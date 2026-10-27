###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit.splice -- replace a compiler-generated "shell" loop of a kernel's ``.s`` with generated code.

The HIP source keeps an ordinary loop (the shell) whose live-ins / live-outs are those of the generated loop; the
compiler produces the prologue, descriptors, addresses and epilogue. This module
  * finds the shell loop: a natural loop of the kernel's CFG (by header label, by WMMA count, or the one with the
    most WMMAs), possibly with internal forward branches (rare paths) and a rotated latch block; its member blocks
    must be contiguous in the text;
  * reads its entry register map (``entry_map``): VGPR / scalar live-ins at the header (CFG liveness) with their
    roles inside the loop (WMMA A / B / C, DS / VMEM address or data, buffer resource, TDM descriptor, loop-carried
    counter), live-outs at every exit, the registers free inside the loop, and what is in flight at entry;
  * replaces the region's lines (``splice``), keeping every block's labels and trailing directives;
  * re-derives the derived lines of a region in a given order (``rebuild``, with asmkit.emit): identity
    (``identity``) or a generator's / scheduler's order;
  * checks the result with the same expert model (``analyze``): no hazard left open inside the region on any path,
    and none that the original did not have in the code after every exit (the code after the loop was compiled for
    the original loop's exit state).
The independent verification (another tokenizer, symbolic equivalence, bounds) is the V track's, not this one's."""

import json
import re

from . import emit as M
from . import expert as E
from .isa import (
    coverage_errors,
    first_tok,
    load_any,
    regs_of,
    sgpr_defs_uses,
)


# --------------------------------------------------------------------------------------------------------------
# CFG, dominators, natural loops
# --------------------------------------------------------------------------------------------------------------
def preds_of(bl):
    pr = [[] for _ in bl]
    for b, (_, _, succ) in enumerate(bl):
        for t in succ:
            pr[t].append(b)
    return pr


def dominators(bl, entry=0):
    n = len(bl)
    pr = preds_of(bl)
    dom = [set(range(n)) for _ in range(n)]
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for b in range(n):
            if b == entry:
                continue
            ps = [dom[p] for p in pr[b]]
            new = (set.intersection(*ps) if ps else set()) | {b}
            if new != dom[b]:
                dom[b], changed = new, True
    return dom


def natural_loops(bl):
    """[(header, members (sorted), latches)]: one per header (back edges b -> h with h dominating b merged)."""
    dom = dominators(bl)
    pr = preds_of(bl)
    loops = {}
    for b, (_, _, succ) in enumerate(bl):
        for h in succ:
            if h in dom[b]:
                mem, stack = {h, b}, ([b] if b != h else [])
                while stack:
                    y = stack.pop()
                    for p in pr[y]:
                        if p not in mem:
                            mem.add(p)
                            stack.append(p)
                m, l = loops.get(h, (set(), set()))
                loops[h] = (m | mem, l | {b})
    return [(h, sorted(m), sorted(l)) for h, (m, l) in sorted(loops.items())]


class Region:
    """A shell loop: member blocks (layout order), header, latches, exits, line range [lo, hi]."""

    def __init__(self, lines, ins, bl, header, members, latches, func):
        self.lines, self.ins, self.bl, self.func = lines, ins, bl, func
        self.header, self.members, self.latches = header, members, latches
        if members != list(range(members[0], members[-1] + 1)):
            raise ValueError(f"loop blocks {members} are not contiguous in the text")
        self.lo, self.hi = bl[members[0]][0], bl[members[-1]][1]
        self.local = {b: k for k, b in enumerate(members)}
        pr = preds_of(bl)
        self.outside_preds = [p for p in pr[header] if p not in self.local]
        for b in members:
            if b != header and any(p not in self.local for p in pr[b]):
                raise ValueError(f"block {b} of the loop is entered from outside (not a natural loop region)")
        self.exits = [(b, t) for b in members for t in bl[b][2] if t not in self.local]

    def nwmma(self):
        return sum(self.ins[i].cls == "XDL" for i in range(self.lo, self.hi + 1))

    def describe(self):
        hl = self.ins[self.bl[self.header][0]].label
        return (
            f"loop header {hl} (block {self.header}), blocks {self.members[0]}..{self.members[-1]}, lines "
            f"{self.lo + 1}-{self.hi + 1}, {self.nwmma()} WMMA, latches {self.latches}, exits {self.exits}"
        )


def find_shell(path_or_lines, label=None, nwmma=None, func=None):
    """The shell loop region of a kernel .s (path or lines): the natural loop whose header block has `label`, or
    with `nwmma` WMMAs, or the innermost loop with the most WMMAs."""
    if isinstance(path_or_lines, (list, tuple)):
        lines = list(path_or_lines)
        from .isa import parse

        ins = parse(lines)
    else:
        lines, ins, fmt = load_any(path_or_lines)
    bl = E.blocks(ins, func)
    loops = natural_loops(bl)
    cands = []
    for h, mem, lat in loops:
        nw = sum(ins[i].cls == "XDL" for b in mem for i in range(bl[b][0], bl[b][1] + 1))
        hl = ins[bl[h][0]].label
        nested = any(set(m2) < set(mem) for _, m2, _ in loops)
        if label is not None and hl != label:
            continue
        if nwmma is not None and nw != nwmma:
            continue
        cands.append((not nested, nw, -h, h, mem, lat))
    if not cands:
        raise LookupError(f"no loop with label {label} / {nwmma} WMMAs")
    cands.sort(reverse=True)
    _, nw, _, h, mem, lat = cands[0]
    return Region(lines, ins, bl, h, mem, lat, func)


def loops_summary(path, func=None):
    lines, ins, fmt = load_any(path)
    bl = E.blocks(ins, func)
    out = []
    for h, mem, lat in natural_loops(bl):
        nw = sum(ins[i].cls == "XDL" for b in mem for i in range(bl[b][0], bl[b][1] + 1))
        out.append((ins[bl[h][0]].label, h, mem, lat, nw))
    return out


# --------------------------------------------------------------------------------------------------------------
# region pieces: per block head lines / items / tail lines
# --------------------------------------------------------------------------------------------------------------
def _is_ins(x):
    return bool(x.mn) and x.cls not in ("LABEL", "DIR")


def block_parts(region, b):
    """(head lines, first, last, tail lines) of block b: head = labels / comments before the first instruction,
    [first, last] the instruction range, tail = directives / comments after the last instruction."""
    s, e, _ = region.bl[b]
    ins = region.ins
    k = s
    while k <= e and not _is_ins(ins[k]):
        k += 1
    j = e
    while j >= k and not _is_ins(ins[j]):
        j -= 1
    for q in range(k, j + 1):
        if ins[q].cls == "DIR" and region.lines[q].strip().startswith("."):
            raise ValueError(f"directive inside block {b} at line {q + 1}: {region.lines[q].strip()}")
    return region.lines[s:k], k, j, region.lines[j + 1 : e + 1]


def region_items(region):
    """[{items, succ (region-local), head, tail, first, last}] per member block, items in source order."""
    out = []
    for b in region.members:
        head, first, last, tail = block_parts(region, b)
        items = M.build_items(region.lines, region.ins, first, last) if first <= last else []
        succ = [region.local[t] for t in region.bl[b][2] if t in region.local]
        out.append(
            {
                "items": items,
                "succ": succ,
                "head": head,
                "tail": tail,
                "first": first,
                "last": last,
                "block": b,
            }
        )
    return out


def _rpo(bl):
    seen, post = set(), []
    stack = [(0, iter(bl[0][2]))]
    seen.add(0)
    while stack:
        b, it = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            post.append(b)
            stack.pop()
        elif nxt not in seen:
            seen.add(nxt)
            stack.append((nxt, iter(bl[nxt][2])))
    return post[::-1] + [b for b in range(len(bl)) if b not in seen]


def walk_block(ins, s, e, fl, expert=True, on_need=None):
    """Advance a Flow over lines [s, e] with the waits in place (also XCNT: an s_set_vgpr_msb or s_wait_xcnt 0
    clears it, a VMEM / TDM op adds its sources)."""
    for i in range(s, e + 1):
        x = ins[i]
        if (
            x.cls in M.VALU_CLS
            and x.cls != "VNOP"
            and on_need is not None
            and any(M.nops_needed(v, x) for v in fl.valus)
        ):
            on_need(i, "v_nop", 0)
        if on_need is not None and _is_ins(x) and x.cls != "MSB":
            sd, _ = sgpr_defs_uses(x)
            guard = {r for r in sd if r != "scc"} | (set() if expert else set(x.defs))
            if guard and any(xs.unknown or (guard & xs.regs) for xs in fl.xs):
                on_need(i, "xcnt", 0)
        for st in fl.sts:
            E.step(
                st,
                ins,
                i,
                (lambda i, f, n: on_need(i, f, n) if (expert or f not in ("va_vdst", "vm_vsrc")) else None)
                if on_need is not None
                else None,
            )
            if not expert:
                E.forget_alu(st)
        if x.cls == "MSB" or (x.cls == "WAIT_CNT" and M.cnt_fields(x).get("xcnt") == 0):
            for xs in fl.xs:
                xs.unknown, xs.regs = False, set()
        elif x.cls in ("VMEM", "FLAT", "TDM"):
            sd, su = sgpr_defs_uses(x)
            for xs in fl.xs:
                xs.regs |= set(su) | (set(x.uses) if not expert else set())
        if x.cls in M.VALU_CLS:
            for v in fl.valus:
                v.insert(0, x)
                del v[12:]
    for st in fl.sts:
        E.cap(st)
    return fl


def function_flows(region, expert=True, iters=40, max_states=8):
    """(in-flow, out-flow) of every block of the kernel function: forward dataflow from the kernel entry (nothing in
    flight, MSB mode 0) with the code's own waits, emit.Flow joins at merges, to a fixed point."""
    ins, bl = region.ins, region.bl
    pr = preds_of(bl)
    inflow, outflow = [None] * len(bl), [None] * len(bl)
    start = M.Flow([E.State(False)], [M.XState(False)], [[]])
    order = _rpo(bl)
    for _ in range(iters):
        changed = False
        for b in order:
            srcs = ([start] if b == 0 else []) + [outflow[p] for p in pr[b] if outflow[p] is not None]
            if not srcs:
                continue
            fl = M.Flow.join(srcs, max_states)
            if inflow[b] is not None and outflow[b] is not None and fl.sig() == inflow[b].sig():
                continue
            inflow[b] = fl
            outflow[b] = walk_block(ins, bl[b][0], bl[b][1], fl.copy(), expert)
            changed = True
        if not changed:
            return inflow, outflow
    raise RuntimeError("function dataflow did not converge")


def entry_paths(region, mode="preheader", expert=True):
    """Ways into the header from outside.
    mode 'preheader' (the original AO / regen_asm_patch model): per outside predecessor block, its instructions
      walked from an unknown state; blind to VMEM loads issued before that block;
    mode 'cfg': the CFG-exact state from the kernel entry (function_flows), joined over the outside predecessors;
    mode 'both': both (every wait satisfies each; the safe default for generated code)."""
    paths = []
    if mode in ("preheader", "both"):
        for p in region.outside_preds:
            s, e, _ = region.bl[p]
            st = E.walk(E.State(unknown=True), region.ins, range(s, e + 1))
            valu = M.preheader_valu_list(region.ins, e + 1)
            paths.append(M.Path(st, M.XState(True), valu))
    if mode in ("cfg", "both"):
        _, outflow = function_flows(region, expert)
        fl = M.Flow.join([outflow[p] for p in region.outside_preds])
        for k, st in enumerate(fl.sts):
            for x in fl.xs if k == 0 else [M.XState(False)]:
                for v in fl.valus if k == 0 else [fl.valus[0]]:
                    paths.append(M.Path(st, x, v))
    return paths


# --------------------------------------------------------------------------------------------------------------
# rebuild / splice
# --------------------------------------------------------------------------------------------------------------
def rebuild(region, blocks_items=None, fmap=None, expert=True, info=None, entry="both"):
    """New lines of the whole text with the region's derived lines re-derived for blocks_items (default: the
    region's own items in source order = identity). blocks_items: [[Item]] per member block (same block structure
    and successors; a generator that changes the block structure builds its own region list for emit_region)."""
    parts = region_items(region)
    if blocks_items is not None:
        if len(blocks_items) != len(parts):
            raise ValueError("blocks_items must have one item list per member block")
        for p, it in zip(parts, blocks_items):
            p["items"] = it
    fmap = fmap if fmap is not None else M.learn_msb_map(region.ins)
    info = {} if info is None else info
    texts = M.emit_region(
        parts,
        fmap,
        None,
        None,
        expert=expert,
        header=region.local[region.header],
        entry_paths=entry_paths(region, entry, expert),
        info=info,
    )
    new_region = []
    for p, t in zip(parts, texts):
        new_region += p["head"] + t + p["tail"]
    return splice(region, new_region)


def splice(region, new_region_lines):
    """The whole text with lines [region.lo, region.hi] replaced."""
    return region.lines[: region.lo] + list(new_region_lines) + region.lines[region.hi + 1 :]


def identity(path, label=None, nwmma=None, expert=True, func=None, entry="both"):
    """(new lines, region, info) of the identity rebuild of a shell loop (entry='preheader' reproduces the original
    AO rebuild, e.g. ao_id e4b638c8)."""
    region = find_shell(path, label=label, nwmma=nwmma, func=func)
    info = {}
    return rebuild(region, expert=expert, info=info, entry=entry), region, info


def set_vgpr_count(lines, n):
    """Raise the kernel's VGPR allocation to at least n (``.amdhsa_next_free_vgpr`` and ``.vgpr_count``)."""
    out = []
    for l in lines:
        m = re.match(r"(\s*\.amdhsa_next_free_vgpr\s+)(\d+)(.*)", l)
        if m and int(m.group(2)) < n:
            l = f"{m.group(1)}{n}{m.group(3)}"
        m = re.match(r"(\s*\.vgpr_count:\s+)(\d+)(.*)", l)
        if m and int(m.group(2)) < n:
            l = f"{m.group(1)}{n}{m.group(3)}"
        out.append(l)
    return out


# --------------------------------------------------------------------------------------------------------------
# analysis of a text's region with its own waits (identity and generated code alike)
# --------------------------------------------------------------------------------------------------------------
def analyze(region, iters=12, max_states=8, expert=True, entry="both"):
    """Walk the region of region.lines with the waits in place over everything that reaches each block (entry +
    back edges to a fixed point, emit.Flow joins). Returns {'open': {(line, field)}, 'nops': [lines],
    'exit_flows': {(b, t): Flow}}."""
    ins, bl = region.ins, region.bl
    loc = region.local
    pr = {b: [p for p in preds_of(bl)[b] if p in loc] for b in region.members}
    order, indeg = [], {b: sum(1 for p in pr[b] if b != region.header) for b in region.members}
    ready = [region.header]
    while ready:
        b = ready.pop(0)
        order.append(b)
        for t in bl[b][2]:
            if t in loc and t != region.header:
                indeg[t] -= 1
                if indeg[t] == 0:
                    ready.append(t)
    entry = M.Flow.of(entry_paths(region, entry, expert))
    if not expert:
        for st in entry.sts:
            E.forget_alu(st)
    back = None
    for _ in range(iters):
        opened, nops, outs = set(), set(), {}
        for b in order:
            if b == region.header:
                fl = M.Flow.join([entry] + ([back] if back is not None else []), max_states)
            else:
                fl = M.Flow.join([outs[q] for q in pr[b]], max_states)
            s, e, _ = bl[b]

            def on_need(i, f, n, opened=opened, nops=nops):
                if f == "v_nop":
                    nops.add(i)
                else:
                    opened.add((i, f))

            walk_block(ins, s, e, fl, expert, on_need)
            outs[b] = fl
        nb = M.Flow.join([outs[q] for q in pr[region.header]], max_states)
        if back is not None and nb.sig() == back.sig():
            break
        back = nb
    return {"open": opened, "nops": sorted(nops), "exit_flows": {(b, t): outs[b] for b, t in region.exits}}


def exit_needs(region, an, limit=400, expert=True):
    """{(b, t): {(line, field)}}: hazards in the code after each exit (block t and its fall-through successors,
    up to `limit` instructions) that the waits there leave open, from the region's exit flows."""
    ins, bl = region.ins, region.bl
    out = {}
    for (b, t), fl in an["exit_flows"].items():
        found = set()
        fl = fl.copy()
        n, blk = 0, t
        while blk is not None and n < limit:
            s, e, succ = bl[blk]
            stop = any(ins[i].mn == "s_endpgm" for i in range(s, e + 1))
            walk_block(ins, s, e, fl, expert, lambda i, f, k, found=found: found.add((i, f)))
            n += sum(1 for i in range(s, e + 1) if _is_ins(ins[i]))
            # follow the fall-through only (a taken branch leaves the walk: the target has its own waits)
            blk = blk + 1 if (not stop and blk + 1 in succ and blk + 1 < len(bl)) else None
        out[(b, t)] = found
    return out


def find_region_like(orig_region, new_lines, func=None):
    """The loop of new_lines whose header has the original header label (a generated region may have any block
    structure inside)."""
    from .isa import parse

    ins_new = parse(new_lines)
    hl = orig_region.ins[orig_region.bl[orig_region.header][0]].label
    bl_new = E.blocks(ins_new, func)
    loops = [(h, m, l) for h, m, l in natural_loops(bl_new) if ins_new[bl_new[h][0]].label == hl]
    if len(loops) != 1:
        raise LookupError(f"rebuilt text: no single loop with header {hl}")
    h, m, l = loops[0]
    return Region(new_lines, ins_new, bl_new, h, m, l, func)


def check(orig_region, new_lines, func=None, expert=True, entry="both"):
    """Static checks of a rebuilt or generated region against the original shell (asmkit's own model): failures
    (empty = pass) and a stats dict. Exits are matched by their target label."""
    try:
        new = find_region_like(orig_region, new_lines, func)
    except (LookupError, ValueError) as e:
        return [str(e)], {}
    fail = []
    if (
        new.lines[: new.lo] != orig_region.lines[: orig_region.lo]
        or new.lines[new.hi + 1 :] != orig_region.lines[orig_region.hi + 1 :]
    ):
        fail.append("text outside the region differs")
    cov = [(i + 1, e) for i in range(new.lo, new.hi + 1) for e in coverage_errors(new.ins[i])]
    if cov:
        fail.append(f"parse coverage: {cov[:4]}")
    a_new, a_old = analyze(new, expert=expert, entry=entry), analyze(orig_region, expert=expert, entry=entry)
    if a_new["open"]:
        fail.append(
            f"{len(a_new['open'])} hazards left open inside the region, e.g. "
            f"{sorted((i + 1, f) for i, f in a_new['open'])[:4]}"
        )
    if a_new["nops"]:
        fail.append(
            f"{len(a_new['nops'])} WMMA/TRANS hazards without enough v_nop, e.g. lines "
            f"{[i + 1 for i in a_new['nops'][:4]]}"
        )

    def by_label(region, en):
        out = {}
        for (b, t), found in en.items():
            key = region.ins[region.bl[t][0]].label or (
                f"region_end+{region.bl[t][0] - region.hi}"
                if t > region.members[-1]
                else f"region_start-{region.lo - region.bl[t][0]}"
            )
            out.setdefault(key, set()).update(found)
        return out

    en_new = by_label(new, exit_needs(new, a_new, expert=expert))
    en_old = by_label(orig_region, exit_needs(orig_region, a_old, expert=expert))
    if set(en_new) - set(en_old):
        fail.append(f"exits to new targets {sorted(set(en_new) - set(en_old))}")
    shift = len(new_lines) - len(orig_region.lines)
    extra_total = 0
    for lab, found in en_new.items():
        old = {(i + shift if i > orig_region.hi else i, f) for i, f in en_old.get(lab, set())}
        extra = found - old
        extra_total += len(extra)
        if extra:
            fail.append(
                f"exit to {lab}: {len(extra)} hazards after the loop the original exit state did not have, "
                f"e.g. {sorted((i + 1, f) for i, f in extra)[:4]}"
            )
    stats = {
        "open_orig": len(a_old["open"]),
        "open_new": len(a_new["open"]),
        "nops_orig": len(a_old["nops"]),
        "nops_new": len(a_new["nops"]),
        "exit_needs_orig": sum(len(v) for v in en_old.values()),
        "exit_needs_new": sum(len(v) for v in en_new.values()),
        "exit_needs_extra": extra_total,
        "region_lines": [new.lo + 1, new.hi + 1],
        "blocks": len(new.members),
    }
    return fail, stats


def generate(shell, gen_lines, fmap=None, expert=True, entry="both", info=None):
    """Splice a generated loop into a shell: gen_lines is the generated region's text (labels and real
    instructions; any derived lines in it are dropped and re-derived). Contract: it holds the shell's header label
    (and starts with it when the preheader falls through into the region), leaves only by branching to the shell's
    exit labels (or by falling through at its end where the shell did); the emitter puts MSB mode 0 at every block
    boundary.
    Returns the whole new text; run check(shell, new) on it."""
    from .isa import parse

    hl = shell.ins[shell.bl[shell.header][0]].label
    gins = parse(gen_lines)
    if not any(x.label == hl for x in gins):
        raise ValueError(f"the generated region has no header label {hl}")
    falls_in = any(
        p == shell.members[0] - 1 and shell.members[0] in shell.bl[p][2] for p in shell.outside_preds
    )
    first = next(x for x in gins if x.cls != "DIR")
    if falls_in and first.label != hl:
        raise ValueError(
            f"the preheader falls through into the region: it must start with the header label {hl}"
        )
    # blocks of the generated text: start at labels and after branches
    starts = [k for k, x in enumerate(gins) if x.label] + [
        k + 1 for k, x in enumerate(gins) if x.mn.startswith(("s_branch", "s_cbranch")) and k + 1 < len(gins)
    ]
    starts = sorted(set(starts) | {0})
    gb = [(s0, (starts[n + 1] - 1) if n + 1 < len(starts) else len(gins) - 1) for n, s0 in enumerate(starts)]
    label_block = {gins[s0].label: n for n, (s0, _) in enumerate(gb) if gins[s0].label}
    blocks = []
    for n, (s0, e0) in enumerate(gb):
        k = s0
        while k <= e0 and not _is_ins(gins[k]):
            k += 1
        j = e0
        while j >= k and not _is_ins(gins[j]):
            j -= 1
        items = M.build_items(gen_lines, gins, k, j) if k <= j else []
        succ, last = [], (gins[j] if k <= j else None)
        if last is not None and last.mn.startswith(("s_branch", "s_cbranch")):
            t = strip_target(last)
            if t in label_block:
                succ.append(label_block[t])
            if last.mn.startswith("s_cbranch") and n + 1 < len(gb):
                succ.append(n + 1)
        elif n + 1 < len(gb):
            succ.append(n + 1)
        blocks.append(
            {"items": items, "succ": succ, "head": gen_lines[s0:k], "tail": gen_lines[j + 1 : e0 + 1]}
        )
    fmap = fmap if fmap is not None else M.learn_msb_map(shell.ins)
    texts = M.emit_region(
        blocks,
        fmap,
        None,
        None,
        expert=expert,
        header=label_block[hl],
        entry_paths=entry_paths(shell, entry, expert),
        info=info,
    )
    region = []
    for b, t in zip(blocks, texts):
        region += b["head"] + t + b["tail"]
    return splice(shell, region)


def strip_target(x):
    from .isa import strip_comment

    return strip_comment(x.ops[0]).strip() if x.ops else ""


# --------------------------------------------------------------------------------------------------------------
# entry register map
# --------------------------------------------------------------------------------------------------------------
def _ranges(regs):
    out, cur = [], None
    for r in sorted(regs):
        if cur and r == cur[1] + 1:
            cur[1] = r
        else:
            cur = [r, r]
            out.append(cur)
    return out


def _fmt(lo, hi, p="v"):
    return f"{p}{lo}" if lo == hi else f"{p}[{lo}:{hi}]"


def _roles(x):
    """[(operand regs (VGPR list), scalar list, role)] of an instruction."""
    mn, ops, out = x.mn, [o for o in x.ops if o != "::"], []
    vr = [regs_of(first_tok(o)) for o in ops]
    from .isa import sregs

    sr = [sregs(first_tok(o)) for o in ops]
    if x.cls == "XDL":
        names = ["wmma_d", "wmma_a", "wmma_b", "wmma_c"]
    elif x.cls == "DS":
        names = (
            ["ds_dst", "ds_addr"]
            if mn.startswith(("ds_load", "ds_read"))
            else ["ds_addr", "ds_data", "ds_data2"]
        )
    elif x.cls in ("VMEM", "FLAT"):
        load = "_load" in mn
        if x.cls == "VMEM":
            names = (
                ["vmem_dst", "vmem_vaddr", "buffer_rsrc", "buffer_soffset"]
                if load
                else ["vmem_data", "vmem_vaddr", "buffer_rsrc", "buffer_soffset"]
            )
        else:
            names = (
                ["vmem_dst", "vmem_vaddr", "vmem_saddr"]
                if load
                else ["vmem_vaddr", "vmem_data", "vmem_saddr"]
            )
    elif x.cls == "TDM":
        names = [f"tdm_desc{k}" for k in range(len(ops))]
    elif x.cls in ("CSMACC", "DPMACC", "TRANS"):
        names = ["valu_dst"] + ["valu_src"] * 8
    elif x.cls in ("SALU", "BARRIER"):
        names = ["salu_dst"] + ["salu_src"] * 8
    else:
        names = ["op"] * 16
    for k, o in enumerate(ops):
        out.append((vr[k], sr[k], names[k] if k < len(names) else "op"))
    return out


def entry_map(region):
    """Register contract of the shell loop for a generator (JSON-serializable)."""
    ins, bl = region.ins, region.bl
    lout = E.liveness(ins, bl)
    lin = E.live_in(ins, bl, lout)
    s_in, s_out = E.scalar_liveness(ins, bl, sgpr_defs_uses)
    h = region.header
    live_in_v = lin[h]
    vin = {r for r in range(1024) if live_in_v >> r & 1}
    vdef, sdef = set(), set()
    roles_v, roles_s, tuples = {}, {}, {}
    used_v = set()
    first_use = {}
    for b in region.members:
        s, e, _ = bl[b]
        for i in range(s, e + 1):
            x = ins[i]
            if not _is_ins(x):
                continue
            vdef |= set(x.defs) if not x.rmw else set(x.defs)
            used_v |= set(x.defs) | set(x.uses)
            sd, su = sgpr_defs_uses(x)
            sdef |= set(sd)
            for vregs, sregs_, role in _roles(x):
                if vregs:
                    key = (vregs[0], vregs[-1])
                    tuples.setdefault(key, set()).add(role)
                    for r in vregs:
                        roles_v.setdefault(r, set()).add(role)
                        first_use.setdefault(r, i + 1)
                for r in sregs_:
                    roles_s.setdefault(r, set()).add(role)
    sin = s_in[h]
    # live-in VGPRs grouped by the operand tuples that touch them
    groups = []
    covered = set()
    for (lo, hi), rl in sorted(tuples.items()):
        regs = set(range(lo, hi + 1)) & vin
        if not regs or regs <= covered:
            continue
        covered |= regs
        groups.append(
            {
                "regs": _fmt(lo, hi),
                "live_in": [_fmt(a, b) for a, b in _ranges(regs)],
                "carried": bool(regs & vdef),
                "roles": sorted(rl),
                "first_use_line": first_use.get(lo),
            }
        )
    rest = vin - covered
    for a, b in _ranges(rest):
        groups.append(
            {
                "regs": _fmt(a, b),
                "live_in": [_fmt(a, b)],
                "carried": bool(set(range(a, b + 1)) & vdef),
                "roles": sorted(set().union(*[roles_v.get(r, set()) for r in range(a, b + 1)]))
                or ["passthrough"],
                "first_use_line": None,
            }
        )
    exits = {}
    for b, t in region.exits:
        lv = {r for r in range(1024) if lin[t] >> r & 1}
        exits[region.ins[bl[t][0]].label or f"block{t}"] = {
            "from_block": b,
            "vgpr_live": [_fmt(a, c) for a, c in _ranges(lv)],
            "vgpr_live_defined_in_loop": [_fmt(a, c) for a, c in _ranges(lv & vdef)],
            "scalar_live": sorted(s_in[t], key=_skey),
        }
    vgpr_count = None
    for l in region.lines:
        m = re.match(r"\s*\.vgpr_count:\s+(\d+)", l)
        if m:
            vgpr_count = int(m.group(1))
    live_any_exit = set()
    for b, t in region.exits:
        live_any_exit |= {r for r in range(1024) if lin[t] >> r & 1}
    free = set(range(1024)) - vin - live_any_exit
    ent = entry_paths(region, "cfg")
    pend = sorted({r for p in ent for rec in p.st.loads if not rec.done for r in rec.writes})
    scal = []
    for r in sorted(sin, key=_skey):
        scal.append(
            {"reg": r, "carried": r in sdef, "roles": sorted(roles_s.get(r, set())) or ["passthrough"]}
        )
    return {
        "loop": region.describe(),
        "header_label": region.ins[bl[h][0]].label,
        "lines": [region.lo + 1, region.hi + 1],
        "vgpr_count_kernel": vgpr_count,
        "vgpr_live_in": groups,
        "vgpr_live_in_count": len(vin),
        "scalar_live_in": scal,
        "exits": exits,
        "vgpr_free_in_loop": [_fmt(a, b) for a, b in _ranges(free)],
        "vgpr_free_count": len(free),
        "vgpr_used_in_loop": len(used_v),
        "entry": {
            "msb_mode": 0,
            "pending_load_vgprs": [_fmt(a, b) for a, b in _ranges(pend)],
            "unknown_valu": any(p.st.unknown_v for p in ent),
            "unknown_ds": any(p.st.unknown_ds for p in ent),
            "outside_preds": region.outside_preds,
        },
    }


def _skey(r):
    m = re.match(r"([a-z_]+)(\d*)", r)
    return (m.group(1), int(m.group(2)) if m.group(2) else -1)


def dump_map(m, path):
    with open(path, "w") as f:
        json.dump(m, f, indent=1)
