###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The code the generators of the gfx1250 MLA kernels emit (fwd_sched and bwd_dkdv_sched on the assembly model asm).
Host-only and standard library: it runs wherever the tests run, whatever the GPU.

The kernels' inline asm is the product of two list schedulers and of a model of the hardware's hazards, so a change to
any of them changes code that was validated on an MI455X. REGIONS pins that code. After a deliberate change:

  1. dump the regions before and after it and read the difference (attach it to the pull request):
     ``python tests/pytorch/ops/test_attention_flydsl_gfx1250_mla_codegen.py --dump DIR`` in both trees, then
     ``diff -r`` (as a script this file reads the generators of the tree it is in, whatever is installed);
  2. run test_attention_flydsl_gfx1250_mla.py and test_attention_flydsl_gfx1250_mla_bwd.py on an MI455X, and time
     the kernels against the tree before the change;
  3. update REGIONS with the digests the same command prints.
"""

import argparse
import hashlib
import importlib
import pathlib
import re
import sys
import types

import pytest

# sha256 of every region's asm text and constraint string
REGIONS = {
    ("fwd", "INIT"): "7744dc06fb75337efd6dd738a530138a3cbd503312c715427ec9ae6a18cdbc61",
    ("fwd", "INIT_C0"): "10a255a7963cd68dd7e23b862b6be02a68f2bb920543ef82634bd21fa2f33957",
    ("fwd", "TILE_CL"): "17c62a6a6ca805827a2a76472b7413ae18d4974572d35ce16fa2b605db9518b5",
    ("fwd", "TILE_D0"): "514fd7f3b3e961d08c1d9cdec1d7e35e4646728fb82e8e70482f75ea6201b77f",
    ("fwd", "TILE_D1"): "7af7a29bcde49ed425c342ed6d70155bc08d0c878fa1bcce54bbc1f8af6e9cd3",
    ("fwd", "TILE_D2"): "84c397c6ffb5d547d494436682111fe5f9b198e708a2197005e4f0ad21fcae1e",
    ("fwd", "TILE_D3F"): "5c079c9d73cf4cc3f623865529f54b6d776331ffb3d319f13ffb867ae14257eb",
    ("fwd", "FINAL_A"): "4570480af899fa78f2cb5b33af663c32a2d7343dcb0f524547eae3704d6bf09f",
    ("bwd_dkdv", "MAIN"): "a1bde80d5608aa170089223aadfcf869f26c18761bbc88ab05a27d127614c631",
}
_EVERY_REGION = pytest.mark.parametrize(("sched", "name"), list(REGIONS), ids=["_".join(k) for k in REGIONS])


_PACKAGE = "primus_turbo.flydsl.attention.gfx1250_mla"


def _module(name):
    """asm, fwd_sched or bwd_dkdv_sched, imported on first use (test_attention_flydsl_gfx1250_mla checks that
    collecting the tests imports no kernel module)."""
    return importlib.import_module(f"{_PACKAGE}.{name}")


def _sched(sched):
    return _module(f"{sched}_sched")


def _vr(r, n=1):
    return _module("asm").vr(r, n)


def _digest(sched, name):
    text, constraints = _sched(sched).region_asm(name)
    return hashlib.sha256((text + "\n" + constraints).encode()).hexdigest()


def _code(text):
    """The instruction and label lines of a region, without their comments."""
    return [x for x in (line.split(";")[0].strip() for line in text.split("\n")) if x]


def _instructions(sched, name):
    """The instruction lines of a region (no labels)."""
    return [x for x in _code(_sched(sched).region_asm(name)[0]) if not x.endswith(":")]


def _flow_blocks(sched, name):
    """The region as straight-line blocks: ([instruction lines], [successor blocks]) per block. A block ends before
    a label and behind a branch; control falls through to the next one unless it ends with s_branch."""
    blocks, labels = [[]], {}
    for line in _code(_sched(sched).region_asm(name)[0]):
        if line.endswith(":"):
            if blocks[-1]:
                blocks.append([])
            labels[line[:-1]] = len(blocks) - 1
        else:
            blocks[-1].append(line)
            if line.startswith(("s_branch", "s_cbranch")):
                blocks.append([])
    succ = []
    for i, lines in enumerate(blocks):
        last = lines[-1] if lines else ""
        out = [labels[last.split()[1]]] if last.startswith(("s_branch", "s_cbranch")) else []
        if not last.startswith("s_branch") and i + 1 < len(blocks):
            out.append(i + 1)
        succ.append(out)
    return blocks, succ


def _dataflow(sched, name, entry, step, join):
    """Runs ``step(state, line) -> state`` over every instruction of the region along every path, from ``entry``
    to a fixed point (``join(states)``: the state where paths meet). ``step`` asserts what must hold."""
    blocks, succ = _flow_blocks(sched, name)
    preds = [[i for i, out in enumerate(succ) if j in out] for j in range(len(blocks))]
    after = [None] * len(blocks)
    todo = list(range(len(blocks)))
    while todo:
        i = todo.pop(0)
        reaching = [after[p] for p in preds[i] if after[p] is not None] + ([entry] if i == 0 else [])
        if not reaching:
            continue
        state = join(reaching)
        for line in blocks[i]:
            state = step(state, line)
        if state != after[i]:
            after[i] = state
            todo += [j for j in succ[i] if j not in todo]


def test_every_region_is_pinned():
    assert {(s, n) for s in ("fwd", "bwd_dkdv") for n in _sched(s).OPERANDS} == set(REGIONS)


@_EVERY_REGION
def test_region_is_the_validated_code(sched, name):
    assert _digest(sched, name) == REGIONS[sched, name], (
        f"{sched} {name}: the generated code changed; see this file's docstring before updating its digest"
    )


@_EVERY_REGION
def test_region_names_its_operands_and_its_own_registers_only(sched, name):
    """A region's constraints are its operands, then its clobbers: every VGPR LLVM could allocate from the generated
    code's base on, the scalar registers it uses, vcc, scc and memory. Its text names each operand, and no VGPR
    below the base."""
    mod = _sched(sched)
    text, constraints = mod.region_asm(name)
    constraints = constraints.split(",")
    nops = len(mod.OPERANDS[name])
    operands, clobbers = constraints[:nops], constraints[nops:]
    assert all(c.startswith("~{") for c in clobbers) and not any(c.startswith("~") for c in operands)
    assert {f"~{{v{r}}}" for r in range(mod.BASE, 256)} <= set(clobbers)
    assert {"~{vcc}", "~{scc}", "~{memory}"} <= set(clobbers)
    code = "\n".join(_code(text))
    assert {int(k) for k in re.findall(r"\$(\d+)", code)} == set(range(nops))
    # (a VGPR from v256 on is spelled by its low 8 bits, with its number in a comment)
    named = re.findall(r"(?<![\w$.\[])v(\d+)\b(?!\s*/\*)", code) + re.findall(
        r"(?<![\w$.])v\[(\d+):\d+\](?!\s*/\*)", code
    )
    assert named and min(int(r) for r in named) >= mod.BASE
    # an SGPR the text names is a clobber (the operands are $k)
    sgprs = {int(r) for r in re.findall(r"(?<![\w$.])s\[?(\d+)", code)}
    assert sgprs <= {int(c[3:-1]) for c in clobbers if re.fullmatch(r"~\{s\d+\}", c)}


@_EVERY_REGION
def test_region_is_in_vgpr_msb_mode_0_at_labels_and_branches(sched, name):
    """The VGPR MSB mode (s_set_vgpr_msb) is 0 wherever control can enter or leave a block, and at the end: the
    kernel's own code, and every block, start in mode 0."""
    mode = 0
    for line in _code(_sched(sched).region_asm(name)[0]):
        if line.startswith("s_set_vgpr_msb"):
            new = int(line.split()[1], 0)
            assert new >> 8 == mode, line  # (the instruction also names the mode it leaves)
            mode = new & 0xFF
        elif line.endswith(":") or line.startswith(("s_branch", "s_cbranch")):
            assert mode == 0, line
    assert mode == 0


# The regions whose first VALU-class instruction has no va_vdst(0) wait before it: nothing they could depend on is
# in flight where they start. INIT opens a block: behind the kernel's start, or behind the previous block's
# TILE_D3F / FINAL_A, which end drained; MAIN is the whole kernel body.
_START_WITHOUT_DRAIN = {("fwd", "INIT"), ("bwd_dkdv", "MAIN")}


@_EVERY_REGION
def test_region_drains_the_valu_before_its_first_one(sched, name):
    """The model starts a region with an empty VALU history: its v_nop rules do not see the WMMAs and
    transcendentals the code before the region left in flight. A va_vdst(0) wait before the region's first
    VALU-class instruction ends them (today the wait of the region's first LDS read)."""
    lines = _instructions(sched, name)
    first = next(i for i, x in enumerate(lines) if x.startswith("v_"))
    drained = any("depctr_va_vdst(0)" in x for x in lines[:first])
    assert drained != ((sched, name) in _START_WITHOUT_DRAIN), lines[: first + 1]


@pytest.mark.parametrize("name", ["TILE_D3F", "FINAL_A"])
def test_a_blocks_last_region_ends_drained(name):
    """INIT, the next block's first region, starts with VALUs and no wait of its own."""
    lines = _instructions("fwd", name)
    last = max(i for i, x in enumerate(lines) if "depctr_va_vdst(0)" in x)
    assert not [x for x in lines[last:] if x.startswith(("v_", "ds_", "buffer_", "global_"))]


def test_generated_valu_classes():
    """The wait and v_nop rules of the model are those of three VALU classes: the one WMMA, the transcendentals and
    core VALU. Nothing the generators emit is in another one (DPMACC), or is a WMMA the wait states are not for."""
    asm = _module("asm")
    classes = {}
    for sched, name in REGIONS:
        for line in _instructions(sched, name):
            mn = line.split()[0]
            if mn.startswith("v_"):
                classes.setdefault(asm.classify(mn), set()).add(mn.removesuffix("_e32").removesuffix("_e64"))
    assert set(classes) == {"XDL", "TRANS", "CSMACC", "VNOP"}
    assert classes["XDL"] == {asm.WMMA}
    assert classes["TRANS"] == {"v_exp_f32", "v_log_f32", "v_rcp_f32"}


@_EVERY_REGION
def test_region_gives_no_fresh_valu_written_sgpr_to_memory(sched, name):
    """LLVM keeps 5 instructions between a VALU that writes an SGPR and a buffer, global or TDM instruction that
    reads it; the model does not pad for that. On every path of the region, such an instruction reads no SGPR a
    VALU wrote less than 5 instructions before."""
    asm = _module("asm")

    def step(fresh, line):  # fresh: {SGPR a VALU wrote: instructions since}
        x = asm.parse_line(-1, "\t" + line)
        defs, uses = asm.sgpr_defs_uses(x)
        if x.cls in ("VMEM", "FLAT", "TDM"):
            assert not set(uses) & set(fresh), (line, fresh)
        aged = {r: n + 1 for r, n in fresh.items() if n + 1 < 5 and r not in defs}
        if x.cls in asm.VALU:
            aged.update(dict.fromkeys(defs, 0))
        return aged

    def join(states):
        return {r: min(s[r] for s in states if r in s) for r in set().union(*states)}

    _dataflow(sched, name, {}, step, join)


@_EVERY_REGION
def test_region_keeps_at_most_ten_tdm_ops_in_flight(sched, name):
    """LLVM waits s_wait_tensorcnt 10 before a TDM op that follows another; the model places no such wait. On
    every path of the region, no more than 10 of the ops it issued can be in flight where it issues one."""

    def step(in_flight, line):
        mn = line.split()[0]
        if mn == "s_wait_tensorcnt":
            return min(in_flight, int(line.split()[1], 0))
        if mn.startswith("tensor_"):
            assert in_flight <= 10, line
            return in_flight + 1
        return in_flight

    _dataflow(sched, name, 0, step, max)


@pytest.mark.parametrize("name", ["TILE_D3F", "FINAL_A"])
def test_forward_division_reads_its_own_carry(name):
    """FINAL computes 1.0 / d per q tile as LLVM expands it: v_div_scale (its second one writes vcc), ...,
    v_div_fmas (reads vcc), v_div_fixup. The scheduler interleaves the four q tiles: the last writer of vcc before
    each v_div_fmas is the v_div_scale of its own division (the one on the d its v_div_fixup names)."""
    asm = _module("asm")
    ins = [asm.parse_line(-1, "\t" + line) for line in _instructions("fwd", name)]

    def reg(x, p):
        return asm.regs_of(asm.first_tok(x.ops[p]))

    carry, divisions = None, 0
    for i, x in enumerate(ins):
        if x.mn.startswith("v_div_fmas"):
            fixup = next(y for y in ins[i:] if y.mn.startswith("v_div_fixup") and reg(y, 0) == reg(x, 0))
            assert carry is not None and carry.mn.startswith("v_div_scale"), x.text
            assert reg(carry, 3) == reg(fixup, 2), (carry.text, x.text, fixup.text)
            divisions += 1
        if "vcc" in asm.sgpr_defs_uses(x)[0]:
            carry = x
    assert divisions == 4


def _emit(*texts, unknown=False):
    asm = _module("asm")
    flow = asm.Flow([asm.State(unknown=unknown)], [asm.XState(unknown)], [[]])
    return [line.split(";")[0].strip() for line in asm.emit(list(texts), flow)[0]]


def _wmma(d, a, b, c=None):
    return f"v_wmma_f32_16x16x32_bf16 {_vr(d, 8)}, {_vr(a, 8)}, {_vr(b, 8)}, {_vr(c, 8) if c else 0}"


def test_asm_model_sets_the_vgpr_msb_mode():
    high, mixed = f"v_mov_b32 {_vr(261)}, 0", f"v_add_f32 {_vr(5)}, {_vr(261)}, {_vr(600)}"
    assert _emit(high, mixed) == [
        "s_set_vgpr_msb 0x40",
        high,
        "s_set_vgpr_msb 0x4009",
        mixed,
        "s_set_vgpr_msb 0x900",
    ]
    assert _emit(high, "s_cbranch_scc1 .Lx") == [
        "s_set_vgpr_msb 0x40",
        high,
        "s_set_vgpr_msb 0x4000",
        "s_cbranch_scc1 .Lx",
    ]
    with pytest.raises(ValueError, match="crosses a 256-register block"):
        _vr(252, 8)


def test_asm_model_pads_the_wmma_and_trans_hazards():
    first = _wmma(8, 16, 24)
    # a WMMA's result as the next WMMA's A operand: 5 VALU-class instructions between them; as its accumulator: none
    assert _emit(first, _wmma(32, 8, 24)) == [first] + ["v_nop"] * 5 + [_wmma(32, 8, 24)]
    assert _emit(first, _wmma(8, 16, 24, 8)) == [first, _wmma(8, 16, 24, 8)]
    # a VALU that reads a WMMA's result: 4; one that reads a transcendental's: 1
    add = f"v_add_f32 {_vr(40)}, {_vr(8)}, {_vr(9)}"
    assert _emit(first, add) == [first] + ["v_nop"] * 4 + [add]
    exp, use = f"v_exp_f32 {_vr(3)}, {_vr(4)}", f"v_add_f32 {_vr(5)}, {_vr(3)}, {_vr(3)}"
    assert _emit(exp, use) == [exp, "v_nop", use]
    # a VALU that writes a register the WMMA reads: 4 as well
    move = f"v_mov_b32 {_vr(16)}, 0"
    assert _emit(first, move) == [first] + ["v_nop"] * 4 + [move]


def test_asm_model_knows_one_wmma():
    """The wait states are those of the 8-cycle bf16 WMMA with its accumulator +0 or its own destination; the
    model refuses another WMMA, or another accumulator, instead of padding it wrongly."""
    asm = _module("asm")
    other = f"v_wmma_i32_16x16x64_iu8 {_vr(8, 8)}, {_vr(16, 8)}, {_vr(24, 8)}, 0"
    for text in (other, _wmma(8, 16, 24, 32)):
        with pytest.raises(ValueError, match="WMMA hazard rules"):
            _emit(text, text)
    assert asm.classify("v_swmmac_f32_16x16x64_bf16") == "XDL"  # refused too: not asm.WMMA


def test_asm_model_waits_for_what_is_in_flight():
    add, store = f"v_add_f32 {_vr(5)}, {_vr(3)}, {_vr(3)}", f"ds_store_b32 {_vr(1)}, {_vr(5)}"
    load0, load1 = (
        f"ds_load_b128 {_vr(8, 4)}, {_vr(1)}",
        f"ds_load_b128 {_vr(12, 4)}, {_vr(1)} offset:16",
    )
    use = f"v_add_f32 {_vr(5)}, {_vr(8)}, {_vr(8)}"
    # a memory instruction on a register an in-flight VALU writes
    assert _emit(add, store) == [add, "s_wait_alu depctr_va_vdst(0)", store]
    # a register an in-flight load writes: the wait leaves the younger load in flight
    assert _emit(load0, load1, use) == [load0, load1, "s_wait_dscnt 0x1", use]
    # LDS operations in flight at the barrier
    assert _emit(load0, "s_barrier_signal -1") == [load0, "s_wait_dscnt 0x0", "s_barrier_signal -1"]
    # the rewrite of a register an in-flight memory instruction still reads
    move = f"v_mov_b32 {_vr(1)}, 0"
    assert _emit(store, move) == [store, "s_wait_alu depctr_vm_vsrc(0)", move]
    tdm = "tensor_load_to_lds s[4:7], s[8:15]"
    assert _emit(tdm, "s_mov_b32 s4, 0") == [tdm, "s_wait_xcnt 0x0", "s_mov_b32 s4, 0"]
    # entered from code the model does not follow: everything may be in flight
    assert _emit(load0, unknown=True) == [
        "s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)",
        "s_wait_dscnt 0x0",
        load0,
    ]


def test_asm_model_counts_within_a_class():
    """A class completes in order, so a wait may leave the younger instructions of the class in flight."""
    add0, add1 = f"v_add_f32 {_vr(5)}, {_vr(3)}, {_vr(3)}", f"v_add_f32 {_vr(6)}, {_vr(3)}, {_vr(3)}"
    store = f"ds_store_b32 {_vr(1)}, {_vr(5)}"
    assert _emit(add0, add1, store) == [add0, add1, "s_wait_alu depctr_va_vdst(1)", store]
    # two VALU classes in flight complete in any order: 0, except for the overwrite of a register only an in-flight
    # WMMA still reads, which that WMMA's own class bounds
    wmmas = [_wmma(8, 16, 24), _wmma(32, 40, 48), _wmma(56, 64, 72)]
    read_d, write_a = f"ds_store_b128 {_vr(1)}, {_vr(8, 4)}", f"ds_load_b128 {_vr(16, 4)}, {_vr(1)}"
    assert _emit(*wmmas, add0, read_d) == [*wmmas, add0, "s_wait_alu depctr_va_vdst(0)", read_d]
    assert _emit(*wmmas, add0, write_a) == [*wmmas, add0, "s_wait_alu depctr_va_vdst(2)", write_a]
    # memory instructions read their sources in order within their class too
    st0 = f"buffer_store_b128 {_vr(8, 4)}, {_vr(1)}, s[4:7], null offen"
    st1 = f"buffer_store_b128 {_vr(12, 4)}, {_vr(2)}, s[4:7], null offen"
    move = f"v_mov_b32 {_vr(8)}, 0"
    assert _emit(st0, st1, move) == [st0, st1, "s_wait_alu depctr_vm_vsrc(1)", move]
    load = f"ds_load_b128 {_vr(20, 4)}, {_vr(3)}"
    assert _emit(st0, load, move) == [st0, load, "s_wait_alu depctr_vm_vsrc(0)", move]
    # a TDM op reads no VGPR and is in none of these classes: it does not stand in for a buffer instruction
    tdm = "tensor_load_to_lds s[4:7], s[8:15]"
    assert _emit(st0, tdm, tdm, move) == [st0, tdm, tdm, "s_wait_alu depctr_vm_vsrc(0)", move]


def test_asm_model_loop_block_waits_for_its_back_edge():
    """A block that is its own successor is emitted for the join of its entry and its exit, to a fixed point of the
    flow: the wait its first instruction needs only from the second trip on is there. (The flow keeps changing
    until the VALU histories are full; the block's va_vdst(0) retires its VALU, whose record's distance would
    otherwise grow by a trip at a time up to the model's cap.)"""
    asm = _module("asm")
    use, load = f"v_add_f32 {_vr(5)}, {_vr(8)}, {_vr(8)}", f"ds_load_b128 {_vr(8, 4)}, {_vr(1)}"
    texts = (use, "s_wait_alu depctr_va_vdst(0)", load)
    items = [asm.make_item("\t" + t) for t in texts]
    entry = asm.Flow([asm.State()], [asm.XState(False)], [[]])
    head = entry
    for _ in range(asm.VALU_HISTORY + 4):
        lines, exit_flow = asm.emit_block(items, head)
        joined = asm.Flow.join([entry, exit_flow])
        if joined.sig() == head.sig():
            break
        head = joined
    else:
        pytest.fail("no fixed point")
    assert [line.split(";")[0].strip() for line in lines] == ["s_wait_dscnt 0x0", *texts]


def test_asm_model_scalar_operands_of_a_division():
    """v_div_scale writes its carry-out (operand 1) and v_div_fmas reads vcc, as asm tells bwd_dkdv_sched's
    dependence graph and its own XCNT rule (the forward's graph has its parser, fwd_sched.mk, with the same
    model: test_forward_division_reads_its_own_carry checks the code it produces)."""
    asm = _module("asm")

    def scalars(text):
        return asm.sgpr_defs_uses(asm.parse_line(-1, "\t" + text))

    assert scalars(f"v_div_scale_f32 {_vr(1)}, vcc_lo, 1.0, {_vr(2)}, 1.0") == (["vcc"], ["exec"])
    assert scalars(f"v_div_scale_f32 {_vr(1)}, null, {_vr(2)}, {_vr(2)}, 1.0") == ([], ["exec"])
    assert scalars(f"v_div_fmas_f32 {_vr(1)}, {_vr(1)}, {_vr(3)}, {_vr(4)}") == ([], ["vcc", "exec"])
    assert scalars(f"v_cmp_lt_f32_e64 s91, 0, {_vr(2)}") == (["s91"], ["s91", "exec"])


@pytest.mark.parametrize("masked", [False, True])
def test_backward_tile_schedule_keeps_the_dependence_graph(masked):
    """The slot scheduler reorders a tile's instructions; every instruction still follows the ones it depends on."""
    sched = _sched("bwd_dkdv")
    ops = sched.loop_ops(sched._ops())
    nodes, wmmas = sched.step_nodes(sched.Layout(sched.BASE), ops, masked, 0, sched.loop_scalars(ops, 1, 1))
    order = sched.schedule_slot(nodes, wmmas)
    at = {id(n): i for i, n in enumerate(order)}
    assert sorted(at.values()) == list(range(len(nodes)))
    assert all(at[id(nodes[p])] < at[id(n)] for n in nodes for p, _ in n.preds)
    assert [n for n in order if n.kind == "WMMA"] == [nodes[i] for i in wmmas]


def test_forward_tile_schedule_keeps_the_dependence_graph():
    sched = _sched("fwd")
    ops = {"kb": 0, "vb": 1, "dl": 2, "c2": "s40"}
    nodes = sched.tile_nodes(sched.Layout(sched.BASE), ops, 0)
    order = [n for n in sched.schedule(nodes) if n.idx >= 0]  # (the scheduler adds its s_wait_dscnt lines)
    assert sorted(n.idx for n in order) == list(range(len(nodes)))
    at = {n.idx: i for i, n in enumerate(order)}
    assert all(at[p] < at[n.idx] for n in nodes for p, _ in n.preds)


def _use_this_tree():
    """As a script: the generators come from the tree this file is in, not from an installed primus_turbo. They
    use the standard library only, so their packages are stand-ins (no __init__ runs, nothing else is imported)."""
    root = pathlib.Path(__file__).resolve().parents[3]
    parts = _PACKAGE.split(".")
    for n in range(1, len(parts) + 1):
        package = types.ModuleType(".".join(parts[:n]))
        package.__path__ = [str(root.joinpath(*parts[:n]))]
        sys.modules[package.__name__] = package
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prints REGIONS for this tree's generators.")
    parser.add_argument(
        "--dump",
        metavar="DIR",
        type=pathlib.Path,
        help="also write every region there, <generator>_<region>.s and .constraints, to diff against another tree's",
    )
    args = parser.parse_args()
    print(f"generators of {_use_this_tree()}", file=sys.stderr)
    for key in [(s, n) for s in ("fwd", "bwd_dkdv") for n in _sched(s).OPERANDS]:
        print(f'    ("{key[0]}", "{key[1]}"): "{_digest(*key)}",')
        if args.dump is not None:
            args.dump.mkdir(parents=True, exist_ok=True)
            for suffix, text in zip((".s", ".constraints"), _sched(key[0]).region_asm(key[1])):
                (args.dump / f"{key[0]}_{key[1]}{suffix}").write_text(text + "\n")
