###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The code the generators of the gfx1250 MLA kernels emit (fwd_sched and bwd_dkdv_sched on the assembly model asm).
Host-only and standard library: it runs wherever the tests run, whatever the GPU.

The kernels' inline asm is the product of two list schedulers and of a model of the hardware's hazards, so a change to
any of them changes code that was validated on an MI455X. REGIONS pins that code. After a deliberate change, run
test_attention_flydsl_gfx1250_mla.py and test_attention_flydsl_gfx1250_mla_bwd.py on an MI455X, then update the
digests with ``python tests/pytorch/ops/test_attention_flydsl_gfx1250_mla_codegen.py``.
"""

import hashlib
import importlib
import re

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
    ("bwd_dkdv", "MAIN"): "6443989428f1c2fe2dc5cb27d0a2e3eeb27a4896c6673d143dba36f153cd873f",
}
_EVERY_REGION = pytest.mark.parametrize(("sched", "name"), list(REGIONS), ids=["_".join(k) for k in REGIONS])


def _module(name):
    """asm, fwd_sched or bwd_dkdv_sched, imported on first use (test_attention_flydsl_gfx1250_mla checks that
    collecting the tests imports no kernel module)."""
    return importlib.import_module(f"primus_turbo.flydsl.attention.gfx1250_mla.{name}")


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


if __name__ == "__main__":
    for key in REGIONS:
        print(f'    {key!r}: "{_digest(*key)}",')
