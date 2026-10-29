# asmkit: gfx1250 assembly backend for spliced hot loops

`asmkit` turns a hot loop given as "real instructions in their final order" into assembly that is valid on gfx1250,
splices it into a kernel that the compiler built (the *shell*), assembles it exactly as the clang driver does, and
checks that the code object holds what was intended. It is the backend that the dK/dV loop generator, the forward
hot-loop generator and the AO list scheduler share. Standard library only.

| module | what it does |
|---|---|
| `isa.py` | `.s` / `llvm-objdump` text model. It keeps the true VGPR numbers (0..1023, from the `/*vN*/` comments), the VGPR and scalar defs / uses (with `s[a:b]` ranges, vcc / exec / m0 / scc), and the instruction classes. It also finds loops and functions, and checks parse coverage: every register token of a line must be in its defs / uses. |
| `expert.py` | Expert-scheduling-mode in-flight model: `va_vdst` (LLVM rule), `vm_vsrc`, dscnt / loadcnt / storecnt / tensorcnt. Also the CFG, VGPR and scalar liveness, and a conservative state join. |
| `emit.py` | Re-derives the *derived* lines for a given order: `s_set_vgpr_msb` (field map learned from the compiler's output, VOPD by position), expert waits, `s_wait_xcnt`, hazard `v_nop` (WMMA→WMMA A/B 5, WMMA→VALU 4, TRANS→VALU 1), and WMMA reuse flags. It works over a multi-block loop region: waits are the minimum over every path (entry and back edges, iterated to a fixed point). Fence waits (`tensorcnt`, `kmcnt`, waits before `s_barrier_signal`) stay where they are. |
| `splice.py` | Finds the shell loop (a natural loop; it may have rare-path blocks and a rotated latch) and builds its **entry register map**: live-in VGPR tuples with their roles (WMMA A / B / C, DS / VMEM address or data), scalar live-ins (buffer resource, TDM descriptor, loop-carried counter), live-outs at each exit, free VGPRs, and what is in flight at entry. It also does the identity rebuild (`rebuild`), splices generated code (`generate`), and runs the static check (`check`): no hazard left open inside the region on any path, and no new hazard after any exit. |
| `build.py` | Assembles with the driver's own `-cc1as` / `lld` commands (from `-###`, or rebuilt from the `-mllvm` flags). An unedited `.s` gives the byte-identical code object. Writes the notes and the listing. |
| `verify_obj.py` | Compares the intended `.s` with the objdump, instruction by instruction: mnemonic, true VGPRs, scalars, MSB immediates, branch targets. |

## Entry model (`--entry`)

The state at the loop header can be modelled in three ways:
- `cfg`: exact forward dataflow from the kernel entry. This is the compiler's view, and the compiler's own loops have 0 open hazards under it.
- `preheader`: the preheader block walked from an unknown state. This was the original AO model. It cannot see VMEM loads issued earlier: the AO identity rebuild `ao_id` (e4b638c8) reads prologue K/V fragments (`buffer_load` into v[786:793], …) in its first trip with no `loadcnt` wait on that path.
- `both` (default): every wait satisfies both models. Use it for anything that goes to a GPU.

## Command line

```
PYTHONPATH=primus_turbo/hipkittens python -m asmkit loops    K.s
PYTHONPATH=primus_turbo/hipkittens python -m asmkit map      K.s --nwmma 240 --out map.json
PYTHONPATH=primus_turbo/hipkittens python -m asmkit identity K.s OUT.s --nwmma 240 [--no-expert] [--entry both]
PYTHONPATH=primus_turbo/hipkittens python -m asmkit generate K.s GEN.s OUT.s --label .LBB0_25
PYTHONPATH=primus_turbo/hipkittens python -m asmkit build    OUT.s OUT.hsaco --main-file-name k.hip \
    --mllvm=-amdgpu-expert-scheduling-mode
PYTHONPATH=primus_turbo/hipkittens python -m asmkit verify-obj OUT.s OUT.objdump.s
```

## Generator contract (`splice.generate`)

The generated region text is made of labels and real instructions; derived lines in it are dropped and re-derived.
- It must contain the shell's header label. If the preheader falls through into the region, the text must start with that label.
- It may leave the region only by branching to the shell's exit labels, or by falling through at its end where the shell did.
- The MSB mode is 0 at every block boundary; the emitter makes sure of this.
- VGPRs: the registers in the entry map's `vgpr_live_in` hold the entry values. Every register live at an exit must hold the right value there. Everything else in `vgpr_free_in_loop` is free.
- If the region needs more VGPRs than the kernel allocates, raise `.amdhsa_next_free_vgpr` / `.vgpr_count` with `splice.set_vgpr_count`.

## Scope

These checks share the generator's model, so they catch emitter and splice mistakes. They are not an independent proof. Before any transformed or generated code runs on a GPU, it needs the independent verification in `asmkit/verify`: symbolic multi-trip equivalence against the shell, address bounds with real kernel arguments, and the preflight stamp. That code must not import these modules.
