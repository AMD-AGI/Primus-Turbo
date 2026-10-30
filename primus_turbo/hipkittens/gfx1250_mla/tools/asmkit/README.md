# asmkit: gfx1250 assembly toolkit of the kernel generators

`asmkit` is the backend of the two generators in `tools/`: `fwdgen` (`k_mla_fwd`) and `bwdgen` (`k_dkdv96ds`). A
generator compiles its HIP marker shell and generates the real instructions of every marked region in their final
order. asmkit adds the lines that this order implies (`s_set_vgpr_msb`, expert scheduling mode waits, hazard
`v_nop`s), re-checks the result, and assembles and verifies the spliced kernel. Standard library only.

| module | contents |
|---|---|
| `isa.py` | Text model of a compiler `.s` or an `llvm-objdump -d` listing: one `Ins` per line with the true VGPR numbers (0..1023, from the `/*vN*/` comments), VGPR defs / uses and class; `sgpr_defs_uses`; the kernel function's line range; `coverage_errors` (every register token of a line is in its defs / uses). |
| `expert.py` | In-flight model of expert scheduling mode: `State`, the `va_vdst` rules (`va_need`), `vm_vsrc`, `dscnt` / `loadcnt` / `storecnt`, state joins; `blocks`, the kernel's basic blocks. |
| `emit.py` | `make_item`; the `s_set_vgpr_msb` field map (`learn_msb_map`, `merge_msb_maps`); `Flow`, what may reach a program point; `emit_block`, which adds the derived lines to a block; `walk_block`, which re-walks emitted lines and reports open hazards. |
| `build.py` | `set_vgpr_count`; `driver_steps` / `assemble`: assembly with the clang driver's own `-cc1as` / `lld` commands (an unedited `.s` gives the compiler's code object byte for byte), plus the notes and the listing. |
| `verify_obj.py` | `compare`: the intended `.s` against the code object's listing, instruction by instruction (mnemonic, true VGPRs, scalars, `s_set_vgpr_msb` immediates, branch targets). |

## Hazard and wait model

- `s_wait_alu depctr_va_vdst(N)`, N <= 14: before a memory instruction that reads or writes a register an in-flight
  VALU writes, or writes one it reads. WMMA, TRANS and core VALU complete in order within their class, so N may be
  the number of counted same-class instructions issued since (the per-class bound). Rules (`expert.va_need`):
  `"llvm"` (fwdgen) is LLVM's, N = 0 once more than one class is pending; `"llvm_war_xdl"` (bwdgen) is LLVM's except
  that a write-after-read against an in-flight WMMA keeps the per-class bound; `"class"`, the per-class bound itself,
  is what `walk_block` checks.
- `s_wait_alu depctr_vm_vsrc(N)`, N <= 6 (7 means no wait): before an instruction that writes a register an
  in-flight memory instruction still reads.
- `s_wait_dscnt` / `loadcnt` / `storecnt N`: before an instruction that touches a register an in-flight load
  writes; `s_wait_dscnt 0` before `s_barrier_signal` while LDS instructions are in flight.
- `s_wait_xcnt 0`: before a write of a scalar register that an in-flight VMEM / FLAT / TDM instruction still reads.
  `s_set_vgpr_msb` does not wait for XCNT.
- `v_nop`, counted in VALU-class instructions: WMMA D -> WMMA A / B: 5; WMMA -> VALU that reads its D or writes its
  D / A / B: 4; TRANS -> an immediately following non-TRANS VALU that reads its result or writes its source: 1.
  Other VALU -> VALU dependences are interlocked.
- Not modelled: dependences through SGPRs / vcc from a VALU to a SALU or memory instruction (`va_sdst`, `va_vcc`);
  the generated code must not need those waits.

## Generator contract

- Items (`make_item`) are real instructions in their final order: no `s_set_vgpr_msb`, `s_wait_alu`,
  register-driven `s_wait_*cnt` or `v_nop`, no VOPD, no WMMA reuse flags. `s_wait_*cnt` items (fences) stay in place
  and count.
- The field map covers every VGPR operand of the items: learn it from compiler output (the shell and an
  LLVM-compiled reference) and pin what that leaves unknown or ambiguous.
- The entry `Flow` over-approximates what is in flight when the block starts (`State(unknown=True)`: anything, on
  any VGPR), and the MSB mode is 0 there. `emit_block` returns the flow at the block end and leaves mode 0 there and
  before a branch; the blocks of a loop are emitted until their entry flows (`Flow.join`) reach a fixed point.
- `walk_block` from the same entry flow must report nothing, and `isa.coverage_errors` must be empty for every
  emitted line. The walk shares the emitter's model: it catches emitter mistakes, not model mistakes.
- After splicing: `set_vgpr_count` if the code needs more VGPRs than the kernel allocates, `assemble` with the
  `driver_steps` of the shell's compile command, and `verify_obj.compare` against the listing.
