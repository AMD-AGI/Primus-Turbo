# gfx1250 MLA forward (HipKittens line, W-S1a)

Causal attention forward of the DeepSeek-V3 MLA head dims (q/k 192, v 128, bf16, MHA, bottom-right causal) on gfx1250,
bit for bit the shipped FlyDSL forward (`primus_turbo/flydsl/attention/gfx1250_mla`): same out (bf16) and lse (fp32,
natural log). Status: round 4, step W-S1a (CPU-built and CPU-checked; no GPU run yet). Not wired into Turbo: the
Python integration (`impl.py`, `hsaco.py`, manifest, the `PRIMUS_TURBO_MLA_FWD_IMPL` switch, tests) is FWD-INT's.

## Layout

| path | what |
|---|---|
| `hip/k_mla_fwd.hip` | the **shell**: LLVM compiles the prologue (block remap, TDM descriptors and the 4-stage K/V ring, Q staging), the ring barrier waits, the loop control and the O / LSE buffer stores. The per-tile work sits at inline-asm `HKGEN` markers. `-DHKGEN_REF=1` puts the same per-tile math in HIP instead (the reference; a class-1 kernel of its own). |
| `hip/hkf.h` | the raw gfx1250 builtins the shell uses (a self-contained subset of the backward's `hk1.h`) |
| `gen/fwdgen.py` | the generated regions: `INIT`, `TILE_C` (no mask), `TILE_M` (causal mask on all four q tiles), `FINAL_0..3`; register layout, dependence graph, list scheduler |
| `gen/build.py` | compile the shell, read the markers, generate, derive MSB / waits / v_nops (`asmkit.emit`), splice, assemble (`asmkit.build`), compare (`asmkit.verify_obj`), contract checks |
| `tools/emu.py` | CPU emulation of the generated regions of a built code object (one wave), bit for bit against `tools/ref_softmax.py` |
| `tools/ref_softmax.py` | numpy model of FlyDSL's forward (op order from its gfx1250 ISA) |
| `tools/fwd_ab.py` | GPU driver: canary, bit-exactness vs FlyDSL, poison, determinism, HK bwd fed with HK fwd, same-process A/B |
| `bin/k_mla_fwd.hsaco` | the built code object (`--sched list`, generic masked body; sha256 in `bin/SHA256SUMS`). `--diag` (diagonal-tile bodies `TILE_D0..3`) and `--sched program` build the other variants |

## Kernel

`k_mla_fwd(Q, K, V, O, LSE, scale, Sq, Skv, H, B, nqb)`, grid `(nqb = ceil(Sq / 256), H, B)` through FlyDSL's
longest-first / head-grouped block remap, block 128 (4 waves, 1 per SIMD), LDS 274432 B (one workgroup per CU),
1024 VGPRs. Wave `w` owns the 16-row q tiles `{w, w + 4, w + 8, w + 12}` of the 256-row block. Per 64-kv tile:
Q / K / V fragment reads (FlyDSL's lane patterns), the ring barrier's arrive once the stage's reads are issued, then
per q tile `QK -> [mask] -> softmax -> O *= corr -> PV`, list-scheduled across the four q tiles around an in-order
WMMA stream (no cross-tile pipelining yet). Domain: `Sq == Skv`, `Sq % 64 == 0`, `B * Sq * H * 128 * 2 < 2^31`
(the caller falls back to FlyDSL otherwise).

Bit-exactness argument: every op is FlyDSL's (see the `fwdgen` docstring); two differences in form only: O is
multiplied by `corr` on every tile (FlyDSL: only when some row of the q tile rescales; `corr = exp2(m - m) = 1`
exactly otherwise), and the fully masked q tiles of the diagonal tiles run (FlyDSL skips them): for finite inputs
`corr = 1`, `p = exp2(-inf) = 0`, `d = fma(1, d, +0) = d`, `O + 0 * V = O`. Both rely on `v_exp_f32(+0) == 1.0` and
`v_exp_f32(-inf) == +0` (to be confirmed on hardware by the first bitwise GPU check).

## Generated-code contract (checked by `build.py`)

- writes only VGPRs above every VGPR the shell's code references (`base`), the marker outputs, and the SGPRs the
  markers declare clobbered (`s90..s103`, `vcc`, `scc`); never `exec` / `m0`;
- no global / buffer / flat / scratch access, no TDM; LDS reads only from the marker bases (+ static offsets inside
  the stage / Q layout);
- one `s_barrier_signal` per TILE region (after `s_wait_dscnt 0` + `s_wait_tensorcnt 6`), none elsewhere;
- MSB mode 0 at every region end; no hazard left open (re-walk with asmkit's model from an unknown entry state);
- the shell's code holds no WMMA / TRANS (no hazard into a region); every marker runs with all lanes; the shell never
  references `s90..s103` and writes `vcc` / `scc` before reading them after every marker.

## Build recipe

Inside the `lhz-hk-cpu` container (toolchain `AMD clang version 24.0.0git (… 862306352beb…)`), with the AO track's
`asmkit` on the path (`primus_turbo/hipkittens/asmkit`, or `--asmkit-root`):

```
python3 primus_turbo/hipkittens/gfx1250_mla_fwd/gen/build.py --out OUT [--sched list|program] [--asmkit-root DIR]
```

1. `amdclang++ -x hip --offload-arch=gfx1250 --cuda-device-only --no-gpu-bundle-output -O3 -std=c++20 -DNDEBUG -I.
   -save-temps=obj -mllvm -amdgpu-expert-scheduling-mode -cuid=k_mla_fwd -DHKGEN_REF=0 -c hip/k_mla_fwd.hip`
   (cwd `hip/`); the saved `.s` is `OUT/shell.s`. The same with `-DHKGEN_REF=1` gives `OUT/ref/k_mla_fwd.hsaco`
   (and the DS operand MSB fields for the emitter).
2. markers -> register map (`OUT/build.json`: `base`, marker operands); `fwdgen` regions; `asmkit.emit.emit_region`
   (expert mode, unknown entry state) adds `s_set_vgpr_msb`, waits, hazard `v_nop`s; re-walk must find nothing open.
3. splice in place of the marker bodies, `.amdhsa_next_free_vgpr` / `.vgpr_count` -> 1024: `OUT/k_mla_fwd.s`.
4. assemble with the driver's own `-cc1as` / `lld` commands (`asmkit.build.driver_steps` of step 1's command):
   `OUT/k_mla_fwd.hsaco`, `.notes`, `.objdump.s`; `asmkit.verify_obj` must report 0 mismatches.
5. contract checks on the listing; `VERDICT PASS` and exit 0, else exit 1.

The build is deterministic (two builds: identical sha256).

## CPU checks

```
python3 tools/emu.py --build OUT --ref-dir tools --asmkit-root DIR --shape toy|s640|prod|b1|b4 \
    [--waves BX:HEAD:WAVE,...] [--wmma exact|seq] [--exp2 cr|perturb]
python3 -I <V track>/tools/vbounds.py --kernel fwd --listing OUT/k_mla_fwd.objdump.s --shape toy|s640|prod|b1|b4
```

`emu.py` executes the generated regions of the code object's listing for one wave (shell modelled: LDS image, marker
operands, poisoned shell registers between regions, poisoned scratch registers at every region entry) and compares
out, lse and the fp32 state before the epilogue (m, d, O) bit for bit with `ref_softmax` under the same primitive
models. It shows the operation graph is FlyDSL's; it cannot show that WMMA / `v_exp_f32` / `v_log_f32` round like the
models (the GPU check does). The independent preflight stamp (V track) is required before any GPU run.
