# Offline FlyDSL audit scripts

Every script here runs in a container with **no `/dev/kfd`** and cannot touch the card:

```bash
docker run --rm --network none \
  -v $PWD/output/0917__flydsl/bin/<script>:/tmp/s.py:ro \
  --entrypoint python3 fa-tune:deps /tmp/s.py
```

Results are written up in `../API-DELTA.md`. Re-run them after any flydsl version change.

| script | answers |
|---|---|
| `api_audit_1_symbols.py` | which modules/attributes aiter's gfx1250 forward needs exist in the installed flydsl. **Its `T.*` section reports false MISSINGs** - `T.bf16` needs an MLIR Context; script 2 is the corrected version |
| `api_audit_2_typing_and_gaps.py` | `T.*` inside `with Context():`, and where the four absent symbols actually live |
| `api_audit_3_subattrs_rocdl.py` | sub-attributes, the rocdl gfx1250 primitives, and the `tdm_ops` member list |
| `wave_audit_1_decision_points.py` | every wave-size decision point inside the package |
| `wave_audit_2_is_rdna_arch.py` | the `is_rdna_arch` source and its verdict per arch string |
| `wave_audit_3_env_override.py` | whether `FLYDSL_GPU_ARCH` / `HSA_OVERRIDE_GFX_VERSION` can override it |
| `wave_audit_4_patch_reach.py` | whether patching `runtime.device` reaches the consumers (it does not) |

`wave_audit_4`'s section 4 prints `None` for both arches: it probes a helper by a name that
does not exist. The buffer-descriptor flag difference is read from source instead
(`expr/buffer_ops.py:68`), and that is what `API-DELTA.md` cites.
