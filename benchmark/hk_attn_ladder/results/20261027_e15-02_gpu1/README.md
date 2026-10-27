# 2026-10-27, ctheliosr-rck-g02-e15-02, GPU1 (0002:01:00.0)

Image amdprimus/amdprimus:tas-20261019 (torch 2.14.0+rocm10.2.0a20261011, flydsl 0.3.4.1), branch
dev/lhz/hk_attn_ladder 53a2d33b, container lhz-hkl-g1 (this GPU only). Kernel log clean for the whole session.

* `test_*.json`: stage C correctness (`hkl.test`, every rung + FlyDSL, every shape). All pass except the MLA
  `rect32` shape (b1 Sq 4096 Skv 4128 h4 causal) for r2d .. r7: dq 17.7 dB (the dQ read error, see the ladder
  README); r2c and earlier are bitwise equal to hk_v0 there. At the production shapes every Llama v0 flag variant
  is bitwise equal to v0 and every MLA rung bitwise equal to hk_v0, which is bitwise equal to FlyDSL's backward.
* `ladder_*.md / .json`: stage D timing (`hkl.ladder --side -r 10`, settle 1 s per cell, alternating order,
  flush 512 MB per call). Null controls: fwd 1.0011x +- 0.0030, bwd 0.9985x +- 0.0020, MLA 1.0016x +- 0.0045.
  Sustained sclk 1.2-1.5 GHz for the dense kernels (absolute times ~10% above a08-1's; ratios match).
