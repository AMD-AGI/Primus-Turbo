DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, 7b22589a8076 GPU None, 2026-10-28 02:07:45

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r7c | 2.3656 | 1511 |  |  |  | 1300 | 3.075 | - | - | cache-policy / dq head-order knobs (HKL_DS_ST_AUX, HKL_KV_TDM_CPOL, HKL_DQ_*), all default: == r7 bytes |
| r7cnp | 2.3920 | 1494 | r7c | 0.988x | 0.971x | 1324 | 3.167 | - | - | control: r7c without the dK/dV asm patch |
| cp_st1 | 2.3918 | 1494 | r7cnp | 1.001x | 1.000x | 1324 | 3.168 | - | - | dS^T export stores th NT |
| cp_st3 | 2.3921 | 1494 | r7cnp | 1.001x | 0.999x | 1324 | 3.171 | - | - | dS^T export stores th WB (high-temporal with write-back in MALL) |
| cp_st5 | 2.3963 | 1492 | r7cnp | 1.000x | 0.997x | 1327 | 3.177 | - | - | dS^T export stores th RT_NT (regular in CU / SE, non-temporal in MALL) |
| cp_kv2 | 2.3946 | 1493 | r7cnp | 0.999x | 0.998x | 1326 | 3.173 | - | - | dK/dV Q / dO ring TDM loads th HT |
| cp_dqds1 | 2.3596 | 1515 | r7c | 1.003x | 1.003x | 1302 | 3.071 | - | - | dq dS ring TDM loads th NT (K regular) |
| cp_dqds3 | 2.3586 | 1515 | r7c | 1.004x | 1.003x | 1302 | 3.067 | - | - | dq dS ring TDM loads th LU (last use) |
| cp_dqds3#null | 2.3551 | 1518 |  |  |  | 1302 | 3.069 | - | - | dq dS ring TDM loads th LU (last use) |
