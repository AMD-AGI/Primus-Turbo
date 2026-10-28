DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, 7b22589a8076 GPU None, 2026-10-28 02:12:15

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r7r | 2.3632 | 1512 |  |  |  | 1301 | 3.075 | - | - | WMMA reuse-hint knobs (HKL_ABL_NOREUSE, HKL_DQ_REUSE), all default: == r7 bytes |
| r7rnp | 2.3944 | 1493 | r7r | 0.989x | 0.969x | 1327 | 3.172 | - | - | control: r7r without the dK/dV asm patch |
| rr_noreuse | 2.3980 | 1491 | r7rnp | 0.998x | 0.996x | 1330 | 3.182 | - | - | ablation: dK/dV without any WMMA operand-reuse hint (bits unchanged) |
| rr_dqreuse | 2.3636 | 1512 | r7r | 0.999x | 0.997x | 1304 | 3.083 | - | - | dq: matrix_b_reuse on the 11 of 12 consecutive WMMAs that share a dS operand (bits unchanged) |
| rr_dqreuse#null | 2.3648 | 1511 |  |  |  | 1300 | 3.078 | - | - | dq: matrix_b_reuse on the 11 of 12 consecutive WMMAs that share a dS operand (bits unchanged) |
