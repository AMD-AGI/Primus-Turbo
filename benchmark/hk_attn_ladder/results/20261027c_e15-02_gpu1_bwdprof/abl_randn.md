DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, c822521170f1 GPU None, 2026-10-27 15:13:39

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r7 | 2.4781 | 1442 |  |  |  | 1253 | 3.101 | - | - | redundant causal test dropped from the masked dK/dV loop |
| r7np | 2.5217 | 1417 |  |  |  | 1290 | 3.254 | - | - | control: r7 without the dK/dV asm register-retarget patch (the ablations below cannot carry it) |
| abl_nods | 2.3302 | 1534 | r7np | 1.084x | 1.014x | 1378 | 3.210 | - | - | ablation (wrong results): full loop exports no dS^T (no DPP regroup, no 4.33 GB of stores) |
| abl_notdm | 2.3917 | 1494 | r7np | 1.055x | 1.019x | 1334 | 3.194 | - | - | ablation (wrong results): full loop streams no Q / dO tiles into the ring (stale LDS) |
| abl_noexp | 2.4815 | 1440 | r7np | 1.016x | 1.015x | 1293 | 3.205 | - | - | ablation (wrong results): no v_exp in the softmax |
| abl_nosmx | 2.4348 | 1468 | r7np | 1.036x | 1.048x | 1276 | 3.104 | - | - | ablation (wrong results): no softmax / dS VALU (P = S, dS = dP; NL / ND loads dead) |
| abl_dq_nods | 2.3468 | 1523 | r7np | 1.073x | 1.030x | 1344 | 3.158 | - | - | ablation (wrong results): dq's dS ops are 16-B dummies (no 4.33 GB dS read) |
| abl_dq_nok | 2.4607 | 1453 | r7np | 1.025x | 1.004x | 1318 | 3.240 | - | - | ablation (wrong results): dq's K ops are 16-B dummies |
| abl_dq_nok#null | 2.4591 | 1453 |  |  |  | 1318 | 3.242 | - | - | ablation (wrong results): dq's K ops are 16-B dummies |
