# 0915 opt — HEALTHY — 2026-09-15T08:10:51   uptime 0h47m   idle 0.1%

NOW     (idle) STOP sentinel present
NEXT    nc-time|llama31-8b|1 · nc-time|gqa2k|1 · nc-time|w8s4096|1
        3 queued · 3 blocked on needs · 0 malformed
GPU     HEALTHY  sclk   new faults deg=0 wedge=0  sigkills 0
        budgets: measure 90s  measure_cold 180s  bringup 300s  sweep 300s  e2e 3600s  cpu 600s
DONE    124 ok · 4 wrong · 1 timeout · 2 fault
BEST    bwd 0.732 ms (gen|asm|smoke|1)   must-beat 17.692

LEARNED
  OK final|champ|4            sqnr_db.out >=50  got 53.6196775297861
  OK final|champ|4            sqnr_db.dk >=50  got 50.60819203012865
  OK final|champ|5            sqnr_db.out >=50  got 53.6196775297861
  OK final|champ|5            sqnr_db.dk >=50  got 50.60819203012865
  OK final|prev|1             sqnr_db.out >=50  got 53.6196775297861
  OK final|prev|2             sqnr_db.out >=50  got 53.6196775297861
  OK final|prev|3             sqnr_db.out >=50  got 53.6196775297861
  OK final|prev|4             sqnr_db.out >=50  got 53.6196775297861
  OK final|prev|5             sqnr_db.out >=50  got 53.6196775297861
  OK final|oob|1              sqnr_db.out >=50  got 53.66972664861177
  OK final|oob|2              sqnr_db.out >=50  got 53.66972664861177
  OK final|oob|3              sqnr_db.out >=50  got 53.66972664861177

RECENT
  07:24:37  final|prev|1               ok       15.0s  bwd  17.7662  sqnr 53.62
  07:24:52  final|prev|2               ok       15.0s  bwd  17.6876  sqnr 53.62
  07:25:07  final|prev|3               ok       15.0s  bwd  17.6774  sqnr 53.62
  07:25:22  final|prev|4               ok       15.0s  bwd  17.6766  sqnr 53.62
  07:25:37  final|prev|5               ok       15.0s  bwd  17.6857  sqnr 53.62
  07:25:53  final|oob|1                ok       15.0s  bwd  10.4177  sqnr 53.67
  07:26:08  final|oob|2                ok       15.0s  bwd  10.4817  sqnr 53.67
  07:26:23  final|oob|3                ok       15.0s  bwd   9.0537  sqnr 53.67

ACTION
