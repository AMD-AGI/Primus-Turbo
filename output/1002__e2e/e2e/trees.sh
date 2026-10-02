# sourced by drive.sh / preflight.sh / analyze.sh -- the arms of the 2026-10-02 A0 e2e.
#   fly    = fwd r16 (r13ns, e2e copy without the hipBLASLt re-point) + bwd s6 (re-pinned to flydsl 0.3.4.1)
#   flyr29 = fwd r16 (same copy)                                      + bwd r29 (0.3.4.1 pin, the B0 09-28 e2e arm)
#   asm    = aiter ASM fwd + hand-launched ASM bwd (+ host GQA sum)
KIT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/e2e
B0E=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e
FWD_TREE=$KIT/arms/fwd_r16_imglib
BWD_S6=$B0E/arms/bwd_s6_0341          # created 2026-10-02 08:01 by the s6 re-pin agent (source copy: output/1002__e2e/arms_src/bwd_s6_0341)
BWD_R29=$B0E/arms/bwd_r29_0341
E2E_FLY_TREES_JSON="{\"fly\":{\"fwd\":\"$FWD_TREE\",\"bwd\":\"$BWD_S6\"},\"flyr29\":{\"fwd\":\"$FWD_TREE\",\"bwd\":\"$BWD_R29\"}}"
export E2E_FLY_TREES_JSON
# expected md5 of the files that carry kernels (preflight refuses on a mismatch; PREFLIGHT_ALLOW_SRC_DIFF=1 overrides)
S6_KERNELS_MD5=5e61678d53260a2fc17c68286298f4b9       # output/0930__bwd/armsrc/s6/kernels.py (champion s6)
S6_IMPL_MD5=46e0e8127bd57c888944bbc6ee96249b          # e2e port of output/0930__bwd/armsrc/s6/impl.py (e8015b95) + two env
                                                      # switches, defaults = s6 (output/1002__e2e/patches/s6_impl_env_switch.diff):
                                                      #   FLY_BWD_SIDE_STREAM=0   dQ chain serial on the caller's stream (r29 order)
                                                      #   FLY_BWD_RECORD_STREAM=0 keep the side stream, skip record_stream() at the join
S6_SRC_MD5SUMS=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__e2e/arms_src/bwd_s6_0341/MD5SUMS   # the e2e copy must equal it
# fly arm stream mode for the e2e: side stream ON (as benchmarked), record_stream OFF. The join
# (caller stream waits for the side stream before returning; every block the side stream touches is
# referenced until then) already orders any reuse after the side stream's work; record_stream would
# only defer reuse of ~1.1 GiB per call while the host runs ahead (up to ~4.6 layers between nkfix
# syncs, i.e. up to ~5 GiB) against 2.2 GiB of memguard headroom at 32L. GPU work is identical.
FLY_BWD_ENV="-e FLY_BWD_SIDE_STREAM=${FLY_BWD_SIDE_STREAM:-1} -e FLY_BWD_RECORD_STREAM=${FLY_BWD_RECORD_STREAM:-0}"
R29_KERNELS_MD5=37f37052eb739579555d6bdf62a829cc      # job round 29 kernels.py (B0 09-28 e2e)
R29_IMPL_MD5=08cb8533d75e82198fabb9514fd18ba5
PIN0341_ENV_MD5=7df61bba26309ffc14b84a432fb451a7      # the 0.3.4.1 _env.py shared by bwd_r20/r29/s6 _0341 trees
# schedules: step 1 (and 2 in 3arm) is ASM -- no FlyDSL in step 1 (09-28: both FlyDSL-in-step-1 runs went bad)
SPEC_3ARM_P1="asm;asm,fly,flyr29,asm,flyr29,fly"
SPEC_3ARM_P2="asm;asm,flyr29,fly,asm,fly,flyr29"
STEPS_3ARM=92; PFREQ_3ARM=11          # 11 is coprime with the 6-step cycle: every arm gets profiled
SPEC_2ARM_P1="asm,fly,asm,fly;asm,fly,fly,asm"   # B0 final recipe, process a (ABBA)
SPEC_2ARM_P2="asm,fly,asm,fly;fly,asm,asm,fly"   # B0's process b (BAAB) with the warm-up made ASM-first
STEPS_2ARM=62; PFREQ_2ARM=10
