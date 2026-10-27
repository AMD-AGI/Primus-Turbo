// bwd1/v1_step.inc -- the body of ONE step of k_hk_bwd1_g4 (hip/src/k_attn_bwd_v1.hip), textually included
// (a) by the `step` lambda (every default build: byte-identical to the former inline body) and
// (b) by kernel-scope blocks when BWD1_OPT_REGS != 0 (hk/notes/bwd_v11.md): the llvm.fake.use register holds
//     below must be lexically in the kernel, because the inliner drops the fake uses of an inlined callee.
// The includer provides: constexpr bool MASKED; constexpr int P4, P, NX; int s (step); XSet &X (tile s on
// entry, tile s + 1 on exit); the macro BWD1_KS (1 = kernel scope: holds active, dq loop-carried in dq_c); with
// BWD1_OPT_MASKED=2 also constexpr bool MDEAD (this body is a diagonal step's DEAD-wave path).
// Every BWD1_OPT_* addition sits under #if, so a build without them compiles exactly the former lambda body.
#if BWD1_OPT_MASKED == 1
        constexpr bool IL = b1::SCHED == 1; // BWD1_OPT_MASKED=1: diagonal steps run the FULL step's interleave too
#elif BWD1_OPT_MASKED == 2
        constexpr bool IL = b1::SCHED == 1 && !(MASKED && MDEAD); // =2: live waves interleaved, DEAD waves: v1 bursts
#else
        constexpr bool IL = b1::SCHED == 1 && !MASKED; // interleave patterns (FULL steps only)
#endif
        using Z0  = std::integral_constant<int, 0>;
        using SLN = std::integral_constant<int, NX>;
        const int qt = b0::tile_qt(s, n, G, nqp);
#if BWD1_OPT_MASKED == 1
        // every wave kind computes the whole step (no kind branch): a DEAD wave's exp2 arguments are all
        // MASK_NEG, so P = +0 and dS = +-0 (dK / dV / dQ bits unchanged, as BWD1_SKIP_DEAD=0)
        constexpr bool live = true;
#elif BWD1_OPT_MASKED == 2
        // the wave-kind branch sits outside the step (one body per path): DEAD waves skip S/dP, softmax, dV / dK and
        // the E tr16 exactly as v1's diagonal body; FULL / DIAG waves run the interleaved FULL body with the mask
        constexpr bool live = !(MASKED && MDEAD);
#else
        int       kind = b0::K_FULL;
        if constexpr (MASKED) kind = b0::wave_kind(kvw0, w, qt, cshift, causal);
        const bool live = !MASKED || !BWD1_SKIP_DEAD || kind != b0::K_DEAD;
#endif
#if (BWD1_OPT_REGS & 8) && BWD1_KS
        acc_t      S[2][2], dP[2][2];
        acc_t (&dq)[2][2] = dq_c; // loop-carried: dQ(s - 1)'s registers stay allocated through G0 of step s + 1
#else
        acc_t      S[2][2], dP[2][2], dq[2][2];
#endif
#if (BWD1_OPT_REGS & 16) && BWD1_KS
        u32x4      pu[2][2];
        u32x4 (&du)[2][2] = du_c; // loop-carried: E2's B / A operands stay allocated through G0 of step s + 1
        opnd_t     dOT[8];
        opnd_t (&QT)[8] = qt_c;
#else
        u32x4      pu[2][2], du[2][2];
        opnd_t     QT[8], dOT[8];
#endif
        auto sm = [&](auto hh, auto kh) { // softmax / dS of block (hh, kh), v0 design 6.2
            acc_t t = softmax_arg_rows(S[hh][kh], c1, X.ld[hh][0], X.ld[hh][2]);
            if constexpr (MASKED) t = mask_col_gt_row(t, l, b0::mask_off(qt, hh, kvw0, w, kh, cshift), MASK_NEG);
            const acc_t p = exp2_acc(t), d = dsoftmax(p, dP[hh][kh], X.ld[hh][1], X.ld[hh][3]);
            pu[hh][kh] = cvt_pk_bf16(p);
            du[hh][kh] = cvt_pk_bf16(d);
        };
        auto sdp = [&](auto kh) { // S / dP of kv tile kh: 4 independent chains over dt
            static_for<4>([&](auto dt) {
                static_for<2>([&](auto hh) {
                    if constexpr (dt == 0) S[hh][kh] = wmma0(X.qA[hh][dt], kf[kh][dt]);
                    else S[hh][kh] = wmma(X.qA[hh][dt], kf[kh][dt], S[hh][kh]);
                });
                static_for<2>([&](auto hh) {
                    if constexpr (dt == 0) dP[hh][kh] = wmma0(X.dA[hh][dt], vf[kh][dt]);
                    else dP[hh][kh] = wmma(X.dA[hh][dt], vf[kh][dt], dP[hh][kh]);
                });
            });
        };
        auto load_dot = [&] { static_for<8>([&](auto dt) { dOT[dt] = load_tr<XP>(do_tr + P4 * b0::QT_B, 0, 16 * dt); }); };
        auto load_qt  = [&] { static_for<8>([&](auto dt) { QT[dt] = load_tr<XP>(q_tr + P4 * b0::QT_B, 0, 16 * dt); }); };
        // E WMMAs [LO, HI), kh-major (an A operand dies only in the second kv-tile pass): e < 16 dV^T[dt][kh] +=
        // dOT[dt] P[kh] with kh = e / 8, dt = e % 8; e >= 16 dK^T[dt][kh] += QT[dt] dS[kh] likewise
        auto e_wmma = [&](auto lo_c, auto hi_c) __attribute__((always_inline)) {
            constexpr int LO = decltype(lo_c)::value, HI = decltype(hi_c)::value;
            static_for<b1::NE>([&](auto e) {
                if constexpr ((int) e >= LO && (int) e < HI) {
                    constexpr int kh = (e % 16) / 8, dt = e % 8;
                    if constexpr (e < 16) dVt[dt][kh] = wmma(dOT[dt], cat_u32(pu[0][kh], pu[1][kh]), dVt[dt][kh]);
                    else dKt[dt][kh] = wmma(QT[dt], cat_u32(du[0][kh], du[1][kh]), dKt[dt][kh]);
                }
            });
        };

        // T: ring op s + 3 (group 0 built under E2 of step s - 1)
#if BWD1_OPT_TDM3
        tdm_load(g0n, rg1); // BWD1_OPT_TDM3: wave 3 issues its dummy op (constant descriptor): no role branch
#else
        if (w < b0::ISS_KV) tdm_load(g0n, rg1);
#endif
        sched_barrier<0>();
        // G0: S/dP kv tile 0 (its A operands do not die here) || dSa(s - 1) (buffer P ^ 1), dOT(s)
        if constexpr (!IL) {
            load_dsa(std::integral_constant<int, P ^ 1>{});
            if (live) load_dot();
            sched_barrier<0>();
            if (live) sdp(I0);
        } else {
            sdp(I0);
            load_dsa(std::integral_constant<int, P ^ 1>{});
            load_dot();
            pattern<16, 0, 0, 32>();
        }
        sched_barrier<0>();
#if (BWD1_OPT_REGS & 8) && BWD1_KS
        BWD1_KEEP_ACC4(dq); // dQ(s - 1)'s atomic data + voffset registers: not reused by the G0 loads (vm_vsrc WAR)
        hka_llvm_fake_use(avo_c);
#endif
#if (BWD1_OPT_REGS & 16) && BWD1_KS
        BWD1_KEEP8_1(QT); // step s - 1's E2 operands: not reused by the G0 loads (WAR on the last E2 WMMAs)
        BWD1_KEEP4_U(du);
#endif
        // G1: S/dP kv tile 1 || softmax kv tile 0
        if (live) {
            sdp(I1);
            sm(I0, I0);
            sm(I1, I0);
#if BWD1_OPT_MASKED
            pattern<16, 56 + (MASKED ? b1::MASK_VALU : 0), 0, 0>();  // live bodies only
#else
            pattern<16, 56, 0, 0>();
#endif
        }
        sched_barrier<0>();
#if (BWD1_OPT_REGS & 1) && BWD1_KS
        BWD1_KEEP8(X.qA); // G1's WMMA A operands: the softmax kh0 temporaries go elsewhere (no WAR v_nop)
        BWD1_KEEP8(X.dA);
#endif
        // C: dQ(s - 1) || softmax kv tile 1; the atomic voffset of tile s - 1
        dq_wmma(dq);
        if (live) {
            sm(I0, I1);
            sm(I1, I1);
        }
        if constexpr (MASKED) {
            if (!live) static_for<2>([&](auto hh) { static_for<2>([&](auto kh) { du[hh][kh] = zero_u32x4(); }); });
        }
#if (BWD1_OPT_REGS & 8) && BWD1_KS
        avo_c = voff_nonneg(voff_lane + b0::dqa_voff(0, 0, b1::atom_gh(s, n, G, rot), b1::atom_qt(s, n, G, nqp), nqp));
        const int avo = avo_c;
#else
        const int avo = voff_nonneg(voff_lane + b0::dqa_voff(0, 0, b1::atom_gh(s, n, G, rot), b1::atom_qt(s, n, G, nqp), nqp));
#endif
#if BWD1_OPT_MASKED
        if constexpr (IL) pattern<16, 56 + (MASKED ? b1::MASK_VALU : 0), 0, 0, 2>();
#else
        if constexpr (!MASKED) pattern<16, 56, 0, 0, 2>();
#endif
        order_fence(); // every load above stays older than the stores
#if (BWD1_OPT_REGS & 2) && BWD1_KS
        BWD1_KEEP8(dSa); // C's WMMA A operands: the softmax kh1 temporaries go elsewhere (no WAR v_nop)
#endif
        // all DS loads issued so far (dSa, dOT: G0, ~300 cycles ago) are complete here: one explicit wait instead of
        // LLVM's per-WMMA s_wait_dscnt in E1 (its dV WMMAs read dOT while the QT loads are in flight)
        if constexpr (IL) wait_dscnt<0>();
        // ST: dS^T(s) -> buffer P
        static_for<2>([&](auto kh) {
            static_for<2>([&](auto hh) { ds_store_b128(ds_st, P * b0::DS_B + b0::ct_imm(SP, 16 * kh, 16 * hh), du[hh][kh]); });
        });
        order_fence(); // the stores stay older than every later DS op
        // E1: E WMMAs 0..NE1-1 || QT(s) (first 8 gaps: no A operand dies there) + atomics + readback 0..RB1-1
#if BWD1_OPT_MASKED
        constexpr int RB1 = IL ? b1::RB1 : 0;
        constexpr int AT1 = (b1::ATOM == 0 || !IL) ? b1::NATOM : b1::ATOM_E1;
#else
        constexpr int RB1 = MASKED ? 0 : b1::RB1;
        constexpr int AT1 = (b1::ATOM == 0 || MASKED) ? b1::NATOM : b1::ATOM_E1;
#endif
        if constexpr (!IL) {
            if (live) load_qt();
            atomics(Z0{}, std::integral_constant<int, AT1>{}, dq, avo);
            sched_barrier<0>();
            if (live) e_wmma(Z0{}, std::integral_constant<int, b1::NE1>{});
            readback(Z0{}, std::integral_constant<int, RB1>{}, SLN{}, X);
        } else {
            // source order == issue order per gap (atomics are ordered memory ops: the DS loads cannot pass them)
            static_for<b1::NE1>([&](auto g) {
                e_wmma(g, std::integral_constant<int, g + 1>{});
                if constexpr (g < 8) QT[g] = load_tr<XP>(q_tr + P4 * b0::QT_B, 0, 16 * g);
                constexpr int a0 = b1::ATOM == 0 ? (g == 0 ? 0 : AT1) : (AT1 * g) / b1::NE1;
                constexpr int a1 = b1::ATOM == 0 ? AT1 : (AT1 * (g + 1)) / b1::NE1;
                atomics(std::integral_constant<int, a0>{}, std::integral_constant<int, a1>{}, dq, avo);
                if constexpr (g >= 8) {
                    constexpr int r0 = (RB1 * (g - 8)) / (b1::NE1 - 8), r1 = (RB1 * (g - 7)) / (b1::NE1 - 8);
                    readback(std::integral_constant<int, r0>{}, std::integral_constant<int, r1>{}, SLN{}, X);
                }
            });
            static_for<b1::NE1>([&](auto g) {
                sched_group_barrier<sg::WMMA, 1, 0>();
                if constexpr (g < 8) sched_group_barrier<sg::DS_READ, 2, 0>();
                constexpr int na = b1::ATOM == 0 ? (g == 0 ? AT1 : 0) : (AT1 * (g + 1)) / b1::NE1 - (AT1 * g) / b1::NE1;
                if constexpr (na > 0) sched_group_barrier<sg::VMEM, na, 0>();
                if constexpr (g >= 8) {
                    constexpr int nr = b1::rb_loads((RB1 * (g - 7)) / (b1::NE1 - 8)) - b1::rb_loads((RB1 * (g - 8)) / (b1::NE1 - 8));
                    if constexpr (nr > 0) sched_group_barrier<sg::DS_READ, nr, 0>();
                }
            });
        }
        // SIG: QT(s) and the stores retire (the NUNC readback loads may stay in flight)
#if BWD1_OPT_MASKED
        if constexpr (IL) split_signal_tdm_dscnt<b1::TCNT, b1::NUNC>();
        else split_signal_tdm_lds<b1::TCNT>();
#else
        if constexpr (!MASKED) split_signal_tdm_dscnt<b1::TCNT, b1::NUNC>();
        else split_signal_tdm_lds<b1::TCNT>();
#endif
        // E2: E WMMAs NE1..31 || readback items RB1..23 (+ atomics); group 0 of op(s + 4)
        g0n = ring_g0(s + 1 + b1::LEAD);
        asm volatile("" ::"s"(g0n)); // materialize under E2 (IR sinking would move it behind the wait)
        if constexpr (!IL) {
            readback(std::integral_constant<int, RB1>{}, std::integral_constant<int, b1::RB_ITEMS>{}, SLN{}, X);
            sched_barrier<0>();
            if (live) e_wmma(std::integral_constant<int, b1::NE1>{}, std::integral_constant<int, b1::NE>{});
        } else {
            constexpr int NR = b1::RB_ITEMS - RB1, AT2 = b1::NATOM - AT1;
            static_for<b1::NWIN>([&](auto g) {
                e_wmma(std::integral_constant<int, b1::NE1 + g>{}, std::integral_constant<int, b1::NE1 + g + 1>{});
                constexpr int r0 = RB1 + (NR * g) / b1::NWIN, r1 = RB1 + (NR * (g + 1)) / b1::NWIN;
                readback(std::integral_constant<int, r0>{}, std::integral_constant<int, r1>{}, SLN{}, X);
                constexpr int a0 = AT1 + (AT2 * g) / b1::NWIN, a1 = AT1 + (AT2 * (g + 1)) / b1::NWIN;
                atomics(std::integral_constant<int, a0>{}, std::integral_constant<int, a1>{}, dq, avo);
            });
            static_for<b1::NWIN>([&](auto g) {
                sched_group_barrier<sg::WMMA, 1, 0>();
                constexpr int nr = b1::rb_loads(RB1 + (NR * (g + 1)) / b1::NWIN) - b1::rb_loads(RB1 + (NR * g) / b1::NWIN);
                if constexpr (nr > 0) sched_group_barrier<sg::DS_READ, nr, 0>();
                constexpr int na = (AT2 * (g + 1)) / b1::NWIN - (AT2 * g) / b1::NWIN;
                if constexpr (na > 0) sched_group_barrier<sg::VMEM, na, 0>();
            });
        }
#if (BWD1_OPT_REGS & 4) && BWD1_KS
        // E1 / E2's WMMA operands stay allocated to the end of E2: the readback(s + 1) targets cannot be the
        // registers the last E WMMAs read (expert-mode WAR va_vdst(0) drains in E2)
        sched_barrier<0>();
#if BWD1_OPT_MASKED == 2
        constexpr bool EOPS = live; // a DEAD-wave body loads / computes no E operand
#else
        constexpr bool EOPS = true;
#endif
        if constexpr (EOPS) {
            BWD1_KEEP8_1(QT);
            BWD1_KEEP8_1(dOT);
            BWD1_KEEP4_U(pu);
            BWD1_KEEP4_U(du);
        }
#endif
        // WAIT
        split_wait_lds();
