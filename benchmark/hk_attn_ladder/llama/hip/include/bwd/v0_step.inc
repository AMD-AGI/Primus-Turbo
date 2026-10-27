// bwd/v0_step.inc -- body of one step of k_hk_bwd_g4 (hip/src/k_attn_bwd_v0.hip; the step's phases P1..P9 are in
// that file's header comment). Included in two places of the kernel, both inside k_hk_bwd_g4:
//   * the `step` lambda (every build without HK_BWD_OPT_REGS, the pinned default included): the code is v0's step
//     body verbatim, so the default build stays byte-identical;
//   * kernel-scope blocks of the main loop (HK_BWD_OPT_REGS builds: the two FULL-loop steps; with bit 3 also the two
//     diagonal-loop steps): llvm.fake.use calls survive only in the function they are written in -- the inliner
//     drops those of an inlined callee (lambda) -- so the register holds of HK_BWD_OPT_REGS must be lexically in
//     the kernel.
// Context names it expects: MASKED (constexpr bool), P (constexpr int, the step parity = dS^T buffer), s (step),
// X / Xn (XSet &: this step's readback set, the readback target), plus the kernel's locals (n, nqp, G, w, l, kvw0,
// cshift, causal, c1, lds0, kf, vf, kt, dVt, dKt, dSa, voff_lane, readback, load_dsa, dq_wmma, dq_atomics, ring_g0,
// rg1, ...; variant builds also dq_atom8 / A0 / A1 (HK_BWD_OPT_ATOM), rb_bases / readback_at (HK_BWD_OPT_ADDR)).
// No include guard: included on purpose more than once. Every HK_BWD_OPT_* knob is described in the kernel file's
// header; with all of them 0 only v0's text below is compiled.
        const int      qt = b0::tile_qt(s, n, G, nqp);
        const int      avo = voff_nonneg(voff_lane + b0::dqa_voff(0, 0, b0::atom_gh(s, n, G), b0::atom_qt(s, n, G, nqp), nqp));
        int            kind = b0::K_FULL;
        if constexpr (MASKED) kind = b0::wave_kind(kvw0, w, qt, cshift, causal);
        const bool live = !MASKED || !BWD_SKIP_DEAD || kind != b0::K_DEAD;
        acc_t      S[2][2], dP[2][2], dq[2][2];
        u32x4      pu[2][2], du[2][2];
        auto       sm = [&](auto hh, auto kh) { // softmax / dS of block (hh, kh), design 6.2
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
        constexpr std::integral_constant<int, 0> I0{};
        constexpr std::integral_constant<int, 1> I1{};
#if HK_BWD_OPT_ONEPHASE
        // group 0 of ring op s + 3, built here so that its SALU runs under P1's WMMAs (the TDM follows the wait)
        const u32x4 g0n = ring_g0(s + b0::LEAD);
        asm volatile("" ::"s"(g0n));
#endif
#if HK_BWD_OPT_ADDR
        // the E-operand bases of P4 (slot s % 4), computed and pinned before P1 (their VALU runs ahead of the P1
        // WMMAs, which wait for the readback anyway); ONEPHASE: also the P9 readback bases (slot (s + 1) % 4)
        const lds_u8 *e_dtr = pin_lds(tr_base<XP>(lds0 + b0::lds_do(s & 3), l));
        const lds_u8 *e_qtr = pin_lds(tr_base<XP>(lds0 + b0::lds_q(s & 3), l));
#if HK_BWD_OPT_ONEPHASE
        const auto rbb = rb_bases((s + 1) & 3);
#endif
        sched_barrier<0>();
#endif
        if (live) {
            // P1: G0
            sdp(I0);
            sched_barrier<0>();
            // P2: G1 || softmax kh 0
            sdp(I1);
            sm(I0, I0);
            sm(I1, I0);
#if BWD_HOLD >= 2
            static_for<2>([&](auto hh) { static_for<4>([&](auto dt) { hold_vgprs(X.qA[hh][dt], du[1][0]); hold_vgprs(X.dA[hh][dt], du[1][0]); }); });
#endif
            interleave16<0, 4>();
            sched_barrier<0>();
#if HK_BWD_OPT_REGS & 1
            if constexpr ((P == 0 || (HK_BWD_OPT_REGS & 4)) && (!MASKED || (HK_BWD_OPT_REGS & 8))) {
                HK_BWD_KEEP8(X.qA);
                HK_BWD_KEEP8(X.dA);
            }
#endif
            // P3: dQ(s - 1) || softmax kh 1
            dq_wmma(dq);
            sm(I0, I1);
            sm(I1, I1);
#if BWD_HOLD >= 1
            static_for<2>([&](auto qh) { static_for<4>([&](auto c) { hold_vgprs(dSa[qh][c], du[1][1]); }); });
#endif
            interleave16<2, 4>();
        } else {
            dq_wmma(dq);
            static_for<2>([&](auto hh) { static_for<2>([&](auto kh) { du[hh][kh] = zero_u32x4(); }); });
        }
        sched_barrier<0>();
#if HK_BWD_OPT_REGS & 2
        if constexpr ((P == 0 || (HK_BWD_OPT_REGS & 4)) && (!MASKED || (HK_BWD_OPT_REGS & 8))) HK_BWD_KEEP8(dSa);
#endif
        // P4: dS^T stores, fence, E operands (FULL), atomics of dQ(s - 1)
        {
            lds_u8 *dr = rows_base<SP>(lds0 + b0::lds_ds(P) + b0::ds_rows_wave(w), l);
            static_for<2>([&](auto kh) {
                static_for<2>([&](auto hh) { ds_store_b128(dr, b0::ct_imm(SP, 16 * kh, 16 * hh), du[hh][kh]); });
            });
        }
        order_fence();
        opnd_t        dOT[8], QT[8];
#if HK_BWD_OPT_ADDR
        const lds_u8 *dtr = e_dtr, *qtr = e_qtr;
#else
        const lds_u8 *dtr = tr_base<XP>(lds0 + b0::lds_do(s & 3), l), *qtr = tr_base<XP>(lds0 + b0::lds_q(s & 3), l);
#endif
        auto          load_e = [&] {
            static_for<8>([&](auto dt) { dOT[dt] = load_tr<XP>(dtr, 0, 16 * dt); });
            static_for<8>([&](auto dt) { QT[dt] = load_tr<XP>(qtr, 0, 16 * dt); });
        };
#if HK_BWD_OPT_ANY
        // ---- variant tail (built only when some HK_BWD_OPT_* knob is set; the #else branch is v0's tail verbatim) ----
        auto e_wmma = [&] { // P6: E
            const opnd_t Pb[2]  = {cat_u32(pu[0][0], pu[1][0]), cat_u32(pu[0][1], pu[1][1])};
            const opnd_t dSb[2] = {cat_u32(du[0][0], du[1][0]), cat_u32(du[0][1], du[1][1])};
            static_for<8>([&](auto dt) {
                static_for<2>([&](auto kh) { dVt[dt][kh] = wmma(dOT[dt], Pb[kh], dVt[dt][kh]); });
            });
            static_for<8>([&](auto dt) {
                static_for<2>([&](auto kh) { dKt[dt][kh] = wmma(QT[dt], dSb[kh], dKt[dt][kh]); });
            });
        };
        // MASKED variant: every wave kind loads the E operands here (DEAD waves too: same slot, same window as a
        // FULL wave's), so a diagonal step has the FULL step's 32-destination burst and signal
        constexpr bool E_EARLY = !MASKED || HK_BWD_OPT_MASKED;
#if HK_BWD_OPT_ATOM == 4
        dq_atom8(dq, avo, A0, A0); // sub-burst 1 of 4: before the E tr16
        sched_barrier<0>();
#endif
        if constexpr (E_EARLY) load_e();
        sched_barrier<0>();
#if HK_BWD_OPT_ATOM == 4
        dq_atom8(dq, avo, A0, A1); // sub-burst 2 of 4: after the E tr16
#elif HK_BWD_OPT_ATOM == 2
        dq_atom8(dq, avo, A0, A0); // sub-burst 1 of 2 (qh 0)
        dq_atom8(dq, avo, A0, A1);
#else
        dq_atomics(dq, avo);
#endif
#if HK_BWD_OPT_ONEPHASE
        // P5: signal B_s (<= 7 E tr16 uncovered), P7: wait B_s right away, P8, P9, then P6 E
        if constexpr (E_EARLY) split_signal_tdm_dscnt<b0::TCNT_STEADY, b0opt::ONEPHASE_UNCOVERED>();
        else split_signal_tdm_lds<b0::TCNT_STEADY>();
        split_wait_lds();
#else
        // P5: signal B_s
        if constexpr (E_EARLY) split_signal_tdm_dscnt<b0::TCNT_STEADY, b0::FULL_UNCOVERED>();
        else split_signal_tdm_lds<b0::TCNT_STEADY>();
        // P6: E (+ the SALU of the next ring op's group 0, hidden under the WMMAs)
        const u32x4 g0n = ring_g0(s + b0::LEAD);
        asm volatile("" ::"s"(g0n)); // materialize here (IR sinking would move it behind the wait)
#if HK_BWD_OPT_ADDR
        const auto rbb = rb_bases((s + 1) & 3); // P9's bases, under E
#endif
        if (live) {
            if constexpr (!E_EARLY) load_e();
            e_wmma();
        }
#if HK_BWD_OPT_MASKED
        // diagonal step, every wave kind: retire the E tr16 here. A live wave's E WMMAs already waited for all of
        // them (one issue cycle); a DEAD wave never consumes them, and without this wait the waitcnt pass, which
        // merges both paths, puts WAW waits into the shared readback. No branch: an `else` path changes the
        // register allocation of the whole loop.
        if constexpr (MASKED) wait_dscnt<0>();
#endif
        // P7: wait B_s
        split_wait_lds();
#endif
        // P8: ring op s + 3
#if HK_BWD_OPT_TDM3
        tdm_load(g0n, rg1); // wave 3: its 16-B dummy op
#else
        if (w < b0::ISS_KV) tdm_load(g0n, rg1);
#endif
        sched_barrier<0>();
#if HK_BWD_OPT_ATOM == 4
        dq_atom8(dq, avo, A1, A0); // sub-burst 3 of 4: after the TDM
        sched_barrier<0>();
#elif HK_BWD_OPT_ATOM == 2 && !HK_BWD_OPT_ONEPHASE && !HK_BWD_OPT_ATOM_LATE
        dq_atom8(dq, avo, A1, A0); // sub-burst 2 of 2 (qh 1): after the TDM
        dq_atom8(dq, avo, A1, A1);
        sched_barrier<0>();
#endif
        // P9: readback of tile s + 1, then dSa = dS(s)
#if HK_BWD_OPT_ADDR
        readback_at(Xn, rbb);
#else
        readback(Xn, (s + 1) & 3);
#endif
        sched_barrier<0>();
        load_dsa(P);
        sched_barrier<0>();
#if HK_BWD_OPT_ATOM == 4
        dq_atom8(dq, avo, A1, A1); // sub-burst 4 of 4: after the readback
        sched_barrier<0>();
#elif HK_BWD_OPT_ATOM == 2 && (HK_BWD_OPT_ONEPHASE || HK_BWD_OPT_ATOM_LATE)
        dq_atom8(dq, avo, A1, A0); // sub-burst 2 of 2 (qh 1): after the readback (ONEPHASE: under E)
        dq_atom8(dq, avo, A1, A1);
        sched_barrier<0>();
#endif
#if HK_BWD_OPT_ONEPHASE
        if (live) { // P6: E, its operands loaded in P4; the readback lands under it
            if constexpr (!E_EARLY) load_e();
            e_wmma();
        }
#if HK_BWD_OPT_MASKED
        if constexpr (MASKED) wait_dscnt<0>(); // as above (DEAD waves: the unused E tr16 and the readback)
#endif
        sched_barrier<0>();
#endif
#else
        if constexpr (!MASKED) load_e();
        sched_barrier<0>();
        dq_atomics(dq, avo);
        // P5: signal B_s
        if constexpr (!MASKED) split_signal_tdm_dscnt<b0::TCNT_STEADY, b0::FULL_UNCOVERED>();
        else split_signal_tdm_lds<b0::TCNT_STEADY>();
        // P6: E (+ the SALU of the next ring op's group 0, hidden under the WMMAs)
        const u32x4 g0n = ring_g0(s + b0::LEAD);
        asm volatile("" ::"s"(g0n)); // materialize here (IR sinking would move it behind the wait)
        if (live) {
            if constexpr (MASKED) load_e();
            const opnd_t Pb[2]  = {cat_u32(pu[0][0], pu[1][0]), cat_u32(pu[0][1], pu[1][1])};
            const opnd_t dSb[2] = {cat_u32(du[0][0], du[1][0]), cat_u32(du[0][1], du[1][1])};
            static_for<8>([&](auto dt) {
                static_for<2>([&](auto kh) { dVt[dt][kh] = wmma(dOT[dt], Pb[kh], dVt[dt][kh]); });
            });
            static_for<8>([&](auto dt) {
                static_for<2>([&](auto kh) { dKt[dt][kh] = wmma(QT[dt], dSb[kh], dKt[dt][kh]); });
            });
        }
        // P7: wait B_s
        split_wait_lds();
        // P8: ring op s + 3
        if (w < b0::ISS_KV) tdm_load(g0n, rg1);
        sched_barrier<0>();
        // P9: readback of tile s + 1, then dSa = dS(s)
        readback(Xn, (s + 1) & 3);
        sched_barrier<0>();
        load_dsa(P);
        sched_barrier<0>();
#endif
