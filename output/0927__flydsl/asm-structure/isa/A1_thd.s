	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	kn_fmha_fwd_prefill_a16w16_m32x8_thd_0
	.p2align	8
	.type	kn_fmha_fwd_prefill_a16w16_m32x8_thd_0,@function
kn_fmha_fwd_prefill_a16w16_m32x8_thd_0:
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	v_nop
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b96 s[24:26], s[0:1], 0xb8 nv
	s_load_b96 s[52:54], s[0:1], 0x9c nv
	s_bfe_u32 s2, ttmp6, 0x40010
	s_and_b32 s3, ttmp7, 0xffff
	s_add_co_i32 s2, s2, 1
	s_bfe_u32 s5, ttmp6, 0x4000c
	s_mul_i32 s2, s3, s2
	s_bfe_u32 s4, ttmp6, 0x40004
	s_add_co_i32 s5, s5, 1
	s_bfe_u32 s6, ttmp6, 0x40014
	s_add_co_i32 s4, s4, s2
	s_and_b32 s2, ttmp6, 15
	s_mul_i32 s5, ttmp9, s5
	s_lshr_b32 s7, ttmp7, 16
	s_add_co_i32 s6, s6, 1
	s_add_co_i32 s2, s2, s5
	s_mul_i32 s5, s7, s6
	s_bfe_u32 s6, ttmp6, 0x40008
	s_getreg_b32 s8, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s6, s6, s5
	s_cmp_eq_u32 s8, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s16, s53
	s_cselect_b32 s5, s7, s6
	s_mul_i32 s6, s26, s25
	s_cselect_b32 s2, ttmp9, s2
	s_cselect_b32 s3, s3, s4
	s_abs_i32 s4, s6
	s_mul_i32 s5, s25, s5
	s_cvt_f32_u32 s7, s4
	s_add_co_i32 s3, s5, s3
	s_sub_co_i32 s5, 0, s4
	s_mul_i32 s3, s3, s24
	v_s_rcp_f32 s7, s7
	s_add_co_i32 s3, s3, s2
	s_mul_f32 s7, s7, 0x4f7ffffe
	s_cvt_u32_f32 s7, s7
	s_mul_i32 s5, s5, s7
	s_mul_hi_u32 s2, s7, s5
	s_abs_i32 s5, s3
	s_add_co_i32 s7, s7, s2
	s_mul_hi_u32 s2, s5, s7
	s_xor_b32 s7, s3, s6
	s_mul_i32 s8, s2, s4
	s_ashr_i32 s9, s7, 31
	s_sub_co_i32 s5, s5, s8
	s_add_co_i32 s8, s2, 1
	s_sub_co_i32 s10, s5, s4
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s2, s8, s2
	s_cselect_b32 s5, s10, s5
	s_add_co_i32 s8, s2, 1
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s2, s8, s2
	s_xor_b32 s2, s2, s9
	s_sub_co_i32 s4, s2, s9
	s_mul_i32 s4, s4, s6
	s_cmp_lg_u32 s3, s4
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s5, -1, 0
	s_and_b32 s4, s4, s5
	s_sub_co_ci_u32 s26, s2, s9
	s_abs_i32 s27, s25
	s_mul_i32 s5, s26, s6
	s_cvt_f32_u32 s2, s27
	s_sub_co_i32 s4, 0, s27
	s_sub_co_i32 s6, s3, s5
	s_load_b64 s[8:9], s[0:1], 0x70 nv
	v_s_rcp_f32 s2, s2
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_cvt_u32_f32 s2, s2
	s_mul_i32 s4, s4, s2
	s_mul_hi_u32 s3, s2, s4
	s_abs_i32 s4, s6
	s_add_co_i32 s2, s2, s3
	s_xor_b32 s3, s6, s25
	s_mul_hi_u32 s2, s4, s2
	s_ashr_i32 s10, s3, 31
	s_mul_i32 s5, s2, s27
	s_sub_co_i32 s7, s4, s5
	s_add_co_i32 s4, s2, 1
	s_sub_co_i32 s28, s7, s27
	s_cmp_ge_u32 s7, s27
	s_cselect_b32 s2, s4, s2
	s_cselect_b32 s4, s28, s7
	s_add_co_i32 s5, s2, 1
	s_cmp_ge_u32 s4, s27
	s_cselect_b32 s2, s5, s2
	s_load_b64 s[4:5], s[0:1], 0x60 nv
	s_xor_b32 s11, s2, s10
	s_sub_co_i32 s2, s11, s10
	s_mul_i32 s2, s2, s25
	s_cmp_lg_u32 s6, s2
	s_cselect_b32 s12, -1, 0
	s_cmp_lt_i32 s3, 0
	s_cselect_b32 s3, -1, 0
	s_and_b32 s3, s12, s3
	s_sub_co_ci_u32 s10, s11, s10
	s_ashr_i32 s17, s53, 31
	s_ashr_i32 s11, s10, 31
	s_lshl_b64 s[10:11], s[10:11], 2
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[4:5], s[10:11]
	s_load_b64 s[56:57], s[4:5], 0x0
	s_wait_xcnt 0x0
	s_add_nc_u64 s[4:5], s[8:9], s[10:11]
	s_load_b64 s[64:65], s[4:5], 0x0
	s_clause 0x2
	s_load_b64 s[60:61], s[0:1], 0x0 nv
	s_load_b64 s[34:35], s[0:1], 0x40 nv
	s_load_b256 s[8:15], s[0:1], 0x7c nv
	s_wait_kmcnt 0x0
	s_sub_co_i32 s94, s65, s64
	s_ashr_i32 s5, s57, 31
	s_mov_b32 s4, s57
	s_sub_co_i32 s33, s57, s56
	s_mul_u64 s[16:17], s[16:17], s[4:5]
	s_lshl_b64 s[58:59], s[16:17], 2
	s_cmp_gt_i32 s33, 0
	s_cselect_b32 s3, -1, 0
	s_min_i32 s16, s33, s94
	s_cmp_lt_i32 s16, 1
	s_mov_b32 s16, -1
	s_cbranch_scc0 .LBB0_4
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s3, 1, 0
	s_cmp_lg_u32 s3, 1
	s_cbranch_scc1 .LBB0_3
	v_and_b32_e32 v1, 31, v0
	s_bfe_u32 s3, ttmp8, 0x50019
	s_sub_co_i32 s17, s6, s2
	s_lshl_b32 s22, s3, 5
	s_not_b32 s2, s26
	v_or_b32_e32 v6, s22, v1
	s_add_co_i32 s2, s24, s2
	s_ashr_i32 s19, s12, 31
	s_lshl_b32 s25, s2, 7
	s_mov_b32 s18, s12
	v_lshrrev_b32_e32 v3, 4, v6
	v_bitop3_b32 v2, v1, 0x3f0, s22 bitop3:0xc8
	v_add_nc_u32_e32 v7, 0x80, v6
	s_mul_u64 s[20:21], s[18:19], s[4:5]
	v_bitop3_b32 v1, v1, 15, s22 bitop3:0xc8
	s_lshl_b32 s5, s17, 2
	v_cmp_ne_u32_e32 vcc_lo, v6, v2
	v_or_b32_e32 v2, s25, v3
	v_ashrrev_i32_e32 v3, 31, v7
	v_cmp_gt_i32_e64 s2, 0, v6
	v_add_nc_u32_e32 v8, 0x100, v6
	v_add_nc_u32_e32 v16, 0x180, v6
	s_mov_b32 s16, 0
	s_lshl_b32 s17, s20, 26
	s_and_b32 vcc_lo, s2, vcc_lo
	s_lshr_b64 s[2:3], s[20:21], 6
	v_subrev_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_dual_lshrrev_b32 v3, 28, v3 :: v_dual_ashrrev_i32 v5, 31, v8
	s_and_b64 s[22:23], s[2:3], 0x1ffffffffffffff
	v_ashrrev_i32_e32 v4, 31, v2
	v_cmp_gt_i32_e32 vcc_lo, 0, v7
	v_add_nc_u32_e32 v3, v7, v3
	v_cmp_gt_i32_e64 s2, 0, v2
	s_mov_b32 s18, s16
	v_dual_lshrrev_b32 v4, 30, v4 :: v_dual_lshrrev_b32 v5, 28, v5
	v_and_b32_e32 v9, -16, v3
	s_or_b64 s[20:21], s[60:61], s[16:17]
	s_mov_b32 s17, s16
	v_dual_add_nc_u32 v4, v2, v4 :: v_dual_ashrrev_i32 v3, 4, v3
	v_add_nc_u32_e32 v5, v8, v5
	v_cmp_ne_u32_e64 s3, v7, v9
	s_mov_b32 s19, s16
	v_dual_add_nc_u32 v3, s25, v3 :: v_dual_bitop2_b32 v10, -4, v4 bitop3:0x40
	v_ashrrev_i32_e32 v4, 2, v4
	s_and_b32 vcc_lo, vcc_lo, s3
	v_ashrrev_i32_e32 v17, 31, v16
	v_cmp_ne_u32_e64 s4, v2, v10
	v_sub_nc_u32_e32 v11, v2, v10
	v_subrev_co_ci_u32_e64 v10, null, 0, v3, vcc_lo
	v_sub_nc_u32_e32 v7, v7, v9
	s_and_b32 vcc_lo, s2, s4
	v_cmp_gt_i32_e64 s4, 0, v16
	v_subrev_co_ci_u32_e64 v3, null, 0, v4, vcc_lo
	v_dual_ashrrev_i32 v4, 31, v10 :: v_dual_add_nc_u32 v2, s5, v11
	v_dual_add_nc_u32 v12, s56, v3 :: v_dual_bitop2_b32 v11, -16, v5 bitop3:0x40
	v_ashrrev_i32_e32 v5, 4, v5
	v_cmp_gt_i32_e32 vcc_lo, 0, v8
	v_mul_lo_u32 v2, v2, s52
	v_cmp_ne_u32_e64 s2, v8, v11
	v_dual_lshrrev_b32 v4, 30, v4 :: v_dual_add_nc_u32 v5, s25, v5
	v_dual_lshlrev_b32 v1, 4, v1 :: v_dual_lshlrev_b32 v7, 4, v7
	s_and_b32 vcc_lo, vcc_lo, s2
	v_cmp_gt_i32_e64 s2, 0, v10
	v_mad_u32 v2, v12, s12, v2
	v_add_nc_u32_e32 v12, v10, v4
	v_subrev_co_ci_u32_e64 v13, null, 0, v5, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s33, v3
	v_mov_b64_e32 v[4:5], s[18:19]
	v_dual_ashrrev_i32 v15, 31, v13 :: v_dual_bitop2_b32 v14, -4, v12 bitop3:0x40
	v_lshl_add_u32 v1, v2, 1, v1
	v_ashrrev_i32_e32 v12, 2, v12
	v_mov_b64_e32 v[2:3], s[16:17]
	s_lshl_b32 s17, s58, 25
	v_lshrrev_b32_e32 v15, 30, v15
	v_cndmask_b32_e32 v1, 0x7fffffff, v1, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, v10, v14
	v_sub_nc_u32_e32 v10, v10, v14
	s_lshr_b64 s[18:19], s[58:59], 7
	v_dual_add_nc_u32 v14, v13, v15 :: v_dual_lshrrev_b32 v15, 28, v17
	s_and_b32 vcc_lo, s2, vcc_lo
	v_add_nc_u32_e32 v10, s5, v10
	v_subrev_co_ci_u32_e64 v12, null, 0, v12, vcc_lo
	v_dual_add_nc_u32 v9, v16, v15 :: v_dual_bitop2_b32 v17, -4, v14 bitop3:0x40
	v_cmp_gt_i32_e64 s2, 0, v13
	v_dual_ashrrev_i32 v14, 2, v14 :: v_dual_add_nc_u32 v15, s56, v12
	v_cmp_ne_u32_e32 vcc_lo, v13, v17
	v_sub_nc_u32_e32 v18, v13, v17
	v_and_b32_e32 v19, -16, v9
	v_mul_lo_u32 v10, v10, s52
	v_add_nc_u32_e32 v17, 0x200, v6
	s_and_b32 vcc_lo, s2, vcc_lo
	v_add_nc_u32_e32 v13, s5, v18
	v_subrev_co_ci_u32_e64 v14, null, 0, v14, vcc_lo
	v_ashrrev_i32_e32 v9, 4, v9
	v_cmp_ne_u32_e64 s3, v16, v19
	v_mul_lo_u32 v13, v13, s52
	v_dual_add_nc_u32 v18, s56, v14 :: v_dual_sub_nc_u32 v8, v8, v11
	v_add_nc_u32_e32 v9, s25, v9
	s_and_b32 vcc_lo, s4, s3
	v_mad_u32 v10, v15, s12, v10
	v_ashrrev_i32_e32 v15, 31, v17
	s_wait_alu depctr_va_vdst(0)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	s_wait_xcnt 0x0
	v_subrev_co_ci_u32_e64 v9, null, 0, v9, vcc_lo
	v_mad_u32 v13, v18, s12, v13
	v_lshrrev_b32_e32 v15, 28, v15
	v_cmp_gt_i32_e32 vcc_lo, s33, v12
	v_dual_ashrrev_i32 v11, 31, v9 :: v_dual_lshlrev_b32 v8, 4, v8
	s_wait_alu depctr_vm_vsrc(0)
	v_lshl_add_u32 v1, v10, 1, v7
	v_cmp_gt_i32_e64 s2, 0, v17
	s_or_b64 s[16:17], s[34:35], s[16:17]
	v_lshrrev_b32_e32 v11, 30, v11
	v_lshl_add_u32 v8, v13, 1, v8
	v_cndmask_b32_e32 v1, 0x7fffffff, v1, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s33, v14
	v_add_nc_u32_e32 v7, v17, v15
	v_add_nc_u32_e32 v10, v9, v11
	v_add_nc_u32_e32 v13, 0x280, v6
	v_sub_nc_u32_e32 v16, v16, v19
	v_cndmask_b32_e32 v8, 0x7fffffff, v8, vcc_lo
	v_and_b32_e32 v11, -16, v7
	v_dual_ashrrev_i32 v7, 4, v7 :: v_dual_bitop2_b32 v12, -4, v10 bitop3:0x40
	v_ashrrev_i32_e32 v14, 31, v13
	s_wait_alu depctr_va_vdst(9)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	s_wait_alu depctr_va_vdst(3)
	buffer_store_b128 v[2:5], v8, s[20:23], null offen
	v_cmp_ne_u32_e32 vcc_lo, v17, v11
	v_cmp_ne_u32_e64 s3, v9, v12
	v_dual_add_nc_u32 v7, s25, v7 :: v_dual_sub_nc_u32 v12, v9, v12
	v_lshrrev_b32_e32 v14, 28, v14
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_gt_i32_e64 s2, 0, v13
	s_wait_xcnt 0x0
	v_subrev_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 0, v9
	v_dual_ashrrev_i32 v9, 2, v10 :: v_dual_add_nc_u32 v10, s5, v12
	v_dual_ashrrev_i32 v12, 31, v7 :: v_dual_add_nc_u32 v14, v13, v14
	s_and_b32 vcc_lo, vcc_lo, s3
	v_sub_nc_u32_e32 v11, v17, v11
	v_subrev_co_ci_u32_e64 v9, null, 0, v9, vcc_lo
	v_dual_lshrrev_b32 v12, 30, v12 :: v_dual_bitop2_b32 v15, -16, v14 bitop3:0x40
	v_mul_lo_u32 v10, v10, s52
	v_dual_add_nc_u32 v18, s56, v9 :: v_dual_ashrrev_i32 v14, 4, v14
	v_add_nc_u32_e32 v12, v7, v12
	v_cmp_ne_u32_e32 vcc_lo, v13, v15
	v_sub_nc_u32_e32 v13, v13, v15
	v_dual_lshlrev_b32 v11, 4, v11 :: v_dual_add_nc_u32 v14, s25, v14
	v_mad_u32 v10, v18, s12, v10
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_gt_i32_e64 s2, 0, v7
	v_subrev_co_ci_u32_e64 v14, null, 0, v14, vcc_lo
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_lshlrev_b32 v16, 4, v16 :: v_dual_ashrrev_i32 v8, 31, v14
	v_lshl_add_u32 v1, v10, 1, v16
	v_add_nc_u32_e32 v10, 0x300, v6
	v_dual_ashrrev_i32 v16, 31, v10 :: v_dual_bitop2_b32 v18, -4, v12 bitop3:0x40
	v_cmp_ne_u32_e32 vcc_lo, v7, v18
	v_dual_ashrrev_i32 v12, 2, v12 :: v_dual_sub_nc_u32 v7, v7, v18
	v_lshrrev_b32_e32 v8, 30, v8
	v_cmp_gt_i32_e64 s4, 0, v10
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_gt_i32_e64 s2, 0, v14
	v_subrev_co_ci_u32_e64 v12, null, 0, v12, vcc_lo
	v_add_nc_u32_e32 v7, s5, v7
	v_cmp_gt_i32_e32 vcc_lo, s33, v9
	v_add_nc_u32_e32 v8, v14, v8
	v_add_nc_u32_e32 v18, s56, v12
	v_mul_lo_u32 v7, v7, s52
	v_cndmask_b32_e32 v1, 0x7fffffff, v1, vcc_lo
	v_and_b32_e32 v9, -4, v8
	v_lshrrev_b32_e32 v16, 28, v16
	v_ashrrev_i32_e32 v8, 2, v8
	s_wait_alu depctr_va_vdst(3)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	v_sub_nc_u32_e32 v19, v14, v9
	v_add_nc_u32_e32 v16, v10, v16
	v_cmp_ne_u32_e32 vcc_lo, v14, v9
	v_mad_u32 v7, v18, s12, v7
	v_dual_add_nc_u32 v14, s5, v19 :: v_dual_ashrrev_i32 v9, 4, v16
	s_and_b32 vcc_lo, s2, vcc_lo
	s_wait_xcnt 0x0
	v_subrev_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	v_dual_add_nc_u32 v9, s25, v9 :: v_dual_bitop2_b32 v20, -16, v16 bitop3:0x40
	v_add_nc_u32_e32 v16, 0x380, v6
	v_mul_lo_u32 v14, v14, s52
	v_add_nc_u32_e32 v17, s56, v8
	s_wait_alu depctr_vm_vsrc(0)
	v_lshl_add_u32 v1, v7, 1, v11
	v_ashrrev_i32_e32 v19, 31, v16
	v_cmp_ne_u32_e64 s3, v10, v20
	v_cmp_gt_i32_e64 s2, 0, v16
	v_sub_nc_u32_e32 v10, v10, v20
	v_mad_u32 v14, v17, s12, v14
	v_lshrrev_b32_e32 v17, 28, v19
	s_and_b32 vcc_lo, s4, s3
	v_subrev_co_ci_u32_e64 v9, null, 0, v9, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s33, v12
	v_ashrrev_i32_e32 v18, 31, v9
	v_cmp_gt_i32_e64 s4, 0, v9
	v_cndmask_b32_e32 v1, 0x7fffffff, v1, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s33, v8
	v_dual_lshrrev_b32 v15, 30, v18 :: v_dual_lshlrev_b32 v13, 4, v13
	v_dual_add_nc_u32 v7, v9, v15 :: v_dual_add_nc_u32 v11, v16, v17
	v_lshl_add_u32 v13, v14, 1, v13
	v_and_b32_e32 v12, -4, v7
	v_and_b32_e32 v14, -16, v11
	v_dual_cndmask_b32 v8, 0x7fffffff, v13 :: v_dual_ashrrev_i32 v7, 2, v7
	s_wait_alu depctr_va_vdst(7)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	s_wait_alu depctr_va_vdst(0)
	buffer_store_b128 v[2:5], v8, s[20:23], null offen
	v_dual_sub_nc_u32 v13, v9, v12 :: v_dual_ashrrev_i32 v11, 4, v11
	v_cmp_ne_u32_e32 vcc_lo, v16, v14
	v_cmp_ne_u32_e64 s3, v9, v12
	v_or_b32_e32 v12, 0x400, v6
	v_dual_add_nc_u32 v9, s5, v13 :: v_dual_add_nc_u32 v11, s25, v11
	s_and_b32 vcc_lo, s2, vcc_lo
	s_movk_i32 s2, 0x7f0
	v_lshrrev_b32_e32 v18, 4, v12
	v_bitop3_b32 v15, v6, s2, 0x400 bitop3:0xc8
	s_wait_xcnt 0x0
	v_subrev_co_ci_u32_e64 v11, null, 0, v11, vcc_lo
	s_and_b32 vcc_lo, s4, s3
	v_mul_lo_u32 v9, v9, s52
	v_subrev_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	v_ashrrev_i32_e32 v13, 31, v11
	v_cmp_ne_u32_e32 vcc_lo, v12, v15
	v_cmp_gt_i32_e64 s2, 0, v12
	v_dual_add_nc_u32 v17, s56, v7 :: v_dual_bitop2_b32 v12, s25, v18 bitop3:0x54
	v_dual_lshrrev_b32 v13, 30, v13 :: v_dual_lshlrev_b32 v10, 4, v10
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_gt_i32_e64 s2, 0, v11
	v_mad_u32 v9, v17, s12, v9
	v_add_nc_u32_e32 v13, v11, v13
	v_subrev_co_ci_u32_e64 v12, null, 0, v12, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s33, v7
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_sub_nc_u32 v14, v16, v14 :: v_dual_bitop2_b32 v8, -4, v13 bitop3:0x40
	v_bitop3_b32 v18, v6, 15, 0x400 bitop3:0xc8
	v_lshl_add_u32 v1, v9, 1, v10
	v_ashrrev_i32_e32 v9, 31, v12
	v_add_nc_u32_e32 v10, 0x480, v6
	v_dual_cndmask_b32 v1, 0x7fffffff, v1 :: v_dual_lshlrev_b32 v14, 4, v14
	v_dual_lshrrev_b32 v7, 30, v9 :: v_dual_ashrrev_i32 v9, 31, v10
	v_cmp_ne_u32_e32 vcc_lo, v11, v8
	v_dual_ashrrev_i32 v13, 2, v13 :: v_dual_sub_nc_u32 v8, v11, v8
	v_dual_add_nc_u32 v7, v12, v7 :: v_dual_lshrrev_b32 v9, 28, v9
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_gt_i32_e64 s2, 0, v12
	v_subrev_co_ci_u32_e64 v11, null, 0, v13, vcc_lo
	v_dual_add_nc_u32 v8, s5, v8 :: v_dual_bitop2_b32 v13, -4, v7 bitop3:0x40
	v_dual_add_nc_u32 v9, v10, v9 :: v_dual_add_nc_u32 v15, s56, v11
	v_dual_ashrrev_i32 v7, 2, v7 :: v_dual_sub_nc_u32 v16, v12, v13
	v_cmp_ne_u32_e32 vcc_lo, v12, v13
	v_and_b32_e32 v17, -16, v9
	v_mul_lo_u32 v8, v8, s52
	v_cmp_gt_i32_e64 s4, 0, v10
	v_add_nc_u32_e32 v12, s5, v16
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_ne_u32_e64 s3, v10, v17
	v_subrev_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	v_ashrrev_i32_e32 v9, 4, v9
	v_mul_lo_u32 v12, v12, s52
	v_mad_u32 v8, v15, s12, v8
	v_add_nc_u32_e32 v16, s56, v7
	v_add_nc_u32_e32 v13, 0x500, v6
	v_add_nc_u32_e32 v9, s25, v9
	s_and_b32 vcc_lo, s4, s3
	s_wait_alu depctr_va_vdst(14)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	v_sub_nc_u32_e32 v10, v10, v17
	v_mad_u32 v12, v16, s12, v12
	s_wait_xcnt 0x0
	v_subrev_co_ci_u32_e64 v9, null, 0, v9, vcc_lo
	v_dual_ashrrev_i32 v15, 31, v13 :: v_dual_lshlrev_b32 v16, 4, v18
	s_wait_alu depctr_vm_vsrc(0)
	v_lshl_add_u32 v1, v8, 1, v14
	v_ashrrev_i32_e32 v19, 31, v9
	v_cmp_gt_i32_e32 vcc_lo, s33, v11
	v_lshrrev_b32_e32 v15, 28, v15
	v_lshl_add_u32 v12, v12, 1, v16
	v_cmp_gt_i32_e64 s2, 0, v13
	v_dual_cndmask_b32 v1, 0x7fffffff, v1 :: v_dual_lshrrev_b32 v18, 30, v19
	v_cmp_gt_i32_e32 vcc_lo, s33, v7
	v_add_nc_u32_e32 v8, v13, v15
	v_add_nc_u32_e32 v15, 0x580, v6
	v_dual_cndmask_b32 v7, 0x7fffffff, v12 :: v_dual_lshlrev_b32 v10, 4, v10
	v_dual_add_nc_u32 v14, v9, v18 :: v_dual_bitop2_b32 v11, -16, v8 bitop3:0x40
	v_dual_ashrrev_i32 v8, 4, v8 :: v_dual_ashrrev_i32 v16, 31, v15
	s_wait_alu depctr_va_vdst(6)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	s_wait_alu depctr_va_vdst(2)
	buffer_store_b128 v[2:5], v7, s[20:23], null offen
	v_and_b32_e32 v12, -4, v14
	v_cmp_ne_u32_e32 vcc_lo, v13, v11
	v_dual_add_nc_u32 v8, s25, v8 :: v_dual_sub_nc_u32 v11, v13, v11
	v_cmp_ne_u32_e64 s3, v9, v12
	s_and_b32 vcc_lo, s2, vcc_lo
	v_sub_nc_u32_e32 v12, v9, v12
	s_wait_xcnt 0x0
	v_subrev_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 0, v9
	v_dual_ashrrev_i32 v9, 2, v14 :: v_dual_add_nc_u32 v12, s5, v12
	v_cmp_gt_i32_e64 s2, 0, v15
	v_lshlrev_b32_e32 v11, 4, v11
	s_and_b32 vcc_lo, vcc_lo, s3
	v_subrev_co_ci_u32_e64 v9, null, 0, v9, vcc_lo
	v_ashrrev_i32_e32 v14, 31, v8
	v_mul_lo_u32 v12, v12, s52
	v_dual_add_nc_u32 v17, s56, v9 :: v_dual_lshrrev_b32 v16, 28, v16
	v_dual_lshrrev_b32 v14, 30, v14 :: v_dual_add_nc_u32 v16, v15, v16
	v_add_nc_u32_e32 v14, v8, v14
	v_mad_u32 v12, v17, s12, v12
	v_and_b32_e32 v18, -16, v16
	v_dual_ashrrev_i32 v16, 4, v16 :: v_dual_bitop2_b32 v17, -4, v14 bitop3:0x40
	v_cmp_ne_u32_e32 vcc_lo, v15, v18
	v_add_nc_u32_e32 v16, s25, v16
	s_wait_alu depctr_vm_vsrc(1)
	v_lshl_add_u32 v1, v12, 1, v10
	v_ashrrev_i32_e32 v12, 2, v14
	v_add_nc_u32_e32 v10, 0x600, v6
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_gt_i32_e64 s2, 0, v8
	v_subrev_co_ci_u32_e64 v16, null, 0, v16, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, v8, v17
	v_dual_sub_nc_u32 v8, v8, v17 :: v_dual_ashrrev_i32 v14, 31, v10
	s_wait_alu depctr_vm_vsrc(0)
	v_ashrrev_i32_e32 v7, 31, v16
	v_cmp_gt_i32_e64 s4, 0, v10
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_gt_i32_e64 s2, 0, v16
	v_subrev_co_ci_u32_e64 v12, null, 0, v12, vcc_lo
	v_dual_lshrrev_b32 v7, 30, v7 :: v_dual_lshrrev_b32 v14, 28, v14
	v_cmp_gt_i32_e32 vcc_lo, s33, v9
	v_dual_sub_nc_u32 v15, v15, v18 :: v_dual_add_nc_u32 v7, v16, v7
	v_add_nc_u32_e32 v8, s5, v8
	v_dual_cndmask_b32 v1, 0x7fffffff, v1 :: v_dual_add_nc_u32 v14, v10, v14
	v_dual_add_nc_u32 v17, s56, v12 :: v_dual_bitop2_b32 v9, -4, v7 bitop3:0x40
	v_ashrrev_i32_e32 v7, 2, v7
	v_mul_lo_u32 v8, v8, s52
	v_and_b32_e32 v20, -16, v14
	s_wait_alu depctr_va_vdst(4)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	v_sub_nc_u32_e32 v19, v16, v9
	v_cmp_ne_u32_e32 vcc_lo, v16, v9
	v_ashrrev_i32_e32 v9, 4, v14
	v_cmp_ne_u32_e64 s3, v10, v20
	v_add_nc_u32_e32 v16, 0x680, v6
	v_add_nc_u32_e32 v14, s5, v19
	s_and_b32 vcc_lo, s2, vcc_lo
	v_add_nc_u32_e32 v9, s25, v9
	s_wait_xcnt 0x0
	v_subrev_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	s_and_b32 vcc_lo, s4, s3
	v_mul_lo_u32 v14, v14, s52
	v_subrev_co_ci_u32_e64 v9, null, 0, v9, vcc_lo
	v_mad_u32 v8, v17, s12, v8
	v_add_nc_u32_e32 v13, s56, v7
	v_cmp_gt_i32_e32 vcc_lo, s33, v12
	v_ashrrev_i32_e32 v17, 31, v9
	v_cmp_gt_i32_e64 s3, 0, v16
	v_cmp_gt_i32_e64 s4, 0, v9
	v_mad_u32 v13, v13, s12, v14
	v_dual_ashrrev_i32 v14, 31, v16 :: v_dual_lshrrev_b32 v17, 30, v17
	v_lshlrev_b32_e32 v15, 4, v15
	s_wait_alu depctr_vm_vsrc(0)
	v_lshl_add_u32 v1, v8, 1, v11
	v_dual_lshrrev_b32 v8, 28, v14 :: v_dual_add_nc_u32 v11, v9, v17
	v_lshl_add_u32 v13, v13, 1, v15
	v_add_nc_u32_e32 v14, 0x700, v6
	v_dual_add_nc_u32 v8, v16, v8 :: v_dual_bitop2_b32 v12, -4, v11 bitop3:0x40
	v_cndmask_b32_e32 v1, 0x7fffffff, v1, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s33, v7
	v_ashrrev_i32_e32 v11, 2, v11
	v_cndmask_b32_e32 v7, 0x7fffffff, v13, vcc_lo
	v_and_b32_e32 v13, -16, v8
	v_ashrrev_i32_e32 v8, 4, v8
	v_cmp_ne_u32_e32 vcc_lo, v9, v12
	v_dual_sub_nc_u32 v12, v9, v12 :: v_dual_ashrrev_i32 v9, 31, v14
	v_cmp_ne_u32_e64 s2, v16, v13
	v_add_nc_u32_e32 v8, s25, v8
	s_and_b32 vcc_lo, s4, vcc_lo
	v_dual_add_nc_u32 v12, s5, v12 :: v_dual_lshrrev_b32 v9, 28, v9
	s_and_b32 s2, s3, s2
	v_subrev_co_ci_u32_e64 v11, null, 0, v11, vcc_lo
	v_subrev_co_ci_u32_e64 v8, null, 0, v8, s2
	v_mul_lo_u32 v12, v12, s52
	s_wait_alu depctr_va_vdst(13)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_add_nc_u32 v1, v14, v9 :: v_dual_ashrrev_i32 v15, 31, v8
	v_add_nc_u32_e32 v9, s56, v11
	v_cmp_gt_i32_e64 s2, 0, v14
	v_lshrrev_b32_e32 v15, 30, v15
	v_mad_u32 v9, v9, s12, v12
	v_add_nc_u32_e32 v12, v8, v15
	s_wait_alu depctr_va_vdst(14)
	buffer_store_b128 v[2:5], v7, s[20:23], null offen
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_sub_nc_u32 v7, v10, v20 :: v_dual_bitop2_b32 v10, -16, v1 bitop3:0x40
	v_ashrrev_i32_e32 v1, 4, v1
	v_add_nc_u32_e32 v15, 0x780, v6
	v_lshlrev_b32_e32 v7, 4, v7
	v_cmp_ne_u32_e32 vcc_lo, v14, v10
	v_dual_add_nc_u32 v1, s25, v1 :: v_dual_ashrrev_i32 v17, 31, v15
	v_and_b32_e32 v18, -4, v12
	v_lshl_add_u32 v7, v9, 1, v7
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_gt_i32_e64 s2, 0, v8
	v_subrev_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	v_lshrrev_b32_e32 v17, 28, v17
	v_cmp_ne_u32_e32 vcc_lo, v8, v18
	v_dual_sub_nc_u32 v9, v8, v18 :: v_dual_ashrrev_i32 v18, 31, v1
	v_dual_ashrrev_i32 v8, 2, v12 :: v_dual_add_nc_u32 v17, v15, v17
	s_and_b32 vcc_lo, s2, vcc_lo
	v_dual_add_nc_u32 v9, s5, v9 :: v_dual_lshrrev_b32 v12, 30, v18
	v_subrev_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	v_and_b32_e32 v18, -16, v17
	v_mul_lo_u32 v9, v9, s52
	v_dual_add_nc_u32 v12, v1, v12 :: v_dual_ashrrev_i32 v17, 4, v17
	v_cmp_gt_i32_e64 s2, 0, v15
	v_cmp_ne_u32_e32 vcc_lo, v15, v18
	v_dual_sub_nc_u32 v13, v16, v13 :: v_dual_bitop2_b32 v19, -4, v12 bitop3:0x40
	v_dual_add_nc_u32 v16, s25, v17 :: v_dual_add_nc_u32 v17, s56, v8
	s_and_b32 vcc_lo, s2, vcc_lo
	v_add_nc_u32_e32 v6, s25, v6
	v_cmp_gt_i32_e64 s2, 0, v1
	v_sub_nc_u32_e32 v10, v14, v10
	v_mad_u32 v9, v17, s12, v9
	v_sub_nc_u32_e32 v17, v1, v19
	v_subrev_co_ci_u32_e64 v16, null, 0, v16, vcc_lo
	v_lshlrev_b32_e32 v13, 4, v13
	v_cmp_ne_u32_e32 vcc_lo, v1, v19
	v_ashrrev_i32_e32 v1, 2, v12
	v_dual_ashrrev_i32 v19, 31, v16 :: v_dual_add_nc_u32 v12, s5, v17
	v_lshl_add_u32 v9, v9, 1, v13
	s_and_b32 vcc_lo, s2, vcc_lo
	v_cmp_gt_i32_e64 s2, 0, v16
	v_dual_lshrrev_b32 v17, 30, v19 :: v_dual_ashrrev_i32 v19, 31, v6
	v_subrev_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	v_mul_lo_u32 v12, v12, s52
	v_dual_add_nc_u32 v17, v16, v17 :: v_dual_lshrrev_b32 v19, 30, v19
	v_cmp_gt_i32_e32 vcc_lo, s33, v11
	v_add_nc_u32_e32 v11, s56, v1
	v_cmp_gt_i32_e64 s4, 0, v6
	v_dual_add_nc_u32 v13, v6, v19 :: v_dual_bitop2_b32 v20, -4, v17 bitop3:0x40
	v_cndmask_b32_e32 v7, 0x7fffffff, v7, vcc_lo
	v_mad_u32 v11, v11, s12, v12
	v_dual_ashrrev_i32 v14, 2, v17 :: v_dual_sub_nc_u32 v12, v16, v20
	v_and_b32_e32 v17, -4, v13
	v_cmp_ne_u32_e32 vcc_lo, v16, v20
	v_dual_ashrrev_i32 v13, 2, v13 :: v_dual_lshlrev_b32 v10, 4, v10
	v_add_nc_u32_e32 v12, s5, v12
	v_cmp_ne_u32_e64 s3, v6, v17
	s_and_b32 vcc_lo, s2, vcc_lo
	s_wait_alu depctr_va_vdst(7)
	buffer_store_b128 v[2:5], v7, s[20:23], null offen
	s_wait_xcnt 0x0
	v_subrev_co_ci_u32_e64 v14, null, 0, v14, vcc_lo
	v_mul_lo_u32 v12, v12, s52
	s_and_b32 vcc_lo, s4, s3
	v_dual_sub_nc_u32 v6, v6, v17 :: v_dual_add_nc_u32 v16, s56, v14
	v_subrev_co_ci_u32_e64 v13, null, 0, v13, vcc_lo
	v_sub_nc_u32_e32 v15, v15, v18
	v_cmp_gt_i32_e32 vcc_lo, s33, v8
	v_lshl_add_u32 v10, v11, 1, v10
	v_dual_add_nc_u32 v17, s56, v13 :: v_dual_add_nc_u32 v6, s5, v6
	v_mad_u32 v11, v16, s12, v12
	v_lshlrev_b32_e32 v12, 4, v15
	v_cndmask_b32_e32 v8, 0x7fffffff, v9, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s33, v1
	v_mul_lo_u32 v15, v17, s53
	v_mul_lo_u32 v6, v6, s54
	v_cndmask_b32_e32 v1, 0x7fffffff, v10, vcc_lo
	s_wait_alu depctr_vm_vsrc(0)
	v_lshl_add_u32 v7, v11, 1, v12
	v_cmp_gt_i32_e32 vcc_lo, s33, v14
	s_wait_alu depctr_va_vdst(6)
	buffer_store_b128 v[2:5], v8, s[20:23], null offen
	v_add_lshl_u32 v6, v15, v6, 2
	s_wait_alu depctr_va_vdst(3)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_cndmask_b32_e32 v1, 0x7fffffff, v7, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, s33, v13
	v_mov_b32_e32 v7, 0xff800000
	v_cndmask_b32_e32 v6, 0x7fffffff, v6, vcc_lo
	s_wait_alu depctr_va_vdst(3)
	buffer_store_b128 v[2:5], v1, s[20:23], null offen
	s_wait_alu depctr_va_vdst(0)
	buffer_store_b32 v7, v6, s[16:19], null offen
.LBB0_3:
	s_wait_xcnt 0x0
	s_mov_b32 s16, 0
.LBB0_4:
	s_and_b32 s2, s16, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	s_cbranch_scc1 .LBB0_51
	s_bfe_u32 s96, ttmp8, 0x50019
	s_set_vgpr_msb 64
	v_and_b32_e32 v141 /*v397*/, 15, v0
	s_and_b32 s2, s96, 30
	s_lshr_b32 s21, s96, 1
	s_cmp_lg_u32 s96, s2
	s_mul_i32 s93, s96, 0x2200
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s96, 0
	v_and_b32_e32 v143 /*v399*/, 16, v0
	s_cselect_b32 s3, -1, 0
	s_mov_b32 s16, s9
	s_and_b32 s2, s3, s2
	s_mov_b32 s18, s13
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s22, 1, 0
	s_ashr_i32 s2, s6, 31
	s_cmp_ge_u32 s7, s27
	s_mov_b32 s68, s10
	s_cselect_b32 s3, s28, s7
	s_clause 0x3
	s_load_b64 s[6:7], s[0:1], 0x10 nv
	s_load_b64 s[72:73], s[0:1], 0x20 nv
	s_load_b64 s[74:75], s[0:1], 0x30 nv
	s_load_b32 s23, s[0:1], 0xac nv
	s_wait_xcnt 0x0
	s_sub_co_i32 s0, s3, s27
	s_cmp_ge_u32 s3, s27
	s_mov_b32 s70, s11
	s_cselect_b32 s0, s0, s3
	s_not_b32 s1, s26
	s_xor_b32 s0, s0, s2
	s_add_co_i32 s1, s24, s1
	s_lshl_b32 s3, s96, 5
	s_lshl_b32 s95, s1, 7
	s_sub_co_i32 s2, s0, s2
	s_add_co_i32 s57, s95, s3
	s_lshl_b32 s62, s2, 2
	s_cmp_lt_i32 s57, 0
	v_and_b32_e32 v142 /*v398*/, 31, v0
	s_cselect_b32 s55, -1, 0
	s_add_co_i32 s5, s93, 0x20000
	s_set_vgpr_msb 0x4044
	v_or_b32_e32 v136 /*v392*/, s57, v141 /*v397*/
	s_ashr_i32 s0, s57, 31
	s_ashr_i32 s3, s57, 2
	s_lshr_b32 s0, s0, 30
	s_add_co_i32 s24, s3, s56
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4404
	v_ashrrev_i32_e32 v1, 31, v136 /*v392*/
	s_ashr_i32 s17, s9, 31
	s_ashr_i32 s63, s62, 31
	s_ashr_i32 s19, s13, 31
	s_ashr_i32 s25, s24, 31
	s_set_vgpr_msb 0x401
	v_lshrrev_b32_e32 v1, 30, v1
	s_mul_u64 s[24:25], s[24:25], s[16:17]
	s_mul_u64 s[26:27], s[62:63], s[18:19]
	s_lshl_b64 s[24:25], s[24:25], 1
	s_lshl_b64 s[26:27], s[26:27], 1
	v_add_nc_u32_e32 v1, v136 /*v392*/, v1
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[6:7], s[6:7], s[24:25]
	s_set_vgpr_msb 0x140
	v_bfe_u32 v140 /*v396*/, v0, 4, 1
	s_add_nc_u64 s[6:7], s[6:7], s[26:27]
	s_mov_b32 s26, s15
	v_ashrrev_i32_e32 v137 /*v393*/, 2, v1
	s_set_vgpr_msb 0x4054
	v_mad_u32_u24 v144 /*v400*/, 0x110, v141 /*v397*/, v143 /*v399*/
	v_cvt_pk_bf16_f32 v147 /*v403*/, s8, s8
	v_dual_lshlrev_b32 v145 /*v401*/, 3, v140 /*v396*/ :: v_dual_lshlrev_b32 v148 /*v404*/, 1, v142 /*v398*/
	s_mov_b32 s4, 1
	v_add_nc_u32_e32 v149 /*v405*/, s5, v144 /*v400*/
	s_set_vgpr_msb 0x5404
	v_or_b32_e32 v2, 16, v136 /*v392*/
	s_set_vgpr_msb 0x444
	v_or_b32_e32 v146 /*v402*/, 0x20000, v144 /*v400*/
	s_mov_b32 s43, 0
	s_mov_b32 s48, 16
	s_bitset1_b32 s7, 31
	s_set_vgpr_msb 0x4400
	v_add_nc_u32_e32 v3, s0, v2
	s_sub_co_i32 s0, s33, s3
	s_mov_b32 s25, 0xffff0000
	s_max_i32 s20, s0, 0
	s_mov_b32 s28, 0x80004
	v_and_b32_e32 v4, -4, v3
	v_and_b32_e32 v5, -4, v1
	s_mov_b32 s24, 0x7510000
	v_cmp_ne_u32_e32 vcc_lo, v2, v4
	v_ashrrev_i32_e32 v2, 2, v3
	s_set_vgpr_msb 1
	v_cmp_ne_u32_e64 s0, v136 /*v392*/, v5
	s_and_b32 vcc_lo, s55, vcc_lo
	s_and_b32 s0, s55, s0
	s_cmp_lg_u32 s9, 0x80000000
	s_set_vgpr_msb 0x144
	v_subrev_co_ci_u32_e64 v138 /*v394*/, null, 0, v137 /*v393*/, s0
	s_cselect_b32 s17, s17, 0
	s_cselect_b32 s16, s9, 0x200
	s_cmp_lg_u32 s13, 0x80000000
	v_sub_co_ci_u32_e64 v139 /*v395*/, null, v2, 0, vcc_lo
	s_cselect_b32 s29, s13, 0x80
	s_cselect_b32 s3, s19, 0
	s_bfe_i32 s1, s1, 0x10018
	s_and_b32 s3, s3, 0xffff
	s_lshr_b32 s9, s1, 30
	s_lshl_b32 s13, s16, 16
	s_or_b32 s9, s9, s95
	s_or_b32 s30, s3, s13
	s_addk_co_i32 s9, 0x7f
	s_lshr_b64 s[18:19], s[16:17], 16
	s_ashr_i32 s3, s9, 2
	s_sub_co_i32 s9, s94, s33
	s_add_co_i32 s3, s3, s1
	s_add_co_i32 s1, s33, -1
	s_add_co_i32 s63, s23, s9
	s_min_i32 s1, s3, s1
	s_add_co_i32 s63, s63, 1
	s_and_b32 s3, s18, 0xffff0000
	s_add_co_i32 s1, s63, s1
	s_lshr_b32 s9, s16, 16
	s_min_i32 s1, s1, s94
	s_or_b32 s31, s3, s9
	s_max_i32 s13, s1, 1
	s_ashr_i32 s19, s14, 31
	s_add_co_i32 s3, s13, 63
	s_mov_b32 s18, s14
	s_lshr_b32 s66, s3, 6
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s65, s64, 31
	s_ashr_i32 s69, s10, 31
	s_mul_u64 s[18:19], s[2:3], s[18:19]
	s_ashr_i32 s71, s11, 31
	s_mul_u64 s[16:17], s[64:65], s[68:69]
	s_lshl_b64 s[76:77], s[18:19], 1
	s_mul_u64 s[18:19], s[64:65], s[70:71]
	s_ashr_i32 s27, s15, 31
	s_lshl_b64 s[16:17], s[16:17], 1
	s_lshl_b64 s[14:15], s[18:19], 1
	s_mul_u64 s[2:3], s[2:3], s[26:27]
	s_add_nc_u64 s[16:17], s[72:73], s[16:17]
	s_add_nc_u64 s[18:19], s[74:75], s[14:15]
	s_lshl_b64 s[14:15], s[2:3], 1
	s_min_i32 s97, s13, 64
	s_mov_b32 s1, -1
	s_add_nc_u64 s[80:81], s[16:17], s[76:77]
	s_add_nc_u64 s[78:79], s[18:19], s[14:15]
	s_cmp_lg_u32 s21, s22
	s_mov_b32 s27, 0x807fff
	s_mov_b32 s26, 0xffff7fff
	s_set_vgpr_msb 0x4400
	s_cbranch_scc0 .LBB0_15
	s_mov_b32 s21, s43
	s_mov_b32 s22, s43
	s_mov_b32 s23, s43
	s_mov_b32 s40, s43
	s_mov_b32 s41, s43
	s_mov_b32 s42, s43
	s_cmp_lg_u32 s10, 0x80000000
	tensor_load_to_lds s[4:7], s[24:31], s[20:23], s[40:43]
	s_mov_b64 s[0:1], s[4:5]
	s_mov_b64 s[2:3], s[6:7]
	s_cselect_b32 s3, s69, 0
	s_cselect_b32 s41, s68, 0x80
	s_and_b32 s1, s96, 3
	s_mov_b64 s[16:17], s[4:5]
	s_lshl_b32 s21, s1, 4
	s_lshl_b32 s22, s1, 5
	s_sub_co_i32 s17, s97, s21
	s_and_b32 s42, s3, 0xffff
	s_max_i32 s17, s17, 0
	s_mov_b64 s[18:19], s[6:7]
	s_lshl_b32 s17, s17, 16
	s_mov_b32 s2, s41
	s_or_b32 s38, s17, 0x7fff
	s_cmp_lg_u32 s11, 0x80000000
	s_mul_u64 s[8:9], s[2:3], s[22:23]
	s_cselect_b32 s49, s70, 0x80
	s_cselect_b32 s19, s71, 0
	s_mov_b32 s18, s49
	s_mul_i32 s98, s1, 0x1200
	s_mul_u64 s[22:23], s[18:19], s[22:23]
	s_mul_i32 s65, s1, 0x1100
	s_add_nc_u64 s[2:3], s[8:9], s[80:81]
	s_mov_b32 s39, 0x800000
	s_bitset1_b32 s98, 16
	s_add_nc_u64 s[82:83], s[22:23], s[78:79]
	s_mov_b32 s36, s24
	s_mov_b32 s37, s25
	s_mov_b32 s40, s48
	s_mov_b32 s1, s65
	s_bitset1_b32 s3, 31
	s_mov_b32 s44, 0xf510000
	s_mov_b32 s45, s25
	s_mov_b32 s51, s43
	s_mov_b32 s47, s39
	s_mov_b32 s46, s38
	s_mov_b32 s17, s98
	s_and_b32 s50, s19, 0xffff
	s_or_b32 s19, s83, 0x80000000
	s_mov_b32 s18, s82
	s_add_co_i32 s67, s63, -1
	s_set_vgpr_msb 17
	v_and_or_b32 v64, v142 /*v398*/, 7, v145 /*v401*/
	s_set_vgpr_msb 0x1144
	v_or_b32_e32 v154 /*v410*/, 0x20000, v144 /*v400*/
	s_add_nc_u64 s[82:83], s[74:75], s[14:15]
	s_add_nc_u64 s[84:85], s[72:73], s[76:77]
	s_set_vgpr_msb 0x4401
	v_mul_u32_u24_e32 v64, 0x120, v64
	v_and_or_b32 v64, v148 /*v404*/, 16, v64
	s_set_vgpr_msb 0x140
	v_or_b32_e32 v150 /*v406*/, 0x10000, v64
	v_or_b32_e32 v155 /*v411*/, 0x30000, v64
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4001
	ds_load_b128 v[0:3], v149 /*v405*/
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[4:7], v149 /*v405*/ offset:32
	ds_load_b128 v[8:11], v149 /*v405*/ offset:64
	ds_load_b128 v[12:15], v149 /*v405*/ offset:96
	ds_load_b128 v[16:19], v149 /*v405*/ offset:128
	ds_load_b128 v[20:23], v149 /*v405*/ offset:160
	ds_load_b128 v[24:27], v149 /*v405*/ offset:192
	ds_load_b128 v[28:31], v149 /*v405*/ offset:224
	ds_load_b128 v[32:35], v149 /*v405*/ offset:4352
	ds_load_b128 v[36:39], v149 /*v405*/ offset:4384
	ds_load_b128 v[40:43], v149 /*v405*/ offset:4416
	ds_load_b128 v[44:47], v149 /*v405*/ offset:4448
	ds_load_b128 v[48:51], v149 /*v405*/ offset:4480
	ds_load_b128 v[52:55], v149 /*v405*/ offset:4512
	ds_load_b128 v[56:59], v149 /*v405*/ offset:4544
	ds_load_b128 v[60:63], v149 /*v405*/ offset:4576
	tensor_load_to_lds s[0:3], s[36:43]
	tensor_load_to_lds s[16:19], s[44:51]
	s_ashr_i32 s0, s95, 2
	s_add_co_i32 s2, s66, -1
	s_add_co_i32 s0, s67, s0
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v135, v147 /*v403*/, v7
	s_max_i32 s0, s0, 0
	v_pk_mul_bf16 v134, v147 /*v403*/, v6
	s_add_co_i32 s0, s0, 1
	v_pk_mul_bf16 v133, v147 /*v403*/, v5
	s_ashr_i32 s1, s0, 31
	v_pk_mul_bf16 v132, v147 /*v403*/, v4
	s_lshr_b32 s1, s1, 26
	v_pk_mul_bf16 v131, v147 /*v403*/, v3
	s_add_co_i32 s1, s0, s1
	v_pk_mul_bf16 v130, v147 /*v403*/, v2
	s_and_b32 s16, s1, 0xffffffc0
	s_ashr_i32 s1, s1, 6
	s_cmp_lg_u32 s0, s16
	v_pk_mul_bf16 v129, v147 /*v403*/, v1
	s_cselect_b32 s16, -1, 0
	s_cmp_lt_i32 s0, 0
	v_pk_mul_bf16 v128, v147 /*v403*/, v0
	s_cselect_b32 s0, -1, 0
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v147 /*v403*/, v15
	s_and_b32 s0, s0, s16
	s_sub_co_ci_u32 s0, s1, 0
	v_pk_mul_bf16 v142, v147 /*v403*/, v14
	s_min_i32 s0, s0, s2
	v_pk_mul_bf16 v141, v147 /*v403*/, v13
	v_pk_mul_bf16 v140, v147 /*v403*/, v12
	v_pk_mul_bf16 v139, v147 /*v403*/, v11
	v_pk_mul_bf16 v138, v147 /*v403*/, v10
	v_pk_mul_bf16 v137, v147 /*v403*/, v9
	v_pk_mul_bf16 v136, v147 /*v403*/, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v147 /*v403*/, v23
	v_pk_mul_bf16 v150, v147 /*v403*/, v22
	v_pk_mul_bf16 v149, v147 /*v403*/, v21
	v_pk_mul_bf16 v148, v147 /*v403*/, v20
	v_pk_mul_bf16 v147, v147 /*v403*/, v19
	v_pk_mul_bf16 v146, v147 /*v403*/, v18
	v_pk_mul_bf16 v145, v147 /*v403*/, v17
	v_pk_mul_bf16 v144, v147 /*v403*/, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v147 /*v403*/, v31
	v_pk_mul_bf16 v158, v147 /*v403*/, v30
	v_pk_mul_bf16 v157, v147 /*v403*/, v29
	v_pk_mul_bf16 v156, v147 /*v403*/, v28
	v_pk_mul_bf16 v155, v147 /*v403*/, v27
	v_pk_mul_bf16 v154, v147 /*v403*/, v26
	v_pk_mul_bf16 v153, v147 /*v403*/, v25
	v_pk_mul_bf16 v152, v147 /*v403*/, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v147 /*v403*/, v39
	v_pk_mul_bf16 v166, v147 /*v403*/, v38
	v_pk_mul_bf16 v165, v147 /*v403*/, v37
	v_pk_mul_bf16 v164, v147 /*v403*/, v36
	v_pk_mul_bf16 v163, v147 /*v403*/, v35
	v_pk_mul_bf16 v162, v147 /*v403*/, v34
	v_pk_mul_bf16 v161, v147 /*v403*/, v33
	v_pk_mul_bf16 v160, v147 /*v403*/, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v147 /*v403*/, v47
	v_pk_mul_bf16 v174, v147 /*v403*/, v46
	v_pk_mul_bf16 v173, v147 /*v403*/, v45
	v_pk_mul_bf16 v172, v147 /*v403*/, v44
	v_pk_mul_bf16 v171, v147 /*v403*/, v43
	v_pk_mul_bf16 v170, v147 /*v403*/, v42
	v_pk_mul_bf16 v169, v147 /*v403*/, v41
	v_pk_mul_bf16 v168, v147 /*v403*/, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v147 /*v403*/, v55
	v_pk_mul_bf16 v182, v147 /*v403*/, v54
	v_pk_mul_bf16 v181, v147 /*v403*/, v53
	v_pk_mul_bf16 v180, v147 /*v403*/, v52
	v_pk_mul_bf16 v179, v147 /*v403*/, v51
	v_pk_mul_bf16 v178, v147 /*v403*/, v50
	v_pk_mul_bf16 v177, v147 /*v403*/, v49
	v_pk_mul_bf16 v176, v147 /*v403*/, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v147 /*v403*/, v63
	v_pk_mul_bf16 v190, v147 /*v403*/, v62
	v_pk_mul_bf16 v189, v147 /*v403*/, v61
	v_pk_mul_bf16 v188, v147 /*v403*/, v60
	v_pk_mul_bf16 v187, v147 /*v403*/, v59
	v_pk_mul_bf16 v186, v147 /*v403*/, v58
	v_pk_mul_bf16 v185, v147 /*v403*/, v57
	v_pk_mul_bf16 v184, v147 /*v403*/, v56
	v_mov_b32_e32 v0, 0
	s_max_i32 s2, s0, 0
	s_mov_b32 s3, s43
	s_cmp_lt_i32 s0, 1
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x100
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_16
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[30:31], v[6:7]
	v_mov_b64_e32 v[28:29], v[4:5]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[46:47], v[6:7]
	v_mov_b64_e32 v[44:45], v[4:5]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[40:41], v[0:1]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[62:63], v[6:7]
	v_mov_b64_e32 v[60:61], v[4:5]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[56:57], v[0:1]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[64:65], v[0:1]
	v_mov_b64_e32 v[78:79], v[6:7]
	v_mov_b64_e32 v[76:77], v[4:5]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[72:73], v[0:1]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[88:89], v[0:1]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[96:97], v[0:1]
	v_mov_b64_e32 v[110:111], v[6:7]
	v_mov_b64_e32 v[108:109], v[4:5]
	v_mov_b64_e32 v[106:107], v[2:3]
	v_mov_b64_e32 v[104:105], v[0:1]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
	s_set_vgpr_msb 64
	v_dual_mov_b32 v135 /*v391*/, 0xf149f2ca :: v_dual_mov_b32 v130 /*v386*/, 0xf149f2ca
	s_set_vgpr_msb 0x4001
	v_dual_mov_b32 v200, v155 /*v411*/ :: v_dual_mov_b32 v201, v154 /*v410*/
	s_set_vgpr_msb 0x141
	v_mov_b32_e32 v151 /*v407*/, v144 /*v400*/
	s_set_vgpr_msb 0x4140
	v_dual_mov_b32 v64 /*v320*/, v0 :: v_dual_mov_b32 v65 /*v321*/, v0
	s_lshl_b64 s[86:87], s[2:3], 6
	s_mov_b32 s16, 1
	s_sub_co_i32 s99, s13, 64
	s_add_co_i32 s88, s64, 64
	s_mov_b32 s40, 16
	s_mov_b32 s37, 0xffff0000
	s_mov_b32 s36, 0x7510000
	s_mov_b64 s[90:91], 0xffffffffffffffc0
	s_mov_b32 s100, 0x76543210
	s_mov_b32 s92, 0x3fb8aa3b
	s_mov_b32 s101, 1
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_9
.LBB0_8:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v163 /*v419*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v110 /*v366*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v102 /*v358*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v94 /*v350*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v78 /*v334*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v72 /*v328*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v68 /*v324*/, v70 /*v326*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v111 /*v367*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v103 /*v359*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v95 /*v351*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v79 /*v335*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v73 /*v329*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v69 /*v325*/, v71 /*v327*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[156:163] /*v[412:419]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v75 /*v331*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v74 /*v330*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v73 /*v329*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v72 /*v328*/, v105 /*v361*/, v109 /*v365*/
	v_cvt_pk_bf16_f32 v71 /*v327*/, v97 /*v353*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v70 /*v326*/, v89 /*v345*/, v93 /*v349*/
	v_cvt_pk_bf16_f32 v69 /*v325*/, v81 /*v337*/, v87 /*v343*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[164:171] /*v[420:427]*/, v[56:63]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v68 /*v324*/, v67 /*v323*/, v77 /*v333*/
	s_add_nc_u64 s[86:87], s[86:87], s[90:91]
	s_sub_co_i32 s99, s99, 64
	s_add_co_i32 s88, s88, 64
	s_add_co_i32 s101, s101, 1
	s_cmp_lg_u64 s[86:87], 0
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[156:163] /*v[412:419]*/, v[112:119]
	v_nop
	v_nop
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v108 /*v364*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v100 /*v356*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[164:171] /*v[420:427]*/, v[48:55]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v92 /*v348*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v66 /*v322*/, v76 /*v332*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[156:163] /*v[412:419]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[164:171] /*v[420:427]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[156:163] /*v[412:419]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[164:171] /*v[420:427]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[156:163] /*v[412:419]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[164:171] /*v[420:427]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[156:163] /*v[412:419]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[164:171] /*v[420:427]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[156:163] /*v[412:419]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[164:171] /*v[420:427]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[156:163] /*v[412:419]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[164:171] /*v[420:427]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x415
	v_pk_fma_f32 v[200:201], v[134:135] /*v[390:391]*/, v[64:65] /*v[320:321]*/, v[132:133] /*v[388:389]*/
	s_set_vgpr_msb 0x1541
	v_mov_b32_e32 v135 /*v391*/, v152 /*v408*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[130:131] /*v[386:387]*/, v[200:201]
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_dual_mov_b32 v200, v155 /*v411*/ :: v_dual_mov_b32 v201, v154 /*v410*/
	s_set_vgpr_msb 0x541
	v_mov_b32_e32 v130 /*v386*/, v153 /*v409*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[68:75] /*v[324:331]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[68:75] /*v[324:331]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[68:75] /*v[324:331]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[68:75] /*v[324:331]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[68:75] /*v[324:331]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[68:75] /*v[324:331]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[68:75] /*v[324:331]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[68:75] /*v[324:331]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_scc0 .LBB0_17
.LBB0_9:
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v154 /*v410*/, v151 /*v407*/ :: v_dual_mov_b32 v155 /*v411*/, v150 /*v406*/
	s_set_vgpr_msb 0x4140
	v_dual_mov_b32 v151 /*v407*/, v201 :: v_dual_mov_b32 v150 /*v406*/, v200
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s101, s66
	s_set_vgpr_msb 0x4000
	s_cbranch_scc1 .LBB0_11
	v_nop
	v_nop
	v_med3_i32 v192, s99, 0, 64
	s_lshr_b32 s17, s101, 31
	s_ashr_i32 s89, s88, 31
	s_add_co_i32 s17, s101, s17
	s_mul_u64 s[18:19], s[88:89], s[68:69]
	v_readfirstlane_b32 s38, v192
	s_and_b32 s17, s17, 0x7ffe
	s_lshl_b64 s[18:19], s[18:19], 1
	s_sub_co_i32 s17, s101, s17
	s_mul_u64 s[0:1], s[88:89], s[70:71]
	s_lshl_b32 s45, s17, 17
	s_sub_co_i32 s17, s38, s21
	s_add_nc_u64 s[18:19], s[84:85], s[18:19]
	s_max_i32 s38, s17, 0
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_nc_u64 s[18:19], s[8:9], s[18:19]
	s_lshl_b32 s38, s38, 16
	s_add_nc_u64 s[0:1], s[82:83], s[0:1]
	s_or_b32 s17, s65, s45
	s_bitset1_b32 s19, 31
	s_addk_co_i32 s38, 0x7fff
	s_mov_b32 s47, s39
	tensor_load_to_lds s[16:19], s[36:43]
	s_add_nc_u64 s[18:19], s[22:23], s[0:1]
	s_or_b32 s17, s98, s45
	s_bitset1_b32 s19, 31
	s_mov_b32 s45, s37
	s_mov_b32 s46, s38
	s_mov_b32 s48, s40
	s_mov_b32 s51, s43
	tensor_load_to_lds s[16:19], s[44:51]
.LBB0_11:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 1
	ds_load_b128 v[192:195], v154 /*v410*/
	ds_load_b128 v[196:199], v154 /*v410*/ offset:32
	ds_load_b128 v[200:203], v154 /*v410*/ offset:64
	ds_load_b128 v[204:207], v154 /*v410*/ offset:96
	ds_load_b128 v[208:211], v154 /*v410*/ offset:128
	ds_load_b128 v[212:215], v154 /*v410*/ offset:160
	ds_load_b128 v[216:219], v154 /*v410*/ offset:192
	ds_load_b128 v[220:223], v154 /*v410*/ offset:224
	ds_load_b128 v[224:227], v154 /*v410*/ offset:4352
	ds_load_b128 v[228:231], v154 /*v410*/ offset:4384
	ds_load_b128 v[232:235], v154 /*v410*/ offset:4416
	ds_load_b128 v[236:239], v154 /*v410*/ offset:4448
	ds_load_b128 v[240:243], v154 /*v410*/ offset:4480
	ds_load_b128 v[244:247], v154 /*v410*/ offset:4512
	ds_load_b128 v[248:251], v154 /*v410*/ offset:4544
	ds_load_b128 v[252:255], v154 /*v410*/ offset:4576
	s_set_vgpr_msb 0x141
	ds_load_b128 v[0:3] /*v[256:259]*/, v154 /*v410*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v154 /*v410*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v154 /*v410*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v154 /*v410*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v154 /*v410*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v154 /*v410*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v154 /*v410*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v154 /*v410*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v154 /*v410*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v154 /*v410*/ offset:13088
	ds_load_b128 v[40:43] /*v[296:299]*/, v154 /*v410*/ offset:13120
	ds_load_b128 v[44:47] /*v[300:303]*/, v154 /*v410*/ offset:13152
	ds_load_b128 v[48:51] /*v[304:307]*/, v154 /*v410*/ offset:13184
	ds_load_b128 v[52:55] /*v[308:311]*/, v154 /*v410*/ offset:13216
	ds_load_b128 v[56:59] /*v[312:315]*/, v154 /*v410*/ offset:13248
	ds_load_b128 v[60:63] /*v[316:319]*/, v154 /*v410*/ offset:13280
	s_set_vgpr_msb 0x4140
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[192:199], v[128:135], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[224:231], v[128:135], 0
	s_set_vgpr_msb 0x4041
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	s_set_vgpr_msb 0x4150
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[192:199], v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[200:207], v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[164:171] /*v[420:427]*/, v[224:231], v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[232:239], v[136:143], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[428:435]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[8:15] /*v[264:271]*/, v[136:143], v[120:127] /*v[376:383]*/
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[444:451]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[200:207], v[168:175], v[156:163] /*v[412:419]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[208:215], v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[164:171] /*v[420:427]*/, v[232:239], v[168:175], v[164:171] /*v[420:427]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[240:247], v[144:151], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[428:435]*/, v[8:15] /*v[264:271]*/, v[168:175], v[172:179] /*v[428:435]*/
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[16:23] /*v[272:279]*/, v[144:151], v[120:127] /*v[376:383]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[40:47] /*v[296:303]*/, v[136:143], v[180:187] /*v[436:443]*/
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[444:451]*/, v[40:47] /*v[296:303]*/, v[168:175], v[188:195] /*v[444:451]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[208:215], v[176:183], v[156:163] /*v[412:419]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[216:223], v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[164:171] /*v[420:427]*/, v[240:247], v[176:183], v[164:171] /*v[420:427]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[248:255], v[152:159], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[428:435]*/, v[16:23] /*v[272:279]*/, v[176:183], v[172:179] /*v[428:435]*/
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[24:31] /*v[280:287]*/, v[152:159], v[120:127] /*v[376:383]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[48:55] /*v[304:311]*/, v[144:151], v[180:187] /*v[436:443]*/
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[444:451]*/, v[48:55] /*v[304:311]*/, v[176:183], v[188:195] /*v[444:451]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[216:223], v[184:191], v[156:163] /*v[412:419]*/
	v_wmma_f32_16x16x32_bf16 v[164:171] /*v[420:427]*/, v[248:255], v[184:191], v[164:171] /*v[420:427]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[428:435]*/, v[24:31] /*v[280:287]*/, v[184:191], v[172:179] /*v[428:435]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[56:63] /*v[312:319]*/, v[152:159], v[180:187] /*v[436:443]*/
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[444:451]*/, v[56:63] /*v[312:319]*/, v[184:191], v[188:195] /*v[444:451]*/
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v155 /*v411*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v155 /*v411*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v155 /*v411*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v155 /*v411*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v155 /*v411*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v155 /*v411*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v155 /*v411*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v155 /*v411*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v155 /*v411*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v155 /*v411*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v155 /*v411*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v155 /*v411*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v155 /*v411*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v155 /*v411*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v155 /*v411*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v155 /*v411*/ offset:13920
	s_set_vgpr_msb 0x5101
	ds_load_tr16_b128 v[248:251], v155 /*v411*/ offset:128
	ds_load_tr16_b128 v[232:235], v155 /*v411*/ offset:160
	ds_load_tr16_b128 v[252:255], v155 /*v411*/ offset:4736
	ds_load_tr16_b128 v[236:239], v155 /*v411*/ offset:4768
	ds_load_tr16_b128 v[240:243], v155 /*v411*/ offset:9344
	ds_load_tr16_b128 v[224:227], v155 /*v411*/ offset:9376
	ds_load_tr16_b128 v[244:247], v155 /*v411*/ offset:13952
	ds_load_tr16_b128 v[228:231], v155 /*v411*/ offset:13984
	ds_load_tr16_b128 v[216:219], v155 /*v411*/ offset:192
	ds_load_tr16_b128 v[200:203], v155 /*v411*/ offset:224
	ds_load_tr16_b128 v[220:223], v155 /*v411*/ offset:4800
	ds_load_tr16_b128 v[204:207], v155 /*v411*/ offset:4832
	ds_load_tr16_b128 v[208:211], v155 /*v411*/ offset:9408
	ds_load_tr16_b128 v[192:195], v155 /*v411*/ offset:9440
	ds_load_tr16_b128 v[212:215], v155 /*v411*/ offset:14016
	ds_load_tr16_b128 v[196:199], v155 /*v411*/ offset:14048
	s_set_vgpr_msb 0x155
	v_dual_max_num_f32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/ :: v_dual_max_num_f32 v75 /*v331*/, v156 /*v412*/, v157 /*v413*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v80 /*v336*/, v95 /*v351*/, v96 /*v352*/, v97 /*v353*/
	v_max3_num_f32 v82 /*v338*/, v98 /*v354*/, v99 /*v355*/, v100 /*v356*/
	v_max3_num_f32 v84 /*v340*/, v101 /*v357*/, v120 /*v376*/, v121 /*v377*/
	v_max3_num_f32 v77 /*v333*/, v159 /*v415*/, v160 /*v416*/, v161 /*v417*/
	v_max3_num_f32 v81 /*v337*/, v165 /*v421*/, v166 /*v422*/, v167 /*v423*/
	v_max3_num_f32 v83 /*v339*/, v168 /*v424*/, v169 /*v425*/, v170 /*v426*/
	v_max3_num_f32 v85 /*v341*/, v171 /*v427*/, v172 /*v428*/, v173 /*v429*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v94 /*v350*/
	v_max3_num_f32 v86 /*v342*/, v122 /*v378*/, v123 /*v379*/, v124 /*v380*/
	v_max3_num_f32 v88 /*v344*/, v125 /*v381*/, v126 /*v382*/, v127 /*v383*/
	v_max3_num_f32 v90 /*v346*/, v180 /*v436*/, v181 /*v437*/, v182 /*v438*/
	v_max3_num_f32 v92 /*v348*/, v183 /*v439*/, v184 /*v440*/, v185 /*v441*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v82 /*v338*/, v84 /*v340*/
	v_max3_num_f32 v79 /*v335*/, v162 /*v418*/, v163 /*v419*/, v164 /*v420*/
	v_max3_num_f32 v87 /*v343*/, v174 /*v430*/, v175 /*v431*/, v176 /*v432*/
	v_max3_num_f32 v89 /*v345*/, v177 /*v433*/, v178 /*v434*/, v179 /*v435*/
	v_max3_num_f32 v91 /*v347*/, v188 /*v444*/, v189 /*v445*/, v190 /*v446*/
	v_max3_num_f32 v93 /*v349*/, v191 /*v447*/, v192 /*v448*/, v193 /*v449*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v158 /*v414*/, v77 /*v333*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v83 /*v339*/, v85 /*v341*/
	v_max3_num_f32 v80 /*v336*/, v86 /*v342*/, v88 /*v344*/, v90 /*v346*/
	v_max3_num_f32 v81 /*v337*/, v92 /*v348*/, v186 /*v442*/, v187 /*v443*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	v_max3_num_f32 v76 /*v332*/, v87 /*v343*/, v89 /*v345*/, v91 /*v347*/
	v_max3_num_f32 v78 /*v334*/, v93 /*v349*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v80 /*v336*/, v81 /*v337*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_dual_mov_b32 v76 /*v332*/, v74 /*v330*/ :: v_dual_mov_b32 v77 /*v333*/, v75 /*v331*/
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s100, 0xfedcba98
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s100, 0xfedcba98
	v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v76 /*v332*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v77 /*v333*/
	v_sub_f32_e32 v76 /*v332*/, v74 /*v330*/, v130 /*v386*/
	v_dual_max_num_f32 v74 /*v330*/, v130 /*v386*/, v74 /*v330*/ :: v_dual_sub_f32 v77 /*v333*/, v75 /*v331*/, v135 /*v391*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v76 /*v332*/
	v_cmp_lt_f32_e64 s0, 0x41000000, v77 /*v333*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s1, -1, 0
	s_cmp_lg_u32 s0, 0
	v_dual_cndmask_b32 v153 /*v409*/, v74 /*v330*/, v130 /*v386*/, s1 :: v_dual_max_num_f32 v74 /*v330*/, v135 /*v391*/, v75 /*v331*/
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u32 s0, 0
	s_cselect_b32 s0, -1, 0
	v_mul_f32_e32 v118 /*v374*/, 0xbfb8aa3b, v153 /*v409*/
	v_cndmask_b32_e64 v152 /*v408*/, v74 /*v330*/, v135 /*v391*/, s0
	v_sub_f32_e32 v134 /*v390*/, v130 /*v386*/, v153 /*v409*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_mul_f32_e32 v132 /*v388*/, 0xbfb8aa3b, v152 /*v408*/
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[72:73] /*v[328:329]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v66 /*v322*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v70 /*v326*/, v67 /*v323*/
	v_exp_f32_e32 v72 /*v328*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_exp_f32_e32 v78 /*v334*/, v76 /*v332*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[94:95] /*v[350:351]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_nop
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[96:97] /*v[352:353]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v69 /*v325*/, v156 /*v412*/
	v_exp_f32_e32 v71 /*v327*/, v157 /*v413*/
	v_exp_f32_e32 v73 /*v329*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[162:163] /*v[418:419]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v75 /*v331*/, v159 /*v415*/
	v_exp_f32_e32 v79 /*v335*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[166:167] /*v[422:423]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v80 /*v336*/
	v_exp_f32_e32 v90 /*v346*/, v81 /*v337*/
	v_exp_f32_e32 v94 /*v350*/, v66 /*v322*/
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[98:99] /*v[354:355]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v98 /*v354*/, v67 /*v323*/
	v_exp_f32_e32 v102 /*v358*/, v76 /*v332*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[100:101] /*v[356:357]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v106 /*v362*/, v77 /*v333*/
	v_nop
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[120:121] /*v[376:377]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v156 /*v412*/
	v_exp_f32_e32 v91 /*v347*/, v157 /*v413*/
	v_exp_f32_e32 v95 /*v351*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[168:169] /*v[424:425]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v159 /*v415*/
	v_exp_f32_e32 v103 /*v359*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[170:171] /*v[426:427]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[172:173] /*v[428:429]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v80 /*v336*/
	v_exp_f32_e32 v114 /*v370*/, v81 /*v337*/
	v_exp_f32_e32 v116 /*v372*/, v66 /*v322*/
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[122:123] /*v[378:379]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v122 /*v378*/, v67 /*v323*/
	v_exp_f32_e32 v66 /*v322*/, v76 /*v332*/
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[124:125] /*v[380:381]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v76 /*v332*/, v77 /*v333*/
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[126:127] /*v[382:383]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v111 /*v367*/, v156 /*v412*/
	v_exp_f32_e32 v115 /*v371*/, v157 /*v413*/
	v_exp_f32_e32 v117 /*v373*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[174:175] /*v[430:431]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v159 /*v415*/
	v_exp_f32_e32 v67 /*v323*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[176:177] /*v[432:433]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v77 /*v333*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[178:179] /*v[434:435]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v86 /*v342*/, v81 /*v337*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[180:181] /*v[436:437]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[182:183] /*v[438:439]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v97 /*v353*/
	v_pk_fma_f32 v[120:121] /*v[376:377]*/, v[184:185] /*v[440:441]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v81 /*v337*/, v156 /*v412*/
	v_exp_f32_e32 v87 /*v343*/, v157 /*v413*/
	v_exp_f32_e32 v89 /*v345*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[188:189] /*v[444:445]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v93 /*v349*/, v159 /*v415*/
	v_exp_f32_e32 v97 /*v353*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[190:191] /*v[446:447]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v101 /*v357*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[192:193] /*v[448:449]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v80 /*v336*/, v80 /*v336*/
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_exp_f32_e32 v108 /*v364*/, v105 /*v361*/
	v_pk_fma_f32 v[126:127] /*v[382:383]*/, v[186:187] /*v[442:443]*/, s[92:93], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v124 /*v380*/, v121 /*v377*/
	v_exp_f32_e32 v105 /*v361*/, v156 /*v412*/
	v_exp_f32_e32 v109 /*v365*/, v157 /*v413*/
	v_exp_f32_e32 v113 /*v369*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[194:195] /*v[450:451]*/, s[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v159 /*v415*/
	v_exp_f32_e32 v121 /*v377*/, v160 /*v416*/
	v_exp_f32_e32 v125 /*v381*/, v161 /*v417*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[68:69] /*v[324:325]*/, v[70:71] /*v[326:327]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[74:75] /*v[330:331]*/, v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[98:99] /*v[354:355]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[110:111] /*v[366:367]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v128 /*v384*/, v127 /*v383*/
	v_exp_f32_e32 v127 /*v383*/, v132 /*v388*/
	v_exp_f32_e32 v129 /*v385*/, v133 /*v389*/
	v_nop
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[72:73] /*v[328:329]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[82:83] /*v[338:339]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[122:123] /*v[378:379]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[106:107] /*v[362:363]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[116:117] /*v[372:373]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[92:93] /*v[348:349]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_exp_f32_e32 v120 /*v376*/, v120 /*v376*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[94:95] /*v[350:351]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[76:77] /*v[332:333]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[104:105] /*v[360:361]*/, v[108:109] /*v[364:365]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[88:89] /*v[344:345]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[100:101] /*v[356:357]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[162:163] /*v[418:419]*/
	v_exp_f32_e32 v126 /*v382*/, v126 /*v382*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[112:113] /*v[368:369]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[166:167] /*v[422:423]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[160:161] /*v[416:417]*/
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[124:125] /*v[380:381]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[162:163] /*v[418:419]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[158:159] /*v[414:415]*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[156:157] /*v[412:413]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/, v[132:133] /*v[388:389]*/
	v_dual_mov_b32 v132 /*v388*/, v130 /*v386*/ :: v_dual_mov_b32 v133 /*v389*/, v131 /*v387*/
	v_permlanex16_b32 v132 /*v388*/, v132 /*v388*/, s100, 0xfedcba98
	v_permlanex16_b32 v133 /*v389*/, v133 /*v389*/, s100, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_13
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_13:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v135 /*v391*/, v135 /*v391*/, v152 /*v408*/
	s_and_b32 s0, s1, exec_lo
	s_cselect_b32 s0, 1, 0
	s_cmp_lg_u32 s0, 1
	v_mul_f32_e32 v135 /*v391*/, 0x3fb8aa3b, v135 /*v391*/
	v_exp_f32_e32 v135 /*v391*/, v135 /*v391*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_8
	v_nop
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v156 /*v412*/, v135 /*v391*/
	s_set_vgpr_msb 0x4104
	v_pk_mul_f32 v[62:63], v[62:63], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[156:157] /*v[412:413]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_8
.LBB0_15:
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB0_28
	s_branch .LBB0_48
.LBB0_16:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[30:31], v[6:7]
	v_mov_b64_e32 v[28:29], v[4:5]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[46:47], v[6:7]
	v_mov_b64_e32 v[44:45], v[4:5]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[40:41], v[0:1]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[62:63], v[6:7]
	v_mov_b64_e32 v[60:61], v[4:5]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[56:57], v[0:1]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[64:65], v[0:1]
	v_mov_b64_e32 v[78:79], v[6:7]
	v_mov_b64_e32 v[76:77], v[4:5]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[72:73], v[0:1]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[88:89], v[0:1]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[96:97], v[0:1]
	v_mov_b64_e32 v[110:111], v[6:7]
	v_mov_b64_e32 v[108:109], v[4:5]
	v_mov_b64_e32 v[106:107], v[2:3]
	v_mov_b64_e32 v[104:105], v[0:1]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
	s_set_vgpr_msb 64
	v_dual_mov_b32 v152 /*v408*/, 0xf149f2ca :: v_dual_mov_b32 v65 /*v321*/, v0
	v_dual_mov_b32 v64 /*v320*/, v0 :: v_dual_mov_b32 v153 /*v409*/, 0xf149f2ca
	s_set_vgpr_msb 0x4041
	v_mov_b32_e32 v151 /*v407*/, v144 /*v400*/
	s_set_vgpr_msb 0x4100
.LBB0_17:
	s_cmp_ge_u32 s2, s66
	s_cbranch_scc1 .LBB0_26
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4
	v_dual_add_nc_u32 v192, s67, v138 /*v394*/ :: v_dual_add_nc_u32 v193, s67, v139 /*v395*/
	s_add_co_i32 s0, s94, -1
	s_mov_b32 s43, 0
	s_mov_b32 s16, 1
	s_set_vgpr_msb 0x440
	v_min_i32_e32 v156 /*v412*/, s0, v192
	v_min_i32_e32 v157 /*v413*/, s0, v193
	s_mov_b32 s67, s43
	s_mov_b32 s40, 16
	s_mov_b32 s39, 0x800000
	s_mov_b32 s37, 0xffff0000
	s_mov_b32 s36, 0x7510000
	s_mov_b32 s44, 0xf510000
	s_mov_b32 s1, 0x76543210
	s_mov_b32 s86, 0x3fb8aa3b
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_20
.LBB0_19:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v167 /*v423*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v108 /*v364*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v100 /*v356*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v92 /*v348*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v76 /*v332*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v70 /*v326*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v66 /*v322*/, v68 /*v324*/
	v_cvt_pk_bf16_f32 v175 /*v431*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v109 /*v365*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v101 /*v357*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v93 /*v349*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v77 /*v333*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v71 /*v327*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v67 /*v323*/, v69 /*v325*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[160:167] /*v[416:423]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v101 /*v357*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v100 /*v356*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v99 /*v355*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v98 /*v354*/, v105 /*v361*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v97 /*v353*/, v97 /*v353*/, v103 /*v359*/
	v_dual_fmac_f32 v153 /*v409*/, v132 /*v388*/, v64 /*v320*/ :: v_dual_fmac_f32 v155 /*v411*/, v134 /*v390*/, v65 /*v321*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[168:175] /*v[424:431]*/, v[56:63]
	s_add_nc_u64 s[2:3], s[2:3], 1
	s_set_vgpr_msb 0x545
	v_mov_b32_e32 v152 /*v408*/, v135 /*v391*/
	v_cmp_ge_u64_e64 s0, s[2:3], s[66:67]
	v_dual_add_f32 v64 /*v320*/, v153 /*v409*/, v130 /*v386*/ :: v_dual_add_f32 v65 /*v321*/, v155 /*v411*/, v131 /*v387*/
	v_nop
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[160:167] /*v[416:423]*/, v[112:119]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v110 /*v366*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v102 /*v358*/
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v94 /*v350*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v72 /*v328*/, v78 /*v334*/
	v_cvt_pk_bf16_f32 v96 /*v352*/, v89 /*v345*/, v95 /*v351*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[168:175] /*v[424:431]*/, v[48:55]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v95 /*v351*/, v81 /*v337*/, v87 /*v343*/
	v_cvt_pk_bf16_f32 v94 /*v350*/, v73 /*v329*/, v79 /*v335*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v155 /*v411*/, v150 /*v406*/ :: v_dual_mov_b32 v150 /*v406*/, v154 /*v410*/
	v_dual_mov_b32 v154 /*v410*/, v151 /*v407*/ :: v_dual_mov_b32 v151 /*v407*/, v158 /*v414*/
	v_mov_b32_e32 v153 /*v409*/, v133 /*v389*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[160:167] /*v[416:423]*/, v[104:111]
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[168:175] /*v[424:431]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[160:167] /*v[416:423]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[168:175] /*v[424:431]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[160:167] /*v[416:423]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[168:175] /*v[424:431]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[160:167] /*v[416:423]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[168:175] /*v[424:431]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[160:167] /*v[416:423]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[168:175] /*v[424:431]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[160:167] /*v[416:423]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[168:175] /*v[424:431]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[94:101] /*v[350:357]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[94:101] /*v[350:357]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[94:101] /*v[350:357]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[94:101] /*v[350:357]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[94:101] /*v[350:357]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[94:101] /*v[350:357]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[94:101] /*v[350:357]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[94:101] /*v[350:357]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_27
.LBB0_20:
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v158 /*v414*/, v154 /*v410*/ :: v_dual_mov_b32 v154 /*v410*/, v155 /*v411*/
	s_add_co_i32 s0, s2, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s0, s66
	s_set_vgpr_msb 0x4100
	s_cbranch_scc1 .LBB0_22
	s_lshl_b32 s17, s0, 6
	s_lshl_b32 s0, s0, 17
	s_sub_co_i32 s19, s13, s17
	s_add_co_i32 s18, s17, s64
	v_nop
	v_nop
	v_nop
	v_med3_i32 v192, s19, 0, 64
	s_ashr_i32 s19, s18, 31
	s_and_b32 s0, s0, 0x20000
	s_mul_u64 s[46:47], s[18:19], s[70:71]
	s_mul_u64 s[18:19], s[18:19], s[68:69]
	v_readfirstlane_b32 s38, v192
	s_lshl_b64 s[18:19], s[18:19], 1
	s_lshl_b64 s[46:47], s[46:47], 1
	s_add_nc_u64 s[18:19], s[84:85], s[18:19]
	s_or_b32 s17, s65, s0
	s_sub_co_i32 s38, s38, s21
	s_add_nc_u64 s[18:19], s[8:9], s[18:19]
	s_max_i32 s38, s38, 0
	s_add_nc_u64 s[46:47], s[82:83], s[46:47]
	s_lshl_b32 s38, s38, 16
	s_bitset1_b32 s19, 31
	s_addk_co_i32 s38, 0x7fff
	s_mov_b32 s45, s37
	tensor_load_to_lds s[16:19], s[36:43]
	s_add_nc_u64 s[18:19], s[22:23], s[46:47]
	s_or_b32 s17, s98, s0
	s_bitset1_b32 s19, 31
	s_mov_b32 s46, s38
	s_mov_b32 s47, s39
	s_mov_b32 s48, s40
	s_mov_b32 s51, s43
	tensor_load_to_lds s[16:19], s[44:51]
.LBB0_22:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 1
	ds_load_b128 v[192:195], v151 /*v407*/
	ds_load_b128 v[196:199], v151 /*v407*/ offset:32
	ds_load_b128 v[200:203], v151 /*v407*/ offset:64
	ds_load_b128 v[204:207], v151 /*v407*/ offset:96
	ds_load_b128 v[208:211], v151 /*v407*/ offset:128
	ds_load_b128 v[212:215], v151 /*v407*/ offset:160
	ds_load_b128 v[216:219], v151 /*v407*/ offset:192
	ds_load_b128 v[220:223], v151 /*v407*/ offset:224
	ds_load_b128 v[224:227], v151 /*v407*/ offset:4352
	ds_load_b128 v[228:231], v151 /*v407*/ offset:4384
	ds_load_b128 v[232:235], v151 /*v407*/ offset:4416
	ds_load_b128 v[236:239], v151 /*v407*/ offset:4448
	ds_load_b128 v[240:243], v151 /*v407*/ offset:4480
	ds_load_b128 v[244:247], v151 /*v407*/ offset:4512
	ds_load_b128 v[248:251], v151 /*v407*/ offset:4544
	ds_load_b128 v[252:255], v151 /*v407*/ offset:4576
	s_set_vgpr_msb 0x141
	ds_load_b128 v[0:3] /*v[256:259]*/, v151 /*v407*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v151 /*v407*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v151 /*v407*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v151 /*v407*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v151 /*v407*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v151 /*v407*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v151 /*v407*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v151 /*v407*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v151 /*v407*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v151 /*v407*/ offset:13088
	ds_load_b128 v[40:43] /*v[296:299]*/, v151 /*v407*/ offset:13120
	ds_load_b128 v[44:47] /*v[300:303]*/, v151 /*v407*/ offset:13152
	ds_load_b128 v[48:51] /*v[304:307]*/, v151 /*v407*/ offset:13184
	ds_load_b128 v[52:55] /*v[308:311]*/, v151 /*v407*/ offset:13216
	ds_load_b128 v[56:59] /*v[312:315]*/, v151 /*v407*/ offset:13248
	ds_load_b128 v[60:63] /*v[316:319]*/, v151 /*v407*/ offset:13280
	s_set_vgpr_msb 0x4140
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[192:199], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[192:199], v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[224:231], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0x4041
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x4150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[200:207], v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[200:207], v[168:175], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[232:239], v[136:143], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[232:239], v[168:175], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[8:15] /*v[264:271]*/, v[136:143], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[8:15] /*v[264:271]*/, v[168:175], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[40:47] /*v[296:303]*/, v[136:143], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[40:47] /*v[296:303]*/, v[168:175], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[208:215], v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[208:215], v[176:183], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[240:247], v[144:151], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[240:247], v[176:183], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[16:23] /*v[272:279]*/, v[144:151], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[16:23] /*v[272:279]*/, v[176:183], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[48:55] /*v[304:311]*/, v[144:151], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[48:55] /*v[304:311]*/, v[176:183], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[216:223], v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[216:223], v[184:191], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[248:255], v[152:159], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[248:255], v[184:191], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[24:31] /*v[280:287]*/, v[152:159], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[24:31] /*v[280:287]*/, v[184:191], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[56:63] /*v[312:319]*/, v[152:159], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[56:63] /*v[312:319]*/, v[184:191], v[122:129] /*v[378:385]*/
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v150 /*v406*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v150 /*v406*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v150 /*v406*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v150 /*v406*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v150 /*v406*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v150 /*v406*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v150 /*v406*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v150 /*v406*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v150 /*v406*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v150 /*v406*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v150 /*v406*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v150 /*v406*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v150 /*v406*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v150 /*v406*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v150 /*v406*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v150 /*v406*/ offset:13920
	s_set_vgpr_msb 0x5101
	ds_load_tr16_b128 v[248:251], v150 /*v406*/ offset:128
	ds_load_tr16_b128 v[232:235], v150 /*v406*/ offset:160
	ds_load_tr16_b128 v[252:255], v150 /*v406*/ offset:4736
	ds_load_tr16_b128 v[236:239], v150 /*v406*/ offset:4768
	ds_load_tr16_b128 v[240:243], v150 /*v406*/ offset:9344
	ds_load_tr16_b128 v[224:227], v150 /*v406*/ offset:9376
	ds_load_tr16_b128 v[244:247], v150 /*v406*/ offset:13952
	ds_load_tr16_b128 v[228:231], v150 /*v406*/ offset:13984
	ds_load_tr16_b128 v[216:219], v150 /*v406*/ offset:192
	ds_load_tr16_b128 v[200:203], v150 /*v406*/ offset:224
	ds_load_tr16_b128 v[220:223], v150 /*v406*/ offset:4800
	ds_load_tr16_b128 v[204:207], v150 /*v406*/ offset:4832
	ds_load_tr16_b128 v[208:211], v150 /*v406*/ offset:9408
	ds_load_tr16_b128 v[192:195], v150 /*v406*/ offset:9440
	ds_load_tr16_b128 v[212:215], v150 /*v406*/ offset:14016
	ds_load_tr16_b128 v[196:199], v150 /*v406*/ offset:14048
	s_set_vgpr_msb 0x155
	v_lshl_or_b32 v132 /*v388*/, s2, 6, v145 /*v401*/
	v_cmp_le_i32_e32 vcc_lo, v132 /*v388*/, v156 /*v412*/
	v_dual_add_nc_u32 v177 /*v433*/, 17, v132 /*v388*/ :: v_dual_bitop2_b32 v133 /*v389*/, 2, v132 /*v388*/ bitop3:0x54
	v_dual_add_nc_u32 v178 /*v434*/, 18, v132 /*v388*/ :: v_dual_bitop2_b32 v134 /*v390*/, 3, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e32 v66 /*v322*/, 0xff800000, v66 /*v322*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v132 /*v388*/, v156 /*v412*/
	v_dual_add_nc_u32 v179 /*v435*/, 19, v132 /*v388*/ :: v_dual_bitop2_b32 v135 /*v391*/, 4, v132 /*v388*/ bitop3:0x54
	s_wait_alu depctr_vm_vsrc(6)
	v_or_b32_e32 v155 /*v411*/, 5, v132 /*v388*/
	v_or_b32_e32 v159 /*v415*/, 6, v132 /*v388*/
	v_cndmask_b32_e32 v67 /*v323*/, 0xff800000, v67 /*v323*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v133 /*v389*/, v156 /*v412*/
	v_dual_add_nc_u32 v182 /*v438*/, 22, v132 /*v388*/ :: v_dual_bitop2_b32 v175 /*v431*/, 7, v132 /*v388*/ bitop3:0x54
	v_dual_add_nc_u32 v183 /*v439*/, 23, v132 /*v388*/ :: v_dual_bitop2_b32 v176 /*v432*/, 16, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e32 v68 /*v324*/, 0xff800000, v68 /*v324*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v134 /*v390*/, v156 /*v412*/
	v_dual_add_nc_u32 v193 /*v449*/, 49, v132 /*v388*/ :: v_dual_bitop2_b32 v184 /*v440*/, 32, v132 /*v388*/ bitop3:0x54
	v_dual_add_nc_u32 v194 /*v450*/, 50, v132 /*v388*/ :: v_dual_bitop2_b32 v185 /*v441*/, 33, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e32 v69 /*v325*/, 0xff800000, v69 /*v325*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v135 /*v391*/, v156 /*v412*/
	v_or_b32_e32 v186 /*v442*/, 34, v132 /*v388*/
	v_or_b32_e32 v191 /*v447*/, 39, v132 /*v388*/
	v_dual_add_nc_u32 v199 /*v455*/, 55, v132 /*v388*/ :: v_dual_bitop2_b32 v192 /*v448*/, 48, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e32 v70 /*v326*/, 0xff800000, v70 /*v326*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v155 /*v411*/, v156 /*v412*/
	v_cndmask_b32_e32 v71 /*v327*/, 0xff800000, v71 /*v327*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v159 /*v415*/, v156 /*v412*/
	v_cndmask_b32_e32 v72 /*v328*/, 0xff800000, v72 /*v328*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v175 /*v431*/, v156 /*v412*/
	v_cndmask_b32_e32 v73 /*v329*/, 0xff800000, v73 /*v329*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v176 /*v432*/, v156 /*v412*/
	v_cndmask_b32_e32 v82 /*v338*/, 0xff800000, v82 /*v338*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v177 /*v433*/, v156 /*v412*/
	v_cndmask_b32_e32 v83 /*v339*/, 0xff800000, v83 /*v339*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v178 /*v434*/, v156 /*v412*/
	v_cndmask_b32_e32 v130 /*v386*/, 0xff800000, v84 /*v340*/, vcc_lo
	v_add_nc_u32_e32 v84 /*v340*/, 20, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v179 /*v435*/, v156 /*v412*/
	v_cndmask_b32_e32 v131 /*v387*/, 0xff800000, v85 /*v341*/, vcc_lo
	v_add_nc_u32_e32 v85 /*v341*/, 21, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v84 /*v340*/, v156 /*v412*/
	v_cndmask_b32_e32 v86 /*v342*/, 0xff800000, v86 /*v342*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v85 /*v341*/, v156 /*v412*/
	v_cndmask_b32_e32 v87 /*v343*/, 0xff800000, v87 /*v343*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v182 /*v438*/, v156 /*v412*/
	v_cndmask_b32_e32 v88 /*v344*/, 0xff800000, v88 /*v344*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v183 /*v439*/, v156 /*v412*/
	v_cndmask_b32_e32 v89 /*v345*/, 0xff800000, v89 /*v345*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v184 /*v440*/, v156 /*v412*/
	v_cndmask_b32_e32 v160 /*v416*/, 0xff800000, v98 /*v354*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v185 /*v441*/, v156 /*v412*/
	v_or_b32_e32 v98 /*v354*/, 35, v132 /*v388*/
	v_cndmask_b32_e32 v161 /*v417*/, 0xff800000, v99 /*v355*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v186 /*v442*/, v156 /*v412*/
	v_or_b32_e32 v99 /*v355*/, 36, v132 /*v388*/
	v_cndmask_b32_e32 v162 /*v418*/, 0xff800000, v100 /*v356*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v98 /*v354*/, v156 /*v412*/
	v_or_b32_e32 v100 /*v356*/, 37, v132 /*v388*/
	v_cndmask_b32_e32 v163 /*v419*/, 0xff800000, v101 /*v357*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v99 /*v355*/, v156 /*v412*/
	v_or_b32_e32 v101 /*v357*/, 38, v132 /*v388*/
	v_cndmask_b32_e32 v102 /*v358*/, 0xff800000, v102 /*v358*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v100 /*v356*/, v156 /*v412*/
	v_cndmask_b32_e32 v103 /*v359*/, 0xff800000, v103 /*v359*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v101 /*v357*/, v156 /*v412*/
	v_cndmask_b32_e32 v104 /*v360*/, 0xff800000, v104 /*v360*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v191 /*v447*/, v156 /*v412*/
	v_cndmask_b32_e32 v105 /*v361*/, 0xff800000, v105 /*v361*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v192 /*v448*/, v156 /*v412*/
	v_cndmask_b32_e32 v164 /*v420*/, 0xff800000, v114 /*v370*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v193 /*v449*/, v156 /*v412*/
	v_dual_cndmask_b32 v165 /*v421*/, 0xff800000, v115 /*v371*/ :: v_dual_add_nc_u32 v114 /*v370*/, 51, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v194 /*v450*/, v156 /*v412*/
	v_add_nc_u32_e32 v115 /*v371*/, 52, v132 /*v388*/
	v_cndmask_b32_e32 v166 /*v422*/, 0xff800000, v116 /*v372*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v114 /*v370*/, v156 /*v412*/
	v_dual_cndmask_b32 v167 /*v423*/, 0xff800000, v117 /*v373*/ :: v_dual_add_nc_u32 v116 /*v372*/, 53, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v115 /*v371*/, v156 /*v412*/
	v_dual_cndmask_b32 v118 /*v374*/, 0xff800000, v118 /*v374*/ :: v_dual_add_nc_u32 v117 /*v373*/, 54, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v116 /*v372*/, v156 /*v412*/
	v_cndmask_b32_e32 v119 /*v375*/, 0xff800000, v119 /*v375*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v117 /*v373*/, v156 /*v412*/
	v_cndmask_b32_e32 v120 /*v376*/, 0xff800000, v120 /*v376*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v199 /*v455*/, v156 /*v412*/
	v_cndmask_b32_e32 v121 /*v377*/, 0xff800000, v121 /*v377*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v132 /*v388*/, v157 /*v413*/
	v_cndmask_b32_e32 v168 /*v424*/, 0xff800000, v74 /*v330*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v132 /*v388*/, v157 /*v413*/
	v_max_num_f32_e32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/
	v_cndmask_b32_e32 v169 /*v425*/, 0xff800000, v75 /*v331*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v133 /*v389*/, v157 /*v413*/
	v_dual_max_num_f32 v75 /*v331*/, v168 /*v424*/, v169 /*v425*/ :: v_dual_cndmask_b32 v170 /*v426*/, 0xff800000, v76 /*v332*/
	v_cmp_le_i32_e32 vcc_lo, v134 /*v390*/, v157 /*v413*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_cndmask_b32_e32 v171 /*v427*/, 0xff800000, v77 /*v333*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v135 /*v391*/, v157 /*v413*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	v_cndmask_b32_e32 v172 /*v428*/, 0xff800000, v78 /*v334*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v155 /*v411*/, v157 /*v413*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v82 /*v338*/
	v_cndmask_b32_e32 v173 /*v429*/, 0xff800000, v79 /*v335*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v159 /*v415*/, v157 /*v413*/
	v_max3_num_f32 v77 /*v333*/, v171 /*v427*/, v172 /*v428*/, v173 /*v429*/
	v_cndmask_b32_e32 v174 /*v430*/, 0xff800000, v80 /*v336*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v175 /*v431*/, v157 /*v413*/
	v_max3_num_f32 v80 /*v336*/, v83 /*v339*/, v130 /*v386*/, v131 /*v387*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v170 /*v426*/, v77 /*v333*/
	v_cndmask_b32_e32 v175 /*v431*/, 0xff800000, v81 /*v337*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v176 /*v432*/, v157 /*v413*/
	v_cndmask_b32_e32 v176 /*v432*/, 0xff800000, v90 /*v346*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v177 /*v433*/, v157 /*v413*/
	v_max3_num_f32 v90 /*v346*/, v89 /*v345*/, v160 /*v416*/, v161 /*v417*/
	v_cndmask_b32_e32 v177 /*v433*/, 0xff800000, v91 /*v347*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v178 /*v434*/, v157 /*v413*/
	v_max3_num_f32 v79 /*v335*/, v174 /*v430*/, v175 /*v431*/, v176 /*v432*/
	v_cndmask_b32_e32 v178 /*v434*/, 0xff800000, v92 /*v348*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v179 /*v435*/, v157 /*v413*/
	v_max3_num_f32 v92 /*v348*/, v162 /*v418*/, v163 /*v419*/, v102 /*v358*/
	v_cndmask_b32_e32 v179 /*v435*/, 0xff800000, v93 /*v349*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v84 /*v340*/, v157 /*v413*/
	v_max3_num_f32 v84 /*v340*/, v86 /*v342*/, v87 /*v343*/, v88 /*v344*/
	v_max3_num_f32 v93 /*v349*/, v103 /*v359*/, v104 /*v360*/, v105 /*v361*/
	v_max3_num_f32 v81 /*v337*/, v177 /*v433*/, v178 /*v434*/, v179 /*v435*/
	v_cndmask_b32_e32 v180 /*v436*/, 0xff800000, v94 /*v350*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v85 /*v341*/, v157 /*v413*/
	v_max3_num_f32 v94 /*v350*/, v164 /*v420*/, v165 /*v421*/, v166 /*v422*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v90 /*v346*/
	v_cndmask_b32_e32 v181 /*v437*/, 0xff800000, v95 /*v351*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v182 /*v438*/, v157 /*v413*/
	v_max3_num_f32 v95 /*v351*/, v167 /*v423*/, v118 /*v374*/, v119 /*v375*/
	v_max3_num_f32 v90 /*v346*/, v92 /*v348*/, v93 /*v349*/, v94 /*v350*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	v_cndmask_b32_e32 v182 /*v438*/, 0xff800000, v96 /*v352*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v183 /*v439*/, v157 /*v413*/
	v_max3_num_f32 v92 /*v348*/, v95 /*v351*/, v120 /*v376*/, v121 /*v377*/
	v_cndmask_b32_e32 v183 /*v439*/, 0xff800000, v97 /*v353*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v184 /*v440*/, v157 /*v413*/
	v_max3_num_f32 v85 /*v341*/, v180 /*v436*/, v181 /*v437*/, v182 /*v438*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v90 /*v346*/, v92 /*v348*/
	v_cndmask_b32_e32 v184 /*v440*/, 0xff800000, v106 /*v362*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v185 /*v441*/, v157 /*v413*/
	v_cndmask_b32_e32 v185 /*v441*/, 0xff800000, v107 /*v363*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v186 /*v442*/, v157 /*v413*/
	v_max3_num_f32 v91 /*v347*/, v183 /*v439*/, v184 /*v440*/, v185 /*v441*/
	v_cndmask_b32_e32 v186 /*v442*/, 0xff800000, v108 /*v364*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v98 /*v354*/, v157 /*v413*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v85 /*v341*/, v91 /*v347*/
	v_cndmask_b32_e32 v187 /*v443*/, 0xff800000, v109 /*v365*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v99 /*v355*/, v157 /*v413*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_cndmask_b32_e32 v188 /*v444*/, 0xff800000, v110 /*v366*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v100 /*v356*/, v157 /*v413*/
	v_cndmask_b32_e32 v189 /*v445*/, 0xff800000, v111 /*v367*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v101 /*v357*/, v157 /*v413*/
	v_max3_num_f32 v80 /*v336*/, v186 /*v442*/, v187 /*v443*/, v188 /*v444*/
	v_cndmask_b32_e32 v190 /*v446*/, 0xff800000, v112 /*v368*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v191 /*v447*/, v157 /*v413*/
	v_cndmask_b32_e32 v191 /*v447*/, 0xff800000, v113 /*v369*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v192 /*v448*/, v157 /*v413*/
	v_max3_num_f32 v84 /*v340*/, v189 /*v445*/, v190 /*v446*/, v191 /*v447*/
	v_cndmask_b32_e32 v192 /*v448*/, 0xff800000, v122 /*v378*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v193 /*v449*/, v157 /*v413*/
	v_cndmask_b32_e32 v193 /*v449*/, 0xff800000, v123 /*v379*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v194 /*v450*/, v157 /*v413*/
	v_cndmask_b32_e32 v194 /*v450*/, 0xff800000, v124 /*v380*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v114 /*v370*/, v157 /*v413*/
	v_cndmask_b32_e32 v195 /*v451*/, 0xff800000, v125 /*v381*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v115 /*v371*/, v157 /*v413*/
	v_max3_num_f32 v76 /*v332*/, v192 /*v448*/, v193 /*v449*/, v194 /*v450*/
	v_cndmask_b32_e32 v196 /*v452*/, 0xff800000, v126 /*v382*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v116 /*v372*/, v157 /*v413*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v76 /*v332*/
	v_cndmask_b32_e32 v197 /*v453*/, 0xff800000, v127 /*v383*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v117 /*v373*/, v157 /*v413*/
	v_max3_num_f32 v78 /*v334*/, v195 /*v451*/, v196 /*v452*/, v197 /*v453*/
	v_cndmask_b32_e32 v198 /*v454*/, 0xff800000, v128 /*v384*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v199 /*v455*/, v157 /*v413*/
	v_cndmask_b32_e32 v199 /*v455*/, 0xff800000, v129 /*v385*/, vcc_lo
	v_max3_num_f32 v78 /*v334*/, v78 /*v334*/, v198 /*v454*/, v199 /*v455*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_dual_mov_b32 v77 /*v333*/, v74 /*v330*/ :: v_dual_mov_b32 v76 /*v332*/, v75 /*v331*/
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s1, 0xfedcba98
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s1, 0xfedcba98
	v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v77 /*v333*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/
	v_dual_sub_f32 v77 /*v333*/, v74 /*v330*/, v153 /*v409*/ :: v_dual_max_num_f32 v74 /*v330*/, v153 /*v409*/, v74 /*v330*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v77 /*v333*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s0, -1, 0
	v_dual_sub_f32 v76 /*v332*/, v75 /*v331*/, v152 /*v408*/ :: v_dual_cndmask_b32 v133 /*v389*/, v74 /*v330*/, v153 /*v409*/, s0
	v_max_num_f32_e32 v74 /*v330*/, v152 /*v408*/, v75 /*v331*/
	v_cmp_lt_f32_e64 s0, 0x41000000, v76 /*v332*/
	v_mul_f32_e32 v124 /*v380*/, 0xbfb8aa3b, v133 /*v389*/
	s_cmp_lg_u32 s0, 0
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[72:73] /*v[328:329]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s17, -1, 0
	s_cmp_eq_u32 s0, 0
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[130:131] /*v[386:387]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s0, -1, 0
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v135 /*v391*/, v74 /*v330*/, v152 /*v408*/, s0
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v72 /*v328*/
	v_exp_f32_e32 v90 /*v346*/, v73 /*v329*/
	v_mul_f32_e32 v132 /*v388*/, 0xbfb8aa3b, v135 /*v391*/
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[86:87] /*v[342:343]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v80 /*v336*/
	v_exp_f32_e32 v106 /*v362*/, v81 /*v337*/
	v_nop
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[160:161] /*v[416:417]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[86:87] /*v[342:343]*/, v[162:163] /*v[418:419]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[168:169] /*v[424:425]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[170:171] /*v[426:427]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[162:163] /*v[418:419]*/, v[172:173] /*v[428:429]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v67 /*v323*/
	v_exp_f32_e32 v70 /*v326*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[82:83] /*v[338:339]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_exp_f32_e32 v67 /*v323*/, v130 /*v386*/
	v_exp_f32_e32 v69 /*v325*/, v131 /*v387*/
	v_exp_f32_e32 v71 /*v327*/, v160 /*v416*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[174:175] /*v[430:431]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v75 /*v331*/, v161 /*v417*/
	v_exp_f32_e32 v77 /*v333*/, v162 /*v418*/
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[176:177] /*v[432:433]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v163 /*v419*/
	v_nop
	v_pk_fma_f32 v[162:163] /*v[418:419]*/, v[178:179] /*v[434:435]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v78 /*v334*/
	v_exp_f32_e32 v98 /*v354*/, v79 /*v335*/
	v_nop
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[88:89] /*v[344:345]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v130 /*v386*/
	v_exp_f32_e32 v91 /*v347*/, v131 /*v387*/
	v_exp_f32_e32 v93 /*v349*/, v160 /*v416*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[180:181] /*v[436:437]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v161 /*v417*/
	v_exp_f32_e32 v101 /*v357*/, v162 /*v418*/
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[182:183] /*v[438:439]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v163 /*v419*/
	v_nop
	v_pk_fma_f32 v[162:163] /*v[418:419]*/, v[184:185] /*v[440:441]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v114 /*v370*/, v73 /*v329*/
	v_exp_f32_e32 v122 /*v378*/, v79 /*v335*/
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[102:103] /*v[358:359]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[104:105] /*v[360:361]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v109 /*v365*/, v130 /*v386*/
	v_exp_f32_e32 v115 /*v371*/, v131 /*v387*/
	v_exp_f32_e32 v117 /*v373*/, v160 /*v416*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[186:187] /*v[442:443]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v161 /*v417*/
	v_exp_f32_e32 v73 /*v329*/, v162 /*v418*/
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[188:189] /*v[444:445]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v79 /*v335*/, v163 /*v419*/
	v_nop
	v_pk_fma_f32 v[162:163] /*v[418:419]*/, v[190:191] /*v[446:447]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_exp_f32_e32 v76 /*v332*/, v76 /*v332*/
	v_exp_f32_e32 v108 /*v364*/, v72 /*v328*/
	v_exp_f32_e32 v116 /*v372*/, v78 /*v334*/
	v_exp_f32_e32 v72 /*v328*/, v80 /*v336*/
	v_exp_f32_e32 v78 /*v334*/, v81 /*v337*/
	v_exp_f32_e32 v80 /*v336*/, v86 /*v342*/
	v_exp_f32_e32 v86 /*v342*/, v87 /*v343*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[164:165] /*v[420:421]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v94 /*v350*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[166:167] /*v[422:423]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v102 /*v358*/, v97 /*v353*/
	v_exp_f32_e32 v81 /*v337*/, v130 /*v386*/
	v_exp_f32_e32 v87 /*v343*/, v131 /*v387*/
	v_exp_f32_e32 v89 /*v345*/, v160 /*v416*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[192:193] /*v[448:449]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v95 /*v351*/, v161 /*v417*/
	v_exp_f32_e32 v97 /*v353*/, v162 /*v418*/
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[194:195] /*v[450:451]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v103 /*v359*/, v163 /*v419*/
	v_nop
	v_pk_fma_f32 v[162:163] /*v[418:419]*/, v[196:197] /*v[452:453]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_pk_fma_f32 v[126:127] /*v[382:383]*/, v[118:119] /*v[374:375]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v105 /*v361*/
	v_pk_fma_f32 v[128:129] /*v[384:385]*/, v[120:121] /*v[376:377]*/, s[86:87], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v105 /*v361*/, v130 /*v386*/
	v_exp_f32_e32 v111 /*v367*/, v131 /*v387*/
	v_exp_f32_e32 v113 /*v369*/, v160 /*v416*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[198:199] /*v[454:455]*/, s[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v161 /*v417*/
	v_exp_f32_e32 v121 /*v377*/, v162 /*v418*/
	v_exp_f32_e32 v125 /*v381*/, v163 /*v419*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[74:75] /*v[330:331]*/, v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[98:99] /*v[354:355]*/, v[100:101] /*v[356:357]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[108:109] /*v[364:365]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v120 /*v376*/, v126 /*v382*/
	v_exp_f32_e32 v124 /*v380*/, v127 /*v383*/
	v_exp_f32_e32 v126 /*v382*/, v128 /*v384*/
	v_exp_f32_e32 v128 /*v384*/, v129 /*v385*/
	v_exp_f32_e32 v127 /*v383*/, v130 /*v386*/
	v_exp_f32_e32 v129 /*v385*/, v131 /*v387*/
	v_nop
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[70:71] /*v[326:327]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[82:83] /*v[338:339]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[122:123] /*v[378:379]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[106:107] /*v[362:363]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[116:117] /*v[372:373]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[92:93] /*v[348:349]*/, v[130:131] /*v[386:387]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[78:79] /*v[334:335]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[104:105] /*v[360:361]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[88:89] /*v[344:345]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[102:103] /*v[358:359]*/, v[172:173] /*v[428:429]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[164:165] /*v[420:421]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[112:113] /*v[368:369]*/, v[174:175] /*v[430:431]*/
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[170:171] /*v[426:427]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[168:169] /*v[424:425]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[124:125] /*v[380:381]*/, v[172:173] /*v[428:429]*/
	v_sub_f32_e32 v132 /*v388*/, v153 /*v409*/, v133 /*v389*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[166:167] /*v[422:423]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[168:169] /*v[424:425]*/, v[164:165] /*v[420:421]*/
	v_mul_f32_e32 v132 /*v388*/, 0x3fb8aa3b, v132 /*v388*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[160:161] /*v[416:417]*/, v[130:131] /*v[386:387]*/
	v_exp_f32_e32 v132 /*v388*/, v132 /*v388*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[162:163] /*v[418:419]*/, v[130:131] /*v[386:387]*/
	v_dual_mov_b32 v153 /*v409*/, v130 /*v386*/ :: v_dual_mov_b32 v155 /*v411*/, v131 /*v387*/
	v_permlanex16_b32 v153 /*v409*/, v153 /*v409*/, s1, 0xfedcba98
	v_permlanex16_b32 v155 /*v411*/, v155 /*v411*/, s1, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_24
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_24:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v134 /*v390*/, v152 /*v408*/, v135 /*v391*/
	s_and_b32 s0, s17, exec_lo
	s_cselect_b32 s0, 1, 0
	s_cmp_lg_u32 s0, 1
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_19
	v_nop
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[62:63], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_19
.LBB0_26:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v135 /*v391*/, v152 /*v408*/ :: v_dual_mov_b32 v133 /*v389*/, v153 /*v409*/
	s_set_vgpr_msb 0x4100
.LBB0_27:
	s_branch .LBB0_48
.LBB0_28:
	s_mov_b32 s23, 0
	s_cmp_lg_u32 s10, 0x80000000
	s_mov_b32 s21, s23
	s_mov_b32 s22, s23
	s_mov_b32 s0, s23
	s_mov_b32 s1, s23
	s_mov_b32 s2, s23
	s_mov_b32 s3, s23
	s_mov_b32 s9, s23
	tensor_load_to_lds s[4:7], s[24:31], s[20:23], s[0:3]
	s_cselect_b32 s1, s69, 0
	s_cselect_b32 s21, s68, 0x80
	s_and_b32 s2, s96, 3
	s_mov_b32 s0, s21
	s_lshl_b32 s45, s2, 4
	s_lshl_b32 s8, s2, 5
	s_mul_i32 s46, s2, 0x1100
	s_mul_i32 s47, s2, 0x1200
	s_mul_u64 s[2:3], s[0:1], s[8:9]
	s_sub_co_i32 s0, s97, s45
	s_and_b32 s22, s1, 0xffff
	s_max_i32 s0, s0, 0
	s_add_nc_u64 s[6:7], s[2:3], s[80:81]
	s_lshl_b32 s0, s0, 16
	s_mov_b32 s4, 1
	s_or_b32 s18, s0, 0x7fff
	s_cmp_lg_u32 s11, 0x80000000
	s_mov_b32 s19, 0x800000
	s_cselect_b32 s29, s70, 0x80
	s_cselect_b32 s1, s71, 0
	s_mov_b32 s0, s29
	s_mov_b32 s17, 0xffff0000
	s_mov_b32 s16, 0x7510000
	s_mov_b32 s20, 16
	s_mov_b32 s5, s46
	s_bitset1_b32 s7, 31
	s_mul_u64 s[8:9], s[0:1], s[8:9]
	s_bitset1_b32 s47, 16
	s_mov_b32 s24, 0xf510000
	s_mov_b32 s25, s17
	s_mov_b32 s27, s19
	s_mov_b32 s31, s23
	s_mov_b32 s28, s20
	s_mov_b32 s26, s18
	s_and_b32 s30, s1, 0xffff
	s_ashr_i32 s0, s95, 2
	s_add_co_i32 s48, s63, -1
	s_set_vgpr_msb 17
	v_and_or_b32 v64, v142 /*v398*/, 7, v145 /*v401*/
	s_add_co_i32 s0, s48, s0
	s_add_nc_u64 s[10:11], s[74:75], s[14:15]
	s_max_i32 s0, s0, 0
	s_add_nc_u64 s[14:15], s[72:73], s[76:77]
	s_add_co_i32 s0, s0, 1
	v_mul_u32_u24_e32 v64, 0x120, v64
	s_ashr_i32 s1, s0, 31
	s_mov_b32 s37, s23
	s_lshr_b32 s1, s1, 26
	s_add_co_i32 s1, s0, s1
	s_set_vgpr_msb 0x1101
	v_and_or_b32 v64, v148 /*v404*/, 16, v64
	s_set_vgpr_msb 0x140
	v_or_b32_e32 v148 /*v404*/, 0x10000, v64
	v_or_b32_e32 v151 /*v407*/, 0x30000, v64
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4001
	ds_load_b128 v[0:3], v149 /*v405*/
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[4:7], v149 /*v405*/ offset:32
	ds_load_b128 v[8:11], v149 /*v405*/ offset:64
	ds_load_b128 v[12:15], v149 /*v405*/ offset:96
	ds_load_b128 v[16:19], v149 /*v405*/ offset:128
	ds_load_b128 v[20:23], v149 /*v405*/ offset:160
	ds_load_b128 v[24:27], v149 /*v405*/ offset:192
	ds_load_b128 v[28:31], v149 /*v405*/ offset:224
	ds_load_b128 v[32:35], v149 /*v405*/ offset:4352
	ds_load_b128 v[36:39], v149 /*v405*/ offset:4384
	ds_load_b128 v[40:43], v149 /*v405*/ offset:4416
	ds_load_b128 v[44:47], v149 /*v405*/ offset:4448
	ds_load_b128 v[48:51], v149 /*v405*/ offset:4480
	ds_load_b128 v[52:55], v149 /*v405*/ offset:4512
	ds_load_b128 v[56:59], v149 /*v405*/ offset:4544
	ds_load_b128 v[60:63], v149 /*v405*/ offset:4576
	tensor_load_to_lds s[4:7], s[16:23]
	s_add_nc_u64 s[6:7], s[8:9], s[78:79]
	s_mov_b32 s5, s47
	s_bitset1_b32 s7, 31
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v135, v147 /*v403*/, v7
	tensor_load_to_lds s[4:7], s[24:31]
	s_and_b32 s6, s1, 0xffffffc0
	s_add_co_i32 s5, s66, -1
	s_ashr_i32 s1, s1, 6
	s_cmp_lg_u32 s0, s6
	v_pk_mul_bf16 v134, v147 /*v403*/, v6
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s0, 0
	v_pk_mul_bf16 v133, v147 /*v403*/, v5
	s_cselect_b32 s0, -1, 0
	v_pk_mul_bf16 v132, v147 /*v403*/, v4
	s_and_b32 s0, s0, s6
	s_sub_co_ci_u32 s0, s1, 0
	v_pk_mul_bf16 v131, v147 /*v403*/, v3
	s_min_i32 s0, s0, s5
	v_pk_mul_bf16 v130, v147 /*v403*/, v2
	v_pk_mul_bf16 v129, v147 /*v403*/, v1
	v_pk_mul_bf16 v128, v147 /*v403*/, v0
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v147 /*v403*/, v15
	v_pk_mul_bf16 v142, v147 /*v403*/, v14
	v_pk_mul_bf16 v141, v147 /*v403*/, v13
	v_pk_mul_bf16 v140, v147 /*v403*/, v12
	v_pk_mul_bf16 v139, v147 /*v403*/, v11
	v_pk_mul_bf16 v138, v147 /*v403*/, v10
	v_pk_mul_bf16 v137, v147 /*v403*/, v9
	v_pk_mul_bf16 v136, v147 /*v403*/, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v147 /*v403*/, v23
	v_pk_mul_bf16 v150, v147 /*v403*/, v22
	v_pk_mul_bf16 v149, v147 /*v403*/, v21
	v_pk_mul_bf16 v148, v147 /*v403*/, v20
	v_pk_mul_bf16 v147, v147 /*v403*/, v19
	v_pk_mul_bf16 v146, v147 /*v403*/, v18
	v_pk_mul_bf16 v145, v147 /*v403*/, v17
	v_pk_mul_bf16 v144, v147 /*v403*/, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v147 /*v403*/, v31
	v_pk_mul_bf16 v158, v147 /*v403*/, v30
	v_pk_mul_bf16 v157, v147 /*v403*/, v29
	v_pk_mul_bf16 v156, v147 /*v403*/, v28
	v_pk_mul_bf16 v155, v147 /*v403*/, v27
	v_pk_mul_bf16 v154, v147 /*v403*/, v26
	v_pk_mul_bf16 v153, v147 /*v403*/, v25
	v_pk_mul_bf16 v152, v147 /*v403*/, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v147 /*v403*/, v39
	v_pk_mul_bf16 v166, v147 /*v403*/, v38
	v_pk_mul_bf16 v165, v147 /*v403*/, v37
	v_pk_mul_bf16 v164, v147 /*v403*/, v36
	v_pk_mul_bf16 v163, v147 /*v403*/, v35
	v_pk_mul_bf16 v162, v147 /*v403*/, v34
	v_pk_mul_bf16 v161, v147 /*v403*/, v33
	v_pk_mul_bf16 v160, v147 /*v403*/, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v147 /*v403*/, v47
	v_pk_mul_bf16 v174, v147 /*v403*/, v46
	v_pk_mul_bf16 v173, v147 /*v403*/, v45
	v_pk_mul_bf16 v172, v147 /*v403*/, v44
	v_pk_mul_bf16 v171, v147 /*v403*/, v43
	v_pk_mul_bf16 v170, v147 /*v403*/, v42
	v_pk_mul_bf16 v169, v147 /*v403*/, v41
	v_pk_mul_bf16 v168, v147 /*v403*/, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v147 /*v403*/, v55
	v_pk_mul_bf16 v182, v147 /*v403*/, v54
	v_pk_mul_bf16 v181, v147 /*v403*/, v53
	v_pk_mul_bf16 v180, v147 /*v403*/, v52
	v_pk_mul_bf16 v179, v147 /*v403*/, v51
	v_pk_mul_bf16 v178, v147 /*v403*/, v50
	v_pk_mul_bf16 v177, v147 /*v403*/, v49
	v_pk_mul_bf16 v176, v147 /*v403*/, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v147 /*v403*/, v63
	v_pk_mul_bf16 v190, v147 /*v403*/, v62
	v_pk_mul_bf16 v189, v147 /*v403*/, v61
	v_pk_mul_bf16 v188, v147 /*v403*/, v60
	v_pk_mul_bf16 v187, v147 /*v403*/, v59
	v_pk_mul_bf16 v186, v147 /*v403*/, v58
	v_pk_mul_bf16 v185, v147 /*v403*/, v57
	v_pk_mul_bf16 v184, v147 /*v403*/, v56
	v_mov_b32_e32 v0, 0
	s_max_i32 s36, s0, 0
	s_cmp_lt_i32 s0, 1
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x100
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_37
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[30:31], v[6:7]
	v_mov_b64_e32 v[28:29], v[4:5]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[46:47], v[6:7]
	v_mov_b64_e32 v[44:45], v[4:5]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[40:41], v[0:1]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[62:63], v[6:7]
	v_mov_b64_e32 v[60:61], v[4:5]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[56:57], v[0:1]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[64:65], v[0:1]
	v_mov_b64_e32 v[78:79], v[6:7]
	v_mov_b64_e32 v[76:77], v[4:5]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[72:73], v[0:1]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[88:89], v[0:1]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[96:97], v[0:1]
	v_mov_b64_e32 v[110:111], v[6:7]
	v_mov_b64_e32 v[108:109], v[4:5]
	v_mov_b64_e32 v[106:107], v[2:3]
	v_mov_b64_e32 v[104:105], v[0:1]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
	s_set_vgpr_msb 64
	v_dual_mov_b32 v135 /*v391*/, 0xf149f2ca :: v_dual_mov_b32 v130 /*v386*/, 0xf149f2ca
	s_set_vgpr_msb 0x4001
	v_dual_mov_b32 v200, v151 /*v407*/ :: v_dual_mov_b32 v201, v146 /*v402*/
	s_set_vgpr_msb 0x141
	v_mov_b32_e32 v147 /*v403*/, v144 /*v400*/
	s_set_vgpr_msb 0x4140
	v_dual_mov_b32 v64 /*v320*/, v0 :: v_dual_mov_b32 v65 /*v321*/, v0
	s_lshl_b64 s[38:39], s[36:37], 6
	s_sub_co_i32 s49, s13, 64
	s_add_co_i32 s40, s64, 64
	s_mov_b64 s[42:43], 0xffffffffffffffc0
	s_mov_b32 s50, 0x76543210
	s_mov_b32 s44, 0x3fb8aa3b
	s_mov_b32 s51, 1
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_31
.LBB0_30:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v159 /*v415*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v110 /*v366*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v102 /*v358*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v94 /*v350*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v155 /*v411*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v154 /*v410*/, v78 /*v334*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v153 /*v409*/, v72 /*v328*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v152 /*v408*/, v68 /*v324*/, v70 /*v326*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v111 /*v367*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v103 /*v359*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v95 /*v351*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v79 /*v335*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v73 /*v329*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v69 /*v325*/, v71 /*v327*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[152:159] /*v[408:415]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v75 /*v331*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v74 /*v330*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v73 /*v329*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v72 /*v328*/, v105 /*v361*/, v109 /*v365*/
	v_cvt_pk_bf16_f32 v71 /*v327*/, v97 /*v353*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v70 /*v326*/, v89 /*v345*/, v93 /*v349*/
	v_cvt_pk_bf16_f32 v69 /*v325*/, v81 /*v337*/, v87 /*v343*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[160:167] /*v[416:423]*/, v[56:63]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v68 /*v324*/, v67 /*v323*/, v77 /*v333*/
	s_add_nc_u64 s[38:39], s[38:39], s[42:43]
	s_sub_co_i32 s49, s49, 64
	s_add_co_i32 s40, s40, 64
	s_add_co_i32 s51, s51, 1
	s_cmp_lg_u64 s[38:39], 0
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[152:159] /*v[408:415]*/, v[112:119]
	v_nop
	v_nop
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v108 /*v364*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v100 /*v356*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[160:167] /*v[416:423]*/, v[48:55]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v92 /*v348*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v66 /*v322*/, v76 /*v332*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[152:159] /*v[408:415]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[160:167] /*v[416:423]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[152:159] /*v[408:415]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[160:167] /*v[416:423]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[152:159] /*v[408:415]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[160:167] /*v[416:423]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[152:159] /*v[408:415]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[160:167] /*v[416:423]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[152:159] /*v[408:415]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[160:167] /*v[416:423]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[152:159] /*v[408:415]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[160:167] /*v[416:423]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x415
	v_pk_fma_f32 v[200:201], v[134:135] /*v[390:391]*/, v[64:65] /*v[320:321]*/, v[132:133] /*v[388:389]*/
	s_set_vgpr_msb 0x1541
	v_mov_b32_e32 v135 /*v391*/, v149 /*v405*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[130:131] /*v[386:387]*/, v[200:201]
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_dual_mov_b32 v200, v151 /*v407*/ :: v_dual_mov_b32 v201, v146 /*v402*/
	s_set_vgpr_msb 0x541
	v_mov_b32_e32 v130 /*v386*/, v150 /*v406*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[68:75] /*v[324:331]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[68:75] /*v[324:331]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[68:75] /*v[324:331]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[68:75] /*v[324:331]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[68:75] /*v[324:331]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[68:75] /*v[324:331]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[68:75] /*v[324:331]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[68:75] /*v[324:331]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_scc0 .LBB0_38
.LBB0_31:
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v146 /*v402*/, v147 /*v403*/ :: v_dual_mov_b32 v151 /*v407*/, v148 /*v404*/
	s_set_vgpr_msb 0x4140
	v_dual_mov_b32 v147 /*v403*/, v201 :: v_dual_mov_b32 v148 /*v404*/, v200
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4041
	ds_load_b128 v[56:59] /*v[312:315]*/, v146 /*v402*/
	ds_load_b128 v[60:63] /*v[316:319]*/, v146 /*v402*/ offset:32
	ds_load_b128 v[48:51] /*v[304:307]*/, v146 /*v402*/ offset:64
	ds_load_b128 v[52:55] /*v[308:311]*/, v146 /*v402*/ offset:96
	ds_load_b128 v[40:43] /*v[296:299]*/, v146 /*v402*/ offset:128
	ds_load_b128 v[44:47] /*v[300:303]*/, v146 /*v402*/ offset:160
	ds_load_b128 v[32:35] /*v[288:291]*/, v146 /*v402*/ offset:192
	ds_load_b128 v[36:39] /*v[292:295]*/, v146 /*v402*/ offset:224
	ds_load_b128 v[24:27] /*v[280:283]*/, v146 /*v402*/ offset:4352
	ds_load_b128 v[28:31] /*v[284:287]*/, v146 /*v402*/ offset:4384
	ds_load_b128 v[16:19] /*v[272:275]*/, v146 /*v402*/ offset:4416
	ds_load_b128 v[20:23] /*v[276:279]*/, v146 /*v402*/ offset:4448
	ds_load_b128 v[8:11] /*v[264:267]*/, v146 /*v402*/ offset:4480
	ds_load_b128 v[12:15] /*v[268:271]*/, v146 /*v402*/ offset:4512
	ds_load_b128 v[0:3] /*v[256:259]*/, v146 /*v402*/ offset:4544
	ds_load_b128 v[4:7] /*v[260:263]*/, v146 /*v402*/ offset:4576
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[248:251], v146 /*v402*/ offset:8704
	ds_load_b128 v[252:255], v146 /*v402*/ offset:8736
	ds_load_b128 v[240:243], v146 /*v402*/ offset:8768
	ds_load_b128 v[244:247], v146 /*v402*/ offset:8800
	ds_load_b128 v[232:235], v146 /*v402*/ offset:8832
	ds_load_b128 v[236:239], v146 /*v402*/ offset:8864
	ds_load_b128 v[224:227], v146 /*v402*/ offset:8896
	ds_load_b128 v[228:231], v146 /*v402*/ offset:8928
	ds_load_b128 v[216:219], v146 /*v402*/ offset:13056
	ds_load_b128 v[220:223], v146 /*v402*/ offset:13088
	ds_load_b128 v[208:211], v146 /*v402*/ offset:13120
	ds_load_b128 v[212:215], v146 /*v402*/ offset:13152
	ds_load_b128 v[200:203], v146 /*v402*/ offset:13184
	ds_load_b128 v[204:207], v146 /*v402*/ offset:13216
	ds_load_b128 v[192:195], v146 /*v402*/ offset:13248
	ds_load_b128 v[196:199], v146 /*v402*/ offset:13280
	s_cmp_ge_i32 s51, s66
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_33
	s_set_vgpr_msb 0x41
	v_med3_i32 v66 /*v322*/, s49, 0, 64
	s_lshr_b32 s5, s51, 31
	s_ashr_i32 s41, s40, 31
	s_add_co_i32 s5, s51, s5
	s_mul_u64 s[6:7], s[40:41], s[68:69]
	v_readfirstlane_b32 s18, v66 /*v322*/
	s_and_b32 s5, s5, 0x7ffe
	s_lshl_b64 s[6:7], s[6:7], 1
	s_sub_co_i32 s5, s51, s5
	s_mul_u64 s[0:1], s[40:41], s[70:71]
	s_lshl_b32 s25, s5, 17
	s_sub_co_i32 s5, s18, s45
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_max_i32 s18, s5, 0
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_nc_u64 s[6:7], s[2:3], s[6:7]
	s_lshl_b32 s18, s18, 16
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	s_or_b32 s5, s46, s25
	s_bitset1_b32 s7, 31
	s_addk_co_i32 s18, 0x7fff
	s_mov_b32 s27, s19
	tensor_load_to_lds s[4:7], s[16:23]
	s_add_nc_u64 s[6:7], s[8:9], s[0:1]
	s_or_b32 s5, s47, s25
	s_bitset1_b32 s7, 31
	s_mov_b32 s25, s17
	s_mov_b32 s26, s18
	s_mov_b32 s28, s20
	s_mov_b32 s31, s23
	tensor_load_to_lds s[4:7], s[24:31]
	s_set_vgpr_msb 0x4100
.LBB0_33:
	s_set_vgpr_msb 0x41
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	s_set_vgpr_msb 0x4140
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[248:255], v[128:135], 0
	s_set_vgpr_msb 0x4051
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[48:55] /*v[304:311]*/, v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[16:23] /*v[272:279]*/, v[136:143], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[248:255], v[160:167], 0
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[240:247], v[136:143], v[120:127] /*v[376:383]*/
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[48:55] /*v[304:311]*/, v[168:175], v[152:159] /*v[408:415]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[40:47] /*v[296:303]*/, v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[16:23] /*v[272:279]*/, v[168:175], v[160:167] /*v[416:423]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[8:15] /*v[264:271]*/, v[144:151], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[240:247], v[168:175], v[168:175] /*v[424:431]*/
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[232:239], v[144:151], v[120:127] /*v[376:383]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[208:215], v[136:143], v[176:183] /*v[432:439]*/
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[208:215], v[168:175], v[184:191] /*v[440:447]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[40:47] /*v[296:303]*/, v[176:183], v[152:159] /*v[408:415]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[32:39] /*v[288:295]*/, v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[8:15] /*v[264:271]*/, v[176:183], v[160:167] /*v[416:423]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[0:7] /*v[256:263]*/, v[152:159], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[232:239], v[176:183], v[168:175] /*v[424:431]*/
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[224:231], v[152:159], v[120:127] /*v[376:383]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[200:207], v[144:151], v[176:183] /*v[432:439]*/
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[200:207], v[176:183], v[184:191] /*v[440:447]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[32:39] /*v[288:295]*/, v[184:191], v[152:159] /*v[408:415]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[0:7] /*v[256:263]*/, v[184:191], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[224:231], v[184:191], v[168:175] /*v[424:431]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[192:199], v[152:159], v[176:183] /*v[432:439]*/
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[192:199], v[184:191], v[184:191] /*v[440:447]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5041
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v151 /*v407*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v151 /*v407*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v151 /*v407*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v151 /*v407*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v151 /*v407*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v151 /*v407*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v151 /*v407*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v151 /*v407*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v151 /*v407*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v151 /*v407*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v151 /*v407*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v151 /*v407*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v151 /*v407*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v151 /*v407*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v151 /*v407*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v151 /*v407*/ offset:13920
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[248:251], v151 /*v407*/ offset:128
	ds_load_tr16_b128 v[232:235], v151 /*v407*/ offset:160
	ds_load_tr16_b128 v[252:255], v151 /*v407*/ offset:4736
	ds_load_tr16_b128 v[236:239], v151 /*v407*/ offset:4768
	ds_load_tr16_b128 v[240:243], v151 /*v407*/ offset:9344
	ds_load_tr16_b128 v[224:227], v151 /*v407*/ offset:9376
	ds_load_tr16_b128 v[244:247], v151 /*v407*/ offset:13952
	ds_load_tr16_b128 v[228:231], v151 /*v407*/ offset:13984
	ds_load_tr16_b128 v[216:219], v151 /*v407*/ offset:192
	ds_load_tr16_b128 v[200:203], v151 /*v407*/ offset:224
	ds_load_tr16_b128 v[220:223], v151 /*v407*/ offset:4800
	ds_load_tr16_b128 v[204:207], v151 /*v407*/ offset:4832
	ds_load_tr16_b128 v[208:211], v151 /*v407*/ offset:9408
	ds_load_tr16_b128 v[192:195], v151 /*v407*/ offset:9440
	ds_load_tr16_b128 v[212:215], v151 /*v407*/ offset:14016
	ds_load_tr16_b128 v[196:199], v151 /*v407*/ offset:14048
	s_set_vgpr_msb 0x155
	v_dual_max_num_f32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/ :: v_dual_max_num_f32 v75 /*v331*/, v152 /*v408*/, v153 /*v409*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v80 /*v336*/, v95 /*v351*/, v96 /*v352*/, v97 /*v353*/
	v_max3_num_f32 v82 /*v338*/, v98 /*v354*/, v99 /*v355*/, v100 /*v356*/
	v_max3_num_f32 v84 /*v340*/, v101 /*v357*/, v120 /*v376*/, v121 /*v377*/
	v_max3_num_f32 v77 /*v333*/, v155 /*v411*/, v156 /*v412*/, v157 /*v413*/
	v_max3_num_f32 v81 /*v337*/, v161 /*v417*/, v162 /*v418*/, v163 /*v419*/
	v_max3_num_f32 v83 /*v339*/, v164 /*v420*/, v165 /*v421*/, v166 /*v422*/
	v_max3_num_f32 v85 /*v341*/, v167 /*v423*/, v168 /*v424*/, v169 /*v425*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v94 /*v350*/
	v_max3_num_f32 v86 /*v342*/, v122 /*v378*/, v123 /*v379*/, v124 /*v380*/
	v_max3_num_f32 v88 /*v344*/, v125 /*v381*/, v126 /*v382*/, v127 /*v383*/
	v_max3_num_f32 v90 /*v346*/, v176 /*v432*/, v177 /*v433*/, v178 /*v434*/
	v_max3_num_f32 v92 /*v348*/, v179 /*v435*/, v180 /*v436*/, v181 /*v437*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v82 /*v338*/, v84 /*v340*/
	v_max3_num_f32 v79 /*v335*/, v158 /*v414*/, v159 /*v415*/, v160 /*v416*/
	v_max3_num_f32 v87 /*v343*/, v170 /*v426*/, v171 /*v427*/, v172 /*v428*/
	v_max3_num_f32 v89 /*v345*/, v173 /*v429*/, v174 /*v430*/, v175 /*v431*/
	v_max3_num_f32 v91 /*v347*/, v184 /*v440*/, v185 /*v441*/, v186 /*v442*/
	v_max3_num_f32 v93 /*v349*/, v187 /*v443*/, v188 /*v444*/, v189 /*v445*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v154 /*v410*/, v77 /*v333*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v83 /*v339*/, v85 /*v341*/
	v_max3_num_f32 v80 /*v336*/, v86 /*v342*/, v88 /*v344*/, v90 /*v346*/
	v_max3_num_f32 v81 /*v337*/, v92 /*v348*/, v182 /*v438*/, v183 /*v439*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	v_max3_num_f32 v76 /*v332*/, v87 /*v343*/, v89 /*v345*/, v91 /*v347*/
	v_max3_num_f32 v78 /*v334*/, v93 /*v349*/, v190 /*v446*/, v191 /*v447*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v80 /*v336*/, v81 /*v337*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_dual_mov_b32 v76 /*v332*/, v74 /*v330*/ :: v_dual_mov_b32 v77 /*v333*/, v75 /*v331*/
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s50, 0xfedcba98
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s50, 0xfedcba98
	v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v76 /*v332*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v77 /*v333*/
	v_sub_f32_e32 v76 /*v332*/, v74 /*v330*/, v130 /*v386*/
	v_dual_max_num_f32 v74 /*v330*/, v130 /*v386*/, v74 /*v330*/ :: v_dual_sub_f32 v77 /*v333*/, v75 /*v331*/, v135 /*v391*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v76 /*v332*/
	v_cmp_lt_f32_e64 s0, 0x41000000, v77 /*v333*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s1, -1, 0
	s_cmp_lg_u32 s0, 0
	v_dual_cndmask_b32 v150 /*v406*/, v74 /*v330*/, v130 /*v386*/, s1 :: v_dual_max_num_f32 v74 /*v330*/, v135 /*v391*/, v75 /*v331*/
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u32 s0, 0
	s_cselect_b32 s0, -1, 0
	v_mul_f32_e32 v118 /*v374*/, 0xbfb8aa3b, v150 /*v406*/
	s_wait_alu depctr_vm_vsrc(6)
	v_cndmask_b32_e64 v149 /*v405*/, v74 /*v330*/, v135 /*v391*/, s0
	v_sub_f32_e32 v134 /*v390*/, v130 /*v386*/, v150 /*v406*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_mul_f32_e32 v132 /*v388*/, 0xbfb8aa3b, v149 /*v405*/
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[72:73] /*v[328:329]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v66 /*v322*/
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v70 /*v326*/, v67 /*v323*/
	v_exp_f32_e32 v72 /*v328*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_exp_f32_e32 v78 /*v334*/, v76 /*v332*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[94:95] /*v[350:351]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_nop
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[96:97] /*v[352:353]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v69 /*v325*/, v152 /*v408*/
	v_exp_f32_e32 v71 /*v327*/, v153 /*v409*/
	v_exp_f32_e32 v73 /*v329*/, v154 /*v410*/
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[158:159] /*v[414:415]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v75 /*v331*/, v155 /*v411*/
	v_exp_f32_e32 v79 /*v335*/, v156 /*v412*/
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[160:161] /*v[416:417]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v157 /*v413*/
	v_nop
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[162:163] /*v[418:419]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v80 /*v336*/
	v_exp_f32_e32 v90 /*v346*/, v81 /*v337*/
	v_exp_f32_e32 v94 /*v350*/, v66 /*v322*/
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[98:99] /*v[354:355]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v98 /*v354*/, v67 /*v323*/
	v_exp_f32_e32 v102 /*v358*/, v76 /*v332*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[100:101] /*v[356:357]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v106 /*v362*/, v77 /*v333*/
	v_nop
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[120:121] /*v[376:377]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v152 /*v408*/
	v_exp_f32_e32 v91 /*v347*/, v153 /*v409*/
	v_exp_f32_e32 v95 /*v351*/, v154 /*v410*/
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[164:165] /*v[420:421]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v155 /*v411*/
	v_exp_f32_e32 v103 /*v359*/, v156 /*v412*/
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[166:167] /*v[422:423]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v157 /*v413*/
	v_nop
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[168:169] /*v[424:425]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v80 /*v336*/
	v_exp_f32_e32 v114 /*v370*/, v81 /*v337*/
	v_exp_f32_e32 v116 /*v372*/, v66 /*v322*/
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[122:123] /*v[378:379]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v122 /*v378*/, v67 /*v323*/
	v_exp_f32_e32 v66 /*v322*/, v76 /*v332*/
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[124:125] /*v[380:381]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v76 /*v332*/, v77 /*v333*/
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[126:127] /*v[382:383]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v111 /*v367*/, v152 /*v408*/
	v_exp_f32_e32 v115 /*v371*/, v153 /*v409*/
	v_exp_f32_e32 v117 /*v373*/, v154 /*v410*/
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[170:171] /*v[426:427]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v155 /*v411*/
	v_exp_f32_e32 v67 /*v323*/, v156 /*v412*/
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[172:173] /*v[428:429]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v77 /*v333*/, v157 /*v413*/
	v_nop
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[174:175] /*v[430:431]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v86 /*v342*/, v81 /*v337*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[176:177] /*v[432:433]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[178:179] /*v[434:435]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v97 /*v353*/
	v_pk_fma_f32 v[120:121] /*v[376:377]*/, v[180:181] /*v[436:437]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v81 /*v337*/, v152 /*v408*/
	v_exp_f32_e32 v87 /*v343*/, v153 /*v409*/
	v_exp_f32_e32 v89 /*v345*/, v154 /*v410*/
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[184:185] /*v[440:441]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v93 /*v349*/, v155 /*v411*/
	v_exp_f32_e32 v97 /*v353*/, v156 /*v412*/
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[186:187] /*v[442:443]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v101 /*v357*/, v157 /*v413*/
	v_nop
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[188:189] /*v[444:445]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v80 /*v336*/, v80 /*v336*/
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_exp_f32_e32 v108 /*v364*/, v105 /*v361*/
	v_pk_fma_f32 v[126:127] /*v[382:383]*/, v[182:183] /*v[438:439]*/, s[44:45], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v124 /*v380*/, v121 /*v377*/
	v_exp_f32_e32 v105 /*v361*/, v152 /*v408*/
	v_exp_f32_e32 v109 /*v365*/, v153 /*v409*/
	v_exp_f32_e32 v113 /*v369*/, v154 /*v410*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[190:191] /*v[446:447]*/, s[44:45], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v155 /*v411*/
	v_exp_f32_e32 v121 /*v377*/, v156 /*v412*/
	v_exp_f32_e32 v125 /*v381*/, v157 /*v413*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[68:69] /*v[324:325]*/, v[70:71] /*v[326:327]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[74:75] /*v[330:331]*/, v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[98:99] /*v[354:355]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[110:111] /*v[366:367]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v128 /*v384*/, v127 /*v383*/
	v_exp_f32_e32 v127 /*v383*/, v132 /*v388*/
	v_exp_f32_e32 v129 /*v385*/, v133 /*v389*/
	v_nop
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[72:73] /*v[328:329]*/, v[152:153] /*v[408:409]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[82:83] /*v[338:339]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[122:123] /*v[378:379]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[106:107] /*v[362:363]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[116:117] /*v[372:373]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[92:93] /*v[348:349]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_exp_f32_e32 v120 /*v376*/, v120 /*v376*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[94:95] /*v[350:351]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[76:77] /*v[332:333]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[104:105] /*v[360:361]*/, v[108:109] /*v[364:365]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[88:89] /*v[344:345]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[100:101] /*v[356:357]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[158:159] /*v[414:415]*/
	v_exp_f32_e32 v126 /*v382*/, v126 /*v382*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[112:113] /*v[368:369]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[152:153] /*v[408:409]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[162:163] /*v[418:419]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[160:161] /*v[416:417]*/, v[156:157] /*v[412:413]*/
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[124:125] /*v[380:381]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[158:159] /*v[414:415]*/, v[152:153] /*v[408:409]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[154:155] /*v[410:411]*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[160:161] /*v[416:417]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[152:153] /*v[408:409]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[154:155] /*v[410:411]*/, v[132:133] /*v[388:389]*/
	v_dual_mov_b32 v132 /*v388*/, v130 /*v386*/ :: v_dual_mov_b32 v133 /*v389*/, v131 /*v387*/
	v_permlanex16_b32 v132 /*v388*/, v132 /*v388*/, s50, 0xfedcba98
	v_permlanex16_b32 v133 /*v389*/, v133 /*v389*/, s50, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_35
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_35:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v135 /*v391*/, v135 /*v391*/, v149 /*v405*/
	s_and_b32 s0, s1, exec_lo
	s_cselect_b32 s0, 1, 0
	s_cmp_lg_u32 s0, 1
	v_mul_f32_e32 v135 /*v391*/, 0x3fb8aa3b, v135 /*v391*/
	v_exp_f32_e32 v135 /*v391*/, v135 /*v391*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_30
	v_nop
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v152 /*v408*/, v135 /*v391*/
	s_set_vgpr_msb 0x4104
	v_pk_mul_f32 v[62:63], v[62:63], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_30
.LBB0_37:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 64
	v_dual_mov_b32 v149 /*v405*/, 0xf149f2ca :: v_dual_mov_b32 v64 /*v320*/, v0
	v_dual_mov_b32 v65 /*v321*/, v0 :: v_dual_mov_b32 v150 /*v406*/, 0xf149f2ca
	s_set_vgpr_msb 0x4041
	v_mov_b32_e32 v147 /*v403*/, v144 /*v400*/
	s_set_vgpr_msb 0x4100
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[30:31], v[6:7]
	v_mov_b64_e32 v[28:29], v[4:5]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[46:47], v[6:7]
	v_mov_b64_e32 v[44:45], v[4:5]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[40:41], v[0:1]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[62:63], v[6:7]
	v_mov_b64_e32 v[60:61], v[4:5]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[56:57], v[0:1]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[64:65], v[0:1]
	v_mov_b64_e32 v[78:79], v[6:7]
	v_mov_b64_e32 v[76:77], v[4:5]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[72:73], v[0:1]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[88:89], v[0:1]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[96:97], v[0:1]
	v_mov_b64_e32 v[110:111], v[6:7]
	v_mov_b64_e32 v[108:109], v[4:5]
	v_mov_b64_e32 v[106:107], v[2:3]
	v_mov_b64_e32 v[104:105], v[0:1]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
.LBB0_38:
	s_cmp_ge_u32 s36, s66
	s_cbranch_scc1 .LBB0_47
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4
	v_dual_add_nc_u32 v192, s48, v138 /*v394*/ :: v_dual_add_nc_u32 v193, s48, v139 /*v395*/
	s_add_co_i32 s94, s94, -1
	s_mov_b32 s23, 0
	s_mov_b32 s4, 1
	s_set_vgpr_msb 0x440
	v_min_i32_e32 v152 /*v408*/, s94, v192
	v_min_i32_e32 v153 /*v409*/, s94, v193
	s_mov_b32 s67, s23
	s_mov_b32 s20, 16
	s_mov_b32 s19, 0x800000
	s_mov_b32 s17, 0xffff0000
	s_mov_b32 s16, 0x7510000
	s_mov_b32 s24, 0xf510000
	s_mov_b32 s1, 0x76543210
	s_mov_b32 s38, 0x3fb8aa3b
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_41
.LBB0_40:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v163 /*v419*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v108 /*v364*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v100 /*v356*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v92 /*v348*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v76 /*v332*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v70 /*v326*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v66 /*v322*/, v68 /*v324*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v109 /*v365*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v101 /*v357*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v93 /*v349*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v77 /*v333*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v71 /*v327*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v67 /*v323*/, v69 /*v325*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[156:163] /*v[412:419]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v101 /*v357*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v100 /*v356*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v99 /*v355*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v98 /*v354*/, v105 /*v361*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v97 /*v353*/, v97 /*v353*/, v103 /*v359*/
	v_dual_fmac_f32 v150 /*v406*/, v132 /*v388*/, v64 /*v320*/ :: v_dual_fmac_f32 v151 /*v407*/, v134 /*v390*/, v65 /*v321*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[164:171] /*v[420:427]*/, v[56:63]
	s_add_nc_u64 s[36:37], s[36:37], 1
	s_set_vgpr_msb 0x545
	v_mov_b32_e32 v149 /*v405*/, v135 /*v391*/
	v_cmp_ge_u64_e64 s0, s[36:37], s[66:67]
	v_dual_add_f32 v64 /*v320*/, v150 /*v406*/, v130 /*v386*/ :: v_dual_add_f32 v65 /*v321*/, v151 /*v407*/, v131 /*v387*/
	v_nop
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[156:163] /*v[412:419]*/, v[112:119]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v110 /*v366*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v102 /*v358*/
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v94 /*v350*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v72 /*v328*/, v78 /*v334*/
	v_cvt_pk_bf16_f32 v96 /*v352*/, v89 /*v345*/, v95 /*v351*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[164:171] /*v[420:427]*/, v[48:55]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v95 /*v351*/, v81 /*v337*/, v87 /*v343*/
	v_cvt_pk_bf16_f32 v94 /*v350*/, v73 /*v329*/, v79 /*v335*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v151 /*v407*/, v148 /*v404*/ :: v_dual_mov_b32 v148 /*v404*/, v146 /*v402*/
	v_dual_mov_b32 v146 /*v402*/, v147 /*v403*/ :: v_dual_mov_b32 v147 /*v403*/, v154 /*v410*/
	v_mov_b32_e32 v150 /*v406*/, v133 /*v389*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[156:163] /*v[412:419]*/, v[104:111]
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[164:171] /*v[420:427]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[156:163] /*v[412:419]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[164:171] /*v[420:427]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[156:163] /*v[412:419]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[164:171] /*v[420:427]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[156:163] /*v[412:419]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[164:171] /*v[420:427]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[156:163] /*v[412:419]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[164:171] /*v[420:427]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[156:163] /*v[412:419]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[164:171] /*v[420:427]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[94:101] /*v[350:357]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[94:101] /*v[350:357]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[94:101] /*v[350:357]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[94:101] /*v[350:357]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[94:101] /*v[350:357]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[94:101] /*v[350:357]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[94:101] /*v[350:357]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[94:101] /*v[350:357]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_48
.LBB0_41:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v154 /*v410*/, v146 /*v402*/ :: v_dual_mov_b32 v146 /*v402*/, v151 /*v407*/
	s_add_co_i32 s0, s36, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[56:59] /*v[312:315]*/, v147 /*v403*/
	ds_load_b128 v[60:63] /*v[316:319]*/, v147 /*v403*/ offset:32
	ds_load_b128 v[48:51] /*v[304:307]*/, v147 /*v403*/ offset:64
	ds_load_b128 v[52:55] /*v[308:311]*/, v147 /*v403*/ offset:96
	ds_load_b128 v[40:43] /*v[296:299]*/, v147 /*v403*/ offset:128
	ds_load_b128 v[44:47] /*v[300:303]*/, v147 /*v403*/ offset:160
	ds_load_b128 v[32:35] /*v[288:291]*/, v147 /*v403*/ offset:192
	ds_load_b128 v[36:39] /*v[292:295]*/, v147 /*v403*/ offset:224
	ds_load_b128 v[24:27] /*v[280:283]*/, v147 /*v403*/ offset:4352
	ds_load_b128 v[28:31] /*v[284:287]*/, v147 /*v403*/ offset:4384
	ds_load_b128 v[16:19] /*v[272:275]*/, v147 /*v403*/ offset:4416
	ds_load_b128 v[20:23] /*v[276:279]*/, v147 /*v403*/ offset:4448
	ds_load_b128 v[8:11] /*v[264:267]*/, v147 /*v403*/ offset:4480
	ds_load_b128 v[12:15] /*v[268:271]*/, v147 /*v403*/ offset:4512
	ds_load_b128 v[0:3] /*v[256:259]*/, v147 /*v403*/ offset:4544
	ds_load_b128 v[4:7] /*v[260:263]*/, v147 /*v403*/ offset:4576
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[248:251], v147 /*v403*/ offset:8704
	ds_load_b128 v[252:255], v147 /*v403*/ offset:8736
	ds_load_b128 v[240:243], v147 /*v403*/ offset:8768
	ds_load_b128 v[244:247], v147 /*v403*/ offset:8800
	ds_load_b128 v[232:235], v147 /*v403*/ offset:8832
	ds_load_b128 v[236:239], v147 /*v403*/ offset:8864
	ds_load_b128 v[224:227], v147 /*v403*/ offset:8896
	ds_load_b128 v[228:231], v147 /*v403*/ offset:8928
	ds_load_b128 v[216:219], v147 /*v403*/ offset:13056
	ds_load_b128 v[220:223], v147 /*v403*/ offset:13088
	ds_load_b128 v[208:211], v147 /*v403*/ offset:13120
	ds_load_b128 v[212:215], v147 /*v403*/ offset:13152
	ds_load_b128 v[200:203], v147 /*v403*/ offset:13184
	ds_load_b128 v[204:207], v147 /*v403*/ offset:13216
	ds_load_b128 v[192:195], v147 /*v403*/ offset:13248
	ds_load_b128 v[196:199], v147 /*v403*/ offset:13280
	s_cmp_ge_i32 s0, s66
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_43
	s_lshl_b32 s5, s0, 6
	s_lshl_b32 s0, s0, 17
	s_sub_co_i32 s7, s13, s5
	s_add_co_i32 s6, s5, s64
	s_set_vgpr_msb 0x41
	v_med3_i32 v66 /*v322*/, s7, 0, 64
	s_ashr_i32 s7, s6, 31
	s_and_b32 s0, s0, 0x20000
	s_mul_u64 s[26:27], s[6:7], s[70:71]
	s_mul_u64 s[6:7], s[6:7], s[68:69]
	v_readfirstlane_b32 s18, v66 /*v322*/
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[26:27], s[26:27], 1
	s_add_nc_u64 s[6:7], s[14:15], s[6:7]
	s_or_b32 s5, s46, s0
	s_sub_co_i32 s18, s18, s45
	s_add_nc_u64 s[6:7], s[2:3], s[6:7]
	s_max_i32 s18, s18, 0
	s_add_nc_u64 s[26:27], s[10:11], s[26:27]
	s_lshl_b32 s18, s18, 16
	s_bitset1_b32 s7, 31
	s_addk_co_i32 s18, 0x7fff
	s_mov_b32 s25, s17
	tensor_load_to_lds s[4:7], s[16:23]
	s_add_nc_u64 s[6:7], s[8:9], s[26:27]
	s_or_b32 s5, s47, s0
	s_bitset1_b32 s7, 31
	s_mov_b32 s26, s18
	s_mov_b32 s27, s19
	s_mov_b32 s28, s20
	s_mov_b32 s31, s23
	tensor_load_to_lds s[4:7], s[24:31]
	s_set_vgpr_msb 0x4100
.LBB0_43:
	s_set_vgpr_msb 0x41
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_set_vgpr_msb 0x4140
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[248:255], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[248:255], v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x4051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[48:55] /*v[304:311]*/, v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[48:55] /*v[304:311]*/, v[168:175], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[16:23] /*v[272:279]*/, v[136:143], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[16:23] /*v[272:279]*/, v[168:175], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[240:247], v[136:143], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[240:247], v[168:175], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[208:215], v[136:143], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[208:215], v[168:175], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[40:47] /*v[296:303]*/, v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[40:47] /*v[296:303]*/, v[176:183], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[8:15] /*v[264:271]*/, v[144:151], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[8:15] /*v[264:271]*/, v[176:183], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[232:239], v[144:151], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[232:239], v[176:183], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[200:207], v[144:151], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[200:207], v[176:183], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[32:39] /*v[288:295]*/, v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[32:39] /*v[288:295]*/, v[184:191], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[0:7] /*v[256:263]*/, v[152:159], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[0:7] /*v[256:263]*/, v[184:191], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[224:231], v[152:159], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[224:231], v[184:191], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[192:199], v[152:159], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[192:199], v[184:191], v[122:129] /*v[378:385]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5041
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v148 /*v404*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v148 /*v404*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v148 /*v404*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v148 /*v404*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v148 /*v404*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v148 /*v404*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v148 /*v404*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v148 /*v404*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v148 /*v404*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v148 /*v404*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v148 /*v404*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v148 /*v404*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v148 /*v404*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v148 /*v404*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v148 /*v404*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v148 /*v404*/ offset:13920
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[248:251], v148 /*v404*/ offset:128
	ds_load_tr16_b128 v[232:235], v148 /*v404*/ offset:160
	ds_load_tr16_b128 v[252:255], v148 /*v404*/ offset:4736
	ds_load_tr16_b128 v[236:239], v148 /*v404*/ offset:4768
	ds_load_tr16_b128 v[240:243], v148 /*v404*/ offset:9344
	ds_load_tr16_b128 v[224:227], v148 /*v404*/ offset:9376
	ds_load_tr16_b128 v[244:247], v148 /*v404*/ offset:13952
	ds_load_tr16_b128 v[228:231], v148 /*v404*/ offset:13984
	ds_load_tr16_b128 v[216:219], v148 /*v404*/ offset:192
	ds_load_tr16_b128 v[200:203], v148 /*v404*/ offset:224
	ds_load_tr16_b128 v[220:223], v148 /*v404*/ offset:4800
	ds_load_tr16_b128 v[204:207], v148 /*v404*/ offset:4832
	ds_load_tr16_b128 v[208:211], v148 /*v404*/ offset:9408
	ds_load_tr16_b128 v[192:195], v148 /*v404*/ offset:9440
	ds_load_tr16_b128 v[212:215], v148 /*v404*/ offset:14016
	ds_load_tr16_b128 v[196:199], v148 /*v404*/ offset:14048
	s_set_vgpr_msb 0x155
	v_lshl_or_b32 v132 /*v388*/, s36, 6, v145 /*v401*/
	v_cmp_le_i32_e32 vcc_lo, v132 /*v388*/, v152 /*v408*/
	v_dual_add_nc_u32 v173 /*v429*/, 17, v132 /*v388*/ :: v_dual_bitop2_b32 v133 /*v389*/, 2, v132 /*v388*/ bitop3:0x54
	v_dual_add_nc_u32 v174 /*v430*/, 18, v132 /*v388*/ :: v_dual_bitop2_b32 v134 /*v390*/, 3, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e32 v66 /*v322*/, 0xff800000, v66 /*v322*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v132 /*v388*/, v152 /*v408*/
	v_dual_add_nc_u32 v175 /*v431*/, 19, v132 /*v388*/ :: v_dual_bitop2_b32 v135 /*v391*/, 4, v132 /*v388*/ bitop3:0x54
	s_wait_alu depctr_vm_vsrc(6)
	v_or_b32_e32 v151 /*v407*/, 5, v132 /*v388*/
	v_or_b32_e32 v155 /*v411*/, 6, v132 /*v388*/
	v_cndmask_b32_e32 v67 /*v323*/, 0xff800000, v67 /*v323*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v133 /*v389*/, v152 /*v408*/
	v_dual_add_nc_u32 v178 /*v434*/, 22, v132 /*v388*/ :: v_dual_bitop2_b32 v171 /*v427*/, 7, v132 /*v388*/ bitop3:0x54
	v_dual_add_nc_u32 v179 /*v435*/, 23, v132 /*v388*/ :: v_dual_bitop2_b32 v172 /*v428*/, 16, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e32 v68 /*v324*/, 0xff800000, v68 /*v324*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v134 /*v390*/, v152 /*v408*/
	v_dual_add_nc_u32 v189 /*v445*/, 49, v132 /*v388*/ :: v_dual_bitop2_b32 v180 /*v436*/, 32, v132 /*v388*/ bitop3:0x54
	v_dual_add_nc_u32 v190 /*v446*/, 50, v132 /*v388*/ :: v_dual_bitop2_b32 v181 /*v437*/, 33, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e32 v69 /*v325*/, 0xff800000, v69 /*v325*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v135 /*v391*/, v152 /*v408*/
	v_or_b32_e32 v182 /*v438*/, 34, v132 /*v388*/
	v_or_b32_e32 v187 /*v443*/, 39, v132 /*v388*/
	v_dual_add_nc_u32 v195 /*v451*/, 55, v132 /*v388*/ :: v_dual_bitop2_b32 v188 /*v444*/, 48, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e32 v70 /*v326*/, 0xff800000, v70 /*v326*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v151 /*v407*/, v152 /*v408*/
	v_cndmask_b32_e32 v71 /*v327*/, 0xff800000, v71 /*v327*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v155 /*v411*/, v152 /*v408*/
	v_cndmask_b32_e32 v72 /*v328*/, 0xff800000, v72 /*v328*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v171 /*v427*/, v152 /*v408*/
	v_cndmask_b32_e32 v73 /*v329*/, 0xff800000, v73 /*v329*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v172 /*v428*/, v152 /*v408*/
	v_cndmask_b32_e32 v82 /*v338*/, 0xff800000, v82 /*v338*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v173 /*v429*/, v152 /*v408*/
	v_cndmask_b32_e32 v83 /*v339*/, 0xff800000, v83 /*v339*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v174 /*v430*/, v152 /*v408*/
	v_cndmask_b32_e32 v130 /*v386*/, 0xff800000, v84 /*v340*/, vcc_lo
	v_add_nc_u32_e32 v84 /*v340*/, 20, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v175 /*v431*/, v152 /*v408*/
	v_cndmask_b32_e32 v131 /*v387*/, 0xff800000, v85 /*v341*/, vcc_lo
	v_add_nc_u32_e32 v85 /*v341*/, 21, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v84 /*v340*/, v152 /*v408*/
	v_cndmask_b32_e32 v86 /*v342*/, 0xff800000, v86 /*v342*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v85 /*v341*/, v152 /*v408*/
	v_cndmask_b32_e32 v87 /*v343*/, 0xff800000, v87 /*v343*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v178 /*v434*/, v152 /*v408*/
	v_cndmask_b32_e32 v88 /*v344*/, 0xff800000, v88 /*v344*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v179 /*v435*/, v152 /*v408*/
	v_cndmask_b32_e32 v89 /*v345*/, 0xff800000, v89 /*v345*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v180 /*v436*/, v152 /*v408*/
	v_cndmask_b32_e32 v156 /*v412*/, 0xff800000, v98 /*v354*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v181 /*v437*/, v152 /*v408*/
	v_or_b32_e32 v98 /*v354*/, 35, v132 /*v388*/
	v_cndmask_b32_e32 v157 /*v413*/, 0xff800000, v99 /*v355*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v182 /*v438*/, v152 /*v408*/
	v_or_b32_e32 v99 /*v355*/, 36, v132 /*v388*/
	v_cndmask_b32_e32 v158 /*v414*/, 0xff800000, v100 /*v356*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v98 /*v354*/, v152 /*v408*/
	v_or_b32_e32 v100 /*v356*/, 37, v132 /*v388*/
	v_cndmask_b32_e32 v159 /*v415*/, 0xff800000, v101 /*v357*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v99 /*v355*/, v152 /*v408*/
	v_or_b32_e32 v101 /*v357*/, 38, v132 /*v388*/
	v_cndmask_b32_e32 v102 /*v358*/, 0xff800000, v102 /*v358*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v100 /*v356*/, v152 /*v408*/
	v_cndmask_b32_e32 v103 /*v359*/, 0xff800000, v103 /*v359*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v101 /*v357*/, v152 /*v408*/
	v_cndmask_b32_e32 v104 /*v360*/, 0xff800000, v104 /*v360*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v187 /*v443*/, v152 /*v408*/
	v_cndmask_b32_e32 v105 /*v361*/, 0xff800000, v105 /*v361*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v188 /*v444*/, v152 /*v408*/
	v_cndmask_b32_e32 v160 /*v416*/, 0xff800000, v114 /*v370*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v189 /*v445*/, v152 /*v408*/
	v_dual_cndmask_b32 v161 /*v417*/, 0xff800000, v115 /*v371*/ :: v_dual_add_nc_u32 v114 /*v370*/, 51, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v190 /*v446*/, v152 /*v408*/
	v_add_nc_u32_e32 v115 /*v371*/, 52, v132 /*v388*/
	v_cndmask_b32_e32 v162 /*v418*/, 0xff800000, v116 /*v372*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v114 /*v370*/, v152 /*v408*/
	v_dual_cndmask_b32 v163 /*v419*/, 0xff800000, v117 /*v373*/ :: v_dual_add_nc_u32 v116 /*v372*/, 53, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v115 /*v371*/, v152 /*v408*/
	v_dual_cndmask_b32 v118 /*v374*/, 0xff800000, v118 /*v374*/ :: v_dual_add_nc_u32 v117 /*v373*/, 54, v132 /*v388*/
	v_cmp_le_i32_e32 vcc_lo, v116 /*v372*/, v152 /*v408*/
	v_cndmask_b32_e32 v119 /*v375*/, 0xff800000, v119 /*v375*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v117 /*v373*/, v152 /*v408*/
	v_cndmask_b32_e32 v120 /*v376*/, 0xff800000, v120 /*v376*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v195 /*v451*/, v152 /*v408*/
	v_cndmask_b32_e32 v121 /*v377*/, 0xff800000, v121 /*v377*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v132 /*v388*/, v153 /*v409*/
	v_cndmask_b32_e32 v164 /*v420*/, 0xff800000, v74 /*v330*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v132 /*v388*/, v153 /*v409*/
	v_max_num_f32_e32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/
	v_cndmask_b32_e32 v165 /*v421*/, 0xff800000, v75 /*v331*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v133 /*v389*/, v153 /*v409*/
	v_dual_max_num_f32 v75 /*v331*/, v164 /*v420*/, v165 /*v421*/ :: v_dual_cndmask_b32 v166 /*v422*/, 0xff800000, v76 /*v332*/
	v_cmp_le_i32_e32 vcc_lo, v134 /*v390*/, v153 /*v409*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_cndmask_b32_e32 v167 /*v423*/, 0xff800000, v77 /*v333*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v135 /*v391*/, v153 /*v409*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	v_cndmask_b32_e32 v168 /*v424*/, 0xff800000, v78 /*v334*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v151 /*v407*/, v153 /*v409*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v82 /*v338*/
	v_cndmask_b32_e32 v169 /*v425*/, 0xff800000, v79 /*v335*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v155 /*v411*/, v153 /*v409*/
	v_max3_num_f32 v77 /*v333*/, v167 /*v423*/, v168 /*v424*/, v169 /*v425*/
	v_cndmask_b32_e32 v170 /*v426*/, 0xff800000, v80 /*v336*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v171 /*v427*/, v153 /*v409*/
	v_max3_num_f32 v80 /*v336*/, v83 /*v339*/, v130 /*v386*/, v131 /*v387*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v166 /*v422*/, v77 /*v333*/
	v_cndmask_b32_e32 v171 /*v427*/, 0xff800000, v81 /*v337*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v172 /*v428*/, v153 /*v409*/
	v_cndmask_b32_e32 v172 /*v428*/, 0xff800000, v90 /*v346*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v173 /*v429*/, v153 /*v409*/
	v_max3_num_f32 v90 /*v346*/, v89 /*v345*/, v156 /*v412*/, v157 /*v413*/
	v_cndmask_b32_e32 v173 /*v429*/, 0xff800000, v91 /*v347*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v174 /*v430*/, v153 /*v409*/
	v_max3_num_f32 v79 /*v335*/, v170 /*v426*/, v171 /*v427*/, v172 /*v428*/
	v_cndmask_b32_e32 v174 /*v430*/, 0xff800000, v92 /*v348*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v175 /*v431*/, v153 /*v409*/
	v_max3_num_f32 v92 /*v348*/, v158 /*v414*/, v159 /*v415*/, v102 /*v358*/
	v_cndmask_b32_e32 v175 /*v431*/, 0xff800000, v93 /*v349*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v84 /*v340*/, v153 /*v409*/
	v_max3_num_f32 v84 /*v340*/, v86 /*v342*/, v87 /*v343*/, v88 /*v344*/
	v_max3_num_f32 v93 /*v349*/, v103 /*v359*/, v104 /*v360*/, v105 /*v361*/
	v_max3_num_f32 v81 /*v337*/, v173 /*v429*/, v174 /*v430*/, v175 /*v431*/
	v_cndmask_b32_e32 v176 /*v432*/, 0xff800000, v94 /*v350*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v85 /*v341*/, v153 /*v409*/
	v_max3_num_f32 v94 /*v350*/, v160 /*v416*/, v161 /*v417*/, v162 /*v418*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v90 /*v346*/
	v_cndmask_b32_e32 v177 /*v433*/, 0xff800000, v95 /*v351*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v178 /*v434*/, v153 /*v409*/
	v_max3_num_f32 v95 /*v351*/, v163 /*v419*/, v118 /*v374*/, v119 /*v375*/
	v_max3_num_f32 v90 /*v346*/, v92 /*v348*/, v93 /*v349*/, v94 /*v350*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	v_cndmask_b32_e32 v178 /*v434*/, 0xff800000, v96 /*v352*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v179 /*v435*/, v153 /*v409*/
	v_max3_num_f32 v92 /*v348*/, v95 /*v351*/, v120 /*v376*/, v121 /*v377*/
	v_cndmask_b32_e32 v179 /*v435*/, 0xff800000, v97 /*v353*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v180 /*v436*/, v153 /*v409*/
	v_max3_num_f32 v85 /*v341*/, v176 /*v432*/, v177 /*v433*/, v178 /*v434*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v90 /*v346*/, v92 /*v348*/
	v_cndmask_b32_e32 v180 /*v436*/, 0xff800000, v106 /*v362*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v181 /*v437*/, v153 /*v409*/
	v_cndmask_b32_e32 v181 /*v437*/, 0xff800000, v107 /*v363*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v182 /*v438*/, v153 /*v409*/
	v_max3_num_f32 v91 /*v347*/, v179 /*v435*/, v180 /*v436*/, v181 /*v437*/
	v_cndmask_b32_e32 v182 /*v438*/, 0xff800000, v108 /*v364*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v98 /*v354*/, v153 /*v409*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v85 /*v341*/, v91 /*v347*/
	v_cndmask_b32_e32 v183 /*v439*/, 0xff800000, v109 /*v365*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v99 /*v355*/, v153 /*v409*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_cndmask_b32_e32 v184 /*v440*/, 0xff800000, v110 /*v366*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v100 /*v356*/, v153 /*v409*/
	v_cndmask_b32_e32 v185 /*v441*/, 0xff800000, v111 /*v367*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v101 /*v357*/, v153 /*v409*/
	v_max3_num_f32 v80 /*v336*/, v182 /*v438*/, v183 /*v439*/, v184 /*v440*/
	v_cndmask_b32_e32 v186 /*v442*/, 0xff800000, v112 /*v368*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v187 /*v443*/, v153 /*v409*/
	v_cndmask_b32_e32 v187 /*v443*/, 0xff800000, v113 /*v369*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v188 /*v444*/, v153 /*v409*/
	v_max3_num_f32 v84 /*v340*/, v185 /*v441*/, v186 /*v442*/, v187 /*v443*/
	v_cndmask_b32_e32 v188 /*v444*/, 0xff800000, v122 /*v378*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v189 /*v445*/, v153 /*v409*/
	v_cndmask_b32_e32 v189 /*v445*/, 0xff800000, v123 /*v379*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v190 /*v446*/, v153 /*v409*/
	v_cndmask_b32_e32 v190 /*v446*/, 0xff800000, v124 /*v380*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v114 /*v370*/, v153 /*v409*/
	v_cndmask_b32_e32 v191 /*v447*/, 0xff800000, v125 /*v381*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v115 /*v371*/, v153 /*v409*/
	v_max3_num_f32 v76 /*v332*/, v188 /*v444*/, v189 /*v445*/, v190 /*v446*/
	v_cndmask_b32_e32 v192 /*v448*/, 0xff800000, v126 /*v382*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v116 /*v372*/, v153 /*v409*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v76 /*v332*/
	v_cndmask_b32_e32 v193 /*v449*/, 0xff800000, v127 /*v383*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v117 /*v373*/, v153 /*v409*/
	v_max3_num_f32 v78 /*v334*/, v191 /*v447*/, v192 /*v448*/, v193 /*v449*/
	v_cndmask_b32_e32 v194 /*v450*/, 0xff800000, v128 /*v384*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v195 /*v451*/, v153 /*v409*/
	v_cndmask_b32_e32 v195 /*v451*/, 0xff800000, v129 /*v385*/, vcc_lo
	v_max3_num_f32 v78 /*v334*/, v78 /*v334*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_dual_mov_b32 v77 /*v333*/, v74 /*v330*/ :: v_dual_mov_b32 v76 /*v332*/, v75 /*v331*/
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s1, 0xfedcba98
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s1, 0xfedcba98
	v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v77 /*v333*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/
	v_sub_f32_e32 v77 /*v333*/, v74 /*v330*/, v150 /*v406*/
	v_dual_max_num_f32 v74 /*v330*/, v150 /*v406*/, v74 /*v330*/ :: v_dual_sub_f32 v76 /*v332*/, v75 /*v331*/, v149 /*v405*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v77 /*v333*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s0, -1, 0
	v_dual_cndmask_b32 v133 /*v389*/, v74 /*v330*/, v150 /*v406*/, s0 :: v_dual_max_num_f32 v74 /*v330*/, v149 /*v405*/, v75 /*v331*/
	v_cmp_lt_f32_e64 s0, 0x41000000, v76 /*v332*/
	v_mul_f32_e32 v124 /*v380*/, 0xbfb8aa3b, v133 /*v389*/
	s_cmp_lg_u32 s0, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s0, 0
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[72:73] /*v[328:329]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s0, -1, 0
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[130:131] /*v[386:387]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v135 /*v391*/, v74 /*v330*/, v149 /*v405*/, s0
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v72 /*v328*/
	v_mul_f32_e32 v132 /*v388*/, 0xbfb8aa3b, v135 /*v391*/
	v_exp_f32_e32 v90 /*v346*/, v73 /*v329*/
	v_nop
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[86:87] /*v[342:343]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v80 /*v336*/
	v_exp_f32_e32 v106 /*v362*/, v81 /*v337*/
	v_nop
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[156:157] /*v[412:413]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[86:87] /*v[342:343]*/, v[158:159] /*v[414:415]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[164:165] /*v[420:421]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[166:167] /*v[422:423]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[168:169] /*v[424:425]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v67 /*v323*/
	v_exp_f32_e32 v70 /*v326*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[82:83] /*v[338:339]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_exp_f32_e32 v67 /*v323*/, v130 /*v386*/
	v_exp_f32_e32 v69 /*v325*/, v131 /*v387*/
	v_exp_f32_e32 v71 /*v327*/, v156 /*v412*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[170:171] /*v[426:427]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v75 /*v331*/, v157 /*v413*/
	v_exp_f32_e32 v77 /*v333*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[172:173] /*v[428:429]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v159 /*v415*/
	v_nop
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[174:175] /*v[430:431]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v78 /*v334*/
	v_exp_f32_e32 v98 /*v354*/, v79 /*v335*/
	v_nop
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[88:89] /*v[344:345]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v130 /*v386*/
	v_exp_f32_e32 v91 /*v347*/, v131 /*v387*/
	v_exp_f32_e32 v93 /*v349*/, v156 /*v412*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[176:177] /*v[432:433]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v157 /*v413*/
	v_exp_f32_e32 v101 /*v357*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[178:179] /*v[434:435]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v159 /*v415*/
	v_nop
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[180:181] /*v[436:437]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v114 /*v370*/, v73 /*v329*/
	v_exp_f32_e32 v122 /*v378*/, v79 /*v335*/
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[102:103] /*v[358:359]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[104:105] /*v[360:361]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v109 /*v365*/, v130 /*v386*/
	v_exp_f32_e32 v115 /*v371*/, v131 /*v387*/
	v_exp_f32_e32 v117 /*v373*/, v156 /*v412*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[182:183] /*v[438:439]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v157 /*v413*/
	v_exp_f32_e32 v73 /*v329*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[184:185] /*v[440:441]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v79 /*v335*/, v159 /*v415*/
	v_nop
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[186:187] /*v[442:443]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_exp_f32_e32 v76 /*v332*/, v76 /*v332*/
	v_exp_f32_e32 v108 /*v364*/, v72 /*v328*/
	v_exp_f32_e32 v116 /*v372*/, v78 /*v334*/
	v_exp_f32_e32 v72 /*v328*/, v80 /*v336*/
	v_exp_f32_e32 v78 /*v334*/, v81 /*v337*/
	v_exp_f32_e32 v80 /*v336*/, v86 /*v342*/
	v_exp_f32_e32 v86 /*v342*/, v87 /*v343*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[160:161] /*v[416:417]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v94 /*v350*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[162:163] /*v[418:419]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v102 /*v358*/, v97 /*v353*/
	v_exp_f32_e32 v81 /*v337*/, v130 /*v386*/
	v_exp_f32_e32 v87 /*v343*/, v131 /*v387*/
	v_exp_f32_e32 v89 /*v345*/, v156 /*v412*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[188:189] /*v[444:445]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v95 /*v351*/, v157 /*v413*/
	v_exp_f32_e32 v97 /*v353*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[190:191] /*v[446:447]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v103 /*v359*/, v159 /*v415*/
	v_nop
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[192:193] /*v[448:449]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_pk_fma_f32 v[126:127] /*v[382:383]*/, v[118:119] /*v[374:375]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v105 /*v361*/
	v_pk_fma_f32 v[128:129] /*v[384:385]*/, v[120:121] /*v[376:377]*/, s[38:39], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v105 /*v361*/, v130 /*v386*/
	v_exp_f32_e32 v111 /*v367*/, v131 /*v387*/
	v_exp_f32_e32 v113 /*v369*/, v156 /*v412*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[194:195] /*v[450:451]*/, s[38:39], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v157 /*v413*/
	v_exp_f32_e32 v121 /*v377*/, v158 /*v414*/
	v_exp_f32_e32 v125 /*v381*/, v159 /*v415*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[74:75] /*v[330:331]*/, v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[98:99] /*v[354:355]*/, v[100:101] /*v[356:357]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[108:109] /*v[364:365]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v120 /*v376*/, v126 /*v382*/
	v_exp_f32_e32 v124 /*v380*/, v127 /*v383*/
	v_exp_f32_e32 v126 /*v382*/, v128 /*v384*/
	v_exp_f32_e32 v128 /*v384*/, v129 /*v385*/
	v_exp_f32_e32 v127 /*v383*/, v130 /*v386*/
	v_exp_f32_e32 v129 /*v385*/, v131 /*v387*/
	v_nop
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[70:71] /*v[326:327]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[82:83] /*v[338:339]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[122:123] /*v[378:379]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[106:107] /*v[362:363]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[116:117] /*v[372:373]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[92:93] /*v[348:349]*/, v[130:131] /*v[386:387]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[78:79] /*v[334:335]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[104:105] /*v[360:361]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[88:89] /*v[344:345]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[102:103] /*v[358:359]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[112:113] /*v[368:369]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[166:167] /*v[422:423]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[124:125] /*v[380:381]*/, v[168:169] /*v[424:425]*/
	v_sub_f32_e32 v132 /*v388*/, v150 /*v406*/, v133 /*v389*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[162:163] /*v[418:419]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[160:161] /*v[416:417]*/
	v_mul_f32_e32 v132 /*v388*/, 0x3fb8aa3b, v132 /*v388*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[156:157] /*v[412:413]*/, v[130:131] /*v[386:387]*/
	v_exp_f32_e32 v132 /*v388*/, v132 /*v388*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/, v[130:131] /*v[386:387]*/
	v_dual_mov_b32 v150 /*v406*/, v130 /*v386*/ :: v_dual_mov_b32 v151 /*v407*/, v131 /*v387*/
	v_permlanex16_b32 v150 /*v406*/, v150 /*v406*/, s1, 0xfedcba98
	v_permlanex16_b32 v151 /*v407*/, v151 /*v407*/, s1, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_45
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_45:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v134 /*v390*/, v149 /*v405*/, v135 /*v391*/
	s_and_b32 s0, s5, exec_lo
	s_cselect_b32 s0, 1, 0
	s_cmp_lg_u32 s0, 1
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_40
	v_nop
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[62:63], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_40
.LBB0_47:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v135 /*v391*/, v149 /*v405*/ :: v_dual_mov_b32 v133 /*v389*/, v150 /*v406*/
	s_set_vgpr_msb 0x4100
.LBB0_48:
	s_set_vgpr_msb 5
	v_div_scale_f32 v129, null, v64 /*v320*/, v64 /*v320*/, 1.0
	v_div_scale_f32 v132, vcc_lo, 1.0, v64 /*v320*/, 1.0
	v_div_scale_f32 v150, null, v65 /*v321*/, v65 /*v321*/, 1.0
	s_lshl_b32 s0, s66, 17
	v_mul_u32_u24_e32 v128, 0x110, v141 /*v397*/
	s_set_vgpr_msb 0x504
	v_rcp_f32_e32 v130, v129
	s_and_b32 s1, s0, 0x20000
	v_rcp_f32_e32 v151, v150
	v_cmp_lt_f32_e64 s0, 0, v64 /*v320*/
	s_mov_b32 s2, 0
	s_set_vgpr_msb 0x400
	s_wait_dscnt 0x0
	v_fma_f32 v131, -v129, v130, 1.0
	v_fmac_f32_e32 v130, v131, v130
	v_mul_f32_e32 v131, v132, v130
	v_fma_f32 v133, -v129, v131, v132
	v_fmac_f32_e32 v131, v133, v130
	v_fma_f32 v129, -v129, v131, v132
	v_div_fmas_f32 v129, v129, v130, v131
	v_fma_f32 v130, -v150, v151, 1.0
	s_set_vgpr_msb 4
	v_div_scale_f32 v152, vcc_lo, 1.0, v65 /*v321*/, 1.0
	v_div_fixup_f32 v129, v129, v64 /*v320*/, 1.0
	s_set_vgpr_msb 0x400
	v_dual_fmac_f32 v151, v130, v151 :: v_dual_cndmask_b32 v130, 0, v129, s0
	v_mul_f32_e32 v153, v152, v151
	s_add_co_i32 s0, s1, s93
	s_set_vgpr_msb 4
	v_add_nc_u32_e32 v129, s0, v144 /*v400*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[140:141], v[130:131], v[76:77] op_sel_hi:[0,1]
	v_fma_f32 v76, -v150, v153, v152
	v_pk_mul_f32 v[90:91], v[130:131], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[130:131], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[132:133], v[130:131], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[130:131], v[92:93] op_sel_hi:[0,1]
	v_fmac_f32_e32 v153, v76, v151
	v_cvt_pk_bf16_f32 v81, v90, v91
	v_cvt_pk_bf16_f32 v80, v88, v89
	v_pk_mul_f32 v[96:97], v[96:97], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135], v[130:131], v[82:83] op_sel_hi:[0,1]
	v_fma_f32 v90, -v150, v153, v152
	v_cvt_pk_bf16_f32 v82, v92, v93
	v_pk_mul_f32 v[112:113], v[112:113], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[130:131] op_sel_hi:[1,0]
	v_div_fmas_f32 v88, v90, v151, v153
	s_set_vgpr_msb 4
	v_cmp_lt_f32_e32 vcc_lo, 0, v65 /*v321*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[106:107], v[106:107], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4
	v_div_fixup_f32 v92, v88, v65 /*v321*/, 1.0
	s_set_vgpr_msb 0x410
	v_pk_mul_f32 v[98:99], v[98:99], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[130:131] op_sel_hi:[1,0]
	v_cvt_pk_bf16_f32 v76, v96, v97
	v_cndmask_b32_e32 v96, 0, v92, vcc_lo
	v_pk_mul_f32 v[120:121], v[120:121], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[130:131], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[130:131], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[130:131], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[136:137], v[130:131], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[138:139], v[130:131], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[142:143], v[130:131], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[130:131], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[146:147], v[130:131], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[148:149], v[130:131], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[130:131], v[130:131], v[70:71] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v69, v114, v115
	v_cvt_pk_bf16_f32 v68, v112, v113
	v_cvt_pk_bf16_f32 v75, v110, v111
	v_cvt_pk_bf16_f32 v74, v108, v109
	v_cvt_pk_bf16_f32 v73, v106, v107
	v_cvt_pk_bf16_f32 v72, v104, v105
	v_cvt_pk_bf16_f32 v79, v102, v103
	v_cvt_pk_bf16_f32 v78, v100, v101
	v_cvt_pk_bf16_f32 v77, v98, v99
	v_pk_mul_f32 v[56:57], v[56:57], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[96:97], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[96:97], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[96:97], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[96:97], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[96:97], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[96:97], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[96:97], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[96:97], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[96:97], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[96:97], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[96:97], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[96:97], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[110:111], v[96:97], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[112:113], v[96:97], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[114:115], v[96:97], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[96:97], v[96:97], v[6:7] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v67, v126, v127
	v_cvt_pk_bf16_f32 v66, v124, v125
	v_cvt_pk_bf16_f32 v65, v122, v123
	v_cvt_pk_bf16_f32 v64, v120, v121
	v_cvt_pk_bf16_f32 v71, v118, v119
	v_cvt_pk_bf16_f32 v70, v116, v117
	v_cvt_pk_bf16_f32 v83, v94, v95
	v_cvt_pk_bf16_f32 v87, v86, v87
	v_cvt_pk_bf16_f32 v86, v84, v85
	v_cvt_pk_bf16_f32 v85, v134, v135
	v_cvt_pk_bf16_f32 v84, v132, v133
	v_cvt_pk_bf16_f32 v91, v142, v143
	v_cvt_pk_bf16_f32 v90, v140, v141
	v_cvt_pk_bf16_f32 v89, v138, v139
	v_cvt_pk_bf16_f32 v88, v136, v137
	v_cvt_pk_bf16_f32 v95, v130, v131
	v_cvt_pk_bf16_f32 v94, v148, v149
	v_cvt_pk_bf16_f32 v93, v146, v147
	v_cvt_pk_bf16_f32 v92, v144, v145
	v_add3_u32 v116, v128, s0, v143 /*v399*/
	v_cvt_pk_bf16_f32 v3, v62, v63
	v_cvt_pk_bf16_f32 v2, v60, v61
	v_cvt_pk_bf16_f32 v1, v58, v59
	v_cvt_pk_bf16_f32 v0, v56, v57
	s_wait_alu depctr_vm_vsrc(0)
	v_cvt_pk_bf16_f32 v7, v54, v55
	v_cvt_pk_bf16_f32 v6, v52, v53
	v_cvt_pk_bf16_f32 v5, v50, v51
	v_cvt_pk_bf16_f32 v4, v48, v49
	v_cvt_pk_bf16_f32 v11, v46, v47
	v_cvt_pk_bf16_f32 v10, v44, v45
	v_cvt_pk_bf16_f32 v9, v42, v43
	v_cvt_pk_bf16_f32 v8, v40, v41
	v_cvt_pk_bf16_f32 v15, v38, v39
	v_cvt_pk_bf16_f32 v14, v36, v37
	v_cvt_pk_bf16_f32 v13, v34, v35
	v_cvt_pk_bf16_f32 v12, v32, v33
	v_cvt_pk_bf16_f32 v19, v30, v31
	v_cvt_pk_bf16_f32 v18, v28, v29
	v_cvt_pk_bf16_f32 v17, v26, v27
	v_cvt_pk_bf16_f32 v16, v24, v25
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v20, v21
	v_cvt_pk_bf16_f32 v21, v100, v101
	v_cvt_pk_bf16_f32 v20, v98, v99
	v_cvt_pk_bf16_f32 v27, v108, v109
	v_cvt_pk_bf16_f32 v26, v106, v107
	v_cvt_pk_bf16_f32 v25, v104, v105
	v_cvt_pk_bf16_f32 v24, v102, v103
	v_cvt_pk_bf16_f32 v31, v96, v97
	v_cvt_pk_bf16_f32 v30, v114, v115
	v_cvt_pk_bf16_f32 v29, v112, v113
	v_cvt_pk_bf16_f32 v28, v110, v111
	s_lshl_b32 s1, s33, 2
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v129, v[64:67]
	ds_store_b128 v129, v[68:71] offset:32
	ds_store_b128 v129, v[72:75] offset:64
	ds_store_b128 v129, v[76:79] offset:96
	ds_store_b128 v129, v[80:83] offset:128
	ds_store_b128 v129, v[84:87] offset:160
	ds_store_b128 v129, v[88:91] offset:192
	ds_store_b128 v129, v[92:95] offset:224
	ds_store_b128 v116, v[0:3] offset:4352
	ds_store_b128 v116, v[4:7] offset:4384
	ds_store_b128 v116, v[8:11] offset:4416
	ds_store_b128 v116, v[12:15] offset:4448
	s_sub_co_i32 s1, s1, s57
	ds_store_b128 v116, v[16:19] offset:4480
	ds_store_b128 v116, v[20:23] offset:4512
	ds_store_b128 v116, v[24:27] offset:4544
	ds_store_b128 v116, v[28:31] offset:4576
	s_cmp_lt_i32 s1, 1
	s_set_vgpr_msb 0x1000
	s_wait_dscnt 0x0
	s_cbranch_scc1 .LBB0_50
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 5
	v_dual_lshrrev_b32 v26, 4, v142 /*v398*/ :: v_dual_bitop2_b32 v0, 28, v140 /*v396*/ bitop3:0x54
	s_add_co_i32 s1, s1, -1
	v_lshl_add_u32 v32, v141 /*v397*/, 4, s0
	s_min_u32 s1, s1, 31
	s_set_vgpr_msb 0x500
	v_or_b32_e32 v1, 30, v26
	v_min_u32_e32 v2, s1, v0
	v_or_b32_e32 v3, 26, v26
	s_set_vgpr_msb 4
	v_dual_lshlrev_b32 v0, 3, v141 /*v397*/ :: v_dual_bitop2_b32 v5, 24, v140 /*v396*/ bitop3:0x54
	s_set_vgpr_msb 0x400
	v_min_u32_e32 v4, s1, v1
	v_dual_mov_b32 v1, 0 :: v_dual_bitop2_b32 v6, s57, v2 bitop3:0x54
	v_min_u32_e32 v8, s1, v3
	v_min_u32_e32 v14, s1, v5
	v_dual_ashrrev_i32 v5, 31, v6 :: v_dual_bitop2_b32 v3, s57, v4 bitop3:0x54
	v_mad_u32_u24 v33, 0x110, v2, v32
	v_mad_u32_u24 v34, 0x110, v4, v32
	v_mad_u32_u24 v35, 0x110, v8, v32
	v_dual_ashrrev_i32 v10, 31, v3 :: v_dual_bitop2_b32 v9, s57, v8 bitop3:0x54
	v_lshrrev_b32_e32 v2, 30, v5
	s_set_vgpr_msb 4
	v_or_b32_e32 v21, 16, v140 /*v396*/
	s_set_vgpr_msb 0x400
	v_mad_u32_u24 v36, 0x110, v14, v32
	v_lshrrev_b32_e32 v10, 30, v10
	v_dual_ashrrev_i32 v5, 31, v9 :: v_dual_bitop2_b32 v7, 22, v26 bitop3:0x54
	v_or_b32_e32 v11, s57, v14
	v_min_u32_e32 v21, s1, v21
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 4
	v_or_b32_e32 v30, 8, v140 /*v396*/
	s_set_vgpr_msb 0x400
	v_min_u32_e32 v15, s1, v7
	v_dual_lshrrev_b32 v4, 30, v5 :: v_dual_add_nc_u32 v2, v6, v2
	v_dual_ashrrev_i32 v7, 31, v11 :: v_dual_add_nc_u32 v5, v3, v10
	v_mad_u32_u24 v37, 0x110, v15, v32
	v_dual_add_nc_u32 v12, v9, v4 :: v_dual_bitop2_b32 v10, -4, v2 bitop3:0x40
	v_dual_ashrrev_i32 v2, 2, v2 :: v_dual_bitop2_b32 v4, -4, v5 bitop3:0x40
	v_ashrrev_i32_e32 v5, 2, v5
	v_mad_u32_u24 v40, 0x110, v21, v32
	v_cmp_ne_u32_e32 vcc_lo, v6, v10
	v_add_nc_u32_e32 v13, s56, v2
	v_dual_sub_nc_u32 v2, v6, v10 :: v_dual_bitop2_b32 v10, -4, v12 bitop3:0x40
	v_cmp_ne_u32_e64 s0, v3, v4
	v_dual_sub_nc_u32 v6, v3, v4 :: v_dual_add_nc_u32 v16, s56, v5
	v_add_nc_u32_e32 v2, s62, v2
	s_and_b32 s3, s55, vcc_lo
	s_and_b32 s0, s55, s0
	v_dual_add_nc_u32 v4, s62, v6 :: v_dual_bitop2_b32 v18, s57, v15 bitop3:0x54
	v_cndmask_b32_e64 v6, 0, 1, s3
	v_cndmask_b32_e64 v17, 0, 1, s0
	v_mad_nc_i64_i32 v[2:3], v2, s52, v[0:1]
	v_mad_nc_i64_i32 v[4:5], v4, s52, v[0:1]
	v_cmp_ne_u32_e32 vcc_lo, v9, v10
	v_dual_sub_nc_u32 v6, v13, v6 :: v_dual_sub_nc_u32 v13, v16, v17
	v_dual_sub_nc_u32 v16, v9, v10 :: v_dual_ashrrev_i32 v12, 2, v12
	v_lshrrev_b32_e32 v9, 30, v7
	v_mad_nc_i64_i32 v[2:3], v6, s12, v[2:3]
	v_mad_nc_i64_i32 v[4:5], v13, s12, v[4:5]
	v_add_nc_u32_e32 v6, s62, v16
	s_set_vgpr_msb 4
	v_or_b32_e32 v13, 20, v140 /*v396*/
	s_and_b32 s0, s55, vcc_lo
	s_set_vgpr_msb 0x400
	v_dual_add_nc_u32 v10, s56, v12 :: v_dual_add_nc_u32 v9, v11, v9
	v_cndmask_b32_e64 v12, 0, 1, s0
	v_ashrrev_i32_e32 v16, 31, v18
	v_min_u32_e32 v17, s1, v13
	v_mad_nc_i64_i32 v[6:7], v6, s52, v[0:1]
	v_dual_sub_nc_u32 v10, v10, v12 :: v_dual_bitop2_b32 v8, -4, v9 bitop3:0x40
	v_dual_lshrrev_b32 v12, 30, v16 :: v_dual_ashrrev_i32 v9, 2, v9
	v_or_b32_e32 v13, s57, v17
	v_cmp_ne_u32_e32 vcc_lo, v11, v8
	v_mad_u32_u24 v38, 0x110, v17, v32
	v_mad_nc_i64_i32 v[6:7], v10, s12, v[6:7]
	v_sub_nc_u32_e32 v10, v11, v8
	v_dual_add_nc_u32 v12, v18, v12 :: v_dual_add_nc_u32 v11, s56, v9
	v_ashrrev_i32_e32 v9, 31, v13
	s_and_b32 s0, s55, vcc_lo
	v_dual_add_nc_u32 v8, s62, v10 :: v_dual_bitop2_b32 v10, -4, v12 bitop3:0x40
	v_cndmask_b32_e64 v16, 0, 1, s0
	v_dual_ashrrev_i32 v12, 2, v12 :: v_dual_lshrrev_b32 v19, 30, v9
	v_mad_nc_i64_i32 v[8:9], v8, s52, v[0:1]
	v_cmp_ne_u32_e32 vcc_lo, v18, v10
	v_sub_nc_u32_e32 v11, v11, v16
	v_dual_add_nc_u32 v12, s56, v12 :: v_dual_sub_nc_u32 v10, v18, v10
	v_add_nc_u32_e32 v16, v13, v19
	s_and_b32 s0, s55, vcc_lo
	v_min_u32_e32 v41, s1, v30
	v_cndmask_b32_e64 v18, 0, 1, s0
	v_mad_nc_i64_i32 v[8:9], v11, s12, v[8:9]
	v_dual_add_nc_u32 v10, s62, v10 :: v_dual_bitop2_b32 v19, -4, v16 bitop3:0x40
	v_or_b32_e32 v11, 18, v26
	v_dual_sub_nc_u32 v18, v12, v18 :: v_dual_ashrrev_i32 v12, 2, v16
	v_cmp_ne_u32_e32 vcc_lo, v13, v19
	v_sub_nc_u32_e32 v16, v13, v19
	v_min_u32_e32 v20, s1, v11
	v_mad_nc_i64_i32 v[10:11], v10, s52, v[0:1]
	v_add_nc_u32_e32 v19, s56, v12
	s_and_b32 s0, s55, vcc_lo
	v_add_nc_u32_e32 v12, s62, v16
	v_cndmask_b32_e64 v22, 0, 1, s0
	v_or_b32_e32 v16, s57, v20
	v_mad_u32_u24 v39, 0x110, v20, v32
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[60:61]
	v_mad_nc_i64_i32 v[10:11], v18, s12, v[10:11]
	v_dual_sub_nc_u32 v18, v19, v22 :: v_dual_ashrrev_i32 v23, 31, v16
	v_mad_nc_i64_i32 v[12:13], v12, s52, v[0:1]
	v_or_b32_e32 v22, s57, v21
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[60:61]
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[60:61]
	v_lshrrev_b32_e32 v19, 30, v23
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[60:61]
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[60:61]
	v_or_b32_e32 v30, s57, v41
	v_mad_nc_i64_i32 v[12:13], v18, s12, v[12:13]
	v_dual_ashrrev_i32 v18, 31, v22 :: v_dual_add_nc_u32 v14, v16, v19
	v_mad_u32_u24 v41, 0x110, v41, v32
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshrrev_b32_e32 v17, 30, v18
	v_and_b32_e32 v15, -4, v14
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[60:61]
	v_dual_add_nc_u32 v17, v22, v17 :: v_dual_bitop2_b32 v18, 14, v26 bitop3:0x54
	v_dual_sub_nc_u32 v19, v16, v15 :: v_dual_ashrrev_i32 v14, 2, v14
	v_cmp_ne_u32_e32 vcc_lo, v16, v15
	v_and_b32_e32 v16, -4, v17
	v_min_u32_e32 v23, s1, v18
	v_dual_ashrrev_i32 v17, 2, v17 :: v_dual_add_nc_u32 v18, s56, v14
	v_add_nc_u32_e32 v14, s62, v19
	s_and_b32 s0, s55, vcc_lo
	v_dual_sub_nc_u32 v25, v22, v16 :: v_dual_bitop2_b32 v19, s57, v23 bitop3:0x54
	v_cmp_ne_u32_e32 vcc_lo, v22, v16
	v_cndmask_b32_e64 v24, 0, 1, s0
	v_mad_nc_i64_i32 v[14:15], v14, s52, v[0:1]
	v_dual_ashrrev_i32 v27, 31, v19 :: v_dual_add_nc_u32 v22, s56, v17
	v_add_nc_u32_e32 v16, s62, v25
	s_and_b32 s0, s55, vcc_lo
	v_mad_u32_u24 v42, 0x110, v23, v32
	v_lshrrev_b32_e32 v25, 30, v27
	s_set_vgpr_msb 4
	v_or_b32_e32 v27, 12, v140 /*v396*/
	v_cndmask_b32_e64 v28, 0, 1, s0
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v25, v19, v25
	v_min_u32_e32 v27, s1, v27
	v_mad_nc_i64_i32 v[16:17], v16, s52, v[0:1]
	v_sub_nc_u32_e32 v18, v18, v24
	v_dual_sub_nc_u32 v22, v22, v28 :: v_dual_bitop2_b32 v20, -4, v25 bitop3:0x40
	v_or_b32_e32 v24, s57, v27
	v_or_b32_e32 v28, 10, v26
	v_mad_nc_i64_i32 v[14:15], v18, s12, v[14:15]
	v_ashrrev_i32_e32 v18, 2, v25
	v_mad_nc_i64_i32 v[16:17], v22, s12, v[16:17]
	v_sub_nc_u32_e32 v22, v19, v20
	v_ashrrev_i32_e32 v25, 31, v24
	v_cmp_ne_u32_e32 vcc_lo, v19, v20
	v_add_nc_u32_e32 v20, s56, v18
	v_mad_u32_u24 v44, 0x110, v27, v32
	v_dual_add_nc_u32 v18, s62, v22 :: v_dual_lshrrev_b32 v22, 30, v25
	v_min_u32_e32 v25, s1, v28
	s_and_b32 s0, s55, vcc_lo
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[60:61]
	v_cndmask_b32_e64 v28, 0, 1, s0
	v_dual_add_nc_u32 v22, v24, v22 :: v_dual_bitop2_b32 v29, s57, v25 bitop3:0x54
	v_mad_nc_i64_i32 v[18:19], v18, s52, v[0:1]
	v_mad_u32_u24 v45, 0x110, v25, v32
	v_dual_sub_nc_u32 v20, v20, v28 :: v_dual_bitop2_b32 v21, -4, v22 bitop3:0x40
	v_ashrrev_i32_e32 v28, 31, v29
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[60:61]
	v_mad_nc_i64_i32 v[18:19], v20, s12, v[18:19]
	v_dual_ashrrev_i32 v20, 2, v22 :: v_dual_sub_nc_u32 v22, v24, v21
	v_lshrrev_b32_e32 v28, 30, v28
	v_cmp_ne_u32_e32 vcc_lo, v24, v21
	v_dual_add_nc_u32 v24, s56, v20 :: v_dual_add_nc_u32 v20, s62, v22
	v_add_nc_u32_e32 v22, v29, v28
	s_and_b32 s0, s55, vcc_lo
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[60:61]
	v_cndmask_b32_e64 v28, 0, 1, s0
	v_mad_nc_i64_i32 v[20:21], v20, s52, v[0:1]
	v_and_b32_e32 v23, -4, v22
	v_dual_ashrrev_i32 v22, 2, v22 :: v_dual_sub_nc_u32 v24, v24, v28
	v_sub_nc_u32_e32 v28, v29, v23
	v_cmp_ne_u32_e32 vcc_lo, v29, v23
	v_or_b32_e32 v29, 6, v26
	v_mad_nc_i64_i32 v[20:21], v24, s12, v[20:21]
	v_dual_add_nc_u32 v24, s56, v22 :: v_dual_add_nc_u32 v22, s62, v28
	s_and_b32 s0, s55, vcc_lo
	v_lshrrev_b32_e32 v28, 30, v31
	v_cndmask_b32_e64 v31, 0, 1, s0
	v_min_u32_e32 v43, s1, v29
	v_mad_nc_i64_i32 v[22:23], v22, s52, v[0:1]
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[60:61]
	v_dual_add_nc_u32 v28, v30, v28 :: v_dual_sub_nc_u32 v24, v24, v31
	s_set_vgpr_msb 4
	v_or_b32_e32 v31, 4, v140 /*v396*/
	s_set_vgpr_msb 0x400
	v_and_b32_e32 v27, -4, v28
	v_mad_nc_i64_i32 v[22:23], v24, s12, v[22:23]
	v_dual_ashrrev_i32 v24, 2, v28 :: v_dual_bitop2_b32 v29, s57, v43 bitop3:0x54
	v_min_u32_e32 v46, s1, v31
	v_sub_nc_u32_e32 v25, v30, v27
	v_cmp_ne_u32_e32 vcc_lo, v30, v27
	v_dual_add_nc_u32 v27, s56, v24 :: v_dual_ashrrev_i32 v28, 31, v29
	v_dual_add_nc_u32 v24, s62, v25 :: v_dual_bitop2_b32 v31, s57, v46 bitop3:0x54
	s_and_b32 s0, s55, vcc_lo
	v_mad_u32_u24 v43, 0x110, v43, v32
	v_lshrrev_b32_e32 v28, 30, v28
	v_cndmask_b32_e64 v30, 0, 1, s0
	v_mad_nc_i64_i32 v[24:25], v24, s52, v[0:1]
	v_mad_u32_u24 v46, 0x110, v46, v32
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[60:61]
	v_dual_add_nc_u32 v28, v29, v28 :: v_dual_sub_nc_u32 v27, v27, v30
	v_or_b32_e32 v26, 2, v26
	v_and_b32_e32 v30, -4, v28
	v_dual_ashrrev_i32 v28, 2, v28 :: v_dual_ashrrev_i32 v47, 31, v31
	v_mad_nc_i64_i32 v[24:25], v27, s12, v[24:25]
	v_min_u32_e32 v48, s1, v26
	v_cmp_ne_u32_e32 vcc_lo, v29, v30
	v_dual_add_nc_u32 v26, s56, v28 :: v_dual_lshrrev_b32 v27, 30, v47
	s_set_vgpr_msb 4
	v_min_i32_e32 v47, s1, v140 /*v396*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v28, s57, v48
	s_and_b32 s0, s55, vcc_lo
	v_sub_nc_u32_e32 v29, v29, v30
	v_cndmask_b32_e64 v49, 0, 1, s0
	v_dual_ashrrev_i32 v50, 31, v28 :: v_dual_bitop2_b32 v30, s57, v47 bitop3:0x54
	v_add_nc_u32_e32 v27, v31, v27
	v_mad_u32_u24 v47, 0x110, v47, v32
	v_sub_nc_u32_e32 v49, v26, v49
	v_dual_add_nc_u32 v26, s62, v29 :: v_dual_lshrrev_b32 v50, 30, v50
	v_dual_ashrrev_i32 v29, 31, v30 :: v_dual_bitop2_b32 v51, -4, v27 bitop3:0x40
	v_ashrrev_i32_e32 v52, 2, v27
	v_mad_nc_i64_i32 v[26:27], v26, s52, v[0:1]
	v_dual_add_nc_u32 v50, v28, v50 :: v_dual_lshrrev_b32 v29, 30, v29
	v_cmp_ne_u32_e32 vcc_lo, v31, v51
	v_dual_add_nc_u32 v52, s56, v52 :: v_dual_sub_nc_u32 v31, v31, v51
	v_dual_add_nc_u32 v29, v30, v29 :: v_dual_bitop2_b32 v51, -4, v50 bitop3:0x40
	s_and_b32 s0, s55, vcc_lo
	v_dual_ashrrev_i32 v50, 2, v50 :: v_dual_add_nc_u32 v31, s62, v31
	v_cndmask_b32_e64 v53, 0, 1, s0
	v_and_b32_e32 v54, -4, v29
	v_cmp_ne_u32_e32 vcc_lo, v28, v51
	v_dual_sub_nc_u32 v28, v28, v51 :: v_dual_ashrrev_i32 v29, 2, v29
	v_add_nc_u32_e32 v50, s56, v50
	v_sub_nc_u32_e32 v51, v30, v54
	v_cmp_ne_u32_e64 s0, v30, v54
	v_dual_add_nc_u32 v54, s62, v28 :: v_dual_add_nc_u32 v55, s56, v29
	v_mad_nc_i64_i32 v[30:31], v31, s52, v[0:1]
	v_add_nc_u32_e32 v28, s62, v51
	s_and_b32 s0, s55, s0
	v_mad_nc_i64_i32 v[26:27], v49, s12, v[26:27]
	v_cndmask_b32_e64 v51, 0, 1, s0
	s_and_b32 s0, s55, vcc_lo
	v_mad_nc_i64_i32 v[28:29], v28, s52, v[0:1]
	v_cndmask_b32_e64 v56, 0, 1, s0
	v_mad_nc_i64_i32 v[0:1], v54, s52, v[0:1]
	v_sub_nc_u32_e32 v51, v55, v51
	v_mad_u32_u24 v32, 0x110, v48, v32
	v_lshl_add_u64 v[26:27], v[26:27], 1, s[60:61]
	v_dual_sub_nc_u32 v49, v50, v56 :: v_dual_sub_nc_u32 v50, v52, v53
	v_mad_nc_i64_i32 v[28:29], v51, s12, v[28:29]
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[60:61]
	v_mad_nc_i64_i32 v[0:1], v49, s12, v[0:1]
	v_mad_nc_i64_i32 v[30:31], v50, s12, v[30:31]
	v_lshl_add_u64 v[28:29], v[28:29], 1, s[60:61]
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[60:61]
	v_lshl_add_u64 v[30:31], v[30:31], 1, s[60:61]
	s_wait_alu depctr_va_vdst(2)
	global_store_async_from_lds_b128 v[28:29], v47, off
	s_wait_alu depctr_va_vdst(1)
	global_store_async_from_lds_b128 v[0:1], v32, off
	s_wait_alu depctr_va_vdst(0)
	global_store_async_from_lds_b128 v[30:31], v46, off
	global_store_async_from_lds_b128 v[26:27], v43, off
	global_store_async_from_lds_b128 v[24:25], v41, off
	global_store_async_from_lds_b128 v[22:23], v45, off
	global_store_async_from_lds_b128 v[20:21], v44, off
	global_store_async_from_lds_b128 v[18:19], v42, off
	global_store_async_from_lds_b128 v[16:17], v40, off
	global_store_async_from_lds_b128 v[14:15], v39, off
	global_store_async_from_lds_b128 v[12:13], v38, off
	global_store_async_from_lds_b128 v[10:11], v37, off
	global_store_async_from_lds_b128 v[8:9], v36, off
	global_store_async_from_lds_b128 v[6:7], v35, off
	global_store_async_from_lds_b128 v[2:3], v33, off
	global_store_async_from_lds_b128 v[4:5], v34, off
.LBB0_50:
	s_set_vgpr_msb 4
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v64 /*v320*/
	v_cmp_gt_f32_e64 s0, 0x800000, v65 /*v321*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_lshlrev_b32 v0, 2, v137 /*v393*/ :: v_dual_add_nc_u32 v5, s56, v138 /*v394*/
	v_cmp_gt_i32_e64 s1, s33, v139 /*v395*/
	v_cndmask_b32_e64 v3, 0, 32, vcc_lo
	v_cndmask_b32_e64 v4, 0, 32, s0
	v_cndmask_b32_e64 v1, 0, 0x42000000, vcc_lo
	v_cndmask_b32_e64 v2, 0, 0x42000000, s0
	v_mul_lo_u32 v5, v5, s53
	s_set_vgpr_msb 0x401
	v_ldexp_f32 v3, v64 /*v320*/, v3
	v_ldexp_f32 v4, v65 /*v321*/, v4
	s_set_vgpr_msb 0x104
	v_cmp_eq_u32_e32 vcc_lo, 0, v140 /*v396*/
	v_cmp_gt_i32_e64 s0, s33, v138 /*v394*/
	s_lshl_b32 s3, s58, 25
	v_log_f32_e32 v3, v3
	v_log_f32_e32 v4, v4
	s_lshr_b64 s[6:7], s[58:59], 7
	s_and_b32 s0, vcc_lo, s0
	s_and_b32 vcc_lo, vcc_lo, s1
	s_or_b64 s[4:5], s[34:35], s[2:3]
	s_set_vgpr_msb 0x400
	v_sub_f32_e32 v1, v3, v1
	s_set_vgpr_msb 1
	v_sub_nc_u32_e32 v0, v136 /*v392*/, v0
	s_set_vgpr_msb 0x100
	v_dual_sub_f32 v2, v4, v2 :: v_dual_mul_f32 v1, 0x3f317218, v1
	v_add_nc_u32_e32 v0, s62, v0
	s_set_vgpr_msb 4
	v_add_nc_u32_e32 v6, s56, v139 /*v395*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v2, 0x3f317218, v2
	s_set_vgpr_msb 4
	v_add_f32_e32 v1, v1, v133 /*v389*/
	v_mul_lo_u32 v0, v0, s54
	v_mul_lo_u32 v6, v6, s53
	v_add_f32_e32 v2, v2, v135 /*v391*/
	s_set_vgpr_msb 0x400
	v_add_lshl_u32 v3, v5, v0, 2
	v_add_lshl_u32 v0, v6, v0, 2
	v_cndmask_b32_e64 v3, 0x7fffffff, v3, s0
	v_cndmask_b32_e32 v0, 0x7fffffff, v0, vcc_lo
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	buffer_store_b32 v1, v3, s[4:7], null offen
	buffer_store_b32 v2, v0, s[4:7], null offen
.LBB0_51:
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	kn_fmha_fwd_prefill_a16w16_m32x8_thd_0, .Lfunc_end0-kn_fmha_fwd_prefill_a16w16_m32x8_thd_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kn_fmha_fwd_prefill_a16w16_m32x8_thd_0
		.amdhsa_group_segment_fixed_size 327680
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 440
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 513
		.amdhsa_next_free_sgpr 102
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-kn_fmha_fwd_prefill_a16w16_m32x8_thd_0)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.num_vgpr, 456
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.num_agpr, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.numbered_sgpr, 102
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.num_named_barrier, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.private_seg_size, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.uses_vcc, 1
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.uses_flat_scratch, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.has_dyn_sized_stack, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.has_recursion, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_thd_0.has_indirect_call, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .offset:         120
        .size:           4
        .value_kind:     by_value
      - .offset:         124
        .size:           4
        .value_kind:     by_value
      - .offset:         128
        .size:           4
        .value_kind:     by_value
      - .offset:         132
        .size:           4
        .value_kind:     by_value
      - .offset:         136
        .size:           4
        .value_kind:     by_value
      - .offset:         140
        .size:           4
        .value_kind:     by_value
      - .offset:         144
        .size:           4
        .value_kind:     by_value
      - .offset:         148
        .size:           4
        .value_kind:     by_value
      - .offset:         152
        .size:           4
        .value_kind:     by_value
      - .offset:         156
        .size:           4
        .value_kind:     by_value
      - .offset:         160
        .size:           4
        .value_kind:     by_value
      - .offset:         164
        .size:           4
        .value_kind:     by_value
      - .offset:         168
        .size:           4
        .value_kind:     by_value
      - .offset:         172
        .size:           4
        .value_kind:     by_value
      - .offset:         176
        .size:           4
        .value_kind:     by_value
      - .offset:         180
        .size:           4
        .value_kind:     by_value
      - .offset:         184
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         188
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         192
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         196
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         198
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         200
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         202
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         204
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         206
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         224
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         232
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         240
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         248
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 327680
    .kernarg_segment_align: 8
    .kernarg_segment_size: 440
    .max_flat_workgroup_size: 128
    .name:           kn_fmha_fwd_prefill_a16w16_m32x8_thd_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     104
    .sgpr_spill_count: 0
    .symbol:         kn_fmha_fwd_prefill_a16w16_m32x8_thd_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     456
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
