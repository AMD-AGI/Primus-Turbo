	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	k_delta_bshd_0
	.p2align	8
	.type	k_delta_bshd_0,@function
k_delta_bshd_0:
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	v_nop
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x3
	s_load_b96 s[16:18], s[0:1], 0x84 nv
	s_load_b64 s[2:3], s[0:1], 0x60 nv
	s_load_b64 s[4:5], s[0:1], 0x0 nv
	s_load_b64 s[12:13], s[0:1], 0x30 nv
	s_wait_xcnt 0x0
	s_bfe_u32 s1, ttmp6, 0x4000c
	s_and_b32 s6, ttmp6, 15
	s_add_co_i32 s1, s1, 1
	s_mov_b32 s0, 0
	s_mul_i32 s1, ttmp9, s1
	s_getreg_b32 s14, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s15, s6, s1
	v_lshlrev_b32_e32 v1, 4, v0
	s_wait_kmcnt 0x0
	s_lshl_b32 s6, s18, 8
	s_lshl_b32 s10, s18, 2
	s_lshl_b32 s1, s18, 27
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s11, s10, 31
	s_or_b64 s[8:9], s[2:3], s[0:1]
	s_lshr_b64 s[6:7], s[6:7], 7
	s_lshr_b64 s[10:11], s[10:11], 7
	s_cmp_eq_u32 s14, 0
	s_mov_b32 s14, s6
	s_cselect_b32 s0, ttmp9, s15
	s_mov_b32 s15, s7
	v_lshl_or_b32 v1, s0, 13, v1
	s_mul_i32 s3, s17, s16
	buffer_load_b128 v[6:9], v1, s[12:15], null offen
	buffer_load_b128 v[10:13], v1, s[4:7], null offen offset:4096
	buffer_load_b128 v[14:17], v1, s[12:15], null offen offset:4096
	buffer_load_b128 v[2:5], v1, s[4:7], null offen
	s_abs_i32 s1, s3
	s_wait_xcnt 0x0
	s_bfe_i32 s5, s0, 0x1001a
	s_cvt_f32_u32 s2, s1
	v_lshrrev_b32_e32 v1, 4, v0
	s_lshl_b32 s4, s0, 5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_s_rcp_f32 s2, s2
	v_or_b32_e32 v1, s4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_or_b32_e32 v32, 16, v1
	s_mul_f32 s0, s2, 0x4f7ffffe
	s_sub_co_i32 s2, 0, s1
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_cvt_u32_f32 s0, s0
	v_dual_add_nc_u32 v19, s5, v32 :: v_dual_add_nc_u32 v18, s5, v1
	v_and_b32_e32 v0, 15, v0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s2, s2, s0
	s_mul_hi_u32 s2, s0, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_xor_b32_e32 v18, s5, v18
	s_add_co_i32 s0, s0, s2
	v_xor_b32_e32 v19, s5, v19
	s_ashr_i32 s2, s3, 31
	v_mul_hi_u32 v20, v18, s0
	s_xor_b32 s5, s5, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_hi_u32 v21, v19, s0
	v_mul_lo_u32 v22, v20, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v23, v21, s1
	v_dual_sub_nc_u32 v18, v18, v22 :: v_dual_add_nc_u32 v22, 1, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_sub_nc_u32 v19, v19, v23 :: v_dual_add_nc_u32 v23, 1, v21
	v_subrev_nc_u32_e32 v24, s1, v18
	v_cmp_le_u32_e32 vcc_lo, s1, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_le_u32_e64 s0, s1, v19
	v_dual_cndmask_b32 v20, v20, v22, vcc_lo :: v_dual_cndmask_b32 v21, v21, v23, s0
	v_subrev_nc_u32_e32 v22, s1, v19
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v23, 1, v20 :: v_dual_cndmask_b32 v19, v19, v22, s0
	v_cmp_le_u32_e32 vcc_lo, s1, v18
	v_add_nc_u32_e32 v22, 1, v21
	v_cmp_gt_i32_e64 s0, s18, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v18, v20, v23, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s1, v19
	s_xor_b32 s1, s3, s4
	v_cndmask_b32_e32 v19, v21, v22, vcc_lo
	v_xor_b32_e32 v18, s5, v18
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v19, s5, v19
	v_subrev_nc_u32_e32 v0, s5, v18
	s_and_b32 s0, vcc_lo, s0
	s_cmp_lt_i32 s1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_subrev_nc_u32_e32 v20, s5, v19
	v_mul_lo_u32 v0, v0, s3
	s_cselect_b32 s4, -1, 0
	s_abs_i32 s6, s17
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v20, v20, s3
	s_cvt_f32_u32 s7, s6
	v_cmp_ne_u32_e64 s1, v1, v0
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_s_rcp_f32 s7, s7
	v_cmp_ne_u32_e64 s2, v32, v20
	s_and_b32 s1, s4, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	v_subrev_co_ci_u32_e64 v33, null, s5, v18, s1
	s_and_b32 s1, s4, s2
	s_sub_co_i32 s2, 0, s6
	v_subrev_co_ci_u32_e64 v34, null, s5, v19, s1
	v_mul_lo_u32 v0, v33, s3
	s_mul_f32 s1, s7, 0x4f7ffffe
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_mul_lo_u32 v18, v34, s3
	s_ashr_i32 s3, s17, 31
	s_cvt_u32_f32 s1, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(SALU_CYCLE_2)
	v_sub_nc_u32_e32 v35, v1, v0
	s_mul_i32 s2, s2, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_nc_u32_e32 v36, v32, v18
	s_mul_hi_u32 s2, s1, s2
	v_ashrrev_i32_e32 v22, 31, v35
	s_add_co_i32 s1, s1, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_nc_u32 v1, 0, v36 :: v_dual_sub_nc_u32 v0, 0, v35
	v_max_i32_e32 v1, v1, v36
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max_i32_e32 v0, v0, v35
	v_mul_hi_u32 v19, v1, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_hi_u32 v18, v0, s1
	v_mul_lo_u32 v21, v19, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v20, v18, s6
	v_dual_add_nc_u32 v23, 1, v18 :: v_dual_sub_nc_u32 v1, v1, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_sub_nc_u32_e32 v0, v0, v20
	v_ashrrev_i32_e32 v20, 31, v36
	v_cmp_le_u32_e64 s2, s6, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_subrev_nc_u32_e32 v24, s6, v0
	v_cmp_le_u32_e64 s1, s6, v0
	v_dual_add_nc_u32 v21, 1, v19 :: v_dual_bitop2_b32 v37, s3, v20 bitop3:0x14
	v_xor_b32_e32 v20, s17, v36
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_cndmask_b32 v0, v0, v24, s1 :: v_dual_cndmask_b32 v18, v18, v23, s1
	v_cndmask_b32_e64 v19, v19, v21, s2
	v_subrev_nc_u32_e32 v23, s6, v1
	v_xor_b32_e32 v22, s3, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_le_u32_e64 s1, s6, v0
	v_dual_add_nc_u32 v21, 1, v18 :: v_dual_cndmask_b32 v1, v1, v23, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_nc_u32 v23, 1, v19 :: v_dual_cndmask_b32 v0, v18, v21, s1
	v_cmp_le_u32_e64 s1, s6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_cndmask_b32 v1, v19, v23, s1 :: v_dual_bitop2_b32 v0, v0, v22 bitop3:0x14
	v_xor_b32_e32 v19, s17, v35
	v_dual_sub_nc_u32 v1, v0, v22 :: v_dual_bitop2_b32 v38, v1, v37 bitop3:0x14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, 0, v19
	v_sub_nc_u32_e32 v18, v38, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s17
	v_mul_lo_u32 v18, v18, s17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_ne_u32_e64 s2, v35, v1
	v_cmp_ne_u32_e64 s4, v36, v18
	s_and_b32 s1, s2, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	v_sub_co_ci_u32_e64 v39, null, v0, v22, s1
	s_wait_loadcnt 0x3
	v_and_b32_e32 v23, 0xffff0000, v7
	v_lshlrev_b32_e32 v22, 16, v7
	v_and_b32_e32 v7, 0xffff0000, v8
	s_wait_loadcnt 0x0
	v_and_b32_e32 v1, 0xffff0000, v2
	v_lshlrev_b32_e32 v0, 16, v2
	v_cmp_gt_i32_e64 s3, 0, v20
	v_and_b32_e32 v19, 0xffff0000, v3
	v_dual_lshlrev_b32 v18, 16, v3 :: v_dual_lshlrev_b32 v2, 16, v4
	v_and_b32_e32 v3, 0xffff0000, v4
	v_and_b32_e32 v21, 0xffff0000, v5
	v_dual_lshlrev_b32 v20, 16, v5 :: v_dual_lshlrev_b32 v4, 16, v6
	v_and_b32_e32 v5, 0xffff0000, v6
	v_lshlrev_b32_e32 v6, 16, v8
	v_and_b32_e32 v25, 0xffff0000, v9
	v_dual_lshlrev_b32 v24, 16, v9 :: v_dual_lshlrev_b32 v8, 16, v10
	v_and_b32_e32 v9, 0xffff0000, v10
	v_and_b32_e32 v27, 0xffff0000, v11
	v_dual_lshlrev_b32 v26, 16, v11 :: v_dual_lshlrev_b32 v10, 16, v12
	v_and_b32_e32 v11, 0xffff0000, v12
	v_and_b32_e32 v29, 0xffff0000, v13
	v_dual_lshlrev_b32 v28, 16, v13 :: v_dual_lshlrev_b32 v12, 16, v14
	v_and_b32_e32 v13, 0xffff0000, v14
	v_and_b32_e32 v31, 0xffff0000, v15
	v_dual_lshlrev_b32 v30, 16, v15 :: v_dual_lshlrev_b32 v14, 16, v16
	v_pk_mul_f32 v[0:1], v[0:1], v[4:5]
	s_delay_alu instid0(VALU_DEP_4)
	v_pk_mul_f32 v[4:5], v[8:9], v[12:13]
	v_and_b32_e32 v15, 0xffff0000, v16
	v_pk_mul_f32 v[12:13], v[18:19], v[22:23]
	v_pk_mul_f32 v[18:19], v[26:27], v[30:31]
	v_pk_add_f32 v[0:1], v[0:1], 0 op_sel_hi:[1,0]
	v_pk_add_f32 v[4:5], v[4:5], 0 op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[6:7]
	v_pk_mul_f32 v[6:7], v[10:11], v[14:15]
	v_and_b32_e32 v9, 0xffff0000, v17
	v_pk_add_f32 v[0:1], v[12:13], v[0:1]
	v_pk_add_f32 v[4:5], v[18:19], v[4:5]
	v_lshlrev_b32_e32 v8, 16, v17
	s_and_b32 s1, s4, s3
	v_pk_mul_f32 v[10:11], v[20:21], v[24:25]
	v_pk_add_f32 v[0:1], v[2:3], v[0:1]
	v_pk_add_f32 v[2:3], v[6:7], v[4:5]
	v_sub_nc_u32_e32 v4, v33, v39
	v_pk_mul_f32 v[8:9], v[28:29], v[8:9]
	v_sub_co_ci_u32_e64 v12, null, v38, v37, s1
	v_pk_add_f32 v[0:1], v[10:11], v[0:1]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_u32 v4, v4, s17, v35
	v_pk_add_f32 v[2:3], v[8:9], v[2:3]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_sub_nc_u32_e32 v5, v34, v12
	v_cmp_gt_i32_e64 s1, s18, v32
	v_dual_add_f32 v0, v0, v1 :: v_dual_add_f32 v1, v2, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mad_u32 v5, v5, s17, v36
	v_mad_u32 v2, v4, s16, v39
	v_add_f32_dpp v0, v0, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_and_b32 vcc_lo, vcc_lo, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_dpp v0, v0, v0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mad_u32 v3, v5, s16, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v2, s18, v2, s0
	v_add_f32_dpp v1, v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_xmask:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_f32_dpp v1, v1, v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cndmask_b32_e32 v3, s18, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_dpp v0, v0, v0 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_xmask:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v3, 2, v3
	v_add_f32_dpp v1, v1, v1 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_clause 0x1
	buffer_store_b32 v0, v2, s[8:11], null offen
	buffer_store_b32 v1, v3, s[8:11], null offen
	s_endpgm
.Lfunc_end0:
	.size	k_delta_bshd_0, .Lfunc_end0-k_delta_bshd_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_delta_bshd_0
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 144
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 40
		.amdhsa_next_free_sgpr 19
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-k_delta_bshd_0)<<4)&4080)>>4
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

	.set .Lk_delta_bshd_0.num_vgpr, 40
	.set .Lk_delta_bshd_0.num_agpr, 0
	.set .Lk_delta_bshd_0.numbered_sgpr, 19
	.set .Lk_delta_bshd_0.num_named_barrier, 0
	.set .Lk_delta_bshd_0.private_seg_size, 0
	.set .Lk_delta_bshd_0.uses_vcc, 1
	.set .Lk_delta_bshd_0.uses_flat_scratch, 0
	.set .Lk_delta_bshd_0.has_dyn_sized_stack, 0
	.set .Lk_delta_bshd_0.has_recursion, 0
	.set .Lk_delta_bshd_0.has_indirect_call, 0
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
        .size:           40
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           40
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .offset:         104
        .size:           28
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .max_flat_workgroup_size: 256
    .name:           k_delta_bshd_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     21
    .sgpr_spill_count: 0
    .symbol:         k_delta_bshd_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     40
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
