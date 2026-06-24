.amdgcn_target "amdgcn-amd-amdhsa--gfx950"

.text
.globl grouped_variable_k_dot_scaled_kernel
.p2align 8
.type grouped_variable_k_dot_scaled_kernel,@function
grouped_variable_k_dot_scaled_kernel:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .L0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
.L0:
	s_load_dword s17, s[0:1], 0x38
	s_mov_b64 s[20:21], s[6:7]
	s_mov_b64 s[24:25], s[2:3]
	s_cmpk_gt_i32 s16, 0xff
	s_cbranch_scc1 .L1
	s_ashr_i32 s2, s16, 31
	s_lshr_b32 s2, s2, 29
	s_add_i32 s2, s16, s2
	s_ashr_i32 s3, s2, 3
	s_and_b32 s6, s2, 0x7fffff8
	s_ashr_i32 s2, s2, 31
	s_lshr_b32 s2, s2, 27
	s_add_i32 s2, s3, s2
	s_sub_i32 s6, s16, s6
	s_andn2_b32 s2, s2, 31
	s_lshl_b32 s6, s6, 5
	s_sub_i32 s2, s3, s2
	s_add_i32 s16, s6, s2
.L1:
	s_add_i32 s2, s15, 0xff
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 24
	s_add_i32 s2, s2, s3
	s_ashr_i32 s33, s2, 8
	s_waitcnt lgkmcnt(0)
	s_add_i32 s2, s17, 0xff
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 24
	s_add_i32 s2, s2, s3
	s_ashr_i32 s2, s2, 8
	s_mul_i32 s34, s2, s33
	s_mul_i32 s14, s34, s14
	s_cmp_ge_i32 s16, s14
	s_cbranch_scc1 .L6
	s_load_dwordx4 s[28:31], s[0:1], 0x3c
	s_nop 0
	s_load_dword s0, s[10:11], 0x0
	s_load_dword s1, s[8:9], 0x0
	v_lshrrev_b32_e32 v2, 3, v0
	v_lshrrev_b32_e32 v3, 1, v0
	v_and_b32_e32 v1, 15, v0
	v_and_b32_e32 v2, 0x60, v2
	v_and_b32_e32 v3, 0x60, v3
	v_lshrrev_b32_e32 v4, 2, v0
	v_lshlrev_b32_e32 v96, 4, v1
	v_and_or_b32 v97, v0, 31, v2
	v_and_or_b32 v116, v4, 8, v3
	v_lshrrev_b32_e32 v108, 4, v0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s0
	v_mul_f32_e32 v4, s1, v4
	v_mad_u64_u32 v[124:125], s[0:1], s28, v108, v[96:97]
	v_mad_u64_u32 v[112:113], s[0:1], s29, v108, v[96:97]
	v_mul_f32_e32 v98, 0x3e800000, v4
	v_lshlrev_b32_e32 v4, 4, v0
	s_movk_i32 s0, 0xf0
	v_bitop3_b32 v4, v4, v0, s0 bitop3:0x78
	v_lshlrev_b32_e32 v5, 7, v0
	v_and_b32_e32 v0, 16, v0
	v_and_b32_e32 v5, 0x1700, v5
	v_lshlrev_b32_e32 v1, 3, v1
	v_or_b32_e32 v2, v2, v0
	v_or_b32_e32 v0, v3, v0
	v_or_b32_e32 v122, 16, v116
	v_or_b32_e32 v123, 0x80, v116
	v_or_b32_e32 v110, 0x90, v116
	v_bitop3_b32 v2, v5, v2, v1 bitop3:0x36
	v_bitop3_b32 v0, v5, v0, v1 bitop3:0x36
	v_mul_lo_u32 v1, s31, v97
	s_abs_i32 s44, s34
	v_lshl_add_u32 v3, s31, 7, v1
	v_add_u32_e32 v111, v1, v116
	v_add_u32_e32 v114, v1, v122
	v_add_u32_e32 v113, v1, v123
	v_add_u32_e32 v115, v1, v110
	v_cvt_f32_u32_e32 v1, s44
	s_lshl_b32 s35, s2, 2
	s_abs_i32 s45, s35
	v_add_u32_e32 v121, v3, v116
	v_rcp_iflag_f32_e32 v1, v1
	v_add_u32_e32 v125, v3, v122
	v_add_u32_e32 v126, v3, v123
	v_add_u32_e32 v127, v3, v110
	v_cvt_f32_u32_e32 v3, s45
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_sub_i32 s0, 0, s44
	v_rcp_iflag_f32_e32 v3, v3
	s_mov_b32 s27, 0x27000
	v_readfirstlane_b32 s1, v1
	s_mul_i32 s0, s0, s1
	v_mul_f32_e32 v1, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v1, v1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s47, s1, s0
	s_sub_i32 s0, 0, s45
	v_readfirstlane_b32 s1, v1
	s_mul_i32 s0, s0, s1
	s_mov_b32 s26, 0x7ffffffe
	s_and_b32 s5, s5, 0xffff
	s_mul_hi_u32 s0, s1, s0
	v_or_b32_e32 v109, 64, v108
	s_lshl_b32 s40, s28, 7
	s_lshl_b32 s41, s29, 7
	s_lshl_b32 s42, s28, 6
	s_lshl_b32 s43, s29, 6
	s_and_b32 s25, s25, 0xffff
	s_and_b32 s21, s21, 0xffff
	v_mov_b32_e32 v99, v98
	s_ashr_i32 s46, s34, 31
	s_bfe_i32 s48, s2, 0x1001d
	s_add_i32 s49, s1, s0
	v_bfrev_b32_e32 v117, 1
	v_add_u32_e32 v118, 0, v4
	v_add_u32_e32 v119, 0, v2
	v_add_u32_e32 v120, 0, v0
	s_mov_b32 s36, s4
	s_mov_b32 s37, s5
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	scratch_store_dword off, v97, off
	scratch_store_dword off, v121, off offset:4
	scratch_store_dword off, v125, off offset:8
	scratch_store_dword off, v126, off offset:12
	scratch_store_dword off, v127, off offset:16
	s_branch .L3
.L2:
	s_waitcnt vmcnt(4)
	v_or_b32_e32 v65, 0x80, v97
	s_mul_i32 s18, s18, s30
	v_or_b32_e32 v64, s19, v97
	v_or_b32_e32 v65, s19, v65
	v_or_b32_e32 v66, s50, v116
	s_mul_i32 s19, s19, s31
	s_add_i32 s18, s50, s18
	v_or_b32_e32 v67, s50, v122
	v_or_b32_e32 v68, s50, v123
	v_or_b32_e32 v69, s50, v110
	v_cmp_gt_i32_e64 s[10:11], s15, v64
	v_cmp_gt_i32_e64 s[8:9], s17, v66
	v_pk_mul_f32 v[0:1], v[98:99], v[0:1]
	v_pk_mul_f32 v[2:3], v[98:99], v[2:3]
	v_pk_mul_f32 v[4:5], v[98:99], v[4:5]
	v_pk_mul_f32 v[6:7], v[98:99], v[6:7]
	v_pk_mul_f32 v[8:9], v[98:99], v[8:9]
	v_pk_mul_f32 v[16:17], v[98:99], v[16:17]
	v_pk_mul_f32 v[32:33], v[98:99], v[32:33]
	s_add_i32 s50, s18, s19
	v_cvt_pk_bf16_f32 v0, v0, v1
	v_cvt_pk_bf16_f32 v1, v2, v3
	v_cvt_pk_bf16_f32 v2, v4, v5
	v_cvt_pk_bf16_f32 v3, v6, v7
	v_cvt_pk_bf16_f32 v4, v8, v9
	v_cvt_pk_bf16_f32 v8, v16, v17
	v_cvt_pk_bf16_f32 v16, v32, v33
	v_add_lshl_u32 v32, s50, v111, 1
	s_and_b64 s[18:19], s[10:11], s[8:9]
	v_cmp_gt_i32_e64 s[6:7], s17, v67
	v_pk_mul_f32 v[10:11], v[98:99], v[10:11]
	v_pk_mul_f32 v[12:13], v[98:99], v[12:13]
	v_pk_mul_f32 v[14:15], v[98:99], v[14:15]
	v_permlane32_swap_b32_e32 v0, v2
	v_permlane32_swap_b32_e32 v1, v3
	v_cndmask_b32_e64 v32, v117, v32, s[18:19]
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_cvt_pk_bf16_f32 v5, v10, v11
	v_cvt_pk_bf16_f32 v6, v12, v13
	v_cvt_pk_bf16_f32 v7, v14, v15
	buffer_store_dwordx4 v[0:3], v32, s[20:23], 0 offen
	s_and_b64 s[18:19], s[10:11], s[6:7]
	v_cmp_gt_i32_e64 s[2:3], s17, v68
	v_add_lshl_u32 v0, s50, v114, 1
	v_pk_mul_f32 v[18:19], v[98:99], v[18:19]
	v_pk_mul_f32 v[20:21], v[98:99], v[20:21]
	v_pk_mul_f32 v[22:23], v[98:99], v[22:23]
	v_permlane32_swap_b32_e32 v4, v6
	v_permlane32_swap_b32_e32 v5, v7
	v_cndmask_b32_e64 v0, v117, v0, s[18:19]
	v_cvt_pk_bf16_f32 v9, v18, v19
	v_cvt_pk_bf16_f32 v10, v20, v21
	v_cvt_pk_bf16_f32 v11, v22, v23
	buffer_store_dwordx4 v[4:7], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, s50, v113, 1
	s_and_b64 s[18:19], s[10:11], s[2:3]
	v_cmp_gt_i32_e64 s[0:1], s17, v69
	v_pk_mul_f32 v[24:25], v[98:99], v[24:25]
	v_pk_mul_f32 v[26:27], v[98:99], v[26:27]
	v_pk_mul_f32 v[28:29], v[98:99], v[28:29]
	v_pk_mul_f32 v[30:31], v[98:99], v[30:31]
	v_permlane32_swap_b32_e32 v8, v10
	v_permlane32_swap_b32_e32 v9, v11
	v_cndmask_b32_e64 v0, v117, v0, s[18:19]
	v_cvt_pk_bf16_f32 v12, v24, v25
	v_cvt_pk_bf16_f32 v13, v26, v27
	v_cvt_pk_bf16_f32 v14, v28, v29
	v_cvt_pk_bf16_f32 v15, v30, v31
	buffer_store_dwordx4 v[8:11], v0, s[20:23], 0 offen
	v_add_lshl_u32 v0, s50, v115, 1
	s_and_b64 s[10:11], s[10:11], s[0:1]
	v_cmp_gt_i32_e32 vcc, s15, v65
	v_pk_mul_f32 v[34:35], v[98:99], v[34:35]
	v_pk_mul_f32 v[36:37], v[98:99], v[36:37]
	v_pk_mul_f32 v[38:39], v[98:99], v[38:39]
	v_permlane32_swap_b32_e32 v12, v14
	v_permlane32_swap_b32_e32 v13, v15
	v_cndmask_b32_e64 v0, v117, v0, s[10:11]
	v_cvt_pk_bf16_f32 v17, v34, v35
	v_cvt_pk_bf16_f32 v18, v36, v37
	v_cvt_pk_bf16_f32 v19, v38, v39
	buffer_store_dwordx4 v[12:15], v0, s[20:23], 0 offen
	s_waitcnt vmcnt(7)
	v_add_lshl_u32 v0, s50, v121, 1
	s_and_b64 s[8:9], vcc, s[8:9]
	v_pk_mul_f32 v[40:41], v[98:99], v[40:41]
	v_pk_mul_f32 v[42:43], v[98:99], v[42:43]
	v_pk_mul_f32 v[44:45], v[98:99], v[44:45]
	v_pk_mul_f32 v[46:47], v[98:99], v[46:47]
	v_permlane32_swap_b32_e32 v16, v18
	v_permlane32_swap_b32_e32 v17, v19
	v_cndmask_b32_e64 v0, v117, v0, s[8:9]
	v_cvt_pk_bf16_f32 v20, v40, v41
	v_cvt_pk_bf16_f32 v21, v42, v43
	v_cvt_pk_bf16_f32 v22, v44, v45
	v_cvt_pk_bf16_f32 v23, v46, v47
	buffer_store_dwordx4 v[16:19], v0, s[20:23], 0 offen
	s_waitcnt vmcnt(7)
	v_add_lshl_u32 v0, s50, v125, 1
	s_and_b64 s[6:7], vcc, s[6:7]
	v_pk_mul_f32 v[48:49], v[98:99], v[48:49]
	v_pk_mul_f32 v[50:51], v[98:99], v[50:51]
	v_pk_mul_f32 v[52:53], v[98:99], v[52:53]
	v_pk_mul_f32 v[54:55], v[98:99], v[54:55]
	v_permlane32_swap_b32_e32 v20, v22
	v_permlane32_swap_b32_e32 v21, v23
	v_cndmask_b32_e64 v0, v117, v0, s[6:7]
	v_cvt_pk_bf16_f32 v24, v48, v49
	v_cvt_pk_bf16_f32 v25, v50, v51
	v_cvt_pk_bf16_f32 v26, v52, v53
	v_cvt_pk_bf16_f32 v27, v54, v55
	buffer_store_dwordx4 v[20:23], v0, s[20:23], 0 offen
	s_waitcnt vmcnt(7)
	v_add_lshl_u32 v0, s50, v126, 1
	s_and_b64 s[2:3], vcc, s[2:3]
	v_pk_mul_f32 v[56:57], v[98:99], v[56:57]
	v_pk_mul_f32 v[58:59], v[98:99], v[58:59]
	v_pk_mul_f32 v[60:61], v[98:99], v[60:61]
	v_pk_mul_f32 v[62:63], v[98:99], v[62:63]
	v_permlane32_swap_b32_e32 v24, v26
	v_permlane32_swap_b32_e32 v25, v27
	v_cndmask_b32_e64 v0, v117, v0, s[2:3]
	v_cvt_pk_bf16_f32 v28, v56, v57
	v_cvt_pk_bf16_f32 v29, v58, v59
	v_cvt_pk_bf16_f32 v30, v60, v61
	v_cvt_pk_bf16_f32 v31, v62, v63
	buffer_store_dwordx4 v[24:27], v0, s[20:23], 0 offen
	s_waitcnt vmcnt(7)
	v_add_lshl_u32 v0, s50, v127, 1
	s_and_b64 vcc, vcc, s[0:1]
	s_addk_i32 s16, 0x100
	v_permlane32_swap_b32_e32 v28, v30
	v_permlane32_swap_b32_e32 v29, v31
	v_cndmask_b32_e32 v0, v117, v0, vcc
	s_cmp_lt_i32 s16, s14
	buffer_store_dwordx4 v[28:31], v0, s[20:23], 0 offen
	s_cbranch_scc0 .L6
.L3:
	s_abs_i32 s1, s16
	s_mul_hi_u32 s2, s1, s47
	s_mul_i32 s3, s2, s44
	s_ashr_i32 s0, s16, 31
	s_sub_i32 s1, s1, s3
	s_xor_b32 s0, s0, s46
	s_add_i32 s3, s2, 1
	s_sub_i32 s6, s1, s44
	s_cmp_ge_u32 s1, s44
	s_cselect_b32 s2, s3, s2
	s_cselect_b32 s1, s6, s1
	s_add_i32 s3, s2, 1
	s_cmp_ge_u32 s1, s44
	s_cselect_b32 s1, s3, s2
	s_xor_b32 s1, s1, s0
	s_sub_i32 s18, s1, s0
	s_mul_i32 s0, s18, s34
	s_sub_i32 s0, s16, s0
	s_abs_i32 s2, s0
	s_mul_hi_u32 s3, s2, s49
	s_mul_i32 s6, s3, s45
	s_ashr_i32 s1, s0, 31
	s_sub_i32 s2, s2, s6
	s_xor_b32 s1, s1, s48
	s_add_i32 s6, s3, 1
	s_sub_i32 s7, s2, s45
	s_cmp_ge_u32 s2, s45
	s_cselect_b32 s3, s6, s3
	s_cselect_b32 s2, s7, s2
	s_add_i32 s6, s3, 1
	s_cmp_ge_u32 s2, s45
	s_cselect_b32 s2, s6, s3
	s_xor_b32 s2, s2, s1
	s_sub_i32 s1, s2, s1
	s_lshl_b32 s2, s1, 2
	s_sub_i32 s3, s33, s2
	s_min_i32 s3, s3, 4
	s_abs_i32 s6, s3
	v_cvt_f32_u32_e32 v0, s6
	s_mul_i32 s1, s1, s35
	s_sub_i32 s7, 0, s6
	s_sub_i32 s0, s0, s1
	v_rcp_iflag_f32_e32 v0, v0
	s_xor_b32 s1, s0, s3
	s_ashr_i32 s8, s1, 31
	s_abs_i32 s1, s0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v4, 0
	v_readfirstlane_b32 s9, v0
	s_mul_i32 s7, s7, s9
	s_mul_hi_u32 s7, s9, s7
	s_add_i32 s9, s9, s7
	s_mul_hi_u32 s7, s1, s9
	s_mul_i32 s9, s7, s6
	s_sub_i32 s1, s1, s9
	s_add_i32 s9, s7, 1
	s_sub_i32 s10, s1, s6
	s_cmp_ge_u32 s1, s6
	s_cselect_b32 s7, s9, s7
	s_cselect_b32 s1, s10, s1
	s_add_i32 s9, s7, 1
	s_cmp_ge_u32 s1, s6
	s_cselect_b32 s1, s9, s7
	s_xor_b32 s9, s1, s8
	s_sub_i32 s6, s9, s8
	s_mul_i32 s1, s6, s3
	s_sub_i32 s0, s0, s1
	s_ashr_i32 s19, s18, 31
	s_add_i32 s2, s0, s2
	s_lshl_b64 s[0:1], s[18:19], 3
	s_add_u32 s0, s12, s0
	s_addc_u32 s1, s13, s1
	v_mov_b32_e32 v0, 0
	global_load_dword v64, v0, s[0:1]
	s_add_i32 s0, s18, 1
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 3
	s_add_u32 s0, s12, s0
	s_addc_u32 s1, s13, s1
	global_load_dword v65, v0, s[0:1]
	s_lshl_b32 s19, s2, 8
	s_lshl_b32 s50, s6, 8
	v_or_b32_e32 v0, s19, v96
	v_cmp_gt_i32_e32 vcc, s15, v0
	v_or_b32_e32 v1, s50, v96
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s0, v64
	v_mul_lo_u32 v0, v64, s28
	v_add_u32_e32 v0, s19, v0
	v_add_u32_e32 v0, v0, v124
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s1, v65
	s_sub_i32 s6, s1, s0
	s_add_i32 s10, s6, 0x7f
	v_cmp_gt_i32_e64 s[2:3], s6, v108
	s_cmpk_gt_i32 s10, 0x7f
	s_cselect_b64 s[22:23], -1, 0
	s_and_b64 s[0:1], vcc, s[2:3]
	s_and_b64 s[0:1], s[22:23], s[0:1]
	v_cmp_gt_i32_e64 s[6:7], s6, v109
	v_cndmask_b32_e64 v2, v117, v0, s[0:1]
	v_cmp_gt_i32_e64 s[0:1], s17, v1
	v_mul_lo_u32 v1, v64, s29
	s_and_b64 s[52:53], vcc, s[6:7]
	v_add_u32_e32 v1, s50, v1
	v_add_u32_e32 v0, s42, v0
	s_and_b64 s[54:55], s[0:1], s[2:3]
	s_and_b64 s[2:3], s[22:23], s[52:53]
	v_add_u32_e32 v1, v1, v112
	s_and_b64 s[6:7], s[0:1], s[6:7]
	v_cndmask_b32_e64 v0, v117, v0, s[2:3]
	s_and_b64 s[2:3], s[22:23], s[54:55]
	buffer_load_dwordx4 v[48:51], v2, s[24:27], 0 offen
	v_add_u32_e32 v2, s43, v1
	v_cndmask_b32_e64 v1, v117, v1, s[2:3]
	s_and_b64 s[2:3], s[22:23], s[6:7]
	v_cndmask_b32_e64 v2, v117, v2, s[2:3]
	buffer_load_dwordx4 v[66:69], v0, s[24:27], 0 offen
	buffer_load_dwordx4 v[70:73], v1, s[36:39], 0 offen
	buffer_load_dwordx4 v[74:77], v2, s[36:39], 0 offen
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmpk_lt_i32 s10, 0x100
	s_waitcnt vmcnt(3)
	ds_write_b128 v118, v[48:51]
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	s_waitcnt vmcnt(2)
	ds_write_b128 v118, v[66:69] offset:16384
	s_waitcnt vmcnt(1)
	ds_write_b128 v118, v[70:73] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v118, v[74:77] offset:49152
	s_cbranch_scc1 .L5
	v_or_b32_e32 v0, 0xc0, v108
	v_add_u32_e32 v0, v0, v64
	v_lshl_or_b32 v1, s9, 8, v96
	s_lshl_b32 s3, s8, 8
	v_mul_lo_u32 v88, s29, v0
	v_subrev_u32_e32 v89, s3, v1
	v_or_b32_e32 v1, 0x80, v108
	v_mul_lo_u32 v91, s28, v0
	v_sub_u32_e32 v0, v65, v64
	s_lshr_b32 s2, s10, 7
	v_add_u32_e32 v1, v1, v64
	v_add_u32_e32 v94, 0xffffff80, v0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v97, v116
	v_mov_b32_e32 v121, v115
	v_mov_b32_e32 v116, v113
	v_mov_b32_e32 v113, v114
	v_mov_b32_e32 v115, v111
	v_mov_b32_e32 v114, v112
	v_mov_b32_e32 v112, v124
	v_mov_b32_e32 v111, v110
	v_mov_b32_e32 v110, v123
	v_mov_b32_e32 v127, v122
	v_mul_lo_u32 v90, s29, v1
	v_add_u32_e32 v92, s19, v96
	v_mul_lo_u32 v93, s28, v1
	s_add_i32 s51, s2, -1
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v32, v0
	v_mov_b32_e32 v33, v0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v0
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v0
	v_mov_b32_e32 v40, v0
	v_mov_b32_e32 v41, v0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	v_mov_b32_e32 v44, v0
	v_mov_b32_e32 v45, v0
	v_mov_b32_e32 v46, v0
	v_mov_b32_e32 v47, v0
	v_mov_b32_e32 v48, v0
	v_mov_b32_e32 v49, v0
	v_mov_b32_e32 v50, v0
	v_mov_b32_e32 v51, v0
	v_mov_b32_e32 v52, v0
	v_mov_b32_e32 v53, v0
	v_mov_b32_e32 v54, v0
	v_mov_b32_e32 v55, v0
	v_mov_b32_e32 v56, v0
	v_mov_b32_e32 v57, v0
	v_mov_b32_e32 v58, v0
	v_mov_b32_e32 v59, v0
	v_mov_b32_e32 v60, v0
	v_mov_b32_e32 v61, v0
	v_mov_b32_e32 v62, v0
	v_mov_b32_e32 v63, v0
.L4:
	v_cmp_lt_i32_e64 s[2:3], v108, v94
	v_cmp_lt_i32_e64 s[8:9], v109, v94
	v_add_u32_e32 v64, v93, v92
	v_add_u32_e32 v65, v91, v92
	s_and_b64 s[6:7], vcc, s[2:3]
	s_and_b64 s[10:11], vcc, s[8:9]
	v_cndmask_b32_e64 v64, v117, v64, s[6:7]
	v_cndmask_b32_e64 v65, v117, v65, s[10:11]
	buffer_load_dwordx4 v[80:83], v64, s[24:27], 0 offen
	buffer_load_dwordx4 v[84:87], v65, s[24:27], 0 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b8 v[64:65], v120 offset:32768
	ds_read_b64_tr_b8 v[66:67], v120 offset:34944
	ds_read_b64_tr_b8 v[68:69], v120 offset:40960
	ds_read_b64_tr_b8 v[70:71], v120 offset:43136
	ds_read_b64_tr_b8 v[72:73], v119
	ds_read_b64_tr_b8 v[74:75], v119 offset:2176
	ds_read_b64_tr_b8 v[76:77], v119 offset:8192
	ds_read_b64_tr_b8 v[78:79], v119 offset:10368
	ds_read_b64_tr_b8 v[100:101], v120 offset:32896
	ds_read_b64_tr_b8 v[102:103], v120 offset:34816
	ds_read_b64_tr_b8 v[104:105], v120 offset:41088
	ds_read_b64_tr_b8 v[106:107], v120 offset:43008
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[64:71], v[72:79], v[0:15] cbsz:0
	s_and_b64 s[8:9], s[0:1], s[8:9]
	v_add_u32_e32 v95, v90, v89
	s_and_b64 s[2:3], s[0:1], s[2:3]
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_cndmask_b32_e64 v95, v117, v95, s[2:3]
	s_add_i32 s51, s51, -1
	v_add_u32_e32 v92, s40, v92
	v_add_u32_e32 v94, 0xffffff80, v94
	s_cmp_lg_u32 s51, 0
	buffer_load_dwordx4 v[122:125], v95, s[4:7], 0 offen
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[16:31], v[100:107], v[72:79], v[16:31] cbsz:0
	ds_read_b64_tr_b8 v[72:73], v119 offset:128
	ds_read_b64_tr_b8 v[74:75], v119 offset:2048
	ds_read_b64_tr_b8 v[76:77], v119 offset:8320
	ds_read_b64_tr_b8 v[78:79], v119 offset:10240
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[64:71], v[72:79], v[32:47] cbsz:0
	v_mfma_f32_32x32x64_f8f6f4 v[48:63], v[100:107], v[72:79], v[48:63] cbsz:0
	ds_read_b64_tr_b8 v[64:65], v120 offset:49152
	ds_read_b64_tr_b8 v[66:67], v120 offset:51328
	ds_read_b64_tr_b8 v[68:69], v120 offset:57344
	ds_read_b64_tr_b8 v[70:71], v120 offset:59520
	ds_read_b64_tr_b8 v[72:73], v119 offset:16384
	ds_read_b64_tr_b8 v[74:75], v119 offset:18560
	ds_read_b64_tr_b8 v[76:77], v119 offset:24576
	ds_read_b64_tr_b8 v[78:79], v119 offset:26752
	ds_read_b64_tr_b8 v[100:101], v120 offset:49280
	ds_read_b64_tr_b8 v[102:103], v120 offset:51200
	ds_read_b64_tr_b8 v[104:105], v120 offset:57472
	ds_read_b64_tr_b8 v[106:107], v120 offset:59392
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[64:71], v[72:79], v[0:15] cbsz:0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[16:31], v[100:107], v[72:79], v[16:31] cbsz:0
	v_add_u32_e32 v72, v88, v89
	v_cndmask_b32_e64 v126, v117, v72, s[8:9]
	ds_read_b64_tr_b8 v[72:73], v119 offset:16512
	ds_read_b64_tr_b8 v[74:75], v119 offset:18432
	ds_read_b64_tr_b8 v[76:77], v119 offset:24704
	ds_read_b64_tr_b8 v[78:79], v119 offset:26624
	v_add_u32_e32 v89, s41, v89
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[64:71], v[72:79], v[32:47] cbsz:0
	buffer_load_dwordx4 v[64:67], v126, s[4:7], 0 offen
	s_barrier
	s_waitcnt vmcnt(3)
	ds_write_b128 v118, v[80:83]
	s_waitcnt vmcnt(2)
	ds_write_b128 v118, v[84:87] offset:16384
	s_waitcnt vmcnt(1)
	ds_write_b128 v118, v[122:125] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v118, v[64:67] offset:49152
	v_mfma_f32_32x32x64_f8f6f4 v[48:63], v[100:107], v[72:79], v[48:63] cbsz:0
	s_cbranch_scc1 .L4
	v_mov_b32_e32 v122, v127
	v_mov_b32_e32 v123, v110
	v_mov_b32_e32 v110, v111
	v_mov_b32_e32 v124, v112
	v_mov_b32_e32 v112, v114
	v_mov_b32_e32 v111, v115
	v_mov_b32_e32 v114, v113
	v_mov_b32_e32 v113, v116
	v_mov_b32_e32 v115, v121
	v_mov_b32_e32 v116, v97
	scratch_load_dword v97, off, off
	scratch_load_dword v121, off, off offset:4
	scratch_load_dword v125, off, off offset:8
	scratch_load_dword v126, off, off offset:12
	scratch_load_dword v127, off, off offset:16
.L5:
	s_andn2_b64 vcc, exec, s[22:23]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_vccnz .L2
	ds_read_b64_tr_b8 v[64:65], v120 offset:32768
	ds_read_b64_tr_b8 v[66:67], v120 offset:34944
	ds_read_b64_tr_b8 v[68:69], v120 offset:40960
	ds_read_b64_tr_b8 v[70:71], v120 offset:43136
	ds_read_b64_tr_b8 v[80:81], v119
	ds_read_b64_tr_b8 v[82:83], v119 offset:2176
	ds_read_b64_tr_b8 v[84:85], v119 offset:8192
	ds_read_b64_tr_b8 v[86:87], v119 offset:10368
	ds_read_b64_tr_b8 v[72:73], v120 offset:32896
	ds_read_b64_tr_b8 v[74:75], v120 offset:34816
	ds_read_b64_tr_b8 v[76:77], v120 offset:41088
	ds_read_b64_tr_b8 v[78:79], v120 offset:43008
	ds_read_b64_tr_b8 v[88:89], v120 offset:49152
	ds_read_b64_tr_b8 v[90:91], v120 offset:51328
	ds_read_b64_tr_b8 v[92:93], v120 offset:57344
	ds_read_b64_tr_b8 v[94:95], v120 offset:59520
	ds_read_b64_tr_b8 v[100:101], v119 offset:16384
	ds_read_b64_tr_b8 v[102:103], v119 offset:18560
	ds_read_b64_tr_b8 v[104:105], v119 offset:24576
	ds_read_b64_tr_b8 v[106:107], v119 offset:26752
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[64:71], v[80:87], v[0:15] cbsz:0
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_32x32x64_f8f6f4 v[16:31], v[72:79], v[80:87], v[16:31] cbsz:0
	ds_read_b64_tr_b8 v[80:81], v120 offset:49280
	ds_read_b64_tr_b8 v[82:83], v120 offset:51200
	ds_read_b64_tr_b8 v[84:85], v120 offset:57472
	ds_read_b64_tr_b8 v[86:87], v120 offset:59392
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[88:95], v[100:107], v[0:15] cbsz:0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[16:31], v[80:87], v[100:107], v[16:31] cbsz:0
	ds_read_b64_tr_b8 v[100:101], v119 offset:128
	ds_read_b64_tr_b8 v[102:103], v119 offset:2048
	ds_read_b64_tr_b8 v[104:105], v119 offset:8320
	ds_read_b64_tr_b8 v[106:107], v119 offset:10240
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[64:71], v[100:107], v[32:47] cbsz:0
	ds_read_b64_tr_b8 v[64:65], v119 offset:16512
	ds_read_b64_tr_b8 v[66:67], v119 offset:18432
	ds_read_b64_tr_b8 v[68:69], v119 offset:24704
	ds_read_b64_tr_b8 v[70:71], v119 offset:26624
	v_mfma_f32_32x32x64_f8f6f4 v[48:63], v[72:79], v[100:107], v[48:63] cbsz:0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[88:95], v[64:71], v[32:47] cbsz:0
	v_mfma_f32_32x32x64_f8f6f4 v[48:63], v[80:87], v[64:71], v[48:63] cbsz:0
	s_branch .L2
.L6:
	s_endpgm
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
	s_nop 0
.Lfunc_end:
.size grouped_variable_k_dot_scaled_kernel, .Lfunc_end-grouped_variable_k_dot_scaled_kernel

.rodata
.p2align 6
.amdhsa_kernel grouped_variable_k_dot_scaled_kernel
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 96
  .amdhsa_next_free_vgpr 224
  .amdhsa_next_free_sgpr 59
  .amdhsa_accum_offset 224
  .amdhsa_float_round_mode_32 3
  .amdhsa_float_round_mode_16_64 3
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_ieee_mode 1
  .amdhsa_dx10_clamp 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: grouped_variable_k_dot_scaled_kernel
    .symbol: grouped_variable_k_dot_scaled_kernel.kd
    .kernarg_segment_size: 96
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 59
    .vgpr_count: 221
    .agpr_count: 0
    .max_flat_workgroup_size: 512
    .sgpr_spill_count: 0
    .vgpr_spill_count: 0
    .uses_dynamic_stack: false
    .uniform_work_group_size: 1
    .args:
      - .offset: 0
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 8
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 16
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 24
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 32
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 40
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 48
        .size: 4
        .value_kind: by_value
      - .offset: 52
        .size: 4
        .value_kind: by_value
      - .offset: 56
        .size: 4
        .value_kind: by_value
      - .offset: 60
        .size: 4
        .value_kind: by_value
      - .offset: 64
        .size: 4
        .value_kind: by_value
      - .offset: 68
        .size: 4
        .value_kind: by_value
      - .offset: 72
        .size: 4
        .value_kind: by_value
      - .offset: 80
        .size: 8
        .value_kind: global_buffer
        .address_space: global
      - .offset: 88
        .size: 8
        .value_kind: global_buffer
        .address_space: global
amdhsa.target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...
.end_amdgpu_metadata
