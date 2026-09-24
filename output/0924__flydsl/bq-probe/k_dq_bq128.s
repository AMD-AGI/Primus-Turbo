	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	k_dq_0
	.p2align	8
	.type	k_dq_0,@function
k_dq_0:
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	v_nop
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_bfe_u32 s2, ttmp6, 0x40010
	s_and_b32 s3, ttmp7, 0xffff
	s_add_co_i32 s2, s2, 1
	s_bfe_u32 s4, ttmp6, 0x40004
	s_mul_i32 s2, s3, s2
	s_bfe_u32 s5, ttmp6, 0x4000c
	s_add_co_i32 s2, s4, s2
	s_add_co_i32 s12, s5, 1
	s_load_b256 s[4:11], s[0:1], 0x140 nv
	s_and_b32 s13, ttmp6, 15
	s_mul_i32 s12, ttmp9, s12
	s_getreg_b32 s14, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s13, s13, s12
	s_cmp_eq_u32 s14, 0
	s_set_vgpr_msb 0x80
	v_mov_b32_e32 v198 /*v710*/, v0
	s_cselect_b32 s12, ttmp9, s13
	s_cselect_b32 s2, s3, s2
	s_bfe_u32 s3, ttmp6, 0x40014
	s_lshr_b32 s13, ttmp7, 16
	s_add_co_i32 s3, s3, 1
	s_bfe_u32 s15, ttmp6, 0x40008
	s_mul_i32 s3, s13, s3
	s_set_vgpr_msb 0x8008
	v_lshrrev_b32_e32 v16, 4, v198 /*v710*/
	s_add_co_i32 s15, s15, s3
	s_cmp_eq_u32 s14, 0
	s_clause 0x2
	s_load_b64 s[20:21], s[0:1], 0x0 nv
	s_load_b64 s[24:25], s[0:1], 0x90 nv
	s_load_b64 s[28:29], s[0:1], 0xc0 nv
	s_cselect_b32 s44, s13, s15
	s_wait_kmcnt 0x0
	s_add_co_i32 s3, s5, 0x7f
	s_set_vgpr_msb 0x880
	v_lshlrev_b32_e32 v130 /*v642*/, 3, v16
	s_ashr_i32 s13, s3, 31
	s_set_vgpr_msb 0x8088
	v_and_b32_e32 v201 /*v713*/, 15, v198 /*v710*/
	s_lshr_b32 s13, s13, 25
	s_mul_i32 s41, s5, s44
	s_add_co_i32 s13, s3, s13
	s_mov_b32 s22, 0x800000
	s_and_b32 s14, s13, 0xffffff80
	s_ashr_i32 s13, s13, 7
	s_cmp_lg_u32 s3, s14
	s_mov_b32 s23, 0
	s_cselect_b32 s14, -1, 0
	s_cmp_lt_i32 s3, 0
	s_mov_b32 s26, s22
	s_cselect_b32 s3, -1, 0
	s_not_b32 s2, s2
	s_and_b32 s3, s3, s14
	s_add_co_i32 s13, s13, s2
	s_cmp_lg_u32 s3, 0
	s_mov_b32 s27, s23
	s_sub_co_ci_u32 s2, s13, 0
	s_load_b32 s13, s[0:1], 0x160 nv
	s_lshl_b32 s37, s2, 7
	s_mov_b32 s30, 0x200000
	s_add_co_i32 s42, s37, s11
	s_set_vgpr_msb 0x8808
	v_or_b32_e32 v2, s37, v201 /*v713*/
	s_add_co_i32 s2, s42, 0x9f
	s_mov_b32 s31, s23
	s_ashr_i32 s3, s2, 31
	s_set_vgpr_msb 0x888
	v_or_b32_e32 v132 /*v644*/, 16, v198 /*v710*/
	s_lshr_b32 s3, s3, 27
	s_mov_b64 s[18:19], 0x800000
	s_add_co_i32 s3, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s14, s3, 0xffffffe0
	s_ashr_i32 s3, s3, 5
	s_cmp_lg_u32 s2, s14
	s_cselect_b32 s14, -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, s14
	s_sub_co_ci_u32 s2, s3, 0
	s_max_i32 s2, s2, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s13, 0
	s_cselect_b32 s3, -1, 0
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s43, s2, s10
	s_add_co_i32 s2, s42, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s3, s2, s3
	s_ashr_i32 s3, s3, 5
	s_cmp_gt_i32 s2, -1
	s_cselect_b32 s2, s3, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s43
	s_cmp_lg_u32 s13, 0
	s_cselect_b32 s40, -1, 0
	s_and_b32 s3, s40, exec_lo
	s_cselect_b32 s2, s2, s10
	s_ashr_i32 s10, s12, 31
	s_ashr_i32 s13, s7, 31
	s_lshr_b32 s10, s10, 29
	s_lshr_b32 s13, s13, 29
	s_add_co_i32 s10, s12, s10
	s_add_co_i32 s13, s7, s13
	s_ashr_i32 s14, s10, 3
	s_and_b32 s10, s10, -8
	s_ashr_i32 s15, s13, 3
	s_and_b32 s13, s13, -8
	s_and_b32 s3, s7, 7
	s_sub_co_i32 s16, s12, s10
	s_cmp_lg_u32 s7, s13
	s_cselect_b32 s13, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s17, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s13, s17, s13
	s_sub_co_ci_u32 s13, s15, 0
	s_cmp_lg_u32 s12, s10
	s_mul_i32 s13, s13, s16
	s_cselect_b32 s10, -1, 0
	s_cmp_lt_i32 s12, 0
	s_cselect_b32 s15, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s10, s15, s10
	s_sub_co_ci_u32 s10, s14, 0
	s_add_co_i32 s10, s10, s13
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s38, s10, s12
	s_abs_i32 s12, s9
	s_abs_i32 s13, s38
	s_cvt_f32_u32 s3, s12
	s_sub_co_i32 s10, 0, s12
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s3, s3
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s3, s3
	s_mul_i32 s10, s10, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s10, s3, s10
	s_add_co_i32 s3, s3, s10
	s_xor_b32 s10, s38, s9
	s_mul_hi_u32 s14, s13, s3
	s_ashr_i32 s3, s10, 31
	s_mul_i32 s15, s14, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s13, s13, s15
	s_add_co_i32 s15, s14, 1
	s_sub_co_i32 s16, s13, s12
	s_cmp_ge_u32 s13, s12
	s_cselect_b32 s14, s15, s14
	s_cselect_b32 s13, s16, s13
	s_add_co_i32 s15, s14, 1
	s_cmp_ge_u32 s13, s12
	s_cselect_b32 s12, s15, s14
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s12, s12, s3
	s_sub_co_i32 s13, s12, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s13, s13, s9
	s_cmp_lg_u32 s38, s13
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s10, 0
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s9, s10, s9
	s_sub_co_ci_u32 s3, s12, s3
	s_or_b32 s36, s37, 32
	s_or_b32 s35, s37, 48
	s_lshl_b32 s14, s7, 4
	s_set_vgpr_msb 0x8848
	v_or_b32_e32 v28 /*v284*/, s36, v201 /*v713*/
	v_or_b32_e32 v29 /*v285*/, s35, v201 /*v713*/
	s_mul_i32 s9, s14, s41
	s_or_b32 s39, s37, 16
	s_lshl4_add_u32 s15, s38, s9
	s_set_vgpr_msb 0x4808
	v_or_b32_e32 v17, s39, v201 /*v713*/
	v_mad_u32 v3, v2, s14, s15
	s_set_vgpr_msb 0x801
	v_mad_u32 v5, v28 /*v284*/, s14, s15
	v_mad_u32 v6, v29 /*v285*/, s14, s15
	s_or_b32 s33, s37, 0x50
	v_mad_u32 v4, s14, v17, s15
	s_set_vgpr_msb 0x148
	v_or_b32_e32 v31 /*v287*/, s33, v201 /*v713*/
	s_or_b32 s9, s37, 0x70
	s_or_b32 s34, s37, 64
	s_set_vgpr_msb 0x48c8
	v_or_b32_e32 v81 /*v849*/, s9, v201 /*v713*/
	s_set_vgpr_msb 0xc848
	v_or_b32_e32 v30 /*v286*/, s34, v201 /*v713*/
	s_set_vgpr_msb 0x4800
	v_or_b32_e32 v6, v6, v16
	v_or_b32_e32 v5, v5, v16
	v_or_b32_e32 v3, v3, v16
	s_or_b32 s10, s37, 0x60
	s_clause 0x1
	s_load_b64 s[12:13], s[0:1], 0x30 nv
	s_load_b64 s[16:17], s[0:1], 0x60 nv
	s_set_vgpr_msb 0xc8
	v_or_b32_e32 v80 /*v848*/, s10, v201 /*v713*/
	s_set_vgpr_msb 0xc800
	v_dual_lshlrev_b32 v5, 4, v5 :: v_dual_bitop2_b32 v4, v4, v16 bitop3:0x54
	v_lshlrev_b32_e32 v3, 4, v3
	s_clause 0x1
	buffer_load_b128 v[8:11], v3, s[20:23], null offen
	buffer_load_b128 v[12:15], v3, s[20:23], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:2080 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:2096 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v3, s[20:23], null offen offset:64
	buffer_load_b128 v[12:15], v3, s[20:23], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:1888 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:1904 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v3, s[20:23], null offen offset:128
	buffer_load_b128 v[12:15], v3, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:352 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:368 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[12:15], v3, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:224 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:240 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v3, s[24:27], null offen
	buffer_load_b128 v[12:15], v3, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:1984 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:2000 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v3, s[24:27], null offen offset:64
	buffer_load_b128 v[12:15], v3, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:384 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:400 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v3, s[24:27], null offen offset:128
	buffer_load_b128 v[12:15], v3, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:416 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:432 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v3, s[24:27], null offen offset:192
	buffer_load_b128 v[12:15], v3, s[24:27], null offen offset:224
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v3, 4, v6
	s_set_vgpr_msb 0xc0
	s_clause 0x9
	buffer_load_b128 v[6:9] /*v[774:777]*/, v3, s[20:23], null offen
	buffer_load_b128 v[10:13] /*v[778:781]*/, v3, s[20:23], null offen offset:32
	buffer_load_b128 v[38:41] /*v[806:809]*/, v3, s[20:23], null offen offset:64
	buffer_load_b128 v[42:45] /*v[810:813]*/, v3, s[20:23], null offen offset:96
	s_set_vgpr_msb 0xc040
	buffer_load_b128 v[194:197] /*v[450:453]*/, v3, s[20:23], null offen offset:128
	buffer_load_b128 v[198:201] /*v[454:457]*/, v3, s[20:23], null offen offset:160
	s_set_vgpr_msb 0x40c0
	buffer_load_b128 v[58:61] /*v[826:829]*/, v3, s[20:23], null offen offset:192
	buffer_load_b128 v[62:65] /*v[830:833]*/, v3, s[20:23], null offen offset:224
	s_clause 0x8
	buffer_load_b128 v[50:53] /*v[818:821]*/, v3, s[24:27], null offen
	buffer_load_b128 v[54:57] /*v[822:825]*/, v3, s[24:27], null offen offset:32
	buffer_load_b128 v[114:117] /*v[882:885]*/, v3, s[24:27], null offen offset:64
	buffer_load_b128 v[118:121] /*v[886:889]*/, v3, s[24:27], null offen offset:96
	buffer_load_b128 v[66:69] /*v[834:837]*/, v3, s[24:27], null offen offset:128
	buffer_load_b128 v[70:73] /*v[838:841]*/, v3, s[24:27], null offen offset:160
	s_set_vgpr_msb 0xc000
	buffer_load_b128 v[92:95], v3, s[24:27], null offen offset:192
	buffer_load_b128 v[96:99], v3, s[24:27], null offen offset:224
	s_set_vgpr_msb 1
	v_mad_u32 v3, v31 /*v287*/, s14, s15
	s_wait_loadcnt 0x11
	scratch_store_b128 off, v[8:11], off offset:256 nv
	s_wait_loadcnt 0x10
	scratch_store_b128 off, v[12:15], off offset:272 nv
	s_set_vgpr_msb 0x100
	v_or_b32_e32 v3, v3, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v3, 4, v3
	s_set_vgpr_msb 0xc0
	s_clause 0x8
	buffer_load_b128 v[130:133] /*v[898:901]*/, v3, s[20:23], null offen
	buffer_load_b128 v[134:137] /*v[902:905]*/, v3, s[20:23], null offen offset:32
	s_set_vgpr_msb 0xc040
	buffer_load_b128 v[210:213] /*v[466:469]*/, v3, s[20:23], null offen offset:64
	buffer_load_b128 v[214:217] /*v[470:473]*/, v3, s[20:23], null offen offset:96
	buffer_load_b128 v[218:221] /*v[474:477]*/, v3, s[20:23], null offen offset:128
	buffer_load_b128 v[222:225] /*v[478:481]*/, v3, s[20:23], null offen offset:160
	buffer_load_b128 v[226:229] /*v[482:485]*/, v3, s[20:23], null offen offset:192
	buffer_load_b128 v[230:233] /*v[486:489]*/, v3, s[20:23], null offen offset:224
	s_clause 0x8
	buffer_load_b128 v[234:237] /*v[490:493]*/, v3, s[24:27], null offen
	buffer_load_b128 v[238:241] /*v[494:497]*/, v3, s[24:27], null offen offset:32
	buffer_load_b128 v[242:245] /*v[498:501]*/, v3, s[24:27], null offen offset:64
	buffer_load_b128 v[246:249] /*v[502:505]*/, v3, s[24:27], null offen offset:96
	buffer_load_b128 v[250:253] /*v[506:509]*/, v3, s[24:27], null offen offset:128
	buffer_load_b128 v[254:257] /*v[510:513]*/, v3, s[24:27], null offen offset:160
	s_set_vgpr_msb 0x4080
	buffer_load_b128 v[2:5] /*v[514:517]*/, v3, s[24:27], null offen offset:192
	buffer_load_b128 v[6:9] /*v[518:521]*/, v3, s[24:27], null offen offset:224
	s_set_vgpr_msb 0x8003
	v_mad_u32 v3, v81 /*v849*/, s14, s15
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v3, v3, v16
	v_dual_lshlrev_b32 v4, 4, v4 :: v_dual_lshlrev_b32 v3, 4, v3
	s_clause 0x1
	buffer_load_b128 v[8:11], v4, s[20:23], null offen
	buffer_load_b128 v[12:15], v4, s[20:23], null offen offset:32
	s_set_vgpr_msb 0x80
	s_clause 0x3
	buffer_load_b128 v[122:125] /*v[634:637]*/, v3, s[24:27], null offen offset:128
	buffer_load_b128 v[126:129] /*v[638:641]*/, v3, s[24:27], null offen offset:160
	buffer_load_b128 v[146:149] /*v[658:661]*/, v3, s[24:27], null offen offset:192
	buffer_load_b128 v[150:153] /*v[662:665]*/, v3, s[24:27], null offen offset:224
	s_wait_loadcnt 0x5
	scratch_store_b128 off, v[8:11], off offset:2112 nv
	s_wait_loadcnt 0x4
	scratch_store_b128 off, v[12:15], off offset:2128 nv
	s_set_vgpr_msb 0x8000
	s_clause 0x1
	buffer_load_b128 v[8:11], v4, s[20:23], null offen offset:64
	buffer_load_b128 v[12:15], v4, s[20:23], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:1920 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:1936 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v4, s[20:23], null offen offset:128
	buffer_load_b128 v[12:15], v4, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:448 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:464 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v4, s[20:23], null offen offset:192
	buffer_load_b128 v[12:15], v4, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:288 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:304 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v4, s[24:27], null offen
	buffer_load_b128 v[12:15], v4, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:2016 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:2032 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v4, s[24:27], null offen offset:64
	buffer_load_b128 v[12:15], v4, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:480 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:496 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v4, s[24:27], null offen offset:128
	buffer_load_b128 v[12:15], v4, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:512 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:528 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v4, s[24:27], null offen offset:192
	buffer_load_b128 v[12:15], v4, s[24:27], null offen offset:224
	s_set_vgpr_msb 1
	v_mad_u32 v4, v30 /*v286*/, s14, s15
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:320 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:336 nv
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_load_b128 v[8:11], v5, s[20:23], null offen
	buffer_load_b128 v[12:15], v5, s[20:23], null offen offset:32
	v_or_b32_e32 v4, v4, v16
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:2144 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:2160 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v5, s[20:23], null offen offset:64
	buffer_load_b128 v[12:15], v5, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v4, 4, v4
	s_set_vgpr_msb 0x80
	s_clause 0xa
	buffer_load_b128 v[74:77] /*v[586:589]*/, v4, s[20:23], null offen
	buffer_load_b128 v[78:81] /*v[590:593]*/, v4, s[20:23], null offen offset:32
	s_set_vgpr_msb 0x80c0
	buffer_load_b128 v[90:93] /*v[858:861]*/, v4, s[20:23], null offen offset:64
	buffer_load_b128 v[94:97] /*v[862:865]*/, v4, s[20:23], null offen offset:96
	s_set_vgpr_msb 0xc080
	buffer_load_b128 v[250:253] /*v[762:765]*/, v4, s[20:23], null offen offset:128
	buffer_load_b128 v[254:257] /*v[766:769]*/, v4, s[20:23], null offen offset:160
	s_set_vgpr_msb 0x80c0
	buffer_load_b128 v[98:101] /*v[866:869]*/, v4, s[20:23], null offen offset:192
	buffer_load_b128 v[102:105] /*v[870:873]*/, v4, s[20:23], null offen offset:224
	s_set_vgpr_msb 0xc000
	s_clause 0x9
	buffer_load_b128 v[68:71], v4, s[24:27], null offen
	buffer_load_b128 v[72:75], v4, s[24:27], null offen offset:32
	s_set_vgpr_msb 0xc0
	buffer_load_b128 v[154:157] /*v[922:925]*/, v4, s[24:27], null offen offset:64
	buffer_load_b128 v[158:161] /*v[926:929]*/, v4, s[24:27], null offen offset:96
	buffer_load_b128 v[30:33] /*v[798:801]*/, v4, s[24:27], null offen offset:128
	buffer_load_b128 v[34:37] /*v[802:805]*/, v4, s[24:27], null offen offset:160
	s_set_vgpr_msb 0xc040
	buffer_load_b128 v[202:205] /*v[458:461]*/, v4, s[24:27], null offen offset:192
	buffer_load_b128 v[206:209] /*v[462:465]*/, v4, s[24:27], null offen offset:224
	s_set_vgpr_msb 0x4003
	v_mad_u32 v4, v80 /*v848*/, s14, s15
	s_mul_i32 s14, s7, s44
	s_wait_loadcnt 0x11
	scratch_store_b128 off, v[8:11], off offset:1952 nv
	s_wait_loadcnt 0x10
	scratch_store_b128 off, v[12:15], off offset:1968 nv
	s_set_vgpr_msb 0x300
	s_clause 0x1
	buffer_load_b128 v[8:11], v5, s[20:23], null offen offset:128
	buffer_load_b128 v[12:15], v5, s[20:23], null offen offset:160
	v_or_b32_e32 v4, v4, v16
	s_add_co_i32 s14, s38, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:544 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:560 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v5, s[20:23], null offen offset:192
	buffer_load_b128 v[12:15], v5, s[20:23], null offen offset:224
	v_lshlrev_b32_e32 v4, 4, v4
	s_set_vgpr_msb 0x80
	s_clause 0x9
	buffer_load_b128 v[10:13] /*v[522:525]*/, v4, s[20:23], null offen
	buffer_load_b128 v[14:17] /*v[526:529]*/, v4, s[20:23], null offen offset:32
	buffer_load_b128 v[18:21] /*v[530:533]*/, v4, s[20:23], null offen offset:64
	buffer_load_b128 v[22:25] /*v[534:537]*/, v4, s[20:23], null offen offset:96
	s_set_vgpr_msb 0x80c0
	buffer_load_b128 v[82:85] /*v[850:853]*/, v4, s[20:23], null offen offset:128
	buffer_load_b128 v[86:89] /*v[854:857]*/, v4, s[20:23], null offen offset:160
	s_set_vgpr_msb 0xc080
	buffer_load_b128 v[26:29] /*v[538:541]*/, v4, s[20:23], null offen offset:192
	buffer_load_b128 v[30:33] /*v[542:545]*/, v4, s[20:23], null offen offset:224
	s_mul_i32 s14, s14, s5
	s_clause 0x3
	buffer_load_b128 v[34:37] /*v[546:549]*/, v4, s[24:27], null offen
	buffer_load_b128 v[38:41] /*v[550:553]*/, v4, s[24:27], null offen offset:32
	buffer_load_b128 v[42:45] /*v[554:557]*/, v4, s[24:27], null offen offset:64
	buffer_load_b128 v[46:49] /*v[558:561]*/, v4, s[24:27], null offen offset:96
	s_set_vgpr_msb 0x8000
	v_add_lshl_u32 v2, s14, v2, 2
	s_wait_loadcnt 0xd
	scratch_store_b128 off, v[8:11], off offset:576 nv
	s_wait_loadcnt 0xc
	scratch_store_b128 off, v[12:15], off offset:592 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v5, s[24:27], null offen
	buffer_load_b128 v[12:15], v5, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:2048 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:2064 nv
	s_clause 0x1
	buffer_load_b128 v[8:11], v5, s[24:27], null offen offset:64
	buffer_load_b128 v[12:15], v5, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[8:11], off offset:608 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[12:15], off offset:624 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v5, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v5, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:640 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:656 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v5, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v5, s[24:27], null offen offset:224
	s_set_vgpr_msb 4
	v_add_lshl_u32 v5, s14, v29 /*v285*/, 2
	s_set_vgpr_msb 0x480
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:672 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:688 nv
	s_clause 0x3
	buffer_load_b128 v[50:53] /*v[562:565]*/, v4, s[24:27], null offen offset:128
	buffer_load_b128 v[54:57] /*v[566:569]*/, v4, s[24:27], null offen offset:160
	buffer_load_b128 v[58:61] /*v[570:573]*/, v4, s[24:27], null offen offset:192
	buffer_load_b128 v[62:65] /*v[574:577]*/, v4, s[24:27], null offen offset:224
	s_clause 0x8
	buffer_load_b128 v[66:69] /*v[578:581]*/, v3, s[20:23], null offen
	buffer_load_b128 v[70:73] /*v[582:585]*/, v3, s[20:23], null offen offset:32
	buffer_load_b128 v[82:85] /*v[594:597]*/, v3, s[20:23], null offen offset:64
	buffer_load_b128 v[86:89] /*v[598:601]*/, v3, s[20:23], null offen offset:96
	s_set_vgpr_msb 0x8000
	buffer_load_b128 v[180:183], v3, s[20:23], null offen offset:128
	buffer_load_b128 v[184:187], v3, s[20:23], null offen offset:160
	buffer_load_b128 v[220:223], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[224:227], v3, s[20:23], null offen offset:224
	s_clause 0x4
	buffer_load_b128 v[236:239], v3, s[24:27], null offen
	buffer_load_b128 v[240:243], v3, s[24:27], null offen offset:32
	s_set_vgpr_msb 64
	buffer_load_b128 v[4:7] /*v[260:263]*/, v3, s[24:27], null offen offset:64
	buffer_load_b128 v[8:11] /*v[264:267]*/, v3, s[24:27], null offen offset:96
	s_wait_xcnt 0x4
	s_load_b64 s[20:21], s[0:1], 0xe8 nv
	s_set_vgpr_msb 0x4000
	v_add_lshl_u32 v3, s14, v17, 2
	s_set_vgpr_msb 4
	v_add_lshl_u32 v4, s14, v28 /*v284*/, 2
	v_add_lshl_u32 v6, s14, v30 /*v286*/, 2
	s_mov_b32 s22, s30
	v_add_lshl_u32 v7, s14, v31 /*v287*/, 2
	s_set_vgpr_msb 0x40c
	v_add_lshl_u32 v8, s14, v80 /*v848*/, 2
	v_add_lshl_u32 v9, s14, v81 /*v849*/, 2
	s_set_vgpr_msb 0xcc0
	s_clause 0x8
	buffer_load_b32 v74 /*v842*/, v2, s[28:31], null offen
	s_set_vgpr_msb 0xc080
	buffer_load_b32 v162 /*v674*/, v3, s[28:31], null offen
	buffer_load_b32 v164 /*v676*/, v4, s[28:31], null offen
	buffer_load_b32 v166 /*v678*/, v5, s[28:31], null offen
	buffer_load_b32 v168 /*v680*/, v6, s[28:31], null offen
	buffer_load_b32 v170 /*v682*/, v7, s[28:31], null offen
	buffer_load_b32 v172 /*v684*/, v8, s[28:31], null offen
	buffer_load_b32 v174 /*v686*/, v9, s[28:31], null offen
	s_wait_kmcnt 0x0
	s_clause 0x9
	buffer_load_b32 v176 /*v688*/, v2, s[20:23], null offen
	buffer_load_b32 v178 /*v690*/, v3, s[20:23], null offen
	s_set_vgpr_msb 0x80c0
	buffer_load_b32 v76 /*v844*/, v4, s[20:23], null offen
	buffer_load_b32 v186 /*v954*/, v5, s[20:23], null offen
	buffer_load_b32 v188 /*v956*/, v6, s[20:23], null offen
	buffer_load_b32 v190 /*v958*/, v7, s[20:23], null offen
	buffer_load_b32 v78 /*v846*/, v8, s[20:23], null offen
	s_set_vgpr_msb 0xc008
	buffer_load_b32 v18, v9, s[20:23], null offen
	s_wait_xcnt 0x7
	v_and_b32_e32 v2, 16, v198 /*v710*/
	s_set_vgpr_msb 0x82a
	v_and_or_b32 v3, v198 /*v710*/, 7, v130 /*v642*/
	v_bfe_u32 v4, v198 /*v710*/, 3, 1
	v_lshrrev_b32_e32 v5, 3, v198 /*v710*/
	s_lshl_b32 s22, s3, 4
	s_set_vgpr_msb 0x2a88
	v_mad_u32_u24 v199 /*v711*/, 0x110, v201 /*v713*/, v2
	v_mad_u32_u24 v200 /*v712*/, 0x110, v132 /*v644*/, v2
	s_set_vgpr_msb 0x8880
	v_mul_u32_u24_e32 v135 /*v647*/, 0x110, v3
	v_lshlrev_b32_e32 v136 /*v648*/, 4, v4
	s_set_vgpr_msb 0x80c0
	v_lshlrev_b32_e32 v200 /*v968*/, 4, v5
	s_ashr_i32 s3, s2, 31
	s_mov_b32 s24, 1
	s_mov_b64 s[14:15], 0x800000
	s_cmp_lt_i32 s2, 1
	s_mul_i32 s23, s6, s44
	s_set_vgpr_msb 0xc00c
	s_clause 0x3a
	scratch_store_b128 off, v[6:9] /*v[774:777]*/, off offset:1792 nv
	scratch_store_b128 off, v[10:13] /*v[778:781]*/, off offset:1808 nv
	scratch_store_b128 off, v[38:41] /*v[806:809]*/, off offset:1824 nv
	scratch_store_b128 off, v[42:45] /*v[810:813]*/, off offset:1840 nv
	scratch_store_b128 off, v[50:53] /*v[818:821]*/, off offset:1856 nv
	scratch_store_b128 off, v[54:57] /*v[822:825]*/, off offset:1872 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[194:197] /*v[450:453]*/, off offset:704 nv
	scratch_store_b128 off, v[198:201] /*v[454:457]*/, off offset:720 nv
	s_set_vgpr_msb 0x400
	scratch_store_b128 off, v[92:95], off offset:736 nv
	scratch_store_b128 off, v[96:99], off offset:752 nv
	s_set_vgpr_msb 12
	scratch_store_b128 off, v[66:69] /*v[834:837]*/, off offset:768 nv
	scratch_store_b128 off, v[70:73] /*v[838:841]*/, off offset:784 nv
	scratch_store_b128 off, v[114:117] /*v[882:885]*/, off offset:800 nv
	scratch_store_b128 off, v[118:121] /*v[886:889]*/, off offset:816 nv
	scratch_store_b128 off, v[154:157] /*v[922:925]*/, off offset:832 nv
	scratch_store_b128 off, v[158:161] /*v[926:929]*/, off offset:848 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[202:205] /*v[458:461]*/, off offset:864 nv
	scratch_store_b128 off, v[206:209] /*v[462:465]*/, off offset:880 nv
	s_set_vgpr_msb 0x40c
	scratch_store_b128 off, v[58:61] /*v[826:829]*/, off offset:896 nv
	scratch_store_b128 off, v[62:65] /*v[830:833]*/, off offset:912 nv
	s_set_vgpr_msb 0xc00
	scratch_store_b128 off, v[68:71], off offset:928 nv
	scratch_store_b128 off, v[72:75], off offset:944 nv
	s_set_vgpr_msb 12
	scratch_store_b128 off, v[30:33] /*v[798:801]*/, off offset:960 nv
	scratch_store_b128 off, v[34:37] /*v[802:805]*/, off offset:976 nv
	scratch_store_b128 off, v[98:101] /*v[866:869]*/, off offset:992 nv
	scratch_store_b128 off, v[102:105] /*v[870:873]*/, off offset:1008 nv
	s_set_vgpr_msb 0xc08
	scratch_store_b128 off, v[250:253] /*v[762:765]*/, off offset:1024 nv
	scratch_store_b128 off, v[254:257] /*v[766:769]*/, off offset:1040 nv
	s_set_vgpr_msb 0x804
	scratch_store_b128 off, v[218:221] /*v[474:477]*/, off offset:1056 nv
	scratch_store_b128 off, v[222:225] /*v[478:481]*/, off offset:1072 nv
	scratch_store_b128 off, v[234:237] /*v[490:493]*/, off offset:1088 nv
	scratch_store_b128 off, v[238:241] /*v[494:497]*/, off offset:1104 nv
	scratch_store_b128 off, v[250:253] /*v[506:509]*/, off offset:1120 nv
	scratch_store_b128 off, v[254:257] /*v[510:513]*/, off offset:1136 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[10:13] /*v[522:525]*/, off offset:1152 nv
	scratch_store_b128 off, v[14:17] /*v[526:529]*/, off offset:1168 nv
	s_set_vgpr_msb 0x80c
	scratch_store_b128 off, v[82:85] /*v[850:853]*/, off offset:1184 nv
	scratch_store_b128 off, v[86:89] /*v[854:857]*/, off offset:1200 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[226:229] /*v[482:485]*/, off offset:1216 nv
	scratch_store_b128 off, v[230:233] /*v[486:489]*/, off offset:1232 nv
	scratch_store_b128 off, v[210:213] /*v[466:469]*/, off offset:1248 nv
	scratch_store_b128 off, v[214:217] /*v[470:473]*/, off offset:1264 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[2:5] /*v[514:517]*/, off offset:1280 nv
	scratch_store_b128 off, v[6:9] /*v[518:521]*/, off offset:1296 nv
	scratch_store_b128 off, v[18:21] /*v[530:533]*/, off offset:1312 nv
	scratch_store_b128 off, v[22:25] /*v[534:537]*/, off offset:1328 nv
	s_wait_loadcnt 0x1b
	scratch_store_b128 off, v[66:69] /*v[578:581]*/, off offset:1344 nv
	s_wait_loadcnt 0x1a
	scratch_store_b128 off, v[70:73] /*v[582:585]*/, off offset:1360 nv
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x17
	scratch_store_b128 off, v[180:183], off offset:1376 nv
	s_wait_loadcnt 0x16
	s_clause 0x6
	scratch_store_b128 off, v[184:187], off offset:1392 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[242:245] /*v[498:501]*/, off offset:1408 nv
	scratch_store_b128 off, v[246:249] /*v[502:505]*/, off offset:1424 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[34:37] /*v[546:549]*/, off offset:1440 nv
	scratch_store_b128 off, v[38:41] /*v[550:553]*/, off offset:1456 nv
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x13
	scratch_store_b128 off, v[236:239], off offset:1472 nv
	s_wait_loadcnt 0x12
	s_clause 0x10
	scratch_store_b128 off, v[240:243], off offset:1488 nv
	s_set_vgpr_msb 8
	scratch_store_b128 off, v[50:53] /*v[562:565]*/, off offset:1504 nv
	scratch_store_b128 off, v[54:57] /*v[566:569]*/, off offset:1520 nv
	scratch_store_b128 off, v[122:125] /*v[634:637]*/, off offset:1536 nv
	scratch_store_b128 off, v[126:129] /*v[638:641]*/, off offset:1552 nv
	scratch_store_b128 off, v[26:29] /*v[538:541]*/, off offset:1568 nv
	scratch_store_b128 off, v[30:33] /*v[542:545]*/, off offset:1584 nv
	scratch_store_b128 off, v[42:45] /*v[554:557]*/, off offset:1600 nv
	scratch_store_b128 off, v[46:49] /*v[558:561]*/, off offset:1616 nv
	scratch_store_b128 off, v[58:61] /*v[570:573]*/, off offset:1632 nv
	scratch_store_b128 off, v[62:65] /*v[574:577]*/, off offset:1648 nv
	scratch_store_b128 off, v[82:85] /*v[594:597]*/, off offset:1664 nv
	scratch_store_b128 off, v[86:89] /*v[598:601]*/, off offset:1680 nv
	s_set_vgpr_msb 0x800
	scratch_store_b128 off, v[220:223], off offset:1696 nv
	scratch_store_b128 off, v[224:227], off offset:1712 nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x11
	scratch_store_b128 off, v[4:7] /*v[260:263]*/, off offset:1728 nv
	s_wait_loadcnt 0x10
	s_clause 0x3
	scratch_store_b128 off, v[8:11] /*v[264:267]*/, off offset:1744 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[146:149] /*v[658:661]*/, off offset:1760 nv
	scratch_store_b128 off, v[150:153] /*v[662:665]*/, off offset:1776 nv
	s_set_vgpr_msb 0x800
	s_cbranch_scc1 .LBB0_4
	s_lshl_b32 s25, s8, 4
	s_set_vgpr_msb 0xc3
	v_mov_b32_e32 v218 /*v986*/, 0
	s_mul_i32 s5, s23, s25
	s_clause 0x7
	scratch_load_b128 v[162:165] /*v[930:933]*/, off, off offset:384 nv
	scratch_load_b128 v[166:169] /*v[934:937]*/, off, off offset:400 nv
	scratch_load_b128 v[226:229] /*v[994:997]*/, off, off offset:800 nv
	scratch_load_b128 v[230:233] /*v[998:1001]*/, off, off offset:816 nv
	scratch_load_b128 v[242:245] /*v[1010:1013]*/, off, off offset:832 nv
	scratch_load_b128 v[246:249] /*v[1014:1017]*/, off, off offset:848 nv
	scratch_load_b128 v[154:157] /*v[922:925]*/, off, off offset:1408 nv
	scratch_load_b128 v[158:161] /*v[926:929]*/, off, off offset:1424 nv
	s_add_co_i32 s5, s22, s5
	s_wait_loadcnt 0x9
	v_dual_mov_b32 v79 /*v847*/, v78 /*v846*/ :: v_dual_mov_b32 v191 /*v959*/, v190 /*v958*/
	s_set_vgpr_msb 0xc308
	v_mad_u32 v2, s25, v132 /*v644*/, s5
	v_mad_u32 v3, s25, v201 /*v713*/, s5
	s_set_vgpr_msb 0x882
	v_dual_mov_b32 v179 /*v691*/, v178 /*v690*/ :: v_dual_bitop2_b32 v133 /*v645*/, s5, v16 bitop3:0x54
	s_set_vgpr_msb 0x82c3
	v_dual_mov_b32 v189 /*v957*/, v188 /*v956*/ :: v_dual_mov_b32 v187 /*v955*/, v186 /*v954*/
	v_dual_mov_b32 v77 /*v845*/, v76 /*v844*/ :: v_dual_mov_b32 v75 /*v843*/, v74 /*v842*/
	s_set_vgpr_msb 0xc382
	v_dual_mov_b32 v177 /*v689*/, v176 /*v688*/ :: v_dual_mov_b32 v163 /*v675*/, v162 /*v674*/
	s_set_vgpr_msb 0x8200
	v_or_b32_e32 v3, v3, v16
	v_or_b32_e32 v2, v2, v16
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v165 /*v677*/, v164 /*v676*/ :: v_dual_mov_b32 v167 /*v679*/, v166 /*v678*/
	v_dual_mov_b32 v169 /*v681*/, v168 /*v680*/ :: v_dual_mov_b32 v171 /*v683*/, v170 /*v682*/
	s_set_vgpr_msb 0x8200
	v_dual_lshlrev_b32 v3, 4, v3 :: v_dual_lshlrev_b32 v2, 4, v2
	s_wait_loadcnt 0x8
	v_mov_b32_e32 v19, v18
	s_set_vgpr_msb 3
	v_mov_b32_e32 v204, v218 /*v986*/
	s_set_vgpr_msb 0x382
	v_dual_mov_b32 v173 /*v685*/, v172 /*v684*/ :: v_dual_mov_b32 v175 /*v687*/, v174 /*v686*/
	s_set_vgpr_msb 0x8200
	v_or_b32_e32 v5, 0xc0, v2
	v_or_b32_e32 v4, 0xe0, v2
	v_or_b32_e32 v6, 0xa0, v2
	v_or_b32_e32 v7, 0x80, v2
	v_or_b32_e32 v8, 0x60, v2
	v_or_b32_e32 v9, 64, v2
	v_or_b32_e32 v10, 32, v2
	s_set_vgpr_msb 64
	s_clause 0x1
	buffer_load_b128 v[194:197] /*v[450:453]*/, v5, s[16:19], null offen
	buffer_load_b128 v[214:217] /*v[470:473]*/, v6, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[202:205] /*v[458:461]*/, v5, s[12:15], null offen
	buffer_load_b128 v[230:233] /*v[486:489]*/, v6, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[210:213] /*v[466:469]*/, v7, s[16:19], null offen
	buffer_load_b128 v[238:241] /*v[494:497]*/, v8, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[226:229] /*v[482:485]*/, v7, s[12:15], null offen
	buffer_load_b128 v[254:257] /*v[510:513]*/, v8, s[12:15], null offen
	s_clause 0x3
	buffer_load_b128 v[234:237] /*v[490:493]*/, v9, s[16:19], null offen
	buffer_load_b128 v[222:225] /*v[478:481]*/, v10, s[16:19], null offen
	buffer_load_b128 v[198:201] /*v[454:457]*/, v4, s[16:19], null offen
	buffer_load_b128 v[218:221] /*v[474:477]*/, v2, s[16:19], null offen
	s_clause 0x2
	buffer_load_b128 v[250:253] /*v[506:509]*/, v9, s[12:15], null offen
	s_set_vgpr_msb 0x4080
	buffer_load_b128 v[6:9] /*v[518:521]*/, v10, s[12:15], null offen
	s_set_vgpr_msb 0x8000
	v_or_b32_e32 v5, 0xc0, v3
	v_or_b32_e32 v6, 0xa0, v3
	s_set_vgpr_msb 64
	s_clause 0x2
	buffer_load_b128 v[206:209] /*v[462:465]*/, v4, s[12:15], null offen
	s_set_vgpr_msb 0x4080
	buffer_load_b128 v[2:5] /*v[514:517]*/, v2, s[12:15], null offen
	s_set_vgpr_msb 0x8000
	v_or_b32_e32 v4, 0x80, v3
	v_or_b32_e32 v7, 0x60, v3
	s_set_vgpr_msb 64
	s_clause 0x2
	buffer_load_b128 v[242:245] /*v[498:501]*/, v5, s[16:19], null offen
	s_set_vgpr_msb 0x4080
	buffer_load_b128 v[22:25] /*v[534:537]*/, v6, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[10:13] /*v[522:525]*/, v5, s[12:15], null offen
	buffer_load_b128 v[30:33] /*v[542:545]*/, v6, s[12:15], null offen
	s_set_vgpr_msb 0x8003
	v_dual_mov_b32 v205, v218 /*v986*/ :: v_dual_bitop2_b32 v5, 64, v3 bitop3:0x54
	v_or_b32_e32 v2, 0xe0, v3
	v_dual_mov_b32 v206, v218 /*v986*/ :: v_dual_bitop2_b32 v6, 32, v3 bitop3:0x54
	s_set_vgpr_msb 0x380
	s_clause 0x1
	buffer_load_b128 v[18:21] /*v[530:533]*/, v4, s[16:19], null offen
	buffer_load_b128 v[54:57] /*v[566:569]*/, v7, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[26:29] /*v[538:541]*/, v4, s[12:15], null offen
	buffer_load_b128 v[62:65] /*v[574:577]*/, v7, s[12:15], null offen
	s_clause 0x5
	buffer_load_b128 v[50:53] /*v[562:565]*/, v5, s[16:19], null offen
	buffer_load_b128 v[38:41] /*v[550:553]*/, v6, s[16:19], null offen
	s_set_vgpr_msb 0x8040
	buffer_load_b128 v[246:249] /*v[502:505]*/, v2, s[16:19], null offen
	s_set_vgpr_msb 0x4080
	buffer_load_b128 v[34:37] /*v[546:549]*/, v3, s[16:19], null offen
	s_clause 0x3
	buffer_load_b128 v[58:61] /*v[570:573]*/, v5, s[12:15], null offen
	buffer_load_b128 v[46:49] /*v[558:561]*/, v6, s[12:15], null offen
	buffer_load_b128 v[14:17] /*v[526:529]*/, v2, s[12:15], null offen
	buffer_load_b128 v[42:45] /*v[554:557]*/, v3, s[12:15], null offen
	s_set_vgpr_msb 0x8008
	v_dual_lshlrev_b32 v4, 1, v198 /*v710*/ :: v_dual_bitop2_b32 v3, 64, v136 /*v648*/ bitop3:0x54
	s_set_vgpr_msb 0x80c
	v_or_b32_e32 v7, 0xa0, v200 /*v968*/
	v_or_b32_e32 v5, 0x60, v200 /*v968*/
	s_set_vgpr_msb 0xc08
	v_or_b32_e32 v6, 0x80, v136 /*v648*/
	v_or_b32_e32 v8, 0xc0, v136 /*v648*/
	v_and_or_b32 v4, v4, 16, 0xe0
	s_set_vgpr_msb 0x8c2
	v_add_nc_u32_e32 v201 /*v969*/, v135 /*v647*/, v7
	s_set_vgpr_msb 0xc282
	v_dual_add_nc_u32 v137 /*v649*/, v135 /*v647*/, v3 :: v_dual_add_nc_u32 v196 /*v708*/, v135 /*v647*/, v5
	v_dual_add_nc_u32 v197 /*v709*/, v135 /*v647*/, v6 :: v_dual_add_nc_u32 v82 /*v594*/, v135 /*v647*/, v8
	v_add_nc_u32_e32 v83 /*v595*/, v135 /*v647*/, v4
	s_set_vgpr_msb 0x820f
	v_dual_mov_b32 v3, v218 /*v986*/ :: v_dual_mov_b32 v4, v218 /*v986*/
	v_dual_mov_b32 v5, v218 /*v986*/ :: v_dual_bitop2_b32 v2, 32, v200 /*v968*/ bitop3:0x54
	v_dual_mov_b32 v6, v218 /*v986*/ :: v_dual_mov_b32 v7, v218 /*v986*/
	v_dual_mov_b32 v8, v218 /*v986*/ :: v_dual_mov_b32 v9, v218 /*v986*/
	s_set_vgpr_msb 0xfc3
	v_dual_mov_b32 v219 /*v987*/, v218 /*v986*/ :: v_dual_mov_b32 v220 /*v988*/, v218 /*v986*/
	v_dual_mov_b32 v221 /*v989*/, v218 /*v986*/ :: v_dual_mov_b32 v222 /*v990*/, v218 /*v986*/
	v_dual_mov_b32 v223 /*v991*/, v218 /*v986*/ :: v_dual_mov_b32 v224 /*v992*/, v218 /*v986*/
	v_mov_b32_e32 v225 /*v993*/, v218 /*v986*/
	s_set_vgpr_msb 0xc343
	v_dual_mov_b32 v178 /*v434*/, v218 /*v986*/ :: v_dual_mov_b32 v179 /*v435*/, v218 /*v986*/
	v_dual_mov_b32 v180 /*v436*/, v218 /*v986*/ :: v_dual_mov_b32 v181 /*v437*/, v218 /*v986*/
	v_dual_mov_b32 v182 /*v438*/, v218 /*v986*/ :: v_dual_mov_b32 v183 /*v439*/, v218 /*v986*/
	v_dual_mov_b32 v184 /*v440*/, v218 /*v986*/ :: v_dual_mov_b32 v185 /*v441*/, v218 /*v986*/
	s_set_vgpr_msb 0x43c3
	v_dual_mov_b32 v202 /*v970*/, v218 /*v986*/ :: v_dual_mov_b32 v203 /*v971*/, v218 /*v986*/
	v_dual_mov_b32 v204 /*v972*/, v218 /*v986*/ :: v_dual_mov_b32 v205 /*v973*/, v218 /*v986*/
	v_dual_mov_b32 v206 /*v974*/, v218 /*v986*/ :: v_dual_mov_b32 v207 /*v975*/, v218 /*v986*/
	v_dual_mov_b32 v208 /*v976*/, v218 /*v986*/ :: v_dual_mov_b32 v209 /*v977*/, v218 /*v986*/
	v_mov_b32_e32 v138 /*v906*/, v218 /*v986*/
	s_set_vgpr_msb 0xc343
	v_dual_mov_b32 v162 /*v418*/, v218 /*v986*/ :: v_dual_mov_b32 v163 /*v419*/, v218 /*v986*/
	s_set_vgpr_msb 0x4303
	v_mov_b32_e32 v207, v218 /*v986*/
	s_set_vgpr_msb 0x382
	v_add_nc_u32_e32 v134 /*v646*/, v135 /*v647*/, v2
	s_set_vgpr_msb 0x8203
	v_mov_b32_e32 v2, v218 /*v986*/
	s_set_vgpr_msb 0x343
	v_dual_mov_b32 v164 /*v420*/, v218 /*v986*/ :: v_dual_mov_b32 v165 /*v421*/, v218 /*v986*/
	s_set_vgpr_msb 0x4300
	v_mov_b64_e32 v[178:179], v[8:9]
	v_mov_b64_e32 v[176:177], v[6:7]
	v_mov_b64_e32 v[174:175], v[4:5]
	v_mov_b64_e32 v[172:173], v[2:3]
	v_mov_b64_e32 v[170:171], v[8:9]
	v_mov_b64_e32 v[168:169], v[6:7]
	v_mov_b64_e32 v[166:167], v[4:5]
	v_mov_b64_e32 v[164:165], v[2:3]
	v_mov_b64_e32 v[162:163], v[8:9]
	v_mov_b64_e32 v[160:161], v[6:7]
	v_mov_b64_e32 v[158:159], v[4:5]
	v_mov_b64_e32 v[156:157], v[2:3]
	v_mov_b64_e32 v[154:155], v[8:9]
	v_mov_b64_e32 v[152:153], v[6:7]
	v_mov_b64_e32 v[150:151], v[4:5]
	v_mov_b64_e32 v[148:149], v[2:3]
	v_mov_b64_e32 v[146:147], v[8:9]
	v_mov_b64_e32 v[144:145], v[6:7]
	v_mov_b64_e32 v[142:143], v[4:5]
	v_mov_b64_e32 v[140:141], v[2:3]
	v_mov_b64_e32 v[138:139], v[8:9]
	v_mov_b64_e32 v[136:137], v[6:7]
	v_mov_b64_e32 v[134:135], v[4:5]
	v_mov_b64_e32 v[132:133], v[2:3]
	v_mov_b64_e32 v[130:131], v[8:9]
	v_mov_b64_e32 v[128:129], v[6:7]
	v_mov_b64_e32 v[126:127], v[4:5]
	v_mov_b64_e32 v[124:125], v[2:3]
	v_mov_b64_e32 v[122:123], v[8:9]
	v_mov_b64_e32 v[120:121], v[6:7]
	v_mov_b64_e32 v[118:119], v[4:5]
	v_mov_b64_e32 v[116:117], v[2:3]
	v_mov_b64_e32 v[114:115], v[8:9]
	v_mov_b64_e32 v[112:113], v[6:7]
	v_mov_b64_e32 v[110:111], v[4:5]
	v_mov_b64_e32 v[108:109], v[2:3]
	v_mov_b64_e32 v[106:107], v[8:9]
	v_mov_b64_e32 v[104:105], v[6:7]
	v_mov_b64_e32 v[102:103], v[4:5]
	v_mov_b64_e32 v[100:101], v[2:3]
	v_mov_b64_e32 v[90:91], v[8:9]
	v_mov_b64_e32 v[88:89], v[6:7]
	v_mov_b64_e32 v[86:87], v[4:5]
	v_mov_b64_e32 v[84:85], v[2:3]
	v_mov_b64_e32 v[66:67], v[8:9]
	v_mov_b64_e32 v[64:65], v[6:7]
	v_mov_b64_e32 v[62:63], v[4:5]
	v_mov_b64_e32 v[60:61], v[2:3]
	v_mov_b64_e32 v[82:83], v[8:9]
	v_mov_b64_e32 v[80:81], v[6:7]
	v_mov_b64_e32 v[78:79], v[4:5]
	v_mov_b64_e32 v[76:77], v[2:3]
	v_mov_b64_e32 v[58:59], v[8:9]
	v_mov_b64_e32 v[56:57], v[6:7]
	v_mov_b64_e32 v[54:55], v[4:5]
	v_mov_b64_e32 v[52:53], v[2:3]
	v_mov_b64_e32 v[50:51], v[8:9]
	v_mov_b64_e32 v[48:49], v[6:7]
	v_mov_b64_e32 v[46:47], v[4:5]
	v_mov_b64_e32 v[44:45], v[2:3]
	v_mov_b64_e32 v[42:43], v[8:9]
	v_mov_b64_e32 v[40:41], v[6:7]
	v_mov_b64_e32 v[38:39], v[4:5]
	v_mov_b64_e32 v[36:37], v[2:3]
	s_set_vgpr_msb 0x43
	v_dual_mov_b32 v166 /*v422*/, v218 /*v986*/ :: v_dual_mov_b32 v167 /*v423*/, v218 /*v986*/
	v_dual_mov_b32 v168 /*v424*/, v218 /*v986*/ :: v_dual_mov_b32 v169 /*v425*/, v218 /*v986*/
	s_set_vgpr_msb 0x4303
	v_dual_mov_b32 v208, v218 /*v986*/ :: v_dual_mov_b32 v209, v218 /*v986*/
	v_dual_mov_b32 v210, v218 /*v986*/ :: v_dual_mov_b32 v211, v218 /*v986*/
	s_set_vgpr_msb 0x383
	v_dual_mov_b32 v138 /*v650*/, v218 /*v986*/ :: v_dual_mov_b32 v139 /*v651*/, v218 /*v986*/
	v_dual_mov_b32 v140 /*v652*/, v218 /*v986*/ :: v_dual_mov_b32 v141 /*v653*/, v218 /*v986*/
	v_dual_mov_b32 v142 /*v654*/, v218 /*v986*/ :: v_dual_mov_b32 v143 /*v655*/, v218 /*v986*/
	v_dual_mov_b32 v144 /*v656*/, v218 /*v986*/ :: v_dual_mov_b32 v145 /*v657*/, v218 /*v986*/
	s_set_vgpr_msb 0x8343
	v_dual_mov_b32 v130 /*v386*/, v218 /*v986*/ :: v_dual_mov_b32 v131 /*v387*/, v218 /*v986*/
	v_dual_mov_b32 v132 /*v388*/, v218 /*v986*/ :: v_dual_mov_b32 v133 /*v389*/, v218 /*v986*/
	v_dual_mov_b32 v134 /*v390*/, v218 /*v986*/ :: v_dual_mov_b32 v135 /*v391*/, v218 /*v986*/
	v_dual_mov_b32 v136 /*v392*/, v218 /*v986*/ :: v_dual_mov_b32 v137 /*v393*/, v218 /*v986*/
	v_dual_mov_b32 v170 /*v426*/, v218 /*v986*/ :: v_dual_mov_b32 v171 /*v427*/, v218 /*v986*/
	v_dual_mov_b32 v172 /*v428*/, v218 /*v986*/ :: v_dual_mov_b32 v173 /*v429*/, v218 /*v986*/
	v_dual_mov_b32 v174 /*v430*/, v218 /*v986*/ :: v_dual_mov_b32 v175 /*v431*/, v218 /*v986*/
	v_dual_mov_b32 v176 /*v432*/, v218 /*v986*/ :: v_dual_mov_b32 v177 /*v433*/, v218 /*v986*/
	v_dual_mov_b32 v146 /*v402*/, v218 /*v986*/ :: v_dual_mov_b32 v147 /*v403*/, v218 /*v986*/
	v_dual_mov_b32 v148 /*v404*/, v218 /*v986*/ :: v_dual_mov_b32 v149 /*v405*/, v218 /*v986*/
	v_dual_mov_b32 v150 /*v406*/, v218 /*v986*/ :: v_dual_mov_b32 v151 /*v407*/, v218 /*v986*/
	v_dual_mov_b32 v152 /*v408*/, v218 /*v986*/ :: v_dual_mov_b32 v153 /*v409*/, v218 /*v986*/
	v_dual_mov_b32 v154 /*v410*/, v218 /*v986*/ :: v_dual_mov_b32 v155 /*v411*/, v218 /*v986*/
	v_dual_mov_b32 v156 /*v412*/, v218 /*v986*/ :: v_dual_mov_b32 v157 /*v413*/, v218 /*v986*/
	v_dual_mov_b32 v158 /*v414*/, v218 /*v986*/ :: v_dual_mov_b32 v159 /*v415*/, v218 /*v986*/
	v_dual_mov_b32 v160 /*v416*/, v218 /*v986*/ :: v_dual_mov_b32 v161 /*v417*/, v218 /*v986*/
	v_dual_mov_b32 v98 /*v354*/, v218 /*v986*/ :: v_dual_mov_b32 v99 /*v355*/, v218 /*v986*/
	v_dual_mov_b32 v100 /*v356*/, v218 /*v986*/ :: v_dual_mov_b32 v101 /*v357*/, v218 /*v986*/
	v_dual_mov_b32 v102 /*v358*/, v218 /*v986*/ :: v_dual_mov_b32 v103 /*v359*/, v218 /*v986*/
	v_dual_mov_b32 v104 /*v360*/, v218 /*v986*/ :: v_dual_mov_b32 v105 /*v361*/, v218 /*v986*/
	v_dual_mov_b32 v138 /*v394*/, v218 /*v986*/ :: v_dual_mov_b32 v139 /*v395*/, v218 /*v986*/
	v_dual_mov_b32 v140 /*v396*/, v218 /*v986*/ :: v_dual_mov_b32 v141 /*v397*/, v218 /*v986*/
	v_dual_mov_b32 v142 /*v398*/, v218 /*v986*/ :: v_dual_mov_b32 v143 /*v399*/, v218 /*v986*/
	v_dual_mov_b32 v144 /*v400*/, v218 /*v986*/ :: v_dual_mov_b32 v145 /*v401*/, v218 /*v986*/
	v_dual_mov_b32 v114 /*v370*/, v218 /*v986*/ :: v_dual_mov_b32 v115 /*v371*/, v218 /*v986*/
	v_dual_mov_b32 v116 /*v372*/, v218 /*v986*/ :: v_dual_mov_b32 v117 /*v373*/, v218 /*v986*/
	v_dual_mov_b32 v118 /*v374*/, v218 /*v986*/ :: v_dual_mov_b32 v119 /*v375*/, v218 /*v986*/
	v_dual_mov_b32 v120 /*v376*/, v218 /*v986*/ :: v_dual_mov_b32 v121 /*v377*/, v218 /*v986*/
	v_dual_mov_b32 v122 /*v378*/, v218 /*v986*/ :: v_dual_mov_b32 v123 /*v379*/, v218 /*v986*/
	v_dual_mov_b32 v124 /*v380*/, v218 /*v986*/ :: v_dual_mov_b32 v125 /*v381*/, v218 /*v986*/
	v_dual_mov_b32 v126 /*v382*/, v218 /*v986*/ :: v_dual_mov_b32 v127 /*v383*/, v218 /*v986*/
	v_dual_mov_b32 v128 /*v384*/, v218 /*v986*/ :: v_dual_mov_b32 v129 /*v385*/, v218 /*v986*/
	s_set_vgpr_msb 0x43c3
	v_dual_mov_b32 v139 /*v907*/, v218 /*v986*/ :: v_dual_mov_b32 v140 /*v908*/, v218 /*v986*/
	v_dual_mov_b32 v141 /*v909*/, v218 /*v986*/ :: v_dual_mov_b32 v142 /*v910*/, v218 /*v986*/
	v_dual_mov_b32 v143 /*v911*/, v218 /*v986*/ :: v_dual_mov_b32 v144 /*v912*/, v218 /*v986*/
	v_dual_mov_b32 v145 /*v913*/, v218 /*v986*/ :: v_dual_mov_b32 v210 /*v978*/, v218 /*v986*/
	s_set_vgpr_msb 0xc343
	v_dual_mov_b32 v106 /*v362*/, v218 /*v986*/ :: v_dual_mov_b32 v107 /*v363*/, v218 /*v986*/
	v_dual_mov_b32 v108 /*v364*/, v218 /*v986*/ :: v_dual_mov_b32 v109 /*v365*/, v218 /*v986*/
	v_dual_mov_b32 v110 /*v366*/, v218 /*v986*/ :: v_dual_mov_b32 v111 /*v367*/, v218 /*v986*/
	v_dual_mov_b32 v112 /*v368*/, v218 /*v986*/ :: v_dual_mov_b32 v113 /*v369*/, v218 /*v986*/
	v_dual_mov_b32 v82 /*v338*/, v218 /*v986*/ :: v_dual_mov_b32 v83 /*v339*/, v218 /*v986*/
	v_dual_mov_b32 v84 /*v340*/, v218 /*v986*/ :: v_dual_mov_b32 v85 /*v341*/, v218 /*v986*/
	v_dual_mov_b32 v86 /*v342*/, v218 /*v986*/ :: v_dual_mov_b32 v87 /*v343*/, v218 /*v986*/
	v_dual_mov_b32 v88 /*v344*/, v218 /*v986*/ :: v_dual_mov_b32 v89 /*v345*/, v218 /*v986*/
	v_dual_mov_b32 v90 /*v346*/, v218 /*v986*/ :: v_dual_mov_b32 v91 /*v347*/, v218 /*v986*/
	v_dual_mov_b32 v92 /*v348*/, v218 /*v986*/ :: v_dual_mov_b32 v93 /*v349*/, v218 /*v986*/
	v_dual_mov_b32 v94 /*v350*/, v218 /*v986*/ :: v_dual_mov_b32 v95 /*v351*/, v218 /*v986*/
	v_dual_mov_b32 v96 /*v352*/, v218 /*v986*/ :: v_dual_mov_b32 v97 /*v353*/, v218 /*v986*/
	v_dual_mov_b32 v186 /*v442*/, v218 /*v986*/ :: v_dual_mov_b32 v187 /*v443*/, v218 /*v986*/
	v_dual_mov_b32 v188 /*v444*/, v218 /*v986*/ :: v_dual_mov_b32 v189 /*v445*/, v218 /*v986*/
	v_dual_mov_b32 v190 /*v446*/, v218 /*v986*/ :: v_dual_mov_b32 v191 /*v447*/, v218 /*v986*/
	v_dual_mov_b32 v192 /*v448*/, v218 /*v986*/ :: v_dual_mov_b32 v193 /*v449*/, v218 /*v986*/
	v_dual_mov_b32 v74 /*v330*/, v218 /*v986*/ :: v_dual_mov_b32 v75 /*v331*/, v218 /*v986*/
	v_dual_mov_b32 v76 /*v332*/, v218 /*v986*/ :: v_dual_mov_b32 v77 /*v333*/, v218 /*v986*/
	v_dual_mov_b32 v78 /*v334*/, v218 /*v986*/ :: v_dual_mov_b32 v79 /*v335*/, v218 /*v986*/
	v_dual_mov_b32 v80 /*v336*/, v218 /*v986*/ :: v_dual_mov_b32 v81 /*v337*/, v218 /*v986*/
	s_set_vgpr_msb 0x43c3
	v_dual_mov_b32 v211 /*v979*/, v218 /*v986*/ :: v_dual_mov_b32 v212 /*v980*/, v218 /*v986*/
	v_dual_mov_b32 v213 /*v981*/, v218 /*v986*/ :: v_dual_mov_b32 v214 /*v982*/, v218 /*v986*/
	v_dual_mov_b32 v215 /*v983*/, v218 /*v986*/ :: v_dual_mov_b32 v216 /*v984*/, v218 /*v986*/
	v_dual_mov_b32 v217 /*v985*/, v218 /*v986*/ :: v_dual_mov_b32 v146 /*v914*/, v218 /*v986*/
	s_set_vgpr_msb 0xc343
	v_dual_mov_b32 v58 /*v314*/, v218 /*v986*/ :: v_dual_mov_b32 v59 /*v315*/, v218 /*v986*/
	v_dual_mov_b32 v60 /*v316*/, v218 /*v986*/ :: v_dual_mov_b32 v61 /*v317*/, v218 /*v986*/
	v_dual_mov_b32 v62 /*v318*/, v218 /*v986*/ :: v_dual_mov_b32 v63 /*v319*/, v218 /*v986*/
	v_dual_mov_b32 v64 /*v320*/, v218 /*v986*/ :: v_dual_mov_b32 v65 /*v321*/, v218 /*v986*/
	v_dual_mov_b32 v34 /*v290*/, v218 /*v986*/ :: v_dual_mov_b32 v35 /*v291*/, v218 /*v986*/
	v_dual_mov_b32 v36 /*v292*/, v218 /*v986*/ :: v_dual_mov_b32 v37 /*v293*/, v218 /*v986*/
	v_dual_mov_b32 v38 /*v294*/, v218 /*v986*/ :: v_dual_mov_b32 v39 /*v295*/, v218 /*v986*/
	v_dual_mov_b32 v40 /*v296*/, v218 /*v986*/ :: v_dual_mov_b32 v41 /*v297*/, v218 /*v986*/
	s_set_vgpr_msb 0x4303
	v_dual_mov_b32 v252, v218 /*v986*/ :: v_dual_mov_b32 v253, v218 /*v986*/
	v_dual_mov_b32 v254, v218 /*v986*/ :: v_dual_mov_b32 v255, v218 /*v986*/
	s_set_vgpr_msb 0x343
	v_dual_mov_b32 v0 /*v256*/, v218 /*v986*/ :: v_dual_mov_b32 v1 /*v257*/, v218 /*v986*/
	v_dual_mov_b32 v2 /*v258*/, v218 /*v986*/ :: v_dual_mov_b32 v3 /*v259*/, v218 /*v986*/
	v_dual_mov_b32 v20 /*v276*/, v218 /*v986*/ :: v_dual_mov_b32 v21 /*v277*/, v218 /*v986*/
	v_dual_mov_b32 v22 /*v278*/, v218 /*v986*/ :: v_dual_mov_b32 v23 /*v279*/, v218 /*v986*/
	v_dual_mov_b32 v24 /*v280*/, v218 /*v986*/ :: v_dual_mov_b32 v25 /*v281*/, v218 /*v986*/
	v_dual_mov_b32 v26 /*v282*/, v218 /*v986*/ :: v_dual_mov_b32 v27 /*v283*/, v218 /*v986*/
	v_dual_mov_b32 v50 /*v306*/, v218 /*v986*/ :: v_dual_mov_b32 v51 /*v307*/, v218 /*v986*/
	v_dual_mov_b32 v52 /*v308*/, v218 /*v986*/ :: v_dual_mov_b32 v53 /*v309*/, v218 /*v986*/
	v_dual_mov_b32 v54 /*v310*/, v218 /*v986*/ :: v_dual_mov_b32 v55 /*v311*/, v218 /*v986*/
	v_dual_mov_b32 v56 /*v312*/, v218 /*v986*/ :: v_dual_mov_b32 v57 /*v313*/, v218 /*v986*/
	v_dual_mov_b32 v42 /*v298*/, v218 /*v986*/ :: v_dual_mov_b32 v43 /*v299*/, v218 /*v986*/
	v_dual_mov_b32 v44 /*v300*/, v218 /*v986*/ :: v_dual_mov_b32 v45 /*v301*/, v218 /*v986*/
	v_dual_mov_b32 v46 /*v302*/, v218 /*v986*/ :: v_dual_mov_b32 v47 /*v303*/, v218 /*v986*/
	v_dual_mov_b32 v48 /*v304*/, v218 /*v986*/ :: v_dual_mov_b32 v49 /*v305*/, v218 /*v986*/
	s_set_vgpr_msb 0x4303
	v_dual_mov_b32 v196, v218 /*v986*/ :: v_dual_mov_b32 v197, v218 /*v986*/
	v_dual_mov_b32 v198, v218 /*v986*/ :: v_dual_mov_b32 v199, v218 /*v986*/
	v_dual_mov_b32 v200, v218 /*v986*/ :: v_dual_mov_b32 v201, v218 /*v986*/
	v_dual_mov_b32 v202, v218 /*v986*/ :: v_dual_mov_b32 v203, v218 /*v986*/
	s_set_vgpr_msb 0x3c3
	v_dual_mov_b32 v147 /*v915*/, v218 /*v986*/ :: v_dual_mov_b32 v148 /*v916*/, v218 /*v986*/
	v_dual_mov_b32 v149 /*v917*/, v218 /*v986*/ :: v_dual_mov_b32 v150 /*v918*/, v218 /*v986*/
	v_dual_mov_b32 v151 /*v919*/, v218 /*v986*/ :: v_dual_mov_b32 v152 /*v920*/, v218 /*v986*/
	v_dual_mov_b32 v153 /*v921*/, v218 /*v986*/ :: v_dual_mov_b32 v192 /*v960*/, v218 /*v986*/
	s_set_vgpr_msb 0xc343
	v_dual_mov_b32 v66 /*v322*/, v218 /*v986*/ :: v_dual_mov_b32 v67 /*v323*/, v218 /*v986*/
	v_dual_mov_b32 v68 /*v324*/, v218 /*v986*/ :: v_dual_mov_b32 v69 /*v325*/, v218 /*v986*/
	v_dual_mov_b32 v70 /*v326*/, v218 /*v986*/ :: v_dual_mov_b32 v71 /*v327*/, v218 /*v986*/
	v_dual_mov_b32 v72 /*v328*/, v218 /*v986*/ :: v_dual_mov_b32 v73 /*v329*/, v218 /*v986*/
	s_set_vgpr_msb 0x43c3
	v_dual_mov_b32 v193 /*v961*/, v218 /*v986*/ :: v_dual_mov_b32 v194 /*v962*/, v218 /*v986*/
	v_dual_mov_b32 v195 /*v963*/, v218 /*v986*/ :: v_dual_mov_b32 v196 /*v964*/, v218 /*v986*/
	v_dual_mov_b32 v197 /*v965*/, v218 /*v986*/ :: v_dual_mov_b32 v198 /*v966*/, v218 /*v986*/
	v_dual_mov_b32 v199 /*v967*/, v218 /*v986*/ :: v_dual_mov_b32 v234 /*v1002*/, v218 /*v986*/
	s_set_vgpr_msb 0xc303
	v_dual_mov_b32 v20, v218 /*v986*/ :: v_dual_mov_b32 v21, v218 /*v986*/
	v_dual_mov_b32 v22, v218 /*v986*/ :: v_dual_mov_b32 v23, v218 /*v986*/
	v_dual_mov_b32 v24, v218 /*v986*/ :: v_dual_mov_b32 v25, v218 /*v986*/
	v_dual_mov_b32 v26, v218 /*v986*/ :: v_dual_mov_b32 v27, v218 /*v986*/
	v_dual_mov_b32 v244, v218 /*v986*/ :: v_dual_mov_b32 v245, v218 /*v986*/
	v_dual_mov_b32 v246, v218 /*v986*/ :: v_dual_mov_b32 v247, v218 /*v986*/
	v_dual_mov_b32 v248, v218 /*v986*/ :: v_dual_mov_b32 v249, v218 /*v986*/
	v_dual_mov_b32 v250, v218 /*v986*/ :: v_dual_mov_b32 v251, v218 /*v986*/
	s_set_vgpr_msb 0x3c3
	v_dual_mov_b32 v235 /*v1003*/, v218 /*v986*/ :: v_dual_mov_b32 v236 /*v1004*/, v218 /*v986*/
	v_dual_mov_b32 v237 /*v1005*/, v218 /*v986*/ :: v_dual_mov_b32 v238 /*v1006*/, v218 /*v986*/
	v_dual_mov_b32 v239 /*v1007*/, v218 /*v986*/ :: v_dual_mov_b32 v240 /*v1008*/, v218 /*v986*/
	v_dual_mov_b32 v241 /*v1009*/, v218 /*v986*/ :: v_dual_mov_b32 v178 /*v946*/, v218 /*v986*/
	s_set_vgpr_msb 0xc303
	v_dual_mov_b32 v228, v218 /*v986*/ :: v_dual_mov_b32 v229, v218 /*v986*/
	v_dual_mov_b32 v230, v218 /*v986*/ :: v_dual_mov_b32 v231, v218 /*v986*/
	v_dual_mov_b32 v232, v218 /*v986*/ :: v_dual_mov_b32 v233, v218 /*v986*/
	v_dual_mov_b32 v234, v218 /*v986*/ :: v_dual_mov_b32 v235, v218 /*v986*/
	v_dual_mov_b32 v212, v218 /*v986*/ :: v_dual_mov_b32 v213, v218 /*v986*/
	v_dual_mov_b32 v214, v218 /*v986*/ :: v_dual_mov_b32 v215, v218 /*v986*/
	v_dual_mov_b32 v216, v218 /*v986*/ :: v_dual_mov_b32 v217, v218 /*v986*/
	v_dual_mov_b32 v218, v218 /*v986*/ :: v_dual_mov_b32 v219, v218 /*v986*/
	v_dual_mov_b32 v28, v218 /*v986*/ :: v_dual_mov_b32 v29, v218 /*v986*/
	v_dual_mov_b32 v30, v218 /*v986*/ :: v_dual_mov_b32 v31, v218 /*v986*/
	v_dual_mov_b32 v32, v218 /*v986*/ :: v_dual_mov_b32 v33, v218 /*v986*/
	v_dual_mov_b32 v34, v218 /*v986*/ :: v_dual_mov_b32 v35, v218 /*v986*/
	s_set_vgpr_msb 0x383
	v_dual_mov_b32 v154 /*v666*/, v218 /*v986*/ :: v_dual_mov_b32 v155 /*v667*/, v218 /*v986*/
	v_dual_mov_b32 v156 /*v668*/, v218 /*v986*/ :: v_dual_mov_b32 v157 /*v669*/, v218 /*v986*/
	v_dual_mov_b32 v158 /*v670*/, v218 /*v986*/ :: v_dual_mov_b32 v159 /*v671*/, v218 /*v986*/
	v_dual_mov_b32 v160 /*v672*/, v218 /*v986*/ :: v_dual_mov_b32 v161 /*v673*/, v218 /*v986*/
	s_set_vgpr_msb 0x83c3
	v_dual_mov_b32 v179 /*v947*/, v218 /*v986*/ :: v_dual_mov_b32 v180 /*v948*/, v218 /*v986*/
	v_dual_mov_b32 v181 /*v949*/, v218 /*v986*/ :: v_dual_mov_b32 v182 /*v950*/, v218 /*v986*/
	v_dual_mov_b32 v183 /*v951*/, v218 /*v986*/ :: v_dual_mov_b32 v184 /*v952*/, v218 /*v986*/
	v_dual_mov_b32 v185 /*v953*/, v218 /*v986*/ :: v_dual_mov_b32 v170 /*v938*/, v218 /*v986*/
	s_set_vgpr_msb 0xc303
	v_dual_mov_b32 v188, v218 /*v986*/ :: v_dual_mov_b32 v189, v218 /*v986*/
	v_dual_mov_b32 v190, v218 /*v986*/ :: v_dual_mov_b32 v191, v218 /*v986*/
	v_dual_mov_b32 v192, v218 /*v986*/ :: v_dual_mov_b32 v193, v218 /*v986*/
	v_dual_mov_b32 v194, v218 /*v986*/ :: v_dual_mov_b32 v195, v218 /*v986*/
	s_set_vgpr_msb 0x3c3
	v_dual_mov_b32 v171 /*v939*/, v218 /*v986*/ :: v_dual_mov_b32 v172 /*v940*/, v218 /*v986*/
	v_dual_mov_b32 v173 /*v941*/, v218 /*v986*/ :: v_dual_mov_b32 v174 /*v942*/, v218 /*v986*/
	v_dual_mov_b32 v175 /*v943*/, v218 /*v986*/ :: v_dual_mov_b32 v176 /*v944*/, v218 /*v986*/
	v_mov_b32_e32 v177 /*v945*/, v218 /*v986*/
	s_set_vgpr_msb 0xc303
	v_dual_mov_b32 v0, v218 /*v986*/ :: v_dual_mov_b32 v1, v218 /*v986*/
	s_add_co_i32 s26, s2, 0x7ffffff
	s_mov_b32 s5, s4
	s_mov_b32 s6, 0x3fb8aa3b
	s_mov_b64 s[20:21], s[2:3]
	s_clause 0xd
	scratch_store_b128 off, v[0:3], off offset:160 nv
	scratch_store_b128 off, v[4:7], off offset:176 nv
	scratch_store_b128 off, v[0:3], off offset:128 nv
	scratch_store_b128 off, v[4:7], off offset:144 nv
	scratch_store_b128 off, v[0:3], off offset:96 nv
	scratch_store_b128 off, v[4:7], off offset:112 nv
	scratch_store_b128 off, v[0:3], off offset:64 nv
	scratch_store_b128 off, v[4:7], off offset:80 nv
	scratch_store_b128 off, v[0:3], off offset:32 nv
	scratch_store_b128 off, v[4:7], off offset:48 nv
	scratch_store_b128 off, v[0:3], off nv
	scratch_store_b128 off, v[4:7], off offset:16 nv
	scratch_store_b128 off, v[0:3], off offset:192 nv
	scratch_store_b128 off, v[4:7], off offset:208 nv
	s_set_vgpr_msb 0x300
.LBB0_2:
	s_set_vgpr_msb 0x8e
	s_clause 0x5
	scratch_load_b128 v[234:237] /*v[746:749]*/, off, off offset:2080 nv
	scratch_load_b128 v[238:241] /*v[750:753]*/, off, off offset:2096 nv
	scratch_load_b128 v[242:245] /*v[754:757]*/, off, off offset:2112 nv
	scratch_load_b128 v[246:249] /*v[758:761]*/, off, off offset:2128 nv
	scratch_load_b128 v[250:253] /*v[762:765]*/, off, off offset:2144 nv
	scratch_load_b128 v[254:257] /*v[766:769]*/, off, off offset:2160 nv
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[42:49] /*v[554:561]*/, v[6:13] /*v[774:781]*/, 0
	s_set_vgpr_msb 0x8ec0
	s_clause 0xe
	scratch_load_b128 v[18:21] /*v[786:789]*/, off, off offset:1152 nv
	scratch_load_b128 v[22:25] /*v[790:793]*/, off, off offset:1168 nv
	scratch_load_b128 v[122:125] /*v[890:893]*/, off, off offset:1344 nv
	scratch_load_b128 v[126:129] /*v[894:897]*/, off, off offset:1360 nv
	scratch_load_b128 v[26:29] /*v[794:797]*/, off, off offset:1888 nv
	scratch_load_b128 v[30:33] /*v[798:801]*/, off, off offset:1904 nv
	scratch_load_b128 v[58:61] /*v[826:829]*/, off, off offset:1920 nv
	scratch_load_b128 v[62:65] /*v[830:833]*/, off, off offset:1936 nv
	scratch_load_b128 v[82:85] /*v[850:853]*/, off, off offset:1952 nv
	scratch_load_b128 v[86:89] /*v[854:857]*/, off, off offset:1968 nv
	scratch_load_b128 v[66:69] /*v[834:837]*/, off, off offset:1248 nv
	scratch_load_b128 v[70:73] /*v[838:841]*/, off, off offset:1264 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[68:71], off, off offset:1312 nv
	scratch_load_b128 v[72:75], off, off offset:1328 nv
	s_set_vgpr_msb 0x8a
	v_wmma_f32_16x16x32_bf16 v[116:123] /*v[628:635]*/, v[42:49] /*v[554:561]*/, v[74:81] /*v[586:593]*/, 0
	s_set_vgpr_msb 0x8a03
	v_mov_b64_e32 v[180:181], v[38:39] /*v[806:807]*/
	v_mov_b64_e32 v[182:183], v[40:41] /*v[808:809]*/
	v_mov_b64_e32 v[184:185], v[42:43] /*v[810:811]*/
	v_mov_b64_e32 v[186:187], v[44:45] /*v[812:813]*/
	s_set_vgpr_msb 0x3c0
	s_clause 0x3
	scratch_load_b128 v[98:101] /*v[866:869]*/, off, off offset:2016 nv
	scratch_load_b128 v[102:105] /*v[870:873]*/, off, off offset:2032 nv
	scratch_load_b128 v[106:109] /*v[874:877]*/, off, off offset:2048 nv
	scratch_load_b128 v[110:113] /*v[878:881]*/, off, off offset:2064 nv
	s_set_vgpr_msb 0xc003
	v_mov_b64_e32 v[220:221], v[50:51] /*v[818:819]*/
	s_set_vgpr_msb 0x3ae
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[58:65] /*v[570:577]*/, v[38:45] /*v[806:813]*/, v[108:115] /*v[620:627]*/
	s_set_vgpr_msb 0xaec0
	s_clause 0x3
	scratch_load_b128 v[42:45] /*v[810:813]*/, off, off offset:1664 nv
	scratch_load_b128 v[46:49] /*v[814:817]*/, off, off offset:1680 nv
	scratch_load_b128 v[34:37] /*v[802:805]*/, off, off offset:1984 nv
	scratch_load_b128 v[38:41] /*v[806:809]*/, off, off offset:2000 nv
	s_set_vgpr_msb 0xc003
	v_mov_b64_e32 v[222:223], v[52:53] /*v[820:821]*/
	v_mov_b64_e32 v[224:225], v[54:55] /*v[822:823]*/
	v_mov_b64_e32 v[226:227], v[56:57] /*v[824:825]*/
	s_clause 0x3
	scratch_load_b128 v[92:95], off, off offset:1088 nv
	scratch_load_b128 v[96:99], off, off offset:1104 nv
	scratch_load_b128 v[0:3], off, off offset:1440 nv
	scratch_load_b128 v[4:7], off, off offset:1456 nv
	s_set_vgpr_msb 0x38e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[34:41] /*v[546:553]*/, v[50:57] /*v[818:825]*/, 0
	s_set_vgpr_msb 0x8ec0
	s_clause 0xa
	scratch_load_b128 v[50:53] /*v[818:821]*/, off, off offset:928 nv
	scratch_load_b128 v[54:57] /*v[822:825]*/, off, off offset:944 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[4:7] /*v[260:263]*/, off, off offset:1472 nv
	scratch_load_b128 v[8:11] /*v[264:267]*/, off, off offset:1488 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[8:11], off, off offset:480 nv
	scratch_load_b128 v[12:15], off, off offset:496 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[12:15] /*v[268:271]*/, off, off offset:608 nv
	scratch_load_b128 v[16:19] /*v[272:275]*/, off, off offset:624 nv
	s_set_vgpr_msb 0x40c3
	v_mov_b64_e32 v[120:121] /*v[888:889]*/, v[96:97] /*v[864:865]*/
	v_mov_b64_e32 v[118:119] /*v[886:887]*/, v[94:95] /*v[862:863]*/
	v_mov_b64_e32 v[116:117] /*v[884:885]*/, v[92:93] /*v[860:861]*/
	s_set_vgpr_msb 0xc3ae
	v_wmma_f32_16x16x32_bf16 v[116:123] /*v[628:635]*/, v[58:65] /*v[570:577]*/, v[90:97] /*v[858:865]*/, v[116:123] /*v[628:635]*/
	s_set_vgpr_msb 0xaec3
	v_mov_b64_e32 v[114:115] /*v[882:883]*/, v[90:91] /*v[858:859]*/
	s_clause 0x4
	scratch_load_b128 v[90:93] /*v[858:861]*/, off, off offset:1600 nv
	scratch_load_b128 v[94:97] /*v[862:865]*/, off, off offset:1616 nv
	s_set_vgpr_msb 0xc30a
	scratch_load_b128 v[236:239], off, off offset:1728 nv
	scratch_load_b128 v[240:243], off, off offset:1744 nv
	ds_store_b128 v199 /*v711*/, v[42:45] /*v[554:557]*/
	ds_store_b128 v199 /*v711*/, v[46:49] /*v[558:561]*/ offset:32
	ds_store_b128 v199 /*v711*/, v[58:61] /*v[570:573]*/ offset:64
	ds_store_b128 v199 /*v711*/, v[62:65] /*v[574:577]*/ offset:96
	ds_store_b128 v199 /*v711*/, v[26:29] /*v[538:541]*/ offset:128
	ds_store_b128 v199 /*v711*/, v[30:33] /*v[542:545]*/ offset:160
	ds_store_b128 v199 /*v711*/, v[10:13] /*v[522:525]*/ offset:192
	ds_store_b128 v199 /*v711*/, v[14:17] /*v[526:529]*/ offset:224
	ds_store_b128 v200 /*v712*/, v[2:5] /*v[514:517]*/
	ds_store_b128 v200 /*v712*/, v[6:9] /*v[518:521]*/ offset:32
	s_set_vgpr_msb 0xa06
	ds_store_b128 v200 /*v712*/, v[250:253] /*v[506:509]*/ offset:64
	ds_store_b128 v200 /*v712*/, v[254:257] /*v[510:513]*/ offset:96
	ds_store_b128 v200 /*v712*/, v[226:229] /*v[482:485]*/ offset:128
	ds_store_b128 v200 /*v712*/, v[230:233] /*v[486:489]*/ offset:160
	ds_store_b128 v200 /*v712*/, v[202:205] /*v[458:461]*/ offset:192
	ds_store_b128 v200 /*v712*/, v[206:209] /*v[462:465]*/ offset:224
	s_cmp_lt_i32 s24, s2
	s_add_nc_u64 s[20:21], s[20:21], -1
	s_cselect_b32 s3, s24, s26
	s_set_vgpr_msb 0x6ae
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[42:49] /*v[554:561]*/, v[130:137] /*v[898:905]*/, 0
	s_lshl_b32 s3, s3, 5
	s_add_co_i32 s24, s24, 1
	s_cmp_lg_u64 s[20:21], 0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[50:57] /*v[562:569]*/, v[226:233] /*v[994:1001]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xae8a
	s_wait_loadcnt 0x2a
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[42:49] /*v[554:561]*/, v[234:241] /*v[746:753]*/, 0
	s_wait_loadcnt 0x28
	v_wmma_f32_16x16x32_bf16 v[92:99] /*v[604:611]*/, v[42:49] /*v[554:561]*/, v[242:249] /*v[754:761]*/, 0
	s_wait_loadcnt 0x26
	v_wmma_f32_16x16x32_bf16 v[100:107] /*v[612:619]*/, v[42:49] /*v[554:561]*/, v[250:257] /*v[762:769]*/, 0
	s_set_vgpr_msb 0x8aae
	s_wait_loadcnt 0x24
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[42:49] /*v[554:561]*/, v[18:25] /*v[786:793]*/, 0
	s_wait_loadcnt 0x22
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[692:699]*/, v[42:49] /*v[554:561]*/, v[122:129] /*v[890:897]*/, 0
	s_wait_loadcnt 0x20
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[58:65] /*v[570:577]*/, v[26:33] /*v[794:801]*/, v[84:91] /*v[596:603]*/
	s_wait_loadcnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[92:99] /*v[604:611]*/, v[58:65] /*v[570:577]*/, v[58:65] /*v[826:833]*/, v[92:99] /*v[604:611]*/
	s_wait_loadcnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[100:107] /*v[612:619]*/, v[58:65] /*v[570:577]*/, v[82:89] /*v[850:857]*/, v[100:107] /*v[612:619]*/
	s_wait_loadcnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[58:65] /*v[570:577]*/, v[66:73] /*v[834:841]*/, v[124:131] /*v[636:643]*/
	s_set_vgpr_msb 0xaea2
	s_wait_loadcnt 0x18
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[58:65] /*v[570:577]*/, v[68:75], v[146:153] /*v[658:665]*/
	s_set_vgpr_msb 0xa2ae
	s_wait_loadcnt 0x12
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[692:699]*/, v[58:65] /*v[570:577]*/, v[42:49] /*v[810:817]*/, v[180:187] /*v[692:699]*/
	s_wait_loadcnt 0x10
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[34:41] /*v[546:553]*/, v[34:41] /*v[802:809]*/, 0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[34:41] /*v[546:553]*/, v[98:105] /*v[866:873]*/, 0
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[700:707]*/, v[34:41] /*v[546:553]*/, v[106:113] /*v[874:881]*/, 0
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[714:721]*/, v[34:41] /*v[546:553]*/, v[50:57] /*v[818:825]*/, 0
	s_set_vgpr_msb 0xae82
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[722:729]*/, v[34:41] /*v[546:553]*/, v[92:99], 0
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[34:41] /*v[546:553]*/, v[0:7], 0
	s_set_vgpr_msb 0x8286
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[34:41] /*v[546:553]*/, v[4:11] /*v[260:267]*/, 0
	s_set_vgpr_msb 0x86ae
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[50:57] /*v[562:569]*/, v[162:169] /*v[930:937]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xaea2
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[50:57] /*v[562:569]*/, v[8:15], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa2a6
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[700:707]*/, v[50:57] /*v[562:569]*/, v[12:19] /*v[268:275]*/, v[188:195] /*v[700:707]*/
	s_set_vgpr_msb 0xa6ae
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[714:721]*/, v[50:57] /*v[562:569]*/, v[242:249] /*v[1010:1017]*/, v[202:209] /*v[714:721]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[722:729]*/, v[50:57] /*v[562:569]*/, v[154:161] /*v[922:929]*/, v[210:217] /*v[722:729]*/
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[50:57] /*v[562:569]*/, v[90:97] /*v[858:865]*/, v[218:225] /*v[730:737]*/
	s_set_vgpr_msb 0xaea2
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[50:57] /*v[562:569]*/, v[236:243], v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa28a
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[2:9] /*v[514:521]*/, v[234:241] /*v[746:753]*/, 0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[2:9] /*v[514:521]*/, v[242:249] /*v[754:761]*/, 0
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[2:9] /*v[514:521]*/, v[250:257] /*v[762:769]*/, 0
	s_set_vgpr_msb 0x8a8e
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[2:9] /*v[514:521]*/, v[6:13] /*v[774:781]*/, 0
	s_set_vgpr_msb 0x8e8a
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[2:9] /*v[514:521]*/, v[74:81] /*v[586:593]*/, 0
	s_set_vgpr_msb 0x8ace
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[2:9] /*v[514:521]*/, v[130:137] /*v[898:905]*/, 0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[2:9] /*v[514:521]*/, v[18:25] /*v[786:793]*/, 0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[2:9] /*v[514:521]*/, v[122:129] /*v[890:897]*/, 0
	s_clause 0x1
	scratch_load_b128 v[122:125] /*v[890:893]*/, off, off offset:544 nv
	scratch_load_b128 v[126:129] /*v[894:897]*/, off, off offset:560 nv
	s_set_vgpr_msb 0xcead
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[250:257] /*v[506:513]*/, v[26:33] /*v[794:801]*/, v[34:41] /*v[546:553]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[250:257] /*v[506:513]*/, v[58:65] /*v[826:833]*/, v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[250:257] /*v[506:513]*/, v[82:89] /*v[850:857]*/, v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xadc0
	s_clause 0x1
	scratch_load_b128 v[82:85] /*v[850:853]*/, off, off offset:704 nv
	scratch_load_b128 v[86:89] /*v[854:857]*/, off, off offset:720 nv
	s_set_vgpr_msb 0xc0a1
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[250:257] /*v[506:513]*/, v[180:187], v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa100
	s_clause 0x1
	scratch_load_b128 v[180:183], off, off offset:1376 nv
	scratch_load_b128 v[184:187], off, off offset:1392 nv
	s_set_vgpr_msb 0xad
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[250:257] /*v[506:513]*/, v[114:121] /*v[882:889]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xadfd
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[250:257] /*v[506:513]*/, v[66:73] /*v[834:841]*/, v[2:9] /*v[770:777]*/
	s_set_vgpr_msb 0xfdf1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[250:257] /*v[506:513]*/, v[68:75], v[10:17] /*v[778:785]*/
	s_set_vgpr_msb 0xf100
	s_clause 0x1
	scratch_load_b128 v[68:71], off, off offset:1184 nv
	scratch_load_b128 v[72:75], off, off offset:1200 nv
	s_set_vgpr_msb 0xfd
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[250:257] /*v[506:513]*/, v[42:49] /*v[810:817]*/, v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xfd4d
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[218:225] /*v[474:481]*/, v[34:41] /*v[802:809]*/, 0
	s_set_vgpr_msb 0x4d8d
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[218:225] /*v[474:481]*/, v[98:105] /*v[866:873]*/, 0
	s_set_vgpr_msb 0x8dcd
	s_clause 0x1
	scratch_load_b128 v[98:101] /*v[866:869]*/, off, off offset:1504 nv
	scratch_load_b128 v[102:105] /*v[870:873]*/, off, off offset:1520 nv
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[794:801]*/, v[218:225] /*v[474:481]*/, v[106:113] /*v[874:881]*/, 0
	s_clause 0x1
	scratch_load_b128 v[106:109] /*v[874:877]*/, off, off offset:512 nv
	scratch_load_b128 v[110:113] /*v[878:881]*/, off, off offset:528 nv
	s_set_vgpr_msb 0xcdc1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[802:809]*/, v[218:225] /*v[474:481]*/, v[220:227], 0
	s_set_vgpr_msb 0xc100
	s_clause 0x1
	scratch_load_b128 v[220:223], off, off offset:416 nv
	scratch_load_b128 v[224:227], off, off offset:432 nv
	s_set_vgpr_msb 0xcd
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[810:817]*/, v[218:225] /*v[474:481]*/, v[50:57] /*v[818:825]*/, 0
	s_set_vgpr_msb 0xcdc1
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[818:825]*/, v[218:225] /*v[474:481]*/, v[92:99], 0
	s_set_vgpr_msb 0xc100
	s_clause 0x1
	scratch_load_b128 v[92:95], off, off offset:1120 nv
	scratch_load_b128 v[96:99], off, off offset:1136 nv
	s_set_vgpr_msb 0xc1
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[826:833]*/, v[218:225] /*v[474:481]*/, v[0:7], 0
	s_set_vgpr_msb 0xc100
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:1536 nv
	scratch_load_b128 v[4:7], off, off offset:1552 nv
	s_set_vgpr_msb 0xc5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[834:841]*/, v[218:225] /*v[474:481]*/, v[4:11] /*v[260:267]*/, 0
	s_set_vgpr_msb 0xc55d
	s_clause 0x3
	scratch_load_b128 v[4:7] /*v[260:263]*/, off, off offset:640 nv
	scratch_load_b128 v[8:11] /*v[264:267]*/, off, off offset:656 nv
	scratch_load_b128 v[218:221] /*v[474:477]*/, off, off offset:768 nv
	scratch_load_b128 v[222:225] /*v[478:481]*/, off, off offset:784 nv
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[234:241] /*v[490:497]*/, v[162:169] /*v[930:937]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x5da1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[234:241] /*v[490:497]*/, v[8:15], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa100
	s_clause 0x1
	scratch_load_b128 v[8:11], off, off offset:352 nv
	scratch_load_b128 v[12:15], off, off offset:368 nv
	s_set_vgpr_msb 0xf5
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[794:801]*/, v[234:241] /*v[490:497]*/, v[12:19] /*v[268:275]*/, v[26:33] /*v[794:801]*/
	s_set_vgpr_msb 0xf540
	s_clause 0x1
	scratch_load_b128 v[12:15] /*v[268:271]*/, off, off offset:672 nv
	scratch_load_b128 v[16:19] /*v[272:275]*/, off, off offset:688 nv
	s_set_vgpr_msb 0x40fd
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[802:809]*/, v[234:241] /*v[490:497]*/, v[226:233] /*v[994:1001]*/, v[34:41] /*v[802:809]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[810:817]*/, v[234:241] /*v[490:497]*/, v[242:249] /*v[1010:1017]*/, v[42:49] /*v[810:817]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[818:825]*/, v[234:241] /*v[490:497]*/, v[154:161] /*v[922:929]*/, v[50:57] /*v[818:825]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[826:833]*/, v[234:241] /*v[490:497]*/, v[90:97] /*v[858:865]*/, v[58:65] /*v[826:833]*/
	s_clause 0x1
	scratch_load_b128 v[90:93] /*v[858:861]*/, off, off offset:1056 nv
	scratch_load_b128 v[94:97] /*v[862:865]*/, off, off offset:1072 nv
	s_set_vgpr_msb 0xfdf1
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[834:841]*/, v[234:241] /*v[490:497]*/, v[236:243], v[66:73] /*v[834:841]*/
	s_set_vgpr_msb 0xf100
	s_clause 0x4
	scratch_load_b128 v[236:239], off, off offset:448 nv
	scratch_load_b128 v[240:243], off, off offset:464 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[234:237] /*v[490:493]*/, off, off offset:1024 nv
	scratch_load_b128 v[238:241] /*v[494:497]*/, off, off offset:1040 nv
	s_set_vgpr_msb 0x40a2
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[26:33] /*v[538:545]*/, v[8:15], v[84:91] /*v[596:603]*/
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[92:99] /*v[604:611]*/, v[26:33] /*v[538:545]*/, v[236:243], v[92:99] /*v[604:611]*/
	s_set_vgpr_msb 0xa2ae
	v_wmma_f32_16x16x32_bf16 v[100:107] /*v[612:619]*/, v[26:33] /*v[538:545]*/, v[122:129] /*v[890:897]*/, v[100:107] /*v[612:619]*/
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[26:33] /*v[538:545]*/, v[82:89] /*v[850:857]*/, v[108:115] /*v[620:627]*/
	s_set_vgpr_msb 0xaea6
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[116:123] /*v[628:635]*/, v[26:33] /*v[538:545]*/, v[234:241] /*v[490:497]*/, v[116:123] /*v[628:635]*/
	s_set_vgpr_msb 0xa6ae
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[26:33] /*v[538:545]*/, v[90:97] /*v[858:865]*/, v[124:131] /*v[636:643]*/
	s_set_vgpr_msb 0xaea2
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[26:33] /*v[538:545]*/, v[68:75], v[146:153] /*v[658:665]*/
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[692:699]*/, v[26:33] /*v[538:545]*/, v[180:187], v[180:187] /*v[692:699]*/
	s_clause 0x1
	scratch_load_b128 v[26:29] /*v[538:541]*/, off, off offset:960 nv
	scratch_load_b128 v[30:33] /*v[542:545]*/, off, off offset:976 nv
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[700:707]*/, v[18:25] /*v[530:537]*/, v[4:11] /*v[260:267]*/, v[188:195] /*v[700:707]*/
	s_set_vgpr_msb 0xa6aa
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[714:721]*/, v[18:25] /*v[530:537]*/, v[26:33] /*v[538:545]*/, v[202:209] /*v[714:721]*/
	s_set_vgpr_msb 0xaaa2
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[722:729]*/, v[18:25] /*v[530:537]*/, v[92:99], v[210:217] /*v[722:729]*/
	s_set_vgpr_msb 0xa2a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[226:233] /*v[482:489]*/, v[8:15], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa100
	s_clause 0x1
	scratch_load_b128 v[8:11], off, off offset:224 nv
	scratch_load_b128 v[12:15], off, off offset:240 nv
	s_set_vgpr_msb 0xa1
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[226:233] /*v[482:489]*/, v[236:243], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa100
	s_clause 0x1
	scratch_load_b128 v[236:239], off, off offset:320 nv
	scratch_load_b128 v[240:243], off, off offset:336 nv
	s_set_vgpr_msb 0xad
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[226:233] /*v[482:489]*/, v[122:129] /*v[890:897]*/, v[234:241] /*v[746:753]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[226:233] /*v[482:489]*/, v[82:89] /*v[850:857]*/, v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xada5
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[226:233] /*v[482:489]*/, v[234:241] /*v[490:497]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa540
	s_clause 0x1
	scratch_load_b128 v[234:237] /*v[490:493]*/, off, off offset:864 nv
	scratch_load_b128 v[238:241] /*v[494:497]*/, off, off offset:880 nv
	s_set_vgpr_msb 0x40fd
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[226:233] /*v[482:489]*/, v[90:97] /*v[858:865]*/, v[2:9] /*v[770:777]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xfdc3
	v_mov_b64_e32 v[90:91] /*v[858:859]*/, v[114:115] /*v[882:883]*/
	v_mov_b64_e32 v[92:93] /*v[860:861]*/, v[116:117] /*v[884:885]*/
	v_mov_b64_e32 v[94:95] /*v[862:863]*/, v[118:119] /*v[886:887]*/
	s_set_vgpr_msb 0xc3f1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[226:233] /*v[482:489]*/, v[68:75], v[10:17] /*v[778:785]*/
	s_set_vgpr_msb 0xf100
	s_clause 0x1
	scratch_load_b128 v[68:71], off, off offset:1568 nv
	scratch_load_b128 v[72:75], off, off offset:1584 nv
	s_set_vgpr_msb 0xc3
	v_mov_b64_e32 v[96:97] /*v[864:865]*/, v[120:121] /*v[888:889]*/
	s_set_vgpr_msb 0xc3f1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[226:233] /*v[482:489]*/, v[180:187], v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xf140
	s_clause 0x4
	scratch_load_b128 v[226:229] /*v[482:485]*/, off, off offset:992 nv
	scratch_load_b128 v[230:233] /*v[486:489]*/, off, off offset:1008 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[180:183], off, off offset:1696 nv
	scratch_load_b128 v[184:187], off, off offset:1712 nv
	s_set_vgpr_msb 0x51
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[210:217] /*v[466:473]*/, v[220:227], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51ad
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[210:217] /*v[466:473]*/, v[106:113] /*v[874:881]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xadf5
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[794:801]*/, v[210:217] /*v[466:473]*/, v[4:11] /*v[260:267]*/, v[26:33] /*v[794:801]*/
	s_set_vgpr_msb 0xf540
	s_clause 0x1
	scratch_load_b128 v[4:7] /*v[260:263]*/, off, off offset:288 nv
	scratch_load_b128 v[8:11] /*v[264:267]*/, off, off offset:304 nv
	s_set_vgpr_msb 0x40f5
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[802:809]*/, v[210:217] /*v[466:473]*/, v[218:225] /*v[474:481]*/, v[34:41] /*v[802:809]*/
	s_set_vgpr_msb 0xf5f9
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[810:817]*/, v[210:217] /*v[466:473]*/, v[26:33] /*v[538:545]*/, v[42:49] /*v[810:817]*/
	s_set_vgpr_msb 0xf980
	s_clause 0x1
	scratch_load_b128 v[26:29] /*v[538:541]*/, off, off offset:576 nv
	scratch_load_b128 v[30:33] /*v[542:545]*/, off, off offset:592 nv
	s_set_vgpr_msb 0x80f1
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[818:825]*/, v[210:217] /*v[466:473]*/, v[92:99], v[50:57] /*v[818:825]*/
	s_set_vgpr_msb 0xf100
	s_clause 0x1
	scratch_load_b128 v[92:95], off, off offset:1216 nv
	scratch_load_b128 v[96:99], off, off offset:1232 nv
	s_set_vgpr_msb 0xfd
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[826:833]*/, v[210:217] /*v[466:473]*/, v[98:105] /*v[866:873]*/, v[58:65] /*v[826:833]*/
	s_set_vgpr_msb 0xfdf1
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[834:841]*/, v[210:217] /*v[466:473]*/, v[0:7], v[66:73] /*v[834:841]*/
	s_set_vgpr_msb 0xf140
	s_clause 0x1
	scratch_load_b128 v[210:213] /*v[466:469]*/, off, off offset:896 nv
	scratch_load_b128 v[214:217] /*v[470:473]*/, off, off offset:912 nv
	s_set_vgpr_msb 0x40a2
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[18:25] /*v[530:537]*/, v[0:7], v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:1760 nv
	scratch_load_b128 v[4:7], off, off offset:1776 nv
	s_set_vgpr_msb 0xa2
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[18:25] /*v[530:537]*/, v[220:227], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[220:223], off, off offset:256 nv
	scratch_load_b128 v[224:227], off, off offset:272 nv
	s_set_vgpr_msb 0xae
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[18:25] /*v[530:537]*/, v[106:113] /*v[874:881]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xaea6
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[18:25] /*v[530:537]*/, v[218:225] /*v[474:481]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa640
	s_clause 0x1
	scratch_load_b128 v[218:221] /*v[474:477]*/, off, off offset:736 nv
	scratch_load_b128 v[222:225] /*v[478:481]*/, off, off offset:752 nv
	s_set_vgpr_msb 0x40ae
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[18:25] /*v[530:537]*/, v[98:105] /*v[866:873]*/, v[218:225] /*v[730:737]*/
	s_clause 0x1
	scratch_load_b128 v[18:21] /*v[530:533]*/, off, off offset:1632 nv
	scratch_load_b128 v[22:25] /*v[534:537]*/, off, off offset:1648 nv
	s_set_vgpr_msb 0xaea2
	s_wait_loadcnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[10:17] /*v[522:529]*/, v[8:15], v[84:91] /*v[596:603]*/
	s_set_vgpr_msb 0xa2a6
	s_wait_loadcnt 0xe
	v_wmma_f32_16x16x32_bf16 v[92:99] /*v[604:611]*/, v[10:17] /*v[522:529]*/, v[4:11] /*v[260:267]*/, v[92:99] /*v[604:611]*/
	s_set_vgpr_msb 0xa6aa
	s_wait_loadcnt 0xc
	v_wmma_f32_16x16x32_bf16 v[100:107] /*v[612:619]*/, v[10:17] /*v[522:529]*/, v[26:33] /*v[538:545]*/, v[100:107] /*v[612:619]*/
	s_set_vgpr_msb 0xaaa6
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[10:17] /*v[522:529]*/, v[210:217] /*v[466:473]*/, v[108:115] /*v[620:627]*/
	v_wmma_f32_16x16x32_bf16 v[116:123] /*v[628:635]*/, v[10:17] /*v[522:529]*/, v[226:233] /*v[482:489]*/, v[116:123] /*v[628:635]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[10:17] /*v[522:529]*/, v[92:99], v[124:131] /*v[636:643]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[10:17] /*v[522:529]*/, v[68:75], v[146:153] /*v[658:665]*/
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[692:699]*/, v[10:17] /*v[522:529]*/, v[180:187], v[180:187] /*v[692:699]*/
	s_clause 0x1
	scratch_load_b128 v[10:13] /*v[522:525]*/, off, off offset:1280 nv
	scratch_load_b128 v[14:17] /*v[526:529]*/, off, off offset:1296 nv
	s_set_vgpr_msb 0xa2a1
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[242:249] /*v[498:505]*/, v[0:7], v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[834:841]*/, v[194:201] /*v[450:457]*/, v[0:7], v[66:73] /*v[834:841]*/
	s_set_vgpr_msb 0xf100
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:160 nv
	scratch_load_b128 v[4:7], off, off offset:176 nv
	s_set_vgpr_msb 0xa5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[202:209] /*v[458:465]*/, v[4:11] /*v[260:267]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa551
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[194:201] /*v[450:457]*/, v[220:227], v[250:257] /*v[506:513]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5149
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_add_f32 v[250:251] /*v[506:507]*/, v[250:251] /*v[506:507]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x49a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[194:201] /*v[450:457]*/, v[236:243], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa149
	v_pk_add_f32 v[252:253] /*v[508:509]*/, v[252:253] /*v[508:509]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x498a
	v_pk_add_f32 v[0:1] /*v[512:513]*/, v[0:1] /*v[512:513]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_add_f32 v[6:7] /*v[518:519]*/, v[6:7] /*v[518:519]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8af5
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[794:801]*/, v[194:201] /*v[450:457]*/, v[12:19] /*v[268:275]*/, v[26:33] /*v[794:801]*/
	s_set_vgpr_msb 0xf58a
	v_pk_add_f32 v[2:3] /*v[514:515]*/, v[2:3] /*v[514:515]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5] /*v[516:517]*/, v[4:5] /*v[516:517]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9] /*v[520:521]*/, v[8:9] /*v[520:521]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8af5
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[802:809]*/, v[194:201] /*v[450:457]*/, v[218:225] /*v[474:481]*/, v[34:41] /*v[802:809]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[810:817]*/, v[194:201] /*v[450:457]*/, v[234:241] /*v[490:497]*/, v[42:49] /*v[810:817]*/
	s_set_vgpr_msb 0xf5f9
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[818:825]*/, v[194:201] /*v[450:457]*/, v[10:17] /*v[522:529]*/, v[50:57] /*v[818:825]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[826:833]*/, v[194:201] /*v[450:457]*/, v[18:25] /*v[530:537]*/, v[58:65] /*v[826:833]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xf949
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, s[4:5]
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[194:195] /*v[450:451]*/, v[84:85] /*v[596:597]*/
	s_set_vgpr_msb 0x49a1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[242:249] /*v[498:505]*/, v[220:227], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa149
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[194:195] /*v[450:451]*/, v[88:89] /*v[600:601]*/
	s_set_vgpr_msb 0x4989
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[194:195] /*v[450:451]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x8949
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[194:195] /*v[450:451]*/, v[86:87] /*v[598:599]*/
	s_set_vgpr_msb 0x4989
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[194:195] /*v[450:451]*/, v[152:153] /*v[664:665]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[194:195] /*v[450:451]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[194:195] /*v[450:451]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[194:195] /*v[450:451]*/, v[56:57] /*v[568:569]*/
	s_set_vgpr_msb 0x89a1
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[242:249] /*v[498:505]*/, v[236:243], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa14d
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4d8a
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a4d
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4d8a
	v_pk_add_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a41
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x41a5
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[700:707]*/, v[242:249] /*v[498:505]*/, v[12:19] /*v[268:275]*/, v[188:195] /*v[700:707]*/
	s_set_vgpr_msb 0xa58a
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_add_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v196 /*v452*/, v196 /*v452*/
	v_exp_f32_e32 v197 /*v453*/, v197 /*v453*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x41a5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[242:249] /*v[498:505]*/, v[218:225] /*v[474:481]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa541
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	s_set_vgpr_msb 0x4182
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8249
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[194:195] /*v[450:451]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[220:221] /*v[476:477]*/, v[58:59] /*v[570:571]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[60:61] /*v[572:573]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa5
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[714:721]*/, v[242:249] /*v[498:505]*/, v[234:241] /*v[490:497]*/, v[202:209] /*v[714:721]*/
	s_set_vgpr_msb 0xa589
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[194:195] /*v[450:451]*/, v[130:131] /*v[642:643]*/
	v_pk_mul_f32 v[98:99] /*v[610:611]*/, v[194:195] /*v[450:451]*/, v[184:185] /*v[696:697]*/
	s_set_vgpr_msb 0x8949
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4982
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8249
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[194:195] /*v[450:451]*/, v[106:107] /*v[618:619]*/
	s_set_vgpr_msb 0x494e
	v_pk_add_f32 v[236:237] /*v[492:493]*/, v[188:189] /*v[700:701]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4e8a
	v_pk_add_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa9
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[722:729]*/, v[242:249] /*v[498:505]*/, v[10:17] /*v[522:529]*/, v[210:217] /*v[722:729]*/
	s_set_vgpr_msb 0xa98a
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[98:99] /*v[610:611]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a49
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[234:235] /*v[490:491]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4982
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8289
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[194:195] /*v[450:451]*/, v[114:115] /*v[626:627]*/
	s_set_vgpr_msb 0x898e
	v_pk_add_f32 v[14:15] /*v[526:527]*/, v[68:69] /*v[580:581]*/, v[186:187] /*v[954:955]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17] /*v[528:529]*/, v[70:71] /*v[582:583]*/, v[186:187] /*v[954:955]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ea9
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[242:249] /*v[498:505]*/, v[18:25] /*v[530:537]*/, v[218:225] /*v[730:737]*/
	s_set_vgpr_msb 0xa98e
	v_pk_add_f32 v[60:61] /*v[572:573]*/, v[210:211] /*v[722:723]*/, v[190:191] /*v[958:959]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e89
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[194:195] /*v[450:451]*/, v[146:147] /*v[658:659]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[194:195] /*v[450:451]*/, v[148:149] /*v[660:661]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a49
	v_pk_mul_f32 v[244:245] /*v[500:501]*/, v[194:195] /*v[450:451]*/, v[108:109] /*v[620:621]*/
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[194:195] /*v[450:451]*/, v[110:111] /*v[622:623]*/
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[194:195] /*v[450:451]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x49a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[202:209] /*v[458:465]*/, v[8:15], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa18e
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[72:73] /*v[584:585]*/, v[186:187] /*v[954:955]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e89
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[194:195] /*v[450:451]*/, v[116:117] /*v[628:629]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[194:195] /*v[450:451]*/, v[118:119] /*v[630:631]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[194:195] /*v[450:451]*/, v[120:121] /*v[632:633]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[194:195] /*v[450:451]*/, v[150:151] /*v[662:663]*/
	s_set_vgpr_msb 0x898e
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[220:221] /*v[732:733]*/, v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e49
	v_pk_add_f32 v[244:245] /*v[500:501]*/, v[244:245] /*v[500:501]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x49a9
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[202:209] /*v[458:465]*/, v[26:33] /*v[538:545]*/, v[234:241] /*v[746:753]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[194:195] /*v[450:451]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[194:195] /*v[450:451]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[194:195] /*v[450:451]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[194:195] /*v[450:451]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[194:195] /*v[450:451]*/, v[122:123] /*v[634:635]*/
	s_set_vgpr_msb 0xa98e
	v_pk_add_f32 v[30:31] /*v[542:543]*/, v[204:205] /*v[716:717]*/, v[188:189] /*v[956:957]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33] /*v[544:545]*/, v[206:207] /*v[718:719]*/, v[188:189] /*v[956:957]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ea5
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[202:209] /*v[458:465]*/, v[210:217] /*v[466:473]*/, v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa589
	v_pk_mul_f32 v[110:111] /*v[622:623]*/, v[194:195] /*v[450:451]*/, v[234:235] /*v[746:747]*/
	v_pk_mul_f32 v[112:113] /*v[624:625]*/, v[194:195] /*v[450:451]*/, v[236:237] /*v[748:749]*/
	v_pk_mul_f32 v[114:115] /*v[626:627]*/, v[194:195] /*v[450:451]*/, v[238:239] /*v[750:751]*/
	v_pk_mul_f32 v[116:117] /*v[628:629]*/, v[194:195] /*v[450:451]*/, v[240:241] /*v[752:753]*/
	s_set_vgpr_msb 0x8949
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[194:195] /*v[450:451]*/, v[92:93] /*v[604:605]*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[194:195] /*v[450:451]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[210:211] /*v[466:467]*/, v[48:49] /*v[560:561]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa5
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[202:209] /*v[458:465]*/, v[226:233] /*v[482:489]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa549
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[194:195] /*v[450:451]*/, v[94:95] /*v[606:607]*/
	s_set_vgpr_msb 0x4989
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[194:195] /*v[450:451]*/, v[128:129] /*v[640:641]*/
	s_set_vgpr_msb 0x898e
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[224:225] /*v[736:737]*/, v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e89
	v_pk_mul_f32 v[94:95] /*v[606:607]*/, v[194:195] /*v[450:451]*/, v[180:181] /*v[692:693]*/
	s_set_vgpr_msb 0x8949
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[194:195] /*v[450:451]*/, v[100:101] /*v[612:613]*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[194:195] /*v[450:451]*/, v[102:103] /*v[614:615]*/
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[194:195] /*v[450:451]*/, v[104:105] /*v[616:617]*/
	s_set_vgpr_msb 0x49f1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[202:209] /*v[458:465]*/, v[92:99], v[2:9] /*v[770:777]*/
	s_set_vgpr_msb 0xf189
	v_pk_mul_f32 v[96:97] /*v[608:609]*/, v[194:195] /*v[450:451]*/, v[182:183] /*v[694:695]*/
	v_pk_mul_f32 v[100:101] /*v[612:613]*/, v[194:195] /*v[450:451]*/, v[186:187] /*v[698:699]*/
	s_set_vgpr_msb 0x8982
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[226:227] /*v[738:739]*/, v[18:19] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8289
	v_pk_mul_f32 v[128:129] /*v[640:641]*/, v[194:195] /*v[450:451]*/, v[244:245] /*v[756:757]*/
	v_pk_mul_f32 v[130:131] /*v[642:643]*/, v[194:195] /*v[450:451]*/, v[246:247] /*v[758:759]*/
	v_pk_mul_f32 v[146:147] /*v[658:659]*/, v[194:195] /*v[450:451]*/, v[248:249] /*v[760:761]*/
	v_pk_mul_f32 v[182:183] /*v[694:695]*/, v[194:195] /*v[450:451]*/, v[250:251] /*v[762:763]*/
	s_set_vgpr_msb 0x89f1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[202:209] /*v[458:465]*/, v[68:75], v[10:17] /*v[778:785]*/
	s_set_vgpr_msb 0xf189
	v_pk_mul_f32 v[184:185] /*v[696:697]*/, v[194:195] /*v[450:451]*/, v[252:253] /*v[764:765]*/
	v_pk_mul_f32 v[186:187] /*v[698:699]*/, v[194:195] /*v[450:451]*/, v[254:255] /*v[766:767]*/
	s_set_vgpr_msb 0x898d
	v_pk_mul_f32 v[188:189] /*v[700:701]*/, v[194:195] /*v[450:451]*/, v[0:1] /*v[768:769]*/
	v_pk_mul_f32 v[204:205] /*v[716:717]*/, v[194:195] /*v[450:451]*/, v[2:3] /*v[770:771]*/
	v_pk_mul_f32 v[206:207] /*v[718:719]*/, v[194:195] /*v[450:451]*/, v[4:5] /*v[772:773]*/
	v_pk_mul_f32 v[210:211] /*v[722:723]*/, v[194:195] /*v[450:451]*/, v[8:9] /*v[776:777]*/
	s_set_vgpr_msb 0x8d49
	v_pk_add_f32 v[212:213] /*v[468:469]*/, v[212:213] /*v[468:469]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x49f1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[202:209] /*v[458:465]*/, v[180:187], v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xf18d
	v_pk_mul_f32 v[220:221] /*v[732:733]*/, v[194:195] /*v[450:451]*/, v[10:11] /*v[778:779]*/
	v_pk_mul_f32 v[224:225] /*v[736:737]*/, v[194:195] /*v[450:451]*/, v[14:15] /*v[782:783]*/
	v_pk_mul_f32 v[226:227] /*v[738:739]*/, v[194:195] /*v[450:451]*/, v[16:17] /*v[784:785]*/
	s_set_vgpr_msb 0x8d49
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[194:195] /*v[450:451]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[42:43] /*v[554:555]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[44:45] /*v[556:557]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[46:47] /*v[558:559]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a8e
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[208:209] /*v[720:721]*/, v[188:189] /*v[956:957]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e89
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[194:195] /*v[450:451]*/, v[124:125] /*v[636:637]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[194:195] /*v[450:451]*/, v[126:127] /*v[638:639]*/
	s_set_vgpr_msb 0x898e
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[222:223] /*v[734:735]*/, v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e89
	v_pk_mul_f32 v[126:127] /*v[638:639]*/, v[194:195] /*v[450:451]*/, v[242:243] /*v[754:755]*/
	s_set_vgpr_msb 0x898d
	v_pk_mul_f32 v[208:209] /*v[720:721]*/, v[194:195] /*v[450:451]*/, v[6:7] /*v[774:775]*/
	v_pk_mul_f32 v[222:223] /*v[734:735]*/, v[194:195] /*v[450:451]*/, v[12:13] /*v[780:781]*/
	v_pk_mul_f32 v[236:237] /*v[748:749]*/, v[194:195] /*v[450:451]*/, v[18:19] /*v[786:787]*/
	v_pk_mul_f32 v[238:239] /*v[750:751]*/, v[194:195] /*v[450:451]*/, v[20:21] /*v[788:789]*/
	v_pk_mul_f32 v[240:241] /*v[752:753]*/, v[194:195] /*v[450:451]*/, v[22:23] /*v[790:791]*/
	v_pk_mul_f32 v[242:243] /*v[754:755]*/, v[194:195] /*v[450:451]*/, v[24:25] /*v[792:793]*/
	s_set_vgpr_msb 0x8d4d
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[212:213] /*v[468:469]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4d49
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_add_f32 v[228:229] /*v[484:485]*/, v[228:229] /*v[484:485]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[232:233] /*v[488:489]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[246:247] /*v[502:503]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[248:249] /*v[504:505]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x498a
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[70:71] /*v[582:583]*/, v[70:71] /*v[582:583]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[72:73] /*v[584:585]*/, v[72:73] /*v[584:585]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[94:95] /*v[606:607]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[96:97] /*v[608:609]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[100:101] /*v[612:613]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a8e
	v_pk_add_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e8a
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[110:111] /*v[622:623]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[112:113] /*v[624:625]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[114:115] /*v[626:627]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[116:117] /*v[628:629]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[126:127] /*v[638:639]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[128:129] /*v[640:641]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[130:131] /*v[642:643]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[146:147] /*v[658:659]*/, v[146:147] /*v[658:659]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[182:183] /*v[694:695]*/, v[182:183] /*v[694:695]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[184:185] /*v[696:697]*/, v[184:185] /*v[696:697]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[186:187] /*v[698:699]*/, v[186:187] /*v[698:699]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[188:189] /*v[700:701]*/, v[188:189] /*v[700:701]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[204:205] /*v[716:717]*/, v[204:205] /*v[716:717]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[206:207] /*v[718:719]*/, v[206:207] /*v[718:719]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[208:209] /*v[720:721]*/, v[208:209] /*v[720:721]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[210:211] /*v[722:723]*/, v[210:211] /*v[722:723]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[220:221] /*v[732:733]*/, v[220:221] /*v[732:733]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[222:223] /*v[734:735]*/, v[222:223] /*v[734:735]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[224:225] /*v[736:737]*/, v[224:225] /*v[736:737]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[226:227] /*v[738:739]*/, v[226:227] /*v[738:739]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[236:237] /*v[748:749]*/, v[236:237] /*v[748:749]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[238:239] /*v[750:751]*/, v[238:239] /*v[750:751]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[240:241] /*v[752:753]*/, v[240:241] /*v[752:753]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[242:243] /*v[754:755]*/, v[242:243] /*v[754:755]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v202 /*v458*/, v202 /*v458*/
	v_exp_f32_e32 v203 /*v459*/, v203 /*v459*/
	v_exp_f32_e32 v212 /*v468*/, v212 /*v468*/
	v_exp_f32_e32 v213 /*v469*/, v213 /*v469*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v216 /*v472*/, v216 /*v472*/
	v_exp_f32_e32 v217 /*v473*/, v217 /*v473*/
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[228:229] /*v[484:485]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[232:233] /*v[488:489]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[234:235] /*v[490:491]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[244:245] /*v[500:501]*/, v[244:245] /*v[500:501]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[246:247] /*v[502:503]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[248:249] /*v[504:505]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4182
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[70:71] /*v[582:583]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[72:73] /*v[584:585]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95] /*v[606:607]*/, v[94:95] /*v[606:607]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97] /*v[608:609]*/, v[96:97] /*v[608:609]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99] /*v[610:611]*/, v[98:99] /*v[610:611]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101] /*v[612:613]*/, v[100:101] /*v[612:613]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v54 /*v566*/, v54 /*v566*/
	v_exp_f32_e32 v55 /*v567*/, v55 /*v567*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111] /*v[622:623]*/, v[110:111] /*v[622:623]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113] /*v[624:625]*/, v[112:113] /*v[624:625]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115] /*v[626:627]*/, v[114:115] /*v[626:627]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117] /*v[628:629]*/, v[116:117] /*v[628:629]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127] /*v[638:639]*/, v[126:127] /*v[638:639]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129] /*v[640:641]*/, v[128:129] /*v[640:641]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131] /*v[642:643]*/, v[130:131] /*v[642:643]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147] /*v[658:659]*/, v[146:147] /*v[658:659]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183] /*v[694:695]*/, v[182:183] /*v[694:695]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[184:185] /*v[696:697]*/, v[184:185] /*v[696:697]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[186:187] /*v[698:699]*/, v[186:187] /*v[698:699]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189] /*v[700:701]*/, v[188:189] /*v[700:701]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205] /*v[716:717]*/, v[204:205] /*v[716:717]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207] /*v[718:719]*/, v[206:207] /*v[718:719]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209] /*v[720:721]*/, v[208:209] /*v[720:721]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211] /*v[722:723]*/, v[210:211] /*v[722:723]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221] /*v[732:733]*/, v[220:221] /*v[732:733]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223] /*v[734:735]*/, v[222:223] /*v[734:735]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225] /*v[736:737]*/, v[224:225] /*v[736:737]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[226:227] /*v[738:739]*/, v[226:227] /*v[738:739]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[236:237] /*v[748:749]*/, v[236:237] /*v[748:749]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[238:239] /*v[750:751]*/, v[238:239] /*v[750:751]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[240:241] /*v[752:753]*/, v[240:241] /*v[752:753]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[242:243] /*v[754:755]*/, v[242:243] /*v[754:755]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x824a
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[62:63] /*v[574:575]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a41
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	v_exp_f32_e32 v214 /*v470*/, v214 /*v470*/
	v_exp_f32_e32 v215 /*v471*/, v215 /*v471*/
	v_exp_f32_e32 v218 /*v474*/, v218 /*v474*/
	v_exp_f32_e32 v219 /*v475*/, v219 /*v475*/
	v_exp_f32_e32 v228 /*v484*/, v228 /*v484*/
	v_exp_f32_e32 v229 /*v485*/, v229 /*v485*/
	v_exp_f32_e32 v230 /*v486*/, v230 /*v486*/
	v_exp_f32_e32 v231 /*v487*/, v231 /*v487*/
	v_exp_f32_e32 v232 /*v488*/, v232 /*v488*/
	v_exp_f32_e32 v233 /*v489*/, v233 /*v489*/
	v_exp_f32_e32 v234 /*v490*/, v234 /*v490*/
	v_exp_f32_e32 v235 /*v491*/, v235 /*v491*/
	v_exp_f32_e32 v244 /*v500*/, v244 /*v500*/
	v_exp_f32_e32 v245 /*v501*/, v245 /*v501*/
	v_exp_f32_e32 v246 /*v502*/, v246 /*v502*/
	v_exp_f32_e32 v247 /*v503*/, v247 /*v503*/
	v_exp_f32_e32 v248 /*v504*/, v248 /*v504*/
	v_exp_f32_e32 v249 /*v505*/, v249 /*v505*/
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v10 /*v522*/, v10 /*v522*/
	v_exp_f32_e32 v11 /*v523*/, v11 /*v523*/
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
	v_exp_f32_e32 v22 /*v534*/, v22 /*v534*/
	v_exp_f32_e32 v23 /*v535*/, v23 /*v535*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v25 /*v537*/, v25 /*v537*/
	v_exp_f32_e32 v26 /*v538*/, v26 /*v538*/
	v_exp_f32_e32 v27 /*v539*/, v27 /*v539*/
	v_exp_f32_e32 v44 /*v556*/, v44 /*v556*/
	v_exp_f32_e32 v45 /*v557*/, v45 /*v557*/
	v_exp_f32_e32 v46 /*v558*/, v46 /*v558*/
	v_exp_f32_e32 v47 /*v559*/, v47 /*v559*/
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v49 /*v561*/, v49 /*v561*/
	v_exp_f32_e32 v58 /*v570*/, v58 /*v570*/
	v_exp_f32_e32 v59 /*v571*/, v59 /*v571*/
	v_exp_f32_e32 v68 /*v580*/, v68 /*v580*/
	v_exp_f32_e32 v69 /*v581*/, v69 /*v581*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	v_exp_f32_e32 v71 /*v583*/, v71 /*v583*/
	v_exp_f32_e32 v72 /*v584*/, v72 /*v584*/
	v_exp_f32_e32 v73 /*v585*/, v73 /*v585*/
	v_exp_f32_e32 v84 /*v596*/, v84 /*v596*/
	v_exp_f32_e32 v85 /*v597*/, v85 /*v597*/
	v_exp_f32_e32 v94 /*v606*/, v94 /*v606*/
	v_exp_f32_e32 v95 /*v607*/, v95 /*v607*/
	v_exp_f32_e32 v96 /*v608*/, v96 /*v608*/
	v_exp_f32_e32 v97 /*v609*/, v97 /*v609*/
	v_exp_f32_e32 v98 /*v610*/, v98 /*v610*/
	v_exp_f32_e32 v99 /*v611*/, v99 /*v611*/
	v_exp_f32_e32 v100 /*v612*/, v100 /*v612*/
	v_exp_f32_e32 v101 /*v613*/, v101 /*v613*/
	v_exp_f32_e32 v34 /*v546*/, v34 /*v546*/
	v_exp_f32_e32 v35 /*v547*/, v35 /*v547*/
	v_exp_f32_e32 v36 /*v548*/, v36 /*v548*/
	v_exp_f32_e32 v37 /*v549*/, v37 /*v549*/
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	v_exp_f32_e32 v39 /*v551*/, v39 /*v551*/
	v_exp_f32_e32 v40 /*v552*/, v40 /*v552*/
	v_exp_f32_e32 v41 /*v553*/, v41 /*v553*/
	v_exp_f32_e32 v50 /*v562*/, v50 /*v562*/
	v_exp_f32_e32 v51 /*v563*/, v51 /*v563*/
	v_exp_f32_e32 v52 /*v564*/, v52 /*v564*/
	v_exp_f32_e32 v53 /*v565*/, v53 /*v565*/
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v57 /*v569*/, v57 /*v569*/
	v_exp_f32_e32 v110 /*v622*/, v110 /*v622*/
	v_exp_f32_e32 v111 /*v623*/, v111 /*v623*/
	v_exp_f32_e32 v112 /*v624*/, v112 /*v624*/
	v_exp_f32_e32 v113 /*v625*/, v113 /*v625*/
	v_exp_f32_e32 v114 /*v626*/, v114 /*v626*/
	v_exp_f32_e32 v115 /*v627*/, v115 /*v627*/
	v_exp_f32_e32 v116 /*v628*/, v116 /*v628*/
	v_exp_f32_e32 v117 /*v629*/, v117 /*v629*/
	v_exp_f32_e32 v126 /*v638*/, v126 /*v638*/
	v_exp_f32_e32 v127 /*v639*/, v127 /*v639*/
	v_exp_f32_e32 v128 /*v640*/, v128 /*v640*/
	v_exp_f32_e32 v129 /*v641*/, v129 /*v641*/
	v_exp_f32_e32 v130 /*v642*/, v130 /*v642*/
	v_exp_f32_e32 v131 /*v643*/, v131 /*v643*/
	v_exp_f32_e32 v146 /*v658*/, v146 /*v658*/
	v_exp_f32_e32 v147 /*v659*/, v147 /*v659*/
	v_exp_f32_e32 v182 /*v694*/, v182 /*v694*/
	v_exp_f32_e32 v183 /*v695*/, v183 /*v695*/
	v_exp_f32_e32 v184 /*v696*/, v184 /*v696*/
	v_exp_f32_e32 v185 /*v697*/, v185 /*v697*/
	v_exp_f32_e32 v186 /*v698*/, v186 /*v698*/
	v_exp_f32_e32 v187 /*v699*/, v187 /*v699*/
	v_exp_f32_e32 v188 /*v700*/, v188 /*v700*/
	v_exp_f32_e32 v189 /*v701*/, v189 /*v701*/
	v_exp_f32_e32 v204 /*v716*/, v204 /*v716*/
	v_exp_f32_e32 v205 /*v717*/, v205 /*v717*/
	v_exp_f32_e32 v206 /*v718*/, v206 /*v718*/
	v_exp_f32_e32 v207 /*v719*/, v207 /*v719*/
	v_exp_f32_e32 v208 /*v720*/, v208 /*v720*/
	v_exp_f32_e32 v209 /*v721*/, v209 /*v721*/
	v_exp_f32_e32 v210 /*v722*/, v210 /*v722*/
	v_exp_f32_e32 v211 /*v723*/, v211 /*v723*/
	v_exp_f32_e32 v220 /*v732*/, v220 /*v732*/
	v_exp_f32_e32 v221 /*v733*/, v221 /*v733*/
	v_exp_f32_e32 v222 /*v734*/, v222 /*v734*/
	v_exp_f32_e32 v223 /*v735*/, v223 /*v735*/
	v_exp_f32_e32 v224 /*v736*/, v224 /*v736*/
	v_exp_f32_e32 v225 /*v737*/, v225 /*v737*/
	v_exp_f32_e32 v226 /*v738*/, v226 /*v738*/
	v_exp_f32_e32 v227 /*v739*/, v227 /*v739*/
	v_exp_f32_e32 v236 /*v748*/, v236 /*v748*/
	v_exp_f32_e32 v237 /*v749*/, v237 /*v749*/
	v_exp_f32_e32 v238 /*v750*/, v238 /*v750*/
	v_exp_f32_e32 v239 /*v751*/, v239 /*v751*/
	v_exp_f32_e32 v240 /*v752*/, v240 /*v752*/
	v_exp_f32_e32 v241 /*v753*/, v241 /*v753*/
	v_exp_f32_e32 v242 /*v754*/, v242 /*v754*/
	v_exp_f32_e32 v243 /*v755*/, v243 /*v755*/
	s_set_vgpr_msb 0x824a
	v_pk_add_f32 v[226:227] /*v[482:483]*/, v[64:65] /*v[576:577]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a4e
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[190:191] /*v[702:703]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[240:241] /*v[496:497]*/, v[192:193] /*v[704:705]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[242:243] /*v[498:499]*/, v[194:195] /*v[706:707]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4e8e
	v_pk_add_f32 v[12:13] /*v[524:525]*/, v[66:67] /*v[578:579]*/, v[186:187] /*v[954:955]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[202:203] /*v[714:715]*/, v[188:189] /*v[956:957]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[62:63] /*v[574:575]*/, v[212:213] /*v[724:725]*/, v[190:191] /*v[958:959]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[64:65] /*v[576:577]*/, v[214:215] /*v[726:727]*/, v[190:191] /*v[958:959]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[216:217] /*v[728:729]*/, v[190:191] /*v[958:959]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[86:87] /*v[598:599]*/, v[218:219] /*v[730:731]*/, v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e82
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[228:229] /*v[740:741]*/, v[18:19] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[230:231] /*v[742:743]*/, v[18:19] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[232:233] /*v[744:745]*/, v[18:19] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x828f
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[26:27] /*v[794:795]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[28:29] /*v[796:797]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[30:31] /*v[798:799]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[32:33] /*v[800:801]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[148:149] /*v[660:661]*/, v[34:35] /*v[802:803]*/, v[186:187] /*v[954:955]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[150:151] /*v[662:663]*/, v[36:37] /*v[804:805]*/, v[186:187] /*v[954:955]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[152:153] /*v[664:665]*/, v[38:39] /*v[806:807]*/, v[186:187] /*v[954:955]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[180:181] /*v[692:693]*/, v[40:41] /*v[808:809]*/, v[186:187] /*v[954:955]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[190:191] /*v[702:703]*/, v[42:43] /*v[810:811]*/, v[188:189] /*v[956:957]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[192:193] /*v[704:705]*/, v[44:45] /*v[812:813]*/, v[188:189] /*v[956:957]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[194:195] /*v[706:707]*/, v[46:47] /*v[814:815]*/, v[188:189] /*v[956:957]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[202:203] /*v[714:715]*/, v[48:49] /*v[816:817]*/, v[188:189] /*v[956:957]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[212:213] /*v[724:725]*/, v[50:51] /*v[818:819]*/, v[190:191] /*v[958:959]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[214:215] /*v[726:727]*/, v[52:53] /*v[820:821]*/, v[190:191] /*v[958:959]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[216:217] /*v[728:729]*/, v[54:55] /*v[822:823]*/, v[190:191] /*v[958:959]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[218:219] /*v[730:731]*/, v[56:57] /*v[824:825]*/, v[190:191] /*v[958:959]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[228:229] /*v[740:741]*/, v[58:59] /*v[826:827]*/, v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[230:231] /*v[742:743]*/, v[60:61] /*v[828:829]*/, v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[232:233] /*v[744:745]*/, v[62:63] /*v[830:831]*/, v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[234:235] /*v[746:747]*/, v[64:65] /*v[832:833]*/, v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8f83
	v_pk_add_f32 v[244:245] /*v[756:757]*/, v[66:67] /*v[834:835]*/, v[18:19] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[246:247] /*v[758:759]*/, v[68:69] /*v[836:837]*/, v[18:19] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[248:249] /*v[760:761]*/, v[70:71] /*v[838:839]*/, v[18:19] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[250:251] /*v[762:763]*/, v[72:73] /*v[840:841]*/, v[18:19] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8345
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[204:205] /*v[460:461]*/, v[196:197] /*v[452:453]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[208:209] /*v[464:465]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[210:211] /*v[466:467]*/, v[202:203] /*v[458:459]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[220:221] /*v[476:477]*/, v[212:213] /*v[468:469]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[224:225] /*v[480:481]*/, v[216:217] /*v[472:473]*/
	s_set_vgpr_msb 0x458a
	v_pk_mul_f32 v[6:7] /*v[518:519]*/, v[6:7] /*v[518:519]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x8a45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[206:207] /*v[462:463]*/, v[198:199] /*v[454:455]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[222:223] /*v[478:479]*/, v[214:215] /*v[470:471]*/
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[226:227] /*v[482:483]*/, v[218:219] /*v[474:475]*/
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[236:237] /*v[492:493]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[238:239] /*v[494:495]*/, v[230:231] /*v[486:487]*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[240:241] /*v[496:497]*/, v[232:233] /*v[488:489]*/
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[242:243] /*v[498:499]*/, v[234:235] /*v[490:491]*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[12:13] /*v[524:525]*/, v[244:245] /*v[500:501]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[14:15] /*v[526:527]*/, v[246:247] /*v[502:503]*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[16:17] /*v[528:529]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x464a
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[18:19] /*v[530:531]*/, v[10:11] /*v[522:523]*/
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[28:29] /*v[540:541]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[30:31] /*v[542:543]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[32:33] /*v[544:545]*/, v[24:25] /*v[536:537]*/
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[42:43] /*v[554:555]*/, v[26:27] /*v[538:539]*/
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[60:61] /*v[572:573]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[62:63] /*v[574:575]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[64:65] /*v[576:577]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[242:243] /*v[498:499]*/, v[66:67] /*v[578:579]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[244:245] /*v[500:501]*/, v[86:87] /*v[598:599]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[88:89] /*v[600:601]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[90:91] /*v[602:603]*/, v[72:73] /*v[584:585]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[92:93] /*v[604:605]*/, v[84:85] /*v[596:597]*/
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[102:103] /*v[614:615]*/, v[94:95] /*v[606:607]*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[104:105] /*v[616:617]*/, v[96:97] /*v[608:609]*/
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[106:107] /*v[618:619]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[108:109] /*v[620:621]*/, v[100:101] /*v[612:613]*/
	s_set_vgpr_msb 0x8a49
	v_pk_mul_f32 v[250:251] /*v[506:507]*/, v[250:251] /*v[506:507]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[252:253] /*v[508:509]*/, v[252:253] /*v[508:509]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, v[38:39] /*v[550:551]*/
	s_set_vgpr_msb 0x498a
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[0:1] /*v[512:513]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[2:3] /*v[514:515]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[4:5] /*v[516:517]*/, v[4:5] /*v[516:517]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[8:9] /*v[520:521]*/, v[8:9] /*v[520:521]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[118:119] /*v[630:631]*/, v[110:111] /*v[622:623]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[120:121] /*v[632:633]*/, v[112:113] /*v[624:625]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[122:123] /*v[634:635]*/, v[114:115] /*v[626:627]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[124:125] /*v[636:637]*/, v[116:117] /*v[628:629]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[148:149] /*v[660:661]*/, v[126:127] /*v[638:639]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[150:151] /*v[662:663]*/, v[128:129] /*v[640:641]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[152:153] /*v[664:665]*/, v[130:131] /*v[642:643]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[180:181] /*v[692:693]*/, v[146:147] /*v[658:659]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[190:191] /*v[702:703]*/, v[182:183] /*v[694:695]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[192:193] /*v[704:705]*/, v[184:185] /*v[696:697]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[194:195] /*v[706:707]*/, v[186:187] /*v[698:699]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[202:203] /*v[714:715]*/, v[188:189] /*v[700:701]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[212:213] /*v[724:725]*/, v[204:205] /*v[716:717]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[214:215] /*v[726:727]*/, v[206:207] /*v[718:719]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[216:217] /*v[728:729]*/, v[208:209] /*v[720:721]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[218:219] /*v[730:731]*/, v[210:211] /*v[722:723]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[228:229] /*v[740:741]*/, v[220:221] /*v[732:733]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[230:231] /*v[742:743]*/, v[222:223] /*v[734:735]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[232:233] /*v[744:745]*/, v[224:225] /*v[736:737]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[234:235] /*v[746:747]*/, v[226:227] /*v[738:739]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[244:245] /*v[756:757]*/, v[236:237] /*v[748:749]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[246:247] /*v[758:759]*/, v[238:239] /*v[750:751]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[248:249] /*v[760:761]*/, v[240:241] /*v[752:753]*/
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[250:251] /*v[762:763]*/, v[242:243] /*v[754:755]*/
	s_set_vgpr_msb 0x8a45
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[194:195] /*v[450:451]*/, v[196:197] /*v[452:453]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[194:195] /*v[450:451]*/, v[202:203] /*v[458:459]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[194:195] /*v[450:451]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[194:195] /*v[450:451]*/, v[208:209] /*v[464:465]*/
	s_set_vgpr_msb 0x4589
	v_pk_mul_f32 v[6:7] /*v[518:519]*/, v[194:195] /*v[450:451]*/, v[6:7] /*v[518:519]*/
	s_set_vgpr_msb 0x8945
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[194:195] /*v[450:451]*/, v[198:199] /*v[454:455]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[194:195] /*v[450:451]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[194:195] /*v[450:451]*/, v[206:207] /*v[462:463]*/
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[194:195] /*v[450:451]*/, v[210:211] /*v[466:467]*/
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[194:195] /*v[450:451]*/, v[212:213] /*v[468:469]*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[194:195] /*v[450:451]*/, v[214:215] /*v[470:471]*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[194:195] /*v[450:451]*/, v[216:217] /*v[472:473]*/
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[194:195] /*v[450:451]*/, v[218:219] /*v[474:475]*/
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[194:195] /*v[450:451]*/, v[220:221] /*v[476:477]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[194:195] /*v[450:451]*/, v[222:223] /*v[478:479]*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[194:195] /*v[450:451]*/, v[224:225] /*v[480:481]*/
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[194:195] /*v[450:451]*/, v[226:227] /*v[482:483]*/
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[194:195] /*v[450:451]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[194:195] /*v[450:451]*/, v[230:231] /*v[486:487]*/
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[194:195] /*v[450:451]*/, v[232:233] /*v[488:489]*/
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[194:195] /*v[450:451]*/, v[234:235] /*v[490:491]*/
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[194:195] /*v[450:451]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[194:195] /*v[450:451]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[194:195] /*v[450:451]*/, v[240:241] /*v[496:497]*/
	v_pk_mul_f32 v[242:243] /*v[498:499]*/, v[194:195] /*v[450:451]*/, v[242:243] /*v[498:499]*/
	v_pk_mul_f32 v[244:245] /*v[500:501]*/, v[194:195] /*v[450:451]*/, v[244:245] /*v[500:501]*/
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[194:195] /*v[450:451]*/, v[246:247] /*v[502:503]*/
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[194:195] /*v[450:451]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x4589
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[194:195] /*v[450:451]*/, v[10:11] /*v[522:523]*/
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[194:195] /*v[450:451]*/, v[12:13] /*v[524:525]*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[194:195] /*v[450:451]*/, v[14:15] /*v[526:527]*/
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[194:195] /*v[450:451]*/, v[16:17] /*v[528:529]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[194:195] /*v[450:451]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x8985
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[194:195] /*v[450:451]*/, v[250:251] /*v[506:507]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[194:195] /*v[450:451]*/, v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x8545
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[194:195] /*v[450:451]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x4589
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[194:195] /*v[450:451]*/, v[0:1] /*v[512:513]*/
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[194:195] /*v[450:451]*/, v[2:3] /*v[514:515]*/
	v_pk_mul_f32 v[4:5] /*v[516:517]*/, v[194:195] /*v[450:451]*/, v[4:5] /*v[516:517]*/
	v_pk_mul_f32 v[8:9] /*v[520:521]*/, v[194:195] /*v[450:451]*/, v[8:9] /*v[520:521]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[194:195] /*v[450:451]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[194:195] /*v[450:451]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[194:195] /*v[450:451]*/, v[24:25] /*v[536:537]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[194:195] /*v[450:451]*/, v[26:27] /*v[538:539]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[194:195] /*v[450:451]*/, v[28:29] /*v[540:541]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[194:195] /*v[450:451]*/, v[30:31] /*v[542:543]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[194:195] /*v[450:451]*/, v[32:33] /*v[544:545]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[194:195] /*v[450:451]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[194:195] /*v[450:451]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[194:195] /*v[450:451]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[194:195] /*v[450:451]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[194:195] /*v[450:451]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[194:195] /*v[450:451]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[194:195] /*v[450:451]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[194:195] /*v[450:451]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[194:195] /*v[450:451]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[194:195] /*v[450:451]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[194:195] /*v[450:451]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[194:195] /*v[450:451]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[194:195] /*v[450:451]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[194:195] /*v[450:451]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[194:195] /*v[450:451]*/, v[62:63] /*v[574:575]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[194:195] /*v[450:451]*/, v[64:65] /*v[576:577]*/
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[194:195] /*v[450:451]*/, v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x8945
	v_cvt_pk_bf16_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v208 /*v464*/, v209 /*v465*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v208 /*v464*/, v6 /*v518*/, v7 /*v519*/
	s_set_vgpr_msb 0x4a8a
	v_add_nc_u32_e32 v6 /*v518*/, v135 /*v647*/, v136 /*v648*/
	s_set_vgpr_msb 0x8a45
	v_cvt_pk_bf16_f32 v203 /*v459*/, v206 /*v462*/, v207 /*v463*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v206 /*v462*/, v2 /*v514*/, v3 /*v515*/
	v_cvt_pk_bf16_f32 v207 /*v463*/, v4 /*v516*/, v5 /*v517*/
	v_cvt_pk_bf16_f32 v209 /*v465*/, v8 /*v520*/, v9 /*v521*/
	s_set_vgpr_msb 0x4a82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v6 /*v518*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v6 /*v518*/ offset:4352
	s_set_vgpr_msb 0x8245
	v_cvt_pk_bf16_f32 v195 /*v451*/, v198 /*v454*/, v199 /*v455*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v200 /*v456*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v210 /*v466*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v210 /*v466*/, v212 /*v468*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v211 /*v467*/, v214 /*v470*/, v215 /*v471*/
	v_cvt_pk_bf16_f32 v212 /*v468*/, v216 /*v472*/, v217 /*v473*/
	v_cvt_pk_bf16_f32 v213 /*v469*/, v218 /*v474*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v218 /*v474*/, v220 /*v476*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v219 /*v475*/, v222 /*v478*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v220 /*v476*/, v224 /*v480*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v221 /*v477*/, v226 /*v482*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v226 /*v482*/, v228 /*v484*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v227 /*v483*/, v230 /*v486*/, v231 /*v487*/
	v_cvt_pk_bf16_f32 v228 /*v484*/, v232 /*v488*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v229 /*v485*/, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v234 /*v490*/, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v235 /*v491*/, v238 /*v494*/, v239 /*v495*/
	v_cvt_pk_bf16_f32 v236 /*v492*/, v240 /*v496*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v237 /*v493*/, v242 /*v498*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v242 /*v498*/, v244 /*v500*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v243 /*v499*/, v246 /*v502*/, v247 /*v503*/
	v_cvt_pk_bf16_f32 v244 /*v500*/, v248 /*v504*/, v249 /*v505*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v245 /*v501*/, v10 /*v522*/, v11 /*v523*/
	v_cvt_pk_bf16_f32 v250 /*v506*/, v12 /*v524*/, v13 /*v525*/
	v_cvt_pk_bf16_f32 v251 /*v507*/, v14 /*v526*/, v15 /*v527*/
	v_cvt_pk_bf16_f32 v252 /*v508*/, v16 /*v528*/, v17 /*v529*/
	v_cvt_pk_bf16_f32 v253 /*v509*/, v18 /*v530*/, v19 /*v531*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v68 /*v580*/, v69 /*v581*/
	v_cvt_pk_bf16_f32 v199 /*v455*/, v70 /*v582*/, v71 /*v583*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v200 /*v456*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v201 /*v457*/, v0 /*v512*/, v1 /*v513*/
	v_cvt_pk_bf16_f32 v214 /*v470*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v215 /*v471*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v216 /*v472*/, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v217 /*v473*/, v26 /*v538*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v222 /*v478*/, v28 /*v540*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v223 /*v479*/, v30 /*v542*/, v31 /*v543*/
	v_cvt_pk_bf16_f32 v224 /*v480*/, v32 /*v544*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v225 /*v481*/, v34 /*v546*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v230 /*v486*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v231 /*v487*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v232 /*v488*/, v40 /*v552*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v233 /*v489*/, v42 /*v554*/, v43 /*v555*/
	v_cvt_pk_bf16_f32 v238 /*v494*/, v44 /*v556*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v239 /*v495*/, v46 /*v558*/, v47 /*v559*/
	v_cvt_pk_bf16_f32 v240 /*v496*/, v48 /*v560*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v241 /*v497*/, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v246 /*v502*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v247 /*v503*/, v54 /*v566*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v248 /*v504*/, v56 /*v568*/, v57 /*v569*/
	v_cvt_pk_bf16_f32 v249 /*v505*/, v58 /*v570*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v254 /*v510*/, v60 /*v572*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v255 /*v511*/, v62 /*v574*/, v63 /*v575*/
	s_set_vgpr_msb 0x4a8a
	v_cvt_pk_bf16_f32 v0 /*v512*/, v64 /*v576*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v1 /*v513*/, v66 /*v578*/, v67 /*v579*/
	s_set_vgpr_msb 0x8a59
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x59c0
	s_clause 0x5
	scratch_load_b128 v[38:41] /*v[806:809]*/, off, off offset:1824 nv
	scratch_load_b128 v[42:45] /*v[810:813]*/, off, off offset:1840 nv
	scratch_load_b128 v[50:53] /*v[818:821]*/, off, off offset:1856 nv
	scratch_load_b128 v[54:57] /*v[822:825]*/, off, off offset:1872 nv
	scratch_load_b128 v[6:9] /*v[774:777]*/, off, off offset:1792 nv
	scratch_load_b128 v[10:13] /*v[778:781]*/, off, off offset:1808 nv
	s_set_vgpr_msb 0xc009
	v_wmma_f32_16x16x32_bf16 v[244:251], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[244:251]
	v_wmma_f32_16x16x32_bf16 v[172:179], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[172:179]
	s_set_vgpr_msb 0x9f9
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[986:993]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[218:225] /*v[986:993]*/
	s_set_vgpr_msb 0xf959
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[276:283]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[20:27] /*v[276:283]*/
	s_set_vgpr_msb 0x5909
	v_wmma_f32_16x16x32_bf16 v[108:115], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[108:115]
	s_set_vgpr_msb 0x9f9
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[938:945]*/, v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[170:177] /*v[938:945]*/
	s_set_vgpr_msb 0xf982
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v134 /*v646*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v134 /*v646*/ offset:4352
	s_set_vgpr_msb 0x8209
	s_wait_loadcnt_dscnt 0x600
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x3
	scratch_store_b128 off, v[0:3], off offset:160 nv
	scratch_store_b128 off, v[4:7], off offset:176 nv
	scratch_load_b128 v[0:3], off, off offset:128 nv
	scratch_load_b128 v[4:7], off, off offset:144 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x59f9
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[1002:1009]*/, v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[234:241] /*v[1002:1009]*/
	s_set_vgpr_msb 0xf909
	v_wmma_f32_16x16x32_bf16 v[164:171], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[164:171]
	v_wmma_f32_16x16x32_bf16 v[100:107], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[100:107]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v137 /*v649*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v137 /*v649*/ offset:4352
	s_set_vgpr_msb 0x8209
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x3
	scratch_store_b128 off, v[0:3], off offset:128 nv
	scratch_store_b128 off, v[4:7], off offset:144 nv
	scratch_load_b128 v[0:3], off, off offset:96 nv
	scratch_load_b128 v[4:7], off, off offset:112 nv
	s_set_vgpr_msb 0xf9
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[970:977]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[202:209] /*v[970:977]*/
	s_set_vgpr_msb 0xf959
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5909
	v_wmma_f32_16x16x32_bf16 v[228:235], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[228:235]
	v_wmma_f32_16x16x32_bf16 v[156:163], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[156:163]
	v_wmma_f32_16x16x32_bf16 v[84:91], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[84:91]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v196 /*v708*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v196 /*v708*/ offset:4352
	s_set_vgpr_msb 0x8209
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x3
	scratch_store_b128 off, v[0:3], off offset:96 nv
	scratch_store_b128 off, v[4:7], off offset:112 nv
	scratch_load_b128 v[0:3], off, off offset:64 nv
	scratch_load_b128 v[4:7], off, off offset:80 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[138:145] /*v[394:401]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5909
	v_wmma_f32_16x16x32_bf16 v[196:203], v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[196:203]
	v_wmma_f32_16x16x32_bf16 v[212:219], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[212:219]
	v_wmma_f32_16x16x32_bf16 v[148:155], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[148:155]
	v_wmma_f32_16x16x32_bf16 v[60:67], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[60:67]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v197 /*v709*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v197 /*v709*/ offset:4352
	s_set_vgpr_msb 0x8209
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x3
	scratch_store_b128 off, v[0:3], off offset:64 nv
	scratch_store_b128 off, v[4:7], off offset:80 nv
	scratch_load_b128 v[0:3], off, off offset:32 nv
	scratch_load_b128 v[4:7], off, off offset:48 nv
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[204:211], v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[204:211]
	s_set_vgpr_msb 0x959
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x59f9
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[978:985]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[210:217] /*v[978:985]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[914:921]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[146:153] /*v[914:921]*/
	s_set_vgpr_msb 0xf909
	v_wmma_f32_16x16x32_bf16 v[28:35], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[28:35]
	v_wmma_f32_16x16x32_bf16 v[140:147], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[140:147]
	v_wmma_f32_16x16x32_bf16 v[76:83], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[76:83]
	s_set_vgpr_msb 0x983
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v201 /*v969*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v201 /*v969*/ offset:4352
	s_set_vgpr_msb 0x8309
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x3
	scratch_store_b128 off, v[0:3], off offset:32 nv
	scratch_store_b128 off, v[4:7], off offset:48 nv
	scratch_load_b128 v[0:3], off, off nv
	scratch_load_b128 v[4:7], off, off offset:16 nv
	s_set_vgpr_msb 0xa9
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[650:657]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[138:145] /*v[650:657]*/
	s_set_vgpr_msb 0xa959
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[58:65] /*v[314:321]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[666:673]*/, v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[154:161] /*v[666:673]*/
	s_set_vgpr_msb 0xa909
	v_wmma_f32_16x16x32_bf16 v[132:139], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[132:139]
	v_wmma_f32_16x16x32_bf16 v[52:59], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[52:59]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v82 /*v594*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v82 /*v594*/ offset:4352
	s_set_vgpr_msb 0x8209
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x3
	scratch_store_b128 off, v[0:3], off nv
	scratch_store_b128 off, v[4:7], off offset:16 nv
	scratch_load_b128 v[0:3], off, off offset:192 nv
	scratch_load_b128 v[4:7], off, off offset:208 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x59f9
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[906:913]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[138:145] /*v[906:913]*/
	s_set_vgpr_msb 0xf959
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x59f9
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[960:967]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[192:199] /*v[960:967]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[946:953]*/, v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[178:185] /*v[946:953]*/
	s_set_vgpr_msb 0xf909
	v_wmma_f32_16x16x32_bf16 v[124:131], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[124:131]
	v_wmma_f32_16x16x32_bf16 v[44:51], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[44:51]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v83 /*v595*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v83 /*v595*/ offset:4352
	s_set_vgpr_msb 0x8259
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[170:177] /*v[426:433]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_or_b32_e32 v194 /*v450*/, s3, v132 /*v644*/
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_lo_u32 v194 /*v450*/, v194 /*v450*/, s25
	s_set_vgpr_msb 0x5909
	v_wmma_f32_16x16x32_bf16 v[116:123], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[116:123]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x959
	v_add_lshl_u32 v238 /*v494*/, v194 /*v450*/, v133 /*v645*/, 4
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[106:113] /*v[362:369]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5944
	v_or_b32_e32 v206 /*v462*/, 0xe0, v238 /*v494*/
	v_or_b32_e32 v207 /*v463*/, 0xc0, v238 /*v494*/
	s_set_vgpr_msb 0x4409
	v_wmma_f32_16x16x32_bf16 v[252:259], v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[252:259]
	v_wmma_f32_16x16x32_bf16 v[20:27], v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[20:27]
	v_wmma_f32_16x16x32_bf16 v[188:195], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[188:195]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v230 /*v486*/, 0xa0, v238 /*v494*/
	v_or_b32_e32 v231 /*v487*/, 0x80, v238 /*v494*/
	s_set_vgpr_msb 0x4409
	v_wmma_f32_16x16x32_bf16 v[36:43], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[36:43]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v246 /*v502*/, 0x60, v238 /*v494*/
	v_or_b32_e32 v247 /*v503*/, 64, v238 /*v494*/
	s_set_vgpr_msb 0x4409
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:192 nv
	scratch_store_b128 off, v[4:7], off offset:208 nv
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x84
	v_or_b32_e32 v6 /*v518*/, 32, v238 /*v494*/
	s_set_vgpr_msb 0x8441
	s_clause 0x2
	buffer_load_b128 v[194:197] /*v[450:453]*/, v238 /*v494*/, s[16:19], null offen
	s_set_vgpr_msb 0x4142
	buffer_load_b128 v[222:225] /*v[478:481]*/, v6 /*v518*/, s[16:19], null offen
	s_set_vgpr_msb 0x4249
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[194:195] /*v[450:451]*/
	v_or_b32_e32 v194 /*v450*/, s3, v201 /*v713*/
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[196:197] /*v[452:453]*/
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_lo_u32 v194 /*v450*/, v194 /*v450*/, s25
	s_set_vgpr_msb 0x4989
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v50 /*v562*/, v194 /*v450*/, v133 /*v645*/, 4
	v_or_b32_e32 v66 /*v578*/, 32, v50 /*v562*/
	s_set_vgpr_msb 0x8942
	s_clause 0x2
	buffer_load_b128 v[194:197] /*v[450:453]*/, v50 /*v562*/, s[16:19], null offen
	s_set_vgpr_msb 0x428a
	buffer_load_b128 v[38:41] /*v[550:553]*/, v66 /*v578*/, s[16:19], null offen
	v_or_b32_e32 v14 /*v526*/, 0xe0, v50 /*v562*/
	v_or_b32_e32 v15 /*v527*/, 0xc0, v50 /*v562*/
	v_or_b32_e32 v30 /*v542*/, 0xa0, v50 /*v562*/
	v_or_b32_e32 v31 /*v543*/, 0x80, v50 /*v562*/
	v_or_b32_e32 v58 /*v570*/, 0x60, v50 /*v562*/
	v_or_b32_e32 v59 /*v571*/, 64, v50 /*v562*/
	s_set_vgpr_msb 0x8a81
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[34:35] /*v[546:547]*/, v[194:195] /*v[450:451]*/
	v_mov_b64_e32 v[36:37] /*v[548:549]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x8141
	buffer_load_b128 v[194:197] /*v[450:453]*/, v238 /*v494*/, s[12:15], null offen
	s_set_vgpr_msb 0x4181
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[2:3] /*v[514:515]*/, v[194:195] /*v[450:451]*/
	v_mov_b64_e32 v[4:5] /*v[516:517]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x8142
	buffer_load_b128 v[194:197] /*v[450:453]*/, v50 /*v562*/, s[12:15], null offen
	s_set_vgpr_msb 0x4281
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[42:43] /*v[554:555]*/, v[194:195] /*v[450:451]*/
	v_mov_b64_e32 v[44:45] /*v[556:557]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x8141
	s_clause 0x1
	buffer_load_b128 v[194:197] /*v[450:453]*/, v206 /*v462*/, s[16:19], null offen
	buffer_load_b128 v[202:205] /*v[458:461]*/, v207 /*v463*/, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[198:199] /*v[454:455]*/, v[194:195] /*v[450:451]*/
	v_mov_b64_e32 v[200:201] /*v[456:457]*/, v[196:197] /*v[452:453]*/
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, v[202:203] /*v[458:459]*/
	v_mov_b64_e32 v[196:197] /*v[452:453]*/, v[204:205] /*v[460:461]*/
	s_clause 0x1
	buffer_load_b128 v[202:205] /*v[458:461]*/, v206 /*v462*/, s[12:15], null offen
	buffer_load_b128 v[210:213] /*v[466:469]*/, v207 /*v463*/, s[12:15], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mov_b64_e32 v[206:207] /*v[462:463]*/, v[202:203] /*v[458:459]*/
	v_mov_b64_e32 v[208:209] /*v[464:465]*/, v[204:205] /*v[460:461]*/
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[202:203] /*v[458:459]*/, v[210:211] /*v[466:467]*/
	v_mov_b64_e32 v[204:205] /*v[460:461]*/, v[212:213] /*v[468:469]*/
	s_clause 0x1
	buffer_load_b128 v[210:213] /*v[466:469]*/, v230 /*v486*/, s[16:19], null offen
	buffer_load_b128 v[226:229] /*v[482:485]*/, v231 /*v487*/, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[214:215] /*v[470:471]*/, v[210:211] /*v[466:467]*/
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[212:213] /*v[468:469]*/
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[226:227] /*v[482:483]*/
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[228:229] /*v[484:485]*/
	s_clause 0x1
	buffer_load_b128 v[226:229] /*v[482:485]*/, v230 /*v486*/, s[12:15], null offen
	buffer_load_b128 v[234:237] /*v[490:493]*/, v231 /*v487*/, s[12:15], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[226:227] /*v[482:483]*/
	v_mov_b64_e32 v[232:233] /*v[488:489]*/, v[228:229] /*v[484:485]*/
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[234:235] /*v[490:491]*/
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[236:237] /*v[492:493]*/
	s_clause 0x1
	buffer_load_b128 v[234:237] /*v[490:493]*/, v246 /*v502*/, s[16:19], null offen
	buffer_load_b128 v[242:245] /*v[498:501]*/, v247 /*v503*/, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[238:239] /*v[494:495]*/, v[234:235] /*v[490:491]*/
	v_mov_b64_e32 v[240:241] /*v[496:497]*/, v[236:237] /*v[492:493]*/
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[234:235] /*v[490:491]*/, v[242:243] /*v[498:499]*/
	v_mov_b64_e32 v[236:237] /*v[492:493]*/, v[244:245] /*v[500:501]*/
	s_clause 0x1
	buffer_load_b128 v[242:245] /*v[498:501]*/, v246 /*v502*/, s[12:15], null offen
	buffer_load_b128 v[246:249] /*v[502:505]*/, v247 /*v503*/, s[12:15], null offen
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[254:255] /*v[510:511]*/, v[242:243] /*v[498:499]*/
	s_set_vgpr_msb 0x4181
	v_mov_b64_e32 v[0:1] /*v[512:513]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x8142
	buffer_load_b128 v[242:245] /*v[498:501]*/, v6 /*v518*/, s[12:15], null offen
	s_set_vgpr_msb 0x4241
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[250:251] /*v[506:507]*/, v[246:247] /*v[502:503]*/
	v_mov_b64_e32 v[252:253] /*v[508:509]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x4181
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[6:7] /*v[518:519]*/, v[242:243] /*v[498:499]*/
	v_mov_b64_e32 v[8:9] /*v[520:521]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x8142
	s_clause 0x2
	buffer_load_b128 v[242:245] /*v[498:501]*/, v14 /*v526*/, s[16:19], null offen
	s_set_vgpr_msb 0x4282
	buffer_load_b128 v[10:13] /*v[522:525]*/, v15 /*v527*/, s[16:19], null offen
	s_set_vgpr_msb 0x8241
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[246:247] /*v[502:503]*/, v[242:243] /*v[498:499]*/
	v_mov_b64_e32 v[248:249] /*v[504:505]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x4142
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[242:243] /*v[498:499]*/, v[10:11] /*v[522:523]*/
	v_mov_b64_e32 v[244:245] /*v[500:501]*/, v[12:13] /*v[524:525]*/
	s_set_vgpr_msb 0x4282
	s_clause 0x1
	buffer_load_b128 v[10:13] /*v[522:525]*/, v14 /*v526*/, s[12:15], null offen
	buffer_load_b128 v[18:21] /*v[530:533]*/, v15 /*v527*/, s[12:15], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mov_b64_e32 v[14:15] /*v[526:527]*/, v[10:11] /*v[522:523]*/
	v_mov_b64_e32 v[16:17] /*v[528:529]*/, v[12:13] /*v[524:525]*/
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[10:11] /*v[522:523]*/, v[18:19] /*v[530:531]*/
	v_mov_b64_e32 v[12:13] /*v[524:525]*/, v[20:21] /*v[532:533]*/
	s_clause 0x1
	buffer_load_b128 v[18:21] /*v[530:533]*/, v30 /*v542*/, s[16:19], null offen
	buffer_load_b128 v[26:29] /*v[538:541]*/, v31 /*v543*/, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[22:23] /*v[534:535]*/, v[18:19] /*v[530:531]*/
	v_mov_b64_e32 v[24:25] /*v[536:537]*/, v[20:21] /*v[532:533]*/
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[18:19] /*v[530:531]*/, v[26:27] /*v[538:539]*/
	v_mov_b64_e32 v[20:21] /*v[532:533]*/, v[28:29] /*v[540:541]*/
	s_clause 0x1
	buffer_load_b128 v[26:29] /*v[538:541]*/, v30 /*v542*/, s[12:15], null offen
	buffer_load_b128 v[46:49] /*v[558:561]*/, v31 /*v543*/, s[12:15], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_mov_b64_e32 v[30:31] /*v[542:543]*/, v[26:27] /*v[538:539]*/
	v_mov_b64_e32 v[32:33] /*v[544:545]*/, v[28:29] /*v[540:541]*/
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[26:27] /*v[538:539]*/, v[46:47] /*v[558:559]*/
	v_mov_b64_e32 v[28:29] /*v[540:541]*/, v[48:49] /*v[560:561]*/
	s_clause 0x1
	buffer_load_b128 v[46:49] /*v[558:561]*/, v58 /*v570*/, s[16:19], null offen
	buffer_load_b128 v[50:53] /*v[562:565]*/, v59 /*v571*/, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[54:55] /*v[566:567]*/, v[46:47] /*v[558:559]*/
	v_mov_b64_e32 v[56:57] /*v[568:569]*/, v[48:49] /*v[560:561]*/
	s_clause 0x1
	buffer_load_b128 v[46:49] /*v[558:561]*/, v58 /*v570*/, s[12:15], null offen
	buffer_load_b128 v[58:61] /*v[570:573]*/, v59 /*v571*/, s[12:15], null offen
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[62:63] /*v[574:575]*/, v[46:47] /*v[558:559]*/
	v_mov_b64_e32 v[64:65] /*v[576:577]*/, v[48:49] /*v[560:561]*/
	buffer_load_b128 v[46:49] /*v[558:561]*/, v66 /*v578*/, s[12:15], null offen
	s_set_vgpr_msb 0x8200
	s_cbranch_scc1 .LBB0_2
	s_set_vgpr_msb 64
	s_clause 0x38
	scratch_load_b128 v[194:197] /*v[450:453]*/, off, off offset:704 nv
	scratch_load_b128 v[198:201] /*v[454:457]*/, off, off offset:720 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[58:61] /*v[826:829]*/, off, off offset:896 nv
	scratch_load_b128 v[62:65] /*v[830:833]*/, off, off offset:912 nv
	scratch_load_b128 v[114:117] /*v[882:885]*/, off, off offset:800 nv
	scratch_load_b128 v[118:121] /*v[886:889]*/, off, off offset:816 nv
	scratch_load_b128 v[66:69] /*v[834:837]*/, off, off offset:768 nv
	scratch_load_b128 v[70:73] /*v[838:841]*/, off, off offset:784 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[92:95], off, off offset:736 nv
	scratch_load_b128 v[96:99], off, off offset:752 nv
	s_set_vgpr_msb 0x80
	scratch_load_b128 v[250:253] /*v[762:765]*/, off, off offset:1024 nv
	scratch_load_b128 v[254:257] /*v[766:769]*/, off, off offset:1040 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[98:101] /*v[866:869]*/, off, off offset:992 nv
	scratch_load_b128 v[102:105] /*v[870:873]*/, off, off offset:1008 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[68:71], off, off offset:928 nv
	scratch_load_b128 v[72:75], off, off offset:944 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b128 v[154:157] /*v[922:925]*/, off, off offset:832 nv
	scratch_load_b128 v[158:161] /*v[926:929]*/, off, off offset:848 nv
	scratch_load_b128 v[30:33] /*v[798:801]*/, off, off offset:960 nv
	scratch_load_b128 v[34:37] /*v[802:805]*/, off, off offset:976 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[202:205] /*v[458:461]*/, off, off offset:864 nv
	scratch_load_b128 v[206:209] /*v[462:465]*/, off, off offset:880 nv
	scratch_load_b128 v[210:213] /*v[466:469]*/, off, off offset:1248 nv
	scratch_load_b128 v[214:217] /*v[470:473]*/, off, off offset:1264 nv
	scratch_load_b128 v[218:221] /*v[474:477]*/, off, off offset:1056 nv
	scratch_load_b128 v[222:225] /*v[478:481]*/, off, off offset:1072 nv
	scratch_load_b128 v[226:229] /*v[482:485]*/, off, off offset:1216 nv
	scratch_load_b128 v[230:233] /*v[486:489]*/, off, off offset:1232 nv
	scratch_load_b128 v[234:237] /*v[490:493]*/, off, off offset:1088 nv
	scratch_load_b128 v[238:241] /*v[494:497]*/, off, off offset:1104 nv
	scratch_load_b128 v[242:245] /*v[498:501]*/, off, off offset:1408 nv
	scratch_load_b128 v[246:249] /*v[502:505]*/, off, off offset:1424 nv
	scratch_load_b128 v[250:253] /*v[506:509]*/, off, off offset:1120 nv
	scratch_load_b128 v[254:257] /*v[510:513]*/, off, off offset:1136 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[2:5] /*v[514:517]*/, off, off offset:1280 nv
	scratch_load_b128 v[6:9] /*v[518:521]*/, off, off offset:1296 nv
	scratch_load_b128 v[10:13] /*v[522:525]*/, off, off offset:1152 nv
	scratch_load_b128 v[14:17] /*v[526:529]*/, off, off offset:1168 nv
	scratch_load_b128 v[18:21] /*v[530:533]*/, off, off offset:1312 nv
	scratch_load_b128 v[22:25] /*v[534:537]*/, off, off offset:1328 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[82:85] /*v[850:853]*/, off, off offset:1184 nv
	scratch_load_b128 v[86:89] /*v[854:857]*/, off, off offset:1200 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[26:29] /*v[538:541]*/, off, off offset:1568 nv
	scratch_load_b128 v[30:33] /*v[542:545]*/, off, off offset:1584 nv
	scratch_load_b128 v[34:37] /*v[546:549]*/, off, off offset:1440 nv
	scratch_load_b128 v[38:41] /*v[550:553]*/, off, off offset:1456 nv
	scratch_load_b128 v[42:45] /*v[554:557]*/, off, off offset:1600 nv
	s_wait_loadcnt 0x2f
	s_clause 0x17
	scratch_load_b128 v[46:49] /*v[558:561]*/, off, off offset:1616 nv
	scratch_load_b128 v[50:53] /*v[562:565]*/, off, off offset:1504 nv
	scratch_load_b128 v[54:57] /*v[566:569]*/, off, off offset:1520 nv
	scratch_load_b128 v[58:61] /*v[570:573]*/, off, off offset:1632 nv
	scratch_load_b128 v[62:65] /*v[574:577]*/, off, off offset:1648 nv
	scratch_load_b128 v[66:69] /*v[578:581]*/, off, off offset:1344 nv
	scratch_load_b128 v[70:73] /*v[582:585]*/, off, off offset:1360 nv
	scratch_load_b128 v[82:85] /*v[594:597]*/, off, off offset:1664 nv
	scratch_load_b128 v[86:89] /*v[598:601]*/, off, off offset:1680 nv
	s_set_vgpr_msb 0x8000
	scratch_load_b128 v[180:183], off, off offset:1376 nv
	scratch_load_b128 v[184:187], off, off offset:1392 nv
	scratch_load_b128 v[220:223], off, off offset:1696 nv
	scratch_load_b128 v[224:227], off, off offset:1712 nv
	scratch_load_b128 v[236:239], off, off offset:1472 nv
	scratch_load_b128 v[240:243], off, off offset:1488 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[4:7] /*v[260:263]*/, off, off offset:1728 nv
	scratch_load_b128 v[8:11] /*v[264:267]*/, off, off offset:1744 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[122:125] /*v[634:637]*/, off, off offset:1536 nv
	scratch_load_b128 v[126:129] /*v[638:641]*/, off, off offset:1552 nv
	scratch_load_b128 v[146:149] /*v[658:661]*/, off, off offset:1760 nv
	scratch_load_b128 v[150:153] /*v[662:665]*/, off, off offset:1776 nv
	v_lshlrev_b32_e32 v130 /*v642*/, 3, v16
	s_set_vgpr_msb 0x8000
	s_branch .LBB0_5
.LBB0_4:
	s_set_vgpr_msb 0xcf
	v_mov_b32_e32 v218 /*v986*/, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v219 /*v987*/, v218 /*v986*/ :: v_dual_mov_b32 v220 /*v988*/, v218 /*v986*/
	v_dual_mov_b32 v221 /*v989*/, v218 /*v986*/ :: v_dual_mov_b32 v222 /*v990*/, v218 /*v986*/
	v_dual_mov_b32 v223 /*v991*/, v218 /*v986*/ :: v_dual_mov_b32 v224 /*v992*/, v218 /*v986*/
	v_mov_b32_e32 v225 /*v993*/, v218 /*v986*/
	s_clause 0xb
	scratch_store_b128 off, v[218:221] /*v[986:989]*/, off nv
	scratch_store_b128 off, v[222:225] /*v[990:993]*/, off offset:16 nv
	scratch_store_b128 off, v[218:221] /*v[986:989]*/, off offset:32 nv
	scratch_store_b128 off, v[222:225] /*v[990:993]*/, off offset:48 nv
	scratch_store_b128 off, v[218:221] /*v[986:989]*/, off offset:64 nv
	scratch_store_b128 off, v[222:225] /*v[990:993]*/, off offset:80 nv
	scratch_store_b128 off, v[218:221] /*v[986:989]*/, off offset:96 nv
	scratch_store_b128 off, v[222:225] /*v[990:993]*/, off offset:112 nv
	scratch_store_b128 off, v[218:221] /*v[986:989]*/, off offset:128 nv
	scratch_store_b128 off, v[222:225] /*v[990:993]*/, off offset:144 nv
	scratch_store_b128 off, v[218:221] /*v[986:989]*/, off offset:160 nv
	scratch_store_b128 off, v[222:225] /*v[990:993]*/, off offset:176 nv
	v_mov_b64_e32 v[170:171] /*v[938:939]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[172:173] /*v[940:941]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[174:175] /*v[942:943]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[176:177] /*v[944:945]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0xcf03
	v_mov_b64_e32 v[36:37], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[38:39], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[40:41], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[42:43], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[44:45], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[46:47], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[48:49], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[50:51], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[52:53], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[54:55], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[56:57], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[58:59], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[76:77], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[78:79], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[80:81], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[82:83], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[60:61], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[62:63], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[64:65], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[66:67], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[84:85], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[86:87], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[88:89], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[90:91], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[100:101], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[102:103], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[104:105], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[106:107], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[108:109], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[110:111], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[112:113], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[114:115], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[116:117], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[118:119], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[120:121], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[122:123], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[124:125], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[126:127], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[128:129], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[130:131], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[132:133], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[134:135], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[136:137], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[138:139], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[140:141], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[142:143], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[144:145], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[146:147], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[148:149], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[150:151], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[152:153], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[154:155], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[156:157], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[158:159], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[160:161], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[162:163], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[164:165], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[166:167], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[168:169], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[170:171], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[172:173], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[174:175], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[176:177], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[178:179], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[188:189], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[190:191], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[192:193], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[194:195], v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x3c3
	v_mov_b64_e32 v[178:179] /*v[946:947]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[180:181] /*v[948:949]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[182:183] /*v[950:951]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[184:185] /*v[952:953]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0xc383
	v_mov_b64_e32 v[154:155] /*v[666:667]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[156:157] /*v[668:669]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[158:159] /*v[670:671]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[160:161] /*v[672:673]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x8303
	v_mov_b64_e32 v[28:29], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[30:31], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[32:33], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[34:35], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[212:213], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[214:215], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[216:217], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[218:219], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[228:229], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[230:231], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[232:233], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[234:235], v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x3c3
	v_mov_b64_e32 v[240:241] /*v[1008:1009]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[238:239] /*v[1006:1007]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[236:237] /*v[1004:1005]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[234:235] /*v[1002:1003]*/, v[218:219] /*v[986:987]*/
	s_set_vgpr_msb 0xc303
	v_mov_b64_e32 v[244:245], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[246:247], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[248:249], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[250:251], v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[20:21], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[22:23], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[24:25], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[26:27], v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x3c3
	v_mov_b64_e32 v[192:193] /*v[960:961]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[194:195] /*v[962:963]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[196:197] /*v[964:965]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[198:199] /*v[966:967]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0xc343
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x43c3
	v_mov_b64_e32 v[146:147] /*v[914:915]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[148:149] /*v[916:917]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[150:151] /*v[918:919]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[152:153] /*v[920:921]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0xc303
	v_mov_b64_e32 v[196:197], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[198:199], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[200:201], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[202:203], v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x343
	v_mov_b64_e32 v[42:43] /*v[298:299]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[44:45] /*v[300:301]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[46:47] /*v[302:303]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[48:49] /*v[304:305]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[50:51] /*v[306:307]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[52:53] /*v[308:309]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[54:55] /*v[310:311]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[56:57] /*v[312:313]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[22:23] /*v[278:279]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[24:25] /*v[280:281]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[26:27] /*v[282:283]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x4303
	v_mov_b64_e32 v[252:253], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[254:255], v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0x343
	v_mov_b64_e32 v[0:1] /*v[256:257]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[2:3] /*v[258:259]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[40:41] /*v[296:297]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[58:59] /*v[314:315]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[60:61] /*v[316:317]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[62:63] /*v[318:319]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[64:65] /*v[320:321]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x43c3
	v_mov_b64_e32 v[210:211] /*v[978:979]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[212:213] /*v[980:981]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[214:215] /*v[982:983]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[216:217] /*v[984:985]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0xc343
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[186:187] /*v[442:443]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[188:189] /*v[444:445]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[190:191] /*v[446:447]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[192:193] /*v[448:449]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x43c3
	v_mov_b64_e32 v[138:139] /*v[906:907]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[140:141] /*v[908:909]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[142:143] /*v[910:911]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[144:145] /*v[912:913]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0xc343
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[156:157] /*v[412:413]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[158:159] /*v[414:415]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[160:161] /*v[416:417]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[152:153] /*v[408:409]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[224:225] /*v[992:993]*/
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[134:135] /*v[390:391]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[136:137] /*v[392:393]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x4383
	v_mov_b64_e32 v[138:139] /*v[650:651]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[140:141] /*v[652:653]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[142:143] /*v[654:655]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[144:145] /*v[656:657]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x8303
	v_mov_b64_e32 v[204:205], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[206:207], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[208:209], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[210:211], v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x343
	v_mov_b64_e32 v[162:163] /*v[418:419]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[164:165] /*v[420:421]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[166:167] /*v[422:423]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[168:169] /*v[424:425]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x43c3
	v_mov_b64_e32 v[202:203] /*v[970:971]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[204:205] /*v[972:973]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[206:207] /*v[974:975]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[208:209] /*v[976:977]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0xc34f
	v_mov_b64_e32 v[178:179] /*v[434:435]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[180:181] /*v[436:437]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[182:183] /*v[438:439]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[184:185] /*v[440:441]*/, v[224:225] /*v[992:993]*/
	s_clause 0x1
	scratch_store_b128 off, v[218:221] /*v[986:989]*/, off offset:192 nv
	scratch_store_b128 off, v[222:225] /*v[990:993]*/, off offset:208 nv
	s_set_vgpr_msb 0x4f00
.LBB0_5:
	s_set_vgpr_msb 0x80
	s_clause 0x23
	scratch_load_b128 v[210:213] /*v[722:725]*/, off, off offset:352 nv
	scratch_load_b128 v[214:217] /*v[726:729]*/, off, off offset:368 nv
	scratch_load_b128 v[218:221] /*v[730:733]*/, off, off offset:224 nv
	scratch_load_b128 v[222:225] /*v[734:737]*/, off, off offset:240 nv
	scratch_load_b128 v[188:191] /*v[700:703]*/, off, off offset:384 nv
	scratch_load_b128 v[192:195] /*v[704:707]*/, off, off offset:400 nv
	scratch_load_b128 v[180:183] /*v[692:695]*/, off, off offset:416 nv
	scratch_load_b128 v[184:187] /*v[696:699]*/, off, off offset:432 nv
	scratch_load_b128 v[234:237] /*v[746:749]*/, off, off offset:256 nv
	scratch_load_b128 v[238:241] /*v[750:753]*/, off, off offset:272 nv
	scratch_load_b128 v[202:205] /*v[714:717]*/, off, off offset:448 nv
	scratch_load_b128 v[206:209] /*v[718:721]*/, off, off offset:464 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[22:25] /*v[790:793]*/, off, off offset:288 nv
	scratch_load_b128 v[26:29] /*v[794:797]*/, off, off offset:304 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[226:229] /*v[738:741]*/, off, off offset:480 nv
	scratch_load_b128 v[230:233] /*v[742:745]*/, off, off offset:496 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[50:53] /*v[818:821]*/, off, off offset:512 nv
	scratch_load_b128 v[54:57] /*v[822:825]*/, off, off offset:528 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[242:245] /*v[754:757]*/, off, off offset:320 nv
	scratch_load_b128 v[246:249] /*v[758:761]*/, off, off offset:336 nv
	scratch_load_b128 v[90:93] /*v[602:605]*/, off, off offset:544 nv
	scratch_load_b128 v[94:97] /*v[606:609]*/, off, off offset:560 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[42:45] /*v[810:813]*/, off, off offset:576 nv
	scratch_load_b128 v[46:49] /*v[814:817]*/, off, off offset:592 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[114:117] /*v[626:629]*/, off, off offset:608 nv
	scratch_load_b128 v[118:121] /*v[630:633]*/, off, off offset:624 nv
	scratch_load_b128 v[98:101] /*v[610:613]*/, off, off offset:640 nv
	scratch_load_b128 v[102:105] /*v[614:617]*/, off, off offset:656 nv
	scratch_load_b128 v[106:109] /*v[618:621]*/, off, off offset:672 nv
	scratch_load_b128 v[110:113] /*v[622:625]*/, off, off offset:688 nv
	s_sub_co_i32 s20, s43, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s20, 1
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_9
	s_set_vgpr_msb 0x48
	s_wait_loadcnt 0x3e
	v_dual_lshlrev_b32 v201 /*v457*/, 4, v201 /*v713*/ :: v_dual_bitop2_b32 v195 /*v451*/, 64, v136 /*v648*/ bitop3:0x54
	s_lshl_b32 s3, s23, 4
	s_lshl_b32 s6, s2, 9
	s_set_vgpr_msb 0x48c4
	v_dual_add_nc_u32 v40 /*v808*/, s11, v28 /*v284*/ :: v_dual_add_nc_u32 v38 /*v806*/, s11, v29 /*v285*/
	s_set_vgpr_msb 0xc450
	v_add3_u32 v201 /*v457*/, s3, s6, v201 /*v457*/
	s_set_vgpr_msb 0x5003
	s_wait_loadcnt 0x23
	v_mov_b32_e32 v0, v76 /*v844*/
	s_set_vgpr_msb 0x34e
	v_dual_mov_b32 v202 /*v458*/, v201 /*v713*/ :: v_dual_bitop2_b32 v194 /*v450*/, 32, v200 /*v968*/ bitop3:0x54
	s_set_vgpr_msb 0x4e0b
	s_wait_loadcnt 0x1f
	v_dual_add_nc_u32 v4, s42, v201 /*v713*/ :: v_dual_mov_b32 v2, v78 /*v846*/
	s_set_vgpr_msb 0xb44
	v_mul_lo_u32 v201 /*v457*/, s8, v201 /*v457*/
	s_set_vgpr_msb 0x4400
	v_mov_b32_e32 v1, v0
	s_set_vgpr_msb 0xc3
	v_dual_mov_b32 v189 /*v957*/, v188 /*v956*/ :: v_dual_mov_b32 v187 /*v955*/, v186 /*v954*/
	v_dual_mov_b32 v191 /*v959*/, v190 /*v958*/ :: v_dual_add_nc_u32 v252 /*v1020*/, s11, v17
	scratch_store_b64 off, v[0:1], off offset:3492 nv
	s_set_vgpr_msb 0xc300
	v_mov_b32_e32 v0, v4
	s_set_vgpr_msb 0x44
	v_or_b32_e32 v201 /*v457*/, v16, v201 /*v457*/
	s_set_vgpr_msb 0x4408
	s_wait_loadcnt 0x1e
	v_dual_mov_b32 v19, v18 :: v_dual_mov_b32 v3, v2
	s_clause 0x8
	scratch_store_b128 off, v[74:77] /*v[586:589]*/, off offset:3556 nv
	scratch_store_b128 off, v[78:81] /*v[590:593]*/, off offset:3572 nv
	s_set_vgpr_msb 0x80c
	scratch_store_b128 off, v[90:93] /*v[858:861]*/, off offset:3516 nv
	scratch_store_b128 off, v[94:97] /*v[862:865]*/, off offset:3532 nv
	scratch_store_b128 off, v[130:133] /*v[898:901]*/, off offset:3392 nv
	scratch_store_b128 off, v[134:137] /*v[902:905]*/, off offset:3408 nv
	s_set_vgpr_msb 0xc81
	scratch_store_b64 off, v[0:1], off offset:3548 nv
	v_add_lshl_u32 v201 /*v713*/, v201 /*v457*/, s22, 4
	s_set_vgpr_msb 0x8144
	v_add_lshl_u32 v201 /*v457*/, s23, v202 /*v458*/, 4
	s_set_vgpr_msb 0x440c
	v_mov_b32_e32 v1, v4
	scratch_store_b64 off, v[186:187] /*v[954:955]*/, off offset:3468 nv
	s_set_vgpr_msb 0xccc
	v_dual_add_nc_u32 v76 /*v844*/, s11, v80 /*v848*/ :: v_dual_add_nc_u32 v10 /*v778*/, s11, v81 /*v849*/
	s_set_vgpr_msb 0xcc44
	v_add3_u32 v201 /*v457*/, s6, v201 /*v457*/, 0x100
	s_set_vgpr_msb 0x4403
	scratch_store_b64 off, v[0:1], off offset:3424 nv
	s_wait_xcnt 0x0
	v_mov_b32_e32 v1, v38 /*v806*/
	s_set_vgpr_msb 0x34c
	scratch_store_b64 off, v[188:189] /*v[956:957]*/, off offset:3476 nv
	v_or_b32_e32 v196 /*v452*/, 0x60, v200 /*v968*/
	s_set_vgpr_msb 0x4c44
	v_mul_lo_u32 v201 /*v457*/, s8, v201 /*v457*/
	s_set_vgpr_msb 0x4448
	v_or_b32_e32 v197 /*v453*/, 0x80, v136 /*v648*/
	s_set_vgpr_msb 0x4800
	scratch_store_b64 off, v[0:1], off offset:3432 nv
	s_set_vgpr_msb 0x4c
	v_or_b32_e32 v198 /*v454*/, 0xa0, v200 /*v968*/
	s_set_vgpr_msb 0x4cc4
	v_dual_add_nc_u32 v78 /*v846*/, s11, v30 /*v286*/ :: v_dual_add_nc_u32 v254 /*v1022*/, s11, v31 /*v287*/
	s_set_vgpr_msb 0xc448
	v_or_b32_e32 v199 /*v455*/, 0xc0, v136 /*v648*/
	s_set_vgpr_msb 0x484c
	v_or_b32_e32 v200 /*v456*/, 0xe0, v200 /*v968*/
	s_set_vgpr_msb 0x4c04
	v_or_b32_e32 v0, v16, v201 /*v457*/
	s_mov_b32 s5, s4
	s_set_vgpr_msb 0x482
	v_dual_mov_b32 v179 /*v691*/, v178 /*v690*/ :: v_dual_mov_b32 v177 /*v689*/, v176 /*v688*/
	v_mov_b64_e32 v[196:197] /*v[708:709]*/, s[4:5]
	s_set_vgpr_msb 0x82c0
	v_add_lshl_u32 v12 /*v780*/, v0, s22, 4
	s_set_vgpr_msb 0xc00a
	v_add_nc_u32_e32 v0, v135 /*v647*/, v136 /*v648*/
	s_set_vgpr_msb 0xacf
	scratch_store_b64 off, v[190:191] /*v[958:959]*/, off offset:3484 nv
	v_dual_mov_b32 v75 /*v843*/, v74 /*v842*/ :: v_dual_mov_b32 v39 /*v807*/, v252 /*v1020*/
	s_set_vgpr_msb 0xcf82
	v_dual_mov_b32 v163 /*v675*/, v162 /*v674*/ :: v_dual_mov_b32 v165 /*v677*/, v164 /*v676*/
	scratch_store_b32 off, v0, off offset:3440 nv
	s_set_vgpr_msb 0x8206
	v_add_nc_u32_e32 v0, v135 /*v647*/, v194 /*v450*/
	s_set_vgpr_msb 0x6c3
	scratch_store_b64 off, v[2:3], off offset:3500 nv
	v_dual_mov_b32 v41 /*v809*/, v40 /*v808*/ :: v_dual_mov_b32 v79 /*v847*/, v78 /*v846*/
	s_set_vgpr_msb 0xc382
	v_dual_mov_b32 v167 /*v679*/, v166 /*v678*/ :: v_dual_mov_b32 v169 /*v681*/, v168 /*v680*/
	scratch_store_b32 off, v0, off offset:3444 nv
	s_set_vgpr_msb 0x8206
	v_add_nc_u32_e32 v0, v135 /*v647*/, v195 /*v451*/
	s_set_vgpr_msb 0x6c0
	s_clause 0x6
	scratch_store_b64 off, v[18:19], off offset:3508 nv
	scratch_load_b128 v[242:245] /*v[1010:1013]*/, off, off offset:1792 nv
	scratch_load_b128 v[246:249] /*v[1014:1017]*/, off, off offset:1808 nv
	scratch_load_b128 v[122:125] /*v[890:893]*/, off, off offset:1856 nv
	scratch_load_b128 v[126:129] /*v[894:897]*/, off, off offset:1872 nv
	scratch_load_b128 v[154:157] /*v[922:925]*/, off, off offset:928 nv
	scratch_load_b128 v[158:161] /*v[926:929]*/, off, off offset:944 nv
	s_set_vgpr_msb 0xc080
	s_wait_loadcnt 0x1d
	scratch_load_b128 v[180:183] /*v[692:695]*/, off, off offset:3392 nv
	s_wait_loadcnt 0x1d
	s_clause 0x7
	scratch_load_b128 v[184:187] /*v[696:699]*/, off, off offset:3408 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[130:133] /*v[898:901]*/, off, off offset:1088 nv
	scratch_load_b128 v[134:137] /*v[902:905]*/, off, off offset:1104 nv
	scratch_load_b128 v[2:5] /*v[770:773]*/, off, off offset:1152 nv
	scratch_load_b128 v[6:9] /*v[774:777]*/, off, off offset:1168 nv
	scratch_load_b128 v[82:85] /*v[850:853]*/, off, off offset:1888 nv
	scratch_load_b128 v[86:89] /*v[854:857]*/, off, off offset:1904 nv
	s_wait_loadcnt 0x1b
	scratch_load_b128 v[50:53] /*v[818:821]*/, off, off offset:384 nv
	s_wait_loadcnt 0x1b
	s_clause 0x9
	scratch_load_b128 v[54:57] /*v[822:825]*/, off, off offset:400 nv
	scratch_load_b128 v[30:33] /*v[798:801]*/, off, off offset:1952 nv
	scratch_load_b128 v[34:37] /*v[802:805]*/, off, off offset:1968 nv
	scratch_load_b128 v[98:101] /*v[866:869]*/, off, off offset:800 nv
	scratch_load_b128 v[102:105] /*v[870:873]*/, off, off offset:816 nv
	scratch_load_b128 v[114:117] /*v[882:885]*/, off, off offset:832 nv
	scratch_load_b128 v[118:121] /*v[886:889]*/, off, off offset:848 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[8:11], off, off offset:1312 nv
	scratch_load_b128 v[12:15], off, off offset:1328 nv
	s_set_vgpr_msb 0xc0
	s_wait_loadcnt 0x1f
	scratch_load_b128 v[42:45] /*v[810:813]*/, off, off offset:416 nv
	s_wait_loadcnt 0x1f
	s_clause 0x19
	scratch_load_b128 v[46:49] /*v[814:817]*/, off, off offset:432 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[188:191] /*v[700:703]*/, off, off offset:512 nv
	scratch_load_b128 v[192:195] /*v[704:707]*/, off, off offset:528 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[58:61] /*v[826:829]*/, off, off offset:704 nv
	scratch_load_b128 v[62:65] /*v[830:833]*/, off, off offset:720 nv
	scratch_load_b128 v[90:93] /*v[858:861]*/, off, off offset:768 nv
	scratch_load_b128 v[94:97] /*v[862:865]*/, off, off offset:784 nv
	scratch_load_b128 v[162:165] /*v[930:933]*/, off, off offset:960 nv
	scratch_load_b128 v[166:169] /*v[934:937]*/, off, off offset:976 nv
	scratch_load_b128 v[22:25] /*v[790:793]*/, off, off offset:1120 nv
	scratch_load_b128 v[26:29] /*v[794:797]*/, off, off offset:1136 nv
	scratch_load_b128 v[14:17] /*v[782:785]*/, off, off offset:1184 nv
	scratch_load_b128 v[18:21] /*v[786:789]*/, off, off offset:1200 nv
	scratch_load_b128 v[226:229] /*v[994:997]*/, off, off offset:576 nv
	scratch_load_b128 v[230:233] /*v[998:1001]*/, off, off offset:592 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[10:13] /*v[266:269]*/, off, off offset:864 nv
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:880 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[220:223], off, off offset:1632 nv
	scratch_load_b128 v[224:227], off, off offset:1648 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b64 v[80:81] /*v[848:849]*/, off, off offset:3500 nv
	scratch_load_b64 v[250:251] /*v[1018:1019]*/, off, off offset:3508 nv
	s_set_vgpr_msb 0xc082
	v_dual_mov_b32 v171 /*v683*/, v170 /*v682*/ :: v_dual_mov_b32 v173 /*v685*/, v172 /*v684*/
	scratch_store_b32 off, v0, off offset:3448 nv
	s_set_vgpr_msb 0x8206
	v_add_nc_u32_e32 v0, v135 /*v647*/, v196 /*v452*/
	s_set_vgpr_msb 0x6cb
	scratch_store_b32 off, v198 /*v710*/, off offset:3588 nv
	v_dual_mov_b32 v77 /*v845*/, v254 /*v1022*/ :: v_dual_mov_b32 v253 /*v1021*/, v76 /*v844*/
	s_set_vgpr_msb 0xcb82
	v_mov_b32_e32 v175 /*v687*/, v174 /*v686*/
	scratch_store_b32 off, v0, off offset:3452 nv
	s_set_vgpr_msb 0x8206
	v_add_nc_u32_e32 v0, v135 /*v647*/, v197 /*v453*/
	scratch_store_b32 off, v202 /*v458*/, off offset:3592 nv
	s_set_vgpr_msb 0x6c3
	v_mov_b32_e32 v11 /*v779*/, v10 /*v778*/
	s_set_vgpr_msb 0xc3a0
	v_lshl_or_b32 v198 /*v710*/, s2, 5, v130 /*v642*/
	s_set_vgpr_msb 0xa0c6
	v_add_nc_u32_e32 v13 /*v781*/, v135 /*v647*/, v200 /*v456*/
	s_set_vgpr_msb 0xc600
	scratch_store_b32 off, v0, off offset:3456 nv
	s_set_vgpr_msb 6
	v_add_nc_u32_e32 v0, v135 /*v647*/, v198 /*v454*/
	s_set_vgpr_msb 0x608
	scratch_store_b32 off, v130 /*v642*/, off offset:3596 nv
	s_ashr_i32 s21, s20, 31
	s_lshl_b32 s3, s8, 13
	s_mov_b32 s4, 0x3fb8aa3b
	s_set_vgpr_msb 0x800
	scratch_store_b32 off, v0, off offset:3460 nv
	s_set_vgpr_msb 6
	v_add_nc_u32_e32 v0, v135 /*v647*/, v199 /*v455*/
	s_set_vgpr_msb 0x600
	scratch_store_b32 off, v0, off offset:3464 nv
.LBB0_7:
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v206 /*v462*/, 32, v201 /*v713*/
	s_set_vgpr_msb 0x4800
	s_clause 0x3e
	scratch_store_b128 off, v[156:159], off offset:3360 nv
	scratch_store_b128 off, v[160:163], off offset:3376 nv
	scratch_store_b128 off, v[212:215], off offset:3328 nv
	scratch_store_b128 off, v[216:219], off offset:3344 nv
	scratch_store_b128 off, v[28:31], off offset:3296 nv
	scratch_store_b128 off, v[32:35], off offset:3312 nv
	s_set_vgpr_msb 12
	scratch_store_b128 off, v[178:181] /*v[946:949]*/, off offset:3264 nv
	scratch_store_b128 off, v[182:185] /*v[950:953]*/, off offset:3280 nv
	s_set_vgpr_msb 0xc08
	scratch_store_b128 off, v[154:157] /*v[666:669]*/, off offset:3232 nv
	scratch_store_b128 off, v[158:161] /*v[670:673]*/, off offset:3248 nv
	s_set_vgpr_msb 0x800
	scratch_store_b128 off, v[188:191], off offset:3200 nv
	scratch_store_b128 off, v[192:195], off offset:3216 nv
	scratch_store_b128 off, v[84:87], off offset:3168 nv
	scratch_store_b128 off, v[88:91], off offset:3184 nv
	scratch_store_b128 off, v[148:151], off offset:3136 nv
	scratch_store_b128 off, v[152:155], off offset:3152 nv
	scratch_store_b128 off, v[60:63], off offset:3104 nv
	scratch_store_b128 off, v[64:67], off offset:3120 nv
	scratch_store_b128 off, v[140:143], off offset:3072 nv
	scratch_store_b128 off, v[144:147], off offset:3088 nv
	scratch_store_b128 off, v[76:79], off offset:3040 nv
	scratch_store_b128 off, v[80:83], off offset:3056 nv
	scratch_store_b128 off, v[132:135], off offset:3008 nv
	scratch_store_b128 off, v[136:139], off offset:3024 nv
	scratch_store_b128 off, v[52:55], off offset:2976 nv
	scratch_store_b128 off, v[56:59], off offset:2992 nv
	scratch_store_b128 off, v[124:127], off offset:2944 nv
	scratch_store_b128 off, v[128:131], off offset:2960 nv
	scratch_store_b128 off, v[44:47], off offset:2912 nv
	scratch_store_b128 off, v[48:51], off offset:2928 nv
	scratch_store_b128 off, v[116:119], off offset:2880 nv
	scratch_store_b128 off, v[120:123], off offset:2896 nv
	scratch_store_b128 off, v[36:39], off offset:2848 nv
	scratch_store_b128 off, v[40:43], off offset:2864 nv
	scratch_store_b128 off, v[228:231], off offset:2816 nv
	scratch_store_b128 off, v[232:235], off offset:2832 nv
	scratch_store_b128 off, v[20:23], off offset:2784 nv
	scratch_store_b128 off, v[24:27], off offset:2800 nv
	scratch_store_b128 off, v[244:247], off offset:2752 nv
	scratch_store_b128 off, v[248:251], off offset:2768 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[66:69] /*v[322:325]*/, off offset:2720 nv
	scratch_store_b128 off, v[70:73] /*v[326:329]*/, off offset:2736 nv
	s_set_vgpr_msb 0x40c
	scratch_store_b128 off, v[192:195] /*v[960:963]*/, off offset:2688 nv
	scratch_store_b128 off, v[196:199] /*v[964:967]*/, off offset:2704 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[20:23] /*v[276:279]*/, off offset:2656 nv
	scratch_store_b128 off, v[24:27] /*v[280:283]*/, off offset:2672 nv
	s_set_vgpr_msb 0x40c
	scratch_store_b128 off, v[146:149] /*v[914:917]*/, off offset:2624 nv
	scratch_store_b128 off, v[150:153] /*v[918:921]*/, off offset:2640 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[34:37] /*v[290:293]*/, off offset:2592 nv
	scratch_store_b128 off, v[38:41] /*v[294:297]*/, off offset:2608 nv
	scratch_store_b128 off, v[186:189] /*v[442:445]*/, off offset:2560 nv
	scratch_store_b128 off, v[190:193] /*v[446:449]*/, off offset:2576 nv
	scratch_store_b128 off, v[42:45] /*v[298:301]*/, off offset:2528 nv
	scratch_store_b128 off, v[46:49] /*v[302:305]*/, off offset:2544 nv
	s_set_vgpr_msb 0x400
	scratch_store_b128 off, v[252:255], off offset:2496 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[0:3] /*v[256:259]*/, off offset:2512 nv
	s_set_vgpr_msb 0x40c
	scratch_store_b128 off, v[138:141] /*v[906:909]*/, off offset:2464 nv
	scratch_store_b128 off, v[142:145] /*v[910:913]*/, off offset:2480 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[98:101] /*v[354:357]*/, off offset:2432 nv
	scratch_store_b128 off, v[102:105] /*v[358:361]*/, off offset:2448 nv
	s_set_vgpr_msb 0x40c
	scratch_store_b128 off, v[210:213] /*v[978:981]*/, off offset:2400 nv
	scratch_store_b128 off, v[214:217] /*v[982:985]*/, off offset:2416 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[82:85] /*v[338:341]*/, off offset:2368 nv
	s_clause 0xd
	scratch_store_b128 off, v[86:89] /*v[342:345]*/, off offset:2384 nv
	scratch_store_b128 off, v[130:133] /*v[386:389]*/, off offset:2336 nv
	scratch_store_b128 off, v[134:137] /*v[390:393]*/, off offset:2352 nv
	scratch_store_b128 off, v[178:181] /*v[434:437]*/, off offset:2304 nv
	scratch_store_b128 off, v[182:185] /*v[438:441]*/, off offset:2320 nv
	scratch_store_b128 off, v[114:117] /*v[370:373]*/, off offset:2272 nv
	scratch_store_b128 off, v[118:121] /*v[374:377]*/, off offset:2288 nv
	scratch_store_b128 off, v[146:149] /*v[402:405]*/, off offset:2240 nv
	scratch_store_b128 off, v[150:153] /*v[406:409]*/, off offset:2256 nv
	scratch_store_b128 off, v[162:165] /*v[418:421]*/, off offset:2208 nv
	scratch_store_b128 off, v[166:169] /*v[422:425]*/, off offset:2224 nv
	s_set_vgpr_msb 0x442
	scratch_store_b128 off, v[204:207], off offset:2176 nv
	scratch_store_b128 off, v[208:211], off offset:2192 nv
	buffer_load_b128 v[194:197] /*v[450:453]*/, v201 /*v713*/, s[12:15], null offen
	s_set_vgpr_msb 0x42c0
	s_clause 0x1
	scratch_load_b128 v[66:69] /*v[834:837]*/, off, off offset:2080 nv
	scratch_load_b128 v[70:73] /*v[838:841]*/, off, off offset:2096 nv
	s_set_vgpr_msb 0xc049
	v_or_b32_e32 v218 /*v474*/, 64, v201 /*v713*/
	buffer_load_b128 v[198:201] /*v[454:457]*/, v206 /*v462*/, s[12:15], null offen
	s_set_vgpr_msb 0x4942
	s_clause 0x2
	buffer_load_b128 v[202:205] /*v[458:461]*/, v201 /*v713*/, s[16:19], null offen
	s_set_vgpr_msb 0x4249
	buffer_load_b128 v[206:209] /*v[462:465]*/, v206 /*v462*/, s[16:19], null offen
	v_or_b32_e32 v222 /*v478*/, 0x60, v201 /*v713*/
	v_or_b32_e32 v234 /*v490*/, 0x80, v201 /*v713*/
	v_or_b32_e32 v238 /*v494*/, 0xa0, v201 /*v713*/
	buffer_load_b128 v[210:213] /*v[466:469]*/, v218 /*v474*/, s[12:15], null offen
	s_set_vgpr_msb 0x4900
	s_clause 0x1
	scratch_load_b128 v[120:123], off, off offset:352 nv
	scratch_load_b128 v[124:127], off, off offset:368 nv
	s_set_vgpr_msb 0x8c
	v_or_b32_e32 v10 /*v522*/, 32, v12 /*v780*/
	s_set_vgpr_msb 0x8c49
	s_clause 0x1
	buffer_load_b128 v[226:229] /*v[482:485]*/, v234 /*v490*/, s[12:15], null offen
	buffer_load_b128 v[214:217] /*v[470:473]*/, v222 /*v478*/, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[218:221] /*v[474:477]*/, v218 /*v474*/, s[16:19], null offen
	buffer_load_b128 v[222:225] /*v[478:481]*/, v222 /*v478*/, s[16:19], null offen
	buffer_load_b128 v[230:233] /*v[486:489]*/, v238 /*v494*/, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[234:237] /*v[490:493]*/, v234 /*v490*/, s[16:19], null offen
	buffer_load_b128 v[238:241] /*v[494:497]*/, v238 /*v494*/, s[16:19], null offen
	v_or_b32_e32 v250 /*v506*/, 0xc0, v201 /*v713*/
	v_or_b32_e32 v254 /*v510*/, 0xe0, v201 /*v713*/
	s_set_vgpr_msb 0x4982
	buffer_load_b128 v[6:9] /*v[518:521]*/, v10 /*v522*/, s[12:15], null offen
	s_set_vgpr_msb 0x8283
	s_clause 0x2
	buffer_load_b128 v[202:205] /*v[714:717]*/, v12 /*v780*/, s[16:19], null offen
	s_set_vgpr_msb 0x838e
	buffer_load_b128 v[206:209] /*v[718:721]*/, v10 /*v522*/, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v10 /*v522*/, 64, v12 /*v780*/
	s_set_vgpr_msb 0x8e40
	v_mov_b64_e32 v[152:153] /*v[408:409]*/, v[170:171]
	s_set_vgpr_msb 0x4041
	buffer_load_b128 v[242:245] /*v[498:501]*/, v250 /*v506*/, s[12:15], null offen
	s_set_vgpr_msb 0x4140
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[168:169]
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[166:167]
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[164:165]
	s_set_vgpr_msb 0x4080
	s_wait_loadcnt 0x3e
	v_mov_b64_e32 v[120:121] /*v[632:633]*/, v[178:179]
	v_mov_b64_e32 v[118:119] /*v[630:631]*/, v[176:177]
	v_mov_b64_e32 v[116:117] /*v[628:629]*/, v[174:175]
	v_mov_b64_e32 v[114:115] /*v[626:627]*/, v[172:173]
	s_set_vgpr_msb 0x8003
	v_mov_b64_e32 v[168:169], v[202:203] /*v[970:971]*/
	v_mov_b64_e32 v[170:171], v[204:205] /*v[972:973]*/
	v_mov_b64_e32 v[172:173], v[206:207] /*v[974:975]*/
	v_mov_b64_e32 v[174:175], v[208:209] /*v[976:977]*/
	s_set_vgpr_msb 0x3c1
	v_mov_b64_e32 v[208:209] /*v[976:977]*/, v[176:177] /*v[432:433]*/
	v_mov_b64_e32 v[206:207] /*v[974:975]*/, v[174:175] /*v[430:431]*/
	v_mov_b64_e32 v[204:205] /*v[972:973]*/, v[172:173] /*v[428:429]*/
	v_mov_b64_e32 v[202:203] /*v[970:971]*/, v[170:171] /*v[426:427]*/
	s_set_vgpr_msb 0xc141
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[144:145] /*v[400:401]*/
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[142:143] /*v[398:399]*/
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[140:141] /*v[396:397]*/
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[138:139] /*v[394:395]*/
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[112:113] /*v[368:369]*/
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[110:111] /*v[366:367]*/
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[108:109] /*v[364:365]*/
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[106:107] /*v[362:363]*/
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[80:81] /*v[336:337]*/
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[78:79] /*v[334:335]*/
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[76:77] /*v[332:333]*/
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[74:75] /*v[330:331]*/
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[56:57] /*v[312:313]*/
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[54:55] /*v[310:311]*/
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[52:53] /*v[308:309]*/
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[50:51] /*v[306:307]*/
	s_set_vgpr_msb 0x4140
	v_mov_b64_e32 v[56:57] /*v[312:313]*/, v[202:203]
	v_mov_b64_e32 v[54:55] /*v[310:311]*/, v[200:201]
	v_mov_b64_e32 v[52:53] /*v[308:309]*/, v[198:199]
	v_mov_b64_e32 v[50:51] /*v[306:307]*/, v[196:197]
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:224 nv
	scratch_load_b128 v[196:199], off, off offset:240 nv
	s_set_vgpr_msb 0x82
	s_clause 0x2
	buffer_load_b128 v[210:213] /*v[722:725]*/, v10 /*v522*/, s[12:15], null offen
	s_set_vgpr_msb 0x8241
	buffer_load_b128 v[246:249] /*v[502:505]*/, v254 /*v510*/, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[250:253] /*v[506:509]*/, v250 /*v506*/, s[16:19], null offen
	buffer_load_b128 v[254:257] /*v[510:513]*/, v254 /*v510*/, s[16:19], null offen
	s_set_vgpr_msb 0x418f
	buffer_load_b128 v[2:5] /*v[514:517]*/, v12 /*v780*/, s[12:15], null offen
	v_or_b32_e32 v11 /*v523*/, 0x60, v12 /*v780*/
	s_set_vgpr_msb 0x8f8e
	buffer_load_b128 v[214:217] /*v[726:729]*/, v11 /*v523*/, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[218:221] /*v[730:733]*/, v10 /*v522*/, s[16:19], null offen
	buffer_load_b128 v[222:225] /*v[734:737]*/, v11 /*v523*/, s[16:19], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v10 /*v522*/, 0x80, v12 /*v780*/
	s_wait_xcnt 0x0
	v_or_b32_e32 v11 /*v523*/, 0xa0, v12 /*v780*/
	buffer_load_b128 v[230:233] /*v[742:745]*/, v11 /*v523*/, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[234:237] /*v[746:749]*/, v10 /*v522*/, s[16:19], null offen
	buffer_load_b128 v[238:241] /*v[750:753]*/, v11 /*v523*/, s[16:19], null offen
	buffer_load_b128 v[226:229] /*v[738:741]*/, v10 /*v522*/, s[12:15], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v10 /*v522*/, 0xc0, v12 /*v780*/
	v_or_b32_e32 v11 /*v523*/, 0xe0, v12 /*v780*/
	buffer_load_b128 v[126:129] /*v[638:641]*/, v11 /*v523*/, s[12:15], null offen
	s_set_vgpr_msb 0x8e02
	s_clause 0x1
	buffer_load_b128 v[184:187], v10 /*v522*/, s[16:19], null offen
	buffer_load_b128 v[188:191], v11 /*v523*/, s[16:19], null offen
	s_set_vgpr_msb 0x282
	buffer_load_b128 v[122:125] /*v[634:637]*/, v10 /*v522*/, s[12:15], null offen
	s_set_vgpr_msb 0x82c0
	s_clause 0x5
	scratch_load_b128 v[106:109] /*v[874:877]*/, off, off offset:1984 nv
	scratch_load_b128 v[110:113] /*v[878:881]*/, off, off offset:2000 nv
	scratch_load_b128 v[138:141] /*v[906:909]*/, off, off offset:2112 nv
	scratch_load_b128 v[142:145] /*v[910:913]*/, off, off offset:2128 nv
	scratch_load_b128 v[146:149] /*v[914:917]*/, off, off offset:1920 nv
	scratch_load_b128 v[150:153] /*v[918:921]*/, off, off offset:1936 nv
	s_set_vgpr_msb 0xc040
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[114:115]
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[112:113]
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[110:111]
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[108:109]
	s_set_vgpr_msb 0x4000
	s_clause 0x2b
	scratch_load_b128 v[112:115], off, off offset:448 nv
	scratch_load_b128 v[116:119], off, off offset:464 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[186:189] /*v[442:445]*/, off, off offset:2144 nv
	scratch_load_b128 v[190:193] /*v[446:449]*/, off, off offset:2160 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[0:3], off, off offset:3556 nv
	scratch_load_b128 v[4:7], off, off offset:3572 nv
	scratch_load_b128 v[16:19], off, off offset:1344 nv
	scratch_load_b128 v[20:23], off, off offset:1360 nv
	scratch_load_b128 v[40:43], off, off offset:1824 nv
	scratch_load_b128 v[44:47], off, off offset:1840 nv
	scratch_load_b128 v[24:27], off, off offset:3516 nv
	scratch_load_b128 v[28:31], off, off offset:3532 nv
	scratch_load_b128 v[56:59], off, off offset:1248 nv
	scratch_load_b128 v[60:63], off, off offset:1264 nv
	scratch_load_b128 v[80:83], off, off offset:1664 nv
	scratch_load_b128 v[84:87], off, off offset:1680 nv
	scratch_load_b128 v[136:139], off, off offset:544 nv
	scratch_load_b128 v[140:143], off, off offset:560 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b128 v[194:197] /*v[962:965]*/, off, off offset:1024 nv
	scratch_load_b128 v[198:201] /*v[966:969]*/, off, off offset:1040 nv
	scratch_load_b128 v[186:189] /*v[954:957]*/, off, off offset:1056 nv
	scratch_load_b128 v[190:193] /*v[958:961]*/, off, off offset:1072 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[144:147], off, off offset:1376 nv
	scratch_load_b128 v[148:151], off, off offset:1392 nv
	s_set_vgpr_msb 0x80
	scratch_load_b128 v[154:157] /*v[666:669]*/, off, off offset:896 nv
	scratch_load_b128 v[158:161] /*v[670:673]*/, off, off offset:912 nv
	s_set_vgpr_msb 0x8040
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:992 nv
	scratch_load_b128 v[22:25] /*v[278:281]*/, off, off offset:1008 nv
	scratch_load_b128 v[2:5] /*v[258:261]*/, off, off offset:1216 nv
	scratch_load_b128 v[6:9] /*v[262:265]*/, off, off offset:1232 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[228:231], off, off offset:1568 nv
	scratch_load_b128 v[232:235], off, off offset:1584 nv
	scratch_load_b128 v[212:215], off, off offset:1696 nv
	scratch_load_b128 v[216:219], off, off offset:1712 nv
	s_set_vgpr_msb 0x80
	scratch_load_b128 v[250:253] /*v[762:765]*/, off, off offset:2016 nv
	scratch_load_b128 v[254:257] /*v[766:769]*/, off, off offset:2032 nv
	s_set_vgpr_msb 0x8040
	v_mov_b64_e32 v[32:33] /*v[288:289]*/, v[106:107]
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[104:105]
	v_mov_b64_e32 v[28:29] /*v[284:285]*/, v[102:103]
	v_mov_b64_e32 v[26:27] /*v[282:283]*/, v[100:101]
	s_set_vgpr_msb 0x4003
	s_clause 0x1
	scratch_load_b128 v[104:107], off, off offset:480 nv
	scratch_load_b128 v[108:111], off, off offset:496 nv
	v_mov_b64_e32 v[242:243], v[170:171] /*v[938:939]*/
	v_mov_b64_e32 v[244:245], v[172:173] /*v[940:941]*/
	v_mov_b64_e32 v[246:247], v[174:175] /*v[942:943]*/
	v_mov_b64_e32 v[248:249], v[176:177] /*v[944:945]*/
	s_set_vgpr_msb 0x3c0
	s_clause 0x1d
	scratch_load_b128 v[170:173] /*v[938:941]*/, off, off offset:2048 nv
	scratch_load_b128 v[174:177] /*v[942:945]*/, off, off offset:2064 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[48:51], off, off offset:1440 nv
	scratch_load_b128 v[52:55], off, off offset:1456 nv
	scratch_load_b128 v[64:67], off, off offset:1472 nv
	scratch_load_b128 v[68:71], off, off offset:1488 nv
	scratch_load_b128 v[96:99], off, off offset:608 nv
	scratch_load_b128 v[100:103], off, off offset:624 nv
	scratch_load_b128 v[32:35], off, off offset:1408 nv
	scratch_load_b128 v[36:39], off, off offset:1424 nv
	scratch_load_b128 v[72:75], off, off offset:1600 nv
	scratch_load_b128 v[76:79], off, off offset:1616 nv
	scratch_load_b128 v[88:91], off, off offset:1728 nv
	scratch_load_b128 v[92:95], off, off offset:1744 nv
	scratch_load_b128 v[128:131], off, off offset:640 nv
	scratch_load_b128 v[132:135], off, off offset:656 nv
	scratch_load_b128 v[152:155], off, off offset:1504 nv
	scratch_load_b128 v[156:159], off, off offset:1520 nv
	scratch_load_b128 v[160:163], off, off offset:1536 nv
	scratch_load_b128 v[164:167], off, off offset:1552 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b128 v[178:181] /*v[946:949]*/, off, off offset:672 nv
	scratch_load_b128 v[182:185] /*v[950:953]*/, off, off offset:688 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[42:45] /*v[298:301]*/, off, off offset:736 nv
	scratch_load_b128 v[46:49] /*v[302:305]*/, off, off offset:752 nv
	s_set_vgpr_msb 0x4003
	scratch_load_b128 v[250:253], off, off offset:1280 nv
	scratch_load_b128 v[254:257], off, off offset:1296 nv
	v_mov_b64_e32 v[176:177], v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[178:179], v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[180:181], v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[182:183], v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x3c2
	v_mov_b64_e32 v[224:225] /*v[992:993]*/, v[144:145] /*v[656:657]*/
	v_mov_b64_e32 v[222:223] /*v[990:991]*/, v[142:143] /*v[654:655]*/
	v_mov_b64_e32 v[220:221] /*v[988:989]*/, v[140:141] /*v[652:653]*/
	v_mov_b64_e32 v[218:219] /*v[986:987]*/, v[138:139] /*v[650:651]*/
	s_set_vgpr_msb 0xc281
	v_mov_b64_e32 v[144:145] /*v[656:657]*/, v[160:161] /*v[416:417]*/
	v_mov_b64_e32 v[142:143] /*v[654:655]*/, v[158:159] /*v[414:415]*/
	v_mov_b64_e32 v[140:141] /*v[652:653]*/, v[156:157] /*v[412:413]*/
	v_mov_b64_e32 v[138:139] /*v[650:651]*/, v[154:155] /*v[410:411]*/
	s_set_vgpr_msb 0x8141
	v_mov_b64_e32 v[160:161] /*v[416:417]*/, v[128:129] /*v[384:385]*/
	v_mov_b64_e32 v[158:159] /*v[414:415]*/, v[126:127] /*v[382:383]*/
	v_mov_b64_e32 v[156:157] /*v[412:413]*/, v[124:125] /*v[380:381]*/
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[122:123] /*v[378:379]*/
	s_clause 0x1
	scratch_load_b128 v[162:165] /*v[418:421]*/, off, off offset:2208 th:TH_LOAD_LU nv
	scratch_load_b128 v[166:169] /*v[422:425]*/, off, off offset:2224 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x41cc
	v_add_nc_u32_e32 v12 /*v780*/, s3, v12 /*v780*/
	s_set_vgpr_msb 0xcc41
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[160:161] /*v[416:417]*/
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[158:159] /*v[414:415]*/
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[156:157] /*v[412:413]*/
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[154:155] /*v[410:411]*/
	s_set_vgpr_msb 0x4142
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[138:139] /*v[650:651]*/
	v_mov_b64_e32 v[156:157] /*v[412:413]*/, v[140:141] /*v[652:653]*/
	v_mov_b64_e32 v[158:159] /*v[414:415]*/, v[142:143] /*v[654:655]*/
	v_mov_b64_e32 v[160:161] /*v[416:417]*/, v[144:145] /*v[656:657]*/
	s_set_vgpr_msb 0x4283
	v_mov_b64_e32 v[138:139] /*v[650:651]*/, v[218:219] /*v[986:987]*/
	v_mov_b64_e32 v[140:141] /*v[652:653]*/, v[220:221] /*v[988:989]*/
	v_mov_b64_e32 v[142:143] /*v[654:655]*/, v[222:223] /*v[990:991]*/
	v_mov_b64_e32 v[144:145] /*v[656:657]*/, v[224:225] /*v[992:993]*/
	s_set_vgpr_msb 0x83c0
	v_mov_b64_e32 v[224:225] /*v[992:993]*/, v[182:183]
	v_mov_b64_e32 v[222:223] /*v[990:991]*/, v[180:181]
	v_mov_b64_e32 v[220:221] /*v[988:989]*/, v[178:179]
	v_mov_b64_e32 v[218:219] /*v[986:987]*/, v[176:177]
	s_add_nc_u64 s[20:21], s[20:21], -1
	s_set_vgpr_msb 0xc08d
	s_wait_loadcnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[194:201] /*v[450:457]*/, v[66:73] /*v[834:841]*/, 0
	s_set_vgpr_msb 0x8d06
	ds_store_b128 v199 /*v711*/, v[194:197] /*v[450:453]*/
	ds_store_b128 v199 /*v711*/, v[198:201] /*v[454:457]*/ offset:32
	ds_store_b128 v199 /*v711*/, v[226:229] /*v[482:485]*/ offset:128
	ds_store_b128 v199 /*v711*/, v[230:233] /*v[486:489]*/ offset:160
	s_set_vgpr_msb 0x6ad
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[210:217] /*v[466:473]*/, v[82:89] /*v[850:857]*/, v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xad06
	ds_store_b128 v199 /*v711*/, v[242:245] /*v[498:501]*/ offset:192
	ds_store_b128 v199 /*v711*/, v[246:249] /*v[502:505]*/ offset:224
	s_set_vgpr_msb 0x6a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[226:233] /*v[482:489]*/, v[120:127], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa106
	ds_store_b128 v199 /*v711*/, v[210:213] /*v[466:469]*/ offset:64
	ds_store_b128 v199 /*v711*/, v[214:217] /*v[470:473]*/ offset:96
	s_set_vgpr_msb 0x60a
	ds_store_b128 v200 /*v712*/, v[2:5] /*v[514:517]*/
	ds_store_b128 v200 /*v712*/, v[6:9] /*v[518:521]*/ offset:32
	ds_store_b128 v200 /*v712*/, v[210:213] /*v[722:725]*/ offset:64
	ds_store_b128 v200 /*v712*/, v[214:217] /*v[726:729]*/ offset:96
	ds_store_b128 v200 /*v712*/, v[226:229] /*v[738:741]*/ offset:128
	ds_store_b128 v200 /*v712*/, v[230:233] /*v[742:745]*/ offset:160
	s_set_vgpr_msb 0xaa1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[242:249] /*v[498:505]*/, v[192:199], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa10a
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:256 nv
	scratch_load_b128 v[196:199], off, off offset:272 nv
	ds_store_b128 v200 /*v712*/, v[122:125] /*v[634:637]*/ offset:192
	ds_store_b128 v200 /*v712*/, v[126:129] /*v[638:641]*/ offset:224
	s_set_vgpr_msb 0xa8d
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[202:209] /*v[458:465]*/, v[106:113] /*v[874:881]*/, 0
	s_set_vgpr_msb 0x8d88
	v_add_nc_u32_e32 v201 /*v713*/, s3, v201 /*v713*/
	s_set_vgpr_msb 0x88ad
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[218:225] /*v[474:481]*/, v[50:57] /*v[818:825]*/, v[18:25] /*v[530:537]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[234:241] /*v[490:497]*/, v[42:49] /*v[810:817]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xada1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[250:257] /*v[506:513]*/, v[192:199], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa100
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:288 nv
	scratch_load_b128 v[196:199], off, off offset:304 nv
	s_set_vgpr_msb 0xad
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[194:201] /*v[450:457]*/, v[138:145] /*v[906:913]*/, 0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[210:217] /*v[466:473]*/, v[146:153] /*v[914:921]*/, v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xada1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[226:233] /*v[482:489]*/, v[112:119], v[26:33] /*v[538:545]*/
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[242:249] /*v[498:505]*/, v[192:199], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa100
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:320 nv
	scratch_load_b128 v[196:199], off, off offset:336 nv
	s_set_vgpr_msb 0x85
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[194:201] /*v[450:457]*/, v[186:193] /*v[442:449]*/, 0
	s_set_vgpr_msb 0x858d
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[194:201] /*v[450:457]*/, v[242:249] /*v[1010:1017]*/, 0
	s_set_vgpr_msb 0x8d81
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[194:201] /*v[450:457]*/, v[0:7], 0
	s_set_vgpr_msb 0x8189
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[194:201] /*v[450:457]*/, v[180:187] /*v[692:699]*/, 0
	s_set_vgpr_msb 0x898d
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[770:777]*/, 0
	s_set_vgpr_msb 0x8d81
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[194:201] /*v[450:457]*/, v[16:23], 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x814a
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[196:197] /*v[708:709]*/, v[10:11] /*v[522:523]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[708:709]*/, v[12:13] /*v[524:525]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[14:15] /*v[526:527]*/
	s_set_vgpr_msb 0x4aad
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[210:217] /*v[466:473]*/, v[30:37] /*v[798:805]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xad4a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x4aa1
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[210:217] /*v[466:473]*/, v[40:47], v[58:65] /*v[570:577]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[210:217] /*v[466:473]*/, v[24:31], v[74:81] /*v[586:593]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[210:217] /*v[466:473]*/, v[56:63], v[90:97] /*v[602:609]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[210:217] /*v[466:473]*/, v[8:15], v[106:113] /*v[618:625]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[210:217] /*v[466:473]*/, v[80:87], v[146:153] /*v[658:665]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa148
	v_or_b32_e32 v216 /*v472*/, 3, v198 /*v710*/
	v_or_b32_e32 v217 /*v473*/, 2, v198 /*v710*/
	s_set_vgpr_msb 0x48a1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[226:233] /*v[482:489]*/, v[136:143], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa1ad
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[226:233] /*v[482:489]*/, v[58:65] /*v[826:833]*/, v[58:65] /*v[570:577]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[226:233] /*v[482:489]*/, v[194:201] /*v[962:969]*/, v[74:81] /*v[586:593]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[226:233] /*v[482:489]*/, v[186:193] /*v[954:961]*/, v[90:97] /*v[602:609]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[226:233] /*v[482:489]*/, v[14:21] /*v[782:789]*/, v[106:113] /*v[618:625]*/
	s_set_vgpr_msb 0xada1
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[226:233] /*v[482:489]*/, v[144:151], v[146:153] /*v[658:665]*/
	s_set_vgpr_msb 0xa140
	s_clause 0x1
	scratch_load_b64 v[230:231] /*v[486:487]*/, off, off offset:3424 nv
	scratch_load_b64 v[232:233] /*v[488:489]*/, off, off offset:3432 nv
	s_set_vgpr_msb 0x40ad
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[242:249] /*v[498:505]*/, v[226:233] /*v[994:1001]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xada9
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[242:249] /*v[498:505]*/, v[154:161] /*v[666:673]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa9a5
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[242:249] /*v[498:505]*/, v[18:25] /*v[274:281]*/, v[74:81] /*v[586:593]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[242:249] /*v[498:505]*/, v[2:9] /*v[258:265]*/, v[90:97] /*v[602:609]*/
	s_set_vgpr_msb 0xa5a1
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[242:249] /*v[498:505]*/, v[228:235], v[106:113] /*v[618:625]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[242:249] /*v[498:505]*/, v[212:219], v[146:153] /*v[658:665]*/
	s_set_vgpr_msb 0xa140
	s_clause 0x1
	scratch_load_b64 v[246:247] /*v[502:503]*/, off, off offset:3548 nv
	scratch_load_b64 v[248:249] /*v[504:505]*/, off, off offset:3476 nv
	s_set_vgpr_msb 0x4089
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[202:209] /*v[458:465]*/, v[250:257] /*v[762:769]*/, 0
	s_set_vgpr_msb 0x89a1
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[218:225] /*v[474:481]*/, v[104:111], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa1a9
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[234:241] /*v[490:497]*/, v[188:195] /*v[700:707]*/, v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa9a1
	s_wait_loadcnt 0x4
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[250:257] /*v[506:513]*/, v[192:199], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa100
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:1760 nv
	scratch_load_b128 v[196:199], off, off offset:1776 nv
	s_set_vgpr_msb 0x8d
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[202:209] /*v[458:465]*/, v[170:177] /*v[938:945]*/, 0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[202:209] /*v[458:465]*/, v[122:129] /*v[890:897]*/, 0
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[202:209] /*v[458:465]*/, v[154:161] /*v[922:929]*/, 0
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[202:209] /*v[458:465]*/, v[130:137] /*v[898:905]*/, 0
	s_set_vgpr_msb 0x8d81
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[202:209] /*v[458:465]*/, v[48:55], 0
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[202:209] /*v[458:465]*/, v[64:71], 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x814a
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[18:19] /*v[530:531]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0x4aa1
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[218:225] /*v[474:481]*/, v[96:103], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa1ad
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[218:225] /*v[474:481]*/, v[98:105] /*v[866:873]*/, v[66:73] /*v[578:585]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[218:225] /*v[474:481]*/, v[114:121] /*v[882:889]*/, v[82:89] /*v[594:601]*/
	s_set_vgpr_msb 0xada1
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[218:225] /*v[474:481]*/, v[32:39], v[98:105] /*v[610:617]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[218:225] /*v[474:481]*/, v[72:79], v[130:137] /*v[642:649]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[218:225] /*v[474:481]*/, v[88:95], v[242:249] /*v[754:761]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa148
	v_or_b32_e32 v222 /*v478*/, 5, v198 /*v710*/
	v_or_b32_e32 v223 /*v479*/, 4, v198 /*v710*/
	v_or_b32_e32 v224 /*v480*/, 7, v198 /*v710*/
	s_set_vgpr_msb 0x48a1
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[234:241] /*v[490:497]*/, v[128:135], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa148
	v_or_b32_e32 v225 /*v481*/, 6, v198 /*v710*/
	s_set_vgpr_msb 0x48ad
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[234:241] /*v[490:497]*/, v[90:97] /*v[858:865]*/, v[66:73] /*v[578:585]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[234:241] /*v[490:497]*/, v[162:169] /*v[930:937]*/, v[82:89] /*v[594:601]*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[234:241] /*v[490:497]*/, v[22:29] /*v[790:797]*/, v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xada1
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[234:241] /*v[490:497]*/, v[152:159], v[130:137] /*v[642:649]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[234:241] /*v[490:497]*/, v[160:167], v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa146
	s_clause 0x1
	scratch_load_b64 v[238:239] /*v[494:495]*/, off, off offset:3492 nv
	scratch_load_b64 v[240:241] /*v[496:497]*/, off, off offset:3468 nv
	s_wait_loadcnt 0x5
	v_cmp_ge_i32_e32 vcc_lo, v198 /*v710*/, v246 /*v502*/
	v_cmp_gt_i32_e64 s2, v198 /*v710*/, v246 /*v502*/
	s_set_vgpr_msb 0x46ad
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[250:257] /*v[506:513]*/, v[178:185] /*v[946:953]*/, v[50:57] /*v[562:569]*/
	s_and_b32 s5, s40, vcc_lo
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0xad45
	v_cndmask_b32_e64 v195 /*v451*/, v195 /*v451*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v194 /*v450*/, v194 /*v450*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v231 /*v487*/
	v_cmp_gt_i32_e64 s2, v217 /*v473*/, v246 /*v502*/
	s_set_vgpr_msb 0x45a5
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[250:257] /*v[506:513]*/, v[42:49] /*v[298:305]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa54d
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v196 /*v452*/, v196 /*v452*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4d45
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v231 /*v487*/
	v_cndmask_b32_e64 v197 /*v453*/, v197 /*v453*/, 0xff61b1e6, s5
	v_cmp_gt_i32_e64 s2, v223 /*v479*/, v246 /*v502*/
	v_exp_f32_e32 v194 /*v450*/, v194 /*v450*/
	v_exp_f32_e32 v195 /*v451*/, v195 /*v451*/
	s_set_vgpr_msb 0x45a5
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[250:257] /*v[506:513]*/, v[10:17] /*v[266:273]*/, v[82:89] /*v[594:601]*/
	s_set_vgpr_msb 0xa54d
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4d45
	v_cmp_gt_i32_e64 s2, v225 /*v481*/, v246 /*v502*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v224 /*v480*/, v231 /*v487*/
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v196 /*v452*/, v196 /*v452*/
	v_exp_f32_e32 v197 /*v453*/, v197 /*v453*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[202:203] /*v[458:459]*/, v[194:195] /*v[450:451]*/
	s_set_vgpr_msb 0x454d
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4d4a
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[20:21] /*v[532:533]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a41
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v198 /*v710*/, v252 /*v1020*/
	s_set_vgpr_msb 0xe46
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[196:197] /*v[708:709]*/, v[194:195] /*v[450:451]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[202:203] /*v[458:459]*/, v[196:197] /*v[452:453]*/
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s5
	v_cvt_pk_bf16_f32 v194 /*v450*/, v194 /*v450*/, v195 /*v451*/
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[708:709]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x464d
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4d0e
	v_cmp_ge_i32_e32 vcc_lo, v198 /*v710*/, v252 /*v1020*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0xe4a
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[196:197] /*v[708:709]*/, v[30:31] /*v[542:543]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v195 /*v451*/, v196 /*v452*/, v197 /*v453*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[22:23] /*v[534:535]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4a0d
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v39 /*v807*/
	s_set_vgpr_msb 0xda1
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[250:257] /*v[506:513]*/, v[250:257], v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xa145
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[24:25] /*v[536:537]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[708:709]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x46a1
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[250:257] /*v[506:513]*/, v[220:227], v[130:137] /*v[642:649]*/
	s_set_vgpr_msb 0xa145
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v196 /*v452*/, v196 /*v452*/, v197 /*v453*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v197 /*v453*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[26:27] /*v[538:539]*/
	s_set_vgpr_msb 0x4aa1
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[250:257] /*v[506:513]*/, v[192:199], v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa14d
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v217 /*v473*/, v252 /*v1020*/
	s_set_vgpr_msb 0x4d80
	scratch_load_b64 v[0:1] /*v[512:513]*/, off, off offset:3484 nv
	s_set_vgpr_msb 0x8049
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x4982
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[202:209] /*v[714:721]*/, v[48:55], 0
	s_set_vgpr_msb 0x824d
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v223 /*v479*/, v252 /*v1020*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v39 /*v807*/
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_cndmask_b32_e64 v202 /*v458*/, v202 /*v458*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v225 /*v481*/, v252 /*v1020*/
	s_set_vgpr_msb 0x4d49
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	s_set_vgpr_msb 0x49a2
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[218:225] /*v[730:737]*/, v[72:79], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa201
	v_mov_b64_e32 v[48:49], v[50:51] /*v[306:307]*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x14d
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v224 /*v480*/, v39 /*v807*/
	v_cndmask_b32_e64 v203 /*v459*/, v203 /*v459*/, 0xff61b1e6, s5
	s_set_vgpr_msb 0x4d01
	v_mov_b64_e32 v[50:51], v[52:53] /*v[308:309]*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	s_set_vgpr_msb 0x41a2
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[234:241] /*v[746:753]*/, v[152:159], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa249
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4901
	v_mov_b64_e32 v[52:53], v[54:55] /*v[310:311]*/
	v_mov_b64_e32 v[54:55], v[56:57] /*v[312:313]*/
	s_set_vgpr_msb 0x141
	v_mov_b64_e32 v[50:51] /*v[306:307]*/, v[74:75] /*v[330:331]*/
	v_mov_b64_e32 v[52:53] /*v[308:309]*/, v[76:77] /*v[332:333]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_ge_i32_e32 vcc_lo, v198 /*v710*/, v40 /*v808*/
	s_set_vgpr_msb 0xea0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[184:191], v[220:227], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa041
	v_mov_b64_e32 v[54:55] /*v[310:311]*/, v[78:79] /*v[334:335]*/
	v_exp_f32_e32 v204 /*v460*/, v202 /*v458*/
	v_exp_f32_e32 v205 /*v461*/, v203 /*v459*/
	v_nop
	s_set_vgpr_msb 0x414a
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[196:197] /*v[708:709]*/, v[32:33] /*v[544:545]*/
	s_set_vgpr_msb 0x4a41
	v_mov_b64_e32 v[56:57] /*v[312:313]*/, v[80:81] /*v[336:337]*/
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[106:107] /*v[362:363]*/
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[108:109] /*v[364:365]*/
	s_set_vgpr_msb 0x418e
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[2:9] /*v[514:521]*/, v[2:9] /*v[770:777]*/, 0
	s_set_vgpr_msb 0x8e41
	v_cndmask_b32_e64 v203 /*v459*/, v203 /*v459*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v202 /*v458*/, v202 /*v458*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x418e
	v_cmp_gt_i32_e64 s2, v198 /*v710*/, v40 /*v808*/
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e49
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[110:111] /*v[366:367]*/
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[112:113] /*v[368:369]*/
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x49a2
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[210:217] /*v[722:729]*/, v[8:15], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa24d
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[138:139] /*v[394:395]*/
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[140:141] /*v[396:397]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v41 /*v809*/
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[142:143] /*v[398:399]*/
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[144:145] /*v[400:401]*/
	v_exp_f32_e32 v206 /*v462*/, v202 /*v458*/
	v_exp_f32_e32 v207 /*v463*/, v203 /*v459*/
	v_nop
	s_set_vgpr_msb 0x4d4a
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[34:35] /*v[546:547]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aae
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[226:233] /*v[738:745]*/, v[14:21] /*v[782:789]*/, v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xae45
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[170:171] /*v[426:427]*/
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[172:173] /*v[428:429]*/
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[202:203] /*v[458:459]*/, v[198:199] /*v[454:455]*/
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[176:177] /*v[432:433]*/
	s_set_vgpr_msb 0x4543
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[202:203] /*v[970:971]*/
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[204:205] /*v[972:973]*/
	s_set_vgpr_msb 0x43a2
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[122:129] /*v[634:641]*/, v[228:235], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa246
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4600
	s_clause 0x1
	scratch_load_b128 v[228:231], off, off offset:2816 th:TH_LOAD_LU nv
	scratch_load_b128 v[232:235], off, off offset:2832 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x43
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[206:207] /*v[974:975]*/
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[208:209] /*v[976:977]*/
	s_set_vgpr_msb 0x43c0
	v_mov_b64_e32 v[208:209] /*v[976:977]*/, v[174:175]
	s_set_vgpr_msb 0xc045
	v_cvt_pk_bf16_f32 v202 /*v458*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[36:37] /*v[548:549]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a82
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[2:9] /*v[514:521]*/, v[16:23], 0
	s_set_vgpr_msb 0x82c0
	v_mov_b64_e32 v[206:207] /*v[974:975]*/, v[172:173]
	v_mov_b64_e32 v[204:205] /*v[972:973]*/, v[170:171]
	v_mov_b64_e32 v[202:203] /*v[970:971]*/, v[168:169]
	s_set_vgpr_msb 0xc045
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x4a46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x46a2
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[210:217] /*v[722:729]*/, v[80:87], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[76:79], off, off offset:3040 th:TH_LOAD_LU nv
	scratch_load_b128 v[80:83], off, off offset:3056 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v203 /*v459*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[38:39] /*v[550:551]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a45
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[204:205] /*v[460:461]*/
	s_set_vgpr_msb 0x45a2
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[226:233] /*v[738:745]*/, v[144:151], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[148:151], off, off offset:3136 th:TH_LOAD_LU nv
	scratch_load_b128 v[152:155], off, off offset:3152 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v204 /*v460*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[40:41] /*v[552:553]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a8e
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[202:209] /*v[714:721]*/, v[130:137] /*v[898:905]*/, 0
	s_set_vgpr_msb 0x8e45
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[206:207] /*v[462:463]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4a46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x46a2
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[218:225] /*v[730:737]*/, v[32:39], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa245
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v205 /*v461*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[42:43] /*v[554:555]*/
	s_set_vgpr_msb 0x4a4d
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v217 /*v473*/, v40 /*v808*/
	s_set_vgpr_msb 0x4d8a
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[2:9] /*v[514:521]*/, v[180:187] /*v[692:699]*/, 0
	s_set_vgpr_msb 0x8a49
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_wait_loadcnt 0x8
	v_pk_add_f32 v[210:211] /*v[466:467]*/, v[238:239] /*v[494:495]*/, v[50:51] /*v[562:563]*/ neg_lo:[1,0] neg_hi:[1,0]
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x494d
	v_cmp_gt_i32_e64 s2, v223 /*v479*/, v40 /*v808*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v41 /*v809*/
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	v_cndmask_b32_e64 v206 /*v462*/, v206 /*v462*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4d49
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490d
	v_cmp_gt_i32_e64 s2, v225 /*v481*/, v40 /*v808*/
	s_set_vgpr_msb 0xd46
	s_wait_loadcnt 0x7
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[66:67] /*v[578:579]*/, v[240:241] /*v[496:497]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x46a2
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[210:217] /*v[722:729]*/, v[56:63], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa245
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[210:211] /*v[466:467]*/, v[198:199] /*v[454:455]*/
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x454d
	v_cmp_gt_i32_e32 vcc_lo, v224 /*v480*/, v41 /*v809*/
	v_cndmask_b32_e64 v207 /*v463*/, v207 /*v463*/, 0xff61b1e6, s5
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	s_set_vgpr_msb 0x4d46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4649
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	s_and_b32 s2, s40, s2
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v208 /*v464*/, v208 /*v464*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v210 /*v466*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[52:53] /*v[564:565]*/, v[238:239] /*v[494:495]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x460e
	v_cmp_gt_i32_e64 s2, v198 /*v710*/, v38 /*v806*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_ge_i32_e32 vcc_lo, v198 /*v710*/, v38 /*v806*/
	s_set_vgpr_msb 0xe45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[200:201] /*v[456:457]*/
	v_cndmask_b32_e64 v209 /*v465*/, v209 /*v465*/, 0xff61b1e6, s5
	v_exp_f32_e32 v206 /*v462*/, v206 /*v462*/
	v_exp_f32_e32 v207 /*v463*/, v207 /*v463*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[164:165] /*v[676:677]*/, v[208:209] /*v[464:465]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x464a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[60:61] /*v[572:573]*/
	s_set_vgpr_msb 0x4aae
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[226:233] /*v[738:745]*/, v[186:193] /*v[954:961]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xae45
	v_cvt_pk_bf16_f32 v211 /*v467*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[54:55] /*v[566:567]*/, v[238:239] /*v[494:495]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v233 /*v489*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[206:207] /*v[462:463]*/
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v208 /*v464*/, v208 /*v464*/
	v_exp_f32_e32 v209 /*v465*/, v209 /*v465*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[62:63] /*v[574:575]*/
	s_set_vgpr_msb 0x4aa6
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[122:129] /*v[634:641]*/, v[2:9] /*v[258:265]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa646
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v212 /*v468*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[56:57] /*v[568:569]*/, v[238:239] /*v[494:495]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x468e
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[202:209] /*v[714:721]*/, v[154:161] /*v[922:929]*/, 0
	s_set_vgpr_msb 0x8e45
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[208:209] /*v[464:465]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[64:65] /*v[576:577]*/
	s_set_vgpr_msb 0x4a46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x46ae
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[218:225] /*v[730:737]*/, v[114:121] /*v[882:889]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xae45
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v213 /*v469*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[58:59] /*v[570:571]*/
	s_set_vgpr_msb 0x4a4d
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v217 /*v473*/, v38 /*v806*/
	s_set_vgpr_msb 0x4d82
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[2:9] /*v[514:521]*/, v[0:7], 0
	s_set_vgpr_msb 0x8200
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:224 nv
	scratch_load_b128 v[4:7], off, off offset:240 nv
	s_set_vgpr_msb 0x49
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x494d
	v_cmp_gt_i32_e64 s2, v223 /*v479*/, v38 /*v806*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4d45
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v233 /*v489*/
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	v_cndmask_b32_e64 v206 /*v462*/, v206 /*v462*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490d
	v_cmp_gt_i32_e64 s2, v225 /*v481*/, v38 /*v806*/
	s_set_vgpr_msb 0xda2
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[210:217] /*v[722:729]*/, v[24:31], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x3
	scratch_load_b128 v[28:31], off, off offset:3296 th:TH_LOAD_LU nv
	scratch_load_b128 v[32:35], off, off offset:3312 th:TH_LOAD_LU nv
	scratch_load_b128 v[20:23], off, off offset:2784 th:TH_LOAD_LU nv
	scratch_load_b128 v[24:27], off, off offset:2800 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x45
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[214:215] /*v[470:471]*/, v[198:199] /*v[454:455]*/
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v224 /*v480*/, v233 /*v489*/
	v_cndmask_b32_e64 v207 /*v463*/, v207 /*v463*/, 0xff61b1e6, s5
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4649
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	s_and_b32 s2, s40, s2
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v208 /*v464*/, v208 /*v464*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v218 /*v474*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[68:69] /*v[580:581]*/, v[240:241] /*v[496:497]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x460e
	v_cmp_gt_i32_e64 s2, v198 /*v710*/, v78 /*v846*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_ge_i32_e32 vcc_lo, v198 /*v710*/, v78 /*v846*/
	s_set_vgpr_msb 0xe45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[200:201] /*v[456:457]*/
	v_cndmask_b32_e64 v209 /*v465*/, v209 /*v465*/, 0xff61b1e6, s5
	v_exp_f32_e32 v206 /*v462*/, v206 /*v462*/
	v_exp_f32_e32 v207 /*v463*/, v207 /*v463*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[166:167] /*v[678:679]*/, v[208:209] /*v[464:465]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x464a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[76:77] /*v[588:589]*/
	s_set_vgpr_msb 0x4a46
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[82:83] /*v[594:595]*/, v[248:249] /*v[504:505]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x46ae
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[226:233] /*v[738:745]*/, v[194:201] /*v[962:969]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xae45
	v_cvt_pk_bf16_f32 v219 /*v475*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[70:71] /*v[582:583]*/, v[240:241] /*v[496:497]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x464d
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v79 /*v847*/
	s_set_vgpr_msb 0x4dc0
	s_clause 0x1
	scratch_load_b128 v[192:195] /*v[960:963]*/, off, off offset:2688 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199] /*v[964:967]*/, off, off offset:2704 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc045
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[206:207] /*v[462:463]*/
	v_exp_f32_e32 v208 /*v464*/, v208 /*v464*/
	v_exp_f32_e32 v209 /*v465*/, v209 /*v465*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[78:79] /*v[590:591]*/
	s_set_vgpr_msb 0x4aa6
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[122:129] /*v[634:641]*/, v[18:25] /*v[274:281]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa646
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v220 /*v476*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[72:73] /*v[584:585]*/, v[240:241] /*v[496:497]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x468e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[202:209] /*v[714:721]*/, v[122:129] /*v[890:897]*/, 0
	s_set_vgpr_msb 0x8e45
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[208:209] /*v[464:465]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[80:81] /*v[592:593]*/
	s_set_vgpr_msb 0x4a46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x46ae
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[218:225] /*v[730:737]*/, v[98:105] /*v[866:873]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xae45
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v221 /*v477*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[74:75] /*v[586:587]*/
	s_set_vgpr_msb 0x4a4d
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v217 /*v473*/, v78 /*v846*/
	s_set_vgpr_msb 0x4d8e
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[2:9] /*v[514:521]*/, v[242:249] /*v[1010:1017]*/, 0
	s_set_vgpr_msb 0x8e49
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x494d
	v_cmp_gt_i32_e64 s2, v223 /*v479*/, v78 /*v846*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v79 /*v847*/
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	v_cndmask_b32_e64 v206 /*v462*/, v206 /*v462*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4d49
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490d
	v_cmp_gt_i32_e64 s2, v225 /*v481*/, v78 /*v846*/
	s_set_vgpr_msb 0xda2
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[210:217] /*v[722:729]*/, v[40:47], v[74:81] /*v[586:593]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[36:39], off, off offset:2848 th:TH_LOAD_LU nv
	scratch_load_b128 v[40:43], off, off offset:2864 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x45
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[214:215] /*v[470:471]*/, v[198:199] /*v[454:455]*/
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x454d
	v_cmp_gt_i32_e32 vcc_lo, v224 /*v480*/, v79 /*v847*/
	v_cndmask_b32_e64 v207 /*v463*/, v207 /*v463*/, 0xff61b1e6, s5
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	s_set_vgpr_msb 0x4d46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4649
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	s_and_b32 s2, s40, s2
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v208 /*v464*/, v208 /*v464*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v226 /*v482*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[84:85] /*v[596:597]*/, v[248:249] /*v[504:505]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x460e
	v_cmp_gt_i32_e64 s2, v198 /*v710*/, v254 /*v1022*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_ge_i32_e32 vcc_lo, v198 /*v710*/, v254 /*v1022*/
	s_set_vgpr_msb 0xe45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[200:201] /*v[456:457]*/
	v_cndmask_b32_e64 v209 /*v465*/, v209 /*v465*/, 0xff61b1e6, s5
	v_exp_f32_e32 v206 /*v462*/, v206 /*v462*/
	v_exp_f32_e32 v207 /*v463*/, v207 /*v463*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[168:169] /*v[680:681]*/, v[208:209] /*v[464:465]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x464a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[92:93] /*v[604:605]*/
	s_wait_loadcnt 0x10
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[98:99] /*v[610:611]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aae
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[226:233] /*v[738:745]*/, v[58:65] /*v[826:833]*/, v[74:81] /*v[586:593]*/
	s_set_vgpr_msb 0xae45
	v_cvt_pk_bf16_f32 v227 /*v483*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[86:87] /*v[598:599]*/, v[248:249] /*v[504:505]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x464d
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v77 /*v845*/
	s_set_vgpr_msb 0x4d45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[206:207] /*v[462:463]*/
	v_exp_f32_e32 v208 /*v464*/, v208 /*v464*/
	v_exp_f32_e32 v209 /*v465*/, v209 /*v465*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[94:95] /*v[606:607]*/
	s_set_vgpr_msb 0x4aaa
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[122:129] /*v[634:641]*/, v[154:161] /*v[666:673]*/, v[74:81] /*v[586:593]*/
	s_set_vgpr_msb 0xaa46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4680
	s_clause 0x1
	scratch_load_b128 v[154:157] /*v[666:669]*/, off, off offset:3232 th:TH_LOAD_LU nv
	scratch_load_b128 v[158:161] /*v[670:673]*/, off, off offset:3248 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8045
	v_cvt_pk_bf16_f32 v228 /*v484*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[88:89] /*v[600:601]*/, v[248:249] /*v[504:505]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x468e
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[202:209] /*v[714:721]*/, v[170:177] /*v[938:945]*/, 0
	s_set_vgpr_msb 0x8e45
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[208:209] /*v[464:465]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x4a46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x46a2
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[218:225] /*v[730:737]*/, v[96:103], v[82:89] /*v[594:601]*/
	s_set_vgpr_msb 0xa245
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v229 /*v485*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x4a4d
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v217 /*v473*/, v254 /*v1022*/
	s_set_vgpr_msb 0x4d86
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[2:9] /*v[514:521]*/, v[186:193] /*v[442:449]*/, 0
	s_set_vgpr_msb 0x8649
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x494d
	v_cmp_gt_i32_e64 s2, v223 /*v479*/, v254 /*v1022*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v77 /*v845*/
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	v_cndmask_b32_e64 v206 /*v462*/, v206 /*v462*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4d49
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490d
	v_cmp_gt_i32_e64 s2, v225 /*v481*/, v254 /*v1022*/
	s_set_vgpr_msb 0xdae
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[210:217] /*v[722:729]*/, v[30:37] /*v[798:805]*/, v[90:97] /*v[602:609]*/
	s_set_vgpr_msb 0xae45
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[214:215] /*v[470:471]*/, v[198:199] /*v[454:455]*/
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x454d
	v_cmp_gt_i32_e32 vcc_lo, v224 /*v480*/, v77 /*v845*/
	v_cndmask_b32_e64 v207 /*v463*/, v207 /*v463*/, 0xff61b1e6, s5
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	s_set_vgpr_msb 0x4d46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4649
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	s_and_b32 s2, s40, s2
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v208 /*v464*/, v208 /*v464*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v234 /*v490*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[100:101] /*v[612:613]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a0e
	v_cmp_gt_i32_e64 s2, v198 /*v710*/, v76 /*v844*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_ge_i32_e32 vcc_lo, v198 /*v710*/, v76 /*v844*/
	s_set_vgpr_msb 0xe45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[200:201] /*v[456:457]*/
	v_cndmask_b32_e64 v209 /*v465*/, v209 /*v465*/, 0xff61b1e6, s5
	v_exp_f32_e32 v206 /*v462*/, v206 /*v462*/
	v_exp_f32_e32 v207 /*v463*/, v207 /*v463*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[170:171] /*v[682:683]*/, v[208:209] /*v[464:465]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x464a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[108:109] /*v[620:621]*/
	s_set_vgpr_msb 0x4a4e
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[130:131] /*v[642:643]*/, v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4ea2
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[226:233] /*v[738:745]*/, v[136:143], v[90:97] /*v[602:609]*/
	s_set_vgpr_msb 0xa245
	v_cvt_pk_bf16_f32 v235 /*v491*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[102:103] /*v[614:615]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a4d
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v253 /*v1021*/
	s_set_vgpr_msb 0x4d00
	s_clause 0x1
	scratch_load_b128 v[140:143], off, off offset:3072 th:TH_LOAD_LU nv
	scratch_load_b128 v[144:147], off, off offset:3088 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[206:207] /*v[462:463]*/
	v_exp_f32_e32 v208 /*v464*/, v208 /*v464*/
	v_exp_f32_e32 v209 /*v465*/, v209 /*v465*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x4aae
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[122:129] /*v[634:641]*/, v[226:233] /*v[994:1001]*/, v[90:97] /*v[602:609]*/
	s_set_vgpr_msb 0xae46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v236 /*v492*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[104:105] /*v[616:617]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a8a
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[202:209] /*v[714:721]*/, v[250:257] /*v[762:769]*/, 0
	s_set_vgpr_msb 0x8a45
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[208:209] /*v[464:465]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[112:113] /*v[624:625]*/
	v_nop
	v_nop
	s_set_vgpr_msb 0x4ac0
	v_mov_b64_e32 v[0:1] /*v[768:769]*/, v[248:249]
	s_set_vgpr_msb 0xc080
	v_mov_b64_e32 v[254:255] /*v[766:767]*/, v[246:247]
	s_set_vgpr_msb 0x80a2
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[218:225] /*v[730:737]*/, v[104:111], v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xa246
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4680
	v_mov_b64_e32 v[252:253] /*v[764:765]*/, v[244:245]
	v_mov_b64_e32 v[250:251] /*v[762:763]*/, v[242:243]
	s_set_vgpr_msb 0x8045
	v_cvt_pk_bf16_f32 v237 /*v493*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[106:107] /*v[618:619]*/
	s_set_vgpr_msb 0x4a8e
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[2:9] /*v[514:521]*/, v[138:145] /*v[906:913]*/, 0
	s_set_vgpr_msb 0x8e00
	s_clause 0x10
	scratch_load_b128 v[236:239], off, off offset:2752 th:TH_LOAD_LU nv
	scratch_load_b128 v[240:243], off, off offset:2768 th:TH_LOAD_LU nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[34:37] /*v[290:293]*/, off, off offset:2592 th:TH_LOAD_LU nv
	scratch_load_b128 v[38:41] /*v[294:297]*/, off, off offset:2608 th:TH_LOAD_LU nv
	scratch_load_b128 v[186:189] /*v[442:445]*/, off, off offset:2560 th:TH_LOAD_LU nv
	scratch_load_b128 v[190:193] /*v[446:449]*/, off, off offset:2576 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[138:141] /*v[906:909]*/, off, off offset:2464 th:TH_LOAD_LU nv
	scratch_load_b128 v[142:145] /*v[910:913]*/, off, off offset:2480 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc04d
	scratch_load_b128 v[98:101] /*v[354:357]*/, off, off offset:2432 th:TH_LOAD_LU nv
	scratch_load_b128 v[102:105] /*v[358:361]*/, off, off offset:2448 th:TH_LOAD_LU nv
	scratch_load_b128 v[130:133] /*v[386:389]*/, off, off offset:2336 th:TH_LOAD_LU nv
	scratch_load_b128 v[134:137] /*v[390:393]*/, off, off offset:2352 th:TH_LOAD_LU nv
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:2304 th:TH_LOAD_LU nv
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:2320 th:TH_LOAD_LU nv
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v217 /*v473*/, v76 /*v844*/
	s_set_vgpr_msb 0x4d49
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x49ae
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[210:217] /*v[722:729]*/, v[146:153] /*v[914:921]*/, v[106:113] /*v[618:625]*/
	s_set_vgpr_msb 0xae4d
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v223 /*v479*/, v76 /*v844*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v253 /*v1021*/
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	v_cndmask_b32_e64 v206 /*v462*/, v206 /*v462*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4d49
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490d
	v_cmp_gt_i32_e64 s2, v225 /*v481*/, v76 /*v844*/
	s_set_vgpr_msb 0xda2
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[226:233] /*v[738:745]*/, v[112:119], v[106:113] /*v[618:625]*/
	s_set_vgpr_msb 0xa2c0
	s_clause 0x1
	scratch_load_b128 v[146:149] /*v[914:917]*/, off, off offset:2624 th:TH_LOAD_LU nv
	scratch_load_b128 v[150:153] /*v[918:921]*/, off, off offset:2640 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc045
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[214:215] /*v[470:471]*/, v[198:199] /*v[454:455]*/
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x454d
	v_cmp_gt_i32_e32 vcc_lo, v224 /*v480*/, v253 /*v1021*/
	v_cndmask_b32_e64 v207 /*v463*/, v207 /*v463*/, 0xff61b1e6, s5
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	s_set_vgpr_msb 0x4d46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4649
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	s_and_b32 s2, s40, s2
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v208 /*v464*/, v208 /*v464*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v242 /*v498*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454e
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[132:133] /*v[644:645]*/, v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cmp_gt_i32_e64 s2, v198 /*v710*/, v10 /*v778*/
	s_set_vgpr_msb 0x4e41
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_ge_i32_e32 vcc_lo, v198 /*v710*/, v10 /*v778*/
	s_set_vgpr_msb 0xe45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[200:201] /*v[456:457]*/
	v_cndmask_b32_e64 v209 /*v465*/, v209 /*v465*/, 0xff61b1e6, s5
	v_exp_f32_e32 v206 /*v462*/, v206 /*v462*/
	v_exp_f32_e32 v207 /*v463*/, v207 /*v463*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[148:149] /*v[660:661]*/
	s_set_vgpr_msb 0x4a46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[172:173] /*v[684:685]*/, v[208:209] /*v[464:465]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x464e
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[242:243] /*v[754:755]*/, v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4e81
	v_mov_b64_e32 v[242:243] /*v[754:755]*/, v[66:67] /*v[322:323]*/
	s_set_vgpr_msb 0x8145
	v_cvt_pk_bf16_f32 v243 /*v499*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454e
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[134:135] /*v[646:647]*/, v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4e4d
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v11 /*v779*/
	s_set_vgpr_msb 0x4daa
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[234:241] /*v[746:753]*/, v[188:195] /*v[700:707]*/, v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xaa45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[206:207] /*v[462:463]*/
	v_exp_f32_e32 v208 /*v464*/, v208 /*v464*/
	v_exp_f32_e32 v209 /*v465*/, v209 /*v465*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[150:151] /*v[662:663]*/
	s_set_vgpr_msb 0x4a46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v244 /*v500*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454e
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[136:137] /*v[648:649]*/, v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4e8e
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[202:209] /*v[714:721]*/, v[106:113] /*v[874:881]*/, 0
	s_set_vgpr_msb 0x8e45
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[208:209] /*v[464:465]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[152:153] /*v[664:665]*/
	v_nop
	v_nop
	s_set_vgpr_msb 0x4ac1
	v_mov_b64_e32 v[112:113] /*v[880:881]*/, v[32:33] /*v[288:289]*/
	v_mov_b64_e32 v[110:111] /*v[878:879]*/, v[30:31] /*v[286:287]*/
	s_set_vgpr_msb 0xc1ae
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[218:225] /*v[730:737]*/, v[50:57] /*v[818:825]*/, v[130:137] /*v[642:649]*/
	s_set_vgpr_msb 0xae46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x46c1
	v_mov_b64_e32 v[108:109] /*v[876:877]*/, v[28:29] /*v[284:285]*/
	v_mov_b64_e32 v[106:107] /*v[874:875]*/, v[26:27] /*v[282:283]*/
	s_set_vgpr_msb 0xc145
	v_cvt_pk_bf16_f32 v245 /*v501*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[146:147] /*v[658:659]*/
	s_set_vgpr_msb 0x4a8e
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[2:9] /*v[514:521]*/, v[66:73] /*v[834:841]*/, 0
	s_set_vgpr_msb 0x8e4d
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v217 /*v473*/, v10 /*v778*/
	v_nop
	s_set_vgpr_msb 0x4dc1
	v_mov_b64_e32 v[72:73] /*v[840:841]*/, v[152:153] /*v[408:409]*/
	s_set_vgpr_msb 0xc1ae
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[210:217] /*v[722:729]*/, v[82:89] /*v[850:857]*/, v[146:153] /*v[658:665]*/
	s_set_vgpr_msb 0xaec1
	v_mov_b64_e32 v[70:71] /*v[838:839]*/, v[150:151] /*v[406:407]*/
	s_set_vgpr_msb 0xc149
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x49c1
	v_mov_b64_e32 v[68:69] /*v[836:837]*/, v[148:149] /*v[404:405]*/
	s_set_vgpr_msb 0xc14d
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v223 /*v479*/, v10 /*v778*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4da2
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[226:233] /*v[738:745]*/, v[120:127], v[146:153] /*v[658:665]*/
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0xa24d
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v11 /*v779*/
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s5
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x4dc1
	v_mov_b64_e32 v[66:67] /*v[834:835]*/, v[146:147] /*v[402:403]*/
	s_set_vgpr_msb 0xc1a2
	s_wait_loadcnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[122:129] /*v[634:641]*/, v[0:7], v[146:153] /*v[658:665]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:256 nv
	scratch_load_b128 v[4:7], off, off offset:272 nv
	s_set_vgpr_msb 0x49
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v206 /*v462*/, v206 /*v462*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x490d
	v_cmp_gt_i32_e64 s2, v225 /*v481*/, v10 /*v778*/
	s_set_vgpr_msb 0xd45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[214:215] /*v[470:471]*/, v[198:199] /*v[454:455]*/
	s_clause 0xe
	scratch_load_b128 v[26:29] /*v[282:285]*/, off, off offset:2656 th:TH_LOAD_LU nv
	scratch_load_b128 v[30:33] /*v[286:289]*/, off, off offset:2672 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x45c0
	scratch_load_b128 v[210:213] /*v[978:981]*/, off, off offset:2400 th:TH_LOAD_LU nv
	scratch_load_b128 v[214:217] /*v[982:985]*/, off, off offset:2416 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[82:85] /*v[338:341]*/, off, off offset:2368 th:TH_LOAD_LU nv
	scratch_load_b128 v[86:89] /*v[342:345]*/, off, off offset:2384 th:TH_LOAD_LU nv
	scratch_load_b128 v[114:117] /*v[370:373]*/, off, off offset:2272 th:TH_LOAD_LU nv
	scratch_load_b128 v[118:121] /*v[374:377]*/, off, off offset:2288 th:TH_LOAD_LU nv
	scratch_load_b128 v[146:149] /*v[402:405]*/, off, off offset:2240 th:TH_LOAD_LU nv
	scratch_load_b128 v[150:153] /*v[406:409]*/, off, off offset:2256 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[204:207], off, off offset:2176 th:TH_LOAD_LU nv
	scratch_load_b128 v[208:211], off, off offset:2192 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x41
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x41ae
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[234:241] /*v[746:753]*/, v[42:49] /*v[810:817]*/, v[130:137] /*v[642:649]*/
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0xae46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x464d
	v_cndmask_b32_e64 v207 /*v463*/, v207 /*v463*/, 0xff61b1e6, s5
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	v_cmp_gt_i32_e32 vcc_lo, v224 /*v480*/, v11 /*v779*/
	s_set_vgpr_msb 0x4d45
	v_cvt_pk_bf16_f32 v250 /*v506*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454e
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[244:245] /*v[756:757]*/, v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4e49
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x4981
	v_mov_b64_e32 v[244:245] /*v[756:757]*/, v[68:69] /*v[324:325]*/
	s_set_vgpr_msb 0x8145
	v_cndmask_b32_e64 v208 /*v464*/, v208 /*v464*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[148:149] /*v[660:661]*/
	s_set_vgpr_msb 0x4a41
	v_cndmask_b32_e64 v209 /*v465*/, v209 /*v465*/, 0xff61b1e6, s5
	s_set_vgpr_msb 0x4146
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4641
	v_exp_f32_e32 v206 /*v462*/, v206 /*v462*/
	v_exp_f32_e32 v207 /*v463*/, v207 /*v463*/
	s_set_vgpr_msb 0x41a2
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[234:241] /*v[746:753]*/, v[128:135], v[82:89] /*v[594:601]*/
	s_set_vgpr_msb 0xa249
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v251 /*v507*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454e
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[246:247] /*v[758:759]*/, v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4e81
	v_mov_b64_e32 v[246:247] /*v[758:759]*/, v[70:71] /*v[326:327]*/
	s_set_vgpr_msb 0x814a
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[196:197] /*v[708:709]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x4a41
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4100
	s_clause 0x1
	scratch_load_b128 v[132:135], off, off offset:3008 th:TH_LOAD_LU nv
	scratch_load_b128 v[136:139], off, off offset:3024 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[206:207] /*v[462:463]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[150:151] /*v[662:663]*/
	s_set_vgpr_msb 0x4aac
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[184:191], v[178:185] /*v[946:953]*/, v[82:89] /*v[594:601]*/
	s_set_vgpr_msb 0xac41
	v_exp_f32_e32 v208 /*v464*/, v208 /*v464*/
	v_exp_f32_e32 v209 /*v465*/, v209 /*v465*/
	s_set_vgpr_msb 0x4146
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x46c0
	s_clause 0x6
	scratch_load_b128 v[178:181] /*v[946:949]*/, off, off offset:3264 th:TH_LOAD_LU nv
	scratch_load_b128 v[182:185] /*v[950:953]*/, off, off offset:3280 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[124:127], off, off offset:2944 th:TH_LOAD_LU nv
	scratch_load_b128 v[128:131], off, off offset:2960 th:TH_LOAD_LU nv
	scratch_load_b128 v[116:119], off, off offset:2880 th:TH_LOAD_LU nv
	scratch_load_b128 v[120:123], off, off offset:2896 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v252 /*v508*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454e
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[248:249] /*v[760:761]*/, v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4e81
	v_mov_b64_e32 v[248:249] /*v[760:761]*/, v[72:73] /*v[328:329]*/
	s_set_vgpr_msb 0x8142
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[114:115] /*v[626:627]*/
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[116:117] /*v[628:629]*/
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[118:119] /*v[630:631]*/
	s_set_vgpr_msb 0x4245
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[208:209] /*v[464:465]*/
	s_set_vgpr_msb 0x4542
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[120:121] /*v[632:633]*/
	s_set_vgpr_msb 0x4288
	v_add_nc_u32_e32 v114 /*v626*/, 16, v198 /*v710*/
	s_set_vgpr_msb 0x884a
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[152:153] /*v[664:665]*/
	s_set_vgpr_msb 0x4aae
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[234:241] /*v[746:753]*/, v[90:97] /*v[858:865]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xae46
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4688
	v_add_nc_u32_e32 v198 /*v710*/, 32, v198 /*v710*/
	s_set_vgpr_msb 0x8806
	v_cmp_ge_i32_e32 vcc_lo, v114 /*v626*/, v246 /*v502*/
	v_cmp_gt_i32_e64 s2, v114 /*v626*/, v246 /*v502*/
	s_set_vgpr_msb 0x688
	v_or_b32_e32 v115 /*v627*/, 3, v114 /*v626*/
	s_set_vgpr_msb 0x8845
	v_cvt_pk_bf16_f32 v253 /*v509*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[146:147] /*v[658:659]*/
	s_and_b32 s5, s40, vcc_lo
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x4a88
	v_or_b32_e32 v116 /*v628*/, 2, v114 /*v626*/
	s_set_vgpr_msb 0x8806
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v231 /*v487*/
	s_set_vgpr_msb 0x641
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4188
	v_or_b32_e32 v117 /*v629*/, 5, v114 /*v626*/
	s_set_vgpr_msb 0x8806
	v_cmp_gt_i32_e64 s2, v116 /*v628*/, v246 /*v502*/
	s_set_vgpr_msb 0x688
	v_or_b32_e32 v118 /*v630*/, 4, v114 /*v626*/
	v_or_b32_e32 v119 /*v631*/, 7, v114 /*v626*/
	s_set_vgpr_msb 0x884d
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4d88
	v_or_b32_e32 v120 /*v632*/, 6, v114 /*v626*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x88a4
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[184:191], v[42:49] /*v[298:305]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa441
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4106
	v_cmp_gt_i32_e32 vcc_lo, v117 /*v629*/, v231 /*v487*/
	s_set_vgpr_msb 0x641
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s5
	s_set_vgpr_msb 0x4106
	v_cmp_gt_i32_e64 s2, v118 /*v630*/, v246 /*v502*/
	s_set_vgpr_msb 0x641
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	s_set_vgpr_msb 0x41ae
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[234:241] /*v[746:753]*/, v[162:169] /*v[930:937]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xae4d
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_clause 0x1
	scratch_load_b128 v[42:45] /*v[298:301]*/, off, off offset:2528 th:TH_LOAD_LU nv
	scratch_load_b128 v[46:49] /*v[302:305]*/, off, off offset:2544 th:TH_LOAD_LU nv
	v_cndmask_b32_e64 v206 /*v462*/, v206 /*v462*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4d06
	v_cmp_gt_i32_e64 s2, v120 /*v632*/, v246 /*v502*/
	s_set_vgpr_msb 0x641
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4106
	v_cmp_gt_i32_e32 vcc_lo, v119 /*v631*/, v231 /*v487*/
	s_set_vgpr_msb 0x64d
	v_cndmask_b32_e64 v207 /*v463*/, v207 /*v463*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	v_cndmask_b32_e64 v208 /*v464*/, v208 /*v464*/, 0xff61b1e6, s2
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4d0e
	v_cmp_gt_i32_e64 s2, v114 /*v626*/, v252 /*v1020*/
	s_set_vgpr_msb 0xe46
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[82:83] /*v[594:595]*/, v[238:239] /*v[494:495]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x46a4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[184:191], v[10:17] /*v[266:273]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa441
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_ge_i32_e32 vcc_lo, v114 /*v626*/, v252 /*v1020*/
	s_set_vgpr_msb 0xe41
	v_cndmask_b32_e64 v209 /*v465*/, v209 /*v465*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v206 /*v462*/, v206 /*v462*/
	v_exp_f32_e32 v207 /*v463*/, v207 /*v463*/
	s_set_vgpr_msb 0x4146
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[50:51] /*v[562:563]*/, v[248:249] /*v[504:505]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x464d
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[74:75] /*v[842:843]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4dae
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[234:241] /*v[746:753]*/, v[22:29] /*v[790:797]*/, v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xae41
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v39 /*v807*/
	s_set_vgpr_msb 0xe41
	v_exp_f32_e32 v208 /*v464*/, v208 /*v464*/
	v_exp_f32_e32 v209 /*v465*/, v209 /*v465*/
	s_set_vgpr_msb 0x41a0
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[184:191], v[250:257], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x2
	scratch_load_b128 v[252:255], off, off offset:2496 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4a
	scratch_load_b128 v[0:3] /*v[256:259]*/, off, off offset:2512 th:TH_LOAD_LU nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[34:35] /*v[546:547]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa2
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[122:129] /*v[634:641]*/, v[212:219], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[212:215], off, off offset:3328 th:TH_LOAD_LU nv
	scratch_load_b128 v[216:219], off, off offset:3344 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x82
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[202:209] /*v[714:721]*/, v[64:71], 0
	s_set_vgpr_msb 0x8200
	s_clause 0x1
	scratch_load_b128 v[60:63], off, off offset:3104 th:TH_LOAD_LU nv
	scratch_load_b128 v[64:67], off, off offset:3120 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xa2
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[218:225] /*v[730:737]*/, v[88:95], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[84:87], off, off offset:3168 th:TH_LOAD_LU nv
	scratch_load_b128 v[88:91], off, off offset:3184 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[184:191], v[0:7], v[130:137] /*v[642:649]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:288 nv
	scratch_load_b128 v[4:7], off, off offset:304 nv
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x4a
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[130:131] /*v[642:643]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa2
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[234:241] /*v[746:753]*/, v[160:167], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[156:159], off, off offset:3360 th:TH_LOAD_LU nv
	scratch_load_b128 v[160:163], off, off offset:3376 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[214:215] /*v[470:471]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[132:133] /*v[644:645]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[214:215] /*v[470:471]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x45a0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[184:191], v[192:199], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa045
	v_cvt_pk_bf16_f32 v198 /*v454*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[200:201] /*v[456:457]*/
	v_nop
	v_nop
	s_set_vgpr_msb 0x468e
	v_pk_add_f32 v[2:3] /*v[514:515]*/, v[2:3] /*v[514:515]*/, v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e45
	v_cvt_pk_bf16_f32 v199 /*v455*/, v200 /*v456*/, v201 /*v457*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[134:135] /*v[646:647]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa2
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[122:129] /*v[634:641]*/, v[0:7], v[106:113] /*v[618:625]*/
	s_set_vgpr_msb 0xa200
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:320 nv
	scratch_load_b128 v[4:7], off, off offset:336 nv
	s_set_vgpr_msb 0x45
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[206:207] /*v[462:463]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[136:137] /*v[648:649]*/, v[176:177] /*v[688:689]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[208:209] /*v[464:465]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[108:109] /*v[620:621]*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[196:197] /*v[708:709]*/, v[110:111] /*v[622:623]*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[196:197] /*v[708:709]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v200 /*v456*/, v200 /*v456*/, v201 /*v457*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[206:207] /*v[462:463]*/
	s_set_vgpr_msb 0x46a0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[184:191], v[0:7], v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xa045
	v_cvt_pk_bf16_f32 v201 /*v457*/, v206 /*v462*/, v207 /*v463*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[106:107] /*v[618:619]*/
	s_set_vgpr_msb 0x4a00
	s_clause 0x2
	scratch_load_b32 v0, off, off offset:3440 nv
	scratch_load_b128 v[188:191], off, off offset:3200 th:TH_LOAD_LU nv
	scratch_load_b128 v[192:195], off, off offset:3216 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x41
	v_cndmask_b32_e64 v207 /*v463*/, v207 /*v463*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v206 /*v462*/, v206 /*v462*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v116 /*v628*/, v252 /*v1020*/
	s_set_vgpr_msb 0xe4a
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[98:99] /*v[610:611]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a49
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v208 /*v464*/, v208 /*v464*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v118 /*v630*/, v252 /*v1020*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v117 /*v629*/, v39 /*v807*/
	s_set_vgpr_msb 0xe49
	v_cndmask_b32_e64 v209 /*v465*/, v209 /*v465*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v206 /*v462*/, v206 /*v462*/
	v_cndmask_b32_e64 v214 /*v470*/, v214 /*v470*/, 0xff61b1e6, s2
	v_exp_f32_e32 v207 /*v463*/, v207 /*v463*/
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v120 /*v632*/, v252 /*v1020*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v119 /*v631*/, v39 /*v807*/
	s_set_vgpr_msb 0xe45
	v_cndmask_b32_e64 v215 /*v471*/, v215 /*v471*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v208 /*v464*/, v208 /*v464*/
	v_exp_f32_e32 v209 /*v465*/, v209 /*v465*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[222:223] /*v[478:479]*/, v[206:207] /*v[462:463]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, v[162:163] /*v[674:675]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[100:101] /*v[612:613]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a41
	v_cndmask_b32_e64 v216 /*v472*/, v216 /*v472*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v114 /*v626*/, v40 /*v808*/
	s_set_vgpr_msb 0xe46
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[206:207] /*v[462:463]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[222:223] /*v[478:479]*/, v[208:209] /*v[464:465]*/
	v_cndmask_b32_e64 v217 /*v473*/, v217 /*v473*/, 0xff61b1e6, s5
	v_cvt_pk_bf16_f32 v206 /*v462*/, v206 /*v462*/, v207 /*v463*/
	v_exp_f32_e32 v214 /*v470*/, v214 /*v470*/
	v_exp_f32_e32 v215 /*v471*/, v215 /*v471*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[208:209] /*v[464:465]*/
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[162:163] /*v[674:675]*/, v[216:217] /*v[472:473]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x460e
	v_cmp_ge_i32_e32 vcc_lo, v114 /*v626*/, v40 /*v808*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0xe4a
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[196:197] /*v[708:709]*/, v[94:95] /*v[606:607]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v207 /*v463*/, v208 /*v464*/, v209 /*v465*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[102:103] /*v[614:615]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4a0e
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v41 /*v809*/
	s_set_vgpr_msb 0xe45
	v_exp_f32_e32 v216 /*v472*/, v216 /*v472*/
	v_exp_f32_e32 v217 /*v473*/, v217 /*v473*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[104:105] /*v[616:617]*/, v[178:179] /*v[690:691]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[208:209] /*v[464:465]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, v[216:217] /*v[472:473]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[196:197] /*v[708:709]*/, v[92:93] /*v[604:605]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v208 /*v464*/, v208 /*v464*/, v209 /*v465*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[196:197] /*v[708:709]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v209 /*v465*/, v214 /*v470*/, v215 /*v471*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[196:197] /*v[708:709]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x4a41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v215 /*v471*/, v215 /*v471*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v214 /*v470*/, v214 /*v470*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v116 /*v628*/, v40 /*v808*/
	s_set_vgpr_msb 0xe49
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v216 /*v472*/, v216 /*v472*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v118 /*v630*/, v40 /*v808*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v117 /*v629*/, v41 /*v809*/
	s_set_vgpr_msb 0xe49
	v_cndmask_b32_e64 v217 /*v473*/, v217 /*v473*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v214 /*v470*/, v214 /*v470*/
	v_cndmask_b32_e64 v222 /*v478*/, v222 /*v478*/, 0xff61b1e6, s2
	v_exp_f32_e32 v215 /*v471*/, v215 /*v471*/
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v120 /*v632*/, v40 /*v808*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v119 /*v631*/, v41 /*v809*/
	s_set_vgpr_msb 0xe45
	v_cndmask_b32_e64 v223 /*v479*/, v223 /*v479*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v216 /*v472*/, v216 /*v472*/
	v_exp_f32_e32 v217 /*v473*/, v217 /*v473*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[230:231] /*v[486:487]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[164:165] /*v[676:677]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[238:239] /*v[494:495]*/, v[84:85] /*v[596:597]*/ neg_lo:[1,0] neg_hi:[1,0]
	v_cndmask_b32_e64 v224 /*v480*/, v224 /*v480*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v114 /*v626*/, v38 /*v806*/
	s_set_vgpr_msb 0xe46
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[196:197] /*v[708:709]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[230:231] /*v[486:487]*/, v[216:217] /*v[472:473]*/
	v_cndmask_b32_e64 v225 /*v481*/, v225 /*v481*/, 0xff61b1e6, s5
	v_cvt_pk_bf16_f32 v214 /*v470*/, v214 /*v470*/, v215 /*v471*/
	v_exp_f32_e32 v222 /*v478*/, v222 /*v478*/
	v_exp_f32_e32 v223 /*v479*/, v223 /*v479*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[196:197] /*v[708:709]*/, v[216:217] /*v[472:473]*/
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[164:165] /*v[676:677]*/, v[224:225] /*v[480:481]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x460e
	v_cmp_ge_i32_e32 vcc_lo, v114 /*v626*/, v38 /*v806*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0xe4a
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[196:197] /*v[708:709]*/, v[78:79] /*v[590:591]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v215 /*v471*/, v216 /*v472*/, v217 /*v473*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[86:87] /*v[598:599]*/, v[238:239] /*v[494:495]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v233 /*v489*/
	s_set_vgpr_msb 0x4645
	v_exp_f32_e32 v224 /*v480*/, v224 /*v480*/
	v_exp_f32_e32 v225 /*v481*/, v225 /*v481*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[88:89] /*v[600:601]*/, v[238:239] /*v[494:495]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[66:67] /*v[578:579]*/, v[240:241] /*v[496:497]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_3)
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[196:197] /*v[708:709]*/, v[216:217] /*v[472:473]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_3)
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[224:225] /*v[480:481]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[196:197] /*v[708:709]*/, v[76:77] /*v[588:589]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v216 /*v472*/, v216 /*v472*/, v217 /*v473*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[196:197] /*v[708:709]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v217 /*v473*/, v222 /*v478*/, v223 /*v479*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[196:197] /*v[708:709]*/, v[74:75] /*v[586:587]*/
	s_set_vgpr_msb 0x4a41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v223 /*v479*/, v223 /*v479*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v222 /*v478*/, v222 /*v478*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v116 /*v628*/, v38 /*v806*/
	s_set_vgpr_msb 0xe49
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v224 /*v480*/, v224 /*v480*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v118 /*v630*/, v38 /*v806*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4106
	v_cmp_gt_i32_e32 vcc_lo, v117 /*v629*/, v233 /*v489*/
	s_set_vgpr_msb 0x649
	v_cndmask_b32_e64 v225 /*v481*/, v225 /*v481*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v222 /*v478*/, v222 /*v478*/
	v_cndmask_b32_e64 v230 /*v486*/, v230 /*v486*/, 0xff61b1e6, s2
	v_exp_f32_e32 v223 /*v479*/, v223 /*v479*/
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v120 /*v632*/, v38 /*v806*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4106
	v_cmp_gt_i32_e32 vcc_lo, v119 /*v631*/, v233 /*v489*/
	s_set_vgpr_msb 0x641
	v_cndmask_b32_e64 v231 /*v487*/, v231 /*v487*/, 0xff61b1e6, s5
	s_set_vgpr_msb 0x414a
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[196:197] /*v[708:709]*/, v[80:81] /*v[592:593]*/
	s_set_vgpr_msb 0x4a49
	v_exp_f32_e32 v224 /*v480*/, v224 /*v480*/
	v_exp_f32_e32 v225 /*v481*/, v225 /*v481*/
	s_and_b32 s2, s40, s2
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, v[166:167] /*v[678:679]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[238:239] /*v[494:495]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[68:69] /*v[580:581]*/, v[240:241] /*v[496:497]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4641
	v_cndmask_b32_e64 v232 /*v488*/, v232 /*v488*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v114 /*v626*/, v78 /*v846*/
	s_set_vgpr_msb 0xe45
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[238:239] /*v[494:495]*/, v[224:225] /*v[480:481]*/
	v_cndmask_b32_e64 v233 /*v489*/, v233 /*v489*/, 0xff61b1e6, s5
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[196:197] /*v[708:709]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x4641
	v_exp_f32_e32 v230 /*v486*/, v230 /*v486*/
	v_exp_f32_e32 v231 /*v487*/, v231 /*v487*/
	s_set_vgpr_msb 0x4146
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[196:197] /*v[708:709]*/, v[224:225] /*v[480:481]*/
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[166:167] /*v[678:679]*/, v[232:233] /*v[488:489]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x4645
	v_cvt_pk_bf16_f32 v222 /*v478*/, v222 /*v478*/, v223 /*v479*/
	s_set_vgpr_msb 0x450e
	v_cmp_ge_i32_e32 vcc_lo, v114 /*v626*/, v78 /*v846*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0xe45
	v_cvt_pk_bf16_f32 v223 /*v479*/, v224 /*v480*/, v225 /*v481*/
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[232:233] /*v[488:489]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[70:71] /*v[582:583]*/, v[240:241] /*v[496:497]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x460e
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v79 /*v847*/
	s_set_vgpr_msb 0xe4a
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[196:197] /*v[708:709]*/, v[62:63] /*v[574:575]*/
	s_set_vgpr_msb 0x4a45
	v_exp_f32_e32 v232 /*v488*/, v232 /*v488*/
	v_exp_f32_e32 v233 /*v489*/, v233 /*v489*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[230:231] /*v[486:487]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[72:73] /*v[584:585]*/, v[240:241] /*v[496:497]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x464a
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[196:197] /*v[708:709]*/, v[64:65] /*v[576:577]*/
	s_set_vgpr_msb 0x4a46
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[196:197] /*v[708:709]*/, v[224:225] /*v[480:481]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, v[232:233] /*v[488:489]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[196:197] /*v[708:709]*/, v[60:61] /*v[572:573]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v224 /*v480*/, v224 /*v480*/, v225 /*v481*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[196:197] /*v[708:709]*/, v[230:231] /*v[486:487]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v225 /*v481*/, v230 /*v486*/, v231 /*v487*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[196:197] /*v[708:709]*/, v[58:59] /*v[570:571]*/
	s_set_vgpr_msb 0x4a41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v231 /*v487*/, v231 /*v487*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v230 /*v486*/, v230 /*v486*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v116 /*v628*/, v78 /*v846*/
	s_set_vgpr_msb 0xe49
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v232 /*v488*/, v232 /*v488*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v118 /*v630*/, v78 /*v846*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v117 /*v629*/, v79 /*v847*/
	s_set_vgpr_msb 0xe49
	v_cndmask_b32_e64 v233 /*v489*/, v233 /*v489*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v230 /*v486*/, v230 /*v486*/
	v_cndmask_b32_e64 v238 /*v494*/, v238 /*v494*/, 0xff61b1e6, s2
	v_exp_f32_e32 v231 /*v487*/, v231 /*v487*/
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[232:233] /*v[488:489]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v120 /*v632*/, v78 /*v846*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[232:233] /*v[488:489]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v119 /*v631*/, v79 /*v847*/
	s_set_vgpr_msb 0xe45
	v_cndmask_b32_e64 v239 /*v495*/, v239 /*v495*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v232 /*v488*/, v232 /*v488*/
	v_exp_f32_e32 v233 /*v489*/, v233 /*v489*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[246:247] /*v[502:503]*/, v[230:231] /*v[486:487]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, v[168:169] /*v[680:681]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[248:249] /*v[504:505]*/, v[52:53] /*v[564:565]*/ neg_lo:[1,0] neg_hi:[1,0]
	v_cndmask_b32_e64 v240 /*v496*/, v240 /*v496*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v114 /*v626*/, v254 /*v1022*/
	s_set_vgpr_msb 0xe46
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[196:197] /*v[708:709]*/, v[230:231] /*v[486:487]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[246:247] /*v[502:503]*/, v[232:233] /*v[488:489]*/
	v_cndmask_b32_e64 v241 /*v497*/, v241 /*v497*/, 0xff61b1e6, s5
	v_cvt_pk_bf16_f32 v230 /*v486*/, v230 /*v486*/, v231 /*v487*/
	v_exp_f32_e32 v238 /*v494*/, v238 /*v494*/
	v_exp_f32_e32 v239 /*v495*/, v239 /*v495*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[196:197] /*v[708:709]*/, v[232:233] /*v[488:489]*/
	v_pk_add_f32 v[240:241] /*v[496:497]*/, v[168:169] /*v[680:681]*/, v[240:241] /*v[496:497]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x460e
	v_cmp_ge_i32_e32 vcc_lo, v114 /*v626*/, v254 /*v1022*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0xe4a
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[196:197] /*v[708:709]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v231 /*v487*/, v232 /*v488*/, v233 /*v489*/
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[240:241] /*v[496:497]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[54:55] /*v[566:567]*/, v[248:249] /*v[504:505]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x460e
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v77 /*v845*/
	s_set_vgpr_msb 0xe45
	v_exp_f32_e32 v240 /*v496*/, v240 /*v496*/
	v_exp_f32_e32 v241 /*v497*/, v241 /*v497*/
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[232:233] /*v[488:489]*/, v[238:239] /*v[494:495]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[56:57] /*v[568:569]*/, v[248:249] /*v[504:505]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x464a
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[196:197] /*v[708:709]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0x4a46
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[196:197] /*v[708:709]*/, v[232:233] /*v[488:489]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, v[240:241] /*v[496:497]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[196:197] /*v[708:709]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v232 /*v488*/, v232 /*v488*/, v233 /*v489*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[196:197] /*v[708:709]*/, v[238:239] /*v[494:495]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v233 /*v489*/, v238 /*v494*/, v239 /*v495*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[196:197] /*v[708:709]*/, v[42:43] /*v[554:555]*/
	s_set_vgpr_msb 0x4a41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v239 /*v495*/, v239 /*v495*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v238 /*v494*/, v238 /*v494*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v116 /*v628*/, v254 /*v1022*/
	s_set_vgpr_msb 0xe49
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v240 /*v496*/, v240 /*v496*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v118 /*v630*/, v254 /*v1022*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v117 /*v629*/, v77 /*v845*/
	s_set_vgpr_msb 0xe49
	v_cndmask_b32_e64 v241 /*v497*/, v241 /*v497*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v238 /*v494*/, v238 /*v494*/
	v_cndmask_b32_e64 v246 /*v502*/, v246 /*v502*/, 0xff61b1e6, s2
	v_exp_f32_e32 v239 /*v495*/, v239 /*v495*/
	v_pk_add_f32 v[240:241] /*v[496:497]*/, v[240:241] /*v[496:497]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v120 /*v632*/, v254 /*v1022*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[240:241] /*v[496:497]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v119 /*v631*/, v77 /*v845*/
	s_set_vgpr_msb 0xe45
	v_cndmask_b32_e64 v247 /*v503*/, v247 /*v503*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v240 /*v496*/, v240 /*v496*/
	v_exp_f32_e32 v241 /*v497*/, v241 /*v497*/
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[254:255] /*v[510:511]*/, v[238:239] /*v[494:495]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[246:247] /*v[502:503]*/, v[170:171] /*v[682:683]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[36:37] /*v[548:549]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a41
	v_cndmask_b32_e64 v248 /*v504*/, v248 /*v504*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v114 /*v626*/, v76 /*v844*/
	s_set_vgpr_msb 0xe46
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[196:197] /*v[708:709]*/, v[238:239] /*v[494:495]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[246:247] /*v[502:503]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[254:255] /*v[510:511]*/, v[240:241] /*v[496:497]*/
	v_cndmask_b32_e64 v249 /*v505*/, v249 /*v505*/, 0xff61b1e6, s5
	v_cvt_pk_bf16_f32 v238 /*v494*/, v238 /*v494*/, v239 /*v495*/
	v_exp_f32_e32 v246 /*v502*/, v246 /*v502*/
	v_exp_f32_e32 v247 /*v503*/, v247 /*v503*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[196:197] /*v[708:709]*/, v[240:241] /*v[496:497]*/
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[170:171] /*v[682:683]*/, v[248:249] /*v[504:505]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x460e
	v_cmp_ge_i32_e32 vcc_lo, v114 /*v626*/, v76 /*v844*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0xe4a
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[196:197] /*v[708:709]*/, v[30:31] /*v[542:543]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v239 /*v495*/, v240 /*v496*/, v241 /*v497*/
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[248:249] /*v[504:505]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[240:241] /*v[496:497]*/, v[38:39] /*v[550:551]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4a0e
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v253 /*v1021*/
	s_set_vgpr_msb 0xe45
	v_exp_f32_e32 v248 /*v504*/, v248 /*v504*/
	v_exp_f32_e32 v249 /*v505*/, v249 /*v505*/
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[240:241] /*v[496:497]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[40:41] /*v[552:553]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a8a
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[196:197] /*v[708:709]*/, v[32:33] /*v[544:545]*/
	s_set_vgpr_msb 0x8a46
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[196:197] /*v[708:709]*/, v[240:241] /*v[496:497]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[246:247] /*v[502:503]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[196:197] /*v[708:709]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v240 /*v496*/, v240 /*v496*/, v241 /*v497*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[196:197] /*v[708:709]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v241 /*v497*/, v246 /*v502*/, v247 /*v503*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[196:197] /*v[708:709]*/, v[26:27] /*v[538:539]*/
	s_set_vgpr_msb 0x4a41
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v247 /*v503*/, v247 /*v503*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v246 /*v502*/, v246 /*v502*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v116 /*v628*/, v76 /*v844*/
	s_set_vgpr_msb 0xe49
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[246:247] /*v[502:503]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v248 /*v504*/, v248 /*v504*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v118 /*v630*/, v76 /*v844*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[246:247] /*v[502:503]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v117 /*v629*/, v253 /*v1021*/
	s_set_vgpr_msb 0xe49
	v_cndmask_b32_e64 v249 /*v505*/, v249 /*v505*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v246 /*v502*/, v246 /*v502*/
	v_cndmask_b32_e64 v254 /*v510*/, v254 /*v510*/, 0xff61b1e6, s2
	v_exp_f32_e32 v247 /*v503*/, v247 /*v503*/
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[248:249] /*v[504:505]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x490e
	v_cmp_gt_i32_e64 s2, v120 /*v632*/, v76 /*v844*/
	s_set_vgpr_msb 0xe41
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[248:249] /*v[504:505]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e32 vcc_lo, v119 /*v631*/, v253 /*v1021*/
	s_set_vgpr_msb 0xe41
	v_cndmask_b32_e64 v255 /*v511*/, v255 /*v511*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	v_exp_f32_e32 v248 /*v504*/, v248 /*v504*/
	v_exp_f32_e32 v249 /*v505*/, v249 /*v505*/
	s_set_vgpr_msb 0x4146
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[18:19] /*v[530:531]*/, v[246:247] /*v[502:503]*/
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[172:173] /*v[684:685]*/, v[254:255] /*v[510:511]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x468e
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[20:21] /*v[532:533]*/, v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v0 /*v512*/, v0 /*v512*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v114 /*v626*/, v10 /*v778*/
	s_set_vgpr_msb 0x8e46
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[196:197] /*v[708:709]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x4641
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x4146
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[18:19] /*v[530:531]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x4682
	v_cndmask_b32_e64 v1 /*v513*/, v1 /*v513*/, 0xff61b1e6, s5
	s_set_vgpr_msb 0x8245
	v_cvt_pk_bf16_f32 v246 /*v502*/, v246 /*v502*/, v247 /*v503*/
	v_exp_f32_e32 v254 /*v510*/, v254 /*v510*/
	v_exp_f32_e32 v255 /*v511*/, v255 /*v511*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[196:197] /*v[708:709]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x468a
	v_pk_add_f32 v[0:1] /*v[512:513]*/, v[0:1] /*v[512:513]*/, v[172:173] /*v[684:685]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a0e
	v_cmp_ge_i32_e32 vcc_lo, v114 /*v626*/, v10 /*v778*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0xe45
	v_cvt_pk_bf16_f32 v247 /*v503*/, v248 /*v504*/, v249 /*v505*/
	s_set_vgpr_msb 0x4582
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[0:1] /*v[512:513]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x824e
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[22:23] /*v[534:535]*/, v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v11 /*v779*/
	s_set_vgpr_msb 0x4e82
	v_exp_f32_e32 v0 /*v512*/, v0 /*v512*/
	v_exp_f32_e32 v1 /*v513*/, v1 /*v513*/
	s_set_vgpr_msb 0x8245
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[248:249] /*v[504:505]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x454e
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[24:25] /*v[536:537]*/, v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4e46
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[196:197] /*v[708:709]*/, v[248:249] /*v[504:505]*/
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[0:1] /*v[512:513]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x468a
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[196:197] /*v[708:709]*/, v[12:13] /*v[524:525]*/
	s_set_vgpr_msb 0x8a45
	v_cvt_pk_bf16_f32 v248 /*v504*/, v248 /*v504*/, v249 /*v505*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[196:197] /*v[708:709]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x4645
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v249 /*v505*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[196:197] /*v[708:709]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[196:197] /*v[708:709]*/, v[14:15] /*v[526:527]*/
	s_set_vgpr_msb 0x8a41
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v255 /*v511*/, v255 /*v511*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v254 /*v510*/, v254 /*v510*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x410e
	v_cmp_gt_i32_e64 s2, v116 /*v628*/, v10 /*v778*/
	s_set_vgpr_msb 0xe49
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x498e
	v_cndmask_b32_e64 v0 /*v512*/, v0 /*v512*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v118 /*v630*/, v10 /*v778*/
	s_set_vgpr_msb 0x8e41
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	s_set_vgpr_msb 0x418e
	v_cmp_gt_i32_e32 vcc_lo, v117 /*v629*/, v11 /*v779*/
	v_cndmask_b32_e64 v1 /*v513*/, v1 /*v513*/, 0xff61b1e6, s5
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x8e41
	v_exp_f32_e32 v254 /*v510*/, v254 /*v510*/
	s_set_vgpr_msb 0x418e
	v_cndmask_b32_e64 v10 /*v522*/, v10 /*v522*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v120 /*v632*/, v10 /*v778*/
	s_set_vgpr_msb 0x8e8a
	v_pk_add_f32 v[0:1] /*v[512:513]*/, v[0:1] /*v[512:513]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v255 /*v511*/, v255 /*v511*/
	s_and_b32 s2, s40, s2
	s_set_vgpr_msb 0x418e
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[0:1] /*v[512:513]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v119 /*v631*/, v11 /*v779*/
	v_cndmask_b32_e64 v11 /*v523*/, v11 /*v523*/, 0xff61b1e6, s5
	s_set_vgpr_msb 0x8e46
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[2:3] /*v[514:515]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x468e
	v_exp_f32_e32 v0 /*v512*/, v0 /*v512*/
	v_exp_f32_e32 v1 /*v513*/, v1 /*v513*/
	v_pk_add_f32 v[2:3] /*v[514:515]*/, v[4:5] /*v[516:517]*/, v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e8a
	v_pk_add_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a46
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[196:197] /*v[708:709]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x468a
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, s[4:5] op_sel_hi:[1,0]
	s_and_b32 s5, s40, vcc_lo
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[2:3] /*v[514:515]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x8a45
	v_cvt_pk_bf16_f32 v254 /*v510*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x458e
	v_pk_add_f32 v[2:3] /*v[514:515]*/, v[8:9] /*v[520:521]*/, v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v12 /*v524*/, v10 /*v522*/
	v_exp_f32_e32 v13 /*v525*/, v11 /*v523*/
	v_nop
	s_set_vgpr_msb 0x8e8a
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[196:197] /*v[708:709]*/, v[16:17] /*v[528:529]*/
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[196:197] /*v[708:709]*/, v[0:1] /*v[512:513]*/
	s_cmp_lg_u64 s[20:21], 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v11 /*v523*/, v11 /*v523*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v10 /*v522*/, v10 /*v522*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x8a4a
	v_cvt_pk_bf16_f32 v255 /*v511*/, v0 /*v512*/, v1 /*v513*/
	s_set_vgpr_msb 0x4a8e
	v_pk_add_f32 v[0:1] /*v[512:513]*/, v[6:7] /*v[518:519]*/, v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e8a
	v_pk_add_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, v[174:175] /*v[686:687]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[0:1] /*v[512:513]*/, v[12:13] /*v[524:525]*/
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, s[4:5] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[196:197] /*v[708:709]*/, v[0:1] /*v[512:513]*/
	v_exp_f32_e32 v10 /*v522*/, v10 /*v522*/
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v11 /*v523*/, v11 /*v523*/
	v_cvt_pk_bf16_f32 v0 /*v512*/, v0 /*v512*/, v1 /*v513*/
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[2:3] /*v[514:515]*/, v[10:11] /*v[522:523]*/
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[196:197] /*v[708:709]*/, v[2:3] /*v[514:515]*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v1 /*v513*/, v2 /*v514*/, v3 /*v515*/
	s_set_vgpr_msb 0x8a80
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v0
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v0 offset:4352
	s_set_vgpr_msb 0x8000
	scratch_load_b32 v0, off, off offset:3444 nv
	s_set_vgpr_msb 0xf9
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[986:993]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[218:225] /*v[986:993]*/
	s_set_vgpr_msb 0xf959
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[146:153] /*v[402:409]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[26:33] /*v[282:289]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[26:27] /*v[282:283]*/
	s_set_vgpr_msb 0x5909
	v_wmma_f32_16x16x32_bf16 v[236:243], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[236:243]
	s_set_vgpr_msb 0x941
	v_mov_b64_e32 v[22:23] /*v[278:279]*/, v[28:29] /*v[284:285]*/
	v_mov_b64_e32 v[24:25] /*v[280:281]*/, v[30:31] /*v[286:287]*/
	v_mov_b64_e32 v[26:27] /*v[282:283]*/, v[32:33] /*v[288:289]*/
	v_nop
	s_set_vgpr_msb 0x4100
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mov_b64_e32 v[250:251], v[242:243]
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5900
	v_mov_b64_e32 v[248:249], v[240:241]
	v_mov_b64_e32 v[246:247], v[238:239]
	v_mov_b64_e32 v[244:245], v[236:237]
	v_nop
	s_set_vgpr_msb 1
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mov_b64_e32 v[172:173], v[66:67] /*v[322:323]*/
	s_set_vgpr_msb 0x1a9
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa901
	v_mov_b64_e32 v[174:175], v[68:69] /*v[324:325]*/
	v_mov_b64_e32 v[176:177], v[70:71] /*v[326:327]*/
	v_mov_b64_e32 v[178:179], v[72:73] /*v[328:329]*/
	s_set_vgpr_msb 0x140
	s_clause 0x1
	scratch_load_b128 v[66:69] /*v[322:325]*/, off, off offset:2720 th:TH_LOAD_LU nv
	scratch_load_b128 v[70:73] /*v[326:329]*/, off, off offset:2736 th:TH_LOAD_LU nv
	v_nop
	s_set_vgpr_msb 0x4002
	v_mov_b64_e32 v[108:109], v[242:243] /*v[754:755]*/
	s_set_vgpr_msb 0x2a9
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa980
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v0
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v0 offset:4352
	s_set_vgpr_msb 0x8000
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:160 nv
	scratch_load_b128 v[4:7], off, off offset:176 nv
	s_set_vgpr_msb 0x59
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5902
	v_mov_b64_e32 v[110:111], v[244:245] /*v[756:757]*/
	v_mov_b64_e32 v[112:113], v[246:247] /*v[758:759]*/
	v_mov_b64_e32 v[114:115], v[248:249] /*v[760:761]*/
	s_set_vgpr_msb 0x2c3
	v_mov_b64_e32 v[176:177] /*v[944:945]*/, v[0:1] /*v[768:769]*/
	s_set_vgpr_msb 0xc3c2
	v_mov_b64_e32 v[174:175] /*v[942:943]*/, v[254:255] /*v[766:767]*/
	v_mov_b64_e32 v[172:173] /*v[940:941]*/, v[252:253] /*v[764:765]*/
	v_mov_b64_e32 v[170:171] /*v[938:939]*/, v[250:251] /*v[762:763]*/
	s_set_vgpr_msb 0xc259
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x59f9
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[1002:1009]*/, v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[234:241] /*v[1002:1009]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[834:841]*/, v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[66:73] /*v[834:841]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xf903
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mov_b64_e32 v[164:165], v[66:67] /*v[834:835]*/
	s_set_vgpr_msb 0x3f9
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[874:881]*/, v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[106:113] /*v[874:881]*/
	s_set_vgpr_msb 0xf903
	v_mov_b64_e32 v[166:167], v[68:69] /*v[836:837]*/
	v_mov_b64_e32 v[168:169], v[70:71] /*v[838:839]*/
	v_mov_b64_e32 v[170:171], v[72:73] /*v[840:841]*/
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_mov_b64_e32 v[100:101], v[106:107] /*v[874:875]*/
	v_mov_b64_e32 v[102:103], v[108:109] /*v[876:877]*/
	s_set_vgpr_msb 0x309
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x903
	s_clause 0x2
	scratch_store_b128 off, v[0:3], off offset:160 nv
	scratch_store_b128 off, v[4:7], off offset:176 nv
	scratch_load_b32 v0, off, off offset:3448 nv
	v_mov_b64_e32 v[104:105], v[110:111] /*v[878:879]*/
	v_mov_b64_e32 v[106:107], v[112:113] /*v[880:881]*/
	s_set_vgpr_msb 0x380
	s_wait_loadcnt 0x0
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v0
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v0 offset:4352
	s_set_vgpr_msb 0x8000
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:128 nv
	scratch_load_b128 v[4:7], off, off offset:144 nv
	s_set_vgpr_msb 0xf9
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[970:977]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[202:209] /*v[970:977]*/
	s_set_vgpr_msb 0xf959
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5909
	v_wmma_f32_16x16x32_bf16 v[228:235], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[228:235]
	v_wmma_f32_16x16x32_bf16 v[156:163], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[156:163]
	v_wmma_f32_16x16x32_bf16 v[84:91], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[84:91]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x2
	scratch_store_b128 off, v[0:3], off offset:128 nv
	scratch_store_b128 off, v[4:7], off offset:144 nv
	scratch_load_b32 v0, off, off offset:3452 nv
	s_set_vgpr_msb 0x80
	s_wait_loadcnt 0x0
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v0
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v0 offset:4352
	s_set_vgpr_msb 0x8000
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:96 nv
	scratch_load_b128 v[4:7], off, off offset:112 nv
	s_set_vgpr_msb 0x59
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[138:145] /*v[394:401]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5909
	v_wmma_f32_16x16x32_bf16 v[48:55], v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[48:55]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x900
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mov_b64_e32 v[202:203], v[54:55]
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[212:219], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[212:219]
	s_set_vgpr_msb 0x900
	v_mov_b64_e32 v[200:201], v[52:53]
	v_mov_b64_e32 v[198:199], v[50:51]
	v_mov_b64_e32 v[196:197], v[48:49]
	s_clause 0x3
	scratch_load_b128 v[52:55], off, off offset:2976 th:TH_LOAD_LU nv
	scratch_load_b128 v[56:59], off, off offset:2992 th:TH_LOAD_LU nv
	scratch_load_b128 v[44:47], off, off offset:2912 th:TH_LOAD_LU nv
	scratch_load_b128 v[48:51], off, off offset:2928 th:TH_LOAD_LU nv
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[148:155], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[148:155]
	v_wmma_f32_16x16x32_bf16 v[60:67], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[60:67]
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x2
	scratch_store_b128 off, v[0:3], off offset:96 nv
	scratch_store_b128 off, v[4:7], off offset:112 nv
	scratch_load_b32 v0, off, off offset:3456 nv
	s_set_vgpr_msb 0x80
	s_wait_loadcnt 0x0
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v0
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v0 offset:4352
	s_set_vgpr_msb 0x8009
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:64 nv
	scratch_load_b128 v[4:7], off, off offset:80 nv
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[204:211], v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[204:211]
	s_set_vgpr_msb 0x959
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x59f9
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[978:985]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[210:217] /*v[978:985]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[914:921]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[146:153] /*v[914:921]*/
	s_set_vgpr_msb 0xf909
	v_wmma_f32_16x16x32_bf16 v[28:35], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[28:35]
	v_wmma_f32_16x16x32_bf16 v[140:147], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[140:147]
	v_wmma_f32_16x16x32_bf16 v[76:83], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[76:83]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x2
	scratch_store_b128 off, v[0:3], off offset:64 nv
	scratch_store_b128 off, v[4:7], off offset:80 nv
	scratch_load_b32 v0, off, off offset:3460 nv
	s_set_vgpr_msb 0x80
	s_wait_loadcnt 0x0
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v0
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v0 offset:4352
	s_set_vgpr_msb 0x8000
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off offset:32 nv
	scratch_load_b128 v[4:7], off, off offset:48 nv
	s_set_vgpr_msb 0xa9
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[650:657]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[138:145] /*v[650:657]*/
	s_set_vgpr_msb 0xa959
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[58:65] /*v[314:321]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[666:673]*/, v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[154:161] /*v[666:673]*/
	s_set_vgpr_msb 0xa909
	v_wmma_f32_16x16x32_bf16 v[132:139], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[132:139]
	v_wmma_f32_16x16x32_bf16 v[52:59], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[52:59]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x2
	scratch_store_b128 off, v[0:3], off offset:32 nv
	scratch_store_b128 off, v[4:7], off offset:48 nv
	scratch_load_b32 v0, off, off offset:3464 nv
	s_set_vgpr_msb 0x80
	s_wait_loadcnt 0x0
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v0
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v0 offset:4352
	s_set_vgpr_msb 0x8000
	s_clause 0x1
	scratch_load_b128 v[0:3], off, off nv
	scratch_load_b128 v[4:7], off, off offset:16 nv
	s_set_vgpr_msb 0x59
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x59f9
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[906:913]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[138:145] /*v[906:913]*/
	s_set_vgpr_msb 0xf959
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x59f9
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[960:967]*/, v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[192:199] /*v[960:967]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[946:953]*/, v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[178:185] /*v[946:953]*/
	s_set_vgpr_msb 0xf909
	v_wmma_f32_16x16x32_bf16 v[124:131], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[124:131]
	v_wmma_f32_16x16x32_bf16 v[44:51], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[44:51]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x3
	scratch_store_b128 off, v[0:3], off nv
	scratch_store_b128 off, v[4:7], off offset:16 nv
	scratch_load_b128 v[0:3], off, off offset:192 nv
	scratch_load_b128 v[4:7], off, off offset:208 nv
	s_set_vgpr_msb 0x83
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v13 /*v781*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v13 /*v781*/ offset:4352
	s_set_vgpr_msb 0x8359
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5909
	v_wmma_f32_16x16x32_bf16 v[252:259], v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[252:259]
	v_wmma_f32_16x16x32_bf16 v[20:27], v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[20:27]
	v_wmma_f32_16x16x32_bf16 v[188:195], v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[188:195]
	v_wmma_f32_16x16x32_bf16 v[116:123], v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[116:123]
	v_wmma_f32_16x16x32_bf16 v[36:43], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[36:43]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[0:7]
	s_set_vgpr_msb 0x900
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:192 nv
	scratch_store_b128 off, v[4:7], off offset:208 nv
	s_cbranch_scc1 .LBB0_7
	s_set_vgpr_msb 0x80
	s_clause 0x2
	scratch_load_b32 v198 /*v710*/, off, off offset:3588 nv
	scratch_load_b32 v201 /*v713*/, off, off offset:3592 nv
	scratch_load_b32 v130 /*v642*/, off, off offset:3596 nv
	s_set_vgpr_msb 0x8000
.LBB0_9:
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 8
	v_or_b32_e32 v0, s37, v130 /*v642*/
	s_load_b64 s[0:1], s[0:1], 0x110 nv
	s_mul_i32 s4, s7, s41
	s_mov_b32 s3, 0
	s_add_co_i32 s38, s38, s4
	s_set_vgpr_msb 0x840
	v_or_b32_e32 v204 /*v460*/, 5, v0
	s_mov_b32 s2, 0x800000
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[68:69], v[244:245]
	v_mov_b64_e32 v[70:71], v[246:247]
	v_mov_b64_e32 v[72:73], v[248:249]
	v_mov_b64_e32 v[74:75], v[250:251]
	s_set_vgpr_msb 0x43
	s_wait_kmcnt 0x0
	v_cvt_pk_bf16_f32 v194 /*v450*/, v218 /*v986*/, s0
	v_cvt_pk_bf16_f32 v195 /*v451*/, v219 /*v987*/, s0
	v_cvt_pk_bf16_f32 v198 /*v454*/, v220 /*v988*/, s0
	v_cvt_pk_bf16_f32 v206 /*v462*/, v222 /*v990*/, s0
	v_cvt_pk_bf16_f32 v208 /*v464*/, v223 /*v991*/, s0
	s_set_vgpr_msb 0x4301
	v_cvt_pk_bf16_f32 v2, v178 /*v434*/, s0
	s_set_vgpr_msb 0x143
	v_cvt_pk_bf16_f32 v209 /*v465*/, v225 /*v993*/, s0
	s_set_vgpr_msb 0x4301
	v_cvt_pk_bf16_f32 v3, v179 /*v435*/, s0
	v_cvt_pk_bf16_f32 v4, v180 /*v436*/, s0
	v_cvt_pk_bf16_f32 v5, v181 /*v437*/, s0
	v_cvt_pk_bf16_f32 v8, v184 /*v440*/, s0
	v_cvt_pk_bf16_f32 v6, v183 /*v439*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v178 /*v434*/, v154 /*v410*/, s0
	v_cvt_pk_bf16_f32 v180 /*v436*/, v156 /*v412*/, s0
	v_cvt_pk_bf16_f32 v181 /*v437*/, v157 /*v413*/, s0
	v_cvt_pk_bf16_f32 v183 /*v439*/, v160 /*v416*/, s0
	v_cvt_pk_bf16_f32 v154 /*v410*/, v114 /*v370*/, s0
	v_cvt_pk_bf16_f32 v157 /*v413*/, v117 /*v373*/, s0
	v_cvt_pk_bf16_f32 v156 /*v412*/, v116 /*v372*/, s0
	v_cvt_pk_bf16_f32 v114 /*v370*/, v90 /*v346*/, s0
	v_cvt_pk_bf16_f32 v116 /*v372*/, v92 /*v348*/, s0
	v_cvt_pk_bf16_f32 v117 /*v373*/, v93 /*v349*/, s0
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v90 /*v346*/, v210 /*v978*/, s0
	v_cvt_pk_bf16_f32 v93 /*v349*/, v213 /*v981*/, s0
	v_cvt_pk_bf16_f32 v92 /*v348*/, v212 /*v980*/, s0
	v_cvt_pk_bf16_f32 v29 /*v285*/, v149 /*v917*/, s0
	v_cvt_pk_bf16_f32 v30 /*v286*/, v150 /*v918*/, s0
	v_cvt_pk_bf16_f32 v28 /*v284*/, v148 /*v916*/, s0
	s_set_vgpr_msb 0x4341
	v_cvt_pk_bf16_f32 v18 /*v274*/, v66 /*v322*/, s0
	v_cvt_pk_bf16_f32 v19 /*v275*/, v67 /*v323*/, s0
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v10 /*v266*/, v192 /*v960*/, s0
	v_cvt_pk_bf16_f32 v11 /*v267*/, v193 /*v961*/, s0
	v_cvt_pk_bf16_f32 v12 /*v268*/, v194 /*v962*/, s0
	v_cvt_pk_bf16_f32 v13 /*v269*/, v197 /*v965*/, s0
	s_set_vgpr_msb 0x4303
	v_cvt_pk_bf16_f32 v242, v234 /*v1002*/, s0
	v_or_b32_e32 v1, 1, v0
	v_cvt_pk_bf16_f32 v244, v236 /*v1004*/, s0
	s_set_vgpr_msb 0x340
	v_mul_lo_u32 v196 /*v452*/, v0, s7
	s_set_vgpr_msb 0x4003
	v_cvt_pk_bf16_f32 v245, v237 /*v1005*/, s0
	s_set_vgpr_msb 0x340
	v_or_b32_e32 v197 /*v453*/, 2, v0
	s_set_vgpr_msb 0x4000
	v_mul_lo_u32 v1, v1, s7
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v247, v240 /*v1008*/, s0
	s_set_vgpr_msb 0x340
	v_or_b32_e32 v199 /*v455*/, 3, v0
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v226, v212, s0
	s_set_vgpr_msb 0x41
	v_mul_lo_u32 v197 /*v453*/, v197 /*v453*/, s7
	v_add_lshl_u32 v196 /*v452*/, v196 /*v452*/, s38, 7
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v227, v213, s0
	s_set_vgpr_msb 0x41
	v_mul_lo_u32 v199 /*v455*/, v199 /*v455*/, s7
	s_set_vgpr_msb 0x4100
	v_add_lshl_u32 v1, s38, v1, 7
	v_cvt_pk_bf16_f32 v221, v31, s0
	s_set_vgpr_msb 64
	v_or_b32_e32 v200 /*v456*/, 4, v0
	s_set_vgpr_msb 0x4049
	v_or_b32_e32 v201 /*v457*/, v196 /*v452*/, v201 /*v713*/
	v_add_lshl_u32 v197 /*v453*/, v197 /*v453*/, s38, 7
	s_set_vgpr_msb 0x4948
	v_or_b32_e32 v202 /*v458*/, v1, v201 /*v713*/
	s_set_vgpr_msb 0x4849
	v_or_b32_e32 v196 /*v452*/, v196 /*v452*/, v198 /*v710*/
	v_mul_lo_u32 v200 /*v456*/, v200 /*v456*/, s7
	v_add_lshl_u32 v199 /*v455*/, v199 /*v455*/, s38, 7
	v_or_b32_e32 v203 /*v459*/, v197 /*v453*/, v201 /*v713*/
	s_set_vgpr_msb 0x4944
	v_dual_lshlrev_b32 v202 /*v458*/, 2, v202 /*v458*/ :: v_dual_lshlrev_b32 v201 /*v457*/, 2, v201 /*v457*/
	v_lshlrev_b32_e32 v196 /*v452*/, 2, v196 /*v452*/
	s_set_vgpr_msb 0x4408
	v_or_b32_e32 v1, v1, v198 /*v710*/
	s_set_vgpr_msb 0x849
	v_or_b32_e32 v197 /*v453*/, v197 /*v453*/, v198 /*v710*/
	v_add_lshl_u32 v200 /*v456*/, v200 /*v456*/, s38, 7
	s_clause 0x1
	buffer_store_b16 v194 /*v450*/, v201 /*v457*/, s[0:3], null offen
	buffer_store_b16 v195 /*v451*/, v202 /*v458*/, s[0:3], null offen
	s_set_vgpr_msb 0x4900
	v_lshlrev_b32_e32 v1, 2, v1
	s_set_vgpr_msb 0x49
	v_or_b32_e32 v195 /*v451*/, v199 /*v455*/, v201 /*v713*/
	v_or_b32_e32 v205 /*v461*/, v200 /*v456*/, v201 /*v713*/
	s_set_vgpr_msb 0x4944
	v_lshlrev_b32_e32 v197 /*v453*/, 2, v197 /*v453*/
	s_set_vgpr_msb 0x4400
	v_cvt_pk_bf16_f32 v222, v32, s0
	s_set_vgpr_msb 0x44
	v_lshlrev_b32_e32 v194 /*v450*/, 2, v203 /*v459*/
	v_mul_lo_u32 v203 /*v459*/, s7, v204 /*v460*/
	v_dual_lshlrev_b32 v205 /*v461*/, 2, v205 /*v461*/ :: v_dual_lshlrev_b32 v195 /*v451*/, 2, v195 /*v451*/
	s_set_vgpr_msb 0x4449
	v_or_b32_e32 v200 /*v456*/, v200 /*v456*/, v198 /*v710*/
	buffer_store_b16 v198 /*v454*/, v194 /*v450*/, s[0:3], null offen
	s_set_vgpr_msb 0x4943
	v_or_b32_e32 v198 /*v454*/, 6, v0
	v_cvt_pk_bf16_f32 v204 /*v460*/, v221 /*v989*/, s0
	s_set_vgpr_msb 0x4341
	s_clause 0x1
	buffer_store_b16 v204 /*v460*/, v195 /*v451*/, s[0:3], null offen
	buffer_store_b16 v206 /*v462*/, v205 /*v461*/, s[0:3], null offen
	v_add_lshl_u32 v203 /*v459*/, v203 /*v459*/, s38, 7
	v_mul_lo_u32 v198 /*v454*/, v198 /*v454*/, s7
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v206 /*v462*/, v224 /*v992*/, s0
	s_set_vgpr_msb 0x4300
	v_cvt_pk_bf16_f32 v220, v30, s0
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v213, v157 /*v669*/, s0
	v_or_b32_e32 v0, 7, v0
	s_set_vgpr_msb 0x249
	v_or_b32_e32 v207 /*v463*/, v203 /*v459*/, v201 /*v713*/
	s_set_vgpr_msb 0x4902
	v_cvt_pk_bf16_f32 v212, v156 /*v668*/, s0
	s_set_vgpr_msb 0x200
	v_mov_b64_e32 v[30:31], v[190:191]
	s_set_vgpr_msb 0x44
	v_add_lshl_u32 v198 /*v454*/, s38, v198 /*v454*/, 7
	s_set_vgpr_msb 0x4400
	v_mul_lo_u32 v0, v0, s7
	s_set_vgpr_msb 0x44
	v_lshlrev_b32_e32 v207 /*v463*/, 2, v207 /*v463*/
	s_set_vgpr_msb 0x4400
	v_cvt_pk_bf16_f32 v180, v166, s0
	v_cvt_pk_bf16_f32 v181, v167, s0
	s_set_vgpr_msb 0x49
	v_or_b32_e32 v204 /*v460*/, v198 /*v454*/, v201 /*v713*/
	s_set_vgpr_msb 0x4900
	v_cvt_pk_bf16_f32 v183, v170, s0
	s_set_vgpr_msb 0x41
	buffer_store_b16 v208 /*v464*/, v207 /*v463*/, s[0:3], null offen
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v170, v156, s0
	v_add_lshl_u32 v0, s38, v0, 7
	s_set_vgpr_msb 0x44
	v_lshlrev_b32_e32 v204 /*v460*/, 2, v204 /*v460*/
	s_set_vgpr_msb 0x4449
	v_or_b32_e32 v198 /*v454*/, v198 /*v454*/, v198 /*v710*/
	s_set_vgpr_msb 0x4900
	v_cvt_pk_bf16_f32 v156, v142, s0
	s_set_vgpr_msb 0xc0
	s_clause 0x1
	scratch_load_b128 v[162:165] /*v[930:933]*/, off, off offset:192 th:TH_LOAD_LU nv
	scratch_load_b128 v[166:169] /*v[934:937]*/, off, off offset:208 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc048
	v_or_b32_e32 v208 /*v464*/, v0, v201 /*v713*/
	s_set_vgpr_msb 0x4844
	v_dual_lshlrev_b32 v198 /*v454*/, 2, v198 /*v454*/ :: v_dual_bitop2_b32 v210 /*v466*/, 64, v196 /*v452*/ bitop3:0x54
	s_set_vgpr_msb 0x4408
	v_or_b32_e32 v0, v0, v198 /*v710*/
	s_set_vgpr_msb 0x8c0
	v_mov_b64_e32 v[212:213] /*v[980:981]*/, v[102:103]
	s_set_vgpr_msb 0xc045
	v_lshlrev_b32_e32 v208 /*v464*/, 2, v208 /*v464*/
	s_clause 0x1
	buffer_store_b16 v206 /*v462*/, v204 /*v460*/, s[0:3], null offen
	buffer_store_b16 v209 /*v465*/, v208 /*v464*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v206 /*v462*/, 64, v197 /*v453*/
	s_set_vgpr_msb 0x4500
	v_lshlrev_b32_e32 v0, 2, v0
	v_cvt_pk_bf16_f32 v98, v60, s0
	v_cvt_pk_bf16_f32 v99, v61, s0
	s_set_vgpr_msb 9
	buffer_store_b16 v2, v210 /*v466*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, v199 /*v455*/, v198 /*v710*/
	s_set_vgpr_msb 0x900
	v_or_b32_e32 v9, 64, v0
	v_cvt_pk_bf16_f32 v93, v79, s0
	s_set_vgpr_msb 64
	v_or_b32_e32 v199 /*v455*/, 64, v1
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v94, v80, s0
	v_lshlrev_b32_e32 v2, 2, v2
	v_cvt_pk_bf16_f32 v92, v78, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v60, v173 /*v941*/, s0
	s_set_vgpr_msb 0x309
	buffer_store_b16 v3, v199 /*v455*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, v203 /*v459*/, v198 /*v710*/
	s_set_vgpr_msb 0x940
	v_or_b32_e32 v209 /*v465*/, 64, v2
	v_or_b32_e32 v203 /*v459*/, 0xc0, v2
	s_set_vgpr_msb 0x4005
	buffer_store_b16 v4, v206 /*v462*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v4, 2, v200 /*v456*/
	s_set_vgpr_msb 0x500
	v_lshlrev_b32_e32 v3, 2, v3
	s_set_vgpr_msb 0x43
	v_cvt_pk_bf16_f32 v200 /*v456*/, v204 /*v972*/, s0
	s_set_vgpr_msb 0x4301
	buffer_store_b16 v5, v209 /*v465*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v5, v182 /*v438*/, s0
	v_or_b32_e32 v7, 64, v4
	s_set_vgpr_msb 0x140
	v_or_b32_e32 v199 /*v455*/, 64, v3
	s_set_vgpr_msb 0x4000
	s_clause 0x2
	buffer_store_b16 v5, v7, s[0:3], null offen
	s_set_vgpr_msb 5
	buffer_store_b16 v6, v199 /*v455*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v5, 64, v198 /*v454*/
	s_set_vgpr_msb 0x543
	v_cvt_pk_bf16_f32 v199 /*v455*/, v203 /*v971*/, s0
	s_set_vgpr_msb 0x4300
	v_mov_b16_e32 v6.l, v8.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v7, v185 /*v441*/, s0
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v8, v202 /*v970*/, s0
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:32 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:48 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x300
	buffer_store_b16 v6, v5, s[0:3], null offen
	s_set_vgpr_msb 1
	v_mov_b16_e64 v6.l, v200.l /*v456.l*/
	s_set_vgpr_msb 0x144
	v_or_b32_e32 v200 /*v456*/, 0xc0, v197 /*v453*/
	s_set_vgpr_msb 0x4400
	s_clause 0x2
	buffer_store_b16 v7, v9, s[0:3], null offen
	s_set_vgpr_msb 1
	buffer_store_b16 v8, v201 /*v457*/, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v7, v205 /*v973*/, s0
	s_set_vgpr_msb 0x301
	v_mov_b16_e64 v5.l, v199.l /*v455.l*/
	s_clause 0x1
	buffer_store_b16 v5, v202 /*v458*/, s[0:3], null offen offset:128
	buffer_store_b16 v6, v194 /*v450*/, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v5, v206 /*v974*/, s0
	v_cvt_pk_bf16_f32 v8, v208 /*v976*/, s0
	s_set_vgpr_msb 0x301
	buffer_store_b16 v7, v195 /*v451*/, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v6, v207 /*v975*/, s0
	s_set_vgpr_msb 0x301
	s_clause 0x1
	buffer_store_b16 v5, v205 /*v461*/, s[0:3], null offen offset:128
	buffer_store_b16 v6, v207 /*v463*/, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x100
	v_mov_b16_e32 v7.l, v8.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v5, v162 /*v418*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v199 /*v455*/, v165 /*v421*/, s0
	v_cvt_pk_bf16_f32 v162 /*v418*/, v138 /*v394*/, s0
	v_cvt_pk_bf16_f32 v165 /*v421*/, v141 /*v397*/, s0
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v138 /*v394*/, v138 /*v906*/, s0
	s_set_vgpr_msb 0x4307
	v_or_b32_e32 v6, 0xc0, v196 /*v452*/
	v_cvt_pk_bf16_f32 v9, v209 /*v977*/, s0
	s_set_vgpr_msb 0x743
	v_cvt_pk_bf16_f32 v141 /*v397*/, v143 /*v911*/, s0
	s_set_vgpr_msb 0x4300
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b16_e32 v8.l, v9.l
	s_set_vgpr_msb 1
	s_clause 0x1
	buffer_store_b16 v7, v204 /*v460*/, s[0:3], null offen offset:128
	buffer_store_b16 v8, v208 /*v464*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v7, v163 /*v419*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v163 /*v419*/, v139 /*v395*/, s0
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v139 /*v395*/, v139 /*v907*/, s0
	s_set_vgpr_msb 0x4301
	v_cvt_pk_bf16_f32 v8, v164 /*v420*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v164 /*v420*/, v140 /*v396*/, s0
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v140 /*v396*/, v140 /*v908*/, s0
	s_set_vgpr_msb 0x4300
	v_or_b32_e32 v9, 0xc0, v1
	s_clause 0x1
	buffer_store_b16 v5, v6, s[0:3], null offen
	buffer_store_b16 v7, v9, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v5, v166 /*v422*/, s0
	s_clause 0x2
	buffer_store_b16 v8, v200 /*v456*/, s[0:3], null offen
	s_set_vgpr_msb 0x141
	buffer_store_b16 v199 /*v455*/, v203 /*v459*/, s[0:3], null offen
	s_set_vgpr_msb 0x4101
	v_cvt_pk_bf16_f32 v8, v168 /*v424*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v168 /*v424*/, v145 /*v401*/, s0
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v9, 0xc0, v3
	s_set_vgpr_msb 64
	v_or_b32_e32 v203 /*v459*/, 0x140, v4
	s_set_vgpr_msb 0x4045
	v_or_b32_e32 v199 /*v455*/, 0xc0, v198 /*v454*/
	v_cvt_pk_bf16_f32 v200 /*v456*/, v169 /*v425*/, s0
	s_set_vgpr_msb 0x4501
	v_or_b32_e32 v7, 0xc0, v4
	v_or_b32_e32 v4, 0x1c0, v4
	v_cvt_pk_bf16_f32 v6, v167 /*v423*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v5, v7, s[0:3], null offen
	buffer_store_b16 v6, v9, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v5, 0xc0, v0
	s_set_vgpr_msb 1
	buffer_store_b16 v8, v199 /*v455*/, s[0:3], null offen
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v8, v205, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v205, v183 /*v951*/, s0
	s_set_vgpr_msb 0x300
	v_cvt_pk_bf16_f32 v9, v206, s0
	s_set_vgpr_msb 0x42
	v_cvt_pk_bf16_f32 v199 /*v455*/, v142 /*v654*/, s0
	s_set_vgpr_msb 0x4201
	v_mov_b16_e64 v6.l, v200.l /*v456.l*/
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v7, v204, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v204, v180 /*v948*/, s0
	s_set_vgpr_msb 0x340
	v_or_b32_e32 v200 /*v456*/, 0x140, v2
	s_set_vgpr_msb 0x4000
	buffer_store_b16 v6, v5, s[0:3], null offen
	v_or_b32_e32 v2, 0x1c0, v2
	s_wait_xcnt 0x0
	v_mov_b16_e32 v6.l, v7.l
	v_mov_b16_e32 v7.l, v8.l
	v_mov_b16_e32 v8.l, v9.l
	s_set_vgpr_msb 1
	s_clause 0x2
	buffer_store_b16 v6, v201 /*v457*/, s[0:3], null offen offset:256
	buffer_store_b16 v7, v202 /*v458*/, s[0:3], null offen offset:256
	buffer_store_b16 v8, v194 /*v450*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v7, v210, s0
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v210, v154 /*v666*/, s0
	s_set_vgpr_msb 0x200
	v_cvt_pk_bf16_f32 v5, v207, s0
	v_cvt_pk_bf16_f32 v8, v211, s0
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v211, v155 /*v667*/, s0
	s_set_vgpr_msb 0x200
	v_cvt_pk_bf16_f32 v9, v208, s0
	s_set_vgpr_msb 1
	buffer_store_b16 v5, v195 /*v451*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x104
	v_cvt_pk_bf16_f32 v5, v209, s0
	v_mov_b16_e32 v6.l, v9.l
	v_or_b32_e32 v9, 0x140, v196 /*v452*/
	s_set_vgpr_msb 0x401
	buffer_store_b16 v6, v205 /*v461*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x102
	v_cvt_pk_bf16_f32 v6, v138 /*v650*/, s0
	s_set_vgpr_msb 0x201
	s_clause 0x1
	buffer_store_b16 v5, v207 /*v463*/, s[0:3], null offen offset:256
	buffer_store_b16 v7, v204 /*v460*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x102
	v_cvt_pk_bf16_f32 v5, v139 /*v651*/, s0
	s_set_vgpr_msb 0x201
	s_clause 0x2
	buffer_store_b16 v8, v208 /*v464*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x100
	buffer_store_b16 v6, v9, s[0:3], null offen
	s_set_vgpr_msb 6
	v_cvt_pk_bf16_f32 v8, v141 /*v653*/, s0
	v_cvt_pk_bf16_f32 v7, v140 /*v652*/, s0
	v_or_b32_e32 v9, 0x140, v197 /*v453*/
	s_set_vgpr_msb 0x600
	v_or_b32_e32 v6, 0x140, v1
	s_clause 0x1
	buffer_store_b16 v5, v6, s[0:3], null offen
	buffer_store_b16 v7, v9, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v5, v143 /*v655*/, s0
	s_set_vgpr_msb 0x201
	s_clause 0x2
	buffer_store_b16 v8, v200 /*v456*/, s[0:3], null offen
	s_set_vgpr_msb 0x141
	buffer_store_b16 v199 /*v455*/, v203 /*v459*/, s[0:3], null offen
	s_set_vgpr_msb 0x4106
	v_cvt_pk_bf16_f32 v8, v145 /*v657*/, s0
	v_or_b32_e32 v9, 0x140, v198 /*v454*/
	s_set_vgpr_msb 0x641
	v_cvt_pk_bf16_f32 v199 /*v455*/, v130 /*v386*/, s0
	v_cvt_pk_bf16_f32 v130 /*v386*/, v107 /*v363*/, s0
	v_cvt_pk_bf16_f32 v107 /*v363*/, v187 /*v443*/, s0
	v_or_b32_e32 v200 /*v456*/, 0x140, v0
	s_set_vgpr_msb 0x4102
	v_cvt_pk_bf16_f32 v7, v144 /*v656*/, s0
	v_or_b32_e32 v6, 0x140, v3
	v_or_b32_e32 v1, 0x1c0, v1
	v_or_b32_e32 v3, 0x1c0, v3
	s_set_vgpr_msb 0x200
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_set_vgpr_msb 1
	v_mov_b16_e64 v5.l, v199.l /*v455.l*/
	s_set_vgpr_msb 0x100
	buffer_store_b16 v7, v9, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v7, v132 /*v388*/, s0
	v_cvt_pk_bf16_f32 v6, v131 /*v387*/, s0
	buffer_store_b16 v8, v200 /*v456*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v8, v134 /*v390*/, s0
	v_cvt_pk_bf16_f32 v9, v135 /*v391*/, s0
	buffer_store_b16 v5, v201 /*v457*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v5, v133 /*v389*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v200 /*v456*/, v151 /*v407*/, s0
	s_set_vgpr_msb 0x4101
	buffer_store_b16 v6, v202 /*v458*/, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x100
	v_mov_b16_e32 v6.l, v8.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v8, v171 /*v427*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v171 /*v427*/, v99 /*v355*/, s0
	v_cvt_pk_bf16_f32 v99 /*v355*/, v75 /*v331*/, s0
	v_cvt_pk_bf16_f32 v75 /*v331*/, v35 /*v291*/, s0
	v_cvt_pk_bf16_f32 v201 /*v457*/, v153 /*v409*/, s0
	s_set_vgpr_msb 0x4101
	buffer_store_b16 v7, v194 /*v450*/, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x104
	v_mov_b16_e32 v7.l, v9.l
	v_or_b32_e32 v9, 0x1c0, v196 /*v452*/
	s_set_vgpr_msb 0x441
	v_cvt_pk_bf16_f32 v196 /*v452*/, v149 /*v405*/, s0
	v_cvt_pk_bf16_f32 v149 /*v405*/, v125 /*v381*/, s0
	s_set_vgpr_msb 0x4101
	buffer_store_b16 v5, v195 /*v451*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v5, v136 /*v392*/, s0
	buffer_store_b16 v6, v205 /*v461*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v6, v137 /*v393*/, s0
	buffer_store_b16 v7, v207 /*v463*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v7, v170 /*v426*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v170 /*v426*/, v98 /*v354*/, s0
	v_cvt_pk_bf16_f32 v98 /*v354*/, v74 /*v330*/, s0
	v_cvt_pk_bf16_f32 v74 /*v330*/, v34 /*v290*/, s0
	s_set_vgpr_msb 0x4101
	s_clause 0x1
	buffer_store_b16 v5, v204 /*v460*/, s[0:3], null offen offset:384
	buffer_store_b16 v6, v208 /*v464*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v5, v172 /*v428*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v172 /*v428*/, v100 /*v356*/, s0
	v_cvt_pk_bf16_f32 v100 /*v356*/, v76 /*v332*/, s0
	v_cvt_pk_bf16_f32 v76 /*v332*/, v36 /*v292*/, s0
	v_cvt_pk_bf16_f32 v36 /*v292*/, v23 /*v279*/, s0
	s_set_vgpr_msb 0x4100
	s_clause 0x1
	buffer_store_b16 v7, v9, s[0:3], null offen
	buffer_store_b16 v8, v1, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v7, v174 /*v430*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v174 /*v430*/, v105 /*v361*/, s0
	s_set_vgpr_msb 0x4105
	v_or_b32_e32 v6, 0x1c0, v197 /*v453*/
	v_cvt_pk_bf16_f32 v1, v173 /*v429*/, s0
	s_set_vgpr_msb 0x504
	s_clause 0x1
	buffer_store_b16 v5, v6, s[0:3], null offen
	buffer_store_b16 v1, v2, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v5, 0x1c0, v198 /*v454*/
	s_set_vgpr_msb 0x441
	v_cvt_pk_bf16_f32 v198 /*v454*/, v150 /*v406*/, s0
	v_cvt_pk_bf16_f32 v150 /*v406*/, v126 /*v382*/, s0
	v_cvt_pk_bf16_f32 v126 /*v382*/, v86 /*v342*/, s0
	v_cvt_pk_bf16_f32 v86 /*v342*/, v62 /*v318*/, s0
	s_set_vgpr_msb 0x4108
	v_or_b32_e32 v8, s39, v130 /*v642*/
	buffer_store_b16 v7, v4, s[0:3], null offen
	s_set_vgpr_msb 0x801
	v_cvt_pk_bf16_f32 v4, v176 /*v432*/, s0
	v_cvt_pk_bf16_f32 v1, v175 /*v431*/, s0
	v_mul_lo_u32 v2, s7, v8
	v_or_b32_e32 v7, 3, v8
	s_set_vgpr_msb 0x140
	v_or_b32_e32 v195 /*v451*/, 5, v8
	s_set_vgpr_msb 0x4000
	buffer_store_b16 v1, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v1, v177 /*v433*/, s0
	v_mul_lo_u32 v7, s7, v7
	v_cvt_pk_bf16_f32 v9, v148 /*v404*/, s0
	v_add_lshl_u32 v2, s38, v2, 7
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v5, s[0:3], null offen
	s_set_vgpr_msb 0x41
	v_mul_lo_u32 v195 /*v451*/, v195 /*v451*/, s7
	v_cvt_pk_bf16_f32 v173 /*v429*/, v104 /*v360*/, s0
	v_cvt_pk_bf16_f32 v148 /*v404*/, v124 /*v380*/, s0
	s_set_vgpr_msb 0x4108
	v_or_b32_e32 v4, v2, v201 /*v713*/
	v_add_lshl_u32 v7, v7, s38, 7
	s_set_vgpr_msb 0x841
	v_cvt_pk_bf16_f32 v124 /*v380*/, v85 /*v341*/, s0
	v_cvt_pk_bf16_f32 v104 /*v360*/, v81 /*v337*/, s0
	s_set_vgpr_msb 0x4108
	v_or_b32_e32 v2, v2, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v4, 2, v4
	v_or_b32_e32 v5, 2, v8
	s_set_vgpr_msb 0x41
	v_add_lshl_u32 v195 /*v451*/, v195 /*v451*/, s38, 7
	v_cvt_pk_bf16_f32 v85 /*v341*/, v61 /*v317*/, s0
	s_set_vgpr_msb 0x4100
	v_lshlrev_b32_e32 v2, 2, v2
	v_mul_lo_u32 v5, v5, s7
	s_set_vgpr_msb 0x49
	v_or_b32_e32 v199 /*v455*/, v195 /*v451*/, v201 /*v713*/
	s_set_vgpr_msb 0x4900
	v_or_b32_e32 v0, 0x1c0, v0
	s_set_vgpr_msb 64
	v_or_b32_e32 v202 /*v458*/, 64, v2
	s_set_vgpr_msb 0x4044
	v_lshlrev_b32_e32 v199 /*v455*/, 2, v199 /*v455*/
	s_set_vgpr_msb 0x4400
	v_add_lshl_u32 v5, v5, s38, 7
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v1, v147 /*v403*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v147 /*v403*/, v123 /*v379*/, s0
	s_set_vgpr_msb 0x4148
	v_or_b32_e32 v194 /*v450*/, v5, v201 /*v713*/
	s_set_vgpr_msb 0x4808
	v_or_b32_e32 v5, v5, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v6, 1, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v5, 2, v5
	v_mul_lo_u32 v3, v6, s7
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v6, v146 /*v402*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v146 /*v402*/, v122 /*v378*/, s0
	s_set_vgpr_msb 0x4100
	s_delay_alu instid0(VALU_DEP_3)
	v_add_lshl_u32 v3, v3, s38, 7
	buffer_store_b16 v6, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v6, 4, v8
	s_set_vgpr_msb 8
	v_or_b32_e32 v0, v3, v201 /*v713*/
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_lo_u32 v6, v6, s7
	v_or_b32_e32 v3, v3, v198 /*v710*/
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v0, 2, v0
	v_lshlrev_b32_e32 v3, 2, v3
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_set_vgpr_msb 4
	v_lshlrev_b32_e32 v1, 2, v194 /*v450*/
	v_add_lshl_u32 v6, v6, s38, 7
	s_set_vgpr_msb 0x440
	v_or_b32_e32 v179 /*v435*/, 64, v3
	s_set_vgpr_msb 0x4048
	v_or_b32_e32 v194 /*v450*/, v7, v201 /*v713*/
	s_set_vgpr_msb 0x4800
	buffer_store_b16 v9, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 6, v8
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v197 /*v453*/, v6, v201 /*v713*/
	s_set_vgpr_msb 0x4844
	v_lshlrev_b32_e32 v194 /*v450*/, 2, v194 /*v450*/
	s_set_vgpr_msb 0x4408
	v_or_b32_e32 v6, v6, v198 /*v710*/
	v_mul_lo_u32 v9, v9, s7
	s_set_vgpr_msb 0x845
	v_lshlrev_b32_e32 v197 /*v453*/, 2, v197 /*v453*/
	s_clause 0x1
	buffer_store_b16 v196 /*v452*/, v194 /*v450*/, s[0:3], null offen
	buffer_store_b16 v198 /*v454*/, v197 /*v453*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v198 /*v454*/, v152 /*v408*/, s0
	s_set_vgpr_msb 0x4500
	v_or_b32_e32 v8, 7, v8
	v_add_lshl_u32 v9, v9, s38, 7
	v_lshlrev_b32_e32 v6, 2, v6
	s_set_vgpr_msb 0x41
	buffer_store_b16 v200 /*v456*/, v199 /*v455*/, s[0:3], null offen
	s_set_vgpr_msb 0x4108
	v_or_b32_e32 v7, v7, v198 /*v710*/
	v_mul_lo_u32 v8, v8, s7
	s_set_vgpr_msb 0x848
	v_or_b32_e32 v196 /*v452*/, v9, v201 /*v713*/
	s_set_vgpr_msb 0x4800
	v_lshlrev_b32_e32 v7, 2, v7
	s_set_vgpr_msb 8
	v_or_b32_e32 v9, v9, v198 /*v710*/
	s_set_vgpr_msb 0x844
	v_lshlrev_b32_e32 v196 /*v452*/, 2, v196 /*v452*/
	s_set_vgpr_msb 0x4400
	v_add_lshl_u32 v8, v8, s38, 7
	v_lshlrev_b32_e32 v9, 2, v9
	s_set_vgpr_msb 0x48
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v200 /*v456*/, v8, v201 /*v713*/
	s_set_vgpr_msb 0x4840
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v167 /*v423*/, 0xc0, v9
	s_set_vgpr_msb 0x4008
	v_or_b32_e32 v8, v8, v198 /*v710*/
	s_set_vgpr_msb 0x845
	v_lshlrev_b32_e32 v200 /*v456*/, 2, v200 /*v456*/
	s_clause 0x1
	buffer_store_b16 v198 /*v454*/, v196 /*v452*/, s[0:3], null offen
	buffer_store_b16 v201 /*v457*/, v200 /*v456*/, s[0:3], null offen
	s_set_vgpr_msb 0x4540
	v_or_b32_e32 v198 /*v454*/, 64, v5
	s_set_vgpr_msb 0x4000
	v_lshlrev_b32_e32 v8, 2, v8
	s_set_vgpr_msb 0x41
	buffer_store_b16 v178 /*v434*/, v202 /*v458*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v178 /*v434*/, v155 /*v411*/, s0
	s_set_vgpr_msb 0x4100
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v18, v10, s0
	s_set_vgpr_msb 0x41
	v_or_b32_e32 v151 /*v407*/, 0x140, v8
	v_or_b32_e32 v201 /*v457*/, 64, v7
	v_cvt_pk_bf16_f32 v155 /*v411*/, v115 /*v371*/, s0
	buffer_store_b16 v178 /*v434*/, v179 /*v435*/, s[0:3], null offen
	s_set_vgpr_msb 0x4149
	v_or_b32_e32 v178 /*v434*/, v195 /*v451*/, v198 /*v710*/
	s_set_vgpr_msb 0x4900
	v_cvt_pk_bf16_f32 v19, v11, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v10, v164 /*v932*/, s0
	s_set_vgpr_msb 0x345
	buffer_store_b16 v180 /*v436*/, v198 /*v454*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v180 /*v436*/, v159 /*v415*/, s0
	v_lshlrev_b32_e32 v178 /*v434*/, 2, v178 /*v434*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v158 /*v414*/, s0
	v_cvt_pk_bf16_f32 v158 /*v414*/, v118 /*v374*/, s0
	buffer_store_b16 v181 /*v437*/, v201 /*v457*/, s[0:3], null offen
	s_set_vgpr_msb 0x4540
	v_or_b32_e32 v181 /*v437*/, 64, v6
	s_set_vgpr_msb 0x4045
	v_or_b32_e32 v182 /*v438*/, 64, v178 /*v434*/
	s_clause 0x1
	buffer_store_b16 v179 /*v435*/, v181 /*v437*/, s[0:3], null offen
	buffer_store_b16 v180 /*v436*/, v182 /*v438*/, s[0:3], null offen
	s_set_vgpr_msb 0x4540
	v_or_b32_e32 v179 /*v435*/, 64, v9
	s_set_vgpr_msb 0x4044
	v_or_b32_e32 v166 /*v422*/, 0xc0, v178 /*v434*/
	s_set_vgpr_msb 0x4441
	v_or_b32_e32 v182 /*v438*/, 64, v8
	v_mov_b16_e64 v180.l /*v436.l*/, v183.l /*v439.l*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v161 /*v417*/, s0
	s_clause 0x3
	buffer_store_b16 v180 /*v436*/, v179 /*v435*/, s[0:3], null offen
	buffer_store_b16 v181 /*v437*/, v182 /*v438*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v170 /*v426*/, v4, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x4041
	v_cvt_pk_bf16_f32 v170 /*v426*/, v101 /*v357*/, s0
	v_cvt_pk_bf16_f32 v101 /*v357*/, v77 /*v333*/, s0
	v_cvt_pk_bf16_f32 v77 /*v333*/, v39 /*v295*/, s0
	v_cvt_pk_bf16_f32 v39 /*v295*/, v25 /*v281*/, s0
	s_set_vgpr_msb 0x4140
	s_clause 0x1
	buffer_store_b16 v171 /*v427*/, v0, s[0:3], null offen offset:128
	buffer_store_b16 v172 /*v428*/, v1, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x4041
	v_cvt_pk_bf16_f32 v171 /*v427*/, v102 /*v358*/, s0
	buffer_store_b16 v170 /*v426*/, v194 /*v450*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l /*v426.l*/, v173.l /*v429.l*/
	v_mov_b16_e64 v173.l /*v429.l*/, v174.l /*v430.l*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v103 /*v359*/, s0
	s_clause 0x1
	buffer_store_b16 v171 /*v427*/, v197 /*v453*/, s[0:3], null offen offset:128
	buffer_store_b16 v172 /*v428*/, v199 /*v455*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v171 /*v427*/, 0xc0, v3
	s_clause 0x1
	buffer_store_b16 v170 /*v426*/, v196 /*v452*/, s[0:3], null offen offset:128
	buffer_store_b16 v173 /*v429*/, v200 /*v456*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v170 /*v426*/, 0xc0, v2
	s_clause 0x1
	buffer_store_b16 v162 /*v418*/, v170 /*v426*/, s[0:3], null offen
	buffer_store_b16 v163 /*v419*/, v171 /*v427*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v162 /*v418*/, v142 /*v398*/, s0
	v_or_b32_e32 v172 /*v428*/, 0xc0, v5
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v163 /*v419*/, v143 /*v399*/, s0
	v_or_b32_e32 v173 /*v429*/, 0xc0, v7
	s_clause 0x1
	buffer_store_b16 v164 /*v420*/, v172 /*v428*/, s[0:3], null offen
	buffer_store_b16 v165 /*v421*/, v173 /*v429*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v164 /*v420*/, 0xc0, v6
	s_clause 0x1
	buffer_store_b16 v162 /*v418*/, v164 /*v420*/, s[0:3], null offen
	buffer_store_b16 v163 /*v419*/, v166 /*v422*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v162 /*v418*/, 0xc0, v8
	s_wait_xcnt 0x0
	v_mov_b16_e64 v163.l /*v419.l*/, v168.l /*v424.l*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v144 /*v400*/, s0
	s_clause 0x5
	buffer_store_b16 v165 /*v421*/, v167 /*v423*/, s[0:3], null offen
	buffer_store_b16 v163 /*v419*/, v162 /*v418*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v154 /*v410*/, v4, s[0:3], null offen offset:256
	buffer_store_b16 v155 /*v411*/, v0, s[0:3], null offen offset:256
	buffer_store_b16 v156 /*v412*/, v1, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x4041
	v_mov_b16_e64 v154.l /*v410.l*/, v158.l /*v414.l*/
	buffer_store_b16 v157 /*v413*/, v194 /*v450*/, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v157 /*v413*/, v121 /*v377*/, s0
	v_cvt_pk_bf16_f32 v156 /*v412*/, v120 /*v376*/, s0
	buffer_store_b16 v154 /*v410*/, v197 /*v453*/, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v155 /*v411*/, v119 /*v375*/, s0
	v_cvt_pk_bf16_f32 v119 /*v375*/, v96 /*v352*/, s0
	s_wait_xcnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b16_e64 v154.l /*v410.l*/, v155.l /*v411.l*/
	v_mov_b16_e64 v155.l /*v411.l*/, v156.l /*v412.l*/
	s_clause 0x1
	buffer_store_b16 v154 /*v410*/, v199 /*v455*/, s[0:3], null offen offset:256
	buffer_store_b16 v155 /*v411*/, v196 /*v452*/, s[0:3], null offen offset:256
	s_wait_xcnt 0x1
	v_or_b32_e32 v154 /*v410*/, 0x140, v5
	s_wait_xcnt 0x0
	v_or_b32_e32 v155 /*v411*/, 0x140, v7
	v_mov_b16_e64 v156.l /*v412.l*/, v157.l /*v413.l*/
	v_or_b32_e32 v157 /*v413*/, 0x140, v2
	s_clause 0x1
	buffer_store_b16 v156 /*v412*/, v200 /*v456*/, s[0:3], null offen offset:256
	buffer_store_b16 v146 /*v402*/, v157 /*v413*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v156 /*v412*/, 0x140, v6
	s_wait_xcnt 0x0
	v_or_b32_e32 v146 /*v402*/, 0x140, v3
	s_clause 0x1
	buffer_store_b16 v147 /*v403*/, v146 /*v402*/, s[0:3], null offen
	buffer_store_b16 v148 /*v404*/, v154 /*v410*/, s[0:3], null offen
	s_set_vgpr_msb 0x4144
	v_or_b32_e32 v147 /*v403*/, 0x140, v178 /*v434*/
	s_set_vgpr_msb 0x4400
	v_or_b32_e32 v2, 0x1c0, v2
	s_set_vgpr_msb 0x41
	s_clause 0x1
	buffer_store_b16 v149 /*v405*/, v155 /*v411*/, s[0:3], null offen
	buffer_store_b16 v150 /*v406*/, v156 /*v412*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v149 /*v405*/, v129 /*v385*/, s0
	v_cvt_pk_bf16_f32 v148 /*v404*/, v128 /*v384*/, s0
	v_cvt_pk_bf16_f32 v128 /*v384*/, v89 /*v345*/, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v150 /*v406*/, 0x140, v9
	v_cvt_pk_bf16_f32 v146 /*v402*/, v127 /*v383*/, s0
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v3, 0x1c0, v3
	s_set_vgpr_msb 0x41
	v_cvt_pk_bf16_f32 v127 /*v383*/, v87 /*v343*/, s0
	s_clause 0x4
	buffer_store_b16 v146 /*v402*/, v147 /*v403*/, s[0:3], null offen
	buffer_store_b16 v148 /*v404*/, v150 /*v406*/, s[0:3], null offen
	buffer_store_b16 v149 /*v405*/, v151 /*v407*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v138 /*v394*/, v4, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x4041
	v_mov_b16_e64 v138.l /*v394.l*/, v139.l /*v395.l*/
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v139 /*v395*/, v142 /*v910*/, s0
	s_set_vgpr_msb 0x4303
	v_cvt_pk_bf16_f32 v4, v141 /*v909*/, s0
	s_set_vgpr_msb 0x340
	buffer_store_b16 v138 /*v394*/, v0, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x4001
	v_mov_b16_e64 v0.l, v139.l /*v395.l*/
	s_set_vgpr_msb 0x140
	buffer_store_b16 v140 /*v396*/, v1, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x4001
	v_mov_b16_e64 v1.l, v141.l /*v397.l*/
	buffer_store_b16 v4, v194 /*v450*/, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v4, v144 /*v912*/, s0
	s_set_vgpr_msb 0x301
	buffer_store_b16 v0, v197 /*v453*/, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v0, v145 /*v913*/, s0
	s_set_vgpr_msb 0x301
	buffer_store_b16 v1, v199 /*v455*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v106 /*v362*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v106 /*v362*/, v186 /*v442*/, s0
	s_set_vgpr_msb 0x4101
	s_clause 0x1
	buffer_store_b16 v4, v196 /*v452*/, s[0:3], null offen offset:384
	buffer_store_b16 v0, v200 /*v456*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_or_b32_e32 v4, 0x1c0, v7
	v_cvt_pk_bf16_f32 v7, v82 /*v338*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v82 /*v338*/, v58 /*v314*/, s0
	s_set_vgpr_msb 0x4140
	v_cvt_pk_bf16_f32 v58 /*v314*/, v253, s0
	s_set_vgpr_msb 0x4000
	s_clause 0x2
	buffer_store_b16 v1, v2, s[0:3], null offen
	s_set_vgpr_msb 64
	buffer_store_b16 v130 /*v386*/, v3, s[0:3], null offen
	s_set_vgpr_msb 0x4001
	v_cvt_pk_bf16_f32 v1, v109 /*v365*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v109 /*v365*/, v192 /*v448*/, s0
	s_set_vgpr_msb 0x4101
	v_cvt_pk_bf16_f32 v0, v108 /*v364*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v108 /*v364*/, v188 /*v444*/, s0
	s_set_vgpr_msb 0x4101
	v_cvt_pk_bf16_f32 v3, v110 /*v366*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v110 /*v366*/, v193 /*v449*/, s0
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v2, 0x1c0, v5
	s_clause 0x1
	buffer_store_b16 v0, v2, s[0:3], null offen
	buffer_store_b16 v1, v4, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v0, v111 /*v367*/, s0
	v_or_b32_e32 v4, 0x1c0, v9
	v_cvt_pk_bf16_f32 v9, v84 /*v340*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v84 /*v340*/, v60 /*v316*/, s0
	s_set_vgpr_msb 0x4101
	v_cvt_pk_bf16_f32 v2, v112 /*v368*/, s0
	v_or_b32_e32 v5, 0x1c0, v6
	s_set_vgpr_msb 0x108
	v_or_b32_e32 v6, s36, v130 /*v642*/
	buffer_store_b16 v3, v5, s[0:3], null offen
	s_set_vgpr_msb 0x804
	v_or_b32_e32 v3, 0x1c0, v178 /*v434*/
	v_mul_lo_u32 v1, v6, s7
	s_set_vgpr_msb 0x440
	v_or_b32_e32 v123 /*v379*/, 5, v6
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v5, 1, v6
	buffer_store_b16 v0, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v0, v113 /*v369*/, s0
	s_set_vgpr_msb 0x141
	v_mul_lo_u32 v123 /*v379*/, v123 /*v379*/, s7
	s_set_vgpr_msb 0x4100
	v_add_lshl_u32 v1, s38, v1, 7
	v_mul_lo_u32 v3, v5, s7
	buffer_store_b16 v2, v4, s[0:3], null offen
	v_or_b32_e32 v5, 2, v6
	s_set_vgpr_msb 8
	v_or_b32_e32 v2, v1, v201 /*v713*/
	s_set_vgpr_msb 0x841
	v_add_lshl_u32 v123 /*v379*/, v123 /*v379*/, s38, 7
	s_set_vgpr_msb 0x4100
	v_add_lshl_u32 v3, v3, s38, 7
	v_mul_lo_u32 v5, v5, s7
	v_lshlrev_b32_e32 v2, 2, v2
	v_or_b32_e32 v4, 0x1c0, v8
	s_set_vgpr_msb 0x49
	v_or_b32_e32 v130 /*v386*/, v123 /*v379*/, v201 /*v713*/
	s_set_vgpr_msb 0x4908
	v_or_b32_e32 v1, v1, v198 /*v710*/
	buffer_store_b16 v0, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v0, v3, v201 /*v713*/
	v_add_lshl_u32 v5, v5, s38, 7
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_lshlrev_b32 v0, 2, v0 :: v_dual_bitop2_b32 v8, 3, v6 bitop3:0x54
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v122 /*v378*/, v5, v201 /*v713*/
	s_set_vgpr_msb 0x4808
	v_or_b32_e32 v5, v5, v198 /*v710*/
	v_or_b32_e32 v3, v3, v198 /*v710*/
	v_mul_lo_u32 v8, v8, s7
	s_set_vgpr_msb 0x840
	v_or_b32_e32 v129 /*v385*/, 64, v1
	s_set_vgpr_msb 0x4000
	buffer_store_b16 v7, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v5, 2, v5 :: v_dual_bitop2_b32 v7, 4, v6 bitop3:0x54
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v83 /*v339*/, s0
	v_lshlrev_b32_e32 v3, 2, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_lo_u32 v7, s7, v7
	v_add_lshl_u32 v8, s38, v8, 7
	s_set_vgpr_msb 0x104
	buffer_store_b16 v4, v0, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v4, 2, v122 /*v378*/
	s_set_vgpr_msb 0x441
	v_or_b32_e32 v115 /*v371*/, 64, v3
	v_cvt_pk_bf16_f32 v83 /*v339*/, v59 /*v315*/, s0
	s_set_vgpr_msb 0x4148
	v_or_b32_e32 v122 /*v378*/, v8, v201 /*v713*/
	s_set_vgpr_msb 0x4808
	v_or_b32_e32 v8, v8, v198 /*v710*/
	v_add_lshl_u32 v7, v7, s38, 7
	buffer_store_b16 v9, v4, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v9, 6, v6
	v_lshlrev_b32_e32 v8, 2, v8
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v125 /*v381*/, v7, v201 /*v713*/
	s_set_vgpr_msb 0x4800
	v_mul_lo_u32 v9, v9, s7
	s_set_vgpr_msb 0x44
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_lshlrev_b32 v122 /*v378*/, 2, v122 /*v378*/ :: v_dual_lshlrev_b32 v125 /*v381*/, 2, v125 /*v381*/
	s_set_vgpr_msb 0x4408
	v_or_b32_e32 v7, v7, v198 /*v710*/
	s_set_vgpr_msb 0x841
	s_clause 0x1
	buffer_store_b16 v124 /*v380*/, v122 /*v378*/, s[0:3], null offen
	buffer_store_b16 v126 /*v382*/, v125 /*v381*/, s[0:3], null offen
	s_set_vgpr_msb 0x4100
	v_add_lshl_u32 v9, v9, s38, 7
	v_lshlrev_b32_e32 v7, 2, v7
	s_set_vgpr_msb 0x41
	v_cvt_pk_bf16_f32 v126 /*v382*/, v88 /*v344*/, s0
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v6, 7, v6
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v124 /*v380*/, v9, v201 /*v713*/
	s_set_vgpr_msb 0x4800
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_lo_u32 v6, v6, s7
	s_set_vgpr_msb 0x44
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_lshlrev_b32 v130 /*v386*/, 2, v130 /*v386*/ :: v_dual_lshlrev_b32 v124 /*v380*/, 2, v124 /*v380*/
	s_set_vgpr_msb 0x4408
	v_or_b32_e32 v9, v9, v198 /*v710*/
	s_set_vgpr_msb 0x841
	buffer_store_b16 v127 /*v383*/, v130 /*v386*/, s[0:3], null offen
	s_set_vgpr_msb 0x4100
	v_add_lshl_u32 v6, v6, s38, 7
	s_set_vgpr_msb 0x48
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v127 /*v383*/, v6, v201 /*v713*/
	s_set_vgpr_msb 0x4808
	v_or_b32_e32 v6, v6, v198 /*v710*/
	s_set_vgpr_msb 0x845
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v127 /*v383*/, 2, v127 /*v383*/
	s_clause 0x1
	buffer_store_b16 v126 /*v382*/, v124 /*v380*/, s[0:3], null offen
	buffer_store_b16 v128 /*v384*/, v127 /*v383*/, s[0:3], null offen
	s_set_vgpr_msb 0x4540
	v_or_b32_e32 v126 /*v382*/, 64, v5
	s_set_vgpr_msb 0x4000
	v_dual_lshlrev_b32 v9, 2, v9 :: v_dual_lshlrev_b32 v6, 2, v6
	s_set_vgpr_msb 0x41
	buffer_store_b16 v114 /*v370*/, v129 /*v385*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v114 /*v370*/, v91 /*v347*/, s0
	v_or_b32_e32 v128 /*v384*/, 64, v8
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v91 /*v347*/, v211 /*v979*/, s0
	v_or_b32_e32 v87 /*v343*/, 0x140, v6
	s_set_vgpr_msb 0x4349
	buffer_store_b16 v114 /*v370*/, v115 /*v371*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v114 /*v370*/, v123 /*v379*/, v198 /*v710*/
	s_set_vgpr_msb 0x4940
	v_or_b32_e32 v103 /*v359*/, 0xc0, v9
	s_set_vgpr_msb 0x40c0
	v_mov_b64_e32 v[210:211] /*v[978:979]*/, v[100:101]
	s_set_vgpr_msb 0xc045
	buffer_store_b16 v116 /*v372*/, v126 /*v382*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v116 /*v372*/, v95 /*v351*/, s0
	v_lshlrev_b32_e32 v114 /*v370*/, 2, v114 /*v370*/
	v_cvt_pk_bf16_f32 v115 /*v371*/, v94 /*v350*/, s0
	s_set_vgpr_msb 0x4543
	v_cvt_pk_bf16_f32 v94 /*v350*/, v214 /*v982*/, s0
	s_set_vgpr_msb 0x4341
	buffer_store_b16 v117 /*v373*/, v128 /*v384*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v117 /*v373*/, 64, v7
	s_set_vgpr_msb 0x4145
	v_or_b32_e32 v118 /*v374*/, 64, v114 /*v370*/
	s_clause 0x1
	buffer_store_b16 v115 /*v371*/, v117 /*v373*/, s[0:3], null offen
	buffer_store_b16 v116 /*v372*/, v118 /*v374*/, s[0:3], null offen
	s_set_vgpr_msb 0x4540
	v_or_b32_e32 v115 /*v371*/, 64, v9
	s_set_vgpr_msb 0x4044
	v_or_b32_e32 v102 /*v358*/, 0xc0, v114 /*v370*/
	s_set_vgpr_msb 0x4441
	v_or_b32_e32 v118 /*v374*/, 64, v6
	v_mov_b16_e64 v116.l /*v372.l*/, v119.l /*v375.l*/
	v_cvt_pk_bf16_f32 v117 /*v373*/, v97 /*v353*/, s0
	s_clause 0x3
	buffer_store_b16 v116 /*v372*/, v115 /*v371*/, s[0:3], null offen
	buffer_store_b16 v117 /*v373*/, v118 /*v374*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v106 /*v362*/, v2, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x4041
	v_cvt_pk_bf16_f32 v106 /*v362*/, v189 /*v445*/, s0
	s_set_vgpr_msb 0x4140
	s_clause 0x1
	buffer_store_b16 v107 /*v363*/, v0, s[0:3], null offen offset:128
	buffer_store_b16 v108 /*v364*/, v4, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x4041
	v_cvt_pk_bf16_f32 v107 /*v363*/, v190 /*v446*/, s0
	buffer_store_b16 v106 /*v362*/, v122 /*v378*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v106.l /*v362.l*/, v109.l /*v365.l*/
	v_mov_b16_e64 v109.l /*v365.l*/, v110.l /*v366.l*/
	v_cvt_pk_bf16_f32 v108 /*v364*/, v191 /*v447*/, s0
	s_clause 0x1
	buffer_store_b16 v107 /*v363*/, v125 /*v381*/, s[0:3], null offen offset:128
	buffer_store_b16 v108 /*v364*/, v130 /*v386*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v107 /*v363*/, 0xc0, v3
	s_clause 0x1
	buffer_store_b16 v106 /*v362*/, v124 /*v380*/, s[0:3], null offen offset:128
	buffer_store_b16 v109 /*v365*/, v127 /*v383*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v106 /*v362*/, 0xc0, v1
	s_clause 0x1
	buffer_store_b16 v98 /*v354*/, v106 /*v362*/, s[0:3], null offen
	buffer_store_b16 v99 /*v355*/, v107 /*v363*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v98 /*v354*/, v78 /*v334*/, s0
	v_or_b32_e32 v108 /*v364*/, 0xc0, v5
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v99 /*v355*/, v79 /*v335*/, s0
	v_or_b32_e32 v109 /*v365*/, 0xc0, v8
	s_clause 0x1
	buffer_store_b16 v100 /*v356*/, v108 /*v364*/, s[0:3], null offen
	buffer_store_b16 v101 /*v357*/, v109 /*v365*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v100 /*v356*/, 0xc0, v7
	s_clause 0x1
	buffer_store_b16 v98 /*v354*/, v100 /*v356*/, s[0:3], null offen
	buffer_store_b16 v99 /*v355*/, v102 /*v358*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v98 /*v354*/, 0xc0, v6
	s_wait_xcnt 0x0
	v_mov_b16_e64 v99.l /*v355.l*/, v104.l /*v360.l*/
	v_cvt_pk_bf16_f32 v101 /*v357*/, v80 /*v336*/, s0
	s_clause 0x5
	buffer_store_b16 v101 /*v357*/, v103 /*v359*/, s[0:3], null offen
	buffer_store_b16 v99 /*v355*/, v98 /*v354*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v90 /*v346*/, v2, s[0:3], null offen offset:256
	buffer_store_b16 v91 /*v347*/, v0, s[0:3], null offen offset:256
	buffer_store_b16 v92 /*v348*/, v4, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x4041
	v_mov_b16_e64 v90.l /*v346.l*/, v94.l /*v350.l*/
	buffer_store_b16 v93 /*v349*/, v122 /*v378*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v93 /*v349*/, v217 /*v985*/, s0
	v_cvt_pk_bf16_f32 v92 /*v348*/, v216 /*v984*/, s0
	s_set_vgpr_msb 0x43c0
	v_mov_b64_e32 v[216:217] /*v[984:985]*/, v[106:107]
	s_set_vgpr_msb 0xc041
	buffer_store_b16 v90 /*v346*/, v125 /*v381*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v91 /*v347*/, v215 /*v983*/, s0
	s_set_vgpr_msb 0x43c0
	v_mov_b64_e32 v[214:215] /*v[982:983]*/, v[104:105]
	s_set_vgpr_msb 0xc000
	v_cvt_pk_bf16_f32 v104, v67, s0
	s_set_vgpr_msb 0x41
	v_mov_b16_e64 v90.l /*v346.l*/, v91.l /*v347.l*/
	v_mov_b16_e64 v91.l /*v347.l*/, v92.l /*v348.l*/
	s_clause 0x1
	buffer_store_b16 v90 /*v346*/, v130 /*v386*/, s[0:3], null offen offset:256
	buffer_store_b16 v91 /*v347*/, v124 /*v380*/, s[0:3], null offen offset:256
	s_wait_xcnt 0x1
	v_or_b32_e32 v90 /*v346*/, 0x140, v5
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v107, v85, s0
	v_cvt_pk_bf16_f32 v85, v55, s0
	s_set_vgpr_msb 0x41
	v_or_b32_e32 v91 /*v347*/, 0x140, v8
	v_mov_b16_e64 v92.l /*v348.l*/, v93.l /*v349.l*/
	v_or_b32_e32 v93 /*v349*/, 0x140, v1
	s_clause 0x1
	buffer_store_b16 v92 /*v348*/, v127 /*v383*/, s[0:3], null offen offset:256
	buffer_store_b16 v82 /*v338*/, v93 /*v349*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v92 /*v348*/, 0x140, v7
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v106, v84, s0
	v_cvt_pk_bf16_f32 v84, v54, s0
	s_set_vgpr_msb 0x41
	v_or_b32_e32 v82 /*v338*/, 0x140, v3
	s_clause 0x1
	buffer_store_b16 v83 /*v339*/, v82 /*v338*/, s[0:3], null offen
	buffer_store_b16 v84 /*v340*/, v90 /*v346*/, s[0:3], null offen
	s_set_vgpr_msb 0x4144
	v_or_b32_e32 v83 /*v339*/, 0x140, v114 /*v370*/
	s_set_vgpr_msb 0x4400
	v_or_b32_e32 v1, 0x1c0, v1
	s_set_vgpr_msb 0x41
	s_clause 0x1
	buffer_store_b16 v85 /*v341*/, v91 /*v347*/, s[0:3], null offen
	buffer_store_b16 v86 /*v342*/, v92 /*v348*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v85 /*v341*/, v65 /*v321*/, s0
	v_cvt_pk_bf16_f32 v84 /*v340*/, v64 /*v320*/, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v86 /*v342*/, 0x140, v9
	v_cvt_pk_bf16_f32 v82 /*v338*/, v63 /*v319*/, s0
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v3, 0x1c0, v3
	v_cvt_pk_bf16_f32 v101, v63, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v63, v175 /*v943*/, s0
	s_set_vgpr_msb 0x341
	s_clause 0x4
	buffer_store_b16 v82 /*v338*/, v83 /*v339*/, s[0:3], null offen
	buffer_store_b16 v84 /*v340*/, v86 /*v342*/, s[0:3], null offen
	buffer_store_b16 v85 /*v341*/, v87 /*v343*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v74 /*v330*/, v2, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x4041
	v_mov_b16_e64 v74.l /*v330.l*/, v75.l /*v331.l*/
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v100, v62, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v62, v174 /*v942*/, s0
	s_set_vgpr_msb 0x341
	v_cvt_pk_bf16_f32 v75 /*v331*/, v38 /*v294*/, s0
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v74 /*v330*/, v0, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x4041
	v_cvt_pk_bf16_f32 v38 /*v294*/, v24 /*v280*/, s0
	s_set_vgpr_msb 0x4101
	v_cvt_pk_bf16_f32 v2, v37 /*v293*/, s0
	s_set_vgpr_msb 0x140
	buffer_store_b16 v76 /*v332*/, v4, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x4001
	v_mov_b16_e64 v0.l, v75.l /*v331.l*/
	v_mov_b16_e64 v4.l, v77.l /*v333.l*/
	buffer_store_b16 v2, v122 /*v378*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v40 /*v296*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v40 /*v296*/, v27 /*v283*/, s0
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v27 /*v283*/, v147 /*v915*/, s0
	s_set_vgpr_msb 0x4301
	buffer_store_b16 v0, v125 /*v381*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v0, v41 /*v297*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v41 /*v297*/, v50 /*v306*/, s0
	v_cvt_pk_bf16_f32 v50 /*v306*/, v42 /*v298*/, s0
	s_set_vgpr_msb 0x4140
	v_cvt_pk_bf16_f32 v42 /*v298*/, v198, s0
	s_set_vgpr_msb 0x4001
	buffer_store_b16 v4, v130 /*v386*/, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v4, v252, s0
	v_cvt_pk_bf16_f32 v252, v71, s0
	s_set_vgpr_msb 1
	s_clause 0x1
	buffer_store_b16 v2, v124 /*v380*/, s[0:3], null offen offset:384
	buffer_store_b16 v0, v127 /*v383*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_or_b32_e32 v2, 0x1c0, v5
	v_or_b32_e32 v5, 0x1c0, v7
	s_set_vgpr_msb 0x108
	v_or_b32_e32 v7, s35, v130 /*v642*/
	s_clause 0x2
	buffer_store_b16 v4, v1, s[0:3], null offen
	s_set_vgpr_msb 0x840
	buffer_store_b16 v58 /*v314*/, v3, s[0:3], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v4, 0x1c0, v8
	v_cvt_pk_bf16_f32 v0, v254, s0
	s_set_vgpr_msb 64
	v_or_b32_e32 v35 /*v291*/, 5, v7
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v254, v72, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v3, v0 /*v256*/, s0
	s_set_vgpr_msb 0x140
	v_cvt_pk_bf16_f32 v0 /*v256*/, v75, s0
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v1, v255, s0
	s_clause 0x1
	buffer_store_b16 v0, v2, s[0:3], null offen
	buffer_store_b16 v1, v4, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v0, v1 /*v257*/, s0
	s_set_vgpr_msb 0x141
	v_mul_lo_u32 v35 /*v291*/, v35 /*v291*/, s7
	s_set_vgpr_msb 0x4104
	buffer_store_b16 v3, v5, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v114 /*v370*/
	s_set_vgpr_msb 0x401
	v_or_b32_e32 v4, 0x1c0, v9
	v_cvt_pk_bf16_f32 v9, v22 /*v278*/, s0
	s_set_vgpr_msb 0x141
	v_cvt_pk_bf16_f32 v22 /*v278*/, v70 /*v326*/, s0
	s_set_vgpr_msb 0x4100
	v_mul_lo_u32 v1, v7, s7
	buffer_store_b16 v0, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v0, v3 /*v259*/, s0
	v_cvt_pk_bf16_f32 v2, v2 /*v258*/, s0
	s_set_vgpr_msb 0x141
	v_add_lshl_u32 v35 /*v291*/, v35 /*v291*/, s38, 7
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v255, v73, s0
	v_mov_b64_e32 v[72:73], v[232:233]
	v_add_lshl_u32 v1, s38, v1, 7
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_set_vgpr_msb 0x49
	v_or_b32_e32 v58 /*v314*/, v35 /*v291*/, v201 /*v713*/
	s_set_vgpr_msb 0x4940
	v_cvt_pk_bf16_f32 v2 /*v258*/, v21, s0
	s_set_vgpr_msb 0x4008
	v_or_b32_e32 v2, v1, v201 /*v713*/
	s_set_vgpr_msb 0x849
	v_or_b32_e32 v35 /*v291*/, v35 /*v291*/, v198 /*v710*/
	s_set_vgpr_msb 0x4944
	v_lshlrev_b32_e32 v58 /*v314*/, 2, v58 /*v314*/
	s_set_vgpr_msb 0x4401
	v_or_b32_e32 v4, 0x1c0, v6
	v_dual_lshlrev_b32 v2, 2, v2 :: v_dual_bitop2_b32 v5, 1, v7 bitop3:0x54
	v_cvt_pk_bf16_f32 v6, v20 /*v276*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v0, v4, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v21 /*v277*/, s0
	v_mul_lo_u32 v3, s7, v5
	s_set_vgpr_msb 0x144
	v_lshlrev_b32_e32 v35 /*v291*/, 2, v35 /*v291*/
	s_set_vgpr_msb 0x4408
	v_or_b32_e32 v1, v1, v198 /*v710*/
	s_set_vgpr_msb 0x841
	v_cvt_pk_bf16_f32 v21 /*v277*/, v69 /*v325*/, s0
	s_set_vgpr_msb 0x4100
	buffer_store_b16 v6, v2, s[0:3], null offen
	v_or_b32_e32 v8, 3, v7
	s_set_vgpr_msb 0x41
	v_cvt_pk_bf16_f32 v20 /*v276*/, v68 /*v324*/, s0
	s_set_vgpr_msb 0x4100
	v_lshlrev_b32_e32 v1, 2, v1
	v_add_lshl_u32 v3, v3, s38, 7
	v_mul_lo_u32 v8, v8, s7
	s_set_vgpr_msb 64
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v59 /*v315*/, 64, v1
	s_set_vgpr_msb 0x4008
	v_or_b32_e32 v0, v3, v201 /*v713*/
	v_or_b32_e32 v3, v3, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v5, 2, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v0, 2, v0
	v_add_lshl_u32 v8, v8, s38, 7
	v_mul_lo_u32 v5, v5, s7
	buffer_store_b16 v4, v0, s[0:3], null offen
	v_add_lshl_u32 v5, v5, s38, 7
	s_set_vgpr_msb 0x48
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v34 /*v290*/, v5, v201 /*v713*/
	s_set_vgpr_msb 0x4808
	v_or_b32_e32 v5, v5, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v6, 4, v7
	s_set_vgpr_msb 4
	v_lshlrev_b32_e32 v4, 2, v34 /*v290*/
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v5, 2, v5
	v_mul_lo_u32 v6, v6, s7
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v34 /*v290*/, v8, v201 /*v713*/
	s_set_vgpr_msb 0x4808
	v_or_b32_e32 v8, v8, v198 /*v710*/
	buffer_store_b16 v9, v4, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v9, 6, v7
	s_set_vgpr_msb 64
	v_or_b32_e32 v60 /*v316*/, 64, v5
	s_set_vgpr_msb 0x4044
	v_lshlrev_b32_e32 v34 /*v290*/, 2, v34 /*v290*/
	s_set_vgpr_msb 0x4400
	v_lshlrev_b32_e32 v8, 2, v8
	v_add_lshl_u32 v6, v6, s38, 7
	v_mul_lo_u32 v9, v9, s7
	s_set_vgpr_msb 64
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v61 /*v317*/, 64, v8
	s_set_vgpr_msb 0x4048
	v_or_b32_e32 v37 /*v293*/, v6, v201 /*v713*/
	s_set_vgpr_msb 0x4808
	v_or_b32_e32 v6, v6, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v7, 7, v7
	v_add_lshl_u32 v9, v9, s38, 7
	s_set_vgpr_msb 0x44
	v_lshlrev_b32_e32 v37 /*v293*/, 2, v37 /*v293*/
	s_set_vgpr_msb 0x4400
	v_lshlrev_b32_e32 v6, 2, v6
	v_mul_lo_u32 v7, v7, s7
	s_set_vgpr_msb 0x41
	s_clause 0x1
	buffer_store_b16 v36 /*v292*/, v34 /*v290*/, s[0:3], null offen
	buffer_store_b16 v38 /*v294*/, v37 /*v293*/, s[0:3], null offen
	s_set_vgpr_msb 0x4148
	v_or_b32_e32 v36 /*v292*/, v9, v201 /*v713*/
	s_set_vgpr_msb 0x4841
	buffer_store_b16 v39 /*v295*/, v58 /*v314*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v38 /*v294*/, v26 /*v282*/, s0
	s_set_vgpr_msb 0x4108
	v_or_b32_e32 v9, v9, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v3, 2, v3
	v_add_lshl_u32 v7, v7, s38, 7
	s_set_vgpr_msb 0x47
	v_lshlrev_b32_e32 v36 /*v292*/, 2, v36 /*v292*/
	v_cvt_pk_bf16_f32 v26 /*v282*/, v146 /*v914*/, s0
	s_set_vgpr_msb 0x4700
	v_lshlrev_b32_e32 v9, 2, v9
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v39 /*v295*/, v7, v201 /*v713*/
	s_set_vgpr_msb 0x4808
	v_or_b32_e32 v7, v7, v198 /*v710*/
	s_set_vgpr_msb 0x845
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v39 /*v295*/, 2, v39 /*v295*/
	s_clause 0x2
	buffer_store_b16 v38 /*v294*/, v36 /*v292*/, s[0:3], null offen
	buffer_store_b16 v40 /*v296*/, v39 /*v295*/, s[0:3], null offen
	buffer_store_b16 v41 /*v297*/, v59 /*v315*/, s[0:3], null offen
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v38 /*v294*/, v51 /*v307*/, s0
	s_set_vgpr_msb 0x4541
	v_or_b32_e32 v40 /*v296*/, 64, v3
	v_cvt_pk_bf16_f32 v41 /*v297*/, v52 /*v308*/, s0
	v_cvt_pk_bf16_f32 v59 /*v315*/, v53 /*v309*/, s0
	s_set_vgpr_msb 0x4100
	v_lshlrev_b32_e32 v7, 2, v7
	s_set_vgpr_msb 0x41
	v_cvt_pk_bf16_f32 v51 /*v307*/, v43 /*v299*/, s0
	s_clause 0x1
	buffer_store_b16 v38 /*v294*/, v40 /*v296*/, s[0:3], null offen
	buffer_store_b16 v41 /*v297*/, v60 /*v316*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v38 /*v294*/, v54 /*v310*/, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v41 /*v297*/, 64, v6
	v_cvt_pk_bf16_f32 v60 /*v316*/, v56 /*v312*/, s0
	buffer_store_b16 v59 /*v315*/, v61 /*v317*/, s[0:3], null offen
	v_cvt_pk_bf16_f32 v40 /*v296*/, v55 /*v311*/, s0
	s_set_vgpr_msb 0x4145
	v_or_b32_e32 v59 /*v315*/, 64, v35 /*v291*/
	s_clause 0x1
	buffer_store_b16 v38 /*v294*/, v41 /*v297*/, s[0:3], null offen
	buffer_store_b16 v40 /*v296*/, v59 /*v315*/, s[0:3], null offen
	s_set_vgpr_msb 0x4541
	v_or_b32_e32 v38 /*v294*/, 64, v9
	v_mov_b16_e64 v40.l /*v296.l*/, v60.l /*v316.l*/
	v_cvt_pk_bf16_f32 v41 /*v297*/, v57 /*v313*/, s0
	v_or_b32_e32 v59 /*v315*/, 64, v7
	v_cvt_pk_bf16_f32 v52 /*v308*/, v44 /*v300*/, s0
	v_or_b32_e32 v43 /*v299*/, 0xc0, v3
	buffer_store_b16 v40 /*v296*/, v38 /*v294*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v38.l /*v294.l*/, v51.l /*v307.l*/
	s_clause 0x2
	buffer_store_b16 v41 /*v297*/, v59 /*v315*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v50 /*v306*/, v2, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x4041
	v_mov_b16_e64 v40.l /*v296.l*/, v52.l /*v308.l*/
	v_cvt_pk_bf16_f32 v41 /*v297*/, v45 /*v301*/, s0
	v_cvt_pk_bf16_f32 v50 /*v306*/, v48 /*v304*/, s0
	v_cvt_pk_bf16_f32 v51 /*v307*/, v49 /*v305*/, s0
	s_set_vgpr_msb 0x4140
	s_clause 0x1
	buffer_store_b16 v38 /*v294*/, v0, s[0:3], null offen offset:128
	buffer_store_b16 v40 /*v296*/, v4, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x4041
	v_cvt_pk_bf16_f32 v38 /*v294*/, v46 /*v302*/, s0
	v_cvt_pk_bf16_f32 v40 /*v296*/, v47 /*v303*/, s0
	buffer_store_b16 v41 /*v297*/, v34 /*v290*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v41.l /*v297.l*/, v50.l /*v306.l*/
	v_mov_b16_e64 v50.l /*v306.l*/, v51.l /*v307.l*/
	s_clause 0x1
	buffer_store_b16 v38 /*v294*/, v37 /*v293*/, s[0:3], null offen offset:128
	buffer_store_b16 v40 /*v296*/, v58 /*v314*/, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x4140
	v_cvt_pk_bf16_f32 v38 /*v294*/, v196, s0
	v_or_b32_e32 v40 /*v296*/, 0xc0, v1
	v_or_b32_e32 v45 /*v301*/, 0xc0, v5
	s_set_vgpr_msb 0x4041
	s_clause 0x1
	buffer_store_b16 v41 /*v297*/, v36 /*v292*/, s[0:3], null offen offset:128
	buffer_store_b16 v50 /*v306*/, v39 /*v295*/, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x4140
	v_cvt_pk_bf16_f32 v41 /*v297*/, v197, s0
	v_cvt_pk_bf16_f32 v44 /*v300*/, v199, s0
	v_or_b32_e32 v50 /*v306*/, 0xc0, v8
	s_set_vgpr_msb 0x4041
	s_clause 0x1
	buffer_store_b16 v38 /*v294*/, v40 /*v296*/, s[0:3], null offen
	buffer_store_b16 v41 /*v297*/, v43 /*v299*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	v_cvt_pk_bf16_f32 v38 /*v294*/, v200, s0
	s_set_vgpr_msb 0x4041
	s_clause 0x1
	buffer_store_b16 v42 /*v298*/, v45 /*v301*/, s[0:3], null offen
	buffer_store_b16 v44 /*v300*/, v50 /*v306*/, s[0:3], null offen
	s_wait_xcnt 0x2
	v_or_b32_e32 v41 /*v297*/, 0xc0, v6
	s_set_vgpr_msb 0x4144
	v_cvt_pk_bf16_f32 v45 /*v301*/, v203, s0
	v_cvt_pk_bf16_f32 v40 /*v296*/, v201, s0
	v_or_b32_e32 v43 /*v299*/, 0xc0, v35 /*v291*/
	v_cvt_pk_bf16_f32 v42 /*v298*/, v202, s0
	s_set_vgpr_msb 0x4441
	v_or_b32_e32 v44 /*v300*/, 0xc0, v9
	s_clause 0x1
	buffer_store_b16 v38 /*v294*/, v41 /*v297*/, s[0:3], null offen
	buffer_store_b16 v40 /*v296*/, v43 /*v299*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v38 /*v294*/, 0xc0, v7
	s_wait_xcnt 0x0
	v_mov_b16_e64 v40.l /*v296.l*/, v45.l /*v301.l*/
	buffer_store_b16 v42 /*v298*/, v44 /*v300*/, s[0:3], null offen
	v_or_b32_e32 v23 /*v279*/, 0x140, v7
	s_set_vgpr_msb 0x4103
	v_cvt_pk_bf16_f32 v202, v178 /*v946*/, s0
	v_cvt_pk_bf16_f32 v203, v179 /*v947*/, s0
	s_set_vgpr_msb 0x341
	s_clause 0x4
	buffer_store_b16 v40 /*v296*/, v38 /*v294*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v26 /*v282*/, v2, s[0:3], null offen offset:256
	buffer_store_b16 v27 /*v283*/, v0, s[0:3], null offen offset:256
	buffer_store_b16 v28 /*v284*/, v4, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x4041
	v_mov_b16_e64 v26.l /*v282.l*/, v30.l /*v286.l*/
	buffer_store_b16 v29 /*v285*/, v34 /*v290*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v27 /*v283*/, v151 /*v919*/, s0
	v_cvt_pk_bf16_f32 v28 /*v284*/, v152 /*v920*/, s0
	v_cvt_pk_bf16_f32 v29 /*v285*/, v153 /*v921*/, s0
	s_set_vgpr_msb 0x4341
	buffer_store_b16 v26 /*v282*/, v37 /*v293*/, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v26.l /*v282.l*/, v27.l /*v283.l*/
	v_mov_b16_e64 v27.l /*v283.l*/, v28.l /*v284.l*/
	v_mov_b16_e64 v28.l /*v284.l*/, v29.l /*v285.l*/
	v_or_b32_e32 v29 /*v285*/, 0x140, v1
	s_clause 0x3
	buffer_store_b16 v26 /*v282*/, v58 /*v314*/, s[0:3], null offen offset:256
	buffer_store_b16 v27 /*v283*/, v36 /*v292*/, s[0:3], null offen offset:256
	buffer_store_b16 v28 /*v284*/, v39 /*v295*/, s[0:3], null offen offset:256
	buffer_store_b16 v18 /*v274*/, v29 /*v285*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v18 /*v274*/, 0x140, v3
	v_or_b32_e32 v27 /*v283*/, 0x140, v8
	v_or_b32_e32 v26 /*v282*/, 0x140, v5
	v_or_b32_e32 v28 /*v284*/, 0x140, v6
	s_clause 0x1
	buffer_store_b16 v19 /*v275*/, v18 /*v274*/, s[0:3], null offen
	buffer_store_b16 v20 /*v276*/, v26 /*v282*/, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v18 /*v274*/, v71 /*v327*/, s0
	s_clause 0x1
	buffer_store_b16 v21 /*v277*/, v27 /*v283*/, s[0:3], null offen
	buffer_store_b16 v22 /*v278*/, v28 /*v284*/, s[0:3], null offen
	s_set_vgpr_msb 0x4145
	v_or_b32_e32 v19 /*v275*/, 0x140, v35 /*v291*/
	v_cvt_pk_bf16_f32 v20 /*v276*/, v72 /*v328*/, s0
	v_cvt_pk_bf16_f32 v21 /*v277*/, v73 /*v329*/, s0
	s_set_vgpr_msb 0x4540
	v_or_b32_e32 v22 /*v278*/, 0x140, v9
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v1, 0x1c0, v1
	s_set_vgpr_msb 0x41
	buffer_store_b16 v18 /*v274*/, v19 /*v275*/, s[0:3], null offen
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v3, 0x1c0, v3
	s_set_vgpr_msb 0x41
	s_clause 0x1
	scratch_load_b128 v[66:69] /*v[322:325]*/, off, off offset:96 th:TH_LOAD_LU nv
	scratch_load_b128 v[70:73] /*v[326:329]*/, off, off offset:112 th:TH_LOAD_LU nv
	s_clause 0x3
	buffer_store_b16 v20 /*v276*/, v22 /*v278*/, s[0:3], null offen
	buffer_store_b16 v21 /*v277*/, v23 /*v279*/, s[0:3], null offen
	s_set_vgpr_msb 0x4140
	buffer_store_b16 v10 /*v266*/, v2, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x4041
	v_mov_b16_e64 v10.l /*v266.l*/, v11.l /*v267.l*/
	s_set_vgpr_msb 0x4143
	v_cvt_pk_bf16_f32 v11 /*v267*/, v196 /*v964*/, s0
	s_set_vgpr_msb 0x4303
	v_cvt_pk_bf16_f32 v2, v195 /*v963*/, s0
	s_set_vgpr_msb 0x340
	buffer_store_b16 v10 /*v266*/, v0, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x4001
	v_mov_b16_e64 v0.l, v11.l /*v267.l*/
	s_set_vgpr_msb 0x140
	buffer_store_b16 v12 /*v268*/, v4, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x4001
	v_mov_b16_e64 v4.l, v13.l /*v269.l*/
	buffer_store_b16 v2, v34 /*v290*/, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v2, v198 /*v966*/, s0
	s_set_vgpr_msb 0x301
	buffer_store_b16 v0, v37 /*v293*/, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v0, v199 /*v967*/, s0
	s_set_vgpr_msb 0x301
	s_clause 0x2
	buffer_store_b16 v4, v58 /*v314*/, s[0:3], null offen offset:384
	buffer_store_b16 v2, v36 /*v292*/, s[0:3], null offen offset:384
	buffer_store_b16 v0, v39 /*v295*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_or_b32_e32 v2, 0x1c0, v5
	v_or_b32_e32 v5, 0x1c0, v6
	s_set_vgpr_msb 0x108
	v_or_b32_e32 v6, s34, v130 /*v642*/
	v_cvt_pk_bf16_f32 v4, v20, s0
	v_cvt_pk_bf16_f32 v0, v22, s0
	s_clause 0x2
	buffer_store_b16 v4, v1, s[0:3], null offen
	s_set_vgpr_msb 0x840
	buffer_store_b16 v2 /*v258*/, v3, s[0:3], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v251, 5, v6
	v_cvt_pk_bf16_f32 v3, v24, s0
	v_cvt_pk_bf16_f32 v1, v23, s0
	v_or_b32_e32 v4, 0x1c0, v8
	s_clause 0x1
	buffer_store_b16 v0, v2, s[0:3], null offen
	buffer_store_b16 v1, v4, s[0:3], null offen
	v_mul_lo_u32 v251, v251, s7
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v0, v25, s0
	buffer_store_b16 v3, v5, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 1, v6
	v_cvt_pk_bf16_f32 v2, v26, s0
	v_or_b32_e32 v4, 0x1c0, v9
	v_mul_lo_u32 v1, v6, s7
	v_or_b32_e32 v8, 3, v6
	v_add_lshl_u32 v251, v251, s38, 7
	v_cvt_pk_bf16_f32 v9, v70, s0
	v_mov_b64_e32 v[70:71], v[230:231]
	s_set_vgpr_msb 0xc0
	s_clause 0x1
	scratch_load_b128 v[192:195] /*v[960:963]*/, off, off offset:64 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199] /*v[964:967]*/, off, off offset:80 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc000
	v_mul_lo_u32 v8, v8, s7
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v2 /*v258*/, v251, v201 /*v713*/
	s_set_vgpr_msb 0x4800
	v_add_lshl_u32 v1, s38, v1, 7
	v_cvt_pk_bf16_f32 v236, v70, s0
	s_set_vgpr_msb 0x44
	v_lshlrev_b32_e32 v2 /*v258*/, 2, v2 /*v258*/
	s_set_vgpr_msb 0x4404
	v_or_b32_e32 v3, 0x1c0, v35 /*v291*/
	v_add_lshl_u32 v8, v8, s38, 7
	buffer_store_b16 v0, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v3, v5, s7
	v_cvt_pk_bf16_f32 v0, v27, s0
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v4, 0x1c0, v7
	s_set_vgpr_msb 8
	v_or_b32_e32 v2, v1, v201 /*v713*/
	v_cvt_pk_bf16_f32 v7, v68, s0
	v_or_b32_e32 v1, v1, v198 /*v710*/
	s_clause 0x1
	scratch_load_b128 v[20:23], off, off offset:160 th:TH_LOAD_LU nv
	scratch_load_b128 v[24:27], off, off offset:176 th:TH_LOAD_LU nv
	v_add_lshl_u32 v3, v3, s38, 7
	buffer_store_b16 v0, v4, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v2, 2, v2
	v_cvt_pk_bf16_f32 v4, v69, s0
	v_lshlrev_b32_e32 v1, 2, v1
	s_set_vgpr_msb 8
	v_or_b32_e32 v0, v3, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v5, 2, v6
	buffer_store_b16 v7, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 4, v6
	s_set_vgpr_msb 8
	v_or_b32_e32 v3, v3, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v0, 2, v0
	v_mul_lo_u32 v5, v5, s7
	s_set_vgpr_msb 64
	v_or_b32_e32 v1 /*v257*/, 64, v1
	s_set_vgpr_msb 0x4000
	v_mul_lo_u32 v7, v7, s7
	v_lshlrev_b32_e32 v3, 2, v3
	buffer_store_b16 v4, v0, s[0:3], null offen
	v_mov_b64_e32 v[68:69], v[228:229]
	v_add_lshl_u32 v5, v5, s38, 7
	v_add_lshl_u32 v7, v7, s38, 7
	s_set_vgpr_msb 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_or_b32_e32 v250, v5, v201 /*v713*/
	v_or_b32_e32 v5, v5, v198 /*v710*/
	v_or_b32_e32 v253, v7, v201 /*v713*/
	v_or_b32_e32 v7, v7, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v4, 2, v250
	s_set_vgpr_msb 8
	v_or_b32_e32 v250, v8, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshlrev_b32_e32 v253, 2, v253
	s_set_vgpr_msb 8
	v_or_b32_e32 v8, v8, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v250, 2, v250 :: v_dual_bitop2_b32 v243, 64, v3 bitop3:0x54
	buffer_store_b16 v9, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 6, v6
	v_or_b32_e32 v6, 7, v6
	s_clause 0x3
	buffer_store_b16 v252, v250, s[0:3], null offen
	buffer_store_b16 v254, v253, s[0:3], null offen
	s_set_vgpr_msb 1
	buffer_store_b16 v255, v2 /*v258*/, s[0:3], null offen
	v_mul_lo_u32 v9, s7, v9
	v_mul_lo_u32 v6, s7, v6
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v254, v74, s0
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v7, 2, v7
	v_mov_b64_e32 v[74:75], v[234:235]
	v_cvt_pk_bf16_f32 v234, v68, s0
	v_cvt_pk_bf16_f32 v235, v69, s0
	v_add_lshl_u32 v9, v9, s38, 7
	v_add_lshl_u32 v6, v6, s38, 7
	v_cvt_pk_bf16_f32 v228, v214, s0
	v_cvt_pk_bf16_f32 v237, v74, s0
	v_cvt_pk_bf16_f32 v238, v75, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v252, v9, v201 /*v713*/
	v_or_b32_e32 v255, v6, v201 /*v713*/
	v_or_b32_e32 v9, v9, v198 /*v710*/
	v_or_b32_e32 v6, v6, v198 /*v710*/
	v_cvt_pk_bf16_f32 v229, v215, s0
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v252, 2, v252 :: v_dual_lshlrev_b32 v255, 2, v255
	s_clause 0x4
	buffer_store_b16 v254, v252, s[0:3], null offen
	s_set_vgpr_msb 64
	buffer_store_b16 v0 /*v256*/, v255, s[0:3], null offen
	s_set_vgpr_msb 0x4001
	buffer_store_b16 v242, v1 /*v257*/, s[0:3], null offen
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v242, v235 /*v1003*/, s0
	v_or_b32_e32 v254, 64, v5
	s_set_vgpr_msb 0x340
	v_or_b32_e32 v0 /*v256*/, 64, v8
	s_set_vgpr_msb 0x4000
	v_dual_lshlrev_b32 v9, 2, v9 :: v_dual_lshlrev_b32 v6, 2, v6
	buffer_store_b16 v242, v243, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v242, v251, v198 /*v710*/
	s_clause 0x2
	buffer_store_b16 v244, v254, s[0:3], null offen
	s_set_vgpr_msb 0x801
	buffer_store_b16 v245, v0 /*v256*/, s[0:3], null offen
	s_set_vgpr_msb 0x103
	v_cvt_pk_bf16_f32 v243, v238 /*v1006*/, s0
	v_dual_lshlrev_b32 v242, 2, v242 :: v_dual_bitop2_b32 v245, 64, v7 bitop3:0x54
	v_cvt_pk_bf16_f32 v244, v239 /*v1007*/, s0
	s_set_vgpr_msb 0x300
	v_cvt_pk_bf16_f32 v232, v219, s0
	v_or_b32_e32 v231, 0xc0, v9
	v_cvt_pk_bf16_f32 v219, v29, s0
	v_or_b32_e32 v246, 64, v242
	s_clause 0x1
	buffer_store_b16 v243, v245, s[0:3], null offen
	buffer_store_b16 v244, v246, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v243, 64, v9
	s_wait_xcnt 0x0
	v_mov_b16_e64 v244.l, v247.l
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v245, v241 /*v1009*/, s0
	v_or_b32_e32 v246, 64, v6
	v_or_b32_e32 v230, 0xc0, v242
	s_set_vgpr_msb 0x302
	v_cvt_pk_bf16_f32 v214, v158 /*v670*/, s0
	s_set_vgpr_msb 0x200
	s_clause 0x2
	buffer_store_b16 v244, v243, s[0:3], null offen
	buffer_store_b16 v245, v246, s[0:3], null offen
	buffer_store_b16 v234, v2, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v234, v71, s0
	s_clause 0x1
	buffer_store_b16 v235, v0, s[0:3], null offen offset:128
	buffer_store_b16 v236, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v235, v72, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v236, v73, s0
	v_or_b32_e32 v215, 0x140, v6
	buffer_store_b16 v234, v250, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v234.l, v237.l
	v_mov_b16_e64 v237.l, v238.l
	s_clause 0x5
	buffer_store_b16 v235, v253, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	buffer_store_b16 v236, v2 /*v258*/, s[0:3], null offen offset:128
	s_set_vgpr_msb 0x100
	buffer_store_b16 v234, v252, s[0:3], null offen offset:128
	buffer_store_b16 v237, v255, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v234, 0xc0, v1
	v_or_b32_e32 v236, 0xc0, v5
	v_or_b32_e32 v235, 0xc0, v3
	s_wait_xcnt 0x0
	v_or_b32_e32 v237, 0xc0, v8
	s_clause 0x1
	buffer_store_b16 v226, v234, s[0:3], null offen
	buffer_store_b16 v227, v235, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v226, v216, s0
	s_clause 0x1
	buffer_store_b16 v228, v236, s[0:3], null offen
	buffer_store_b16 v229, v237, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v228, 0xc0, v7
	v_cvt_pk_bf16_f32 v227, v217, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v229, v218, s0
	s_clause 0x1
	buffer_store_b16 v226, v228, s[0:3], null offen
	buffer_store_b16 v227, v230, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v226, 0xc0, v6
	s_wait_xcnt 0x0
	v_mov_b16_e64 v227.l, v232.l
	v_cvt_pk_bf16_f32 v218, v28, s0
	buffer_store_b16 v229, v231, s[0:3], null offen
	v_mov_b64_e32 v[28:29], v[188:189]
	s_set_vgpr_msb 0xc0
	s_clause 0x1
	scratch_load_b128 v[234:237] /*v[1002:1005]*/, off, off th:TH_LOAD_LU nv
	scratch_load_b128 v[238:241] /*v[1006:1009]*/, off, off offset:16 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc000
	s_clause 0x3
	buffer_store_b16 v227, v226, s[0:3], null offen
	buffer_store_b16 v218, v2, s[0:3], null offen offset:256
	buffer_store_b16 v219, v0, s[0:3], null offen offset:256
	buffer_store_b16 v220, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e64 v218.l, v222.l
	buffer_store_b16 v221, v250, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v219, v33, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v220, v34, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v221, v35, s0
	buffer_store_b16 v218, v253, s[0:3], null offen offset:256
	v_mov_b64_e32 v[32:33], v[192:193]
	s_wait_xcnt 0x0
	v_mov_b16_e64 v218.l, v219.l
	v_mov_b16_e64 v219.l, v220.l
	v_mov_b16_e64 v220.l, v221.l
	v_or_b32_e32 v221, 0x140, v1
	s_set_vgpr_msb 1
	s_clause 0x4
	buffer_store_b16 v218, v2 /*v258*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x100
	buffer_store_b16 v219, v252, s[0:3], null offen offset:256
	buffer_store_b16 v220, v255, s[0:3], null offen offset:256
	buffer_store_b16 v210, v221, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v210, 0x140, v3
	v_or_b32_e32 v219, 0x140, v8
	v_or_b32_e32 v218, 0x140, v5
	v_or_b32_e32 v220, 0x140, v7
	s_clause 0x1
	buffer_store_b16 v211, v210, s[0:3], null offen
	buffer_store_b16 v212, v218, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v210, v159 /*v671*/, s0
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_store_b16 v213, v219, s[0:3], null offen
	buffer_store_b16 v214, v220, s[0:3], null offen
	v_or_b32_e32 v211, 0x140, v242
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v212, v160 /*v672*/, s0
	v_cvt_pk_bf16_f32 v213, v161 /*v673*/, s0
	v_or_b32_e32 v214, 0x140, v9
	v_or_b32_e32 v1, 0x1c0, v1
	s_set_vgpr_msb 0x200
	buffer_store_b16 v210, v211, s[0:3], null offen
	v_mov_b64_e32 v[34:35], v[194:195]
	v_cvt_pk_bf16_f32 v194, v29, s0
	s_clause 0x2
	buffer_store_b16 v212, v214, s[0:3], null offen
	buffer_store_b16 v213, v215, s[0:3], null offen
	buffer_store_b16 v202, v2, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v202.l, v203.l
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v203, v182 /*v950*/, s0
	v_cvt_pk_bf16_f32 v2, v181 /*v949*/, s0
	v_or_b32_e32 v3, 0x1c0, v3
	s_set_vgpr_msb 0x300
	v_cvt_pk_bf16_f32 v188, v175, s0
	buffer_store_b16 v202, v0, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v0.l, v203.l
	buffer_store_b16 v204, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v4.l, v205.l
	buffer_store_b16 v2, v250, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v2, v184 /*v952*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v0, v253, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v0, v185 /*v953*/, s0
	s_set_vgpr_msb 0x301
	buffer_store_b16 v4, v2 /*v258*/, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v4, v28, s0
	s_clause 0x1
	buffer_store_b16 v2, v252, s[0:3], null offen offset:384
	buffer_store_b16 v0, v255, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v0, v30, s0
	v_or_b32_e32 v2, 0x1c0, v5
	v_or_b32_e32 v5, 0x1c0, v7
	s_set_vgpr_msb 8
	v_or_b32_e32 v7, s33, v130 /*v642*/
	s_clause 0x1
	buffer_store_b16 v4, v1, s[0:3], null offen
	buffer_store_b16 v194, v3, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v31, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v4, 0x1c0, v8
	s_clause 0x1
	buffer_store_b16 v0, v2, s[0:3], null offen
	buffer_store_b16 v1, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v1, v7, s7
	v_cvt_pk_bf16_f32 v3, v32, s0
	v_cvt_pk_bf16_f32 v0, v33, s0
	v_cvt_pk_bf16_f32 v2, v34, s0
	v_or_b32_e32 v4, 0x1c0, v9
	v_or_b32_e32 v8, 3, v7
	buffer_store_b16 v3, v5, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v242
	v_add_lshl_u32 v1, s38, v1, 7
	v_or_b32_e32 v187, 5, v7
	v_mul_lo_u32 v8, v8, s7
	v_cvt_pk_bf16_f32 v9, v174, s0
	s_clause 0x1
	buffer_store_b16 v0, v3, s[0:3], null offen
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v2, v1, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v5, 1, v7
	v_cvt_pk_bf16_f32 v0, v35, s0
	v_or_b32_e32 v4, 0x1c0, v6
	v_cvt_pk_bf16_f32 v6, v172, s0
	v_lshlrev_b32_e32 v2, 2, v2
	v_mul_lo_u32 v3, v5, s7
	v_or_b32_e32 v5, 2, v7
	buffer_store_b16 v0, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v173, s0
	buffer_store_b16 v6, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v6, 4, v7
	v_mul_lo_u32 v5, v5, s7
	v_mul_lo_u32 v187, v187, s7
	v_add_lshl_u32 v3, v3, s38, 7
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, v1, v198 /*v710*/
	v_mul_lo_u32 v6, v6, s7
	v_add_lshl_u32 v8, v8, s38, 7
	v_cvt_pk_bf16_f32 v191, v177, s0
	v_or_b32_e32 v0, v3, v201 /*v713*/
	v_add_lshl_u32 v5, v5, s38, 7
	v_add_lshl_u32 v187, v187, s38, 7
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v1, 2, v1
	v_cvt_pk_bf16_f32 v190, v176, s0
	v_lshlrev_b32_e32 v0, 2, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v186, v5, v201 /*v713*/
	v_add_lshl_u32 v6, v6, s38, 7
	v_or_b32_e32 v194, v187, v201 /*v713*/
	v_or_b32_e32 v3, v3, v198 /*v710*/
	buffer_store_b16 v4, v0, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v4, 2, v186
	s_set_vgpr_msb 8
	v_or_b32_e32 v186, v8, v201 /*v713*/
	v_or_b32_e32 v189, v6, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v194, 2, v194 :: v_dual_bitop2_b32 v193, 64, v1 bitop3:0x54
	buffer_store_b16 v9, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 6, v7
	v_dual_lshlrev_b32 v186, 2, v186 :: v_dual_bitop2_b32 v7, 7, v7 bitop3:0x54
	v_lshlrev_b32_e32 v189, 2, v189
	s_clause 0x1
	buffer_store_b16 v188, v186, s[0:3], null offen
	buffer_store_b16 v190, v189, s[0:3], null offen
	v_mul_lo_u32 v9, v9, s7
	v_mul_lo_u32 v7, v7, s7
	buffer_store_b16 v191, v194, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v190, v178, s0
	v_cvt_pk_bf16_f32 v178, v164, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v5, v5, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v3, 2, v3
	v_cvt_pk_bf16_f32 v192, v179, s0
	v_add_lshl_u32 v9, v9, s38, 7
	v_add_lshl_u32 v7, v7, s38, 7
	s_set_vgpr_msb 8
	v_or_b32_e32 v8, v8, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v5, 2, v5 :: v_dual_bitop2_b32 v179, 64, v3 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v188, v9, v201 /*v713*/
	v_or_b32_e32 v191, v7, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v8, 2, v8
	s_set_vgpr_msb 8
	v_or_b32_e32 v6, v6, v198 /*v710*/
	v_or_b32_e32 v9, v9, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v188, 2, v188 :: v_dual_lshlrev_b32 v191, 2, v191
	s_clause 0x2
	buffer_store_b16 v190, v188, s[0:3], null offen
	buffer_store_b16 v192, v191, s[0:3], null offen
	buffer_store_b16 v178, v193, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v178, v165, s0
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v190, 64, v5 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v7, v7, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v9, 2, v9
	buffer_store_b16 v178, v179, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v178, v187, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v192, 64, v8
	buffer_store_b16 v180, v190, s[0:3], null offen
	v_cvt_pk_bf16_f32 v179, v168, s0
	v_dual_lshlrev_b32 v7, 2, v7 :: v_dual_lshlrev_b32 v178, 2, v178
	buffer_store_b16 v181, v192, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v181, 64, v6
	v_cvt_pk_bf16_f32 v180, v169, s0
	v_cvt_pk_bf16_f32 v173, v162, s0
	v_or_b32_e32 v182, 64, v178
	s_clause 0x1
	buffer_store_b16 v179, v181, s[0:3], null offen
	buffer_store_b16 v180, v182, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v179, 64, v9
	s_wait_xcnt 0x0
	v_mov_b16_e64 v180.l, v183.l
	v_cvt_pk_bf16_f32 v181, v171, s0
	v_or_b32_e32 v182, 64, v7
	v_cvt_pk_bf16_f32 v171, v157, s0
	v_cvt_pk_bf16_f32 v172, v158, s0
	s_clause 0x2
	buffer_store_b16 v180, v179, s[0:3], null offen
	buffer_store_b16 v181, v182, s[0:3], null offen
	buffer_store_b16 v170, v2, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v170, v159, s0
	v_cvt_pk_bf16_f32 v174, v163, s0
	s_clause 0x1
	buffer_store_b16 v171, v0, s[0:3], null offen offset:128
	buffer_store_b16 v172, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v171, v160, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v172, v161, s0
	buffer_store_b16 v170, v186, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v173.l
	v_mov_b16_e64 v173.l, v174.l
	s_clause 0x1
	buffer_store_b16 v171, v189, s[0:3], null offen offset:128
	buffer_store_b16 v172, v194, s[0:3], null offen offset:128
	v_cvt_pk_bf16_f32 v162, v148, s0
	s_clause 0x1
	buffer_store_b16 v170, v188, s[0:3], null offen offset:128
	buffer_store_b16 v173, v191, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v170, 0xc0, v1
	v_cvt_pk_bf16_f32 v164, v150, s0
	v_or_b32_e32 v172, 0xc0, v5
	v_cvt_pk_bf16_f32 v163, v149, s0
	v_or_b32_e32 v171, 0xc0, v3
	v_cvt_pk_bf16_f32 v165, v151, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v173, 0xc0, v8
	s_clause 0x1
	buffer_store_b16 v162, v170, s[0:3], null offen
	buffer_store_b16 v163, v171, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v162, v152, s0
	s_clause 0x1
	buffer_store_b16 v164, v172, s[0:3], null offen
	buffer_store_b16 v165, v173, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v164, 0xc0, v6
	v_cvt_pk_bf16_f32 v168, v155, s0
	v_cvt_pk_bf16_f32 v163, v153, s0
	v_or_b32_e32 v166, 0xc0, v178
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v165, v154, s0
	v_or_b32_e32 v167, 0xc0, v9
	s_clause 0x1
	buffer_store_b16 v162, v164, s[0:3], null offen
	buffer_store_b16 v163, v166, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v162, 0xc0, v7
	s_wait_xcnt 0x0
	v_mov_b16_e64 v163.l, v168.l
	v_cvt_pk_bf16_f32 v154, v140, s0
	v_cvt_pk_bf16_f32 v157, v143, s0
	v_cvt_pk_bf16_f32 v158, v144, s0
	v_cvt_pk_bf16_f32 v155, v141, s0
	s_clause 0x4
	buffer_store_b16 v165, v167, s[0:3], null offen
	buffer_store_b16 v163, v162, s[0:3], null offen
	buffer_store_b16 v154, v2, s[0:3], null offen offset:256
	buffer_store_b16 v155, v0, s[0:3], null offen offset:256
	buffer_store_b16 v156, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e64 v154.l, v158.l
	buffer_store_b16 v157, v186, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v155, v145, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v156, v146, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v157, v147, s0
	buffer_store_b16 v154, v189, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v146, v132, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v154.l, v155.l
	v_mov_b16_e64 v155.l, v156.l
	v_mov_b16_e64 v156.l, v157.l
	v_or_b32_e32 v157, 0x140, v1
	s_clause 0x1
	scratch_load_b128 v[28:31], off, off offset:128 th:TH_LOAD_LU nv
	scratch_load_b128 v[32:35], off, off offset:144 th:TH_LOAD_LU nv
	s_clause 0x1
	buffer_store_b16 v154, v194, s[0:3], null offen offset:256
	buffer_store_b16 v155, v188, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v147, v133, s0
	s_clause 0x1
	buffer_store_b16 v156, v191, s[0:3], null offen offset:256
	buffer_store_b16 v146, v157, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v146, 0x140, v3
	v_cvt_pk_bf16_f32 v149, v135, s0
	v_or_b32_e32 v155, 0x140, v8
	v_cvt_pk_bf16_f32 v148, v134, s0
	v_or_b32_e32 v154, 0x140, v5
	v_cvt_pk_bf16_f32 v150, v136, s0
	v_or_b32_e32 v156, 0x140, v6
	s_clause 0x1
	buffer_store_b16 v147, v146, s[0:3], null offen
	buffer_store_b16 v148, v154, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v146, v137, s0
	s_clause 0x1
	buffer_store_b16 v149, v155, s[0:3], null offen
	buffer_store_b16 v150, v156, s[0:3], null offen
	v_or_b32_e32 v147, 0x140, v178
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v148, v138, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v149, v139, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v150, 0x140, v9
	v_cvt_pk_bf16_f32 v138, v124, s0
	v_or_b32_e32 v151, 0x140, v7
	v_cvt_pk_bf16_f32 v139, v125, s0
	s_clause 0x1
	buffer_store_b16 v146, v147, s[0:3], null offen
	buffer_store_b16 v148, v150, s[0:3], null offen
	v_cvt_pk_bf16_f32 v140, v126, s0
	s_clause 0x1
	buffer_store_b16 v149, v151, s[0:3], null offen
	buffer_store_b16 v138, v2, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v138.l, v139.l
	v_cvt_pk_bf16_f32 v139, v128, s0
	v_cvt_pk_bf16_f32 v141, v129, s0
	v_cvt_pk_bf16_f32 v2, v127, s0
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v138, v0, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v0.l, v139.l
	buffer_store_b16 v140, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v4.l, v141.l
	buffer_store_b16 v2, v186, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v130, s0
	buffer_store_b16 v0, v189, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v0, v131, s0
	buffer_store_b16 v4, v194, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v116, s0
	v_cvt_pk_bf16_f32 v130, v117, s0
	v_or_b32_e32 v3, 0x1c0, v3
	s_clause 0x1
	buffer_store_b16 v2, v188, s[0:3], null offen offset:384
	buffer_store_b16 v0, v191, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v0, v118, s0
	v_or_b32_e32 v2, 0x1c0, v5
	v_or_b32_e32 v5, 0x1c0, v6
	s_set_vgpr_msb 8
	v_or_b32_e32 v6, s10, v130 /*v642*/
	s_clause 0x1
	buffer_store_b16 v4, v1, s[0:3], null offen
	buffer_store_b16 v130, v3, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v119, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v4, 0x1c0, v8
	s_clause 0x1
	buffer_store_b16 v0, v2, s[0:3], null offen
	buffer_store_b16 v1, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v1, v6, s7
	v_cvt_pk_bf16_f32 v3, v120, s0
	v_cvt_pk_bf16_f32 v0, v121, s0
	v_cvt_pk_bf16_f32 v2, v122, s0
	v_or_b32_e32 v4, 0x1c0, v9
	v_cvt_pk_bf16_f32 v9, v110, s0
	buffer_store_b16 v3, v5, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v178
	v_add_lshl_u32 v1, s38, v1, 7
	v_cvt_pk_bf16_f32 v124, v111, s0
	v_cvt_pk_bf16_f32 v126, v112, s0
	v_cvt_pk_bf16_f32 v127, v113, s0
	s_clause 0x1
	buffer_store_b16 v0, v3, s[0:3], null offen
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v2, v1, v201 /*v713*/
	v_cvt_pk_bf16_f32 v0, v123, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v4, 0x1c0, v7
	v_cvt_pk_bf16_f32 v7, v108, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, v1, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v2, 2, v2
	v_or_b32_e32 v8, 3, v6
	buffer_store_b16 v0, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v109, s0
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v7, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 4, v6
	v_mul_lo_u32 v8, v8, s7
	v_cvt_pk_bf16_f32 v128, v115, s0
	v_or_b32_e32 v129, 64, v1
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v116, v212 /*v980*/, s0
	v_mul_lo_u32 v7, s7, v7
	v_cvt_pk_bf16_f32 v117, v213 /*v981*/, s0
	v_cvt_pk_bf16_f32 v119, v216 /*v984*/, s0
	s_set_vgpr_msb 0x308
	v_cvt_pk_bf16_f32 v109, v90, s0
	v_add_lshl_u32 v8, v8, s38, 7
	v_cvt_pk_bf16_f32 v108, v86, s0
	v_cvt_pk_bf16_f32 v110, v91, s0
	v_cvt_pk_bf16_f32 v90, v76, s0
	v_add_lshl_u32 v7, v7, s38, 7
	v_cvt_pk_bf16_f32 v91, v77, s0
	v_cvt_pk_bf16_f32 v86, v56, s0
	v_cvt_pk_bf16_f32 v74, v44, s0
	v_cvt_pk_bf16_f32 v75, v45, s0
	v_or_b32_e32 v125, v7, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v5, 1, v6
	s_set_vgpr_msb 8
	v_or_b32_e32 v7, v7, v198 /*v710*/
	v_cvt_pk_bf16_f32 v76, v46, s0
	v_cvt_pk_bf16_f32 v77, v49, s0
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v125, 2, v125
	v_mul_lo_u32 v3, v5, s7
	v_dual_lshlrev_b32 v7, 2, v7 :: v_dual_bitop2_b32 v5, 2, v6 bitop3:0x54
	s_wait_loadcnt 0x4
	v_cvt_pk_bf16_f32 v55, v26, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v26, v192 /*v960*/, s0
	v_mul_lo_u32 v5, s7, v5
	v_add_lshl_u32 v3, s38, v3, 7
	s_set_vgpr_msb 0x308
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v0, v3, v201 /*v713*/
	v_add_lshl_u32 v5, v5, s38, 7
	v_or_b32_e32 v3, v3, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v123, 5, v6
	v_lshlrev_b32_e32 v0, 2, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v122, v5, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v3, 2, v3
	v_mul_lo_u32 v123, v123, s7
	s_set_vgpr_msb 8
	v_or_b32_e32 v5, v5, v198 /*v710*/
	buffer_store_b16 v4, v0, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v4, 2, v122
	s_set_vgpr_msb 8
	v_or_b32_e32 v122, v8, v201 /*v713*/
	v_or_b32_e32 v8, v8, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v5, 2, v5 :: v_dual_bitop2_b32 v115, 64, v3 bitop3:0x54
	v_add_lshl_u32 v123, v123, s38, 7
	v_lshlrev_b32_e32 v122, 2, v122
	buffer_store_b16 v9, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 6, v6
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_bitop2_b32 v6, 7, v6 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v130, v123, v201 /*v713*/
	s_clause 0x1
	buffer_store_b16 v124, v122, s[0:3], null offen
	buffer_store_b16 v126, v125, s[0:3], null offen
	v_mul_lo_u32 v9, v9, s7
	v_mul_lo_u32 v6, v6, s7
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v126, v114, s0
	s_set_vgpr_msb 0x803
	v_lshlrev_b32_e32 v130, 2, v130
	v_cvt_pk_bf16_f32 v114, v210 /*v978*/, s0
	v_add_lshl_u32 v9, s38, v9, 7
	v_add_lshl_u32 v6, s38, v6, 7
	s_set_vgpr_msb 0x308
	buffer_store_b16 v127, v130, s[0:3], null offen
	v_or_b32_e32 v124, v9, v201 /*v713*/
	s_wait_xcnt 0x0
	v_or_b32_e32 v127, v6, v201 /*v713*/
	v_or_b32_e32 v9, v9, v198 /*v710*/
	v_or_b32_e32 v6, v6, v198 /*v710*/
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_lshlrev_b32 v124, 2, v124 :: v_dual_lshlrev_b32 v127, 2, v127
	s_clause 0x2
	buffer_store_b16 v126, v124, s[0:3], null offen
	buffer_store_b16 v128, v127, s[0:3], null offen
	buffer_store_b16 v114, v129, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v114, v211 /*v979*/, s0
	v_or_b32_e32 v126, 64, v5
	v_dual_lshlrev_b32 v9, 2, v9 :: v_dual_bitop2_b32 v128, 64, v8 bitop3:0x54
	v_lshlrev_b32_e32 v6, 2, v6
	s_set_vgpr_msb 0x308
	buffer_store_b16 v114, v115, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v114, v123, v198 /*v710*/
	s_clause 0x1
	buffer_store_b16 v116, v126, s[0:3], null offen
	buffer_store_b16 v117, v128, s[0:3], null offen
	s_set_vgpr_msb 0x803
	v_cvt_pk_bf16_f32 v115, v214 /*v982*/, s0
	v_dual_lshlrev_b32 v114, 2, v114 :: v_dual_bitop2_b32 v117, 64, v7 bitop3:0x54
	v_cvt_pk_bf16_f32 v116, v215 /*v983*/, s0
	v_or_b32_e32 v103, 0xc0, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v118, 64, v114
	s_set_vgpr_msb 0x300
	s_clause 0x1
	buffer_store_b16 v115, v117, s[0:3], null offen
	buffer_store_b16 v116, v118, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v115, 64, v9
	s_wait_xcnt 0x0
	v_mov_b16_e32 v116.l, v119.l
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v117, v217 /*v985*/, s0
	v_or_b32_e32 v118, 64, v6
	v_or_b32_e32 v102, 0xc0, v114
	s_set_vgpr_msb 0x300
	s_clause 0x2
	buffer_store_b16 v116, v115, s[0:3], null offen
	buffer_store_b16 v117, v118, s[0:3], null offen
	buffer_store_b16 v106, v2, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v106, v87, s0
	s_clause 0x1
	buffer_store_b16 v107, v0, s[0:3], null offen offset:128
	buffer_store_b16 v108, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v107, v88, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v108, v89, s0
	v_or_b32_e32 v87, 0x140, v6
	buffer_store_b16 v106, v122, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v106.l, v109.l
	v_mov_b16_e32 v109.l, v110.l
	s_clause 0x3
	buffer_store_b16 v107, v125, s[0:3], null offen offset:128
	buffer_store_b16 v108, v130, s[0:3], null offen offset:128
	buffer_store_b16 v106, v124, s[0:3], null offen offset:128
	buffer_store_b16 v109, v127, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v106, 0xc0, v1
	v_or_b32_e32 v108, 0xc0, v5
	v_or_b32_e32 v107, 0xc0, v3
	s_wait_xcnt 0x0
	v_or_b32_e32 v109, 0xc0, v8
	s_clause 0x1
	buffer_store_b16 v98, v106, s[0:3], null offen
	buffer_store_b16 v99, v107, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v98, v64, s0
	s_clause 0x1
	buffer_store_b16 v100, v108, s[0:3], null offen
	buffer_store_b16 v101, v109, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v100, 0xc0, v7
	v_cvt_pk_bf16_f32 v99, v65, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v101, v66, s0
	s_clause 0x1
	buffer_store_b16 v98, v100, s[0:3], null offen
	buffer_store_b16 v99, v102, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v98, 0xc0, v6
	s_wait_xcnt 0x0
	v_mov_b16_e32 v99.l, v104.l
	buffer_store_b16 v101, v103, s[0:3], null offen
	v_cvt_pk_bf16_f32 v66, v37, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v64, v177 /*v945*/, s0
	s_set_vgpr_msb 0x301
	v_cvt_pk_bf16_f32 v37, v69 /*v325*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x3
	buffer_store_b16 v99, v98, s[0:3], null offen
	buffer_store_b16 v90, v2, s[0:3], null offen offset:256
	buffer_store_b16 v91, v0, s[0:3], null offen offset:256
	buffer_store_b16 v92, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e32 v90.l, v94.l
	buffer_store_b16 v93, v122, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v91, v81, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v92, v82, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v93, v83, s0
	buffer_store_b16 v90, v125, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v82, v52, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v90.l, v91.l
	v_mov_b16_e32 v91.l, v92.l
	v_mov_b16_e32 v92.l, v93.l
	v_or_b32_e32 v93, 0x140, v1
	s_clause 0x1
	buffer_store_b16 v90, v130, s[0:3], null offen offset:256
	buffer_store_b16 v91, v124, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v83, v53, s0
	s_clause 0x1
	buffer_store_b16 v92, v127, s[0:3], null offen offset:256
	buffer_store_b16 v82, v93, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 0x140, v3
	v_or_b32_e32 v91, 0x140, v8
	v_or_b32_e32 v90, 0x140, v5
	v_or_b32_e32 v92, 0x140, v7
	s_clause 0x1
	buffer_store_b16 v83, v82, s[0:3], null offen
	buffer_store_b16 v84, v90, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v82, v57, s0
	s_clause 0x1
	buffer_store_b16 v85, v91, s[0:3], null offen
	buffer_store_b16 v86, v92, s[0:3], null offen
	v_or_b32_e32 v83, 0x140, v114
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v84, v58, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v85, v59, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v86, 0x140, v9
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v82, v83, s[0:3], null offen
	v_or_b32_e32 v3, 0x1c0, v3
	v_cvt_pk_bf16_f32 v52, v22, s0
	s_clause 0x2
	buffer_store_b16 v84, v86, s[0:3], null offen
	buffer_store_b16 v85, v87, s[0:3], null offen
	buffer_store_b16 v74, v2, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v74.l, v75.l
	v_cvt_pk_bf16_f32 v75, v48, s0
	v_cvt_pk_bf16_f32 v2, v47, s0
	v_cvt_pk_bf16_f32 v53, v23, s0
	v_cvt_pk_bf16_f32 v22, v14, s0
	buffer_store_b16 v74, v0, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v0.l, v75.l
	buffer_store_b16 v76, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v4.l, v77.l
	buffer_store_b16 v2, v122, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v50, s0
	buffer_store_b16 v0, v125, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v0, v51, s0
	buffer_store_b16 v4, v130, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v36, s0
	s_clause 0x1
	buffer_store_b16 v2, v124, s[0:3], null offen offset:384
	buffer_store_b16 v0, v127, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v0, v38, s0
	v_or_b32_e32 v2, 0x1c0, v5
	v_or_b32_e32 v5, 0x1c0, v7
	s_set_vgpr_msb 8
	v_or_b32_e32 v7, s9, v130 /*v642*/
	s_clause 0x1
	buffer_store_b16 v4, v1, s[0:3], null offen
	buffer_store_b16 v66, v3, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v39, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v4, 0x1c0, v8
	s_clause 0x1
	buffer_store_b16 v0, v2, s[0:3], null offen
	buffer_store_b16 v1, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v1, v7, s7
	v_cvt_pk_bf16_f32 v3, v40, s0
	v_cvt_pk_bf16_f32 v0, v41, s0
	v_cvt_pk_bf16_f32 v2, v42, s0
	v_or_b32_e32 v4, 0x1c0, v9
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v9, v172 /*v940*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v3, v5, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v114
	v_add_lshl_u32 v1, s38, v1, 7
	v_or_b32_e32 v5, 1, v7
	v_cvt_pk_bf16_f32 v50, v20, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v36, v68 /*v324*/, s0
	s_set_vgpr_msb 0x108
	s_clause 0x1
	buffer_store_b16 v0, v3, s[0:3], null offen
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, v1, v201 /*v713*/
	v_or_b32_e32 v1, v1, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v59, 5, v7
	v_mul_lo_u32 v3, v5, s7
	v_or_b32_e32 v5, 2, v7
	v_cvt_pk_bf16_f32 v0, v43, s0
	v_lshlrev_b32_e32 v1, 2, v1
	v_mul_lo_u32 v59, v59, s7
	v_or_b32_e32 v4, 0x1c0, v6
	v_mul_lo_u32 v5, v5, s7
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v6, v170 /*v938*/, s0
	v_add_lshl_u32 v3, s38, v3, 7
	v_lshlrev_b32_e32 v2, 2, v2
	s_set_vgpr_msb 0x300
	buffer_store_b16 v0, v4, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v4, v171 /*v939*/, s0
	v_add_lshl_u32 v59, s38, v59, 7
	s_set_vgpr_msb 0x308
	v_or_b32_e32 v0, v3, v201 /*v713*/
	v_add_lshl_u32 v5, v5, s38, 7
	buffer_store_b16 v6, v2, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v6, 4, v7
	s_set_vgpr_msb 8
	v_or_b32_e32 v66, v59, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v0, 2, v0 :: v_dual_bitop2_b32 v8, 3, v7 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v58, v5, v201 /*v713*/
	v_mul_lo_u32 v6, v6, s7
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v66, 2, v66
	v_mul_lo_u32 v8, v8, s7
	buffer_store_b16 v4, v0, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v4, 2, v58
	s_set_vgpr_msb 8
	v_or_b32_e32 v5, v5, v198 /*v710*/
	v_or_b32_e32 v3, v3, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v65, 64, v1
	v_add_lshl_u32 v6, v6, s38, 7
	buffer_store_b16 v9, v4, s[0:3], null offen
	v_add_lshl_u32 v8, v8, s38, 7
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v5, 2, v5 :: v_dual_bitop2_b32 v9, 6, v7 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v61, v6, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v3, 2, v3
	s_set_vgpr_msb 8
	v_or_b32_e32 v58, v8, v201 /*v713*/
	v_or_b32_e32 v8, v8, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v7, 7, v7
	v_mul_lo_u32 v9, v9, s7
	v_dual_lshlrev_b32 v61, 2, v61 :: v_dual_lshlrev_b32 v58, 2, v58
	v_lshlrev_b32_e32 v8, 2, v8
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v7, v7, s7
	s_clause 0x1
	buffer_store_b16 v60, v58, s[0:3], null offen
	buffer_store_b16 v62, v61, s[0:3], null offen
	v_add_lshl_u32 v9, v9, s38, 7
	buffer_store_b16 v63, v66, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v62, v176 /*v944*/, s0
	s_set_vgpr_msb 0x308
	v_or_b32_e32 v6, v6, v198 /*v710*/
	v_add_lshl_u32 v7, v7, s38, 7
	v_or_b32_e32 v60, v9, v201 /*v713*/
	v_or_b32_e32 v9, v9, v198 /*v710*/
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v42, v28, s0
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v6, 2, v6
	s_set_vgpr_msb 8
	v_or_b32_e32 v63, v7, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v60, 2, v60
	s_set_vgpr_msb 8
	v_or_b32_e32 v7, v7, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v9, 2, v9
	v_cvt_pk_bf16_f32 v43, v29, s0
	v_lshlrev_b32_e32 v63, 2, v63
	s_clause 0x2
	buffer_store_b16 v62, v60, s[0:3], null offen
	buffer_store_b16 v64, v63, s[0:3], null offen
	buffer_store_b16 v50, v65, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v50, v21, s0
	v_or_b32_e32 v62, 64, v5
	v_or_b32_e32 v51, 64, v3
	v_lshlrev_b32_e32 v7, 2, v7
	s_wait_loadcnt 0x0
	v_cvt_pk_bf16_f32 v45, v34, s0
	v_cvt_pk_bf16_f32 v44, v30, s0
	v_cvt_pk_bf16_f32 v46, v35, s0
	buffer_store_b16 v50, v51, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v50, v59, v198 /*v710*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v64, 64, v8
	buffer_store_b16 v52, v62, s[0:3], null offen
	v_cvt_pk_bf16_f32 v51, v24, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v52, v25, s0
	v_lshlrev_b32_e32 v50, 2, v50
	buffer_store_b16 v53, v64, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v53, 64, v6
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v34, v66 /*v322*/, s0
	v_cvt_pk_bf16_f32 v35, v67 /*v323*/, s0
	v_or_b32_e32 v54, 64, v50
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v51, v53, s[0:3], null offen
	buffer_store_b16 v52, v54, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v51, 64, v9
	s_wait_xcnt 0x0
	v_mov_b16_e32 v52.l, v55.l
	v_cvt_pk_bf16_f32 v53, v27, s0
	v_or_b32_e32 v54, 64, v7
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v40, v73 /*v329*/, s0
	v_or_b32_e32 v38, 0xc0, v50
	s_set_vgpr_msb 0x100
	s_clause 0x2
	buffer_store_b16 v52, v51, s[0:3], null offen
	buffer_store_b16 v53, v54, s[0:3], null offen
	buffer_store_b16 v42, v2, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v42, v31, s0
	s_clause 0x1
	buffer_store_b16 v43, v0, s[0:3], null offen offset:128
	buffer_store_b16 v44, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v43, v32, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v44, v33, s0
	v_or_b32_e32 v39, 0xc0, v9
	buffer_store_b16 v42, v58, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v42.l, v45.l
	v_mov_b16_e32 v45.l, v46.l
	s_clause 0x3
	buffer_store_b16 v43, v61, s[0:3], null offen offset:128
	buffer_store_b16 v44, v66, s[0:3], null offen offset:128
	buffer_store_b16 v42, v60, s[0:3], null offen offset:128
	buffer_store_b16 v45, v63, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v42, 0xc0, v1
	v_or_b32_e32 v44, 0xc0, v5
	v_or_b32_e32 v43, 0xc0, v3
	s_wait_xcnt 0x0
	v_or_b32_e32 v45, 0xc0, v8
	s_clause 0x1
	buffer_store_b16 v34, v42, s[0:3], null offen
	buffer_store_b16 v35, v43, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v34, v70 /*v326*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v36, v44, s[0:3], null offen
	buffer_store_b16 v37, v45, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v36, 0xc0, v6
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v35, v71 /*v327*/, s0
	v_cvt_pk_bf16_f32 v37, v72 /*v328*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v34, v36, s[0:3], null offen
	buffer_store_b16 v35, v38, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v34, 0xc0, v7
	s_wait_xcnt 0x0
	v_mov_b16_e32 v35.l, v40.l
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v29, v195 /*v963*/, s0
	v_cvt_pk_bf16_f32 v30, v196 /*v964*/, s0
	v_cvt_pk_bf16_f32 v27, v193 /*v961*/, s0
	v_cvt_pk_bf16_f32 v28, v194 /*v962*/, s0
	s_set_vgpr_msb 0x300
	s_clause 0x4
	buffer_store_b16 v37, v39, s[0:3], null offen
	buffer_store_b16 v35, v34, s[0:3], null offen
	buffer_store_b16 v26, v2, s[0:3], null offen offset:256
	buffer_store_b16 v27, v0, s[0:3], null offen offset:256
	buffer_store_b16 v28, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e32 v26.l, v30.l
	buffer_store_b16 v29, v58, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v27, v197 /*v965*/, s0
	v_cvt_pk_bf16_f32 v28, v198 /*v966*/, s0
	v_cvt_pk_bf16_f32 v29, v199 /*v967*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v26, v61, s[0:3], null offen offset:256
	v_or_b32_e32 v20, 0x140, v3
	s_wait_xcnt 0x0
	v_mov_b16_e32 v26.l, v27.l
	v_mov_b16_e32 v27.l, v28.l
	v_mov_b16_e32 v28.l, v29.l
	v_or_b32_e32 v29, 0x140, v1
	s_clause 0x3
	buffer_store_b16 v26, v66, s[0:3], null offen offset:256
	buffer_store_b16 v27, v60, s[0:3], null offen offset:256
	buffer_store_b16 v28, v63, s[0:3], null offen offset:256
	buffer_store_b16 v18, v29, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v18, v12, s0
	v_or_b32_e32 v26, 0x140, v5
	v_cvt_pk_bf16_f32 v21, v13, s0
	v_or_b32_e32 v27, 0x140, v8
	v_or_b32_e32 v28, 0x140, v6
	v_cvt_pk_bf16_f32 v23, v15, s0
	v_or_b32_e32 v29, 0x140, v50
	s_clause 0x4
	buffer_store_b16 v19, v20, s[0:3], null offen
	buffer_store_b16 v18, v26, s[0:3], null offen
	buffer_store_b16 v21, v27, s[0:3], null offen
	buffer_store_b16 v22, v28, s[0:3], null offen
	buffer_store_b16 v23, v29, s[0:3], null offen
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v18, v16, s0
	v_or_b32_e32 v19, 0x140, v9
	v_cvt_pk_bf16_f32 v20, v17, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v23, 0x140, v7
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v21, v234 /*v1002*/, s0
	v_cvt_pk_bf16_f32 v22, v235 /*v1003*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v18, v19, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v18, v236 /*v1004*/, s0
	s_set_vgpr_msb 0x300
	s_clause 0x2
	buffer_store_b16 v20, v23, s[0:3], null offen
	buffer_store_b16 v21, v2, s[0:3], null offen offset:384
	buffer_store_b16 v22, v0, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v19, v239 /*v1007*/, s0
	v_cvt_pk_bf16_f32 v20, v240 /*v1008*/, s0
	v_cvt_pk_bf16_f32 v0, v237 /*v1005*/, s0
	s_set_vgpr_msb 0x300
	v_mov_b16_e32 v2.l, v18.l
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v18, v238 /*v1006*/, s0
	v_or_b32_e32 v1, 0x1c0, v1
	v_or_b32_e32 v3, 0x1c0, v3
	v_or_b32_e32 v5, 0x1c0, v5
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v2.l, v18.l
	v_mov_b16_e32 v4.l, v19.l
	v_mov_b16_e32 v18.l, v20.l
	buffer_store_b16 v0, v58, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v0, v241 /*v1009*/, s0
	s_set_vgpr_msb 0x300
	s_clause 0x2
	buffer_store_b16 v2, v61, s[0:3], null offen offset:384
	buffer_store_b16 v4, v66, s[0:3], null offen offset:384
	buffer_store_b16 v18, v60, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v2, v162 /*v930*/, s0
	v_cvt_pk_bf16_f32 v4, v163 /*v931*/, s0
	s_set_vgpr_msb 0x300
	s_clause 0x1
	buffer_store_b16 v0, v63, s[0:3], null offen offset:384
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v0, v165 /*v933*/, s0
	v_or_b32_e32 v2, 0x1c0, v8
	s_set_vgpr_msb 0x300
	s_clause 0x1
	buffer_store_b16 v4, v3, s[0:3], null offen
	buffer_store_b16 v10, v5, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v1, v166 /*v934*/, s0
	v_or_b32_e32 v3, 0x1c0, v6
	v_cvt_pk_bf16_f32 v4, v167 /*v935*/, s0
	v_or_b32_e32 v6, 0x1c0, v50
	v_cvt_pk_bf16_f32 v5, v168 /*v936*/, s0
	v_or_b32_e32 v9, 0x1c0, v9
	v_cvt_pk_bf16_f32 v8, v169 /*v937*/, s0
	v_or_b32_e32 v7, 0x1c0, v7
	s_set_vgpr_msb 0x300
	s_clause 0x4
	buffer_store_b16 v0, v2, s[0:3], null offen
	buffer_store_b16 v1, v3, s[0:3], null offen
	buffer_store_b16 v4, v6, s[0:3], null offen
	buffer_store_b16 v5, v9, s[0:3], null offen
	buffer_store_b16 v8, v7, s[0:3], null offen
	s_endpgm
.Lfunc_end0:
	.size	k_dq_0, .Lfunc_end0-k_dq_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dq_0
		.amdhsa_group_segment_fixed_size 8704
		.amdhsa_private_segment_fixed_size 3604
		.amdhsa_kernarg_size 356
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 1023
		.amdhsa_next_free_sgpr 45
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-k_dq_0)<<4)&4080)>>4
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

	.set .Lk_dq_0.num_vgpr, 1023
	.set .Lk_dq_0.num_agpr, 0
	.set .Lk_dq_0.numbered_sgpr, 45
	.set .Lk_dq_0.num_named_barrier, 0
	.set .Lk_dq_0.private_seg_size, 3604
	.set .Lk_dq_0.uses_vcc, 1
	.set .Lk_dq_0.uses_flat_scratch, 1
	.set .Lk_dq_0.has_dyn_sized_stack, 0
	.set .Lk_dq_0.has_recursion, 0
	.set .Lk_dq_0.has_indirect_call, 0
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
        .size:           40
        .value_kind:     by_value
      - .address_space:  global
        .offset:         144
        .size:           8
        .value_kind:     global_buffer
      - .offset:         152
        .size:           40
        .value_kind:     by_value
      - .address_space:  global
        .offset:         192
        .size:           8
        .value_kind:     global_buffer
      - .offset:         200
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         232
        .size:           8
        .value_kind:     global_buffer
      - .offset:         240
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         272
        .size:           8
        .value_kind:     global_buffer
      - .offset:         280
        .size:           40
        .value_kind:     by_value
      - .offset:         320
        .size:           4
        .value_kind:     by_value
      - .offset:         324
        .size:           4
        .value_kind:     by_value
      - .offset:         328
        .size:           4
        .value_kind:     by_value
      - .offset:         332
        .size:           4
        .value_kind:     by_value
      - .offset:         336
        .size:           4
        .value_kind:     by_value
      - .offset:         340
        .size:           4
        .value_kind:     by_value
      - .offset:         344
        .size:           4
        .value_kind:     by_value
      - .offset:         348
        .size:           4
        .value_kind:     by_value
      - .offset:         352
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 8704
    .kernarg_segment_align: 8
    .kernarg_segment_size: 356
    .max_flat_workgroup_size: 32
    .name:           k_dq_0
    .private_segment_fixed_size: 3604
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     47
    .sgpr_spill_count: 0
    .symbol:         k_dq_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1023
    .vgpr_spill_count: 1068
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
