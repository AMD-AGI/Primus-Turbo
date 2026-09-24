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
	v_dual_lshrrev_b32 v2, 4, v0 :: v_dual_bitop2_b32 v14, 15, v0 bitop3:0x40
	s_cselect_b32 s12, ttmp9, s13
	s_cselect_b32 s2, s3, s2
	s_bfe_u32 s3, ttmp6, 0x40014
	s_lshr_b32 s13, ttmp7, 16
	s_add_co_i32 s3, s3, 1
	s_bfe_u32 s15, ttmp6, 0x40008
	s_mul_i32 s3, s13, s3
	s_clause 0x1
	s_load_b64 s[20:21], s[0:1], 0x0 nv
	s_load_b64 s[24:25], s[0:1], 0x90 nv
	s_add_co_i32 s15, s15, s3
	s_cmp_eq_u32 s14, 0
	s_mov_b32 s22, 0x800000
	s_cselect_b32 s3, s13, s15
	s_wait_kmcnt 0x0
	s_add_co_i32 s13, s5, 0xff
	s_mul_i32 s46, s5, s3
	s_ashr_i32 s14, s13, 31
	s_mov_b32 s23, 0
	s_lshr_b32 s14, s14, 24
	s_mov_b32 s26, s22
	s_add_co_i32 s14, s13, s14
	s_mov_b32 s27, s23
	s_and_b32 s15, s14, 0xffffff00
	s_ashr_i32 s14, s14, 8
	s_cmp_lg_u32 s13, s15
	s_set_vgpr_msb 64
	v_lshlrev_b32_e32 v155 /*v411*/, 3, v2
	s_cselect_b32 s15, -1, 0
	s_cmp_lt_i32 s13, 0
	s_mul_i32 s6, s6, s3
	s_cselect_b32 s13, -1, 0
	s_not_b32 s2, s2
	s_and_b32 s13, s13, s15
	s_add_co_i32 s14, s14, s2
	s_cmp_lg_u32 s13, 0
	s_mul_i32 s3, s7, s3
	s_sub_co_ci_u32 s2, s14, 0
	s_load_b32 s14, s[0:1], 0x160 nv
	s_lshl_b32 s44, s2, 8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s49, s44, s11
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v1, s44, v14
	s_add_co_i32 s2, s49, 0x11f
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s13, s2, 31
	s_lshr_b32 s13, s13, 27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s13, s2, s13
	s_and_b32 s15, s13, 0xffffffe0
	s_ashr_i32 s13, s13, 5
	s_cmp_lg_u32 s2, s15
	s_cselect_b32 s15, -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, s15
	s_sub_co_ci_u32 s2, s13, 0
	s_max_i32 s2, s2, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s10
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s14, 0
	s_cselect_b32 s13, -1, 0
	s_and_b32 s13, s13, exec_lo
	s_cselect_b32 s50, s2, s10
	s_add_co_i32 s2, s49, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s13, s2, 31
	s_lshr_b32 s13, s13, 27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s13, s2, s13
	s_ashr_i32 s13, s13, 5
	s_cmp_gt_i32 s2, -1
	s_cselect_b32 s2, s13, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s50
	s_cmp_lg_u32 s14, 0
	s_cselect_b32 s47, -1, 0
	s_and_b32 s13, s47, exec_lo
	s_cselect_b32 s2, s2, s10
	s_ashr_i32 s13, s12, 31
	s_ashr_i32 s14, s7, 31
	s_lshr_b32 s13, s13, 29
	s_lshr_b32 s14, s14, 29
	s_add_co_i32 s13, s12, s13
	s_add_co_i32 s14, s7, s14
	s_ashr_i32 s15, s13, 3
	s_and_b32 s13, s13, -8
	s_ashr_i32 s16, s14, 3
	s_and_b32 s14, s14, -8
	s_and_b32 s10, s7, 7
	s_sub_co_i32 s17, s12, s13
	s_cmp_lg_u32 s7, s14
	s_cselect_b32 s14, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s18, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s14, s18, s14
	s_sub_co_ci_u32 s14, s16, 0
	s_cmp_lg_u32 s12, s13
	s_mul_i32 s14, s14, s17
	s_cselect_b32 s13, -1, 0
	s_cmp_lt_i32 s12, 0
	s_mov_b64 s[18:19], 0x800000
	s_cselect_b32 s16, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s13, s16, s13
	s_sub_co_ci_u32 s13, s15, 0
	s_add_co_i32 s13, s13, s14
	s_cmp_eq_u32 s10, 0
	s_cselect_b32 s45, s13, s12
	s_abs_i32 s10, s9
	s_abs_i32 s14, s45
	s_cvt_f32_u32 s12, s10
	s_sub_co_i32 s13, 0, s10
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s12, s12
	s_mul_f32 s12, s12, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s12, s12
	s_mul_i32 s13, s13, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s13, s12, s13
	s_add_co_i32 s12, s12, s13
	s_xor_b32 s13, s45, s9
	s_mul_hi_u32 s12, s14, s12
	s_ashr_i32 s16, s13, 31
	s_mul_i32 s15, s12, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s14, s14, s15
	s_add_co_i32 s15, s12, 1
	s_sub_co_i32 s17, s14, s10
	s_cmp_ge_u32 s14, s10
	s_cselect_b32 s12, s15, s12
	s_cselect_b32 s14, s17, s14
	s_add_co_i32 s15, s12, 1
	s_cmp_ge_u32 s14, s10
	s_cselect_b32 s10, s15, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s10, s10, s16
	s_sub_co_i32 s12, s10, s16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s12, s12, s9
	s_cmp_lg_u32 s45, s12
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s13, 0
	s_cselect_b32 s12, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s9, s12, s9
	s_sub_co_ci_u32 s9, s10, s16
	s_lshl_b32 s10, s7, 4
	s_or_b32 s43, s44, 16
	s_mul_i32 s12, s10, s46
	s_or_b32 s42, s44, 32
	s_lshl4_add_u32 s14, s45, s12
	v_or_b32_e32 v171, s43, v14
	v_or_b32_e32 v173, s42, v14
	v_mad_u32 v3, v1, s10, s14
	s_or_b32 s40, s44, 64
	s_or_b32 s41, s44, 48
	v_mad_u32 v4, v171, s10, s14
	v_mad_u32 v5, v173, s10, s14
	s_set_vgpr_msb 64
	v_or_b32_e32 v104 /*v360*/, s40, v14
	v_or_b32_e32 v103 /*v359*/, s41, v14
	s_or_b32 s39, s44, 0x50
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v3, v3, v2
	s_set_vgpr_msb 64
	v_or_b32_e32 v105 /*v361*/, s39, v14
	s_or_b32 s38, s44, 0x60
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v4, v4, v2
	v_dual_lshlrev_b32 v3, 4, v3 :: v_dual_bitop2_b32 v5, v5, v2 bitop3:0x54
	s_set_vgpr_msb 1
	v_mad_u32 v6, v103 /*v359*/, s10, s14
	s_set_vgpr_msb 0x180
	v_or_b32_e32 v77 /*v589*/, s38, v14
	s_set_vgpr_msb 0x8000
	v_dual_lshlrev_b32 v4, 4, v4 :: v_dual_lshlrev_b32 v5, 4, v5
	s_clause 0x9
	buffer_load_b128 v[18:21], v3, s[20:23], null offen
	buffer_load_b128 v[22:25], v3, s[20:23], null offen offset:32
	buffer_load_b128 v[246:249], v3, s[20:23], null offen offset:64
	buffer_load_b128 v[250:253], v3, s[20:23], null offen offset:96
	s_set_vgpr_msb 64
	buffer_load_b128 v[6:9] /*v[262:265]*/, v3, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13] /*v[266:269]*/, v3, s[20:23], null offen offset:160
	s_set_vgpr_msb 0x4000
	buffer_load_b128 v[98:101], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[102:105], v3, s[20:23], null offen offset:224
	s_set_vgpr_msb 64
	s_clause 0x8
	buffer_load_b128 v[14:17] /*v[270:273]*/, v3, s[24:27], null offen
	buffer_load_b128 v[18:21] /*v[274:277]*/, v3, s[24:27], null offen offset:32
	s_set_vgpr_msb 0x4000
	buffer_load_b128 v[106:109], v3, s[24:27], null offen offset:64
	buffer_load_b128 v[110:113], v3, s[24:27], null offen offset:96
	buffer_load_b128 v[114:117], v3, s[24:27], null offen offset:128
	buffer_load_b128 v[118:121], v3, s[24:27], null offen offset:160
	buffer_load_b128 v[122:125], v3, s[24:27], null offen offset:192
	buffer_load_b128 v[126:129], v3, s[24:27], null offen offset:224
	s_clause 0x9
	buffer_load_b128 v[130:133], v4, s[20:23], null offen
	buffer_load_b128 v[134:137], v4, s[20:23], null offen offset:32
	s_set_vgpr_msb 64
	buffer_load_b128 v[94:97] /*v[350:353]*/, v4, s[20:23], null offen offset:64
	buffer_load_b128 v[98:101] /*v[354:357]*/, v4, s[20:23], null offen offset:96
	buffer_load_b128 v[170:173] /*v[426:429]*/, v4, s[20:23], null offen offset:128
	buffer_load_b128 v[174:177] /*v[430:433]*/, v4, s[20:23], null offen offset:160
	s_set_vgpr_msb 0x4080
	buffer_load_b128 v[138:141] /*v[650:653]*/, v4, s[20:23], null offen offset:192
	buffer_load_b128 v[142:145] /*v[654:657]*/, v4, s[20:23], null offen offset:224
	s_set_vgpr_msb 0x8040
	s_clause 0x8
	buffer_load_b128 v[210:213] /*v[466:469]*/, v4, s[24:27], null offen
	buffer_load_b128 v[214:217] /*v[470:473]*/, v4, s[24:27], null offen offset:32
	s_set_vgpr_msb 0x4080
	buffer_load_b128 v[2:5] /*v[514:517]*/, v4, s[24:27], null offen offset:64
	buffer_load_b128 v[6:9] /*v[518:521]*/, v4, s[24:27], null offen offset:96
	buffer_load_b128 v[28:31] /*v[540:543]*/, v4, s[24:27], null offen offset:128
	buffer_load_b128 v[32:35] /*v[544:547]*/, v4, s[24:27], null offen offset:160
	buffer_load_b128 v[84:87] /*v[596:599]*/, v4, s[24:27], null offen offset:192
	buffer_load_b128 v[88:91] /*v[600:603]*/, v4, s[24:27], null offen offset:224
	s_clause 0xa
	buffer_load_b128 v[100:103] /*v[612:615]*/, v5, s[20:23], null offen
	buffer_load_b128 v[104:107] /*v[616:619]*/, v5, s[20:23], null offen offset:32
	s_set_vgpr_msb 0x8040
	buffer_load_b128 v[242:245] /*v[498:501]*/, v5, s[20:23], null offen offset:64
	buffer_load_b128 v[246:249] /*v[502:505]*/, v5, s[20:23], null offen offset:96
	s_set_vgpr_msb 0x4000
	buffer_load_b128 v[222:225], v5, s[20:23], null offen offset:128
	buffer_load_b128 v[226:229], v5, s[20:23], null offen offset:160
	buffer_load_b128 v[254:257], v5, s[20:23], null offen offset:192
	s_set_vgpr_msb 64
	buffer_load_b128 v[2:5] /*v[258:261]*/, v5, s[20:23], null offen offset:224
	s_set_vgpr_msb 0x4000
	s_clause 0x3
	buffer_load_b128 v[238:241], v5, s[24:27], null offen
	buffer_load_b128 v[242:245], v5, s[24:27], null offen offset:32
	buffer_load_b128 v[90:93], v5, s[24:27], null offen offset:64
	buffer_load_b128 v[94:97], v5, s[24:27], null offen offset:96
	s_set_vgpr_msb 1
	v_mad_u32 v4, v104 /*v360*/, s10, s14
	s_or_b32 s37, s44, 0x70
	s_or_b32 s36, s44, 0x80
	s_set_vgpr_msb 0x100
	v_or_b32_e32 v6, v6, v2
	s_set_vgpr_msb 0x80
	v_or_b32_e32 v79 /*v591*/, s37, v14
	v_or_b32_e32 v81 /*v593*/, s36, v14
	s_or_b32 s35, s44, 0x90
	s_or_b32 s34, s44, 0xa0
	s_set_vgpr_msb 0x8000
	v_or_b32_e32 v4, v4, v2
	v_lshlrev_b32_e32 v3, 4, v6
	s_clause 0x7
	buffer_load_b128 v[174:177], v3, s[20:23], null offen
	buffer_load_b128 v[178:181], v3, s[20:23], null offen offset:32
	buffer_load_b128 v[182:185], v3, s[20:23], null offen offset:64
	buffer_load_b128 v[186:189], v3, s[20:23], null offen offset:96
	buffer_load_b128 v[190:193], v3, s[20:23], null offen offset:128
	buffer_load_b128 v[194:197], v3, s[20:23], null offen offset:160
	buffer_load_b128 v[198:201], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[202:205], v3, s[20:23], null offen offset:224
	v_lshlrev_b32_e32 v4, 4, v4
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:32
	s_set_vgpr_msb 0x80
	s_clause 0x9
	buffer_load_b128 v[92:95] /*v[604:607]*/, v3, s[24:27], null offen
	buffer_load_b128 v[96:99] /*v[608:611]*/, v3, s[24:27], null offen offset:32
	s_set_vgpr_msb 0x8000
	buffer_load_b128 v[162:165], v3, s[24:27], null offen offset:64
	buffer_load_b128 v[166:169], v3, s[24:27], null offen offset:96
	buffer_load_b128 v[154:157], v3, s[24:27], null offen offset:128
	buffer_load_b128 v[158:161], v3, s[24:27], null offen offset:160
	s_set_vgpr_msb 0x80
	buffer_load_b128 v[36:39] /*v[548:551]*/, v3, s[24:27], null offen offset:192
	buffer_load_b128 v[40:43] /*v[552:555]*/, v3, s[24:27], null offen offset:224
	s_set_vgpr_msb 0x8001
	v_mad_u32 v3, v105 /*v361*/, s10, s14
	s_set_vgpr_msb 0x180
	v_or_b32_e32 v125 /*v637*/, s35, v14
	v_or_b32_e32 v127 /*v639*/, s34, v14
	s_or_b32 s33, s44, 0xb0
	s_or_b32 s31, s44, 0xc0
	v_or_b32_e32 v128 /*v640*/, s33, v14
	v_or_b32_e32 v129 /*v641*/, s31, v14
	s_or_b32 s30, s44, 0xd0
	s_set_vgpr_msb 0x8000
	v_or_b32_e32 v3, v3, v2
	s_set_vgpr_msb 0x80
	v_or_b32_e32 v243 /*v755*/, s30, v14
	s_or_b32 s29, s44, 0xe0
	s_or_b32 s28, s44, 0xf0
	v_or_b32_e32 v245 /*v757*/, s29, v14
	s_set_vgpr_msb 0x8000
	v_lshlrev_b32_e32 v3, 4, v3
	s_set_vgpr_msb 0x80
	v_or_b32_e32 v247 /*v759*/, s28, v14
	s_set_vgpr_msb 0x8000
	s_clause 0x3
	buffer_load_b128 v[138:141], v5, s[24:27], null offen offset:128
	buffer_load_b128 v[142:145], v5, s[24:27], null offen offset:160
	buffer_load_b128 v[146:149], v5, s[24:27], null offen offset:192
	buffer_load_b128 v[150:153], v5, s[24:27], null offen offset:224
	s_add_co_i32 s3, s45, s3
	s_lshl_b32 s48, s8, 4
	s_mul_i32 s5, s3, s5
	s_mul_i32 s6, s6, s48
	v_add_lshl_u32 v1, s5, v1, 2
	s_set_vgpr_msb 8
	v_add_lshl_u32 v15, s5, v243 /*v755*/, 2
	v_add_lshl_u32 v16, s5, v245 /*v757*/, 2
	s_lshl4_add_u32 s6, s9, s6
	v_add_lshl_u32 v17, s5, v247 /*v759*/, 2
	s_clause 0x1
	s_load_b64 s[12:13], s[0:1], 0x30 nv
	s_load_b64 s[16:17], s[0:1], 0x60 nv
	s_ashr_i32 s3, s2, 31
	s_cmp_lt_i32 s2, 1
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0xd
	scratch_store_b128 off, v[6:9], off offset:3648 nv
	s_wait_loadcnt 0xc
	scratch_store_b128 off, v[10:13], off offset:3664 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:96
	s_set_vgpr_msb 0x80
	v_mov_b64_e32 v[74:75] /*v[586:587]*/, v[24:25]
	v_mov_b64_e32 v[72:73] /*v[584:585]*/, v[22:23]
	v_mov_b64_e32 v[70:71] /*v[582:583]*/, v[20:21]
	v_mov_b64_e32 v[68:69] /*v[580:581]*/, v[18:19]
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6640 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6656 nv
	s_set_vgpr_msb 0x8000
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6672 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6688 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3680 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3696 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3712 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3728 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6704 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6720 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6736 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6752 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v4, v77 /*v589*/, s10, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3744 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3760 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:32
	v_or_b32_e32 v4, v4, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3776 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3792 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v4, 4, v4
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6768 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6784 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6800 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6816 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3808 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3824 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3840 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3856 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6832 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6848 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6864 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6880 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v3, v79 /*v591*/, s10, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3872 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3888 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:32
	v_or_b32_e32 v3, v3, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3904 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3920 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v3, 4, v3
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6896 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6912 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6928 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6944 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3936 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3952 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:3968 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:3984 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6960 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:6976 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:6992 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7008 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v4, v81 /*v593*/, s10, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4000 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4016 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:32
	v_or_b32_e32 v4, v4, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5616 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5632 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v4, 4, v4
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7024 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7040 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7056 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7072 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4032 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4048 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7088 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7104 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7120 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7136 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7152 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7168 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v3, v125 /*v637*/, s10, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4064 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4080 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:32
	v_or_b32_e32 v3, v3, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7184 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7200 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v3, 4, v3
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7216 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7232 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7248 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7264 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4096 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4112 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5648 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5664 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7280 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7296 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7312 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7328 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v4, v127 /*v639*/, s10, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4128 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4144 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:32
	v_or_b32_e32 v4, v4, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7344 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7360 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v4, 4, v4
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7376 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7392 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7408 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7424 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4160 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4176 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7440 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7456 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7472 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7488 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7504 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7520 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v3, v128 /*v640*/, s10, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4192 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4208 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:32
	v_or_b32_e32 v3, v3, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5680 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5696 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v3, 4, v3
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7536 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7552 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7568 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7584 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4224 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4240 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5712 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5728 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7600 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7616 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7632 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7648 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v4, v129 /*v641*/, s10, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4256 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4272 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:32
	v_or_b32_e32 v4, v4, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7664 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7680 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v4, 4, v4
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7696 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7712 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7728 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7744 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4288 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4304 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7760 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7776 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7792 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7808 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7856 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7872 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v3, v243 /*v755*/, s10, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4320 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4336 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:32
	v_or_b32_e32 v3, v3, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5744 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5760 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v3, 4, v3
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7888 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7904 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7920 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7936 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4352 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4368 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5776 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5792 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7952 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:7968 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:7984 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:8000 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v4, v245 /*v757*/, s10, s14
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4384 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4400 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:32
	v_or_b32_e32 v4, v4, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5808 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5824 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v4, 4, v4
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:8048 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:8064 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:8080 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:8096 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4416 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4432 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5840 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5856 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:8112 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:8128 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:8144 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:8160 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v3, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v3, s[24:27], null offen offset:224
	s_set_vgpr_msb 2
	v_mad_u32 v3, v247 /*v759*/, s10, s14
	s_mov_b32 s10, 1
	s_mov_b64 s[14:15], 0x800000
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4448 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4464 nv
	s_set_vgpr_msb 0x200
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:32
	v_or_b32_e32 v3, v3, v2
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5872 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5888 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:96
	v_lshlrev_b32_e32 v3, 4, v3
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:8176 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:8192 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5904 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5920 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[20:23], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4512 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4528 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5936 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5952 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:64
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:8208 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:8224 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:128
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:5968 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:5984 nv
	s_clause 0x1
	buffer_load_b128 v[6:9], v4, s[24:27], null offen offset:192
	buffer_load_b128 v[10:13], v4, s[24:27], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[6:9], off offset:4544 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[10:13], off offset:4560 nv
	s_clause 0x1
	buffer_load_b128 v[4:7], v3, s[20:23], null offen
	buffer_load_b128 v[8:11], v3, s[20:23], null offen offset:32
	s_set_vgpr_msb 8
	v_add_lshl_u32 v12, s5, v127 /*v639*/, 2
	v_add_lshl_u32 v13, s5, v128 /*v640*/, 2
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[4:7], off offset:6000 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[8:11], off offset:6016 nv
	s_clause 0x1
	buffer_load_b128 v[4:7], v3, s[20:23], null offen offset:64
	buffer_load_b128 v[8:11], v3, s[20:23], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[4:7], off offset:8240 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[8:11], off offset:8256 nv
	s_clause 0x1
	buffer_load_b128 v[4:7], v3, s[20:23], null offen offset:128
	buffer_load_b128 v[8:11], v3, s[20:23], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[4:7], off offset:6032 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[8:11], off offset:6048 nv
	s_clause 0x1
	buffer_load_b128 v[4:7], v3, s[20:23], null offen offset:192
	buffer_load_b128 v[8:11], v3, s[20:23], null offen offset:224
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[4:7], off offset:4576 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[8:11], off offset:4592 nv
	s_clause 0x1
	buffer_load_b128 v[4:7], v3, s[24:27], null offen
	buffer_load_b128 v[8:11], v3, s[24:27], null offen offset:32
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[4:7], off offset:6064 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[8:11], off offset:6080 nv
	s_clause 0x1
	buffer_load_b128 v[4:7], v3, s[24:27], null offen offset:64
	buffer_load_b128 v[8:11], v3, s[24:27], null offen offset:96
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[4:7], off offset:8272 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[8:11], off offset:8288 nv
	s_clause 0x1
	buffer_load_b128 v[4:7], v3, s[24:27], null offen offset:128
	buffer_load_b128 v[8:11], v3, s[24:27], null offen offset:160
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[4:7], off offset:6096 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[8:11], off offset:6112 nv
	s_clause 0x1
	buffer_load_b128 v[4:7], v3, s[24:27], null offen offset:192
	buffer_load_b128 v[8:11], v3, s[24:27], null offen offset:224
	s_wait_xcnt 0x0
	v_and_b32_e32 v3, 16, v0
	s_clause 0x1
	s_load_b64 s[24:25], s[0:1], 0xc0 nv
	s_load_b64 s[20:21], s[0:1], 0xe8 nv
	s_mov_b32 s26, 0x200000
	s_wait_loadcnt 0x1
	scratch_store_b128 off, v[4:7], off offset:4608 nv
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[8:11], off offset:4624 nv
	s_wait_xcnt 0x0
	v_or_b32_e32 v4, 16, v0
	s_set_vgpr_msb 64
	v_mad_u32_u24 v157 /*v413*/, 0x110, v14, v3
	s_mov_b32 s22, s26
	s_set_vgpr_msb 0x4004
	v_add_lshl_u32 v5, s5, v103 /*v359*/, 2
	v_add_lshl_u32 v6, s5, v104 /*v360*/, 2
	s_set_vgpr_msb 0x400
	v_mad_u32_u24 v221, 0x110, v4, v3
	s_set_vgpr_msb 16
	v_and_or_b32 v3, v0, 7, v155 /*v411*/
	scratch_store_b32 off, v4, off offset:8924 nv
	s_wait_xcnt 0x0
	v_add_lshl_u32 v4, s5, v173, 2
	s_set_vgpr_msb 0x1004
	v_add_lshl_u32 v7, s5, v105 /*v361*/, 2
	s_set_vgpr_msb 0x408
	v_add_lshl_u32 v9, s5, v79 /*v591*/, 2
	s_set_vgpr_msb 0x800
	v_mul_u32_u24_e32 v220, 0x110, v3
	v_bfe_u32 v3, v0, 3, 1
	s_set_vgpr_msb 8
	v_add_lshl_u32 v10, s5, v81 /*v593*/, 2
	v_add_lshl_u32 v11, s5, v125 /*v637*/, 2
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v230, 4, v3 :: v_dual_lshrrev_b32 v3, 3, v0
	s_set_vgpr_msb 0x80
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v249 /*v761*/, 4, v3
	s_set_vgpr_msb 0x8000
	v_add_lshl_u32 v3, s5, v171, 2
	s_set_vgpr_msb 0x80
	s_wait_kmcnt 0x0
	buffer_load_b32 v242 /*v754*/, v1, s[20:23], null offen
	s_set_vgpr_msb 0x80c0
	s_clause 0x1
	buffer_load_b32 v34 /*v802*/, v1, s[24:27], null offen
	buffer_load_b32 v104 /*v872*/, v3, s[24:27], null offen
	s_set_vgpr_msb 0xc080
	s_clause 0x2
	buffer_load_b32 v244 /*v756*/, v3, s[20:23], null offen
	s_set_vgpr_msb 0x8000
	buffer_load_b32 v82, v4, s[20:23], null offen
	s_set_vgpr_msb 0xc0
	s_clause 0x1
	buffer_load_b32 v254 /*v1022*/, v4, s[24:27], null offen
	buffer_load_b32 v252 /*v1020*/, v5, s[24:27], null offen
	s_set_vgpr_msb 0xc008
	buffer_load_b32 v4, v5, s[20:23], null offen
	v_add_lshl_u32 v8, s5, v77 /*v589*/, 2
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v1, s6, v2
	scratch_store_b32 off, v1, off offset:8304 nv
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[4:5], off offset:6152 nv
	buffer_load_b32 v4, v6, s[20:23], null offen
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[4:5], off offset:6144 nv
	s_set_vgpr_msb 64
	s_clause 0x2
	buffer_load_b32 v158 /*v414*/, v6, s[24:27], null offen
	s_set_vgpr_msb 0x4000
	buffer_load_b32 v218, v7, s[24:27], null offen
	buffer_load_b32 v4, v7, s[20:23], null offen
	scratch_store_b32 off, v14, off offset:6160 nv
	s_set_vgpr_msb 8
	v_add_lshl_u32 v14, s5, v129 /*v641*/, 2
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[4:5], off offset:6128 nv
	buffer_load_b32 v4, v8, s[20:23], null offen
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[4:5], off offset:6136 nv
	s_set_vgpr_msb 0xc0
	s_clause 0x1
	buffer_load_b32 v250 /*v1018*/, v8, s[24:27], null offen
	buffer_load_b32 v94 /*v862*/, v9, s[24:27], null offen
	s_clause 0x1
	buffer_load_b32 v236 /*v1004*/, v9, s[20:23], null offen
	buffer_load_b32 v212 /*v980*/, v10, s[20:23], null offen
	s_clause 0x2
	buffer_load_b32 v6 /*v774*/, v10, s[24:27], null offen
	s_set_vgpr_msb 0xc040
	buffer_load_b32 v156 /*v412*/, v11, s[24:27], null offen
	s_clause 0x2
	buffer_load_b32 v154 /*v410*/, v11, s[20:23], null offen
	s_set_vgpr_msb 0x40c0
	buffer_load_b32 v42 /*v810*/, v12, s[20:23], null offen
	s_clause 0x2
	buffer_load_b32 v170 /*v938*/, v12, s[24:27], null offen
	s_set_vgpr_msb 0xc040
	buffer_load_b32 v160 /*v416*/, v13, s[24:27], null offen
	s_set_vgpr_msb 0x40c0
	s_clause 0x1
	buffer_load_b32 v214 /*v982*/, v13, s[20:23], null offen
	buffer_load_b32 v216 /*v984*/, v14, s[20:23], null offen
	s_clause 0x2
	buffer_load_b32 v238 /*v1006*/, v14, s[24:27], null offen
	s_set_vgpr_msb 0xc040
	buffer_load_b32 v102 /*v358*/, v15, s[24:27], null offen
	s_set_vgpr_msb 0x4000
	buffer_load_b32 v4, v15, s[20:23], null offen
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[4:5], off offset:8308 nv
	buffer_load_b32 v4, v16, s[20:23], null offen
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[4:5], off offset:5536 nv
	buffer_load_b32 v4, v16, s[24:27], null offen
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[4:5], off offset:8820 nv
	buffer_load_b32 v4, v17, s[24:27], null offen
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[4:5], off offset:8828 nv
	buffer_load_b32 v4, v17, s[20:23], null offen
	s_clause 0x3e
	scratch_store_b128 off, v[18:21], off offset:5552 nv
	scratch_store_b128 off, v[22:25], off offset:5568 nv
	s_set_vgpr_msb 4
	scratch_store_b32 off, v155 /*v411*/, off offset:8836 nv
	s_set_vgpr_msb 0x400
	scratch_store_b32 off, v0, off offset:8980 nv
	scratch_store_b128 off, v[246:249], off offset:5584 nv
	scratch_store_b128 off, v[250:253], off offset:5600 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[14:17] /*v[270:273]*/, off offset:3584 nv
	scratch_store_b128 off, v[18:21] /*v[274:277]*/, off offset:3600 nv
	s_set_vgpr_msb 0x400
	scratch_store_b128 off, v[154:157], off offset:6608 nv
	scratch_store_b128 off, v[158:161], off offset:6624 nv
	s_set_vgpr_msb 8
	scratch_store_b128 off, v[36:39] /*v[548:551]*/, off offset:3616 nv
	scratch_store_b128 off, v[40:43] /*v[552:555]*/, off offset:3632 nv
	s_set_vgpr_msb 0x800
	scratch_store_b128 off, v[162:165], off offset:7824 nv
	scratch_store_b128 off, v[166:169], off offset:7840 nv
	scratch_store_b128 off, v[106:109], off offset:8016 nv
	scratch_store_b128 off, v[110:113], off offset:8032 nv
	s_set_vgpr_msb 8
	scratch_store_b128 off, v[138:141] /*v[650:653]*/, off offset:4480 nv
	scratch_store_b128 off, v[142:145] /*v[654:657]*/, off offset:4496 nv
	s_set_vgpr_msb 0x800
	scratch_store_b128 off, v[138:141], off offset:6220 nv
	scratch_store_b128 off, v[142:145], off offset:6236 nv
	scratch_store_b128 off, v[146:149], off offset:3040 nv
	scratch_store_b128 off, v[150:153], off offset:3056 nv
	scratch_store_b128 off, v[174:177], off offset:3072 nv
	scratch_store_b128 off, v[178:181], off offset:3088 nv
	scratch_store_b128 off, v[182:185], off offset:6252 nv
	scratch_store_b128 off, v[186:189], off offset:6268 nv
	scratch_store_b128 off, v[190:193], off offset:6284 nv
	scratch_store_b128 off, v[194:197], off offset:6300 nv
	scratch_store_b128 off, v[198:201], off offset:3104 nv
	scratch_store_b128 off, v[202:205], off offset:3120 nv
	scratch_store_b32 off, v221, off offset:6316 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[6:9] /*v[262:265]*/, off offset:6320 nv
	scratch_store_b128 off, v[10:13] /*v[266:269]*/, off offset:6336 nv
	s_set_vgpr_msb 0x400
	scratch_store_b128 off, v[122:125], off offset:3136 nv
	scratch_store_b128 off, v[126:129], off offset:3152 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[94:97] /*v[350:353]*/, off offset:6352 nv
	scratch_store_b128 off, v[98:101] /*v[354:357]*/, off offset:6368 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[2:5] /*v[514:517]*/, off offset:6384 nv
	scratch_store_b128 off, v[6:9] /*v[518:521]*/, off offset:6400 nv
	scratch_store_b128 off, v[84:87] /*v[596:599]*/, off offset:3168 nv
	scratch_store_b128 off, v[88:91] /*v[600:603]*/, off offset:3184 nv
	s_set_vgpr_msb 0x804
	scratch_store_b128 off, v[242:245] /*v[498:501]*/, off offset:6416 nv
	scratch_store_b128 off, v[246:249] /*v[502:505]*/, off offset:6432 nv
	s_set_vgpr_msb 0x400
	scratch_store_b128 off, v[254:257], off offset:3200 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[2:5] /*v[258:261]*/, off offset:3216 nv
	s_set_vgpr_msb 0x400
	scratch_store_b128 off, v[90:93], off offset:6448 nv
	scratch_store_b128 off, v[94:97], off offset:6464 nv
	scratch_store_b128 off, v[98:101], off offset:3232 nv
	scratch_store_b128 off, v[102:105], off offset:3248 nv
	scratch_store_b128 off, v[114:117], off offset:6480 nv
	scratch_store_b128 off, v[118:121], off offset:6496 nv
	scratch_store_b128 off, v[130:133], off offset:3264 nv
	scratch_store_b128 off, v[134:137], off offset:3280 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:6512 nv
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:6528 nv
	scratch_store_b128 off, v[210:213] /*v[466:469]*/, off offset:3296 nv
	scratch_store_b128 off, v[214:217] /*v[470:473]*/, off offset:3312 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[28:31] /*v[540:543]*/, off offset:6544 nv
	scratch_store_b128 off, v[32:35] /*v[544:547]*/, off offset:6560 nv
	scratch_store_b128 off, v[100:103] /*v[612:615]*/, off offset:3328 nv
	scratch_store_b128 off, v[104:107] /*v[616:619]*/, off offset:3344 nv
	s_set_vgpr_msb 0x800
	scratch_store_b128 off, v[222:225], off offset:6576 nv
	scratch_store_b128 off, v[226:229], off offset:6592 nv
	s_clause 0x4
	scratch_store_b128 off, v[238:241], off offset:3360 nv
	scratch_store_b128 off, v[242:245], off offset:3376 nv
	s_set_vgpr_msb 8
	scratch_store_b128 off, v[92:95] /*v[604:607]*/, off offset:3392 nv
	scratch_store_b128 off, v[96:99] /*v[608:611]*/, off offset:3408 nv
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	scratch_store_b64 off, v[4:5], off offset:5544 nv
	s_cbranch_scc1 .LBB0_4
	v_or_b32_e32 v1, 16, v0
	s_set_vgpr_msb 8
	s_clause 0x10
	scratch_store_b32 off, v247 /*v759*/, off offset:9040 nv
	scratch_store_b32 off, v245 /*v757*/, off offset:9036 nv
	scratch_store_b32 off, v243 /*v755*/, off offset:9032 nv
	scratch_store_b32 off, v129 /*v641*/, off offset:9028 nv
	scratch_store_b32 off, v128 /*v640*/, off offset:9024 nv
	scratch_store_b32 off, v127 /*v639*/, off offset:9020 nv
	scratch_store_b32 off, v125 /*v637*/, off offset:9016 nv
	scratch_store_b32 off, v81 /*v593*/, off offset:9012 nv
	scratch_store_b32 off, v79 /*v591*/, off offset:9008 nv
	scratch_store_b32 off, v77 /*v589*/, off offset:9004 nv
	s_set_vgpr_msb 0x804
	scratch_store_b32 off, v105 /*v361*/, off offset:9000 nv
	scratch_store_b32 off, v104 /*v360*/, off offset:8996 nv
	scratch_store_b32 off, v103 /*v359*/, off offset:8992 nv
	s_set_vgpr_msb 0x4c3
	scratch_store_b32 off, v173, off offset:8988 nv
	scratch_store_b32 off, v171, off offset:8984 nv
	v_dual_mov_b32 v217 /*v985*/, v216 /*v984*/ :: v_dual_mov_b32 v215 /*v983*/, v214 /*v982*/
	s_set_vgpr_msb 0xc382
	v_dual_mov_b32 v245 /*v757*/, v244 /*v756*/ :: v_dual_mov_b32 v243 /*v755*/, v242 /*v754*/
	s_set_vgpr_msb 0x8200
	v_mad_u32 v1, s48, v1, s6
	s_set_vgpr_msb 0xc3
	v_dual_mov_b32 v43 /*v811*/, v42 /*v810*/ :: v_dual_mov_b32 v213 /*v981*/, v212 /*v980*/
	v_dual_mov_b32 v171 /*v939*/, v170 /*v938*/ :: v_dual_mov_b32 v239 /*v1007*/, v238 /*v1006*/
	v_dual_mov_b32 v44 /*v812*/, 0 :: v_dual_mov_b32 v95 /*v863*/, v94 /*v862*/
	v_mov_b32_e32 v7 /*v775*/, v6 /*v774*/
	s_set_vgpr_msb 0xc300
	v_or_b32_e32 v1, v1, v2
	v_and_b32_e32 v3, 15, v0
	s_set_vgpr_msb 0xc3
	v_dual_mov_b32 v105 /*v873*/, v104 /*v872*/ :: v_dual_mov_b32 v255 /*v1023*/, v254 /*v1022*/
	s_set_vgpr_msb 0xc341
	v_mov_b32_e32 v159 /*v415*/, v158 /*v414*/
	s_set_vgpr_msb 0x4100
	v_lshlrev_b32_e32 v1, 4, v1
	v_mad_u32 v3, s48, v3, s6
	s_set_vgpr_msb 12
	s_clause 0x2
	scratch_store_b64 off, v[94:95] /*v[862:863]*/, off offset:8892 nv
	s_set_vgpr_msb 0xc45
	scratch_store_b32 off, v157 /*v413*/, off offset:8840 nv
	s_wait_xcnt 0x0
	v_mov_b32_e32 v157 /*v413*/, v156 /*v412*/
	s_set_vgpr_msb 0x4500
	v_dual_mov_b32 v219, v218 :: v_dual_bitop2_b32 v8, 64, v1 bitop3:0x54
	v_or_b32_e32 v4, 0xc0, v1
	v_or_b32_e32 v5, 0xa0, v1
	v_or_b32_e32 v6, 0x80, v1
	v_or_b32_e32 v9, 32, v1
	v_dual_mov_b32 v83, v82 :: v_dual_bitop2_b32 v2, v3, v2 bitop3:0x54
	v_or_b32_e32 v3, 0xe0, v1
	v_or_b32_e32 v7, 0x60, v1
	s_clause 0x1
	buffer_load_b128 v[10:13], v4, s[16:19], null offen
	buffer_load_b128 v[210:213], v5, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[138:141], v4, s[12:15], null offen
	buffer_load_b128 v[150:153], v5, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[206:209], v6, s[16:19], null offen
	buffer_load_b128 v[174:177], v7, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[146:149], v6, s[12:15], null offen
	buffer_load_b128 v[182:185], v7, s[12:15], null offen
	s_clause 0x3
	buffer_load_b128 v[170:173], v8, s[16:19], null offen
	buffer_load_b128 v[190:193], v9, s[16:19], null offen
	buffer_load_b128 v[14:17], v3, s[16:19], null offen
	buffer_load_b128 v[186:189], v1, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[178:181], v8, s[12:15], null offen
	buffer_load_b128 v[202:205], v9, s[12:15], null offen
	scratch_load_b64 v[8:9], off, off offset:5544 nv
	v_lshlrev_b32_e32 v2, 4, v2
	s_clause 0x1
	buffer_load_b128 v[142:145], v3, s[12:15], null offen
	buffer_load_b128 v[198:201], v1, s[12:15], null offen
	v_or_b32_e32 v7, 0xc0, v230
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v161 /*v417*/, v160 /*v416*/ :: v_dual_mov_b32 v103 /*v359*/, v102 /*v358*/
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v4, 0xc0, v2
	v_or_b32_e32 v5, 0xa0, v2
	v_or_b32_e32 v3, 0x80, v2
	v_or_b32_e32 v6, 0x60, v2
	v_or_b32_e32 v1, 0xe0, v2
	s_clause 0x1
	buffer_load_b128 v[18:21], v4, s[16:19], null offen
	buffer_load_b128 v[38:41], v5, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[26:29], v4, s[12:15], null offen
	buffer_load_b128 v[46:49], v5, s[12:15], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v4, 64, v2
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 32, v2
	s_clause 0x1
	buffer_load_b128 v[34:37], v3, s[16:19], null offen
	buffer_load_b128 v[54:57], v6, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[42:45], v3, s[12:15], null offen
	buffer_load_b128 v[62:65], v6, s[12:15], null offen
	s_clause 0x3
	buffer_load_b128 v[50:53], v4, s[16:19], null offen
	buffer_load_b128 v[70:73], v5, s[16:19], null offen
	buffer_load_b128 v[22:25], v1, s[16:19], null offen
	buffer_load_b128 v[66:69], v2, s[16:19], null offen
	s_clause 0x3
	buffer_load_b128 v[58:61], v4, s[12:15], null offen
	buffer_load_b128 v[78:81], v5, s[12:15], null offen
	buffer_load_b128 v[30:33], v1, s[12:15], null offen
	buffer_load_b128 v[74:77], v2, s[12:15], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, 32, v249 /*v761*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v3, 1, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v4, 0x60, v249 /*v761*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v5, 0x80, v230
	s_set_vgpr_msb 8
	v_or_b32_e32 v6, 0xa0, v249 /*v761*/
	s_set_vgpr_msb 0x8c3
	v_dual_mov_b32 v237 /*v1005*/, v236 /*v1004*/ :: v_dual_mov_b32 v35 /*v803*/, v34 /*v802*/
	s_set_vgpr_msb 0xc300
	v_and_or_b32 v3, v3, 16, 0xe0
	s_set_vgpr_msb 0xc3
	v_dual_mov_b32 v253 /*v1021*/, v252 /*v1020*/ :: v_dual_mov_b32 v251 /*v1019*/, v250 /*v1018*/
	v_dual_mov_b32 v45 /*v813*/, v44 /*v812*/ :: v_dual_mov_b32 v46 /*v814*/, v44 /*v812*/
	v_dual_mov_b32 v47 /*v815*/, v44 /*v812*/ :: v_dual_mov_b32 v48 /*v816*/, v44 /*v812*/
	v_dual_mov_b32 v49 /*v817*/, v44 /*v812*/ :: v_dual_mov_b32 v50 /*v818*/, v44 /*v812*/
	v_dual_mov_b32 v51 /*v819*/, v44 /*v812*/ :: v_dual_mov_b32 v204 /*v972*/, v44 /*v812*/
	s_set_vgpr_msb 0xc383
	v_dual_mov_b32 v20 /*v532*/, v44 /*v812*/ :: v_dual_mov_b32 v21 /*v533*/, v44 /*v812*/
	v_dual_mov_b32 v22 /*v534*/, v44 /*v812*/ :: v_dual_mov_b32 v23 /*v535*/, v44 /*v812*/
	v_dual_mov_b32 v24 /*v536*/, v44 /*v812*/ :: v_dual_mov_b32 v25 /*v537*/, v44 /*v812*/
	v_dual_mov_b32 v26 /*v538*/, v44 /*v812*/ :: v_dual_mov_b32 v27 /*v539*/, v44 /*v812*/
	s_set_vgpr_msb 0x83c3
	v_dual_mov_b32 v205 /*v973*/, v44 /*v812*/ :: v_dual_mov_b32 v206 /*v974*/, v44 /*v812*/
	v_dual_mov_b32 v207 /*v975*/, v44 /*v812*/ :: v_dual_mov_b32 v208 /*v976*/, v44 /*v812*/
	v_dual_mov_b32 v209 /*v977*/, v44 /*v812*/ :: v_dual_mov_b32 v210 /*v978*/, v44 /*v812*/
	v_dual_mov_b32 v211 /*v979*/, v44 /*v812*/ :: v_dual_mov_b32 v196 /*v964*/, v44 /*v812*/
	s_set_vgpr_msb 0xc383
	v_dual_mov_b32 v76 /*v588*/, v44 /*v812*/ :: v_dual_mov_b32 v77 /*v589*/, v44 /*v812*/
	v_dual_mov_b32 v78 /*v590*/, v44 /*v812*/ :: v_dual_mov_b32 v79 /*v591*/, v44 /*v812*/
	v_dual_mov_b32 v80 /*v592*/, v44 /*v812*/ :: v_dual_mov_b32 v81 /*v593*/, v44 /*v812*/
	v_dual_mov_b32 v82 /*v594*/, v44 /*v812*/ :: v_dual_mov_b32 v83 /*v595*/, v44 /*v812*/
	s_set_vgpr_msb 0x83c3
	v_dual_mov_b32 v197 /*v965*/, v44 /*v812*/ :: v_dual_mov_b32 v198 /*v966*/, v44 /*v812*/
	v_dual_mov_b32 v199 /*v967*/, v44 /*v812*/ :: v_dual_mov_b32 v200 /*v968*/, v44 /*v812*/
	v_dual_mov_b32 v201 /*v969*/, v44 /*v812*/ :: v_dual_mov_b32 v202 /*v970*/, v44 /*v812*/
	v_mov_b32_e32 v203 /*v971*/, v44 /*v812*/
	s_set_vgpr_msb 0xc383
	v_dual_mov_b32 v170 /*v682*/, v44 /*v812*/ :: v_dual_mov_b32 v171 /*v683*/, v44 /*v812*/
	v_dual_mov_b32 v172 /*v684*/, v44 /*v812*/ :: v_dual_mov_b32 v173 /*v685*/, v44 /*v812*/
	v_dual_mov_b32 v174 /*v686*/, v44 /*v812*/ :: v_dual_mov_b32 v175 /*v687*/, v44 /*v812*/
	v_dual_mov_b32 v176 /*v688*/, v44 /*v812*/ :: v_dual_mov_b32 v177 /*v689*/, v44 /*v812*/
	s_set_vgpr_msb 0x8343
	v_dual_mov_b32 v186 /*v442*/, v44 /*v812*/ :: v_dual_mov_b32 v187 /*v443*/, v44 /*v812*/
	v_dual_mov_b32 v188 /*v444*/, v44 /*v812*/ :: v_dual_mov_b32 v189 /*v445*/, v44 /*v812*/
	v_dual_mov_b32 v190 /*v446*/, v44 /*v812*/ :: v_dual_mov_b32 v191 /*v447*/, v44 /*v812*/
	v_dual_mov_b32 v192 /*v448*/, v44 /*v812*/ :: v_dual_mov_b32 v193 /*v449*/, v44 /*v812*/
	v_dual_mov_b32 v138 /*v394*/, v44 /*v812*/ :: v_dual_mov_b32 v139 /*v395*/, v44 /*v812*/
	v_dual_mov_b32 v140 /*v396*/, v44 /*v812*/ :: v_dual_mov_b32 v141 /*v397*/, v44 /*v812*/
	v_dual_mov_b32 v142 /*v398*/, v44 /*v812*/ :: v_dual_mov_b32 v143 /*v399*/, v44 /*v812*/
	v_dual_mov_b32 v144 /*v400*/, v44 /*v812*/ :: v_dual_mov_b32 v145 /*v401*/, v44 /*v812*/
	s_set_vgpr_msb 0x4383
	v_dual_mov_b32 v44 /*v556*/, v44 /*v812*/ :: v_dual_mov_b32 v45 /*v557*/, v44 /*v812*/
	v_dual_mov_b32 v46 /*v558*/, v44 /*v812*/ :: v_dual_mov_b32 v47 /*v559*/, v44 /*v812*/
	v_dual_mov_b32 v48 /*v560*/, v44 /*v812*/ :: v_dual_mov_b32 v49 /*v561*/, v44 /*v812*/
	v_dual_mov_b32 v50 /*v562*/, v44 /*v812*/ :: v_dual_mov_b32 v51 /*v563*/, v44 /*v812*/
	v_dual_mov_b32 v154 /*v666*/, v44 /*v812*/ :: v_dual_mov_b32 v155 /*v667*/, v44 /*v812*/
	v_dual_mov_b32 v156 /*v668*/, v44 /*v812*/ :: v_dual_mov_b32 v157 /*v669*/, v44 /*v812*/
	v_dual_mov_b32 v158 /*v670*/, v44 /*v812*/ :: v_dual_mov_b32 v159 /*v671*/, v44 /*v812*/
	v_dual_mov_b32 v160 /*v672*/, v44 /*v812*/ :: v_dual_mov_b32 v161 /*v673*/, v44 /*v812*/
	s_set_vgpr_msb 0x8343
	v_dual_mov_b32 v130 /*v386*/, v44 /*v812*/ :: v_dual_mov_b32 v131 /*v387*/, v44 /*v812*/
	v_dual_mov_b32 v132 /*v388*/, v44 /*v812*/ :: v_dual_mov_b32 v133 /*v389*/, v44 /*v812*/
	v_dual_mov_b32 v134 /*v390*/, v44 /*v812*/ :: v_dual_mov_b32 v135 /*v391*/, v44 /*v812*/
	v_dual_mov_b32 v136 /*v392*/, v44 /*v812*/ :: v_dual_mov_b32 v137 /*v393*/, v44 /*v812*/
	v_dual_mov_b32 v122 /*v378*/, v44 /*v812*/ :: v_dual_mov_b32 v123 /*v379*/, v44 /*v812*/
	v_dual_mov_b32 v124 /*v380*/, v44 /*v812*/ :: v_dual_mov_b32 v125 /*v381*/, v44 /*v812*/
	v_dual_mov_b32 v126 /*v382*/, v44 /*v812*/ :: v_dual_mov_b32 v127 /*v383*/, v44 /*v812*/
	v_dual_mov_b32 v128 /*v384*/, v44 /*v812*/ :: v_dual_mov_b32 v129 /*v385*/, v44 /*v812*/
	v_dual_mov_b32 v70 /*v326*/, v44 /*v812*/ :: v_dual_mov_b32 v71 /*v327*/, v44 /*v812*/
	v_dual_mov_b32 v72 /*v328*/, v44 /*v812*/ :: v_dual_mov_b32 v73 /*v329*/, v44 /*v812*/
	v_dual_mov_b32 v74 /*v330*/, v44 /*v812*/ :: v_dual_mov_b32 v75 /*v331*/, v44 /*v812*/
	v_dual_mov_b32 v76 /*v332*/, v44 /*v812*/ :: v_dual_mov_b32 v77 /*v333*/, v44 /*v812*/
	v_dual_mov_b32 v62 /*v318*/, v44 /*v812*/ :: v_dual_mov_b32 v63 /*v319*/, v44 /*v812*/
	v_dual_mov_b32 v64 /*v320*/, v44 /*v812*/ :: v_dual_mov_b32 v65 /*v321*/, v44 /*v812*/
	v_dual_mov_b32 v66 /*v322*/, v44 /*v812*/ :: v_dual_mov_b32 v67 /*v323*/, v44 /*v812*/
	v_dual_mov_b32 v68 /*v324*/, v44 /*v812*/ :: v_dual_mov_b32 v69 /*v325*/, v44 /*v812*/
	s_set_vgpr_msb 0x4383
	v_dual_mov_b32 v52 /*v564*/, v44 /*v812*/ :: v_dual_mov_b32 v53 /*v565*/, v44 /*v812*/
	v_dual_mov_b32 v54 /*v566*/, v44 /*v812*/ :: v_dual_mov_b32 v55 /*v567*/, v44 /*v812*/
	v_dual_mov_b32 v56 /*v568*/, v44 /*v812*/ :: v_dual_mov_b32 v57 /*v569*/, v44 /*v812*/
	v_dual_mov_b32 v58 /*v570*/, v44 /*v812*/ :: v_dual_mov_b32 v59 /*v571*/, v44 /*v812*/
	s_set_vgpr_msb 0x8343
	v_dual_mov_b32 v194 /*v450*/, v44 /*v812*/ :: v_dual_mov_b32 v195 /*v451*/, v44 /*v812*/
	v_dual_mov_b32 v196 /*v452*/, v44 /*v812*/ :: v_dual_mov_b32 v197 /*v453*/, v44 /*v812*/
	v_dual_mov_b32 v198 /*v454*/, v44 /*v812*/ :: v_dual_mov_b32 v199 /*v455*/, v44 /*v812*/
	v_dual_mov_b32 v200 /*v456*/, v44 /*v812*/ :: v_dual_mov_b32 v201 /*v457*/, v44 /*v812*/
	v_dual_mov_b32 v234 /*v490*/, v44 /*v812*/ :: v_dual_mov_b32 v235 /*v491*/, v44 /*v812*/
	v_dual_mov_b32 v236 /*v492*/, v44 /*v812*/ :: v_dual_mov_b32 v237 /*v493*/, v44 /*v812*/
	v_dual_mov_b32 v238 /*v494*/, v44 /*v812*/ :: v_dual_mov_b32 v239 /*v495*/, v44 /*v812*/
	v_dual_mov_b32 v240 /*v496*/, v44 /*v812*/ :: v_dual_mov_b32 v241 /*v497*/, v44 /*v812*/
	v_dual_mov_b32 v178 /*v434*/, v44 /*v812*/ :: v_dual_mov_b32 v179 /*v435*/, v44 /*v812*/
	v_dual_mov_b32 v180 /*v436*/, v44 /*v812*/ :: v_dual_mov_b32 v181 /*v437*/, v44 /*v812*/
	v_dual_mov_b32 v182 /*v438*/, v44 /*v812*/ :: v_dual_mov_b32 v183 /*v439*/, v44 /*v812*/
	v_dual_mov_b32 v184 /*v440*/, v44 /*v812*/ :: v_dual_mov_b32 v185 /*v441*/, v44 /*v812*/
	v_dual_mov_b32 v114 /*v370*/, v44 /*v812*/ :: v_dual_mov_b32 v115 /*v371*/, v44 /*v812*/
	v_dual_mov_b32 v116 /*v372*/, v44 /*v812*/ :: v_dual_mov_b32 v117 /*v373*/, v44 /*v812*/
	v_dual_mov_b32 v118 /*v374*/, v44 /*v812*/ :: v_dual_mov_b32 v119 /*v375*/, v44 /*v812*/
	v_dual_mov_b32 v120 /*v376*/, v44 /*v812*/ :: v_dual_mov_b32 v121 /*v377*/, v44 /*v812*/
	v_dual_mov_b32 v106 /*v362*/, v44 /*v812*/ :: v_dual_mov_b32 v107 /*v363*/, v44 /*v812*/
	v_dual_mov_b32 v108 /*v364*/, v44 /*v812*/ :: v_dual_mov_b32 v109 /*v365*/, v44 /*v812*/
	v_dual_mov_b32 v110 /*v366*/, v44 /*v812*/ :: v_dual_mov_b32 v111 /*v367*/, v44 /*v812*/
	v_dual_mov_b32 v112 /*v368*/, v44 /*v812*/ :: v_dual_mov_b32 v113 /*v369*/, v44 /*v812*/
	v_dual_mov_b32 v162 /*v418*/, v44 /*v812*/ :: v_dual_mov_b32 v163 /*v419*/, v44 /*v812*/
	v_dual_mov_b32 v164 /*v420*/, v44 /*v812*/ :: v_dual_mov_b32 v165 /*v421*/, v44 /*v812*/
	v_dual_mov_b32 v166 /*v422*/, v44 /*v812*/ :: v_dual_mov_b32 v167 /*v423*/, v44 /*v812*/
	v_dual_mov_b32 v168 /*v424*/, v44 /*v812*/ :: v_dual_mov_b32 v169 /*v425*/, v44 /*v812*/
	v_dual_mov_b32 v146 /*v402*/, v44 /*v812*/ :: v_dual_mov_b32 v147 /*v403*/, v44 /*v812*/
	v_dual_mov_b32 v148 /*v404*/, v44 /*v812*/ :: v_dual_mov_b32 v149 /*v405*/, v44 /*v812*/
	v_dual_mov_b32 v150 /*v406*/, v44 /*v812*/ :: v_dual_mov_b32 v151 /*v407*/, v44 /*v812*/
	v_dual_mov_b32 v152 /*v408*/, v44 /*v812*/ :: v_dual_mov_b32 v153 /*v409*/, v44 /*v812*/
	v_dual_mov_b32 v86 /*v342*/, v44 /*v812*/ :: v_dual_mov_b32 v87 /*v343*/, v44 /*v812*/
	v_dual_mov_b32 v88 /*v344*/, v44 /*v812*/ :: v_dual_mov_b32 v89 /*v345*/, v44 /*v812*/
	v_dual_mov_b32 v90 /*v346*/, v44 /*v812*/ :: v_dual_mov_b32 v91 /*v347*/, v44 /*v812*/
	v_dual_mov_b32 v92 /*v348*/, v44 /*v812*/ :: v_dual_mov_b32 v93 /*v349*/, v44 /*v812*/
	v_dual_mov_b32 v78 /*v334*/, v44 /*v812*/ :: v_dual_mov_b32 v79 /*v335*/, v44 /*v812*/
	v_dual_mov_b32 v80 /*v336*/, v44 /*v812*/ :: v_dual_mov_b32 v81 /*v337*/, v44 /*v812*/
	v_dual_mov_b32 v82 /*v338*/, v44 /*v812*/ :: v_dual_mov_b32 v83 /*v339*/, v44 /*v812*/
	v_dual_mov_b32 v84 /*v340*/, v44 /*v812*/ :: v_dual_mov_b32 v85 /*v341*/, v44 /*v812*/
	s_set_vgpr_msb 0x4383
	v_dual_mov_b32 v108 /*v620*/, v44 /*v812*/ :: v_dual_mov_b32 v109 /*v621*/, v44 /*v812*/
	v_dual_mov_b32 v110 /*v622*/, v44 /*v812*/ :: v_dual_mov_b32 v111 /*v623*/, v44 /*v812*/
	v_dual_mov_b32 v112 /*v624*/, v44 /*v812*/ :: v_dual_mov_b32 v113 /*v625*/, v44 /*v812*/
	v_dual_mov_b32 v114 /*v626*/, v44 /*v812*/ :: v_dual_mov_b32 v115 /*v627*/, v44 /*v812*/
	v_dual_mov_b32 v10 /*v522*/, v44 /*v812*/ :: v_dual_mov_b32 v11 /*v523*/, v44 /*v812*/
	v_dual_mov_b32 v12 /*v524*/, v44 /*v812*/ :: v_dual_mov_b32 v13 /*v525*/, v44 /*v812*/
	v_dual_mov_b32 v14 /*v526*/, v44 /*v812*/ :: v_dual_mov_b32 v15 /*v527*/, v44 /*v812*/
	v_dual_mov_b32 v16 /*v528*/, v44 /*v812*/ :: v_dual_mov_b32 v17 /*v529*/, v44 /*v812*/
	s_set_vgpr_msb 0x830f
	v_dual_mov_b32 v84, v44 /*v812*/ :: v_dual_mov_b32 v85, v44 /*v812*/
	v_dual_mov_b32 v86, v44 /*v812*/ :: v_dual_mov_b32 v87, v44 /*v812*/
	v_dual_mov_b32 v88, v44 /*v812*/ :: v_dual_mov_b32 v89, v44 /*v812*/
	s_add_co_i32 s20, s2, 0x7ffffff
	s_mov_b32 s5, s4
	s_mov_b32 s6, 0x3fb8aa3b
	s_mov_b64 s[8:9], s[2:3]
	s_clause 0x4
	scratch_store_b64 off, v[254:255] /*v[1022:1023]*/, off offset:8928 nv
	scratch_store_b64 off, v[252:253] /*v[1020:1021]*/, off offset:8936 nv
	scratch_store_b64 off, v[250:251] /*v[1018:1019]*/, off offset:8944 nv
	s_set_vgpr_msb 0xf4d
	scratch_load_b64 v[104:105] /*v[360:361]*/, off, off offset:8308 th:TH_LOAD_LU nv
	v_mov_b32_e32 v155 /*v411*/, v154 /*v410*/
	s_clause 0x2
	scratch_store_b64 off, v[214:215] /*v[982:983]*/, off offset:6164 nv
	scratch_store_b64 off, v[236:237] /*v[1004:1005]*/, off offset:6212 nv
	scratch_store_b64 off, v[6:7] /*v[774:775]*/, off offset:8876 nv
	s_set_vgpr_msb 0x4dc9
	v_mov_b64_e32 v[214:215] /*v[982:983]*/, v[102:103] /*v[358:359]*/
	s_clause 0x9
	scratch_store_b64 off, v[242:243] /*v[754:755]*/, off offset:6188 nv
	s_set_vgpr_msb 0xc904
	scratch_store_b64 off, v[154:155] /*v[410:411]*/, off offset:8908 nv
	scratch_store_b64 off, v[158:159] /*v[414:415]*/, off offset:8856 nv
	s_set_vgpr_msb 0x408
	scratch_store_b64 off, v[244:245] /*v[756:757]*/, off offset:6180 nv
	s_set_vgpr_msb 0x804
	scratch_store_b64 off, v[156:157] /*v[412:413]*/, off offset:8900 nv
	s_set_vgpr_msb 0x4cf
	scratch_store_b64 off, v[170:171] /*v[938:939]*/, off offset:8848 nv
	s_wait_xcnt 0x0
	v_mov_b64_e32 v[170:171] /*v[938:939]*/, v[42:43] /*v[810:811]*/
	s_set_vgpr_msb 0xcf03
	s_clause 0x1
	scratch_store_b32 off, v220, off offset:8864 nv
	scratch_store_b64 off, v[82:83], off offset:6204 nv
	s_wait_xcnt 0x0
	v_dual_mov_b32 v82, v44 /*v812*/ :: v_dual_mov_b32 v83, v44 /*v812*/
	s_set_vgpr_msb 0x300
	s_wait_loadcnt 0x13
	v_mov_b32_e32 v9, v8
	v_add_nc_u32_e32 v0, v220, v1
	s_set_vgpr_msb 3
	v_mov_b32_e32 v1, v44 /*v812*/
	s_clause 0x1
	scratch_store_b64 off, v[8:9], off offset:5544 nv
	scratch_load_b64 v[8:9], off, off offset:5536 nv
	v_or_b32_e32 v2, 64, v230
	scratch_store_b32 off, v0, off offset:8960 nv
	s_set_vgpr_msb 0x341
	s_wait_loadcnt 0x1
	v_mov_b32_e32 v105 /*v361*/, v104 /*v360*/
	s_set_vgpr_msb 0x4100
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v8
	v_add_nc_u32_e32 v0, v220, v2
	s_set_vgpr_msb 3
	v_mov_b32_e32 v2, v44 /*v812*/
	s_clause 0x2
	scratch_store_b64 off, v[8:9], off offset:5536 nv
	scratch_load_b64 v[8:9], off, off offset:6136 nv
	scratch_store_b32 off, v0, off offset:8964 nv
	s_set_vgpr_msb 0x300
	v_add_nc_u32_e32 v0, v220, v4
	s_set_vgpr_msb 3
	v_mov_b32_e32 v4, v44 /*v812*/
	scratch_store_b32 off, v0, off offset:8968 nv
	s_set_vgpr_msb 0x300
	v_add_nc_u32_e32 v0, v220, v5
	s_set_vgpr_msb 3
	v_mov_b32_e32 v5, v44 /*v812*/
	scratch_store_b32 off, v0, off offset:8972 nv
	s_set_vgpr_msb 0x300
	v_add_nc_u32_e32 v0, v220, v6
	s_set_vgpr_msb 3
	v_mov_b32_e32 v6, v44 /*v812*/
	scratch_store_b32 off, v0, off offset:8976 nv
	s_set_vgpr_msb 0x300
	v_add_nc_u32_e32 v0, v220, v7
	s_set_vgpr_msb 15
	scratch_store_b64 off, v[238:239] /*v[1006:1007]*/, off offset:8868 nv
	v_mov_b32_e32 v7, v44 /*v812*/
	s_set_vgpr_msb 0xf00
	scratch_store_b32 off, v0, off offset:8952 nv
	s_wait_xcnt 0x0
	v_add_nc_u32_e32 v0, v220, v3
	s_set_vgpr_msb 11
	scratch_store_b32 off, v249 /*v761*/, off offset:9044 nv
	v_mov_b32_e32 v3, v44 /*v812*/
	s_set_vgpr_msb 0xb03
	scratch_store_b32 off, v0, off offset:8956 nv
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, v44 /*v812*/
	s_clause 0x5
	scratch_store_b128 off, v[0:3], off offset:3488 nv
	scratch_store_b128 off, v[4:7], off offset:3504 nv
	scratch_store_b128 off, v[0:3], off offset:3520 nv
	scratch_store_b128 off, v[4:7], off offset:3536 nv
	scratch_store_b128 off, v[0:3], off offset:3552 nv
	scratch_store_b128 off, v[4:7], off offset:3568 nv
	s_set_vgpr_msb 0x300
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v8
	s_clause 0xb
	scratch_store_b128 off, v[0:3], off offset:3424 nv
	scratch_store_b128 off, v[4:7], off offset:3440 nv
	scratch_store_b128 off, v[0:3], off offset:2912 nv
	scratch_store_b128 off, v[4:7], off offset:2928 nv
	scratch_store_b64 off, v[8:9], off offset:6136 nv
	scratch_load_b64 v[8:9], off, off offset:6128 nv
	scratch_store_b128 off, v[0:3], off offset:3456 nv
	scratch_store_b128 off, v[4:7], off offset:3472 nv
	scratch_store_b128 off, v[0:3], off offset:2944 nv
	scratch_store_b128 off, v[4:7], off offset:2960 nv
	scratch_store_b128 off, v[0:3], off offset:2816 nv
	scratch_store_b128 off, v[4:7], off offset:2832 nv
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v8
	s_clause 0x1
	scratch_store_b64 off, v[8:9], off offset:6128 nv
	scratch_load_b64 v[8:9], off, off offset:6144 nv
	s_set_vgpr_msb 0xc0
	v_mov_b64_e32 v[66:67] /*v[834:835]*/, v[6:7]
	v_mov_b64_e32 v[64:65] /*v[832:833]*/, v[4:5]
	v_mov_b64_e32 v[62:63] /*v[830:831]*/, v[2:3]
	v_mov_b64_e32 v[60:61] /*v[828:829]*/, v[0:1]
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:2976 nv
	scratch_store_b128 off, v[4:7], off offset:2992 nv
	s_set_vgpr_msb 0xc00c
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v8
	s_clause 0x2
	scratch_store_b64 off, v[212:213] /*v[980:981]*/, off offset:6196 nv
	s_set_vgpr_msb 0xcc7
	scratch_store_b64 off, v[160:161] /*v[416:417]*/, off offset:8916 nv
	s_wait_xcnt 0x1
	v_mov_b64_e32 v[212:213] /*v[980:981]*/, v[104:105] /*v[872:873]*/
	s_set_vgpr_msb 0xc700
	s_clause 0x1
	scratch_store_b64 off, v[8:9], off offset:6144 nv
	scratch_load_b64 v[8:9], off, off offset:6152 nv
	s_set_vgpr_msb 0xc0
	v_mov_b64_e32 v[128:129] /*v[896:897]*/, v[6:7]
	v_mov_b64_e32 v[126:127] /*v[894:895]*/, v[4:5]
	v_mov_b64_e32 v[124:125] /*v[892:893]*/, v[2:3]
	v_mov_b64_e32 v[122:123] /*v[890:891]*/, v[0:1]
	s_clause 0x37
	scratch_store_b128 off, v[0:3], off offset:2752 nv
	scratch_store_b128 off, v[4:7], off offset:2768 nv
	scratch_store_b128 off, v[0:3], off offset:992 nv
	scratch_store_b128 off, v[4:7], off offset:1008 nv
	scratch_store_b128 off, v[0:3], off offset:2720 nv
	scratch_store_b128 off, v[4:7], off offset:2736 nv
	scratch_store_b128 off, v[0:3], off offset:2688 nv
	scratch_store_b128 off, v[4:7], off offset:2704 nv
	scratch_store_b128 off, v[0:3], off offset:2656 nv
	scratch_store_b128 off, v[4:7], off offset:2672 nv
	scratch_store_b128 off, v[0:3], off offset:2624 nv
	scratch_store_b128 off, v[4:7], off offset:2640 nv
	scratch_store_b128 off, v[0:3], off offset:2592 nv
	scratch_store_b128 off, v[4:7], off offset:2608 nv
	scratch_store_b128 off, v[0:3], off offset:2560 nv
	scratch_store_b128 off, v[4:7], off offset:2576 nv
	scratch_store_b128 off, v[0:3], off offset:2528 nv
	scratch_store_b128 off, v[4:7], off offset:2544 nv
	scratch_store_b128 off, v[0:3], off offset:2496 nv
	scratch_store_b128 off, v[4:7], off offset:2512 nv
	scratch_store_b128 off, v[0:3], off offset:2464 nv
	scratch_store_b128 off, v[4:7], off offset:2480 nv
	scratch_store_b128 off, v[0:3], off offset:2432 nv
	scratch_store_b128 off, v[4:7], off offset:2448 nv
	scratch_store_b128 off, v[0:3], off offset:2368 nv
	scratch_store_b128 off, v[4:7], off offset:2384 nv
	scratch_store_b128 off, v[0:3], off offset:2336 nv
	scratch_store_b128 off, v[4:7], off offset:2352 nv
	scratch_store_b128 off, v[0:3], off offset:2304 nv
	scratch_store_b128 off, v[4:7], off offset:2320 nv
	scratch_store_b128 off, v[0:3], off offset:2272 nv
	scratch_store_b128 off, v[4:7], off offset:2288 nv
	scratch_store_b128 off, v[0:3], off offset:448 nv
	scratch_store_b128 off, v[4:7], off offset:464 nv
	scratch_store_b128 off, v[0:3], off offset:384 nv
	scratch_store_b128 off, v[4:7], off offset:400 nv
	scratch_store_b128 off, v[0:3], off offset:320 nv
	scratch_store_b128 off, v[4:7], off offset:336 nv
	scratch_store_b128 off, v[0:3], off offset:256 nv
	scratch_store_b128 off, v[4:7], off offset:272 nv
	scratch_store_b128 off, v[0:3], off offset:224 nv
	scratch_store_b128 off, v[4:7], off offset:240 nv
	scratch_store_b128 off, v[0:3], off offset:192 nv
	scratch_store_b128 off, v[4:7], off offset:208 nv
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
	s_set_vgpr_msb 0xc000
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v8
	s_clause 0x3e
	scratch_store_b128 off, v[0:3], off offset:3008 nv
	scratch_store_b128 off, v[4:7], off offset:3024 nv
	scratch_store_b64 off, v[8:9], off offset:6152 nv
	scratch_load_b64 v[8:9], off, off offset:8820 nv
	scratch_store_b128 off, v[82:85], off offset:2208 nv
	scratch_store_b128 off, v[86:89], off offset:2224 nv
	scratch_store_b128 off, v[82:85], off offset:2112 nv
	scratch_store_b128 off, v[86:89], off offset:2128 nv
	scratch_store_b128 off, v[82:85], off offset:2240 nv
	scratch_store_b128 off, v[86:89], off offset:2256 nv
	scratch_store_b128 off, v[82:85], off offset:2176 nv
	scratch_store_b128 off, v[86:89], off offset:2192 nv
	scratch_store_b128 off, v[82:85], off offset:2048 nv
	scratch_store_b128 off, v[86:89], off offset:2064 nv
	scratch_store_b128 off, v[0:3], off offset:2880 nv
	scratch_store_b128 off, v[4:7], off offset:2896 nv
	scratch_store_b128 off, v[82:85], off offset:480 nv
	scratch_store_b128 off, v[86:89], off offset:496 nv
	scratch_store_b128 off, v[82:85], off offset:1888 nv
	scratch_store_b128 off, v[86:89], off offset:1904 nv
	scratch_store_b128 off, v[82:85], off offset:1856 nv
	scratch_store_b128 off, v[86:89], off offset:1872 nv
	scratch_store_b128 off, v[82:85], off offset:1824 nv
	scratch_store_b128 off, v[86:89], off offset:1840 nv
	scratch_store_b128 off, v[82:85], off offset:2144 nv
	scratch_store_b128 off, v[86:89], off offset:2160 nv
	scratch_store_b128 off, v[82:85], off offset:2400 nv
	scratch_store_b128 off, v[86:89], off offset:2416 nv
	scratch_store_b128 off, v[82:85], off offset:1984 nv
	scratch_store_b128 off, v[86:89], off offset:2000 nv
	scratch_store_b128 off, v[82:85], off offset:2784 nv
	scratch_store_b128 off, v[86:89], off offset:2800 nv
	scratch_store_b128 off, v[82:85], off offset:2080 nv
	scratch_store_b128 off, v[86:89], off offset:2096 nv
	scratch_store_b128 off, v[82:85], off offset:352 nv
	scratch_store_b128 off, v[86:89], off offset:368 nv
	scratch_store_b128 off, v[82:85], off offset:288 nv
	scratch_store_b128 off, v[86:89], off offset:304 nv
	scratch_store_b128 off, v[82:85], off offset:2016 nv
	scratch_store_b128 off, v[86:89], off offset:2032 nv
	scratch_store_b128 off, v[82:85], off offset:1792 nv
	scratch_store_b128 off, v[86:89], off offset:1808 nv
	scratch_store_b128 off, v[82:85], off offset:1504 nv
	scratch_store_b128 off, v[86:89], off offset:1520 nv
	scratch_store_b128 off, v[82:85], off offset:1952 nv
	scratch_store_b128 off, v[86:89], off offset:1968 nv
	scratch_store_b128 off, v[0:3], off offset:2848 nv
	scratch_store_b128 off, v[4:7], off offset:2864 nv
	scratch_store_b128 off, v[82:85], off offset:1664 nv
	scratch_store_b128 off, v[86:89], off offset:1680 nv
	scratch_store_b128 off, v[82:85], off offset:1600 nv
	scratch_store_b128 off, v[86:89], off offset:1616 nv
	scratch_store_b128 off, v[82:85], off offset:1760 nv
	scratch_store_b128 off, v[86:89], off offset:1776 nv
	scratch_store_b128 off, v[82:85], off offset:1920 nv
	scratch_store_b128 off, v[86:89], off offset:1936 nv
	scratch_store_b128 off, v[82:85], off offset:1696 nv
	scratch_store_b128 off, v[86:89], off offset:1712 nv
	scratch_store_b128 off, v[82:85], off offset:1248 nv
	scratch_store_b128 off, v[86:89], off offset:1264 nv
	scratch_store_b128 off, v[82:85], off offset:1216 nv
	scratch_store_b128 off, v[86:89], off offset:1232 nv
	scratch_store_b128 off, v[82:85], off offset:1568 nv
	s_clause 0x3e
	scratch_store_b128 off, v[86:89], off offset:1584 nv
	scratch_store_b128 off, v[82:85], off offset:1728 nv
	scratch_store_b128 off, v[86:89], off offset:1744 nv
	scratch_store_b128 off, v[82:85], off offset:416 nv
	scratch_store_b128 off, v[86:89], off offset:432 nv
	scratch_store_b128 off, v[82:85], off offset:1472 nv
	scratch_store_b128 off, v[86:89], off offset:1488 nv
	scratch_store_b128 off, v[82:85], off offset:1440 nv
	scratch_store_b128 off, v[86:89], off offset:1456 nv
	scratch_store_b128 off, v[82:85], off offset:1344 nv
	scratch_store_b128 off, v[86:89], off offset:1360 nv
	scratch_store_b128 off, v[82:85], off offset:1312 nv
	scratch_store_b128 off, v[86:89], off offset:1328 nv
	scratch_store_b128 off, v[82:85], off offset:1536 nv
	scratch_store_b128 off, v[86:89], off offset:1552 nv
	scratch_store_b128 off, v[82:85], off offset:1376 nv
	scratch_store_b128 off, v[86:89], off offset:1392 nv
	scratch_store_b128 off, v[82:85], off offset:1408 nv
	scratch_store_b128 off, v[86:89], off offset:1424 nv
	scratch_store_b128 off, v[82:85], off offset:1280 nv
	scratch_store_b128 off, v[86:89], off offset:1296 nv
	scratch_store_b128 off, v[82:85], off offset:1184 nv
	scratch_store_b128 off, v[86:89], off offset:1200 nv
	scratch_store_b128 off, v[82:85], off offset:1152 nv
	scratch_store_b128 off, v[86:89], off offset:1168 nv
	scratch_store_b128 off, v[82:85], off offset:1120 nv
	scratch_store_b128 off, v[86:89], off offset:1136 nv
	scratch_store_b128 off, v[82:85], off offset:1024 nv
	scratch_store_b128 off, v[86:89], off offset:1040 nv
	scratch_store_b128 off, v[82:85], off offset:576 nv
	scratch_store_b128 off, v[86:89], off offset:592 nv
	scratch_store_b128 off, v[82:85], off offset:1088 nv
	scratch_store_b128 off, v[86:89], off offset:1104 nv
	scratch_store_b128 off, v[82:85], off offset:1056 nv
	scratch_store_b128 off, v[86:89], off offset:1072 nv
	scratch_store_b128 off, v[82:85], off offset:960 nv
	scratch_store_b128 off, v[86:89], off offset:976 nv
	scratch_store_b128 off, v[82:85], off offset:928 nv
	scratch_store_b128 off, v[86:89], off offset:944 nv
	scratch_store_b128 off, v[82:85], off offset:896 nv
	scratch_store_b128 off, v[86:89], off offset:912 nv
	scratch_store_b128 off, v[82:85], off offset:864 nv
	scratch_store_b128 off, v[86:89], off offset:880 nv
	scratch_store_b128 off, v[82:85], off offset:832 nv
	scratch_store_b128 off, v[86:89], off offset:848 nv
	scratch_store_b128 off, v[82:85], off offset:800 nv
	scratch_store_b128 off, v[86:89], off offset:816 nv
	scratch_store_b128 off, v[82:85], off offset:768 nv
	scratch_store_b128 off, v[86:89], off offset:784 nv
	scratch_store_b128 off, v[82:85], off offset:736 nv
	scratch_store_b128 off, v[86:89], off offset:752 nv
	scratch_store_b128 off, v[82:85], off offset:704 nv
	scratch_store_b128 off, v[86:89], off offset:720 nv
	scratch_store_b128 off, v[82:85], off offset:672 nv
	scratch_store_b128 off, v[86:89], off offset:688 nv
	scratch_store_b128 off, v[82:85], off offset:640 nv
	scratch_store_b128 off, v[86:89], off offset:656 nv
	scratch_store_b128 off, v[82:85], off offset:608 nv
	scratch_store_b128 off, v[86:89], off offset:624 nv
	scratch_store_b128 off, v[82:85], off offset:544 nv
	scratch_store_b128 off, v[86:89], off offset:560 nv
	scratch_store_b128 off, v[82:85], off offset:512 nv
	scratch_store_b128 off, v[86:89], off offset:528 nv
	s_clause 0x1
	scratch_store_b128 off, v[82:85], off offset:1632 nv
	scratch_store_b128 off, v[86:89], off offset:1648 nv
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v8
	s_set_vgpr_msb 12
	s_clause 0x4
	scratch_store_b64 off, v[216:217] /*v[984:985]*/, off offset:6172 nv
	s_set_vgpr_msb 0xc00
	scratch_store_b64 off, v[218:219], off offset:8884 nv
	scratch_store_b64 off, v[8:9], off offset:8820 nv
	scratch_load_b64 v[8:9], off, off offset:8828 nv
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v9, v8
	s_clause 0xd
	scratch_store_b64 off, v[8:9], off offset:8828 nv
	scratch_store_b32 off, v230, off offset:8844 nv
	scratch_load_b64 v[194:195], off, off offset:6128 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b128 v[242:245] /*v[1010:1013]*/, off, off offset:7664 nv
	scratch_load_b128 v[246:249] /*v[1014:1017]*/, off, off offset:7680 nv
	scratch_load_b128 v[114:117] /*v[882:885]*/, off, off offset:7952 nv
	scratch_load_b128 v[118:121] /*v[886:889]*/, off, off offset:7968 nv
	scratch_load_b128 v[130:133] /*v[898:901]*/, off, off offset:7504 nv
	scratch_load_b128 v[134:137] /*v[902:905]*/, off, off offset:7520 nv
	scratch_load_b64 v[78:79] /*v[846:847]*/, off, off offset:8820 nv
	scratch_load_b64 v[76:77] /*v[844:845]*/, off, off offset:5536 nv
	scratch_load_b64 v[80:81] /*v[848:849]*/, off, off offset:8828 nv
	scratch_load_b64 v[216:217] /*v[984:985]*/, off, off offset:5544 nv
	s_set_vgpr_msb 0xc000
.LBB0_2:
	s_clause 0x1
	scratch_load_b32 v0, off, off offset:6160 nv
	scratch_load_b32 v2, off, off offset:8304 nv
	s_cmp_lt_i32 s10, s2
	v_mov_b64_e32 v[160:161], v[16:17]
	s_cselect_b32 s3, s10, s20
	v_mov_b64_e32 v[158:159], v[14:15]
	s_lshl_b32 s3, s3, 5
	v_mov_b64_e32 v[156:157], v[12:13]
	v_mov_b64_e32 v[154:155], v[10:11]
	s_clause 0x3e
	scratch_load_b64 v[12:13], off, off offset:6152 nv
	scratch_load_b64 v[14:15], off, off offset:6136 nv
	s_set_vgpr_msb 12
	scratch_store_b128 off, v[44:47] /*v[812:815]*/, off offset:5504 nv
	scratch_store_b128 off, v[48:51] /*v[816:819]*/, off offset:5520 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[146:149] /*v[402:405]*/, off offset:5248 nv
	scratch_store_b128 off, v[150:153] /*v[406:409]*/, off offset:5264 nv
	scratch_store_b128 off, v[162:165] /*v[418:421]*/, off offset:5280 nv
	scratch_store_b128 off, v[166:169] /*v[422:425]*/, off offset:5296 nv
	s_set_vgpr_msb 0x40c
	scratch_store_b128 off, v[60:63] /*v[828:831]*/, off offset:5408 nv
	scratch_store_b128 off, v[64:67] /*v[832:835]*/, off offset:5424 nv
	s_set_vgpr_msb 0xc08
	scratch_store_b128 off, v[44:47] /*v[556:559]*/, off offset:5312 nv
	scratch_store_b128 off, v[48:51] /*v[560:563]*/, off offset:5328 nv
	s_set_vgpr_msb 0x80c
	scratch_store_b128 off, v[122:125] /*v[890:893]*/, off offset:5376 nv
	scratch_store_b128 off, v[126:129] /*v[894:897]*/, off offset:5392 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[194:197] /*v[450:453]*/, off offset:5344 nv
	scratch_store_b128 off, v[198:201] /*v[454:457]*/, off offset:5360 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[10:13] /*v[522:525]*/, off offset:5472 nv
	scratch_store_b128 off, v[14:17] /*v[526:529]*/, off offset:5488 nv
	scratch_store_b128 off, v[170:173] /*v[682:685]*/, off offset:5216 nv
	scratch_store_b128 off, v[174:177] /*v[686:689]*/, off offset:5232 nv
	s_set_vgpr_msb 0x804
	scratch_store_b128 off, v[234:237] /*v[490:493]*/, off offset:5440 nv
	scratch_store_b128 off, v[238:241] /*v[494:497]*/, off offset:5456 nv
	s_set_vgpr_msb 0x40c
	scratch_store_b128 off, v[204:207] /*v[972:975]*/, off offset:5184 nv
	scratch_store_b128 off, v[208:211] /*v[976:979]*/, off offset:5200 nv
	s_set_vgpr_msb 0xc08
	scratch_store_b128 off, v[20:23] /*v[532:535]*/, off offset:5152 nv
	scratch_store_b128 off, v[24:27] /*v[536:539]*/, off offset:5168 nv
	scratch_store_b128 off, v[76:79] /*v[588:591]*/, off offset:5120 nv
	scratch_store_b128 off, v[80:83] /*v[592:595]*/, off offset:5136 nv
	s_set_vgpr_msb 0x80c
	scratch_store_b128 off, v[196:199] /*v[964:967]*/, off offset:5088 nv
	scratch_store_b128 off, v[200:203] /*v[968:971]*/, off offset:5104 nv
	s_set_vgpr_msb 0xc08
	scratch_store_b128 off, v[52:55] /*v[564:567]*/, off offset:5056 nv
	scratch_store_b128 off, v[56:59] /*v[568:571]*/, off offset:5072 nv
	s_set_vgpr_msb 0x804
	scratch_store_b128 off, v[186:189] /*v[442:445]*/, off offset:5024 nv
	scratch_store_b128 off, v[190:193] /*v[446:449]*/, off offset:5040 nv
	scratch_store_b128 off, v[138:141] /*v[394:397]*/, off offset:4992 nv
	scratch_store_b128 off, v[142:145] /*v[398:401]*/, off offset:5008 nv
	scratch_store_b128 off, v[178:181] /*v[434:437]*/, off offset:4960 nv
	scratch_store_b128 off, v[182:185] /*v[438:441]*/, off offset:4976 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[154:157] /*v[666:669]*/, off offset:4928 nv
	scratch_store_b128 off, v[158:161] /*v[670:673]*/, off offset:4944 nv
	s_set_vgpr_msb 0x804
	scratch_store_b128 off, v[114:117] /*v[370:373]*/, off offset:4896 nv
	scratch_store_b128 off, v[118:121] /*v[374:377]*/, off offset:4912 nv
	scratch_store_b128 off, v[130:133] /*v[386:389]*/, off offset:4864 nv
	scratch_store_b128 off, v[134:137] /*v[390:393]*/, off offset:4880 nv
	scratch_store_b128 off, v[106:109] /*v[362:365]*/, off offset:4832 nv
	scratch_store_b128 off, v[110:113] /*v[366:369]*/, off offset:4848 nv
	scratch_store_b128 off, v[122:125] /*v[378:381]*/, off offset:4800 nv
	scratch_store_b128 off, v[126:129] /*v[382:385]*/, off offset:4816 nv
	scratch_store_b128 off, v[70:73] /*v[326:329]*/, off offset:4768 nv
	scratch_store_b128 off, v[74:77] /*v[330:333]*/, off offset:4784 nv
	scratch_store_b128 off, v[62:65] /*v[318:321]*/, off offset:4736 nv
	scratch_store_b128 off, v[66:69] /*v[322:325]*/, off offset:4752 nv
	scratch_store_b128 off, v[86:89] /*v[342:345]*/, off offset:4704 nv
	scratch_store_b128 off, v[90:93] /*v[346:349]*/, off offset:4720 nv
	scratch_store_b128 off, v[78:81] /*v[334:337]*/, off offset:4672 nv
	scratch_store_b128 off, v[82:85] /*v[338:341]*/, off offset:4688 nv
	s_set_vgpr_msb 0x488
	scratch_store_b128 off, v[108:111] /*v[620:623]*/, off offset:4640 nv
	scratch_store_b128 off, v[112:115] /*v[624:627]*/, off offset:4656 nv
	scratch_load_b128 v[106:109] /*v[618:621]*/, off, off offset:5616 nv
	scratch_load_b128 v[110:113] /*v[622:625]*/, off, off offset:5632 nv
	scratch_load_b128 v[122:125] /*v[634:637]*/, off, off offset:7184 nv
	scratch_load_b128 v[126:129] /*v[638:641]*/, off, off offset:7200 nv
	scratch_load_b128 v[90:93] /*v[602:605]*/, off, off offset:7344 nv
	s_clause 0x3e
	scratch_load_b128 v[94:97] /*v[606:609]*/, off, off offset:7360 nv
	scratch_load_b128 v[98:101] /*v[610:613]*/, off, off offset:5680 nv
	scratch_load_b128 v[102:105] /*v[614:617]*/, off, off offset:5696 nv
	s_set_vgpr_msb 0x88c0
	scratch_load_b128 v[180:183] /*v[948:951]*/, off, off offset:5744 nv
	scratch_load_b128 v[184:187] /*v[952:955]*/, off, off offset:5760 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[226:229] /*v[482:485]*/, off, off offset:5808 nv
	scratch_load_b128 v[230:233] /*v[486:489]*/, off, off offset:5824 nv
	scratch_load_b128 v[62:65] /*v[318:321]*/, off, off offset:5872 nv
	scratch_load_b128 v[66:69] /*v[322:325]*/, off, off offset:5888 nv
	scratch_load_b128 v[30:33] /*v[286:289]*/, off, off offset:6000 nv
	scratch_load_b128 v[34:37] /*v[290:293]*/, off, off offset:6016 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[114:117] /*v[626:629]*/, off, off offset:7088 nv
	scratch_load_b128 v[118:121] /*v[630:633]*/, off, off offset:7104 nv
	scratch_load_b128 v[82:85] /*v[594:597]*/, off, off offset:7440 nv
	scratch_load_b128 v[86:89] /*v[598:601]*/, off, off offset:7456 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[90:93] /*v[858:861]*/, off, off offset:7760 nv
	scratch_load_b128 v[94:97] /*v[862:865]*/, off, off offset:7776 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[218:221] /*v[474:477]*/, off, off offset:5776 nv
	scratch_load_b128 v[222:225] /*v[478:481]*/, off, off offset:5792 nv
	scratch_load_b128 v[202:205] /*v[458:461]*/, off, off offset:5840 nv
	scratch_load_b128 v[206:209] /*v[462:465]*/, off, off offset:5856 nv
	scratch_load_b128 v[54:57] /*v[310:313]*/, off, off offset:5936 nv
	scratch_load_b128 v[58:61] /*v[314:317]*/, off, off offset:5952 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[238:241], off, off offset:6064 nv
	scratch_load_b128 v[242:245], off, off offset:6080 nv
	s_set_vgpr_msb 0x80
	scratch_load_b128 v[154:157] /*v[666:669]*/, off, off offset:6352 nv
	scratch_load_b128 v[158:161] /*v[670:673]*/, off, off offset:6368 nv
	scratch_load_b128 v[202:205] /*v[714:717]*/, off, off offset:6416 nv
	scratch_load_b128 v[206:209] /*v[718:721]*/, off, off offset:6432 nv
	s_set_vgpr_msb 0x8040
	scratch_load_b128 v[234:237] /*v[490:493]*/, off, off offset:6252 nv
	scratch_load_b128 v[238:241] /*v[494:497]*/, off, off offset:6268 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[138:141] /*v[906:909]*/, off, off offset:6640 nv
	scratch_load_b128 v[142:145] /*v[910:913]*/, off, off offset:6656 nv
	scratch_load_b128 v[188:191] /*v[956:959]*/, off, off offset:6768 nv
	scratch_load_b128 v[192:195] /*v[960:963]*/, off, off offset:6784 nv
	scratch_load_b128 v[226:229] /*v[994:997]*/, off, off offset:6896 nv
	scratch_load_b128 v[230:233] /*v[998:1001]*/, off, off offset:6912 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[242:245] /*v[498:501]*/, off, off offset:7024 nv
	scratch_load_b128 v[246:249] /*v[502:505]*/, off, off offset:7040 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[242:245] /*v[754:757]*/, off, off offset:7216 nv
	scratch_load_b128 v[246:249] /*v[758:761]*/, off, off offset:7232 nv
	s_set_vgpr_msb 0x8000
	scratch_load_b128 v[254:257], off, off offset:7376 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[2:5] /*v[258:261]*/, off, off offset:7392 nv
	scratch_load_b128 v[46:49] /*v[302:305]*/, off, off offset:7536 nv
	scratch_load_b128 v[50:53] /*v[306:309]*/, off, off offset:7552 nv
	scratch_load_b128 v[250:253] /*v[506:509]*/, off, off offset:7696 nv
	scratch_load_b128 v[254:257] /*v[510:513]*/, off, off offset:7712 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[18:21] /*v[786:789]*/, off, off offset:7888 nv
	scratch_load_b128 v[22:25] /*v[790:793]*/, off, off offset:7904 nv
	scratch_load_b128 v[10:13] /*v[778:781]*/, off, off offset:8048 nv
	scratch_load_b128 v[14:17] /*v[782:785]*/, off, off offset:8064 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[162:165], off, off offset:8240 nv
	scratch_load_b128 v[166:169], off, off offset:8256 nv
	s_set_vgpr_msb 0x8c
	s_wait_loadcnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[74:81], v[242:249] /*v[1010:1017]*/, 0
	s_set_vgpr_msb 0x8cc0
	s_clause 0xf
	scratch_load_b128 v[106:109] /*v[874:877]*/, off, off offset:8016 nv
	scratch_load_b128 v[110:113] /*v[878:881]*/, off, off offset:8032 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[178:181] /*v[690:693]*/, off, off offset:6384 nv
	scratch_load_b128 v[182:185] /*v[694:697]*/, off, off offset:6400 nv
	scratch_load_b128 v[234:237] /*v[746:749]*/, off, off offset:6448 nv
	scratch_load_b128 v[238:241] /*v[750:753]*/, off, off offset:6464 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[60:63] /*v[828:831]*/, off, off offset:7824 nv
	scratch_load_b128 v[64:67] /*v[832:835]*/, off, off offset:7840 nv
	scratch_load_b128 v[172:175] /*v[940:943]*/, off, off offset:6704 nv
	scratch_load_b128 v[176:179] /*v[944:947]*/, off, off offset:6720 nv
	scratch_load_b128 v[204:207] /*v[972:975]*/, off, off offset:6832 nv
	scratch_load_b128 v[208:211] /*v[976:979]*/, off, off offset:6848 nv
	scratch_load_b128 v[36:39] /*v[804:807]*/, off, off offset:6960 nv
	scratch_load_b128 v[40:43] /*v[808:811]*/, off, off offset:6976 nv
	s_set_vgpr_msb 0xc004
	v_wmma_f32_16x16x32_bf16 v[90:97], v[66:73], v[14:21] /*v[270:277]*/, 0
	s_set_vgpr_msb 0x4c0
	s_clause 0x3e
	scratch_load_b128 v[98:101] /*v[866:869]*/, off, off offset:7120 nv
	scratch_load_b128 v[102:105] /*v[870:873]*/, off, off offset:7136 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[222:225], off, off offset:7280 nv
	scratch_load_b128 v[226:229], off, off offset:7296 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b128 v[52:55] /*v[820:823]*/, off, off offset:7600 nv
	scratch_load_b128 v[56:59] /*v[824:827]*/, off, off offset:7616 nv
	scratch_load_b128 v[44:47] /*v[812:815]*/, off, off offset:7792 nv
	scratch_load_b128 v[48:51] /*v[816:819]*/, off, off offset:7808 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[38:41] /*v[294:297]*/, off, off offset:8112 nv
	scratch_load_b128 v[42:45] /*v[298:301]*/, off, off offset:8128 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[214:217], off, off offset:8208 nv
	scratch_load_b128 v[218:221], off, off offset:8224 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[70:73] /*v[326:329]*/, off, off offset:6320 nv
	scratch_load_b128 v[74:77] /*v[330:333]*/, off, off offset:6336 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[162:165] /*v[674:677]*/, off, off offset:6512 nv
	scratch_load_b128 v[166:169] /*v[678:681]*/, off, off offset:6528 nv
	scratch_load_b128 v[218:221] /*v[730:733]*/, off, off offset:6576 nv
	scratch_load_b128 v[222:225] /*v[734:737]*/, off, off offset:6592 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[146:149] /*v[914:917]*/, off, off offset:6284 nv
	scratch_load_b128 v[150:153] /*v[918:921]*/, off, off offset:6300 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[230:233], off, off offset:6672 nv
	scratch_load_b128 v[234:237], off, off offset:6688 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[6:9] /*v[262:265]*/, off, off offset:6800 nv
	scratch_load_b128 v[10:13] /*v[266:269]*/, off, off offset:6816 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[146:149] /*v[658:661]*/, off, off offset:6928 nv
	scratch_load_b128 v[150:153] /*v[662:665]*/, off, off offset:6944 nv
	scratch_load_b128 v[194:197] /*v[706:709]*/, off, off offset:7056 nv
	scratch_load_b128 v[198:201] /*v[710:713]*/, off, off offset:7072 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[26:29] /*v[794:797]*/, off, off offset:7248 nv
	scratch_load_b128 v[30:33] /*v[798:801]*/, off, off offset:7264 nv
	scratch_load_b128 v[82:85] /*v[850:853]*/, off, off offset:7408 nv
	scratch_load_b128 v[86:89] /*v[854:857]*/, off, off offset:7424 nv
	scratch_load_b128 v[162:165] /*v[930:933]*/, off, off offset:7568 nv
	scratch_load_b128 v[166:169] /*v[934:937]*/, off, off offset:7584 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[226:229] /*v[738:741]*/, off, off offset:7728 nv
	scratch_load_b128 v[230:233] /*v[742:745]*/, off, off offset:7744 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[218:221] /*v[986:989]*/, off, off offset:7920 nv
	scratch_load_b128 v[222:225] /*v[990:993]*/, off, off offset:7936 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[250:253] /*v[762:765]*/, off, off offset:8080 nv
	scratch_load_b128 v[254:257] /*v[766:769]*/, off, off offset:8096 nv
	scratch_load_b128 v[138:141] /*v[650:653]*/, off, off offset:6480 nv
	scratch_load_b128 v[142:145] /*v[654:657]*/, off, off offset:6496 nv
	scratch_load_b128 v[186:189] /*v[698:701]*/, off, off offset:6544 nv
	scratch_load_b128 v[190:193] /*v[702:705]*/, off, off offset:6560 nv
	scratch_load_b128 v[130:133] /*v[642:645]*/, off, off offset:6220 nv
	scratch_load_b128 v[134:137] /*v[646:649]*/, off, off offset:6236 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[122:125] /*v[890:893]*/, off, off offset:6608 nv
	scratch_load_b128 v[126:129] /*v[894:897]*/, off, off offset:6624 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[210:213] /*v[466:469]*/, off, off offset:6736 nv
	scratch_load_b128 v[214:217] /*v[470:473]*/, off, off offset:6752 nv
	scratch_load_b128 v[22:25] /*v[278:281]*/, off, off offset:6864 nv
	scratch_load_b128 v[26:29] /*v[282:285]*/, off, off offset:6880 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[170:173] /*v[682:685]*/, off, off offset:6992 nv
	scratch_load_b128 v[174:177] /*v[686:689]*/, off, off offset:7008 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[2:5] /*v[770:773]*/, off, off offset:7152 nv
	scratch_load_b128 v[6:9] /*v[774:777]*/, off, off offset:7168 nv
	scratch_load_b128 v[68:71] /*v[836:839]*/, off, off offset:7312 nv
	scratch_load_b128 v[72:75] /*v[840:843]*/, off, off offset:7328 nv
	scratch_load_b128 v[196:199] /*v[964:967]*/, off, off offset:7632 nv
	scratch_load_b128 v[200:203] /*v[968:971]*/, off, off offset:7648 nv
	scratch_load_b128 v[154:157] /*v[922:925]*/, off, off offset:7856 nv
	scratch_load_b128 v[158:161] /*v[926:929]*/, off, off offset:7872 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[210:213] /*v[722:725]*/, off, off offset:7984 nv
	s_clause 0x3
	scratch_load_b128 v[214:217] /*v[726:729]*/, off, off offset:8000 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[234:237] /*v[1002:1005]*/, off, off offset:8144 nv
	scratch_load_b128 v[238:241] /*v[1006:1009]*/, off, off offset:8160 nv
	s_set_vgpr_msb 0xc048
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, s[4:5]
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:5552 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:5568 nv
	s_add_nc_u64 s[8:9], s[8:9], -1
	s_add_co_i32 s10, s10, 1
	s_cmp_lg_u64 s[8:9], 0
	s_wait_loadcnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[74:81], v[122:129] /*v[634:641]*/, 0
	s_set_vgpr_msb 0x4800
	v_or_b32_e32 v1, s3, v0
	s_clause 0x1
	scratch_load_b32 v0, off, off offset:8924 nv
	scratch_load_b32 v17, off, off offset:6316 nv
	v_mul_lo_u32 v1, v1, s48
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v1, v1, v2, 4
	s_set_vgpr_msb 0x48
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[74:81], v[106:113] /*v[618:625]*/, 0
	s_set_vgpr_msb 0x4800
	buffer_load_b128 v[4:7], v1, s[12:15], null offen
	v_or_b32_e32 v82, 32, v1
	v_or_b32_e32 v83, 0x60, v1
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8308 nv
	buffer_load_b128 v[4:7], v82, s[12:15], null offen
	s_set_vgpr_msb 0x48
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[74:81], v[90:97] /*v[602:609]*/, 0
	s_set_vgpr_msb 0x4800
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8324 nv
	buffer_load_b128 v[4:7], v1, s[16:19], null offen
	s_set_vgpr_msb 0x48
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[74:81], v[98:105] /*v[610:617]*/, 0
	s_set_vgpr_msb 0x4800
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8340 nv
	buffer_load_b128 v[4:7], v82, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 64, v1
	s_set_vgpr_msb 0x8c
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[74:81], v[180:187] /*v[948:955]*/, 0
	s_set_vgpr_msb 0x8c00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8356 nv
	buffer_load_b128 v[4:7], v82, s[12:15], null offen
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[74:81], v[226:233] /*v[482:489]*/, 0
	s_set_vgpr_msb 0x8400
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8372 nv
	buffer_load_b128 v[4:7], v83, s[12:15], null offen
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[74:81], v[62:69] /*v[318:325]*/, 0
	s_set_vgpr_msb 0x8400
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8388 nv
	buffer_load_b128 v[4:7], v82, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 0x80, v1
	s_set_vgpr_msb 0x48
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[66:73], v[114:121] /*v[626:633]*/, 0
	s_set_vgpr_msb 0x4800
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8404 nv
	buffer_load_b128 v[4:7], v83, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v83, 0xa0, v1
	s_set_vgpr_msb 0x48
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[66:73], v[82:89] /*v[594:601]*/, 0
	s_set_vgpr_msb 0x4800
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8420 nv
	buffer_load_b128 v[4:7], v82, s[12:15], null offen
	s_set_vgpr_msb 0x8c
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[66:73], v[90:97] /*v[858:865]*/, 0
	s_set_vgpr_msb 0x8c00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8436 nv
	buffer_load_b128 v[4:7], v83, s[12:15], null offen
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[66:73], v[218:225] /*v[474:481]*/, 0
	s_set_vgpr_msb 0x8400
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8452 nv
	buffer_load_b128 v[4:7], v82, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 0xc0, v1
	v_or_b32_e32 v1, 0xe0, v1
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[66:73], v[202:209] /*v[458:465]*/, 0
	s_set_vgpr_msb 0x8400
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8468 nv
	buffer_load_b128 v[4:7], v83, s[16:19], null offen
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[66:73], v[54:61] /*v[310:317]*/, 0
	s_set_vgpr_msb 0x8400
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8484 nv
	buffer_load_b128 v[4:7], v82, s[12:15], null offen
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[58:65], v[242:249] /*v[498:505]*/, v[162:169] /*v[418:425]*/
	s_set_vgpr_msb 0x5400
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8500 nv
	buffer_load_b128 v[4:7], v1, s[12:15], null offen
	s_set_vgpr_msb 0x58
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[58:65], v[242:249] /*v[754:761]*/, v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5800
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8516 nv
	buffer_load_b128 v[4:7], v82, s[16:19], null offen
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[58:65], v[254:261], v[194:201] /*v[450:457]*/
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8532 nv
	s_set_vgpr_msb 0x5000
	buffer_load_b128 v[4:7], v1, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v1, s3, v0
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[58:65], v[46:53] /*v[302:309]*/, v[88:95] /*v[344:351]*/
	s_set_vgpr_msb 0x5400
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v1, v1, s48
	v_add_lshl_u32 v1, v1, v2, 4
	s_set_vgpr_msb 0xa4
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[58:65], v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa400
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v82, 32, v1
	v_or_b32_e32 v83, 0x60, v1
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[4:7], off offset:8548 nv
	buffer_load_b128 v[2:5], v1, s[12:15], null offen
	s_set_vgpr_msb 0xac
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[58:65], v[18:25] /*v[786:793]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xac00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8564 nv
	buffer_load_b128 v[2:5], v82, s[12:15], null offen
	s_set_vgpr_msb 0xac
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[58:65], v[10:17] /*v[778:785]*/, v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xac00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8580 nv
	buffer_load_b128 v[2:5], v1, s[16:19], null offen
	s_set_vgpr_msb 12
	v_wmma_f32_16x16x32_bf16 v[90:97], v[50:57], v[106:113] /*v[874:881]*/, v[90:97]
	s_set_vgpr_msb 0xc00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8596 nv
	buffer_load_b128 v[2:5], v82, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 64, v1
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[50:57], v[98:105] /*v[866:873]*/, v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5c00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8612 nv
	buffer_load_b128 v[2:5], v82, s[12:15], null offen
	s_set_vgpr_msb 0xac
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[50:57], v[44:51] /*v[812:819]*/, v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xac00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8628 nv
	buffer_load_b128 v[2:5], v83, s[12:15], null offen
	s_set_vgpr_msb 0xac
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[50:57], v[114:121] /*v[882:889]*/, v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xac00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8644 nv
	buffer_load_b128 v[2:5], v82, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 0x80, v1
	s_set_vgpr_msb 0xa4
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[50:57], v[38:45] /*v[294:301]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa400
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8660 nv
	buffer_load_b128 v[2:5], v83, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v83, 0xa0, v1
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[50:57], v[214:221], v[58:65] /*v[570:577]*/
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8676 nv
	s_set_vgpr_msb 0xa000
	buffer_load_b128 v[2:5], v82, s[12:15], null offen
	s_set_vgpr_msb 0x58
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[42:49], v[194:201] /*v[706:713]*/, v[162:169] /*v[418:425]*/
	s_set_vgpr_msb 0x5800
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8692 nv
	buffer_load_b128 v[2:5], v83, s[12:15], null offen
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[42:49], v[26:33] /*v[794:801]*/, v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5c00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8708 nv
	buffer_load_b128 v[2:5], v82, s[16:19], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 0xc0, v1
	v_or_b32_e32 v1, 0xe0, v1
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[42:49], v[82:89] /*v[850:857]*/, v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5c00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8724 nv
	buffer_load_b128 v[2:5], v83, s[16:19], null offen
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[42:49], v[162:169] /*v[930:937]*/, v[88:95] /*v[344:351]*/
	s_set_vgpr_msb 0x5c00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8740 nv
	buffer_load_b128 v[2:5], v82, s[12:15], null offen
	s_set_vgpr_msb 0xa8
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[42:49], v[226:233] /*v[738:745]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa800
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8756 nv
	buffer_load_b128 v[2:5], v1, s[12:15], null offen
	s_set_vgpr_msb 0xac
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[42:49], v[218:225] /*v[986:993]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xac00
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8772 nv
	buffer_load_b128 v[2:5], v82, s[16:19], null offen
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[82:89], v[74:81], v[68:75] /*v[580:587]*/, 0
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	scratch_store_b128 off, v[2:5], off offset:8788 nv
	buffer_load_b128 v[0:3], v1, s[16:19], null offen
	v_wmma_f32_16x16x32_bf16 v[82:89], v[58:65], v[246:253], v[82:89]
	s_clause 0x1
	scratch_load_b128 v[246:249], off, off offset:8176 nv
	scratch_load_b128 v[250:253], off, off offset:8192 nv
	s_wait_loadcnt 0x2
	s_clause 0x2
	scratch_store_b128 off, v[0:3], off offset:8804 nv
	scratch_load_b128 v[2:5], off, off offset:3264 nv
	scratch_load_b128 v[6:9], off, off offset:3280 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[74:81], v[2:9], 0
	s_clause 0x2
	scratch_load_b128 v[2:5], off, off offset:3296 nv
	scratch_load_b128 v[6:9], off, off offset:3312 nv
	scratch_load_b32 v0, off, off offset:8840 nv
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[106:113], v[66:73], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3328 nv
	scratch_load_b128 v[6:9], off, off offset:3344 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[74:81], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3360 nv
	scratch_load_b128 v[6:9], off, off offset:3376 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129], v[66:73], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3072 nv
	scratch_load_b128 v[6:9], off, off offset:3088 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[130:137], v[74:81], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3392 nv
	scratch_load_b128 v[6:9], off, off offset:3408 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[66:73], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3648 nv
	scratch_load_b128 v[6:9], off, off offset:3664 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[74:81], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3712 nv
	scratch_load_b128 v[6:9], off, off offset:3728 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[66:73], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3776 nv
	scratch_load_b128 v[6:9], off, off offset:3792 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[74:81], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3840 nv
	scratch_load_b128 v[6:9], off, off offset:3856 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[66:73], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3904 nv
	scratch_load_b128 v[6:9], off, off offset:3920 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[74:81], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3968 nv
	scratch_load_b128 v[6:9], off, off offset:3984 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[66:73], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5648 nv
	scratch_load_b128 v[6:9], off, off offset:5664 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[66:73], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5712 nv
	scratch_load_b128 v[6:9], off, off offset:5728 nv
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[74:81], v[30:37] /*v[286:293]*/, 0
	s_set_vgpr_msb 0x8440
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[66:73], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:7472 nv
	scratch_load_b128 v[8:11], off, off offset:7488 nv
	s_set_vgpr_msb 0x80
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[66:73], v[238:245], 0
	s_set_vgpr_msb 0x8000
	scratch_load_b64 v[70:71], off, off offset:8856 nv
	ds_store_b128 v0, v[58:61] offset:64
	ds_store_b128 v0, v[62:65] offset:96
	ds_store_b128 v0, v[42:45] offset:128
	ds_store_b128 v0, v[46:49] offset:160
	ds_store_b128 v0, v[26:29] offset:192
	ds_store_b128 v0, v[30:33] offset:224
	scratch_load_b64 v[72:73], off, off offset:6144 nv
	ds_store_b128 v0, v[74:77]
	ds_store_b128 v0, v[78:81] offset:32
	scratch_load_b64 v[78:79], off, off offset:8884 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[98:105], v[58:65], v[154:161] /*v[666:673]*/, v[98:105]
	s_set_vgpr_msb 0x800
	ds_store_b128 v17, v[198:201]
	ds_store_b128 v17, v[202:205] offset:32
	ds_store_b128 v17, v[138:141] offset:192
	ds_store_b128 v17, v[142:145] offset:224
	ds_store_b128 v17, v[146:149] offset:128
	ds_store_b128 v17, v[150:153] offset:160
	ds_store_b128 v17, v[178:181] offset:64
	ds_store_b128 v17, v[182:185] offset:96
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[114:121], v[58:65], v[202:209] /*v[714:721]*/, v[114:121]
	s_set_vgpr_msb 0x804
	v_wmma_f32_16x16x32_bf16 v[130:137], v[58:65], v[234:241] /*v[490:497]*/, v[130:137]
	s_set_vgpr_msb 0x45c
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[58:65], v[138:145] /*v[906:913]*/, v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[58:65], v[188:195] /*v[956:963]*/, v[130:137] /*v[386:393]*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[58:65], v[226:233] /*v[994:1001]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5ca0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[58:65], v[246:253], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[58:65], v[162:169], v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_load_b128 v[58:61], off, off offset:8272 nv
	scratch_load_b128 v[62:65], off, off offset:8288 nv
	v_wmma_f32_16x16x32_bf16 v[106:113], v[50:57], v[178:185] /*v[690:697]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[122:129], v[50:57], v[234:241] /*v[746:753]*/, v[122:129]
	s_set_vgpr_msb 0x85c
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[50:57], v[60:67] /*v[828:835]*/, v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[50:57], v[172:179] /*v[940:947]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[50:57], v[204:211] /*v[972:979]*/, v[138:145] /*v[394:401]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[50:57], v[36:43] /*v[804:811]*/, v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5c50
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[50:57], v[222:229], v[186:193] /*v[442:449]*/
	s_wait_loadcnt 0x5
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[50:57], v[4:11], v[80:87] /*v[336:343]*/
	s_set_vgpr_msb 0x505c
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[50:57], v[52:59] /*v[820:827]*/, v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5ca0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[50:57], v[58:65], v[74:81] /*v[586:593]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[50:53], off, off offset:5904 nv
	scratch_load_b128 v[54:57], off, off offset:5920 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[42:49], v[50:57], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[50:53], off, off offset:5968 nv
	scratch_load_b128 v[54:57], off, off offset:5984 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[34:41], v[50:57], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa004
	s_clause 0x1
	scratch_load_b128 v[50:53], off, off offset:6032 nv
	scratch_load_b128 v[54:57], off, off offset:6048 nv
	v_wmma_f32_16x16x32_bf16 v[82:89], v[42:49], v[70:77] /*v[326:333]*/, v[82:89]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[98:105], v[42:49], v[162:169] /*v[674:681]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[114:121], v[42:49], v[218:225] /*v[730:737]*/, v[114:121]
	s_set_vgpr_msb 0x80c
	v_wmma_f32_16x16x32_bf16 v[130:137], v[42:49], v[146:153] /*v[914:921]*/, v[130:137]
	s_set_vgpr_msb 0xc50
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[42:49], v[230:237], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5054
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[42:49], v[6:13] /*v[262:269]*/, v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5458
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[42:49], v[146:153] /*v[658:665]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x58a8
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[42:49], v[250:257] /*v[762:769]*/, v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa8a0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[42:49], v[50:57], v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x2
	scratch_load_b128 v[42:45], off, off offset:6096 nv
	scratch_load_b128 v[46:49], off, off offset:6112 nv
	scratch_load_b64 v[56:57], off, off offset:6204 nv
	v_wmma_f32_16x16x32_bf16 v[90:97], v[34:41], v[138:145] /*v[650:657]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[106:113], v[34:41], v[186:193] /*v[698:705]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[122:129], v[34:41], v[130:137] /*v[642:649]*/, v[122:129]
	s_set_vgpr_msb 0x85c
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[34:41], v[122:129] /*v[890:897]*/, v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5c54
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[34:41], v[210:217] /*v[466:473]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[34:41], v[22:29] /*v[278:285]*/, v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5458
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[34:41], v[170:177] /*v[682:689]*/, v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x585c
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[34:41], v[2:9] /*v[770:777]*/, v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[34:41], v[68:75] /*v[836:843]*/, v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[34:41], v[130:137] /*v[898:905]*/, v[80:87] /*v[336:343]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[34:41], v[196:203] /*v[964:971]*/, v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5cac
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[34:41], v[154:161] /*v[922:929]*/, v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xaca8
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[34:41], v[210:217] /*v[722:729]*/, v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa8ac
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[34:41], v[234:241] /*v[1002:1009]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xaca0
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[34:41], v[42:49], v[74:81] /*v[586:593]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x2
	scratch_load_b128 v[34:37], off, off offset:3232 nv
	scratch_load_b128 v[38:41], off, off offset:3248 nv
	scratch_load_b64 v[48:49], off, off offset:6180 nv
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[82:89], v[26:33], v[34:41], v[82:89]
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3136 nv
	scratch_load_b128 v[38:41], off, off offset:3152 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[18:25], v[34:41], v[90:97]
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4480 nv
	scratch_load_b128 v[38:41], off, off offset:4496 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[26:33], v[34:41], v[98:105]
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3168 nv
	scratch_load_b128 v[38:41], off, off offset:3184 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113], v[18:25], v[34:41], v[106:113]
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3200 nv
	scratch_load_b128 v[38:41], off, off offset:3216 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[26:33], v[34:41], v[114:121]
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3040 nv
	scratch_load_b128 v[38:41], off, off offset:3056 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129], v[18:25], v[34:41], v[122:129]
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3104 nv
	scratch_load_b128 v[38:41], off, off offset:3120 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[130:137], v[26:33], v[34:41], v[130:137]
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3616 nv
	scratch_load_b128 v[38:41], off, off offset:3632 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[18:25], v[34:41], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3680 nv
	scratch_load_b128 v[38:41], off, off offset:3696 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[26:33], v[34:41], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3744 nv
	scratch_load_b128 v[38:41], off, off offset:3760 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[18:25], v[34:41], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3808 nv
	scratch_load_b128 v[38:41], off, off offset:3824 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[26:33], v[34:41], v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3872 nv
	scratch_load_b128 v[38:41], off, off offset:3888 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[18:25], v[34:41], v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:3936 nv
	scratch_load_b128 v[38:41], off, off offset:3952 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[26:33], v[34:41], v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4000 nv
	scratch_load_b128 v[38:41], off, off offset:4016 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[18:25], v[34:41], v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4032 nv
	scratch_load_b128 v[38:41], off, off offset:4048 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[26:33], v[34:41], v[162:169] /*v[418:425]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4064 nv
	scratch_load_b128 v[38:41], off, off offset:4080 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[18:25], v[34:41], v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4096 nv
	scratch_load_b128 v[38:41], off, off offset:4112 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[26:33], v[34:41], v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4128 nv
	scratch_load_b128 v[38:41], off, off offset:4144 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[18:25], v[34:41], v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4160 nv
	scratch_load_b128 v[38:41], off, off offset:4176 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[26:33], v[34:41], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4192 nv
	scratch_load_b128 v[38:41], off, off offset:4208 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[18:25], v[34:41], v[80:87] /*v[336:343]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4224 nv
	scratch_load_b128 v[38:41], off, off offset:4240 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[26:33], v[34:41], v[88:95] /*v[344:351]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4256 nv
	scratch_load_b128 v[38:41], off, off offset:4272 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[18:25], v[34:41], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4288 nv
	scratch_load_b128 v[38:41], off, off offset:4304 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[26:33], v[34:41], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4320 nv
	scratch_load_b128 v[38:41], off, off offset:4336 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[18:25], v[34:41], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4352 nv
	scratch_load_b128 v[38:41], off, off offset:4368 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[26:33], v[34:41], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4384 nv
	scratch_load_b128 v[38:41], off, off offset:4400 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[18:25], v[34:41], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4416 nv
	scratch_load_b128 v[38:41], off, off offset:4432 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[26:33], v[34:41], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4448 nv
	scratch_load_b128 v[38:41], off, off offset:4464 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[18:25], v[34:41], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4512 nv
	scratch_load_b128 v[38:41], off, off offset:4528 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[26:33], v[34:41], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4544 nv
	scratch_load_b128 v[38:41], off, off offset:4560 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[18:25], v[34:41], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[34:37], off, off offset:4576 nv
	scratch_load_b128 v[38:41], off, off offset:4592 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[26:33], v[34:41], v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x2
	scratch_load_b128 v[26:29], off, off offset:4608 nv
	scratch_load_b128 v[30:33], off, off offset:4624 nv
	scratch_load_b64 v[40:41], off, off offset:6188 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[34:35], v[122:123], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[18:25], v[26:33], v[74:81] /*v[586:593]*/
	s_set_vgpr_msb 0xa001
	v_pk_add_f32 v[38:39], v[106:107] /*v[362:363]*/, v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	v_nop
	v_nop
	v_nop
	v_pk_mul_f32 v[18:19], v[78:79] /*v[334:335]*/, v[82:83]
	v_pk_mul_f32 v[20:21], v[78:79] /*v[334:335]*/, v[84:85]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_pk_add_f32 v[26:27], v[90:91], v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[86:87]
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[88:89]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[18:19], v[18:19], v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21], v[20:21], v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc01
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[120:121]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[22:23], v[22:23], v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25], v[24:25], v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[18:19], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], s[6:7] op_sel_hi:[1,0]
	v_pk_add_f32 v[32:33], v[32:33], v[254:255] /*v[1022:1023]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v18, v18
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[18:19], v[26:27], v[18:19]
	v_pk_add_f32 v[26:27], v[92:93], v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v32, v32
	s_set_vgpr_msb 1
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[18:19], v[78:79] /*v[334:335]*/, v[18:19]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[20:21], v[26:27], v[20:21]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[26:27], v[78:79] /*v[334:335]*/, v[102:103]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v33, v33
	scratch_load_b64 v[102:103], off, off offset:8876 nv
	v_cvt_pk_bf16_f32 v18, v18, v19
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[20:21], v[78:79] /*v[334:335]*/, v[20:21]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[26:27], v[26:27], v[212:213] /*v[980:981]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v19, v20, v21
	v_pk_add_f32 v[20:21], v[94:95], v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[26:27], s[6:7] op_sel_hi:[1,0]
	scratch_load_b64 v[94:95], off, off offset:8892 nv
	v_pk_mul_f32 v[20:21], v[20:21], v[22:23]
	v_pk_add_f32 v[22:23], v[96:97], v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v28, v26
	v_exp_f32_e32 v29, v27
	v_nop
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[26:27], v[78:79] /*v[334:335]*/, v[104:105]
	v_pk_mul_f32 v[20:21], v[78:79] /*v[334:335]*/, v[20:21]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[100:101]
	s_clause 0x1
	scratch_load_b64 v[96:97], off, off offset:6212 nv
	scratch_load_b64 v[104:105], off, off offset:6196 nv
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v20, v20, v21
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[26:27], v[26:27], v[212:213] /*v[980:981]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25], v[24:25], v[212:213] /*v[980:981]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v21, v22, v23
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[98:99]
	s_set_vgpr_msb 0x10c
	v_pk_mul_f32 v[26:27], v[26:27], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_add_f32 v[22:23], v[22:23], v[212:213] /*v[980:981]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v30, v26
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v31, v27
	v_nop
	s_set_vgpr_msb 0xc00
	v_pk_add_f32 v[26:27], v[106:107], v[48:49] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v24, v24
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[26:27], v[22:23]
	s_set_vgpr_msb 1
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v26, v22, v23
	v_pk_add_f32 v[22:23], v[108:109], v[48:49] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[116:117]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[254:255] /*v[1022:1023]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v27, v22, v23
	v_pk_add_f32 v[22:23], v[110:111], v[48:49] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	scratch_load_b64 v[110:111], off, off offset:8900 nv
	v_pk_mul_f32 v[22:23], v[22:23], v[28:29]
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 1
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v28, v22, v23
	v_pk_add_f32 v[22:23], v[112:113], v[48:49] neg_lo:[0,1] neg_hi:[0,1]
	scratch_load_b64 v[112:113], off, off offset:8908 nv
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[118:119]
	scratch_load_b64 v[118:119], off, off offset:8848 nv
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[30:31], v[30:31], v[254:255] /*v[1022:1023]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v29, v22, v23
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[114:115]
	s_set_vgpr_msb 0x10c
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[22:23], v[22:23], v[254:255] /*v[1022:1023]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v31, v31
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[34:35], v[22:23]
	s_set_vgpr_msb 1
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v34, v22, v23
	v_pk_add_f32 v[22:23], v[124:125], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[132:133]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[252:253] /*v[1020:1021]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v35, v22, v23
	v_pk_add_f32 v[22:23], v[126:127], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	scratch_load_b64 v[126:127], off, off offset:8916 nv
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[134:135]
	scratch_load_b64 v[134:135], off, off offset:8868 nv
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[30:31], v[30:31], v[252:253] /*v[1020:1021]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v36, v22, v23
	v_pk_add_f32 v[22:23], v[128:129], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	scratch_load_b64 v[128:129], off, off offset:6164 nv
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[136:137]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	scratch_load_b64 v[136:137], off, off offset:6172 nv
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[32:33], v[32:33], v[252:253] /*v[1020:1021]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v37, v22, v23
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[130:131]
	s_set_vgpr_msb 0x10c
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[22:23], v[22:23], v[252:253] /*v[1020:1021]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v33, v33
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[38:39], v[122:123] /*v[378:379]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v42, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[108:109] /*v[364:365]*/, v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[116:117] /*v[372:373]*/
	s_set_vgpr_msb 0x501
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v43, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[110:111] /*v[366:367]*/, v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[118:119] /*v[374:375]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[30:31], v[30:31], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v44, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[112:113] /*v[368:369]*/, v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[120:121] /*v[376:377]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[32:33], v[32:33], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v45, v22, v23
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[114:115] /*v[370:371]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[22:23], v[22:23], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v33, v33
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[38:39], v[138:139] /*v[394:395]*/, v[194:195] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v50, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[124:125] /*v[380:381]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[132:133] /*v[388:389]*/
	s_set_vgpr_msb 0x501
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v51, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[126:127] /*v[382:383]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[134:135] /*v[390:391]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[30:31], v[30:31], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v52, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[128:129] /*v[384:385]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[136:137] /*v[392:393]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[32:33], v[32:33], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v53, v22, v23
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[130:131] /*v[386:387]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[22:23], v[22:23], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v33, v33
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[38:39], v[154:155] /*v[410:411]*/, v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v86, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[140:141] /*v[396:397]*/, v[194:195] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[148:149] /*v[404:405]*/
	s_set_vgpr_msb 0x501
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v87, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[142:143] /*v[398:399]*/, v[194:195] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[150:151] /*v[406:407]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[30:31], v[30:31], v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v88, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[144:145] /*v[400:401]*/, v[194:195] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[152:153] /*v[408:409]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[32:33], v[32:33], v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v89, v22, v23
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[146:147] /*v[402:403]*/
	s_set_vgpr_msb 0x50c
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[22:23], v[22:23], v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v33, v33
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 1
	s_wait_loadcnt 0x8
	v_pk_add_f32 v[38:39], v[170:171] /*v[426:427]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v66, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[156:157] /*v[412:413]*/, v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[164:165] /*v[420:421]*/
	s_set_vgpr_msb 0x501
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v67, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[158:159] /*v[414:415]*/, v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[166:167] /*v[422:423]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[30:31], v[30:31], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v68, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[160:161] /*v[416:417]*/, v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[168:169] /*v[424:425]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[32:33], v[32:33], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v69, v22, v23
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[162:163] /*v[418:419]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[22:23], v[22:23], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v33, v33
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 1
	s_wait_loadcnt 0x7
	v_pk_add_f32 v[38:39], v[186:187] /*v[442:443]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v74, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[172:173] /*v[428:429]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[180:181] /*v[436:437]*/
	s_set_vgpr_msb 0x501
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v75, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[174:175] /*v[430:431]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[182:183] /*v[438:439]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[30:31], v[30:31], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v76, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[176:177] /*v[432:433]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[184:185] /*v[440:441]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[32:33], v[32:33], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v77, v22, v23
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[178:179] /*v[434:435]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[22:23], v[22:23], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v33, v33
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 1
	s_wait_loadcnt 0x5
	v_pk_add_f32 v[38:39], v[80:81] /*v[336:337]*/, v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v82, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[188:189] /*v[444:445]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x501
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v83, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[190:191] /*v[446:447]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[30:31], v[30:31], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v84, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[192:193] /*v[448:449]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[32:33], v[32:33], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v85, v22, v23
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[194:195] /*v[450:451]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x48
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[198:205], v[106:113] /*v[618:625]*/, 0
	s_set_vgpr_msb 0x4800
	v_pk_add_f32 v[22:23], v[22:23], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[198:205], v[62:69] /*v[318:325]*/, 0
	s_set_vgpr_msb 0x8400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 13
	v_pk_add_f32 v[38:39], v[96:97] /*v[352:353]*/, v[170:171] /*v[938:939]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xda0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[178:185], v[246:253], v[106:113] /*v[618:625]*/
	s_set_vgpr_msb 0xa001
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v90, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[82:83] /*v[338:339]*/, v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x154
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[178:185], v[242:249] /*v[498:505]*/, v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5400
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[90:91] /*v[346:347]*/
	s_set_vgpr_msb 0x501
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x4
	v_pk_add_f32 v[24:25], v[24:25], v[118:119] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x58
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[146:153], v[194:201] /*v[706:713]*/, v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5800
	v_cvt_pk_bf16_f32 v91, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[84:85] /*v[340:341]*/, v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[92:93] /*v[348:349]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[30:31], v[30:31], v[118:119] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v92, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[86:87] /*v[342:343]*/, v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x144
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[198:205], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3584 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3600 nv
	s_set_vgpr_msb 0x4400
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[94:95] /*v[350:351]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[32:33], v[32:33], v[118:119] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v93, v22, v23
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[88:89] /*v[344:345]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x44
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[186:193], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3264 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3280 nv
	s_set_vgpr_msb 0x4400
	v_pk_add_f32 v[22:23], v[22:23], v[118:119] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[170:177], v[106:113] /*v[874:881]*/, v[88:95] /*v[344:351]*/
	s_set_vgpr_msb 0x5c00
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[38:39], v[10:11] /*v[522:523]*/, v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x258
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[206:213], v[138:145] /*v[650:657]*/, v[88:95] /*v[344:351]*/
	s_set_vgpr_msb 0x5801
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v98, v22, v23
	s_set_vgpr_msb 13
	v_pk_add_f32 v[22:23], v[98:99] /*v[354:355]*/, v[170:171] /*v[938:939]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[4:5] /*v[516:517]*/
	s_set_vgpr_msb 0x901
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v99, v22, v23
	s_set_vgpr_msb 13
	v_pk_add_f32 v[22:23], v[100:101] /*v[356:357]*/, v[170:171] /*v[938:939]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd00
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[6:7] /*v[518:519]*/
	s_set_vgpr_msb 0x901
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_add_f32 v[30:31], v[30:31], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v100, v22, v23
	s_set_vgpr_msb 13
	v_pk_add_f32 v[22:23], v[102:103] /*v[358:359]*/, v[170:171] /*v[938:939]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd44
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[198:205], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3296 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3312 nv
	s_set_vgpr_msb 0x4400
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[8:9] /*v[520:521]*/
	s_set_vgpr_msb 0x901
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	v_pk_add_f32 v[32:33], v[32:33], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x58
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[178:185], v[154:161] /*v[666:673]*/, v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5800
	v_cvt_pk_bf16_f32 v101, v22, v23
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x900
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[22:23], v[22:23], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x88
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[186:193], v[114:121] /*v[626:633]*/, 0
	s_set_vgpr_msb 0x8800
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[186:193], v[54:61] /*v[310:317]*/, 0
	s_set_vgpr_msb 0x8400
	s_delay_alu instid0(TRANS32_DEP_2)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[38:39], v[26:27] /*v[538:539]*/, v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x2a0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[170:177], v[214:221], v[114:121] /*v[626:633]*/
	s_set_vgpr_msb 0xa001
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v106, v22, v23
	s_set_vgpr_msb 2
	v_pk_add_f32 v[22:23], v[12:13] /*v[524:525]*/, v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x258
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[146:153], v[162:169] /*v[674:681]*/, v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5800
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[20:21] /*v[532:533]*/
	s_set_vgpr_msb 0x901
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xac
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[170:177], v[98:105] /*v[866:873]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xac00
	v_cvt_pk_bf16_f32 v107, v22, v23
	s_set_vgpr_msb 2
	v_pk_add_f32 v[22:23], v[14:15] /*v[526:527]*/, v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[22:23] /*v[534:535]*/
	s_set_vgpr_msb 0x9ac
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[206:213], v[2:9] /*v[770:777]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xac01
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x1c0
	s_clause 0x1
	scratch_load_b128 v[2:5] /*v[770:773]*/, off, off offset:1632 nv
	scratch_load_b128 v[6:9] /*v[774:777]*/, off, off offset:1648 nv
	s_set_vgpr_msb 0xc000
	v_pk_add_f32 v[30:31], v[30:31], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v108, v22, v23
	s_set_vgpr_msb 2
	v_pk_add_f32 v[22:23], v[16:17] /*v[528:529]*/, v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x288
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[198:205], v[122:129] /*v[634:641]*/, 0
	s_set_vgpr_msb 0x8800
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[24:25] /*v[536:537]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[198:205], v[30:37] /*v[286:293]*/, 0
	s_set_vgpr_msb 0x8401
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[32:33], v[32:33], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v109, v22, v23
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x944
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[186:193], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3328 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3344 nv
	s_set_vgpr_msb 0x4400
	v_pk_add_f32 v[22:23], v[22:23], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v33, v33
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[178:185], v[162:169], v[122:129] /*v[634:641]*/
	s_set_vgpr_msb 0xa000
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 6
	v_pk_add_f32 v[38:39], v[42:43] /*v[554:555]*/, v[104:105] /*v[360:361]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x658
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[170:177], v[178:185] /*v[690:697]*/, v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5801
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v114, v22, v23
	s_set_vgpr_msb 2
	v_pk_add_f32 v[22:23], v[28:29] /*v[540:541]*/, v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x258
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[206:213], v[186:193] /*v[698:705]*/, v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5800
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[36:37] /*v[548:549]*/
	s_set_vgpr_msb 0x901
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[214:215] /*v[982:983]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xca8
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[178:185], v[242:249] /*v[754:761]*/, v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa800
	v_cvt_pk_bf16_f32 v115, v22, v23
	s_set_vgpr_msb 2
	v_pk_add_f32 v[22:23], v[30:31] /*v[542:543]*/, v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[38:39] /*v[550:551]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 0xac
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[146:153], v[26:33] /*v[794:801]*/, v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xac01
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[30:31], v[30:31], v[214:215] /*v[982:983]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v116, v22, v23
	s_set_vgpr_msb 2
	v_pk_add_f32 v[22:23], v[32:33] /*v[544:545]*/, v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x88
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[198:205], v[90:97] /*v[602:609]*/, 0
	s_set_vgpr_msb 0x8800
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[40:41] /*v[552:553]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[32:33], v[32:33], v[214:215] /*v[982:983]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc84
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[198:205], v[226:233] /*v[482:489]*/, 0
	s_set_vgpr_msb 0x8400
	v_cvt_pk_bf16_f32 v117, v22, v23
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[34:35] /*v[546:547]*/
	s_set_vgpr_msb 0x90c
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[22:23], v[22:23], v[214:215] /*v[982:983]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc88
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[186:193], v[82:89] /*v[594:601]*/, 0
	s_set_vgpr_msb 0x8800
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v22, v22
	s_set_vgpr_msb 0x44
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[198:205], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3360 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3376 nv
	s_set_vgpr_msb 0x4400
	v_exp_f32_e32 v23, v23
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[170:177], v[4:11], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa00e
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:5904 nv
	scratch_load_b128 v[8:11], off, off offset:5920 nv
	v_pk_add_f32 v[38:39], v[58:59] /*v[570:571]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe01
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v122, v22, v23
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[146:153], v[4:11], v[106:113] /*v[618:625]*/
	s_set_vgpr_msb 0xa006
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:5968 nv
	scratch_load_b128 v[8:11], off, off offset:5984 nv
	v_pk_add_f32 v[22:23], v[44:45] /*v[556:557]*/, v[104:105] /*v[360:361]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x600
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 0x44
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[186:193], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3072 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3088 nv
	s_set_vgpr_msb 0x4409
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x901
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xca0
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[206:213], v[4:11], v[114:121] /*v[626:633]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:6032 nv
	scratch_load_b128 v[8:11], off, off offset:6048 nv
	v_cvt_pk_bf16_f32 v123, v22, v23
	s_set_vgpr_msb 6
	v_pk_add_f32 v[22:23], v[46:47] /*v[558:559]*/, v[104:105] /*v[360:361]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x600
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 0x44
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[198:205], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3392 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3408 nv
	s_set_vgpr_msb 0x4400
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x109
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x900
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v124, v22, v23
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[146:153], v[4:11], v[122:129] /*v[634:641]*/
	s_set_vgpr_msb 0xa006
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:6096 nv
	scratch_load_b128 v[8:11], off, off offset:6112 nv
	v_pk_add_f32 v[22:23], v[48:49] /*v[560:561]*/, v[104:105] /*v[360:361]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x60c
	v_pk_add_f32 v[30:31], v[30:31], v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 0x44
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[186:193], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3648 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3664 nv
	s_set_vgpr_msb 0x4400
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x109
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[56:57] /*v[568:569]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v30, v30
	v_cvt_pk_bf16_f32 v125, v22, v23
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[50:51] /*v[562:563]*/
	s_set_vgpr_msb 0x90c
	v_exp_f32_e32 v31, v31
	v_pk_add_f32 v[32:33], v[32:33], v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc88
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[198:205], v[98:105] /*v[610:617]*/, 0
	s_set_vgpr_msb 0x880c
	v_pk_add_f32 v[22:23], v[22:23], v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc84
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[186:193], v[218:225] /*v[474:481]*/, 0
	s_set_vgpr_msb 0x8400
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 14
	v_pk_add_f32 v[38:39], v[74:75] /*v[586:587]*/, v[216:217] /*v[984:985]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe84
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[186:193], v[202:209] /*v[458:465]*/, 0
	s_set_vgpr_msb 0x8401
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v130, v22, v23
	s_set_vgpr_msb 14
	v_pk_add_f32 v[22:23], v[60:61] /*v[572:573]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe58
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[178:185], v[202:209] /*v[714:721]*/, v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5800
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[68:69] /*v[580:581]*/
	s_set_vgpr_msb 0x901
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[24:25], v[24:25], v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc58
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[146:153], v[218:225] /*v[730:737]*/, v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5800
	v_cvt_pk_bf16_f32 v131, v22, v23
	s_set_vgpr_msb 14
	v_pk_add_f32 v[22:23], v[62:63] /*v[574:575]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe00
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_exp_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v25, v25
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[70:71] /*v[582:583]*/
	s_set_vgpr_msb 0x944
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[198:205], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3712 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3728 nv
	s_set_vgpr_msb 0x4401
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[30:31], v[30:31], v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v132, v22, v23
	s_set_vgpr_msb 14
	v_pk_add_f32 v[22:23], v[64:65] /*v[576:577]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe00
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8c
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[198:205], v[242:249] /*v[1010:1017]*/, 0
	s_set_vgpr_msb 0x8c00
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[72:73] /*v[584:585]*/
	s_set_vgpr_msb 0x901
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x158
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[170:177], v[234:241] /*v[746:753]*/, v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x580c
	v_pk_add_f32 v[32:33], v[32:33], v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v133, v22, v23
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x900
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8c
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[186:193], v[90:97] /*v[858:865]*/, 0
	s_set_vgpr_msb 0x8c0c
	v_pk_add_f32 v[22:23], v[22:23], v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc58
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[206:213], v[130:137] /*v[642:649]*/, v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5800
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_nop
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[178:185], v[234:241] /*v[490:497]*/, v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5401
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v0, v22, v23
	s_set_vgpr_msb 14
	v_pk_add_f32 v[22:23], v[76:77] /*v[588:589]*/, v[216:217] /*v[984:985]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe5c
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[146:153], v[146:153] /*v[914:921]*/, v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5c00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 1
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x144
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[186:193], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3776 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3792 nv
	s_set_vgpr_msb 0x4400
	v_cvt_pk_bf16_f32 v1, v22, v23
	s_set_vgpr_msb 14
	v_pk_add_f32 v[22:23], v[78:79] /*v[590:591]*/, v[216:217] /*v[984:985]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[170:177], v[60:67] /*v[828:835]*/, v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5c01
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v2, v22, v23
	s_set_vgpr_msb 14
	v_pk_add_f32 v[22:23], v[80:81] /*v[592:593]*/, v[216:217] /*v[984:985]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe8c
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[198:205], v[180:187] /*v[948:955]*/, 0
	s_set_vgpr_msb 0x8c00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 1
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x15c
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[206:213], v[122:129] /*v[890:897]*/, v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5c00
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v3, v22, v23
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[178:185], v[138:145] /*v[906:913]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5c50
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[146:153], v[230:237], v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x505c
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[170:177], v[172:179] /*v[940:947]*/, v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5c54
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[206:213], v[210:217] /*v[466:473]*/, v[154:161] /*v[410:417]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[88:89]
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[86:87]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[198:205], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3840 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3856 nv
	s_set_vgpr_msb 0x545c
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[178:185], v[188:195] /*v[956:963]*/, v[162:169] /*v[418:425]*/
	s_set_vgpr_msb 0x5c54
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[146:153], v[6:13] /*v[262:269]*/, v[162:169] /*v[418:425]*/
	s_set_vgpr_msb 0x54a0
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[178:185], v[254:261], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa0ac
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[146:153], v[82:89] /*v[850:857]*/, v[26:33] /*v[538:545]*/
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[206:213], v[130:137] /*v[898:905]*/, v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xaca4
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[178:185], v[46:53] /*v[302:309]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa4ac
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[146:153], v[162:169] /*v[930:937]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xaca4
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[178:185], v[250:257] /*v[506:513]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa444
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[186:193], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3904 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3920 nv
	s_set_vgpr_msb 0x445c
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[170:177], v[204:211] /*v[972:979]*/, v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5c54
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[206:213], v[22:29] /*v[278:285]*/, v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x54a8
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[146:153], v[226:233] /*v[738:745]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa8ac
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[170:177], v[44:51] /*v[812:819]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xacc0
	s_clause 0x1
	scratch_load_b128 v[44:47] /*v[812:815]*/, off, off offset:5504 th:TH_LOAD_LU nv
	scratch_load_b128 v[48:51] /*v[816:819]*/, off, off offset:5520 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc0ac
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[206:213], v[154:161] /*v[922:929]*/, v[66:73] /*v[578:585]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[178:185], v[18:25] /*v[786:793]*/, v[74:81] /*v[586:593]*/
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[146:153], v[218:225] /*v[986:993]*/, v[74:81] /*v[586:593]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[170:177], v[114:121] /*v[882:889]*/, v[82:89] /*v[594:601]*/
	s_set_vgpr_msb 0xac44
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[198:205], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3968 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3984 nv
	s_set_vgpr_msb 0x4400
	v_wmma_f32_16x16x32_bf16 v[198:205], v[186:193], v[238:245], 0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[198:205], v[170:177], v[58:65], v[198:205]
	v_wmma_f32_16x16x32_bf16 v[198:205], v[206:213], v[4:11], v[198:205]
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3232 nv
	scratch_load_b128 v[8:11], off, off offset:3248 nv
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[178:185], v[226:233] /*v[994:1001]*/, v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5c58
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[146:153], v[146:153] /*v[658:665]*/, v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x58a8
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[206:213], v[210:217] /*v[722:729]*/, v[82:89] /*v[594:601]*/
	s_set_vgpr_msb 0xa8ac
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[178:185], v[10:17] /*v[778:785]*/, v[90:97] /*v[602:609]*/
	s_set_vgpr_msb 0xac44
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[186:193], v[14:21] /*v[270:277]*/, 0
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:5648 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:5664 nv
	s_set_vgpr_msb 0x445c
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[170:177], v[36:43] /*v[804:811]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5c58
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[206:213], v[170:177] /*v[682:689]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x58a8
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[146:153], v[250:257] /*v[762:769]*/, v[90:97] /*v[602:609]*/
	s_set_vgpr_msb 0xa8a4
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[170:177], v[38:45] /*v[294:301]*/, v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xa4ac
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[206:213], v[234:241] /*v[1002:1009]*/, v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xac84
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[186:193], v[14:21] /*v[270:277]*/, 0
	s_set_vgpr_msb 0x8440
	s_clause 0x1
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:5712 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:5728 nv
	s_set_vgpr_msb 0x40a0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[170:177], v[222:229], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa0ac
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[206:213], v[68:75] /*v[836:843]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xac84
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[186:193], v[14:21] /*v[270:277]*/, 0
	s_set_vgpr_msb 0x8400
	s_clause 0x1
	scratch_load_b128 v[186:189], off, off offset:5584 nv
	scratch_load_b128 v[190:193], off, off offset:5600 nv
	s_set_vgpr_msb 0xac
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[170:177], v[52:59] /*v[820:827]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xac00
	s_clause 0x6
	scratch_load_b128 v[170:173], off, off offset:544 nv
	scratch_load_b128 v[174:177], off, off offset:560 nv
	scratch_load_b128 v[162:165], off, off offset:576 nv
	scratch_load_b128 v[166:169], off, off offset:592 nv
	s_set_vgpr_msb 0xac
	scratch_load_b128 v[130:133] /*v[642:645]*/, off, off offset:1216 nv
	scratch_load_b128 v[134:137] /*v[646:649]*/, off, off offset:1232 nv
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[206:213], v[196:203] /*v[964:971]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xacc0
	s_clause 0x1
	scratch_load_b128 v[196:199] /*v[964:967]*/, off, off offset:5088 th:TH_LOAD_LU nv
	scratch_load_b128 v[200:203] /*v[968:971]*/, off, off offset:5104 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc050
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[178:185], v[186:193], v[80:87] /*v[336:343]*/
	s_set_vgpr_msb 0x5054
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[146:153], v[70:77] /*v[326:333]*/, v[80:87] /*v[336:343]*/
	s_set_vgpr_msb 0x5450
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[138:145], v[4:11], v[80:87] /*v[336:343]*/
	s_set_vgpr_msb 0x5005
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3136 nv
	scratch_load_b128 v[8:11], off, off offset:3152 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[80:81] /*v[336:337]*/
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[82:83] /*v[338:339]*/
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[84:85] /*v[340:341]*/
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[86:87] /*v[342:343]*/
	s_set_vgpr_msb 0x550
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[154:161], v[4:11], v[88:95] /*v[344:351]*/
	s_set_vgpr_msb 0x500c
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4480 nv
	scratch_load_b128 v[8:11], off, off offset:4496 nv
	v_pk_add_f32 v[22:23], v[22:23], v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25], v[24:25], v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[30:31], v[30:31], v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33], v[32:33], v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_mul_f32 v[22:23], v[22:23], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc01
	v_pk_add_f32 v[38:39], v[88:89] /*v[344:345]*/, v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[138:145], v[4:11], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3168 nv
	scratch_load_b128 v[8:11], off, off offset:3184 nv
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[38:39], v[90:91] /*v[346:347]*/, v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[78:79] /*v[334:335]*/, v[22:23]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[24:25], v[38:39], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[38:39], v[78:79] /*v[334:335]*/, v[100:101] /*v[356:357]*/
	s_set_vgpr_msb 0x550
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[154:161], v[4:11], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5001
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3200 nv
	scratch_load_b128 v[8:11], off, off offset:3216 nv
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[24:25]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v22, v22, v23
	s_set_vgpr_msb 12
	v_pk_add_f32 v[38:39], v[38:39], v[212:213] /*v[980:981]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v23, v24, v25
	s_set_vgpr_msb 1
	v_pk_add_f32 v[24:25], v[92:93] /*v[348:349]*/, v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[46:47], v[106:107] /*v[362:363]*/, v[48:49] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[38:39], v[38:39], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[138:145], v[4:11], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3040 nv
	scratch_load_b128 v[8:11], off, off offset:3056 nv
	v_pk_mul_f32 v[24:25], v[24:25], v[30:31]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[30:31], v[94:95] /*v[350:351]*/, v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[40:41], v[78:79] /*v[334:335]*/, v[102:103] /*v[358:359]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[24:25], v[78:79] /*v[334:335]*/, v[24:25]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[30:31], v[30:31], v[32:33]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[98:99] /*v[354:355]*/
	s_set_vgpr_msb 0x50c
	v_pk_add_f32 v[40:41], v[40:41], v[212:213] /*v[980:981]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[154:161], v[4:11], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5001
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3104 nv
	scratch_load_b128 v[8:11], off, off offset:3120 nv
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[30:31]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v24, v24, v25
	s_set_vgpr_msb 12
	v_pk_add_f32 v[32:33], v[32:33], v[212:213] /*v[980:981]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[40:41], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v25, v30, v31
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[96:97] /*v[352:353]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[32:33], v[32:33], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v40, v40
	v_exp_f32_e32 v41, v41
	s_set_vgpr_msb 1
	v_pk_add_f32 v[54:55], v[122:123] /*v[378:379]*/, v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[30:31], v[30:31], v[212:213] /*v[980:981]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	s_set_vgpr_msb 0xc40
	s_clause 0x9
	scratch_load_b128 v[96:99] /*v[352:355]*/, off, off offset:768 nv
	scratch_load_b128 v[100:103] /*v[356:359]*/, off, off offset:784 nv
	scratch_load_b128 v[250:253] /*v[506:509]*/, off, off offset:1408 nv
	scratch_load_b128 v[254:257] /*v[510:513]*/, off, off offset:1424 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[68:71] /*v[836:839]*/, off, off offset:1600 nv
	scratch_load_b128 v[72:75] /*v[840:843]*/, off, off offset:1616 nv
	s_set_vgpr_msb 0xc050
	scratch_load_b128 v[226:229] /*v[482:485]*/, off, off offset:1856 nv
	scratch_load_b128 v[230:233] /*v[486:489]*/, off, off offset:1872 nv
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[138:145], v[4:11], v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3616 nv
	scratch_load_b128 v[8:11], off, off offset:3632 nv
	v_pk_mul_f32 v[30:31], v[30:31], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[154:161], v[4:11], v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3680 nv
	scratch_load_b128 v[8:11], off, off offset:3696 nv
	v_pk_mul_f32 v[30:31], v[46:47], v[30:31]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[46:47], v[108:109] /*v[364:365]*/, v[48:49] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[30:31], v[78:79] /*v[334:335]*/, v[30:31]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[32:33], v[46:47], v[32:33]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[46:47], v[78:79] /*v[334:335]*/, v[118:119] /*v[374:375]*/
	s_set_vgpr_msb 0x501
	v_pk_add_f32 v[62:63], v[138:139] /*v[394:395]*/, v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x150
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[138:145], v[4:11], v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5001
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3744 nv
	scratch_load_b128 v[8:11], off, off offset:3760 nv
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[32:33]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v30, v30, v31
	s_set_vgpr_msb 12
	v_pk_add_f32 v[46:47], v[46:47], v[254:255] /*v[1022:1023]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v31, v32, v33
	s_set_vgpr_msb 1
	v_pk_add_f32 v[32:33], v[110:111] /*v[366:367]*/, v[48:49] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[46:47], v[46:47], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[64:65], v[78:79] /*v[334:335]*/, v[152:153] /*v[408:409]*/
	s_set_vgpr_msb 0x550
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[154:161], v[4:11], v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3808 nv
	scratch_load_b128 v[8:11], off, off offset:3824 nv
	v_pk_mul_f32 v[32:33], v[32:33], v[38:39]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[38:39], v[112:113] /*v[368:369]*/, v[48:49] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[48:49], v[78:79] /*v[334:335]*/, v[120:121] /*v[376:377]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[32:33], v[78:79] /*v[334:335]*/, v[32:33]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[38:39], v[38:39], v[40:41]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[40:41], v[78:79] /*v[334:335]*/, v[116:117] /*v[372:373]*/
	s_set_vgpr_msb 0x50c
	v_pk_add_f32 v[48:49], v[48:49], v[254:255] /*v[1022:1023]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	v_pk_add_f32 v[64:65], v[64:65], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v32, v32, v33
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[38:39], v[78:79] /*v[334:335]*/, v[38:39]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[40:41], v[40:41], v[254:255] /*v[1022:1023]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[48:49], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc40
	s_clause 0xa
	scratch_load_b128 v[106:109] /*v[362:365]*/, off, off offset:800 nv
	scratch_load_b128 v[110:113] /*v[366:369]*/, off, off offset:816 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[242:245] /*v[754:757]*/, off, off offset:1376 nv
	scratch_load_b128 v[246:249] /*v[758:761]*/, off, off offset:1392 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[180:183] /*v[948:951]*/, off, off offset:1664 nv
	scratch_load_b128 v[184:187] /*v[952:955]*/, off, off offset:1680 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[234:237] /*v[490:493]*/, off, off offset:1888 nv
	scratch_load_b128 v[238:241] /*v[494:497]*/, off, off offset:1904 nv
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v33, v38, v39
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[38:39], v[78:79] /*v[334:335]*/, v[114:115] /*v[370:371]*/
	s_set_vgpr_msb 0x50c
	v_pk_mul_f32 v[40:41], v[40:41], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	v_exp_f32_e32 v64, v64
	v_pk_add_f32 v[38:39], v[38:39], v[254:255] /*v[1022:1023]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v40, v40
	v_exp_f32_e32 v41, v41
	v_exp_f32_e32 v65, v65
	s_set_vgpr_msb 0xc50
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[138:145], v[4:11], v[162:169] /*v[418:425]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3872 nv
	scratch_load_b128 v[8:11], off, off offset:3888 nv
	v_pk_mul_f32 v[38:39], v[38:39], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v38, v38
	v_exp_f32_e32 v39, v39
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[154:161], v[4:11], v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:3936 nv
	scratch_load_b128 v[8:11], off, off offset:3952 nv
	v_pk_mul_f32 v[38:39], v[54:55], v[38:39]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[54:55], v[124:125] /*v[380:381]*/, v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[38:39], v[78:79] /*v[334:335]*/, v[38:39]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[40:41], v[54:55], v[40:41]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[54:55], v[78:79] /*v[334:335]*/, v[134:135] /*v[390:391]*/
	s_set_vgpr_msb 0x550
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[138:145], v[4:11], v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5001
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4000 nv
	scratch_load_b128 v[8:11], off, off offset:4016 nv
	v_pk_mul_f32 v[40:41], v[78:79] /*v[334:335]*/, v[40:41]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v38, v38, v39
	s_set_vgpr_msb 12
	v_pk_add_f32 v[54:55], v[54:55], v[252:253] /*v[1020:1021]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v39, v40, v41
	s_set_vgpr_msb 1
	v_pk_add_f32 v[40:41], v[126:127] /*v[382:383]*/, v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[54:55], v[54:55], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[80:81], v[78:79] /*v[334:335]*/, v[184:185] /*v[440:441]*/
	s_set_vgpr_msb 0x550
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[154:161], v[4:11], v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4032 nv
	scratch_load_b128 v[8:11], off, off offset:4048 nv
	v_pk_mul_f32 v[40:41], v[40:41], v[46:47]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[46:47], v[128:129] /*v[384:385]*/, v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[56:57], v[78:79] /*v[334:335]*/, v[136:137] /*v[392:393]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[40:41], v[78:79] /*v[334:335]*/, v[40:41]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[46:47], v[46:47], v[48:49]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[48:49], v[78:79] /*v[334:335]*/, v[132:133] /*v[388:389]*/
	s_set_vgpr_msb 0x50c
	v_pk_add_f32 v[56:57], v[56:57], v[252:253] /*v[1020:1021]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc01
	v_pk_add_f32 v[86:87], v[186:187] /*v[442:443]*/, v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v40, v40, v41
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[46:47], v[78:79] /*v[334:335]*/, v[46:47]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[48:49], v[48:49], v[252:253] /*v[1020:1021]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[56:57], s[6:7] op_sel_hi:[1,0]
	v_pk_add_f32 v[80:81], v[80:81], v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[138:145], v[4:11], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4064 nv
	scratch_load_b128 v[8:11], off, off offset:4080 nv
	v_cvt_pk_bf16_f32 v41, v46, v47
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[46:47], v[78:79] /*v[334:335]*/, v[130:131] /*v[386:387]*/
	s_set_vgpr_msb 0x50c
	v_pk_mul_f32 v[48:49], v[48:49], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_pk_mul_f32 v[80:81], v[80:81], s[6:7] op_sel_hi:[1,0]
	v_pk_add_f32 v[46:47], v[46:47], v[252:253] /*v[1020:1021]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v48, v48
	v_exp_f32_e32 v49, v49
	s_set_vgpr_msb 0xc05
	v_pk_mul_f32 v[88:89], v[78:79] /*v[334:335]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v80, v80
	v_pk_mul_f32 v[46:47], v[46:47], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v81, v81
	s_set_vgpr_msb 0xc0
	s_clause 0x1
	scratch_load_b64 v[252:253] /*v[1020:1021]*/, off, off offset:8936 nv
	scratch_load_b64 v[254:255] /*v[1022:1023]*/, off, off offset:8928 nv
	s_set_vgpr_msb 0xc000
	v_pk_add_f32 v[88:89], v[88:89], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[154:161], v[4:11], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4096 nv
	scratch_load_b128 v[8:11], off, off offset:4112 nv
	v_exp_f32_e32 v46, v46
	v_exp_f32_e32 v47, v47
	v_pk_mul_f32 v[88:89], v[88:89], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v89, v89
	s_delay_alu instid0(TRANS32_DEP_3)
	v_pk_mul_f32 v[46:47], v[62:63], v[46:47]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[62:63], v[140:141] /*v[396:397]*/, v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x1a0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[138:145], v[4:11], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4128 nv
	scratch_load_b128 v[8:11], off, off offset:4144 nv
	v_pk_mul_f32 v[48:49], v[62:63], v[48:49]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[46:47], v[78:79] /*v[334:335]*/, v[46:47]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[62:63], v[78:79] /*v[334:335]*/, v[150:151] /*v[406:407]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[48:49], v[78:79] /*v[334:335]*/, v[48:49]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v46, v46, v47
	v_pk_add_f32 v[62:63], v[62:63], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[154:161], v[4:11], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4160 nv
	scratch_load_b128 v[8:11], off, off offset:4176 nv
	v_cvt_pk_bf16_f32 v47, v48, v49
	s_set_vgpr_msb 1
	v_pk_add_f32 v[48:49], v[142:143] /*v[398:399]*/, v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[62:63], v[62:63], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[48:49], v[48:49], v[54:55]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[54:55], v[144:145] /*v[400:401]*/, v[12:13] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v62, v62
	v_exp_f32_e32 v63, v63
	s_clause 0xc
	scratch_load_b128 v[178:181], off, off offset:608 nv
	scratch_load_b128 v[182:185], off, off offset:624 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[138:141] /*v[394:397]*/, off, off offset:1024 nv
	scratch_load_b128 v[142:145] /*v[398:401]*/, off, off offset:1040 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[178:181] /*v[690:693]*/, off, off offset:1248 nv
	scratch_load_b128 v[182:185] /*v[694:697]*/, off, off offset:1264 nv
	scratch_load_b128 v[218:221] /*v[730:733]*/, off, off offset:2784 nv
	scratch_load_b128 v[222:225] /*v[734:737]*/, off, off offset:2800 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[188:191] /*v[956:959]*/, off, off offset:2240 nv
	scratch_load_b128 v[192:195] /*v[960:963]*/, off, off offset:2256 nv
	s_set_vgpr_msb 0xc001
	v_pk_mul_f32 v[48:49], v[78:79] /*v[334:335]*/, v[48:49]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[54:55], v[54:55], v[56:57]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[56:57], v[78:79] /*v[334:335]*/, v[148:149] /*v[404:405]*/
	s_set_vgpr_msb 0x5a0
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[138:145], v[4:11], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa001
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4192 nv
	scratch_load_b128 v[8:11], off, off offset:4208 nv
	v_pk_mul_f32 v[54:55], v[78:79] /*v[334:335]*/, v[54:55]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v48, v48, v49
	v_pk_add_f32 v[56:57], v[56:57], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v49, v54, v55
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[54:55], v[78:79] /*v[334:335]*/, v[146:147] /*v[402:403]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[56:57], v[56:57], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 64
	s_clause 0x1
	scratch_load_b128 v[146:149] /*v[402:405]*/, off, off offset:5248 th:TH_LOAD_LU nv
	scratch_load_b128 v[150:153] /*v[406:409]*/, off, off offset:5264 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x40a0
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[154:161], v[4:11], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4224 nv
	scratch_load_b128 v[8:11], off, off offset:4240 nv
	v_pk_add_f32 v[54:55], v[54:55], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[70:71], v[154:155] /*v[410:411]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v57, v57
	v_pk_mul_f32 v[54:55], v[54:55], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[138:145], v[4:11], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4256 nv
	scratch_load_b128 v[8:11], off, off offset:4272 nv
	v_exp_f32_e32 v54, v54
	v_exp_f32_e32 v55, v55
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[54:55], v[70:71], v[54:55]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[70:71], v[156:157] /*v[412:413]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x1a0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[154:161], v[4:11], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4288 nv
	scratch_load_b128 v[8:11], off, off offset:4304 nv
	v_pk_mul_f32 v[56:57], v[70:71], v[56:57]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[54:55], v[78:79] /*v[334:335]*/, v[54:55]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[70:71], v[78:79] /*v[334:335]*/, v[166:167] /*v[422:423]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[56:57], v[78:79] /*v[334:335]*/, v[56:57]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v54, v54, v55
	v_pk_add_f32 v[70:71], v[70:71], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[138:145], v[4:11], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4320 nv
	scratch_load_b128 v[8:11], off, off offset:4336 nv
	v_cvt_pk_bf16_f32 v55, v56, v57
	s_set_vgpr_msb 1
	v_pk_add_f32 v[56:57], v[158:159] /*v[414:415]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[70:71], v[70:71], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[56:57], v[56:57], v[62:63]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[62:63], v[160:161] /*v[416:417]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[72:73], v[78:79] /*v[334:335]*/, v[168:169] /*v[424:425]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[56:57], v[78:79] /*v[334:335]*/, v[56:57]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[62:63], v[62:63], v[64:65]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[64:65], v[78:79] /*v[334:335]*/, v[164:165] /*v[420:421]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[72:73], v[72:73], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[120:121], v[78:79] /*v[334:335]*/, v[64:65] /*v[576:577]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v56, v56, v57
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[62:63], v[78:79] /*v[334:335]*/, v[62:63]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[64:65], v[64:65], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[72:73], s[6:7] op_sel_hi:[1,0]
	v_pk_add_f32 v[120:121], v[120:121], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[154:161], v[4:11], v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4352 nv
	scratch_load_b128 v[8:11], off, off offset:4368 nv
	v_cvt_pk_bf16_f32 v57, v62, v63
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[62:63], v[78:79] /*v[334:335]*/, v[162:163] /*v[418:419]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[64:65], v[64:65], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v72, v72
	v_exp_f32_e32 v73, v73
	v_pk_mul_f32 v[120:121], v[120:121], s[6:7] op_sel_hi:[1,0]
	v_pk_add_f32 v[62:63], v[62:63], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[78:79], v[170:171] /*v[426:427]*/, v[194:195] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_exp_f32_e32 v120, v120
	v_pk_mul_f32 v[62:63], v[62:63], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v121, v121
	s_set_vgpr_msb 64
	s_clause 0x1
	scratch_load_b128 v[162:165] /*v[418:421]*/, off, off offset:5280 th:TH_LOAD_LU nv
	scratch_load_b128 v[166:169] /*v[422:425]*/, off, off offset:5296 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x40a0
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[138:145], v[4:11], v[74:81] /*v[586:593]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4384 nv
	scratch_load_b128 v[8:11], off, off offset:4400 nv
	v_exp_f32_e32 v62, v62
	v_exp_f32_e32 v63, v63
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[62:63], v[78:79], v[62:63]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[78:79], v[172:173] /*v[428:429]*/, v[194:195] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x1a0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[154:161], v[4:11], v[82:89] /*v[594:601]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4416 nv
	scratch_load_b128 v[8:11], off, off offset:4432 nv
	v_pk_mul_f32 v[64:65], v[78:79], v[64:65]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[78:79], v[78:79] /*v[334:335]*/, v[182:183] /*v[438:439]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[62:63], v[78:79] /*v[334:335]*/, v[62:63]
	v_pk_mul_f32 v[64:65], v[78:79] /*v[334:335]*/, v[64:65]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[78:79], v[78:79], v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc40
	v_cvt_pk_bf16_f32 v214 /*v470*/, v62, v63
	s_set_vgpr_msb 0x40a0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[138:145], v[4:11], v[90:97] /*v[602:609]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4448 nv
	scratch_load_b128 v[8:11], off, off offset:4464 nv
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v215 /*v471*/, v64, v65
	s_set_vgpr_msb 0x4001
	v_pk_add_f32 v[64:65], v[174:175] /*v[430:431]*/, v[194:195] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[78:79], v[78:79], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[64:65], v[64:65], v[70:71]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[70:71], v[176:177] /*v[432:433]*/, v[194:195] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v78, v78
	v_exp_f32_e32 v79, v79
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[154:161], v[4:11], v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4512 nv
	scratch_load_b128 v[8:11], off, off offset:4528 nv
	v_pk_mul_f32 v[70:71], v[70:71], v[72:73]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[72:73], v[78:79] /*v[334:335]*/, v[180:181] /*v[436:437]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[64:65], v[78:79] /*v[334:335]*/, v[64:65]
	v_pk_mul_f32 v[70:71], v[78:79] /*v[334:335]*/, v[70:71]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[72:73], v[72:73], v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc40
	v_cvt_pk_bf16_f32 v216 /*v472*/, v64, v65
	s_set_vgpr_msb 0x4000
	s_clause 0x4
	scratch_load_b128 v[58:61], off, off offset:5408 th:TH_LOAD_LU nv
	scratch_load_b128 v[62:65], off, off offset:5424 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc0
	scratch_load_b128 v[234:237] /*v[1002:1005]*/, off, off offset:2752 nv
	scratch_load_b128 v[238:241] /*v[1006:1009]*/, off, off offset:2768 nv
	s_set_vgpr_msb 0xc0a0
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[138:145], v[4:11], v[106:113] /*v[618:625]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4544 nv
	scratch_load_b128 v[8:11], off, off offset:4560 nv
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v217 /*v473*/, v70, v71
	s_set_vgpr_msb 0x4005
	v_pk_mul_f32 v[70:71], v[78:79] /*v[334:335]*/, v[178:179] /*v[434:435]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[72:73], v[72:73], s[6:7] op_sel_hi:[1,0]
	s_clause 0x9
	scratch_load_b128 v[186:189], off, off offset:640 nv
	scratch_load_b128 v[190:193], off, off offset:656 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:1120 nv
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:1136 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[146:149] /*v[914:917]*/, off, off offset:1728 nv
	scratch_load_b128 v[150:153] /*v[918:921]*/, off, off offset:1744 nv
	scratch_load_b128 v[218:221] /*v[986:989]*/, off, off offset:1984 nv
	scratch_load_b128 v[222:225] /*v[990:993]*/, off, off offset:2000 nv
	s_set_vgpr_msb 0xc00c
	v_pk_add_f32 v[70:71], v[70:71], v[250:251] /*v[1018:1019]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v72, v72
	v_exp_f32_e32 v73, v73
	s_set_vgpr_msb 0xcc0
	scratch_load_b64 v[250:251] /*v[1018:1019]*/, off, off offset:8944 nv
	s_set_vgpr_msb 0xc0a0
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[154:161], v[4:11], v[114:121] /*v[626:633]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4576 nv
	scratch_load_b128 v[8:11], off, off offset:4592 nv
	v_pk_mul_f32 v[70:71], v[70:71], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_nop
	s_set_vgpr_msb 14
	v_pk_add_f32 v[146:147], v[114:115] /*v[626:627]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xea0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[138:145], v[4:11], v[122:129] /*v[634:641]*/
	s_set_vgpr_msb 0xa000
	v_pk_mul_f32 v[70:71], v[86:87], v[70:71]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[86:87], v[188:189] /*v[444:445]*/, v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	s_clause 0x1
	scratch_load_b128 v[4:7], off, off offset:4608 nv
	scratch_load_b128 v[8:11], off, off offset:4624 nv
	v_pk_mul_f32 v[70:71], v[78:79] /*v[334:335]*/, v[70:71]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[86:87], v[78:79] /*v[334:335]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x506
	v_pk_add_f32 v[142:143], v[98:99] /*v[610:611]*/, v[104:105] /*v[360:361]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[112:113] /*v[624:625]*/, v[78:79] /*v[334:335]*/
	s_set_vgpr_msb 0x600
	v_cvt_pk_bf16_f32 v70, v70, v71
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[72:73], v[78:79] /*v[334:335]*/, v[72:73]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[86:87], v[86:87], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[148:149], v[78:79] /*v[334:335]*/, v[128:129] /*v[640:641]*/
	s_set_vgpr_msb 0x90c
	v_pk_add_f32 v[144:145], v[144:145], v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[198:205], v[154:161], v[4:11], v[198:205]
	v_cvt_pk_bf16_f32 v71, v72, v73
	s_set_vgpr_msb 1
	v_pk_add_f32 v[72:73], v[190:191] /*v[446:447]*/, v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x10c
	v_pk_mul_f32 v[86:87], v[86:87], s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145], v[144:145], s[6:7] op_sel_hi:[1,0]
	v_pk_add_f32 v[148:149], v[148:149], v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_clause 0x4
	scratch_load_b128 v[154:157], off, off offset:512 nv
	scratch_load_b128 v[158:161], off, off offset:528 nv
	s_set_vgpr_msb 0xc40
	scratch_load_b128 v[170:173] /*v[426:429]*/, off, off offset:1088 nv
	scratch_load_b128 v[174:177] /*v[430:433]*/, off, off offset:1104 nv
	s_set_vgpr_msb 0x4000
	v_pk_mul_f32 v[72:73], v[72:73], v[78:79]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[78:79], v[192:193] /*v[448:449]*/, v[14:15] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v144, v144
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[72:73], v[78:79] /*v[334:335]*/, v[72:73]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[78:79], v[78:79], v[80:81]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[80:81], v[78:79] /*v[334:335]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x50c
	v_exp_f32_e32 v145, v145
	v_pk_add_f32 v[150:151], v[198:199], v[216:217] /*v[984:985]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v72, v72, v73
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[78:79], v[78:79] /*v[334:335]*/, v[78:79]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[80:81], v[80:81], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[148:149], v[148:149], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 64
	s_clause 0x9
	scratch_load_b128 v[30:33] /*v[286:289]*/, off, off offset:2976 nv
	scratch_load_b128 v[34:37] /*v[290:293]*/, off, off offset:2992 nv
	scratch_load_b128 v[62:65] /*v[318:321]*/, off, off offset:864 nv
	scratch_load_b128 v[66:69] /*v[322:325]*/, off, off offset:880 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[36:39] /*v[804:807]*/, off, off offset:1312 nv
	scratch_load_b128 v[40:43] /*v[808:811]*/, off, off offset:1328 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[10:13], off, off offset:1952 nv
	scratch_load_b128 v[14:17], off, off offset:1968 nv
	v_cvt_pk_bf16_f32 v73, v78, v79
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[78:79], v[78:79] /*v[334:335]*/, v[194:195] /*v[450:451]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[80:81], v[80:81], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v148, v148
	v_exp_f32_e32 v149, v149
	v_pk_add_f32 v[78:79], v[78:79], v[94:95] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[94:95], v[2:3] /*v[514:515]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_exp_f32_e32 v80, v80
	v_exp_f32_e32 v81, v81
	v_pk_mul_f32 v[78:79], v[78:79], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v78, v78
	v_exp_f32_e32 v79, v79
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[78:79], v[94:95], v[78:79]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[94:95], v[4:5] /*v[516:517]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x201
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[78:79], v[78:79] /*v[334:335]*/, v[78:79]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[94:95], v[78:79] /*v[334:335]*/, v[14:15] /*v[526:527]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v78, v78, v79
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[80:81], v[78:79] /*v[334:335]*/, v[80:81]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[94:95], v[94:95], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v79, v80, v81
	s_set_vgpr_msb 2
	v_pk_add_f32 v[80:81], v[6:7] /*v[518:519]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_pk_mul_f32 v[94:95], v[94:95], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[80:81], v[80:81], v[86:87]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[86:87], v[8:9] /*v[520:521]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x209
	v_pk_mul_f32 v[96:97], v[78:79] /*v[334:335]*/, v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[80:81], v[78:79] /*v[334:335]*/, v[80:81]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[86:87], v[86:87], v[88:89]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[88:89], v[78:79] /*v[334:335]*/, v[12:13] /*v[524:525]*/
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[96:97], v[96:97], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v80, v80, v81
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[86:87], v[78:79] /*v[334:335]*/, v[86:87]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[88:89], v[88:89], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[96:97], v[96:97], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v81, v86, v87
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[86:87], v[78:79] /*v[334:335]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x900
	v_pk_mul_f32 v[88:89], v[88:89], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_pk_add_f32 v[86:87], v[86:87], v[102:103] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[102:103], v[18:19] /*v[530:531]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v89, v89
	v_pk_mul_f32 v[86:87], v[86:87], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v87, v87
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[86:87], v[102:103], v[86:87]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[102:103], v[20:21] /*v[532:533]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x201
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[86:87], v[78:79] /*v[334:335]*/, v[86:87]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[88:89], v[102:103], v[88:89]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[102:103], v[78:79] /*v[334:335]*/, v[30:31] /*v[542:543]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v86, v86, v87
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[88:89], v[78:79] /*v[334:335]*/, v[88:89]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[102:103], v[102:103], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v87, v88, v89
	s_set_vgpr_msb 2
	v_pk_add_f32 v[88:89], v[22:23] /*v[534:535]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_pk_mul_f32 v[102:103], v[102:103], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[88:89], v[88:89], v[94:95]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[94:95], v[24:25] /*v[536:537]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x209
	v_pk_mul_f32 v[104:105], v[78:79] /*v[334:335]*/, v[32:33] /*v[544:545]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v103, v103
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[88:89], v[78:79] /*v[334:335]*/, v[88:89]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[94:95], v[94:95], v[96:97]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[96:97], v[78:79] /*v[334:335]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[104:105], v[104:105], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v88, v88, v89
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[94:95], v[78:79] /*v[334:335]*/, v[94:95]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[96:97], v[96:97], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[104:105], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v89, v94, v95
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[94:95], v[78:79] /*v[334:335]*/, v[26:27] /*v[538:539]*/
	s_set_vgpr_msb 0x900
	v_pk_mul_f32 v[96:97], v[96:97], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v104, v104
	v_exp_f32_e32 v105, v105
	s_set_vgpr_msb 64
	s_clause 0xb
	scratch_load_b128 v[194:197] /*v[450:453]*/, off, off offset:5344 th:TH_LOAD_LU nv
	scratch_load_b128 v[198:201] /*v[454:457]*/, off, off offset:5360 th:TH_LOAD_LU nv
	scratch_load_b128 v[88:91] /*v[344:347]*/, off, off offset:736 nv
	scratch_load_b128 v[92:95] /*v[348:351]*/, off, off offset:752 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[210:213] /*v[722:725]*/, off, off offset:1280 nv
	scratch_load_b128 v[214:217] /*v[726:729]*/, off, off offset:1296 nv
	scratch_load_b128 v[20:23] /*v[532:535]*/, off, off offset:1760 nv
	scratch_load_b128 v[24:27] /*v[536:539]*/, off, off offset:1776 nv
	s_set_vgpr_msb 0x8040
	scratch_load_b128 v[218:221] /*v[474:477]*/, off, off offset:1824 nv
	scratch_load_b128 v[222:225] /*v[478:481]*/, off, off offset:1840 nv
	s_set_vgpr_msb 0x4000
	v_pk_add_f32 v[94:95], v[94:95], v[110:111] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[110:111], v[34:35] /*v[546:547]*/, v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_pk_mul_f32 v[94:95], v[94:95], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[94:95], v[110:111], v[94:95]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[110:111], v[36:37] /*v[548:549]*/, v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x201
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[94:95], v[78:79] /*v[334:335]*/, v[94:95]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[96:97], v[110:111], v[96:97]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[110:111], v[78:79] /*v[334:335]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v94, v94, v95
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[96:97], v[78:79] /*v[334:335]*/, v[96:97]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[110:111], v[110:111], v[118:119] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v95, v96, v97
	s_set_vgpr_msb 2
	v_pk_add_f32 v[96:97], v[38:39] /*v[550:551]*/, v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_pk_mul_f32 v[110:111], v[110:111], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[96:97], v[96:97], v[102:103]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[102:103], v[40:41] /*v[552:553]*/, v[112:113] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x209
	v_pk_mul_f32 v[112:113], v[78:79] /*v[334:335]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v110, v110
	v_exp_f32_e32 v111, v111
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[96:97], v[78:79] /*v[334:335]*/, v[96:97]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[102:103], v[102:103], v[104:105]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[104:105], v[78:79] /*v[334:335]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[112:113], v[112:113], v[118:119] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v96, v96, v97
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[102:103], v[78:79] /*v[334:335]*/, v[102:103]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[104:105], v[104:105], v[118:119] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[112:113], v[112:113], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v97, v102, v103
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[102:103], v[78:79] /*v[334:335]*/, v[42:43] /*v[554:555]*/
	s_set_vgpr_msb 0x900
	v_pk_mul_f32 v[104:105], v[104:105], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v112, v112
	v_exp_f32_e32 v113, v113
	s_set_vgpr_msb 0xc0
	s_clause 0xc
	scratch_load_b128 v[122:125] /*v[890:893]*/, off, off offset:5376 th:TH_LOAD_LU nv
	scratch_load_b128 v[126:129] /*v[894:897]*/, off, off offset:5392 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[80:83] /*v[336:339]*/, off, off offset:704 nv
	scratch_load_b128 v[84:87] /*v[340:343]*/, off, off offset:720 nv
	scratch_load_b128 v[202:205] /*v[458:461]*/, off, off offset:1184 nv
	scratch_load_b128 v[206:209] /*v[462:465]*/, off, off offset:1200 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[162:165] /*v[930:933]*/, off, off offset:1920 nv
	scratch_load_b128 v[166:169] /*v[934:937]*/, off, off offset:1936 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[36:39] /*v[548:551]*/, off, off offset:2144 nv
	scratch_load_b128 v[40:43] /*v[552:555]*/, off, off offset:2160 nv
	s_set_vgpr_msb 0x8000
	v_pk_add_f32 v[102:103], v[102:103], v[118:119] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 14
	v_pk_add_f32 v[118:119], v[50:51] /*v[562:563]*/, v[170:171] /*v[938:939]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe00
	v_exp_f32_e32 v104, v104
	v_exp_f32_e32 v105, v105
	s_set_vgpr_msb 0x80
	s_clause 0xc
	scratch_load_b128 v[44:47] /*v[556:559]*/, off, off offset:5312 th:TH_LOAD_LU nv
	scratch_load_b128 v[48:51] /*v[560:563]*/, off, off offset:5328 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8040
	scratch_load_b128 v[54:57] /*v[310:313]*/, off, off offset:672 nv
	scratch_load_b128 v[58:61] /*v[314:317]*/, off, off offset:688 nv
	scratch_load_b128 v[186:189] /*v[442:445]*/, off, off offset:1152 nv
	scratch_load_b128 v[190:193] /*v[446:449]*/, off, off offset:1168 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[154:157] /*v[922:925]*/, off, off offset:1696 nv
	scratch_load_b128 v[158:161] /*v[926:929]*/, off, off offset:1712 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[28:31] /*v[540:543]*/, off, off offset:2400 nv
	scratch_load_b128 v[32:35] /*v[544:547]*/, off, off offset:2416 nv
	s_set_vgpr_msb 0x8000
	v_pk_mul_f32 v[102:103], v[102:103], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v103, v103
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[102:103], v[118:119], v[102:103]
	s_set_vgpr_msb 14
	v_pk_add_f32 v[118:119], v[52:53] /*v[564:565]*/, v[170:171] /*v[938:939]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe01
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[102:103], v[78:79] /*v[334:335]*/, v[102:103]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[104:105], v[118:119], v[104:105]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[118:119], v[78:79] /*v[334:335]*/, v[62:63] /*v[574:575]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v102, v102, v103
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[104:105], v[78:79] /*v[334:335]*/, v[104:105]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[118:119], v[118:119], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v103, v104, v105
	s_set_vgpr_msb 14
	v_pk_add_f32 v[104:105], v[54:55] /*v[566:567]*/, v[170:171] /*v[938:939]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe00
	v_pk_mul_f32 v[118:119], v[118:119], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[104:105], v[104:105], v[110:111]
	s_set_vgpr_msb 14
	v_pk_add_f32 v[110:111], v[56:57] /*v[568:569]*/, v[170:171] /*v[938:939]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe00
	v_exp_f32_e32 v118, v118
	v_exp_f32_e32 v119, v119
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[104:105], v[78:79] /*v[334:335]*/, v[104:105]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[110:111], v[110:111], v[112:113]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[112:113], v[78:79] /*v[334:335]*/, v[60:61] /*v[572:573]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v104, v104, v105
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[110:111], v[78:79] /*v[334:335]*/, v[110:111]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[112:113], v[112:113], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v105, v110, v111
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[110:111], v[78:79] /*v[334:335]*/, v[58:59] /*v[570:571]*/
	s_set_vgpr_msb 0x900
	v_pk_mul_f32 v[112:113], v[112:113], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_add_f32 v[110:111], v[110:111], v[126:127] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[126:127], v[66:67] /*v[578:579]*/, v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_exp_f32_e32 v112, v112
	v_exp_f32_e32 v113, v113
	v_pk_mul_f32 v[110:111], v[110:111], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v110, v110
	v_exp_f32_e32 v111, v111
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[110:111], v[126:127], v[110:111]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[126:127], v[68:69] /*v[580:581]*/, v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x201
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[110:111], v[78:79] /*v[334:335]*/, v[110:111]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[112:113], v[126:127], v[112:113]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[126:127], v[78:79] /*v[334:335]*/, v[78:79] /*v[590:591]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v110, v110, v111
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[112:113], v[78:79] /*v[334:335]*/, v[112:113]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[126:127], v[126:127], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v111, v112, v113
	s_set_vgpr_msb 2
	v_pk_add_f32 v[112:113], v[70:71] /*v[582:583]*/, v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_pk_mul_f32 v[126:127], v[126:127], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[112:113], v[112:113], v[118:119]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[118:119], v[72:73] /*v[584:585]*/, v[128:129] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x209
	v_pk_mul_f32 v[128:129], v[78:79] /*v[334:335]*/, v[80:81] /*v[592:593]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v126, v126
	v_exp_f32_e32 v127, v127
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[112:113], v[78:79] /*v[334:335]*/, v[112:113]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[118:119], v[118:119], v[120:121]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[120:121], v[78:79] /*v[334:335]*/, v[76:77] /*v[588:589]*/
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[128:129], v[128:129], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v112, v112, v113
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[118:119], v[78:79] /*v[334:335]*/, v[118:119]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[120:121], v[120:121], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[128:129], v[128:129], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v113, v118, v119
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[118:119], v[78:79] /*v[334:335]*/, v[74:75] /*v[586:587]*/
	s_set_vgpr_msb 0x900
	v_pk_mul_f32 v[120:121], v[120:121], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v128, v128
	v_exp_f32_e32 v129, v129
	s_set_vgpr_msb 64
	s_clause 0x7
	scratch_load_b128 v[114:117] /*v[370:373]*/, off, off offset:832 nv
	scratch_load_b128 v[118:121] /*v[374:377]*/, off, off offset:848 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[68:71] /*v[580:583]*/, off, off offset:1536 nv
	scratch_load_b128 v[72:75] /*v[584:587]*/, off, off offset:1552 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[52:55] /*v[820:823]*/, off, off offset:2848 nv
	scratch_load_b128 v[56:59] /*v[824:827]*/, off, off offset:2864 nv
	s_set_vgpr_msb 0xc000
	v_pk_add_f32 v[118:119], v[118:119], v[134:135] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[134:135], v[82:83] /*v[594:595]*/, v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_exp_f32_e32 v120, v120
	v_exp_f32_e32 v121, v121
	s_set_vgpr_msb 64
	s_clause 0x6
	scratch_load_b128 v[70:73] /*v[326:329]*/, off, off offset:896 nv
	scratch_load_b128 v[74:77] /*v[330:333]*/, off, off offset:912 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[76:79] /*v[588:591]*/, off, off offset:1344 nv
	scratch_load_b128 v[80:83] /*v[592:595]*/, off, off offset:1360 nv
	scratch_load_b128 v[250:253] /*v[762:765]*/, off, off offset:1504 nv
	scratch_load_b128 v[254:257] /*v[766:769]*/, off, off offset:1520 nv
	s_set_vgpr_msb 0x8000
	v_pk_mul_f32 v[118:119], v[118:119], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v118, v118
	v_exp_f32_e32 v119, v119
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[118:119], v[134:135], v[118:119]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[134:135], v[84:85] /*v[596:597]*/, v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x201
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[118:119], v[78:79] /*v[334:335]*/, v[118:119]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[120:121], v[134:135], v[120:121]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[134:135], v[78:79] /*v[334:335]*/, v[94:95] /*v[606:607]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v118, v118, v119
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[120:121], v[78:79] /*v[334:335]*/, v[120:121]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[134:135], v[134:135], v[214:215] /*v[982:983]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v119, v120, v121
	s_set_vgpr_msb 2
	v_pk_add_f32 v[120:121], v[86:87] /*v[598:599]*/, v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x200
	v_pk_mul_f32 v[134:135], v[134:135], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[120:121], v[120:121], v[126:127]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[126:127], v[88:89] /*v[600:601]*/, v[136:137] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x209
	v_pk_mul_f32 v[136:137], v[78:79] /*v[334:335]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v134, v134
	v_exp_f32_e32 v135, v135
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[120:121], v[78:79] /*v[334:335]*/, v[120:121]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[126:127], v[126:127], v[128:129]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[128:129], v[78:79] /*v[334:335]*/, v[92:93] /*v[604:605]*/
	s_set_vgpr_msb 0x90c
	v_pk_add_f32 v[136:137], v[136:137], v[214:215] /*v[982:983]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v120, v120, v121
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[126:127], v[78:79] /*v[334:335]*/, v[126:127]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[128:129], v[128:129], v[214:215] /*v[982:983]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[136:137], v[136:137], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v121, v126, v127
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[126:127], v[78:79] /*v[334:335]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x90c
	v_pk_mul_f32 v[128:129], v[128:129], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v136, v136
	v_exp_f32_e32 v137, v137
	v_pk_add_f32 v[126:127], v[126:127], v[214:215] /*v[982:983]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v128, v128
	v_exp_f32_e32 v129, v129
	v_pk_mul_f32 v[126:127], v[126:127], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v126, v126
	v_exp_f32_e32 v127, v127
	v_nop
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[126:127], v[142:143], v[126:127]
	s_set_vgpr_msb 6
	v_pk_add_f32 v[142:143], v[100:101] /*v[612:613]*/, v[104:105] /*v[360:361]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x601
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[126:127], v[78:79] /*v[334:335]*/, v[126:127]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[128:129], v[142:143], v[128:129]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[142:143], v[78:79] /*v[334:335]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v126, v126, v127
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[128:129], v[78:79] /*v[334:335]*/, v[128:129]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[142:143], v[142:143], v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v127, v128, v129
	s_set_vgpr_msb 6
	v_pk_add_f32 v[128:129], v[102:103] /*v[614:615]*/, v[104:105] /*v[360:361]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x600
	v_pk_mul_f32 v[142:143], v[142:143], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[128:129], v[128:129], v[134:135]
	s_set_vgpr_msb 6
	v_pk_add_f32 v[134:135], v[104:105] /*v[616:617]*/, v[104:105] /*v[360:361]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x600
	v_exp_f32_e32 v142, v142
	v_exp_f32_e32 v143, v143
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[128:129], v[78:79] /*v[334:335]*/, v[128:129]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[134:135], v[134:135], v[136:137]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[136:137], v[78:79] /*v[334:335]*/, v[108:109] /*v[620:621]*/
	s_set_vgpr_msb 0x940
	s_clause 0x35
	scratch_load_b128 v[130:133] /*v[386:389]*/, off, off offset:960 nv
	scratch_load_b128 v[134:137] /*v[390:393]*/, off, off offset:976 nv
	scratch_load_b128 v[242:245] /*v[498:501]*/, off, off offset:1472 nv
	scratch_load_b128 v[246:249] /*v[502:505]*/, off, off offset:1488 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[98:101] /*v[866:869]*/, off, off offset:2016 nv
	scratch_load_b128 v[102:105] /*v[870:873]*/, off, off offset:2032 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[10:13] /*v[522:525]*/, off, off offset:5472 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17] /*v[526:529]*/, off, off offset:5488 th:TH_LOAD_LU nv
	scratch_load_b128 v[108:111] /*v[620:623]*/, off, off offset:3008 nv
	scratch_load_b128 v[112:115] /*v[624:627]*/, off, off offset:3024 nv
	s_set_vgpr_msb 0x8000
	scratch_load_b128 v[214:217], off, off nv
	scratch_load_b128 v[218:221], off, off offset:16 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b128 v[26:29] /*v[794:797]*/, off, off offset:32 nv
	scratch_load_b128 v[30:33] /*v[798:801]*/, off, off offset:48 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b128 v[230:233], off, off offset:64 nv
	scratch_load_b128 v[234:237], off, off offset:80 nv
	scratch_load_b128 v[222:225], off, off offset:96 nv
	scratch_load_b128 v[226:229], off, off offset:112 nv
	scratch_load_b128 v[246:249], off, off offset:128 nv
	scratch_load_b128 v[250:253], off, off offset:144 nv
	scratch_load_b128 v[238:241], off, off offset:160 nv
	scratch_load_b128 v[242:245], off, off offset:176 nv
	scratch_load_b128 v[206:209], off, off offset:192 nv
	scratch_load_b128 v[210:213], off, off offset:208 nv
	scratch_load_b128 v[254:257], off, off offset:224 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[2:5] /*v[258:261]*/, off, off offset:240 nv
	scratch_load_b128 v[22:25] /*v[278:281]*/, off, off offset:256 nv
	scratch_load_b128 v[26:29] /*v[282:285]*/, off, off offset:272 nv
	scratch_load_b128 v[6:9] /*v[262:265]*/, off, off offset:320 nv
	scratch_load_b128 v[10:13] /*v[266:269]*/, off, off offset:336 nv
	scratch_load_b128 v[38:41] /*v[294:297]*/, off, off offset:384 nv
	scratch_load_b128 v[42:45] /*v[298:301]*/, off, off offset:400 nv
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:448 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:464 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[82:85] /*v[850:853]*/, off, off offset:992 nv
	scratch_load_b128 v[86:89] /*v[854:857]*/, off, off offset:1008 nv
	scratch_load_b128 v[10:13] /*v[778:781]*/, off, off offset:2944 nv
	scratch_load_b128 v[14:17] /*v[782:785]*/, off, off offset:2960 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[170:173] /*v[682:685]*/, off, off offset:5216 th:TH_LOAD_LU nv
	scratch_load_b128 v[174:177] /*v[686:689]*/, off, off offset:5232 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8040
	scratch_load_b128 v[46:49] /*v[302:305]*/, off, off offset:2816 nv
	scratch_load_b128 v[50:53] /*v[306:309]*/, off, off offset:2832 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[2:5] /*v[514:517]*/, off, off offset:2208 nv
	scratch_load_b128 v[6:9] /*v[518:521]*/, off, off offset:2224 nv
	s_set_vgpr_msb 0x8000
	v_cvt_pk_bf16_f32 v128, v128, v129
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[134:135], v[78:79] /*v[334:335]*/, v[134:135]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[136:137], v[136:137], v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v129, v134, v135
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[134:135], v[78:79] /*v[334:335]*/, v[106:107] /*v[618:619]*/
	s_set_vgpr_msb 0x900
	v_pk_mul_f32 v[136:137], v[136:137], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 64
	s_clause 0xe
	scratch_load_b128 v[122:125] /*v[378:381]*/, off, off offset:928 nv
	scratch_load_b128 v[126:129] /*v[382:385]*/, off, off offset:944 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[90:93] /*v[858:861]*/, off, off offset:1440 nv
	scratch_load_b128 v[94:97] /*v[862:865]*/, off, off offset:1456 nv
	scratch_load_b128 v[204:207] /*v[972:975]*/, off, off offset:1792 nv
	scratch_load_b128 v[208:211] /*v[976:979]*/, off, off offset:1808 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[100:103] /*v[612:615]*/, off, off offset:2176 nv
	scratch_load_b128 v[104:107] /*v[616:619]*/, off, off offset:2192 nv
	scratch_load_b128 v[84:87] /*v[596:599]*/, off, off offset:2048 nv
	scratch_load_b128 v[88:91] /*v[600:603]*/, off, off offset:2064 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[138:141] /*v[906:909]*/, off, off offset:2112 nv
	scratch_load_b128 v[142:145] /*v[910:913]*/, off, off offset:2128 nv
	s_set_vgpr_msb 0xc00c
	v_pk_add_f32 v[134:135], v[134:135], v[78:79] /*v[846:847]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v136, v136
	v_exp_f32_e32 v137, v137
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[134:135], v[134:135], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v134, v134
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v135, v135
	v_nop
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[134:135], v[146:147], v[134:135]
	s_set_vgpr_msb 14
	v_pk_add_f32 v[146:147], v[116:117] /*v[628:629]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe01
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[134:135], v[78:79] /*v[334:335]*/, v[134:135]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[136:137], v[146:147], v[136:137]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[146:147], v[78:79] /*v[334:335]*/, v[126:127] /*v[638:639]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v134, v134, v135
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[136:137], v[78:79] /*v[334:335]*/, v[136:137]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[146:147], v[146:147], v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v135, v136, v137
	s_set_vgpr_msb 14
	v_pk_add_f32 v[136:137], v[118:119] /*v[630:631]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe00
	v_pk_mul_f32 v[146:147], v[146:147], s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[136:137], v[136:137], v[142:143]
	s_set_vgpr_msb 14
	v_pk_add_f32 v[142:143], v[120:121] /*v[632:633]*/, v[76:77] /*v[844:845]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xe00
	v_exp_f32_e32 v146, v146
	v_exp_f32_e32 v147, v147
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[136:137], v[78:79] /*v[334:335]*/, v[136:137]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[142:143], v[142:143], v[144:145]
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[144:145], v[78:79] /*v[334:335]*/, v[124:125] /*v[636:637]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v136, v136, v137
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[142:143], v[78:79] /*v[334:335]*/, v[142:143]
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[144:145], v[144:145], v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v137, v142, v143
	s_set_vgpr_msb 9
	v_pk_mul_f32 v[142:143], v[78:79] /*v[334:335]*/, v[122:123] /*v[634:635]*/
	s_set_vgpr_msb 0x900
	v_pk_mul_f32 v[144:145], v[144:145], s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 64
	s_clause 0x2e
	scratch_load_b128 v[154:157] /*v[410:413]*/, off, off offset:1056 nv
	scratch_load_b128 v[158:161] /*v[414:417]*/, off, off offset:1072 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[122:125] /*v[634:637]*/, off, off offset:1568 nv
	scratch_load_b128 v[126:129] /*v[638:641]*/, off, off offset:1584 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[60:63] /*v[828:831]*/, off, off offset:2080 nv
	scratch_load_b128 v[64:67] /*v[832:835]*/, off, off offset:2096 nv
	scratch_load_b128 v[106:109] /*v[874:877]*/, off, off offset:2912 nv
	scratch_load_b128 v[110:113] /*v[878:881]*/, off, off offset:2928 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[52:55] /*v[564:567]*/, off, off offset:2272 nv
	scratch_load_b128 v[56:59] /*v[568:571]*/, off, off offset:2288 nv
	scratch_load_b128 v[162:165] /*v[674:677]*/, off, off offset:2304 nv
	scratch_load_b128 v[166:169] /*v[678:681]*/, off, off offset:2320 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[18:21] /*v[786:789]*/, off, off offset:2336 nv
	scratch_load_b128 v[22:25] /*v[790:793]*/, off, off offset:2352 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[154:157] /*v[666:669]*/, off, off offset:2368 nv
	scratch_load_b128 v[158:161] /*v[670:673]*/, off, off offset:2384 nv
	scratch_load_b128 v[194:197] /*v[706:709]*/, off, off offset:2432 nv
	scratch_load_b128 v[198:201] /*v[710:713]*/, off, off offset:2448 nv
	scratch_load_b128 v[202:205] /*v[714:717]*/, off, off offset:2464 nv
	scratch_load_b128 v[206:209] /*v[718:721]*/, off, off offset:2480 nv
	scratch_load_b128 v[138:141] /*v[650:653]*/, off, off offset:2496 nv
	scratch_load_b128 v[142:145] /*v[654:657]*/, off, off offset:2512 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[172:175] /*v[940:943]*/, off, off offset:2528 nv
	scratch_load_b128 v[176:179] /*v[944:947]*/, off, off offset:2544 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[60:63] /*v[572:575]*/, off, off offset:2560 nv
	scratch_load_b128 v[64:67] /*v[576:579]*/, off, off offset:2576 nv
	scratch_load_b128 v[92:95] /*v[604:607]*/, off, off offset:2592 nv
	scratch_load_b128 v[96:99] /*v[608:611]*/, off, off offset:2608 nv
	scratch_load_b128 v[186:189] /*v[698:701]*/, off, off offset:2624 nv
	scratch_load_b128 v[190:193] /*v[702:705]*/, off, off offset:2640 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[226:229] /*v[994:997]*/, off, off offset:2656 nv
	scratch_load_b128 v[230:233] /*v[998:1001]*/, off, off offset:2672 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[226:229] /*v[738:741]*/, off, off offset:2688 nv
	scratch_load_b128 v[230:233] /*v[742:745]*/, off, off offset:2704 nv
	scratch_load_b128 v[234:237] /*v[746:749]*/, off, off offset:2720 nv
	scratch_load_b128 v[238:241] /*v[750:753]*/, off, off offset:2736 nv
	scratch_load_b128 v[146:149] /*v[658:661]*/, off, off offset:2880 nv
	scratch_load_b128 v[150:153] /*v[662:665]*/, off, off offset:2896 nv
	s_set_vgpr_msb 0x800c
	v_pk_add_f32 v[142:143], v[142:143], v[80:81] /*v[848:849]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v144, v144
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[142:143], v[142:143], s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v142, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_exp_f32_e32 v143, v143
	v_nop
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[142:143], v[150:151], v[142:143]
	s_set_vgpr_msb 12
	v_pk_add_f32 v[150:151], v[200:201], v[216:217] /*v[984:985]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc01
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[142:143], v[78:79] /*v[334:335]*/, v[142:143]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[144:145], v[150:151], v[144:145]
	v_cvt_pk_bf16_f32 v4, v142, v143
	s_set_vgpr_msb 1
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[144:145], v[78:79] /*v[334:335]*/, v[144:145]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v5, v144, v145
	s_set_vgpr_msb 12
	v_pk_add_f32 v[144:145], v[202:203], v[216:217] /*v[984:985]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[144:145], v[144:145], v[146:147]
	s_set_vgpr_msb 12
	v_pk_add_f32 v[146:147], v[204:205], v[216:217] /*v[984:985]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_clause 0x1
	scratch_load_b128 v[198:201], off, off offset:480 nv
	scratch_load_b128 v[202:205], off, off offset:496 nv
	s_set_vgpr_msb 0xc01
	v_pk_mul_f32 v[144:145], v[78:79] /*v[334:335]*/, v[144:145]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[146:147], v[146:147], v[148:149]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v6, v144, v145
	s_set_vgpr_msb 1
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[146:147], v[78:79] /*v[334:335]*/, v[146:147]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cvt_pk_bf16_f32 v7, v146, v147
	v_mov_b64_e32 v[142:143], v[4:5]
	v_mov_b64_e32 v[140:141], v[2:3]
	v_mov_b64_e32 v[138:139], v[0:1]
	v_mov_b64_e32 v[144:145], v[6:7]
	s_clause 0x1
	scratch_load_b32 v0, off, off offset:8844 nv
	scratch_load_b32 v1, off, off offset:8864 nv
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v1, v1, v0
	ds_load_tr16_b128 v[146:149], v1
	ds_load_tr16_b128 v[150:153], v1 offset:4352
	scratch_load_b32 v0, off, off offset:8960 nv
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[198:205], v[90:97], v[146:153], v[198:205]
	s_clause 0x3
	scratch_store_b128 off, v[198:201], off offset:480 nv
	scratch_store_b128 off, v[202:205], off offset:496 nv
	scratch_load_b128 v[198:201], off, off offset:3488 nv
	scratch_load_b128 v[202:205], off, off offset:3504 nv
	s_set_vgpr_msb 0xf0
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[812:819]*/, v[18:25], v[146:153], v[44:51] /*v[812:819]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[874:881]*/, v[26:33], v[146:153], v[106:113] /*v[874:881]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[106:109] /*v[874:877]*/, off offset:2912 nv
	scratch_store_b128 off, v[110:113] /*v[878:881]*/, off offset:2928 nv
	s_set_vgpr_msb 0xca0
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[34:41], v[146:153], v[44:51] /*v[556:563]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[286:293]*/, v[42:49], v[146:153], v[30:37] /*v[286:293]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[30:33] /*v[286:289]*/, off offset:2976 nv
	scratch_store_b128 off, v[34:37] /*v[290:293]*/, off offset:2992 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[50:57], v[146:153], v[162:169] /*v[418:425]*/
	s_set_vgpr_msb 0x50f1
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[994:1001]*/, v[210:217] /*v[466:473]*/, v[146:153], v[226:233] /*v[994:1001]*/
	s_set_vgpr_msb 0xf10c
	s_clause 0x1
	scratch_store_b128 off, v[226:229] /*v[994:997]*/, off offset:2656 nv
	scratch_store_b128 off, v[230:233] /*v[998:1001]*/, off offset:2672 nv
	s_set_vgpr_msb 0xca0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[666:673]*/, v[66:73], v[146:153], v[154:161] /*v[666:673]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[154:157] /*v[666:669]*/, off offset:2368 nv
	scratch_store_b128 off, v[158:161] /*v[670:673]*/, off offset:2384 nv
	s_set_vgpr_msb 0x800
	v_wmma_f32_16x16x32_bf16 v[254:261], v[74:81], v[146:153], v[254:261]
	s_set_vgpr_msb 0x80
	s_clause 0x4
	scratch_load_b128 v[154:157] /*v[666:669]*/, off, off offset:4928 th:TH_LOAD_LU nv
	scratch_load_b128 v[158:161] /*v[670:673]*/, off, off offset:4944 th:TH_LOAD_LU nv
	scratch_store_b128 off, v[254:257], off offset:224 nv
	s_set_vgpr_msb 0x8004
	scratch_store_b128 off, v[2:5] /*v[258:261]*/, off offset:240 nv
	s_set_vgpr_msb 0x4a0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[82:89], v[146:153], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa0f0
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[828:835]*/, v[98:105], v[146:153], v[60:67] /*v[828:835]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[60:63] /*v[828:831]*/, off offset:2080 nv
	scratch_store_b128 off, v[64:67] /*v[832:835]*/, off offset:2096 nv
	s_set_vgpr_msb 0xcf0
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[948:955]*/, v[106:113], v[146:153], v[180:187] /*v[948:955]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[180:183] /*v[948:951]*/, off offset:1664 nv
	scratch_store_b128 off, v[184:187] /*v[952:955]*/, off offset:1680 nv
	s_set_vgpr_msb 0xcf0
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[914:921]*/, v[114:121], v[146:153], v[146:153] /*v[914:921]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[146:149] /*v[914:917]*/, off offset:1728 nv
	scratch_store_b128 off, v[150:153] /*v[918:921]*/, off offset:1744 nv
	s_set_vgpr_msb 0xc50
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[122:129], v[146:153], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[250:253] /*v[506:509]*/, off offset:1408 nv
	scratch_store_b128 off, v[254:257] /*v[510:513]*/, off offset:1424 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[130:137], v[146:153], v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[154:157] /*v[410:413]*/, off offset:1056 nv
	scratch_store_b128 off, v[158:161] /*v[414:417]*/, off offset:1072 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[344:351]*/, v[138:145], v[146:153], v[88:95] /*v[344:351]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x4
	ds_load_tr16_b128 v[146:149], v0
	ds_load_tr16_b128 v[150:153], v0 offset:4352
	s_wait_loadcnt_dscnt 0x200
	v_wmma_f32_16x16x32_bf16 v[198:205], v[18:25], v[146:153], v[198:205]
	s_clause 0x1
	scratch_store_b128 off, v[198:201], off offset:3488 nv
	scratch_store_b128 off, v[202:205], off offset:3504 nv
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[90:97], v[146:153], v[234:241] /*v[490:497]*/
	s_set_vgpr_msb 0x5044
	s_clause 0x6
	scratch_store_b128 off, v[234:237] /*v[490:493]*/, off offset:1888 nv
	scratch_store_b128 off, v[238:241] /*v[494:497]*/, off offset:1904 nv
	scratch_load_b128 v[234:237] /*v[490:493]*/, off, off offset:5440 th:TH_LOAD_LU nv
	scratch_load_b128 v[238:241] /*v[494:497]*/, off, off offset:5456 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4400
	scratch_load_b128 v[198:201], off, off offset:352 nv
	scratch_load_b128 v[202:205], off, off offset:368 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[198:205], v[98:105], v[146:153], v[198:205]
	s_clause 0x3
	scratch_store_b128 off, v[198:201], off offset:352 nv
	scratch_store_b128 off, v[202:205], off offset:368 nv
	scratch_load_b128 v[198:201], off, off offset:416 nv
	scratch_load_b128 v[202:205], off, off offset:432 nv
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[138:145], v[146:153], v[80:87] /*v[336:343]*/
	s_set_vgpr_msb 0x5044
	s_clause 0x9
	scratch_store_b128 off, v[88:91] /*v[344:347]*/, off offset:736 nv
	scratch_store_b128 off, v[92:95] /*v[348:351]*/, off offset:752 nv
	scratch_store_b128 off, v[80:83] /*v[336:339]*/, off offset:704 nv
	scratch_store_b128 off, v[84:87] /*v[340:343]*/, off offset:720 nv
	scratch_load_b128 v[86:89] /*v[342:345]*/, off, off offset:4704 th:TH_LOAD_LU nv
	scratch_load_b128 v[90:93] /*v[346:349]*/, off, off offset:4720 th:TH_LOAD_LU nv
	scratch_load_b128 v[78:81] /*v[334:337]*/, off, off offset:4672 th:TH_LOAD_LU nv
	scratch_load_b128 v[82:85] /*v[338:341]*/, off, off offset:4688 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4400
	scratch_load_b32 v0, off, off offset:8964 nv
	s_wait_loadcnt 0x5
	v_wmma_f32_16x16x32_bf16 v[198:205], v[114:121], v[146:153], v[198:205]
	s_clause 0x3
	scratch_store_b128 off, v[198:201], off offset:416 nv
	scratch_store_b128 off, v[202:205], off offset:432 nv
	scratch_load_b128 v[198:201], off, off offset:3520 nv
	scratch_load_b128 v[202:205], off, off offset:3536 nv
	s_set_vgpr_msb 0xf0
	v_wmma_f32_16x16x32_bf16 v[196:203] /*v[964:971]*/, v[26:33], v[146:153], v[196:203] /*v[964:971]*/
	s_set_vgpr_msb 0xf000
	v_wmma_f32_16x16x32_bf16 v[58:65], v[34:41], v[146:153], v[58:65]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xc0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mov_b64_e32 v[66:67] /*v[834:835]*/, v[64:65]
	s_set_vgpr_msb 0xc050
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[42:49], v[146:153], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x50c0
	v_mov_b64_e32 v[64:65] /*v[832:833]*/, v[62:63]
	v_mov_b64_e32 v[62:63] /*v[830:831]*/, v[60:61]
	v_mov_b64_e32 v[60:61] /*v[828:829]*/, v[58:59]
	s_set_vgpr_msb 0xc050
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[50:57], v[146:153], v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[698:705]*/, v[210:217] /*v[466:473]*/, v[146:153], v[186:193] /*v[698:705]*/
	s_set_vgpr_msb 0xa108
	s_clause 0x1
	scratch_store_b128 off, v[186:189] /*v[698:701]*/, off offset:2624 nv
	scratch_store_b128 off, v[190:193] /*v[702:705]*/, off offset:2640 nv
	s_set_vgpr_msb 0x8f0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[66:73], v[146:153], v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[18:21] /*v[786:789]*/, off offset:2336 nv
	scratch_store_b128 off, v[22:25] /*v[790:793]*/, off offset:2352 nv
	s_set_vgpr_msb 0xc00
	v_wmma_f32_16x16x32_bf16 v[206:213], v[74:81], v[146:153], v[206:213]
	s_clause 0x1
	scratch_store_b128 off, v[206:209], off offset:192 nv
	scratch_store_b128 off, v[210:213], off offset:208 nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[82:89], v[146:153], v[108:115] /*v[620:627]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[108:111] /*v[620:623]*/, off offset:3008 nv
	scratch_store_b128 off, v[112:115] /*v[624:627]*/, off offset:3024 nv
	s_set_vgpr_msb 0x8f0
	v_wmma_f32_16x16x32_bf16 v[68:75] /*v[836:843]*/, v[106:113], v[146:153], v[68:75] /*v[836:843]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[68:71] /*v[836:839]*/, off offset:1600 nv
	scratch_store_b128 off, v[72:75] /*v[840:843]*/, off offset:1616 nv
	s_set_vgpr_msb 0xca0
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[722:729]*/, v[122:129], v[146:153], v[210:217] /*v[722:729]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[210:213] /*v[722:725]*/, off offset:1280 nv
	scratch_store_b128 off, v[214:217] /*v[726:729]*/, off offset:1296 nv
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[130:137], v[146:153], v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[146:149], v0
	ds_load_tr16_b128 v[150:153], v0 offset:4352
	s_set_vgpr_msb 0x80
	s_clause 0x5
	scratch_load_b128 v[108:111] /*v[620:623]*/, off, off offset:4640 th:TH_LOAD_LU nv
	scratch_load_b128 v[112:115] /*v[624:627]*/, off, off offset:4656 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8004
	scratch_load_b32 v0, off, off offset:8968 nv
	scratch_store_b128 off, v[130:133] /*v[386:389]*/, off offset:960 nv
	scratch_store_b128 off, v[134:137] /*v[390:393]*/, off offset:976 nv
	s_set_vgpr_msb 0x400
	s_wait_loadcnt_dscnt 0x300
	v_wmma_f32_16x16x32_bf16 v[198:205], v[18:25], v[146:153], v[198:205]
	s_clause 0x3
	scratch_store_b128 off, v[198:201], off offset:3520 nv
	scratch_store_b128 off, v[202:205], off offset:3536 nv
	scratch_load_b128 v[198:201], off, off offset:3456 nv
	scratch_load_b128 v[202:205], off, off offset:3472 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[198:205], v[26:33], v[146:153], v[198:205]
	s_clause 0x3
	scratch_store_b128 off, v[198:201], off offset:3456 nv
	scratch_store_b128 off, v[202:205], off offset:3472 nv
	scratch_load_b128 v[198:201], off, off offset:288 nv
	scratch_load_b128 v[202:205], off, off offset:304 nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[532:539]*/, v[106:113], v[146:153], v[20:27] /*v[532:539]*/
	s_set_vgpr_msb 0xa048
	s_clause 0x3
	scratch_store_b128 off, v[20:23] /*v[532:535]*/, off offset:1760 nv
	scratch_store_b128 off, v[24:27] /*v[536:539]*/, off offset:1776 nv
	scratch_load_b128 v[130:133] /*v[386:389]*/, off, off offset:4864 th:TH_LOAD_LU nv
	scratch_load_b128 v[134:137] /*v[390:393]*/, off, off offset:4880 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x48a0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[666:673]*/, v[34:41], v[146:153], v[154:161] /*v[666:673]*/
	s_clause 0x1
	scratch_load_b128 v[20:23] /*v[532:535]*/, off, off offset:5152 th:TH_LOAD_LU nv
	scratch_load_b128 v[24:27] /*v[536:539]*/, off, off offset:5168 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xa0f0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[890:897]*/, v[42:49], v[146:153], v[122:129] /*v[890:897]*/
	s_set_vgpr_msb 0xf050
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[50:57], v[146:153], v[86:93] /*v[342:349]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[92:99] /*v[604:611]*/, v[210:217] /*v[466:473]*/, v[146:153], v[92:99] /*v[604:611]*/
	s_set_vgpr_msb 0xa108
	s_clause 0x1
	scratch_store_b128 off, v[92:95] /*v[604:607]*/, off offset:2592 nv
	scratch_store_b128 off, v[96:99] /*v[608:611]*/, off offset:2608 nv
	s_set_vgpr_msb 0x8a0
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[674:681]*/, v[66:73], v[146:153], v[162:169] /*v[674:681]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[162:165] /*v[674:677]*/, off offset:2304 nv
	scratch_store_b128 off, v[166:169] /*v[678:681]*/, off offset:2320 nv
	s_set_vgpr_msb 0x800
	v_wmma_f32_16x16x32_bf16 v[238:245], v[74:81], v[146:153], v[238:245]
	s_clause 0x1
	scratch_store_b128 off, v[238:241], off offset:160 nv
	scratch_store_b128 off, v[242:245], off offset:176 nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[82:89], v[146:153], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[2:5] /*v[514:517]*/, off offset:2208 nv
	scratch_store_b128 off, v[6:9] /*v[518:521]*/, off offset:2224 nv
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[90:97], v[146:153], v[226:233] /*v[482:489]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[226:229] /*v[482:485]*/, off offset:1856 nv
	scratch_store_b128 off, v[230:233] /*v[486:489]*/, off offset:1872 nv
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[198:205], v[98:105], v[146:153], v[198:205]
	s_clause 0x1
	scratch_store_b128 off, v[198:201], off offset:288 nv
	scratch_store_b128 off, v[202:205], off offset:304 nv
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[114:121], v[146:153], v[242:249] /*v[498:505]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[242:245] /*v[498:501]*/, off offset:1472 nv
	scratch_store_b128 off, v[246:249] /*v[502:505]*/, off offset:1488 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[122:129], v[146:153], v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[202:205] /*v[458:461]*/, off offset:1184 nv
	scratch_store_b128 off, v[206:209] /*v[462:465]*/, off offset:1200 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[130:137], v[146:153], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[122:125] /*v[378:381]*/, off offset:928 nv
	scratch_store_b128 off, v[126:129] /*v[382:385]*/, off offset:944 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[310:317]*/, v[138:145], v[146:153], v[54:61] /*v[310:317]*/
	s_set_vgpr_msb 0x5000
	ds_load_tr16_b128 v[146:149], v0
	ds_load_tr16_b128 v[150:153], v0 offset:4352
	s_clause 0x5
	scratch_load_b32 v0, off, off offset:8972 nv
	s_set_vgpr_msb 0x44
	scratch_load_b128 v[122:125] /*v[378:381]*/, off, off offset:4800 th:TH_LOAD_LU nv
	scratch_load_b128 v[126:129] /*v[382:385]*/, off, off offset:4816 th:TH_LOAD_LU nv
	scratch_store_b128 off, v[54:57] /*v[310:313]*/, off offset:672 nv
	scratch_store_b128 off, v[58:61] /*v[314:317]*/, off offset:688 nv
	s_set_vgpr_msb 0x4400
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[186:193], v[138:145], v[146:153], v[186:193]
	s_clause 0x1
	scratch_store_b128 off, v[186:189], off offset:640 nv
	scratch_store_b128 off, v[190:193], off offset:656 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[532:539]*/, v[18:25], v[146:153], v[20:27] /*v[532:539]*/
	s_set_vgpr_msb 0xa000
	s_clause 0x1
	scratch_load_b128 v[186:189], off, off offset:3552 nv
	scratch_load_b128 v[190:193], off, off offset:3568 nv
	s_set_vgpr_msb 0xf0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[26:33], v[146:153], v[10:17] /*v[778:785]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[10:13] /*v[778:781]*/, off offset:2944 nv
	scratch_store_b128 off, v[14:17] /*v[782:785]*/, off offset:2960 nv
	s_set_vgpr_msb 0xc50
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[34:41], v[146:153], v[130:137] /*v[386:393]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[42:49], v[146:153], v[234:241] /*v[490:497]*/
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[50:57], v[146:153], v[78:85] /*v[334:341]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[210:217] /*v[466:473]*/, v[146:153], v[60:67] /*v[572:579]*/
	s_set_vgpr_msb 0xa108
	s_clause 0x1
	scratch_store_b128 off, v[60:63] /*v[572:575]*/, off offset:2560 nv
	scratch_store_b128 off, v[64:67] /*v[576:579]*/, off offset:2576 nv
	s_set_vgpr_msb 0x8a0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[66:73], v[146:153], v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[52:55] /*v[564:567]*/, off offset:2272 nv
	scratch_store_b128 off, v[56:59] /*v[568:571]*/, off offset:2288 nv
	s_set_vgpr_msb 0x800
	v_wmma_f32_16x16x32_bf16 v[246:253], v[74:81], v[146:153], v[246:253]
	s_set_vgpr_msb 0x80
	s_clause 0x3
	scratch_load_b128 v[52:55] /*v[564:567]*/, off, off offset:5056 th:TH_LOAD_LU nv
	scratch_load_b128 v[56:59] /*v[568:571]*/, off, off offset:5072 th:TH_LOAD_LU nv
	scratch_store_b128 off, v[246:249], off offset:128 nv
	scratch_store_b128 off, v[250:253], off offset:144 nv
	s_set_vgpr_msb 0x80f0
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[906:913]*/, v[82:89], v[146:153], v[138:145] /*v[906:913]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x3
	scratch_load_b128 v[246:249], off, off offset:5584 nv
	scratch_load_b128 v[250:253], off, off offset:5600 nv
	scratch_store_b128 off, v[138:141] /*v[906:909]*/, off offset:2112 nv
	scratch_store_b128 off, v[142:145] /*v[910:913]*/, off offset:2128 nv
	s_set_vgpr_msb 0xc50
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[90:97], v[146:153], v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[218:221] /*v[474:477]*/, off offset:1824 nv
	scratch_store_b128 off, v[222:225] /*v[478:481]*/, off offset:1840 nv
	s_set_vgpr_msb 0x4f0
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[866:873]*/, v[98:105], v[146:153], v[98:105] /*v[866:873]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[98:101] /*v[866:869]*/, off offset:2016 nv
	scratch_store_b128 off, v[102:105] /*v[870:873]*/, off offset:2032 nv
	s_set_vgpr_msb 0xcf0
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[930:937]*/, v[106:113], v[146:153], v[162:169] /*v[930:937]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[162:165] /*v[930:933]*/, off offset:1920 nv
	scratch_store_b128 off, v[166:169] /*v[934:937]*/, off offset:1936 nv
	s_set_vgpr_msb 0xcf0
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[858:865]*/, v[114:121], v[146:153], v[90:97] /*v[858:865]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[90:93] /*v[858:861]*/, off offset:1440 nv
	scratch_store_b128 off, v[94:97] /*v[862:865]*/, off offset:1456 nv
	s_set_vgpr_msb 0xc50
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[122:129], v[146:153], v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[186:189] /*v[442:445]*/, off offset:1152 nv
	scratch_store_b128 off, v[190:193] /*v[446:449]*/, off offset:1168 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[130:137], v[146:153], v[70:77] /*v[326:333]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x8
	ds_load_tr16_b128 v[146:149], v0
	ds_load_tr16_b128 v[150:153], v0 offset:4352
	scratch_load_b32 v0, off, off offset:8976 nv
	s_set_vgpr_msb 0xf0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[972:979]*/, v[98:105], v[146:153], v[204:211] /*v[972:979]*/
	s_set_vgpr_msb 0xf004
	s_clause 0x4
	scratch_store_b128 off, v[70:73] /*v[326:329]*/, off offset:896 nv
	scratch_store_b128 off, v[74:77] /*v[330:333]*/, off offset:912 nv
	s_set_vgpr_msb 0x40c
	scratch_store_b128 off, v[204:207] /*v[972:975]*/, off offset:1792 nv
	scratch_store_b128 off, v[208:211] /*v[976:979]*/, off offset:1808 nv
	s_set_vgpr_msb 0xc50
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[122:129], v[146:153], v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5044
	s_clause 0x3
	scratch_store_b128 off, v[178:181] /*v[434:437]*/, off offset:1120 nv
	scratch_store_b128 off, v[182:185] /*v[438:441]*/, off offset:1136 nv
	scratch_load_b128 v[70:73] /*v[326:329]*/, off, off offset:4768 th:TH_LOAD_LU nv
	scratch_load_b128 v[74:77] /*v[330:333]*/, off, off offset:4784 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4400
	s_wait_loadcnt 0x7
	v_wmma_f32_16x16x32_bf16 v[186:193], v[18:25], v[146:153], v[186:193]
	s_set_vgpr_msb 0xc0
	s_clause 0x8
	scratch_load_b128 v[204:207] /*v[972:975]*/, off, off offset:5184 th:TH_LOAD_LU nv
	scratch_load_b128 v[208:211] /*v[976:979]*/, off, off offset:5200 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc050
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:4960 th:TH_LOAD_LU nv
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:4976 th:TH_LOAD_LU nv
	scratch_load_b128 v[186:189] /*v[442:445]*/, off, off offset:5024 th:TH_LOAD_LU nv
	scratch_load_b128 v[190:193] /*v[446:449]*/, off, off offset:5040 th:TH_LOAD_LU nv
	scratch_store_b128 off, v[186:189], off offset:3552 nv
	scratch_store_b128 off, v[190:193], off offset:3568 nv
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[302:309]*/, v[26:33], v[146:153], v[46:53] /*v[302:309]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[46:49] /*v[302:305]*/, off offset:2816 nv
	scratch_store_b128 off, v[50:53] /*v[306:309]*/, off offset:2832 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[34:41], v[146:153], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x50f0
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[1002:1009]*/, v[42:49], v[146:153], v[234:241] /*v[1002:1009]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[234:237] /*v[1002:1005]*/, off offset:2752 nv
	scratch_store_b128 off, v[238:241] /*v[1006:1009]*/, off offset:2768 nv
	s_set_vgpr_msb 0xca0
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[50:57], v[146:153], v[108:115] /*v[620:627]*/
	s_set_vgpr_msb 0xa0f1
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[940:947]*/, v[210:217] /*v[466:473]*/, v[146:153], v[172:179] /*v[940:947]*/
	s_set_vgpr_msb 0xf10c
	s_clause 0x1
	scratch_store_b128 off, v[172:175] /*v[940:943]*/, off offset:2528 nv
	scratch_store_b128 off, v[176:179] /*v[944:947]*/, off offset:2544 nv
	s_set_vgpr_msb 0xc50
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[270:277]*/, v[66:73], v[146:153], v[14:21] /*v[270:277]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[14:17] /*v[270:273]*/, off offset:448 nv
	scratch_store_b128 off, v[18:21] /*v[274:277]*/, off offset:464 nv
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[222:229], v[74:81], v[146:153], v[222:229]
	s_set_vgpr_msb 64
	s_clause 0x3
	scratch_load_b128 v[14:17] /*v[270:273]*/, off, off offset:3584 nv
	scratch_load_b128 v[18:21] /*v[274:277]*/, off, off offset:3600 nv
	scratch_store_b128 off, v[222:225], off offset:96 nv
	scratch_store_b128 off, v[226:229], off offset:112 nv
	s_set_vgpr_msb 0x40f0
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[956:963]*/, v[82:89], v[146:153], v[188:195] /*v[956:963]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[188:191] /*v[956:959]*/, off offset:2240 nv
	scratch_store_b128 off, v[192:195] /*v[960:963]*/, off offset:2256 nv
	s_set_vgpr_msb 0xca0
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[90:97], v[146:153], v[36:43] /*v[548:555]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[36:39] /*v[548:551]*/, off offset:2144 nv
	scratch_store_b128 off, v[40:43] /*v[552:555]*/, off offset:2160 nv
	s_set_vgpr_msb 0x8f0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[922:929]*/, v[106:113], v[146:153], v[154:161] /*v[922:929]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[154:157] /*v[922:925]*/, off offset:1696 nv
	scratch_store_b128 off, v[158:161] /*v[926:929]*/, off offset:1712 nv
	s_set_vgpr_msb 0xca0
	v_wmma_f32_16x16x32_bf16 v[76:83] /*v[588:595]*/, v[114:121], v[146:153], v[76:83] /*v[588:595]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[76:79] /*v[588:591]*/, off offset:1344 nv
	scratch_store_b128 off, v[80:83] /*v[592:595]*/, off offset:1360 nv
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[318:325]*/, v[130:137], v[146:153], v[62:69] /*v[318:325]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[62:65] /*v[318:321]*/, off offset:864 nv
	scratch_store_b128 off, v[66:69] /*v[322:325]*/, off offset:880 nv
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[178:185], v[138:145], v[146:153], v[178:185]
	s_wait_loadcnt 0xa
	ds_load_tr16_b128 v[146:149], v0
	ds_load_tr16_b128 v[150:153], v0 offset:4352
	scratch_load_b32 v0, off, off offset:8952 nv
	s_set_vgpr_msb 0x50
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[130:137], v[146:153], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5084
	s_clause 0x6
	scratch_store_b128 off, v[114:117] /*v[370:373]*/, off offset:832 nv
	scratch_store_b128 off, v[118:121] /*v[374:377]*/, off offset:848 nv
	scratch_load_b128 v[76:79] /*v[588:591]*/, off, off offset:5120 th:TH_LOAD_LU nv
	scratch_load_b128 v[80:83] /*v[592:595]*/, off, off offset:5136 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8440
	scratch_load_b128 v[62:65] /*v[318:321]*/, off, off offset:4736 th:TH_LOAD_LU nv
	scratch_load_b128 v[66:69] /*v[322:325]*/, off, off offset:4752 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x40f0
	s_wait_loadcnt 0xb
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[972:979]*/, v[18:25], v[146:153], v[204:211] /*v[972:979]*/
	s_set_vgpr_msb 0xf040
	s_clause 0x3
	scratch_load_b128 v[114:117] /*v[370:373]*/, off, off offset:4896 th:TH_LOAD_LU nv
	scratch_load_b128 v[118:121] /*v[374:377]*/, off, off offset:4912 th:TH_LOAD_LU nv
	scratch_store_b128 off, v[178:181], off offset:608 nv
	scratch_store_b128 off, v[182:185], off offset:624 nv
	s_set_vgpr_msb 0x40a0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[26:33], v[146:153], v[170:177] /*v[682:689]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[34:41], v[146:153], v[70:77] /*v[326:333]*/
	s_wait_loadcnt 0xb
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[42:49], v[146:153], v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x50f0
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[850:857]*/, v[50:57], v[146:153], v[82:89] /*v[850:857]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[82:85] /*v[850:853]*/, off offset:992 nv
	scratch_store_b128 off, v[86:89] /*v[854:857]*/, off offset:1008 nv
	s_set_vgpr_msb 0xca1
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[650:657]*/, v[210:217] /*v[466:473]*/, v[146:153], v[138:145] /*v[650:657]*/
	s_set_vgpr_msb 0xa108
	s_clause 0x1
	scratch_store_b128 off, v[138:141] /*v[650:653]*/, off offset:2496 nv
	scratch_store_b128 off, v[142:145] /*v[654:657]*/, off offset:2512 nv
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[294:301]*/, v[66:73], v[146:153], v[38:45] /*v[294:301]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[38:41] /*v[294:297]*/, off offset:384 nv
	scratch_store_b128 off, v[42:45] /*v[298:301]*/, off offset:400 nv
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[230:237], v[74:81], v[146:153], v[230:237]
	s_clause 0x1
	scratch_store_b128 off, v[230:233], off offset:64 nv
	scratch_store_b128 off, v[234:237], off offset:80 nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[100:107] /*v[612:619]*/, v[82:89], v[146:153], v[100:107] /*v[612:619]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[100:103] /*v[612:615]*/, off offset:2176 nv
	scratch_store_b128 off, v[104:107] /*v[616:619]*/, off offset:2192 nv
	s_set_vgpr_msb 0x8a0
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[90:97], v[146:153], v[28:35] /*v[540:547]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[28:31] /*v[540:543]*/, off offset:2400 nv
	scratch_store_b128 off, v[32:35] /*v[544:547]*/, off offset:2416 nv
	s_set_vgpr_msb 0x8a0
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[98:105], v[146:153], v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[250:253] /*v[762:765]*/, off offset:1504 nv
	scratch_store_b128 off, v[254:257] /*v[766:769]*/, off offset:1520 nv
	s_set_vgpr_msb 0x8a0
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[106:113], v[146:153], v[178:185] /*v[690:697]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[178:181] /*v[690:693]*/, off offset:1248 nv
	scratch_store_b128 off, v[182:185] /*v[694:697]*/, off offset:1264 nv
	s_set_vgpr_msb 0x8f0
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[804:811]*/, v[114:121], v[146:153], v[36:43] /*v[804:811]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[36:39] /*v[804:807]*/, off offset:1312 nv
	scratch_store_b128 off, v[40:43] /*v[808:811]*/, off offset:1328 nv
	s_set_vgpr_msb 0xc50
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[122:129], v[146:153], v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[138:141] /*v[394:397]*/, off offset:1024 nv
	scratch_store_b128 off, v[142:145] /*v[398:401]*/, off offset:1040 nv
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[170:177], v[138:145], v[146:153], v[170:177]
	s_wait_loadcnt 0x6
	ds_load_tr16_b128 v[146:149], v0
	ds_load_tr16_b128 v[150:153], v0 offset:4352
	scratch_load_b32 v0, off, off offset:8956 nv
	s_set_vgpr_msb 0xa0
	s_wait_loadcnt_dscnt 0x500
	v_wmma_f32_16x16x32_bf16 v[76:83] /*v[588:595]*/, v[18:25], v[146:153], v[76:83] /*v[588:595]*/
	s_set_vgpr_msb 0xa050
	s_clause 0x3
	scratch_load_b128 v[138:141] /*v[394:397]*/, off, off offset:4992 th:TH_LOAD_LU nv
	scratch_load_b128 v[142:145] /*v[398:401]*/, off, off offset:5008 th:TH_LOAD_LU nv
	scratch_store_b128 off, v[170:173], off offset:544 nv
	scratch_store_b128 off, v[174:177], off offset:560 nv
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[26:33], v[146:153], v[186:193] /*v[442:449]*/
	s_wait_loadcnt 0x5
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[318:325]*/, v[34:41], v[146:153], v[62:69] /*v[318:325]*/
	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[42:49], v[146:153], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[50:57], v[146:153], v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[234:237] /*v[746:749]*/, off offset:2720 nv
	scratch_store_b128 off, v[238:241] /*v[750:753]*/, off offset:2736 nv
	s_set_vgpr_msb 0x8a1
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[714:721]*/, v[210:217] /*v[466:473]*/, v[146:153], v[202:209] /*v[714:721]*/
	s_set_vgpr_msb 0xa108
	s_clause 0x1
	scratch_store_b128 off, v[202:205] /*v[714:717]*/, off offset:2464 nv
	scratch_store_b128 off, v[206:209] /*v[718:721]*/, off offset:2480 nv
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[262:269]*/, v[66:73], v[146:153], v[6:13] /*v[262:269]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[6:9] /*v[262:265]*/, off offset:320 nv
	scratch_store_b128 off, v[10:13] /*v[266:269]*/, off offset:336 nv
	s_set_vgpr_msb 0x4f0
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[794:801]*/, v[74:81], v[146:153], v[26:33] /*v[794:801]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[26:29] /*v[794:797]*/, off offset:32 nv
	scratch_store_b128 off, v[30:33] /*v[798:801]*/, off offset:48 nv
	s_set_vgpr_msb 0xca0
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[82:89], v[146:153], v[84:91] /*v[596:603]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[84:87] /*v[596:599]*/, off offset:2048 nv
	scratch_store_b128 off, v[88:91] /*v[600:603]*/, off offset:2064 nv
	s_set_vgpr_msb 0x8f0
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[986:993]*/, v[90:97], v[146:153], v[218:225] /*v[986:993]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[218:221] /*v[986:989]*/, off offset:1984 nv
	scratch_store_b128 off, v[222:225] /*v[990:993]*/, off offset:2000 nv
	s_set_vgpr_msb 0xc00
	v_wmma_f32_16x16x32_bf16 v[10:17], v[98:105], v[146:153], v[10:17]
	s_clause 0x1
	scratch_store_b128 off, v[10:13], off offset:1952 nv
	scratch_store_b128 off, v[14:17], off offset:1968 nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[106:113], v[146:153], v[130:137] /*v[642:649]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[130:133] /*v[642:645]*/, off offset:1216 nv
	scratch_store_b128 off, v[134:137] /*v[646:649]*/, off offset:1232 nv
	s_set_vgpr_msb 0x8a0
	v_wmma_f32_16x16x32_bf16 v[68:75] /*v[580:587]*/, v[114:121], v[146:153], v[68:75] /*v[580:587]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[68:71] /*v[580:583]*/, off offset:1536 nv
	scratch_store_b128 off, v[72:75] /*v[584:587]*/, off offset:1552 nv
	s_set_vgpr_msb 0x800
	v_wmma_f32_16x16x32_bf16 v[162:169], v[122:129], v[146:153], v[162:169]
	s_set_vgpr_msb 0x80
	s_clause 0x3
	scratch_load_b128 v[68:71] /*v[580:583]*/, off, off offset:5552 nv
	scratch_load_b128 v[72:75] /*v[584:587]*/, off, off offset:5568 nv
	scratch_store_b128 off, v[162:165], off offset:576 nv
	scratch_store_b128 off, v[166:169], off offset:592 nv
	s_set_vgpr_msb 0x8050
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[130:137], v[146:153], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[106:109] /*v[362:365]*/, off offset:800 nv
	scratch_store_b128 off, v[110:113] /*v[366:369]*/, off offset:816 nv
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145], v[146:153], v[154:161]
	s_wait_loadcnt 0x4
	ds_load_tr16_b128 v[146:149], v0
	ds_load_tr16_b128 v[150:153], v0 offset:4352
	scratch_load_b128 v[0:3], off, off offset:8804 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xf0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[138:145], v[146:153], v[2:9] /*v[770:777]*/
	s_clause 0x9
	scratch_store_b128 off, v[154:157], off offset:512 nv
	scratch_store_b128 off, v[158:161], off offset:528 nv
	s_set_vgpr_msb 0xf040
	scratch_load_b128 v[106:109] /*v[362:365]*/, off, off offset:4832 th:TH_LOAD_LU nv
	scratch_load_b128 v[110:113] /*v[366:369]*/, off, off offset:4848 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x400c
	scratch_load_b128 v[154:157], off, off offset:3424 nv
	scratch_load_b128 v[158:161], off, off offset:3440 nv
	scratch_store_b128 off, v[2:5] /*v[770:773]*/, off offset:1632 nv
	scratch_store_b128 off, v[6:9] /*v[774:777]*/, off offset:1648 nv
	s_set_vgpr_msb 0xc00
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[18:25], v[146:153], v[154:161]
	v_mov_b64_e32 v[14:15], v[0:1]
	v_mov_b64_e32 v[16:17], v[2:3]
	s_clause 0x2
	scratch_load_b128 v[0:3], off, off offset:8788 th:TH_LOAD_LU nv
	scratch_store_b128 off, v[154:157], off offset:3424 nv
	scratch_store_b128 off, v[158:161], off offset:3440 nv
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[26:33], v[146:153], v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[10:11], v[0:1]
	v_mov_b64_e32 v[12:13], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8772 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[34:41], v[146:153], v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa000
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[142:143], v[0:1]
	v_mov_b64_e32 v[144:145], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8756 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[42:49], v[146:153], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[138:139], v[0:1]
	v_mov_b64_e32 v[140:141], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8740 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[50:57], v[146:153], v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[226:229] /*v[738:741]*/, off offset:2688 nv
	scratch_store_b128 off, v[230:233] /*v[742:745]*/, off offset:2704 nv
	s_set_vgpr_msb 0x8a1
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[706:713]*/, v[210:217] /*v[466:473]*/, v[146:153], v[194:201] /*v[706:713]*/
	s_set_vgpr_msb 0xa108
	s_clause 0x1
	scratch_store_b128 off, v[194:197] /*v[706:709]*/, off offset:2432 nv
	scratch_store_b128 off, v[198:201] /*v[710:713]*/, off offset:2448 nv
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[66:73], v[146:153], v[22:29] /*v[278:285]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[22:25] /*v[278:281]*/, off offset:256 nv
	scratch_store_b128 off, v[26:29] /*v[282:285]*/, off offset:272 nv
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[214:221], v[74:81], v[146:153], v[214:221]
	s_clause 0x1
	scratch_store_b128 off, v[214:217], off nv
	scratch_store_b128 off, v[218:221], off offset:16 nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[82:89], v[146:153], v[146:153] /*v[658:665]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[146:149] /*v[658:661]*/, off offset:2880 nv
	scratch_store_b128 off, v[150:153] /*v[662:665]*/, off offset:2896 nv
	s_set_vgpr_msb 0x8a0
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[90:97], v[146:153], v[218:225] /*v[730:737]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[218:221] /*v[730:733]*/, off offset:2784 nv
	scratch_store_b128 off, v[222:225] /*v[734:737]*/, off offset:2800 nv
	s_set_vgpr_msb 0x8f0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[820:827]*/, v[98:105], v[146:153], v[52:59] /*v[820:827]*/
	s_set_vgpr_msb 0xf00c
	s_clause 0x1
	scratch_store_b128 off, v[52:55] /*v[820:823]*/, off offset:2848 nv
	scratch_store_b128 off, v[56:59] /*v[824:827]*/, off offset:2864 nv
	s_set_vgpr_msb 0xca0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[106:113], v[146:153], v[122:129] /*v[634:641]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[122:125] /*v[634:637]*/, off offset:1568 nv
	scratch_store_b128 off, v[126:129] /*v[638:641]*/, off offset:1584 nv
	s_set_vgpr_msb 0x8a0
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[114:121], v[146:153], v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa008
	s_clause 0x1
	scratch_store_b128 off, v[242:245] /*v[754:757]*/, off offset:1376 nv
	scratch_store_b128 off, v[246:249] /*v[758:761]*/, off offset:1392 nv
	s_set_vgpr_msb 0x850
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[122:129], v[146:153], v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:1088 nv
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:1104 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[130:137], v[146:153], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[96:99] /*v[352:355]*/, off offset:768 nv
	scratch_store_b128 off, v[100:103] /*v[356:359]*/, off offset:784 nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[210:211], v[0:1]
	v_mov_b64_e32 v[212:213], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8724 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[206:207], v[0:1]
	v_mov_b64_e32 v[208:209], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8708 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[150:151], v[0:1]
	v_mov_b64_e32 v[152:153], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8692 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[146:147], v[0:1]
	v_mov_b64_e32 v[148:149], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8676 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[174:175], v[0:1]
	v_mov_b64_e32 v[176:177], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8660 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[170:171], v[0:1]
	v_mov_b64_e32 v[172:173], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8644 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[182:183], v[0:1]
	v_mov_b64_e32 v[184:185], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8628 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[178:179], v[0:1]
	v_mov_b64_e32 v[180:181], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8612 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[190:191], v[0:1]
	v_mov_b64_e32 v[192:193], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8596 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[186:187], v[0:1]
	v_mov_b64_e32 v[188:189], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8580 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[202:203], v[0:1]
	v_mov_b64_e32 v[204:205], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8564 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[198:199], v[0:1]
	v_mov_b64_e32 v[200:201], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8548 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[22:23], v[0:1]
	v_mov_b64_e32 v[24:25], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8532 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[18:19], v[0:1]
	v_mov_b64_e32 v[20:21], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8516 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[30:31], v[0:1]
	v_mov_b64_e32 v[32:33], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8500 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[26:27], v[0:1]
	v_mov_b64_e32 v[28:29], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8484 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[38:39], v[0:1]
	v_mov_b64_e32 v[40:41], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8468 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[34:35], v[0:1]
	v_mov_b64_e32 v[36:37], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8452 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[46:47], v[0:1]
	v_mov_b64_e32 v[48:49], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8436 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[42:43], v[0:1]
	v_mov_b64_e32 v[44:45], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8420 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[54:55], v[0:1]
	v_mov_b64_e32 v[56:57], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8404 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[50:51], v[0:1]
	v_mov_b64_e32 v[52:53], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8388 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[62:63], v[0:1]
	v_mov_b64_e32 v[64:65], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8372 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[58:59], v[0:1]
	v_mov_b64_e32 v[60:61], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8356 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[70:71], v[0:1]
	v_mov_b64_e32 v[72:73], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8340 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[66:67], v[0:1]
	v_mov_b64_e32 v[68:69], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8324 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[78:79], v[0:1]
	v_mov_b64_e32 v[80:81], v[2:3]
	scratch_load_b128 v[0:3], off, off offset:8308 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_mov_b64_e32 v[74:75], v[0:1]
	v_mov_b64_e32 v[76:77], v[2:3]
	s_set_vgpr_msb 0x400
	s_cbranch_scc1 .LBB0_2
	s_set_vgpr_msb 0xc3
	v_dual_mov_b32 v104 /*v872*/, v212 /*v980*/ :: v_dual_mov_b32 v42 /*v810*/, v170 /*v938*/
	s_clause 0xe
	scratch_load_b64 v[94:95] /*v[862:863]*/, off, off offset:8892 nv
	scratch_load_b64 v[6:7] /*v[774:775]*/, off, off offset:8876 nv
	s_set_vgpr_msb 0xc340
	scratch_load_b32 v157 /*v413*/, off, off offset:8840 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b64 v[242:243] /*v[754:755]*/, off, off offset:6188 nv
	scratch_load_b64 v[244:245] /*v[756:757]*/, off, off offset:6180 nv
	s_set_vgpr_msb 0x8000
	scratch_load_b64 v[4:5], off, off offset:6204 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b64 v[236:237] /*v[1004:1005]*/, off, off offset:6212 nv
	scratch_load_b64 v[212:213] /*v[980:981]*/, off, off offset:6196 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b32 v221, off, off offset:6316 nv
	scratch_load_b64 v[0:1], off, off offset:8900 nv
	s_set_vgpr_msb 0x83
	v_mov_b64_e32 v[28:29] /*v[540:541]*/, v[44:45] /*v[812:813]*/
	v_mov_b64_e32 v[30:31] /*v[542:543]*/, v[46:47] /*v[814:815]*/
	v_mov_b64_e32 v[32:33] /*v[544:545]*/, v[48:49] /*v[816:817]*/
	v_mov_b64_e32 v[34:35] /*v[546:547]*/, v[50:51] /*v[818:819]*/
	s_set_vgpr_msb 0x8300
	s_clause 0x2e
	scratch_load_b128 v[82:85], off, off offset:6448 nv
	scratch_load_b128 v[86:89], off, off offset:6464 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b128 v[44:47] /*v[812:815]*/, off, off offset:3360 nv
	scratch_load_b128 v[48:51] /*v[816:819]*/, off, off offset:3376 nv
	scratch_load_b128 v[114:117] /*v[882:885]*/, off, off offset:3200 nv
	scratch_load_b128 v[118:121] /*v[886:889]*/, off, off offset:3216 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[250:253] /*v[506:509]*/, off, off offset:6576 nv
	scratch_load_b128 v[254:257] /*v[510:513]*/, off, off offset:6592 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[74:77], off, off offset:6416 nv
	scratch_load_b128 v[78:81], off, off offset:6432 nv
	scratch_load_b128 v[66:69], off, off offset:3328 nv
	scratch_load_b128 v[70:73], off, off offset:3344 nv
	scratch_load_b128 v[58:61], off, off offset:3168 nv
	scratch_load_b128 v[62:65], off, off offset:3184 nv
	scratch_load_b128 v[50:53], off, off offset:6544 nv
	scratch_load_b128 v[54:57], off, off offset:6560 nv
	scratch_load_b128 v[42:45], off, off offset:6384 nv
	scratch_load_b128 v[46:49], off, off offset:6400 nv
	scratch_load_b128 v[34:37], off, off offset:3296 nv
	scratch_load_b128 v[38:41], off, off offset:3312 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[170:173] /*v[426:429]*/, off, off offset:6512 nv
	scratch_load_b128 v[174:177] /*v[430:433]*/, off, off offset:6528 nv
	scratch_load_b128 v[94:97] /*v[350:353]*/, off, off offset:6352 nv
	scratch_load_b128 v[98:101] /*v[354:357]*/, off, off offset:6368 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[130:133], off, off offset:3264 nv
	scratch_load_b128 v[134:137], off, off offset:3280 nv
	scratch_load_b128 v[122:125], off, off offset:3136 nv
	scratch_load_b128 v[126:129], off, off offset:3152 nv
	scratch_load_b128 v[114:117], off, off offset:6480 nv
	scratch_load_b128 v[118:121], off, off offset:6496 nv
	scratch_load_b128 v[98:101], off, off offset:3232 nv
	scratch_load_b128 v[102:105], off, off offset:3248 nv
	scratch_load_b128 v[90:93], off, off offset:6320 nv
	scratch_load_b128 v[94:97], off, off offset:6336 nv
	s_set_vgpr_msb 64
	scratch_load_b32 v155 /*v411*/, off, off offset:8836 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b64 v[126:127] /*v[638:639]*/, off, off offset:5544 nv
	scratch_load_b64 v[124:125] /*v[636:637]*/, off, off offset:8828 nv
	s_set_vgpr_msb 0x8002
	scratch_load_b64 v[172:173], off, off offset:8820 nv
	scratch_load_b64 v[170:171], off, off offset:5536 nv
	v_mov_b64_e32 v[162:163], v[76:77] /*v[588:589]*/
	v_mov_b64_e32 v[164:165], v[78:79] /*v[590:591]*/
	v_mov_b64_e32 v[166:167], v[80:81] /*v[592:593]*/
	v_mov_b64_e32 v[168:169], v[82:83] /*v[594:595]*/
	s_clause 0x10
	scratch_load_b128 v[138:141], off, off offset:6220 nv
	scratch_load_b128 v[142:145], off, off offset:6236 nv
	scratch_load_b128 v[146:149], off, off offset:3040 nv
	scratch_load_b128 v[150:153], off, off offset:3056 nv
	scratch_load_b128 v[174:177], off, off offset:3072 nv
	scratch_load_b128 v[178:181], off, off offset:3088 nv
	scratch_load_b128 v[182:185], off, off offset:6252 nv
	scratch_load_b128 v[186:189], off, off offset:6268 nv
	scratch_load_b128 v[190:193], off, off offset:6284 nv
	scratch_load_b128 v[194:197], off, off offset:6300 nv
	scratch_load_b128 v[198:201], off, off offset:3104 nv
	scratch_load_b128 v[202:205], off, off offset:3120 nv
	s_set_vgpr_msb 0x284
	scratch_load_b128 v[92:95] /*v[604:607]*/, off, off offset:3392 nv
	scratch_load_b128 v[96:99] /*v[608:611]*/, off, off offset:3408 nv
	scratch_load_b64 v[76:77] /*v[588:589]*/, off, off offset:8856 nv
	scratch_store_b64 off, v[104:105] /*v[360:361]*/, off offset:8308 nv
	s_set_vgpr_msb 0x8400
	s_wait_loadcnt 0xf
	s_clause 0x5
	scratch_load_b32 v171, off, off offset:8984 nv
	scratch_load_b32 v173, off, off offset:8988 nv
	s_set_vgpr_msb 64
	scratch_load_b32 v103 /*v359*/, off, off offset:8992 nv
	scratch_load_b32 v104 /*v360*/, off, off offset:8996 nv
	scratch_load_b32 v105 /*v361*/, off, off offset:9000 nv
	s_set_vgpr_msb 0x4080
	s_wait_loadcnt 0x5
	s_clause 0x12
	scratch_load_b32 v77 /*v589*/, off, off offset:9004 nv
	scratch_load_b32 v79 /*v591*/, off, off offset:9008 nv
	scratch_load_b32 v81 /*v593*/, off, off offset:9012 nv
	scratch_load_b32 v125 /*v637*/, off, off offset:9016 nv
	scratch_load_b32 v127 /*v639*/, off, off offset:9020 nv
	scratch_load_b32 v128 /*v640*/, off, off offset:9024 nv
	scratch_load_b32 v129 /*v641*/, off, off offset:9028 nv
	scratch_load_b32 v243 /*v755*/, off, off offset:9032 nv
	scratch_load_b32 v245 /*v757*/, off, off offset:9036 nv
	scratch_load_b32 v247 /*v759*/, off, off offset:9040 nv
	s_set_vgpr_msb 0x8000
	scratch_load_b32 v220, off, off offset:8864 nv
	scratch_load_b32 v230, off, off offset:8844 nv
	s_set_vgpr_msb 0x80
	scratch_load_b32 v249 /*v761*/, off, off offset:9044 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b64 v[238:239] /*v[1006:1007]*/, off, off offset:8868 nv
	s_set_vgpr_msb 0xc043
	scratch_load_b64 v[206:207] /*v[462:463]*/, off, off offset:8884 nv
	v_mov_b32_e32 v102 /*v358*/, v214 /*v982*/
	s_set_vgpr_msb 0x43c0
	s_clause 0x13
	scratch_load_b128 v[148:151] /*v[916:919]*/, off, off offset:3488 nv
	scratch_load_b128 v[152:155] /*v[920:923]*/, off, off offset:3504 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[216:219] /*v[472:475]*/, off, off offset:3520 nv
	scratch_load_b128 v[220:223] /*v[476:479]*/, off, off offset:3536 nv
	s_set_vgpr_msb 0x4000
	scratch_load_b128 v[206:209], off, off offset:3456 nv
	scratch_load_b128 v[210:213], off, off offset:3472 nv
	scratch_load_b128 v[154:157], off, off offset:3424 nv
	scratch_load_b128 v[158:161], off, off offset:3440 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b64 v[170:171] /*v[938:939]*/, off, off offset:8848 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b64 v[160:161] /*v[416:417]*/, off, off offset:8916 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b64 v[214:215] /*v[982:983]*/, off, off offset:6164 nv
	scratch_load_b64 v[216:217] /*v[984:985]*/, off, off offset:6172 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[224:227] /*v[480:483]*/, off, off offset:3552 nv
	scratch_load_b128 v[228:231] /*v[484:487]*/, off, off offset:3568 nv
	s_set_vgpr_msb 0x4002
	v_mov_b64_e32 v[26:27], v[20:21] /*v[532:533]*/
	v_mov_b64_e32 v[28:29], v[22:23] /*v[534:535]*/
	v_mov_b64_e32 v[30:31], v[24:25] /*v[536:537]*/
	v_mov_b64_e32 v[32:33], v[26:27] /*v[538:539]*/
	s_set_vgpr_msb 0x203
	v_mov_b64_e32 v[106:107], v[204:205] /*v[972:973]*/
	v_mov_b64_e32 v[108:109], v[206:207] /*v[974:975]*/
	v_mov_b64_e32 v[110:111], v[208:209] /*v[976:977]*/
	v_mov_b64_e32 v[112:113], v[210:211] /*v[978:979]*/
	s_set_vgpr_msb 0x3c1
	v_mov_b64_e32 v[210:211] /*v[978:979]*/, v[240:241] /*v[496:497]*/
	v_mov_b64_e32 v[208:209] /*v[976:977]*/, v[238:239] /*v[494:495]*/
	v_mov_b64_e32 v[206:207] /*v[974:975]*/, v[236:237] /*v[492:493]*/
	v_mov_b64_e32 v[204:205] /*v[972:973]*/, v[234:235] /*v[490:491]*/
	s_set_vgpr_msb 0xc140
	v_mov_b32_e32 v156 /*v412*/, v0
	s_set_vgpr_msb 0x4000
	scratch_load_b64 v[0:1], off, off offset:8908 nv
	s_set_vgpr_msb 0x42
	v_mov_b32_e32 v158 /*v414*/, v76 /*v588*/
	s_set_vgpr_msb 0x4201
	s_wait_loadcnt 0xf
	v_mov_b32_e32 v218, v206 /*v462*/
	s_set_vgpr_msb 0x140
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v154 /*v410*/, v0
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_5
.LBB0_4:
	scratch_load_b64 v[2:3], off, off offset:5544 nv
	v_mov_b32_e32 v18, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v19, v18 :: v_dual_mov_b32 v20, v18
	v_dual_mov_b32 v21, v18 :: v_dual_mov_b32 v22, v18
	v_dual_mov_b32 v23, v18 :: v_dual_mov_b32 v24, v18
	v_mov_b32_e32 v25, v18
	s_clause 0x3e
	scratch_store_b128 off, v[18:21], off offset:512 nv
	scratch_store_b128 off, v[22:25], off offset:528 nv
	scratch_store_b128 off, v[18:21], off offset:544 nv
	scratch_store_b128 off, v[22:25], off offset:560 nv
	scratch_store_b128 off, v[18:21], off offset:608 nv
	scratch_store_b128 off, v[22:25], off offset:624 nv
	scratch_store_b128 off, v[18:21], off offset:640 nv
	scratch_store_b128 off, v[22:25], off offset:656 nv
	scratch_store_b128 off, v[18:21], off offset:672 nv
	scratch_store_b128 off, v[22:25], off offset:688 nv
	scratch_store_b128 off, v[18:21], off offset:704 nv
	scratch_store_b128 off, v[22:25], off offset:720 nv
	scratch_store_b128 off, v[18:21], off offset:736 nv
	scratch_store_b128 off, v[22:25], off offset:752 nv
	scratch_store_b128 off, v[18:21], off offset:768 nv
	scratch_store_b128 off, v[22:25], off offset:784 nv
	scratch_store_b128 off, v[18:21], off offset:800 nv
	scratch_store_b128 off, v[22:25], off offset:816 nv
	scratch_store_b128 off, v[18:21], off offset:832 nv
	scratch_store_b128 off, v[22:25], off offset:848 nv
	scratch_store_b128 off, v[18:21], off offset:864 nv
	scratch_store_b128 off, v[22:25], off offset:880 nv
	scratch_store_b128 off, v[18:21], off offset:896 nv
	scratch_store_b128 off, v[22:25], off offset:912 nv
	scratch_store_b128 off, v[18:21], off offset:928 nv
	scratch_store_b128 off, v[22:25], off offset:944 nv
	scratch_store_b128 off, v[18:21], off offset:960 nv
	scratch_store_b128 off, v[22:25], off offset:976 nv
	scratch_store_b128 off, v[18:21], off offset:1056 nv
	scratch_store_b128 off, v[22:25], off offset:1072 nv
	scratch_store_b128 off, v[18:21], off offset:1088 nv
	scratch_store_b128 off, v[22:25], off offset:1104 nv
	scratch_store_b128 off, v[18:21], off offset:576 nv
	scratch_store_b128 off, v[22:25], off offset:592 nv
	scratch_store_b128 off, v[18:21], off offset:1024 nv
	scratch_store_b128 off, v[22:25], off offset:1040 nv
	scratch_store_b128 off, v[18:21], off offset:1120 nv
	scratch_store_b128 off, v[22:25], off offset:1136 nv
	scratch_store_b128 off, v[18:21], off offset:1152 nv
	scratch_store_b128 off, v[22:25], off offset:1168 nv
	scratch_store_b128 off, v[18:21], off offset:1184 nv
	scratch_store_b128 off, v[22:25], off offset:1200 nv
	scratch_store_b128 off, v[18:21], off offset:1280 nv
	scratch_store_b128 off, v[22:25], off offset:1296 nv
	scratch_store_b128 off, v[18:21], off offset:1408 nv
	scratch_store_b128 off, v[22:25], off offset:1424 nv
	scratch_store_b128 off, v[18:21], off offset:1376 nv
	scratch_store_b128 off, v[22:25], off offset:1392 nv
	scratch_store_b128 off, v[18:21], off offset:1536 nv
	scratch_store_b128 off, v[22:25], off offset:1552 nv
	scratch_store_b128 off, v[18:21], off offset:1312 nv
	scratch_store_b128 off, v[22:25], off offset:1328 nv
	scratch_store_b128 off, v[18:21], off offset:1344 nv
	scratch_store_b128 off, v[22:25], off offset:1360 nv
	scratch_store_b128 off, v[18:21], off offset:1440 nv
	scratch_store_b128 off, v[22:25], off offset:1456 nv
	scratch_store_b128 off, v[18:21], off offset:1472 nv
	scratch_store_b128 off, v[22:25], off offset:1488 nv
	scratch_store_b128 off, v[18:21], off offset:416 nv
	scratch_store_b128 off, v[22:25], off offset:432 nv
	scratch_store_b128 off, v[18:21], off offset:1728 nv
	scratch_store_b128 off, v[22:25], off offset:1744 nv
	scratch_store_b128 off, v[18:21], off offset:1568 nv
	s_clause 0x3c
	scratch_store_b128 off, v[22:25], off offset:1584 nv
	scratch_store_b128 off, v[18:21], off offset:1216 nv
	scratch_store_b128 off, v[22:25], off offset:1232 nv
	scratch_store_b128 off, v[18:21], off offset:1248 nv
	scratch_store_b128 off, v[22:25], off offset:1264 nv
	scratch_store_b128 off, v[18:21], off offset:1696 nv
	scratch_store_b128 off, v[22:25], off offset:1712 nv
	scratch_store_b128 off, v[18:21], off offset:1920 nv
	scratch_store_b128 off, v[22:25], off offset:1936 nv
	scratch_store_b128 off, v[18:21], off offset:1760 nv
	scratch_store_b128 off, v[22:25], off offset:1776 nv
	scratch_store_b128 off, v[18:21], off offset:1600 nv
	scratch_store_b128 off, v[22:25], off offset:1616 nv
	scratch_store_b128 off, v[18:21], off offset:1664 nv
	scratch_store_b128 off, v[22:25], off offset:1680 nv
	scratch_store_b128 off, v[18:21], off offset:2848 nv
	scratch_store_b128 off, v[22:25], off offset:2864 nv
	scratch_store_b128 off, v[18:21], off offset:1952 nv
	scratch_store_b128 off, v[22:25], off offset:1968 nv
	scratch_store_b128 off, v[18:21], off offset:1504 nv
	scratch_store_b128 off, v[22:25], off offset:1520 nv
	scratch_store_b128 off, v[18:21], off offset:1792 nv
	scratch_store_b128 off, v[22:25], off offset:1808 nv
	scratch_store_b128 off, v[18:21], off offset:2016 nv
	scratch_store_b128 off, v[22:25], off offset:2032 nv
	scratch_store_b128 off, v[18:21], off offset:288 nv
	scratch_store_b128 off, v[22:25], off offset:304 nv
	scratch_store_b128 off, v[18:21], off offset:352 nv
	scratch_store_b128 off, v[22:25], off offset:368 nv
	scratch_store_b128 off, v[18:21], off offset:2080 nv
	scratch_store_b128 off, v[22:25], off offset:2096 nv
	scratch_store_b128 off, v[18:21], off offset:2784 nv
	scratch_store_b128 off, v[22:25], off offset:2800 nv
	scratch_store_b128 off, v[18:21], off offset:1984 nv
	scratch_store_b128 off, v[22:25], off offset:2000 nv
	scratch_store_b128 off, v[18:21], off offset:2400 nv
	scratch_store_b128 off, v[22:25], off offset:2416 nv
	scratch_store_b128 off, v[18:21], off offset:2144 nv
	scratch_store_b128 off, v[22:25], off offset:2160 nv
	scratch_store_b128 off, v[18:21], off offset:1824 nv
	scratch_store_b128 off, v[22:25], off offset:1840 nv
	scratch_store_b128 off, v[18:21], off offset:1856 nv
	scratch_store_b128 off, v[22:25], off offset:1872 nv
	scratch_store_b128 off, v[18:21], off offset:1888 nv
	scratch_store_b128 off, v[22:25], off offset:1904 nv
	scratch_store_b128 off, v[18:21], off offset:480 nv
	scratch_store_b128 off, v[22:25], off offset:496 nv
	scratch_store_b128 off, v[18:21], off offset:2880 nv
	scratch_store_b128 off, v[22:25], off offset:2896 nv
	scratch_store_b128 off, v[18:21], off offset:2048 nv
	scratch_store_b128 off, v[22:25], off offset:2064 nv
	scratch_store_b128 off, v[18:21], off offset:2176 nv
	scratch_store_b128 off, v[22:25], off offset:2192 nv
	scratch_store_b128 off, v[18:21], off offset:2240 nv
	scratch_store_b128 off, v[22:25], off offset:2256 nv
	scratch_store_b128 off, v[18:21], off offset:2112 nv
	scratch_store_b128 off, v[22:25], off offset:2128 nv
	scratch_store_b128 off, v[18:21], off offset:2208 nv
	scratch_store_b128 off, v[22:25], off offset:2224 nv
	scratch_store_b128 off, v[18:21], off offset:3008 nv
	scratch_store_b128 off, v[22:25], off offset:3024 nv
	s_set_vgpr_msb 0x80
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v126 /*v638*/, v2
	s_set_vgpr_msb 0x8000
	scratch_load_b64 v[2:3], off, off offset:8828 nv
	s_set_vgpr_msb 0x80
	v_mov_b64_e32 v[16:17] /*v[528:529]*/, v[24:25]
	v_mov_b64_e32 v[14:15] /*v[526:527]*/, v[22:23]
	v_mov_b64_e32 v[12:13] /*v[524:525]*/, v[20:21]
	v_mov_b64_e32 v[10:11] /*v[522:523]*/, v[18:19]
	s_clause 0x35
	scratch_store_b128 off, v[18:21], off nv
	scratch_store_b128 off, v[22:25], off offset:16 nv
	scratch_store_b128 off, v[18:21], off offset:32 nv
	scratch_store_b128 off, v[22:25], off offset:48 nv
	scratch_store_b128 off, v[18:21], off offset:64 nv
	scratch_store_b128 off, v[22:25], off offset:80 nv
	scratch_store_b128 off, v[18:21], off offset:96 nv
	scratch_store_b128 off, v[22:25], off offset:112 nv
	scratch_store_b128 off, v[18:21], off offset:128 nv
	scratch_store_b128 off, v[22:25], off offset:144 nv
	scratch_store_b128 off, v[18:21], off offset:160 nv
	scratch_store_b128 off, v[22:25], off offset:176 nv
	scratch_store_b128 off, v[18:21], off offset:192 nv
	scratch_store_b128 off, v[22:25], off offset:208 nv
	scratch_store_b128 off, v[18:21], off offset:224 nv
	scratch_store_b128 off, v[22:25], off offset:240 nv
	scratch_store_b128 off, v[18:21], off offset:256 nv
	scratch_store_b128 off, v[22:25], off offset:272 nv
	scratch_store_b128 off, v[18:21], off offset:320 nv
	scratch_store_b128 off, v[22:25], off offset:336 nv
	scratch_store_b128 off, v[18:21], off offset:384 nv
	scratch_store_b128 off, v[22:25], off offset:400 nv
	scratch_store_b128 off, v[18:21], off offset:448 nv
	scratch_store_b128 off, v[22:25], off offset:464 nv
	scratch_store_b128 off, v[18:21], off offset:2272 nv
	scratch_store_b128 off, v[22:25], off offset:2288 nv
	scratch_store_b128 off, v[18:21], off offset:2304 nv
	scratch_store_b128 off, v[22:25], off offset:2320 nv
	scratch_store_b128 off, v[18:21], off offset:2336 nv
	scratch_store_b128 off, v[22:25], off offset:2352 nv
	scratch_store_b128 off, v[18:21], off offset:2368 nv
	scratch_store_b128 off, v[22:25], off offset:2384 nv
	scratch_store_b128 off, v[18:21], off offset:2432 nv
	scratch_store_b128 off, v[22:25], off offset:2448 nv
	scratch_store_b128 off, v[18:21], off offset:2464 nv
	scratch_store_b128 off, v[22:25], off offset:2480 nv
	scratch_store_b128 off, v[18:21], off offset:2496 nv
	scratch_store_b128 off, v[22:25], off offset:2512 nv
	scratch_store_b128 off, v[18:21], off offset:2528 nv
	scratch_store_b128 off, v[22:25], off offset:2544 nv
	scratch_store_b128 off, v[18:21], off offset:2560 nv
	scratch_store_b128 off, v[22:25], off offset:2576 nv
	scratch_store_b128 off, v[18:21], off offset:2592 nv
	scratch_store_b128 off, v[22:25], off offset:2608 nv
	scratch_store_b128 off, v[18:21], off offset:2624 nv
	scratch_store_b128 off, v[22:25], off offset:2640 nv
	scratch_store_b128 off, v[18:21], off offset:2656 nv
	scratch_store_b128 off, v[22:25], off offset:2672 nv
	scratch_store_b128 off, v[18:21], off offset:2688 nv
	scratch_store_b128 off, v[22:25], off offset:2704 nv
	scratch_store_b128 off, v[18:21], off offset:2720 nv
	scratch_store_b128 off, v[22:25], off offset:2736 nv
	scratch_store_b128 off, v[18:21], off offset:992 nv
	scratch_store_b128 off, v[22:25], off offset:1008 nv
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v124 /*v636*/, v2
	s_set_vgpr_msb 0x8000
	scratch_load_b64 v[2:3], off, off offset:8820 nv
	v_mov_b32_e32 v4, v82
	s_set_vgpr_msb 0x80
	v_mov_b64_e32 v[114:115] /*v[626:627]*/, v[24:25]
	v_mov_b64_e32 v[112:113] /*v[624:625]*/, v[22:23]
	v_mov_b64_e32 v[110:111] /*v[622:623]*/, v[20:21]
	v_mov_b64_e32 v[108:109] /*v[620:621]*/, v[18:19]
	s_set_vgpr_msb 0x8040
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[24:25]
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[22:23]
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[20:21]
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[18:19]
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, v[24:25]
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, v[22:23]
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[20:21]
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[18:19]
	v_mov_b64_e32 v[152:153] /*v[408:409]*/, v[24:25]
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[22:23]
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[20:21]
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[18:19]
	v_mov_b64_e32 v[168:169] /*v[424:425]*/, v[24:25]
	v_mov_b64_e32 v[166:167] /*v[422:423]*/, v[22:23]
	v_mov_b64_e32 v[164:165] /*v[420:421]*/, v[20:21]
	v_mov_b64_e32 v[162:163] /*v[418:419]*/, v[18:19]
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[24:25]
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[22:23]
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[20:21]
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[18:19]
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[24:25]
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[22:23]
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[20:21]
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[18:19]
	v_mov_b64_e32 v[184:185] /*v[440:441]*/, v[24:25]
	v_mov_b64_e32 v[182:183] /*v[438:439]*/, v[22:23]
	v_mov_b64_e32 v[180:181] /*v[436:437]*/, v[20:21]
	v_mov_b64_e32 v[178:179] /*v[434:435]*/, v[18:19]
	s_clause 0x1
	scratch_store_b128 off, v[18:21], off offset:2752 nv
	scratch_store_b128 off, v[22:25], off offset:2768 nv
	s_set_vgpr_msb 0x4000
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v172, v2
	scratch_load_b64 v[2:3], off, off offset:5536 nv
	s_set_vgpr_msb 0xc0
	v_mov_b64_e32 v[210:211] /*v[978:979]*/, v[24:25]
	v_mov_b64_e32 v[208:209] /*v[976:977]*/, v[22:23]
	v_mov_b64_e32 v[206:207] /*v[974:975]*/, v[20:21]
	v_mov_b64_e32 v[204:205] /*v[972:973]*/, v[18:19]
	v_mov_b64_e32 v[128:129] /*v[896:897]*/, v[24:25]
	v_mov_b64_e32 v[126:127] /*v[894:895]*/, v[22:23]
	v_mov_b64_e32 v[124:125] /*v[892:893]*/, v[20:21]
	v_mov_b64_e32 v[122:123] /*v[890:891]*/, v[18:19]
	s_set_vgpr_msb 0xc040
	v_mov_b64_e32 v[200:201] /*v[456:457]*/, v[24:25]
	v_mov_b64_e32 v[198:199] /*v[454:455]*/, v[22:23]
	v_mov_b64_e32 v[196:197] /*v[452:453]*/, v[20:21]
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, v[18:19]
	s_clause 0x1
	scratch_store_b128 off, v[18:21], off offset:2976 nv
	scratch_store_b128 off, v[22:25], off offset:2992 nv
	s_set_vgpr_msb 0x4000
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v170, v2
	s_set_vgpr_msb 0x80
	v_mov_b64_e32 v[58:59] /*v[570:571]*/, v[24:25]
	v_mov_b64_e32 v[56:57] /*v[568:569]*/, v[22:23]
	v_mov_b64_e32 v[54:55] /*v[566:567]*/, v[20:21]
	v_mov_b64_e32 v[52:53] /*v[564:565]*/, v[18:19]
	s_set_vgpr_msb 0x8040
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[24:25]
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[22:23]
	v_mov_b64_e32 v[64:65] /*v[320:321]*/, v[20:21]
	v_mov_b64_e32 v[62:63] /*v[318:319]*/, v[18:19]
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[24:25]
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[22:23]
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[20:21]
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[18:19]
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[24:25]
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[22:23]
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[20:21]
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[18:19]
	v_mov_b64_e32 v[136:137] /*v[392:393]*/, v[24:25]
	v_mov_b64_e32 v[134:135] /*v[390:391]*/, v[22:23]
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, v[20:21]
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[18:19]
	s_set_vgpr_msb 0x4080
	v_mov_b64_e32 v[160:161] /*v[672:673]*/, v[24:25]
	v_mov_b64_e32 v[158:159] /*v[670:671]*/, v[22:23]
	v_mov_b64_e32 v[156:157] /*v[668:669]*/, v[20:21]
	v_mov_b64_e32 v[154:155] /*v[666:667]*/, v[18:19]
	s_set_vgpr_msb 0x80c0
	v_mov_b64_e32 v[66:67] /*v[834:835]*/, v[24:25]
	v_mov_b64_e32 v[64:65] /*v[832:833]*/, v[22:23]
	v_mov_b64_e32 v[62:63] /*v[830:831]*/, v[20:21]
	v_mov_b64_e32 v[60:61] /*v[828:829]*/, v[18:19]
	s_set_vgpr_msb 0xc080
	v_mov_b64_e32 v[50:51] /*v[562:563]*/, v[24:25]
	v_mov_b64_e32 v[48:49] /*v[560:561]*/, v[22:23]
	v_mov_b64_e32 v[46:47] /*v[558:559]*/, v[20:21]
	v_mov_b64_e32 v[44:45] /*v[556:557]*/, v[18:19]
	s_set_vgpr_msb 0x8040
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[24:25]
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[22:23]
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[20:21]
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[18:19]
	v_mov_b64_e32 v[192:193] /*v[448:449]*/, v[24:25]
	v_mov_b64_e32 v[190:191] /*v[446:447]*/, v[22:23]
	v_mov_b64_e32 v[188:189] /*v[444:445]*/, v[20:21]
	v_mov_b64_e32 v[186:187] /*v[442:443]*/, v[18:19]
	s_set_vgpr_msb 0x4080
	v_mov_b64_e32 v[176:177] /*v[688:689]*/, v[24:25]
	v_mov_b64_e32 v[174:175] /*v[686:687]*/, v[22:23]
	v_mov_b64_e32 v[172:173] /*v[684:685]*/, v[20:21]
	v_mov_b64_e32 v[170:171] /*v[682:683]*/, v[18:19]
	s_clause 0x3
	scratch_store_b128 off, v[18:21], off offset:2816 nv
	scratch_store_b128 off, v[22:25], off offset:2832 nv
	scratch_store_b128 off, v[18:21], off offset:2944 nv
	scratch_store_b128 off, v[22:25], off offset:2960 nv
	s_set_vgpr_msb 0x8000
	v_mov_b64_e32 v[212:213], v[24:25]
	v_mov_b64_e32 v[210:211], v[22:23]
	v_mov_b64_e32 v[208:209], v[20:21]
	v_mov_b64_e32 v[206:207], v[18:19]
	s_set_vgpr_msb 0xc0
	v_mov_b64_e32 v[202:203] /*v[970:971]*/, v[24:25]
	v_mov_b64_e32 v[200:201] /*v[968:969]*/, v[22:23]
	v_mov_b64_e32 v[198:199] /*v[966:967]*/, v[20:21]
	v_mov_b64_e32 v[196:197] /*v[964:965]*/, v[18:19]
	s_clause 0x1
	scratch_store_b128 off, v[18:21], off offset:2912 nv
	scratch_store_b128 off, v[22:25], off offset:2928 nv
	s_set_vgpr_msb 0xc000
	v_mov_b64_e32 v[160:161], v[24:25]
	v_mov_b64_e32 v[158:159], v[22:23]
	v_mov_b64_e32 v[156:157], v[20:21]
	v_mov_b64_e32 v[154:155], v[18:19]
	v_mov_b64_e32 v[168:169], v[24:25]
	v_mov_b64_e32 v[166:167], v[22:23]
	v_mov_b64_e32 v[164:165], v[20:21]
	v_mov_b64_e32 v[162:163], v[18:19]
	v_mov_b64_e32 v[112:113], v[24:25]
	v_mov_b64_e32 v[110:111], v[22:23]
	v_mov_b64_e32 v[108:109], v[20:21]
	v_mov_b64_e32 v[106:107], v[18:19]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[24:25]
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[22:23]
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[20:21]
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[18:19]
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[32:33], v[24:25]
	v_mov_b64_e32 v[30:31], v[22:23]
	v_mov_b64_e32 v[28:29], v[20:21]
	v_mov_b64_e32 v[26:27], v[18:19]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[24:25]
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[22:23]
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[20:21]
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[18:19]
	s_set_vgpr_msb 0x40c0
	v_mov_b64_e32 v[154:155] /*v[922:923]*/, v[24:25]
	v_mov_b64_e32 v[152:153] /*v[920:921]*/, v[22:23]
	v_mov_b64_e32 v[150:151] /*v[918:919]*/, v[20:21]
	v_mov_b64_e32 v[148:149] /*v[916:917]*/, v[18:19]
	s_clause 0x1
	scratch_store_b128 off, v[18:21], off offset:1632 nv
	scratch_store_b128 off, v[22:25], off offset:1648 nv
	s_set_vgpr_msb 0xc080
	v_mov_b64_e32 v[34:35] /*v[546:547]*/, v[24:25]
	v_mov_b64_e32 v[32:33] /*v[544:545]*/, v[22:23]
	v_mov_b64_e32 v[30:31] /*v[542:543]*/, v[20:21]
	v_mov_b64_e32 v[28:29] /*v[540:541]*/, v[18:19]
	s_set_vgpr_msb 0x8000
.LBB0_5:
	s_sub_co_i32 s8, s50, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s8, 1
	s_cbranch_scc1 .LBB0_9
	s_clause 0x1
	scratch_load_b32 v1, off, off offset:6160 nv
	scratch_load_b64 v[2:3], off, off offset:8308 nv
	s_set_vgpr_msb 0xc3
	v_dual_add_nc_u32 v241 /*v1009*/, s11, v173 :: v_dual_mov_b32 v217 /*v985*/, v216 /*v984*/
	v_dual_mov_b32 v43 /*v811*/, v42 /*v810*/ :: v_dual_mov_b32 v213 /*v981*/, v212 /*v980*/
	s_set_vgpr_msb 0xc3c7
	v_dual_add_nc_u32 v105 /*v873*/, s11, v104 /*v360*/ :: v_dual_mov_b32 v215 /*v983*/, v214 /*v982*/
	s_set_vgpr_msb 0xc784
	v_add_nc_u32_e32 v43 /*v555*/, s11, v105 /*v361*/
	s_set_vgpr_msb 0x84c8
	v_dual_add_nc_u32 v239 /*v1007*/, s11, v81 /*v593*/ :: v_dual_add_nc_u32 v7 /*v775*/, s11, v127 /*v639*/
	s_set_vgpr_msb 0xc882
	v_mov_b32_e32 v127 /*v639*/, v126 /*v638*/
	s_set_vgpr_msb 0x8200
	v_or_b32_e32 v18, 64, v230
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v9 /*v521*/, s11, v243 /*v755*/
	s_set_vgpr_msb 0x88c8
	v_dual_add_nc_u32 v253 /*v1021*/, s11, v245 /*v757*/ :: v_dual_add_nc_u32 v240 /*v1008*/, s11, v247 /*v759*/
	s_set_vgpr_msb 0xc882
	v_mov_b32_e32 v245 /*v757*/, v244 /*v756*/
	s_set_vgpr_msb 0x8208
	v_or_b32_e32 v19, 0x60, v249 /*v761*/
	s_set_vgpr_msb 0x8c3
	v_mov_b32_e32 v237 /*v1005*/, v236 /*v1004*/
	s_set_vgpr_msb 0xc382
	v_mov_b32_e32 v243 /*v755*/, v242 /*v754*/
	s_set_vgpr_msb 0x824b
	v_dual_mov_b32 v202 /*v458*/, v34 /*v802*/ :: v_dual_add_nc_u32 v205 /*v461*/, s11, v125 /*v637*/
	s_set_vgpr_msb 0x4bc8
	v_dual_add_nc_u32 v255 /*v1023*/, s11, v77 /*v589*/ :: v_dual_add_nc_u32 v251 /*v1019*/, s11, v79 /*v591*/
	v_dual_add_nc_u32 v95 /*v863*/, s11, v128 /*v640*/ :: v_dual_add_nc_u32 v171 /*v939*/, s11, v129 /*v641*/
	s_set_vgpr_msb 0xc808
	v_or_b32_e32 v21, 0xa0, v249 /*v761*/
	v_or_b32_e32 v23, 0xe0, v249 /*v761*/
	s_set_vgpr_msb 0x881
	v_mov_b64_e32 v[90:91] /*v[602:603]*/, v[230:231] /*v[486:487]*/
	v_mov_b64_e32 v[88:89] /*v[600:601]*/, v[228:229] /*v[484:485]*/
	v_mov_b64_e32 v[86:87] /*v[598:599]*/, v[226:227] /*v[482:483]*/
	v_mov_b64_e32 v[84:85] /*v[596:597]*/, v[224:225] /*v[480:481]*/
	s_set_vgpr_msb 0x8141
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[216:217] /*v[472:473]*/
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[218:219] /*v[474:475]*/
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[220:221] /*v[476:477]*/
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x4143
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[148:149] /*v[916:917]*/
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[150:151] /*v[918:919]*/
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[152:153] /*v[920:921]*/
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[154:155] /*v[922:923]*/
	s_set_vgpr_msb 0x4382
	v_mov_b64_e32 v[66:67] /*v[578:579]*/, v[34:35] /*v[546:547]*/
	v_mov_b64_e32 v[64:65] /*v[576:577]*/, v[32:33] /*v[544:545]*/
	v_mov_b64_e32 v[62:63] /*v[574:575]*/, v[30:31] /*v[542:543]*/
	v_mov_b64_e32 v[60:61] /*v[572:573]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0x8242
	v_mov_b32_e32 v206 /*v462*/, v124 /*v636*/
	s_set_vgpr_msb 0x4200
	v_or_b32_e32 v20, 0x80, v230
	v_or_b32_e32 v22, 0xc0, v230
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[10:11], v[206:207]
	v_mov_b64_e32 v[12:13], v[208:209]
	v_mov_b64_e32 v[14:15], v[210:211]
	v_mov_b64_e32 v[16:17], v[212:213]
	s_set_vgpr_msb 0xc0
	v_mov_b64_e32 v[220:221] /*v[988:989]*/, s[4:5]
	s_set_vgpr_msb 0xc000
	v_mov_b64_e32 v[180:181], v[112:113]
	v_mov_b64_e32 v[178:179], v[110:111]
	v_mov_b64_e32 v[176:177], v[108:109]
	v_mov_b64_e32 v[174:175], v[106:107]
	v_mov_b64_e32 v[188:189], v[32:33]
	v_mov_b64_e32 v[186:187], v[30:31]
	v_mov_b64_e32 v[184:185], v[28:29]
	v_mov_b64_e32 v[182:183], v[26:27]
	s_set_vgpr_msb 0x84
	v_dual_mov_b32 v42 /*v554*/, v218 :: v_dual_add_nc_u32 v8 /*v520*/, s11, v103 /*v359*/
	s_set_vgpr_msb 0x8441
	v_mov_b32_e32 v155 /*v411*/, v154 /*v410*/
	s_set_vgpr_msb 0x4140
	v_mov_b32_e32 v204 /*v460*/, v172
	s_ashr_i32 s9, s8, 31
	s_set_vgpr_msb 0x400c
	v_mov_b32_e32 v5, v4
	scratch_store_b64 off, v[212:213] /*v[980:981]*/, off offset:6196 nv
	s_set_vgpr_msb 0xc40
	v_add_nc_u32_e32 v203 /*v459*/, s11, v171
	s_set_vgpr_msb 0x4008
	v_mov_b32_e32 v171, v170
	s_mov_b64 s[10:11], 0
	s_clause 0x3
	scratch_store_b64 off, v[244:245] /*v[756:757]*/, off offset:6180 nv
	scratch_store_b64 off, v[242:243] /*v[754:755]*/, off offset:6188 nv
	s_set_vgpr_msb 0x800
	scratch_store_b64 off, v[170:171], off offset:5536 nv
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v3, v2
	s_clause 0x1
	scratch_store_b64 off, v[2:3], off offset:8308 nv
	scratch_load_b64 v[2:3], off, off offset:6136 nv
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v3, v2
	s_set_vgpr_msb 12
	s_clause 0x4
	scratch_store_b64 off, v[216:217] /*v[984:985]*/, off offset:6172 nv
	scratch_store_b64 off, v[42:43] /*v[810:811]*/, off offset:8324 nv
	s_set_vgpr_msb 0xc01
	scratch_store_b64 off, v[2:3], off offset:6136 nv
	scratch_load_b64 v[2:3], off, off offset:6128 nv
	v_mov_b32_e32 v0, v102 /*v358*/
	s_set_vgpr_msb 0x140
	v_add_nc_u32_e32 v207 /*v463*/, s49, v1
	s_set_vgpr_msb 0x4008
	v_or_b32_e32 v1, 32, v249 /*v761*/
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, v220, v1
	scratch_store_b32 off, v1, off offset:8356 nv
	s_wait_xcnt 0x0
	v_add_nc_u32_e32 v1, v220, v18
	scratch_store_b32 off, v1, off offset:8372 nv
	s_wait_xcnt 0x0
	v_add_nc_u32_e32 v1, v220, v19
	scratch_store_b32 off, v1, off offset:8388 nv
	s_wait_xcnt 0x0
	v_add_nc_u32_e32 v1, v220, v20
	scratch_store_b32 off, v1, off offset:8404 nv
	s_wait_xcnt 0x0
	v_add_nc_u32_e32 v1, v220, v21
	scratch_store_b32 off, v1, off offset:8420 nv
	s_wait_xcnt 0x0
	v_add_nc_u32_e32 v1, v220, v22
	scratch_store_b32 off, v1, off offset:8436 nv
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_dual_add_nc_u32 v1, v220, v23 :: v_dual_mov_b32 v3, v2
	s_set_vgpr_msb 8
	s_clause 0x5
	scratch_store_b64 off, v[126:127] /*v[638:639]*/, off offset:5544 nv
	s_set_vgpr_msb 0x80c
	scratch_store_b64 off, v[214:215] /*v[982:983]*/, off offset:6164 nv
	s_set_vgpr_msb 0xc00
	scratch_store_b64 off, v[2:3], off offset:6128 nv
	scratch_load_b64 v[2:3], off, off offset:6144 nv
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v3, v2
	s_clause 0x1
	scratch_store_b64 off, v[2:3], off offset:6144 nv
	scratch_load_b64 v[2:3], off, off offset:6152 nv
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v3, v2
	s_clause 0x2
	scratch_store_b64 off, v[2:3], off offset:6152 nv
	s_set_vgpr_msb 12
	scratch_store_b64 off, v[236:237] /*v[1004:1005]*/, off offset:6212 nv
	s_set_vgpr_msb 0xc00
	v_add_nc_u32_e32 v2, v220, v230
	s_clause 0x3e
	scratch_store_b64 off, v[4:5], off offset:6204 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b128 v[230:233] /*v[998:1001]*/, off, off offset:7760 nv
	scratch_load_b128 v[234:237] /*v[1002:1005]*/, off, off offset:7776 nv
	scratch_load_b128 v[44:47] /*v[812:815]*/, off, off offset:5584 nv
	scratch_load_b128 v[48:51] /*v[816:819]*/, off, off offset:5600 nv
	scratch_load_b128 v[138:141] /*v[906:909]*/, off, off offset:6384 nv
	scratch_load_b128 v[142:145] /*v[910:913]*/, off, off offset:6400 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[100:103] /*v[612:615]*/, off, off offset:6416 nv
	scratch_load_b128 v[104:107] /*v[616:619]*/, off, off offset:6432 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[222:225] /*v[990:993]*/, off, off offset:6252 nv
	scratch_load_b128 v[226:229] /*v[994:997]*/, off, off offset:6268 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[254:257] /*v[766:769]*/, off, off offset:7824 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[2:5] /*v[770:773]*/, off, off offset:7840 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[188:191] /*v[700:703]*/, off, off offset:6832 nv
	scratch_load_b128 v[192:195] /*v[704:707]*/, off, off offset:6848 nv
	scratch_load_b128 v[204:207] /*v[716:719]*/, off, off offset:6896 nv
	scratch_load_b128 v[208:211] /*v[720:723]*/, off, off offset:6912 nv
	scratch_load_b128 v[220:223] /*v[732:735]*/, off, off offset:6960 nv
	scratch_load_b128 v[224:227] /*v[736:739]*/, off, off offset:6976 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[114:117] /*v[882:885]*/, off, off offset:7024 nv
	scratch_load_b128 v[118:121] /*v[886:889]*/, off, off offset:7040 nv
	scratch_load_b128 v[84:87] /*v[852:855]*/, off, off offset:7120 nv
	scratch_load_b128 v[88:91] /*v[856:859]*/, off, off offset:7136 nv
	scratch_load_b128 v[76:79] /*v[844:847]*/, off, off offset:7216 nv
	scratch_load_b128 v[80:83] /*v[848:851]*/, off, off offset:7232 nv
	scratch_load_b128 v[68:71] /*v[836:839]*/, off, off offset:7280 nv
	scratch_load_b128 v[72:75] /*v[840:843]*/, off, off offset:7296 nv
	scratch_load_b128 v[36:39] /*v[804:807]*/, off, off offset:7376 nv
	scratch_load_b128 v[40:43] /*v[808:811]*/, off, off offset:7392 nv
	scratch_load_b128 v[28:31] /*v[796:799]*/, off, off offset:7472 nv
	scratch_load_b128 v[32:35] /*v[800:803]*/, off, off offset:7488 nv
	scratch_load_b128 v[20:23] /*v[788:791]*/, off, off offset:7536 nv
	scratch_load_b128 v[24:27] /*v[792:795]*/, off, off offset:7552 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[236:239] /*v[748:751]*/, off, off offset:7600 nv
	scratch_load_b128 v[240:243] /*v[752:755]*/, off, off offset:7616 nv
	scratch_load_b128 v[124:127] /*v[636:639]*/, off, off offset:7792 nv
	scratch_load_b128 v[128:131] /*v[640:643]*/, off, off offset:7808 nv
	scratch_load_b128 v[212:215] /*v[724:727]*/, off, off offset:7888 nv
	scratch_load_b128 v[216:219] /*v[728:731]*/, off, off offset:7904 nv
	scratch_load_b128 v[76:79] /*v[588:591]*/, off, off offset:7952 nv
	scratch_load_b128 v[80:83] /*v[592:595]*/, off, off offset:7968 nv
	scratch_load_b128 v[180:183] /*v[692:695]*/, off, off offset:8048 nv
	scratch_load_b128 v[184:187] /*v[696:699]*/, off, off offset:8064 nv
	scratch_load_b128 v[132:135] /*v[644:647]*/, off, off offset:8112 nv
	scratch_load_b128 v[136:139] /*v[648:651]*/, off, off offset:8128 nv
	scratch_load_b128 v[116:119] /*v[628:631]*/, off, off offset:6320 nv
	scratch_load_b128 v[120:123] /*v[632:635]*/, off, off offset:6336 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[130:133] /*v[898:901]*/, off, off offset:6480 nv
	scratch_load_b128 v[134:137] /*v[902:905]*/, off, off offset:6496 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[224:227] /*v[480:483]*/, off, off offset:6576 nv
	scratch_load_b128 v[228:231] /*v[484:487]*/, off, off offset:6592 nv
	scratch_load_b128 v[240:243] /*v[496:499]*/, off, off offset:6220 nv
	scratch_load_b128 v[244:247] /*v[500:503]*/, off, off offset:6236 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[96:99] /*v[864:867]*/, off, off offset:6284 nv
	scratch_load_b128 v[100:103] /*v[868:871]*/, off, off offset:6300 nv
	scratch_load_b128 v[146:149] /*v[914:917]*/, off, off offset:6608 nv
	scratch_load_b128 v[150:153] /*v[918:921]*/, off, off offset:6624 nv
	scratch_load_b128 v[162:165] /*v[930:933]*/, off, off offset:6672 nv
	scratch_load_b128 v[166:169] /*v[934:937]*/, off, off offset:6688 nv
	scratch_load_b128 v[12:15] /*v[780:783]*/, off, off offset:6736 nv
	scratch_load_b128 v[16:19] /*v[784:787]*/, off, off offset:6752 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[228:231] /*v[740:743]*/, off, off offset:6992 nv
	scratch_load_b128 v[232:235] /*v[744:747]*/, off, off offset:7008 nv
	s_clause 0x25
	scratch_load_b128 v[246:249] /*v[758:761]*/, off, off offset:7056 nv
	scratch_load_b128 v[250:253] /*v[762:765]*/, off, off offset:7072 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[52:55] /*v[820:823]*/, off, off offset:7152 nv
	scratch_load_b128 v[56:59] /*v[824:827]*/, off, off offset:7168 nv
	scratch_load_b128 v[172:175] /*v[940:943]*/, off, off offset:7312 nv
	scratch_load_b128 v[176:179] /*v[944:947]*/, off, off offset:7328 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[196:199] /*v[708:711]*/, off, off offset:7408 nv
	scratch_load_b128 v[200:203] /*v[712:715]*/, off, off offset:7424 nv
	scratch_load_b128 v[140:143] /*v[652:655]*/, off, off offset:7568 nv
	scratch_load_b128 v[144:147] /*v[656:659]*/, off, off offset:7584 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[106:109] /*v[874:877]*/, off, off offset:7632 nv
	scratch_load_b128 v[110:113] /*v[878:881]*/, off, off offset:7648 nv
	scratch_load_b128 v[188:191] /*v[956:959]*/, off, off offset:7728 nv
	scratch_load_b128 v[192:195] /*v[960:963]*/, off, off offset:7744 nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[232:235] /*v[488:491]*/, off, off offset:8080 nv
	scratch_load_b128 v[236:239] /*v[492:495]*/, off, off offset:8096 nv
	scratch_load_b128 v[248:251] /*v[504:507]*/, off, off offset:8144 nv
	scratch_load_b128 v[252:255] /*v[508:511]*/, off, off offset:8160 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[0:3] /*v[512:515]*/, off, off offset:5904 nv
	scratch_load_b128 v[4:7] /*v[516:519]*/, off, off offset:5920 nv
	scratch_load_b128 v[34:37] /*v[546:549]*/, off, off offset:5968 nv
	scratch_load_b128 v[38:41] /*v[550:553]*/, off, off offset:5984 nv
	scratch_load_b128 v[18:21] /*v[530:533]*/, off, off offset:6032 nv
	scratch_load_b128 v[22:25] /*v[534:537]*/, off, off offset:6048 nv
	scratch_load_b128 v[26:29] /*v[538:541]*/, off, off offset:6096 nv
	scratch_load_b128 v[30:33] /*v[542:545]*/, off, off offset:6112 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b64 v[10:11] /*v[778:779]*/, off, off offset:8324 nv
	scratch_load_b64 v[8:9] /*v[776:777]*/, off, off offset:6164 nv
	scratch_load_b64 v[92:93] /*v[860:861]*/, off, off offset:6172 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b64 v[244:245] /*v[756:757]*/, off, off offset:8308 nv
	scratch_store_b32 off, v2, off offset:8340 nv
	s_set_vgpr_msb 0x8000
.LBB0_7:
	s_clause 0x1
	scratch_load_b32 v82, off, off offset:6160 nv
	scratch_load_b32 v2, off, off offset:8304 nv
	s_add_co_i32 s3, s2, s10
	s_set_vgpr_msb 8
	s_clause 0x3e
	scratch_store_b128 off, v[60:63] /*v[572:575]*/, off offset:5504 nv
	scratch_store_b128 off, v[64:67] /*v[576:579]*/, off offset:5520 nv
	scratch_store_b128 off, v[10:13] /*v[522:525]*/, off offset:5472 nv
	scratch_store_b128 off, v[14:17] /*v[526:529]*/, off offset:5488 nv
	s_set_vgpr_msb 0x80c
	scratch_store_b128 off, v[204:207] /*v[972:975]*/, off offset:5440 nv
	scratch_store_b128 off, v[208:211] /*v[976:979]*/, off offset:5456 nv
	s_set_vgpr_msb 0xc08
	scratch_store_b128 off, v[84:87] /*v[596:599]*/, off offset:3552 nv
	scratch_store_b128 off, v[88:91] /*v[600:603]*/, off offset:3568 nv
	s_set_vgpr_msb 0x80c
	scratch_store_b128 off, v[60:63] /*v[828:831]*/, off offset:5408 nv
	scratch_store_b128 off, v[64:67] /*v[832:835]*/, off offset:5424 nv
	scratch_store_b128 off, v[122:125] /*v[890:893]*/, off offset:5376 nv
	scratch_store_b128 off, v[126:129] /*v[894:897]*/, off offset:5392 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[194:197] /*v[450:453]*/, off offset:5344 nv
	scratch_store_b128 off, v[198:201] /*v[454:457]*/, off offset:5360 nv
	s_set_vgpr_msb 0x40c
	scratch_store_b128 off, v[196:199] /*v[964:967]*/, off offset:5088 nv
	scratch_store_b128 off, v[200:203] /*v[968:971]*/, off offset:5104 nv
	s_set_vgpr_msb 0xc04
	scratch_store_b128 off, v[186:189] /*v[442:445]*/, off offset:5024 nv
	scratch_store_b128 off, v[190:193] /*v[446:449]*/, off offset:5040 nv
	scratch_store_b128 off, v[178:181] /*v[434:437]*/, off offset:4960 nv
	scratch_store_b128 off, v[182:185] /*v[438:441]*/, off offset:4976 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[44:47] /*v[556:559]*/, off offset:5312 nv
	scratch_store_b128 off, v[48:51] /*v[560:563]*/, off offset:5328 nv
	s_set_vgpr_msb 0x804
	scratch_store_b128 off, v[162:165] /*v[418:421]*/, off offset:5280 nv
	scratch_store_b128 off, v[166:169] /*v[422:425]*/, off offset:5296 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[154:157] /*v[666:669]*/, off offset:4928 nv
	scratch_store_b128 off, v[158:161] /*v[670:673]*/, off offset:4944 nv
	s_set_vgpr_msb 0x804
	scratch_store_b128 off, v[146:149] /*v[402:405]*/, off offset:5248 nv
	scratch_store_b128 off, v[150:153] /*v[406:409]*/, off offset:5264 nv
	s_set_vgpr_msb 0x408
	scratch_store_b128 off, v[170:173] /*v[682:685]*/, off offset:5216 nv
	scratch_store_b128 off, v[174:177] /*v[686:689]*/, off offset:5232 nv
	scratch_store_b128 off, v[52:55] /*v[564:567]*/, off offset:5056 nv
	scratch_store_b128 off, v[56:59] /*v[568:571]*/, off offset:5072 nv
	s_set_vgpr_msb 0x800
	scratch_store_b128 off, v[10:13], off offset:3456 nv
	scratch_store_b128 off, v[14:17], off offset:3472 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[216:219] /*v[472:475]*/, off offset:3488 nv
	scratch_store_b128 off, v[220:223] /*v[476:479]*/, off offset:3504 nv
	scratch_store_b128 off, v[170:173] /*v[426:429]*/, off offset:3520 nv
	scratch_store_b128 off, v[174:177] /*v[430:433]*/, off offset:3536 nv
	s_set_vgpr_msb 0x400
	scratch_store_b128 off, v[182:185], off offset:5152 nv
	scratch_store_b128 off, v[186:189], off offset:5168 nv
	scratch_store_b128 off, v[174:177], off offset:5184 nv
	scratch_store_b128 off, v[178:181], off offset:5200 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[138:141] /*v[394:397]*/, off offset:4992 nv
	scratch_store_b128 off, v[142:145] /*v[398:401]*/, off offset:5008 nv
	s_set_vgpr_msb 0x400
	scratch_store_b128 off, v[162:165], off offset:5120 nv
	scratch_store_b128 off, v[166:169], off offset:5136 nv
	scratch_store_b128 off, v[154:157], off offset:3424 nv
	scratch_store_b128 off, v[158:161], off offset:3440 nv
	s_set_vgpr_msb 4
	scratch_store_b128 off, v[130:133] /*v[386:389]*/, off offset:4864 nv
	scratch_store_b128 off, v[134:137] /*v[390:393]*/, off offset:4880 nv
	scratch_store_b128 off, v[114:117] /*v[370:373]*/, off offset:4896 nv
	scratch_store_b128 off, v[118:121] /*v[374:377]*/, off offset:4912 nv
	scratch_store_b128 off, v[122:125] /*v[378:381]*/, off offset:4800 nv
	scratch_store_b128 off, v[126:129] /*v[382:385]*/, off offset:4816 nv
	scratch_store_b128 off, v[106:109] /*v[362:365]*/, off offset:4832 nv
	scratch_store_b128 off, v[110:113] /*v[366:369]*/, off offset:4848 nv
	scratch_store_b128 off, v[86:89] /*v[342:345]*/, off offset:4704 nv
	scratch_store_b128 off, v[90:93] /*v[346:349]*/, off offset:4720 nv
	scratch_store_b128 off, v[70:73] /*v[326:329]*/, off offset:4768 nv
	scratch_store_b128 off, v[74:77] /*v[330:333]*/, off offset:4784 nv
	scratch_store_b128 off, v[62:65] /*v[318:321]*/, off offset:4736 nv
	scratch_store_b128 off, v[66:69] /*v[322:325]*/, off offset:4752 nv
	scratch_store_b128 off, v[78:81] /*v[334:337]*/, off offset:4672 nv
	s_clause 0x3
	scratch_store_b128 off, v[82:85] /*v[338:341]*/, off offset:4688 nv
	s_set_vgpr_msb 0x4c8
	scratch_store_b128 off, v[108:111] /*v[620:623]*/, off offset:4640 nv
	scratch_store_b128 off, v[112:115] /*v[624:627]*/, off offset:4656 nv
	s_lshl_b32 s5, s3, 5
	s_clause 0x1
	scratch_load_b128 v[180:183] /*v[948:951]*/, off, off offset:7184 nv
	scratch_load_b128 v[184:187] /*v[952:955]*/, off, off offset:7200 nv
	s_or_b32 s3, s5, 16
	s_clause 0x3
	scratch_load_b128 v[154:157] /*v[922:925]*/, off, off offset:7344 nv
	scratch_load_b128 v[158:161] /*v[926:929]*/, off, off offset:7360 nv
	scratch_load_b128 v[196:199] /*v[964:967]*/, off, off offset:7664 nv
	scratch_load_b128 v[200:203] /*v[968:971]*/, off, off offset:7680 nv
	s_set_vgpr_msb 0xc842
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[68:69] /*v[580:581]*/
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[70:71] /*v[582:583]*/
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[72:73] /*v[584:585]*/
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[74:75] /*v[586:587]*/
	s_set_vgpr_msb 0x42c0
	s_clause 0x38
	scratch_load_b128 v[60:63] /*v[828:831]*/, off, off offset:8016 nv
	scratch_load_b128 v[64:67] /*v[832:835]*/, off, off offset:8032 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[84:87] /*v[596:599]*/, off, off offset:6352 nv
	scratch_load_b128 v[88:91] /*v[600:603]*/, off, off offset:6368 nv
	scratch_load_b128 v[10:13] /*v[522:525]*/, off, off offset:6512 nv
	scratch_load_b128 v[14:17] /*v[526:529]*/, off, off offset:6528 nv
	s_set_vgpr_msb 0x8040
	scratch_load_b128 v[208:211] /*v[464:467]*/, off, off offset:6544 nv
	scratch_load_b128 v[212:215] /*v[468:471]*/, off, off offset:6560 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[60:63] /*v[572:575]*/, off, off offset:6448 nv
	scratch_load_b128 v[64:67] /*v[576:579]*/, off, off offset:6464 nv
	scratch_load_b128 v[148:151] /*v[660:663]*/, off, off offset:6640 nv
	scratch_load_b128 v[152:155] /*v[664:667]*/, off, off offset:6656 nv
	scratch_load_b128 v[68:71] /*v[580:583]*/, off, off offset:6704 nv
	scratch_load_b128 v[72:75] /*v[584:587]*/, off, off offset:6720 nv
	scratch_load_b128 v[172:175] /*v[684:687]*/, off, off offset:6768 nv
	scratch_load_b128 v[176:179] /*v[688:691]*/, off, off offset:6784 nv
	s_set_vgpr_msb 0x8000
	scratch_load_b128 v[10:13], off, off offset:6800 nv
	scratch_load_b128 v[14:17], off, off offset:6816 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[114:117] /*v[370:373]*/, off, off offset:6864 nv
	scratch_load_b128 v[118:121] /*v[374:377]*/, off, off offset:6880 nv
	scratch_load_b128 v[122:125] /*v[378:381]*/, off, off offset:6928 nv
	scratch_load_b128 v[126:129] /*v[382:385]*/, off, off offset:6944 nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[212:215] /*v[980:983]*/, off, off offset:7088 nv
	scratch_load_b128 v[216:219] /*v[984:987]*/, off, off offset:7104 nv
	scratch_load_b128 v[204:207] /*v[972:975]*/, off, off offset:5648 nv
	scratch_load_b128 v[208:211] /*v[976:979]*/, off, off offset:5664 nv
	scratch_load_b128 v[122:125] /*v[890:893]*/, off, off offset:7440 nv
	scratch_load_b128 v[126:129] /*v[894:897]*/, off, off offset:7456 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[164:167] /*v[676:679]*/, off, off offset:7504 nv
	scratch_load_b128 v[168:171] /*v[680:683]*/, off, off offset:7520 nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[242:245] /*v[1010:1013]*/, off, off offset:5712 nv
	scratch_load_b128 v[246:249] /*v[1014:1017]*/, off, off offset:5728 nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[156:159] /*v[668:671]*/, off, off offset:7696 nv
	scratch_load_b128 v[160:163] /*v[672:675]*/, off, off offset:7712 nv
	s_set_vgpr_msb 0x8040
	scratch_load_b128 v[130:133] /*v[386:389]*/, off, off offset:7856 nv
	scratch_load_b128 v[134:137] /*v[390:393]*/, off, off offset:7872 nv
	scratch_load_b128 v[146:149] /*v[402:405]*/, off, off offset:7920 nv
	scratch_load_b128 v[150:153] /*v[406:409]*/, off, off offset:7936 nv
	scratch_load_b128 v[216:219] /*v[472:475]*/, off, off offset:7984 nv
	scratch_load_b128 v[220:223] /*v[476:479]*/, off, off offset:8000 nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[44:47] /*v[556:559]*/, off, off offset:8208 nv
	scratch_load_b128 v[48:51] /*v[560:563]*/, off, off offset:8224 nv
	scratch_load_b128 v[52:55] /*v[564:567]*/, off, off offset:8240 nv
	scratch_load_b128 v[56:59] /*v[568:571]*/, off, off offset:8256 nv
	scratch_load_b128 v[108:111] /*v[620:623]*/, off, off offset:8272 nv
	scratch_load_b128 v[112:115] /*v[624:627]*/, off, off offset:8288 nv
	s_add_nc_u64 s[10:11], s[10:11], 1
	s_set_vgpr_msb 0x8000
	s_wait_loadcnt 0x35
	v_or_b32_e32 v18, s5, v82
	v_or_b32_e32 v82, s3, v82
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v18, v18, s48
	v_mul_lo_u32 v82, v82, s48
	s_wait_loadcnt 0x34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v70, v18, v2, 4
	v_add_lshl_u32 v82, v82, v2, 4
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3584 nv
	scratch_load_b128 v[6:9], off, off offset:3600 nv
	v_or_b32_e32 v30, 32, v70
	buffer_load_b128 v[18:21], v70, s[12:15], null offen
	v_or_b32_e32 v74, 0xc0, v70
	v_or_b32_e32 v42, 64, v70
	v_or_b32_e32 v83, 32, v82
	buffer_load_b128 v[22:25], v30, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[26:29], v70, s[16:19], null offen
	buffer_load_b128 v[30:33], v30, s[16:19], null offen
	v_or_b32_e32 v46, 0x60, v70
	buffer_load_b128 v[66:69], v74, s[12:15], null offen
	v_or_b32_e32 v58, 0x80, v70
	s_clause 0x1
	buffer_load_b128 v[34:37], v42, s[12:15], null offen
	buffer_load_b128 v[146:149], v83, s[12:15], null offen
	s_set_vgpr_msb 64
	s_clause 0x1
	buffer_load_b128 v[194:197] /*v[450:453]*/, v82, s[16:19], null offen
	buffer_load_b128 v[198:201] /*v[454:457]*/, v83, s[16:19], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v83, 64, v82
	v_or_b32_e32 v84, 0x60, v82
	s_clause 0x1
	buffer_load_b128 v[50:53], v58, s[12:15], null offen
	buffer_load_b128 v[38:41], v46, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[42:45], v42, s[16:19], null offen
	buffer_load_b128 v[46:49], v46, s[16:19], null offen
	v_or_b32_e32 v62, 0xa0, v70
	s_clause 0x3
	buffer_load_b128 v[142:145], v82, s[12:15], null offen
	s_set_vgpr_msb 64
	buffer_load_b128 v[178:181] /*v[434:437]*/, v83, s[12:15], null offen
	buffer_load_b128 v[182:185] /*v[438:441]*/, v84, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[186:189] /*v[442:445]*/, v83, s[16:19], null offen
	buffer_load_b128 v[190:193] /*v[446:449]*/, v84, s[16:19], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v83, 0x80, v82
	v_or_b32_e32 v84, 0xa0, v82
	buffer_load_b128 v[54:57], v62, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[58:61], v58, s[16:19], null offen
	buffer_load_b128 v[62:65], v62, s[16:19], null offen
	s_set_vgpr_msb 64
	buffer_load_b128 v[166:169] /*v[422:425]*/, v84, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[170:173] /*v[426:429]*/, v83, s[16:19], null offen
	buffer_load_b128 v[174:177] /*v[430:433]*/, v84, s[16:19], null offen
	buffer_load_b128 v[162:165] /*v[418:421]*/, v83, s[12:15], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v83, 0xc0, v82
	v_or_b32_e32 v82, 0xe0, v82
	v_or_b32_e32 v78, 0xe0, v70
	s_set_vgpr_msb 64
	buffer_load_b128 v[110:113] /*v[366:369]*/, v82, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[98:101] /*v[354:357]*/, v83, s[16:19], null offen
	buffer_load_b128 v[102:105] /*v[358:361]*/, v82, s[16:19], null offen
	s_clause 0x2
	buffer_load_b128 v[106:109] /*v[362:365]*/, v83, s[12:15], null offen
	s_set_vgpr_msb 0x4000
	buffer_load_b128 v[70:73], v78, s[12:15], null offen
	s_clause 0x1
	buffer_load_b128 v[74:77], v74, s[16:19], null offen
	buffer_load_b128 v[78:81], v78, s[16:19], null offen
	s_set_vgpr_msb 1
	s_wait_loadcnt 0x1f
	ds_store_b128 v157 /*v413*/, v[18:21]
	s_wait_loadcnt 0x1e
	ds_store_b128 v157 /*v413*/, v[22:25] offset:32
	s_set_vgpr_msb 0x104
	v_wmma_f32_16x16x32_bf16 v[82:89], v[18:25], v[138:145] /*v[394:401]*/, 0
	s_set_vgpr_msb 0x401
	s_wait_loadcnt 0x16
	ds_store_b128 v157 /*v413*/, v[50:53] offset:128
	s_wait_loadcnt 0xd
	ds_store_b128 v157 /*v413*/, v[54:57] offset:160
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[90:97], v[26:33], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3264 nv
	scratch_load_b128 v[6:9], off, off offset:3280 nv
	s_set_vgpr_msb 1
	ds_store_b128 v157 /*v413*/, v[34:37] offset:64
	ds_store_b128 v157 /*v413*/, v[38:41] offset:96
	ds_store_b128 v157 /*v413*/, v[66:69] offset:192
	s_wait_loadcnt 0x4
	ds_store_b128 v157 /*v413*/, v[70:73] offset:224
	s_set_vgpr_msb 0x14c
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[286:293]*/, v[18:25], v[180:187] /*v[948:955]*/, 0
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[302:309]*/, v[18:25], v[154:161] /*v[922:929]*/, 0
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[18:25], v[196:203] /*v[964:971]*/, 0
	s_set_vgpr_msb 0x4c0c
	v_wmma_f32_16x16x32_bf16 v[82:89], v[34:41], v[44:51] /*v[812:819]*/, v[82:89]
	s_set_vgpr_msb 0xc08
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[82:89], v[50:57], v[116:123] /*v[628:635]*/, v[82:89]
	s_set_vgpr_msb 0x80c
	v_wmma_f32_16x16x32_bf16 v[90:97], v[42:49], v[60:67] /*v[828:835]*/, v[90:97]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[90:97], v[58:65], v[130:137] /*v[898:905]*/, v[90:97]
	s_set_vgpr_msb 0xc08
	v_wmma_f32_16x16x32_bf16 v[214:221], v[26:33], v[92:99] /*v[604:611]*/, 0
	s_set_vgpr_msb 0x880
	s_clause 0x1
	scratch_load_b128 v[92:95] /*v[604:607]*/, off, off offset:8176 nv
	scratch_load_b128 v[96:99] /*v[608:611]*/, off, off offset:8192 nv
	s_set_vgpr_msb 0x8008
	v_wmma_f32_16x16x32_bf16 v[214:221], v[42:49], v[254:261] /*v[766:773]*/, v[214:221]
	s_set_vgpr_msb 0x80c
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[214:221], v[58:65], v[146:153] /*v[914:921]*/, v[214:221]
	s_set_vgpr_msb 0xc5c
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[26:33], v[212:219] /*v[980:987]*/, 0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[42:49], v[84:91] /*v[852:859]*/, v[22:29] /*v[278:285]*/
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[58:65], v[52:59] /*v[820:827]*/, v[22:29] /*v[278:285]*/
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[286:293]*/, v[34:41], v[76:83] /*v[844:851]*/, v[30:37] /*v[286:293]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[294:301]*/, v[26:33], v[204:211] /*v[972:979]*/, 0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[294:301]*/, v[42:49], v[68:75] /*v[836:843]*/, v[38:45] /*v[294:301]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[294:301]*/, v[58:65], v[172:179] /*v[940:947]*/, v[38:45] /*v[294:301]*/
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[302:309]*/, v[34:41], v[36:43] /*v[804:811]*/, v[46:53] /*v[302:309]*/
	s_set_vgpr_msb 0x5c58
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[302:309]*/, v[50:57], v[196:203] /*v[708:715]*/, v[46:53] /*v[302:309]*/
	s_set_vgpr_msb 0x585c
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[310:317]*/, v[26:33], v[122:129] /*v[890:897]*/, 0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[310:317]*/, v[42:49], v[28:35] /*v[796:803]*/, v[54:61] /*v[310:317]*/
	s_set_vgpr_msb 0x5c58
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[310:317]*/, v[58:65], v[164:171] /*v[676:683]*/, v[54:61] /*v[310:317]*/
	s_set_vgpr_msb 0x584c
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[26:33], v[242:249] /*v[1010:1017]*/, 0
	s_set_vgpr_msb 0x4c58
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[42:49], v[236:243] /*v[748:755]*/, v[70:77] /*v[326:333]*/
	s_set_vgpr_msb 0x585c
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[58:65], v[106:113] /*v[874:881]*/, v[70:77] /*v[326:333]*/
	s_set_vgpr_msb 0x5c58
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[34:41], v[156:163] /*v[668:675]*/, v[78:85] /*v[334:341]*/
	s_set_vgpr_msb 0x585c
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[50:57], v[188:195] /*v[956:963]*/, v[78:85] /*v[334:341]*/
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[26:33], v[230:237] /*v[998:1005]*/, 0
	s_set_vgpr_msb 0x5c58
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[42:49], v[124:131] /*v[636:643]*/, v[86:93] /*v[342:349]*/
	s_set_vgpr_msb 0x5854
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[58:65], v[130:137] /*v[386:393]*/, v[86:93] /*v[342:349]*/
	s_set_vgpr_msb 0x5400
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[98:105], v[18:25], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3296 nv
	scratch_load_b128 v[6:9], off, off offset:3312 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[98:105], v[34:41], v[84:91] /*v[596:603]*/, v[98:105]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[98:105], v[50:57], v[10:17] /*v[522:529]*/, v[98:105]
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[182:189], v[26:33], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3328 nv
	scratch_load_b128 v[6:9], off, off offset:3344 nv
	s_set_vgpr_msb 12
	v_wmma_f32_16x16x32_bf16 v[182:189], v[42:49], v[138:145] /*v[906:913]*/, v[182:189]
	s_set_vgpr_msb 0xc04
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[182:189], v[58:65], v[208:215] /*v[464:471]*/, v[182:189]
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[190:197], v[18:25], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3360 nv
	scratch_load_b128 v[6:9], off, off offset:3376 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[190:197], v[34:41], v[100:107] /*v[612:619]*/, v[190:197]
	s_set_vgpr_msb 0x804
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[190:197], v[50:57], v[224:231] /*v[480:487]*/, v[190:197]
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[198:205], v[26:33], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3072 nv
	scratch_load_b128 v[6:9], off, off offset:3088 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[198:205], v[42:49], v[60:67] /*v[572:579]*/, v[198:205]
	s_set_vgpr_msb 0x804
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[198:205], v[58:65], v[240:247] /*v[496:503]*/, v[198:205]
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[206:213], v[18:25], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3648 nv
	scratch_load_b128 v[6:9], off, off offset:3664 nv
	s_set_vgpr_msb 12
	v_wmma_f32_16x16x32_bf16 v[206:213], v[34:41], v[222:229] /*v[990:997]*/, v[206:213]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[206:213], v[50:57], v[96:103] /*v[864:871]*/, v[206:213]
	s_set_vgpr_msb 0xc00
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[222:229], v[18:25], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3712 nv
	scratch_load_b128 v[6:9], off, off offset:3728 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[222:229], v[34:41], v[148:155] /*v[660:667]*/, v[222:229]
	s_set_vgpr_msb 0x80c
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[222:229], v[50:57], v[162:169] /*v[930:937]*/, v[222:229]
	s_set_vgpr_msb 0xc00
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[230:237], v[26:33], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3776 nv
	scratch_load_b128 v[6:9], off, off offset:3792 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[230:237], v[42:49], v[68:75] /*v[580:587]*/, v[230:237]
	s_set_vgpr_msb 0x80c
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[230:237], v[58:65], v[12:19] /*v[780:787]*/, v[230:237]
	s_set_vgpr_msb 0xc00
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[238:245], v[18:25], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3840 nv
	scratch_load_b128 v[6:9], off, off offset:3856 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[238:245], v[34:41], v[172:179] /*v[684:691]*/, v[238:245]
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[238:245], v[50:57], v[10:17], v[238:245]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[246:253], v[26:33], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3904 nv
	scratch_load_b128 v[6:9], off, off offset:3920 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[246:253], v[42:49], v[188:195] /*v[700:707]*/, v[246:253]
	s_set_vgpr_msb 0x804
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[246:253], v[58:65], v[114:121] /*v[370:377]*/, v[246:253]
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[254:261], v[18:25], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3968 nv
	scratch_load_b128 v[6:9], off, off offset:3984 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[254:261], v[34:41], v[204:211] /*v[716:723]*/, v[254:261]
	s_set_vgpr_msb 0x804
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[254:261], v[50:57], v[122:129] /*v[378:385]*/, v[254:261]
	s_set_vgpr_msb 0x440
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[262:269]*/, v[26:33], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5616 nv
	scratch_load_b128 v[6:9], off, off offset:5632 nv
	s_set_vgpr_msb 0x58
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[262:269]*/, v[42:49], v[220:227] /*v[732:739]*/, v[6:13] /*v[262:269]*/
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[262:269]*/, v[58:65], v[228:235] /*v[740:747]*/, v[6:13] /*v[262:269]*/
	s_set_vgpr_msb 0x5840
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[270:277]*/, v[18:25], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5680 nv
	scratch_load_b128 v[6:9], off, off offset:5696 nv
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[270:277]*/, v[34:41], v[114:121] /*v[882:889]*/, v[14:21] /*v[270:277]*/
	s_set_vgpr_msb 0x5c58
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[270:277]*/, v[50:57], v[246:253] /*v[758:765]*/, v[14:21] /*v[270:277]*/
	s_set_vgpr_msb 0x5840
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[318:325]*/, v[18:25], v[2:9], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5744 nv
	scratch_load_b128 v[6:9], off, off offset:5760 nv
	s_set_vgpr_msb 0x5c
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[318:325]*/, v[34:41], v[20:27] /*v[788:795]*/, v[62:69] /*v[318:325]*/
	s_set_vgpr_msb 0x5c58
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[318:325]*/, v[50:57], v[140:147] /*v[652:659]*/, v[62:69] /*v[318:325]*/
	s_set_vgpr_msb 0x5800
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[110:117], v[18:25], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5776 nv
	scratch_load_b128 v[6:9], off, off offset:5792 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[110:117], v[34:41], v[212:219] /*v[724:731]*/, v[110:117]
	s_set_vgpr_msb 0x804
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[110:117], v[50:57], v[146:153] /*v[402:409]*/, v[110:117]
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[174:181], v[26:33], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5808 nv
	scratch_load_b128 v[6:9], off, off offset:5824 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[174:181], v[42:49], v[76:83] /*v[588:595]*/, v[174:181]
	s_set_vgpr_msb 0x804
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[174:181], v[58:65], v[216:223] /*v[472:479]*/, v[174:181]
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[118:125], v[18:25], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5840 nv
	scratch_load_b128 v[6:9], off, off offset:5856 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[118:125], v[34:41], v[180:187] /*v[692:699]*/, v[118:125]
	s_set_vgpr_msb 0x804
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[118:125], v[50:57], v[232:239] /*v[488:495]*/, v[118:125]
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[166:173], v[26:33], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5872 nv
	scratch_load_b128 v[6:9], off, off offset:5888 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[166:173], v[42:49], v[132:139] /*v[644:651]*/, v[166:173]
	s_set_vgpr_msb 0x804
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[166:173], v[58:65], v[248:255] /*v[504:511]*/, v[166:173]
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[126:133], v[18:25], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:5936 nv
	scratch_load_b128 v[6:9], off, off offset:5952 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[126:133], v[34:41], v[92:99] /*v[604:611]*/, v[126:133]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[126:133], v[50:57], v[0:7] /*v[512:519]*/, v[126:133]
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[158:165], v[26:33], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:6000 nv
	scratch_load_b128 v[6:9], off, off offset:6016 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[158:165], v[42:49], v[44:51] /*v[556:563]*/, v[158:165]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[158:165], v[58:65], v[34:41] /*v[546:553]*/, v[158:165]
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[134:141], v[18:25], v[2:9], 0
	s_clause 0x3
	scratch_load_b128 v[18:21], off, off offset:3232 nv
	scratch_load_b128 v[22:25], off, off offset:3248 nv
	scratch_load_b128 v[2:5], off, off offset:6064 nv
	scratch_load_b128 v[6:9], off, off offset:6080 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[134:141], v[34:41], v[52:59] /*v[564:571]*/, v[134:141]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[134:141], v[50:57], v[18:25] /*v[530:537]*/, v[134:141]
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[82:89], v[66:73], v[18:25], v[82:89]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3136 nv
	scratch_load_b128 v[22:25], off, off offset:3152 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[150:157], v[26:33], v[2:9], 0
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:7248 nv
	scratch_load_b128 v[6:9], off, off offset:7264 nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[150:157], v[42:49], v[108:115] /*v[620:627]*/, v[150:157]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[150:157], v[58:65], v[26:33] /*v[538:545]*/, v[150:157]
	s_clause 0x1
	scratch_load_b64 v[62:63], off, off offset:6188 nv
	scratch_load_b64 v[64:65], off, off offset:6204 nv
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[90:97], v[74:81], v[18:25], v[90:97]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4480 nv
	scratch_load_b128 v[22:25], off, off offset:4496 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[286:293]*/, v[50:57], v[2:9], v[30:37] /*v[286:293]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b32 v55, off, off offset:8836 nv
	scratch_load_b64 v[56:57], off, off offset:6180 nv
	s_wait_loadcnt 0x1
	v_or_b32_e32 v40, s5, v55
	v_wmma_f32_16x16x32_bf16 v[98:105], v[66:73], v[18:25], v[98:105]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3168 nv
	scratch_load_b128 v[22:25], off, off offset:3184 nv
	v_nop
	v_pk_add_f32 v[26:27], v[90:91], v[62:63] neg_lo:[0,1] neg_hi:[0,1]
	v_nop
	v_nop
	v_mul_f32_e32 v28, s4, v104
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[182:189], v[74:81], v[18:25], v[182:189]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3200 nv
	scratch_load_b128 v[22:25], off, off offset:3216 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[190:197], v[66:73], v[18:25], v[190:197]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3040 nv
	scratch_load_b128 v[22:25], off, off offset:3056 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_mul_f32_e32 v33, s4, v197
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[198:205], v[74:81], v[18:25], v[198:205]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3104 nv
	scratch_load_b128 v[22:25], off, off offset:3120 nv
	v_mul_f32_e32 v32, s4, v196
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[34:35], v[198:199], v[64:65] neg_lo:[0,1] neg_hi:[0,1]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[206:213], v[66:73], v[18:25], v[206:213]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3616 nv
	scratch_load_b128 v[22:25], off, off offset:3632 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[214:221], v[74:81], v[18:25], v[214:221]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3680 nv
	scratch_load_b128 v[22:25], off, off offset:3696 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[222:229], v[66:73], v[18:25], v[222:229]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3744 nv
	scratch_load_b128 v[22:25], off, off offset:3760 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[230:237], v[74:81], v[18:25], v[230:237]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3808 nv
	scratch_load_b128 v[22:25], off, off offset:3824 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[238:245], v[66:73], v[18:25], v[238:245]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3872 nv
	scratch_load_b128 v[22:25], off, off offset:3888 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[246:253], v[74:81], v[18:25], v[246:253]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:3936 nv
	scratch_load_b128 v[22:25], off, off offset:3952 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[254:261], v[66:73], v[18:25], v[254:261]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4000 nv
	scratch_load_b128 v[22:25], off, off offset:4016 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[262:269]*/, v[74:81], v[18:25], v[6:13] /*v[262:269]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4032 nv
	scratch_load_b128 v[22:25], off, off offset:4048 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[270:277]*/, v[66:73], v[18:25], v[14:21] /*v[270:277]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4064 nv
	scratch_load_b128 v[22:25], off, off offset:4080 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[74:81], v[18:25], v[22:29] /*v[278:285]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4096 nv
	scratch_load_b128 v[22:25], off, off offset:4112 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[286:293]*/, v[66:73], v[18:25], v[30:37] /*v[286:293]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4128 nv
	scratch_load_b128 v[22:25], off, off offset:4144 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[294:301]*/, v[74:81], v[18:25], v[38:45] /*v[294:301]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4160 nv
	scratch_load_b128 v[22:25], off, off offset:4176 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[302:309]*/, v[66:73], v[18:25], v[46:53] /*v[302:309]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4192 nv
	scratch_load_b128 v[22:25], off, off offset:4208 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[310:317]*/, v[74:81], v[18:25], v[54:61] /*v[310:317]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4224 nv
	scratch_load_b128 v[22:25], off, off offset:4240 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[318:325]*/, v[66:73], v[18:25], v[62:69] /*v[318:325]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4256 nv
	scratch_load_b128 v[22:25], off, off offset:4272 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[74:81], v[18:25], v[70:77] /*v[326:333]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4288 nv
	scratch_load_b128 v[22:25], off, off offset:4304 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[66:73], v[18:25], v[78:85] /*v[334:341]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4320 nv
	scratch_load_b128 v[22:25], off, off offset:4336 nv
	s_set_vgpr_msb 0x50
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[74:81], v[18:25], v[86:93] /*v[342:349]*/
	s_set_vgpr_msb 0x5000
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4352 nv
	scratch_load_b128 v[22:25], off, off offset:4368 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[110:117], v[66:73], v[18:25], v[110:117]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4384 nv
	scratch_load_b128 v[22:25], off, off offset:4400 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[174:181], v[74:81], v[18:25], v[174:181]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4416 nv
	scratch_load_b128 v[22:25], off, off offset:4432 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[118:125], v[66:73], v[18:25], v[118:125]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4448 nv
	scratch_load_b128 v[22:25], off, off offset:4464 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[166:173], v[74:81], v[18:25], v[166:173]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4512 nv
	scratch_load_b128 v[22:25], off, off offset:4528 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[126:133], v[66:73], v[18:25], v[126:133]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4544 nv
	scratch_load_b128 v[22:25], off, off offset:4560 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[158:165], v[74:81], v[18:25], v[158:165]
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:4576 nv
	scratch_load_b128 v[22:25], off, off offset:4592 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[134:141], v[66:73], v[18:25], v[134:141]
	s_clause 0x3
	scratch_load_b128 v[18:21], off, off offset:4608 nv
	scratch_load_b128 v[22:25], off, off offset:4624 nv
	scratch_load_b64 v[70:71], off, off offset:6152 nv
	scratch_load_b64 v[72:73], off, off offset:6144 nv
	s_wait_loadcnt 0x1
	v_pk_add_f32 v[38:39], v[214:215], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[150:157], v[74:81], v[18:25], v[150:157]
	scratch_load_b64 v[80:81], off, off offset:6128 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_mul_f32_e32 v18, s4, v82
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v40, v207 /*v463*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v19, s4, v83 :: v_dual_bitop2_b32 v41, 2, v40 bitop3:0x54
	v_dual_mul_f32 v22, s4, v86 :: v_dual_bitop2_b32 v46, 3, v40 bitop3:0x54
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_ge_i32_e32 vcc_lo, v40, v207 /*v463*/
	v_cndmask_b32_e64 v18, v18, 0xff61b1e6, s5
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v20, s4, v84 :: v_dual_mul_f32 v21, s4, v85
	v_dual_mul_f32 v23, s4, v87 :: v_dual_bitop2_b32 v47, 4, v40 bitop3:0x54
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v41, v207 /*v463*/
	v_cndmask_b32_e64 v19, v19, 0xff61b1e6, s5
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v25, s4, v89 :: v_dual_bitop2_b32 v48, 5, v40 bitop3:0x54
	s_set_vgpr_msb 4
	v_sub_f32_e32 v18, v18, v202 /*v458*/
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v46, v207 /*v463*/
	v_cndmask_b32_e64 v20, v20, 0xff61b1e6, s5
	v_sub_f32_e32 v19, v19, v202 /*v458*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v49, 6, v40
	v_mul_f32_e32 v24, s4, v88
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v47, v207 /*v463*/
	v_cndmask_b32_e64 v21, v21, 0xff61b1e6, s5
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v19, 0x3fb8aa3b, v19
	s_set_vgpr_msb 4
	v_sub_f32_e32 v20, v20, v202 /*v458*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v54, 7, v40
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v48, v207 /*v463*/
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	v_sub_f32_e32 v21, v21, v202 /*v458*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v18, 0x3fb8aa3b, v18
	v_exp_f32_e32 v19, v19
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v49, v207 /*v463*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v21, 0x3fb8aa3b, v21
	v_exp_f32_e32 v18, v18
	s_set_vgpr_msb 4
	v_sub_f32_e32 v22, v22, v202 /*v458*/
	s_and_b32 s5, s47, vcc_lo
	v_sub_f32_e32 v23, v23, v202 /*v458*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v20, 0x3fb8aa3b, v20
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v54, v207 /*v463*/
	v_exp_f32_e32 v21, v21
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v23, 0x3fb8aa3b, v23
	v_exp_f32_e32 v20, v20
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[18:19], v[26:27], v[18:19]
	v_pk_add_f32 v[26:27], v[92:93], v[62:63] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	s_set_vgpr_msb 4
	v_sub_f32_e32 v24, v24, v202 /*v458*/
	v_exp_f32_e32 v23, v23
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[18:19], v[220:221] /*v[988:989]*/, v[18:19]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[20:21], v[26:27], v[20:21]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v25, v25, v202 /*v458*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	v_mul_f32_e32 v24, 0x3fb8aa3b, v24
	v_cvt_pk_bf16_f32 v18, v18, v19
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[20:21], v[220:221] /*v[988:989]*/, v[20:21]
	v_mul_f32_e32 v25, 0x3fb8aa3b, v25
	s_set_vgpr_msb 0x304
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v24, v24
	v_cmp_gt_i32_e32 vcc_lo, v40, v203 /*v459*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v19, v20, v21
	v_pk_add_f32 v[20:21], v[94:95], v[62:63] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v25, v25
	v_mul_f32_e32 v26, s4, v102
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_ge_i32_e32 vcc_lo, v40, v203 /*v459*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[20:21], v[20:21], v[22:23]
	v_pk_add_f32 v[22:23], v[96:97], v[62:63] neg_lo:[0,1] neg_hi:[0,1]
	v_mul_f32_e32 v27, s4, v103
	s_clause 0x1
	scratch_load_b64 v[88:89], off, off offset:6136 nv
	scratch_load_b64 v[78:79], off, off offset:5536 nv
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[20:21], v[220:221] /*v[988:989]*/, v[20:21]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	v_dual_mul_f32 v24, s4, v100 :: v_dual_mul_f32 v25, s4, v101
	scratch_load_b64 v[96:97], off, off offset:6212 nv
	v_cvt_pk_bf16_f32 v20, v20, v21
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v21, v22, v23
	v_dual_mul_f32 v22, s4, v98 :: v_dual_mul_f32 v23, s4, v99
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v41, v203 /*v459*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 0x40c
	v_sub_f32_e32 v22, v22, v104 /*v872*/
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc04
	v_cmp_gt_i32_e32 vcc_lo, v46, v203 /*v459*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	s_set_vgpr_msb 0x40c
	v_sub_f32_e32 v23, v23, v104 /*v872*/
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc04
	v_cmp_gt_i32_e32 vcc_lo, v47, v203 /*v459*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v23, 0x3fb8aa3b, v23
	s_set_vgpr_msb 12
	v_sub_f32_e32 v24, v24, v104 /*v872*/
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc04
	v_cmp_gt_i32_e32 vcc_lo, v48, v203 /*v459*/
	v_cndmask_b32_e64 v26, v26, 0xff61b1e6, s5
	s_set_vgpr_msb 0x40c
	v_sub_f32_e32 v25, v25, v104 /*v872*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	v_exp_f32_e32 v23, v23
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v49, v203 /*v459*/
	s_set_vgpr_msb 0x40c
	v_sub_f32_e32 v26, v26, v104 /*v872*/
	v_cndmask_b32_e64 v27, v27, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v25, 0x3fb8aa3b, v25
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v54, v203 /*v459*/
	v_cndmask_b32_e64 v30, v28, 0xff61b1e6, s5
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v28, s4, v105
	v_mul_f32_e32 v26, 0x3fb8aa3b, v26
	v_mul_f32_e32 v24, 0x3fb8aa3b, v24
	s_and_b32 s5, s47, vcc_lo
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v31, v28, 0xff61b1e6, s5
	v_exp_f32_e32 v28, v26
	v_nop
	s_set_vgpr_msb 12
	v_sub_f32_e32 v26, v27, v104 /*v872*/
	v_exp_f32_e32 v24, v24
	v_cmp_gt_i32_e32 vcc_lo, v40, v241 /*v1009*/
	scratch_load_b64 v[104:105], off, off offset:6196 nv
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v26, 0x3fb8aa3b, v26
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_ge_i32_e32 vcc_lo, v40, v241 /*v1009*/
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v29, v26
	v_nop
	v_sub_f32_e32 v26, v30, v104 /*v872*/
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v26, 0x3fb8aa3b, v26
	v_exp_f32_e32 v30, v26
	v_nop
	s_set_vgpr_msb 12
	v_sub_f32_e32 v26, v31, v104 /*v872*/
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v26, 0x3fb8aa3b, v26
	v_exp_f32_e32 v31, v26
	v_nop
	v_pk_add_f32 v[26:27], v[182:183], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[26:27], v[22:23]
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v26, v22, v23
	v_pk_add_f32 v[22:23], v[184:185], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	v_dual_mul_f32 v24, s4, v192 :: v_dual_mul_f32 v25, s4, v193
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v27, v22, v23
	v_pk_add_f32 v[22:23], v[186:187], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[28:29]
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v28, v22, v23
	v_pk_add_f32 v[22:23], v[188:189], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_dual_mul_f32 v30, s4, v194 :: v_dual_mul_f32 v31, s4, v195
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v29, v22, v23
	v_dual_mul_f32 v22, s4, v190 :: v_dual_mul_f32 v23, s4, v191
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v41, v241 /*v1009*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	v_sub_f32_e32 v22, v22, v254 /*v1022*/
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v46, v241 /*v1009*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	v_sub_f32_e32 v23, v23, v254 /*v1022*/
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v47, v241 /*v1009*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v23, 0x3fb8aa3b, v23
	s_set_vgpr_msb 12
	v_sub_f32_e32 v24, v24, v254 /*v1022*/
	s_and_b32 s5, s47, vcc_lo
	v_sub_f32_e32 v25, v25, v254 /*v1022*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v241 /*v1009*/
	v_exp_f32_e32 v23, v23
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v25, 0x3fb8aa3b, v25
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v49, v241 /*v1009*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	v_sub_f32_e32 v30, v30, v254 /*v1022*/
	v_exp_f32_e32 v25, v25
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[22:23], v[34:35], v[22:23]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v31, v31, v254 /*v1022*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v241 /*v1009*/
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_pk_mul_f32 v[22:23], v[22:23], v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v31, 0x3fb8aa3b, v31
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v40, v8 /*v520*/
	s_set_vgpr_msb 0x800
	v_cvt_pk_bf16_f32 v34, v22, v23
	v_pk_add_f32 v[22:23], v[200:201], v[64:65] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	s_set_vgpr_msb 12
	v_sub_f32_e32 v32, v32, v254 /*v1022*/
	v_exp_f32_e32 v31, v31
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v33, v33, v254 /*v1022*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v30, 0x3fb8aa3b, v30
	v_mul_f32_e32 v32, 0x3fb8aa3b, v32
	s_set_vgpr_msb 8
	v_cmp_ge_i32_e32 vcc_lo, v40, v8 /*v520*/
	s_set_vgpr_msb 0x803
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v32, v32
	v_mul_f32_e32 v24, s4, v208
	v_cvt_pk_bf16_f32 v35, v22, v23
	v_pk_add_f32 v[22:23], v[202:203], v[64:65] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v33, v33
	v_mul_f32_e32 v25, s4, v209
	s_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_dual_mul_f32 v30, s4, v210 :: v_dual_mul_f32 v31, s4, v211
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v36, v22, v23
	v_pk_add_f32 v[22:23], v[204:205], v[64:65] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	v_mul_f32_e32 v32, s4, v212
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v37, v22, v23
	v_dual_mul_f32 v22, s4, v206 :: v_dual_mul_f32 v23, s4, v207
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v41, v8 /*v520*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 0x80c
	v_sub_f32_e32 v22, v22, v252 /*v1020*/
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc08
	v_cmp_gt_i32_e32 vcc_lo, v46, v8 /*v520*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	s_set_vgpr_msb 0x80c
	v_sub_f32_e32 v23, v23, v252 /*v1020*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v33, s4, v213 :: v_dual_mul_f32 v22, 0x3fb8aa3b, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v47, v8 /*v520*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	s_set_vgpr_msb 0x80c
	v_sub_f32_e32 v24, v24, v252 /*v1020*/
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	v_sub_f32_e32 v25, v25, v252 /*v1020*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v48, v8 /*v520*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v8 /*v520*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	s_set_vgpr_msb 0x80c
	v_sub_f32_e32 v30, v30, v252 /*v1020*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v31, v31, v252 /*v1020*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v54, v8 /*v520*/
	s_set_vgpr_msb 0x803
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	v_cvt_pk_bf16_f32 v42, v22, v23
	v_pk_add_f32 v[22:23], v[216:217], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	s_set_vgpr_msb 12
	v_sub_f32_e32 v32, v32, v252 /*v1020*/
	v_cmp_gt_i32_e32 vcc_lo, v40, v105 /*v873*/
	s_set_vgpr_msb 0xc00
	s_wait_loadcnt 0x5
	v_pk_add_f32 v[38:39], v[230:231], v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v33, v33, v252 /*v1020*/
	s_set_vgpr_msb 0xc03
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x30c
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v105 /*v873*/
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v43, v22, v23
	v_pk_add_f32 v[22:23], v[218:219], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v33, v33
	v_dual_mul_f32 v24, s4, v224 :: v_dual_mul_f32 v25, s4, v225
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_dual_mul_f32 v30, s4, v226 :: v_dual_mul_f32 v31, s4, v227
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v44, v22, v23
	v_pk_add_f32 v[22:23], v[220:221], v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	v_mul_f32_e32 v32, s4, v228
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v45, v22, v23
	v_dual_mul_f32 v22, s4, v222 :: v_dual_mul_f32 v23, s4, v223
	s_set_vgpr_msb 12
	v_wmma_f32_16x16x32_bf16 v[218:225], v[142:149], v[196:203] /*v[964:971]*/, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v41, v105 /*v873*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v22, v22, v158 /*v414*/
	s_set_vgpr_msb 0x409
	v_wmma_f32_16x16x32_bf16 v[218:225], v[178:185] /*v[434:441]*/, v[156:163] /*v[668:675]*/, v[218:225]
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x90c
	v_cmp_gt_i32_e32 vcc_lo, v46, v105 /*v873*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v23, v23, v158 /*v414*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v33, s4, v229 :: v_dual_mul_f32 v22, 0x3fb8aa3b, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v47, v105 /*v873*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v24, v24, v158 /*v414*/
	v_exp_f32_e32 v22, v22
	s_set_vgpr_msb 0x40d
	v_wmma_f32_16x16x32_bf16 v[218:225], v[162:169] /*v[418:425]*/, v[188:195] /*v[956:963]*/, v[218:225]
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xd04
	v_sub_f32_e32 v25, v25, v158 /*v414*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v105 /*v873*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v105 /*v873*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v30, v30, v158 /*v414*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v31, v31, v158 /*v414*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v105 /*v873*/
	v_pk_mul_f32 v[22:23], v[22:23], v[220:221] /*v[988:989]*/
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v50, v22, v23
	v_pk_add_f32 v[22:23], v[232:233], v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	s_set_vgpr_msb 4
	v_sub_f32_e32 v32, v32, v158 /*v414*/
	s_set_vgpr_msb 0x408
	v_cmp_gt_i32_e32 vcc_lo, v40, v43 /*v555*/
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x4
	v_pk_add_f32 v[38:39], v[246:247], v[80:81] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v33, v33, v158 /*v414*/
	s_set_vgpr_msb 0x403
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x308
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v43 /*v555*/
	s_set_vgpr_msb 0x800
	v_cvt_pk_bf16_f32 v51, v22, v23
	v_pk_add_f32 v[22:23], v[234:235], v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v33, v33
	v_dual_mul_f32 v24, s4, v240 :: v_dual_mul_f32 v25, s4, v241
	s_set_vgpr_msb 13
	v_wmma_f32_16x16x32_bf16 v[226:233], v[194:201] /*v[450:457]*/, v[242:249] /*v[1010:1017]*/, 0
	s_set_vgpr_msb 0xd00
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_dual_mul_f32 v30, s4, v242 :: v_dual_mul_f32 v31, s4, v243
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v52, v22, v23
	v_pk_add_f32 v[22:23], v[236:237], v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[226:233], v[186:193] /*v[442:449]*/, v[236:243] /*v[748:755]*/, v[226:233]
	s_set_vgpr_msb 0x900
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	v_mul_f32_e32 v32, s4, v244
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x30d
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[226:233], v[170:177] /*v[426:433]*/, v[106:113] /*v[874:881]*/, v[226:233]
	s_set_vgpr_msb 0xd00
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v53, v22, v23
	v_dual_mul_f32 v22, s4, v238 :: v_dual_mul_f32 v23, s4, v239
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v41, v43 /*v555*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	v_sub_f32_e32 v22, v22, v42 /*v554*/
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v46, v43 /*v555*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	v_sub_f32_e32 v23, v23, v42 /*v554*/
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v33, s4, v245 :: v_dual_mul_f32 v22, 0x3fb8aa3b, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v47, v43 /*v555*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	v_sub_f32_e32 v24, v24, v42 /*v554*/
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v25, v25, v42 /*v554*/
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v48, v43 /*v555*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v43 /*v555*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	v_sub_f32_e32 v30, v30, v42 /*v554*/
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(TRANS32_DEP_2)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 8
	v_sub_f32_e32 v31, v31, v42 /*v554*/
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v54, v43 /*v555*/
	s_set_vgpr_msb 0x803
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	v_cvt_pk_bf16_f32 v58, v22, v23
	v_pk_add_f32 v[22:23], v[248:249], v[80:81] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	s_set_vgpr_msb 8
	v_sub_f32_e32 v32, v32, v42 /*v554*/
	s_set_vgpr_msb 0x80c
	v_cmp_gt_i32_e32 vcc_lo, v40, v255 /*v1023*/
	s_set_vgpr_msb 0xc01
	s_wait_loadcnt 0x3
	v_pk_add_f32 v[38:39], v[6:7] /*v[262:263]*/, v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 8
	v_sub_f32_e32 v33, v33, v42 /*v554*/
	s_set_vgpr_msb 0x803
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x30c
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v255 /*v1023*/
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v59, v22, v23
	v_pk_add_f32 v[22:23], v[250:251], v[80:81] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v33, v33
	s_set_vgpr_msb 4
	v_dual_mul_f32 v24, s4, v0 /*v256*/ :: v_dual_mul_f32 v25, s4, v1 /*v257*/
	s_set_vgpr_msb 0x40d
	v_wmma_f32_16x16x32_bf16 v[242:249], v[194:201] /*v[450:457]*/, v[122:129] /*v[890:897]*/, 0
	s_set_vgpr_msb 0xd00
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v30, s4, v2 /*v258*/ :: v_dual_mul_f32 v31, s4, v3 /*v259*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v60, v22, v23
	v_pk_add_f32 v[22:23], v[252:253], v[80:81] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 13
	v_wmma_f32_16x16x32_bf16 v[242:249], v[186:193] /*v[442:449]*/, v[28:35] /*v[796:803]*/, v[242:249]
	s_set_vgpr_msb 0xd00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v32, s4, v4 /*v260*/ :: v_dual_mul_f32 v33, s4, v5 /*v261*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x309
	v_wmma_f32_16x16x32_bf16 v[242:249], v[170:177] /*v[426:433]*/, v[164:171] /*v[676:683]*/, v[242:249]
	s_set_vgpr_msb 0x900
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v61, v22, v23
	v_mul_f32_e32 v22, s4, v254
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v41, v255 /*v1023*/
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v22, v22, v250 /*v1018*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v23, s4, v255
	s_set_vgpr_msb 12
	v_wmma_f32_16x16x32_bf16 v[250:257], v[142:149], v[154:161] /*v[922:929]*/, 0
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v46, v255 /*v1023*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	v_exp_f32_e32 v22, v22
	v_sub_f32_e32 v23, v23, v250 /*v1018*/
	s_set_vgpr_msb 0xc0d
	v_wmma_f32_16x16x32_bf16 v[250:257], v[178:185] /*v[434:441]*/, v[36:43] /*v[804:811]*/, v[250:257]
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xd0c
	v_cmp_gt_i32_e32 vcc_lo, v47, v255 /*v1023*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	v_sub_f32_e32 v24, v24, v250 /*v1018*/
	s_and_b32 s5, s47, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v25, v25, v250 /*v1018*/
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v255 /*v1023*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc09
	v_wmma_f32_16x16x32_bf16 v[250:257], v[162:169] /*v[418:425]*/, v[196:203] /*v[708:715]*/, v[250:257]
	s_set_vgpr_msb 0x90c
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v255 /*v1023*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	v_sub_f32_e32 v30, v30, v250 /*v1018*/
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_2)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v31, v31, v250 /*v1018*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v255 /*v1023*/
	v_pk_mul_f32 v[22:23], v[22:23], v[220:221] /*v[988:989]*/
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v66, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[8:9] /*v[264:265]*/, v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	v_sub_f32_e32 v32, v32, v250 /*v1018*/
	v_cmp_gt_i32_e32 vcc_lo, v40, v251 /*v1019*/
	s_set_vgpr_msb 0xc01
	s_wait_loadcnt 0x1
	v_pk_add_f32 v[38:39], v[22:23] /*v[278:279]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v33, v33, v250 /*v1018*/
	s_set_vgpr_msb 0xc03
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x30c
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v251 /*v1019*/
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v67, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[10:11] /*v[266:267]*/, v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x104
	v_exp_f32_e32 v33, v33
	v_dual_mul_f32 v24, s4, v16 /*v272*/ :: v_dual_mul_f32 v25, s4, v17 /*v273*/
	s_set_vgpr_msb 0x44d
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[194:201] /*v[450:457]*/, v[204:211] /*v[972:979]*/, 0
	s_set_vgpr_msb 0x4d00
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v30, s4, v18 /*v274*/ :: v_dual_mul_f32 v31, s4, v19 /*v275*/
	s_set_vgpr_msb 0x4c0
	s_clause 0x1
	scratch_load_b128 v[204:207] /*v[972:975]*/, off, off offset:5440 th:TH_LOAD_LU nv
	scratch_load_b128 v[208:211] /*v[976:979]*/, off, off offset:5456 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc003
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v68, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[12:13] /*v[268:269]*/, v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x15d
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[186:193] /*v[442:449]*/, v[68:75] /*v[836:843]*/, v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5d00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v32, s4, v20 /*v276*/ :: v_dual_mul_f32 v33, s4, v21 /*v277*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x35d
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[170:177] /*v[426:433]*/, v[172:179] /*v[940:947]*/, v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5d00
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v69, v22, v23
	s_set_vgpr_msb 4
	v_dual_mul_f32 v22, s4, v14 /*v270*/ :: v_dual_mul_f32 v23, s4, v15 /*v271*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v41, v251 /*v1019*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc4c
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[142:149], v[180:187] /*v[948:955]*/, 0
	s_set_vgpr_msb 0x4c0c
	v_sub_f32_e32 v22, v22, v94 /*v862*/
	s_set_vgpr_msb 0xccc
	s_clause 0x1
	scratch_load_b128 v[180:183] /*v[948:951]*/, off, off offset:6000 nv
	scratch_load_b128 v[184:187] /*v[952:955]*/, off, off offset:6016 nv
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v46, v251 /*v1019*/
	s_set_vgpr_msb 0xcc0c
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	v_sub_f32_e32 v23, v23, v94 /*v862*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	s_set_vgpr_msb 0x5d
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[178:185] /*v[434:441]*/, v[76:83] /*v[844:851]*/, v[10:17] /*v[266:273]*/
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x5d0c
	v_cmp_gt_i32_e32 vcc_lo, v47, v251 /*v1019*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	v_sub_f32_e32 v24, v24, v94 /*v862*/
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v25, v25, v94 /*v862*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v251 /*v1019*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc51
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[162:169] /*v[418:425]*/, v[2:9], v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x510c
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v251 /*v1019*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	v_sub_f32_e32 v30, v30, v94 /*v862*/
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3232 nv
	scratch_load_b128 v[6:9], off, off offset:3248 nv
	s_set_vgpr_msb 0xccd
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[922:929]*/, v[194:201] /*v[450:457]*/, v[230:237] /*v[998:1005]*/, 0
	s_set_vgpr_msb 0xcd00
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v31, v31, v94 /*v862*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v251 /*v1019*/
	v_pk_mul_f32 v[22:23], v[22:23], v[220:221] /*v[988:989]*/
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v74, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[24:25] /*v[280:281]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	v_sub_f32_e32 v32, v32, v94 /*v862*/
	v_cmp_gt_i32_e32 vcc_lo, v40, v239 /*v1007*/
	s_set_vgpr_msb 0xc01
	s_wait_loadcnt 0x6
	v_pk_add_f32 v[38:39], v[38:39] /*v[294:295]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v33, v33, v94 /*v862*/
	s_set_vgpr_msb 0xc03
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x30c
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v239 /*v1007*/
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v75, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[26:27] /*v[282:283]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x104
	v_exp_f32_e32 v33, v33
	v_dual_mul_f32 v24, s4, v32 /*v288*/ :: v_dual_mul_f32 v25, s4, v33 /*v289*/
	s_set_vgpr_msb 0x44d
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[194:201] /*v[450:457]*/, v[212:219] /*v[980:987]*/, 0
	s_set_vgpr_msb 0x4d00
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v30, s4, v34 /*v290*/ :: v_dual_mul_f32 v31, s4, v35 /*v291*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v76, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[28:29] /*v[284:285]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x15d
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[186:193] /*v[442:449]*/, v[84:91] /*v[852:859]*/, v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5d00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v32, s4, v36 /*v292*/ :: v_dual_mul_f32 v33, s4, v37 /*v293*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x35d
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[170:177] /*v[426:433]*/, v[52:59] /*v[820:827]*/, v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5d00
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v77, v22, v23
	s_set_vgpr_msb 4
	v_dual_mul_f32 v22, s4, v30 /*v286*/ :: v_dual_mul_f32 v23, s4, v31 /*v287*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v41, v239 /*v1007*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 0xcf9
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[922:929]*/, v[186:193] /*v[442:449]*/, v[124:131] /*v[636:643]*/, v[154:161] /*v[922:929]*/
	s_set_vgpr_msb 0xf90c
	v_sub_f32_e32 v22, v22, v6 /*v774*/
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v46, v239 /*v1007*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	v_sub_f32_e32 v23, v23, v6 /*v774*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	s_set_vgpr_msb 0xf5
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[922:929]*/, v[170:177] /*v[426:433]*/, v[130:137] /*v[386:393]*/, v[154:161] /*v[922:929]*/
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xf50c
	v_cmp_gt_i32_e32 vcc_lo, v47, v239 /*v1007*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	v_sub_f32_e32 v24, v24, v6 /*v774*/
	v_exp_f32_e32 v22, v22
	s_set_vgpr_msb 0xc40
	s_clause 0x1
	scratch_load_b128 v[130:133] /*v[386:389]*/, off, off offset:4864 th:TH_LOAD_LU nv
	scratch_load_b128 v[134:137] /*v[390:393]*/, off, off offset:4880 th:TH_LOAD_LU nv
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x400c
	v_sub_f32_e32 v25, v25, v6 /*v774*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v239 /*v1007*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v239 /*v1007*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	v_sub_f32_e32 v30, v30, v6 /*v774*/
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_2)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v31, v31, v6 /*v774*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v239 /*v1007*/
	v_pk_mul_f32 v[22:23], v[22:23], v[220:221] /*v[988:989]*/
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v82, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[40:41] /*v[296:297]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	v_sub_f32_e32 v32, v32, v6 /*v774*/
	s_set_vgpr_msb 0xc04
	v_cmp_gt_i32_e32 vcc_lo, v40, v205 /*v461*/
	s_set_vgpr_msb 0x405
	v_pk_add_f32 v[38:39], v[54:55] /*v[310:311]*/, v[154:155] /*v[410:411]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v33, v33, v6 /*v774*/
	s_set_vgpr_msb 0xc03
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x304
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v205 /*v461*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v83, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[42:43] /*v[298:299]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x104
	v_exp_f32_e32 v33, v33
	v_dual_mul_f32 v24, s4, v48 /*v304*/ :: v_dual_mul_f32 v25, s4, v49 /*v305*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v30, s4, v50 /*v306*/ :: v_dual_mul_f32 v31, s4, v51 /*v307*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v84, v22, v23
	s_set_vgpr_msb 1
	v_pk_add_f32 v[22:23], v[44:45] /*v[300:301]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v32, s4, v52 /*v308*/ :: v_dual_mul_f32 v33, s4, v53 /*v309*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v85, v22, v23
	s_set_vgpr_msb 4
	v_dual_mul_f32 v22, s4, v46 /*v302*/ :: v_dual_mul_f32 v23, s4, v47 /*v303*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v41, v205 /*v461*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_sub_f32_e32 v22, v22, v156 /*v412*/
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v46, v205 /*v461*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	v_sub_f32_e32 v23, v23, v156 /*v412*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v47, v205 /*v461*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	v_sub_f32_e32 v24, v24, v156 /*v412*/
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v25, v25, v156 /*v412*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v48, v205 /*v461*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v205 /*v461*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	v_sub_f32_e32 v30, v30, v156 /*v412*/
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(TRANS32_DEP_2)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v31, v31, v156 /*v412*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v54, v205 /*v461*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	v_cvt_pk_bf16_f32 v90, v22, v23
	s_set_vgpr_msb 5
	v_pk_add_f32 v[22:23], v[56:57] /*v[312:313]*/, v[154:155] /*v[410:411]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x504
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	v_sub_f32_e32 v32, v32, v156 /*v412*/
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v40, v7 /*v775*/
	s_set_vgpr_msb 0xc0d
	v_pk_add_f32 v[38:39], v[70:71] /*v[326:327]*/, v[10:11] /*v[778:779]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd00
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v33, v33, v156 /*v412*/
	s_set_vgpr_msb 0x403
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x30c
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v7 /*v775*/
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v91, v22, v23
	s_set_vgpr_msb 5
	v_pk_add_f32 v[22:23], v[58:59] /*v[314:315]*/, v[154:155] /*v[410:411]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x504
	v_exp_f32_e32 v33, v33
	v_dual_mul_f32 v24, s4, v64 /*v320*/ :: v_dual_mul_f32 v25, s4, v65 /*v321*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v30, s4, v66 /*v322*/ :: v_dual_mul_f32 v31, s4, v67 /*v323*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v92, v22, v23
	s_set_vgpr_msb 5
	v_pk_add_f32 v[22:23], v[60:61] /*v[316:317]*/, v[154:155] /*v[410:411]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x500
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v32, s4, v68 /*v324*/ :: v_dual_mul_f32 v33, s4, v69 /*v325*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v93, v22, v23
	s_set_vgpr_msb 4
	v_dual_mul_f32 v22, s4, v62 /*v318*/ :: v_dual_mul_f32 v23, s4, v63 /*v319*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v41, v7 /*v775*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	v_sub_f32_e32 v22, v22, v170 /*v938*/
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v46, v7 /*v775*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	v_sub_f32_e32 v23, v23, v170 /*v938*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v47, v7 /*v775*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	v_sub_f32_e32 v24, v24, v170 /*v938*/
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v25, v25, v170 /*v938*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v7 /*v775*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v7 /*v775*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	v_sub_f32_e32 v30, v30, v170 /*v938*/
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_2)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v31, v31, v170 /*v938*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v7 /*v775*/
	v_pk_mul_f32 v[22:23], v[22:23], v[220:221] /*v[988:989]*/
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v98, v22, v23
	s_set_vgpr_msb 13
	v_pk_add_f32 v[22:23], v[72:73] /*v[328:329]*/, v[10:11] /*v[778:779]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	v_sub_f32_e32 v32, v32, v170 /*v938*/
	v_cmp_gt_i32_e32 vcc_lo, v40, v95 /*v863*/
	s_set_vgpr_msb 0xc0d
	v_pk_add_f32 v[38:39], v[86:87] /*v[342:343]*/, v[8:9] /*v[776:777]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd00
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v33, v33, v170 /*v938*/
	s_set_vgpr_msb 0xc03
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x30c
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v95 /*v863*/
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v99, v22, v23
	s_set_vgpr_msb 13
	v_pk_add_f32 v[22:23], v[74:75] /*v[330:331]*/, v[10:11] /*v[778:779]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd04
	v_exp_f32_e32 v33, v33
	v_dual_mul_f32 v24, s4, v80 /*v336*/ :: v_dual_mul_f32 v25, s4, v81 /*v337*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v30, s4, v82 /*v338*/ :: v_dual_mul_f32 v31, s4, v83 /*v339*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v100, v22, v23
	s_set_vgpr_msb 13
	v_pk_add_f32 v[22:23], v[76:77] /*v[332:333]*/, v[10:11] /*v[778:779]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v32, s4, v84 /*v340*/ :: v_dual_mul_f32 v33, s4, v85 /*v341*/
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v101, v22, v23
	s_set_vgpr_msb 4
	v_dual_mul_f32 v22, s4, v78 /*v334*/ :: v_dual_mul_f32 v23, s4, v79 /*v335*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v41, v95 /*v863*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v22, v22, v160 /*v416*/
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v46, v95 /*v863*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v23, v23, v160 /*v416*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v47, v95 /*v863*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v24, v24, v160 /*v416*/
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	v_sub_f32_e32 v25, v25, v160 /*v416*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v95 /*v863*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v95 /*v863*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v30, v30, v160 /*v416*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v31, v31, v160 /*v416*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v95 /*v863*/
	v_pk_mul_f32 v[22:23], v[22:23], v[220:221] /*v[988:989]*/
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v106, v22, v23
	s_set_vgpr_msb 13
	v_pk_add_f32 v[22:23], v[88:89] /*v[344:345]*/, v[8:9] /*v[776:777]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd04
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	v_sub_f32_e32 v32, v32, v160 /*v416*/
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v40, v171 /*v939*/
	v_pk_add_f32 v[38:39], v[174:175], v[92:93] /*v[860:861]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v33, v33, v160 /*v416*/
	s_set_vgpr_msb 0x403
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x30c
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v171 /*v939*/
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v107, v22, v23
	s_set_vgpr_msb 13
	v_pk_add_f32 v[22:23], v[90:91] /*v[346:347]*/, v[8:9] /*v[776:777]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd00
	v_exp_f32_e32 v33, v33
	v_dual_mul_f32 v24, s4, v112 :: v_dual_mul_f32 v25, s4, v113
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_dual_mul_f32 v30, s4, v114 :: v_dual_mul_f32 v31, s4, v115
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v108, v22, v23
	s_set_vgpr_msb 13
	v_pk_add_f32 v[22:23], v[92:93] /*v[348:349]*/, v[8:9] /*v[776:777]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xd00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	v_mul_f32_e32 v32, s4, v116
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v109, v22, v23
	v_dual_mul_f32 v22, s4, v110 :: v_dual_mul_f32 v23, s4, v111
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v41, v171 /*v939*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	v_sub_f32_e32 v22, v22, v238 /*v1006*/
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v46, v171 /*v939*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	v_sub_f32_e32 v23, v23, v238 /*v1006*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v33, s4, v117 :: v_dual_mul_f32 v22, 0x3fb8aa3b, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v47, v171 /*v939*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	v_sub_f32_e32 v24, v24, v238 /*v1006*/
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v25, v25, v238 /*v1006*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v171 /*v939*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v171 /*v939*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	v_sub_f32_e32 v30, v30, v238 /*v1006*/
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(TRANS32_DEP_2)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v31, v31, v238 /*v1006*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v171 /*v939*/
	v_pk_mul_f32 v[22:23], v[22:23], v[220:221] /*v[988:989]*/
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v114, v22, v23
	s_set_vgpr_msb 12
	v_pk_add_f32 v[22:23], v[176:177], v[92:93] /*v[860:861]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	v_sub_f32_e32 v32, v32, v238 /*v1006*/
	s_set_vgpr_msb 0xc08
	v_cmp_gt_i32_e32 vcc_lo, v40, v9 /*v521*/
	v_pk_add_f32 v[38:39], v[166:167], v[244:245] /*v[756:757]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v33, v33, v238 /*v1006*/
	s_set_vgpr_msb 0xc03
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x308
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v9 /*v521*/
	s_set_vgpr_msb 0x800
	v_cvt_pk_bf16_f32 v115, v22, v23
	s_set_vgpr_msb 12
	v_pk_add_f32 v[22:23], v[178:179], v[92:93] /*v[860:861]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v33, v33
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v24, s4, v120 :: v_dual_mul_f32 v25, s4, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_dual_mul_f32 v30, s4, v122 :: v_dual_mul_f32 v31, s4, v123
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v116, v22, v23
	s_set_vgpr_msb 12
	v_pk_add_f32 v[22:23], v[180:181], v[92:93] /*v[860:861]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	v_mul_f32_e32 v32, s4, v124
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v117, v22, v23
	v_dual_mul_f32 v22, s4, v118 :: v_dual_mul_f32 v23, s4, v119
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	v_mul_f32_e32 v33, s4, v125
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v41, v9 /*v521*/
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v22, v22, v0 :: v_dual_sub_f32 v23, v23, v0
	s_and_b32 s5, s47, vcc_lo
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v46, v9 /*v521*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	s_set_vgpr_msb 0x800
	v_mul_f32_e32 v23, 0x3fb8aa3b, v23
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v47, v9 /*v521*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	s_set_vgpr_msb 0x800
	v_sub_f32_e32 v24, v24, v0
	v_exp_f32_e32 v23, v23
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v48, v9 /*v521*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_set_vgpr_msb 0x800
	v_sub_f32_e32 v25, v25, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	v_sub_f32_e32 v30, v30, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v24, 0x3fb8aa3b, v24 :: v_dual_mul_f32 v25, 0x3fb8aa3b, v25
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x308
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_cmp_gt_i32_e32 vcc_lo, v49, v9 /*v521*/
	s_set_vgpr_msb 0x800
	v_cvt_pk_bf16_f32 v122, v22, v23
	s_set_vgpr_msb 8
	v_pk_add_f32 v[22:23], v[168:169], v[244:245] /*v[756:757]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v30, 0x3fb8aa3b, v30 :: v_dual_sub_f32 v31, v31, v0
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v54, v9 /*v521*/
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_mul_f32_e32 v31, 0x3fb8aa3b, v31
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	v_sub_f32_e32 v32, v32, v0
	v_exp_f32_e32 v31, v31
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v40, v253 /*v1021*/
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v123, v22, v23
	s_set_vgpr_msb 8
	v_pk_add_f32 v[22:23], v[170:171], v[244:245] /*v[756:757]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x800
	v_sub_f32_e32 v33, v33, v0
	v_mul_f32_e32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_ge_i32_e32 vcc_lo, v40, v253 /*v1021*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	v_exp_f32_e32 v32, v32
	v_dual_mul_f32 v24, s4, v128 :: v_dual_mul_f32 v25, s4, v129
	v_mul_f32_e32 v30, s4, v130
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v33, v33
	v_mul_f32_e32 v31, s4, v131
	v_pk_add_f32 v[38:39], v[158:159], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v124, v22, v23
	s_set_vgpr_msb 8
	v_pk_add_f32 v[22:23], v[172:173], v[244:245] /*v[756:757]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	v_mul_f32_e32 v32, s4, v132
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v125, v22, v23
	v_dual_mul_f32 v22, s4, v126 :: v_dual_mul_f32 v23, s4, v127
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v41, v253 /*v1021*/
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v22, v22, v204 /*v460*/
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v46, v253 /*v1021*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v23, v23, v204 /*v460*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v33, s4, v133 :: v_dual_mul_f32 v22, 0x3fb8aa3b, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v47, v253 /*v1021*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v24, v24, v204 /*v460*/
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	v_sub_f32_e32 v25, v25, v204 /*v460*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v253 /*v1021*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v253 /*v1021*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v30, v30, v204 /*v460*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v31, v31, v204 /*v460*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v253 /*v1021*/
	v_pk_mul_f32 v[22:23], v[22:23], v[220:221] /*v[988:989]*/
	v_exp_f32_e32 v25, v25
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v130, v22, v23
	v_pk_add_f32 v[22:23], v[160:161], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	s_set_vgpr_msb 4
	v_sub_f32_e32 v32, v32, v204 /*v460*/
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v40, v240 /*v1008*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v33, v33, v204 /*v460*/
	s_set_vgpr_msb 0x403
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s5, s47, vcc_lo
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x30c
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v40, v240 /*v1008*/
	s_set_vgpr_msb 0xc00
	v_cvt_pk_bf16_f32 v131, v22, v23
	v_pk_add_f32 v[22:23], v[162:163], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v33, v33
	v_dual_mul_f32 v24, s4, v136 :: v_dual_mul_f32 v25, s4, v137
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	v_dual_mul_f32 v30, s4, v138 :: v_dual_mul_f32 v31, s4, v139
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v132, v22, v23
	v_pk_add_f32 v[22:23], v[164:165], v[78:79] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	v_mul_f32_e32 v32, s4, v140
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v133, v22, v23
	v_dual_mul_f32 v22, s4, v134 :: v_dual_mul_f32 v23, s4, v135
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s5
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v41, v240 /*v1008*/
	scratch_load_b64 v[40:41], off, off offset:5544 nv
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v22, v22, v206 /*v462*/
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v46, v240 /*v1008*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v23, v23, v206 /*v462*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v33, s4, v141 :: v_dual_mul_f32 v22, 0x3fb8aa3b, v22
	s_and_b32 s5, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v47, v240 /*v1008*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v24, v24, v206 /*v462*/
	v_exp_f32_e32 v22, v22
	s_and_b32 s5, s47, vcc_lo
	v_sub_f32_e32 v25, v25, v206 /*v462*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v48, v240 /*v1008*/
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_and_b32 s5, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v49, v240 /*v1008*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v30, v30, v206 /*v462*/
	s_and_b32 s5, s47, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v31, v31, v206 /*v462*/
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v54, v240 /*v1008*/
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v30, v30
	s_and_b32 s5, s47, vcc_lo
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s5
	s_set_vgpr_msb 0xc04
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_sub_f32 v32, v32, v206 /*v462*/ :: v_dual_sub_f32 v33, v33, v206 /*v462*/
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v33, v33
	s_wait_loadcnt 0x0
	v_pk_add_f32 v[38:39], v[150:151], v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 1
	v_mov_b64_e32 v[38:39], v[154:155] /*v[410:411]*/
	s_set_vgpr_msb 0x103
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v138, v22, v23
	v_pk_add_f32 v[22:23], v[152:153], v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[24:25]
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_1)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v139, v22, v23
	v_pk_add_f32 v[22:23], v[154:155], v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[30:31]
	s_set_vgpr_msb 1
	v_dual_mov_b32 v30, v156 /*v412*/ :: v_dual_mov_b32 v31, v157 /*v413*/
	s_set_vgpr_msb 0x103
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v140, v22, v23
	v_pk_add_f32 v[22:23], v[156:157], v[40:41] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 1
	v_mov_b32_e32 v40, v160 /*v416*/
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3584 nv
	scratch_load_b128 v[154:157], off, off offset:3600 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[194:201] /*v[450:457]*/, v[150:157], 0
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33]
	s_set_vgpr_msb 1
	v_mov_b32_e32 v32, v158 /*v414*/
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3264 nv
	scratch_load_b128 v[154:157], off, off offset:3280 nv
	s_set_vgpr_msb 0x103
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v141, v22, v23
	s_set_vgpr_msb 0x44
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[142:149], v[138:145] /*v[394:401]*/, 0
	s_set_vgpr_msb 0x4404
	scratch_load_b32 v22, off, off offset:6316 nv
	s_wait_loadcnt 0x0
	ds_store_b128 v22, v[106:109] /*v[362:365]*/ offset:192
	ds_store_b128 v22, v[110:113] /*v[366:369]*/ offset:224
	s_set_vgpr_msb 0x45d
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[178:185] /*v[434:441]*/, v[44:51] /*v[812:819]*/, v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5d00
	ds_store_b128 v22, v[142:145]
	ds_store_b128 v22, v[146:149] offset:32
	s_set_vgpr_msb 4
	ds_store_b128 v22, v[178:181] /*v[434:437]*/ offset:64
	ds_store_b128 v22, v[182:185] /*v[438:441]*/ offset:96
	ds_store_b128 v22, v[162:165] /*v[418:421]*/ offset:128
	ds_store_b128 v22, v[166:169] /*v[422:425]*/ offset:160
	s_set_vgpr_msb 0x459
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[162:169] /*v[418:425]*/, v[116:123] /*v[628:635]*/, v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5951
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[106:113] /*v[362:369]*/, v[2:9], v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5104
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3136 nv
	scratch_load_b128 v[6:9], off, off offset:3152 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_dual_mul_f32 v22, s4, v154 /*v410*/ :: v_dual_mul_f32 v23, s4, v155 /*v411*/
	s_set_vgpr_msb 0x40d
	v_wmma_f32_16x16x32_bf16 v[178:185], v[186:193] /*v[442:449]*/, v[60:67] /*v[828:835]*/, v[178:185]
	s_set_vgpr_msb 0xd04
	v_dual_mul_f32 v24, s4, v156 /*v412*/ :: v_dual_mul_f32 v25, s4, v157 /*v413*/
	s_set_vgpr_msb 0x440
	v_dual_mov_b32 v157 /*v413*/, v31 :: v_dual_mov_b32 v156 /*v412*/, v30
	s_set_vgpr_msb 0x4004
	v_dual_mul_f32 v30, s4, v158 /*v414*/ :: v_dual_mul_f32 v31, s4, v159 /*v415*/
	s_set_vgpr_msb 0x440
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[38:39]
	s_set_vgpr_msb 0x400d
	v_wmma_f32_16x16x32_bf16 v[178:185], v[170:177] /*v[426:433]*/, v[130:137] /*v[898:905]*/, v[178:185]
	s_set_vgpr_msb 0xd40
	v_mov_b32_e32 v158 /*v414*/, v32
	s_set_vgpr_msb 0x4004
	v_dual_mul_f32 v32, s4, v160 /*v416*/ :: v_dual_mul_f32 v33, s4, v161 /*v417*/
	s_set_vgpr_msb 0x440
	v_mov_b32_e32 v160 /*v416*/, v40
	s_set_vgpr_msb 0x4001
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[178:185], v[98:105] /*v[354:361]*/, v[2:9], v[178:185]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4480 nv
	scratch_load_b128 v[6:9], off, off offset:4496 nv
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[38:39], v[178:179], v[62:63] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 64
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[142:149], v[150:157], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3296 nv
	scratch_load_b128 v[154:157], off, off offset:3312 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[178:185] /*v[434:441]*/, v[84:91] /*v[596:603]*/, v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5980
	s_clause 0x1
	scratch_load_b128 v[84:87] /*v[596:599]*/, off, off offset:3552 th:TH_LOAD_LU nv
	scratch_load_b128 v[88:91] /*v[600:603]*/, off, off offset:3568 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8059
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[162:169] /*v[418:425]*/, v[10:17] /*v[522:529]*/, v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5980
	s_clause 0x1
	scratch_load_b128 v[10:13] /*v[522:525]*/, off, off offset:5472 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17] /*v[526:529]*/, off, off offset:5488 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8051
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[106:113] /*v[362:369]*/, v[2:9], v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5104
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3168 nv
	scratch_load_b128 v[6:9], off, off offset:3184 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_dual_mul_f32 v40, s4, v144 /*v400*/ :: v_dual_mul_f32 v41, s4, v145 /*v401*/
	s_set_vgpr_msb 0x401
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[162:169], v[194:201] /*v[450:457]*/, v[150:157], 0
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3328 nv
	scratch_load_b128 v[154:157], off, off offset:3344 nv
	s_set_vgpr_msb 0x10d
	v_wmma_f32_16x16x32_bf16 v[162:169], v[186:193] /*v[442:449]*/, v[138:145] /*v[906:913]*/, v[162:169]
	s_set_vgpr_msb 0xd05
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[162:169], v[170:177] /*v[426:433]*/, v[208:215] /*v[464:471]*/, v[162:169]
	s_set_vgpr_msb 0x501
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[162:169], v[98:105] /*v[354:361]*/, v[2:9], v[162:169]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3200 nv
	scratch_load_b128 v[6:9], off, off offset:3216 nv
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[46:47], v[162:163], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[194:201], v[142:149], v[150:157], 0
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3360 nv
	scratch_load_b128 v[154:157], off, off offset:3376 nv
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[194:201], v[178:185] /*v[434:441]*/, v[100:107] /*v[612:619]*/, v[194:201]
	s_set_vgpr_msb 0x905
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[194:201], v[162:169] /*v[418:425]*/, v[224:231] /*v[480:487]*/, v[194:201]
	s_set_vgpr_msb 0x501
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[194:201], v[106:113] /*v[362:369]*/, v[2:9], v[194:201]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3040 nv
	scratch_load_b128 v[6:9], off, off offset:3056 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_mul_f32_e32 v48, s4, v200
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[210:217], v[194:201] /*v[450:457]*/, v[150:157], 0
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3072 nv
	scratch_load_b128 v[154:157], off, off offset:3088 nv
	s_set_vgpr_msb 0x109
	v_wmma_f32_16x16x32_bf16 v[210:217], v[186:193] /*v[442:449]*/, v[60:67] /*v[572:579]*/, v[210:217]
	s_set_vgpr_msb 0x980
	s_clause 0x1
	scratch_load_b128 v[60:63] /*v[572:575]*/, off, off offset:5504 th:TH_LOAD_LU nv
	scratch_load_b128 v[64:67] /*v[576:579]*/, off, off offset:5520 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8005
	v_wmma_f32_16x16x32_bf16 v[210:217], v[170:177] /*v[426:433]*/, v[240:247] /*v[496:503]*/, v[210:217]
	s_set_vgpr_msb 0x501
	s_wait_loadcnt 0x4
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[210:217], v[98:105] /*v[354:361]*/, v[2:9], v[210:217]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3104 nv
	scratch_load_b128 v[6:9], off, off offset:3120 nv
	s_set_vgpr_msb 0x140
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[142:149], v[150:157], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3392 nv
	scratch_load_b128 v[154:157], off, off offset:3408 nv
	s_set_vgpr_msb 0x5d
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[178:185] /*v[434:441]*/, v[222:229] /*v[990:997]*/, v[90:97] /*v[346:353]*/
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[162:169] /*v[418:425]*/, v[96:103] /*v[864:871]*/, v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5d51
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[106:113] /*v[362:369]*/, v[2:9], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5100
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3616 nv
	scratch_load_b128 v[6:9], off, off offset:3632 nv
	s_set_vgpr_msb 0x41
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[194:201] /*v[450:457]*/, v[150:157], 0
	s_set_vgpr_msb 0x4100
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3648 nv
	scratch_load_b128 v[154:157], off, off offset:3664 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[186:193] /*v[442:449]*/, v[254:261] /*v[766:773]*/, v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x595d
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[170:177] /*v[426:433]*/, v[146:153] /*v[914:921]*/, v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5d51
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[98:105] /*v[354:361]*/, v[2:9], v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5100
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3680 nv
	scratch_load_b128 v[6:9], off, off offset:3696 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[142:149], v[150:157], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3712 nv
	scratch_load_b128 v[154:157], off, off offset:3728 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[178:185] /*v[434:441]*/, v[148:155] /*v[660:667]*/, v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x595d
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[162:169] /*v[418:425]*/, v[162:169] /*v[930:937]*/, v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5d51
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[106:113] /*v[362:369]*/, v[2:9], v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5100
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3744 nv
	scratch_load_b128 v[6:9], off, off offset:3760 nv
	s_set_vgpr_msb 0x41
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[194:201] /*v[450:457]*/, v[150:157], 0
	s_set_vgpr_msb 0x4100
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3776 nv
	scratch_load_b128 v[154:157], off, off offset:3792 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[186:193] /*v[442:449]*/, v[68:75] /*v[580:587]*/, v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x595d
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[170:177] /*v[426:433]*/, v[12:19] /*v[780:787]*/, v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5d51
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[98:105] /*v[354:361]*/, v[2:9], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5100
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3808 nv
	scratch_load_b128 v[6:9], off, off offset:3824 nv
	s_set_vgpr_msb 64
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[142:149], v[150:157], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3840 nv
	scratch_load_b128 v[154:157], off, off offset:3856 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[178:185] /*v[434:441]*/, v[172:179] /*v[684:691]*/, v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5951
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[162:169] /*v[418:425]*/, v[10:17], v[58:65] /*v[314:321]*/
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[106:113] /*v[362:369]*/, v[2:9], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5100
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3872 nv
	scratch_load_b128 v[6:9], off, off offset:3888 nv
	s_set_vgpr_msb 0x41
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[194:201] /*v[450:457]*/, v[150:157], 0
	s_set_vgpr_msb 0x4100
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3904 nv
	scratch_load_b128 v[154:157], off, off offset:3920 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[186:193] /*v[442:449]*/, v[188:195] /*v[700:707]*/, v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5955
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[170:177] /*v[426:433]*/, v[114:121] /*v[370:377]*/, v[50:57] /*v[306:313]*/
	s_clause 0x1
	scratch_load_b128 v[114:117] /*v[370:373]*/, off, off offset:4896 th:TH_LOAD_LU nv
	scratch_load_b128 v[118:121] /*v[374:377]*/, off, off offset:4912 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x5551
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[98:105] /*v[354:361]*/, v[2:9], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5101
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:3936 nv
	scratch_load_b128 v[6:9], off, off offset:3952 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[78:79], v[50:51] /*v[306:307]*/, v[80:81] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x140
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[142:149], v[150:157], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:3968 nv
	scratch_load_b128 v[154:157], off, off offset:3984 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[178:185] /*v[434:441]*/, v[204:211] /*v[716:723]*/, v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5955
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[162:169] /*v[418:425]*/, v[122:129] /*v[378:385]*/, v[42:49] /*v[298:305]*/
	s_clause 0x1
	scratch_load_b128 v[122:125] /*v[378:381]*/, off, off offset:4800 th:TH_LOAD_LU nv
	scratch_load_b128 v[126:129] /*v[382:385]*/, off, off offset:4816 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x5551
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[106:113] /*v[362:369]*/, v[2:9], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5100
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4000 nv
	scratch_load_b128 v[6:9], off, off offset:4016 nv
	s_set_vgpr_msb 0x41
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[194:201] /*v[450:457]*/, v[150:157], 0
	s_set_vgpr_msb 0x4100
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:5616 nv
	scratch_load_b128 v[154:157], off, off offset:5632 nv
	s_set_vgpr_msb 0x59
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[186:193] /*v[442:449]*/, v[220:227] /*v[732:739]*/, v[34:41] /*v[290:297]*/
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[170:177] /*v[426:433]*/, v[228:235] /*v[740:747]*/, v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5951
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[98:105] /*v[354:361]*/, v[2:9], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5101
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4032 nv
	scratch_load_b128 v[6:9], off, off offset:4048 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[86:87], v[34:35] /*v[290:291]*/, v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x140
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[142:149], v[150:157], 0
	s_set_vgpr_msb 0x4000
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:5680 nv
	scratch_load_b128 v[154:157], off, off offset:5696 nv
	s_set_vgpr_msb 0x5d
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[178:185] /*v[434:441]*/, v[114:121] /*v[882:889]*/, v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5d59
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[162:169] /*v[418:425]*/, v[246:253] /*v[758:765]*/, v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5951
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[106:113] /*v[362:369]*/, v[2:9], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5100
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4064 nv
	scratch_load_b128 v[6:9], off, off offset:4080 nv
	s_set_vgpr_msb 0x51
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[98:105] /*v[354:361]*/, v[2:9], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5101
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4096 nv
	scratch_load_b128 v[6:9], off, off offset:4112 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[94:95], v[18:19] /*v[274:275]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x151
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[106:113] /*v[362:369]*/, v[2:9], v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x5100
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4128 nv
	scratch_load_b128 v[6:9], off, off offset:4144 nv
	s_set_vgpr_msb 0x51
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[98:105] /*v[354:361]*/, v[2:9], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5101
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4160 nv
	scratch_load_b128 v[6:9], off, off offset:4176 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[102:103], v[2:3] /*v[258:259]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[250:257], v[106:113] /*v[362:369]*/, v[2:9], v[250:257]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4192 nv
	scratch_load_b128 v[6:9], off, off offset:4208 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[242:249], v[98:105] /*v[354:361]*/, v[2:9], v[242:249]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4224 nv
	scratch_load_b128 v[6:9], off, off offset:4240 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[110:111], v[154:155] /*v[410:411]*/, v[242:243] neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[234:241], v[142:149], v[150:157], 0
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:5744 nv
	scratch_load_b128 v[154:157], off, off offset:5760 nv
	s_set_vgpr_msb 13
	v_wmma_f32_16x16x32_bf16 v[234:241], v[178:185] /*v[434:441]*/, v[20:27] /*v[788:795]*/, v[234:241]
	s_set_vgpr_msb 0xd09
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[234:241], v[162:169] /*v[418:425]*/, v[140:147] /*v[652:659]*/, v[234:241]
	s_set_vgpr_msb 0x901
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[234:241], v[106:113] /*v[362:369]*/, v[2:9], v[234:241]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4256 nv
	scratch_load_b128 v[6:9], off, off offset:4272 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_mul_f32_e32 v112, s4, v240
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[226:233], v[98:105] /*v[354:361]*/, v[2:9], v[226:233]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4288 nv
	scratch_load_b128 v[6:9], off, off offset:4304 nv
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x10c
	v_pk_add_f32 v[118:119], v[226:227], v[10:11] /*v[778:779]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xc01
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[218:225], v[106:113] /*v[362:369]*/, v[2:9], v[218:225]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4320 nv
	scratch_load_b128 v[6:9], off, off offset:4336 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_mul_f32_e32 v120, s4, v224
	s_set_vgpr_msb 0x1f1
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[922:929]*/, v[98:105] /*v[354:361]*/, v[2:9], v[154:161] /*v[922:929]*/
	s_set_vgpr_msb 0xf10f
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4352 nv
	scratch_load_b128 v[6:9], off, off offset:4368 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[126:127], v[154:155] /*v[922:923]*/, v[8:9] /*v[776:777]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xf00
	v_wmma_f32_16x16x32_bf16 v[202:209], v[142:149], v[150:157], 0
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:5776 nv
	scratch_load_b128 v[154:157], off, off offset:5792 nv
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[202:209], v[178:185] /*v[434:441]*/, v[212:219] /*v[724:731]*/, v[202:209]
	s_set_vgpr_msb 0x905
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[202:209], v[162:169] /*v[418:425]*/, v[146:153] /*v[402:409]*/, v[202:209]
	s_set_vgpr_msb 0x501
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[202:209], v[106:113] /*v[362:369]*/, v[2:9], v[202:209]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4384 nv
	scratch_load_b128 v[6:9], off, off offset:4400 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_mul_f32_e32 v128, s4, v208
	s_set_vgpr_msb 0x1c1
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[1010:1017]*/, v[194:201] /*v[450:457]*/, v[150:157], 0
	s_set_vgpr_msb 0xc100
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:5808 nv
	scratch_load_b128 v[154:157], off, off offset:5824 nv
	s_set_vgpr_msb 0xf9
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[1010:1017]*/, v[186:193] /*v[442:449]*/, v[76:83] /*v[588:595]*/, v[242:249] /*v[1010:1017]*/
	s_set_vgpr_msb 0xf9f5
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[1010:1017]*/, v[170:177] /*v[426:433]*/, v[216:223] /*v[472:479]*/, v[242:249] /*v[1010:1017]*/
	s_set_vgpr_msb 0xf5f1
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[1010:1017]*/, v[98:105] /*v[354:361]*/, v[2:9], v[242:249] /*v[1010:1017]*/
	s_set_vgpr_msb 0xf10f
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4416 nv
	scratch_load_b128 v[6:9], off, off offset:4432 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[134:135], v[242:243] /*v[1010:1011]*/, v[92:93] /*v[860:861]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xf00
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[186:193], v[142:149], v[150:157], 0
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:5840 nv
	scratch_load_b128 v[154:157], off, off offset:5856 nv
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[186:193], v[178:185] /*v[434:441]*/, v[180:187] /*v[692:699]*/, v[186:193]
	s_set_vgpr_msb 0x905
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[162:169] /*v[418:425]*/, v[232:239] /*v[488:495]*/, v[186:193]
	s_set_vgpr_msb 0x501
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[106:113] /*v[362:369]*/, v[2:9], v[186:193]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4448 nv
	scratch_load_b128 v[6:9], off, off offset:4464 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_mul_f32_e32 v136, s4, v192
	s_set_vgpr_msb 0x1c1
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[980:987]*/, v[194:201] /*v[450:457]*/, v[150:157], 0
	s_set_vgpr_msb 0xc100
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:5872 nv
	scratch_load_b128 v[154:157], off, off offset:5888 nv
	s_set_vgpr_msb 0xf9
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[980:987]*/, v[186:193] /*v[442:449]*/, v[132:139] /*v[644:651]*/, v[212:219] /*v[980:987]*/
	s_set_vgpr_msb 0xf9f5
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[980:987]*/, v[170:177] /*v[426:433]*/, v[248:255] /*v[504:511]*/, v[212:219] /*v[980:987]*/
	s_set_vgpr_msb 0xf5f1
	s_wait_loadcnt 0x2
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[980:987]*/, v[98:105] /*v[354:361]*/, v[2:9], v[212:219] /*v[980:987]*/
	s_set_vgpr_msb 0xf10b
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4512 nv
	scratch_load_b128 v[6:9], off, off offset:4528 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[178:179], v[212:213] /*v[980:981]*/, v[244:245] /*v[756:757]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xb00
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[170:177], v[142:149], v[150:157], 0
	s_clause 0x1
	scratch_load_b128 v[150:153], off, off offset:5936 nv
	scratch_load_b128 v[154:157], off, off offset:5952 nv
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[170:177], v[178:185] /*v[434:441]*/, v[92:99] /*v[604:611]*/, v[170:177]
	s_set_vgpr_msb 0x980
	s_clause 0x3
	scratch_load_b128 v[154:157] /*v[666:669]*/, off, off offset:4928 th:TH_LOAD_LU nv
	scratch_load_b128 v[158:161] /*v[670:673]*/, off, off offset:4944 th:TH_LOAD_LU nv
	scratch_load_b128 v[92:95] /*v[604:607]*/, off, off offset:3392 nv
	scratch_load_b128 v[96:99] /*v[608:611]*/, off, off offset:3408 nv
	s_set_vgpr_msb 0x8009
	v_wmma_f32_16x16x32_bf16 v[170:177], v[162:169] /*v[418:425]*/, v[0:7] /*v[512:519]*/, v[170:177]
	s_set_vgpr_msb 0x901
	s_wait_loadcnt 0x6
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[170:177], v[106:113] /*v[362:369]*/, v[2:9], v[170:177]
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4544 nv
	scratch_load_b128 v[6:9], off, off offset:4560 nv
	s_set_vgpr_msb 0x1c1
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[890:897]*/, v[194:201] /*v[450:457]*/, v[150:157], 0
	s_set_vgpr_msb 0xc1f9
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[890:897]*/, v[186:193] /*v[442:449]*/, v[44:51] /*v[556:563]*/, v[122:129] /*v[890:897]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[890:897]*/, v[170:177] /*v[426:433]*/, v[34:41] /*v[546:553]*/, v[122:129] /*v[890:897]*/
	s_set_vgpr_msb 0xf9f1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[890:897]*/, v[98:105] /*v[354:361]*/, v[2:9], v[122:129] /*v[890:897]*/
	s_set_vgpr_msb 0xf10c
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:4576 nv
	scratch_load_b128 v[6:9], off, off offset:4592 nv
	v_wmma_f32_16x16x32_bf16 v[154:161], v[142:149], v[180:187] /*v[948:955]*/, 0
	s_set_vgpr_msb 0xcc0
	s_clause 0x1
	scratch_load_b128 v[180:183] /*v[948:951]*/, off, off offset:6064 nv
	scratch_load_b128 v[184:187] /*v[952:955]*/, off, off offset:6080 nv
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xc000
	v_or_b32_e32 v142, s3, v55
	v_pk_add_f32 v[54:55], v[210:211], v[64:65] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[154:161], v[178:185] /*v[434:441]*/, v[52:59] /*v[564:571]*/, v[154:161]
	s_set_vgpr_msb 0x940
	s_clause 0x4
	scratch_load_b128 v[178:181] /*v[434:437]*/, off, off offset:4960 th:TH_LOAD_LU nv
	scratch_load_b128 v[182:185] /*v[438:441]*/, off, off offset:4976 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4084
	scratch_load_b128 v[44:47] /*v[556:559]*/, off, off offset:5312 th:TH_LOAD_LU nv
	scratch_load_b128 v[48:51] /*v[560:563]*/, off, off offset:5328 th:TH_LOAD_LU nv
	v_cmp_gt_i32_e32 vcc_lo, v142, v207 /*v463*/
	s_set_vgpr_msb 0x8400
	v_or_b32_e32 v143, 2, v142
	v_or_b32_e32 v144, 3, v142
	v_or_b32_e32 v145, 4, v142
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[154:161], v[162:169] /*v[418:425]*/, v[18:25] /*v[530:537]*/, v[154:161]
	s_set_vgpr_msb 0x904
	v_cmp_ge_i32_e32 vcc_lo, v142, v207 /*v463*/
	v_cndmask_b32_e64 v22, v22, 0xff61b1e6, s3
	s_set_vgpr_msb 0x444
	s_clause 0x1
	scratch_load_b128 v[162:165] /*v[418:421]*/, off, off offset:5280 th:TH_LOAD_LU nv
	scratch_load_b128 v[166:169] /*v[422:425]*/, off, off offset:5296 th:TH_LOAD_LU nv
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v143, v207 /*v463*/
	s_set_vgpr_msb 0x4401
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113] /*v[362:369]*/, v[2:9], v[154:161]
	s_set_vgpr_msb 0x140
	s_clause 0x4
	scratch_load_b128 v[106:109] /*v[362:365]*/, off, off offset:4832 th:TH_LOAD_LU nv
	scratch_load_b128 v[110:113] /*v[366:369]*/, off, off offset:4848 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4004
	scratch_load_b128 v[2:5], off, off offset:4608 nv
	scratch_load_b128 v[6:9], off, off offset:4624 nv
	v_cndmask_b32_e64 v23, v23, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144, v207 /*v463*/
	v_cndmask_b32_e64 v24, v24, 0xff61b1e6, s3
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_sub_f32 v22, v22, v202 /*v458*/ :: v_dual_sub_f32 v23, v23, v202 /*v458*/
	s_set_vgpr_msb 0x40d
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x32_bf16 v[146:153], v[194:201] /*v[450:457]*/, v[180:187] /*v[948:955]*/, 0
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xd04
	v_cmp_gt_i32_e32 vcc_lo, v145, v207 /*v463*/
	v_cndmask_b32_e64 v25, v25, 0xff61b1e6, s3
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v22, 0x3fb8aa3b, v22
	s_set_vgpr_msb 4
	v_sub_f32_e32 v24, v24, v202 /*v458*/
	s_set_vgpr_msb 0x4c0
	s_clause 0x7
	scratch_load_b128 v[180:183] /*v[948:951]*/, off, off offset:5408 th:TH_LOAD_LU nv
	scratch_load_b128 v[184:187] /*v[952:955]*/, off, off offset:5424 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc040
	scratch_load_b128 v[194:197] /*v[450:453]*/, off, off offset:5344 th:TH_LOAD_LU nv
	scratch_load_b128 v[198:201] /*v[454:457]*/, off, off offset:5360 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x40c0
	scratch_load_b128 v[196:199] /*v[964:967]*/, off, off offset:5088 th:TH_LOAD_LU nv
	scratch_load_b128 v[200:203] /*v[968:971]*/, off, off offset:5104 th:TH_LOAD_LU nv
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc009
	v_wmma_f32_16x16x32_bf16 v[146:153], v[186:193] /*v[442:449]*/, v[108:115] /*v[620:627]*/, v[146:153]
	s_set_vgpr_msb 0x904
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s3
	v_sub_f32_e32 v25, v25, v202 /*v458*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v23, 0x3fb8aa3b, v23 :: v_dual_mul_f32 v24, 0x3fb8aa3b, v24
	v_exp_f32_e32 v22, v22
	s_set_vgpr_msb 0x80
	s_clause 0x4
	scratch_load_b128 v[108:111] /*v[620:623]*/, off, off offset:4640 th:TH_LOAD_LU nv
	scratch_load_b128 v[112:115] /*v[624:627]*/, off, off offset:4656 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8040
	scratch_load_b128 v[186:189] /*v[442:445]*/, off, off offset:5024 th:TH_LOAD_LU nv
	scratch_load_b128 v[190:193] /*v[446:449]*/, off, off offset:5040 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4009
	v_wmma_f32_16x16x32_bf16 v[146:153], v[170:177] /*v[426:433]*/, v[26:33] /*v[538:545]*/, v[146:153]
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	s_set_vgpr_msb 64
	s_clause 0x11
	scratch_load_b128 v[146:149] /*v[402:405]*/, off, off offset:5248 th:TH_LOAD_LU nv
	scratch_load_b128 v[150:153] /*v[406:409]*/, off, off offset:5264 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[170:173] /*v[682:685]*/, off, off offset:5216 th:TH_LOAD_LU nv
	scratch_load_b128 v[174:177] /*v[686:689]*/, off, off offset:5232 th:TH_LOAD_LU nv
	scratch_load_b128 v[52:55] /*v[564:567]*/, off, off offset:5056 th:TH_LOAD_LU nv
	scratch_load_b128 v[56:59] /*v[568:571]*/, off, off offset:5072 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8000
	scratch_load_b128 v[10:13], off, off offset:3456 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:3472 th:TH_LOAD_LU nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[216:219] /*v[472:475]*/, off, off offset:3488 th:TH_LOAD_LU nv
	scratch_load_b128 v[220:223] /*v[476:479]*/, off, off offset:3504 th:TH_LOAD_LU nv
	scratch_load_b128 v[170:173] /*v[426:429]*/, off, off offset:3520 th:TH_LOAD_LU nv
	scratch_load_b128 v[174:177] /*v[430:433]*/, off, off offset:3536 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4080
	scratch_load_b128 v[68:71] /*v[580:583]*/, off, off offset:5552 nv
	scratch_load_b128 v[72:75] /*v[584:587]*/, off, off offset:5568 nv
	s_set_vgpr_msb 0x8000
	v_mul_f32_e32 v154, s4, v154
	v_pk_mul_f32 v[22:23], v[38:39], v[22:23]
	s_set_vgpr_msb 1
	s_wait_loadcnt 0x18
	v_wmma_f32_16x16x32_bf16 v[146:153], v[98:105] /*v[354:361]*/, v[2:9], v[146:153]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[38:39], v[180:181], v[62:63] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[22:23], v[220:221] /*v[988:989]*/, v[22:23]
	scratch_load_b64 v[2:3], off, off offset:5536 nv
	s_set_vgpr_msb 0x300
	v_cvt_pk_bf16_f32 v22, v22, v23
	v_nop
	s_set_vgpr_msb 64
	v_or_b32_e32 v98 /*v354*/, 5, v142
	v_or_b32_e32 v99 /*v355*/, 6, v142
	v_or_b32_e32 v100 /*v356*/, 7, v142
	s_set_vgpr_msb 0x4005
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v207 /*v463*/
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v207 /*v463*/
	s_set_vgpr_msb 0x504
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s3
	v_sub_f32_e32 v30, v30, v202 /*v458*/
	s_and_b32 s3, s47, vcc_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v31, v31, v202 /*v458*/
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v25, 0x3fb8aa3b, v25 :: v_dual_mul_f32 v30, 0x3fb8aa3b, v30
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v207 /*v463*/
	s_set_vgpr_msb 0x504
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s3
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v30, v30
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v142, v203 /*v459*/
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s3
	v_sub_f32_e32 v32, v32, v202 /*v458*/
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(TRANS32_DEP_2)
	v_pk_mul_f32 v[24:25], v[38:39], v[24:25]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v33, v33, v202 /*v458*/
	s_set_vgpr_msb 0x403
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s3, s47, vcc_lo
	v_pk_mul_f32 v[24:25], v[220:221] /*v[988:989]*/, v[24:25]
	v_mul_f32_e32 v33, 0x3fb8aa3b, v33
	s_set_vgpr_msb 0x304
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_cmp_ge_i32_e32 vcc_lo, v142, v203 /*v459*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v23, v24, v25
	v_pk_add_f32 v[24:25], v[182:183], v[62:63] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v33, v33
	s_set_vgpr_msb 4
	v_dual_mul_f32 v38, s4, v142 /*v398*/ :: v_dual_mul_f32 v39, s4, v143 /*v399*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[24:25], v[24:25], v[30:31]
	v_pk_add_f32 v[30:31], v[184:185], v[62:63] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[62:63], v[82:83] /*v[338:339]*/, v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x103
	v_pk_mul_f32 v[24:25], v[220:221] /*v[988:989]*/, v[24:25]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[30:31], v[30:31], v[32:33]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v32, s4, v140 /*v396*/ :: v_dual_mul_f32 v33, s4, v141 /*v397*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v24, v24, v25
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[30:31], v[220:221] /*v[988:989]*/, v[30:31]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v25, v30, v31
	s_set_vgpr_msb 4
	v_dual_mul_f32 v30, s4, v138 /*v394*/ :: v_dual_mul_f32 v31, s4, v139 /*v395*/
	s_set_vgpr_msb 0x440
	s_clause 0x3
	scratch_load_b128 v[138:141] /*v[394:397]*/, off, off offset:4992 th:TH_LOAD_LU nv
	scratch_load_b128 v[142:145] /*v[398:401]*/, off, off offset:5008 th:TH_LOAD_LU nv
	scratch_load_b128 v[208:211] /*v[464:467]*/, off, off offset:2944 nv
	scratch_load_b128 v[212:215] /*v[468:471]*/, off, off offset:2960 nv
	s_set_vgpr_msb 0x4004
	v_cndmask_b32_e64 v30, v30, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v143, v203 /*v459*/
	v_cndmask_b32_e64 v31, v31, 0xff61b1e6, s3
	s_set_vgpr_msb 0x40c
	v_sub_f32_e32 v30, v30, v104 /*v872*/
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc04
	v_cmp_gt_i32_e32 vcc_lo, v144, v203 /*v459*/
	v_cndmask_b32_e64 v32, v32, 0xff61b1e6, s3
	s_set_vgpr_msb 0x40c
	v_sub_f32_e32 v31, v31, v104 /*v872*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v30, 0x3fb8aa3b, v30
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v145, v203 /*v459*/
	v_cndmask_b32_e64 v33, v33, 0xff61b1e6, s3
	s_set_vgpr_msb 0x40c
	v_sub_f32_e32 v32, v32, v104 /*v872*/
	v_exp_f32_e32 v30, v30
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v203 /*v459*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e64 v38, v38, 0xff61b1e6, s3
	v_sub_f32_e32 v33, v33, v104 /*v872*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v31, 0x3fb8aa3b, v31 :: v_dual_mul_f32 v32, 0x3fb8aa3b, v32
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v203 /*v459*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e64 v39, v39, 0xff61b1e6, s3
	v_exp_f32_e32 v31, v31
	v_sub_f32_e32 v38, v38, v104 /*v872*/
	v_exp_f32_e32 v32, v32
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v39, v39, v104 /*v872*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v33, 0x3fb8aa3b, v33 :: v_dual_mul_f32 v38, 0x3fb8aa3b, v38
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v203 /*v459*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v40, v40, 0xff61b1e6, s3
	v_pk_mul_f32 v[30:31], v[46:47], v[30:31]
	v_exp_f32_e32 v33, v33
	v_pk_add_f32 v[46:47], v[164:165], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	v_exp_f32_e32 v38, v38
	v_cndmask_b32_e64 v41, v41, 0xff61b1e6, s3
	s_set_vgpr_msb 12
	v_sub_f32_e32 v40, v40, v104 /*v872*/
	v_pk_mul_f32 v[30:31], v[30:31], v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[32:33], v[46:47], v[32:33]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v41, v41, v104 /*v872*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v39, 0x3fb8aa3b, v39 :: v_dual_mul_f32 v40, 0x3fb8aa3b, v40
	v_cvt_pk_bf16_f32 v30, v30, v31
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[32:33], v[220:221] /*v[988:989]*/, v[32:33]
	v_dual_mul_f32 v41, 0x3fb8aa3b, v41 :: v_dual_mul_f32 v46, s4, v198
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v39, v39
	v_exp_f32_e32 v40, v40
	v_cvt_pk_bf16_f32 v31, v32, v33
	v_pk_add_f32 v[32:33], v[166:167], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v41, v41
	v_mul_f32_e32 v47, s4, v199
	s_delay_alu instid0(TRANS32_DEP_3) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[32:33], v[32:33], v[38:39]
	v_pk_add_f32 v[38:39], v[168:169], v[56:57] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v56, s4, v96 /*v352*/ :: v_dual_mul_f32 v57, s4, v97 /*v353*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[32:33], v[220:221] /*v[988:989]*/, v[32:33]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[38:39], v[38:39], v[40:41]
	v_dual_mul_f32 v40, s4, v196 :: v_dual_mul_f32 v41, s4, v197
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v32, v32, v33
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_3)
	v_pk_mul_f32 v[38:39], v[220:221] /*v[988:989]*/, v[38:39]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v33, v38, v39
	v_mul_f32_e32 v38, s4, v194
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v142, v241 /*v1009*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v39, s4, v195
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_ge_i32_e32 vcc_lo, v142, v241 /*v1009*/
	v_cndmask_b32_e64 v38, v38, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v143, v241 /*v1009*/
	v_cndmask_b32_e64 v39, v39, 0xff61b1e6, s3
	s_delay_alu instid0(VALU_DEP_3)
	v_sub_f32_e32 v38, v38, v254 /*v1022*/
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144, v241 /*v1009*/
	v_cndmask_b32_e64 v40, v40, 0xff61b1e6, s3
	v_sub_f32_e32 v39, v39, v254 /*v1022*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v49, s4, v201 :: v_dual_mul_f32 v38, 0x3fb8aa3b, v38
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v241 /*v1009*/
	v_cndmask_b32_e64 v41, v41, 0xff61b1e6, s3
	v_sub_f32_e32 v40, v40, v254 /*v1022*/
	v_exp_f32_e32 v38, v38
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc0d
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v241 /*v1009*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v46, v46, 0xff61b1e6, s3
	v_sub_f32_e32 v41, v41, v254 /*v1022*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v39, 0x3fb8aa3b, v39 :: v_dual_mul_f32 v40, 0x3fb8aa3b, v40
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v241 /*v1009*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v47, v47, 0xff61b1e6, s3
	v_exp_f32_e32 v39, v39
	v_sub_f32_e32 v46, v46, v254 /*v1022*/
	v_exp_f32_e32 v40, v40
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v47, v47, v254 /*v1022*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v41, 0x3fb8aa3b, v41 :: v_dual_mul_f32 v46, 0x3fb8aa3b, v46
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v241 /*v1009*/
	s_set_vgpr_msb 0xd00
	v_cndmask_b32_e64 v48, v48, 0xff61b1e6, s3
	v_pk_mul_f32 v[38:39], v[54:55], v[38:39]
	v_exp_f32_e32 v41, v41
	v_pk_add_f32 v[54:55], v[212:213], v[64:65] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	v_exp_f32_e32 v46, v46
	v_cndmask_b32_e64 v49, v49, 0xff61b1e6, s3
	s_set_vgpr_msb 12
	v_sub_f32_e32 v48, v48, v254 /*v1022*/
	v_pk_mul_f32 v[38:39], v[38:39], v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0xc08
	v_cmp_gt_i32_e32 vcc_lo, v142, v8 /*v520*/
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[40:41], v[54:55], v[40:41]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v49, v49, v254 /*v1022*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v47, 0x3fb8aa3b, v47 :: v_dual_mul_f32 v48, 0x3fb8aa3b, v48
	v_cvt_pk_bf16_f32 v38, v38, v39
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[40:41], v[220:221] /*v[988:989]*/, v[40:41]
	v_mul_f32_e32 v49, 0x3fb8aa3b, v49
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v47, v47
	v_exp_f32_e32 v48, v48
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v39, v40, v41
	v_pk_add_f32 v[40:41], v[214:215], v[64:65] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v49, v49
	s_set_vgpr_msb 8
	v_cmp_ge_i32_e32 vcc_lo, v142, v8 /*v520*/
	s_set_vgpr_msb 0x804
	v_dual_mul_f32 v54, s4, v94 /*v350*/ :: v_dual_mul_f32 v55, s4, v95 /*v351*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[40:41], v[40:41], v[46:47]
	v_pk_add_f32 v[46:47], v[216:217], v[64:65] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v64, s4, v80 /*v336*/ :: v_dual_mul_f32 v65, s4, v81 /*v337*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[40:41], v[220:221] /*v[988:989]*/, v[40:41]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[46:47], v[46:47], v[48:49]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v48, s4, v92 /*v348*/ :: v_dual_mul_f32 v49, s4, v93 /*v349*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v40, v40, v41
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[46:47], v[220:221] /*v[988:989]*/, v[46:47]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v41, v46, v47
	s_set_vgpr_msb 4
	v_dual_mul_f32 v46, s4, v90 /*v346*/ :: v_dual_mul_f32 v47, s4, v91 /*v347*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v46, v46, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x408
	v_cmp_gt_i32_e32 vcc_lo, v143, v8 /*v520*/
	v_cndmask_b32_e64 v47, v47, 0xff61b1e6, s3
	s_set_vgpr_msb 0x80c
	v_sub_f32_e32 v46, v46, v252 /*v1020*/
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc08
	v_cmp_gt_i32_e32 vcc_lo, v144, v8 /*v520*/
	v_cndmask_b32_e64 v48, v48, 0xff61b1e6, s3
	s_set_vgpr_msb 0x80c
	v_sub_f32_e32 v47, v47, v252 /*v1020*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v46, 0x3fb8aa3b, v46
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v145, v8 /*v520*/
	v_cndmask_b32_e64 v49, v49, 0xff61b1e6, s3
	s_set_vgpr_msb 0x80c
	v_sub_f32_e32 v48, v48, v252 /*v1020*/
	v_exp_f32_e32 v46, v46
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc09
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v8 /*v520*/
	s_set_vgpr_msb 0x90c
	v_cndmask_b32_e64 v54, v54, 0xff61b1e6, s3
	v_sub_f32_e32 v49, v49, v252 /*v1020*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v47, 0x3fb8aa3b, v47 :: v_dual_mul_f32 v48, 0x3fb8aa3b, v48
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 9
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v8 /*v520*/
	s_set_vgpr_msb 0x90c
	v_cndmask_b32_e64 v55, v55, 0xff61b1e6, s3
	v_exp_f32_e32 v47, v47
	v_sub_f32_e32 v54, v54, v252 /*v1020*/
	v_exp_f32_e32 v48, v48
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v55, v55, v252 /*v1020*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v49, 0x3fb8aa3b, v49 :: v_dual_mul_f32 v54, 0x3fb8aa3b, v54
	s_set_vgpr_msb 9
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v8 /*v520*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v56, v56, 0xff61b1e6, s3
	v_pk_mul_f32 v[46:47], v[62:63], v[46:47]
	v_exp_f32_e32 v49, v49
	s_set_vgpr_msb 1
	v_pk_add_f32 v[62:63], v[84:85] /*v[340:341]*/, v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x10c
	v_exp_f32_e32 v54, v54
	v_cndmask_b32_e64 v57, v57, 0xff61b1e6, s3
	v_sub_f32_e32 v56, v56, v252 /*v1020*/
	v_pk_mul_f32 v[46:47], v[46:47], v[220:221] /*v[988:989]*/
	v_cmp_gt_i32_e32 vcc_lo, v142, v105 /*v873*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[48:49], v[62:63], v[48:49]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v57, v57, v252 /*v1020*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v55, 0x3fb8aa3b, v55 :: v_dual_mul_f32 v56, 0x3fb8aa3b, v56
	v_cvt_pk_bf16_f32 v46, v46, v47
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[48:49], v[220:221] /*v[988:989]*/, v[48:49]
	v_mul_f32_e32 v57, 0x3fb8aa3b, v57
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v55, v55
	v_exp_f32_e32 v56, v56
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v47, v48, v49
	s_set_vgpr_msb 1
	v_pk_add_f32 v[48:49], v[86:87] /*v[342:343]*/, v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x10c
	v_exp_f32_e32 v57, v57
	v_cmp_ge_i32_e32 vcc_lo, v142, v105 /*v873*/
	s_set_vgpr_msb 0xc04
	v_dual_mul_f32 v62, s4, v78 /*v334*/ :: v_dual_mul_f32 v63, s4, v79 /*v335*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[48:49], v[48:49], v[54:55]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[54:55], v[88:89] /*v[344:345]*/, v[70:71] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[70:71], v[66:67] /*v[322:323]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x140
	s_clause 0x3
	scratch_load_b128 v[86:89] /*v[342:345]*/, off, off offset:4704 th:TH_LOAD_LU nv
	scratch_load_b128 v[90:93] /*v[346:349]*/, off, off offset:4720 th:TH_LOAD_LU nv
	scratch_load_b128 v[78:81] /*v[334:337]*/, off, off offset:4672 th:TH_LOAD_LU nv
	scratch_load_b128 v[82:85] /*v[338:341]*/, off, off offset:4688 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4003
	v_pk_mul_f32 v[48:49], v[220:221] /*v[988:989]*/, v[48:49]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[54:55], v[54:55], v[56:57]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v56, s4, v76 /*v332*/ :: v_dual_mul_f32 v57, s4, v77 /*v333*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v48, v48, v49
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[54:55], v[220:221] /*v[988:989]*/, v[54:55]
	s_wait_loadcnt 0x8
	v_pk_add_f32 v[162:163], v[122:123] /*v[890:891]*/, v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v49, v54, v55
	s_set_vgpr_msb 4
	v_dual_mul_f32 v54, s4, v74 /*v330*/ :: v_dual_mul_f32 v55, s4, v75 /*v331*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v54, v54, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v143, v105 /*v873*/
	v_cndmask_b32_e64 v55, v55, 0xff61b1e6, s3
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v54, v54, v158 /*v414*/
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v144, v105 /*v873*/
	v_cndmask_b32_e64 v56, v56, 0xff61b1e6, s3
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v55, v55, v158 /*v414*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v54, 0x3fb8aa3b, v54
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v105 /*v873*/
	v_cndmask_b32_e64 v57, v57, 0xff61b1e6, s3
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v56, v56, v158 /*v414*/
	v_exp_f32_e32 v54, v54
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40d
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v105 /*v873*/
	s_set_vgpr_msb 0xd04
	v_cndmask_b32_e64 v62, v62, 0xff61b1e6, s3
	v_sub_f32_e32 v57, v57, v158 /*v414*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v55, 0x3fb8aa3b, v55 :: v_dual_mul_f32 v56, 0x3fb8aa3b, v56
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v105 /*v873*/
	s_set_vgpr_msb 0xd04
	v_cndmask_b32_e64 v63, v63, 0xff61b1e6, s3
	v_exp_f32_e32 v55, v55
	v_sub_f32_e32 v62, v62, v158 /*v414*/
	v_exp_f32_e32 v56, v56
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v63, v63, v158 /*v414*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v57, 0x3fb8aa3b, v57 :: v_dual_mul_f32 v62, 0x3fb8aa3b, v62
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v105 /*v873*/
	s_set_vgpr_msb 0xd00
	v_cndmask_b32_e64 v64, v64, 0xff61b1e6, s3
	v_pk_mul_f32 v[54:55], v[70:71], v[54:55]
	v_exp_f32_e32 v57, v57
	s_set_vgpr_msb 1
	v_pk_add_f32 v[70:71], v[68:69] /*v[324:325]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x104
	v_exp_f32_e32 v62, v62
	v_cndmask_b32_e64 v65, v65, 0xff61b1e6, s3
	v_sub_f32_e32 v64, v64, v158 /*v414*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[54:55], v[220:221] /*v[988:989]*/, v[54:55]
	s_set_vgpr_msb 0x308
	v_cmp_gt_i32_e32 vcc_lo, v142, v43 /*v555*/
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[56:57], v[70:71], v[56:57]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v65, v65, v158 /*v414*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v63, 0x3fb8aa3b, v63 :: v_dual_mul_f32 v64, 0x3fb8aa3b, v64
	v_cvt_pk_bf16_f32 v54, v54, v55
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[56:57], v[220:221] /*v[988:989]*/, v[56:57]
	v_mul_f32_e32 v65, 0x3fb8aa3b, v65
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v63, v63
	v_exp_f32_e32 v64, v64
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v55, v56, v57
	s_set_vgpr_msb 1
	v_pk_add_f32 v[56:57], v[70:71] /*v[326:327]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x108
	v_exp_f32_e32 v65, v65
	v_cmp_ge_i32_e32 vcc_lo, v142, v43 /*v555*/
	s_set_vgpr_msb 0x804
	v_dual_mul_f32 v70, s4, v62 /*v318*/ :: v_dual_mul_f32 v71, s4, v63 /*v319*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[56:57], v[56:57], v[62:63]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[62:63], v[72:73] /*v[328:329]*/, v[72:73] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x104
	v_dual_mul_f32 v72, s4, v64 /*v320*/ :: v_dual_mul_f32 v73, s4, v65 /*v321*/
	s_set_vgpr_msb 0x440
	s_clause 0x1
	scratch_load_b128 v[70:73] /*v[326:329]*/, off, off offset:4768 th:TH_LOAD_LU nv
	scratch_load_b128 v[74:77] /*v[330:333]*/, off, off offset:4784 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4003
	v_pk_mul_f32 v[56:57], v[220:221] /*v[988:989]*/, v[56:57]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[62:63], v[62:63], v[64:65]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v64, s4, v60 /*v316*/ :: v_dual_mul_f32 v65, s4, v61 /*v317*/
	s_set_vgpr_msb 0x440
	s_clause 0x1
	scratch_load_b128 v[62:65] /*v[318:321]*/, off, off offset:4736 th:TH_LOAD_LU nv
	scratch_load_b128 v[66:69] /*v[322:325]*/, off, off offset:4752 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v56, v56, v57
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[62:63], v[220:221] /*v[988:989]*/, v[62:63]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v57, v62, v63
	s_set_vgpr_msb 4
	v_dual_mul_f32 v62, s4, v58 /*v314*/ :: v_dual_mul_f32 v63, s4, v59 /*v315*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v62, v62, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x408
	v_cmp_gt_i32_e32 vcc_lo, v143, v43 /*v555*/
	v_cndmask_b32_e64 v63, v63, 0xff61b1e6, s3
	v_sub_f32_e32 v62, v62, v42 /*v554*/
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144, v43 /*v555*/
	v_cndmask_b32_e64 v64, v64, 0xff61b1e6, s3
	v_sub_f32_e32 v63, v63, v42 /*v554*/
	s_set_vgpr_msb 0x800
	v_mul_f32_e32 v62, 0x3fb8aa3b, v62
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v145, v43 /*v555*/
	v_cndmask_b32_e64 v65, v65, 0xff61b1e6, s3
	v_sub_f32_e32 v64, v64, v42 /*v554*/
	v_exp_f32_e32 v62, v62
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x809
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v43 /*v555*/
	s_set_vgpr_msb 0x908
	v_cndmask_b32_e64 v70, v70, 0xff61b1e6, s3
	v_sub_f32_e32 v65, v65, v42 /*v554*/
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v63, 0x3fb8aa3b, v63 :: v_dual_mul_f32 v64, 0x3fb8aa3b, v64
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 9
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v43 /*v555*/
	s_set_vgpr_msb 0x908
	v_cndmask_b32_e64 v71, v71, 0xff61b1e6, s3
	v_exp_f32_e32 v63, v63
	v_sub_f32_e32 v70, v70, v42 /*v554*/
	v_exp_f32_e32 v64, v64
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v71, v71, v42 /*v554*/
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v65, 0x3fb8aa3b, v65 :: v_dual_mul_f32 v70, 0x3fb8aa3b, v70
	s_set_vgpr_msb 9
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v43 /*v555*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v72, v72, 0xff61b1e6, s3
	v_pk_mul_f32 v[62:63], v[78:79], v[62:63]
	v_exp_f32_e32 v65, v65
	s_set_vgpr_msb 1
	v_pk_add_f32 v[78:79], v[52:53] /*v[308:309]*/, v[80:81] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x108
	v_exp_f32_e32 v70, v70
	v_cndmask_b32_e64 v73, v73, 0xff61b1e6, s3
	v_sub_f32_e32 v72, v72, v42 /*v554*/
	s_set_vgpr_msb 0x803
	v_pk_mul_f32 v[62:63], v[220:221] /*v[988:989]*/, v[62:63]
	v_cmp_lt_i32_e32 vcc_lo, v255 /*v1023*/, v142
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[64:65], v[78:79], v[64:65]
	s_set_vgpr_msb 8
	v_sub_f32_e32 v73, v73, v42 /*v554*/
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v71, 0x3fb8aa3b, v71 :: v_dual_mul_f32 v72, 0x3fb8aa3b, v72
	v_cvt_pk_bf16_f32 v62, v62, v63
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[64:65], v[220:221] /*v[988:989]*/, v[64:65]
	v_mul_f32_e32 v73, 0x3fb8aa3b, v73
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v71, v71
	v_exp_f32_e32 v72, v72
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v63, v64, v65
	s_set_vgpr_msb 1
	v_pk_add_f32 v[64:65], v[54:55] /*v[310:311]*/, v[80:81] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x10c
	v_exp_f32_e32 v73, v73
	v_cmp_ge_i32_e32 vcc_lo, v142, v255 /*v1023*/
	s_set_vgpr_msb 0xc04
	v_dual_mul_f32 v78, s4, v46 /*v302*/ :: v_dual_mul_f32 v79, s4, v47 /*v303*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[64:65], v[64:65], v[70:71]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[70:71], v[56:57] /*v[312:313]*/, v[80:81] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x104
	v_dual_mul_f32 v80, s4, v48 /*v304*/ :: v_dual_mul_f32 v81, s4, v49 /*v305*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[64:65], v[220:221] /*v[988:989]*/, v[64:65]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[70:71], v[70:71], v[72:73]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v72, s4, v44 /*v300*/ :: v_dual_mul_f32 v73, s4, v45 /*v301*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v64, v64, v65
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[70:71], v[220:221] /*v[988:989]*/, v[70:71]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v65, v70, v71
	s_set_vgpr_msb 4
	v_dual_mul_f32 v70, s4, v42 /*v298*/ :: v_dual_mul_f32 v71, s4, v43 /*v299*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v70, v70, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v143, v255 /*v1023*/
	v_cndmask_b32_e64 v71, v71, 0xff61b1e6, s3
	v_sub_f32_e32 v70, v70, v250 /*v1018*/
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144, v255 /*v1023*/
	v_cndmask_b32_e64 v72, v72, 0xff61b1e6, s3
	v_sub_f32_e32 v71, v71, v250 /*v1018*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v70, 0x3fb8aa3b, v70
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v255 /*v1023*/
	v_cndmask_b32_e64 v73, v73, 0xff61b1e6, s3
	v_sub_f32_e32 v72, v72, v250 /*v1018*/
	v_exp_f32_e32 v70, v70
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc0d
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v255 /*v1023*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v78, v78, 0xff61b1e6, s3
	v_sub_f32_e32 v73, v73, v250 /*v1018*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v71, 0x3fb8aa3b, v71 :: v_dual_mul_f32 v72, 0x3fb8aa3b, v72
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v255 /*v1023*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v79, v79, 0xff61b1e6, s3
	v_exp_f32_e32 v71, v71
	v_sub_f32_e32 v78, v78, v250 /*v1018*/
	v_exp_f32_e32 v72, v72
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v79, v79, v250 /*v1018*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v73, 0x3fb8aa3b, v73 :: v_dual_mul_f32 v78, 0x3fb8aa3b, v78
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v255 /*v1023*/
	s_set_vgpr_msb 0xd00
	v_cndmask_b32_e64 v80, v80, 0xff61b1e6, s3
	v_pk_mul_f32 v[70:71], v[86:87], v[70:71]
	v_exp_f32_e32 v73, v73
	s_set_vgpr_msb 1
	v_pk_add_f32 v[86:87], v[36:37] /*v[292:293]*/, v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x10c
	v_exp_f32_e32 v78, v78
	v_cndmask_b32_e64 v81, v81, 0xff61b1e6, s3
	v_sub_f32_e32 v80, v80, v250 /*v1018*/
	v_pk_mul_f32 v[70:71], v[70:71], v[220:221] /*v[988:989]*/
	v_cmp_gt_i32_e32 vcc_lo, v142, v251 /*v1019*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v81, v81, v250 /*v1018*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v79, 0x3fb8aa3b, v79 :: v_dual_mul_f32 v80, 0x3fb8aa3b, v80
	v_cvt_pk_bf16_f32 v70, v70, v71
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[72:73], v[220:221] /*v[988:989]*/, v[72:73]
	v_mul_f32_e32 v81, 0x3fb8aa3b, v81
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v79, v79
	v_exp_f32_e32 v80, v80
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v71, v72, v73
	s_set_vgpr_msb 1
	v_pk_add_f32 v[72:73], v[38:39] /*v[294:295]*/, v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x10c
	v_exp_f32_e32 v81, v81
	v_cmp_ge_i32_e32 vcc_lo, v142, v251 /*v1019*/
	s_set_vgpr_msb 0xc04
	v_dual_mul_f32 v86, s4, v30 /*v286*/ :: v_dual_mul_f32 v87, s4, v31 /*v287*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[72:73], v[72:73], v[78:79]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[78:79], v[40:41] /*v[296:297]*/, v[88:89] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x104
	v_dual_mul_f32 v88, s4, v32 /*v288*/ :: v_dual_mul_f32 v89, s4, v33 /*v289*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[72:73], v[220:221] /*v[988:989]*/, v[72:73]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[78:79], v[78:79], v[80:81]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v80, s4, v28 /*v284*/ :: v_dual_mul_f32 v81, s4, v29 /*v285*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v72, v72, v73
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[78:79], v[220:221] /*v[988:989]*/, v[78:79]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v73, v78, v79
	s_set_vgpr_msb 4
	v_dual_mul_f32 v78, s4, v26 /*v282*/ :: v_dual_mul_f32 v79, s4, v27 /*v283*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v78, v78, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v143, v251 /*v1019*/
	v_cndmask_b32_e64 v79, v79, 0xff61b1e6, s3
	v_sub_f32_e32 v78, v78, v94 /*v862*/
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144, v251 /*v1019*/
	v_cndmask_b32_e64 v80, v80, 0xff61b1e6, s3
	v_sub_f32_e32 v79, v79, v94 /*v862*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v78, 0x3fb8aa3b, v78
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v251 /*v1019*/
	v_cndmask_b32_e64 v81, v81, 0xff61b1e6, s3
	v_sub_f32_e32 v80, v80, v94 /*v862*/
	v_exp_f32_e32 v78, v78
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc0d
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v251 /*v1019*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v86, v86, 0xff61b1e6, s3
	v_sub_f32_e32 v81, v81, v94 /*v862*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v79, 0x3fb8aa3b, v79 :: v_dual_mul_f32 v80, 0x3fb8aa3b, v80
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v251 /*v1019*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v87, v87, 0xff61b1e6, s3
	v_exp_f32_e32 v79, v79
	v_sub_f32_e32 v86, v86, v94 /*v862*/
	v_exp_f32_e32 v80, v80
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v87, v87, v94 /*v862*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v81, 0x3fb8aa3b, v81 :: v_dual_mul_f32 v86, 0x3fb8aa3b, v86
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v251 /*v1019*/
	s_set_vgpr_msb 0xd00
	v_cndmask_b32_e64 v88, v88, 0xff61b1e6, s3
	v_pk_mul_f32 v[78:79], v[94:95], v[78:79]
	v_exp_f32_e32 v81, v81
	s_set_vgpr_msb 1
	v_pk_add_f32 v[94:95], v[20:21] /*v[276:277]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x10c
	v_exp_f32_e32 v86, v86
	v_cndmask_b32_e64 v89, v89, 0xff61b1e6, s3
	v_sub_f32_e32 v88, v88, v94 /*v862*/
	v_pk_mul_f32 v[78:79], v[78:79], v[220:221] /*v[988:989]*/
	v_cmp_gt_i32_e32 vcc_lo, v142, v239 /*v1007*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[80:81], v[94:95], v[80:81]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v89, v89, v94 /*v862*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v87, 0x3fb8aa3b, v87 :: v_dual_mul_f32 v88, 0x3fb8aa3b, v88
	v_cvt_pk_bf16_f32 v78, v78, v79
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[80:81], v[220:221] /*v[988:989]*/, v[80:81]
	v_mul_f32_e32 v89, 0x3fb8aa3b, v89
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v88, v88
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v79, v80, v81
	s_set_vgpr_msb 1
	v_pk_add_f32 v[80:81], v[22:23] /*v[278:279]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x10c
	v_exp_f32_e32 v89, v89
	v_cmp_ge_i32_e32 vcc_lo, v142, v239 /*v1007*/
	s_set_vgpr_msb 0xc04
	v_dual_mul_f32 v94, s4, v14 /*v270*/ :: v_dual_mul_f32 v95, s4, v15 /*v271*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[80:81], v[80:81], v[86:87]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[86:87], v[24:25] /*v[280:281]*/, v[96:97] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x104
	v_dual_mul_f32 v96, s4, v16 /*v272*/ :: v_dual_mul_f32 v97, s4, v17 /*v273*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[80:81], v[220:221] /*v[988:989]*/, v[80:81]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[86:87], v[86:87], v[88:89]
	s_set_vgpr_msb 4
	v_dual_mul_f32 v88, s4, v12 /*v268*/ :: v_dual_mul_f32 v89, s4, v13 /*v269*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v80, v80, v81
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[86:87], v[220:221] /*v[988:989]*/, v[86:87]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v81, v86, v87
	s_set_vgpr_msb 4
	v_dual_mul_f32 v86, s4, v10 /*v266*/ :: v_dual_mul_f32 v87, s4, v11 /*v267*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v86, v86, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v143, v239 /*v1007*/
	v_cndmask_b32_e64 v87, v87, 0xff61b1e6, s3
	v_sub_f32_e32 v86, v86, v6 /*v774*/
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144, v239 /*v1007*/
	v_cndmask_b32_e64 v88, v88, 0xff61b1e6, s3
	v_sub_f32_e32 v87, v87, v6 /*v774*/
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v86, 0x3fb8aa3b, v86
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v239 /*v1007*/
	v_cndmask_b32_e64 v89, v89, 0xff61b1e6, s3
	v_sub_f32_e32 v88, v88, v6 /*v774*/
	v_exp_f32_e32 v86, v86
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc0d
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v239 /*v1007*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v94, v94, 0xff61b1e6, s3
	v_sub_f32_e32 v89, v89, v6 /*v774*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v87, 0x3fb8aa3b, v87 :: v_dual_mul_f32 v88, 0x3fb8aa3b, v88
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v239 /*v1007*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v95, v95, 0xff61b1e6, s3
	v_exp_f32_e32 v87, v87
	v_sub_f32_e32 v94, v94, v6 /*v774*/
	v_exp_f32_e32 v88, v88
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v95, v95, v6 /*v774*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v89, 0x3fb8aa3b, v89 :: v_dual_mul_f32 v94, 0x3fb8aa3b, v94
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v239 /*v1007*/
	s_set_vgpr_msb 0xd00
	v_cndmask_b32_e64 v96, v96, 0xff61b1e6, s3
	v_pk_mul_f32 v[86:87], v[102:103], v[86:87]
	v_exp_f32_e32 v89, v89
	s_set_vgpr_msb 1
	v_pk_add_f32 v[102:103], v[4:5] /*v[260:261]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x10c
	v_exp_f32_e32 v94, v94
	v_cndmask_b32_e64 v97, v97, 0xff61b1e6, s3
	v_sub_f32_e32 v96, v96, v6 /*v774*/
	v_pk_mul_f32 v[86:87], v[86:87], v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0xc04
	v_cmp_gt_i32_e32 vcc_lo, v142, v205 /*v461*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[88:89], v[102:103], v[88:89]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v97, v97, v6 /*v774*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v95, 0x3fb8aa3b, v95 :: v_dual_mul_f32 v96, 0x3fb8aa3b, v96
	v_cvt_pk_bf16_f32 v86, v86, v87
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[88:89], v[220:221] /*v[988:989]*/, v[88:89]
	v_mul_f32_e32 v97, 0x3fb8aa3b, v97
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v95, v95
	v_exp_f32_e32 v96, v96
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v87, v88, v89
	s_set_vgpr_msb 1
	v_pk_add_f32 v[88:89], v[6:7] /*v[262:263]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x104
	v_exp_f32_e32 v97, v97
	v_cmp_ge_i32_e32 vcc_lo, v142, v205 /*v461*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v102, s4, v254
	v_pk_mul_f32 v[88:89], v[88:89], v[94:95]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[94:95], v[8:9] /*v[264:265]*/, v[104:105] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x104
	v_dual_mul_f32 v104, s4, v0 /*v256*/ :: v_dual_mul_f32 v105, s4, v1 /*v257*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[88:89], v[220:221] /*v[988:989]*/, v[88:89]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[94:95], v[94:95], v[96:97]
	v_dual_mul_f32 v96, s4, v252 :: v_dual_mul_f32 v97, s4, v253
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v88, v88, v89
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_3)
	v_pk_mul_f32 v[94:95], v[220:221] /*v[988:989]*/, v[94:95]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v89, v94, v95
	v_dual_mul_f32 v94, s4, v250 :: v_dual_mul_f32 v95, s4, v251
	v_cndmask_b32_e64 v94, v94, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v143, v205 /*v461*/
	v_cndmask_b32_e64 v95, v95, 0xff61b1e6, s3
	v_sub_f32_e32 v94, v94, v156 /*v412*/
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144, v205 /*v461*/
	v_cndmask_b32_e64 v96, v96, 0xff61b1e6, s3
	v_sub_f32_e32 v95, v95, v156 /*v412*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v103, s4, v255 :: v_dual_mul_f32 v94, 0x3fb8aa3b, v94
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e32 vcc_lo, v145, v205 /*v461*/
	v_cndmask_b32_e64 v97, v97, 0xff61b1e6, s3
	v_sub_f32_e32 v96, v96, v156 /*v412*/
	v_exp_f32_e32 v94, v94
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x405
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v205 /*v461*/
	s_set_vgpr_msb 0x504
	v_cndmask_b32_e64 v102, v102, 0xff61b1e6, s3
	v_sub_f32_e32 v97, v97, v156 /*v412*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v95, 0x3fb8aa3b, v95 :: v_dual_mul_f32 v96, 0x3fb8aa3b, v96
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v205 /*v461*/
	s_set_vgpr_msb 0x504
	v_cndmask_b32_e64 v103, v103, 0xff61b1e6, s3
	v_exp_f32_e32 v95, v95
	v_sub_f32_e32 v102, v102, v156 /*v412*/
	v_exp_f32_e32 v96, v96
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v103, v103, v156 /*v412*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v97, 0x3fb8aa3b, v97 :: v_dual_mul_f32 v102, 0x3fb8aa3b, v102
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v205 /*v461*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v104, v104, 0xff61b1e6, s3
	v_pk_mul_f32 v[94:95], v[110:111], v[94:95]
	v_exp_f32_e32 v97, v97
	s_set_vgpr_msb 4
	v_pk_add_f32 v[110:111], v[244:245], v[154:155] /*v[410:411]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	v_exp_f32_e32 v102, v102
	v_cndmask_b32_e64 v105, v105, 0xff61b1e6, s3
	v_sub_f32_e32 v104, v104, v156 /*v412*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[94:95], v[220:221] /*v[988:989]*/, v[94:95]
	v_cmp_lt_i32_e32 vcc_lo, v7 /*v775*/, v142
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[96:97], v[110:111], v[96:97]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v105, v105, v156 /*v412*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v103, 0x3fb8aa3b, v103 :: v_dual_mul_f32 v104, 0x3fb8aa3b, v104
	v_cvt_pk_bf16_f32 v94, v94, v95
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[96:97], v[220:221] /*v[988:989]*/, v[96:97]
	v_mul_f32_e32 v105, 0x3fb8aa3b, v105
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v103, v103
	v_exp_f32_e32 v104, v104
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v95, v96, v97
	s_set_vgpr_msb 4
	v_pk_add_f32 v[96:97], v[246:247], v[154:155] /*v[410:411]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v105, v105
	s_set_vgpr_msb 0x40c
	v_cmp_ge_i32_e32 vcc_lo, v142, v7 /*v775*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v110, s4, v238 :: v_dual_mul_f32 v111, s4, v239
	v_pk_mul_f32 v[96:97], v[96:97], v[102:103]
	s_set_vgpr_msb 4
	v_pk_add_f32 v[102:103], v[248:249], v[154:155] /*v[410:411]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x403
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[96:97], v[220:221] /*v[988:989]*/, v[96:97]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_pk_mul_f32 v[102:103], v[102:103], v[104:105]
	v_dual_mul_f32 v104, s4, v236 :: v_dual_mul_f32 v105, s4, v237
	v_cvt_pk_bf16_f32 v96, v96, v97
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_3)
	v_pk_mul_f32 v[102:103], v[220:221] /*v[988:989]*/, v[102:103]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v97, v102, v103
	v_dual_mul_f32 v102, s4, v234 :: v_dual_mul_f32 v103, s4, v235
	v_cndmask_b32_e64 v102, v102, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v143, v7 /*v775*/
	v_cndmask_b32_e64 v103, v103, 0xff61b1e6, s3
	v_sub_f32_e32 v102, v102, v170 /*v938*/
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144, v7 /*v775*/
	v_cndmask_b32_e64 v104, v104, 0xff61b1e6, s3
	v_sub_f32_e32 v103, v103, v170 /*v938*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v113, s4, v241 :: v_dual_mul_f32 v102, 0x3fb8aa3b, v102
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v7 /*v775*/
	v_cndmask_b32_e64 v105, v105, 0xff61b1e6, s3
	v_sub_f32_e32 v104, v104, v170 /*v938*/
	v_exp_f32_e32 v102, v102
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc0d
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v7 /*v775*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v110, v110, 0xff61b1e6, s3
	v_sub_f32_e32 v105, v105, v170 /*v938*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v103, 0x3fb8aa3b, v103 :: v_dual_mul_f32 v104, 0x3fb8aa3b, v104
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v7 /*v775*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v111, v111, 0xff61b1e6, s3
	v_exp_f32_e32 v103, v103
	v_sub_f32_e32 v110, v110, v170 /*v938*/
	v_exp_f32_e32 v104, v104
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v111, v111, v170 /*v938*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v105, 0x3fb8aa3b, v105 :: v_dual_mul_f32 v110, 0x3fb8aa3b, v110
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v7 /*v775*/
	s_set_vgpr_msb 0xd00
	v_cndmask_b32_e64 v112, v112, 0xff61b1e6, s3
	v_pk_mul_f32 v[102:103], v[118:119], v[102:103]
	v_exp_f32_e32 v105, v105
	s_set_vgpr_msb 12
	v_pk_add_f32 v[118:119], v[228:229], v[10:11] /*v[778:779]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	v_exp_f32_e32 v110, v110
	v_cndmask_b32_e64 v113, v113, 0xff61b1e6, s3
	v_sub_f32_e32 v112, v112, v170 /*v938*/
	v_pk_mul_f32 v[102:103], v[102:103], v[220:221] /*v[988:989]*/
	v_cmp_gt_i32_e32 vcc_lo, v142, v95 /*v863*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[104:105], v[118:119], v[104:105]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v113, v113, v170 /*v938*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v111, 0x3fb8aa3b, v111 :: v_dual_mul_f32 v112, 0x3fb8aa3b, v112
	v_cvt_pk_bf16_f32 v102, v102, v103
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[104:105], v[220:221] /*v[988:989]*/, v[104:105]
	v_mul_f32_e32 v113, 0x3fb8aa3b, v113
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v111, v111
	v_exp_f32_e32 v112, v112
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v103, v104, v105
	s_set_vgpr_msb 12
	v_pk_add_f32 v[104:105], v[230:231], v[10:11] /*v[778:779]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v113, v113
	v_cmp_ge_i32_e32 vcc_lo, v142, v95 /*v863*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v118, s4, v222 :: v_dual_mul_f32 v119, s4, v223
	v_pk_mul_f32 v[104:105], v[104:105], v[110:111]
	s_set_vgpr_msb 12
	v_pk_add_f32 v[110:111], v[232:233], v[10:11] /*v[778:779]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[104:105], v[104:105], v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_pk_mul_f32 v[110:111], v[110:111], v[112:113]
	v_dual_mul_f32 v112, s4, v220 :: v_dual_mul_f32 v113, s4, v221
	v_cvt_pk_bf16_f32 v104, v104, v105
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_3)
	v_pk_mul_f32 v[110:111], v[220:221] /*v[988:989]*/, v[110:111]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v105, v110, v111
	v_dual_mul_f32 v110, s4, v218 :: v_dual_mul_f32 v111, s4, v219
	v_cndmask_b32_e64 v110, v110, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v143, v95 /*v863*/
	v_cndmask_b32_e64 v111, v111, 0xff61b1e6, s3
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v110, v110, v160 /*v416*/
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v144, v95 /*v863*/
	v_cndmask_b32_e64 v112, v112, 0xff61b1e6, s3
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v111, v111, v160 /*v416*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v121, s4, v225 :: v_dual_mul_f32 v110, 0x3fb8aa3b, v110
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v95 /*v863*/
	v_cndmask_b32_e64 v113, v113, 0xff61b1e6, s3
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v112, v112, v160 /*v416*/
	v_exp_f32_e32 v110, v110
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40d
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v95 /*v863*/
	s_set_vgpr_msb 0xd04
	v_cndmask_b32_e64 v118, v118, 0xff61b1e6, s3
	v_sub_f32_e32 v113, v113, v160 /*v416*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v111, 0x3fb8aa3b, v111 :: v_dual_mul_f32 v112, 0x3fb8aa3b, v112
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v95 /*v863*/
	s_set_vgpr_msb 0xd04
	v_cndmask_b32_e64 v119, v119, 0xff61b1e6, s3
	v_exp_f32_e32 v111, v111
	v_sub_f32_e32 v118, v118, v160 /*v416*/
	v_exp_f32_e32 v112, v112
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v119, v119, v160 /*v416*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v113, 0x3fb8aa3b, v113 :: v_dual_mul_f32 v118, 0x3fb8aa3b, v118
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v95 /*v863*/
	s_set_vgpr_msb 0xd00
	v_cndmask_b32_e64 v120, v120, 0xff61b1e6, s3
	v_pk_mul_f32 v[110:111], v[126:127], v[110:111]
	v_exp_f32_e32 v113, v113
	s_set_vgpr_msb 15
	v_pk_add_f32 v[126:127], v[156:157] /*v[924:925]*/, v[8:9] /*v[776:777]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xf04
	v_exp_f32_e32 v118, v118
	v_cndmask_b32_e64 v121, v121, 0xff61b1e6, s3
	v_sub_f32_e32 v120, v120, v160 /*v416*/
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[110:111], v[220:221] /*v[988:989]*/, v[110:111]
	v_cmp_lt_i32_e32 vcc_lo, v171 /*v939*/, v142
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[112:113], v[126:127], v[112:113]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v121, v121, v160 /*v416*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v119, 0x3fb8aa3b, v119 :: v_dual_mul_f32 v120, 0x3fb8aa3b, v120
	v_cvt_pk_bf16_f32 v110, v110, v111
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[112:113], v[220:221] /*v[988:989]*/, v[112:113]
	v_mul_f32_e32 v121, 0x3fb8aa3b, v121
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v119, v119
	v_exp_f32_e32 v120, v120
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v111, v112, v113
	s_set_vgpr_msb 15
	v_pk_add_f32 v[112:113], v[158:159] /*v[926:927]*/, v[8:9] /*v[776:777]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xf0c
	v_exp_f32_e32 v121, v121
	v_cmp_ge_i32_e32 vcc_lo, v142, v171 /*v939*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v126, s4, v206 :: v_dual_mul_f32 v127, s4, v207
	v_pk_mul_f32 v[112:113], v[112:113], v[118:119]
	s_set_vgpr_msb 15
	v_pk_add_f32 v[118:119], v[160:161] /*v[928:929]*/, v[8:9] /*v[776:777]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xf03
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[112:113], v[220:221] /*v[988:989]*/, v[112:113]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_pk_mul_f32 v[118:119], v[118:119], v[120:121]
	v_dual_mul_f32 v120, s4, v204 :: v_dual_mul_f32 v121, s4, v205
	v_cvt_pk_bf16_f32 v112, v112, v113
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_3)
	v_pk_mul_f32 v[118:119], v[220:221] /*v[988:989]*/, v[118:119]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v113, v118, v119
	v_dual_mul_f32 v118, s4, v202 :: v_dual_mul_f32 v119, s4, v203
	v_cndmask_b32_e64 v118, v118, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v143, v171 /*v939*/
	v_cndmask_b32_e64 v119, v119, 0xff61b1e6, s3
	v_sub_f32_e32 v118, v118, v238 /*v1006*/
	s_and_b32 s3, s47, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144, v171 /*v939*/
	v_cndmask_b32_e64 v120, v120, 0xff61b1e6, s3
	v_sub_f32_e32 v119, v119, v238 /*v1006*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v129, s4, v209 :: v_dual_mul_f32 v118, 0x3fb8aa3b, v118
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v171 /*v939*/
	v_cndmask_b32_e64 v121, v121, 0xff61b1e6, s3
	v_sub_f32_e32 v120, v120, v238 /*v1006*/
	v_exp_f32_e32 v118, v118
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xc0d
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v171 /*v939*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v126, v126, 0xff61b1e6, s3
	v_sub_f32_e32 v121, v121, v238 /*v1006*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v119, 0x3fb8aa3b, v119 :: v_dual_mul_f32 v120, 0x3fb8aa3b, v120
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v171 /*v939*/
	s_set_vgpr_msb 0xd0c
	v_cndmask_b32_e64 v127, v127, 0xff61b1e6, s3
	v_exp_f32_e32 v119, v119
	v_sub_f32_e32 v126, v126, v238 /*v1006*/
	v_exp_f32_e32 v120, v120
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v127, v127, v238 /*v1006*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v121, 0x3fb8aa3b, v121 :: v_dual_mul_f32 v126, 0x3fb8aa3b, v126
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v171 /*v939*/
	s_set_vgpr_msb 0xd00
	v_cndmask_b32_e64 v128, v128, 0xff61b1e6, s3
	v_pk_mul_f32 v[118:119], v[134:135], v[118:119]
	v_exp_f32_e32 v121, v121
	s_set_vgpr_msb 15
	v_pk_add_f32 v[134:135], v[244:245] /*v[1012:1013]*/, v[92:93] /*v[860:861]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xf0c
	v_exp_f32_e32 v126, v126
	v_cndmask_b32_e64 v129, v129, 0xff61b1e6, s3
	v_sub_f32_e32 v128, v128, v238 /*v1006*/
	v_pk_mul_f32 v[118:119], v[118:119], v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0xc08
	v_cmp_gt_i32_e32 vcc_lo, v142, v9 /*v521*/
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[120:121], v[134:135], v[120:121]
	s_set_vgpr_msb 12
	v_sub_f32_e32 v129, v129, v238 /*v1006*/
	s_set_vgpr_msb 0xc00
	v_dual_mul_f32 v127, 0x3fb8aa3b, v127 :: v_dual_mul_f32 v128, 0x3fb8aa3b, v128
	v_cvt_pk_bf16_f32 v118, v118, v119
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[120:121], v[220:221] /*v[988:989]*/, v[120:121]
	v_mul_f32_e32 v129, 0x3fb8aa3b, v129
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v127, v127
	v_exp_f32_e32 v128, v128
	s_and_b32 s3, s47, vcc_lo
	v_cvt_pk_bf16_f32 v119, v120, v121
	s_set_vgpr_msb 15
	v_pk_add_f32 v[120:121], v[246:247] /*v[1014:1015]*/, v[92:93] /*v[860:861]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xf08
	v_exp_f32_e32 v129, v129
	v_cmp_ge_i32_e32 vcc_lo, v142, v9 /*v521*/
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v134, s4, v190 :: v_dual_mul_f32 v135, s4, v191
	v_pk_mul_f32 v[120:121], v[120:121], v[126:127]
	s_set_vgpr_msb 15
	v_pk_add_f32 v[126:127], v[248:249] /*v[1016:1017]*/, v[92:93] /*v[860:861]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xf03
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[120:121], v[220:221] /*v[988:989]*/, v[120:121]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_pk_mul_f32 v[126:127], v[126:127], v[128:129]
	v_dual_mul_f32 v128, s4, v188 :: v_dual_mul_f32 v129, s4, v189
	v_cvt_pk_bf16_f32 v120, v120, v121
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_3)
	v_pk_mul_f32 v[126:127], v[220:221] /*v[988:989]*/, v[126:127]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v121, v126, v127
	v_dual_mul_f32 v126, s4, v186 :: v_dual_mul_f32 v127, s4, v187
	s_clause 0x1
	scratch_load_b128 v[182:185], off, off offset:5152 th:TH_LOAD_LU nv
	scratch_load_b128 v[186:189], off, off offset:5168 th:TH_LOAD_LU nv
	v_cndmask_b32_e64 v126, v126, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v143, v9 /*v521*/
	v_cndmask_b32_e64 v127, v127, 0xff61b1e6, s3
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v137, s4, v193 :: v_dual_sub_f32 v126, v126, v0
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v144, v9 /*v521*/
	v_cndmask_b32_e64 v128, v128, 0xff61b1e6, s3
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v126, 0x3fb8aa3b, v126 :: v_dual_sub_f32 v127, v127, v0
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e32 vcc_lo, v145, v9 /*v521*/
	v_cndmask_b32_e64 v129, v129, 0xff61b1e6, s3
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v127, 0x3fb8aa3b, v127 :: v_dual_sub_f32 v128, v128, v0
	v_exp_f32_e32 v126, v126
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 9
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v9 /*v521*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v134, v134, 0xff61b1e6, s3
	v_sub_f32_e32 v129, v129, v0
	v_exp_f32_e32 v127, v127
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 9
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v9 /*v521*/
	s_set_vgpr_msb 0x900
	v_sub_f32_e32 v134, v134, v0
	v_dual_mul_f32 v128, 0x3fb8aa3b, v128 :: v_dual_mul_f32 v129, 0x3fb8aa3b, v129
	v_cndmask_b32_e64 v135, v135, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 9
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v9 /*v521*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v128, v128
	v_exp_f32_e32 v129, v129
	v_cndmask_b32_e64 v136, v136, 0xff61b1e6, s3
	v_pk_mul_f32 v[126:127], v[178:179], v[126:127]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 11
	v_pk_add_f32 v[178:179], v[214:215] /*v[982:983]*/, v[244:245] /*v[756:757]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xb00
	v_cndmask_b32_e64 v137, v137, 0xff61b1e6, s3
	v_dual_mul_f32 v134, 0x3fb8aa3b, v134 :: v_dual_sub_f32 v135, v135, v0
	v_sub_f32_e32 v136, v136, v0
	v_pk_mul_f32 v[128:129], v[178:179], v[128:129]
	s_delay_alu instid0(VALU_DEP_4)
	v_sub_f32_e32 v137, v137, v0
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[126:127], v[220:221] /*v[988:989]*/, v[126:127]
	v_mul_f32_e32 v135, 0x3fb8aa3b, v135
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v134, v134
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[128:129], v[220:221] /*v[988:989]*/, v[128:129]
	v_dual_mul_f32 v136, 0x3fb8aa3b, v136 :: v_dual_mul_f32 v137, 0x3fb8aa3b, v137
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v135, v135
	v_cvt_pk_bf16_f32 v126, v126, v127
	v_cvt_pk_bf16_f32 v127, v128, v129
	s_set_vgpr_msb 11
	v_pk_add_f32 v[128:129], v[216:217] /*v[984:985]*/, v[244:245] /*v[756:757]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xb0c
	v_exp_f32_e32 v136, v136
	v_exp_f32_e32 v137, v137
	v_cmp_gt_i32_e32 vcc_lo, v142, v253 /*v1021*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[128:129], v[128:129], v[134:135]
	s_set_vgpr_msb 11
	v_pk_add_f32 v[134:135], v[218:219] /*v[986:987]*/, v[244:245] /*v[756:757]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0xb0c
	v_cmp_ge_i32_e32 vcc_lo, v142, v253 /*v1021*/
	v_pk_mul_f32 v[128:129], v[128:129], v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[134:135], v[134:135], v[136:137]
	v_dual_mul_f32 v136, s4, v172 :: v_dual_mul_f32 v137, s4, v173
	v_mul_f32_e32 v172, s4, v176
	v_cvt_pk_bf16_f32 v128, v128, v129
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[134:135], v[220:221] /*v[988:989]*/, v[134:135]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v129, v134, v135
	v_dual_mul_f32 v134, s4, v170 :: v_dual_mul_f32 v135, s4, v171
	v_dual_mul_f32 v170, s4, v174 :: v_dual_mul_f32 v171, s4, v175
	v_cndmask_b32_e64 v134, v134, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v143, v253 /*v1021*/
	v_cndmask_b32_e64 v135, v135, 0xff61b1e6, s3
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v134, v134, v204 /*v460*/
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40c
	v_cmp_gt_i32_e32 vcc_lo, v144, v253 /*v1021*/
	v_cndmask_b32_e64 v136, v136, 0xff61b1e6, s3
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v135, v135, v204 /*v460*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v173, s4, v177 :: v_dual_mul_f32 v134, 0x3fb8aa3b, v134
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v253 /*v1021*/
	v_cndmask_b32_e64 v137, v137, 0xff61b1e6, s3
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v136, v136, v204 /*v460*/
	v_exp_f32_e32 v134, v134
	s_clause 0x1
	scratch_load_b128 v[174:177], off, off offset:5184 th:TH_LOAD_LU nv
	scratch_load_b128 v[178:181], off, off offset:5200 th:TH_LOAD_LU nv
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x40d
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v253 /*v1021*/
	s_set_vgpr_msb 0xd04
	v_cndmask_b32_e64 v170, v170, 0xff61b1e6, s3
	v_sub_f32_e32 v137, v137, v204 /*v460*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v135, 0x3fb8aa3b, v135 :: v_dual_mul_f32 v136, 0x3fb8aa3b, v136
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v253 /*v1021*/
	s_set_vgpr_msb 0xd04
	v_cndmask_b32_e64 v171, v171, 0xff61b1e6, s3
	v_exp_f32_e32 v135, v135
	v_sub_f32_e32 v170, v170, v204 /*v460*/
	v_exp_f32_e32 v136, v136
	s_and_b32 s3, s47, vcc_lo
	v_sub_f32_e32 v171, v171, v204 /*v460*/
	s_set_vgpr_msb 0x400
	v_dual_mul_f32 v137, 0x3fb8aa3b, v137 :: v_dual_mul_f32 v170, 0x3fb8aa3b, v170
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v253 /*v1021*/
	s_set_vgpr_msb 0xd00
	v_pk_mul_f32 v[134:135], v[162:163], v[134:135]
	s_set_vgpr_msb 3
	v_pk_add_f32 v[162:163], v[124:125] /*v[892:893]*/, v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v137, v137
	v_cndmask_b32_e64 v172, v172, 0xff61b1e6, s3
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[134:135], v[220:221] /*v[988:989]*/, v[134:135]
	s_set_vgpr_msb 0x30c
	v_cndmask_b32_e64 v173, v173, 0xff61b1e6, s3
	v_cmp_gt_i32_e32 vcc_lo, v142, v240 /*v1008*/
	s_set_vgpr_msb 0xc04
	v_sub_f32_e32 v172, v172, v204 /*v460*/
	v_exp_f32_e32 v170, v170
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[136:137], v[162:163], v[136:137]
	v_cvt_pk_bf16_f32 v134, v134, v135
	s_set_vgpr_msb 3
	v_pk_add_f32 v[162:163], v[128:129] /*v[896:897]*/, v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x304
	v_sub_f32_e32 v173, v173, v204 /*v460*/
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 0x403
	v_pk_mul_f32 v[136:137], v[220:221] /*v[988:989]*/, v[136:137]
	v_cmp_le_i32_e32 vcc_lo, v240 /*v1008*/, v142
	v_dual_mul_f32 v142, s4, v155 :: v_dual_mul_f32 v173, 0x3fb8aa3b, v173
	s_set_vgpr_msb 0x300
	v_cndmask_b32_e64 v154, v154, 0xff61b1e6, s3
	v_cvt_pk_bf16_f32 v135, v136, v137
	s_set_vgpr_msb 3
	v_pk_add_f32 v[136:137], v[126:127] /*v[894:895]*/, v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	scratch_load_b64 v[2:3], off, off offset:5544 nv
	s_and_b32 s3, s47, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v240 /*v1008*/, v143
	s_set_vgpr_msb 0x300
	v_cndmask_b32_e64 v155, v142, 0xff61b1e6, s3
	v_dual_mul_f32 v171, 0x3fb8aa3b, v171 :: v_dual_mul_f32 v172, 0x3fb8aa3b, v172
	v_exp_f32_e32 v173, v173
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 4
	v_sub_f32_e32 v143, v155, v206 /*v462*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v142, s4, v156
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v144, v240 /*v1008*/
	v_exp_f32_e32 v171, v171
	v_exp_f32_e32 v172, v172
	s_set_vgpr_msb 0xc00
	v_mul_f32_e32 v143, 0x3fb8aa3b, v143
	v_cndmask_b32_e64 v156, v142, 0xff61b1e6, s3
	v_mul_f32_e32 v142, s4, v157
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 12
	v_cmp_gt_i32_e32 vcc_lo, v145, v240 /*v1008*/
	v_exp_f32_e32 v143, v143
	s_set_vgpr_msb 0xc00
	v_pk_mul_f32 v[136:137], v[136:137], v[170:171]
	v_cndmask_b32_e64 v144, v142, 0xff61b1e6, s3
	v_mul_f32_e32 v142, s4, v158
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v98 /*v354*/, v240 /*v1008*/
	s_set_vgpr_msb 0xd00
	v_pk_mul_f32 v[162:163], v[162:163], v[172:173]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v144, v144, v206 /*v462*/
	v_cndmask_b32_e64 v145, v142, 0xff61b1e6, s3
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v142, s4, v159
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v240 /*v1008*/
	s_set_vgpr_msb 0xd03
	v_mul_f32_e32 v144, 0x3fb8aa3b, v144
	v_pk_mul_f32 v[136:137], v[220:221] /*v[988:989]*/, v[136:137]
	s_set_vgpr_msb 0x300
	v_cndmask_b32_e64 v158, v142, 0xff61b1e6, s3
	v_mul_f32_e32 v142, s4, v160
	s_and_b32 s3, s47, vcc_lo
	s_set_vgpr_msb 13
	v_cmp_gt_i32_e32 vcc_lo, v100 /*v356*/, v240 /*v1008*/
	s_set_vgpr_msb 0xd04
	v_exp_f32_e32 v157, v144
	v_dual_sub_f32 v144, v145, v206 /*v462*/ :: v_dual_sub_f32 v145, v158, v206 /*v462*/
	v_cndmask_b32_e64 v159, v142, 0xff61b1e6, s3
	s_set_vgpr_msb 0x403
	v_mul_f32_e32 v142, s4, v161
	s_and_b32 s3, s47, vcc_lo
	v_dual_mul_f32 v144, 0x3fb8aa3b, v144 :: v_dual_mul_f32 v145, 0x3fb8aa3b, v145
	v_pk_mul_f32 v[162:163], v[220:221] /*v[988:989]*/, v[162:163]
	s_set_vgpr_msb 0x304
	v_cndmask_b32_e64 v160, v142, 0xff61b1e6, s3
	v_dual_sub_f32 v142, v154, v206 /*v462*/ :: v_dual_sub_f32 v154, v156, v206 /*v462*/
	v_exp_f32_e32 v144, v144
	v_exp_f32_e32 v145, v145
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v155, v160, v206 /*v462*/
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v142, 0x3fb8aa3b, v142
	v_mul_f32_e32 v154, 0x3fb8aa3b, v154
	v_cvt_pk_bf16_f32 v136, v136, v137
	v_cvt_pk_bf16_f32 v137, v162, v163
	v_mul_f32_e32 v155, 0x3fb8aa3b, v155
	v_exp_f32_e32 v142, v142
	v_exp_f32_e32 v156, v154
	v_nop
	s_set_vgpr_msb 4
	v_sub_f32_e32 v154, v159, v206 /*v462*/
	s_set_vgpr_msb 0x4c0
	s_clause 0x1
	scratch_load_b128 v[122:125] /*v[890:893]*/, off, off offset:5376 th:TH_LOAD_LU nv
	scratch_load_b128 v[126:129] /*v[894:897]*/, off, off offset:5392 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v155, v155
	s_clause 0x1
	scratch_load_b128 v[162:165], off, off offset:5120 th:TH_LOAD_LU nv
	scratch_load_b128 v[166:169], off, off offset:5136 th:TH_LOAD_LU nv
	s_cmp_lg_u64 s[10:11], s[8:9]
	v_mul_f32_e32 v154, 0x3fb8aa3b, v154
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v154, v154
	s_wait_loadcnt 0x4
	v_pk_add_f32 v[146:147], v[146:147], v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[142:143], v[146:147], v[142:143]
	v_pk_add_f32 v[146:147], v[148:149], v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[142:143], v[220:221] /*v[988:989]*/, v[142:143]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[146:147], v[146:147], v[156:157]
	v_cvt_pk_bf16_f32 v142, v142, v143
	s_set_vgpr_msb 3
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[146:147], v[220:221] /*v[988:989]*/, v[146:147]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v143, v146, v147
	v_pk_add_f32 v[146:147], v[150:151], v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[146:147], v[144:145]
	v_pk_add_f32 v[146:147], v[152:153], v[2:3] neg_lo:[0,1] neg_hi:[0,1]
	scratch_load_b32 v2, off, off offset:8340 nv
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[144:145], v[220:221] /*v[988:989]*/, v[144:145]
	s_set_vgpr_msb 0x300
	v_pk_mul_f32 v[146:147], v[146:147], v[154:155]
	s_clause 0x1
	scratch_load_b128 v[154:157], off, off offset:2656 nv
	scratch_load_b128 v[158:161], off, off offset:2672 nv
	v_cvt_pk_bf16_f32 v144, v144, v145
	s_set_vgpr_msb 3
	v_pk_mul_f32 v[146:147], v[220:221] /*v[988:989]*/, v[146:147]
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v145, v146, v147
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[146:149], v2
	ds_load_tr16_b128 v[150:153], v2 offset:4352
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[58:65], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2656 nv
	scratch_store_b128 off, v[158:161], off offset:2672 nv
	scratch_load_b128 v[154:157], off, off offset:2368 nv
	scratch_load_b128 v[158:161], off, off offset:2384 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[66:73], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2368 nv
	scratch_store_b128 off, v[158:161], off offset:2384 nv
	scratch_load_b128 v[154:157], off, off offset:480 nv
	scratch_load_b128 v[158:161], off, off offset:496 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[90:97], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:480 nv
	scratch_store_b128 off, v[158:161], off offset:496 nv
	scratch_load_b128 v[154:157], off, off offset:2080 nv
	scratch_load_b128 v[158:161], off, off offset:2096 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[98:105], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2080 nv
	scratch_store_b128 off, v[158:161], off offset:2096 nv
	scratch_load_b128 v[154:157], off, off offset:1664 nv
	scratch_load_b128 v[158:161], off, off offset:1680 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113], v[146:153], v[154:161]
	s_clause 0x5
	scratch_store_b128 off, v[154:157], off offset:1664 nv
	scratch_store_b128 off, v[158:161], off offset:1680 nv
	scratch_load_b128 v[154:157], off, off offset:1728 nv
	scratch_load_b128 v[158:161], off, off offset:1744 nv
	scratch_load_b128 v[2:5], off, off offset:2912 nv
	scratch_load_b128 v[6:9], off, off offset:2928 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[114:121], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1728 nv
	scratch_store_b128 off, v[158:161], off offset:1744 nv
	scratch_load_b128 v[154:157], off, off offset:1408 nv
	scratch_load_b128 v[158:161], off, off offset:1424 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[26:33], v[146:153], v[2:9]
	s_clause 0x3
	scratch_store_b128 off, v[2:5], off offset:2912 nv
	scratch_store_b128 off, v[6:9], off offset:2928 nv
	scratch_load_b128 v[2:5], off, off offset:2976 nv
	scratch_load_b128 v[6:9], off, off offset:2992 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[122:129], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1408 nv
	scratch_store_b128 off, v[158:161], off offset:1424 nv
	scratch_load_b128 v[154:157], off, off offset:1056 nv
	scratch_load_b128 v[158:161], off, off offset:1072 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[42:49], v[146:153], v[2:9]
	s_clause 0x3
	scratch_store_b128 off, v[2:5], off offset:2976 nv
	scratch_store_b128 off, v[6:9], off offset:2992 nv
	scratch_load_b128 v[2:5], off, off offset:224 nv
	scratch_load_b128 v[6:9], off, off offset:240 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[130:137], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1056 nv
	scratch_store_b128 off, v[158:161], off offset:1072 nv
	scratch_load_b128 v[154:157], off, off offset:736 nv
	scratch_load_b128 v[158:161], off, off offset:752 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81], v[146:153], v[2:9]
	s_clause 0x2
	scratch_store_b128 off, v[2:5], off offset:224 nv
	scratch_store_b128 off, v[6:9], off offset:240 nv
	scratch_load_b32 v2, off, off offset:8356 nv
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:736 nv
	scratch_store_b128 off, v[158:161], off offset:752 nv
	scratch_load_b128 v[154:157], off, off offset:2624 nv
	scratch_load_b128 v[158:161], off, off offset:2640 nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[18:25], v[146:153], v[60:67] /*v[572:579]*/
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[34:41], v[146:153], v[44:51] /*v[556:563]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[50:57], v[146:153], v[162:169] /*v[418:425]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[82:89], v[146:153], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa000
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[146:149], v2
	ds_load_tr16_b128 v[150:153], v2 offset:4352
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:192 nv
	scratch_load_b128 v[6:9], off, off offset:208 nv
	s_wait_loadcnt_dscnt 0x200
	v_wmma_f32_16x16x32_bf16 v[154:161], v[58:65], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2624 nv
	scratch_store_b128 off, v[158:161], off offset:2640 nv
	scratch_load_b128 v[154:157], off, off offset:2336 nv
	scratch_load_b128 v[158:161], off, off offset:2352 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[66:73], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2336 nv
	scratch_store_b128 off, v[158:161], off offset:2352 nv
	scratch_load_b128 v[154:157], off, off offset:1888 nv
	scratch_load_b128 v[158:161], off, off offset:1904 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[90:97], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1888 nv
	scratch_store_b128 off, v[158:161], off offset:1904 nv
	scratch_load_b128 v[154:157], off, off offset:352 nv
	scratch_load_b128 v[158:161], off, off offset:368 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[98:105], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:352 nv
	scratch_store_b128 off, v[158:161], off offset:368 nv
	scratch_load_b128 v[154:157], off, off offset:1600 nv
	scratch_load_b128 v[158:161], off, off offset:1616 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1600 nv
	scratch_store_b128 off, v[158:161], off offset:1616 nv
	scratch_load_b128 v[154:157], off, off offset:416 nv
	scratch_load_b128 v[158:161], off, off offset:432 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[114:121], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:416 nv
	scratch_store_b128 off, v[158:161], off offset:432 nv
	scratch_load_b128 v[154:157], off, off offset:1280 nv
	scratch_load_b128 v[158:161], off, off offset:1296 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[122:129], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1280 nv
	scratch_store_b128 off, v[158:161], off offset:1296 nv
	scratch_load_b128 v[154:157], off, off offset:960 nv
	scratch_load_b128 v[158:161], off, off offset:976 nv
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81], v[146:153], v[2:9]
	s_clause 0x3
	scratch_store_b128 off, v[2:5], off offset:192 nv
	scratch_store_b128 off, v[6:9], off offset:208 nv
	scratch_load_b128 v[2:5], off, off offset:3008 nv
	scratch_load_b128 v[6:9], off, off offset:3024 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[130:137], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:960 nv
	scratch_store_b128 off, v[158:161], off offset:976 nv
	scratch_load_b128 v[154:157], off, off offset:704 nv
	scratch_load_b128 v[158:161], off, off offset:720 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[82:89], v[146:153], v[2:9]
	s_clause 0x2
	scratch_store_b128 off, v[2:5], off offset:3008 nv
	scratch_store_b128 off, v[6:9], off offset:3024 nv
	scratch_load_b32 v2, off, off offset:8372 nv
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:704 nv
	scratch_store_b128 off, v[158:161], off offset:720 nv
	scratch_load_b128 v[154:157], off, off offset:2592 nv
	scratch_load_b128 v[158:161], off, off offset:2608 nv
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[18:25], v[146:153], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x50f0
	v_wmma_f32_16x16x32_bf16 v[196:203] /*v[964:971]*/, v[26:33], v[146:153], v[196:203] /*v[964:971]*/
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[948:955]*/, v[34:41], v[146:153], v[180:187] /*v[948:955]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xf0c3
	s_delay_alu instid0(TRANS32_DEP_1)
	v_mov_b64_e32 v[60:61] /*v[828:829]*/, v[180:181] /*v[948:949]*/
	s_set_vgpr_msb 0xc350
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[42:49], v[146:153], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x50c3
	v_mov_b64_e32 v[62:63] /*v[830:831]*/, v[182:183] /*v[950:951]*/
	v_mov_b64_e32 v[64:65] /*v[832:833]*/, v[184:185] /*v[952:953]*/
	v_mov_b64_e32 v[66:67] /*v[834:835]*/, v[186:187] /*v[954:955]*/
	s_set_vgpr_msb 0xc350
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[50:57], v[146:153], v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[146:149], v2
	ds_load_tr16_b128 v[150:153], v2 offset:4352
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:160 nv
	scratch_load_b128 v[6:9], off, off offset:176 nv
	s_wait_loadcnt_dscnt 0x200
	v_wmma_f32_16x16x32_bf16 v[154:161], v[58:65], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2592 nv
	scratch_store_b128 off, v[158:161], off offset:2608 nv
	scratch_load_b128 v[154:157], off, off offset:2304 nv
	scratch_load_b128 v[158:161], off, off offset:2320 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[66:73], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2304 nv
	scratch_store_b128 off, v[158:161], off offset:2320 nv
	scratch_load_b128 v[154:157], off, off offset:2208 nv
	scratch_load_b128 v[158:161], off, off offset:2224 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[82:89], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2208 nv
	scratch_store_b128 off, v[158:161], off offset:2224 nv
	scratch_load_b128 v[154:157], off, off offset:1856 nv
	scratch_load_b128 v[158:161], off, off offset:1872 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[90:97], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1856 nv
	scratch_store_b128 off, v[158:161], off offset:1872 nv
	scratch_load_b128 v[154:157], off, off offset:288 nv
	scratch_load_b128 v[158:161], off, off offset:304 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[98:105], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:288 nv
	scratch_store_b128 off, v[158:161], off offset:304 nv
	scratch_load_b128 v[154:157], off, off offset:1760 nv
	scratch_load_b128 v[158:161], off, off offset:1776 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1760 nv
	scratch_store_b128 off, v[158:161], off offset:1776 nv
	scratch_load_b128 v[154:157], off, off offset:1472 nv
	scratch_load_b128 v[158:161], off, off offset:1488 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[114:121], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1472 nv
	scratch_store_b128 off, v[158:161], off offset:1488 nv
	scratch_load_b128 v[154:157], off, off offset:1184 nv
	scratch_load_b128 v[158:161], off, off offset:1200 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[122:129], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1184 nv
	scratch_store_b128 off, v[158:161], off offset:1200 nv
	scratch_load_b128 v[154:157], off, off offset:928 nv
	scratch_load_b128 v[158:161], off, off offset:944 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[130:137], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:928 nv
	scratch_store_b128 off, v[158:161], off offset:944 nv
	scratch_load_b128 v[154:157], off, off offset:672 nv
	scratch_load_b128 v[158:161], off, off offset:688 nv
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81], v[146:153], v[2:9]
	s_clause 0x2
	scratch_store_b128 off, v[2:5], off offset:160 nv
	scratch_store_b128 off, v[6:9], off offset:176 nv
	scratch_load_b32 v2, off, off offset:8388 nv
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:672 nv
	scratch_store_b128 off, v[158:161], off offset:688 nv
	scratch_load_b128 v[154:157], off, off offset:2560 nv
	scratch_load_b128 v[158:161], off, off offset:2576 nv
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[18:25], v[146:153], v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[10:17], v[26:33], v[146:153], v[10:17]
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[666:673]*/, v[34:41], v[146:153], v[154:161] /*v[666:673]*/
	s_set_vgpr_msb 0xa0f0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[890:897]*/, v[42:49], v[146:153], v[122:129] /*v[890:897]*/
	s_set_vgpr_msb 0xf050
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[50:57], v[146:153], v[86:93] /*v[342:349]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[146:149], v2
	ds_load_tr16_b128 v[150:153], v2 offset:4352
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:128 nv
	scratch_load_b128 v[6:9], off, off offset:144 nv
	s_wait_loadcnt_dscnt 0x200
	v_wmma_f32_16x16x32_bf16 v[154:161], v[58:65], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2560 nv
	scratch_store_b128 off, v[158:161], off offset:2576 nv
	scratch_load_b128 v[154:157], off, off offset:2272 nv
	scratch_load_b128 v[158:161], off, off offset:2288 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[66:73], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2272 nv
	scratch_store_b128 off, v[158:161], off offset:2288 nv
	scratch_load_b128 v[154:157], off, off offset:2112 nv
	scratch_load_b128 v[158:161], off, off offset:2128 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[82:89], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2112 nv
	scratch_store_b128 off, v[158:161], off offset:2128 nv
	scratch_load_b128 v[154:157], off, off offset:1824 nv
	scratch_load_b128 v[158:161], off, off offset:1840 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[90:97], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1824 nv
	scratch_store_b128 off, v[158:161], off offset:1840 nv
	scratch_load_b128 v[154:157], off, off offset:2016 nv
	scratch_load_b128 v[158:161], off, off offset:2032 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[98:105], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2016 nv
	scratch_store_b128 off, v[158:161], off offset:2032 nv
	scratch_load_b128 v[154:157], off, off offset:1920 nv
	scratch_load_b128 v[158:161], off, off offset:1936 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1920 nv
	scratch_store_b128 off, v[158:161], off offset:1936 nv
	scratch_load_b128 v[154:157], off, off offset:1440 nv
	scratch_load_b128 v[158:161], off, off offset:1456 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[114:121], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1440 nv
	scratch_store_b128 off, v[158:161], off offset:1456 nv
	scratch_load_b128 v[154:157], off, off offset:1152 nv
	scratch_load_b128 v[158:161], off, off offset:1168 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[122:129], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1152 nv
	scratch_store_b128 off, v[158:161], off offset:1168 nv
	scratch_load_b128 v[154:157], off, off offset:896 nv
	scratch_load_b128 v[158:161], off, off offset:912 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[130:137], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:896 nv
	scratch_store_b128 off, v[158:161], off offset:912 nv
	scratch_load_b128 v[154:157], off, off offset:640 nv
	scratch_load_b128 v[158:161], off, off offset:656 nv
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81], v[146:153], v[2:9]
	s_clause 0x2
	scratch_store_b128 off, v[2:5], off offset:128 nv
	scratch_store_b128 off, v[6:9], off offset:144 nv
	scratch_load_b32 v2, off, off offset:8404 nv
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:640 nv
	scratch_store_b128 off, v[158:161], off offset:656 nv
	scratch_load_b128 v[154:157], off, off offset:2816 nv
	scratch_load_b128 v[158:161], off, off offset:2832 nv
	v_wmma_f32_16x16x32_bf16 v[182:189], v[18:25], v[146:153], v[182:189]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[26:33], v[146:153], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_store_b128 off, v[208:211] /*v[464:467]*/, off offset:2944 nv
	scratch_store_b128 off, v[212:215] /*v[468:471]*/, off offset:2960 nv
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[34:41], v[146:153], v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x50f0
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[972:979]*/, v[42:49], v[146:153], v[204:211] /*v[972:979]*/
	s_set_vgpr_msb 0xf050
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[50:57], v[146:153], v[78:85] /*v[334:341]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[146:149], v2
	ds_load_tr16_b128 v[150:153], v2 offset:4352
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:448 nv
	scratch_load_b128 v[6:9], off, off offset:464 nv
	s_wait_loadcnt_dscnt 0x200
	v_wmma_f32_16x16x32_bf16 v[154:161], v[26:33], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2816 nv
	scratch_store_b128 off, v[158:161], off offset:2832 nv
	scratch_load_b128 v[154:157], off, off offset:2752 nv
	scratch_load_b128 v[158:161], off, off offset:2768 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[42:49], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2752 nv
	scratch_store_b128 off, v[158:161], off offset:2768 nv
	scratch_load_b128 v[154:157], off, off offset:2528 nv
	scratch_load_b128 v[158:161], off, off offset:2544 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[58:65], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2528 nv
	scratch_store_b128 off, v[158:161], off offset:2544 nv
	scratch_load_b128 v[154:157], off, off offset:2240 nv
	scratch_load_b128 v[158:161], off, off offset:2256 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[82:89], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2240 nv
	scratch_store_b128 off, v[158:161], off offset:2256 nv
	scratch_load_b128 v[154:157], off, off offset:2144 nv
	scratch_load_b128 v[158:161], off, off offset:2160 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[90:97], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2144 nv
	scratch_store_b128 off, v[158:161], off offset:2160 nv
	scratch_load_b128 v[154:157], off, off offset:1792 nv
	scratch_load_b128 v[158:161], off, off offset:1808 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[98:105], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1792 nv
	scratch_store_b128 off, v[158:161], off offset:1808 nv
	scratch_load_b128 v[154:157], off, off offset:1696 nv
	scratch_load_b128 v[158:161], off, off offset:1712 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1696 nv
	scratch_store_b128 off, v[158:161], off offset:1712 nv
	scratch_load_b128 v[154:157], off, off offset:1344 nv
	scratch_load_b128 v[158:161], off, off offset:1360 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[114:121], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1344 nv
	scratch_store_b128 off, v[158:161], off offset:1360 nv
	scratch_load_b128 v[154:157], off, off offset:1120 nv
	scratch_load_b128 v[158:161], off, off offset:1136 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[122:129], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1120 nv
	scratch_store_b128 off, v[158:161], off offset:1136 nv
	scratch_load_b128 v[154:157], off, off offset:864 nv
	scratch_load_b128 v[158:161], off, off offset:880 nv
	v_wmma_f32_16x16x32_bf16 v[2:9], v[66:73], v[146:153], v[2:9]
	s_clause 0x3
	scratch_store_b128 off, v[2:5], off offset:448 nv
	scratch_store_b128 off, v[6:9], off offset:464 nv
	scratch_load_b128 v[2:5], off, off offset:96 nv
	scratch_load_b128 v[6:9], off, off offset:112 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[130:137], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:864 nv
	scratch_store_b128 off, v[158:161], off offset:880 nv
	scratch_load_b128 v[154:157], off, off offset:608 nv
	scratch_load_b128 v[158:161], off, off offset:624 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81], v[146:153], v[2:9]
	s_clause 0x2
	scratch_store_b128 off, v[2:5], off offset:96 nv
	scratch_store_b128 off, v[6:9], off offset:112 nv
	scratch_load_b32 v2, off, off offset:8420 nv
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:608 nv
	scratch_store_b128 off, v[158:161], off offset:624 nv
	scratch_load_b128 v[154:157], off, off offset:2496 nv
	scratch_load_b128 v[158:161], off, off offset:2512 nv
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[18:25], v[146:153], v[84:91] /*v[596:603]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[34:41], v[146:153], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[50:57], v[146:153], v[108:115] /*v[620:627]*/
	s_set_vgpr_msb 0xa000
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[146:149], v2
	ds_load_tr16_b128 v[150:153], v2 offset:4352
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:992 nv
	scratch_load_b128 v[6:9], off, off offset:1008 nv
	s_wait_loadcnt_dscnt 0x200
	v_wmma_f32_16x16x32_bf16 v[154:161], v[58:65], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2496 nv
	scratch_store_b128 off, v[158:161], off offset:2512 nv
	scratch_load_b128 v[154:157], off, off offset:2176 nv
	scratch_load_b128 v[158:161], off, off offset:2192 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[82:89], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2176 nv
	scratch_store_b128 off, v[158:161], off offset:2192 nv
	scratch_load_b128 v[154:157], off, off offset:2400 nv
	scratch_load_b128 v[158:161], off, off offset:2416 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[90:97], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2400 nv
	scratch_store_b128 off, v[158:161], off offset:2416 nv
	scratch_load_b128 v[154:157], off, off offset:1504 nv
	scratch_load_b128 v[158:161], off, off offset:1520 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[98:105], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1504 nv
	scratch_store_b128 off, v[158:161], off offset:1520 nv
	scratch_load_b128 v[154:157], off, off offset:1248 nv
	scratch_load_b128 v[158:161], off, off offset:1264 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1248 nv
	scratch_store_b128 off, v[158:161], off offset:1264 nv
	scratch_load_b128 v[154:157], off, off offset:1312 nv
	scratch_load_b128 v[158:161], off, off offset:1328 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[114:121], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1312 nv
	scratch_store_b128 off, v[158:161], off offset:1328 nv
	scratch_load_b128 v[154:157], off, off offset:1024 nv
	scratch_load_b128 v[158:161], off, off offset:1040 nv
	v_wmma_f32_16x16x32_bf16 v[2:9], v[50:57], v[146:153], v[2:9]
	s_clause 0x3
	scratch_store_b128 off, v[2:5], off offset:992 nv
	scratch_store_b128 off, v[6:9], off offset:1008 nv
	scratch_load_b128 v[2:5], off, off offset:384 nv
	scratch_load_b128 v[6:9], off, off offset:400 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[122:129], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1024 nv
	scratch_store_b128 off, v[158:161], off offset:1040 nv
	scratch_load_b128 v[154:157], off, off offset:832 nv
	scratch_load_b128 v[158:161], off, off offset:848 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[66:73], v[146:153], v[2:9]
	s_clause 0x3
	scratch_store_b128 off, v[2:5], off offset:384 nv
	scratch_store_b128 off, v[6:9], off offset:400 nv
	scratch_load_b128 v[2:5], off, off offset:64 nv
	scratch_load_b128 v[6:9], off, off offset:80 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[130:137], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:832 nv
	scratch_store_b128 off, v[158:161], off offset:848 nv
	scratch_load_b128 v[154:157], off, off offset:544 nv
	scratch_load_b128 v[158:161], off, off offset:560 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81], v[146:153], v[2:9]
	s_clause 0x2
	scratch_store_b128 off, v[2:5], off offset:64 nv
	scratch_store_b128 off, v[6:9], off offset:80 nv
	scratch_load_b32 v2, off, off offset:8436 nv
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:544 nv
	scratch_store_b128 off, v[158:161], off offset:560 nv
	scratch_load_b128 v[154:157], off, off offset:2720 nv
	scratch_load_b128 v[158:161], off, off offset:2736 nv
	v_wmma_f32_16x16x32_bf16 v[174:181], v[18:25], v[146:153], v[174:181]
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[26:33], v[146:153], v[170:177] /*v[682:689]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[34:41], v[146:153], v[70:77] /*v[326:333]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[42:49], v[146:153], v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x2
	ds_load_tr16_b128 v[146:149], v2
	ds_load_tr16_b128 v[150:153], v2 offset:4352
	s_clause 0x1
	scratch_load_b128 v[2:5], off, off offset:320 nv
	scratch_load_b128 v[6:9], off, off offset:336 nv
	s_wait_loadcnt_dscnt 0x200
	v_wmma_f32_16x16x32_bf16 v[154:161], v[50:57], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2720 nv
	scratch_store_b128 off, v[158:161], off offset:2736 nv
	scratch_load_b128 v[154:157], off, off offset:2464 nv
	scratch_load_b128 v[158:161], off, off offset:2480 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[58:65], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2464 nv
	scratch_store_b128 off, v[158:161], off offset:2480 nv
	scratch_load_b128 v[154:157], off, off offset:2048 nv
	scratch_load_b128 v[158:161], off, off offset:2064 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[82:89], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:2048 nv
	scratch_store_b128 off, v[158:161], off offset:2064 nv
	scratch_load_b128 v[154:157], off, off offset:1984 nv
	scratch_load_b128 v[158:161], off, off offset:2000 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[90:97], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1984 nv
	scratch_store_b128 off, v[158:161], off offset:2000 nv
	scratch_load_b128 v[154:157], off, off offset:1952 nv
	scratch_load_b128 v[158:161], off, off offset:1968 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[98:105], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1952 nv
	scratch_store_b128 off, v[158:161], off offset:1968 nv
	scratch_load_b128 v[154:157], off, off offset:1216 nv
	scratch_load_b128 v[158:161], off, off offset:1232 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[106:113], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1216 nv
	scratch_store_b128 off, v[158:161], off offset:1232 nv
	scratch_load_b128 v[154:157], off, off offset:1536 nv
	scratch_load_b128 v[158:161], off, off offset:1552 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[114:121], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:1536 nv
	scratch_store_b128 off, v[158:161], off offset:1552 nv
	scratch_load_b128 v[154:157], off, off offset:576 nv
	scratch_load_b128 v[158:161], off, off offset:592 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[122:129], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:576 nv
	scratch_store_b128 off, v[158:161], off offset:592 nv
	scratch_load_b128 v[154:157], off, off offset:800 nv
	scratch_load_b128 v[158:161], off, off offset:816 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[130:137], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:800 nv
	scratch_store_b128 off, v[158:161], off offset:816 nv
	scratch_load_b128 v[154:157], off, off offset:512 nv
	scratch_load_b128 v[158:161], off, off offset:528 nv
	v_wmma_f32_16x16x32_bf16 v[2:9], v[66:73], v[146:153], v[2:9]
	s_clause 0x3
	scratch_store_b128 off, v[2:5], off offset:320 nv
	scratch_store_b128 off, v[6:9], off offset:336 nv
	scratch_load_b128 v[2:5], off, off offset:32 nv
	scratch_load_b128 v[6:9], off, off offset:48 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145], v[146:153], v[154:161]
	s_clause 0x3
	scratch_store_b128 off, v[154:157], off offset:512 nv
	scratch_store_b128 off, v[158:161], off offset:528 nv
	scratch_load_b128 v[154:157], off, off offset:3424 th:TH_LOAD_LU nv
	scratch_load_b128 v[158:161], off, off offset:3440 th:TH_LOAD_LU nv
	v_wmma_f32_16x16x32_bf16 v[162:169], v[18:25], v[146:153], v[162:169]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[26:33], v[146:153], v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[318:325]*/, v[34:41], v[146:153], v[62:69] /*v[318:325]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[42:49], v[146:153], v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81], v[146:153], v[2:9]
	ds_load_tr16_b128 v[146:149], v1
	ds_load_tr16_b128 v[150:153], v1 offset:4352
	s_clause 0x1
	scratch_store_b128 off, v[2:5], off offset:32 nv
	scratch_store_b128 off, v[6:9], off offset:48 nv
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[154:161], v[18:25], v[146:153], v[154:161]
	s_clause 0x3
	scratch_load_b128 v[18:21], off, off offset:2688 nv
	scratch_load_b128 v[22:25], off, off offset:2704 nv
	scratch_load_b128 v[2:5], off, off offset:256 nv
	scratch_load_b128 v[6:9], off, off offset:272 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[18:25], v[50:57], v[146:153], v[18:25]
	s_clause 0x3
	scratch_store_b128 off, v[18:21], off offset:2688 nv
	scratch_store_b128 off, v[22:25], off offset:2704 nv
	scratch_load_b128 v[18:21], off, off offset:2432 nv
	scratch_load_b128 v[22:25], off, off offset:2448 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[58:65], v[146:153], v[18:25]
	s_clause 0x3
	scratch_store_b128 off, v[18:21], off offset:2432 nv
	scratch_store_b128 off, v[22:25], off offset:2448 nv
	scratch_load_b128 v[18:21], off, off offset:2784 nv
	scratch_load_b128 v[22:25], off, off offset:2800 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[90:97], v[146:153], v[18:25]
	s_clause 0x3
	scratch_store_b128 off, v[18:21], off offset:2784 nv
	scratch_store_b128 off, v[22:25], off offset:2800 nv
	scratch_load_b128 v[18:21], off, off offset:1568 nv
	scratch_load_b128 v[22:25], off, off offset:1584 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[106:113], v[146:153], v[18:25]
	s_clause 0x3
	scratch_store_b128 off, v[18:21], off offset:1568 nv
	scratch_store_b128 off, v[22:25], off offset:1584 nv
	scratch_load_b128 v[18:21], off, off offset:1376 nv
	scratch_load_b128 v[22:25], off, off offset:1392 nv
	v_wmma_f32_16x16x32_bf16 v[2:9], v[66:73], v[146:153], v[2:9]
	s_clause 0x1
	scratch_store_b128 off, v[2:5], off offset:256 nv
	scratch_store_b128 off, v[6:9], off offset:272 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[114:121], v[146:153], v[18:25]
	s_clause 0x5
	scratch_store_b128 off, v[18:21], off offset:1376 nv
	scratch_store_b128 off, v[22:25], off offset:1392 nv
	scratch_load_b128 v[2:5], off, off nv
	scratch_load_b128 v[6:9], off, off offset:16 nv
	scratch_load_b128 v[18:21], off, off offset:1088 nv
	scratch_load_b128 v[22:25], off, off offset:1104 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[74:81], v[146:153], v[2:9]
	s_clause 0x1
	scratch_store_b128 off, v[2:5], off nv
	scratch_store_b128 off, v[6:9], off offset:16 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[122:129], v[146:153], v[18:25]
	s_clause 0x5
	scratch_store_b128 off, v[18:21], off offset:1088 nv
	scratch_store_b128 off, v[22:25], off offset:1104 nv
	scratch_load_b128 v[2:5], off, off offset:2880 nv
	scratch_load_b128 v[6:9], off, off offset:2896 nv
	scratch_load_b128 v[18:21], off, off offset:768 nv
	scratch_load_b128 v[22:25], off, off offset:784 nv
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[82:89], v[146:153], v[2:9]
	s_clause 0x1
	scratch_store_b128 off, v[2:5], off offset:2880 nv
	scratch_store_b128 off, v[6:9], off offset:2896 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[130:137], v[146:153], v[18:25]
	s_clause 0x5
	scratch_store_b128 off, v[18:21], off offset:768 nv
	scratch_store_b128 off, v[22:25], off offset:784 nv
	scratch_load_b128 v[2:5], off, off offset:2848 nv
	scratch_load_b128 v[6:9], off, off offset:2864 nv
	scratch_load_b128 v[18:21], off, off offset:1632 nv
	scratch_load_b128 v[22:25], off, off offset:1648 nv
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[26:33], v[146:153], v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[34:41], v[146:153], v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[42:49], v[146:153], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5000
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[2:9], v[98:105], v[146:153], v[2:9]
	s_clause 0x1
	scratch_store_b128 off, v[2:5], off offset:2848 nv
	scratch_store_b128 off, v[6:9], off offset:2864 nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25], v[138:145], v[146:153], v[18:25]
	s_clause 0x1
	scratch_store_b128 off, v[18:21], off offset:1632 nv
	scratch_store_b128 off, v[22:25], off offset:1648 nv
	s_cbranch_scc1 .LBB0_7
	s_set_vgpr_msb 64
	scratch_load_b32 v155 /*v411*/, off, off offset:8836 nv
	s_set_vgpr_msb 0x4082
	v_mov_b64_e32 v[28:29] /*v[540:541]*/, v[60:61] /*v[572:573]*/
	v_mov_b64_e32 v[30:31] /*v[542:543]*/, v[62:63] /*v[574:575]*/
	v_mov_b64_e32 v[32:33] /*v[544:545]*/, v[64:65] /*v[576:577]*/
	v_mov_b64_e32 v[34:35] /*v[546:547]*/, v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x82c1
	v_mov_b64_e32 v[154:155] /*v[922:923]*/, v[222:223] /*v[478:479]*/
	v_mov_b64_e32 v[152:153] /*v[920:921]*/, v[220:221] /*v[476:477]*/
	v_mov_b64_e32 v[150:151] /*v[918:919]*/, v[218:219] /*v[474:475]*/
	v_mov_b64_e32 v[148:149] /*v[916:917]*/, v[216:217] /*v[472:473]*/
	s_set_vgpr_msb 0xc141
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[176:177] /*v[432:433]*/
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[174:175] /*v[430:431]*/
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[172:173] /*v[428:429]*/
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[170:171] /*v[426:427]*/
	s_set_vgpr_msb 0x4100
	v_mov_b64_e32 v[26:27], v[182:183]
	v_mov_b64_e32 v[28:29], v[184:185]
	v_mov_b64_e32 v[30:31], v[186:187]
	v_mov_b64_e32 v[32:33], v[188:189]
	s_set_vgpr_msb 0x42
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[84:85] /*v[596:597]*/
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[86:87] /*v[598:599]*/
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[88:89] /*v[600:601]*/
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x4200
	v_mov_b64_e32 v[106:107], v[174:175]
	v_mov_b64_e32 v[108:109], v[176:177]
	v_mov_b64_e32 v[110:111], v[178:179]
	v_mov_b64_e32 v[112:113], v[180:181]
	v_mov_b64_e32 v[212:213], v[16:17]
	v_mov_b64_e32 v[210:211], v[14:15]
	v_mov_b64_e32 v[208:209], v[12:13]
	v_mov_b64_e32 v[206:207], v[10:11]
.LBB0_9:
	s_clause 0x3
	scratch_load_b32 v35, off, off offset:6160 th:TH_LOAD_LU nv
	scratch_load_b128 v[36:39], off, off offset:3008 nv
	scratch_load_b128 v[40:43], off, off offset:3024 nv
	scratch_load_b32 v0, off, off offset:8980 nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x4
	v_or_b32_e32 v1, s44, v155 /*v411*/
	s_load_b64 s[0:1], s[0:1], 0x110 nv
	s_mul_i32 s4, s7, s46
	s_set_vgpr_msb 0x482
	v_mov_b64_e32 v[36:37] /*v[548:549]*/, v[28:29] /*v[540:541]*/
	s_add_co_i32 s45, s45, s4
	s_set_vgpr_msb 0x8200
	v_mul_lo_u32 v19, v1, s7
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0x800000
	s_set_vgpr_msb 0x82
	v_mov_b64_e32 v[38:39] /*v[550:551]*/, v[30:31] /*v[542:543]*/
	s_set_vgpr_msb 0x82c0
	v_mov_b64_e32 v[178:179] /*v[946:947]*/, v[32:33]
	v_mov_b64_e32 v[176:177] /*v[944:945]*/, v[30:31]
	v_mov_b64_e32 v[174:175] /*v[942:943]*/, v[28:29]
	v_mov_b64_e32 v[172:173] /*v[940:941]*/, v[26:27]
	s_set_vgpr_msb 0xc000
	v_add_lshl_u32 v19, s45, v19, 7
	s_set_vgpr_msb 0x82
	v_mov_b64_e32 v[40:41] /*v[552:553]*/, v[32:33] /*v[544:545]*/
	v_mov_b64_e32 v[42:43] /*v[554:555]*/, v[34:35] /*v[546:547]*/
	s_set_vgpr_msb 0x8201
	v_mov_b64_e32 v[10:11], v[224:225] /*v[480:481]*/
	v_mov_b64_e32 v[12:13], v[226:227] /*v[482:483]*/
	v_mov_b64_e32 v[14:15], v[228:229] /*v[484:485]*/
	v_mov_b64_e32 v[16:17], v[230:231] /*v[486:487]*/
	s_set_vgpr_msb 0x1c0
	s_clause 0x1
	scratch_load_b128 v[180:183] /*v[948:951]*/, off, off offset:2816 th:TH_LOAD_LU nv
	scratch_load_b128 v[184:187] /*v[952:955]*/, off, off offset:2832 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc002
	s_wait_kmcnt 0x0
	v_cvt_pk_bf16_f32 v18, v36 /*v548*/, s0
	v_cvt_pk_bf16_f32 v21, v37 /*v549*/, s0
	v_cvt_pk_bf16_f32 v23, v38 /*v550*/, s0
	v_cvt_pk_bf16_f32 v25, v39 /*v551*/, s0
	v_cvt_pk_bf16_f32 v27, v40 /*v552*/, s0
	v_cvt_pk_bf16_f32 v29, v41 /*v553*/, s0
	v_cvt_pk_bf16_f32 v31, v42 /*v554*/, s0
	s_set_vgpr_msb 0x200
	v_cvt_pk_bf16_f32 v2, v154, s0
	s_set_vgpr_msb 0x43
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[60:61] /*v[828:829]*/
	v_mov_b64_e32 v[32:33] /*v[288:289]*/, v[62:63] /*v[830:831]*/
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[64:65] /*v[832:833]*/
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[66:67] /*v[834:835]*/
	s_set_vgpr_msb 0x4302
	v_mov_b64_e32 v[44:45], v[10:11] /*v[522:523]*/
	v_mov_b64_e32 v[46:47], v[12:13] /*v[524:525]*/
	v_mov_b64_e32 v[48:49], v[14:15] /*v[526:527]*/
	v_mov_b64_e32 v[50:51], v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x200
	s_wait_loadcnt 0x5
	v_or_b32_e32 v20, v19, v35
	s_wait_loadcnt 0x2
	v_or_b32_e32 v19, v19, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 2, v20
	buffer_store_b16 v18, v20, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v18, 1, v1
	v_mul_lo_u32 v18, v18, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v18, s45, v18, 7
	v_or_b32_e32 v22, v18, v35
	v_or_b32_e32 v18, v18, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 2, v22
	buffer_store_b16 v21, v22, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v21, 2, v1
	v_mul_lo_u32 v21, v21, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v21, s45, v21, 7
	v_or_b32_e32 v24, v21, v35
	v_or_b32_e32 v21, v21, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 2, v24
	buffer_store_b16 v23, v24, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v23, 3, v1
	v_mul_lo_u32 v23, v23, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v23, s45, v23, 7
	v_or_b32_e32 v26, v23, v35
	v_or_b32_e32 v23, v23, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v26, 2, v26
	v_lshlrev_b32_e32 v19, 2, v19
	buffer_store_b16 v25, v26, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v25, 4, v1
	v_or_b32_e32 v34, 64, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v25, v25, s7
	v_add_lshl_u32 v25, s45, v25, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_or_b32_e32 v28, v25, v35
	v_or_b32_e32 v25, v25, v0
	v_dual_lshlrev_b32 v28, 2, v28 :: v_dual_lshlrev_b32 v18, 2, v18
	buffer_store_b16 v27, v28, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v27, 5, v1
	v_or_b32_e32 v3, 0x1c0, v18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v27, v27, s7
	v_add_lshl_u32 v27, s45, v27, 7
	v_lshlrev_b32_e32 v21, 2, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_or_b32_e32 v30, v27, v35
	v_or_b32_e32 v27, v27, v0
	v_dual_lshlrev_b32 v30, 2, v30 :: v_dual_lshlrev_b32 v23, 2, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v27, 2, v27
	buffer_store_b16 v29, v30, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v29, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v29, v29, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_lo_u32 v1, v1, s7
	v_lshlrev_b32_e32 v25, 2, v25
	v_add_lshl_u32 v29, s45, v29, 7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, s45, v1, 7
	v_or_b32_e32 v32, v29, v35
	v_or_b32_e32 v29, v29, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v32, 2, v32
	v_dual_lshlrev_b32 v29, 2, v29 :: v_dual_bitop2_b32 v33, v1, v35 bitop3:0x54
	v_or_b32_e32 v1, v1, v0
	buffer_store_b16 v31, v32, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v31, v43 /*v555*/, s0
	v_lshlrev_b32_e32 v33, 2, v33
	v_lshlrev_b32_e32 v1, 2, v1
	s_set_vgpr_msb 0x200
	buffer_store_b16 v31, v33, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v148 /*v916*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v149 /*v917*/, s0
	v_or_b32_e32 v34, 64, v18
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v150 /*v918*/, s0
	v_or_b32_e32 v34, 64, v21
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v151 /*v919*/, s0
	v_or_b32_e32 v34, 64, v23
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v152 /*v920*/, s0
	v_or_b32_e32 v34, 64, v25
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v153 /*v921*/, s0
	v_or_b32_e32 v34, 64, v27
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v154 /*v922*/, s0
	v_or_b32_e32 v34, 64, v29
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v155 /*v923*/, s0
	v_or_b32_e32 v34, 64, v1
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v31, v216 /*v472*/, s0
	v_or_b32_e32 v34, 0xc0, v19
	s_set_vgpr_msb 0x100
	buffer_store_b16 v31, v20, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v31, v217 /*v473*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v31, v22, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v31, v218 /*v474*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v31, v24, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v31, v219 /*v475*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v31, v26, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v31, v220 /*v476*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v31, v28, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v31, v221 /*v477*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v31, v30, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v31, v222 /*v478*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v31, v32, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v31, v223 /*v479*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v31, v33, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v172 /*v940*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v173 /*v941*/, s0
	v_or_b32_e32 v34, 0xc0, v18
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v174 /*v942*/, s0
	v_or_b32_e32 v34, 0xc0, v21
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v175 /*v943*/, s0
	v_or_b32_e32 v34, 0xc0, v23
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v176 /*v944*/, s0
	v_or_b32_e32 v34, 0xc0, v25
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v177 /*v945*/, s0
	v_or_b32_e32 v34, 0xc0, v27
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v178 /*v946*/, s0
	v_or_b32_e32 v34, 0xc0, v29
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v31, v179 /*v947*/, s0
	v_or_b32_e32 v34, 0xc0, v1
	s_set_vgpr_msb 0x300
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v10, s0
	v_cvt_pk_bf16_f32 v10, v162, s0
	v_or_b32_e32 v34, 0x140, v19
	buffer_store_b16 v31, v20, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v11, s0
	buffer_store_b16 v31, v22, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v12, s0
	buffer_store_b16 v31, v24, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v13, s0
	buffer_store_b16 v31, v26, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v14, s0
	buffer_store_b16 v31, v28, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v15, s0
	buffer_store_b16 v31, v30, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v16, s0
	buffer_store_b16 v31, v32, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v17, s0
	buffer_store_b16 v31, v33, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v106, s0
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v107, s0
	v_or_b32_e32 v34, 0x140, v18
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v108, s0
	v_or_b32_e32 v34, 0x140, v21
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v109, s0
	v_or_b32_e32 v34, 0x140, v23
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v110, s0
	v_or_b32_e32 v34, 0x140, v25
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v111, s0
	v_or_b32_e32 v34, 0x140, v27
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v112, s0
	v_or_b32_e32 v34, 0x140, v29
	buffer_store_b16 v31, v34, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v31, v113, s0
	v_or_b32_e32 v34, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	s_clause 0x1
	buffer_store_b16 v31, v34, s[0:3], null offen
	buffer_store_b16 v10, v20, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v10, v163, s0
	buffer_store_b16 v10, v22, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v10, v164, s0
	buffer_store_b16 v10, v24, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v10, v165, s0
	buffer_store_b16 v10, v26, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v10, v166, s0
	buffer_store_b16 v10, v28, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v10, v167, s0
	buffer_store_b16 v10, v30, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v10, v168, s0
	buffer_store_b16 v10, v32, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v10, v169, s0
	buffer_store_b16 v10, v33, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_or_b32_e32 v10, 0x1c0, v19
	buffer_store_b16 v2, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v155, s0
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:2912 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:2928 th:TH_LOAD_LU nv
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v156, s0
	v_or_b32_e32 v3, 0x1c0, v21
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v157, s0
	v_or_b32_e32 v3, 0x1c0, v23
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v158, s0
	v_or_b32_e32 v3, 0x1c0, v25
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v159, s0
	v_or_b32_e32 v3, 0x1c0, v27
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v160, s0
	v_or_b32_e32 v3, 0x1c0, v29
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v161, s0
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v1, s43, v155 /*v411*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v1, s7
	v_add_lshl_u32 v3, v3, s45, 7
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_lshlrev_b32_e32 v4, 2, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v3, 2, v3
	v_or_b32_e32 v18, 64, v3
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v2, v10, s0
	v_cvt_pk_bf16_f32 v5, v11, s0
	v_cvt_pk_bf16_f32 v7, v12, s0
	v_cvt_pk_bf16_f32 v9, v13, s0
	s_wait_loadcnt 0x0
	v_cvt_pk_bf16_f32 v11, v14, s0
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_cvt_pk_bf16_f32 v13, v15, s0
	v_cvt_pk_bf16_f32 v15, v16, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_or_b32_e32 v6, v2, v35
	v_or_b32_e32 v2, v2, v0
	v_lshlrev_b32_e32 v6, 2, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	v_or_b32_e32 v7, v7, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_lshlrev_b32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	v_mul_lo_u32 v9, v9, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v9, v9, s45, 7
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	v_mul_lo_u32 v11, v11, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v11, v11, s45, 7
	v_or_b32_e32 v14, v11, v35
	v_or_b32_e32 v11, v11, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_lshlrev_b32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	v_or_b32_e32 v13, v13, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v13, 2, v13
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v17, s0
	v_or_b32_e32 v17, v1, v35
	v_dual_lshlrev_b32 v17, 2, v17 :: v_dual_bitop2_b32 v1, v1, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v196 /*v964*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v197 /*v965*/, s0
	v_or_b32_e32 v18, 64, v2
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v198 /*v966*/, s0
	v_or_b32_e32 v18, 64, v5
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v199 /*v967*/, s0
	v_or_b32_e32 v18, 64, v7
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v200 /*v968*/, s0
	v_or_b32_e32 v18, 64, v9
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v201 /*v969*/, s0
	v_or_b32_e32 v18, 64, v11
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v202 /*v970*/, s0
	v_or_b32_e32 v18, 64, v13
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v203 /*v971*/, s0
	v_or_b32_e32 v18, 64, v1
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2944 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2960 th:TH_LOAD_LU nv
	v_cvt_pk_bf16_f32 v15, v206, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v207, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v208, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v209, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v210, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v211, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v212, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v213, s0
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v180 /*v948*/, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v181 /*v949*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v182 /*v950*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v183 /*v951*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v184 /*v952*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v185 /*v953*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v186 /*v954*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v187 /*v955*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v170 /*v682*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v171 /*v683*/, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v172 /*v684*/, s0
	v_or_b32_e32 v18, 0x140, v5
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v173 /*v685*/, s0
	v_or_b32_e32 v18, 0x140, v7
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v174 /*v686*/, s0
	v_or_b32_e32 v18, 0x140, v9
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v175 /*v687*/, s0
	v_or_b32_e32 v18, 0x140, v11
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v176 /*v688*/, s0
	v_or_b32_e32 v18, 0x140, v13
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v177 /*v689*/, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v186 /*v442*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v187 /*v443*/, s0
	s_set_vgpr_msb 0x102
	v_cvt_pk_bf16_f32 v15, v50 /*v562*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v188 /*v444*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v189 /*v445*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v190 /*v446*/, s0
	s_set_vgpr_msb 0x1c0
	s_clause 0x7
	scratch_load_b128 v[28:31] /*v[796:799]*/, off, off offset:2976 th:TH_LOAD_LU nv
	scratch_load_b128 v[32:35] /*v[800:803]*/, off, off offset:2992 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[28:31] /*v[540:543]*/, off, off offset:2432 th:TH_LOAD_LU nv
	scratch_load_b128 v[32:35] /*v[544:547]*/, off, off offset:2448 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[130:133] /*v[898:901]*/, off, off offset:2176 th:TH_LOAD_LU nv
	scratch_load_b128 v[134:137] /*v[902:905]*/, off, off offset:2192 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc000
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v191 /*v447*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v192 /*v448*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v193 /*v449*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v138 /*v394*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v3, v139 /*v395*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v140 /*v396*/, s0
	v_or_b32_e32 v3, 0x1c0, v5
	s_set_vgpr_msb 0x102
	v_cvt_pk_bf16_f32 v5, v45 /*v557*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v141 /*v397*/, s0
	v_or_b32_e32 v3, 0x1c0, v7
	s_set_vgpr_msb 0x102
	v_cvt_pk_bf16_f32 v7, v46 /*v558*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v142 /*v398*/, s0
	v_or_b32_e32 v3, 0x1c0, v9
	s_set_vgpr_msb 0x102
	v_cvt_pk_bf16_f32 v9, v47 /*v559*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v143 /*v399*/, s0
	v_or_b32_e32 v3, 0x1c0, v11
	s_set_vgpr_msb 0x102
	v_cvt_pk_bf16_f32 v11, v48 /*v560*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v144 /*v400*/, s0
	v_or_b32_e32 v3, 0x1c0, v13
	s_set_vgpr_msb 0x102
	v_cvt_pk_bf16_f32 v13, v49 /*v561*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v145 /*v401*/, s0
	s_set_vgpr_msb 0x104
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v1, s42, v155 /*v411*/
	s_set_vgpr_msb 0x402
	v_cvt_pk_bf16_f32 v2, v44 /*v556*/, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, s7, v1
	v_add_lshl_u32 v3, s45, v3, 7
	s_set_vgpr_msb 0x200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_or_b32_e32 v18, 64, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_or_b32_e32 v17, v1, v35
	v_or_b32_e32 v13, v13, v0
	v_or_b32_e32 v1, v1, v0
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v17, 2, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v51 /*v563*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v30 /*v286*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v31 /*v287*/, s0
	v_or_b32_e32 v18, 64, v2
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v32 /*v288*/, s0
	v_or_b32_e32 v18, 64, v5
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v33 /*v289*/, s0
	v_or_b32_e32 v18, 64, v7
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v34 /*v290*/, s0
	v_or_b32_e32 v18, 64, v9
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v35 /*v291*/, s0
	v_or_b32_e32 v18, 64, v11
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v36 /*v292*/, s0
	v_or_b32_e32 v18, 64, v13
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v37 /*v293*/, s0
	v_or_b32_e32 v18, 64, v1
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v154 /*v666*/, s0
	v_or_b32_e32 v18, 0xc0, v3
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v155 /*v667*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v156 /*v668*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v157 /*v669*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v158 /*v670*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v159 /*v671*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v160 /*v672*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v161 /*v673*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v130 /*v386*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v131 /*v387*/, s0
	v_or_b32_e32 v18, 0xc0, v2
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v132 /*v388*/, s0
	v_or_b32_e32 v18, 0xc0, v5
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v133 /*v389*/, s0
	v_or_b32_e32 v18, 0xc0, v7
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v134 /*v390*/, s0
	v_or_b32_e32 v18, 0xc0, v9
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v135 /*v391*/, s0
	v_or_b32_e32 v18, 0xc0, v11
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v136 /*v392*/, s0
	v_or_b32_e32 v18, 0xc0, v13
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v137 /*v393*/, s0
	v_or_b32_e32 v18, 0xc0, v1
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v122 /*v378*/, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v123 /*v379*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v124 /*v380*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v125 /*v381*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v126 /*v382*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v127 /*v383*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v128 /*v384*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v129 /*v385*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v70 /*v326*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v71 /*v327*/, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v72 /*v328*/, s0
	v_or_b32_e32 v18, 0x140, v5
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v73 /*v329*/, s0
	v_or_b32_e32 v18, 0x140, v7
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v74 /*v330*/, s0
	v_or_b32_e32 v18, 0x140, v9
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v75 /*v331*/, s0
	v_or_b32_e32 v18, 0x140, v11
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v76 /*v332*/, s0
	v_or_b32_e32 v18, 0x140, v13
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v77 /*v333*/, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v62 /*v318*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v63 /*v319*/, s0
	s_set_vgpr_msb 0x103
	s_wait_loadcnt 0x4
	v_cvt_pk_bf16_f32 v15, v34 /*v802*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v64 /*v320*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v65 /*v321*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v66 /*v322*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v67 /*v323*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v68 /*v324*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v69 /*v325*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v52 /*v564*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v3, v53 /*v565*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v54 /*v566*/, s0
	v_or_b32_e32 v3, 0x1c0, v5
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v5, v29 /*v797*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v55 /*v567*/, s0
	v_or_b32_e32 v3, 0x1c0, v7
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v7, v30 /*v798*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v56 /*v568*/, s0
	v_or_b32_e32 v3, 0x1c0, v9
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v9, v31 /*v799*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v57 /*v569*/, s0
	v_or_b32_e32 v3, 0x1c0, v11
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v11, v32 /*v800*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v58 /*v570*/, s0
	v_or_b32_e32 v3, 0x1c0, v13
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v13, v33 /*v801*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v59 /*v571*/, s0
	s_set_vgpr_msb 0x204
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v1, s41, v155 /*v411*/
	s_set_vgpr_msb 0x403
	v_cvt_pk_bf16_f32 v2, v28 /*v796*/, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, s7, v1
	v_add_lshl_u32 v3, s45, v3, 7
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_or_b32_e32 v18, 64, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_or_b32_e32 v17, v1, v35
	v_or_b32_e32 v13, v13, v0
	v_or_b32_e32 v1, v1, v0
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v17, 2, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v35 /*v803*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v194 /*v450*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v195 /*v451*/, s0
	v_or_b32_e32 v18, 64, v2
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v196 /*v452*/, s0
	v_or_b32_e32 v18, 64, v5
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v197 /*v453*/, s0
	v_or_b32_e32 v18, 64, v7
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v198 /*v454*/, s0
	v_or_b32_e32 v18, 64, v9
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v199 /*v455*/, s0
	v_or_b32_e32 v18, 64, v11
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v200 /*v456*/, s0
	v_or_b32_e32 v18, 64, v13
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v201 /*v457*/, s0
	v_or_b32_e32 v18, 64, v1
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v122 /*v890*/, s0
	v_or_b32_e32 v18, 0xc0, v3
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v123 /*v891*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v124 /*v892*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v125 /*v893*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v126 /*v894*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v127 /*v895*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v128 /*v896*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v129 /*v897*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v204 /*v972*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v205 /*v973*/, s0
	v_or_b32_e32 v18, 0xc0, v2
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v206 /*v974*/, s0
	v_or_b32_e32 v18, 0xc0, v5
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v207 /*v975*/, s0
	v_or_b32_e32 v18, 0xc0, v7
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v208 /*v976*/, s0
	v_or_b32_e32 v18, 0xc0, v9
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v209 /*v977*/, s0
	v_or_b32_e32 v18, 0xc0, v11
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v210 /*v978*/, s0
	v_or_b32_e32 v18, 0xc0, v13
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v211 /*v979*/, s0
	v_or_b32_e32 v18, 0xc0, v1
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2752 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2768 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v178 /*v434*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v179 /*v435*/, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v180 /*v436*/, s0
	v_or_b32_e32 v18, 0x140, v5
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v181 /*v437*/, s0
	v_or_b32_e32 v18, 0x140, v7
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v182 /*v438*/, s0
	v_or_b32_e32 v18, 0x140, v9
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v183 /*v439*/, s0
	v_or_b32_e32 v18, 0x140, v11
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v184 /*v440*/, s0
	v_or_b32_e32 v18, 0x140, v13
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v185 /*v441*/, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v114 /*v370*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v115 /*v371*/, s0
	v_cvt_pk_bf16_f32 v15, v168 /*v424*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v116 /*v372*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v117 /*v373*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v118 /*v374*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v119 /*v375*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v120 /*v376*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v121 /*v377*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v4, v106 /*v362*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v3, v107 /*v363*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v108 /*v364*/, s0
	v_or_b32_e32 v3, 0x1c0, v5
	v_cvt_pk_bf16_f32 v5, v163 /*v419*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v109 /*v365*/, s0
	v_or_b32_e32 v3, 0x1c0, v7
	v_cvt_pk_bf16_f32 v7, v164 /*v420*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v110 /*v366*/, s0
	v_or_b32_e32 v3, 0x1c0, v9
	v_cvt_pk_bf16_f32 v9, v165 /*v421*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v111 /*v367*/, s0
	v_or_b32_e32 v3, 0x1c0, v11
	v_cvt_pk_bf16_f32 v11, v166 /*v422*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v112 /*v368*/, s0
	v_or_b32_e32 v3, 0x1c0, v13
	v_cvt_pk_bf16_f32 v13, v167 /*v423*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v2, v113 /*v369*/, s0
	s_set_vgpr_msb 0x104
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v1, s40, v155 /*v411*/
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v2, v162 /*v418*/, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, s7, v1
	v_add_lshl_u32 v3, s45, v3, 7
	s_set_vgpr_msb 0x100
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_or_b32_e32 v18, 64, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_or_b32_e32 v17, v1, v35
	v_or_b32_e32 v13, v13, v0
	v_or_b32_e32 v1, v1, v0
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v17, 2, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v169 /*v425*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v146 /*v402*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v147 /*v403*/, s0
	v_or_b32_e32 v18, 64, v2
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v148 /*v404*/, s0
	v_or_b32_e32 v18, 64, v5
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v149 /*v405*/, s0
	v_or_b32_e32 v18, 64, v7
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v150 /*v406*/, s0
	v_or_b32_e32 v18, 64, v9
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v151 /*v407*/, s0
	v_or_b32_e32 v18, 64, v11
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v152 /*v408*/, s0
	v_or_b32_e32 v18, 64, v13
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v153 /*v409*/, s0
	v_or_b32_e32 v18, 64, v1
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v86 /*v342*/, s0
	v_or_b32_e32 v18, 0xc0, v3
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v87 /*v343*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v88 /*v344*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v89 /*v345*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v90 /*v346*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v91 /*v347*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v92 /*v348*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v93 /*v349*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v78 /*v334*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v79 /*v335*/, s0
	v_or_b32_e32 v18, 0xc0, v2
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v80 /*v336*/, s0
	v_or_b32_e32 v18, 0xc0, v5
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v81 /*v337*/, s0
	v_or_b32_e32 v18, 0xc0, v7
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v82 /*v338*/, s0
	v_or_b32_e32 v18, 0xc0, v9
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v83 /*v339*/, s0
	v_or_b32_e32 v18, 0xc0, v11
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v84 /*v340*/, s0
	v_or_b32_e32 v18, 0xc0, v13
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v15, v85 /*v341*/, s0
	v_or_b32_e32 v18, 0xc0, v1
	s_set_vgpr_msb 0x100
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:992 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1008 th:TH_LOAD_LU nv
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v108 /*v620*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v109 /*v621*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v110 /*v622*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v111 /*v623*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v112 /*v624*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v113 /*v625*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v114 /*v626*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v115 /*v627*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0x140, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0x140, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0x140, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0x140, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0x140, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2720 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2736 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v19, s0
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v20, s0
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v21, s0
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v22, s0
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v23, s0
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v24, s0
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v25, s0
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_clause 0x1
	scratch_load_b128 v[14:17], off, off offset:2688 th:TH_LOAD_LU nv
	scratch_load_b128 v[18:21], off, off offset:2704 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v4, v14, s0
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v15, s0
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v16, s0
	v_or_b32_e32 v3, 0x1c0, v5
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v17, s0
	v_or_b32_e32 v3, 0x1c0, v7
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v18, s0
	v_or_b32_e32 v3, 0x1c0, v9
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v19, s0
	v_or_b32_e32 v3, 0x1c0, v11
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v13
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:2656 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:2672 th:TH_LOAD_LU nv
	v_cvt_pk_bf16_f32 v2, v20, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v21, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2624 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2640 th:TH_LOAD_LU nv
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v1, s39, v155 /*v411*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v1, s7
	v_add_lshl_u32 v3, v3, s45, 7
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	s_wait_loadcnt 0x3
	v_cvt_pk_bf16_f32 v2, v10, s0
	v_cvt_pk_bf16_f32 v5, v11, s0
	v_cvt_pk_bf16_f32 v7, v12, s0
	v_cvt_pk_bf16_f32 v9, v13, s0
	s_wait_loadcnt 0x2
	v_cvt_pk_bf16_f32 v11, v14, s0
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_cvt_pk_bf16_f32 v13, v15, s0
	v_cvt_pk_bf16_f32 v15, v16, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	v_or_b32_e32 v13, v13, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v13, 2, v13
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v17, s0
	v_or_b32_e32 v17, v1, v35
	v_dual_lshlrev_b32 v17, 2, v17 :: v_dual_bitop2_b32 v1, v1, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 64, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 64, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2592 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2608 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2560 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2576 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2528 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2544 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2496 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2512 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0x140, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0x140, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0x140, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0x140, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0x140, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2464 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2480 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v19, s0
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v20, s0
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v21, s0
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v22, s0
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v23, s0
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v24, s0
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2336 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2352 th:TH_LOAD_LU nv
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v28 /*v540*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v3, v29 /*v541*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v30 /*v542*/, s0
	v_or_b32_e32 v3, 0x1c0, v5
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v31 /*v543*/, s0
	v_or_b32_e32 v3, 0x1c0, v7
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v32 /*v544*/, s0
	v_or_b32_e32 v3, 0x1c0, v9
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v33 /*v545*/, s0
	v_or_b32_e32 v3, 0x1c0, v11
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v13
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:2368 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:2384 th:TH_LOAD_LU nv
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v34 /*v546*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v35 /*v547*/, s0
	s_set_vgpr_msb 0x204
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v1, s38, v155 /*v411*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v1, s7
	v_add_lshl_u32 v3, v3, s45, 7
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v2, v10, s0
	v_cvt_pk_bf16_f32 v5, v11, s0
	v_cvt_pk_bf16_f32 v7, v12, s0
	v_cvt_pk_bf16_f32 v9, v13, s0
	s_wait_loadcnt 0x0
	v_cvt_pk_bf16_f32 v11, v14, s0
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_cvt_pk_bf16_f32 v13, v15, s0
	v_cvt_pk_bf16_f32 v15, v16, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	v_or_b32_e32 v13, v13, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v13, 2, v13
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v17, s0
	v_or_b32_e32 v17, v1, v35
	v_dual_lshlrev_b32 v17, 2, v17 :: v_dual_bitop2_b32 v1, v1, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 64, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 64, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2304 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2320 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2272 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2288 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:448 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:464 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:384 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:400 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0x140, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0x140, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0x140, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0x140, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0x140, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:320 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:336 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v19, s0
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v20, s0
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v21, s0
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v22, s0
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v23, s0
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v24, s0
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v25, s0
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_clause 0x1
	scratch_load_b128 v[14:17], off, off offset:256 th:TH_LOAD_LU nv
	scratch_load_b128 v[18:21], off, off offset:272 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v4, v14, s0
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v15, s0
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v16, s0
	v_or_b32_e32 v3, 0x1c0, v5
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v17, s0
	v_or_b32_e32 v3, 0x1c0, v7
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v18, s0
	v_or_b32_e32 v3, 0x1c0, v9
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v19, s0
	v_or_b32_e32 v3, 0x1c0, v11
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v13
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:224 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:240 th:TH_LOAD_LU nv
	v_cvt_pk_bf16_f32 v2, v20, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v21, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:192 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:208 th:TH_LOAD_LU nv
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v1, s37, v155 /*v411*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v1, s7
	v_add_lshl_u32 v3, v3, s45, 7
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	s_wait_loadcnt 0x3
	v_cvt_pk_bf16_f32 v2, v10, s0
	v_cvt_pk_bf16_f32 v5, v11, s0
	v_cvt_pk_bf16_f32 v7, v12, s0
	v_cvt_pk_bf16_f32 v9, v13, s0
	s_wait_loadcnt 0x2
	v_cvt_pk_bf16_f32 v11, v14, s0
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_cvt_pk_bf16_f32 v13, v15, s0
	v_cvt_pk_bf16_f32 v15, v16, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	v_or_b32_e32 v13, v13, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v13, 2, v13
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v17, s0
	v_or_b32_e32 v17, v1, v35
	v_dual_lshlrev_b32 v17, 2, v17 :: v_dual_bitop2_b32 v1, v1, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 64, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 64, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:160 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:176 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:128 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:144 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:96 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:112 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:64 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:80 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0x140, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0x140, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0x140, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0x140, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0x140, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:32 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:48 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v19, s0
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v20, s0
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v21, s0
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v22, s0
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v23, s0
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v24, s0
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v25, s0
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_clause 0x1
	scratch_load_b128 v[14:17], off, off th:TH_LOAD_LU nv
	scratch_load_b128 v[18:21], off, off offset:16 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v4, v14, s0
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v15, s0
	v_cvt_pk_bf16_f32 v15, v50, s0
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v16, s0
	v_or_b32_e32 v3, 0x1c0, v5
	v_cvt_pk_bf16_f32 v5, v45, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v17, s0
	v_or_b32_e32 v3, 0x1c0, v7
	v_cvt_pk_bf16_f32 v7, v46, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v18, s0
	v_or_b32_e32 v3, 0x1c0, v9
	v_cvt_pk_bf16_f32 v9, v47, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v19, s0
	v_or_b32_e32 v3, 0x1c0, v11
	v_cvt_pk_bf16_f32 v11, v48, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v20, s0
	v_or_b32_e32 v3, 0x1c0, v13
	v_cvt_pk_bf16_f32 v13, v49, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v21, s0
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v1, s36, v155 /*v411*/
	v_cvt_pk_bf16_f32 v2, v44, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v1, s7
	v_add_lshl_u32 v3, v3, s45, 7
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_or_b32_e32 v18, 64, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_or_b32_e32 v17, v1, v35
	v_or_b32_e32 v13, v13, v0
	v_or_b32_e32 v1, v1, v0
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v17, 2, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v51, s0
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v36, s0
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v37, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v38, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v39, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v40, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v41, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v42, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v43, s0
	v_or_b32_e32 v18, 64, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2208 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2224 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2112 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2128 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2240 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2256 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v130 /*v898*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v131 /*v899*/, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v132 /*v900*/, s0
	v_or_b32_e32 v18, 0x140, v5
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v133 /*v901*/, s0
	v_or_b32_e32 v18, 0x140, v7
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v134 /*v902*/, s0
	v_or_b32_e32 v18, 0x140, v9
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v135 /*v903*/, s0
	v_or_b32_e32 v18, 0x140, v11
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v136 /*v904*/, s0
	v_or_b32_e32 v18, 0x140, v13
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v137 /*v905*/, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2048 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2064 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v19, s0
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v20, s0
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v21, s0
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v22, s0
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v23, s0
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v24, s0
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v25, s0
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_clause 0x1
	scratch_load_b128 v[14:17], off, off offset:2880 th:TH_LOAD_LU nv
	scratch_load_b128 v[18:21], off, off offset:2896 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v4, v14, s0
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v15, s0
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v16, s0
	v_or_b32_e32 v3, 0x1c0, v5
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v17, s0
	v_or_b32_e32 v3, 0x1c0, v7
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v18, s0
	v_or_b32_e32 v3, 0x1c0, v9
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v19, s0
	v_or_b32_e32 v3, 0x1c0, v11
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v13
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:480 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:496 th:TH_LOAD_LU nv
	v_cvt_pk_bf16_f32 v2, v20, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v21, s0
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v1, s35, v155 /*v411*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v1, s7
	v_add_lshl_u32 v3, v3, s45, 7
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v2, v10, s0
	v_cvt_pk_bf16_f32 v5, v11, s0
	v_cvt_pk_bf16_f32 v7, v12, s0
	v_cvt_pk_bf16_f32 v9, v13, s0
	s_wait_loadcnt 0x0
	v_cvt_pk_bf16_f32 v11, v14, s0
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_cvt_pk_bf16_f32 v13, v15, s0
	v_cvt_pk_bf16_f32 v15, v16, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1888 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1904 th:TH_LOAD_LU nv
	s_wait_xcnt 0x2
	v_or_b32_e32 v9, 4, v1
	s_set_vgpr_msb 0xc0
	s_clause 0x8
	scratch_load_b128 v[130:133] /*v[898:901]*/, off, off offset:2144 th:TH_LOAD_LU nv
	scratch_load_b128 v[134:137] /*v[902:905]*/, off, off offset:2160 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[28:31] /*v[540:543]*/, off, off offset:2400 th:TH_LOAD_LU nv
	scratch_load_b128 v[32:35] /*v[544:547]*/, off, off offset:2416 th:TH_LOAD_LU nv
	scratch_load_b128 v[20:23] /*v[532:535]*/, off, off offset:1984 th:TH_LOAD_LU nv
	scratch_load_b128 v[24:27] /*v[536:539]*/, off, off offset:2000 th:TH_LOAD_LU nv
	scratch_load_b128 v[12:15] /*v[524:527]*/, off, off offset:2784 th:TH_LOAD_LU nv
	scratch_load_b128 v[16:19] /*v[528:531]*/, off, off offset:2800 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x8000
	v_mul_lo_u32 v9, v9, s7
	s_set_vgpr_msb 0xc0
	s_clause 0xf
	scratch_load_b128 v[204:207] /*v[972:975]*/, off, off offset:1952 th:TH_LOAD_LU nv
	scratch_load_b128 v[208:211] /*v[976:979]*/, off, off offset:1968 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc080
	scratch_load_b128 v[68:71] /*v[580:583]*/, off, off offset:2848 th:TH_LOAD_LU nv
	scratch_load_b128 v[72:75] /*v[584:587]*/, off, off offset:2864 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x80c0
	scratch_load_b128 v[186:189] /*v[954:957]*/, off, off offset:1664 th:TH_LOAD_LU nv
	scratch_load_b128 v[190:193] /*v[958:961]*/, off, off offset:1680 th:TH_LOAD_LU nv
	scratch_load_b128 v[178:181] /*v[946:949]*/, off, off offset:1600 th:TH_LOAD_LU nv
	scratch_load_b128 v[182:185] /*v[950:953]*/, off, off offset:1616 th:TH_LOAD_LU nv
	scratch_load_b128 v[164:167] /*v[932:935]*/, off, off offset:1920 th:TH_LOAD_LU nv
	scratch_load_b128 v[168:171] /*v[936:939]*/, off, off offset:1936 th:TH_LOAD_LU nv
	scratch_load_b128 v[154:157] /*v[922:925]*/, off, off offset:1696 th:TH_LOAD_LU nv
	scratch_load_b128 v[158:161] /*v[926:929]*/, off, off offset:1712 th:TH_LOAD_LU nv
	scratch_load_b128 v[68:71] /*v[836:839]*/, off, off offset:1568 th:TH_LOAD_LU nv
	scratch_load_b128 v[72:75] /*v[840:843]*/, off, off offset:1584 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc000
	v_add_lshl_u32 v9, v9, s45, 7
	s_set_vgpr_msb 0xc0
	s_clause 0x5
	scratch_load_b128 v[146:149] /*v[914:917]*/, off, off offset:1728 th:TH_LOAD_LU nv
	scratch_load_b128 v[150:153] /*v[918:921]*/, off, off offset:1744 th:TH_LOAD_LU nv
	scratch_load_b128 v[84:87] /*v[852:855]*/, off, off offset:1408 th:TH_LOAD_LU nv
	scratch_load_b128 v[88:91] /*v[856:859]*/, off, off offset:1424 th:TH_LOAD_LU nv
	scratch_load_b128 v[12:15] /*v[780:783]*/, off, off offset:1024 th:TH_LOAD_LU nv
	scratch_load_b128 v[16:19] /*v[784:787]*/, off, off offset:1040 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xc000
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	v_mul_lo_u32 v11, v11, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v11, v11, s45, 7
	v_or_b32_e32 v14, v11, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v13, v13, s7
	v_mul_lo_u32 v1, v1, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v13, v13, s45, 7
	v_add_lshl_u32 v1, v1, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v16, v13, v35
	v_or_b32_e32 v13, v13, v0
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v13, 2, v13
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v17, s0
	v_or_b32_e32 v17, v1, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v17, 2, v17 :: v_dual_bitop2_b32 v1, v1, v0 bitop3:0x54
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_loadcnt 0x1d
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 64, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x1c
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 64, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1856 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1872 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1824 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1840 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v130 /*v898*/, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v131 /*v899*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v132 /*v900*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v133 /*v901*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v134 /*v902*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v135 /*v903*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v136 /*v904*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v137 /*v905*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v28 /*v540*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v29 /*v541*/, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v30 /*v542*/, s0
	v_or_b32_e32 v18, 0x140, v5
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v31 /*v543*/, s0
	v_or_b32_e32 v18, 0x140, v7
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v32 /*v544*/, s0
	v_or_b32_e32 v18, 0x140, v9
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v33 /*v545*/, s0
	v_or_b32_e32 v18, 0x140, v11
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v34 /*v546*/, s0
	v_or_b32_e32 v18, 0x140, v13
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v35 /*v547*/, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v15, v20 /*v532*/, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:352 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:368 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x200
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v21 /*v533*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v22 /*v534*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v23 /*v535*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v24 /*v536*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v25 /*v537*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v26 /*v538*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v27 /*v539*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v12 /*v524*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v3, v13 /*v525*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v14 /*v526*/, s0
	v_or_b32_e32 v3, 0x1c0, v5
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v15 /*v527*/, s0
	v_or_b32_e32 v3, 0x1c0, v7
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v16 /*v528*/, s0
	v_or_b32_e32 v3, 0x1c0, v9
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v17 /*v529*/, s0
	v_or_b32_e32 v3, 0x1c0, v11
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v13
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:2080 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:2096 th:TH_LOAD_LU nv
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v18 /*v530*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v19 /*v531*/, s0
	s_set_vgpr_msb 0x204
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v1, s34, v155 /*v411*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v1, s7
	v_add_lshl_u32 v3, v3, s45, 7
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v2, v10, s0
	v_cvt_pk_bf16_f32 v5, v11, s0
	v_cvt_pk_bf16_f32 v7, v12, s0
	v_cvt_pk_bf16_f32 v9, v13, s0
	s_wait_loadcnt 0x0
	v_cvt_pk_bf16_f32 v11, v14, s0
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_cvt_pk_bf16_f32 v13, v15, s0
	v_cvt_pk_bf16_f32 v15, v16, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	v_or_b32_e32 v13, v13, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v13, 2, v13
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v17, s0
	v_or_b32_e32 v17, v1, v35
	v_dual_lshlrev_b32 v17, 2, v17 :: v_dual_bitop2_b32 v1, v1, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 64, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 64, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:288 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:304 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:2016 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:2032 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1792 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1808 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1504 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1520 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0x140, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0x140, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0x140, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0x140, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0x140, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v204 /*v972*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v4, v205 /*v973*/, s0
	v_cvt_pk_bf16_f32 v15, v192 /*v960*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v4, v206 /*v974*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v4, v207 /*v975*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v4, v208 /*v976*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v4, v209 /*v977*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v4, v210 /*v978*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v4, v211 /*v979*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v4, v68 /*v580*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v3, v69 /*v581*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v70 /*v582*/, s0
	v_or_b32_e32 v3, 0x1c0, v5
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v5, v187 /*v955*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v71 /*v583*/, s0
	v_or_b32_e32 v3, 0x1c0, v7
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v7, v188 /*v956*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v72 /*v584*/, s0
	v_or_b32_e32 v3, 0x1c0, v9
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v9, v189 /*v957*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v73 /*v585*/, s0
	v_or_b32_e32 v3, 0x1c0, v11
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v11, v190 /*v958*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v74 /*v586*/, s0
	v_or_b32_e32 v3, 0x1c0, v13
	s_set_vgpr_msb 0x203
	v_cvt_pk_bf16_f32 v13, v191 /*v959*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v2, v75 /*v587*/, s0
	s_set_vgpr_msb 0x204
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v1, s33, v155 /*v411*/
	s_set_vgpr_msb 0x403
	v_cvt_pk_bf16_f32 v2, v186 /*v954*/, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, s7, v1
	v_add_lshl_u32 v3, s45, v3, 7
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_or_b32_e32 v18, 64, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_or_b32_e32 v17, v1, v35
	v_or_b32_e32 v13, v13, v0
	v_or_b32_e32 v1, v1, v0
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v17, 2, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v193 /*v961*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v178 /*v946*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v179 /*v947*/, s0
	v_or_b32_e32 v18, 64, v2
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v180 /*v948*/, s0
	v_or_b32_e32 v18, 64, v5
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v181 /*v949*/, s0
	v_or_b32_e32 v18, 64, v7
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v182 /*v950*/, s0
	v_or_b32_e32 v18, 64, v9
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v183 /*v951*/, s0
	v_or_b32_e32 v18, 64, v11
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v184 /*v952*/, s0
	v_or_b32_e32 v18, 64, v13
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v185 /*v953*/, s0
	v_or_b32_e32 v18, 64, v1
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1760 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1776 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v164 /*v932*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v165 /*v933*/, s0
	v_or_b32_e32 v18, 0xc0, v2
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v166 /*v934*/, s0
	v_or_b32_e32 v18, 0xc0, v5
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v167 /*v935*/, s0
	v_or_b32_e32 v18, 0xc0, v7
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v168 /*v936*/, s0
	v_or_b32_e32 v18, 0xc0, v9
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v169 /*v937*/, s0
	v_or_b32_e32 v18, 0xc0, v11
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v170 /*v938*/, s0
	v_or_b32_e32 v18, 0xc0, v13
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v171 /*v939*/, s0
	v_or_b32_e32 v18, 0xc0, v1
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1248 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1264 th:TH_LOAD_LU nv
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v154 /*v922*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v155 /*v923*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v156 /*v924*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v157 /*v925*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v158 /*v926*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v159 /*v927*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v160 /*v928*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v161 /*v929*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0x140, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0x140, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0x140, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0x140, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0x140, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1216 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1232 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v19, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v152 /*v920*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v20, s0
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v21, s0
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v22, s0
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v23, s0
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v24, s0
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:416 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:432 th:TH_LOAD_LU nv
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v4, v68 /*v836*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v3, v69 /*v837*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v2, v70 /*v838*/, s0
	v_or_b32_e32 v3, 0x1c0, v5
	v_cvt_pk_bf16_f32 v5, v147 /*v915*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v2, v71 /*v839*/, s0
	v_or_b32_e32 v3, 0x1c0, v7
	v_cvt_pk_bf16_f32 v7, v148 /*v916*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v2, v72 /*v840*/, s0
	v_or_b32_e32 v3, 0x1c0, v9
	v_cvt_pk_bf16_f32 v9, v149 /*v917*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v2, v73 /*v841*/, s0
	v_or_b32_e32 v3, 0x1c0, v11
	v_cvt_pk_bf16_f32 v11, v150 /*v918*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v2, v74 /*v842*/, s0
	v_or_b32_e32 v3, 0x1c0, v13
	v_cvt_pk_bf16_f32 v13, v151 /*v919*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v2, v75 /*v843*/, s0
	s_set_vgpr_msb 0x304
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v1, s31, v155 /*v411*/
	s_set_vgpr_msb 0x403
	v_cvt_pk_bf16_f32 v2, v146 /*v914*/, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, s7, v1
	v_add_lshl_u32 v3, s45, v3, 7
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_or_b32_e32 v17, v1, v35
	v_or_b32_e32 v13, v13, v0
	v_or_b32_e32 v1, v1, v0
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v17, 2, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v153 /*v921*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 64, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 64, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1472 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1488 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1440 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1456 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1344 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1360 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1312 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1328 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0x140, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0x140, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0x140, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0x140, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0x140, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1536 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1552 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v19, s0
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v20, s0
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v21, s0
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v22, s0
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v23, s0
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v24, s0
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v25, s0
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_clause 0x1
	scratch_load_b128 v[14:17], off, off offset:1376 th:TH_LOAD_LU nv
	scratch_load_b128 v[18:21], off, off offset:1392 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v4, v14, s0
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v15, s0
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v90 /*v858*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v16, s0
	v_or_b32_e32 v3, 0x1c0, v5
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v5, v85 /*v853*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v17, s0
	v_or_b32_e32 v3, 0x1c0, v7
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v7, v86 /*v854*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v18, s0
	v_or_b32_e32 v3, 0x1c0, v9
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v9, v87 /*v855*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v19, s0
	v_or_b32_e32 v3, 0x1c0, v11
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v11, v88 /*v856*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v20, s0
	v_or_b32_e32 v3, 0x1c0, v13
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v13, v89 /*v857*/, s0
	s_set_vgpr_msb 0x304
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v21, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1280 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1296 th:TH_LOAD_LU nv
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v1, s30, v155 /*v411*/
	s_set_vgpr_msb 0x403
	v_cvt_pk_bf16_f32 v2, v84 /*v852*/, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, s7, v1
	v_add_lshl_u32 v3, s45, v3, 7
	s_set_vgpr_msb 0x300
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_or_b32_e32 v17, v1, v35
	v_or_b32_e32 v13, v13, v0
	v_or_b32_e32 v1, v1, v0
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v17, 2, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v13, 2, v13
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v91 /*v859*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 64, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 64, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1184 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1200 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1152 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1168 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:1120 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:1136 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v12 /*v780*/, s0
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v13 /*v781*/, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v14 /*v782*/, s0
	v_or_b32_e32 v18, 0x140, v5
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v15 /*v783*/, s0
	v_or_b32_e32 v18, 0x140, v7
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v16 /*v784*/, s0
	v_or_b32_e32 v18, 0x140, v9
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v17 /*v785*/, s0
	v_or_b32_e32 v18, 0x140, v11
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v18 /*v786*/, s0
	v_or_b32_e32 v18, 0x140, v13
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v15, v19 /*v787*/, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	s_set_vgpr_msb 0x300
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:576 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:592 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v19, s0
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v20, s0
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v21, s0
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v22, s0
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v23, s0
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v24, s0
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v25, s0
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_clause 0x1
	scratch_load_b128 v[14:17], off, off offset:1088 th:TH_LOAD_LU nv
	scratch_load_b128 v[18:21], off, off offset:1104 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v4, v14, s0
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v15, s0
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v16, s0
	v_or_b32_e32 v3, 0x1c0, v5
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v17, s0
	v_or_b32_e32 v3, 0x1c0, v7
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v18, s0
	v_or_b32_e32 v3, 0x1c0, v9
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v19, s0
	v_or_b32_e32 v3, 0x1c0, v11
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v13
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:1056 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:1072 th:TH_LOAD_LU nv
	v_cvt_pk_bf16_f32 v2, v20, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v21, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:960 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:976 th:TH_LOAD_LU nv
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v1, s29, v155 /*v411*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v1, s7
	v_add_lshl_u32 v3, v3, s45, 7
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	s_wait_loadcnt 0x3
	v_cvt_pk_bf16_f32 v2, v10, s0
	v_cvt_pk_bf16_f32 v5, v11, s0
	v_cvt_pk_bf16_f32 v7, v12, s0
	v_cvt_pk_bf16_f32 v9, v13, s0
	s_wait_loadcnt 0x2
	v_cvt_pk_bf16_f32 v11, v14, s0
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_cvt_pk_bf16_f32 v13, v15, s0
	v_cvt_pk_bf16_f32 v15, v16, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	v_or_b32_e32 v13, v13, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v13, 2, v13
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v17, s0
	v_or_b32_e32 v17, v1, v35
	v_dual_lshlrev_b32 v17, 2, v17 :: v_dual_bitop2_b32 v1, v1, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 64, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 64, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:928 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:944 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:896 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:912 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0xc0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0xc0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0xc0, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0xc0, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0xc0, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0xc0, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0xc0, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0xc0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:864 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:880 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	buffer_store_b16 v15, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	buffer_store_b16 v15, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	buffer_store_b16 v15, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	buffer_store_b16 v15, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	buffer_store_b16 v15, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	buffer_store_b16 v15, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:832 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:848 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v17, s[0:3], null offen offset:256
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 0x140, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 0x140, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 0x140, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 0x140, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 0x140, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	v_or_b32_e32 v18, 0x140, v1
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:800 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:816 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v15, v18, s0
	buffer_store_b16 v15, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v19, s0
	buffer_store_b16 v4, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v20, s0
	buffer_store_b16 v4, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v21, s0
	buffer_store_b16 v4, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v22, s0
	buffer_store_b16 v4, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v23, s0
	buffer_store_b16 v4, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v24, s0
	buffer_store_b16 v4, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v25, s0
	buffer_store_b16 v4, v17, s[0:3], null offen offset:384
	s_clause 0x1
	scratch_load_b128 v[14:17], off, off offset:768 th:TH_LOAD_LU nv
	scratch_load_b128 v[18:21], off, off offset:784 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v4, v14, s0
	buffer_store_b16 v4, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v15, s0
	buffer_store_b16 v3, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v16, s0
	v_or_b32_e32 v3, 0x1c0, v5
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v17, s0
	v_or_b32_e32 v3, 0x1c0, v7
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v18, s0
	v_or_b32_e32 v3, 0x1c0, v9
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v19, s0
	v_or_b32_e32 v3, 0x1c0, v11
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v13
	s_clause 0x1
	scratch_load_b128 v[10:13], off, off offset:736 th:TH_LOAD_LU nv
	scratch_load_b128 v[14:17], off, off offset:752 th:TH_LOAD_LU nv
	v_cvt_pk_bf16_f32 v2, v20, s0
	buffer_store_b16 v2, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v21, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:704 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:720 th:TH_LOAD_LU nv
	buffer_store_b16 v2, v1, s[0:3], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v1, s28, v155 /*v411*/
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v3, v1, s7
	v_add_lshl_u32 v3, v3, s45, 7
	s_set_vgpr_msb 0x400
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v3, v35
	v_or_b32_e32 v3, v3, v0
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v3, 2, v3
	s_wait_loadcnt 0x3
	v_cvt_pk_bf16_f32 v2, v10, s0
	v_cvt_pk_bf16_f32 v5, v11, s0
	v_cvt_pk_bf16_f32 v7, v12, s0
	v_cvt_pk_bf16_f32 v9, v13, s0
	s_wait_loadcnt 0x2
	v_cvt_pk_bf16_f32 v11, v14, s0
	buffer_store_b16 v2, v4, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 1, v1
	v_cvt_pk_bf16_f32 v13, v15, s0
	v_cvt_pk_bf16_f32 v15, v16, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v2, v2, s7
	v_add_lshl_u32 v2, v2, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v6, v2, v35
	v_dual_lshlrev_b32 v6, 2, v6 :: v_dual_bitop2_b32 v2, v2, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_store_b16 v5, v6, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, 2, v1
	v_mul_lo_u32 v5, v5, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v5, v5, s45, 7
	v_or_b32_e32 v8, v5, v35
	v_or_b32_e32 v5, v5, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v8, 2, v8 :: v_dual_lshlrev_b32 v5, 2, v5
	buffer_store_b16 v7, v8, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v7, 3, v1
	v_mul_lo_u32 v7, v7, s7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v7, v7, s45, 7
	v_or_b32_e32 v10, v7, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v10, 2, v10 :: v_dual_bitop2_b32 v7, v7, v0 bitop3:0x54
	v_lshlrev_b32_e32 v7, 2, v7
	buffer_store_b16 v9, v10, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v9, 4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, v9, s7
	v_add_lshl_u32 v9, v9, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, v9, v35
	v_or_b32_e32 v9, v9, v0
	v_dual_lshlrev_b32 v12, 2, v12 :: v_dual_lshlrev_b32 v9, 2, v9
	buffer_store_b16 v11, v12, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, v11, s7
	v_add_lshl_u32 v11, v11, s45, 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, v11, v35
	v_dual_lshlrev_b32 v14, 2, v14 :: v_dual_bitop2_b32 v11, v11, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v11
	buffer_store_b16 v13, v14, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v13, 6, v1
	v_or_b32_e32 v1, 7, v1
	v_mul_lo_u32 v13, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_lo_u32 v1, v1, s7
	v_add_lshl_u32 v13, v13, s45, 7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v1, v1, s45, 7
	v_or_b32_e32 v16, v13, v35
	v_or_b32_e32 v13, v13, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v0, v1, v0
	v_dual_lshlrev_b32 v16, 2, v16 :: v_dual_lshlrev_b32 v13, 2, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v0, 2, v0
	buffer_store_b16 v15, v16, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v17, s0
	v_or_b32_e32 v17, v1, v35
	v_dual_lshlrev_b32 v17, 2, v17 :: v_dual_bitop2_b32 v1, 64, v0 bitop3:0x54
	buffer_store_b16 v15, v17, s[0:3], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v18, s0
	v_or_b32_e32 v18, 64, v3
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v19, s0
	v_or_b32_e32 v18, 64, v2
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v20, s0
	v_or_b32_e32 v18, 64, v5
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v21, s0
	v_or_b32_e32 v18, 64, v7
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v22, s0
	v_or_b32_e32 v18, 64, v9
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v23, s0
	v_or_b32_e32 v18, 64, v11
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v24, s0
	v_or_b32_e32 v18, 64, v13
	buffer_store_b16 v15, v18, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v15, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:672 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:688 th:TH_LOAD_LU nv
	buffer_store_b16 v15, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v15, 0xc0, v3
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v1, v18, s0
	buffer_store_b16 v1, v4, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v19, s0
	buffer_store_b16 v1, v6, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v20, s0
	buffer_store_b16 v1, v8, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v21, s0
	buffer_store_b16 v1, v10, s[0:3], null offen offset:128
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v22, s0
	buffer_store_b16 v1, v12, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v23, s0
	buffer_store_b16 v1, v14, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v24, s0
	buffer_store_b16 v1, v16, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:640 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:656 th:TH_LOAD_LU nv
	buffer_store_b16 v1, v17, s[0:3], null offen offset:128
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v18, s0
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v19, s0
	v_or_b32_e32 v15, 0xc0, v2
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v20, s0
	v_or_b32_e32 v15, 0xc0, v5
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v21, s0
	v_or_b32_e32 v15, 0xc0, v7
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v22, s0
	v_or_b32_e32 v15, 0xc0, v9
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v23, s0
	v_or_b32_e32 v15, 0xc0, v11
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v24, s0
	v_or_b32_e32 v15, 0xc0, v13
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:608 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:624 th:TH_LOAD_LU nv
	v_or_b32_e32 v15, 0xc0, v0
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v15, 0x140, v3
	v_or_b32_e32 v3, 0x1c0, v3
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v1, v18, s0
	buffer_store_b16 v1, v4, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v19, s0
	buffer_store_b16 v1, v6, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v20, s0
	buffer_store_b16 v1, v8, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v21, s0
	buffer_store_b16 v1, v10, s[0:3], null offen offset:256
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v22, s0
	buffer_store_b16 v1, v12, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v23, s0
	buffer_store_b16 v1, v14, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v24, s0
	buffer_store_b16 v1, v16, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:544 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:560 th:TH_LOAD_LU nv
	buffer_store_b16 v1, v17, s[0:3], null offen offset:256
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v18, s0
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v19, s0
	v_or_b32_e32 v15, 0x140, v2
	v_or_b32_e32 v2, 0x1c0, v2
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v20, s0
	v_or_b32_e32 v15, 0x140, v5
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v21, s0
	v_or_b32_e32 v15, 0x140, v7
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v22, s0
	v_or_b32_e32 v15, 0x140, v9
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v23, s0
	v_or_b32_e32 v15, 0x140, v11
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v24, s0
	v_or_b32_e32 v15, 0x140, v13
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v25, s0
	s_clause 0x1
	scratch_load_b128 v[18:21], off, off offset:512 th:TH_LOAD_LU nv
	scratch_load_b128 v[22:25], off, off offset:528 th:TH_LOAD_LU nv
	v_or_b32_e32 v15, 0x140, v0
	v_or_b32_e32 v0, 0x1c0, v0
	buffer_store_b16 v1, v15, s[0:3], null offen
	s_wait_loadcnt 0x1
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v18, s0
	buffer_store_b16 v1, v4, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v19, s0
	buffer_store_b16 v1, v6, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v20, s0
	buffer_store_b16 v1, v8, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v21, s0
	buffer_store_b16 v1, v10, s[0:3], null offen offset:384
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v22, s0
	buffer_store_b16 v1, v12, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v23, s0
	buffer_store_b16 v1, v14, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v24, s0
	buffer_store_b16 v1, v16, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v25, s0
	buffer_store_b16 v1, v17, s[0:3], null offen offset:384
	s_clause 0x1
	scratch_load_b128 v[14:17], off, off offset:1632 th:TH_LOAD_LU nv
	scratch_load_b128 v[18:21], off, off offset:1648 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x1
	v_cvt_pk_bf16_f32 v1, v14, s0
	buffer_store_b16 v1, v3, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v15, s0
	buffer_store_b16 v1, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v16, s0
	v_or_b32_e32 v2, 0x1c0, v5
	buffer_store_b16 v1, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v17, s0
	v_or_b32_e32 v2, 0x1c0, v7
	buffer_store_b16 v1, v2, s[0:3], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v18, s0
	v_or_b32_e32 v2, 0x1c0, v9
	buffer_store_b16 v1, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v19, s0
	v_or_b32_e32 v2, 0x1c0, v11
	buffer_store_b16 v1, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v20, s0
	v_or_b32_e32 v2, 0x1c0, v13
	buffer_store_b16 v1, v2, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v21, s0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_endpgm
.Lfunc_end0:
	.size	k_dq_0, .Lfunc_end0-k_dq_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dq_0
		.amdhsa_group_segment_fixed_size 8704
		.amdhsa_private_segment_fixed_size 9052
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
		.amdhsa_next_free_vgpr 1024
		.amdhsa_next_free_sgpr 51
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

	.set .Lk_dq_0.num_vgpr, 1024
	.set .Lk_dq_0.num_agpr, 0
	.set .Lk_dq_0.numbered_sgpr, 51
	.set .Lk_dq_0.num_named_barrier, 0
	.set .Lk_dq_0.private_seg_size, 9052
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
    .private_segment_fixed_size: 9052
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     53
    .sgpr_spill_count: 0
    .symbol:         k_dq_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1024
    .vgpr_spill_count: 4903
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
