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
	s_load_b256 s[36:43], s[0:1], 0x140 nv
	s_and_b32 s3, ttmp7, 0xffff
	s_add_co_i32 s2, s2, 1
	s_bfe_u32 s5, ttmp6, 0x4000c
	s_mul_i32 s2, s3, s2
	s_bfe_u32 s4, ttmp6, 0x40004
	s_add_co_i32 s5, s5, 1
	s_add_co_i32 s4, s4, s2
	s_and_b32 s2, ttmp6, 15
	s_mul_i32 s5, ttmp9, s5
	s_getreg_b32 s6, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s2, s2, s5
	s_cmp_eq_u32 s6, 0
	s_set_vgpr_msb 0x80
	v_dual_lshrrev_b32 v6 /*v518*/, 4, v0 :: v_dual_bitop2_b32 v2 /*v514*/, 15, v0 bitop3:0x40
	s_cselect_b32 s5, ttmp9, s2
	s_cselect_b32 s2, s3, s4
	s_bfe_u32 s3, ttmp6, 0x40014
	s_lshr_b32 s4, ttmp7, 16
	s_add_co_i32 s3, s3, 1
	s_bfe_u32 s7, ttmp6, 0x40008
	s_mul_i32 s3, s4, s3
	s_mov_b32 s18, 0x200000
	s_add_co_i32 s7, s7, s3
	s_cmp_eq_u32 s6, 0
	s_load_b64 s[16:17], s[0:1], 0xc0 nv
	s_cselect_b32 s22, s4, s7
	s_wait_kmcnt 0x0
	s_add_co_i32 s3, s37, 31
	s_mul_i32 s54, s37, s22
	s_ashr_i32 s4, s3, 31
	s_mul_i32 s15, s39, s22
	s_lshr_b32 s4, s4, 27
	s_set_vgpr_msb 0x8088
	v_lshlrev_b32_e32 v3 /*v515*/, 3, v6 /*v518*/
	s_add_co_i32 s4, s3, s4
	s_mov_b64 s[50:51], 0x800000
	s_and_b32 s6, s4, 0xffffffe0
	s_ashr_i32 s4, s4, 5
	s_cmp_lg_u32 s3, s6
	s_mov_b64 s[46:47], 0x800000
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s3, 0
	s_cselect_b32 s3, -1, 0
	s_not_b32 s2, s2
	s_and_b32 s3, s3, s6
	s_add_co_i32 s4, s4, s2
	s_cmp_lg_u32 s3, 0
	s_sub_co_ci_u32 s2, s4, 0
	s_load_b32 s4, s[0:1], 0x160 nv
	s_lshl_b32 s52, s2, 5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s20, s52, s43
	s_set_vgpr_msb 0x8808
	v_or_b32_e32 v2, s52, v2 /*v514*/
	s_add_co_i32 s2, s20, 63
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s3, s2, s3
	s_and_b32 s6, s3, 0xffffffe0
	s_ashr_i32 s3, s3, 5
	s_cmp_lg_u32 s2, s6
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, s6
	s_sub_co_ci_u32 s2, s3, 0
	s_max_i32 s2, s2, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s42
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s3, -1, 0
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s21, s2, s42
	s_add_co_i32 s2, s20, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s3, s2, s3
	s_ashr_i32 s3, s3, 5
	s_cmp_gt_i32 s2, -1
	s_cselect_b32 s2, s3, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s21
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s53, -1, 0
	s_and_b32 s3, s53, exec_lo
	s_cselect_b32 s2, s2, s42
	s_ashr_i32 s4, s5, 31
	s_ashr_i32 s6, s39, 31
	s_lshr_b32 s4, s4, 29
	s_lshr_b32 s6, s6, 29
	s_add_co_i32 s4, s5, s4
	s_add_co_i32 s6, s39, s6
	s_ashr_i32 s7, s4, 3
	s_and_b32 s4, s4, -8
	s_ashr_i32 s8, s6, 3
	s_and_b32 s6, s6, -8
	s_and_b32 s3, s39, 7
	s_sub_co_i32 s9, s5, s4
	s_cmp_lg_u32 s39, s6
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s39, 0
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s6, s10, s6
	s_sub_co_ci_u32 s6, s8, 0
	s_cmp_lg_u32 s5, s4
	s_mul_i32 s6, s6, s9
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s5, 0
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s4, s8, s4
	s_sub_co_ci_u32 s4, s7, 0
	s_add_co_i32 s4, s4, s6
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s42, s4, s5
	s_abs_i32 s4, s41
	s_abs_i32 s6, s42
	s_cvt_f32_u32 s3, s4
	s_sub_co_i32 s5, 0, s4
	s_xor_b32 s12, s42, s41
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s3, s3
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s3, s3
	s_mul_i32 s5, s5, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s5, s3, s5
	s_add_co_i32 s3, s3, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_hi_u32 s5, s6, s3
	s_ashr_i32 s3, s12, 31
	s_mul_i32 s7, s5, s4
	s_add_co_i32 s8, s5, 1
	s_sub_co_i32 s7, s6, s7
	s_mov_b32 s6, 0x800000
	s_sub_co_i32 s9, s7, s4
	s_cmp_ge_u32 s7, s4
	s_mov_b32 s10, s6
	s_cselect_b32 s5, s8, s5
	s_cselect_b32 s7, s9, s7
	s_add_co_i32 s8, s5, 1
	s_cmp_ge_u32 s7, s4
	s_mov_b32 s7, 0
	s_cselect_b32 s4, s8, s5
	s_mov_b32 s19, s7
	s_xor_b32 s13, s4, s3
	s_clause 0x3
	s_load_b64 s[4:5], s[0:1], 0x0 nv
	s_load_b64 s[8:9], s[0:1], 0x90 nv
	s_load_b64 s[44:45], s[0:1], 0x30 nv
	s_load_b64 s[48:49], s[0:1], 0x60 nv
	s_sub_co_i32 s11, s13, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s14, s11, s41
	s_mov_b32 s11, s7
	s_cmp_lg_u32 s42, s14
	s_cselect_b32 s14, -1, 0
	s_cmp_lt_i32 s12, 0
	s_cselect_b32 s12, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s12, s12, s14
	s_sub_co_ci_u32 s23, s13, s3
	s_lshl_b32 s3, s39, 4
	s_or_b32 s41, s52, 16
	s_mul_i32 s12, s3, s54
	v_or_b32_e32 v1, s41, v2 /*v514*/
	s_lshl4_add_u32 s14, s42, s12
	s_load_b64 s[12:13], s[0:1], 0xe8 nv
	v_mad_u32 v3, v2, s3, s14
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_u32 v4, v1, s3, s14
	s_add_co_i32 s3, s42, s15
	s_mov_b32 s14, s18
	s_mul_i32 s3, s3, s37
	s_mov_b32 s15, s7
	v_add_lshl_u32 v2, v2, s3, 2
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v3, v3, v6 /*v518*/
	v_add_lshl_u32 v5, v1, s3, 2
	v_or_b32_e32 v4, v4, v6 /*v518*/
	s_set_vgpr_msb 0x840
	s_clause 0x1
	buffer_load_b32 v250 /*v506*/, v2, s[16:19], null offen
	buffer_load_b32 v252 /*v508*/, v5, s[16:19], null offen
	s_set_vgpr_msb 0x4000
	v_dual_lshlrev_b32 v3, 4, v3 :: v_dual_lshlrev_b32 v4, 4, v4
	s_wait_kmcnt 0x0
	s_clause 0x7
	buffer_load_b128 v[66:69], v3, s[4:7], null offen
	buffer_load_b128 v[70:73], v3, s[4:7], null offen offset:32
	buffer_load_b128 v[74:77], v3, s[4:7], null offen offset:64
	buffer_load_b128 v[78:81], v3, s[4:7], null offen offset:96
	buffer_load_b128 v[82:85], v3, s[4:7], null offen offset:128
	buffer_load_b128 v[86:89], v3, s[4:7], null offen offset:160
	buffer_load_b128 v[98:101], v3, s[4:7], null offen offset:192
	buffer_load_b128 v[102:105], v3, s[4:7], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[106:109], v3, s[8:11], null offen
	buffer_load_b128 v[110:113], v3, s[8:11], null offen offset:32
	buffer_load_b128 v[114:117], v3, s[8:11], null offen offset:64
	buffer_load_b128 v[118:121], v3, s[8:11], null offen offset:96
	buffer_load_b128 v[122:125], v3, s[8:11], null offen offset:128
	buffer_load_b128 v[126:129], v3, s[8:11], null offen offset:160
	buffer_load_b128 v[130:133], v3, s[8:11], null offen offset:192
	buffer_load_b128 v[134:137], v3, s[8:11], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[138:141], v4, s[4:7], null offen
	buffer_load_b128 v[142:145], v4, s[4:7], null offen offset:32
	buffer_load_b128 v[146:149], v4, s[4:7], null offen offset:64
	buffer_load_b128 v[150:153], v4, s[4:7], null offen offset:96
	buffer_load_b128 v[154:157], v4, s[4:7], null offen offset:128
	buffer_load_b128 v[158:161], v4, s[4:7], null offen offset:160
	buffer_load_b128 v[162:165], v4, s[4:7], null offen offset:192
	buffer_load_b128 v[166:169], v4, s[4:7], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[170:173], v4, s[8:11], null offen
	buffer_load_b128 v[174:177], v4, s[8:11], null offen offset:32
	buffer_load_b128 v[178:181], v4, s[8:11], null offen offset:64
	buffer_load_b128 v[182:185], v4, s[8:11], null offen offset:96
	buffer_load_b128 v[186:189], v4, s[8:11], null offen offset:128
	buffer_load_b128 v[190:193], v4, s[8:11], null offen offset:160
	buffer_load_b128 v[194:197], v4, s[8:11], null offen offset:192
	buffer_load_b128 v[198:201], v4, s[8:11], null offen offset:224
	s_set_vgpr_msb 64
	s_clause 0x2
	buffer_load_b32 v254 /*v510*/, v2, s[12:15], null offen
	s_set_vgpr_msb 0x4080
	buffer_load_b32 v0 /*v512*/, v5, s[12:15], null offen
	s_set_vgpr_msb 0x8000
	v_dual_lshrrev_b32 v5, 3, v0 :: v_dual_bitop2_b32 v2, 16, v0 bitop3:0x40
	s_set_vgpr_msb 0x80
	v_or_b32_e32 v10 /*v522*/, 16, v0
	s_set_vgpr_msb 0x8020
	v_and_or_b32 v3, v0, 7, v3 /*v515*/
	v_bfe_u32 v4, v0, 3, 1
	s_set_vgpr_msb 0x2088
	v_mad_u32_u24 v4 /*v516*/, 0x110, v2 /*v514*/, v2
	s_set_vgpr_msb 0x8880
	v_lshlrev_b32_e32 v9 /*v521*/, 4, v5
	s_set_vgpr_msb 0x8088
	v_mad_u32_u24 v5 /*v517*/, 0x110, v10 /*v522*/, v2
	s_set_vgpr_msb 0x8880
	v_mul_u32_u24_e32 v7 /*v519*/, 0x110, v3
	v_lshlrev_b32_e32 v8 /*v520*/, 4, v4
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s5, s23, 4
	s_mov_b32 s9, 1
	s_cmp_lt_i32 s2, 1
	s_mul_i32 s8, s38, s22
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_3
	s_lshl_b32 s10, s40, 4
	s_set_vgpr_msb 0x41
	s_wait_loadcnt 0x1
	v_dual_mov_b32 v255 /*v511*/, v254 /*v510*/ :: v_dual_mov_b32 v251 /*v507*/, v250 /*v506*/
	s_mul_i32 s4, s8, s10
	v_mov_b32_e32 v253 /*v509*/, v252 /*v508*/
	s_add_co_i32 s4, s5, s4
	s_set_vgpr_msb 0x418a
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v1 /*v513*/, v0 /*v512*/ :: v_dual_bitop2_b32 v11 /*v523*/, s4, v6 /*v518*/ bitop3:0x54
	s_set_vgpr_msb 0x8a08
	v_mad_u32 v2, s10, v10 /*v522*/, s4
	v_mad_u32 v3, s10, v2 /*v514*/, s4
	s_add_co_i32 s11, s2, 0x7ffffff
	s_mov_b32 s37, s36
	s_mov_b32 s4, 0x3fb8aa3b
	s_mov_b64 s[6:7], s[2:3]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v242, 0 :: v_dual_bitop2_b32 v2, v2, v6 /*v518*/ bitop3:0x54
	v_dual_mov_b32 v243, v242 :: v_dual_bitop2_b32 v3, v3, v6 /*v518*/ bitop3:0x54
	v_dual_mov_b32 v249, v242 :: v_dual_mov_b32 v250, v242
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_lshlrev_b32 v2, 4, v2 :: v_dual_lshlrev_b32 v3, 4, v3
	v_dual_mov_b32 v244, v242 :: v_dual_mov_b32 v251, v242
	v_mov_b32_e32 v252, v242
	v_or_b32_e32 v5, 0xc0, v2
	v_or_b32_e32 v4, 0xe0, v2
	v_or_b32_e32 v6, 0xa0, v2
	v_or_b32_e32 v7, 0x80, v2
	v_or_b32_e32 v8, 0x60, v2
	v_or_b32_e32 v9, 64, v2
	v_or_b32_e32 v10, 32, v2
	s_set_vgpr_msb 64
	s_clause 0x1
	buffer_load_b128 v[2:5] /*v[258:261]*/, v5, s[48:51], null offen
	buffer_load_b128 v[14:17] /*v[270:273]*/, v6, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[42:45] /*v[298:301]*/, v5, s[44:47], null offen
	buffer_load_b128 v[54:57] /*v[310:313]*/, v6, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[10:13] /*v[266:269]*/, v7, s[48:51], null offen
	buffer_load_b128 v[38:41] /*v[294:297]*/, v8, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[50:53] /*v[306:309]*/, v7, s[44:47], null offen
	buffer_load_b128 v[70:73] /*v[326:329]*/, v8, s[44:47], null offen
	s_clause 0x3
	buffer_load_b128 v[34:37] /*v[290:293]*/, v9, s[48:51], null offen
	buffer_load_b128 v[30:33] /*v[286:289]*/, v10, s[48:51], null offen
	buffer_load_b128 v[6:9] /*v[262:265]*/, v4, s[48:51], null offen
	buffer_load_b128 v[26:29] /*v[282:285]*/, v2, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[66:69] /*v[322:325]*/, v9, s[44:47], null offen
	buffer_load_b128 v[78:81] /*v[334:337]*/, v10, s[44:47], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v5, 0xc0, v3
	v_or_b32_e32 v6, 0xa0, v3
	s_set_vgpr_msb 64
	s_clause 0x1
	buffer_load_b128 v[46:49] /*v[302:305]*/, v4, s[44:47], null offen
	buffer_load_b128 v[74:77] /*v[330:333]*/, v2, s[44:47], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v4, 0x80, v3
	v_or_b32_e32 v7, 0x60, v3
	s_set_vgpr_msb 64
	s_clause 0x1
	buffer_load_b128 v[18:21] /*v[274:277]*/, v5, s[48:51], null offen
	buffer_load_b128 v[62:65] /*v[318:321]*/, v6, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[82:85] /*v[338:341]*/, v5, s[44:47], null offen
	buffer_load_b128 v[110:113] /*v[366:369]*/, v6, s[44:47], null offen
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v245, v242 :: v_dual_bitop2_b32 v5, 64, v3 bitop3:0x54
	v_or_b32_e32 v2, 0xe0, v3
	v_dual_mov_b32 v246, v242 :: v_dual_bitop2_b32 v6, 32, v3 bitop3:0x54
	s_set_vgpr_msb 64
	s_clause 0x1
	buffer_load_b128 v[58:61] /*v[314:317]*/, v4, s[48:51], null offen
	buffer_load_b128 v[102:105] /*v[358:361]*/, v7, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[106:109] /*v[362:365]*/, v4, s[44:47], null offen
	buffer_load_b128 v[118:121] /*v[374:377]*/, v7, s[44:47], null offen
	s_clause 0x3
	buffer_load_b128 v[98:101] /*v[354:357]*/, v5, s[48:51], null offen
	buffer_load_b128 v[94:97] /*v[350:353]*/, v6, s[48:51], null offen
	buffer_load_b128 v[22:25] /*v[278:281]*/, v2, s[48:51], null offen
	buffer_load_b128 v[90:93] /*v[346:349]*/, v3, s[48:51], null offen
	s_clause 0x3
	buffer_load_b128 v[114:117] /*v[370:373]*/, v5, s[44:47], null offen
	buffer_load_b128 v[126:129] /*v[382:385]*/, v6, s[44:47], null offen
	buffer_load_b128 v[86:89] /*v[342:345]*/, v2, s[44:47], null offen
	buffer_load_b128 v[122:125] /*v[378:381]*/, v3, s[44:47], null offen
	s_set_vgpr_msb 0x4000
	v_lshlrev_b32_e32 v4, 1, v0
	s_set_vgpr_msb 8
	v_dual_mov_b32 v247, v242 :: v_dual_bitop2_b32 v2, 32, v9 /*v521*/ bitop3:0x54
	v_dual_mov_b32 v248, v242 :: v_dual_bitop2_b32 v3, 64, v8 /*v520*/ bitop3:0x54
	v_or_b32_e32 v5, 0x60, v9 /*v521*/
	v_or_b32_e32 v6, 0x80, v8 /*v520*/
	v_or_b32_e32 v7, 0xa0, v9 /*v521*/
	v_or_b32_e32 v8, 0xc0, v8 /*v520*/
	v_and_or_b32 v4, v4, 16, 0xe0
	s_set_vgpr_msb 0x882
	v_dual_add_nc_u32 v12 /*v524*/, v7 /*v519*/, v2 :: v_dual_add_nc_u32 v13 /*v525*/, v7 /*v519*/, v3
	v_dual_add_nc_u32 v14 /*v526*/, v7 /*v519*/, v5 :: v_dual_add_nc_u32 v15 /*v527*/, v7 /*v519*/, v6
	v_dual_add_nc_u32 v16 /*v528*/, v7 /*v519*/, v7 :: v_dual_add_nc_u32 v17 /*v529*/, v7 /*v519*/, v8
	v_add_nc_u32_e32 v18 /*v530*/, v7 /*v519*/, v4
	s_set_vgpr_msb 0x8200
	v_dual_mov_b32 v253, v242 :: v_dual_mov_b32 v254, v242
	v_dual_mov_b32 v255, v242 :: v_dual_mov_b32 v234, v242
	s_set_vgpr_msb 64
	v_dual_mov_b32 v0 /*v256*/, v242 :: v_dual_mov_b32 v1 /*v257*/, v242
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v235, v242 :: v_dual_mov_b32 v236, v242
	v_dual_mov_b32 v237, v242 :: v_dual_mov_b32 v238, v242
	v_dual_mov_b32 v239, v242 :: v_dual_mov_b32 v240, v242
	v_dual_mov_b32 v241, v242 :: v_dual_mov_b32 v226, v242
	v_dual_mov_b32 v227, v242 :: v_dual_mov_b32 v228, v242
	v_dual_mov_b32 v229, v242 :: v_dual_mov_b32 v230, v242
	v_dual_mov_b32 v231, v242 :: v_dual_mov_b32 v232, v242
	v_dual_mov_b32 v233, v242 :: v_dual_mov_b32 v218, v242
	v_dual_mov_b32 v219, v242 :: v_dual_mov_b32 v220, v242
	v_dual_mov_b32 v221, v242 :: v_dual_mov_b32 v222, v242
	v_dual_mov_b32 v223, v242 :: v_dual_mov_b32 v224, v242
	v_dual_mov_b32 v225, v242 :: v_dual_mov_b32 v210, v242
	v_dual_mov_b32 v211, v242 :: v_dual_mov_b32 v212, v242
	v_dual_mov_b32 v213, v242 :: v_dual_mov_b32 v214, v242
	v_dual_mov_b32 v215, v242 :: v_dual_mov_b32 v216, v242
	v_dual_mov_b32 v217, v242 :: v_dual_mov_b32 v202, v242
	v_dual_mov_b32 v203, v242 :: v_dual_mov_b32 v204, v242
	v_dual_mov_b32 v205, v242 :: v_dual_mov_b32 v206, v242
	v_dual_mov_b32 v207, v242 :: v_dual_mov_b32 v208, v242
	v_dual_mov_b32 v209, v242 :: v_dual_mov_b32 v90, v242
	v_dual_mov_b32 v91, v242 :: v_dual_mov_b32 v92, v242
	v_dual_mov_b32 v93, v242 :: v_dual_mov_b32 v94, v242
	v_dual_mov_b32 v95, v242 :: v_dual_mov_b32 v96, v242
	v_dual_mov_b32 v97, v242 :: v_dual_mov_b32 v58, v242
	v_dual_mov_b32 v59, v242 :: v_dual_mov_b32 v60, v242
	v_dual_mov_b32 v61, v242 :: v_dual_mov_b32 v62, v242
	v_dual_mov_b32 v63, v242 :: v_dual_mov_b32 v64, v242
	v_dual_mov_b32 v65, v242 :: v_dual_mov_b32 v50, v242
	v_dual_mov_b32 v51, v242 :: v_dual_mov_b32 v52, v242
	v_dual_mov_b32 v53, v242 :: v_dual_mov_b32 v54, v242
	v_dual_mov_b32 v55, v242 :: v_dual_mov_b32 v56, v242
	v_dual_mov_b32 v57, v242 :: v_dual_mov_b32 v42, v242
	v_dual_mov_b32 v43, v242 :: v_dual_mov_b32 v44, v242
	v_dual_mov_b32 v45, v242 :: v_dual_mov_b32 v46, v242
	v_dual_mov_b32 v47, v242 :: v_dual_mov_b32 v48, v242
	v_dual_mov_b32 v49, v242 :: v_dual_mov_b32 v34, v242
	v_dual_mov_b32 v35, v242 :: v_dual_mov_b32 v36, v242
	v_dual_mov_b32 v37, v242 :: v_dual_mov_b32 v38, v242
	v_dual_mov_b32 v39, v242 :: v_dual_mov_b32 v40, v242
	v_dual_mov_b32 v41, v242 :: v_dual_mov_b32 v26, v242
	v_dual_mov_b32 v27, v242 :: v_dual_mov_b32 v28, v242
	v_dual_mov_b32 v29, v242 :: v_dual_mov_b32 v30, v242
	v_dual_mov_b32 v31, v242 :: v_dual_mov_b32 v32, v242
	v_dual_mov_b32 v33, v242 :: v_dual_mov_b32 v18, v242
	v_dual_mov_b32 v19, v242 :: v_dual_mov_b32 v20, v242
	v_dual_mov_b32 v21, v242 :: v_dual_mov_b32 v22, v242
	v_dual_mov_b32 v23, v242 :: v_dual_mov_b32 v24, v242
	v_dual_mov_b32 v25, v242 :: v_dual_mov_b32 v10, v242
	v_dual_mov_b32 v11, v242 :: v_dual_mov_b32 v12, v242
	v_dual_mov_b32 v13, v242 :: v_dual_mov_b32 v14, v242
	v_dual_mov_b32 v15, v242 :: v_dual_mov_b32 v16, v242
	v_dual_mov_b32 v17, v242 :: v_dual_mov_b32 v2, v242
	v_dual_mov_b32 v3, v242 :: v_dual_mov_b32 v4, v242
	v_dual_mov_b32 v5, v242 :: v_dual_mov_b32 v6, v242
	v_dual_mov_b32 v7, v242 :: v_dual_mov_b32 v8, v242
	v_mov_b32_e32 v9, v242
.LBB0_2:
	s_cmp_lt_i32 s9, s2
	s_set_vgpr_msb 0x81
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[532:539]*/, v[122:129] /*v[378:385]*/, v[66:73], 0
	s_cselect_b32 s3, s9, s11
	s_set_vgpr_msb 0x8106
	ds_store_b128 v4 /*v516*/, v[122:125] /*v[378:381]*/
	ds_store_b128 v4 /*v516*/, v[126:129] /*v[382:385]*/ offset:32
	s_lshl_b32 s3, s3, 5
	ds_store_b128 v4 /*v516*/, v[114:117] /*v[370:373]*/ offset:64
	ds_store_b128 v4 /*v516*/, v[118:121] /*v[374:377]*/ offset:96
	ds_store_b128 v4 /*v516*/, v[106:109] /*v[362:365]*/ offset:128
	ds_store_b128 v4 /*v516*/, v[110:113] /*v[366:369]*/ offset:160
	s_set_vgpr_msb 0x648
	v_or_b32_e32 v130 /*v386*/, s3, v2 /*v514*/
	v_or_b32_e32 v131 /*v387*/, s3, v10 /*v522*/
	s_set_vgpr_msb 0x4881
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[122:129] /*v[378:385]*/, v[138:145], 0
	s_set_vgpr_msb 0x8186
	ds_store_b128 v4 /*v516*/, v[82:85] /*v[338:341]*/ offset:192
	ds_store_b128 v4 /*v516*/, v[86:89] /*v[342:345]*/ offset:224
	ds_store_b128 v5 /*v517*/, v[74:77] /*v[330:333]*/
	ds_store_b128 v5 /*v517*/, v[78:81] /*v[334:337]*/ offset:32
	v_mov_b64_e32 v[68:69] /*v[580:581]*/, s[36:37]
	s_set_vgpr_msb 0x868a
	v_add_nc_u32_e32 v19 /*v531*/, v7 /*v519*/, v8 /*v520*/
	s_add_nc_u64 s[6:7], s[6:7], -1
	s_add_co_i32 s9, s9, 1
	s_cmp_lg_u64 s[6:7], 0
	s_set_vgpr_msb 0x8a81
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[74:81] /*v[330:337]*/, v[66:73], 0
	v_nop
	s_set_vgpr_msb 0x8149
	v_mul_lo_u32 v122 /*v378*/, v130 /*v386*/, s10
	v_mul_lo_u32 v123 /*v379*/, v131 /*v387*/, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_lshl_u32 v182 /*v438*/, v122 /*v378*/, v11 /*v523*/, 4
	v_add_lshl_u32 v190 /*v446*/, v123 /*v379*/, v11 /*v523*/, 4
	s_set_vgpr_msb 0x4981
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[74:81] /*v[330:337]*/, v[138:145], 0
	s_set_vgpr_msb 0x8145
	v_or_b32_e32 v183 /*v439*/, 32, v182 /*v438*/
	v_or_b32_e32 v184 /*v440*/, 64, v182 /*v438*/
	v_or_b32_e32 v185 /*v441*/, 0x60, v182 /*v438*/
	v_or_b32_e32 v186 /*v442*/, 0x80, v182 /*v438*/
	v_or_b32_e32 v187 /*v443*/, 0xa0, v182 /*v438*/
	v_or_b32_e32 v188 /*v444*/, 0xc0, v182 /*v438*/
	v_or_b32_e32 v191 /*v447*/, 0xe0, v182 /*v438*/
	v_or_b32_e32 v192 /*v448*/, 32, v190 /*v446*/
	v_or_b32_e32 v202 /*v458*/, 64, v190 /*v446*/
	v_or_b32_e32 v206 /*v462*/, 0x60, v190 /*v446*/
	v_or_b32_e32 v218 /*v474*/, 0x80, v190 /*v446*/
	v_or_b32_e32 v222 /*v478*/, 0xa0, v190 /*v446*/
	v_or_b32_e32 v234 /*v490*/, 0xc0, v190 /*v446*/
	v_or_b32_e32 v238 /*v494*/, 0xe0, v190 /*v446*/
	s_clause 0x1
	buffer_load_b128 v[122:125] /*v[378:381]*/, v182 /*v438*/, s[44:47], null offen
	buffer_load_b128 v[126:129] /*v[382:385]*/, v183 /*v439*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[158:161] /*v[414:417]*/, v182 /*v438*/, s[48:51], null offen
	buffer_load_b128 v[130:133] /*v[386:389]*/, v183 /*v439*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[134:137] /*v[390:393]*/, v184 /*v440*/, s[44:47], null offen
	buffer_load_b128 v[138:141] /*v[394:397]*/, v185 /*v441*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[142:145] /*v[398:401]*/, v184 /*v440*/, s[48:51], null offen
	buffer_load_b128 v[146:149] /*v[402:405]*/, v185 /*v441*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[150:153] /*v[406:409]*/, v186 /*v442*/, s[44:47], null offen
	buffer_load_b128 v[154:157] /*v[410:413]*/, v187 /*v443*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[162:165] /*v[418:421]*/, v186 /*v442*/, s[48:51], null offen
	buffer_load_b128 v[166:169] /*v[422:425]*/, v187 /*v443*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[170:173] /*v[426:429]*/, v188 /*v444*/, s[44:47], null offen
	buffer_load_b128 v[174:177] /*v[430:433]*/, v191 /*v447*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[178:181] /*v[434:437]*/, v188 /*v444*/, s[48:51], null offen
	buffer_load_b128 v[182:185] /*v[438:441]*/, v191 /*v447*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[242:245] /*v[498:501]*/, v190 /*v446*/, s[44:47], null offen
	buffer_load_b128 v[186:189] /*v[442:445]*/, v192 /*v448*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[246:249] /*v[502:505]*/, v190 /*v446*/, s[48:51], null offen
	buffer_load_b128 v[190:193] /*v[446:449]*/, v192 /*v448*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[194:197] /*v[450:453]*/, v202 /*v458*/, s[44:47], null offen
	buffer_load_b128 v[198:201] /*v[454:457]*/, v206 /*v462*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[202:205] /*v[458:461]*/, v202 /*v458*/, s[48:51], null offen
	buffer_load_b128 v[206:209] /*v[462:465]*/, v206 /*v462*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[210:213] /*v[466:469]*/, v218 /*v474*/, s[44:47], null offen
	buffer_load_b128 v[214:217] /*v[470:473]*/, v222 /*v478*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[218:221] /*v[474:477]*/, v218 /*v474*/, s[48:51], null offen
	buffer_load_b128 v[222:225] /*v[478:481]*/, v222 /*v478*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[226:229] /*v[482:485]*/, v234 /*v490*/, s[44:47], null offen
	buffer_load_b128 v[230:233] /*v[486:489]*/, v238 /*v494*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[234:237] /*v[490:493]*/, v234 /*v490*/, s[48:51], null offen
	buffer_load_b128 v[238:241] /*v[494:497]*/, v238 /*v494*/, s[48:51], null offen
	s_set_vgpr_msb 0x45a1
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[66:73] /*v[322:329]*/, v[146:153], v[44:51] /*v[556:563]*/
	s_set_vgpr_msb 0xa106
	ds_store_b128 v5 /*v517*/, v[66:69] /*v[322:325]*/ offset:64
	ds_store_b128 v5 /*v517*/, v[70:73] /*v[326:329]*/ offset:96
	ds_store_b128 v5 /*v517*/, v[50:53] /*v[306:309]*/ offset:128
	ds_store_b128 v5 /*v517*/, v[54:57] /*v[310:313]*/ offset:160
	ds_store_b128 v5 /*v517*/, v[42:45] /*v[298:301]*/ offset:192
	ds_store_b128 v5 /*v517*/, v[46:49] /*v[302:305]*/ offset:224
	s_set_vgpr_msb 0x6a1
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[66:73] /*v[322:329]*/, v[74:81], v[36:43] /*v[548:555]*/
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[114:121] /*v[370:377]*/, v[146:153], v[28:35] /*v[540:547]*/
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[532:539]*/, v[114:121] /*v[370:377]*/, v[74:81], v[20:27] /*v[532:539]*/
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[50:57] /*v[306:313]*/, v[154:161], v[44:51] /*v[556:563]*/
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[50:57] /*v[306:313]*/, v[82:89], v[36:43] /*v[548:555]*/
	s_delay_alu instid0(TRANS32_DEP_3)
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[532:539]*/, v[106:113] /*v[362:369]*/, v[82:89], v[20:27] /*v[532:539]*/
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[106:113] /*v[362:369]*/, v[154:161], v[28:35] /*v[540:547]*/
	s_set_vgpr_msb 0xa141
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[90:97] /*v[346:353]*/, v[170:177], 0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[90:97] /*v[346:353]*/, v[106:113], 0
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[26:33] /*v[282:289]*/, v[106:113], 0
	s_set_vgpr_msb 0x41a1
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[26:33] /*v[282:289]*/, v[170:177], 0
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[42:49] /*v[298:305]*/, v[162:169], v[44:51] /*v[556:563]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa18a
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[68:69] /*v[580:581]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x8aa1
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[548:555]*/, v[42:49] /*v[298:305]*/, v[98:105], v[36:43] /*v[548:555]*/
	s_set_vgpr_msb 0xa18a
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[68:69] /*v[580:581]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[68:69] /*v[580:581]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[68:69] /*v[580:581]*/, v[50:51] /*v[562:563]*/
	s_set_vgpr_msb 0x8a86
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v14 /*v526*/ offset:4352
	s_set_vgpr_msb 0x8642
	ds_load_tr16_b128 v[66:69] /*v[322:325]*/, v15 /*v527*/
	ds_load_tr16_b128 v[70:73] /*v[326:329]*/, v15 /*v527*/ offset:4352
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v16 /*v528*/
	ds_load_tr16_b128 v[54:57] /*v[310:313]*/, v16 /*v528*/ offset:4352
	ds_load_tr16_b128 v[42:45] /*v[298:301]*/, v17 /*v529*/
	ds_load_tr16_b128 v[46:49] /*v[302:305]*/, v17 /*v529*/ offset:4352
	s_set_vgpr_msb 0x4286
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x86a1
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[532:539]*/, v[82:89] /*v[338:345]*/, v[98:105], v[20:27] /*v[532:539]*/
	s_set_vgpr_msb 0xa18a
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[68:69] /*v[580:581]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[68:69] /*v[580:581]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[68:69] /*v[580:581]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[68:69] /*v[580:581]*/, v[42:43] /*v[554:555]*/
	s_set_vgpr_msb 0x8a86
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x86a1
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[540:547]*/, v[82:89] /*v[338:345]*/, v[162:169], v[28:35] /*v[540:547]*/
	s_set_vgpr_msb 0xa14a
	v_pk_mul_f32 v[26:27] /*v[282:283]*/, v[68:69] /*v[580:581]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[28:29] /*v[284:285]*/, v[68:69] /*v[580:581]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[30:31] /*v[286:287]*/, v[68:69] /*v[580:581]*/, v[24:25] /*v[536:537]*/
	v_pk_mul_f32 v[32:33] /*v[288:289]*/, v[68:69] /*v[580:581]*/, v[26:27] /*v[538:539]*/
	s_set_vgpr_msb 0x4a86
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8645
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[26:27] /*v[282:283]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[98:105] /*v[354:361]*/, v[114:121], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x514a
	v_pk_mul_f32 v[82:83] /*v[338:339]*/, v[68:69] /*v[580:581]*/, v[28:29] /*v[540:541]*/
	v_pk_mul_f32 v[84:85] /*v[340:341]*/, v[68:69] /*v[580:581]*/, v[30:31] /*v[542:543]*/
	v_pk_mul_f32 v[86:87] /*v[342:343]*/, v[68:69] /*v[580:581]*/, v[32:33] /*v[544:545]*/
	v_pk_mul_f32 v[88:89] /*v[344:345]*/, v[68:69] /*v[580:581]*/, v[34:35] /*v[546:547]*/
	s_set_vgpr_msb 0x4a45
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[28:29] /*v[284:285]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[82:83] /*v[338:339]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[84:85] /*v[340:341]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[98:105] /*v[354:361]*/, v[178:185], v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[30:31] /*v[286:287]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[32:33] /*v[288:289]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[86:87] /*v[342:343]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[88:89] /*v[344:345]*/, v[88:89] /*v[344:345]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4582
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8242
	v_pk_mul_f32 v[114:115] /*v[370:371]*/, v[40:41] /*v[552:553]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[34:41] /*v[290:297]*/, v[114:121], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5142
	v_pk_mul_f32 v[116:117] /*v[372:373]*/, v[42:43] /*v[554:555]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119] /*v[374:375]*/, v[44:45] /*v[556:557]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121] /*v[376:377]*/, v[46:47] /*v[558:559]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4282
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[48:49] /*v[560:561]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[50:51] /*v[562:563]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8241
	v_pk_mul_f32 v[26:27] /*v[282:283]*/, v[26:27] /*v[282:283]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29] /*v[284:285]*/, v[28:29] /*v[284:285]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x41a1
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[34:41] /*v[290:297]*/, v[178:185], v[60:67] /*v[572:579]*/
	s_set_vgpr_msb 0xa141
	v_pk_mul_f32 v[82:83] /*v[338:339]*/, v[82:83] /*v[338:339]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85] /*v[340:341]*/, v[84:85] /*v[340:341]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31] /*v[286:287]*/, v[30:31] /*v[286:287]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33] /*v[288:289]*/, v[32:33] /*v[288:289]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87] /*v[342:343]*/, v[86:87] /*v[342:343]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89] /*v[344:345]*/, v[88:89] /*v[344:345]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v70 /*v582*/, v36 /*v548*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[58:65] /*v[314:321]*/, v[122:129], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v71 /*v583*/, v37 /*v549*/
	v_exp_f32_e32 v72 /*v584*/, v38 /*v550*/
	v_exp_f32_e32 v73 /*v585*/, v39 /*v551*/
	s_set_vgpr_msb 0x8281
	v_exp_f32_e32 v74 /*v586*/, v114 /*v370*/
	v_exp_f32_e32 v75 /*v587*/, v115 /*v371*/
	v_exp_f32_e32 v76 /*v588*/, v116 /*v372*/
	v_exp_f32_e32 v77 /*v589*/, v117 /*v373*/
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[58:65] /*v[314:321]*/, v[186:193], v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5181
	v_exp_f32_e32 v78 /*v590*/, v118 /*v374*/
	v_exp_f32_e32 v79 /*v591*/, v119 /*v375*/
	v_exp_f32_e32 v80 /*v592*/, v120 /*v376*/
	v_exp_f32_e32 v81 /*v593*/, v121 /*v377*/
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v82 /*v594*/, v40 /*v552*/
	v_exp_f32_e32 v83 /*v595*/, v41 /*v553*/
	v_exp_f32_e32 v84 /*v596*/, v42 /*v554*/
	s_set_vgpr_msb 0x8251
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[10:17] /*v[266:273]*/, v[122:129], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5182
	v_exp_f32_e32 v85 /*v597*/, v43 /*v555*/
	s_set_vgpr_msb 0x8241
	v_exp_f32_e32 v98 /*v354*/, v26 /*v282*/
	v_exp_f32_e32 v99 /*v355*/, v27 /*v283*/
	v_exp_f32_e32 v34 /*v290*/, v28 /*v284*/
	v_exp_f32_e32 v35 /*v291*/, v29 /*v285*/
	v_exp_f32_e32 v36 /*v292*/, v30 /*v286*/
	v_exp_f32_e32 v37 /*v293*/, v31 /*v287*/
	s_set_vgpr_msb 0x41a1
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[10:17] /*v[266:273]*/, v[186:193], v[60:67] /*v[572:579]*/
	s_set_vgpr_msb 0xa151
	v_exp_f32_e32 v38 /*v294*/, v32 /*v288*/
	v_exp_f32_e32 v39 /*v295*/, v33 /*v289*/
	v_exp_f32_e32 v40 /*v296*/, v82 /*v338*/
	v_exp_f32_e32 v41 /*v297*/, v83 /*v339*/
	v_exp_f32_e32 v82 /*v338*/, v84 /*v340*/
	v_exp_f32_e32 v83 /*v339*/, v85 /*v341*/
	v_exp_f32_e32 v84 /*v340*/, v86 /*v342*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[18:25] /*v[274:281]*/, v[130:137], v[106:113] /*v[362:369]*/
	v_exp_f32_e32 v85 /*v341*/, v87 /*v343*/
	v_exp_f32_e32 v86 /*v342*/, v88 /*v344*/
	v_exp_f32_e32 v87 /*v343*/, v89 /*v345*/
	s_set_vgpr_msb 0x5142
	ds_load_tr16_b128 v[114:117] /*v[370:373]*/, v19 /*v531*/
	ds_load_tr16_b128 v[118:121] /*v[374:377]*/, v19 /*v531*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v12 /*v524*/
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v12 /*v524*/ offset:4352
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v13 /*v525*/
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v13 /*v525*/ offset:4352
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v14 /*v526*/
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[26:29] /*v[282:285]*/, v18 /*v530*/
	ds_load_tr16_b128 v[30:33] /*v[286:289]*/, v18 /*v530*/ offset:4352
	s_set_vgpr_msb 0x4251
	s_wait_loadcnt 0x19
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[144:145] /*v[400:401]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[18:25] /*v[274:281]*/, v[194:201], v[74:81] /*v[330:337]*/
	s_wait_loadcnt 0x18
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[146:147] /*v[402:403]*/
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[148:149] /*v[404:405]*/
	s_wait_loadcnt 0x12
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[176:177] /*v[432:433]*/
	v_nop
	s_set_vgpr_msb 0x5149
	v_pk_add_f32 v[10:11] /*v[266:267]*/, v[74:75] /*v[330:331]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4951
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[2:9] /*v[258:265]*/, v[130:137], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5149
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[76:77] /*v[332:333]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[78:79] /*v[334:335]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[80:81] /*v[336:337]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[10:11] /*v[266:267]*/, v[10:11] /*v[266:267]*/, v[40:41] /*v[296:297]*/
	s_wait_loadcnt 0xf
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[242:243] /*v[498:499]*/
	v_pk_mul_f32 v[12:13] /*v[268:269]*/, v[12:13] /*v[268:269]*/, v[82:83] /*v[338:339]*/
	v_pk_mul_f32 v[14:15] /*v[270:271]*/, v[14:15] /*v[270:271]*/, v[84:85] /*v[340:341]*/
	s_set_vgpr_msb 0x45a1
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[2:9] /*v[258:265]*/, v[194:201], v[60:67] /*v[572:579]*/
	s_set_vgpr_msb 0xa145
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[90:91] /*v[346:347]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[92:93] /*v[348:349]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[94:95] /*v[350:351]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[96:97] /*v[352:353]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[2:3] /*v[258:259]*/, v[106:107] /*v[362:363]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5] /*v[260:261]*/, v[108:109] /*v[364:365]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[6:7] /*v[262:263]*/, v[110:111] /*v[366:367]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[112:113] /*v[368:369]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[58:59] /*v[314:315]*/, v[60:61] /*v[572:573]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[62:63] /*v[574:575]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[64:65] /*v[576:577]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[66:67] /*v[578:579]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a45
	v_pk_mul_f32 v[2:3] /*v[258:259]*/, v[2:3] /*v[258:259]*/, v[98:99] /*v[354:355]*/
	v_pk_mul_f32 v[4:5] /*v[260:261]*/, v[4:5] /*v[260:261]*/, v[34:35] /*v[290:291]*/
	v_pk_mul_f32 v[6:7] /*v[262:263]*/, v[6:7] /*v[262:263]*/, v[36:37] /*v[292:293]*/
	v_pk_mul_f32 v[8:9] /*v[264:265]*/, v[8:9] /*v[264:265]*/, v[38:39] /*v[294:295]*/
	v_pk_mul_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, v[86:87] /*v[342:343]*/
	s_set_vgpr_msb 0x4549
	v_pk_mul_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, v[74:75] /*v[586:587]*/
	v_pk_mul_f32 v[24:25] /*v[280:281]*/, v[24:25] /*v[280:281]*/, v[76:77] /*v[588:589]*/
	v_pk_mul_f32 v[34:35] /*v[290:291]*/, v[58:59] /*v[314:315]*/, v[78:79] /*v[590:591]*/
	v_pk_mul_f32 v[36:37] /*v[292:293]*/, v[60:61] /*v[316:317]*/, v[80:81] /*v[592:593]*/
	v_pk_mul_f32 v[38:39] /*v[294:295]*/, v[62:63] /*v[318:319]*/, v[82:83] /*v[594:595]*/
	v_pk_mul_f32 v[40:41] /*v[296:297]*/, v[64:65] /*v[320:321]*/, v[84:85] /*v[596:597]*/
	v_pk_mul_f32 v[2:3] /*v[258:259]*/, v[2:3] /*v[258:259]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[4:5] /*v[260:261]*/, v[4:5] /*v[260:261]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[6:7] /*v[262:263]*/, v[6:7] /*v[262:263]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[8:9] /*v[264:265]*/, v[8:9] /*v[264:265]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[10:11] /*v[266:267]*/, v[10:11] /*v[266:267]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[12:13] /*v[268:269]*/, v[12:13] /*v[268:269]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[14:15] /*v[270:271]*/, v[14:15] /*v[270:271]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[24:25] /*v[280:281]*/, v[24:25] /*v[280:281]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[34:35] /*v[290:291]*/, v[34:35] /*v[290:291]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[36:37] /*v[292:293]*/, v[36:37] /*v[292:293]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[38:39] /*v[294:295]*/, v[38:39] /*v[294:295]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[40:41] /*v[296:297]*/, v[40:41] /*v[296:297]*/, v[68:69] /*v[580:581]*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v2 /*v258*/, v2 /*v258*/, v3 /*v259*/
	v_cvt_pk_bf16_f32 v3 /*v259*/, v4 /*v260*/, v5 /*v261*/
	v_cvt_pk_bf16_f32 v4 /*v260*/, v6 /*v262*/, v7 /*v263*/
	v_cvt_pk_bf16_f32 v5 /*v261*/, v8 /*v264*/, v9 /*v265*/
	v_cvt_pk_bf16_f32 v10 /*v266*/, v10 /*v266*/, v11 /*v267*/
	v_cvt_pk_bf16_f32 v6 /*v262*/, v18 /*v274*/, v19 /*v275*/
	v_cvt_pk_bf16_f32 v7 /*v263*/, v20 /*v276*/, v21 /*v277*/
	v_cvt_pk_bf16_f32 v8 /*v264*/, v22 /*v278*/, v23 /*v279*/
	v_cvt_pk_bf16_f32 v9 /*v265*/, v24 /*v280*/, v25 /*v281*/
	v_cvt_pk_bf16_f32 v11 /*v267*/, v12 /*v268*/, v13 /*v269*/
	v_cvt_pk_bf16_f32 v12 /*v268*/, v14 /*v270*/, v15 /*v271*/
	v_cvt_pk_bf16_f32 v13 /*v269*/, v16 /*v272*/, v17 /*v273*/
	v_cvt_pk_bf16_f32 v14 /*v270*/, v34 /*v290*/, v35 /*v291*/
	v_cvt_pk_bf16_f32 v15 /*v271*/, v36 /*v292*/, v37 /*v293*/
	v_cvt_pk_bf16_f32 v16 /*v272*/, v38 /*v294*/, v39 /*v295*/
	v_cvt_pk_bf16_f32 v17 /*v273*/, v40 /*v296*/, v41 /*v297*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x32_bf16 v[242:249], v[2:9] /*v[258:265]*/, v[114:121] /*v[370:377]*/, v[242:249]
	s_set_vgpr_msb 0x541
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[244:245] /*v[500:501]*/
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, v[158:159] /*v[414:415]*/
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, v[160:161] /*v[416:417]*/
	s_wait_loadcnt 0x8
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[206:207] /*v[462:463]*/
	v_mov_b64_e32 v[40:41] /*v[296:297]*/, v[208:209] /*v[464:465]*/
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[202:203] /*v[458:459]*/
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[204:205] /*v[460:461]*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[58:65], v[10:17] /*v[266:273]*/, v[114:121] /*v[370:377]*/, v[58:65]
	s_set_vgpr_msb 0x541
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[186:187] /*v[442:443]*/
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[188:189] /*v[444:445]*/
	v_mov_b64_e32 v[22:23] /*v[278:279]*/, v[182:183] /*v[438:439]*/
	v_mov_b64_e32 v[24:25] /*v[280:281]*/, v[184:185] /*v[440:441]*/
	v_mov_b64_e32 v[18:19] /*v[274:275]*/, v[178:179] /*v[434:435]*/
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[180:181] /*v[436:437]*/
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[174:175] /*v[430:431]*/
	s_set_vgpr_msb 0x4109
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[250:257], v[2:9] /*v[258:265]*/, v[36:43] /*v[548:555]*/, v[250:257]
	s_set_vgpr_msb 0x941
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[170:171] /*v[426:427]*/
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[172:173] /*v[428:429]*/
	v_mov_b64_e32 v[62:63] /*v[318:319]*/, v[166:167] /*v[422:423]*/
	v_mov_b64_e32 v[64:65] /*v[320:321]*/, v[168:169] /*v[424:425]*/
	v_mov_b64_e32 v[58:59] /*v[314:315]*/, v[162:163] /*v[418:419]*/
	v_mov_b64_e32 v[60:61] /*v[316:317]*/, v[164:165] /*v[420:421]*/
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[154:155] /*v[410:411]*/
	s_set_vgpr_msb 0x4109
	v_wmma_f32_16x16x32_bf16 v[50:57], v[10:17] /*v[266:273]*/, v[36:43] /*v[548:555]*/, v[50:57]
	s_set_vgpr_msb 0x941
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[156:157] /*v[412:413]*/
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[150:151] /*v[406:407]*/
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[152:153] /*v[408:409]*/
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[142:143] /*v[398:399]*/
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[138:139] /*v[394:395]*/
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[140:141] /*v[396:397]*/
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[134:135] /*v[390:391]*/
	s_set_vgpr_msb 0x4109
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[234:241], v[2:9] /*v[258:265]*/, v[44:51] /*v[556:563]*/, v[234:241]
	s_set_vgpr_msb 0x941
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[136:137] /*v[392:393]*/
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, v[130:131] /*v[386:387]*/
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, v[132:133] /*v[388:389]*/
	s_set_vgpr_msb 0x4109
	v_wmma_f32_16x16x32_bf16 v[42:49], v[10:17] /*v[266:273]*/, v[44:51] /*v[556:563]*/, v[42:49]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[226:233], v[2:9] /*v[258:265]*/, v[52:59] /*v[564:571]*/, v[226:233]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[10:17] /*v[266:273]*/, v[52:59] /*v[564:571]*/, v[34:41]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[218:225], v[2:9] /*v[258:265]*/, v[66:73] /*v[322:329]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[10:17] /*v[266:273]*/, v[66:73] /*v[322:329]*/, v[26:33]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x541
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[198:199] /*v[454:455]*/
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[200:201] /*v[456:457]*/
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[194:195] /*v[450:451]*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[210:217], v[2:9] /*v[258:265]*/, v[50:57] /*v[306:313]*/, v[210:217]
	s_set_vgpr_msb 0x541
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[18:25], v[10:17] /*v[266:273]*/, v[50:57] /*v[306:313]*/, v[18:25]
	s_wait_loadcnt 0x6
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x541
	v_mov_b64_e32 v[54:55] /*v[310:311]*/, v[214:215] /*v[470:471]*/
	v_mov_b64_e32 v[56:57] /*v[312:313]*/, v[216:217] /*v[472:473]*/
	v_mov_b64_e32 v[50:51] /*v[306:307]*/, v[210:211] /*v[466:467]*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[202:209], v[2:9] /*v[258:265]*/, v[42:49] /*v[298:305]*/, v[202:209]
	s_set_vgpr_msb 0x541
	v_mov_b64_e32 v[52:53] /*v[308:309]*/, v[212:213] /*v[468:469]*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[10:17], v[10:17] /*v[266:273]*/, v[42:49] /*v[298:305]*/, v[10:17]
	s_wait_loadcnt 0x2
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x541
	v_mov_b64_e32 v[46:47] /*v[302:303]*/, v[230:231] /*v[486:487]*/
	v_mov_b64_e32 v[48:49] /*v[304:305]*/, v[232:233] /*v[488:489]*/
	v_mov_b64_e32 v[42:43] /*v[298:299]*/, v[226:227] /*v[482:483]*/
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[2:9] /*v[258:265]*/, v[26:33] /*v[282:289]*/, v[90:97]
	s_set_vgpr_msb 0x541
	v_mov_b64_e32 v[44:45] /*v[300:301]*/, v[228:229] /*v[484:485]*/
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[6:7] /*v[262:263]*/, v[238:239] /*v[494:495]*/
	v_mov_b64_e32 v[8:9] /*v[264:265]*/, v[240:241] /*v[496:497]*/
	v_mov_b64_e32 v[2:3] /*v[258:259]*/, v[234:235] /*v[490:491]*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[2:9], v[10:17] /*v[266:273]*/, v[26:33] /*v[282:289]*/, v[2:9]
	s_set_vgpr_msb 0x541
	v_mov_b64_e32 v[4:5] /*v[260:261]*/, v[236:237] /*v[492:493]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[26:27] /*v[282:283]*/, v[246:247] /*v[502:503]*/
	v_mov_b64_e32 v[28:29] /*v[284:285]*/, v[248:249] /*v[504:505]*/
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, v[222:223] /*v[478:479]*/
	v_mov_b64_e32 v[16:17] /*v[272:273]*/, v[224:225] /*v[480:481]*/
	v_mov_b64_e32 v[10:11] /*v[266:267]*/, v[218:219] /*v[474:475]*/
	v_mov_b64_e32 v[12:13] /*v[268:269]*/, v[220:221] /*v[476:477]*/
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[190:191] /*v[446:447]*/
	v_mov_b64_e32 v[32:33] /*v[288:289]*/, v[192:193] /*v[448:449]*/
	s_set_vgpr_msb 0x4100
	s_cbranch_scc1 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	v_mov_b32_e32 v2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v4, v2
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v6, v2
	v_dual_mov_b32 v7, v2 :: v_dual_mov_b32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[10:11], v[2:3]
	s_delay_alu instid0(VALU_DEP_4)
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[16:17], v[8:9]
	v_mov_b64_e32 v[24:25], v[8:9]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[32:33], v[8:9]
	v_mov_b64_e32 v[30:31], v[6:7]
	v_mov_b64_e32 v[28:29], v[4:5]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[40:41], v[8:9]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[48:49], v[8:9]
	v_mov_b64_e32 v[46:47], v[6:7]
	v_mov_b64_e32 v[44:45], v[4:5]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[56:57], v[8:9]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[64:65], v[8:9]
	v_mov_b64_e32 v[62:63], v[6:7]
	v_mov_b64_e32 v[60:61], v[4:5]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[96:97], v[8:9]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[208:209], v[8:9]
	v_mov_b64_e32 v[206:207], v[6:7]
	v_mov_b64_e32 v[204:205], v[4:5]
	v_mov_b64_e32 v[202:203], v[2:3]
	v_mov_b64_e32 v[216:217], v[8:9]
	v_mov_b64_e32 v[214:215], v[6:7]
	v_mov_b64_e32 v[212:213], v[4:5]
	v_mov_b64_e32 v[210:211], v[2:3]
	v_mov_b64_e32 v[224:225], v[8:9]
	v_mov_b64_e32 v[222:223], v[6:7]
	v_mov_b64_e32 v[220:221], v[4:5]
	v_mov_b64_e32 v[218:219], v[2:3]
	v_mov_b64_e32 v[232:233], v[8:9]
	v_mov_b64_e32 v[230:231], v[6:7]
	v_mov_b64_e32 v[228:229], v[4:5]
	v_mov_b64_e32 v[226:227], v[2:3]
	v_mov_b64_e32 v[240:241], v[8:9]
	v_mov_b64_e32 v[238:239], v[6:7]
	v_mov_b64_e32 v[236:237], v[4:5]
	v_mov_b64_e32 v[234:235], v[2:3]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[0:1] /*v[256:257]*/, v[8:9]
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[254:255], v[6:7]
	v_mov_b64_e32 v[252:253], v[4:5]
	v_mov_b64_e32 v[250:251], v[2:3]
	v_mov_b64_e32 v[248:249], v[8:9]
	v_mov_b64_e32 v[246:247], v[6:7]
	v_mov_b64_e32 v[244:245], v[4:5]
	v_mov_b64_e32 v[242:243], v[2:3]
.LBB0_4:
	s_sub_co_i32 s34, s21, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s34, 1
	s_cbranch_scc1 .LBB0_7
	s_set_vgpr_msb 0x48
	v_add_lshl_u32 v3 /*v259*/, s8, v2 /*v514*/, 4
	s_set_vgpr_msb 0x4841
	s_wait_loadcnt 0x1
	v_dual_mov_b32 v255 /*v511*/, v254 /*v510*/ :: v_dual_add_nc_u32 v4 /*v260*/, s43, v1
	s_set_vgpr_msb 0x4108
	v_lshlrev_b32_e32 v1, 4, v2 /*v514*/
	s_lshl_b32 s4, s2, 9
	s_lshl_b32 s3, s8, 4
	s_set_vgpr_msb 0x844
	v_add3_u32 v3 /*v259*/, s4, v3 /*v259*/, 0x100
	s_set_vgpr_msb 0x4449
	v_dual_mov_b32 v251 /*v507*/, v250 /*v506*/ :: v_dual_bitop2_b32 v12 /*v268*/, 64, v8 /*v520*/ bitop3:0x54
	s_set_vgpr_msb 0x4900
	v_add3_u32 v1, s3, s4, v1
	s_set_vgpr_msb 0x48
	v_dual_add_nc_u32 v2 /*v258*/, s20, v2 /*v514*/ :: v_dual_bitop2_b32 v11 /*v267*/, 32, v9 /*v521*/ bitop3:0x54
	s_set_vgpr_msb 0x4844
	v_mul_lo_u32 v3 /*v259*/, s40, v3 /*v259*/
	s_set_vgpr_msb 0x4448
	v_or_b32_e32 v13 /*v269*/, 0x60, v9 /*v521*/
	s_set_vgpr_msb 0x4800
	v_mul_lo_u32 v1, s40, v1
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v14 /*v270*/, 0x80, v8 /*v520*/
	v_or_b32_e32 v15 /*v271*/, 0xa0, v9 /*v521*/
	v_or_b32_e32 v16 /*v272*/, 0xc0, v8 /*v520*/
	v_or_b32_e32 v17 /*v273*/, 0xe0, v9 /*v521*/
	s_mov_b32 s37, s36
	s_set_vgpr_msb 0x4846
	v_dual_add_nc_u32 v12 /*v268*/, v7 /*v519*/, v12 /*v268*/ :: v_dual_bitop2_b32 v6 /*v262*/, v6 /*v518*/, v3 /*v259*/ bitop3:0x54
	s_set_vgpr_msb 0x4642
	v_or_b32_e32 v5 /*v261*/, v6 /*v518*/, v1
	s_set_vgpr_msb 0x4282
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v1 /*v513*/, v0 /*v512*/
	s_set_vgpr_msb 0x8201
	v_mov_b32_e32 v1, v2 /*v258*/
	s_set_vgpr_msb 0x161
	v_add_lshl_u32 v9 /*v265*/, v6 /*v262*/, s5, 4
	v_mov_b64_e32 v[6:7] /*v[262:263]*/, s[36:37]
	v_dual_mov_b32 v253 /*v509*/, v252 /*v508*/ :: v_dual_mov_b32 v3 /*v259*/, v4 /*v260*/
	v_add_lshl_u32 v5 /*v261*/, v5 /*v261*/, s5, 4
	v_lshl_or_b32 v8 /*v264*/, s2, 5, v3 /*v515*/
	s_set_vgpr_msb 0x614a
	v_add_nc_u32_e32 v10 /*v266*/, v7 /*v519*/, v8 /*v520*/
	s_set_vgpr_msb 0x4a46
	v_dual_add_nc_u32 v11 /*v267*/, v7 /*v519*/, v11 /*v267*/ :: v_dual_add_nc_u32 v13 /*v269*/, v7 /*v519*/, v13 /*v269*/
	v_dual_add_nc_u32 v14 /*v270*/, v7 /*v519*/, v14 /*v270*/ :: v_dual_add_nc_u32 v15 /*v271*/, v7 /*v519*/, v15 /*v271*/
	v_dual_add_nc_u32 v16 /*v272*/, v7 /*v519*/, v16 /*v272*/ :: v_dual_add_nc_u32 v17 /*v273*/, v7 /*v519*/, v17 /*v273*/
	s_ashr_i32 s35, s34, 31
	s_lshl_b32 s37, s40, 13
	s_mov_b32 s36, 0x3fb8aa3b
	s_set_vgpr_msb 0x4600
.LBB0_6:
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x45
	v_or_b32_e32 v42 /*v298*/, 32, v5 /*v261*/
	v_or_b32_e32 v43 /*v299*/, 64, v5 /*v261*/
	s_clause 0x1
	buffer_load_b128 v[18:21] /*v[274:277]*/, v5 /*v261*/, s[44:47], null offen
	buffer_load_b128 v[26:29] /*v[282:285]*/, v9 /*v265*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[30:33] /*v[286:289]*/, v5 /*v261*/, s[48:51], null offen
	buffer_load_b128 v[38:41] /*v[294:297]*/, v9 /*v265*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[22:25] /*v[278:281]*/, v42 /*v298*/, s[44:47], null offen
	buffer_load_b128 v[46:49] /*v[302:305]*/, v43 /*v299*/, s[44:47], null offen
	v_or_b32_e32 v130 /*v386*/, 0x60, v5 /*v261*/
	v_or_b32_e32 v131 /*v387*/, 0x80, v5 /*v261*/
	s_clause 0x3
	buffer_load_b128 v[34:37] /*v[290:293]*/, v42 /*v298*/, s[48:51], null offen
	buffer_load_b128 v[62:65] /*v[318:321]*/, v43 /*v299*/, s[48:51], null offen
	buffer_load_b128 v[66:69] /*v[322:325]*/, v130 /*v386*/, s[48:51], null offen
	buffer_load_b128 v[126:129] /*v[382:385]*/, v131 /*v387*/, s[48:51], null offen
	s_wait_xcnt 0x3
	v_or_b32_e32 v42 /*v298*/, 32, v9 /*v265*/
	v_or_b32_e32 v166 /*v422*/, 0xe0, v5 /*v261*/
	v_or_b32_e32 v170 /*v426*/, 0xc0, v9 /*v265*/
	v_or_b32_e32 v171 /*v427*/, 0xe0, v9 /*v265*/
	v_or_b32_e32 v174 /*v430*/, 3, v8 /*v264*/
	v_or_b32_e32 v175 /*v431*/, 2, v8 /*v264*/
	v_cmp_ge_i32_e32 vcc_lo, v8 /*v264*/, v2 /*v258*/
	v_cmp_gt_i32_e64 s2, v8 /*v264*/, v2 /*v258*/
	v_or_b32_e32 v176 /*v432*/, 5, v8 /*v264*/
	v_cmp_ge_i32_e64 s3, v8 /*v264*/, v4 /*v260*/
	v_cmp_gt_i32_e64 s4, v8 /*v264*/, v4 /*v260*/
	s_and_b32 s38, s53, vcc_lo
	s_and_b32 s2, s53, s2
	s_set_vgpr_msb 0x4501
	v_cmp_gt_i32_e64 s13, v176 /*v432*/, v1
	s_set_vgpr_msb 0x105
	v_cmp_gt_i32_e64 s8, v176 /*v432*/, v3 /*v259*/
	s_and_b32 s3, s53, s3
	s_and_b32 s4, s53, s4
	s_add_nc_u64 s[34:35], s[34:35], -1
	s_and_b32 s13, s53, s13
	s_and_b32 s8, s53, s8
	s_set_vgpr_msb 0x541
	s_wait_loadcnt 0x5
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[310:317]*/, v[18:25] /*v[274:281]*/, v[66:73], 0
	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[30:37] /*v[286:293]*/, v[106:113], 0
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[30:37] /*v[286:293]*/, v[170:177], 0
	s_clause 0x2
	buffer_load_b128 v[30:33] /*v[286:289]*/, v42 /*v298*/, s[44:47], null offen
	buffer_load_b128 v[50:53] /*v[306:309]*/, v130 /*v386*/, s[44:47], null offen
	buffer_load_b128 v[34:37] /*v[290:293]*/, v131 /*v387*/, s[44:47], null offen
	s_set_vgpr_msb 0x4106
	ds_store_b128 v4 /*v516*/, v[18:21] /*v[274:277]*/
	ds_store_b128 v4 /*v516*/, v[22:25] /*v[278:281]*/ offset:32
	ds_store_b128 v4 /*v516*/, v[46:49] /*v[302:305]*/ offset:64
	s_wait_loadcnt 0x1
	ds_store_b128 v4 /*v516*/, v[50:53] /*v[306:309]*/ offset:96
	s_wait_loadcnt 0x0
	ds_store_b128 v4 /*v516*/, v[34:37] /*v[290:293]*/ offset:128
	s_set_vgpr_msb 0x651
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[62:69] /*v[318:325]*/, v[114:121], v[70:77] /*v[326:333]*/
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[62:69] /*v[318:325]*/, v[178:185], v[86:93] /*v[342:349]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5145
	v_or_b32_e32 v68 /*v324*/, 0x80, v9 /*v265*/
	v_or_b32_e32 v69 /*v325*/, 0xa0, v9 /*v265*/
	v_or_b32_e32 v66 /*v322*/, 0xa0, v5 /*v261*/
	v_or_b32_e32 v67 /*v323*/, 0xc0, v5 /*v261*/
	s_clause 0x1
	buffer_load_b128 v[130:133] /*v[386:389]*/, v66 /*v322*/, s[48:51], null offen
	buffer_load_b128 v[142:145] /*v[398:401]*/, v67 /*v323*/, s[48:51], null offen
	buffer_load_b128 v[150:153] /*v[406:409]*/, v68 /*v324*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[42:45] /*v[298:301]*/, v42 /*v298*/, s[48:51], null offen
	buffer_load_b128 v[162:165] /*v[418:421]*/, v69 /*v325*/, s[48:51], null offen
	s_set_vgpr_msb 0x4541
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[102:109] /*v[358:365]*/, v[38:45] /*v[294:301]*/, v[106:113], 0
	buffer_load_b128 v[154:157] /*v[410:413]*/, v69 /*v325*/, s[44:47], null offen
	buffer_load_b128 v[158:161] /*v[414:417]*/, v68 /*v324*/, s[48:51], null offen
	s_set_vgpr_msb 0x4144
	v_add_nc_u32_e32 v5 /*v261*/, s37, v5 /*v261*/
	v_cmp_lt_i32_e64 s15, v1, v174 /*v430*/
	s_set_vgpr_msb 0x4405
	v_cmp_gt_i32_e64 s9, v174 /*v430*/, v3 /*v259*/
	s_and_b32 s15, s53, s15
	s_set_vgpr_msb 0x541
	v_wmma_f32_16x16x32_bf16 v[118:125] /*v[374:381]*/, v[38:45] /*v[294:301]*/, v[170:177], 0
	s_and_b32 s9, s53, s9
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x4145
	v_or_b32_e32 v42 /*v298*/, 64, v9 /*v265*/
	v_or_b32_e32 v43 /*v299*/, 0x60, v9 /*v265*/
	buffer_load_b128 v[138:141] /*v[394:397]*/, v43 /*v299*/, s[44:47], null offen
	buffer_load_b128 v[38:41] /*v[294:297]*/, v42 /*v298*/, s[48:51], null offen
	buffer_load_b128 v[134:137] /*v[390:393]*/, v42 /*v298*/, s[44:47], null offen
	buffer_load_b128 v[42:45] /*v[298:301]*/, v43 /*v299*/, s[48:51], null offen
	s_set_vgpr_msb 0x4551
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[102:109] /*v[358:365]*/, v[38:45] /*v[294:301]*/, v[114:121], v[102:109] /*v[358:365]*/
	s_set_vgpr_msb 0x5145
	v_add_nc_u32_e32 v9 /*v265*/, s37, v9 /*v265*/
	v_cmp_gt_i32_e64 s18, v175 /*v431*/, v2 /*v258*/
	v_cmp_gt_i32_e64 s12, v175 /*v431*/, v4 /*v260*/
	s_and_b32 s18, s53, s18
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[118:125] /*v[374:381]*/, v[38:45] /*v[294:301]*/, v[178:185], v[118:125] /*v[374:381]*/
	s_clause 0x2
	buffer_load_b128 v[38:41] /*v[294:297]*/, v66 /*v322*/, s[44:47], null offen
	buffer_load_b128 v[62:65] /*v[318:321]*/, v67 /*v323*/, s[44:47], null offen
	buffer_load_b128 v[66:69] /*v[322:325]*/, v166 /*v422*/, s[44:47], null offen
	s_set_vgpr_msb 0x5106
	s_wait_loadcnt 0x2
	ds_store_b128 v4 /*v516*/, v[38:41] /*v[294:297]*/ offset:160
	s_wait_loadcnt 0x1
	ds_store_b128 v4 /*v516*/, v[62:65] /*v[318:321]*/ offset:192
	s_wait_loadcnt 0x0
	ds_store_b128 v4 /*v516*/, v[66:69] /*v[322:325]*/ offset:224
	ds_store_b128 v5 /*v517*/, v[26:29] /*v[282:285]*/
	ds_store_b128 v5 /*v517*/, v[30:33] /*v[286:289]*/ offset:32
	ds_store_b128 v5 /*v517*/, v[134:137] /*v[390:393]*/ offset:64
	ds_store_b128 v5 /*v517*/, v[138:141] /*v[394:397]*/ offset:96
	s_set_vgpr_msb 0x651
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[126:133] /*v[382:389]*/, v[122:129], v[70:77] /*v[326:333]*/
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5145
	v_or_b32_e32 v42 /*v298*/, 4, v8 /*v264*/
	v_or_b32_e32 v43 /*v299*/, 7, v8 /*v264*/
	v_dual_add_nc_u32 v45 /*v301*/, 16, v8 /*v264*/ :: v_dual_bitop2_b32 v44 /*v300*/, 6, v8 /*v264*/ bitop3:0x54
	s_and_b32 s12, s53, s12
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s16, v42 /*v298*/, v2 /*v258*/
	s_set_vgpr_msb 0x4551
	v_cmp_gt_i32_e64 s14, v43 /*v299*/, v1
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[126:133] /*v[382:389]*/, v[186:193], v[86:93] /*v[342:349]*/
	buffer_load_b128 v[146:149] /*v[402:405]*/, v166 /*v422*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[126:129] /*v[382:385]*/, v170 /*v426*/, s[44:47], null offen
	buffer_load_b128 v[130:133] /*v[386:389]*/, v171 /*v427*/, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[166:169] /*v[422:425]*/, v170 /*v426*/, s[48:51], null offen
	buffer_load_b128 v[170:173] /*v[426:429]*/, v171 /*v427*/, s[48:51], null offen
	s_set_vgpr_msb 0x5105
	v_cmp_gt_i32_e64 s17, v44 /*v300*/, v2 /*v258*/
	v_cmp_gt_i32_e64 s10, v42 /*v298*/, v4 /*v260*/
	v_cmp_gt_i32_e64 s7, v43 /*v299*/, v3 /*v259*/
	v_cmp_gt_i32_e64 s11, v44 /*v300*/, v4 /*v260*/
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[310:317]*/, v[46:53] /*v[302:309]*/, v[74:81], v[54:61] /*v[310:317]*/
	s_set_vgpr_msb 0x5145
	v_cmp_ge_i32_e64 s5, v45 /*v301*/, v2 /*v258*/
	v_cmp_gt_i32_e64 s6, v45 /*v301*/, v2 /*v258*/
	v_or_b32_e32 v42 /*v298*/, 3, v45 /*v301*/
	v_or_b32_e32 v43 /*v299*/, 2, v45 /*v301*/
	v_or_b32_e32 v44 /*v300*/, 5, v45 /*v301*/
	s_and_b32 s16, s53, s16
	v_cmp_ge_i32_e64 s19, v45 /*v301*/, v4 /*v260*/
	s_set_vgpr_msb 0x4541
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[18:25] /*v[274:281]*/, v[138:145], 0
	s_set_vgpr_msb 0x4105
	v_cmp_gt_i32_e64 s20, v45 /*v301*/, v4 /*v260*/
	s_set_vgpr_msb 0x501
	v_cmp_gt_i32_e64 s21, v42 /*v298*/, v1
	s_set_vgpr_msb 0x105
	v_cmp_gt_i32_e64 s22, v43 /*v299*/, v2 /*v258*/
	s_set_vgpr_msb 0x501
	v_cmp_gt_i32_e64 s23, v44 /*v300*/, v1
	s_set_vgpr_msb 0x105
	v_cmp_gt_i32_e64 s27, v42 /*v298*/, v3 /*v259*/
	v_cmp_gt_i32_e64 s28, v43 /*v299*/, v4 /*v260*/
	v_cmp_gt_i32_e64 s29, v44 /*v300*/, v3 /*v259*/
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[26:33] /*v[282:289]*/, v[66:73], 0
	s_and_b32 s5, s53, s5
	s_and_b32 s6, s53, s6
	s_and_b32 s17, s53, s17
	s_and_b32 s14, s53, s14
	s_and_b32 s10, s53, s10
	s_and_b32 s11, s53, s11
	s_and_b32 s7, s53, s7
	v_wmma_f32_16x16x32_bf16 v[110:117] /*v[366:373]*/, v[26:33] /*v[282:289]*/, v[138:145], 0
	s_and_b32 s22, s53, s22
	s_and_b32 s21, s53, s21
	s_and_b32 s23, s53, s23
	s_and_b32 s19, s53, s19
	s_and_b32 s20, s53, s20
	s_and_b32 s28, s53, s28
	s_and_b32 s27, s53, s27
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[310:317]*/, v[34:41] /*v[290:297]*/, v[82:89], v[54:61] /*v[310:317]*/
	s_and_b32 s29, s53, s29
	s_set_vgpr_msb 0x5106
	ds_store_b128 v5 /*v517*/, v[150:153] /*v[406:409]*/ offset:128
	ds_store_b128 v5 /*v517*/, v[154:157] /*v[410:413]*/ offset:160
	s_wait_loadcnt 0x3
	ds_store_b128 v5 /*v517*/, v[126:129] /*v[382:385]*/ offset:192
	s_wait_loadcnt 0x2
	ds_store_b128 v5 /*v517*/, v[130:133] /*v[386:389]*/ offset:224
	s_set_vgpr_msb 0x651
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[46:53] /*v[302:309]*/, v[146:153], v[78:85] /*v[334:341]*/
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v10 /*v266*/
	ds_load_tr16_b128 v[22:25] /*v[278:281]*/, v10 /*v266*/ offset:4352
	ds_load_tr16_b128 v[26:29] /*v[282:285]*/, v11 /*v267*/
	s_set_vgpr_msb 0x5144
	v_add_nc_u32_e32 v8 /*v264*/, 32, v8 /*v264*/
	s_set_vgpr_msb 0x4451
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[134:141] /*v[390:397]*/, v[74:81], v[94:101] /*v[350:357]*/
	v_wmma_f32_16x16x32_bf16 v[110:117] /*v[366:373]*/, v[134:141] /*v[390:397]*/, v[146:153], v[110:117] /*v[366:373]*/
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[310:317]*/, v[62:69] /*v[318:325]*/, v[98:105], v[54:61] /*v[310:317]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5145
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[30:31] /*v[286:287]*/, v[6:7] /*v[262:263]*/, v[54:55] /*v[310:311]*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[34:41] /*v[290:297]*/, v[154:161], v[78:85] /*v[334:341]*/
	s_set_vgpr_msb 0x5145
	v_pk_mul_f32 v[32:33] /*v[288:289]*/, v[6:7] /*v[262:263]*/, v[56:57] /*v[312:313]*/
	v_cndmask_b32_e64 v31 /*v287*/, v31 /*v287*/, 0xff61b1e6, s38
	v_nop
	v_nop
	v_pk_mul_f32 v[34:35] /*v[290:291]*/, v[6:7] /*v[262:263]*/, v[58:59] /*v[314:315]*/
	v_cndmask_b32_e64 v30 /*v286*/, v30 /*v286*/, 0xff61b1e6, s2
	v_cndmask_b32_e64 v33 /*v289*/, v33 /*v289*/, 0xff61b1e6, s15
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[150:157] /*v[406:413]*/, v[82:89], v[94:101] /*v[350:357]*/
	v_cndmask_b32_e64 v32 /*v288*/, v32 /*v288*/, 0xff61b1e6, s18
	v_cndmask_b32_e64 v35 /*v291*/, v35 /*v291*/, 0xff61b1e6, s13
	v_cndmask_b32_e64 v34 /*v290*/, v34 /*v290*/, 0xff61b1e6, s16
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[30:31] /*v[286:287]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[36:37] /*v[292:293]*/, v[6:7] /*v[262:263]*/, v[60:61] /*v[316:317]*/
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[32:33] /*v[288:289]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[34:35] /*v[290:291]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[110:117] /*v[366:373]*/, v[150:157] /*v[406:413]*/, v[154:161], v[110:117] /*v[366:373]*/
	v_pk_mul_f32 v[30:31] /*v[286:287]*/, v[30:31] /*v[286:287]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33] /*v[288:289]*/, v[32:33] /*v[288:289]*/, s[36:37] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v37 /*v293*/, v37 /*v293*/, 0xff61b1e6, s14
	v_cndmask_b32_e64 v36 /*v292*/, v36 /*v292*/, 0xff61b1e6, s17
	v_pk_mul_f32 v[34:35] /*v[290:291]*/, v[34:35] /*v[290:291]*/, s[36:37] op_sel_hi:[1,0]
	v_exp_f32_e32 v30 /*v286*/, v30 /*v286*/
	v_exp_f32_e32 v31 /*v287*/, v31 /*v287*/
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[334:341]*/, v[62:69] /*v[318:325]*/, v[162:169], v[78:85] /*v[334:341]*/
	v_exp_f32_e32 v32 /*v288*/, v32 /*v288*/
	v_exp_f32_e32 v33 /*v289*/, v33 /*v289*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[36:37] /*v[292:293]*/, v[36:37] /*v[292:293]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v34 /*v290*/, v34 /*v290*/
	v_exp_f32_e32 v35 /*v291*/, v35 /*v291*/
	v_pk_mul_f32 v[46:47] /*v[302:303]*/, v[6:7] /*v[262:263]*/, v[78:79] /*v[334:335]*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[126:133] /*v[382:389]*/, v[98:105], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5145
	v_pk_mul_f32 v[48:49] /*v[304:305]*/, v[6:7] /*v[262:263]*/, v[80:81] /*v[336:337]*/
	v_pk_mul_f32 v[50:51] /*v[306:307]*/, v[6:7] /*v[262:263]*/, v[82:83] /*v[338:339]*/
	v_pk_mul_f32 v[52:53] /*v[308:309]*/, v[6:7] /*v[262:263]*/, v[84:85] /*v[340:341]*/
	v_cndmask_b32_e64 v47 /*v303*/, v47 /*v303*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v46 /*v302*/, v46 /*v302*/, 0xff61b1e6, s4
	v_cndmask_b32_e64 v49 /*v305*/, v49 /*v305*/, 0xff61b1e6, s9
	v_cndmask_b32_e64 v48 /*v304*/, v48 /*v304*/, 0xff61b1e6, s12
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[110:117] /*v[366:373]*/, v[126:133] /*v[382:389]*/, v[162:169], v[110:117] /*v[366:373]*/
	s_set_vgpr_msb 0x5145
	v_pk_mul_f32 v[62:63] /*v[318:319]*/, v[6:7] /*v[262:263]*/, v[94:95] /*v[350:351]*/
	v_pk_mul_f32 v[64:65] /*v[320:321]*/, v[6:7] /*v[262:263]*/, v[96:97] /*v[352:353]*/
	v_pk_mul_f32 v[66:67] /*v[322:323]*/, v[6:7] /*v[262:263]*/, v[98:99] /*v[354:355]*/
	v_pk_mul_f32 v[68:69] /*v[324:325]*/, v[6:7] /*v[262:263]*/, v[100:101] /*v[356:357]*/
	v_cndmask_b32_e64 v51 /*v307*/, v51 /*v307*/, 0xff61b1e6, s8
	v_cndmask_b32_e64 v63 /*v319*/, v63 /*v319*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v62 /*v318*/, v62 /*v318*/, 0xff61b1e6, s6
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[102:109] /*v[358:365]*/, v[158:165] /*v[414:421]*/, v[122:129], v[102:109] /*v[358:365]*/
	s_set_vgpr_msb 0x5145
	v_pk_mul_f32 v[78:79] /*v[334:335]*/, v[6:7] /*v[262:263]*/, v[110:111] /*v[366:367]*/
	v_pk_mul_f32 v[80:81] /*v[336:337]*/, v[6:7] /*v[262:263]*/, v[112:113] /*v[368:369]*/
	v_pk_mul_f32 v[82:83] /*v[338:339]*/, v[6:7] /*v[262:263]*/, v[114:115] /*v[370:371]*/
	v_pk_mul_f32 v[84:85] /*v[340:341]*/, v[6:7] /*v[262:263]*/, v[116:117] /*v[372:373]*/
	v_cndmask_b32_e64 v50 /*v306*/, v50 /*v306*/, 0xff61b1e6, s10
	v_cndmask_b32_e64 v53 /*v309*/, v53 /*v309*/, 0xff61b1e6, s7
	v_cndmask_b32_e64 v52 /*v308*/, v52 /*v308*/, 0xff61b1e6, s11
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[118:125] /*v[374:381]*/, v[158:165] /*v[414:421]*/, v[186:193], v[118:125] /*v[374:381]*/
	v_cndmask_b32_e64 v65 /*v321*/, v65 /*v321*/, 0xff61b1e6, s21
	v_cndmask_b32_e64 v64 /*v320*/, v64 /*v320*/, 0xff61b1e6, s22
	v_cndmask_b32_e64 v67 /*v323*/, v67 /*v323*/, 0xff61b1e6, s23
	v_cndmask_b32_e64 v79 /*v335*/, v79 /*v335*/, 0xff61b1e6, s19
	s_set_vgpr_msb 0x5144
	v_or_b32_e32 v158 /*v414*/, 4, v45 /*v301*/
	v_or_b32_e32 v159 /*v415*/, 7, v45 /*v301*/
	v_or_b32_e32 v160 /*v416*/, 6, v45 /*v301*/
	s_set_vgpr_msb 0x4451
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[326:333]*/, v[142:149] /*v[398:405]*/, v[130:137], v[70:77] /*v[326:333]*/
	v_cndmask_b32_e64 v78 /*v334*/, v78 /*v334*/, 0xff61b1e6, s20
	s_set_vgpr_msb 0x5105
	v_cmp_gt_i32_e64 s24, v158 /*v414*/, v2 /*v258*/
	s_set_vgpr_msb 0x501
	v_cmp_gt_i32_e64 s25, v159 /*v415*/, v1
	s_set_vgpr_msb 0x145
	v_cmp_gt_i32_e64 s26, v160 /*v416*/, v2 /*v258*/
	v_cmp_gt_i32_e64 s30, v158 /*v414*/, v4 /*v260*/
	v_cmp_gt_i32_e64 s31, v159 /*v415*/, v3 /*v259*/
	v_cmp_gt_i32_e64 s33, v160 /*v416*/, v4 /*v260*/
	s_and_b32 s24, s53, s24
	s_and_b32 s26, s53, s26
	s_and_b32 s25, s53, s25
	s_and_b32 s30, s53, s30
	s_and_b32 s33, s53, s33
	s_and_b32 s31, s53, s31
	v_cndmask_b32_e64 v66 /*v322*/, v66 /*v322*/, 0xff61b1e6, s24
	v_cndmask_b32_e64 v69 /*v325*/, v69 /*v325*/, 0xff61b1e6, s25
	v_cndmask_b32_e64 v68 /*v324*/, v68 /*v324*/, 0xff61b1e6, s26
	v_cndmask_b32_e64 v81 /*v337*/, v81 /*v337*/, 0xff61b1e6, s27
	v_cndmask_b32_e64 v80 /*v336*/, v80 /*v336*/, 0xff61b1e6, s28
	v_cndmask_b32_e64 v83 /*v339*/, v83 /*v339*/, 0xff61b1e6, s29
	v_cndmask_b32_e64 v82 /*v338*/, v82 /*v338*/, 0xff61b1e6, s30
	v_cndmask_b32_e64 v85 /*v341*/, v85 /*v341*/, 0xff61b1e6, s31
	v_cndmask_b32_e64 v84 /*v340*/, v84 /*v340*/, 0xff61b1e6, s33
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[46:47] /*v[302:303]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[62:63] /*v[318:319]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[38:39] /*v[294:295]*/, v[70:71] /*v[326:327]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[72:73] /*v[328:329]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[342:349]*/, v[142:149] /*v[398:405]*/, v[194:201], v[86:93] /*v[342:349]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[48:49] /*v[304:305]*/, v[48:49] /*v[304:305]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[50:51] /*v[306:307]*/, v[50:51] /*v[306:307]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[52:53] /*v[308:309]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[64:65] /*v[320:321]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[68:69] /*v[324:325]*/, v[250:251] /*v[506:507]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[78:79] /*v[334:335]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4551
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[102:109] /*v[358:365]*/, v[166:173] /*v[422:429]*/, v[130:137], v[102:109] /*v[358:365]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[80:81] /*v[336:337]*/, v[80:81] /*v[336:337]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[82:83] /*v[338:339]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[84:85] /*v[340:341]*/, v[252:253] /*v[508:509]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[46:47] /*v[302:303]*/, v[46:47] /*v[302:303]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63] /*v[318:319]*/, v[62:63] /*v[318:319]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[74:75] /*v[330:331]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[30:31] /*v[286:287]*/, v[38:39] /*v[294:295]*/, v[30:31] /*v[286:287]*/
	v_pk_mul_f32 v[32:33] /*v[288:289]*/, v[40:41] /*v[296:297]*/, v[32:33] /*v[288:289]*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[118:125] /*v[374:381]*/, v[166:173] /*v[422:429]*/, v[194:201], v[118:125] /*v[374:381]*/
	v_pk_mul_f32 v[36:37] /*v[292:293]*/, v[36:37] /*v[292:293]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49] /*v[304:305]*/, v[48:49] /*v[304:305]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51] /*v[306:307]*/, v[50:51] /*v[306:307]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53] /*v[308:309]*/, v[52:53] /*v[308:309]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65] /*v[320:321]*/, v[64:65] /*v[320:321]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69] /*v[324:325]*/, v[68:69] /*v[324:325]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79] /*v[334:335]*/, v[78:79] /*v[334:335]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81] /*v[336:337]*/, v[80:81] /*v[336:337]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83] /*v[338:339]*/, v[82:83] /*v[338:339]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85] /*v[340:341]*/, v[84:85] /*v[340:341]*/, s[36:37] op_sel_hi:[1,0]
	v_exp_f32_e32 v46 /*v302*/, v46 /*v302*/
	v_exp_f32_e32 v47 /*v303*/, v47 /*v303*/
	v_exp_f32_e32 v62 /*v318*/, v62 /*v318*/
	v_exp_f32_e32 v63 /*v319*/, v63 /*v319*/
	s_set_vgpr_msb 0x5149
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[86:87] /*v[342:343]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[102:103] /*v[358:359]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[34:35] /*v[290:291]*/, v[42:43] /*v[298:299]*/, v[34:35] /*v[290:291]*/
	v_pk_mul_f32 v[30:31] /*v[286:287]*/, v[6:7] /*v[262:263]*/, v[30:31] /*v[286:287]*/
	v_pk_mul_f32 v[32:33] /*v[288:289]*/, v[6:7] /*v[262:263]*/, v[32:33] /*v[288:289]*/
	v_exp_f32_e32 v36 /*v292*/, v36 /*v292*/
	v_exp_f32_e32 v37 /*v293*/, v37 /*v293*/
	v_exp_f32_e32 v48 /*v304*/, v48 /*v304*/
	v_exp_f32_e32 v49 /*v305*/, v49 /*v305*/
	v_exp_f32_e32 v50 /*v306*/, v50 /*v306*/
	v_exp_f32_e32 v51 /*v307*/, v51 /*v307*/
	v_exp_f32_e32 v52 /*v308*/, v52 /*v308*/
	v_exp_f32_e32 v53 /*v309*/, v53 /*v309*/
	v_exp_f32_e32 v64 /*v320*/, v64 /*v320*/
	v_exp_f32_e32 v65 /*v321*/, v65 /*v321*/
	v_exp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_exp_f32_e32 v67 /*v323*/, v67 /*v323*/
	v_exp_f32_e32 v68 /*v324*/, v68 /*v324*/
	v_exp_f32_e32 v69 /*v325*/, v69 /*v325*/
	v_exp_f32_e32 v78 /*v334*/, v78 /*v334*/
	v_exp_f32_e32 v79 /*v335*/, v79 /*v335*/
	v_exp_f32_e32 v80 /*v336*/, v80 /*v336*/
	v_exp_f32_e32 v81 /*v337*/, v81 /*v337*/
	v_exp_f32_e32 v82 /*v338*/, v82 /*v338*/
	v_exp_f32_e32 v83 /*v339*/, v83 /*v339*/
	v_exp_f32_e32 v84 /*v340*/, v84 /*v340*/
	v_exp_f32_e32 v85 /*v341*/, v85 /*v341*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[76:77] /*v[332:333]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[56:57] /*v[312:313]*/, v[88:89] /*v[344:345]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[58:59] /*v[314:315]*/, v[90:91] /*v[346:347]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[92:93] /*v[348:349]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_add_f32 v[72:73] /*v[328:329]*/, v[104:105] /*v[360:361]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[106:107] /*v[362:363]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[108:109] /*v[364:365]*/, v[254:255] /*v[510:511]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[118:119] /*v[374:375]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[88:89] /*v[344:345]*/, v[120:121] /*v[376:377]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[122:123] /*v[378:379]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[124:125] /*v[380:381]*/, v[0:1] /*v[512:513]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[38:39] /*v[294:295]*/, v[54:55] /*v[310:311]*/, v[46:47] /*v[302:303]*/
	v_pk_mul_f32 v[46:47] /*v[302:303]*/, v[70:71] /*v[326:327]*/, v[62:63] /*v[318:319]*/
	v_pk_mul_f32 v[62:63] /*v[318:319]*/, v[6:7] /*v[262:263]*/, v[34:35] /*v[290:291]*/
	v_cvt_pk_bf16_f32 v34 /*v290*/, v30 /*v286*/, v31 /*v287*/
	v_cvt_pk_bf16_f32 v35 /*v291*/, v32 /*v288*/, v33 /*v289*/
	ds_load_tr16_b128 v[30:33] /*v[286:289]*/, v11 /*v267*/ offset:4352
	v_pk_mul_f32 v[36:37] /*v[292:293]*/, v[44:45] /*v[300:301]*/, v[36:37] /*v[292:293]*/
	v_pk_mul_f32 v[40:41] /*v[296:297]*/, v[56:57] /*v[312:313]*/, v[48:49] /*v[304:305]*/
	v_pk_mul_f32 v[42:43] /*v[298:299]*/, v[58:59] /*v[314:315]*/, v[50:51] /*v[306:307]*/
	v_pk_mul_f32 v[44:45] /*v[300:301]*/, v[60:61] /*v[316:317]*/, v[52:53] /*v[308:309]*/
	v_pk_mul_f32 v[48:49] /*v[304:305]*/, v[72:73] /*v[328:329]*/, v[64:65] /*v[320:321]*/
	v_pk_mul_f32 v[50:51] /*v[306:307]*/, v[74:75] /*v[330:331]*/, v[66:67] /*v[322:323]*/
	v_pk_mul_f32 v[52:53] /*v[308:309]*/, v[76:77] /*v[332:333]*/, v[68:69] /*v[324:325]*/
	v_pk_mul_f32 v[54:55] /*v[310:311]*/, v[86:87] /*v[342:343]*/, v[78:79] /*v[334:335]*/
	v_pk_mul_f32 v[56:57] /*v[312:313]*/, v[88:89] /*v[344:345]*/, v[80:81] /*v[336:337]*/
	v_pk_mul_f32 v[58:59] /*v[314:315]*/, v[90:91] /*v[346:347]*/, v[82:83] /*v[338:339]*/
	v_pk_mul_f32 v[60:61] /*v[316:317]*/, v[92:93] /*v[348:349]*/, v[84:85] /*v[340:341]*/
	v_pk_mul_f32 v[64:65] /*v[320:321]*/, v[6:7] /*v[262:263]*/, v[36:37] /*v[292:293]*/
	v_pk_mul_f32 v[38:39] /*v[294:295]*/, v[6:7] /*v[262:263]*/, v[38:39] /*v[294:295]*/
	v_pk_mul_f32 v[66:67] /*v[322:323]*/, v[6:7] /*v[262:263]*/, v[40:41] /*v[296:297]*/
	v_pk_mul_f32 v[68:69] /*v[324:325]*/, v[6:7] /*v[262:263]*/, v[42:43] /*v[298:299]*/
	v_pk_mul_f32 v[70:71] /*v[326:327]*/, v[6:7] /*v[262:263]*/, v[44:45] /*v[300:301]*/
	v_pk_mul_f32 v[40:41] /*v[296:297]*/, v[6:7] /*v[262:263]*/, v[46:47] /*v[302:303]*/
	v_pk_mul_f32 v[44:45] /*v[300:301]*/, v[6:7] /*v[262:263]*/, v[48:49] /*v[304:305]*/
	v_pk_mul_f32 v[46:47] /*v[302:303]*/, v[6:7] /*v[262:263]*/, v[50:51] /*v[306:307]*/
	v_pk_mul_f32 v[48:49] /*v[304:305]*/, v[6:7] /*v[262:263]*/, v[52:53] /*v[308:309]*/
	v_pk_mul_f32 v[50:51] /*v[306:307]*/, v[6:7] /*v[262:263]*/, v[54:55] /*v[310:311]*/
	v_pk_mul_f32 v[52:53] /*v[308:309]*/, v[6:7] /*v[262:263]*/, v[56:57] /*v[312:313]*/
	v_pk_mul_f32 v[54:55] /*v[310:311]*/, v[6:7] /*v[262:263]*/, v[58:59] /*v[314:315]*/
	v_pk_mul_f32 v[56:57] /*v[312:313]*/, v[6:7] /*v[262:263]*/, v[60:61] /*v[316:317]*/
	v_cvt_pk_bf16_f32 v36 /*v292*/, v62 /*v318*/, v63 /*v319*/
	v_cvt_pk_bf16_f32 v37 /*v293*/, v64 /*v320*/, v65 /*v321*/
	v_cvt_pk_bf16_f32 v42 /*v298*/, v38 /*v294*/, v39 /*v295*/
	v_cvt_pk_bf16_f32 v38 /*v294*/, v40 /*v296*/, v41 /*v297*/
	v_cvt_pk_bf16_f32 v39 /*v295*/, v44 /*v300*/, v45 /*v301*/
	v_cvt_pk_bf16_f32 v40 /*v296*/, v46 /*v302*/, v47 /*v303*/
	v_cvt_pk_bf16_f32 v41 /*v297*/, v48 /*v304*/, v49 /*v305*/
	v_cvt_pk_bf16_f32 v43 /*v299*/, v66 /*v322*/, v67 /*v323*/
	v_cvt_pk_bf16_f32 v44 /*v300*/, v68 /*v324*/, v69 /*v325*/
	v_cvt_pk_bf16_f32 v45 /*v301*/, v70 /*v326*/, v71 /*v327*/
	v_cvt_pk_bf16_f32 v46 /*v302*/, v50 /*v306*/, v51 /*v307*/
	v_cvt_pk_bf16_f32 v47 /*v303*/, v52 /*v308*/, v53 /*v309*/
	v_cvt_pk_bf16_f32 v48 /*v304*/, v54 /*v310*/, v55 /*v311*/
	v_cvt_pk_bf16_f32 v49 /*v305*/, v56 /*v312*/, v57 /*v313*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[242:249], v[34:41] /*v[290:297]*/, v[18:25] /*v[274:281]*/, v[242:249]
	s_cmp_lg_u64 s[34:35], 0
	v_wmma_f32_16x16x32_bf16 v[58:65], v[42:49] /*v[298:305]*/, v[18:25] /*v[274:281]*/, v[58:65]
	s_set_vgpr_msb 0x541
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v12 /*v268*/
	ds_load_tr16_b128 v[22:25] /*v[278:281]*/, v12 /*v268*/ offset:4352
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[250:257], v[34:41] /*v[290:297]*/, v[26:33] /*v[282:289]*/, v[250:257]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[42:49] /*v[298:305]*/, v[26:33] /*v[282:289]*/, v[50:57]
	s_set_vgpr_msb 0x541
	ds_load_tr16_b128 v[26:29] /*v[282:285]*/, v13 /*v269*/
	ds_load_tr16_b128 v[30:33] /*v[286:289]*/, v13 /*v269*/ offset:4352
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[234:241], v[34:41] /*v[290:297]*/, v[18:25] /*v[274:281]*/, v[234:241]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[42:49] /*v[298:305]*/, v[18:25] /*v[274:281]*/, v[42:49]
	s_set_vgpr_msb 0x541
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v14 /*v270*/
	ds_load_tr16_b128 v[22:25] /*v[278:281]*/, v14 /*v270*/ offset:4352
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[226:233], v[34:41] /*v[290:297]*/, v[26:33] /*v[282:289]*/, v[226:233]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[42:49] /*v[298:305]*/, v[26:33] /*v[282:289]*/, v[34:41]
	s_set_vgpr_msb 0x541
	ds_load_tr16_b128 v[26:29] /*v[282:285]*/, v15 /*v271*/
	ds_load_tr16_b128 v[30:33] /*v[286:289]*/, v15 /*v271*/ offset:4352
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[218:225], v[34:41] /*v[290:297]*/, v[18:25] /*v[274:281]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[42:49] /*v[298:305]*/, v[18:25] /*v[274:281]*/, v[26:33]
	s_set_vgpr_msb 0x541
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v16 /*v272*/
	ds_load_tr16_b128 v[22:25] /*v[278:281]*/, v16 /*v272*/ offset:4352
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[210:217], v[34:41] /*v[290:297]*/, v[26:33] /*v[282:289]*/, v[210:217]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[42:49] /*v[298:305]*/, v[26:33] /*v[282:289]*/, v[18:25]
	s_set_vgpr_msb 0x541
	ds_load_tr16_b128 v[26:29] /*v[282:285]*/, v17 /*v273*/
	ds_load_tr16_b128 v[30:33] /*v[286:289]*/, v17 /*v273*/ offset:4352
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[202:209], v[34:41] /*v[290:297]*/, v[18:25] /*v[274:281]*/, v[202:209]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[42:49] /*v[298:305]*/, v[18:25] /*v[274:281]*/, v[10:17]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[90:97], v[34:41] /*v[290:297]*/, v[26:33] /*v[282:289]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[42:49] /*v[298:305]*/, v[26:33] /*v[282:289]*/, v[2:9]
	s_set_vgpr_msb 0x500
	s_cbranch_scc1 .LBB0_6
.LBB0_7:
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, s52, v3 /*v515*/
	s_mul_i32 s4, s39, s54
	s_load_b64 s[0:1], s[0:1], 0x110 nv
	s_add_co_i32 s42, s42, s4
	s_mov_b32 s3, 0
	s_wait_loadcnt 0x21
	v_mul_lo_u32 v69, v1, s39
	s_mov_b32 s2, 0x800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v69, v69, s42, 7
	s_wait_loadcnt 0x1f
	v_or_b32_e32 v74, v69, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v66, 1, v1
	s_wait_kmcnt 0x0
	v_cvt_pk_bf16_f32 v67, v242, s0
	v_lshlrev_b32_e32 v74, 2, v74
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_lo_u32 v66, v66, s39
	v_cvt_pk_bf16_f32 v68, v243, s0
	v_cvt_pk_bf16_f32 v71, v244, s0
	s_wait_loadcnt 0x1e
	v_cvt_pk_bf16_f32 v79, v246, s0
	buffer_store_b16 v67, v74, s[0:3], null offen
	v_cvt_pk_bf16_f32 v81, v247, s0
	s_wait_loadcnt 0x1d
	v_cvt_pk_bf16_f32 v82, v249, s0
	v_cvt_pk_bf16_f32 v83, v250, s0
	v_add_lshl_u32 v66, s42, v66, 7
	s_wait_loadcnt 0x1c
	v_cvt_pk_bf16_f32 v87, v236, s0
	v_cvt_pk_bf16_f32 v58, v58, s0
	v_cvt_pk_bf16_f32 v59, v59, s0
	v_cvt_pk_bf16_f32 v60, v60, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v75, v66, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v70, 2, v1
	v_cvt_pk_bf16_f32 v61, v61, s0
	v_cvt_pk_bf16_f32 v62, v62, s0
	v_cvt_pk_bf16_f32 v63, v63, s0
	v_lshlrev_b32_e32 v75, 2, v75
	v_mul_lo_u32 v70, v70, s39
	v_cvt_pk_bf16_f32 v65, v65, s0
	v_cvt_pk_bf16_f32 v50, v50, s0
	v_cvt_pk_bf16_f32 v51, v51, s0
	buffer_store_b16 v68, v75, s[0:3], null offen
	v_cvt_pk_bf16_f32 v52, v52, s0
	v_cvt_pk_bf16_f32 v53, v53, s0
	v_cvt_pk_bf16_f32 v56, v56, s0
	v_add_lshl_u32 v70, s42, v70, 7
	v_cvt_pk_bf16_f32 v42, v42, s0
	v_cvt_pk_bf16_f32 v43, v43, s0
	v_cvt_pk_bf16_f32 v44, v44, s0
	v_cvt_pk_bf16_f32 v34, v34, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v76, v70, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v72, 3, v1
	v_cvt_pk_bf16_f32 v36, v36, s0
	v_cvt_pk_bf16_f32 v35, v35, s0
	v_cvt_pk_bf16_f32 v37, v37, s0
	v_lshlrev_b32_e32 v67, 2, v76
	v_mul_lo_u32 v72, v72, s39
	v_cvt_pk_bf16_f32 v26, v26, s0
	v_cvt_pk_bf16_f32 v29, v29, s0
	v_cvt_pk_bf16_f32 v30, v30, s0
	buffer_store_b16 v71, v67, s[0:3], null offen
	v_cvt_pk_bf16_f32 v27, v27, s0
	v_cvt_pk_bf16_f32 v28, v28, s0
	v_cvt_pk_bf16_f32 v18, v18, s0
	v_add_lshl_u32 v72, s42, v72, 7
	v_cvt_pk_bf16_f32 v19, v19, s0
	v_cvt_pk_bf16_f32 v21, v21, s0
	v_cvt_pk_bf16_f32 v22, v22, s0
	v_cvt_pk_bf16_f32 v23, v23, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v68, v72, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v73, 4, v1
	v_cvt_pk_bf16_f32 v12, v12, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_lshlrev_b32_e32 v68, 2, v68
	v_mul_lo_u32 v73, v73, s39
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_cvt_pk_bf16_f32 v4, v4, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_lshl_u32 v73, s42, v73, 7
	s_set_vgpr_msb 8
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v78, v73, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v77, 5, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v78, 2, v78
	v_mul_lo_u32 v76, v77, s39
	v_cvt_pk_bf16_f32 v77, v245, s0
	s_clause 0x1
	buffer_store_b16 v77, v68, s[0:3], null offen
	buffer_store_b16 v79, v78, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v79, v248, s0
	v_add_lshl_u32 v76, s42, v76, 7
	s_set_vgpr_msb 8
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v80, v76, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v71, 6, v1
	v_or_b32_e32 v76, v76, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v80, 2, v80
	v_mul_lo_u32 v71, v71, s39
	buffer_store_b16 v81, v80, s[0:3], null offen
	v_add_lshl_u32 v71, s42, v71, 7
	v_or_b32_e32 v66, v66, v0
	s_set_vgpr_msb 8
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v77, v71, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v69, v69, v0
	v_or_b32_e32 v1, 7, v1
	v_dual_lshlrev_b32 v66, 2, v66 :: v_dual_bitop2_b32 v71, v71, v0 bitop3:0x54
	v_lshlrev_b32_e32 v77, 2, v77
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b32_e32 v69, 2, v69
	v_mul_lo_u32 v1, v1, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v1, s42, v1, 7
	s_set_vgpr_msb 8
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v81, v1, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v1, v1, v0
	v_or_b32_e32 v70, v70, v0
	v_or_b32_e32 v73, v73, v0
	v_or_b32_e32 v84, 64, v69
	v_lshlrev_b32_e32 v81, 2, v81
	v_dual_lshlrev_b32 v1, 2, v1 :: v_dual_bitop2_b32 v72, v72, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshlrev_b32 v73, 2, v73 :: v_dual_lshlrev_b32 v70, 2, v70
	s_clause 0x1
	buffer_store_b16 v79, v77, s[0:3], null offen
	buffer_store_b16 v82, v81, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v72, 2, v72 :: v_dual_bitop2_b32 v82, 64, v66 bitop3:0x54
	buffer_store_b16 v83, v84, s[0:3], null offen
	v_cvt_pk_bf16_f32 v79, v251, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v83, v252, s0
	v_dual_lshlrev_b32 v76, 2, v76 :: v_dual_bitop2_b32 v85, 64, v70 bitop3:0x54
	v_cvt_pk_bf16_f32 v84, v253, s0
	v_dual_lshlrev_b32 v71, 2, v71 :: v_dual_bitop2_b32 v86, 64, v72 bitop3:0x54
	s_clause 0x1
	buffer_store_b16 v79, v82, s[0:3], null offen
	buffer_store_b16 v83, v85, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v79, v254, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v83, 64, v73
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v85, v0 /*v256*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v84, v86, s[0:3], null offen
	v_cvt_pk_bf16_f32 v82, v255, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v84, 64, v76
	s_clause 0x1
	buffer_store_b16 v79, v83, s[0:3], null offen
	buffer_store_b16 v82, v84, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v79, 64, v71
	s_wait_xcnt 0x0
	v_mov_b16_e32 v82.l, v85.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v83, v1 /*v257*/, s0
	v_or_b32_e32 v85, 64, v1
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v86, v235, s0
	v_cvt_pk_bf16_f32 v84, v234, s0
	s_clause 0x2
	buffer_store_b16 v82, v79, s[0:3], null offen
	buffer_store_b16 v83, v85, s[0:3], null offen
	buffer_store_b16 v84, v74, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_mov_b16_e32 v79.l, v86.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v83, v237, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v84, v240, s0
	v_mov_b16_e32 v82.l, v87.l
	v_cvt_pk_bf16_f32 v85, v241, s0
	s_clause 0x1
	buffer_store_b16 v79, v75, s[0:3], null offen offset:128
	buffer_store_b16 v82, v67, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v79, v238, s0
	buffer_store_b16 v83, v68, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v83.l, v84.l
	v_cvt_pk_bf16_f32 v82, v239, s0
	v_mov_b16_e32 v84.l, v85.l
	s_clause 0x1
	buffer_store_b16 v79, v78, s[0:3], null offen offset:128
	buffer_store_b16 v82, v80, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v79, v226, s0
	s_clause 0x1
	buffer_store_b16 v83, v77, s[0:3], null offen offset:128
	buffer_store_b16 v84, v81, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_or_b32_e32 v82, 0xc0, v69
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v84, v228, s0
	v_or_b32_e32 v87, 0xc0, v70
	v_cvt_pk_bf16_f32 v83, v227, s0
	v_or_b32_e32 v85, 0xc0, v66
	v_cvt_pk_bf16_f32 v86, v229, s0
	v_or_b32_e32 v88, 0xc0, v72
	s_clause 0x1
	buffer_store_b16 v79, v82, s[0:3], null offen
	buffer_store_b16 v83, v85, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v79, v230, s0
	s_clause 0x1
	buffer_store_b16 v84, v87, s[0:3], null offen
	buffer_store_b16 v86, v88, s[0:3], null offen
	s_wait_xcnt 0x2
	v_or_b32_e32 v83, 0xc0, v73
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v84, v232, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v86, 0xc0, v71
	v_cvt_pk_bf16_f32 v87, v233, s0
	v_cvt_pk_bf16_f32 v82, v231, s0
	v_or_b32_e32 v85, 0xc0, v76
	s_clause 0x1
	buffer_store_b16 v79, v83, s[0:3], null offen
	buffer_store_b16 v82, v85, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v79, 0xc0, v1
	s_wait_xcnt 0x0
	v_mov_b16_e32 v82.l, v87.l
	buffer_store_b16 v84, v86, s[0:3], null offen
	v_cvt_pk_bf16_f32 v83, v218, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v84, v219, s0
	v_cvt_pk_bf16_f32 v85, v220, s0
	buffer_store_b16 v82, v79, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v79, v221, s0
	v_mov_b16_e32 v82.l, v83.l
	v_mov_b16_e32 v83.l, v84.l
	v_mov_b16_e32 v84.l, v85.l
	v_cvt_pk_bf16_f32 v85, v222, s0
	s_clause 0x3
	buffer_store_b16 v82, v74, s[0:3], null offen offset:256
	buffer_store_b16 v83, v75, s[0:3], null offen offset:256
	buffer_store_b16 v84, v67, s[0:3], null offen offset:256
	buffer_store_b16 v79, v68, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v79, v223, s0
	v_mov_b16_e32 v82.l, v85.l
	v_cvt_pk_bf16_f32 v84, v225, s0
	v_cvt_pk_bf16_f32 v83, v224, s0
	v_or_b32_e32 v85, 0x140, v69
	v_or_b32_e32 v87, 0x140, v72
	buffer_store_b16 v82, v78, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v82, v210, s0
	s_clause 0x1
	buffer_store_b16 v79, v80, s[0:3], null offen offset:256
	buffer_store_b16 v83, v77, s[0:3], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v79, v211, s0
	s_clause 0x1
	buffer_store_b16 v84, v81, s[0:3], null offen offset:256
	buffer_store_b16 v82, v85, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 0x140, v66
	v_cvt_pk_bf16_f32 v84, v213, s0
	v_cvt_pk_bf16_f32 v83, v212, s0
	v_or_b32_e32 v85, 0x140, v70
	v_cvt_pk_bf16_f32 v86, v214, s0
	v_or_b32_e32 v88, 0x140, v73
	s_clause 0x1
	buffer_store_b16 v79, v82, s[0:3], null offen
	buffer_store_b16 v83, v85, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v79, v215, s0
	s_clause 0x1
	buffer_store_b16 v84, v87, s[0:3], null offen
	buffer_store_b16 v86, v88, s[0:3], null offen
	v_or_b32_e32 v82, 0x140, v76
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v86, v202, s0
	v_cvt_pk_bf16_f32 v83, v216, s0
	v_cvt_pk_bf16_f32 v84, v217, s0
	v_or_b32_e32 v85, 0x140, v71
	v_or_b32_e32 v87, 0x140, v1
	buffer_store_b16 v79, v82, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v79.l, v86.l
	v_cvt_pk_bf16_f32 v82, v203, s0
	s_clause 0x1
	buffer_store_b16 v83, v85, s[0:3], null offen
	buffer_store_b16 v84, v87, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v83, v204, s0
	buffer_store_b16 v79, v74, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v79.l, v82.l
	v_cvt_pk_bf16_f32 v82, v206, s0
	v_cvt_pk_bf16_f32 v84, v207, s0
	v_cvt_pk_bf16_f32 v74, v205, s0
	v_or_b32_e32 v69, 0x1c0, v69
	buffer_store_b16 v79, v75, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v75.l, v82.l
	buffer_store_b16 v83, v67, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v67.l, v84.l
	buffer_store_b16 v74, v68, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v68, v208, s0
	buffer_store_b16 v75, v78, s[0:3], null offen offset:384
	v_cvt_pk_bf16_f32 v74, v209, s0
	buffer_store_b16 v67, v80, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v67, v90, s0
	v_cvt_pk_bf16_f32 v75, v91, s0
	v_or_b32_e32 v66, 0x1c0, v66
	s_clause 0x1
	buffer_store_b16 v68, v77, s[0:3], null offen offset:384
	buffer_store_b16 v74, v81, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v68, v92, s0
	s_clause 0x1
	buffer_store_b16 v67, v69, s[0:3], null offen
	buffer_store_b16 v75, v66, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v67, 0x1c0, v70
	v_or_b32_e32 v70, 0x1c0, v72
	v_or_b32_e32 v72, 0x1c0, v73
	s_set_vgpr_msb 8
	v_or_b32_e32 v73, s41, v3 /*v515*/
	v_cvt_pk_bf16_f32 v66, v93, s0
	v_cvt_pk_bf16_f32 v69, v94, s0
	s_clause 0x1
	buffer_store_b16 v68, v67, s[0:3], null offen
	buffer_store_b16 v66, v70, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mul_lo_u32 v67, v73, s39
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v66, v95, s0
	buffer_store_b16 v69, v72, s[0:3], null offen
	v_cvt_pk_bf16_f32 v68, v96, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v69, 0x1c0, v76
	v_or_b32_e32 v70, 0x1c0, v71
	v_or_b32_e32 v71, 1, v73
	v_or_b32_e32 v1, 0x1c0, v1
	v_add_lshl_u32 v67, s42, v67, 7
	s_clause 0x1
	buffer_store_b16 v66, v69, s[0:3], null offen
	buffer_store_b16 v68, v70, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mul_lo_u32 v69, v71, s39
	s_wait_xcnt 0x0
	v_or_b32_e32 v70, 2, v73
	s_set_vgpr_msb 8
	v_or_b32_e32 v68, v67, v2 /*v514*/
	v_cvt_pk_bf16_f32 v66, v97, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v71, 3, v73
	v_mul_lo_u32 v70, v70, s39
	v_lshlrev_b32_e32 v68, 2, v68
	v_add_lshl_u32 v69, v69, s42, 7
	buffer_store_b16 v66, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v66, v71, s39
	buffer_store_b16 v58, v68, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, v69, v2 /*v514*/
	v_add_lshl_u32 v58, v70, s42, 7
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v70, 4, v73
	v_or_b32_e32 v69, v69, v0
	v_lshlrev_b32_e32 v1, 2, v1
	s_set_vgpr_msb 8
	v_or_b32_e32 v71, v58, v2 /*v514*/
	v_mul_lo_u32 v70, v70, s39
	v_add_lshl_u32 v66, v66, s42, 7
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v58, v58, v0
	buffer_store_b16 v59, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v59, 2, v71
	s_set_vgpr_msb 8
	v_or_b32_e32 v71, v66, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v58, 2, v58
	v_add_lshl_u32 v70, v70, s42, 7
	buffer_store_b16 v60, v59, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v71, 2, v71 :: v_dual_bitop2_b32 v60, 6, v73 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v74, v70, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v70, v70, v0
	v_mul_lo_u32 v60, v60, s39
	v_or_b32_e32 v72, 5, v73
	v_lshlrev_b32_e32 v74, 2, v74
	s_clause 0x1
	buffer_store_b16 v61, v71, s[0:3], null offen
	buffer_store_b16 v62, v74, s[0:3], null offen
	v_mul_lo_u32 v72, v72, s39
	v_add_lshl_u32 v60, v60, s42, 7
	s_set_vgpr_msb 8
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v61, v60, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v60, v60, v0
	v_or_b32_e32 v73, 7, v73
	v_add_lshl_u32 v72, v72, s42, 7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_lshlrev_b32 v61, 2, v61 :: v_dual_lshlrev_b32 v60, 2, v60
	v_mul_lo_u32 v73, v73, s39
	s_set_vgpr_msb 8
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v75, v72, v2 /*v514*/
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v75, 2, v75
	v_add_lshl_u32 v62, v73, s42, 7
	buffer_store_b16 v63, v75, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v63, v64, s0
	v_or_b32_e32 v64, v67, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v67, v62, v2 /*v514*/
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v67, 2, v67
	s_clause 0x1
	buffer_store_b16 v63, v61, s[0:3], null offen
	buffer_store_b16 v65, v67, s[0:3], null offen
	s_wait_xcnt 0x1
	v_dual_lshlrev_b32 v63, 2, v69 :: v_dual_lshlrev_b32 v64, 2, v64
	s_wait_xcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v65, 64, v63
	v_or_b32_e32 v73, 64, v64
	buffer_store_b16 v50, v73, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v50, v66, v0
	v_or_b32_e32 v66, 64, v58
	buffer_store_b16 v51, v65, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v51, v72, v0
	v_dual_lshlrev_b32 v50, 2, v50 :: v_dual_bitop2_b32 v0, v62, v0 bitop3:0x54
	buffer_store_b16 v52, v66, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v52, 2, v70 :: v_dual_lshlrev_b32 v51, 2, v51
	v_dual_lshlrev_b32 v0, 2, v0 :: v_dual_bitop2_b32 v69, 64, v50 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v65, 64, v51
	buffer_store_b16 v53, v69, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v53, v54, s0
	v_cvt_pk_bf16_f32 v54, v55, s0
	v_or_b32_e32 v55, 64, v52
	s_clause 0x1
	buffer_store_b16 v53, v55, s[0:3], null offen
	buffer_store_b16 v54, v65, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v53, 64, v60
	s_wait_xcnt 0x0
	v_mov_b16_e32 v54.l, v56.l
	v_cvt_pk_bf16_f32 v55, v57, s0
	v_or_b32_e32 v56, 64, v0
	s_clause 0x2
	buffer_store_b16 v54, v53, s[0:3], null offen
	buffer_store_b16 v55, v56, s[0:3], null offen
	buffer_store_b16 v42, v68, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v42, v45, s0
	v_cvt_pk_bf16_f32 v45, v48, s0
	s_clause 0x1
	buffer_store_b16 v43, v1, s[0:3], null offen offset:128
	buffer_store_b16 v44, v59, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v43, v46, s0
	v_cvt_pk_bf16_f32 v46, v49, s0
	buffer_store_b16 v42, v71, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v42.l, v45.l
	v_cvt_pk_bf16_f32 v44, v47, s0
	s_clause 0x1
	buffer_store_b16 v43, v74, s[0:3], null offen offset:128
	buffer_store_b16 v44, v75, s[0:3], null offen offset:128
	v_mov_b16_e32 v45.l, v46.l
	s_clause 0x1
	buffer_store_b16 v42, v61, s[0:3], null offen offset:128
	buffer_store_b16 v45, v67, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v42, 0xc0, v64
	v_or_b32_e32 v44, 0xc0, v58
	v_or_b32_e32 v43, 0xc0, v63
	s_wait_xcnt 0x0
	v_or_b32_e32 v45, 0xc0, v50
	s_clause 0x1
	buffer_store_b16 v34, v42, s[0:3], null offen
	buffer_store_b16 v35, v43, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v34, v38, s0
	s_clause 0x1
	buffer_store_b16 v36, v44, s[0:3], null offen
	buffer_store_b16 v37, v45, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v36, 0xc0, v52
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v37, v40, s0
	v_cvt_pk_bf16_f32 v40, v41, s0
	v_cvt_pk_bf16_f32 v35, v39, s0
	v_or_b32_e32 v38, 0xc0, v51
	v_or_b32_e32 v39, 0xc0, v60
	s_clause 0x1
	buffer_store_b16 v34, v36, s[0:3], null offen
	buffer_store_b16 v35, v38, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v34, 0xc0, v0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v35.l, v40.l
	buffer_store_b16 v37, v39, s[0:3], null offen
	s_clause 0x3
	buffer_store_b16 v35, v34, s[0:3], null offen
	buffer_store_b16 v26, v68, s[0:3], null offen offset:256
	buffer_store_b16 v27, v1, s[0:3], null offen offset:256
	buffer_store_b16 v28, v59, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e32 v26.l, v30.l
	buffer_store_b16 v29, v71, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v27, v31, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v28, v32, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v29, v33, s0
	buffer_store_b16 v26, v74, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e32 v26.l, v27.l
	v_mov_b16_e32 v27.l, v28.l
	v_mov_b16_e32 v28.l, v29.l
	v_or_b32_e32 v29, 0x140, v64
	s_clause 0x3
	buffer_store_b16 v26, v75, s[0:3], null offen offset:256
	buffer_store_b16 v27, v61, s[0:3], null offen offset:256
	buffer_store_b16 v28, v67, s[0:3], null offen offset:256
	buffer_store_b16 v18, v29, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v18, v20, s0
	v_or_b32_e32 v20, 0x140, v63
	v_or_b32_e32 v26, 0x140, v58
	v_or_b32_e32 v27, 0x140, v50
	v_or_b32_e32 v28, 0x140, v52
	v_or_b32_e32 v29, 0x140, v51
	s_clause 0x4
	buffer_store_b16 v19, v20, s[0:3], null offen
	buffer_store_b16 v18, v26, s[0:3], null offen
	buffer_store_b16 v21, v27, s[0:3], null offen
	buffer_store_b16 v22, v28, s[0:3], null offen
	buffer_store_b16 v23, v29, s[0:3], null offen
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v18, v24, s0
	v_or_b32_e32 v19, 0x140, v60
	v_cvt_pk_bf16_f32 v20, v25, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v21, 0x140, v0
	v_or_b32_e32 v0, 0x1c0, v0
	s_clause 0x3
	buffer_store_b16 v18, v19, s[0:3], null offen
	buffer_store_b16 v20, v21, s[0:3], null offen
	buffer_store_b16 v10, v68, s[0:3], null offen offset:384
	buffer_store_b16 v11, v1, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_mov_b16_e32 v10.l, v12.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v11, v14, s0
	v_cvt_pk_bf16_f32 v12, v15, s0
	v_cvt_pk_bf16_f32 v1, v13, s0
	v_cvt_pk_bf16_f32 v13, v16, s0
	buffer_store_b16 v10, v59, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v10.l, v11.l
	v_mov_b16_e32 v11.l, v12.l
	buffer_store_b16 v1, v71, s[0:3], null offen offset:384
	v_mov_b16_e32 v12.l, v13.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v17, s0
	s_clause 0x2
	buffer_store_b16 v10, v74, s[0:3], null offen offset:384
	buffer_store_b16 v11, v75, s[0:3], null offen offset:384
	buffer_store_b16 v12, v61, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_or_b32_e32 v11, 0x1c0, v63
	v_or_b32_e32 v10, 0x1c0, v64
	s_wait_xcnt 0x0
	v_or_b32_e32 v12, 0x1c0, v58
	s_clause 0x1
	buffer_store_b16 v1, v67, s[0:3], null offen offset:384
	buffer_store_b16 v2, v10, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v5, s0
	s_clause 0x1
	buffer_store_b16 v3, v11, s[0:3], null offen
	buffer_store_b16 v4, v12, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v3, 0x1c0, v50
	v_cvt_pk_bf16_f32 v2, v6, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v4, 0x1c0, v52
	v_cvt_pk_bf16_f32 v5, v7, s0
	v_or_b32_e32 v7, 0x1c0, v51
	v_cvt_pk_bf16_f32 v6, v8, s0
	v_cvt_pk_bf16_f32 v8, v9, s0
	v_or_b32_e32 v9, 0x1c0, v60
	s_clause 0x4
	buffer_store_b16 v1, v3, s[0:3], null offen
	buffer_store_b16 v2, v4, s[0:3], null offen
	buffer_store_b16 v5, v7, s[0:3], null offen
	buffer_store_b16 v6, v9, s[0:3], null offen
	buffer_store_b16 v8, v0, s[0:3], null offen
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	k_dq_0, .Lfunc_end0-k_dq_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dq_0
		.amdhsa_group_segment_fixed_size 8704
		.amdhsa_private_segment_fixed_size 0
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
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 598
		.amdhsa_next_free_sgpr 55
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

	.set .Lk_dq_0.num_vgpr, 598
	.set .Lk_dq_0.num_agpr, 0
	.set .Lk_dq_0.numbered_sgpr, 55
	.set .Lk_dq_0.num_named_barrier, 0
	.set .Lk_dq_0.private_seg_size, 0
	.set .Lk_dq_0.uses_vcc, 1
	.set .Lk_dq_0.uses_flat_scratch, 0
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
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     57
    .sgpr_spill_count: 0
    .symbol:         k_dq_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     598
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
