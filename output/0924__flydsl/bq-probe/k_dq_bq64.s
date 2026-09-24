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
	s_load_b256 s[64:71], s[0:1], 0x140 nv
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
	s_set_vgpr_msb 0xc0
	v_dual_lshrrev_b32 v56 /*v824*/, 4, v0 :: v_dual_bitop2_b32 v52 /*v820*/, 15, v0 bitop3:0x40
	s_cselect_b32 s5, ttmp9, s2
	s_cselect_b32 s2, s3, s4
	s_bfe_u32 s3, ttmp6, 0x40014
	s_lshr_b32 s4, ttmp7, 16
	s_add_co_i32 s3, s3, 1
	s_bfe_u32 s7, ttmp6, 0x40008
	s_mul_i32 s3, s4, s3
	s_mov_b32 s14, 0x200000
	s_add_co_i32 s7, s7, s3
	s_cmp_eq_u32 s6, 0
	s_clause 0x1
	s_load_b64 s[72:73], s[0:1], 0x30 nv
	s_load_b64 s[76:77], s[0:1], 0x60 nv
	s_cselect_b32 s18, s4, s7
	s_wait_kmcnt 0x0
	s_add_co_i32 s3, s65, 63
	s_mul_i32 s85, s65, s18
	s_ashr_i32 s4, s3, 31
	s_set_vgpr_msb 0xc0cc
	v_lshlrev_b32_e32 v53 /*v821*/, 3, v56 /*v824*/
	s_lshr_b32 s4, s4, 26
	s_mov_b64 s[78:79], 0x800000
	s_add_co_i32 s4, s3, s4
	s_mov_b64 s[74:75], 0x800000
	s_and_b32 s6, s4, 0xffffffc0
	s_ashr_i32 s4, s4, 6
	s_cmp_lg_u32 s3, s6
	s_load_b64 s[12:13], s[0:1], 0xc0 nv
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s3, 0
	s_cselect_b32 s3, -1, 0
	s_not_b32 s2, s2
	s_and_b32 s3, s3, s6
	s_add_co_i32 s4, s4, s2
	s_cmp_lg_u32 s3, 0
	s_sub_co_ci_u32 s2, s4, 0
	s_load_b32 s4, s[0:1], 0x160 nv
	s_lshl_b32 s82, s2, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s16, s82, s71
	s_set_vgpr_msb 0xcc0c
	v_or_b32_e32 v2, s82, v52 /*v820*/
	s_add_co_i32 s2, s16, 0x5f
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
	s_min_i32 s2, s2, s70
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s3, -1, 0
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s17, s2, s70
	s_add_co_i32 s2, s16, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s3, s2, s3
	s_ashr_i32 s3, s3, 5
	s_cmp_gt_i32 s2, -1
	s_cselect_b32 s2, s3, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s17
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s84, -1, 0
	s_and_b32 s3, s84, exec_lo
	s_cselect_b32 s2, s2, s70
	s_ashr_i32 s4, s5, 31
	s_ashr_i32 s6, s67, 31
	s_lshr_b32 s4, s4, 29
	s_lshr_b32 s6, s6, 29
	s_add_co_i32 s4, s5, s4
	s_add_co_i32 s6, s67, s6
	s_ashr_i32 s7, s4, 3
	s_and_b32 s4, s4, -8
	s_ashr_i32 s8, s6, 3
	s_and_b32 s6, s6, -8
	s_and_b32 s3, s67, 7
	s_sub_co_i32 s9, s5, s4
	s_cmp_lg_u32 s67, s6
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s67, 0
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
	s_cselect_b32 s83, s4, s5
	s_abs_i32 s3, s69
	s_abs_i32 s6, s83
	s_cvt_f32_u32 s4, s3
	s_sub_co_i32 s5, 0, s3
	s_xor_b32 s7, s83, s69
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s4, s4
	s_ashr_i32 s8, s7, 31
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s4, s4
	s_mul_i32 s5, s5, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s5, s4, s5
	s_add_co_i32 s4, s4, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s4, s6, s4
	s_mul_i32 s5, s4, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s5, s6, s5
	s_add_co_i32 s6, s4, 1
	s_sub_co_i32 s9, s5, s3
	s_cmp_ge_u32 s5, s3
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s5, s9, s5
	s_add_co_i32 s6, s4, 1
	s_cmp_ge_u32 s5, s3
	s_cselect_b32 s3, s6, s4
	s_mov_b32 s6, 0x800000
	s_xor_b32 s3, s3, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s4, s3, s8
	s_mul_i32 s9, s4, s69
	s_load_b64 s[4:5], s[0:1], 0x0 nv
	s_cmp_lg_u32 s83, s9
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s7, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s7, s9
	s_sub_co_ci_u32 s3, s3, s8
	s_lshl_b32 s10, s67, 4
	s_or_b32 s81, s82, 16
	s_mul_i32 s7, s10, s85
	s_or_b32 s80, s82, 32
	s_lshl4_add_u32 s11, s83, s7
	v_or_b32_e32 v1, s81, v52 /*v820*/
	s_set_vgpr_msb 0xccc
	v_or_b32_e32 v57 /*v825*/, s80, v52 /*v820*/
	s_set_vgpr_msb 0xcc00
	v_mad_u32 v3, v2, s10, s11
	s_or_b32 s69, s82, 48
	s_load_b64 s[8:9], s[0:1], 0x90 nv
	s_set_vgpr_msb 0xcc
	v_or_b32_e32 v58 /*v826*/, s69, v52 /*v820*/
	s_set_vgpr_msb 0xcc00
	v_mad_u32 v4, v1, s10, s11
	s_set_vgpr_msb 3
	v_mad_u32 v5, v57 /*v825*/, s10, s11
	s_mov_b32 s7, 0
	v_or_b32_e32 v3, v56 /*v824*/, v3
	v_mad_u32 v6, v58 /*v826*/, s10, s11
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	v_or_b32_e32 v4, v56 /*v824*/, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_or_b32_e32 v5, v56 /*v824*/, v5
	v_lshlrev_b32_e32 v3, 4, v3
	s_mov_b32 s15, s7
	v_or_b32_e32 v6, v56 /*v824*/, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_lshlrev_b32 v4, 4, v4 :: v_dual_lshlrev_b32 v5, 4, v5
	s_set_vgpr_msb 0x300
	s_wait_kmcnt 0x0
	s_clause 0x7
	buffer_load_b128 v[186:189], v3, s[4:7], null offen
	buffer_load_b128 v[190:193], v3, s[4:7], null offen offset:32
	buffer_load_b128 v[194:197], v3, s[4:7], null offen offset:64
	buffer_load_b128 v[198:201], v3, s[4:7], null offen offset:96
	buffer_load_b128 v[202:205], v3, s[4:7], null offen offset:128
	buffer_load_b128 v[206:209], v3, s[4:7], null offen offset:160
	buffer_load_b128 v[210:213], v3, s[4:7], null offen offset:192
	buffer_load_b128 v[214:217], v3, s[4:7], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[218:221], v3, s[8:11], null offen
	buffer_load_b128 v[222:225], v3, s[8:11], null offen offset:32
	buffer_load_b128 v[226:229], v3, s[8:11], null offen offset:64
	buffer_load_b128 v[230:233], v3, s[8:11], null offen offset:96
	buffer_load_b128 v[234:237], v3, s[8:11], null offen offset:128
	buffer_load_b128 v[238:241], v3, s[8:11], null offen offset:160
	buffer_load_b128 v[242:245], v3, s[8:11], null offen offset:192
	buffer_load_b128 v[246:249], v3, s[8:11], null offen offset:224
	s_clause 0x8
	buffer_load_b128 v[250:253], v4, s[4:7], null offen
	buffer_load_b128 v[254:257], v4, s[4:7], null offen offset:32
	s_set_vgpr_msb 64
	buffer_load_b128 v[2:5] /*v[258:261]*/, v4, s[4:7], null offen offset:64
	buffer_load_b128 v[6:9] /*v[262:265]*/, v4, s[4:7], null offen offset:96
	buffer_load_b128 v[10:13] /*v[266:269]*/, v4, s[4:7], null offen offset:128
	buffer_load_b128 v[14:17] /*v[270:273]*/, v4, s[4:7], null offen offset:160
	buffer_load_b128 v[18:21] /*v[274:277]*/, v4, s[4:7], null offen offset:192
	buffer_load_b128 v[22:25] /*v[278:281]*/, v4, s[4:7], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[26:29] /*v[282:285]*/, v4, s[8:11], null offen
	buffer_load_b128 v[30:33] /*v[286:289]*/, v4, s[8:11], null offen offset:32
	buffer_load_b128 v[34:37] /*v[290:293]*/, v4, s[8:11], null offen offset:64
	buffer_load_b128 v[38:41] /*v[294:297]*/, v4, s[8:11], null offen offset:96
	buffer_load_b128 v[42:45] /*v[298:301]*/, v4, s[8:11], null offen offset:128
	buffer_load_b128 v[46:49] /*v[302:305]*/, v4, s[8:11], null offen offset:160
	buffer_load_b128 v[50:53] /*v[306:309]*/, v4, s[8:11], null offen offset:192
	buffer_load_b128 v[54:57] /*v[310:313]*/, v4, s[8:11], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[58:61] /*v[314:317]*/, v5, s[4:7], null offen
	buffer_load_b128 v[62:65] /*v[318:321]*/, v5, s[4:7], null offen offset:32
	buffer_load_b128 v[74:77] /*v[330:333]*/, v5, s[4:7], null offen offset:64
	buffer_load_b128 v[78:81] /*v[334:337]*/, v5, s[4:7], null offen offset:96
	buffer_load_b128 v[82:85] /*v[338:341]*/, v5, s[4:7], null offen offset:128
	buffer_load_b128 v[86:89] /*v[342:345]*/, v5, s[4:7], null offen offset:160
	buffer_load_b128 v[90:93] /*v[346:349]*/, v5, s[4:7], null offen offset:192
	buffer_load_b128 v[94:97] /*v[350:353]*/, v5, s[4:7], null offen offset:224
	s_clause 0x3
	buffer_load_b128 v[98:101] /*v[354:357]*/, v5, s[8:11], null offen
	buffer_load_b128 v[102:105] /*v[358:361]*/, v5, s[8:11], null offen offset:32
	buffer_load_b128 v[106:109] /*v[362:365]*/, v5, s[8:11], null offen offset:64
	buffer_load_b128 v[110:113] /*v[366:369]*/, v5, s[8:11], null offen offset:96
	s_set_vgpr_msb 0x4000
	v_lshlrev_b32_e32 v3, 4, v6
	s_set_vgpr_msb 64
	s_clause 0x3
	buffer_load_b128 v[114:117] /*v[370:373]*/, v5, s[8:11], null offen offset:128
	buffer_load_b128 v[118:121] /*v[374:377]*/, v5, s[8:11], null offen offset:160
	buffer_load_b128 v[122:125] /*v[378:381]*/, v5, s[8:11], null offen offset:192
	buffer_load_b128 v[126:129] /*v[382:385]*/, v5, s[8:11], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[130:133] /*v[386:389]*/, v3, s[4:7], null offen
	buffer_load_b128 v[134:137] /*v[390:393]*/, v3, s[4:7], null offen offset:32
	buffer_load_b128 v[138:141] /*v[394:397]*/, v3, s[4:7], null offen offset:64
	buffer_load_b128 v[142:145] /*v[398:401]*/, v3, s[4:7], null offen offset:96
	buffer_load_b128 v[146:149] /*v[402:405]*/, v3, s[4:7], null offen offset:128
	buffer_load_b128 v[150:153] /*v[406:409]*/, v3, s[4:7], null offen offset:160
	buffer_load_b128 v[154:157] /*v[410:413]*/, v3, s[4:7], null offen offset:192
	buffer_load_b128 v[158:161] /*v[414:417]*/, v3, s[4:7], null offen offset:224
	s_wait_xcnt 0x0
	s_load_b64 s[4:5], s[0:1], 0xe8 nv
	s_mul_i32 s6, s67, s18
	s_clause 0x3
	buffer_load_b128 v[162:165] /*v[418:421]*/, v3, s[8:11], null offen
	buffer_load_b128 v[166:169] /*v[422:425]*/, v3, s[8:11], null offen offset:32
	buffer_load_b128 v[170:173] /*v[426:429]*/, v3, s[8:11], null offen offset:64
	buffer_load_b128 v[174:177] /*v[430:433]*/, v3, s[8:11], null offen offset:96
	s_add_co_i32 s6, s83, s6
	s_clause 0x3
	buffer_load_b128 v[178:181] /*v[434:437]*/, v3, s[8:11], null offen offset:128
	buffer_load_b128 v[182:185] /*v[438:441]*/, v3, s[8:11], null offen offset:160
	buffer_load_b128 v[186:189] /*v[442:445]*/, v3, s[8:11], null offen offset:192
	buffer_load_b128 v[190:193] /*v[446:449]*/, v3, s[8:11], null offen offset:224
	s_wait_xcnt 0x0
	s_mul_i32 s8, s6, s65
	s_mov_b32 s6, s14
	s_set_vgpr_msb 0x4000
	v_add_lshl_u32 v2, s8, v2, 2
	v_add_lshl_u32 v3, s8, v1, 2
	s_set_vgpr_msb 12
	v_add_lshl_u32 v4, s8, v57 /*v825*/, 2
	v_add_lshl_u32 v5, s8, v58 /*v826*/, 2
	s_set_vgpr_msb 0xcc0
	s_clause 0x3
	buffer_load_b32 v34 /*v802*/, v2, s[12:15], null offen
	buffer_load_b32 v36 /*v804*/, v3, s[12:15], null offen
	buffer_load_b32 v38 /*v806*/, v4, s[12:15], null offen
	buffer_load_b32 v40 /*v808*/, v5, s[12:15], null offen
	s_wait_kmcnt 0x0
	s_clause 0x3
	buffer_load_b32 v42 /*v810*/, v2, s[4:7], null offen
	buffer_load_b32 v44 /*v812*/, v3, s[4:7], null offen
	buffer_load_b32 v46 /*v814*/, v4, s[4:7], null offen
	buffer_load_b32 v48 /*v816*/, v5, s[4:7], null offen
	s_set_vgpr_msb 0xc000
	v_dual_lshrrev_b32 v5, 3, v0 :: v_dual_bitop2_b32 v2, 16, v0 bitop3:0x40
	s_set_vgpr_msb 0xc0
	v_or_b32_e32 v62 /*v830*/, 16, v0
	s_set_vgpr_msb 0xc030
	v_and_or_b32 v3, v0, 7, v53 /*v821*/
	v_bfe_u32 v4, v0, 3, 1
	s_set_vgpr_msb 0x30cc
	v_mad_u32_u24 v54 /*v822*/, 0x110, v52 /*v820*/, v2
	s_set_vgpr_msb 0xccc0
	v_lshlrev_b32_e32 v61 /*v829*/, 4, v5
	s_set_vgpr_msb 0xc0cc
	v_mad_u32_u24 v55 /*v823*/, 0x110, v62 /*v830*/, v2
	s_set_vgpr_msb 0xccc0
	v_mul_u32_u24_e32 v59 /*v827*/, 0x110, v3
	v_lshlrev_b32_e32 v60 /*v828*/, 4, v4
	s_lshl_b32 s5, s3, 4
	s_ashr_i32 s3, s2, 31
	s_mov_b32 s9, 1
	s_cmp_lt_i32 s2, 1
	s_mul_i32 s8, s66, s18
	s_set_vgpr_msb 0xc000
	s_cbranch_scc1 .LBB0_3
	s_lshl_b32 s10, s68, 4
	s_set_vgpr_msb 64
	v_mov_b32_e32 v242 /*v498*/, 0
	s_mul_i32 s4, s8, s10
	s_set_vgpr_msb 0x40cf
	s_wait_loadcnt 0x1
	v_dual_mov_b32 v47 /*v815*/, v46 /*v814*/ :: v_dual_mov_b32 v45 /*v813*/, v44 /*v812*/
	s_add_co_i32 s4, s5, s4
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v49 /*v817*/, v48 /*v816*/ :: v_dual_bitop2_b32 v63 /*v831*/, s4, v56 /*v824*/ bitop3:0x54
	s_set_vgpr_msb 0xcf0c
	v_mad_u32 v2, s10, v62 /*v830*/, s4
	v_mad_u32 v3, s10, v52 /*v820*/, s4
	s_set_vgpr_msb 0xcc3
	v_dual_mov_b32 v43 /*v811*/, v42 /*v810*/ :: v_dual_mov_b32 v35 /*v803*/, v34 /*v802*/
	v_dual_mov_b32 v37 /*v805*/, v36 /*v804*/ :: v_dual_mov_b32 v39 /*v807*/, v38 /*v806*/
	s_set_vgpr_msb 0xc341
	v_dual_mov_b32 v243 /*v499*/, v242 /*v498*/ :: v_dual_mov_b32 v244 /*v500*/, v242 /*v498*/
	s_set_vgpr_msb 0x410c
	v_or_b32_e32 v2, v2, v56 /*v824*/
	v_or_b32_e32 v3, v3, v56 /*v824*/
	s_set_vgpr_msb 0xc41
	v_dual_mov_b32 v245 /*v501*/, v242 /*v498*/ :: v_dual_mov_b32 v246 /*v502*/, v242 /*v498*/
	v_dual_mov_b32 v247 /*v503*/, v242 /*v498*/ :: v_dual_mov_b32 v248 /*v504*/, v242 /*v498*/
	s_set_vgpr_msb 0x4101
	v_dual_lshlrev_b32 v2, 4, v2 :: v_dual_lshlrev_b32 v3, 4, v3
	v_mov_b32_e32 v178, v242 /*v498*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v249 /*v505*/, v242 /*v498*/ :: v_dual_mov_b32 v250 /*v506*/, v242 /*v498*/
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v5, 0xc0, v2
	v_or_b32_e32 v4, 0xe0, v2
	v_or_b32_e32 v6, 0xa0, v2
	v_or_b32_e32 v7, 0x80, v2
	v_or_b32_e32 v8, 0x60, v2
	v_or_b32_e32 v9, 64, v2
	v_or_b32_e32 v10, 32, v2
	s_set_vgpr_msb 0x80
	s_clause 0x1
	buffer_load_b128 v[2:5] /*v[514:517]*/, v5, s[76:79], null offen
	buffer_load_b128 v[22:25] /*v[534:537]*/, v6, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[10:13] /*v[522:525]*/, v5, s[72:75], null offen
	buffer_load_b128 v[30:33] /*v[542:545]*/, v6, s[72:75], null offen
	s_clause 0x1
	buffer_load_b128 v[18:21] /*v[530:533]*/, v7, s[76:79], null offen
	buffer_load_b128 v[46:49] /*v[558:561]*/, v8, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[26:29] /*v[538:541]*/, v7, s[72:75], null offen
	buffer_load_b128 v[102:105] /*v[614:617]*/, v8, s[72:75], null offen
	s_clause 0x3
	buffer_load_b128 v[42:45] /*v[554:557]*/, v9, s[76:79], null offen
	buffer_load_b128 v[38:41] /*v[550:553]*/, v10, s[76:79], null offen
	buffer_load_b128 v[6:9] /*v[518:521]*/, v4, s[76:79], null offen
	buffer_load_b128 v[34:37] /*v[546:549]*/, v2, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[98:101] /*v[610:613]*/, v9, s[72:75], null offen
	buffer_load_b128 v[94:97] /*v[606:609]*/, v10, s[72:75], null offen
	s_set_vgpr_msb 0x8000
	v_or_b32_e32 v5, 0xc0, v3
	v_or_b32_e32 v6, 0xa0, v3
	s_set_vgpr_msb 0x80
	s_clause 0x1
	buffer_load_b128 v[14:17] /*v[526:529]*/, v4, s[72:75], null offen
	buffer_load_b128 v[90:93] /*v[602:605]*/, v2, s[72:75], null offen
	s_set_vgpr_msb 0x8000
	v_or_b32_e32 v4, 0x80, v3
	v_or_b32_e32 v7, 0x60, v3
	s_set_vgpr_msb 0x80
	s_clause 0x1
	buffer_load_b128 v[50:53] /*v[562:565]*/, v5, s[76:79], null offen
	buffer_load_b128 v[70:73] /*v[582:585]*/, v6, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[58:61] /*v[570:573]*/, v5, s[72:75], null offen
	buffer_load_b128 v[78:81] /*v[590:593]*/, v6, s[72:75], null offen
	s_set_vgpr_msb 0x8001
	v_dual_mov_b32 v179, v242 /*v498*/ :: v_dual_bitop2_b32 v5, 64, v3 bitop3:0x54
	v_or_b32_e32 v2, 0xe0, v3
	v_dual_mov_b32 v180, v242 /*v498*/ :: v_dual_bitop2_b32 v6, 32, v3 bitop3:0x54
	s_set_vgpr_msb 0x180
	s_clause 0x1
	buffer_load_b128 v[66:69] /*v[578:581]*/, v4, s[76:79], null offen
	buffer_load_b128 v[86:89] /*v[598:601]*/, v7, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[74:77] /*v[586:589]*/, v4, s[72:75], null offen
	buffer_load_b128 v[118:121] /*v[630:633]*/, v7, s[72:75], null offen
	s_clause 0x3
	buffer_load_b128 v[82:85] /*v[594:597]*/, v5, s[76:79], null offen
	buffer_load_b128 v[110:113] /*v[622:625]*/, v6, s[76:79], null offen
	buffer_load_b128 v[54:57] /*v[566:569]*/, v2, s[76:79], null offen
	buffer_load_b128 v[106:109] /*v[618:621]*/, v3, s[76:79], null offen
	s_clause 0x3
	buffer_load_b128 v[114:117] /*v[626:629]*/, v5, s[72:75], null offen
	buffer_load_b128 v[126:129] /*v[638:641]*/, v6, s[72:75], null offen
	buffer_load_b128 v[62:65] /*v[574:577]*/, v2, s[72:75], null offen
	buffer_load_b128 v[122:125] /*v[634:637]*/, v3, s[72:75], null offen
	s_set_vgpr_msb 0x8000
	v_lshlrev_b32_e32 v4, 1, v0
	s_set_vgpr_msb 13
	v_dual_mov_b32 v181, v242 /*v498*/ :: v_dual_bitop2_b32 v2, 32, v61 /*v829*/ bitop3:0x54
	v_dual_mov_b32 v182, v242 /*v498*/ :: v_dual_bitop2_b32 v3, 64, v60 /*v828*/ bitop3:0x54
	v_or_b32_e32 v5, 0x60, v61 /*v829*/
	v_or_b32_e32 v6, 0x80, v60 /*v828*/
	v_or_b32_e32 v7, 0xa0, v61 /*v829*/
	v_or_b32_e32 v8, 0xc0, v60 /*v828*/
	s_set_vgpr_msb 0xd00
	v_and_or_b32 v4, v4, 16, 0xe0
	s_set_vgpr_msb 0xc3
	v_dual_mov_b32 v41 /*v809*/, v40 /*v808*/ :: v_dual_add_nc_u32 v64 /*v832*/, v59 /*v827*/, v2
	v_dual_add_nc_u32 v65 /*v833*/, v59 /*v827*/, v3 :: v_dual_add_nc_u32 v66 /*v834*/, v59 /*v827*/, v5
	v_dual_add_nc_u32 v67 /*v835*/, v59 /*v827*/, v6 :: v_dual_add_nc_u32 v68 /*v836*/, v59 /*v827*/, v7
	v_add_nc_u32_e32 v69 /*v837*/, v59 /*v827*/, v8
	v_add_nc_u32_e32 v70 /*v838*/, v59 /*v827*/, v4
	s_set_vgpr_msb 0xc341
	v_dual_mov_b32 v251 /*v507*/, v242 /*v498*/ :: v_dual_mov_b32 v252 /*v508*/, v242 /*v498*/
	v_dual_mov_b32 v253 /*v509*/, v242 /*v498*/ :: v_dual_mov_b32 v254 /*v510*/, v242 /*v498*/
	v_dual_mov_b32 v255 /*v511*/, v242 /*v498*/ :: v_dual_mov_b32 v234 /*v490*/, v242 /*v498*/
	s_set_vgpr_msb 0x4181
	v_dual_mov_b32 v0 /*v512*/, v242 /*v498*/ :: v_dual_mov_b32 v1 /*v513*/, v242 /*v498*/
	s_set_vgpr_msb 0x8141
	v_dual_mov_b32 v235 /*v491*/, v242 /*v498*/ :: v_dual_mov_b32 v236 /*v492*/, v242 /*v498*/
	v_dual_mov_b32 v237 /*v493*/, v242 /*v498*/ :: v_dual_mov_b32 v238 /*v494*/, v242 /*v498*/
	v_dual_mov_b32 v239 /*v495*/, v242 /*v498*/ :: v_dual_mov_b32 v240 /*v496*/, v242 /*v498*/
	v_dual_mov_b32 v241 /*v497*/, v242 /*v498*/ :: v_dual_mov_b32 v226 /*v482*/, v242 /*v498*/
	v_dual_mov_b32 v227 /*v483*/, v242 /*v498*/ :: v_dual_mov_b32 v228 /*v484*/, v242 /*v498*/
	v_dual_mov_b32 v229 /*v485*/, v242 /*v498*/ :: v_dual_mov_b32 v230 /*v486*/, v242 /*v498*/
	v_dual_mov_b32 v231 /*v487*/, v242 /*v498*/ :: v_dual_mov_b32 v232 /*v488*/, v242 /*v498*/
	v_dual_mov_b32 v233 /*v489*/, v242 /*v498*/ :: v_dual_mov_b32 v218 /*v474*/, v242 /*v498*/
	v_dual_mov_b32 v219 /*v475*/, v242 /*v498*/ :: v_dual_mov_b32 v220 /*v476*/, v242 /*v498*/
	v_dual_mov_b32 v221 /*v477*/, v242 /*v498*/ :: v_dual_mov_b32 v222 /*v478*/, v242 /*v498*/
	v_dual_mov_b32 v223 /*v479*/, v242 /*v498*/ :: v_dual_mov_b32 v224 /*v480*/, v242 /*v498*/
	v_dual_mov_b32 v225 /*v481*/, v242 /*v498*/ :: v_dual_mov_b32 v210 /*v466*/, v242 /*v498*/
	v_dual_mov_b32 v211 /*v467*/, v242 /*v498*/ :: v_dual_mov_b32 v212 /*v468*/, v242 /*v498*/
	v_dual_mov_b32 v213 /*v469*/, v242 /*v498*/ :: v_dual_mov_b32 v214 /*v470*/, v242 /*v498*/
	v_dual_mov_b32 v215 /*v471*/, v242 /*v498*/ :: v_dual_mov_b32 v216 /*v472*/, v242 /*v498*/
	v_dual_mov_b32 v217 /*v473*/, v242 /*v498*/ :: v_dual_mov_b32 v202 /*v458*/, v242 /*v498*/
	v_dual_mov_b32 v203 /*v459*/, v242 /*v498*/ :: v_dual_mov_b32 v204 /*v460*/, v242 /*v498*/
	v_dual_mov_b32 v205 /*v461*/, v242 /*v498*/ :: v_dual_mov_b32 v206 /*v462*/, v242 /*v498*/
	v_dual_mov_b32 v207 /*v463*/, v242 /*v498*/ :: v_dual_mov_b32 v208 /*v464*/, v242 /*v498*/
	v_dual_mov_b32 v209 /*v465*/, v242 /*v498*/ :: v_dual_mov_b32 v194 /*v450*/, v242 /*v498*/
	v_dual_mov_b32 v195 /*v451*/, v242 /*v498*/ :: v_dual_mov_b32 v196 /*v452*/, v242 /*v498*/
	v_dual_mov_b32 v197 /*v453*/, v242 /*v498*/ :: v_dual_mov_b32 v198 /*v454*/, v242 /*v498*/
	v_dual_mov_b32 v199 /*v455*/, v242 /*v498*/ :: v_dual_mov_b32 v200 /*v456*/, v242 /*v498*/
	v_dual_mov_b32 v201 /*v457*/, v242 /*v498*/ :: v_dual_mov_b32 v66 /*v322*/, v242 /*v498*/
	v_dual_mov_b32 v67 /*v323*/, v242 /*v498*/ :: v_dual_mov_b32 v68 /*v324*/, v242 /*v498*/
	v_dual_mov_b32 v69 /*v325*/, v242 /*v498*/ :: v_dual_mov_b32 v70 /*v326*/, v242 /*v498*/
	v_dual_mov_b32 v71 /*v327*/, v242 /*v498*/ :: v_dual_mov_b32 v72 /*v328*/, v242 /*v498*/
	v_mov_b32_e32 v73 /*v329*/, v242 /*v498*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v183, v242 /*v498*/ :: v_dual_mov_b32 v184, v242 /*v498*/
	v_dual_mov_b32 v185, v242 /*v498*/ :: v_dual_mov_b32 v170, v242 /*v498*/
	v_dual_mov_b32 v171, v242 /*v498*/ :: v_dual_mov_b32 v172, v242 /*v498*/
	v_dual_mov_b32 v173, v242 /*v498*/ :: v_dual_mov_b32 v174, v242 /*v498*/
	v_dual_mov_b32 v175, v242 /*v498*/ :: v_dual_mov_b32 v176, v242 /*v498*/
	v_dual_mov_b32 v177, v242 /*v498*/ :: v_dual_mov_b32 v162, v242 /*v498*/
	v_dual_mov_b32 v163, v242 /*v498*/ :: v_dual_mov_b32 v164, v242 /*v498*/
	v_dual_mov_b32 v165, v242 /*v498*/ :: v_dual_mov_b32 v166, v242 /*v498*/
	v_dual_mov_b32 v167, v242 /*v498*/ :: v_dual_mov_b32 v168, v242 /*v498*/
	v_dual_mov_b32 v169, v242 /*v498*/ :: v_dual_mov_b32 v154, v242 /*v498*/
	v_dual_mov_b32 v155, v242 /*v498*/ :: v_dual_mov_b32 v156, v242 /*v498*/
	v_dual_mov_b32 v157, v242 /*v498*/ :: v_dual_mov_b32 v158, v242 /*v498*/
	v_dual_mov_b32 v159, v242 /*v498*/ :: v_dual_mov_b32 v160, v242 /*v498*/
	v_dual_mov_b32 v161, v242 /*v498*/ :: v_dual_mov_b32 v146, v242 /*v498*/
	v_dual_mov_b32 v147, v242 /*v498*/ :: v_dual_mov_b32 v148, v242 /*v498*/
	v_dual_mov_b32 v149, v242 /*v498*/ :: v_dual_mov_b32 v150, v242 /*v498*/
	v_dual_mov_b32 v151, v242 /*v498*/ :: v_dual_mov_b32 v152, v242 /*v498*/
	v_dual_mov_b32 v153, v242 /*v498*/ :: v_dual_mov_b32 v138, v242 /*v498*/
	v_dual_mov_b32 v139, v242 /*v498*/ :: v_dual_mov_b32 v140, v242 /*v498*/
	v_dual_mov_b32 v141, v242 /*v498*/ :: v_dual_mov_b32 v142, v242 /*v498*/
	v_dual_mov_b32 v143, v242 /*v498*/ :: v_dual_mov_b32 v144, v242 /*v498*/
	v_dual_mov_b32 v145, v242 /*v498*/ :: v_dual_mov_b32 v130, v242 /*v498*/
	v_dual_mov_b32 v131, v242 /*v498*/ :: v_dual_mov_b32 v132, v242 /*v498*/
	v_dual_mov_b32 v133, v242 /*v498*/ :: v_dual_mov_b32 v134, v242 /*v498*/
	v_dual_mov_b32 v135, v242 /*v498*/ :: v_dual_mov_b32 v136, v242 /*v498*/
	v_dual_mov_b32 v137, v242 /*v498*/ :: v_dual_mov_b32 v122, v242 /*v498*/
	v_dual_mov_b32 v123, v242 /*v498*/ :: v_dual_mov_b32 v124, v242 /*v498*/
	v_dual_mov_b32 v125, v242 /*v498*/ :: v_dual_mov_b32 v126, v242 /*v498*/
	v_dual_mov_b32 v127, v242 /*v498*/ :: v_dual_mov_b32 v128, v242 /*v498*/
	v_dual_mov_b32 v129, v242 /*v498*/ :: v_dual_mov_b32 v114, v242 /*v498*/
	v_dual_mov_b32 v115, v242 /*v498*/ :: v_dual_mov_b32 v116, v242 /*v498*/
	v_dual_mov_b32 v117, v242 /*v498*/ :: v_dual_mov_b32 v118, v242 /*v498*/
	v_dual_mov_b32 v119, v242 /*v498*/ :: v_dual_mov_b32 v120, v242 /*v498*/
	v_dual_mov_b32 v121, v242 /*v498*/ :: v_dual_mov_b32 v106, v242 /*v498*/
	v_dual_mov_b32 v107, v242 /*v498*/ :: v_dual_mov_b32 v108, v242 /*v498*/
	v_dual_mov_b32 v109, v242 /*v498*/ :: v_dual_mov_b32 v110, v242 /*v498*/
	v_dual_mov_b32 v111, v242 /*v498*/ :: v_dual_mov_b32 v112, v242 /*v498*/
	v_dual_mov_b32 v113, v242 /*v498*/ :: v_dual_mov_b32 v98, v242 /*v498*/
	v_dual_mov_b32 v99, v242 /*v498*/ :: v_dual_mov_b32 v100, v242 /*v498*/
	v_dual_mov_b32 v101, v242 /*v498*/ :: v_dual_mov_b32 v102, v242 /*v498*/
	v_dual_mov_b32 v103, v242 /*v498*/ :: v_dual_mov_b32 v104, v242 /*v498*/
	v_dual_mov_b32 v105, v242 /*v498*/ :: v_dual_mov_b32 v90, v242 /*v498*/
	v_dual_mov_b32 v91, v242 /*v498*/ :: v_dual_mov_b32 v92, v242 /*v498*/
	v_dual_mov_b32 v93, v242 /*v498*/ :: v_dual_mov_b32 v94, v242 /*v498*/
	v_dual_mov_b32 v95, v242 /*v498*/ :: v_dual_mov_b32 v96, v242 /*v498*/
	v_dual_mov_b32 v97, v242 /*v498*/ :: v_dual_mov_b32 v82, v242 /*v498*/
	v_dual_mov_b32 v83, v242 /*v498*/ :: v_dual_mov_b32 v84, v242 /*v498*/
	v_dual_mov_b32 v85, v242 /*v498*/ :: v_dual_mov_b32 v86, v242 /*v498*/
	v_dual_mov_b32 v87, v242 /*v498*/ :: v_dual_mov_b32 v88, v242 /*v498*/
	v_dual_mov_b32 v89, v242 /*v498*/ :: v_dual_mov_b32 v74, v242 /*v498*/
	v_dual_mov_b32 v75, v242 /*v498*/ :: v_dual_mov_b32 v76, v242 /*v498*/
	v_dual_mov_b32 v77, v242 /*v498*/ :: v_dual_mov_b32 v78, v242 /*v498*/
	v_dual_mov_b32 v79, v242 /*v498*/ :: v_dual_mov_b32 v80, v242 /*v498*/
	v_dual_mov_b32 v81, v242 /*v498*/ :: v_dual_mov_b32 v66, v242 /*v498*/
	v_dual_mov_b32 v67, v242 /*v498*/ :: v_dual_mov_b32 v68, v242 /*v498*/
	v_dual_mov_b32 v69, v242 /*v498*/ :: v_dual_mov_b32 v70, v242 /*v498*/
	v_dual_mov_b32 v71, v242 /*v498*/ :: v_dual_mov_b32 v72, v242 /*v498*/
	v_dual_mov_b32 v73, v242 /*v498*/ :: v_dual_mov_b32 v58, v242 /*v498*/
	v_dual_mov_b32 v59, v242 /*v498*/ :: v_dual_mov_b32 v60, v242 /*v498*/
	v_dual_mov_b32 v61, v242 /*v498*/ :: v_dual_mov_b32 v62, v242 /*v498*/
	v_dual_mov_b32 v63, v242 /*v498*/ :: v_dual_mov_b32 v64, v242 /*v498*/
	v_dual_mov_b32 v65, v242 /*v498*/ :: v_dual_mov_b32 v50, v242 /*v498*/
	v_dual_mov_b32 v51, v242 /*v498*/ :: v_dual_mov_b32 v52, v242 /*v498*/
	v_dual_mov_b32 v53, v242 /*v498*/ :: v_dual_mov_b32 v54, v242 /*v498*/
	v_dual_mov_b32 v55, v242 /*v498*/ :: v_dual_mov_b32 v56, v242 /*v498*/
	v_dual_mov_b32 v57, v242 /*v498*/ :: v_dual_mov_b32 v42, v242 /*v498*/
	v_dual_mov_b32 v43, v242 /*v498*/ :: v_dual_mov_b32 v44, v242 /*v498*/
	v_dual_mov_b32 v45, v242 /*v498*/ :: v_dual_mov_b32 v46, v242 /*v498*/
	v_dual_mov_b32 v47, v242 /*v498*/ :: v_dual_mov_b32 v48, v242 /*v498*/
	v_dual_mov_b32 v49, v242 /*v498*/ :: v_dual_mov_b32 v34, v242 /*v498*/
	v_dual_mov_b32 v35, v242 /*v498*/ :: v_dual_mov_b32 v36, v242 /*v498*/
	v_dual_mov_b32 v37, v242 /*v498*/ :: v_dual_mov_b32 v38, v242 /*v498*/
	v_dual_mov_b32 v39, v242 /*v498*/ :: v_dual_mov_b32 v40, v242 /*v498*/
	v_dual_mov_b32 v41, v242 /*v498*/ :: v_dual_mov_b32 v26, v242 /*v498*/
	v_dual_mov_b32 v27, v242 /*v498*/ :: v_dual_mov_b32 v28, v242 /*v498*/
	v_dual_mov_b32 v29, v242 /*v498*/ :: v_dual_mov_b32 v30, v242 /*v498*/
	v_dual_mov_b32 v31, v242 /*v498*/ :: v_dual_mov_b32 v32, v242 /*v498*/
	v_dual_mov_b32 v33, v242 /*v498*/ :: v_dual_mov_b32 v18, v242 /*v498*/
	v_dual_mov_b32 v19, v242 /*v498*/ :: v_dual_mov_b32 v20, v242 /*v498*/
	v_dual_mov_b32 v21, v242 /*v498*/ :: v_dual_mov_b32 v22, v242 /*v498*/
	v_dual_mov_b32 v23, v242 /*v498*/ :: v_dual_mov_b32 v24, v242 /*v498*/
	v_dual_mov_b32 v25, v242 /*v498*/ :: v_dual_mov_b32 v10, v242 /*v498*/
	v_dual_mov_b32 v11, v242 /*v498*/ :: v_dual_mov_b32 v12, v242 /*v498*/
	v_dual_mov_b32 v13, v242 /*v498*/ :: v_dual_mov_b32 v14, v242 /*v498*/
	v_dual_mov_b32 v15, v242 /*v498*/ :: v_dual_mov_b32 v16, v242 /*v498*/
	v_dual_mov_b32 v17, v242 /*v498*/ :: v_dual_mov_b32 v2, v242 /*v498*/
	v_dual_mov_b32 v3, v242 /*v498*/ :: v_dual_mov_b32 v4, v242 /*v498*/
	v_dual_mov_b32 v5, v242 /*v498*/ :: v_dual_mov_b32 v6, v242 /*v498*/
	v_dual_mov_b32 v7, v242 /*v498*/ :: v_dual_mov_b32 v8, v242 /*v498*/
	v_mov_b32_e32 v9, v242 /*v498*/
	s_add_co_i32 s11, s2, 0x7ffffff
	s_mov_b32 s65, s64
	s_mov_b32 s4, 0x3fb8aa3b
	s_mov_b64 s[6:7], s[2:3]
	s_set_vgpr_msb 0x100
.LBB0_2:
	s_cmp_lt_i32 s9, s2
	s_set_vgpr_msb 11
	s_wait_loadcnt 0x0
	ds_store_b128 v54 /*v822*/, v[122:125] /*v[634:637]*/
	ds_store_b128 v54 /*v822*/, v[126:129] /*v[638:641]*/ offset:32
	s_cselect_b32 s3, s9, s11
	ds_store_b128 v54 /*v822*/, v[114:117] /*v[626:629]*/ offset:64
	ds_store_b128 v54 /*v822*/, v[118:121] /*v[630:633]*/ offset:96
	ds_store_b128 v54 /*v822*/, v[74:77] /*v[586:589]*/ offset:128
	ds_store_b128 v54 /*v822*/, v[78:81] /*v[590:593]*/ offset:160
	s_lshl_b32 s3, s3, 5
	s_set_vgpr_msb 0xb8f
	v_dual_add_nc_u32 v138 /*v650*/, v59 /*v827*/, v60 /*v828*/ :: v_dual_bitop2_b32 v131 /*v643*/, s3, v62 /*v830*/ bitop3:0x54
	v_or_b32_e32 v130 /*v642*/, s3, v52 /*v820*/
	s_set_vgpr_msb 0x8f0b
	ds_store_b128 v54 /*v822*/, v[58:61] /*v[570:573]*/ offset:192
	ds_store_b128 v54 /*v822*/, v[62:65] /*v[574:577]*/ offset:224
	ds_store_b128 v55 /*v823*/, v[90:93] /*v[602:605]*/
	ds_store_b128 v55 /*v823*/, v[94:97] /*v[606:609]*/ offset:32
	s_set_vgpr_msb 0xbc2
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[122:129] /*v[634:641]*/, v[186:193], 0
	s_set_vgpr_msb 0xc282
	v_mul_lo_u32 v131 /*v643*/, v131 /*v643*/, s10
	v_mul_lo_u32 v130 /*v642*/, v130 /*v642*/, s10
	s_set_vgpr_msb 0x82c0
	v_mov_b64_e32 v[50:51] /*v[818:819]*/, s[64:65]
	s_add_nc_u64 s[6:7], s[6:7], -1
	s_add_co_i32 s9, s9, 1
	s_cmp_lg_u64 s[6:7], 0
	s_set_vgpr_msb 0xc08e
	v_add_lshl_u32 v131 /*v643*/, v131 /*v643*/, v63 /*v831*/, 4
	v_add_lshl_u32 v130 /*v642*/, v130 /*v642*/, v63 /*v831*/, 4
	s_set_vgpr_msb 0x8ec2
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[122:129] /*v[634:641]*/, v[250:257], 0
	s_set_vgpr_msb 0xc28a
	v_or_b32_e32 v140 /*v652*/, 32, v131 /*v643*/
	v_or_b32_e32 v132 /*v644*/, 32, v130 /*v642*/
	v_or_b32_e32 v133 /*v645*/, 64, v130 /*v642*/
	v_or_b32_e32 v134 /*v646*/, 0x60, v130 /*v642*/
	v_or_b32_e32 v135 /*v647*/, 0x80, v130 /*v642*/
	v_or_b32_e32 v136 /*v648*/, 0xa0, v130 /*v642*/
	v_or_b32_e32 v137 /*v649*/, 0xc0, v130 /*v642*/
	v_or_b32_e32 v139 /*v651*/, 0xe0, v130 /*v642*/
	v_or_b32_e32 v141 /*v653*/, 64, v131 /*v643*/
	v_or_b32_e32 v142 /*v654*/, 0x60, v131 /*v643*/
	v_or_b32_e32 v143 /*v655*/, 0x80, v131 /*v643*/
	v_or_b32_e32 v144 /*v656*/, 0xa0, v131 /*v643*/
	v_or_b32_e32 v145 /*v657*/, 0xc0, v131 /*v643*/
	v_or_b32_e32 v146 /*v658*/, 0xe0, v131 /*v643*/
	s_clause 0x1
	buffer_load_b128 v[186:189] /*v[698:701]*/, v130 /*v642*/, s[72:75], null offen
	buffer_load_b128 v[162:165] /*v[674:677]*/, v132 /*v644*/, s[72:75], null offen
	s_clause 0x1
	buffer_load_b128 v[198:201] /*v[710:713]*/, v130 /*v642*/, s[76:79], null offen
	buffer_load_b128 v[166:169] /*v[678:681]*/, v132 /*v644*/, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[170:173] /*v[682:685]*/, v133 /*v645*/, s[72:75], null offen
	buffer_load_b128 v[174:177] /*v[686:689]*/, v134 /*v646*/, s[72:75], null offen
	s_clause 0x1
	buffer_load_b128 v[178:181] /*v[690:693]*/, v133 /*v645*/, s[76:79], null offen
	buffer_load_b128 v[182:185] /*v[694:697]*/, v134 /*v646*/, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[190:193] /*v[702:705]*/, v135 /*v647*/, s[72:75], null offen
	buffer_load_b128 v[194:197] /*v[706:709]*/, v136 /*v648*/, s[72:75], null offen
	s_clause 0x1
	buffer_load_b128 v[202:205] /*v[714:717]*/, v135 /*v647*/, s[76:79], null offen
	buffer_load_b128 v[206:209] /*v[718:721]*/, v136 /*v648*/, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[210:213] /*v[722:725]*/, v137 /*v649*/, s[72:75], null offen
	buffer_load_b128 v[214:217] /*v[726:729]*/, v139 /*v651*/, s[72:75], null offen
	s_clause 0x1
	buffer_load_b128 v[218:221] /*v[730:733]*/, v137 /*v649*/, s[76:79], null offen
	buffer_load_b128 v[222:225] /*v[734:737]*/, v139 /*v651*/, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[254:257] /*v[766:769]*/, v131 /*v643*/, s[72:75], null offen
	buffer_load_b128 v[226:229] /*v[738:741]*/, v140 /*v652*/, s[72:75], null offen
	s_set_vgpr_msb 0x8ac2
	s_clause 0x2
	buffer_load_b128 v[10:13] /*v[778:781]*/, v131 /*v643*/, s[76:79], null offen
	s_set_vgpr_msb 0xc282
	buffer_load_b128 v[230:233] /*v[742:745]*/, v140 /*v652*/, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[234:237] /*v[746:749]*/, v141 /*v653*/, s[72:75], null offen
	buffer_load_b128 v[238:241] /*v[750:753]*/, v142 /*v654*/, s[72:75], null offen
	s_clause 0x1
	buffer_load_b128 v[242:245] /*v[754:757]*/, v141 /*v653*/, s[76:79], null offen
	buffer_load_b128 v[246:249] /*v[758:761]*/, v142 /*v654*/, s[76:79], null offen
	s_clause 0x2
	buffer_load_b128 v[250:253] /*v[762:765]*/, v143 /*v655*/, s[72:75], null offen
	s_set_vgpr_msb 0x82c6
	buffer_load_b128 v[2:5] /*v[770:773]*/, v144 /*v656*/, s[72:75], null offen
	s_clause 0x1
	buffer_load_b128 v[6:9] /*v[774:777]*/, v143 /*v655*/, s[76:79], null offen
	buffer_load_b128 v[14:17] /*v[782:785]*/, v144 /*v656*/, s[76:79], null offen
	s_clause 0x1
	buffer_load_b128 v[18:21] /*v[786:789]*/, v145 /*v657*/, s[72:75], null offen
	buffer_load_b128 v[22:25] /*v[790:793]*/, v146 /*v658*/, s[72:75], null offen
	s_clause 0x1
	buffer_load_b128 v[26:29] /*v[794:797]*/, v145 /*v657*/, s[76:79], null offen
	buffer_load_b128 v[30:33] /*v[798:801]*/, v146 /*v658*/, s[76:79], null offen
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[122:129] /*v[634:641]*/, v[58:65] /*v[314:321]*/, 0
	s_set_vgpr_msb 0xc60b
	ds_store_b128 v55 /*v823*/, v[98:101] /*v[610:613]*/ offset:64
	ds_store_b128 v55 /*v823*/, v[102:105] /*v[614:617]*/ offset:96
	ds_store_b128 v55 /*v823*/, v[26:29] /*v[538:541]*/ offset:128
	ds_store_b128 v55 /*v823*/, v[30:33] /*v[542:545]*/ offset:160
	ds_store_b128 v55 /*v823*/, v[10:13] /*v[522:525]*/ offset:192
	ds_store_b128 v55 /*v823*/, v[14:17] /*v[526:529]*/ offset:224
	s_set_vgpr_msb 0xbc2
	ds_load_tr16_b128 v[120:123] /*v[888:891]*/, v138 /*v650*/
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v138 /*v650*/ offset:4352
	s_set_vgpr_msb 0xc283
	ds_load_tr16_b128 v[154:157] /*v[666:669]*/, v64 /*v832*/
	ds_load_tr16_b128 v[158:161] /*v[670:673]*/, v64 /*v832*/ offset:4352
	ds_load_tr16_b128 v[146:149] /*v[658:661]*/, v65 /*v833*/
	ds_load_tr16_b128 v[150:153] /*v[662:665]*/, v65 /*v833*/ offset:4352
	ds_load_tr16_b128 v[138:141] /*v[650:653]*/, v66 /*v834*/
	ds_load_tr16_b128 v[142:145] /*v[654:657]*/, v66 /*v834*/ offset:4352
	s_set_vgpr_msb 0x8386
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[122:129] /*v[634:641]*/, v[130:137] /*v[386:393]*/, 0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[90:97] /*v[602:609]*/, v[130:137] /*v[386:393]*/, 0
	s_set_vgpr_msb 0x86c2
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[90:97] /*v[602:609]*/, v[186:193], 0
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[90:97] /*v[602:609]*/, v[250:257], 0
	s_set_vgpr_msb 0xc2c6
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[90:97] /*v[602:609]*/, v[58:65] /*v[314:321]*/, 0
	s_set_vgpr_msb 0xc6a6
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[98:105] /*v[610:617]*/, v[138:145] /*v[394:401]*/, v[122:129] /*v[634:641]*/
	s_set_vgpr_msb 0xa6f2
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[98:105] /*v[610:617]*/, v[194:201], v[96:103] /*v[864:871]*/
	s_set_vgpr_msb 0xf2f6
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[98:105] /*v[610:617]*/, v[2:9] /*v[258:265]*/, v[104:111] /*v[872:879]*/
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[98:105] /*v[610:617]*/, v[74:81] /*v[330:337]*/, v[112:119] /*v[880:887]*/
	s_set_vgpr_msb 0xf682
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[106:113] /*v[618:625]*/, v[218:225], 0
	s_set_vgpr_msb 0x82a6
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[114:121] /*v[626:633]*/, v[138:145] /*v[394:401]*/, v[130:137] /*v[642:649]*/
	s_set_vgpr_msb 0xa6f2
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[114:121] /*v[626:633]*/, v[194:201], v[72:79] /*v[840:847]*/
	s_set_vgpr_msb 0xf2f6
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[114:121] /*v[626:633]*/, v[2:9] /*v[258:265]*/, v[80:87] /*v[848:855]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[114:121] /*v[626:633]*/, v[74:81] /*v[330:337]*/, v[88:95] /*v[856:863]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[106:113] /*v[618:625]*/, v[26:33] /*v[282:289]*/, 0
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[106:113] /*v[618:625]*/, v[98:105] /*v[354:361]*/, 0
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[106:113] /*v[618:625]*/, v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0xf682
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[34:41] /*v[546:553]*/, v[218:225], 0
	s_set_vgpr_msb 0x82c6
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[34:41] /*v[546:553]*/, v[26:33] /*v[282:289]*/, 0
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[34:41] /*v[546:553]*/, v[98:105] /*v[354:361]*/, 0
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[34:41] /*v[546:553]*/, v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0xc6a2
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[82:89] /*v[594:601]*/, v[226:233], v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[74:81] /*v[586:593]*/, v[146:153] /*v[402:409]*/, v[130:137] /*v[642:649]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[26:33] /*v[538:545]*/, v[146:153] /*v[402:409]*/, v[122:129] /*v[634:641]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[42:49] /*v[554:561]*/, v[226:233], v[106:113] /*v[618:625]*/
	s_set_vgpr_msb 0xa2f6
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[42:49] /*v[554:561]*/, v[34:41] /*v[290:297]*/, v[152:159] /*v[920:927]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[42:49] /*v[554:561]*/, v[106:113] /*v[362:369]*/, v[160:167] /*v[928:935]*/
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[42:49] /*v[554:561]*/, v[170:177] /*v[426:433]*/, v[168:175] /*v[936:943]*/
	s_set_vgpr_msb 0xf6f2
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[74:81] /*v[586:593]*/, v[202:209], v[72:79] /*v[840:847]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[66:73] /*v[578:585]*/, v[234:241], v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xa2f6
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[74:81] /*v[586:593]*/, v[10:17] /*v[266:273]*/, v[80:87] /*v[848:855]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[74:81] /*v[586:593]*/, v[82:89] /*v[338:345]*/, v[88:95] /*v[856:863]*/
	s_set_vgpr_msb 0xf6f2
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[26:33] /*v[538:545]*/, v[202:209], v[96:103] /*v[864:871]*/
	s_set_vgpr_msb 0xf2f6
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[26:33] /*v[538:545]*/, v[10:17] /*v[266:273]*/, v[104:111] /*v[872:879]*/
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[26:33] /*v[538:545]*/, v[82:89] /*v[338:345]*/, v[112:119] /*v[880:887]*/
	s_set_vgpr_msb 0xf6a6
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[58:65] /*v[570:577]*/, v[154:161] /*v[410:417]*/, v[130:137] /*v[642:649]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa68b
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[130:131] /*v[642:643]*/, v[50:51] /*v[818:819]*/, v[130:131] /*v[642:643]*/
	s_set_vgpr_msb 0x8ba6
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[10:17] /*v[522:529]*/, v[154:161] /*v[410:417]*/, v[122:129] /*v[634:641]*/
	s_set_vgpr_msb 0xa68b
	v_pk_mul_f32 v[132:133] /*v[644:645]*/, v[50:51] /*v[818:819]*/, v[132:133] /*v[644:645]*/
	v_pk_mul_f32 v[134:135] /*v[646:647]*/, v[50:51] /*v[818:819]*/, v[134:135] /*v[646:647]*/
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[50:51] /*v[818:819]*/, v[136:137] /*v[648:649]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[40:41] /*v[808:809]*/, v[130:131] /*v[642:643]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[40:41] /*v[808:809]*/, v[132:133] /*v[644:645]*/ neg_lo:[1,0] neg_hi:[1,0]
	v_pk_add_f32 v[134:135] /*v[646:647]*/, v[40:41] /*v[808:809]*/, v[134:135] /*v[646:647]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x8ba2
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[18:25] /*v[530:537]*/, v[234:241], v[106:113] /*v[618:625]*/
	s_set_vgpr_msb 0xa28b
	v_pk_mul_f32 v[122:123] /*v[634:635]*/, v[50:51] /*v[818:819]*/, v[122:123] /*v[634:635]*/
	v_pk_mul_f32 v[124:125] /*v[636:637]*/, v[50:51] /*v[818:819]*/, v[124:125] /*v[636:637]*/
	v_pk_mul_f32 v[126:127] /*v[638:639]*/, v[50:51] /*v[818:819]*/, v[126:127] /*v[638:639]*/
	v_pk_mul_f32 v[128:129] /*v[640:641]*/, v[50:51] /*v[818:819]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[136:137] /*v[648:649]*/, v[40:41] /*v[808:809]*/, v[136:137] /*v[648:649]*/ neg_lo:[1,0] neg_hi:[1,0]
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[40:41] /*v[808:809]*/, v[122:123] /*v[634:635]*/ neg_lo:[1,0] neg_hi:[1,0]
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[40:41] /*v[808:809]*/, v[124:125] /*v[636:637]*/ neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x8bf6
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[18:25] /*v[530:537]*/, v[42:49] /*v[298:305]*/, v[152:159] /*v[920:927]*/
	s_set_vgpr_msb 0xf68e
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[126:127] /*v[638:639]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[128:129] /*v[640:641]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[130:131] /*v[642:643]*/, v[130:131] /*v[642:643]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133] /*v[644:645]*/, v[132:133] /*v[644:645]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135] /*v[646:647]*/, v[134:135] /*v[646:647]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91] /*v[602:603]*/, v[136:137] /*v[648:649]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93] /*v[604:605]*/, v[122:123] /*v[634:635]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8ef6
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[18:25] /*v[530:537]*/, v[114:121] /*v[370:377]*/, v[160:167] /*v[928:935]*/
	s_set_vgpr_msb 0xf682
	v_pk_mul_f32 v[94:95] /*v[606:607]*/, v[124:125] /*v[636:637]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97] /*v[608:609]*/, v[126:127] /*v[638:639]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123] /*v[634:635]*/, v[128:129] /*v[640:641]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x82f6
	v_exp_f32_e32 v176 /*v944*/, v130 /*v642*/
	v_exp_f32_e32 v177 /*v945*/, v131 /*v643*/
	v_exp_f32_e32 v178 /*v946*/, v132 /*v644*/
	v_exp_f32_e32 v179 /*v947*/, v133 /*v645*/
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[18:25] /*v[530:537]*/, v[178:185] /*v[434:441]*/, v[168:175] /*v[936:943]*/
	v_exp_f32_e32 v180 /*v948*/, v134 /*v646*/
	v_exp_f32_e32 v181 /*v949*/, v135 /*v647*/
	v_exp_f32_e32 v182 /*v950*/, v90 /*v602*/
	v_exp_f32_e32 v183 /*v951*/, v91 /*v603*/
	v_exp_f32_e32 v184 /*v952*/, v92 /*v604*/
	v_exp_f32_e32 v185 /*v953*/, v93 /*v605*/
	v_exp_f32_e32 v186 /*v954*/, v94 /*v606*/
	s_set_vgpr_msb 0xf6f2
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[58:65] /*v[570:577]*/, v[210:217], v[72:79] /*v[840:847]*/
	v_exp_f32_e32 v187 /*v955*/, v95 /*v607*/
	v_exp_f32_e32 v188 /*v956*/, v96 /*v608*/
	v_exp_f32_e32 v189 /*v957*/, v97 /*v609*/
	v_exp_f32_e32 v190 /*v958*/, v122 /*v634*/
	v_exp_f32_e32 v191 /*v959*/, v123 /*v635*/
	s_set_vgpr_msb 0xf283
	ds_load_tr16_b128 v[130:133] /*v[642:645]*/, v67 /*v835*/
	ds_load_tr16_b128 v[134:137] /*v[646:649]*/, v67 /*v835*/ offset:4352
	ds_load_tr16_b128 v[122:125] /*v[634:637]*/, v68 /*v836*/
	ds_load_tr16_b128 v[126:129] /*v[638:641]*/, v68 /*v836*/ offset:4352
	ds_load_tr16_b128 v[114:117] /*v[626:629]*/, v69 /*v837*/
	ds_load_tr16_b128 v[118:121] /*v[630:633]*/, v69 /*v837*/ offset:4352
	ds_load_tr16_b128 v[90:93] /*v[602:605]*/, v70 /*v838*/
	ds_load_tr16_b128 v[94:97] /*v[606:609]*/, v70 /*v838*/ offset:4352
	s_set_vgpr_msb 0x83a2
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[50:57] /*v[562:569]*/, v[242:249], v[98:105] /*v[610:617]*/
	s_set_vgpr_msb 0xa2f6
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[58:65] /*v[570:577]*/, v[18:25] /*v[274:281]*/, v[80:87] /*v[848:855]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xf68f
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[50:51] /*v[818:819]*/, v[80:81] /*v[848:849]*/
	s_set_vgpr_msb 0x8ff6
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[58:65] /*v[570:577]*/, v[90:97] /*v[346:353]*/, v[88:95] /*v[856:863]*/
	s_set_vgpr_msb 0xf68f
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[50:51] /*v[818:819]*/, v[82:83] /*v[850:851]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[50:51] /*v[818:819]*/, v[84:85] /*v[852:853]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[50:51] /*v[818:819]*/, v[86:87] /*v[854:855]*/
	s_set_vgpr_msb 0x8f8e
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ef2
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[10:17] /*v[522:529]*/, v[210:217], v[96:103] /*v[864:871]*/
	s_set_vgpr_msb 0xf28f
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[50:51] /*v[818:819]*/, v[88:89] /*v[856:857]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[50:51] /*v[818:819]*/, v[90:91] /*v[858:859]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[50:51] /*v[818:819]*/, v[92:93] /*v[860:861]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[50:51] /*v[818:819]*/, v[94:95] /*v[862:863]*/
	s_set_vgpr_msb 0x8f8e
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ef6
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[10:17] /*v[522:529]*/, v[18:25] /*v[274:281]*/, v[104:111] /*v[872:879]*/
	s_set_vgpr_msb 0xf68f
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[50:51] /*v[818:819]*/, v[96:97] /*v[864:865]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[50:51] /*v[818:819]*/, v[98:99] /*v[866:867]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[50:51] /*v[818:819]*/, v[100:101] /*v[868:869]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[50:51] /*v[818:819]*/, v[102:103] /*v[870:871]*/
	s_set_vgpr_msb 0x8f8e
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ef6
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[10:17] /*v[522:529]*/, v[90:97] /*v[346:353]*/, v[112:119] /*v[880:887]*/
	s_set_vgpr_msb 0xf68f
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[50:51] /*v[818:819]*/, v[104:105] /*v[872:873]*/
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[50:51] /*v[818:819]*/, v[106:107] /*v[874:875]*/
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[50:51] /*v[818:819]*/, v[108:109] /*v[876:877]*/
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[50:51] /*v[818:819]*/, v[110:111] /*v[878:879]*/
	s_set_vgpr_msb 0x8f8e
	v_pk_add_f32 v[10:11] /*v[522:523]*/, v[98:99] /*v[610:611]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13] /*v[524:525]*/, v[100:101] /*v[612:613]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[14:15] /*v[526:527]*/, v[102:103] /*v[614:615]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ef6
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[82:89] /*v[594:601]*/, v[34:41] /*v[290:297]*/, v[128:135] /*v[896:903]*/
	s_set_vgpr_msb 0xf68e
	v_pk_add_f32 v[16:17] /*v[528:529]*/, v[104:105] /*v[616:617]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e8f
	v_pk_mul_f32 v[98:99] /*v[610:611]*/, v[50:51] /*v[818:819]*/, v[112:113] /*v[880:881]*/
	v_pk_mul_f32 v[100:101] /*v[612:613]*/, v[50:51] /*v[818:819]*/, v[114:115] /*v[882:883]*/
	v_pk_mul_f32 v[102:103] /*v[614:615]*/, v[50:51] /*v[818:819]*/, v[116:117] /*v[884:885]*/
	v_pk_mul_f32 v[104:105] /*v[616:617]*/, v[50:51] /*v[818:819]*/, v[118:119] /*v[886:887]*/
	s_set_vgpr_msb 0x8f8e
	v_pk_add_f32 v[60:61] /*v[572:573]*/, v[60:61] /*v[572:573]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[62:63] /*v[574:575]*/, v[62:63] /*v[574:575]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ef6
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[82:89] /*v[594:601]*/, v[106:113] /*v[362:369]*/, v[136:143] /*v[904:911]*/
	s_set_vgpr_msb 0xf68e
	v_pk_add_f32 v[64:65] /*v[576:577]*/, v[64:65] /*v[576:577]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[74:75] /*v[586:587]*/, v[74:75] /*v[586:587]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[76:77] /*v[588:589]*/, v[76:77] /*v[588:589]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[78:79] /*v[590:591]*/, v[78:79] /*v[590:591]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[98:99] /*v[610:611]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[100:101] /*v[612:613]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ef6
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[82:89] /*v[594:601]*/, v[170:177] /*v[426:433]*/, v[144:151] /*v[912:919]*/
	s_set_vgpr_msb 0xf68e
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[102:103] /*v[614:615]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[104:105] /*v[616:617]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8ea2
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[2:9] /*v[514:521]*/, v[242:249], v[106:113] /*v[618:625]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[60:61] /*v[572:573]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[62:63] /*v[574:575]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[64:65] /*v[576:577]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xa2f6
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[2:9] /*v[514:521]*/, v[50:57] /*v[306:313]*/, v[152:159] /*v[920:927]*/
	s_set_vgpr_msb 0xf682
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[74:75] /*v[586:587]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[76:77] /*v[588:589]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[78:79] /*v[590:591]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99] /*v[610:611]*/, v[98:99] /*v[610:611]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101] /*v[612:613]*/, v[100:101] /*v[612:613]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103] /*v[614:615]*/, v[102:103] /*v[614:615]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x82f6
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[2:9] /*v[514:521]*/, v[122:129] /*v[378:385]*/, v[160:167] /*v[928:935]*/
	s_set_vgpr_msb 0xf682
	v_pk_mul_f32 v[104:105] /*v[616:617]*/, v[104:105] /*v[616:617]*/, s[4:5] op_sel_hi:[1,0]
	v_exp_f32_e32 v18 /*v530*/, v18 /*v530*/
	v_exp_f32_e32 v19 /*v531*/, v19 /*v531*/
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
	v_exp_f32_e32 v22 /*v534*/, v22 /*v534*/
	v_exp_f32_e32 v23 /*v535*/, v23 /*v535*/
	s_set_vgpr_msb 0x82f6
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[2:9] /*v[514:521]*/, v[186:193] /*v[442:449]*/, v[168:175] /*v[936:943]*/
	s_set_vgpr_msb 0xf682
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v25 /*v537*/, v25 /*v537*/
	v_exp_f32_e32 v34 /*v546*/, v34 /*v546*/
	v_exp_f32_e32 v35 /*v547*/, v35 /*v547*/
	s_set_vgpr_msb 0x828f
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[50:51] /*v[818:819]*/, v[72:73] /*v[840:841]*/
	v_pk_mul_f32 v[4:5] /*v[516:517]*/, v[50:51] /*v[818:819]*/, v[74:75] /*v[842:843]*/
	v_pk_mul_f32 v[6:7] /*v[518:519]*/, v[50:51] /*v[818:819]*/, v[76:77] /*v[844:845]*/
	v_pk_mul_f32 v[8:9] /*v[520:521]*/, v[50:51] /*v[818:819]*/, v[78:79] /*v[846:847]*/
	s_set_vgpr_msb 0x8ff6
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[66:73] /*v[578:585]*/, v[42:49] /*v[298:305]*/, v[128:135] /*v[896:903]*/
	s_set_vgpr_msb 0xf68e
	v_pk_add_f32 v[2:3] /*v[514:515]*/, v[2:3] /*v[514:515]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5] /*v[516:517]*/, v[4:5] /*v[516:517]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[6:7] /*v[518:519]*/, v[6:7] /*v[518:519]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9] /*v[520:521]*/, v[8:9] /*v[520:521]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v36 /*v548*/, v36 /*v548*/
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[2:3] /*v[514:515]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5] /*v[516:517]*/, v[4:5] /*v[516:517]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8ef6
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[66:73] /*v[578:585]*/, v[114:121] /*v[370:377]*/, v[136:143] /*v[904:911]*/
	s_set_vgpr_msb 0xf682
	v_pk_mul_f32 v[6:7] /*v[518:519]*/, v[6:7] /*v[518:519]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9] /*v[520:521]*/, v[8:9] /*v[520:521]*/, s[4:5] op_sel_hi:[1,0]
	v_exp_f32_e32 v2 /*v514*/, v2 /*v514*/
	v_exp_f32_e32 v3 /*v515*/, v3 /*v515*/
	v_exp_f32_e32 v4 /*v516*/, v4 /*v516*/
	v_exp_f32_e32 v5 /*v517*/, v5 /*v517*/
	v_exp_f32_e32 v6 /*v518*/, v6 /*v518*/
	s_set_vgpr_msb 0x82f6
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[66:73] /*v[578:585]*/, v[178:185] /*v[434:441]*/, v[144:151] /*v[912:919]*/
	s_set_vgpr_msb 0xf682
	v_exp_f32_e32 v7 /*v519*/, v7 /*v519*/
	v_exp_f32_e32 v8 /*v520*/, v8 /*v520*/
	v_exp_f32_e32 v9 /*v521*/, v9 /*v521*/
	v_exp_f32_e32 v37 /*v549*/, v37 /*v549*/
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	v_exp_f32_e32 v39 /*v551*/, v39 /*v551*/
	v_exp_f32_e32 v40 /*v552*/, v40 /*v552*/
	s_set_vgpr_msb 0x82f6
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[50:57] /*v[562:569]*/, v[50:57] /*v[306:313]*/, v[128:135] /*v[896:903]*/
	s_set_vgpr_msb 0xf682
	v_exp_f32_e32 v41 /*v553*/, v41 /*v553*/
	v_exp_f32_e32 v58 /*v570*/, v58 /*v570*/
	v_exp_f32_e32 v59 /*v571*/, v59 /*v571*/
	v_exp_f32_e32 v60 /*v572*/, v60 /*v572*/
	v_exp_f32_e32 v61 /*v573*/, v61 /*v573*/
	v_exp_f32_e32 v62 /*v574*/, v62 /*v574*/
	v_exp_f32_e32 v63 /*v575*/, v63 /*v575*/
	s_set_vgpr_msb 0x82f6
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[50:57] /*v[562:569]*/, v[122:129] /*v[378:385]*/, v[136:143] /*v[904:911]*/
	s_set_vgpr_msb 0xf682
	v_exp_f32_e32 v64 /*v576*/, v64 /*v576*/
	v_exp_f32_e32 v65 /*v577*/, v65 /*v577*/
	v_exp_f32_e32 v74 /*v586*/, v74 /*v586*/
	v_exp_f32_e32 v75 /*v587*/, v75 /*v587*/
	v_exp_f32_e32 v76 /*v588*/, v76 /*v588*/
	v_exp_f32_e32 v77 /*v589*/, v77 /*v589*/
	v_exp_f32_e32 v78 /*v590*/, v78 /*v590*/
	s_set_vgpr_msb 0x82f6
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[50:57] /*v[562:569]*/, v[186:193] /*v[442:449]*/, v[144:151] /*v[912:919]*/
	s_set_vgpr_msb 0xf682
	v_exp_f32_e32 v79 /*v591*/, v79 /*v591*/
	v_exp_f32_e32 v80 /*v592*/, v80 /*v592*/
	v_exp_f32_e32 v81 /*v593*/, v81 /*v593*/
	v_exp_f32_e32 v98 /*v610*/, v98 /*v610*/
	v_exp_f32_e32 v99 /*v611*/, v99 /*v611*/
	v_exp_f32_e32 v100 /*v612*/, v100 /*v612*/
	v_exp_f32_e32 v101 /*v613*/, v101 /*v613*/
	v_exp_f32_e32 v102 /*v614*/, v102 /*v614*/
	v_exp_f32_e32 v103 /*v615*/, v103 /*v615*/
	v_exp_f32_e32 v104 /*v616*/, v104 /*v616*/
	v_exp_f32_e32 v105 /*v617*/, v105 /*v617*/
	s_set_vgpr_msb 0x828f
	v_pk_add_f32 v[26:27] /*v[538:539]*/, v[128:129] /*v[896:897]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[130:131] /*v[898:899]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[30:31] /*v[542:543]*/, v[132:133] /*v[900:901]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33] /*v[544:545]*/, v[134:135] /*v[902:903]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[136:137] /*v[904:905]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[138:139] /*v[906:907]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[140:141] /*v[908:909]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[142:143] /*v[910:911]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[144:145] /*v[912:913]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[146:147] /*v[914:915]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[148:149] /*v[916:917]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[56:57] /*v[568:569]*/, v[150:151] /*v[918:919]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8f8e
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[106:107] /*v[618:619]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[108:109] /*v[620:621]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[70:71] /*v[582:583]*/, v[110:111] /*v[622:623]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[72:73] /*v[584:585]*/, v[112:113] /*v[624:625]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e8f
	v_pk_add_f32 v[82:83] /*v[594:595]*/, v[152:153] /*v[920:921]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[84:85] /*v[596:597]*/, v[154:155] /*v[922:923]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[86:87] /*v[598:599]*/, v[156:157] /*v[924:925]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[158:159] /*v[926:927]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[160:161] /*v[928:929]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[162:163] /*v[930:931]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[164:165] /*v[932:933]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[166:167] /*v[934:935]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8fcf
	v_pk_add_f32 v[72:73] /*v[840:841]*/, v[168:169] /*v[936:937]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[74:75] /*v[842:843]*/, v[170:171] /*v[938:939]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[76:77] /*v[844:845]*/, v[172:173] /*v[940:941]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[78:79] /*v[846:847]*/, v[174:175] /*v[942:943]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0xcf8a
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[10:11] /*v[522:523]*/, v[2:3] /*v[514:515]*/
	v_pk_mul_f32 v[4:5] /*v[516:517]*/, v[12:13] /*v[524:525]*/, v[4:5] /*v[516:517]*/
	v_pk_mul_f32 v[6:7] /*v[518:519]*/, v[14:15] /*v[526:527]*/, v[6:7] /*v[518:519]*/
	v_pk_mul_f32 v[8:9] /*v[520:521]*/, v[16:17] /*v[528:529]*/, v[8:9] /*v[520:521]*/
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[26:27] /*v[538:539]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[28:29] /*v[540:541]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[30:31] /*v[542:543]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[32:33] /*v[544:545]*/, v[24:25] /*v[536:537]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[42:43] /*v[554:555]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[44:45] /*v[556:557]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[46:47] /*v[558:559]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[48:49] /*v[560:561]*/, v[40:41] /*v[552:553]*/
	s_set_vgpr_msb 0x8a8e
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[50:51] /*v[562:563]*/, v[176:177] /*v[944:945]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[52:53] /*v[564:565]*/, v[178:179] /*v[946:947]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[54:55] /*v[566:567]*/, v[180:181] /*v[948:949]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[56:57] /*v[568:569]*/, v[182:183] /*v[950:951]*/
	s_set_vgpr_msb 0x8e8a
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[66:67] /*v[578:579]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[68:69] /*v[580:581]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[70:71] /*v[582:583]*/, v[62:63] /*v[574:575]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[72:73] /*v[584:585]*/, v[64:65] /*v[576:577]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[82:83] /*v[594:595]*/, v[74:75] /*v[586:587]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[84:85] /*v[596:597]*/, v[76:77] /*v[588:589]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[86:87] /*v[598:599]*/, v[78:79] /*v[590:591]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[88:89] /*v[600:601]*/, v[80:81] /*v[592:593]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[106:107] /*v[618:619]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[108:109] /*v[620:621]*/, v[100:101] /*v[612:613]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[110:111] /*v[622:623]*/, v[102:103] /*v[614:615]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[112:113] /*v[624:625]*/, v[104:105] /*v[616:617]*/
	s_set_vgpr_msb 0x8a8f
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[72:73] /*v[840:841]*/, v[184:185] /*v[952:953]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[74:75] /*v[842:843]*/, v[186:187] /*v[954:955]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[76:77] /*v[844:845]*/, v[188:189] /*v[956:957]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[78:79] /*v[846:847]*/, v[190:191] /*v[958:959]*/
	s_set_vgpr_msb 0x8f8b
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[50:51] /*v[818:819]*/, v[2:3] /*v[514:515]*/
	v_pk_mul_f32 v[4:5] /*v[516:517]*/, v[50:51] /*v[818:819]*/, v[4:5] /*v[516:517]*/
	v_pk_mul_f32 v[6:7] /*v[518:519]*/, v[50:51] /*v[818:819]*/, v[6:7] /*v[518:519]*/
	v_pk_mul_f32 v[8:9] /*v[520:521]*/, v[50:51] /*v[818:819]*/, v[8:9] /*v[520:521]*/
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[50:51] /*v[818:819]*/, v[10:11] /*v[522:523]*/
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[50:51] /*v[818:819]*/, v[12:13] /*v[524:525]*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[50:51] /*v[818:819]*/, v[14:15] /*v[526:527]*/
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[50:51] /*v[818:819]*/, v[16:17] /*v[528:529]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[50:51] /*v[818:819]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[50:51] /*v[818:819]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[50:51] /*v[818:819]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[50:51] /*v[818:819]*/, v[24:25] /*v[536:537]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[50:51] /*v[818:819]*/, v[26:27] /*v[538:539]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[50:51] /*v[818:819]*/, v[28:29] /*v[540:541]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[50:51] /*v[818:819]*/, v[30:31] /*v[542:543]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[50:51] /*v[818:819]*/, v[32:33] /*v[544:545]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[50:51] /*v[818:819]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[50:51] /*v[818:819]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[50:51] /*v[818:819]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[50:51] /*v[818:819]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[50:51] /*v[818:819]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[50:51] /*v[818:819]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[50:51] /*v[818:819]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[50:51] /*v[818:819]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[818:819]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[50:51] /*v[818:819]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[50:51] /*v[818:819]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[50:51] /*v[818:819]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[50:51] /*v[818:819]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[50:51] /*v[818:819]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[50:51] /*v[818:819]*/, v[62:63] /*v[574:575]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[50:51] /*v[818:819]*/, v[64:65] /*v[576:577]*/
	s_set_vgpr_msb 0x8b8a
	v_cvt_pk_bf16_f32 v2 /*v514*/, v2 /*v514*/, v3 /*v515*/
	v_cvt_pk_bf16_f32 v3 /*v515*/, v4 /*v516*/, v5 /*v517*/
	v_cvt_pk_bf16_f32 v4 /*v516*/, v6 /*v518*/, v7 /*v519*/
	v_cvt_pk_bf16_f32 v5 /*v517*/, v8 /*v520*/, v9 /*v521*/
	v_cvt_pk_bf16_f32 v10 /*v522*/, v10 /*v522*/, v11 /*v523*/
	v_cvt_pk_bf16_f32 v11 /*v523*/, v12 /*v524*/, v13 /*v525*/
	v_cvt_pk_bf16_f32 v12 /*v524*/, v14 /*v526*/, v15 /*v527*/
	v_cvt_pk_bf16_f32 v6 /*v518*/, v34 /*v546*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v7 /*v519*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v8 /*v520*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v9 /*v521*/, v40 /*v552*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v13 /*v525*/, v16 /*v528*/, v17 /*v529*/
	v_cvt_pk_bf16_f32 v18 /*v530*/, v18 /*v530*/, v19 /*v531*/
	v_cvt_pk_bf16_f32 v19 /*v531*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v14 /*v526*/, v42 /*v554*/, v43 /*v555*/
	v_cvt_pk_bf16_f32 v15 /*v527*/, v44 /*v556*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v16 /*v528*/, v46 /*v558*/, v47 /*v559*/
	v_cvt_pk_bf16_f32 v17 /*v529*/, v48 /*v560*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v20 /*v532*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v21 /*v533*/, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v26 /*v538*/, v26 /*v538*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v22 /*v534*/, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v23 /*v535*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v24 /*v536*/, v54 /*v566*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v25 /*v537*/, v56 /*v568*/, v57 /*v569*/
	v_cvt_pk_bf16_f32 v27 /*v539*/, v28 /*v540*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v28 /*v540*/, v30 /*v542*/, v31 /*v543*/
	v_cvt_pk_bf16_f32 v29 /*v541*/, v32 /*v544*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v30 /*v542*/, v58 /*v570*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v31 /*v543*/, v60 /*v572*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v32 /*v544*/, v62 /*v574*/, v63 /*v575*/
	v_cvt_pk_bf16_f32 v33 /*v545*/, v64 /*v576*/, v65 /*v577*/
	s_set_vgpr_msb 0x8a5e
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[120:127] /*v[888:895]*/, v[242:249] /*v[498:505]*/
	s_set_vgpr_msb 0x5e83
	s_wait_loadcnt 0xd
	v_mov_b64_e32 v[34:35] /*v[546:547]*/, v[10:11] /*v[778:779]*/
	v_mov_b64_e32 v[36:37] /*v[548:549]*/, v[12:13] /*v[780:781]*/
	s_set_vgpr_msb 0x8382
	v_mov_b64_e32 v[106:107] /*v[618:619]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[108:109] /*v[620:621]*/, v[200:201] /*v[712:713]*/
	s_wait_loadcnt 0x8
	v_mov_b64_e32 v[46:47] /*v[558:559]*/, v[246:247] /*v[758:759]*/
	v_mov_b64_e32 v[48:49] /*v[560:561]*/, v[248:249] /*v[760:761]*/
	v_mov_b64_e32 v[42:43] /*v[554:555]*/, v[242:243] /*v[754:755]*/
	s_set_vgpr_msb 0x825e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[10:17] /*v[522:529]*/, v[120:127] /*v[888:895]*/, v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5e82
	v_mov_b64_e32 v[44:45] /*v[556:557]*/, v[244:245] /*v[756:757]*/
	v_mov_b64_e32 v[102:103] /*v[614:615]*/, v[238:239] /*v[750:751]*/
	v_mov_b64_e32 v[104:105] /*v[616:617]*/, v[240:241] /*v[752:753]*/
	v_mov_b64_e32 v[98:99] /*v[610:611]*/, v[234:235] /*v[746:747]*/
	v_mov_b64_e32 v[100:101] /*v[612:613]*/, v[236:237] /*v[748:749]*/
	v_mov_b64_e32 v[38:39] /*v[550:551]*/, v[230:231] /*v[742:743]*/
	v_mov_b64_e32 v[40:41] /*v[552:553]*/, v[232:233] /*v[744:745]*/
	s_set_vgpr_msb 0x820e
	v_wmma_f32_16x16x32_bf16 v[122:129], v[18:25] /*v[530:537]*/, v[120:127] /*v[888:895]*/, v[122:129]
	s_set_vgpr_msb 0xe82
	v_mov_b64_e32 v[54:55] /*v[566:567]*/, v[222:223] /*v[734:735]*/
	v_mov_b64_e32 v[56:57] /*v[568:569]*/, v[224:225] /*v[736:737]*/
	v_mov_b64_e32 v[50:51] /*v[562:563]*/, v[218:219] /*v[730:731]*/
	v_mov_b64_e32 v[52:53] /*v[564:565]*/, v[220:221] /*v[732:733]*/
	v_mov_b64_e32 v[62:63] /*v[574:575]*/, v[214:215] /*v[726:727]*/
	v_mov_b64_e32 v[64:65] /*v[576:577]*/, v[216:217] /*v[728:729]*/
	v_mov_b64_e32 v[58:59] /*v[570:571]*/, v[210:211] /*v[722:723]*/
	s_set_vgpr_msb 0x820e
	v_wmma_f32_16x16x32_bf16 v[58:65], v[26:33] /*v[538:545]*/, v[120:127] /*v[888:895]*/, v[58:65]
	s_set_vgpr_msb 0xe82
	v_mov_b64_e32 v[60:61] /*v[572:573]*/, v[212:213] /*v[724:725]*/
	v_mov_b64_e32 v[70:71] /*v[582:583]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[72:73] /*v[584:585]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[66:67] /*v[578:579]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[68:69] /*v[580:581]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[78:79] /*v[590:591]*/, v[194:195] /*v[706:707]*/
	v_mov_b64_e32 v[80:81] /*v[592:593]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[2:9] /*v[514:521]*/, v[154:161] /*v[666:673]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x5a82
	v_mov_b64_e32 v[74:75] /*v[586:587]*/, v[190:191] /*v[702:703]*/
	v_mov_b64_e32 v[76:77] /*v[588:589]*/, v[192:193] /*v[704:705]*/
	v_mov_b64_e32 v[86:87] /*v[598:599]*/, v[182:183] /*v[694:695]*/
	v_mov_b64_e32 v[88:89] /*v[600:601]*/, v[184:185] /*v[696:697]*/
	v_mov_b64_e32 v[82:83] /*v[594:595]*/, v[178:179] /*v[690:691]*/
	v_mov_b64_e32 v[84:85] /*v[596:597]*/, v[180:181] /*v[692:693]*/
	v_mov_b64_e32 v[110:111] /*v[622:623]*/, v[166:167] /*v[678:679]*/
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[178:185], v[10:17] /*v[522:529]*/, v[154:161] /*v[666:673]*/, v[178:185]
	s_set_vgpr_msb 0xa82
	v_mov_b64_e32 v[112:113] /*v[624:625]*/, v[168:169] /*v[680:681]*/
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[114:121], v[18:25] /*v[530:537]*/, v[154:161] /*v[666:673]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[26:33] /*v[538:545]*/, v[154:161] /*v[666:673]*/, v[50:57]
	s_set_vgpr_msb 0xa5a
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[2:9] /*v[514:521]*/, v[146:153] /*v[658:665]*/, v[234:241] /*v[490:497]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[170:177], v[10:17] /*v[522:529]*/, v[146:153] /*v[658:665]*/, v[170:177]
	v_wmma_f32_16x16x32_bf16 v[106:113], v[18:25] /*v[530:537]*/, v[146:153] /*v[658:665]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[26:33] /*v[538:545]*/, v[146:153] /*v[658:665]*/, v[42:49]
	s_set_vgpr_msb 0xa5a
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[2:9] /*v[514:521]*/, v[138:145] /*v[650:657]*/, v[226:233] /*v[482:489]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[162:169], v[10:17] /*v[522:529]*/, v[138:145] /*v[650:657]*/, v[162:169]
	v_wmma_f32_16x16x32_bf16 v[98:105], v[18:25] /*v[530:537]*/, v[138:145] /*v[650:657]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[26:33] /*v[538:545]*/, v[138:145] /*v[650:657]*/, v[34:41]
	s_set_vgpr_msb 0xa5a
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[2:9] /*v[514:521]*/, v[130:137] /*v[642:649]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[154:161], v[10:17] /*v[522:529]*/, v[130:137] /*v[642:649]*/, v[154:161]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[18:25] /*v[530:537]*/, v[130:137] /*v[642:649]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[26:33] /*v[538:545]*/, v[130:137] /*v[642:649]*/, v[26:33]
	s_set_vgpr_msb 0xa5a
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/, v[122:129] /*v[634:641]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[146:153], v[10:17] /*v[522:529]*/, v[122:129] /*v[634:641]*/, v[146:153]
	v_wmma_f32_16x16x32_bf16 v[82:89], v[18:25] /*v[530:537]*/, v[122:129] /*v[634:641]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[26:33] /*v[538:545]*/, v[122:129] /*v[634:641]*/, v[18:25]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa82
	v_mov_b64_e32 v[122:123] /*v[634:635]*/, v[186:187] /*v[698:699]*/
	v_mov_b64_e32 v[124:125] /*v[636:637]*/, v[188:189] /*v[700:701]*/
	v_mov_b64_e32 v[126:127] /*v[638:639]*/, v[162:163] /*v[674:675]*/
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/, v[114:121] /*v[626:633]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5a82
	v_mov_b64_e32 v[128:129] /*v[640:641]*/, v[164:165] /*v[676:677]*/
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[138:145], v[10:17] /*v[522:529]*/, v[114:121] /*v[626:633]*/, v[138:145]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[18:25] /*v[530:537]*/, v[114:121] /*v[626:633]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[26:33] /*v[538:545]*/, v[114:121] /*v[626:633]*/, v[10:17]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa82
	v_mov_b64_e32 v[118:119] /*v[630:631]*/, v[174:175] /*v[686:687]*/
	v_mov_b64_e32 v[120:121] /*v[632:633]*/, v[176:177] /*v[688:689]*/
	v_mov_b64_e32 v[114:115] /*v[626:627]*/, v[170:171] /*v[682:683]*/
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/, v[90:97] /*v[602:609]*/, v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5a82
	v_mov_b64_e32 v[116:117] /*v[628:629]*/, v[172:173] /*v[684:685]*/
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x8283
	v_mov_b64_e32 v[6:7] /*v[518:519]*/, v[30:31] /*v[798:799]*/
	v_mov_b64_e32 v[8:9] /*v[520:521]*/, v[32:33] /*v[800:801]*/
	v_mov_b64_e32 v[2:3] /*v[514:515]*/, v[26:27] /*v[794:795]*/
	s_set_vgpr_msb 0x830a
	v_wmma_f32_16x16x32_bf16 v[130:137], v[10:17] /*v[522:529]*/, v[90:97] /*v[602:609]*/, v[130:137]
	s_set_vgpr_msb 0xa83
	v_mov_b64_e32 v[4:5] /*v[516:517]*/, v[28:29] /*v[796:797]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[14:15] /*v[526:527]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[16:17] /*v[528:529]*/, v[24:25] /*v[792:793]*/
	v_mov_b64_e32 v[10:11] /*v[522:523]*/, v[18:19] /*v[786:787]*/
	s_set_vgpr_msb 0x830a
	v_wmma_f32_16x16x32_bf16 v[66:73], v[18:25] /*v[530:537]*/, v[90:97] /*v[602:609]*/, v[66:73]
	s_set_vgpr_msb 0xa83
	v_mov_b64_e32 v[12:13] /*v[524:525]*/, v[20:21] /*v[788:789]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[22:23] /*v[534:535]*/, v[14:15] /*v[782:783]*/
	v_mov_b64_e32 v[24:25] /*v[536:537]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[18:19] /*v[530:531]*/, v[6:7] /*v[774:775]*/
	s_set_vgpr_msb 0x830a
	v_wmma_f32_16x16x32_bf16 v[2:9], v[26:33] /*v[538:545]*/, v[90:97] /*v[602:609]*/, v[2:9]
	s_set_vgpr_msb 0xa83
	v_mov_b64_e32 v[20:21] /*v[532:533]*/, v[8:9] /*v[776:777]*/
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x8382
	v_mov_b64_e32 v[90:91] /*v[602:603]*/, v[254:255] /*v[766:767]*/
	s_set_vgpr_msb 0x8283
	v_mov_b64_e32 v[92:93] /*v[604:605]*/, v[0:1] /*v[768:769]*/
	v_mov_b64_e32 v[30:31] /*v[542:543]*/, v[2:3] /*v[770:771]*/
	v_mov_b64_e32 v[32:33] /*v[544:545]*/, v[4:5] /*v[772:773]*/
	s_set_vgpr_msb 0x8382
	v_mov_b64_e32 v[26:27] /*v[538:539]*/, v[250:251] /*v[762:763]*/
	v_mov_b64_e32 v[28:29] /*v[540:541]*/, v[252:253] /*v[764:765]*/
	v_mov_b64_e32 v[94:95] /*v[606:607]*/, v[226:227] /*v[738:739]*/
	v_mov_b64_e32 v[96:97] /*v[608:609]*/, v[228:229] /*v[740:741]*/
	s_set_vgpr_msb 0x8200
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
	v_mov_b64_e32 v[72:73], v[8:9]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[80:81], v[8:9]
	v_mov_b64_e32 v[78:79], v[6:7]
	v_mov_b64_e32 v[76:77], v[4:5]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[88:89], v[8:9]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[96:97], v[8:9]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[104:105], v[8:9]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[112:113], v[8:9]
	v_mov_b64_e32 v[110:111], v[6:7]
	v_mov_b64_e32 v[108:109], v[4:5]
	v_mov_b64_e32 v[106:107], v[2:3]
	v_mov_b64_e32 v[120:121], v[8:9]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[128:129], v[8:9]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[136:137], v[8:9]
	v_mov_b64_e32 v[134:135], v[6:7]
	v_mov_b64_e32 v[132:133], v[4:5]
	v_mov_b64_e32 v[130:131], v[2:3]
	v_mov_b64_e32 v[144:145], v[8:9]
	v_mov_b64_e32 v[142:143], v[6:7]
	v_mov_b64_e32 v[140:141], v[4:5]
	v_mov_b64_e32 v[138:139], v[2:3]
	v_mov_b64_e32 v[152:153], v[8:9]
	v_mov_b64_e32 v[150:151], v[6:7]
	v_mov_b64_e32 v[148:149], v[4:5]
	v_mov_b64_e32 v[146:147], v[2:3]
	v_mov_b64_e32 v[160:161], v[8:9]
	v_mov_b64_e32 v[158:159], v[6:7]
	v_mov_b64_e32 v[156:157], v[4:5]
	v_mov_b64_e32 v[154:155], v[2:3]
	v_mov_b64_e32 v[168:169], v[8:9]
	v_mov_b64_e32 v[166:167], v[6:7]
	v_mov_b64_e32 v[164:165], v[4:5]
	v_mov_b64_e32 v[162:163], v[2:3]
	v_mov_b64_e32 v[176:177], v[8:9]
	v_mov_b64_e32 v[174:175], v[6:7]
	v_mov_b64_e32 v[172:173], v[4:5]
	v_mov_b64_e32 v[170:171], v[2:3]
	v_mov_b64_e32 v[184:185], v[8:9]
	v_mov_b64_e32 v[182:183], v[6:7]
	v_mov_b64_e32 v[180:181], v[4:5]
	v_mov_b64_e32 v[178:179], v[2:3]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[8:9]
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[6:7]
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[4:5]
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[2:3]
	v_mov_b64_e32 v[200:201] /*v[456:457]*/, v[8:9]
	v_mov_b64_e32 v[198:199] /*v[454:455]*/, v[6:7]
	v_mov_b64_e32 v[196:197] /*v[452:453]*/, v[4:5]
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, v[2:3]
	v_mov_b64_e32 v[208:209] /*v[464:465]*/, v[8:9]
	v_mov_b64_e32 v[206:207] /*v[462:463]*/, v[6:7]
	v_mov_b64_e32 v[204:205] /*v[460:461]*/, v[4:5]
	v_mov_b64_e32 v[202:203] /*v[458:459]*/, v[2:3]
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[8:9]
	v_mov_b64_e32 v[214:215] /*v[470:471]*/, v[6:7]
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[4:5]
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[2:3]
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[8:9]
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[6:7]
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[4:5]
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[2:3]
	v_mov_b64_e32 v[232:233] /*v[488:489]*/, v[8:9]
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[6:7]
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[4:5]
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[2:3]
	v_mov_b64_e32 v[240:241] /*v[496:497]*/, v[8:9]
	v_mov_b64_e32 v[238:239] /*v[494:495]*/, v[6:7]
	v_mov_b64_e32 v[236:237] /*v[492:493]*/, v[4:5]
	v_mov_b64_e32 v[234:235] /*v[490:491]*/, v[2:3]
	s_set_vgpr_msb 0x4080
	v_mov_b64_e32 v[0:1] /*v[512:513]*/, v[8:9]
	s_set_vgpr_msb 0x8040
	v_mov_b64_e32 v[254:255] /*v[510:511]*/, v[6:7]
	v_mov_b64_e32 v[252:253] /*v[508:509]*/, v[4:5]
	v_mov_b64_e32 v[250:251] /*v[506:507]*/, v[2:3]
	v_mov_b64_e32 v[248:249] /*v[504:505]*/, v[8:9]
	v_mov_b64_e32 v[246:247] /*v[502:503]*/, v[6:7]
	v_mov_b64_e32 v[244:245] /*v[500:501]*/, v[4:5]
	v_mov_b64_e32 v[242:243] /*v[498:499]*/, v[2:3]
	s_set_vgpr_msb 0x4000
.LBB0_4:
	s_sub_co_i32 s70, s17, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s70, 1
	s_cbranch_scc1 .LBB0_7
	s_set_vgpr_msb 0x8c
	v_dual_add_nc_u32 v102 /*v614*/, s16, v52 /*v820*/ :: v_dual_add_nc_u32 v106 /*v618*/, s71, v57 /*v825*/
	s_set_vgpr_msb 0x8c80
	v_add_nc_u32_e32 v104 /*v616*/, s71, v1
	s_set_vgpr_msb 0x800c
	v_lshlrev_b32_e32 v1, 4, v52 /*v820*/
	s_set_vgpr_msb 0xc8c
	v_add_lshl_u32 v9 /*v521*/, s8, v52 /*v820*/, 4
	s_lshl_b32 s3, s8, 4
	s_lshl_b32 s4, s2, 9
	v_dual_add_nc_u32 v108 /*v620*/, s71, v58 /*v826*/ :: v_dual_bitop2_b32 v2 /*v514*/, 32, v61 /*v829*/ bitop3:0x54
	s_set_vgpr_msb 0x8c00
	v_add3_u32 v1, s3, s4, v1
	s_set_vgpr_msb 0x88
	v_add3_u32 v9 /*v521*/, s4, v9 /*v521*/, 0x100
	s_set_vgpr_msb 0x888c
	v_or_b32_e32 v4 /*v516*/, 0x60, v61 /*v829*/
	v_or_b32_e32 v5 /*v517*/, 0x80, v60 /*v828*/
	v_or_b32_e32 v6 /*v518*/, 0xa0, v61 /*v829*/
	v_mul_lo_u32 v10 /*v522*/, v1, s68
	s_set_vgpr_msb 0x8c88
	v_mul_lo_u32 v9 /*v521*/, s68, v9 /*v521*/
	s_set_vgpr_msb 0x888e
	v_or_b32_e32 v7 /*v519*/, 0xc0, v60 /*v828*/
	v_or_b32_e32 v8 /*v520*/, 0xe0, v61 /*v829*/
	v_dual_mov_b32 v105 /*v617*/, v106 /*v618*/ :: v_dual_bitop2_b32 v3 /*v515*/, 64, v60 /*v828*/ bitop3:0x54
	v_mov_b32_e32 v103 /*v615*/, v104 /*v616*/
	s_mov_b32 s65, s64
	v_or_b32_e32 v10 /*v522*/, v10 /*v522*/, v56 /*v824*/
	s_set_vgpr_msb 0x8e8b
	v_dual_add_nc_u32 v115 /*v627*/, v59 /*v827*/, v2 /*v514*/ :: v_dual_bitop2_b32 v9 /*v521*/, v56 /*v824*/, v9 /*v521*/ bitop3:0x54
	v_mov_b64_e32 v[110:111] /*v[622:623]*/, s[64:65]
	s_set_vgpr_msb 0x8bc3
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v49 /*v817*/, v48 /*v816*/ :: v_dual_mov_b32 v47 /*v815*/, v46 /*v814*/
	v_dual_mov_b32 v45 /*v813*/, v44 /*v812*/ :: v_dual_mov_b32 v43 /*v811*/, v42 /*v810*/
	v_dual_mov_b32 v35 /*v803*/, v34 /*v802*/ :: v_dual_mov_b32 v37 /*v805*/, v36 /*v804*/
	s_set_vgpr_msb 0xc302
	v_mov_b32_e32 v1, v102 /*v614*/
	s_set_vgpr_msb 0x2c3
	v_dual_mov_b32 v39 /*v807*/, v38 /*v806*/ :: v_dual_mov_b32 v41 /*v809*/, v40 /*v808*/
	s_set_vgpr_msb 0xc3b2
	v_mov_b32_e32 v107 /*v619*/, v108 /*v620*/
	v_lshl_or_b32 v112 /*v624*/, s2, 5, v53 /*v821*/
	v_add_lshl_u32 v113 /*v625*/, v9 /*v521*/, s5, 4
	s_set_vgpr_msb 0xb28f
	v_add_nc_u32_e32 v114 /*v626*/, v59 /*v827*/, v60 /*v828*/
	s_set_vgpr_msb 0x8f8b
	v_add_nc_u32_e32 v116 /*v628*/, v59 /*v827*/, v3 /*v515*/
	v_add_lshl_u32 v109 /*v621*/, s5, v10 /*v522*/, 4
	v_dual_add_nc_u32 v117 /*v629*/, v59 /*v827*/, v4 /*v516*/ :: v_dual_add_nc_u32 v118 /*v630*/, v59 /*v827*/, v5 /*v517*/
	v_dual_add_nc_u32 v119 /*v631*/, v59 /*v827*/, v6 /*v518*/ :: v_dual_add_nc_u32 v120 /*v632*/, v59 /*v827*/, v7 /*v519*/
	v_add_nc_u32_e32 v121 /*v633*/, v59 /*v827*/, v8 /*v520*/
	s_ashr_i32 s71, s70, 31
	s_lshl_b32 s68, s68, 13
	s_mov_b32 s66, 0x3fb8aa3b
	s_set_vgpr_msb 0x8b00
.LBB0_6:
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x8a
	v_or_b32_e32 v2 /*v514*/, 32, v109 /*v621*/
	v_or_b32_e32 v3 /*v515*/, 0xe0, v109 /*v621*/
	v_or_b32_e32 v4 /*v516*/, 32, v113 /*v625*/
	s_clause 0x4
	buffer_load_b128 v[62:65] /*v[574:577]*/, v109 /*v621*/, s[72:75], null offen
	buffer_load_b128 v[122:125] /*v[634:637]*/, v113 /*v625*/, s[72:75], null offen
	buffer_load_b128 v[66:69] /*v[578:581]*/, v2 /*v514*/, s[72:75], null offen
	buffer_load_b128 v[134:137] /*v[646:649]*/, v3 /*v515*/, s[72:75], null offen
	buffer_load_b128 v[126:129] /*v[638:641]*/, v4 /*v516*/, s[72:75], null offen
	v_or_b32_e32 v5 /*v517*/, 64, v109 /*v621*/
	v_or_b32_e32 v6 /*v518*/, 0x60, v109 /*v621*/
	v_or_b32_e32 v7 /*v519*/, 64, v113 /*v625*/
	v_or_b32_e32 v9 /*v521*/, 0x80, v109 /*v621*/
	v_or_b32_e32 v8 /*v520*/, 0x60, v113 /*v625*/
	v_or_b32_e32 v10 /*v522*/, 0xa0, v109 /*v621*/
	v_or_b32_e32 v11 /*v523*/, 0x80, v113 /*v625*/
	s_clause 0x3
	buffer_load_b128 v[138:141] /*v[650:653]*/, v5 /*v517*/, s[72:75], null offen
	buffer_load_b128 v[142:145] /*v[654:657]*/, v6 /*v518*/, s[72:75], null offen
	buffer_load_b128 v[146:149] /*v[658:661]*/, v7 /*v519*/, s[72:75], null offen
	buffer_load_b128 v[150:153] /*v[662:665]*/, v8 /*v520*/, s[72:75], null offen
	v_or_b32_e32 v12 /*v524*/, 0xa0, v113 /*v625*/
	s_clause 0x3
	buffer_load_b128 v[154:157] /*v[666:669]*/, v9 /*v521*/, s[72:75], null offen
	buffer_load_b128 v[158:161] /*v[670:673]*/, v10 /*v522*/, s[72:75], null offen
	buffer_load_b128 v[162:165] /*v[674:677]*/, v11 /*v523*/, s[72:75], null offen
	buffer_load_b128 v[166:169] /*v[678:681]*/, v12 /*v524*/, s[72:75], null offen
	v_or_b32_e32 v13 /*v525*/, 0xc0, v109 /*v621*/
	v_or_b32_e32 v14 /*v526*/, 0xc0, v113 /*v625*/
	v_or_b32_e32 v15 /*v527*/, 0xe0, v113 /*v625*/
	s_clause 0x4
	buffer_load_b128 v[170:173] /*v[682:685]*/, v109 /*v621*/, s[76:79], null offen
	buffer_load_b128 v[178:181] /*v[690:693]*/, v113 /*v625*/, s[76:79], null offen
	buffer_load_b128 v[174:177] /*v[686:689]*/, v2 /*v514*/, s[76:79], null offen
	buffer_load_b128 v[190:193] /*v[702:705]*/, v3 /*v515*/, s[76:79], null offen
	buffer_load_b128 v[182:185] /*v[694:697]*/, v4 /*v516*/, s[76:79], null offen
	s_clause 0x2
	buffer_load_b128 v[130:133] /*v[642:645]*/, v13 /*v525*/, s[72:75], null offen
	buffer_load_b128 v[194:197] /*v[706:709]*/, v14 /*v526*/, s[72:75], null offen
	buffer_load_b128 v[198:201] /*v[710:713]*/, v15 /*v527*/, s[72:75], null offen
	s_clause 0xa
	buffer_load_b128 v[202:205] /*v[714:717]*/, v5 /*v517*/, s[76:79], null offen
	buffer_load_b128 v[206:209] /*v[718:721]*/, v6 /*v518*/, s[76:79], null offen
	buffer_load_b128 v[210:213] /*v[722:725]*/, v7 /*v519*/, s[76:79], null offen
	buffer_load_b128 v[214:217] /*v[726:729]*/, v8 /*v520*/, s[76:79], null offen
	buffer_load_b128 v[218:221] /*v[730:733]*/, v9 /*v521*/, s[76:79], null offen
	buffer_load_b128 v[222:225] /*v[734:737]*/, v10 /*v522*/, s[76:79], null offen
	buffer_load_b128 v[226:229] /*v[738:741]*/, v11 /*v523*/, s[76:79], null offen
	buffer_load_b128 v[230:233] /*v[742:745]*/, v12 /*v524*/, s[76:79], null offen
	buffer_load_b128 v[186:189] /*v[698:701]*/, v13 /*v525*/, s[76:79], null offen
	buffer_load_b128 v[22:25] /*v[534:537]*/, v14 /*v526*/, s[76:79], null offen
	buffer_load_b128 v[26:29] /*v[538:541]*/, v15 /*v527*/, s[76:79], null offen
	s_wait_xcnt 0xf
	v_dual_add_nc_u32 v2 /*v514*/, 16, v112 /*v624*/ :: v_dual_bitop2_b32 v3 /*v515*/, 3, v112 /*v624*/ bitop3:0x54
	s_wait_xcnt 0xe
	v_dual_add_nc_u32 v109 /*v621*/, s68, v109 /*v621*/ :: v_dual_bitop2_b32 v4 /*v516*/, 2, v112 /*v624*/ bitop3:0x54
	s_wait_xcnt 0x1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_nc_u32 v113 /*v625*/, s68, v113 /*v625*/ :: v_dual_bitop2_b32 v14 /*v526*/, 6, v2 /*v514*/ bitop3:0x54
	v_or_b32_e32 v5 /*v517*/, 5, v112 /*v624*/
	v_or_b32_e32 v6 /*v518*/, 4, v112 /*v624*/
	v_or_b32_e32 v7 /*v519*/, 7, v112 /*v624*/
	v_or_b32_e32 v8 /*v520*/, 6, v112 /*v624*/
	v_cmp_gt_i32_e64 s29, v14 /*v526*/, v102 /*v614*/
	v_cmp_gt_i32_e64 s21, v14 /*v526*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s13, v14 /*v526*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s5, v14 /*v526*/, v108 /*v620*/
	v_or_b32_e32 v9 /*v521*/, 3, v2 /*v514*/
	v_or_b32_e32 v10 /*v522*/, 2, v2 /*v514*/
	v_or_b32_e32 v11 /*v523*/, 5, v2 /*v514*/
	v_or_b32_e32 v12 /*v524*/, 4, v2 /*v514*/
	v_or_b32_e32 v13 /*v525*/, 7, v2 /*v514*/
	v_cmp_ge_i32_e32 vcc_lo, v112 /*v624*/, v102 /*v614*/
	v_cmp_gt_i32_e64 s64, v112 /*v624*/, v102 /*v614*/
	s_set_vgpr_msb 0x8a02
	v_cmp_gt_i32_e64 s61, v3 /*v515*/, v1
	s_set_vgpr_msb 0x20a
	v_cmp_gt_i32_e64 s65, v4 /*v516*/, v102 /*v614*/
	s_set_vgpr_msb 0xa02
	v_cmp_gt_i32_e64 s60, v5 /*v517*/, v1
	s_set_vgpr_msb 0x20a
	v_cmp_gt_i32_e64 s63, v6 /*v518*/, v102 /*v614*/
	s_set_vgpr_msb 0xa02
	v_cmp_gt_i32_e64 s59, v7 /*v519*/, v1
	s_set_vgpr_msb 0x20a
	v_cmp_gt_i32_e64 s62, v8 /*v520*/, v102 /*v614*/
	v_cmp_ge_i32_e64 s56, v112 /*v624*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s57, v112 /*v624*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s53, v3 /*v515*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s58, v4 /*v516*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s52, v5 /*v517*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s55, v6 /*v518*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s51, v7 /*v519*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s54, v8 /*v520*/, v104 /*v616*/
	v_cmp_ge_i32_e64 s48, v112 /*v624*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s49, v112 /*v624*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s45, v3 /*v515*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s50, v4 /*v516*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s44, v5 /*v517*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s47, v6 /*v518*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s43, v7 /*v519*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s46, v8 /*v520*/, v106 /*v618*/
	v_cmp_ge_i32_e64 s40, v112 /*v624*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s41, v112 /*v624*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s37, v3 /*v515*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s42, v4 /*v516*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s36, v5 /*v517*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s39, v6 /*v518*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s35, v7 /*v519*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s38, v8 /*v520*/, v108 /*v620*/
	v_cmp_ge_i32_e64 s31, v2 /*v514*/, v102 /*v614*/
	v_cmp_gt_i32_e64 s33, v2 /*v514*/, v102 /*v614*/
	s_set_vgpr_msb 0xa02
	v_cmp_gt_i32_e64 s28, v9 /*v521*/, v1
	s_set_vgpr_msb 0x20a
	v_cmp_gt_i32_e64 s34, v10 /*v522*/, v102 /*v614*/
	s_set_vgpr_msb 0xa02
	v_cmp_gt_i32_e64 s27, v11 /*v523*/, v1
	s_set_vgpr_msb 0x20a
	v_cmp_gt_i32_e64 s30, v12 /*v524*/, v102 /*v614*/
	s_set_vgpr_msb 0xa02
	v_cmp_gt_i32_e64 s26, v13 /*v525*/, v1
	s_set_vgpr_msb 0x28a
	v_cmp_ge_i32_e64 s23, v2 /*v514*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s24, v2 /*v514*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s20, v9 /*v521*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s25, v10 /*v522*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s19, v11 /*v523*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s22, v12 /*v524*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s18, v13 /*v525*/, v103 /*v615*/
	v_cmp_ge_i32_e64 s15, v2 /*v514*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s16, v2 /*v514*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s12, v9 /*v521*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s17, v10 /*v522*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s11, v11 /*v523*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s14, v12 /*v524*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s10, v13 /*v525*/, v105 /*v617*/
	v_cmp_ge_i32_e64 s7, v2 /*v514*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s8, v2 /*v514*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s4, v9 /*v521*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s9, v10 /*v522*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s3, v11 /*v523*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s6, v12 /*v524*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s2, v13 /*v525*/, v107 /*v619*/
	s_and_b32 s64, s84, s64
	s_and_b32 s65, s84, s65
	s_and_b32 s61, s84, s61
	s_and_b32 s63, s84, s63
	s_and_b32 s60, s84, s60
	s_and_b32 s62, s84, s62
	s_and_b32 s59, s84, s59
	s_and_b32 s56, s84, s56
	s_and_b32 s57, s84, s57
	s_and_b32 s58, s84, s58
	s_and_b32 s53, s84, s53
	s_and_b32 s55, s84, s55
	s_and_b32 s52, s84, s52
	s_and_b32 s54, s84, s54
	s_and_b32 s51, s84, s51
	s_and_b32 s48, s84, s48
	s_and_b32 s49, s84, s49
	s_and_b32 s50, s84, s50
	s_and_b32 s45, s84, s45
	s_and_b32 s47, s84, s47
	s_and_b32 s44, s84, s44
	s_and_b32 s46, s84, s46
	s_and_b32 s43, s84, s43
	s_and_b32 s40, s84, s40
	s_and_b32 s41, s84, s41
	s_and_b32 s42, s84, s42
	s_and_b32 s37, s84, s37
	s_and_b32 s39, s84, s39
	s_and_b32 s36, s84, s36
	s_and_b32 s38, s84, s38
	s_and_b32 s35, s84, s35
	s_and_b32 s31, s84, s31
	s_and_b32 s33, s84, s33
	s_and_b32 s34, s84, s34
	s_and_b32 s28, s84, s28
	s_and_b32 s30, s84, s30
	s_and_b32 s27, s84, s27
	s_and_b32 s29, s84, s29
	s_and_b32 s26, s84, s26
	s_and_b32 s23, s84, s23
	s_and_b32 s24, s84, s24
	s_and_b32 s25, s84, s25
	s_and_b32 s20, s84, s20
	s_and_b32 s22, s84, s22
	s_and_b32 s19, s84, s19
	s_and_b32 s21, s84, s21
	s_and_b32 s18, s84, s18
	s_and_b32 s15, s84, s15
	s_and_b32 s16, s84, s16
	s_and_b32 s17, s84, s17
	s_and_b32 s12, s84, s12
	s_and_b32 s14, s84, s14
	s_and_b32 s11, s84, s11
	s_and_b32 s13, s84, s13
	s_and_b32 s10, s84, s10
	s_and_b32 s7, s84, s7
	s_and_b32 s8, s84, s8
	s_and_b32 s9, s84, s9
	s_and_b32 s4, s84, s4
	s_and_b32 s6, s84, s6
	s_and_b32 s3, s84, s3
	s_and_b32 s5, s84, s5
	s_and_b32 s2, s84, s2
	s_and_b32 s86, s84, vcc_lo
	v_add_nc_u32_e32 v112 /*v624*/, 32, v112 /*v624*/
	s_add_nc_u64 s[70:71], s[70:71], -1
	s_set_vgpr_msb 0x8a82
	s_wait_loadcnt 0x10
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[170:177] /*v[682:689]*/, v[218:225], 0
	s_set_vgpr_msb 0x820b
	ds_store_b128 v54 /*v822*/, v[62:65] /*v[574:577]*/
	ds_store_b128 v54 /*v822*/, v[66:69] /*v[578:581]*/ offset:32
	ds_store_b128 v54 /*v822*/, v[138:141] /*v[650:653]*/ offset:64
	ds_store_b128 v54 /*v822*/, v[142:145] /*v[654:657]*/ offset:96
	s_set_vgpr_msb 0xb82
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[62:69] /*v[574:581]*/, v[186:193], 0
	s_set_vgpr_msb 0x820b
	ds_store_b128 v54 /*v822*/, v[154:157] /*v[666:669]*/ offset:128
	ds_store_b128 v54 /*v822*/, v[158:161] /*v[670:673]*/ offset:160
	s_wait_loadcnt 0xd
	ds_store_b128 v54 /*v822*/, v[130:133] /*v[642:645]*/ offset:192
	ds_store_b128 v54 /*v822*/, v[134:137] /*v[646:649]*/ offset:224
	ds_store_b128 v55 /*v823*/, v[122:125] /*v[634:637]*/
	ds_store_b128 v55 /*v823*/, v[126:129] /*v[638:641]*/ offset:32
	ds_store_b128 v55 /*v823*/, v[146:149] /*v[658:661]*/ offset:64
	ds_store_b128 v55 /*v823*/, v[150:153] /*v[662:665]*/ offset:96
	ds_store_b128 v55 /*v823*/, v[162:165] /*v[674:677]*/ offset:128
	ds_store_b128 v55 /*v823*/, v[166:169] /*v[678:681]*/ offset:160
	s_wait_loadcnt 0xc
	ds_store_b128 v55 /*v823*/, v[194:197] /*v[706:709]*/ offset:192
	s_wait_loadcnt 0xb
	ds_store_b128 v55 /*v823*/, v[198:201] /*v[710:713]*/ offset:224
	s_set_vgpr_msb 0xb86
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v114 /*v626*/
	ds_load_tr16_b128 v[10:13] /*v[522:525]*/, v114 /*v626*/ offset:4352
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v115 /*v627*/
	s_cmp_lg_u64 s[70:71], 0
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[122:129] /*v[634:641]*/, v[58:65] /*v[314:321]*/, 0
	s_set_vgpr_msb 0x8682
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[122:129] /*v[634:641]*/, v[186:193], 0
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[598:605]*/, v[62:69] /*v[574:581]*/, v[250:257], 0
	s_set_vgpr_msb 0x8286
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[62:69] /*v[574:581]*/, v[58:65] /*v[314:321]*/, 0
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[62:69] /*v[574:581]*/, v[130:137] /*v[386:393]*/, 0
	s_set_vgpr_msb 0x8682
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[122:129] /*v[634:641]*/, v[250:257], 0
	s_set_vgpr_msb 0x82c6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[122:129] /*v[634:641]*/, v[130:137] /*v[386:393]*/, 0
	s_set_vgpr_msb 0xc6a2
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[138:145] /*v[650:657]*/, v[194:201], v[46:53] /*v[558:565]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[170:177] /*v[682:689]*/, v[98:105] /*v[354:361]*/, 0
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[146:153] /*v[658:665]*/, v[74:81] /*v[330:337]*/, v[14:21] /*v[526:533]*/
	s_set_vgpr_msb 0xa6a2
	s_wait_loadcnt 0x9
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[202:209] /*v[714:721]*/, v[226:233], v[54:61] /*v[566:573]*/
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[146:153] /*v[658:665]*/, v[194:201], v[30:37] /*v[542:549]*/
	s_set_vgpr_msb 0xa286
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[170:177] /*v[682:689]*/, v[26:33] /*v[282:289]*/, 0
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[170:177] /*v[682:689]*/, v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x8682
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[178:185] /*v[690:697]*/, v[218:225], 0
	s_set_vgpr_msb 0x82a6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[178:185] /*v[690:697]*/, v[26:33] /*v[282:289]*/, 0
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[178:185] /*v[690:697]*/, v[98:105] /*v[354:361]*/, 0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[178:185] /*v[690:697]*/, v[162:169] /*v[418:425]*/, 0
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[598:605]*/, v[138:145] /*v[650:657]*/, v[2:9] /*v[258:265]*/, v[86:93] /*v[598:605]*/
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[138:145] /*v[650:657]*/, v[74:81] /*v[330:337]*/, v[78:85] /*v[590:597]*/
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[138:145] /*v[650:657]*/, v[138:145] /*v[394:401]*/, v[70:77] /*v[582:589]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[146:153] /*v[658:665]*/, v[2:9] /*v[258:265]*/, v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[146:153] /*v[658:665]*/, v[138:145] /*v[394:401]*/, v[2:9] /*v[770:777]*/
	s_set_vgpr_msb 0xf6a2
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[154:161] /*v[666:673]*/, v[202:209], v[46:53] /*v[558:565]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[162:169] /*v[674:681]*/, v[82:89] /*v[338:345]*/, v[14:21] /*v[526:533]*/
	s_set_vgpr_msb 0xa6a2
	s_wait_loadcnt 0x5
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[218:225] /*v[730:737]*/, v[234:241], v[54:61] /*v[566:573]*/
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[162:169] /*v[674:681]*/, v[202:209], v[30:37] /*v[542:549]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[210:217] /*v[722:729]*/, v[226:233], v[170:177] /*v[682:689]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[210:217] /*v[722:729]*/, v[34:41] /*v[290:297]*/, v[242:249] /*v[754:761]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[210:217] /*v[722:729]*/, v[106:113] /*v[362:369]*/, v[250:257] /*v[762:769]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[210:217] /*v[722:729]*/, v[170:177] /*v[426:433]*/, v[122:129] /*v[634:641]*/
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[598:605]*/, v[154:161] /*v[666:673]*/, v[10:17] /*v[266:273]*/, v[86:93] /*v[598:605]*/
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[154:161] /*v[666:673]*/, v[82:89] /*v[338:345]*/, v[78:85] /*v[590:597]*/
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[154:161] /*v[666:673]*/, v[146:153] /*v[402:409]*/, v[70:77] /*v[582:589]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[162:169] /*v[674:681]*/, v[10:17] /*v[266:273]*/, v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[162:169] /*v[674:681]*/, v[146:153] /*v[402:409]*/, v[2:9] /*v[770:777]*/
	s_set_vgpr_msb 0xf6a2
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[130:137] /*v[642:649]*/, v[210:217], v[46:53] /*v[558:565]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[194:201] /*v[706:713]*/, v[90:97] /*v[346:353]*/, v[14:21] /*v[526:533]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa68a
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[110:111] /*v[622:623]*/, v[14:15] /*v[526:527]*/
	s_set_vgpr_msb 0x8aa2
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[186:193] /*v[698:705]*/, v[242:249], v[54:61] /*v[566:573]*/
	s_set_vgpr_msb 0xa28a
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[110:111] /*v[622:623]*/, v[16:17] /*v[528:529]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[110:111] /*v[622:623]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[110:111] /*v[622:623]*/, v[20:21] /*v[532:533]*/
	v_cndmask_b32_e64 v15 /*v527*/, v15 /*v527*/, 0xff61b1e6, s15
	v_cndmask_b32_e64 v14 /*v526*/, v14 /*v526*/, 0xff61b1e6, s16
	v_cndmask_b32_e64 v17 /*v529*/, v17 /*v529*/, 0xff61b1e6, s12
	v_cndmask_b32_e64 v16 /*v528*/, v16 /*v528*/, 0xff61b1e6, s17
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[226:233] /*v[738:745]*/, v[234:241], v[170:177] /*v[682:689]*/
	v_cndmask_b32_e64 v19 /*v531*/, v19 /*v531*/, 0xff61b1e6, s11
	v_cndmask_b32_e64 v18 /*v530*/, v18 /*v530*/, 0xff61b1e6, s14
	v_cndmask_b32_e64 v21 /*v533*/, v21 /*v533*/, 0xff61b1e6, s10
	v_cndmask_b32_e64 v20 /*v532*/, v20 /*v532*/, 0xff61b1e6, s13
	s_set_vgpr_msb 0xa28e
	v_pk_add_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[226:233] /*v[738:745]*/, v[42:49] /*v[298:305]*/, v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa68e
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, s[66:67] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, s[66:67] op_sel_hi:[1,0]
	v_exp_f32_e32 v14 /*v526*/, v14 /*v526*/
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[226:233] /*v[738:745]*/, v[114:121] /*v[370:377]*/, v[250:257] /*v[762:769]*/
	v_exp_f32_e32 v15 /*v527*/, v15 /*v527*/
	v_exp_f32_e32 v16 /*v528*/, v16 /*v528*/
	v_exp_f32_e32 v17 /*v529*/, v17 /*v529*/
	v_exp_f32_e32 v18 /*v530*/, v18 /*v530*/
	v_exp_f32_e32 v19 /*v531*/, v19 /*v531*/
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[226:233] /*v[738:745]*/, v[178:185] /*v[434:441]*/, v[122:129] /*v[634:641]*/
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[598:605]*/, v[130:137] /*v[642:649]*/, v[18:25] /*v[274:281]*/, v[86:93] /*v[598:605]*/
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[130:137] /*v[642:649]*/, v[90:97] /*v[346:353]*/, v[78:85] /*v[590:597]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa68a
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[110:111] /*v[622:623]*/, v[78:79] /*v[590:591]*/
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[130:137] /*v[642:649]*/, v[154:161] /*v[410:417]*/, v[70:77] /*v[582:589]*/
	s_set_vgpr_msb 0xa68a
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[110:111] /*v[622:623]*/, v[80:81] /*v[592:593]*/
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[110:111] /*v[622:623]*/, v[82:83] /*v[594:595]*/
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[110:111] /*v[622:623]*/, v[84:85] /*v[596:597]*/
	v_cndmask_b32_e64 v79 /*v591*/, v79 /*v591*/, 0xff61b1e6, s48
	v_cndmask_b32_e64 v78 /*v590*/, v78 /*v590*/, 0xff61b1e6, s49
	v_cndmask_b32_e64 v81 /*v593*/, v81 /*v593*/, 0xff61b1e6, s45
	v_cndmask_b32_e64 v80 /*v592*/, v80 /*v592*/, 0xff61b1e6, s50
	s_set_vgpr_msb 0x8aa2
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[194:201] /*v[706:713]*/, v[210:217], v[30:37] /*v[542:549]*/
	s_set_vgpr_msb 0xa28a
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[110:111] /*v[622:623]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[110:111] /*v[622:623]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[110:111] /*v[622:623]*/, v[74:75] /*v[586:587]*/
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[110:111] /*v[622:623]*/, v[76:77] /*v[588:589]*/
	v_cndmask_b32_e64 v83 /*v595*/, v83 /*v595*/, 0xff61b1e6, s44
	v_cndmask_b32_e64 v82 /*v594*/, v82 /*v594*/, 0xff61b1e6, s47
	v_cndmask_b32_e64 v85 /*v597*/, v85 /*v597*/, 0xff61b1e6, s43
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[194:201] /*v[706:713]*/, v[18:25] /*v[274:281]*/, v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa68a
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[110:111] /*v[622:623]*/, v[30:31] /*v[542:543]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[110:111] /*v[622:623]*/, v[32:33] /*v[544:545]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[110:111] /*v[622:623]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[110:111] /*v[622:623]*/, v[36:37] /*v[548:549]*/
	v_cndmask_b32_e64 v84 /*v596*/, v84 /*v596*/, 0xff61b1e6, s46
	v_cndmask_b32_e64 v71 /*v583*/, v71 /*v583*/, 0xff61b1e6, s40
	v_cndmask_b32_e64 v70 /*v582*/, v70 /*v582*/, 0xff61b1e6, s41
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[194:201] /*v[706:713]*/, v[154:161] /*v[410:417]*/, v[2:9] /*v[770:777]*/
	s_set_vgpr_msb 0xf68a
	v_pk_mul_f32 v[130:131] /*v[642:643]*/, v[110:111] /*v[622:623]*/, v[234:235] /*v[746:747]*/
	v_pk_mul_f32 v[132:133] /*v[644:645]*/, v[110:111] /*v[622:623]*/, v[236:237] /*v[748:749]*/
	v_pk_mul_f32 v[134:135] /*v[646:647]*/, v[110:111] /*v[622:623]*/, v[238:239] /*v[750:751]*/
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[110:111] /*v[622:623]*/, v[240:241] /*v[752:753]*/
	v_cndmask_b32_e64 v73 /*v585*/, v73 /*v585*/, 0xff61b1e6, s37
	v_cndmask_b32_e64 v72 /*v584*/, v72 /*v584*/, 0xff61b1e6, s42
	v_cndmask_b32_e64 v75 /*v587*/, v75 /*v587*/, 0xff61b1e6, s36
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[202:209] /*v[714:721]*/, v[106:113] /*v[362:369]*/, v[38:45] /*v[550:557]*/
	s_set_vgpr_msb 0xa68e
	v_pk_mul_f32 v[154:155] /*v[666:667]*/, v[110:111] /*v[622:623]*/, v[2:3] /*v[770:771]*/
	v_pk_mul_f32 v[156:157] /*v[668:669]*/, v[110:111] /*v[622:623]*/, v[4:5] /*v[772:773]*/
	v_pk_mul_f32 v[158:159] /*v[670:671]*/, v[110:111] /*v[622:623]*/, v[6:7] /*v[774:775]*/
	v_pk_mul_f32 v[160:161] /*v[672:673]*/, v[110:111] /*v[622:623]*/, v[8:9] /*v[776:777]*/
	v_cndmask_b32_e64 v74 /*v586*/, v74 /*v586*/, 0xff61b1e6, s39
	v_cndmask_b32_e64 v77 /*v589*/, v77 /*v589*/, 0xff61b1e6, s35
	v_cndmask_b32_e64 v76 /*v588*/, v76 /*v588*/, 0xff61b1e6, s38
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[202:209] /*v[714:721]*/, v[34:41] /*v[290:297]*/, v[94:101] /*v[606:613]*/
	v_cndmask_b32_e64 v31 /*v543*/, v31 /*v543*/, 0xff61b1e6, s31
	v_cndmask_b32_e64 v30 /*v542*/, v30 /*v542*/, 0xff61b1e6, s33
	v_cndmask_b32_e64 v33 /*v545*/, v33 /*v545*/, 0xff61b1e6, s28
	v_cndmask_b32_e64 v32 /*v544*/, v32 /*v544*/, 0xff61b1e6, s34
	v_cndmask_b32_e64 v35 /*v547*/, v35 /*v547*/, 0xff61b1e6, s27
	v_cndmask_b32_e64 v34 /*v546*/, v34 /*v546*/, 0xff61b1e6, s30
	v_cndmask_b32_e64 v37 /*v549*/, v37 /*v549*/, 0xff61b1e6, s26
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[202:209] /*v[714:721]*/, v[170:177] /*v[426:433]*/, v[62:69] /*v[574:581]*/
	v_cndmask_b32_e64 v36 /*v548*/, v36 /*v548*/, 0xff61b1e6, s29
	v_cndmask_b32_e64 v131 /*v643*/, v131 /*v643*/, 0xff61b1e6, s23
	v_cndmask_b32_e64 v130 /*v642*/, v130 /*v642*/, 0xff61b1e6, s24
	v_cndmask_b32_e64 v133 /*v645*/, v133 /*v645*/, 0xff61b1e6, s20
	v_cndmask_b32_e64 v132 /*v644*/, v132 /*v644*/, 0xff61b1e6, s25
	v_cndmask_b32_e64 v135 /*v647*/, v135 /*v647*/, 0xff61b1e6, s19
	v_cndmask_b32_e64 v134 /*v646*/, v134 /*v646*/, 0xff61b1e6, s22
	s_set_vgpr_msb 0xa6a2
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[22:29] /*v[534:541]*/, v[242:249], v[170:177] /*v[682:689]*/
	v_cndmask_b32_e64 v137 /*v649*/, v137 /*v649*/, 0xff61b1e6, s18
	v_cndmask_b32_e64 v136 /*v648*/, v136 /*v648*/, 0xff61b1e6, s21
	v_cndmask_b32_e64 v155 /*v667*/, v155 /*v667*/, 0xff61b1e6, s7
	v_cndmask_b32_e64 v154 /*v666*/, v154 /*v666*/, 0xff61b1e6, s8
	v_cndmask_b32_e64 v157 /*v669*/, v157 /*v669*/, 0xff61b1e6, s4
	v_cndmask_b32_e64 v156 /*v668*/, v156 /*v668*/, 0xff61b1e6, s9
	v_cndmask_b32_e64 v159 /*v671*/, v159 /*v671*/, 0xff61b1e6, s3
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[22:29] /*v[534:541]*/, v[50:57] /*v[306:313]*/, v[242:249] /*v[754:761]*/
	v_cndmask_b32_e64 v158 /*v670*/, v158 /*v670*/, 0xff61b1e6, s6
	v_cndmask_b32_e64 v161 /*v673*/, v161 /*v673*/, 0xff61b1e6, s2
	v_cndmask_b32_e64 v160 /*v672*/, v160 /*v672*/, 0xff61b1e6, s5
	s_set_vgpr_msb 0xa68e
	v_pk_add_f32 v[78:79] /*v[590:591]*/, v[78:79] /*v[590:591]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[82:83] /*v[594:595]*/, v[82:83] /*v[594:595]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, v[38:39] /*v[806:807]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[22:29] /*v[534:541]*/, v[122:129] /*v[378:385]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa68e
	v_pk_add_f32 v[70:71] /*v[582:583]*/, v[70:71] /*v[582:583]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[72:73] /*v[584:585]*/, v[72:73] /*v[584:585]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[74:75] /*v[586:587]*/, v[74:75] /*v[586:587]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[76:77] /*v[588:589]*/, v[76:77] /*v[588:589]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[30:31] /*v[542:543]*/, v[30:31] /*v[542:543]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[22:29] /*v[534:541]*/, v[186:193] /*v[442:449]*/, v[122:129] /*v[634:641]*/
	s_set_vgpr_msb 0xa68e
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[130:131] /*v[642:643]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[132:133] /*v[644:645]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[134:135] /*v[646:647]*/, v[134:135] /*v[646:647]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e8a
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[110:111] /*v[622:623]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[110:111] /*v[622:623]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[110:111] /*v[622:623]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[110:111] /*v[622:623]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a8e
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[54:55] /*v[566:567]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[56:57] /*v[568:569]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[58:59] /*v[570:571]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[60:61] /*v[572:573]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e8a
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[110:111] /*v[622:623]*/, v[86:87] /*v[598:599]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[110:111] /*v[622:623]*/, v[88:89] /*v[600:601]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[110:111] /*v[622:623]*/, v[90:91] /*v[602:603]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[110:111] /*v[622:623]*/, v[92:93] /*v[604:605]*/
	v_cndmask_b32_e64 v23 /*v535*/, v23 /*v535*/, 0xff61b1e6, s86
	v_cndmask_b32_e64 v22 /*v534*/, v22 /*v534*/, 0xff61b1e6, s64
	v_cndmask_b32_e64 v25 /*v537*/, v25 /*v537*/, 0xff61b1e6, s61
	v_cndmask_b32_e64 v24 /*v536*/, v24 /*v536*/, 0xff61b1e6, s65
	v_cndmask_b32_e64 v27 /*v539*/, v27 /*v539*/, 0xff61b1e6, s60
	v_cndmask_b32_e64 v26 /*v538*/, v26 /*v538*/, 0xff61b1e6, s63
	v_cndmask_b32_e64 v29 /*v541*/, v29 /*v541*/, 0xff61b1e6, s59
	v_cndmask_b32_e64 v28 /*v540*/, v28 /*v540*/, 0xff61b1e6, s62
	v_cndmask_b32_e64 v55 /*v567*/, v55 /*v567*/, 0xff61b1e6, s56
	v_cndmask_b32_e64 v54 /*v566*/, v54 /*v566*/, 0xff61b1e6, s57
	v_cndmask_b32_e64 v57 /*v569*/, v57 /*v569*/, 0xff61b1e6, s53
	v_cndmask_b32_e64 v56 /*v568*/, v56 /*v568*/, 0xff61b1e6, s58
	v_cndmask_b32_e64 v59 /*v571*/, v59 /*v571*/, 0xff61b1e6, s52
	v_cndmask_b32_e64 v58 /*v570*/, v58 /*v570*/, 0xff61b1e6, s55
	v_cndmask_b32_e64 v61 /*v573*/, v61 /*v573*/, 0xff61b1e6, s51
	v_cndmask_b32_e64 v60 /*v572*/, v60 /*v572*/, 0xff61b1e6, s54
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[218:225] /*v[730:737]*/, v[114:121] /*v[370:377]*/, v[38:45] /*v[550:557]*/
	s_set_vgpr_msb 0xa68e
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[28:29] /*v[540:541]*/, v[34:35] /*v[802:803]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[218:225] /*v[730:737]*/, v[42:49] /*v[298:305]*/, v[94:101] /*v[606:613]*/
	s_set_vgpr_msb 0xa68e
	v_pk_add_f32 v[60:61] /*v[572:573]*/, v[60:61] /*v[572:573]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[36:37] /*v[804:805]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[154:155] /*v[666:667]*/, v[154:155] /*v[666:667]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[156:157] /*v[668:669]*/, v[156:157] /*v[668:669]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159] /*v[670:671]*/, v[158:159] /*v[670:671]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[160:161] /*v[672:673]*/, v[160:161] /*v[672:673]*/, v[40:41] /*v[808:809]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, s[66:67] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[218:225] /*v[730:737]*/, v[178:185] /*v[434:441]*/, v[62:69] /*v[574:581]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[28:29] /*v[540:541]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[60:61] /*v[572:573]*/, s[66:67] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[186:193] /*v[698:705]*/, v[122:129] /*v[378:385]*/, v[38:45] /*v[550:557]*/
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[78:79] /*v[590:591]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[82:83] /*v[594:595]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[70:71] /*v[582:583]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[72:73] /*v[584:585]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[74:75] /*v[586:587]*/, s[66:67] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[186:193] /*v[698:705]*/, v[50:57] /*v[306:313]*/, v[94:101] /*v[606:613]*/
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[76:77] /*v[588:589]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[30:31] /*v[542:543]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131] /*v[642:643]*/, v[130:131] /*v[642:643]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133] /*v[644:645]*/, v[132:133] /*v[644:645]*/, s[66:67] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[186:193] /*v[698:705]*/, v[186:193] /*v[442:449]*/, v[62:69] /*v[574:581]*/
	v_pk_mul_f32 v[134:135] /*v[646:647]*/, v[134:135] /*v[646:647]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155] /*v[666:667]*/, v[154:155] /*v[666:667]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157] /*v[668:669]*/, v[156:157] /*v[668:669]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159] /*v[670:671]*/, v[158:159] /*v[670:671]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161] /*v[672:673]*/, v[160:161] /*v[672:673]*/, s[66:67] op_sel_hi:[1,0]
	v_exp_f32_e32 v22 /*v534*/, v22 /*v534*/
	v_exp_f32_e32 v23 /*v535*/, v23 /*v535*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v25 /*v537*/, v25 /*v537*/
	v_exp_f32_e32 v26 /*v538*/, v26 /*v538*/
	v_exp_f32_e32 v27 /*v539*/, v27 /*v539*/
	v_exp_f32_e32 v28 /*v540*/, v28 /*v540*/
	v_exp_f32_e32 v29 /*v541*/, v29 /*v541*/
	v_exp_f32_e32 v54 /*v566*/, v54 /*v566*/
	v_exp_f32_e32 v55 /*v567*/, v55 /*v567*/
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v57 /*v569*/, v57 /*v569*/
	v_exp_f32_e32 v58 /*v570*/, v58 /*v570*/
	v_exp_f32_e32 v59 /*v571*/, v59 /*v571*/
	v_exp_f32_e32 v60 /*v572*/, v60 /*v572*/
	v_exp_f32_e32 v61 /*v573*/, v61 /*v573*/
	v_exp_f32_e32 v78 /*v590*/, v78 /*v590*/
	v_exp_f32_e32 v79 /*v591*/, v79 /*v591*/
	v_exp_f32_e32 v80 /*v592*/, v80 /*v592*/
	v_exp_f32_e32 v81 /*v593*/, v81 /*v593*/
	v_exp_f32_e32 v82 /*v594*/, v82 /*v594*/
	v_exp_f32_e32 v83 /*v595*/, v83 /*v595*/
	v_exp_f32_e32 v84 /*v596*/, v84 /*v596*/
	v_exp_f32_e32 v85 /*v597*/, v85 /*v597*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	v_exp_f32_e32 v71 /*v583*/, v71 /*v583*/
	v_exp_f32_e32 v72 /*v584*/, v72 /*v584*/
	v_exp_f32_e32 v73 /*v585*/, v73 /*v585*/
	v_exp_f32_e32 v74 /*v586*/, v74 /*v586*/
	v_exp_f32_e32 v75 /*v587*/, v75 /*v587*/
	v_exp_f32_e32 v76 /*v588*/, v76 /*v588*/
	v_exp_f32_e32 v77 /*v589*/, v77 /*v589*/
	v_exp_f32_e32 v30 /*v542*/, v30 /*v542*/
	v_exp_f32_e32 v31 /*v543*/, v31 /*v543*/
	v_exp_f32_e32 v32 /*v544*/, v32 /*v544*/
	v_exp_f32_e32 v33 /*v545*/, v33 /*v545*/
	v_exp_f32_e32 v34 /*v546*/, v34 /*v546*/
	v_exp_f32_e32 v35 /*v547*/, v35 /*v547*/
	v_exp_f32_e32 v36 /*v548*/, v36 /*v548*/
	v_exp_f32_e32 v37 /*v549*/, v37 /*v549*/
	v_exp_f32_e32 v130 /*v642*/, v130 /*v642*/
	v_exp_f32_e32 v131 /*v643*/, v131 /*v643*/
	v_exp_f32_e32 v132 /*v644*/, v132 /*v644*/
	v_exp_f32_e32 v133 /*v645*/, v133 /*v645*/
	v_exp_f32_e32 v134 /*v646*/, v134 /*v646*/
	v_exp_f32_e32 v135 /*v647*/, v135 /*v647*/
	v_exp_f32_e32 v136 /*v648*/, v136 /*v648*/
	v_exp_f32_e32 v137 /*v649*/, v137 /*v649*/
	v_exp_f32_e32 v154 /*v666*/, v154 /*v666*/
	v_exp_f32_e32 v155 /*v667*/, v155 /*v667*/
	v_exp_f32_e32 v156 /*v668*/, v156 /*v668*/
	v_exp_f32_e32 v157 /*v669*/, v157 /*v669*/
	v_exp_f32_e32 v158 /*v670*/, v158 /*v670*/
	v_exp_f32_e32 v159 /*v671*/, v159 /*v671*/
	v_exp_f32_e32 v160 /*v672*/, v160 /*v672*/
	v_exp_f32_e32 v161 /*v673*/, v161 /*v673*/
	s_set_vgpr_msb 0xa68e
	v_pk_add_f32 v[86:87] /*v[598:599]*/, v[94:95] /*v[606:607]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[96:97] /*v[608:609]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[98:99] /*v[610:611]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[100:101] /*v[612:613]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[62:63] /*v[574:575]*/, v[62:63] /*v[574:575]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[64:65] /*v[576:577]*/, v[64:65] /*v[576:577]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[170:171] /*v[682:683]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[172:173] /*v[684:685]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[174:175] /*v[686:687]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[176:177] /*v[688:689]*/, v[42:43] /*v[810:811]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[138:139] /*v[650:651]*/, v[242:243] /*v[754:755]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[140:141] /*v[652:653]*/, v[244:245] /*v[756:757]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[142:143] /*v[654:655]*/, v[246:247] /*v[758:759]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[144:145] /*v[656:657]*/, v[248:249] /*v[760:761]*/, v[44:45] /*v[812:813]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[146:147] /*v[658:659]*/, v[250:251] /*v[762:763]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[148:149] /*v[660:661]*/, v[252:253] /*v[764:765]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[150:151] /*v[662:663]*/, v[254:255] /*v[766:767]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e8f
	v_pk_add_f32 v[152:153] /*v[664:665]*/, v[0:1] /*v[768:769]*/, v[46:47] /*v[814:815]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8f8e
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[122:123] /*v[634:635]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[124:125] /*v[636:637]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[126:127] /*v[638:639]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[128:129] /*v[640:641]*/, v[48:49] /*v[816:817]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8e8a
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[46:47] /*v[558:559]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[48:49] /*v[560:561]*/, v[24:25] /*v[536:537]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[50:51] /*v[562:563]*/, v[26:27] /*v[538:539]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[52:53] /*v[564:565]*/, v[28:29] /*v[540:541]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[86:87] /*v[598:599]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[88:89] /*v[600:601]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[90:91] /*v[602:603]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[92:93] /*v[604:605]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[78:79] /*v[590:591]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[80:81] /*v[592:593]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, v[82:83] /*v[594:595]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[84:85] /*v[596:597]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[62:63] /*v[574:575]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[64:65] /*v[576:577]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[66:67] /*v[578:579]*/, v[74:75] /*v[586:587]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[68:69] /*v[580:581]*/, v[76:77] /*v[588:589]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[94:95] /*v[606:607]*/, v[30:31] /*v[542:543]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[96:97] /*v[608:609]*/, v[32:33] /*v[544:545]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[98:99] /*v[610:611]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[100:101] /*v[612:613]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[138:139] /*v[650:651]*/, v[130:131] /*v[642:643]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[140:141] /*v[652:653]*/, v[132:133] /*v[644:645]*/
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[142:143] /*v[654:655]*/, v[134:135] /*v[646:647]*/
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[144:145] /*v[656:657]*/, v[136:137] /*v[648:649]*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[146:147] /*v[658:659]*/, v[14:15] /*v[526:527]*/
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[148:149] /*v[660:661]*/, v[16:17] /*v[528:529]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[150:151] /*v[662:663]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[152:153] /*v[664:665]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[122:123] /*v[634:635]*/, v[154:155] /*v[666:667]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[124:125] /*v[636:637]*/, v[156:157] /*v[668:669]*/
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[126:127] /*v[638:639]*/, v[158:159] /*v[670:671]*/
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[128:129] /*v[640:641]*/, v[160:161] /*v[672:673]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[110:111] /*v[622:623]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[110:111] /*v[622:623]*/, v[24:25] /*v[536:537]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[110:111] /*v[622:623]*/, v[26:27] /*v[538:539]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[110:111] /*v[622:623]*/, v[28:29] /*v[540:541]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[110:111] /*v[622:623]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[110:111] /*v[622:623]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[110:111] /*v[622:623]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[110:111] /*v[622:623]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[110:111] /*v[622:623]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[110:111] /*v[622:623]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[110:111] /*v[622:623]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[110:111] /*v[622:623]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[110:111] /*v[622:623]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[110:111] /*v[622:623]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[110:111] /*v[622:623]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[110:111] /*v[622:623]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[110:111] /*v[622:623]*/, v[30:31] /*v[542:543]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[110:111] /*v[622:623]*/, v[32:33] /*v[544:545]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[110:111] /*v[622:623]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[110:111] /*v[622:623]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[110:111] /*v[622:623]*/, v[62:63] /*v[574:575]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[110:111] /*v[622:623]*/, v[64:65] /*v[576:577]*/
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[110:111] /*v[622:623]*/, v[66:67] /*v[578:579]*/
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[110:111] /*v[622:623]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[110:111] /*v[622:623]*/, v[14:15] /*v[526:527]*/
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[110:111] /*v[622:623]*/, v[16:17] /*v[528:529]*/
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[110:111] /*v[622:623]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[110:111] /*v[622:623]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[110:111] /*v[622:623]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[110:111] /*v[622:623]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[110:111] /*v[622:623]*/, v[74:75] /*v[586:587]*/
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[110:111] /*v[622:623]*/, v[76:77] /*v[588:589]*/
	v_cvt_pk_bf16_f32 v14 /*v526*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v15 /*v527*/, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v16 /*v528*/, v26 /*v538*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v17 /*v529*/, v28 /*v540*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v22 /*v534*/, v46 /*v558*/, v47 /*v559*/
	v_cvt_pk_bf16_f32 v23 /*v535*/, v48 /*v560*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v24 /*v536*/, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v18 /*v530*/, v30 /*v542*/, v31 /*v543*/
	v_cvt_pk_bf16_f32 v19 /*v531*/, v32 /*v544*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v20 /*v532*/, v34 /*v546*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v21 /*v533*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v25 /*v537*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v30 /*v542*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v31 /*v543*/, v40 /*v552*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v26 /*v538*/, v62 /*v574*/, v63 /*v575*/
	v_cvt_pk_bf16_f32 v27 /*v539*/, v64 /*v576*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v28 /*v540*/, v66 /*v578*/, v67 /*v579*/
	v_cvt_pk_bf16_f32 v29 /*v541*/, v68 /*v580*/, v69 /*v581*/
	v_cvt_pk_bf16_f32 v32 /*v544*/, v42 /*v554*/, v43 /*v555*/
	v_cvt_pk_bf16_f32 v33 /*v545*/, v44 /*v556*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v38 /*v550*/, v54 /*v566*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v34 /*v546*/, v78 /*v590*/, v79 /*v591*/
	v_cvt_pk_bf16_f32 v35 /*v547*/, v80 /*v592*/, v81 /*v593*/
	v_cvt_pk_bf16_f32 v36 /*v548*/, v82 /*v594*/, v83 /*v595*/
	v_cvt_pk_bf16_f32 v37 /*v549*/, v84 /*v596*/, v85 /*v597*/
	v_cvt_pk_bf16_f32 v39 /*v551*/, v56 /*v568*/, v57 /*v569*/
	v_cvt_pk_bf16_f32 v40 /*v552*/, v58 /*v570*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v41 /*v553*/, v60 /*v572*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v42 /*v554*/, v70 /*v582*/, v71 /*v583*/
	v_cvt_pk_bf16_f32 v43 /*v555*/, v72 /*v584*/, v73 /*v585*/
	v_cvt_pk_bf16_f32 v44 /*v556*/, v74 /*v586*/, v75 /*v587*/
	v_cvt_pk_bf16_f32 v45 /*v557*/, v76 /*v588*/, v77 /*v589*/
	s_set_vgpr_msb 0x8a5a
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[14:21] /*v[526:533]*/, v[6:13] /*v[518:525]*/, v[242:249] /*v[498:505]*/
	s_set_vgpr_msb 0x5a82
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v116 /*v628*/
	ds_load_tr16_b128 v[50:53] /*v[562:565]*/, v116 /*v628*/ offset:4352
	s_set_vgpr_msb 0x825a
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[22:29] /*v[534:541]*/, v[6:13] /*v[518:525]*/, v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[122:129], v[30:37] /*v[542:549]*/, v[6:13] /*v[518:525]*/, v[122:129]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[38:45] /*v[550:557]*/, v[6:13] /*v[518:525]*/, v[58:65]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v115 /*v627*/ offset:4352
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[14:21] /*v[526:533]*/, v[2:9] /*v[514:521]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[178:185], v[22:29] /*v[534:541]*/, v[2:9] /*v[514:521]*/, v[178:185]
	v_wmma_f32_16x16x32_bf16 v[114:121], v[30:37] /*v[542:549]*/, v[2:9] /*v[514:521]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[38:45] /*v[550:557]*/, v[2:9] /*v[514:521]*/, v[50:57]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v117 /*v629*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v117 /*v629*/ offset:4352
	s_set_vgpr_msb 0x825a
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[14:21] /*v[526:533]*/, v[46:53] /*v[558:565]*/, v[234:241] /*v[490:497]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[170:177], v[22:29] /*v[534:541]*/, v[46:53] /*v[558:565]*/, v[170:177]
	v_wmma_f32_16x16x32_bf16 v[106:113], v[30:37] /*v[542:549]*/, v[46:53] /*v[558:565]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[38:45] /*v[550:557]*/, v[46:53] /*v[558:565]*/, v[42:49]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v118 /*v630*/
	ds_load_tr16_b128 v[50:53] /*v[562:565]*/, v118 /*v630*/ offset:4352
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[14:21] /*v[526:533]*/, v[2:9] /*v[514:521]*/, v[226:233] /*v[482:489]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[162:169], v[22:29] /*v[534:541]*/, v[2:9] /*v[514:521]*/, v[162:169]
	v_wmma_f32_16x16x32_bf16 v[98:105], v[30:37] /*v[542:549]*/, v[2:9] /*v[514:521]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[38:45] /*v[550:557]*/, v[2:9] /*v[514:521]*/, v[34:41]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v119 /*v631*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v119 /*v631*/ offset:4352
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[14:21] /*v[526:533]*/, v[46:53] /*v[558:565]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[154:161], v[22:29] /*v[534:541]*/, v[46:53] /*v[558:565]*/, v[154:161]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[30:37] /*v[542:549]*/, v[46:53] /*v[558:565]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[38:45] /*v[550:557]*/, v[46:53] /*v[558:565]*/, v[26:33]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v120 /*v632*/
	ds_load_tr16_b128 v[50:53] /*v[562:565]*/, v120 /*v632*/ offset:4352
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[14:21] /*v[526:533]*/, v[2:9] /*v[514:521]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[146:153], v[22:29] /*v[534:541]*/, v[2:9] /*v[514:521]*/, v[146:153]
	v_wmma_f32_16x16x32_bf16 v[82:89], v[30:37] /*v[542:549]*/, v[2:9] /*v[514:521]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[38:45] /*v[550:557]*/, v[2:9] /*v[514:521]*/, v[18:25]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v121 /*v633*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v121 /*v633*/ offset:4352
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[14:21] /*v[526:533]*/, v[46:53] /*v[558:565]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[138:145], v[22:29] /*v[534:541]*/, v[46:53] /*v[558:565]*/, v[138:145]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[30:37] /*v[542:549]*/, v[46:53] /*v[558:565]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[38:45] /*v[550:557]*/, v[46:53] /*v[558:565]*/, v[10:17]
	s_set_vgpr_msb 0xa5a
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[14:21] /*v[526:533]*/, v[2:9] /*v[514:521]*/, v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[130:137], v[22:29] /*v[534:541]*/, v[2:9] /*v[514:521]*/, v[130:137]
	v_wmma_f32_16x16x32_bf16 v[66:73], v[30:37] /*v[542:549]*/, v[2:9] /*v[514:521]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[38:45] /*v[550:557]*/, v[2:9] /*v[514:521]*/, v[2:9]
	s_set_vgpr_msb 0xa00
	s_cbranch_scc1 .LBB0_6
.LBB0_7:
	s_set_vgpr_msb 12
	v_or_b32_e32 v1, s82, v53 /*v821*/
	s_mul_i32 s4, s67, s85
	s_load_b64 s[0:1], s[0:1], 0x110 nv
	s_add_co_i32 s83, s83, s4
	s_mov_b32 s3, 0
	s_wait_loadcnt 0x3e
	v_mul_lo_u32 v189, v1, s67
	s_mov_b32 s2, 0x800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v189, v189, s83, 7
	v_or_b32_e32 v194, v189, v52 /*v820*/
	s_set_vgpr_msb 0xc01
	v_or_b32_e32 v186, 1, v1
	s_wait_kmcnt 0x0
	v_cvt_pk_bf16_f32 v187, v242 /*v498*/, s0
	v_lshlrev_b32_e32 v194, 2, v194
	v_cvt_pk_bf16_f32 v188, v243 /*v499*/, s0
	v_cvt_pk_bf16_f32 v191, v244 /*v500*/, s0
	v_mul_lo_u32 v186, s67, v186
	v_cvt_pk_bf16_f32 v199, v246 /*v502*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v187, v194, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v201, v247 /*v503*/, s0
	v_cvt_pk_bf16_f32 v202, v249 /*v505*/, s0
	v_cvt_pk_bf16_f32 v203, v250 /*v506*/, s0
	v_cvt_pk_bf16_f32 v207, v236 /*v492*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v178, v178, s0
	v_add_lshl_u32 v186, s83, v186, 7
	v_cvt_pk_bf16_f32 v179, v179, s0
	v_cvt_pk_bf16_f32 v180, v180, s0
	v_cvt_pk_bf16_f32 v181, v181, s0
	v_cvt_pk_bf16_f32 v184, v184, s0
	s_set_vgpr_msb 12
	v_or_b32_e32 v195, v186, v52 /*v820*/
	v_cvt_pk_bf16_f32 v170, v170, s0
	v_cvt_pk_bf16_f32 v171, v171, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v190, 2, v1
	v_cvt_pk_bf16_f32 v172, v172, s0
	v_lshlrev_b32_e32 v195, 2, v195
	v_cvt_pk_bf16_f32 v162, v162, s0
	v_cvt_pk_bf16_f32 v164, v164, s0
	v_mul_lo_u32 v190, v190, s67
	v_cvt_pk_bf16_f32 v163, v163, s0
	buffer_store_b16 v188, v195, s[0:3], null offen
	v_cvt_pk_bf16_f32 v165, v165, s0
	v_cvt_pk_bf16_f32 v154, v154, s0
	v_cvt_pk_bf16_f32 v157, v157, s0
	v_cvt_pk_bf16_f32 v158, v158, s0
	v_cvt_pk_bf16_f32 v155, v155, s0
	v_add_lshl_u32 v190, s83, v190, 7
	v_cvt_pk_bf16_f32 v156, v156, s0
	v_cvt_pk_bf16_f32 v146, v146, s0
	v_cvt_pk_bf16_f32 v147, v147, s0
	v_cvt_pk_bf16_f32 v149, v149, s0
	s_set_vgpr_msb 12
	v_or_b32_e32 v196, v190, v52 /*v820*/
	v_cvt_pk_bf16_f32 v148, v148, s0
	v_cvt_pk_bf16_f32 v150, v150, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v192, 3, v1
	v_cvt_pk_bf16_f32 v138, v138, s0
	v_lshlrev_b32_e32 v187, 2, v196
	v_cvt_pk_bf16_f32 v139, v139, s0
	v_cvt_pk_bf16_f32 v140, v140, s0
	v_mul_lo_u32 v192, v192, s67
	v_cvt_pk_bf16_f32 v130, v130, s0
	buffer_store_b16 v191, v187, s[0:3], null offen
	v_cvt_pk_bf16_f32 v131, v131, s0
	v_cvt_pk_bf16_f32 v122, v122, s0
	v_cvt_pk_bf16_f32 v123, v123, s0
	v_cvt_pk_bf16_f32 v124, v124, s0
	v_cvt_pk_bf16_f32 v125, v125, s0
	v_add_lshl_u32 v192, s83, v192, 7
	v_cvt_pk_bf16_f32 v126, v126, s0
	v_cvt_pk_bf16_f32 v127, v127, s0
	v_cvt_pk_bf16_f32 v129, v129, s0
	v_cvt_pk_bf16_f32 v114, v114, s0
	s_set_vgpr_msb 12
	v_or_b32_e32 v188, v192, v52 /*v820*/
	v_cvt_pk_bf16_f32 v115, v115, s0
	v_cvt_pk_bf16_f32 v116, v116, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v193, 4, v1
	v_cvt_pk_bf16_f32 v117, v117, s0
	v_lshlrev_b32_e32 v188, 2, v188
	v_cvt_pk_bf16_f32 v120, v120, s0
	v_cvt_pk_bf16_f32 v106, v106, s0
	v_mul_lo_u32 v193, v193, s67
	v_cvt_pk_bf16_f32 v107, v107, s0
	v_cvt_pk_bf16_f32 v108, v108, s0
	v_cvt_pk_bf16_f32 v98, v98, s0
	v_cvt_pk_bf16_f32 v100, v100, s0
	v_cvt_pk_bf16_f32 v99, v99, s0
	v_cvt_pk_bf16_f32 v101, v101, s0
	v_cvt_pk_bf16_f32 v90, v90, s0
	v_add_lshl_u32 v193, s83, v193, 7
	v_cvt_pk_bf16_f32 v93, v93, s0
	v_cvt_pk_bf16_f32 v94, v94, s0
	v_cvt_pk_bf16_f32 v91, v91, s0
	v_cvt_pk_bf16_f32 v92, v92, s0
	s_set_vgpr_msb 12
	v_or_b32_e32 v198, v193, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v197, 5, v1
	v_cvt_pk_bf16_f32 v82, v82, s0
	v_cvt_pk_bf16_f32 v83, v83, s0
	v_cvt_pk_bf16_f32 v85, v85, s0
	v_lshlrev_b32_e32 v198, 2, v198
	v_mul_lo_u32 v196, v197, s67
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v197, v245 /*v501*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v197, v188, s[0:3], null offen
	buffer_store_b16 v199, v198, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v248 /*v504*/, s0
	v_add_lshl_u32 v196, s83, v196, 7
	s_set_vgpr_msb 0x10c
	v_cvt_pk_bf16_f32 v84, v84, s0
	v_cvt_pk_bf16_f32 v86, v86, s0
	v_cvt_pk_bf16_f32 v74, v74, s0
	v_cvt_pk_bf16_f32 v75, v75, s0
	v_or_b32_e32 v200, v196, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v191, 6, v1
	v_or_b32_e32 v196, v196, v0
	v_cvt_pk_bf16_f32 v76, v76, s0
	v_cvt_pk_bf16_f32 v66, v66, s0
	v_lshlrev_b32_e32 v200, 2, v200
	v_mul_lo_u32 v191, v191, s67
	v_cvt_pk_bf16_f32 v67, v67, s0
	v_cvt_pk_bf16_f32 v58, v58, s0
	v_cvt_pk_bf16_f32 v59, v59, s0
	v_or_b32_e32 v186, v186, v0
	v_or_b32_e32 v1, 7, v1
	buffer_store_b16 v201, v200, s[0:3], null offen
	v_cvt_pk_bf16_f32 v60, v60, s0
	v_add_lshl_u32 v191, s83, v191, 7
	v_lshlrev_b32_e32 v186, 2, v186
	v_mul_lo_u32 v1, v1, s67
	v_cvt_pk_bf16_f32 v61, v61, s0
	v_cvt_pk_bf16_f32 v62, v62, s0
	s_set_vgpr_msb 12
	v_or_b32_e32 v197, v191, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v189, v189, v0
	v_or_b32_e32 v191, v191, v0
	v_cvt_pk_bf16_f32 v63, v63, s0
	v_cvt_pk_bf16_f32 v65, v65, s0
	v_lshlrev_b32_e32 v197, 2, v197
	v_lshlrev_b32_e32 v189, 2, v189
	v_add_lshl_u32 v1, s83, v1, 7
	v_cvt_pk_bf16_f32 v50, v50, s0
	v_cvt_pk_bf16_f32 v51, v51, s0
	v_cvt_pk_bf16_f32 v52, v52, s0
	v_cvt_pk_bf16_f32 v53, v53, s0
	s_set_vgpr_msb 12
	v_or_b32_e32 v201, v1, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v1, v1, v0
	v_or_b32_e32 v190, v190, v0
	v_or_b32_e32 v193, v193, v0
	v_or_b32_e32 v204, 64, v189
	v_lshlrev_b32_e32 v201, 2, v201
	v_dual_lshlrev_b32 v1, 2, v1 :: v_dual_bitop2_b32 v192, v192, v0 bitop3:0x54
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshlrev_b32 v193, 2, v193 :: v_dual_lshlrev_b32 v190, 2, v190
	s_clause 0x1
	buffer_store_b16 v199, v197, s[0:3], null offen
	buffer_store_b16 v202, v201, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v192, 2, v192 :: v_dual_bitop2_b32 v202, 64, v186 bitop3:0x54
	buffer_store_b16 v203, v204, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v251 /*v507*/, s0
	v_cvt_pk_bf16_f32 v203, v252 /*v508*/, s0
	v_dual_lshlrev_b32 v196, 2, v196 :: v_dual_bitop2_b32 v205, 64, v190 bitop3:0x54
	v_cvt_pk_bf16_f32 v204, v253 /*v509*/, s0
	v_dual_lshlrev_b32 v191, 2, v191 :: v_dual_bitop2_b32 v206, 64, v192 bitop3:0x54
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v199, v202, s[0:3], null offen
	buffer_store_b16 v203, v205, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v254 /*v510*/, s0
	v_or_b32_e32 v203, 64, v193
	s_set_vgpr_msb 0x102
	v_cvt_pk_bf16_f32 v205, v0 /*v512*/, s0
	s_set_vgpr_msb 0x200
	buffer_store_b16 v204, v206, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v202, v255 /*v511*/, s0
	v_or_b32_e32 v204, 64, v196
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v199, v203, s[0:3], null offen
	buffer_store_b16 v202, v204, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v199, 64, v191
	s_wait_xcnt 0x0
	v_mov_b16_e64 v202.l, v205.l
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v203, v1 /*v513*/, s0
	v_or_b32_e32 v205, 64, v1
	s_set_vgpr_msb 0x201
	v_cvt_pk_bf16_f32 v206, v235 /*v491*/, s0
	v_cvt_pk_bf16_f32 v204, v234 /*v490*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x2
	buffer_store_b16 v202, v199, s[0:3], null offen
	buffer_store_b16 v203, v205, s[0:3], null offen
	buffer_store_b16 v204, v194, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_mov_b16_e64 v199.l, v206.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v203, v237 /*v493*/, s0
	v_cvt_pk_bf16_f32 v204, v240 /*v496*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v202.l, v207.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v205, v241 /*v497*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v199, v195, s[0:3], null offen offset:128
	buffer_store_b16 v202, v187, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v238 /*v494*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v203, v188, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v203.l, v204.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v202, v239 /*v495*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v204.l, v205.l
	s_clause 0x1
	buffer_store_b16 v199, v198, s[0:3], null offen offset:128
	buffer_store_b16 v202, v200, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v226 /*v482*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v203, v197, s[0:3], null offen offset:128
	buffer_store_b16 v204, v201, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_or_b32_e32 v202, 0xc0, v189
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v204, v228 /*v484*/, s0
	v_or_b32_e32 v207, 0xc0, v190
	v_cvt_pk_bf16_f32 v203, v227 /*v483*/, s0
	v_or_b32_e32 v205, 0xc0, v186
	v_cvt_pk_bf16_f32 v206, v229 /*v485*/, s0
	v_or_b32_e32 v208, 0xc0, v192
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v199, v202, s[0:3], null offen
	buffer_store_b16 v203, v205, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v230 /*v486*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v204, v207, s[0:3], null offen
	buffer_store_b16 v206, v208, s[0:3], null offen
	s_wait_xcnt 0x2
	v_or_b32_e32 v203, 0xc0, v193
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v204, v232 /*v488*/, s0
	v_or_b32_e32 v206, 0xc0, v191
	v_cvt_pk_bf16_f32 v207, v233 /*v489*/, s0
	v_cvt_pk_bf16_f32 v202, v231 /*v487*/, s0
	v_or_b32_e32 v205, 0xc0, v196
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v199, v203, s[0:3], null offen
	buffer_store_b16 v202, v205, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v199, 0xc0, v1
	s_wait_xcnt 0x0
	v_mov_b16_e64 v202.l, v207.l
	buffer_store_b16 v204, v206, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v203, v218 /*v474*/, s0
	v_cvt_pk_bf16_f32 v204, v219 /*v475*/, s0
	v_cvt_pk_bf16_f32 v205, v220 /*v476*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v202, v199, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v221 /*v477*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v202.l, v203.l
	v_mov_b16_e64 v203.l, v204.l
	v_mov_b16_e64 v204.l, v205.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v205, v222 /*v478*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x3
	buffer_store_b16 v202, v194, s[0:3], null offen offset:256
	buffer_store_b16 v203, v195, s[0:3], null offen offset:256
	buffer_store_b16 v204, v187, s[0:3], null offen offset:256
	buffer_store_b16 v199, v188, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v223 /*v479*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v202.l, v205.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v204, v225 /*v481*/, s0
	v_cvt_pk_bf16_f32 v203, v224 /*v480*/, s0
	v_or_b32_e32 v205, 0x140, v189
	v_or_b32_e32 v207, 0x140, v192
	s_set_vgpr_msb 0x100
	buffer_store_b16 v202, v198, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v202, v210 /*v466*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v199, v200, s[0:3], null offen offset:256
	buffer_store_b16 v203, v197, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v211 /*v467*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v204, v201, s[0:3], null offen offset:256
	buffer_store_b16 v202, v205, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v202, 0x140, v186
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v204, v213 /*v469*/, s0
	v_cvt_pk_bf16_f32 v203, v212 /*v468*/, s0
	v_or_b32_e32 v205, 0x140, v190
	v_cvt_pk_bf16_f32 v206, v214 /*v470*/, s0
	v_or_b32_e32 v208, 0x140, v193
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v199, v202, s[0:3], null offen
	buffer_store_b16 v203, v205, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v215 /*v471*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v204, v207, s[0:3], null offen
	buffer_store_b16 v206, v208, s[0:3], null offen
	v_or_b32_e32 v202, 0x140, v196
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v206, v202 /*v458*/, s0
	v_cvt_pk_bf16_f32 v203, v216 /*v472*/, s0
	v_cvt_pk_bf16_f32 v204, v217 /*v473*/, s0
	v_or_b32_e32 v205, 0x140, v191
	v_or_b32_e32 v207, 0x140, v1
	s_set_vgpr_msb 0x100
	buffer_store_b16 v199, v202, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v199.l, v206.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v202, v203 /*v459*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v203, v205, s[0:3], null offen
	buffer_store_b16 v204, v207, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v203, v204 /*v460*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v199, v194, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v199.l, v202.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v202, v206 /*v462*/, s0
	v_cvt_pk_bf16_f32 v204, v207 /*v463*/, s0
	v_cvt_pk_bf16_f32 v194, v205 /*v461*/, s0
	v_or_b32_e32 v189, 0x1c0, v189
	s_set_vgpr_msb 0x100
	buffer_store_b16 v199, v195, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v195.l, v202.l
	buffer_store_b16 v203, v187, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v187.l, v204.l
	buffer_store_b16 v194, v188, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v188, v208 /*v464*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v195, v198, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v194, v209 /*v465*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v187, v200, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v187, v194 /*v450*/, s0
	v_cvt_pk_bf16_f32 v195, v195 /*v451*/, s0
	v_or_b32_e32 v186, 0x1c0, v186
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v188, v197, s[0:3], null offen offset:384
	buffer_store_b16 v194, v201, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v188, v196 /*v452*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v187, v189, s[0:3], null offen
	buffer_store_b16 v195, v186, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v187, 0x1c0, v190
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v189, v198 /*v454*/, s0
	v_or_b32_e32 v190, 0x1c0, v192
	v_or_b32_e32 v192, 0x1c0, v193
	v_cvt_pk_bf16_f32 v186, v197 /*v453*/, s0
	s_set_vgpr_msb 0x10c
	v_or_b32_e32 v193, s81, v53 /*v821*/
	s_clause 0x1
	buffer_store_b16 v188, v187, s[0:3], null offen
	buffer_store_b16 v186, v190, s[0:3], null offen
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v186, v199 /*v455*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v189, v192, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v188, v200 /*v456*/, s0
	v_or_b32_e32 v189, 0x1c0, v196
	v_or_b32_e32 v190, 0x1c0, v191
	v_mul_lo_u32 v187, s67, v193
	v_or_b32_e32 v1, 0x1c0, v1
	v_or_b32_e32 v196, 5, v193
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v186, v189, s[0:3], null offen
	buffer_store_b16 v188, v190, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v190, 2, v193
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v186, v201 /*v457*/, s0
	v_cvt_pk_bf16_f32 v194, v68 /*v324*/, s0
	v_add_lshl_u32 v187, s83, v187, 7
	v_mul_lo_u32 v196, s67, v196
	v_mul_lo_u32 v190, s67, v190
	s_set_vgpr_msb 0x100
	buffer_store_b16 v186, v1, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v186, v67 /*v323*/, s0
	s_set_vgpr_msb 0x10c
	v_or_b32_e32 v188, v187, v52 /*v820*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v197, v69 /*v325*/, s0
	v_cvt_pk_bf16_f32 v199, v70 /*v326*/, s0
	v_cvt_pk_bf16_f32 v201, v71 /*v327*/, s0
	v_add_lshl_u32 v196, s83, v196, 7
	v_add_lshl_u32 v190, s83, v190, 7
	v_lshlrev_b32_e32 v188, 2, v188
	v_cvt_pk_bf16_f32 v202, v73 /*v329*/, s0
	s_set_vgpr_msb 0x10c
	v_cvt_pk_bf16_f32 v56, v56, s0
	v_or_b32_e32 v200, v196, v52 /*v820*/
	v_or_b32_e32 v195, v190, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v190, v190, v0
	v_or_b32_e32 v191, 1, v193
	v_cvt_pk_bf16_f32 v42, v42, s0
	v_lshlrev_b32_e32 v200, 2, v200
	v_cvt_pk_bf16_f32 v43, v43, s0
	v_lshlrev_b32_e32 v190, 2, v190
	v_mul_lo_u32 v189, v191, s67
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v191, v66 /*v322*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v44, v44, s0
	v_cvt_pk_bf16_f32 v34, v34, s0
	v_cvt_pk_bf16_f32 v36, v36, s0
	v_cvt_pk_bf16_f32 v35, v35, s0
	buffer_store_b16 v191, v188, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v191, 4, v193
	v_add_lshl_u32 v189, v189, s83, 7
	v_cvt_pk_bf16_f32 v37, v37, s0
	v_cvt_pk_bf16_f32 v26, v26, s0
	v_cvt_pk_bf16_f32 v29, v29, s0
	v_mul_lo_u32 v191, v191, s67
	s_set_vgpr_msb 12
	v_or_b32_e32 v1, v189, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v189, v189, v0
	v_cvt_pk_bf16_f32 v30, v30, s0
	v_cvt_pk_bf16_f32 v27, v27, s0
	v_cvt_pk_bf16_f32 v28, v28, s0
	v_lshlrev_b32_e32 v1, 2, v1
	v_lshlrev_b32_e32 v189, 2, v189
	v_add_lshl_u32 v191, v191, s83, 7
	v_cvt_pk_bf16_f32 v18, v18, s0
	v_cvt_pk_bf16_f32 v19, v19, s0
	buffer_store_b16 v186, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v186, 2, v195
	s_set_vgpr_msb 12
	v_or_b32_e32 v198, v191, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v191, v191, v0
	v_or_b32_e32 v192, 3, v193
	v_cvt_pk_bf16_f32 v21, v21, s0
	buffer_store_b16 v194, v186, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v194, 6, v193
	v_or_b32_e32 v193, 7, v193
	v_mul_lo_u32 v192, v192, s67
	v_lshlrev_b32_e32 v198, 2, v198
	v_cvt_pk_bf16_f32 v22, v22, s0
	v_mul_lo_u32 v194, v194, s67
	v_mul_lo_u32 v193, v193, s67
	v_cvt_pk_bf16_f32 v23, v23, s0
	v_cvt_pk_bf16_f32 v12, v12, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_add_lshl_u32 v192, v192, s83, 7
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_add_lshl_u32 v194, v194, s83, 7
	v_add_lshl_u32 v193, v193, s83, 7
	s_set_vgpr_msb 12
	v_or_b32_e32 v195, v192, v52 /*v820*/
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_cvt_pk_bf16_f32 v4, v4, s0
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v195, 2, v195
	s_clause 0x1
	buffer_store_b16 v197, v195, s[0:3], null offen
	buffer_store_b16 v199, v198, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v197, v194, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v187, v187, v0
	buffer_store_b16 v201, v200, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v201, v193, v52 /*v820*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v199, v72 /*v328*/, s0
	v_dual_lshlrev_b32 v197, 2, v197 :: v_dual_lshlrev_b32 v187, 2, v187
	s_set_vgpr_msb 0x100
	v_dual_lshlrev_b32 v201, 2, v201 :: v_dual_bitop2_b32 v193, v193, v0 bitop3:0x54
	s_clause 0x1
	buffer_store_b16 v199, v197, s[0:3], null offen
	buffer_store_b16 v202, v201, s[0:3], null offen
	v_or_b32_e32 v203, 64, v187
	s_wait_xcnt 0x1
	v_or_b32_e32 v199, 64, v190
	buffer_store_b16 v178, v203, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v178, v192, v0
	v_or_b32_e32 v192, 64, v189
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v178, 2, v178
	buffer_store_b16 v179, v192, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v179, v196, v0
	buffer_store_b16 v180, v199, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v180, 2, v191 :: v_dual_bitop2_b32 v202, 64, v178 bitop3:0x54
	v_dual_lshlrev_b32 v179, 2, v179 :: v_dual_bitop2_b32 v191, v194, v0 bitop3:0x54
	buffer_store_b16 v181, v202, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v181, v182, s0
	v_cvt_pk_bf16_f32 v182, v183, s0
	v_or_b32_e32 v183, 64, v180
	v_or_b32_e32 v192, 64, v179
	v_lshlrev_b32_e32 v191, 2, v191
	s_clause 0x1
	buffer_store_b16 v181, v183, s[0:3], null offen
	buffer_store_b16 v182, v192, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v181, 2, v193 :: v_dual_bitop2_b32 v182, 64, v191 bitop3:0x54
	v_mov_b16_e64 v183.l, v184.l
	v_cvt_pk_bf16_f32 v184, v185, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v185, 64, v181
	s_clause 0x2
	buffer_store_b16 v183, v182, s[0:3], null offen
	buffer_store_b16 v184, v185, s[0:3], null offen
	buffer_store_b16 v170, v188, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v170, v173, s0
	v_cvt_pk_bf16_f32 v173, v176, s0
	s_clause 0x1
	buffer_store_b16 v171, v1, s[0:3], null offen offset:128
	buffer_store_b16 v172, v186, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v171, v174, s0
	v_cvt_pk_bf16_f32 v174, v177, s0
	buffer_store_b16 v170, v195, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v173.l
	v_cvt_pk_bf16_f32 v172, v175, s0
	s_clause 0x1
	buffer_store_b16 v171, v198, s[0:3], null offen offset:128
	buffer_store_b16 v172, v200, s[0:3], null offen offset:128
	v_mov_b16_e64 v173.l, v174.l
	s_clause 0x1
	buffer_store_b16 v170, v197, s[0:3], null offen offset:128
	buffer_store_b16 v173, v201, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v170, 0xc0, v187
	v_or_b32_e32 v172, 0xc0, v190
	v_or_b32_e32 v171, 0xc0, v189
	s_wait_xcnt 0x0
	v_or_b32_e32 v173, 0xc0, v178
	s_clause 0x1
	buffer_store_b16 v162, v170, s[0:3], null offen
	buffer_store_b16 v163, v171, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v162, v166, s0
	s_clause 0x1
	buffer_store_b16 v164, v172, s[0:3], null offen
	buffer_store_b16 v165, v173, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v164, 0xc0, v180
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v165, v168, s0
	v_cvt_pk_bf16_f32 v168, v169, s0
	v_cvt_pk_bf16_f32 v163, v167, s0
	v_or_b32_e32 v166, 0xc0, v179
	v_or_b32_e32 v167, 0xc0, v191
	s_clause 0x1
	buffer_store_b16 v162, v164, s[0:3], null offen
	buffer_store_b16 v163, v166, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v162, 0xc0, v181
	s_wait_xcnt 0x0
	v_mov_b16_e64 v163.l, v168.l
	buffer_store_b16 v165, v167, s[0:3], null offen
	s_clause 0x3
	buffer_store_b16 v163, v162, s[0:3], null offen
	buffer_store_b16 v154, v188, s[0:3], null offen offset:256
	buffer_store_b16 v155, v1, s[0:3], null offen offset:256
	buffer_store_b16 v156, v186, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e64 v154.l, v158.l
	buffer_store_b16 v157, v195, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v155, v159, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v156, v160, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v157, v161, s0
	buffer_store_b16 v154, v198, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v154.l, v155.l
	v_mov_b16_e64 v155.l, v156.l
	v_mov_b16_e64 v156.l, v157.l
	v_or_b32_e32 v157, 0x140, v187
	s_clause 0x3
	buffer_store_b16 v154, v200, s[0:3], null offen offset:256
	buffer_store_b16 v155, v197, s[0:3], null offen offset:256
	buffer_store_b16 v156, v201, s[0:3], null offen offset:256
	buffer_store_b16 v146, v157, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v146, 0x140, v189
	v_or_b32_e32 v155, 0x140, v178
	v_or_b32_e32 v154, 0x140, v190
	v_or_b32_e32 v156, 0x140, v180
	s_clause 0x1
	buffer_store_b16 v147, v146, s[0:3], null offen
	buffer_store_b16 v148, v154, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v146, v151, s0
	s_clause 0x1
	buffer_store_b16 v149, v155, s[0:3], null offen
	buffer_store_b16 v150, v156, s[0:3], null offen
	v_or_b32_e32 v147, 0x140, v179
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v148, v152, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v149, v153, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v150, 0x140, v191
	v_or_b32_e32 v151, 0x140, v181
	buffer_store_b16 v146, v147, s[0:3], null offen
	s_clause 0x2
	buffer_store_b16 v148, v150, s[0:3], null offen
	buffer_store_b16 v149, v151, s[0:3], null offen
	buffer_store_b16 v138, v188, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v138, v141, s0
	v_cvt_pk_bf16_f32 v141, v142, s0
	v_cvt_pk_bf16_f32 v142, v143, s0
	s_clause 0x2
	buffer_store_b16 v139, v1, s[0:3], null offen offset:384
	buffer_store_b16 v140, v186, s[0:3], null offen offset:384
	buffer_store_b16 v138, v195, s[0:3], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e64 v1.l, v141.l
	v_mov_b16_e64 v139.l, v142.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v138, v144, s0
	v_or_b32_e32 v140, 0x1c0, v189
	s_clause 0x1
	buffer_store_b16 v1, v198, s[0:3], null offen offset:384
	buffer_store_b16 v139, v200, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_or_b32_e32 v139, 0x1c0, v187
	v_cvt_pk_bf16_f32 v1, v145, s0
	s_clause 0x1
	buffer_store_b16 v138, v197, s[0:3], null offen offset:384
	buffer_store_b16 v1, v201, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v132, s0
	s_clause 0x1
	buffer_store_b16 v130, v139, s[0:3], null offen
	buffer_store_b16 v131, v140, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v131, 0x1c0, v190
	s_set_vgpr_msb 12
	v_or_b32_e32 v138, s80, v53 /*v821*/
	v_cvt_pk_bf16_f32 v130, v133, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v133, 0x1c0, v178
	v_cvt_pk_bf16_f32 v132, v134, s0
	v_or_b32_e32 v134, 0x1c0, v180
	s_clause 0x1
	buffer_store_b16 v1, v131, s[0:3], null offen
	buffer_store_b16 v130, v133, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v130, v138, s67
	v_cvt_pk_bf16_f32 v1, v135, s0
	buffer_store_b16 v132, v134, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v132, 0x1c0, v179
	v_or_b32_e32 v134, 1, v138
	v_cvt_pk_bf16_f32 v131, v136, s0
	v_or_b32_e32 v133, 0x1c0, v191
	v_or_b32_e32 v135, 3, v138
	v_add_lshl_u32 v130, s83, v130, 7
	buffer_store_b16 v1, v132, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v132, v134, s67
	v_or_b32_e32 v134, 2, v138
	buffer_store_b16 v131, v133, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v131, v130, v52 /*v820*/
	v_cvt_pk_bf16_f32 v1, v137, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v133, 0x1c0, v181
	v_mul_lo_u32 v134, v134, s67
	v_dual_lshlrev_b32 v131, 2, v131 :: v_dual_bitop2_b32 v136, 5, v138 bitop3:0x54
	v_add_lshl_u32 v132, v132, s83, 7
	buffer_store_b16 v1, v133, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v133, v135, s67
	v_mul_lo_u32 v136, v136, s67
	buffer_store_b16 v122, v131, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v1, v132, v52 /*v820*/
	v_add_lshl_u32 v122, v134, s83, 7
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v134, 4, v138
	v_or_b32_e32 v132, v132, v0
	v_lshlrev_b32_e32 v1, 2, v1
	s_set_vgpr_msb 12
	v_or_b32_e32 v135, v122, v52 /*v820*/
	v_mul_lo_u32 v134, v134, s67
	v_add_lshl_u32 v133, v133, s83, 7
	v_add_lshl_u32 v136, v136, s83, 7
	buffer_store_b16 v123, v1, s[0:3], null offen
	s_set_vgpr_msb 0xc00
	v_dual_lshlrev_b32 v123, 2, v135 :: v_dual_bitop2_b32 v122, v122, v0 bitop3:0x54
	s_set_vgpr_msb 12
	v_or_b32_e32 v135, v133, v52 /*v820*/
	v_or_b32_e32 v139, v136, v52 /*v820*/
	v_add_lshl_u32 v134, v134, s83, 7
	buffer_store_b16 v124, v123, s[0:3], null offen
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v124, 6, v138
	v_dual_lshlrev_b32 v135, 2, v135 :: v_dual_bitop2_b32 v138, 7, v138 bitop3:0x54
	s_set_vgpr_msb 12
	v_or_b32_e32 v137, v134, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v139, 2, v139
	v_mul_lo_u32 v124, v124, s67
	v_mul_lo_u32 v138, v138, s67
	v_dual_lshlrev_b32 v122, 2, v122 :: v_dual_lshlrev_b32 v137, 2, v137
	s_clause 0x1
	buffer_store_b16 v125, v135, s[0:3], null offen
	buffer_store_b16 v126, v137, s[0:3], null offen
	v_add_lshl_u32 v124, v124, s83, 7
	s_wait_xcnt 0x0
	v_add_lshl_u32 v126, v138, s83, 7
	buffer_store_b16 v127, v139, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v127, v128, s0
	v_or_b32_e32 v128, v130, v0
	s_set_vgpr_msb 12
	v_or_b32_e32 v125, v124, v52 /*v820*/
	v_or_b32_e32 v130, v126, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v134, v134, v0
	v_or_b32_e32 v124, v124, v0
	v_dual_lshlrev_b32 v128, 2, v128 :: v_dual_lshlrev_b32 v125, 2, v125
	v_lshlrev_b32_e32 v130, 2, v130
	s_clause 0x1
	buffer_store_b16 v127, v125, s[0:3], null offen
	buffer_store_b16 v129, v130, s[0:3], null offen
	v_or_b32_e32 v138, 64, v128
	s_wait_xcnt 0x1
	v_dual_lshlrev_b32 v127, 2, v132 :: v_dual_bitop2_b32 v132, 64, v122 bitop3:0x54
	v_or_b32_e32 v126, v126, v0
	v_lshlrev_b32_e32 v124, 2, v124
	buffer_store_b16 v114, v138, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v114, v133, v0
	v_or_b32_e32 v129, 64, v127
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v114, 2, v114
	buffer_store_b16 v115, v129, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v115, v136, v0
	buffer_store_b16 v116, v132, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v116, 2, v134
	v_or_b32_e32 v133, 64, v114
	v_lshlrev_b32_e32 v115, 2, v115
	buffer_store_b16 v117, v133, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v117, v118, s0
	v_cvt_pk_bf16_f32 v118, v119, s0
	v_or_b32_e32 v119, 64, v116
	v_or_b32_e32 v129, 64, v115
	s_clause 0x1
	buffer_store_b16 v117, v119, s[0:3], null offen
	buffer_store_b16 v118, v129, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v117, 2, v126 :: v_dual_bitop2_b32 v118, 64, v124 bitop3:0x54
	v_mov_b16_e32 v119.l, v120.l
	v_cvt_pk_bf16_f32 v120, v121, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v121, 64, v117
	s_clause 0x2
	buffer_store_b16 v119, v118, s[0:3], null offen
	buffer_store_b16 v120, v121, s[0:3], null offen
	buffer_store_b16 v106, v131, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v106, v109, s0
	v_cvt_pk_bf16_f32 v109, v112, s0
	s_clause 0x1
	buffer_store_b16 v107, v1, s[0:3], null offen offset:128
	buffer_store_b16 v108, v123, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v107, v110, s0
	v_cvt_pk_bf16_f32 v110, v113, s0
	buffer_store_b16 v106, v135, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v106.l, v109.l
	v_cvt_pk_bf16_f32 v108, v111, s0
	s_clause 0x1
	buffer_store_b16 v107, v137, s[0:3], null offen offset:128
	buffer_store_b16 v108, v139, s[0:3], null offen offset:128
	v_mov_b16_e32 v109.l, v110.l
	s_clause 0x1
	buffer_store_b16 v106, v125, s[0:3], null offen offset:128
	buffer_store_b16 v109, v130, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v106, 0xc0, v128
	v_or_b32_e32 v108, 0xc0, v122
	v_or_b32_e32 v107, 0xc0, v127
	s_wait_xcnt 0x0
	v_or_b32_e32 v109, 0xc0, v114
	s_clause 0x1
	buffer_store_b16 v98, v106, s[0:3], null offen
	buffer_store_b16 v99, v107, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v98, v102, s0
	s_clause 0x1
	buffer_store_b16 v100, v108, s[0:3], null offen
	buffer_store_b16 v101, v109, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v100, 0xc0, v116
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v101, v104, s0
	v_cvt_pk_bf16_f32 v104, v105, s0
	v_cvt_pk_bf16_f32 v99, v103, s0
	v_or_b32_e32 v102, 0xc0, v115
	v_or_b32_e32 v103, 0xc0, v124
	s_clause 0x1
	buffer_store_b16 v98, v100, s[0:3], null offen
	buffer_store_b16 v99, v102, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v98, 0xc0, v117
	s_wait_xcnt 0x0
	v_mov_b16_e32 v99.l, v104.l
	buffer_store_b16 v101, v103, s[0:3], null offen
	s_clause 0x3
	buffer_store_b16 v99, v98, s[0:3], null offen
	buffer_store_b16 v90, v131, s[0:3], null offen offset:256
	buffer_store_b16 v91, v1, s[0:3], null offen offset:256
	buffer_store_b16 v92, v123, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e32 v90.l, v94.l
	buffer_store_b16 v93, v135, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v91, v95, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v92, v96, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v93, v97, s0
	buffer_store_b16 v90, v137, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e32 v90.l, v91.l
	v_mov_b16_e32 v91.l, v92.l
	v_mov_b16_e32 v92.l, v93.l
	v_or_b32_e32 v93, 0x140, v128
	s_clause 0x3
	buffer_store_b16 v90, v139, s[0:3], null offen offset:256
	buffer_store_b16 v91, v125, s[0:3], null offen offset:256
	buffer_store_b16 v92, v130, s[0:3], null offen offset:256
	buffer_store_b16 v82, v93, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 0x140, v127
	v_or_b32_e32 v91, 0x140, v114
	v_or_b32_e32 v90, 0x140, v122
	v_or_b32_e32 v92, 0x140, v116
	s_clause 0x1
	buffer_store_b16 v83, v82, s[0:3], null offen
	buffer_store_b16 v84, v90, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v82, v87, s0
	s_clause 0x1
	buffer_store_b16 v85, v91, s[0:3], null offen
	buffer_store_b16 v86, v92, s[0:3], null offen
	v_or_b32_e32 v83, 0x140, v115
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v84, v88, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v85, v89, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v86, 0x140, v124
	v_or_b32_e32 v87, 0x140, v117
	buffer_store_b16 v82, v83, s[0:3], null offen
	s_clause 0x2
	buffer_store_b16 v84, v86, s[0:3], null offen
	buffer_store_b16 v85, v87, s[0:3], null offen
	buffer_store_b16 v74, v131, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v74, v77, s0
	v_cvt_pk_bf16_f32 v77, v78, s0
	v_cvt_pk_bf16_f32 v78, v79, s0
	s_clause 0x2
	buffer_store_b16 v75, v1, s[0:3], null offen offset:384
	buffer_store_b16 v76, v123, s[0:3], null offen offset:384
	buffer_store_b16 v74, v135, s[0:3], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e32 v1.l, v77.l
	v_mov_b16_e32 v75.l, v78.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v74, v80, s0
	v_or_b32_e32 v76, 0x1c0, v127
	s_clause 0x1
	buffer_store_b16 v1, v137, s[0:3], null offen offset:384
	buffer_store_b16 v75, v139, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_or_b32_e32 v75, 0x1c0, v128
	v_cvt_pk_bf16_f32 v1, v81, s0
	s_clause 0x1
	buffer_store_b16 v74, v125, s[0:3], null offen offset:384
	buffer_store_b16 v1, v130, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v68, s0
	s_clause 0x1
	buffer_store_b16 v66, v75, s[0:3], null offen
	buffer_store_b16 v67, v76, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v67, 0x1c0, v122
	s_set_vgpr_msb 12
	v_or_b32_e32 v74, s69, v53 /*v821*/
	v_cvt_pk_bf16_f32 v66, v69, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v69, 0x1c0, v114
	v_cvt_pk_bf16_f32 v68, v70, s0
	v_or_b32_e32 v70, 0x1c0, v116
	s_clause 0x1
	buffer_store_b16 v1, v67, s[0:3], null offen
	buffer_store_b16 v66, v69, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v66, v74, s67
	v_cvt_pk_bf16_f32 v1, v71, s0
	buffer_store_b16 v68, v70, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v68, 0x1c0, v115
	v_or_b32_e32 v70, 1, v74
	v_cvt_pk_bf16_f32 v67, v72, s0
	v_or_b32_e32 v69, 0x1c0, v124
	v_or_b32_e32 v71, 3, v74
	v_add_lshl_u32 v66, s83, v66, 7
	buffer_store_b16 v1, v68, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v68, v70, s67
	v_or_b32_e32 v70, 2, v74
	buffer_store_b16 v67, v69, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v67, v66, v52 /*v820*/
	v_cvt_pk_bf16_f32 v1, v73, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v69, 0x1c0, v117
	v_mul_lo_u32 v70, v70, s67
	v_dual_lshlrev_b32 v67, 2, v67 :: v_dual_bitop2_b32 v72, 5, v74 bitop3:0x54
	v_add_lshl_u32 v68, v68, s83, 7
	buffer_store_b16 v1, v69, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v69, v71, s67
	v_mul_lo_u32 v72, v72, s67
	buffer_store_b16 v58, v67, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v1, v68, v52 /*v820*/
	v_add_lshl_u32 v58, v70, s83, 7
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v70, 4, v74
	v_or_b32_e32 v68, v68, v0
	v_lshlrev_b32_e32 v1, 2, v1
	s_set_vgpr_msb 12
	v_or_b32_e32 v71, v58, v52 /*v820*/
	v_mul_lo_u32 v70, v70, s67
	v_add_lshl_u32 v69, v69, s83, 7
	v_add_lshl_u32 v72, v72, s83, 7
	buffer_store_b16 v59, v1, s[0:3], null offen
	s_set_vgpr_msb 0xc00
	v_dual_lshlrev_b32 v59, 2, v71 :: v_dual_bitop2_b32 v58, v58, v0 bitop3:0x54
	s_set_vgpr_msb 12
	v_or_b32_e32 v71, v69, v52 /*v820*/
	v_or_b32_e32 v75, v72, v52 /*v820*/
	v_add_lshl_u32 v70, v70, s83, 7
	buffer_store_b16 v60, v59, s[0:3], null offen
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v60, 6, v74
	v_dual_lshlrev_b32 v71, 2, v71 :: v_dual_bitop2_b32 v74, 7, v74 bitop3:0x54
	s_set_vgpr_msb 12
	v_or_b32_e32 v73, v70, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v75, 2, v75
	v_mul_lo_u32 v60, v60, s67
	v_mul_lo_u32 v74, v74, s67
	v_dual_lshlrev_b32 v58, 2, v58 :: v_dual_lshlrev_b32 v73, 2, v73
	s_clause 0x1
	buffer_store_b16 v61, v71, s[0:3], null offen
	buffer_store_b16 v62, v73, s[0:3], null offen
	v_add_lshl_u32 v60, v60, s83, 7
	s_wait_xcnt 0x0
	v_add_lshl_u32 v62, v74, s83, 7
	buffer_store_b16 v63, v75, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v63, v64, s0
	v_or_b32_e32 v64, v66, v0
	s_set_vgpr_msb 12
	v_or_b32_e32 v61, v60, v52 /*v820*/
	v_or_b32_e32 v66, v62, v52 /*v820*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v70, v70, v0
	v_or_b32_e32 v60, v60, v0
	v_dual_lshlrev_b32 v64, 2, v64 :: v_dual_lshlrev_b32 v61, 2, v61
	v_lshlrev_b32_e32 v66, 2, v66
	s_clause 0x1
	buffer_store_b16 v63, v61, s[0:3], null offen
	buffer_store_b16 v65, v66, s[0:3], null offen
	v_or_b32_e32 v74, 64, v64
	s_wait_xcnt 0x1
	v_dual_lshlrev_b32 v63, 2, v68 :: v_dual_bitop2_b32 v68, 64, v58 bitop3:0x54
	v_lshlrev_b32_e32 v60, 2, v60
	buffer_store_b16 v50, v74, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v50, v69, v0
	v_or_b32_e32 v65, 64, v63
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v50, 2, v50
	buffer_store_b16 v51, v65, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v51, v72, v0
	buffer_store_b16 v52, v68, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v52, 2, v70
	v_or_b32_e32 v69, 64, v50
	v_dual_lshlrev_b32 v51, 2, v51 :: v_dual_bitop2_b32 v0, v62, v0 bitop3:0x54
	buffer_store_b16 v53, v69, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v53, v54, s0
	v_cvt_pk_bf16_f32 v54, v55, s0
	v_or_b32_e32 v55, 64, v52
	v_dual_lshlrev_b32 v0, 2, v0 :: v_dual_bitop2_b32 v65, 64, v51 bitop3:0x54
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
	buffer_store_b16 v42, v67, s[0:3], null offen offset:128
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
	buffer_store_b16 v43, v73, s[0:3], null offen offset:128
	buffer_store_b16 v44, v75, s[0:3], null offen offset:128
	v_mov_b16_e32 v45.l, v46.l
	s_clause 0x1
	buffer_store_b16 v42, v61, s[0:3], null offen offset:128
	buffer_store_b16 v45, v66, s[0:3], null offen offset:128
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
	buffer_store_b16 v26, v67, s[0:3], null offen offset:256
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
	buffer_store_b16 v26, v73, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e32 v26.l, v27.l
	v_mov_b16_e32 v27.l, v28.l
	v_mov_b16_e32 v28.l, v29.l
	v_or_b32_e32 v29, 0x140, v64
	s_clause 0x3
	buffer_store_b16 v26, v75, s[0:3], null offen offset:256
	buffer_store_b16 v27, v61, s[0:3], null offen offset:256
	buffer_store_b16 v28, v66, s[0:3], null offen offset:256
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
	buffer_store_b16 v10, v67, s[0:3], null offen offset:384
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
	buffer_store_b16 v10, v73, s[0:3], null offen offset:384
	buffer_store_b16 v11, v75, s[0:3], null offen offset:384
	buffer_store_b16 v12, v61, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_or_b32_e32 v11, 0x1c0, v63
	v_or_b32_e32 v10, 0x1c0, v64
	s_wait_xcnt 0x0
	v_or_b32_e32 v12, 0x1c0, v58
	s_clause 0x1
	buffer_store_b16 v1, v66, s[0:3], null offen offset:384
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
		.amdhsa_next_free_vgpr 960
		.amdhsa_next_free_sgpr 87
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

	.set .Lk_dq_0.num_vgpr, 960
	.set .Lk_dq_0.num_agpr, 0
	.set .Lk_dq_0.numbered_sgpr, 87
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
    .sgpr_count:     89
    .sgpr_spill_count: 0
    .symbol:         k_dq_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     960
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
