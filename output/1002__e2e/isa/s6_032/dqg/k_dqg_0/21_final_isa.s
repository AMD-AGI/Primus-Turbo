	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	k_dqg_0
	.p2align	8
	.type	k_dqg_0,@function
k_dqg_0:
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	v_nop
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_bfe_u32 s2, ttmp6, 0x40010
	s_load_b256 s[56:63], s[0:1], 0x170 nv
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
	v_and_b32_e32 v209 /*v721*/, 15, v0
	s_cselect_b32 s5, ttmp9, s2
	s_cselect_b32 s2, s3, s4
	s_bfe_u32 s3, ttmp6, 0x40014
	s_lshr_b32 s4, ttmp7, 16
	s_add_co_i32 s3, s3, 1
	s_bfe_u32 s7, ttmp6, 0x40008
	s_mul_i32 s3, s4, s3
	s_clause 0x3
	s_load_b64 s[16:17], s[0:1], 0x0 nv
	s_load_b64 s[76:77], s[0:1], 0x30 nv
	s_load_b64 s[78:79], s[0:1], 0x60 nv
	s_load_b64 s[20:21], s[0:1], 0x90 nv
	s_add_co_i32 s7, s7, s3
	s_cmp_eq_u32 s6, 0
	s_set_vgpr_msb 0x8000
	v_lshrrev_b32_e32 v4, 4, v0
	s_cselect_b32 s3, s4, s7
	s_wait_kmcnt 0x0
	s_ashr_i32 s4, s57, 31
	s_mul_i32 s91, s57, s3
	s_lshr_b32 s4, s4, 26
	s_mov_b32 s71, 0
	s_add_co_i32 s4, s57, s4
	s_mov_b32 s18, 0x800000
	s_and_b32 s6, s4, 0xffffffc0
	s_ashr_i32 s4, s4, 6
	s_cmp_lg_u32 s57, s6
	s_mov_b32 s19, s71
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s57, 0
	s_mov_b32 s22, s18
	s_cselect_b32 s7, -1, 0
	s_not_b32 s2, s2
	s_and_b32 s6, s7, s6
	s_add_co_i32 s4, s4, s2
	s_cmp_lg_u32 s6, 0
	s_load_b32 s6, s[0:1], 0x190 nv
	s_sub_co_ci_u32 s2, s4, 0
	s_mov_b32 s23, s71
	s_lshl_b32 s88, s2, 6
	s_mul_i32 s82, s58, s3
	s_add_co_i32 s24, s88, s63
	s_set_vgpr_msb 8
	v_or_b32_e32 v2, s88, v209 /*v721*/
	s_add_co_i32 s2, s24, 0x5f
	s_mov_b32 s68, 32
	s_ashr_i32 s4, s2, 31
	s_mov_b32 s15, s71
	s_lshr_b32 s4, s4, 27
	s_mov_b32 s65, 0xffff0000
	s_add_co_i32 s4, s2, s4
	s_mov_b32 s64, 0x7510000
	s_and_b32 s7, s4, 0xffffffe0
	s_ashr_i32 s4, s4, 5
	s_cmp_lg_u32 s2, s7
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, s7
	s_sub_co_ci_u32 s2, s4, 0
	s_max_i32 s2, s2, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s62
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s6, 0
	s_cselect_b32 s4, -1, 0
	s_and_b32 s4, s4, exec_lo
	s_cselect_b32 s25, s2, s62
	s_add_co_i32 s2, s24, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s4, s2, 31
	s_lshr_b32 s4, s4, 27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s4, s2, s4
	s_ashr_i32 s4, s4, 5
	s_cmp_gt_i32 s2, -1
	s_cselect_b32 s2, s4, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s25
	s_cmp_lg_u32 s6, 0
	s_cselect_b32 s90, -1, 0
	s_and_b32 s4, s90, exec_lo
	s_cselect_b32 s2, s2, s62
	s_ashr_i32 s6, s5, 31
	s_ashr_i32 s7, s59, 31
	s_lshr_b32 s6, s6, 29
	s_lshr_b32 s7, s7, 29
	s_add_co_i32 s6, s5, s6
	s_add_co_i32 s7, s59, s7
	s_ashr_i32 s8, s6, 3
	s_and_b32 s6, s6, -8
	s_ashr_i32 s9, s7, 3
	s_and_b32 s7, s7, -8
	s_and_b32 s4, s59, 7
	s_sub_co_i32 s10, s5, s6
	s_cmp_lg_u32 s59, s7
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s59, 0
	s_cselect_b32 s11, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s11, s7
	s_sub_co_ci_u32 s7, s9, 0
	s_cmp_lg_u32 s5, s6
	s_mul_i32 s7, s7, s10
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s5, 0
	s_mov_b32 s10, 0x200000
	s_cselect_b32 s9, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s9, s6
	s_sub_co_ci_u32 s6, s8, 0
	s_add_co_i32 s6, s6, s7
	s_cmp_eq_u32 s4, 0
	s_cselect_b32 s89, s6, s5
	s_abs_i32 s5, s61
	s_abs_i32 s8, s89
	s_cvt_f32_u32 s4, s5
	s_sub_co_i32 s7, 0, s5
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s4, s4
	s_mul_f32 s6, s4, 0x4f7ffffe
	s_mov_b32 s4, 1
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s6, s6
	s_mul_i32 s7, s7, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s7, s6, s7
	s_add_co_i32 s6, s6, s7
	s_xor_b32 s7, s89, s61
	s_mul_hi_u32 s6, s8, s6
	s_ashr_i32 s14, s7, 31
	s_mul_i32 s9, s6, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s8, s8, s9
	s_add_co_i32 s9, s6, 1
	s_sub_co_i32 s11, s8, s5
	s_cmp_ge_u32 s8, s5
	s_cselect_b32 s6, s9, s6
	s_cselect_b32 s8, s11, s8
	s_add_co_i32 s11, s6, 1
	s_cmp_ge_u32 s8, s5
	s_clause 0x1
	s_load_b64 s[8:9], s[0:1], 0xf0 nv
	s_load_b64 s[12:13], s[0:1], 0x118 nv
	s_cselect_b32 s5, s11, s6
	s_mov_b32 s11, s71
	s_xor_b32 s5, s5, s14
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s6, s5, s14
	s_mul_i32 s6, s6, s61
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_cmp_lg_u32 s89, s6
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s7, -1, 0
	s_and_b32 s6, s7, s6
	s_sub_co_ci_u32 s80, s5, s14
	s_lshl_b32 s5, s59, 4
	s_or_b32 s62, s88, 48
	s_mul_i32 s6, s5, s91
	s_set_vgpr_msb 0x888
	v_or_b32_e32 v135 /*v647*/, s62, v209 /*v721*/
	s_lshl4_add_u32 s6, s89, s6
	s_or_b32 s87, s88, 16
	s_set_vgpr_msb 0x8808
	v_mad_u32 v3, v2, s5, s6
	s_or_b32 s86, s88, 32
	v_or_b32_e32 v1, s87, v209 /*v721*/
	s_set_vgpr_msb 0x888
	v_or_b32_e32 v134 /*v646*/, s86, v209 /*v721*/
	s_set_vgpr_msb 0x8802
	v_mad_u32 v7, v135 /*v647*/, s5, s6
	s_ashr_i32 s83, s82, 31
	s_ashr_i32 s61, s60, 31
	v_mad_u32 v5, s5, v1, s6
	s_set_vgpr_msb 0x200
	v_or_b32_e32 v3, v3, v4
	s_set_vgpr_msb 2
	v_mad_u32 v6, v134 /*v646*/, s5, s6
	s_mul_i32 s5, s59, s3
	s_mul_u64 s[6:7], s[60:61], s[82:83]
	s_add_co_i32 s5, s89, s5
	s_set_vgpr_msb 0x200
	v_dual_lshlrev_b32 v3, 4, v3 :: v_dual_bitop2_b32 v7, v7, v4 bitop3:0x54
	s_mul_i32 s5, s5, s57
	s_ashr_i32 s81, s80, 31
	v_or_b32_e32 v6, v6, v4
	v_add_lshl_u32 v2, s5, v2, 2
	v_dual_lshlrev_b32 v7, 4, v7 :: v_dual_bitop2_b32 v5, v5, v4 bitop3:0x54
	s_add_nc_u64 s[6:7], s[6:7], s[80:81]
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v6, 4, v6
	s_lshl_b32 s3, s60, 7
	s_add_co_i32 s83, s25, -1
	v_lshlrev_b32_e32 v5, 4, v5
	s_clause 0x20
	buffer_load_b128 v[194:197], v3, s[16:19], null offen
	buffer_load_b128 v[198:201], v3, s[16:19], null offen offset:32
	buffer_load_b128 v[202:205], v3, s[16:19], null offen offset:64
	buffer_load_b128 v[206:209], v3, s[16:19], null offen offset:96
	buffer_load_b128 v[210:213], v3, s[16:19], null offen offset:128
	buffer_load_b128 v[214:217], v3, s[16:19], null offen offset:160
	buffer_load_b128 v[218:221], v3, s[16:19], null offen offset:192
	buffer_load_b128 v[222:225], v3, s[16:19], null offen offset:224
	buffer_load_b128 v[226:229], v5, s[16:19], null offen
	buffer_load_b128 v[230:233], v5, s[16:19], null offen offset:32
	buffer_load_b128 v[234:237], v5, s[16:19], null offen offset:64
	buffer_load_b128 v[238:241], v5, s[16:19], null offen offset:96
	buffer_load_b128 v[242:245], v5, s[16:19], null offen offset:128
	buffer_load_b128 v[246:249], v5, s[16:19], null offen offset:160
	buffer_load_b128 v[250:253], v5, s[16:19], null offen offset:192
	buffer_load_b128 v[254:257], v5, s[16:19], null offen offset:224
	s_set_vgpr_msb 64
	buffer_load_b128 v[2:5] /*v[258:261]*/, v6, s[16:19], null offen
	buffer_load_b128 v[6:9] /*v[262:265]*/, v6, s[16:19], null offen offset:32
	buffer_load_b128 v[10:13] /*v[266:269]*/, v6, s[16:19], null offen offset:64
	buffer_load_b128 v[14:17] /*v[270:273]*/, v6, s[16:19], null offen offset:96
	buffer_load_b128 v[18:21] /*v[274:277]*/, v6, s[16:19], null offen offset:128
	buffer_load_b128 v[22:25] /*v[278:281]*/, v6, s[16:19], null offen offset:160
	buffer_load_b128 v[26:29] /*v[282:285]*/, v6, s[16:19], null offen offset:192
	buffer_load_b128 v[30:33] /*v[286:289]*/, v6, s[16:19], null offen offset:224
	buffer_load_b128 v[34:37] /*v[290:293]*/, v7, s[16:19], null offen
	buffer_load_b128 v[38:41] /*v[294:297]*/, v7, s[16:19], null offen offset:32
	buffer_load_b128 v[50:53] /*v[306:309]*/, v7, s[16:19], null offen offset:64
	buffer_load_b128 v[54:57] /*v[310:313]*/, v7, s[16:19], null offen offset:96
	buffer_load_b128 v[58:61] /*v[314:317]*/, v7, s[16:19], null offen offset:128
	buffer_load_b128 v[62:65] /*v[318:321]*/, v7, s[16:19], null offen offset:160
	buffer_load_b128 v[66:69] /*v[322:325]*/, v7, s[16:19], null offen offset:192
	buffer_load_b128 v[70:73] /*v[326:329]*/, v7, s[16:19], null offen offset:224
	s_clause 0x1f
	buffer_load_b128 v[74:77] /*v[330:333]*/, v3, s[20:23], null offen
	buffer_load_b128 v[78:81] /*v[334:337]*/, v3, s[20:23], null offen offset:32
	buffer_load_b128 v[82:85] /*v[338:341]*/, v3, s[20:23], null offen offset:64
	buffer_load_b128 v[86:89] /*v[342:345]*/, v3, s[20:23], null offen offset:96
	buffer_load_b128 v[90:93] /*v[346:349]*/, v3, s[20:23], null offen offset:128
	buffer_load_b128 v[94:97] /*v[350:353]*/, v3, s[20:23], null offen offset:160
	buffer_load_b128 v[106:109] /*v[362:365]*/, v3, s[20:23], null offen offset:192
	buffer_load_b128 v[110:113] /*v[366:369]*/, v3, s[20:23], null offen offset:224
	buffer_load_b128 v[114:117] /*v[370:373]*/, v5, s[20:23], null offen
	buffer_load_b128 v[118:121] /*v[374:377]*/, v5, s[20:23], null offen offset:32
	buffer_load_b128 v[122:125] /*v[378:381]*/, v5, s[20:23], null offen offset:64
	buffer_load_b128 v[126:129] /*v[382:385]*/, v5, s[20:23], null offen offset:96
	buffer_load_b128 v[130:133] /*v[386:389]*/, v5, s[20:23], null offen offset:128
	buffer_load_b128 v[134:137] /*v[390:393]*/, v5, s[20:23], null offen offset:160
	buffer_load_b128 v[146:149] /*v[402:405]*/, v5, s[20:23], null offen offset:192
	buffer_load_b128 v[150:153] /*v[406:409]*/, v5, s[20:23], null offen offset:224
	buffer_load_b128 v[154:157] /*v[410:413]*/, v6, s[20:23], null offen
	buffer_load_b128 v[158:161] /*v[414:417]*/, v6, s[20:23], null offen offset:32
	buffer_load_b128 v[162:165] /*v[418:421]*/, v6, s[20:23], null offen offset:64
	buffer_load_b128 v[166:169] /*v[422:425]*/, v6, s[20:23], null offen offset:96
	buffer_load_b128 v[170:173] /*v[426:429]*/, v6, s[20:23], null offen offset:128
	buffer_load_b128 v[174:177] /*v[430:433]*/, v6, s[20:23], null offen offset:160
	buffer_load_b128 v[186:189] /*v[442:445]*/, v6, s[20:23], null offen offset:192
	buffer_load_b128 v[190:193] /*v[446:449]*/, v6, s[20:23], null offen offset:224
	buffer_load_b128 v[194:197] /*v[450:453]*/, v7, s[20:23], null offen
	buffer_load_b128 v[198:201] /*v[454:457]*/, v7, s[20:23], null offen offset:32
	buffer_load_b128 v[202:205] /*v[458:461]*/, v7, s[20:23], null offen offset:64
	buffer_load_b128 v[206:209] /*v[462:465]*/, v7, s[20:23], null offen offset:96
	buffer_load_b128 v[218:221] /*v[474:477]*/, v7, s[20:23], null offen offset:128
	buffer_load_b128 v[222:225] /*v[478:481]*/, v7, s[20:23], null offen offset:160
	buffer_load_b128 v[226:229] /*v[482:485]*/, v7, s[20:23], null offen offset:192
	buffer_load_b128 v[230:233] /*v[486:489]*/, v7, s[20:23], null offen offset:224
	s_set_vgpr_msb 0x4000
	v_add_lshl_u32 v3, s5, v1, 2
	s_set_vgpr_msb 8
	v_add_lshl_u32 v5, s5, v134 /*v646*/, 2
	v_add_lshl_u32 v6, s5, v135 /*v647*/, 2
	s_wait_kmcnt 0x0
	s_clause 0x3
	buffer_load_b32 v7, v2, s[8:11], null offen
	buffer_load_b32 v8, v3, s[8:11], null offen
	buffer_load_b32 v9, v5, s[8:11], null offen
	buffer_load_b32 v10, v6, s[8:11], null offen
	s_wait_xcnt 0x0
	s_lshl_b64 s[8:9], s[6:7], 8
	s_cmp_lg_u32 s3, 0x80000000
	s_mov_b32 s14, s10
	s_cselect_b32 s69, s3, 0x80
	s_max_i32 s3, s58, 0
	s_add_nc_u64 s[6:7], s[76:77], s[8:9]
	s_lshl_b32 s10, s3, 16
	s_lshr_b32 s3, s3, 16
	s_or_b32 s66, s10, 0x7fff
	s_or_b32 s67, s3, 0x800000
	s_min_i32 s3, s83, 1
	s_ashr_i32 s10, s69, 31
	s_lshl_b32 s3, s3, 5
	s_and_b32 s70, s10, 0xffff
	s_add_co_i32 s10, s3, s82
	s_bitset1_b32 s7, 31
	s_mov_b32 s5, s71
	s_ashr_i32 s11, s10, 31
	s_clause 0x3
	buffer_load_b32 v11, v2, s[12:15], null offen
	buffer_load_b32 v12, v3, s[12:15], null offen
	buffer_load_b32 v13, v5, s[12:15], null offen
	buffer_load_b32 v14, v6, s[12:15], null offen
	tensor_load_to_lds s[4:7], s[64:71]
	s_add_nc_u64 s[6:7], s[78:79], s[8:9]
	s_mul_u64 s[8:9], s[10:11], s[60:61]
	s_sub_co_i32 s3, s58, s3
	s_add_nc_u64 s[8:9], s[8:9], s[80:81]
	s_bitset1_b32 s7, 31
	s_movk_i32 s5, 0x2200
	s_lshl_b64 s[8:9], s[8:9], 8
	s_max_i32 s3, s3, 0
	tensor_load_to_lds s[4:7], s[64:71]
	s_add_nc_u64 s[6:7], s[76:77], s[8:9]
	s_lshl_b32 s10, s3, 16
	s_lshr_b32 s3, s3, 16
	s_bitset1_b32 s7, 31
	s_movk_i32 s5, 0x4400
	s_or_b32 s66, s10, 0x7fff
	s_or_b32 s67, s3, 0x800000
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v3, 1, v0 :: v_dual_bitop2_b32 v2, 16, v0 bitop3:0x40
	tensor_load_to_lds s[4:7], s[64:71]
	s_add_nc_u64 s[6:7], s[78:79], s[8:9]
	s_movk_i32 s5, 0x6600
	s_bitset1_b32 s7, 31
	s_set_vgpr_msb 0x80
	v_lshlrev_b32_e32 v214 /*v726*/, 3, v4
	tensor_load_to_lds s[4:7], s[64:71]
	s_set_vgpr_msb 0x8000
	v_and_b32_e32 v3, 16, v3
	s_set_vgpr_msb 0x88
	v_mad_u32_u24 v216 /*v728*/, 0x110, v209 /*v721*/, v2
	s_mul_f32 s8, s56, 0x3fb8aa3b
	s_mov_b32 s12, 2
	s_set_vgpr_msb 0x8880
	s_wait_loadcnt 0x7
	v_mul_f32_e32 v186 /*v698*/, 0xbfb8aa3b, v7
	s_wait_loadcnt 0x6
	v_mul_f32_e32 v188 /*v700*/, 0xbfb8aa3b, v8
	s_wait_loadcnt 0x5
	v_mul_f32_e32 v190 /*v702*/, 0xbfb8aa3b, v9
	s_wait_loadcnt 0x4
	v_mul_f32_e32 v192 /*v704*/, 0xbfb8aa3b, v10
	s_wait_loadcnt 0x3
	v_mul_f32_e64 v194 /*v706*/, -s56, v11
	s_set_vgpr_msb 0x8020
	v_and_or_b32 v4, v0, 7, v214 /*v726*/
	s_set_vgpr_msb 0x2082
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v196 /*v708*/, -s56, v12 :: v_dual_mul_f32 v198 /*v710*/, -s56, v13
	s_wait_loadcnt 0x0
	v_mul_f32_e64 v200 /*v712*/, -s56, v14
	v_mad_u32_u24 v215 /*v727*/, 0x110, v4, v3
	s_wait_tensorcnt 0x2
	ds_load_b128 v[114:117] /*v[626:629]*/, v216 /*v728*/
	ds_load_b128 v[118:121] /*v[630:633]*/, v216 /*v728*/ offset:32
	ds_load_b128 v[122:125] /*v[634:637]*/, v216 /*v728*/ offset:8704
	ds_load_b128 v[126:129] /*v[638:641]*/, v216 /*v728*/ offset:8736
	ds_load_b128 v[98:101] /*v[610:613]*/, v216 /*v728*/ offset:64
	ds_load_b128 v[102:105] /*v[614:617]*/, v216 /*v728*/ offset:96
	ds_load_b128 v[106:109] /*v[618:621]*/, v216 /*v728*/ offset:8768
	ds_load_b128 v[110:113] /*v[622:625]*/, v216 /*v728*/ offset:8800
	ds_load_b128 v[82:85] /*v[594:597]*/, v216 /*v728*/ offset:128
	ds_load_b128 v[86:89] /*v[598:601]*/, v216 /*v728*/ offset:160
	ds_load_b128 v[90:93] /*v[602:605]*/, v216 /*v728*/ offset:8832
	ds_load_b128 v[94:97] /*v[606:609]*/, v216 /*v728*/ offset:8864
	ds_load_b128 v[66:69] /*v[578:581]*/, v216 /*v728*/ offset:192
	ds_load_b128 v[70:73] /*v[582:585]*/, v216 /*v728*/ offset:224
	ds_load_b128 v[74:77] /*v[586:589]*/, v216 /*v728*/ offset:8896
	ds_load_b128 v[78:81] /*v[590:593]*/, v216 /*v728*/ offset:8928
	ds_load_b128 v[50:53] /*v[562:565]*/, v216 /*v728*/ offset:4352
	ds_load_b128 v[54:57] /*v[566:569]*/, v216 /*v728*/ offset:4384
	ds_load_b128 v[58:61] /*v[570:573]*/, v216 /*v728*/ offset:13056
	ds_load_b128 v[62:65] /*v[574:577]*/, v216 /*v728*/ offset:13088
	ds_load_b128 v[34:37] /*v[546:549]*/, v216 /*v728*/ offset:4416
	ds_load_b128 v[38:41] /*v[550:553]*/, v216 /*v728*/ offset:4448
	ds_load_b128 v[42:45] /*v[554:557]*/, v216 /*v728*/ offset:13120
	ds_load_b128 v[46:49] /*v[558:561]*/, v216 /*v728*/ offset:13152
	ds_load_b128 v[18:21] /*v[530:533]*/, v216 /*v728*/ offset:4480
	ds_load_b128 v[22:25] /*v[534:537]*/, v216 /*v728*/ offset:4512
	ds_load_b128 v[26:29] /*v[538:541]*/, v216 /*v728*/ offset:13184
	ds_load_b128 v[30:33] /*v[542:545]*/, v216 /*v728*/ offset:13216
	ds_load_b128 v[2:5] /*v[514:517]*/, v216 /*v728*/ offset:4544
	ds_load_b128 v[6:9] /*v[518:521]*/, v216 /*v728*/ offset:4576
	ds_load_b128 v[10:13] /*v[522:525]*/, v216 /*v728*/ offset:13248
	ds_load_b128 v[14:17] /*v[526:529]*/, v216 /*v728*/ offset:13280
	s_ashr_i32 s3, s2, 31
	s_cmp_lt_i32 s2, 1
	s_set_vgpr_msb 0x8200
	s_cbranch_scc1 .LBB0_3
	s_set_vgpr_msb 64
	v_mov_b32_e32 v250 /*v506*/, 0
	s_mov_b32 s9, s8
	s_mov_b32 s57, s56
	s_set_vgpr_msb 0x4082
	v_mov_b64_e32 v[130:131] /*v[642:643]*/, s[8:9]
	v_mov_b64_e32 v[132:133] /*v[644:645]*/, s[56:57]
	v_dual_mov_b32 v187 /*v699*/, v186 /*v698*/ :: v_dual_mov_b32 v195 /*v707*/, v194 /*v706*/
	v_dual_mov_b32 v189 /*v701*/, v188 /*v700*/ :: v_dual_mov_b32 v197 /*v709*/, v196 /*v708*/
	v_dual_mov_b32 v191 /*v703*/, v190 /*v702*/ :: v_dual_mov_b32 v199 /*v711*/, v198 /*v710*/
	v_dual_mov_b32 v193 /*v705*/, v192 /*v704*/ :: v_dual_mov_b32 v201 /*v713*/, v200 /*v712*/
	s_set_vgpr_msb 0x8241
	v_dual_mov_b32 v251 /*v507*/, v250 /*v506*/ :: v_dual_mov_b32 v252 /*v508*/, v250 /*v506*/
	v_dual_mov_b32 v253 /*v509*/, v250 /*v506*/ :: v_dual_mov_b32 v254 /*v510*/, v250 /*v506*/
	v_dual_mov_b32 v255 /*v511*/, v250 /*v506*/ :: v_dual_mov_b32 v242 /*v498*/, v250 /*v506*/
	s_set_vgpr_msb 0x4181
	v_dual_mov_b32 v0 /*v512*/, v250 /*v506*/ :: v_dual_mov_b32 v1 /*v513*/, v250 /*v506*/
	s_set_vgpr_msb 0x8141
	v_dual_mov_b32 v243 /*v499*/, v250 /*v506*/ :: v_dual_mov_b32 v244 /*v500*/, v250 /*v506*/
	v_dual_mov_b32 v245 /*v501*/, v250 /*v506*/ :: v_dual_mov_b32 v246 /*v502*/, v250 /*v506*/
	v_dual_mov_b32 v247 /*v503*/, v250 /*v506*/ :: v_dual_mov_b32 v248 /*v504*/, v250 /*v506*/
	v_dual_mov_b32 v249 /*v505*/, v250 /*v506*/ :: v_dual_mov_b32 v234 /*v490*/, v250 /*v506*/
	v_dual_mov_b32 v235 /*v491*/, v250 /*v506*/ :: v_dual_mov_b32 v236 /*v492*/, v250 /*v506*/
	v_dual_mov_b32 v237 /*v493*/, v250 /*v506*/ :: v_dual_mov_b32 v238 /*v494*/, v250 /*v506*/
	v_dual_mov_b32 v239 /*v495*/, v250 /*v506*/ :: v_dual_mov_b32 v240 /*v496*/, v250 /*v506*/
	v_dual_mov_b32 v241 /*v497*/, v250 /*v506*/ :: v_dual_mov_b32 v210 /*v466*/, v250 /*v506*/
	v_dual_mov_b32 v211 /*v467*/, v250 /*v506*/ :: v_dual_mov_b32 v212 /*v468*/, v250 /*v506*/
	v_dual_mov_b32 v213 /*v469*/, v250 /*v506*/ :: v_dual_mov_b32 v214 /*v470*/, v250 /*v506*/
	v_dual_mov_b32 v215 /*v471*/, v250 /*v506*/ :: v_dual_mov_b32 v216 /*v472*/, v250 /*v506*/
	v_dual_mov_b32 v217 /*v473*/, v250 /*v506*/ :: v_dual_mov_b32 v178 /*v434*/, v250 /*v506*/
	v_dual_mov_b32 v179 /*v435*/, v250 /*v506*/ :: v_dual_mov_b32 v180 /*v436*/, v250 /*v506*/
	v_dual_mov_b32 v181 /*v437*/, v250 /*v506*/ :: v_dual_mov_b32 v182 /*v438*/, v250 /*v506*/
	v_dual_mov_b32 v183 /*v439*/, v250 /*v506*/ :: v_dual_mov_b32 v184 /*v440*/, v250 /*v506*/
	v_dual_mov_b32 v185 /*v441*/, v250 /*v506*/ :: v_dual_mov_b32 v138 /*v394*/, v250 /*v506*/
	v_dual_mov_b32 v139 /*v395*/, v250 /*v506*/ :: v_dual_mov_b32 v140 /*v396*/, v250 /*v506*/
	v_dual_mov_b32 v141 /*v397*/, v250 /*v506*/ :: v_dual_mov_b32 v142 /*v398*/, v250 /*v506*/
	v_dual_mov_b32 v143 /*v399*/, v250 /*v506*/ :: v_dual_mov_b32 v144 /*v400*/, v250 /*v506*/
	v_dual_mov_b32 v145 /*v401*/, v250 /*v506*/ :: v_dual_mov_b32 v98 /*v354*/, v250 /*v506*/
	v_dual_mov_b32 v99 /*v355*/, v250 /*v506*/ :: v_dual_mov_b32 v100 /*v356*/, v250 /*v506*/
	v_dual_mov_b32 v101 /*v357*/, v250 /*v506*/ :: v_dual_mov_b32 v102 /*v358*/, v250 /*v506*/
	v_dual_mov_b32 v103 /*v359*/, v250 /*v506*/ :: v_dual_mov_b32 v104 /*v360*/, v250 /*v506*/
	v_dual_mov_b32 v105 /*v361*/, v250 /*v506*/ :: v_dual_mov_b32 v42 /*v298*/, v250 /*v506*/
	v_dual_mov_b32 v43 /*v299*/, v250 /*v506*/ :: v_dual_mov_b32 v44 /*v300*/, v250 /*v506*/
	v_dual_mov_b32 v45 /*v301*/, v250 /*v506*/ :: v_dual_mov_b32 v46 /*v302*/, v250 /*v506*/
	v_dual_mov_b32 v47 /*v303*/, v250 /*v506*/ :: v_dual_mov_b32 v48 /*v304*/, v250 /*v506*/
	v_mov_b32_e32 v49 /*v305*/, v250 /*v506*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v186, v250 /*v506*/ :: v_dual_mov_b32 v187, v250 /*v506*/
	v_dual_mov_b32 v188, v250 /*v506*/ :: v_dual_mov_b32 v189, v250 /*v506*/
	v_dual_mov_b32 v190, v250 /*v506*/ :: v_dual_mov_b32 v191, v250 /*v506*/
	v_dual_mov_b32 v192, v250 /*v506*/ :: v_dual_mov_b32 v193, v250 /*v506*/
	v_dual_mov_b32 v178, v250 /*v506*/ :: v_dual_mov_b32 v179, v250 /*v506*/
	v_dual_mov_b32 v180, v250 /*v506*/ :: v_dual_mov_b32 v181, v250 /*v506*/
	v_dual_mov_b32 v182, v250 /*v506*/ :: v_dual_mov_b32 v183, v250 /*v506*/
	v_dual_mov_b32 v184, v250 /*v506*/ :: v_dual_mov_b32 v185, v250 /*v506*/
	v_dual_mov_b32 v170, v250 /*v506*/ :: v_dual_mov_b32 v171, v250 /*v506*/
	v_dual_mov_b32 v172, v250 /*v506*/ :: v_dual_mov_b32 v173, v250 /*v506*/
	v_dual_mov_b32 v174, v250 /*v506*/ :: v_dual_mov_b32 v175, v250 /*v506*/
	v_dual_mov_b32 v176, v250 /*v506*/ :: v_dual_mov_b32 v177, v250 /*v506*/
	v_dual_mov_b32 v162, v250 /*v506*/ :: v_dual_mov_b32 v163, v250 /*v506*/
	v_dual_mov_b32 v164, v250 /*v506*/ :: v_dual_mov_b32 v165, v250 /*v506*/
	v_dual_mov_b32 v166, v250 /*v506*/ :: v_dual_mov_b32 v167, v250 /*v506*/
	v_dual_mov_b32 v168, v250 /*v506*/ :: v_dual_mov_b32 v169, v250 /*v506*/
	v_dual_mov_b32 v154, v250 /*v506*/ :: v_dual_mov_b32 v155, v250 /*v506*/
	v_dual_mov_b32 v156, v250 /*v506*/ :: v_dual_mov_b32 v157, v250 /*v506*/
	v_dual_mov_b32 v158, v250 /*v506*/ :: v_dual_mov_b32 v159, v250 /*v506*/
	v_dual_mov_b32 v160, v250 /*v506*/ :: v_dual_mov_b32 v161, v250 /*v506*/
	v_dual_mov_b32 v146, v250 /*v506*/ :: v_dual_mov_b32 v147, v250 /*v506*/
	v_dual_mov_b32 v148, v250 /*v506*/ :: v_dual_mov_b32 v149, v250 /*v506*/
	v_dual_mov_b32 v150, v250 /*v506*/ :: v_dual_mov_b32 v151, v250 /*v506*/
	v_dual_mov_b32 v152, v250 /*v506*/ :: v_dual_mov_b32 v153, v250 /*v506*/
	v_dual_mov_b32 v138, v250 /*v506*/ :: v_dual_mov_b32 v139, v250 /*v506*/
	v_dual_mov_b32 v140, v250 /*v506*/ :: v_dual_mov_b32 v141, v250 /*v506*/
	v_dual_mov_b32 v142, v250 /*v506*/ :: v_dual_mov_b32 v143, v250 /*v506*/
	v_dual_mov_b32 v144, v250 /*v506*/ :: v_dual_mov_b32 v145, v250 /*v506*/
	v_dual_mov_b32 v130, v250 /*v506*/ :: v_dual_mov_b32 v131, v250 /*v506*/
	v_dual_mov_b32 v132, v250 /*v506*/ :: v_dual_mov_b32 v133, v250 /*v506*/
	v_dual_mov_b32 v134, v250 /*v506*/ :: v_dual_mov_b32 v135, v250 /*v506*/
	v_dual_mov_b32 v136, v250 /*v506*/ :: v_dual_mov_b32 v137, v250 /*v506*/
	v_dual_mov_b32 v122, v250 /*v506*/ :: v_dual_mov_b32 v123, v250 /*v506*/
	v_dual_mov_b32 v124, v250 /*v506*/ :: v_dual_mov_b32 v125, v250 /*v506*/
	v_dual_mov_b32 v126, v250 /*v506*/ :: v_dual_mov_b32 v127, v250 /*v506*/
	v_dual_mov_b32 v128, v250 /*v506*/ :: v_dual_mov_b32 v129, v250 /*v506*/
	v_dual_mov_b32 v114, v250 /*v506*/ :: v_dual_mov_b32 v115, v250 /*v506*/
	v_dual_mov_b32 v116, v250 /*v506*/ :: v_dual_mov_b32 v117, v250 /*v506*/
	v_dual_mov_b32 v118, v250 /*v506*/ :: v_dual_mov_b32 v119, v250 /*v506*/
	v_dual_mov_b32 v120, v250 /*v506*/ :: v_dual_mov_b32 v121, v250 /*v506*/
	v_dual_mov_b32 v106, v250 /*v506*/ :: v_dual_mov_b32 v107, v250 /*v506*/
	v_dual_mov_b32 v108, v250 /*v506*/ :: v_dual_mov_b32 v109, v250 /*v506*/
	v_dual_mov_b32 v110, v250 /*v506*/ :: v_dual_mov_b32 v111, v250 /*v506*/
	v_dual_mov_b32 v112, v250 /*v506*/ :: v_dual_mov_b32 v113, v250 /*v506*/
	v_dual_mov_b32 v98, v250 /*v506*/ :: v_dual_mov_b32 v99, v250 /*v506*/
	v_dual_mov_b32 v100, v250 /*v506*/ :: v_dual_mov_b32 v101, v250 /*v506*/
	v_dual_mov_b32 v102, v250 /*v506*/ :: v_dual_mov_b32 v103, v250 /*v506*/
	v_dual_mov_b32 v104, v250 /*v506*/ :: v_dual_mov_b32 v105, v250 /*v506*/
	v_dual_mov_b32 v90, v250 /*v506*/ :: v_dual_mov_b32 v91, v250 /*v506*/
	v_dual_mov_b32 v92, v250 /*v506*/ :: v_dual_mov_b32 v93, v250 /*v506*/
	v_dual_mov_b32 v94, v250 /*v506*/ :: v_dual_mov_b32 v95, v250 /*v506*/
	v_dual_mov_b32 v96, v250 /*v506*/ :: v_dual_mov_b32 v97, v250 /*v506*/
	v_dual_mov_b32 v82, v250 /*v506*/ :: v_dual_mov_b32 v83, v250 /*v506*/
	v_dual_mov_b32 v84, v250 /*v506*/ :: v_dual_mov_b32 v85, v250 /*v506*/
	v_dual_mov_b32 v86, v250 /*v506*/ :: v_dual_mov_b32 v87, v250 /*v506*/
	v_dual_mov_b32 v88, v250 /*v506*/ :: v_dual_mov_b32 v89, v250 /*v506*/
	v_dual_mov_b32 v74, v250 /*v506*/ :: v_dual_mov_b32 v75, v250 /*v506*/
	v_dual_mov_b32 v76, v250 /*v506*/ :: v_dual_mov_b32 v77, v250 /*v506*/
	v_dual_mov_b32 v78, v250 /*v506*/ :: v_dual_mov_b32 v79, v250 /*v506*/
	v_dual_mov_b32 v80, v250 /*v506*/ :: v_dual_mov_b32 v81, v250 /*v506*/
	v_dual_mov_b32 v66, v250 /*v506*/ :: v_dual_mov_b32 v67, v250 /*v506*/
	v_dual_mov_b32 v68, v250 /*v506*/ :: v_dual_mov_b32 v69, v250 /*v506*/
	v_dual_mov_b32 v70, v250 /*v506*/ :: v_dual_mov_b32 v71, v250 /*v506*/
	v_dual_mov_b32 v72, v250 /*v506*/ :: v_dual_mov_b32 v73, v250 /*v506*/
	v_dual_mov_b32 v58, v250 /*v506*/ :: v_dual_mov_b32 v59, v250 /*v506*/
	v_dual_mov_b32 v60, v250 /*v506*/ :: v_dual_mov_b32 v61, v250 /*v506*/
	v_dual_mov_b32 v62, v250 /*v506*/ :: v_dual_mov_b32 v63, v250 /*v506*/
	v_dual_mov_b32 v64, v250 /*v506*/ :: v_dual_mov_b32 v65, v250 /*v506*/
	v_dual_mov_b32 v50, v250 /*v506*/ :: v_dual_mov_b32 v51, v250 /*v506*/
	v_dual_mov_b32 v52, v250 /*v506*/ :: v_dual_mov_b32 v53, v250 /*v506*/
	v_dual_mov_b32 v54, v250 /*v506*/ :: v_dual_mov_b32 v55, v250 /*v506*/
	v_dual_mov_b32 v56, v250 /*v506*/ :: v_dual_mov_b32 v57, v250 /*v506*/
	v_dual_mov_b32 v42, v250 /*v506*/ :: v_dual_mov_b32 v43, v250 /*v506*/
	v_dual_mov_b32 v44, v250 /*v506*/ :: v_dual_mov_b32 v45, v250 /*v506*/
	v_dual_mov_b32 v46, v250 /*v506*/ :: v_dual_mov_b32 v47, v250 /*v506*/
	v_dual_mov_b32 v48, v250 /*v506*/ :: v_dual_mov_b32 v49, v250 /*v506*/
	v_dual_mov_b32 v34, v250 /*v506*/ :: v_dual_mov_b32 v35, v250 /*v506*/
	v_dual_mov_b32 v36, v250 /*v506*/ :: v_dual_mov_b32 v37, v250 /*v506*/
	v_dual_mov_b32 v38, v250 /*v506*/ :: v_dual_mov_b32 v39, v250 /*v506*/
	v_dual_mov_b32 v40, v250 /*v506*/ :: v_dual_mov_b32 v41, v250 /*v506*/
	v_dual_mov_b32 v26, v250 /*v506*/ :: v_dual_mov_b32 v27, v250 /*v506*/
	v_dual_mov_b32 v28, v250 /*v506*/ :: v_dual_mov_b32 v29, v250 /*v506*/
	v_dual_mov_b32 v30, v250 /*v506*/ :: v_dual_mov_b32 v31, v250 /*v506*/
	v_dual_mov_b32 v32, v250 /*v506*/ :: v_dual_mov_b32 v33, v250 /*v506*/
	v_dual_mov_b32 v18, v250 /*v506*/ :: v_dual_mov_b32 v19, v250 /*v506*/
	v_dual_mov_b32 v20, v250 /*v506*/ :: v_dual_mov_b32 v21, v250 /*v506*/
	v_dual_mov_b32 v22, v250 /*v506*/ :: v_dual_mov_b32 v23, v250 /*v506*/
	v_dual_mov_b32 v24, v250 /*v506*/ :: v_dual_mov_b32 v25, v250 /*v506*/
	v_dual_mov_b32 v10, v250 /*v506*/ :: v_dual_mov_b32 v11, v250 /*v506*/
	v_dual_mov_b32 v12, v250 /*v506*/ :: v_dual_mov_b32 v13, v250 /*v506*/
	v_dual_mov_b32 v14, v250 /*v506*/ :: v_dual_mov_b32 v15, v250 /*v506*/
	v_dual_mov_b32 v16, v250 /*v506*/ :: v_dual_mov_b32 v17, v250 /*v506*/
	v_dual_mov_b32 v2, v250 /*v506*/ :: v_dual_mov_b32 v3, v250 /*v506*/
	v_dual_mov_b32 v4, v250 /*v506*/ :: v_dual_mov_b32 v5, v250 /*v506*/
	v_dual_mov_b32 v6, v250 /*v506*/ :: v_dual_mov_b32 v7, v250 /*v506*/
	v_dual_mov_b32 v8, v250 /*v506*/ :: v_dual_mov_b32 v9, v250 /*v506*/
	s_mov_b64 s[10:11], s[2:3]
	s_mov_b32 s3, s71
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x0
.LBB0_2:
	s_mov_b32 s9, s3
	s_min_i32 s6, s12, s83
	s_addk_co_i32 s3, 0xbc00
	s_cmp_lg_u32 s9, 0
	s_cselect_b32 s5, s3, 0x8800
	s_add_co_i32 s3, s9, 0x4400
	s_cmp_lg_u32 s9, 0x8800
	s_cselect_b32 s3, s3, 0
	s_lshl_b32 s7, s6, 5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s6, s7, s82
	s_sub_co_i32 s13, s58, s7
	s_ashr_i32 s7, s6, 31
	s_max_i32 s13, s13, 0
	s_mul_u64 s[6:7], s[6:7], s[60:61]
	s_lshl_b32 s16, s13, 16
	s_add_nc_u64 s[6:7], s[6:7], s[80:81]
	s_lshr_b32 s13, s13, 16
	s_lshl_b64 s[14:15], s[6:7], 8
	s_or_b32 s66, s16, 0x7fff
	s_add_nc_u64 s[6:7], s[76:77], s[14:15]
	s_or_b32 s67, s13, 0x800000
	s_bitset1_b32 s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[4:7], s[64:71]
	s_add_nc_u64 s[6:7], s[78:79], s[14:15]
	s_addk_co_i32 s5, 0x2200
	s_bitset1_b32 s7, 31
	tensor_load_to_lds s[4:7], s[64:71]
	s_set_vgpr_msb 0x82
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[114:121] /*v[626:633]*/, v[194:201], 0
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[114:121] /*v[626:633]*/, v[226:233], 0
	s_set_vgpr_msb 0x82a6
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[680:687]*/, v[114:121] /*v[626:633]*/, v[2:9] /*v[258:265]*/, 0
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[114:121] /*v[626:633]*/, v[34:41] /*v[290:297]*/, 0
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[122:129] /*v[634:641]*/, v[194:201] /*v[450:457]*/, 0
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[122:129] /*v[634:641]*/, v[74:81] /*v[330:337]*/, 0
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[122:129] /*v[634:641]*/, v[114:121] /*v[370:377]*/, 0
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[688:695]*/, v[122:129] /*v[634:641]*/, v[154:161] /*v[410:417]*/, 0
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[106:113] /*v[618:625]*/, v[202:209] /*v[458:465]*/, v[114:121] /*v[626:633]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[98:105] /*v[610:617]*/, v[202:209], v[136:143] /*v[648:655]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[106:113] /*v[618:625]*/, v[82:89] /*v[338:345]*/, v[144:151] /*v[656:663]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[98:105] /*v[610:617]*/, v[234:241], v[152:159] /*v[664:671]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[106:113] /*v[618:625]*/, v[122:129] /*v[378:385]*/, v[160:167] /*v[672:679]*/
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[680:687]*/, v[98:105] /*v[610:617]*/, v[10:17] /*v[266:273]*/, v[168:175] /*v[680:687]*/
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[688:695]*/, v[106:113] /*v[618:625]*/, v[162:169] /*v[418:425]*/, v[176:183] /*v[688:695]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[98:105] /*v[610:617]*/, v[50:57] /*v[306:313]*/, v[218:225] /*v[730:737]*/
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[90:97] /*v[602:609]*/, v[218:225] /*v[474:481]*/, v[114:121] /*v[626:633]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[82:89] /*v[594:601]*/, v[210:217], v[136:143] /*v[648:655]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[90:97] /*v[602:609]*/, v[90:97] /*v[346:353]*/, v[144:151] /*v[656:663]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[82:89] /*v[594:601]*/, v[242:249], v[152:159] /*v[664:671]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[90:97] /*v[602:609]*/, v[130:137] /*v[386:393]*/, v[160:167] /*v[672:679]*/
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[680:687]*/, v[82:89] /*v[594:601]*/, v[18:25] /*v[274:281]*/, v[168:175] /*v[680:687]*/
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[688:695]*/, v[90:97] /*v[602:609]*/, v[170:177] /*v[426:433]*/, v[176:183] /*v[688:695]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[82:89] /*v[594:601]*/, v[58:65] /*v[314:321]*/, v[218:225] /*v[730:737]*/
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[74:81] /*v[586:593]*/, v[226:233] /*v[482:489]*/, v[114:121] /*v[626:633]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[66:73] /*v[578:585]*/, v[218:225], v[136:143] /*v[648:655]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[74:81] /*v[586:593]*/, v[106:113] /*v[362:369]*/, v[144:151] /*v[656:663]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[66:73] /*v[578:585]*/, v[250:257], v[152:159] /*v[664:671]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[74:81] /*v[586:593]*/, v[146:153] /*v[402:409]*/, v[160:167] /*v[672:679]*/
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[680:687]*/, v[66:73] /*v[578:585]*/, v[26:33] /*v[282:289]*/, v[168:175] /*v[680:687]*/
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[688:695]*/, v[74:81] /*v[586:593]*/, v[186:193] /*v[442:449]*/, v[176:183] /*v[688:695]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[66:73] /*v[578:585]*/, v[66:73] /*v[322:329]*/, v[218:225] /*v[730:737]*/
	s_set_vgpr_msb 0xa682
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[50:57] /*v[562:569]*/, v[194:201], 0
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x82aa
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[136:137] /*v[648:649]*/, v[130:131] /*v[642:643]*/, v[186:187] /*v[698:699]*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[138:139] /*v[650:651]*/, v[130:131] /*v[642:643]*/, v[186:187] /*v[698:699]*/
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v66 /*v578*/, v66 /*v578*/
	s_set_vgpr_msb 0xaa86
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[58:65] /*v[570:577]*/, v[74:81] /*v[330:337]*/, 0
	s_set_vgpr_msb 0x86aa
	v_pk_fma_f32 v[70:71] /*v[582:583]*/, v[140:141] /*v[652:653]*/, v[130:131] /*v[642:643]*/, v[186:187] /*v[698:699]*/
	v_pk_fma_f32 v[72:73] /*v[584:585]*/, v[142:143] /*v[654:655]*/, v[130:131] /*v[642:643]*/, v[186:187] /*v[698:699]*/
	v_exp_f32_e32 v67 /*v579*/, v67 /*v579*/
	s_set_vgpr_msb 0xaa82
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[50:57] /*v[562:569]*/, v[226:233], 0
	s_set_vgpr_msb 0x82aa
	v_pk_fma_f32 v[74:75] /*v[586:587]*/, v[144:145] /*v[656:657]*/, v[132:133] /*v[644:645]*/, v[194:195] /*v[706:707]*/
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[74:75] /*v[586:587]*/
	v_exp_f32_e32 v68 /*v580*/, v68 /*v580*/
	s_set_vgpr_msb 0xaa86
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[58:65] /*v[570:577]*/, v[114:121] /*v[370:377]*/, 0
	s_set_vgpr_msb 0x86aa
	v_cvt_pk_bf16_f32 v144 /*v656*/, v66 /*v578*/, v67 /*v579*/
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[146:147] /*v[658:659]*/, v[132:133] /*v[644:645]*/, v[194:195] /*v[706:707]*/
	v_exp_f32_e32 v69 /*v581*/, v69 /*v581*/
	s_set_vgpr_msb 0xaa86
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[50:57] /*v[562:569]*/, v[2:9] /*v[258:265]*/, 0
	s_set_vgpr_msb 0x868a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[68:69] /*v[580:581]*/, v[66:67] /*v[578:579]*/
	v_cvt_pk_bf16_f32 v145 /*v657*/, v66 /*v578*/, v67 /*v579*/
	v_exp_f32_e32 v66 /*v578*/, v70 /*v582*/
	s_set_vgpr_msb 0x8ac6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[58:65] /*v[570:577]*/, v[154:161] /*v[410:417]*/, 0
	s_set_vgpr_msb 0xc6aa
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[148:149] /*v[660:661]*/, v[132:133] /*v[644:645]*/, v[194:195] /*v[706:707]*/
	v_pk_fma_f32 v[74:75] /*v[586:587]*/, v[150:151] /*v[662:663]*/, v[132:133] /*v[644:645]*/, v[194:195] /*v[706:707]*/
	v_exp_f32_e32 v67 /*v579*/, v71 /*v583*/
	s_set_vgpr_msb 0xaac6
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[50:57] /*v[562:569]*/, v[34:41] /*v[290:297]*/, 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xc68a
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[66:67] /*v[578:579]*/, v[68:69] /*v[580:581]*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v146 /*v658*/, v50 /*v562*/, v51 /*v563*/
	v_exp_f32_e32 v50 /*v562*/, v72 /*v584*/
	s_set_vgpr_msb 0x8ac6
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[58:65] /*v[570:577]*/, v[194:201] /*v[450:457]*/, 0
	s_set_vgpr_msb 0xc68a
	v_exp_f32_e32 v51 /*v563*/, v73 /*v585*/
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, v[74:75] /*v[586:587]*/
	v_cvt_pk_bf16_f32 v147 /*v659*/, v50 /*v562*/, v51 /*v563*/
	s_set_vgpr_msb 0x8aa2
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[34:41] /*v[546:553]*/, v[202:209], v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa2aa
	v_pk_fma_f32 v[50:51] /*v[562:563]*/, v[152:153] /*v[664:665]*/, v[130:131] /*v[642:643]*/, v[188:189] /*v[700:701]*/
	v_pk_fma_f32 v[52:53] /*v[564:565]*/, v[154:155] /*v[666:667]*/, v[130:131] /*v[642:643]*/, v[188:189] /*v[700:701]*/
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v50 /*v562*/, v50 /*v562*/
	s_set_vgpr_msb 0xaaa6
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[42:49] /*v[554:561]*/, v[82:89] /*v[338:345]*/, v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa6aa
	v_pk_fma_f32 v[54:55] /*v[566:567]*/, v[156:157] /*v[668:669]*/, v[130:131] /*v[642:643]*/, v[188:189] /*v[700:701]*/
	v_pk_fma_f32 v[56:57] /*v[568:569]*/, v[158:159] /*v[670:671]*/, v[130:131] /*v[642:643]*/, v[188:189] /*v[700:701]*/
	v_exp_f32_e32 v51 /*v563*/, v51 /*v563*/
	s_set_vgpr_msb 0xaaa2
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[34:41] /*v[546:553]*/, v[234:241], v[136:143] /*v[648:655]*/
	s_set_vgpr_msb 0xa2aa
	v_pk_fma_f32 v[58:59] /*v[570:571]*/, v[160:161] /*v[672:673]*/, v[132:133] /*v[644:645]*/, v[196:197] /*v[708:709]*/
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, v[58:59] /*v[570:571]*/
	v_exp_f32_e32 v52 /*v564*/, v52 /*v564*/
	s_set_vgpr_msb 0xaaa6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[42:49] /*v[554:561]*/, v[122:129] /*v[378:385]*/, v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa6aa
	v_cvt_pk_bf16_f32 v152 /*v664*/, v50 /*v562*/, v51 /*v563*/
	v_pk_fma_f32 v[50:51] /*v[562:563]*/, v[162:163] /*v[674:675]*/, v[132:133] /*v[644:645]*/, v[196:197] /*v[708:709]*/
	v_exp_f32_e32 v53 /*v565*/, v53 /*v565*/
	s_set_vgpr_msb 0xaaa6
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[34:41] /*v[546:553]*/, v[10:17] /*v[266:273]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa68a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/, v[50:51] /*v[562:563]*/
	v_cvt_pk_bf16_f32 v153 /*v665*/, v50 /*v562*/, v51 /*v563*/
	v_exp_f32_e32 v50 /*v562*/, v54 /*v566*/
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[42:49] /*v[554:561]*/, v[162:169] /*v[418:425]*/, v[2:9] /*v[770:777]*/
	s_set_vgpr_msb 0xf6aa
	v_pk_fma_f32 v[52:53] /*v[564:565]*/, v[164:165] /*v[676:677]*/, v[132:133] /*v[644:645]*/, v[196:197] /*v[708:709]*/
	v_pk_fma_f32 v[58:59] /*v[570:571]*/, v[166:167] /*v[678:679]*/, v[132:133] /*v[644:645]*/, v[196:197] /*v[708:709]*/
	v_exp_f32_e32 v51 /*v563*/, v55 /*v567*/
	s_set_vgpr_msb 0xaaf6
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[34:41] /*v[546:553]*/, v[50:57] /*v[306:313]*/, v[10:17] /*v[778:785]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xf68a
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v154 /*v666*/, v34 /*v546*/, v35 /*v547*/
	v_exp_f32_e32 v34 /*v546*/, v56 /*v568*/
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[42:49] /*v[554:561]*/, v[202:209] /*v[458:465]*/, v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xf68a
	v_exp_f32_e32 v35 /*v547*/, v57 /*v569*/
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[58:59] /*v[570:571]*/
	v_cvt_pk_bf16_f32 v155 /*v667*/, v34 /*v546*/, v35 /*v547*/
	s_set_vgpr_msb 0x8aa2
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[18:25] /*v[530:537]*/, v[210:217], v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa2aa
	v_pk_fma_f32 v[34:35] /*v[546:547]*/, v[168:169] /*v[680:681]*/, v[130:131] /*v[642:643]*/, v[190:191] /*v[702:703]*/
	v_pk_fma_f32 v[36:37] /*v[548:549]*/, v[170:171] /*v[682:683]*/, v[130:131] /*v[642:643]*/, v[190:191] /*v[702:703]*/
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v34 /*v546*/, v34 /*v546*/
	s_set_vgpr_msb 0xaaa6
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[26:33] /*v[538:545]*/, v[90:97] /*v[346:353]*/, v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa6aa
	v_pk_fma_f32 v[38:39] /*v[550:551]*/, v[172:173] /*v[684:685]*/, v[130:131] /*v[642:643]*/, v[190:191] /*v[702:703]*/
	v_pk_fma_f32 v[40:41] /*v[552:553]*/, v[174:175] /*v[686:687]*/, v[130:131] /*v[642:643]*/, v[190:191] /*v[702:703]*/
	v_exp_f32_e32 v35 /*v547*/, v35 /*v547*/
	s_set_vgpr_msb 0xaaa2
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[18:25] /*v[530:537]*/, v[242:249], v[136:143] /*v[648:655]*/
	s_set_vgpr_msb 0xa2aa
	v_pk_fma_f32 v[42:43] /*v[554:555]*/, v[176:177] /*v[688:689]*/, v[132:133] /*v[644:645]*/, v[198:199] /*v[710:711]*/
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[42:43] /*v[554:555]*/
	v_exp_f32_e32 v36 /*v548*/, v36 /*v548*/
	s_set_vgpr_msb 0xaaa6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[26:33] /*v[538:545]*/, v[130:137] /*v[386:393]*/, v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa6aa
	v_cvt_pk_bf16_f32 v160 /*v672*/, v34 /*v546*/, v35 /*v547*/
	v_pk_fma_f32 v[34:35] /*v[546:547]*/, v[178:179] /*v[690:691]*/, v[132:133] /*v[644:645]*/, v[198:199] /*v[710:711]*/
	v_exp_f32_e32 v37 /*v549*/, v37 /*v549*/
	s_set_vgpr_msb 0xaaa6
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[18:25] /*v[530:537]*/, v[18:25] /*v[274:281]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa68a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[36:37] /*v[548:549]*/, v[34:35] /*v[546:547]*/
	v_cvt_pk_bf16_f32 v161 /*v673*/, v34 /*v546*/, v35 /*v547*/
	v_exp_f32_e32 v34 /*v546*/, v38 /*v550*/
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[26:33] /*v[538:545]*/, v[170:177] /*v[426:433]*/, v[2:9] /*v[770:777]*/
	s_set_vgpr_msb 0xf6aa
	v_pk_fma_f32 v[36:37] /*v[548:549]*/, v[180:181] /*v[692:693]*/, v[132:133] /*v[644:645]*/, v[198:199] /*v[710:711]*/
	v_pk_fma_f32 v[42:43] /*v[554:555]*/, v[182:183] /*v[694:695]*/, v[132:133] /*v[644:645]*/, v[198:199] /*v[710:711]*/
	v_exp_f32_e32 v35 /*v547*/, v39 /*v551*/
	s_set_vgpr_msb 0xaaf6
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[18:25] /*v[530:537]*/, v[58:65] /*v[314:321]*/, v[10:17] /*v[778:785]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xf68a
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[34:35] /*v[546:547]*/, v[36:37] /*v[548:549]*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v162 /*v674*/, v18 /*v530*/, v19 /*v531*/
	v_exp_f32_e32 v18 /*v530*/, v40 /*v552*/
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[26:33] /*v[538:545]*/, v[218:225] /*v[474:481]*/, v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xf68a
	v_exp_f32_e32 v19 /*v531*/, v41 /*v553*/
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, v[42:43] /*v[554:555]*/
	v_cvt_pk_bf16_f32 v163 /*v675*/, v18 /*v530*/, v19 /*v531*/
	s_set_vgpr_msb 0x8aa2
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[2:9] /*v[514:521]*/, v[218:225], v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa2aa
	v_pk_fma_f32 v[18:19] /*v[530:531]*/, v[218:219] /*v[730:731]*/, v[130:131] /*v[642:643]*/, v[192:193] /*v[704:705]*/
	v_pk_fma_f32 v[20:21] /*v[532:533]*/, v[220:221] /*v[732:733]*/, v[130:131] /*v[642:643]*/, v[192:193] /*v[704:705]*/
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v18 /*v530*/, v18 /*v530*/
	s_set_vgpr_msb 0xaaa6
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[10:17] /*v[522:529]*/, v[106:113] /*v[362:369]*/, v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa6aa
	v_pk_fma_f32 v[22:23] /*v[534:535]*/, v[222:223] /*v[734:735]*/, v[130:131] /*v[642:643]*/, v[192:193] /*v[704:705]*/
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[224:225] /*v[736:737]*/, v[130:131] /*v[642:643]*/, v[192:193] /*v[704:705]*/
	v_exp_f32_e32 v19 /*v531*/, v19 /*v531*/
	s_set_vgpr_msb 0xaaa2
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[2:9] /*v[514:521]*/, v[250:257], v[136:143] /*v[648:655]*/
	s_set_vgpr_msb 0xa2aa
	v_pk_fma_f32 v[26:27] /*v[538:539]*/, v[114:115] /*v[626:627]*/, v[132:133] /*v[644:645]*/, v[200:201] /*v[712:713]*/
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, v[26:27] /*v[538:539]*/
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	s_set_vgpr_msb 0xaaa6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[10:17] /*v[522:529]*/, v[146:153] /*v[402:409]*/, v[242:249] /*v[754:761]*/
	s_set_vgpr_msb 0xa6aa
	v_cvt_pk_bf16_f32 v168 /*v680*/, v18 /*v530*/, v19 /*v531*/
	v_pk_fma_f32 v[18:19] /*v[530:531]*/, v[116:117] /*v[628:629]*/, v[132:133] /*v[644:645]*/, v[200:201] /*v[712:713]*/
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
	s_set_vgpr_msb 0xaaa6
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[2:9] /*v[514:521]*/, v[26:33] /*v[282:289]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa68a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[20:21] /*v[532:533]*/, v[18:19] /*v[530:531]*/
	v_cvt_pk_bf16_f32 v169 /*v681*/, v18 /*v530*/, v19 /*v531*/
	v_exp_f32_e32 v18 /*v530*/, v22 /*v534*/
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[10:17] /*v[522:529]*/, v[186:193] /*v[442:449]*/, v[2:9] /*v[770:777]*/
	s_set_vgpr_msb 0xf6aa
	v_pk_fma_f32 v[20:21] /*v[532:533]*/, v[118:119] /*v[630:631]*/, v[132:133] /*v[644:645]*/, v[200:201] /*v[712:713]*/
	v_pk_fma_f32 v[26:27] /*v[538:539]*/, v[120:121] /*v[632:633]*/, v[132:133] /*v[644:645]*/, v[200:201] /*v[712:713]*/
	v_exp_f32_e32 v19 /*v531*/, v23 /*v535*/
	s_set_vgpr_msb 0xaaf6
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[2:9] /*v[514:521]*/, v[66:73] /*v[322:329]*/, v[10:17] /*v[778:785]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xf68a
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[18:19] /*v[530:531]*/, v[20:21] /*v[532:533]*/
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_pk_bf16_f32 v170 /*v682*/, v2 /*v514*/, v3 /*v515*/
	v_exp_f32_e32 v2 /*v514*/, v24 /*v536*/
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[10:17] /*v[522:529]*/, v[226:233] /*v[482:489]*/, v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xf68a
	v_exp_f32_e32 v3 /*v515*/, v25 /*v537*/
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[2:3] /*v[514:515]*/, v[26:27] /*v[538:539]*/
	v_cvt_pk_bf16_f32 v171 /*v683*/, v2 /*v514*/, v3 /*v515*/
	v_add_nc_u32_e32 v2 /*v514*/, s9, v215 /*v727*/
	ds_load_tr16_b128 v[176:179] /*v[688:691]*/, v2 /*v514*/
	ds_load_tr16_b128 v[218:221] /*v[730:733]*/, v2 /*v514*/ offset:32
	ds_load_tr16_b128 v[180:183] /*v[692:695]*/, v2 /*v514*/ offset:4352
	ds_load_tr16_b128 v[222:225] /*v[734:737]*/, v2 /*v514*/ offset:4384
	s_set_vgpr_msb 0x8ac2
	ds_load_tr16_b128 v[26:29] /*v[794:797]*/, v2 /*v514*/ offset:64
	ds_load_tr16_b128 v[34:37] /*v[802:805]*/, v2 /*v514*/ offset:96
	ds_load_tr16_b128 v[30:33] /*v[798:801]*/, v2 /*v514*/ offset:4416
	ds_load_tr16_b128 v[38:41] /*v[806:809]*/, v2 /*v514*/ offset:4448
	ds_load_tr16_b128 v[42:45] /*v[810:813]*/, v2 /*v514*/ offset:128
	ds_load_tr16_b128 v[50:53] /*v[818:821]*/, v2 /*v514*/ offset:160
	ds_load_tr16_b128 v[46:49] /*v[814:817]*/, v2 /*v514*/ offset:4480
	ds_load_tr16_b128 v[54:57] /*v[822:825]*/, v2 /*v514*/ offset:4512
	ds_load_tr16_b128 v[58:61] /*v[826:829]*/, v2 /*v514*/ offset:192
	ds_load_tr16_b128 v[66:69] /*v[834:837]*/, v2 /*v514*/ offset:224
	ds_load_tr16_b128 v[62:65] /*v[830:833]*/, v2 /*v514*/ offset:4544
	ds_load_tr16_b128 v[70:73] /*v[838:841]*/, v2 /*v514*/ offset:4576
	s_wait_tensorcnt 0x2
	s_set_vgpr_msb 0xc2aa
	v_add_nc_u32_e32 v14 /*v526*/, s3, v216 /*v728*/
	ds_load_b128 v[114:117] /*v[626:629]*/, v14 /*v526*/
	ds_load_b128 v[118:121] /*v[630:633]*/, v14 /*v526*/ offset:32
	ds_load_b128 v[122:125] /*v[634:637]*/, v14 /*v526*/ offset:8704
	ds_load_b128 v[126:129] /*v[638:641]*/, v14 /*v526*/ offset:8736
	ds_load_b128 v[98:101] /*v[610:613]*/, v14 /*v526*/ offset:64
	ds_load_b128 v[102:105] /*v[614:617]*/, v14 /*v526*/ offset:96
	ds_load_b128 v[106:109] /*v[618:621]*/, v14 /*v526*/ offset:8768
	ds_load_b128 v[110:113] /*v[622:625]*/, v14 /*v526*/ offset:8800
	ds_load_b128 v[82:85] /*v[594:597]*/, v14 /*v526*/ offset:128
	ds_load_b128 v[86:89] /*v[598:601]*/, v14 /*v526*/ offset:160
	ds_load_b128 v[90:93] /*v[602:605]*/, v14 /*v526*/ offset:8832
	ds_load_b128 v[94:97] /*v[606:609]*/, v14 /*v526*/ offset:8864
	ds_load_b128 v[66:69] /*v[578:581]*/, v14 /*v526*/ offset:192
	ds_load_b128 v[70:73] /*v[582:585]*/, v14 /*v526*/ offset:224
	ds_load_b128 v[74:77] /*v[586:589]*/, v14 /*v526*/ offset:8896
	ds_load_b128 v[78:81] /*v[590:593]*/, v14 /*v526*/ offset:8928
	ds_load_b128 v[50:53] /*v[562:565]*/, v14 /*v526*/ offset:4352
	ds_load_b128 v[54:57] /*v[566:569]*/, v14 /*v526*/ offset:4384
	ds_load_b128 v[58:61] /*v[570:573]*/, v14 /*v526*/ offset:13056
	ds_load_b128 v[62:65] /*v[574:577]*/, v14 /*v526*/ offset:13088
	ds_load_b128 v[34:37] /*v[546:549]*/, v14 /*v526*/ offset:4416
	ds_load_b128 v[38:41] /*v[550:553]*/, v14 /*v526*/ offset:4448
	ds_load_b128 v[42:45] /*v[554:557]*/, v14 /*v526*/ offset:13120
	ds_load_b128 v[46:49] /*v[558:561]*/, v14 /*v526*/ offset:13152
	ds_load_b128 v[18:21] /*v[530:533]*/, v14 /*v526*/ offset:4480
	ds_load_b128 v[22:25] /*v[534:537]*/, v14 /*v526*/ offset:4512
	ds_load_b128 v[26:29] /*v[538:541]*/, v14 /*v526*/ offset:13184
	ds_load_b128 v[30:33] /*v[542:545]*/, v14 /*v526*/ offset:13216
	ds_load_b128 v[2:5] /*v[514:517]*/, v14 /*v526*/ offset:4544
	ds_load_b128 v[6:9] /*v[518:521]*/, v14 /*v526*/ offset:4576
	ds_load_b128 v[10:13] /*v[522:525]*/, v14 /*v526*/ offset:13248
	ds_load_b128 v[14:17] /*v[526:529]*/, v14 /*v526*/ offset:13280
	v_pk_fma_f32 v[148:149] /*v[660:661]*/, v[226:227] /*v[738:739]*/, v[130:131] /*v[642:643]*/, v[186:187] /*v[698:699]*/
	v_pk_fma_f32 v[150:151] /*v[662:663]*/, v[228:229] /*v[740:741]*/, v[130:131] /*v[642:643]*/, v[186:187] /*v[698:699]*/
	v_pk_fma_f32 v[156:157] /*v[668:669]*/, v[230:231] /*v[742:743]*/, v[130:131] /*v[642:643]*/, v[186:187] /*v[698:699]*/
	v_pk_fma_f32 v[158:159] /*v[670:671]*/, v[232:233] /*v[744:745]*/, v[130:131] /*v[642:643]*/, v[186:187] /*v[698:699]*/
	v_pk_fma_f32 v[164:165] /*v[676:677]*/, v[234:235] /*v[746:747]*/, v[132:133] /*v[644:645]*/, v[194:195] /*v[706:707]*/
	v_exp_f32_e32 v148 /*v660*/, v148 /*v660*/
	v_exp_f32_e32 v149 /*v661*/, v149 /*v661*/
	v_exp_f32_e32 v150 /*v662*/, v150 /*v662*/
	v_exp_f32_e32 v151 /*v663*/, v151 /*v663*/
	v_exp_f32_e32 v156 /*v668*/, v156 /*v668*/
	v_exp_f32_e32 v157 /*v669*/, v157 /*v669*/
	v_exp_f32_e32 v158 /*v670*/, v158 /*v670*/
	v_exp_f32_e32 v159 /*v671*/, v159 /*v671*/
	v_pk_fma_f32 v[166:167] /*v[678:679]*/, v[236:237] /*v[748:749]*/, v[132:133] /*v[644:645]*/, v[194:195] /*v[706:707]*/
	v_pk_fma_f32 v[172:173] /*v[684:685]*/, v[238:239] /*v[750:751]*/, v[132:133] /*v[644:645]*/, v[194:195] /*v[706:707]*/
	v_pk_fma_f32 v[174:175] /*v[686:687]*/, v[240:241] /*v[752:753]*/, v[132:133] /*v[644:645]*/, v[194:195] /*v[706:707]*/
	v_pk_mul_f32 v[148:149] /*v[660:661]*/, v[148:149] /*v[660:661]*/, v[164:165] /*v[676:677]*/
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_mul_f32 v[150:151] /*v[662:663]*/, v[150:151] /*v[662:663]*/, v[166:167] /*v[678:679]*/
	v_pk_mul_f32 v[156:157] /*v[668:669]*/, v[156:157] /*v[668:669]*/, v[172:173] /*v[684:685]*/
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_4)
	v_pk_mul_f32 v[158:159] /*v[670:671]*/, v[158:159] /*v[670:671]*/, v[174:175] /*v[686:687]*/
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_pk_bf16_f32 v148 /*v660*/, v148 /*v660*/, v149 /*v661*/
	v_cvt_pk_bf16_f32 v149 /*v661*/, v150 /*v662*/, v151 /*v663*/
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_pk_bf16_f32 v150 /*v662*/, v156 /*v668*/, v157 /*v669*/
	v_cvt_pk_bf16_f32 v151 /*v663*/, v158 /*v670*/, v159 /*v671*/
	s_set_vgpr_msb 0xaa5a
	s_wait_dscnt 0x2d
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[144:151] /*v[656:663]*/, v[176:183] /*v[688:695]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x5aaa
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[130:131] /*v[642:643]*/, v[188:189] /*v[700:701]*/
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[138:139] /*v[650:651]*/, v[130:131] /*v[642:643]*/, v[188:189] /*v[700:701]*/
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v136 /*v648*/, v136 /*v648*/
	s_set_vgpr_msb 0xaa5a
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[144:151] /*v[656:663]*/, v[218:225] /*v[730:737]*/, v[242:249] /*v[498:505]*/
	s_set_vgpr_msb 0x5aaa
	v_pk_fma_f32 v[140:141] /*v[652:653]*/, v[140:141] /*v[652:653]*/, v[130:131] /*v[642:643]*/, v[188:189] /*v[700:701]*/
	v_pk_fma_f32 v[142:143] /*v[654:655]*/, v[142:143] /*v[654:655]*/, v[130:131] /*v[642:643]*/, v[188:189] /*v[700:701]*/
	v_exp_f32_e32 v137 /*v649*/, v137 /*v649*/
	s_set_vgpr_msb 0xaa5e
	s_wait_dscnt 0x29
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[144:151] /*v[656:663]*/, v[26:33] /*v[794:801]*/, v[234:241] /*v[490:497]*/
	s_set_vgpr_msb 0x5eaa
	v_pk_fma_f32 v[156:157] /*v[668:669]*/, v[242:243] /*v[754:755]*/, v[132:133] /*v[644:645]*/, v[196:197] /*v[708:709]*/
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[156:157] /*v[668:669]*/
	v_exp_f32_e32 v138 /*v650*/, v138 /*v650*/
	s_set_vgpr_msb 0xaa5e
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[144:151] /*v[656:663]*/, v[34:41] /*v[802:809]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5eaa
	v_cvt_pk_bf16_f32 v156 /*v668*/, v136 /*v648*/, v137 /*v649*/
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[244:245] /*v[756:757]*/, v[132:133] /*v[644:645]*/, v[196:197] /*v[708:709]*/
	v_exp_f32_e32 v139 /*v651*/, v139 /*v651*/
	s_set_vgpr_msb 0xaa5e
	s_wait_dscnt 0x25
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[144:151] /*v[656:663]*/, v[42:49] /*v[810:817]*/, v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5e8a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[138:139] /*v[650:651]*/, v[136:137] /*v[648:649]*/
	v_cvt_pk_bf16_f32 v157 /*v669*/, v136 /*v648*/, v137 /*v649*/
	v_exp_f32_e32 v136 /*v648*/, v140 /*v652*/
	s_set_vgpr_msb 0x8a5e
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[144:151] /*v[656:663]*/, v[50:57] /*v[818:825]*/, v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5eaa
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[246:247] /*v[758:759]*/, v[132:133] /*v[644:645]*/, v[196:197] /*v[708:709]*/
	v_pk_fma_f32 v[164:165] /*v[676:677]*/, v[248:249] /*v[760:761]*/, v[132:133] /*v[644:645]*/, v[196:197] /*v[708:709]*/
	v_exp_f32_e32 v137 /*v649*/, v141 /*v653*/
	s_set_vgpr_msb 0xaa5e
	s_wait_dscnt 0x21
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[144:151] /*v[656:663]*/, v[58:65] /*v[826:833]*/, v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5e8a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[138:139] /*v[650:651]*/
	v_cvt_pk_bf16_f32 v158 /*v670*/, v136 /*v648*/, v137 /*v649*/
	v_exp_f32_e32 v136 /*v648*/, v142 /*v654*/
	s_set_vgpr_msb 0x8a5e
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[144:151] /*v[656:663]*/, v[66:73] /*v[834:841]*/, v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5e8a
	v_exp_f32_e32 v137 /*v649*/, v143 /*v655*/
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[164:165] /*v[676:677]*/
	v_cvt_pk_bf16_f32 v159 /*v671*/, v136 /*v648*/, v137 /*v649*/
	s_set_vgpr_msb 0x8a0a
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[186:193], v[152:159] /*v[664:671]*/, v[176:183] /*v[688:695]*/, v[186:193]
	s_set_vgpr_msb 0xaaa
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[250:251] /*v[762:763]*/, v[130:131] /*v[642:643]*/, v[190:191] /*v[702:703]*/
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[252:253] /*v[764:765]*/, v[130:131] /*v[642:643]*/, v[190:191] /*v[702:703]*/
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v136 /*v648*/, v136 /*v648*/
	s_set_vgpr_msb 0xaa0a
	v_wmma_f32_16x16x32_bf16 v[178:185], v[152:159] /*v[664:671]*/, v[218:225] /*v[730:737]*/, v[178:185]
	s_set_vgpr_msb 0xaaa
	v_pk_fma_f32 v[140:141] /*v[652:653]*/, v[254:255] /*v[766:767]*/, v[130:131] /*v[642:643]*/, v[190:191] /*v[702:703]*/
	s_set_vgpr_msb 0xaaab
	v_pk_fma_f32 v[142:143] /*v[654:655]*/, v[0:1] /*v[768:769]*/, v[130:131] /*v[642:643]*/, v[190:191] /*v[702:703]*/
	s_set_vgpr_msb 0xab82
	v_exp_f32_e32 v137 /*v649*/, v137 /*v649*/
	s_set_vgpr_msb 0x820e
	v_wmma_f32_16x16x32_bf16 v[170:177], v[152:159] /*v[664:671]*/, v[26:33] /*v[794:801]*/, v[170:177]
	s_set_vgpr_msb 0xeab
	v_pk_fma_f32 v[144:145] /*v[656:657]*/, v[2:3] /*v[770:771]*/, v[132:133] /*v[644:645]*/, v[198:199] /*v[710:711]*/
	s_set_vgpr_msb 0xab8a
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[144:145] /*v[656:657]*/
	v_exp_f32_e32 v138 /*v650*/, v138 /*v650*/
	s_set_vgpr_msb 0x8a0e
	v_wmma_f32_16x16x32_bf16 v[162:169], v[152:159] /*v[664:671]*/, v[34:41] /*v[802:809]*/, v[162:169]
	s_set_vgpr_msb 0xe8a
	v_cvt_pk_bf16_f32 v164 /*v676*/, v136 /*v648*/, v137 /*v649*/
	s_set_vgpr_msb 0x8aab
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[4:5] /*v[772:773]*/, v[132:133] /*v[644:645]*/, v[198:199] /*v[710:711]*/
	s_set_vgpr_msb 0xab82
	v_exp_f32_e32 v139 /*v651*/, v139 /*v651*/
	s_set_vgpr_msb 0x820e
	v_wmma_f32_16x16x32_bf16 v[154:161], v[152:159] /*v[664:671]*/, v[42:49] /*v[810:817]*/, v[154:161]
	s_set_vgpr_msb 0xe8a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[138:139] /*v[650:651]*/, v[136:137] /*v[648:649]*/
	v_cvt_pk_bf16_f32 v165 /*v677*/, v136 /*v648*/, v137 /*v649*/
	v_exp_f32_e32 v136 /*v648*/, v140 /*v652*/
	s_set_vgpr_msb 0x8a0e
	v_wmma_f32_16x16x32_bf16 v[146:153], v[152:159] /*v[664:671]*/, v[50:57] /*v[818:825]*/, v[146:153]
	s_set_vgpr_msb 0xeab
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[6:7] /*v[774:775]*/, v[132:133] /*v[644:645]*/, v[198:199] /*v[710:711]*/
	v_pk_fma_f32 v[144:145] /*v[656:657]*/, v[8:9] /*v[776:777]*/, v[132:133] /*v[644:645]*/, v[198:199] /*v[710:711]*/
	s_set_vgpr_msb 0xab82
	v_exp_f32_e32 v137 /*v649*/, v141 /*v653*/
	s_set_vgpr_msb 0x820e
	v_wmma_f32_16x16x32_bf16 v[138:145], v[152:159] /*v[664:671]*/, v[58:65] /*v[826:833]*/, v[138:145]
	s_set_vgpr_msb 0xe8a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[138:139] /*v[650:651]*/
	v_cvt_pk_bf16_f32 v166 /*v678*/, v136 /*v648*/, v137 /*v649*/
	v_exp_f32_e32 v136 /*v648*/, v142 /*v654*/
	s_set_vgpr_msb 0x8a0e
	v_wmma_f32_16x16x32_bf16 v[130:137], v[152:159] /*v[664:671]*/, v[66:73] /*v[834:841]*/, v[130:137]
	s_set_vgpr_msb 0xe8a
	v_exp_f32_e32 v137 /*v649*/, v143 /*v655*/
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[144:145] /*v[656:657]*/
	v_cvt_pk_bf16_f32 v167 /*v679*/, v136 /*v648*/, v137 /*v649*/
	s_set_vgpr_msb 0x8a0a
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[122:129], v[160:167] /*v[672:679]*/, v[176:183] /*v[688:695]*/, v[122:129]
	s_set_vgpr_msb 0xaab
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[10:11] /*v[778:779]*/, v[130:131] /*v[642:643]*/, v[192:193] /*v[704:705]*/
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[12:13] /*v[780:781]*/, v[130:131] /*v[642:643]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0xab82
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v136 /*v648*/, v136 /*v648*/
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[114:121], v[160:167] /*v[672:679]*/, v[218:225] /*v[730:737]*/, v[114:121]
	s_set_vgpr_msb 0xaab
	v_pk_fma_f32 v[140:141] /*v[652:653]*/, v[14:15] /*v[782:783]*/, v[130:131] /*v[642:643]*/, v[192:193] /*v[704:705]*/
	v_pk_fma_f32 v[142:143] /*v[654:655]*/, v[16:17] /*v[784:785]*/, v[130:131] /*v[642:643]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0xab82
	v_exp_f32_e32 v137 /*v649*/, v137 /*v649*/
	s_set_vgpr_msb 0x820e
	v_wmma_f32_16x16x32_bf16 v[106:113], v[160:167] /*v[672:679]*/, v[26:33] /*v[794:801]*/, v[106:113]
	s_set_vgpr_msb 0xeab
	v_pk_fma_f32 v[144:145] /*v[656:657]*/, v[18:19] /*v[786:787]*/, v[132:133] /*v[644:645]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0xab8a
	s_delay_alu instid0(TRANS32_DEP_2) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[144:145] /*v[656:657]*/
	v_exp_f32_e32 v138 /*v650*/, v138 /*v650*/
	s_set_vgpr_msb 0x8a0e
	v_wmma_f32_16x16x32_bf16 v[98:105], v[160:167] /*v[672:679]*/, v[34:41] /*v[802:809]*/, v[98:105]
	s_set_vgpr_msb 0xe8a
	v_cvt_pk_bf16_f32 v172 /*v684*/, v136 /*v648*/, v137 /*v649*/
	s_set_vgpr_msb 0x8aab
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[20:21] /*v[788:789]*/, v[132:133] /*v[644:645]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0xab82
	v_exp_f32_e32 v139 /*v651*/, v139 /*v651*/
	s_set_vgpr_msb 0x820e
	v_wmma_f32_16x16x32_bf16 v[90:97], v[160:167] /*v[672:679]*/, v[42:49] /*v[810:817]*/, v[90:97]
	s_set_vgpr_msb 0xe8a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[138:139] /*v[650:651]*/, v[136:137] /*v[648:649]*/
	v_cvt_pk_bf16_f32 v173 /*v685*/, v136 /*v648*/, v137 /*v649*/
	v_exp_f32_e32 v136 /*v648*/, v140 /*v652*/
	s_set_vgpr_msb 0x8a0e
	v_wmma_f32_16x16x32_bf16 v[82:89], v[160:167] /*v[672:679]*/, v[50:57] /*v[818:825]*/, v[82:89]
	s_set_vgpr_msb 0xeab
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[22:23] /*v[790:791]*/, v[132:133] /*v[644:645]*/, v[200:201] /*v[712:713]*/
	v_pk_fma_f32 v[144:145] /*v[656:657]*/, v[24:25] /*v[792:793]*/, v[132:133] /*v[644:645]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0xab82
	v_exp_f32_e32 v137 /*v649*/, v141 /*v653*/
	s_set_vgpr_msb 0x820e
	v_wmma_f32_16x16x32_bf16 v[74:81], v[160:167] /*v[672:679]*/, v[58:65] /*v[826:833]*/, v[74:81]
	s_set_vgpr_msb 0xe8a
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[138:139] /*v[650:651]*/
	v_cvt_pk_bf16_f32 v174 /*v686*/, v136 /*v648*/, v137 /*v649*/
	v_exp_f32_e32 v136 /*v648*/, v142 /*v654*/
	s_set_vgpr_msb 0x8a0e
	v_wmma_f32_16x16x32_bf16 v[66:73], v[160:167] /*v[672:679]*/, v[66:73] /*v[834:841]*/, v[66:73]
	s_set_vgpr_msb 0xe8a
	v_exp_f32_e32 v137 /*v649*/, v143 /*v655*/
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[136:137] /*v[648:649]*/, v[136:137] /*v[648:649]*/, v[144:145] /*v[656:657]*/
	v_cvt_pk_bf16_f32 v175 /*v687*/, v136 /*v648*/, v137 /*v649*/
	s_set_vgpr_msb 0x8a0a
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[58:65], v[168:175] /*v[680:687]*/, v[176:183] /*v[688:695]*/, v[58:65]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[168:175] /*v[680:687]*/, v[218:225] /*v[730:737]*/, v[50:57]
	s_set_vgpr_msb 0xa0e
	v_wmma_f32_16x16x32_bf16 v[42:49], v[168:175] /*v[680:687]*/, v[26:33] /*v[794:801]*/, v[42:49]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[168:175] /*v[680:687]*/, v[34:41] /*v[802:809]*/, v[34:41]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[168:175] /*v[680:687]*/, v[42:49] /*v[810:817]*/, v[26:33]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[168:175] /*v[680:687]*/, v[50:57] /*v[818:825]*/, v[18:25]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[168:175] /*v[680:687]*/, v[58:65] /*v[826:833]*/, v[10:17]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[168:175] /*v[680:687]*/, v[66:73] /*v[834:841]*/, v[2:9]
	s_add_nc_u64 s[10:11], s[10:11], -1
	s_add_co_i32 s12, s12, 1
	s_cmp_lg_u64 s[10:11], 0
	s_set_vgpr_msb 0xe00
	s_cbranch_scc1 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	v_mov_b32_e32 v2, 0
	s_mov_b32 s3, s71
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
	v_mov_b64_e32 v[192:193], v[8:9]
	v_mov_b64_e32 v[190:191], v[6:7]
	v_mov_b64_e32 v[188:189], v[4:5]
	v_mov_b64_e32 v[186:187], v[2:3]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[48:49] /*v[304:305]*/, v[8:9]
	v_mov_b64_e32 v[46:47] /*v[302:303]*/, v[6:7]
	v_mov_b64_e32 v[44:45] /*v[300:301]*/, v[4:5]
	v_mov_b64_e32 v[42:43] /*v[298:299]*/, v[2:3]
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[8:9]
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[6:7]
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[4:5]
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[2:3]
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[8:9]
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[6:7]
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[4:5]
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[2:3]
	v_mov_b64_e32 v[184:185] /*v[440:441]*/, v[8:9]
	v_mov_b64_e32 v[182:183] /*v[438:439]*/, v[6:7]
	v_mov_b64_e32 v[180:181] /*v[436:437]*/, v[4:5]
	v_mov_b64_e32 v[178:179] /*v[434:435]*/, v[2:3]
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[8:9]
	v_mov_b64_e32 v[214:215] /*v[470:471]*/, v[6:7]
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[4:5]
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[2:3]
	v_mov_b64_e32 v[240:241] /*v[496:497]*/, v[8:9]
	v_mov_b64_e32 v[238:239] /*v[494:495]*/, v[6:7]
	v_mov_b64_e32 v[236:237] /*v[492:493]*/, v[4:5]
	v_mov_b64_e32 v[234:235] /*v[490:491]*/, v[2:3]
	v_mov_b64_e32 v[248:249] /*v[504:505]*/, v[8:9]
	v_mov_b64_e32 v[246:247] /*v[502:503]*/, v[6:7]
	v_mov_b64_e32 v[244:245] /*v[500:501]*/, v[4:5]
	v_mov_b64_e32 v[242:243] /*v[498:499]*/, v[2:3]
	s_set_vgpr_msb 0x4080
	v_mov_b64_e32 v[0:1] /*v[512:513]*/, v[8:9]
	s_set_vgpr_msb 0x8040
	v_mov_b64_e32 v[254:255] /*v[510:511]*/, v[6:7]
	v_mov_b64_e32 v[252:253] /*v[508:509]*/, v[4:5]
	v_mov_b64_e32 v[250:251] /*v[506:507]*/, v[2:3]
	s_set_vgpr_msb 0x4000
.LBB0_4:
	s_sub_co_i32 s84, s25, s2
	s_mov_b32 s72, 1
	s_cmp_lt_i32 s84, 1
	s_cbranch_scc1 .LBB0_7
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v202 /*v714*/, s24, v209 /*v721*/
	s_set_vgpr_msb 0x8880
	v_add_nc_u32_e32 v204 /*v716*/, s63, v1
	s_set_vgpr_msb 0x808a
	v_dual_add_nc_u32 v206 /*v718*/, s63, v134 /*v646*/ :: v_dual_add_nc_u32 v208 /*v720*/, s63, v135 /*v647*/
	s_mov_b32 s57, s56
	s_mov_b32 s9, s8
	v_mov_b64_e32 v[212:213] /*v[724:725]*/, s[56:57]
	v_mov_b64_e32 v[210:211] /*v[722:723]*/, s[8:9]
	v_dual_mov_b32 v201 /*v713*/, v200 /*v712*/ :: v_dual_mov_b32 v199 /*v711*/, v198 /*v710*/
	v_dual_mov_b32 v197 /*v709*/, v196 /*v708*/ :: v_dual_mov_b32 v195 /*v707*/, v194 /*v706*/
	v_mov_b32_e32 v187 /*v699*/, v186 /*v698*/
	s_set_vgpr_msb 0x8a02
	v_mov_b32_e32 v1, v202 /*v714*/
	s_set_vgpr_msb 0x2a2
	v_dual_mov_b32 v189 /*v701*/, v188 /*v700*/ :: v_dual_mov_b32 v203 /*v715*/, v204 /*v716*/
	v_dual_mov_b32 v191 /*v703*/, v190 /*v702*/ :: v_dual_mov_b32 v205 /*v717*/, v206 /*v718*/
	v_dual_mov_b32 v193 /*v705*/, v192 /*v704*/ :: v_dual_mov_b32 v207 /*v719*/, v208 /*v720*/
	v_lshl_or_b32 v217 /*v729*/, s2, 5, v214 /*v726*/
	s_ashr_i32 s85, s84, 31
	s_add_co_i32 s63, s2, 2
	s_mov_b32 s71, 0
	s_mov_b32 s68, 32
	s_mov_b32 s65, 0xffff0000
	s_mov_b32 s64, 0x7510000
	s_set_vgpr_msb 0xa200
	s_wait_dscnt 0x0
.LBB0_6:
	s_min_i32 s2, s63, s83
	s_add_co_i32 s4, s3, 0xffffbc00
	s_cmp_lg_u32 s3, 0
	s_cselect_b32 s73, s4, 0x8800
	s_add_co_i32 s4, s3, 0x4400
	s_cmp_lg_u32 s3, 0x8800
	s_cselect_b32 s92, s4, 0
	s_lshl_b32 s2, s2, 5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s4, s2, s82
	s_sub_co_i32 s2, s58, s2
	s_ashr_i32 s5, s4, 31
	s_max_i32 s2, s2, 0
	s_mul_u64 s[4:5], s[4:5], s[60:61]
	s_lshl_b32 s6, s2, 16
	s_add_nc_u64 s[4:5], s[4:5], s[80:81]
	s_lshr_b32 s2, s2, 16
	s_lshl_b64 s[4:5], s[4:5], 8
	s_or_b32 s66, s6, 0x7fff
	s_add_nc_u64 s[74:75], s[76:77], s[4:5]
	s_or_b32 s67, s2, 0x800000
	s_bitset1_b32 s75, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[72:75], s[64:71]
	s_add_nc_u64 s[74:75], s[78:79], s[4:5]
	s_addk_co_i32 s73, 0x2200
	s_bitset1_b32 s75, 31
	tensor_load_to_lds s[72:75], s[64:71]
	s_set_vgpr_msb 0x82
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[114:121] /*v[626:633]*/, v[194:201], 0
	s_set_vgpr_msb 0x8286
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[122:129] /*v[634:641]*/, v[74:81] /*v[330:337]*/, 0
	s_set_vgpr_msb 0x8682
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[114:121] /*v[626:633]*/, v[226:233], 0
	s_set_vgpr_msb 0x8286
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[122:129] /*v[634:641]*/, v[114:121] /*v[370:377]*/, 0
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[114:121] /*v[626:633]*/, v[2:9] /*v[258:265]*/, 0
	s_set_vgpr_msb 0x86c6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[122:129] /*v[634:641]*/, v[154:161] /*v[410:417]*/, 0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[114:121] /*v[626:633]*/, v[34:41] /*v[290:297]*/, 0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[122:129] /*v[634:641]*/, v[194:201] /*v[450:457]*/, 0
	s_set_vgpr_msb 0xc6c2
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[794:801]*/, v[50:57] /*v[562:569]*/, v[194:201], 0
	s_set_vgpr_msb 0xc2c6
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[802:809]*/, v[58:65] /*v[570:577]*/, v[74:81] /*v[330:337]*/, 0
	s_set_vgpr_msb 0xc6c2
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[810:817]*/, v[50:57] /*v[562:569]*/, v[226:233], 0
	s_set_vgpr_msb 0xc2c6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[818:825]*/, v[58:65] /*v[570:577]*/, v[114:121] /*v[370:377]*/, 0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[826:833]*/, v[50:57] /*v[562:569]*/, v[2:9] /*v[258:265]*/, 0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[834:841]*/, v[58:65] /*v[570:577]*/, v[154:161] /*v[410:417]*/, 0
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[842:849]*/, v[50:57] /*v[562:569]*/, v[34:41] /*v[290:297]*/, 0
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[850:857]*/, v[58:65] /*v[570:577]*/, v[194:201] /*v[450:457]*/, 0
	s_set_vgpr_msb 0xc6a2
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[98:105] /*v[610:617]*/, v[202:209], v[218:225] /*v[730:737]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[106:113] /*v[618:625]*/, v[82:89] /*v[338:345]*/, v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[98:105] /*v[610:617]*/, v[234:241], v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[106:113] /*v[618:625]*/, v[122:129] /*v[378:385]*/, v[242:249] /*v[754:761]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[98:105] /*v[610:617]*/, v[10:17] /*v[266:273]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[106:113] /*v[618:625]*/, v[162:169] /*v[418:425]*/, v[2:9] /*v[770:777]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[98:105] /*v[610:617]*/, v[50:57] /*v[306:313]*/, v[10:17] /*v[778:785]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[106:113] /*v[618:625]*/, v[202:209] /*v[458:465]*/, v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xf6f2
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[794:801]*/, v[34:41] /*v[546:553]*/, v[202:209], v[26:33] /*v[794:801]*/
	s_set_vgpr_msb 0xf2f6
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[802:809]*/, v[42:49] /*v[554:561]*/, v[82:89] /*v[338:345]*/, v[34:41] /*v[802:809]*/
	s_set_vgpr_msb 0xf6f2
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[810:817]*/, v[34:41] /*v[546:553]*/, v[234:241], v[42:49] /*v[810:817]*/
	s_set_vgpr_msb 0xf2f6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[818:825]*/, v[42:49] /*v[554:561]*/, v[122:129] /*v[378:385]*/, v[50:57] /*v[818:825]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[826:833]*/, v[34:41] /*v[546:553]*/, v[10:17] /*v[266:273]*/, v[58:65] /*v[826:833]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[834:841]*/, v[42:49] /*v[554:561]*/, v[162:169] /*v[418:425]*/, v[66:73] /*v[834:841]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[842:849]*/, v[34:41] /*v[546:553]*/, v[50:57] /*v[306:313]*/, v[74:81] /*v[842:849]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[850:857]*/, v[42:49] /*v[554:561]*/, v[202:209] /*v[458:465]*/, v[82:89] /*v[850:857]*/
	s_set_vgpr_msb 0xf6a2
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[82:89] /*v[594:601]*/, v[210:217], v[218:225] /*v[730:737]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[90:97] /*v[602:609]*/, v[90:97] /*v[346:353]*/, v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[82:89] /*v[594:601]*/, v[242:249], v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[90:97] /*v[602:609]*/, v[130:137] /*v[386:393]*/, v[242:249] /*v[754:761]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[82:89] /*v[594:601]*/, v[18:25] /*v[274:281]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[90:97] /*v[602:609]*/, v[170:177] /*v[426:433]*/, v[2:9] /*v[770:777]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[82:89] /*v[594:601]*/, v[58:65] /*v[314:321]*/, v[10:17] /*v[778:785]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[90:97] /*v[602:609]*/, v[218:225] /*v[474:481]*/, v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xf6f2
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[794:801]*/, v[18:25] /*v[530:537]*/, v[210:217], v[26:33] /*v[794:801]*/
	s_set_vgpr_msb 0xf2f6
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[802:809]*/, v[26:33] /*v[538:545]*/, v[90:97] /*v[346:353]*/, v[34:41] /*v[802:809]*/
	s_set_vgpr_msb 0xf6f2
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[810:817]*/, v[18:25] /*v[530:537]*/, v[242:249], v[42:49] /*v[810:817]*/
	s_set_vgpr_msb 0xf2f6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[818:825]*/, v[26:33] /*v[538:545]*/, v[130:137] /*v[386:393]*/, v[50:57] /*v[818:825]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[826:833]*/, v[18:25] /*v[530:537]*/, v[18:25] /*v[274:281]*/, v[58:65] /*v[826:833]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[834:841]*/, v[26:33] /*v[538:545]*/, v[170:177] /*v[426:433]*/, v[66:73] /*v[834:841]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[842:849]*/, v[18:25] /*v[530:537]*/, v[58:65] /*v[314:321]*/, v[74:81] /*v[842:849]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[850:857]*/, v[26:33] /*v[538:545]*/, v[218:225] /*v[474:481]*/, v[82:89] /*v[850:857]*/
	s_set_vgpr_msb 0xf6a2
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[66:73] /*v[578:585]*/, v[218:225], v[218:225] /*v[730:737]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[74:81] /*v[586:593]*/, v[106:113] /*v[362:369]*/, v[226:233] /*v[738:745]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[66:73] /*v[578:585]*/, v[250:257], v[234:241] /*v[746:753]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[74:81] /*v[586:593]*/, v[146:153] /*v[402:409]*/, v[242:249] /*v[754:761]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[66:73] /*v[578:585]*/, v[26:33] /*v[282:289]*/, v[250:257] /*v[762:769]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[770:777]*/, v[74:81] /*v[586:593]*/, v[186:193] /*v[442:449]*/, v[2:9] /*v[770:777]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[778:785]*/, v[66:73] /*v[578:585]*/, v[66:73] /*v[322:329]*/, v[10:17] /*v[778:785]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[786:793]*/, v[74:81] /*v[586:593]*/, v[226:233] /*v[482:489]*/, v[18:25] /*v[786:793]*/
	s_set_vgpr_msb 0xf6f2
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[794:801]*/, v[2:9] /*v[514:521]*/, v[218:225], v[26:33] /*v[794:801]*/
	s_set_vgpr_msb 0xf2f6
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[802:809]*/, v[10:17] /*v[522:529]*/, v[106:113] /*v[362:369]*/, v[34:41] /*v[802:809]*/
	s_set_vgpr_msb 0xf6f2
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[810:817]*/, v[2:9] /*v[514:521]*/, v[250:257], v[42:49] /*v[810:817]*/
	s_set_vgpr_msb 0xf2f6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[818:825]*/, v[10:17] /*v[522:529]*/, v[146:153] /*v[402:409]*/, v[50:57] /*v[818:825]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[826:833]*/, v[2:9] /*v[514:521]*/, v[26:33] /*v[282:289]*/, v[58:65] /*v[826:833]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[834:841]*/, v[10:17] /*v[522:529]*/, v[186:193] /*v[442:449]*/, v[66:73] /*v[834:841]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[842:849]*/, v[2:9] /*v[514:521]*/, v[66:73] /*v[322:329]*/, v[74:81] /*v[842:849]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[850:857]*/, v[10:17] /*v[522:529]*/, v[226:233] /*v[482:489]*/, v[82:89] /*v[850:857]*/
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xf68a
	v_add_nc_u32_e32 v2 /*v514*/, s3, v215 /*v727*/
	ds_load_tr16_b128 v[138:141] /*v[650:653]*/, v2 /*v514*/
	ds_load_tr16_b128 v[130:133] /*v[642:645]*/, v2 /*v514*/ offset:32
	ds_load_tr16_b128 v[142:145] /*v[654:657]*/, v2 /*v514*/ offset:4352
	ds_load_tr16_b128 v[134:137] /*v[646:649]*/, v2 /*v514*/ offset:4384
	ds_load_tr16_b128 v[146:149] /*v[658:661]*/, v2 /*v514*/ offset:64
	ds_load_tr16_b128 v[154:157] /*v[666:669]*/, v2 /*v514*/ offset:96
	ds_load_tr16_b128 v[150:153] /*v[662:665]*/, v2 /*v514*/ offset:4416
	ds_load_tr16_b128 v[158:161] /*v[670:673]*/, v2 /*v514*/ offset:4448
	ds_load_tr16_b128 v[162:165] /*v[674:677]*/, v2 /*v514*/ offset:128
	ds_load_tr16_b128 v[170:173] /*v[682:685]*/, v2 /*v514*/ offset:160
	ds_load_tr16_b128 v[166:169] /*v[678:681]*/, v2 /*v514*/ offset:4480
	ds_load_tr16_b128 v[174:177] /*v[686:689]*/, v2 /*v514*/ offset:4512
	ds_load_tr16_b128 v[178:181] /*v[690:693]*/, v2 /*v514*/ offset:192
	s_set_vgpr_msb 0x8ac2
	ds_load_tr16_b128 v[90:93] /*v[858:861]*/, v2 /*v514*/ offset:224
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[182:185] /*v[694:697]*/, v2 /*v514*/ offset:4544
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[94:97] /*v[862:865]*/, v2 /*v514*/ offset:4576
	s_wait_tensorcnt 0x2
	s_set_vgpr_msb 0xc28a
	v_add_nc_u32_e32 v14 /*v526*/, s92, v216 /*v728*/
	ds_load_b128 v[114:117] /*v[626:629]*/, v14 /*v526*/
	ds_load_b128 v[118:121] /*v[630:633]*/, v14 /*v526*/ offset:32
	ds_load_b128 v[122:125] /*v[634:637]*/, v14 /*v526*/ offset:8704
	ds_load_b128 v[126:129] /*v[638:641]*/, v14 /*v526*/ offset:8736
	ds_load_b128 v[98:101] /*v[610:613]*/, v14 /*v526*/ offset:64
	ds_load_b128 v[102:105] /*v[614:617]*/, v14 /*v526*/ offset:96
	ds_load_b128 v[106:109] /*v[618:621]*/, v14 /*v526*/ offset:8768
	ds_load_b128 v[110:113] /*v[622:625]*/, v14 /*v526*/ offset:8800
	ds_load_b128 v[82:85] /*v[594:597]*/, v14 /*v526*/ offset:128
	ds_load_b128 v[86:89] /*v[598:601]*/, v14 /*v526*/ offset:160
	ds_load_b128 v[90:93] /*v[602:605]*/, v14 /*v526*/ offset:8832
	ds_load_b128 v[94:97] /*v[606:609]*/, v14 /*v526*/ offset:8864
	ds_load_b128 v[66:69] /*v[578:581]*/, v14 /*v526*/ offset:192
	ds_load_b128 v[70:73] /*v[582:585]*/, v14 /*v526*/ offset:224
	ds_load_b128 v[74:77] /*v[586:589]*/, v14 /*v526*/ offset:8896
	ds_load_b128 v[78:81] /*v[590:593]*/, v14 /*v526*/ offset:8928
	ds_load_b128 v[50:53] /*v[562:565]*/, v14 /*v526*/ offset:4352
	ds_load_b128 v[54:57] /*v[566:569]*/, v14 /*v526*/ offset:4384
	ds_load_b128 v[58:61] /*v[570:573]*/, v14 /*v526*/ offset:13056
	ds_load_b128 v[62:65] /*v[574:577]*/, v14 /*v526*/ offset:13088
	ds_load_b128 v[34:37] /*v[546:549]*/, v14 /*v526*/ offset:4416
	ds_load_b128 v[38:41] /*v[550:553]*/, v14 /*v526*/ offset:4448
	ds_load_b128 v[42:45] /*v[554:557]*/, v14 /*v526*/ offset:13120
	ds_load_b128 v[46:49] /*v[558:561]*/, v14 /*v526*/ offset:13152
	ds_load_b128 v[18:21] /*v[530:533]*/, v14 /*v526*/ offset:4480
	ds_load_b128 v[22:25] /*v[534:537]*/, v14 /*v526*/ offset:4512
	ds_load_b128 v[26:29] /*v[538:541]*/, v14 /*v526*/ offset:13184
	ds_load_b128 v[30:33] /*v[542:545]*/, v14 /*v526*/ offset:13216
	ds_load_b128 v[2:5] /*v[514:517]*/, v14 /*v526*/ offset:4544
	ds_load_b128 v[6:9] /*v[518:521]*/, v14 /*v526*/ offset:4576
	ds_load_b128 v[10:13] /*v[522:525]*/, v14 /*v526*/ offset:13248
	ds_load_b128 v[14:17] /*v[526:529]*/, v14 /*v526*/ offset:13280
	s_set_vgpr_msb 0x8ac8
	v_or_b32_e32 v98 /*v866*/, 3, v217 /*v729*/
	v_or_b32_e32 v99 /*v867*/, 2, v217 /*v729*/
	v_or_b32_e32 v100 /*v868*/, 5, v217 /*v729*/
	v_or_b32_e32 v101 /*v869*/, 4, v217 /*v729*/
	s_set_vgpr_msb 0xc8aa
	v_pk_fma_f32 v[220:221] /*v[732:733]*/, v[220:221] /*v[732:733]*/, v[210:211] /*v[722:723]*/, v[186:187] /*v[698:699]*/
	s_set_vgpr_msb 0xaa03
	v_cmp_gt_i32_e64 s21, v98 /*v866*/, v1
	s_set_vgpr_msb 0x3cb
	v_cmp_gt_i32_e64 s22, v99 /*v867*/, v202 /*v714*/
	v_or_b32_e32 v102 /*v870*/, 7, v217 /*v729*/
	s_set_vgpr_msb 0xcb03
	v_cmp_gt_i32_e64 s23, v100 /*v868*/, v1
	s_set_vgpr_msb 0x3c8
	v_or_b32_e32 v103 /*v871*/, 6, v217 /*v729*/
	s_and_b32 s21, s90, s21
	s_set_vgpr_msb 0xc8aa
	v_pk_fma_f32 v[222:223] /*v[734:735]*/, v[222:223] /*v[734:735]*/, v[210:211] /*v[722:723]*/, v[186:187] /*v[698:699]*/
	s_set_vgpr_msb 0xaa0b
	v_cmp_gt_i32_e64 s24, v101 /*v869*/, v202 /*v714*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v221 /*v733*/, v221 /*v733*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s22
	s_set_vgpr_msb 0x8203
	v_cmp_gt_i32_e64 s25, v102 /*v870*/, v1
	s_set_vgpr_msb 0x3aa
	v_cndmask_b32_e64 v220 /*v732*/, v220 /*v732*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s23
	v_pk_fma_f32 v[224:225] /*v[736:737]*/, v[224:225] /*v[736:737]*/, v[210:211] /*v[722:723]*/, v[186:187] /*v[698:699]*/
	s_set_vgpr_msb 0xaa0b
	v_cmp_gt_i32_e64 s26, v103 /*v871*/, v202 /*v714*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v223 /*v735*/, v223 /*v735*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s24
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s27, v98 /*v866*/, v203 /*v715*/
	s_set_vgpr_msb 0xbaa
	v_cndmask_b32_e64 v222 /*v734*/, v222 /*v734*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s25
	v_pk_fma_f32 v[236:237] /*v[748:749]*/, v[236:237] /*v[748:749]*/, v[210:211] /*v[722:723]*/, v[188:189] /*v[700:701]*/
	s_set_vgpr_msb 0xaa0b
	v_cmp_gt_i32_e64 s28, v99 /*v867*/, v204 /*v716*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v225 /*v737*/, v225 /*v737*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s26
	v_cmp_ge_i32_e64 s2, v217 /*v729*/, v202 /*v714*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s29, v100 /*v868*/, v203 /*v715*/
	s_set_vgpr_msb 0xbaa
	v_cndmask_b32_e64 v224 /*v736*/, v224 /*v736*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s27
	v_pk_fma_f32 v[218:219] /*v[730:731]*/, v[218:219] /*v[730:731]*/, v[210:211] /*v[722:723]*/, v[186:187] /*v[698:699]*/
	v_cmp_gt_i32_e32 vcc_lo, v217 /*v729*/, v202 /*v714*/
	v_pk_fma_f32 v[238:239] /*v[750:751]*/, v[238:239] /*v[750:751]*/, v[210:211] /*v[722:723]*/, v[188:189] /*v[700:701]*/
	s_set_vgpr_msb 0xaa0b
	v_cmp_gt_i32_e64 s30, v101 /*v869*/, v204 /*v716*/
	v_cmp_gt_i32_e64 s34, v98 /*v866*/, v205 /*v717*/
	v_cmp_gt_i32_e64 s40, v98 /*v866*/, v207 /*v719*/
	s_set_vgpr_msb 0xbca
	v_cndmask_b32_e64 v98 /*v866*/, v237 /*v749*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s28
	v_cmp_ge_i32_e64 s4, v217 /*v729*/, v204 /*v716*/
	s_set_vgpr_msb 0xca0b
	v_cmp_gt_i32_e64 s31, v102 /*v870*/, v203 /*v715*/
	s_and_b32 s2, s90, s2
	v_cmp_gt_i32_e64 s35, v99 /*v867*/, v206 /*v718*/
	v_cmp_gt_i32_e64 s41, v99 /*v867*/, v208 /*v720*/
	s_set_vgpr_msb 0xbc2
	v_cndmask_b32_e64 v99 /*v867*/, v236 /*v748*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s29
	s_set_vgpr_msb 0xc2aa
	v_pk_fma_f32 v[234:235] /*v[746:747]*/, v[234:235] /*v[746:747]*/, v[210:211] /*v[722:723]*/, v[188:189] /*v[700:701]*/
	v_cmp_gt_i32_e64 s3, v217 /*v729*/, v204 /*v716*/
	v_pk_fma_f32 v[240:241] /*v[752:753]*/, v[240:241] /*v[752:753]*/, v[210:211] /*v[722:723]*/, v[188:189] /*v[700:701]*/
	s_set_vgpr_msb 0xaa0b
	v_cmp_gt_i32_e64 s33, v103 /*v871*/, v204 /*v716*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v219 /*v731*/, v219 /*v731*/, 0xff61b1e6, s2
	s_and_b32 s2, s90, vcc_lo
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s36, v100 /*v868*/, v205 /*v717*/
	v_cmp_gt_i32_e64 s12, v100 /*v868*/, v207 /*v719*/
	s_set_vgpr_msb 0xbca
	v_cndmask_b32_e64 v100 /*v868*/, v239 /*v751*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s30
	v_cmp_ge_i32_e64 s6, v217 /*v729*/, v206 /*v718*/
	s_set_vgpr_msb 0xca82
	v_cndmask_b32_e64 v218 /*v730*/, v218 /*v730*/, 0xff61b1e6, s2
	s_and_b32 s2, s90, s4
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s37, v101 /*v869*/, v206 /*v718*/
	v_cmp_gt_i32_e64 s11, v101 /*v869*/, v208 /*v720*/
	s_set_vgpr_msb 0xbc2
	v_cndmask_b32_e64 v101 /*v869*/, v238 /*v750*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s31
	s_set_vgpr_msb 0xc2aa
	v_pk_fma_f32 v[250:251] /*v[762:763]*/, v[250:251] /*v[762:763]*/, v[210:211] /*v[722:723]*/, v[190:191] /*v[702:703]*/
	v_cmp_gt_i32_e64 s5, v217 /*v729*/, v206 /*v718*/
	v_pk_fma_f32 v[252:253] /*v[764:765]*/, v[252:253] /*v[764:765]*/, v[210:211] /*v[722:723]*/, v[190:191] /*v[702:703]*/
	s_set_vgpr_msb 0xaac8
	v_or_b32_e32 v104 /*v872*/, 17, v217 /*v729*/
	s_set_vgpr_msb 0xc882
	v_cndmask_b32_e64 v235 /*v747*/, v235 /*v747*/, 0xff61b1e6, s2
	s_and_b32 s2, s90, s3
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s38, v102 /*v870*/, v205 /*v717*/
	v_cmp_gt_i32_e64 s10, v102 /*v870*/, v207 /*v719*/
	s_set_vgpr_msb 0xbca
	v_cndmask_b32_e64 v102 /*v870*/, v241 /*v753*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s33
	v_cmp_ge_i32_e64 s8, v217 /*v729*/, v208 /*v720*/
	v_or_b32_e32 v105 /*v873*/, 16, v217 /*v729*/
	s_set_vgpr_msb 0xca82
	v_cndmask_b32_e64 v234 /*v746*/, v234 /*v746*/, 0xff61b1e6, s2
	s_and_b32 s2, s90, s6
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s39, v103 /*v871*/, v206 /*v718*/
	v_cmp_gt_i32_e32 vcc_lo, v103 /*v871*/, v208 /*v720*/
	s_set_vgpr_msb 0xbc2
	v_cndmask_b32_e64 v103 /*v871*/, v240 /*v752*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s34
	s_set_vgpr_msb 0xc2aa
	v_pk_fma_f32 v[254:255] /*v[766:767]*/, v[254:255] /*v[766:767]*/, v[210:211] /*v[722:723]*/, v[190:191] /*v[702:703]*/
	s_set_vgpr_msb 0xaac2
	v_cndmask_b32_e64 v112 /*v880*/, v251 /*v763*/, 0xff61b1e6, s2
	s_and_b32 s2, s90, s5
	s_set_vgpr_msb 0xc203
	v_cmp_gt_i32_e64 s42, v104 /*v872*/, v1
	s_set_vgpr_msb 0x30b
	v_cmp_gt_i32_e64 s50, v104 /*v872*/, v203 /*v715*/
	v_cmp_gt_i32_e64 s20, v104 /*v872*/, v205 /*v717*/
	v_cmp_gt_i32_e64 s9, v104 /*v872*/, v207 /*v719*/
	s_set_vgpr_msb 0xbc2
	v_cndmask_b32_e64 v104 /*v872*/, v253 /*v765*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s35
	s_set_vgpr_msb 0xc282
	v_cndmask_b32_e64 v250 /*v762*/, v250 /*v762*/, 0xff61b1e6, s2
	s_and_b32 s2, s90, s8
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s43, v105 /*v873*/, v202 /*v714*/
	v_cmp_gt_i32_e64 s51, v105 /*v873*/, v204 /*v716*/
	v_cmp_gt_i32_e64 s19, v105 /*v873*/, v206 /*v718*/
	v_cmp_gt_i32_e64 s8, v105 /*v873*/, v208 /*v720*/
	s_set_vgpr_msb 0xbc2
	v_cndmask_b32_e64 v105 /*v873*/, v252 /*v764*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s36
	s_set_vgpr_msb 0xc2eb
	v_pk_fma_f32 v[0:1] /*v[768:769]*/, v[0:1] /*v[768:769]*/, v[210:211] /*v[722:723]*/, v[190:191] /*v[702:703]*/
	v_or_b32_e32 v106 /*v874*/, 19, v217 /*v729*/
	s_set_vgpr_msb 0xeb82
	v_cndmask_b32_e64 v255 /*v767*/, v255 /*v767*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s37
	s_set_vgpr_msb 0x82c8
	v_or_b32_e32 v107 /*v875*/, 18, v217 /*v729*/
	s_set_vgpr_msb 0xc882
	v_cndmask_b32_e64 v254 /*v766*/, v254 /*v766*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s38
	s_set_vgpr_msb 0x82eb
	v_pk_fma_f32 v[26:27] /*v[794:795]*/, v[26:27] /*v[794:795]*/, v[210:211] /*v[722:723]*/, v[186:187] /*v[698:699]*/
	v_or_b32_e32 v108 /*v876*/, 21, v217 /*v729*/
	v_cndmask_b32_e64 v1 /*v769*/, v1 /*v769*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s39
	s_set_vgpr_msb 0xeb03
	v_cmp_gt_i32_e64 s44, v106 /*v874*/, v1
	s_set_vgpr_msb 0x3eb
	v_or_b32_e32 v109 /*v877*/, 20, v217 /*v729*/
	v_cndmask_b32_e64 v0 /*v768*/, v0 /*v768*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s42
	v_pk_fma_f32 v[28:29] /*v[796:797]*/, v[28:29] /*v[796:797]*/, v[210:211] /*v[722:723]*/, v[186:187] /*v[698:699]*/
	v_cmp_gt_i32_e64 s45, v107 /*v875*/, v202 /*v714*/
	v_or_b32_e32 v110 /*v878*/, 23, v217 /*v729*/
	s_set_vgpr_msb 0xeb83
	v_cndmask_b32_e64 v237 /*v749*/, v27 /*v795*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s43
	v_cmp_gt_i32_e64 s46, v108 /*v876*/, v1
	s_set_vgpr_msb 0x83c8
	v_or_b32_e32 v111 /*v879*/, 22, v217 /*v729*/
	s_set_vgpr_msb 0xc883
	v_cndmask_b32_e64 v238 /*v750*/, v26 /*v794*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s44
	s_set_vgpr_msb 0x83eb
	v_pk_fma_f32 v[30:31] /*v[798:799]*/, v[30:31] /*v[798:799]*/, v[210:211] /*v[722:723]*/, v[186:187] /*v[698:699]*/
	v_cmp_gt_i32_e64 s47, v109 /*v877*/, v202 /*v714*/
	s_set_vgpr_msb 0xeb83
	v_cndmask_b32_e64 v241 /*v753*/, v29 /*v797*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s45
	v_cmp_gt_i32_e64 s48, v110 /*v878*/, v1
	v_cndmask_b32_e64 v240 /*v752*/, v28 /*v796*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s46
	s_set_vgpr_msb 0x83eb
	v_pk_fma_f32 v[32:33] /*v[800:801]*/, v[32:33] /*v[800:801]*/, v[210:211] /*v[722:723]*/, v[186:187] /*v[698:699]*/
	v_cmp_gt_i32_e64 s49, v111 /*v879*/, v202 /*v714*/
	s_set_vgpr_msb 0xeb82
	v_exp_f32_e32 v218 /*v730*/, v218 /*v730*/
	v_exp_f32_e32 v219 /*v731*/, v219 /*v731*/
	v_exp_f32_e32 v220 /*v732*/, v220 /*v732*/
	v_exp_f32_e32 v221 /*v733*/, v221 /*v733*/
	v_exp_f32_e32 v222 /*v734*/, v222 /*v734*/
	v_exp_f32_e32 v223 /*v735*/, v223 /*v735*/
	v_exp_f32_e32 v238 /*v750*/, v238 /*v750*/
	v_exp_f32_e32 v239 /*v751*/, v237 /*v749*/
	s_set_vgpr_msb 0x8283
	v_cndmask_b32_e64 v251 /*v763*/, v31 /*v799*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s47
	s_set_vgpr_msb 0x83aa
	v_pk_fma_f32 v[226:227] /*v[738:739]*/, v[226:227] /*v[738:739]*/, v[212:213] /*v[724:725]*/, v[194:195] /*v[706:707]*/
	v_pk_fma_f32 v[228:229] /*v[740:741]*/, v[228:229] /*v[740:741]*/, v[212:213] /*v[724:725]*/, v[194:195] /*v[706:707]*/
	v_pk_fma_f32 v[230:231] /*v[742:743]*/, v[230:231] /*v[742:743]*/, v[212:213] /*v[724:725]*/, v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0xaaeb
	v_pk_fma_f32 v[34:35] /*v[802:803]*/, v[34:35] /*v[802:803]*/, v[212:213] /*v[724:725]*/, v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0xeb82
	v_exp_f32_e32 v236 /*v748*/, v250 /*v762*/
	v_nop
	s_set_vgpr_msb 0x8283
	v_cndmask_b32_e64 v250 /*v762*/, v30 /*v798*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s48
	s_set_vgpr_msb 0x83eb
	v_pk_fma_f32 v[12:13] /*v[780:781]*/, v[12:13] /*v[780:781]*/, v[210:211] /*v[722:723]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0xeb83
	v_cndmask_b32_e64 v253 /*v765*/, v33 /*v801*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s49
	s_set_vgpr_msb 0x83eb
	v_pk_fma_f32 v[42:43] /*v[810:811]*/, v[42:43] /*v[810:811]*/, v[210:211] /*v[722:723]*/, v[188:189] /*v[700:701]*/
	s_set_vgpr_msb 0xeb83
	v_cndmask_b32_e64 v252 /*v764*/, v32 /*v800*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s40
	s_set_vgpr_msb 0x838a
	v_pk_mul_f32 v[218:219] /*v[730:731]*/, v[226:227] /*v[738:739]*/, v[218:219] /*v[730:731]*/
	v_pk_mul_f32 v[220:221] /*v[732:733]*/, v[228:229] /*v[740:741]*/, v[220:221] /*v[732:733]*/
	v_pk_mul_f32 v[222:223] /*v[734:735]*/, v[230:231] /*v[742:743]*/, v[222:223] /*v[734:735]*/
	s_set_vgpr_msb 0x8a8b
	v_pk_mul_f32 v[226:227] /*v[738:739]*/, v[34:35] /*v[802:803]*/, v[238:239] /*v[750:751]*/
	s_set_vgpr_msb 0x8b82
	v_exp_f32_e32 v240 /*v752*/, v240 /*v752*/
	v_exp_f32_e32 v241 /*v753*/, v241 /*v753*/
	s_set_vgpr_msb 0x82cb
	v_cndmask_b32_e64 v13 /*v781*/, v13 /*v781*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s41
	v_cmp_gt_i32_e64 s52, v106 /*v874*/, v203 /*v715*/
	s_set_vgpr_msb 0xcb82
	v_exp_f32_e32 v224 /*v736*/, v224 /*v736*/
	v_exp_f32_e32 v225 /*v737*/, v225 /*v737*/
	v_exp_f32_e32 v250 /*v762*/, v250 /*v762*/
	v_exp_f32_e32 v251 /*v763*/, v251 /*v763*/
	s_set_vgpr_msb 0x82eb
	v_cndmask_b32_e64 v12 /*v780*/, v12 /*v780*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s50
	v_pk_fma_f32 v[36:37] /*v[804:805]*/, v[36:37] /*v[804:805]*/, v[212:213] /*v[724:725]*/, v[194:195] /*v[706:707]*/
	v_pk_fma_f32 v[44:45] /*v[812:813]*/, v[44:45] /*v[812:813]*/, v[210:211] /*v[722:723]*/, v[188:189] /*v[700:701]*/
	v_cmp_gt_i32_e64 s53, v107 /*v875*/, v204 /*v716*/
	s_set_vgpr_msb 0xeb8a
	v_cvt_pk_bf16_f32 v218 /*v730*/, v218 /*v730*/, v219 /*v731*/
	v_cvt_pk_bf16_f32 v219 /*v731*/, v220 /*v732*/, v221 /*v733*/
	v_cvt_pk_bf16_f32 v220 /*v732*/, v222 /*v734*/, v223 /*v735*/
	v_cvt_pk_bf16_f32 v222 /*v734*/, v226 /*v738*/, v227 /*v739*/
	s_set_vgpr_msb 0x8a83
	v_cndmask_b32_e64 v226 /*v738*/, v43 /*v811*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s51
	s_set_vgpr_msb 0x83aa
	v_pk_fma_f32 v[232:233] /*v[744:745]*/, v[232:233] /*v[744:745]*/, v[212:213] /*v[724:725]*/, v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0xaaeb
	v_pk_fma_f32 v[38:39] /*v[806:807]*/, v[38:39] /*v[806:807]*/, v[212:213] /*v[724:725]*/, v[194:195] /*v[706:707]*/
	v_cmp_gt_i32_e64 s54, v108 /*v876*/, v203 /*v715*/
	s_set_vgpr_msb 0xeb82
	v_exp_f32_e32 v252 /*v764*/, v252 /*v764*/
	v_exp_f32_e32 v253 /*v765*/, v253 /*v765*/
	s_set_vgpr_msb 0x8283
	v_cndmask_b32_e64 v227 /*v739*/, v42 /*v810*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s52
	s_set_vgpr_msb 0x83eb
	v_pk_fma_f32 v[46:47] /*v[814:815]*/, v[46:47] /*v[814:815]*/, v[210:211] /*v[722:723]*/, v[188:189] /*v[700:701]*/
	v_cmp_gt_i32_e64 s55, v109 /*v877*/, v204 /*v716*/
	s_set_vgpr_msb 0xeb8b
	v_pk_mul_f32 v[228:229] /*v[740:741]*/, v[36:37] /*v[804:805]*/, v[240:241] /*v[752:753]*/
	v_cndmask_b32_e64 v241 /*v753*/, v45 /*v813*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s53
	s_set_vgpr_msb 0x8beb
	v_pk_fma_f32 v[40:41] /*v[808:809]*/, v[40:41] /*v[808:809]*/, v[212:213] /*v[724:725]*/, v[194:195] /*v[706:707]*/
	v_cmp_gt_i32_e64 s56, v110 /*v878*/, v203 /*v715*/
	s_set_vgpr_msb 0xeb8a
	v_pk_mul_f32 v[224:225] /*v[736:737]*/, v[232:233] /*v[744:745]*/, v[224:225] /*v[736:737]*/
	s_set_vgpr_msb 0x8a8b
	v_pk_mul_f32 v[230:231] /*v[742:743]*/, v[38:39] /*v[806:807]*/, v[250:251] /*v[762:763]*/
	v_cndmask_b32_e64 v240 /*v752*/, v44 /*v812*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s54
	s_set_vgpr_msb 0x8beb
	v_pk_fma_f32 v[48:49] /*v[816:817]*/, v[48:49] /*v[816:817]*/, v[210:211] /*v[722:723]*/, v[188:189] /*v[700:701]*/
	v_cmp_gt_i32_e64 s57, v111 /*v879*/, v204 /*v716*/
	s_set_vgpr_msb 0xeb8b
	v_cndmask_b32_e64 v251 /*v763*/, v47 /*v815*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s55
	v_pk_mul_f32 v[232:233] /*v[744:745]*/, v[40:41] /*v[808:809]*/, v[252:253] /*v[764:765]*/
	s_set_vgpr_msb 0x8b8a
	v_cvt_pk_bf16_f32 v223 /*v735*/, v228 /*v740*/, v229 /*v741*/
	s_set_vgpr_msb 0x8a83
	v_cndmask_b32_e64 v250 /*v762*/, v46 /*v814*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s56
	v_exp_f32_e32 v228 /*v740*/, v99 /*v867*/
	v_exp_f32_e32 v229 /*v741*/, v98 /*v866*/
	s_set_vgpr_msb 0x838a
	v_exp_f32_e32 v234 /*v746*/, v234 /*v746*/
	v_exp_f32_e32 v235 /*v747*/, v235 /*v747*/
	v_cvt_pk_bf16_f32 v221 /*v733*/, v224 /*v736*/, v225 /*v737*/
	v_cvt_pk_bf16_f32 v224 /*v736*/, v230 /*v742*/, v231 /*v743*/
	s_set_vgpr_msb 0x8a83
	v_exp_f32_e32 v230 /*v742*/, v101 /*v869*/
	v_exp_f32_e32 v231 /*v743*/, v100 /*v868*/
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v238 /*v750*/, v227 /*v739*/
	v_exp_f32_e32 v239 /*v751*/, v226 /*v738*/
	v_exp_f32_e32 v240 /*v752*/, v240 /*v752*/
	v_exp_f32_e32 v241 /*v753*/, v241 /*v753*/
	s_set_vgpr_msb 0x8283
	v_cndmask_b32_e64 v253 /*v765*/, v49 /*v817*/, 0xff61b1e6, s21
	s_and_b32 s21, s90, s57
	s_set_vgpr_msb 0x83aa
	v_pk_fma_f32 v[244:245] /*v[756:757]*/, v[244:245] /*v[756:757]*/, v[212:213] /*v[724:725]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0xaaeb
	v_pk_fma_f32 v[14:15] /*v[782:783]*/, v[14:15] /*v[782:783]*/, v[210:211] /*v[722:723]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0xebaa
	v_pk_fma_f32 v[242:243] /*v[754:755]*/, v[242:243] /*v[754:755]*/, v[212:213] /*v[724:725]*/, v[196:197] /*v[708:709]*/
	v_pk_fma_f32 v[246:247] /*v[758:759]*/, v[246:247] /*v[758:759]*/, v[212:213] /*v[724:725]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0xaaeb
	v_pk_fma_f32 v[50:51] /*v[818:819]*/, v[50:51] /*v[818:819]*/, v[212:213] /*v[724:725]*/, v[196:197] /*v[708:709]*/
	v_pk_fma_f32 v[52:53] /*v[820:821]*/, v[52:53] /*v[820:821]*/, v[212:213] /*v[724:725]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0xeb83
	v_cndmask_b32_e64 v252 /*v764*/, v48 /*v816*/, 0xff61b1e6, s21
	s_set_vgpr_msb 0x83eb
	v_pk_fma_f32 v[58:59] /*v[826:827]*/, v[58:59] /*v[826:827]*/, v[210:211] /*v[722:723]*/, v[190:191] /*v[702:703]*/
	s_set_vgpr_msb 0xeb8a
	v_cvt_pk_bf16_f32 v225 /*v737*/, v232 /*v744*/, v233 /*v745*/
	s_set_vgpr_msb 0x8a83
	v_exp_f32_e32 v232 /*v744*/, v103 /*v871*/
	v_exp_f32_e32 v233 /*v745*/, v102 /*v870*/
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v250 /*v762*/, v250 /*v762*/
	v_exp_f32_e32 v251 /*v763*/, v251 /*v763*/
	s_and_b32 s11, s90, s11
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s18, v106 /*v874*/, v205 /*v717*/
	s_set_vgpr_msb 0xbaa
	v_pk_fma_f32 v[248:249] /*v[760:761]*/, v[248:249] /*v[760:761]*/, v[212:213] /*v[724:725]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0xaaeb
	v_pk_fma_f32 v[54:55] /*v[822:823]*/, v[54:55] /*v[822:823]*/, v[212:213] /*v[724:725]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0xeb8a
	v_pk_mul_f32 v[228:229] /*v[740:741]*/, v[244:245] /*v[756:757]*/, v[228:229] /*v[740:741]*/
	s_set_vgpr_msb 0x8a83
	v_cndmask_b32_e64 v245 /*v757*/, v14 /*v782*/, 0xff61b1e6, s11
	s_and_b32 s11, s90, s20
	s_set_vgpr_msb 0x83eb
	v_pk_fma_f32 v[60:61] /*v[828:829]*/, v[60:61] /*v[828:829]*/, v[210:211] /*v[722:723]*/, v[190:191] /*v[702:703]*/
	v_cmp_gt_i32_e64 s17, v107 /*v875*/, v206 /*v718*/
	s_set_vgpr_msb 0xeb8a
	v_exp_f32_e32 v252 /*v764*/, v252 /*v764*/
	v_exp_f32_e32 v253 /*v765*/, v253 /*v765*/
	v_pk_mul_f32 v[226:227] /*v[738:739]*/, v[242:243] /*v[754:755]*/, v[234:235] /*v[746:747]*/
	v_pk_mul_f32 v[230:231] /*v[742:743]*/, v[246:247] /*v[758:759]*/, v[230:231] /*v[742:743]*/
	s_set_vgpr_msb 0x8a8b
	v_pk_mul_f32 v[234:235] /*v[746:747]*/, v[50:51] /*v[818:819]*/, v[238:239] /*v[750:751]*/
	v_pk_mul_f32 v[238:239] /*v[750:751]*/, v[52:53] /*v[820:821]*/, v[240:241] /*v[752:753]*/
	s_set_vgpr_msb 0x8b5a
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[218:225] /*v[730:737]*/, v[138:145] /*v[650:657]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x5aeb
	v_cmp_gt_i32_e64 s16, v108 /*v876*/, v205 /*v717*/
	v_pk_fma_f32 v[56:57] /*v[824:825]*/, v[56:57] /*v[824:825]*/, v[212:213] /*v[724:725]*/, v[196:197] /*v[708:709]*/
	v_pk_fma_f32 v[62:63] /*v[830:831]*/, v[62:63] /*v[830:831]*/, v[210:211] /*v[722:723]*/, v[190:191] /*v[702:703]*/
	v_cmp_gt_i32_e64 s15, v109 /*v877*/, v206 /*v718*/
	s_set_vgpr_msb 0xeb8a
	v_pk_mul_f32 v[232:233] /*v[744:745]*/, v[248:249] /*v[760:761]*/, v[232:233] /*v[744:745]*/
	s_set_vgpr_msb 0x8a8b
	v_pk_mul_f32 v[240:241] /*v[752:753]*/, v[54:55] /*v[822:823]*/, v[250:251] /*v[762:763]*/
	s_set_vgpr_msb 0x8b8a
	v_cvt_pk_bf16_f32 v226 /*v738*/, v226 /*v738*/, v227 /*v739*/
	s_set_vgpr_msb 0x8a5a
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[218:225] /*v[730:737]*/, v[130:137] /*v[642:649]*/, v[242:249] /*v[498:505]*/
	s_set_vgpr_msb 0x5a8a
	v_cvt_pk_bf16_f32 v227 /*v739*/, v228 /*v740*/, v229 /*v741*/
	v_cvt_pk_bf16_f32 v228 /*v740*/, v230 /*v742*/, v231 /*v743*/
	v_cvt_pk_bf16_f32 v231 /*v743*/, v238 /*v750*/, v239 /*v751*/
	s_set_vgpr_msb 0x8aeb
	v_cmp_gt_i32_e64 s14, v110 /*v878*/, v205 /*v717*/
	v_pk_fma_f32 v[64:65] /*v[832:833]*/, v[64:65] /*v[832:833]*/, v[210:211] /*v[722:723]*/, v[190:191] /*v[702:703]*/
	v_cmp_gt_i32_e64 s13, v111 /*v879*/, v206 /*v718*/
	s_set_vgpr_msb 0xeb8b
	v_pk_mul_f32 v[242:243] /*v[754:755]*/, v[56:57] /*v[824:825]*/, v[252:253] /*v[764:765]*/
	s_set_vgpr_msb 0x8b5a
	s_wait_dscnt 0x29
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[218:225] /*v[730:737]*/, v[146:153] /*v[658:665]*/, v[234:241] /*v[490:497]*/
	s_set_vgpr_msb 0x5a8a
	v_cvt_pk_bf16_f32 v229 /*v741*/, v232 /*v744*/, v233 /*v745*/
	v_cvt_pk_bf16_f32 v232 /*v744*/, v240 /*v752*/, v241 /*v753*/
	s_set_vgpr_msb 0x8aeb
	v_pk_fma_f32 v[10:11] /*v[778:779]*/, v[10:11] /*v[778:779]*/, v[210:211] /*v[722:723]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0xeb8a
	v_cvt_pk_bf16_f32 v233 /*v745*/, v242 /*v754*/, v243 /*v755*/
	v_cmp_gt_i32_e64 s7, v217 /*v729*/, v208 /*v720*/
	s_set_vgpr_msb 0x8a83
	v_exp_f32_e32 v237 /*v749*/, v112 /*v880*/
	s_set_vgpr_msb 0x838a
	v_cvt_pk_bf16_f32 v230 /*v742*/, v234 /*v746*/, v235 /*v747*/
	s_set_vgpr_msb 0x8a5a
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[218:225] /*v[730:737]*/, v[154:161] /*v[666:673]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5aeb
	v_pk_fma_f32 v[2:3] /*v[770:771]*/, v[2:3] /*v[770:771]*/, v[212:213] /*v[724:725]*/, v[198:199] /*v[710:711]*/
	v_pk_fma_f32 v[4:5] /*v[772:773]*/, v[4:5] /*v[772:773]*/, v[212:213] /*v[724:725]*/, v[198:199] /*v[710:711]*/
	v_pk_fma_f32 v[6:7] /*v[774:775]*/, v[6:7] /*v[774:775]*/, v[212:213] /*v[724:725]*/, v[198:199] /*v[710:711]*/
	v_pk_fma_f32 v[8:9] /*v[776:777]*/, v[8:9] /*v[776:777]*/, v[212:213] /*v[724:725]*/, v[198:199] /*v[710:711]*/
	v_pk_fma_f32 v[66:67] /*v[834:835]*/, v[66:67] /*v[834:835]*/, v[212:213] /*v[724:725]*/, v[198:199] /*v[710:711]*/
	v_pk_fma_f32 v[68:69] /*v[836:837]*/, v[68:69] /*v[836:837]*/, v[212:213] /*v[724:725]*/, v[198:199] /*v[710:711]*/
	v_pk_fma_f32 v[70:71] /*v[838:839]*/, v[70:71] /*v[838:839]*/, v[212:213] /*v[724:725]*/, v[198:199] /*v[710:711]*/
	s_set_vgpr_msb 0xeb5a
	s_wait_dscnt 0x25
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[218:225] /*v[730:737]*/, v[162:169] /*v[674:681]*/, v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5aeb
	v_pk_fma_f32 v[72:73] /*v[840:841]*/, v[72:73] /*v[840:841]*/, v[212:213] /*v[724:725]*/, v[198:199] /*v[710:711]*/
	v_cndmask_b32_e64 v11 /*v779*/, v11 /*v779*/, 0xff61b1e6, s2
	s_and_b32 s2, s90, s7
	v_pk_fma_f32 v[16:17] /*v[784:785]*/, v[16:17] /*v[784:785]*/, v[210:211] /*v[722:723]*/, v[192:193] /*v[704:705]*/
	v_cndmask_b32_e64 v10 /*v778*/, v10 /*v778*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s7, v106 /*v874*/, v207 /*v719*/
	v_cmp_gt_i32_e64 s6, v107 /*v875*/, v208 /*v720*/
	s_set_vgpr_msb 0xeb5a
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[218:225] /*v[730:737]*/, v[170:177] /*v[682:689]*/, v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5aeb
	v_cmp_gt_i32_e64 s5, v108 /*v876*/, v207 /*v719*/
	v_cmp_gt_i32_e64 s4, v109 /*v877*/, v208 /*v720*/
	v_cmp_gt_i32_e64 s3, v110 /*v878*/, v207 /*v719*/
	v_cmp_gt_i32_e64 s2, v111 /*v879*/, v208 /*v720*/
	v_pk_fma_f32 v[74:75] /*v[842:843]*/, v[74:75] /*v[842:843]*/, v[210:211] /*v[722:723]*/, v[192:193] /*v[704:705]*/
	v_pk_fma_f32 v[76:77] /*v[844:845]*/, v[76:77] /*v[844:845]*/, v[210:211] /*v[722:723]*/, v[192:193] /*v[704:705]*/
	v_pk_fma_f32 v[78:79] /*v[846:847]*/, v[78:79] /*v[846:847]*/, v[210:211] /*v[722:723]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0xeb5a
	s_wait_dscnt 0x21
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[218:225] /*v[730:737]*/, v[178:185] /*v[690:697]*/, v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5aeb
	v_pk_fma_f32 v[80:81] /*v[848:849]*/, v[80:81] /*v[848:849]*/, v[210:211] /*v[722:723]*/, v[192:193] /*v[704:705]*/
	s_and_b32 s10, s90, s10
	s_and_b32 s12, s90, s12
	s_and_b32 s9, s90, s9
	s_and_b32 s8, s90, s8
	s_and_b32 s7, s90, s7
	s_and_b32 s6, s90, s6
	s_set_vgpr_msb 0xeb5e
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[218:225] /*v[730:737]*/, v[90:97] /*v[858:865]*/, v[42:49] /*v[298:305]*/
	s_and_b32 s5, s90, s5
	s_and_b32 s4, s90, s4
	s_and_b32 s3, s90, s3
	s_and_b32 s2, s90, s2
	s_set_vgpr_msb 0x5e83
	v_cndmask_b32_e64 v244 /*v756*/, v15 /*v783*/, 0xff61b1e6, s12
	s_set_vgpr_msb 0x83eb
	v_pk_fma_f32 v[18:19] /*v[786:787]*/, v[18:19] /*v[786:787]*/, v[212:213] /*v[724:725]*/, v[200:201] /*v[712:713]*/
	v_pk_fma_f32 v[20:21] /*v[788:789]*/, v[20:21] /*v[788:789]*/, v[212:213] /*v[724:725]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0xeb0a
	v_wmma_f32_16x16x32_bf16 v[186:193], v[226:233] /*v[738:745]*/, v[138:145] /*v[650:657]*/, v[186:193]
	s_set_vgpr_msb 0xa83
	v_cndmask_b32_e64 v218 /*v730*/, v59 /*v827*/, 0xff61b1e6, s11
	s_and_b32 s11, s90, s19
	v_exp_f32_e32 v220 /*v732*/, v105 /*v873*/
	v_cndmask_b32_e64 v219 /*v731*/, v58 /*v826*/, 0xff61b1e6, s11
	s_and_b32 s11, s90, s18
	v_exp_f32_e32 v221 /*v733*/, v104 /*v872*/
	v_cndmask_b32_e64 v239 /*v751*/, v61 /*v829*/, 0xff61b1e6, s11
	s_and_b32 s11, s90, s17
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v222 /*v734*/, v254 /*v766*/
	s_set_vgpr_msb 0x8283
	v_cndmask_b32_e64 v238 /*v750*/, v60 /*v828*/, 0xff61b1e6, s11
	s_and_b32 s11, s90, s16
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v223 /*v735*/, v255 /*v767*/
	s_set_vgpr_msb 0x8283
	v_cndmask_b32_e64 v241 /*v753*/, v63 /*v831*/, 0xff61b1e6, s11
	s_and_b32 s11, s90, s15
	v_exp_f32_e32 v224 /*v736*/, v0 /*v768*/
	v_cndmask_b32_e64 v240 /*v752*/, v62 /*v830*/, 0xff61b1e6, s11
	s_and_b32 s11, s90, s14
	v_exp_f32_e32 v225 /*v737*/, v1 /*v769*/
	v_cndmask_b32_e64 v243 /*v755*/, v65 /*v833*/, 0xff61b1e6, s11
	s_and_b32 s11, s90, s13
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v234 /*v746*/, v219 /*v731*/
	s_set_vgpr_msb 0x8283
	v_cndmask_b32_e64 v242 /*v754*/, v64 /*v832*/, 0xff61b1e6, s11
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v235 /*v747*/, v218 /*v730*/
	v_exp_f32_e32 v238 /*v750*/, v238 /*v750*/
	v_exp_f32_e32 v239 /*v751*/, v239 /*v751*/
	v_exp_f32_e32 v240 /*v752*/, v240 /*v752*/
	v_exp_f32_e32 v241 /*v753*/, v241 /*v753*/
	v_exp_f32_e32 v242 /*v754*/, v242 /*v754*/
	v_exp_f32_e32 v243 /*v755*/, v243 /*v755*/
	s_set_vgpr_msb 0x828b
	v_pk_mul_f32 v[218:219] /*v[730:731]*/, v[2:3] /*v[770:771]*/, v[236:237] /*v[748:749]*/
	v_pk_mul_f32 v[220:221] /*v[732:733]*/, v[4:5] /*v[772:773]*/, v[220:221] /*v[732:733]*/
	v_pk_mul_f32 v[222:223] /*v[734:735]*/, v[6:7] /*v[774:775]*/, v[222:223] /*v[734:735]*/
	v_pk_mul_f32 v[224:225] /*v[736:737]*/, v[8:9] /*v[776:777]*/, v[224:225] /*v[736:737]*/
	v_pk_mul_f32 v[234:235] /*v[746:747]*/, v[66:67] /*v[834:835]*/, v[234:235] /*v[746:747]*/
	v_pk_mul_f32 v[236:237] /*v[748:749]*/, v[68:69] /*v[836:837]*/, v[238:239] /*v[750:751]*/
	v_pk_mul_f32 v[238:239] /*v[750:751]*/, v[70:71] /*v[838:839]*/, v[240:241] /*v[752:753]*/
	v_pk_mul_f32 v[240:241] /*v[752:753]*/, v[72:73] /*v[840:841]*/, v[242:243] /*v[754:755]*/
	s_set_vgpr_msb 0x8b0a
	v_wmma_f32_16x16x32_bf16 v[178:185], v[226:233] /*v[738:745]*/, v[130:137] /*v[642:649]*/, v[178:185]
	s_set_vgpr_msb 0xa8a
	v_cvt_pk_bf16_f32 v218 /*v730*/, v218 /*v730*/, v219 /*v731*/
	v_cvt_pk_bf16_f32 v219 /*v731*/, v220 /*v732*/, v221 /*v733*/
	v_cvt_pk_bf16_f32 v220 /*v732*/, v222 /*v734*/, v223 /*v735*/
	v_cvt_pk_bf16_f32 v221 /*v733*/, v224 /*v736*/, v225 /*v737*/
	v_cvt_pk_bf16_f32 v222 /*v734*/, v234 /*v746*/, v235 /*v747*/
	v_cvt_pk_bf16_f32 v223 /*v735*/, v236 /*v748*/, v237 /*v749*/
	v_cvt_pk_bf16_f32 v224 /*v736*/, v238 /*v750*/, v239 /*v751*/
	s_set_vgpr_msb 0x8a0a
	v_wmma_f32_16x16x32_bf16 v[170:177], v[226:233] /*v[738:745]*/, v[146:153] /*v[658:665]*/, v[170:177]
	s_set_vgpr_msb 0xa8a
	v_cvt_pk_bf16_f32 v225 /*v737*/, v240 /*v752*/, v241 /*v753*/
	s_set_vgpr_msb 0x8a83
	v_cndmask_b32_e64 v235 /*v747*/, v75 /*v843*/, 0xff61b1e6, s9
	v_cndmask_b32_e64 v234 /*v746*/, v74 /*v842*/, 0xff61b1e6, s8
	v_cndmask_b32_e64 v237 /*v749*/, v77 /*v845*/, 0xff61b1e6, s7
	v_cndmask_b32_e64 v236 /*v748*/, v76 /*v844*/, 0xff61b1e6, s6
	v_cndmask_b32_e64 v239 /*v751*/, v79 /*v847*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v238 /*v750*/, v78 /*v846*/, 0xff61b1e6, s4
	s_set_vgpr_msb 0x830a
	v_wmma_f32_16x16x32_bf16 v[162:169], v[226:233] /*v[738:745]*/, v[154:161] /*v[666:673]*/, v[162:169]
	s_set_vgpr_msb 0xa83
	v_cndmask_b32_e64 v241 /*v753*/, v81 /*v849*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v240 /*v752*/, v80 /*v848*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v234 /*v746*/, v234 /*v746*/
	v_exp_f32_e32 v235 /*v747*/, v235 /*v747*/
	v_exp_f32_e32 v236 /*v748*/, v236 /*v748*/
	v_exp_f32_e32 v237 /*v749*/, v237 /*v749*/
	v_exp_f32_e32 v238 /*v750*/, v238 /*v750*/
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[154:161], v[226:233] /*v[738:745]*/, v[162:169] /*v[674:681]*/, v[154:161]
	s_set_vgpr_msb 0xa82
	v_exp_f32_e32 v239 /*v751*/, v239 /*v751*/
	v_exp_f32_e32 v240 /*v752*/, v240 /*v752*/
	v_exp_f32_e32 v241 /*v753*/, v241 /*v753*/
	s_set_vgpr_msb 0x82eb
	v_pk_fma_f32 v[22:23] /*v[790:791]*/, v[22:23] /*v[790:791]*/, v[212:213] /*v[724:725]*/, v[200:201] /*v[712:713]*/
	v_pk_fma_f32 v[24:25] /*v[792:793]*/, v[24:25] /*v[792:793]*/, v[212:213] /*v[724:725]*/, v[200:201] /*v[712:713]*/
	v_pk_fma_f32 v[82:83] /*v[850:851]*/, v[82:83] /*v[850:851]*/, v[212:213] /*v[724:725]*/, v[200:201] /*v[712:713]*/
	v_pk_fma_f32 v[84:85] /*v[852:853]*/, v[84:85] /*v[852:853]*/, v[212:213] /*v[724:725]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0xeb0a
	v_wmma_f32_16x16x32_bf16 v[146:153], v[226:233] /*v[738:745]*/, v[170:177] /*v[682:689]*/, v[146:153]
	s_set_vgpr_msb 0xaeb
	v_pk_fma_f32 v[86:87] /*v[854:855]*/, v[86:87] /*v[854:855]*/, v[212:213] /*v[724:725]*/, v[200:201] /*v[712:713]*/
	v_pk_fma_f32 v[88:89] /*v[856:857]*/, v[88:89] /*v[856:857]*/, v[212:213] /*v[724:725]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0xeb8b
	v_pk_mul_f32 v[234:235] /*v[746:747]*/, v[82:83] /*v[850:851]*/, v[234:235] /*v[746:747]*/
	v_pk_mul_f32 v[236:237] /*v[748:749]*/, v[84:85] /*v[852:853]*/, v[236:237] /*v[748:749]*/
	v_add_nc_u32_e32 v217 /*v729*/, 32, v217 /*v729*/
	v_pk_mul_f32 v[238:239] /*v[750:751]*/, v[86:87] /*v[854:855]*/, v[238:239] /*v[750:751]*/
	v_pk_mul_f32 v[240:241] /*v[752:753]*/, v[88:89] /*v[856:857]*/, v[240:241] /*v[752:753]*/
	s_set_vgpr_msb 0x8b0a
	v_wmma_f32_16x16x32_bf16 v[138:145], v[226:233] /*v[738:745]*/, v[178:185] /*v[690:697]*/, v[138:145]
	s_add_nc_u64 s[84:85], s[84:85], -1
	s_add_co_i32 s63, s63, 1
	s_mov_b32 s3, s92
	s_set_vgpr_msb 0xa0e
	v_wmma_f32_16x16x32_bf16 v[130:137], v[226:233] /*v[738:745]*/, v[90:97] /*v[858:865]*/, v[130:137]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xe83
	v_cndmask_b32_e64 v233 /*v745*/, v17 /*v785*/, 0xff61b1e6, s10
	s_and_b32 s10, s90, vcc_lo
	v_exp_f32_e32 v226 /*v738*/, v10 /*v778*/
	v_cndmask_b32_e64 v232 /*v744*/, v16 /*v784*/, 0xff61b1e6, s10
	v_exp_f32_e32 v227 /*v739*/, v11 /*v779*/
	v_exp_f32_e32 v228 /*v740*/, v12 /*v780*/
	v_exp_f32_e32 v229 /*v741*/, v13 /*v781*/
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v230 /*v742*/, v245 /*v757*/
	v_exp_f32_e32 v231 /*v743*/, v244 /*v756*/
	v_exp_f32_e32 v232 /*v744*/, v232 /*v744*/
	v_exp_f32_e32 v233 /*v745*/, v233 /*v745*/
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[122:129], v[218:225] /*v[730:737]*/, v[138:145] /*v[650:657]*/, v[122:129]
	s_set_vgpr_msb 0xa8b
	v_pk_mul_f32 v[226:227] /*v[738:739]*/, v[18:19] /*v[786:787]*/, v[226:227] /*v[738:739]*/
	s_cmp_lg_u64 s[84:85], 0
	v_pk_mul_f32 v[228:229] /*v[740:741]*/, v[20:21] /*v[788:789]*/, v[228:229] /*v[740:741]*/
	v_pk_mul_f32 v[230:231] /*v[742:743]*/, v[22:23] /*v[790:791]*/, v[230:231] /*v[742:743]*/
	s_set_vgpr_msb 0x8b8a
	v_cvt_pk_bf16_f32 v226 /*v738*/, v226 /*v738*/, v227 /*v739*/
	s_set_vgpr_msb 0x8a8b
	v_pk_mul_f32 v[232:233] /*v[744:745]*/, v[24:25] /*v[792:793]*/, v[232:233] /*v[744:745]*/
	s_set_vgpr_msb 0x8b8a
	v_cvt_pk_bf16_f32 v227 /*v739*/, v228 /*v740*/, v229 /*v741*/
	s_set_vgpr_msb 0x8a0a
	v_wmma_f32_16x16x32_bf16 v[114:121], v[218:225] /*v[730:737]*/, v[130:137] /*v[642:649]*/, v[114:121]
	s_set_vgpr_msb 0xa8a
	v_cvt_pk_bf16_f32 v228 /*v740*/, v230 /*v742*/, v231 /*v743*/
	v_cvt_pk_bf16_f32 v230 /*v742*/, v234 /*v746*/, v235 /*v747*/
	v_cvt_pk_bf16_f32 v229 /*v741*/, v232 /*v744*/, v233 /*v745*/
	v_cvt_pk_bf16_f32 v231 /*v743*/, v236 /*v748*/, v237 /*v749*/
	v_cvt_pk_bf16_f32 v232 /*v744*/, v238 /*v750*/, v239 /*v751*/
	v_cvt_pk_bf16_f32 v233 /*v745*/, v240 /*v752*/, v241 /*v753*/
	s_set_vgpr_msb 0x8a0a
	v_wmma_f32_16x16x32_bf16 v[106:113], v[218:225] /*v[730:737]*/, v[146:153] /*v[658:665]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[98:105], v[218:225] /*v[730:737]*/, v[154:161] /*v[666:673]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[218:225] /*v[730:737]*/, v[162:169] /*v[674:681]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[82:89], v[218:225] /*v[730:737]*/, v[170:177] /*v[682:689]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[218:225] /*v[730:737]*/, v[178:185] /*v[690:697]*/, v[74:81]
	s_set_vgpr_msb 0xa0e
	v_wmma_f32_16x16x32_bf16 v[66:73], v[218:225] /*v[730:737]*/, v[90:97] /*v[858:865]*/, v[66:73]
	s_set_vgpr_msb 0xe0a
	v_wmma_f32_16x16x32_bf16 v[58:65], v[226:233] /*v[738:745]*/, v[138:145] /*v[650:657]*/, v[58:65]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[226:233] /*v[738:745]*/, v[130:137] /*v[642:649]*/, v[50:57]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[226:233] /*v[738:745]*/, v[146:153] /*v[658:665]*/, v[42:49]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[226:233] /*v[738:745]*/, v[154:161] /*v[666:673]*/, v[34:41]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[226:233] /*v[738:745]*/, v[162:169] /*v[674:681]*/, v[26:33]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[226:233] /*v[738:745]*/, v[170:177] /*v[682:689]*/, v[18:25]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[226:233] /*v[738:745]*/, v[178:185] /*v[690:697]*/, v[10:17]
	s_set_vgpr_msb 0xa0e
	v_wmma_f32_16x16x32_bf16 v[2:9], v[226:233] /*v[738:745]*/, v[90:97] /*v[858:865]*/, v[2:9]
	s_set_vgpr_msb 0xe00
	s_cbranch_scc1 .LBB0_6
.LBB0_7:
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, s88, v214 /*v726*/
	s_load_b64 s[0:1], s[0:1], 0x140 nv
	s_mul_i32 s4, s59, s91
	s_mov_b32 s3, 0
	s_add_co_i32 s89, s89, s4
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v194, 1, v1
	v_mul_lo_u32 v196, v1, s59
	v_or_b32_e32 v197, 2, v1
	v_or_b32_e32 v200, 3, v1
	v_or_b32_e32 v203, 4, v1
	v_mul_lo_u32 v194, v194, s59
	v_or_b32_e32 v204, 5, v1
	v_mul_lo_u32 v197, v197, s59
	s_mov_b32 s2, 0x800000
	v_add_lshl_u32 v196, s89, v196, 7
	s_wait_tensorcnt 0x0
	v_mul_lo_u32 v200, v200, s59
	v_mul_lo_u32 v203, v203, s59
	v_add_lshl_u32 v194, s89, v194, 7
	s_set_vgpr_msb 8
	v_or_b32_e32 v201, v196, v209 /*v721*/
	v_add_lshl_u32 v197, v197, s89, 7
	s_set_vgpr_msb 0x801
	s_wait_kmcnt 0x0
	v_cvt_pk_bf16_f32 v195, v250 /*v506*/, s0
	v_cvt_pk_bf16_f32 v198, v251 /*v507*/, s0
	s_set_vgpr_msb 0x108
	v_or_b32_e32 v202, v194, v209 /*v721*/
	s_set_vgpr_msb 0x801
	v_cvt_pk_bf16_f32 v199, v252 /*v508*/, s0
	v_lshlrev_b32_e32 v201, 1, v201
	s_set_vgpr_msb 0x108
	v_or_b32_e32 v205, v197, v209 /*v721*/
	v_mul_lo_u32 v204, v204, s59
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v202, 1, v202
	v_add_lshl_u32 v200, s89, v200, 7
	buffer_store_b16 v195, v201, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v195, 1, v205
	v_add_lshl_u32 v203, s89, v203, 7
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v205, v253 /*v509*/, s0
	s_set_vgpr_msb 0x100
	v_or_b32_e32 v196, v196, v0
	v_add_lshl_u32 v204, s89, v204, 7
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v207, v254 /*v510*/, s0
	s_set_vgpr_msb 0x108
	buffer_store_b16 v198, v202, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v198.l, v199.l
	v_or_b32_e32 v206, v203, v209 /*v721*/
	v_or_b32_e32 v208, v204, v209 /*v721*/
	s_set_vgpr_msb 0x801
	v_cvt_pk_bf16_f32 v209, v255 /*v511*/, s0
	v_lshlrev_b32_e32 v196, 1, v196
	s_set_vgpr_msb 0x100
	buffer_store_b16 v198, v195, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v206, 1, v206 :: v_dual_bitop2_b32 v198, 6, v1 bitop3:0x54
	v_dual_lshlrev_b32 v208, 1, v208 :: v_dual_bitop2_b32 v1, 7, v1 bitop3:0x54
	v_or_b32_e32 v194, v194, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_lo_u32 v198, v198, s59
	v_or_b32_e32 v197, v197, v0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v211, v242 /*v498*/, s0
	v_mul_lo_u32 v1, s59, v1
	v_or_b32_e32 v212, 32, v196
	s_set_vgpr_msb 0x108
	v_or_b32_e32 v199, v200, v209 /*v721*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v194, 1, v194 :: v_dual_lshlrev_b32 v197, 1, v197
	v_add_lshl_u32 v198, s89, v198, 7
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_lshlrev_b32 v199, 1, v199 :: v_dual_bitop2_b32 v200, v200, v0 bitop3:0x54
	v_add_lshl_u32 v1, s89, v1, 7
	s_clause 0x1
	buffer_store_b16 v205, v199, s[0:3], null offen
	buffer_store_b16 v207, v206, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v205, v198, v209 /*v721*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v203, v203, v0
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v210, v1 /*v513*/, s0
	v_lshlrev_b32_e32 v200, 1, v200
	s_set_vgpr_msb 0x200
	v_dual_lshlrev_b32 v205, 1, v205 :: v_dual_bitop2_b32 v204, v204, v0 bitop3:0x54
	v_dual_lshlrev_b32 v203, 1, v203 :: v_dual_bitop2_b32 v213, 32, v197 bitop3:0x54
	buffer_store_b16 v209, v208, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v209, v1, v209 /*v721*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v198, v198, v0
	v_or_b32_e32 v214, 32, v200
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v207, v0 /*v512*/, s0
	s_set_vgpr_msb 0x200
	v_dual_lshlrev_b32 v209, 1, v209 :: v_dual_bitop2_b32 v1, v1, v0 bitop3:0x54
	s_clause 0x1
	buffer_store_b16 v207, v205, s[0:3], null offen
	buffer_store_b16 v210, v209, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v207, v243 /*v499*/, s0
	v_lshlrev_b32_e32 v204, 1, v204
	s_set_vgpr_msb 0x100
	buffer_store_b16 v211, v212, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v211, v244 /*v500*/, s0
	v_or_b32_e32 v210, 32, v194
	v_lshlrev_b32_e32 v198, 1, v198
	v_cvt_pk_bf16_f32 v212, v245 /*v501*/, s0
	v_lshlrev_b32_e32 v1, 1, v1
	v_cvt_pk_bf16_f32 v215, v236 /*v492*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v207, v210, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v207, v246 /*v502*/, s0
	v_or_b32_e32 v216, 0x60, v200
	s_set_vgpr_msb 0x100
	buffer_store_b16 v211, v213, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v211, 32, v203
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v210, v247 /*v503*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v212, v214, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v212, 32, v204
	s_clause 0x1
	buffer_store_b16 v207, v211, s[0:3], null offen
	buffer_store_b16 v210, v212, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v207, 32, v198
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v213, v248 /*v504*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v186, v186, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v212, v234 /*v490*/, s0
	v_cvt_pk_bf16_f32 v211, v249 /*v505*/, s0
	v_cvt_pk_bf16_f32 v214, v235 /*v491*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v210.l, v213.l
	v_cvt_pk_bf16_f32 v187, v187, s0
	v_cvt_pk_bf16_f32 v188, v188, s0
	v_or_b32_e32 v213, 32, v1
	v_cvt_pk_bf16_f32 v189, v189, s0
	buffer_store_b16 v210, v207, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v210.l, v215.l
	v_or_b32_e32 v215, 0x60, v197
	v_cvt_pk_bf16_f32 v190, v190, s0
	s_clause 0x1
	buffer_store_b16 v211, v213, s[0:3], null offen
	buffer_store_b16 v212, v201, s[0:3], null offen offset:64
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v211, v237 /*v493*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v207.l, v214.l
	s_clause 0x1
	buffer_store_b16 v207, v202, s[0:3], null offen offset:64
	buffer_store_b16 v210, v195, s[0:3], null offen offset:64
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v207, v238 /*v494*/, s0
	v_cvt_pk_bf16_f32 v212, v240 /*v496*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v211, v199, s[0:3], null offen offset:64
	v_cvt_pk_bf16_f32 v191, v191, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v210, v239 /*v495*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v207, v206, s[0:3], null offen offset:64
	buffer_store_b16 v210, v208, s[0:3], null offen offset:64
	s_wait_xcnt 0x2
	v_mov_b16_e64 v211.l, v212.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v207, v210 /*v466*/, s0
	v_or_b32_e32 v210, 0x60, v196
	v_cvt_pk_bf16_f32 v213, v241 /*v497*/, s0
	v_cvt_pk_bf16_f32 v214, v213 /*v469*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v193, v193, s0
	v_cvt_pk_bf16_f32 v178, v178, s0
	v_cvt_pk_bf16_f32 v179, v179, s0
	v_mov_b16_e64 v212.l, v213.l
	s_clause 0x1
	buffer_store_b16 v211, v205, s[0:3], null offen offset:64
	buffer_store_b16 v212, v209, s[0:3], null offen offset:64
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v211, v211 /*v467*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v180, v180, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v212, v212 /*v468*/, s0
	v_or_b32_e32 v213, 0x60, v194
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v207, v210, s[0:3], null offen
	buffer_store_b16 v211, v213, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v207, v214 /*v470*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v181, v181, s0
	s_clause 0x1
	buffer_store_b16 v212, v215, s[0:3], null offen
	buffer_store_b16 v214, v216, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v212, v216 /*v472*/, s0
	v_or_b32_e32 v213, 0x60, v204
	v_or_b32_e32 v216, 0xa0, v203
	v_or_b32_e32 v214, 0x60, v198
	v_cvt_pk_bf16_f32 v215, v217 /*v473*/, s0
	v_or_b32_e32 v211, 0x60, v203
	v_cvt_pk_bf16_f32 v210, v215 /*v471*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v207, v211, s[0:3], null offen
	buffer_store_b16 v210, v213, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v207, 0x60, v1
	v_cvt_pk_bf16_f32 v184, v184, s0
	buffer_store_b16 v212, v214, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v212, v179 /*v435*/, s0
	v_cvt_pk_bf16_f32 v213, v180 /*v436*/, s0
	v_cvt_pk_bf16_f32 v214, v142 /*v398*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v210.l, v215.l
	v_cvt_pk_bf16_f32 v170, v170, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v211, v178 /*v434*/, s0
	v_or_b32_e32 v215, 0xa0, v200
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v171, v171, s0
	buffer_store_b16 v210, v207, s[0:3], null offen
	v_cvt_pk_bf16_f32 v172, v172, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v210.l, v211.l
	v_mov_b16_e64 v211.l, v212.l
	v_mov_b16_e64 v212.l, v213.l
	s_clause 0x2
	buffer_store_b16 v210, v201, s[0:3], null offen offset:128
	buffer_store_b16 v211, v202, s[0:3], null offen offset:128
	buffer_store_b16 v212, v195, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v211, v184 /*v440*/, s0
	v_cvt_pk_bf16_f32 v207, v181 /*v437*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v162, v162, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v212, v185 /*v441*/, s0
	v_cvt_pk_bf16_f32 v213, v182 /*v438*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v164, v164, s0
	buffer_store_b16 v207, v199, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v207, v183 /*v439*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v163, v163, s0
	v_mov_b16_e64 v210.l, v213.l
	v_cvt_pk_bf16_f32 v165, v165, s0
	v_cvt_pk_bf16_f32 v154, v154, s0
	v_or_b32_e32 v213, 0xa0, v196
	v_or_b32_e32 v196, 0xe0, v196
	buffer_store_b16 v210, v206, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v210, v138 /*v394*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v157, v157, s0
	s_clause 0x1
	buffer_store_b16 v207, v208, s[0:3], null offen offset:128
	buffer_store_b16 v211, v205, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v207, v139 /*v395*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v158, v158, s0
	s_clause 0x1
	buffer_store_b16 v212, v209, s[0:3], null offen offset:128
	buffer_store_b16 v210, v213, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v212, v141 /*v397*/, s0
	v_cvt_pk_bf16_f32 v211, v140 /*v396*/, s0
	v_or_b32_e32 v213, 0xa0, v197
	v_or_b32_e32 v210, 0xa0, v194
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v207, v210, s[0:3], null offen
	buffer_store_b16 v211, v213, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v207, v143 /*v399*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v155, v155, s0
	s_clause 0x1
	buffer_store_b16 v212, v215, s[0:3], null offen
	buffer_store_b16 v214, v216, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v212, v145 /*v401*/, s0
	v_or_b32_e32 v213, 0xa0, v198
	v_or_b32_e32 v198, 0xe0, v198
	v_cvt_pk_bf16_f32 v214, v98 /*v354*/, s0
	v_or_b32_e32 v215, 0xa0, v1
	v_or_b32_e32 v1, 0xe0, v1
	v_cvt_pk_bf16_f32 v211, v144 /*v400*/, s0
	v_or_b32_e32 v210, 0xa0, v204
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v156, v156, s0
	v_or_b32_e32 v194, 0xe0, v194
	v_cvt_pk_bf16_f32 v146, v146, s0
	v_cvt_pk_bf16_f32 v147, v147, s0
	buffer_store_b16 v207, v210, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v207.l, v214.l
	v_cvt_pk_bf16_f32 v149, v149, s0
	buffer_store_b16 v211, v213, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v211, v100 /*v356*/, s0
	v_cvt_pk_bf16_f32 v210, v99 /*v355*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v212, v215, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v212, v103 /*v359*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v148, v148, s0
	buffer_store_b16 v207, v201, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_mov_b16_e64 v207.l, v210.l
	v_cvt_pk_bf16_f32 v150, v150, s0
	v_cvt_pk_bf16_f32 v138, v138, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v210, v102 /*v358*/, s0
	v_cvt_pk_bf16_f32 v201, v101 /*v357*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v207, v202, s[0:3], null offen offset:192
	v_cvt_pk_bf16_f32 v139, v139, s0
	v_cvt_pk_bf16_f32 v140, v140, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v202.l, v210.l
	buffer_store_b16 v211, v195, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_mov_b16_e64 v195.l, v212.l
	v_cvt_pk_bf16_f32 v130, v130, s0
	buffer_store_b16 v201, v199, s[0:3], null offen offset:192
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v201, v105 /*v361*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v131, v131, s0
	buffer_store_b16 v202, v206, s[0:3], null offen offset:192
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v202, v43 /*v299*/, s0
	v_cvt_pk_bf16_f32 v199, v104 /*v360*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v195, v208, s[0:3], null offen offset:192
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v195, v42 /*v298*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v122, v122, s0
	s_clause 0x1
	buffer_store_b16 v199, v205, s[0:3], null offen offset:192
	buffer_store_b16 v201, v209, s[0:3], null offen offset:192
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v44 /*v300*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v123, v123, s0
	s_clause 0x1
	buffer_store_b16 v195, v196, s[0:3], null offen
	buffer_store_b16 v202, v194, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v195, 0xe0, v197
	s_set_vgpr_msb 9
	v_cvt_pk_bf16_f32 v197, v46 /*v302*/, s0
	v_or_b32_e32 v201, s87, v214 /*v726*/
	v_cvt_pk_bf16_f32 v194, v45 /*v301*/, s0
	s_set_vgpr_msb 0x900
	v_mov_b16_e64 v196.l, v199.l
	v_or_b32_e32 v199, 0xe0, v200
	s_clause 0x1
	buffer_store_b16 v196, v195, s[0:3], null offen
	buffer_store_b16 v194, v199, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v48 /*v304*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v124, v124, s0
	v_or_b32_e32 v199, 1, v201
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v194, v47 /*v303*/, s0
	v_mul_lo_u32 v195, s59, v201
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v125, v125, s0
	v_or_b32_e32 v200, 0xe0, v203
	v_cvt_pk_bf16_f32 v126, v126, s0
	v_cvt_pk_bf16_f32 v127, v127, s0
	v_cvt_pk_bf16_f32 v129, v129, s0
	v_cvt_pk_bf16_f32 v114, v114, s0
	buffer_store_b16 v197, v200, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v197, 0xe0, v204
	v_add_lshl_u32 v195, s89, v195, 7
	v_cvt_pk_bf16_f32 v115, v115, s0
	v_or_b32_e32 v200, 5, v201
	v_cvt_pk_bf16_f32 v116, v116, s0
	buffer_store_b16 v194, v197, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v194, v49 /*v305*/, s0
	s_set_vgpr_msb 0x108
	v_cvt_pk_bf16_f32 v117, v117, s0
	buffer_store_b16 v196, v198, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v196, v195, v209 /*v721*/
	v_mul_lo_u32 v200, v200, s59
	v_cvt_pk_bf16_f32 v120, v120, s0
	v_mul_lo_u32 v197, v199, s59
	v_cvt_pk_bf16_f32 v106, v106, s0
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v196, 1, v196 :: v_dual_bitop2_b32 v199, 3, v201 bitop3:0x54
	v_or_b32_e32 v198, 2, v201
	buffer_store_b16 v194, v1, s[0:3], null offen
	v_add_lshl_u32 v200, s89, v200, 7
	v_cvt_pk_bf16_f32 v107, v107, s0
	v_add_lshl_u32 v197, s89, v197, 7
	v_mul_lo_u32 v198, v198, s59
	buffer_store_b16 v186, v196, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mul_lo_u32 v194, v199, s59
	s_set_vgpr_msb 8
	v_or_b32_e32 v203, v200, v209 /*v721*/
	v_or_b32_e32 v1, v197, v209 /*v721*/
	v_cvt_pk_bf16_f32 v108, v108, s0
	v_cvt_pk_bf16_f32 v98, v98, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v197, v197, v0
	v_add_lshl_u32 v186, s89, v198, 7
	v_lshlrev_b32_e32 v1, 1, v1
	v_add_lshl_u32 v194, s89, v194, 7
	v_dual_lshlrev_b32 v203, 1, v203 :: v_dual_bitop2_b32 v198, 4, v201 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v199, v186, v209 /*v721*/
	buffer_store_b16 v187, v1, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v186, v186, v0
	v_cvt_pk_bf16_f32 v100, v100, s0
	v_mul_lo_u32 v198, v198, s59
	v_lshlrev_b32_e32 v187, 1, v199
	v_cvt_pk_bf16_f32 v99, v99, s0
	v_lshlrev_b32_e32 v186, 1, v186
	s_set_vgpr_msb 8
	v_or_b32_e32 v199, v194, v209 /*v721*/
	v_cvt_pk_bf16_f32 v101, v101, s0
	buffer_store_b16 v188, v187, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v188, 6, v201
	v_add_lshl_u32 v198, s89, v198, 7
	v_lshlrev_b32_e32 v199, 1, v199
	v_cvt_pk_bf16_f32 v90, v90, s0
	v_or_b32_e32 v201, 7, v201
	v_mul_lo_u32 v188, v188, s59
	s_set_vgpr_msb 8
	v_or_b32_e32 v202, v198, v209 /*v721*/
	v_cvt_pk_bf16_f32 v93, v93, s0
	v_cvt_pk_bf16_f32 v94, v94, s0
	v_mul_lo_u32 v201, v201, s59
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v202, 1, v202 :: v_dual_bitop2_b32 v198, v198, v0 bitop3:0x54
	s_clause 0x1
	buffer_store_b16 v189, v199, s[0:3], null offen
	buffer_store_b16 v190, v202, s[0:3], null offen
	v_add_lshl_u32 v188, s89, v188, 7
	s_wait_xcnt 0x0
	v_add_lshl_u32 v190, s89, v201, 7
	v_cvt_pk_bf16_f32 v91, v91, s0
	buffer_store_b16 v191, v203, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v191, v192, s0
	v_or_b32_e32 v192, v195, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v189, v188, v209 /*v721*/
	v_cvt_pk_bf16_f32 v92, v92, s0
	v_or_b32_e32 v195, v190, v209 /*v721*/
	v_cvt_pk_bf16_f32 v82, v82, s0
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v192, 1, v192 :: v_dual_lshlrev_b32 v189, 1, v189
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_lshlrev_b32 v195, 1, v195 :: v_dual_bitop2_b32 v188, v188, v0 bitop3:0x54
	s_clause 0x1
	buffer_store_b16 v191, v189, s[0:3], null offen
	buffer_store_b16 v193, v195, s[0:3], null offen
	s_wait_xcnt 0x1
	v_dual_lshlrev_b32 v191, 1, v197 :: v_dual_bitop2_b32 v201, 32, v192 bitop3:0x54
	v_or_b32_e32 v190, v190, v0
	v_lshlrev_b32_e32 v188, 1, v188
	v_cvt_pk_bf16_f32 v83, v83, s0
	buffer_store_b16 v178, v201, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v178, v194, v0
	v_or_b32_e32 v193, 32, v191
	v_or_b32_e32 v194, 32, v186
	v_cvt_pk_bf16_f32 v85, v85, s0
	v_cvt_pk_bf16_f32 v84, v84, s0
	v_lshlrev_b32_e32 v178, 1, v178
	buffer_store_b16 v179, v193, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v179, v200, v0
	buffer_store_b16 v180, v194, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v180, 1, v198
	v_or_b32_e32 v197, 32, v178
	v_cvt_pk_bf16_f32 v86, v86, s0
	v_lshlrev_b32_e32 v179, 1, v179
	v_cvt_pk_bf16_f32 v74, v74, s0
	v_cvt_pk_bf16_f32 v75, v75, s0
	buffer_store_b16 v181, v197, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v181, v182, s0
	v_cvt_pk_bf16_f32 v182, v183, s0
	v_or_b32_e32 v183, 32, v180
	v_or_b32_e32 v193, 32, v179
	s_clause 0x1
	buffer_store_b16 v181, v183, s[0:3], null offen
	buffer_store_b16 v182, v193, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v181, 1, v190 :: v_dual_bitop2_b32 v182, 32, v188 bitop3:0x54
	v_mov_b16_e64 v183.l, v184.l
	v_cvt_pk_bf16_f32 v184, v185, s0
	v_cvt_pk_bf16_f32 v76, v76, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v185, 32, v181
	v_cvt_pk_bf16_f32 v66, v66, s0
	s_clause 0x2
	buffer_store_b16 v183, v182, s[0:3], null offen
	buffer_store_b16 v184, v185, s[0:3], null offen
	buffer_store_b16 v170, v196, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v170, v173, s0
	v_cvt_pk_bf16_f32 v173, v176, s0
	s_clause 0x1
	buffer_store_b16 v171, v1, s[0:3], null offen offset:64
	buffer_store_b16 v172, v187, s[0:3], null offen offset:64
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v171, v174, s0
	v_cvt_pk_bf16_f32 v174, v177, s0
	buffer_store_b16 v170, v199, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v173.l
	v_cvt_pk_bf16_f32 v172, v175, s0
	s_clause 0x1
	buffer_store_b16 v171, v202, s[0:3], null offen offset:64
	buffer_store_b16 v172, v203, s[0:3], null offen offset:64
	v_mov_b16_e64 v173.l, v174.l
	s_clause 0x1
	buffer_store_b16 v170, v189, s[0:3], null offen offset:64
	buffer_store_b16 v173, v195, s[0:3], null offen offset:64
	s_wait_xcnt 0x1
	v_or_b32_e32 v170, 0x60, v192
	v_or_b32_e32 v172, 0x60, v186
	v_or_b32_e32 v171, 0x60, v191
	s_wait_xcnt 0x0
	v_or_b32_e32 v173, 0x60, v178
	s_clause 0x1
	buffer_store_b16 v162, v170, s[0:3], null offen
	buffer_store_b16 v163, v171, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v162, v166, s0
	s_clause 0x1
	buffer_store_b16 v164, v172, s[0:3], null offen
	buffer_store_b16 v165, v173, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v164, 0x60, v180
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v165, v168, s0
	v_cvt_pk_bf16_f32 v168, v169, s0
	v_cvt_pk_bf16_f32 v163, v167, s0
	v_or_b32_e32 v166, 0x60, v179
	v_or_b32_e32 v167, 0x60, v188
	s_clause 0x1
	buffer_store_b16 v162, v164, s[0:3], null offen
	buffer_store_b16 v163, v166, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v162, 0x60, v181
	s_wait_xcnt 0x0
	v_mov_b16_e64 v163.l, v168.l
	buffer_store_b16 v165, v167, s[0:3], null offen
	v_cvt_pk_bf16_f32 v67, v67, s0
	v_cvt_pk_bf16_f32 v58, v58, s0
	v_cvt_pk_bf16_f32 v59, v59, s0
	s_clause 0x3
	buffer_store_b16 v163, v162, s[0:3], null offen
	buffer_store_b16 v154, v196, s[0:3], null offen offset:128
	buffer_store_b16 v155, v1, s[0:3], null offen offset:128
	buffer_store_b16 v156, v187, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_mov_b16_e64 v154.l, v158.l
	buffer_store_b16 v157, v199, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v155, v159, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v156, v160, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v157, v161, s0
	buffer_store_b16 v154, v202, s[0:3], null offen offset:128
	v_cvt_pk_bf16_f32 v60, v60, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v154.l, v155.l
	v_mov_b16_e64 v155.l, v156.l
	v_mov_b16_e64 v156.l, v157.l
	v_or_b32_e32 v157, 0xa0, v192
	s_clause 0x3
	buffer_store_b16 v154, v203, s[0:3], null offen offset:128
	buffer_store_b16 v155, v189, s[0:3], null offen offset:128
	buffer_store_b16 v156, v195, s[0:3], null offen offset:128
	buffer_store_b16 v146, v157, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v146, 0xa0, v191
	v_or_b32_e32 v155, 0xa0, v178
	v_or_b32_e32 v154, 0xa0, v186
	v_or_b32_e32 v156, 0xa0, v180
	s_clause 0x1
	buffer_store_b16 v147, v146, s[0:3], null offen
	buffer_store_b16 v148, v154, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v146, v151, s0
	s_clause 0x1
	buffer_store_b16 v149, v155, s[0:3], null offen
	buffer_store_b16 v150, v156, s[0:3], null offen
	v_or_b32_e32 v147, 0xa0, v179
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v148, v152, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v149, v153, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v150, 0xa0, v188
	v_or_b32_e32 v151, 0xa0, v181
	buffer_store_b16 v146, v147, s[0:3], null offen
	v_cvt_pk_bf16_f32 v61, v61, s0
	v_cvt_pk_bf16_f32 v62, v62, s0
	s_clause 0x2
	buffer_store_b16 v148, v150, s[0:3], null offen
	buffer_store_b16 v149, v151, s[0:3], null offen
	buffer_store_b16 v138, v196, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v138, v141, s0
	v_cvt_pk_bf16_f32 v141, v142, s0
	v_cvt_pk_bf16_f32 v142, v143, s0
	s_clause 0x2
	buffer_store_b16 v139, v1, s[0:3], null offen offset:192
	buffer_store_b16 v140, v187, s[0:3], null offen offset:192
	buffer_store_b16 v138, v199, s[0:3], null offen offset:192
	s_wait_xcnt 0x2
	v_mov_b16_e64 v1.l, v141.l
	v_mov_b16_e64 v139.l, v142.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v138, v144, s0
	v_or_b32_e32 v140, 0xe0, v191
	v_cvt_pk_bf16_f32 v63, v63, s0
	s_clause 0x1
	buffer_store_b16 v1, v202, s[0:3], null offen offset:192
	buffer_store_b16 v139, v203, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_or_b32_e32 v139, 0xe0, v192
	v_cvt_pk_bf16_f32 v1, v145, s0
	s_clause 0x1
	buffer_store_b16 v138, v189, s[0:3], null offen offset:192
	buffer_store_b16 v1, v195, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v132, s0
	s_clause 0x1
	buffer_store_b16 v130, v139, s[0:3], null offen
	buffer_store_b16 v131, v140, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v131, 0xe0, v186
	s_set_vgpr_msb 8
	v_or_b32_e32 v138, s86, v214 /*v726*/
	v_cvt_pk_bf16_f32 v130, v133, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v133, 0xe0, v178
	v_cvt_pk_bf16_f32 v132, v134, s0
	v_or_b32_e32 v134, 0xe0, v180
	s_clause 0x1
	buffer_store_b16 v1, v131, s[0:3], null offen
	buffer_store_b16 v130, v133, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v130, v138, s59
	v_cvt_pk_bf16_f32 v1, v135, s0
	buffer_store_b16 v132, v134, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v132, 0xe0, v179
	v_or_b32_e32 v134, 1, v138
	v_cvt_pk_bf16_f32 v131, v136, s0
	v_or_b32_e32 v133, 0xe0, v188
	v_or_b32_e32 v135, 3, v138
	v_add_lshl_u32 v130, s89, v130, 7
	buffer_store_b16 v1, v132, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v132, v134, s59
	v_or_b32_e32 v134, 2, v138
	buffer_store_b16 v131, v133, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v131, v130, v209 /*v721*/
	v_cvt_pk_bf16_f32 v1, v137, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v133, 0xe0, v181
	v_mul_lo_u32 v134, v134, s59
	v_dual_lshlrev_b32 v131, 1, v131 :: v_dual_bitop2_b32 v136, 5, v138 bitop3:0x54
	v_add_lshl_u32 v132, s89, v132, 7
	buffer_store_b16 v1, v133, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v133, v135, s59
	v_mul_lo_u32 v136, v136, s59
	buffer_store_b16 v122, v131, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, v132, v209 /*v721*/
	v_add_lshl_u32 v122, v134, s89, 7
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v134, 4, v138
	v_or_b32_e32 v132, v132, v0
	v_cvt_pk_bf16_f32 v65, v65, s0
	v_lshlrev_b32_e32 v1, 1, v1
	s_set_vgpr_msb 8
	v_or_b32_e32 v135, v122, v209 /*v721*/
	v_mul_lo_u32 v134, v134, s59
	v_add_lshl_u32 v133, v133, s89, 7
	v_add_lshl_u32 v136, v136, s89, 7
	buffer_store_b16 v123, v1, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v123, 1, v135 :: v_dual_bitop2_b32 v122, v122, v0 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v135, v133, v209 /*v721*/
	v_or_b32_e32 v139, v136, v209 /*v721*/
	v_add_lshl_u32 v134, v134, s89, 7
	buffer_store_b16 v124, v123, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v124, 6, v138
	v_dual_lshlrev_b32 v135, 1, v135 :: v_dual_bitop2_b32 v138, 7, v138 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v137, v134, v209 /*v721*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v139, 1, v139
	v_mul_lo_u32 v124, v124, s59
	v_mul_lo_u32 v138, v138, s59
	v_dual_lshlrev_b32 v122, 1, v122 :: v_dual_lshlrev_b32 v137, 1, v137
	s_clause 0x1
	buffer_store_b16 v125, v135, s[0:3], null offen
	buffer_store_b16 v126, v137, s[0:3], null offen
	v_add_lshl_u32 v124, s89, v124, 7
	s_wait_xcnt 0x0
	v_add_lshl_u32 v126, s89, v138, 7
	buffer_store_b16 v127, v139, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v127, v128, s0
	v_or_b32_e32 v128, v130, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v125, v124, v209 /*v721*/
	v_or_b32_e32 v130, v126, v209 /*v721*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v134, v134, v0
	v_or_b32_e32 v124, v124, v0
	v_dual_lshlrev_b32 v128, 1, v128 :: v_dual_lshlrev_b32 v125, 1, v125
	v_lshlrev_b32_e32 v130, 1, v130
	s_clause 0x1
	buffer_store_b16 v127, v125, s[0:3], null offen
	buffer_store_b16 v129, v130, s[0:3], null offen
	v_or_b32_e32 v138, 32, v128
	s_wait_xcnt 0x1
	v_dual_lshlrev_b32 v127, 1, v132 :: v_dual_bitop2_b32 v132, 32, v122 bitop3:0x54
	v_or_b32_e32 v126, v126, v0
	v_lshlrev_b32_e32 v124, 1, v124
	buffer_store_b16 v114, v138, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v114, v133, v0
	v_or_b32_e32 v129, 32, v127
	v_cvt_pk_bf16_f32 v50, v50, s0
	v_cvt_pk_bf16_f32 v51, v51, s0
	v_cvt_pk_bf16_f32 v52, v52, s0
	v_lshlrev_b32_e32 v114, 1, v114
	buffer_store_b16 v115, v129, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v115, v136, v0
	buffer_store_b16 v116, v132, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v116, 1, v134
	v_or_b32_e32 v133, 32, v114
	v_cvt_pk_bf16_f32 v53, v53, s0
	v_lshlrev_b32_e32 v115, 1, v115
	v_cvt_pk_bf16_f32 v56, v56, s0
	v_cvt_pk_bf16_f32 v42, v42, s0
	buffer_store_b16 v117, v133, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v117, v118, s0
	v_cvt_pk_bf16_f32 v118, v119, s0
	v_or_b32_e32 v119, 32, v116
	v_or_b32_e32 v129, 32, v115
	s_clause 0x1
	buffer_store_b16 v117, v119, s[0:3], null offen
	buffer_store_b16 v118, v129, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v117, 1, v126 :: v_dual_bitop2_b32 v118, 32, v124 bitop3:0x54
	v_mov_b16_e32 v119.l, v120.l
	v_cvt_pk_bf16_f32 v120, v121, s0
	v_cvt_pk_bf16_f32 v43, v43, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v121, 32, v117
	v_cvt_pk_bf16_f32 v44, v44, s0
	s_clause 0x2
	buffer_store_b16 v119, v118, s[0:3], null offen
	buffer_store_b16 v120, v121, s[0:3], null offen
	buffer_store_b16 v106, v131, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v106, v109, s0
	v_cvt_pk_bf16_f32 v109, v112, s0
	s_clause 0x1
	buffer_store_b16 v107, v1, s[0:3], null offen offset:64
	buffer_store_b16 v108, v123, s[0:3], null offen offset:64
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v107, v110, s0
	v_cvt_pk_bf16_f32 v110, v113, s0
	buffer_store_b16 v106, v135, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e32 v106.l, v109.l
	v_cvt_pk_bf16_f32 v108, v111, s0
	s_clause 0x1
	buffer_store_b16 v107, v137, s[0:3], null offen offset:64
	buffer_store_b16 v108, v139, s[0:3], null offen offset:64
	v_mov_b16_e32 v109.l, v110.l
	s_clause 0x1
	buffer_store_b16 v106, v125, s[0:3], null offen offset:64
	buffer_store_b16 v109, v130, s[0:3], null offen offset:64
	s_wait_xcnt 0x1
	v_or_b32_e32 v106, 0x60, v128
	v_or_b32_e32 v108, 0x60, v122
	v_or_b32_e32 v107, 0x60, v127
	s_wait_xcnt 0x0
	v_or_b32_e32 v109, 0x60, v114
	s_clause 0x1
	buffer_store_b16 v98, v106, s[0:3], null offen
	buffer_store_b16 v99, v107, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v98, v102, s0
	s_clause 0x1
	buffer_store_b16 v100, v108, s[0:3], null offen
	buffer_store_b16 v101, v109, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v100, 0x60, v116
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v101, v104, s0
	v_cvt_pk_bf16_f32 v104, v105, s0
	v_cvt_pk_bf16_f32 v99, v103, s0
	v_or_b32_e32 v102, 0x60, v115
	v_or_b32_e32 v103, 0x60, v124
	s_clause 0x1
	buffer_store_b16 v98, v100, s[0:3], null offen
	buffer_store_b16 v99, v102, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v98, 0x60, v117
	s_wait_xcnt 0x0
	v_mov_b16_e32 v99.l, v104.l
	buffer_store_b16 v101, v103, s[0:3], null offen
	v_cvt_pk_bf16_f32 v34, v34, s0
	v_cvt_pk_bf16_f32 v36, v36, s0
	v_cvt_pk_bf16_f32 v35, v35, s0
	s_clause 0x3
	buffer_store_b16 v99, v98, s[0:3], null offen
	buffer_store_b16 v90, v131, s[0:3], null offen offset:128
	buffer_store_b16 v91, v1, s[0:3], null offen offset:128
	buffer_store_b16 v92, v123, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_mov_b16_e32 v90.l, v94.l
	buffer_store_b16 v93, v135, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v91, v95, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v92, v96, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v93, v97, s0
	buffer_store_b16 v90, v137, s[0:3], null offen offset:128
	v_cvt_pk_bf16_f32 v37, v37, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v90.l, v91.l
	v_mov_b16_e32 v91.l, v92.l
	v_mov_b16_e32 v92.l, v93.l
	v_or_b32_e32 v93, 0xa0, v128
	s_clause 0x3
	buffer_store_b16 v90, v139, s[0:3], null offen offset:128
	buffer_store_b16 v91, v125, s[0:3], null offen offset:128
	buffer_store_b16 v92, v130, s[0:3], null offen offset:128
	buffer_store_b16 v82, v93, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v82, 0xa0, v127
	v_or_b32_e32 v91, 0xa0, v114
	v_or_b32_e32 v90, 0xa0, v122
	v_or_b32_e32 v92, 0xa0, v116
	s_clause 0x1
	buffer_store_b16 v83, v82, s[0:3], null offen
	buffer_store_b16 v84, v90, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v82, v87, s0
	s_clause 0x1
	buffer_store_b16 v85, v91, s[0:3], null offen
	buffer_store_b16 v86, v92, s[0:3], null offen
	v_or_b32_e32 v83, 0xa0, v115
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v84, v88, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v85, v89, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v86, 0xa0, v124
	v_or_b32_e32 v87, 0xa0, v117
	buffer_store_b16 v82, v83, s[0:3], null offen
	v_cvt_pk_bf16_f32 v26, v26, s0
	v_cvt_pk_bf16_f32 v29, v29, s0
	s_clause 0x2
	buffer_store_b16 v84, v86, s[0:3], null offen
	buffer_store_b16 v85, v87, s[0:3], null offen
	buffer_store_b16 v74, v131, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v74, v77, s0
	v_cvt_pk_bf16_f32 v77, v78, s0
	v_cvt_pk_bf16_f32 v78, v79, s0
	s_clause 0x2
	buffer_store_b16 v75, v1, s[0:3], null offen offset:192
	buffer_store_b16 v76, v123, s[0:3], null offen offset:192
	buffer_store_b16 v74, v135, s[0:3], null offen offset:192
	s_wait_xcnt 0x2
	v_mov_b16_e32 v1.l, v77.l
	v_mov_b16_e32 v75.l, v78.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v74, v80, s0
	v_or_b32_e32 v76, 0xe0, v127
	v_cvt_pk_bf16_f32 v30, v30, s0
	s_clause 0x1
	buffer_store_b16 v1, v137, s[0:3], null offen offset:192
	buffer_store_b16 v75, v139, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_or_b32_e32 v75, 0xe0, v128
	v_cvt_pk_bf16_f32 v1, v81, s0
	s_clause 0x1
	buffer_store_b16 v74, v125, s[0:3], null offen offset:192
	buffer_store_b16 v1, v130, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v68, s0
	s_clause 0x1
	buffer_store_b16 v66, v75, s[0:3], null offen
	buffer_store_b16 v67, v76, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v67, 0xe0, v122
	s_set_vgpr_msb 8
	v_or_b32_e32 v74, s62, v214 /*v726*/
	v_cvt_pk_bf16_f32 v66, v69, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v69, 0xe0, v114
	v_cvt_pk_bf16_f32 v68, v70, s0
	v_or_b32_e32 v70, 0xe0, v116
	s_clause 0x1
	buffer_store_b16 v1, v67, s[0:3], null offen
	buffer_store_b16 v66, v69, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v66, v74, s59
	v_cvt_pk_bf16_f32 v1, v71, s0
	buffer_store_b16 v68, v70, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v68, 0xe0, v115
	v_or_b32_e32 v70, 1, v74
	v_cvt_pk_bf16_f32 v67, v72, s0
	v_or_b32_e32 v69, 0xe0, v124
	v_or_b32_e32 v71, 3, v74
	v_add_lshl_u32 v66, s89, v66, 7
	buffer_store_b16 v1, v68, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v68, v70, s59
	v_or_b32_e32 v70, 2, v74
	buffer_store_b16 v67, v69, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v67, v66, v209 /*v721*/
	v_cvt_pk_bf16_f32 v1, v73, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v69, 0xe0, v117
	v_mul_lo_u32 v70, v70, s59
	v_dual_lshlrev_b32 v67, 1, v67 :: v_dual_bitop2_b32 v72, 5, v74 bitop3:0x54
	v_add_lshl_u32 v68, s89, v68, 7
	buffer_store_b16 v1, v69, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v69, v71, s59
	v_mul_lo_u32 v72, v72, s59
	buffer_store_b16 v58, v67, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, v68, v209 /*v721*/
	v_add_lshl_u32 v58, v70, s89, 7
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v70, 4, v74
	v_or_b32_e32 v68, v68, v0
	v_cvt_pk_bf16_f32 v27, v27, s0
	v_lshlrev_b32_e32 v1, 1, v1
	s_set_vgpr_msb 8
	v_or_b32_e32 v71, v58, v209 /*v721*/
	v_mul_lo_u32 v70, v70, s59
	v_add_lshl_u32 v69, v69, s89, 7
	v_add_lshl_u32 v72, v72, s89, 7
	buffer_store_b16 v59, v1, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v59, 1, v71 :: v_dual_bitop2_b32 v58, v58, v0 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v71, v69, v209 /*v721*/
	v_or_b32_e32 v75, v72, v209 /*v721*/
	v_add_lshl_u32 v70, v70, s89, 7
	buffer_store_b16 v60, v59, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v60, 6, v74
	v_dual_lshlrev_b32 v71, 1, v71 :: v_dual_bitop2_b32 v74, 7, v74 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v73, v70, v209 /*v721*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v75, 1, v75
	v_mul_lo_u32 v60, v60, s59
	v_mul_lo_u32 v74, v74, s59
	v_dual_lshlrev_b32 v58, 1, v58 :: v_dual_lshlrev_b32 v73, 1, v73
	s_clause 0x1
	buffer_store_b16 v61, v71, s[0:3], null offen
	buffer_store_b16 v62, v73, s[0:3], null offen
	v_add_lshl_u32 v60, s89, v60, 7
	s_wait_xcnt 0x0
	v_add_lshl_u32 v62, s89, v74, 7
	buffer_store_b16 v63, v75, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v63, v64, s0
	v_or_b32_e32 v64, v66, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v61, v60, v209 /*v721*/
	v_or_b32_e32 v66, v62, v209 /*v721*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v70, v70, v0
	v_or_b32_e32 v60, v60, v0
	v_dual_lshlrev_b32 v64, 1, v64 :: v_dual_lshlrev_b32 v61, 1, v61
	v_lshlrev_b32_e32 v66, 1, v66
	s_clause 0x1
	buffer_store_b16 v63, v61, s[0:3], null offen
	buffer_store_b16 v65, v66, s[0:3], null offen
	v_or_b32_e32 v74, 32, v64
	s_wait_xcnt 0x1
	v_dual_lshlrev_b32 v63, 1, v68 :: v_dual_bitop2_b32 v68, 32, v58 bitop3:0x54
	v_lshlrev_b32_e32 v60, 1, v60
	v_cvt_pk_bf16_f32 v28, v28, s0
	buffer_store_b16 v50, v74, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v50, v69, v0
	v_or_b32_e32 v65, 32, v63
	v_cvt_pk_bf16_f32 v18, v18, s0
	v_cvt_pk_bf16_f32 v19, v19, s0
	v_cvt_pk_bf16_f32 v21, v21, s0
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_b16 v51, v65, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v51, v72, v0
	buffer_store_b16 v52, v68, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v52, 1, v70
	v_or_b32_e32 v69, 32, v50
	v_dual_lshlrev_b32 v51, 1, v51 :: v_dual_bitop2_b32 v0, v62, v0 bitop3:0x54
	v_cvt_pk_bf16_f32 v22, v22, s0
	v_cvt_pk_bf16_f32 v23, v23, s0
	buffer_store_b16 v53, v69, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v53, v54, s0
	v_cvt_pk_bf16_f32 v54, v55, s0
	v_or_b32_e32 v55, 32, v52
	v_dual_lshlrev_b32 v0, 1, v0 :: v_dual_bitop2_b32 v65, 32, v51 bitop3:0x54
	s_clause 0x1
	buffer_store_b16 v53, v55, s[0:3], null offen
	buffer_store_b16 v54, v65, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v53, 32, v60
	s_wait_xcnt 0x0
	v_mov_b16_e32 v54.l, v56.l
	v_cvt_pk_bf16_f32 v55, v57, s0
	v_or_b32_e32 v56, 32, v0
	v_cvt_pk_bf16_f32 v12, v12, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	s_clause 0x2
	buffer_store_b16 v54, v53, s[0:3], null offen
	buffer_store_b16 v55, v56, s[0:3], null offen
	buffer_store_b16 v42, v67, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v42, v45, s0
	v_cvt_pk_bf16_f32 v45, v48, s0
	s_clause 0x1
	buffer_store_b16 v43, v1, s[0:3], null offen offset:64
	buffer_store_b16 v44, v59, s[0:3], null offen offset:64
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v43, v46, s0
	v_cvt_pk_bf16_f32 v46, v49, s0
	buffer_store_b16 v42, v71, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e32 v42.l, v45.l
	v_cvt_pk_bf16_f32 v44, v47, s0
	s_clause 0x1
	buffer_store_b16 v43, v73, s[0:3], null offen offset:64
	buffer_store_b16 v44, v75, s[0:3], null offen offset:64
	v_mov_b16_e32 v45.l, v46.l
	s_clause 0x1
	buffer_store_b16 v42, v61, s[0:3], null offen offset:64
	buffer_store_b16 v45, v66, s[0:3], null offen offset:64
	s_wait_xcnt 0x1
	v_or_b32_e32 v42, 0x60, v64
	v_or_b32_e32 v44, 0x60, v58
	v_or_b32_e32 v43, 0x60, v63
	s_wait_xcnt 0x0
	v_or_b32_e32 v45, 0x60, v50
	s_clause 0x1
	buffer_store_b16 v34, v42, s[0:3], null offen
	buffer_store_b16 v35, v43, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v34, v38, s0
	s_clause 0x1
	buffer_store_b16 v36, v44, s[0:3], null offen
	buffer_store_b16 v37, v45, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v36, 0x60, v52
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v37, v40, s0
	v_cvt_pk_bf16_f32 v40, v41, s0
	v_cvt_pk_bf16_f32 v35, v39, s0
	v_or_b32_e32 v38, 0x60, v51
	v_or_b32_e32 v39, 0x60, v60
	s_clause 0x1
	buffer_store_b16 v34, v36, s[0:3], null offen
	buffer_store_b16 v35, v38, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v34, 0x60, v0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v35.l, v40.l
	buffer_store_b16 v37, v39, s[0:3], null offen
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_cvt_pk_bf16_f32 v2, v2, s0
	s_clause 0x3
	buffer_store_b16 v35, v34, s[0:3], null offen
	buffer_store_b16 v26, v67, s[0:3], null offen offset:128
	buffer_store_b16 v27, v1, s[0:3], null offen offset:128
	buffer_store_b16 v28, v59, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_mov_b16_e32 v26.l, v30.l
	buffer_store_b16 v29, v71, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v27, v31, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v28, v32, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v29, v33, s0
	buffer_store_b16 v26, v73, s[0:3], null offen offset:128
	v_cvt_pk_bf16_f32 v4, v4, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v26.l, v27.l
	v_mov_b16_e32 v27.l, v28.l
	v_mov_b16_e32 v28.l, v29.l
	v_or_b32_e32 v29, 0xa0, v64
	s_clause 0x3
	buffer_store_b16 v26, v75, s[0:3], null offen offset:128
	buffer_store_b16 v27, v61, s[0:3], null offen offset:128
	buffer_store_b16 v28, v66, s[0:3], null offen offset:128
	buffer_store_b16 v18, v29, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v18, v20, s0
	v_or_b32_e32 v20, 0xa0, v63
	v_or_b32_e32 v26, 0xa0, v58
	v_or_b32_e32 v27, 0xa0, v50
	v_or_b32_e32 v28, 0xa0, v52
	v_or_b32_e32 v29, 0xa0, v51
	s_clause 0x4
	buffer_store_b16 v19, v20, s[0:3], null offen
	buffer_store_b16 v18, v26, s[0:3], null offen
	buffer_store_b16 v21, v27, s[0:3], null offen
	buffer_store_b16 v22, v28, s[0:3], null offen
	buffer_store_b16 v23, v29, s[0:3], null offen
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v18, v24, s0
	v_or_b32_e32 v19, 0xa0, v60
	v_cvt_pk_bf16_f32 v20, v25, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v21, 0xa0, v0
	v_or_b32_e32 v0, 0xe0, v0
	s_clause 0x3
	buffer_store_b16 v18, v19, s[0:3], null offen
	buffer_store_b16 v20, v21, s[0:3], null offen
	buffer_store_b16 v10, v67, s[0:3], null offen offset:192
	buffer_store_b16 v11, v1, s[0:3], null offen offset:192
	s_wait_xcnt 0x1
	v_mov_b16_e32 v10.l, v12.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v11, v14, s0
	v_cvt_pk_bf16_f32 v12, v15, s0
	v_cvt_pk_bf16_f32 v1, v13, s0
	v_cvt_pk_bf16_f32 v13, v16, s0
	buffer_store_b16 v10, v59, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_mov_b16_e32 v10.l, v11.l
	v_mov_b16_e32 v11.l, v12.l
	buffer_store_b16 v1, v71, s[0:3], null offen offset:192
	v_mov_b16_e32 v12.l, v13.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v17, s0
	s_clause 0x2
	buffer_store_b16 v10, v73, s[0:3], null offen offset:192
	buffer_store_b16 v11, v75, s[0:3], null offen offset:192
	buffer_store_b16 v12, v61, s[0:3], null offen offset:192
	s_wait_xcnt 0x1
	v_or_b32_e32 v11, 0xe0, v63
	v_or_b32_e32 v10, 0xe0, v64
	s_wait_xcnt 0x0
	v_or_b32_e32 v12, 0xe0, v58
	s_clause 0x1
	buffer_store_b16 v1, v66, s[0:3], null offen offset:192
	buffer_store_b16 v2, v10, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v5, s0
	s_clause 0x1
	buffer_store_b16 v3, v11, s[0:3], null offen
	buffer_store_b16 v4, v12, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v3, 0xe0, v50
	v_cvt_pk_bf16_f32 v2, v6, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v4, 0xe0, v52
	v_cvt_pk_bf16_f32 v5, v7, s0
	v_or_b32_e32 v7, 0xe0, v51
	v_cvt_pk_bf16_f32 v6, v8, s0
	v_cvt_pk_bf16_f32 v8, v9, s0
	v_or_b32_e32 v9, 0xe0, v60
	s_clause 0x4
	buffer_store_b16 v1, v3, s[0:3], null offen
	buffer_store_b16 v2, v4, s[0:3], null offen
	buffer_store_b16 v5, v7, s[0:3], null offen
	buffer_store_b16 v6, v9, s[0:3], null offen
	buffer_store_b16 v8, v0, s[0:3], null offen
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	k_dqg_0, .Lfunc_end0-k_dqg_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dqg_0
		.amdhsa_group_segment_fixed_size 52224
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 404
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
		.amdhsa_next_free_vgpr 881
		.amdhsa_next_free_sgpr 93
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-k_dqg_0)<<4)&4080)>>4
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

	.set .Lk_dqg_0.num_vgpr, 881
	.set .Lk_dqg_0.num_agpr, 0
	.set .Lk_dqg_0.numbered_sgpr, 93
	.set .Lk_dqg_0.num_named_barrier, 0
	.set .Lk_dqg_0.private_seg_size, 0
	.set .Lk_dqg_0.uses_vcc, 1
	.set .Lk_dqg_0.uses_flat_scratch, 0
	.set .Lk_dqg_0.has_dyn_sized_stack, 0
	.set .Lk_dqg_0.has_recursion, 0
	.set .Lk_dqg_0.has_indirect_call, 0
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
        .size:           40
        .value_kind:     by_value
      - .address_space:  global
        .offset:         240
        .size:           8
        .value_kind:     global_buffer
      - .offset:         248
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         280
        .size:           8
        .value_kind:     global_buffer
      - .offset:         288
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         320
        .size:           8
        .value_kind:     global_buffer
      - .offset:         328
        .size:           40
        .value_kind:     by_value
      - .offset:         368
        .size:           4
        .value_kind:     by_value
      - .offset:         372
        .size:           4
        .value_kind:     by_value
      - .offset:         376
        .size:           4
        .value_kind:     by_value
      - .offset:         380
        .size:           4
        .value_kind:     by_value
      - .offset:         384
        .size:           4
        .value_kind:     by_value
      - .offset:         388
        .size:           4
        .value_kind:     by_value
      - .offset:         392
        .size:           4
        .value_kind:     by_value
      - .offset:         396
        .size:           4
        .value_kind:     by_value
      - .offset:         400
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 52224
    .kernarg_segment_align: 8
    .kernarg_segment_size: 404
    .max_flat_workgroup_size: 32
    .name:           k_dqg_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     95
    .sgpr_spill_count: 0
    .symbol:         k_dqg_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     881
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
