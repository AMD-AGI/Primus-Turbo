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
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_bfe_u32 s4, ttmp6, 0x40010
	s_add_co_i32 s2, s2, 1
	s_load_b256 s[64:71], s[0:1], 0x140 nv
	s_and_b32 s3, ttmp6, 15
	s_mul_i32 s2, ttmp9, s2
	s_and_b32 s5, ttmp7, 0xffff
	s_add_co_i32 s4, s4, 1
	s_add_co_i32 s3, s3, s2
	s_mul_i32 s2, s5, s4
	s_bfe_u32 s4, ttmp6, 0x40004
	s_getreg_b32 s6, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s4, s4, s2
	s_cmp_eq_u32 s6, 0
	s_set_vgpr_msb 0x80
	v_dual_lshrrev_b32 v16 /*v528*/, 4, v0 :: v_dual_bitop2_b32 v109 /*v621*/, 15, v0 bitop3:0x40
	s_cselect_b32 s82, s5, s4
	s_cselect_b32 s2, ttmp9, s3
	s_bfe_u32 s3, ttmp6, 0x40014
	s_lshr_b32 s4, ttmp7, 16
	s_add_co_i32 s3, s3, 1
	s_bfe_u32 s5, ttmp6, 0x40008
	s_mul_i32 s3, s4, s3
	s_mov_b32 s14, 0x200000
	s_add_co_i32 s5, s5, s3
	s_cmp_eq_u32 s6, 0
	s_mov_b64 s[78:79], 0x800000
	s_cselect_b32 s18, s4, s5
	s_lshl_b32 s83, s2, 6
	s_load_b32 s4, s[0:1], 0x160 nv
	s_wait_kmcnt 0x0
	s_add_co_i32 s16, s71, s83
	s_mul_i32 s85, s65, s18
	s_add_co_i32 s2, s16, 0x5f
	s_set_vgpr_msb 0x8008
	v_or_b32_e32 v2, s83, v109 /*v621*/
	s_ashr_i32 s3, s2, 31
	s_mov_b64 s[74:75], 0x800000
	s_lshr_b32 s3, s3, 27
	s_set_vgpr_msb 0x888
	v_lshlrev_b32_e32 v112 /*v624*/, 3, v16 /*v528*/
	s_add_co_i32 s3, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s5, s3, 0xffffffe0
	s_ashr_i32 s3, s3, 5
	s_cmp_lg_u32 s2, s5
	s_cselect_b32 s5, -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, s5
	s_sub_co_ci_u32 s2, s3, 0
	s_max_i32 s2, s2, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s2, s2, s70
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
	s_abs_i32 s3, s69
	s_xor_b32 s7, s69, s82
	s_cvt_f32_u32 s4, s3
	s_sub_co_i32 s5, 0, s3
	s_ashr_i32 s8, s7, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_s_rcp_f32 s4, s4
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s4, s4
	s_mul_i32 s5, s5, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s5, s4, s5
	s_add_co_i32 s4, s4, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s4, s82, s4
	s_mul_i32 s5, s4, s3
	s_add_co_i32 s6, s4, 1
	s_sub_co_i32 s5, s82, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s9, s5, s3
	s_cmp_ge_u32 s5, s3
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s5, s9, s5
	s_add_co_i32 s6, s4, 1
	s_cmp_ge_u32 s5, s3
	s_cselect_b32 s3, s6, s4
	s_mov_b32 s6, 0x800000
	s_xor_b32 s3, s3, s8
	s_mov_b32 s10, s6
	s_sub_co_i32 s4, s3, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s4, s4, s69
	s_cmp_lg_u32 s82, s4
	s_load_b64 s[4:5], s[0:1], 0x0 nv
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s7, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s7, s9
	s_sub_co_ci_u32 s3, s3, s8
	s_lshl_b32 s12, s67, 4
	s_or_b32 s81, s83, 16
	s_mul_i32 s7, s12, s85
	s_or_b32 s80, s83, 32
	s_lshl4_add_u32 s13, s82, s7
	s_set_vgpr_msb 0x8808
	v_or_b32_e32 v1, s81, v109 /*v621*/
	s_set_vgpr_msb 0x888
	v_or_b32_e32 v17 /*v529*/, s80, v109 /*v621*/
	s_set_vgpr_msb 0x8800
	v_mad_u32 v3, s12, v2, s13
	s_clause 0x2
	s_load_b64 s[8:9], s[0:1], 0x90 nv
	s_load_b64 s[72:73], s[0:1], 0x30 nv
	s_load_b64 s[76:77], s[0:1], 0x60 nv
	s_or_b32 s69, s83, 48
	v_mad_u32 v4, s12, v1, s13
	s_set_vgpr_msb 8
	v_mad_u32 v5, s12, v17 /*v529*/, s13
	s_set_vgpr_msb 0x888
	v_or_b32_e32 v18 /*v530*/, s69, v109 /*v621*/
	s_mov_b32 s7, 0
	s_set_vgpr_msb 0x8808
	v_or_b32_e32 v3, v3, v16 /*v528*/
	s_mov_b32 s11, s7
	v_mad_u32 v6, s12, v18 /*v530*/, s13
	v_or_b32_e32 v4, v4, v16 /*v528*/
	v_or_b32_e32 v5, v5, v16 /*v528*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v3, 4, v3
	s_load_b64 s[12:13], s[0:1], 0xc0 nv
	s_mov_b32 s15, s7
	v_dual_lshlrev_b32 v4, 4, v4 :: v_dual_lshlrev_b32 v5, 4, v5
	s_wait_kmcnt 0x0
	s_clause 0x8
	buffer_load_b128 v[242:245], v3, s[4:7], null offen
	buffer_load_b128 v[246:249], v3, s[4:7], null offen offset:32
	buffer_load_b128 v[250:253], v3, s[4:7], null offen offset:64
	buffer_load_b128 v[254:257], v3, s[4:7], null offen offset:96
	s_set_vgpr_msb 64
	buffer_load_b128 v[2:5] /*v[258:261]*/, v3, s[4:7], null offen offset:128
	buffer_load_b128 v[6:9] /*v[262:265]*/, v3, s[4:7], null offen offset:160
	buffer_load_b128 v[10:13] /*v[266:269]*/, v3, s[4:7], null offen offset:192
	buffer_load_b128 v[14:17] /*v[270:273]*/, v3, s[4:7], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[18:21] /*v[274:277]*/, v3, s[8:11], null offen
	buffer_load_b128 v[22:25] /*v[278:281]*/, v3, s[8:11], null offen offset:32
	buffer_load_b128 v[26:29] /*v[282:285]*/, v3, s[8:11], null offen offset:64
	buffer_load_b128 v[30:33] /*v[286:289]*/, v3, s[8:11], null offen offset:96
	buffer_load_b128 v[34:37] /*v[290:293]*/, v3, s[8:11], null offen offset:128
	buffer_load_b128 v[38:41] /*v[294:297]*/, v3, s[8:11], null offen offset:160
	buffer_load_b128 v[42:45] /*v[298:301]*/, v3, s[8:11], null offen offset:192
	buffer_load_b128 v[46:49] /*v[302:305]*/, v3, s[8:11], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[50:53] /*v[306:309]*/, v4, s[4:7], null offen
	buffer_load_b128 v[54:57] /*v[310:313]*/, v4, s[4:7], null offen offset:32
	buffer_load_b128 v[58:61] /*v[314:317]*/, v4, s[4:7], null offen offset:64
	buffer_load_b128 v[62:65] /*v[318:321]*/, v4, s[4:7], null offen offset:96
	buffer_load_b128 v[66:69] /*v[322:325]*/, v4, s[4:7], null offen offset:128
	buffer_load_b128 v[70:73] /*v[326:329]*/, v4, s[4:7], null offen offset:160
	buffer_load_b128 v[74:77] /*v[330:333]*/, v4, s[4:7], null offen offset:192
	buffer_load_b128 v[78:81] /*v[334:337]*/, v4, s[4:7], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[82:85] /*v[338:341]*/, v4, s[8:11], null offen
	buffer_load_b128 v[86:89] /*v[342:345]*/, v4, s[8:11], null offen offset:32
	buffer_load_b128 v[90:93] /*v[346:349]*/, v4, s[8:11], null offen offset:64
	buffer_load_b128 v[94:97] /*v[350:353]*/, v4, s[8:11], null offen offset:96
	buffer_load_b128 v[106:109] /*v[362:365]*/, v4, s[8:11], null offen offset:128
	buffer_load_b128 v[110:113] /*v[366:369]*/, v4, s[8:11], null offen offset:160
	buffer_load_b128 v[114:117] /*v[370:373]*/, v4, s[8:11], null offen offset:192
	buffer_load_b128 v[118:121] /*v[374:377]*/, v4, s[8:11], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[122:125] /*v[378:381]*/, v5, s[4:7], null offen
	buffer_load_b128 v[126:129] /*v[382:385]*/, v5, s[4:7], null offen offset:32
	buffer_load_b128 v[130:133] /*v[386:389]*/, v5, s[4:7], null offen offset:64
	buffer_load_b128 v[134:137] /*v[390:393]*/, v5, s[4:7], null offen offset:96
	buffer_load_b128 v[138:141] /*v[394:397]*/, v5, s[4:7], null offen offset:128
	buffer_load_b128 v[142:145] /*v[398:401]*/, v5, s[4:7], null offen offset:160
	buffer_load_b128 v[146:149] /*v[402:405]*/, v5, s[4:7], null offen offset:192
	buffer_load_b128 v[150:153] /*v[406:409]*/, v5, s[4:7], null offen offset:224
	s_set_vgpr_msb 0x4008
	v_or_b32_e32 v3, v6, v16 /*v528*/
	s_set_vgpr_msb 0x800
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_lshrrev_b32 v6, 3, v0 :: v_dual_lshlrev_b32 v3, 4, v3
	s_set_vgpr_msb 64
	s_clause 0x3
	buffer_load_b128 v[170:173] /*v[426:429]*/, v5, s[8:11], null offen offset:128
	buffer_load_b128 v[174:177] /*v[430:433]*/, v5, s[8:11], null offen offset:160
	buffer_load_b128 v[178:181] /*v[434:437]*/, v5, s[8:11], null offen offset:192
	buffer_load_b128 v[182:185] /*v[438:441]*/, v5, s[8:11], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[186:189] /*v[442:445]*/, v3, s[4:7], null offen
	buffer_load_b128 v[190:193] /*v[446:449]*/, v3, s[4:7], null offen offset:32
	buffer_load_b128 v[194:197] /*v[450:453]*/, v3, s[4:7], null offen offset:64
	buffer_load_b128 v[198:201] /*v[454:457]*/, v3, s[4:7], null offen offset:96
	buffer_load_b128 v[202:205] /*v[458:461]*/, v3, s[4:7], null offen offset:128
	buffer_load_b128 v[206:209] /*v[462:465]*/, v3, s[4:7], null offen offset:160
	buffer_load_b128 v[210:213] /*v[466:469]*/, v3, s[4:7], null offen offset:192
	buffer_load_b128 v[214:217] /*v[470:473]*/, v3, s[4:7], null offen offset:224
	s_wait_xcnt 0x0
	s_load_b64 s[4:5], s[0:1], 0xe8 nv
	s_mul_i32 s6, s67, s18
	s_clause 0x3
	buffer_load_b128 v[154:157] /*v[410:413]*/, v5, s[8:11], null offen
	buffer_load_b128 v[158:161] /*v[414:417]*/, v5, s[8:11], null offen offset:32
	buffer_load_b128 v[162:165] /*v[418:421]*/, v5, s[8:11], null offen offset:64
	buffer_load_b128 v[166:169] /*v[422:425]*/, v5, s[8:11], null offen offset:96
	s_add_co_i32 s6, s6, s82
	s_clause 0x7
	buffer_load_b128 v[226:229] /*v[482:485]*/, v3, s[8:11], null offen
	buffer_load_b128 v[230:233] /*v[486:489]*/, v3, s[8:11], null offen offset:32
	buffer_load_b128 v[234:237] /*v[490:493]*/, v3, s[8:11], null offen offset:64
	buffer_load_b128 v[238:241] /*v[494:497]*/, v3, s[8:11], null offen offset:96
	buffer_load_b128 v[242:245] /*v[498:501]*/, v3, s[8:11], null offen offset:128
	buffer_load_b128 v[246:249] /*v[502:505]*/, v3, s[8:11], null offen offset:160
	buffer_load_b128 v[250:253] /*v[506:509]*/, v3, s[8:11], null offen offset:192
	buffer_load_b128 v[254:257] /*v[510:513]*/, v3, s[8:11], null offen offset:224
	s_wait_xcnt 0x0
	s_mul_i32 s8, s6, s65
	s_mov_b32 s6, s14
	s_set_vgpr_msb 0x4000
	v_add_lshl_u32 v2, s8, v2, 2
	v_add_lshl_u32 v3, s8, v1, 2
	s_set_vgpr_msb 8
	v_add_lshl_u32 v4, s8, v17 /*v529*/, 2
	v_add_lshl_u32 v5, s8, v18 /*v530*/, 2
	s_set_vgpr_msb 0x880
	s_clause 0x3
	buffer_load_b32 v86 /*v598*/, v2, s[12:15], null offen
	buffer_load_b32 v88 /*v600*/, v3, s[12:15], null offen
	buffer_load_b32 v90 /*v602*/, v4, s[12:15], null offen
	buffer_load_b32 v92 /*v604*/, v5, s[12:15], null offen
	s_wait_kmcnt 0x0
	s_clause 0x3
	buffer_load_b32 v94 /*v606*/, v2, s[4:7], null offen
	buffer_load_b32 v96 /*v608*/, v3, s[4:7], null offen
	buffer_load_b32 v98 /*v610*/, v4, s[4:7], null offen
	buffer_load_b32 v100 /*v612*/, v5, s[4:7], null offen
	s_set_vgpr_msb 0x8020
	v_and_b32_e32 v3, 16, v0
	v_or_b32_e32 v2, 16, v0
	v_and_or_b32 v4, v0, 7, v112 /*v624*/
	v_bfe_u32 v5, v0, 3, 1
	s_set_vgpr_msb 0x2080
	v_lshlrev_b32_e32 v21 /*v533*/, 4, v6
	s_set_vgpr_msb 0x8088
	v_mad_u32_u24 v113 /*v625*/, 0x110, v109 /*v621*/, v3
	s_set_vgpr_msb 0x8880
	v_mad_u32_u24 v114 /*v626*/, 0x110, v2, v3
	v_mul_u32_u24_e32 v19 /*v531*/, 0x110, v4
	v_lshlrev_b32_e32 v20 /*v532*/, 4, v5
	s_lshl_b32 s5, s3, 4
	s_ashr_i32 s3, s2, 31
	s_cmp_lt_i32 s2, 1
	s_mul_i32 s8, s18, s66
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_3
	s_lshl_b32 s4, s8, 4
	s_set_vgpr_msb 10
	v_or_b32_e32 v5, 0x60, v21 /*v533*/
	v_lshl_add_u32 v8, v109 /*v621*/, 4, s4
	s_set_vgpr_msb 0xa08
	v_lshl_add_u32 v2, v2, 4, s4
	v_or_b32_e32 v6, 0x80, v20 /*v532*/
	v_or_b32_e32 v9, 0xa0, v21 /*v533*/
	v_or_b32_e32 v10, 0xc0, v20 /*v532*/
	v_mul_lo_u32 v8, v8, s68
	v_mul_lo_u32 v2, v2, s68
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v7, 1, v0
	s_set_vgpr_msb 64
	v_mov_b32_e32 v218 /*v474*/, 0
	s_set_vgpr_msb 0x4082
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v101 /*v613*/, v100 /*v612*/ :: v_dual_mov_b32 v99 /*v611*/, v98 /*v610*/
	s_set_vgpr_msb 0x8209
	v_or_b32_e32 v4, 64, v20 /*v532*/
	v_dual_mov_b32 v234, v218 /*v474*/ :: v_dual_bitop2_b32 v3, 32, v21 /*v533*/ bitop3:0x54
	s_set_vgpr_msb 0x902
	v_or_b32_e32 v8, v16 /*v528*/, v8
	v_or_b32_e32 v2, v16 /*v528*/, v2
	s_set_vgpr_msb 0x282
	v_dual_mov_b32 v97 /*v609*/, v96 /*v608*/ :: v_dual_mov_b32 v95 /*v607*/, v94 /*v606*/
	s_set_vgpr_msb 0x8200
	v_and_or_b32 v7, v7, 16, 0xe0
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v87 /*v599*/, v86 /*v598*/ :: v_dual_mov_b32 v89 /*v601*/, v88 /*v600*/
	v_dual_mov_b32 v91 /*v603*/, v90 /*v602*/ :: v_dual_mov_b32 v93 /*v605*/, v92 /*v604*/
	v_add_lshl_u32 v22 /*v534*/, s5, v8, 4
	v_add_lshl_u32 v23 /*v535*/, s5, v2, 4
	v_dual_add_nc_u32 v26 /*v538*/, v19 /*v531*/, v5 :: v_dual_add_nc_u32 v27 /*v539*/, v19 /*v531*/, v6
	v_dual_add_nc_u32 v28 /*v540*/, v19 /*v531*/, v9 :: v_dual_add_nc_u32 v29 /*v541*/, v19 /*v531*/, v10
	v_add_nc_u32_e32 v30 /*v542*/, v19 /*v531*/, v7
	s_set_vgpr_msb 0x8241
	v_dual_mov_b32 v219 /*v475*/, v218 /*v474*/ :: v_dual_mov_b32 v220 /*v476*/, v218 /*v474*/
	v_dual_mov_b32 v221 /*v477*/, v218 /*v474*/ :: v_dual_mov_b32 v222 /*v478*/, v218 /*v474*/
	v_dual_mov_b32 v223 /*v479*/, v218 /*v474*/ :: v_dual_mov_b32 v224 /*v480*/, v218 /*v474*/
	v_dual_mov_b32 v225 /*v481*/, v218 /*v474*/ :: v_dual_mov_b32 v98 /*v354*/, v218 /*v474*/
	v_dual_mov_b32 v99 /*v355*/, v218 /*v474*/ :: v_dual_mov_b32 v100 /*v356*/, v218 /*v474*/
	v_dual_mov_b32 v101 /*v357*/, v218 /*v474*/ :: v_dual_mov_b32 v102 /*v358*/, v218 /*v474*/
	v_dual_mov_b32 v103 /*v359*/, v218 /*v474*/ :: v_dual_mov_b32 v104 /*v360*/, v218 /*v474*/
	v_mov_b32_e32 v105 /*v361*/, v218 /*v474*/
	s_set_vgpr_msb 0x4101
	v_mov_b32_e32 v235, v218 /*v474*/
	s_set_vgpr_msb 0x182
	v_dual_add_nc_u32 v24 /*v536*/, v19 /*v531*/, v3 :: v_dual_add_nc_u32 v25 /*v537*/, v19 /*v531*/, v4
	s_set_vgpr_msb 0x8201
	v_dual_mov_b32 v236, v218 /*v474*/ :: v_dual_mov_b32 v237, v218 /*v474*/
	v_dual_mov_b32 v238, v218 /*v474*/ :: v_dual_mov_b32 v239, v218 /*v474*/
	v_dual_mov_b32 v240, v218 /*v474*/ :: v_dual_mov_b32 v241, v218 /*v474*/
	v_dual_mov_b32 v226, v218 /*v474*/ :: v_dual_mov_b32 v227, v218 /*v474*/
	v_dual_mov_b32 v228, v218 /*v474*/ :: v_dual_mov_b32 v229, v218 /*v474*/
	v_dual_mov_b32 v230, v218 /*v474*/ :: v_dual_mov_b32 v231, v218 /*v474*/
	v_dual_mov_b32 v232, v218 /*v474*/ :: v_dual_mov_b32 v233, v218 /*v474*/
	v_dual_mov_b32 v218, v218 /*v474*/ :: v_dual_mov_b32 v219, v218 /*v474*/
	v_dual_mov_b32 v220, v218 /*v474*/ :: v_dual_mov_b32 v221, v218 /*v474*/
	v_dual_mov_b32 v222, v218 /*v474*/ :: v_dual_mov_b32 v223, v218 /*v474*/
	v_dual_mov_b32 v224, v218 /*v474*/ :: v_dual_mov_b32 v225, v218 /*v474*/
	v_dual_mov_b32 v210, v218 /*v474*/ :: v_dual_mov_b32 v211, v218 /*v474*/
	v_dual_mov_b32 v212, v218 /*v474*/ :: v_dual_mov_b32 v213, v218 /*v474*/
	v_dual_mov_b32 v214, v218 /*v474*/ :: v_dual_mov_b32 v215, v218 /*v474*/
	v_dual_mov_b32 v216, v218 /*v474*/ :: v_dual_mov_b32 v217, v218 /*v474*/
	v_dual_mov_b32 v202, v218 /*v474*/ :: v_dual_mov_b32 v203, v218 /*v474*/
	v_dual_mov_b32 v204, v218 /*v474*/ :: v_dual_mov_b32 v205, v218 /*v474*/
	v_dual_mov_b32 v206, v218 /*v474*/ :: v_dual_mov_b32 v207, v218 /*v474*/
	v_dual_mov_b32 v208, v218 /*v474*/ :: v_dual_mov_b32 v209, v218 /*v474*/
	v_dual_mov_b32 v194, v218 /*v474*/ :: v_dual_mov_b32 v195, v218 /*v474*/
	v_dual_mov_b32 v196, v218 /*v474*/ :: v_dual_mov_b32 v197, v218 /*v474*/
	v_dual_mov_b32 v198, v218 /*v474*/ :: v_dual_mov_b32 v199, v218 /*v474*/
	v_dual_mov_b32 v200, v218 /*v474*/ :: v_dual_mov_b32 v201, v218 /*v474*/
	v_dual_mov_b32 v186, v218 /*v474*/ :: v_dual_mov_b32 v187, v218 /*v474*/
	v_dual_mov_b32 v188, v218 /*v474*/ :: v_dual_mov_b32 v189, v218 /*v474*/
	v_dual_mov_b32 v190, v218 /*v474*/ :: v_dual_mov_b32 v191, v218 /*v474*/
	v_dual_mov_b32 v192, v218 /*v474*/ :: v_dual_mov_b32 v193, v218 /*v474*/
	v_dual_mov_b32 v178, v218 /*v474*/ :: v_dual_mov_b32 v179, v218 /*v474*/
	v_dual_mov_b32 v180, v218 /*v474*/ :: v_dual_mov_b32 v181, v218 /*v474*/
	v_dual_mov_b32 v182, v218 /*v474*/ :: v_dual_mov_b32 v183, v218 /*v474*/
	v_dual_mov_b32 v184, v218 /*v474*/ :: v_dual_mov_b32 v185, v218 /*v474*/
	v_dual_mov_b32 v170, v218 /*v474*/ :: v_dual_mov_b32 v171, v218 /*v474*/
	v_dual_mov_b32 v172, v218 /*v474*/ :: v_dual_mov_b32 v173, v218 /*v474*/
	v_dual_mov_b32 v174, v218 /*v474*/ :: v_dual_mov_b32 v175, v218 /*v474*/
	v_dual_mov_b32 v176, v218 /*v474*/ :: v_dual_mov_b32 v177, v218 /*v474*/
	v_dual_mov_b32 v162, v218 /*v474*/ :: v_dual_mov_b32 v163, v218 /*v474*/
	v_dual_mov_b32 v164, v218 /*v474*/ :: v_dual_mov_b32 v165, v218 /*v474*/
	v_dual_mov_b32 v166, v218 /*v474*/ :: v_dual_mov_b32 v167, v218 /*v474*/
	v_dual_mov_b32 v168, v218 /*v474*/ :: v_dual_mov_b32 v169, v218 /*v474*/
	v_dual_mov_b32 v154, v218 /*v474*/ :: v_dual_mov_b32 v155, v218 /*v474*/
	v_dual_mov_b32 v156, v218 /*v474*/ :: v_dual_mov_b32 v157, v218 /*v474*/
	v_dual_mov_b32 v158, v218 /*v474*/ :: v_dual_mov_b32 v159, v218 /*v474*/
	v_dual_mov_b32 v160, v218 /*v474*/ :: v_dual_mov_b32 v161, v218 /*v474*/
	v_dual_mov_b32 v146, v218 /*v474*/ :: v_dual_mov_b32 v147, v218 /*v474*/
	v_dual_mov_b32 v148, v218 /*v474*/ :: v_dual_mov_b32 v149, v218 /*v474*/
	v_dual_mov_b32 v150, v218 /*v474*/ :: v_dual_mov_b32 v151, v218 /*v474*/
	v_dual_mov_b32 v152, v218 /*v474*/ :: v_dual_mov_b32 v153, v218 /*v474*/
	v_dual_mov_b32 v138, v218 /*v474*/ :: v_dual_mov_b32 v139, v218 /*v474*/
	v_dual_mov_b32 v140, v218 /*v474*/ :: v_dual_mov_b32 v141, v218 /*v474*/
	v_dual_mov_b32 v142, v218 /*v474*/ :: v_dual_mov_b32 v143, v218 /*v474*/
	v_dual_mov_b32 v144, v218 /*v474*/ :: v_dual_mov_b32 v145, v218 /*v474*/
	v_dual_mov_b32 v130, v218 /*v474*/ :: v_dual_mov_b32 v131, v218 /*v474*/
	v_dual_mov_b32 v132, v218 /*v474*/ :: v_dual_mov_b32 v133, v218 /*v474*/
	v_dual_mov_b32 v134, v218 /*v474*/ :: v_dual_mov_b32 v135, v218 /*v474*/
	v_dual_mov_b32 v136, v218 /*v474*/ :: v_dual_mov_b32 v137, v218 /*v474*/
	v_dual_mov_b32 v122, v218 /*v474*/ :: v_dual_mov_b32 v123, v218 /*v474*/
	v_dual_mov_b32 v124, v218 /*v474*/ :: v_dual_mov_b32 v125, v218 /*v474*/
	v_dual_mov_b32 v126, v218 /*v474*/ :: v_dual_mov_b32 v127, v218 /*v474*/
	v_dual_mov_b32 v128, v218 /*v474*/ :: v_dual_mov_b32 v129, v218 /*v474*/
	v_dual_mov_b32 v114, v218 /*v474*/ :: v_dual_mov_b32 v115, v218 /*v474*/
	v_dual_mov_b32 v116, v218 /*v474*/ :: v_dual_mov_b32 v117, v218 /*v474*/
	v_dual_mov_b32 v118, v218 /*v474*/ :: v_dual_mov_b32 v119, v218 /*v474*/
	v_dual_mov_b32 v120, v218 /*v474*/ :: v_dual_mov_b32 v121, v218 /*v474*/
	v_dual_mov_b32 v106, v218 /*v474*/ :: v_dual_mov_b32 v107, v218 /*v474*/
	v_dual_mov_b32 v108, v218 /*v474*/ :: v_dual_mov_b32 v109, v218 /*v474*/
	v_dual_mov_b32 v110, v218 /*v474*/ :: v_dual_mov_b32 v111, v218 /*v474*/
	v_dual_mov_b32 v112, v218 /*v474*/ :: v_dual_mov_b32 v113, v218 /*v474*/
	v_dual_mov_b32 v98, v218 /*v474*/ :: v_dual_mov_b32 v99, v218 /*v474*/
	v_dual_mov_b32 v100, v218 /*v474*/ :: v_dual_mov_b32 v101, v218 /*v474*/
	v_dual_mov_b32 v102, v218 /*v474*/ :: v_dual_mov_b32 v103, v218 /*v474*/
	v_dual_mov_b32 v104, v218 /*v474*/ :: v_dual_mov_b32 v105, v218 /*v474*/
	v_dual_mov_b32 v90, v218 /*v474*/ :: v_dual_mov_b32 v91, v218 /*v474*/
	v_dual_mov_b32 v92, v218 /*v474*/ :: v_dual_mov_b32 v93, v218 /*v474*/
	v_dual_mov_b32 v94, v218 /*v474*/ :: v_dual_mov_b32 v95, v218 /*v474*/
	v_dual_mov_b32 v96, v218 /*v474*/ :: v_dual_mov_b32 v97, v218 /*v474*/
	v_dual_mov_b32 v82, v218 /*v474*/ :: v_dual_mov_b32 v83, v218 /*v474*/
	v_dual_mov_b32 v84, v218 /*v474*/ :: v_dual_mov_b32 v85, v218 /*v474*/
	v_dual_mov_b32 v86, v218 /*v474*/ :: v_dual_mov_b32 v87, v218 /*v474*/
	v_dual_mov_b32 v88, v218 /*v474*/ :: v_dual_mov_b32 v89, v218 /*v474*/
	v_dual_mov_b32 v74, v218 /*v474*/ :: v_dual_mov_b32 v75, v218 /*v474*/
	v_dual_mov_b32 v76, v218 /*v474*/ :: v_dual_mov_b32 v77, v218 /*v474*/
	v_dual_mov_b32 v78, v218 /*v474*/ :: v_dual_mov_b32 v79, v218 /*v474*/
	v_dual_mov_b32 v80, v218 /*v474*/ :: v_dual_mov_b32 v81, v218 /*v474*/
	v_dual_mov_b32 v66, v218 /*v474*/ :: v_dual_mov_b32 v67, v218 /*v474*/
	v_dual_mov_b32 v68, v218 /*v474*/ :: v_dual_mov_b32 v69, v218 /*v474*/
	v_dual_mov_b32 v70, v218 /*v474*/ :: v_dual_mov_b32 v71, v218 /*v474*/
	v_dual_mov_b32 v72, v218 /*v474*/ :: v_dual_mov_b32 v73, v218 /*v474*/
	v_dual_mov_b32 v58, v218 /*v474*/ :: v_dual_mov_b32 v59, v218 /*v474*/
	v_dual_mov_b32 v60, v218 /*v474*/ :: v_dual_mov_b32 v61, v218 /*v474*/
	v_dual_mov_b32 v62, v218 /*v474*/ :: v_dual_mov_b32 v63, v218 /*v474*/
	v_dual_mov_b32 v64, v218 /*v474*/ :: v_dual_mov_b32 v65, v218 /*v474*/
	v_dual_mov_b32 v50, v218 /*v474*/ :: v_dual_mov_b32 v51, v218 /*v474*/
	v_dual_mov_b32 v52, v218 /*v474*/ :: v_dual_mov_b32 v53, v218 /*v474*/
	v_dual_mov_b32 v54, v218 /*v474*/ :: v_dual_mov_b32 v55, v218 /*v474*/
	v_dual_mov_b32 v56, v218 /*v474*/ :: v_dual_mov_b32 v57, v218 /*v474*/
	v_dual_mov_b32 v42, v218 /*v474*/ :: v_dual_mov_b32 v43, v218 /*v474*/
	v_dual_mov_b32 v44, v218 /*v474*/ :: v_dual_mov_b32 v45, v218 /*v474*/
	v_dual_mov_b32 v46, v218 /*v474*/ :: v_dual_mov_b32 v47, v218 /*v474*/
	v_dual_mov_b32 v48, v218 /*v474*/ :: v_dual_mov_b32 v49, v218 /*v474*/
	v_dual_mov_b32 v34, v218 /*v474*/ :: v_dual_mov_b32 v35, v218 /*v474*/
	v_dual_mov_b32 v36, v218 /*v474*/ :: v_dual_mov_b32 v37, v218 /*v474*/
	v_dual_mov_b32 v38, v218 /*v474*/ :: v_dual_mov_b32 v39, v218 /*v474*/
	v_dual_mov_b32 v40, v218 /*v474*/ :: v_dual_mov_b32 v41, v218 /*v474*/
	v_dual_mov_b32 v26, v218 /*v474*/ :: v_dual_mov_b32 v27, v218 /*v474*/
	v_dual_mov_b32 v28, v218 /*v474*/ :: v_dual_mov_b32 v29, v218 /*v474*/
	v_dual_mov_b32 v30, v218 /*v474*/ :: v_dual_mov_b32 v31, v218 /*v474*/
	v_dual_mov_b32 v32, v218 /*v474*/ :: v_dual_mov_b32 v33, v218 /*v474*/
	v_dual_mov_b32 v18, v218 /*v474*/ :: v_dual_mov_b32 v19, v218 /*v474*/
	v_dual_mov_b32 v20, v218 /*v474*/ :: v_dual_mov_b32 v21, v218 /*v474*/
	v_dual_mov_b32 v22, v218 /*v474*/ :: v_dual_mov_b32 v23, v218 /*v474*/
	v_dual_mov_b32 v24, v218 /*v474*/ :: v_dual_mov_b32 v25, v218 /*v474*/
	v_dual_mov_b32 v10, v218 /*v474*/ :: v_dual_mov_b32 v11, v218 /*v474*/
	v_dual_mov_b32 v12, v218 /*v474*/ :: v_dual_mov_b32 v13, v218 /*v474*/
	v_dual_mov_b32 v14, v218 /*v474*/ :: v_dual_mov_b32 v15, v218 /*v474*/
	v_dual_mov_b32 v16, v218 /*v474*/ :: v_dual_mov_b32 v17, v218 /*v474*/
	v_dual_mov_b32 v2, v218 /*v474*/ :: v_dual_mov_b32 v3, v218 /*v474*/
	v_dual_mov_b32 v4, v218 /*v474*/ :: v_dual_mov_b32 v5, v218 /*v474*/
	v_dual_mov_b32 v6, v218 /*v474*/ :: v_dual_mov_b32 v7, v218 /*v474*/
	v_dual_mov_b32 v8, v218 /*v474*/ :: v_dual_mov_b32 v9, v218 /*v474*/
	s_mov_b32 s65, s64
	s_lshl_b32 s9, s68, 13
	s_mov_b32 s4, 0x3fb8aa3b
	s_mov_b64 s[6:7], s[2:3]
	s_set_vgpr_msb 0x100
.LBB0_2:
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x8a
	v_or_b32_e32 v2 /*v514*/, 32, v22 /*v534*/
	v_or_b32_e32 v5 /*v517*/, 64, v22 /*v534*/
	v_or_b32_e32 v9 /*v521*/, 0x80, v22 /*v534*/
	v_or_b32_e32 v3 /*v515*/, 0xe0, v22 /*v534*/
	v_or_b32_e32 v6 /*v518*/, 0x60, v22 /*v534*/
	v_or_b32_e32 v10 /*v522*/, 0xa0, v22 /*v534*/
	v_or_b32_e32 v4 /*v516*/, 32, v23 /*v535*/
	v_or_b32_e32 v7 /*v519*/, 64, v23 /*v535*/
	v_or_b32_e32 v11 /*v523*/, 0x80, v23 /*v535*/
	s_clause 0x4
	buffer_load_b128 v[32:35] /*v[544:547]*/, v22 /*v534*/, s[72:75], null offen
	buffer_load_b128 v[40:43] /*v[552:555]*/, v23 /*v535*/, s[72:75], null offen
	buffer_load_b128 v[36:39] /*v[548:551]*/, v2 /*v514*/, s[72:75], null offen
	buffer_load_b128 v[52:55] /*v[564:567]*/, v3 /*v515*/, s[72:75], null offen
	buffer_load_b128 v[44:47] /*v[556:559]*/, v4 /*v516*/, s[72:75], null offen
	v_or_b32_e32 v8 /*v520*/, 0x60, v23 /*v535*/
	v_or_b32_e32 v12 /*v524*/, 0xa0, v23 /*v535*/
	s_clause 0x3
	buffer_load_b128 v[72:75] /*v[584:587]*/, v9 /*v521*/, s[72:75], null offen
	buffer_load_b128 v[76:79] /*v[588:591]*/, v10 /*v522*/, s[72:75], null offen
	buffer_load_b128 v[116:119] /*v[628:631]*/, v11 /*v523*/, s[72:75], null offen
	buffer_load_b128 v[120:123] /*v[632:635]*/, v12 /*v524*/, s[72:75], null offen
	s_clause 0x4
	buffer_load_b128 v[124:127] /*v[636:639]*/, v22 /*v534*/, s[76:79], null offen
	buffer_load_b128 v[132:135] /*v[644:647]*/, v23 /*v535*/, s[76:79], null offen
	buffer_load_b128 v[128:131] /*v[640:643]*/, v2 /*v514*/, s[76:79], null offen
	buffer_load_b128 v[144:147] /*v[656:659]*/, v3 /*v515*/, s[76:79], null offen
	buffer_load_b128 v[136:139] /*v[648:651]*/, v4 /*v516*/, s[76:79], null offen
	s_wait_xcnt 0x2
	v_or_b32_e32 v2 /*v514*/, 0xc0, v22 /*v534*/
	v_add_nc_u32_e32 v22 /*v534*/, s9, v22 /*v534*/
	s_clause 0x3
	buffer_load_b128 v[56:59] /*v[568:571]*/, v5 /*v517*/, s[72:75], null offen
	buffer_load_b128 v[60:63] /*v[572:575]*/, v6 /*v518*/, s[72:75], null offen
	buffer_load_b128 v[64:67] /*v[576:579]*/, v7 /*v519*/, s[72:75], null offen
	buffer_load_b128 v[68:71] /*v[580:583]*/, v8 /*v520*/, s[72:75], null offen
	s_wait_xcnt 0x5
	v_or_b32_e32 v3 /*v515*/, 0xc0, v23 /*v535*/
	s_wait_xcnt 0x4
	v_or_b32_e32 v4 /*v516*/, 0xe0, v23 /*v535*/
	s_clause 0x2
	buffer_load_b128 v[48:51] /*v[560:563]*/, v2 /*v514*/, s[72:75], null offen
	buffer_load_b128 v[148:151] /*v[660:663]*/, v3 /*v515*/, s[72:75], null offen
	buffer_load_b128 v[152:155] /*v[664:667]*/, v4 /*v516*/, s[72:75], null offen
	s_clause 0xa
	buffer_load_b128 v[156:159] /*v[668:671]*/, v5 /*v517*/, s[76:79], null offen
	buffer_load_b128 v[160:163] /*v[672:675]*/, v6 /*v518*/, s[76:79], null offen
	buffer_load_b128 v[164:167] /*v[676:679]*/, v7 /*v519*/, s[76:79], null offen
	buffer_load_b128 v[168:171] /*v[680:683]*/, v8 /*v520*/, s[76:79], null offen
	buffer_load_b128 v[172:175] /*v[684:687]*/, v9 /*v521*/, s[76:79], null offen
	buffer_load_b128 v[176:179] /*v[688:691]*/, v10 /*v522*/, s[76:79], null offen
	buffer_load_b128 v[180:183] /*v[692:695]*/, v11 /*v523*/, s[76:79], null offen
	buffer_load_b128 v[184:187] /*v[696:699]*/, v12 /*v524*/, s[76:79], null offen
	buffer_load_b128 v[140:143] /*v[652:655]*/, v2 /*v514*/, s[76:79], null offen
	buffer_load_b128 v[188:191] /*v[700:703]*/, v3 /*v515*/, s[76:79], null offen
	buffer_load_b128 v[192:195] /*v[704:707]*/, v4 /*v516*/, s[76:79], null offen
	v_mov_b64_e32 v[14:15] /*v[526:527]*/, s[64:65]
	s_wait_xcnt 0x2
	v_dual_add_nc_u32 v2 /*v514*/, v19 /*v531*/, v20 /*v532*/ :: v_dual_add_nc_u32 v23 /*v535*/, s9, v23 /*v535*/
	s_add_nc_u64 s[6:7], s[6:7], -1
	s_set_vgpr_msb 0x8a82
	s_wait_loadcnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[196:203] /*v[708:715]*/, v[32:39] /*v[544:551]*/, v[242:249], 0
	s_set_vgpr_msb 0x820a
	ds_store_b128 v113 /*v625*/, v[32:35] /*v[544:547]*/
	ds_store_b128 v113 /*v625*/, v[36:39] /*v[548:551]*/ offset:32
	s_wait_loadcnt 0x11
	ds_store_b128 v113 /*v625*/, v[56:59] /*v[568:571]*/ offset:64
	s_wait_loadcnt 0x10
	ds_store_b128 v113 /*v625*/, v[60:63] /*v[572:575]*/ offset:96
	ds_store_b128 v113 /*v625*/, v[72:75] /*v[584:587]*/ offset:128
	ds_store_b128 v113 /*v625*/, v[76:79] /*v[588:591]*/ offset:160
	s_wait_loadcnt 0xd
	ds_store_b128 v113 /*v625*/, v[48:51] /*v[560:563]*/ offset:192
	ds_store_b128 v113 /*v625*/, v[52:55] /*v[564:567]*/ offset:224
	s_set_vgpr_msb 0xa86
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[716:723]*/, v[124:131] /*v[636:643]*/, v[82:89] /*v[338:345]*/, 0
	s_set_vgpr_msb 0x860a
	ds_store_b128 v114 /*v626*/, v[40:43] /*v[552:555]*/
	ds_store_b128 v114 /*v626*/, v[44:47] /*v[556:559]*/ offset:32
	ds_store_b128 v114 /*v626*/, v[64:67] /*v[576:579]*/ offset:64
	ds_store_b128 v114 /*v626*/, v[68:71] /*v[580:583]*/ offset:96
	ds_store_b128 v114 /*v626*/, v[116:119] /*v[628:631]*/ offset:128
	ds_store_b128 v114 /*v626*/, v[120:123] /*v[632:635]*/ offset:160
	s_wait_loadcnt 0xc
	ds_store_b128 v114 /*v626*/, v[148:151] /*v[660:663]*/ offset:192
	s_wait_loadcnt 0xb
	ds_store_b128 v114 /*v626*/, v[152:155] /*v[664:667]*/ offset:224
	s_cmp_lg_u64 s[6:7], 0
	s_set_vgpr_msb 0xa86
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[724:731]*/, v[32:39] /*v[544:551]*/, v[50:57] /*v[306:313]*/, 0
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v2 /*v514*/
	ds_load_tr16_b128 v[10:13] /*v[522:525]*/, v2 /*v514*/ offset:4352
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v24 /*v536*/
	v_wmma_f32_16x16x32_bf16 v[220:227] /*v[732:739]*/, v[32:39] /*v[544:551]*/, v[122:129] /*v[378:385]*/, 0
	v_wmma_f32_16x16x32_bf16 v[228:235] /*v[740:747]*/, v[124:131] /*v[636:643]*/, v[18:25] /*v[274:281]*/, 0
	v_wmma_f32_16x16x32_bf16 v[236:243] /*v[748:755]*/, v[124:131] /*v[636:643]*/, v[154:161] /*v[410:417]*/, 0
	v_wmma_f32_16x16x32_bf16 v[244:251] /*v[756:763]*/, v[32:39] /*v[544:551]*/, v[186:193] /*v[442:449]*/, 0
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[124:131] /*v[636:643]*/, v[226:233] /*v[482:489]*/, 0
	s_set_vgpr_msb 0x8682
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[40:47] /*v[552:559]*/, v[242:249], 0
	s_set_vgpr_msb 0x82c6
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[772:779]*/, v[40:47] /*v[552:559]*/, v[50:57] /*v[306:313]*/, 0
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[788:795]*/, v[40:47] /*v[552:559]*/, v[122:129] /*v[378:385]*/, 0
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[804:811]*/, v[40:47] /*v[552:559]*/, v[186:193] /*v[442:449]*/, 0
	s_set_vgpr_msb 0xc6a2
	v_wmma_f32_16x16x32_bf16 v[196:203] /*v[708:715]*/, v[56:63] /*v[568:575]*/, v[250:257], v[196:203] /*v[708:715]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[724:731]*/, v[56:63] /*v[568:575]*/, v[58:65] /*v[314:321]*/, v[212:219] /*v[724:731]*/
	v_wmma_f32_16x16x32_bf16 v[220:227] /*v[732:739]*/, v[56:63] /*v[568:575]*/, v[130:137] /*v[386:393]*/, v[220:227] /*v[732:739]*/
	v_wmma_f32_16x16x32_bf16 v[244:251] /*v[756:763]*/, v[56:63] /*v[568:575]*/, v[194:201] /*v[450:457]*/, v[244:251] /*v[756:763]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[64:71] /*v[576:583]*/, v[250:257], v[124:131] /*v[636:643]*/
	s_set_vgpr_msb 0xa2f6
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[772:779]*/, v[64:71] /*v[576:583]*/, v[58:65] /*v[314:321]*/, v[4:11] /*v[772:779]*/
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[788:795]*/, v[64:71] /*v[576:583]*/, v[130:137] /*v[386:393]*/, v[20:27] /*v[788:795]*/
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[804:811]*/, v[64:71] /*v[576:583]*/, v[194:201] /*v[450:457]*/, v[36:43] /*v[804:811]*/
	s_set_vgpr_msb 0xf6a6
	v_wmma_f32_16x16x32_bf16 v[196:203] /*v[708:715]*/, v[72:79] /*v[584:591]*/, v[2:9] /*v[258:265]*/, v[196:203] /*v[708:715]*/
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[716:723]*/, v[156:163] /*v[668:675]*/, v[90:97] /*v[346:353]*/, v[204:211] /*v[716:723]*/
	v_wmma_f32_16x16x32_bf16 v[252:259] /*v[764:771]*/, v[132:139] /*v[644:651]*/, v[18:25] /*v[274:281]*/, 0
	s_set_vgpr_msb 0xa6c6
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[780:787]*/, v[132:139] /*v[644:651]*/, v[82:89] /*v[338:345]*/, 0
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[796:803]*/, v[132:139] /*v[644:651]*/, v[154:161] /*v[410:417]*/, 0
	s_set_vgpr_msb 0xc6a6
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[132:139] /*v[644:651]*/, v[226:233] /*v[482:489]*/, 0
	v_wmma_f32_16x16x32_bf16 v[228:235] /*v[740:747]*/, v[156:163] /*v[668:675]*/, v[26:33] /*v[282:289]*/, v[228:235] /*v[740:747]*/
	v_wmma_f32_16x16x32_bf16 v[236:243] /*v[748:755]*/, v[156:163] /*v[668:675]*/, v[162:169] /*v[418:425]*/, v[236:243] /*v[748:755]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[156:163] /*v[668:675]*/, v[234:241] /*v[490:497]*/, v[32:39] /*v[544:551]*/
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[724:731]*/, v[72:79] /*v[584:591]*/, v[66:73] /*v[322:329]*/, v[212:219] /*v[724:731]*/
	v_wmma_f32_16x16x32_bf16 v[220:227] /*v[732:739]*/, v[72:79] /*v[584:591]*/, v[138:145] /*v[394:401]*/, v[220:227] /*v[732:739]*/
	v_wmma_f32_16x16x32_bf16 v[244:251] /*v[756:763]*/, v[72:79] /*v[584:591]*/, v[202:209] /*v[458:465]*/, v[244:251] /*v[756:763]*/
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[116:123] /*v[628:635]*/, v[2:9] /*v[258:265]*/, v[124:131] /*v[636:643]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[772:779]*/, v[116:123] /*v[628:635]*/, v[66:73] /*v[322:329]*/, v[4:11] /*v[772:779]*/
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[788:795]*/, v[116:123] /*v[628:635]*/, v[138:145] /*v[394:401]*/, v[20:27] /*v[788:795]*/
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[804:811]*/, v[116:123] /*v[628:635]*/, v[202:209] /*v[458:465]*/, v[36:43] /*v[804:811]*/
	s_set_vgpr_msb 0xf6a6
	v_wmma_f32_16x16x32_bf16 v[196:203] /*v[708:715]*/, v[48:55] /*v[560:567]*/, v[10:17] /*v[266:273]*/, v[196:203] /*v[708:715]*/
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[716:723]*/, v[172:179] /*v[684:691]*/, v[106:113] /*v[362:369]*/, v[204:211] /*v[716:723]*/
	v_wmma_f32_16x16x32_bf16 v[252:259] /*v[764:771]*/, v[164:171] /*v[676:683]*/, v[26:33] /*v[282:289]*/, v[252:259] /*v[764:771]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[780:787]*/, v[164:171] /*v[676:683]*/, v[90:97] /*v[346:353]*/, v[12:19] /*v[780:787]*/
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[796:803]*/, v[164:171] /*v[676:683]*/, v[162:169] /*v[418:425]*/, v[28:35] /*v[796:803]*/
	s_set_vgpr_msb 0xf6a6
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[164:171] /*v[676:683]*/, v[234:241] /*v[490:497]*/, v[40:47] /*v[552:559]*/
	v_wmma_f32_16x16x32_bf16 v[228:235] /*v[740:747]*/, v[172:179] /*v[684:691]*/, v[34:41] /*v[290:297]*/, v[228:235] /*v[740:747]*/
	v_wmma_f32_16x16x32_bf16 v[236:243] /*v[748:755]*/, v[172:179] /*v[684:691]*/, v[170:177] /*v[426:433]*/, v[236:243] /*v[748:755]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[172:179] /*v[684:691]*/, v[242:249] /*v[498:505]*/, v[32:39] /*v[544:551]*/
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[724:731]*/, v[48:55] /*v[560:567]*/, v[74:81] /*v[330:337]*/, v[212:219] /*v[724:731]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa68a
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[14:15] /*v[526:527]*/, v[212:213] /*v[724:725]*/
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[220:227] /*v[732:739]*/, v[48:55] /*v[560:567]*/, v[146:153] /*v[402:409]*/, v[220:227] /*v[732:739]*/
	s_set_vgpr_msb 0xa68a
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[14:15] /*v[526:527]*/, v[214:215] /*v[726:727]*/
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[14:15] /*v[526:527]*/, v[216:217] /*v[728:729]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[14:15] /*v[526:527]*/, v[218:219] /*v[730:731]*/
	v_pk_add_f32 v[64:65] /*v[576:577]*/, v[64:65] /*v[576:577]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[244:251] /*v[756:763]*/, v[48:55] /*v[560:567]*/, v[210:217] /*v[466:473]*/, v[244:251] /*v[756:763]*/
	s_set_vgpr_msb 0xa68a
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[14:15] /*v[526:527]*/, v[220:221] /*v[732:733]*/
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[14:15] /*v[526:527]*/, v[222:223] /*v[734:735]*/
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[14:15] /*v[526:527]*/, v[224:225] /*v[736:737]*/
	v_pk_mul_f32 v[102:103] /*v[614:615]*/, v[14:15] /*v[526:527]*/, v[226:227] /*v[738:739]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[14:15] /*v[526:527]*/, v[196:197] /*v[708:709]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[14:15] /*v[526:527]*/, v[198:199] /*v[710:711]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[14:15] /*v[526:527]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[148:155] /*v[660:667]*/, v[10:17] /*v[266:273]*/, v[124:131] /*v[636:643]*/
	s_set_vgpr_msb 0xa68a
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[14:15] /*v[526:527]*/, v[202:203] /*v[714:715]*/
	v_pk_mul_f32 v[118:119] /*v[630:631]*/, v[14:15] /*v[526:527]*/, v[244:245] /*v[756:757]*/
	v_pk_mul_f32 v[120:121] /*v[632:633]*/, v[14:15] /*v[526:527]*/, v[246:247] /*v[758:759]*/
	v_pk_mul_f32 v[122:123] /*v[634:635]*/, v[14:15] /*v[526:527]*/, v[248:249] /*v[760:761]*/
	v_pk_mul_f32 v[132:133] /*v[644:645]*/, v[14:15] /*v[526:527]*/, v[250:251] /*v[762:763]*/
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[772:779]*/, v[148:155] /*v[660:667]*/, v[74:81] /*v[330:337]*/, v[4:11] /*v[772:779]*/
	s_set_vgpr_msb 0xf68a
	v_pk_mul_f32 v[124:125] /*v[636:637]*/, v[14:15] /*v[526:527]*/, v[124:125] /*v[636:637]*/
	v_pk_mul_f32 v[126:127] /*v[638:639]*/, v[14:15] /*v[526:527]*/, v[126:127] /*v[638:639]*/
	v_pk_mul_f32 v[128:129] /*v[640:641]*/, v[14:15] /*v[526:527]*/, v[128:129] /*v[640:641]*/
	v_pk_mul_f32 v[130:131] /*v[642:643]*/, v[14:15] /*v[526:527]*/, v[130:131] /*v[642:643]*/
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[70:71] /*v[582:583]*/, v[70:71] /*v[582:583]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[20:27] /*v[788:795]*/, v[148:155] /*v[660:667]*/, v[146:153] /*v[402:409]*/, v[20:27] /*v[788:795]*/
	s_set_vgpr_msb 0xf68a
	v_pk_add_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[82:83] /*v[594:595]*/, v[82:83] /*v[594:595]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[102:103] /*v[614:615]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[118:119] /*v[630:631]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[120:121] /*v[632:633]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[122:123] /*v[634:635]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[36:43] /*v[804:811]*/, v[148:155] /*v[660:667]*/, v[210:217] /*v[466:473]*/, v[36:43] /*v[804:811]*/
	s_set_vgpr_msb 0xf68e
	v_pk_mul_f32 v[158:159] /*v[670:671]*/, v[14:15] /*v[526:527]*/, v[20:21] /*v[788:789]*/
	v_pk_mul_f32 v[160:161] /*v[672:673]*/, v[14:15] /*v[526:527]*/, v[22:23] /*v[790:791]*/
	v_pk_mul_f32 v[162:163] /*v[674:675]*/, v[14:15] /*v[526:527]*/, v[24:25] /*v[792:793]*/
	v_pk_mul_f32 v[164:165] /*v[676:677]*/, v[14:15] /*v[526:527]*/, v[26:27] /*v[794:795]*/
	v_pk_mul_f32 v[148:149] /*v[660:661]*/, v[14:15] /*v[526:527]*/, v[10:11] /*v[778:779]*/
	s_set_vgpr_msb 0x8e8a
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[132:133] /*v[644:645]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[124:125] /*v[636:637]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[716:723]*/, v[140:147] /*v[652:659]*/, v[114:121] /*v[370:377]*/, v[204:211] /*v[716:723]*/
	s_set_vgpr_msb 0xa68e
	v_pk_mul_f32 v[174:175] /*v[686:687]*/, v[14:15] /*v[526:527]*/, v[36:37] /*v[804:805]*/
	v_pk_mul_f32 v[176:177] /*v[688:689]*/, v[14:15] /*v[526:527]*/, v[38:39] /*v[806:807]*/
	v_pk_mul_f32 v[178:179] /*v[690:691]*/, v[14:15] /*v[526:527]*/, v[40:41] /*v[808:809]*/
	s_set_vgpr_msb 0x8e8a
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[126:127] /*v[638:639]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[128:129] /*v[640:641]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[130:131] /*v[642:643]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[148:149] /*v[660:661]*/, v[148:149] /*v[660:661]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[252:259] /*v[764:771]*/, v[180:187] /*v[692:699]*/, v[34:41] /*v[290:297]*/, v[252:259] /*v[764:771]*/
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[158:159] /*v[670:671]*/, v[158:159] /*v[670:671]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[160:161] /*v[672:673]*/, v[160:161] /*v[672:673]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[162:163] /*v[674:675]*/, v[162:163] /*v[674:675]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[164:165] /*v[676:677]*/, v[164:165] /*v[676:677]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[174:175] /*v[686:687]*/, v[174:175] /*v[686:687]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[176:177] /*v[688:689]*/, v[176:177] /*v[688:689]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[178:179] /*v[690:691]*/, v[178:179] /*v[690:691]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[780:787]*/, v[180:187] /*v[692:699]*/, v[106:113] /*v[362:369]*/, v[12:19] /*v[780:787]*/
	s_set_vgpr_msb 0xf682
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[64:65] /*v[576:577]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x82f6
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[796:803]*/, v[180:187] /*v[692:699]*/, v[170:177] /*v[426:433]*/, v[28:35] /*v[796:803]*/
	s_set_vgpr_msb 0xf6a6
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[70:71] /*v[582:583]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[82:83] /*v[594:595]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103] /*v[614:615]*/, v[102:103] /*v[614:615]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119] /*v[630:631]*/, v[118:119] /*v[630:631]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121] /*v[632:633]*/, v[120:121] /*v[632:633]*/, s[4:5] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[180:187] /*v[692:699]*/, v[242:249] /*v[498:505]*/, v[40:47] /*v[552:559]*/
	v_pk_mul_f32 v[122:123] /*v[634:635]*/, v[122:123] /*v[634:635]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133] /*v[644:645]*/, v[132:133] /*v[644:645]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125] /*v[636:637]*/, v[124:125] /*v[636:637]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127] /*v[638:639]*/, v[126:127] /*v[638:639]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xa68e
	v_pk_mul_f32 v[180:181] /*v[692:693]*/, v[14:15] /*v[526:527]*/, v[42:43] /*v[810:811]*/
	v_pk_mul_f32 v[128:129] /*v[640:641]*/, v[128:129] /*v[640:641]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131] /*v[642:643]*/, v[130:131] /*v[642:643]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[228:235] /*v[740:747]*/, v[140:147] /*v[652:659]*/, v[42:49] /*v[298:305]*/, v[228:235] /*v[740:747]*/
	v_pk_mul_f32 v[148:149] /*v[660:661]*/, v[148:149] /*v[660:661]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[180:181] /*v[692:693]*/, v[180:181] /*v[692:693]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[158:159] /*v[670:671]*/, v[158:159] /*v[670:671]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161] /*v[672:673]*/, v[160:161] /*v[672:673]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163] /*v[674:675]*/, v[162:163] /*v[674:675]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165] /*v[676:677]*/, v[164:165] /*v[676:677]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175] /*v[686:687]*/, v[174:175] /*v[686:687]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[236:243] /*v[748:755]*/, v[140:147] /*v[652:659]*/, v[178:185] /*v[434:441]*/, v[236:243] /*v[748:755]*/
	v_pk_mul_f32 v[176:177] /*v[688:689]*/, v[176:177] /*v[688:689]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[690:691]*/, v[178:179] /*v[690:691]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[692:693]*/, v[180:181] /*v[692:693]*/, s[4:5] op_sel_hi:[1,0]
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v49 /*v561*/, v49 /*v561*/
	v_exp_f32_e32 v50 /*v562*/, v50 /*v562*/
	v_exp_f32_e32 v51 /*v563*/, v51 /*v563*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[140:147] /*v[652:659]*/, v[250:257] /*v[506:513]*/, v[32:39] /*v[544:551]*/
	v_exp_f32_e32 v52 /*v564*/, v52 /*v564*/
	v_exp_f32_e32 v53 /*v565*/, v53 /*v565*/
	v_exp_f32_e32 v54 /*v566*/, v54 /*v566*/
	v_exp_f32_e32 v55 /*v567*/, v55 /*v567*/
	s_set_vgpr_msb 0xa68e
	v_pk_mul_f32 v[142:143] /*v[654:655]*/, v[14:15] /*v[526:527]*/, v[4:5] /*v[772:773]*/
	v_pk_mul_f32 v[144:145] /*v[656:657]*/, v[14:15] /*v[526:527]*/, v[6:7] /*v[774:775]*/
	v_pk_mul_f32 v[146:147] /*v[658:659]*/, v[14:15] /*v[526:527]*/, v[8:9] /*v[776:777]*/
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[252:259] /*v[764:771]*/, v[188:195] /*v[700:707]*/, v[42:49] /*v[298:305]*/, v[252:259] /*v[764:771]*/
	v_exp_f32_e32 v64 /*v576*/, v64 /*v576*/
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[142:143] /*v[654:655]*/, v[142:143] /*v[654:655]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[144:145] /*v[656:657]*/, v[144:145] /*v[656:657]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[146:147] /*v[658:659]*/, v[146:147] /*v[658:659]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v65 /*v577*/, v65 /*v577*/
	v_exp_f32_e32 v66 /*v578*/, v66 /*v578*/
	v_pk_mul_f32 v[142:143] /*v[654:655]*/, v[142:143] /*v[654:655]*/, s[4:5] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[780:787]*/, v[188:195] /*v[700:707]*/, v[114:121] /*v[370:377]*/, v[12:19] /*v[780:787]*/
	s_set_vgpr_msb 0xf682
	v_pk_mul_f32 v[144:145] /*v[656:657]*/, v[144:145] /*v[656:657]*/, s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147] /*v[658:659]*/, v[146:147] /*v[658:659]*/, s[4:5] op_sel_hi:[1,0]
	v_exp_f32_e32 v67 /*v579*/, v67 /*v579*/
	v_exp_f32_e32 v68 /*v580*/, v68 /*v580*/
	v_exp_f32_e32 v69 /*v581*/, v69 /*v581*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	v_exp_f32_e32 v71 /*v583*/, v71 /*v583*/
	s_set_vgpr_msb 0x82f6
	v_wmma_f32_16x16x32_bf16 v[28:35] /*v[796:803]*/, v[188:195] /*v[700:707]*/, v[178:185] /*v[434:441]*/, v[28:35] /*v[796:803]*/
	s_set_vgpr_msb 0xf6a6
	v_exp_f32_e32 v80 /*v592*/, v80 /*v592*/
	v_exp_f32_e32 v81 /*v593*/, v81 /*v593*/
	v_exp_f32_e32 v82 /*v594*/, v82 /*v594*/
	v_exp_f32_e32 v83 /*v595*/, v83 /*v595*/
	v_exp_f32_e32 v84 /*v596*/, v84 /*v596*/
	v_exp_f32_e32 v85 /*v597*/, v85 /*v597*/
	v_exp_f32_e32 v102 /*v614*/, v102 /*v614*/
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[188:195] /*v[700:707]*/, v[250:257] /*v[506:513]*/, v[40:47] /*v[552:559]*/
	v_exp_f32_e32 v103 /*v615*/, v103 /*v615*/
	v_exp_f32_e32 v118 /*v630*/, v118 /*v630*/
	v_exp_f32_e32 v119 /*v631*/, v119 /*v631*/
	v_exp_f32_e32 v120 /*v632*/, v120 /*v632*/
	v_exp_f32_e32 v121 /*v633*/, v121 /*v633*/
	v_exp_f32_e32 v122 /*v634*/, v122 /*v634*/
	v_exp_f32_e32 v123 /*v635*/, v123 /*v635*/
	v_exp_f32_e32 v132 /*v644*/, v132 /*v644*/
	v_exp_f32_e32 v133 /*v645*/, v133 /*v645*/
	v_exp_f32_e32 v124 /*v636*/, v124 /*v636*/
	v_exp_f32_e32 v125 /*v637*/, v125 /*v637*/
	v_exp_f32_e32 v126 /*v638*/, v126 /*v638*/
	v_exp_f32_e32 v127 /*v639*/, v127 /*v639*/
	v_exp_f32_e32 v128 /*v640*/, v128 /*v640*/
	v_exp_f32_e32 v129 /*v641*/, v129 /*v641*/
	v_exp_f32_e32 v130 /*v642*/, v130 /*v642*/
	v_exp_f32_e32 v131 /*v643*/, v131 /*v643*/
	v_exp_f32_e32 v142 /*v654*/, v142 /*v654*/
	v_exp_f32_e32 v143 /*v655*/, v143 /*v655*/
	v_exp_f32_e32 v144 /*v656*/, v144 /*v656*/
	v_exp_f32_e32 v145 /*v657*/, v145 /*v657*/
	v_exp_f32_e32 v146 /*v658*/, v146 /*v658*/
	v_exp_f32_e32 v147 /*v659*/, v147 /*v659*/
	v_exp_f32_e32 v148 /*v660*/, v148 /*v660*/
	v_exp_f32_e32 v149 /*v661*/, v149 /*v661*/
	v_exp_f32_e32 v158 /*v670*/, v158 /*v670*/
	v_exp_f32_e32 v159 /*v671*/, v159 /*v671*/
	v_exp_f32_e32 v160 /*v672*/, v160 /*v672*/
	v_exp_f32_e32 v161 /*v673*/, v161 /*v673*/
	v_exp_f32_e32 v162 /*v674*/, v162 /*v674*/
	v_exp_f32_e32 v163 /*v675*/, v163 /*v675*/
	v_exp_f32_e32 v164 /*v676*/, v164 /*v676*/
	v_exp_f32_e32 v165 /*v677*/, v165 /*v677*/
	v_exp_f32_e32 v174 /*v686*/, v174 /*v686*/
	v_exp_f32_e32 v175 /*v687*/, v175 /*v687*/
	v_exp_f32_e32 v176 /*v688*/, v176 /*v688*/
	v_exp_f32_e32 v177 /*v689*/, v177 /*v689*/
	v_exp_f32_e32 v178 /*v690*/, v178 /*v690*/
	v_exp_f32_e32 v179 /*v691*/, v179 /*v691*/
	v_exp_f32_e32 v180 /*v692*/, v180 /*v692*/
	v_exp_f32_e32 v181 /*v693*/, v181 /*v693*/
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[56:57] /*v[568:569]*/, v[228:229] /*v[740:741]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[58:59] /*v[570:571]*/, v[230:231] /*v[742:743]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[60:61] /*v[572:573]*/, v[232:233] /*v[744:745]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[62:63] /*v[574:575]*/, v[234:235] /*v[746:747]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[72:73] /*v[584:585]*/, v[204:205] /*v[716:717]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[74:75] /*v[586:587]*/, v[206:207] /*v[718:719]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[76:77] /*v[588:589]*/, v[208:209] /*v[720:721]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[78:79] /*v[590:591]*/, v[210:211] /*v[722:723]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[236:237] /*v[748:749]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[238:239] /*v[750:751]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[240:241] /*v[752:753]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[242:243] /*v[754:755]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[134:135] /*v[646:647]*/, v[252:253] /*v[764:765]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[136:137] /*v[648:649]*/, v[254:255] /*v[766:767]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a8b
	v_pk_add_f32 v[138:139] /*v[650:651]*/, v[0:1] /*v[768:769]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[140:141] /*v[652:653]*/, v[2:3] /*v[770:771]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[150:151] /*v[662:663]*/, v[12:13] /*v[780:781]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[152:153] /*v[664:665]*/, v[14:15] /*v[782:783]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[154:155] /*v[666:667]*/, v[16:17] /*v[784:785]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[156:157] /*v[668:669]*/, v[18:19] /*v[786:787]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[166:167] /*v[678:679]*/, v[28:29] /*v[796:797]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[168:169] /*v[680:681]*/, v[30:31] /*v[798:799]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[170:171] /*v[682:683]*/, v[32:33] /*v[800:801]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[172:173] /*v[684:685]*/, v[34:35] /*v[802:803]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8b8a
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[56:57] /*v[568:569]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[58:59] /*v[570:571]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[60:61] /*v[572:573]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[62:63] /*v[574:575]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[72:73] /*v[584:585]*/, v[64:65] /*v[576:577]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[74:75] /*v[586:587]*/, v[66:67] /*v[578:579]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[76:77] /*v[588:589]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[78:79] /*v[590:591]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[104:105] /*v[616:617]*/, v[80:81] /*v[592:593]*/
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[106:107] /*v[618:619]*/, v[82:83] /*v[594:595]*/
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[110:111] /*v[622:623]*/, v[84:85] /*v[596:597]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[116:117] /*v[628:629]*/, v[102:103] /*v[614:615]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, v[118:119] /*v[630:631]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[120:121] /*v[632:633]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[122:123] /*v[634:635]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[132:133] /*v[644:645]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[134:135] /*v[646:647]*/, v[124:125] /*v[636:637]*/
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[136:137] /*v[648:649]*/, v[126:127] /*v[638:639]*/
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[138:139] /*v[650:651]*/, v[128:129] /*v[640:641]*/
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[140:141] /*v[652:653]*/, v[130:131] /*v[642:643]*/
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[150:151] /*v[662:663]*/, v[142:143] /*v[654:655]*/
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[152:153] /*v[664:665]*/, v[144:145] /*v[656:657]*/
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[154:155] /*v[666:667]*/, v[146:147] /*v[658:659]*/
	v_pk_mul_f32 v[102:103] /*v[614:615]*/, v[156:157] /*v[668:669]*/, v[148:149] /*v[660:661]*/
	v_pk_mul_f32 v[104:105] /*v[616:617]*/, v[166:167] /*v[678:679]*/, v[158:159] /*v[670:671]*/
	v_pk_mul_f32 v[106:107] /*v[618:619]*/, v[168:169] /*v[680:681]*/, v[160:161] /*v[672:673]*/
	v_pk_mul_f32 v[110:111] /*v[622:623]*/, v[170:171] /*v[682:683]*/, v[162:163] /*v[674:675]*/
	v_pk_mul_f32 v[116:117] /*v[628:629]*/, v[172:173] /*v[684:685]*/, v[164:165] /*v[676:677]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[174:175] /*v[686:687]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, v[176:177] /*v[688:689]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[178:179] /*v[690:691]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, v[180:181] /*v[692:693]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[14:15] /*v[526:527]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[14:15] /*v[526:527]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[14:15] /*v[526:527]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[14:15] /*v[526:527]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[14:15] /*v[526:527]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[14:15] /*v[526:527]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[14:15] /*v[526:527]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[14:15] /*v[526:527]*/, v[62:63] /*v[574:575]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[14:15] /*v[526:527]*/, v[64:65] /*v[576:577]*/
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[14:15] /*v[526:527]*/, v[66:67] /*v[578:579]*/
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[14:15] /*v[526:527]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[14:15] /*v[526:527]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[118:119] /*v[630:631]*/, v[14:15] /*v[526:527]*/, v[32:33] /*v[544:545]*/
	v_pk_mul_f32 v[120:121] /*v[632:633]*/, v[14:15] /*v[526:527]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[122:123] /*v[634:635]*/, v[14:15] /*v[526:527]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[124:125] /*v[636:637]*/, v[14:15] /*v[526:527]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[14:15] /*v[526:527]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[14:15] /*v[526:527]*/, v[74:75] /*v[586:587]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[14:15] /*v[526:527]*/, v[76:77] /*v[588:589]*/
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[14:15] /*v[526:527]*/, v[78:79] /*v[590:591]*/
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[14:15] /*v[526:527]*/, v[80:81] /*v[592:593]*/
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[14:15] /*v[526:527]*/, v[82:83] /*v[594:595]*/
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[14:15] /*v[526:527]*/, v[84:85] /*v[596:597]*/
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[14:15] /*v[526:527]*/, v[102:103] /*v[614:615]*/
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[14:15] /*v[526:527]*/, v[104:105] /*v[616:617]*/
	v_pk_mul_f32 v[102:103] /*v[614:615]*/, v[14:15] /*v[526:527]*/, v[106:107] /*v[618:619]*/
	v_pk_mul_f32 v[104:105] /*v[616:617]*/, v[14:15] /*v[526:527]*/, v[110:111] /*v[622:623]*/
	v_pk_mul_f32 v[106:107] /*v[618:619]*/, v[14:15] /*v[526:527]*/, v[116:117] /*v[628:629]*/
	v_pk_mul_f32 v[110:111] /*v[622:623]*/, v[14:15] /*v[526:527]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[116:117] /*v[628:629]*/, v[14:15] /*v[526:527]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[126:127] /*v[638:639]*/, v[14:15] /*v[526:527]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, v[46:47] /*v[558:559]*/
	v_cvt_pk_bf16_f32 v32 /*v544*/, v48 /*v560*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v33 /*v545*/, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v34 /*v546*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v35 /*v547*/, v54 /*v566*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v40 /*v552*/, v56 /*v568*/, v57 /*v569*/
	v_cvt_pk_bf16_f32 v41 /*v553*/, v58 /*v570*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v42 /*v554*/, v60 /*v572*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v36 /*v548*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v37 /*v549*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v38 /*v550*/, v72 /*v584*/, v73 /*v585*/
	v_cvt_pk_bf16_f32 v39 /*v551*/, v74 /*v586*/, v75 /*v587*/
	v_cvt_pk_bf16_f32 v43 /*v555*/, v62 /*v574*/, v63 /*v575*/
	v_cvt_pk_bf16_f32 v48 /*v560*/, v64 /*v576*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v49 /*v561*/, v66 /*v578*/, v67 /*v579*/
	v_cvt_pk_bf16_f32 v44 /*v556*/, v76 /*v588*/, v77 /*v589*/
	v_cvt_pk_bf16_f32 v45 /*v557*/, v78 /*v590*/, v79 /*v591*/
	v_cvt_pk_bf16_f32 v46 /*v558*/, v80 /*v592*/, v81 /*v593*/
	v_cvt_pk_bf16_f32 v47 /*v559*/, v82 /*v594*/, v83 /*v595*/
	v_cvt_pk_bf16_f32 v50 /*v562*/, v68 /*v580*/, v69 /*v581*/
	v_cvt_pk_bf16_f32 v51 /*v563*/, v70 /*v582*/, v71 /*v583*/
	v_cvt_pk_bf16_f32 v56 /*v568*/, v118 /*v630*/, v119 /*v631*/
	v_cvt_pk_bf16_f32 v52 /*v564*/, v84 /*v596*/, v85 /*v597*/
	v_cvt_pk_bf16_f32 v53 /*v565*/, v102 /*v614*/, v103 /*v615*/
	v_cvt_pk_bf16_f32 v54 /*v566*/, v104 /*v616*/, v105 /*v617*/
	v_cvt_pk_bf16_f32 v55 /*v567*/, v106 /*v618*/, v107 /*v619*/
	v_cvt_pk_bf16_f32 v57 /*v569*/, v120 /*v632*/, v121 /*v633*/
	v_cvt_pk_bf16_f32 v58 /*v570*/, v122 /*v634*/, v123 /*v635*/
	v_cvt_pk_bf16_f32 v59 /*v571*/, v124 /*v636*/, v125 /*v637*/
	v_cvt_pk_bf16_f32 v60 /*v572*/, v110 /*v622*/, v111 /*v623*/
	v_cvt_pk_bf16_f32 v61 /*v573*/, v116 /*v628*/, v117 /*v629*/
	v_cvt_pk_bf16_f32 v62 /*v574*/, v126 /*v638*/, v127 /*v639*/
	v_cvt_pk_bf16_f32 v63 /*v575*/, v14 /*v526*/, v15 /*v527*/
	s_set_vgpr_msb 0x8a5a
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[32:39] /*v[544:551]*/, v[6:13] /*v[518:525]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5a82
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v25 /*v537*/
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v25 /*v537*/ offset:4352
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[186:193], v[40:47] /*v[552:559]*/, v[6:13] /*v[518:525]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[122:129], v[48:55] /*v[560:567]*/, v[6:13] /*v[518:525]*/, v[122:129]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[56:63] /*v[568:575]*/, v[6:13] /*v[518:525]*/, v[58:65]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v24 /*v536*/ offset:4352
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[32:39] /*v[544:551]*/, v[2:9] /*v[514:521]*/, v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[178:185], v[40:47] /*v[552:559]*/, v[2:9] /*v[514:521]*/, v[178:185]
	v_wmma_f32_16x16x32_bf16 v[114:121], v[48:55] /*v[560:567]*/, v[2:9] /*v[514:521]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[56:63] /*v[568:575]*/, v[2:9] /*v[514:521]*/, v[50:57]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v26 /*v538*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v26 /*v538*/ offset:4352
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[234:241], v[32:39] /*v[544:551]*/, v[64:71] /*v[576:583]*/, v[234:241]
	v_wmma_f32_16x16x32_bf16 v[170:177], v[40:47] /*v[552:559]*/, v[64:71] /*v[576:583]*/, v[170:177]
	v_wmma_f32_16x16x32_bf16 v[106:113], v[48:55] /*v[560:567]*/, v[64:71] /*v[576:583]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[56:63] /*v[568:575]*/, v[64:71] /*v[576:583]*/, v[42:49]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v27 /*v539*/
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v27 /*v539*/ offset:4352
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[226:233], v[32:39] /*v[544:551]*/, v[2:9] /*v[514:521]*/, v[226:233]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[40:47] /*v[552:559]*/, v[2:9] /*v[514:521]*/, v[162:169]
	v_wmma_f32_16x16x32_bf16 v[98:105], v[48:55] /*v[560:567]*/, v[2:9] /*v[514:521]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[56:63] /*v[568:575]*/, v[2:9] /*v[514:521]*/, v[34:41]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v28 /*v540*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v28 /*v540*/ offset:4352
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[218:225], v[32:39] /*v[544:551]*/, v[64:71] /*v[576:583]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[154:161], v[40:47] /*v[552:559]*/, v[64:71] /*v[576:583]*/, v[154:161]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[48:55] /*v[560:567]*/, v[64:71] /*v[576:583]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[56:63] /*v[568:575]*/, v[64:71] /*v[576:583]*/, v[26:33]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v29 /*v541*/
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v29 /*v541*/ offset:4352
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[210:217], v[32:39] /*v[544:551]*/, v[2:9] /*v[514:521]*/, v[210:217]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[40:47] /*v[552:559]*/, v[2:9] /*v[514:521]*/, v[146:153]
	v_wmma_f32_16x16x32_bf16 v[82:89], v[48:55] /*v[560:567]*/, v[2:9] /*v[514:521]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[56:63] /*v[568:575]*/, v[2:9] /*v[514:521]*/, v[18:25]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v30 /*v542*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v30 /*v542*/ offset:4352
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[202:209], v[32:39] /*v[544:551]*/, v[64:71] /*v[576:583]*/, v[202:209]
	v_wmma_f32_16x16x32_bf16 v[138:145], v[40:47] /*v[552:559]*/, v[64:71] /*v[576:583]*/, v[138:145]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[48:55] /*v[560:567]*/, v[64:71] /*v[576:583]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[56:63] /*v[568:575]*/, v[64:71] /*v[576:583]*/, v[10:17]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[32:39] /*v[544:551]*/, v[2:9] /*v[514:521]*/, v[194:201]
	v_wmma_f32_16x16x32_bf16 v[130:137], v[40:47] /*v[552:559]*/, v[2:9] /*v[514:521]*/, v[130:137]
	v_wmma_f32_16x16x32_bf16 v[66:73], v[48:55] /*v[560:567]*/, v[2:9] /*v[514:521]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[56:63] /*v[568:575]*/, v[2:9] /*v[514:521]*/, v[2:9]
	s_set_vgpr_msb 0xa00
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
	v_mov_b64_e32 v[192:193], v[8:9]
	v_mov_b64_e32 v[190:191], v[6:7]
	v_mov_b64_e32 v[188:189], v[4:5]
	v_mov_b64_e32 v[186:187], v[2:3]
	v_mov_b64_e32 v[200:201], v[8:9]
	v_mov_b64_e32 v[198:199], v[6:7]
	v_mov_b64_e32 v[196:197], v[4:5]
	v_mov_b64_e32 v[194:195], v[2:3]
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
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[8:9]
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[6:7]
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[4:5]
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[2:3]
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[8:9]
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[6:7]
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[4:5]
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[2:3]
	s_set_vgpr_msb 0x4000
.LBB0_4:
	s_sub_co_i32 s70, s17, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s70, 1
	s_cbranch_scc1 .LBB0_7
	s_set_vgpr_msb 0x82
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v101 /*v613*/, v100 /*v612*/ :: v_dual_add_nc_u32 v104 /*v616*/, s71, v1
	s_set_vgpr_msb 0x8208
	v_lshlrev_b32_e32 v1, 4, v109 /*v621*/
	v_nop
	v_nop
	s_set_vgpr_msb 0x888
	v_add_lshl_u32 v9 /*v521*/, s8, v109 /*v621*/, 4
	s_lshl_b32 s3, s8, 4
	s_lshl_b32 s4, s2, 9
	v_dual_add_nc_u32 v102 /*v614*/, s16, v109 /*v621*/ :: v_dual_add_nc_u32 v108 /*v620*/, s71, v18 /*v530*/
	s_set_vgpr_msb 0x8800
	v_add3_u32 v1, s3, s4, v1
	s_set_vgpr_msb 0x8a
	v_add3_u32 v9 /*v521*/, s4, v9 /*v521*/, 0x100
	v_dual_add_nc_u32 v106 /*v618*/, s71, v17 /*v529*/ :: v_dual_bitop2_b32 v3 /*v515*/, 64, v20 /*v532*/ bitop3:0x54
	v_dual_mov_b32 v99 /*v611*/, v98 /*v610*/ :: v_dual_bitop2_b32 v2 /*v514*/, 32, v21 /*v533*/ bitop3:0x54
	s_set_vgpr_msb 0x8a80
	v_mul_lo_u32 v10 /*v522*/, s68, v1
	s_set_vgpr_msb 0x808a
	v_mul_lo_u32 v9 /*v521*/, s68, v9 /*v521*/
	v_or_b32_e32 v4 /*v516*/, 0x60, v21 /*v533*/
	v_or_b32_e32 v5 /*v517*/, 0x80, v20 /*v532*/
	v_or_b32_e32 v6 /*v518*/, 0xa0, v21 /*v533*/
	v_or_b32_e32 v7 /*v519*/, 0xc0, v20 /*v532*/
	v_or_b32_e32 v8 /*v520*/, 0xe0, v21 /*v533*/
	s_mov_b32 s65, s64
	v_dual_mov_b32 v105 /*v617*/, v106 /*v618*/ :: v_dual_bitop2_b32 v10 /*v522*/, v16 /*v528*/, v10 /*v522*/ bitop3:0x54
	v_dual_add_nc_u32 v118 /*v630*/, v19 /*v531*/, v20 /*v532*/ :: v_dual_bitop2_b32 v9 /*v521*/, v16 /*v528*/, v9 /*v521*/ bitop3:0x54
	v_mov_b64_e32 v[110:111] /*v[622:623]*/, s[64:65]
	v_dual_mov_b32 v97 /*v609*/, v96 /*v608*/ :: v_dual_mov_b32 v95 /*v607*/, v94 /*v606*/
	v_dual_mov_b32 v87 /*v599*/, v86 /*v598*/ :: v_dual_mov_b32 v89 /*v601*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a02
	v_mov_b32_e32 v1, v102 /*v614*/
	s_set_vgpr_msb 0x2aa
	v_dual_mov_b32 v103 /*v615*/, v104 /*v616*/ :: v_dual_mov_b32 v91 /*v603*/, v90 /*v602*/
	v_dual_mov_b32 v93 /*v605*/, v92 /*v604*/ :: v_dual_mov_b32 v107 /*v619*/, v108 /*v620*/
	v_add_lshl_u32 v115 /*v627*/, v10 /*v522*/, s5, 4
	v_add_lshl_u32 v116 /*v628*/, v9 /*v521*/, s5, 4
	v_lshl_or_b32 v117 /*v629*/, s2, 5, v112 /*v624*/
	v_dual_add_nc_u32 v119 /*v631*/, v19 /*v531*/, v2 /*v514*/ :: v_dual_add_nc_u32 v120 /*v632*/, v19 /*v531*/, v3 /*v515*/
	v_dual_add_nc_u32 v121 /*v633*/, v19 /*v531*/, v4 /*v516*/ :: v_dual_add_nc_u32 v122 /*v634*/, v19 /*v531*/, v5 /*v517*/
	v_dual_add_nc_u32 v123 /*v635*/, v19 /*v531*/, v6 /*v518*/ :: v_dual_add_nc_u32 v124 /*v636*/, v19 /*v531*/, v7 /*v519*/
	v_add_nc_u32_e32 v125 /*v637*/, v19 /*v531*/, v8 /*v520*/
	s_ashr_i32 s71, s70, 31
	s_lshl_b32 s68, s68, 13
	s_mov_b32 s66, 0x3fb8aa3b
	s_set_vgpr_msb 0xaa00
.LBB0_6:
	s_set_vgpr_msb 0x8a
	v_or_b32_e32 v10 /*v522*/, 32, v115 /*v627*/
	v_or_b32_e32 v11 /*v523*/, 0xe0, v115 /*v627*/
	v_or_b32_e32 v12 /*v524*/, 32, v116 /*v628*/
	s_clause 0x4
	buffer_load_b128 v[2:5] /*v[514:517]*/, v115 /*v627*/, s[72:75], null offen
	buffer_load_b128 v[126:129] /*v[638:641]*/, v116 /*v628*/, s[72:75], null offen
	buffer_load_b128 v[6:9] /*v[518:521]*/, v10 /*v522*/, s[72:75], null offen
	buffer_load_b128 v[138:141] /*v[650:653]*/, v11 /*v523*/, s[72:75], null offen
	buffer_load_b128 v[130:133] /*v[642:645]*/, v12 /*v524*/, s[72:75], null offen
	v_or_b32_e32 v13 /*v525*/, 64, v115 /*v627*/
	v_or_b32_e32 v14 /*v526*/, 0x60, v115 /*v627*/
	v_or_b32_e32 v15 /*v527*/, 64, v116 /*v628*/
	v_or_b32_e32 v17 /*v529*/, 0x80, v115 /*v627*/
	v_or_b32_e32 v16 /*v528*/, 0x60, v116 /*v628*/
	v_or_b32_e32 v18 /*v530*/, 0xa0, v115 /*v627*/
	v_or_b32_e32 v19 /*v531*/, 0x80, v116 /*v628*/
	s_clause 0x3
	buffer_load_b128 v[142:145] /*v[654:657]*/, v13 /*v525*/, s[72:75], null offen
	buffer_load_b128 v[146:149] /*v[658:661]*/, v14 /*v526*/, s[72:75], null offen
	buffer_load_b128 v[150:153] /*v[662:665]*/, v15 /*v527*/, s[72:75], null offen
	buffer_load_b128 v[154:157] /*v[666:669]*/, v16 /*v528*/, s[72:75], null offen
	v_or_b32_e32 v20 /*v532*/, 0xa0, v116 /*v628*/
	s_clause 0x3
	buffer_load_b128 v[158:161] /*v[670:673]*/, v17 /*v529*/, s[72:75], null offen
	buffer_load_b128 v[162:165] /*v[674:677]*/, v18 /*v530*/, s[72:75], null offen
	buffer_load_b128 v[166:169] /*v[678:681]*/, v19 /*v531*/, s[72:75], null offen
	buffer_load_b128 v[170:173] /*v[682:685]*/, v20 /*v532*/, s[72:75], null offen
	v_or_b32_e32 v21 /*v533*/, 0xc0, v115 /*v627*/
	v_or_b32_e32 v22 /*v534*/, 0xc0, v116 /*v628*/
	v_or_b32_e32 v23 /*v535*/, 0xe0, v116 /*v628*/
	s_clause 0x4
	buffer_load_b128 v[174:177] /*v[686:689]*/, v115 /*v627*/, s[76:79], null offen
	buffer_load_b128 v[182:185] /*v[694:697]*/, v116 /*v628*/, s[76:79], null offen
	buffer_load_b128 v[178:181] /*v[690:693]*/, v10 /*v522*/, s[76:79], null offen
	buffer_load_b128 v[194:197] /*v[706:709]*/, v11 /*v523*/, s[76:79], null offen
	buffer_load_b128 v[186:189] /*v[698:701]*/, v12 /*v524*/, s[76:79], null offen
	s_clause 0x2
	buffer_load_b128 v[134:137] /*v[646:649]*/, v21 /*v533*/, s[72:75], null offen
	buffer_load_b128 v[198:201] /*v[710:713]*/, v22 /*v534*/, s[72:75], null offen
	buffer_load_b128 v[202:205] /*v[714:717]*/, v23 /*v535*/, s[72:75], null offen
	s_clause 0xa
	buffer_load_b128 v[206:209] /*v[718:721]*/, v13 /*v525*/, s[76:79], null offen
	buffer_load_b128 v[210:213] /*v[722:725]*/, v14 /*v526*/, s[76:79], null offen
	buffer_load_b128 v[214:217] /*v[726:729]*/, v15 /*v527*/, s[76:79], null offen
	buffer_load_b128 v[218:221] /*v[730:733]*/, v16 /*v528*/, s[76:79], null offen
	buffer_load_b128 v[222:225] /*v[734:737]*/, v17 /*v529*/, s[76:79], null offen
	buffer_load_b128 v[226:229] /*v[738:741]*/, v18 /*v530*/, s[76:79], null offen
	buffer_load_b128 v[230:233] /*v[742:745]*/, v19 /*v531*/, s[76:79], null offen
	buffer_load_b128 v[234:237] /*v[746:749]*/, v20 /*v532*/, s[76:79], null offen
	buffer_load_b128 v[190:193] /*v[702:705]*/, v21 /*v533*/, s[76:79], null offen
	buffer_load_b128 v[14:17] /*v[526:529]*/, v22 /*v534*/, s[76:79], null offen
	buffer_load_b128 v[18:21] /*v[530:533]*/, v23 /*v535*/, s[76:79], null offen
	s_wait_xcnt 0x1
	v_dual_add_nc_u32 v10 /*v522*/, 16, v117 /*v629*/ :: v_dual_bitop2_b32 v22 /*v534*/, 4, v117 /*v629*/ bitop3:0x54
	s_wait_xcnt 0x0
	v_or_b32_e32 v23 /*v535*/, 7, v117 /*v629*/
	v_or_b32_e32 v24 /*v536*/, 6, v117 /*v629*/
	v_or_b32_e32 v11 /*v523*/, 3, v117 /*v629*/
	v_or_b32_e32 v25 /*v537*/, 3, v10 /*v522*/
	v_or_b32_e32 v26 /*v538*/, 2, v10 /*v522*/
	v_or_b32_e32 v27 /*v539*/, 5, v10 /*v522*/
	v_or_b32_e32 v28 /*v540*/, 4, v10 /*v522*/
	v_or_b32_e32 v29 /*v541*/, 7, v10 /*v522*/
	v_or_b32_e32 v30 /*v542*/, 6, v10 /*v522*/
	v_cmp_gt_i32_e64 s63, v22 /*v534*/, v102 /*v614*/
	s_set_vgpr_msb 0x8a02
	v_cmp_gt_i32_e64 s59, v23 /*v535*/, v1
	s_set_vgpr_msb 0x20a
	v_cmp_gt_i32_e64 s62, v24 /*v536*/, v102 /*v614*/
	v_cmp_gt_i32_e64 s55, v22 /*v534*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s51, v23 /*v535*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s54, v24 /*v536*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s47, v22 /*v534*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s43, v23 /*v535*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s46, v24 /*v536*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s39, v22 /*v534*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s35, v23 /*v535*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s38, v24 /*v536*/, v108 /*v620*/
	s_set_vgpr_msb 0xa02
	v_cmp_gt_i32_e64 s28, v25 /*v537*/, v1
	s_set_vgpr_msb 0x20a
	v_cmp_gt_i32_e64 s34, v26 /*v538*/, v102 /*v614*/
	s_set_vgpr_msb 0xa02
	v_cmp_gt_i32_e64 s27, v27 /*v539*/, v1
	s_set_vgpr_msb 0x20a
	v_cmp_gt_i32_e64 s30, v28 /*v540*/, v102 /*v614*/
	s_set_vgpr_msb 0xa02
	v_cmp_gt_i32_e64 s26, v29 /*v541*/, v1
	s_set_vgpr_msb 0x28a
	v_cmp_gt_i32_e64 s29, v30 /*v542*/, v102 /*v614*/
	v_cmp_gt_i32_e64 s20, v25 /*v537*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s25, v26 /*v538*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s19, v27 /*v539*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s22, v28 /*v540*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s18, v29 /*v541*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s21, v30 /*v542*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s12, v25 /*v537*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s17, v26 /*v538*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s11, v27 /*v539*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s14, v28 /*v540*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s10, v29 /*v541*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s13, v30 /*v542*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s4, v25 /*v537*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s9, v26 /*v538*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s3, v27 /*v539*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s6, v28 /*v540*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s2, v29 /*v541*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s5, v30 /*v542*/, v108 /*v620*/
	v_or_b32_e32 v12 /*v524*/, 2, v117 /*v629*/
	v_or_b32_e32 v13 /*v525*/, 5, v117 /*v629*/
	v_cmp_ge_i32_e32 vcc_lo, v117 /*v629*/, v102 /*v614*/
	v_cmp_gt_i32_e64 s64, v117 /*v629*/, v102 /*v614*/
	s_set_vgpr_msb 0x8a02
	v_cmp_gt_i32_e64 s61, v11 /*v523*/, v1
	s_set_vgpr_msb 0x20a
	v_cmp_gt_i32_e64 s65, v12 /*v524*/, v102 /*v614*/
	s_set_vgpr_msb 0xa02
	v_cmp_gt_i32_e64 s60, v13 /*v525*/, v1
	s_set_vgpr_msb 0x28a
	v_cmp_ge_i32_e64 s56, v117 /*v629*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s57, v117 /*v629*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s53, v11 /*v523*/, v103 /*v615*/
	v_cmp_gt_i32_e64 s58, v12 /*v524*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s52, v13 /*v525*/, v103 /*v615*/
	v_cmp_ge_i32_e64 s48, v117 /*v629*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s49, v117 /*v629*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s45, v11 /*v523*/, v105 /*v617*/
	v_cmp_gt_i32_e64 s50, v12 /*v524*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s44, v13 /*v525*/, v105 /*v617*/
	v_cmp_ge_i32_e64 s40, v117 /*v629*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s41, v117 /*v629*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s37, v11 /*v523*/, v107 /*v619*/
	v_cmp_gt_i32_e64 s42, v12 /*v524*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s36, v13 /*v525*/, v107 /*v619*/
	v_cmp_ge_i32_e64 s31, v10 /*v522*/, v102 /*v614*/
	v_cmp_gt_i32_e64 s33, v10 /*v522*/, v102 /*v614*/
	v_cmp_ge_i32_e64 s23, v10 /*v522*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s24, v10 /*v522*/, v104 /*v616*/
	v_cmp_ge_i32_e64 s15, v10 /*v522*/, v106 /*v618*/
	v_cmp_gt_i32_e64 s16, v10 /*v522*/, v106 /*v618*/
	v_cmp_ge_i32_e64 s7, v10 /*v522*/, v108 /*v620*/
	v_cmp_gt_i32_e64 s8, v10 /*v522*/, v108 /*v620*/
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
	v_dual_add_nc_u32 v115 /*v627*/, s68, v115 /*v627*/ :: v_dual_add_nc_u32 v116 /*v628*/, s68, v116 /*v628*/
	v_add_nc_u32_e32 v117 /*v629*/, 32, v117 /*v629*/
	s_add_nc_u64 s[70:71], s[70:71], -1
	s_set_vgpr_msb 0x8a86
	s_wait_loadcnt 0x10
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[534:541]*/, v[174:181] /*v[686:693]*/, v[154:161] /*v[410:417]*/, 0
	s_set_vgpr_msb 0x860a
	ds_store_b128 v113 /*v625*/, v[2:5] /*v[514:517]*/
	ds_store_b128 v113 /*v625*/, v[6:9] /*v[518:521]*/ offset:32
	ds_store_b128 v113 /*v625*/, v[142:145] /*v[654:657]*/ offset:64
	ds_store_b128 v113 /*v625*/, v[146:149] /*v[658:661]*/ offset:96
	s_set_vgpr_msb 0xa82
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[2:9] /*v[514:521]*/, v[242:249], 0
	s_set_vgpr_msb 0x820a
	ds_store_b128 v113 /*v625*/, v[158:161] /*v[670:673]*/ offset:128
	ds_store_b128 v113 /*v625*/, v[162:165] /*v[674:677]*/ offset:160
	s_wait_loadcnt 0xd
	ds_store_b128 v113 /*v625*/, v[134:137] /*v[646:649]*/ offset:192
	ds_store_b128 v113 /*v625*/, v[138:141] /*v[650:653]*/ offset:224
	ds_store_b128 v114 /*v626*/, v[126:129] /*v[638:641]*/
	ds_store_b128 v114 /*v626*/, v[130:133] /*v[642:645]*/ offset:32
	ds_store_b128 v114 /*v626*/, v[150:153] /*v[662:665]*/ offset:64
	ds_store_b128 v114 /*v626*/, v[154:157] /*v[666:669]*/ offset:96
	ds_store_b128 v114 /*v626*/, v[166:169] /*v[678:681]*/ offset:128
	ds_store_b128 v114 /*v626*/, v[170:173] /*v[682:685]*/ offset:160
	s_wait_loadcnt 0xc
	ds_store_b128 v114 /*v626*/, v[198:201] /*v[710:713]*/ offset:192
	s_wait_loadcnt 0xb
	ds_store_b128 v114 /*v626*/, v[202:205] /*v[714:717]*/ offset:224
	s_cmp_lg_u64 s[70:71], 0
	s_set_vgpr_msb 0xa86
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[2:9] /*v[514:521]*/, v[50:57] /*v[306:313]*/, 0
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[174:181] /*v[686:693]*/, v[82:89] /*v[338:345]*/, 0
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[174:181] /*v[686:693]*/, v[18:25] /*v[274:281]*/, 0
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[2:9] /*v[514:521]*/, v[122:129] /*v[378:385]*/, 0
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[2:9] /*v[514:521]*/, v[186:193] /*v[442:449]*/, 0
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v118 /*v630*/
	ds_load_tr16_b128 v[10:13] /*v[522:525]*/, v118 /*v630*/ offset:4352
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v119 /*v631*/
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[174:181] /*v[686:693]*/, v[226:233] /*v[482:489]*/, 0
	s_set_vgpr_msb 0x8682
	v_wmma_f32_16x16x32_bf16 v[174:181] /*v[686:693]*/, v[126:133] /*v[638:645]*/, v[242:249], 0
	s_set_vgpr_msb 0x8286
	v_wmma_f32_16x16x32_bf16 v[246:253] /*v[758:765]*/, v[126:133] /*v[638:645]*/, v[50:57] /*v[306:313]*/, 0
	s_set_vgpr_msb 0x86c6
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[774:781]*/, v[126:133] /*v[638:645]*/, v[122:129] /*v[378:385]*/, 0
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[790:797]*/, v[126:133] /*v[638:645]*/, v[186:193] /*v[442:449]*/, 0
	s_set_vgpr_msb 0xc6a2
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[142:149] /*v[654:661]*/, v[250:257], v[46:53] /*v[558:565]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[142:149] /*v[654:661]*/, v[58:65] /*v[314:321]*/, v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[206:213] /*v[718:725]*/, v[26:33] /*v[282:289]*/, v[54:61] /*v[566:573]*/
	v_wmma_f32_16x16x32_bf16 v[238:245] /*v[750:757]*/, v[182:189] /*v[694:701]*/, v[18:25] /*v[274:281]*/, 0
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[766:773]*/, v[182:189] /*v[694:701]*/, v[82:89] /*v[338:345]*/, 0
	s_set_vgpr_msb 0xa6c6
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[782:789]*/, v[182:189] /*v[694:701]*/, v[154:161] /*v[410:417]*/, 0
	s_set_vgpr_msb 0xc6a6
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[638:645]*/, v[182:189] /*v[694:701]*/, v[226:233] /*v[482:489]*/, 0
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[142:149] /*v[654:661]*/, v[130:137] /*v[386:393]*/, v[78:85] /*v[590:597]*/
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[142:149] /*v[654:661]*/, v[194:201] /*v[450:457]*/, v[70:77] /*v[582:589]*/
	s_set_vgpr_msb 0xa6a2
	v_wmma_f32_16x16x32_bf16 v[174:181] /*v[686:693]*/, v[150:157] /*v[662:669]*/, v[250:257], v[174:181] /*v[686:693]*/
	s_set_vgpr_msb 0xa2a6
	v_wmma_f32_16x16x32_bf16 v[246:253] /*v[758:765]*/, v[150:157] /*v[662:669]*/, v[58:65] /*v[314:321]*/, v[246:253] /*v[758:765]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[774:781]*/, v[150:157] /*v[662:669]*/, v[130:137] /*v[386:393]*/, v[6:13] /*v[774:781]*/
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[790:797]*/, v[150:157] /*v[662:669]*/, v[194:201] /*v[450:457]*/, v[22:29] /*v[790:797]*/
	s_set_vgpr_msb 0xf6a6
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[158:165] /*v[670:677]*/, v[2:9] /*v[258:265]*/, v[46:53] /*v[558:565]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[158:165] /*v[670:677]*/, v[66:73] /*v[322:329]*/, v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[222:229] /*v[734:741]*/, v[34:41] /*v[290:297]*/, v[54:61] /*v[566:573]*/
	v_wmma_f32_16x16x32_bf16 v[238:245] /*v[750:757]*/, v[214:221] /*v[726:733]*/, v[26:33] /*v[282:289]*/, v[238:245] /*v[750:757]*/
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[766:773]*/, v[214:221] /*v[726:733]*/, v[90:97] /*v[346:353]*/, v[254:261] /*v[766:773]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[782:789]*/, v[214:221] /*v[726:733]*/, v[162:169] /*v[418:425]*/, v[14:21] /*v[782:789]*/
	s_set_vgpr_msb 0xf6a6
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[638:645]*/, v[214:221] /*v[726:733]*/, v[234:241] /*v[490:497]*/, v[126:133] /*v[638:645]*/
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[158:165] /*v[670:677]*/, v[138:145] /*v[394:401]*/, v[78:85] /*v[590:597]*/
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[158:165] /*v[670:677]*/, v[202:209] /*v[458:465]*/, v[70:77] /*v[582:589]*/
	v_wmma_f32_16x16x32_bf16 v[174:181] /*v[686:693]*/, v[166:173] /*v[678:685]*/, v[2:9] /*v[258:265]*/, v[174:181] /*v[686:693]*/
	v_wmma_f32_16x16x32_bf16 v[246:253] /*v[758:765]*/, v[166:173] /*v[678:685]*/, v[66:73] /*v[322:329]*/, v[246:253] /*v[758:765]*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[774:781]*/, v[166:173] /*v[678:685]*/, v[138:145] /*v[394:401]*/, v[6:13] /*v[774:781]*/
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[790:797]*/, v[166:173] /*v[678:685]*/, v[202:209] /*v[458:465]*/, v[22:29] /*v[790:797]*/
	s_set_vgpr_msb 0xf6a6
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[134:141] /*v[646:653]*/, v[10:17] /*v[266:273]*/, v[46:53] /*v[558:565]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[134:141] /*v[646:653]*/, v[74:81] /*v[330:337]*/, v[38:45] /*v[550:557]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa68a
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[110:111] /*v[622:623]*/, v[38:39] /*v[550:551]*/
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[238:245] /*v[750:757]*/, v[230:237] /*v[742:749]*/, v[34:41] /*v[290:297]*/, v[238:245] /*v[750:757]*/
	s_set_vgpr_msb 0xa68a
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[110:111] /*v[622:623]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[110:111] /*v[622:623]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[110:111] /*v[622:623]*/, v[44:45] /*v[556:557]*/
	v_cndmask_b32_e64 v39 /*v551*/, v39 /*v551*/, 0xff61b1e6, s56
	v_cndmask_b32_e64 v38 /*v550*/, v38 /*v550*/, 0xff61b1e6, s57
	v_cndmask_b32_e64 v41 /*v553*/, v41 /*v553*/, 0xff61b1e6, s53
	v_cndmask_b32_e64 v40 /*v552*/, v40 /*v552*/, 0xff61b1e6, s58
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[766:773]*/, v[230:237] /*v[742:749]*/, v[106:113] /*v[362:369]*/, v[254:261] /*v[766:773]*/
	v_cndmask_b32_e64 v43 /*v555*/, v43 /*v555*/, 0xff61b1e6, s52
	v_cndmask_b32_e64 v42 /*v554*/, v42 /*v554*/, 0xff61b1e6, s55
	v_cndmask_b32_e64 v45 /*v557*/, v45 /*v557*/, 0xff61b1e6, s51
	v_cndmask_b32_e64 v44 /*v556*/, v44 /*v556*/, 0xff61b1e6, s54
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[782:789]*/, v[230:237] /*v[742:749]*/, v[170:177] /*v[426:433]*/, v[14:21] /*v[782:789]*/
	s_set_vgpr_msb 0xf68a
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, s[66:67] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, s[66:67] op_sel_hi:[1,0]
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[638:645]*/, v[230:237] /*v[742:749]*/, v[242:249] /*v[498:505]*/, v[126:133] /*v[638:645]*/
	v_exp_f32_e32 v39 /*v551*/, v39 /*v551*/
	v_exp_f32_e32 v40 /*v552*/, v40 /*v552*/
	v_exp_f32_e32 v41 /*v553*/, v41 /*v553*/
	v_exp_f32_e32 v42 /*v554*/, v42 /*v554*/
	v_exp_f32_e32 v43 /*v555*/, v43 /*v555*/
	v_exp_f32_e32 v44 /*v556*/, v44 /*v556*/
	v_exp_f32_e32 v45 /*v557*/, v45 /*v557*/
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[190:197] /*v[702:709]*/, v[42:49] /*v[298:305]*/, v[54:61] /*v[566:573]*/
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[134:141] /*v[646:653]*/, v[146:153] /*v[402:409]*/, v[78:85] /*v[590:597]*/
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[134:141] /*v[646:653]*/, v[210:217] /*v[466:473]*/, v[70:77] /*v[582:589]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa68a
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[110:111] /*v[622:623]*/, v[70:71] /*v[582:583]*/
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[174:181] /*v[686:693]*/, v[198:205] /*v[710:717]*/, v[10:17] /*v[266:273]*/, v[174:181] /*v[686:693]*/
	s_set_vgpr_msb 0xa68a
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[110:111] /*v[622:623]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[110:111] /*v[622:623]*/, v[74:75] /*v[586:587]*/
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[110:111] /*v[622:623]*/, v[76:77] /*v[588:589]*/
	v_cndmask_b32_e64 v71 /*v583*/, v71 /*v583*/, 0xff61b1e6, s40
	v_cndmask_b32_e64 v70 /*v582*/, v70 /*v582*/, 0xff61b1e6, s41
	v_cndmask_b32_e64 v73 /*v585*/, v73 /*v585*/, 0xff61b1e6, s37
	v_cndmask_b32_e64 v72 /*v584*/, v72 /*v584*/, 0xff61b1e6, s42
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[246:253] /*v[758:765]*/, v[198:205] /*v[710:717]*/, v[74:81] /*v[330:337]*/, v[246:253] /*v[758:765]*/
	v_cndmask_b32_e64 v75 /*v587*/, v75 /*v587*/, 0xff61b1e6, s36
	v_cndmask_b32_e64 v74 /*v586*/, v74 /*v586*/, 0xff61b1e6, s39
	v_cndmask_b32_e64 v77 /*v589*/, v77 /*v589*/, 0xff61b1e6, s35
	v_cndmask_b32_e64 v76 /*v588*/, v76 /*v588*/, 0xff61b1e6, s38
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[70:71] /*v[582:583]*/, v[70:71] /*v[582:583]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[72:73] /*v[584:585]*/, v[72:73] /*v[584:585]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[74:75] /*v[586:587]*/, v[74:75] /*v[586:587]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[774:781]*/, v[198:205] /*v[710:717]*/, v[146:153] /*v[402:409]*/, v[6:13] /*v[774:781]*/
	s_set_vgpr_msb 0xf68a
	v_pk_mul_f32 v[142:143] /*v[654:655]*/, v[110:111] /*v[622:623]*/, v[246:247] /*v[758:759]*/
	v_pk_mul_f32 v[144:145] /*v[656:657]*/, v[110:111] /*v[622:623]*/, v[248:249] /*v[760:761]*/
	v_pk_mul_f32 v[146:147] /*v[658:659]*/, v[110:111] /*v[622:623]*/, v[250:251] /*v[762:763]*/
	v_pk_mul_f32 v[148:149] /*v[660:661]*/, v[110:111] /*v[622:623]*/, v[252:253] /*v[764:765]*/
	v_pk_add_f32 v[76:77] /*v[588:589]*/, v[76:77] /*v[588:589]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v143 /*v655*/, v143 /*v655*/, 0xff61b1e6, s23
	v_cndmask_b32_e64 v142 /*v654*/, v142 /*v654*/, 0xff61b1e6, s24
	s_set_vgpr_msb 0x8af6
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[790:797]*/, v[198:205] /*v[710:717]*/, v[210:217] /*v[466:473]*/, v[22:29] /*v[790:797]*/
	s_set_vgpr_msb 0xf68e
	v_pk_mul_f32 v[158:159] /*v[670:671]*/, v[110:111] /*v[622:623]*/, v[6:7] /*v[774:775]*/
	v_pk_mul_f32 v[160:161] /*v[672:673]*/, v[110:111] /*v[622:623]*/, v[8:9] /*v[776:777]*/
	v_pk_mul_f32 v[162:163] /*v[674:675]*/, v[110:111] /*v[622:623]*/, v[10:11] /*v[778:779]*/
	v_pk_mul_f32 v[164:165] /*v[676:677]*/, v[110:111] /*v[622:623]*/, v[12:13] /*v[780:781]*/
	v_cndmask_b32_e64 v145 /*v657*/, v145 /*v657*/, 0xff61b1e6, s20
	v_cndmask_b32_e64 v144 /*v656*/, v144 /*v656*/, 0xff61b1e6, s25
	v_cndmask_b32_e64 v147 /*v659*/, v147 /*v659*/, 0xff61b1e6, s19
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[534:541]*/, v[206:213] /*v[718:725]*/, v[162:169] /*v[418:425]*/, v[22:29] /*v[534:541]*/
	v_cndmask_b32_e64 v146 /*v658*/, v146 /*v658*/, 0xff61b1e6, s22
	v_cndmask_b32_e64 v149 /*v661*/, v149 /*v661*/, 0xff61b1e6, s18
	v_cndmask_b32_e64 v148 /*v660*/, v148 /*v660*/, 0xff61b1e6, s21
	v_cndmask_b32_e64 v159 /*v671*/, v159 /*v671*/, 0xff61b1e6, s15
	v_cndmask_b32_e64 v158 /*v670*/, v158 /*v670*/, 0xff61b1e6, s16
	v_cndmask_b32_e64 v161 /*v673*/, v161 /*v673*/, 0xff61b1e6, s12
	v_cndmask_b32_e64 v160 /*v672*/, v160 /*v672*/, 0xff61b1e6, s17
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[206:213] /*v[718:725]*/, v[90:97] /*v[346:353]*/, v[30:37] /*v[542:549]*/
	v_cndmask_b32_e64 v163 /*v675*/, v163 /*v675*/, 0xff61b1e6, s11
	v_cndmask_b32_e64 v162 /*v674*/, v162 /*v674*/, 0xff61b1e6, s14
	v_cndmask_b32_e64 v165 /*v677*/, v165 /*v677*/, 0xff61b1e6, s10
	v_cndmask_b32_e64 v164 /*v676*/, v164 /*v676*/, 0xff61b1e6, s13
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[142:143] /*v[654:655]*/, v[142:143] /*v[654:655]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[144:145] /*v[656:657]*/, v[144:145] /*v[656:657]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[146:147] /*v[658:659]*/, v[146:147] /*v[658:659]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[206:213] /*v[718:725]*/, v[234:241] /*v[490:497]*/, v[62:69] /*v[574:581]*/
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[148:149] /*v[660:661]*/, v[148:149] /*v[660:661]*/, v[88:89] /*v[600:601]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159] /*v[670:671]*/, v[158:159] /*v[670:671]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[160:161] /*v[672:673]*/, v[160:161] /*v[672:673]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[162:163] /*v[674:675]*/, v[162:163] /*v[674:675]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[164:165] /*v[676:677]*/, v[164:165] /*v[676:677]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[70:71] /*v[582:583]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[72:73] /*v[584:585]*/, s[66:67] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[238:245] /*v[750:757]*/, v[14:21] /*v[526:533]*/, v[42:49] /*v[298:305]*/, v[238:245] /*v[750:757]*/
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[74:75] /*v[586:587]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[76:77] /*v[588:589]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143] /*v[654:655]*/, v[142:143] /*v[654:655]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145] /*v[656:657]*/, v[144:145] /*v[656:657]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147] /*v[658:659]*/, v[146:147] /*v[658:659]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149] /*v[660:661]*/, v[148:149] /*v[660:661]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159] /*v[670:671]*/, v[158:159] /*v[670:671]*/, s[66:67] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[766:773]*/, v[14:21] /*v[526:533]*/, v[114:121] /*v[370:377]*/, v[254:261] /*v[766:773]*/
	v_pk_mul_f32 v[160:161] /*v[672:673]*/, v[160:161] /*v[672:673]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163] /*v[674:675]*/, v[162:163] /*v[674:675]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165] /*v[676:677]*/, v[164:165] /*v[676:677]*/, s[66:67] op_sel_hi:[1,0]
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	v_exp_f32_e32 v71 /*v583*/, v71 /*v583*/
	v_exp_f32_e32 v72 /*v584*/, v72 /*v584*/
	v_exp_f32_e32 v73 /*v585*/, v73 /*v585*/
	s_set_vgpr_msb 0xa6f6
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[782:789]*/, v[14:21] /*v[526:533]*/, v[178:185] /*v[434:441]*/, v[14:21] /*v[782:789]*/
	s_set_vgpr_msb 0xf6a6
	v_exp_f32_e32 v74 /*v586*/, v74 /*v586*/
	v_exp_f32_e32 v75 /*v587*/, v75 /*v587*/
	v_exp_f32_e32 v76 /*v588*/, v76 /*v588*/
	v_exp_f32_e32 v77 /*v589*/, v77 /*v589*/
	v_exp_f32_e32 v142 /*v654*/, v142 /*v654*/
	v_exp_f32_e32 v143 /*v655*/, v143 /*v655*/
	v_exp_f32_e32 v144 /*v656*/, v144 /*v656*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[638:645]*/, v[14:21] /*v[526:533]*/, v[250:257] /*v[506:513]*/, v[126:133] /*v[638:645]*/
	v_exp_f32_e32 v145 /*v657*/, v145 /*v657*/
	v_exp_f32_e32 v146 /*v658*/, v146 /*v658*/
	v_exp_f32_e32 v147 /*v659*/, v147 /*v659*/
	v_exp_f32_e32 v148 /*v660*/, v148 /*v660*/
	s_set_vgpr_msb 0xa68a
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[110:111] /*v[622:623]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[110:111] /*v[622:623]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[110:111] /*v[622:623]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[110:111] /*v[622:623]*/, v[52:53] /*v[564:565]*/
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[54:55] /*v[566:567]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[56:57] /*v[568:569]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[58:59] /*v[570:571]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[60:61] /*v[572:573]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[110:111] /*v[622:623]*/, v[78:79] /*v[590:591]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[110:111] /*v[622:623]*/, v[80:81] /*v[592:593]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[110:111] /*v[622:623]*/, v[82:83] /*v[594:595]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[110:111] /*v[622:623]*/, v[84:85] /*v[596:597]*/
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[110:111] /*v[622:623]*/, v[174:175] /*v[686:687]*/
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[110:111] /*v[622:623]*/, v[176:177] /*v[688:689]*/
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[110:111] /*v[622:623]*/, v[178:179] /*v[690:691]*/
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[110:111] /*v[622:623]*/, v[180:181] /*v[692:693]*/
	s_set_vgpr_msb 0x8a8e
	v_pk_mul_f32 v[174:175] /*v[686:687]*/, v[110:111] /*v[622:623]*/, v[22:23] /*v[790:791]*/
	v_pk_mul_f32 v[176:177] /*v[688:689]*/, v[110:111] /*v[622:623]*/, v[24:25] /*v[792:793]*/
	v_pk_mul_f32 v[178:179] /*v[690:691]*/, v[110:111] /*v[622:623]*/, v[26:27] /*v[794:795]*/
	v_pk_mul_f32 v[180:181] /*v[692:693]*/, v[110:111] /*v[622:623]*/, v[28:29] /*v[796:797]*/
	v_cndmask_b32_e64 v15 /*v527*/, v15 /*v527*/, 0xff61b1e6, s86
	v_cndmask_b32_e64 v14 /*v526*/, v14 /*v526*/, 0xff61b1e6, s64
	v_cndmask_b32_e64 v17 /*v529*/, v17 /*v529*/, 0xff61b1e6, s61
	v_cndmask_b32_e64 v16 /*v528*/, v16 /*v528*/, 0xff61b1e6, s65
	v_cndmask_b32_e64 v19 /*v531*/, v19 /*v531*/, 0xff61b1e6, s60
	v_cndmask_b32_e64 v18 /*v530*/, v18 /*v530*/, 0xff61b1e6, s63
	v_cndmask_b32_e64 v21 /*v533*/, v21 /*v533*/, 0xff61b1e6, s59
	v_cndmask_b32_e64 v20 /*v532*/, v20 /*v532*/, 0xff61b1e6, s62
	v_cndmask_b32_e64 v55 /*v567*/, v55 /*v567*/, 0xff61b1e6, s48
	v_cndmask_b32_e64 v54 /*v566*/, v54 /*v566*/, 0xff61b1e6, s49
	v_cndmask_b32_e64 v57 /*v569*/, v57 /*v569*/, 0xff61b1e6, s45
	v_cndmask_b32_e64 v56 /*v568*/, v56 /*v568*/, 0xff61b1e6, s50
	v_cndmask_b32_e64 v59 /*v571*/, v59 /*v571*/, 0xff61b1e6, s44
	v_cndmask_b32_e64 v58 /*v570*/, v58 /*v570*/, 0xff61b1e6, s47
	v_cndmask_b32_e64 v61 /*v573*/, v61 /*v573*/, 0xff61b1e6, s43
	v_cndmask_b32_e64 v60 /*v572*/, v60 /*v572*/, 0xff61b1e6, s46
	v_cndmask_b32_e64 v79 /*v591*/, v79 /*v591*/, 0xff61b1e6, s31
	v_cndmask_b32_e64 v78 /*v590*/, v78 /*v590*/, 0xff61b1e6, s33
	v_cndmask_b32_e64 v81 /*v593*/, v81 /*v593*/, 0xff61b1e6, s28
	v_cndmask_b32_e64 v80 /*v592*/, v80 /*v592*/, 0xff61b1e6, s34
	v_cndmask_b32_e64 v83 /*v595*/, v83 /*v595*/, 0xff61b1e6, s27
	v_cndmask_b32_e64 v82 /*v594*/, v82 /*v594*/, 0xff61b1e6, s30
	v_cndmask_b32_e64 v85 /*v597*/, v85 /*v597*/, 0xff61b1e6, s26
	v_cndmask_b32_e64 v84 /*v596*/, v84 /*v596*/, 0xff61b1e6, s29
	v_cndmask_b32_e64 v175 /*v687*/, v175 /*v687*/, 0xff61b1e6, s7
	v_cndmask_b32_e64 v174 /*v686*/, v174 /*v686*/, 0xff61b1e6, s8
	v_cndmask_b32_e64 v177 /*v689*/, v177 /*v689*/, 0xff61b1e6, s4
	v_cndmask_b32_e64 v176 /*v688*/, v176 /*v688*/, 0xff61b1e6, s9
	v_cndmask_b32_e64 v179 /*v691*/, v179 /*v691*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v178 /*v690*/, v178 /*v690*/, 0xff61b1e6, s6
	v_cndmask_b32_e64 v181 /*v693*/, v181 /*v693*/, 0xff61b1e6, s2
	v_cndmask_b32_e64 v180 /*v692*/, v180 /*v692*/, 0xff61b1e6, s5
	s_set_vgpr_msb 0x8ea6
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[534:541]*/, v[222:229] /*v[734:741]*/, v[170:177] /*v[426:433]*/, v[22:29] /*v[534:541]*/
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[222:229] /*v[734:741]*/, v[106:113] /*v[362:369]*/, v[30:37] /*v[542:549]*/
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[60:61] /*v[572:573]*/, v[60:61] /*v[572:573]*/, v[90:91] /*v[602:603]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[78:79] /*v[590:591]*/, v[78:79] /*v[590:591]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[82:83] /*v[594:595]*/, v[82:83] /*v[594:595]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, v[86:87] /*v[598:599]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[174:175] /*v[686:687]*/, v[174:175] /*v[686:687]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[176:177] /*v[688:689]*/, v[176:177] /*v[688:689]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[222:229] /*v[734:741]*/, v[242:249] /*v[498:505]*/, v[62:69] /*v[574:581]*/
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[178:179] /*v[690:691]*/, v[178:179] /*v[690:691]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[180:181] /*v[692:693]*/, v[180:181] /*v[692:693]*/, v[92:93] /*v[604:605]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, s[66:67] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8aa6
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[534:541]*/, v[190:197] /*v[702:709]*/, v[178:185] /*v[434:441]*/, v[22:29] /*v[534:541]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[60:61] /*v[572:573]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[78:79] /*v[590:591]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[80:81] /*v[592:593]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[82:83] /*v[594:595]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[84:85] /*v[596:597]*/, s[66:67] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[190:197] /*v[702:709]*/, v[114:121] /*v[370:377]*/, v[30:37] /*v[542:549]*/
	v_pk_mul_f32 v[174:175] /*v[686:687]*/, v[174:175] /*v[686:687]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177] /*v[688:689]*/, v[176:177] /*v[688:689]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[690:691]*/, v[178:179] /*v[690:691]*/, s[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[692:693]*/, v[180:181] /*v[692:693]*/, s[66:67] op_sel_hi:[1,0]
	v_exp_f32_e32 v14 /*v526*/, v14 /*v526*/
	v_exp_f32_e32 v15 /*v527*/, v15 /*v527*/
	v_exp_f32_e32 v16 /*v528*/, v16 /*v528*/
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[190:197] /*v[702:709]*/, v[250:257] /*v[506:513]*/, v[62:69] /*v[574:581]*/
	v_exp_f32_e32 v17 /*v529*/, v17 /*v529*/
	v_exp_f32_e32 v18 /*v530*/, v18 /*v530*/
	v_exp_f32_e32 v19 /*v531*/, v19 /*v531*/
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
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
	v_exp_f32_e32 v149 /*v661*/, v149 /*v661*/
	v_exp_f32_e32 v158 /*v670*/, v158 /*v670*/
	v_exp_f32_e32 v159 /*v671*/, v159 /*v671*/
	v_exp_f32_e32 v160 /*v672*/, v160 /*v672*/
	v_exp_f32_e32 v161 /*v673*/, v161 /*v673*/
	v_exp_f32_e32 v162 /*v674*/, v162 /*v674*/
	v_exp_f32_e32 v163 /*v675*/, v163 /*v675*/
	v_exp_f32_e32 v164 /*v676*/, v164 /*v676*/
	v_exp_f32_e32 v165 /*v677*/, v165 /*v677*/
	v_exp_f32_e32 v174 /*v686*/, v174 /*v686*/
	v_exp_f32_e32 v175 /*v687*/, v175 /*v687*/
	v_exp_f32_e32 v176 /*v688*/, v176 /*v688*/
	v_exp_f32_e32 v177 /*v689*/, v177 /*v689*/
	v_exp_f32_e32 v178 /*v690*/, v178 /*v690*/
	v_exp_f32_e32 v179 /*v691*/, v179 /*v691*/
	v_exp_f32_e32 v180 /*v692*/, v180 /*v692*/
	v_exp_f32_e32 v181 /*v693*/, v181 /*v693*/
	s_set_vgpr_msb 0xa68a
	v_pk_add_f32 v[30:31] /*v[542:543]*/, v[30:31] /*v[542:543]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[28:29] /*v[540:541]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[62:63] /*v[574:575]*/, v[62:63] /*v[574:575]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[64:65] /*v[576:577]*/, v[64:65] /*v[576:577]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[134:135] /*v[646:647]*/, v[238:239] /*v[750:751]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[136:137] /*v[648:649]*/, v[240:241] /*v[752:753]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[138:139] /*v[650:651]*/, v[242:243] /*v[754:755]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[140:141] /*v[652:653]*/, v[244:245] /*v[756:757]*/, v[94:95] /*v[606:607]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[150:151] /*v[662:663]*/, v[254:255] /*v[766:767]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a8b
	v_pk_add_f32 v[152:153] /*v[664:665]*/, v[0:1] /*v[768:769]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[154:155] /*v[666:667]*/, v[2:3] /*v[770:771]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[156:157] /*v[668:669]*/, v[4:5] /*v[772:773]*/, v[96:97] /*v[608:609]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[166:167] /*v[678:679]*/, v[14:15] /*v[782:783]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[168:169] /*v[680:681]*/, v[16:17] /*v[784:785]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[170:171] /*v[682:683]*/, v[18:19] /*v[786:787]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[172:173] /*v[684:685]*/, v[20:21] /*v[788:789]*/, v[98:99] /*v[610:611]*/ neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8b8a
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[126:127] /*v[638:639]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[128:129] /*v[640:641]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[130:131] /*v[642:643]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[132:133] /*v[644:645]*/, v[100:101] /*v[612:613]*/ neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[46:47] /*v[558:559]*/, v[14:15] /*v[526:527]*/
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[48:49] /*v[560:561]*/, v[16:17] /*v[528:529]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[50:51] /*v[562:563]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[52:53] /*v[564:565]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[30:31] /*v[542:543]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[28:29] /*v[540:541]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[62:63] /*v[574:575]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[64:65] /*v[576:577]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[66:67] /*v[578:579]*/, v[74:75] /*v[586:587]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[68:69] /*v[580:581]*/, v[76:77] /*v[588:589]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[134:135] /*v[646:647]*/, v[78:79] /*v[590:591]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[136:137] /*v[648:649]*/, v[80:81] /*v[592:593]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[138:139] /*v[650:651]*/, v[82:83] /*v[594:595]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[140:141] /*v[652:653]*/, v[84:85] /*v[596:597]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[150:151] /*v[662:663]*/, v[142:143] /*v[654:655]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[152:153] /*v[664:665]*/, v[144:145] /*v[656:657]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[154:155] /*v[666:667]*/, v[146:147] /*v[658:659]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[156:157] /*v[668:669]*/, v[148:149] /*v[660:661]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[166:167] /*v[678:679]*/, v[158:159] /*v[670:671]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[168:169] /*v[680:681]*/, v[160:161] /*v[672:673]*/
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[170:171] /*v[682:683]*/, v[162:163] /*v[674:675]*/
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[172:173] /*v[684:685]*/, v[164:165] /*v[676:677]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[126:127] /*v[638:639]*/, v[174:175] /*v[686:687]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[128:129] /*v[640:641]*/, v[176:177] /*v[688:689]*/
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[130:131] /*v[642:643]*/, v[178:179] /*v[690:691]*/
	v_pk_mul_f32 v[76:77] /*v[588:589]*/, v[132:133] /*v[644:645]*/, v[180:181] /*v[692:693]*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[110:111] /*v[622:623]*/, v[14:15] /*v[526:527]*/
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[110:111] /*v[622:623]*/, v[16:17] /*v[528:529]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[110:111] /*v[622:623]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[110:111] /*v[622:623]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[110:111] /*v[622:623]*/, v[30:31] /*v[542:543]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[110:111] /*v[622:623]*/, v[32:33] /*v[544:545]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[110:111] /*v[622:623]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[110:111] /*v[622:623]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[78:79] /*v[590:591]*/, v[110:111] /*v[622:623]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[80:81] /*v[592:593]*/, v[110:111] /*v[622:623]*/, v[24:25] /*v[536:537]*/
	v_pk_mul_f32 v[82:83] /*v[594:595]*/, v[110:111] /*v[622:623]*/, v[26:27] /*v[538:539]*/
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[110:111] /*v[622:623]*/, v[28:29] /*v[540:541]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[110:111] /*v[622:623]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[110:111] /*v[622:623]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[110:111] /*v[622:623]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[110:111] /*v[622:623]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[110:111] /*v[622:623]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[110:111] /*v[622:623]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[110:111] /*v[622:623]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[110:111] /*v[622:623]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[110:111] /*v[622:623]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[110:111] /*v[622:623]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[110:111] /*v[622:623]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[110:111] /*v[622:623]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[110:111] /*v[622:623]*/, v[62:63] /*v[574:575]*/
	v_pk_mul_f32 v[60:61] /*v[572:573]*/, v[110:111] /*v[622:623]*/, v[64:65] /*v[576:577]*/
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[110:111] /*v[622:623]*/, v[66:67] /*v[578:579]*/
	v_pk_mul_f32 v[64:65] /*v[576:577]*/, v[110:111] /*v[622:623]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[110:111] /*v[622:623]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[110:111] /*v[622:623]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[110:111] /*v[622:623]*/, v[74:75] /*v[586:587]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[110:111] /*v[622:623]*/, v[76:77] /*v[588:589]*/
	v_cvt_pk_bf16_f32 v14 /*v526*/, v14 /*v526*/, v15 /*v527*/
	v_cvt_pk_bf16_f32 v15 /*v527*/, v16 /*v528*/, v17 /*v529*/
	v_cvt_pk_bf16_f32 v16 /*v528*/, v18 /*v530*/, v19 /*v531*/
	v_cvt_pk_bf16_f32 v17 /*v529*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v22 /*v534*/, v30 /*v542*/, v31 /*v543*/
	v_cvt_pk_bf16_f32 v23 /*v535*/, v32 /*v544*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v24 /*v536*/, v34 /*v546*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v18 /*v530*/, v26 /*v538*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v19 /*v531*/, v28 /*v540*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v20 /*v532*/, v46 /*v558*/, v47 /*v559*/
	v_cvt_pk_bf16_f32 v21 /*v533*/, v48 /*v560*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v25 /*v537*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v30 /*v542*/, v78 /*v590*/, v79 /*v591*/
	v_cvt_pk_bf16_f32 v31 /*v543*/, v80 /*v592*/, v81 /*v593*/
	v_cvt_pk_bf16_f32 v26 /*v538*/, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v27 /*v539*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v28 /*v540*/, v54 /*v566*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v29 /*v541*/, v56 /*v568*/, v57 /*v569*/
	v_cvt_pk_bf16_f32 v32 /*v544*/, v82 /*v594*/, v83 /*v595*/
	v_cvt_pk_bf16_f32 v33 /*v545*/, v84 /*v596*/, v85 /*v597*/
	v_cvt_pk_bf16_f32 v38 /*v550*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v34 /*v546*/, v58 /*v570*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v35 /*v547*/, v60 /*v572*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v36 /*v548*/, v62 /*v574*/, v63 /*v575*/
	v_cvt_pk_bf16_f32 v37 /*v549*/, v64 /*v576*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v39 /*v551*/, v40 /*v552*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v40 /*v552*/, v42 /*v554*/, v43 /*v555*/
	v_cvt_pk_bf16_f32 v41 /*v553*/, v44 /*v556*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v42 /*v554*/, v66 /*v578*/, v67 /*v579*/
	v_cvt_pk_bf16_f32 v43 /*v555*/, v68 /*v580*/, v69 /*v581*/
	v_cvt_pk_bf16_f32 v44 /*v556*/, v70 /*v582*/, v71 /*v583*/
	v_cvt_pk_bf16_f32 v45 /*v557*/, v72 /*v584*/, v73 /*v585*/
	s_set_vgpr_msb 0x8a5a
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[14:21] /*v[526:533]*/, v[6:13] /*v[518:525]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5a82
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v120 /*v632*/
	ds_load_tr16_b128 v[50:53] /*v[562:565]*/, v120 /*v632*/ offset:4352
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[186:193], v[22:29] /*v[534:541]*/, v[6:13] /*v[518:525]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[122:129], v[30:37] /*v[542:549]*/, v[6:13] /*v[518:525]*/, v[122:129]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[38:45] /*v[550:557]*/, v[6:13] /*v[518:525]*/, v[58:65]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v119 /*v631*/ offset:4352
	s_set_vgpr_msb 0x825a
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[14:21] /*v[526:533]*/, v[2:9] /*v[514:521]*/, v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[178:185], v[22:29] /*v[534:541]*/, v[2:9] /*v[514:521]*/, v[178:185]
	v_wmma_f32_16x16x32_bf16 v[114:121], v[30:37] /*v[542:549]*/, v[2:9] /*v[514:521]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[38:45] /*v[550:557]*/, v[2:9] /*v[514:521]*/, v[50:57]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v121 /*v633*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v121 /*v633*/ offset:4352
	s_set_vgpr_msb 0x820a
	v_wmma_f32_16x16x32_bf16 v[234:241], v[14:21] /*v[526:533]*/, v[46:53] /*v[558:565]*/, v[234:241]
	v_wmma_f32_16x16x32_bf16 v[170:177], v[22:29] /*v[534:541]*/, v[46:53] /*v[558:565]*/, v[170:177]
	v_wmma_f32_16x16x32_bf16 v[106:113], v[30:37] /*v[542:549]*/, v[46:53] /*v[558:565]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[38:45] /*v[550:557]*/, v[46:53] /*v[558:565]*/, v[42:49]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v122 /*v634*/
	ds_load_tr16_b128 v[50:53] /*v[562:565]*/, v122 /*v634*/ offset:4352
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[226:233], v[14:21] /*v[526:533]*/, v[2:9] /*v[514:521]*/, v[226:233]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[22:29] /*v[534:541]*/, v[2:9] /*v[514:521]*/, v[162:169]
	v_wmma_f32_16x16x32_bf16 v[98:105], v[30:37] /*v[542:549]*/, v[2:9] /*v[514:521]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[38:45] /*v[550:557]*/, v[2:9] /*v[514:521]*/, v[34:41]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v123 /*v635*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v123 /*v635*/ offset:4352
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[218:225], v[14:21] /*v[526:533]*/, v[46:53] /*v[558:565]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[154:161], v[22:29] /*v[534:541]*/, v[46:53] /*v[558:565]*/, v[154:161]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[30:37] /*v[542:549]*/, v[46:53] /*v[558:565]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[38:45] /*v[550:557]*/, v[46:53] /*v[558:565]*/, v[26:33]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v124 /*v636*/
	ds_load_tr16_b128 v[50:53] /*v[562:565]*/, v124 /*v636*/ offset:4352
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[210:217], v[14:21] /*v[526:533]*/, v[2:9] /*v[514:521]*/, v[210:217]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[22:29] /*v[534:541]*/, v[2:9] /*v[514:521]*/, v[146:153]
	v_wmma_f32_16x16x32_bf16 v[82:89], v[30:37] /*v[542:549]*/, v[2:9] /*v[514:521]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[38:45] /*v[550:557]*/, v[2:9] /*v[514:521]*/, v[18:25]
	s_set_vgpr_msb 0xa82
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v125 /*v637*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v125 /*v637*/ offset:4352
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[202:209], v[14:21] /*v[526:533]*/, v[46:53] /*v[558:565]*/, v[202:209]
	v_wmma_f32_16x16x32_bf16 v[138:145], v[22:29] /*v[534:541]*/, v[46:53] /*v[558:565]*/, v[138:145]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[30:37] /*v[542:549]*/, v[46:53] /*v[558:565]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[38:45] /*v[550:557]*/, v[46:53] /*v[558:565]*/, v[10:17]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[14:21] /*v[526:533]*/, v[2:9] /*v[514:521]*/, v[194:201]
	v_wmma_f32_16x16x32_bf16 v[130:137], v[22:29] /*v[534:541]*/, v[2:9] /*v[514:521]*/, v[130:137]
	v_wmma_f32_16x16x32_bf16 v[66:73], v[30:37] /*v[542:549]*/, v[2:9] /*v[514:521]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[38:45] /*v[550:557]*/, v[2:9] /*v[514:521]*/, v[2:9]
	s_set_vgpr_msb 0xa00
	s_cbranch_scc1 .LBB0_6
.LBB0_7:
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, s83, v112 /*v624*/
	s_mul_i32 s4, s67, s85
	s_load_b64 s[0:1], s[0:1], 0x110 nv
	s_add_co_i32 s4, s4, s82
	s_mov_b32 s3, 0
	s_wait_loadcnt 0x3e
	v_mul_lo_u32 v245, v1, s67
	s_mov_b32 s2, 0x800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v245, v245, s4, 7
	v_or_b32_e32 v250, v245, v109 /*v621*/
	s_set_vgpr_msb 0x801
	v_or_b32_e32 v242, 1, v1
	s_wait_kmcnt 0x0
	v_cvt_pk_bf16_f32 v243, v218 /*v474*/, s0
	v_lshlrev_b32_e32 v250, 2, v250
	v_cvt_pk_bf16_f32 v244, v219 /*v475*/, s0
	v_cvt_pk_bf16_f32 v247, v220 /*v476*/, s0
	v_mul_lo_u32 v242, s67, v242
	v_cvt_pk_bf16_f32 v255, v222 /*v478*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v243, v250, s[0:3], null offen
	s_set_vgpr_msb 0x41
	v_cvt_pk_bf16_f32 v1 /*v257*/, v223 /*v479*/, s0
	v_cvt_pk_bf16_f32 v3 /*v259*/, v98 /*v354*/, s0
	v_cvt_pk_bf16_f32 v2 /*v258*/, v225 /*v481*/, s0
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v234, v234, s0
	v_cvt_pk_bf16_f32 v235, v235, s0
	v_add_lshl_u32 v242, s4, v242, 7
	v_cvt_pk_bf16_f32 v236, v236, s0
	v_cvt_pk_bf16_f32 v226, v226, s0
	v_cvt_pk_bf16_f32 v228, v228, s0
	v_cvt_pk_bf16_f32 v227, v227, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v251, v242, v109 /*v621*/
	v_cvt_pk_bf16_f32 v229, v229, s0
	v_cvt_pk_bf16_f32 v218, v218, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v246, 2, v1
	v_cvt_pk_bf16_f32 v221, v221, s0
	v_lshlrev_b32_e32 v251, 2, v251
	v_cvt_pk_bf16_f32 v222, v222, s0
	v_cvt_pk_bf16_f32 v219, v219, s0
	v_mul_lo_u32 v246, s67, v246
	v_cvt_pk_bf16_f32 v220, v220, s0
	buffer_store_b16 v244, v251, s[0:3], null offen
	v_cvt_pk_bf16_f32 v210, v210, s0
	v_cvt_pk_bf16_f32 v211, v211, s0
	v_cvt_pk_bf16_f32 v213, v213, s0
	v_cvt_pk_bf16_f32 v212, v212, s0
	v_cvt_pk_bf16_f32 v214, v214, s0
	v_add_lshl_u32 v246, s4, v246, 7
	v_cvt_pk_bf16_f32 v202, v202, s0
	v_cvt_pk_bf16_f32 v203, v203, s0
	v_cvt_pk_bf16_f32 v204, v204, s0
	v_cvt_pk_bf16_f32 v194, v194, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v252, v246, v109 /*v621*/
	v_cvt_pk_bf16_f32 v195, v195, s0
	v_cvt_pk_bf16_f32 v196, v196, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v248, 3, v1
	v_cvt_pk_bf16_f32 v186, v186, s0
	v_lshlrev_b32_e32 v243, 2, v252
	v_cvt_pk_bf16_f32 v187, v187, s0
	v_cvt_pk_bf16_f32 v188, v188, s0
	v_mul_lo_u32 v248, s67, v248
	v_cvt_pk_bf16_f32 v189, v189, s0
	buffer_store_b16 v247, v243, s[0:3], null offen
	v_cvt_pk_bf16_f32 v190, v190, s0
	v_cvt_pk_bf16_f32 v191, v191, s0
	v_cvt_pk_bf16_f32 v193, v193, s0
	v_cvt_pk_bf16_f32 v178, v178, s0
	v_cvt_pk_bf16_f32 v179, v179, s0
	v_add_lshl_u32 v248, s4, v248, 7
	v_cvt_pk_bf16_f32 v180, v180, s0
	v_cvt_pk_bf16_f32 v181, v181, s0
	v_cvt_pk_bf16_f32 v184, v184, s0
	v_cvt_pk_bf16_f32 v170, v170, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v244, v248, v109 /*v621*/
	v_cvt_pk_bf16_f32 v171, v171, s0
	v_cvt_pk_bf16_f32 v172, v172, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v249, 4, v1
	v_cvt_pk_bf16_f32 v162, v162, s0
	v_lshlrev_b32_e32 v244, 2, v244
	v_cvt_pk_bf16_f32 v164, v164, s0
	v_cvt_pk_bf16_f32 v163, v163, s0
	v_mul_lo_u32 v249, s67, v249
	v_cvt_pk_bf16_f32 v165, v165, s0
	v_cvt_pk_bf16_f32 v154, v154, s0
	v_cvt_pk_bf16_f32 v157, v157, s0
	v_cvt_pk_bf16_f32 v158, v158, s0
	v_cvt_pk_bf16_f32 v155, v155, s0
	v_cvt_pk_bf16_f32 v156, v156, s0
	v_cvt_pk_bf16_f32 v146, v146, s0
	v_add_lshl_u32 v249, s4, v249, 7
	v_cvt_pk_bf16_f32 v147, v147, s0
	v_cvt_pk_bf16_f32 v149, v149, s0
	v_cvt_pk_bf16_f32 v148, v148, s0
	v_cvt_pk_bf16_f32 v150, v150, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v254, v249, v109 /*v621*/
	v_cvt_pk_bf16_f32 v138, v138, s0
	v_cvt_pk_bf16_f32 v139, v139, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v253, 5, v1
	v_dual_lshlrev_b32 v254, 2, v254 :: v_dual_bitop2_b32 v249, v249, v0 bitop3:0x54
	v_cvt_pk_bf16_f32 v140, v140, s0
	v_cvt_pk_bf16_f32 v130, v130, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_lshlrev_b32 v249, 2, v249 :: v_dual_bitop2_b32 v242, v242, v0 bitop3:0x54
	v_mul_lo_u32 v252, s67, v253
	v_cvt_pk_bf16_f32 v131, v131, s0
	v_cvt_pk_bf16_f32 v122, v122, s0
	v_dual_lshlrev_b32 v242, 2, v242 :: v_dual_bitop2_b32 v247, 6, v1 bitop3:0x54
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v253, v221 /*v477*/, s0
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v253, v244, s[0:3], null offen
	buffer_store_b16 v255, v254, s[0:3], null offen
	v_mul_lo_u32 v247, s67, v247
	v_or_b32_e32 v246, v246, v0
	v_add_lshl_u32 v252, s4, v252, 7
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v255, v224 /*v480*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v123, v123, s0
	v_dual_lshlrev_b32 v246, 2, v246 :: v_dual_bitop2_b32 v1, 7, v1 bitop3:0x54
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v0 /*v256*/, v252, v109 /*v621*/
	s_set_vgpr_msb 0x4800
	v_add_lshl_u32 v247, s4, v247, 7
	v_cvt_pk_bf16_f32 v124, v124, s0
	v_cvt_pk_bf16_f32 v125, v125, s0
	v_or_b32_e32 v252, v252, v0
	v_mul_lo_u32 v1, s67, v1
	s_set_vgpr_msb 8
	v_or_b32_e32 v253, v247, v109 /*v621*/
	s_set_vgpr_msb 0x844
	v_lshlrev_b32_e32 v0 /*v256*/, 2, v0 /*v256*/
	s_set_vgpr_msb 0x4440
	v_or_b32_e32 v5 /*v261*/, 64, v246
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v126, v126, s0
	v_dual_lshlrev_b32 v253, 2, v253 :: v_dual_bitop2_b32 v245, v245, v0 bitop3:0x54
	s_set_vgpr_msb 0x41
	buffer_store_b16 v1 /*v257*/, v0 /*v256*/, s[0:3], null offen
	s_set_vgpr_msb 0x4100
	v_add_lshl_u32 v1, s4, v1, 7
	v_dual_lshlrev_b32 v245, 2, v245 :: v_dual_bitop2_b32 v247, v247, v0 bitop3:0x54
	v_cvt_pk_bf16_f32 v127, v127, s0
	v_cvt_pk_bf16_f32 v129, v129, s0
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v1 /*v257*/, v1, v109 /*v621*/
	s_set_vgpr_msb 0x4800
	v_dual_lshlrev_b32 v247, 2, v247 :: v_dual_bitop2_b32 v248, v248, v0 bitop3:0x54
	s_set_vgpr_msb 64
	v_or_b32_e32 v4 /*v260*/, 64, v245
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v1, v1, v0
	s_set_vgpr_msb 0x44
	v_lshlrev_b32_e32 v1 /*v257*/, 2, v1 /*v257*/
	s_set_vgpr_msb 0x4400
	s_clause 0x2
	buffer_store_b16 v255, v253, s[0:3], null offen
	s_set_vgpr_msb 0x41
	buffer_store_b16 v2 /*v258*/, v1 /*v257*/, s[0:3], null offen
	s_set_vgpr_msb 0x4101
	v_cvt_pk_bf16_f32 v255, v99 /*v355*/, s0
	v_lshlrev_b32_e32 v252, 2, v252
	s_set_vgpr_msb 0x141
	buffer_store_b16 v3 /*v259*/, v4 /*v260*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3 /*v259*/, v100 /*v356*/, s0
	v_or_b32_e32 v2 /*v258*/, 64, v242
	s_set_vgpr_msb 0x4100
	v_lshlrev_b32_e32 v1, 2, v1
	s_set_vgpr_msb 0x41
	v_cvt_pk_bf16_f32 v4 /*v260*/, v101 /*v357*/, s0
	s_set_vgpr_msb 0x4100
	v_lshlrev_b32_e32 v248, 2, v248
	v_cvt_pk_bf16_f32 v114, v114, s0
	s_set_vgpr_msb 1
	buffer_store_b16 v255, v2 /*v258*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v255, v102 /*v358*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v115, v115, s0
	s_set_vgpr_msb 0x41
	v_or_b32_e32 v6 /*v262*/, 64, v248
	buffer_store_b16 v3 /*v259*/, v5 /*v261*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3 /*v259*/, 64, v249
	v_cvt_pk_bf16_f32 v2 /*v258*/, v103 /*v359*/, s0
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v116, v116, s0
	s_set_vgpr_msb 0x41
	v_cvt_pk_bf16_f32 v5 /*v261*/, v104 /*v360*/, s0
	buffer_store_b16 v4 /*v260*/, v6 /*v262*/, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v4 /*v260*/, 64, v252
	s_set_vgpr_msb 0x4101
	s_clause 0x2
	buffer_store_b16 v255, v3 /*v259*/, s[0:3], null offen
	s_set_vgpr_msb 0x141
	buffer_store_b16 v2 /*v258*/, v4 /*v260*/, s[0:3], null offen
	s_set_vgpr_msb 0x4100
	v_or_b32_e32 v255, 64, v247
	v_cvt_pk_bf16_f32 v117, v117, s0
	s_set_vgpr_msb 0x41
	v_or_b32_e32 v4 /*v260*/, 64, v1
	v_mov_b16_e64 v2.l /*v258.l*/, v5.l /*v261.l*/
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v120, v120, s0
	s_set_vgpr_msb 0x41
	v_cvt_pk_bf16_f32 v3 /*v259*/, v105 /*v361*/, s0
	s_set_vgpr_msb 0x4100
	v_cvt_pk_bf16_f32 v106, v106, s0
	v_cvt_pk_bf16_f32 v107, v107, s0
	s_set_vgpr_msb 64
	s_clause 0x4
	buffer_store_b16 v2 /*v258*/, v255, s[0:3], null offen
	s_set_vgpr_msb 0x4041
	buffer_store_b16 v3 /*v259*/, v4 /*v260*/, s[0:3], null offen
	s_set_vgpr_msb 0x4100
	buffer_store_b16 v234, v250, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v234, v237, s0
	v_cvt_pk_bf16_f32 v237, v240, s0
	v_cvt_pk_bf16_f32 v108, v108, s0
	s_clause 0x1
	buffer_store_b16 v235, v251, s[0:3], null offen offset:128
	buffer_store_b16 v236, v243, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v235, v238, s0
	v_cvt_pk_bf16_f32 v238, v241, s0
	v_cvt_pk_bf16_f32 v98, v98, s0
	buffer_store_b16 v234, v244, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v234.l, v237.l
	v_cvt_pk_bf16_f32 v100, v100, s0
	v_cvt_pk_bf16_f32 v236, v239, s0
	v_mov_b16_e64 v237.l, v238.l
	s_clause 0x2
	buffer_store_b16 v235, v254, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	buffer_store_b16 v236, v0 /*v256*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_or_b32_e32 v236, 0xc0, v246
	s_set_vgpr_msb 0x100
	s_clause 0x2
	buffer_store_b16 v234, v253, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	buffer_store_b16 v237, v1 /*v257*/, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v234, 0xc0, v245
	v_or_b32_e32 v235, 0xc0, v242
	s_wait_xcnt 0x0
	v_or_b32_e32 v237, 0xc0, v248
	s_set_vgpr_msb 0x100
	s_clause 0x1
	buffer_store_b16 v226, v234, s[0:3], null offen
	buffer_store_b16 v227, v235, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v226, v230, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v227, v231, s0
	v_or_b32_e32 v230, 0xc0, v252
	s_clause 0x1
	buffer_store_b16 v228, v236, s[0:3], null offen
	buffer_store_b16 v229, v237, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v228, 0xc0, v249
	v_or_b32_e32 v231, 0xc0, v247
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v229, v232, s0
	v_cvt_pk_bf16_f32 v232, v233, s0
	s_clause 0x1
	buffer_store_b16 v226, v228, s[0:3], null offen
	buffer_store_b16 v227, v230, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v226, 0xc0, v1
	buffer_store_b16 v229, v231, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v227.l, v232.l
	v_cvt_pk_bf16_f32 v99, v99, s0
	v_cvt_pk_bf16_f32 v101, v101, s0
	v_cvt_pk_bf16_f32 v90, v90, s0
	v_cvt_pk_bf16_f32 v93, v93, s0
	s_clause 0x3
	buffer_store_b16 v227, v226, s[0:3], null offen
	buffer_store_b16 v218, v250, s[0:3], null offen offset:256
	buffer_store_b16 v219, v251, s[0:3], null offen offset:256
	buffer_store_b16 v220, v243, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e64 v218.l, v222.l
	buffer_store_b16 v221, v244, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v219, v223, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v220, v224, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v221, v225, s0
	buffer_store_b16 v218, v254, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v94, v94, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v218.l, v219.l
	v_mov_b16_e64 v219.l, v220.l
	v_mov_b16_e64 v220.l, v221.l
	v_or_b32_e32 v221, 0x140, v245
	s_set_vgpr_msb 1
	s_clause 0x6
	buffer_store_b16 v218, v0 /*v256*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x100
	buffer_store_b16 v219, v253, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	buffer_store_b16 v220, v1 /*v257*/, s[0:3], null offen offset:256
	s_set_vgpr_msb 0x100
	buffer_store_b16 v210, v221, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v210, 0x140, v242
	v_or_b32_e32 v219, 0x140, v248
	v_or_b32_e32 v218, 0x140, v246
	v_or_b32_e32 v220, 0x140, v249
	s_clause 0x1
	buffer_store_b16 v211, v210, s[0:3], null offen
	buffer_store_b16 v212, v218, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v210, v215, s0
	s_clause 0x1
	buffer_store_b16 v213, v219, s[0:3], null offen
	buffer_store_b16 v214, v220, s[0:3], null offen
	v_or_b32_e32 v211, 0x140, v252
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v212, v216, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v213, v217, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v214, 0x140, v247
	v_or_b32_e32 v215, 0x140, v1
	buffer_store_b16 v210, v211, s[0:3], null offen
	v_or_b32_e32 v1, 0x1c0, v1
	v_cvt_pk_bf16_f32 v91, v91, s0
	s_clause 0x2
	buffer_store_b16 v212, v214, s[0:3], null offen
	buffer_store_b16 v213, v215, s[0:3], null offen
	buffer_store_b16 v202, v250, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v202, v205, s0
	v_cvt_pk_bf16_f32 v205, v206, s0
	v_cvt_pk_bf16_f32 v206, v207, s0
	s_clause 0x2
	buffer_store_b16 v203, v251, s[0:3], null offen offset:384
	buffer_store_b16 v204, v243, s[0:3], null offen offset:384
	buffer_store_b16 v202, v244, s[0:3], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e64 v203.l, v205.l
	s_wait_xcnt 0x1
	v_mov_b16_e64 v204.l, v206.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v202, v208, s0
	v_or_b32_e32 v205, 0x1c0, v242
	v_cvt_pk_bf16_f32 v92, v92, s0
	s_clause 0x2
	buffer_store_b16 v203, v254, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	buffer_store_b16 v204, v0 /*v256*/, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_or_b32_e32 v204, 0x1c0, v245
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v203, v209, s0
	s_clause 0x5
	buffer_store_b16 v202, v253, s[0:3], null offen offset:384
	s_set_vgpr_msb 1
	buffer_store_b16 v203, v1 /*v257*/, s[0:3], null offen offset:384
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v204, s[0:3], null offen
	buffer_store_b16 v195, v205, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v195, 0x1c0, v246
	s_set_vgpr_msb 8
	v_or_b32_e32 v203, s81, v112 /*v624*/
	v_cvt_pk_bf16_f32 v194, v197, s0
	v_cvt_pk_bf16_f32 v197, v198, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v198, 0x1c0, v248
	s_clause 0x1
	buffer_store_b16 v196, v195, s[0:3], null offen
	buffer_store_b16 v194, v198, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mul_lo_u32 v195, s67, v203
	v_or_b32_e32 v202, 0x1c0, v249
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v194, v199, s0
	v_cvt_pk_bf16_f32 v196, v200, s0
	v_or_b32_e32 v198, 0x1c0, v247
	v_cvt_pk_bf16_f32 v82, v82, s0
	buffer_store_b16 v197, v202, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v197, 0x1c0, v252
	v_add_lshl_u32 v195, s4, v195, 7
	v_cvt_pk_bf16_f32 v83, v83, s0
	v_cvt_pk_bf16_f32 v85, v85, s0
	v_cvt_pk_bf16_f32 v84, v84, s0
	s_clause 0x1
	buffer_store_b16 v194, v197, s[0:3], null offen
	buffer_store_b16 v196, v198, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v196, v195, v109 /*v621*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v199, 1, v203
	v_or_b32_e32 v198, 2, v203
	v_cvt_pk_bf16_f32 v194, v201, s0
	v_cvt_pk_bf16_f32 v86, v86, s0
	v_lshlrev_b32_e32 v196, 2, v196
	v_mul_lo_u32 v197, s67, v199
	v_mul_lo_u32 v198, s67, v198
	buffer_store_b16 v194, v1, s[0:3], null offen
	v_or_b32_e32 v199, 3, v203
	buffer_store_b16 v186, v196, s[0:3], null offen
	v_cvt_pk_bf16_f32 v74, v74, s0
	v_cvt_pk_bf16_f32 v75, v75, s0
	v_cvt_pk_bf16_f32 v76, v76, s0
	v_add_lshl_u32 v197, s4, v197, 7
	s_wait_xcnt 0x0
	v_add_lshl_u32 v186, s4, v198, 7
	v_mul_lo_u32 v194, s67, v199
	v_cvt_pk_bf16_f32 v66, v66, s0
	v_cvt_pk_bf16_f32 v67, v67, s0
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, v197, v109 /*v621*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v197, v197, v0
	v_or_b32_e32 v198, 4, v203
	s_set_vgpr_msb 8
	v_or_b32_e32 v199, v186, v109 /*v621*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v186, v186, v0
	v_dual_lshlrev_b32 v1, 2, v1 :: v_dual_bitop2_b32 v200, 5, v203 bitop3:0x54
	v_mul_lo_u32 v198, s67, v198
	v_add_lshl_u32 v194, s4, v194, 7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b32_e32 v186, 2, v186
	v_mul_lo_u32 v200, s67, v200
	buffer_store_b16 v187, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v187, 2, v199
	s_set_vgpr_msb 8
	v_or_b32_e32 v199, v194, v109 /*v621*/
	v_cvt_pk_bf16_f32 v58, v58, s0
	v_add_lshl_u32 v198, v198, s4, 7
	v_cvt_pk_bf16_f32 v59, v59, s0
	buffer_store_b16 v188, v187, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v188, 6, v203
	v_add_lshl_u32 v200, s4, v200, 7
	s_set_vgpr_msb 8
	v_or_b32_e32 v201, v198, v109 /*v621*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v198, v198, v0
	v_or_b32_e32 v202, 7, v203
	v_mul_lo_u32 v188, s67, v188
	s_set_vgpr_msb 8
	v_or_b32_e32 v203, v200, v109 /*v621*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v199, 2, v199 :: v_dual_lshlrev_b32 v201, 2, v201
	v_mul_lo_u32 v202, s67, v202
	s_clause 0x1
	buffer_store_b16 v189, v199, s[0:3], null offen
	buffer_store_b16 v190, v201, s[0:3], null offen
	v_lshlrev_b32_e32 v203, 2, v203
	v_add_lshl_u32 v188, s4, v188, 7
	v_cvt_pk_bf16_f32 v60, v60, s0
	v_cvt_pk_bf16_f32 v61, v61, s0
	s_wait_xcnt 0x0
	v_add_lshl_u32 v190, s4, v202, 7
	buffer_store_b16 v191, v203, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v191, v192, s0
	v_or_b32_e32 v192, v195, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v189, v188, v109 /*v621*/
	v_or_b32_e32 v195, v190, v109 /*v621*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v188, v188, v0
	v_or_b32_e32 v190, v190, v0
	v_dual_lshlrev_b32 v192, 2, v192 :: v_dual_lshlrev_b32 v189, 2, v189
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_lshlrev_b32 v195, 2, v195 :: v_dual_lshlrev_b32 v188, 2, v188
	v_cvt_pk_bf16_f32 v62, v62, s0
	v_or_b32_e32 v202, 64, v192
	s_clause 0x1
	buffer_store_b16 v191, v189, s[0:3], null offen
	buffer_store_b16 v193, v195, s[0:3], null offen
	s_wait_xcnt 0x1
	v_lshlrev_b32_e32 v191, 2, v197
	v_cvt_pk_bf16_f32 v63, v63, s0
	v_cvt_pk_bf16_f32 v65, v65, s0
	buffer_store_b16 v178, v202, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v178, v194, v0
	v_or_b32_e32 v193, 64, v191
	v_or_b32_e32 v194, 64, v186
	v_cvt_pk_bf16_f32 v50, v50, s0
	v_cvt_pk_bf16_f32 v51, v51, s0
	v_lshlrev_b32_e32 v178, 2, v178
	buffer_store_b16 v179, v193, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v179, v200, v0
	buffer_store_b16 v180, v194, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v180, 2, v198
	v_or_b32_e32 v197, 64, v178
	v_cvt_pk_bf16_f32 v52, v52, s0
	v_lshlrev_b32_e32 v179, 2, v179
	v_cvt_pk_bf16_f32 v53, v53, s0
	v_cvt_pk_bf16_f32 v56, v56, s0
	buffer_store_b16 v181, v197, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v181, v182, s0
	v_cvt_pk_bf16_f32 v182, v183, s0
	v_or_b32_e32 v183, 64, v180
	v_or_b32_e32 v193, 64, v179
	s_clause 0x1
	buffer_store_b16 v181, v183, s[0:3], null offen
	buffer_store_b16 v182, v193, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v181, 2, v190 :: v_dual_bitop2_b32 v182, 64, v188 bitop3:0x54
	v_mov_b16_e64 v183.l, v184.l
	v_cvt_pk_bf16_f32 v184, v185, s0
	v_cvt_pk_bf16_f32 v42, v42, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v185, 64, v181
	v_cvt_pk_bf16_f32 v43, v43, s0
	s_clause 0x2
	buffer_store_b16 v183, v182, s[0:3], null offen
	buffer_store_b16 v184, v185, s[0:3], null offen
	buffer_store_b16 v170, v196, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v170, v173, s0
	v_cvt_pk_bf16_f32 v173, v176, s0
	s_clause 0x1
	buffer_store_b16 v171, v1, s[0:3], null offen offset:128
	buffer_store_b16 v172, v187, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v171, v174, s0
	v_cvt_pk_bf16_f32 v174, v177, s0
	buffer_store_b16 v170, v199, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v173.l
	v_cvt_pk_bf16_f32 v172, v175, s0
	s_clause 0x1
	buffer_store_b16 v171, v201, s[0:3], null offen offset:128
	buffer_store_b16 v172, v203, s[0:3], null offen offset:128
	v_mov_b16_e64 v173.l, v174.l
	s_clause 0x1
	buffer_store_b16 v170, v189, s[0:3], null offen offset:128
	buffer_store_b16 v173, v195, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_or_b32_e32 v170, 0xc0, v192
	v_or_b32_e32 v172, 0xc0, v186
	v_or_b32_e32 v171, 0xc0, v191
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
	v_or_b32_e32 v167, 0xc0, v188
	s_clause 0x1
	buffer_store_b16 v162, v164, s[0:3], null offen
	buffer_store_b16 v163, v166, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v162, 0xc0, v181
	s_wait_xcnt 0x0
	v_mov_b16_e64 v163.l, v168.l
	buffer_store_b16 v165, v167, s[0:3], null offen
	v_cvt_pk_bf16_f32 v44, v44, s0
	v_cvt_pk_bf16_f32 v34, v34, s0
	v_cvt_pk_bf16_f32 v36, v36, s0
	s_clause 0x3
	buffer_store_b16 v163, v162, s[0:3], null offen
	buffer_store_b16 v154, v196, s[0:3], null offen offset:256
	buffer_store_b16 v155, v1, s[0:3], null offen offset:256
	buffer_store_b16 v156, v187, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e64 v154.l, v158.l
	buffer_store_b16 v157, v199, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v155, v159, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v156, v160, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v157, v161, s0
	buffer_store_b16 v154, v201, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v35, v35, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v154.l, v155.l
	v_mov_b16_e64 v155.l, v156.l
	v_mov_b16_e64 v156.l, v157.l
	v_or_b32_e32 v157, 0x140, v192
	s_clause 0x3
	buffer_store_b16 v154, v203, s[0:3], null offen offset:256
	buffer_store_b16 v155, v189, s[0:3], null offen offset:256
	buffer_store_b16 v156, v195, s[0:3], null offen offset:256
	buffer_store_b16 v146, v157, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v146, 0x140, v191
	v_or_b32_e32 v155, 0x140, v178
	v_or_b32_e32 v154, 0x140, v186
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
	v_or_b32_e32 v150, 0x140, v188
	v_or_b32_e32 v151, 0x140, v181
	buffer_store_b16 v146, v147, s[0:3], null offen
	v_cvt_pk_bf16_f32 v37, v37, s0
	v_cvt_pk_bf16_f32 v26, v26, s0
	s_clause 0x2
	buffer_store_b16 v148, v150, s[0:3], null offen
	buffer_store_b16 v149, v151, s[0:3], null offen
	buffer_store_b16 v138, v196, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v138, v141, s0
	v_cvt_pk_bf16_f32 v141, v142, s0
	v_cvt_pk_bf16_f32 v142, v143, s0
	s_clause 0x2
	buffer_store_b16 v139, v1, s[0:3], null offen offset:384
	buffer_store_b16 v140, v187, s[0:3], null offen offset:384
	buffer_store_b16 v138, v199, s[0:3], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e64 v1.l, v141.l
	v_mov_b16_e64 v139.l, v142.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v138, v144, s0
	v_or_b32_e32 v140, 0x1c0, v191
	v_cvt_pk_bf16_f32 v29, v29, s0
	s_clause 0x1
	buffer_store_b16 v1, v201, s[0:3], null offen offset:384
	buffer_store_b16 v139, v203, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_or_b32_e32 v139, 0x1c0, v192
	v_cvt_pk_bf16_f32 v1, v145, s0
	s_clause 0x1
	buffer_store_b16 v138, v189, s[0:3], null offen offset:384
	buffer_store_b16 v1, v195, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v132, s0
	s_clause 0x1
	buffer_store_b16 v130, v139, s[0:3], null offen
	buffer_store_b16 v131, v140, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v131, 0x1c0, v186
	s_set_vgpr_msb 8
	v_or_b32_e32 v138, s80, v112 /*v624*/
	v_cvt_pk_bf16_f32 v130, v133, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v133, 0x1c0, v178
	v_cvt_pk_bf16_f32 v132, v134, s0
	v_or_b32_e32 v134, 0x1c0, v180
	s_clause 0x1
	buffer_store_b16 v1, v131, s[0:3], null offen
	buffer_store_b16 v130, v133, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v130, s67, v138
	v_cvt_pk_bf16_f32 v1, v135, s0
	buffer_store_b16 v132, v134, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v132, 0x1c0, v179
	v_or_b32_e32 v134, 1, v138
	v_cvt_pk_bf16_f32 v131, v136, s0
	v_or_b32_e32 v133, 0x1c0, v188
	v_or_b32_e32 v135, 3, v138
	v_add_lshl_u32 v130, s4, v130, 7
	buffer_store_b16 v1, v132, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v132, s67, v134
	v_or_b32_e32 v134, 2, v138
	buffer_store_b16 v131, v133, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v131, v130, v109 /*v621*/
	v_cvt_pk_bf16_f32 v1, v137, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v133, 0x1c0, v181
	v_mul_lo_u32 v134, s67, v134
	v_dual_lshlrev_b32 v131, 2, v131 :: v_dual_bitop2_b32 v136, 5, v138 bitop3:0x54
	v_add_lshl_u32 v132, s4, v132, 7
	buffer_store_b16 v1, v133, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v133, s67, v135
	v_mul_lo_u32 v136, s67, v136
	buffer_store_b16 v122, v131, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, v132, v109 /*v621*/
	v_add_lshl_u32 v122, v134, s4, 7
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v134, 4, v138
	v_cvt_pk_bf16_f32 v30, v30, s0
	v_cvt_pk_bf16_f32 v27, v27, s0
	v_lshlrev_b32_e32 v1, 2, v1
	s_set_vgpr_msb 8
	v_or_b32_e32 v135, v122, v109 /*v621*/
	v_mul_lo_u32 v134, v134, s67
	v_add_lshl_u32 v133, v133, s4, 7
	v_add_lshl_u32 v136, v136, s4, 7
	buffer_store_b16 v123, v1, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v123, 2, v135 :: v_dual_bitop2_b32 v122, v122, v0 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v135, v133, v109 /*v621*/
	v_or_b32_e32 v139, v136, v109 /*v621*/
	v_add_lshl_u32 v134, v134, s4, 7
	buffer_store_b16 v124, v123, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v124, 6, v138
	v_dual_lshlrev_b32 v135, 2, v135 :: v_dual_bitop2_b32 v138, 7, v138 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v137, v134, v109 /*v621*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v139, 2, v139
	v_mul_lo_u32 v124, s67, v124
	v_mul_lo_u32 v138, s67, v138
	v_dual_lshlrev_b32 v137, 2, v137 :: v_dual_bitop2_b32 v134, v134, v0 bitop3:0x54
	s_clause 0x1
	buffer_store_b16 v125, v135, s[0:3], null offen
	buffer_store_b16 v126, v137, s[0:3], null offen
	v_add_lshl_u32 v124, s4, v124, 7
	s_wait_xcnt 0x0
	v_add_lshl_u32 v126, s4, v138, 7
	buffer_store_b16 v127, v139, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v127, v128, s0
	v_or_b32_e32 v128, v130, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v125, v124, v109 /*v621*/
	v_or_b32_e32 v130, v126, v109 /*v621*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v132, v132, v0
	v_or_b32_e32 v126, v126, v0
	v_dual_lshlrev_b32 v128, 2, v128 :: v_dual_lshlrev_b32 v125, 2, v125
	v_lshlrev_b32_e32 v130, 2, v130
	s_clause 0x1
	buffer_store_b16 v127, v125, s[0:3], null offen
	buffer_store_b16 v129, v130, s[0:3], null offen
	v_or_b32_e32 v138, 64, v128
	s_wait_xcnt 0x1
	v_dual_lshlrev_b32 v127, 2, v132 :: v_dual_lshlrev_b32 v122, 2, v122
	v_or_b32_e32 v124, v124, v0
	v_cvt_pk_bf16_f32 v28, v28, s0
	buffer_store_b16 v114, v138, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v114, v133, v0
	v_or_b32_e32 v129, 64, v127
	v_dual_lshlrev_b32 v124, 2, v124 :: v_dual_bitop2_b32 v132, 64, v122 bitop3:0x54
	v_cvt_pk_bf16_f32 v18, v18, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v114, 2, v114
	buffer_store_b16 v115, v129, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v115, v136, v0
	buffer_store_b16 v116, v132, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v116, 2, v134
	v_or_b32_e32 v133, 64, v114
	v_cvt_pk_bf16_f32 v19, v19, s0
	v_lshlrev_b32_e32 v115, 2, v115
	v_cvt_pk_bf16_f32 v21, v21, s0
	v_cvt_pk_bf16_f32 v22, v22, s0
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
	v_cvt_pk_bf16_f32 v23, v23, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v121, 64, v117
	v_cvt_pk_bf16_f32 v12, v12, s0
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
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_cvt_pk_bf16_f32 v3, v3, s0
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
	v_cvt_pk_bf16_f32 v2, v2, s0
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
	v_cvt_pk_bf16_f32 v4, v4, s0
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
	s_set_vgpr_msb 8
	v_or_b32_e32 v74, s69, v112 /*v624*/
	v_cvt_pk_bf16_f32 v66, v69, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v69, 0x1c0, v114
	v_cvt_pk_bf16_f32 v68, v70, s0
	v_or_b32_e32 v70, 0x1c0, v116
	s_clause 0x1
	buffer_store_b16 v1, v67, s[0:3], null offen
	buffer_store_b16 v66, v69, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v66, s67, v74
	v_cvt_pk_bf16_f32 v1, v71, s0
	buffer_store_b16 v68, v70, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v68, 0x1c0, v115
	v_or_b32_e32 v70, 1, v74
	v_cvt_pk_bf16_f32 v67, v72, s0
	v_or_b32_e32 v69, 0x1c0, v124
	v_or_b32_e32 v71, 3, v74
	v_add_lshl_u32 v66, s4, v66, 7
	buffer_store_b16 v1, v68, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v68, s67, v70
	v_or_b32_e32 v70, 2, v74
	buffer_store_b16 v67, v69, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v67, v66, v109 /*v621*/
	v_cvt_pk_bf16_f32 v1, v73, s0
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v69, 0x1c0, v117
	v_mul_lo_u32 v70, s67, v70
	v_dual_lshlrev_b32 v67, 2, v67 :: v_dual_bitop2_b32 v72, 5, v74 bitop3:0x54
	v_add_lshl_u32 v68, s4, v68, 7
	buffer_store_b16 v1, v69, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mul_lo_u32 v69, s67, v71
	v_mul_lo_u32 v72, s67, v72
	buffer_store_b16 v58, v67, s[0:3], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, v68, v109 /*v621*/
	v_add_lshl_u32 v58, v70, s4, 7
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v70, 4, v74
	v_or_b32_e32 v68, v68, v0
	v_lshlrev_b32_e32 v1, 2, v1
	s_set_vgpr_msb 8
	v_or_b32_e32 v71, v58, v109 /*v621*/
	v_mul_lo_u32 v70, v70, s67
	v_add_lshl_u32 v69, v69, s4, 7
	v_add_lshl_u32 v72, v72, s4, 7
	buffer_store_b16 v59, v1, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v59, 2, v71 :: v_dual_bitop2_b32 v58, v58, v0 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v71, v69, v109 /*v621*/
	v_or_b32_e32 v75, v72, v109 /*v621*/
	v_add_lshl_u32 v70, v70, s4, 7
	buffer_store_b16 v60, v59, s[0:3], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v60, 6, v74
	v_dual_lshlrev_b32 v71, 2, v71 :: v_dual_bitop2_b32 v74, 7, v74 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v73, v70, v109 /*v621*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v75, 2, v75
	v_mul_lo_u32 v60, s67, v60
	v_mul_lo_u32 v74, s67, v74
	v_dual_lshlrev_b32 v58, 2, v58 :: v_dual_lshlrev_b32 v73, 2, v73
	s_clause 0x1
	buffer_store_b16 v61, v71, s[0:3], null offen
	buffer_store_b16 v62, v73, s[0:3], null offen
	v_add_lshl_u32 v60, s4, v60, 7
	s_wait_xcnt 0x0
	v_add_lshl_u32 v62, s4, v74, 7
	buffer_store_b16 v63, v75, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v63, v64, s0
	v_or_b32_e32 v64, v66, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v61, v60, v109 /*v621*/
	v_or_b32_e32 v66, v62, v109 /*v621*/
	s_set_vgpr_msb 0x800
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
		.amdhsa_next_free_vgpr 812
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

	.set .Lk_dq_0.num_vgpr, 812
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
    .vgpr_count:     812
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
