	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	k_dkdv_0
	.p2align	8
	.type	k_dkdv_0,@function
k_dkdv_0:
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	v_nop
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_bfe_u32 s2, ttmp6, 0x40010
	s_load_b256 s[36:43], s[0:1], 0x170 nv
	s_and_b32 s3, ttmp7, 0xffff
	s_add_co_i32 s2, s2, 1
	s_bfe_u32 s5, ttmp6, 0x40014
	s_mul_i32 s2, s3, s2
	s_bfe_u32 s4, ttmp6, 0x40004
	s_lshr_b32 s6, ttmp7, 16
	s_add_co_i32 s5, s5, 1
	s_add_co_i32 s4, s4, s2
	s_mul_i32 s2, s6, s5
	s_bfe_u32 s5, ttmp6, 0x40008
	s_getreg_b32 s7, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s5, s5, s2
	s_cmp_eq_u32 s7, 0
	s_set_vgpr_msb 64
	v_dual_lshrrev_b32 v255 /*v511*/, 4, v0 :: v_dual_bitop2_b32 v253 /*v509*/, 15, v0 bitop3:0x40
	s_cselect_b32 s8, s6, s5
	s_cselect_b32 s4, s3, s4
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_wait_kmcnt 0x0
	s_mul_i32 s67, s38, s8
	s_mul_i32 s2, ttmp9, s2
	s_load_b64 s[44:45], s[0:1], 0x30 nv
	s_add_co_i32 s5, s3, s2
	s_cmp_eq_u32 s7, 0
	s_load_b64 s[2:3], s[0:1], 0x190 nv
	s_cselect_b32 s66, ttmp9, s5
	s_lshr_b32 s5, s42, 31
	s_lshl_b32 s9, s4, 5
	s_add_co_i32 s5, s42, s5
	s_set_vgpr_msb 0x4004
	v_or_b32_e32 v1, s9, v253 /*v509*/
	s_and_b32 s4, s5, -2
	s_ashr_i32 s5, s5, 1
	s_cmp_lg_u32 s42, s4
	s_clause 0x1
	s_load_b64 s[48:49], s[0:1], 0x90 nv
	s_load_b64 s[52:53], s[0:1], 0x0 nv
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s42, 0
	s_mul_i32 s68, s41, s66
	s_cselect_b32 s6, -1, 0
	s_mul_i32 s69, s39, s8
	s_and_b32 s4, s6, s4
	s_sub_co_ci_u32 s6, s5, 0
	s_sub_co_i32 s7, s9, s43
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_max_i32 s7, s7, 0
	s_lshr_b32 s7, s7, 5
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s10, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_and_b32 s10, s10, exec_lo
	s_cselect_b32 s71, s7, 0
	s_cmp_lg_u32 s4, 0
	s_sub_co_ci_u32 s72, s5, s71
	s_or_b32 s4, s9, 31
	s_sub_co_i32 s4, s4, s43
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s5, s4, 31
	s_ashr_i32 s7, s5, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s7, s7, 27
	s_add_co_i32 s7, s5, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s10, s7, 0xffffffe0
	s_ashr_i32 s7, s7, 5
	s_cmp_lg_u32 s5, s10
	s_cselect_b32 s10, -1, 0
	s_cmp_lt_i32 s5, 0
	s_cselect_b32 s5, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_and_b32 s5, s5, s10
	s_sub_co_ci_u32 s5, s7, 0
	s_cmp_gt_i32 s4, -1
	s_mul_i32 s10, s39, s37
	s_cselect_b32 s4, s5, 0
	s_min_i32 s4, s4, s6
	s_mul_i32 s6, s38, s40
	s_sub_co_i32 s4, s4, s71
	s_mul_i32 s6, s6, s3
	s_max_i32 s4, s4, 0
	s_mul_i32 s3, s10, s3
	s_min_i32 s4, s4, s72
	s_cmp_lg_u32 s2, 0
	s_mov_b32 s10, 0
	s_cselect_b32 s74, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s2, s74, exec_lo
	s_cselect_b32 s73, s4, 0
	s_lshl_b32 s4, s40, 4
	s_or_b32 s2, s9, 16
	s_mul_i32 s5, s4, s67
	v_or_b32_e32 v2, s2, v253 /*v509*/
	s_lshl4_add_u32 s5, s66, s5
	s_lshl_b32 s34, s6, 8
	v_mad_u32 v1, v1, s4, s5
	s_ashr_i32 s35, s34, 31
	v_mad_u32 v2, v2, s4, s5
	s_load_b64 s[4:5], s[0:1], 0x60 nv
	s_lshr_b64 s[46:47], s[34:35], 7
	s_lshl_b32 s12, s3, 8
	s_mov_b32 s6, s46
	s_mov_b32 s7, s47
	v_or_b32_e32 v1, v1, v255 /*v511*/
	s_lshl_b32 s14, s3, 2
	v_or_b32_e32 v2, v2, v255 /*v511*/
	s_lshl_b32 s35, s39, 4
	s_ashr_i32 s13, s12, 31
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v1, 4, v1
	s_ashr_i32 s15, s14, 31
	v_lshlrev_b32_e32 v2, 4, v2
	s_clause 0x6
	buffer_load_b128 v[242:245], v1, s[44:47], null offen
	buffer_load_b128 v[246:249], v1, s[44:47], null offen offset:32
	buffer_load_b128 v[250:253], v1, s[44:47], null offen offset:64
	buffer_load_b128 v[254:257], v1, s[44:47], null offen offset:96
	s_set_vgpr_msb 64
	buffer_load_b128 v[2:5] /*v[258:261]*/, v1, s[44:47], null offen offset:128
	buffer_load_b128 v[6:9] /*v[262:265]*/, v1, s[44:47], null offen offset:160
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v3, 0xe0, v1
	v_add_nc_u32_e32 v4, 0xe0, v2
	s_set_vgpr_msb 64
	s_clause 0x5
	buffer_load_b128 v[42:45] /*v[298:301]*/, v2, s[44:47], null offen offset:128
	buffer_load_b128 v[46:49] /*v[302:305]*/, v2, s[44:47], null offen offset:160
	buffer_load_b128 v[50:53] /*v[306:309]*/, v1, s[44:47], null offen offset:192
	buffer_load_b128 v[54:57] /*v[310:313]*/, v3, s[44:47], null offen
	buffer_load_b128 v[58:61] /*v[314:317]*/, v2, s[44:47], null offen offset:192
	buffer_load_b128 v[62:65] /*v[318:321]*/, v4, s[44:47], null offen
	s_wait_kmcnt 0x0
	s_clause 0xf
	buffer_load_b128 v[66:69] /*v[322:325]*/, v1, s[4:7], null offen offset:64
	buffer_load_b128 v[70:73] /*v[326:329]*/, v1, s[4:7], null offen offset:96
	buffer_load_b128 v[74:77] /*v[330:333]*/, v1, s[4:7], null offen offset:128
	buffer_load_b128 v[78:81] /*v[334:337]*/, v1, s[4:7], null offen offset:160
	buffer_load_b128 v[82:85] /*v[338:341]*/, v1, s[4:7], null offen offset:192
	buffer_load_b128 v[86:89] /*v[342:345]*/, v3, s[4:7], null offen
	buffer_load_b128 v[90:93] /*v[346:349]*/, v2, s[4:7], null offen
	buffer_load_b128 v[94:97] /*v[350:353]*/, v2, s[4:7], null offen offset:32
	buffer_load_b128 v[98:101] /*v[354:357]*/, v2, s[4:7], null offen offset:64
	buffer_load_b128 v[102:105] /*v[358:361]*/, v2, s[4:7], null offen offset:96
	buffer_load_b128 v[106:109] /*v[362:365]*/, v2, s[4:7], null offen offset:128
	buffer_load_b128 v[110:113] /*v[366:369]*/, v2, s[4:7], null offen offset:160
	buffer_load_b128 v[114:117] /*v[370:373]*/, v2, s[4:7], null offen offset:192
	buffer_load_b128 v[118:121] /*v[374:377]*/, v4, s[4:7], null offen
	buffer_load_b128 v[18:21] /*v[274:277]*/, v1, s[4:7], null offen
	buffer_load_b128 v[22:25] /*v[278:281]*/, v1, s[4:7], null offen offset:32
	s_clause 0x3
	buffer_load_b128 v[26:29] /*v[282:285]*/, v2, s[44:47], null offen
	buffer_load_b128 v[30:33] /*v[286:289]*/, v2, s[44:47], null offen offset:32
	buffer_load_b128 v[34:37] /*v[290:293]*/, v2, s[44:47], null offen offset:64
	buffer_load_b128 v[38:41] /*v[294:297]*/, v2, s[44:47], null offen offset:96
	s_wait_xcnt 0x4
	s_clause 0x1
	s_load_b64 s[4:5], s[0:1], 0xc0 nv
	s_load_b64 s[6:7], s[0:1], 0xe8 nv
	s_set_vgpr_msb 0x4004
	v_lshlrev_b32_e32 v2, 3, v255 /*v511*/
	s_set_vgpr_msb 0x400
	v_and_b32_e32 v1, 48, v0
	s_set_vgpr_msb 0x80
	v_lshrrev_b32_e32 v5 /*v517*/, 3, v0
	s_set_vgpr_msb 0x8000
	v_bfe_u32 v3, v0, 3, 1
	s_lshl_b32 s11, s3, 27
	s_set_vgpr_msb 0x80
	v_and_or_b32 v4 /*v516*/, v0, 7, v2
	v_or_b32_e32 v0 /*v512*/, s9, v2
	s_set_vgpr_msb 0x8040
	v_add_nc_u32_e32 v254 /*v510*/, s2, v2
	s_set_vgpr_msb 0x4084
	v_mad_u32_u24 v1 /*v513*/, 0x110, v253 /*v509*/, v1
	s_set_vgpr_msb 0x8480
	v_lshlrev_b32_e32 v2 /*v514*/, 4, v3
	s_set_vgpr_msb 0x8088
	v_mul_u32_u24_e32 v3 /*v515*/, 0x50, v4 /*v516*/
	s_set_vgpr_msb 0x8848
	v_or_b32_e32 v251 /*v507*/, 3, v0 /*v512*/
	v_or_b32_e32 v252 /*v508*/, 2, v0 /*v512*/
	v_or_b32_e32 v249 /*v505*/, 5, v0 /*v512*/
	v_or_b32_e32 v250 /*v506*/, 4, v0 /*v512*/
	v_or_b32_e32 v247 /*v503*/, 7, v0 /*v512*/
	v_or_b32_e32 v248 /*v504*/, 6, v0 /*v512*/
	s_set_vgpr_msb 0x4844
	v_or_b32_e32 v245 /*v501*/, 3, v254 /*v510*/
	v_or_b32_e32 v246 /*v502*/, 2, v254 /*v510*/
	v_or_b32_e32 v243 /*v499*/, 5, v254 /*v510*/
	v_or_b32_e32 v244 /*v500*/, 4, v254 /*v510*/
	s_set_vgpr_msb 0x4404
	v_or_b32_e32 v1, 7, v254 /*v510*/
	s_set_vgpr_msb 0x444
	v_or_b32_e32 v242 /*v498*/, 6, v254 /*v510*/
	s_mul_i32 s38, s73, s41
	s_mul_i32 s70, s37, s35
	s_lshr_b64 s[54:55], s[12:13], 7
	s_lshr_b64 s[58:59], s[14:15], 7
	s_wait_kmcnt 0x0
	s_or_b64 s[56:57], s[4:5], s[10:11]
	s_or_b64 s[60:61], s[6:7], s[10:11]
	s_cmp_lt_i32 s38, 1
	s_mul_i32 s70, s70, s8
	s_set_vgpr_msb 0x4400
	s_cbranch_scc1 .LBB0_3
	s_abs_i32 s75, s41
	s_movk_i32 s3, 0x1400
	s_cvt_f32_u32 s2, s75
	v_dual_lshlrev_b32 v2, 1, v0 :: v_dual_mov_b32 v234, 0
	s_set_vgpr_msb 0x88
	v_mad_u32_u24 v13 /*v525*/, 0x110, v4 /*v516*/, s3
	v_s_rcp_f32 s2, s2
	s_sub_co_i32 s3, 0, s75
	s_movk_i32 s4, 0x3600
	s_movk_i32 s5, 0xa00
	s_set_vgpr_msb 0x88a4
	v_mad_i32_i24 v6 /*v518*/, 0xffffff40, v253 /*v509*/, v1 /*v513*/
	s_set_vgpr_msb 0xa48a
	v_or_b32_e32 v7 /*v519*/, 32, v2 /*v514*/
	v_or_b32_e32 v8 /*v520*/, 64, v2 /*v514*/
	v_lshl_or_b32 v9 /*v521*/, v5 /*v517*/, 4, 0x60
	s_mul_f32 s2, s2, 0x4f7ffffe
	v_or_b32_e32 v10 /*v522*/, 0x80, v2 /*v514*/
	v_or_b32_e32 v11 /*v523*/, 0xa0, v2 /*v514*/
	v_or_b32_e32 v12 /*v524*/, 0xc0, v2 /*v514*/
	s_cvt_u32_f32 s2, s2
	v_mad_u32_u24 v14 /*v526*/, 0x110, v4 /*v516*/, s4
	v_mad_u32_u24 v15 /*v527*/, 0x50, v4 /*v516*/, s5
	s_set_vgpr_msb 0x8a80
	v_and_or_b32 v16 /*v528*/, v2, 16, 0xe0
	s_mul_i32 s3, s3, s2
	s_set_vgpr_msb 0x8000
	v_dual_mov_b32 v235, v234 :: v_dual_mov_b32 v236, v234
	v_dual_mov_b32 v237, v234 :: v_dual_mov_b32 v238, v234
	v_dual_mov_b32 v239, v234 :: v_dual_mov_b32 v240, v234
	v_dual_mov_b32 v241, v234 :: v_dual_mov_b32 v218, v234
	v_dual_mov_b32 v219, v234 :: v_dual_mov_b32 v220, v234
	v_dual_mov_b32 v221, v234 :: v_dual_mov_b32 v222, v234
	v_dual_mov_b32 v223, v234 :: v_dual_mov_b32 v224, v234
	v_dual_mov_b32 v225, v234 :: v_dual_mov_b32 v210, v234
	v_dual_mov_b32 v211, v234 :: v_dual_mov_b32 v212, v234
	v_dual_mov_b32 v213, v234 :: v_dual_mov_b32 v214, v234
	v_dual_mov_b32 v215, v234 :: v_dual_mov_b32 v216, v234
	v_dual_mov_b32 v217, v234 :: v_dual_mov_b32 v186, v234
	v_dual_mov_b32 v187, v234 :: v_dual_mov_b32 v188, v234
	v_dual_mov_b32 v189, v234 :: v_dual_mov_b32 v190, v234
	v_dual_mov_b32 v191, v234 :: v_dual_mov_b32 v192, v234
	v_dual_mov_b32 v193, v234 :: v_dual_mov_b32 v170, v234
	v_dual_mov_b32 v171, v234 :: v_dual_mov_b32 v172, v234
	v_dual_mov_b32 v173, v234 :: v_dual_mov_b32 v174, v234
	v_dual_mov_b32 v175, v234 :: v_dual_mov_b32 v176, v234
	v_dual_mov_b32 v177, v234 :: v_dual_mov_b32 v154, v234
	v_dual_mov_b32 v155, v234 :: v_dual_mov_b32 v156, v234
	v_dual_mov_b32 v157, v234 :: v_dual_mov_b32 v158, v234
	v_dual_mov_b32 v159, v234 :: v_dual_mov_b32 v160, v234
	v_dual_mov_b32 v161, v234 :: v_dual_mov_b32 v138, v234
	v_dual_mov_b32 v139, v234 :: v_dual_mov_b32 v140, v234
	v_dual_mov_b32 v141, v234 :: v_dual_mov_b32 v142, v234
	v_dual_mov_b32 v143, v234 :: v_dual_mov_b32 v144, v234
	v_dual_mov_b32 v145, v234 :: v_dual_mov_b32 v130, v234
	v_dual_mov_b32 v131, v234 :: v_dual_mov_b32 v132, v234
	v_dual_mov_b32 v133, v234 :: v_dual_mov_b32 v134, v234
	v_dual_mov_b32 v135, v234 :: v_dual_mov_b32 v136, v234
	v_dual_mov_b32 v137, v234 :: v_dual_mov_b32 v106, v234
	v_dual_mov_b32 v107, v234 :: v_dual_mov_b32 v108, v234
	v_dual_mov_b32 v109, v234 :: v_dual_mov_b32 v110, v234
	v_dual_mov_b32 v111, v234 :: v_dual_mov_b32 v112, v234
	v_dual_mov_b32 v113, v234 :: v_dual_mov_b32 v90, v234
	v_dual_mov_b32 v91, v234 :: v_dual_mov_b32 v92, v234
	v_dual_mov_b32 v93, v234 :: v_dual_mov_b32 v94, v234
	v_dual_mov_b32 v95, v234 :: v_dual_mov_b32 v96, v234
	v_dual_mov_b32 v97, v234 :: v_dual_mov_b32 v82, v234
	v_dual_mov_b32 v83, v234 :: v_dual_mov_b32 v84, v234
	v_dual_mov_b32 v85, v234 :: v_dual_mov_b32 v86, v234
	v_dual_mov_b32 v87, v234 :: v_dual_mov_b32 v88, v234
	v_dual_mov_b32 v89, v234 :: v_dual_mov_b32 v58, v234
	v_dual_mov_b32 v59, v234 :: v_dual_mov_b32 v60, v234
	v_dual_mov_b32 v61, v234 :: v_dual_mov_b32 v62, v234
	v_dual_mov_b32 v63, v234 :: v_dual_mov_b32 v64, v234
	v_dual_mov_b32 v65, v234 :: v_dual_mov_b32 v42, v234
	v_dual_mov_b32 v43, v234 :: v_dual_mov_b32 v44, v234
	v_dual_mov_b32 v45, v234 :: v_dual_mov_b32 v46, v234
	v_dual_mov_b32 v47, v234 :: v_dual_mov_b32 v48, v234
	v_dual_mov_b32 v49, v234 :: v_dual_mov_b32 v26, v234
	v_dual_mov_b32 v27, v234 :: v_dual_mov_b32 v28, v234
	v_dual_mov_b32 v29, v234 :: v_dual_mov_b32 v30, v234
	v_dual_mov_b32 v31, v234 :: v_dual_mov_b32 v32, v234
	v_dual_mov_b32 v33, v234 :: v_dual_mov_b32 v10, v234
	v_dual_mov_b32 v11, v234 :: v_dual_mov_b32 v12, v234
	v_dual_mov_b32 v13, v234 :: v_dual_mov_b32 v14, v234
	v_dual_mov_b32 v15, v234 :: v_dual_mov_b32 v16, v234
	v_dual_mov_b32 v17, v234 :: v_dual_mov_b32 v2, v234
	v_dual_mov_b32 v3, v234 :: v_dual_mov_b32 v4, v234
	v_dual_mov_b32 v5, v234 :: v_dual_mov_b32 v6, v234
	v_dual_mov_b32 v7, v234 :: v_dual_mov_b32 v8, v234
	v_dual_mov_b32 v9, v234 :: v_dual_mov_b32 v226, v234
	s_set_vgpr_msb 64
	v_dual_mov_b32 v122 /*v378*/, v234 :: v_dual_mov_b32 v123 /*v379*/, v234
	v_dual_mov_b32 v124 /*v380*/, v234 :: v_dual_mov_b32 v125 /*v381*/, v234
	v_dual_mov_b32 v126 /*v382*/, v234 :: v_dual_mov_b32 v127 /*v383*/, v234
	v_dual_mov_b32 v128 /*v384*/, v234 :: v_dual_mov_b32 v129 /*v385*/, v234
	v_dual_mov_b32 v10 /*v266*/, v234 :: v_dual_mov_b32 v11 /*v267*/, v234
	v_dual_mov_b32 v12 /*v268*/, v234 :: v_dual_mov_b32 v13 /*v269*/, v234
	v_dual_mov_b32 v14 /*v270*/, v234 :: v_dual_mov_b32 v15 /*v271*/, v234
	v_dual_mov_b32 v16 /*v272*/, v234 :: v_dual_mov_b32 v17 /*v273*/, v234
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v227, v234 :: v_dual_mov_b32 v228, v234
	v_dual_mov_b32 v229, v234 :: v_dual_mov_b32 v230, v234
	v_dual_mov_b32 v231, v234 :: v_dual_mov_b32 v232, v234
	v_dual_mov_b32 v233, v234 :: v_dual_mov_b32 v202, v234
	v_dual_mov_b32 v203, v234 :: v_dual_mov_b32 v204, v234
	v_dual_mov_b32 v205, v234 :: v_dual_mov_b32 v206, v234
	v_dual_mov_b32 v207, v234 :: v_dual_mov_b32 v208, v234
	v_dual_mov_b32 v209, v234 :: v_dual_mov_b32 v194, v234
	v_dual_mov_b32 v195, v234 :: v_dual_mov_b32 v196, v234
	v_dual_mov_b32 v197, v234 :: v_dual_mov_b32 v198, v234
	v_dual_mov_b32 v199, v234 :: v_dual_mov_b32 v200, v234
	v_dual_mov_b32 v201, v234 :: v_dual_mov_b32 v178, v234
	v_dual_mov_b32 v179, v234 :: v_dual_mov_b32 v180, v234
	v_dual_mov_b32 v181, v234 :: v_dual_mov_b32 v182, v234
	v_dual_mov_b32 v183, v234 :: v_dual_mov_b32 v184, v234
	v_dual_mov_b32 v185, v234 :: v_dual_mov_b32 v162, v234
	v_dual_mov_b32 v163, v234 :: v_dual_mov_b32 v164, v234
	v_dual_mov_b32 v165, v234 :: v_dual_mov_b32 v166, v234
	v_dual_mov_b32 v167, v234 :: v_dual_mov_b32 v168, v234
	v_dual_mov_b32 v169, v234 :: v_dual_mov_b32 v146, v234
	v_dual_mov_b32 v147, v234 :: v_dual_mov_b32 v148, v234
	v_dual_mov_b32 v149, v234 :: v_dual_mov_b32 v150, v234
	v_dual_mov_b32 v151, v234 :: v_dual_mov_b32 v152, v234
	v_dual_mov_b32 v153, v234 :: v_dual_mov_b32 v122, v234
	v_dual_mov_b32 v123, v234 :: v_dual_mov_b32 v124, v234
	v_dual_mov_b32 v125, v234 :: v_dual_mov_b32 v126, v234
	v_dual_mov_b32 v127, v234 :: v_dual_mov_b32 v128, v234
	v_dual_mov_b32 v129, v234 :: v_dual_mov_b32 v114, v234
	v_dual_mov_b32 v115, v234 :: v_dual_mov_b32 v116, v234
	v_dual_mov_b32 v117, v234 :: v_dual_mov_b32 v118, v234
	v_dual_mov_b32 v119, v234 :: v_dual_mov_b32 v120, v234
	v_dual_mov_b32 v121, v234 :: v_dual_mov_b32 v98, v234
	v_dual_mov_b32 v99, v234 :: v_dual_mov_b32 v100, v234
	v_dual_mov_b32 v101, v234 :: v_dual_mov_b32 v102, v234
	v_dual_mov_b32 v103, v234 :: v_dual_mov_b32 v104, v234
	v_dual_mov_b32 v105, v234 :: v_dual_mov_b32 v74, v234
	v_dual_mov_b32 v75, v234 :: v_dual_mov_b32 v76, v234
	v_dual_mov_b32 v77, v234 :: v_dual_mov_b32 v78, v234
	v_dual_mov_b32 v79, v234 :: v_dual_mov_b32 v80, v234
	v_dual_mov_b32 v81, v234 :: v_dual_mov_b32 v66, v234
	v_dual_mov_b32 v67, v234 :: v_dual_mov_b32 v68, v234
	v_dual_mov_b32 v69, v234 :: v_dual_mov_b32 v70, v234
	v_dual_mov_b32 v71, v234 :: v_dual_mov_b32 v72, v234
	v_dual_mov_b32 v73, v234 :: v_dual_mov_b32 v50, v234
	v_dual_mov_b32 v51, v234 :: v_dual_mov_b32 v52, v234
	v_dual_mov_b32 v53, v234 :: v_dual_mov_b32 v54, v234
	v_dual_mov_b32 v55, v234 :: v_dual_mov_b32 v56, v234
	v_dual_mov_b32 v57, v234 :: v_dual_mov_b32 v34, v234
	v_dual_mov_b32 v35, v234 :: v_dual_mov_b32 v36, v234
	v_dual_mov_b32 v37, v234 :: v_dual_mov_b32 v38, v234
	v_dual_mov_b32 v39, v234 :: v_dual_mov_b32 v40, v234
	v_dual_mov_b32 v41, v234 :: v_dual_mov_b32 v18, v234
	v_dual_mov_b32 v19, v234 :: v_dual_mov_b32 v20, v234
	v_dual_mov_b32 v21, v234 :: v_dual_mov_b32 v22, v234
	v_dual_mov_b32 v23, v234 :: v_dual_mov_b32 v24, v234
	v_mov_b32_e32 v25, v234
	s_mul_hi_u32 s3, s2, s3
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_ashr_i32 s76, s41, 31
	s_add_co_i32 s77, s2, s3
	s_mov_b64 s[64:65], 0
	s_mov_b32 s42, 0x3fb8aa3b
	s_ashr_i32 s39, s38, 31
	s_mov_b32 s50, s54
	s_mov_b32 s51, s55
.LBB0_2:
	s_abs_i32 s2, s64
	s_ashr_i32 s3, s64, 31
	s_mul_hi_u32 s4, s2, s77
	s_xor_b32 s3, s3, s76
	s_mul_i32 s5, s4, s75
	s_add_co_i32 s6, s4, 1
	s_sub_co_i32 s2, s2, s5
	s_mov_b32 s62, s58
	s_sub_co_i32 s5, s2, s75
	s_cmp_ge_u32 s2, s75
	s_mov_b32 s63, s59
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s2, s5, s2
	s_add_co_i32 s5, s4, 1
	s_cmp_ge_u32 s2, s75
	s_set_vgpr_msb 0x9a
	v_mov_b64_e32 v[98:99] /*v[610:611]*/, s[44:45]
	s_cselect_b32 s2, s5, s4
	v_dual_add_nc_u32 v107 /*v619*/, v14 /*v526*/, v7 /*v519*/ :: v_dual_add_nc_u32 v108 /*v620*/, v13 /*v525*/, v8 /*v520*/
	s_xor_b32 s2, s2, s3
	v_dual_add_nc_u32 v103 /*v615*/, v14 /*v526*/, v2 /*v514*/ :: v_dual_add_nc_u32 v105 /*v617*/, v13 /*v525*/, v7 /*v519*/
	s_sub_co_i32 s4, s2, s3
	v_dual_add_nc_u32 v109 /*v621*/, v13 /*v525*/, v9 /*v521*/ :: v_dual_add_nc_u32 v110 /*v622*/, v13 /*v525*/, v10 /*v522*/
	s_mul_i32 s4, s4, s41
	v_dual_add_nc_u32 v111 /*v623*/, v13 /*v525*/, v11 /*v523*/ :: v_dual_add_nc_u32 v112 /*v624*/, v13 /*v525*/, v12 /*v524*/
	s_cmp_lg_u32 s64, s4
	v_dual_add_nc_u32 v113 /*v625*/, v13 /*v525*/, v16 /*v528*/ :: v_dual_add_nc_u32 v114 /*v626*/, v3 /*v515*/, v2 /*v514*/
	s_cselect_b32 s4, -1, 0
	s_xor_b32 s5, s41, s64
	v_dual_add_nc_u32 v115 /*v627*/, v14 /*v526*/, v8 /*v520*/ :: v_dual_add_nc_u32 v116 /*v628*/, v14 /*v526*/, v9 /*v521*/
	s_cmp_lt_i32 s5, 0
	v_dual_add_nc_u32 v119 /*v631*/, v15 /*v527*/, v2 /*v514*/ :: v_dual_add_nc_u32 v120 /*v632*/, v14 /*v526*/, v12 /*v524*/
	s_cselect_b32 s5, -1, 0
	v_dual_add_nc_u32 v117 /*v629*/, v14 /*v526*/, v10 /*v522*/ :: v_dual_add_nc_u32 v118 /*v630*/, v14 /*v526*/, v11 /*v523*/
	s_and_b32 s4, s5, s4
	s_sub_co_ci_u32 s2, s2, s3
	v_add_nc_u32_e32 v101 /*v613*/, v13 /*v525*/, v2 /*v514*/
	s_mul_i32 s3, s2, s41
	s_add_co_i32 s2, s2, s71
	s_sub_co_i32 s3, s64, s3
	v_lshl_or_b32 v17 /*v529*/, s2, 5, v253 /*v509*/
	s_add_co_i32 s2, s3, s68
	v_add_nc_u32_e32 v123 /*v635*/, v15 /*v527*/, v7 /*v519*/
	s_lshl4_add_u32 s3, s2, s70
	s_add_co_i32 s2, s2, s69
	s_set_vgpr_msb 0x9a42
	v_mad_u32 v130 /*v386*/, v17 /*v529*/, s35, s3
	s_set_vgpr_msb 0x428a
	v_or_b32_e32 v67 /*v579*/, 16, v17 /*v529*/
	s_mul_i32 s2, s2, s37
	v_add_nc_u32_e32 v68 /*v580*/, s43, v17 /*v529*/
	v_add_lshl_u32 v17 /*v529*/, s2, v17 /*v529*/, 2
	v_dual_add_nc_u32 v121 /*v633*/, v14 /*v526*/, v16 /*v528*/ :: v_dual_add_nc_u32 v122 /*v634*/, v3 /*v515*/, v7 /*v519*/
	s_set_vgpr_msb 0x8a42
	v_mad_u32 v138 /*v394*/, v67 /*v579*/, s35, s3
	s_set_vgpr_msb 0x4245
	v_or_b32_e32 v130 /*v386*/, v130 /*v386*/, v255 /*v511*/
	s_set_vgpr_msb 0x458a
	v_add_lshl_u32 v83 /*v595*/, s2, v67 /*v579*/, 2
	v_add_nc_u32_e32 v74 /*v586*/, s43, v67 /*v579*/
	buffer_load_b32 v100 /*v612*/, v17 /*v529*/, s[56:59], null offen
	s_clause 0x1
	buffer_load_b32 v102 /*v614*/, v17 /*v529*/, s[60:63], null offen
	buffer_load_b32 v104 /*v616*/, v83 /*v595*/, s[60:63], null offen
	s_set_vgpr_msb 0x8a84
	v_lshlrev_b32_e32 v66 /*v578*/, 4, v130 /*v386*/
	s_set_vgpr_msb 0x8442
	s_clause 0x1
	buffer_load_b128 v[130:133] /*v[386:389]*/, v66 /*v578*/, s[52:55], null offen
	buffer_load_b128 v[134:137] /*v[390:393]*/, v66 /*v578*/, s[52:55], null offen offset:32
	s_set_vgpr_msb 0x4245
	v_or_b32_e32 v138 /*v394*/, v138 /*v394*/, v255 /*v511*/
	s_set_vgpr_msb 0x4548
	v_or_b32_e32 v210 /*v466*/, 64, v66 /*v578*/
	v_or_b32_e32 v211 /*v467*/, 0x80, v66 /*v578*/
	s_set_vgpr_msb 0x4882
	s_clause 0x4
	buffer_load_b128 v[30:33] /*v[542:545]*/, v66 /*v578*/, s[52:55], null offen offset:96
	buffer_load_b128 v[38:41] /*v[550:553]*/, v66 /*v578*/, s[52:55], null offen offset:160
	s_set_vgpr_msb 0x8285
	buffer_load_b128 v[26:29] /*v[538:541]*/, v210 /*v466*/, s[52:55], null offen
	buffer_load_b128 v[34:37] /*v[546:549]*/, v211 /*v467*/, s[52:55], null offen
	v_lshlrev_b32_e32 v82 /*v594*/, 4, v138 /*v394*/
	s_clause 0x1
	buffer_load_b128 v[50:53] /*v[562:565]*/, v210 /*v466*/, s[48:51], null offen
	buffer_load_b128 v[58:61] /*v[570:573]*/, v211 /*v467*/, s[48:51], null offen
	s_set_vgpr_msb 0x854a
	s_clause 0x1
	buffer_load_b128 v[222:225] /*v[478:481]*/, v82 /*v594*/, s[52:55], null offen offset:96
	buffer_load_b128 v[214:217] /*v[470:473]*/, v82 /*v594*/, s[52:55], null offen offset:160
	s_clause 0x1
	buffer_load_b128 v[238:241] /*v[494:497]*/, v82 /*v594*/, s[48:51], null offen offset:96
	buffer_load_b128 v[230:233] /*v[486:489]*/, v82 /*v594*/, s[48:51], null offen offset:160
	s_clause 0x1
	buffer_load_b128 v[138:141] /*v[394:397]*/, v82 /*v594*/, s[52:55], null offen
	buffer_load_b128 v[142:145] /*v[398:401]*/, v82 /*v594*/, s[52:55], null offen offset:32
	v_or_b32_e32 v226 /*v482*/, 64, v82 /*v594*/
	v_or_b32_e32 v227 /*v483*/, 0x80, v82 /*v594*/
	s_set_vgpr_msb 0x4a41
	s_clause 0x1
	buffer_load_b128 v[218:221] /*v[474:477]*/, v226 /*v482*/, s[52:55], null offen
	buffer_load_b128 v[210:213] /*v[466:469]*/, v227 /*v483*/, s[52:55], null offen
	s_set_vgpr_msb 0x4188
	v_or_b32_e32 v17 /*v529*/, 0xc0, v66 /*v578*/
	v_add_nc_u32_e32 v78 /*v590*/, 0xe0, v66 /*v578*/
	s_set_vgpr_msb 0x8842
	s_clause 0x8
	buffer_load_b128 v[170:173] /*v[426:429]*/, v66 /*v578*/, s[48:51], null offen
	buffer_load_b128 v[174:177] /*v[430:433]*/, v66 /*v578*/, s[48:51], null offen offset:32
	buffer_load_b128 v[146:149] /*v[402:405]*/, v82 /*v594*/, s[48:51], null offen
	buffer_load_b128 v[150:153] /*v[406:409]*/, v82 /*v594*/, s[48:51], null offen offset:32
	s_set_vgpr_msb 0x4282
	buffer_load_b128 v[54:57] /*v[566:569]*/, v66 /*v578*/, s[48:51], null offen offset:96
	buffer_load_b128 v[62:65] /*v[574:577]*/, v66 /*v578*/, s[48:51], null offen offset:160
	s_set_vgpr_msb 0x8241
	buffer_load_b128 v[234:237] /*v[490:493]*/, v226 /*v482*/, s[48:51], null offen
	s_set_vgpr_msb 0x410a
	v_cmp_ge_i32_e32 vcc_lo, v0 /*v512*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s2, v0 /*v512*/, v68 /*v580*/
	s_set_vgpr_msb 0xa09
	v_cmp_gt_i32_e64 s3, v251 /*v507*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s4, v252 /*v508*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s5, v249 /*v505*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s6, v250 /*v506*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s7, v247 /*v503*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s8, v248 /*v504*/, v68 /*v580*/
	v_cmp_ge_i32_e64 s9, v254 /*v510*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s10, v254 /*v510*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s11, v245 /*v501*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s12, v246 /*v502*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s13, v243 /*v499*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s14, v244 /*v500*/, v68 /*v580*/
	s_set_vgpr_msb 0x908
	v_cmp_gt_i32_e64 s15, v1, v68 /*v580*/
	s_set_vgpr_msb 0x809
	v_cmp_gt_i32_e64 s16, v242 /*v498*/, v68 /*v580*/
	v_cmp_gt_i32_e64 s17, v247 /*v503*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s18, v248 /*v504*/, v74 /*v586*/
	s_set_vgpr_msb 0x90a
	v_cmp_ge_i32_e64 s19, v0 /*v512*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s20, v0 /*v512*/, v74 /*v586*/
	s_set_vgpr_msb 0xa09
	v_cmp_gt_i32_e64 s21, v251 /*v507*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s22, v252 /*v508*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s23, v249 /*v505*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s24, v250 /*v506*/, v74 /*v586*/
	s_set_vgpr_msb 0x908
	v_cmp_gt_i32_e64 s25, v1, v74 /*v586*/
	s_set_vgpr_msb 0x809
	v_cmp_gt_i32_e64 s26, v242 /*v498*/, v74 /*v586*/
	s_set_vgpr_msb 0x982
	buffer_load_b128 v[70:73] /*v[582:585]*/, v78 /*v590*/, s[52:55], null offen
	s_set_vgpr_msb 0x8209
	v_cmp_ge_i32_e64 s27, v254 /*v510*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s28, v254 /*v510*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s29, v245 /*v501*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s30, v246 /*v502*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s31, v243 /*v499*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s33, v244 /*v500*/, v74 /*v586*/
	s_set_vgpr_msb 0x982
	s_clause 0x4
	buffer_load_b128 v[74:77] /*v[586:589]*/, v17 /*v529*/, s[48:51], null offen
	s_set_vgpr_msb 0x8241
	buffer_load_b128 v[226:229] /*v[482:485]*/, v227 /*v483*/, s[48:51], null offen
	s_set_vgpr_msb 0x418a
	buffer_load_b128 v[78:81] /*v[590:593]*/, v78 /*v590*/, s[48:51], null offen
	buffer_load_b128 v[66:69] /*v[578:581]*/, v17 /*v529*/, s[52:55], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v17 /*v529*/, 0xc0, v82 /*v594*/
	v_add_nc_u32_e32 v94 /*v606*/, 0xe0, v82 /*v594*/
	buffer_load_b32 v106 /*v618*/, v83 /*v595*/, s[56:59], null offen
	buffer_load_b128 v[86:89] /*v[598:601]*/, v94 /*v606*/, s[52:55], null offen
	buffer_load_b128 v[90:93] /*v[602:605]*/, v17 /*v529*/, s[48:51], null offen
	buffer_load_b128 v[82:85] /*v[594:597]*/, v17 /*v529*/, s[52:55], null offen
	buffer_load_b128 v[94:97] /*v[606:609]*/, v94 /*v606*/, s[48:51], null offen
	s_and_b32 s62, s74, vcc_lo
	s_and_b32 s2, s74, s2
	s_and_b32 s4, s74, s4
	s_and_b32 s3, s74, s3
	s_and_b32 s6, s74, s6
	s_and_b32 s5, s74, s5
	s_and_b32 s8, s74, s8
	s_and_b32 s7, s74, s7
	s_and_b32 s9, s74, s9
	s_and_b32 s10, s74, s10
	s_and_b32 s12, s74, s12
	s_and_b32 s11, s74, s11
	s_and_b32 s14, s74, s14
	s_and_b32 s13, s74, s13
	s_and_b32 s16, s74, s16
	s_and_b32 s15, s74, s15
	s_and_b32 s17, s74, s17
	s_and_b32 s18, s74, s18
	s_and_b32 s19, s74, s19
	s_and_b32 s20, s74, s20
	s_and_b32 s21, s74, s21
	s_and_b32 s22, s74, s22
	s_and_b32 s23, s74, s23
	s_and_b32 s24, s74, s24
	s_and_b32 s25, s74, s25
	s_and_b32 s26, s74, s26
	s_and_b32 s27, s74, s27
	s_and_b32 s28, s74, s28
	s_and_b32 s29, s74, s29
	s_and_b32 s30, s74, s30
	s_and_b32 s31, s74, s31
	s_and_b32 s33, s74, s33
	s_add_nc_u64 s[64:65], s[64:65], 1
	s_set_vgpr_msb 0x8a44
	s_wait_loadcnt 0x1f
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[242:249], v[130:137] /*v[386:393]*/, 0
	s_set_vgpr_msb 0x4406
	ds_store_b128 v1 /*v513*/, v[130:133] /*v[386:389]*/ offset:13824
	ds_store_b128 v1 /*v513*/, v[134:137] /*v[390:393]*/ offset:13856
	s_set_vgpr_msb 0x60a
	s_wait_loadcnt 0x1d
	ds_store_b128 v1 /*v513*/, v[38:41] /*v[550:553]*/ offset:13984
	s_set_vgpr_msb 0xa06
	s_wait_loadcnt 0x10
	ds_store_b128 v1 /*v513*/, v[170:173] /*v[426:429]*/ offset:5120
	s_wait_loadcnt 0xf
	ds_store_b128 v1 /*v513*/, v[174:177] /*v[430:433]*/ offset:5152
	s_set_vgpr_msb 0x60a
	ds_store_b128 v1 /*v513*/, v[30:33] /*v[542:545]*/ offset:13920
	ds_store_b128 v1 /*v513*/, v[26:29] /*v[538:541]*/ offset:13888
	ds_store_b128 v1 /*v513*/, v[34:37] /*v[546:549]*/ offset:13952
	ds_store_b128 v1 /*v513*/, v[50:53] /*v[562:565]*/ offset:5184
	s_set_vgpr_msb 0xa45
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[26:33] /*v[282:289]*/, v[130:137] /*v[386:393]*/, 0
	s_set_vgpr_msb 0x450a
	s_wait_loadcnt 0xc
	ds_store_b128 v1 /*v513*/, v[54:57] /*v[566:569]*/ offset:5216
	ds_store_b128 v1 /*v513*/, v[58:61] /*v[570:573]*/ offset:5248
	s_wait_loadcnt 0xb
	ds_store_b128 v1 /*v513*/, v[62:65] /*v[574:577]*/ offset:5280
	s_wait_loadcnt 0x5
	ds_store_b128 v1 /*v513*/, v[66:69] /*v[578:581]*/ offset:14016
	ds_store_b128 v1 /*v513*/, v[70:73] /*v[582:585]*/ offset:14048
	ds_store_b128 v1 /*v513*/, v[74:77] /*v[586:589]*/ offset:5312
	ds_store_b128 v1 /*v513*/, v[78:81] /*v[590:593]*/ offset:5344
	s_set_vgpr_msb 0xa44
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[242:249], v[138:145] /*v[394:401]*/, 0
	s_cmp_lg_u64 s[64:65], s[38:39]
	s_set_vgpr_msb 0x4445
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[26:33] /*v[282:289]*/, v[138:145] /*v[394:401]*/, 0
	s_set_vgpr_msb 0x4558
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[250:257], v[26:33] /*v[538:545]*/, v[162:169] /*v[418:425]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[34:41] /*v[290:297]*/, v[26:33] /*v[538:545]*/, v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5954
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[250:257], v[218:225] /*v[474:481]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5455
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[34:41] /*v[290:297]*/, v[218:225] /*v[474:481]*/, v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5559
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[2:9] /*v[258:265]*/, v[34:41] /*v[546:553]*/, v[162:169] /*v[418:425]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[42:49] /*v[298:305]*/, v[34:41] /*v[546:553]*/, v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5945
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[18:25] /*v[274:281]*/, v[170:177] /*v[426:433]*/, 0
	s_set_vgpr_msb 0x4585
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[18:25] /*v[274:281]*/, v[146:153] /*v[402:409]*/, 0
	s_set_vgpr_msb 0x8555
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[2:9] /*v[258:265]*/, v[210:217] /*v[466:473]*/, v[186:193] /*v[442:449]*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[42:49] /*v[298:305]*/, v[210:217] /*v[466:473]*/, v[178:185] /*v[434:441]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[90:97] /*v[346:353]*/, v[170:177] /*v[426:433]*/, 0
	s_set_vgpr_msb 0x5559
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[418:425]*/, v[50:57] /*v[306:313]*/, v[66:73] /*v[578:585]*/, v[162:169] /*v[418:425]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[162:163] /*v[418:419]*/, v[98:99] /*v[610:611]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[58:65] /*v[314:321]*/, v[66:73] /*v[578:585]*/, v[154:161] /*v[410:417]*/
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[164:165] /*v[420:421]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[166:167] /*v[422:423]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[168:169] /*v[424:425]*/, v[98:99] /*v[610:611]*/
	v_cndmask_b32_e64 v131 /*v387*/, v131 /*v387*/, 0xff61b1e6, s62
	v_cndmask_b32_e64 v130 /*v386*/, v130 /*v386*/, 0xff61b1e6, s2
	v_cndmask_b32_e64 v133 /*v389*/, v133 /*v389*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v132 /*v388*/, v132 /*v388*/, 0xff61b1e6, s4
	s_set_vgpr_msb 0x5985
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[90:97] /*v[346:353]*/, v[146:153] /*v[402:409]*/, 0
	s_set_vgpr_msb 0x8546
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[98:99] /*v[610:611]*/, v[154:155] /*v[410:411]*/
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[98:99] /*v[610:611]*/, v[156:157] /*v[412:413]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[98:99] /*v[610:611]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[98:99] /*v[610:611]*/, v[160:161] /*v[416:417]*/
	s_set_vgpr_msb 0x4659
	v_cndmask_b32_e64 v135 /*v391*/, v135 /*v391*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v134 /*v390*/, v134 /*v390*/, 0xff61b1e6, s6
	v_cndmask_b32_e64 v137 /*v393*/, v137 /*v393*/, 0xff61b1e6, s7
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[66:73] /*v[322:329]*/, v[50:57] /*v[562:569]*/, v[194:201] /*v[450:457]*/
	v_cndmask_b32_e64 v136 /*v392*/, v136 /*v392*/, 0xff61b1e6, s8
	v_cndmask_b32_e64 v161 /*v417*/, v161 /*v417*/, 0xff61b1e6, s15
	v_cndmask_b32_e64 v160 /*v416*/, v160 /*v416*/, 0xff61b1e6, s16
	v_cndmask_b32_e64 v155 /*v411*/, v155 /*v411*/, 0xff61b1e6, s9
	v_cndmask_b32_e64 v154 /*v410*/, v154 /*v410*/, 0xff61b1e6, s10
	v_cndmask_b32_e64 v157 /*v413*/, v157 /*v413*/, 0xff61b1e6, s11
	v_cndmask_b32_e64 v156 /*v412*/, v156 /*v412*/, 0xff61b1e6, s12
	s_set_vgpr_msb 0x59a5
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[66:73] /*v[322:329]*/, v[234:241] /*v[490:497]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa559
	v_cndmask_b32_e64 v159 /*v415*/, v159 /*v415*/, 0xff61b1e6, s13
	v_cndmask_b32_e64 v158 /*v414*/, v158 /*v414*/, 0xff61b1e6, s14
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[134:135] /*v[390:391]*/, v[134:135] /*v[390:391]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[136:137] /*v[392:393]*/, v[136:137] /*v[392:393]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[50:57] /*v[306:313]*/, v[82:89] /*v[594:601]*/, v[186:193] /*v[442:449]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[134:135] /*v[390:391]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[136:137] /*v[392:393]*/, s[42:43] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[58:65] /*v[314:321]*/, v[82:89] /*v[594:601]*/, v[178:185] /*v[434:441]*/
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, s[42:43] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v187 /*v443*/, v187 /*v443*/, 0xff61b1e6, s19
	v_cndmask_b32_e64 v186 /*v442*/, v186 /*v442*/, 0xff61b1e6, s20
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[98:105] /*v[354:361]*/, v[50:57] /*v[562:569]*/, v[202:209] /*v[458:465]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, v[98:99] /*v[610:611]*/
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[184:185] /*v[440:441]*/, v[98:99] /*v[610:611]*/
	v_cndmask_b32_e64 v193 /*v449*/, v193 /*v449*/, 0xff61b1e6, s17
	v_cndmask_b32_e64 v192 /*v448*/, v192 /*v448*/, 0xff61b1e6, s18
	v_cndmask_b32_e64 v189 /*v445*/, v189 /*v445*/, 0xff61b1e6, s21
	s_set_vgpr_msb 0x59a5
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[98:105] /*v[354:361]*/, v[234:241] /*v[490:497]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa559
	v_cndmask_b32_e64 v188 /*v444*/, v188 /*v444*/, 0xff61b1e6, s22
	v_cndmask_b32_e64 v191 /*v447*/, v191 /*v447*/, 0xff61b1e6, s23
	v_cndmask_b32_e64 v190 /*v446*/, v190 /*v446*/, 0xff61b1e6, s24
	v_cndmask_b32_e64 v185 /*v441*/, v185 /*v441*/, 0xff61b1e6, s25
	v_cndmask_b32_e64 v184 /*v440*/, v184 /*v440*/, 0xff61b1e6, s26
	v_cndmask_b32_e64 v179 /*v435*/, v179 /*v435*/, 0xff61b1e6, s27
	v_cndmask_b32_e64 v178 /*v434*/, v178 /*v434*/, 0xff61b1e6, s28
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[74:81] /*v[330:337]*/, v[58:65] /*v[570:577]*/, v[194:201] /*v[450:457]*/
	v_cndmask_b32_e64 v181 /*v437*/, v181 /*v437*/, 0xff61b1e6, s29
	v_cndmask_b32_e64 v180 /*v436*/, v180 /*v436*/, 0xff61b1e6, s30
	v_cndmask_b32_e64 v183 /*v439*/, v183 /*v439*/, 0xff61b1e6, s31
	v_cndmask_b32_e64 v182 /*v438*/, v182 /*v438*/, 0xff61b1e6, s33
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x59a5
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[74:81] /*v[330:337]*/, v[226:233] /*v[482:489]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa559
	v_pk_add_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[184:185] /*v[440:441]*/, v[184:185] /*v[440:441]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, s[42:43] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[106:113] /*v[362:369]*/, v[58:65] /*v[570:577]*/, v[202:209] /*v[458:465]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, s[42:43] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x59a5
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[106:113] /*v[362:369]*/, v[226:233] /*v[482:489]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa559
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[184:185] /*v[440:441]*/, s[42:43] op_sel_hi:[1,0]
	v_exp_f32_e32 v154 /*v410*/, v154 /*v410*/
	v_exp_f32_e32 v155 /*v411*/, v155 /*v411*/
	v_exp_f32_e32 v156 /*v412*/, v156 /*v412*/
	v_exp_f32_e32 v157 /*v413*/, v157 /*v413*/
	v_exp_f32_e32 v158 /*v414*/, v158 /*v414*/
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[82:89] /*v[338:345]*/, v[74:81] /*v[586:593]*/, v[194:201] /*v[450:457]*/
	v_exp_f32_e32 v159 /*v415*/, v159 /*v415*/
	v_exp_f32_e32 v160 /*v416*/, v160 /*v416*/
	v_exp_f32_e32 v161 /*v417*/, v161 /*v417*/
	v_exp_f32_e32 v186 /*v442*/, v186 /*v442*/
	v_exp_f32_e32 v187 /*v443*/, v187 /*v443*/
	v_exp_f32_e32 v188 /*v444*/, v188 /*v444*/
	v_exp_f32_e32 v189 /*v445*/, v189 /*v445*/
	s_set_vgpr_msb 0x59a9
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[82:89] /*v[338:345]*/, v[90:97] /*v[602:609]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa959
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[194:195] /*v[450:451]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[196:197] /*v[452:453]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[198:199] /*v[454:455]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[200:201] /*v[456:457]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v190 /*v446*/, v190 /*v446*/
	v_exp_f32_e32 v191 /*v447*/, v191 /*v447*/
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[114:121] /*v[370:377]*/, v[74:81] /*v[586:593]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x594a
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[18:19] /*v[530:531]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[20:21] /*v[532:533]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[22:23] /*v[534:535]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[24:25] /*v[536:537]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa9
	v_exp_f32_e32 v18 /*v530*/, v130 /*v386*/
	v_exp_f32_e32 v19 /*v531*/, v131 /*v387*/
	v_exp_f32_e32 v20 /*v532*/, v132 /*v388*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[114:121] /*v[370:377]*/, v[90:97] /*v[602:609]*/, v[42:49] /*v[554:561]*/
	v_exp_f32_e32 v21 /*v533*/, v133 /*v389*/
	v_exp_f32_e32 v22 /*v534*/, v134 /*v390*/
	v_exp_f32_e32 v23 /*v535*/, v135 /*v391*/
	v_exp_f32_e32 v24 /*v536*/, v136 /*v392*/
	v_exp_f32_e32 v25 /*v537*/, v137 /*v393*/
	s_set_vgpr_msb 0xa949
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[202:203] /*v[458:459]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[204:205] /*v[460:461]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[206:207] /*v[462:463]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[208:209] /*v[464:465]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v193 /*v449*/, v193 /*v449*/
	v_exp_f32_e32 v178 /*v434*/, v178 /*v434*/
	v_exp_f32_e32 v179 /*v435*/, v179 /*v435*/
	v_exp_f32_e32 v180 /*v436*/, v180 /*v436*/
	v_exp_f32_e32 v181 /*v437*/, v181 /*v437*/
	v_exp_f32_e32 v182 /*v438*/, v182 /*v438*/
	v_exp_f32_e32 v183 /*v439*/, v183 /*v439*/
	v_exp_f32_e32 v184 /*v440*/, v184 /*v440*/
	v_exp_f32_e32 v185 /*v441*/, v185 /*v441*/
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[42:43] /*v[554:555]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[44:45] /*v[556:557]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[46:47] /*v[558:559]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[48:49] /*v[560:561]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v133 /*v389*/, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v132 /*v388*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v131 /*v387*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v130 /*v386*/, v18 /*v530*/, v19 /*v531*/
	s_set_vgpr_msb 0x4a49
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[164:165] /*v[420:421]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[166:167] /*v[422:423]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[168:169] /*v[424:425]*/, v[24:25] /*v[536:537]*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v135 /*v391*/, v156 /*v412*/, v157 /*v413*/
	v_cvt_pk_bf16_f32 v136 /*v392*/, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v134 /*v390*/, v154 /*v410*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v137 /*v393*/, v160 /*v416*/, v161 /*v417*/
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[170:171] /*v[426:427]*/, v[154:155] /*v[410:411]*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, v[156:157] /*v[412:413]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, v[160:161] /*v[416:417]*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[186:187] /*v[442:443]*/
	v_cvt_pk_bf16_f32 v154 /*v410*/, v186 /*v442*/, v187 /*v443*/
	v_cvt_pk_bf16_f32 v155 /*v411*/, v188 /*v444*/, v189 /*v445*/
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[196:197] /*v[452:453]*/, v[188:189] /*v[444:445]*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v190 /*v446*/, v191 /*v447*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[198:199] /*v[454:455]*/, v[190:191] /*v[446:447]*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v192 /*v448*/, v193 /*v449*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[200:201] /*v[456:457]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[202:203] /*v[458:459]*/, v[178:179] /*v[434:435]*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v180 /*v436*/, v181 /*v437*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[204:205] /*v[460:461]*/, v[180:181] /*v[436:437]*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v182 /*v438*/, v183 /*v439*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[206:207] /*v[462:463]*/, v[182:183] /*v[438:439]*/
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[208:209] /*v[464:465]*/, v[184:185] /*v[440:441]*/
	s_set_vgpr_msb 0x4546
	ds_store_b128 v6 /*v518*/, v[130:133] /*v[386:389]*/
	ds_store_b128 v6 /*v518*/, v[134:137] /*v[390:393]*/ offset:32
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[98:99] /*v[610:611]*/, v[162:163] /*v[418:419]*/
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[98:99] /*v[610:611]*/, v[164:165] /*v[420:421]*/
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[98:99] /*v[610:611]*/, v[166:167] /*v[422:423]*/
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[98:99] /*v[610:611]*/, v[168:169] /*v[424:425]*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[98:99] /*v[610:611]*/, v[170:171] /*v[426:427]*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[98:99] /*v[610:611]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[98:99] /*v[610:611]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[98:99] /*v[610:611]*/, v[176:177] /*v[432:433]*/
	s_set_vgpr_msb 0x4645
	v_cvt_pk_bf16_f32 v161 /*v417*/, v184 /*v440*/, v185 /*v441*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[98:99] /*v[610:611]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[98:99] /*v[610:611]*/, v[186:187] /*v[442:443]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[98:99] /*v[610:611]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[98:99] /*v[610:611]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[98:99] /*v[610:611]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[98:99] /*v[610:611]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[98:99] /*v[610:611]*/, v[180:181] /*v[436:437]*/
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[98:99] /*v[610:611]*/, v[182:183] /*v[438:439]*/
	s_set_vgpr_msb 0x4645
	v_cvt_pk_bf16_f32 v130 /*v386*/, v130 /*v386*/, v131 /*v387*/
	v_cvt_pk_bf16_f32 v131 /*v387*/, v132 /*v388*/, v133 /*v389*/
	v_cvt_pk_bf16_f32 v132 /*v388*/, v134 /*v390*/, v135 /*v391*/
	v_cvt_pk_bf16_f32 v133 /*v389*/, v136 /*v392*/, v137 /*v393*/
	v_cvt_pk_bf16_f32 v134 /*v390*/, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v135 /*v391*/, v164 /*v420*/, v165 /*v421*/
	v_cvt_pk_bf16_f32 v136 /*v392*/, v166 /*v422*/, v167 /*v423*/
	v_cvt_pk_bf16_f32 v137 /*v393*/, v168 /*v424*/, v169 /*v425*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v172 /*v428*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v184 /*v440*/, v185 /*v441*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v182 /*v438*/, v183 /*v439*/
	s_set_vgpr_msb 0x4506
	ds_store_b128 v6 /*v518*/, v[130:133] /*v[386:389]*/ offset:2560
	ds_store_b128 v6 /*v518*/, v[134:137] /*v[390:393]*/ offset:2592
	ds_store_b128 v1 /*v513*/, v[146:149] /*v[402:405]*/ offset:9472
	ds_store_b128 v1 /*v513*/, v[150:153] /*v[406:409]*/ offset:9504
	ds_store_b128 v1 /*v513*/, v[138:141] /*v[394:397]*/ offset:18176
	ds_store_b128 v1 /*v513*/, v[142:145] /*v[398:401]*/ offset:18208
	ds_store_b128 v1 /*v513*/, v[234:237] /*v[490:493]*/ offset:9536
	ds_store_b128 v1 /*v513*/, v[238:241] /*v[494:497]*/ offset:9568
	ds_store_b128 v1 /*v513*/, v[218:221] /*v[474:477]*/ offset:18240
	ds_store_b128 v1 /*v513*/, v[222:225] /*v[478:481]*/ offset:18272
	ds_store_b128 v1 /*v513*/, v[226:229] /*v[482:485]*/ offset:9600
	ds_store_b128 v1 /*v513*/, v[230:233] /*v[486:489]*/ offset:9632
	ds_store_b128 v1 /*v513*/, v[210:213] /*v[466:469]*/ offset:18304
	ds_store_b128 v1 /*v513*/, v[214:217] /*v[470:473]*/ offset:18336
	s_set_vgpr_msb 0x60a
	ds_store_b128 v1 /*v513*/, v[90:93] /*v[602:605]*/ offset:9664
	ds_store_b128 v1 /*v513*/, v[94:97] /*v[606:609]*/ offset:9696
	ds_store_b128 v1 /*v513*/, v[82:85] /*v[594:597]*/ offset:18368
	ds_store_b128 v1 /*v513*/, v[86:89] /*v[598:601]*/ offset:18400
	s_set_vgpr_msb 0xa46
	ds_store_b128 v6 /*v518*/, v[154:157] /*v[410:413]*/ offset:1280
	ds_store_b128 v6 /*v518*/, v[158:161] /*v[414:417]*/ offset:1312
	ds_store_b128 v6 /*v518*/, v[162:165] /*v[418:421]*/ offset:3840
	ds_store_b128 v6 /*v518*/, v[166:169] /*v[422:425]*/ offset:3872
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_tr16_b128 v[130:133] /*v[386:389]*/, v101 /*v613*/
	ds_load_tr16_b128 v[134:137] /*v[390:393]*/, v101 /*v613*/ offset:4352
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v114 /*v626*/
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v114 /*v626*/ offset:1280
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v105 /*v617*/
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v105 /*v617*/ offset:4352
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v108 /*v620*/
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v108 /*v620*/ offset:4352
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v109 /*v621*/
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v109 /*v621*/ offset:4352
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v110 /*v622*/
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v110 /*v622*/ offset:4352
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v111 /*v623*/
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v111 /*v623*/ offset:4352
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v112 /*v624*/
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v112 /*v624*/ offset:4352
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v113 /*v625*/
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v113 /*v625*/ offset:4352
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v115 /*v627*/
	ds_load_tr16_b128 v[206:209] /*v[462:465]*/, v115 /*v627*/ offset:4352
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v119 /*v631*/
	s_set_vgpr_msb 0x4605
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[234:241], v[138:145] /*v[394:401]*/, v[130:137] /*v[386:393]*/, v[234:241]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[222:225] /*v[478:481]*/, v103 /*v615*/ offset:4352
	ds_load_tr16_b128 v[226:229] /*v[482:485]*/, v116 /*v628*/
	ds_load_tr16_b128 v[230:233] /*v[486:489]*/, v116 /*v628*/ offset:4352
	ds_load_tr16_b128 v[234:237] /*v[490:493]*/, v117 /*v629*/
	ds_load_tr16_b128 v[238:241] /*v[494:497]*/, v117 /*v629*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[18:21] /*v[530:533]*/, v118 /*v630*/
	ds_load_tr16_b128 v[22:25] /*v[534:537]*/, v118 /*v630*/ offset:4352
	ds_load_tr16_b128 v[26:29] /*v[538:541]*/, v120 /*v632*/
	ds_load_tr16_b128 v[30:33] /*v[542:545]*/, v120 /*v632*/ offset:4352
	ds_load_tr16_b128 v[34:37] /*v[546:549]*/, v121 /*v633*/
	ds_load_tr16_b128 v[38:41] /*v[550:553]*/, v121 /*v633*/ offset:4352
	ds_load_tr16_b128 v[42:45] /*v[554:557]*/, v122 /*v634*/
	s_set_vgpr_msb 0x8205
	s_wait_dscnt 0x1b
	v_wmma_f32_16x16x32_bf16 v[218:225], v[138:145] /*v[394:401]*/, v[146:153] /*v[402:409]*/, v[218:225]
	s_wait_dscnt 0x19
	v_wmma_f32_16x16x32_bf16 v[210:217], v[138:145] /*v[394:401]*/, v[154:161] /*v[410:417]*/, v[210:217]
	s_wait_dscnt 0x17
	v_wmma_f32_16x16x32_bf16 v[186:193], v[138:145] /*v[394:401]*/, v[162:169] /*v[418:425]*/, v[186:193]
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[170:177], v[138:145] /*v[394:401]*/, v[170:177] /*v[426:433]*/, v[170:177]
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145] /*v[394:401]*/, v[178:185] /*v[434:441]*/, v[154:161]
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[138:145], v[138:145] /*v[394:401]*/, v[186:193] /*v[442:449]*/, v[138:145]
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[130:137], v[138:145] /*v[394:401]*/, v[194:201] /*v[450:457]*/, v[130:137]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v119 /*v631*/ offset:1280
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v107 /*v619*/
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v107 /*v619*/ offset:4352
	ds_load_tr16_b128 v[218:221] /*v[474:477]*/, v103 /*v615*/
	s_set_vgpr_msb 0x4205
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[226:233], v[210:217] /*v[466:473]*/, v[202:209] /*v[458:465]*/, v[226:233]
	s_set_vgpr_msb 0x555
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[210:217] /*v[466:473]*/, v[138:145] /*v[394:401]*/, v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[210:217] /*v[466:473]*/, v[218:225] /*v[474:481]*/, v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[202:209], v[210:217] /*v[466:473]*/, v[226:233] /*v[482:489]*/, v[202:209]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[210:217] /*v[466:473]*/, v[234:241] /*v[490:497]*/, v[194:201]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[178:185], v[210:217] /*v[466:473]*/, v[18:25] /*v[530:537]*/, v[178:185]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[210:217] /*v[466:473]*/, v[26:33] /*v[538:545]*/, v[162:169]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[210:217] /*v[466:473]*/, v[34:41] /*v[546:553]*/, v[146:153]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v122 /*v634*/ offset:1280
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v123 /*v635*/
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v123 /*v635*/ offset:1280
	s_set_vgpr_msb 0x4206
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_wmma_f32_16x16x32_bf16 v[106:113], v[42:49] /*v[554:561]*/, v[130:137] /*v[386:393]*/, v[106:113]
	s_set_vgpr_msb 0x605
	s_barrier_wait -1
	v_wmma_f32_16x16x32_bf16 v[122:129], v[210:217] /*v[466:473]*/, v[218:225] /*v[474:481]*/, v[122:129]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[90:97], v[42:49] /*v[554:561]*/, v[146:153] /*v[402:409]*/, v[90:97]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[114:121], v[210:217] /*v[466:473]*/, v[138:145] /*v[394:401]*/, v[114:121]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[82:89], v[42:49] /*v[554:561]*/, v[154:161] /*v[410:417]*/, v[82:89]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[98:105], v[210:217] /*v[466:473]*/, v[202:209] /*v[458:465]*/, v[98:105]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[58:65], v[42:49] /*v[554:561]*/, v[162:169] /*v[418:425]*/, v[58:65]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[74:81], v[210:217] /*v[466:473]*/, v[226:233] /*v[482:489]*/, v[74:81]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[42:49], v[42:49] /*v[554:561]*/, v[170:177] /*v[426:433]*/, v[42:49]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[66:73], v[210:217] /*v[466:473]*/, v[234:241] /*v[490:497]*/, v[66:73]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[26:33], v[42:49] /*v[554:561]*/, v[178:185] /*v[434:441]*/, v[26:33]
	s_set_vgpr_msb 0x609
	v_wmma_f32_16x16x32_bf16 v[50:57], v[210:217] /*v[466:473]*/, v[18:25] /*v[530:537]*/, v[50:57]
	s_set_vgpr_msb 0x906
	v_wmma_f32_16x16x32_bf16 v[10:17], v[42:49] /*v[554:561]*/, v[186:193] /*v[442:449]*/, v[10:17]
	s_set_vgpr_msb 0x609
	v_wmma_f32_16x16x32_bf16 v[34:41], v[210:217] /*v[466:473]*/, v[26:33] /*v[538:545]*/, v[34:41]
	s_set_vgpr_msb 0x906
	v_wmma_f32_16x16x32_bf16 v[2:9], v[42:49] /*v[554:561]*/, v[194:201] /*v[450:457]*/, v[2:9]
	s_set_vgpr_msb 0x609
	v_wmma_f32_16x16x32_bf16 v[18:25], v[210:217] /*v[466:473]*/, v[34:41] /*v[546:553]*/, v[18:25]
	s_set_vgpr_msb 0x900
	s_cbranch_scc1 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	v_mov_b32_e32 v18, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v19, v18 :: v_dual_mov_b32 v20, v18
	v_dual_mov_b32 v21, v18 :: v_dual_mov_b32 v22, v18
	v_dual_mov_b32 v23, v18 :: v_dual_mov_b32 v24, v18
	v_mov_b32_e32 v25, v18
	v_mov_b64_e32 v[36:37], v[20:21]
	v_mov_b64_e32 v[34:35], v[18:19]
	s_delay_alu instid0(VALU_DEP_4)
	v_mov_b64_e32 v[38:39], v[22:23]
	v_mov_b64_e32 v[54:55], v[22:23]
	v_mov_b64_e32 v[40:41], v[24:25]
	v_mov_b64_e32 v[56:57], v[24:25]
	v_mov_b64_e32 v[52:53], v[20:21]
	v_mov_b64_e32 v[50:51], v[18:19]
	v_mov_b64_e32 v[72:73], v[24:25]
	v_mov_b64_e32 v[70:71], v[22:23]
	v_mov_b64_e32 v[68:69], v[20:21]
	v_mov_b64_e32 v[66:67], v[18:19]
	v_mov_b64_e32 v[80:81], v[24:25]
	v_mov_b64_e32 v[78:79], v[22:23]
	v_mov_b64_e32 v[76:77], v[20:21]
	v_mov_b64_e32 v[74:75], v[18:19]
	v_mov_b64_e32 v[104:105], v[24:25]
	v_mov_b64_e32 v[102:103], v[22:23]
	v_mov_b64_e32 v[100:101], v[20:21]
	v_mov_b64_e32 v[98:99], v[18:19]
	v_mov_b64_e32 v[120:121], v[24:25]
	v_mov_b64_e32 v[118:119], v[22:23]
	v_mov_b64_e32 v[116:117], v[20:21]
	v_mov_b64_e32 v[114:115], v[18:19]
	v_mov_b64_e32 v[128:129], v[24:25]
	v_mov_b64_e32 v[126:127], v[22:23]
	v_mov_b64_e32 v[124:125], v[20:21]
	v_mov_b64_e32 v[122:123], v[18:19]
	v_mov_b64_e32 v[152:153], v[24:25]
	v_mov_b64_e32 v[150:151], v[22:23]
	v_mov_b64_e32 v[148:149], v[20:21]
	v_mov_b64_e32 v[146:147], v[18:19]
	v_mov_b64_e32 v[168:169], v[24:25]
	v_mov_b64_e32 v[166:167], v[22:23]
	v_mov_b64_e32 v[164:165], v[20:21]
	v_mov_b64_e32 v[162:163], v[18:19]
	v_mov_b64_e32 v[184:185], v[24:25]
	v_mov_b64_e32 v[182:183], v[22:23]
	v_mov_b64_e32 v[180:181], v[20:21]
	v_mov_b64_e32 v[178:179], v[18:19]
	v_mov_b64_e32 v[200:201], v[24:25]
	v_mov_b64_e32 v[198:199], v[22:23]
	v_mov_b64_e32 v[196:197], v[20:21]
	v_mov_b64_e32 v[194:195], v[18:19]
	v_mov_b64_e32 v[208:209], v[24:25]
	v_mov_b64_e32 v[206:207], v[22:23]
	v_mov_b64_e32 v[204:205], v[20:21]
	v_mov_b64_e32 v[202:203], v[18:19]
	v_mov_b64_e32 v[232:233], v[24:25]
	v_mov_b64_e32 v[230:231], v[22:23]
	v_mov_b64_e32 v[228:229], v[20:21]
	v_mov_b64_e32 v[226:227], v[18:19]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[16:17] /*v[272:273]*/, v[24:25]
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, v[22:23]
	v_mov_b64_e32 v[12:13] /*v[268:269]*/, v[20:21]
	v_mov_b64_e32 v[10:11] /*v[266:267]*/, v[18:19]
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[24:25]
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[22:23]
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[20:21]
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[18:19]
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[2:3], v[18:19]
	v_mov_b64_e32 v[4:5], v[20:21]
	v_mov_b64_e32 v[6:7], v[22:23]
	v_mov_b64_e32 v[8:9], v[24:25]
	v_mov_b64_e32 v[10:11], v[18:19]
	v_mov_b64_e32 v[12:13], v[20:21]
	v_mov_b64_e32 v[14:15], v[22:23]
	v_mov_b64_e32 v[16:17], v[24:25]
	v_mov_b64_e32 v[32:33], v[24:25]
	v_mov_b64_e32 v[30:31], v[22:23]
	v_mov_b64_e32 v[28:29], v[20:21]
	v_mov_b64_e32 v[26:27], v[18:19]
	v_mov_b64_e32 v[48:49], v[24:25]
	v_mov_b64_e32 v[46:47], v[22:23]
	v_mov_b64_e32 v[44:45], v[20:21]
	v_mov_b64_e32 v[42:43], v[18:19]
	v_mov_b64_e32 v[64:65], v[24:25]
	v_mov_b64_e32 v[62:63], v[22:23]
	v_mov_b64_e32 v[60:61], v[20:21]
	v_mov_b64_e32 v[58:59], v[18:19]
	v_mov_b64_e32 v[88:89], v[24:25]
	v_mov_b64_e32 v[86:87], v[22:23]
	v_mov_b64_e32 v[84:85], v[20:21]
	v_mov_b64_e32 v[82:83], v[18:19]
	v_mov_b64_e32 v[96:97], v[24:25]
	v_mov_b64_e32 v[94:95], v[22:23]
	v_mov_b64_e32 v[92:93], v[20:21]
	v_mov_b64_e32 v[90:91], v[18:19]
	v_mov_b64_e32 v[112:113], v[24:25]
	v_mov_b64_e32 v[110:111], v[22:23]
	v_mov_b64_e32 v[108:109], v[20:21]
	v_mov_b64_e32 v[106:107], v[18:19]
	v_mov_b64_e32 v[136:137], v[24:25]
	v_mov_b64_e32 v[134:135], v[22:23]
	v_mov_b64_e32 v[132:133], v[20:21]
	v_mov_b64_e32 v[130:131], v[18:19]
	v_mov_b64_e32 v[144:145], v[24:25]
	v_mov_b64_e32 v[142:143], v[22:23]
	v_mov_b64_e32 v[140:141], v[20:21]
	v_mov_b64_e32 v[138:139], v[18:19]
	v_mov_b64_e32 v[160:161], v[24:25]
	v_mov_b64_e32 v[158:159], v[22:23]
	v_mov_b64_e32 v[156:157], v[20:21]
	v_mov_b64_e32 v[154:155], v[18:19]
	v_mov_b64_e32 v[176:177], v[24:25]
	v_mov_b64_e32 v[174:175], v[22:23]
	v_mov_b64_e32 v[172:173], v[20:21]
	v_mov_b64_e32 v[170:171], v[18:19]
	v_mov_b64_e32 v[192:193], v[24:25]
	v_mov_b64_e32 v[190:191], v[22:23]
	v_mov_b64_e32 v[188:189], v[20:21]
	v_mov_b64_e32 v[186:187], v[18:19]
	v_mov_b64_e32 v[216:217], v[24:25]
	v_mov_b64_e32 v[214:215], v[22:23]
	v_mov_b64_e32 v[212:213], v[20:21]
	v_mov_b64_e32 v[210:211], v[18:19]
	v_mov_b64_e32 v[224:225], v[24:25]
	v_mov_b64_e32 v[222:223], v[22:23]
	v_mov_b64_e32 v[220:221], v[20:21]
	v_mov_b64_e32 v[218:219], v[18:19]
	v_mov_b64_e32 v[240:241], v[24:25]
	v_mov_b64_e32 v[238:239], v[22:23]
	v_mov_b64_e32 v[236:237], v[20:21]
	v_mov_b64_e32 v[234:235], v[18:19]
.LBB0_4:
	s_sub_co_i32 s2, s72, s73
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s2, s2, s41
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB0_7
	s_abs_i32 s8, s41
	s_set_vgpr_msb 0x49
	v_dual_lshlrev_b32 v130 /*v386*/, 4, v5 /*v517*/ :: v_dual_bitop2_b32 v152 /*v408*/, 32, v2 /*v514*/ bitop3:0x54
	s_cvt_f32_u32 s4, s8
	s_movk_i32 s5, 0x1400
	s_movk_i32 s6, 0x3600
	v_mad_u32_u24 v147 /*v403*/, 0x110, v4 /*v516*/, s5
	v_s_rcp_f32 s4, s4
	s_movk_i32 s5, 0xa00
	v_mad_u32_u24 v148 /*v404*/, 0x110, v4 /*v516*/, s6
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_nc_u32 v133 /*v389*/, v147 /*v403*/, v2 /*v514*/ :: v_dual_bitop2_b32 v138 /*v394*/, 64, v2 /*v514*/ bitop3:0x54
	s_set_vgpr_msb 0x4944
	v_or_b32_e32 v140 /*v396*/, 0x60, v130 /*v386*/
	s_set_vgpr_msb 0x4448
	v_or_b32_e32 v142 /*v398*/, 0x80, v2 /*v514*/
	v_or_b32_e32 v144 /*v400*/, 0xa0, v2 /*v514*/
	s_mul_f32 s4, s4, 0x4f7ffffe
	v_or_b32_e32 v146 /*v402*/, 0xc0, v2 /*v514*/
	s_set_vgpr_msb 0x4844
	v_or_b32_e32 v149 /*v405*/, 0xe0, v130 /*v386*/
	s_set_vgpr_msb 0x4448
	v_mad_u32_u24 v153 /*v409*/, 0x50, v4 /*v516*/, s5
	s_cvt_u32_f32 s4, s4
	s_sub_co_i32 s5, 0, s8
	s_mov_b32 s12, s36
	s_mov_b32 s13, s36
	s_mul_i32 s5, s5, s4
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, s[12:13]
	s_set_vgpr_msb 0x4864
	v_mad_i32_i24 v132 /*v388*/, 0xffffff40, v253 /*v509*/, v1 /*v513*/
	s_set_vgpr_msb 0x6449
	v_dual_add_nc_u32 v134 /*v390*/, v148 /*v404*/, v2 /*v514*/ :: v_dual_add_nc_u32 v150 /*v406*/, v153 /*v409*/, v2 /*v514*/
	s_set_vgpr_msb 0x4945
	v_dual_add_nc_u32 v135 /*v391*/, v147 /*v403*/, v152 /*v408*/ :: v_dual_add_nc_u32 v136 /*v392*/, v148 /*v404*/, v152 /*v408*/
	v_dual_add_nc_u32 v137 /*v393*/, v147 /*v403*/, v138 /*v394*/ :: v_dual_add_nc_u32 v138 /*v394*/, v148 /*v404*/, v138 /*v394*/
	v_dual_add_nc_u32 v139 /*v395*/, v147 /*v403*/, v140 /*v396*/ :: v_dual_add_nc_u32 v140 /*v396*/, v148 /*v404*/, v140 /*v396*/
	v_dual_add_nc_u32 v141 /*v397*/, v147 /*v403*/, v142 /*v398*/ :: v_dual_add_nc_u32 v142 /*v398*/, v148 /*v404*/, v142 /*v398*/
	v_dual_add_nc_u32 v143 /*v399*/, v147 /*v403*/, v144 /*v400*/ :: v_dual_add_nc_u32 v144 /*v400*/, v148 /*v404*/, v144 /*v400*/
	v_dual_add_nc_u32 v145 /*v401*/, v147 /*v403*/, v146 /*v402*/ :: v_dual_add_nc_u32 v146 /*v402*/, v148 /*v404*/, v146 /*v402*/
	v_dual_add_nc_u32 v147 /*v403*/, v147 /*v403*/, v149 /*v405*/ :: v_dual_add_nc_u32 v148 /*v404*/, v148 /*v404*/, v149 /*v405*/
	s_set_vgpr_msb 0x454a
	v_add_nc_u32_e32 v149 /*v405*/, v3 /*v515*/, v2 /*v514*/
	s_set_vgpr_msb 0x4a46
	v_add_nc_u32_e32 v151 /*v407*/, v3 /*v515*/, v152 /*v408*/
	s_set_vgpr_msb 0x4645
	v_add_nc_u32_e32 v152 /*v408*/, v153 /*v409*/, v152 /*v408*/
	s_mul_hi_u32 s5, s4, s5
	s_add_co_i32 s7, s73, s71
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s9, s41, 31
	s_add_co_i32 s10, s4, s5
	s_mov_b64 s[4:5], 0
	s_mov_b32 s50, s54
	s_mov_b32 s51, s55
	s_mov_b32 s62, s58
	s_mov_b32 s63, s59
	s_mov_b32 s6, 0x3fb8aa3b
	s_set_vgpr_msb 0x4500
.LBB0_6:
	s_abs_i32 s11, s4
	s_ashr_i32 s12, s4, 31
	s_mul_hi_u32 s13, s11, s10
	s_xor_b32 s12, s12, s9
	s_mul_i32 s14, s13, s8
	s_add_co_i32 s15, s13, 1
	s_sub_co_i32 s11, s11, s14
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s14, s11, s8
	s_cmp_ge_u32 s11, s8
	s_cselect_b32 s13, s15, s13
	s_cselect_b32 s11, s14, s11
	s_add_co_i32 s14, s13, 1
	s_cmp_ge_u32 s11, s8
	s_cselect_b32 s11, s14, s13
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s11, s11, s12
	s_sub_co_i32 s13, s11, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s13, s13, s41
	s_cmp_lg_u32 s4, s13
	s_cselect_b32 s13, -1, 0
	s_xor_b32 s14, s41, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s14, 0
	s_cselect_b32 s14, -1, 0
	s_and_b32 s13, s14, s13
	s_sub_co_ci_u32 s11, s11, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s12, s11, s41
	s_add_co_i32 s11, s7, s11
	s_sub_co_i32 s12, s4, s12
	s_set_vgpr_msb 0x55
	v_lshl_or_b32 v153 /*v409*/, s11, 5, v253 /*v509*/
	s_add_co_i32 s11, s12, s68
	s_add_nc_u64 s[4:5], s[4:5], 1
	s_lshl4_add_u32 s12, s11, s70
	s_add_co_i32 s11, s11, s69
	v_mad_u32 v154 /*v410*/, v153 /*v409*/, s35, s12
	s_mul_i32 s11, s11, s37
	s_cmp_lg_u64 s[4:5], s[2:3]
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v154 /*v410*/, v154 /*v410*/, v255 /*v511*/
	s_set_vgpr_msb 0x5584
	v_or_b32_e32 v90 /*v602*/, 16, v153 /*v409*/
	s_set_vgpr_msb 0x8444
	v_add_lshl_u32 v153 /*v409*/, s11, v153 /*v409*/, 2
	s_set_vgpr_msb 0x4484
	v_lshlrev_b32_e32 v74 /*v586*/, 4, v154 /*v410*/
	s_set_vgpr_msb 0x8442
	v_mad_u32 v194 /*v450*/, v90 /*v602*/, s35, s12
	s_set_vgpr_msb 0x4289
	v_add_lshl_u32 v90 /*v602*/, s11, v90 /*v602*/, 2
	buffer_load_b32 v106 /*v618*/, v153 /*v409*/, s[60:63], null offen
	s_clause 0x2
	buffer_load_b32 v108 /*v620*/, v153 /*v409*/, s[56:59], null offen
	s_set_vgpr_msb 0x8982
	buffer_load_b32 v110 /*v622*/, v90 /*v602*/, s[56:59], null offen
	s_set_vgpr_msb 0x8242
	s_clause 0x1
	buffer_load_b128 v[154:157] /*v[410:413]*/, v74 /*v586*/, s[52:55], null offen
	buffer_load_b128 v[158:161] /*v[414:417]*/, v74 /*v586*/, s[52:55], null offen offset:32
	buffer_load_b128 v[162:165] /*v[418:421]*/, v74 /*v586*/, s[48:51], null offen
	s_set_vgpr_msb 0x4288
	v_or_b32_e32 v10 /*v522*/, 64, v74 /*v586*/
	v_or_b32_e32 v26 /*v538*/, 0x80, v74 /*v586*/
	s_set_vgpr_msb 0x8845
	v_or_b32_e32 v194 /*v450*/, v194 /*v450*/, v255 /*v511*/
	s_set_vgpr_msb 0x4542
	s_clause 0x6
	buffer_load_b128 v[234:237] /*v[490:493]*/, v10 /*v522*/, s[52:55], null offen
	s_set_vgpr_msb 0x4282
	buffer_load_b128 v[2:5] /*v[514:517]*/, v26 /*v538*/, s[52:55], null offen
	s_set_vgpr_msb 0x8242
	buffer_load_b128 v[238:241] /*v[494:497]*/, v74 /*v586*/, s[52:55], null offen offset:96
	s_set_vgpr_msb 0x4286
	buffer_load_b128 v[6:9] /*v[518:521]*/, v74 /*v586*/, s[52:55], null offen offset:160
	buffer_load_b128 v[14:17] /*v[526:529]*/, v74 /*v586*/, s[48:51], null offen offset:96
	v_lshlrev_b32_e32 v91 /*v603*/, 4, v194 /*v450*/
	s_set_vgpr_msb 0x8642
	buffer_load_b128 v[166:169] /*v[422:425]*/, v74 /*v586*/, s[48:51], null offen offset:32
	s_set_vgpr_msb 0x4288
	v_or_b32_e32 v82 /*v594*/, 0xc0, v74 /*v586*/
	s_set_vgpr_msb 0x8842
	buffer_load_b128 v[198:201] /*v[454:457]*/, v91 /*v603*/, s[52:55], null offen offset:32
	buffer_load_b128 v[210:213] /*v[466:469]*/, v91 /*v603*/, s[48:51], null offen
	buffer_load_b128 v[194:197] /*v[450:453]*/, v91 /*v603*/, s[52:55], null offen
	s_set_vgpr_msb 0x428a
	v_or_b32_e32 v42 /*v554*/, 64, v91 /*v603*/
	v_or_b32_e32 v58 /*v570*/, 0x80, v91 /*v603*/
	s_clause 0x1
	buffer_load_b128 v[38:41] /*v[550:553]*/, v91 /*v603*/, s[52:55], null offen offset:96
	buffer_load_b128 v[46:49] /*v[558:561]*/, v91 /*v603*/, s[52:55], null offen offset:160
	s_clause 0x1
	buffer_load_b128 v[54:57] /*v[566:569]*/, v91 /*v603*/, s[48:51], null offen offset:96
	buffer_load_b128 v[62:65] /*v[574:577]*/, v91 /*v603*/, s[48:51], null offen offset:160
	v_add_nc_u32_e32 v86 /*v598*/, 0xe0, v74 /*v586*/
	s_set_vgpr_msb 0x8a42
	buffer_load_b128 v[214:217] /*v[470:473]*/, v91 /*v603*/, s[48:51], null offen offset:32
	s_set_vgpr_msb 0x4282
	buffer_load_b128 v[34:37] /*v[546:549]*/, v42 /*v554*/, s[52:55], null offen
	buffer_load_b128 v[50:53] /*v[562:565]*/, v42 /*v554*/, s[48:51], null offen
	buffer_load_b128 v[42:45] /*v[554:557]*/, v58 /*v570*/, s[52:55], null offen
	s_clause 0x1
	buffer_load_b128 v[30:33] /*v[542:545]*/, v74 /*v586*/, s[48:51], null offen offset:160
	buffer_load_b128 v[58:61] /*v[570:573]*/, v58 /*v570*/, s[48:51], null offen
	buffer_load_b128 v[74:77] /*v[586:589]*/, v82 /*v594*/, s[52:55], null offen
	s_set_vgpr_msb 0x8248
	v_or_b32_e32 v153 /*v409*/, 0xc0, v91 /*v603*/
	s_set_vgpr_msb 0x488a
	buffer_load_b128 v[78:81] /*v[590:593]*/, v86 /*v598*/, s[52:55], null offen
	buffer_load_b128 v[82:85] /*v[594:597]*/, v82 /*v594*/, s[48:51], null offen
	v_add_nc_u32_e32 v102 /*v614*/, 0xe0, v91 /*v603*/
	s_clause 0x2
	buffer_load_b128 v[10:13] /*v[522:525]*/, v10 /*v522*/, s[48:51], null offen
	buffer_load_b128 v[26:29] /*v[538:541]*/, v26 /*v538*/, s[48:51], null offen
	buffer_load_b128 v[86:89] /*v[598:601]*/, v86 /*v598*/, s[48:51], null offen
	buffer_load_b32 v112 /*v624*/, v90 /*v602*/, s[60:63], null offen
	s_set_vgpr_msb 0x8a81
	s_clause 0x2
	buffer_load_b128 v[90:93] /*v[602:605]*/, v153 /*v409*/, s[52:55], null offen
	s_set_vgpr_msb 0x8182
	buffer_load_b128 v[94:97] /*v[606:609]*/, v102 /*v614*/, s[52:55], null offen
	s_set_vgpr_msb 0x8281
	s_clause 0x2
	buffer_load_b128 v[98:101] /*v[610:613]*/, v153 /*v409*/, s[48:51], null offen
	s_set_vgpr_msb 0x8182
	buffer_load_b128 v[102:105] /*v[614:617]*/, v102 /*v614*/, s[48:51], null offen
	s_set_vgpr_msb 0x8244
	s_wait_loadcnt 0x1f
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[242:249], v[154:161] /*v[410:417]*/, 0
	s_set_vgpr_msb 0x4406
	s_wait_loadcnt 0x1e
	ds_store_b128 v1 /*v513*/, v[162:165] /*v[418:421]*/ offset:5120
	s_wait_loadcnt 0x18
	ds_store_b128 v1 /*v513*/, v[166:169] /*v[422:425]*/ offset:5152
	s_set_vgpr_msb 0x645
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[26:33] /*v[282:289]*/, v[154:161] /*v[410:417]*/, 0
	s_set_vgpr_msb 0x4506
	ds_store_b128 v1 /*v513*/, v[154:157] /*v[410:413]*/ offset:13824
	ds_store_b128 v1 /*v513*/, v[158:161] /*v[414:417]*/ offset:13856
	s_set_vgpr_msb 0x60a
	s_wait_loadcnt 0x7
	ds_store_b128 v1 /*v513*/, v[10:13] /*v[522:525]*/ offset:5184
	ds_store_b128 v1 /*v513*/, v[14:17] /*v[526:529]*/ offset:5216
	s_set_vgpr_msb 0xa06
	ds_store_b128 v1 /*v513*/, v[234:237] /*v[490:493]*/ offset:13888
	ds_store_b128 v1 /*v513*/, v[238:241] /*v[494:497]*/ offset:13920
	s_set_vgpr_msb 0x60a
	s_wait_loadcnt 0x6
	ds_store_b128 v1 /*v513*/, v[26:29] /*v[538:541]*/ offset:5248
	s_set_vgpr_msb 0xa44
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[242:249], v[194:201] /*v[450:457]*/, 0
	s_set_vgpr_msb 0x440a
	ds_store_b128 v1 /*v513*/, v[30:33] /*v[542:545]*/ offset:5280
	ds_store_b128 v1 /*v513*/, v[2:5] /*v[514:517]*/ offset:13952
	ds_store_b128 v1 /*v513*/, v[6:9] /*v[518:521]*/ offset:13984
	ds_store_b128 v1 /*v513*/, v[82:85] /*v[594:597]*/ offset:5312
	s_wait_loadcnt 0x5
	ds_store_b128 v1 /*v513*/, v[86:89] /*v[598:601]*/ offset:5344
	ds_store_b128 v1 /*v513*/, v[74:77] /*v[586:589]*/ offset:14016
	ds_store_b128 v1 /*v513*/, v[78:81] /*v[590:593]*/ offset:14048
	s_set_vgpr_msb 0xa85
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[26:33] /*v[282:289]*/, v[194:201] /*v[450:457]*/, 0
	s_set_vgpr_msb 0x8554
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[250:257], v[234:241] /*v[490:497]*/, v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5455
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[34:41] /*v[290:297]*/, v[234:241] /*v[490:497]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5558
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[250:257], v[34:41] /*v[546:553]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x58a9
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[34:41] /*v[290:297]*/, v[34:41] /*v[546:553]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa959
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[2:9] /*v[258:265]*/, v[2:9] /*v[514:521]*/, v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[42:49] /*v[298:305]*/, v[2:9] /*v[514:521]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5945
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[18:25] /*v[274:281]*/, v[162:169] /*v[418:425]*/, 0
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[90:97] /*v[346:353]*/, v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[2:9] /*v[258:265]*/, v[42:49] /*v[554:561]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[42:49] /*v[298:305]*/, v[42:49] /*v[554:561]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa945
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[18:25] /*v[274:281]*/, v[210:217] /*v[466:473]*/, 0
	s_set_vgpr_msb 0x4585
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[90:97] /*v[346:353]*/, v[210:217] /*v[466:473]*/, 0
	s_set_vgpr_msb 0x8559
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[50:57] /*v[306:313]*/, v[74:81] /*v[586:593]*/, v[170:177] /*v[426:433]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5945
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[130:131] /*v[386:387]*/, v[170:171] /*v[426:427]*/
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[58:65] /*v[314:321]*/, v[74:81] /*v[586:593]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[130:131] /*v[386:387]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[130:131] /*v[386:387]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[130:131] /*v[386:387]*/, v[176:177] /*v[432:433]*/
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[108:109] /*v[620:621]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[108:109] /*v[620:621]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[108:109] /*v[620:621]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[66:73] /*v[322:329]*/, v[10:17] /*v[522:529]*/, v[178:185] /*v[434:441]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[130:131] /*v[386:387]*/, v[186:187] /*v[442:443]*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[130:131] /*v[386:387]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[130:131] /*v[386:387]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[130:131] /*v[386:387]*/, v[192:193] /*v[448:449]*/
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[108:109] /*v[620:621]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[170:171] /*v[426:427]*/, v[108:109] /*v[620:621]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, v[108:109] /*v[620:621]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[98:105] /*v[354:361]*/, v[10:17] /*v[522:529]*/, v[202:209] /*v[458:465]*/
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[108:109] /*v[620:621]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, v[108:109] /*v[620:621]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[170:171] /*v[426:427]*/, s[6:7] op_sel_hi:[1,0]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[50:57] /*v[306:313]*/, v[90:97] /*v[602:609]*/, v[218:225] /*v[474:481]*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v234 /*v490*/, v154 /*v410*/
	v_exp_f32_e32 v235 /*v491*/, v155 /*v411*/
	v_exp_f32_e32 v154 /*v410*/, v156 /*v412*/
	v_exp_f32_e32 v155 /*v411*/, v157 /*v413*/
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[58:65] /*v[314:321]*/, v[90:97] /*v[602:609]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa945
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[130:131] /*v[386:387]*/, v[218:219] /*v[474:475]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[130:131] /*v[386:387]*/, v[220:221] /*v[476:477]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[130:131] /*v[386:387]*/, v[222:223] /*v[478:479]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[130:131] /*v[386:387]*/, v[224:225] /*v[480:481]*/
	v_exp_f32_e32 v158 /*v414*/, v158 /*v414*/
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[66:73] /*v[322:329]*/, v[50:57] /*v[562:569]*/, v[226:233] /*v[482:489]*/
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[130:131] /*v[386:387]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[130:131] /*v[386:387]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[130:131] /*v[386:387]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[130:131] /*v[386:387]*/, v[24:25] /*v[536:537]*/
	v_pk_add_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[98:105] /*v[354:361]*/, v[50:57] /*v[562:569]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa959
	v_pk_add_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, s[6:7] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[74:81] /*v[330:337]*/, v[26:33] /*v[538:545]*/, v[178:185] /*v[434:441]*/
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v159 /*v415*/, v159 /*v415*/
	v_exp_f32_e32 v156 /*v412*/, v160 /*v416*/
	v_exp_f32_e32 v157 /*v413*/, v161 /*v417*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[106:113] /*v[362:369]*/, v[26:33] /*v[538:545]*/, v[202:209] /*v[458:465]*/
	v_exp_f32_e32 v170 /*v426*/, v170 /*v426*/
	v_exp_f32_e32 v171 /*v427*/, v171 /*v427*/
	v_exp_f32_e32 v172 /*v428*/, v172 /*v428*/
	v_exp_f32_e32 v173 /*v429*/, v173 /*v429*/
	v_exp_f32_e32 v174 /*v430*/, v174 /*v430*/
	v_exp_f32_e32 v175 /*v431*/, v175 /*v431*/
	v_exp_f32_e32 v160 /*v416*/, v176 /*v432*/
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[74:81] /*v[330:337]*/, v[58:65] /*v[570:577]*/, v[226:233] /*v[482:489]*/
	v_exp_f32_e32 v161 /*v417*/, v177 /*v433*/
	v_exp_f32_e32 v176 /*v432*/, v186 /*v442*/
	v_exp_f32_e32 v177 /*v433*/, v187 /*v443*/
	v_exp_f32_e32 v186 /*v442*/, v188 /*v444*/
	v_exp_f32_e32 v187 /*v443*/, v189 /*v445*/
	v_exp_f32_e32 v188 /*v444*/, v190 /*v446*/
	v_exp_f32_e32 v189 /*v445*/, v191 /*v447*/
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[106:113] /*v[362:369]*/, v[58:65] /*v[570:577]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa959
	v_exp_f32_e32 v190 /*v446*/, v192 /*v448*/
	v_exp_f32_e32 v191 /*v447*/, v193 /*v449*/
	v_exp_f32_e32 v192 /*v448*/, v218 /*v474*/
	v_exp_f32_e32 v193 /*v449*/, v219 /*v475*/
	v_exp_f32_e32 v218 /*v474*/, v220 /*v476*/
	v_exp_f32_e32 v219 /*v475*/, v221 /*v477*/
	v_exp_f32_e32 v220 /*v476*/, v222 /*v478*/
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[434:441]*/, v[82:89] /*v[338:345]*/, v[82:89] /*v[594:601]*/, v[178:185] /*v[434:441]*/
	v_exp_f32_e32 v221 /*v477*/, v223 /*v479*/
	v_exp_f32_e32 v222 /*v478*/, v224 /*v480*/
	v_exp_f32_e32 v223 /*v479*/, v225 /*v481*/
	v_nop
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[178:179] /*v[434:435]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[114:121] /*v[370:377]*/, v[82:89] /*v[594:601]*/, v[202:209] /*v[458:465]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[180:181] /*v[436:437]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[182:183] /*v[438:439]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[184:185] /*v[440:441]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[162:163] /*v[418:419]*/, v[234:235] /*v[490:491]*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v186 /*v442*/, v187 /*v443*/
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[164:165] /*v[420:421]*/, v[154:155] /*v[410:411]*/
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[166:167] /*v[422:423]*/, v[158:159] /*v[414:415]*/
	s_set_vgpr_msb 0x4559
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[82:89] /*v[338:345]*/, v[98:105] /*v[610:617]*/, v[226:233] /*v[482:489]*/
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[202:203] /*v[458:459]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[180:181] /*v[436:437]*/, v[204:205] /*v[460:461]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[206:207] /*v[462:463]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[184:185] /*v[440:441]*/, v[208:209] /*v[464:465]*/, v[106:107] /*v[618:619]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[168:169] /*v[424:425]*/, v[156:157] /*v[412:413]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[170:171] /*v[426:427]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, v[172:173] /*v[428:429]*/
	s_set_vgpr_msb 0x45a9
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[114:121] /*v[370:377]*/, v[98:105] /*v[610:617]*/, v[66:73] /*v[578:585]*/
	s_set_vgpr_msb 0xa949
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[226:227] /*v[482:483]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[228:229] /*v[484:485]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[230:231] /*v[486:487]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[232:233] /*v[488:489]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[184:185] /*v[440:441]*/, v[160:161] /*v[416:417]*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v156 /*v412*/, v157 /*v413*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[226:227] /*v[482:483]*/, v[66:67] /*v[578:579]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[228:229] /*v[484:485]*/, v[68:69] /*v[580:581]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[70:71] /*v[582:583]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[72:73] /*v[584:585]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v156 /*v412*/, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v155 /*v411*/, v154 /*v410*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v154 /*v410*/, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v160 /*v416*/, v161 /*v417*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v172 /*v428*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v170 /*v426*/, v171 /*v427*/
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[202:203] /*v[458:459]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[204:205] /*v[460:461]*/, v[186:187] /*v[442:443]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[206:207] /*v[462:463]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[208:209] /*v[464:465]*/, v[190:191] /*v[446:447]*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v190 /*v446*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v188 /*v444*/, v189 /*v445*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v176 /*v432*/, v177 /*v433*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[226:227] /*v[482:483]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[228:229] /*v[484:485]*/, v[218:219] /*v[474:475]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[230:231] /*v[486:487]*/, v[220:221] /*v[476:477]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[232:233] /*v[488:489]*/, v[222:223] /*v[478:479]*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v192 /*v448*/, v193 /*v449*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[130:131] /*v[386:387]*/, v[224:225] /*v[480:481]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[130:131] /*v[386:387]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[130:131] /*v[386:387]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[130:131] /*v[386:387]*/, v[240:241] /*v[496:497]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[130:131] /*v[386:387]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[130:131] /*v[386:387]*/, v[180:181] /*v[436:437]*/
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[130:131] /*v[386:387]*/, v[182:183] /*v[438:439]*/
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[130:131] /*v[386:387]*/, v[184:185] /*v[440:441]*/
	ds_store_b128 v132 /*v388*/, v[154:157] /*v[410:413]*/
	ds_store_b128 v132 /*v388*/, v[158:161] /*v[414:417]*/ offset:32
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[130:131] /*v[386:387]*/, v[170:171] /*v[426:427]*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[130:131] /*v[386:387]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[130:131] /*v[386:387]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[130:131] /*v[386:387]*/, v[202:203] /*v[458:459]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[130:131] /*v[386:387]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[130:131] /*v[386:387]*/, v[186:187] /*v[442:443]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[130:131] /*v[386:387]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[130:131] /*v[386:387]*/, v[190:191] /*v[446:447]*/
	v_cvt_pk_bf16_f32 v154 /*v410*/, v192 /*v448*/, v193 /*v449*/
	v_cvt_pk_bf16_f32 v155 /*v411*/, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v208 /*v464*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v182 /*v438*/, v183 /*v439*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v184 /*v440*/, v185 /*v441*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v222 /*v478*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v220 /*v476*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v218 /*v474*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v172 /*v428*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v175 /*v431*/, v186 /*v442*/, v187 /*v443*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v188 /*v444*/, v189 /*v445*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v190 /*v446*/, v191 /*v447*/
	ds_store_b128 v132 /*v388*/, v[154:157] /*v[410:413]*/ offset:2560
	ds_store_b128 v132 /*v388*/, v[158:161] /*v[414:417]*/ offset:2592
	s_set_vgpr_msb 0x4506
	ds_store_b128 v1 /*v513*/, v[210:213] /*v[466:469]*/ offset:9472
	ds_store_b128 v1 /*v513*/, v[214:217] /*v[470:473]*/ offset:9504
	ds_store_b128 v1 /*v513*/, v[194:197] /*v[450:453]*/ offset:18176
	ds_store_b128 v1 /*v513*/, v[198:201] /*v[454:457]*/ offset:18208
	s_set_vgpr_msb 0x60a
	ds_store_b128 v1 /*v513*/, v[50:53] /*v[562:565]*/ offset:9536
	ds_store_b128 v1 /*v513*/, v[54:57] /*v[566:569]*/ offset:9568
	ds_store_b128 v1 /*v513*/, v[34:37] /*v[546:549]*/ offset:18240
	ds_store_b128 v1 /*v513*/, v[38:41] /*v[550:553]*/ offset:18272
	ds_store_b128 v1 /*v513*/, v[58:61] /*v[570:573]*/ offset:9600
	ds_store_b128 v1 /*v513*/, v[62:65] /*v[574:577]*/ offset:9632
	ds_store_b128 v1 /*v513*/, v[42:45] /*v[554:557]*/ offset:18304
	ds_store_b128 v1 /*v513*/, v[46:49] /*v[558:561]*/ offset:18336
	ds_store_b128 v1 /*v513*/, v[98:101] /*v[610:613]*/ offset:9664
	ds_store_b128 v1 /*v513*/, v[102:105] /*v[614:617]*/ offset:9696
	ds_store_b128 v1 /*v513*/, v[90:93] /*v[602:605]*/ offset:18368
	ds_store_b128 v1 /*v513*/, v[94:97] /*v[606:609]*/ offset:18400
	s_set_vgpr_msb 0xa45
	ds_store_b128 v132 /*v388*/, v[162:165] /*v[418:421]*/ offset:1280
	ds_store_b128 v132 /*v388*/, v[166:169] /*v[422:425]*/ offset:1312
	ds_store_b128 v132 /*v388*/, v[170:173] /*v[426:429]*/ offset:3840
	ds_store_b128 v132 /*v388*/, v[174:177] /*v[430:433]*/ offset:3872
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v150 /*v406*/ offset:1280
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v135 /*v391*/
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v136 /*v392*/ offset:4352
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v137 /*v393*/
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v138 /*v394*/ offset:4352
	ds_load_tr16_b128 v[218:221] /*v[474:477]*/, v139 /*v395*/
	ds_load_tr16_b128 v[230:233] /*v[486:489]*/, v140 /*v396*/ offset:4352
	ds_load_tr16_b128 v[234:237] /*v[490:493]*/, v141 /*v397*/
	s_set_vgpr_msb 0x4581
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v142 /*v398*/ offset:4352
	ds_load_tr16_b128 v[10:13] /*v[522:525]*/, v143 /*v399*/
	ds_load_tr16_b128 v[22:25] /*v[534:537]*/, v144 /*v400*/ offset:4352
	ds_load_tr16_b128 v[26:29] /*v[538:541]*/, v145 /*v401*/
	ds_load_tr16_b128 v[38:41] /*v[550:553]*/, v146 /*v402*/ offset:4352
	ds_load_tr16_b128 v[42:45] /*v[554:557]*/, v147 /*v403*/
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v135 /*v391*/ offset:4352
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v136 /*v392*/
	ds_load_tr16_b128 v[206:209] /*v[462:465]*/, v137 /*v393*/ offset:4352
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v138 /*v394*/
	ds_load_tr16_b128 v[222:225] /*v[478:481]*/, v139 /*v395*/ offset:4352
	ds_load_tr16_b128 v[226:229] /*v[482:485]*/, v140 /*v396*/
	ds_load_tr16_b128 v[238:241] /*v[494:497]*/, v141 /*v397*/ offset:4352
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v142 /*v398*/
	ds_load_tr16_b128 v[14:17] /*v[526:529]*/, v143 /*v399*/ offset:4352
	ds_load_tr16_b128 v[18:21] /*v[530:533]*/, v144 /*v400*/
	ds_load_tr16_b128 v[30:33] /*v[542:545]*/, v145 /*v401*/ offset:4352
	ds_load_tr16_b128 v[34:37] /*v[546:549]*/, v146 /*v402*/
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v147 /*v403*/ offset:4352
	ds_load_tr16_b128 v[50:53] /*v[562:565]*/, v148 /*v404*/
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v133 /*v389*/
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v133 /*v389*/ offset:4352
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v149 /*v405*/
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v149 /*v405*/ offset:1280
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v134 /*v390*/
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v134 /*v390*/ offset:4352
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v150 /*v406*/
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[234:241], v[162:169] /*v[418:425]*/, v[154:161] /*v[410:417]*/, v[234:241]
	v_wmma_f32_16x16x32_bf16 v[218:225], v[162:169] /*v[418:425]*/, v[186:193] /*v[442:449]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[210:217], v[162:169] /*v[418:425]*/, v[202:209] /*v[458:465]*/, v[210:217]
	v_wmma_f32_16x16x32_bf16 v[186:193], v[162:169] /*v[418:425]*/, v[218:225] /*v[474:481]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[170:177], v[162:169] /*v[418:425]*/, v[234:241] /*v[490:497]*/, v[170:177]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[154:161], v[162:169] /*v[418:425]*/, v[10:17] /*v[522:529]*/, v[154:161]
	v_wmma_f32_16x16x32_bf16 v[138:145], v[162:169] /*v[418:425]*/, v[26:33] /*v[538:545]*/, v[138:145]
	v_wmma_f32_16x16x32_bf16 v[130:137], v[162:169] /*v[418:425]*/, v[42:49] /*v[554:561]*/, v[130:137]
	s_set_vgpr_msb 0x981
	ds_load_tr16_b128 v[54:57] /*v[566:569]*/, v148 /*v404*/ offset:4352
	s_set_vgpr_msb 0x8155
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v151 /*v407*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[178:185] /*v[434:441]*/, v[170:177] /*v[426:433]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[178:185] /*v[434:441]*/, v[194:201] /*v[450:457]*/, v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[226:233], v[178:185] /*v[434:441]*/, v[210:217] /*v[466:473]*/, v[226:233]
	v_wmma_f32_16x16x32_bf16 v[202:209], v[178:185] /*v[434:441]*/, v[226:233] /*v[482:489]*/, v[202:209]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[194:201], v[178:185] /*v[434:441]*/, v[2:9] /*v[514:521]*/, v[194:201]
	v_wmma_f32_16x16x32_bf16 v[178:185], v[178:185] /*v[434:441]*/, v[18:25] /*v[530:537]*/, v[178:185]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[178:185] /*v[434:441]*/, v[34:41] /*v[546:553]*/, v[162:169]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[146:153], v[178:185] /*v[434:441]*/, v[50:57] /*v[562:569]*/, v[146:153]
	s_set_vgpr_msb 0x941
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v151 /*v407*/ offset:1280
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v152 /*v408*/
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v152 /*v408*/ offset:1280
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_wmma_f32_16x16x32_bf16 v[106:113], v[162:169] /*v[418:425]*/, v[154:161] /*v[410:417]*/, v[106:113]
	s_barrier_wait -1
	v_wmma_f32_16x16x32_bf16 v[122:129], v[178:185] /*v[434:441]*/, v[170:177] /*v[426:433]*/, v[122:129]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[162:169] /*v[418:425]*/, v[186:193] /*v[442:449]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[114:121], v[178:185] /*v[434:441]*/, v[194:201] /*v[450:457]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[82:89], v[162:169] /*v[418:425]*/, v[202:209] /*v[458:465]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[98:105], v[178:185] /*v[434:441]*/, v[210:217] /*v[466:473]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[162:169] /*v[418:425]*/, v[218:225] /*v[474:481]*/, v[58:65]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[178:185] /*v[434:441]*/, v[226:233] /*v[482:489]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[162:169] /*v[418:425]*/, v[234:241] /*v[490:497]*/, v[42:49]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[66:73], v[178:185] /*v[434:441]*/, v[2:9] /*v[514:521]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[162:169] /*v[418:425]*/, v[10:17] /*v[522:529]*/, v[26:33]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[178:185] /*v[434:441]*/, v[18:25] /*v[530:537]*/, v[50:57]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[162:169] /*v[418:425]*/, v[26:33] /*v[538:545]*/, v[10:17]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[178:185] /*v[434:441]*/, v[34:41] /*v[546:553]*/, v[34:41]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[162:169] /*v[418:425]*/, v[42:49] /*v[554:561]*/, v[2:9]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[178:185] /*v[434:441]*/, v[50:57] /*v[562:569]*/, v[18:25]
	s_set_vgpr_msb 0x900
	s_cbranch_scc1 .LBB0_6
.LBB0_7:
	s_clause 0x1
	s_load_b64 s[6:7], s[0:1], 0x110 nv
	s_load_b64 s[8:9], s[0:1], 0x140 nv
	s_set_vgpr_msb 8
	s_wait_loadcnt 0x1f
	v_or_b32_e32 v243, 1, v0 /*v512*/
	v_mul_lo_u32 v242, s40, v0 /*v512*/
	s_mul_i32 s4, s40, s67
	s_lshl_b32 s1, s34, 25
	s_add_co_i32 s4, s4, s66
	v_mul_lo_u32 v243, v243, s40
	s_mov_b32 s0, 0
	s_set_vgpr_msb 0x804
	s_wait_loadcnt 0x1e
	v_mul_lo_u32 v247, s40, v252 /*v508*/
	v_mul_lo_u32 v249, s40, v251 /*v507*/
	v_add_lshl_u32 v242, v242, s4, 7
	v_cvt_pk_bf16_f32 v234, v234, s0
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v245, v122 /*v378*/, s0
	s_mov_b32 s2, s46
	v_add_lshl_u32 v243, s4, v243, 7
	v_or_b32_e32 v244, v253 /*v509*/, v242
	s_mov_b32 s3, s47
	v_add_lshl_u32 v247, s4, v247, 7
	s_wait_loadcnt 0x1d
	v_mul_lo_u32 v251, v249 /*v505*/, s40
	v_or_b32_e32 v246, v253 /*v509*/, v243
	s_wait_kmcnt 0x0
	s_or_b64 s[44:45], s[6:7], s[0:1]
	s_or_b64 s[0:1], s[8:9], s[0:1]
	v_lshlrev_b32_e32 v244, 2, v244
	v_cvt_pk_bf16_f32 v248, v123 /*v379*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v235, v235, s0
	v_lshlrev_b32_e32 v246, 2, v246
	v_cvt_pk_bf16_f32 v236, v236, s0
	buffer_store_b16 v234, v244, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v234.l, v248.l
	buffer_store_b16 v245, v244, s[0:3], null offen
	buffer_store_b16 v235, v246, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v235, s4, v249, 7
	s_set_vgpr_msb 5
	v_mul_lo_u32 v249, s40, v250 /*v506*/
	v_cvt_pk_bf16_f32 v245, v124 /*v380*/, s0
	s_set_vgpr_msb 0x504
	buffer_store_b16 v234, v246, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v234, v247, v253 /*v509*/
	v_or_b32_e32 v248, v235, v253 /*v509*/
	v_cvt_pk_bf16_f32 v237, v237, s0
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v250, v125 /*v381*/, s0
	v_mul_lo_u32 v253, v247 /*v503*/, s40
	v_dual_lshlrev_b32 v234, 2, v234 :: v_dual_lshlrev_b32 v248, 2, v248
	v_add_lshl_u32 v249, s4, v249, 7
	v_cvt_pk_bf16_f32 v252, v127 /*v383*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v239, v239, s0
	buffer_store_b16 v236, v234, s[44:47], null offen
	buffer_store_b16 v245, v234, s[0:3], null offen
	buffer_store_b16 v237, v248, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v237, s4, v251, 7
	s_set_vgpr_msb 4
	v_mul_lo_u32 v251, s40, v248 /*v504*/
	v_mov_b16_e64 v236.l, v250.l
	v_or_b32_e32 v245, v249, v253 /*v509*/
	v_cvt_pk_bf16_f32 v241, v241, s0
	v_or_b32_e32 v250, v237, v253 /*v509*/
	v_cvt_pk_bf16_f32 v218, v218, s0
	buffer_store_b16 v236, v248, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v236, v238, s0
	s_set_vgpr_msb 0x401
	v_dual_lshlrev_b32 v238, 2, v245 :: v_dual_lshlrev_b32 v250, 2, v250
	v_cvt_pk_bf16_f32 v245, v126 /*v382*/, s0
	v_add_lshl_u32 v251, s4, v251, 7
	s_set_vgpr_msb 0x104
	v_cvt_pk_bf16_f32 v220, v220, s0
	buffer_store_b16 v236, v238, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v236.l, v252.l
	buffer_store_b16 v245, v238, s[0:3], null offen
	buffer_store_b16 v239, v250, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v239, v251, v253 /*v509*/
	v_add_lshl_u32 v245, v253, s4, 7
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v253, v129 /*v385*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v236, v250, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v236, v240, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v240, v128 /*v384*/, s0
	v_or_b32_e32 v252, v253 /*v509*/, v245
	v_lshlrev_b32_e32 v239, 2, v239
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v219, v219, s0
	v_cvt_pk_bf16_f32 v210, v210, s0
	v_cvt_pk_bf16_f32 v211, v211, s0
	v_lshlrev_b32_e32 v252, 2, v252
	buffer_store_b16 v236, v239, s[44:47], null offen
	buffer_store_b16 v240, v239, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v236.l, v253.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v240, v10 /*v266*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v212, v212, s0
	buffer_store_b16 v241, v252, s[44:47], null offen
	v_cvt_pk_bf16_f32 v214, v214, s0
	buffer_store_b16 v236, v252, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v236, v11 /*v267*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v218, v244, s[44:47], null offen offset:64
	buffer_store_b16 v240, v244, s[0:3], null offen offset:64
	buffer_store_b16 v219, v246, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v219.l, v220.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v220, v12 /*v268*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v218.l, v236.l
	v_cvt_pk_bf16_f32 v213, v213, s0
	v_cvt_pk_bf16_f32 v186, v186, s0
	v_cvt_pk_bf16_f32 v202, v202, s0
	v_cvt_pk_bf16_f32 v187, v187, s0
	buffer_store_b16 v218, v246, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v218, v221, s0
	buffer_store_b16 v219, v234, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v219.l, v220.l
	v_cvt_pk_bf16_f32 v220, v222, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v222, v14 /*v270*/, s0
	v_cvt_pk_bf16_f32 v221, v13 /*v269*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v188, v188, s0
	buffer_store_b16 v219, v234, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v219.l, v220.l
	buffer_store_b16 v218, v248, s[44:47], null offen offset:64
	buffer_store_b16 v221, v248, s[0:3], null offen offset:64
	s_wait_xcnt 0x1
	v_mov_b16_e64 v218.l, v222.l
	v_cvt_pk_bf16_f32 v220, v223, s0
	v_cvt_pk_bf16_f32 v222, v225, s0
	buffer_store_b16 v219, v238, s[44:47], null offen offset:64
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v219, v15 /*v271*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v218, v238, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v218.l, v220.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v220, v16 /*v272*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v221, v224, s0
	v_cvt_pk_bf16_f32 v204, v204, s0
	v_cvt_pk_bf16_f32 v190, v190, s0
	buffer_store_b16 v218, v250, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v218.l, v220.l
	buffer_store_b16 v219, v250, s[0:3], null offen offset:64
	buffer_store_b16 v221, v239, s[44:47], null offen offset:64
	s_wait_xcnt 0x1
	v_mov_b16_e64 v219.l, v222.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v220, v17 /*v273*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v192, v192, s0
	buffer_store_b16 v218, v239, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v218, v226, s0
	buffer_store_b16 v219, v252, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v219.l, v220.l
	v_cvt_pk_bf16_f32 v220, v227, s0
	v_cvt_pk_bf16_f32 v194, v194, s0
	v_cvt_pk_bf16_f32 v170, v170, s0
	v_cvt_pk_bf16_f32 v171, v171, s0
	buffer_store_b16 v219, v252, s[0:3], null offen offset:64
	buffer_store_b16 v210, v244, s[44:47], null offen offset:128
	buffer_store_b16 v218, v244, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_mov_b16_e64 v210.l, v220.l
	buffer_store_b16 v211, v246, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v211, v228, s0
	v_cvt_pk_bf16_f32 v172, v172, s0
	v_cvt_pk_bf16_f32 v173, v173, s0
	buffer_store_b16 v210, v246, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v210.l, v212.l
	v_cvt_pk_bf16_f32 v212, v229, s0
	v_cvt_pk_bf16_f32 v154, v154, s0
	v_cvt_pk_bf16_f32 v156, v156, s0
	v_cvt_pk_bf16_f32 v138, v138, s0
	buffer_store_b16 v210, v234, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v210.l, v212.l
	buffer_store_b16 v211, v234, s[0:3], null offen offset:128
	buffer_store_b16 v213, v248, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_mov_b16_e64 v211.l, v214.l
	v_cvt_pk_bf16_f32 v212, v230, s0
	v_cvt_pk_bf16_f32 v214, v232, s0
	buffer_store_b16 v210, v248, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v210, v215, s0
	buffer_store_b16 v211, v238, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v211.l, v212.l
	v_cvt_pk_bf16_f32 v212, v216, s0
	v_cvt_pk_bf16_f32 v213, v231, s0
	v_cvt_pk_bf16_f32 v139, v139, s0
	v_cvt_pk_bf16_f32 v140, v140, s0
	buffer_store_b16 v211, v238, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v211.l, v212.l
	buffer_store_b16 v210, v250, s[44:47], null offen offset:128
	buffer_store_b16 v213, v250, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_mov_b16_e64 v210.l, v214.l
	v_or_b32_e32 v214, v243, v0
	v_or_b32_e32 v212, v242, v0
	buffer_store_b16 v211, v239, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v211, v217, s0
	v_cvt_pk_bf16_f32 v213, v233, s0
	buffer_store_b16 v210, v239, s[0:3], null offen offset:128
	v_cvt_pk_bf16_f32 v130, v130, s0
	v_cvt_pk_bf16_f32 v131, v131, s0
	buffer_store_b16 v211, v252, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_or_b32_e32 v211, v247, v0
	v_lshlrev_b32_e32 v210, 2, v212
	v_mov_b16_e64 v212.l, v213.l
	v_cvt_pk_bf16_f32 v133, v133, s0
	v_cvt_pk_bf16_f32 v106, v106, s0
	v_lshlrev_b32_e32 v211, 2, v211
	v_or_b32_e32 v213, 0xc0, v210
	buffer_store_b16 v212, v252, s[0:3], null offen offset:128
	buffer_store_b16 v186, v213, s[44:47], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v186, 2, v214
	buffer_store_b16 v202, v213, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v202, v203, s0
	v_or_b32_e32 v212, 0xc0, v211
	v_or_b32_e32 v213, v235, v0
	v_or_b32_e32 v203, 0xc0, v186
	v_cvt_pk_bf16_f32 v122, v122, s0
	v_cvt_pk_bf16_f32 v123, v123, s0
	v_cvt_pk_bf16_f32 v107, v107, s0
	v_cvt_pk_bf16_f32 v108, v108, s0
	buffer_store_b16 v187, v203, s[44:47], null offen
	buffer_store_b16 v202, v203, s[0:3], null offen
	buffer_store_b16 v188, v212, s[44:47], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v202, v249, v0
	v_mov_b16_e64 v187.l, v204.l
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v188, 2, v213
	v_cvt_pk_bf16_f32 v204, v206, s0
	v_cvt_pk_bf16_f32 v206, v208, s0
	v_lshlrev_b32_e32 v202, 2, v202
	buffer_store_b16 v187, v212, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v187, v189, s0
	v_cvt_pk_bf16_f32 v189, v205, s0
	v_or_b32_e32 v203, 0xc0, v188
	v_or_b32_e32 v205, 0xc0, v202
	v_cvt_pk_bf16_f32 v109, v109, s0
	v_cvt_pk_bf16_f32 v125, v125, s0
	v_mul_lo_u32 v1, s40, v1
	buffer_store_b16 v187, v203, s[44:47], null offen
	buffer_store_b16 v189, v203, s[0:3], null offen
	buffer_store_b16 v190, v205, s[44:47], null offen
	buffer_store_b16 v204, v205, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v205, v245, v0
	v_or_b32_e32 v187, v237, v0
	v_or_b32_e32 v189, v251, v0
	v_cvt_pk_bf16_f32 v190, v191, s0
	v_cvt_pk_bf16_f32 v191, v207, s0
	v_cvt_pk_bf16_f32 v127, v127, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_lshlrev_b32 v187, 2, v187 :: v_dual_lshlrev_b32 v189, 2, v189
	v_cvt_pk_bf16_f32 v111, v111, s0
	v_add_lshl_u32 v1, s4, v1, 7
	v_cvt_pk_bf16_f32 v113, v113, s0
	v_or_b32_e32 v203, 0xc0, v187
	v_or_b32_e32 v204, 0xc0, v189
	buffer_store_b16 v190, v203, s[44:47], null offen
	buffer_store_b16 v191, v203, s[0:3], null offen
	s_wait_xcnt 0x1
	v_lshlrev_b32_e32 v190, 2, v205
	s_wait_xcnt 0x0
	v_mov_b16_e64 v191.l, v206.l
	buffer_store_b16 v192, v204, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v192, v193, s0
	v_cvt_pk_bf16_f32 v193, v209, s0
	v_or_b32_e32 v203, 0xc0, v190
	buffer_store_b16 v191, v204, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v191.l, v194.l
	buffer_store_b16 v192, v203, s[44:47], null offen
	buffer_store_b16 v193, v203, s[0:3], null offen
	buffer_store_b16 v170, v244, s[44:47], null offen offset:256
	buffer_store_b16 v191, v244, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v191, v196, s0
	v_cvt_pk_bf16_f32 v170, v195, s0
	buffer_store_b16 v171, v246, s[44:47], null offen offset:256
	buffer_store_b16 v170, v246, s[0:3], null offen offset:256
	buffer_store_b16 v172, v234, s[44:47], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e64 v171.l, v191.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v170, v197, s0
	buffer_store_b16 v171, v234, s[0:3], null offen offset:256
	buffer_store_b16 v173, v248, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v173, v175, s0
	v_cvt_pk_bf16_f32 v171, v174, s0
	v_cvt_pk_bf16_f32 v174, v199, s0
	buffer_store_b16 v170, v248, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v172, v198, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v173.l
	buffer_store_b16 v171, v238, s[44:47], null offen offset:256
	buffer_store_b16 v172, v238, s[0:3], null offen offset:256
	v_mov_b16_e64 v173.l, v174.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v171, v176, s0
	buffer_store_b16 v170, v250, s[44:47], null offen offset:256
	buffer_store_b16 v173, v250, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v173, v201, s0
	v_cvt_pk_bf16_f32 v170, v200, s0
	buffer_store_b16 v171, v239, s[44:47], null offen offset:256
	v_cvt_pk_bf16_f32 v172, v177, s0
	buffer_store_b16 v170, v239, s[0:3], null offen offset:256
	buffer_store_b16 v172, v252, s[44:47], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e64 v171.l, v173.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v170, v178, s0
	buffer_store_b16 v171, v252, s[0:3], null offen offset:256
	buffer_store_b16 v154, v244, s[44:47], null offen offset:320
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v171, v180, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v154, v155, s0
	v_cvt_pk_bf16_f32 v155, v179, s0
	buffer_store_b16 v170, v244, s[0:3], null offen offset:320
	buffer_store_b16 v154, v246, s[44:47], null offen offset:320
	buffer_store_b16 v155, v246, s[0:3], null offen offset:320
	s_wait_xcnt 0x2
	v_mov_b16_e64 v170.l, v171.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v154, v157, s0
	v_cvt_pk_bf16_f32 v157, v182, s0
	buffer_store_b16 v156, v234, s[44:47], null offen offset:320
	buffer_store_b16 v170, v234, s[0:3], null offen offset:320
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v156, v158, s0
	v_cvt_pk_bf16_f32 v158, v159, s0
	v_cvt_pk_bf16_f32 v155, v181, s0
	buffer_store_b16 v154, v248, s[44:47], null offen offset:320
	s_wait_xcnt 0x0
	v_mov_b16_e64 v154.l, v157.l
	buffer_store_b16 v155, v248, s[0:3], null offen offset:320
	buffer_store_b16 v156, v238, s[44:47], null offen offset:320
	v_mov_b16_e64 v157.l, v158.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v155, v183, s0
	buffer_store_b16 v154, v238, s[0:3], null offen offset:320
	buffer_store_b16 v157, v250, s[44:47], null offen offset:320
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v157, v161, s0
	v_cvt_pk_bf16_f32 v158, v185, s0
	v_cvt_pk_bf16_f32 v154, v160, s0
	buffer_store_b16 v155, v250, s[0:3], null offen offset:320
	v_cvt_pk_bf16_f32 v156, v184, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v155.l, v157.l
	v_mov_b16_e64 v157.l, v158.l
	buffer_store_b16 v154, v239, s[44:47], null offen offset:320
	buffer_store_b16 v156, v239, s[0:3], null offen offset:320
	buffer_store_b16 v155, v252, s[44:47], null offen offset:320
	buffer_store_b16 v157, v252, s[0:3], null offen offset:320
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v155, v163, s0
	v_cvt_pk_bf16_f32 v154, v162, s0
	buffer_store_b16 v138, v244, s[44:47], null offen offset:384
	buffer_store_b16 v154, v244, s[0:3], null offen offset:384
	buffer_store_b16 v139, v246, s[44:47], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e64 v138.l, v155.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v139, v164, s0
	buffer_store_b16 v138, v246, s[0:3], null offen offset:384
	buffer_store_b16 v140, v234, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v138, v141, s0
	v_cvt_pk_bf16_f32 v141, v142, s0
	v_cvt_pk_bf16_f32 v142, v166, s0
	buffer_store_b16 v139, v234, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v140, v165, s0
	buffer_store_b16 v138, v248, s[44:47], null offen offset:384
	buffer_store_b16 v140, v248, s[0:3], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e64 v139.l, v141.l
	v_mov_b16_e64 v141.l, v142.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v138, v143, s0
	buffer_store_b16 v139, v238, s[44:47], null offen offset:384
	buffer_store_b16 v141, v238, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v141, v168, s0
	v_cvt_pk_bf16_f32 v142, v145, s0
	v_cvt_pk_bf16_f32 v139, v167, s0
	v_cvt_pk_bf16_f32 v140, v144, s0
	buffer_store_b16 v138, v250, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v138.l, v141.l
	v_mov_b16_e64 v141.l, v142.l
	buffer_store_b16 v139, v250, s[0:3], null offen offset:384
	buffer_store_b16 v140, v239, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v139, v169, s0
	buffer_store_b16 v138, v239, s[0:3], null offen offset:384
	buffer_store_b16 v141, v252, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v138, v146, s0
	v_or_b32_e32 v140, 0x1c0, v210
	s_wait_xcnt 0x0
	v_or_b32_e32 v141, 0x1c0, v186
	buffer_store_b16 v139, v252, s[0:3], null offen offset:384
	buffer_store_b16 v130, v140, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v130, v147, s0
	buffer_store_b16 v138, v140, s[0:3], null offen
	buffer_store_b16 v131, v141, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v131, v132, s0
	v_cvt_pk_bf16_f32 v132, v148, s0
	v_or_b32_e32 v138, 0x1c0, v211
	v_or_b32_e32 v139, 0x1c0, v188
	buffer_store_b16 v130, v141, s[0:3], null offen
	buffer_store_b16 v131, v138, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v130, v149, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v131, v134, s0
	v_cvt_pk_bf16_f32 v134, v135, s0
	v_cvt_pk_bf16_f32 v135, v151, s0
	buffer_store_b16 v132, v138, s[0:3], null offen
	buffer_store_b16 v133, v139, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v133, 0x1c0, v202
	v_cvt_pk_bf16_f32 v132, v150, s0
	v_or_b32_e32 v138, 0x1c0, v187
	buffer_store_b16 v130, v139, s[0:3], null offen
	buffer_store_b16 v131, v133, s[44:47], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v130.l, v135.l
	buffer_store_b16 v132, v133, s[0:3], null offen
	buffer_store_b16 v134, v138, s[44:47], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v134, 1, v254 /*v510*/
	v_cvt_pk_bf16_f32 v131, v136, s0
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v133, 0x1c0, v189
	buffer_store_b16 v130, v138, s[0:3], null offen
	s_set_vgpr_msb 4
	v_mul_lo_u32 v130, s40, v254 /*v510*/
	v_mul_lo_u32 v134, v134, s40
	v_cvt_pk_bf16_f32 v135, v137, s0
	v_cvt_pk_bf16_f32 v137, v153, s0
	v_cvt_pk_bf16_f32 v132, v152, s0
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v136, 0x1c0, v190
	buffer_store_b16 v131, v133, s[44:47], null offen
	buffer_store_b16 v132, v133, s[0:3], null offen
	v_add_lshl_u32 v130, s4, v130, 7
	s_wait_xcnt 0x1
	v_mov_b16_e64 v131.l, v137.l
	s_wait_xcnt 0x0
	v_add_lshl_u32 v133, s4, v134, 7
	buffer_store_b16 v135, v136, s[44:47], null offen
	s_set_vgpr_msb 4
	v_mul_lo_u32 v134, s40, v246 /*v502*/
	v_or_b32_e32 v132, v130, v253 /*v509*/
	buffer_store_b16 v131, v136, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mul_lo_u32 v135, s40, v245 /*v501*/
	v_cvt_pk_bf16_f32 v90, v90, s0
	v_cvt_pk_bf16_f32 v92, v92, s0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v131, 2, v132
	s_set_vgpr_msb 4
	v_or_b32_e32 v132, v133, v253 /*v509*/
	v_cvt_pk_bf16_f32 v91, v91, s0
	v_cvt_pk_bf16_f32 v82, v82, s0
	v_cvt_pk_bf16_f32 v83, v83, s0
	v_cvt_pk_bf16_f32 v84, v84, s0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v132, 2, v132
	buffer_store_b16 v106, v131, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v106.l, v123.l
	v_add_lshl_u32 v123, s4, v134, 7
	buffer_store_b16 v122, v131, s[0:3], null offen
	buffer_store_b16 v107, v132, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v107, s4, v135, 7
	v_cvt_pk_bf16_f32 v122, v124, s0
	buffer_store_b16 v106, v132, s[0:3], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v106, v123, v253 /*v509*/
	v_mul_lo_u32 v134, s40, v244 /*v500*/
	v_or_b32_e32 v124, v107, v253 /*v509*/
	v_mul_lo_u32 v135, s40, v243 /*v499*/
	v_cvt_pk_bf16_f32 v86, v86, s0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v106, 2, v106
	v_cvt_pk_bf16_f32 v85, v85, s0
	v_lshlrev_b32_e32 v124, 2, v124
	v_cvt_pk_bf16_f32 v58, v58, s0
	v_add_lshl_u32 v134, s4, v134, 7
	buffer_store_b16 v108, v106, s[44:47], null offen
	buffer_store_b16 v122, v106, s[0:3], null offen
	buffer_store_b16 v109, v124, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v109, s4, v135, 7
	v_mov_b16_e32 v108.l, v125.l
	s_set_vgpr_msb 4
	v_or_b32_e32 v122, v134, v253 /*v509*/
	v_cvt_pk_bf16_f32 v74, v74, s0
	v_cvt_pk_bf16_f32 v59, v59, s0
	v_or_b32_e32 v125, v109, v253 /*v509*/
	v_cvt_pk_bf16_f32 v60, v60, s0
	v_cvt_pk_bf16_f32 v76, v76, s0
	v_cvt_pk_bf16_f32 v62, v62, s0
	v_cvt_pk_bf16_f32 v64, v64, s0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v125, 2, v125
	buffer_store_b16 v108, v124, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v108, v110, s0
	v_lshlrev_b32_e32 v110, 2, v122
	v_cvt_pk_bf16_f32 v122, v126, s0
	s_set_vgpr_msb 4
	v_mul_lo_u32 v126, s40, v242 /*v498*/
	v_cvt_pk_bf16_f32 v42, v42, s0
	v_cvt_pk_bf16_f32 v43, v43, s0
	buffer_store_b16 v108, v110, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v108.l, v127.l
	buffer_store_b16 v122, v110, s[0:3], null offen
	buffer_store_b16 v111, v125, s[44:47], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v122, v1, v253 /*v509*/
	v_cvt_pk_bf16_f32 v127, v129, s0
	v_add_lshl_u32 v126, v126, s4, 7
	buffer_store_b16 v108, v125, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v108, v112, s0
	v_cvt_pk_bf16_f32 v112, v128, s0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v122, 2, v122
	s_set_vgpr_msb 4
	v_or_b32_e32 v111, v126, v253 /*v509*/
	v_cvt_pk_bf16_f32 v45, v45, s0
	v_cvt_pk_bf16_f32 v26, v26, s0
	v_cvt_pk_bf16_f32 v28, v28, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v111, 2, v111
	buffer_store_b16 v108, v111, s[44:47], null offen
	buffer_store_b16 v112, v111, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e32 v108.l, v127.l
	buffer_store_b16 v113, v122, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v112, v114, s0
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_cvt_pk_bf16_f32 v12, v12, s0
	buffer_store_b16 v108, v122, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v108, v115, s0
	buffer_store_b16 v90, v131, s[44:47], null offen offset:64
	buffer_store_b16 v112, v131, s[0:3], null offen offset:64
	buffer_store_b16 v91, v132, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e32 v91.l, v92.l
	v_cvt_pk_bf16_f32 v92, v116, s0
	v_mov_b16_e32 v90.l, v108.l
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_cvt_pk_bf16_f32 v3, v3, s0
	buffer_store_b16 v90, v132, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v90, v93, s0
	buffer_store_b16 v91, v106, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e32 v91.l, v92.l
	v_cvt_pk_bf16_f32 v92, v94, s0
	v_cvt_pk_bf16_f32 v94, v118, s0
	v_cvt_pk_bf16_f32 v93, v117, s0
	buffer_store_b16 v91, v106, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e32 v91.l, v92.l
	buffer_store_b16 v90, v124, s[44:47], null offen offset:64
	buffer_store_b16 v93, v124, s[0:3], null offen offset:64
	s_wait_xcnt 0x1
	v_mov_b16_e32 v90.l, v94.l
	v_cvt_pk_bf16_f32 v92, v95, s0
	v_cvt_pk_bf16_f32 v94, v97, s0
	buffer_store_b16 v91, v110, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v91, v119, s0
	buffer_store_b16 v90, v110, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e32 v90.l, v92.l
	v_cvt_pk_bf16_f32 v92, v120, s0
	v_cvt_pk_bf16_f32 v93, v96, s0
	buffer_store_b16 v90, v125, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e32 v90.l, v92.l
	buffer_store_b16 v91, v125, s[0:3], null offen offset:64
	buffer_store_b16 v93, v111, s[44:47], null offen offset:64
	s_wait_xcnt 0x1
	v_mov_b16_e32 v91.l, v94.l
	v_cvt_pk_bf16_f32 v92, v121, s0
	buffer_store_b16 v90, v111, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v90, v98, s0
	buffer_store_b16 v91, v122, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e32 v91.l, v92.l
	v_cvt_pk_bf16_f32 v92, v99, s0
	buffer_store_b16 v91, v122, s[0:3], null offen offset:64
	buffer_store_b16 v82, v131, s[44:47], null offen offset:128
	buffer_store_b16 v90, v131, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_mov_b16_e32 v82.l, v92.l
	buffer_store_b16 v83, v132, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v83, v100, s0
	buffer_store_b16 v82, v132, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v82.l, v84.l
	v_cvt_pk_bf16_f32 v84, v101, s0
	buffer_store_b16 v82, v106, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v82.l, v84.l
	buffer_store_b16 v83, v106, s[0:3], null offen offset:128
	buffer_store_b16 v85, v124, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_mov_b16_e32 v83.l, v86.l
	v_cvt_pk_bf16_f32 v84, v102, s0
	v_cvt_pk_bf16_f32 v86, v104, s0
	buffer_store_b16 v82, v124, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v82, v87, s0
	buffer_store_b16 v83, v110, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v83.l, v84.l
	v_cvt_pk_bf16_f32 v84, v88, s0
	v_cvt_pk_bf16_f32 v85, v103, s0
	buffer_store_b16 v83, v110, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v83.l, v84.l
	buffer_store_b16 v82, v125, s[44:47], null offen offset:128
	buffer_store_b16 v85, v125, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_mov_b16_e32 v82.l, v86.l
	v_or_b32_e32 v86, v133, v0
	v_or_b32_e32 v84, v130, v0
	buffer_store_b16 v83, v111, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v83, v89, s0
	v_cvt_pk_bf16_f32 v85, v105, s0
	buffer_store_b16 v82, v111, s[0:3], null offen offset:128
	buffer_store_b16 v83, v122, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_or_b32_e32 v83, v123, v0
	v_lshlrev_b32_e32 v82, 2, v84
	v_mov_b16_e32 v84.l, v85.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v83, 2, v83
	v_or_b32_e32 v85, 0xc0, v82
	buffer_store_b16 v84, v122, s[0:3], null offen offset:128
	buffer_store_b16 v58, v85, s[44:47], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v58, 2, v86
	buffer_store_b16 v74, v85, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v74, v75, s0
	v_or_b32_e32 v84, 0xc0, v83
	v_or_b32_e32 v85, v107, v0
	v_or_b32_e32 v75, 0xc0, v58
	buffer_store_b16 v59, v75, s[44:47], null offen
	buffer_store_b16 v74, v75, s[0:3], null offen
	buffer_store_b16 v60, v84, s[44:47], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v74, v134, v0
	v_mov_b16_e32 v59.l, v76.l
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v60, 2, v85
	v_cvt_pk_bf16_f32 v76, v78, s0
	v_lshlrev_b32_e32 v74, 2, v74
	buffer_store_b16 v59, v84, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v59, v61, s0
	v_cvt_pk_bf16_f32 v61, v77, s0
	v_or_b32_e32 v75, 0xc0, v60
	v_or_b32_e32 v77, 0xc0, v74
	buffer_store_b16 v59, v75, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v59, v109, v0
	buffer_store_b16 v61, v75, s[0:3], null offen
	buffer_store_b16 v62, v77, s[44:47], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v61, v126, v0
	buffer_store_b16 v76, v77, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v62, v63, s0
	v_dual_lshlrev_b32 v59, 2, v59 :: v_dual_bitop2_b32 v0, v1, v0 bitop3:0x54
	v_lshlrev_b32_e32 v61, 2, v61
	v_mov_b16_e32 v1.l, v64.l
	v_cvt_pk_bf16_f32 v64, v80, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v75, 0xc0, v59
	v_cvt_pk_bf16_f32 v63, v79, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v76, 0xc0, v61
	buffer_store_b16 v62, v75, s[44:47], null offen
	buffer_store_b16 v63, v75, s[0:3], null offen
	v_lshlrev_b32_e32 v0, 2, v0
	buffer_store_b16 v1, v76, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v64.l
	v_cvt_pk_bf16_f32 v62, v65, s0
	v_cvt_pk_bf16_f32 v65, v66, s0
	v_or_b32_e32 v64, 0xc0, v0
	v_cvt_pk_bf16_f32 v63, v81, s0
	buffer_store_b16 v1, v76, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v42.l
	v_mov_b16_e32 v42.l, v65.l
	buffer_store_b16 v62, v64, s[44:47], null offen
	buffer_store_b16 v63, v64, s[0:3], null offen
	buffer_store_b16 v1, v131, s[44:47], null offen offset:256
	buffer_store_b16 v42, v131, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v42, v44, s0
	v_cvt_pk_bf16_f32 v44, v68, s0
	v_cvt_pk_bf16_f32 v1, v67, s0
	buffer_store_b16 v43, v132, s[44:47], null offen offset:256
	buffer_store_b16 v1, v132, s[0:3], null offen offset:256
	buffer_store_b16 v42, v106, s[44:47], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e32 v43.l, v44.l
	v_mov_b16_e32 v44.l, v45.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v69, s0
	buffer_store_b16 v43, v106, s[0:3], null offen offset:256
	buffer_store_b16 v44, v124, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v44, v47, s0
	v_cvt_pk_bf16_f32 v45, v71, s0
	v_cvt_pk_bf16_f32 v42, v46, s0
	buffer_store_b16 v1, v124, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v43, v70, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v44.l
	v_mov_b16_e32 v44.l, v45.l
	buffer_store_b16 v42, v110, s[44:47], null offen offset:256
	buffer_store_b16 v43, v110, s[0:3], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v42, v48, s0
	buffer_store_b16 v1, v125, s[44:47], null offen offset:256
	buffer_store_b16 v44, v125, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v44, v73, s0
	v_cvt_pk_bf16_f32 v1, v72, s0
	v_cvt_pk_bf16_f32 v43, v49, s0
	buffer_store_b16 v42, v111, s[44:47], null offen offset:256
	buffer_store_b16 v1, v111, s[0:3], null offen offset:256
	buffer_store_b16 v43, v122, s[44:47], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e32 v42.l, v44.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v50, s0
	buffer_store_b16 v42, v122, s[0:3], null offen offset:256
	buffer_store_b16 v26, v131, s[44:47], null offen offset:320
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v42, v52, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v26, v27, s0
	buffer_store_b16 v1, v131, s[0:3], null offen offset:320
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v28.l
	v_cvt_pk_bf16_f32 v27, v51, s0
	v_mov_b16_e32 v28.l, v42.l
	buffer_store_b16 v26, v132, s[44:47], null offen offset:320
	buffer_store_b16 v27, v132, s[0:3], null offen offset:320
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v26, v29, s0
	buffer_store_b16 v1, v106, s[44:47], null offen offset:320
	buffer_store_b16 v28, v106, s[0:3], null offen offset:320
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v28, v54, s0
	v_cvt_pk_bf16_f32 v29, v31, s0
	v_cvt_pk_bf16_f32 v1, v53, s0
	buffer_store_b16 v26, v124, s[44:47], null offen offset:320
	v_cvt_pk_bf16_f32 v27, v30, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v26.l, v28.l
	v_mov_b16_e32 v28.l, v29.l
	buffer_store_b16 v1, v124, s[0:3], null offen offset:320
	buffer_store_b16 v27, v110, s[44:47], null offen offset:320
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v55, s0
	buffer_store_b16 v26, v110, s[0:3], null offen offset:320
	buffer_store_b16 v28, v125, s[44:47], null offen offset:320
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v26, v32, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v28, v33, s0
	v_cvt_pk_bf16_f32 v27, v56, s0
	v_cvt_pk_bf16_f32 v29, v57, s0
	buffer_store_b16 v1, v125, s[0:3], null offen offset:320
	buffer_store_b16 v26, v111, s[44:47], null offen offset:320
	buffer_store_b16 v27, v111, s[0:3], null offen offset:320
	s_wait_xcnt 0x2
	v_mov_b16_e32 v1.l, v28.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v26, v35, s0
	v_mov_b16_e32 v28.l, v29.l
	buffer_store_b16 v1, v122, s[44:47], null offen offset:320
	buffer_store_b16 v28, v122, s[0:3], null offen offset:320
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v34, s0
	buffer_store_b16 v10, v131, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v10.l, v26.l
	buffer_store_b16 v1, v131, s[0:3], null offen offset:384
	buffer_store_b16 v11, v132, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v36, s0
	buffer_store_b16 v10, v132, s[0:3], null offen offset:384
	buffer_store_b16 v12, v106, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v12, v14, s0
	v_cvt_pk_bf16_f32 v10, v13, s0
	v_cvt_pk_bf16_f32 v13, v38, s0
	buffer_store_b16 v1, v106, s[0:3], null offen offset:384
	v_cvt_pk_bf16_f32 v11, v37, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v12.l
	buffer_store_b16 v10, v124, s[44:47], null offen offset:384
	buffer_store_b16 v11, v124, s[0:3], null offen offset:384
	v_mov_b16_e32 v12.l, v13.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v10, v15, s0
	buffer_store_b16 v1, v110, s[44:47], null offen offset:384
	buffer_store_b16 v12, v110, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v12, v40, s0
	v_cvt_pk_bf16_f32 v13, v17, s0
	v_cvt_pk_bf16_f32 v1, v39, s0
	buffer_store_b16 v10, v125, s[44:47], null offen offset:384
	v_cvt_pk_bf16_f32 v11, v16, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v10.l, v12.l
	v_mov_b16_e32 v12.l, v13.l
	buffer_store_b16 v1, v125, s[0:3], null offen offset:384
	buffer_store_b16 v11, v111, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v41, s0
	buffer_store_b16 v10, v111, s[0:3], null offen offset:384
	buffer_store_b16 v12, v122, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v10, v18, s0
	v_or_b32_e32 v11, 0x1c0, v82
	s_wait_xcnt 0x0
	v_or_b32_e32 v12, 0x1c0, v58
	buffer_store_b16 v1, v122, s[0:3], null offen offset:384
	buffer_store_b16 v2, v11, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v19, s0
	buffer_store_b16 v10, v11, s[0:3], null offen
	buffer_store_b16 v3, v12, s[44:47], null offen
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v2, v4, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v20, s0
	v_cvt_pk_bf16_f32 v4, v5, s0
	v_or_b32_e32 v5, 0x1c0, v83
	v_or_b32_e32 v10, 0x1c0, v60
	buffer_store_b16 v1, v12, s[0:3], null offen
	buffer_store_b16 v2, v5, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v21, s0
	buffer_store_b16 v3, v5, s[0:3], null offen
	buffer_store_b16 v4, v10, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v7, s0
	v_cvt_pk_bf16_f32 v7, v23, s0
	v_cvt_pk_bf16_f32 v2, v6, s0
	v_or_b32_e32 v5, 0x1c0, v74
	v_cvt_pk_bf16_f32 v3, v22, s0
	v_or_b32_e32 v6, 0x1c0, v59
	buffer_store_b16 v1, v10, s[0:3], null offen
	buffer_store_b16 v2, v5, s[44:47], null offen
	buffer_store_b16 v3, v5, s[0:3], null offen
	buffer_store_b16 v4, v6, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v1.l, v7.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v2, v8, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v4, 0x1c0, v61
	v_cvt_pk_bf16_f32 v3, v24, s0
	v_cvt_pk_bf16_f32 v5, v9, s0
	v_or_b32_e32 v0, 0x1c0, v0
	v_cvt_pk_bf16_f32 v7, v25, s0
	buffer_store_b16 v1, v6, s[0:3], null offen
	buffer_store_b16 v2, v4, s[44:47], null offen
	buffer_store_b16 v3, v4, s[0:3], null offen
	buffer_store_b16 v5, v0, s[44:47], null offen
	buffer_store_b16 v7, v0, s[0:3], null offen
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	k_dkdv_0, .Lfunc_end0-k_dkdv_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dkdv_0
		.amdhsa_group_segment_fixed_size 22528
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 408
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
		.amdhsa_next_free_vgpr 636
		.amdhsa_next_free_sgpr 78
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-k_dkdv_0)<<4)&4080)>>4
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

	.set .Lk_dkdv_0.num_vgpr, 636
	.set .Lk_dkdv_0.num_agpr, 0
	.set .Lk_dkdv_0.numbered_sgpr, 78
	.set .Lk_dkdv_0.num_named_barrier, 0
	.set .Lk_dkdv_0.private_seg_size, 0
	.set .Lk_dkdv_0.uses_vcc, 1
	.set .Lk_dkdv_0.uses_flat_scratch, 0
	.set .Lk_dkdv_0.has_dyn_sized_stack, 0
	.set .Lk_dkdv_0.has_recursion, 0
	.set .Lk_dkdv_0.has_indirect_call, 0
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
      - .offset:         404
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 22528
    .kernarg_segment_align: 8
    .kernarg_segment_size: 408
    .max_flat_workgroup_size: 64
    .name:           k_dkdv_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 64
      - 1
      - 1
    .sgpr_count:     80
    .sgpr_spill_count: 0
    .symbol:         k_dkdv_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     636
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
