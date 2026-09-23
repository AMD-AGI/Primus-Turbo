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
	v_dual_lshrrev_b32 v145 /*v401*/, 4, v0 :: v_dual_bitop2_b32 v141 /*v397*/, 15, v0 bitop3:0x40
	s_cselect_b32 s8, s6, s5
	s_cselect_b32 s4, s3, s4
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_wait_kmcnt 0x0
	s_mul_i32 s68, s38, s8
	s_mul_i32 s2, ttmp9, s2
	s_load_b64 s[44:45], s[0:1], 0x30 nv
	s_add_co_i32 s5, s3, s2
	s_cmp_eq_u32 s7, 0
	s_load_b64 s[2:3], s[0:1], 0x190 nv
	s_cselect_b32 s67, ttmp9, s5
	s_lshr_b32 s5, s42, 31
	s_lshl_b32 s9, s4, 5
	s_add_co_i32 s5, s42, s5
	s_set_vgpr_msb 0x4004
	v_or_b32_e32 v1, s9, v141 /*v397*/
	s_and_b32 s4, s5, -2
	s_ashr_i32 s5, s5, 1
	s_cmp_lg_u32 s42, s4
	s_clause 0x1
	s_load_b64 s[48:49], s[0:1], 0x90 nv
	s_load_b64 s[52:53], s[0:1], 0x0 nv
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s42, 0
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v2, 16, v1
	s_cselect_b32 s6, -1, 0
	s_mul_i32 s69, s41, s67
	s_and_b32 s4, s6, s4
	s_sub_co_ci_u32 s6, s5, 0
	s_sub_co_i32 s7, s9, s43
	s_mul_i32 s70, s39, s8
	s_max_i32 s7, s7, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s7, s7, 5
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s10, -1, 0
	s_and_b32 s10, s10, exec_lo
	s_cselect_b32 s72, s7, 0
	s_cmp_lg_u32 s4, 0
	s_sub_co_ci_u32 s73, s5, s72
	s_or_b32 s4, s9, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s4, s4, s43
	s_add_co_i32 s5, s4, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s7, s5, 31
	s_lshr_b32 s7, s7, 27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s7, s5, s7
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
	s_mov_b32 s10, 0
	s_cselect_b32 s4, s5, 0
	s_min_i32 s4, s4, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s4, s4, s72
	s_max_i32 s4, s4, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_min_i32 s4, s4, s73
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s75, -1, 0
	s_and_b32 s2, s75, exec_lo
	s_cselect_b32 s74, s4, 0
	s_lshl_b32 s66, s40, 4
	s_mul_i32 s4, s38, s40
	s_mul_i32 s2, s66, s68
	s_mul_i32 s38, s74, s41
	s_lshl4_add_u32 s2, s67, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u32 v1, s66, v1, s2
	v_mad_u32 v2, s66, v2, s2
	s_mul_i32 s2, s4, s3
	s_load_b64 s[4:5], s[0:1], 0x60 nv
	s_lshl_b32 s34, s2, 8
	s_mul_i32 s2, s39, s37
	s_ashr_i32 s35, s34, 31
	s_mul_i32 s11, s2, s3
	s_set_vgpr_msb 4
	v_or_b32_e32 v1, v1, v145 /*v401*/
	v_or_b32_e32 v2, v2, v145 /*v401*/
	s_lshr_b64 s[46:47], s[34:35], 7
	s_lshl_b32 s2, s11, 8
	s_mov_b32 s6, s46
	s_set_vgpr_msb 0x400
	v_dual_lshlrev_b32 v1, 4, v1 :: v_dual_lshlrev_b32 v2, 4, v2
	s_mov_b32 s7, s47
	s_clause 0x3
	buffer_load_b128 v[242:245], v1, s[44:47], null offen
	buffer_load_b128 v[246:249], v1, s[44:47], null offen offset:32
	buffer_load_b128 v[250:253], v1, s[44:47], null offen offset:64
	buffer_load_b128 v[254:257], v1, s[44:47], null offen offset:96
	v_add_nc_u32_e32 v3, 0xa0, v1
	v_add_nc_u32_e32 v4, 0xc0, v1
	v_add_nc_u32_e32 v5, 0xe0, v1
	v_add_nc_u32_e32 v6, 0xa0, v2
	s_set_vgpr_msb 64
	s_clause 0x1
	buffer_load_b128 v[18:21] /*v[274:277]*/, v1, s[44:47], null offen offset:128
	buffer_load_b128 v[22:25] /*v[278:281]*/, v3, s[44:47], null offen
	s_wait_kmcnt 0x0
	s_clause 0x1
	buffer_load_b128 v[26:29] /*v[282:285]*/, v1, s[4:7], null offen offset:128
	buffer_load_b128 v[30:33] /*v[286:289]*/, v3, s[4:7], null offen
	s_clause 0x1
	buffer_load_b128 v[34:37] /*v[290:293]*/, v4, s[44:47], null offen
	buffer_load_b128 v[38:41] /*v[294:297]*/, v5, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[42:45] /*v[298:301]*/, v4, s[4:7], null offen
	buffer_load_b128 v[46:49] /*v[302:305]*/, v5, s[4:7], null offen
	s_clause 0x1
	buffer_load_b128 v[50:53] /*v[306:309]*/, v2, s[44:47], null offen offset:128
	buffer_load_b128 v[54:57] /*v[310:313]*/, v6, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[62:65] /*v[318:321]*/, v6, s[4:7], null offen
	buffer_load_b128 v[58:61] /*v[314:317]*/, v2, s[4:7], null offen offset:128
	s_clause 0x3
	buffer_load_b128 v[66:69] /*v[322:325]*/, v2, s[44:47], null offen
	buffer_load_b128 v[70:73] /*v[326:329]*/, v2, s[44:47], null offen offset:32
	buffer_load_b128 v[74:77] /*v[330:333]*/, v2, s[44:47], null offen offset:64
	buffer_load_b128 v[78:81] /*v[334:337]*/, v2, s[44:47], null offen offset:96
	s_clause 0x3
	buffer_load_b128 v[2:5] /*v[258:261]*/, v1, s[4:7], null offen
	buffer_load_b128 v[6:9] /*v[262:265]*/, v1, s[4:7], null offen offset:32
	buffer_load_b128 v[10:13] /*v[266:269]*/, v1, s[4:7], null offen offset:64
	buffer_load_b128 v[14:17] /*v[270:273]*/, v1, s[4:7], null offen offset:96
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v1, 0xc0, v2
	v_add_nc_u32_e32 v3, 0xe0, v2
	s_set_vgpr_msb 64
	s_clause 0x3
	buffer_load_b128 v[90:93] /*v[346:349]*/, v2, s[4:7], null offen
	buffer_load_b128 v[94:97] /*v[350:353]*/, v2, s[4:7], null offen offset:32
	buffer_load_b128 v[98:101] /*v[354:357]*/, v2, s[4:7], null offen offset:64
	buffer_load_b128 v[102:105] /*v[358:361]*/, v2, s[4:7], null offen offset:96
	s_clause 0x1
	buffer_load_b128 v[106:109] /*v[362:365]*/, v1, s[44:47], null offen
	buffer_load_b128 v[110:113] /*v[366:369]*/, v3, s[44:47], null offen
	s_clause 0x1
	buffer_load_b128 v[114:117] /*v[370:373]*/, v1, s[4:7], null offen
	buffer_load_b128 v[118:121] /*v[374:377]*/, v3, s[4:7], null offen
	s_wait_xcnt 0x0
	s_clause 0x1
	s_load_b64 s[4:5], s[0:1], 0xc0 nv
	s_load_b64 s[6:7], s[0:1], 0xe8 nv
	s_set_vgpr_msb 0x4004
	v_lshlrev_b32_e32 v1, 3, v145 /*v401*/
	s_set_vgpr_msb 0x400
	v_and_b32_e32 v2, 0x70, v0
	v_bfe_u32 v3, v0, 3, 1
	s_lshl_b32 s12, s11, 2
	s_lshl_b32 s35, s39, 4
	s_set_vgpr_msb 64
	v_and_or_b32 v149 /*v405*/, v0, 7, v1
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s13, s12, 31
	s_lshl_b32 s11, s11, 27
	v_dual_add_nc_u32 v144 /*v400*/, s9, v1 :: v_dual_lshlrev_b32 v147 /*v403*/, 4, v3
	s_set_vgpr_msb 0x4044
	v_mad_u32_u24 v146 /*v402*/, 0x110, v141 /*v397*/, v2
	v_mul_u32_u24_e32 v148 /*v404*/, 0x50, v149 /*v405*/
	s_mul_i32 s71, s37, s35
	s_lshr_b64 s[54:55], s[2:3], 7
	s_lshr_b64 s[58:59], s[12:13], 7
	s_mul_i32 s71, s71, s8
	s_wait_kmcnt 0x0
	s_or_b64 s[56:57], s[4:5], s[10:11]
	s_or_b64 s[60:61], s[6:7], s[10:11]
	s_cmp_lt_i32 s38, 1
	s_set_vgpr_msb 0x4400
	s_cbranch_scc1 .LBB0_3
	s_abs_i32 s76, s41
	s_movk_i32 s2, 0x1400
	s_cvt_f32_u32 s3, s76
	s_set_vgpr_msb 0x44
	v_mad_u32_u24 v152 /*v408*/, 0x110, v149 /*v405*/, s2
	s_movk_i32 s2, 0x3600
	s_set_vgpr_msb 0x4404
	v_dual_mov_b32 v234, 0 :: v_dual_bitop2_b32 v1, 3, v144 /*v400*/ bitop3:0x54
	v_s_rcp_f32 s3, s3
	s_set_vgpr_msb 0x444
	v_mad_u32_u24 v153 /*v409*/, 0x110, v149 /*v405*/, s2
	v_dual_add_nc_u32 v150 /*v406*/, 16, v144 /*v400*/ :: v_dual_bitop2_b32 v130 /*v386*/, 2, v144 /*v400*/ bitop3:0x54
	s_set_vgpr_msb 0x4400
	v_lshlrev_b32_e32 v2, 1, v0
	s_set_vgpr_msb 0x54
	v_mad_i32_i24 v151 /*v407*/, 0xffffff40, v141 /*v397*/, v146 /*v402*/
	v_dual_mov_b32 v85 /*v341*/, v234 :: v_dual_bitop2_b32 v155 /*v411*/, 64, v147 /*v403*/ bitop3:0x54
	s_mul_f32 s2, s3, 0x4f7ffffe
	v_dual_mov_b32 v126 /*v382*/, v234 :: v_dual_bitop2_b32 v135 /*v391*/, 3, v150 /*v406*/ bitop3:0x54
	s_movk_i32 s3, 0xa00
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cvt_u32_f32 s2, s2
	v_mad_u32_u24 v161 /*v417*/, 0x50, v149 /*v405*/, s3
	s_sub_co_i32 s3, 0, s76
	v_dual_mov_b32 v127 /*v383*/, v234 :: v_dual_bitop2_b32 v136 /*v392*/, 2, v150 /*v406*/ bitop3:0x54
	s_mul_i32 s3, s3, s2
	v_dual_mov_b32 v125 /*v381*/, v234 :: v_dual_bitop2_b32 v134 /*v390*/, 6, v144 /*v400*/ bitop3:0x54
	v_dual_mov_b32 v123 /*v379*/, v234 :: v_dual_bitop2_b32 v132 /*v388*/, 4, v144 /*v400*/ bitop3:0x54
	v_dual_mov_b32 v122 /*v378*/, v234 :: v_dual_bitop2_b32 v131 /*v387*/, 5, v144 /*v400*/ bitop3:0x54
	v_dual_mov_b32 v124 /*v380*/, v234 :: v_dual_bitop2_b32 v133 /*v389*/, 7, v144 /*v400*/ bitop3:0x54
	v_dual_mov_b32 v128 /*v384*/, v234 :: v_dual_bitop2_b32 v137 /*v393*/, 5, v150 /*v406*/ bitop3:0x54
	v_dual_mov_b32 v129 /*v385*/, v234 :: v_dual_bitop2_b32 v138 /*v394*/, 4, v150 /*v406*/ bitop3:0x54
	v_dual_mov_b32 v82 /*v338*/, v234 :: v_dual_bitop2_b32 v139 /*v395*/, 7, v150 /*v406*/ bitop3:0x54
	v_dual_mov_b32 v83 /*v339*/, v234 :: v_dual_bitop2_b32 v140 /*v396*/, 6, v150 /*v406*/ bitop3:0x54
	v_dual_mov_b32 v84 /*v340*/, v234 :: v_dual_bitop2_b32 v154 /*v410*/, 32, v147 /*v403*/ bitop3:0x54
	v_or_b32_e32 v156 /*v412*/, 0x60, v147 /*v403*/
	v_or_b32_e32 v157 /*v413*/, 0x80, v147 /*v403*/
	v_or_b32_e32 v158 /*v414*/, 0xa0, v147 /*v403*/
	v_or_b32_e32 v159 /*v415*/, 0xc0, v147 /*v403*/
	s_set_vgpr_msb 0x5400
	v_mov_b32_e32 v235, v234
	s_set_vgpr_msb 64
	v_and_or_b32 v160 /*v416*/, v2, 16, 0xe0
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v236, v234 :: v_dual_mov_b32 v237, v234
	v_dual_mov_b32 v238, v234 :: v_dual_mov_b32 v239, v234
	v_dual_mov_b32 v240, v234 :: v_dual_mov_b32 v241, v234
	v_dual_mov_b32 v218, v234 :: v_dual_mov_b32 v219, v234
	v_dual_mov_b32 v220, v234 :: v_dual_mov_b32 v221, v234
	v_dual_mov_b32 v222, v234 :: v_dual_mov_b32 v223, v234
	v_dual_mov_b32 v224, v234 :: v_dual_mov_b32 v225, v234
	v_dual_mov_b32 v202, v234 :: v_dual_mov_b32 v203, v234
	v_dual_mov_b32 v204, v234 :: v_dual_mov_b32 v205, v234
	v_dual_mov_b32 v206, v234 :: v_dual_mov_b32 v207, v234
	v_dual_mov_b32 v208, v234 :: v_dual_mov_b32 v209, v234
	v_dual_mov_b32 v186, v234 :: v_dual_mov_b32 v187, v234
	v_dual_mov_b32 v188, v234 :: v_dual_mov_b32 v189, v234
	v_dual_mov_b32 v190, v234 :: v_dual_mov_b32 v191, v234
	v_dual_mov_b32 v192, v234 :: v_dual_mov_b32 v193, v234
	v_dual_mov_b32 v170, v234 :: v_dual_mov_b32 v171, v234
	v_dual_mov_b32 v172, v234 :: v_dual_mov_b32 v173, v234
	v_dual_mov_b32 v174, v234 :: v_dual_mov_b32 v175, v234
	v_dual_mov_b32 v176, v234 :: v_dual_mov_b32 v177, v234
	v_dual_mov_b32 v154, v234 :: v_dual_mov_b32 v155, v234
	v_dual_mov_b32 v156, v234 :: v_dual_mov_b32 v157, v234
	v_dual_mov_b32 v158, v234 :: v_dual_mov_b32 v159, v234
	v_dual_mov_b32 v160, v234 :: v_dual_mov_b32 v161, v234
	v_dual_mov_b32 v138, v234 :: v_dual_mov_b32 v139, v234
	v_dual_mov_b32 v140, v234 :: v_dual_mov_b32 v141, v234
	v_dual_mov_b32 v142, v234 :: v_dual_mov_b32 v143, v234
	v_dual_mov_b32 v144, v234 :: v_dual_mov_b32 v145, v234
	v_dual_mov_b32 v130, v234 :: v_dual_mov_b32 v131, v234
	v_dual_mov_b32 v132, v234 :: v_dual_mov_b32 v133, v234
	v_dual_mov_b32 v134, v234 :: v_dual_mov_b32 v135, v234
	v_dual_mov_b32 v136, v234 :: v_dual_mov_b32 v137, v234
	v_dual_mov_b32 v106, v234 :: v_dual_mov_b32 v107, v234
	v_dual_mov_b32 v108, v234 :: v_dual_mov_b32 v109, v234
	v_dual_mov_b32 v110, v234 :: v_dual_mov_b32 v111, v234
	v_dual_mov_b32 v112, v234 :: v_dual_mov_b32 v113, v234
	v_dual_mov_b32 v90, v234 :: v_dual_mov_b32 v91, v234
	v_dual_mov_b32 v92, v234 :: v_dual_mov_b32 v93, v234
	v_dual_mov_b32 v94, v234 :: v_dual_mov_b32 v95, v234
	v_dual_mov_b32 v96, v234 :: v_dual_mov_b32 v97, v234
	v_dual_mov_b32 v74, v234 :: v_dual_mov_b32 v75, v234
	v_dual_mov_b32 v76, v234 :: v_dual_mov_b32 v77, v234
	v_dual_mov_b32 v78, v234 :: v_dual_mov_b32 v79, v234
	v_dual_mov_b32 v80, v234 :: v_dual_mov_b32 v81, v234
	v_dual_mov_b32 v58, v234 :: v_dual_mov_b32 v59, v234
	v_dual_mov_b32 v60, v234 :: v_dual_mov_b32 v61, v234
	v_dual_mov_b32 v62, v234 :: v_dual_mov_b32 v63, v234
	v_dual_mov_b32 v64, v234 :: v_dual_mov_b32 v65, v234
	v_dual_mov_b32 v42, v234 :: v_dual_mov_b32 v43, v234
	v_dual_mov_b32 v44, v234 :: v_dual_mov_b32 v45, v234
	v_dual_mov_b32 v46, v234 :: v_dual_mov_b32 v47, v234
	v_dual_mov_b32 v48, v234 :: v_dual_mov_b32 v49, v234
	v_dual_mov_b32 v26, v234 :: v_dual_mov_b32 v27, v234
	v_dual_mov_b32 v28, v234 :: v_dual_mov_b32 v29, v234
	v_dual_mov_b32 v30, v234 :: v_dual_mov_b32 v31, v234
	v_dual_mov_b32 v32, v234 :: v_dual_mov_b32 v33, v234
	v_dual_mov_b32 v10, v234 :: v_dual_mov_b32 v11, v234
	v_dual_mov_b32 v12, v234 :: v_dual_mov_b32 v13, v234
	v_dual_mov_b32 v14, v234 :: v_dual_mov_b32 v15, v234
	v_dual_mov_b32 v16, v234 :: v_dual_mov_b32 v17, v234
	v_dual_mov_b32 v2, v234 :: v_dual_mov_b32 v3, v234
	v_dual_mov_b32 v4, v234 :: v_dual_mov_b32 v5, v234
	v_dual_mov_b32 v6, v234 :: v_dual_mov_b32 v7, v234
	v_dual_mov_b32 v8, v234 :: v_dual_mov_b32 v9, v234
	s_set_vgpr_msb 64
	v_dual_mov_b32 v86 /*v342*/, v234 :: v_dual_mov_b32 v87 /*v343*/, v234
	v_dual_mov_b32 v88 /*v344*/, v234 :: v_dual_mov_b32 v89 /*v345*/, v234
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v226, v234 :: v_dual_mov_b32 v227, v234
	v_dual_mov_b32 v228, v234 :: v_dual_mov_b32 v229, v234
	v_dual_mov_b32 v230, v234 :: v_dual_mov_b32 v231, v234
	v_dual_mov_b32 v232, v234 :: v_dual_mov_b32 v233, v234
	v_dual_mov_b32 v210, v234 :: v_dual_mov_b32 v211, v234
	v_dual_mov_b32 v212, v234 :: v_dual_mov_b32 v213, v234
	v_dual_mov_b32 v214, v234 :: v_dual_mov_b32 v215, v234
	v_dual_mov_b32 v216, v234 :: v_dual_mov_b32 v217, v234
	v_dual_mov_b32 v194, v234 :: v_dual_mov_b32 v195, v234
	v_dual_mov_b32 v196, v234 :: v_dual_mov_b32 v197, v234
	v_dual_mov_b32 v198, v234 :: v_dual_mov_b32 v199, v234
	v_dual_mov_b32 v200, v234 :: v_dual_mov_b32 v201, v234
	v_dual_mov_b32 v178, v234 :: v_dual_mov_b32 v179, v234
	v_dual_mov_b32 v180, v234 :: v_dual_mov_b32 v181, v234
	v_dual_mov_b32 v182, v234 :: v_dual_mov_b32 v183, v234
	v_dual_mov_b32 v184, v234 :: v_dual_mov_b32 v185, v234
	v_dual_mov_b32 v162, v234 :: v_dual_mov_b32 v163, v234
	v_dual_mov_b32 v164, v234 :: v_dual_mov_b32 v165, v234
	v_dual_mov_b32 v166, v234 :: v_dual_mov_b32 v167, v234
	v_dual_mov_b32 v168, v234 :: v_dual_mov_b32 v169, v234
	v_dual_mov_b32 v146, v234 :: v_dual_mov_b32 v147, v234
	v_dual_mov_b32 v148, v234 :: v_dual_mov_b32 v149, v234
	v_dual_mov_b32 v150, v234 :: v_dual_mov_b32 v151, v234
	v_dual_mov_b32 v152, v234 :: v_dual_mov_b32 v153, v234
	v_dual_mov_b32 v122, v234 :: v_dual_mov_b32 v123, v234
	v_dual_mov_b32 v124, v234 :: v_dual_mov_b32 v125, v234
	v_dual_mov_b32 v126, v234 :: v_dual_mov_b32 v127, v234
	v_dual_mov_b32 v128, v234 :: v_dual_mov_b32 v129, v234
	v_dual_mov_b32 v114, v234 :: v_dual_mov_b32 v115, v234
	v_dual_mov_b32 v116, v234 :: v_dual_mov_b32 v117, v234
	v_dual_mov_b32 v118, v234 :: v_dual_mov_b32 v119, v234
	v_dual_mov_b32 v120, v234 :: v_dual_mov_b32 v121, v234
	v_dual_mov_b32 v98, v234 :: v_dual_mov_b32 v99, v234
	v_dual_mov_b32 v100, v234 :: v_dual_mov_b32 v101, v234
	v_dual_mov_b32 v102, v234 :: v_dual_mov_b32 v103, v234
	v_dual_mov_b32 v104, v234 :: v_dual_mov_b32 v105, v234
	v_dual_mov_b32 v82, v234 :: v_dual_mov_b32 v83, v234
	v_dual_mov_b32 v84, v234 :: v_dual_mov_b32 v85, v234
	v_dual_mov_b32 v86, v234 :: v_dual_mov_b32 v87, v234
	v_dual_mov_b32 v88, v234 :: v_dual_mov_b32 v89, v234
	v_dual_mov_b32 v66, v234 :: v_dual_mov_b32 v67, v234
	v_dual_mov_b32 v68, v234 :: v_dual_mov_b32 v69, v234
	v_dual_mov_b32 v70, v234 :: v_dual_mov_b32 v71, v234
	v_dual_mov_b32 v72, v234 :: v_dual_mov_b32 v73, v234
	v_dual_mov_b32 v50, v234 :: v_dual_mov_b32 v51, v234
	v_dual_mov_b32 v52, v234 :: v_dual_mov_b32 v53, v234
	v_dual_mov_b32 v54, v234 :: v_dual_mov_b32 v55, v234
	v_dual_mov_b32 v56, v234 :: v_dual_mov_b32 v57, v234
	v_dual_mov_b32 v34, v234 :: v_dual_mov_b32 v35, v234
	v_dual_mov_b32 v36, v234 :: v_dual_mov_b32 v37, v234
	v_dual_mov_b32 v38, v234 :: v_dual_mov_b32 v39, v234
	v_dual_mov_b32 v40, v234 :: v_dual_mov_b32 v41, v234
	v_dual_mov_b32 v18, v234 :: v_dual_mov_b32 v19, v234
	v_dual_mov_b32 v20, v234 :: v_dual_mov_b32 v21, v234
	v_dual_mov_b32 v22, v234 :: v_dual_mov_b32 v23, v234
	v_dual_mov_b32 v24, v234 :: v_dual_mov_b32 v25, v234
	s_mul_hi_u32 s3, s2, s3
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_ashr_i32 s77, s41, 31
	s_add_co_i32 s78, s2, s3
	s_mov_b64 s[64:65], 0
	s_mov_b32 s42, 0x3fb8aa3b
	s_ashr_i32 s39, s38, 31
	s_mov_b32 s50, s54
	s_mov_b32 s51, s55
.LBB0_2:
	s_abs_i32 s2, s64
	s_ashr_i32 s3, s64, 31
	s_mul_hi_u32 s4, s2, s78
	s_xor_b32 s3, s3, s77
	s_mul_i32 s5, s4, s76
	s_add_co_i32 s6, s4, 1
	s_sub_co_i32 s2, s2, s5
	s_mov_b32 s62, s58
	s_sub_co_i32 s5, s2, s76
	s_cmp_ge_u32 s2, s76
	s_mov_b32 s63, s59
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s2, s5, s2
	s_add_co_i32 s5, s4, 1
	s_cmp_ge_u32 s2, s76
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, s[44:45]
	s_cselect_b32 s2, s5, s4
	s_set_vgpr_msb 0x4095
	v_dual_add_nc_u32 v99 /*v611*/, v152 /*v408*/, v147 /*v403*/ :: v_dual_add_nc_u32 v101 /*v613*/, v153 /*v409*/, v147 /*v403*/
	s_xor_b32 s2, s2, s3
	v_dual_add_nc_u32 v103 /*v615*/, v152 /*v408*/, v154 /*v410*/ :: v_dual_add_nc_u32 v105 /*v617*/, v153 /*v409*/, v154 /*v410*/
	s_sub_co_i32 s4, s2, s3
	v_dual_add_nc_u32 v106 /*v618*/, v152 /*v408*/, v155 /*v411*/ :: v_dual_add_nc_u32 v107 /*v619*/, v152 /*v408*/, v156 /*v412*/
	s_mul_i32 s4, s4, s41
	v_dual_add_nc_u32 v108 /*v620*/, v152 /*v408*/, v157 /*v413*/ :: v_dual_add_nc_u32 v109 /*v621*/, v152 /*v408*/, v158 /*v414*/
	s_cmp_lg_u32 s64, s4
	v_dual_add_nc_u32 v110 /*v622*/, v152 /*v408*/, v159 /*v415*/ :: v_dual_add_nc_u32 v111 /*v623*/, v152 /*v408*/, v160 /*v416*/
	s_cselect_b32 s4, -1, 0
	s_xor_b32 s5, s41, s64
	v_dual_add_nc_u32 v117 /*v629*/, v161 /*v417*/, v147 /*v403*/ :: v_dual_add_nc_u32 v120 /*v632*/, v148 /*v404*/, v154 /*v410*/
	s_cmp_lt_i32 s5, 0
	v_dual_add_nc_u32 v114 /*v626*/, v153 /*v409*/, v156 /*v412*/ :: v_dual_add_nc_u32 v115 /*v627*/, v153 /*v409*/, v157 /*v413*/
	s_cselect_b32 s5, -1, 0
	v_dual_add_nc_u32 v116 /*v628*/, v153 /*v409*/, v158 /*v414*/ :: v_dual_add_nc_u32 v118 /*v630*/, v153 /*v409*/, v159 /*v415*/
	s_and_b32 s4, s5, s4
	s_sub_co_ci_u32 s2, s2, s3
	v_add_nc_u32_e32 v119 /*v631*/, v153 /*v409*/, v160 /*v416*/
	s_mul_i32 s3, s2, s41
	s_add_co_i32 s2, s2, s72
	s_sub_co_i32 s3, s64, s3
	v_lshl_or_b32 v10 /*v522*/, s2, 5, v141 /*v397*/
	s_add_co_i32 s2, s3, s69
	v_add_nc_u32_e32 v121 /*v633*/, v161 /*v417*/, v154 /*v410*/
	s_lshl4_add_u32 s3, s2, s71
	s_add_co_i32 s2, s2, s70
	s_set_vgpr_msb 0x9542
	v_mad_u32 v162 /*v418*/, v10 /*v522*/, s35, s3
	s_set_vgpr_msb 0x4288
	v_dual_add_nc_u32 v42 /*v554*/, s43, v10 /*v522*/ :: v_dual_bitop2_b32 v34 /*v546*/, 16, v10 /*v522*/ bitop3:0x54
	s_mul_i32 s2, s2, s37
	s_set_vgpr_msb 0x8885
	v_add_nc_u32_e32 v112 /*v624*/, v148 /*v404*/, v147 /*v403*/
	s_set_vgpr_msb 0x8588
	v_add_lshl_u32 v43 /*v555*/, s2, v10 /*v522*/, 2
	s_set_vgpr_msb 0x8842
	v_mad_u32 v202 /*v458*/, v34 /*v546*/, s35, s3
	s_set_vgpr_msb 0x4288
	v_add_lshl_u32 v83 /*v595*/, s2, v34 /*v546*/, 2
	s_set_vgpr_msb 0x8845
	v_or_b32_e32 v162 /*v418*/, v162 /*v418*/, v145 /*v401*/
	s_set_vgpr_msb 0x4585
	v_add_nc_u32_e32 v113 /*v625*/, v153 /*v409*/, v155 /*v411*/
	s_set_vgpr_msb 0x8589
	v_add_nc_u32_e32 v74 /*v586*/, s43, v34 /*v546*/
	v_cmp_ge_i32_e32 vcc_lo, v144 /*v400*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s2, v144 /*v400*/, v42 /*v554*/
	s_set_vgpr_msb 0x8984
	v_lshlrev_b32_e32 v66 /*v578*/, 4, v162 /*v418*/
	s_set_vgpr_msb 0x8445
	v_or_b32_e32 v202 /*v458*/, v202 /*v458*/, v145 /*v401*/
	s_set_vgpr_msb 0x4542
	s_clause 0x1
	buffer_load_b128 v[162:165] /*v[418:421]*/, v66 /*v578*/, s[52:55], null offen
	buffer_load_b128 v[166:169] /*v[422:425]*/, v66 /*v578*/, s[52:55], null offen offset:32
	s_set_vgpr_msb 0x4284
	v_lshlrev_b32_e32 v82 /*v594*/, 4, v202 /*v458*/
	s_set_vgpr_msb 0x8442
	s_clause 0x1
	buffer_load_b128 v[178:181] /*v[434:437]*/, v66 /*v578*/, s[48:51], null offen
	buffer_load_b128 v[182:185] /*v[438:441]*/, v66 /*v578*/, s[48:51], null offen offset:32
	s_clause 0x3
	buffer_load_b128 v[250:253] /*v[506:509]*/, v66 /*v578*/, s[52:55], null offen offset:64
	buffer_load_b128 v[254:257] /*v[510:513]*/, v66 /*v578*/, s[52:55], null offen offset:96
	buffer_load_b128 v[202:205] /*v[458:461]*/, v82 /*v594*/, s[52:55], null offen
	buffer_load_b128 v[206:209] /*v[462:465]*/, v82 /*v594*/, s[52:55], null offen offset:32
	s_set_vgpr_msb 0x428a
	v_or_b32_e32 v44 /*v556*/, 0x80, v66 /*v578*/
	v_add_nc_u32_e32 v46 /*v558*/, 0xa0, v66 /*v578*/
	s_clause 0x1
	buffer_load_b128 v[10:13] /*v[522:525]*/, v66 /*v578*/, s[48:51], null offen offset:64
	buffer_load_b128 v[14:17] /*v[526:529]*/, v66 /*v578*/, s[48:51], null offen offset:96
	s_clause 0x3
	buffer_load_b128 v[18:21] /*v[530:533]*/, v82 /*v594*/, s[52:55], null offen offset:64
	buffer_load_b128 v[22:25] /*v[534:537]*/, v82 /*v594*/, s[52:55], null offen offset:96
	buffer_load_b128 v[34:37] /*v[546:549]*/, v44 /*v556*/, s[52:55], null offen
	buffer_load_b128 v[38:41] /*v[550:553]*/, v46 /*v558*/, s[52:55], null offen
	v_or_b32_e32 v58 /*v570*/, 0x80, v82 /*v594*/
	v_add_nc_u32_e32 v62 /*v574*/, 0xa0, v82 /*v594*/
	v_add_nc_u32_e32 v75 /*v587*/, 0xc0, v66 /*v578*/
	v_add_nc_u32_e32 v78 /*v590*/, 0xe0, v66 /*v578*/
	s_set_vgpr_msb 0x8a42
	s_clause 0x1
	buffer_load_b128 v[226:229] /*v[482:485]*/, v82 /*v594*/, s[48:51], null offen
	buffer_load_b128 v[230:233] /*v[486:489]*/, v82 /*v594*/, s[48:51], null offen offset:32
	v_cmp_lt_i32_e64 s3, v42 /*v554*/, v1
	s_set_vgpr_msb 0x4209
	v_cmp_gt_i32_e64 s4, v130 /*v386*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s5, v131 /*v387*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s6, v132 /*v388*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s7, v133 /*v389*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s8, v134 /*v390*/, v42 /*v554*/
	v_cmp_ge_i32_e64 s9, v150 /*v406*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s10, v150 /*v406*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s11, v135 /*v391*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s12, v136 /*v392*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s13, v137 /*v393*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s14, v138 /*v394*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s15, v139 /*v395*/, v42 /*v554*/
	v_cmp_gt_i32_e64 s16, v140 /*v396*/, v42 /*v554*/
	s_set_vgpr_msb 0x98a
	buffer_load_b32 v98 /*v610*/, v43 /*v555*/, s[56:59], null offen
	s_clause 0x1
	buffer_load_b32 v100 /*v612*/, v43 /*v555*/, s[60:63], null offen
	buffer_load_b32 v102 /*v614*/, v83 /*v595*/, s[60:63], null offen
	s_clause 0x1
	buffer_load_b128 v[42:45] /*v[554:557]*/, v44 /*v556*/, s[48:51], null offen
	buffer_load_b128 v[46:49] /*v[558:561]*/, v46 /*v558*/, s[48:51], null offen
	s_clause 0x3
	buffer_load_b128 v[50:53] /*v[562:565]*/, v58 /*v570*/, s[52:55], null offen
	buffer_load_b128 v[54:57] /*v[566:569]*/, v62 /*v574*/, s[52:55], null offen
	buffer_load_b128 v[66:69] /*v[578:581]*/, v75 /*v587*/, s[52:55], null offen
	buffer_load_b128 v[70:73] /*v[582:585]*/, v78 /*v590*/, s[52:55], null offen
	v_add_nc_u32_e32 v90 /*v602*/, 0xc0, v82 /*v594*/
	v_add_nc_u32_e32 v94 /*v606*/, 0xe0, v82 /*v594*/
	s_clause 0x1
	buffer_load_b128 v[26:29] /*v[538:541]*/, v82 /*v594*/, s[48:51], null offen offset:64
	buffer_load_b128 v[30:33] /*v[542:545]*/, v82 /*v594*/, s[48:51], null offen offset:96
	s_set_vgpr_msb 0x8a09
	v_cmp_gt_i32_e64 s17, v133 /*v389*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s18, v134 /*v390*/, v74 /*v586*/
	v_cmp_ge_i32_e64 s19, v144 /*v400*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s20, v144 /*v400*/, v74 /*v586*/
	s_set_vgpr_msb 0x908
	v_cmp_gt_i32_e64 s21, v1, v74 /*v586*/
	s_set_vgpr_msb 0x809
	v_cmp_gt_i32_e64 s22, v130 /*v386*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s23, v131 /*v387*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s24, v132 /*v388*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s25, v139 /*v395*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s26, v140 /*v396*/, v74 /*v586*/
	v_cmp_ge_i32_e64 s27, v150 /*v406*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s28, v150 /*v406*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s29, v135 /*v391*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s30, v136 /*v392*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s31, v137 /*v393*/, v74 /*v586*/
	v_cmp_gt_i32_e64 s33, v138 /*v394*/, v74 /*v586*/
	s_set_vgpr_msb 0x982
	s_clause 0x1
	buffer_load_b128 v[74:77] /*v[586:589]*/, v75 /*v587*/, s[48:51], null offen
	buffer_load_b128 v[78:81] /*v[590:593]*/, v78 /*v590*/, s[48:51], null offen
	buffer_load_b32 v104 /*v616*/, v83 /*v595*/, s[56:59], null offen
	s_clause 0x1
	buffer_load_b128 v[82:85] /*v[594:597]*/, v90 /*v602*/, s[52:55], null offen
	buffer_load_b128 v[86:89] /*v[598:601]*/, v94 /*v606*/, s[52:55], null offen
	s_clause 0x3
	buffer_load_b128 v[90:93] /*v[602:605]*/, v90 /*v602*/, s[48:51], null offen
	buffer_load_b128 v[58:61] /*v[570:573]*/, v58 /*v570*/, s[48:51], null offen
	buffer_load_b128 v[62:65] /*v[574:577]*/, v62 /*v574*/, s[48:51], null offen
	buffer_load_b128 v[94:97] /*v[606:609]*/, v94 /*v606*/, s[48:51], null offen
	s_and_b32 s62, s75, vcc_lo
	s_and_b32 s2, s75, s2
	s_and_b32 s4, s75, s4
	s_and_b32 s3, s75, s3
	s_and_b32 s6, s75, s6
	s_and_b32 s5, s75, s5
	s_and_b32 s8, s75, s8
	s_and_b32 s7, s75, s7
	s_and_b32 s9, s75, s9
	s_and_b32 s10, s75, s10
	s_and_b32 s12, s75, s12
	s_and_b32 s11, s75, s11
	s_and_b32 s14, s75, s14
	s_and_b32 s13, s75, s13
	s_and_b32 s16, s75, s16
	s_and_b32 s15, s75, s15
	s_and_b32 s17, s75, s17
	s_and_b32 s18, s75, s18
	s_and_b32 s19, s75, s19
	s_and_b32 s20, s75, s20
	s_and_b32 s21, s75, s21
	s_and_b32 s22, s75, s22
	s_and_b32 s23, s75, s23
	s_and_b32 s24, s75, s24
	s_and_b32 s25, s75, s25
	s_and_b32 s26, s75, s26
	s_and_b32 s27, s75, s27
	s_and_b32 s28, s75, s28
	s_and_b32 s29, s75, s29
	s_and_b32 s30, s75, s30
	s_and_b32 s31, s75, s31
	s_and_b32 s33, s75, s33
	s_add_nc_u64 s[64:65], s[64:65], 1
	s_set_vgpr_msb 0x8244
	s_wait_loadcnt 0x22
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[242:249], v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x4405
	ds_store_b128 v146 /*v402*/, v[162:165] /*v[418:421]*/ offset:13824
	ds_store_b128 v146 /*v402*/, v[166:169] /*v[422:425]*/ offset:13856
	s_wait_loadcnt 0x1f
	ds_store_b128 v146 /*v402*/, v[250:253] /*v[506:509]*/ offset:13888
	s_wait_loadcnt 0x1e
	ds_store_b128 v146 /*v402*/, v[254:257] /*v[510:513]*/ offset:13920
	ds_store_b128 v146 /*v402*/, v[178:181] /*v[434:437]*/ offset:5120
	ds_store_b128 v146 /*v402*/, v[182:185] /*v[438:441]*/ offset:5152
	s_set_vgpr_msb 0x509
	s_wait_loadcnt 0x1b
	ds_store_b128 v146 /*v402*/, v[10:13] /*v[522:525]*/ offset:5184
	s_wait_loadcnt 0x1a
	ds_store_b128 v146 /*v402*/, v[14:17] /*v[526:529]*/ offset:5216
	s_wait_loadcnt 0x17
	ds_store_b128 v146 /*v402*/, v[34:37] /*v[546:549]*/ offset:13952
	s_set_vgpr_msb 0x945
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[66:73] /*v[322:329]*/, v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x4509
	s_wait_loadcnt 0x16
	ds_store_b128 v146 /*v402*/, v[38:41] /*v[550:553]*/ offset:13984
	s_wait_loadcnt 0x10
	ds_store_b128 v146 /*v402*/, v[42:45] /*v[554:557]*/ offset:5248
	s_wait_loadcnt 0xf
	ds_store_b128 v146 /*v402*/, v[46:49] /*v[558:561]*/ offset:5280
	s_wait_loadcnt 0xc
	ds_store_b128 v146 /*v402*/, v[66:69] /*v[578:581]*/ offset:14016
	s_wait_loadcnt 0xb
	ds_store_b128 v146 /*v402*/, v[70:73] /*v[582:585]*/ offset:14048
	s_wait_loadcnt 0x8
	ds_store_b128 v146 /*v402*/, v[74:77] /*v[586:589]*/ offset:5312
	s_wait_loadcnt 0x7
	ds_store_b128 v146 /*v402*/, v[78:81] /*v[590:593]*/ offset:5344
	s_set_vgpr_msb 0x945
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[2:9] /*v[258:265]*/, v[178:185] /*v[434:441]*/, 0
	s_cmp_lg_u64 s[64:65], s[38:39]
	s_set_vgpr_msb 0x4544
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[242:249], v[202:209] /*v[458:465]*/, 0
	s_set_vgpr_msb 0x4445
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[66:73] /*v[322:329]*/, v[202:209] /*v[458:465]*/, 0
	s_set_vgpr_msb 0x4554
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[250:257], v[250:257] /*v[506:513]*/, v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5455
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[74:81] /*v[330:337]*/, v[250:257] /*v[506:513]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5559
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[10:17] /*v[266:273]*/, v[10:17] /*v[522:529]*/, v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5958
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[250:257], v[18:25] /*v[530:537]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[74:81] /*v[330:337]*/, v[18:25] /*v[530:537]*/, v[234:241] /*v[490:497]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[18:25] /*v[274:281]*/, v[34:41] /*v[546:553]*/, v[170:177] /*v[426:433]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[50:57] /*v[306:313]*/, v[34:41] /*v[546:553]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5945
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[90:97] /*v[346:353]*/, v[178:185] /*v[434:441]*/, 0
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[2:9] /*v[258:265]*/, v[226:233] /*v[482:489]*/, 0
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[26:33] /*v[282:289]*/, v[42:49] /*v[554:561]*/, v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[18:25] /*v[274:281]*/, v[50:57] /*v[562:569]*/, v[218:225] /*v[474:481]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[50:57] /*v[306:313]*/, v[50:57] /*v[562:569]*/, v[234:241] /*v[490:497]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[34:41] /*v[290:297]*/, v[66:73] /*v[578:585]*/, v[170:177] /*v[426:433]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5945
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[142:143] /*v[398:399]*/, v[170:171] /*v[426:427]*/
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[442:449]*/, v[106:113] /*v[362:369]*/, v[66:73] /*v[578:585]*/, v[186:193] /*v[442:449]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[142:143] /*v[398:399]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[142:143] /*v[398:399]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[142:143] /*v[398:399]*/, v[176:177] /*v[432:433]*/
	v_cndmask_b32_e64 v163 /*v419*/, v163 /*v419*/, 0xff61b1e6, s62
	v_cndmask_b32_e64 v162 /*v418*/, v162 /*v418*/, 0xff61b1e6, s2
	v_cndmask_b32_e64 v165 /*v421*/, v165 /*v421*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v164 /*v420*/, v164 /*v420*/, 0xff61b1e6, s4
	s_set_vgpr_msb 0x4585
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[90:97] /*v[346:353]*/, v[226:233] /*v[482:489]*/, 0
	s_set_vgpr_msb 0x8545
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[142:143] /*v[398:399]*/, v[186:187] /*v[442:443]*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[142:143] /*v[398:399]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[142:143] /*v[398:399]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[142:143] /*v[398:399]*/, v[192:193] /*v[448:449]*/
	v_cndmask_b32_e64 v167 /*v423*/, v167 /*v423*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v166 /*v422*/, v166 /*v422*/, 0xff61b1e6, s6
	v_cndmask_b32_e64 v169 /*v425*/, v169 /*v425*/, 0xff61b1e6, s7
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[98:105] /*v[354:361]*/, v[10:17] /*v[522:529]*/, v[210:217] /*v[466:473]*/
	v_cndmask_b32_e64 v168 /*v424*/, v168 /*v424*/, 0xff61b1e6, s8
	v_cndmask_b32_e64 v177 /*v433*/, v177 /*v433*/, 0xff61b1e6, s15
	v_cndmask_b32_e64 v176 /*v432*/, v176 /*v432*/, 0xff61b1e6, s16
	v_cndmask_b32_e64 v171 /*v427*/, v171 /*v427*/, 0xff61b1e6, s9
	v_cndmask_b32_e64 v170 /*v426*/, v170 /*v426*/, 0xff61b1e6, s10
	v_cndmask_b32_e64 v173 /*v429*/, v173 /*v429*/, 0xff61b1e6, s11
	v_cndmask_b32_e64 v172 /*v428*/, v172 /*v428*/, 0xff61b1e6, s12
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[10:17] /*v[266:273]*/, v[26:33] /*v[538:545]*/, v[242:249] /*v[498:505]*/
	v_cndmask_b32_e64 v175 /*v431*/, v175 /*v431*/, 0xff61b1e6, s13
	v_cndmask_b32_e64 v174 /*v430*/, v174 /*v430*/, 0xff61b1e6, s14
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[164:165] /*v[420:421]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[166:167] /*v[422:423]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[168:169] /*v[424:425]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[42:49] /*v[298:305]*/, v[74:81] /*v[586:593]*/, v[194:201] /*v[450:457]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[170:171] /*v[426:427]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[164:165] /*v[420:421]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[166:167] /*v[422:423]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[168:169] /*v[424:425]*/, s[42:43] op_sel_hi:[1,0]
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[34:41] /*v[290:297]*/, v[82:89] /*v[594:601]*/, v[218:225] /*v[474:481]*/
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[194:195] /*v[450:451]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[180:181] /*v[436:437]*/, v[196:197] /*v[452:453]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[198:199] /*v[454:455]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[184:185] /*v[440:441]*/, v[200:201] /*v[456:457]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[170:171] /*v[426:427]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, s[42:43] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[106:113] /*v[362:369]*/, v[82:89] /*v[594:601]*/, v[234:241] /*v[490:497]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[142:143] /*v[398:399]*/, v[218:219] /*v[474:475]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[142:143] /*v[398:399]*/, v[220:221] /*v[476:477]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[142:143] /*v[398:399]*/, v[222:223] /*v[478:479]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[142:143] /*v[398:399]*/, v[224:225] /*v[480:481]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, s[42:43] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v195 /*v451*/, v195 /*v451*/, 0xff61b1e6, s19
	v_cndmask_b32_e64 v194 /*v450*/, v194 /*v450*/, 0xff61b1e6, s20
	s_set_vgpr_msb 0x45a9
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[98:105] /*v[354:361]*/, v[26:33] /*v[538:545]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa945
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[142:143] /*v[398:399]*/, v[234:235] /*v[490:491]*/
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[142:143] /*v[398:399]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[142:143] /*v[398:399]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[142:143] /*v[398:399]*/, v[240:241] /*v[496:497]*/
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s17
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s18
	v_cndmask_b32_e64 v197 /*v453*/, v197 /*v453*/, 0xff61b1e6, s21
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[58:65] /*v[314:321]*/, v[42:49] /*v[554:561]*/, v[210:217] /*v[466:473]*/
	v_cndmask_b32_e64 v196 /*v452*/, v196 /*v452*/, 0xff61b1e6, s22
	v_cndmask_b32_e64 v199 /*v455*/, v199 /*v455*/, 0xff61b1e6, s23
	v_cndmask_b32_e64 v198 /*v454*/, v198 /*v454*/, 0xff61b1e6, s24
	v_cndmask_b32_e64 v241 /*v497*/, v241 /*v497*/, 0xff61b1e6, s25
	v_cndmask_b32_e64 v240 /*v496*/, v240 /*v496*/, 0xff61b1e6, s26
	v_cndmask_b32_e64 v235 /*v491*/, v235 /*v491*/, 0xff61b1e6, s27
	v_cndmask_b32_e64 v234 /*v490*/, v234 /*v490*/, 0xff61b1e6, s28
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[26:33] /*v[282:289]*/, v[58:65] /*v[570:577]*/, v[242:249] /*v[498:505]*/
	v_cndmask_b32_e64 v237 /*v493*/, v237 /*v493*/, 0xff61b1e6, s29
	v_cndmask_b32_e64 v236 /*v492*/, v236 /*v492*/, 0xff61b1e6, s30
	v_cndmask_b32_e64 v239 /*v495*/, v239 /*v495*/, 0xff61b1e6, s31
	v_cndmask_b32_e64 v238 /*v494*/, v238 /*v494*/, 0xff61b1e6, s33
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[58:65] /*v[314:321]*/, v[58:65] /*v[570:577]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa959
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[240:241] /*v[496:497]*/, v[240:241] /*v[496:497]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[234:235] /*v[490:491]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[236:237] /*v[492:493]*/, v[236:237] /*v[492:493]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, v[104:105] /*v[616:617]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, s[42:43] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[114:121] /*v[370:377]*/, v[74:81] /*v[586:593]*/, v[210:217] /*v[466:473]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[234:235] /*v[490:491]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[236:237] /*v[492:493]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[240:241] /*v[496:497]*/, s[42:43] op_sel_hi:[1,0]
	v_exp_f32_e32 v170 /*v426*/, v170 /*v426*/
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[42:49] /*v[298:305]*/, v[90:97] /*v[602:609]*/, v[242:249] /*v[498:505]*/
	v_pk_add_f32 v[186:187] /*v[442:443]*/, v[210:211] /*v[466:467]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[212:213] /*v[468:469]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[190:191] /*v[446:447]*/, v[214:215] /*v[470:471]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[216:217] /*v[472:473]*/, v[100:101] /*v[612:613]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v171 /*v427*/, v171 /*v427*/
	v_exp_f32_e32 v172 /*v428*/, v172 /*v428*/
	v_exp_f32_e32 v173 /*v429*/, v173 /*v429*/
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[114:121] /*v[370:377]*/, v[90:97] /*v[602:609]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa949
	v_pk_add_f32 v[210:211] /*v[466:467]*/, v[242:243] /*v[498:499]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[212:213] /*v[468:469]*/, v[244:245] /*v[500:501]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[246:247] /*v[502:503]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[248:249] /*v[504:505]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v242 /*v498*/, v162 /*v418*/
	v_exp_f32_e32 v243 /*v499*/, v163 /*v419*/
	v_exp_f32_e32 v244 /*v500*/, v164 /*v420*/
	v_exp_f32_e32 v245 /*v501*/, v165 /*v421*/
	v_exp_f32_e32 v246 /*v502*/, v166 /*v422*/
	v_exp_f32_e32 v247 /*v503*/, v167 /*v423*/
	v_exp_f32_e32 v248 /*v504*/, v168 /*v424*/
	v_exp_f32_e32 v249 /*v505*/, v169 /*v425*/
	v_exp_f32_e32 v174 /*v430*/, v174 /*v430*/
	v_exp_f32_e32 v175 /*v431*/, v175 /*v431*/
	v_exp_f32_e32 v176 /*v432*/, v176 /*v432*/
	v_exp_f32_e32 v177 /*v433*/, v177 /*v433*/
	v_exp_f32_e32 v194 /*v450*/, v194 /*v450*/
	v_exp_f32_e32 v195 /*v451*/, v195 /*v451*/
	v_exp_f32_e32 v196 /*v452*/, v196 /*v452*/
	v_exp_f32_e32 v197 /*v453*/, v197 /*v453*/
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	v_exp_f32_e32 v199 /*v455*/, v199 /*v455*/
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	v_exp_f32_e32 v234 /*v490*/, v234 /*v490*/
	v_exp_f32_e32 v235 /*v491*/, v235 /*v491*/
	v_exp_f32_e32 v236 /*v492*/, v236 /*v492*/
	v_exp_f32_e32 v237 /*v493*/, v237 /*v493*/
	v_exp_f32_e32 v238 /*v494*/, v238 /*v494*/
	v_exp_f32_e32 v239 /*v495*/, v239 /*v495*/
	v_exp_f32_e32 v240 /*v496*/, v240 /*v496*/
	v_exp_f32_e32 v241 /*v497*/, v241 /*v497*/
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[2:3] /*v[514:515]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[220:221] /*v[476:477]*/, v[4:5] /*v[516:517]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[6:7] /*v[518:519]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[8:9] /*v[520:521]*/, v[102:103] /*v[614:615]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v165 /*v421*/, v248 /*v504*/, v249 /*v505*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v246 /*v502*/, v247 /*v503*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v244 /*v500*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v242 /*v498*/, v243 /*v499*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[242:243] /*v[498:499]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, v[244:245] /*v[500:501]*/
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, v[246:247] /*v[502:503]*/
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[184:185] /*v[440:441]*/, v[248:249] /*v[504:505]*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v172 /*v428*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v176 /*v432*/, v177 /*v433*/
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, v[170:171] /*v[426:427]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[210:211] /*v[466:467]*/, v[194:195] /*v[450:451]*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v196 /*v452*/, v197 /*v453*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[212:213] /*v[468:469]*/, v[196:197] /*v[452:453]*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v198 /*v454*/, v199 /*v455*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[214:215] /*v[470:471]*/, v[198:199] /*v[454:455]*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v200 /*v456*/, v201 /*v457*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[216:217] /*v[472:473]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[218:219] /*v[474:475]*/, v[234:235] /*v[490:491]*/
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[220:221] /*v[476:477]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[222:223] /*v[478:479]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[224:225] /*v[480:481]*/, v[240:241] /*v[496:497]*/
	ds_store_b128 v151 /*v407*/, v[162:165] /*v[418:421]*/
	ds_store_b128 v151 /*v407*/, v[166:169] /*v[422:425]*/ offset:32
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[142:143] /*v[398:399]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[142:143] /*v[398:399]*/, v[180:181] /*v[436:437]*/
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[142:143] /*v[398:399]*/, v[182:183] /*v[438:439]*/
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[142:143] /*v[398:399]*/, v[184:185] /*v[440:441]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[142:143] /*v[398:399]*/, v[186:187] /*v[442:443]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[142:143] /*v[398:399]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[142:143] /*v[398:399]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[142:143] /*v[398:399]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[142:143] /*v[398:399]*/, v[210:211] /*v[466:467]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[142:143] /*v[398:399]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[142:143] /*v[398:399]*/, v[196:197] /*v[452:453]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[142:143] /*v[398:399]*/, v[198:199] /*v[454:455]*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[142:143] /*v[398:399]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[142:143] /*v[398:399]*/, v[212:213] /*v[468:469]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[142:143] /*v[398:399]*/, v[214:215] /*v[470:471]*/
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[216:217] /*v[472:473]*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v164 /*v420*/, v165 /*v421*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v166 /*v422*/, v167 /*v423*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v168 /*v424*/, v169 /*v425*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v182 /*v438*/, v183 /*v439*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v184 /*v440*/, v185 /*v441*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v175 /*v431*/, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v238 /*v494*/, v239 /*v495*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v240 /*v496*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v178 /*v434*/, v186 /*v442*/, v187 /*v443*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v188 /*v444*/, v189 /*v445*/
	v_cvt_pk_bf16_f32 v180 /*v436*/, v190 /*v446*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v192 /*v448*/, v193 /*v449*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v183 /*v439*/, v196 /*v452*/, v197 /*v453*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v198 /*v454*/, v199 /*v455*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v142 /*v398*/, v143 /*v399*/
	ds_store_b128 v151 /*v407*/, v[162:165] /*v[418:421]*/ offset:2560
	ds_store_b128 v151 /*v407*/, v[166:169] /*v[422:425]*/ offset:2592
	ds_store_b128 v146 /*v402*/, v[226:229] /*v[482:485]*/ offset:9472
	ds_store_b128 v146 /*v402*/, v[230:233] /*v[486:489]*/ offset:9504
	ds_store_b128 v146 /*v402*/, v[202:205] /*v[458:461]*/ offset:18176
	ds_store_b128 v146 /*v402*/, v[206:209] /*v[462:465]*/ offset:18208
	s_set_vgpr_msb 0x4509
	ds_store_b128 v146 /*v402*/, v[26:29] /*v[538:541]*/ offset:9536
	ds_store_b128 v146 /*v402*/, v[30:33] /*v[542:545]*/ offset:9568
	ds_store_b128 v146 /*v402*/, v[18:21] /*v[530:533]*/ offset:18240
	ds_store_b128 v146 /*v402*/, v[22:25] /*v[534:537]*/ offset:18272
	ds_store_b128 v146 /*v402*/, v[58:61] /*v[570:573]*/ offset:9600
	ds_store_b128 v146 /*v402*/, v[62:65] /*v[574:577]*/ offset:9632
	ds_store_b128 v146 /*v402*/, v[50:53] /*v[562:565]*/ offset:18304
	ds_store_b128 v146 /*v402*/, v[54:57] /*v[566:569]*/ offset:18336
	ds_store_b128 v146 /*v402*/, v[90:93] /*v[602:605]*/ offset:9664
	ds_store_b128 v146 /*v402*/, v[94:97] /*v[606:609]*/ offset:9696
	ds_store_b128 v146 /*v402*/, v[82:85] /*v[594:597]*/ offset:18368
	ds_store_b128 v146 /*v402*/, v[86:89] /*v[598:601]*/ offset:18400
	s_set_vgpr_msb 0x905
	ds_store_b128 v151 /*v407*/, v[170:173] /*v[426:429]*/ offset:1280
	ds_store_b128 v151 /*v407*/, v[174:177] /*v[430:433]*/ offset:1312
	ds_store_b128 v151 /*v407*/, v[178:181] /*v[434:437]*/ offset:3840
	ds_store_b128 v151 /*v407*/, v[182:185] /*v[438:441]*/ offset:3872
	s_set_vgpr_msb 0x542
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v99 /*v611*/
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v99 /*v611*/ offset:4352
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v112 /*v624*/
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v112 /*v624*/ offset:1280
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v103 /*v615*/
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v103 /*v615*/ offset:4352
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v106 /*v618*/
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v106 /*v618*/ offset:4352
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v107 /*v619*/
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v107 /*v619*/ offset:4352
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v108 /*v620*/
	ds_load_tr16_b128 v[206:209] /*v[462:465]*/, v108 /*v620*/ offset:4352
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v109 /*v621*/
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v109 /*v621*/ offset:4352
	ds_load_tr16_b128 v[218:221] /*v[474:477]*/, v110 /*v622*/
	ds_load_tr16_b128 v[222:225] /*v[478:481]*/, v110 /*v622*/ offset:4352
	ds_load_tr16_b128 v[226:229] /*v[482:485]*/, v111 /*v623*/
	ds_load_tr16_b128 v[230:233] /*v[486:489]*/, v111 /*v623*/ offset:4352
	ds_load_tr16_b128 v[234:237] /*v[490:493]*/, v113 /*v625*/
	ds_load_tr16_b128 v[238:241] /*v[494:497]*/, v113 /*v625*/ offset:4352
	ds_load_tr16_b128 v[242:245] /*v[498:501]*/, v117 /*v629*/
	s_set_vgpr_msb 0x4205
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[234:241], v[170:177] /*v[426:433]*/, v[162:169] /*v[418:425]*/, v[234:241]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[254:257] /*v[510:513]*/, v101 /*v613*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v114 /*v626*/
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v114 /*v626*/ offset:4352
	ds_load_tr16_b128 v[10:13] /*v[522:525]*/, v115 /*v627*/
	ds_load_tr16_b128 v[14:17] /*v[526:529]*/, v115 /*v627*/ offset:4352
	ds_load_tr16_b128 v[18:21] /*v[530:533]*/, v116 /*v628*/
	ds_load_tr16_b128 v[22:25] /*v[534:537]*/, v116 /*v628*/ offset:4352
	ds_load_tr16_b128 v[26:29] /*v[538:541]*/, v118 /*v630*/
	ds_load_tr16_b128 v[30:33] /*v[542:545]*/, v118 /*v630*/ offset:4352
	ds_load_tr16_b128 v[34:37] /*v[546:549]*/, v119 /*v631*/
	ds_load_tr16_b128 v[38:41] /*v[550:553]*/, v119 /*v631*/ offset:4352
	ds_load_tr16_b128 v[42:45] /*v[554:557]*/, v120 /*v632*/
	s_set_vgpr_msb 0x8205
	s_wait_dscnt 0x1b
	v_wmma_f32_16x16x32_bf16 v[218:225], v[170:177] /*v[426:433]*/, v[178:185] /*v[434:441]*/, v[218:225]
	s_wait_dscnt 0x19
	v_wmma_f32_16x16x32_bf16 v[202:209], v[170:177] /*v[426:433]*/, v[186:193] /*v[442:449]*/, v[202:209]
	s_wait_dscnt 0x17
	v_wmma_f32_16x16x32_bf16 v[186:193], v[170:177] /*v[426:433]*/, v[194:201] /*v[450:457]*/, v[186:193]
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[170:177], v[170:177] /*v[426:433]*/, v[202:209] /*v[458:465]*/, v[170:177]
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[154:161], v[170:177] /*v[426:433]*/, v[210:217] /*v[466:473]*/, v[154:161]
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[138:145], v[170:177] /*v[426:433]*/, v[218:225] /*v[474:481]*/, v[138:145]
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[130:137], v[170:177] /*v[426:433]*/, v[226:233] /*v[482:489]*/, v[130:137]
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[246:249] /*v[502:505]*/, v117 /*v629*/ offset:1280
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v105 /*v617*/
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v105 /*v617*/ offset:4352
	ds_load_tr16_b128 v[250:253] /*v[506:509]*/, v101 /*v613*/
	s_set_vgpr_msb 0x4205
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[226:233], v[242:249] /*v[498:505]*/, v[234:241] /*v[490:497]*/, v[226:233]
	s_set_vgpr_msb 0x555
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[242:249] /*v[498:505]*/, v[170:177] /*v[426:433]*/, v[82:89] /*v[338:345]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[242:249] /*v[498:505]*/, v[250:257] /*v[506:513]*/, v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[210:217], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[210:217]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[242:249] /*v[498:505]*/, v[10:17] /*v[522:529]*/, v[194:201]
	v_wmma_f32_16x16x32_bf16 v[178:185], v[242:249] /*v[498:505]*/, v[18:25] /*v[530:537]*/, v[178:185]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[242:249] /*v[498:505]*/, v[26:33] /*v[538:545]*/, v[162:169]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[242:249] /*v[498:505]*/, v[34:41] /*v[546:553]*/, v[146:153]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v120 /*v632*/ offset:1280
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[242:245] /*v[498:501]*/, v121 /*v633*/
	ds_load_tr16_b128 v[246:249] /*v[502:505]*/, v121 /*v633*/ offset:1280
	s_set_vgpr_msb 0x4206
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_wmma_f32_16x16x32_bf16 v[106:113], v[42:49] /*v[554:561]*/, v[162:169] /*v[418:425]*/, v[106:113]
	s_set_vgpr_msb 0x605
	s_barrier_wait -1
	v_wmma_f32_16x16x32_bf16 v[122:129], v[242:249] /*v[498:505]*/, v[250:257] /*v[506:513]*/, v[122:129]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[90:97], v[42:49] /*v[554:561]*/, v[178:185] /*v[434:441]*/, v[90:97]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[114:121], v[242:249] /*v[498:505]*/, v[170:177] /*v[426:433]*/, v[114:121]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[74:81], v[42:49] /*v[554:561]*/, v[186:193] /*v[442:449]*/, v[74:81]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[98:105], v[242:249] /*v[498:505]*/, v[234:241] /*v[490:497]*/, v[98:105]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[58:65], v[42:49] /*v[554:561]*/, v[194:201] /*v[450:457]*/, v[58:65]
	s_set_vgpr_msb 0x609
	v_wmma_f32_16x16x32_bf16 v[82:89], v[242:249] /*v[498:505]*/, v[2:9] /*v[514:521]*/, v[82:89]
	s_set_vgpr_msb 0x906
	v_wmma_f32_16x16x32_bf16 v[42:49], v[42:49] /*v[554:561]*/, v[202:209] /*v[458:465]*/, v[42:49]
	s_set_vgpr_msb 0x609
	v_wmma_f32_16x16x32_bf16 v[66:73], v[242:249] /*v[498:505]*/, v[10:17] /*v[522:529]*/, v[66:73]
	s_set_vgpr_msb 0x906
	v_wmma_f32_16x16x32_bf16 v[26:33], v[42:49] /*v[554:561]*/, v[210:217] /*v[466:473]*/, v[26:33]
	s_set_vgpr_msb 0x609
	v_wmma_f32_16x16x32_bf16 v[50:57], v[242:249] /*v[498:505]*/, v[18:25] /*v[530:537]*/, v[50:57]
	s_set_vgpr_msb 0x906
	v_wmma_f32_16x16x32_bf16 v[10:17], v[42:49] /*v[554:561]*/, v[218:225] /*v[474:481]*/, v[10:17]
	s_set_vgpr_msb 0x609
	v_wmma_f32_16x16x32_bf16 v[34:41], v[242:249] /*v[498:505]*/, v[26:33] /*v[538:545]*/, v[34:41]
	s_set_vgpr_msb 0x906
	v_wmma_f32_16x16x32_bf16 v[2:9], v[42:49] /*v[554:561]*/, v[226:233] /*v[482:489]*/, v[2:9]
	s_set_vgpr_msb 0x609
	v_wmma_f32_16x16x32_bf16 v[18:25], v[242:249] /*v[498:505]*/, v[34:41] /*v[546:553]*/, v[18:25]
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
	v_mov_b64_e32 v[88:89], v[24:25]
	v_mov_b64_e32 v[86:87], v[22:23]
	v_mov_b64_e32 v[84:85], v[20:21]
	v_mov_b64_e32 v[82:83], v[18:19]
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
	v_mov_b64_e32 v[216:217], v[24:25]
	v_mov_b64_e32 v[214:215], v[22:23]
	v_mov_b64_e32 v[212:213], v[20:21]
	v_mov_b64_e32 v[210:211], v[18:19]
	v_mov_b64_e32 v[232:233], v[24:25]
	v_mov_b64_e32 v[230:231], v[22:23]
	v_mov_b64_e32 v[228:229], v[20:21]
	v_mov_b64_e32 v[226:227], v[18:19]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[24:25]
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[22:23]
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[20:21]
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[18:19]
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
	v_mov_b64_e32 v[80:81], v[24:25]
	v_mov_b64_e32 v[78:79], v[22:23]
	v_mov_b64_e32 v[76:77], v[20:21]
	v_mov_b64_e32 v[74:75], v[18:19]
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
	v_mov_b64_e32 v[208:209], v[24:25]
	v_mov_b64_e32 v[206:207], v[22:23]
	v_mov_b64_e32 v[204:205], v[20:21]
	v_mov_b64_e32 v[202:203], v[18:19]
	v_mov_b64_e32 v[224:225], v[24:25]
	v_mov_b64_e32 v[222:223], v[22:23]
	v_mov_b64_e32 v[220:221], v[20:21]
	v_mov_b64_e32 v[218:219], v[18:19]
	v_mov_b64_e32 v[240:241], v[24:25]
	v_mov_b64_e32 v[238:239], v[22:23]
	v_mov_b64_e32 v[236:237], v[20:21]
	v_mov_b64_e32 v[234:235], v[18:19]
.LBB0_4:
	s_sub_co_i32 s2, s73, s74
	s_set_vgpr_msb 64
	v_lshrrev_b32_e32 v130 /*v386*/, 3, v0
	s_mul_i32 s2, s2, s41
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s2, 1
	s_set_vgpr_msb 0x4000
	s_cbranch_scc1 .LBB0_7
	s_abs_i32 s8, s41
	s_movk_i32 s5, 0x1400
	s_cvt_f32_u32 s4, s8
	s_movk_i32 s6, 0x3600
	s_set_vgpr_msb 0x45
	v_mad_u32_u24 v152 /*v408*/, 0x110, v149 /*v405*/, s5
	s_movk_i32 s5, 0xa00
	v_s_rcp_f32 s4, s4
	v_mad_u32_u24 v153 /*v409*/, 0x110, v149 /*v405*/, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(TRANS32_DEP_1)
	v_dual_add_nc_u32 v132 /*v388*/, v152 /*v408*/, v147 /*v403*/ :: v_dual_bitop2_b32 v155 /*v411*/, 32, v147 /*v403*/ bitop3:0x54
	v_or_b32_e32 v137 /*v393*/, 64, v147 /*v403*/
	v_or_b32_e32 v139 /*v395*/, 0x60, v147 /*v403*/
	v_or_b32_e32 v142 /*v398*/, 0x80, v147 /*v403*/
	v_or_b32_e32 v150 /*v406*/, 0xa0, v147 /*v403*/
	s_mul_f32 s4, s4, 0x4f7ffffe
	v_or_b32_e32 v151 /*v407*/, 0xc0, v147 /*v403*/
	v_lshl_or_b32 v154 /*v410*/, v130 /*v386*/, 4, 0xe0
	v_mad_u32_u24 v156 /*v412*/, 0x50, v149 /*v405*/, s5
	s_cvt_u32_f32 s4, s4
	s_sub_co_i32 s5, 0, s8
	s_mov_b32 s12, s36
	s_mov_b32 s13, s36
	s_mul_i32 s5, s5, s4
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, s[12:13]
	s_set_vgpr_msb 0x4514
	v_mad_i32_i24 v1, 0xffffff40, v141 /*v397*/, v146 /*v402*/
	s_set_vgpr_msb 0x1445
	v_add_nc_u32_e32 v133 /*v389*/, v153 /*v409*/, v147 /*v403*/
	v_dual_add_nc_u32 v134 /*v390*/, v152 /*v408*/, v155 /*v411*/ :: v_dual_add_nc_u32 v135 /*v391*/, v153 /*v409*/, v155 /*v411*/
	v_dual_add_nc_u32 v136 /*v392*/, v152 /*v408*/, v137 /*v393*/ :: v_dual_add_nc_u32 v137 /*v393*/, v153 /*v409*/, v137 /*v393*/
	v_dual_add_nc_u32 v138 /*v394*/, v152 /*v408*/, v139 /*v395*/ :: v_dual_add_nc_u32 v139 /*v395*/, v153 /*v409*/, v139 /*v395*/
	v_dual_add_nc_u32 v140 /*v396*/, v152 /*v408*/, v142 /*v398*/ :: v_dual_add_nc_u32 v142 /*v398*/, v153 /*v409*/, v142 /*v398*/
	v_dual_add_nc_u32 v143 /*v399*/, v152 /*v408*/, v150 /*v406*/ :: v_dual_add_nc_u32 v149 /*v405*/, v153 /*v409*/, v150 /*v406*/
	v_dual_add_nc_u32 v150 /*v406*/, v152 /*v408*/, v151 /*v407*/ :: v_dual_add_nc_u32 v151 /*v407*/, v153 /*v409*/, v151 /*v407*/
	v_dual_add_nc_u32 v152 /*v408*/, v152 /*v408*/, v154 /*v410*/ :: v_dual_add_nc_u32 v153 /*v409*/, v153 /*v409*/, v154 /*v410*/
	v_add_nc_u32_e32 v154 /*v410*/, v148 /*v404*/, v147 /*v403*/
	v_add_nc_u32_e32 v147 /*v403*/, v156 /*v412*/, v147 /*v403*/
	v_add_nc_u32_e32 v148 /*v404*/, v148 /*v404*/, v155 /*v411*/
	v_add_nc_u32_e32 v155 /*v411*/, v156 /*v412*/, v155 /*v411*/
	s_mul_hi_u32 s5, s4, s5
	s_add_co_i32 s7, s74, s72
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
	s_set_vgpr_msb 0x98
	v_lshl_or_b32 v76 /*v588*/, s11, 5, v141 /*v397*/
	s_add_co_i32 s11, s12, s69
	s_add_nc_u64 s[4:5], s[4:5], 1
	s_lshl4_add_u32 s12, s11, s71
	s_add_co_i32 s11, s11, s70
	v_or_b32_e32 v77 /*v589*/, 16, v76 /*v588*/
	s_set_vgpr_msb 0x9842
	v_mad_u32 v156 /*v412*/, v76 /*v588*/, s35, s12
	s_mul_i32 s11, s11, s37
	s_cmp_lg_u64 s[4:5], s[2:3]
	s_set_vgpr_msb 0x4288
	v_add_lshl_u32 v76 /*v588*/, s11, v76 /*v588*/, 2
	s_set_vgpr_msb 0x8842
	v_mad_u32 v196 /*v452*/, v77 /*v589*/, s35, s12
	s_set_vgpr_msb 0x428a
	v_add_lshl_u32 v77 /*v589*/, s11, v77 /*v589*/, 2
	buffer_load_b32 v92 /*v604*/, v76 /*v588*/, s[60:63], null offen
	s_clause 0x1
	buffer_load_b32 v94 /*v606*/, v76 /*v588*/, s[56:59], null offen
	buffer_load_b32 v96 /*v608*/, v77 /*v589*/, s[56:59], null offen
	s_set_vgpr_msb 0x8a45
	v_or_b32_e32 v156 /*v412*/, v156 /*v412*/, v145 /*v401*/
	s_set_vgpr_msb 0x4582
	buffer_load_b32 v98 /*v610*/, v77 /*v589*/, s[60:63], null offen
	s_set_vgpr_msb 0x8245
	v_or_b32_e32 v196 /*v452*/, v196 /*v452*/, v145 /*v401*/
	s_set_vgpr_msb 0x4584
	v_lshlrev_b32_e32 v60 /*v572*/, 4, v156 /*v412*/
	s_set_vgpr_msb 0x8442
	s_clause 0x1
	buffer_load_b128 v[156:159] /*v[412:415]*/, v60 /*v572*/, s[52:55], null offen
	buffer_load_b128 v[160:163] /*v[416:419]*/, v60 /*v572*/, s[52:55], null offen offset:32
	buffer_load_b128 v[164:167] /*v[420:423]*/, v60 /*v572*/, s[48:51], null offen
	s_set_vgpr_msb 0x4284
	v_lshlrev_b32_e32 v78 /*v590*/, 4, v196 /*v452*/
	s_set_vgpr_msb 0x8442
	buffer_load_b128 v[200:203] /*v[456:459]*/, v78 /*v590*/, s[52:55], null offen offset:32
	buffer_load_b128 v[212:215] /*v[468:471]*/, v78 /*v590*/, s[48:51], null offen
	s_clause 0x2
	buffer_load_b128 v[196:199] /*v[452:455]*/, v78 /*v590*/, s[52:55], null offen
	buffer_load_b128 v[244:247] /*v[500:503]*/, v60 /*v572*/, s[52:55], null offen offset:64
	buffer_load_b128 v[248:251] /*v[504:507]*/, v60 /*v572*/, s[52:55], null offen offset:96
	s_set_vgpr_msb 0x428a
	buffer_load_b128 v[4:7] /*v[516:519]*/, v60 /*v572*/, s[48:51], null offen offset:64
	v_or_b32_e32 v36 /*v548*/, 0x80, v60 /*v572*/
	v_add_nc_u32_e32 v40 /*v552*/, 0xa0, v60 /*v572*/
	v_or_b32_e32 v52 /*v564*/, 0x80, v78 /*v590*/
	s_clause 0x1
	buffer_load_b128 v[12:15] /*v[524:527]*/, v78 /*v590*/, s[52:55], null offen offset:64
	buffer_load_b128 v[16:19] /*v[528:531]*/, v78 /*v590*/, s[52:55], null offen offset:96
	buffer_load_b128 v[20:23] /*v[532:535]*/, v78 /*v590*/, s[48:51], null offen offset:64
	buffer_load_b128 v[28:31] /*v[540:543]*/, v36 /*v548*/, s[52:55], null offen
	v_add_nc_u32_e32 v56 /*v568*/, 0xa0, v78 /*v590*/
	s_clause 0x1
	buffer_load_b128 v[44:47] /*v[556:559]*/, v52 /*v564*/, s[52:55], null offen
	buffer_load_b128 v[32:35] /*v[544:547]*/, v40 /*v552*/, s[52:55], null offen
	s_clause 0x4
	buffer_load_b128 v[36:39] /*v[548:551]*/, v36 /*v548*/, s[48:51], null offen
	s_set_vgpr_msb 0x8a42
	buffer_load_b128 v[168:171] /*v[424:427]*/, v60 /*v572*/, s[48:51], null offen offset:32
	s_set_vgpr_msb 0x428a
	buffer_load_b128 v[8:11] /*v[520:523]*/, v60 /*v572*/, s[48:51], null offen offset:96
	v_add_nc_u32_e32 v68 /*v580*/, 0xc0, v60 /*v572*/
	v_add_nc_u32_e32 v72 /*v584*/, 0xe0, v60 /*v572*/
	buffer_load_b128 v[48:51] /*v[560:563]*/, v56 /*v568*/, s[52:55], null offen
	s_clause 0x2
	buffer_load_b128 v[52:55] /*v[564:567]*/, v52 /*v564*/, s[48:51], null offen
	s_set_vgpr_msb 0x8a42
	buffer_load_b128 v[216:219] /*v[472:475]*/, v78 /*v590*/, s[48:51], null offen offset:32
	s_set_vgpr_msb 0x428a
	v_add_nc_u32_e32 v84 /*v596*/, 0xc0, v78 /*v590*/
	s_clause 0x1
	buffer_load_b128 v[60:63] /*v[572:575]*/, v68 /*v580*/, s[52:55], null offen
	buffer_load_b128 v[64:67] /*v[576:579]*/, v72 /*v584*/, s[52:55], null offen
	buffer_load_b128 v[68:71] /*v[580:583]*/, v68 /*v580*/, s[48:51], null offen
	v_add_nc_u32_e32 v88 /*v600*/, 0xe0, v78 /*v590*/
	s_clause 0x1
	buffer_load_b128 v[24:27] /*v[536:539]*/, v78 /*v590*/, s[48:51], null offen offset:96
	buffer_load_b128 v[72:75] /*v[584:587]*/, v72 /*v584*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[76:79] /*v[588:591]*/, v84 /*v596*/, s[52:55], null offen
	buffer_load_b128 v[80:83] /*v[592:595]*/, v88 /*v600*/, s[52:55], null offen
	s_clause 0x3
	buffer_load_b128 v[84:87] /*v[596:599]*/, v84 /*v596*/, s[48:51], null offen
	buffer_load_b128 v[40:43] /*v[552:555]*/, v40 /*v552*/, s[48:51], null offen
	buffer_load_b128 v[56:59] /*v[568:571]*/, v56 /*v568*/, s[48:51], null offen
	buffer_load_b128 v[88:91] /*v[600:603]*/, v88 /*v600*/, s[48:51], null offen
	s_set_vgpr_msb 0x8a44
	s_wait_loadcnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[428:435]*/, v[242:249], v[156:163] /*v[412:419]*/, 0
	s_set_vgpr_msb 0x4445
	s_wait_loadcnt 0x1d
	ds_store_b128 v146 /*v402*/, v[164:167] /*v[420:423]*/ offset:5120
	s_wait_loadcnt 0xf
	ds_store_b128 v146 /*v402*/, v[168:171] /*v[424:427]*/ offset:5152
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[444:451]*/, v[66:73] /*v[322:329]*/, v[156:163] /*v[412:419]*/, 0
	ds_store_b128 v146 /*v402*/, v[156:159] /*v[412:415]*/ offset:13824
	ds_store_b128 v146 /*v402*/, v[160:163] /*v[416:419]*/ offset:13856
	s_set_vgpr_msb 0x4509
	ds_store_b128 v146 /*v402*/, v[4:7] /*v[516:519]*/ offset:5184
	s_wait_loadcnt 0xe
	ds_store_b128 v146 /*v402*/, v[8:11] /*v[520:523]*/ offset:5216
	s_set_vgpr_msb 0x905
	ds_store_b128 v146 /*v402*/, v[244:247] /*v[500:503]*/ offset:13888
	ds_store_b128 v146 /*v402*/, v[248:251] /*v[504:507]*/ offset:13920
	s_set_vgpr_msb 0x509
	ds_store_b128 v146 /*v402*/, v[36:39] /*v[548:551]*/ offset:5248
	s_wait_loadcnt 0x2
	ds_store_b128 v146 /*v402*/, v[40:43] /*v[552:555]*/ offset:5280
	ds_store_b128 v146 /*v402*/, v[28:31] /*v[540:543]*/ offset:13952
	ds_store_b128 v146 /*v402*/, v[32:35] /*v[544:547]*/ offset:13984
	ds_store_b128 v146 /*v402*/, v[68:71] /*v[580:583]*/ offset:5312
	ds_store_b128 v146 /*v402*/, v[72:75] /*v[584:587]*/ offset:5344
	ds_store_b128 v146 /*v402*/, v[60:63] /*v[572:575]*/ offset:14016
	ds_store_b128 v146 /*v402*/, v[64:67] /*v[576:579]*/ offset:14048
	s_set_vgpr_msb 0x944
	v_wmma_f32_16x16x32_bf16 v[220:227] /*v[476:483]*/, v[242:249], v[196:203] /*v[452:459]*/, 0
	s_set_vgpr_msb 0x4445
	v_wmma_f32_16x16x32_bf16 v[236:243] /*v[492:499]*/, v[66:73] /*v[322:329]*/, v[196:203] /*v[452:459]*/, 0
	s_set_vgpr_msb 0x4554
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[428:435]*/, v[250:257], v[244:251] /*v[500:507]*/, v[172:179] /*v[428:435]*/
	s_set_vgpr_msb 0x5455
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[444:451]*/, v[74:81] /*v[330:337]*/, v[244:251] /*v[500:507]*/, v[188:195] /*v[444:451]*/
	s_set_vgpr_msb 0x5558
	v_wmma_f32_16x16x32_bf16 v[220:227] /*v[476:483]*/, v[250:257], v[12:19] /*v[524:531]*/, v[220:227] /*v[476:483]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[236:243] /*v[492:499]*/, v[74:81] /*v[330:337]*/, v[12:19] /*v[524:531]*/, v[236:243] /*v[492:499]*/
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[428:435]*/, v[18:25] /*v[274:281]*/, v[28:35] /*v[540:547]*/, v[172:179] /*v[428:435]*/
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[444:451]*/, v[50:57] /*v[306:313]*/, v[28:35] /*v[540:547]*/, v[188:195] /*v[444:451]*/
	s_set_vgpr_msb 0x5945
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[2:9] /*v[258:265]*/, v[164:171] /*v[420:427]*/, 0
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[90:97] /*v[346:353]*/, v[164:171] /*v[420:427]*/, 0
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[220:227] /*v[476:483]*/, v[18:25] /*v[274:281]*/, v[44:51] /*v[556:563]*/, v[220:227] /*v[476:483]*/
	v_wmma_f32_16x16x32_bf16 v[236:243] /*v[492:499]*/, v[50:57] /*v[306:313]*/, v[44:51] /*v[556:563]*/, v[236:243] /*v[492:499]*/
	s_set_vgpr_msb 0x5945
	v_wmma_f32_16x16x32_bf16 v[228:235] /*v[484:491]*/, v[2:9] /*v[258:265]*/, v[212:219] /*v[468:475]*/, 0
	v_wmma_f32_16x16x32_bf16 v[252:259] /*v[508:515]*/, v[90:97] /*v[346:353]*/, v[212:219] /*v[468:475]*/, 0
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[428:435]*/, v[34:41] /*v[290:297]*/, v[60:67] /*v[572:579]*/, v[172:179] /*v[428:435]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5945
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[130:131] /*v[386:387]*/, v[172:173] /*v[428:429]*/
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[444:451]*/, v[106:113] /*v[362:369]*/, v[60:67] /*v[572:579]*/, v[188:195] /*v[444:451]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[130:131] /*v[386:387]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[130:131] /*v[386:387]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[130:131] /*v[386:387]*/, v[178:179] /*v[434:435]*/
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[10:17] /*v[266:273]*/, v[4:11] /*v[516:523]*/, v[180:187] /*v[436:443]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[130:131] /*v[386:387]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[130:131] /*v[386:387]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[130:131] /*v[386:387]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[130:131] /*v[386:387]*/, v[194:195] /*v[450:451]*/
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[98:105] /*v[354:361]*/, v[4:11] /*v[516:523]*/, v[204:211] /*v[460:467]*/
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, s[6:7] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[220:227] /*v[476:483]*/, v[34:41] /*v[290:297]*/, v[76:83] /*v[588:595]*/, v[220:227] /*v[476:483]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v160 /*v416*/, v160 /*v416*/
	v_exp_f32_e32 v161 /*v417*/, v161 /*v417*/
	v_exp_f32_e32 v172 /*v428*/, v172 /*v428*/
	v_exp_f32_e32 v173 /*v429*/, v173 /*v429*/
	v_wmma_f32_16x16x32_bf16 v[236:243] /*v[492:499]*/, v[106:113] /*v[362:369]*/, v[76:83] /*v[588:595]*/, v[236:243] /*v[492:499]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[130:131] /*v[386:387]*/, v[220:221] /*v[476:477]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[130:131] /*v[386:387]*/, v[222:223] /*v[478:479]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[130:131] /*v[386:387]*/, v[224:225] /*v[480:481]*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[130:131] /*v[386:387]*/, v[226:227] /*v[482:483]*/
	v_exp_f32_e32 v174 /*v430*/, v174 /*v430*/
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[228:235] /*v[484:491]*/, v[10:17] /*v[266:273]*/, v[20:27] /*v[532:539]*/, v[228:235] /*v[484:491]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[130:131] /*v[386:387]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[130:131] /*v[386:387]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[130:131] /*v[386:387]*/, v[240:241] /*v[496:497]*/
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[130:131] /*v[386:387]*/, v[242:243] /*v[498:499]*/
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[252:259] /*v[508:515]*/, v[98:105] /*v[354:361]*/, v[20:27] /*v[532:539]*/, v[252:259] /*v[508:515]*/
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[226:227] /*v[482:483]*/, v[226:227] /*v[482:483]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[6:7] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[26:33] /*v[282:289]*/, v[36:43] /*v[548:555]*/, v[180:187] /*v[436:443]*/
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[226:227] /*v[482:483]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v236 /*v492*/, v156 /*v412*/
	v_exp_f32_e32 v237 /*v493*/, v157 /*v413*/
	v_exp_f32_e32 v156 /*v412*/, v158 /*v414*/
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[58:65] /*v[314:321]*/, v[36:43] /*v[548:555]*/, v[204:211] /*v[460:467]*/
	v_exp_f32_e32 v157 /*v413*/, v159 /*v415*/
	v_exp_f32_e32 v158 /*v414*/, v162 /*v418*/
	v_exp_f32_e32 v159 /*v415*/, v163 /*v419*/
	v_exp_f32_e32 v175 /*v431*/, v175 /*v431*/
	v_exp_f32_e32 v176 /*v432*/, v176 /*v432*/
	v_exp_f32_e32 v177 /*v433*/, v177 /*v433*/
	v_exp_f32_e32 v162 /*v418*/, v178 /*v434*/
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[228:235] /*v[484:491]*/, v[26:33] /*v[282:289]*/, v[52:59] /*v[564:571]*/, v[228:235] /*v[484:491]*/
	v_exp_f32_e32 v163 /*v419*/, v179 /*v435*/
	v_exp_f32_e32 v178 /*v434*/, v188 /*v444*/
	v_exp_f32_e32 v179 /*v435*/, v189 /*v445*/
	v_exp_f32_e32 v188 /*v444*/, v190 /*v446*/
	v_exp_f32_e32 v189 /*v445*/, v191 /*v447*/
	v_exp_f32_e32 v190 /*v446*/, v192 /*v448*/
	v_exp_f32_e32 v191 /*v447*/, v193 /*v449*/
	v_wmma_f32_16x16x32_bf16 v[252:259] /*v[508:515]*/, v[58:65] /*v[314:321]*/, v[52:59] /*v[564:571]*/, v[252:259] /*v[508:515]*/
	v_exp_f32_e32 v192 /*v448*/, v194 /*v450*/
	v_exp_f32_e32 v193 /*v449*/, v195 /*v451*/
	v_exp_f32_e32 v194 /*v450*/, v220 /*v476*/
	v_exp_f32_e32 v195 /*v451*/, v221 /*v477*/
	v_exp_f32_e32 v220 /*v476*/, v222 /*v478*/
	v_exp_f32_e32 v221 /*v477*/, v223 /*v479*/
	v_exp_f32_e32 v222 /*v478*/, v224 /*v480*/
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[436:443]*/, v[42:49] /*v[298:305]*/, v[68:75] /*v[580:587]*/, v[180:187] /*v[436:443]*/
	v_exp_f32_e32 v223 /*v479*/, v225 /*v481*/
	v_exp_f32_e32 v224 /*v480*/, v226 /*v482*/
	v_exp_f32_e32 v225 /*v481*/, v227 /*v483*/
	v_nop
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[180:181] /*v[436:437]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[204:211] /*v[460:467]*/, v[114:121] /*v[370:377]*/, v[68:75] /*v[580:587]*/, v[204:211] /*v[460:467]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[182:183] /*v[438:439]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[184:185] /*v[440:441]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[186:187] /*v[442:443]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[164:165] /*v[420:421]*/, v[236:237] /*v[492:493]*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v188 /*v444*/, v189 /*v445*/
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[166:167] /*v[422:423]*/, v[156:157] /*v[412:413]*/
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[168:169] /*v[424:425]*/, v[160:161] /*v[416:417]*/
	s_set_vgpr_msb 0x4559
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[228:235] /*v[484:491]*/, v[42:49] /*v[298:305]*/, v[84:91] /*v[596:603]*/, v[228:235] /*v[484:491]*/
	v_pk_add_f32 v[180:181] /*v[436:437]*/, v[204:205] /*v[460:461]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[206:207] /*v[462:463]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[184:185] /*v[440:441]*/, v[208:209] /*v[464:465]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[186:187] /*v[442:443]*/, v[210:211] /*v[466:467]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[242:243] /*v[498:499]*/, v[170:171] /*v[426:427]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, v[174:175] /*v[430:431]*/
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[252:259] /*v[508:515]*/, v[114:121] /*v[370:377]*/, v[84:91] /*v[596:603]*/, v[252:259] /*v[508:515]*/
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[228:229] /*v[484:485]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[230:231] /*v[486:487]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[232:233] /*v[488:489]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[210:211] /*v[466:467]*/, v[234:235] /*v[490:491]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[184:185] /*v[440:441]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, v[162:163] /*v[418:419]*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v158 /*v414*/, v159 /*v415*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[228:229] /*v[484:485]*/, v[252:253] /*v[508:509]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[254:255] /*v[510:511]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[0:1] /*v[512:513]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[2:3] /*v[514:515]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v158 /*v414*/, v160 /*v416*/, v161 /*v417*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v156 /*v412*/, v157 /*v413*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v172 /*v428*/, v173 /*v429*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[204:205] /*v[460:461]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[206:207] /*v[462:463]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[208:209] /*v[464:465]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[210:211] /*v[466:467]*/, v[192:193] /*v[448:449]*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v192 /*v448*/, v193 /*v449*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v190 /*v446*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v178 /*v434*/, v179 /*v435*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[228:229] /*v[484:485]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[230:231] /*v[486:487]*/, v[220:221] /*v[476:477]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[232:233] /*v[488:489]*/, v[222:223] /*v[478:479]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[234:235] /*v[490:491]*/, v[224:225] /*v[480:481]*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v194 /*v450*/, v195 /*v451*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[130:131] /*v[386:387]*/, v[226:227] /*v[482:483]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[130:131] /*v[386:387]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[130:131] /*v[386:387]*/, v[240:241] /*v[496:497]*/
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[130:131] /*v[386:387]*/, v[242:243] /*v[498:499]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[130:131] /*v[386:387]*/, v[180:181] /*v[436:437]*/
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[130:131] /*v[386:387]*/, v[182:183] /*v[438:439]*/
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[130:131] /*v[386:387]*/, v[184:185] /*v[440:441]*/
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[130:131] /*v[386:387]*/, v[186:187] /*v[442:443]*/
	s_set_vgpr_msb 0x4504
	ds_store_b128 v1, v[156:159] /*v[412:415]*/
	ds_store_b128 v1, v[160:163] /*v[416:419]*/ offset:32
	s_set_vgpr_msb 0x445
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[130:131] /*v[386:387]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[130:131] /*v[386:387]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[130:131] /*v[386:387]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[130:131] /*v[386:387]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[130:131] /*v[386:387]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[130:131] /*v[386:387]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[130:131] /*v[386:387]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[130:131] /*v[386:387]*/, v[192:193] /*v[448:449]*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v208 /*v464*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v210 /*v466*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v182 /*v438*/, v183 /*v439*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v184 /*v440*/, v185 /*v441*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v186 /*v442*/, v187 /*v443*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v224 /*v480*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v222 /*v478*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v220 /*v476*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v172 /*v428*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v175 /*v431*/, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v188 /*v444*/, v189 /*v445*/
	v_cvt_pk_bf16_f32 v178 /*v434*/, v190 /*v446*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v192 /*v448*/, v193 /*v449*/
	s_set_vgpr_msb 0x4504
	ds_store_b128 v1, v[156:159] /*v[412:415]*/ offset:2560
	ds_store_b128 v1, v[160:163] /*v[416:419]*/ offset:2592
	s_set_vgpr_msb 0x405
	ds_store_b128 v146 /*v402*/, v[212:215] /*v[468:471]*/ offset:9472
	ds_store_b128 v146 /*v402*/, v[216:219] /*v[472:475]*/ offset:9504
	ds_store_b128 v146 /*v402*/, v[196:199] /*v[452:455]*/ offset:18176
	ds_store_b128 v146 /*v402*/, v[200:203] /*v[456:459]*/ offset:18208
	s_set_vgpr_msb 0x509
	ds_store_b128 v146 /*v402*/, v[20:23] /*v[532:535]*/ offset:9536
	ds_store_b128 v146 /*v402*/, v[24:27] /*v[536:539]*/ offset:9568
	ds_store_b128 v146 /*v402*/, v[12:15] /*v[524:527]*/ offset:18240
	ds_store_b128 v146 /*v402*/, v[16:19] /*v[528:531]*/ offset:18272
	ds_store_b128 v146 /*v402*/, v[52:55] /*v[564:567]*/ offset:9600
	ds_store_b128 v146 /*v402*/, v[56:59] /*v[568:571]*/ offset:9632
	ds_store_b128 v146 /*v402*/, v[44:47] /*v[556:559]*/ offset:18304
	ds_store_b128 v146 /*v402*/, v[48:51] /*v[560:563]*/ offset:18336
	ds_store_b128 v146 /*v402*/, v[84:87] /*v[596:599]*/ offset:9664
	ds_store_b128 v146 /*v402*/, v[88:91] /*v[600:603]*/ offset:9696
	ds_store_b128 v146 /*v402*/, v[76:79] /*v[588:591]*/ offset:18368
	ds_store_b128 v146 /*v402*/, v[80:83] /*v[592:595]*/ offset:18400
	s_set_vgpr_msb 0x904
	ds_store_b128 v1, v[164:167] /*v[420:423]*/ offset:1280
	ds_store_b128 v1, v[168:171] /*v[424:427]*/ offset:1312
	ds_store_b128 v1, v[172:175] /*v[428:431]*/ offset:3840
	ds_store_b128 v1, v[176:179] /*v[432:435]*/ offset:3872
	s_set_vgpr_msb 0x441
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v147 /*v403*/ offset:1280
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v134 /*v390*/
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v135 /*v391*/ offset:4352
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v136 /*v392*/
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v137 /*v393*/ offset:4352
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v138 /*v394*/
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v139 /*v395*/ offset:4352
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v140 /*v396*/
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v142 /*v398*/ offset:4352
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v143 /*v399*/
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[8:11] /*v[520:523]*/, v149 /*v405*/ offset:4352
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v150 /*v406*/
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v151 /*v407*/ offset:4352
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v152 /*v408*/
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v134 /*v390*/ offset:4352
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v135 /*v391*/
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v136 /*v392*/ offset:4352
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v137 /*v393*/
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v138 /*v394*/ offset:4352
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v139 /*v395*/
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v140 /*v396*/ offset:4352
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v142 /*v398*/
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v143 /*v399*/ offset:4352
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v149 /*v405*/
	ds_load_tr16_b128 v[16:19] /*v[528:531]*/, v150 /*v406*/ offset:4352
	ds_load_tr16_b128 v[20:23] /*v[532:535]*/, v151 /*v407*/
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v152 /*v408*/ offset:4352
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v153 /*v409*/
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v132 /*v388*/
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v132 /*v388*/ offset:4352
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v154 /*v410*/
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v154 /*v410*/ offset:1280
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v133 /*v389*/
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v133 /*v389*/ offset:4352
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v147 /*v403*/
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[234:241], v[164:171] /*v[420:427]*/, v[156:163] /*v[412:419]*/, v[234:241]
	v_wmma_f32_16x16x32_bf16 v[218:225], v[164:171] /*v[420:427]*/, v[188:195] /*v[444:451]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[202:209], v[164:171] /*v[420:427]*/, v[204:211] /*v[460:467]*/, v[202:209]
	v_wmma_f32_16x16x32_bf16 v[186:193], v[164:171] /*v[420:427]*/, v[220:227] /*v[476:483]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[170:177], v[164:171] /*v[420:427]*/, v[236:243] /*v[492:499]*/, v[170:177]
	v_wmma_f32_16x16x32_bf16 v[154:161], v[164:171] /*v[420:427]*/, v[252:259] /*v[508:515]*/, v[154:161]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[138:145], v[164:171] /*v[420:427]*/, v[12:19] /*v[524:531]*/, v[138:145]
	v_wmma_f32_16x16x32_bf16 v[130:137], v[164:171] /*v[420:427]*/, v[28:35] /*v[540:547]*/, v[130:137]
	s_set_vgpr_msb 0x981
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v153 /*v409*/ offset:4352
	s_set_vgpr_msb 0x8155
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v148 /*v404*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[180:187] /*v[436:443]*/, v[172:179] /*v[428:435]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[180:187] /*v[436:443]*/, v[196:203] /*v[452:459]*/, v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[226:233], v[180:187] /*v[436:443]*/, v[212:219] /*v[468:475]*/, v[226:233]
	v_wmma_f32_16x16x32_bf16 v[210:217], v[180:187] /*v[436:443]*/, v[228:235] /*v[484:491]*/, v[210:217]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[180:187] /*v[436:443]*/, v[244:251] /*v[500:507]*/, v[194:201]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[178:185], v[180:187] /*v[436:443]*/, v[4:11] /*v[516:523]*/, v[178:185]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[180:187] /*v[436:443]*/, v[20:27] /*v[532:539]*/, v[162:169]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[146:153], v[180:187] /*v[436:443]*/, v[36:43] /*v[548:555]*/, v[146:153]
	s_set_vgpr_msb 0x941
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v148 /*v404*/ offset:1280
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v155 /*v411*/
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v155 /*v411*/ offset:1280
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_wmma_f32_16x16x32_bf16 v[106:113], v[164:171] /*v[420:427]*/, v[156:163] /*v[412:419]*/, v[106:113]
	s_barrier_wait -1
	v_wmma_f32_16x16x32_bf16 v[122:129], v[180:187] /*v[436:443]*/, v[172:179] /*v[428:435]*/, v[122:129]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[164:171] /*v[420:427]*/, v[188:195] /*v[444:451]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[114:121], v[180:187] /*v[436:443]*/, v[196:203] /*v[452:459]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[164:171] /*v[420:427]*/, v[204:211] /*v[460:467]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[98:105], v[180:187] /*v[436:443]*/, v[212:219] /*v[468:475]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[164:171] /*v[420:427]*/, v[220:227] /*v[476:483]*/, v[58:65]
	v_wmma_f32_16x16x32_bf16 v[82:89], v[180:187] /*v[436:443]*/, v[228:235] /*v[484:491]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[164:171] /*v[420:427]*/, v[236:243] /*v[492:499]*/, v[42:49]
	v_wmma_f32_16x16x32_bf16 v[66:73], v[180:187] /*v[436:443]*/, v[244:251] /*v[500:507]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[164:171] /*v[420:427]*/, v[252:259] /*v[508:515]*/, v[26:33]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[50:57], v[180:187] /*v[436:443]*/, v[4:11] /*v[516:523]*/, v[50:57]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[164:171] /*v[420:427]*/, v[12:19] /*v[524:531]*/, v[10:17]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[180:187] /*v[436:443]*/, v[20:27] /*v[532:539]*/, v[34:41]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[164:171] /*v[420:427]*/, v[28:35] /*v[540:547]*/, v[2:9]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[180:187] /*v[436:443]*/, v[36:43] /*v[548:555]*/, v[18:25]
	s_set_vgpr_msb 0x900
	s_cbranch_scc1 .LBB0_6
.LBB0_7:
	s_mul_i32 s4, s40, s68
	s_clause 0x1
	s_load_b64 s[6:7], s[0:1], 0x110 nv
	s_load_b64 s[8:9], s[0:1], 0x140 nv
	s_add_co_i32 s4, s4, s67
	s_wait_xcnt 0x0
	s_mov_b32 s0, 0
	s_set_vgpr_msb 4
	v_mad_u32 v1, s40, v144 /*v400*/, s4
	s_lshl_b32 s5, s40, 1
	v_cvt_pk_bf16_f32 v234, v234, s0
	s_lshl_b32 s1, s34, 25
	s_set_vgpr_msb 0x401
	s_wait_loadcnt 0x1f
	v_cvt_pk_bf16_f32 v243, v122 /*v378*/, s0
	s_set_vgpr_msb 0x104
	v_cvt_pk_bf16_f32 v235, v235, s0
	s_mov_b32 s2, s46
	s_mov_b32 s3, s47
	v_add_lshl_u32 v244, v1, s40, 7
	s_wait_loadcnt 0x1e
	v_add_lshl_u32 v247, v1, s5, 7
	s_mul_i32 s5, s40, 3
	s_delay_alu instid0(SALU_CYCLE_1)
	v_add_lshl_u32 v248, v1, s5, 7
	v_or_b32_e32 v246, v244, v141 /*v397*/
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v242, 7, v1
	s_lshl_b32 s5, s40, 2
	s_wait_kmcnt 0x0
	s_or_b64 s[44:45], s[6:7], s[0:1]
	s_or_b64 s[0:1], s[8:9], s[0:1]
	v_lshlrev_b32_e32 v246, 2, v246
	s_set_vgpr_msb 4
	v_or_b32_e32 v245, v242, v141 /*v397*/
	v_cvt_pk_bf16_f32 v236, v236, s0
	v_or_b32_e32 v249, v248, v141 /*v397*/
	v_cvt_pk_bf16_f32 v237, v237, s0
	v_cvt_pk_bf16_f32 v238, v238, s0
	s_set_vgpr_msb 0x401
	v_lshlrev_b32_e32 v245, 2, v245
	s_wait_loadcnt 0x1d
	v_cvt_pk_bf16_f32 v250, v126 /*v382*/, s0
	s_set_vgpr_msb 0x104
	v_cvt_pk_bf16_f32 v239, v239, s0
	v_cvt_pk_bf16_f32 v241, v241, s0
	buffer_store_b16 v234, v245, s[44:47], null offen
	buffer_store_b16 v243, v245, s[0:3], null offen
	buffer_store_b16 v235, v246, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v235, v247, v141 /*v397*/
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v234, v123 /*v379*/, s0
	v_cvt_pk_bf16_f32 v243, v124 /*v380*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v240, v240, s0
	v_cvt_pk_bf16_f32 v218, v218, s0
	v_lshlrev_b32_e32 v235, 2, v235
	v_cvt_pk_bf16_f32 v219, v219, s0
	buffer_store_b16 v234, v246, s[0:3], null offen
	buffer_store_b16 v236, v235, s[44:47], null offen
	buffer_store_b16 v243, v235, s[0:3], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v243, v1, s5, 7
	v_lshlrev_b32_e32 v234, 2, v249
	v_mov_b16_e64 v236.l, v237.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v237, v125 /*v381*/, s0
	s_mul_i32 s5, s40, 5
	v_or_b32_e32 v249, v141 /*v397*/, v243
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v221, v221, s0
	v_cvt_pk_bf16_f32 v222, v222, s0
	v_cvt_pk_bf16_f32 v203, v203, s0
	v_cvt_pk_bf16_f32 v202, v202, s0
	v_lshlrev_b32_e32 v249, 2, v249
	buffer_store_b16 v236, v234, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v236, v1, s5, 7
	s_mul_i32 s5, s40, 6
	buffer_store_b16 v237, v234, s[0:3], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v237, v1, s5, 7
	s_mul_i32 s5, s40, 7
	s_set_vgpr_msb 4
	v_or_b32_e32 v251, v236, v141 /*v397*/
	v_add_lshl_u32 v252, v1, s5, 7
	v_cvt_pk_bf16_f32 v204, v204, s0
	v_cvt_pk_bf16_f32 v205, v205, s0
	v_cvt_pk_bf16_f32 v186, v186, s0
	v_cvt_pk_bf16_f32 v187, v187, s0
	v_or_b32_e32 v253, v252, v141 /*v397*/
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v251, 2, v251
	buffer_store_b16 v238, v249, s[44:47], null offen
	buffer_store_b16 v250, v249, s[0:3], null offen
	buffer_store_b16 v239, v251, s[44:47], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v239, v237, v141 /*v397*/
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v238, v127 /*v383*/, s0
	v_cvt_pk_bf16_f32 v250, v128 /*v384*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v189, v189, s0
	v_cvt_pk_bf16_f32 v188, v188, s0
	v_lshlrev_b32_e32 v239, 2, v239
	buffer_store_b16 v238, v251, s[0:3], null offen
	buffer_store_b16 v240, v239, s[44:47], null offen
	buffer_store_b16 v250, v239, s[0:3], null offen
	s_wait_xcnt 0x2
	v_lshlrev_b32_e32 v238, 2, v253
	s_wait_xcnt 0x1
	v_mov_b16_e64 v240.l, v241.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v241, v129 /*v385*/, s0
	v_cvt_pk_bf16_f32 v250, v82 /*v338*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v170, v170, s0
	v_cvt_pk_bf16_f32 v171, v171, s0
	buffer_store_b16 v240, v238, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v240.l, v241.l
	v_mov_b16_e64 v241.l, v250.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v250, v83 /*v339*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v240, v238, s[0:3], null offen
	buffer_store_b16 v218, v245, s[44:47], null offen offset:64
	buffer_store_b16 v241, v245, s[0:3], null offen offset:64
	buffer_store_b16 v219, v246, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v219, v220, s0
	v_mov_b16_e64 v218.l, v250.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v220, v84 /*v340*/, s0
	v_cvt_pk_bf16_f32 v240, v85 /*v341*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v172, v172, s0
	v_cvt_pk_bf16_f32 v154, v154, s0
	buffer_store_b16 v218, v246, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v218.l, v219.l
	v_mov_b16_e64 v219.l, v220.l
	v_mov_b16_e64 v220.l, v221.l
	v_mov_b16_e64 v221.l, v240.l
	buffer_store_b16 v218, v235, s[44:47], null offen offset:64
	buffer_store_b16 v219, v235, s[0:3], null offen offset:64
	buffer_store_b16 v220, v234, s[44:47], null offen offset:64
	buffer_store_b16 v221, v234, s[0:3], null offen offset:64
	s_wait_xcnt 0x3
	v_mov_b16_e64 v218.l, v222.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v219, v86 /*v342*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v220, v223, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v221, v87 /*v343*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v222, v224, s0
	buffer_store_b16 v218, v249, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v218.l, v219.l
	v_mov_b16_e64 v219.l, v220.l
	v_mov_b16_e64 v220.l, v221.l
	v_mov_b16_e64 v221.l, v222.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v222, v88 /*v344*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v218, v249, s[0:3], null offen offset:64
	buffer_store_b16 v219, v251, s[44:47], null offen offset:64
	buffer_store_b16 v220, v251, s[0:3], null offen offset:64
	buffer_store_b16 v221, v239, s[44:47], null offen offset:64
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v219, v225, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v220, v89 /*v345*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v221, v226, s0
	v_mov_b16_e64 v218.l, v222.l
	v_cvt_pk_bf16_f32 v156, v156, s0
	v_cvt_pk_bf16_f32 v155, v155, s0
	v_cvt_pk_bf16_f32 v158, v158, s0
	v_cvt_pk_bf16_f32 v138, v138, s0
	buffer_store_b16 v218, v239, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v218.l, v219.l
	v_mov_b16_e64 v219.l, v220.l
	v_mov_b16_e64 v220.l, v221.l
	buffer_store_b16 v218, v238, s[44:47], null offen offset:64
	buffer_store_b16 v219, v238, s[0:3], null offen offset:64
	buffer_store_b16 v202, v245, s[44:47], null offen offset:128
	buffer_store_b16 v220, v245, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_mov_b16_e64 v202.l, v203.l
	v_cvt_pk_bf16_f32 v203, v227, s0
	v_cvt_pk_bf16_f32 v218, v228, s0
	v_cvt_pk_bf16_f32 v139, v139, s0
	v_cvt_pk_bf16_f32 v141, v141, s0
	buffer_store_b16 v202, v246, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v202.l, v203.l
	v_mov_b16_e64 v203.l, v204.l
	v_mov_b16_e64 v204.l, v218.l
	v_cvt_pk_bf16_f32 v218, v229, s0
	buffer_store_b16 v202, v246, s[0:3], null offen offset:128
	buffer_store_b16 v203, v235, s[44:47], null offen offset:128
	buffer_store_b16 v204, v235, s[0:3], null offen offset:128
	buffer_store_b16 v205, v234, s[44:47], null offen offset:128
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v203, v206, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v204, v230, s0
	v_mov_b16_e64 v202.l, v218.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v205, v207, s0
	v_cvt_pk_bf16_f32 v206, v231, s0
	v_cvt_pk_bf16_f32 v142, v142, s0
	v_cvt_pk_bf16_f32 v130, v130, s0
	buffer_store_b16 v202, v234, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v202.l, v203.l
	v_mov_b16_e64 v203.l, v204.l
	v_mov_b16_e64 v204.l, v205.l
	v_mov_b16_e64 v205.l, v206.l
	v_cvt_pk_bf16_f32 v206, v208, s0
	buffer_store_b16 v202, v249, s[44:47], null offen offset:128
	buffer_store_b16 v203, v249, s[0:3], null offen offset:128
	buffer_store_b16 v204, v251, s[44:47], null offen offset:128
	buffer_store_b16 v205, v251, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v203, v232, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v204, v209, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v205, v233, s0
	v_mov_b16_e64 v202.l, v206.l
	v_cvt_pk_bf16_f32 v131, v131, s0
	v_cvt_pk_bf16_f32 v132, v132, s0
	v_cvt_pk_bf16_f32 v134, v134, s0
	v_cvt_pk_bf16_f32 v136, v136, s0
	buffer_store_b16 v202, v239, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v202.l, v203.l
	v_mov_b16_e64 v203.l, v204.l
	v_mov_b16_e64 v204.l, v205.l
	v_cvt_pk_bf16_f32 v205, v210, s0
	buffer_store_b16 v202, v239, s[0:3], null offen offset:128
	buffer_store_b16 v203, v238, s[44:47], null offen offset:128
	buffer_store_b16 v204, v238, s[0:3], null offen offset:128
	buffer_store_b16 v186, v245, s[44:47], null offen offset:192
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v202, v211, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v203, v212, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v186.l, v205.l
	v_add_lshl_u32 v1, v1, s66, 7
	v_cvt_pk_bf16_f32 v106, v106, s0
	v_cvt_pk_bf16_f32 v122, v122, s0
	v_cvt_pk_bf16_f32 v107, v107, s0
	buffer_store_b16 v186, v245, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_mov_b16_e64 v186.l, v187.l
	v_mov_b16_e64 v187.l, v202.l
	v_mov_b16_e64 v202.l, v203.l
	buffer_store_b16 v186, v246, s[44:47], null offen offset:192
	buffer_store_b16 v187, v246, s[0:3], null offen offset:192
	buffer_store_b16 v188, v235, s[44:47], null offen offset:192
	buffer_store_b16 v202, v235, s[0:3], null offen offset:192
	s_wait_xcnt 0x3
	v_mov_b16_e64 v186.l, v189.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v187, v213, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v188, v190, s0
	v_cvt_pk_bf16_f32 v189, v214, s0
	v_cvt_pk_bf16_f32 v190, v191, s0
	buffer_store_b16 v186, v234, s[44:47], null offen offset:192
	s_wait_xcnt 0x0
	v_mov_b16_e64 v186.l, v187.l
	v_mov_b16_e64 v187.l, v188.l
	v_mov_b16_e64 v188.l, v189.l
	v_mov_b16_e64 v189.l, v190.l
	v_cvt_pk_bf16_f32 v190, v215, s0
	buffer_store_b16 v186, v234, s[0:3], null offen offset:192
	buffer_store_b16 v187, v249, s[44:47], null offen offset:192
	buffer_store_b16 v188, v249, s[0:3], null offen offset:192
	buffer_store_b16 v189, v251, s[44:47], null offen offset:192
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v187, v192, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v188, v216, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v189, v193, s0
	v_mov_b16_e64 v186.l, v190.l
	v_cvt_pk_bf16_f32 v190, v217, s0
	v_cvt_pk_bf16_f32 v124, v124, s0
	v_cvt_pk_bf16_f32 v123, v123, s0
	v_cvt_pk_bf16_f32 v108, v108, s0
	buffer_store_b16 v186, v251, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_mov_b16_e64 v186.l, v187.l
	v_mov_b16_e64 v187.l, v188.l
	v_mov_b16_e64 v188.l, v189.l
	v_mov_b16_e64 v189.l, v190.l
	buffer_store_b16 v186, v239, s[44:47], null offen offset:192
	buffer_store_b16 v187, v239, s[0:3], null offen offset:192
	buffer_store_b16 v188, v238, s[44:47], null offen offset:192
	buffer_store_b16 v189, v238, s[0:3], null offen offset:192
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v186, v194, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v187, v195, s0
	buffer_store_b16 v170, v245, s[44:47], null offen offset:256
	v_cvt_pk_bf16_f32 v109, v109, s0
	v_cvt_pk_bf16_f32 v110, v110, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v186.l
	v_mov_b16_e64 v186.l, v187.l
	v_cvt_pk_bf16_f32 v187, v196, s0
	buffer_store_b16 v170, v245, s[0:3], null offen offset:256
	buffer_store_b16 v171, v246, s[44:47], null offen offset:256
	buffer_store_b16 v186, v246, s[0:3], null offen offset:256
	buffer_store_b16 v172, v235, s[44:47], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v171, v173, s0
	v_mov_b16_e64 v170.l, v187.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v172, v197, s0
	v_cvt_pk_bf16_f32 v173, v174, s0
	v_cvt_pk_bf16_f32 v174, v198, s0
	v_cvt_pk_bf16_f32 v126, v126, s0
	buffer_store_b16 v170, v235, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v171.l
	v_mov_b16_e64 v171.l, v172.l
	v_mov_b16_e64 v172.l, v173.l
	v_mov_b16_e64 v173.l, v174.l
	v_cvt_pk_bf16_f32 v174, v175, s0
	buffer_store_b16 v170, v234, s[44:47], null offen offset:256
	buffer_store_b16 v171, v234, s[0:3], null offen offset:256
	buffer_store_b16 v172, v249, s[44:47], null offen offset:256
	buffer_store_b16 v173, v249, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v171, v199, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v172, v176, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v173, v200, s0
	v_mov_b16_e64 v170.l, v174.l
	v_cvt_pk_bf16_f32 v174, v177, s0
	v_cvt_pk_bf16_f32 v127, v127, s0
	v_cvt_pk_bf16_f32 v111, v111, s0
	v_cvt_pk_bf16_f32 v112, v112, s0
	buffer_store_b16 v170, v251, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v171.l
	v_mov_b16_e64 v171.l, v172.l
	v_mov_b16_e64 v172.l, v173.l
	v_mov_b16_e64 v173.l, v174.l
	v_cvt_pk_bf16_f32 v174, v201, s0
	buffer_store_b16 v170, v251, s[0:3], null offen offset:256
	buffer_store_b16 v171, v239, s[44:47], null offen offset:256
	buffer_store_b16 v172, v239, s[0:3], null offen offset:256
	buffer_store_b16 v173, v238, s[44:47], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v171, v178, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v172, v179, s0
	v_cvt_pk_bf16_f32 v113, v113, s0
	v_mov_b16_e64 v170.l, v174.l
	v_cvt_pk_bf16_f32 v90, v90, s0
	v_cvt_pk_bf16_f32 v91, v91, s0
	v_cvt_pk_bf16_f32 v92, v92, s0
	v_cvt_pk_bf16_f32 v74, v74, s0
	buffer_store_b16 v170, v238, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v171.l
	v_mov_b16_e64 v171.l, v172.l
	buffer_store_b16 v154, v245, s[44:47], null offen offset:320
	buffer_store_b16 v170, v245, s[0:3], null offen offset:320
	buffer_store_b16 v155, v246, s[44:47], null offen offset:320
	buffer_store_b16 v171, v246, s[0:3], null offen offset:320
	s_wait_xcnt 0x3
	v_mov_b16_e64 v154.l, v156.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v155, v180, s0
	v_cvt_pk_bf16_f32 v156, v157, s0
	v_cvt_pk_bf16_f32 v157, v181, s0
	v_cvt_pk_bf16_f32 v75, v75, s0
	buffer_store_b16 v154, v235, s[44:47], null offen offset:320
	s_wait_xcnt 0x0
	v_mov_b16_e64 v154.l, v155.l
	v_mov_b16_e64 v155.l, v156.l
	v_mov_b16_e64 v156.l, v157.l
	v_mov_b16_e64 v157.l, v158.l
	v_cvt_pk_bf16_f32 v158, v182, s0
	buffer_store_b16 v154, v235, s[0:3], null offen offset:320
	buffer_store_b16 v155, v234, s[44:47], null offen offset:320
	buffer_store_b16 v156, v234, s[0:3], null offen offset:320
	buffer_store_b16 v157, v249, s[44:47], null offen offset:320
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v155, v159, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v156, v183, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v157, v160, s0
	v_mov_b16_e64 v154.l, v158.l
	v_cvt_pk_bf16_f32 v158, v184, s0
	v_cvt_pk_bf16_f32 v76, v76, s0
	v_cvt_pk_bf16_f32 v58, v58, s0
	v_cvt_pk_bf16_f32 v59, v59, s0
	buffer_store_b16 v154, v249, s[0:3], null offen offset:320
	s_wait_xcnt 0x0
	v_mov_b16_e64 v154.l, v155.l
	v_mov_b16_e64 v155.l, v156.l
	v_mov_b16_e64 v156.l, v157.l
	v_mov_b16_e64 v157.l, v158.l
	v_cvt_pk_bf16_f32 v158, v161, s0
	buffer_store_b16 v154, v251, s[44:47], null offen offset:320
	buffer_store_b16 v155, v251, s[0:3], null offen offset:320
	buffer_store_b16 v156, v239, s[44:47], null offen offset:320
	buffer_store_b16 v157, v239, s[0:3], null offen offset:320
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v155, v185, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v156, v162, s0
	v_cvt_pk_bf16_f32 v62, v62, s0
	v_mov_b16_e64 v154.l, v158.l
	v_cvt_pk_bf16_f32 v42, v42, s0
	v_cvt_pk_bf16_f32 v43, v43, s0
	v_cvt_pk_bf16_f32 v45, v45, s0
	v_cvt_pk_bf16_f32 v26, v26, s0
	buffer_store_b16 v154, v238, s[44:47], null offen offset:320
	s_wait_xcnt 0x0
	v_mov_b16_e64 v154.l, v155.l
	v_mov_b16_e64 v155.l, v156.l
	v_cvt_pk_bf16_f32 v156, v163, s0
	buffer_store_b16 v154, v238, s[0:3], null offen offset:320
	buffer_store_b16 v138, v245, s[44:47], null offen offset:384
	buffer_store_b16 v155, v245, s[0:3], null offen offset:384
	buffer_store_b16 v139, v246, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v139, v140, s0
	v_mov_b16_e64 v138.l, v156.l
	v_cvt_pk_bf16_f32 v140, v164, s0
	v_cvt_pk_bf16_f32 v154, v165, s0
	v_cvt_pk_bf16_f32 v28, v28, s0
	v_cvt_pk_bf16_f32 v29, v29, s0
	buffer_store_b16 v138, v246, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v138.l, v139.l
	v_mov_b16_e64 v139.l, v140.l
	v_mov_b16_e64 v140.l, v141.l
	v_mov_b16_e64 v141.l, v154.l
	buffer_store_b16 v138, v235, s[44:47], null offen offset:384
	buffer_store_b16 v139, v235, s[0:3], null offen offset:384
	buffer_store_b16 v140, v234, s[44:47], null offen offset:384
	buffer_store_b16 v141, v234, s[0:3], null offen offset:384
	s_wait_xcnt 0x3
	v_mov_b16_e64 v138.l, v142.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v139, v166, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v140, v143, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v141, v167, s0
	v_cvt_pk_bf16_f32 v142, v144, s0
	buffer_store_b16 v138, v249, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v138.l, v139.l
	v_mov_b16_e64 v139.l, v140.l
	v_mov_b16_e64 v140.l, v141.l
	v_mov_b16_e64 v141.l, v142.l
	v_cvt_pk_bf16_f32 v142, v168, s0
	buffer_store_b16 v138, v249, s[0:3], null offen offset:384
	buffer_store_b16 v139, v251, s[44:47], null offen offset:384
	buffer_store_b16 v140, v251, s[0:3], null offen offset:384
	buffer_store_b16 v141, v239, s[44:47], null offen offset:384
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v139, v145, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v140, v169, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v141, v242, v0
	v_mov_b16_e64 v138.l, v142.l
	v_cvt_pk_bf16_f32 v142, v148, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_cvt_pk_bf16_f32 v12, v12, s0
	buffer_store_b16 v138, v239, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v138.l, v139.l
	v_mov_b16_e64 v139.l, v140.l
	v_lshl_or_b32 v140, v141, 2, 0x1c0
	v_cvt_pk_bf16_f32 v141, v146, s0
	v_cvt_pk_bf16_f32 v2, v2, s0
	buffer_store_b16 v138, v238, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_or_b32_e32 v138, v244, v0
	buffer_store_b16 v139, v238, s[0:3], null offen offset:384
	buffer_store_b16 v130, v140, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v130.l, v141.l
	v_or_b32_e32 v141, v247, v0
	v_cvt_pk_bf16_f32 v139, v147, s0
	v_lshl_or_b32 v138, v138, 2, 0x1c0
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_lshl_or_b32 v141, v141, 2, 0x1c0
	buffer_store_b16 v130, v140, s[0:3], null offen
	buffer_store_b16 v131, v138, s[44:47], null offen
	buffer_store_b16 v139, v138, s[0:3], null offen
	buffer_store_b16 v132, v141, s[44:47], null offen
	buffer_store_b16 v142, v141, s[0:3], null offen
	s_wait_xcnt 0x4
	v_or_b32_e32 v130, v248, v0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v131, v133, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v132, v149, s0
	v_or_b32_e32 v133, v243, v0
	v_cvt_pk_bf16_f32 v138, v150, s0
	v_lshl_or_b32 v130, v130, 2, 0x1c0
	buffer_store_b16 v131, v130, s[44:47], null offen
	buffer_store_b16 v132, v130, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v130, v236, v0
	v_lshl_or_b32 v133, v133, 2, 0x1c0
	v_mov_b16_e64 v131.l, v138.l
	v_cvt_pk_bf16_f32 v132, v135, s0
	v_or_b32_e32 v135, v237, v0
	v_cvt_pk_bf16_f32 v138, v152, s0
	v_lshl_or_b32 v130, v130, 2, 0x1c0
	buffer_store_b16 v134, v133, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v134, v151, s0
	v_lshl_or_b32 v135, v135, 2, 0x1c0
	buffer_store_b16 v131, v133, s[0:3], null offen
	buffer_store_b16 v132, v130, s[44:47], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v131.l, v138.l
	buffer_store_b16 v134, v130, s[0:3], null offen
	buffer_store_b16 v136, v135, s[44:47], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v130, v252, v0
	s_set_vgpr_msb 4
	v_or_b32_e32 v134, v1, v141 /*v397*/
	buffer_store_b16 v131, v135, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_add_nc_u32 v131, 17, v144 /*v400*/ :: v_dual_add_nc_u32 v135, 18, v144 /*v400*/
	v_cvt_pk_bf16_f32 v132, v137, s0
	v_lshl_or_b32 v130, v130, 2, 0x1c0
	v_cvt_pk_bf16_f32 v133, v153, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_lo_u32 v131, v131, s40
	v_mul_lo_u32 v135, v135, s40
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v134, 2, v134
	buffer_store_b16 v132, v130, s[44:47], null offen
	buffer_store_b16 v133, v130, s[0:3], null offen
	buffer_store_b16 v106, v134, s[44:47], null offen
	s_set_vgpr_msb 4
	v_add_nc_u32_e32 v130, 19, v144 /*v400*/
	buffer_store_b16 v122, v134, s[0:3], null offen
	v_add_lshl_u32 v131, v131, s4, 7
	s_wait_xcnt 0x0
	v_add_lshl_u32 v122, v135, s4, 7
	v_add_nc_u32_e32 v133, 20, v144 /*v400*/
	v_mul_lo_u32 v130, v130, s40
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v1, v1, v0
	s_set_vgpr_msb 4
	v_or_b32_e32 v106, v131, v141 /*v397*/
	v_or_b32_e32 v132, v122, v141 /*v397*/
	v_mul_lo_u32 v133, v133, s40
	v_lshl_or_b32 v1, v1, 2, 0x1c0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v106, 2, v106
	v_add_lshl_u32 v130, s4, v130, 7
	v_lshlrev_b32_e32 v132, 2, v132
	buffer_store_b16 v107, v106, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v107.l, v124.l
	buffer_store_b16 v123, v106, s[0:3], null offen
	buffer_store_b16 v108, v132, s[44:47], null offen
	s_set_vgpr_msb 4
	v_dual_add_nc_u32 v123, 21, v144 /*v400*/ :: v_dual_bitop2_b32 v108, v130, v141 /*v397*/ bitop3:0x54
	v_cvt_pk_bf16_f32 v124, v125, s0
	buffer_store_b16 v107, v132, s[0:3], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v107, v133, s4, 7
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v108, 2, v108
	v_mul_lo_u32 v123, s40, v123
	s_set_vgpr_msb 4
	v_or_b32_e32 v125, v107, v141 /*v397*/
	buffer_store_b16 v109, v108, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v109.l, v124.l
	v_add_nc_u32_e32 v124, 22, v144 /*v400*/
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v125, 2, v125
	buffer_store_b16 v109, v108, s[0:3], null offen
	buffer_store_b16 v110, v125, s[44:47], null offen
	v_mul_lo_u32 v124, s40, v124
	s_set_vgpr_msb 4
	v_add_nc_u32_e32 v110, 23, v144 /*v400*/
	v_add_lshl_u32 v123, v123, s4, 7
	v_mov_b16_e32 v109.l, v126.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_lo_u32 v110, v110, s40
	v_or_b32_e32 v126, v123, v141 /*v397*/
	v_add_lshl_u32 v124, v124, s4, 7
	buffer_store_b16 v109, v125, s[0:3], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v109.l, v127.l
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v126, 2, v126
	s_set_vgpr_msb 4
	v_or_b32_e32 v127, v124, v141 /*v397*/
	v_add_lshl_u32 v110, v110, s4, 7
	buffer_store_b16 v111, v126, s[44:47], null offen
	buffer_store_b16 v109, v126, s[0:3], null offen
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v109, 2, v127
	s_set_vgpr_msb 4
	v_or_b32_e32 v127, v110, v141 /*v397*/
	v_mov_b16_e32 v111.l, v112.l
	v_cvt_pk_bf16_f32 v112, v128, s0
	v_cvt_pk_bf16_f32 v128, v129, s0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v127, 2, v127
	buffer_store_b16 v111, v109, s[44:47], null offen
	buffer_store_b16 v112, v109, s[0:3], null offen
	buffer_store_b16 v113, v127, s[44:47], null offen
	buffer_store_b16 v128, v127, s[0:3], null offen
	buffer_store_b16 v90, v134, s[44:47], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v90, v114, s0
	v_cvt_pk_bf16_f32 v111, v115, s0
	v_cvt_pk_bf16_f32 v112, v116, s0
	buffer_store_b16 v90, v134, s[0:3], null offen offset:64
	buffer_store_b16 v91, v106, s[44:47], null offen offset:64
	buffer_store_b16 v111, v106, s[0:3], null offen offset:64
	buffer_store_b16 v92, v132, s[44:47], null offen offset:64
	buffer_store_b16 v112, v132, s[0:3], null offen offset:64
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v90, v93, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v91, v117, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v92, v94, s0
	v_cvt_pk_bf16_f32 v93, v118, s0
	v_cvt_pk_bf16_f32 v94, v95, s0
	buffer_store_b16 v90, v108, s[44:47], null offen offset:64
	buffer_store_b16 v91, v108, s[0:3], null offen offset:64
	buffer_store_b16 v92, v125, s[44:47], null offen offset:64
	buffer_store_b16 v93, v125, s[0:3], null offen offset:64
	buffer_store_b16 v94, v126, s[44:47], null offen offset:64
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v90, v119, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v91, v96, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v92, v120, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v93, v97, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v94, v121, s0
	buffer_store_b16 v90, v126, s[0:3], null offen offset:64
	buffer_store_b16 v91, v109, s[44:47], null offen offset:64
	buffer_store_b16 v92, v109, s[0:3], null offen offset:64
	buffer_store_b16 v93, v127, s[44:47], null offen offset:64
	buffer_store_b16 v94, v127, s[0:3], null offen offset:64
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v90, v98, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v91, v99, s0
	buffer_store_b16 v74, v134, s[44:47], null offen offset:128
	buffer_store_b16 v90, v134, s[0:3], null offen offset:128
	buffer_store_b16 v75, v106, s[44:47], null offen offset:128
	buffer_store_b16 v91, v106, s[0:3], null offen offset:128
	buffer_store_b16 v76, v132, s[44:47], null offen offset:128
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v74, v100, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v75, v77, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v76, v101, s0
	v_cvt_pk_bf16_f32 v77, v78, s0
	v_cvt_pk_bf16_f32 v78, v102, s0
	buffer_store_b16 v74, v132, s[0:3], null offen offset:128
	buffer_store_b16 v75, v108, s[44:47], null offen offset:128
	buffer_store_b16 v76, v108, s[0:3], null offen offset:128
	buffer_store_b16 v77, v125, s[44:47], null offen offset:128
	buffer_store_b16 v78, v125, s[0:3], null offen offset:128
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v74, v79, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v75, v103, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v76, v80, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v77, v104, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v78, v81, s0
	buffer_store_b16 v74, v126, s[44:47], null offen offset:128
	buffer_store_b16 v75, v126, s[0:3], null offen offset:128
	buffer_store_b16 v76, v109, s[44:47], null offen offset:128
	buffer_store_b16 v77, v109, s[0:3], null offen offset:128
	buffer_store_b16 v78, v127, s[44:47], null offen offset:128
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v74, v105, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v75, v82, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v76, v83, s0
	buffer_store_b16 v74, v127, s[0:3], null offen offset:128
	buffer_store_b16 v58, v134, s[44:47], null offen offset:192
	buffer_store_b16 v75, v134, s[0:3], null offen offset:192
	buffer_store_b16 v59, v106, s[44:47], null offen offset:192
	buffer_store_b16 v76, v106, s[0:3], null offen offset:192
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v58, v60, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v59, v84, s0
	v_cvt_pk_bf16_f32 v60, v61, s0
	v_cvt_pk_bf16_f32 v61, v85, s0
	buffer_store_b16 v58, v132, s[44:47], null offen offset:192
	buffer_store_b16 v59, v132, s[0:3], null offen offset:192
	buffer_store_b16 v60, v108, s[44:47], null offen offset:192
	buffer_store_b16 v61, v108, s[0:3], null offen offset:192
	buffer_store_b16 v62, v125, s[44:47], null offen offset:192
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v58, v86, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v59, v63, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v60, v87, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v61, v64, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v62, v88, s0
	buffer_store_b16 v58, v125, s[0:3], null offen offset:192
	buffer_store_b16 v59, v126, s[44:47], null offen offset:192
	buffer_store_b16 v60, v126, s[0:3], null offen offset:192
	buffer_store_b16 v61, v109, s[44:47], null offen offset:192
	buffer_store_b16 v62, v109, s[0:3], null offen offset:192
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v58, v65, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v59, v89, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v60, v66, s0
	buffer_store_b16 v58, v127, s[44:47], null offen offset:192
	buffer_store_b16 v59, v127, s[0:3], null offen offset:192
	buffer_store_b16 v42, v134, s[44:47], null offen offset:256
	buffer_store_b16 v60, v134, s[0:3], null offen offset:256
	buffer_store_b16 v43, v106, s[44:47], null offen offset:256
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v42, v67, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v43, v44, s0
	v_cvt_pk_bf16_f32 v44, v68, s0
	v_cvt_pk_bf16_f32 v58, v69, s0
	buffer_store_b16 v42, v106, s[0:3], null offen offset:256
	buffer_store_b16 v43, v132, s[44:47], null offen offset:256
	buffer_store_b16 v44, v132, s[0:3], null offen offset:256
	buffer_store_b16 v45, v108, s[44:47], null offen offset:256
	buffer_store_b16 v58, v108, s[0:3], null offen offset:256
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v42, v46, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v43, v70, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v44, v47, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v45, v71, s0
	v_cvt_pk_bf16_f32 v46, v48, s0
	buffer_store_b16 v42, v125, s[44:47], null offen offset:256
	buffer_store_b16 v43, v125, s[0:3], null offen offset:256
	buffer_store_b16 v44, v126, s[44:47], null offen offset:256
	buffer_store_b16 v45, v126, s[0:3], null offen offset:256
	buffer_store_b16 v46, v109, s[44:47], null offen offset:256
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v42, v72, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v43, v49, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v44, v73, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v45, v50, s0
	buffer_store_b16 v42, v109, s[0:3], null offen offset:256
	buffer_store_b16 v43, v127, s[44:47], null offen offset:256
	buffer_store_b16 v44, v127, s[0:3], null offen offset:256
	buffer_store_b16 v26, v134, s[44:47], null offen offset:320
	buffer_store_b16 v45, v134, s[0:3], null offen offset:320
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v26, v27, s0
	v_cvt_pk_bf16_f32 v27, v51, s0
	v_cvt_pk_bf16_f32 v42, v52, s0
	buffer_store_b16 v26, v106, s[44:47], null offen offset:320
	buffer_store_b16 v27, v106, s[0:3], null offen offset:320
	buffer_store_b16 v28, v132, s[44:47], null offen offset:320
	buffer_store_b16 v42, v132, s[0:3], null offen offset:320
	buffer_store_b16 v29, v108, s[44:47], null offen offset:320
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v26, v53, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v27, v30, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v28, v54, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v29, v31, s0
	v_cvt_pk_bf16_f32 v30, v55, s0
	buffer_store_b16 v26, v108, s[0:3], null offen offset:320
	buffer_store_b16 v27, v125, s[44:47], null offen offset:320
	buffer_store_b16 v28, v125, s[0:3], null offen offset:320
	buffer_store_b16 v29, v126, s[44:47], null offen offset:320
	buffer_store_b16 v30, v126, s[0:3], null offen offset:320
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v26, v32, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v27, v56, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v28, v33, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v29, v57, s0
	buffer_store_b16 v26, v109, s[44:47], null offen offset:320
	buffer_store_b16 v27, v109, s[0:3], null offen offset:320
	buffer_store_b16 v28, v127, s[44:47], null offen offset:320
	buffer_store_b16 v29, v127, s[0:3], null offen offset:320
	buffer_store_b16 v10, v134, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v10, v34, s0
	v_cvt_pk_bf16_f32 v26, v35, s0
	v_cvt_pk_bf16_f32 v27, v36, s0
	buffer_store_b16 v10, v134, s[0:3], null offen offset:384
	buffer_store_b16 v11, v106, s[44:47], null offen offset:384
	buffer_store_b16 v26, v106, s[0:3], null offen offset:384
	buffer_store_b16 v12, v132, s[44:47], null offen offset:384
	buffer_store_b16 v27, v132, s[0:3], null offen offset:384
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v10, v13, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v11, v37, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v12, v14, s0
	v_cvt_pk_bf16_f32 v13, v38, s0
	v_cvt_pk_bf16_f32 v14, v15, s0
	buffer_store_b16 v10, v108, s[44:47], null offen offset:384
	buffer_store_b16 v11, v108, s[0:3], null offen offset:384
	buffer_store_b16 v12, v125, s[44:47], null offen offset:384
	buffer_store_b16 v13, v125, s[0:3], null offen offset:384
	buffer_store_b16 v14, v126, s[44:47], null offen offset:384
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v10, v39, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v11, v16, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v12, v40, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v13, v17, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v14, v41, s0
	buffer_store_b16 v10, v126, s[0:3], null offen offset:384
	buffer_store_b16 v11, v109, s[44:47], null offen offset:384
	buffer_store_b16 v12, v109, s[0:3], null offen offset:384
	buffer_store_b16 v13, v127, s[44:47], null offen offset:384
	buffer_store_b16 v14, v127, s[0:3], null offen offset:384
	s_wait_xcnt 0x3
	v_or_b32_e32 v11, v131, v0
	v_cvt_pk_bf16_f32 v10, v18, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v12, v19, s0
	buffer_store_b16 v2, v1, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, v122, v0
	v_lshl_or_b32 v11, v11, 2, 0x1c0
	buffer_store_b16 v10, v1, s[0:3], null offen
	buffer_store_b16 v3, v11, s[44:47], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v10, v130, v0
	v_mov_b16_e32 v1.l, v12.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v4, s0
	v_lshl_or_b32 v2, v2, 2, 0x1c0
	v_cvt_pk_bf16_f32 v4, v20, s0
	v_lshl_or_b32 v10, v10, 2, 0x1c0
	v_cvt_pk_bf16_f32 v12, v21, s0
	buffer_store_b16 v1, v11, s[0:3], null offen
	buffer_store_b16 v3, v2, s[44:47], null offen
	buffer_store_b16 v4, v2, s[0:3], null offen
	buffer_store_b16 v5, v10, s[44:47], null offen
	buffer_store_b16 v12, v10, s[0:3], null offen
	s_wait_xcnt 0x4
	v_or_b32_e32 v1, v107, v0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v2, v6, s0
	v_or_b32_e32 v4, v123, v0
	v_cvt_pk_bf16_f32 v3, v22, s0
	v_cvt_pk_bf16_f32 v6, v23, s0
	v_lshl_or_b32 v1, v1, 2, 0x1c0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v5, v7, s0
	v_lshl_or_b32 v4, v4, 2, 0x1c0
	v_cvt_pk_bf16_f32 v7, v25, s0
	buffer_store_b16 v2, v1, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, v124, v0
	v_or_b32_e32 v0, v110, v0
	buffer_store_b16 v3, v1, s[0:3], null offen
	buffer_store_b16 v5, v4, s[44:47], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e32 v1.l, v6.l
	v_cvt_pk_bf16_f32 v3, v8, s0
	v_lshl_or_b32 v2, v2, 2, 0x1c0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v5, v24, s0
	v_cvt_pk_bf16_f32 v6, v9, s0
	v_lshl_or_b32 v0, v0, 2, 0x1c0
	buffer_store_b16 v1, v4, s[0:3], null offen
	buffer_store_b16 v3, v2, s[44:47], null offen
	buffer_store_b16 v5, v2, s[0:3], null offen
	buffer_store_b16 v6, v0, s[44:47], null offen
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
		.amdhsa_next_free_vgpr 634
		.amdhsa_next_free_sgpr 79
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

	.set .Lk_dkdv_0.num_vgpr, 634
	.set .Lk_dkdv_0.num_agpr, 0
	.set .Lk_dkdv_0.numbered_sgpr, 79
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
    .max_flat_workgroup_size: 128
    .name:           k_dkdv_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     81
    .sgpr_spill_count: 0
    .symbol:         k_dkdv_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     634
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
