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
	v_dual_lshrrev_b32 v169 /*v425*/, 4, v0 :: v_dual_bitop2_b32 v165 /*v421*/, 15, v0 bitop3:0x40
	s_cselect_b32 s8, s6, s5
	s_cselect_b32 s4, s3, s4
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_wait_kmcnt 0x0
	s_mul_i32 s65, s38, s8
	s_mul_i32 s2, ttmp9, s2
	s_clause 0x1
	s_load_b64 s[44:45], s[0:1], 0x30 nv
	s_load_b64 s[52:53], s[0:1], 0x0 nv
	s_add_co_i32 s5, s3, s2
	s_cmp_eq_u32 s7, 0
	s_load_b64 s[2:3], s[0:1], 0x190 nv
	s_cselect_b32 s64, ttmp9, s5
	s_lshr_b32 s5, s42, 31
	s_lshl_b32 s9, s4, 5
	s_add_co_i32 s5, s42, s5
	s_set_vgpr_msb 0x4004
	v_or_b32_e32 v1, s9, v165 /*v421*/
	s_and_b32 s4, s5, -2
	s_ashr_i32 s5, s5, 1
	s_cmp_lg_u32 s42, s4
	v_bfe_u32 v4, v0, 3, 1
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s42, 0
	s_mul_i32 s66, s41, s64
	s_cselect_b32 s6, -1, 0
	s_mul_i32 s67, s39, s8
	s_and_b32 s4, s6, s4
	s_sub_co_ci_u32 s6, s5, 0
	s_sub_co_i32 s7, s9, s43
	s_set_vgpr_msb 0x400
	v_lshrrev_b32_e32 v3, 3, v0
	s_max_i32 s7, s7, 0
	s_set_vgpr_msb 64
	v_or_b32_e32 v171 /*v427*/, 16, v0
	s_lshr_b32 s7, s7, 5
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s2, 0
	v_lshlrev_b32_e32 v174 /*v430*/, 4, v4
	s_cselect_b32 s10, -1, 0
	v_lshlrev_b32_e32 v177 /*v433*/, 4, v3
	s_and_b32 s10, s10, exec_lo
	s_cselect_b32 s69, s7, 0
	s_cmp_lg_u32 s4, 0
	s_sub_co_ci_u32 s70, s5, s69
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
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s5, s5, s10
	s_sub_co_ci_u32 s5, s7, 0
	s_cmp_gt_i32 s4, -1
	s_cselect_b32 s4, s5, 0
	s_mul_i32 s5, s38, s40
	s_min_i32 s4, s4, s6
	s_mul_i32 s5, s5, s3
	s_sub_co_i32 s4, s4, s69
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_max_i32 s4, s4, 0
	s_min_i32 s4, s4, s70
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s72, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s2, s72, exec_lo
	s_cselect_b32 s71, s4, 0
	s_or_b32 s14, s9, 16
	s_lshl_b32 s2, s40, 4
	s_set_vgpr_msb 0x4004
	v_or_b32_e32 v2, s14, v165 /*v421*/
	s_mul_i32 s4, s2, s65
	s_lshl_b32 s34, s5, 8
	s_lshl4_add_u32 s4, s64, s4
	s_ashr_i32 s35, s34, 31
	v_mad_u32 v1, v1, s2, s4
	v_mad_u32 v2, v2, s2, s4
	s_clause 0x1
	s_load_b64 s[4:5], s[0:1], 0x60 nv
	s_load_b64 s[48:49], s[0:1], 0x90 nv
	s_lshr_b64 s[46:47], s[34:35], 7
	s_mul_i32 s2, s39, s37
	s_mov_b32 s6, s46
	s_mov_b32 s7, s47
	s_mul_i32 s3, s2, s3
	v_or_b32_e32 v1, v1, v169 /*v425*/
	v_or_b32_e32 v2, v2, v169 /*v425*/
	s_lshl_b32 s10, s3, 8
	s_lshl_b32 s12, s3, 2
	s_lshl_b32 s35, s39, 4
	s_set_vgpr_msb 0x400
	v_dual_lshlrev_b32 v1, 4, v1 :: v_dual_lshlrev_b32 v2, 4, v2
	s_clause 0x10
	buffer_load_b128 v[234:237], v1, s[44:47], null offen
	buffer_load_b128 v[238:241], v1, s[44:47], null offen offset:32
	buffer_load_b128 v[242:245], v1, s[44:47], null offen offset:64
	buffer_load_b128 v[246:249], v1, s[44:47], null offen offset:96
	buffer_load_b128 v[250:253], v1, s[44:47], null offen offset:128
	buffer_load_b128 v[254:257], v1, s[44:47], null offen offset:160
	s_set_vgpr_msb 64
	buffer_load_b128 v[2:5] /*v[258:261]*/, v1, s[44:47], null offen offset:192
	buffer_load_b128 v[6:9] /*v[262:265]*/, v1, s[44:47], null offen offset:224
	buffer_load_b128 v[10:13] /*v[266:269]*/, v2, s[44:47], null offen
	buffer_load_b128 v[14:17] /*v[270:273]*/, v2, s[44:47], null offen offset:32
	buffer_load_b128 v[18:21] /*v[274:277]*/, v2, s[44:47], null offen offset:64
	buffer_load_b128 v[22:25] /*v[278:281]*/, v2, s[44:47], null offen offset:96
	buffer_load_b128 v[26:29] /*v[282:285]*/, v2, s[44:47], null offen offset:128
	buffer_load_b128 v[30:33] /*v[286:289]*/, v2, s[44:47], null offen offset:160
	buffer_load_b128 v[42:45] /*v[298:301]*/, v2, s[44:47], null offen offset:192
	buffer_load_b128 v[46:49] /*v[302:305]*/, v2, s[44:47], null offen offset:224
	s_wait_kmcnt 0x0
	s_clause 0xf
	buffer_load_b128 v[50:53] /*v[306:309]*/, v1, s[4:7], null offen
	buffer_load_b128 v[54:57] /*v[310:313]*/, v1, s[4:7], null offen offset:32
	buffer_load_b128 v[58:61] /*v[314:317]*/, v1, s[4:7], null offen offset:64
	buffer_load_b128 v[62:65] /*v[318:321]*/, v1, s[4:7], null offen offset:96
	buffer_load_b128 v[74:77] /*v[330:333]*/, v1, s[4:7], null offen offset:128
	buffer_load_b128 v[78:81] /*v[334:337]*/, v1, s[4:7], null offen offset:160
	buffer_load_b128 v[82:85] /*v[338:341]*/, v1, s[4:7], null offen offset:192
	buffer_load_b128 v[86:89] /*v[342:345]*/, v1, s[4:7], null offen offset:224
	buffer_load_b128 v[90:93] /*v[346:349]*/, v2, s[4:7], null offen
	buffer_load_b128 v[94:97] /*v[350:353]*/, v2, s[4:7], null offen offset:32
	buffer_load_b128 v[98:101] /*v[354:357]*/, v2, s[4:7], null offen offset:64
	buffer_load_b128 v[102:105] /*v[358:361]*/, v2, s[4:7], null offen offset:96
	buffer_load_b128 v[106:109] /*v[362:365]*/, v2, s[4:7], null offen offset:128
	buffer_load_b128 v[110:113] /*v[366:369]*/, v2, s[4:7], null offen offset:160
	buffer_load_b128 v[114:117] /*v[370:373]*/, v2, s[4:7], null offen offset:192
	buffer_load_b128 v[118:121] /*v[374:377]*/, v2, s[4:7], null offen offset:224
	s_wait_xcnt 0x0
	s_clause 0x1
	s_load_b64 s[4:5], s[0:1], 0xc0 nv
	s_load_b64 s[6:7], s[0:1], 0xe8 nv
	s_set_vgpr_msb 0x4004
	v_lshlrev_b32_e32 v1, 3, v169 /*v425*/
	s_set_vgpr_msb 0x400
	v_and_b32_e32 v2, 16, v0
	s_mov_b32 s2, 0
	s_ashr_i32 s11, s10, 31
	s_ashr_i32 s13, s12, 31
	s_set_vgpr_msb 64
	v_and_or_b32 v176 /*v432*/, v0, 7, v1
	v_or_b32_e32 v170 /*v426*/, s9, v1
	v_or_b32_e32 v168 /*v424*/, s14, v1
	s_lshl_b32 s3, s3, 27
	s_set_vgpr_msb 0x4044
	v_mad_u32_u24 v172 /*v428*/, 0x110, v165 /*v421*/, v2
	v_mad_u32_u24 v173 /*v429*/, 0x110, v171 /*v427*/, v2
	v_or_b32_e32 v163 /*v419*/, 3, v170 /*v426*/
	v_or_b32_e32 v164 /*v420*/, 2, v170 /*v426*/
	v_or_b32_e32 v161 /*v417*/, 5, v170 /*v426*/
	v_or_b32_e32 v162 /*v418*/, 4, v170 /*v426*/
	v_or_b32_e32 v159 /*v415*/, 7, v170 /*v426*/
	v_or_b32_e32 v160 /*v416*/, 6, v170 /*v426*/
	v_or_b32_e32 v157 /*v413*/, 3, v168 /*v424*/
	v_or_b32_e32 v158 /*v414*/, 2, v168 /*v424*/
	v_or_b32_e32 v155 /*v411*/, 5, v168 /*v424*/
	v_or_b32_e32 v156 /*v412*/, 4, v168 /*v424*/
	s_set_vgpr_msb 0x4404
	v_or_b32_e32 v1, 7, v168 /*v424*/
	s_set_vgpr_msb 0x444
	v_or_b32_e32 v154 /*v410*/, 6, v168 /*v424*/
	v_mul_u32_u24_e32 v175 /*v431*/, 0x50, v176 /*v432*/
	s_mul_i32 s38, s71, s41
	s_mul_i32 s68, s37, s35
	s_lshr_b64 s[54:55], s[10:11], 7
	s_lshr_b64 s[58:59], s[12:13], 7
	s_wait_kmcnt 0x0
	s_or_b64 s[56:57], s[4:5], s[2:3]
	s_or_b64 s[60:61], s[6:7], s[2:3]
	s_cmp_lt_i32 s38, 1
	s_mul_i32 s68, s68, s8
	s_set_vgpr_msb 0x4400
	s_cbranch_scc1 .LBB0_3
	s_abs_i32 s73, s41
	s_movk_i32 s2, 0x1400
	s_cvt_f32_u32 s3, s73
	s_set_vgpr_msb 4
	v_mad_u32_u24 v2, 0x110, v176 /*v432*/, s2
	s_set_vgpr_msb 0x440
	v_mov_b32_e32 v34 /*v290*/, 0
	s_movk_i32 s4, 0x3600
	v_s_rcp_f32 s2, s3
	s_set_vgpr_msb 0x4004
	v_mad_u32_u24 v3, 0x110, v176 /*v432*/, s4
	s_set_vgpr_msb 0x444
	v_add_nc_u32_e32 v180 /*v436*/, v2, v174 /*v430*/
	s_movk_i32 s3, 0xa00
	s_set_vgpr_msb 0x4405
	v_or_b32_e32 v6, 0x60, v177 /*v433*/
	v_or_b32_e32 v7, 0x80, v174 /*v430*/
	v_or_b32_e32 v9, 0xa0, v177 /*v433*/
	v_or_b32_e32 v10, 0xc0, v174 /*v430*/
	s_mul_f32 s2, s2, 0x4f7ffffe
	v_mad_u32_u24 v11, 0x50, v176 /*v432*/, s3
	v_mov_b32_e32 v220, v34 /*v290*/
	s_set_vgpr_msb 0x554
	v_mad_i32_i24 v178 /*v434*/, 0xffffff40, v165 /*v421*/, v172 /*v428*/
	s_cvt_u32_f32 s4, s2
	s_sub_co_i32 s3, 0, s73
	s_mov_b32 s2, s36
	v_mad_i32_i24 v179 /*v435*/, 0xffffff40, v171 /*v427*/, v173 /*v429*/
	s_mul_i32 s5, s3, s4
	v_add_nc_u32_e32 v181 /*v437*/, v3, v174 /*v430*/
	s_set_vgpr_msb 0x5440
	v_add_nc_u32_e32 v187 /*v443*/, v3, v6
	s_mov_b32 s3, s36
	v_dual_add_nc_u32 v188 /*v444*/, v2, v7 :: v_dual_add_nc_u32 v189 /*v445*/, v3, v7
	v_mov_b64_e32 v[166:167] /*v[422:423]*/, s[2:3]
	v_dual_add_nc_u32 v186 /*v442*/, v2, v6 :: v_dual_add_nc_u32 v190 /*v446*/, v2, v9
	v_add_nc_u32_e32 v191 /*v447*/, v3, v9
	s_set_vgpr_msb 0x4005
	v_dual_mov_b32 v221, v34 /*v290*/ :: v_dual_bitop2_b32 v12, 32, v174 /*v430*/ bitop3:0x54
	s_set_vgpr_msb 0x500
	v_lshlrev_b32_e32 v8, 1, v0
	s_set_vgpr_msb 64
	v_dual_add_nc_u32 v192 /*v448*/, v2, v10 :: v_dual_add_nc_u32 v193 /*v449*/, v3, v10
	s_set_vgpr_msb 0x4005
	v_dual_mov_b32 v219, v34 /*v290*/ :: v_dual_bitop2_b32 v5, 64, v174 /*v430*/ bitop3:0x54
	s_set_vgpr_msb 0x500
	v_and_or_b32 v8, v8, 16, 0xe0
	s_set_vgpr_msb 0x45
	v_add_nc_u32_e32 v196 /*v452*/, v175 /*v431*/, v174 /*v430*/
	s_set_vgpr_msb 0x4505
	v_dual_mov_b32 v218, v34 /*v290*/ :: v_dual_bitop2_b32 v4, 32, v177 /*v433*/ bitop3:0x54
	s_set_vgpr_msb 0x540
	v_dual_add_nc_u32 v185 /*v441*/, v3, v5 :: v_dual_add_nc_u32 v184 /*v440*/, v2, v5
	v_dual_add_nc_u32 v194 /*v450*/, v2, v8 :: v_dual_add_nc_u32 v195 /*v451*/, v3, v8
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_nc_u32 v182 /*v438*/, v2, v4 :: v_dual_add_nc_u32 v183 /*v439*/, v3, v4
	s_set_vgpr_msb 0x4044
	v_add_nc_u32_e32 v197 /*v453*/, v11, v174 /*v430*/
	v_add_nc_u32_e32 v198 /*v454*/, v12, v175 /*v431*/
	s_set_vgpr_msb 0x4440
	v_add_nc_u32_e32 v199 /*v455*/, v11, v12
	s_set_vgpr_msb 0x4041
	v_dual_mov_b32 v35 /*v291*/, v34 /*v290*/ :: v_dual_mov_b32 v36 /*v292*/, v34 /*v290*/
	v_dual_mov_b32 v37 /*v293*/, v34 /*v290*/ :: v_dual_mov_b32 v38 /*v294*/, v34 /*v290*/
	v_dual_mov_b32 v39 /*v295*/, v34 /*v290*/ :: v_dual_mov_b32 v40 /*v296*/, v34 /*v290*/
	v_dual_mov_b32 v41 /*v297*/, v34 /*v290*/ :: v_dual_mov_b32 v122 /*v378*/, v34 /*v290*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v222, v34 /*v290*/ :: v_dual_mov_b32 v223, v34 /*v290*/
	v_dual_mov_b32 v224, v34 /*v290*/ :: v_dual_mov_b32 v225, v34 /*v290*/
	v_dual_mov_b32 v202, v34 /*v290*/ :: v_dual_mov_b32 v203, v34 /*v290*/
	v_dual_mov_b32 v204, v34 /*v290*/ :: v_dual_mov_b32 v205, v34 /*v290*/
	v_dual_mov_b32 v206, v34 /*v290*/ :: v_dual_mov_b32 v207, v34 /*v290*/
	v_dual_mov_b32 v208, v34 /*v290*/ :: v_dual_mov_b32 v209, v34 /*v290*/
	v_dual_mov_b32 v186, v34 /*v290*/ :: v_dual_mov_b32 v187, v34 /*v290*/
	v_dual_mov_b32 v188, v34 /*v290*/ :: v_dual_mov_b32 v189, v34 /*v290*/
	v_dual_mov_b32 v190, v34 /*v290*/ :: v_dual_mov_b32 v191, v34 /*v290*/
	v_dual_mov_b32 v192, v34 /*v290*/ :: v_dual_mov_b32 v193, v34 /*v290*/
	v_dual_mov_b32 v170, v34 /*v290*/ :: v_dual_mov_b32 v171, v34 /*v290*/
	v_dual_mov_b32 v172, v34 /*v290*/ :: v_dual_mov_b32 v173, v34 /*v290*/
	v_dual_mov_b32 v174, v34 /*v290*/ :: v_dual_mov_b32 v175, v34 /*v290*/
	v_dual_mov_b32 v176, v34 /*v290*/ :: v_dual_mov_b32 v177, v34 /*v290*/
	v_dual_mov_b32 v154, v34 /*v290*/ :: v_dual_mov_b32 v155, v34 /*v290*/
	v_dual_mov_b32 v156, v34 /*v290*/ :: v_dual_mov_b32 v157, v34 /*v290*/
	v_dual_mov_b32 v158, v34 /*v290*/ :: v_dual_mov_b32 v159, v34 /*v290*/
	v_dual_mov_b32 v160, v34 /*v290*/ :: v_dual_mov_b32 v161, v34 /*v290*/
	v_dual_mov_b32 v138, v34 /*v290*/ :: v_dual_mov_b32 v139, v34 /*v290*/
	v_dual_mov_b32 v140, v34 /*v290*/ :: v_dual_mov_b32 v141, v34 /*v290*/
	v_dual_mov_b32 v142, v34 /*v290*/ :: v_dual_mov_b32 v143, v34 /*v290*/
	v_dual_mov_b32 v144, v34 /*v290*/ :: v_dual_mov_b32 v145, v34 /*v290*/
	v_dual_mov_b32 v130, v34 /*v290*/ :: v_dual_mov_b32 v131, v34 /*v290*/
	v_dual_mov_b32 v132, v34 /*v290*/ :: v_dual_mov_b32 v133, v34 /*v290*/
	v_dual_mov_b32 v134, v34 /*v290*/ :: v_dual_mov_b32 v135, v34 /*v290*/
	v_dual_mov_b32 v136, v34 /*v290*/ :: v_dual_mov_b32 v137, v34 /*v290*/
	v_dual_mov_b32 v114, v34 /*v290*/ :: v_dual_mov_b32 v115, v34 /*v290*/
	v_dual_mov_b32 v116, v34 /*v290*/ :: v_dual_mov_b32 v117, v34 /*v290*/
	v_dual_mov_b32 v118, v34 /*v290*/ :: v_dual_mov_b32 v119, v34 /*v290*/
	v_dual_mov_b32 v120, v34 /*v290*/ :: v_dual_mov_b32 v121, v34 /*v290*/
	v_dual_mov_b32 v90, v34 /*v290*/ :: v_dual_mov_b32 v91, v34 /*v290*/
	v_dual_mov_b32 v92, v34 /*v290*/ :: v_dual_mov_b32 v93, v34 /*v290*/
	v_dual_mov_b32 v94, v34 /*v290*/ :: v_dual_mov_b32 v95, v34 /*v290*/
	v_dual_mov_b32 v96, v34 /*v290*/ :: v_dual_mov_b32 v97, v34 /*v290*/
	v_dual_mov_b32 v74, v34 /*v290*/ :: v_dual_mov_b32 v75, v34 /*v290*/
	v_dual_mov_b32 v76, v34 /*v290*/ :: v_dual_mov_b32 v77, v34 /*v290*/
	v_dual_mov_b32 v78, v34 /*v290*/ :: v_dual_mov_b32 v79, v34 /*v290*/
	v_dual_mov_b32 v80, v34 /*v290*/ :: v_dual_mov_b32 v81, v34 /*v290*/
	v_dual_mov_b32 v58, v34 /*v290*/ :: v_dual_mov_b32 v59, v34 /*v290*/
	v_dual_mov_b32 v60, v34 /*v290*/ :: v_dual_mov_b32 v61, v34 /*v290*/
	v_dual_mov_b32 v62, v34 /*v290*/ :: v_dual_mov_b32 v63, v34 /*v290*/
	v_dual_mov_b32 v64, v34 /*v290*/ :: v_dual_mov_b32 v65, v34 /*v290*/
	v_dual_mov_b32 v42, v34 /*v290*/ :: v_dual_mov_b32 v43, v34 /*v290*/
	v_dual_mov_b32 v44, v34 /*v290*/ :: v_dual_mov_b32 v45, v34 /*v290*/
	v_dual_mov_b32 v46, v34 /*v290*/ :: v_dual_mov_b32 v47, v34 /*v290*/
	v_dual_mov_b32 v48, v34 /*v290*/ :: v_dual_mov_b32 v49, v34 /*v290*/
	v_dual_mov_b32 v26, v34 /*v290*/ :: v_dual_mov_b32 v27, v34 /*v290*/
	v_dual_mov_b32 v28, v34 /*v290*/ :: v_dual_mov_b32 v29, v34 /*v290*/
	v_dual_mov_b32 v30, v34 /*v290*/ :: v_dual_mov_b32 v31, v34 /*v290*/
	v_dual_mov_b32 v32, v34 /*v290*/ :: v_dual_mov_b32 v33, v34 /*v290*/
	v_dual_mov_b32 v10, v34 /*v290*/ :: v_dual_mov_b32 v11, v34 /*v290*/
	v_dual_mov_b32 v12, v34 /*v290*/ :: v_dual_mov_b32 v13, v34 /*v290*/
	v_dual_mov_b32 v14, v34 /*v290*/ :: v_dual_mov_b32 v15, v34 /*v290*/
	v_dual_mov_b32 v16, v34 /*v290*/ :: v_dual_mov_b32 v17, v34 /*v290*/
	v_dual_mov_b32 v2, v34 /*v290*/ :: v_dual_mov_b32 v3, v34 /*v290*/
	v_dual_mov_b32 v4, v34 /*v290*/ :: v_dual_mov_b32 v5, v34 /*v290*/
	v_dual_mov_b32 v6, v34 /*v290*/ :: v_dual_mov_b32 v7, v34 /*v290*/
	v_dual_mov_b32 v8, v34 /*v290*/ :: v_dual_mov_b32 v9, v34 /*v290*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v123 /*v379*/, v34 /*v290*/ :: v_dual_mov_b32 v124 /*v380*/, v34 /*v290*/
	v_dual_mov_b32 v125 /*v381*/, v34 /*v290*/ :: v_dual_mov_b32 v126 /*v382*/, v34 /*v290*/
	v_dual_mov_b32 v127 /*v383*/, v34 /*v290*/ :: v_dual_mov_b32 v128 /*v384*/, v34 /*v290*/
	v_dual_mov_b32 v129 /*v385*/, v34 /*v290*/ :: v_dual_mov_b32 v66 /*v322*/, v34 /*v290*/
	v_dual_mov_b32 v67 /*v323*/, v34 /*v290*/ :: v_dual_mov_b32 v68 /*v324*/, v34 /*v290*/
	v_dual_mov_b32 v69 /*v325*/, v34 /*v290*/ :: v_dual_mov_b32 v70 /*v326*/, v34 /*v290*/
	v_dual_mov_b32 v71 /*v327*/, v34 /*v290*/ :: v_dual_mov_b32 v72 /*v328*/, v34 /*v290*/
	v_mov_b32_e32 v73 /*v329*/, v34 /*v290*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v226, v34 /*v290*/ :: v_dual_mov_b32 v227, v34 /*v290*/
	v_dual_mov_b32 v228, v34 /*v290*/ :: v_dual_mov_b32 v229, v34 /*v290*/
	v_dual_mov_b32 v230, v34 /*v290*/ :: v_dual_mov_b32 v231, v34 /*v290*/
	v_dual_mov_b32 v232, v34 /*v290*/ :: v_dual_mov_b32 v233, v34 /*v290*/
	v_dual_mov_b32 v210, v34 /*v290*/ :: v_dual_mov_b32 v211, v34 /*v290*/
	v_dual_mov_b32 v212, v34 /*v290*/ :: v_dual_mov_b32 v213, v34 /*v290*/
	v_dual_mov_b32 v214, v34 /*v290*/ :: v_dual_mov_b32 v215, v34 /*v290*/
	v_dual_mov_b32 v216, v34 /*v290*/ :: v_dual_mov_b32 v217, v34 /*v290*/
	v_dual_mov_b32 v194, v34 /*v290*/ :: v_dual_mov_b32 v195, v34 /*v290*/
	v_dual_mov_b32 v196, v34 /*v290*/ :: v_dual_mov_b32 v197, v34 /*v290*/
	v_dual_mov_b32 v198, v34 /*v290*/ :: v_dual_mov_b32 v199, v34 /*v290*/
	v_dual_mov_b32 v200, v34 /*v290*/ :: v_dual_mov_b32 v201, v34 /*v290*/
	v_dual_mov_b32 v178, v34 /*v290*/ :: v_dual_mov_b32 v179, v34 /*v290*/
	v_dual_mov_b32 v180, v34 /*v290*/ :: v_dual_mov_b32 v181, v34 /*v290*/
	v_dual_mov_b32 v182, v34 /*v290*/ :: v_dual_mov_b32 v183, v34 /*v290*/
	v_dual_mov_b32 v184, v34 /*v290*/ :: v_dual_mov_b32 v185, v34 /*v290*/
	v_dual_mov_b32 v162, v34 /*v290*/ :: v_dual_mov_b32 v163, v34 /*v290*/
	v_dual_mov_b32 v164, v34 /*v290*/ :: v_dual_mov_b32 v165, v34 /*v290*/
	v_dual_mov_b32 v166, v34 /*v290*/ :: v_dual_mov_b32 v167, v34 /*v290*/
	v_dual_mov_b32 v168, v34 /*v290*/ :: v_dual_mov_b32 v169, v34 /*v290*/
	v_dual_mov_b32 v146, v34 /*v290*/ :: v_dual_mov_b32 v147, v34 /*v290*/
	v_dual_mov_b32 v148, v34 /*v290*/ :: v_dual_mov_b32 v149, v34 /*v290*/
	v_dual_mov_b32 v150, v34 /*v290*/ :: v_dual_mov_b32 v151, v34 /*v290*/
	v_dual_mov_b32 v152, v34 /*v290*/ :: v_dual_mov_b32 v153, v34 /*v290*/
	v_dual_mov_b32 v122, v34 /*v290*/ :: v_dual_mov_b32 v123, v34 /*v290*/
	v_dual_mov_b32 v124, v34 /*v290*/ :: v_dual_mov_b32 v125, v34 /*v290*/
	v_dual_mov_b32 v126, v34 /*v290*/ :: v_dual_mov_b32 v127, v34 /*v290*/
	v_dual_mov_b32 v128, v34 /*v290*/ :: v_dual_mov_b32 v129, v34 /*v290*/
	v_dual_mov_b32 v106, v34 /*v290*/ :: v_dual_mov_b32 v107, v34 /*v290*/
	v_dual_mov_b32 v108, v34 /*v290*/ :: v_dual_mov_b32 v109, v34 /*v290*/
	v_dual_mov_b32 v110, v34 /*v290*/ :: v_dual_mov_b32 v111, v34 /*v290*/
	v_dual_mov_b32 v112, v34 /*v290*/ :: v_dual_mov_b32 v113, v34 /*v290*/
	v_dual_mov_b32 v98, v34 /*v290*/ :: v_dual_mov_b32 v99, v34 /*v290*/
	v_dual_mov_b32 v100, v34 /*v290*/ :: v_dual_mov_b32 v101, v34 /*v290*/
	v_dual_mov_b32 v102, v34 /*v290*/ :: v_dual_mov_b32 v103, v34 /*v290*/
	v_dual_mov_b32 v104, v34 /*v290*/ :: v_dual_mov_b32 v105, v34 /*v290*/
	v_dual_mov_b32 v82, v34 /*v290*/ :: v_dual_mov_b32 v83, v34 /*v290*/
	v_dual_mov_b32 v84, v34 /*v290*/ :: v_dual_mov_b32 v85, v34 /*v290*/
	v_dual_mov_b32 v86, v34 /*v290*/ :: v_dual_mov_b32 v87, v34 /*v290*/
	v_dual_mov_b32 v88, v34 /*v290*/ :: v_dual_mov_b32 v89, v34 /*v290*/
	v_dual_mov_b32 v66, v34 /*v290*/ :: v_dual_mov_b32 v67, v34 /*v290*/
	v_dual_mov_b32 v68, v34 /*v290*/ :: v_dual_mov_b32 v69, v34 /*v290*/
	v_dual_mov_b32 v70, v34 /*v290*/ :: v_dual_mov_b32 v71, v34 /*v290*/
	v_dual_mov_b32 v72, v34 /*v290*/ :: v_dual_mov_b32 v73, v34 /*v290*/
	v_dual_mov_b32 v50, v34 /*v290*/ :: v_dual_mov_b32 v51, v34 /*v290*/
	v_dual_mov_b32 v52, v34 /*v290*/ :: v_dual_mov_b32 v53, v34 /*v290*/
	v_dual_mov_b32 v54, v34 /*v290*/ :: v_dual_mov_b32 v55, v34 /*v290*/
	v_dual_mov_b32 v56, v34 /*v290*/ :: v_dual_mov_b32 v57, v34 /*v290*/
	v_dual_mov_b32 v34, v34 /*v290*/ :: v_dual_mov_b32 v35, v34 /*v290*/
	v_dual_mov_b32 v36, v34 /*v290*/ :: v_dual_mov_b32 v37, v34 /*v290*/
	v_dual_mov_b32 v38, v34 /*v290*/ :: v_dual_mov_b32 v39, v34 /*v290*/
	v_dual_mov_b32 v40, v34 /*v290*/ :: v_dual_mov_b32 v41, v34 /*v290*/
	v_dual_mov_b32 v18, v34 /*v290*/ :: v_dual_mov_b32 v19, v34 /*v290*/
	v_dual_mov_b32 v20, v34 /*v290*/ :: v_dual_mov_b32 v21, v34 /*v290*/
	v_dual_mov_b32 v22, v34 /*v290*/ :: v_dual_mov_b32 v23, v34 /*v290*/
	v_dual_mov_b32 v24, v34 /*v290*/ :: v_dual_mov_b32 v25, v34 /*v290*/
	s_mul_hi_u32 s5, s4, s5
	s_ashr_i32 s39, s38, 31
	s_ashr_i32 s74, s41, 31
	s_add_co_i32 s75, s4, s5
	s_mov_b64 s[44:45], 0
	s_mov_b32 s42, 0x3fb8aa3b
	s_mov_b32 s50, s54
	s_mov_b32 s51, s55
	s_set_vgpr_msb 0x100
.LBB0_2:
	s_abs_i32 s2, s44
	s_ashr_i32 s3, s44, 31
	s_mul_hi_u32 s4, s2, s75
	s_xor_b32 s3, s3, s74
	s_mul_i32 s5, s4, s73
	s_add_co_i32 s6, s4, 1
	s_sub_co_i32 s2, s2, s5
	s_mov_b32 s62, s58
	s_sub_co_i32 s5, s2, s73
	s_cmp_ge_u32 s2, s73
	s_mov_b32 s63, s59
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s2, s5, s2
	s_add_co_i32 s5, s4, 1
	s_cmp_ge_u32 s2, s73
	s_cselect_b32 s2, s5, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s2, s2, s3
	s_sub_co_i32 s4, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s4, s4, s41
	s_cmp_lg_u32 s44, s4
	s_cselect_b32 s4, -1, 0
	s_xor_b32 s5, s41, s44
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s5, 0
	s_cselect_b32 s5, -1, 0
	s_and_b32 s4, s5, s4
	s_sub_co_ci_u32 s2, s2, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s3, s2, s69
	s_mul_i32 s2, s2, s41
	s_lshl_b32 s3, s3, 5
	s_sub_co_i32 s2, s44, s2
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x44
	v_or_b32_e32 v240 /*v496*/, s3, v165 /*v421*/
	s_add_co_i32 s2, s2, s66
	v_or_b32_e32 v241 /*v497*/, s3, v171 /*v427*/
	s_lshl4_add_u32 s3, s2, s68
	s_add_co_i32 s2, s2, s67
	v_mad_u32 v130 /*v386*/, s35, v240 /*v496*/, s3
	s_set_vgpr_msb 0x4484
	v_dual_add_nc_u32 v56 /*v568*/, s43, v240 /*v496*/ :: v_dual_add_nc_u32 v104 /*v616*/, s43, v241 /*v497*/
	s_set_vgpr_msb 0x8441
	v_mad_u32 v131 /*v387*/, v241 /*v497*/, s35, s3
	s_mul_i32 s2, s2, s37
	s_add_nc_u64 s[44:45], s[44:45], 1
	s_set_vgpr_msb 0x4184
	v_add_lshl_u32 v105 /*v617*/, s2, v240 /*v496*/, 2
	v_add_lshl_u32 v106 /*v618*/, s2, v241 /*v497*/, 2
	s_set_vgpr_msb 0x8445
	v_or_b32_e32 v130 /*v386*/, v130 /*v386*/, v169 /*v425*/
	s_set_vgpr_msb 0x4509
	v_cmp_ge_i32_e32 vcc_lo, v170 /*v426*/, v56 /*v568*/
	v_cmp_gt_i32_e64 s2, v170 /*v426*/, v56 /*v568*/
	s_set_vgpr_msb 0x945
	v_or_b32_e32 v131 /*v387*/, v131 /*v387*/, v169 /*v425*/
	s_set_vgpr_msb 0x4509
	v_cmp_gt_i32_e64 s3, v163 /*v419*/, v56 /*v568*/
	s_set_vgpr_msb 0x984
	v_lshlrev_b32_e32 v80 /*v592*/, 4, v130 /*v386*/
	s_set_vgpr_msb 0x8409
	v_cmp_gt_i32_e64 s4, v164 /*v420*/, v56 /*v568*/
	v_cmp_gt_i32_e64 s5, v161 /*v417*/, v56 /*v568*/
	s_set_vgpr_msb 0x984
	v_lshlrev_b32_e32 v88 /*v600*/, 4, v131 /*v387*/
	s_set_vgpr_msb 0x8449
	v_cmp_gt_i32_e64 s6, v162 /*v418*/, v56 /*v568*/
	v_or_b32_e32 v220 /*v476*/, 32, v80 /*v592*/
	s_set_vgpr_msb 0x4988
	v_or_b32_e32 v32 /*v544*/, 64, v80 /*v592*/
	v_or_b32_e32 v36 /*v548*/, 0x60, v80 /*v592*/
	s_set_vgpr_msb 0x884a
	v_or_b32_e32 v244 /*v500*/, 32, v88 /*v600*/
	s_clause 0x4
	buffer_load_b128 v[138:141] /*v[394:397]*/, v80 /*v592*/, s[52:55], null offen
	s_set_vgpr_msb 0x4a41
	buffer_load_b128 v[142:145] /*v[398:401]*/, v220 /*v476*/, s[52:55], null offen
	s_set_vgpr_msb 0x4142
	buffer_load_b128 v[130:133] /*v[386:389]*/, v88 /*v600*/, s[52:55], null offen
	s_set_vgpr_msb 0x4288
	v_or_b32_e32 v40 /*v552*/, 64, v88 /*v600*/
	v_or_b32_e32 v44 /*v556*/, 0x60, v88 /*v600*/
	s_set_vgpr_msb 0x8841
	s_clause 0x4
	buffer_load_b128 v[134:137] /*v[390:393]*/, v244 /*v500*/, s[52:55], null offen
	s_set_vgpr_msb 0x418a
	buffer_load_b128 v[8:11] /*v[520:523]*/, v32 /*v544*/, s[52:55], null offen
	buffer_load_b128 v[12:15] /*v[524:527]*/, v36 /*v548*/, s[52:55], null offen
	buffer_load_b128 v[24:27] /*v[536:539]*/, v40 /*v552*/, s[52:55], null offen
	v_or_b32_e32 v64 /*v576*/, 0x80, v80 /*v592*/
	v_or_b32_e32 v68 /*v580*/, 0xa0, v80 /*v592*/
	buffer_load_b128 v[28:31] /*v[540:543]*/, v44 /*v556*/, s[52:55], null offen
	v_or_b32_e32 v72 /*v584*/, 0x80, v88 /*v600*/
	v_or_b32_e32 v76 /*v588*/, 0xa0, v88 /*v600*/
	s_clause 0x1
	buffer_load_b128 v[48:51] /*v[560:563]*/, v64 /*v576*/, s[52:55], null offen
	buffer_load_b128 v[52:55] /*v[564:567]*/, v68 /*v580*/, s[52:55], null offen
	v_or_b32_e32 v96 /*v608*/, 0xc0, v80 /*v592*/
	v_or_b32_e32 v100 /*v612*/, 0xe0, v80 /*v592*/
	s_set_vgpr_msb 0x8a42
	s_clause 0x4
	buffer_load_b128 v[216:219] /*v[472:475]*/, v80 /*v592*/, s[48:51], null offen
	buffer_load_b128 v[240:243] /*v[496:499]*/, v88 /*v600*/, s[48:51], null offen
	s_set_vgpr_msb 0x4249
	buffer_load_b128 v[220:223] /*v[476:479]*/, v220 /*v476*/, s[48:51], null offen
	buffer_load_b128 v[244:247] /*v[500:503]*/, v244 /*v500*/, s[48:51], null offen
	v_cmp_gt_i32_e64 s7, v159 /*v415*/, v56 /*v568*/
	v_cmp_gt_i32_e64 s8, v160 /*v416*/, v56 /*v568*/
	v_cmp_ge_i32_e64 s9, v168 /*v424*/, v56 /*v568*/
	v_cmp_gt_i32_e64 s10, v168 /*v424*/, v56 /*v568*/
	v_cmp_gt_i32_e64 s11, v157 /*v413*/, v56 /*v568*/
	v_cmp_gt_i32_e64 s12, v158 /*v414*/, v56 /*v568*/
	v_cmp_gt_i32_e64 s13, v155 /*v411*/, v56 /*v568*/
	v_cmp_gt_i32_e64 s14, v156 /*v412*/, v56 /*v568*/
	s_set_vgpr_msb 0x4908
	v_cmp_gt_i32_e64 s15, v1, v56 /*v568*/
	s_set_vgpr_msb 0x809
	v_cmp_gt_i32_e64 s16, v154 /*v410*/, v56 /*v568*/
	s_set_vgpr_msb 0x98a
	s_clause 0x1
	buffer_load_b128 v[56:59] /*v[568:571]*/, v72 /*v584*/, s[52:55], null offen
	buffer_load_b128 v[60:63] /*v[572:575]*/, v76 /*v588*/, s[52:55], null offen
	v_or_b32_e32 v107 /*v619*/, 0xc0, v88 /*v600*/
	s_clause 0x1
	buffer_load_b128 v[80:83] /*v[592:595]*/, v96 /*v608*/, s[52:55], null offen
	buffer_load_b128 v[84:87] /*v[596:599]*/, v100 /*v612*/, s[52:55], null offen
	v_or_b32_e32 v108 /*v620*/, 0xe0, v88 /*v600*/
	s_clause 0x3
	buffer_load_b128 v[32:35] /*v[544:547]*/, v32 /*v544*/, s[48:51], null offen
	buffer_load_b128 v[36:39] /*v[548:551]*/, v36 /*v548*/, s[48:51], null offen
	buffer_load_b128 v[40:43] /*v[552:555]*/, v40 /*v552*/, s[48:51], null offen
	buffer_load_b128 v[44:47] /*v[556:559]*/, v44 /*v556*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[88:91] /*v[600:603]*/, v107 /*v619*/, s[52:55], null offen
	buffer_load_b128 v[92:95] /*v[604:607]*/, v108 /*v620*/, s[52:55], null offen
	s_clause 0x4
	buffer_load_b128 v[96:99] /*v[608:611]*/, v96 /*v608*/, s[48:51], null offen
	buffer_load_b128 v[64:67] /*v[576:579]*/, v64 /*v576*/, s[48:51], null offen
	buffer_load_b128 v[68:71] /*v[580:583]*/, v68 /*v580*/, s[48:51], null offen
	buffer_load_b128 v[72:75] /*v[584:587]*/, v72 /*v584*/, s[48:51], null offen
	buffer_load_b128 v[76:79] /*v[588:591]*/, v76 /*v588*/, s[48:51], null offen
	s_set_vgpr_msb 0x8a09
	v_cmp_ge_i32_e64 s17, v170 /*v426*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s18, v170 /*v426*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s19, v163 /*v419*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s20, v164 /*v420*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s21, v161 /*v417*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s22, v162 /*v418*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s23, v159 /*v415*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s24, v160 /*v416*/, v104 /*v616*/
	v_cmp_ge_i32_e64 s25, v168 /*v424*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s26, v168 /*v424*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s27, v157 /*v413*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s28, v158 /*v414*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s29, v155 /*v411*/, v104 /*v616*/
	v_cmp_gt_i32_e64 s30, v156 /*v412*/, v104 /*v616*/
	s_set_vgpr_msb 0x982
	s_clause 0x1
	buffer_load_b128 v[100:103] /*v[612:615]*/, v100 /*v612*/, s[48:51], null offen
	buffer_load_b128 v[108:111] /*v[620:623]*/, v108 /*v620*/, s[48:51], null offen
	v_cmp_lt_i32_e64 s31, v104 /*v616*/, v1
	s_set_vgpr_msb 0x8209
	v_cmp_gt_i32_e64 s33, v154 /*v410*/, v104 /*v616*/
	s_set_vgpr_msb 0x982
	s_clause 0x1
	buffer_load_b32 v112 /*v624*/, v105 /*v617*/, s[60:63], null offen
	buffer_load_b32 v114 /*v626*/, v106 /*v618*/, s[60:63], null offen
	s_clause 0x1
	buffer_load_b32 v116 /*v628*/, v105 /*v617*/, s[56:59], null offen
	buffer_load_b32 v118 /*v630*/, v106 /*v618*/, s[56:59], null offen
	buffer_load_b128 v[104:107] /*v[616:619]*/, v107 /*v619*/, s[48:51], null offen
	s_and_b32 s2, s72, s2
	s_and_b32 s4, s72, s4
	s_and_b32 s3, s72, s3
	s_and_b32 s6, s72, s6
	s_and_b32 s5, s72, s5
	s_and_b32 s8, s72, s8
	s_and_b32 s7, s72, s7
	s_and_b32 s62, s72, vcc_lo
	s_and_b32 s9, s72, s9
	s_and_b32 s10, s72, s10
	s_and_b32 s12, s72, s12
	s_and_b32 s11, s72, s11
	s_and_b32 s14, s72, s14
	s_and_b32 s13, s72, s13
	s_and_b32 s16, s72, s16
	s_and_b32 s15, s72, s15
	s_and_b32 s17, s72, s17
	s_and_b32 s18, s72, s18
	s_and_b32 s19, s72, s19
	s_and_b32 s20, s72, s20
	s_and_b32 s21, s72, s21
	s_and_b32 s22, s72, s22
	s_and_b32 s23, s72, s23
	s_and_b32 s24, s72, s24
	s_and_b32 s25, s72, s25
	s_and_b32 s26, s72, s26
	s_and_b32 s27, s72, s27
	s_and_b32 s28, s72, s28
	s_and_b32 s29, s72, s29
	s_and_b32 s30, s72, s30
	s_and_b32 s31, s72, s31
	s_and_b32 s33, s72, s33
	s_cmp_lg_u64 s[44:45], s[38:39]
	s_set_vgpr_msb 0x8205
	s_wait_loadcnt 0x23
	ds_store_b128 v172 /*v428*/, v[138:141] /*v[394:397]*/ offset:13824
	s_wait_loadcnt 0x19
	ds_store_b128 v172 /*v428*/, v[216:219] /*v[472:475]*/ offset:5120
	s_set_vgpr_msb 0x544
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[234:241], v[138:145] /*v[394:401]*/, 0
	s_set_vgpr_msb 0x4405
	ds_store_b128 v172 /*v428*/, v[142:145] /*v[398:401]*/ offset:13856
	s_set_vgpr_msb 0x509
	ds_store_b128 v172 /*v428*/, v[8:11] /*v[520:523]*/ offset:13888
	ds_store_b128 v172 /*v428*/, v[12:15] /*v[524:527]*/ offset:13920
	ds_store_b128 v172 /*v428*/, v[48:51] /*v[560:563]*/ offset:13952
	ds_store_b128 v172 /*v428*/, v[52:55] /*v[564:567]*/ offset:13984
	s_set_vgpr_msb 0x905
	s_wait_loadcnt 0x17
	ds_store_b128 v172 /*v428*/, v[220:223] /*v[476:479]*/ offset:5152
	s_set_vgpr_msb 0x509
	s_wait_loadcnt 0x11
	ds_store_b128 v172 /*v428*/, v[32:35] /*v[544:547]*/ offset:5184
	s_set_vgpr_msb 0x945
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[10:17] /*v[266:273]*/, v[138:145] /*v[394:401]*/, 0
	s_set_vgpr_msb 0x4509
	s_wait_loadcnt 0x10
	ds_store_b128 v172 /*v428*/, v[36:39] /*v[548:551]*/ offset:5216
	s_wait_loadcnt 0xa
	ds_store_b128 v172 /*v428*/, v[64:67] /*v[576:579]*/ offset:5248
	s_wait_loadcnt 0x9
	ds_store_b128 v172 /*v428*/, v[68:71] /*v[580:583]*/ offset:5280
	ds_store_b128 v172 /*v428*/, v[80:83] /*v[592:595]*/ offset:14016
	ds_store_b128 v172 /*v428*/, v[84:87] /*v[596:599]*/ offset:14048
	ds_store_b128 v172 /*v428*/, v[96:99] /*v[608:611]*/ offset:5312
	s_wait_loadcnt 0x6
	ds_store_b128 v172 /*v428*/, v[100:103] /*v[612:615]*/ offset:5344
	s_set_vgpr_msb 0x944
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[234:241], v[130:137] /*v[386:393]*/, 0
	s_set_vgpr_msb 0x4445
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[10:17] /*v[266:273]*/, v[130:137] /*v[386:393]*/, 0
	s_set_vgpr_msb 0x4558
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[242:249], v[8:15] /*v[520:527]*/, v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[18:25] /*v[274:281]*/, v[8:15] /*v[520:527]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5958
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[242:249], v[24:31] /*v[536:543]*/, v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[18:25] /*v[274:281]*/, v[24:31] /*v[536:543]*/, v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5958
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[250:257], v[48:55] /*v[560:567]*/, v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[26:33] /*v[282:289]*/, v[48:55] /*v[560:567]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5945
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[50:57] /*v[306:313]*/, v[216:223] /*v[472:479]*/, 0
	s_set_vgpr_msb 0x4585
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[512:519]*/, v[50:57] /*v[306:313]*/, v[240:247] /*v[496:503]*/, 0
	s_set_vgpr_msb 0x8558
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[250:257], v[56:63] /*v[568:575]*/, v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[26:33] /*v[282:289]*/, v[56:63] /*v[568:575]*/, v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5945
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[90:97] /*v[346:353]*/, v[216:223] /*v[472:479]*/, 0
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[2:9] /*v[258:265]*/, v[80:87] /*v[592:599]*/, v[200:207] /*v[456:463]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5945
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[166:167] /*v[422:423]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[42:49] /*v[298:305]*/, v[80:87] /*v[592:599]*/, v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[166:167] /*v[422:423]*/, v[202:203] /*v[458:459]*/
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[166:167] /*v[422:423]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[166:167] /*v[422:423]*/, v[206:207] /*v[462:463]*/
	v_cndmask_b32_e64 v139 /*v395*/, v139 /*v395*/, 0xff61b1e6, s62
	v_cndmask_b32_e64 v138 /*v394*/, v138 /*v394*/, 0xff61b1e6, s2
	v_cndmask_b32_e64 v141 /*v397*/, v141 /*v397*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v140 /*v396*/, v140 /*v396*/, 0xff61b1e6, s4
	s_set_vgpr_msb 0x4585
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[90:97] /*v[346:353]*/, v[240:247] /*v[496:503]*/, 0
	s_set_vgpr_msb 0x8545
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[166:167] /*v[422:423]*/, v[146:147] /*v[402:403]*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[166:167] /*v[422:423]*/, v[148:149] /*v[404:405]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[166:167] /*v[422:423]*/, v[150:151] /*v[406:407]*/
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[166:167] /*v[422:423]*/, v[152:153] /*v[408:409]*/
	v_cndmask_b32_e64 v143 /*v399*/, v143 /*v399*/, 0xff61b1e6, s5
	v_cndmask_b32_e64 v142 /*v398*/, v142 /*v398*/, 0xff61b1e6, s6
	v_cndmask_b32_e64 v145 /*v401*/, v145 /*v401*/, 0xff61b1e6, s7
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[58:65] /*v[314:321]*/, v[32:39] /*v[544:551]*/, v[232:239] /*v[488:495]*/
	v_cndmask_b32_e64 v144 /*v400*/, v144 /*v400*/, 0xff61b1e6, s8
	v_cndmask_b32_e64 v153 /*v409*/, v153 /*v409*/, 0xff61b1e6, s15
	v_cndmask_b32_e64 v152 /*v408*/, v152 /*v408*/, 0xff61b1e6, s16
	v_cndmask_b32_e64 v147 /*v403*/, v147 /*v403*/, 0xff61b1e6, s9
	v_cndmask_b32_e64 v146 /*v402*/, v146 /*v402*/, 0xff61b1e6, s10
	v_cndmask_b32_e64 v149 /*v405*/, v149 /*v405*/, 0xff61b1e6, s11
	v_cndmask_b32_e64 v148 /*v404*/, v148 /*v404*/, 0xff61b1e6, s12
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[512:519]*/, v[58:65] /*v[314:321]*/, v[40:47] /*v[552:559]*/, v[0:7] /*v[512:519]*/
	s_set_vgpr_msb 0xa959
	v_cndmask_b32_e64 v151 /*v407*/, v151 /*v407*/, 0xff61b1e6, s13
	v_cndmask_b32_e64 v150 /*v406*/, v150 /*v406*/, 0xff61b1e6, s14
	s_wait_loadcnt 0x2
	v_pk_add_f32 v[138:139] /*v[394:395]*/, v[138:139] /*v[394:395]*/, v[116:117] /*v[628:629]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, v[116:117] /*v[628:629]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[116:117] /*v[628:629]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, v[116:117] /*v[628:629]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[116:117] /*v[628:629]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[2:9] /*v[258:265]*/, v[88:95] /*v[600:607]*/, v[208:215] /*v[464:471]*/
	v_pk_add_f32 v[146:147] /*v[402:403]*/, v[146:147] /*v[402:403]*/, v[116:117] /*v[628:629]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[148:149] /*v[404:405]*/, v[148:149] /*v[404:405]*/, v[116:117] /*v[628:629]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[150:151] /*v[406:407]*/, v[150:151] /*v[406:407]*/, v[116:117] /*v[628:629]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[138:139] /*v[394:395]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, s[42:43] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[42:49] /*v[298:305]*/, v[88:95] /*v[600:607]*/, v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[166:167] /*v[422:423]*/, v[208:209] /*v[464:465]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[166:167] /*v[422:423]*/, v[210:211] /*v[466:467]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[166:167] /*v[422:423]*/, v[212:213] /*v[468:469]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[166:167] /*v[422:423]*/, v[214:215] /*v[470:471]*/
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, s[42:43] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v201 /*v457*/, v201 /*v457*/, 0xff61b1e6, s17
	v_cndmask_b32_e64 v200 /*v456*/, v200 /*v456*/, 0xff61b1e6, s18
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[98:105] /*v[354:361]*/, v[32:39] /*v[544:551]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[166:167] /*v[422:423]*/, v[224:225] /*v[480:481]*/
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[166:167] /*v[422:423]*/, v[226:227] /*v[482:483]*/
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[166:167] /*v[422:423]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[166:167] /*v[422:423]*/, v[230:231] /*v[486:487]*/
	v_cndmask_b32_e64 v203 /*v459*/, v203 /*v459*/, 0xff61b1e6, s19
	v_cndmask_b32_e64 v202 /*v458*/, v202 /*v458*/, 0xff61b1e6, s20
	v_cndmask_b32_e64 v205 /*v461*/, v205 /*v461*/, 0xff61b1e6, s21
	s_set_vgpr_msb 0x45a9
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[98:105] /*v[354:361]*/, v[40:47] /*v[552:559]*/, v[16:23] /*v[528:535]*/
	s_set_vgpr_msb 0xa959
	v_cndmask_b32_e64 v204 /*v460*/, v204 /*v460*/, 0xff61b1e6, s22
	v_cndmask_b32_e64 v207 /*v463*/, v207 /*v463*/, 0xff61b1e6, s23
	v_cndmask_b32_e64 v206 /*v462*/, v206 /*v462*/, 0xff61b1e6, s24
	v_cndmask_b32_e64 v225 /*v481*/, v225 /*v481*/, 0xff61b1e6, s25
	v_cndmask_b32_e64 v224 /*v480*/, v224 /*v480*/, 0xff61b1e6, s26
	v_cndmask_b32_e64 v227 /*v483*/, v227 /*v483*/, 0xff61b1e6, s27
	v_cndmask_b32_e64 v226 /*v482*/, v226 /*v482*/, 0xff61b1e6, s28
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[74:81] /*v[330:337]*/, v[64:71] /*v[576:583]*/, v[232:239] /*v[488:495]*/
	v_cndmask_b32_e64 v229 /*v485*/, v229 /*v485*/, 0xff61b1e6, s29
	v_cndmask_b32_e64 v228 /*v484*/, v228 /*v484*/, 0xff61b1e6, s30
	v_cndmask_b32_e64 v231 /*v487*/, v231 /*v487*/, 0xff61b1e6, s31
	v_cndmask_b32_e64 v230 /*v486*/, v230 /*v486*/, 0xff61b1e6, s33
	s_wait_loadcnt 0x1
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[118:119] /*v[630:631]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, v[118:119] /*v[630:631]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[204:205] /*v[460:461]*/, v[118:119] /*v[630:631]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[512:519]*/, v[74:81] /*v[330:337]*/, v[72:79] /*v[584:591]*/, v[0:7] /*v[512:519]*/
	s_set_vgpr_msb 0xa959
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[118:119] /*v[630:631]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[118:119] /*v[630:631]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[226:227] /*v[482:483]*/, v[226:227] /*v[482:483]*/, v[118:119] /*v[630:631]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[228:229] /*v[484:485]*/, v[228:229] /*v[484:485]*/, v[118:119] /*v[630:631]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, v[118:119] /*v[630:631]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[146:147] /*v[402:403]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[148:149] /*v[404:405]*/, s[42:43] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[106:113] /*v[362:369]*/, v[64:71] /*v[576:583]*/, v[248:255] /*v[504:511]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[150:151] /*v[406:407]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[204:205] /*v[460:461]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[226:227] /*v[482:483]*/, s[42:43] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[106:113] /*v[362:369]*/, v[72:79] /*v[584:591]*/, v[16:23] /*v[528:535]*/
	s_set_vgpr_msb 0xa959
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[228:229] /*v[484:485]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, s[42:43] op_sel_hi:[1,0]
	v_exp_f32_e32 v146 /*v402*/, v146 /*v402*/
	v_exp_f32_e32 v147 /*v403*/, v147 /*v403*/
	v_exp_f32_e32 v148 /*v404*/, v148 /*v404*/
	v_exp_f32_e32 v149 /*v405*/, v149 /*v405*/
	v_exp_f32_e32 v150 /*v406*/, v150 /*v406*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[82:89] /*v[338:345]*/, v[96:103] /*v[608:615]*/, v[232:239] /*v[488:495]*/
	v_exp_f32_e32 v151 /*v407*/, v151 /*v407*/
	v_exp_f32_e32 v152 /*v408*/, v152 /*v408*/
	v_exp_f32_e32 v153 /*v409*/, v153 /*v409*/
	v_exp_f32_e32 v200 /*v456*/, v200 /*v456*/
	v_exp_f32_e32 v201 /*v457*/, v201 /*v457*/
	v_exp_f32_e32 v202 /*v458*/, v202 /*v458*/
	v_exp_f32_e32 v203 /*v459*/, v203 /*v459*/
	s_set_vgpr_msb 0x59a9
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[512:519]*/, v[82:89] /*v[338:345]*/, v[104:111] /*v[616:623]*/, v[0:7] /*v[512:519]*/
	s_set_vgpr_msb 0xa959
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[232:233] /*v[488:489]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[210:211] /*v[466:467]*/, v[234:235] /*v[490:491]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[212:213] /*v[468:469]*/, v[236:237] /*v[492:493]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[238:239] /*v[494:495]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v204 /*v460*/, v204 /*v460*/
	v_exp_f32_e32 v205 /*v461*/, v205 /*v461*/
	v_exp_f32_e32 v206 /*v462*/, v206 /*v462*/
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[114:121] /*v[370:377]*/, v[96:103] /*v[608:615]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x594a
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[0:1] /*v[512:513]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[2:3] /*v[514:515]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[236:237] /*v[492:493]*/, v[4:5] /*v[516:517]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[6:7] /*v[518:519]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa9
	v_exp_f32_e32 v0 /*v512*/, v138 /*v394*/
	v_exp_f32_e32 v1 /*v513*/, v139 /*v395*/
	v_exp_f32_e32 v2 /*v514*/, v140 /*v396*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[114:121] /*v[370:377]*/, v[104:111] /*v[616:623]*/, v[16:23] /*v[528:535]*/
	v_exp_f32_e32 v3 /*v515*/, v141 /*v397*/
	v_exp_f32_e32 v4 /*v516*/, v142 /*v398*/
	v_exp_f32_e32 v5 /*v517*/, v143 /*v399*/
	v_exp_f32_e32 v6 /*v518*/, v144 /*v400*/
	v_exp_f32_e32 v7 /*v519*/, v145 /*v401*/
	s_set_vgpr_msb 0xa949
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[248:249] /*v[504:505]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[250:251] /*v[506:507]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[220:221] /*v[476:477]*/, v[252:253] /*v[508:509]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[254:255] /*v[510:511]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v207 /*v463*/, v207 /*v463*/
	v_exp_f32_e32 v224 /*v480*/, v224 /*v480*/
	v_exp_f32_e32 v225 /*v481*/, v225 /*v481*/
	v_exp_f32_e32 v226 /*v482*/, v226 /*v482*/
	v_exp_f32_e32 v227 /*v483*/, v227 /*v483*/
	v_exp_f32_e32 v228 /*v484*/, v228 /*v484*/
	v_exp_f32_e32 v229 /*v485*/, v229 /*v485*/
	v_exp_f32_e32 v230 /*v486*/, v230 /*v486*/
	v_exp_f32_e32 v231 /*v487*/, v231 /*v487*/
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[16:17] /*v[528:529]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[250:251] /*v[506:507]*/, v[18:19] /*v[530:531]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[252:253] /*v[508:509]*/, v[20:21] /*v[532:533]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[22:23] /*v[534:535]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v141 /*v397*/, v6 /*v518*/, v7 /*v519*/
	v_cvt_pk_bf16_f32 v140 /*v396*/, v4 /*v516*/, v5 /*v517*/
	v_cvt_pk_bf16_f32 v139 /*v395*/, v2 /*v514*/, v3 /*v515*/
	v_cvt_pk_bf16_f32 v138 /*v394*/, v0 /*v512*/, v1 /*v513*/
	s_set_vgpr_msb 0x4a49
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[0:1] /*v[512:513]*/
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[210:211] /*v[466:467]*/, v[2:3] /*v[514:515]*/
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[212:213] /*v[468:469]*/, v[4:5] /*v[516:517]*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, v[6:7] /*v[518:519]*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v143 /*v399*/, v148 /*v404*/, v149 /*v405*/
	v_cvt_pk_bf16_f32 v144 /*v400*/, v150 /*v406*/, v151 /*v407*/
	v_cvt_pk_bf16_f32 v142 /*v398*/, v146 /*v402*/, v147 /*v403*/
	v_cvt_pk_bf16_f32 v145 /*v401*/, v152 /*v408*/, v153 /*v409*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, v[146:147] /*v[402:403]*/
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[148:149] /*v[404:405]*/
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, v[150:151] /*v[406:407]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[152:153] /*v[408:409]*/
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[236:237] /*v[492:493]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, v[206:207] /*v[462:463]*/
	v_cvt_pk_bf16_f32 v149 /*v405*/, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v148 /*v404*/, v204 /*v460*/, v205 /*v461*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[234:235] /*v[490:491]*/, v[202:203] /*v[458:459]*/
	v_cvt_pk_bf16_f32 v147 /*v403*/, v202 /*v458*/, v203 /*v459*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[232:233] /*v[488:489]*/, v[200:201] /*v[456:457]*/
	v_cvt_pk_bf16_f32 v146 /*v402*/, v200 /*v456*/, v201 /*v457*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[252:253] /*v[508:509]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[254:255] /*v[510:511]*/, v[230:231] /*v[486:487]*/
	v_cvt_pk_bf16_f32 v152 /*v408*/, v228 /*v484*/, v229 /*v485*/
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[250:251] /*v[506:507]*/, v[226:227] /*v[482:483]*/
	v_cvt_pk_bf16_f32 v151 /*v407*/, v226 /*v482*/, v227 /*v483*/
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[248:249] /*v[504:505]*/, v[224:225] /*v[480:481]*/
	ds_store_b128 v178 /*v434*/, v[138:141] /*v[394:397]*/
	ds_store_b128 v178 /*v434*/, v[142:145] /*v[398:401]*/ offset:32
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[166:167] /*v[422:423]*/, v[208:209] /*v[464:465]*/
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[166:167] /*v[422:423]*/, v[210:211] /*v[466:467]*/
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[166:167] /*v[422:423]*/, v[212:213] /*v[468:469]*/
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[166:167] /*v[422:423]*/, v[214:215] /*v[470:471]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[166:167] /*v[422:423]*/, v[216:217] /*v[472:473]*/
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[166:167] /*v[422:423]*/, v[218:219] /*v[474:475]*/
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[166:167] /*v[422:423]*/, v[220:221] /*v[476:477]*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[166:167] /*v[422:423]*/, v[222:223] /*v[478:479]*/
	v_cvt_pk_bf16_f32 v150 /*v406*/, v224 /*v480*/, v225 /*v481*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[166:167] /*v[422:423]*/, v[202:203] /*v[458:459]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[166:167] /*v[422:423]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[166:167] /*v[422:423]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[166:167] /*v[422:423]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[166:167] /*v[422:423]*/, v[226:227] /*v[482:483]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[166:167] /*v[422:423]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[166:167] /*v[422:423]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[166:167] /*v[422:423]*/, v[206:207] /*v[462:463]*/
	v_cvt_pk_bf16_f32 v138 /*v394*/, v138 /*v394*/, v139 /*v395*/
	v_cvt_pk_bf16_f32 v139 /*v395*/, v140 /*v396*/, v141 /*v397*/
	v_cvt_pk_bf16_f32 v140 /*v396*/, v142 /*v398*/, v143 /*v399*/
	v_cvt_pk_bf16_f32 v141 /*v397*/, v144 /*v400*/, v145 /*v401*/
	v_cvt_pk_bf16_f32 v142 /*v398*/, v208 /*v464*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v143 /*v399*/, v210 /*v466*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v144 /*v400*/, v212 /*v468*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v145 /*v401*/, v214 /*v470*/, v215 /*v471*/
	v_cvt_pk_bf16_f32 v153 /*v409*/, v230 /*v486*/, v231 /*v487*/
	v_cvt_pk_bf16_f32 v200 /*v456*/, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v201 /*v457*/, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v216 /*v472*/, v217 /*v473*/
	v_cvt_pk_bf16_f32 v203 /*v459*/, v218 /*v474*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v220 /*v476*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v222 /*v478*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v224 /*v480*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v207 /*v463*/, v226 /*v482*/, v227 /*v483*/
	ds_store_b128 v178 /*v434*/, v[138:141] /*v[394:397]*/ offset:2560
	ds_store_b128 v178 /*v434*/, v[142:145] /*v[398:401]*/ offset:2592
	ds_store_b128 v173 /*v429*/, v[240:243] /*v[496:499]*/ offset:5120
	ds_store_b128 v173 /*v429*/, v[244:247] /*v[500:503]*/ offset:5152
	ds_store_b128 v173 /*v429*/, v[130:133] /*v[386:389]*/ offset:13824
	ds_store_b128 v173 /*v429*/, v[134:137] /*v[390:393]*/ offset:13856
	s_set_vgpr_msb 0x4509
	ds_store_b128 v173 /*v429*/, v[40:43] /*v[552:555]*/ offset:5184
	ds_store_b128 v173 /*v429*/, v[44:47] /*v[556:559]*/ offset:5216
	ds_store_b128 v173 /*v429*/, v[24:27] /*v[536:539]*/ offset:13888
	ds_store_b128 v173 /*v429*/, v[28:31] /*v[540:543]*/ offset:13920
	ds_store_b128 v173 /*v429*/, v[72:75] /*v[584:587]*/ offset:5248
	ds_store_b128 v173 /*v429*/, v[76:79] /*v[588:591]*/ offset:5280
	ds_store_b128 v173 /*v429*/, v[56:59] /*v[568:571]*/ offset:13952
	ds_store_b128 v173 /*v429*/, v[60:63] /*v[572:575]*/ offset:13984
	ds_store_b128 v173 /*v429*/, v[104:107] /*v[616:619]*/ offset:5312
	ds_store_b128 v173 /*v429*/, v[108:111] /*v[620:623]*/ offset:5344
	ds_store_b128 v173 /*v429*/, v[88:91] /*v[600:603]*/ offset:14016
	ds_store_b128 v173 /*v429*/, v[92:95] /*v[604:607]*/ offset:14048
	s_set_vgpr_msb 0x945
	ds_store_b128 v179 /*v435*/, v[146:149] /*v[402:405]*/
	ds_store_b128 v179 /*v435*/, v[150:153] /*v[406:409]*/ offset:32
	ds_store_b128 v179 /*v435*/, v[200:203] /*v[456:459]*/ offset:2560
	ds_store_b128 v179 /*v435*/, v[204:207] /*v[460:463]*/ offset:2592
	s_wait_dscnt 0x0
	ds_load_tr16_b128 v[130:133] /*v[386:389]*/, v180 /*v436*/
	ds_load_tr16_b128 v[134:137] /*v[390:393]*/, v180 /*v436*/ offset:4352
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v196 /*v452*/
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v196 /*v452*/ offset:1280
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v182 /*v438*/
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v182 /*v438*/ offset:4352
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v184 /*v440*/
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v184 /*v440*/ offset:4352
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v186 /*v442*/
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v186 /*v442*/ offset:4352
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v188 /*v444*/
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v188 /*v444*/ offset:4352
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v190 /*v446*/
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v190 /*v446*/ offset:4352
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v192 /*v448*/
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v192 /*v448*/ offset:4352
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v194 /*v450*/
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v194 /*v450*/ offset:4352
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v181 /*v437*/
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v181 /*v437*/ offset:4352
	s_set_vgpr_msb 0x4581
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v197 /*v453*/
	s_set_vgpr_msb 0x8155
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[138:145] /*v[394:401]*/, v[130:137] /*v[386:393]*/, v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5581
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v185 /*v441*/ offset:4352
	ds_load_tr16_b128 v[16:19] /*v[528:531]*/, v187 /*v443*/
	ds_load_tr16_b128 v[20:23] /*v[532:535]*/, v187 /*v443*/ offset:4352
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v189 /*v445*/
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v189 /*v445*/ offset:4352
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v191 /*v447*/
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v191 /*v447*/ offset:4352
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v193 /*v449*/
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v193 /*v449*/ offset:4352
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v195 /*v451*/
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v195 /*v451*/ offset:4352
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v198 /*v454*/
	s_set_vgpr_msb 0x8105
	s_wait_dscnt 0x1b
	v_wmma_f32_16x16x32_bf16 v[218:225], v[138:145] /*v[394:401]*/, v[146:153] /*v[402:409]*/, v[218:225]
	s_wait_dscnt 0x19
	v_wmma_f32_16x16x32_bf16 v[202:209], v[138:145] /*v[394:401]*/, v[200:207] /*v[456:463]*/, v[202:209]
	s_wait_dscnt 0x17
	v_wmma_f32_16x16x32_bf16 v[186:193], v[138:145] /*v[394:401]*/, v[208:215] /*v[464:471]*/, v[186:193]
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[170:177], v[138:145] /*v[394:401]*/, v[216:223] /*v[472:479]*/, v[170:177]
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x32_bf16 v[154:161], v[138:145] /*v[394:401]*/, v[224:231] /*v[480:487]*/, v[154:161]
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x32_bf16 v[138:145], v[138:145] /*v[394:401]*/, v[232:239] /*v[488:495]*/, v[138:145]
	s_wait_dscnt 0xf
	v_wmma_f32_16x16x32_bf16 v[130:137], v[138:145] /*v[394:401]*/, v[240:247] /*v[496:503]*/, v[130:137]
	s_set_vgpr_msb 0x581
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v197 /*v453*/ offset:1280
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v183 /*v439*/
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v183 /*v439*/ offset:4352
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[8:11] /*v[520:523]*/, v185 /*v441*/
	s_set_vgpr_msb 0x8156
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[0:7] /*v[512:519]*/, v[248:255] /*v[504:511]*/, v[122:129] /*v[378:385]*/
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[0:7] /*v[512:519]*/, v[138:145] /*v[394:401]*/, v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x560a
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[226:233], v[0:7] /*v[512:519]*/, v[8:15] /*v[520:527]*/, v[226:233]
	v_wmma_f32_16x16x32_bf16 v[210:217], v[0:7] /*v[512:519]*/, v[16:23] /*v[528:535]*/, v[210:217]
	v_wmma_f32_16x16x32_bf16 v[194:201], v[0:7] /*v[512:519]*/, v[24:31] /*v[536:543]*/, v[194:201]
	v_wmma_f32_16x16x32_bf16 v[178:185], v[0:7] /*v[512:519]*/, v[32:39] /*v[544:551]*/, v[178:185]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[0:7] /*v[512:519]*/, v[40:47] /*v[552:559]*/, v[162:169]
	v_wmma_f32_16x16x32_bf16 v[146:153], v[0:7] /*v[512:519]*/, v[48:55] /*v[560:567]*/, v[146:153]
	s_set_vgpr_msb 0xa81
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v198 /*v454*/ offset:1280
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v199 /*v455*/
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v199 /*v455*/ offset:1280
	s_set_vgpr_msb 0x8106
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[56:63] /*v[568:575]*/, v[130:137] /*v[386:393]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[122:129], v[0:7] /*v[512:519]*/, v[248:255] /*v[504:511]*/, v[122:129]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[56:63] /*v[568:575]*/, v[146:153] /*v[402:409]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[106:113], v[0:7] /*v[512:519]*/, v[138:145] /*v[394:401]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[56:63] /*v[568:575]*/, v[200:207] /*v[456:463]*/, v[74:81]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[98:105], v[0:7] /*v[512:519]*/, v[8:15] /*v[520:527]*/, v[98:105]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[58:65], v[56:63] /*v[568:575]*/, v[208:215] /*v[464:471]*/, v[58:65]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[82:89], v[0:7] /*v[512:519]*/, v[16:23] /*v[528:535]*/, v[82:89]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[42:49], v[56:63] /*v[568:575]*/, v[216:223] /*v[472:479]*/, v[42:49]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[66:73], v[0:7] /*v[512:519]*/, v[24:31] /*v[536:543]*/, v[66:73]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[26:33], v[56:63] /*v[568:575]*/, v[224:231] /*v[480:487]*/, v[26:33]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[50:57], v[0:7] /*v[512:519]*/, v[32:39] /*v[544:551]*/, v[50:57]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[10:17], v[56:63] /*v[568:575]*/, v[232:239] /*v[488:495]*/, v[10:17]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[34:41], v[0:7] /*v[512:519]*/, v[40:47] /*v[552:559]*/, v[34:41]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[2:9], v[56:63] /*v[568:575]*/, v[240:247] /*v[496:503]*/, v[2:9]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[18:25], v[0:7] /*v[512:519]*/, v[48:55] /*v[560:567]*/, v[18:25]
	s_set_vgpr_msb 0xa00
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
	v_mov_b64_e32 v[112:113], v[24:25]
	v_mov_b64_e32 v[110:111], v[22:23]
	v_mov_b64_e32 v[108:109], v[20:21]
	v_mov_b64_e32 v[106:107], v[18:19]
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
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[24:25]
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[22:23]
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[20:21]
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[18:19]
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
	v_mov_b64_e32 v[120:121], v[24:25]
	v_mov_b64_e32 v[118:119], v[22:23]
	v_mov_b64_e32 v[116:117], v[20:21]
	v_mov_b64_e32 v[114:115], v[18:19]
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
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[40:41] /*v[296:297]*/, v[24:25]
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[22:23]
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[20:21]
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[18:19]
	s_set_vgpr_msb 0x4000
.LBB0_4:
	s_sub_co_i32 s2, s70, s71
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s2, s2, s41
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB0_7
	s_abs_i32 s8, s41
	s_movk_i32 s4, 0x1400
	s_cvt_f32_u32 s5, s8
	s_set_vgpr_msb 0x55
	v_mad_u32_u24 v148 /*v404*/, 0x110, v176 /*v432*/, s4
	s_movk_i32 s4, 0x3600
	v_or_b32_e32 v137 /*v393*/, 32, v177 /*v433*/
	v_s_rcp_f32 s5, s5
	v_mad_u32_u24 v149 /*v405*/, 0x110, v176 /*v432*/, s4
	v_or_b32_e32 v139 /*v395*/, 64, v174 /*v430*/
	v_or_b32_e32 v141 /*v397*/, 0x60, v177 /*v433*/
	v_or_b32_e32 v143 /*v399*/, 0x80, v174 /*v430*/
	v_or_b32_e32 v145 /*v401*/, 0xa0, v177 /*v433*/
	v_or_b32_e32 v147 /*v403*/, 0xc0, v174 /*v430*/
	v_or_b32_e32 v150 /*v406*/, 0xe0, v177 /*v433*/
	s_mul_f32 s4, s5, 0x4f7ffffe
	s_movk_i32 s5, 0xa00
	v_add_nc_u32_e32 v134 /*v390*/, v148 /*v404*/, v174 /*v430*/
	v_mad_u32_u24 v153 /*v409*/, 0x50, v176 /*v432*/, s5
	s_cvt_u32_f32 s4, s4
	v_dual_add_nc_u32 v136 /*v392*/, v148 /*v404*/, v137 /*v393*/ :: v_dual_bitop2_b32 v166 /*v422*/, 32, v174 /*v430*/ bitop3:0x54
	s_sub_co_i32 s5, 0, s8
	s_mov_b32 s12, s36
	s_mov_b32 s13, s36
	s_mul_i32 s5, s5, s4
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, s[12:13]
	v_mad_i32_i24 v132 /*v388*/, 0xffffff40, v165 /*v421*/, v172 /*v428*/
	v_mad_i32_i24 v133 /*v389*/, 0xffffff40, v171 /*v427*/, v173 /*v429*/
	v_dual_add_nc_u32 v135 /*v391*/, v149 /*v405*/, v174 /*v430*/ :: v_dual_add_nc_u32 v137 /*v393*/, v149 /*v405*/, v137 /*v393*/
	v_dual_add_nc_u32 v138 /*v394*/, v148 /*v404*/, v139 /*v395*/ :: v_dual_add_nc_u32 v139 /*v395*/, v149 /*v405*/, v139 /*v395*/
	v_dual_add_nc_u32 v140 /*v396*/, v148 /*v404*/, v141 /*v397*/ :: v_dual_add_nc_u32 v141 /*v397*/, v149 /*v405*/, v141 /*v397*/
	v_dual_add_nc_u32 v142 /*v398*/, v148 /*v404*/, v143 /*v399*/ :: v_dual_add_nc_u32 v143 /*v399*/, v149 /*v405*/, v143 /*v399*/
	v_dual_add_nc_u32 v144 /*v400*/, v148 /*v404*/, v145 /*v401*/ :: v_dual_add_nc_u32 v145 /*v401*/, v149 /*v405*/, v145 /*v401*/
	v_dual_add_nc_u32 v146 /*v402*/, v148 /*v404*/, v147 /*v403*/ :: v_dual_add_nc_u32 v147 /*v403*/, v149 /*v405*/, v147 /*v403*/
	v_dual_add_nc_u32 v148 /*v404*/, v148 /*v404*/, v150 /*v406*/ :: v_dual_add_nc_u32 v149 /*v405*/, v149 /*v405*/, v150 /*v406*/
	v_dual_add_nc_u32 v150 /*v406*/, v175 /*v431*/, v174 /*v430*/ :: v_dual_add_nc_u32 v151 /*v407*/, v153 /*v409*/, v174 /*v430*/
	v_dual_add_nc_u32 v152 /*v408*/, v175 /*v431*/, v166 /*v422*/ :: v_dual_add_nc_u32 v153 /*v409*/, v153 /*v409*/, v166 /*v422*/
	s_mul_hi_u32 s5, s4, s5
	s_add_co_i32 s7, s71, s69
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s9, s41, 31
	s_add_co_i32 s10, s4, s5
	s_mov_b64 s[4:5], 0
	s_mov_b32 s50, s54
	s_mov_b32 s51, s55
	s_mov_b32 s62, s58
	s_mov_b32 s63, s59
	s_mov_b32 s6, 0x3fb8aa3b
	s_set_vgpr_msb 0x5500
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
	s_add_co_i32 s12, s7, s11
	s_mul_i32 s11, s11, s41
	s_lshl_b32 s12, s12, 5
	s_sub_co_i32 s11, s4, s11
	s_set_vgpr_msb 0x44
	v_or_b32_e32 v166 /*v422*/, s12, v165 /*v421*/
	s_add_co_i32 s11, s11, s66
	v_or_b32_e32 v167 /*v423*/, s12, v171 /*v427*/
	s_lshl4_add_u32 s12, s11, s68
	s_add_co_i32 s11, s11, s67
	v_mad_u32 v174 /*v430*/, s35, v166 /*v422*/, s12
	s_mul_i32 s11, s11, s37
	v_mad_u32 v206 /*v462*/, s35, v167 /*v423*/, s12
	s_set_vgpr_msb 0x4484
	v_add_lshl_u32 v95 /*v607*/, s11, v166 /*v422*/, 2
	s_set_vgpr_msb 0x8446
	v_add_lshl_u32 v167 /*v423*/, s11, v167 /*v423*/, 2
	buffer_load_b32 v166 /*v422*/, v95 /*v607*/, s[60:63], null offen
	s_set_vgpr_msb 0x4682
	s_clause 0x2
	buffer_load_b32 v110 /*v622*/, v95 /*v607*/, s[56:59], null offen
	s_set_vgpr_msb 0x8281
	buffer_load_b32 v112 /*v624*/, v167 /*v423*/, s[56:59], null offen
	s_set_vgpr_msb 0x8145
	v_or_b32_e32 v174 /*v430*/, v174 /*v430*/, v169 /*v425*/
	v_or_b32_e32 v214 /*v470*/, v206 /*v462*/, v169 /*v425*/
	s_set_vgpr_msb 0x4585
	buffer_load_b32 v114 /*v626*/, v167 /*v423*/, s[60:63], null offen
	s_add_nc_u64 s[4:5], s[4:5], 1
	v_lshlrev_b32_e32 v78 /*v590*/, 4, v174 /*v430*/
	v_lshlrev_b32_e32 v94 /*v606*/, 4, v214 /*v470*/
	s_cmp_lg_u64 s[4:5], s[2:3]
	s_set_vgpr_msb 0x854a
	buffer_load_b128 v[214:217] /*v[470:473]*/, v94 /*v606*/, s[52:55], null offen
	buffer_load_b128 v[222:225] /*v[478:481]*/, v94 /*v606*/, s[48:51], null offen
	v_or_b32_e32 v178 /*v434*/, 32, v78 /*v590*/
	buffer_load_b128 v[174:177] /*v[430:433]*/, v78 /*v590*/, s[48:51], null offen
	s_clause 0x2
	buffer_load_b128 v[182:185] /*v[438:441]*/, v78 /*v590*/, s[52:55], null offen
	s_set_vgpr_msb 0x4a49
	buffer_load_b128 v[186:189] /*v[442:445]*/, v178 /*v434*/, s[52:55], null offen
	v_or_b32_e32 v226 /*v482*/, 32, v94 /*v606*/
	s_set_vgpr_msb 0x4988
	v_or_b32_e32 v22 /*v534*/, 64, v78 /*v590*/
	v_or_b32_e32 v26 /*v538*/, 0x60, v78 /*v590*/
	v_or_b32_e32 v38 /*v550*/, 64, v94 /*v606*/
	s_set_vgpr_msb 0x8841
	buffer_load_b128 v[178:181] /*v[434:437]*/, v178 /*v434*/, s[48:51], null offen
	s_clause 0x2
	buffer_load_b128 v[218:221] /*v[474:477]*/, v226 /*v482*/, s[52:55], null offen
	s_set_vgpr_msb 0x418a
	buffer_load_b128 v[6:9] /*v[518:521]*/, v22 /*v534*/, s[52:55], null offen
	v_or_b32_e32 v42 /*v554*/, 0x60, v94 /*v606*/
	s_clause 0x1
	buffer_load_b128 v[30:33] /*v[542:545]*/, v38 /*v550*/, s[52:55], null offen
	buffer_load_b128 v[10:13] /*v[522:525]*/, v26 /*v538*/, s[52:55], null offen
	buffer_load_b128 v[22:25] /*v[534:537]*/, v22 /*v534*/, s[48:51], null offen
	s_set_vgpr_msb 0x8a48
	v_or_b32_e32 v167 /*v423*/, 0xc0, v94 /*v606*/
	s_set_vgpr_msb 0x488a
	v_or_b32_e32 v54 /*v566*/, 0x80, v78 /*v590*/
	v_or_b32_e32 v58 /*v570*/, 0xa0, v78 /*v590*/
	v_or_b32_e32 v70 /*v582*/, 0x80, v94 /*v606*/
	buffer_load_b128 v[26:29] /*v[538:541]*/, v26 /*v538*/, s[48:51], null offen
	buffer_load_b128 v[34:37] /*v[546:549]*/, v42 /*v554*/, s[52:55], null offen
	buffer_load_b128 v[38:41] /*v[550:553]*/, v38 /*v550*/, s[48:51], null offen
	v_or_b32_e32 v74 /*v586*/, 0xa0, v94 /*v606*/
	v_or_b32_e32 v106 /*v618*/, 0xe0, v94 /*v606*/
	s_set_vgpr_msb 0x8a81
	buffer_load_b128 v[94:97] /*v[606:609]*/, v167 /*v423*/, s[52:55], null offen
	s_set_vgpr_msb 0x8141
	buffer_load_b128 v[226:229] /*v[482:485]*/, v226 /*v482*/, s[48:51], null offen
	s_set_vgpr_msb 0x418a
	s_clause 0x2
	buffer_load_b128 v[46:49] /*v[558:561]*/, v54 /*v566*/, s[52:55], null offen
	buffer_load_b128 v[62:65] /*v[574:577]*/, v70 /*v582*/, s[52:55], null offen
	buffer_load_b128 v[50:53] /*v[562:565]*/, v58 /*v570*/, s[52:55], null offen
	buffer_load_b128 v[54:57] /*v[566:569]*/, v54 /*v566*/, s[48:51], null offen
	v_or_b32_e32 v86 /*v598*/, 0xc0, v78 /*v590*/
	v_or_b32_e32 v90 /*v602*/, 0xe0, v78 /*v590*/
	buffer_load_b128 v[58:61] /*v[570:573]*/, v58 /*v570*/, s[48:51], null offen
	buffer_load_b128 v[66:69] /*v[578:581]*/, v74 /*v586*/, s[52:55], null offen
	s_clause 0x1
	buffer_load_b128 v[70:73] /*v[582:585]*/, v70 /*v582*/, s[48:51], null offen
	buffer_load_b128 v[42:45] /*v[554:557]*/, v42 /*v554*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[78:81] /*v[590:593]*/, v86 /*v598*/, s[52:55], null offen
	buffer_load_b128 v[82:85] /*v[594:597]*/, v90 /*v602*/, s[52:55], null offen
	s_clause 0x1
	buffer_load_b128 v[86:89] /*v[598:601]*/, v86 /*v598*/, s[48:51], null offen
	buffer_load_b128 v[90:93] /*v[602:605]*/, v90 /*v602*/, s[48:51], null offen
	buffer_load_b128 v[98:101] /*v[610:613]*/, v106 /*v618*/, s[52:55], null offen
	s_set_vgpr_msb 0x8a81
	s_clause 0x3
	buffer_load_b128 v[102:105] /*v[614:617]*/, v167 /*v423*/, s[48:51], null offen
	s_set_vgpr_msb 0x8182
	buffer_load_b128 v[74:77] /*v[586:589]*/, v74 /*v586*/, s[48:51], null offen
	buffer_load_b128 v[106:109] /*v[618:621]*/, v106 /*v618*/, s[48:51], null offen
	s_set_vgpr_msb 0x8205
	s_wait_loadcnt 0x1d
	ds_store_b128 v172 /*v428*/, v[174:177] /*v[430:433]*/ offset:5120
	s_wait_loadcnt 0x1c
	ds_store_b128 v172 /*v428*/, v[182:185] /*v[438:441]*/ offset:13824
	s_set_vgpr_msb 0x544
	s_wait_loadcnt 0x1b
	v_wmma_f32_16x16x32_bf16 v[190:197] /*v[446:453]*/, v[234:241], v[182:189] /*v[438:445]*/, 0
	s_set_vgpr_msb 0x4405
	s_wait_loadcnt 0x1a
	ds_store_b128 v172 /*v428*/, v[178:181] /*v[434:437]*/ offset:5152
	ds_store_b128 v172 /*v428*/, v[186:189] /*v[442:445]*/ offset:13856
	s_set_vgpr_msb 0x509
	s_wait_loadcnt 0x15
	ds_store_b128 v172 /*v428*/, v[22:25] /*v[534:537]*/ offset:5184
	s_wait_loadcnt 0x14
	ds_store_b128 v172 /*v428*/, v[26:29] /*v[538:541]*/ offset:5216
	ds_store_b128 v172 /*v428*/, v[6:9] /*v[518:521]*/ offset:13888
	ds_store_b128 v172 /*v428*/, v[10:13] /*v[522:525]*/ offset:13920
	s_wait_loadcnt 0xc
	ds_store_b128 v172 /*v428*/, v[54:57] /*v[566:569]*/ offset:5248
	s_set_vgpr_msb 0x945
	v_wmma_f32_16x16x32_bf16 v[206:213] /*v[462:469]*/, v[10:17] /*v[266:273]*/, v[182:189] /*v[438:445]*/, 0
	s_set_vgpr_msb 0x4509
	s_wait_loadcnt 0xb
	ds_store_b128 v172 /*v428*/, v[58:61] /*v[570:573]*/ offset:5280
	ds_store_b128 v172 /*v428*/, v[46:49] /*v[558:561]*/ offset:13952
	ds_store_b128 v172 /*v428*/, v[50:53] /*v[562:565]*/ offset:13984
	s_wait_loadcnt 0x5
	ds_store_b128 v172 /*v428*/, v[86:89] /*v[598:601]*/ offset:5312
	s_wait_loadcnt 0x4
	ds_store_b128 v172 /*v428*/, v[90:93] /*v[602:605]*/ offset:5344
	ds_store_b128 v172 /*v428*/, v[78:81] /*v[590:593]*/ offset:14016
	ds_store_b128 v172 /*v428*/, v[82:85] /*v[594:597]*/ offset:14048
	s_set_vgpr_msb 0x945
	v_wmma_f32_16x16x32_bf16 v[198:205] /*v[454:461]*/, v[50:57] /*v[306:313]*/, v[174:181] /*v[430:437]*/, 0
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[90:97] /*v[346:353]*/, v[174:181] /*v[430:437]*/, 0
	s_set_vgpr_msb 0x4544
	v_wmma_f32_16x16x32_bf16 v[238:245] /*v[494:501]*/, v[234:241], v[214:221] /*v[470:477]*/, 0
	s_set_vgpr_msb 0x4445
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[510:517]*/, v[10:17] /*v[266:273]*/, v[214:221] /*v[470:477]*/, 0
	s_set_vgpr_msb 0x4558
	v_wmma_f32_16x16x32_bf16 v[190:197] /*v[446:453]*/, v[242:249], v[6:13] /*v[518:525]*/, v[190:197] /*v[446:453]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[206:213] /*v[462:469]*/, v[18:25] /*v[274:281]*/, v[6:13] /*v[518:525]*/, v[206:213] /*v[462:469]*/
	v_wmma_f32_16x16x32_bf16 v[198:205] /*v[454:461]*/, v[58:65] /*v[314:321]*/, v[22:29] /*v[534:541]*/, v[198:205] /*v[454:461]*/
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[98:105] /*v[354:361]*/, v[22:29] /*v[534:541]*/, v[230:237] /*v[486:493]*/
	s_set_vgpr_msb 0x5958
	v_wmma_f32_16x16x32_bf16 v[238:245] /*v[494:501]*/, v[242:249], v[30:37] /*v[542:549]*/, v[238:245] /*v[494:501]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[510:517]*/, v[18:25] /*v[274:281]*/, v[30:37] /*v[542:549]*/, v[254:261] /*v[510:517]*/
	s_set_vgpr_msb 0x5945
	v_wmma_f32_16x16x32_bf16 v[246:253] /*v[502:509]*/, v[50:57] /*v[306:313]*/, v[222:229] /*v[478:485]*/, 0
	s_set_vgpr_msb 0x4558
	v_wmma_f32_16x16x32_bf16 v[190:197] /*v[446:453]*/, v[250:257], v[46:53] /*v[558:565]*/, v[190:197] /*v[446:453]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[206:213] /*v[462:469]*/, v[26:33] /*v[282:289]*/, v[46:53] /*v[558:565]*/, v[206:213] /*v[462:469]*/
	v_wmma_f32_16x16x32_bf16 v[198:205] /*v[454:461]*/, v[74:81] /*v[330:337]*/, v[54:61] /*v[566:573]*/, v[198:205] /*v[454:461]*/
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[106:113] /*v[362:369]*/, v[54:61] /*v[566:573]*/, v[230:237] /*v[486:493]*/
	s_set_vgpr_msb 0x5958
	v_wmma_f32_16x16x32_bf16 v[238:245] /*v[494:501]*/, v[250:257], v[62:69] /*v[574:581]*/, v[238:245] /*v[494:501]*/
	s_set_vgpr_msb 0x5859
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[510:517]*/, v[26:33] /*v[282:289]*/, v[62:69] /*v[574:581]*/, v[254:261] /*v[510:517]*/
	s_set_vgpr_msb 0x5985
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[90:97] /*v[346:353]*/, v[222:229] /*v[478:485]*/, 0
	s_set_vgpr_msb 0x8559
	v_wmma_f32_16x16x32_bf16 v[246:253] /*v[502:509]*/, v[58:65] /*v[314:321]*/, v[38:45] /*v[550:557]*/, v[246:253] /*v[502:509]*/
	v_wmma_f32_16x16x32_bf16 v[190:197] /*v[446:453]*/, v[2:9] /*v[258:265]*/, v[78:85] /*v[590:597]*/, v[190:197] /*v[446:453]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5945
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[130:131] /*v[386:387]*/, v[190:191] /*v[446:447]*/
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[206:213] /*v[462:469]*/, v[42:49] /*v[298:305]*/, v[78:85] /*v[590:597]*/, v[206:213] /*v[462:469]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[130:131] /*v[386:387]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[130:131] /*v[386:387]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[130:131] /*v[386:387]*/, v[196:197] /*v[452:453]*/
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[198:205] /*v[454:461]*/, v[82:89] /*v[338:345]*/, v[86:93] /*v[598:605]*/, v[198:205] /*v[454:461]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[130:131] /*v[386:387]*/, v[206:207] /*v[462:463]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[130:131] /*v[386:387]*/, v[208:209] /*v[464:465]*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[130:131] /*v[386:387]*/, v[210:211] /*v[466:467]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[130:131] /*v[386:387]*/, v[212:213] /*v[468:469]*/
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[114:121] /*v[370:377]*/, v[86:93] /*v[598:605]*/, v[230:237] /*v[486:493]*/
	s_set_vgpr_msb 0x5945
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[198:199] /*v[454:455]*/, v[166:167] /*v[422:423]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[184:185] /*v[440:441]*/, v[200:201] /*v[456:457]*/, v[166:167] /*v[422:423]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[186:187] /*v[442:443]*/, v[202:203] /*v[458:459]*/, v[166:167] /*v[422:423]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[204:205] /*v[460:461]*/, v[166:167] /*v[422:423]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4559
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[110:111] /*v[622:623]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, s[6:7] op_sel_hi:[1,0]
	s_wait_loadcnt 0x3
	v_wmma_f32_16x16x32_bf16 v[238:245] /*v[494:501]*/, v[2:9] /*v[258:265]*/, v[94:101] /*v[606:613]*/, v[238:245] /*v[494:501]*/
	s_set_vgpr_msb 0x5945
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[230:231] /*v[486:487]*/, v[166:167] /*v[422:423]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[232:233] /*v[488:489]*/, v[166:167] /*v[422:423]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[234:235] /*v[490:491]*/, v[166:167] /*v[422:423]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[236:237] /*v[492:493]*/, v[166:167] /*v[422:423]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[510:517]*/, v[42:49] /*v[298:305]*/, v[94:101] /*v[606:613]*/, v[254:261] /*v[510:517]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[130:131] /*v[386:387]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[130:131] /*v[386:387]*/, v[240:241] /*v[496:497]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[130:131] /*v[386:387]*/, v[242:243] /*v[498:499]*/
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[130:131] /*v[386:387]*/, v[244:245] /*v[500:501]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[204:205] /*v[460:461]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x49a9
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[98:105] /*v[354:361]*/, v[38:45] /*v[550:557]*/, v[14:21] /*v[526:533]*/
	s_set_vgpr_msb 0xa945
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[130:131] /*v[386:387]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x4559
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[130:131] /*v[386:387]*/, v[0:1] /*v[512:513]*/
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[130:131] /*v[386:387]*/, v[2:3] /*v[514:515]*/
	v_pk_mul_f32 v[242:243] /*v[498:499]*/, v[130:131] /*v[386:387]*/, v[4:5] /*v[516:517]*/
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[210:211] /*v[466:467]*/, v[210:211] /*v[466:467]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[236:237] /*v[492:493]*/, v[236:237] /*v[492:493]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[246:253] /*v[502:509]*/, v[74:81] /*v[330:337]*/, v[70:77] /*v[582:589]*/, v[246:253] /*v[502:509]*/
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[240:241] /*v[496:497]*/, v[240:241] /*v[496:497]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[242:243] /*v[498:499]*/, v[242:243] /*v[498:499]*/, v[112:113] /*v[624:625]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[204:205] /*v[460:461]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[106:113] /*v[362:369]*/, v[70:77] /*v[582:589]*/, v[14:21] /*v[526:533]*/
	s_set_vgpr_msb 0xa959
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[210:211] /*v[466:467]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[236:237] /*v[492:493]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[238:239] /*v[494:495]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[240:241] /*v[496:497]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[242:243] /*v[498:499]*/, v[242:243] /*v[498:499]*/, s[6:7] op_sel_hi:[1,0]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[246:253] /*v[502:509]*/, v[82:89] /*v[338:345]*/, v[102:109] /*v[614:621]*/, v[246:253] /*v[502:509]*/
	v_exp_f32_e32 v178 /*v434*/, v178 /*v434*/
	v_exp_f32_e32 v179 /*v435*/, v179 /*v435*/
	v_exp_f32_e32 v190 /*v446*/, v190 /*v446*/
	v_exp_f32_e32 v191 /*v447*/, v191 /*v447*/
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	v_exp_f32_e32 v193 /*v449*/, v193 /*v449*/
	v_exp_f32_e32 v194 /*v450*/, v194 /*v450*/
	s_set_vgpr_msb 0x59a9
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[114:121] /*v[370:377]*/, v[102:109] /*v[614:621]*/, v[14:21] /*v[526:533]*/
	s_set_vgpr_msb 0xa949
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[252:253] /*v[508:509]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v252 /*v508*/, v174 /*v430*/
	v_exp_f32_e32 v253 /*v509*/, v175 /*v431*/
	v_exp_f32_e32 v174 /*v430*/, v176 /*v432*/
	v_exp_f32_e32 v175 /*v431*/, v177 /*v433*/
	v_exp_f32_e32 v176 /*v432*/, v180 /*v436*/
	v_exp_f32_e32 v177 /*v433*/, v181 /*v437*/
	v_exp_f32_e32 v195 /*v451*/, v195 /*v451*/
	v_exp_f32_e32 v180 /*v436*/, v196 /*v452*/
	v_exp_f32_e32 v181 /*v437*/, v197 /*v453*/
	v_exp_f32_e32 v196 /*v452*/, v204 /*v460*/
	v_exp_f32_e32 v197 /*v453*/, v205 /*v461*/
	v_exp_f32_e32 v204 /*v460*/, v206 /*v462*/
	v_exp_f32_e32 v205 /*v461*/, v207 /*v463*/
	v_exp_f32_e32 v206 /*v462*/, v208 /*v464*/
	v_exp_f32_e32 v207 /*v463*/, v209 /*v465*/
	v_exp_f32_e32 v208 /*v464*/, v210 /*v466*/
	v_exp_f32_e32 v209 /*v465*/, v211 /*v467*/
	v_exp_f32_e32 v210 /*v466*/, v236 /*v492*/
	v_exp_f32_e32 v211 /*v467*/, v237 /*v493*/
	v_exp_f32_e32 v236 /*v492*/, v238 /*v494*/
	v_exp_f32_e32 v237 /*v493*/, v239 /*v495*/
	v_exp_f32_e32 v238 /*v494*/, v240 /*v496*/
	v_exp_f32_e32 v239 /*v495*/, v241 /*v497*/
	v_exp_f32_e32 v240 /*v496*/, v242 /*v498*/
	v_exp_f32_e32 v241 /*v497*/, v243 /*v499*/
	v_pk_add_f32 v[212:213] /*v[468:469]*/, v[246:247] /*v[502:503]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[248:249] /*v[504:505]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[250:251] /*v[506:507]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[244:245] /*v[500:501]*/, v[14:15] /*v[526:527]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[16:17] /*v[528:529]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[18:19] /*v[530:531]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[250:251] /*v[506:507]*/, v[20:21] /*v[532:533]*/, v[114:115] /*v[626:627]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a45
	v_pk_mul_f32 v[242:243] /*v[498:499]*/, v[182:183] /*v[438:439]*/, v[252:253] /*v[508:509]*/
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[184:185] /*v[440:441]*/, v[174:175] /*v[430:431]*/
	s_set_vgpr_msb 0x4585
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[186:187] /*v[442:443]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[2:3] /*v[514:515]*/, v[188:189] /*v[444:445]*/, v[176:177] /*v[432:433]*/
	s_set_vgpr_msb 0x8545
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[166:167] /*v[422:423]*/, v[180:181] /*v[436:437]*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v175 /*v431*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v252 /*v508*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v180 /*v436*/, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v192 /*v448*/, v193 /*v449*/
	v_cvt_pk_bf16_f32 v178 /*v434*/, v190 /*v446*/, v191 /*v447*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[212:213] /*v[468:469]*/, v[196:197] /*v[452:453]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[230:231] /*v[486:487]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[232:233] /*v[488:489]*/, v[206:207] /*v[462:463]*/
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[234:235] /*v[490:491]*/, v[208:209] /*v[464:465]*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v208 /*v464*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v183 /*v439*/, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v196 /*v452*/, v197 /*v453*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[244:245] /*v[500:501]*/, v[210:211] /*v[466:467]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[246:247] /*v[502:503]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[248:249] /*v[504:505]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[250:251] /*v[506:507]*/, v[240:241] /*v[496:497]*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v210 /*v466*/, v211 /*v467*/
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[130:131] /*v[386:387]*/, v[242:243] /*v[498:499]*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[130:131] /*v[386:387]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x4549
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[130:131] /*v[386:387]*/, v[0:1] /*v[512:513]*/
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[130:131] /*v[386:387]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[130:131] /*v[386:387]*/, v[198:199] /*v[454:455]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[130:131] /*v[386:387]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[130:131] /*v[386:387]*/, v[202:203] /*v[458:459]*/
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[130:131] /*v[386:387]*/, v[166:167] /*v[422:423]*/
	ds_store_b128 v132 /*v388*/, v[174:177] /*v[430:433]*/
	ds_store_b128 v132 /*v388*/, v[178:181] /*v[434:437]*/ offset:32
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[130:131] /*v[386:387]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[130:131] /*v[386:387]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[130:131] /*v[386:387]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[130:131] /*v[386:387]*/, v[212:213] /*v[468:469]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[130:131] /*v[386:387]*/, v[196:197] /*v[452:453]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[130:131] /*v[386:387]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[130:131] /*v[386:387]*/, v[206:207] /*v[462:463]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[130:131] /*v[386:387]*/, v[208:209] /*v[464:465]*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v210 /*v466*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v175 /*v431*/, v230 /*v486*/, v231 /*v487*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v232 /*v488*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v178 /*v434*/, v198 /*v454*/, v199 /*v455*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v200 /*v456*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v180 /*v436*/, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v166 /*v422*/, v167 /*v423*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v240 /*v496*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v188 /*v444*/, v238 /*v494*/, v239 /*v495*/
	v_cvt_pk_bf16_f32 v187 /*v443*/, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v190 /*v446*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v192 /*v448*/, v193 /*v449*/
	v_cvt_pk_bf16_f32 v192 /*v448*/, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v193 /*v449*/, v212 /*v468*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v208 /*v464*/, v209 /*v465*/
	ds_store_b128 v132 /*v388*/, v[174:177] /*v[430:433]*/ offset:2560
	ds_store_b128 v132 /*v388*/, v[178:181] /*v[434:437]*/ offset:2592
	ds_store_b128 v173 /*v429*/, v[222:225] /*v[478:481]*/ offset:5120
	ds_store_b128 v173 /*v429*/, v[226:229] /*v[482:485]*/ offset:5152
	ds_store_b128 v173 /*v429*/, v[214:217] /*v[470:473]*/ offset:13824
	ds_store_b128 v173 /*v429*/, v[218:221] /*v[474:477]*/ offset:13856
	s_set_vgpr_msb 0x4509
	ds_store_b128 v173 /*v429*/, v[38:41] /*v[550:553]*/ offset:5184
	ds_store_b128 v173 /*v429*/, v[42:45] /*v[554:557]*/ offset:5216
	ds_store_b128 v173 /*v429*/, v[30:33] /*v[542:545]*/ offset:13888
	ds_store_b128 v173 /*v429*/, v[34:37] /*v[546:549]*/ offset:13920
	ds_store_b128 v173 /*v429*/, v[70:73] /*v[582:585]*/ offset:5248
	ds_store_b128 v173 /*v429*/, v[74:77] /*v[586:589]*/ offset:5280
	ds_store_b128 v173 /*v429*/, v[62:65] /*v[574:577]*/ offset:13952
	ds_store_b128 v173 /*v429*/, v[66:69] /*v[578:581]*/ offset:13984
	ds_store_b128 v173 /*v429*/, v[102:105] /*v[614:617]*/ offset:5312
	ds_store_b128 v173 /*v429*/, v[106:109] /*v[618:621]*/ offset:5344
	ds_store_b128 v173 /*v429*/, v[94:97] /*v[606:609]*/ offset:14016
	ds_store_b128 v173 /*v429*/, v[98:101] /*v[610:613]*/ offset:14048
	s_set_vgpr_msb 0x945
	ds_store_b128 v133 /*v389*/, v[182:185] /*v[438:441]*/
	ds_store_b128 v133 /*v389*/, v[186:189] /*v[442:445]*/ offset:32
	ds_store_b128 v133 /*v389*/, v[190:193] /*v[446:449]*/ offset:2560
	ds_store_b128 v133 /*v389*/, v[194:197] /*v[450:453]*/ offset:2592
	s_wait_dscnt 0x0
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v151 /*v407*/ offset:1280
	ds_load_tr16_b128 v[206:209] /*v[462:465]*/, v136 /*v392*/
	ds_load_tr16_b128 v[218:221] /*v[474:477]*/, v137 /*v393*/ offset:4352
	ds_load_tr16_b128 v[222:225] /*v[478:481]*/, v138 /*v394*/
	ds_load_tr16_b128 v[234:237] /*v[490:493]*/, v139 /*v395*/ offset:4352
	ds_load_tr16_b128 v[238:241] /*v[494:497]*/, v140 /*v396*/
	ds_load_tr16_b128 v[250:253] /*v[506:509]*/, v141 /*v397*/ offset:4352
	ds_load_tr16_b128 v[254:257] /*v[510:513]*/, v142 /*v398*/
	s_set_vgpr_msb 0x4581
	ds_load_tr16_b128 v[10:13] /*v[522:525]*/, v143 /*v399*/ offset:4352
	ds_load_tr16_b128 v[14:17] /*v[526:529]*/, v144 /*v400*/
	ds_load_tr16_b128 v[26:29] /*v[538:541]*/, v145 /*v401*/ offset:4352
	ds_load_tr16_b128 v[30:33] /*v[542:545]*/, v146 /*v402*/
	ds_load_tr16_b128 v[42:45] /*v[554:557]*/, v147 /*v403*/ offset:4352
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v148 /*v404*/
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v136 /*v392*/ offset:4352
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v137 /*v393*/
	ds_load_tr16_b128 v[226:229] /*v[482:485]*/, v138 /*v394*/ offset:4352
	ds_load_tr16_b128 v[230:233] /*v[486:489]*/, v139 /*v395*/
	ds_load_tr16_b128 v[242:245] /*v[498:501]*/, v140 /*v396*/ offset:4352
	ds_load_tr16_b128 v[246:249] /*v[502:505]*/, v141 /*v397*/
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v142 /*v398*/ offset:4352
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v143 /*v399*/
	ds_load_tr16_b128 v[18:21] /*v[530:533]*/, v144 /*v400*/ offset:4352
	ds_load_tr16_b128 v[22:25] /*v[534:537]*/, v145 /*v401*/
	ds_load_tr16_b128 v[34:37] /*v[546:549]*/, v146 /*v402*/ offset:4352
	ds_load_tr16_b128 v[38:41] /*v[550:553]*/, v147 /*v403*/
	ds_load_tr16_b128 v[50:53] /*v[562:565]*/, v148 /*v404*/ offset:4352
	ds_load_tr16_b128 v[54:57] /*v[566:569]*/, v149 /*v405*/
	s_set_vgpr_msb 0x8155
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v134 /*v390*/
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v134 /*v390*/ offset:4352
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v150 /*v406*/
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v150 /*v406*/ offset:1280
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v135 /*v391*/
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v135 /*v391*/ offset:4352
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v151 /*v407*/
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[182:189] /*v[438:445]*/, v[174:181] /*v[430:437]*/, v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[218:225], v[182:189] /*v[438:445]*/, v[206:213] /*v[462:469]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[202:209], v[182:189] /*v[438:445]*/, v[222:229] /*v[478:485]*/, v[202:209]
	v_wmma_f32_16x16x32_bf16 v[186:193], v[182:189] /*v[438:445]*/, v[238:245] /*v[494:501]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[170:177], v[182:189] /*v[438:445]*/, v[254:261] /*v[510:517]*/, v[170:177]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[154:161], v[182:189] /*v[438:445]*/, v[14:21] /*v[526:533]*/, v[154:161]
	v_wmma_f32_16x16x32_bf16 v[138:145], v[182:189] /*v[438:445]*/, v[30:37] /*v[542:549]*/, v[138:145]
	v_wmma_f32_16x16x32_bf16 v[130:137], v[182:189] /*v[438:445]*/, v[46:53] /*v[558:565]*/, v[130:137]
	s_set_vgpr_msb 0x981
	ds_load_tr16_b128 v[58:61] /*v[570:573]*/, v149 /*v405*/ offset:4352
	s_set_vgpr_msb 0x8155
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v152 /*v408*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[198:205] /*v[454:461]*/, v[190:197] /*v[446:453]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[198:205] /*v[454:461]*/, v[214:221] /*v[470:477]*/, v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[226:233], v[198:205] /*v[454:461]*/, v[230:237] /*v[486:493]*/, v[226:233]
	v_wmma_f32_16x16x32_bf16 v[210:217], v[198:205] /*v[454:461]*/, v[246:253] /*v[502:509]*/, v[210:217]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[194:201], v[198:205] /*v[454:461]*/, v[6:13] /*v[518:525]*/, v[194:201]
	v_wmma_f32_16x16x32_bf16 v[178:185], v[198:205] /*v[454:461]*/, v[22:29] /*v[534:541]*/, v[178:185]
	v_wmma_f32_16x16x32_bf16 v[162:169], v[198:205] /*v[454:461]*/, v[38:45] /*v[550:557]*/, v[162:169]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[146:153], v[198:205] /*v[454:461]*/, v[54:61] /*v[566:573]*/, v[146:153]
	s_set_vgpr_msb 0x941
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v152 /*v408*/ offset:1280
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v153 /*v409*/
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v153 /*v409*/ offset:1280
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121], v[182:189] /*v[438:445]*/, v[174:181] /*v[430:437]*/, v[114:121]
	v_wmma_f32_16x16x32_bf16 v[122:129], v[198:205] /*v[454:461]*/, v[190:197] /*v[446:453]*/, v[122:129]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[182:189] /*v[438:445]*/, v[206:213] /*v[462:469]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[106:113], v[198:205] /*v[454:461]*/, v[214:221] /*v[470:477]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[182:189] /*v[438:445]*/, v[222:229] /*v[478:485]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[98:105], v[198:205] /*v[454:461]*/, v[230:237] /*v[486:493]*/, v[98:105]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[182:189] /*v[438:445]*/, v[238:245] /*v[494:501]*/, v[58:65]
	v_wmma_f32_16x16x32_bf16 v[82:89], v[198:205] /*v[454:461]*/, v[246:253] /*v[502:509]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[182:189] /*v[438:445]*/, v[254:261] /*v[510:517]*/, v[42:49]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[66:73], v[198:205] /*v[454:461]*/, v[6:13] /*v[518:525]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[182:189] /*v[438:445]*/, v[14:21] /*v[526:533]*/, v[26:33]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[198:205] /*v[454:461]*/, v[22:29] /*v[534:541]*/, v[50:57]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[182:189] /*v[438:445]*/, v[30:37] /*v[542:549]*/, v[10:17]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[198:205] /*v[454:461]*/, v[38:45] /*v[550:557]*/, v[34:41]
	v_wmma_f32_16x16x32_bf16 v[2:9], v[182:189] /*v[438:445]*/, v[46:53] /*v[558:565]*/, v[2:9]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[198:205] /*v[454:461]*/, v[54:61] /*v[566:573]*/, v[18:25]
	s_set_vgpr_msb 0x900
	s_cbranch_scc1 .LBB0_6
.LBB0_7:
	s_clause 0x1
	s_load_b64 s[6:7], s[0:1], 0x110 nv
	s_load_b64 s[8:9], s[0:1], 0x140 nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x1f
	v_or_b32_e32 v235, 1, v170 /*v426*/
	v_mul_lo_u32 v234, s40, v170 /*v426*/
	s_mul_i32 s4, s40, s65
	s_lshl_b32 s1, s34, 25
	s_add_co_i32 s4, s4, s64
	v_mul_lo_u32 v235, v235, s40
	s_mov_b32 s0, 0
	s_wait_loadcnt 0x1e
	v_mul_lo_u32 v241, s40, v164 /*v420*/
	s_wait_loadcnt 0x1d
	v_mul_lo_u32 v243, s40, v163 /*v419*/
	v_add_lshl_u32 v234, v234, s4, 7
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v236, v34 /*v290*/, s0
	v_cvt_pk_bf16_f32 v238, v122 /*v378*/, s0
	s_mov_b32 s2, s46
	v_add_lshl_u32 v235, s4, v235, 7
	v_or_b32_e32 v237, v165 /*v421*/, v234
	s_mov_b32 s3, s47
	v_add_lshl_u32 v241, s4, v241, 7
	v_mul_lo_u32 v244, v162 /*v418*/, s40
	v_or_b32_e32 v239, v165 /*v421*/, v235
	s_wait_kmcnt 0x0
	s_or_b64 s[44:45], s[6:7], s[0:1]
	s_or_b64 s[0:1], s[8:9], s[0:1]
	v_lshlrev_b32_e32 v237, 2, v237
	s_wait_loadcnt 0x1c
	v_mul_lo_u32 v247, v161 /*v417*/, s40
	v_cvt_pk_bf16_f32 v240, v35 /*v291*/, s0
	v_cvt_pk_bf16_f32 v242, v123 /*v379*/, s0
	v_lshlrev_b32_e32 v239, 2, v239
	s_set_vgpr_msb 0x100
	buffer_store_b16 v236, v237, s[44:47], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v245, v37 /*v293*/, s0
	v_cvt_pk_bf16_f32 v246, v125 /*v381*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v236.l, v242.l
	buffer_store_b16 v238, v237, s[0:3], null offen
	buffer_store_b16 v240, v239, s[44:47], null offen
	s_wait_xcnt 0x1
	v_add_lshl_u32 v238, s4, v243, 7
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v242, v124 /*v380*/, s0
	v_add_lshl_u32 v244, s4, v244, 7
	s_set_vgpr_msb 0x104
	buffer_store_b16 v236, v239, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v236, v241, v165 /*v421*/
	v_or_b32_e32 v243, v238, v165 /*v421*/
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v240, v36 /*v292*/, s0
	v_mul_lo_u32 v248, v160 /*v416*/, s40
	s_wait_loadcnt 0x1b
	v_mul_lo_u32 v251, v159 /*v415*/, s40
	v_dual_lshlrev_b32 v236, 2, v236 :: v_dual_lshlrev_b32 v243, 2, v243
	v_cvt_pk_bf16_f32 v249, v39 /*v295*/, s0
	v_cvt_pk_bf16_f32 v250, v127 /*v383*/, s0
	s_set_vgpr_msb 0x100
	v_or_b32_e32 v235, v235, v0
	buffer_store_b16 v240, v236, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v240.l, v246.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v246, v126 /*v382*/, s0
	v_add_lshl_u32 v248, s4, v248, 7
	s_set_vgpr_msb 0x104
	buffer_store_b16 v242, v236, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v242, v244, v165 /*v421*/
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v253, v129 /*v385*/, s0
	s_set_vgpr_msb 0x100
	v_dual_lshlrev_b32 v235, 2, v235 :: v_dual_bitop2_b32 v234, v234, v0 bitop3:0x54
	buffer_store_b16 v245, v243, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v245, s4, v247, 7
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v252, v41 /*v297*/, s0
	v_lshlrev_b32_e32 v234, 2, v234
	s_set_vgpr_msb 0x100
	buffer_store_b16 v240, v243, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v240, v38 /*v294*/, s0
	v_or_b32_e32 v247, v165 /*v421*/, v245
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v218, v218, s0
	v_cvt_pk_bf16_f32 v219, v219, s0
	v_lshlrev_b32_e32 v242, 2, v242
	v_cvt_pk_bf16_f32 v220, v220, s0
	v_dual_lshlrev_b32 v247, 2, v247 :: v_dual_bitop2_b32 v244, v244, v0 bitop3:0x54
	v_cvt_pk_bf16_f32 v221, v221, s0
	buffer_store_b16 v240, v242, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v240.l, v250.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v250, v128 /*v384*/, s0
	s_set_vgpr_msb 0x104
	v_cvt_pk_bf16_f32 v222, v222, s0
	buffer_store_b16 v246, v242, s[0:3], null offen
	buffer_store_b16 v249, v247, s[44:47], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v246, v248, v165 /*v421*/
	s_wait_xcnt 0x0
	v_add_lshl_u32 v249, v251, s4, 7
	v_cvt_pk_bf16_f32 v223, v223, s0
	buffer_store_b16 v240, v247, s[0:3], null offen
	s_set_vgpr_msb 0x401
	v_cvt_pk_bf16_f32 v240, v40 /*v296*/, s0
	s_set_vgpr_msb 0x104
	v_cvt_pk_bf16_f32 v225, v225, s0
	v_or_b32_e32 v251, v249, v165 /*v421*/
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v246, 2, v246
	v_cvt_pk_bf16_f32 v202, v202, s0
	v_cvt_pk_bf16_f32 v203, v203, s0
	v_cvt_pk_bf16_f32 v204, v204, s0
	v_lshlrev_b32_e32 v251, 2, v251
	buffer_store_b16 v240, v246, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v240.l, v253.l
	buffer_store_b16 v250, v246, s[0:3], null offen
	buffer_store_b16 v252, v251, s[44:47], null offen
	v_or_b32_e32 v253, 64, v235
	s_wait_xcnt 0x1
	v_or_b32_e32 v250, 64, v234
	buffer_store_b16 v240, v251, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v240, v66 /*v322*/, s0
	v_cvt_pk_bf16_f32 v252, v67 /*v323*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v186, v186, s0
	buffer_store_b16 v218, v250, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v218, v241, v0
	buffer_store_b16 v240, v250, s[0:3], null offen
	buffer_store_b16 v219, v253, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v219, v238, v0
	v_mov_b16_e64 v241.l, v252.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v238, v68 /*v324*/, s0
	v_lshlrev_b32_e32 v218, 2, v218
	v_cvt_pk_bf16_f32 v250, v69 /*v325*/, s0
	v_lshlrev_b32_e32 v219, 2, v219
	s_set_vgpr_msb 0x100
	buffer_store_b16 v241, v253, s[0:3], null offen
	v_cvt_pk_bf16_f32 v187, v187, s0
	v_or_b32_e32 v240, 64, v218
	buffer_store_b16 v220, v240, s[44:47], null offen
	buffer_store_b16 v238, v240, s[0:3], null offen
	s_wait_xcnt 0x2
	v_or_b32_e32 v241, 64, v219
	s_wait_xcnt 0x0
	v_or_b32_e32 v238, v245, v0
	v_mov_b16_e64 v220.l, v250.l
	v_or_b32_e32 v245, v248, v0
	v_cvt_pk_bf16_f32 v189, v189, s0
	buffer_store_b16 v221, v241, s[44:47], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v221, 2, v244 :: v_dual_lshlrev_b32 v238, 2, v238
	buffer_store_b16 v220, v241, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v220, v70 /*v326*/, s0
	v_cvt_pk_bf16_f32 v241, v71 /*v327*/, s0
	v_or_b32_e32 v240, 64, v221
	v_or_b32_e32 v244, 64, v238
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v170, v170, s0
	v_cvt_pk_bf16_f32 v171, v171, s0
	v_cvt_pk_bf16_f32 v172, v172, s0
	buffer_store_b16 v222, v240, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v222.l, v241.l
	buffer_store_b16 v220, v240, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v220, 2, v245
	buffer_store_b16 v223, v244, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v223, v249, v0
	buffer_store_b16 v222, v244, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v222, v224, s0
	v_or_b32_e32 v224, 64, v220
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v240, v72 /*v328*/, s0
	v_lshlrev_b32_e32 v223, 2, v223
	s_set_vgpr_msb 0x100
	buffer_store_b16 v222, v224, s[44:47], null offen
	buffer_store_b16 v240, v224, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v224, v227, s0
	v_or_b32_e32 v244, 64, v223
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v241, v73 /*v329*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v225, v244, s[44:47], null offen
	buffer_store_b16 v241, v244, s[0:3], null offen
	v_cvt_pk_bf16_f32 v222, v226, s0
	buffer_store_b16 v202, v237, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v202.l, v224.l
	buffer_store_b16 v222, v237, s[0:3], null offen offset:128
	buffer_store_b16 v203, v239, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v203, v228, s0
	buffer_store_b16 v202, v239, s[0:3], null offen offset:128
	buffer_store_b16 v204, v236, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v202, v205, s0
	v_cvt_pk_bf16_f32 v205, v206, s0
	v_cvt_pk_bf16_f32 v206, v230, s0
	buffer_store_b16 v203, v236, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v204, v229, s0
	buffer_store_b16 v202, v243, s[44:47], null offen offset:128
	buffer_store_b16 v204, v243, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_mov_b16_e64 v203.l, v205.l
	v_mov_b16_e64 v205.l, v206.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v202, v207, s0
	buffer_store_b16 v203, v242, s[44:47], null offen offset:128
	buffer_store_b16 v205, v242, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v205, v232, s0
	v_cvt_pk_bf16_f32 v206, v209, s0
	v_cvt_pk_bf16_f32 v203, v231, s0
	buffer_store_b16 v202, v247, s[44:47], null offen offset:128
	v_cvt_pk_bf16_f32 v204, v208, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v202.l, v205.l
	v_mov_b16_e64 v205.l, v206.l
	buffer_store_b16 v203, v247, s[0:3], null offen offset:128
	buffer_store_b16 v204, v246, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v203, v233, s0
	buffer_store_b16 v202, v246, s[0:3], null offen offset:128
	buffer_store_b16 v205, v251, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v202, v210, s0
	v_or_b32_e32 v204, 0xc0, v234
	s_wait_xcnt 0x0
	v_or_b32_e32 v205, 0xc0, v235
	buffer_store_b16 v203, v251, s[0:3], null offen offset:128
	buffer_store_b16 v186, v204, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v186, v211, s0
	buffer_store_b16 v202, v204, s[0:3], null offen
	buffer_store_b16 v187, v205, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v204, v213, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v187, v188, s0
	v_or_b32_e32 v202, 0xc0, v218
	v_cvt_pk_bf16_f32 v188, v212, s0
	v_or_b32_e32 v203, 0xc0, v219
	buffer_store_b16 v186, v205, s[0:3], null offen
	buffer_store_b16 v187, v202, s[44:47], null offen
	buffer_store_b16 v188, v202, s[0:3], null offen
	buffer_store_b16 v189, v203, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e64 v186.l, v204.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v187, v190, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v189, 0xc0, v221
	v_cvt_pk_bf16_f32 v188, v214, s0
	v_cvt_pk_bf16_f32 v190, v191, s0
	v_or_b32_e32 v202, 0xc0, v238
	v_cvt_pk_bf16_f32 v191, v215, s0
	buffer_store_b16 v186, v203, s[0:3], null offen
	buffer_store_b16 v187, v189, s[44:47], null offen
	buffer_store_b16 v188, v189, s[0:3], null offen
	buffer_store_b16 v190, v202, s[44:47], null offen
	buffer_store_b16 v191, v202, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v186, v192, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v188, 0xc0, v220
	v_cvt_pk_bf16_f32 v187, v216, s0
	v_cvt_pk_bf16_f32 v189, v193, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v191, 0xc0, v223
	buffer_store_b16 v186, v188, s[44:47], null offen
	buffer_store_b16 v187, v188, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v187, v195, s0
	v_cvt_pk_bf16_f32 v190, v217, s0
	buffer_store_b16 v189, v191, s[44:47], null offen
	buffer_store_b16 v190, v191, s[0:3], null offen
	v_cvt_pk_bf16_f32 v186, v194, s0
	buffer_store_b16 v170, v237, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v187.l
	buffer_store_b16 v186, v237, s[0:3], null offen offset:256
	buffer_store_b16 v171, v239, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v171, v196, s0
	buffer_store_b16 v170, v239, s[0:3], null offen offset:256
	buffer_store_b16 v172, v236, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v170, v173, s0
	v_cvt_pk_bf16_f32 v173, v174, s0
	v_cvt_pk_bf16_f32 v174, v198, s0
	buffer_store_b16 v171, v236, s[0:3], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v172, v197, s0
	buffer_store_b16 v170, v243, s[44:47], null offen offset:256
	buffer_store_b16 v172, v243, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e64 v171.l, v173.l
	v_mov_b16_e64 v173.l, v174.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v170, v175, s0
	buffer_store_b16 v171, v242, s[44:47], null offen offset:256
	buffer_store_b16 v173, v242, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v173, v200, s0
	v_cvt_pk_bf16_f32 v174, v177, s0
	v_cvt_pk_bf16_f32 v171, v199, s0
	buffer_store_b16 v170, v247, s[44:47], null offen offset:256
	v_cvt_pk_bf16_f32 v172, v176, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v170.l, v173.l
	v_mov_b16_e64 v173.l, v174.l
	buffer_store_b16 v171, v247, s[0:3], null offen offset:256
	buffer_store_b16 v172, v246, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v171, v201, s0
	buffer_store_b16 v170, v246, s[0:3], null offen offset:256
	buffer_store_b16 v173, v251, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v170, v178, s0
	v_or_b32_e32 v172, 0x140, v234
	v_cvt_pk_bf16_f32 v154, v154, s0
	v_cvt_pk_bf16_f32 v155, v155, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v173, 0x140, v235
	buffer_store_b16 v171, v251, s[0:3], null offen offset:256
	buffer_store_b16 v154, v172, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v154, v179, s0
	buffer_store_b16 v170, v172, s[0:3], null offen
	buffer_store_b16 v155, v173, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v172, v181, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v155, v156, s0
	v_or_b32_e32 v170, 0x140, v218
	v_cvt_pk_bf16_f32 v156, v180, s0
	v_cvt_pk_bf16_f32 v157, v157, s0
	v_or_b32_e32 v171, 0x140, v219
	buffer_store_b16 v154, v173, s[0:3], null offen
	buffer_store_b16 v155, v170, s[44:47], null offen
	buffer_store_b16 v156, v170, s[0:3], null offen
	buffer_store_b16 v157, v171, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e64 v154.l, v172.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v155, v158, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v157, 0x140, v221
	v_cvt_pk_bf16_f32 v156, v182, s0
	v_cvt_pk_bf16_f32 v158, v159, s0
	v_or_b32_e32 v170, 0x140, v238
	v_cvt_pk_bf16_f32 v159, v183, s0
	buffer_store_b16 v154, v171, s[0:3], null offen
	buffer_store_b16 v155, v157, s[44:47], null offen
	buffer_store_b16 v156, v157, s[0:3], null offen
	buffer_store_b16 v158, v170, s[44:47], null offen
	buffer_store_b16 v159, v170, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v154, v160, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v156, 0x140, v220
	v_cvt_pk_bf16_f32 v155, v184, s0
	v_cvt_pk_bf16_f32 v157, v161, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v159, 0x140, v223
	buffer_store_b16 v154, v156, s[44:47], null offen
	buffer_store_b16 v155, v156, s[0:3], null offen
	v_cvt_pk_bf16_f32 v138, v138, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v155, v163, s0
	v_cvt_pk_bf16_f32 v158, v185, s0
	buffer_store_b16 v157, v159, s[44:47], null offen
	buffer_store_b16 v158, v159, s[0:3], null offen
	v_cvt_pk_bf16_f32 v154, v162, s0
	buffer_store_b16 v138, v237, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v138.l, v155.l
	v_cvt_pk_bf16_f32 v139, v139, s0
	v_cvt_pk_bf16_f32 v140, v140, s0
	buffer_store_b16 v154, v237, s[0:3], null offen offset:384
	buffer_store_b16 v139, v239, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v139, v164, s0
	buffer_store_b16 v138, v239, s[0:3], null offen offset:384
	buffer_store_b16 v140, v236, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v138, v141, s0
	v_cvt_pk_bf16_f32 v141, v142, s0
	v_cvt_pk_bf16_f32 v142, v166, s0
	buffer_store_b16 v139, v236, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v140, v165, s0
	buffer_store_b16 v138, v243, s[44:47], null offen offset:384
	buffer_store_b16 v140, v243, s[0:3], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e64 v139.l, v141.l
	v_mov_b16_e64 v141.l, v142.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v138, v143, s0
	buffer_store_b16 v139, v242, s[44:47], null offen offset:384
	buffer_store_b16 v141, v242, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v141, v168, s0
	v_cvt_pk_bf16_f32 v142, v145, s0
	v_cvt_pk_bf16_f32 v139, v167, s0
	v_cvt_pk_bf16_f32 v140, v144, s0
	buffer_store_b16 v138, v247, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v138.l, v141.l
	v_mov_b16_e64 v141.l, v142.l
	buffer_store_b16 v139, v247, s[0:3], null offen offset:384
	buffer_store_b16 v140, v246, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v139, v169, s0
	buffer_store_b16 v138, v246, s[0:3], null offen offset:384
	buffer_store_b16 v141, v251, s[44:47], null offen offset:384
	v_cvt_pk_bf16_f32 v130, v130, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v138, v146, s0
	v_or_b32_e32 v140, 0x1c0, v234
	v_cvt_pk_bf16_f32 v131, v131, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v141, 0x1c0, v235
	buffer_store_b16 v139, v251, s[0:3], null offen offset:384
	buffer_store_b16 v130, v140, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v130, v147, s0
	buffer_store_b16 v138, v140, s[0:3], null offen
	buffer_store_b16 v131, v141, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v131, v132, s0
	v_cvt_pk_bf16_f32 v132, v148, s0
	v_or_b32_e32 v138, 0x1c0, v218
	v_cvt_pk_bf16_f32 v133, v133, s0
	v_or_b32_e32 v139, 0x1c0, v219
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
	v_or_b32_e32 v133, 0x1c0, v221
	v_cvt_pk_bf16_f32 v132, v150, s0
	v_or_b32_e32 v138, 0x1c0, v238
	buffer_store_b16 v130, v139, s[0:3], null offen
	buffer_store_b16 v131, v133, s[44:47], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v130.l, v135.l
	buffer_store_b16 v132, v133, s[0:3], null offen
	buffer_store_b16 v134, v138, s[44:47], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v134, 1, v168 /*v424*/
	v_cvt_pk_bf16_f32 v131, v136, s0
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v133, 0x1c0, v220
	buffer_store_b16 v130, v138, s[0:3], null offen
	s_set_vgpr_msb 4
	v_mul_lo_u32 v130, s40, v168 /*v424*/
	v_mul_lo_u32 v134, v134, s40
	v_cvt_pk_bf16_f32 v135, v137, s0
	v_cvt_pk_bf16_f32 v137, v153, s0
	v_cvt_pk_bf16_f32 v132, v152, s0
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v136, 0x1c0, v223
	buffer_store_b16 v131, v133, s[44:47], null offen
	buffer_store_b16 v132, v133, s[0:3], null offen
	v_add_lshl_u32 v130, s4, v130, 7
	s_wait_xcnt 0x1
	v_mov_b16_e64 v131.l, v137.l
	s_wait_xcnt 0x0
	v_add_lshl_u32 v133, s4, v134, 7
	s_set_vgpr_msb 4
	v_mul_lo_u32 v134, s40, v158 /*v414*/
	buffer_store_b16 v135, v136, s[44:47], null offen
	v_or_b32_e32 v132, v130, v165 /*v421*/
	v_cvt_pk_bf16_f32 v114, v114, s0
	buffer_store_b16 v131, v136, s[0:3], null offen
	v_cvt_pk_bf16_f32 v123, v123, s0
	s_wait_xcnt 0x1
	v_mul_lo_u32 v135, s40, v157 /*v413*/
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v131, 2, v132
	s_set_vgpr_msb 4
	v_or_b32_e32 v132, v133, v165 /*v421*/
	v_cvt_pk_bf16_f32 v122, v122, s0
	v_cvt_pk_bf16_f32 v115, v115, s0
	v_cvt_pk_bf16_f32 v116, v116, s0
	buffer_store_b16 v114, v131, s[44:47], null offen
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v132, 2, v132
	v_mov_b16_e32 v114.l, v123.l
	v_add_lshl_u32 v123, s4, v134, 7
	s_set_vgpr_msb 4
	v_mul_lo_u32 v134, s40, v156 /*v412*/
	buffer_store_b16 v122, v131, s[0:3], null offen
	buffer_store_b16 v115, v132, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v115, v135, s4, 7
	buffer_store_b16 v114, v132, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v114, v123, v165 /*v421*/
	v_cvt_pk_bf16_f32 v122, v124, s0
	v_cvt_pk_bf16_f32 v125, v125, s0
	v_or_b32_e32 v124, v115, v165 /*v421*/
	v_add_lshl_u32 v134, v134, s4, 7
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v114, 2, v114
	s_set_vgpr_msb 4
	v_mul_lo_u32 v135, s40, v155 /*v411*/
	v_cvt_pk_bf16_f32 v117, v117, s0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v124, 2, v124
	v_mul_lo_u32 v1, s40, v1
	buffer_store_b16 v116, v114, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v116.l, v125.l
	buffer_store_b16 v122, v114, s[0:3], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v122, v134, v165 /*v421*/
	buffer_store_b16 v117, v124, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v117, v135, s4, 7
	buffer_store_b16 v116, v124, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v116, v118, s0
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v118, 2, v122
	v_cvt_pk_bf16_f32 v122, v126, s0
	s_set_vgpr_msb 4
	v_mul_lo_u32 v126, s40, v154 /*v410*/
	v_or_b32_e32 v125, v117, v165 /*v421*/
	v_cvt_pk_bf16_f32 v127, v127, s0
	v_cvt_pk_bf16_f32 v119, v119, s0
	v_add_lshl_u32 v1, v1, s4, 7
	buffer_store_b16 v116, v118, s[44:47], null offen
	s_set_vgpr_msb 0x400
	v_lshlrev_b32_e32 v125, 2, v125
	v_mov_b16_e32 v116.l, v127.l
	v_add_lshl_u32 v126, s4, v126, 7
	buffer_store_b16 v122, v118, s[0:3], null offen
	buffer_store_b16 v119, v125, s[44:47], null offen
	s_set_vgpr_msb 4
	v_or_b32_e32 v122, v1, v165 /*v421*/
	buffer_store_b16 v116, v125, s[0:3], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v119, v126, v165 /*v421*/
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v116, v120, s0
	v_cvt_pk_bf16_f32 v120, v128, s0
	v_cvt_pk_bf16_f32 v121, v121, s0
	s_set_vgpr_msb 0x400
	v_dual_lshlrev_b32 v122, 2, v122 :: v_dual_lshlrev_b32 v119, 2, v119
	v_cvt_pk_bf16_f32 v127, v129, s0
	v_cvt_pk_bf16_f32 v90, v90, s0
	v_cvt_pk_bf16_f32 v106, v106, s0
	v_cvt_pk_bf16_f32 v91, v91, s0
	buffer_store_b16 v116, v119, s[44:47], null offen
	buffer_store_b16 v120, v119, s[0:3], null offen
	buffer_store_b16 v121, v122, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v121, v133, v0
	v_or_b32_e32 v128, v130, v0
	v_mov_b16_e32 v116.l, v127.l
	v_cvt_pk_bf16_f32 v107, v107, s0
	v_cvt_pk_bf16_f32 v92, v92, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshlrev_b32 v121, 2, v121 :: v_dual_lshlrev_b32 v120, 2, v128
	buffer_store_b16 v116, v122, s[0:3], null offen
	v_cvt_pk_bf16_f32 v93, v93, s0
	v_cvt_pk_bf16_f32 v109, v109, s0
	v_or_b32_e32 v127, 64, v121
	s_wait_xcnt 0x0
	v_or_b32_e32 v116, 64, v120
	v_cvt_pk_bf16_f32 v94, v94, s0
	v_cvt_pk_bf16_f32 v95, v95, s0
	v_cvt_pk_bf16_f32 v75, v75, s0
	v_cvt_pk_bf16_f32 v76, v76, s0
	buffer_store_b16 v90, v116, s[44:47], null offen
	buffer_store_b16 v106, v116, s[0:3], null offen
	buffer_store_b16 v91, v127, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v91, v115, v0
	v_or_b32_e32 v115, v134, v0
	v_or_b32_e32 v90, v123, v0
	buffer_store_b16 v107, v127, s[0:3], null offen
	v_cvt_pk_bf16_f32 v106, v108, s0
	v_cvt_pk_bf16_f32 v59, v59, s0
	v_cvt_pk_bf16_f32 v43, v43, s0
	v_lshlrev_b32_e32 v90, 2, v90
	v_cvt_pk_bf16_f32 v44, v44, s0
	v_cvt_pk_bf16_f32 v27, v27, s0
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_cvt_pk_bf16_f32 v12, v12, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v107, 64, v90
	buffer_store_b16 v92, v107, s[44:47], null offen
	buffer_store_b16 v106, v107, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v106, v117, v0
	v_mov_b16_e32 v92.l, v109.l
	v_cvt_pk_bf16_f32 v3, v3, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v106, 2, v106 :: v_dual_lshlrev_b32 v91, 2, v91
	v_or_b32_e32 v109, 64, v106
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v108, 64, v91
	buffer_store_b16 v93, v108, s[44:47], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v93, 2, v115
	buffer_store_b16 v92, v108, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v92, v110, s0
	v_cvt_pk_bf16_f32 v108, v111, s0
	v_or_b32_e32 v110, v126, v0
	v_or_b32_e32 v107, 64, v93
	v_or_b32_e32 v0, v1, v0
	v_cvt_pk_bf16_f32 v1, v96, s0
	v_cvt_pk_bf16_f32 v96, v97, s0
	v_cvt_pk_bf16_f32 v97, v113, s0
	buffer_store_b16 v94, v107, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v94.l, v108.l
	buffer_store_b16 v92, v107, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v92, 2, v110
	buffer_store_b16 v95, v109, s[44:47], null offen
	v_lshlrev_b32_e32 v0, 2, v0
	buffer_store_b16 v94, v109, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v95, v112, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v94, 64, v92
	buffer_store_b16 v1, v94, s[44:47], null offen
	buffer_store_b16 v95, v94, s[0:3], null offen
	v_or_b32_e32 v107, 64, v0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v74, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v94, v99, s0
	buffer_store_b16 v96, v107, s[44:47], null offen
	buffer_store_b16 v97, v107, s[0:3], null offen
	v_cvt_pk_bf16_f32 v74, v98, s0
	buffer_store_b16 v1, v131, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v94.l
	buffer_store_b16 v74, v131, s[0:3], null offen offset:128
	buffer_store_b16 v75, v132, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v74, v100, s0
	buffer_store_b16 v1, v132, s[0:3], null offen offset:128
	buffer_store_b16 v76, v114, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v76, v78, s0
	v_cvt_pk_bf16_f32 v1, v77, s0
	v_cvt_pk_bf16_f32 v77, v102, s0
	buffer_store_b16 v74, v114, s[0:3], null offen offset:128
	v_cvt_pk_bf16_f32 v75, v101, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v74.l, v76.l
	buffer_store_b16 v1, v124, s[44:47], null offen offset:128
	buffer_store_b16 v75, v124, s[0:3], null offen offset:128
	v_mov_b16_e32 v76.l, v77.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v79, s0
	buffer_store_b16 v74, v118, s[44:47], null offen offset:128
	buffer_store_b16 v76, v118, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v76, v104, s0
	v_cvt_pk_bf16_f32 v77, v81, s0
	v_cvt_pk_bf16_f32 v74, v103, s0
	buffer_store_b16 v1, v125, s[44:47], null offen offset:128
	v_cvt_pk_bf16_f32 v75, v80, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v76.l
	v_mov_b16_e32 v76.l, v77.l
	buffer_store_b16 v74, v125, s[0:3], null offen offset:128
	buffer_store_b16 v75, v119, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v74, v105, s0
	buffer_store_b16 v1, v119, s[0:3], null offen offset:128
	buffer_store_b16 v76, v122, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v58, s0
	v_cvt_pk_bf16_f32 v58, v82, s0
	v_or_b32_e32 v75, 0xc0, v120
	s_wait_xcnt 0x0
	v_or_b32_e32 v76, 0xc0, v121
	buffer_store_b16 v74, v122, s[0:3], null offen offset:128
	buffer_store_b16 v1, v75, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v83, s0
	buffer_store_b16 v58, v75, s[0:3], null offen
	buffer_store_b16 v59, v76, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v75, v85, s0
	v_cvt_pk_bf16_f32 v58, v60, s0
	v_cvt_pk_bf16_f32 v60, v61, s0
	v_or_b32_e32 v61, 0xc0, v90
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v59, v84, s0
	v_or_b32_e32 v74, 0xc0, v91
	buffer_store_b16 v1, v76, s[0:3], null offen
	buffer_store_b16 v58, v61, s[44:47], null offen
	buffer_store_b16 v59, v61, s[0:3], null offen
	buffer_store_b16 v60, v74, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v1.l, v75.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v58, v62, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v60, 0xc0, v93
	v_cvt_pk_bf16_f32 v59, v86, s0
	v_cvt_pk_bf16_f32 v61, v63, s0
	v_or_b32_e32 v63, 0xc0, v106
	v_cvt_pk_bf16_f32 v62, v87, s0
	buffer_store_b16 v1, v74, s[0:3], null offen
	buffer_store_b16 v58, v60, s[44:47], null offen
	buffer_store_b16 v59, v60, s[0:3], null offen
	buffer_store_b16 v61, v63, s[44:47], null offen
	buffer_store_b16 v62, v63, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v1, v64, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v59, 0xc0, v92
	v_cvt_pk_bf16_f32 v58, v88, s0
	v_cvt_pk_bf16_f32 v60, v65, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v62, 0xc0, v0
	buffer_store_b16 v1, v59, s[44:47], null offen
	buffer_store_b16 v58, v59, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v42, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v58, v67, s0
	v_cvt_pk_bf16_f32 v61, v89, s0
	buffer_store_b16 v60, v62, s[44:47], null offen
	buffer_store_b16 v61, v62, s[0:3], null offen
	v_cvt_pk_bf16_f32 v42, v66, s0
	buffer_store_b16 v1, v131, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v58.l
	buffer_store_b16 v42, v131, s[0:3], null offen offset:256
	buffer_store_b16 v43, v132, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v42, v68, s0
	buffer_store_b16 v1, v132, s[0:3], null offen offset:256
	buffer_store_b16 v44, v114, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v44, v46, s0
	v_cvt_pk_bf16_f32 v1, v45, s0
	v_cvt_pk_bf16_f32 v45, v70, s0
	buffer_store_b16 v42, v114, s[0:3], null offen offset:256
	v_cvt_pk_bf16_f32 v43, v69, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v42.l, v44.l
	buffer_store_b16 v1, v124, s[44:47], null offen offset:256
	buffer_store_b16 v43, v124, s[0:3], null offen offset:256
	v_mov_b16_e32 v44.l, v45.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v47, s0
	buffer_store_b16 v42, v118, s[44:47], null offen offset:256
	buffer_store_b16 v44, v118, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v44, v72, s0
	v_cvt_pk_bf16_f32 v45, v49, s0
	v_cvt_pk_bf16_f32 v42, v71, s0
	buffer_store_b16 v1, v125, s[44:47], null offen offset:256
	v_cvt_pk_bf16_f32 v43, v48, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v44.l
	v_mov_b16_e32 v44.l, v45.l
	buffer_store_b16 v42, v125, s[0:3], null offen offset:256
	buffer_store_b16 v43, v119, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v42, v73, s0
	buffer_store_b16 v1, v119, s[0:3], null offen offset:256
	buffer_store_b16 v44, v122, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v26, s0
	v_cvt_pk_bf16_f32 v26, v50, s0
	v_or_b32_e32 v43, 0x140, v120
	s_wait_xcnt 0x0
	v_or_b32_e32 v44, 0x140, v121
	buffer_store_b16 v42, v122, s[0:3], null offen offset:256
	buffer_store_b16 v1, v43, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v51, s0
	buffer_store_b16 v26, v43, s[0:3], null offen
	buffer_store_b16 v27, v44, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v43, v53, s0
	v_cvt_pk_bf16_f32 v26, v28, s0
	v_cvt_pk_bf16_f32 v28, v29, s0
	v_or_b32_e32 v29, 0x140, v90
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v27, v52, s0
	v_or_b32_e32 v42, 0x140, v91
	buffer_store_b16 v1, v44, s[0:3], null offen
	buffer_store_b16 v26, v29, s[44:47], null offen
	buffer_store_b16 v27, v29, s[0:3], null offen
	buffer_store_b16 v28, v42, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v1.l, v43.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v26, v30, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v28, 0x140, v93
	v_cvt_pk_bf16_f32 v27, v54, s0
	v_cvt_pk_bf16_f32 v29, v31, s0
	v_or_b32_e32 v31, 0x140, v106
	v_cvt_pk_bf16_f32 v30, v55, s0
	buffer_store_b16 v1, v42, s[0:3], null offen
	buffer_store_b16 v26, v28, s[44:47], null offen
	buffer_store_b16 v27, v28, s[0:3], null offen
	buffer_store_b16 v29, v31, s[44:47], null offen
	buffer_store_b16 v30, v31, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v1, v32, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v27, 0x140, v92
	v_cvt_pk_bf16_f32 v26, v56, s0
	v_cvt_pk_bf16_f32 v28, v33, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v30, 0x140, v0
	buffer_store_b16 v1, v27, s[44:47], null offen
	buffer_store_b16 v26, v27, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v10, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v26, v35, s0
	v_cvt_pk_bf16_f32 v29, v57, s0
	buffer_store_b16 v28, v30, s[44:47], null offen
	buffer_store_b16 v29, v30, s[0:3], null offen
	v_cvt_pk_bf16_f32 v10, v34, s0
	buffer_store_b16 v1, v131, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v26.l
	buffer_store_b16 v10, v131, s[0:3], null offen offset:384
	buffer_store_b16 v11, v132, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v10, v36, s0
	buffer_store_b16 v1, v132, s[0:3], null offen offset:384
	buffer_store_b16 v12, v114, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v12, v14, s0
	v_cvt_pk_bf16_f32 v1, v13, s0
	v_cvt_pk_bf16_f32 v13, v38, s0
	buffer_store_b16 v10, v114, s[0:3], null offen offset:384
	v_cvt_pk_bf16_f32 v11, v37, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v10.l, v12.l
	buffer_store_b16 v1, v124, s[44:47], null offen offset:384
	buffer_store_b16 v11, v124, s[0:3], null offen offset:384
	v_mov_b16_e32 v12.l, v13.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v15, s0
	buffer_store_b16 v10, v118, s[44:47], null offen offset:384
	buffer_store_b16 v12, v118, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v12, v40, s0
	v_cvt_pk_bf16_f32 v13, v17, s0
	v_cvt_pk_bf16_f32 v10, v39, s0
	buffer_store_b16 v1, v125, s[44:47], null offen offset:384
	v_cvt_pk_bf16_f32 v11, v16, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v12.l
	v_mov_b16_e32 v12.l, v13.l
	buffer_store_b16 v10, v125, s[0:3], null offen offset:384
	buffer_store_b16 v11, v119, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v10, v41, s0
	buffer_store_b16 v1, v119, s[0:3], null offen offset:384
	buffer_store_b16 v12, v122, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v2, s0
	v_cvt_pk_bf16_f32 v2, v18, s0
	v_or_b32_e32 v11, 0x1c0, v120
	s_wait_xcnt 0x0
	v_or_b32_e32 v12, 0x1c0, v121
	buffer_store_b16 v10, v122, s[0:3], null offen offset:384
	buffer_store_b16 v1, v11, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v19, s0
	buffer_store_b16 v2, v11, s[0:3], null offen
	buffer_store_b16 v3, v12, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v2, v4, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v20, s0
	v_cvt_pk_bf16_f32 v4, v5, s0
	v_or_b32_e32 v5, 0x1c0, v90
	v_or_b32_e32 v10, 0x1c0, v91
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
	v_or_b32_e32 v5, 0x1c0, v93
	v_cvt_pk_bf16_f32 v3, v22, s0
	v_or_b32_e32 v6, 0x1c0, v106
	buffer_store_b16 v1, v10, s[0:3], null offen
	buffer_store_b16 v2, v5, s[44:47], null offen
	buffer_store_b16 v3, v5, s[0:3], null offen
	buffer_store_b16 v4, v6, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v1.l, v7.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v2, v8, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v4, 0x1c0, v92
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
		.amdhsa_next_free_vgpr 632
		.amdhsa_next_free_sgpr 76
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

	.set .Lk_dkdv_0.num_vgpr, 632
	.set .Lk_dkdv_0.num_agpr, 0
	.set .Lk_dkdv_0.numbered_sgpr, 76
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
    .max_flat_workgroup_size: 32
    .name:           k_dkdv_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     78
    .sgpr_spill_count: 0
    .symbol:         k_dkdv_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     632
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
