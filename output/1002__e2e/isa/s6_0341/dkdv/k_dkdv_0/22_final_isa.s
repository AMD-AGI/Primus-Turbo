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
	s_load_b256 s[12:19], s[0:1], 0x170 nv
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
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v13 /*v525*/, 15, v0
	s_cselect_b32 s8, s6, s5
	s_cselect_b32 s4, s3, s4
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_set_vgpr_msb 0x8000
	v_lshrrev_b32_e32 v3, 4, v0
	s_mul_i32 s2, ttmp9, s2
	s_load_b64 s[28:29], s[0:1], 0x30 nv
	s_add_co_i32 s5, s3, s2
	s_cmp_eq_u32 s7, 0
	s_load_b64 s[2:3], s[0:1], 0x190 nv
	s_cselect_b32 s10, ttmp9, s5
	s_wait_kmcnt 0x0
	s_lshr_b32 s5, s18, 31
	s_lshl_b32 s9, s4, 5
	s_add_co_i32 s5, s18, s5
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, s9, v13 /*v525*/
	s_and_b32 s4, s5, -2
	s_ashr_i32 s5, s5, 1
	s_cmp_lg_u32 s18, s4
	s_mov_b32 s45, 0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s18, 0
	s_set_vgpr_msb 0x880
	v_and_b32_e32 v12 /*v524*/, 16, v0
	s_cselect_b32 s6, -1, 0
	s_and_b32 s4, s6, s4
	s_sub_co_ci_u32 s62, s5, 0
	s_sub_co_i32 s6, s9, s19
	s_max_i32 s6, s6, 0
	s_lshr_b32 s6, s6, 5
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s7, -1, 0
	s_and_b32 s7, s7, exec_lo
	s_cselect_b32 s18, s6, 0
	s_cmp_lg_u32 s4, 0
	s_sub_co_ci_u32 s60, s5, s18
	s_or_b32 s4, s9, 31
	s_sub_co_i32 s4, s4, s19
	s_add_co_i32 s5, s4, 31
	s_ashr_i32 s6, s5, 31
	s_lshr_b32 s6, s6, 27
	s_add_co_i32 s6, s5, s6
	s_and_b32 s7, s6, 0xffffffe0
	s_ashr_i32 s6, s6, 5
	s_cmp_lg_u32 s5, s7
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s5, 0
	s_cselect_b32 s5, -1, 0
	s_and_b32 s5, s5, s7
	s_sub_co_ci_u32 s5, s6, 0
	s_cmp_gt_i32 s4, -1
	s_cselect_b32 s4, s5, 0
	s_min_i32 s4, s4, s62
	s_sub_co_i32 s4, s4, s18
	s_max_i32 s4, s4, 0
	s_min_i32 s4, s4, s60
	s_cmp_lg_u32 s2, 0
	s_mul_i32 s2, s14, s8
	s_cselect_b32 s64, -1, 0
	s_and_b32 s5, s64, exec_lo
	s_cselect_b32 s61, s4, 0
	s_lshl_b32 s4, s16, 4
	s_or_b32 s11, s9, 16
	s_mul_i32 s2, s2, s4
	s_mul_i32 s5, s14, s16
	s_lshl4_add_u32 s2, s10, s2
	s_mul_i32 s5, s5, s3
	s_set_vgpr_msb 0x8000
	v_mad_u32 v1, s4, v1, s2
	s_lshl_b32 s34, s5, 8
	s_lshl_b32 s33, s13, 2
	s_ashr_i32 s35, s34, 31
	s_mul_i32 s56, s61, s17
	s_lshr_b64 s[30:31], s[34:35], 7
	s_mov_b32 s16, -1
	s_mov_b32 s22, s30
	v_or_b32_e32 v1, v1, v3
	s_mov_b32 s23, s31
	s_set_vgpr_msb 0x80
	v_lshlrev_b32_e32 v7 /*v519*/, 4, v1
	s_set_vgpr_msb 0x8008
	v_or_b32_e32 v2, s11, v13 /*v525*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v1, 1, v0
	s_set_vgpr_msb 0xa8
	v_mad_u32_u24 v14 /*v526*/, 0x110, v13 /*v525*/, v12 /*v524*/
	s_set_vgpr_msb 0xa800
	v_mad_u32 v2, s4, v2, s2
	s_clause 0x2
	s_load_b64 s[20:21], s[0:1], 0x60 nv
	s_load_b64 s[4:5], s[0:1], 0xc0 nv
	s_load_b64 s[6:7], s[0:1], 0xe8 nv
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v11 /*v523*/, 16, v1
	s_mul_i32 s2, s33, s15
	s_mul_i32 s2, s2, s3
	s_ashr_i32 s3, s2, 31
	s_set_vgpr_msb 0x8000
	v_or_b32_e32 v2, v2, v3
	s_lshr_b64 s[38:39], s[2:3], 7
	s_lshl_b32 s3, s2, 25
	s_mov_b32 s2, s45
	s_set_vgpr_msb 0x80
	v_lshlrev_b32_e32 v5 /*v517*/, 4, v2
	s_set_vgpr_msb 0x8002
	s_clause 0xf
	buffer_load_b128 v[34:37], v7 /*v519*/, s[28:31], null offen
	buffer_load_b128 v[38:41], v7 /*v519*/, s[28:31], null offen offset:32
	buffer_load_b128 v[42:45], v7 /*v519*/, s[28:31], null offen offset:64
	buffer_load_b128 v[46:49], v7 /*v519*/, s[28:31], null offen offset:96
	buffer_load_b128 v[50:53], v7 /*v519*/, s[28:31], null offen offset:128
	buffer_load_b128 v[54:57], v7 /*v519*/, s[28:31], null offen offset:160
	buffer_load_b128 v[58:61], v7 /*v519*/, s[28:31], null offen offset:192
	buffer_load_b128 v[62:65], v7 /*v519*/, s[28:31], null offen offset:224
	buffer_load_b128 v[74:77], v5 /*v517*/, s[28:31], null offen
	buffer_load_b128 v[78:81], v5 /*v517*/, s[28:31], null offen offset:32
	buffer_load_b128 v[82:85], v5 /*v517*/, s[28:31], null offen offset:64
	buffer_load_b128 v[86:89], v5 /*v517*/, s[28:31], null offen offset:96
	buffer_load_b128 v[90:93], v5 /*v517*/, s[28:31], null offen offset:128
	buffer_load_b128 v[94:97], v5 /*v517*/, s[28:31], null offen offset:160
	buffer_load_b128 v[98:101], v5 /*v517*/, s[28:31], null offen offset:192
	buffer_load_b128 v[102:105], v5 /*v517*/, s[28:31], null offen offset:224
	s_wait_kmcnt 0x0
	s_clause 0xf
	buffer_load_b128 v[106:109], v7 /*v519*/, s[20:23], null offen
	buffer_load_b128 v[110:113], v7 /*v519*/, s[20:23], null offen offset:32
	buffer_load_b128 v[122:125], v7 /*v519*/, s[20:23], null offen offset:64
	buffer_load_b128 v[126:129], v7 /*v519*/, s[20:23], null offen offset:96
	buffer_load_b128 v[130:133], v7 /*v519*/, s[20:23], null offen offset:128
	buffer_load_b128 v[134:137], v7 /*v519*/, s[20:23], null offen offset:160
	buffer_load_b128 v[138:141], v7 /*v519*/, s[20:23], null offen offset:192
	buffer_load_b128 v[142:145], v7 /*v519*/, s[20:23], null offen offset:224
	buffer_load_b128 v[146:149], v5 /*v517*/, s[20:23], null offen
	buffer_load_b128 v[150:153], v5 /*v517*/, s[20:23], null offen offset:32
	buffer_load_b128 v[154:157], v5 /*v517*/, s[20:23], null offen offset:64
	buffer_load_b128 v[158:161], v5 /*v517*/, s[20:23], null offen offset:96
	buffer_load_b128 v[162:165], v5 /*v517*/, s[20:23], null offen offset:128
	buffer_load_b128 v[166:169], v5 /*v517*/, s[20:23], null offen offset:160
	buffer_load_b128 v[170:173], v5 /*v517*/, s[20:23], null offen offset:192
	buffer_load_b128 v[174:177], v5 /*v517*/, s[20:23], null offen offset:224
	v_lshlrev_b32_e32 v2, 3, v3
	s_wait_xcnt 0x10
	s_ashr_i32 s29, s15, 31
	s_mov_b32 s28, s15
	s_cmp_gt_i32 s56, 0
	s_set_vgpr_msb 0x280
	v_and_or_b32 v9 /*v521*/, v0, 7, v2
	s_set_vgpr_msb 0x80a8
	v_mad_u32_u24 v15 /*v527*/, 0x110, v9 /*v521*/, v11 /*v523*/
	s_set_vgpr_msb 0xa800
	s_cbranch_scc1 .LBB0_2
	s_mov_b32 s16, 0
.LBB0_2:
	s_clause 0x1
	s_load_b64 s[52:53], s[0:1], 0x0 nv
	s_load_b64 s[54:55], s[0:1], 0x90 nv
	s_set_vgpr_msb 0x80
	v_or_b32_e32 v16 /*v528*/, 16, v0
	s_or_b64 s[36:37], s[4:5], s[2:3]
	s_or_b64 s[40:41], s[6:7], s[2:3]
	s_lshl_b32 s63, s15, 7
	s_and_b32 s2, s16, exec_lo
	s_set_vgpr_msb 0x80a8
	v_mul_i32_i24_e32 v18 /*v530*/, 0xffffff40, v13 /*v525*/
	v_mad_u32_u24 v2 /*v514*/, 0x50, v16 /*v528*/, v12 /*v524*/
	v_mul_u32_u24_e32 v3 /*v515*/, 0x50, v9 /*v521*/
	v_or_b32_e32 v17 /*v529*/, 32, v11 /*v523*/
	s_cselect_b32 s2, 1, 0
	s_mul_i32 s14, s17, s10
	s_mul_i32 s15, s15, s8
	s_cmp_lg_u32 s2, 1
	s_mul_i32 s35, s13, s8
	s_set_vgpr_msb 0xa800
	s_cbranch_scc1 .LBB0_5
	s_mov_b32 s3, 0x10a00
	s_set_vgpr_msb 64
	v_dual_mov_b32 v50 /*v306*/, 0 :: v_dual_bitop2_b32 v141 /*v397*/, s9, v2 bitop3:0x54
	s_set_vgpr_msb 0x4008
	v_mad_u32_u24 v3, 0x50, v9 /*v521*/, s3
	s_ashr_i32 s57, s56, 31
	s_cmp_lg_u32 s63, 0x80000000
	s_mov_b32 s4, s12
	s_mov_b32 s5, s12
	s_cselect_b32 s25, s63, 0x80
	s_set_vgpr_msb 0x84a
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, s[4:5]
	v_add3_u32 v145 /*v401*/, v14 /*v526*/, v18 /*v530*/, 0x10000
	v_or_b32_e32 v146 /*v402*/, 0x10000, v2 /*v514*/
	s_set_vgpr_msb 0x4a40
	v_or_b32_e32 v144 /*v400*/, s11, v2
	s_set_vgpr_msb 0x4008
	v_or_b32_e32 v2, 0x10000, v3 /*v515*/
	s_set_vgpr_msb 0x848
	v_add_nc_u32_e32 v148 /*v404*/, v3, v11 /*v523*/
	s_set_vgpr_msb 0x4805
	v_dual_mov_b32 v250, v50 /*v306*/ :: v_dual_bitop2_b32 v1, 3, v141 /*v397*/ bitop3:0x54
	s_set_vgpr_msb 0x544
	v_or_b32_e32 v130 /*v386*/, 2, v141 /*v397*/
	v_or_b32_e32 v131 /*v387*/, 5, v141 /*v397*/
	v_or_b32_e32 v132 /*v388*/, 4, v141 /*v397*/
	v_or_b32_e32 v133 /*v389*/, 7, v141 /*v397*/
	v_or_b32_e32 v134 /*v390*/, 6, v141 /*v397*/
	s_set_vgpr_msb 0x4448
	v_dual_add_nc_u32 v147 /*v403*/, v2, v11 /*v523*/ :: v_dual_add_nc_u32 v149 /*v405*/, v2, v17 /*v529*/
	v_add_nc_u32_e32 v150 /*v406*/, v3, v17 /*v529*/
	s_set_vgpr_msb 0x4845
	v_dual_mov_b32 v51 /*v307*/, v50 /*v306*/ :: v_dual_mov_b32 v52 /*v308*/, v50 /*v306*/
	v_dual_mov_b32 v53 /*v309*/, v50 /*v306*/ :: v_dual_bitop2_b32 v140 /*v396*/, 6, v144 /*v400*/ bitop3:0x54
	v_dual_mov_b32 v62 /*v318*/, v50 /*v306*/ :: v_dual_bitop2_b32 v135 /*v391*/, 3, v144 /*v400*/ bitop3:0x54
	v_or_b32_e32 v136 /*v392*/, 2, v144 /*v400*/
	v_or_b32_e32 v137 /*v393*/, 5, v144 /*v400*/
	v_or_b32_e32 v138 /*v394*/, 4, v144 /*v400*/
	v_dual_mov_b32 v54 /*v310*/, v50 /*v306*/ :: v_dual_bitop2_b32 v139 /*v395*/, 7, v144 /*v400*/ bitop3:0x54
	v_dual_mov_b32 v55 /*v311*/, v50 /*v306*/ :: v_dual_mov_b32 v56 /*v312*/, v50 /*v306*/
	v_dual_mov_b32 v57 /*v313*/, v50 /*v306*/ :: v_dual_mov_b32 v58 /*v314*/, v50 /*v306*/
	v_dual_mov_b32 v59 /*v315*/, v50 /*v306*/ :: v_dual_mov_b32 v60 /*v316*/, v50 /*v306*/
	v_dual_mov_b32 v61 /*v317*/, v50 /*v306*/ :: v_dual_mov_b32 v63 /*v319*/, v50 /*v306*/
	v_dual_mov_b32 v64 /*v320*/, v50 /*v306*/ :: v_dual_mov_b32 v65 /*v321*/, v50 /*v306*/
	v_dual_mov_b32 v42 /*v298*/, v50 /*v306*/ :: v_dual_mov_b32 v43 /*v299*/, v50 /*v306*/
	v_dual_mov_b32 v44 /*v300*/, v50 /*v306*/ :: v_dual_mov_b32 v45 /*v301*/, v50 /*v306*/
	v_dual_mov_b32 v46 /*v302*/, v50 /*v306*/ :: v_dual_mov_b32 v47 /*v303*/, v50 /*v306*/
	v_dual_mov_b32 v48 /*v304*/, v50 /*v306*/ :: v_dual_mov_b32 v49 /*v305*/, v50 /*v306*/
	v_dual_mov_b32 v34 /*v290*/, v50 /*v306*/ :: v_dual_mov_b32 v35 /*v291*/, v50 /*v306*/
	v_dual_mov_b32 v36 /*v292*/, v50 /*v306*/ :: v_dual_mov_b32 v37 /*v293*/, v50 /*v306*/
	v_dual_mov_b32 v38 /*v294*/, v50 /*v306*/ :: v_dual_mov_b32 v39 /*v295*/, v50 /*v306*/
	v_dual_mov_b32 v40 /*v296*/, v50 /*v306*/ :: v_dual_mov_b32 v41 /*v297*/, v50 /*v306*/
	v_dual_mov_b32 v26 /*v282*/, v50 /*v306*/ :: v_dual_mov_b32 v27 /*v283*/, v50 /*v306*/
	v_dual_mov_b32 v28 /*v284*/, v50 /*v306*/ :: v_dual_mov_b32 v29 /*v285*/, v50 /*v306*/
	v_dual_mov_b32 v30 /*v286*/, v50 /*v306*/ :: v_dual_mov_b32 v31 /*v287*/, v50 /*v306*/
	v_dual_mov_b32 v32 /*v288*/, v50 /*v306*/ :: v_dual_mov_b32 v33 /*v289*/, v50 /*v306*/
	v_dual_mov_b32 v10 /*v266*/, v50 /*v306*/ :: v_dual_mov_b32 v11 /*v267*/, v50 /*v306*/
	v_dual_mov_b32 v12 /*v268*/, v50 /*v306*/ :: v_dual_mov_b32 v13 /*v269*/, v50 /*v306*/
	v_dual_mov_b32 v14 /*v270*/, v50 /*v306*/ :: v_dual_mov_b32 v15 /*v271*/, v50 /*v306*/
	v_dual_mov_b32 v16 /*v272*/, v50 /*v306*/ :: v_dual_mov_b32 v17 /*v273*/, v50 /*v306*/
	v_mov_b32_e32 v0 /*v256*/, v50 /*v306*/
	s_set_vgpr_msb 0x4501
	v_dual_mov_b32 v251, v50 /*v306*/ :: v_dual_mov_b32 v252, v50 /*v306*/
	v_dual_mov_b32 v253, v50 /*v306*/ :: v_dual_mov_b32 v254, v50 /*v306*/
	v_dual_mov_b32 v255, v50 /*v306*/ :: v_dual_mov_b32 v226, v50 /*v306*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v1 /*v257*/, v50 /*v306*/ :: v_dual_mov_b32 v122 /*v378*/, v50 /*v306*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v227, v50 /*v306*/ :: v_dual_mov_b32 v228, v50 /*v306*/
	v_dual_mov_b32 v229, v50 /*v306*/ :: v_dual_mov_b32 v230, v50 /*v306*/
	v_dual_mov_b32 v231, v50 /*v306*/ :: v_dual_mov_b32 v232, v50 /*v306*/
	v_dual_mov_b32 v233, v50 /*v306*/ :: v_dual_mov_b32 v186, v50 /*v306*/
	v_dual_mov_b32 v187, v50 /*v306*/ :: v_dual_mov_b32 v188, v50 /*v306*/
	v_dual_mov_b32 v189, v50 /*v306*/ :: v_dual_mov_b32 v190, v50 /*v306*/
	v_dual_mov_b32 v191, v50 /*v306*/ :: v_dual_mov_b32 v192, v50 /*v306*/
	v_dual_mov_b32 v193, v50 /*v306*/ :: v_dual_mov_b32 v178, v50 /*v306*/
	v_dual_mov_b32 v179, v50 /*v306*/ :: v_dual_mov_b32 v180, v50 /*v306*/
	v_dual_mov_b32 v181, v50 /*v306*/ :: v_dual_mov_b32 v182, v50 /*v306*/
	v_dual_mov_b32 v183, v50 /*v306*/ :: v_dual_mov_b32 v184, v50 /*v306*/
	v_dual_mov_b32 v185, v50 /*v306*/ :: v_dual_mov_b32 v114, v50 /*v306*/
	v_dual_mov_b32 v115, v50 /*v306*/ :: v_dual_mov_b32 v116, v50 /*v306*/
	v_dual_mov_b32 v117, v50 /*v306*/ :: v_dual_mov_b32 v118, v50 /*v306*/
	v_dual_mov_b32 v119, v50 /*v306*/ :: v_dual_mov_b32 v120, v50 /*v306*/
	v_dual_mov_b32 v121, v50 /*v306*/ :: v_dual_mov_b32 v66, v50 /*v306*/
	v_dual_mov_b32 v67, v50 /*v306*/ :: v_dual_mov_b32 v68, v50 /*v306*/
	v_dual_mov_b32 v69, v50 /*v306*/ :: v_dual_mov_b32 v70, v50 /*v306*/
	v_dual_mov_b32 v71, v50 /*v306*/ :: v_dual_mov_b32 v72, v50 /*v306*/
	v_dual_mov_b32 v73, v50 /*v306*/ :: v_dual_mov_b32 v26, v50 /*v306*/
	v_dual_mov_b32 v27, v50 /*v306*/ :: v_dual_mov_b32 v28, v50 /*v306*/
	v_dual_mov_b32 v29, v50 /*v306*/ :: v_dual_mov_b32 v30, v50 /*v306*/
	v_dual_mov_b32 v31, v50 /*v306*/ :: v_dual_mov_b32 v32, v50 /*v306*/
	v_dual_mov_b32 v33, v50 /*v306*/ :: v_dual_mov_b32 v18, v50 /*v306*/
	v_dual_mov_b32 v19, v50 /*v306*/ :: v_dual_mov_b32 v20, v50 /*v306*/
	v_dual_mov_b32 v21, v50 /*v306*/ :: v_dual_mov_b32 v22, v50 /*v306*/
	v_dual_mov_b32 v23, v50 /*v306*/ :: v_dual_mov_b32 v24, v50 /*v306*/
	v_dual_mov_b32 v25, v50 /*v306*/ :: v_dual_mov_b32 v10, v50 /*v306*/
	v_dual_mov_b32 v11, v50 /*v306*/ :: v_dual_mov_b32 v12, v50 /*v306*/
	v_dual_mov_b32 v13, v50 /*v306*/ :: v_dual_mov_b32 v14, v50 /*v306*/
	v_dual_mov_b32 v15, v50 /*v306*/ :: v_dual_mov_b32 v16, v50 /*v306*/
	v_dual_mov_b32 v17, v50 /*v306*/ :: v_dual_mov_b32 v2, v50 /*v306*/
	v_dual_mov_b32 v3, v50 /*v306*/ :: v_dual_mov_b32 v4, v50 /*v306*/
	v_dual_mov_b32 v5, v50 /*v306*/ :: v_dual_mov_b32 v6, v50 /*v306*/
	v_dual_mov_b32 v7, v50 /*v306*/ :: v_dual_mov_b32 v8, v50 /*v306*/
	v_dual_mov_b32 v9, v50 /*v306*/ :: v_dual_mov_b32 v242, v50 /*v306*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v123 /*v379*/, v50 /*v306*/ :: v_dual_mov_b32 v124 /*v380*/, v50 /*v306*/
	v_dual_mov_b32 v125 /*v381*/, v50 /*v306*/ :: v_dual_mov_b32 v126 /*v382*/, v50 /*v306*/
	v_dual_mov_b32 v127 /*v383*/, v50 /*v306*/ :: v_dual_mov_b32 v128 /*v384*/, v50 /*v306*/
	v_dual_mov_b32 v129 /*v385*/, v50 /*v306*/ :: v_dual_mov_b32 v114 /*v370*/, v50 /*v306*/
	v_dual_mov_b32 v115 /*v371*/, v50 /*v306*/ :: v_dual_mov_b32 v116 /*v372*/, v50 /*v306*/
	v_dual_mov_b32 v117 /*v373*/, v50 /*v306*/ :: v_dual_mov_b32 v118 /*v374*/, v50 /*v306*/
	v_dual_mov_b32 v119 /*v375*/, v50 /*v306*/ :: v_dual_mov_b32 v120 /*v376*/, v50 /*v306*/
	v_dual_mov_b32 v121 /*v377*/, v50 /*v306*/ :: v_dual_mov_b32 v106 /*v362*/, v50 /*v306*/
	v_dual_mov_b32 v107 /*v363*/, v50 /*v306*/ :: v_dual_mov_b32 v108 /*v364*/, v50 /*v306*/
	v_dual_mov_b32 v109 /*v365*/, v50 /*v306*/ :: v_dual_mov_b32 v110 /*v366*/, v50 /*v306*/
	v_dual_mov_b32 v111 /*v367*/, v50 /*v306*/ :: v_dual_mov_b32 v112 /*v368*/, v50 /*v306*/
	v_dual_mov_b32 v113 /*v369*/, v50 /*v306*/ :: v_dual_mov_b32 v98 /*v354*/, v50 /*v306*/
	v_dual_mov_b32 v99 /*v355*/, v50 /*v306*/ :: v_dual_mov_b32 v100 /*v356*/, v50 /*v306*/
	v_dual_mov_b32 v101 /*v357*/, v50 /*v306*/ :: v_dual_mov_b32 v102 /*v358*/, v50 /*v306*/
	v_dual_mov_b32 v103 /*v359*/, v50 /*v306*/ :: v_dual_mov_b32 v104 /*v360*/, v50 /*v306*/
	v_dual_mov_b32 v105 /*v361*/, v50 /*v306*/ :: v_dual_mov_b32 v90 /*v346*/, v50 /*v306*/
	v_dual_mov_b32 v91 /*v347*/, v50 /*v306*/ :: v_dual_mov_b32 v92 /*v348*/, v50 /*v306*/
	v_dual_mov_b32 v93 /*v349*/, v50 /*v306*/ :: v_dual_mov_b32 v94 /*v350*/, v50 /*v306*/
	v_dual_mov_b32 v95 /*v351*/, v50 /*v306*/ :: v_dual_mov_b32 v96 /*v352*/, v50 /*v306*/
	v_dual_mov_b32 v97 /*v353*/, v50 /*v306*/ :: v_dual_mov_b32 v82 /*v338*/, v50 /*v306*/
	v_dual_mov_b32 v83 /*v339*/, v50 /*v306*/ :: v_dual_mov_b32 v84 /*v340*/, v50 /*v306*/
	v_dual_mov_b32 v85 /*v341*/, v50 /*v306*/ :: v_dual_mov_b32 v86 /*v342*/, v50 /*v306*/
	v_dual_mov_b32 v87 /*v343*/, v50 /*v306*/ :: v_dual_mov_b32 v88 /*v344*/, v50 /*v306*/
	v_dual_mov_b32 v89 /*v345*/, v50 /*v306*/ :: v_dual_mov_b32 v74 /*v330*/, v50 /*v306*/
	v_dual_mov_b32 v75 /*v331*/, v50 /*v306*/ :: v_dual_mov_b32 v76 /*v332*/, v50 /*v306*/
	v_dual_mov_b32 v77 /*v333*/, v50 /*v306*/ :: v_dual_mov_b32 v78 /*v334*/, v50 /*v306*/
	v_dual_mov_b32 v79 /*v335*/, v50 /*v306*/ :: v_dual_mov_b32 v80 /*v336*/, v50 /*v306*/
	v_dual_mov_b32 v81 /*v337*/, v50 /*v306*/ :: v_dual_mov_b32 v66 /*v322*/, v50 /*v306*/
	v_dual_mov_b32 v67 /*v323*/, v50 /*v306*/ :: v_dual_mov_b32 v68 /*v324*/, v50 /*v306*/
	v_dual_mov_b32 v69 /*v325*/, v50 /*v306*/ :: v_dual_mov_b32 v70 /*v326*/, v50 /*v306*/
	v_dual_mov_b32 v71 /*v327*/, v50 /*v306*/ :: v_dual_mov_b32 v72 /*v328*/, v50 /*v306*/
	v_dual_mov_b32 v73 /*v329*/, v50 /*v306*/ :: v_dual_mov_b32 v18 /*v274*/, v50 /*v306*/
	v_dual_mov_b32 v19 /*v275*/, v50 /*v306*/ :: v_dual_mov_b32 v20 /*v276*/, v50 /*v306*/
	v_dual_mov_b32 v21 /*v277*/, v50 /*v306*/ :: v_dual_mov_b32 v22 /*v278*/, v50 /*v306*/
	v_dual_mov_b32 v23 /*v279*/, v50 /*v306*/ :: v_dual_mov_b32 v24 /*v280*/, v50 /*v306*/
	v_dual_mov_b32 v25 /*v281*/, v50 /*v306*/ :: v_dual_mov_b32 v2 /*v258*/, v50 /*v306*/
	v_dual_mov_b32 v3 /*v259*/, v50 /*v306*/ :: v_dual_mov_b32 v4 /*v260*/, v50 /*v306*/
	v_dual_mov_b32 v5 /*v261*/, v50 /*v306*/ :: v_dual_mov_b32 v6 /*v262*/, v50 /*v306*/
	v_dual_mov_b32 v7 /*v263*/, v50 /*v306*/ :: v_dual_mov_b32 v8 /*v264*/, v50 /*v306*/
	v_mov_b32_e32 v9 /*v265*/, v50 /*v306*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v243, v50 /*v306*/ :: v_dual_mov_b32 v244, v50 /*v306*/
	v_dual_mov_b32 v245, v50 /*v306*/ :: v_dual_mov_b32 v246, v50 /*v306*/
	v_dual_mov_b32 v247, v50 /*v306*/ :: v_dual_mov_b32 v248, v50 /*v306*/
	v_dual_mov_b32 v249, v50 /*v306*/ :: v_dual_mov_b32 v234, v50 /*v306*/
	v_dual_mov_b32 v235, v50 /*v306*/ :: v_dual_mov_b32 v236, v50 /*v306*/
	v_dual_mov_b32 v237, v50 /*v306*/ :: v_dual_mov_b32 v238, v50 /*v306*/
	v_dual_mov_b32 v239, v50 /*v306*/ :: v_dual_mov_b32 v240, v50 /*v306*/
	v_dual_mov_b32 v241, v50 /*v306*/ :: v_dual_mov_b32 v218, v50 /*v306*/
	v_dual_mov_b32 v219, v50 /*v306*/ :: v_dual_mov_b32 v220, v50 /*v306*/
	v_dual_mov_b32 v221, v50 /*v306*/ :: v_dual_mov_b32 v222, v50 /*v306*/
	v_dual_mov_b32 v223, v50 /*v306*/ :: v_dual_mov_b32 v224, v50 /*v306*/
	v_dual_mov_b32 v225, v50 /*v306*/ :: v_dual_mov_b32 v210, v50 /*v306*/
	v_dual_mov_b32 v211, v50 /*v306*/ :: v_dual_mov_b32 v212, v50 /*v306*/
	v_dual_mov_b32 v213, v50 /*v306*/ :: v_dual_mov_b32 v214, v50 /*v306*/
	v_dual_mov_b32 v215, v50 /*v306*/ :: v_dual_mov_b32 v216, v50 /*v306*/
	v_dual_mov_b32 v217, v50 /*v306*/ :: v_dual_mov_b32 v202, v50 /*v306*/
	v_dual_mov_b32 v203, v50 /*v306*/ :: v_dual_mov_b32 v204, v50 /*v306*/
	v_dual_mov_b32 v205, v50 /*v306*/ :: v_dual_mov_b32 v206, v50 /*v306*/
	v_dual_mov_b32 v207, v50 /*v306*/ :: v_dual_mov_b32 v208, v50 /*v306*/
	v_dual_mov_b32 v209, v50 /*v306*/ :: v_dual_mov_b32 v194, v50 /*v306*/
	v_dual_mov_b32 v195, v50 /*v306*/ :: v_dual_mov_b32 v196, v50 /*v306*/
	v_dual_mov_b32 v197, v50 /*v306*/ :: v_dual_mov_b32 v198, v50 /*v306*/
	v_dual_mov_b32 v199, v50 /*v306*/ :: v_dual_mov_b32 v200, v50 /*v306*/
	v_mov_b32_e32 v201, v50 /*v306*/
	s_ashr_i32 s2, s25, 31
	s_mov_b32 s24, 32
	s_mov_b64 s[58:59], 0
	s_mov_b32 s44, 1
	s_mov_b32 s42, s38
	s_mov_b32 s43, s39
	s_mov_b32 s21, 0xffff0000
	s_mov_b32 s20, 0x7510000
	s_movk_i32 s49, 0x2200
	s_mov_b32 s16, 0x3fb8aa3b
	s_and_b32 s26, s2, 0xffff
	s_mov_b32 s2, s45
	s_mov_b32 s3, s45
	s_set_vgpr_msb 0x100
.LBB0_4:
	s_add_co_i32 s4, s2, 1
	s_mov_b32 s27, s45
	s_cmp_ge_i32 s4, s17
	s_mov_b32 s48, s44
	s_cselect_b32 s5, -1, 0
	s_and_b32 s6, s5, exec_lo
	s_cselect_b32 s65, 0, s4
	s_cmp_lg_u32 s5, 0
	s_add_co_ci_u32 s66, s3, 0
	s_add_co_i32 s3, s3, s18
	s_add_co_i32 s2, s2, s14
	s_lshl_b32 s6, s3, 5
	s_add_co_i32 s5, s2, s15
	s_add_co_i32 s4, s6, s35
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v151 /*v407*/, s6, v13 /*v525*/
	s_mul_i32 s7, s5, s13
	s_ashr_i32 s5, s4, 31
	s_set_vgpr_msb 0x4888
	v_or_b32_e32 v1 /*v513*/, s6, v16 /*v528*/
	s_ashr_i32 s3, s2, 31
	s_mul_u64 s[4:5], s[4:5], s[28:29]
	s_set_vgpr_msb 0x8844
	v_add_lshl_u32 v152 /*v408*/, s7, v151 /*v407*/, 2
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	s_sub_co_i32 s4, s13, s6
	s_set_vgpr_msb 0x4448
	v_add_lshl_u32 v153 /*v409*/, s7, v1 /*v513*/, 2
	s_lshl_b64 s[2:3], s[2:3], 8
	s_max_i32 s4, s4, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[46:47], s[54:55], s[2:3]
	s_lshl_b32 s5, s4, 16
	s_lshr_b32 s4, s4, 16
	s_add_nc_u64 s[50:51], s[52:53], s[2:3]
	s_set_vgpr_msb 0x4881
	s_clause 0x1
	buffer_load_b32 v0 /*v512*/, v152 /*v408*/, s[36:39], null offen
	buffer_load_b32 v4 /*v516*/, v153 /*v409*/, s[36:39], null offen
	s_clause 0x1
	buffer_load_b32 v6 /*v518*/, v152 /*v408*/, s[40:43], null offen
	buffer_load_b32 v8 /*v520*/, v153 /*v409*/, s[40:43], null offen
	s_bitset1_b32 s47, 31
	s_or_b32 s22, s5, 0x7fff
	s_or_b32 s23, s4, 0x800000
	s_bitset1_b32 s51, 31
	tensor_load_to_lds s[44:47], s[20:27]
	tensor_load_to_lds s[48:51], s[20:27]
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x8142
	ds_load_b128 v[152:155] /*v[408:411]*/, v14 /*v526*/ offset:8704
	ds_load_b128 v[156:159] /*v[412:415]*/, v14 /*v526*/ offset:8736
	ds_load_b128 v[160:163] /*v[416:419]*/, v14 /*v526*/ offset:8768
	ds_load_b128 v[164:167] /*v[420:423]*/, v14 /*v526*/ offset:8800
	ds_load_b128 v[168:171] /*v[424:427]*/, v14 /*v526*/ offset:8832
	ds_load_b128 v[172:175] /*v[428:431]*/, v14 /*v526*/ offset:8864
	ds_load_b128 v[176:179] /*v[432:435]*/, v14 /*v526*/ offset:8896
	ds_load_b128 v[180:183] /*v[436:439]*/, v14 /*v526*/ offset:8928
	ds_load_b128 v[184:187] /*v[440:443]*/, v14 /*v526*/
	ds_load_b128 v[188:191] /*v[444:447]*/, v14 /*v526*/ offset:32
	ds_load_b128 v[192:195] /*v[448:451]*/, v14 /*v526*/ offset:64
	ds_load_b128 v[196:199] /*v[452:455]*/, v14 /*v526*/ offset:96
	ds_load_b128 v[200:203] /*v[456:459]*/, v14 /*v526*/ offset:128
	ds_load_b128 v[204:207] /*v[460:463]*/, v14 /*v526*/ offset:160
	ds_load_b128 v[208:211] /*v[464:467]*/, v14 /*v526*/ offset:192
	ds_load_b128 v[212:215] /*v[468:471]*/, v14 /*v526*/ offset:224
	ds_load_b128 v[216:219] /*v[472:475]*/, v14 /*v526*/ offset:13056
	ds_load_b128 v[220:223] /*v[476:479]*/, v14 /*v526*/ offset:13088
	ds_load_b128 v[224:227] /*v[480:483]*/, v14 /*v526*/ offset:13120
	ds_load_b128 v[228:231] /*v[484:487]*/, v14 /*v526*/ offset:13152
	ds_load_b128 v[232:235] /*v[488:491]*/, v14 /*v526*/ offset:13184
	ds_load_b128 v[236:239] /*v[492:495]*/, v14 /*v526*/ offset:13216
	ds_load_b128 v[240:243] /*v[496:499]*/, v14 /*v526*/ offset:13248
	ds_load_b128 v[244:247] /*v[500:503]*/, v14 /*v526*/ offset:13280
	ds_load_b128 v[248:251] /*v[504:507]*/, v14 /*v526*/ offset:4352
	ds_load_b128 v[252:255] /*v[508:511]*/, v14 /*v526*/ offset:4384
	s_set_vgpr_msb 0x4282
	ds_load_b128 v[20:23] /*v[532:535]*/, v14 /*v526*/ offset:4416
	ds_load_b128 v[24:27] /*v[536:539]*/, v14 /*v526*/ offset:4448
	ds_load_b128 v[28:31] /*v[540:543]*/, v14 /*v526*/ offset:4480
	ds_load_b128 v[32:35] /*v[544:547]*/, v14 /*v526*/ offset:4512
	ds_load_b128 v[36:39] /*v[548:551]*/, v14 /*v526*/ offset:4544
	ds_load_b128 v[40:43] /*v[552:555]*/, v14 /*v526*/ offset:4576
	s_set_vgpr_msb 0x8284
	s_wait_loadcnt_dscnt 0x221e
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[34:41], v[152:159] /*v[408:415]*/, 0
	s_set_vgpr_msb 0x8445
	v_add_nc_u32_e32 v151 /*v407*/, s19, v151 /*v407*/
	v_cmp_gt_i32_e64 s2, v141 /*v397*/, v151 /*v407*/
	v_cmp_ge_i32_e32 vcc_lo, v141 /*v397*/, v151 /*v407*/
	s_set_vgpr_msb 0x45a4
	s_wait_loadcnt_dscnt 0x201c
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[42:49], v[160:167] /*v[416:423]*/, v[44:51] /*v[556:563]*/
	v_cmp_gt_i32_e64 s3, v1, v151 /*v407*/
	s_set_vgpr_msb 0xa405
	v_cmp_gt_i32_e64 s4, v130 /*v386*/, v151 /*v407*/
	s_and_b32 s2, s64, s2
	s_and_b32 s22, s64, vcc_lo
	v_cmp_gt_i32_e64 s5, v131 /*v387*/, v151 /*v407*/
	s_and_b32 s3, s64, s3
	v_cmp_gt_i32_e64 s6, v132 /*v388*/, v151 /*v407*/
	s_set_vgpr_msb 0x5a4
	s_wait_loadcnt_dscnt 0x1e1a
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[50:57], v[168:175] /*v[424:431]*/, v[44:51] /*v[556:563]*/
	s_set_vgpr_msb 0xa405
	v_cmp_gt_i32_e64 s8, v134 /*v390*/, v151 /*v407*/
	v_cmp_gt_i32_e64 s7, v133 /*v389*/, v151 /*v407*/
	v_cmp_ge_i32_e64 s9, v144 /*v400*/, v151 /*v407*/
	v_cmp_gt_i32_e64 s10, v144 /*v400*/, v151 /*v407*/
	v_cmp_gt_i32_e64 s11, v135 /*v391*/, v151 /*v407*/
	v_cmp_gt_i32_e32 vcc_lo, v136 /*v392*/, v151 /*v407*/
	s_set_vgpr_msb 0x5a4
	s_wait_loadcnt_dscnt 0x1c18
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[58:65], v[176:183] /*v[432:439]*/, v[44:51] /*v[556:563]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa489
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[142:143] /*v[398:399]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x8984
	s_wait_loadcnt_dscnt 0x1216
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[106:113], v[184:191] /*v[440:447]*/, 0
	s_set_vgpr_msb 0x8489
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[142:143] /*v[398:399]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[142:143] /*v[398:399]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[142:143] /*v[398:399]*/, v[50:51] /*v[562:563]*/
	s_set_vgpr_msb 0x8982
	v_cndmask_b32_e64 v44 /*v556*/, v44 /*v556*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, s4
	v_cndmask_b32_e64 v45 /*v557*/, v45 /*v557*/, 0xff61b1e6, s22
	v_cndmask_b32_e64 v47 /*v559*/, v47 /*v559*/, 0xff61b1e6, s3
	s_set_vgpr_msb 0x82a4
	s_wait_loadcnt_dscnt 0x1014
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[122:129], v[192:199] /*v[448:455]*/, v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa48a
	v_cndmask_b32_e64 v46 /*v558*/, v46 /*v558*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, s6
	s_wait_loadcnt 0x3
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s64, s5
	v_cndmask_b32_e64 v48 /*v560*/, v48 /*v560*/, 0xff61b1e6, s2
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v49 /*v561*/, v49 /*v561*/, 0xff61b1e6, s3
	s_set_vgpr_msb 0x8aa4
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[130:137], v[200:207] /*v[456:463]*/, v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa482
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, s[16:17] op_sel_hi:[1,0]
	s_and_b32 s2, s64, s8
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, s[16:17] op_sel_hi:[1,0]
	v_exp_f32_e32 v60 /*v572*/, v44 /*v556*/
	v_exp_f32_e32 v61 /*v573*/, v45 /*v557*/
	s_set_vgpr_msb 0x82a4
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[138:145], v[208:215] /*v[464:471]*/, v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa48a
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[48:49] /*v[560:561]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v46 /*v558*/, v46 /*v558*/
	v_exp_f32_e32 v47 /*v559*/, v47 /*v559*/
	v_cndmask_b32_e64 v48 /*v560*/, v50 /*v562*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, s7
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, s[16:17] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v49 /*v561*/, v51 /*v563*/, 0xff61b1e6, s2
	s_wait_loadcnt 0x1
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[54:55] /*v[566:567]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[56:57] /*v[568:569]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v44 /*v556*/, v44 /*v556*/
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, v[46:47] /*v[558:559]*/
	v_exp_f32_e32 v45 /*v557*/, v45 /*v557*/
	v_pk_add_f32 v[62:63] /*v[574:575]*/, v[58:59] /*v[570:571]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, s[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8a89
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[142:143] /*v[398:399]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[142:143] /*v[398:399]*/, v[52:53] /*v[564:565]*/
	s_and_b32 s2, s64, s9
	s_set_vgpr_msb 0x898a
	v_exp_f32_e32 v64 /*v576*/, v48 /*v560*/
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[54:55] /*v[566:567]*/, v[44:45] /*v[556:557]*/
	v_cvt_pk_bf16_f32 v54 /*v566*/, v44 /*v556*/, v45 /*v557*/
	v_exp_f32_e32 v65 /*v577*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v56 /*v568*/, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v57 /*v569*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v53 /*v565*/, v46 /*v558*/, v47 /*v559*/
	s_set_vgpr_msb 0x8a84
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[74:81], v[152:159] /*v[408:415]*/, 0
	s_set_vgpr_msb 0x8489
	v_pk_mul_f32 v[58:59] /*v[570:571]*/, v[142:143] /*v[398:399]*/, v[58:59] /*v[570:571]*/
	s_set_vgpr_msb 0x898a
	v_cvt_pk_bf16_f32 v52 /*v564*/, v60 /*v572*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v55 /*v567*/, v64 /*v576*/, v65 /*v577*/
	v_nop
	s_set_vgpr_msb 0x8a4a
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[62:63] /*v[574:575]*/, v[64:65] /*v[576:577]*/
	s_set_vgpr_msb 0x4a8a
	v_cvt_pk_bf16_f32 v58 /*v570*/, v58 /*v570*/, v59 /*v571*/
	s_set_vgpr_msb 0x8a85
	v_pk_mul_f32 v[62:63] /*v[574:575]*/, v[142:143] /*v[398:399]*/, v[152:153] /*v[408:409]*/
	s_set_vgpr_msb 0x85a4
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[82:89], v[160:167] /*v[416:423]*/, v[44:51] /*v[556:563]*/
	s_set_vgpr_msb 0xa48a
	v_cvt_pk_bf16_f32 v59 /*v571*/, v62 /*v574*/, v63 /*v575*/
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[90:97], v[168:175] /*v[424:431]*/, v[44:51] /*v[556:563]*/
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[98:105], v[176:183] /*v[432:439]*/, v[44:51] /*v[556:563]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa449
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[142:143] /*v[398:399]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[142:143] /*v[398:399]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4944
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[146:153], v[184:191] /*v[440:447]*/, 0
	s_set_vgpr_msb 0x4449
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[142:143] /*v[398:399]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[142:143] /*v[398:399]*/, v[50:51] /*v[562:563]*/
	v_cndmask_b32_e64 v161 /*v417*/, v161 /*v417*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, s10
	v_cndmask_b32_e64 v160 /*v416*/, v160 /*v416*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, s11
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[154:161], v[192:199] /*v[448:455]*/, v[152:159] /*v[408:415]*/
	s_set_vgpr_msb 0x5445
	v_cndmask_b32_e64 v163 /*v419*/, v163 /*v419*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v137 /*v393*/, v151 /*v407*/
	v_cndmask_b32_e64 v162 /*v418*/, v162 /*v418*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v138 /*v394*/, v151 /*v407*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s64, vcc_lo
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[162:163] /*v[418:419]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s64, s2
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[160:161] /*v[416:417]*/, s[16:17] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v171 /*v427*/, v165 /*v421*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v170 /*v426*/, v164 /*v420*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4944
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[34:41], v[216:223] /*v[472:479]*/, 0
	s_set_vgpr_msb 0x4445
	v_cmp_gt_i32_e32 vcc_lo, v139 /*v395*/, v151 /*v407*/
	v_cmp_gt_i32_e64 s2, v140 /*v396*/, v151 /*v407*/
	v_exp_f32_e32 v184 /*v440*/, v176 /*v432*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[170:171] /*v[426:427]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v185 /*v441*/, v177 /*v433*/
	s_and_b32 s3, s64, vcc_lo
	s_and_b32 s2, s64, s2
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[162:169], v[200:207] /*v[456:463]*/, v[152:159] /*v[408:415]*/
	s_set_vgpr_msb 0x5449
	v_cndmask_b32_e64 v179 /*v435*/, v179 /*v435*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v178 /*v434*/, v178 /*v434*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[182:183] /*v[438:439]*/, s[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[168:169] /*v[424:425]*/, s[16:17] op_sel_hi:[1,0]
	v_add_nc_u32_e32 v151 /*v407*/, s19, v1 /*v513*/
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4954
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[42:49], v[224:231] /*v[480:487]*/, v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5445
	v_exp_f32_e32 v186 /*v442*/, v176 /*v432*/
	v_exp_f32_e32 v187 /*v443*/, v177 /*v433*/
	v_exp_f32_e32 v180 /*v436*/, v180 /*v436*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[178:179] /*v[434:435]*/, s[16:17] op_sel_hi:[1,0]
	v_exp_f32_e32 v181 /*v437*/, v181 /*v437*/
	v_cmp_ge_i32_e32 vcc_lo, v141 /*v397*/, v151 /*v407*/
	v_cmp_gt_i32_e64 s2, v141 /*v397*/, v151 /*v407*/
	s_set_vgpr_msb 0x4554
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[170:177], v[208:215] /*v[464:471]*/, v[152:159] /*v[408:415]*/
	s_set_vgpr_msb 0x5441
	v_exp_f32_e32 v182 /*v438*/, v176 /*v432*/
	v_exp_f32_e32 v183 /*v439*/, v177 /*v433*/
	s_and_b32 s3, s64, vcc_lo
	s_and_b32 s2, s64, s2
	v_cmp_lt_i32_e32 vcc_lo, v151 /*v407*/, v1
	v_nop
	s_set_vgpr_msb 0x4149
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4954
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[50:57], v[232:239] /*v[488:495]*/, v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5449
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[184:185] /*v[440:441]*/
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[180:181] /*v[436:437]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[182:183] /*v[438:439]*/
	s_set_vgpr_msb 0x4554
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[58:65], v[240:247] /*v[496:503]*/, v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5445
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[142:143] /*v[398:399]*/, v[152:153] /*v[408:409]*/
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[186:187] /*v[442:443]*/
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[142:143] /*v[398:399]*/, v[154:155] /*v[410:411]*/
	v_cvt_pk_bf16_f32 v183 /*v439*/, v182 /*v438*/, v183 /*v439*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v186 /*v442*/, v187 /*v443*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v152 /*v408*/, v153 /*v409*/
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[142:143] /*v[398:399]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[142:143] /*v[398:399]*/, v[160:161] /*v[416:417]*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v154 /*v410*/, v155 /*v411*/
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[142:143] /*v[398:399]*/, v[156:157] /*v[412:413]*/
	s_set_vgpr_msb 0x4544
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[106:113], v[248:255] /*v[504:511]*/, 0
	s_set_vgpr_msb 0x4445
	v_cvt_pk_bf16_f32 v179 /*v435*/, v152 /*v408*/, v153 /*v409*/
	v_cndmask_b32_e64 v161 /*v417*/, v161 /*v417*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v160 /*v416*/, v160 /*v416*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[142:143] /*v[398:399]*/, v[162:163] /*v[418:419]*/
	v_cmp_gt_i32_e64 s2, v130 /*v386*/, v151 /*v407*/
	s_and_b32 s3, s64, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v131 /*v387*/, v151 /*v407*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[160:161] /*v[416:417]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v153 /*v409*/, v153 /*v409*/, 0xff61b1e6, s3
	s_set_vgpr_msb 0x4945
	v_cmp_gt_i32_e64 s3, v132 /*v388*/, v151 /*v407*/
	s_and_b32 s2, s64, s2
	v_cvt_pk_bf16_f32 v178 /*v434*/, v156 /*v412*/, v157 /*v413*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[154:155] /*v[410:411]*/, s[16:17] op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[142:143] /*v[398:399]*/, v[164:165] /*v[420:421]*/
	v_cndmask_b32_e64 v152 /*v408*/, v152 /*v408*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, vcc_lo
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[142:143] /*v[398:399]*/, v[166:167] /*v[422:423]*/
	v_cmp_gt_i32_e32 vcc_lo, v133 /*v389*/, v151 /*v407*/
	v_cndmask_b32_e64 v155 /*v411*/, v155 /*v411*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, s3
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[152:153] /*v[408:409]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v154 /*v410*/, v154 /*v410*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s2, v134 /*v390*/, v151 /*v407*/
	s_set_vgpr_msb 0x558
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[122:129], v[20:27] /*v[532:539]*/, v[168:175] /*v[424:431]*/
	s_and_b32 s3, s64, vcc_lo
	s_set_vgpr_msb 0x5805
	v_cmp_ge_i32_e32 vcc_lo, v144 /*v400*/, v151 /*v407*/
	s_set_vgpr_msb 0x549
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[154:155] /*v[410:411]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s64, s2
	v_cndmask_b32_e64 v189 /*v445*/, v165 /*v421*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v188 /*v444*/, v164 /*v420*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s2, v144 /*v400*/, v151 /*v407*/
	s_set_vgpr_msb 0x544
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[74:81], v[216:223] /*v[472:479]*/, 0
	s_and_b32 s3, s64, vcc_lo
	s_set_vgpr_msb 0x4445
	v_cmp_gt_i32_e32 vcc_lo, v135 /*v391*/, v151 /*v407*/
	v_exp_f32_e32 v186 /*v442*/, v160 /*v416*/
	s_and_b32 s2, s64, s2
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[162:163] /*v[418:419]*/, s[16:17] op_sel_hi:[1,0]
	v_exp_f32_e32 v187 /*v443*/, v161 /*v417*/
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[166:167] /*v[422:423]*/, s[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4554
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[82:89], v[224:231] /*v[480:487]*/, v[152:159] /*v[408:415]*/
	s_set_vgpr_msb 0x5449
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v190 /*v446*/, v190 /*v446*/
	v_exp_f32_e32 v191 /*v447*/, v191 /*v447*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v181 /*v437*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v180 /*v436*/, v184 /*v440*/, v185 /*v441*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, s[16:17] op_sel_hi:[1,0]
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	s_set_vgpr_msb 0x4554
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[90:97], v[232:239] /*v[488:495]*/, v[152:159] /*v[408:415]*/
	s_set_vgpr_msb 0x5441
	v_exp_f32_e32 v193 /*v449*/, v193 /*v449*/
	v_exp_f32_e32 v188 /*v444*/, v188 /*v444*/
	v_exp_f32_e32 v189 /*v445*/, v189 /*v445*/
	s_set_vgpr_msb 0x4154
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[98:105], v[240:247] /*v[496:503]*/, v[152:159] /*v[408:415]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5445
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[142:143] /*v[398:399]*/, v[152:153] /*v[408:409]*/
	s_set_vgpr_msb 0x4558
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[130:137], v[28:35] /*v[540:547]*/, v[168:175] /*v[424:431]*/
	s_set_vgpr_msb 0x5845
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[142:143] /*v[398:399]*/, v[154:155] /*v[410:411]*/
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[142:143] /*v[398:399]*/, v[156:157] /*v[412:413]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[142:143] /*v[398:399]*/, v[158:159] /*v[414:415]*/
	v_cndmask_b32_e64 v152 /*v408*/, v152 /*v408*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v136 /*v392*/, v151 /*v407*/
	v_cndmask_b32_e64 v153 /*v409*/, v153 /*v409*/, 0xff61b1e6, s3
	s_and_b32 s3, s64, vcc_lo
	s_set_vgpr_msb 0x4544
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[146:153], v[248:255] /*v[504:511]*/, 0
	s_set_vgpr_msb 0x4445
	v_cmp_gt_i32_e32 vcc_lo, v137 /*v393*/, v151 /*v407*/
	s_and_b32 s2, s64, s2
	v_cndmask_b32_e64 v155 /*v411*/, v155 /*v411*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v154 /*v410*/, v154 /*v410*/, 0xff61b1e6, s2
	v_cmp_gt_i32_e64 s2, v138 /*v394*/, v151 /*v407*/
	s_and_b32 s3, s64, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v139 /*v395*/, v151 /*v407*/
	s_set_vgpr_msb 0x4558
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[138:145], v[36:43] /*v[548:555]*/, v[168:175] /*v[424:431]*/
	s_set_vgpr_msb 0x5845
	v_cndmask_b32_e64 v157 /*v413*/, v157 /*v413*/, 0xff61b1e6, s3
	v_cmp_gt_i32_e64 s3, v140 /*v396*/, v151 /*v407*/
	s_and_b32 s2, s64, s2
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v156 /*v412*/, v156 /*v412*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, vcc_lo
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4958
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[154:161], v[20:27] /*v[532:539]*/, v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5849
	v_cndmask_b32_e64 v159 /*v415*/, v159 /*v415*/, 0xff61b1e6, s2
	s_and_b32 s2, s64, s3
	s_wait_loadcnt 0x0
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[168:169] /*v[424:425]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[170:171] /*v[426:427]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v158 /*v414*/, v158 /*v414*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, s[16:17] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4958
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[162:169], v[28:35] /*v[540:547]*/, v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5845
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[168:169] /*v[424:425]*/, v[186:187] /*v[442:443]*/
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[170:171] /*v[426:427]*/, v[190:191] /*v[446:447]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[188:189] /*v[444:445]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[142:143] /*v[398:399]*/, v[168:169] /*v[424:425]*/
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[142:143] /*v[398:399]*/, v[170:171] /*v[426:427]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[142:143] /*v[398:399]*/, v[174:175] /*v[430:431]*/
	s_set_vgpr_msb 0x4558
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[170:177], v[36:43] /*v[548:555]*/, v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5845
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, s[16:17] op_sel_hi:[1,0]
	v_exp_f32_e32 v184 /*v440*/, v152 /*v408*/
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, s[16:17] op_sel_hi:[1,0]
	v_exp_f32_e32 v185 /*v441*/, v153 /*v409*/
	v_nop
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[158:159] /*v[414:415]*/, s[16:17] op_sel_hi:[1,0]
	v_cvt_pk_bf16_f32 v168 /*v424*/, v168 /*v424*/, v169 /*v425*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v175 /*v431*/, v188 /*v444*/, v189 /*v445*/
	v_exp_f32_e32 v188 /*v444*/, v154 /*v410*/
	v_exp_f32_e32 v189 /*v445*/, v155 /*v411*/
	v_exp_f32_e32 v156 /*v412*/, v156 /*v412*/
	v_exp_f32_e32 v157 /*v413*/, v157 /*v413*/
	v_exp_f32_e32 v158 /*v414*/, v152 /*v408*/
	v_exp_f32_e32 v159 /*v415*/, v153 /*v409*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[160:161] /*v[416:417]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[162:163] /*v[418:419]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[164:165] /*v[420:421]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[166:167] /*v[422:423]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[184:185] /*v[440:441]*/
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[188:189] /*v[444:445]*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[156:157] /*v[412:413]*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[142:143] /*v[398:399]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[142:143] /*v[398:399]*/, v[152:153] /*v[408:409]*/
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[142:143] /*v[398:399]*/, v[154:155] /*v[410:411]*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[142:143] /*v[398:399]*/, v[160:161] /*v[416:417]*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[142:143] /*v[398:399]*/, v[162:163] /*v[418:419]*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v172 /*v428*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v192 /*v448*/, v193 /*v449*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v190 /*v446*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v186 /*v442*/, v187 /*v443*/
	v_cvt_pk_bf16_f32 v152 /*v408*/, v152 /*v408*/, v153 /*v409*/
	v_cvt_pk_bf16_f32 v153 /*v409*/, v154 /*v410*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v154 /*v410*/, v160 /*v416*/, v161 /*v417*/
	v_cvt_pk_bf16_f32 v155 /*v411*/, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v156 /*v412*/, v157 /*v413*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v188 /*v444*/, v189 /*v445*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v184 /*v440*/, v185 /*v441*/
	s_set_vgpr_msb 0x4509
	ds_store_b128 v145 /*v401*/, v[52:55] /*v[564:567]*/
	s_set_vgpr_msb 0x905
	ds_store_b128 v145 /*v401*/, v[180:183] /*v[436:439]*/ offset:32
	s_set_vgpr_msb 0x509
	ds_store_b128 v145 /*v401*/, v[56:59] /*v[568:571]*/ offset:2560
	s_set_vgpr_msb 0x945
	ds_store_b128 v145 /*v401*/, v[176:179] /*v[432:435]*/ offset:2592
	ds_store_b128 v146 /*v402*/, v[172:175] /*v[428:431]*/
	ds_store_b128 v146 /*v402*/, v[156:159] /*v[412:415]*/ offset:32
	ds_store_b128 v146 /*v402*/, v[168:171] /*v[424:427]*/ offset:2560
	ds_store_b128 v146 /*v402*/, v[152:155] /*v[408:411]*/ offset:2592
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v147 /*v403*/
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v147 /*v403*/ offset:1280
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v148 /*v404*/
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v148 /*v404*/ offset:1280
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v149 /*v405*/
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v149 /*v405*/ offset:1280
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v150 /*v406*/
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v150 /*v406*/ offset:1280
	s_set_vgpr_msb 0x4542
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v15 /*v527*/
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v15 /*v527*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v15 /*v527*/ offset:4352
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v15 /*v527*/ offset:4384
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v15 /*v527*/ offset:8704
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v15 /*v527*/ offset:8736
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v15 /*v527*/ offset:13056
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v15 /*v527*/ offset:13088
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v15 /*v527*/ offset:64
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v15 /*v527*/ offset:96
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v15 /*v527*/ offset:4416
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v15 /*v527*/ offset:4448
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v15 /*v527*/ offset:8768
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v15 /*v527*/ offset:8800
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v15 /*v527*/ offset:13120
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v15 /*v527*/ offset:13152
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v15 /*v527*/ offset:128
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[20:23] /*v[532:535]*/, v15 /*v527*/ offset:160
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v15 /*v527*/ offset:4480
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v15 /*v527*/ offset:4512
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v15 /*v527*/ offset:8832
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v15 /*v527*/ offset:8864
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v15 /*v527*/ offset:13184
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v15 /*v527*/ offset:13216
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v15 /*v527*/ offset:192
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v15 /*v527*/ offset:224
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v15 /*v527*/ offset:4544
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v15 /*v527*/ offset:4576
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v15 /*v527*/ offset:8896
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v15 /*v527*/ offset:8928
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v15 /*v527*/ offset:13248
	ds_load_tr16_b128 v[72:75] /*v[584:587]*/, v15 /*v527*/ offset:13280
	s_set_vgpr_msb 0x8255
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[152:159] /*v[408:415]*/, v[184:191] /*v[440:447]*/, v[50:57] /*v[306:313]*/
	s_add_nc_u64 s[58:59], s[58:59], 1
	s_mov_b32 s2, s65
	s_cmp_lg_u64 s[58:59], s[56:57]
	s_mov_b32 s3, s66
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[152:159] /*v[408:415]*/, v[192:199] /*v[448:455]*/, v[58:65] /*v[314:321]*/ matrix_a_reuse
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[152:159] /*v[408:415]*/, v[216:223] /*v[472:479]*/, v[42:49] /*v[298:305]*/ matrix_a_reuse
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[152:159] /*v[408:415]*/, v[224:231] /*v[480:487]*/, v[34:41] /*v[290:297]*/ matrix_a_reuse
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[152:159] /*v[408:415]*/, v[248:255] /*v[504:511]*/, v[26:33] /*v[282:289]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5559
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[152:159] /*v[408:415]*/, v[20:27] /*v[532:539]*/, v[10:17] /*v[266:273]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5909
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[250:257], v[152:159] /*v[408:415]*/, v[44:51] /*v[556:563]*/, v[250:257] matrix_a_reuse
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[226:233], v[152:159] /*v[408:415]*/, v[52:59] /*v[564:571]*/, v[226:233] matrix_a_reuse
	s_set_vgpr_msb 0x955
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[160:167] /*v[416:423]*/, v[200:207] /*v[456:463]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[160:167] /*v[416:423]*/, v[208:215] /*v[464:471]*/, v[114:121] /*v[370:377]*/ matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[160:167] /*v[416:423]*/, v[232:239] /*v[488:495]*/, v[106:113] /*v[362:369]*/ matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[160:167] /*v[416:423]*/, v[240:247] /*v[496:503]*/, v[98:105] /*v[354:361]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5559
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[160:167] /*v[416:423]*/, v[28:35] /*v[540:547]*/, v[90:97] /*v[346:353]*/ matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[160:167] /*v[416:423]*/, v[36:43] /*v[548:555]*/, v[82:89] /*v[338:345]*/ matrix_a_reuse
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[160:167] /*v[416:423]*/, v[60:67] /*v[572:579]*/, v[74:81] /*v[330:337]*/ matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[160:167] /*v[416:423]*/, v[68:75] /*v[580:587]*/, v[66:73] /*v[322:329]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5905
	v_wmma_f32_16x16x32_bf16 v[186:193], v[168:175] /*v[424:431]*/, v[184:191] /*v[440:447]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[178:185], v[168:175] /*v[424:431]*/, v[192:199] /*v[448:455]*/, v[178:185] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[168:175] /*v[424:431]*/, v[216:223] /*v[472:479]*/, v[114:121] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[168:175] /*v[424:431]*/, v[224:231] /*v[480:487]*/, v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[168:175] /*v[424:431]*/, v[248:255] /*v[504:511]*/, v[26:33] matrix_a_reuse
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[18:25], v[168:175] /*v[424:431]*/, v[20:27] /*v[532:539]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[168:175] /*v[424:431]*/, v[44:51] /*v[556:563]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[168:175] /*v[424:431]*/, v[52:59] /*v[564:571]*/, v[2:9] matrix_a_reuse
	s_set_vgpr_msb 0x955
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[176:183] /*v[432:439]*/, v[200:207] /*v[456:463]*/, v[18:25] /*v[274:281]*/
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[176:183] /*v[432:439]*/, v[208:215] /*v[464:471]*/, v[2:9] /*v[258:265]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[242:249], v[176:183] /*v[432:439]*/, v[232:239] /*v[488:495]*/, v[242:249] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[234:241], v[176:183] /*v[432:439]*/, v[240:247] /*v[496:503]*/, v[234:241] matrix_a_reuse
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[218:225], v[176:183] /*v[432:439]*/, v[28:35] /*v[540:547]*/, v[218:225] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[176:183] /*v[432:439]*/, v[36:43] /*v[548:555]*/, v[210:217] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[176:183] /*v[432:439]*/, v[60:67] /*v[572:579]*/, v[202:209] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[176:183] /*v[432:439]*/, v[68:75] /*v[580:587]*/, v[194:201] matrix_a_reuse
	s_set_vgpr_msb 0x900
	s_cbranch_scc1 .LBB0_4
	s_branch .LBB0_6
.LBB0_5:
	v_mov_b32_e32 v194, 0
	v_dual_mov_b32 v195, v194 :: v_dual_mov_b32 v196, v194
	v_dual_mov_b32 v197, v194 :: v_dual_mov_b32 v198, v194
	v_dual_mov_b32 v199, v194 :: v_dual_mov_b32 v200, v194
	v_mov_b32_e32 v201, v194
	v_mov_b64_e32 v[204:205], v[196:197]
	v_mov_b64_e32 v[202:203], v[194:195]
	v_mov_b64_e32 v[206:207], v[198:199]
	v_mov_b64_e32 v[214:215], v[198:199]
	v_mov_b64_e32 v[208:209], v[200:201]
	v_mov_b64_e32 v[216:217], v[200:201]
	v_mov_b64_e32 v[212:213], v[196:197]
	v_mov_b64_e32 v[210:211], v[194:195]
	v_mov_b64_e32 v[224:225], v[200:201]
	v_mov_b64_e32 v[222:223], v[198:199]
	v_mov_b64_e32 v[220:221], v[196:197]
	v_mov_b64_e32 v[218:219], v[194:195]
	v_mov_b64_e32 v[240:241], v[200:201]
	v_mov_b64_e32 v[238:239], v[198:199]
	v_mov_b64_e32 v[236:237], v[196:197]
	v_mov_b64_e32 v[234:235], v[194:195]
	v_mov_b64_e32 v[248:249], v[200:201]
	v_mov_b64_e32 v[246:247], v[198:199]
	v_mov_b64_e32 v[244:245], v[196:197]
	v_mov_b64_e32 v[242:243], v[194:195]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[8:9] /*v[264:265]*/, v[200:201]
	v_mov_b64_e32 v[6:7] /*v[262:263]*/, v[198:199]
	v_mov_b64_e32 v[4:5] /*v[260:261]*/, v[196:197]
	v_mov_b64_e32 v[2:3] /*v[258:259]*/, v[194:195]
	v_mov_b64_e32 v[24:25] /*v[280:281]*/, v[200:201]
	v_mov_b64_e32 v[22:23] /*v[278:279]*/, v[198:199]
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[196:197]
	v_mov_b64_e32 v[18:19] /*v[274:275]*/, v[194:195]
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[200:201]
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[198:199]
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[196:197]
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[194:195]
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[200:201]
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[198:199]
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[196:197]
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[194:195]
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[200:201]
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[198:199]
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[196:197]
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[194:195]
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, v[200:201]
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, v[198:199]
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, v[196:197]
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, v[194:195]
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[200:201]
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[198:199]
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[196:197]
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[194:195]
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[200:201]
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[198:199]
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[196:197]
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[194:195]
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[200:201]
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[198:199]
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[196:197]
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[194:195]
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[200:201]
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[198:199]
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[196:197]
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[194:195]
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[2:3], v[194:195]
	v_mov_b64_e32 v[4:5], v[196:197]
	v_mov_b64_e32 v[6:7], v[198:199]
	v_mov_b64_e32 v[8:9], v[200:201]
	v_mov_b64_e32 v[10:11], v[194:195]
	v_mov_b64_e32 v[12:13], v[196:197]
	v_mov_b64_e32 v[14:15], v[198:199]
	v_mov_b64_e32 v[16:17], v[200:201]
	v_mov_b64_e32 v[18:19], v[194:195]
	v_mov_b64_e32 v[20:21], v[196:197]
	v_mov_b64_e32 v[22:23], v[198:199]
	v_mov_b64_e32 v[24:25], v[200:201]
	v_mov_b64_e32 v[26:27], v[194:195]
	v_mov_b64_e32 v[28:29], v[196:197]
	v_mov_b64_e32 v[30:31], v[198:199]
	v_mov_b64_e32 v[32:33], v[200:201]
	v_mov_b64_e32 v[66:67], v[194:195]
	v_mov_b64_e32 v[68:69], v[196:197]
	v_mov_b64_e32 v[70:71], v[198:199]
	v_mov_b64_e32 v[72:73], v[200:201]
	v_mov_b64_e32 v[114:115], v[194:195]
	v_mov_b64_e32 v[116:117], v[196:197]
	v_mov_b64_e32 v[118:119], v[198:199]
	v_mov_b64_e32 v[120:121], v[200:201]
	v_mov_b64_e32 v[178:179], v[194:195]
	v_mov_b64_e32 v[180:181], v[196:197]
	v_mov_b64_e32 v[182:183], v[198:199]
	v_mov_b64_e32 v[184:185], v[200:201]
	v_mov_b64_e32 v[186:187], v[194:195]
	v_mov_b64_e32 v[188:189], v[196:197]
	v_mov_b64_e32 v[190:191], v[198:199]
	v_mov_b64_e32 v[192:193], v[200:201]
	v_mov_b64_e32 v[232:233], v[200:201]
	v_mov_b64_e32 v[230:231], v[198:199]
	v_mov_b64_e32 v[228:229], v[196:197]
	v_mov_b64_e32 v[226:227], v[194:195]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[0:1] /*v[256:257]*/, v[200:201]
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[254:255], v[198:199]
	v_mov_b64_e32 v[252:253], v[196:197]
	v_mov_b64_e32 v[250:251], v[194:195]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[16:17] /*v[272:273]*/, v[200:201]
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, v[198:199]
	v_mov_b64_e32 v[12:13] /*v[268:269]*/, v[196:197]
	v_mov_b64_e32 v[10:11] /*v[266:267]*/, v[194:195]
	v_mov_b64_e32 v[32:33] /*v[288:289]*/, v[200:201]
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[198:199]
	v_mov_b64_e32 v[28:29] /*v[284:285]*/, v[196:197]
	v_mov_b64_e32 v[26:27] /*v[282:283]*/, v[194:195]
	v_mov_b64_e32 v[40:41] /*v[296:297]*/, v[200:201]
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[198:199]
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[196:197]
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[194:195]
	v_mov_b64_e32 v[48:49] /*v[304:305]*/, v[200:201]
	v_mov_b64_e32 v[46:47] /*v[302:303]*/, v[198:199]
	v_mov_b64_e32 v[44:45] /*v[300:301]*/, v[196:197]
	v_mov_b64_e32 v[42:43] /*v[298:299]*/, v[194:195]
	v_mov_b64_e32 v[64:65] /*v[320:321]*/, v[200:201]
	v_mov_b64_e32 v[62:63] /*v[318:319]*/, v[198:199]
	v_mov_b64_e32 v[60:61] /*v[316:317]*/, v[196:197]
	v_mov_b64_e32 v[58:59] /*v[314:315]*/, v[194:195]
	v_mov_b64_e32 v[56:57] /*v[312:313]*/, v[200:201]
	v_mov_b64_e32 v[54:55] /*v[310:311]*/, v[198:199]
	v_mov_b64_e32 v[52:53] /*v[308:309]*/, v[196:197]
	v_mov_b64_e32 v[50:51] /*v[306:307]*/, v[194:195]
	s_set_vgpr_msb 0x4000
.LBB0_6:
	s_add_co_i32 s16, s61, s18
	s_add_co_i32 s62, s62, -1
	s_add_co_i32 s24, s14, s15
	s_min_i32 s2, s16, s62
	s_mul_i32 s3, s24, s13
	s_max_i32 s2, s2, 0
	s_ashr_i32 s15, s14, 31
	s_lshl_b32 s4, s2, 5
	v_nop
	s_set_vgpr_msb 0x8a
	v_dual_lshlrev_b32 v32 /*v544*/, 2, v13 /*v525*/ :: v_dual_bitop2_b32 v31 /*v543*/, 32, v7 /*v519*/ bitop3:0x54
	s_add_co_i32 s2, s4, s3
	s_sub_co_i32 s19, s60, s61
	s_lshl_b32 s5, s2, 2
	s_add_co_i32 s2, s4, s35
	s_add_co_i32 s6, s5, 64
	s_ashr_i32 s3, s2, 31
	s_sub_co_i32 s4, s13, s4
	s_mul_u64 s[2:3], s[28:29], s[2:3]
	s_mov_b32 s42, s38
	s_add_nc_u64 s[2:3], s[2:3], s[14:15]
	s_mov_b32 s43, s39
	s_lshl_b64 s[2:3], s[2:3], 8
	s_cmp_lg_u32 s63, 0x80000000
	s_clause 0x1
	buffer_load_b32 v10 /*v522*/, v32 /*v544*/, s[36:39], s5 offen
	buffer_load_b32 v8 /*v520*/, v32 /*v544*/, s[36:39], s6 offen
	s_cselect_b32 s9, s63, 0x80
	s_max_i32 s4, s4, 0
	s_clause 0x1
	buffer_load_b32 v6 /*v518*/, v32 /*v544*/, s[40:43], s5 offen
	buffer_load_b32 v4 /*v516*/, v32 /*v544*/, s[40:43], s6 offen
	s_wait_xcnt 0x1
	s_lshl_b32 s5, s4, 16
	s_lshr_b32 s4, s4, 16
	s_wait_xcnt 0x0
	s_or_b32 s6, s5, 0x7fff
	s_ashr_i32 s5, s9, 31
	s_mov_b32 s11, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[22:23], s[54:55], s[2:3]
	s_or_b32 s7, s4, 0x800000
	s_and_b32 s10, s5, 0xffff
	s_bitset1_b32 s23, 31
	s_mov_b32 s20, 1
	s_mov_b32 s21, s11
	s_mov_b32 s8, 32
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, 0x7510000
	s_cmp_gt_i32 s17, 1
	tensor_load_to_lds s[20:23], s[4:11]
	s_add_nc_u64 s[22:23], s[52:53], s[2:3]
	s_cselect_b32 s3, -1, 0
	s_cmp_lt_i32 s17, 2
	s_mul_i32 s2, s19, s17
	s_cselect_b32 s15, -1, 0
	s_cmp_gt_i32 s2, 1
	s_bitset1_b32 s23, 31
	s_cselect_b32 s19, -1, 0
	s_movk_i32 s21, 0x2200
	s_and_b32 s15, s15, s19
	s_and_b32 s3, s3, s19
	s_cmp_lg_u32 s15, 0
	tensor_load_to_lds s[20:23], s[4:11]
	s_add_co_ci_u32 s15, s61, s18
	s_movk_i32 s21, 0x4400
	s_min_i32 s15, s15, s62
	v_or_b32_e32 v30 /*v542*/, 64, v7 /*v519*/
	s_max_i32 s15, s15, 0
	s_cmp_lg_u32 s3, 0
	v_or_b32_e32 v29 /*v541*/, 0x60, v7 /*v519*/
	s_add_co_ci_u32 s18, s14, 0
	s_lshl_b32 s3, s15, 5
	s_ashr_i32 s19, s18, 31
	s_add_co_i32 s26, s3, s35
	s_sub_co_i32 s3, s13, s3
	s_ashr_i32 s27, s26, 31
	s_max_i32 s3, s3, 0
	s_mul_u64 s[26:27], s[28:29], s[26:27]
	s_lshl_b32 s6, s3, 16
	s_add_nc_u64 s[18:19], s[26:27], s[18:19]
	s_lshr_b32 s3, s3, 16
	s_lshl_b64 s[18:19], s[18:19], 8
	s_addk_co_i32 s6, 0x7fff
	s_add_nc_u64 s[22:23], s[54:55], s[18:19]
	s_or_b32 s7, s3, 0x800000
	s_bitset1_b32 s23, 31
	v_or_b32_e32 v28 /*v540*/, 0x80, v7 /*v519*/
	tensor_load_to_lds s[20:23], s[4:11]
	s_add_nc_u64 s[22:23], s[52:53], s[18:19]
	s_movk_i32 s21, 0x6600
	s_bitset1_b32 s23, 31
	v_or_b32_e32 v27 /*v539*/, 0xa0, v7 /*v519*/
	tensor_load_to_lds s[20:23], s[4:11]
	v_or_b32_e32 v26 /*v538*/, 0xc0, v7 /*v519*/
	v_or_b32_e32 v25 /*v537*/, 0xe0, v7 /*v519*/
	v_or_b32_e32 v24 /*v536*/, 32, v5 /*v517*/
	v_or_b32_e32 v23 /*v535*/, 64, v5 /*v517*/
	v_or_b32_e32 v22 /*v534*/, 0x60, v5 /*v517*/
	v_or_b32_e32 v21 /*v533*/, 0x80, v5 /*v517*/
	v_or_b32_e32 v20 /*v532*/, 0xa0, v5 /*v517*/
	v_or_b32_e32 v19 /*v531*/, 0xc0, v5 /*v517*/
	s_set_vgpr_msb 0x8a08
	v_or_b32_e32 v1, 0xe0, v5 /*v517*/
	s_mov_b32 s15, 2
	s_wait_tensorcnt 0x2
	s_cmp_lt_i32 s2, 1
	s_set_vgpr_msb 0x800
	s_cbranch_scc1 .LBB0_9
	s_set_vgpr_msb 0x42
	ds_load_b128 v[134:137] /*v[390:393]*/, v14 /*v526*/ offset:4576
	ds_load_b128 v[130:133] /*v[386:389]*/, v14 /*v526*/ offset:4544
	ds_load_b128 v[150:153] /*v[406:409]*/, v14 /*v526*/ offset:4512
	ds_load_b128 v[146:149] /*v[402:405]*/, v14 /*v526*/ offset:4480
	ds_load_b128 v[166:169] /*v[422:425]*/, v14 /*v526*/ offset:4448
	ds_load_b128 v[162:165] /*v[418:421]*/, v14 /*v526*/ offset:4416
	ds_load_b128 v[198:201] /*v[454:457]*/, v14 /*v526*/ offset:4384
	ds_load_b128 v[194:197] /*v[450:453]*/, v14 /*v526*/ offset:4352
	ds_load_b128 v[182:185] /*v[438:441]*/, v14 /*v526*/ offset:13280
	ds_load_b128 v[178:181] /*v[434:437]*/, v14 /*v526*/ offset:13248
	ds_load_b128 v[214:217] /*v[470:473]*/, v14 /*v526*/ offset:13216
	ds_load_b128 v[210:213] /*v[466:469]*/, v14 /*v526*/ offset:13184
	ds_load_b128 v[230:233] /*v[486:489]*/, v14 /*v526*/ offset:13152
	ds_load_b128 v[226:229] /*v[482:485]*/, v14 /*v526*/ offset:13120
	ds_load_b128 v[246:249] /*v[502:505]*/, v14 /*v526*/ offset:13088
	ds_load_b128 v[242:245] /*v[498:501]*/, v14 /*v526*/ offset:13056
	ds_load_b128 v[142:145] /*v[398:401]*/, v14 /*v526*/ offset:224
	ds_load_b128 v[138:141] /*v[394:397]*/, v14 /*v526*/ offset:192
	ds_load_b128 v[158:161] /*v[414:417]*/, v14 /*v526*/ offset:160
	ds_load_b128 v[154:157] /*v[410:413]*/, v14 /*v526*/ offset:128
	ds_load_b128 v[174:177] /*v[430:433]*/, v14 /*v526*/ offset:96
	ds_load_b128 v[170:173] /*v[426:429]*/, v14 /*v526*/ offset:64
	ds_load_b128 v[206:209] /*v[462:465]*/, v14 /*v526*/ offset:32
	ds_load_b128 v[202:205] /*v[458:461]*/, v14 /*v526*/
	ds_load_b128 v[190:193] /*v[446:449]*/, v14 /*v526*/ offset:8928
	ds_load_b128 v[186:189] /*v[442:445]*/, v14 /*v526*/ offset:8896
	ds_load_b128 v[222:225] /*v[478:481]*/, v14 /*v526*/ offset:8864
	ds_load_b128 v[218:221] /*v[474:477]*/, v14 /*v526*/ offset:8832
	ds_load_b128 v[238:241] /*v[494:497]*/, v14 /*v526*/ offset:8800
	ds_load_b128 v[234:237] /*v[490:493]*/, v14 /*v526*/ offset:8768
	ds_load_b128 v[254:257] /*v[510:513]*/, v14 /*v526*/ offset:8736
	ds_load_b128 v[250:253] /*v[506:509]*/, v14 /*v526*/ offset:8704
	s_mov_b32 s6, 0x10a00
	s_set_vgpr_msb 0x428a
	v_or_b32_e32 v36 /*v548*/, 0x10000, v3 /*v515*/
	v_mad_u32_u24 v37 /*v549*/, 0x50, v9 /*v521*/, s6
	s_mov_b32 s6, s12
	s_mov_b32 s7, s12
	v_or_b32_e32 v33 /*v545*/, 0x10000, v2 /*v514*/
	v_mov_b64_e32 v[2:3] /*v[514:515]*/, s[6:7]
	v_add3_u32 v18 /*v530*/, v14 /*v526*/, v18 /*v530*/, 0x10000
	v_dual_add_nc_u32 v34 /*v546*/, v36 /*v548*/, v11 /*v523*/ :: v_dual_add_nc_u32 v35 /*v547*/, v37 /*v549*/, v11 /*v523*/
	v_dual_add_nc_u32 v36 /*v548*/, v36 /*v548*/, v17 /*v529*/ :: v_dual_add_nc_u32 v17 /*v529*/, v37 /*v549*/, v17 /*v529*/
	s_ashr_i32 s3, s2, 31
	s_mov_b32 s12, 0x3fb8aa3b
	s_mov_b64 s[18:19], s[2:3]
	s_mov_b32 s7, s11
	s_mov_b32 s6, s11
	s_mov_b32 s27, s11
	s_set_vgpr_msb 0x8a00
.LBB0_8:
	s_add_co_i32 s3, s7, 1
	s_add_co_i32 s21, s15, -1
	s_cmp_ge_i32 s3, s17
	s_cselect_b32 s22, -1, 0
	s_and_b32 s23, s22, exec_lo
	s_cselect_b32 s23, 1, 0
	s_cselect_b32 s3, 0, s3
	s_cmp_lg_u32 s22, 0
	s_add_co_ci_u32 s25, s6, 0
	s_cmp_lt_i32 s21, s2
	s_cselect_b32 s22, s25, s6
	s_cselect_b32 s44, s3, s7
	s_add_co_i32 s7, s3, 1
	s_cmp_ge_i32 s7, s17
	s_cselect_b32 s21, -1, 0
	s_and_b32 s26, s21, exec_lo
	s_cselect_b32 s7, 0, s7
	s_cmp_lg_u32 s21, 0
	s_add_co_ci_u32 s6, s6, s23
	s_cmp_lt_i32 s15, s2
	s_cselect_b32 s6, s6, s22
	s_cselect_b32 s7, s7, s44
	s_add_co_i32 s21, s27, 0xffffbc00
	s_add_co_i32 s6, s6, s16
	s_cmp_lg_u32 s27, 0
	s_cselect_b32 s21, s21, 0x8800
	s_add_co_i32 s23, s27, 0x4400
	s_cmp_lg_u32 s27, 0x8800
	s_cselect_b32 s26, s23, 0
	s_add_co_i32 s45, s22, s16
	s_lshl_b32 s42, s6, 5
	s_add_co_i32 s6, s7, s14
	s_add_co_i32 s22, s42, s35
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s23, s22, 31
	s_sub_co_i32 s42, s13, s42
	s_mul_u64 s[22:23], s[28:29], s[22:23]
	s_max_i32 s46, s42, 0
	s_add_nc_u64 s[6:7], s[22:23], s[6:7]
	s_lshl_b32 s47, s46, 16
	s_lshl_b64 s[42:43], s[6:7], 8
	s_lshr_b32 s7, s46, 16
	s_add_nc_u64 s[22:23], s[54:55], s[42:43]
	s_or_b32 s6, s47, 0x7fff
	s_bitset1_b32 s23, 31
	s_bitset1_b32 s7, 23
	tensor_load_to_lds s[20:23], s[4:11]
	s_add_nc_u64 s[22:23], s[52:53], s[42:43]
	s_addk_co_i32 s21, 0x2200
	s_bitset1_b32 s23, 31
	tensor_load_to_lds s[20:23], s[4:11]
	s_add_co_i32 s6, s24, s44
	s_lshl_b32 s7, s45, 7
	s_mul_i32 s6, s33, s6
	s_mov_b32 s42, s38
	s_add_co_i32 s6, s7, s6
	s_mov_b32 s43, s39
	s_add_co_i32 s7, s6, 64
	s_set_vgpr_msb 0x82
	s_clause 0x1
	buffer_load_b32 v198 /*v710*/, v32 /*v544*/, s[40:43], s6 offen
	buffer_load_b32 v199 /*v711*/, v32 /*v544*/, s[40:43], s7 offen
	s_clause 0x1
	buffer_load_b32 v200 /*v712*/, v32 /*v544*/, s[36:39], s7 offen
	buffer_load_b32 v37 /*v549*/, v32 /*v544*/, s[36:39], s6 offen
	s_set_vgpr_msb 0x8284
	s_wait_loadcnt_dscnt 0x2600
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[34:41], v[250:257] /*v[506:513]*/, 0
	s_wait_loadcnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[74:81], v[250:257] /*v[506:513]*/, 0
	s_set_vgpr_msb 0x8444
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[34:41], v[242:249] /*v[498:505]*/, 0
	s_set_vgpr_msb 0x44a4
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[74:81], v[242:249] /*v[498:505]*/, 0
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[42:49], v[234:241] /*v[490:497]*/, v[38:45] /*v[550:557]*/
	s_wait_loadcnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[82:89], v[234:241] /*v[490:497]*/, v[46:53] /*v[558:565]*/
	s_set_vgpr_msb 0xa454
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[42:49], v[226:233] /*v[482:489]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x54a4
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[82:89], v[226:233] /*v[482:489]*/, v[54:61] /*v[566:573]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[50:57], v[218:225] /*v[474:481]*/, v[38:45] /*v[550:557]*/
	s_wait_loadcnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[90:97], v[218:225] /*v[474:481]*/, v[46:53] /*v[558:565]*/
	s_set_vgpr_msb 0xa454
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[50:57], v[210:217] /*v[466:473]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x54a4
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[90:97], v[210:217] /*v[466:473]*/, v[54:61] /*v[566:573]*/
	s_set_vgpr_msb 0xa444
	s_wait_loadcnt 0x16
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[106:113], v[202:209] /*v[458:465]*/, 0
	s_wait_loadcnt 0xe
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[146:153], v[202:209] /*v[458:465]*/, 0
	s_set_vgpr_msb 0x44a4
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[58:65], v[186:193] /*v[442:449]*/, v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[98:105], v[186:193] /*v[442:449]*/, v[46:53] /*v[558:565]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa44a
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[2:3] /*v[514:515]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[2:3] /*v[514:515]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[2:3] /*v[514:515]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x4a54
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[122:129], v[170:177] /*v[426:433]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5449
	s_wait_loadcnt 0x7
	v_pk_add_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, v[10:11] /*v[522:523]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, v[10:11] /*v[522:523]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, v[10:11] /*v[522:523]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, s[12:13] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[154:161], v[170:177] /*v[426:433]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5441
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, s[12:13] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4144
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[106:113], v[194:201] /*v[450:457]*/, 0
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[146:153], v[194:201] /*v[450:457]*/, 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x444a
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[2:3] /*v[514:515]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[2:3] /*v[514:515]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[2:3] /*v[514:515]*/, v[50:51] /*v[562:563]*/
	s_set_vgpr_msb 0x4a54
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[58:65], v[178:185] /*v[434:441]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x544a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[2:3] /*v[514:515]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x4a49
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/, v[10:11] /*v[522:523]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[2:3] /*v[514:515]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4a49
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[10:11] /*v[522:523]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[10:11] /*v[522:523]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[10:11] /*v[522:523]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, s[12:13] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x49a4
	v_wmma_f32_16x16x32_bf16 v[54:61] /*v[566:573]*/, v[98:105], v[178:185] /*v[434:441]*/, v[54:61] /*v[566:573]*/
	s_set_vgpr_msb 0xa449
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[10:11] /*v[522:523]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[254:255] /*v[510:511]*/, v[2:3] /*v[514:515]*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[250:251] /*v[506:507]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[130:137], v[154:161] /*v[410:417]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x544a
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[2:3] /*v[514:515]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[2:3] /*v[514:515]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[2:3] /*v[514:515]*/, v[58:59] /*v[570:571]*/
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[2:3] /*v[514:515]*/, v[60:61] /*v[572:573]*/
	s_set_vgpr_msb 0x4a49
	s_wait_loadcnt 0x6
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[234:235] /*v[490:491]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[236:237] /*v[492:493]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[162:169], v[154:161] /*v[410:417]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5449
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[238:239] /*v[494:495]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[240:241] /*v[496:497]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[252:253] /*v[508:509]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x494a
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[2:3] /*v[514:515]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x4a41
	v_exp_f32_e32 v154 /*v410*/, v194 /*v450*/
	v_exp_f32_e32 v155 /*v411*/, v195 /*v451*/
	v_exp_f32_e32 v156 /*v412*/, v196 /*v452*/
	s_set_vgpr_msb 0x4154
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[122:129], v[162:169] /*v[418:425]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5449
	v_exp_f32_e32 v157 /*v413*/, v197 /*v453*/
	v_exp_f32_e32 v158 /*v414*/, v198 /*v454*/
	v_exp_f32_e32 v159 /*v415*/, v199 /*v455*/
	v_exp_f32_e32 v160 /*v416*/, v200 /*v456*/
	v_exp_f32_e32 v161 /*v417*/, v201 /*v457*/
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[154:161], v[162:169] /*v[418:425]*/, v[226:233] /*v[482:489]*/
	s_set_vgpr_msb 0x5449
	v_pk_add_f32 v[184:185] /*v[440:441]*/, v[184:185] /*v[440:441]*/, v[8:9] /*v[520:521]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, s[12:13] op_sel_hi:[1,0]
	v_nop
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[182:183] /*v[438:439]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[170:171] /*v[426:427]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[172:173] /*v[428:429]*/, s[12:13] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[138:145], v[138:145] /*v[394:401]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5441
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[174:175] /*v[430:431]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[176:177] /*v[432:433]*/, s[12:13] op_sel_hi:[1,0]
	v_exp_f32_e32 v174 /*v430*/, v192 /*v448*/
	v_exp_f32_e32 v175 /*v431*/, v193 /*v449*/
	v_exp_f32_e32 v176 /*v432*/, v186 /*v442*/
	v_exp_f32_e32 v177 /*v433*/, v187 /*v443*/
	v_exp_f32_e32 v182 /*v438*/, v188 /*v444*/
	s_set_vgpr_msb 0x4154
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[170:177], v[138:145] /*v[394:401]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5441
	v_exp_f32_e32 v183 /*v439*/, v189 /*v445*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[184:185] /*v[440:441]*/, s[12:13] op_sel_hi:[1,0]
	v_exp_f32_e32 v178 /*v434*/, v178 /*v434*/
	v_exp_f32_e32 v179 /*v435*/, v179 /*v435*/
	v_exp_f32_e32 v180 /*v436*/, v180 /*v436*/
	v_exp_f32_e32 v181 /*v437*/, v181 /*v437*/
	v_exp_f32_e32 v162 /*v418*/, v162 /*v418*/
	s_set_vgpr_msb 0x4154
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[130:137], v[146:153] /*v[402:409]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5449
	s_wait_loadcnt 0x5
	v_pk_add_f32 v[138:139] /*v[394:395]*/, v[218:219] /*v[474:475]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[220:221] /*v[476:477]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[222:223] /*v[478:479]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[144:145] /*v[400:401]*/, v[224:225] /*v[480:481]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v163 /*v419*/, v163 /*v419*/
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[138:139] /*v[394:395]*/, v[154:155] /*v[410:411]*/
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, v[156:157] /*v[412:413]*/
	s_set_vgpr_msb 0x4554
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[162:169], v[146:153] /*v[402:409]*/, v[226:233] /*v[482:489]*/
	s_set_vgpr_msb 0x5445
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, v[160:161] /*v[416:417]*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[2:3] /*v[514:515]*/, v[138:139] /*v[394:395]*/
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[2:3] /*v[514:515]*/, v[140:141] /*v[396:397]*/
	v_pk_add_f32 v[146:147] /*v[402:403]*/, v[6:7] /*v[518:519]*/, v[210:211] /*v[466:467]*/ op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[1,0]
	v_pk_add_f32 v[148:149] /*v[404:405]*/, v[6:7] /*v[518:519]*/, v[212:213] /*v[468:469]*/ op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[1,0]
	v_pk_add_f32 v[150:151] /*v[406:407]*/, v[6:7] /*v[518:519]*/, v[214:215] /*v[470:471]*/ op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x4654
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[138:145], v[130:137] /*v[386:393]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5446
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[2:3] /*v[514:515]*/, v[142:143] /*v[398:399]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[146:147] /*v[402:403]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[148:149] /*v[404:405]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[150:151] /*v[406:407]*/, v[182:183] /*v[438:439]*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[2:3] /*v[514:515]*/, v[144:145] /*v[400:401]*/
	s_set_vgpr_msb 0x4645
	v_cvt_pk_bf16_f32 v138 /*v394*/, v138 /*v394*/, v139 /*v395*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[2:3] /*v[514:515]*/, v[146:147] /*v[402:403]*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[2:3] /*v[514:515]*/, v[148:149] /*v[404:405]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[2:3] /*v[514:515]*/, v[150:151] /*v[406:407]*/
	s_set_vgpr_msb 0x4645
	v_cvt_pk_bf16_f32 v139 /*v395*/, v140 /*v396*/, v141 /*v397*/
	v_cvt_pk_bf16_f32 v140 /*v396*/, v142 /*v398*/, v143 /*v399*/
	v_cvt_pk_bf16_f32 v146 /*v402*/, v146 /*v402*/, v147 /*v403*/
	v_cvt_pk_bf16_f32 v147 /*v403*/, v148 /*v404*/, v149 /*v405*/
	v_cvt_pk_bf16_f32 v148 /*v404*/, v150 /*v406*/, v151 /*v407*/
	v_cvt_pk_bf16_f32 v150 /*v406*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v141 /*v397*/, v144 /*v400*/, v145 /*v401*/
	v_cvt_pk_bf16_f32 v145 /*v401*/, v160 /*v416*/, v161 /*v417*/
	v_exp_f32_e32 v160 /*v416*/, v164 /*v420*/
	v_exp_f32_e32 v161 /*v417*/, v165 /*v421*/
	s_set_vgpr_msb 0x4549
	s_wait_loadcnt 0x4
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[202:203] /*v[458:459]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[204:205] /*v[460:461]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[206:207] /*v[462:463]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v144 /*v400*/, v158 /*v414*/, v159 /*v415*/
	s_set_vgpr_msb 0x4554
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[170:177], v[130:137] /*v[386:393]*/, v[226:233] /*v[482:489]*/
	s_set_vgpr_msb 0x5445
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[142:143] /*v[398:399]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[164:165] /*v[420:421]*/, v[180:181] /*v[436:437]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[162:163] /*v[418:419]*/
	v_cvt_pk_bf16_f32 v143 /*v399*/, v156 /*v412*/, v157 /*v413*/
	v_exp_f32_e32 v184 /*v440*/, v190 /*v446*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[2:3] /*v[514:515]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[2:3] /*v[514:515]*/, v[164:165] /*v[420:421]*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[2:3] /*v[514:515]*/, v[174:175] /*v[430:431]*/
	s_set_vgpr_msb 0x4645
	v_exp_f32_e32 v185 /*v441*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v142 /*v398*/, v154 /*v410*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v154 /*v410*/, v156 /*v412*/, v157 /*v413*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[216:217] /*v[472:473]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v156 /*v412*/, v164 /*v420*/, v165 /*v421*/
	v_exp_f32_e32 v164 /*v420*/, v166 /*v422*/
	v_exp_f32_e32 v165 /*v421*/, v167 /*v423*/
	v_exp_f32_e32 v166 /*v422*/, v168 /*v424*/
	v_exp_f32_e32 v167 /*v423*/, v169 /*v425*/
	v_exp_f32_e32 v168 /*v424*/, v170 /*v426*/
	v_exp_f32_e32 v169 /*v425*/, v171 /*v427*/
	v_exp_f32_e32 v170 /*v426*/, v172 /*v428*/
	v_exp_f32_e32 v171 /*v427*/, v173 /*v429*/
	v_cvt_pk_bf16_f32 v151 /*v407*/, v176 /*v432*/, v177 /*v433*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[208:209] /*v[464:465]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[226:227] /*v[482:483]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[228:229] /*v[484:485]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[134:135] /*v[390:391]*/, v[230:231] /*v[486:487]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[136:137] /*v[392:393]*/, v[232:233] /*v[488:489]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[184:185] /*v[440:441]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, v[160:161] /*v[416:417]*/
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[164:165] /*v[420:421]*/
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[166:167] /*v[422:423]*/
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[134:135] /*v[390:391]*/, v[168:169] /*v[424:425]*/
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[136:137] /*v[392:393]*/, v[170:171] /*v[426:427]*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[2:3] /*v[514:515]*/, v[152:153] /*v[408:409]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[2:3] /*v[514:515]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[2:3] /*v[514:515]*/, v[130:131] /*v[386:387]*/
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[2:3] /*v[514:515]*/, v[132:133] /*v[388:389]*/
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[2:3] /*v[514:515]*/, v[134:135] /*v[390:391]*/
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[2:3] /*v[514:515]*/, v[136:137] /*v[392:393]*/
	s_set_vgpr_msb 0x4645
	v_cvt_pk_bf16_f32 v149 /*v405*/, v152 /*v408*/, v153 /*v409*/
	v_cvt_pk_bf16_f32 v153 /*v409*/, v184 /*v440*/, v185 /*v441*/
	v_cvt_pk_bf16_f32 v152 /*v408*/, v182 /*v438*/, v183 /*v439*/
	v_cvt_pk_bf16_f32 v155 /*v411*/, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v160 /*v416*/, v161 /*v417*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v130 /*v386*/, v130 /*v386*/, v131 /*v387*/
	v_cvt_pk_bf16_f32 v131 /*v387*/, v132 /*v388*/, v133 /*v389*/
	v_cvt_pk_bf16_f32 v132 /*v388*/, v134 /*v390*/, v135 /*v391*/
	v_cvt_pk_bf16_f32 v133 /*v389*/, v136 /*v392*/, v137 /*v393*/
	v_cvt_pk_bf16_f32 v137 /*v393*/, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v136 /*v392*/, v168 /*v424*/, v169 /*v425*/
	v_cvt_pk_bf16_f32 v135 /*v391*/, v166 /*v422*/, v167 /*v423*/
	v_cvt_pk_bf16_f32 v134 /*v390*/, v164 /*v420*/, v165 /*v421*/
	s_set_vgpr_msb 0x4548
	v_add_nc_u32_e32 v162 /*v418*/, s27, v15 /*v527*/
	s_set_vgpr_msb 0x4888
	v_add_nc_u32_e32 v4 /*v516*/, s26, v14 /*v526*/
	s_set_vgpr_msb 0x8886
	ds_store_b128 v18 /*v530*/, v[150:153] /*v[406:409]*/
	ds_store_b128 v18 /*v530*/, v[142:145] /*v[398:401]*/ offset:32
	ds_store_b128 v18 /*v530*/, v[146:149] /*v[402:405]*/ offset:2560
	ds_store_b128 v18 /*v530*/, v[138:141] /*v[394:397]*/ offset:2592
	ds_store_b128 v33 /*v545*/, v[158:161] /*v[414:417]*/
	ds_store_b128 v33 /*v545*/, v[134:137] /*v[390:393]*/ offset:32
	ds_store_b128 v33 /*v545*/, v[154:157] /*v[410:413]*/ offset:2560
	ds_store_b128 v33 /*v545*/, v[130:133] /*v[386:389]*/ offset:2592
	ds_load_tr16_b128 v[38:41] /*v[550:553]*/, v34 /*v546*/
	ds_load_tr16_b128 v[42:45] /*v[554:557]*/, v34 /*v546*/ offset:1280
	s_set_vgpr_msb 0x8681
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v162 /*v418*/
	ds_load_tr16_b128 v[54:57] /*v[566:569]*/, v162 /*v418*/ offset:32
	ds_load_tr16_b128 v[50:53] /*v[562:565]*/, v162 /*v418*/ offset:4352
	ds_load_tr16_b128 v[58:61] /*v[570:573]*/, v162 /*v418*/ offset:4384
	ds_load_tr16_b128 v[62:65] /*v[574:577]*/, v162 /*v418*/ offset:64
	ds_load_tr16_b128 v[70:73] /*v[582:585]*/, v162 /*v418*/ offset:96
	ds_load_tr16_b128 v[66:69] /*v[578:581]*/, v162 /*v418*/ offset:4416
	ds_load_tr16_b128 v[74:77] /*v[586:589]*/, v162 /*v418*/ offset:4448
	ds_load_tr16_b128 v[78:81] /*v[590:593]*/, v162 /*v418*/ offset:128
	ds_load_tr16_b128 v[86:89] /*v[598:601]*/, v162 /*v418*/ offset:160
	ds_load_tr16_b128 v[82:85] /*v[594:597]*/, v162 /*v418*/ offset:4480
	ds_load_tr16_b128 v[90:93] /*v[602:605]*/, v162 /*v418*/ offset:4512
	ds_load_tr16_b128 v[94:97] /*v[606:609]*/, v162 /*v418*/ offset:192
	ds_load_tr16_b128 v[102:105] /*v[614:617]*/, v162 /*v418*/ offset:224
	ds_load_tr16_b128 v[98:101] /*v[610:613]*/, v162 /*v418*/ offset:4544
	ds_load_tr16_b128 v[106:109] /*v[618:621]*/, v162 /*v418*/ offset:4576
	s_set_vgpr_msb 0x8182
	ds_load_tr16_b128 v[110:113] /*v[622:625]*/, v35 /*v547*/
	ds_load_tr16_b128 v[114:117] /*v[626:629]*/, v35 /*v547*/ offset:1280
	s_set_vgpr_msb 0x8281
	ds_load_tr16_b128 v[118:121] /*v[630:633]*/, v162 /*v418*/ offset:8704
	ds_load_tr16_b128 v[126:129] /*v[638:641]*/, v162 /*v418*/ offset:8736
	ds_load_tr16_b128 v[122:125] /*v[634:637]*/, v162 /*v418*/ offset:13056
	ds_load_tr16_b128 v[130:133] /*v[642:645]*/, v162 /*v418*/ offset:13088
	ds_load_tr16_b128 v[134:137] /*v[646:649]*/, v162 /*v418*/ offset:8768
	ds_load_tr16_b128 v[142:145] /*v[654:657]*/, v162 /*v418*/ offset:8800
	ds_load_tr16_b128 v[138:141] /*v[650:653]*/, v162 /*v418*/ offset:13120
	ds_load_tr16_b128 v[146:149] /*v[658:661]*/, v162 /*v418*/ offset:13152
	ds_load_tr16_b128 v[150:153] /*v[662:665]*/, v162 /*v418*/ offset:8832
	ds_load_tr16_b128 v[158:161] /*v[670:673]*/, v162 /*v418*/ offset:8864
	ds_load_tr16_b128 v[154:157] /*v[666:669]*/, v162 /*v418*/ offset:13184
	ds_load_tr16_b128 v[162:165] /*v[674:677]*/, v162 /*v418*/ offset:13216
	ds_load_tr16_b128 v[166:169] /*v[678:681]*/, v162 /*v418*/ offset:8896
	ds_load_tr16_b128 v[174:177] /*v[686:689]*/, v162 /*v418*/ offset:8928
	ds_load_tr16_b128 v[170:173] /*v[682:685]*/, v162 /*v418*/ offset:13248
	ds_load_tr16_b128 v[178:181] /*v[690:693]*/, v162 /*v418*/ offset:13280
	s_set_vgpr_msb 0x8182
	ds_load_tr16_b128 v[182:185] /*v[694:697]*/, v36 /*v548*/
	ds_load_tr16_b128 v[186:189] /*v[698:701]*/, v36 /*v548*/ offset:1280
	ds_load_tr16_b128 v[190:193] /*v[702:705]*/, v17 /*v529*/
	ds_load_tr16_b128 v[194:197] /*v[706:709]*/, v17 /*v529*/ offset:1280
	s_wait_tensorcnt 0x2
	s_set_vgpr_msb 0x825a
	ds_load_b128 v[250:253] /*v[506:509]*/, v4 /*v516*/ offset:8704
	ds_load_b128 v[254:257] /*v[510:513]*/, v4 /*v516*/ offset:8736
	ds_load_b128 v[234:237] /*v[490:493]*/, v4 /*v516*/ offset:8768
	ds_load_b128 v[238:241] /*v[494:497]*/, v4 /*v516*/ offset:8800
	ds_load_b128 v[218:221] /*v[474:477]*/, v4 /*v516*/ offset:8832
	ds_load_b128 v[222:225] /*v[478:481]*/, v4 /*v516*/ offset:8864
	ds_load_b128 v[186:189] /*v[442:445]*/, v4 /*v516*/ offset:8896
	ds_load_b128 v[190:193] /*v[446:449]*/, v4 /*v516*/ offset:8928
	ds_load_b128 v[202:205] /*v[458:461]*/, v4 /*v516*/
	ds_load_b128 v[206:209] /*v[462:465]*/, v4 /*v516*/ offset:32
	ds_load_b128 v[170:173] /*v[426:429]*/, v4 /*v516*/ offset:64
	ds_load_b128 v[174:177] /*v[430:433]*/, v4 /*v516*/ offset:96
	ds_load_b128 v[154:157] /*v[410:413]*/, v4 /*v516*/ offset:128
	ds_load_b128 v[158:161] /*v[414:417]*/, v4 /*v516*/ offset:160
	ds_load_b128 v[138:141] /*v[394:397]*/, v4 /*v516*/ offset:192
	ds_load_b128 v[142:145] /*v[398:401]*/, v4 /*v516*/ offset:224
	ds_load_b128 v[242:245] /*v[498:501]*/, v4 /*v516*/ offset:13056
	ds_load_b128 v[246:249] /*v[502:505]*/, v4 /*v516*/ offset:13088
	ds_load_b128 v[226:229] /*v[482:485]*/, v4 /*v516*/ offset:13120
	ds_load_b128 v[230:233] /*v[486:489]*/, v4 /*v516*/ offset:13152
	ds_load_b128 v[210:213] /*v[466:469]*/, v4 /*v516*/ offset:13184
	ds_load_b128 v[214:217] /*v[470:473]*/, v4 /*v516*/ offset:13216
	ds_load_b128 v[178:181] /*v[434:437]*/, v4 /*v516*/ offset:13248
	ds_load_b128 v[182:185] /*v[438:441]*/, v4 /*v516*/ offset:13280
	ds_load_b128 v[194:197] /*v[450:453]*/, v4 /*v516*/ offset:4352
	ds_load_b128 v[198:201] /*v[454:457]*/, v4 /*v516*/ offset:4384
	ds_load_b128 v[162:165] /*v[418:421]*/, v4 /*v516*/ offset:4416
	ds_load_b128 v[166:169] /*v[422:425]*/, v4 /*v516*/ offset:4448
	ds_load_b128 v[146:149] /*v[402:405]*/, v4 /*v516*/ offset:4480
	ds_load_b128 v[150:153] /*v[406:409]*/, v4 /*v516*/ offset:4512
	ds_load_b128 v[130:133] /*v[386:389]*/, v4 /*v516*/ offset:4544
	ds_load_b128 v[134:137] /*v[390:393]*/, v4 /*v516*/ offset:4576
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[38:45] /*v[550:557]*/, v[46:53] /*v[558:565]*/, v[50:57] /*v[306:313]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[38:45] /*v[550:557]*/, v[54:61] /*v[566:573]*/, v[58:65] /*v[314:321]*/ matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[38:45] /*v[550:557]*/, v[62:69] /*v[574:581]*/, v[42:49] /*v[298:305]*/ matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[38:45] /*v[550:557]*/, v[70:77] /*v[582:589]*/, v[34:41] /*v[290:297]*/ matrix_a_reuse
	s_wait_dscnt 0x3b
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[38:45] /*v[550:557]*/, v[78:85] /*v[590:597]*/, v[26:33] /*v[282:289]*/ matrix_a_reuse
	s_wait_dscnt 0x3a
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[38:45] /*v[550:557]*/, v[86:93] /*v[598:605]*/, v[10:17] /*v[266:273]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5a0a
	s_wait_dscnt 0x37
	v_wmma_f32_16x16x32_bf16 v[250:257], v[38:45] /*v[550:557]*/, v[94:101] /*v[606:613]*/, v[250:257] matrix_a_reuse
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[226:233], v[38:45] /*v[550:557]*/, v[102:109] /*v[614:621]*/, v[226:233] matrix_a_reuse
	s_set_vgpr_msb 0xa5a
	s_wait_dscnt 0x31
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[110:117] /*v[622:629]*/, v[118:125] /*v[630:637]*/, v[122:129] /*v[378:385]*/
	s_wait_dscnt 0x30
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[110:117] /*v[622:629]*/, v[126:133] /*v[638:645]*/, v[114:121] /*v[370:377]*/ matrix_a_reuse
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[110:117] /*v[622:629]*/, v[134:141] /*v[646:653]*/, v[106:113] /*v[362:369]*/ matrix_a_reuse
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[110:117] /*v[622:629]*/, v[142:149] /*v[654:661]*/, v[98:105] /*v[354:361]*/ matrix_a_reuse
	s_wait_dscnt 0x29
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[110:117] /*v[622:629]*/, v[150:157] /*v[662:669]*/, v[90:97] /*v[346:353]*/ matrix_a_reuse
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[110:117] /*v[622:629]*/, v[158:165] /*v[670:677]*/, v[82:89] /*v[338:345]*/ matrix_a_reuse
	s_wait_dscnt 0x25
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[110:117] /*v[622:629]*/, v[166:173] /*v[678:685]*/, v[74:81] /*v[330:337]*/ matrix_a_reuse
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[110:117] /*v[622:629]*/, v[174:181] /*v[686:693]*/, v[66:73] /*v[322:329]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5a0a
	s_wait_dscnt 0x22
	v_wmma_f32_16x16x32_bf16 v[186:193], v[182:189] /*v[694:701]*/, v[46:53] /*v[558:565]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[178:185], v[182:189] /*v[694:701]*/, v[54:61] /*v[566:573]*/, v[178:185] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[114:121], v[182:189] /*v[694:701]*/, v[62:69] /*v[574:581]*/, v[114:121] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[182:189] /*v[694:701]*/, v[70:77] /*v[582:589]*/, v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[182:189] /*v[694:701]*/, v[78:85] /*v[590:597]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[182:189] /*v[694:701]*/, v[86:93] /*v[598:605]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[182:189] /*v[694:701]*/, v[94:101] /*v[606:613]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[182:189] /*v[694:701]*/, v[102:109] /*v[614:621]*/, v[2:9] matrix_a_reuse
	s_set_vgpr_msb 0xa5a
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[190:197] /*v[702:709]*/, v[118:125] /*v[630:637]*/, v[18:25] /*v[274:281]*/
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[190:197] /*v[702:709]*/, v[126:133] /*v[638:645]*/, v[2:9] /*v[258:265]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[242:249], v[190:197] /*v[702:709]*/, v[134:141] /*v[646:653]*/, v[242:249] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[234:241], v[190:197] /*v[702:709]*/, v[142:149] /*v[654:661]*/, v[234:241] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[218:225], v[190:197] /*v[702:709]*/, v[150:157] /*v[662:669]*/, v[218:225] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[210:217], v[190:197] /*v[702:709]*/, v[158:165] /*v[670:677]*/, v[210:217] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[202:209], v[190:197] /*v[702:709]*/, v[166:173] /*v[678:685]*/, v[202:209] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[194:201], v[190:197] /*v[702:709]*/, v[174:181] /*v[686:693]*/, v[194:201] matrix_a_reuse
	s_set_vgpr_msb 0xa82
	s_wait_loadcnt 0x1
	v_dual_mov_b32 v4 /*v516*/, v199 /*v711*/ :: v_dual_mov_b32 v8 /*v520*/, v200 /*v712*/
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v6 /*v518*/, v198 /*v710*/ :: v_dual_mov_b32 v10 /*v522*/, v37 /*v549*/
	s_add_nc_u64 s[18:19], s[18:19], -1
	s_add_co_i32 s15, s15, 1
	s_cmp_lg_u64 s[18:19], 0
	s_mov_b32 s7, s3
	s_mov_b32 s6, s25
	s_mov_b32 s27, s26
	s_set_vgpr_msb 0x8200
	s_cbranch_scc1 .LBB0_8
.LBB0_9:
	s_set_vgpr_msb 40
	s_wait_loadcnt 0x17
	v_mad_i32_i24 v90, 0xffffff20, v13 /*v525*/, v14 /*v526*/
	s_set_vgpr_msb 0x2805
	v_cvt_pk_bf16_f32 v37, v56 /*v312*/, v57 /*v313*/
	v_cvt_pk_bf16_f32 v36, v54 /*v310*/, v55 /*v311*/
	v_cvt_pk_bf16_f32 v35, v52 /*v308*/, v53 /*v309*/
	v_cvt_pk_bf16_f32 v34, v50 /*v306*/, v51 /*v307*/
	v_cvt_pk_bf16_f32 v41, v128 /*v384*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v40, v126 /*v382*/, v127 /*v383*/
	v_cvt_pk_bf16_f32 v39, v124 /*v380*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v38, v122 /*v378*/, v123 /*v379*/
	s_set_vgpr_msb 0x500
	v_or_b32_e32 v50, 48, v0
	s_clause 0x1
	s_load_b64 s[2:3], s[0:1], 0x110 nv
	s_load_b64 s[4:5], s[0:1], 0x140 nv
	s_wait_tensorcnt 0x0
	ds_store_b128 v90, v[34:37]
	ds_store_b128 v90, v[38:41] offset:6144
	s_set_vgpr_msb 34
	v_mad_u32_u24 v91, v16 /*v528*/, 48, v12 /*v524*/
	s_set_vgpr_msb 0x2205
	v_cvt_pk_bf16_f32 v37, v64 /*v320*/, v65 /*v321*/
	v_cvt_pk_bf16_f32 v36, v62 /*v318*/, v63 /*v319*/
	v_cvt_pk_bf16_f32 v35, v60 /*v316*/, v61 /*v317*/
	v_cvt_pk_bf16_f32 v34, v58 /*v314*/, v59 /*v315*/
	v_cvt_pk_bf16_f32 v41, v120 /*v376*/, v121 /*v377*/
	v_cvt_pk_bf16_f32 v40, v118 /*v374*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v39, v116 /*v372*/, v117 /*v373*/
	v_cvt_pk_bf16_f32 v38, v114 /*v370*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v45, v48 /*v304*/, v49 /*v305*/
	v_cvt_pk_bf16_f32 v44, v46 /*v302*/, v47 /*v303*/
	v_cvt_pk_bf16_f32 v43, v44 /*v300*/, v45 /*v301*/
	v_cvt_pk_bf16_f32 v42, v42 /*v298*/, v43 /*v299*/
	v_cvt_pk_bf16_f32 v49, v112 /*v368*/, v113 /*v369*/
	v_cvt_pk_bf16_f32 v48, v110 /*v366*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v47, v108 /*v364*/, v109 /*v365*/
	v_cvt_pk_bf16_f32 v46, v106 /*v362*/, v107 /*v363*/
	s_set_vgpr_msb 0x520
	v_mad_u32_u24 v92, v50, 48, v12 /*v524*/
	v_or_b32_e32 v50, 0x50, v0
	ds_store_b128 v91, v[34:37]
	ds_store_b128 v91, v[38:41] offset:6144
	ds_store_b128 v90, v[42:45] offset:1536
	ds_store_b128 v90, v[46:49] offset:7680
	s_set_vgpr_msb 0x2005
	v_cvt_pk_bf16_f32 v37, v40 /*v296*/, v41 /*v297*/
	v_cvt_pk_bf16_f32 v36, v38 /*v294*/, v39 /*v295*/
	v_cvt_pk_bf16_f32 v35, v36 /*v292*/, v37 /*v293*/
	v_cvt_pk_bf16_f32 v34, v34 /*v290*/, v35 /*v291*/
	v_cvt_pk_bf16_f32 v41, v104 /*v360*/, v105 /*v361*/
	v_cvt_pk_bf16_f32 v40, v102 /*v358*/, v103 /*v359*/
	v_cvt_pk_bf16_f32 v39, v100 /*v356*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v38, v98 /*v354*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v45, v32 /*v288*/, v33 /*v289*/
	v_cvt_pk_bf16_f32 v44, v30 /*v286*/, v31 /*v287*/
	v_cvt_pk_bf16_f32 v43, v28 /*v284*/, v29 /*v285*/
	v_cvt_pk_bf16_f32 v42, v26 /*v282*/, v27 /*v283*/
	s_set_vgpr_msb 0x500
	v_or_b32_e32 v0, 0x70, v0
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v49, v96 /*v352*/, v97 /*v353*/
	v_cvt_pk_bf16_f32 v48, v94 /*v350*/, v95 /*v351*/
	v_cvt_pk_bf16_f32 v47, v92 /*v348*/, v93 /*v349*/
	v_cvt_pk_bf16_f32 v46, v90 /*v346*/, v91 /*v347*/
	s_set_vgpr_msb 0x520
	v_mad_u32_u24 v93, v50, 48, v12 /*v524*/
	s_set_vgpr_msb 0x2005
	v_cvt_pk_bf16_f32 v53, v16 /*v272*/, v17 /*v273*/
	v_cvt_pk_bf16_f32 v52, v14 /*v270*/, v15 /*v271*/
	v_cvt_pk_bf16_f32 v51, v12 /*v268*/, v13 /*v269*/
	v_cvt_pk_bf16_f32 v50, v10 /*v266*/, v11 /*v267*/
	v_cvt_pk_bf16_f32 v57, v88 /*v344*/, v89 /*v345*/
	v_cvt_pk_bf16_f32 v56, v86 /*v342*/, v87 /*v343*/
	v_cvt_pk_bf16_f32 v55, v84 /*v340*/, v85 /*v341*/
	v_cvt_pk_bf16_f32 v54, v82 /*v338*/, v83 /*v339*/
	s_set_vgpr_msb 0x500
	ds_store_b128 v92, v[34:37]
	ds_store_b128 v92, v[38:41] offset:6144
	ds_store_b128 v90, v[42:45] offset:3072
	ds_store_b128 v90, v[46:49] offset:9216
	ds_store_b128 v93, v[50:53]
	ds_store_b128 v93, v[54:57] offset:6144
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v37, v0 /*v256*/, v1 /*v257*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v36, v254, v255
	v_cvt_pk_bf16_f32 v35, v252, v253
	v_cvt_pk_bf16_f32 v34, v250, v251
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v41, v80 /*v336*/, v81 /*v337*/
	v_cvt_pk_bf16_f32 v40, v78 /*v334*/, v79 /*v335*/
	v_cvt_pk_bf16_f32 v39, v76 /*v332*/, v77 /*v333*/
	v_cvt_pk_bf16_f32 v38, v74 /*v330*/, v75 /*v331*/
	s_set_vgpr_msb 0x522
	s_wait_loadcnt 0x16
	v_mad_u32_u24 v94, v9 /*v521*/, 48, v11 /*v523*/
	v_mad_u32_u24 v0, 48, v0, v12 /*v524*/
	s_set_vgpr_msb 0x2200
	v_cvt_pk_bf16_f32 v45, v232, v233
	v_cvt_pk_bf16_f32 v44, v230, v231
	v_cvt_pk_bf16_f32 v43, v228, v229
	v_cvt_pk_bf16_f32 v42, v226, v227
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v49, v72 /*v328*/, v73 /*v329*/
	v_cvt_pk_bf16_f32 v48, v70 /*v326*/, v71 /*v327*/
	v_cvt_pk_bf16_f32 v47, v68 /*v324*/, v69 /*v325*/
	v_cvt_pk_bf16_f32 v46, v66 /*v322*/, v67 /*v323*/
	s_set_vgpr_msb 0x500
	ds_store_b128 v90, v[34:37] offset:4608
	ds_store_b128 v90, v[38:41] offset:10752
	ds_store_b128 v0, v[42:45]
	ds_store_b128 v0, v[46:49] offset:6144
	ds_load_tr16_b128 v[34:37], v94
	ds_load_tr16_b128 v[38:41], v94 offset:6144
	ds_load_tr16_b128 v[42:45], v94 offset:768
	ds_load_tr16_b128 v[46:49], v94 offset:6912
	ds_load_tr16_b128 v[50:53], v94 offset:1536
	ds_load_tr16_b128 v[54:57], v94 offset:7680
	ds_load_tr16_b128 v[58:61], v94 offset:2304
	ds_load_tr16_b128 v[62:65], v94 offset:8448
	ds_load_tr16_b128 v[74:77], v94 offset:3072
	ds_load_tr16_b128 v[78:81], v94 offset:9216
	ds_load_tr16_b128 v[82:85], v94 offset:3840
	ds_load_tr16_b128 v[86:89], v94 offset:9984
	s_lshl_b32 s1, s34, 25
	s_mov_b32 s0, 0
	v_cvt_pk_bf16_f32 v25, v24, v25
	s_wait_kmcnt 0x0
	s_or_b64 s[28:29], s[2:3], s[0:1]
	s_or_b64 s[0:1], s[4:5], s[0:1]
	s_mov_b32 s2, s30
	s_mov_b32 s3, s31
	s_set_vgpr_msb 2
	s_wait_dscnt 0xb
	buffer_store_b128 v[34:37], v7 /*v519*/, s[28:31], null offen
	s_wait_dscnt 0xa
	buffer_store_b128 v[38:41], v7 /*v519*/, s[0:3], null offen
	s_wait_dscnt 0x9
	buffer_store_b128 v[42:45], v31 /*v543*/, s[28:31], null offen
	s_wait_dscnt 0x8
	buffer_store_b128 v[46:49], v31 /*v543*/, s[0:3], null offen
	s_wait_dscnt 0x7
	buffer_store_b128 v[50:53], v30 /*v542*/, s[28:31], null offen
	s_wait_dscnt 0x6
	buffer_store_b128 v[54:57], v30 /*v542*/, s[0:3], null offen
	s_wait_dscnt 0x5
	buffer_store_b128 v[58:61], v29 /*v541*/, s[28:31], null offen
	s_wait_dscnt 0x4
	buffer_store_b128 v[62:65], v29 /*v541*/, s[0:3], null offen
	s_wait_dscnt 0x3
	buffer_store_b128 v[74:77], v28 /*v540*/, s[28:31], null offen
	s_wait_dscnt 0x2
	buffer_store_b128 v[78:81], v28 /*v540*/, s[0:3], null offen
	s_wait_dscnt 0x1
	buffer_store_b128 v[82:85], v27 /*v539*/, s[28:31], null offen
	s_wait_dscnt 0x0
	buffer_store_b128 v[86:89], v27 /*v539*/, s[0:3], null offen
	s_set_vgpr_msb 0x200
	v_cvt_pk_bf16_f32 v45, v192, v193
	v_cvt_pk_bf16_f32 v44, v190, v191
	v_cvt_pk_bf16_f32 v43, v188, v189
	v_cvt_pk_bf16_f32 v42, v186, v187
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v49, v24 /*v280*/, v25 /*v281*/
	v_cvt_pk_bf16_f32 v48, v22 /*v278*/, v23 /*v279*/
	v_cvt_pk_bf16_f32 v47, v20 /*v276*/, v21 /*v277*/
	v_cvt_pk_bf16_f32 v46, v18 /*v274*/, v19 /*v275*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v53, v184, v185
	v_cvt_pk_bf16_f32 v52, v182, v183
	v_cvt_pk_bf16_f32 v51, v180, v181
	v_cvt_pk_bf16_f32 v50, v178, v179
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v57, v8 /*v264*/, v9 /*v265*/
	v_cvt_pk_bf16_f32 v56, v6 /*v262*/, v7 /*v263*/
	v_cvt_pk_bf16_f32 v55, v4 /*v260*/, v5 /*v261*/
	v_cvt_pk_bf16_f32 v54, v2 /*v258*/, v3 /*v259*/
	s_set_vgpr_msb 0x500
	ds_load_tr16_b128 v[34:37], v94 offset:4608
	ds_load_tr16_b128 v[38:41], v94 offset:5376
	ds_load_tr16_b128 v[58:61], v94 offset:10752
	ds_load_tr16_b128 v[62:65], v94 offset:11520
	ds_store_b128 v90, v[42:45] offset:12288
	ds_store_b128 v90, v[46:49] offset:18432
	ds_store_b128 v91, v[50:53] offset:12288
	ds_store_b128 v91, v[54:57] offset:18432
	v_cvt_pk_bf16_f32 v45, v120, v121
	v_cvt_pk_bf16_f32 v44, v118, v119
	v_cvt_pk_bf16_f32 v43, v116, v117
	v_cvt_pk_bf16_f32 v42, v114, v115
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v23, v20, v21
	v_cvt_pk_bf16_f32 v22, v18, v19
	v_cvt_pk_bf16_f32 v49, v248, v249
	v_cvt_pk_bf16_f32 v48, v246, v247
	v_cvt_pk_bf16_f32 v47, v244, v245
	v_cvt_pk_bf16_f32 v46, v242, v243
	v_cvt_pk_bf16_f32 v21, v216, v217
	v_cvt_pk_bf16_f32 v20, v214, v215
	v_cvt_pk_bf16_f32 v19, v212, v213
	v_cvt_pk_bf16_f32 v18, v210, v211
	v_cvt_pk_bf16_f32 v53, v72, v73
	v_cvt_pk_bf16_f32 v52, v70, v71
	v_cvt_pk_bf16_f32 v51, v68, v69
	v_cvt_pk_bf16_f32 v50, v66, v67
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v15, v12, v13
	v_cvt_pk_bf16_f32 v14, v10, v11
	v_cvt_pk_bf16_f32 v57, v240, v241
	v_cvt_pk_bf16_f32 v56, v238, v239
	v_cvt_pk_bf16_f32 v55, v236, v237
	v_cvt_pk_bf16_f32 v54, v234, v235
	v_cvt_pk_bf16_f32 v13, v208, v209
	v_cvt_pk_bf16_f32 v12, v206, v207
	v_cvt_pk_bf16_f32 v11, v204, v205
	v_cvt_pk_bf16_f32 v10, v202, v203
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v28, v29
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v7, v4, v5
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_cvt_pk_bf16_f32 v29, v224, v225
	v_cvt_pk_bf16_f32 v28, v222, v223
	v_cvt_pk_bf16_f32 v27, v220, v221
	v_cvt_pk_bf16_f32 v26, v218, v219
	ds_store_b128 v90, v[42:45] offset:13824
	ds_store_b128 v90, v[46:49] offset:19968
	ds_store_b128 v92, v[50:53] offset:12288
	ds_store_b128 v92, v[54:57] offset:18432
	ds_store_b128 v90, v[30:33] offset:15360
	ds_store_b128 v90, v[26:29] offset:21504
	v_cvt_pk_bf16_f32 v5, v200, v201
	v_cvt_pk_bf16_f32 v4, v198, v199
	v_cvt_pk_bf16_f32 v3, v196, v197
	v_cvt_pk_bf16_f32 v2, v194, v195
	ds_store_b128 v93, v[22:25] offset:12288
	ds_store_b128 v93, v[18:21] offset:18432
	ds_store_b128 v90, v[14:17] offset:16896
	ds_store_b128 v90, v[10:13] offset:23040
	ds_store_b128 v0, v[6:9] offset:12288
	ds_store_b128 v0, v[2:5] offset:18432
	ds_load_tr16_b128 v[2:5], v94 offset:12288
	ds_load_tr16_b128 v[6:9], v94 offset:18432
	ds_load_tr16_b128 v[10:13], v94 offset:13056
	ds_load_tr16_b128 v[14:17], v94 offset:19200
	ds_load_tr16_b128 v[18:21], v94 offset:13824
	ds_load_tr16_b128 v[22:25], v94 offset:19968
	ds_load_tr16_b128 v[26:29], v94 offset:14592
	ds_load_tr16_b128 v[30:33], v94 offset:20736
	ds_load_tr16_b128 v[42:45], v94 offset:15360
	ds_load_tr16_b128 v[46:49], v94 offset:21504
	ds_load_tr16_b128 v[50:53], v94 offset:16128
	ds_load_tr16_b128 v[54:57], v94 offset:22272
	ds_load_tr16_b128 v[66:69], v94 offset:16896
	ds_load_tr16_b128 v[70:73], v94 offset:23040
	ds_load_tr16_b128 v[74:77], v94 offset:17664
	ds_load_tr16_b128 v[78:81], v94 offset:23808
	s_set_vgpr_msb 2
	s_wait_dscnt 0x23
	buffer_store_b128 v[34:37], v26 /*v538*/, s[28:31], null offen
	s_wait_dscnt 0x21
	buffer_store_b128 v[58:61], v26 /*v538*/, s[0:3], null offen
	buffer_store_b128 v[38:41], v25 /*v537*/, s[28:31], null offen
	s_wait_dscnt 0x20
	buffer_store_b128 v[62:65], v25 /*v537*/, s[0:3], null offen
	s_wait_dscnt 0xf
	buffer_store_b128 v[2:5], v5 /*v517*/, s[28:31], null offen
	s_wait_dscnt 0xe
	buffer_store_b128 v[6:9], v5 /*v517*/, s[0:3], null offen
	s_wait_dscnt 0xd
	buffer_store_b128 v[10:13], v24 /*v536*/, s[28:31], null offen
	s_wait_dscnt 0xc
	buffer_store_b128 v[14:17], v24 /*v536*/, s[0:3], null offen
	s_wait_dscnt 0xb
	buffer_store_b128 v[18:21], v23 /*v535*/, s[28:31], null offen
	s_wait_dscnt 0xa
	buffer_store_b128 v[22:25], v23 /*v535*/, s[0:3], null offen
	s_wait_dscnt 0x9
	buffer_store_b128 v[26:29], v22 /*v534*/, s[28:31], null offen
	s_wait_dscnt 0x8
	buffer_store_b128 v[30:33], v22 /*v534*/, s[0:3], null offen
	s_wait_dscnt 0x7
	buffer_store_b128 v[42:45], v21 /*v533*/, s[28:31], null offen
	s_wait_dscnt 0x6
	buffer_store_b128 v[46:49], v21 /*v533*/, s[0:3], null offen
	s_wait_dscnt 0x5
	buffer_store_b128 v[50:53], v20 /*v532*/, s[28:31], null offen
	s_wait_dscnt 0x4
	buffer_store_b128 v[54:57], v20 /*v532*/, s[0:3], null offen
	s_wait_dscnt 0x3
	buffer_store_b128 v[66:69], v19 /*v531*/, s[28:31], null offen
	s_wait_dscnt 0x2
	buffer_store_b128 v[70:73], v19 /*v531*/, s[0:3], null offen
	s_set_vgpr_msb 0x200
	s_wait_dscnt 0x1
	buffer_store_b128 v[74:77], v1, s[28:31], null offen
	s_wait_dscnt 0x0
	buffer_store_b128 v[78:81], v1, s[0:3], null offen
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	k_dkdv_0, .Lfunc_end0-k_dkdv_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dkdv_0
		.amdhsa_group_segment_fixed_size 70656
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
		.amdhsa_next_free_vgpr 713
		.amdhsa_next_free_sgpr 67
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

	.set .Lk_dkdv_0.num_vgpr, 713
	.set .Lk_dkdv_0.num_agpr, 0
	.set .Lk_dkdv_0.numbered_sgpr, 67
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
    .group_segment_fixed_size: 70656
    .kernarg_segment_align: 8
    .kernarg_segment_size: 408
    .max_flat_workgroup_size: 32
    .name:           k_dkdv_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     69
    .sgpr_spill_count: 0
    .symbol:         k_dkdv_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     713
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
