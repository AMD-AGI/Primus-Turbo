	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	k_dkdv_sp_0
	.p2align	8
	.type	k_dkdv_sp_0,@function
k_dkdv_sp_0:
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	v_nop
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_bfe_u32 s2, ttmp6, 0x40010
	s_load_b256 s[12:19], s[0:1], 0x18c nv
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
	s_load_b96 s[48:50], s[0:1], 0x1ac nv
	s_cselect_b32 s66, ttmp9, s2
	s_cselect_b32 s2, s3, s4
	s_bfe_u32 s3, ttmp6, 0x40014
	s_lshr_b32 s4, ttmp7, 16
	s_add_co_i32 s3, s3, 1
	s_bfe_u32 s5, ttmp6, 0x40008
	s_mul_i32 s3, s4, s3
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v13 /*v525*/, 15, v0
	s_add_co_i32 s5, s5, s3
	s_cmp_eq_u32 s6, 0
	s_set_vgpr_msb 0x8000
	v_lshrrev_b32_e32 v3, 4, v0
	s_cselect_b32 s33, s4, s5
	s_wait_kmcnt 0x0
	s_lshr_b32 s3, s18, 31
	s_lshl_b32 s22, s2, 5
	s_add_co_i32 s3, s18, s3
	s_mul_i32 s8, s14, s33
	s_and_b32 s2, s3, -2
	s_ashr_i32 s3, s3, 1
	s_cmp_lg_u32 s18, s2
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, s22, v13 /*v525*/
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s18, 0
	s_mul_i32 s51, s16, s14
	s_cselect_b32 s4, -1, 0
	s_mul_i32 s60, s51, s49
	s_and_b32 s2, s4, s2
	s_sub_co_ci_u32 s63, s3, 0
	s_sub_co_i32 s4, s22, s19
	s_mov_b32 s41, 0
	s_max_i32 s4, s4, 0
	s_mov_b32 s34, s15
	s_lshr_b32 s4, s4, 5
	s_cmp_lg_u32 s48, 0
	s_cselect_b32 s5, -1, 0
	s_and_b32 s5, s5, exec_lo
	s_cselect_b32 s64, s4, 0
	s_cmp_lg_u32 s2, 0
	s_sub_co_ci_u32 s67, s3, s64
	s_or_b32 s2, s22, 31
	s_sub_co_i32 s2, s2, s19
	s_add_co_i32 s3, s2, 31
	s_ashr_i32 s4, s3, 31
	s_lshr_b32 s4, s4, 27
	s_add_co_i32 s4, s3, s4
	s_and_b32 s5, s4, 0xffffffe0
	s_ashr_i32 s4, s4, 5
	s_cmp_lg_u32 s3, s5
	s_cselect_b32 s5, -1, 0
	s_cmp_lt_i32 s3, 0
	s_cselect_b32 s3, -1, 0
	s_and_b32 s3, s3, s5
	s_sub_co_ci_u32 s3, s4, 0
	s_cmp_gt_i32 s2, -1
	s_cselect_b32 s2, s3, 0
	s_min_i32 s2, s2, s63
	s_sub_co_i32 s2, s2, s64
	s_max_i32 s2, s2, 0
	s_min_i32 s2, s2, s67
	s_cmp_lg_u32 s48, 0
	s_cselect_b32 s72, -1, 0
	s_and_b32 s3, s72, exec_lo
	s_cselect_b32 s65, s2, 0
	s_abs_i32 s68, s50
	s_abs_i32 s3, s66
	s_cvt_f32_u32 s2, s68
	s_ashr_i32 s69, s50, 31
	v_s_rcp_f32 s2, s2
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_cvt_u32_f32 s70, s2
	s_sub_co_i32 s2, 0, s68
	s_mul_i32 s2, s2, s70
	s_mul_hi_u32 s2, s70, s2
	s_add_co_i32 s70, s70, s2
	s_ashr_i32 s2, s66, 31
	s_mul_hi_u32 s4, s3, s70
	s_xor_b32 s2, s2, s69
	s_mul_i32 s5, s4, s68
	s_sub_co_i32 s3, s3, s5
	s_add_co_i32 s5, s4, 1
	s_sub_co_i32 s6, s3, s68
	s_cmp_ge_u32 s3, s68
	s_cselect_b32 s4, s5, s4
	s_cselect_b32 s3, s6, s3
	s_add_co_i32 s5, s4, 1
	s_cmp_ge_u32 s3, s68
	s_cselect_b32 s3, s5, s4
	s_xor_b32 s3, s3, s2
	s_sub_co_i32 s4, s3, s2
	s_mul_i32 s4, s4, s50
	s_cmp_lg_u32 s66, s4
	s_load_b64 s[4:5], s[0:1], 0x30 nv
	s_cselect_b32 s6, -1, 0
	s_xor_b32 s7, s50, s66
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s7, -1, 0
	s_and_b32 s6, s7, s6
	s_sub_co_ci_u32 s48, s3, s2
	s_lshl_b32 s2, s16, 4
	s_or_b32 s18, s22, 16
	s_mul_i32 s8, s8, s2
	v_or_b32_e32 v2, s18, v13 /*v525*/
	s_lshl4_add_u32 s3, s48, s8
	s_load_b64 s[8:9], s[0:1], 0x60 nv
	v_mad_u32 v1, v1, s2, s3
	s_lshl_b32 s61, s13, 2
	v_mad_u32 v2, v2, s2, s3
	s_lshl_b32 s2, s60, 8
	s_mul_i32 s73, s48, s50
	s_ashr_i32 s3, s2, 31
	s_lshr_b64 s[6:7], s[2:3], 7
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v1, v1, v3
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_clause 0x1
	s_load_b64 s[2:3], s[0:1], 0xc0 nv
	s_load_b64 s[20:21], s[0:1], 0xe8 nv
	v_dual_lshlrev_b32 v1, 4, v1 :: v_dual_bitop2_b32 v2, v2, v3 bitop3:0x54
	v_lshlrev_b32_e32 v2, 4, v2
	s_wait_kmcnt 0x0
	s_clause 0x7
	buffer_load_b128 v[154:157], v1, s[4:7], null offen
	buffer_load_b128 v[158:161], v1, s[4:7], null offen offset:32
	buffer_load_b128 v[162:165], v1, s[4:7], null offen offset:64
	buffer_load_b128 v[166:169], v1, s[4:7], null offen offset:96
	buffer_load_b128 v[170:173], v1, s[4:7], null offen offset:128
	buffer_load_b128 v[174:177], v1, s[4:7], null offen offset:160
	buffer_load_b128 v[178:181], v1, s[4:7], null offen offset:192
	buffer_load_b128 v[182:185], v1, s[4:7], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[194:197], v1, s[8:11], null offen
	buffer_load_b128 v[198:201], v1, s[8:11], null offen offset:32
	buffer_load_b128 v[202:205], v1, s[8:11], null offen offset:64
	buffer_load_b128 v[206:209], v1, s[8:11], null offen offset:96
	buffer_load_b128 v[210:213], v1, s[8:11], null offen offset:128
	buffer_load_b128 v[214:217], v1, s[8:11], null offen offset:160
	buffer_load_b128 v[226:229], v1, s[8:11], null offen offset:192
	buffer_load_b128 v[230:233], v1, s[8:11], null offen offset:224
	s_clause 0x8
	buffer_load_b128 v[234:237], v2, s[4:7], null offen
	buffer_load_b128 v[238:241], v2, s[4:7], null offen offset:32
	buffer_load_b128 v[242:245], v2, s[4:7], null offen offset:64
	buffer_load_b128 v[246:249], v2, s[4:7], null offen offset:96
	s_set_vgpr_msb 64
	buffer_load_b128 v[2:5] /*v[258:261]*/, v2, s[4:7], null offen offset:128
	buffer_load_b128 v[6:9] /*v[262:265]*/, v2, s[4:7], null offen offset:160
	buffer_load_b128 v[10:13] /*v[266:269]*/, v2, s[4:7], null offen offset:192
	buffer_load_b128 v[14:17] /*v[270:273]*/, v2, s[4:7], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[18:21] /*v[274:277]*/, v2, s[8:11], null offen
	buffer_load_b128 v[22:25] /*v[278:281]*/, v2, s[8:11], null offen offset:32
	buffer_load_b128 v[26:29] /*v[282:285]*/, v2, s[8:11], null offen offset:64
	buffer_load_b128 v[30:33] /*v[286:289]*/, v2, s[8:11], null offen offset:96
	buffer_load_b128 v[34:37] /*v[290:293]*/, v2, s[8:11], null offen offset:128
	buffer_load_b128 v[38:41] /*v[294:297]*/, v2, s[8:11], null offen offset:160
	buffer_load_b128 v[42:45] /*v[298:301]*/, v2, s[8:11], null offen offset:192
	buffer_load_b128 v[46:49] /*v[302:305]*/, v2, s[8:11], null offen offset:224
	s_wait_xcnt 0x8
	s_mul_i32 s4, s61, s15
	s_set_vgpr_msb 0x4000
	v_dual_lshlrev_b32 v1, 3, v3 :: v_dual_lshlrev_b32 v2, 1, v0
	s_mul_i32 s4, s4, s49
	s_mov_b32 s6, -1
	s_ashr_i32 s5, s4, 31
	s_set_vgpr_msb 0x80
	v_and_or_b32 v14 /*v526*/, v0, 7, v1
	v_and_b32_e32 v24 /*v536*/, 16, v2
	s_set_vgpr_msb 0x8000
	v_and_b32_e32 v2, 16, v0
	s_lshr_b64 s[30:31], s[4:5], 7
	s_lshl_b32 s5, s4, 25
	s_cmp_eq_u32 s66, s73
	s_set_vgpr_msb 0xa8
	v_mad_u32_u24 v19 /*v531*/, 0x110, v14 /*v526*/, v24 /*v536*/
	s_cselect_b32 s4, s65, 0
	s_set_vgpr_msb 0xa888
	v_mad_u32_u24 v21 /*v533*/, 0x110, v13 /*v525*/, v2
	s_mul_i32 s56, s4, s17
	s_ashr_i32 s35, s15, 31
	s_cmp_gt_i32 s56, 0
	s_mov_b32 s4, s41
	s_set_vgpr_msb 0x8800
	s_cbranch_scc1 .LBB0_2
	s_mov_b32 s6, 0
.LBB0_2:
	s_clause 0x1
	s_load_b64 s[52:53], s[0:1], 0x0 nv
	s_load_b64 s[54:55], s[0:1], 0x90 nv
	s_set_vgpr_msb 64
	v_or_b32_e32 v132 /*v388*/, 16, v0
	s_set_vgpr_msb 0x4080
	v_or_b32_e32 v23 /*v535*/, s22, v1
	v_or_b32_e32 v17 /*v529*/, s18, v1
	s_or_b64 s[28:29], s[2:3], s[4:5]
	s_or_b64 s[36:37], s[20:21], s[4:5]
	s_lshl_b32 s71, s15, 7
	s_and_b32 s2, s6, exec_lo
	s_set_vgpr_msb 0x8088
	v_or_b32_e32 v11 /*v523*/, 3, v23 /*v535*/
	v_or_b32_e32 v12 /*v524*/, 2, v23 /*v535*/
	v_or_b32_e32 v9 /*v521*/, 5, v23 /*v535*/
	v_or_b32_e32 v10 /*v522*/, 4, v23 /*v535*/
	v_or_b32_e32 v7 /*v519*/, 7, v23 /*v535*/
	v_or_b32_e32 v8 /*v520*/, 6, v23 /*v535*/
	v_mul_i32_i24_e32 v15 /*v527*/, 0xffffff40, v13 /*v525*/
	v_or_b32_e32 v5 /*v517*/, 3, v17 /*v529*/
	v_or_b32_e32 v6 /*v518*/, 2, v17 /*v529*/
	v_or_b32_e32 v3 /*v515*/, 5, v17 /*v529*/
	v_or_b32_e32 v4 /*v516*/, 4, v17 /*v529*/
	s_set_vgpr_msb 0x8808
	v_or_b32_e32 v1, 7, v17 /*v529*/
	s_set_vgpr_msb 0x888
	v_or_b32_e32 v2 /*v514*/, 6, v17 /*v529*/
	s_set_vgpr_msb 0x8884
	v_mad_u32_u24 v27 /*v539*/, 0x50, v132 /*v388*/, v2
	s_set_vgpr_msb 0x8488
	v_mul_u32_u24_e32 v26 /*v538*/, 0x50, v14 /*v526*/
	v_or_b32_e32 v25 /*v537*/, 32, v24 /*v536*/
	s_cselect_b32 s2, 1, 0
	s_mul_i32 s14, s48, s17
	s_mul_i32 s15, s15, s33
	s_cmp_lg_u32 s2, 1
	s_mul_i32 s62, s13, s33
	s_set_vgpr_msb 0x8800
	s_cbranch_scc1 .LBB0_5
	s_mov_b32 s3, 0x10a00
	s_set_vgpr_msb 8
	v_or_b32_e32 v2, 0x10000, v26 /*v538*/
	v_mad_u32_u24 v3, 0x50, v14 /*v526*/, s3
	s_set_vgpr_msb 0x84a
	v_mov_b32_e32 v74 /*v330*/, 0
	s_ashr_i32 s57, s56, 31
	s_cmp_lg_u32 s71, 0x80000000
	s_mov_b32 s4, s12
	s_mov_b32 s5, s12
	s_cselect_b32 s25, s71, 0x80
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, s[4:5]
	v_add3_u32 v133 /*v389*/, v21 /*v533*/, v15 /*v527*/, 0x10000
	v_or_b32_e32 v134 /*v390*/, 0x10000, v27 /*v539*/
	s_set_vgpr_msb 0x4a48
	v_dual_add_nc_u32 v135 /*v391*/, v2, v24 /*v536*/ :: v_dual_add_nc_u32 v136 /*v392*/, v3, v24 /*v536*/
	v_dual_add_nc_u32 v137 /*v393*/, v2, v25 /*v537*/ :: v_dual_add_nc_u32 v138 /*v394*/, v3, v25 /*v537*/
	s_set_vgpr_msb 0x4841
	v_dual_mov_b32 v61 /*v317*/, v74 /*v330*/ :: v_dual_mov_b32 v62 /*v318*/, v74 /*v330*/
	v_dual_mov_b32 v75 /*v331*/, v74 /*v330*/ :: v_dual_mov_b32 v76 /*v332*/, v74 /*v330*/
	v_dual_mov_b32 v77 /*v333*/, v74 /*v330*/ :: v_dual_mov_b32 v78 /*v334*/, v74 /*v330*/
	v_dual_mov_b32 v79 /*v335*/, v74 /*v330*/ :: v_dual_mov_b32 v80 /*v336*/, v74 /*v330*/
	v_dual_mov_b32 v81 /*v337*/, v74 /*v330*/ :: v_dual_mov_b32 v58 /*v314*/, v74 /*v330*/
	v_dual_mov_b32 v59 /*v315*/, v74 /*v330*/ :: v_dual_mov_b32 v60 /*v316*/, v74 /*v330*/
	v_dual_mov_b32 v63 /*v319*/, v74 /*v330*/ :: v_dual_mov_b32 v64 /*v320*/, v74 /*v330*/
	v_dual_mov_b32 v65 /*v321*/, v74 /*v330*/ :: v_dual_mov_b32 v50 /*v306*/, v74 /*v330*/
	v_dual_mov_b32 v51 /*v307*/, v74 /*v330*/ :: v_dual_mov_b32 v52 /*v308*/, v74 /*v330*/
	v_dual_mov_b32 v53 /*v309*/, v74 /*v330*/ :: v_dual_mov_b32 v54 /*v310*/, v74 /*v330*/
	v_dual_mov_b32 v55 /*v311*/, v74 /*v330*/ :: v_dual_mov_b32 v56 /*v312*/, v74 /*v330*/
	v_dual_mov_b32 v57 /*v313*/, v74 /*v330*/ :: v_dual_mov_b32 v0 /*v256*/, v74 /*v330*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v250, v74 /*v330*/ :: v_dual_mov_b32 v251, v74 /*v330*/
	v_dual_mov_b32 v252, v74 /*v330*/ :: v_dual_mov_b32 v253, v74 /*v330*/
	v_dual_mov_b32 v254, v74 /*v330*/ :: v_dual_mov_b32 v255, v74 /*v330*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v1 /*v257*/, v74 /*v330*/ :: v_dual_mov_b32 v122 /*v378*/, v74 /*v330*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v186, v74 /*v330*/ :: v_dual_mov_b32 v187, v74 /*v330*/
	v_dual_mov_b32 v188, v74 /*v330*/ :: v_dual_mov_b32 v189, v74 /*v330*/
	v_dual_mov_b32 v190, v74 /*v330*/ :: v_dual_mov_b32 v191, v74 /*v330*/
	v_dual_mov_b32 v192, v74 /*v330*/ :: v_dual_mov_b32 v193, v74 /*v330*/
	v_dual_mov_b32 v146, v74 /*v330*/ :: v_dual_mov_b32 v147, v74 /*v330*/
	v_dual_mov_b32 v148, v74 /*v330*/ :: v_dual_mov_b32 v149, v74 /*v330*/
	v_dual_mov_b32 v150, v74 /*v330*/ :: v_dual_mov_b32 v151, v74 /*v330*/
	v_dual_mov_b32 v152, v74 /*v330*/ :: v_dual_mov_b32 v153, v74 /*v330*/
	v_dual_mov_b32 v138, v74 /*v330*/ :: v_dual_mov_b32 v139, v74 /*v330*/
	v_dual_mov_b32 v140, v74 /*v330*/ :: v_dual_mov_b32 v141, v74 /*v330*/
	v_dual_mov_b32 v142, v74 /*v330*/ :: v_dual_mov_b32 v143, v74 /*v330*/
	v_dual_mov_b32 v144, v74 /*v330*/ :: v_dual_mov_b32 v145, v74 /*v330*/
	v_dual_mov_b32 v122, v74 /*v330*/ :: v_dual_mov_b32 v123, v74 /*v330*/
	v_dual_mov_b32 v124, v74 /*v330*/ :: v_dual_mov_b32 v125, v74 /*v330*/
	v_dual_mov_b32 v126, v74 /*v330*/ :: v_dual_mov_b32 v127, v74 /*v330*/
	v_dual_mov_b32 v128, v74 /*v330*/ :: v_dual_mov_b32 v129, v74 /*v330*/
	v_dual_mov_b32 v82, v74 /*v330*/ :: v_dual_mov_b32 v83, v74 /*v330*/
	v_dual_mov_b32 v84, v74 /*v330*/ :: v_dual_mov_b32 v85, v74 /*v330*/
	v_dual_mov_b32 v86, v74 /*v330*/ :: v_dual_mov_b32 v87, v74 /*v330*/
	v_dual_mov_b32 v88, v74 /*v330*/ :: v_dual_mov_b32 v89, v74 /*v330*/
	v_dual_mov_b32 v58, v74 /*v330*/ :: v_dual_mov_b32 v59, v74 /*v330*/
	v_dual_mov_b32 v60, v74 /*v330*/ :: v_dual_mov_b32 v61, v74 /*v330*/
	v_dual_mov_b32 v62, v74 /*v330*/ :: v_dual_mov_b32 v63, v74 /*v330*/
	v_dual_mov_b32 v64, v74 /*v330*/ :: v_dual_mov_b32 v65, v74 /*v330*/
	v_dual_mov_b32 v50, v74 /*v330*/ :: v_dual_mov_b32 v51, v74 /*v330*/
	v_dual_mov_b32 v52, v74 /*v330*/ :: v_dual_mov_b32 v53, v74 /*v330*/
	v_dual_mov_b32 v54, v74 /*v330*/ :: v_dual_mov_b32 v55, v74 /*v330*/
	v_dual_mov_b32 v56, v74 /*v330*/ :: v_dual_mov_b32 v57, v74 /*v330*/
	v_dual_mov_b32 v34, v74 /*v330*/ :: v_dual_mov_b32 v35, v74 /*v330*/
	v_dual_mov_b32 v36, v74 /*v330*/ :: v_dual_mov_b32 v37, v74 /*v330*/
	v_dual_mov_b32 v38, v74 /*v330*/ :: v_dual_mov_b32 v39, v74 /*v330*/
	v_dual_mov_b32 v40, v74 /*v330*/ :: v_dual_mov_b32 v41, v74 /*v330*/
	v_dual_mov_b32 v26, v74 /*v330*/ :: v_dual_mov_b32 v27, v74 /*v330*/
	v_dual_mov_b32 v28, v74 /*v330*/ :: v_dual_mov_b32 v29, v74 /*v330*/
	v_dual_mov_b32 v30, v74 /*v330*/ :: v_dual_mov_b32 v31, v74 /*v330*/
	v_dual_mov_b32 v32, v74 /*v330*/ :: v_dual_mov_b32 v33, v74 /*v330*/
	v_dual_mov_b32 v18, v74 /*v330*/ :: v_dual_mov_b32 v19, v74 /*v330*/
	v_dual_mov_b32 v20, v74 /*v330*/ :: v_dual_mov_b32 v21, v74 /*v330*/
	v_dual_mov_b32 v22, v74 /*v330*/ :: v_dual_mov_b32 v23, v74 /*v330*/
	v_dual_mov_b32 v24, v74 /*v330*/ :: v_dual_mov_b32 v25, v74 /*v330*/
	v_dual_mov_b32 v10, v74 /*v330*/ :: v_dual_mov_b32 v11, v74 /*v330*/
	v_dual_mov_b32 v12, v74 /*v330*/ :: v_dual_mov_b32 v13, v74 /*v330*/
	v_dual_mov_b32 v14, v74 /*v330*/ :: v_dual_mov_b32 v15, v74 /*v330*/
	v_dual_mov_b32 v16, v74 /*v330*/ :: v_dual_mov_b32 v17, v74 /*v330*/
	v_dual_mov_b32 v2, v74 /*v330*/ :: v_dual_mov_b32 v3, v74 /*v330*/
	v_dual_mov_b32 v4, v74 /*v330*/ :: v_dual_mov_b32 v5, v74 /*v330*/
	v_dual_mov_b32 v6, v74 /*v330*/ :: v_dual_mov_b32 v7, v74 /*v330*/
	v_dual_mov_b32 v8, v74 /*v330*/ :: v_dual_mov_b32 v9, v74 /*v330*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v123 /*v379*/, v74 /*v330*/ :: v_dual_mov_b32 v124 /*v380*/, v74 /*v330*/
	v_dual_mov_b32 v125 /*v381*/, v74 /*v330*/ :: v_dual_mov_b32 v126 /*v382*/, v74 /*v330*/
	v_dual_mov_b32 v127 /*v383*/, v74 /*v330*/ :: v_dual_mov_b32 v128 /*v384*/, v74 /*v330*/
	v_dual_mov_b32 v129 /*v385*/, v74 /*v330*/ :: v_dual_mov_b32 v114 /*v370*/, v74 /*v330*/
	v_dual_mov_b32 v115 /*v371*/, v74 /*v330*/ :: v_dual_mov_b32 v116 /*v372*/, v74 /*v330*/
	v_dual_mov_b32 v117 /*v373*/, v74 /*v330*/ :: v_dual_mov_b32 v118 /*v374*/, v74 /*v330*/
	v_dual_mov_b32 v119 /*v375*/, v74 /*v330*/ :: v_dual_mov_b32 v120 /*v376*/, v74 /*v330*/
	v_dual_mov_b32 v121 /*v377*/, v74 /*v330*/ :: v_dual_mov_b32 v106 /*v362*/, v74 /*v330*/
	v_dual_mov_b32 v107 /*v363*/, v74 /*v330*/ :: v_dual_mov_b32 v108 /*v364*/, v74 /*v330*/
	v_dual_mov_b32 v109 /*v365*/, v74 /*v330*/ :: v_dual_mov_b32 v110 /*v366*/, v74 /*v330*/
	v_dual_mov_b32 v111 /*v367*/, v74 /*v330*/ :: v_dual_mov_b32 v112 /*v368*/, v74 /*v330*/
	v_dual_mov_b32 v113 /*v369*/, v74 /*v330*/ :: v_dual_mov_b32 v98 /*v354*/, v74 /*v330*/
	v_dual_mov_b32 v99 /*v355*/, v74 /*v330*/ :: v_dual_mov_b32 v100 /*v356*/, v74 /*v330*/
	v_dual_mov_b32 v101 /*v357*/, v74 /*v330*/ :: v_dual_mov_b32 v102 /*v358*/, v74 /*v330*/
	v_dual_mov_b32 v103 /*v359*/, v74 /*v330*/ :: v_dual_mov_b32 v104 /*v360*/, v74 /*v330*/
	v_dual_mov_b32 v105 /*v361*/, v74 /*v330*/ :: v_dual_mov_b32 v90 /*v346*/, v74 /*v330*/
	v_dual_mov_b32 v91 /*v347*/, v74 /*v330*/ :: v_dual_mov_b32 v92 /*v348*/, v74 /*v330*/
	v_dual_mov_b32 v93 /*v349*/, v74 /*v330*/ :: v_dual_mov_b32 v94 /*v350*/, v74 /*v330*/
	v_dual_mov_b32 v95 /*v351*/, v74 /*v330*/ :: v_dual_mov_b32 v96 /*v352*/, v74 /*v330*/
	v_dual_mov_b32 v97 /*v353*/, v74 /*v330*/ :: v_dual_mov_b32 v82 /*v338*/, v74 /*v330*/
	v_dual_mov_b32 v83 /*v339*/, v74 /*v330*/ :: v_dual_mov_b32 v84 /*v340*/, v74 /*v330*/
	v_dual_mov_b32 v85 /*v341*/, v74 /*v330*/ :: v_dual_mov_b32 v86 /*v342*/, v74 /*v330*/
	v_dual_mov_b32 v87 /*v343*/, v74 /*v330*/ :: v_dual_mov_b32 v88 /*v344*/, v74 /*v330*/
	v_dual_mov_b32 v89 /*v345*/, v74 /*v330*/ :: v_dual_mov_b32 v66 /*v322*/, v74 /*v330*/
	v_dual_mov_b32 v67 /*v323*/, v74 /*v330*/ :: v_dual_mov_b32 v68 /*v324*/, v74 /*v330*/
	v_dual_mov_b32 v69 /*v325*/, v74 /*v330*/ :: v_dual_mov_b32 v70 /*v326*/, v74 /*v330*/
	v_dual_mov_b32 v71 /*v327*/, v74 /*v330*/ :: v_dual_mov_b32 v72 /*v328*/, v74 /*v330*/
	v_mov_b32_e32 v73 /*v329*/, v74 /*v330*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v218, v74 /*v330*/ :: v_dual_mov_b32 v219, v74 /*v330*/
	v_dual_mov_b32 v220, v74 /*v330*/ :: v_dual_mov_b32 v221, v74 /*v330*/
	v_dual_mov_b32 v222, v74 /*v330*/ :: v_dual_mov_b32 v223, v74 /*v330*/
	v_dual_mov_b32 v224, v74 /*v330*/ :: v_dual_mov_b32 v225, v74 /*v330*/
	v_dual_mov_b32 v130, v74 /*v330*/ :: v_dual_mov_b32 v131, v74 /*v330*/
	v_dual_mov_b32 v132, v74 /*v330*/ :: v_dual_mov_b32 v133, v74 /*v330*/
	v_dual_mov_b32 v134, v74 /*v330*/ :: v_dual_mov_b32 v135, v74 /*v330*/
	v_dual_mov_b32 v136, v74 /*v330*/ :: v_dual_mov_b32 v137, v74 /*v330*/
	v_dual_mov_b32 v114, v74 /*v330*/ :: v_dual_mov_b32 v115, v74 /*v330*/
	v_dual_mov_b32 v116, v74 /*v330*/ :: v_dual_mov_b32 v117, v74 /*v330*/
	v_dual_mov_b32 v118, v74 /*v330*/ :: v_dual_mov_b32 v119, v74 /*v330*/
	v_dual_mov_b32 v120, v74 /*v330*/ :: v_dual_mov_b32 v121, v74 /*v330*/
	v_dual_mov_b32 v106, v74 /*v330*/ :: v_dual_mov_b32 v107, v74 /*v330*/
	v_dual_mov_b32 v108, v74 /*v330*/ :: v_dual_mov_b32 v109, v74 /*v330*/
	v_dual_mov_b32 v110, v74 /*v330*/ :: v_dual_mov_b32 v111, v74 /*v330*/
	v_dual_mov_b32 v112, v74 /*v330*/ :: v_dual_mov_b32 v113, v74 /*v330*/
	v_dual_mov_b32 v98, v74 /*v330*/ :: v_dual_mov_b32 v99, v74 /*v330*/
	v_dual_mov_b32 v100, v74 /*v330*/ :: v_dual_mov_b32 v101, v74 /*v330*/
	v_dual_mov_b32 v102, v74 /*v330*/ :: v_dual_mov_b32 v103, v74 /*v330*/
	v_dual_mov_b32 v104, v74 /*v330*/ :: v_dual_mov_b32 v105, v74 /*v330*/
	v_dual_mov_b32 v90, v74 /*v330*/ :: v_dual_mov_b32 v91, v74 /*v330*/
	v_dual_mov_b32 v92, v74 /*v330*/ :: v_dual_mov_b32 v93, v74 /*v330*/
	v_dual_mov_b32 v94, v74 /*v330*/ :: v_dual_mov_b32 v95, v74 /*v330*/
	v_dual_mov_b32 v96, v74 /*v330*/ :: v_dual_mov_b32 v97, v74 /*v330*/
	v_dual_mov_b32 v74, v74 /*v330*/ :: v_dual_mov_b32 v75, v74 /*v330*/
	v_dual_mov_b32 v76, v74 /*v330*/ :: v_dual_mov_b32 v77, v74 /*v330*/
	v_dual_mov_b32 v78, v74 /*v330*/ :: v_dual_mov_b32 v79, v74 /*v330*/
	v_dual_mov_b32 v80, v74 /*v330*/ :: v_dual_mov_b32 v81, v74 /*v330*/
	v_dual_mov_b32 v66, v74 /*v330*/ :: v_dual_mov_b32 v67, v74 /*v330*/
	v_dual_mov_b32 v68, v74 /*v330*/ :: v_dual_mov_b32 v69, v74 /*v330*/
	v_dual_mov_b32 v70, v74 /*v330*/ :: v_dual_mov_b32 v71, v74 /*v330*/
	v_dual_mov_b32 v72, v74 /*v330*/ :: v_dual_mov_b32 v73, v74 /*v330*/
	v_dual_mov_b32 v42, v74 /*v330*/ :: v_dual_mov_b32 v43, v74 /*v330*/
	v_dual_mov_b32 v44, v74 /*v330*/ :: v_dual_mov_b32 v45, v74 /*v330*/
	v_dual_mov_b32 v46, v74 /*v330*/ :: v_dual_mov_b32 v47, v74 /*v330*/
	v_dual_mov_b32 v48, v74 /*v330*/ :: v_dual_mov_b32 v49, v74 /*v330*/
	s_ashr_i32 s2, s25, 31
	s_mov_b32 s24, 32
	s_mov_b64 s[58:59], 0
	s_mov_b32 s40, 1
	s_mov_b32 s38, s30
	s_mov_b32 s39, s31
	s_mov_b32 s21, 0xffff0000
	s_mov_b32 s20, 0x7510000
	s_movk_i32 s45, 0x2200
	s_mov_b32 s18, 0x3fb8aa3b
	s_and_b32 s26, s2, 0xffff
	s_mov_b32 s2, s41
	s_mov_b32 s3, s41
	s_set_vgpr_msb 0x100
.LBB0_4:
	s_add_co_i32 s4, s2, 1
	s_mov_b32 s27, s41
	s_cmp_ge_i32 s4, s17
	s_mov_b32 s44, s40
	s_cselect_b32 s5, -1, 0
	s_and_b32 s6, s5, exec_lo
	s_cselect_b32 s74, 0, s4
	s_cmp_lg_u32 s5, 0
	s_add_co_ci_u32 s75, s3, 0
	s_add_co_i32 s3, s3, s64
	s_add_co_i32 s2, s2, s14
	s_lshl_b32 s6, s3, 5
	s_add_co_i32 s5, s2, s15
	s_add_co_i32 s4, s6, s62
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v139 /*v395*/, s6, v13 /*v525*/
	s_mul_i32 s7, s5, s13
	s_ashr_i32 s5, s4, 31
	s_set_vgpr_msb 0x4884
	v_or_b32_e32 v1 /*v513*/, s6, v132 /*v388*/
	s_ashr_i32 s3, s2, 31
	s_mul_u64 s[4:5], s[4:5], s[34:35]
	s_set_vgpr_msb 0x8444
	v_add_lshl_u32 v140 /*v396*/, s7, v139 /*v395*/, 2
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	s_sub_co_i32 s4, s13, s6
	s_set_vgpr_msb 0x4448
	v_add_lshl_u32 v141 /*v397*/, s7, v1 /*v513*/, 2
	s_lshl_b64 s[2:3], s[2:3], 8
	s_max_i32 s4, s4, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[42:43], s[54:55], s[2:3]
	s_lshl_b32 s5, s4, 16
	s_lshr_b32 s4, s4, 16
	s_add_nc_u64 s[46:47], s[52:53], s[2:3]
	s_set_vgpr_msb 0x4881
	s_clause 0x1
	buffer_load_b32 v0 /*v512*/, v140 /*v396*/, s[28:31], null offen
	buffer_load_b32 v16 /*v528*/, v141 /*v397*/, s[28:31], null offen
	s_clause 0x1
	buffer_load_b32 v18 /*v530*/, v140 /*v396*/, s[36:39], null offen
	buffer_load_b32 v20 /*v532*/, v141 /*v397*/, s[36:39], null offen
	s_bitset1_b32 s43, 31
	s_or_b32 s22, s5, 0x7fff
	s_or_b32 s23, s4, 0x800000
	s_bitset1_b32 s47, 31
	tensor_load_to_lds s[40:43], s[20:27]
	tensor_load_to_lds s[44:47], s[20:27]
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x8142
	ds_load_b128 v[140:143] /*v[396:399]*/, v21 /*v533*/ offset:8704
	ds_load_b128 v[144:147] /*v[400:403]*/, v21 /*v533*/ offset:8736
	ds_load_b128 v[148:151] /*v[404:407]*/, v21 /*v533*/ offset:8768
	ds_load_b128 v[152:155] /*v[408:411]*/, v21 /*v533*/ offset:8800
	ds_load_b128 v[156:159] /*v[412:415]*/, v21 /*v533*/ offset:8832
	ds_load_b128 v[160:163] /*v[416:419]*/, v21 /*v533*/ offset:8864
	ds_load_b128 v[164:167] /*v[420:423]*/, v21 /*v533*/ offset:8896
	ds_load_b128 v[168:171] /*v[424:427]*/, v21 /*v533*/ offset:8928
	ds_load_b128 v[172:175] /*v[428:431]*/, v21 /*v533*/
	ds_load_b128 v[176:179] /*v[432:435]*/, v21 /*v533*/ offset:32
	ds_load_b128 v[180:183] /*v[436:439]*/, v21 /*v533*/ offset:64
	ds_load_b128 v[184:187] /*v[440:443]*/, v21 /*v533*/ offset:96
	ds_load_b128 v[188:191] /*v[444:447]*/, v21 /*v533*/ offset:128
	ds_load_b128 v[192:195] /*v[448:451]*/, v21 /*v533*/ offset:160
	ds_load_b128 v[196:199] /*v[452:455]*/, v21 /*v533*/ offset:192
	ds_load_b128 v[200:203] /*v[456:459]*/, v21 /*v533*/ offset:224
	ds_load_b128 v[204:207] /*v[460:463]*/, v21 /*v533*/ offset:13056
	ds_load_b128 v[208:211] /*v[464:467]*/, v21 /*v533*/ offset:13088
	ds_load_b128 v[212:215] /*v[468:471]*/, v21 /*v533*/ offset:13120
	ds_load_b128 v[216:219] /*v[472:475]*/, v21 /*v533*/ offset:13152
	ds_load_b128 v[220:223] /*v[476:479]*/, v21 /*v533*/ offset:13184
	ds_load_b128 v[224:227] /*v[480:483]*/, v21 /*v533*/ offset:13216
	ds_load_b128 v[228:231] /*v[484:487]*/, v21 /*v533*/ offset:13248
	ds_load_b128 v[232:235] /*v[488:491]*/, v21 /*v533*/ offset:13280
	ds_load_b128 v[236:239] /*v[492:495]*/, v21 /*v533*/ offset:4352
	ds_load_b128 v[240:243] /*v[496:499]*/, v21 /*v533*/ offset:4384
	ds_load_b128 v[244:247] /*v[500:503]*/, v21 /*v533*/ offset:4416
	ds_load_b128 v[248:251] /*v[504:507]*/, v21 /*v533*/ offset:4448
	s_set_vgpr_msb 0x4282
	ds_load_b128 v[28:31] /*v[540:543]*/, v21 /*v533*/ offset:4480
	ds_load_b128 v[32:35] /*v[544:547]*/, v21 /*v533*/ offset:4512
	ds_load_b128 v[36:39] /*v[548:551]*/, v21 /*v533*/ offset:4544
	ds_load_b128 v[40:43] /*v[552:555]*/, v21 /*v533*/ offset:4576
	s_set_vgpr_msb 0x8284
	s_wait_loadcnt_dscnt 0x221e
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[154:161], v[140:147] /*v[396:403]*/, 0
	s_set_vgpr_msb 0x8446
	v_add_nc_u32_e32 v139 /*v395*/, s19, v139 /*v395*/
	v_cmp_ge_i32_e32 vcc_lo, v23 /*v535*/, v139 /*v395*/
	v_cmp_gt_i32_e64 s2, v23 /*v535*/, v139 /*v395*/
	s_set_vgpr_msb 0x46a4
	s_wait_loadcnt_dscnt 0x201c
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[162:169], v[148:155] /*v[404:411]*/, v[44:51] /*v[556:563]*/
	s_set_vgpr_msb 0xa406
	v_cmp_gt_i32_e64 s3, v11 /*v523*/, v139 /*v395*/
	v_cmp_gt_i32_e64 s4, v12 /*v524*/, v139 /*v395*/
	s_and_b32 s22, s72, vcc_lo
	s_and_b32 s2, s72, s2
	v_cmp_gt_i32_e64 s5, v9 /*v521*/, v139 /*v395*/
	v_cmp_gt_i32_e64 s6, v10 /*v522*/, v139 /*v395*/
	s_and_b32 s3, s72, s3
	s_set_vgpr_msb 0x6a4
	s_wait_loadcnt_dscnt 0x1e1a
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[170:177], v[156:163] /*v[412:419]*/, v[44:51] /*v[556:563]*/
	s_set_vgpr_msb 0xa406
	v_cmp_gt_i32_e64 s8, v8 /*v520*/, v139 /*v395*/
	v_cmp_gt_i32_e64 s7, v7 /*v519*/, v139 /*v395*/
	v_cmp_ge_i32_e64 s9, v17 /*v529*/, v139 /*v395*/
	v_cmp_gt_i32_e64 s10, v17 /*v529*/, v139 /*v395*/
	v_cmp_gt_i32_e64 s11, v5 /*v517*/, v139 /*v395*/
	v_cmp_gt_i32_e32 vcc_lo, v6 /*v518*/, v139 /*v395*/
	s_set_vgpr_msb 0x6a4
	s_wait_loadcnt_dscnt 0x1c18
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[178:185], v[164:171] /*v[420:427]*/, v[44:51] /*v[556:563]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa449
	v_pk_mul_f32 v[252:253] /*v[508:509]*/, v[130:131] /*v[386:387]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x4984
	s_wait_loadcnt_dscnt 0x1a16
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[194:201], v[172:179] /*v[428:435]*/, 0
	s_set_vgpr_msb 0x8449
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[130:131] /*v[386:387]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4989
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[130:131] /*v[386:387]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[130:131] /*v[386:387]*/, v[50:51] /*v[562:563]*/
	s_set_vgpr_msb 0x8941
	v_cndmask_b32_e64 v253 /*v509*/, v253 /*v509*/, 0xff61b1e6, s22
	v_cndmask_b32_e64 v252 /*v508*/, v252 /*v508*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s4
	v_cndmask_b32_e64 v255 /*v511*/, v255 /*v511*/, 0xff61b1e6, s3
	s_set_vgpr_msb 0x41a4
	s_wait_loadcnt_dscnt 0x1814
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[202:209], v[180:187] /*v[436:443]*/, v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa449
	v_cndmask_b32_e64 v254 /*v510*/, v254 /*v510*/, 0xff61b1e6, s2
	s_wait_loadcnt 0x3
	v_pk_add_f32 v[252:253] /*v[508:509]*/, v[252:253] /*v[508:509]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s72, s6
	s_and_b32 s3, s72, s5
	s_set_vgpr_msb 0x4982
	v_cndmask_b32_e64 v44 /*v556*/, v44 /*v556*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x8249
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[252:253] /*v[508:509]*/, v[252:253] /*v[508:509]*/, s[18:19] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x49a4
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[210:217], v[188:195] /*v[444:451]*/, v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa482
	v_cndmask_b32_e64 v45 /*v557*/, v45 /*v557*/, 0xff61b1e6, s3
	s_and_b32 s2, s72, s8
	s_set_vgpr_msb 0x8241
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, s[18:19] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v60 /*v572*/, v252 /*v508*/
	v_exp_f32_e32 v61 /*v573*/, v253 /*v509*/
	v_nop
	s_set_vgpr_msb 0x814a
	v_pk_add_f32 v[252:253] /*v[508:509]*/, v[44:45] /*v[556:557]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a82
	v_cndmask_b32_e64 v44 /*v556*/, v46 /*v558*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x82a4
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[226:233], v[196:203] /*v[452:459]*/, v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa481
	v_exp_f32_e32 v48 /*v560*/, v254 /*v510*/
	v_exp_f32_e32 v49 /*v561*/, v255 /*v511*/
	s_set_vgpr_msb 0x8141
	v_pk_mul_f32 v[252:253] /*v[508:509]*/, v[252:253] /*v[508:509]*/, s[18:19] op_sel_hi:[1,0]
	s_and_b32 s2, s72, s7
	s_set_vgpr_msb 0x4182
	v_cndmask_b32_e64 v45 /*v557*/, v47 /*v559*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s9
	s_set_vgpr_msb 0x824a
	s_wait_loadcnt 0x1
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[52:53] /*v[564:565]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[54:55] /*v[566:567]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v252 /*v508*/, v252 /*v508*/
	v_exp_f32_e32 v253 /*v509*/, v253 /*v509*/
	s_set_vgpr_msb 0x418a
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a49
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, v[60:61] /*v[572:573]*/
	s_set_vgpr_msb 0x498a
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, v[48:49] /*v[560:561]*/
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[56:57] /*v[568:569]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[56:57] /*v[568:569]*/, v[58:59] /*v[570:571]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, s[18:19] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8a85
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[130:131] /*v[386:387]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x8589
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[130:131] /*v[386:387]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[252:253] /*v[508:509]*/, v[50:51] /*v[562:563]*/
	s_set_vgpr_msb 0x8945
	v_cvt_pk_bf16_f32 v254 /*v510*/, v252 /*v508*/, v253 /*v509*/
	s_set_vgpr_msb 0x458a
	v_exp_f32_e32 v58 /*v570*/, v44 /*v556*/
	v_exp_f32_e32 v59 /*v571*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v52 /*v564*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v53 /*v565*/, v46 /*v558*/, v47 /*v559*/
	s_set_vgpr_msb 0x8a89
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[130:131] /*v[386:387]*/, v[50:51] /*v[562:563]*/
	s_set_vgpr_msb 0x894a
	v_cvt_pk_bf16_f32 v253 /*v509*/, v48 /*v560*/, v49 /*v561*/
	s_set_vgpr_msb 0x4a84
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[234:241], v[140:147] /*v[396:403]*/, 0
	s_set_vgpr_msb 0x844a
	v_cvt_pk_bf16_f32 v252 /*v508*/, v60 /*v572*/, v61 /*v573*/
	s_set_vgpr_msb 0x4a8a
	v_cvt_pk_bf16_f32 v54 /*v566*/, v54 /*v566*/, v55 /*v567*/
	s_set_vgpr_msb 0x8a4a
	v_cvt_pk_bf16_f32 v255 /*v511*/, v58 /*v570*/, v59 /*v571*/
	v_nop
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[56:57] /*v[568:569]*/, v[58:59] /*v[570:571]*/
	s_set_vgpr_msb 0x4a85
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[130:131] /*v[386:387]*/, v[140:141] /*v[396:397]*/
	s_set_vgpr_msb 0x85a4
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[242:249], v[148:155] /*v[404:411]*/, v[44:51] /*v[556:563]*/
	s_set_vgpr_msb 0xa48a
	v_cvt_pk_bf16_f32 v55 /*v567*/, v56 /*v568*/, v57 /*v569*/
	s_set_vgpr_msb 0x8aa5
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[2:9] /*v[258:265]*/, v[156:163] /*v[412:419]*/, v[44:51] /*v[556:563]*/
	v_wmma_f32_16x16x32_bf16 v[44:51] /*v[556:563]*/, v[10:17] /*v[266:273]*/, v[164:171] /*v[420:427]*/, v[44:51] /*v[556:563]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa549
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[130:131] /*v[386:387]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[130:131] /*v[386:387]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4945
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[18:25] /*v[274:281]*/, v[172:179] /*v[428:435]*/, 0
	s_set_vgpr_msb 0x4549
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[130:131] /*v[386:387]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[130:131] /*v[386:387]*/, v[50:51] /*v[562:563]*/
	v_cndmask_b32_e64 v149 /*v405*/, v149 /*v405*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s10
	v_cndmask_b32_e64 v148 /*v404*/, v148 /*v404*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s11
	s_set_vgpr_msb 0x4955
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[26:33] /*v[282:289]*/, v[180:187] /*v[436:443]*/, v[140:147] /*v[396:403]*/
	v_cndmask_b32_e64 v151 /*v407*/, v151 /*v407*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, vcc_lo
	s_set_vgpr_msb 0x5506
	v_cmp_gt_i32_e32 vcc_lo, v3 /*v515*/, v139 /*v395*/
	s_set_vgpr_msb 0x641
	v_cndmask_b32_e64 v150 /*v406*/, v150 /*v406*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4106
	v_cmp_gt_i32_e64 s2, v4 /*v516*/, v139 /*v395*/
	s_set_vgpr_msb 0x649
	v_pk_add_f32 v[148:149] /*v[404:405]*/, v[148:149] /*v[404:405]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s3, s72, vcc_lo
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[150:151] /*v[406:407]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s72, s2
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[148:149] /*v[404:405]*/, s[18:19] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v159 /*v415*/, v153 /*v409*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v158 /*v414*/, v152 /*v408*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4944
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[154:161], v[204:211] /*v[460:467]*/, 0
	v_cmp_gt_i32_e32 vcc_lo, v1, v139 /*v395*/
	s_set_vgpr_msb 0x4406
	v_cmp_gt_i32_e64 s2, v2 /*v514*/, v139 /*v395*/
	s_set_vgpr_msb 0x649
	v_exp_f32_e32 v172 /*v428*/, v164 /*v420*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[158:159] /*v[414:415]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v173 /*v429*/, v165 /*v421*/
	s_and_b32 s3, s72, vcc_lo
	s_and_b32 s2, s72, s2
	s_set_vgpr_msb 0x4955
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[34:41] /*v[290:297]*/, v[188:195] /*v[444:451]*/, v[140:147] /*v[396:403]*/
	v_cndmask_b32_e64 v167 /*v423*/, v167 /*v423*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v166 /*v422*/, v166 /*v422*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[170:171] /*v[426:427]*/, s[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[156:157] /*v[412:413]*/, s[18:19] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x5549
	v_add_nc_u32_e32 v139 /*v395*/, s19, v1 /*v513*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[166:167] /*v[422:423]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4954
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[162:169], v[212:219] /*v[468:475]*/, v[148:155] /*v[404:411]*/
	s_set_vgpr_msb 0x5441
	v_exp_f32_e32 v174 /*v430*/, v164 /*v420*/
	v_exp_f32_e32 v175 /*v431*/, v165 /*v421*/
	v_exp_f32_e32 v168 /*v424*/, v168 /*v424*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[166:167] /*v[422:423]*/, s[18:19] op_sel_hi:[1,0]
	v_exp_f32_e32 v169 /*v425*/, v169 /*v425*/
	s_set_vgpr_msb 0x4106
	v_cmp_ge_i32_e32 vcc_lo, v23 /*v535*/, v139 /*v395*/
	v_cmp_gt_i32_e64 s2, v23 /*v535*/, v139 /*v395*/
	s_set_vgpr_msb 0x655
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[42:49] /*v[298:305]*/, v[196:203] /*v[452:459]*/, v[140:147] /*v[396:403]*/
	v_exp_f32_e32 v170 /*v426*/, v164 /*v420*/
	v_exp_f32_e32 v171 /*v427*/, v165 /*v421*/
	s_and_b32 s3, s72, vcc_lo
	s_and_b32 s2, s72, s2
	s_set_vgpr_msb 0x5506
	v_cmp_gt_i32_e32 vcc_lo, v11 /*v523*/, v139 /*v395*/
	v_nop
	s_set_vgpr_msb 0x649
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4954
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[170:177], v[220:227] /*v[476:483]*/, v[148:155] /*v[404:411]*/
	s_set_vgpr_msb 0x5449
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[146:147] /*v[402:403]*/, v[146:147] /*v[402:403]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[168:169] /*v[424:425]*/
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[146:147] /*v[402:403]*/, v[170:171] /*v[426:427]*/
	s_set_vgpr_msb 0x4554
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[178:185], v[228:235] /*v[484:491]*/, v[148:155] /*v[404:411]*/
	s_set_vgpr_msb 0x5445
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[130:131] /*v[386:387]*/, v[140:141] /*v[396:397]*/
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[130:131] /*v[386:387]*/, v[142:143] /*v[398:399]*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v170 /*v426*/, v171 /*v427*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v140 /*v396*/, v141 /*v397*/
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[130:131] /*v[386:387]*/, v[146:147] /*v[402:403]*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[130:131] /*v[386:387]*/, v[148:149] /*v[404:405]*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v142 /*v398*/, v143 /*v399*/
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[130:131] /*v[386:387]*/, v[144:145] /*v[400:401]*/
	s_set_vgpr_msb 0x4544
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[194:201], v[236:243] /*v[492:499]*/, 0
	s_set_vgpr_msb 0x4445
	v_cvt_pk_bf16_f32 v167 /*v423*/, v140 /*v396*/, v141 /*v397*/
	v_cndmask_b32_e64 v149 /*v405*/, v149 /*v405*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v148 /*v404*/, v148 /*v404*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[130:131] /*v[386:387]*/, v[150:151] /*v[406:407]*/
	s_set_vgpr_msb 0x4506
	v_cmp_gt_i32_e64 s2, v12 /*v524*/, v139 /*v395*/
	s_and_b32 s3, s72, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v9 /*v521*/, v139 /*v395*/
	s_set_vgpr_msb 0x649
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[148:149] /*v[404:405]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v141 /*v397*/, v141 /*v397*/, 0xff61b1e6, s3
	v_cmp_lt_i32_e64 s3, v139 /*v395*/, v10 /*v522*/
	s_and_b32 s2, s72, s2
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v166 /*v422*/, v144 /*v400*/, v145 /*v401*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[142:143] /*v[398:399]*/, s[18:19] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[130:131] /*v[386:387]*/, v[152:153] /*v[408:409]*/
	v_cndmask_b32_e64 v140 /*v396*/, v140 /*v396*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, vcc_lo
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[130:131] /*v[386:387]*/, v[154:155] /*v[410:411]*/
	s_set_vgpr_msb 0x4506
	v_cmp_gt_i32_e32 vcc_lo, v7 /*v519*/, v139 /*v395*/
	s_set_vgpr_msb 0x649
	v_cndmask_b32_e64 v143 /*v399*/, v143 /*v399*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s3
	v_pk_add_f32 v[150:151] /*v[406:407]*/, v[140:141] /*v[396:397]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v142 /*v398*/, v142 /*v398*/, 0xff61b1e6, s2
	v_cmp_lt_i32_e64 s2, v139 /*v395*/, v8 /*v520*/
	s_set_vgpr_msb 0x4954
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[202:209], v[244:251] /*v[500:507]*/, v[156:163] /*v[412:419]*/
	s_and_b32 s3, s72, vcc_lo
	s_set_vgpr_msb 0x5406
	v_cmp_ge_i32_e32 vcc_lo, v17 /*v529*/, v139 /*v395*/
	s_set_vgpr_msb 0x649
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[142:143] /*v[398:399]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_and_b32 s2, s72, s2
	v_cndmask_b32_e64 v177 /*v433*/, v153 /*v409*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v176 /*v432*/, v152 /*v408*/, 0xff61b1e6, s2
	v_cmp_lt_i32_e64 s2, v139 /*v395*/, v17 /*v529*/
	s_set_vgpr_msb 0x4944
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[234:241], v[204:211] /*v[460:467]*/, 0
	s_and_b32 s3, s72, vcc_lo
	s_set_vgpr_msb 0x4406
	v_cmp_gt_i32_e32 vcc_lo, v5 /*v517*/, v139 /*v395*/
	s_set_vgpr_msb 0x641
	v_exp_f32_e32 v174 /*v430*/, v148 /*v404*/
	s_and_b32 s2, s72, s2
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[150:151] /*v[406:407]*/, s[18:19] op_sel_hi:[1,0]
	v_exp_f32_e32 v175 /*v431*/, v149 /*v405*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[154:155] /*v[410:411]*/, s[18:19] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4154
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[242:249], v[212:219] /*v[468:475]*/, v[140:147] /*v[396:403]*/
	s_set_vgpr_msb 0x5449
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v178 /*v434*/, v178 /*v434*/
	v_exp_f32_e32 v179 /*v435*/, v179 /*v435*/
	s_set_vgpr_msb 0x4955
	v_cvt_pk_bf16_f32 v169 /*v425*/, v168 /*v424*/, v169 /*v425*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v172 /*v428*/, v173 /*v429*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, s[18:19] op_sel_hi:[1,0]
	v_exp_f32_e32 v180 /*v436*/, v180 /*v436*/
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[2:9] /*v[258:265]*/, v[220:227] /*v[476:483]*/, v[140:147] /*v[396:403]*/
	v_exp_f32_e32 v181 /*v437*/, v181 /*v437*/
	v_exp_f32_e32 v176 /*v432*/, v176 /*v432*/
	v_exp_f32_e32 v177 /*v433*/, v177 /*v433*/
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[396:403]*/, v[10:17] /*v[266:273]*/, v[228:235] /*v[484:491]*/, v[140:147] /*v[396:403]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[130:131] /*v[386:387]*/, v[140:141] /*v[396:397]*/
	s_set_vgpr_msb 0x5558
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[210:217], v[28:35] /*v[540:547]*/, v[156:163] /*v[412:419]*/
	s_set_vgpr_msb 0x5845
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[130:131] /*v[386:387]*/, v[142:143] /*v[398:399]*/
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[130:131] /*v[386:387]*/, v[144:145] /*v[400:401]*/
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[130:131] /*v[386:387]*/, v[146:147] /*v[402:403]*/
	v_cndmask_b32_e64 v140 /*v396*/, v140 /*v396*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4506
	v_cmp_gt_i32_e64 s2, v6 /*v518*/, v139 /*v395*/
	s_set_vgpr_msb 0x645
	v_cndmask_b32_e64 v141 /*v397*/, v141 /*v397*/, 0xff61b1e6, s3
	s_and_b32 s3, s72, vcc_lo
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[18:25] /*v[274:281]*/, v[236:243] /*v[492:499]*/, 0
	s_set_vgpr_msb 0x4506
	v_cmp_gt_i32_e32 vcc_lo, v3 /*v515*/, v139 /*v395*/
	s_and_b32 s2, s72, s2
	s_set_vgpr_msb 0x641
	v_cndmask_b32_e64 v143 /*v399*/, v143 /*v399*/, 0xff61b1e6, s3
	v_cndmask_b32_e64 v142 /*v398*/, v142 /*v398*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4106
	v_cmp_gt_i32_e64 s2, v4 /*v516*/, v139 /*v395*/
	s_and_b32 s3, s72, vcc_lo
	s_set_vgpr_msb 0x604
	v_cmp_gt_i32_e32 vcc_lo, v1, v139 /*v395*/
	s_set_vgpr_msb 0x458
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[412:419]*/, v[226:233], v[36:43] /*v[548:555]*/, v[156:163] /*v[412:419]*/
	s_set_vgpr_msb 0x5841
	v_cndmask_b32_e64 v145 /*v401*/, v145 /*v401*/, 0xff61b1e6, s3
	s_set_vgpr_msb 0x4106
	v_cmp_gt_i32_e64 s3, v2 /*v514*/, v139 /*v395*/
	s_and_b32 s2, s72, s2
	s_set_vgpr_msb 0x649
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v144 /*v400*/, v144 /*v400*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, vcc_lo
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4955
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[26:33] /*v[282:289]*/, v[244:251] /*v[500:507]*/, v[148:155] /*v[404:411]*/
	v_cndmask_b32_e64 v147 /*v403*/, v147 /*v403*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s3
	s_set_vgpr_msb 0x5559
	s_wait_loadcnt 0x0
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v146 /*v402*/, v146 /*v402*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, s[18:19] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[34:41] /*v[290:297]*/, v[28:35] /*v[540:547]*/, v[148:155] /*v[404:411]*/
	s_set_vgpr_msb 0x5945
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, v[176:177] /*v[432:433]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[146:147] /*v[402:403]*/, v[146:147] /*v[402:403]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[130:131] /*v[386:387]*/, v[156:157] /*v[412:413]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[130:131] /*v[386:387]*/, v[162:163] /*v[418:419]*/
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[404:411]*/, v[42:49] /*v[298:305]*/, v[36:43] /*v[548:555]*/, v[148:155] /*v[404:411]*/
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, s[18:19] op_sel_hi:[1,0]
	v_exp_f32_e32 v172 /*v428*/, v140 /*v396*/
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, s[18:19] op_sel_hi:[1,0]
	v_exp_f32_e32 v173 /*v429*/, v141 /*v397*/
	v_nop
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[146:147] /*v[402:403]*/, s[18:19] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x5945
	v_cvt_pk_bf16_f32 v156 /*v412*/, v156 /*v412*/, v157 /*v413*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v176 /*v432*/, v177 /*v433*/
	v_exp_f32_e32 v176 /*v432*/, v142 /*v398*/
	v_exp_f32_e32 v177 /*v433*/, v143 /*v399*/
	v_exp_f32_e32 v144 /*v400*/, v144 /*v400*/
	v_exp_f32_e32 v145 /*v401*/, v145 /*v401*/
	v_exp_f32_e32 v146 /*v402*/, v140 /*v396*/
	v_exp_f32_e32 v147 /*v403*/, v141 /*v397*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[148:149] /*v[404:405]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[150:151] /*v[406:407]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[148:149] /*v[404:405]*/, v[152:153] /*v[408:409]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[150:151] /*v[406:407]*/, v[154:155] /*v[410:411]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[180:181] /*v[436:437]*/
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[148:149] /*v[404:405]*/, v[144:145] /*v[400:401]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[150:151] /*v[406:407]*/, v[146:147] /*v[402:403]*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[130:131] /*v[386:387]*/, v[160:161] /*v[416:417]*/
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[130:131] /*v[386:387]*/, v[140:141] /*v[396:397]*/
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[130:131] /*v[386:387]*/, v[142:143] /*v[398:399]*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[130:131] /*v[386:387]*/, v[148:149] /*v[404:405]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[130:131] /*v[386:387]*/, v[150:151] /*v[406:407]*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v160 /*v416*/, v161 /*v417*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v174 /*v430*/, v175 /*v431*/
	v_cvt_pk_bf16_f32 v140 /*v396*/, v140 /*v396*/, v141 /*v397*/
	v_cvt_pk_bf16_f32 v141 /*v397*/, v142 /*v398*/, v143 /*v399*/
	v_cvt_pk_bf16_f32 v142 /*v398*/, v148 /*v404*/, v149 /*v405*/
	v_cvt_pk_bf16_f32 v143 /*v399*/, v150 /*v406*/, v151 /*v407*/
	v_cvt_pk_bf16_f32 v147 /*v403*/, v146 /*v402*/, v147 /*v403*/
	v_cvt_pk_bf16_f32 v146 /*v402*/, v144 /*v400*/, v145 /*v401*/
	v_cvt_pk_bf16_f32 v145 /*v401*/, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v144 /*v400*/, v172 /*v428*/, v173 /*v429*/
	ds_store_b128 v133 /*v389*/, v[252:255] /*v[508:511]*/
	ds_store_b128 v133 /*v389*/, v[168:171] /*v[424:427]*/ offset:32
	s_set_vgpr_msb 0x4509
	ds_store_b128 v133 /*v389*/, v[52:55] /*v[564:567]*/ offset:2560
	s_set_vgpr_msb 0x945
	ds_store_b128 v133 /*v389*/, v[164:167] /*v[420:423]*/ offset:2592
	ds_store_b128 v134 /*v390*/, v[160:163] /*v[416:419]*/
	ds_store_b128 v134 /*v390*/, v[144:147] /*v[400:403]*/ offset:32
	ds_store_b128 v134 /*v390*/, v[156:159] /*v[412:415]*/ offset:2560
	ds_store_b128 v134 /*v390*/, v[140:143] /*v[396:399]*/ offset:2592
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v135 /*v391*/
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v135 /*v391*/ offset:1280
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v136 /*v392*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v136 /*v392*/ offset:1280
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v137 /*v393*/
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v137 /*v393*/ offset:1280
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v138 /*v394*/
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v138 /*v394*/ offset:1280
	s_set_vgpr_msb 0x4542
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v19 /*v531*/
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v19 /*v531*/ offset:32
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v19 /*v531*/ offset:4352
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v19 /*v531*/ offset:4384
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v19 /*v531*/ offset:8704
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v19 /*v531*/ offset:8736
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v19 /*v531*/ offset:13056
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v19 /*v531*/ offset:13088
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v19 /*v531*/ offset:64
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v19 /*v531*/ offset:96
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v19 /*v531*/ offset:4416
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v19 /*v531*/ offset:4448
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v19 /*v531*/ offset:8768
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v19 /*v531*/ offset:8800
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v19 /*v531*/ offset:13120
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v19 /*v531*/ offset:13152
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v19 /*v531*/ offset:128
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v19 /*v531*/ offset:160
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v19 /*v531*/ offset:4480
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v19 /*v531*/ offset:4512
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v19 /*v531*/ offset:8832
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v19 /*v531*/ offset:8864
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v19 /*v531*/ offset:13184
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v19 /*v531*/ offset:13216
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v19 /*v531*/ offset:192
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v19 /*v531*/ offset:224
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v19 /*v531*/ offset:4544
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v19 /*v531*/ offset:4576
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v19 /*v531*/ offset:8896
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v19 /*v531*/ offset:8928
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v19 /*v531*/ offset:13248
	ds_load_tr16_b128 v[72:75] /*v[584:587]*/, v19 /*v531*/ offset:13280
	s_set_vgpr_msb 0x8255
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[140:147] /*v[396:403]*/, v[172:179] /*v[428:435]*/, v[74:81] /*v[330:337]*/
	s_add_nc_u64 s[58:59], s[58:59], 1
	s_mov_b32 s2, s74
	s_cmp_lg_u64 s[58:59], s[56:57]
	s_mov_b32 s3, s75
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[140:147] /*v[396:403]*/, v[180:187] /*v[436:443]*/, v[58:65] /*v[314:321]*/ matrix_a_reuse
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[140:147] /*v[396:403]*/, v[204:211] /*v[460:467]*/, v[50:57] /*v[306:313]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5505
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[250:257], v[140:147] /*v[396:403]*/, v[212:219] /*v[468:475]*/, v[250:257] matrix_a_reuse
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[186:193], v[140:147] /*v[396:403]*/, v[236:243] /*v[492:499]*/, v[186:193] matrix_a_reuse
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[146:153], v[140:147] /*v[396:403]*/, v[244:251] /*v[500:507]*/, v[146:153] matrix_a_reuse
	s_set_vgpr_msb 0x509
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[138:145], v[140:147] /*v[396:403]*/, v[44:51] /*v[556:563]*/, v[138:145] matrix_a_reuse
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[122:129], v[140:147] /*v[396:403]*/, v[52:59] /*v[564:571]*/, v[122:129] matrix_a_reuse
	s_set_vgpr_msb 0x955
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[148:155] /*v[404:411]*/, v[188:195] /*v[444:451]*/, v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[148:155] /*v[404:411]*/, v[196:203] /*v[452:459]*/, v[114:121] /*v[370:377]*/ matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[148:155] /*v[404:411]*/, v[220:227] /*v[476:483]*/, v[106:113] /*v[362:369]*/ matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[148:155] /*v[404:411]*/, v[228:235] /*v[484:491]*/, v[98:105] /*v[354:361]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5559
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[148:155] /*v[404:411]*/, v[28:35] /*v[540:547]*/, v[90:97] /*v[346:353]*/ matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[148:155] /*v[404:411]*/, v[36:43] /*v[548:555]*/, v[82:89] /*v[338:345]*/ matrix_a_reuse
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[148:155] /*v[404:411]*/, v[60:67] /*v[572:579]*/, v[66:73] /*v[322:329]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5909
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[218:225], v[148:155] /*v[404:411]*/, v[68:75] /*v[580:587]*/, v[218:225] matrix_a_reuse
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[82:89], v[156:163] /*v[412:419]*/, v[172:179] /*v[428:435]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[156:163] /*v[412:419]*/, v[180:187] /*v[436:443]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[156:163] /*v[412:419]*/, v[204:211] /*v[460:467]*/, v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[156:163] /*v[412:419]*/, v[212:219] /*v[468:475]*/, v[34:41] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[156:163] /*v[412:419]*/, v[236:243] /*v[492:499]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[156:163] /*v[412:419]*/, v[244:251] /*v[500:507]*/, v[18:25] matrix_a_reuse
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[10:17], v[156:163] /*v[412:419]*/, v[44:51] /*v[556:563]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[156:163] /*v[412:419]*/, v[52:59] /*v[564:571]*/, v[2:9] matrix_a_reuse
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[130:137], v[164:171] /*v[420:427]*/, v[188:195] /*v[444:451]*/, v[130:137]
	v_wmma_f32_16x16x32_bf16 v[114:121], v[164:171] /*v[420:427]*/, v[196:203] /*v[452:459]*/, v[114:121] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[164:171] /*v[420:427]*/, v[220:227] /*v[476:483]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[164:171] /*v[420:427]*/, v[228:235] /*v[484:491]*/, v[98:105] matrix_a_reuse
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[90:97], v[164:171] /*v[420:427]*/, v[28:35] /*v[540:547]*/, v[90:97] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[164:171] /*v[420:427]*/, v[36:43] /*v[548:555]*/, v[74:81] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[164:171] /*v[420:427]*/, v[60:67] /*v[572:579]*/, v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[164:171] /*v[420:427]*/, v[68:75] /*v[580:587]*/, v[42:49] matrix_a_reuse
	s_set_vgpr_msb 0x900
	s_cbranch_scc1 .LBB0_4
	s_branch .LBB0_6
.LBB0_5:
	v_mov_b32_e32 v42, 0
	v_dual_mov_b32 v43, v42 :: v_dual_mov_b32 v44, v42
	v_dual_mov_b32 v45, v42 :: v_dual_mov_b32 v46, v42
	v_dual_mov_b32 v47, v42 :: v_dual_mov_b32 v48, v42
	v_mov_b32_e32 v49, v42
	v_mov_b64_e32 v[68:69], v[44:45]
	v_mov_b64_e32 v[66:67], v[42:43]
	v_mov_b64_e32 v[70:71], v[46:47]
	v_mov_b64_e32 v[78:79], v[46:47]
	v_mov_b64_e32 v[72:73], v[48:49]
	v_mov_b64_e32 v[80:81], v[48:49]
	v_mov_b64_e32 v[76:77], v[44:45]
	v_mov_b64_e32 v[74:75], v[42:43]
	v_mov_b64_e32 v[96:97], v[48:49]
	v_mov_b64_e32 v[94:95], v[46:47]
	v_mov_b64_e32 v[92:93], v[44:45]
	v_mov_b64_e32 v[90:91], v[42:43]
	v_mov_b64_e32 v[104:105], v[48:49]
	v_mov_b64_e32 v[102:103], v[46:47]
	v_mov_b64_e32 v[100:101], v[44:45]
	v_mov_b64_e32 v[98:99], v[42:43]
	v_mov_b64_e32 v[112:113], v[48:49]
	v_mov_b64_e32 v[110:111], v[46:47]
	v_mov_b64_e32 v[108:109], v[44:45]
	v_mov_b64_e32 v[106:107], v[42:43]
	v_mov_b64_e32 v[120:121], v[48:49]
	v_mov_b64_e32 v[118:119], v[46:47]
	v_mov_b64_e32 v[116:117], v[44:45]
	v_mov_b64_e32 v[114:115], v[42:43]
	v_mov_b64_e32 v[136:137], v[48:49]
	v_mov_b64_e32 v[134:135], v[46:47]
	v_mov_b64_e32 v[132:133], v[44:45]
	v_mov_b64_e32 v[130:131], v[42:43]
	v_mov_b64_e32 v[224:225], v[48:49]
	v_mov_b64_e32 v[222:223], v[46:47]
	v_mov_b64_e32 v[220:221], v[44:45]
	v_mov_b64_e32 v[218:219], v[42:43]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[48:49]
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[46:47]
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[44:45]
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[42:43]
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[48:49]
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[46:47]
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[44:45]
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[42:43]
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, v[48:49]
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, v[46:47]
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, v[44:45]
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, v[42:43]
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[48:49]
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[46:47]
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[44:45]
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[42:43]
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[48:49]
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[46:47]
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[44:45]
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[42:43]
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[48:49]
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[46:47]
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[44:45]
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[42:43]
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[48:49]
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[46:47]
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[44:45]
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[42:43]
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[2:3], v[42:43]
	v_mov_b64_e32 v[4:5], v[44:45]
	v_mov_b64_e32 v[6:7], v[46:47]
	v_mov_b64_e32 v[8:9], v[48:49]
	v_mov_b64_e32 v[10:11], v[42:43]
	v_mov_b64_e32 v[12:13], v[44:45]
	v_mov_b64_e32 v[14:15], v[46:47]
	v_mov_b64_e32 v[16:17], v[48:49]
	v_mov_b64_e32 v[18:19], v[42:43]
	v_mov_b64_e32 v[20:21], v[44:45]
	v_mov_b64_e32 v[22:23], v[46:47]
	v_mov_b64_e32 v[24:25], v[48:49]
	v_mov_b64_e32 v[26:27], v[42:43]
	v_mov_b64_e32 v[28:29], v[44:45]
	v_mov_b64_e32 v[30:31], v[46:47]
	v_mov_b64_e32 v[32:33], v[48:49]
	v_mov_b64_e32 v[34:35], v[42:43]
	v_mov_b64_e32 v[36:37], v[44:45]
	v_mov_b64_e32 v[38:39], v[46:47]
	v_mov_b64_e32 v[40:41], v[48:49]
	v_mov_b64_e32 v[56:57], v[48:49]
	v_mov_b64_e32 v[54:55], v[46:47]
	v_mov_b64_e32 v[52:53], v[44:45]
	v_mov_b64_e32 v[50:51], v[42:43]
	v_mov_b64_e32 v[64:65], v[48:49]
	v_mov_b64_e32 v[62:63], v[46:47]
	v_mov_b64_e32 v[60:61], v[44:45]
	v_mov_b64_e32 v[58:59], v[42:43]
	v_mov_b64_e32 v[88:89], v[48:49]
	v_mov_b64_e32 v[86:87], v[46:47]
	v_mov_b64_e32 v[84:85], v[44:45]
	v_mov_b64_e32 v[82:83], v[42:43]
	v_mov_b64_e32 v[128:129], v[48:49]
	v_mov_b64_e32 v[126:127], v[46:47]
	v_mov_b64_e32 v[124:125], v[44:45]
	v_mov_b64_e32 v[122:123], v[42:43]
	v_mov_b64_e32 v[144:145], v[48:49]
	v_mov_b64_e32 v[142:143], v[46:47]
	v_mov_b64_e32 v[140:141], v[44:45]
	v_mov_b64_e32 v[138:139], v[42:43]
	v_mov_b64_e32 v[152:153], v[48:49]
	v_mov_b64_e32 v[150:151], v[46:47]
	v_mov_b64_e32 v[148:149], v[44:45]
	v_mov_b64_e32 v[146:147], v[42:43]
	v_mov_b64_e32 v[192:193], v[48:49]
	v_mov_b64_e32 v[190:191], v[46:47]
	v_mov_b64_e32 v[188:189], v[44:45]
	v_mov_b64_e32 v[186:187], v[42:43]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[0:1] /*v[256:257]*/, v[48:49]
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[254:255], v[46:47]
	v_mov_b64_e32 v[252:253], v[44:45]
	v_mov_b64_e32 v[250:251], v[42:43]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[56:57] /*v[312:313]*/, v[48:49]
	v_mov_b64_e32 v[54:55] /*v[310:311]*/, v[46:47]
	v_mov_b64_e32 v[52:53] /*v[308:309]*/, v[44:45]
	v_mov_b64_e32 v[50:51] /*v[306:307]*/, v[42:43]
	v_mov_b64_e32 v[64:65] /*v[320:321]*/, v[48:49]
	v_mov_b64_e32 v[62:63] /*v[318:319]*/, v[46:47]
	v_mov_b64_e32 v[60:61] /*v[316:317]*/, v[44:45]
	v_mov_b64_e32 v[58:59] /*v[314:315]*/, v[42:43]
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[48:49]
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[46:47]
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[44:45]
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[42:43]
	s_set_vgpr_msb 0x4000
.LBB0_6:
	s_sub_co_i32 s2, s67, s65
	s_sub_co_i32 s24, s66, s73
	s_max_i32 s2, s2, 0
	v_nop
	s_set_vgpr_msb 0x8a
	v_lshlrev_b32_e32 v28 /*v540*/, 2, v13 /*v525*/
	s_add_co_i32 s3, s50, s2
	s_mov_b32 s38, s30
	s_add_co_i32 s3, s3, -1
	s_mov_b32 s39, s31
	s_abs_i32 s4, s3
	s_ashr_i32 s6, s3, 31
	s_mul_hi_u32 s5, s4, s70
	s_xor_b32 s6, s6, s69
	s_mul_i32 s7, s5, s68
	s_mov_b32 s11, 0
	s_sub_co_i32 s4, s4, s7
	s_add_co_i32 s7, s5, 1
	s_sub_co_i32 s8, s4, s68
	s_cmp_ge_u32 s4, s68
	s_mov_b32 s20, 1
	s_cselect_b32 s5, s7, s5
	s_cselect_b32 s4, s8, s4
	s_add_co_i32 s7, s5, 1
	s_cmp_ge_u32 s4, s68
	s_mov_b32 s21, s11
	s_cselect_b32 s4, s7, s5
	s_mov_b32 s8, 32
	s_xor_b32 s4, s4, s6
	s_sub_co_i32 s5, s4, s6
	s_mul_i32 s5, s5, s50
	s_cmp_lg_u32 s3, s5
	s_cselect_b32 s5, -1, 0
	s_xor_b32 s3, s3, s50
	s_cmp_lt_i32 s3, 0
	s_cselect_b32 s3, -1, 0
	s_and_b32 s3, s3, s5
	s_sub_co_ci_u32 s3, s4, s6
	s_add_co_i32 s18, s65, s64
	s_mul_i32 s19, s3, s24
	s_add_co_i32 s63, s63, -1
	s_add_co_i32 s26, s18, s19
	s_add_co_i32 s25, s14, s15
	s_min_i32 s5, s26, s63
	s_sub_co_i32 s2, s2, s19
	s_max_i32 s5, s5, 0
	s_mul_i32 s4, s25, s13
	s_max_i32 s2, s2, 0
	s_lshl_b32 s5, s5, 5
	s_min_i32 s27, s2, s3
	s_add_co_i32 s3, s5, s4
	s_add_co_i32 s2, s5, s62
	s_sub_co_i32 s4, s13, s5
	s_lshl_b32 s5, s3, 2
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s15, s14, 31
	s_mul_u64 s[2:3], s[34:35], s[2:3]
	s_add_co_i32 s6, s5, 64
	s_add_nc_u64 s[2:3], s[2:3], s[14:15]
	s_clause 0x1
	buffer_load_b32 v22 /*v534*/, v28 /*v540*/, s[28:31], s5 offen
	buffer_load_b32 v20 /*v532*/, v28 /*v540*/, s[28:31], s6 offen
	s_lshl_b64 s[2:3], s[2:3], 8
	s_cmp_lg_u32 s71, 0x80000000
	s_clause 0x1
	buffer_load_b32 v18 /*v530*/, v28 /*v540*/, s[36:39], s5 offen
	buffer_load_b32 v16 /*v528*/, v28 /*v540*/, s[36:39], s6 offen
	s_cselect_b32 s9, s71, 0x80
	s_max_i32 s4, s4, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[22:23], s[54:55], s[2:3]
	s_wait_xcnt 0x1
	s_lshl_b32 s5, s4, 16
	s_lshr_b32 s4, s4, 16
	s_wait_xcnt 0x0
	s_or_b32 s6, s5, 0x7fff
	s_ashr_i32 s5, s9, 31
	s_or_b32 s7, s4, 0x800000
	s_and_b32 s10, s5, 0xffff
	s_bitset1_b32 s23, 31
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, 0x7510000
	s_cmp_gt_i32 s17, 1
	tensor_load_to_lds s[20:23], s[4:11]
	s_add_nc_u64 s[22:23], s[52:53], s[2:3]
	s_cselect_b32 s3, -1, 0
	s_cmp_lt_i32 s17, 2
	s_mul_i32 s2, s27, s17
	s_cselect_b32 s15, -1, 0
	s_cmp_gt_i32 s2, 1
	s_bitset1_b32 s23, 31
	s_cselect_b32 s21, -1, 0
	s_and_b32 s15, s15, s21
	s_and_b32 s3, s3, s21
	s_cmp_lg_u32 s15, 0
	s_movk_i32 s21, 0x2200
	s_add_co_ci_u32 s15, s18, s19
	tensor_load_to_lds s[20:23], s[4:11]
	s_min_i32 s15, s15, s63
	s_movk_i32 s21, 0x4400
	s_max_i32 s15, s15, 0
	s_cmp_lg_u32 s3, 0
	s_add_co_ci_u32 s18, s14, 0
	s_lshl_b32 s3, s15, 5
	s_ashr_i32 s19, s18, 31
	s_add_co_i32 s38, s3, s62
	s_sub_co_i32 s3, s13, s3
	s_ashr_i32 s39, s38, 31
	s_max_i32 s3, s3, 0
	s_mul_u64 s[38:39], s[34:35], s[38:39]
	s_lshl_b32 s6, s3, 16
	s_add_nc_u64 s[18:19], s[38:39], s[18:19]
	s_lshr_b32 s3, s3, 16
	s_lshl_b64 s[18:19], s[18:19], 8
	s_addk_co_i32 s6, 0x7fff
	s_add_nc_u64 s[22:23], s[54:55], s[18:19]
	s_or_b32 s7, s3, 0x800000
	s_bitset1_b32 s23, 31
	s_mov_b32 s15, 2
	tensor_load_to_lds s[20:23], s[4:11]
	s_add_nc_u64 s[22:23], s[52:53], s[18:19]
	s_movk_i32 s21, 0x6600
	s_bitset1_b32 s23, 31
	tensor_load_to_lds s[20:23], s[4:11]
	s_wait_tensorcnt 0x2
	s_cmp_lt_i32 s2, 1
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_9
	s_set_vgpr_msb 0x42
	ds_load_b128 v[134:137] /*v[390:393]*/, v21 /*v533*/ offset:4576
	ds_load_b128 v[130:133] /*v[386:389]*/, v21 /*v533*/ offset:4544
	ds_load_b128 v[150:153] /*v[406:409]*/, v21 /*v533*/ offset:4512
	ds_load_b128 v[146:149] /*v[402:405]*/, v21 /*v533*/ offset:4480
	ds_load_b128 v[166:169] /*v[422:425]*/, v21 /*v533*/ offset:4448
	ds_load_b128 v[162:165] /*v[418:421]*/, v21 /*v533*/ offset:4416
	ds_load_b128 v[198:201] /*v[454:457]*/, v21 /*v533*/ offset:4384
	ds_load_b128 v[194:197] /*v[450:453]*/, v21 /*v533*/ offset:4352
	ds_load_b128 v[182:185] /*v[438:441]*/, v21 /*v533*/ offset:13280
	ds_load_b128 v[178:181] /*v[434:437]*/, v21 /*v533*/ offset:13248
	ds_load_b128 v[214:217] /*v[470:473]*/, v21 /*v533*/ offset:13216
	ds_load_b128 v[210:213] /*v[466:469]*/, v21 /*v533*/ offset:13184
	ds_load_b128 v[230:233] /*v[486:489]*/, v21 /*v533*/ offset:13152
	ds_load_b128 v[226:229] /*v[482:485]*/, v21 /*v533*/ offset:13120
	ds_load_b128 v[246:249] /*v[502:505]*/, v21 /*v533*/ offset:13088
	ds_load_b128 v[242:245] /*v[498:501]*/, v21 /*v533*/ offset:13056
	ds_load_b128 v[142:145] /*v[398:401]*/, v21 /*v533*/ offset:224
	ds_load_b128 v[138:141] /*v[394:397]*/, v21 /*v533*/ offset:192
	ds_load_b128 v[158:161] /*v[414:417]*/, v21 /*v533*/ offset:160
	ds_load_b128 v[154:157] /*v[410:413]*/, v21 /*v533*/ offset:128
	ds_load_b128 v[174:177] /*v[430:433]*/, v21 /*v533*/ offset:96
	ds_load_b128 v[170:173] /*v[426:429]*/, v21 /*v533*/ offset:64
	ds_load_b128 v[206:209] /*v[462:465]*/, v21 /*v533*/ offset:32
	ds_load_b128 v[202:205] /*v[458:461]*/, v21 /*v533*/
	ds_load_b128 v[190:193] /*v[446:449]*/, v21 /*v533*/ offset:8928
	ds_load_b128 v[186:189] /*v[442:445]*/, v21 /*v533*/ offset:8896
	ds_load_b128 v[222:225] /*v[478:481]*/, v21 /*v533*/ offset:8864
	ds_load_b128 v[218:221] /*v[474:477]*/, v21 /*v533*/ offset:8832
	ds_load_b128 v[238:241] /*v[494:497]*/, v21 /*v533*/ offset:8800
	ds_load_b128 v[234:237] /*v[490:493]*/, v21 /*v533*/ offset:8768
	ds_load_b128 v[254:257] /*v[510:513]*/, v21 /*v533*/ offset:8736
	ds_load_b128 v[250:253] /*v[506:509]*/, v21 /*v533*/ offset:8704
	s_mov_b32 s3, 0x10a00
	s_set_vgpr_msb 0x428a
	v_or_b32_e32 v30 /*v542*/, 0x10000, v26 /*v538*/
	v_mad_u32_u24 v31 /*v543*/, 0x50, v14 /*v526*/, s3
	s_mov_b32 s6, s12
	s_mov_b32 s7, s12
	v_add3_u32 v29 /*v541*/, v21 /*v533*/, v15 /*v527*/, 0x10000
	v_mov_b64_e32 v[14:15] /*v[526:527]*/, s[6:7]
	v_or_b32_e32 v27 /*v539*/, 0x10000, v27 /*v539*/
	v_dual_add_nc_u32 v26 /*v538*/, v30 /*v542*/, v24 /*v536*/ :: v_dual_add_nc_u32 v24 /*v536*/, v31 /*v543*/, v24 /*v536*/
	v_dual_add_nc_u32 v30 /*v542*/, v30 /*v542*/, v25 /*v537*/ :: v_dual_add_nc_u32 v25 /*v537*/, v31 /*v543*/, v25 /*v537*/
	s_mov_b32 s3, 0
	s_mov_b32 s12, 0x3fb8aa3b
	s_sub_nc_u64 s[18:19], 0, s[2:3]
	s_mov_b32 s7, s11
	s_mov_b32 s6, s11
	s_mov_b32 s41, s11
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
	s_add_co_ci_u32 s27, s6, 0
	s_cmp_lt_i32 s21, s2
	s_cselect_b32 s22, s27, s6
	s_cselect_b32 s42, s3, s7
	s_add_co_i32 s7, s3, 1
	s_cmp_ge_i32 s7, s17
	s_cselect_b32 s21, -1, 0
	s_and_b32 s38, s21, exec_lo
	s_cselect_b32 s7, 0, s7
	s_cmp_lg_u32 s21, 0
	s_add_co_ci_u32 s6, s6, s23
	s_cmp_lt_i32 s15, s2
	s_cselect_b32 s6, s6, s22
	s_cselect_b32 s7, s7, s42
	s_add_co_i32 s21, s41, 0xffffbc00
	s_add_co_i32 s6, s6, s26
	s_cmp_lg_u32 s41, 0
	s_cselect_b32 s21, s21, 0x8800
	s_add_co_i32 s23, s41, 0x4400
	s_cmp_lg_u32 s41, 0x8800
	s_cselect_b32 s40, s23, 0
	s_add_co_i32 s43, s22, s26
	s_lshl_b32 s38, s6, 5
	s_add_co_i32 s6, s7, s14
	s_add_co_i32 s22, s38, s62
	s_ashr_i32 s7, s6, 31
	s_ashr_i32 s23, s22, 31
	s_sub_co_i32 s38, s13, s38
	s_mul_u64 s[22:23], s[34:35], s[22:23]
	s_max_i32 s44, s38, 0
	s_add_nc_u64 s[6:7], s[22:23], s[6:7]
	s_lshl_b32 s45, s44, 16
	s_lshl_b64 s[38:39], s[6:7], 8
	s_lshr_b32 s7, s44, 16
	s_add_nc_u64 s[22:23], s[54:55], s[38:39]
	s_or_b32 s6, s45, 0x7fff
	s_bitset1_b32 s23, 31
	s_bitset1_b32 s7, 23
	tensor_load_to_lds s[20:23], s[4:11]
	s_add_nc_u64 s[22:23], s[52:53], s[38:39]
	s_addk_co_i32 s21, 0x2200
	s_bitset1_b32 s23, 31
	tensor_load_to_lds s[20:23], s[4:11]
	s_add_co_i32 s6, s25, s42
	s_lshl_b32 s7, s43, 7
	s_mul_i32 s6, s61, s6
	s_mov_b32 s38, s30
	s_add_co_i32 s6, s7, s6
	s_mov_b32 s39, s31
	s_add_co_i32 s7, s6, 64
	s_set_vgpr_msb 0x82
	s_clause 0x1
	buffer_load_b32 v192 /*v704*/, v28 /*v540*/, s[36:39], s6 offen
	buffer_load_b32 v193 /*v705*/, v28 /*v540*/, s[36:39], s7 offen
	s_clause 0x1
	buffer_load_b32 v194 /*v706*/, v28 /*v540*/, s[28:31], s7 offen
	buffer_load_b32 v31 /*v543*/, v28 /*v540*/, s[28:31], s6 offen
	s_set_vgpr_msb 0x8284
	s_wait_loadcnt_dscnt 0x2600
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[154:161], v[250:257] /*v[506:513]*/, 0
	s_wait_loadcnt 0x16
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[234:241], v[250:257] /*v[506:513]*/, 0
	s_set_vgpr_msb 0x8444
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[154:161], v[242:249] /*v[498:505]*/, 0
	s_set_vgpr_msb 0x44a4
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[560:567]*/, v[234:241], v[242:249] /*v[498:505]*/, 0
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[162:169], v[234:241] /*v[490:497]*/, v[32:39] /*v[544:551]*/
	s_wait_loadcnt 0x14
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[242:249], v[234:241] /*v[490:497]*/, v[40:47] /*v[552:559]*/
	s_set_vgpr_msb 0xa454
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[162:169], v[226:233] /*v[482:489]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x54a4
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[560:567]*/, v[242:249], v[226:233] /*v[482:489]*/, v[48:55] /*v[560:567]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[170:177], v[218:225] /*v[474:481]*/, v[32:39] /*v[544:551]*/
	s_set_vgpr_msb 0xa4a5
	s_wait_loadcnt 0x12
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[2:9] /*v[258:265]*/, v[218:225] /*v[474:481]*/, v[40:47] /*v[552:559]*/
	s_set_vgpr_msb 0xa554
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[170:177], v[210:217] /*v[466:473]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x54a5
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[560:567]*/, v[2:9] /*v[258:265]*/, v[210:217] /*v[466:473]*/, v[48:55] /*v[560:567]*/
	s_set_vgpr_msb 0xa544
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[194:201], v[202:209] /*v[458:465]*/, 0
	s_set_vgpr_msb 0x4445
	s_wait_loadcnt 0xe
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[18:25] /*v[274:281]*/, v[202:209] /*v[458:465]*/, 0
	s_set_vgpr_msb 0x45a4
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[178:185], v[186:193] /*v[442:449]*/, v[32:39] /*v[544:551]*/
	s_set_vgpr_msb 0xa4a5
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[10:17] /*v[266:273]*/, v[186:193] /*v[442:449]*/, v[40:47] /*v[552:559]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa54a
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[14:15] /*v[526:527]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[14:15] /*v[526:527]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[14:15] /*v[526:527]*/, v[38:39] /*v[550:551]*/
	s_set_vgpr_msb 0x4a54
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[202:209], v[170:177] /*v[426:433]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5449
	s_wait_loadcnt 0x7
	v_pk_add_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, v[22:23] /*v[534:535]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, v[22:23] /*v[534:535]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, v[22:23] /*v[534:535]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, s[12:13] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4955
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[26:33] /*v[282:289]*/, v[170:177] /*v[426:433]*/, v[218:225] /*v[474:481]*/
	v_pk_mul_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, s[12:13] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x5544
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[194:201], v[194:201] /*v[450:457]*/, 0
	s_set_vgpr_msb 0x4445
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[18:25] /*v[274:281]*/, v[194:201] /*v[450:457]*/, 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x454a
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[14:15] /*v[526:527]*/, v[32:33] /*v[544:545]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[14:15] /*v[526:527]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[14:15] /*v[526:527]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x4a54
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[178:185], v[178:185] /*v[434:441]*/, v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x544a
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[14:15] /*v[526:527]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4a49
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/, v[22:23] /*v[534:535]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[14:15] /*v[526:527]*/, v[40:41] /*v[552:553]*/
	s_set_vgpr_msb 0x4a49
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[22:23] /*v[534:535]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[22:23] /*v[534:535]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[22:23] /*v[534:535]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, s[12:13] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x49a5
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[560:567]*/, v[10:17] /*v[266:273]*/, v[178:185] /*v[434:441]*/, v[48:55] /*v[560:567]*/
	s_set_vgpr_msb 0xa549
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[22:23] /*v[534:535]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[254:255] /*v[510:511]*/, v[14:15] /*v[526:527]*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[250:251] /*v[506:507]*/, v[14:15] /*v[526:527]*/
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[210:217], v[154:161] /*v[410:417]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x544a
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[14:15] /*v[526:527]*/, v[48:49] /*v[560:561]*/
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[14:15] /*v[526:527]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[14:15] /*v[526:527]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[14:15] /*v[526:527]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x4a49
	s_wait_loadcnt 0x6
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[234:235] /*v[490:491]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[236:237] /*v[492:493]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4955
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[34:41] /*v[290:297]*/, v[154:161] /*v[410:417]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5549
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[238:239] /*v[494:495]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[240:241] /*v[496:497]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[252:253] /*v[508:509]*/, v[14:15] /*v[526:527]*/
	s_set_vgpr_msb 0x494a
	v_pk_mul_f32 v[184:185] /*v[440:441]*/, v[14:15] /*v[526:527]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x4a41
	v_exp_f32_e32 v154 /*v410*/, v194 /*v450*/
	v_exp_f32_e32 v155 /*v411*/, v195 /*v451*/
	v_exp_f32_e32 v156 /*v412*/, v196 /*v452*/
	s_set_vgpr_msb 0x4154
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[202:209], v[162:169] /*v[418:425]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5449
	v_exp_f32_e32 v157 /*v413*/, v197 /*v453*/
	v_exp_f32_e32 v158 /*v414*/, v198 /*v454*/
	v_exp_f32_e32 v159 /*v415*/, v199 /*v455*/
	v_exp_f32_e32 v160 /*v416*/, v200 /*v456*/
	v_exp_f32_e32 v161 /*v417*/, v201 /*v457*/
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4955
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[26:33] /*v[282:289]*/, v[162:169] /*v[418:425]*/, v[226:233] /*v[482:489]*/
	s_set_vgpr_msb 0x5549
	v_pk_add_f32 v[184:185] /*v[440:441]*/, v[184:185] /*v[440:441]*/, v[20:21] /*v[532:533]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, s[12:13] op_sel_hi:[1,0]
	v_nop
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[182:183] /*v[438:439]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[170:171] /*v[426:427]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[172:173] /*v[428:429]*/, s[12:13] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[226:233], v[138:145] /*v[394:401]*/, v[210:217] /*v[466:473]*/
	s_set_vgpr_msb 0x5455
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[174:175] /*v[430:431]*/, s[12:13] op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[176:177] /*v[432:433]*/, s[12:13] op_sel_hi:[1,0]
	v_exp_f32_e32 v174 /*v430*/, v192 /*v448*/
	v_exp_f32_e32 v175 /*v431*/, v193 /*v449*/
	v_exp_f32_e32 v176 /*v432*/, v186 /*v442*/
	v_exp_f32_e32 v177 /*v433*/, v187 /*v443*/
	v_exp_f32_e32 v182 /*v438*/, v188 /*v444*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[42:49] /*v[298:305]*/, v[138:145] /*v[394:401]*/, v[218:225] /*v[474:481]*/
	v_exp_f32_e32 v183 /*v439*/, v189 /*v445*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[184:185] /*v[440:441]*/, s[12:13] op_sel_hi:[1,0]
	v_exp_f32_e32 v178 /*v434*/, v178 /*v434*/
	v_exp_f32_e32 v179 /*v435*/, v179 /*v435*/
	v_exp_f32_e32 v180 /*v436*/, v180 /*v436*/
	v_exp_f32_e32 v181 /*v437*/, v181 /*v437*/
	v_exp_f32_e32 v162 /*v418*/, v162 /*v418*/
	s_set_vgpr_msb 0x5554
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[210:217], v[146:153] /*v[402:409]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5449
	s_wait_loadcnt 0x5
	v_pk_add_f32 v[138:139] /*v[394:395]*/, v[218:219] /*v[474:475]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[220:221] /*v[476:477]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[222:223] /*v[478:479]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[144:145] /*v[400:401]*/, v[224:225] /*v[480:481]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v163 /*v419*/, v163 /*v419*/
	s_set_vgpr_msb 0x4955
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[138:139] /*v[394:395]*/, v[154:155] /*v[410:411]*/
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, v[156:157] /*v[412:413]*/
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[34:41] /*v[290:297]*/, v[146:153] /*v[402:409]*/, v[226:233] /*v[482:489]*/
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, v[160:161] /*v[416:417]*/
	s_set_vgpr_msb 0x5546
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[14:15] /*v[526:527]*/, v[138:139] /*v[394:395]*/
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[14:15] /*v[526:527]*/, v[140:141] /*v[396:397]*/
	v_pk_add_f32 v[146:147] /*v[402:403]*/, v[18:19] /*v[530:531]*/, v[210:211] /*v[466:467]*/ op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[1,0]
	v_pk_add_f32 v[148:149] /*v[404:405]*/, v[18:19] /*v[530:531]*/, v[212:213] /*v[468:469]*/ op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[1,0]
	v_pk_add_f32 v[150:151] /*v[406:407]*/, v[18:19] /*v[530:531]*/, v[214:215] /*v[470:471]*/ op_sel_hi:[0,1] neg_lo:[1,0] neg_hi:[1,0]
	s_set_vgpr_msb 0x4654
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[226:233], v[130:137] /*v[386:393]*/, v[202:209] /*v[458:465]*/
	s_set_vgpr_msb 0x5446
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[14:15] /*v[526:527]*/, v[142:143] /*v[398:399]*/
	s_set_vgpr_msb 0x4645
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[146:147] /*v[402:403]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[148:149] /*v[404:405]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[150:151] /*v[406:407]*/, v[182:183] /*v[438:439]*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[14:15] /*v[526:527]*/, v[144:145] /*v[400:401]*/
	s_set_vgpr_msb 0x4645
	v_cvt_pk_bf16_f32 v138 /*v394*/, v138 /*v394*/, v139 /*v395*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[14:15] /*v[526:527]*/, v[146:147] /*v[402:403]*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[14:15] /*v[526:527]*/, v[148:149] /*v[404:405]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[14:15] /*v[526:527]*/, v[150:151] /*v[406:407]*/
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
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[202:203] /*v[458:459]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[204:205] /*v[460:461]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[206:207] /*v[462:463]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4955
	v_cvt_pk_bf16_f32 v144 /*v400*/, v158 /*v414*/, v159 /*v415*/
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[42:49] /*v[298:305]*/, v[130:137] /*v[386:393]*/, v[226:233] /*v[482:489]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[142:143] /*v[398:399]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[164:165] /*v[420:421]*/, v[180:181] /*v[436:437]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[162:163] /*v[418:419]*/
	v_cvt_pk_bf16_f32 v143 /*v399*/, v156 /*v412*/, v157 /*v413*/
	v_exp_f32_e32 v184 /*v440*/, v190 /*v446*/
	s_set_vgpr_msb 0x5546
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[14:15] /*v[526:527]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[14:15] /*v[526:527]*/, v[164:165] /*v[420:421]*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[14:15] /*v[526:527]*/, v[174:175] /*v[430:431]*/
	s_set_vgpr_msb 0x4645
	v_exp_f32_e32 v185 /*v441*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v142 /*v398*/, v154 /*v410*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v154 /*v410*/, v156 /*v412*/, v157 /*v413*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[216:217] /*v[472:473]*/, v[18:19] /*v[530:531]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
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
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[208:209] /*v[464:465]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[226:227] /*v[482:483]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[228:229] /*v[484:485]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[134:135] /*v[390:391]*/, v[230:231] /*v[486:487]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[136:137] /*v[392:393]*/, v[232:233] /*v[488:489]*/, v[16:17] /*v[528:529]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[184:185] /*v[440:441]*/
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, v[160:161] /*v[416:417]*/
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[164:165] /*v[420:421]*/
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[166:167] /*v[422:423]*/
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[134:135] /*v[390:391]*/, v[168:169] /*v[424:425]*/
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[136:137] /*v[392:393]*/, v[170:171] /*v[426:427]*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[14:15] /*v[526:527]*/, v[152:153] /*v[408:409]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[14:15] /*v[526:527]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[14:15] /*v[526:527]*/, v[130:131] /*v[386:387]*/
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[14:15] /*v[526:527]*/, v[132:133] /*v[388:389]*/
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[14:15] /*v[526:527]*/, v[134:135] /*v[390:391]*/
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[14:15] /*v[526:527]*/, v[136:137] /*v[392:393]*/
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
	v_add_nc_u32_e32 v162 /*v418*/, s41, v19 /*v531*/
	s_set_vgpr_msb 0x4888
	v_add_nc_u32_e32 v16 /*v528*/, s40, v21 /*v533*/
	s_set_vgpr_msb 0x8886
	ds_store_b128 v29 /*v541*/, v[150:153] /*v[406:409]*/
	ds_store_b128 v29 /*v541*/, v[142:145] /*v[398:401]*/ offset:32
	ds_store_b128 v29 /*v541*/, v[146:149] /*v[402:405]*/ offset:2560
	ds_store_b128 v29 /*v541*/, v[138:141] /*v[394:397]*/ offset:2592
	ds_store_b128 v27 /*v539*/, v[158:161] /*v[414:417]*/
	ds_store_b128 v27 /*v539*/, v[134:137] /*v[390:393]*/ offset:32
	ds_store_b128 v27 /*v539*/, v[154:157] /*v[410:413]*/ offset:2560
	ds_store_b128 v27 /*v539*/, v[130:133] /*v[386:389]*/ offset:2592
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v26 /*v538*/
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v26 /*v538*/ offset:1280
	s_set_vgpr_msb 0x8681
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v162 /*v418*/
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v162 /*v418*/ offset:32
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v162 /*v418*/ offset:4352
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v162 /*v418*/ offset:4384
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v162 /*v418*/ offset:64
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v162 /*v418*/ offset:96
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v162 /*v418*/ offset:4416
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v162 /*v418*/ offset:4448
	ds_load_tr16_b128 v[72:75] /*v[584:587]*/, v162 /*v418*/ offset:128
	ds_load_tr16_b128 v[80:83] /*v[592:595]*/, v162 /*v418*/ offset:160
	ds_load_tr16_b128 v[76:79] /*v[588:591]*/, v162 /*v418*/ offset:4480
	ds_load_tr16_b128 v[84:87] /*v[596:599]*/, v162 /*v418*/ offset:4512
	ds_load_tr16_b128 v[88:91] /*v[600:603]*/, v162 /*v418*/ offset:192
	ds_load_tr16_b128 v[96:99] /*v[608:611]*/, v162 /*v418*/ offset:224
	ds_load_tr16_b128 v[92:95] /*v[604:607]*/, v162 /*v418*/ offset:4544
	ds_load_tr16_b128 v[100:103] /*v[612:615]*/, v162 /*v418*/ offset:4576
	s_set_vgpr_msb 0x8182
	ds_load_tr16_b128 v[104:107] /*v[616:619]*/, v24 /*v536*/
	ds_load_tr16_b128 v[108:111] /*v[620:623]*/, v24 /*v536*/ offset:1280
	s_set_vgpr_msb 0x8281
	ds_load_tr16_b128 v[112:115] /*v[624:627]*/, v162 /*v418*/ offset:8704
	ds_load_tr16_b128 v[120:123] /*v[632:635]*/, v162 /*v418*/ offset:8736
	ds_load_tr16_b128 v[116:119] /*v[628:631]*/, v162 /*v418*/ offset:13056
	ds_load_tr16_b128 v[124:127] /*v[636:639]*/, v162 /*v418*/ offset:13088
	ds_load_tr16_b128 v[128:131] /*v[640:643]*/, v162 /*v418*/ offset:8768
	ds_load_tr16_b128 v[136:139] /*v[648:651]*/, v162 /*v418*/ offset:8800
	ds_load_tr16_b128 v[132:135] /*v[644:647]*/, v162 /*v418*/ offset:13120
	ds_load_tr16_b128 v[140:143] /*v[652:655]*/, v162 /*v418*/ offset:13152
	ds_load_tr16_b128 v[144:147] /*v[656:659]*/, v162 /*v418*/ offset:8832
	ds_load_tr16_b128 v[152:155] /*v[664:667]*/, v162 /*v418*/ offset:8864
	ds_load_tr16_b128 v[148:151] /*v[660:663]*/, v162 /*v418*/ offset:13184
	ds_load_tr16_b128 v[156:159] /*v[668:671]*/, v162 /*v418*/ offset:13216
	ds_load_tr16_b128 v[160:163] /*v[672:675]*/, v162 /*v418*/ offset:8896
	ds_load_tr16_b128 v[168:171] /*v[680:683]*/, v162 /*v418*/ offset:8928
	ds_load_tr16_b128 v[164:167] /*v[676:679]*/, v162 /*v418*/ offset:13248
	ds_load_tr16_b128 v[172:175] /*v[684:687]*/, v162 /*v418*/ offset:13280
	s_set_vgpr_msb 0x8182
	ds_load_tr16_b128 v[176:179] /*v[688:691]*/, v30 /*v542*/
	ds_load_tr16_b128 v[180:183] /*v[692:695]*/, v30 /*v542*/ offset:1280
	ds_load_tr16_b128 v[184:187] /*v[696:699]*/, v25 /*v537*/
	ds_load_tr16_b128 v[188:191] /*v[700:703]*/, v25 /*v537*/ offset:1280
	s_wait_tensorcnt 0x2
	s_set_vgpr_msb 0x825a
	ds_load_b128 v[250:253] /*v[506:509]*/, v16 /*v528*/ offset:8704
	ds_load_b128 v[254:257] /*v[510:513]*/, v16 /*v528*/ offset:8736
	ds_load_b128 v[234:237] /*v[490:493]*/, v16 /*v528*/ offset:8768
	ds_load_b128 v[238:241] /*v[494:497]*/, v16 /*v528*/ offset:8800
	ds_load_b128 v[218:221] /*v[474:477]*/, v16 /*v528*/ offset:8832
	ds_load_b128 v[222:225] /*v[478:481]*/, v16 /*v528*/ offset:8864
	ds_load_b128 v[186:189] /*v[442:445]*/, v16 /*v528*/ offset:8896
	ds_load_b128 v[190:193] /*v[446:449]*/, v16 /*v528*/ offset:8928
	ds_load_b128 v[202:205] /*v[458:461]*/, v16 /*v528*/
	ds_load_b128 v[206:209] /*v[462:465]*/, v16 /*v528*/ offset:32
	ds_load_b128 v[170:173] /*v[426:429]*/, v16 /*v528*/ offset:64
	ds_load_b128 v[174:177] /*v[430:433]*/, v16 /*v528*/ offset:96
	ds_load_b128 v[154:157] /*v[410:413]*/, v16 /*v528*/ offset:128
	ds_load_b128 v[158:161] /*v[414:417]*/, v16 /*v528*/ offset:160
	ds_load_b128 v[138:141] /*v[394:397]*/, v16 /*v528*/ offset:192
	ds_load_b128 v[142:145] /*v[398:401]*/, v16 /*v528*/ offset:224
	ds_load_b128 v[242:245] /*v[498:501]*/, v16 /*v528*/ offset:13056
	ds_load_b128 v[246:249] /*v[502:505]*/, v16 /*v528*/ offset:13088
	ds_load_b128 v[226:229] /*v[482:485]*/, v16 /*v528*/ offset:13120
	ds_load_b128 v[230:233] /*v[486:489]*/, v16 /*v528*/ offset:13152
	ds_load_b128 v[210:213] /*v[466:469]*/, v16 /*v528*/ offset:13184
	ds_load_b128 v[214:217] /*v[470:473]*/, v16 /*v528*/ offset:13216
	ds_load_b128 v[178:181] /*v[434:437]*/, v16 /*v528*/ offset:13248
	ds_load_b128 v[182:185] /*v[438:441]*/, v16 /*v528*/ offset:13280
	ds_load_b128 v[194:197] /*v[450:453]*/, v16 /*v528*/ offset:4352
	ds_load_b128 v[198:201] /*v[454:457]*/, v16 /*v528*/ offset:4384
	ds_load_b128 v[162:165] /*v[418:421]*/, v16 /*v528*/ offset:4416
	ds_load_b128 v[166:169] /*v[422:425]*/, v16 /*v528*/ offset:4448
	ds_load_b128 v[146:149] /*v[402:405]*/, v16 /*v528*/ offset:4480
	ds_load_b128 v[150:153] /*v[406:409]*/, v16 /*v528*/ offset:4512
	ds_load_b128 v[130:133] /*v[386:389]*/, v16 /*v528*/ offset:4544
	ds_load_b128 v[134:137] /*v[390:393]*/, v16 /*v528*/ offset:4576
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[32:39] /*v[544:551]*/, v[40:47] /*v[552:559]*/, v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[32:39] /*v[544:551]*/, v[48:55] /*v[560:567]*/, v[58:65] /*v[314:321]*/ matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[32:39] /*v[544:551]*/, v[56:63] /*v[568:575]*/, v[50:57] /*v[306:313]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[250:257], v[32:39] /*v[544:551]*/, v[64:71] /*v[576:583]*/, v[250:257] matrix_a_reuse
	s_wait_dscnt 0x3b
	v_wmma_f32_16x16x32_bf16 v[186:193], v[32:39] /*v[544:551]*/, v[72:79] /*v[584:591]*/, v[186:193] matrix_a_reuse
	s_wait_dscnt 0x3a
	v_wmma_f32_16x16x32_bf16 v[146:153], v[32:39] /*v[544:551]*/, v[80:87] /*v[592:599]*/, v[146:153] matrix_a_reuse
	s_wait_dscnt 0x37
	v_wmma_f32_16x16x32_bf16 v[138:145], v[32:39] /*v[544:551]*/, v[88:95] /*v[600:607]*/, v[138:145] matrix_a_reuse
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[122:129], v[32:39] /*v[544:551]*/, v[96:103] /*v[608:615]*/, v[122:129] matrix_a_reuse
	s_set_vgpr_msb 0xa5a
	s_wait_dscnt 0x31
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[104:111] /*v[616:623]*/, v[112:119] /*v[624:631]*/, v[122:129] /*v[378:385]*/
	s_wait_dscnt 0x30
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[104:111] /*v[616:623]*/, v[120:127] /*v[632:639]*/, v[114:121] /*v[370:377]*/ matrix_a_reuse
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[104:111] /*v[616:623]*/, v[128:135] /*v[640:647]*/, v[106:113] /*v[362:369]*/ matrix_a_reuse
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[104:111] /*v[616:623]*/, v[136:143] /*v[648:655]*/, v[98:105] /*v[354:361]*/ matrix_a_reuse
	s_wait_dscnt 0x29
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[104:111] /*v[616:623]*/, v[144:151] /*v[656:663]*/, v[90:97] /*v[346:353]*/ matrix_a_reuse
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[104:111] /*v[616:623]*/, v[152:159] /*v[664:671]*/, v[82:89] /*v[338:345]*/ matrix_a_reuse
	s_wait_dscnt 0x25
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[104:111] /*v[616:623]*/, v[160:167] /*v[672:679]*/, v[66:73] /*v[322:329]*/ matrix_a_reuse
	s_set_vgpr_msb 0x5a0a
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[218:225], v[104:111] /*v[616:623]*/, v[168:175] /*v[680:687]*/, v[218:225] matrix_a_reuse
	s_wait_dscnt 0x22
	v_wmma_f32_16x16x32_bf16 v[82:89], v[176:183] /*v[688:695]*/, v[40:47] /*v[552:559]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[176:183] /*v[688:695]*/, v[48:55] /*v[560:567]*/, v[58:65] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[50:57], v[176:183] /*v[688:695]*/, v[56:63] /*v[568:575]*/, v[50:57] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[34:41], v[176:183] /*v[688:695]*/, v[64:71] /*v[576:583]*/, v[34:41] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[26:33], v[176:183] /*v[688:695]*/, v[72:79] /*v[584:591]*/, v[26:33] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[18:25], v[176:183] /*v[688:695]*/, v[80:87] /*v[592:599]*/, v[18:25] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[10:17], v[176:183] /*v[688:695]*/, v[88:95] /*v[600:607]*/, v[10:17] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[2:9], v[176:183] /*v[688:695]*/, v[96:103] /*v[608:615]*/, v[2:9] matrix_a_reuse
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[130:137], v[184:191] /*v[696:703]*/, v[112:119] /*v[624:631]*/, v[130:137]
	v_wmma_f32_16x16x32_bf16 v[114:121], v[184:191] /*v[696:703]*/, v[120:127] /*v[632:639]*/, v[114:121] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[106:113], v[184:191] /*v[696:703]*/, v[128:135] /*v[640:647]*/, v[106:113] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[98:105], v[184:191] /*v[696:703]*/, v[136:143] /*v[648:655]*/, v[98:105] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[90:97], v[184:191] /*v[696:703]*/, v[144:151] /*v[656:663]*/, v[90:97] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[74:81], v[184:191] /*v[696:703]*/, v[152:159] /*v[664:671]*/, v[74:81] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[66:73], v[184:191] /*v[696:703]*/, v[160:167] /*v[672:679]*/, v[66:73] matrix_a_reuse
	v_wmma_f32_16x16x32_bf16 v[42:49], v[184:191] /*v[696:703]*/, v[168:175] /*v[680:687]*/, v[42:49] matrix_a_reuse
	s_add_co_u32 s18, s18, 1
	s_add_co_ci_u32 s19, s19, 0
	s_cselect_b32 s6, -1, 0
	s_add_co_i32 s15, s15, 1
	s_and_b32 s6, s6, exec_lo
	s_set_vgpr_msb 0xa82
	s_wait_loadcnt 0x1
	v_dual_mov_b32 v16 /*v528*/, v193 /*v705*/ :: v_dual_mov_b32 v20 /*v532*/, v194 /*v706*/
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v18 /*v530*/, v192 /*v704*/ :: v_dual_mov_b32 v22 /*v534*/, v31 /*v543*/
	s_cselect_b32 s6, 1, 0
	s_mov_b32 s7, s3
	s_cmp_lg_u32 s6, 1
	s_mov_b32 s6, s27
	s_mov_b32 s41, s40
	s_set_vgpr_msb 0x8200
	s_cbranch_scc1 .LBB0_8
.LBB0_9:
	s_set_vgpr_msb 8
	s_wait_loadcnt 0x23
	v_or_b32_e32 v154, 1, v23 /*v535*/
	s_clause 0x1
	s_load_b64 s[4:5], s[0:1], 0x110 nv
	s_load_b64 s[8:9], s[0:1], 0x150 nv
	s_wait_xcnt 0x0
	s_mul_i32 s0, s24, s49
	v_mul_lo_u32 v155, s16, v23 /*v535*/
	s_wait_loadcnt 0x22
	v_mul_lo_u32 v161, s16, v10 /*v522*/
	v_mul_lo_u32 v154, v154, s16
	s_wait_loadcnt 0x20
	v_mul_lo_u32 v166, s16, v7 /*v519*/
	s_add_co_i32 s0, s0, s33
	v_mul_lo_u32 v156, s16, v12 /*v524*/
	s_mul_i32 s0, s51, s0
	v_mul_lo_u32 v159, s16, v11 /*v523*/
	v_mul_lo_u32 v162, s16, v9 /*v521*/
	v_mul_lo_u32 v164, s16, v8 /*v520*/
	s_add_co_i32 s0, s0, s48
	s_mul_i32 s60, s60, s50
	v_add_lshl_u32 v155, v155, s0, 7
	v_add_lshl_u32 v154, v154, s0, 7
	v_add_lshl_u32 v161, v161, s0, 7
	v_add_lshl_u32 v166, v166, s0, 7
	v_add_lshl_u32 v156, v156, s0, 7
	v_or_b32_e32 v157, v155, v13 /*v525*/
	v_or_b32_e32 v158, v154, v13 /*v525*/
	v_add_lshl_u32 v159, v159, s0, 7
	v_or_b32_e32 v165, v161, v13 /*v525*/
	v_add_lshl_u32 v162, v162, s0, 7
	v_add_lshl_u32 v164, v164, s0, 7
	v_or_b32_e32 v169, v166, v13 /*v525*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v154, v154, v0
	v_or_b32_e32 v155, v155, v0
	v_or_b32_e32 v166, v166, v0
	v_or_b32_e32 v161, v161, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v160, v156, v13 /*v525*/
	v_or_b32_e32 v163, v159, v13 /*v525*/
	v_or_b32_e32 v167, v162, v13 /*v525*/
	v_or_b32_e32 v168, v164, v13 /*v525*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v154, 2, v154 :: v_dual_lshlrev_b32 v155, 2, v155
	v_dual_lshlrev_b32 v166, 2, v166 :: v_dual_bitop2_b32 v159, v159, v0 bitop3:0x54
	v_dual_lshlrev_b32 v161, 2, v161 :: v_dual_bitop2_b32 v156, v156, v0 bitop3:0x54
	s_lshl_b32 s2, s60, 9
	v_dual_lshlrev_b32 v157, 2, v157 :: v_dual_lshlrev_b32 v158, 2, v158
	s_ashr_i32 s3, s2, 31
	v_dual_lshlrev_b32 v160, 2, v160 :: v_dual_lshlrev_b32 v163, 2, v163
	v_dual_lshlrev_b32 v165, 2, v165 :: v_dual_lshlrev_b32 v167, 2, v167
	v_dual_lshlrev_b32 v168, 2, v168 :: v_dual_lshlrev_b32 v169, 2, v169
	v_lshlrev_b32_e32 v159, 2, v159
	s_wait_loadcnt 0x1f
	v_dual_lshlrev_b32 v156, 2, v156 :: v_dual_bitop2_b32 v170, 64, v155 bitop3:0x54
	s_lshr_b64 s[6:7], s[2:3], 7
	v_or_b32_e32 v162, v162, v0
	v_or_b32_e32 v164, v164, v0
	v_or_b32_e32 v171, 64, v154
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 64
	s_wait_kmcnt 0x0
	buffer_store_b32 v74 /*v330*/, v157, s[4:7], null offen
	buffer_store_b32 v122 /*v378*/, v157, s[8:11], null offen
	buffer_store_b32 v75 /*v331*/, v158, s[4:7], null offen
	buffer_store_b32 v123 /*v379*/, v158, s[8:11], null offen
	buffer_store_b32 v76 /*v332*/, v160, s[4:7], null offen
	buffer_store_b32 v124 /*v380*/, v160, s[8:11], null offen
	buffer_store_b32 v77 /*v333*/, v163, s[4:7], null offen
	buffer_store_b32 v125 /*v381*/, v163, s[8:11], null offen
	buffer_store_b32 v78 /*v334*/, v165, s[4:7], null offen
	buffer_store_b32 v126 /*v382*/, v165, s[8:11], null offen
	buffer_store_b32 v79 /*v335*/, v167, s[4:7], null offen
	buffer_store_b32 v127 /*v383*/, v167, s[8:11], null offen
	buffer_store_b32 v80 /*v336*/, v168, s[4:7], null offen
	buffer_store_b32 v128 /*v384*/, v168, s[8:11], null offen
	buffer_store_b32 v81 /*v337*/, v169, s[4:7], null offen
	buffer_store_b32 v129 /*v385*/, v169, s[8:11], null offen
	buffer_store_b32 v58 /*v314*/, v170, s[4:7], null offen
	buffer_store_b32 v114 /*v370*/, v170, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	v_dual_lshlrev_b32 v162, 2, v162 :: v_dual_bitop2_b32 v170, 64, v156 bitop3:0x54
	v_lshlrev_b32_e32 v164, 2, v164
	s_set_vgpr_msb 64
	buffer_store_b32 v59 /*v315*/, v171, s[4:7], null offen
	buffer_store_b32 v115 /*v371*/, v171, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v171, 64, v159
	s_set_vgpr_msb 64
	buffer_store_b32 v60 /*v316*/, v170, s[4:7], null offen
	buffer_store_b32 v116 /*v372*/, v170, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v170, 64, v161
	v_or_b32_e32 v172, 64, v164
	s_set_vgpr_msb 64
	buffer_store_b32 v61 /*v317*/, v171, s[4:7], null offen
	buffer_store_b32 v117 /*v373*/, v171, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v171, 64, v162
	s_set_vgpr_msb 64
	buffer_store_b32 v62 /*v318*/, v170, s[4:7], null offen
	buffer_store_b32 v118 /*v374*/, v170, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v170, 64, v166
	s_set_vgpr_msb 64
	buffer_store_b32 v63 /*v319*/, v171, s[4:7], null offen
	buffer_store_b32 v119 /*v375*/, v171, s[8:11], null offen
	buffer_store_b32 v64 /*v320*/, v172, s[4:7], null offen
	buffer_store_b32 v120 /*v376*/, v172, s[8:11], null offen
	buffer_store_b32 v65 /*v321*/, v170, s[4:7], null offen
	buffer_store_b32 v121 /*v377*/, v170, s[8:11], null offen
	buffer_store_b32 v50 /*v306*/, v157, s[4:7], null offen offset:128
	buffer_store_b32 v106 /*v362*/, v157, s[8:11], null offen offset:128
	buffer_store_b32 v51 /*v307*/, v158, s[4:7], null offen offset:128
	buffer_store_b32 v107 /*v363*/, v158, s[8:11], null offen offset:128
	buffer_store_b32 v52 /*v308*/, v160, s[4:7], null offen offset:128
	buffer_store_b32 v108 /*v364*/, v160, s[8:11], null offen offset:128
	buffer_store_b32 v53 /*v309*/, v163, s[4:7], null offen offset:128
	buffer_store_b32 v109 /*v365*/, v163, s[8:11], null offen offset:128
	buffer_store_b32 v54 /*v310*/, v165, s[4:7], null offen offset:128
	buffer_store_b32 v110 /*v366*/, v165, s[8:11], null offen offset:128
	buffer_store_b32 v55 /*v311*/, v167, s[4:7], null offen offset:128
	buffer_store_b32 v111 /*v367*/, v167, s[8:11], null offen offset:128
	buffer_store_b32 v56 /*v312*/, v168, s[4:7], null offen offset:128
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v170, 0xc0, v155
	v_or_b32_e32 v171, 0xc0, v154
	v_or_b32_e32 v172, 0xc0, v156
	s_set_vgpr_msb 64
	buffer_store_b32 v112 /*v368*/, v168, s[8:11], null offen offset:128
	buffer_store_b32 v57 /*v313*/, v169, s[4:7], null offen offset:128
	buffer_store_b32 v113 /*v369*/, v169, s[8:11], null offen offset:128
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v250, v170, s[4:7], null offen
	s_set_vgpr_msb 64
	buffer_store_b32 v98 /*v354*/, v170, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v251, v171, s[4:7], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v170, 0xc0, v159
	s_set_vgpr_msb 64
	buffer_store_b32 v99 /*v355*/, v171, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v252, v172, s[4:7], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v171, 0xc0, v161
	s_set_vgpr_msb 64
	buffer_store_b32 v100 /*v356*/, v172, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v253, v170, s[4:7], null offen
	s_set_vgpr_msb 64
	buffer_store_b32 v101 /*v357*/, v170, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v170, 0xc0, v162
	v_or_b32_e32 v172, 0xc0, v164
	buffer_store_b32 v254, v171, s[4:7], null offen
	s_set_vgpr_msb 64
	buffer_store_b32 v102 /*v358*/, v171, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v171, 0xc0, v166
	buffer_store_b32 v255, v170, s[4:7], null offen
	s_set_vgpr_msb 64
	buffer_store_b32 v103 /*v359*/, v170, s[8:11], null offen
	buffer_store_b32 v0 /*v256*/, v172, s[4:7], null offen
	buffer_store_b32 v104 /*v360*/, v172, s[8:11], null offen
	buffer_store_b32 v1 /*v257*/, v171, s[4:7], null offen
	buffer_store_b32 v105 /*v361*/, v171, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v186, v157, s[4:7], null offen offset:256
	s_set_vgpr_msb 64
	buffer_store_b32 v90 /*v346*/, v157, s[8:11], null offen offset:256
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v187, v158, s[4:7], null offen offset:256
	s_set_vgpr_msb 64
	buffer_store_b32 v91 /*v347*/, v158, s[8:11], null offen offset:256
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v188, v160, s[4:7], null offen offset:256
	s_set_vgpr_msb 64
	buffer_store_b32 v92 /*v348*/, v160, s[8:11], null offen offset:256
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v189, v163, s[4:7], null offen offset:256
	s_set_vgpr_msb 64
	buffer_store_b32 v93 /*v349*/, v163, s[8:11], null offen offset:256
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v190, v165, s[4:7], null offen offset:256
	s_set_vgpr_msb 64
	buffer_store_b32 v94 /*v350*/, v165, s[8:11], null offen offset:256
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v191, v167, s[4:7], null offen offset:256
	s_set_vgpr_msb 64
	buffer_store_b32 v95 /*v351*/, v167, s[8:11], null offen offset:256
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v192, v168, s[4:7], null offen offset:256
	s_wait_xcnt 0x11
	v_or_b32_e32 v170, 0x140, v155
	s_wait_xcnt 0xd
	v_or_b32_e32 v171, 0x140, v154
	s_set_vgpr_msb 64
	buffer_store_b32 v96 /*v352*/, v168, s[8:11], null offen offset:256
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v193, v169, s[4:7], null offen offset:256
	s_set_vgpr_msb 64
	buffer_store_b32 v97 /*v353*/, v169, s[8:11], null offen offset:256
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v146, v170, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v146, 0x140, v156
	s_set_vgpr_msb 64
	buffer_store_b32 v82 /*v338*/, v170, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v147, v171, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v147, 0x140, v159
	s_set_vgpr_msb 64
	buffer_store_b32 v83 /*v339*/, v171, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v148, v146, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v148, 0x140, v161
	s_set_vgpr_msb 64
	buffer_store_b32 v84 /*v340*/, v146, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v149, v147, s[4:7], null offen
	s_set_vgpr_msb 64
	buffer_store_b32 v85 /*v341*/, v147, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v146, 0x140, v162
	v_or_b32_e32 v147, 0x140, v164
	v_mul_lo_u32 v1, s16, v1
	buffer_store_b32 v150, v148, s[4:7], null offen
	s_set_vgpr_msb 64
	buffer_store_b32 v86 /*v342*/, v148, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v148, 0x140, v166
	buffer_store_b32 v151, v146, s[4:7], null offen
	s_set_vgpr_msb 64
	buffer_store_b32 v87 /*v343*/, v146, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v152, v147, s[4:7], null offen
	s_set_vgpr_msb 64
	buffer_store_b32 v88 /*v344*/, v147, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v153, v148, s[4:7], null offen
	s_set_vgpr_msb 64
	buffer_store_b32 v89 /*v345*/, v148, s[8:11], null offen
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v138, v157, s[4:7], null offen offset:384
	s_set_vgpr_msb 64
	buffer_store_b32 v66 /*v322*/, v157, s[8:11], null offen offset:384
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v139, v158, s[4:7], null offen offset:384
	s_set_vgpr_msb 64
	buffer_store_b32 v67 /*v323*/, v158, s[8:11], null offen offset:384
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v140, v160, s[4:7], null offen offset:384
	s_set_vgpr_msb 64
	buffer_store_b32 v68 /*v324*/, v160, s[8:11], null offen offset:384
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v141, v163, s[4:7], null offen offset:384
	s_set_vgpr_msb 64
	buffer_store_b32 v69 /*v325*/, v163, s[8:11], null offen offset:384
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v142, v165, s[4:7], null offen offset:384
	s_set_vgpr_msb 64
	buffer_store_b32 v70 /*v326*/, v165, s[8:11], null offen offset:384
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v143, v167, s[4:7], null offen offset:384
	s_set_vgpr_msb 64
	buffer_store_b32 v71 /*v327*/, v167, s[8:11], null offen offset:384
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v144, v168, s[4:7], null offen offset:384
	s_wait_xcnt 0xc
	v_or_b32_e32 v138, 0x1c0, v155
	s_wait_xcnt 0xa
	v_or_b32_e32 v139, 0x1c0, v154
	s_set_vgpr_msb 64
	buffer_store_b32 v72 /*v328*/, v168, s[8:11], null offen offset:384
	s_set_vgpr_msb 0x4000
	buffer_store_b32 v145, v169, s[4:7], null offen offset:384
	s_set_vgpr_msb 64
	buffer_store_b32 v73 /*v329*/, v169, s[8:11], null offen offset:384
	s_set_vgpr_msb 0x4000
	v_add_lshl_u32 v1, s0, v1, 7
	buffer_store_b32 v122, v138, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v122, 0x1c0, v156
	buffer_store_b32 v218, v138, s[8:11], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v138, 0x1c0, v159
	buffer_store_b32 v123, v139, s[4:7], null offen
	buffer_store_b32 v219, v139, s[8:11], null offen
	buffer_store_b32 v124, v122, s[4:7], null offen
	buffer_store_b32 v220, v122, s[8:11], null offen
	buffer_store_b32 v125, v138, s[4:7], null offen
	s_set_vgpr_msb 8
	v_or_b32_e32 v122, 1, v17 /*v529*/
	v_mul_lo_u32 v125, s16, v17 /*v529*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v123, 0x1c0, v161
	v_or_b32_e32 v124, 0x1c0, v162
	buffer_store_b32 v221, v138, s[8:11], null offen
	v_mul_lo_u32 v122, s16, v122
	buffer_store_b32 v126, v123, s[4:7], null offen
	buffer_store_b32 v222, v123, s[8:11], null offen
	buffer_store_b32 v127, v124, s[4:7], null offen
	buffer_store_b32 v223, v124, s[8:11], null offen
	s_set_vgpr_msb 8
	v_mul_lo_u32 v124, s16, v6 /*v518*/
	v_add_lshl_u32 v125, v125, s0, 7
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v123, 0x1c0, v164
	buffer_store_b32 v128, v123, s[4:7], null offen
	buffer_store_b32 v224, v123, s[8:11], null offen
	v_add_lshl_u32 v122, s0, v122, 7
	s_set_vgpr_msb 8
	v_or_b32_e32 v126, v125, v13 /*v525*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v123, 0x1c0, v166
	v_add_lshl_u32 v124, s0, v124, 7
	s_set_vgpr_msb 8
	v_mul_lo_u32 v128, s16, v5 /*v517*/
	v_or_b32_e32 v127, v122, v13 /*v525*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v126, 2, v126
	buffer_store_b32 v129, v123, s[4:7], null offen
	buffer_store_b32 v225, v123, s[8:11], null offen
	s_set_vgpr_msb 8
	v_mul_lo_u32 v129, s16, v4 /*v516*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v123, 2, v127
	s_set_vgpr_msb 8
	v_or_b32_e32 v127, v124, v13 /*v525*/
	buffer_store_b32 v82, v126, s[4:7], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v82, v128, s0, 7
	buffer_store_b32 v130, v126, s[8:11], null offen
	buffer_store_b32 v83, v123, s[4:7], null offen
	s_wait_xcnt 0x1
	v_mul_lo_u32 v130, s16, v2 /*v514*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v83, 2, v127
	s_set_vgpr_msb 8
	v_mul_lo_u32 v127, s16, v3 /*v515*/
	v_or_b32_e32 v128, v82, v13 /*v525*/
	v_add_lshl_u32 v129, v129, s0, 7
	buffer_store_b32 v131, v123, s[8:11], null offen
	buffer_store_b32 v84, v83, s[4:7], null offen
	buffer_store_b32 v132, v83, s[8:11], null offen
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v122, v122, v0
	v_lshlrev_b32_e32 v84, 2, v128
	s_set_vgpr_msb 8
	v_or_b32_e32 v128, v129, v13 /*v525*/
	v_add_lshl_u32 v127, v127, s0, 7
	v_add_lshl_u32 v130, v130, s0, 7
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v82, v82, v0
	buffer_store_b32 v85, v84, s[4:7], null offen
	v_lshlrev_b32_e32 v128, 2, v128
	s_set_vgpr_msb 8
	v_or_b32_e32 v85, v127, v13 /*v525*/
	v_or_b32_e32 v131, v130, v13 /*v525*/
	buffer_store_b32 v133, v84, s[8:11], null offen
	buffer_store_b32 v86, v128, s[4:7], null offen
	buffer_store_b32 v134, v128, s[8:11], null offen
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v85, 2, v85 :: v_dual_lshlrev_b32 v86, 2, v131
	s_set_vgpr_msb 8
	v_or_b32_e32 v131, v1, v13 /*v525*/
	s_set_vgpr_msb 0x800
	v_dual_lshlrev_b32 v82, 2, v82 :: v_dual_bitop2_b32 v124, v124, v0 bitop3:0x54
	buffer_store_b32 v87, v85, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v87, v125, v0
	buffer_store_b32 v135, v85, s[8:11], null offen
	buffer_store_b32 v88, v86, s[4:7], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v88, 2, v131
	buffer_store_b32 v136, v86, s[8:11], null offen
	v_lshlrev_b32_e32 v124, 2, v124
	buffer_store_b32 v89, v88, s[4:7], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v89, 2, v122 :: v_dual_lshlrev_b32 v87, 2, v87
	buffer_store_b32 v137, v88, s[8:11], null offen
	v_or_b32_e32 v125, 64, v89
	v_or_b32_e32 v122, 64, v87
	buffer_store_b32 v58, v122, s[4:7], null offen
	buffer_store_b32 v114, v122, s[8:11], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v114, v130, v0
	v_or_b32_e32 v58, 64, v124
	buffer_store_b32 v59, v125, s[4:7], null offen
	buffer_store_b32 v115, v125, s[8:11], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v59, v129, v0
	buffer_store_b32 v60, v58, s[4:7], null offen
	buffer_store_b32 v116, v58, s[8:11], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v59, 2, v59 :: v_dual_bitop2_b32 v58, v127, v0 bitop3:0x54
	v_or_b32_e32 v0, v1, v0
	v_or_b32_e32 v60, 64, v82
	v_dual_lshlrev_b32 v1, 2, v58 :: v_dual_bitop2_b32 v58, 64, v59 bitop3:0x54
	v_lshlrev_b32_e32 v0, 2, v0
	buffer_store_b32 v61, v60, s[4:7], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v61, 2, v114
	buffer_store_b32 v117, v60, s[8:11], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v60, 64, v1
	buffer_store_b32 v62, v58, s[4:7], null offen
	buffer_store_b32 v118, v58, s[8:11], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v62, 64, v61
	s_wait_xcnt 0x0
	v_or_b32_e32 v58, 64, v0
	buffer_store_b32 v63, v60, s[4:7], null offen
	buffer_store_b32 v119, v60, s[8:11], null offen
	buffer_store_b32 v64, v62, s[4:7], null offen
	buffer_store_b32 v120, v62, s[8:11], null offen
	buffer_store_b32 v65, v58, s[4:7], null offen
	buffer_store_b32 v121, v58, s[8:11], null offen
	buffer_store_b32 v50, v126, s[4:7], null offen offset:128
	buffer_store_b32 v106, v126, s[8:11], null offen offset:128
	buffer_store_b32 v51, v123, s[4:7], null offen offset:128
	buffer_store_b32 v107, v123, s[8:11], null offen offset:128
	buffer_store_b32 v52, v83, s[4:7], null offen offset:128
	buffer_store_b32 v108, v83, s[8:11], null offen offset:128
	buffer_store_b32 v53, v84, s[4:7], null offen offset:128
	buffer_store_b32 v109, v84, s[8:11], null offen offset:128
	buffer_store_b32 v54, v128, s[4:7], null offen offset:128
	buffer_store_b32 v110, v128, s[8:11], null offen offset:128
	buffer_store_b32 v55, v85, s[4:7], null offen offset:128
	buffer_store_b32 v111, v85, s[8:11], null offen offset:128
	buffer_store_b32 v56, v86, s[4:7], null offen offset:128
	s_wait_xcnt 0xc
	v_or_b32_e32 v50, 0xc0, v87
	s_wait_xcnt 0xa
	v_or_b32_e32 v51, 0xc0, v89
	buffer_store_b32 v112, v86, s[8:11], null offen offset:128
	buffer_store_b32 v57, v88, s[4:7], null offen offset:128
	buffer_store_b32 v113, v88, s[8:11], null offen offset:128
	buffer_store_b32 v34, v50, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v34, 0xc0, v124
	buffer_store_b32 v98, v50, s[8:11], null offen
	buffer_store_b32 v35, v51, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v35, 0xc0, v82
	buffer_store_b32 v99, v51, s[8:11], null offen
	buffer_store_b32 v36, v34, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v36, 0xc0, v59
	buffer_store_b32 v100, v34, s[8:11], null offen
	buffer_store_b32 v37, v35, s[4:7], null offen
	buffer_store_b32 v101, v35, s[8:11], null offen
	s_wait_xcnt 0x2
	v_or_b32_e32 v34, 0xc0, v1
	s_wait_xcnt 0x0
	v_or_b32_e32 v35, 0xc0, v61
	buffer_store_b32 v38, v36, s[4:7], null offen
	buffer_store_b32 v102, v36, s[8:11], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v36, 0xc0, v0
	buffer_store_b32 v39, v34, s[4:7], null offen
	buffer_store_b32 v103, v34, s[8:11], null offen
	buffer_store_b32 v40, v35, s[4:7], null offen
	buffer_store_b32 v104, v35, s[8:11], null offen
	buffer_store_b32 v41, v36, s[4:7], null offen
	buffer_store_b32 v105, v36, s[8:11], null offen
	buffer_store_b32 v26, v126, s[4:7], null offen offset:256
	buffer_store_b32 v90, v126, s[8:11], null offen offset:256
	buffer_store_b32 v27, v123, s[4:7], null offen offset:256
	buffer_store_b32 v91, v123, s[8:11], null offen offset:256
	buffer_store_b32 v28, v83, s[4:7], null offen offset:256
	buffer_store_b32 v92, v83, s[8:11], null offen offset:256
	buffer_store_b32 v29, v84, s[4:7], null offen offset:256
	buffer_store_b32 v93, v84, s[8:11], null offen offset:256
	buffer_store_b32 v30, v128, s[4:7], null offen offset:256
	buffer_store_b32 v94, v128, s[8:11], null offen offset:256
	buffer_store_b32 v31, v85, s[4:7], null offen offset:256
	buffer_store_b32 v95, v85, s[8:11], null offen offset:256
	buffer_store_b32 v32, v86, s[4:7], null offen offset:256
	s_wait_xcnt 0xc
	v_or_b32_e32 v26, 0x140, v87
	s_wait_xcnt 0xa
	v_or_b32_e32 v27, 0x140, v89
	buffer_store_b32 v96, v86, s[8:11], null offen offset:256
	buffer_store_b32 v33, v88, s[4:7], null offen offset:256
	buffer_store_b32 v97, v88, s[8:11], null offen offset:256
	buffer_store_b32 v18, v26, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v18, 0x140, v124
	buffer_store_b32 v74, v26, s[8:11], null offen
	buffer_store_b32 v19, v27, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v19, 0x140, v82
	buffer_store_b32 v75, v27, s[8:11], null offen
	buffer_store_b32 v20, v18, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v20, 0x140, v59
	buffer_store_b32 v76, v18, s[8:11], null offen
	buffer_store_b32 v21, v19, s[4:7], null offen
	buffer_store_b32 v77, v19, s[8:11], null offen
	s_wait_xcnt 0x2
	v_or_b32_e32 v18, 0x140, v1
	s_wait_xcnt 0x0
	v_or_b32_e32 v19, 0x140, v61
	v_or_b32_e32 v1, 0x1c0, v1
	buffer_store_b32 v22, v20, s[4:7], null offen
	buffer_store_b32 v78, v20, s[8:11], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v20, 0x140, v0
	buffer_store_b32 v23, v18, s[4:7], null offen
	buffer_store_b32 v79, v18, s[8:11], null offen
	buffer_store_b32 v24, v19, s[4:7], null offen
	buffer_store_b32 v80, v19, s[8:11], null offen
	buffer_store_b32 v25, v20, s[4:7], null offen
	buffer_store_b32 v81, v20, s[8:11], null offen
	buffer_store_b32 v10, v126, s[4:7], null offen offset:384
	buffer_store_b32 v66, v126, s[8:11], null offen offset:384
	buffer_store_b32 v11, v123, s[4:7], null offen offset:384
	buffer_store_b32 v67, v123, s[8:11], null offen offset:384
	buffer_store_b32 v12, v83, s[4:7], null offen offset:384
	buffer_store_b32 v68, v83, s[8:11], null offen offset:384
	buffer_store_b32 v13, v84, s[4:7], null offen offset:384
	buffer_store_b32 v69, v84, s[8:11], null offen offset:384
	buffer_store_b32 v14, v128, s[4:7], null offen offset:384
	buffer_store_b32 v70, v128, s[8:11], null offen offset:384
	buffer_store_b32 v15, v85, s[4:7], null offen offset:384
	buffer_store_b32 v71, v85, s[8:11], null offen offset:384
	buffer_store_b32 v16, v86, s[4:7], null offen offset:384
	s_wait_xcnt 0xc
	v_or_b32_e32 v10, 0x1c0, v87
	s_wait_xcnt 0xa
	v_or_b32_e32 v11, 0x1c0, v89
	buffer_store_b32 v72, v86, s[8:11], null offen offset:384
	buffer_store_b32 v17, v88, s[4:7], null offen offset:384
	buffer_store_b32 v73, v88, s[8:11], null offen offset:384
	buffer_store_b32 v2, v10, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v2, 0x1c0, v124
	buffer_store_b32 v42, v10, s[8:11], null offen
	buffer_store_b32 v3, v11, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v82
	buffer_store_b32 v43, v11, s[8:11], null offen
	buffer_store_b32 v4, v2, s[4:7], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v4, 0x1c0, v59
	buffer_store_b32 v44, v2, s[8:11], null offen
	buffer_store_b32 v5, v3, s[4:7], null offen
	buffer_store_b32 v45, v3, s[8:11], null offen
	s_wait_xcnt 0x2
	v_or_b32_e32 v2, 0x1c0, v61
	v_or_b32_e32 v0, 0x1c0, v0
	buffer_store_b32 v6, v4, s[4:7], null offen
	buffer_store_b32 v46, v4, s[8:11], null offen
	buffer_store_b32 v7, v1, s[4:7], null offen
	buffer_store_b32 v47, v1, s[8:11], null offen
	buffer_store_b32 v8, v2, s[4:7], null offen
	buffer_store_b32 v48, v2, s[8:11], null offen
	buffer_store_b32 v9, v0, s[4:7], null offen
	buffer_store_b32 v49, v0, s[8:11], null offen
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	k_dkdv_sp_0, .Lfunc_end0-k_dkdv_sp_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dkdv_sp_0
		.amdhsa_group_segment_fixed_size 70656
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 440
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
		.amdhsa_next_free_vgpr 707
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-k_dkdv_sp_0)<<4)&4080)>>4
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

	.set .Lk_dkdv_sp_0.num_vgpr, 707
	.set .Lk_dkdv_sp_0.num_agpr, 0
	.set .Lk_dkdv_sp_0.numbered_sgpr, 76
	.set .Lk_dkdv_sp_0.num_named_barrier, 0
	.set .Lk_dkdv_sp_0.private_seg_size, 0
	.set .Lk_dkdv_sp_0.uses_vcc, 1
	.set .Lk_dkdv_sp_0.uses_flat_scratch, 0
	.set .Lk_dkdv_sp_0.has_dyn_sized_stack, 0
	.set .Lk_dkdv_sp_0.has_recursion, 0
	.set .Lk_dkdv_sp_0.has_indirect_call, 0
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
        .size:           52
        .value_kind:     by_value
      - .address_space:  global
        .offset:         336
        .size:           8
        .value_kind:     global_buffer
      - .offset:         344
        .size:           52
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
      - .offset:         408
        .size:           4
        .value_kind:     by_value
      - .offset:         412
        .size:           4
        .value_kind:     by_value
      - .offset:         416
        .size:           4
        .value_kind:     by_value
      - .offset:         420
        .size:           4
        .value_kind:     by_value
      - .offset:         424
        .size:           4
        .value_kind:     by_value
      - .offset:         428
        .size:           4
        .value_kind:     by_value
      - .offset:         432
        .size:           4
        .value_kind:     by_value
      - .offset:         436
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 70656
    .kernarg_segment_align: 8
    .kernarg_segment_size: 440
    .max_flat_workgroup_size: 32
    .name:           k_dkdv_sp_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 32
      - 1
      - 1
    .sgpr_count:     78
    .sgpr_spill_count: 0
    .symbol:         k_dkdv_sp_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     707
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
