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
	s_set_vgpr_msb 0xc0
	v_and_b32_e32 v31 /*v799*/, 15, v0
	s_cselect_b32 s68, s6, s5
	s_cselect_b32 s4, s3, s4
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_wait_kmcnt 0x0
	s_mul_i32 s65, s38, s68
	s_mul_i32 s2, ttmp9, s2
	v_bfe_u32 v33 /*v801*/, v0, 4, 1
	s_add_co_i32 s5, s3, s2
	s_cmp_eq_u32 s7, 0
	s_load_b64 s[2:3], s[0:1], 0x190 nv
	s_cselect_b32 s64, ttmp9, s5
	s_lshr_b32 s5, s42, 31
	s_lshl_b32 s4, s4, 7
	s_add_co_i32 s5, s42, s5
	s_set_vgpr_msb 0xc000
	v_and_or_b32 v1, 0x60, v0, s4
	s_and_b32 s6, s5, -2
	s_ashr_i32 s5, s5, 1
	s_cmp_lg_u32 s42, s6
	s_set_vgpr_msb 0xc0
	v_and_b32_e32 v30 /*v798*/, 31, v0
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s42, 0
	s_set_vgpr_msb 0xc003
	v_or_b32_e32 v3, v31 /*v799*/, v1
	s_cselect_b32 s7, -1, 0
	s_load_b64 s[44:45], s[0:1], 0x30 nv
	s_and_b32 s6, s7, s6
	s_sub_co_ci_u32 s7, s5, 0
	s_sub_co_i32 s8, s4, s43
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_max_i32 s8, s8, 0
	s_lshr_b32 s8, s8, 5
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s9, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_and_b32 s9, s9, exec_lo
	s_cselect_b32 s69, s8, 0
	s_cmp_lg_u32 s6, 0
	s_sub_co_ci_u32 s66, s5, s69
	s_or_b32 s5, s4, 0x7f
	s_sub_co_i32 s5, s5, s43
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s5, 31
	s_ashr_i32 s8, s6, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s8, s8, 27
	s_add_co_i32 s8, s6, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s9, s8, 0xffffffe0
	s_ashr_i32 s8, s8, 5
	s_cmp_lg_u32 s6, s9
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s6, 0
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s6, s9
	s_sub_co_ci_u32 s6, s8, 0
	s_cmp_gt_i32 s5, -1
	s_mul_i32 s8, s39, s37
	s_cselect_b32 s5, s6, 0
	s_min_i32 s5, s5, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s5, s5, s69
	s_max_i32 s5, s5, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_i32 s4, s5, s66
	s_cmp_lg_u32 s2, 0
	s_mul_i32 s5, s38, s40
	s_cselect_b32 s72, -1, 0
	s_mul_i32 s5, s5, s3
	s_and_b32 s2, s72, exec_lo
	s_cselect_b32 s67, s4, 0
	s_lshl_b32 s2, s40, 4
	s_lshl_b32 s34, s5, 8
	s_mul_i32 s4, s2, s65
	s_ashr_i32 s35, s34, 31
	s_lshl4_add_u32 s4, s64, s4
	s_lshr_b64 s[46:47], s[34:35], 7
	v_mad_u32 v3, s2, v3, s4
	s_mov_b32 s6, s46
	s_mov_b32 s7, s47
	s_mul_i32 s3, s8, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_lshl_b32 s8, s3, 8
	s_lshl_b32 s10, s3, 2
	s_ashr_i32 s9, s8, 31
	v_or_b32_e32 v3, v33 /*v801*/, v3
	v_or_b32_e32 v2, 16, v1
	s_ashr_i32 s11, s10, 31
	s_lshr_b64 s[54:55], s[8:9], 7
	s_lshr_b64 s[58:59], s[10:11], 7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v3, 4, v3 :: v_dual_bitop2_b32 v4, v30 /*v798*/, v2 bitop3:0x54
	s_lshl_b32 s3, s3, 27
	s_mov_b32 s8, -1
	v_mad_u32 v4, s2, v4, s4
	s_clause 0x1
	s_load_b64 s[4:5], s[0:1], 0x60 nv
	s_load_b64 s[48:49], s[0:1], 0x90 nv
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v4, v33 /*v801*/, v4
	v_lshlrev_b32_e32 v4, 4, v4
	s_set_vgpr_msb 0x300
	s_clause 0x10
	buffer_load_b128 v[216:219], v3, s[44:47], null offen
	buffer_load_b128 v[220:223], v3, s[44:47], null offen offset:32
	buffer_load_b128 v[224:227], v3, s[44:47], null offen offset:64
	buffer_load_b128 v[228:231], v3, s[44:47], null offen offset:96
	buffer_load_b128 v[232:235], v3, s[44:47], null offen offset:128
	buffer_load_b128 v[236:239], v3, s[44:47], null offen offset:160
	buffer_load_b128 v[240:243], v3, s[44:47], null offen offset:192
	buffer_load_b128 v[244:247], v3, s[44:47], null offen offset:224
	buffer_load_b128 v[248:251], v4, s[44:47], null offen
	buffer_load_b128 v[252:255], v4, s[44:47], null offen offset:32
	s_set_vgpr_msb 64
	buffer_load_b128 v[0:3] /*v[256:259]*/, v4, s[44:47], null offen offset:64
	buffer_load_b128 v[4:7] /*v[260:263]*/, v4, s[44:47], null offen offset:96
	buffer_load_b128 v[8:11] /*v[264:267]*/, v4, s[44:47], null offen offset:128
	buffer_load_b128 v[12:15] /*v[268:271]*/, v4, s[44:47], null offen offset:160
	buffer_load_b128 v[16:19] /*v[272:275]*/, v4, s[44:47], null offen offset:192
	buffer_load_b128 v[20:23] /*v[276:279]*/, v4, s[44:47], null offen offset:224
	s_wait_kmcnt 0x0
	s_clause 0xf
	buffer_load_b128 v[24:27] /*v[280:283]*/, v3, s[4:7], null offen
	buffer_load_b128 v[28:31] /*v[284:287]*/, v3, s[4:7], null offen offset:32
	buffer_load_b128 v[40:43] /*v[296:299]*/, v3, s[4:7], null offen offset:64
	buffer_load_b128 v[44:47] /*v[300:303]*/, v3, s[4:7], null offen offset:96
	buffer_load_b128 v[48:51] /*v[304:307]*/, v3, s[4:7], null offen offset:128
	buffer_load_b128 v[52:55] /*v[308:311]*/, v3, s[4:7], null offen offset:160
	buffer_load_b128 v[56:59] /*v[312:315]*/, v3, s[4:7], null offen offset:192
	buffer_load_b128 v[60:63] /*v[316:319]*/, v3, s[4:7], null offen offset:224
	buffer_load_b128 v[64:67] /*v[320:323]*/, v4, s[4:7], null offen
	buffer_load_b128 v[68:71] /*v[324:327]*/, v4, s[4:7], null offen offset:32
	buffer_load_b128 v[72:75] /*v[328:331]*/, v4, s[4:7], null offen offset:64
	buffer_load_b128 v[76:79] /*v[332:335]*/, v4, s[4:7], null offen offset:96
	buffer_load_b128 v[80:83] /*v[336:339]*/, v4, s[4:7], null offen offset:128
	buffer_load_b128 v[84:87] /*v[340:343]*/, v4, s[4:7], null offen offset:160
	buffer_load_b128 v[88:91] /*v[344:347]*/, v4, s[4:7], null offen offset:192
	buffer_load_b128 v[92:95] /*v[348:351]*/, v4, s[4:7], null offen offset:224
	s_clause 0x2
	s_load_b64 s[52:53], s[0:1], 0x0 nv
	s_load_b64 s[4:5], s[0:1], 0xc0 nv
	s_load_b64 s[6:7], s[0:1], 0xe8 nv
	s_set_vgpr_msb 0x4000
	v_lshlrev_b32_e32 v4, 1, v0
	s_set_vgpr_msb 12
	v_lshlrev_b32_e32 v3, 3, v33 /*v801*/
	s_mul_i32 s44, s67, s41
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_gt_i32 s44, 0
	s_set_vgpr_msb 0xc00
	s_cbranch_scc1 .LBB0_2
	s_mul_i32 s71, s41, s64
	s_mul_i32 s70, s39, s68
	s_mov_b32 s8, s2
.LBB0_2:
	v_and_b32_e32 v5, 16, v0
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v1 /*v513*/, 0xc0, v4
	v_and_or_b32 v3 /*v515*/, v0, 7, v3
	s_set_vgpr_msb 0x8000
	v_bfe_u32 v0, v0, 3, 2
	s_set_vgpr_msb 3
	v_bfe_u32 v4, v30 /*v798*/, 3, 1
	s_set_vgpr_msb 0x3cc
	v_or_b32_e32 v35 /*v803*/, 16, v30 /*v798*/
	s_set_vgpr_msb 0xccc0
	v_or_b32_e32 v34 /*v802*/, v3, v1
	v_or_b32_e32 v32 /*v800*/, v3, v2
	s_wait_kmcnt 0x0
	s_or_b64 s[56:57], s[4:5], s[2:3]
	s_or_b64 s[60:61], s[6:7], s[2:3]
	s_lshl_b32 s35, s39, 4
	s_and_b32 s2, s8, exec_lo
	s_set_vgpr_msb 0xc0cc
	v_mad_u32_u24 v36 /*v804*/, 0x110, v31 /*v799*/, v5
	s_set_vgpr_msb 0xcc8c
	v_or_b32_e32 v165 /*v677*/, 3, v32 /*v800*/
	v_or_b32_e32 v164 /*v676*/, 2, v32 /*v800*/
	v_or_b32_e32 v163 /*v675*/, 5, v32 /*v800*/
	v_or_b32_e32 v162 /*v674*/, 4, v32 /*v800*/
	v_or_b32_e32 v161 /*v673*/, 7, v32 /*v800*/
	v_or_b32_e32 v160 /*v672*/, 6, v32 /*v800*/
	s_set_vgpr_msb 0x8ccc
	v_mad_u32_u24 v37 /*v805*/, 0x110, v35 /*v803*/, v5
	s_set_vgpr_msb 0xccc0
	v_lshlrev_b32_e32 v38 /*v806*/, 4, v4
	s_set_vgpr_msb 0xc08c
	v_or_b32_e32 v171 /*v683*/, 3, v34 /*v802*/
	v_or_b32_e32 v170 /*v682*/, 2, v34 /*v802*/
	v_or_b32_e32 v169 /*v681*/, 5, v34 /*v802*/
	v_or_b32_e32 v168 /*v680*/, 4, v34 /*v802*/
	v_or_b32_e32 v167 /*v679*/, 7, v34 /*v802*/
	v_or_b32_e32 v166 /*v678*/, 6, v34 /*v802*/
	s_set_vgpr_msb 0x8cc8
	v_mul_u32_u24_e32 v39 /*v807*/, 0x110, v3 /*v515*/
	s_set_vgpr_msb 0xc880
	v_lshlrev_b32_e32 v5 /*v517*/, 4, v0
	s_cselect_b32 s2, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s2, 1
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_5
	s_abs_i32 s74, s41
	s_set_vgpr_msb 11
	v_dual_add_nc_u32 v0, v36 /*v804*/, v1 /*v513*/ :: v_dual_add_nc_u32 v1, v37 /*v805*/, v1 /*v513*/
	s_cvt_f32_u32 s3, s74
	s_movk_i32 s2, 0x2200
	v_or_b32_e32 v8, v38 /*v806*/, v1 /*v513*/
	s_set_vgpr_msb 0xb40
	v_or_b32_e32 v227 /*v483*/, 0x10000, v0
	s_set_vgpr_msb 0x4008
	v_mad_u32_u24 v0, 0x110, v3 /*v515*/, s2
	v_s_rcp_f32 s2, s3
	s_set_vgpr_msb 0x84f
	v_dual_mov_b32 v104 /*v360*/, 0 :: v_dual_add_nc_u32 v231 /*v487*/, v39 /*v807*/, v38 /*v806*/
	s_mov_b32 s3, 0x12200
	s_set_vgpr_msb 0x4f40
	v_or_b32_e32 v229 /*v485*/, 0x10000, v1
	s_set_vgpr_msb 0x4009
	v_dual_mov_b32 v200, v104 /*v360*/ :: v_dual_bitop2_b32 v1, 32, v5 /*v517*/ bitop3:0x54
	v_or_b32_e32 v4, 0x60, v5 /*v517*/
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_set_vgpr_msb 0x90d
	v_dual_mov_b32 v202, v104 /*v360*/ :: v_dual_lshlrev_b32 v3, 1, v30 /*v798*/
	v_dual_mov_b32 v201, v104 /*v360*/ :: v_dual_bitop2_b32 v2, 64, v38 /*v806*/ bitop3:0x54
	s_cvt_u32_f32 s4, s2
	v_or_b32_e32 v5, 0x80, v38 /*v806*/
	s_set_vgpr_msb 0xd08
	v_or_b32_e32 v6, 0xa0, v5 /*v517*/
	s_set_vgpr_msb 0x80c
	v_or_b32_e32 v7, 0xc0, v38 /*v806*/
	v_and_or_b32 v3, v3, 16, 0xe0
	v_or_b32_e32 v9, 0x10000, v39 /*v807*/
	s_set_vgpr_msb 0xc08
	v_mad_u32_u24 v10, 0x110, v3 /*v515*/, s3
	s_set_vgpr_msb 0x801
	v_dual_mov_b32 v203, v104 /*v360*/ :: v_dual_bitop2_b32 v11, 32, v8 bitop3:0x54
	s_sub_co_i32 s3, 0, s74
	s_mov_b32 s2, s36
	s_mul_i32 s5, s3, s4
	s_set_vgpr_msb 0x14c
	v_add_nc_u32_e32 v233 /*v489*/, v0, v38 /*v806*/
	s_set_vgpr_msb 0x4c40
	v_add_nc_u32_e32 v237 /*v493*/, v0, v2
	s_mov_b32 s3, s36
	v_dual_add_nc_u32 v239 /*v495*/, v0, v4 :: v_dual_add_nc_u32 v241 /*v497*/, v0, v5
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, s[2:3]
	s_set_vgpr_msb 0x4043
	v_dual_add_nc_u32 v242 /*v498*/, v39 /*v807*/, v6 :: v_dual_add_nc_u32 v244 /*v500*/, v39 /*v807*/, v7
	v_add_nc_u32_e32 v238 /*v494*/, v39 /*v807*/, v4
	s_set_vgpr_msb 0x4340
	v_dual_add_nc_u32 v243 /*v499*/, v0, v6 :: v_dual_add_nc_u32 v245 /*v501*/, v0, v7
	s_set_vgpr_msb 0x4043
	v_dual_add_nc_u32 v246 /*v502*/, v39 /*v807*/, v3 :: v_dual_add_nc_u32 v240 /*v496*/, v39 /*v807*/, v5
	s_set_vgpr_msb 0x4340
	v_dual_add_nc_u32 v247 /*v503*/, v0, v3 :: v_dual_add_nc_u32 v248 /*v504*/, v9, v8
	v_dual_add_nc_u32 v235 /*v491*/, v0, v1 :: v_dual_add_nc_u32 v249 /*v505*/, v10, v8
	v_dual_add_nc_u32 v250 /*v506*/, v9, v11 :: v_dual_add_nc_u32 v251 /*v507*/, v10, v11
	s_set_vgpr_msb 0x4041
	v_dual_mov_b32 v105 /*v361*/, v104 /*v360*/ :: v_dual_mov_b32 v106 /*v362*/, v104 /*v360*/
	s_set_vgpr_msb 0x4143
	v_add_nc_u32_e32 v234 /*v490*/, v39 /*v807*/, v1
	s_set_vgpr_msb 0x4341
	v_mov_b32_e32 v110 /*v366*/, v104 /*v360*/
	s_set_vgpr_msb 0x4143
	v_add_nc_u32_e32 v236 /*v492*/, v39 /*v807*/, v2
	s_set_vgpr_msb 0x4341
	v_dual_mov_b32 v111 /*v367*/, v104 /*v360*/ :: v_dual_mov_b32 v32 /*v288*/, v104 /*v360*/
	v_dual_mov_b32 v107 /*v363*/, v104 /*v360*/ :: v_dual_mov_b32 v108 /*v364*/, v104 /*v360*/
	v_dual_mov_b32 v109 /*v365*/, v104 /*v360*/ :: v_dual_mov_b32 v33 /*v289*/, v104 /*v360*/
	v_dual_mov_b32 v34 /*v290*/, v104 /*v360*/ :: v_dual_mov_b32 v35 /*v291*/, v104 /*v360*/
	v_dual_mov_b32 v36 /*v292*/, v104 /*v360*/ :: v_dual_mov_b32 v37 /*v293*/, v104 /*v360*/
	v_dual_mov_b32 v38 /*v294*/, v104 /*v360*/ :: v_dual_mov_b32 v39 /*v295*/, v104 /*v360*/
	v_mov_b32_e32 v120 /*v376*/, v104 /*v360*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v204, v104 /*v360*/ :: v_dual_mov_b32 v205, v104 /*v360*/
	v_dual_mov_b32 v206, v104 /*v360*/ :: v_dual_mov_b32 v207, v104 /*v360*/
	v_dual_mov_b32 v184, v104 /*v360*/ :: v_dual_mov_b32 v185, v104 /*v360*/
	v_dual_mov_b32 v186, v104 /*v360*/ :: v_dual_mov_b32 v187, v104 /*v360*/
	v_dual_mov_b32 v188, v104 /*v360*/ :: v_dual_mov_b32 v189, v104 /*v360*/
	v_dual_mov_b32 v190, v104 /*v360*/ :: v_dual_mov_b32 v191, v104 /*v360*/
	v_dual_mov_b32 v168, v104 /*v360*/ :: v_dual_mov_b32 v169, v104 /*v360*/
	v_dual_mov_b32 v170, v104 /*v360*/ :: v_dual_mov_b32 v171, v104 /*v360*/
	v_dual_mov_b32 v172, v104 /*v360*/ :: v_dual_mov_b32 v173, v104 /*v360*/
	v_dual_mov_b32 v174, v104 /*v360*/ :: v_dual_mov_b32 v175, v104 /*v360*/
	v_dual_mov_b32 v152, v104 /*v360*/ :: v_dual_mov_b32 v153, v104 /*v360*/
	v_dual_mov_b32 v154, v104 /*v360*/ :: v_dual_mov_b32 v155, v104 /*v360*/
	v_dual_mov_b32 v156, v104 /*v360*/ :: v_dual_mov_b32 v157, v104 /*v360*/
	v_dual_mov_b32 v158, v104 /*v360*/ :: v_dual_mov_b32 v159, v104 /*v360*/
	v_dual_mov_b32 v136, v104 /*v360*/ :: v_dual_mov_b32 v137, v104 /*v360*/
	v_dual_mov_b32 v138, v104 /*v360*/ :: v_dual_mov_b32 v139, v104 /*v360*/
	v_dual_mov_b32 v140, v104 /*v360*/ :: v_dual_mov_b32 v141, v104 /*v360*/
	v_dual_mov_b32 v142, v104 /*v360*/ :: v_dual_mov_b32 v143, v104 /*v360*/
	v_dual_mov_b32 v128, v104 /*v360*/ :: v_dual_mov_b32 v129, v104 /*v360*/
	v_dual_mov_b32 v130, v104 /*v360*/ :: v_dual_mov_b32 v131, v104 /*v360*/
	v_dual_mov_b32 v132, v104 /*v360*/ :: v_dual_mov_b32 v133, v104 /*v360*/
	v_dual_mov_b32 v134, v104 /*v360*/ :: v_dual_mov_b32 v135, v104 /*v360*/
	v_dual_mov_b32 v112, v104 /*v360*/ :: v_dual_mov_b32 v113, v104 /*v360*/
	v_dual_mov_b32 v114, v104 /*v360*/ :: v_dual_mov_b32 v115, v104 /*v360*/
	v_dual_mov_b32 v116, v104 /*v360*/ :: v_dual_mov_b32 v117, v104 /*v360*/
	v_dual_mov_b32 v118, v104 /*v360*/ :: v_dual_mov_b32 v119, v104 /*v360*/
	v_dual_mov_b32 v88, v104 /*v360*/ :: v_dual_mov_b32 v89, v104 /*v360*/
	v_dual_mov_b32 v90, v104 /*v360*/ :: v_dual_mov_b32 v91, v104 /*v360*/
	v_dual_mov_b32 v92, v104 /*v360*/ :: v_dual_mov_b32 v93, v104 /*v360*/
	v_dual_mov_b32 v94, v104 /*v360*/ :: v_dual_mov_b32 v95, v104 /*v360*/
	v_dual_mov_b32 v72, v104 /*v360*/ :: v_dual_mov_b32 v73, v104 /*v360*/
	v_dual_mov_b32 v74, v104 /*v360*/ :: v_dual_mov_b32 v75, v104 /*v360*/
	v_dual_mov_b32 v76, v104 /*v360*/ :: v_dual_mov_b32 v77, v104 /*v360*/
	v_dual_mov_b32 v78, v104 /*v360*/ :: v_dual_mov_b32 v79, v104 /*v360*/
	v_dual_mov_b32 v56, v104 /*v360*/ :: v_dual_mov_b32 v57, v104 /*v360*/
	v_dual_mov_b32 v58, v104 /*v360*/ :: v_dual_mov_b32 v59, v104 /*v360*/
	v_dual_mov_b32 v60, v104 /*v360*/ :: v_dual_mov_b32 v61, v104 /*v360*/
	v_dual_mov_b32 v62, v104 /*v360*/ :: v_dual_mov_b32 v63, v104 /*v360*/
	v_dual_mov_b32 v40, v104 /*v360*/ :: v_dual_mov_b32 v41, v104 /*v360*/
	v_dual_mov_b32 v42, v104 /*v360*/ :: v_dual_mov_b32 v43, v104 /*v360*/
	v_dual_mov_b32 v44, v104 /*v360*/ :: v_dual_mov_b32 v45, v104 /*v360*/
	v_dual_mov_b32 v46, v104 /*v360*/ :: v_dual_mov_b32 v47, v104 /*v360*/
	v_dual_mov_b32 v24, v104 /*v360*/ :: v_dual_mov_b32 v25, v104 /*v360*/
	v_dual_mov_b32 v26, v104 /*v360*/ :: v_dual_mov_b32 v27, v104 /*v360*/
	v_dual_mov_b32 v28, v104 /*v360*/ :: v_dual_mov_b32 v29, v104 /*v360*/
	v_dual_mov_b32 v30, v104 /*v360*/ :: v_dual_mov_b32 v31, v104 /*v360*/
	v_dual_mov_b32 v8, v104 /*v360*/ :: v_dual_mov_b32 v9, v104 /*v360*/
	v_dual_mov_b32 v10, v104 /*v360*/ :: v_dual_mov_b32 v11, v104 /*v360*/
	v_dual_mov_b32 v12, v104 /*v360*/ :: v_dual_mov_b32 v13, v104 /*v360*/
	v_dual_mov_b32 v14, v104 /*v360*/ :: v_dual_mov_b32 v15, v104 /*v360*/
	v_dual_mov_b32 v0, v104 /*v360*/ :: v_dual_mov_b32 v1, v104 /*v360*/
	v_dual_mov_b32 v2, v104 /*v360*/ :: v_dual_mov_b32 v3, v104 /*v360*/
	v_dual_mov_b32 v4, v104 /*v360*/ :: v_dual_mov_b32 v5, v104 /*v360*/
	v_dual_mov_b32 v6, v104 /*v360*/ :: v_dual_mov_b32 v7, v104 /*v360*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v121 /*v377*/, v104 /*v360*/ :: v_dual_mov_b32 v122 /*v378*/, v104 /*v360*/
	v_dual_mov_b32 v123 /*v379*/, v104 /*v360*/ :: v_dual_mov_b32 v124 /*v380*/, v104 /*v360*/
	v_dual_mov_b32 v125 /*v381*/, v104 /*v360*/ :: v_dual_mov_b32 v126 /*v382*/, v104 /*v360*/
	v_dual_mov_b32 v127 /*v383*/, v104 /*v360*/ :: v_dual_mov_b32 v112 /*v368*/, v104 /*v360*/
	v_dual_mov_b32 v113 /*v369*/, v104 /*v360*/ :: v_dual_mov_b32 v114 /*v370*/, v104 /*v360*/
	v_dual_mov_b32 v115 /*v371*/, v104 /*v360*/ :: v_dual_mov_b32 v116 /*v372*/, v104 /*v360*/
	v_dual_mov_b32 v117 /*v373*/, v104 /*v360*/ :: v_dual_mov_b32 v118 /*v374*/, v104 /*v360*/
	v_dual_mov_b32 v119 /*v375*/, v104 /*v360*/ :: v_dual_mov_b32 v96 /*v352*/, v104 /*v360*/
	v_dual_mov_b32 v97 /*v353*/, v104 /*v360*/ :: v_dual_mov_b32 v98 /*v354*/, v104 /*v360*/
	v_dual_mov_b32 v99 /*v355*/, v104 /*v360*/ :: v_dual_mov_b32 v100 /*v356*/, v104 /*v360*/
	v_dual_mov_b32 v101 /*v357*/, v104 /*v360*/ :: v_dual_mov_b32 v102 /*v358*/, v104 /*v360*/
	v_mov_b32_e32 v103 /*v359*/, v104 /*v360*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v208, v104 /*v360*/ :: v_dual_mov_b32 v209, v104 /*v360*/
	v_dual_mov_b32 v210, v104 /*v360*/ :: v_dual_mov_b32 v211, v104 /*v360*/
	v_dual_mov_b32 v212, v104 /*v360*/ :: v_dual_mov_b32 v213, v104 /*v360*/
	v_dual_mov_b32 v214, v104 /*v360*/ :: v_dual_mov_b32 v215, v104 /*v360*/
	v_dual_mov_b32 v192, v104 /*v360*/ :: v_dual_mov_b32 v193, v104 /*v360*/
	v_dual_mov_b32 v194, v104 /*v360*/ :: v_dual_mov_b32 v195, v104 /*v360*/
	v_dual_mov_b32 v196, v104 /*v360*/ :: v_dual_mov_b32 v197, v104 /*v360*/
	v_dual_mov_b32 v198, v104 /*v360*/ :: v_dual_mov_b32 v199, v104 /*v360*/
	v_dual_mov_b32 v176, v104 /*v360*/ :: v_dual_mov_b32 v177, v104 /*v360*/
	v_dual_mov_b32 v178, v104 /*v360*/ :: v_dual_mov_b32 v179, v104 /*v360*/
	v_dual_mov_b32 v180, v104 /*v360*/ :: v_dual_mov_b32 v181, v104 /*v360*/
	v_dual_mov_b32 v182, v104 /*v360*/ :: v_dual_mov_b32 v183, v104 /*v360*/
	v_dual_mov_b32 v160, v104 /*v360*/ :: v_dual_mov_b32 v161, v104 /*v360*/
	v_dual_mov_b32 v162, v104 /*v360*/ :: v_dual_mov_b32 v163, v104 /*v360*/
	v_dual_mov_b32 v164, v104 /*v360*/ :: v_dual_mov_b32 v165, v104 /*v360*/
	v_dual_mov_b32 v166, v104 /*v360*/ :: v_dual_mov_b32 v167, v104 /*v360*/
	v_dual_mov_b32 v144, v104 /*v360*/ :: v_dual_mov_b32 v145, v104 /*v360*/
	v_dual_mov_b32 v146, v104 /*v360*/ :: v_dual_mov_b32 v147, v104 /*v360*/
	v_dual_mov_b32 v148, v104 /*v360*/ :: v_dual_mov_b32 v149, v104 /*v360*/
	v_dual_mov_b32 v150, v104 /*v360*/ :: v_dual_mov_b32 v151, v104 /*v360*/
	v_dual_mov_b32 v120, v104 /*v360*/ :: v_dual_mov_b32 v121, v104 /*v360*/
	v_dual_mov_b32 v122, v104 /*v360*/ :: v_dual_mov_b32 v123, v104 /*v360*/
	v_dual_mov_b32 v124, v104 /*v360*/ :: v_dual_mov_b32 v125, v104 /*v360*/
	v_dual_mov_b32 v126, v104 /*v360*/ :: v_dual_mov_b32 v127, v104 /*v360*/
	v_dual_mov_b32 v104, v104 /*v360*/ :: v_dual_mov_b32 v105, v104 /*v360*/
	v_dual_mov_b32 v106, v104 /*v360*/ :: v_dual_mov_b32 v107, v104 /*v360*/
	v_dual_mov_b32 v108, v104 /*v360*/ :: v_dual_mov_b32 v109, v104 /*v360*/
	v_dual_mov_b32 v110, v104 /*v360*/ :: v_dual_mov_b32 v111, v104 /*v360*/
	v_dual_mov_b32 v96, v104 /*v360*/ :: v_dual_mov_b32 v97, v104 /*v360*/
	v_dual_mov_b32 v98, v104 /*v360*/ :: v_dual_mov_b32 v99, v104 /*v360*/
	v_dual_mov_b32 v100, v104 /*v360*/ :: v_dual_mov_b32 v101, v104 /*v360*/
	v_dual_mov_b32 v102, v104 /*v360*/ :: v_dual_mov_b32 v103, v104 /*v360*/
	v_dual_mov_b32 v80, v104 /*v360*/ :: v_dual_mov_b32 v81, v104 /*v360*/
	v_dual_mov_b32 v82, v104 /*v360*/ :: v_dual_mov_b32 v83, v104 /*v360*/
	v_dual_mov_b32 v84, v104 /*v360*/ :: v_dual_mov_b32 v85, v104 /*v360*/
	v_dual_mov_b32 v86, v104 /*v360*/ :: v_dual_mov_b32 v87, v104 /*v360*/
	v_dual_mov_b32 v64, v104 /*v360*/ :: v_dual_mov_b32 v65, v104 /*v360*/
	v_dual_mov_b32 v66, v104 /*v360*/ :: v_dual_mov_b32 v67, v104 /*v360*/
	v_dual_mov_b32 v68, v104 /*v360*/ :: v_dual_mov_b32 v69, v104 /*v360*/
	v_dual_mov_b32 v70, v104 /*v360*/ :: v_dual_mov_b32 v71, v104 /*v360*/
	v_dual_mov_b32 v48, v104 /*v360*/ :: v_dual_mov_b32 v49, v104 /*v360*/
	v_dual_mov_b32 v50, v104 /*v360*/ :: v_dual_mov_b32 v51, v104 /*v360*/
	v_dual_mov_b32 v52, v104 /*v360*/ :: v_dual_mov_b32 v53, v104 /*v360*/
	v_dual_mov_b32 v54, v104 /*v360*/ :: v_dual_mov_b32 v55, v104 /*v360*/
	v_dual_mov_b32 v32, v104 /*v360*/ :: v_dual_mov_b32 v33, v104 /*v360*/
	v_dual_mov_b32 v34, v104 /*v360*/ :: v_dual_mov_b32 v35, v104 /*v360*/
	v_dual_mov_b32 v36, v104 /*v360*/ :: v_dual_mov_b32 v37, v104 /*v360*/
	v_dual_mov_b32 v38, v104 /*v360*/ :: v_dual_mov_b32 v39, v104 /*v360*/
	v_dual_mov_b32 v16, v104 /*v360*/ :: v_dual_mov_b32 v17, v104 /*v360*/
	v_dual_mov_b32 v18, v104 /*v360*/ :: v_dual_mov_b32 v19, v104 /*v360*/
	v_dual_mov_b32 v20, v104 /*v360*/ :: v_dual_mov_b32 v21, v104 /*v360*/
	v_dual_mov_b32 v22, v104 /*v360*/ :: v_dual_mov_b32 v23, v104 /*v360*/
	s_mul_i32 s73, s37, s35
	s_mul_hi_u32 s5, s4, s5
	s_ashr_i32 s45, s44, 31
	s_mul_i32 s71, s41, s64
	s_mul_i32 s73, s73, s68
	s_mul_i32 s70, s39, s68
	s_ashr_i32 s75, s41, 31
	s_add_co_i32 s76, s4, s5
	s_mov_b64 s[38:39], 0
	s_mov_b32 s62, s58
	s_mov_b32 s63, s59
	s_mov_b32 s50, s54
	s_mov_b32 s51, s55
	s_mov_b32 s42, 0x3fb8aa3b
	s_set_vgpr_msb 0x100
.LBB0_4:
	s_abs_i32 s2, s38
	s_ashr_i32 s4, s38, 31
	s_mul_hi_u32 s3, s2, s76
	s_xor_b32 s4, s4, s75
	s_mul_i32 s5, s3, s74
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s2, s2, s5
	s_add_co_i32 s5, s3, 1
	s_sub_co_i32 s6, s2, s74
	s_cmp_ge_u32 s2, s74
	s_cselect_b32 s3, s5, s3
	s_cselect_b32 s2, s6, s2
	s_add_co_i32 s5, s3, 1
	s_cmp_ge_u32 s2, s74
	s_cselect_b32 s2, s5, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s2, s2, s4
	s_sub_co_i32 s3, s2, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s3, s3, s41
	s_cmp_lg_u32 s38, s3
	s_cselect_b32 s3, -1, 0
	s_xor_b32 s5, s41, s38
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s5, 0
	s_cselect_b32 s5, -1, 0
	s_and_b32 s3, s5, s3
	s_sub_co_ci_u32 s2, s2, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s3, s2, s41
	s_add_co_i32 s2, s2, s69
	s_sub_co_i32 s3, s38, s3
	s_lshl_b32 s2, s2, 5
	s_add_co_i32 s3, s3, s71
	s_set_vgpr_msb 0x4c
	v_or_b32_e32 v252 /*v508*/, s2, v31 /*v799*/
	v_or_b32_e32 v253 /*v509*/, s2, v35 /*v803*/
	s_add_co_i32 s2, s3, s70
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s2, s2, s37
	v_nop
	v_nop
	s_set_vgpr_msb 0x4c45
	v_add_lshl_u32 v128 /*v384*/, s2, v252 /*v508*/, 2
	v_add_lshl_u32 v129 /*v385*/, s2, v253 /*v509*/, 2
	s_lshl4_add_u32 s2, s3, s73
	s_clause 0x1
	buffer_load_b32 v228 /*v484*/, v128 /*v384*/, s[56:59], null offen
	buffer_load_b32 v226 /*v482*/, v129 /*v385*/, s[56:59], null offen
	s_clause 0x1
	buffer_load_b32 v232 /*v488*/, v128 /*v384*/, s[60:63], null offen
	buffer_load_b32 v230 /*v486*/, v129 /*v385*/, s[60:63], null offen
	s_wait_xcnt 0x1
	v_mad_u32 v128 /*v384*/, v252 /*v508*/, s35, s2
	v_mad_u32 v200 /*v456*/, v253 /*v509*/, s35, s2
	s_add_nc_u64 s[38:39], s[38:39], 1
	v_dual_add_nc_u32 v253 /*v509*/, s43, v253 /*v509*/ :: v_dual_add_nc_u32 v252 /*v508*/, s43, v252 /*v508*/
	s_set_vgpr_msb 0x454d
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v128 /*v384*/, v128 /*v384*/, v33 /*v801*/
	v_or_b32_e32 v200 /*v456*/, v200 /*v456*/, v33 /*v801*/
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_i32_e64 s17, v253 /*v509*/, v34 /*v802*/
	v_cmp_lt_i32_e64 s18, v253 /*v509*/, v34 /*v802*/
	s_set_vgpr_msb 0x4d46
	v_cmp_gt_i32_e64 s19, v171 /*v683*/, v253 /*v509*/
	v_lshlrev_b32_e32 v254 /*v510*/, 4, v128 /*v384*/
	v_lshlrev_b32_e32 v255 /*v511*/, 4, v200 /*v456*/
	v_cmp_gt_i32_e64 s20, v170 /*v682*/, v253 /*v509*/
	v_cmp_gt_i32_e64 s21, v169 /*v681*/, v253 /*v509*/
	v_cmp_gt_i32_e64 s22, v168 /*v680*/, v253 /*v509*/
	v_or_b32_e32 v160 /*v416*/, 32, v254 /*v510*/
	v_or_b32_e32 v192 /*v448*/, 64, v254 /*v510*/
	v_or_b32_e32 v196 /*v452*/, 0x60, v254 /*v510*/
	v_cmp_gt_i32_e64 s23, v167 /*v679*/, v253 /*v509*/
	s_set_vgpr_msb 0x4641
	s_clause 0x1
	buffer_load_b128 v[152:155] /*v[408:411]*/, v254 /*v510*/, s[52:55], null offen
	buffer_load_b128 v[156:159] /*v[412:415]*/, v160 /*v416*/, s[52:55], null offen
	s_set_vgpr_msb 0x4184
	v_or_b32_e32 v0 /*v512*/, 32, v255 /*v511*/
	v_or_b32_e32 v2 /*v514*/, 0x60, v255 /*v511*/
	s_set_vgpr_msb 0x8441
	s_clause 0x1
	buffer_load_b128 v[176:179] /*v[432:435]*/, v192 /*v448*/, s[52:55], null offen
	buffer_load_b128 v[180:183] /*v[436:439]*/, v196 /*v452*/, s[52:55], null offen
	s_clause 0x1
	buffer_load_b128 v[184:187] /*v[440:443]*/, v254 /*v510*/, s[48:51], null offen
	buffer_load_b128 v[188:191] /*v[444:447]*/, v160 /*v416*/, s[48:51], null offen
	s_clause 0x2
	buffer_load_b128 v[200:203] /*v[456:459]*/, v255 /*v511*/, s[52:55], null offen
	s_set_vgpr_msb 0x4142
	buffer_load_b128 v[204:207] /*v[460:463]*/, v0 /*v512*/, s[52:55], null offen
	s_set_vgpr_msb 0x4281
	s_clause 0x2
	buffer_load_b128 v[22:25] /*v[534:537]*/, v255 /*v511*/, s[48:51], null offen
	s_set_vgpr_msb 0x8186
	buffer_load_b128 v[26:29] /*v[538:541]*/, v0 /*v512*/, s[48:51], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v0 /*v512*/, 64, v255 /*v511*/
	s_set_vgpr_msb 0x8641
	s_clause 0x1
	buffer_load_b128 v[192:195] /*v[448:451]*/, v192 /*v448*/, s[48:51], null offen
	buffer_load_b128 v[196:199] /*v[452:455]*/, v196 /*v452*/, s[48:51], null offen
	s_set_vgpr_msb 0x4107
	v_cmp_gt_i32_e64 s2, v34 /*v802*/, v252 /*v508*/
	s_set_vgpr_msb 0x786
	s_clause 0x1
	buffer_load_b128 v[38:41] /*v[550:553]*/, v0 /*v512*/, s[52:55], null offen
	buffer_load_b128 v[42:45] /*v[554:557]*/, v2 /*v514*/, s[52:55], null offen
	s_clause 0x1
	buffer_load_b128 v[54:57] /*v[566:569]*/, v0 /*v512*/, s[48:51], null offen
	buffer_load_b128 v[58:61] /*v[570:573]*/, v2 /*v514*/, s[48:51], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v0 /*v512*/, 0x80, v254 /*v510*/
	s_wait_xcnt 0x0
	v_or_b32_e32 v2 /*v514*/, 0xa0, v254 /*v510*/
	s_clause 0x1
	buffer_load_b128 v[102:105] /*v[614:617]*/, v0 /*v512*/, s[52:55], null offen
	buffer_load_b128 v[106:109] /*v[618:621]*/, v2 /*v514*/, s[52:55], null offen
	s_clause 0x1
	buffer_load_b128 v[110:113] /*v[622:625]*/, v0 /*v512*/, s[48:51], null offen
	buffer_load_b128 v[114:117] /*v[626:629]*/, v2 /*v514*/, s[48:51], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v0 /*v512*/, 0xc0, v254 /*v510*/
	s_set_vgpr_msb 0x8644
	v_or_b32_e32 v254 /*v510*/, 0xe0, v254 /*v510*/
	s_set_vgpr_msb 0x4482
	s_clause 0x2
	buffer_load_b128 v[118:121] /*v[630:633]*/, v0 /*v512*/, s[52:55], null offen
	s_set_vgpr_msb 0x8281
	buffer_load_b128 v[122:125] /*v[634:637]*/, v254 /*v510*/, s[52:55], null offen
	s_set_vgpr_msb 0x8182
	s_clause 0x2
	buffer_load_b128 v[126:129] /*v[638:641]*/, v0 /*v512*/, s[48:51], null offen
	s_set_vgpr_msb 0x8281
	buffer_load_b128 v[130:133] /*v[642:645]*/, v254 /*v510*/, s[48:51], null offen
	s_set_vgpr_msb 0x8144
	v_or_b32_e32 v254 /*v510*/, 0x80, v255 /*v511*/
	s_set_vgpr_msb 0x4485
	v_or_b32_e32 v0 /*v512*/, 0xa0, v255 /*v511*/
	s_clause 0x2
	buffer_load_b128 v[134:137] /*v[646:649]*/, v254 /*v510*/, s[52:55], null offen
	s_set_vgpr_msb 0x8582
	buffer_load_b128 v[138:141] /*v[650:653]*/, v0 /*v512*/, s[52:55], null offen
	s_set_vgpr_msb 0x8281
	s_clause 0x2
	buffer_load_b128 v[142:145] /*v[654:657]*/, v254 /*v510*/, s[48:51], null offen
	s_set_vgpr_msb 0x8182
	buffer_load_b128 v[146:149] /*v[658:661]*/, v0 /*v512*/, s[48:51], null offen
	s_set_vgpr_msb 0x8244
	v_or_b32_e32 v254 /*v510*/, 0xc0, v255 /*v511*/
	v_or_b32_e32 v255 /*v511*/, 0xe0, v255 /*v511*/
	s_set_vgpr_msb 0x4481
	s_clause 0x1
	buffer_load_b128 v[150:153] /*v[662:665]*/, v254 /*v510*/, s[52:55], null offen
	buffer_load_b128 v[154:157] /*v[666:669]*/, v255 /*v511*/, s[52:55], null offen
	s_clause 0x1
	buffer_load_b128 v[172:175] /*v[684:687]*/, v254 /*v510*/, s[48:51], null offen
	buffer_load_b128 v[176:179] /*v[688:691]*/, v255 /*v511*/, s[48:51], null offen
	s_set_vgpr_msb 0x8106
	v_cmp_gt_i32_e64 s3, v171 /*v683*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s4, v170 /*v682*/, v252 /*v508*/
	s_and_b32 s2, s72, s2
	v_cmp_gt_i32_e64 s5, v169 /*v681*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s6, v168 /*v680*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s7, v167 /*v679*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s8, v166 /*v678*/, v252 /*v508*/
	s_set_vgpr_msb 0x607
	v_cmp_ge_i32_e64 s9, v32 /*v800*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s10, v32 /*v800*/, v252 /*v508*/
	s_set_vgpr_msb 0x706
	v_cmp_gt_i32_e64 s11, v165 /*v677*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s12, v164 /*v676*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s13, v163 /*v675*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s14, v162 /*v674*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s15, v161 /*v673*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s16, v160 /*v672*/, v252 /*v508*/
	v_cmp_gt_i32_e64 s24, v166 /*v678*/, v253 /*v509*/
	s_set_vgpr_msb 0x607
	v_cmp_ge_i32_e64 s25, v32 /*v800*/, v253 /*v509*/
	v_cmp_gt_i32_e64 s26, v32 /*v800*/, v253 /*v509*/
	s_set_vgpr_msb 0x706
	v_cmp_gt_i32_e64 s27, v165 /*v677*/, v253 /*v509*/
	v_cmp_gt_i32_e64 s28, v164 /*v676*/, v253 /*v509*/
	v_cmp_gt_i32_e64 s29, v163 /*v675*/, v253 /*v509*/
	s_set_vgpr_msb 0x607
	v_cmp_ge_i32_e32 vcc_lo, v34 /*v802*/, v252 /*v508*/
	s_set_vgpr_msb 0x706
	v_cmp_gt_i32_e64 s30, v162 /*v674*/, v253 /*v509*/
	v_cmp_gt_i32_e64 s31, v161 /*v673*/, v253 /*v509*/
	v_cmp_gt_i32_e64 s33, v160 /*v672*/, v253 /*v509*/
	s_set_vgpr_msb 0x644
	s_wait_loadcnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[216:223], v[152:159] /*v[408:415]*/, 0
	s_set_vgpr_msb 0x4407
	s_wait_loadcnt 0x1b
	ds_store_b128 v36 /*v804*/, v[184:187] /*v[440:443]*/
	s_wait_loadcnt 0x1a
	ds_store_b128 v36 /*v804*/, v[188:191] /*v[444:447]*/ offset:32
	ds_store_b128 v36 /*v804*/, v[152:155] /*v[408:411]*/ offset:8704
	ds_store_b128 v36 /*v804*/, v[156:159] /*v[412:415]*/ offset:8736
	s_wait_loadcnt 0x15
	ds_store_b128 v36 /*v804*/, v[192:195] /*v[448:451]*/ offset:64
	s_wait_loadcnt 0x14
	ds_store_b128 v36 /*v804*/, v[196:199] /*v[452:455]*/ offset:96
	ds_store_b128 v36 /*v804*/, v[176:179] /*v[432:435]*/ offset:8768
	ds_store_b128 v36 /*v804*/, v[180:183] /*v[436:439]*/ offset:8800
	s_set_vgpr_msb 0x744
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[224:231], v[176:183] /*v[432:439]*/, 0
	s_and_b32 s77, s72, vcc_lo
	s_set_vgpr_msb 0x440b
	s_wait_loadcnt 0xd
	ds_store_b128 v36 /*v804*/, v[110:113] /*v[622:625]*/ offset:128
	s_wait_loadcnt 0xc
	ds_store_b128 v36 /*v804*/, v[114:117] /*v[626:629]*/ offset:160
	ds_store_b128 v36 /*v804*/, v[102:105] /*v[614:617]*/ offset:8832
	ds_store_b128 v36 /*v804*/, v[106:109] /*v[618:621]*/ offset:8864
	s_wait_loadcnt 0x9
	ds_store_b128 v36 /*v804*/, v[126:129] /*v[638:641]*/ offset:192
	s_wait_loadcnt 0x8
	ds_store_b128 v36 /*v804*/, v[130:133] /*v[642:645]*/ offset:224
	ds_store_b128 v36 /*v804*/, v[118:121] /*v[630:633]*/ offset:8896
	ds_store_b128 v36 /*v804*/, v[122:125] /*v[634:637]*/ offset:8928
	s_set_vgpr_msb 0xb58
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[392:399]*/, v[232:239], v[102:109] /*v[614:621]*/, v[136:143] /*v[392:399]*/
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[240:247], v[118:125] /*v[630:637]*/, v[144:151] /*v[400:407]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5845
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_add_f32 v[136:137] /*v[392:393]*/, v[136:137] /*v[392:393]*/, v[144:145] /*v[400:401]*/
	s_set_vgpr_msb 0x4544
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[248:255], v[152:159] /*v[408:415]*/, 0
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[138:139] /*v[394:395]*/, v[138:139] /*v[394:395]*/, v[146:147] /*v[402:403]*/
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, v[148:149] /*v[404:405]*/
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[150:151] /*v[406:407]*/
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[224:225] /*v[480:481]*/, v[136:137] /*v[392:393]*/
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[224:225] /*v[480:481]*/, v[138:139] /*v[394:395]*/
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[224:225] /*v[480:481]*/, v[140:141] /*v[396:397]*/
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[0:7] /*v[256:263]*/, v[176:183] /*v[432:439]*/, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v136 /*v392*/, v136 /*v392*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s3
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[224:225] /*v[480:481]*/, v[142:143] /*v[398:399]*/
	v_cndmask_b32_e64 v139 /*v395*/, v139 /*v395*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s4
	v_cndmask_b32_e64 v137 /*v393*/, v137 /*v393*/, 0xff61b1e6, s77
	v_cndmask_b32_e64 v138 /*v394*/, v138 /*v394*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[8:15] /*v[264:271]*/, v[102:109] /*v[614:621]*/, v[128:135] /*v[384:391]*/
	s_and_b32 s2, s72, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v141 /*v397*/, v141 /*v397*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s6
	s_set_vgpr_msb 0x5945
	v_pk_add_f32 v[138:139] /*v[394:395]*/, v[138:139] /*v[394:395]*/, v[228:229] /*v[484:485]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v140 /*v396*/, v140 /*v396*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s7
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[272:279]*/, v[118:125] /*v[630:637]*/, v[168:175] /*v[424:431]*/
	v_cndmask_b32_e64 v143 /*v399*/, v143 /*v399*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s8
	s_set_vgpr_msb 0x5945
	v_pk_add_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, v[228:229] /*v[484:485]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v142 /*v398*/, v142 /*v398*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s9
	v_pk_add_f32 v[136:137] /*v[392:393]*/, v[136:137] /*v[392:393]*/, v[228:229] /*v[484:485]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[138:139] /*v[394:395]*/, v[138:139] /*v[394:395]*/, s[42:43] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[64:71] /*v[320:327]*/, v[184:191] /*v[440:447]*/, 0
	v_pk_add_f32 v[128:129] /*v[384:385]*/, v[128:129] /*v[384:385]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[172:173] /*v[428:429]*/
	v_pk_add_f32 v[134:135] /*v[390:391]*/, v[134:135] /*v[390:391]*/, v[174:175] /*v[430:431]*/
	v_pk_add_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, v[228:229] /*v[484:485]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[128:129] /*v[384:385]*/, v[224:225] /*v[480:481]*/, v[128:129] /*v[384:385]*/
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[224:225] /*v[480:481]*/, v[130:131] /*v[386:387]*/
	s_set_vgpr_msb 0x4585
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[518:525]*/, v[72:79] /*v[328:335]*/, v[192:199] /*v[448:455]*/, 0
	s_set_vgpr_msb 0x8545
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[224:225] /*v[480:481]*/, v[132:133] /*v[388:389]*/
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[224:225] /*v[480:481]*/, v[134:135] /*v[390:391]*/
	v_cndmask_b32_e64 v129 /*v385*/, v129 /*v385*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s10
	v_pk_mul_f32 v[140:141] /*v[396:397]*/, v[140:141] /*v[396:397]*/, s[42:43] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v128 /*v384*/, v128 /*v384*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s11
	s_set_vgpr_msb 0x4584
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[216:223], v[200:207] /*v[456:463]*/, 0
	s_set_vgpr_msb 0x8445
	v_cndmask_b32_e64 v131 /*v387*/, v131 /*v387*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s12
	v_pk_add_f32 v[128:129] /*v[384:385]*/, v[128:129] /*v[384:385]*/, v[228:229] /*v[484:485]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v130 /*v386*/, v130 /*v386*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s13
	v_pk_mul_f32 v[142:143] /*v[398:399]*/, v[142:143] /*v[398:399]*/, s[42:43] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v133 /*v389*/, v133 /*v389*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x4588
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[224:231], v[38:45] /*v[550:557]*/, 0
	s_and_b32 s2, s72, s14
	s_set_vgpr_msb 0x8845
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[228:229] /*v[484:485]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v132 /*v388*/, v132 /*v388*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s15
	v_pk_mul_f32 v[136:137] /*v[392:393]*/, v[136:137] /*v[392:393]*/, s[42:43] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v135 /*v391*/, v135 /*v391*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s16
	s_set_vgpr_msb 0x4559
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[80:87] /*v[336:343]*/, v[110:117] /*v[622:629]*/, v[160:167] /*v[416:423]*/
	v_cndmask_b32_e64 v134 /*v390*/, v134 /*v390*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s17
	s_set_vgpr_msb 0x5945
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[228:229] /*v[484:485]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129] /*v[384:385]*/, v[128:129] /*v[384:385]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_add_f32 v[134:135] /*v[390:391]*/, v[134:135] /*v[390:391]*/, v[228:229] /*v[484:485]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v138 /*v394*/, v138 /*v394*/
	s_set_vgpr_msb 0x45a9
	v_wmma_f32_16x16x32_bf16 v[6:13] /*v[518:525]*/, v[88:95] /*v[344:351]*/, v[126:133] /*v[638:645]*/, v[6:13] /*v[518:525]*/
	s_set_vgpr_msb 0xa941
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, s[42:43] op_sel_hi:[1,0]
	v_exp_f32_e32 v139 /*v395*/, v139 /*v395*/
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[134:135] /*v[390:391]*/, s[42:43] op_sel_hi:[1,0]
	v_exp_f32_e32 v140 /*v396*/, v140 /*v396*/
	v_exp_f32_e32 v141 /*v397*/, v141 /*v397*/
	v_exp_f32_e32 v142 /*v398*/, v142 /*v398*/
	v_exp_f32_e32 v143 /*v399*/, v143 /*v399*/
	s_set_vgpr_msb 0x41a8
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[14:21] /*v[526:533]*/, v[232:239], v[134:141] /*v[646:653]*/, v[14:21] /*v[526:533]*/
	s_set_vgpr_msb 0xa849
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[160:161] /*v[416:417]*/, v[6:7] /*v[518:519]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[162:163] /*v[418:419]*/, v[8:9] /*v[520:521]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[164:165] /*v[420:421]*/, v[10:11] /*v[522:523]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[166:167] /*v[422:423]*/, v[12:13] /*v[524:525]*/
	v_exp_f32_e32 v136 /*v392*/, v136 /*v392*/
	v_exp_f32_e32 v137 /*v393*/, v137 /*v393*/
	s_set_vgpr_msb 0x4945
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[232:233] /*v[488:489]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x45a8
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[46:53] /*v[558:565]*/, v[240:247], v[150:157] /*v[662:669]*/, v[46:53] /*v[558:565]*/
	s_set_vgpr_msb 0xa845
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[232:233] /*v[488:489]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[232:233] /*v[488:489]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[232:233] /*v[488:489]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_nop
	s_set_vgpr_msb 0x454a
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[14:15] /*v[526:527]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x4a84
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[248:255], v[200:207] /*v[456:463]*/, 0
	s_set_vgpr_msb 0x844a
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[16:17] /*v[528:529]*/, v[48:49] /*v[560:561]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[18:19] /*v[530:531]*/, v[50:51] /*v[562:563]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[20:21] /*v[532:533]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x4a45
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[224:225] /*v[480:481]*/, v[160:161] /*v[416:417]*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[224:225] /*v[480:481]*/, v[162:163] /*v[418:419]*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[224:225] /*v[480:481]*/, v[164:165] /*v[420:421]*/
	s_set_vgpr_msb 0x4589
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[598:605]*/, v[0:7] /*v[256:263]*/, v[38:45] /*v[550:557]*/, 0
	s_set_vgpr_msb 0x8945
	v_cndmask_b32_e64 v161 /*v417*/, v161 /*v417*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s18
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[224:225] /*v[480:481]*/, v[166:167] /*v[422:423]*/
	v_cndmask_b32_e64 v160 /*v416*/, v160 /*v416*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s19
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v163 /*v419*/, v163 /*v419*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x45a9
	v_wmma_f32_16x16x32_bf16 v[70:77] /*v[582:589]*/, v[8:15] /*v[264:271]*/, v[134:141] /*v[646:653]*/, v[70:77] /*v[582:589]*/
	s_and_b32 s2, s72, s20
	s_set_vgpr_msb 0xa945
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[226:227] /*v[482:483]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v162 /*v418*/, v162 /*v418*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s21
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cndmask_b32_e64 v165 /*v421*/, v165 /*v421*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s22
	s_set_vgpr_msb 0x45a9
	v_wmma_f32_16x16x32_bf16 v[86:93] /*v[598:605]*/, v[16:23] /*v[272:279]*/, v[150:157] /*v[662:669]*/, v[86:93] /*v[598:605]*/
	s_set_vgpr_msb 0xa945
	v_cndmask_b32_e64 v164 /*v420*/, v164 /*v420*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s23
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, v[226:227] /*v[482:483]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v167 /*v423*/, v167 /*v423*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s24
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[164:165] /*v[420:421]*/, v[226:227] /*v[482:483]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v166 /*v422*/, v166 /*v422*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[70:71] /*v[582:583]*/, v[86:87] /*v[598:599]*/
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[72:73] /*v[584:585]*/, v[88:89] /*v[600:601]*/
	v_pk_add_f32 v[180:181] /*v[436:437]*/, v[74:75] /*v[586:587]*/, v[90:91] /*v[602:603]*/
	s_and_b32 s2, s72, s25
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[76:77] /*v[588:589]*/, v[92:93] /*v[604:605]*/
	s_set_vgpr_msb 0x4a45
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[224:225] /*v[480:481]*/, v[176:177] /*v[432:433]*/
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[224:225] /*v[480:481]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[224:225] /*v[480:481]*/, v[180:181] /*v[436:437]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[24:31] /*v[280:287]*/, v[184:191] /*v[440:447]*/, 0
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[224:225] /*v[480:481]*/, v[182:183] /*v[438:439]*/
	v_cndmask_b32_e64 v177 /*v433*/, v177 /*v433*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s26
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[166:167] /*v[422:423]*/, v[226:227] /*v[482:483]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v176 /*v432*/, v176 /*v432*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s27
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, s[42:43] op_sel_hi:[1,0]
	v_cndmask_b32_e64 v179 /*v435*/, v179 /*v435*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s28
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[40:47] /*v[296:303]*/, v[192:199] /*v[448:455]*/, 0
	v_cndmask_b32_e64 v178 /*v434*/, v178 /*v434*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s29
	v_pk_add_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, v[226:227] /*v[482:483]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v181 /*v437*/, v181 /*v437*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s30
	v_pk_add_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, v[226:227] /*v[482:483]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v180 /*v436*/, v180 /*v436*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s31
	s_set_vgpr_msb 0x4589
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[24:31] /*v[280:287]*/, v[22:29] /*v[534:541]*/, 0
	s_set_vgpr_msb 0x8945
	v_cndmask_b32_e64 v183 /*v439*/, v183 /*v439*/, 0xff61b1e6, s2
	s_and_b32 s2, s72, s33
	v_pk_add_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, v[226:227] /*v[482:483]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cndmask_b32_e64 v182 /*v438*/, v182 /*v438*/, 0xff61b1e6, s2
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[164:165] /*v[420:421]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[166:167] /*v[422:423]*/, s[42:43] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4589
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[40:47] /*v[296:303]*/, v[54:61] /*v[566:573]*/, 0
	s_set_vgpr_msb 0x8945
	v_pk_add_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, v[226:227] /*v[482:483]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[176:177] /*v[432:433]*/, v[176:177] /*v[432:433]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[434:435]*/, v[178:179] /*v[434:435]*/, s[42:43] op_sel_hi:[1,0]
	v_pk_mul_f32 v[180:181] /*v[436:437]*/, v[180:181] /*v[436:437]*/, s[42:43] op_sel_hi:[1,0]
	v_exp_f32_e32 v192 /*v448*/, v130 /*v386*/
	v_pk_mul_f32 v[182:183] /*v[438:439]*/, v[182:183] /*v[438:439]*/, s[42:43] op_sel_hi:[1,0]
	v_exp_f32_e32 v193 /*v449*/, v131 /*v387*/
	s_set_vgpr_msb 0x4589
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[64:71] /*v[320:327]*/, v[22:29] /*v[534:541]*/, 0
	s_set_vgpr_msb 0x8941
	v_exp_f32_e32 v194 /*v450*/, v132 /*v388*/
	v_exp_f32_e32 v195 /*v451*/, v133 /*v389*/
	v_exp_f32_e32 v196 /*v452*/, v134 /*v390*/
	v_exp_f32_e32 v197 /*v453*/, v135 /*v391*/
	v_exp_f32_e32 v198 /*v454*/, v128 /*v384*/
	v_exp_f32_e32 v199 /*v455*/, v129 /*v385*/
	v_exp_f32_e32 v160 /*v416*/, v160 /*v416*/
	s_set_vgpr_msb 0x4189
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[72:79] /*v[328:335]*/, v[54:61] /*v[566:573]*/, 0
	s_set_vgpr_msb 0x8959
	v_exp_f32_e32 v161 /*v417*/, v161 /*v417*/
	v_exp_f32_e32 v162 /*v418*/, v162 /*v418*/
	v_exp_f32_e32 v163 /*v419*/, v163 /*v419*/
	v_exp_f32_e32 v164 /*v420*/, v164 /*v420*/
	v_exp_f32_e32 v165 /*v421*/, v165 /*v421*/
	v_exp_f32_e32 v166 /*v422*/, v166 /*v422*/
	v_exp_f32_e32 v167 /*v423*/, v167 /*v423*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[48:55] /*v[304:311]*/, v[110:117] /*v[622:629]*/, v[208:215] /*v[464:471]*/
	v_exp_f32_e32 v176 /*v432*/, v176 /*v432*/
	v_exp_f32_e32 v177 /*v433*/, v177 /*v433*/
	v_exp_f32_e32 v178 /*v434*/, v178 /*v434*/
	v_exp_f32_e32 v179 /*v435*/, v179 /*v435*/
	v_exp_f32_e32 v180 /*v436*/, v180 /*v436*/
	v_exp_f32_e32 v181 /*v437*/, v181 /*v437*/
	v_exp_f32_e32 v182 /*v438*/, v182 /*v438*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[56:63] /*v[312:319]*/, v[126:133] /*v[638:645]*/, v[216:223] /*v[472:479]*/
	v_exp_f32_e32 v183 /*v439*/, v183 /*v439*/
	s_set_vgpr_msb 0x5945
	v_cvt_pk_bf16_f32 v131 /*v387*/, v142 /*v398*/, v143 /*v399*/
	v_cvt_pk_bf16_f32 v130 /*v386*/, v140 /*v396*/, v141 /*v397*/
	v_cvt_pk_bf16_f32 v129 /*v385*/, v138 /*v394*/, v139 /*v395*/
	v_cvt_pk_bf16_f32 v128 /*v384*/, v136 /*v392*/, v137 /*v393*/
	v_cvt_pk_bf16_f32 v135 /*v391*/, v196 /*v452*/, v197 /*v453*/
	v_cvt_pk_bf16_f32 v134 /*v390*/, v194 /*v450*/, v195 /*v451*/
	s_set_vgpr_msb 0x45a9
	v_wmma_f32_16x16x32_bf16 v[30:37] /*v[542:549]*/, v[48:55] /*v[304:311]*/, v[142:149] /*v[654:661]*/, v[30:37] /*v[542:549]*/
	s_set_vgpr_msb 0xa945
	v_pk_add_f32 v[144:145] /*v[400:401]*/, v[208:209] /*v[464:465]*/, v[216:217] /*v[472:473]*/
	v_pk_add_f32 v[146:147] /*v[402:403]*/, v[210:211] /*v[466:467]*/, v[218:219] /*v[474:475]*/
	v_pk_add_f32 v[148:149] /*v[404:405]*/, v[212:213] /*v[468:469]*/, v[220:221] /*v[476:477]*/
	v_pk_add_f32 v[150:151] /*v[406:407]*/, v[214:215] /*v[470:471]*/, v[222:223] /*v[478:479]*/
	v_cvt_pk_bf16_f32 v133 /*v389*/, v192 /*v448*/, v193 /*v449*/
	v_pk_add_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, v[232:233] /*v[488:489]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[146:147] /*v[402:403]*/, v[146:147] /*v[402:403]*/, v[232:233] /*v[488:489]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x45a9
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[56:63] /*v[312:319]*/, v[172:179] /*v[684:691]*/, v[62:69] /*v[574:581]*/
	s_set_vgpr_msb 0xa945
	v_pk_add_f32 v[148:149] /*v[404:405]*/, v[148:149] /*v[404:405]*/, v[232:233] /*v[488:489]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[150:151] /*v[406:407]*/, v[150:151] /*v[406:407]*/, v[232:233] /*v[488:489]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[144:145] /*v[400:401]*/, v[136:137] /*v[392:393]*/
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[146:147] /*v[402:403]*/, v[138:139] /*v[394:395]*/
	v_cvt_pk_bf16_f32 v132 /*v388*/, v198 /*v454*/, v199 /*v455*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[148:149] /*v[404:405]*/, v[140:141] /*v[396:397]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[150:151] /*v[406:407]*/, v[142:143] /*v[398:399]*/
	s_set_vgpr_msb 0x45a9
	v_wmma_f32_16x16x32_bf16 v[78:85] /*v[590:597]*/, v[80:87] /*v[336:343]*/, v[142:149] /*v[654:661]*/, v[78:85] /*v[590:597]*/
	s_set_vgpr_msb 0xa94a
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[30:31] /*v[542:543]*/, v[62:63] /*v[574:575]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[32:33] /*v[544:545]*/, v[64:65] /*v[576:577]*/
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[34:35] /*v[546:547]*/, v[66:67] /*v[578:579]*/
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[36:37] /*v[548:549]*/, v[68:69] /*v[580:581]*/
	s_set_vgpr_msb 0x4a45
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[152:153] /*v[408:409]*/, v[198:199] /*v[454:455]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[168:169] /*v[424:425]*/, v[230:231] /*v[486:487]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[170:171] /*v[426:427]*/, v[230:231] /*v[486:487]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x45a9
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[88:95] /*v[344:351]*/, v[172:179] /*v[684:691]*/, v[94:101] /*v[606:613]*/
	s_set_vgpr_msb 0xa945
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, v[230:231] /*v[486:487]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[230:231] /*v[486:487]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[196:197] /*v[452:453]*/
	v_pk_mul_f32 v[172:173] /*v[428:429]*/, v[172:173] /*v[428:429]*/, v[164:165] /*v[420:421]*/
	v_pk_mul_f32 v[174:175] /*v[430:431]*/, v[174:175] /*v[430:431]*/, v[166:167] /*v[422:423]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[184:185] /*v[440:441]*/, v[78:79] /*v[590:591]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[186:187] /*v[442:443]*/, v[80:81] /*v[592:593]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[82:83] /*v[594:595]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[190:191] /*v[446:447]*/, v[84:85] /*v[596:597]*/, v[100:101] /*v[612:613]*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v139 /*v395*/, v166 /*v422*/, v167 /*v423*/
	v_pk_add_f32 v[184:185] /*v[440:441]*/, v[184:185] /*v[440:441]*/, v[230:231] /*v[486:487]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[186:187] /*v[442:443]*/, v[186:187] /*v[442:443]*/, v[230:231] /*v[486:487]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[188:189] /*v[444:445]*/, v[188:189] /*v[444:445]*/, v[230:231] /*v[486:487]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[190:191] /*v[446:447]*/, v[190:191] /*v[446:447]*/, v[230:231] /*v[486:487]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_cvt_pk_bf16_f32 v138 /*v394*/, v164 /*v420*/, v165 /*v421*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[170:171] /*v[426:427]*/, v[162:163] /*v[418:419]*/
	v_cvt_pk_bf16_f32 v137 /*v393*/, v162 /*v418*/, v163 /*v419*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[168:169] /*v[424:425]*/, v[160:161] /*v[416:417]*/
	v_cvt_pk_bf16_f32 v136 /*v392*/, v160 /*v416*/, v161 /*v417*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[188:189] /*v[444:445]*/, v[180:181] /*v[436:437]*/
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[190:191] /*v[446:447]*/, v[182:183] /*v[438:439]*/
	v_pk_mul_f32 v[168:169] /*v[424:425]*/, v[186:187] /*v[442:443]*/, v[178:179] /*v[434:435]*/
	v_pk_mul_f32 v[170:171] /*v[426:427]*/, v[184:185] /*v[440:441]*/, v[176:177] /*v[432:433]*/
	ds_store_b128 v227 /*v483*/, v[128:131] /*v[384:387]*/
	ds_store_b128 v227 /*v483*/, v[132:135] /*v[388:391]*/ offset:32
	v_pk_mul_f32 v[128:129] /*v[384:385]*/, v[224:225] /*v[480:481]*/, v[144:145] /*v[400:401]*/
	v_pk_mul_f32 v[130:131] /*v[386:387]*/, v[224:225] /*v[480:481]*/, v[146:147] /*v[402:403]*/
	v_pk_mul_f32 v[132:133] /*v[388:389]*/, v[224:225] /*v[480:481]*/, v[148:149] /*v[404:405]*/
	v_pk_mul_f32 v[134:135] /*v[390:391]*/, v[224:225] /*v[480:481]*/, v[150:151] /*v[406:407]*/
	v_pk_mul_f32 v[144:145] /*v[400:401]*/, v[224:225] /*v[480:481]*/, v[152:153] /*v[408:409]*/
	v_pk_mul_f32 v[146:147] /*v[402:403]*/, v[224:225] /*v[480:481]*/, v[154:155] /*v[410:411]*/
	v_pk_mul_f32 v[148:149] /*v[404:405]*/, v[224:225] /*v[480:481]*/, v[156:157] /*v[412:413]*/
	v_pk_mul_f32 v[150:151] /*v[406:407]*/, v[224:225] /*v[480:481]*/, v[158:159] /*v[414:415]*/
	v_pk_mul_f32 v[152:153] /*v[408:409]*/, v[224:225] /*v[480:481]*/, v[162:163] /*v[418:419]*/
	v_pk_mul_f32 v[154:155] /*v[410:411]*/, v[224:225] /*v[480:481]*/, v[164:165] /*v[420:421]*/
	v_pk_mul_f32 v[156:157] /*v[412:413]*/, v[224:225] /*v[480:481]*/, v[172:173] /*v[428:429]*/
	v_pk_mul_f32 v[158:159] /*v[414:415]*/, v[224:225] /*v[480:481]*/, v[174:175] /*v[430:431]*/
	v_pk_mul_f32 v[162:163] /*v[418:419]*/, v[224:225] /*v[480:481]*/, v[170:171] /*v[426:427]*/
	v_pk_mul_f32 v[164:165] /*v[420:421]*/, v[224:225] /*v[480:481]*/, v[168:169] /*v[424:425]*/
	v_pk_mul_f32 v[160:161] /*v[416:417]*/, v[224:225] /*v[480:481]*/, v[160:161] /*v[416:417]*/
	v_pk_mul_f32 v[166:167] /*v[422:423]*/, v[224:225] /*v[480:481]*/, v[166:167] /*v[422:423]*/
	v_cvt_pk_bf16_f32 v128 /*v384*/, v128 /*v384*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v129 /*v385*/, v130 /*v386*/, v131 /*v387*/
	v_cvt_pk_bf16_f32 v130 /*v386*/, v132 /*v388*/, v133 /*v389*/
	v_cvt_pk_bf16_f32 v131 /*v387*/, v134 /*v390*/, v135 /*v391*/
	v_cvt_pk_bf16_f32 v132 /*v388*/, v144 /*v400*/, v145 /*v401*/
	v_cvt_pk_bf16_f32 v133 /*v389*/, v146 /*v402*/, v147 /*v403*/
	v_cvt_pk_bf16_f32 v134 /*v390*/, v148 /*v404*/, v149 /*v405*/
	v_cvt_pk_bf16_f32 v135 /*v391*/, v150 /*v406*/, v151 /*v407*/
	v_cvt_pk_bf16_f32 v143 /*v399*/, v182 /*v438*/, v183 /*v439*/
	v_cvt_pk_bf16_f32 v142 /*v398*/, v180 /*v436*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v141 /*v397*/, v178 /*v434*/, v179 /*v435*/
	v_cvt_pk_bf16_f32 v140 /*v396*/, v176 /*v432*/, v177 /*v433*/
	v_cvt_pk_bf16_f32 v144 /*v400*/, v152 /*v408*/, v153 /*v409*/
	v_cvt_pk_bf16_f32 v145 /*v401*/, v154 /*v410*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v146 /*v402*/, v156 /*v412*/, v157 /*v413*/
	v_cvt_pk_bf16_f32 v147 /*v403*/, v158 /*v414*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v148 /*v404*/, v162 /*v418*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v149 /*v405*/, v164 /*v420*/, v165 /*v421*/
	v_cvt_pk_bf16_f32 v150 /*v406*/, v160 /*v416*/, v161 /*v417*/
	v_cvt_pk_bf16_f32 v151 /*v407*/, v166 /*v422*/, v167 /*v423*/
	ds_store_b128 v227 /*v483*/, v[128:131] /*v[384:387]*/ offset:8704
	ds_store_b128 v227 /*v483*/, v[132:135] /*v[388:391]*/ offset:8736
	s_set_vgpr_msb 0x450b
	ds_store_b128 v37 /*v805*/, v[22:25] /*v[534:537]*/
	ds_store_b128 v37 /*v805*/, v[26:29] /*v[538:541]*/ offset:32
	s_set_vgpr_msb 0xb07
	ds_store_b128 v37 /*v805*/, v[200:203] /*v[456:459]*/ offset:8704
	ds_store_b128 v37 /*v805*/, v[204:207] /*v[460:463]*/ offset:8736
	s_set_vgpr_msb 0x70b
	ds_store_b128 v37 /*v805*/, v[54:57] /*v[566:569]*/ offset:64
	ds_store_b128 v37 /*v805*/, v[58:61] /*v[570:573]*/ offset:96
	ds_store_b128 v37 /*v805*/, v[38:41] /*v[550:553]*/ offset:8768
	ds_store_b128 v37 /*v805*/, v[42:45] /*v[554:557]*/ offset:8800
	ds_store_b128 v37 /*v805*/, v[142:145] /*v[654:657]*/ offset:128
	ds_store_b128 v37 /*v805*/, v[146:149] /*v[658:661]*/ offset:160
	ds_store_b128 v37 /*v805*/, v[134:137] /*v[646:649]*/ offset:8832
	ds_store_b128 v37 /*v805*/, v[138:141] /*v[650:653]*/ offset:8864
	ds_store_b128 v37 /*v805*/, v[172:175] /*v[684:687]*/ offset:192
	ds_store_b128 v37 /*v805*/, v[176:179] /*v[688:691]*/ offset:224
	ds_store_b128 v37 /*v805*/, v[150:153] /*v[662:665]*/ offset:8896
	ds_store_b128 v37 /*v805*/, v[154:157] /*v[666:669]*/ offset:8928
	s_set_vgpr_msb 0xb55
	ds_store_b128 v229 /*v485*/, v[136:139] /*v[392:395]*/
	ds_store_b128 v229 /*v485*/, v[140:143] /*v[396:399]*/ offset:32
	ds_store_b128 v229 /*v485*/, v[144:147] /*v[400:403]*/ offset:8704
	ds_store_b128 v229 /*v485*/, v[148:151] /*v[404:407]*/ offset:8736
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_cmp_lg_u64 s[38:39], s[44:45]
	s_barrier_wait -1
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v231 /*v487*/
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v231 /*v487*/ offset:4352
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v248 /*v504*/
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v248 /*v504*/ offset:4352
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v234 /*v490*/
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v234 /*v490*/ offset:4352
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v236 /*v492*/
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v236 /*v492*/ offset:4352
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v238 /*v494*/
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v238 /*v494*/ offset:4352
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v240 /*v496*/
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v240 /*v496*/ offset:4352
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v242 /*v498*/
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v242 /*v498*/ offset:4352
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v244 /*v500*/
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v244 /*v500*/ offset:4352
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v246 /*v502*/
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v246 /*v502*/ offset:4352
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[136:143] /*v[392:399]*/, v[128:135] /*v[384:391]*/, v[104:111] /*v[360:367]*/
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v235 /*v491*/
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v235 /*v491*/ offset:4352
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v237 /*v493*/
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v237 /*v493*/ offset:4352
	s_set_vgpr_msb 0x5581
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v239 /*v495*/
	ds_load_tr16_b128 v[10:13] /*v[522:525]*/, v239 /*v495*/ offset:4352
	ds_load_tr16_b128 v[14:17] /*v[526:529]*/, v241 /*v497*/
	ds_load_tr16_b128 v[18:21] /*v[530:533]*/, v241 /*v497*/ offset:4352
	ds_load_tr16_b128 v[22:25] /*v[534:537]*/, v243 /*v499*/
	ds_load_tr16_b128 v[26:29] /*v[538:541]*/, v243 /*v499*/ offset:4352
	ds_load_tr16_b128 v[30:33] /*v[542:545]*/, v245 /*v501*/
	ds_load_tr16_b128 v[34:37] /*v[546:549]*/, v245 /*v501*/ offset:4352
	ds_load_tr16_b128 v[38:41] /*v[550:553]*/, v247 /*v503*/
	ds_load_tr16_b128 v[42:45] /*v[554:557]*/, v247 /*v503*/ offset:4352
	s_set_vgpr_msb 0x8155
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[136:143] /*v[392:399]*/, v[144:151] /*v[400:407]*/, v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5505
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[200:207], v[136:143] /*v[392:399]*/, v[152:159] /*v[408:415]*/, v[200:207]
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[184:191], v[136:143] /*v[392:399]*/, v[160:167] /*v[416:423]*/, v[184:191]
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[168:175], v[136:143] /*v[392:399]*/, v[168:175] /*v[424:431]*/, v[168:175]
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[152:159], v[136:143] /*v[392:399]*/, v[176:183] /*v[432:439]*/, v[152:159]
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[136:143], v[136:143] /*v[392:399]*/, v[184:191] /*v[440:447]*/, v[136:143]
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[128:135], v[136:143] /*v[392:399]*/, v[192:199] /*v[448:455]*/, v[128:135]
	s_set_vgpr_msb 0x555
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v233 /*v489*/
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v233 /*v489*/ offset:4352
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v249 /*v505*/
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v249 /*v505*/ offset:4352
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[200:207] /*v[456:463]*/, v[136:143] /*v[392:399]*/, v[120:127] /*v[376:383]*/
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[200:207] /*v[456:463]*/, v[208:215] /*v[464:471]*/, v[112:119] /*v[368:375]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[200:207] /*v[456:463]*/, v[216:223] /*v[472:479]*/, v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[208:215], v[200:207] /*v[456:463]*/, v[6:13] /*v[518:525]*/, v[208:215]
	v_wmma_f32_16x16x32_bf16 v[192:199], v[200:207] /*v[456:463]*/, v[14:21] /*v[526:533]*/, v[192:199]
	v_wmma_f32_16x16x32_bf16 v[176:183], v[200:207] /*v[456:463]*/, v[22:29] /*v[534:541]*/, v[176:183]
	v_wmma_f32_16x16x32_bf16 v[160:167], v[200:207] /*v[456:463]*/, v[30:37] /*v[542:549]*/, v[160:167]
	v_wmma_f32_16x16x32_bf16 v[144:151], v[200:207] /*v[456:463]*/, v[38:45] /*v[550:557]*/, v[144:151]
	s_set_vgpr_msb 0x941
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v250 /*v506*/
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v250 /*v506*/ offset:4352
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[112:119], v[200:207] /*v[456:463]*/, v[128:135] /*v[384:391]*/, v[112:119]
	s_set_vgpr_msb 0x541
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v251 /*v507*/
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v251 /*v507*/ offset:4352
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_wmma_f32_16x16x32_bf16 v[120:127], v[128:135] /*v[384:391]*/, v[136:143] /*v[392:399]*/, v[120:127]
	s_barrier_wait -1
	v_wmma_f32_16x16x32_bf16 v[88:95], v[200:207] /*v[456:463]*/, v[144:151] /*v[400:407]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[128:135] /*v[384:391]*/, v[208:215] /*v[464:471]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[200:207] /*v[456:463]*/, v[152:159] /*v[408:415]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[128:135] /*v[384:391]*/, v[216:223] /*v[472:479]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[200:207] /*v[456:463]*/, v[160:167] /*v[416:423]*/, v[56:63]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[80:87], v[128:135] /*v[384:391]*/, v[6:13] /*v[518:525]*/, v[80:87]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[40:47], v[200:207] /*v[456:463]*/, v[168:175] /*v[424:431]*/, v[40:47]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[64:71], v[128:135] /*v[384:391]*/, v[14:21] /*v[526:533]*/, v[64:71]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[24:31], v[200:207] /*v[456:463]*/, v[176:183] /*v[432:439]*/, v[24:31]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[22:29] /*v[534:541]*/, v[48:55]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[8:15], v[200:207] /*v[456:463]*/, v[184:191] /*v[440:447]*/, v[8:15]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[128:135] /*v[384:391]*/, v[30:37] /*v[542:549]*/, v[32:39]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207] /*v[456:463]*/, v[192:199] /*v[448:455]*/, v[0:7]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[16:23], v[128:135] /*v[384:391]*/, v[38:45] /*v[550:557]*/, v[16:23]
	s_set_vgpr_msb 0x900
	s_cbranch_scc1 .LBB0_4
	s_branch .LBB0_6
.LBB0_5:
	v_mov_b32_e32 v16, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v17, v16 :: v_dual_mov_b32 v18, v16
	v_dual_mov_b32 v19, v16 :: v_dual_mov_b32 v20, v16
	v_dual_mov_b32 v21, v16 :: v_dual_mov_b32 v22, v16
	v_mov_b32_e32 v23, v16
	v_mov_b64_e32 v[34:35], v[18:19]
	v_mov_b64_e32 v[32:33], v[16:17]
	s_delay_alu instid0(VALU_DEP_4)
	v_mov_b64_e32 v[36:37], v[20:21]
	v_mov_b64_e32 v[52:53], v[20:21]
	v_mov_b64_e32 v[38:39], v[22:23]
	v_mov_b64_e32 v[54:55], v[22:23]
	v_mov_b64_e32 v[50:51], v[18:19]
	v_mov_b64_e32 v[48:49], v[16:17]
	v_mov_b64_e32 v[70:71], v[22:23]
	v_mov_b64_e32 v[68:69], v[20:21]
	v_mov_b64_e32 v[66:67], v[18:19]
	v_mov_b64_e32 v[64:65], v[16:17]
	v_mov_b64_e32 v[86:87], v[22:23]
	v_mov_b64_e32 v[84:85], v[20:21]
	v_mov_b64_e32 v[82:83], v[18:19]
	v_mov_b64_e32 v[80:81], v[16:17]
	v_mov_b64_e32 v[102:103], v[22:23]
	v_mov_b64_e32 v[100:101], v[20:21]
	v_mov_b64_e32 v[98:99], v[18:19]
	v_mov_b64_e32 v[96:97], v[16:17]
	v_mov_b64_e32 v[110:111], v[22:23]
	v_mov_b64_e32 v[108:109], v[20:21]
	v_mov_b64_e32 v[106:107], v[18:19]
	v_mov_b64_e32 v[104:105], v[16:17]
	v_mov_b64_e32 v[126:127], v[22:23]
	v_mov_b64_e32 v[124:125], v[20:21]
	v_mov_b64_e32 v[122:123], v[18:19]
	v_mov_b64_e32 v[120:121], v[16:17]
	v_mov_b64_e32 v[150:151], v[22:23]
	v_mov_b64_e32 v[148:149], v[20:21]
	v_mov_b64_e32 v[146:147], v[18:19]
	v_mov_b64_e32 v[144:145], v[16:17]
	v_mov_b64_e32 v[166:167], v[22:23]
	v_mov_b64_e32 v[164:165], v[20:21]
	v_mov_b64_e32 v[162:163], v[18:19]
	v_mov_b64_e32 v[160:161], v[16:17]
	v_mov_b64_e32 v[182:183], v[22:23]
	v_mov_b64_e32 v[180:181], v[20:21]
	v_mov_b64_e32 v[178:179], v[18:19]
	v_mov_b64_e32 v[176:177], v[16:17]
	v_mov_b64_e32 v[198:199], v[22:23]
	v_mov_b64_e32 v[196:197], v[20:21]
	v_mov_b64_e32 v[194:195], v[18:19]
	v_mov_b64_e32 v[192:193], v[16:17]
	v_mov_b64_e32 v[214:215], v[22:23]
	v_mov_b64_e32 v[212:213], v[20:21]
	v_mov_b64_e32 v[210:211], v[18:19]
	v_mov_b64_e32 v[208:209], v[16:17]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[22:23]
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[20:21]
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[18:19]
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, v[16:17]
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[22:23]
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[20:21]
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[18:19]
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[16:17]
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[22:23]
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[20:21]
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[18:19]
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[16:17]
	s_set_vgpr_msb 0x4000
	v_mov_b64_e32 v[0:1], v[16:17]
	v_mov_b64_e32 v[2:3], v[18:19]
	v_mov_b64_e32 v[4:5], v[20:21]
	v_mov_b64_e32 v[6:7], v[22:23]
	v_mov_b64_e32 v[8:9], v[16:17]
	v_mov_b64_e32 v[10:11], v[18:19]
	v_mov_b64_e32 v[12:13], v[20:21]
	v_mov_b64_e32 v[14:15], v[22:23]
	v_mov_b64_e32 v[30:31], v[22:23]
	v_mov_b64_e32 v[28:29], v[20:21]
	v_mov_b64_e32 v[26:27], v[18:19]
	v_mov_b64_e32 v[24:25], v[16:17]
	v_mov_b64_e32 v[46:47], v[22:23]
	v_mov_b64_e32 v[44:45], v[20:21]
	v_mov_b64_e32 v[42:43], v[18:19]
	v_mov_b64_e32 v[40:41], v[16:17]
	v_mov_b64_e32 v[62:63], v[22:23]
	v_mov_b64_e32 v[60:61], v[20:21]
	v_mov_b64_e32 v[58:59], v[18:19]
	v_mov_b64_e32 v[56:57], v[16:17]
	v_mov_b64_e32 v[78:79], v[22:23]
	v_mov_b64_e32 v[76:77], v[20:21]
	v_mov_b64_e32 v[74:75], v[18:19]
	v_mov_b64_e32 v[72:73], v[16:17]
	v_mov_b64_e32 v[94:95], v[22:23]
	v_mov_b64_e32 v[92:93], v[20:21]
	v_mov_b64_e32 v[90:91], v[18:19]
	v_mov_b64_e32 v[88:89], v[16:17]
	v_mov_b64_e32 v[118:119], v[22:23]
	v_mov_b64_e32 v[116:117], v[20:21]
	v_mov_b64_e32 v[114:115], v[18:19]
	v_mov_b64_e32 v[112:113], v[16:17]
	v_mov_b64_e32 v[134:135], v[22:23]
	v_mov_b64_e32 v[132:133], v[20:21]
	v_mov_b64_e32 v[130:131], v[18:19]
	v_mov_b64_e32 v[128:129], v[16:17]
	v_mov_b64_e32 v[142:143], v[22:23]
	v_mov_b64_e32 v[140:141], v[20:21]
	v_mov_b64_e32 v[138:139], v[18:19]
	v_mov_b64_e32 v[136:137], v[16:17]
	v_mov_b64_e32 v[158:159], v[22:23]
	v_mov_b64_e32 v[156:157], v[20:21]
	v_mov_b64_e32 v[154:155], v[18:19]
	v_mov_b64_e32 v[152:153], v[16:17]
	v_mov_b64_e32 v[174:175], v[22:23]
	v_mov_b64_e32 v[172:173], v[20:21]
	v_mov_b64_e32 v[170:171], v[18:19]
	v_mov_b64_e32 v[168:169], v[16:17]
	v_mov_b64_e32 v[190:191], v[22:23]
	v_mov_b64_e32 v[188:189], v[20:21]
	v_mov_b64_e32 v[186:187], v[18:19]
	v_mov_b64_e32 v[184:185], v[16:17]
	v_mov_b64_e32 v[206:207], v[22:23]
	v_mov_b64_e32 v[204:205], v[20:21]
	v_mov_b64_e32 v[202:203], v[18:19]
	v_mov_b64_e32 v[200:201], v[16:17]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[22:23]
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[20:21]
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[18:19]
	v_mov_b64_e32 v[32:33] /*v[288:289]*/, v[16:17]
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[22:23]
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[20:21]
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[18:19]
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[16:17]
	s_set_vgpr_msb 0x4000
.LBB0_6:
	s_add_co_i32 s2, s41, 1
	s_mov_b32 s62, s58
	s_cmp_lt_u32 s2, 3
	s_mov_b32 s63, s59
	s_cselect_b32 s2, s41, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s3, s2, s41
	s_cmp_lg_u32 s3, 1
	s_cselect_b32 s3, -1, 0
	s_cmp_lt_i32 s41, 0
	s_cselect_b32 s4, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s3, s4, s3
	s_sub_co_ci_u32 s2, s2, 0
	s_add_co_i32 s9, s67, s69
	s_add_co_i32 s10, s71, s70
	s_lshl_b32 s3, s9, 5
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x4d
	v_or_b32_e32 v128 /*v384*/, s3, v31 /*v799*/
	v_or3_b32 v129 /*v385*/, s3, v30 /*v798*/, 16
	s_mul_i32 s3, s10, s37
	s_delay_alu instid0(VALU_DEP_2) | instid1(SALU_CYCLE_1)
	v_add_lshl_u32 v130 /*v386*/, v128 /*v384*/, s3, 2
	s_delay_alu instid0(VALU_DEP_2)
	v_add_lshl_u32 v131 /*v387*/, v129 /*v385*/, s3, 2
	s_mul_i32 s3, s37, s68
	s_set_vgpr_msb 0x4d81
	s_clause 0x1
	buffer_load_b32 v6 /*v518*/, v130 /*v386*/, s[56:59], null offen
	buffer_load_b32 v4 /*v516*/, v131 /*v387*/, s[56:59], null offen
	s_clause 0x1
	buffer_load_b32 v2 /*v514*/, v130 /*v386*/, s[60:63], null offen
	buffer_load_b32 v0 /*v512*/, v131 /*v387*/, s[60:63], null offen
	s_mul_i32 s3, s3, s35
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl4_add_u32 s11, s71, s3
	s_set_vgpr_msb 0x8141
	v_mad_u32 v128 /*v384*/, v128 /*v384*/, s35, s11
	v_mad_u32 v129 /*v385*/, v129 /*v385*/, s35, s11
	s_sub_co_i32 s4, s64, s2
	s_add_co_i32 s2, s9, s2
	s_mul_i32 s4, s4, s41
	s_lshl_b32 s2, s2, 5
	s_add_co_i32 s4, s4, 1
	s_set_vgpr_msb 0x418c
	v_or_b32_e32 v7 /*v519*/, s2, v31 /*v799*/
	s_set_vgpr_msb 0x8c4d
	v_or_b32_e32 v128 /*v384*/, v128 /*v384*/, v33 /*v801*/
	v_or_b32_e32 v129 /*v385*/, v129 /*v385*/, v33 /*v801*/
	s_add_co_i32 s5, s4, s70
	s_set_vgpr_msb 0x4d8c
	v_or3_b32 v8 /*v520*/, s2, v30 /*v798*/, 16
	s_mov_b32 s50, s54
	s_set_vgpr_msb 0x8c44
	v_dual_lshlrev_b32 v128 /*v384*/, 4, v128 /*v384*/ :: v_dual_lshlrev_b32 v160 /*v416*/, 4, v129 /*v385*/
	s_mov_b32 s51, s55
	s_mul_i32 s5, s5, s37
	s_set_vgpr_msb 0x4482
	v_add_lshl_u32 v9 /*v521*/, v7 /*v519*/, s5, 2
	s_set_vgpr_msb 0x8245
	v_or_b32_e32 v129 /*v385*/, 32, v128 /*v384*/
	v_or_b32_e32 v130 /*v386*/, 64, v128 /*v384*/
	v_or_b32_e32 v131 /*v387*/, 0x60, v128 /*v384*/
	v_or_b32_e32 v132 /*v388*/, 0x80, v128 /*v384*/
	v_or_b32_e32 v162 /*v418*/, 64, v160 /*v416*/
	v_or_b32_e32 v164 /*v420*/, 0x80, v160 /*v416*/
	v_or_b32_e32 v133 /*v389*/, 0xa0, v128 /*v384*/
	v_or_b32_e32 v134 /*v390*/, 0xc0, v128 /*v384*/
	v_or_b32_e32 v135 /*v391*/, 0xe0, v128 /*v384*/
	s_clause 0x7
	buffer_load_b128 v[248:251] /*v[504:507]*/, v128 /*v384*/, s[52:55], null offen
	buffer_load_b128 v[252:255] /*v[508:511]*/, v129 /*v385*/, s[52:55], null offen
	buffer_load_b128 v[240:243] /*v[496:499]*/, v130 /*v386*/, s[52:55], null offen
	buffer_load_b128 v[244:247] /*v[500:503]*/, v131 /*v387*/, s[52:55], null offen
	buffer_load_b128 v[232:235] /*v[488:491]*/, v132 /*v388*/, s[52:55], null offen
	buffer_load_b128 v[236:239] /*v[492:495]*/, v133 /*v389*/, s[52:55], null offen
	buffer_load_b128 v[224:227] /*v[480:483]*/, v134 /*v390*/, s[52:55], null offen
	buffer_load_b128 v[228:231] /*v[484:487]*/, v135 /*v391*/, s[52:55], null offen
	s_clause 0x3
	buffer_load_b128 v[216:219] /*v[472:475]*/, v128 /*v384*/, s[48:51], null offen
	buffer_load_b128 v[220:223] /*v[476:479]*/, v129 /*v385*/, s[48:51], null offen
	buffer_load_b128 v[208:211] /*v[464:467]*/, v130 /*v386*/, s[48:51], null offen
	buffer_load_b128 v[212:215] /*v[468:471]*/, v131 /*v387*/, s[48:51], null offen
	v_or_b32_e32 v161 /*v417*/, 32, v160 /*v416*/
	v_or_b32_e32 v163 /*v419*/, 0x60, v160 /*v416*/
	v_or_b32_e32 v165 /*v421*/, 0xa0, v160 /*v416*/
	v_or_b32_e32 v166 /*v422*/, 0xc0, v160 /*v416*/
	v_or_b32_e32 v167 /*v423*/, 0xe0, v160 /*v416*/
	s_clause 0x3
	buffer_load_b128 v[200:203] /*v[456:459]*/, v132 /*v388*/, s[48:51], null offen
	buffer_load_b128 v[204:207] /*v[460:463]*/, v133 /*v389*/, s[48:51], null offen
	buffer_load_b128 v[192:195] /*v[448:451]*/, v134 /*v390*/, s[48:51], null offen
	buffer_load_b128 v[196:199] /*v[452:455]*/, v135 /*v391*/, s[48:51], null offen
	s_clause 0x7
	buffer_load_b128 v[136:139] /*v[392:395]*/, v160 /*v416*/, s[52:55], null offen
	buffer_load_b128 v[140:143] /*v[396:399]*/, v161 /*v417*/, s[52:55], null offen
	buffer_load_b128 v[152:155] /*v[408:411]*/, v162 /*v418*/, s[52:55], null offen
	buffer_load_b128 v[156:159] /*v[412:415]*/, v163 /*v419*/, s[52:55], null offen
	buffer_load_b128 v[144:147] /*v[400:403]*/, v164 /*v420*/, s[52:55], null offen
	buffer_load_b128 v[148:151] /*v[404:407]*/, v165 /*v421*/, s[52:55], null offen
	buffer_load_b128 v[128:131] /*v[384:387]*/, v166 /*v422*/, s[52:55], null offen
	buffer_load_b128 v[132:135] /*v[388:391]*/, v167 /*v423*/, s[52:55], null offen
	s_clause 0x3
	buffer_load_b128 v[168:171] /*v[424:427]*/, v160 /*v416*/, s[48:51], null offen
	buffer_load_b128 v[172:175] /*v[428:431]*/, v161 /*v417*/, s[48:51], null offen
	buffer_load_b128 v[184:187] /*v[440:443]*/, v162 /*v418*/, s[48:51], null offen
	buffer_load_b128 v[188:191] /*v[444:447]*/, v163 /*v419*/, s[48:51], null offen
	s_set_vgpr_msb 0x4582
	v_add_lshl_u32 v10 /*v522*/, v8 /*v520*/, s5, 2
	s_set_vgpr_msb 0x8241
	s_clause 0x3
	buffer_load_b128 v[176:179] /*v[432:435]*/, v164 /*v420*/, s[48:51], null offen
	buffer_load_b128 v[180:183] /*v[436:439]*/, v165 /*v421*/, s[48:51], null offen
	buffer_load_b128 v[160:163] /*v[416:419]*/, v166 /*v422*/, s[48:51], null offen
	buffer_load_b128 v[164:167] /*v[420:423]*/, v167 /*v423*/, s[48:51], null offen
	s_set_vgpr_msb 0x41c2
	s_clause 0x1
	buffer_load_b32 v41 /*v809*/, v9 /*v521*/, s[56:59], null offen
	buffer_load_b32 v40 /*v808*/, v10 /*v522*/, s[56:59], null offen
	s_clause 0x1
	buffer_load_b32 v43 /*v811*/, v9 /*v521*/, s[60:63], null offen
	buffer_load_b32 v42 /*v810*/, v10 /*v522*/, s[60:63], null offen
	s_sub_co_i32 s2, s66, s67
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s2, s2, s41
	s_cmp_lt_i32 s2, 1
	s_set_vgpr_msb 0xc200
	s_cbranch_scc1 .LBB0_9
	s_lshl4_add_u32 s3, s4, s3
	s_abs_i32 s12, s41
	s_set_vgpr_msb 0x82
	v_mad_u32 v8 /*v520*/, v8 /*v520*/, s35, s3
	v_mad_u32 v7 /*v519*/, v7 /*v519*/, s35, s3
	s_cvt_f32_u32 s5, s12
	s_movk_i32 s4, 0x2200
	s_set_vgpr_msb 0x82cb
	v_or_b32_e32 v54 /*v822*/, v38 /*v806*/, v1 /*v513*/
	v_mad_u32_u24 v46 /*v814*/, 0x110, v3 /*v515*/, s4
	v_s_rcp_f32 s4, s5
	s_mov_b32 s5, 0x12200
	s_set_vgpr_msb 0xcb8e
	v_or_b32_e32 v8 /*v520*/, v8 /*v520*/, v33 /*v801*/
	v_or_b32_e32 v7 /*v519*/, v7 /*v519*/, v33 /*v801*/
	s_set_vgpr_msb 0x8ec8
	v_mad_u32_u24 v56 /*v824*/, 0x110, v3 /*v515*/, s5
	s_sub_co_i32 s5, 0, s12
	v_or_b32_e32 v47 /*v815*/, 32, v5 /*v517*/
	s_set_vgpr_msb 0xc888
	v_dual_lshlrev_b32 v8 /*v520*/, 4, v8 /*v520*/ :: v_dual_lshlrev_b32 v7 /*v519*/, 4, v7 /*v519*/
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_set_vgpr_msb 0x88cc
	v_or_b32_e32 v48 /*v816*/, 64, v38 /*v806*/
	s_set_vgpr_msb 0xccc8
	v_or_b32_e32 v49 /*v817*/, 0x60, v5 /*v517*/
	s_set_vgpr_msb 0xc88a
	v_or_b32_e32 v10 /*v522*/, 0xc0, v8 /*v520*/
	v_or_b32_e32 v11 /*v523*/, 0xa0, v8 /*v520*/
	v_or_b32_e32 v12 /*v524*/, 0x80, v8 /*v520*/
	v_or_b32_e32 v13 /*v525*/, 0x60, v8 /*v520*/
	v_or_b32_e32 v9 /*v521*/, 0xe0, v8 /*v520*/
	v_or_b32_e32 v14 /*v526*/, 64, v8 /*v520*/
	v_or_b32_e32 v15 /*v527*/, 32, v8 /*v520*/
	s_clause 0x7
	buffer_load_b128 v[40:43] /*v[552:555]*/, v10 /*v522*/, s[48:51], null offen
	buffer_load_b128 v[36:39] /*v[548:551]*/, v11 /*v523*/, s[48:51], null offen
	buffer_load_b128 v[32:35] /*v[544:547]*/, v12 /*v524*/, s[48:51], null offen
	buffer_load_b128 v[84:87] /*v[596:599]*/, v13 /*v525*/, s[48:51], null offen
	buffer_load_b128 v[72:75] /*v[584:587]*/, v14 /*v526*/, s[48:51], null offen
	buffer_load_b128 v[64:67] /*v[576:579]*/, v15 /*v527*/, s[48:51], null offen
	buffer_load_b128 v[44:47] /*v[556:559]*/, v9 /*v521*/, s[48:51], null offen
	buffer_load_b128 v[60:63] /*v[572:575]*/, v8 /*v520*/, s[48:51], null offen
	s_clause 0x7
	buffer_load_b128 v[56:59] /*v[568:571]*/, v10 /*v522*/, s[52:55], null offen
	buffer_load_b128 v[68:71] /*v[580:583]*/, v11 /*v523*/, s[52:55], null offen
	buffer_load_b128 v[76:79] /*v[588:591]*/, v12 /*v524*/, s[52:55], null offen
	buffer_load_b128 v[88:91] /*v[600:603]*/, v13 /*v525*/, s[52:55], null offen
	buffer_load_b128 v[92:95] /*v[604:607]*/, v14 /*v526*/, s[52:55], null offen
	buffer_load_b128 v[48:51] /*v[560:563]*/, v15 /*v527*/, s[52:55], null offen
	buffer_load_b128 v[80:83] /*v[592:595]*/, v9 /*v521*/, s[52:55], null offen
	buffer_load_b128 v[52:55] /*v[564:567]*/, v8 /*v520*/, s[52:55], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v9 /*v521*/, 0xc0, v7 /*v519*/
	v_or_b32_e32 v10 /*v522*/, 0xa0, v7 /*v519*/
	v_or_b32_e32 v11 /*v523*/, 0x80, v7 /*v519*/
	v_or_b32_e32 v12 /*v524*/, 0x60, v7 /*v519*/
	v_or_b32_e32 v13 /*v525*/, 64, v7 /*v519*/
	s_wait_xcnt 0x0
	v_or_b32_e32 v8 /*v520*/, 0xe0, v7 /*v519*/
	v_or_b32_e32 v14 /*v526*/, 32, v7 /*v519*/
	s_clause 0x7
	buffer_load_b128 v[96:99] /*v[608:611]*/, v9 /*v521*/, s[48:51], null offen
	buffer_load_b128 v[108:111] /*v[620:623]*/, v10 /*v522*/, s[48:51], null offen
	buffer_load_b128 v[112:115] /*v[624:627]*/, v11 /*v523*/, s[48:51], null offen
	buffer_load_b128 v[100:103] /*v[612:615]*/, v12 /*v524*/, s[48:51], null offen
	buffer_load_b128 v[104:107] /*v[616:619]*/, v13 /*v525*/, s[48:51], null offen
	buffer_load_b128 v[120:123] /*v[632:635]*/, v14 /*v526*/, s[48:51], null offen
	buffer_load_b128 v[116:119] /*v[628:631]*/, v8 /*v520*/, s[48:51], null offen
	buffer_load_b128 v[124:127] /*v[636:639]*/, v7 /*v519*/, s[48:51], null offen
	s_clause 0x7
	buffer_load_b128 v[128:131] /*v[640:643]*/, v9 /*v521*/, s[52:55], null offen
	buffer_load_b128 v[132:135] /*v[644:647]*/, v10 /*v522*/, s[52:55], null offen
	buffer_load_b128 v[136:139] /*v[648:651]*/, v11 /*v523*/, s[52:55], null offen
	buffer_load_b128 v[140:143] /*v[652:655]*/, v12 /*v524*/, s[52:55], null offen
	buffer_load_b128 v[144:147] /*v[656:659]*/, v13 /*v525*/, s[52:55], null offen
	buffer_load_b128 v[152:155] /*v[664:667]*/, v14 /*v526*/, s[52:55], null offen
	buffer_load_b128 v[148:151] /*v[660:663]*/, v8 /*v520*/, s[52:55], null offen
	buffer_load_b128 v[156:159] /*v[668:671]*/, v7 /*v519*/, s[52:55], null offen
	s_set_vgpr_msb 0x8a8b
	v_dual_add_nc_u32 v7 /*v519*/, v36 /*v804*/, v1 /*v513*/ :: v_dual_add_nc_u32 v8 /*v520*/, v37 /*v805*/, v1 /*v513*/
	s_cvt_u32_f32 s6, s4
	s_set_vgpr_msb 0x8bcc
	v_or_b32_e32 v50 /*v818*/, 0x80, v38 /*v806*/
	s_set_vgpr_msb 0xccc8
	v_or_b32_e32 v51 /*v819*/, 0xa0, v5 /*v517*/
	v_or_b32_e32 v44 /*v812*/, 0x10000, v7 /*v519*/
	s_mul_i32 s7, s5, s6
	v_or_b32_e32 v45 /*v813*/, 0x10000, v8 /*v520*/
	s_set_vgpr_msb 0xc8cc
	v_or_b32_e32 v52 /*v820*/, 0xc0, v38 /*v806*/
	s_set_vgpr_msb 0xccc8
	v_or_b32_e32 v53 /*v821*/, 0xe0, v5 /*v517*/
	s_set_vgpr_msb 0xc8cc
	v_or_b32_e32 v55 /*v823*/, 0x10000, v39 /*v807*/
	v_or_b32_e32 v57 /*v825*/, 32, v54 /*v822*/
	s_mul_hi_u32 s7, s6, s7
	s_ashr_i32 s3, s2, 31
	s_mov_b32 s4, s36
	s_mov_b32 s5, s36
	s_ashr_i32 s13, s41, 31
	s_add_co_i32 s14, s6, s7
	s_mov_b64 s[6:7], 0
	s_mov_b32 s8, 0x3fb8aa3b
	s_set_vgpr_msb 0xcc00
.LBB0_8:
	s_set_vgpr_msb 0x84
	s_wait_loadcnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[520:527]*/, v[216:223], v[248:255] /*v[504:511]*/, 0
	s_set_vgpr_msb 0x8487
	s_wait_loadcnt 0x3b
	ds_store_b128 v36 /*v804*/, v[216:219] /*v[472:475]*/
	s_wait_loadcnt 0x3a
	ds_store_b128 v36 /*v804*/, v[220:223] /*v[476:479]*/ offset:32
	ds_store_b128 v36 /*v804*/, v[248:251] /*v[504:507]*/ offset:8704
	ds_store_b128 v36 /*v804*/, v[252:255] /*v[508:511]*/ offset:8736
	s_wait_loadcnt 0x39
	ds_store_b128 v36 /*v804*/, v[208:211] /*v[464:467]*/ offset:64
	s_wait_loadcnt 0x38
	ds_store_b128 v36 /*v804*/, v[212:215] /*v[468:471]*/ offset:96
	ds_store_b128 v36 /*v804*/, v[240:243] /*v[496:499]*/ offset:8768
	ds_store_b128 v36 /*v804*/, v[244:247] /*v[500:503]*/ offset:8800
	s_wait_loadcnt 0x37
	ds_store_b128 v36 /*v804*/, v[200:203] /*v[456:459]*/ offset:128
	s_wait_loadcnt 0x36
	ds_store_b128 v36 /*v804*/, v[204:207] /*v[460:463]*/ offset:160
	ds_store_b128 v36 /*v804*/, v[232:235] /*v[488:491]*/ offset:8832
	ds_store_b128 v36 /*v804*/, v[236:239] /*v[492:495]*/ offset:8864
	s_wait_loadcnt 0x35
	ds_store_b128 v36 /*v804*/, v[192:195] /*v[448:451]*/ offset:192
	v_mov_b64_e32 v[196:197] /*v[708:709]*/, s[4:5]
	ds_store_b128 v36 /*v804*/, v[224:227] /*v[480:483]*/ offset:8896
	ds_store_b128 v36 /*v804*/, v[228:231] /*v[484:487]*/ offset:8928
	s_wait_loadcnt 0x34
	ds_store_b128 v36 /*v804*/, v[196:199] /*v[452:455]*/ offset:224
	s_add_co_i32 s15, s6, 2
	s_set_vgpr_msb 0x878f
	v_dual_add_nc_u32 v1 /*v513*/, v39 /*v807*/, v38 /*v806*/ :: v_dual_add_nc_u32 v3 /*v515*/, v46 /*v814*/, v38 /*v806*/
	s_set_vgpr_msb 0x8f84
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[224:231], v[240:247] /*v[496:503]*/, 0
	s_abs_i32 s16, s15
	s_ashr_i32 s19, s15, 31
	s_mul_hi_u32 s17, s16, s14
	s_set_vgpr_msb 0x84c3
	s_wait_loadcnt 0x20
	v_dual_mov_b32 v62 /*v830*/, v40 /*v808*/ :: v_dual_mov_b32 v63 /*v831*/, v42 /*v810*/
	s_mul_i32 s18, s17, s12
	s_set_vgpr_msb 0xc382
	s_wait_loadcnt 0x7
	v_mov_b64_e32 v[198:199] /*v[710:711]*/, v[128:129] /*v[640:641]*/
	s_set_vgpr_msb 0x8284
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[536:543]*/, v[248:255], v[248:255] /*v[504:511]*/, 0
	s_sub_co_i32 s16, s16, s18
	s_xor_b32 s18, s19, s13
	s_add_co_i32 s19, s17, 1
	s_sub_co_i32 s20, s16, s12
	s_cmp_ge_u32 s16, s12
	s_set_vgpr_msb 0x8482
	s_wait_loadcnt 0x1
	v_mov_b64_e32 v[200:201] /*v[712:713]*/, v[150:151] /*v[662:663]*/
	s_cselect_b32 s17, s19, s17
	s_set_vgpr_msb 0x8285
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[684:691]*/, v[0:7] /*v[256:263]*/, v[240:247] /*v[496:503]*/, 0
	s_cselect_b32 s16, s20, s16
	s_add_co_i32 s19, s17, 1
	s_cmp_ge_u32 s16, s12
	s_set_vgpr_msb 0x8582
	v_mov_b64_e32 v[202:203] /*v[714:715]*/, v[148:149] /*v[660:661]*/
	s_cselect_b32 s16, s19, s17
	v_mov_b64_e32 v[204:205] /*v[716:717]*/, v[126:127] /*v[638:639]*/
	s_xor_b32 s16, s16, s18
	s_set_vgpr_msb 0x8284
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[692:699]*/, v[216:223], v[136:143] /*v[392:399]*/, 0
	s_sub_co_i32 s17, s16, s18
	s_set_vgpr_msb 0x8482
	v_mov_b64_e32 v[206:207] /*v[718:719]*/, v[124:125] /*v[636:637]*/
	s_mul_i32 s17, s17, s41
	v_mov_b64_e32 v[208:209] /*v[720:721]*/, v[122:123] /*v[634:635]*/
	s_cmp_lg_u32 s15, s17
	v_mov_b64_e32 v[210:211] /*v[722:723]*/, v[120:121] /*v[632:633]*/
	s_cselect_b32 s17, -1, 0
	s_set_vgpr_msb 0x8284
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[700:707]*/, v[224:231], v[152:159] /*v[408:415]*/, 0
	s_xor_b32 s19, s15, s41
	s_set_vgpr_msb 0x84c3
	v_mov_b32_e32 v60 /*v828*/, v41 /*v809*/
	s_cmp_lt_i32 s19, 0
	v_mov_b32_e32 v61 /*v829*/, v43 /*v811*/
	s_cselect_b32 s19, -1, 0
	s_set_vgpr_msb 0xc382
	v_mov_b64_e32 v[212:213] /*v[724:725]*/, v[106:107] /*v[618:619]*/
	s_and_b32 s17, s19, s17
	s_set_vgpr_msb 0x8244
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[248:255], v[136:143] /*v[392:399]*/, 0
	s_sub_co_ci_u32 s16, s16, s18
	s_set_vgpr_msb 0x4482
	v_mov_b64_e32 v[214:215] /*v[726:727]*/, v[104:105] /*v[616:617]*/
	s_add_co_i32 s17, s16, s9
	s_mul_i32 s16, s16, s41
	s_lshl_b32 s17, s17, 5
	s_sub_co_i32 s15, s15, s16
	s_set_vgpr_msb 0x82cc
	v_or_b32_e32 v58 /*v826*/, s17, v31 /*v799*/
	s_set_vgpr_msb 0xcc45
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[0:7] /*v[256:263]*/, v[152:159] /*v[408:415]*/, 0
	s_set_vgpr_msb 0x45cc
	v_or_b32_e32 v59 /*v827*/, s17, v35 /*v803*/
	s_add_co_i32 s16, s10, s15
	s_set_vgpr_msb 0xcc82
	v_mov_b64_e32 v[216:217] /*v[728:729]*/, v[102:103] /*v[614:615]*/
	s_mul_i32 s16, s16, s37
	v_mov_b64_e32 v[218:219] /*v[730:731]*/, v[100:101] /*v[612:613]*/
	v_mov_b64_e32 v[220:221] /*v[732:733]*/, v[114:115] /*v[626:627]*/
	v_mov_b64_e32 v[222:223] /*v[734:735]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[520:527]*/, v[232:239], v[232:239] /*v[488:495]*/, v[8:15] /*v[520:527]*/
	s_set_vgpr_msb 0xa482
	v_mov_b64_e32 v[224:225] /*v[736:737]*/, v[110:111] /*v[622:623]*/
	v_mov_b64_e32 v[226:227] /*v[738:739]*/, v[108:109] /*v[620:621]*/
	v_mov_b64_e32 v[228:229] /*v[740:741]*/, v[98:99] /*v[610:611]*/
	v_mov_b64_e32 v[230:231] /*v[742:743]*/, v[96:97] /*v[608:609]*/
	v_mov_b64_e32 v[232:233] /*v[744:745]*/, v[118:119] /*v[630:631]*/
	v_mov_b64_e32 v[234:235] /*v[746:747]*/, v[116:117] /*v[628:629]*/
	v_mov_b64_e32 v[236:237] /*v[748:749]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[240:247], v[224:231] /*v[480:487]*/, v[16:23] /*v[528:535]*/
	s_set_vgpr_msb 0xa482
	v_mov_b64_e32 v[238:239] /*v[750:751]*/, v[52:53] /*v[564:565]*/
	v_mov_b64_e32 v[240:241] /*v[752:753]*/, v[50:51] /*v[562:563]*/
	v_mov_b64_e32 v[242:243] /*v[754:755]*/, v[48:49] /*v[560:561]*/
	v_mov_b64_e32 v[244:245] /*v[756:757]*/, v[94:95] /*v[606:607]*/
	v_mov_b64_e32 v[246:247] /*v[758:759]*/, v[92:93] /*v[604:605]*/
	v_mov_b64_e32 v[248:249] /*v[760:761]*/, v[90:91] /*v[602:603]*/
	v_mov_b64_e32 v[250:251] /*v[762:763]*/, v[88:89] /*v[600:601]*/
	s_set_vgpr_msb 0x82a5
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[536:543]*/, v[8:15] /*v[264:271]*/, v[232:239] /*v[488:495]*/, v[24:31] /*v[536:543]*/
	s_set_vgpr_msb 0xa58a
	v_pk_add_f32 v[8:9] /*v[520:521]*/, v[8:9] /*v[520:521]*/, v[16:17] /*v[528:529]*/
	v_mov_b64_e32 v[252:253] /*v[764:765]*/, v[78:79] /*v[590:591]*/
	v_mov_b64_e32 v[254:255] /*v[766:767]*/, v[76:77] /*v[588:589]*/
	s_set_vgpr_msb 0x8ac2
	v_mov_b64_e32 v[0:1] /*v[768:769]*/, v[70:71] /*v[582:583]*/
	s_set_vgpr_msb 0xc24a
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[12:13] /*v[524:525]*/, v[20:21] /*v[532:533]*/
	v_pk_add_f32 v[236:237] /*v[492:493]*/, v[14:15] /*v[526:527]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[10:11] /*v[522:523]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x4aa5
	v_wmma_f32_16x16x32_bf16 v[172:179] /*v[684:691]*/, v[16:23] /*v[272:279]*/, v[224:231] /*v[480:487]*/, v[172:179] /*v[684:691]*/
	s_set_vgpr_msb 0xa54a
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[196:197] /*v[708:709]*/, v[8:9] /*v[520:521]*/
	s_set_vgpr_msb 0x4a86
	v_pk_mul_f32 v[8:9] /*v[520:521]*/, v[196:197] /*v[708:709]*/, v[234:235] /*v[490:491]*/
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[196:197] /*v[708:709]*/, v[236:237] /*v[492:493]*/
	s_set_vgpr_msb 0x8646
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[196:197] /*v[708:709]*/, v[232:233] /*v[488:489]*/
	s_set_vgpr_msb 0x46c2
	v_mov_b64_e32 v[2:3] /*v[770:771]*/, v[68:69] /*v[580:581]*/
	v_mov_b64_e32 v[4:5] /*v[772:773]*/, v[58:59] /*v[570:571]*/
	s_set_vgpr_msb 0xc28a
	v_pk_add_f32 v[8:9] /*v[520:521]*/, v[8:9] /*v[520:521]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[180:187] /*v[692:699]*/, v[232:239], v[144:151] /*v[400:407]*/, v[180:187] /*v[692:699]*/
	s_set_vgpr_msb 0xa44a
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[24:25] /*v[536:537]*/, v[172:173] /*v[684:685]*/
	v_pk_add_f32 v[226:227] /*v[482:483]*/, v[26:27] /*v[538:539]*/, v[174:175] /*v[686:687]*/
	v_pk_add_f32 v[228:229] /*v[484:485]*/, v[28:29] /*v[540:541]*/, v[176:177] /*v[688:689]*/
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[30:31] /*v[542:543]*/, v[178:179] /*v[690:691]*/
	s_set_vgpr_msb 0x4a89
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[238:239] /*v[494:495]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[224:225] /*v[480:481]*/, v[196:197] /*v[708:709]*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[226:227] /*v[482:483]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0x89a4
	v_wmma_f32_16x16x32_bf16 v[188:195] /*v[700:707]*/, v[240:247], v[128:135] /*v[384:391]*/, v[188:195] /*v[700:707]*/
	s_set_vgpr_msb 0xa486
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[196:197] /*v[708:709]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[196:197] /*v[708:709]*/, v[230:231] /*v[486:487]*/
	s_set_vgpr_msb 0x868a
	v_pk_add_f32 v[12:13] /*v[524:525]*/, v[12:13] /*v[524:525]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[30:31] /*v[542:543]*/, v[232:233] /*v[488:489]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a55
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[8:15] /*v[264:271]*/, v[144:151] /*v[400:407]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x554a
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[180:181] /*v[692:693]*/, v[188:189] /*v[700:701]*/
	v_pk_add_f32 v[226:227] /*v[482:483]*/, v[182:183] /*v[694:695]*/, v[190:191] /*v[702:703]*/
	v_pk_add_f32 v[228:229] /*v[484:485]*/, v[184:185] /*v[696:697]*/, v[192:193] /*v[704:705]*/
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[186:187] /*v[698:699]*/, v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[6:7] /*v[518:519]*/, v[18:19] /*v[530:531]*/, v[6:7] /*v[518:519]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a86
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[196:197] /*v[708:709]*/, v[224:225] /*v[480:481]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[196:197] /*v[708:709]*/, v[226:227] /*v[482:483]*/
	s_set_vgpr_msb 0x8655
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[16:23] /*v[272:279]*/, v[128:135] /*v[384:391]*/, v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5586
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[196:197] /*v[708:709]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[12:13] /*v[524:525]*/, s[8:9] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x868a
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[20:21] /*v[532:533]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[22:23] /*v[534:535]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, s[8:9] op_sel_hi:[1,0]
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[24:25] /*v[536:537]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[172:173] /*v[684:685]*/, v[10:11] /*v[522:523]*/, s[8:9] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8a45
	v_pk_add_f32 v[236:237] /*v[492:493]*/, v[248:249] /*v[504:505]*/, v[240:241] /*v[496:497]*/
	v_pk_add_f32 v[240:241] /*v[496:497]*/, v[250:251] /*v[506:507]*/, v[242:243] /*v[498:499]*/
	v_pk_add_f32 v[242:243] /*v[498:499]*/, v[252:253] /*v[508:509]*/, v[244:245] /*v[500:501]*/
	v_pk_add_f32 v[244:245] /*v[500:501]*/, v[254:255] /*v[510:511]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[196:197] /*v[708:709]*/, v[234:235] /*v[490:491]*/
	v_pk_mul_f32 v[250:251] /*v[506:507]*/, v[196:197] /*v[708:709]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[252:253] /*v[508:509]*/, v[196:197] /*v[708:709]*/, v[240:241] /*v[496:497]*/
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[196:197] /*v[708:709]*/, v[242:243] /*v[498:499]*/
	s_set_vgpr_msb 0x4686
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[196:197] /*v[708:709]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x8645
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[24:31] /*v[280:287]*/, v[216:223] /*v[472:479]*/, 0
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[248:249] /*v[504:505]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[250:251] /*v[506:507]*/, v[250:251] /*v[506:507]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[252:253] /*v[508:509]*/, v[252:253] /*v[508:509]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4989
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[254:255] /*v[510:511]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[4:5] /*v[516:517]*/, v[26:27] /*v[538:539]*/, v[4:5] /*v[516:517]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[28:29] /*v[540:541]*/, s[8:9] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[30:31] /*v[542:543]*/, s[8:9] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8a45
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[40:47] /*v[296:303]*/, v[208:215] /*v[464:471]*/, 0
	s_set_vgpr_msb 0x4582
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[8:9] /*v[520:521]*/, s[8:9] op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175] /*v[686:687]*/, v[16:17] /*v[528:529]*/, s[8:9] op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177] /*v[688:689]*/, v[6:7] /*v[518:519]*/, s[8:9] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179] /*v[690:691]*/, v[18:19] /*v[530:531]*/, s[8:9] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8281
	v_pk_mul_f32 v[180:181] /*v[692:693]*/, v[248:249] /*v[504:505]*/, s[8:9] op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183] /*v[694:695]*/, v[250:251] /*v[506:507]*/, s[8:9] op_sel_hi:[1,0]
	v_pk_mul_f32 v[184:185] /*v[696:697]*/, v[252:253] /*v[508:509]*/, s[8:9] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8145
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[64:71] /*v[320:327]*/, v[216:223] /*v[472:479]*/, 0
	s_set_vgpr_msb 0x4582
	v_pk_mul_f32 v[186:187] /*v[698:699]*/, v[4:5] /*v[516:517]*/, s[8:9] op_sel_hi:[1,0]
	v_exp_f32_e32 v188 /*v700*/, v12 /*v524*/
	v_exp_f32_e32 v189 /*v701*/, v13 /*v525*/
	v_exp_f32_e32 v190 /*v702*/, v14 /*v526*/
	v_exp_f32_e32 v191 /*v703*/, v15 /*v527*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, s[8:9] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, s[8:9] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8245
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[72:79] /*v[328:335]*/, v[208:215] /*v[464:471]*/, 0
	s_set_vgpr_msb 0x4582
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, s[8:9] op_sel_hi:[1,0]
	v_exp_f32_e32 v26 /*v538*/, v26 /*v538*/
	v_exp_f32_e32 v27 /*v539*/, v27 /*v539*/
	v_exp_f32_e32 v172 /*v684*/, v172 /*v684*/
	v_exp_f32_e32 v173 /*v685*/, v173 /*v685*/
	v_exp_f32_e32 v28 /*v540*/, v28 /*v540*/
	v_exp_f32_e32 v29 /*v541*/, v29 /*v541*/
	s_set_vgpr_msb 0x8245
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[24:31] /*v[280:287]*/, v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4582
	v_exp_f32_e32 v30 /*v542*/, v30 /*v542*/
	v_exp_f32_e32 v31 /*v543*/, v31 /*v543*/
	v_exp_f32_e32 v174 /*v686*/, v174 /*v686*/
	v_exp_f32_e32 v175 /*v687*/, v175 /*v687*/
	v_exp_f32_e32 v176 /*v688*/, v176 /*v688*/
	v_exp_f32_e32 v177 /*v689*/, v177 /*v689*/
	v_exp_f32_e32 v178 /*v690*/, v178 /*v690*/
	s_set_vgpr_msb 0x8245
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[40:47] /*v[296:303]*/, v[184:191] /*v[440:447]*/, 0
	s_set_vgpr_msb 0x4582
	v_exp_f32_e32 v179 /*v691*/, v179 /*v691*/
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
	v_exp_f32_e32 v22 /*v534*/, v22 /*v534*/
	v_exp_f32_e32 v23 /*v535*/, v23 /*v535*/
	v_exp_f32_e32 v180 /*v692*/, v180 /*v692*/
	v_exp_f32_e32 v181 /*v693*/, v181 /*v693*/
	s_set_vgpr_msb 0x8285
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[516:523]*/, v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x8582
	v_exp_f32_e32 v182 /*v694*/, v182 /*v694*/
	v_exp_f32_e32 v183 /*v695*/, v183 /*v695*/
	v_mov_b64_e32 v[192:193] /*v[704:705]*/, v[134:135] /*v[646:647]*/
	v_mov_b64_e32 v[194:195] /*v[706:707]*/, v[132:133] /*v[644:645]*/
	s_set_vgpr_msb 0x82c2
	v_mov_b64_e32 v[6:7] /*v[774:775]*/, v[56:57] /*v[568:569]*/
	v_mov_b64_e32 v[8:9] /*v[776:777]*/, v[82:83] /*v[594:595]*/
	v_mov_b64_e32 v[10:11] /*v[778:779]*/, v[80:81] /*v[592:593]*/
	s_set_vgpr_msb 0xc285
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[524:531]*/, v[72:79] /*v[328:335]*/, v[184:191] /*v[440:447]*/, 0
	s_set_vgpr_msb 0x85c2
	v_mov_b64_e32 v[12:13] /*v[780:781]*/, v[62:63] /*v[574:575]*/
	v_mov_b64_e32 v[14:15] /*v[782:783]*/, v[60:61] /*v[572:573]*/
	v_mov_b64_e32 v[16:17] /*v[784:785]*/, v[66:67] /*v[578:579]*/
	v_mov_b64_e32 v[18:19] /*v[786:787]*/, v[64:65] /*v[576:577]*/
	v_mov_b64_e32 v[20:21] /*v[788:789]*/, v[74:75] /*v[586:587]*/
	v_mov_b64_e32 v[22:23] /*v[790:791]*/, v[72:73] /*v[584:585]*/
	v_mov_b64_e32 v[24:25] /*v[792:793]*/, v[86:87] /*v[598:599]*/
	s_set_vgpr_msb 0xc255
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[48:55] /*v[304:311]*/, v[200:207] /*v[456:463]*/, v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x55c2
	v_mov_b64_e32 v[26:27] /*v[794:795]*/, v[84:85] /*v[596:597]*/
	v_mov_b64_e32 v[28:29] /*v[796:797]*/, v[34:35] /*v[546:547]*/
	s_lshl4_add_u32 s15, s15, s11
	s_set_vgpr_msb 0xc255
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[56:63] /*v[312:319]*/, v[192:199] /*v[448:455]*/, v[232:239] /*v[488:495]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1)
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[232:233] /*v[488:489]*/
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[80:87] /*v[336:343]*/, v[200:207] /*v[456:463]*/, v[240:247] /*v[496:503]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[230:231] /*v[486:487]*/, v[238:239] /*v[494:495]*/
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[226:227] /*v[482:483]*/, v[234:235] /*v[490:491]*/
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[228:229] /*v[484:485]*/, v[236:237] /*v[492:493]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[88:95] /*v[344:351]*/, v[192:199] /*v[448:455]*/, v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5549
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[224:225] /*v[480:481]*/, v[2:3] /*v[514:515]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[204:205] /*v[460:461]*/, v[2:3] /*v[514:515]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4942
	v_exp_f32_e32 v226 /*v482*/, v184 /*v696*/
	v_exp_f32_e32 v227 /*v483*/, v185 /*v697*/
	v_exp_f32_e32 v228 /*v484*/, v24 /*v536*/
	v_exp_f32_e32 v229 /*v485*/, v25 /*v537*/
	s_set_vgpr_msb 0x4249
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[2:3] /*v[514:515]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4955
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[48:55] /*v[304:311]*/, v[176:183] /*v[432:439]*/, v[208:215] /*v[464:471]*/
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[240:241] /*v[496:497]*/, v[216:217] /*v[472:473]*/
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[242:243] /*v[498:499]*/, v[218:219] /*v[474:475]*/
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[244:245] /*v[500:501]*/, v[220:221] /*v[476:477]*/
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[246:247] /*v[502:503]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x5542
	v_exp_f32_e32 v220 /*v476*/, v186 /*v698*/
	v_exp_f32_e32 v221 /*v477*/, v187 /*v699*/
	s_set_vgpr_msb 0x4249
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, v[2:3] /*v[514:515]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4955
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[56:63] /*v[312:319]*/, v[160:167] /*v[416:423]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5549
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, v[2:3] /*v[514:515]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[2:3] /*v[514:515]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[2:3] /*v[514:515]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[2:3] /*v[514:515]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[26:27] /*v[538:539]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[204:205] /*v[460:461]*/, v[172:173] /*v[684:685]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[28:29] /*v[540:541]*/
	s_set_vgpr_msb 0x49a5
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[516:523]*/, v[80:87] /*v[336:343]*/, v[176:183] /*v[432:439]*/, v[4:11] /*v[516:523]*/
	s_set_vgpr_msb 0xa545
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[248:249] /*v[504:505]*/
	v_pk_add_f32 v[210:211] /*v[466:467]*/, v[210:211] /*v[466:467]*/, v[250:251] /*v[506:507]*/
	v_pk_add_f32 v[212:213] /*v[468:469]*/, v[212:213] /*v[468:469]*/, v[252:253] /*v[508:509]*/
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x4549
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, v[30:31] /*v[542:543]*/
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[210:211] /*v[466:467]*/, v[210:211] /*v[466:467]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x49a5
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[524:531]*/, v[88:95] /*v[344:351]*/, v[160:167] /*v[416:423]*/, v[12:19] /*v[524:531]*/
	s_set_vgpr_msb 0xa549
	v_pk_add_f32 v[212:213] /*v[468:469]*/, v[212:213] /*v[468:469]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, v[188:189] /*v[700:701]*/
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[190:191] /*v[702:703]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[174:175] /*v[686:687]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[176:177] /*v[688:689]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[178:179] /*v[690:691]*/
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[4:5] /*v[516:517]*/, v[12:13] /*v[524:525]*/
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[6:7] /*v[518:519]*/, v[14:15] /*v[526:527]*/
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[8:9] /*v[520:521]*/, v[16:17] /*v[528:529]*/
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[10:11] /*v[522:523]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x4a49
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[210:211] /*v[466:467]*/, v[20:21] /*v[532:533]*/
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[0:1] /*v[512:513]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[212:213] /*v[468:469]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, v[180:181] /*v[692:693]*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, v[182:183] /*v[694:695]*/
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[226:227] /*v[482:483]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[220:221] /*v[476:477]*/
	s_set_vgpr_msb 0x4546
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[196:197] /*v[708:709]*/, v[206:207] /*v[462:463]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[196:197] /*v[708:709]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[196:197] /*v[708:709]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[196:197] /*v[708:709]*/, v[202:203] /*v[458:459]*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[196:197] /*v[708:709]*/, v[192:193] /*v[448:449]*/
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[196:197] /*v[708:709]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[196:197] /*v[708:709]*/, v[196:197] /*v[452:453]*/
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[454:455]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[196:197] /*v[708:709]*/, v[208:209] /*v[464:465]*/
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[196:197] /*v[708:709]*/, v[210:211] /*v[466:467]*/
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[196:197] /*v[708:709]*/, v[212:213] /*v[468:469]*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[196:197] /*v[708:709]*/, v[214:215] /*v[470:471]*/
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[196:197] /*v[708:709]*/, v[216:217] /*v[472:473]*/
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[196:197] /*v[708:709]*/, v[218:219] /*v[474:475]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[196:197] /*v[708:709]*/, v[222:223] /*v[478:479]*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[196:197] /*v[708:709]*/, v[224:225] /*v[480:481]*/
	s_set_vgpr_msb 0x464a
	v_cvt_pk_bf16_f32 v199 /*v455*/, v172 /*v684*/, v173 /*v685*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v30 /*v542*/, v31 /*v543*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v28 /*v540*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v26 /*v538*/, v27 /*v539*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v192 /*v448*/, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v204 /*v460*/, v205 /*v461*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v207 /*v463*/, v176 /*v688*/, v177 /*v689*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v174 /*v686*/, v175 /*v687*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v190 /*v702*/, v191 /*v703*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v188 /*v700*/, v189 /*v701*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v193 /*v449*/, v200 /*v456*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v194 /*v450*/, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v200 /*v456*/, v230 /*v486*/, v231 /*v487*/
	v_cvt_pk_bf16_f32 v201 /*v457*/, v232 /*v488*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v203 /*v459*/, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v208 /*v464*/, v208 /*v464*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v209 /*v465*/, v210 /*v466*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v210 /*v466*/, v212 /*v468*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v211 /*v467*/, v214 /*v470*/, v215 /*v471*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v215 /*v471*/, v180 /*v692*/, v181 /*v693*/
	v_cvt_pk_bf16_f32 v214 /*v470*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v213 /*v469*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v212 /*v468*/, v178 /*v690*/, v179 /*v691*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v216 /*v472*/, v216 /*v472*/, v217 /*v473*/
	v_cvt_pk_bf16_f32 v217 /*v473*/, v218 /*v474*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v218 /*v474*/, v222 /*v478*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v219 /*v475*/, v224 /*v480*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v223 /*v479*/, v220 /*v476*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v222 /*v478*/, v228 /*v484*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v221 /*v477*/, v226 /*v482*/, v227 /*v483*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v220 /*v476*/, v182 /*v694*/, v183 /*v695*/
	s_set_vgpr_msb 0x4a07
	ds_store_b128 v44 /*v812*/, v[196:199] /*v[452:455]*/
	ds_store_b128 v44 /*v812*/, v[204:207] /*v[460:463]*/ offset:32
	ds_store_b128 v44 /*v812*/, v[192:195] /*v[448:451]*/ offset:8704
	ds_store_b128 v44 /*v812*/, v[200:203] /*v[456:459]*/ offset:8736
	ds_store_b128 v37 /*v805*/, v[168:171] /*v[424:427]*/
	ds_store_b128 v37 /*v805*/, v[172:175] /*v[428:431]*/ offset:32
	ds_store_b128 v37 /*v805*/, v[136:139] /*v[392:395]*/ offset:8704
	ds_store_b128 v37 /*v805*/, v[140:143] /*v[396:399]*/ offset:8736
	ds_store_b128 v37 /*v805*/, v[184:187] /*v[440:443]*/ offset:64
	ds_store_b128 v37 /*v805*/, v[188:191] /*v[444:447]*/ offset:96
	ds_store_b128 v37 /*v805*/, v[152:155] /*v[408:411]*/ offset:8768
	ds_store_b128 v37 /*v805*/, v[156:159] /*v[412:415]*/ offset:8800
	ds_store_b128 v37 /*v805*/, v[176:179] /*v[432:435]*/ offset:128
	ds_store_b128 v37 /*v805*/, v[180:183] /*v[436:439]*/ offset:160
	ds_store_b128 v37 /*v805*/, v[144:147] /*v[400:403]*/ offset:8832
	ds_store_b128 v37 /*v805*/, v[148:151] /*v[404:407]*/ offset:8864
	ds_store_b128 v37 /*v805*/, v[160:163] /*v[416:419]*/ offset:192
	ds_store_b128 v37 /*v805*/, v[164:167] /*v[420:423]*/ offset:224
	ds_store_b128 v37 /*v805*/, v[128:131] /*v[384:387]*/ offset:8896
	ds_store_b128 v37 /*v805*/, v[132:135] /*v[388:391]*/ offset:8928
	ds_store_b128 v45 /*v813*/, v[212:215] /*v[468:471]*/
	ds_store_b128 v45 /*v813*/, v[220:223] /*v[476:479]*/ offset:32
	ds_store_b128 v45 /*v813*/, v[208:211] /*v[464:467]*/ offset:8704
	ds_store_b128 v45 /*v813*/, v[216:219] /*v[472:475]*/ offset:8736
	s_set_vgpr_msb 0x78c
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_lshl_u32 v173 /*v685*/, s16, v58 /*v826*/, 2
	v_add_lshl_u32 v174 /*v686*/, s16, v59 /*v827*/, 2
	s_set_vgpr_msb 0x8c4f
	v_dual_add_nc_u32 v128 /*v384*/, v39 /*v807*/, v47 /*v815*/ :: v_dual_add_nc_u32 v129 /*v385*/, v46 /*v814*/, v47 /*v815*/
	v_dual_add_nc_u32 v130 /*v386*/, v39 /*v807*/, v48 /*v816*/ :: v_dual_add_nc_u32 v131 /*v387*/, v46 /*v814*/, v48 /*v816*/
	v_dual_add_nc_u32 v132 /*v388*/, v39 /*v807*/, v49 /*v817*/ :: v_dual_add_nc_u32 v133 /*v389*/, v46 /*v814*/, v49 /*v817*/
	v_dual_add_nc_u32 v134 /*v390*/, v39 /*v807*/, v50 /*v818*/ :: v_dual_add_nc_u32 v135 /*v391*/, v46 /*v814*/, v50 /*v818*/
	v_dual_add_nc_u32 v136 /*v392*/, v39 /*v807*/, v51 /*v819*/ :: v_dual_add_nc_u32 v137 /*v393*/, v46 /*v814*/, v51 /*v819*/
	v_dual_add_nc_u32 v138 /*v394*/, v39 /*v807*/, v52 /*v820*/ :: v_dual_add_nc_u32 v139 /*v395*/, v46 /*v814*/, v52 /*v820*/
	v_dual_add_nc_u32 v140 /*v396*/, v39 /*v807*/, v53 /*v821*/ :: v_dual_add_nc_u32 v141 /*v397*/, v46 /*v814*/, v53 /*v821*/
	v_dual_add_nc_u32 v142 /*v398*/, v55 /*v823*/, v54 /*v822*/ :: v_dual_add_nc_u32 v143 /*v399*/, v56 /*v824*/, v54 /*v822*/
	s_set_vgpr_msb 0x4f8f
	v_dual_add_nc_u32 v12 /*v524*/, v55 /*v823*/, v57 /*v825*/ :: v_dual_add_nc_u32 v172 /*v684*/, v56 /*v824*/, v57 /*v825*/
	s_set_vgpr_msb 0x8f42
	s_barrier_wait -1
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v1 /*v513*/
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v1 /*v513*/ offset:4352
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v3 /*v515*/
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v3 /*v515*/ offset:4352
	s_set_vgpr_msb 0x4241
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v128 /*v384*/
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v128 /*v384*/ offset:4352
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v129 /*v385*/
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v129 /*v385*/ offset:4352
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v130 /*v386*/
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v130 /*v386*/ offset:4352
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v131 /*v387*/
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v131 /*v387*/ offset:4352
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v132 /*v388*/
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v132 /*v388*/ offset:4352
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v133 /*v389*/
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v133 /*v389*/ offset:4352
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v134 /*v390*/
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v134 /*v390*/ offset:4352
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v135 /*v391*/
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v135 /*v391*/ offset:4352
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v136 /*v392*/
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v136 /*v392*/ offset:4352
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v137 /*v393*/
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v137 /*v393*/ offset:4352
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v138 /*v394*/
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v138 /*v394*/ offset:4352
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v139 /*v395*/
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v139 /*v395*/ offset:4352
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v140 /*v396*/
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v140 /*v396*/ offset:4352
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v141 /*v397*/
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v141 /*v397*/ offset:4352
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[16:19] /*v[528:531]*/, v142 /*v398*/
	ds_load_tr16_b128 v[20:23] /*v[532:535]*/, v142 /*v398*/ offset:4352
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v143 /*v399*/
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v143 /*v399*/ offset:4352
	s_set_vgpr_msb 0x8182
	ds_load_tr16_b128 v[8:11] /*v[520:523]*/, v12 /*v524*/
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v12 /*v524*/ offset:4352
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v172 /*v684*/
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v172 /*v684*/ offset:4352
	s_set_vgpr_msb 0x42c2
	s_barrier_signal -1
	s_clause 0x1
	buffer_load_b32 v41 /*v809*/, v173 /*v685*/, s[56:59], null offen
	buffer_load_b32 v40 /*v808*/, v174 /*v686*/, s[56:59], null offen
	s_clause 0x1
	buffer_load_b32 v43 /*v811*/, v173 /*v685*/, s[60:63], null offen
	buffer_load_b32 v42 /*v810*/, v174 /*v686*/, s[60:63], null offen
	s_set_vgpr_msb 0xc256
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[16:23] /*v[528:535]*/, v[144:151] /*v[400:407]*/, v[104:111] /*v[360:367]*/
	s_set_vgpr_msb 0x5682
	v_mov_b64_e32 v[172:173] /*v[684:685]*/, v[158:159] /*v[670:671]*/
	v_mov_b64_e32 v[174:175] /*v[686:687]*/, v[156:157] /*v[668:669]*/
	v_mov_b64_e32 v[176:177] /*v[688:689]*/, v[154:155] /*v[666:667]*/
	v_mov_b64_e32 v[178:179] /*v[690:691]*/, v[152:153] /*v[664:665]*/
	v_mov_b64_e32 v[180:181] /*v[692:693]*/, v[146:147] /*v[658:659]*/
	v_mov_b64_e32 v[182:183] /*v[694:695]*/, v[144:145] /*v[656:657]*/
	v_mov_b64_e32 v[184:185] /*v[696:697]*/, v[142:143] /*v[654:655]*/
	s_set_vgpr_msb 0x8256
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[24:31] /*v[536:543]*/, v[152:159] /*v[408:415]*/, v[120:127] /*v[376:383]*/
	s_set_vgpr_msb 0x5682
	v_mov_b64_e32 v[186:187] /*v[698:699]*/, v[140:141] /*v[652:653]*/
	v_mov_b64_e32 v[188:189] /*v[700:701]*/, v[138:139] /*v[650:651]*/
	v_mov_b64_e32 v[190:191] /*v[702:703]*/, v[136:137] /*v[648:649]*/
	v_mov_b64_e32 v[196:197] /*v[708:709]*/, v[130:131] /*v[642:643]*/
	s_set_vgpr_msb 0x8256
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[16:23] /*v[528:535]*/, v[160:167] /*v[416:423]*/, v[32:39] /*v[288:295]*/
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[368:375]*/, v[24:31] /*v[536:543]*/, v[168:175] /*v[424:431]*/, v[112:119] /*v[368:375]*/
	s_set_vgpr_msb 0x5606
	v_wmma_f32_16x16x32_bf16 v[200:207], v[16:23] /*v[528:535]*/, v[176:183] /*v[432:439]*/, v[200:207]
	s_set_vgpr_msb 0x656
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[24:31] /*v[536:543]*/, v[184:191] /*v[440:447]*/, v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5606
	v_wmma_f32_16x16x32_bf16 v[184:191], v[16:23] /*v[528:535]*/, v[192:199] /*v[448:455]*/, v[184:191]
	v_wmma_f32_16x16x32_bf16 v[208:215], v[24:31] /*v[536:543]*/, v[200:207] /*v[456:463]*/, v[208:215]
	v_wmma_f32_16x16x32_bf16 v[168:175], v[16:23] /*v[528:535]*/, v[208:215] /*v[464:471]*/, v[168:175]
	v_wmma_f32_16x16x32_bf16 v[192:199], v[24:31] /*v[536:543]*/, v[216:223] /*v[472:479]*/, v[192:199]
	v_wmma_f32_16x16x32_bf16 v[152:159], v[16:23] /*v[528:535]*/, v[224:231] /*v[480:487]*/, v[152:159]
	v_wmma_f32_16x16x32_bf16 v[176:183], v[24:31] /*v[536:543]*/, v[232:239] /*v[488:495]*/, v[176:183]
	v_wmma_f32_16x16x32_bf16 v[136:143], v[16:23] /*v[528:535]*/, v[240:247] /*v[496:503]*/, v[136:143]
	v_wmma_f32_16x16x32_bf16 v[160:167], v[24:31] /*v[536:543]*/, v[248:255] /*v[504:511]*/, v[160:167]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[128:135], v[16:23] /*v[528:535]*/, v[0:7] /*v[512:519]*/, v[128:135]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[144:151], v[24:31] /*v[536:543]*/, v[128:135] /*v[384:391]*/, v[144:151]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[112:119], v[8:15] /*v[520:527]*/, v[144:151] /*v[400:407]*/, v[112:119]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[32:33] /*v[544:545]*/
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[38:39] /*v[550:551]*/
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[36:37] /*v[548:549]*/
	s_set_vgpr_msb 0x4205
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[120:127], v[136:143] /*v[392:399]*/, v[152:159] /*v[408:415]*/, v[120:127]
	s_set_vgpr_msb 0x542
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[42:43] /*v[554:555]*/
	s_set_vgpr_msb 0x4206
	v_wmma_f32_16x16x32_bf16 v[88:95], v[8:15] /*v[520:527]*/, v[160:167] /*v[416:423]*/, v[88:95]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[104:111], v[136:143] /*v[392:399]*/, v[168:175] /*v[424:431]*/, v[104:111]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[72:79], v[8:15] /*v[520:527]*/, v[176:183] /*v[432:439]*/, v[72:79]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[96:103], v[136:143] /*v[392:399]*/, v[184:191] /*v[440:447]*/, v[96:103]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[56:63], v[8:15] /*v[520:527]*/, v[192:199] /*v[448:455]*/, v[56:63]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[80:87], v[136:143] /*v[392:399]*/, v[200:207] /*v[456:463]*/, v[80:87]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[40:47], v[8:15] /*v[520:527]*/, v[208:215] /*v[464:471]*/, v[40:47]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[64:71], v[136:143] /*v[392:399]*/, v[216:223] /*v[472:479]*/, v[64:71]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[24:31], v[8:15] /*v[520:527]*/, v[224:231] /*v[480:487]*/, v[24:31]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[232:239] /*v[488:495]*/, v[48:55]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[8:15], v[8:15] /*v[520:527]*/, v[240:247] /*v[496:503]*/, v[8:15]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[32:39], v[136:143] /*v[392:399]*/, v[248:255] /*v[504:511]*/, v[32:39]
	s_set_vgpr_msb 0x50a
	v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15] /*v[520:527]*/, v[0:7] /*v[512:519]*/, v[0:7]
	s_set_vgpr_msb 0xa05
	v_wmma_f32_16x16x32_bf16 v[16:23], v[136:143] /*v[392:399]*/, v[128:135] /*v[384:391]*/, v[16:23]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x542
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[40:41] /*v[552:553]*/
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[46:47] /*v[558:559]*/
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x4243
	v_mad_u32 v134 /*v390*/, v58 /*v826*/, s35, s15
	v_mad_u32 v135 /*v391*/, v59 /*v827*/, s35, s15
	s_set_vgpr_msb 0x434d
	s_delay_alu instid0(VALU_DEP_3)
	v_mov_b64_e32 v[164:165] /*v[420:421]*/, v[132:133] /*v[388:389]*/
	v_mov_b64_e32 v[166:167] /*v[422:423]*/, v[130:131] /*v[386:387]*/
	v_mov_b64_e32 v[160:161] /*v[416:417]*/, v[128:129] /*v[384:385]*/
	v_mov_b64_e32 v[162:163] /*v[418:419]*/, v[150:151] /*v[406:407]*/
	v_mov_b64_e32 v[180:181] /*v[436:437]*/, v[148:149] /*v[404:405]*/
	v_mov_b64_e32 v[182:183] /*v[438:439]*/, v[146:147] /*v[402:403]*/
	v_or_b32_e32 v134 /*v390*/, v134 /*v390*/, v33 /*v801*/
	v_or_b32_e32 v135 /*v391*/, v135 /*v391*/, v33 /*v801*/
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[144:145] /*v[400:401]*/
	s_set_vgpr_msb 0x4d47
	v_mov_b64_e32 v[178:179] /*v[434:435]*/, v[28:29] /*v[796:797]*/
	v_mov_b64_e32 v[188:189] /*v[444:445]*/, v[26:27] /*v[794:795]*/
	v_dual_lshlrev_b32 v134 /*v390*/, 4, v134 /*v390*/ :: v_dual_lshlrev_b32 v135 /*v391*/, 4, v135 /*v391*/
	v_mov_b64_e32 v[190:191] /*v[446:447]*/, v[24:25] /*v[792:793]*/
	v_mov_b64_e32 v[184:185] /*v[440:441]*/, v[22:23] /*v[790:791]*/
	v_mov_b64_e32 v[186:187] /*v[442:443]*/, v[20:21] /*v[788:789]*/
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v136 /*v392*/, 32, v134 /*v390*/
	v_or_b32_e32 v137 /*v393*/, 64, v134 /*v390*/
	v_or_b32_e32 v138 /*v394*/, 0x60, v134 /*v390*/
	v_or_b32_e32 v139 /*v395*/, 0x80, v134 /*v390*/
	v_or_b32_e32 v140 /*v396*/, 0xa0, v134 /*v390*/
	v_or_b32_e32 v141 /*v397*/, 0xc0, v134 /*v390*/
	v_or_b32_e32 v142 /*v398*/, 0xe0, v134 /*v390*/
	s_set_vgpr_msb 0x4781
	s_clause 0x7
	buffer_load_b128 v[156:159] /*v[668:671]*/, v134 /*v390*/, s[52:55], null offen
	buffer_load_b128 v[152:155] /*v[664:667]*/, v136 /*v392*/, s[52:55], null offen
	buffer_load_b128 v[144:147] /*v[656:659]*/, v137 /*v393*/, s[52:55], null offen
	buffer_load_b128 v[140:143] /*v[652:655]*/, v138 /*v394*/, s[52:55], null offen
	buffer_load_b128 v[136:139] /*v[648:651]*/, v139 /*v395*/, s[52:55], null offen
	buffer_load_b128 v[132:135] /*v[644:647]*/, v140 /*v396*/, s[52:55], null offen
	buffer_load_b128 v[128:131] /*v[640:643]*/, v141 /*v397*/, s[52:55], null offen
	buffer_load_b128 v[148:151] /*v[660:663]*/, v142 /*v398*/, s[52:55], null offen
	s_clause 0x3
	buffer_load_b128 v[124:127] /*v[636:639]*/, v134 /*v390*/, s[48:51], null offen
	buffer_load_b128 v[120:123] /*v[632:635]*/, v136 /*v392*/, s[48:51], null offen
	buffer_load_b128 v[104:107] /*v[616:619]*/, v137 /*v393*/, s[48:51], null offen
	buffer_load_b128 v[100:103] /*v[612:615]*/, v138 /*v394*/, s[48:51], null offen
	s_set_vgpr_msb 0x8144
	v_or_b32_e32 v136 /*v392*/, 64, v135 /*v391*/
	v_or_b32_e32 v137 /*v393*/, 0x60, v135 /*v391*/
	v_or_b32_e32 v138 /*v394*/, 0x80, v135 /*v391*/
	v_or_b32_e32 v134 /*v390*/, 32, v135 /*v391*/
	s_set_vgpr_msb 0x4481
	s_clause 0x3
	buffer_load_b128 v[112:115] /*v[624:627]*/, v139 /*v395*/, s[48:51], null offen
	buffer_load_b128 v[108:111] /*v[620:623]*/, v140 /*v396*/, s[48:51], null offen
	buffer_load_b128 v[96:99] /*v[608:611]*/, v141 /*v397*/, s[48:51], null offen
	buffer_load_b128 v[116:119] /*v[628:631]*/, v142 /*v398*/, s[48:51], null offen
	s_clause 0x1
	buffer_load_b128 v[52:55] /*v[564:567]*/, v135 /*v391*/, s[52:55], null offen
	buffer_load_b128 v[48:51] /*v[560:563]*/, v134 /*v390*/, s[52:55], null offen
	s_set_vgpr_msb 0x8144
	v_or_b32_e32 v139 /*v395*/, 0xa0, v135 /*v391*/
	v_or_b32_e32 v140 /*v396*/, 0xc0, v135 /*v391*/
	v_or_b32_e32 v141 /*v397*/, 0xe0, v135 /*v391*/
	s_set_vgpr_msb 0x4481
	s_clause 0x5
	buffer_load_b128 v[92:95] /*v[604:607]*/, v136 /*v392*/, s[52:55], null offen
	buffer_load_b128 v[88:91] /*v[600:603]*/, v137 /*v393*/, s[52:55], null offen
	buffer_load_b128 v[76:79] /*v[588:591]*/, v138 /*v394*/, s[52:55], null offen
	buffer_load_b128 v[68:71] /*v[580:583]*/, v139 /*v395*/, s[52:55], null offen
	buffer_load_b128 v[56:59] /*v[568:571]*/, v140 /*v396*/, s[52:55], null offen
	buffer_load_b128 v[80:83] /*v[592:595]*/, v141 /*v397*/, s[52:55], null offen
	s_clause 0x7
	buffer_load_b128 v[60:63] /*v[572:575]*/, v135 /*v391*/, s[48:51], null offen
	buffer_load_b128 v[64:67] /*v[576:579]*/, v134 /*v390*/, s[48:51], null offen
	buffer_load_b128 v[72:75] /*v[584:587]*/, v136 /*v392*/, s[48:51], null offen
	buffer_load_b128 v[84:87] /*v[596:599]*/, v137 /*v393*/, s[48:51], null offen
	buffer_load_b128 v[32:35] /*v[544:547]*/, v138 /*v394*/, s[48:51], null offen
	buffer_load_b128 v[36:39] /*v[548:551]*/, v139 /*v395*/, s[48:51], null offen
	buffer_load_b128 v[40:43] /*v[552:555]*/, v140 /*v396*/, s[48:51], null offen
	buffer_load_b128 v[44:47] /*v[556:559]*/, v141 /*v397*/, s[48:51], null offen
	s_set_vgpr_msb 0x8143
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[18:19] /*v[786:787]*/
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[16:17] /*v[784:785]*/
	v_mov_b64_e32 v[168:169] /*v[424:425]*/, v[14:15] /*v[782:783]*/
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[12:13] /*v[780:781]*/
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, v[10:11] /*v[778:779]*/
	v_mov_b64_e32 v[134:135] /*v[390:391]*/, v[8:9] /*v[776:777]*/
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[6:7] /*v[774:775]*/
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[4:5] /*v[772:773]*/
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[2:3] /*v[770:771]*/
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[0:1] /*v[768:769]*/
	s_set_vgpr_msb 0x4342
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[254:255] /*v[766:767]*/
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[252:253] /*v[764:765]*/
	v_mov_b64_e32 v[156:157] /*v[412:413]*/, v[250:251] /*v[762:763]*/
	v_mov_b64_e32 v[158:159] /*v[414:415]*/, v[248:249] /*v[760:761]*/
	v_mov_b64_e32 v[152:153] /*v[408:409]*/, v[246:247] /*v[758:759]*/
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[244:245] /*v[756:757]*/
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[242:243] /*v[754:755]*/
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[240:241] /*v[752:753]*/
	v_mov_b64_e32 v[136:137] /*v[392:393]*/, v[238:239] /*v[750:751]*/
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[236:237] /*v[748:749]*/
	v_mov_b64_e32 v[196:197] /*v[452:453]*/, v[234:235] /*v[746:747]*/
	v_mov_b64_e32 v[198:199] /*v[454:455]*/, v[232:233] /*v[744:745]*/
	v_mov_b64_e32 v[192:193] /*v[448:449]*/, v[230:231] /*v[742:743]*/
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, v[228:229] /*v[740:741]*/
	v_mov_b64_e32 v[204:205] /*v[460:461]*/, v[226:227] /*v[738:739]*/
	v_mov_b64_e32 v[206:207] /*v[462:463]*/, v[224:225] /*v[736:737]*/
	v_mov_b64_e32 v[200:201] /*v[456:457]*/, v[222:223] /*v[734:735]*/
	v_mov_b64_e32 v[202:203] /*v[458:459]*/, v[220:221] /*v[732:733]*/
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[218:219] /*v[730:731]*/
	v_mov_b64_e32 v[214:215] /*v[470:471]*/, v[216:217] /*v[728:729]*/
	v_mov_b64_e32 v[208:209] /*v[464:465]*/, v[214:215] /*v[726:727]*/
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[212:213] /*v[724:725]*/
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[236:237] /*v[492:493]*/, v[194:195] /*v[706:707]*/
	v_mov_b64_e32 v[238:239] /*v[494:495]*/, v[192:193] /*v[704:705]*/
	v_mov_b64_e32 v[232:233] /*v[488:489]*/, v[190:191] /*v[702:703]*/
	v_mov_b64_e32 v[234:235] /*v[490:491]*/, v[188:189] /*v[700:701]*/
	v_mov_b64_e32 v[244:245] /*v[500:501]*/, v[186:187] /*v[698:699]*/
	v_mov_b64_e32 v[246:247] /*v[502:503]*/, v[184:185] /*v[696:697]*/
	v_mov_b64_e32 v[240:241] /*v[496:497]*/, v[182:183] /*v[694:695]*/
	v_mov_b64_e32 v[242:243] /*v[498:499]*/, v[180:181] /*v[692:693]*/
	v_mov_b64_e32 v[252:253] /*v[508:509]*/, v[178:179] /*v[690:691]*/
	v_mov_b64_e32 v[254:255] /*v[510:511]*/, v[176:177] /*v[688:689]*/
	v_mov_b64_e32 v[248:249] /*v[504:505]*/, v[174:175] /*v[686:687]*/
	v_mov_b64_e32 v[250:251] /*v[506:507]*/, v[172:173] /*v[684:685]*/
	s_set_vgpr_msb 0x4283
	v_dual_mov_b32 v0 /*v512*/, v63 /*v831*/ :: v_dual_mov_b32 v4 /*v516*/, v62 /*v830*/
	v_dual_mov_b32 v2 /*v514*/, v61 /*v829*/ :: v_dual_mov_b32 v6 /*v518*/, v60 /*v828*/
	s_add_nc_u64 s[6:7], s[6:7], 1
	s_barrier_wait -1
	s_cmp_lg_u64 s[6:7], s[2:3]
	s_set_vgpr_msb 0x8300
	s_cbranch_scc1 .LBB0_8
.LBB0_9:
	s_clause 0x1
	s_load_b64 s[6:7], s[0:1], 0x110 nv
	s_load_b64 s[8:9], s[0:1], 0x140 nv
	s_set_vgpr_msb 12
	s_wait_loadcnt 0x3e
	v_or_b32_e32 v217, 1, v34 /*v802*/
	v_mul_lo_u32 v216, s40, v34 /*v802*/
	s_mul_i32 s4, s40, s65
	s_lshl_b32 s1, s34, 25
	s_add_co_i32 s4, s4, s64
	v_mul_lo_u32 v217, v217, s40
	s_mov_b32 s0, 0
	s_set_vgpr_msb 0xc02
	v_mul_lo_u32 v223, v170 /*v682*/, s40
	v_mul_lo_u32 v225, v171 /*v683*/, s40
	v_add_lshl_u32 v216, s4, v216, 7
	s_set_vgpr_msb 0x201
	v_cvt_pk_bf16_f32 v218, v104 /*v360*/, s0
	v_cvt_pk_bf16_f32 v220, v120 /*v376*/, s0
	s_mov_b32 s2, s46
	v_add_lshl_u32 v217, s4, v217, 7
	s_set_vgpr_msb 0x10c
	v_or_b32_e32 v219, v216, v31 /*v799*/
	s_mov_b32 s3, s47
	v_add_lshl_u32 v223, v223, s4, 7
	s_set_vgpr_msb 0xc02
	v_mul_lo_u32 v226, v168 /*v680*/, s40
	s_set_vgpr_msb 0x20c
	v_or_b32_e32 v221, v217, v31 /*v799*/
	s_wait_kmcnt 0x0
	s_or_b64 s[44:45], s[6:7], s[0:1]
	s_or_b64 s[0:1], s[8:9], s[0:1]
	s_set_vgpr_msb 0xc02
	v_lshlrev_b32_e32 v219, 2, v219
	v_mul_lo_u32 v229, v169 /*v681*/, s40
	s_set_vgpr_msb 0x201
	v_cvt_pk_bf16_f32 v222, v105 /*v361*/, s0
	v_cvt_pk_bf16_f32 v224, v121 /*v377*/, s0
	v_lshlrev_b32_e32 v221, 2, v221
	s_set_vgpr_msb 0x100
	buffer_store_b16 v218, v219, s[44:47], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v227, v107 /*v363*/, s0
	v_cvt_pk_bf16_f32 v228, v123 /*v379*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v218.l, v224.l
	buffer_store_b16 v220, v219, s[0:3], null offen
	buffer_store_b16 v222, v221, s[44:47], null offen
	s_wait_xcnt 0x1
	v_add_lshl_u32 v220, v225, s4, 7
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v224, v122 /*v378*/, s0
	v_add_lshl_u32 v226, s4, v226, 7
	s_set_vgpr_msb 0x10c
	buffer_store_b16 v218, v221, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v218, v223, v31 /*v799*/
	v_or_b32_e32 v225, v220, v31 /*v799*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v222, v106 /*v362*/, s0
	s_set_vgpr_msb 0x102
	v_mul_lo_u32 v230, v166 /*v678*/, s40
	v_mul_lo_u32 v233, v167 /*v679*/, s40
	v_dual_lshlrev_b32 v218, 2, v218 :: v_dual_lshlrev_b32 v225, 2, v225
	s_set_vgpr_msb 0x201
	v_cvt_pk_bf16_f32 v232, v125 /*v381*/, s0
	v_cvt_pk_bf16_f32 v231, v109 /*v365*/, s0
	v_cvt_pk_bf16_f32 v235, v127 /*v383*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v222, v218, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v222.l, v228.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v228, v124 /*v380*/, s0
	v_add_lshl_u32 v230, s4, v230, 7
	s_set_vgpr_msb 0x10c
	buffer_store_b16 v224, v218, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v224, v226, v31 /*v799*/
	v_or_b32_e32 v216, v216, v30 /*v798*/
	v_or_b32_e32 v217, v217, v30 /*v798*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v234, v111 /*v367*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v227, v225, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v227, v229, s4, 7
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v236, v113 /*v369*/, s0
	v_dual_lshlrev_b32 v216, 2, v216 :: v_dual_lshlrev_b32 v217, 2, v217
	s_set_vgpr_msb 0x10c
	buffer_store_b16 v222, v225, s[0:3], null offen
	v_or_b32_e32 v229, v227, v31 /*v799*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v222, v108 /*v364*/, s0
	s_set_vgpr_msb 0x10c
	v_or_b32_e32 v220, v220, v30 /*v798*/
	v_or_b32_e32 v226, v226, v30 /*v798*/
	v_cvt_pk_bf16_f32 v200, v200, s0
	s_set_vgpr_msb 0xc00
	v_dual_lshlrev_b32 v224, 2, v224 :: v_dual_lshlrev_b32 v229, 2, v229
	v_dual_lshlrev_b32 v220, 2, v220 :: v_dual_bitop2_b32 v237, 64, v217 bitop3:0x54
	s_set_vgpr_msb 12
	v_or_b32_e32 v227, v227, v30 /*v798*/
	buffer_store_b16 v222, v224, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v222.l, v232.l
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v232, v126 /*v382*/, s0
	s_set_vgpr_msb 0x10c
	v_cvt_pk_bf16_f32 v201, v201, s0
	buffer_store_b16 v228, v224, s[0:3], null offen
	buffer_store_b16 v231, v229, s[44:47], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v228, v230, v31 /*v799*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v226, 2, v226
	s_set_vgpr_msb 12
	v_or_b32_e32 v230, v230, v30 /*v798*/
	buffer_store_b16 v222, v229, s[0:3], null offen
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v222, v110 /*v366*/, s0
	v_add_lshl_u32 v231, s4, v233, 7
	s_set_vgpr_msb 0x10c
	v_cvt_pk_bf16_f32 v202, v202, s0
	v_cvt_pk_bf16_f32 v184, v184, s0
	v_cvt_pk_bf16_f32 v185, v185, s0
	v_cvt_pk_bf16_f32 v187, v187, s0
	v_or_b32_e32 v233, v231, v31 /*v799*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v227, 2, v227
	v_cvt_pk_bf16_f32 v168, v168, s0
	v_lshlrev_b32_e32 v228, 2, v228
	v_cvt_pk_bf16_f32 v169, v169, s0
	v_lshlrev_b32_e32 v233, 2, v233
	v_cvt_pk_bf16_f32 v170, v170, s0
	v_cvt_pk_bf16_f32 v152, v152, s0
	buffer_store_b16 v222, v228, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v222.l, v235.l
	buffer_store_b16 v232, v228, s[0:3], null offen
	buffer_store_b16 v234, v233, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v234, 64, v216
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v232, v112 /*v368*/, s0
	v_cvt_pk_bf16_f32 v235, v33 /*v289*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v222, v233, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v222, v32 /*v288*/, s0
	s_set_vgpr_msb 0x10c
	v_cvt_pk_bf16_f32 v153, v153, s0
	v_cvt_pk_bf16_f32 v155, v155, s0
	v_cvt_pk_bf16_f32 v136, v136, s0
	v_cvt_pk_bf16_f32 v137, v137, s0
	buffer_store_b16 v222, v234, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v222, v223, v30 /*v798*/
	v_mov_b16_e64 v223.l, v236.l
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v236, 64, v220
	v_cvt_pk_bf16_f32 v138, v138, s0
	v_cvt_pk_bf16_f32 v128, v128, s0
	v_lshlrev_b32_e32 v222, 2, v222
	buffer_store_b16 v232, v234, s[0:3], null offen
	buffer_store_b16 v235, v237, s[44:47], null offen
	buffer_store_b16 v223, v237, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v223, v34 /*v290*/, s0
	v_cvt_pk_bf16_f32 v237, v115 /*v371*/, s0
	v_or_b32_e32 v234, 64, v222
	v_cvt_pk_bf16_f32 v232, v114 /*v370*/, s0
	v_cvt_pk_bf16_f32 v235, v35 /*v291*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v223, v234, s[44:47], null offen
	buffer_store_b16 v232, v234, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v223.l, v237.l
	buffer_store_b16 v235, v236, s[44:47], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v232, v36 /*v292*/, s0
	v_or_b32_e32 v234, 64, v226
	v_cvt_pk_bf16_f32 v235, v37 /*v293*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v223, v236, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v223, v116 /*v372*/, s0
	v_cvt_pk_bf16_f32 v236, v117 /*v373*/, s0
	v_or_b32_e32 v237, 64, v227
	s_set_vgpr_msb 0x100
	buffer_store_b16 v232, v234, s[44:47], null offen
	v_cvt_pk_bf16_f32 v129, v129, s0
	buffer_store_b16 v223, v234, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v232.l, v236.l
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v223, 2, v230
	s_set_vgpr_msb 12
	v_or_b32_e32 v230, v231, v30 /*v798*/
	buffer_store_b16 v235, v237, s[44:47], null offen
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v231, v38 /*v294*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v232, v237, s[0:3], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v230, 2, v230 :: v_dual_bitop2_b32 v232, 64, v223 bitop3:0x54
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v234, v118 /*v374*/, s0
	v_cvt_pk_bf16_f32 v235, v39 /*v295*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v231, v232, s[44:47], null offen
	buffer_store_b16 v234, v232, s[0:3], null offen
	v_or_b32_e32 v237, 64, v230
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v232, v97 /*v353*/, s0
	v_cvt_pk_bf16_f32 v236, v119 /*v375*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v235, v237, s[44:47], null offen
	buffer_store_b16 v236, v237, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v231, v96 /*v352*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v200, v219, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v200.l, v232.l
	buffer_store_b16 v231, v219, s[0:3], null offen offset:128
	buffer_store_b16 v201, v221, s[44:47], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v201, v98 /*v354*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v200, v221, s[0:3], null offen offset:128
	buffer_store_b16 v202, v218, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v200, v203, s0
	v_cvt_pk_bf16_f32 v203, v204, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v204, v100 /*v356*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v201, v218, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v202, v99 /*v355*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v200, v225, s[44:47], null offen offset:128
	buffer_store_b16 v202, v225, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_mov_b16_e64 v201.l, v203.l
	v_mov_b16_e64 v203.l, v204.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v200, v205, s0
	buffer_store_b16 v201, v224, s[44:47], null offen offset:128
	buffer_store_b16 v203, v224, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v203, v102 /*v358*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v204, v207, s0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v201, v101 /*v357*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v200, v229, s[44:47], null offen offset:128
	v_cvt_pk_bf16_f32 v202, v206, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v200.l, v203.l
	v_mov_b16_e64 v203.l, v204.l
	buffer_store_b16 v201, v229, s[0:3], null offen offset:128
	buffer_store_b16 v202, v228, s[44:47], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v201, v103 /*v359*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v200, v228, s[0:3], null offen offset:128
	buffer_store_b16 v203, v233, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v200, v208, s0
	v_or_b32_e32 v202, 0xc0, v216
	s_wait_xcnt 0x0
	v_or_b32_e32 v203, 0xc0, v217
	buffer_store_b16 v201, v233, s[0:3], null offen offset:128
	buffer_store_b16 v184, v202, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v184, v209, s0
	buffer_store_b16 v200, v202, s[0:3], null offen
	buffer_store_b16 v185, v203, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v202, v211, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v185, v186, s0
	v_or_b32_e32 v200, 0xc0, v222
	v_cvt_pk_bf16_f32 v186, v210, s0
	v_or_b32_e32 v201, 0xc0, v220
	buffer_store_b16 v184, v203, s[0:3], null offen
	buffer_store_b16 v185, v200, s[44:47], null offen
	buffer_store_b16 v186, v200, s[0:3], null offen
	buffer_store_b16 v187, v201, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e64 v184.l, v202.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v185, v188, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v187, 0xc0, v226
	v_cvt_pk_bf16_f32 v186, v212, s0
	v_cvt_pk_bf16_f32 v188, v189, s0
	v_or_b32_e32 v200, 0xc0, v227
	v_cvt_pk_bf16_f32 v189, v213, s0
	buffer_store_b16 v184, v201, s[0:3], null offen
	buffer_store_b16 v185, v187, s[44:47], null offen
	buffer_store_b16 v186, v187, s[0:3], null offen
	buffer_store_b16 v188, v200, s[44:47], null offen
	buffer_store_b16 v189, v200, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v184, v190, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v186, 0xc0, v223
	v_cvt_pk_bf16_f32 v185, v214, s0
	v_cvt_pk_bf16_f32 v187, v191, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v189, 0xc0, v230
	buffer_store_b16 v184, v186, s[44:47], null offen
	buffer_store_b16 v185, v186, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v185, v193, s0
	v_cvt_pk_bf16_f32 v188, v215, s0
	buffer_store_b16 v187, v189, s[44:47], null offen
	buffer_store_b16 v188, v189, s[0:3], null offen
	v_cvt_pk_bf16_f32 v184, v192, s0
	buffer_store_b16 v168, v219, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v168.l, v185.l
	buffer_store_b16 v184, v219, s[0:3], null offen offset:256
	buffer_store_b16 v169, v221, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v169, v194, s0
	buffer_store_b16 v168, v221, s[0:3], null offen offset:256
	buffer_store_b16 v170, v218, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v168, v171, s0
	v_cvt_pk_bf16_f32 v171, v172, s0
	v_cvt_pk_bf16_f32 v172, v196, s0
	buffer_store_b16 v169, v218, s[0:3], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v170, v195, s0
	buffer_store_b16 v168, v225, s[44:47], null offen offset:256
	buffer_store_b16 v170, v225, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e64 v169.l, v171.l
	v_mov_b16_e64 v171.l, v172.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v168, v173, s0
	buffer_store_b16 v169, v224, s[44:47], null offen offset:256
	buffer_store_b16 v171, v224, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v171, v198, s0
	v_cvt_pk_bf16_f32 v172, v175, s0
	v_cvt_pk_bf16_f32 v169, v197, s0
	buffer_store_b16 v168, v229, s[44:47], null offen offset:256
	v_cvt_pk_bf16_f32 v170, v174, s0
	s_wait_xcnt 0x0
	v_mov_b16_e64 v168.l, v171.l
	v_mov_b16_e64 v171.l, v172.l
	buffer_store_b16 v169, v229, s[0:3], null offen offset:256
	buffer_store_b16 v170, v228, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v169, v199, s0
	buffer_store_b16 v168, v228, s[0:3], null offen offset:256
	buffer_store_b16 v171, v233, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v168, v176, s0
	v_or_b32_e32 v170, 0x140, v216
	s_wait_xcnt 0x0
	v_or_b32_e32 v171, 0x140, v217
	buffer_store_b16 v169, v233, s[0:3], null offen offset:256
	buffer_store_b16 v152, v170, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v152, v177, s0
	buffer_store_b16 v168, v170, s[0:3], null offen
	buffer_store_b16 v153, v171, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v170, v179, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v153, v154, s0
	v_or_b32_e32 v168, 0x140, v222
	v_cvt_pk_bf16_f32 v154, v178, s0
	v_or_b32_e32 v169, 0x140, v220
	buffer_store_b16 v152, v171, s[0:3], null offen
	buffer_store_b16 v153, v168, s[44:47], null offen
	buffer_store_b16 v154, v168, s[0:3], null offen
	buffer_store_b16 v155, v169, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e64 v152.l, v170.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v153, v156, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v155, 0x140, v226
	v_cvt_pk_bf16_f32 v154, v180, s0
	v_cvt_pk_bf16_f32 v156, v157, s0
	v_or_b32_e32 v168, 0x140, v227
	v_cvt_pk_bf16_f32 v157, v181, s0
	buffer_store_b16 v152, v169, s[0:3], null offen
	buffer_store_b16 v153, v155, s[44:47], null offen
	buffer_store_b16 v154, v155, s[0:3], null offen
	buffer_store_b16 v156, v168, s[44:47], null offen
	buffer_store_b16 v157, v168, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v152, v158, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v154, 0x140, v223
	v_cvt_pk_bf16_f32 v153, v182, s0
	v_cvt_pk_bf16_f32 v155, v159, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v157, 0x140, v230
	buffer_store_b16 v152, v154, s[44:47], null offen
	buffer_store_b16 v153, v154, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v153, v161, s0
	v_cvt_pk_bf16_f32 v156, v183, s0
	buffer_store_b16 v155, v157, s[44:47], null offen
	buffer_store_b16 v156, v157, s[0:3], null offen
	v_cvt_pk_bf16_f32 v152, v160, s0
	buffer_store_b16 v136, v219, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v136.l, v153.l
	buffer_store_b16 v152, v219, s[0:3], null offen offset:384
	buffer_store_b16 v137, v221, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v137, v162, s0
	buffer_store_b16 v136, v221, s[0:3], null offen offset:384
	buffer_store_b16 v138, v218, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v136, v139, s0
	v_cvt_pk_bf16_f32 v139, v140, s0
	v_cvt_pk_bf16_f32 v140, v164, s0
	buffer_store_b16 v137, v218, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v138, v163, s0
	buffer_store_b16 v136, v225, s[44:47], null offen offset:384
	buffer_store_b16 v138, v225, s[0:3], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e64 v137.l, v139.l
	v_mov_b16_e64 v139.l, v140.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v136, v141, s0
	buffer_store_b16 v137, v224, s[44:47], null offen offset:384
	buffer_store_b16 v139, v224, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v139, v166, s0
	v_cvt_pk_bf16_f32 v140, v143, s0
	v_cvt_pk_bf16_f32 v137, v165, s0
	v_cvt_pk_bf16_f32 v138, v142, s0
	buffer_store_b16 v136, v229, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v136.l, v139.l
	v_mov_b16_e64 v139.l, v140.l
	buffer_store_b16 v137, v229, s[0:3], null offen offset:384
	buffer_store_b16 v138, v228, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v137, v167, s0
	buffer_store_b16 v136, v228, s[0:3], null offen offset:384
	buffer_store_b16 v139, v233, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v136, v144, s0
	v_or_b32_e32 v138, 0x1c0, v216
	s_wait_xcnt 0x0
	v_or_b32_e32 v139, 0x1c0, v217
	buffer_store_b16 v137, v233, s[0:3], null offen offset:384
	buffer_store_b16 v128, v138, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v128, v145, s0
	buffer_store_b16 v136, v138, s[0:3], null offen
	buffer_store_b16 v129, v139, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v129, v130, s0
	v_cvt_pk_bf16_f32 v130, v146, s0
	v_or_b32_e32 v136, 0x1c0, v222
	v_cvt_pk_bf16_f32 v131, v131, s0
	v_or_b32_e32 v137, 0x1c0, v220
	buffer_store_b16 v128, v139, s[0:3], null offen
	buffer_store_b16 v129, v136, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v128, v147, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v129, v132, s0
	v_cvt_pk_bf16_f32 v132, v133, s0
	v_cvt_pk_bf16_f32 v133, v149, s0
	buffer_store_b16 v130, v136, s[0:3], null offen
	buffer_store_b16 v131, v137, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v131, 0x1c0, v226
	v_cvt_pk_bf16_f32 v130, v148, s0
	v_or_b32_e32 v136, 0x1c0, v227
	buffer_store_b16 v128, v137, s[0:3], null offen
	buffer_store_b16 v129, v131, s[44:47], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v128.l, v133.l
	buffer_store_b16 v130, v131, s[0:3], null offen
	buffer_store_b16 v132, v136, s[44:47], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v132, 1, v32 /*v800*/
	v_cvt_pk_bf16_f32 v129, v134, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v131, 0x1c0, v223
	buffer_store_b16 v128, v136, s[0:3], null offen
	s_set_vgpr_msb 3
	v_mul_lo_u32 v128, v32 /*v800*/, s40
	v_mul_lo_u32 v132, s40, v132
	s_set_vgpr_msb 0x300
	v_cvt_pk_bf16_f32 v133, v135, s0
	v_cvt_pk_bf16_f32 v135, v151, s0
	v_cvt_pk_bf16_f32 v130, v150, s0
	v_or_b32_e32 v134, 0x1c0, v230
	buffer_store_b16 v129, v131, s[44:47], null offen
	buffer_store_b16 v130, v131, s[0:3], null offen
	v_add_lshl_u32 v128, v128, s4, 7
	s_wait_xcnt 0x1
	v_mov_b16_e64 v129.l, v135.l
	s_wait_xcnt 0x0
	v_add_lshl_u32 v131, v132, s4, 7
	buffer_store_b16 v133, v134, s[44:47], null offen
	s_set_vgpr_msb 2
	v_mul_lo_u32 v132, v164 /*v676*/, s40
	s_set_vgpr_msb 0x20c
	v_or_b32_e32 v130, v128, v31 /*v799*/
	buffer_store_b16 v129, v134, s[0:3], null offen
	s_set_vgpr_msb 0xc02
	v_mul_lo_u32 v133, v165 /*v677*/, s40
	s_set_vgpr_msb 0x200
	v_cvt_pk_bf16_f32 v112, v112, s0
	v_cvt_pk_bf16_f32 v120, v120, s0
	v_lshlrev_b32_e32 v129, 2, v130
	s_set_vgpr_msb 12
	v_or_b32_e32 v130, v131, v31 /*v799*/
	v_cvt_pk_bf16_f32 v121, v121, s0
	v_cvt_pk_bf16_f32 v113, v113, s0
	v_cvt_pk_bf16_f32 v114, v114, s0
	v_cvt_pk_bf16_f32 v115, v115, s0
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v130, 2, v130
	buffer_store_b16 v112, v129, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v112.l, v121.l
	v_add_lshl_u32 v121, v132, s4, 7
	buffer_store_b16 v120, v129, s[0:3], null offen
	buffer_store_b16 v113, v130, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v113, v133, s4, 7
	v_cvt_pk_bf16_f32 v120, v122, s0
	buffer_store_b16 v112, v130, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v112, v121, v31 /*v799*/
	s_set_vgpr_msb 0xc02
	v_mul_lo_u32 v132, v162 /*v674*/, s40
	s_set_vgpr_msb 0x20c
	v_or_b32_e32 v122, v113, v31 /*v799*/
	s_set_vgpr_msb 0xc02
	v_mul_lo_u32 v133, v163 /*v675*/, s40
	s_set_vgpr_msb 0x200
	v_cvt_pk_bf16_f32 v123, v123, s0
	v_lshlrev_b32_e32 v112, 2, v112
	v_cvt_pk_bf16_f32 v117, v117, s0
	v_lshlrev_b32_e32 v122, 2, v122
	v_cvt_pk_bf16_f32 v125, v125, s0
	v_add_lshl_u32 v132, v132, s4, 7
	buffer_store_b16 v114, v112, s[44:47], null offen
	buffer_store_b16 v120, v112, s[0:3], null offen
	buffer_store_b16 v115, v122, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v115, v133, s4, 7
	v_mov_b16_e32 v114.l, v123.l
	s_set_vgpr_msb 12
	v_or_b32_e32 v120, v132, v31 /*v799*/
	s_set_vgpr_msb 0xc02
	v_mul_lo_u32 v133, v161 /*v673*/, s40
	s_set_vgpr_msb 0x20c
	v_cvt_pk_bf16_f32 v119, v119, s0
	v_or_b32_e32 v123, v115, v31 /*v799*/
	v_cvt_pk_bf16_f32 v88, v88, s0
	v_cvt_pk_bf16_f32 v104, v104, s0
	v_cvt_pk_bf16_f32 v89, v89, s0
	v_cvt_pk_bf16_f32 v105, v105, s0
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v123, 2, v123
	buffer_store_b16 v114, v122, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v114, v116, s0
	v_lshlrev_b32_e32 v116, 2, v120
	v_cvt_pk_bf16_f32 v120, v124, s0
	s_set_vgpr_msb 2
	v_mul_lo_u32 v124, v160 /*v672*/, s40
	s_set_vgpr_msb 0x20c
	v_cvt_pk_bf16_f32 v90, v90, s0
	v_cvt_pk_bf16_f32 v91, v91, s0
	buffer_store_b16 v114, v116, s[44:47], null offen
	buffer_store_b16 v120, v116, s[0:3], null offen
	buffer_store_b16 v117, v123, s[44:47], null offen
	s_wait_xcnt 0x1
	v_add_lshl_u32 v120, v133, s4, 7
	v_mov_b16_e32 v114.l, v125.l
	v_cvt_pk_bf16_f32 v107, v107, s0
	v_add_lshl_u32 v124, v124, s4, 7
	v_cvt_pk_bf16_f32 v92, v92, s0
	v_or_b32_e32 v125, v120, v31 /*v799*/
	buffer_store_b16 v114, v123, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v114, v118, s0
	v_or_b32_e32 v117, v124, v31 /*v799*/
	v_cvt_pk_bf16_f32 v118, v126, s0
	v_cvt_pk_bf16_f32 v126, v127, s0
	v_or_b32_e32 v127, v128, v30 /*v798*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v125, 2, v125
	v_lshlrev_b32_e32 v117, 2, v117
	v_cvt_pk_bf16_f32 v93, v93, s0
	v_cvt_pk_bf16_f32 v95, v95, s0
	v_cvt_pk_bf16_f32 v72, v72, s0
	v_cvt_pk_bf16_f32 v73, v73, s0
	buffer_store_b16 v114, v117, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v114.l, v126.l
	buffer_store_b16 v118, v117, s[0:3], null offen
	buffer_store_b16 v119, v125, s[44:47], null offen
	s_wait_xcnt 0x1
	v_lshlrev_b32_e32 v118, 2, v127
	s_set_vgpr_msb 12
	v_or_b32_e32 v119, v131, v30 /*v798*/
	v_cvt_pk_bf16_f32 v74, v74, s0
	buffer_store_b16 v114, v125, s[0:3], null offen
	v_cvt_pk_bf16_f32 v56, v56, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v114, 64, v118
	v_cvt_pk_bf16_f32 v57, v57, s0
	v_cvt_pk_bf16_f32 v59, v59, s0
	v_cvt_pk_bf16_f32 v40, v40, s0
	v_cvt_pk_bf16_f32 v41, v41, s0
	buffer_store_b16 v88, v114, s[44:47], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v88, v121, v30 /*v798*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v119, 2, v119
	v_cvt_pk_bf16_f32 v42, v42, s0
	v_cvt_pk_bf16_f32 v24, v24, s0
	v_cvt_pk_bf16_f32 v25, v25, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_lshlrev_b32 v88, 2, v88 :: v_dual_bitop2_b32 v126, 64, v119 bitop3:0x54
	buffer_store_b16 v104, v114, s[0:3], null offen
	buffer_store_b16 v89, v126, s[44:47], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v89, v113, v30 /*v798*/
	v_or_b32_e32 v113, v132, v30 /*v798*/
	v_cvt_pk_bf16_f32 v104, v106, s0
	v_cvt_pk_bf16_f32 v27, v27, s0
	v_cvt_pk_bf16_f32 v8, v8, s0
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v89, 2, v89
	buffer_store_b16 v105, v126, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v105, 64, v88
	buffer_store_b16 v90, v105, s[44:47], null offen
	buffer_store_b16 v104, v105, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v104, v115, v30 /*v798*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v106, 64, v89
	v_mov_b16_e32 v90.l, v107.l
	v_cvt_pk_bf16_f32 v9, v9, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_lshlrev_b32_e32 v104, 2, v104
	buffer_store_b16 v91, v106, s[44:47], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v91, 2, v113
	buffer_store_b16 v90, v106, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v90, v108, s0
	v_cvt_pk_bf16_f32 v106, v109, s0
	v_or_b32_e32 v107, 64, v104
	s_set_vgpr_msb 12
	v_or_b32_e32 v108, v124, v30 /*v798*/
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v105, 64, v91
	v_cvt_pk_bf16_f32 v0, v0, s0
	v_cvt_pk_bf16_f32 v1, v1, s0
	v_cvt_pk_bf16_f32 v3, v3, s0
	buffer_store_b16 v92, v105, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v92.l, v106.l
	buffer_store_b16 v90, v105, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v90, 2, v108
	buffer_store_b16 v93, v107, s[44:47], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v93, v120, v30 /*v798*/
	v_cvt_pk_bf16_f32 v105, v110, s0
	v_cvt_pk_bf16_f32 v106, v111, s0
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v93, 2, v93
	buffer_store_b16 v92, v107, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v92, v94, s0
	v_or_b32_e32 v94, 64, v90
	buffer_store_b16 v92, v94, s[44:47], null offen
	buffer_store_b16 v105, v94, s[0:3], null offen
	v_or_b32_e32 v107, 64, v93
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v94, v97, s0
	buffer_store_b16 v95, v107, s[44:47], null offen
	buffer_store_b16 v106, v107, s[0:3], null offen
	v_cvt_pk_bf16_f32 v92, v96, s0
	buffer_store_b16 v72, v129, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v72.l, v94.l
	buffer_store_b16 v92, v129, s[0:3], null offen offset:128
	buffer_store_b16 v73, v130, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v73, v98, s0
	buffer_store_b16 v72, v130, s[0:3], null offen offset:128
	buffer_store_b16 v74, v112, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v72, v75, s0
	v_cvt_pk_bf16_f32 v75, v76, s0
	v_cvt_pk_bf16_f32 v76, v100, s0
	buffer_store_b16 v73, v112, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v74, v99, s0
	buffer_store_b16 v72, v122, s[44:47], null offen offset:128
	buffer_store_b16 v74, v122, s[0:3], null offen offset:128
	s_wait_xcnt 0x2
	v_mov_b16_e32 v73.l, v75.l
	v_mov_b16_e32 v75.l, v76.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v72, v77, s0
	buffer_store_b16 v73, v116, s[44:47], null offen offset:128
	buffer_store_b16 v75, v116, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v75, v102, s0
	v_cvt_pk_bf16_f32 v76, v79, s0
	v_cvt_pk_bf16_f32 v73, v101, s0
	buffer_store_b16 v72, v123, s[44:47], null offen offset:128
	v_cvt_pk_bf16_f32 v74, v78, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v72.l, v75.l
	v_mov_b16_e32 v75.l, v76.l
	buffer_store_b16 v73, v123, s[0:3], null offen offset:128
	buffer_store_b16 v74, v117, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v73, v103, s0
	buffer_store_b16 v72, v117, s[0:3], null offen offset:128
	buffer_store_b16 v75, v125, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v72, v80, s0
	v_or_b32_e32 v74, 0xc0, v118
	s_wait_xcnt 0x0
	v_or_b32_e32 v75, 0xc0, v119
	buffer_store_b16 v73, v125, s[0:3], null offen offset:128
	buffer_store_b16 v56, v74, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v56, v81, s0
	buffer_store_b16 v72, v74, s[0:3], null offen
	buffer_store_b16 v57, v75, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v74, v83, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v57, v58, s0
	v_or_b32_e32 v72, 0xc0, v88
	v_cvt_pk_bf16_f32 v58, v82, s0
	v_or_b32_e32 v73, 0xc0, v89
	buffer_store_b16 v56, v75, s[0:3], null offen
	buffer_store_b16 v57, v72, s[44:47], null offen
	buffer_store_b16 v58, v72, s[0:3], null offen
	buffer_store_b16 v59, v73, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v56.l, v74.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v57, v60, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v59, 0xc0, v91
	v_cvt_pk_bf16_f32 v58, v84, s0
	v_cvt_pk_bf16_f32 v60, v61, s0
	v_or_b32_e32 v72, 0xc0, v104
	v_cvt_pk_bf16_f32 v61, v85, s0
	buffer_store_b16 v56, v73, s[0:3], null offen
	buffer_store_b16 v57, v59, s[44:47], null offen
	buffer_store_b16 v58, v59, s[0:3], null offen
	buffer_store_b16 v60, v72, s[44:47], null offen
	buffer_store_b16 v61, v72, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v56, v62, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v58, 0xc0, v90
	v_cvt_pk_bf16_f32 v57, v86, s0
	v_cvt_pk_bf16_f32 v59, v63, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v61, 0xc0, v93
	buffer_store_b16 v56, v58, s[44:47], null offen
	buffer_store_b16 v57, v58, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v57, v65, s0
	v_cvt_pk_bf16_f32 v60, v87, s0
	buffer_store_b16 v59, v61, s[44:47], null offen
	buffer_store_b16 v60, v61, s[0:3], null offen
	v_cvt_pk_bf16_f32 v56, v64, s0
	buffer_store_b16 v40, v129, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e32 v40.l, v57.l
	buffer_store_b16 v56, v129, s[0:3], null offen offset:256
	buffer_store_b16 v41, v130, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v41, v66, s0
	buffer_store_b16 v40, v130, s[0:3], null offen offset:256
	buffer_store_b16 v42, v112, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v40, v43, s0
	v_cvt_pk_bf16_f32 v43, v44, s0
	v_cvt_pk_bf16_f32 v44, v68, s0
	buffer_store_b16 v41, v112, s[0:3], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v42, v67, s0
	buffer_store_b16 v40, v122, s[44:47], null offen offset:256
	buffer_store_b16 v42, v122, s[0:3], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e32 v41.l, v43.l
	v_mov_b16_e32 v43.l, v44.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v40, v45, s0
	buffer_store_b16 v41, v116, s[44:47], null offen offset:256
	buffer_store_b16 v43, v116, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v43, v70, s0
	v_cvt_pk_bf16_f32 v44, v47, s0
	v_cvt_pk_bf16_f32 v41, v69, s0
	buffer_store_b16 v40, v123, s[44:47], null offen offset:256
	v_cvt_pk_bf16_f32 v42, v46, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v40.l, v43.l
	v_mov_b16_e32 v43.l, v44.l
	buffer_store_b16 v41, v123, s[0:3], null offen offset:256
	buffer_store_b16 v42, v117, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v41, v71, s0
	buffer_store_b16 v40, v117, s[0:3], null offen offset:256
	buffer_store_b16 v43, v125, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v40, v48, s0
	v_or_b32_e32 v42, 0x140, v118
	s_wait_xcnt 0x0
	v_or_b32_e32 v43, 0x140, v119
	buffer_store_b16 v41, v125, s[0:3], null offen offset:256
	buffer_store_b16 v24, v42, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v24, v49, s0
	buffer_store_b16 v40, v42, s[0:3], null offen
	buffer_store_b16 v25, v43, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v42, v51, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v25, v26, s0
	v_or_b32_e32 v40, 0x140, v88
	v_cvt_pk_bf16_f32 v26, v50, s0
	v_or_b32_e32 v41, 0x140, v89
	buffer_store_b16 v24, v43, s[0:3], null offen
	buffer_store_b16 v25, v40, s[44:47], null offen
	buffer_store_b16 v26, v40, s[0:3], null offen
	buffer_store_b16 v27, v41, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v24.l, v42.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v25, v28, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v27, 0x140, v91
	v_cvt_pk_bf16_f32 v26, v52, s0
	v_cvt_pk_bf16_f32 v28, v29, s0
	v_or_b32_e32 v40, 0x140, v104
	v_cvt_pk_bf16_f32 v29, v53, s0
	buffer_store_b16 v24, v41, s[0:3], null offen
	buffer_store_b16 v25, v27, s[44:47], null offen
	buffer_store_b16 v26, v27, s[0:3], null offen
	buffer_store_b16 v28, v40, s[44:47], null offen
	buffer_store_b16 v29, v40, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v24, v30, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v26, 0x140, v90
	v_cvt_pk_bf16_f32 v25, v54, s0
	v_cvt_pk_bf16_f32 v27, v31, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v29, 0x140, v93
	buffer_store_b16 v24, v26, s[44:47], null offen
	buffer_store_b16 v25, v26, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v25, v33, s0
	v_cvt_pk_bf16_f32 v28, v55, s0
	buffer_store_b16 v27, v29, s[44:47], null offen
	buffer_store_b16 v28, v29, s[0:3], null offen
	v_cvt_pk_bf16_f32 v24, v32, s0
	buffer_store_b16 v8, v129, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v8.l, v25.l
	buffer_store_b16 v24, v129, s[0:3], null offen offset:384
	buffer_store_b16 v9, v130, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v9, v34, s0
	buffer_store_b16 v8, v130, s[0:3], null offen offset:384
	buffer_store_b16 v10, v112, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v8, v11, s0
	v_cvt_pk_bf16_f32 v11, v12, s0
	v_cvt_pk_bf16_f32 v12, v36, s0
	buffer_store_b16 v9, v112, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v10, v35, s0
	buffer_store_b16 v8, v122, s[44:47], null offen offset:384
	buffer_store_b16 v10, v122, s[0:3], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e32 v9.l, v11.l
	v_mov_b16_e32 v11.l, v12.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v8, v13, s0
	buffer_store_b16 v9, v116, s[44:47], null offen offset:384
	buffer_store_b16 v11, v116, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v11, v38, s0
	v_cvt_pk_bf16_f32 v12, v15, s0
	v_cvt_pk_bf16_f32 v9, v37, s0
	buffer_store_b16 v8, v123, s[44:47], null offen offset:384
	v_cvt_pk_bf16_f32 v10, v14, s0
	s_wait_xcnt 0x0
	v_mov_b16_e32 v8.l, v11.l
	v_mov_b16_e32 v11.l, v12.l
	buffer_store_b16 v9, v123, s[0:3], null offen offset:384
	buffer_store_b16 v10, v117, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v9, v39, s0
	buffer_store_b16 v8, v117, s[0:3], null offen offset:384
	buffer_store_b16 v11, v125, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v8, v16, s0
	v_or_b32_e32 v10, 0x1c0, v118
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 0x1c0, v119
	buffer_store_b16 v9, v125, s[0:3], null offen offset:384
	buffer_store_b16 v0, v10, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v0, v17, s0
	buffer_store_b16 v8, v10, s[0:3], null offen
	buffer_store_b16 v1, v11, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v2, s0
	v_cvt_pk_bf16_f32 v2, v18, s0
	v_or_b32_e32 v8, 0x1c0, v88
	v_or_b32_e32 v9, 0x1c0, v89
	buffer_store_b16 v0, v11, s[0:3], null offen
	buffer_store_b16 v1, v8, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v0, v19, s0
	buffer_store_b16 v2, v8, s[0:3], null offen
	buffer_store_b16 v3, v9, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v8, v21, s0
	v_cvt_pk_bf16_f32 v1, v4, s0
	v_or_b32_e32 v4, 0x1c0, v91
	v_cvt_pk_bf16_f32 v2, v20, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v3, v5, s0
	v_or_b32_e32 v5, 0x1c0, v104
	buffer_store_b16 v0, v9, s[0:3], null offen
	buffer_store_b16 v1, v4, s[44:47], null offen
	buffer_store_b16 v2, v4, s[0:3], null offen
	buffer_store_b16 v3, v5, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v0.l, v8.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v1, v6, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, 0x1c0, v90
	v_cvt_pk_bf16_f32 v2, v22, s0
	v_cvt_pk_bf16_f32 v4, v7, s0
	v_or_b32_e32 v7, 0x1c0, v93
	v_cvt_pk_bf16_f32 v6, v23, s0
	buffer_store_b16 v0, v5, s[0:3], null offen
	buffer_store_b16 v1, v3, s[44:47], null offen
	buffer_store_b16 v2, v3, s[0:3], null offen
	buffer_store_b16 v4, v7, s[44:47], null offen
	buffer_store_b16 v6, v7, s[0:3], null offen
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	k_dkdv_0, .Lfunc_end0-k_dkdv_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel k_dkdv_0
		.amdhsa_group_segment_fixed_size 82944
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
		.amdhsa_next_free_vgpr 832
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

	.set .Lk_dkdv_0.num_vgpr, 832
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
    .group_segment_fixed_size: 82944
    .kernarg_segment_align: 8
    .kernarg_segment_size: 408
    .max_flat_workgroup_size: 128
    .name:           k_dkdv_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     80
    .sgpr_spill_count: 0
    .symbol:         k_dkdv_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     832
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
