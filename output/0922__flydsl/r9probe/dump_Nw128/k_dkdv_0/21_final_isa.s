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
	s_load_b256 s[16:23], s[0:1], 0x170 nv
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
	v_dual_lshrrev_b32 v143 /*v911*/, 4, v0 :: v_dual_bitop2_b32 v141 /*v909*/, 15, v0 bitop3:0x40
	s_cselect_b32 s8, s6, s5
	s_cselect_b32 s4, s3, s4
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_wait_kmcnt 0x0
	s_mul_i32 s51, s18, s8
	s_mul_i32 s2, ttmp9, s2
	s_load_b64 s[24:25], s[0:1], 0x30 nv
	s_add_co_i32 s5, s3, s2
	s_cmp_eq_u32 s7, 0
	s_load_b64 s[2:3], s[0:1], 0x190 nv
	s_cselect_b32 s50, ttmp9, s5
	s_lshr_b32 s5, s22, 31
	s_lshl_b32 s9, s4, 5
	s_add_co_i32 s5, s22, s5
	s_set_vgpr_msb 0xc00c
	v_or_b32_e32 v1, s9, v141 /*v909*/
	s_and_b32 s4, s5, -2
	s_ashr_i32 s5, s5, 1
	s_cmp_lg_u32 s22, s4
	s_mul_i32 s53, s17, s8
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s22, 0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v2, 16, v1
	s_cselect_b32 s6, -1, 0
	s_mul_i32 s52, s21, s50
	s_and_b32 s4, s6, s4
	s_sub_co_ci_u32 s6, s5, 0
	s_sub_co_i32 s7, s9, s23
	s_clause 0x1
	s_load_b64 s[28:29], s[0:1], 0x90 nv
	s_load_b64 s[36:37], s[0:1], 0x0 nv
	s_max_i32 s7, s7, 0
	s_set_vgpr_msb 0x82
	v_mov_b32_e32 v2 /*v514*/, 0
	s_lshr_b32 s7, s7, 5
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s2, 0
	s_mov_b32 s12, 0
	s_cselect_b32 s10, -1, 0
	v_dual_mov_b32 v3 /*v515*/, v2 /*v514*/ :: v_dual_mov_b32 v4 /*v516*/, v2 /*v514*/
	s_and_b32 s10, s10, exec_lo
	s_cselect_b32 s58, s7, 0
	s_cmp_lg_u32 s4, 0
	v_dual_mov_b32 v5 /*v517*/, v2 /*v514*/ :: v_dual_mov_b32 v6 /*v518*/, v2 /*v514*/
	s_sub_co_ci_u32 s59, s5, s58
	s_or_b32 s4, s9, 31
	v_dual_mov_b32 v7 /*v519*/, v2 /*v514*/ :: v_dual_mov_b32 v8 /*v520*/, v2 /*v514*/
	s_sub_co_i32 s4, s4, s23
	v_mov_b32_e32 v9 /*v521*/, v2 /*v514*/
	s_add_co_i32 s5, s4, 31
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b64_e32 v[14:15] /*v[526:527]*/, v[6:7] /*v[518:519]*/
	s_ashr_i32 s7, s5, 31
	v_mov_b64_e32 v[12:13] /*v[524:525]*/, v[4:5] /*v[516:517]*/
	s_lshr_b32 s7, s7, 27
	v_mov_b64_e32 v[16:17] /*v[528:529]*/, v[8:9] /*v[520:521]*/
	s_add_co_i32 s7, s5, s7
	v_mov_b64_e32 v[10:11] /*v[522:523]*/, v[2:3] /*v[514:515]*/
	s_and_b32 s10, s7, 0xffffffe0
	s_ashr_i32 s7, s7, 5
	s_cmp_lg_u32 s5, s10
	v_mov_b64_e32 v[24:25] /*v[536:537]*/, v[8:9] /*v[520:521]*/
	s_cselect_b32 s10, -1, 0
	s_cmp_lt_i32 s5, 0
	v_mov_b64_e32 v[22:23] /*v[534:535]*/, v[6:7] /*v[518:519]*/
	s_cselect_b32 s5, -1, 0
	v_mov_b64_e32 v[20:21] /*v[532:533]*/, v[4:5] /*v[516:517]*/
	s_and_b32 s5, s5, s10
	s_sub_co_ci_u32 s5, s7, 0
	s_cmp_gt_i32 s4, -1
	s_mul_i32 s10, s19, s17
	s_cselect_b32 s4, s5, 0
	v_mov_b64_e32 v[18:19] /*v[530:531]*/, v[2:3] /*v[514:515]*/
	s_min_i32 s4, s4, s6
	v_mov_b64_e32 v[40:41] /*v[552:553]*/, v[8:9] /*v[520:521]*/
	s_sub_co_i32 s4, s4, s58
	v_mov_b64_e32 v[38:39] /*v[550:551]*/, v[6:7] /*v[518:519]*/
	s_max_i32 s4, s4, 0
	v_mov_b64_e32 v[36:37] /*v[548:549]*/, v[4:5] /*v[516:517]*/
	s_min_i32 s4, s4, s59
	s_cmp_lg_u32 s2, 0
	v_mov_b64_e32 v[34:35] /*v[546:547]*/, v[2:3] /*v[514:515]*/
	s_cselect_b32 s61, -1, 0
	v_mov_b64_e32 v[56:57] /*v[568:569]*/, v[8:9] /*v[520:521]*/
	s_and_b32 s2, s61, exec_lo
	s_cselect_b32 s60, s4, 0
	s_lshl_b32 s33, s20, 4
	s_mul_i32 s4, s18, s20
	s_mul_i32 s2, s33, s51
	s_abs_i32 s55, s21
	s_lshl4_add_u32 s2, s50, s2
	v_mov_b64_e32 v[54:55] /*v[566:567]*/, v[6:7] /*v[518:519]*/
	s_set_vgpr_msb 0x8200
	v_mad_u32 v1, s33, v1, s2
	v_mad_u32 v2, s33, v2, s2
	s_mul_i32 s2, s4, s3
	s_load_b64 s[4:5], s[0:1], 0x60 nv
	s_lshl_b32 s34, s2, 8
	s_set_vgpr_msb 0x82
	v_mov_b64_e32 v[52:53] /*v[564:565]*/, v[4:5] /*v[516:517]*/
	s_ashr_i32 s35, s34, 31
	v_mov_b64_e32 v[50:51] /*v[562:563]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x820c
	v_or_b32_e32 v1, v1, v143 /*v911*/
	v_or_b32_e32 v2, v2, v143 /*v911*/
	s_lshr_b64 s[26:27], s[34:35], 7
	s_lshl_b32 s35, s19, 4
	s_mov_b32 s6, s26
	s_set_vgpr_msb 0xc00
	v_dual_lshlrev_b32 v1, 4, v1 :: v_dual_lshlrev_b32 v2, 4, v2
	s_mov_b32 s7, s27
	s_clause 0x3
	buffer_load_b128 v[170:173], v1, s[24:27], null offen
	buffer_load_b128 v[174:177], v1, s[24:27], null offen offset:32
	buffer_load_b128 v[178:181], v1, s[24:27], null offen offset:64
	buffer_load_b128 v[182:185], v1, s[24:27], null offen offset:96
	v_add_nc_u32_e32 v3, 0xa0, v1
	v_add_nc_u32_e32 v4, 0xc0, v1
	v_add_nc_u32_e32 v5, 0xe0, v1
	s_wait_kmcnt 0x0
	s_clause 0x3
	buffer_load_b128 v[194:197], v1, s[4:7], null offen
	buffer_load_b128 v[198:201], v1, s[4:7], null offen offset:32
	buffer_load_b128 v[202:205], v1, s[4:7], null offen offset:64
	buffer_load_b128 v[206:209], v1, s[4:7], null offen offset:96
	v_add_nc_u32_e32 v6, 0xa0, v2
	s_clause 0x1
	buffer_load_b128 v[210:213], v1, s[24:27], null offen offset:128
	buffer_load_b128 v[214:217], v3, s[24:27], null offen
	s_clause 0x1
	buffer_load_b128 v[218:221], v1, s[4:7], null offen offset:128
	buffer_load_b128 v[222:225], v3, s[4:7], null offen
	s_clause 0x1
	buffer_load_b128 v[226:229], v4, s[24:27], null offen
	buffer_load_b128 v[230:233], v5, s[24:27], null offen
	s_clause 0x1
	buffer_load_b128 v[234:237], v4, s[4:7], null offen
	buffer_load_b128 v[238:241], v5, s[4:7], null offen
	s_clause 0x1
	buffer_load_b128 v[242:245], v2, s[24:27], null offen offset:128
	buffer_load_b128 v[246:249], v6, s[24:27], null offen
	s_clause 0x1
	buffer_load_b128 v[254:257], v6, s[4:7], null offen
	buffer_load_b128 v[250:253], v2, s[4:7], null offen offset:128
	s_set_vgpr_msb 64
	s_clause 0x3
	buffer_load_b128 v[2:5] /*v[258:261]*/, v2, s[24:27], null offen
	buffer_load_b128 v[6:9] /*v[262:265]*/, v2, s[24:27], null offen offset:32
	buffer_load_b128 v[10:13] /*v[266:269]*/, v2, s[24:27], null offen offset:64
	buffer_load_b128 v[14:17] /*v[270:273]*/, v2, s[24:27], null offen offset:96
	s_set_vgpr_msb 0x4030
	v_lshl_or_b32 v1, s58, 5, v141 /*v909*/
	s_mul_i32 s53, s53, s35
	s_set_vgpr_msb 0x3040
	s_clause 0x1
	buffer_load_b128 v[18:21] /*v[274:277]*/, v2, s[4:7], null offen
	buffer_load_b128 v[22:25] /*v[278:281]*/, v2, s[4:7], null offen offset:32
	s_lshl4_add_u32 s2, s52, s53
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v4, 0xc0, v2
	v_or_b32_e32 v3, 16, v1
	v_mad_u32 v1, v1, s35, s2
	v_add_nc_u32_e32 v5, 0xe0, v2
	s_set_vgpr_msb 64
	s_clause 0x1
	buffer_load_b128 v[26:29] /*v[282:285]*/, v2, s[4:7], null offen offset:64
	buffer_load_b128 v[30:33] /*v[286:289]*/, v2, s[4:7], null offen offset:96
	s_clause 0x1
	buffer_load_b128 v[42:45] /*v[298:301]*/, v4, s[24:27], null offen
	buffer_load_b128 v[46:49] /*v[302:305]*/, v5, s[24:27], null offen
	s_set_vgpr_msb 0x400c
	v_mad_u32 v3, v3, s35, s2
	s_mul_i32 s2, s10, s3
	v_or_b32_e32 v1, v1, v143 /*v911*/
	s_lshl_b32 s10, s2, 8
	s_set_vgpr_msb 0xc40
	s_clause 0x1
	buffer_load_b128 v[50:53] /*v[306:309]*/, v4, s[4:7], null offen
	buffer_load_b128 v[54:57] /*v[310:313]*/, v5, s[4:7], null offen
	s_ashr_i32 s11, s10, 31
	s_lshl_b32 s13, s2, 27
	s_set_vgpr_msb 0x4000
	v_lshlrev_b32_e32 v1, 4, v1
	s_set_vgpr_msb 12
	v_or_b32_e32 v2, v3, v143 /*v911*/
	s_lshr_b64 s[38:39], s[10:11], 7
	s_lshl_b32 s10, s2, 2
	s_mov_b32 s30, s38
	s_mov_b32 s31, s39
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v2, 4, v2
	v_add_nc_u32_e32 v3, 0xa0, v1
	v_add_nc_u32_e32 v4, 0xc0, v1
	v_add_nc_u32_e32 v5, 0xe0, v1
	s_mul_i32 s2, s19, s8
	v_add_nc_u32_e32 v6, 0xa0, v2
	v_add_nc_u32_e32 v7, 0xc0, v2
	v_add_nc_u32_e32 v8, 0xe0, v2
	s_set_vgpr_msb 0xc0
	s_clause 0x3
	buffer_load_b128 v[2:5] /*v[770:773]*/, v1, s[36:39], null offen
	buffer_load_b128 v[6:9] /*v[774:777]*/, v1, s[36:39], null offen offset:32
	buffer_load_b128 v[10:13] /*v[778:781]*/, v1, s[36:39], null offen offset:64
	buffer_load_b128 v[14:17] /*v[782:785]*/, v1, s[36:39], null offen offset:96
	s_clause 0x3
	buffer_load_b128 v[30:33] /*v[798:801]*/, v1, s[28:31], null offen
	buffer_load_b128 v[38:41] /*v[806:809]*/, v1, s[28:31], null offen offset:32
	buffer_load_b128 v[42:45] /*v[810:813]*/, v1, s[28:31], null offen offset:64
	buffer_load_b128 v[46:49] /*v[814:817]*/, v1, s[28:31], null offen offset:96
	s_clause 0x1
	buffer_load_b128 v[18:21] /*v[786:789]*/, v1, s[36:39], null offen offset:128
	buffer_load_b128 v[22:25] /*v[790:793]*/, v3, s[36:39], null offen
	s_clause 0x1
	buffer_load_b128 v[50:53] /*v[818:821]*/, v1, s[28:31], null offen offset:128
	buffer_load_b128 v[54:57] /*v[822:825]*/, v3, s[28:31], null offen
	s_clause 0x1
	buffer_load_b128 v[26:29] /*v[794:797]*/, v4, s[36:39], null offen
	buffer_load_b128 v[34:37] /*v[802:805]*/, v5, s[36:39], null offen
	s_clause 0x1
	buffer_load_b128 v[58:61] /*v[826:829]*/, v4, s[28:31], null offen
	buffer_load_b128 v[62:65] /*v[830:833]*/, v5, s[28:31], null offen
	s_clause 0x3
	buffer_load_b128 v[66:69] /*v[834:837]*/, v2, s[36:39], null offen
	buffer_load_b128 v[70:73] /*v[838:841]*/, v2, s[36:39], null offen offset:32
	buffer_load_b128 v[74:77] /*v[842:845]*/, v2, s[36:39], null offen offset:64
	buffer_load_b128 v[78:81] /*v[846:849]*/, v2, s[36:39], null offen offset:96
	s_clause 0x3
	buffer_load_b128 v[90:93] /*v[858:861]*/, v2, s[28:31], null offen
	buffer_load_b128 v[98:101] /*v[866:869]*/, v2, s[28:31], null offen offset:32
	buffer_load_b128 v[106:109] /*v[874:877]*/, v2, s[28:31], null offen offset:64
	buffer_load_b128 v[110:113] /*v[878:881]*/, v2, s[28:31], null offen offset:96
	s_clause 0x1
	buffer_load_b128 v[82:85] /*v[850:853]*/, v2, s[36:39], null offen offset:128
	buffer_load_b128 v[86:89] /*v[854:857]*/, v6, s[36:39], null offen
	s_clause 0x1
	buffer_load_b128 v[114:117] /*v[882:885]*/, v2, s[28:31], null offen offset:128
	buffer_load_b128 v[118:121] /*v[886:889]*/, v6, s[28:31], null offen
	s_clause 0x1
	buffer_load_b128 v[94:97] /*v[862:865]*/, v7, s[36:39], null offen
	buffer_load_b128 v[102:105] /*v[870:873]*/, v8, s[36:39], null offen
	s_clause 0x1
	buffer_load_b128 v[122:125] /*v[890:893]*/, v7, s[28:31], null offen
	buffer_load_b128 v[126:129] /*v[894:897]*/, v8, s[28:31], null offen
	s_clause 0x1
	s_load_b64 s[4:5], s[0:1], 0xc0 nv
	s_load_b64 s[6:7], s[0:1], 0xe8 nv
	s_add_co_i32 s54, s52, s2
	s_cvt_f32_u32 s2, s55
	s_set_vgpr_msb 0xc00c
	v_lshlrev_b32_e32 v2, 3, v143 /*v911*/
	s_set_vgpr_msb 0xc00
	v_and_b32_e32 v1, 0x70, v0
	v_lshlrev_b32_e32 v4, 1, v0
	v_s_rcp_f32 s2, s2
	s_mul_i32 s3, s60, s21
	s_set_vgpr_msb 0xc0
	v_add_nc_u32_e32 v144 /*v912*/, s9, v2
	s_set_vgpr_msb 0xc00c
	v_mad_u32_u24 v1, 0x110, v141 /*v909*/, v1
	v_and_or_b32 v3, v0, 7, v2
	s_set_vgpr_msb 0xcc0
	v_and_b32_e32 v147 /*v915*/, 16, v4
	s_max_i32 s24, s3, 0
	s_set_vgpr_msb 0xc0cc
	v_add_nc_u32_e32 v142 /*v910*/, 16, v144 /*v912*/
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_sub_co_i32 s3, 0, s55
	s_set_vgpr_msb 0xcc82
	v_mov_b64_e32 v[72:73] /*v[584:585]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[70:71] /*v[582:583]*/, v[6:7] /*v[518:519]*/
	s_cvt_u32_f32 s2, s2
	s_wait_kmcnt 0x0
	s_or_b64 s[40:41], s[4:5], s[12:13]
	s_or_b64 s[44:45], s[6:7], s[12:13]
	s_movk_i32 s4, 0x1400
	s_movk_i32 s5, 0x3600
	s_movk_i32 s6, 0xa00
	s_mul_i32 s3, s3, s2
	v_mov_b64_e32 v[68:69] /*v[580:581]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[66:67] /*v[578:579]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[88:89] /*v[600:601]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[86:87] /*v[598:599]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[84:85] /*v[596:597]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[82:83] /*v[594:595]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[104:105] /*v[616:617]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[102:103] /*v[614:615]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[100:101] /*v[612:613]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[98:99] /*v[610:611]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[136:137] /*v[648:649]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[134:135] /*v[646:647]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[132:133] /*v[644:645]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[130:131] /*v[642:643]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[144:145] /*v[656:657]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[142:143] /*v[654:655]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[140:141] /*v[652:653]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[138:139] /*v[650:651]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[152:153] /*v[664:665]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[150:151] /*v[662:663]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[148:149] /*v[660:661]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[146:147] /*v[658:659]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[160:161] /*v[672:673]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[158:159] /*v[670:671]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[156:157] /*v[668:669]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[154:155] /*v[666:667]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[184:185] /*v[696:697]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[182:183] /*v[694:695]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[180:181] /*v[692:693]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[178:179] /*v[690:691]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[208:209] /*v[720:721]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[206:207] /*v[718:719]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[204:205] /*v[716:717]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[202:203] /*v[714:715]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[224:225] /*v[736:737]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[222:223] /*v[734:735]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[220:221] /*v[732:733]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[218:219] /*v[730:731]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[240:241] /*v[752:753]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[238:239] /*v[750:751]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[236:237] /*v[748:749]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[234:235] /*v[746:747]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[32:33] /*v[544:545]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[30:31] /*v[542:543]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[28:29] /*v[540:541]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[26:27] /*v[538:539]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[48:49] /*v[560:561]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[46:47] /*v[558:559]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[44:45] /*v[556:557]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[42:43] /*v[554:555]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[64:65] /*v[576:577]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[62:63] /*v[574:575]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[60:61] /*v[572:573]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[58:59] /*v[570:571]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[80:81] /*v[592:593]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[78:79] /*v[590:591]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[76:77] /*v[588:589]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[74:75] /*v[586:587]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[96:97] /*v[608:609]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[94:95] /*v[606:607]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[92:93] /*v[604:605]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[90:91] /*v[602:603]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[112:113] /*v[624:625]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[110:111] /*v[622:623]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[108:109] /*v[620:621]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[106:107] /*v[618:619]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[120:121] /*v[632:633]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[118:119] /*v[630:631]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[116:117] /*v[628:629]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[114:115] /*v[626:627]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[128:129] /*v[640:641]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[126:127] /*v[638:639]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[124:125] /*v[636:637]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[122:123] /*v[634:635]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[168:169] /*v[680:681]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[166:167] /*v[678:679]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[164:165] /*v[676:677]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[162:163] /*v[674:675]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[176:177] /*v[688:689]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[174:175] /*v[686:687]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[172:173] /*v[684:685]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[170:171] /*v[682:683]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[192:193] /*v[704:705]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[190:191] /*v[702:703]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[188:189] /*v[700:701]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[186:187] /*v[698:699]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[200:201] /*v[712:713]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[198:199] /*v[710:711]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[196:197] /*v[708:709]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[194:195] /*v[706:707]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[216:217] /*v[728:729]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[214:215] /*v[726:727]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[212:213] /*v[724:725]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[210:211] /*v[722:723]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[232:233] /*v[744:745]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[230:231] /*v[742:743]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[228:229] /*v[740:741]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[226:227] /*v[738:739]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[248:249] /*v[760:761]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[246:247] /*v[758:759]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[244:245] /*v[756:757]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[242:243] /*v[754:755]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x82c2
	v_mov_b64_e32 v[0:1] /*v[768:769]*/, v[8:9] /*v[520:521]*/
	s_set_vgpr_msb 0xc282
	v_mov_b64_e32 v[254:255] /*v[766:767]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[252:253] /*v[764:765]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[250:251] /*v[762:763]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x82c0
	v_add_nc_u32_e32 v145 /*v913*/, 0x1400, v1
	s_set_vgpr_msb 0xc0cc
	v_mad_i32_i24 v146 /*v914*/, 0xffffff40, v141 /*v909*/, v1
	s_set_vgpr_msb 0xccc0
	v_and_b32_e32 v158 /*v926*/, 0xf0, v4
	v_mad_u32_u24 v148 /*v916*/, 0x110, v3, s4
	v_mad_u32_u24 v149 /*v917*/, 0x110, v3, s5
	v_mul_u32_u24_e32 v150 /*v918*/, 0x50, v3
	v_mad_u32_u24 v151 /*v919*/, 0x50, v3, s6
	s_set_vgpr_msb 0xc0cc
	v_or_b32_e32 v152 /*v920*/, 32, v147 /*v915*/
	v_or_b32_e32 v153 /*v921*/, 64, v147 /*v915*/
	v_or_b32_e32 v154 /*v922*/, 0x60, v147 /*v915*/
	v_or_b32_e32 v155 /*v923*/, 0x80, v147 /*v915*/
	v_or_b32_e32 v156 /*v924*/, 0xa0, v147 /*v915*/
	v_or_b32_e32 v157 /*v925*/, 0xc0, v147 /*v915*/
	v_or_b32_e32 v159 /*v927*/, 0xe0, v147 /*v915*/
	v_or_b32_e32 v139 /*v907*/, 3, v144 /*v912*/
	v_or_b32_e32 v140 /*v908*/, 2, v144 /*v912*/
	v_or_b32_e32 v137 /*v905*/, 5, v144 /*v912*/
	v_or_b32_e32 v138 /*v906*/, 4, v144 /*v912*/
	v_or_b32_e32 v135 /*v903*/, 7, v144 /*v912*/
	v_or_b32_e32 v136 /*v904*/, 6, v144 /*v912*/
	v_or_b32_e32 v133 /*v901*/, 3, v142 /*v910*/
	v_or_b32_e32 v134 /*v902*/, 2, v142 /*v910*/
	v_or_b32_e32 v131 /*v899*/, 5, v142 /*v910*/
	v_or_b32_e32 v132 /*v900*/, 4, v142 /*v910*/
	s_set_vgpr_msb 0xcc0c
	v_or_b32_e32 v1, 7, v142 /*v910*/
	s_set_vgpr_msb 0xccc
	v_or_b32_e32 v130 /*v898*/, 6, v142 /*v910*/
	s_ashr_i32 s11, s10, 31
	s_mul_hi_u32 s3, s2, s3
	s_lshr_b64 s[42:43], s[10:11], 7
	s_mov_b32 s25, s12
	s_mov_b32 s18, s16
	s_mov_b32 s19, s16
	s_ashr_i32 s56, s21, 31
	s_add_co_i32 s57, s2, s3
	s_mov_b64 s[48:49], 0
	s_mov_b32 s22, 0x3fb8aa3b
	s_set_vgpr_msb 0xcc00
	s_branch .LBB0_2
.LBB0_1:
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s2, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s2, 1
	s_cbranch_scc0 .LBB0_4
.LBB0_2:
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 2
	v_mov_b64_e32 v[2:3], v[250:251] /*v[762:763]*/
	v_mov_b64_e32 v[4:5], v[252:253] /*v[764:765]*/
	v_mov_b64_e32 v[6:7], v[254:255] /*v[766:767]*/
	s_set_vgpr_msb 0x203
	v_mov_b64_e32 v[8:9], v[0:1] /*v[768:769]*/
	s_set_vgpr_msb 0x302
	v_mov_b64_e32 v[18:19], v[242:243] /*v[754:755]*/
	v_mov_b64_e32 v[20:21], v[244:245] /*v[756:757]*/
	v_mov_b64_e32 v[22:23], v[246:247] /*v[758:759]*/
	v_mov_b64_e32 v[24:25], v[248:249] /*v[760:761]*/
	v_mov_b64_e32 v[34:35], v[226:227] /*v[738:739]*/
	v_mov_b64_e32 v[36:37], v[228:229] /*v[740:741]*/
	v_mov_b64_e32 v[38:39], v[230:231] /*v[742:743]*/
	v_mov_b64_e32 v[40:41], v[232:233] /*v[744:745]*/
	v_mov_b64_e32 v[50:51], v[210:211] /*v[722:723]*/
	v_mov_b64_e32 v[52:53], v[212:213] /*v[724:725]*/
	v_mov_b64_e32 v[54:55], v[214:215] /*v[726:727]*/
	v_mov_b64_e32 v[56:57], v[216:217] /*v[728:729]*/
	v_mov_b64_e32 v[66:67], v[194:195] /*v[706:707]*/
	v_mov_b64_e32 v[68:69], v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[70:71], v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[72:73], v[200:201] /*v[712:713]*/
	v_mov_b64_e32 v[82:83], v[186:187] /*v[698:699]*/
	v_mov_b64_e32 v[84:85], v[188:189] /*v[700:701]*/
	v_mov_b64_e32 v[86:87], v[190:191] /*v[702:703]*/
	v_mov_b64_e32 v[88:89], v[192:193] /*v[704:705]*/
	v_mov_b64_e32 v[98:99], v[170:171] /*v[682:683]*/
	v_mov_b64_e32 v[100:101], v[172:173] /*v[684:685]*/
	v_mov_b64_e32 v[102:103], v[174:175] /*v[686:687]*/
	v_mov_b64_e32 v[104:105], v[176:177] /*v[688:689]*/
	v_mov_b64_e32 v[114:115], v[162:163] /*v[674:675]*/
	v_mov_b64_e32 v[116:117], v[164:165] /*v[676:677]*/
	v_mov_b64_e32 v[118:119], v[166:167] /*v[678:679]*/
	v_mov_b64_e32 v[120:121], v[168:169] /*v[680:681]*/
	v_mov_b64_e32 v[130:131], v[122:123] /*v[634:635]*/
	v_mov_b64_e32 v[132:133], v[124:125] /*v[636:637]*/
	v_mov_b64_e32 v[134:135], v[126:127] /*v[638:639]*/
	v_mov_b64_e32 v[136:137], v[128:129] /*v[640:641]*/
	v_mov_b64_e32 v[146:147], v[114:115] /*v[626:627]*/
	v_mov_b64_e32 v[148:149], v[116:117] /*v[628:629]*/
	v_mov_b64_e32 v[150:151], v[118:119] /*v[630:631]*/
	v_mov_b64_e32 v[152:153], v[120:121] /*v[632:633]*/
	v_mov_b64_e32 v[162:163], v[106:107] /*v[618:619]*/
	v_mov_b64_e32 v[164:165], v[108:109] /*v[620:621]*/
	v_mov_b64_e32 v[166:167], v[110:111] /*v[622:623]*/
	v_mov_b64_e32 v[168:169], v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x242
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[90:91] /*v[602:603]*/
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[92:93] /*v[604:605]*/
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[94:95] /*v[606:607]*/
	v_mov_b64_e32 v[40:41] /*v[296:297]*/, v[96:97] /*v[608:609]*/
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[74:75] /*v[586:587]*/
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[76:77] /*v[588:589]*/
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[78:79] /*v[590:591]*/
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[80:81] /*v[592:593]*/
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[58:59] /*v[570:571]*/
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[60:61] /*v[572:573]*/
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[62:63] /*v[574:575]*/
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[64:65] /*v[576:577]*/
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[42:43] /*v[554:555]*/
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[44:45] /*v[556:557]*/
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[46:47] /*v[558:559]*/
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[48:49] /*v[560:561]*/
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[26:27] /*v[538:539]*/
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[28:29] /*v[540:541]*/
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[30:31] /*v[542:543]*/
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[32:33] /*v[544:545]*/
	s_set_vgpr_msb 0x4202
	v_mov_b64_e32 v[10:11], v[234:235] /*v[746:747]*/
	v_mov_b64_e32 v[12:13], v[236:237] /*v[748:749]*/
	v_mov_b64_e32 v[14:15], v[238:239] /*v[750:751]*/
	v_mov_b64_e32 v[16:17], v[240:241] /*v[752:753]*/
	v_mov_b64_e32 v[26:27], v[218:219] /*v[730:731]*/
	v_mov_b64_e32 v[28:29], v[220:221] /*v[732:733]*/
	v_mov_b64_e32 v[30:31], v[222:223] /*v[734:735]*/
	v_mov_b64_e32 v[32:33], v[224:225] /*v[736:737]*/
	v_mov_b64_e32 v[42:43], v[202:203] /*v[714:715]*/
	v_mov_b64_e32 v[44:45], v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[46:47], v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[48:49], v[208:209] /*v[720:721]*/
	v_mov_b64_e32 v[58:59], v[178:179] /*v[690:691]*/
	v_mov_b64_e32 v[60:61], v[180:181] /*v[692:693]*/
	v_mov_b64_e32 v[62:63], v[182:183] /*v[694:695]*/
	v_mov_b64_e32 v[64:65], v[184:185] /*v[696:697]*/
	v_mov_b64_e32 v[74:75], v[154:155] /*v[666:667]*/
	v_mov_b64_e32 v[76:77], v[156:157] /*v[668:669]*/
	v_mov_b64_e32 v[78:79], v[158:159] /*v[670:671]*/
	v_mov_b64_e32 v[80:81], v[160:161] /*v[672:673]*/
	v_mov_b64_e32 v[90:91], v[146:147] /*v[658:659]*/
	v_mov_b64_e32 v[92:93], v[148:149] /*v[660:661]*/
	v_mov_b64_e32 v[94:95], v[150:151] /*v[662:663]*/
	v_mov_b64_e32 v[96:97], v[152:153] /*v[664:665]*/
	v_mov_b64_e32 v[106:107], v[138:139] /*v[650:651]*/
	v_mov_b64_e32 v[108:109], v[140:141] /*v[652:653]*/
	v_mov_b64_e32 v[110:111], v[142:143] /*v[654:655]*/
	v_mov_b64_e32 v[112:113], v[144:145] /*v[656:657]*/
	v_mov_b64_e32 v[122:123], v[130:131] /*v[642:643]*/
	v_mov_b64_e32 v[124:125], v[132:133] /*v[644:645]*/
	v_mov_b64_e32 v[126:127], v[134:135] /*v[646:647]*/
	v_mov_b64_e32 v[128:129], v[136:137] /*v[648:649]*/
	v_mov_b64_e32 v[138:139], v[98:99] /*v[610:611]*/
	v_mov_b64_e32 v[140:141], v[100:101] /*v[612:613]*/
	v_mov_b64_e32 v[142:143], v[102:103] /*v[614:615]*/
	v_mov_b64_e32 v[144:145], v[104:105] /*v[616:617]*/
	v_mov_b64_e32 v[154:155], v[82:83] /*v[594:595]*/
	v_mov_b64_e32 v[156:157], v[84:85] /*v[596:597]*/
	v_mov_b64_e32 v[158:159], v[86:87] /*v[598:599]*/
	v_mov_b64_e32 v[160:161], v[88:89] /*v[600:601]*/
	v_mov_b64_e32 v[186:187], v[66:67] /*v[578:579]*/
	v_mov_b64_e32 v[188:189], v[68:69] /*v[580:581]*/
	v_mov_b64_e32 v[190:191], v[70:71] /*v[582:583]*/
	v_mov_b64_e32 v[192:193], v[72:73] /*v[584:585]*/
	s_set_vgpr_msb 0x242
	v_mov_b64_e32 v[58:59] /*v[314:315]*/, v[50:51] /*v[562:563]*/
	v_mov_b64_e32 v[60:61] /*v[316:317]*/, v[52:53] /*v[564:565]*/
	v_mov_b64_e32 v[62:63] /*v[318:319]*/, v[54:55] /*v[566:567]*/
	v_mov_b64_e32 v[64:65] /*v[320:321]*/, v[56:57] /*v[568:569]*/
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[34:35] /*v[546:547]*/
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[36:37] /*v[548:549]*/
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[38:39] /*v[550:551]*/
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[40:41] /*v[552:553]*/
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, v[18:19] /*v[530:531]*/
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, v[20:21] /*v[532:533]*/
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, v[22:23] /*v[534:535]*/
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, v[24:25] /*v[536:537]*/
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[10:11] /*v[522:523]*/
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[12:13] /*v[524:525]*/
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[14:15] /*v[526:527]*/
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[16:17] /*v[528:529]*/
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[2:3] /*v[514:515]*/
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[4:5] /*v[516:517]*/
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[8:9] /*v[520:521]*/
	s_set_vgpr_msb 0x4243
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v134 /*v390*/, v126 /*v894*/ :: v_dual_mov_b32 v135 /*v391*/, v127 /*v895*/
	v_dual_mov_b32 v136 /*v392*/, v128 /*v896*/ :: v_dual_mov_b32 v137 /*v393*/, v129 /*v897*/
	v_dual_mov_b32 v130 /*v386*/, v122 /*v890*/ :: v_dual_mov_b32 v131 /*v387*/, v123 /*v891*/
	v_dual_mov_b32 v132 /*v388*/, v124 /*v892*/ :: v_dual_mov_b32 v133 /*v389*/, v125 /*v893*/
	v_dual_mov_b32 v142 /*v398*/, v118 /*v886*/ :: v_dual_mov_b32 v143 /*v399*/, v119 /*v887*/
	v_dual_mov_b32 v144 /*v400*/, v120 /*v888*/ :: v_dual_mov_b32 v145 /*v401*/, v121 /*v889*/
	v_dual_mov_b32 v138 /*v394*/, v114 /*v882*/ :: v_dual_mov_b32 v139 /*v395*/, v115 /*v883*/
	v_dual_mov_b32 v140 /*v396*/, v116 /*v884*/ :: v_dual_mov_b32 v141 /*v397*/, v117 /*v885*/
	v_dual_mov_b32 v150 /*v406*/, v110 /*v878*/ :: v_dual_mov_b32 v151 /*v407*/, v111 /*v879*/
	v_dual_mov_b32 v152 /*v408*/, v112 /*v880*/ :: v_dual_mov_b32 v153 /*v409*/, v113 /*v881*/
	v_dual_mov_b32 v146 /*v402*/, v106 /*v874*/ :: v_dual_mov_b32 v147 /*v403*/, v107 /*v875*/
	v_dual_mov_b32 v148 /*v404*/, v108 /*v876*/ :: v_dual_mov_b32 v149 /*v405*/, v109 /*v877*/
	v_dual_mov_b32 v166 /*v422*/, v98 /*v866*/ :: v_dual_mov_b32 v167 /*v423*/, v99 /*v867*/
	v_dual_mov_b32 v168 /*v424*/, v100 /*v868*/ :: v_dual_mov_b32 v169 /*v425*/, v101 /*v869*/
	v_dual_mov_b32 v162 /*v418*/, v90 /*v858*/ :: v_dual_mov_b32 v163 /*v419*/, v91 /*v859*/
	v_dual_mov_b32 v164 /*v420*/, v92 /*v860*/ :: v_dual_mov_b32 v165 /*v421*/, v93 /*v861*/
	v_dual_mov_b32 v158 /*v414*/, v102 /*v870*/ :: v_dual_mov_b32 v159 /*v415*/, v103 /*v871*/
	v_dual_mov_b32 v160 /*v416*/, v104 /*v872*/ :: v_dual_mov_b32 v161 /*v417*/, v105 /*v873*/
	v_dual_mov_b32 v154 /*v410*/, v94 /*v862*/ :: v_dual_mov_b32 v155 /*v411*/, v95 /*v863*/
	v_dual_mov_b32 v156 /*v412*/, v96 /*v864*/ :: v_dual_mov_b32 v157 /*v413*/, v97 /*v865*/
	v_dual_mov_b32 v174 /*v430*/, v86 /*v854*/ :: v_dual_mov_b32 v175 /*v431*/, v87 /*v855*/
	v_dual_mov_b32 v176 /*v432*/, v88 /*v856*/ :: v_dual_mov_b32 v177 /*v433*/, v89 /*v857*/
	v_dual_mov_b32 v170 /*v426*/, v82 /*v850*/ :: v_dual_mov_b32 v171 /*v427*/, v83 /*v851*/
	v_dual_mov_b32 v172 /*v428*/, v84 /*v852*/ :: v_dual_mov_b32 v173 /*v429*/, v85 /*v853*/
	v_dual_mov_b32 v182 /*v438*/, v78 /*v846*/ :: v_dual_mov_b32 v183 /*v439*/, v79 /*v847*/
	v_dual_mov_b32 v184 /*v440*/, v80 /*v848*/ :: v_dual_mov_b32 v185 /*v441*/, v81 /*v849*/
	v_dual_mov_b32 v178 /*v434*/, v74 /*v842*/ :: v_dual_mov_b32 v179 /*v435*/, v75 /*v843*/
	v_dual_mov_b32 v180 /*v436*/, v76 /*v844*/ :: v_dual_mov_b32 v181 /*v437*/, v77 /*v845*/
	v_dual_mov_b32 v190 /*v446*/, v70 /*v838*/ :: v_dual_mov_b32 v191 /*v447*/, v71 /*v839*/
	v_dual_mov_b32 v192 /*v448*/, v72 /*v840*/ :: v_dual_mov_b32 v193 /*v449*/, v73 /*v841*/
	v_dual_mov_b32 v186 /*v442*/, v66 /*v834*/ :: v_dual_mov_b32 v187 /*v443*/, v67 /*v835*/
	v_dual_mov_b32 v188 /*v444*/, v68 /*v836*/ :: v_dual_mov_b32 v189 /*v445*/, v69 /*v837*/
	v_dual_mov_b32 v198 /*v454*/, v62 /*v830*/ :: v_dual_mov_b32 v199 /*v455*/, v63 /*v831*/
	v_dual_mov_b32 v200 /*v456*/, v64 /*v832*/ :: v_dual_mov_b32 v201 /*v457*/, v65 /*v833*/
	v_dual_mov_b32 v194 /*v450*/, v58 /*v826*/ :: v_dual_mov_b32 v195 /*v451*/, v59 /*v827*/
	v_dual_mov_b32 v196 /*v452*/, v60 /*v828*/ :: v_dual_mov_b32 v197 /*v453*/, v61 /*v829*/
	v_dual_mov_b32 v206 /*v462*/, v54 /*v822*/ :: v_dual_mov_b32 v207 /*v463*/, v55 /*v823*/
	v_dual_mov_b32 v208 /*v464*/, v56 /*v824*/ :: v_dual_mov_b32 v209 /*v465*/, v57 /*v825*/
	v_dual_mov_b32 v202 /*v458*/, v50 /*v818*/ :: v_dual_mov_b32 v203 /*v459*/, v51 /*v819*/
	v_dual_mov_b32 v204 /*v460*/, v52 /*v820*/ :: v_dual_mov_b32 v205 /*v461*/, v53 /*v821*/
	v_dual_mov_b32 v214 /*v470*/, v46 /*v814*/ :: v_dual_mov_b32 v215 /*v471*/, v47 /*v815*/
	v_dual_mov_b32 v216 /*v472*/, v48 /*v816*/ :: v_dual_mov_b32 v217 /*v473*/, v49 /*v817*/
	v_dual_mov_b32 v210 /*v466*/, v42 /*v810*/ :: v_dual_mov_b32 v211 /*v467*/, v43 /*v811*/
	v_dual_mov_b32 v212 /*v468*/, v44 /*v812*/ :: v_dual_mov_b32 v213 /*v469*/, v45 /*v813*/
	v_dual_mov_b32 v222 /*v478*/, v38 /*v806*/ :: v_dual_mov_b32 v223 /*v479*/, v39 /*v807*/
	v_dual_mov_b32 v224 /*v480*/, v40 /*v808*/ :: v_dual_mov_b32 v225 /*v481*/, v41 /*v809*/
	v_dual_mov_b32 v218 /*v474*/, v30 /*v798*/ :: v_dual_mov_b32 v219 /*v475*/, v31 /*v799*/
	v_dual_mov_b32 v220 /*v476*/, v32 /*v800*/ :: v_dual_mov_b32 v221 /*v477*/, v33 /*v801*/
	v_dual_mov_b32 v230 /*v486*/, v34 /*v802*/ :: v_dual_mov_b32 v231 /*v487*/, v35 /*v803*/
	v_dual_mov_b32 v232 /*v488*/, v36 /*v804*/ :: v_dual_mov_b32 v233 /*v489*/, v37 /*v805*/
	v_dual_mov_b32 v226 /*v482*/, v26 /*v794*/ :: v_dual_mov_b32 v227 /*v483*/, v27 /*v795*/
	v_dual_mov_b32 v228 /*v484*/, v28 /*v796*/ :: v_dual_mov_b32 v229 /*v485*/, v29 /*v797*/
	v_dual_mov_b32 v238 /*v494*/, v22 /*v790*/ :: v_dual_mov_b32 v239 /*v495*/, v23 /*v791*/
	v_dual_mov_b32 v240 /*v496*/, v24 /*v792*/ :: v_dual_mov_b32 v241 /*v497*/, v25 /*v793*/
	v_dual_mov_b32 v234 /*v490*/, v18 /*v786*/ :: v_dual_mov_b32 v235 /*v491*/, v19 /*v787*/
	v_dual_mov_b32 v236 /*v492*/, v20 /*v788*/ :: v_dual_mov_b32 v237 /*v493*/, v21 /*v789*/
	v_dual_mov_b32 v246 /*v502*/, v14 /*v782*/ :: v_dual_mov_b32 v247 /*v503*/, v15 /*v783*/
	v_dual_mov_b32 v248 /*v504*/, v16 /*v784*/ :: v_dual_mov_b32 v249 /*v505*/, v17 /*v785*/
	v_dual_mov_b32 v242 /*v498*/, v10 /*v778*/ :: v_dual_mov_b32 v243 /*v499*/, v11 /*v779*/
	v_dual_mov_b32 v244 /*v500*/, v12 /*v780*/ :: v_dual_mov_b32 v245 /*v501*/, v13 /*v781*/
	v_dual_mov_b32 v254 /*v510*/, v6 /*v774*/ :: v_dual_mov_b32 v255 /*v511*/, v7 /*v775*/
	s_set_vgpr_msb 0x4383
	v_dual_mov_b32 v0 /*v512*/, v8 /*v776*/ :: v_dual_mov_b32 v1 /*v513*/, v9 /*v777*/
	s_set_vgpr_msb 0x8343
	v_dual_mov_b32 v250 /*v506*/, v2 /*v770*/ :: v_dual_mov_b32 v251 /*v507*/, v3 /*v771*/
	v_dual_mov_b32 v252 /*v508*/, v4 /*v772*/ :: v_dual_mov_b32 v253 /*v509*/, v5 /*v773*/
	s_cmp_eq_u64 s[48:49], s[24:25]
	s_mov_b32 s2, -1
	s_set_vgpr_msb 0x4300
	s_cbranch_scc1 .LBB0_1
	s_abs_i32 s3, s48
	s_ashr_i32 s2, s48, 31
	s_mul_hi_u32 s4, s3, s57
	s_xor_b32 s2, s2, s56
	s_mul_i32 s5, s4, s55
	s_add_co_i32 s6, s4, 1
	s_sub_co_i32 s3, s3, s5
	s_mov_b32 s46, s42
	s_sub_co_i32 s5, s3, s55
	s_cmp_ge_u32 s3, s55
	s_mov_b32 s47, s43
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s3, s5, s3
	s_add_co_i32 s5, s4, 1
	s_cmp_ge_u32 s3, s55
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[170:177], v[250:257] /*v[506:513]*/, 0
	s_cselect_b32 s3, s5, s4
	v_mov_b64_e32 v[74:75] /*v[586:587]*/, s[18:19]
	s_xor_b32 s3, s3, s2
	s_set_vgpr_msb 0x8407
	ds_store_b128 v145 /*v913*/, v[218:221] /*v[474:477]*/
	ds_store_b128 v145 /*v913*/, v[222:225] /*v[478:481]*/ offset:32
	ds_store_b128 v145 /*v913*/, v[250:253] /*v[506:509]*/ offset:8704
	ds_store_b128 v145 /*v913*/, v[254:257] /*v[510:513]*/ offset:8736
	s_sub_co_i32 s4, s3, s2
	ds_store_b128 v145 /*v913*/, v[210:213] /*v[466:469]*/ offset:64
	ds_store_b128 v145 /*v913*/, v[214:217] /*v[470:473]*/ offset:96
	ds_store_b128 v145 /*v913*/, v[242:245] /*v[498:501]*/ offset:8768
	ds_store_b128 v145 /*v913*/, v[246:249] /*v[502:505]*/ offset:8800
	ds_store_b128 v145 /*v913*/, v[202:205] /*v[458:461]*/ offset:128
	ds_store_b128 v145 /*v913*/, v[206:209] /*v[462:465]*/ offset:160
	ds_store_b128 v145 /*v913*/, v[234:237] /*v[490:493]*/ offset:8832
	ds_store_b128 v145 /*v913*/, v[238:241] /*v[494:497]*/ offset:8864
	s_mul_i32 s4, s4, s21
	s_set_vgpr_msb 0x7a4
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[178:185], v[242:249] /*v[498:505]*/, v[10:17] /*v[522:529]*/
	s_cmp_lg_u32 s48, s4
	s_set_vgpr_msb 0xa407
	ds_store_b128 v145 /*v913*/, v[194:197] /*v[450:453]*/ offset:192
	ds_store_b128 v145 /*v913*/, v[198:201] /*v[454:457]*/ offset:224
	s_cselect_b32 s4, -1, 0
	s_xor_b32 s5, s21, s48
	ds_store_b128 v145 /*v913*/, v[226:229] /*v[482:485]*/ offset:8896
	ds_store_b128 v145 /*v913*/, v[230:233] /*v[486:489]*/ offset:8928
	s_cmp_lt_i32 s5, 0
	s_set_vgpr_msb 0x78f
	v_dual_add_nc_u32 v76 /*v588*/, v148 /*v916*/, v147 /*v915*/ :: v_dual_add_nc_u32 v77 /*v589*/, v150 /*v918*/, v147 /*v915*/
	s_cselect_b32 s5, -1, 0
	s_set_vgpr_msb 0x8fa4
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[210:217], v[234:241] /*v[490:497]*/, v[10:17] /*v[522:529]*/
	s_and_b32 s4, s5, s4
	s_sub_co_ci_u32 s2, s3, s2
	s_add_co_i32 s30, s48, 1
	s_mul_i32 s3, s21, s2
	s_abs_i32 s5, s30
	s_ashr_i32 s4, s30, 31
	s_mul_hi_u32 s6, s5, s57
	s_xor_b32 s4, s4, s56
	s_mul_i32 s7, s6, s55
	s_add_co_i32 s8, s6, 1
	s_sub_co_i32 s5, s5, s7
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[226:233], v[226:233] /*v[482:489]*/, v[10:17] /*v[522:529]*/
	s_sub_co_i32 s7, s5, s55
	s_cmp_ge_u32 s5, s55
	s_set_vgpr_msb 0xa48f
	v_add_nc_u32_e32 v82 /*v594*/, v149 /*v917*/, v156 /*v924*/
	s_cselect_b32 s6, s8, s6
	s_cselect_b32 s5, s7, s5
	s_add_co_i32 s7, s6, 1
	s_cmp_ge_u32 s5, s55
	s_set_vgpr_msb 0x8f84
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[170:177], v[186:193] /*v[442:449]*/, 0
	s_cselect_b32 s5, s7, s6
	s_set_vgpr_msb 0x848f
	v_add_nc_u32_e32 v98 /*v610*/, v149 /*v917*/, v157 /*v925*/
	s_xor_b32 s5, s5, s4
	v_add_nc_u32_e32 v162 /*v674*/, v149 /*v917*/, v159 /*v927*/
	s_sub_co_i32 s6, s5, s4
	s_set_vgpr_msb 0x8f8a
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[74:75] /*v[586:587]*/, v[10:11] /*v[522:523]*/
	s_mul_i32 s6, s6, s21
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[74:75] /*v[586:587]*/, v[12:13] /*v[524:525]*/
	s_cmp_lg_u32 s30, s6
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[74:75] /*v[586:587]*/, v[14:15] /*v[526:527]*/
	s_cselect_b32 s6, -1, 0
	s_xor_b32 s7, s30, s21
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[74:75] /*v[586:587]*/, v[16:17] /*v[528:529]*/
	s_cmp_lt_i32 s7, 0
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[178:185], v[178:185] /*v[434:441]*/, v[34:41] /*v[546:553]*/
	s_cselect_b32 s7, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s6, s7, s6
	s_sub_co_ci_u32 s31, s5, s4
	s_add_co_i32 s2, s2, s58
	s_add_co_i32 s4, s54, s48
	s_set_vgpr_msb 0xa4b5
	v_lshl_or_b32 v67 /*v579*/, s2, 5, v141 /*v909*/
	s_sub_co_i32 s2, s4, s3
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[2:9] /*v[258:265]*/, v[186:193] /*v[442:449]*/, 0
	s_mul_i32 s2, s2, s17
	s_add_co_i32 s30, s30, s52
	s_set_vgpr_msb 0xb58a
	v_or_b32_e32 v69 /*v581*/, 16, v67 /*v579*/
	v_add_lshl_u32 v18 /*v530*/, s2, v67 /*v579*/, 2
	buffer_load_b32 v66 /*v578*/, v18 /*v530*/, s[40:43], null offen
	buffer_load_b32 v68 /*v580*/, v18 /*v530*/, s[44:47], null offen
	v_add_lshl_u32 v71 /*v583*/, s2, v69 /*v581*/, 2
	s_set_vgpr_msb 0x8a85
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[2:9] /*v[258:265]*/, v[250:257] /*v[506:513]*/, 0
	s_set_vgpr_msb 0x858a
	v_add_nc_u32_e32 v67 /*v579*/, s23, v67 /*v579*/
	s_add_nc_u64 s[48:49], s[48:49], 1
	buffer_load_b32 v70 /*v582*/, v71 /*v583*/, s[40:43], null offen
	buffer_load_b32 v72 /*v584*/, v71 /*v583*/, s[44:47], null offen
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s2, v144 /*v912*/, v67 /*v579*/
	v_cmp_gt_i32_e64 s3, v139 /*v907*/, v67 /*v579*/
	s_set_vgpr_msb 0xba5
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[10:17] /*v[266:273]*/, v[242:249] /*v[498:505]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa50b
	v_cmp_gt_i32_e64 s4, v140 /*v908*/, v67 /*v579*/
	v_cmp_gt_i32_e64 s5, v137 /*v905*/, v67 /*v579*/
	s_and_b32 s2, s61, s2
	v_cmp_gt_i32_e64 s6, v138 /*v906*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v10 /*v522*/, v10 /*v522*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s3
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s7, v135 /*v903*/, v67 /*v579*/
	s_set_vgpr_msb 0xba4
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[242:249], v[234:241] /*v[490:497]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa482
	v_cndmask_b32_e64 v13 /*v525*/, v13 /*v525*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s4
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s8, v136 /*v904*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v12 /*v524*/, v12 /*v524*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s5
	s_set_vgpr_msb 0x820b
	v_cmp_ge_i32_e64 s9, v142 /*v910*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v15 /*v527*/, v15 /*v527*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x82a5
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[42:49] /*v[298:305]*/, v[226:233] /*v[482:489]*/, v[18:25] /*v[530:537]*/
	s_and_b32 s2, s61, s6
	s_set_vgpr_msb 0xa50b
	v_cmp_gt_i32_e64 s10, v142 /*v910*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v14 /*v526*/, v14 /*v526*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s7
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s11, v133 /*v901*/, v67 /*v579*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v17 /*v529*/, v17 /*v529*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s8
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[74:75] /*v[586:587]*/, v[18:19] /*v[530:531]*/
	v_cndmask_b32_e64 v16 /*v528*/, v16 /*v528*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s9
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[74:75] /*v[586:587]*/, v[20:21] /*v[532:533]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s12, v134 /*v902*/, v67 /*v579*/
	s_set_vgpr_msb 0xba4
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[210:217], v[170:177] /*v[426:433]*/, v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa482
	v_cndmask_b32_e64 v19 /*v531*/, v19 /*v531*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s10
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s13, v131 /*v899*/, v67 /*v579*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v18 /*v530*/, v18 /*v530*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s11
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[74:75] /*v[586:587]*/, v[22:23] /*v[534:535]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s14, v132 /*v900*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v21 /*v533*/, v21 /*v533*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s12
	v_cmp_lt_i32_e64 s15, v67 /*v579*/, v1
	s_set_vgpr_msb 0x82a5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[10:17] /*v[266:273]*/, v[178:185] /*v[434:441]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa58a
	v_cndmask_b32_e64 v20 /*v532*/, v20 /*v532*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s13
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[74:75] /*v[586:587]*/, v[24:25] /*v[536:537]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s16, v130 /*v898*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v23 /*v535*/, v23 /*v535*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s14
	s_set_vgpr_msb 0x820b
	v_cmp_ge_i32_e32 vcc_lo, v144 /*v912*/, v67 /*v579*/
	s_set_vgpr_msb 0xba4
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[226:233], v[154:161] /*v[410:417]*/, v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa48a
	v_add_nc_u32_e32 v67 /*v579*/, s23, v69 /*v581*/
	v_cndmask_b32_e64 v22 /*v534*/, v22 /*v534*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s15
	s_and_b32 s46, s61, vcc_lo
	v_cndmask_b32_e64 v25 /*v537*/, v25 /*v537*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s16
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s3, v139 /*v907*/, v67 /*v579*/
	s_set_vgpr_msb 0xba4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[242:249], v[170:177] /*v[426:433]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa482
	v_cndmask_b32_e64 v24 /*v536*/, v24 /*v536*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s2, v144 /*v912*/, v67 /*v579*/
	s_set_vgpr_msb 0xb8a
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[74:75] /*v[586:587]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[74:75] /*v[586:587]*/, v[36:37] /*v[548:549]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s4, v140 /*v908*/, v67 /*v579*/
	v_cmp_gt_i32_e64 s5, v137 /*v905*/, v67 /*v579*/
	s_and_b32 s2, s61, s2
	s_set_vgpr_msb 0xba5
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[42:49] /*v[298:305]*/, v[154:161] /*v[410:417]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa58a
	v_cndmask_b32_e64 v34 /*v546*/, v34 /*v546*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s3
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[74:75] /*v[586:587]*/, v[38:39] /*v[550:551]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s6, v138 /*v906*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v37 /*v549*/, v37 /*v549*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s4
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s7, v135 /*v903*/, v67 /*v579*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v36 /*v548*/, v36 /*v548*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s5
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[74:75] /*v[586:587]*/, v[40:41] /*v[552:553]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s8, v136 /*v904*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v39 /*v551*/, v39 /*v551*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s6
	s_set_vgpr_msb 0x820b
	v_cmp_ge_i32_e64 s9, v142 /*v910*/, v67 /*v579*/
	s_set_vgpr_msb 0xb84
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[194:201], v[218:225] /*v[474:481]*/, 0
	s_set_vgpr_msb 0x848a
	v_cndmask_b32_e64 v38 /*v550*/, v38 /*v550*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s7
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[74:75] /*v[586:587]*/, v[50:51] /*v[562:563]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s10, v142 /*v910*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v41 /*v553*/, v41 /*v553*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s8
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s11, v133 /*v901*/, v67 /*v579*/
	s_set_vgpr_msb 0xb85
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[18:25] /*v[274:281]*/, v[218:225] /*v[474:481]*/, 0
	s_set_vgpr_msb 0x858a
	v_cndmask_b32_e64 v40 /*v552*/, v40 /*v552*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s9
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[74:75] /*v[586:587]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s12, v134 /*v902*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v51 /*v563*/, v51 /*v563*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s10
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s13, v131 /*v899*/, v67 /*v579*/
	s_set_vgpr_msb 0xb84
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[194:201], v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x848a
	v_cndmask_b32_e64 v50 /*v562*/, v50 /*v562*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s11
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[74:75] /*v[586:587]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s14, v132 /*v900*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v53 /*v565*/, v53 /*v565*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s12
	v_cmp_lt_i32_e64 s15, v67 /*v579*/, v1
	s_set_vgpr_msb 0x8285
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[18:25] /*v[274:281]*/, v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x850b
	v_cmp_ge_i32_e32 vcc_lo, v144 /*v912*/, v67 /*v579*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v52 /*v564*/, v52 /*v564*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s13
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[74:75] /*v[586:587]*/, v[56:57] /*v[568:569]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s16, v130 /*v898*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v11 /*v523*/, v11 /*v523*/, 0xff61b1e6, s46
	v_cndmask_b32_e64 v55 /*v567*/, v55 /*v567*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[202:209], v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/
	s_and_b32 s2, s61, s14
	s_and_b32 s46, s61, vcc_lo
	s_set_vgpr_msb 0xa482
	v_cndmask_b32_e64 v54 /*v566*/, v54 /*v566*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s15
	v_cndmask_b32_e64 v35 /*v547*/, v35 /*v547*/, 0xff61b1e6, s46
	v_cndmask_b32_e64 v57 /*v569*/, v57 /*v569*/, 0xff61b1e6, s2
	s_and_b32 s2, s61, s16
	s_set_vgpr_msb 0x82a5
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[26:33] /*v[282:289]*/, v[210:217] /*v[466:473]*/, v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa582
	v_cndmask_b32_e64 v56 /*v568*/, v56 /*v568*/, 0xff61b1e6, s2
	s_add_co_i32 s2, s31, s58
	s_mul_i32 s3, s21, s31
	s_set_vgpr_msb 0x82f0
	v_lshl_or_b32 v18 /*v786*/, s2, 5, v141 /*v909*/
	s_sub_co_i32 s2, s30, s3
	s_mov_b32 s30, s38
	s_lshl4_add_u32 s2, s2, s53
	s_set_vgpr_msb 0xf0a4
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[202:209], v[146:153] /*v[402:409]*/, v[42:49] /*v[554:561]*/
	s_mov_b32 s31, s39
	s_set_vgpr_msb 0xa48a
	s_wait_loadcnt 0x3
	v_pk_add_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[26:33] /*v[282:289]*/, v[146:153] /*v[402:409]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa58a
	v_pk_add_f32 v[12:13] /*v[524:525]*/, v[12:13] /*v[524:525]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[218:225], v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa48a
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[12:13] /*v[524:525]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, s[22:23] op_sel_hi:[1,0]
	s_wait_loadcnt 0x1
	v_pk_add_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[250:257], v[202:209] /*v[458:465]*/, v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa48a
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[218:225], v[138:145] /*v[394:401]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa48a
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v10 /*v522*/, v10 /*v522*/
	v_exp_f32_e32 v11 /*v523*/, v11 /*v523*/
	v_exp_f32_e32 v12 /*v524*/, v12 /*v524*/
	v_exp_f32_e32 v13 /*v525*/, v13 /*v525*/
	v_exp_f32_e32 v14 /*v526*/, v14 /*v526*/
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[250:257], v[138:145] /*v[394:401]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa482
	v_exp_f32_e32 v15 /*v527*/, v15 /*v527*/
	v_exp_f32_e32 v16 /*v528*/, v16 /*v528*/
	v_exp_f32_e32 v17 /*v529*/, v17 /*v529*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, s[22:23] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[234:241], v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa482
	v_exp_f32_e32 v18 /*v530*/, v18 /*v530*/
	v_exp_f32_e32 v19 /*v531*/, v19 /*v531*/
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
	v_exp_f32_e32 v22 /*v534*/, v22 /*v534*/
	v_exp_f32_e32 v23 /*v535*/, v23 /*v535*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	s_set_vgpr_msb 0x82a5
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[50:57] /*v[306:313]*/, v[194:201] /*v[450:457]*/, v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa58a
	v_exp_f32_e32 v25 /*v537*/, v25 /*v537*/
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, s[22:23] op_sel_hi:[1,0]
	v_pk_add_f32 v[2:3] /*v[514:515]*/, v[2:3] /*v[514:515]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5] /*v[516:517]*/, v[4:5] /*v[516:517]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[234:241], v[130:137] /*v[386:393]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa48a
	v_pk_add_f32 v[6:7] /*v[518:519]*/, v[6:7] /*v[518:519]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9] /*v[520:521]*/, v[8:9] /*v[520:521]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[28:29] /*v[540:541]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[30:31] /*v[542:543]*/, v[30:31] /*v[542:543]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v34 /*v546*/, v34 /*v546*/
	s_set_vgpr_msb 0x8aa5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[50:57] /*v[306:313]*/, v[130:137] /*v[386:393]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa58a
	v_exp_f32_e32 v35 /*v547*/, v35 /*v547*/
	v_exp_f32_e32 v36 /*v548*/, v36 /*v548*/
	v_exp_f32_e32 v37 /*v549*/, v37 /*v549*/
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	v_exp_f32_e32 v39 /*v551*/, v39 /*v551*/
	v_exp_f32_e32 v40 /*v552*/, v40 /*v552*/
	v_exp_f32_e32 v41 /*v553*/, v41 /*v553*/
	v_exp_f32_e32 v50 /*v562*/, v50 /*v562*/
	v_exp_f32_e32 v51 /*v563*/, v51 /*v563*/
	v_exp_f32_e32 v52 /*v564*/, v52 /*v564*/
	v_exp_f32_e32 v53 /*v565*/, v53 /*v565*/
	v_exp_f32_e32 v54 /*v566*/, v54 /*v566*/
	v_exp_f32_e32 v55 /*v567*/, v55 /*v567*/
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v57 /*v569*/, v57 /*v569*/
	s_wait_loadcnt 0x0
	v_pk_add_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[58:59] /*v[570:571]*/, v[58:59] /*v[570:571]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[60:61] /*v[572:573]*/, v[60:61] /*v[572:573]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[62:63] /*v[574:575]*/, v[62:63] /*v[574:575]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[64:65] /*v[576:577]*/, v[64:65] /*v[576:577]*/, v[72:73] /*v[584:585]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[66:67] /*v[578:579]*/, v[2:3] /*v[514:515]*/, v[10:11] /*v[522:523]*/
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[4:5] /*v[516:517]*/, v[12:13] /*v[524:525]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[6:7] /*v[518:519]*/, v[14:15] /*v[526:527]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[8:9] /*v[520:521]*/, v[16:17] /*v[528:529]*/
	v_cvt_pk_bf16_f32 v5 /*v517*/, v16 /*v528*/, v17 /*v529*/
	v_cvt_pk_bf16_f32 v4 /*v516*/, v14 /*v526*/, v15 /*v527*/
	v_cvt_pk_bf16_f32 v3 /*v515*/, v12 /*v524*/, v13 /*v525*/
	v_cvt_pk_bf16_f32 v2 /*v514*/, v10 /*v522*/, v11 /*v523*/
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[26:27] /*v[538:539]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[28:29] /*v[540:541]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[30:31] /*v[542:543]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[32:33] /*v[544:545]*/, v[24:25] /*v[536:537]*/
	v_cvt_pk_bf16_f32 v9 /*v521*/, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v8 /*v520*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v7 /*v519*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v6 /*v518*/, v18 /*v530*/, v19 /*v531*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[74:75] /*v[586:587]*/, v[66:67] /*v[578:579]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[74:75] /*v[586:587]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[74:75] /*v[586:587]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[74:75] /*v[586:587]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[42:43] /*v[554:555]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[44:45] /*v[556:557]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[46:47] /*v[558:559]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[48:49] /*v[560:561]*/, v[40:41] /*v[552:553]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[74:75] /*v[586:587]*/, v[10:11] /*v[522:523]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[74:75] /*v[586:587]*/, v[12:13] /*v[524:525]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[74:75] /*v[586:587]*/, v[14:15] /*v[526:527]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[74:75] /*v[586:587]*/, v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x8a0b
	ds_store_b128 v146 /*v914*/, v[2:5] /*v[514:517]*/
	ds_store_b128 v146 /*v914*/, v[6:9] /*v[518:521]*/ offset:32
	s_set_vgpr_msb 0xb8a
	v_cvt_pk_bf16_f32 v5 /*v517*/, v40 /*v552*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v4 /*v516*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v3 /*v515*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v2 /*v514*/, v34 /*v546*/, v35 /*v547*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[58:59] /*v[570:571]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[60:61] /*v[572:573]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[62:63] /*v[574:575]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[64:65] /*v[576:577]*/, v[56:57] /*v[568:569]*/
	v_cvt_pk_bf16_f32 v10 /*v522*/, v18 /*v530*/, v19 /*v531*/
	v_cvt_pk_bf16_f32 v11 /*v523*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v12 /*v524*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v13 /*v525*/, v24 /*v536*/, v25 /*v537*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[74:75] /*v[586:587]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[74:75] /*v[586:587]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[74:75] /*v[586:587]*/, v[46:47] /*v[558:559]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[74:75] /*v[586:587]*/, v[48:49] /*v[560:561]*/
	v_cvt_pk_bf16_f32 v14 /*v526*/, v26 /*v538*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v15 /*v527*/, v28 /*v540*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v16 /*v528*/, v30 /*v542*/, v31 /*v543*/
	v_cvt_pk_bf16_f32 v17 /*v529*/, v32 /*v544*/, v33 /*v545*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[74:75] /*v[586:587]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[74:75] /*v[586:587]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[74:75] /*v[586:587]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[74:75] /*v[586:587]*/, v[40:41] /*v[552:553]*/
	v_cvt_pk_bf16_f32 v9 /*v521*/, v56 /*v568*/, v57 /*v569*/
	v_cvt_pk_bf16_f32 v8 /*v520*/, v54 /*v566*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v7 /*v519*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v6 /*v518*/, v50 /*v562*/, v51 /*v563*/
	s_set_vgpr_msb 0x8a0b
	ds_store_b128 v146 /*v914*/, v[10:13] /*v[522:525]*/ offset:2560
	ds_store_b128 v146 /*v914*/, v[14:17] /*v[526:529]*/ offset:2592
	s_set_vgpr_msb 0xb07
	ds_store_b128 v145 /*v913*/, v[162:165] /*v[418:421]*/ offset:4352
	ds_store_b128 v145 /*v913*/, v[166:169] /*v[422:425]*/ offset:4384
	ds_store_b128 v145 /*v913*/, v[186:189] /*v[442:445]*/ offset:13056
	ds_store_b128 v145 /*v913*/, v[190:193] /*v[446:449]*/ offset:13088
	ds_store_b128 v145 /*v913*/, v[146:149] /*v[402:405]*/ offset:4416
	ds_store_b128 v145 /*v913*/, v[150:153] /*v[406:409]*/ offset:4448
	ds_store_b128 v145 /*v913*/, v[178:181] /*v[434:437]*/ offset:13120
	ds_store_b128 v145 /*v913*/, v[182:185] /*v[438:441]*/ offset:13152
	ds_store_b128 v145 /*v913*/, v[138:141] /*v[394:397]*/ offset:4480
	ds_store_b128 v145 /*v913*/, v[142:145] /*v[398:401]*/ offset:4512
	ds_store_b128 v145 /*v913*/, v[170:173] /*v[426:429]*/ offset:13184
	ds_store_b128 v145 /*v913*/, v[174:177] /*v[430:433]*/ offset:13216
	ds_store_b128 v145 /*v913*/, v[130:133] /*v[386:389]*/ offset:4544
	ds_store_b128 v145 /*v913*/, v[134:137] /*v[390:393]*/ offset:4576
	s_set_vgpr_msb 0x78a
	v_cvt_pk_bf16_f32 v10 /*v522*/, v18 /*v530*/, v19 /*v531*/
	v_cvt_pk_bf16_f32 v11 /*v523*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v12 /*v524*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v13 /*v525*/, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v14 /*v526*/, v26 /*v538*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v15 /*v527*/, v28 /*v540*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v16 /*v528*/, v30 /*v542*/, v31 /*v543*/
	v_cvt_pk_bf16_f32 v17 /*v529*/, v32 /*v544*/, v33 /*v545*/
	s_set_vgpr_msb 0x8a07
	ds_store_b128 v145 /*v913*/, v[154:157] /*v[410:413]*/ offset:13248
	ds_store_b128 v145 /*v913*/, v[158:161] /*v[414:417]*/ offset:13280
	s_set_vgpr_msb 0x70b
	ds_store_b128 v146 /*v914*/, v[2:5] /*v[514:517]*/ offset:1280
	ds_store_b128 v146 /*v914*/, v[6:9] /*v[518:521]*/ offset:1312
	ds_store_b128 v146 /*v914*/, v[10:13] /*v[522:525]*/ offset:3840
	ds_store_b128 v146 /*v914*/, v[14:17] /*v[526:529]*/ offset:3872
	s_set_vgpr_msb 0xb8f
	v_dual_add_nc_u32 v2 /*v514*/, v149 /*v917*/, v147 /*v915*/ :: v_dual_add_nc_u32 v10 /*v522*/, v151 /*v919*/, v147 /*v915*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[130:133] /*v[642:645]*/, v10 /*v522*/
	ds_load_tr16_b128 v[134:137] /*v[646:649]*/, v10 /*v522*/ offset:1280
	s_set_vgpr_msb 0x828f
	v_dual_add_nc_u32 v10 /*v522*/, v148 /*v916*/, v152 /*v920*/ :: v_dual_add_nc_u32 v18 /*v530*/, v149 /*v917*/, v152 /*v920*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[178:181] /*v[690:693]*/, v18 /*v530*/
	ds_load_tr16_b128 v[182:185] /*v[694:697]*/, v18 /*v530*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_dual_add_nc_u32 v18 /*v530*/, v148 /*v916*/, v153 /*v921*/ :: v_dual_add_nc_u32 v34 /*v546*/, v149 /*v917*/, v153 /*v921*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[194:197] /*v[706:709]*/, v34 /*v546*/
	ds_load_tr16_b128 v[198:201] /*v[710:713]*/, v34 /*v546*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_dual_add_nc_u32 v34 /*v546*/, v148 /*v916*/, v154 /*v922*/ :: v_dual_add_nc_u32 v50 /*v562*/, v149 /*v917*/, v154 /*v922*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[210:213] /*v[722:725]*/, v50 /*v562*/
	ds_load_tr16_b128 v[214:217] /*v[726:729]*/, v50 /*v562*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_dual_add_nc_u32 v50 /*v562*/, v148 /*v916*/, v155 /*v923*/ :: v_dual_add_nc_u32 v66 /*v578*/, v149 /*v917*/, v155 /*v923*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[226:229] /*v[738:741]*/, v66 /*v578*/
	ds_load_tr16_b128 v[230:233] /*v[742:745]*/, v66 /*v578*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_add_nc_u32_e32 v66 /*v578*/, v148 /*v916*/, v156 /*v924*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[242:245] /*v[754:757]*/, v82 /*v594*/
	ds_load_tr16_b128 v[246:249] /*v[758:761]*/, v82 /*v594*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_add_nc_u32_e32 v82 /*v594*/, v148 /*v916*/, v157 /*v925*/
	s_set_vgpr_msb 0x8fc2
	ds_load_tr16_b128 v[90:93] /*v[858:861]*/, v98 /*v610*/
	ds_load_tr16_b128 v[94:97] /*v[862:865]*/, v98 /*v610*/ offset:4352
	s_set_vgpr_msb 0xc28f
	v_add_nc_u32_e32 v98 /*v610*/, v148 /*v916*/, v159 /*v927*/
	s_set_vgpr_msb 0x8fc2
	ds_load_tr16_b128 v[160:163] /*v[928:931]*/, v162 /*v674*/
	ds_load_tr16_b128 v[164:167] /*v[932:935]*/, v162 /*v674*/ offset:4352
	s_set_vgpr_msb 0xc28f
	v_add_nc_u32_e32 v162 /*v674*/, v150 /*v918*/, v152 /*v920*/
	s_set_vgpr_msb 0x8fc2
	ds_load_tr16_b128 v[168:171] /*v[936:939]*/, v162 /*v674*/
	ds_load_tr16_b128 v[172:175] /*v[940:943]*/, v162 /*v674*/ offset:1280
	s_set_vgpr_msb 0xc28f
	v_add_nc_u32_e32 v162 /*v674*/, v151 /*v919*/, v152 /*v920*/
	s_set_vgpr_msb 0x8fc2
	ds_load_tr16_b128 v[176:179] /*v[944:947]*/, v162 /*v674*/
	ds_load_tr16_b128 v[180:183] /*v[948:951]*/, v162 /*v674*/ offset:1280
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[138:141] /*v[650:653]*/, v76 /*v588*/
	ds_load_tr16_b128 v[142:145] /*v[654:657]*/, v76 /*v588*/ offset:4352
	ds_load_tr16_b128 v[122:125] /*v[634:637]*/, v77 /*v589*/
	ds_load_tr16_b128 v[126:129] /*v[638:641]*/, v77 /*v589*/ offset:1280
	ds_load_tr16_b128 v[146:149] /*v[658:661]*/, v2 /*v514*/
	ds_load_tr16_b128 v[150:153] /*v[662:665]*/, v2 /*v514*/ offset:4352
	ds_load_tr16_b128 v[154:157] /*v[666:669]*/, v10 /*v522*/
	ds_load_tr16_b128 v[158:161] /*v[670:673]*/, v10 /*v522*/ offset:4352
	ds_load_tr16_b128 v[186:189] /*v[698:701]*/, v18 /*v530*/
	ds_load_tr16_b128 v[190:193] /*v[702:705]*/, v18 /*v530*/ offset:4352
	ds_load_tr16_b128 v[202:205] /*v[714:717]*/, v34 /*v546*/
	ds_load_tr16_b128 v[206:209] /*v[718:721]*/, v34 /*v546*/ offset:4352
	ds_load_tr16_b128 v[218:221] /*v[730:733]*/, v50 /*v562*/
	ds_load_tr16_b128 v[222:225] /*v[734:737]*/, v50 /*v562*/ offset:4352
	ds_load_tr16_b128 v[234:237] /*v[746:749]*/, v66 /*v578*/
	ds_load_tr16_b128 v[238:241] /*v[750:753]*/, v66 /*v578*/ offset:4352
	ds_load_tr16_b128 v[250:253] /*v[762:765]*/, v82 /*v594*/
	ds_load_tr16_b128 v[254:257] /*v[766:769]*/, v82 /*v594*/ offset:4352
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[122:125] /*v[890:893]*/, v98 /*v610*/
	ds_load_tr16_b128 v[126:129] /*v[894:897]*/, v98 /*v610*/ offset:4352
	s_set_vgpr_msb 0xc29a
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[130:137] /*v[642:649]*/, v[178:185] /*v[690:697]*/, v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x9a8b
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[176:183] /*v[944:951]*/, v[178:185] /*v[690:697]*/, v[98:105]
	v_nop
	v_nop
	v_nop
	v_nop
	v_mad_u32 v178 /*v690*/, v18 /*v786*/, s35, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v178 /*v690*/, v143 /*v911*/, v178 /*v690*/
	s_set_vgpr_msb 0x8b9a
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[122:129] /*v[634:641]*/, v[138:145] /*v[650:657]*/, v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x9acb
	v_lshlrev_b32_e32 v46 /*v814*/, 4, v178 /*v690*/
	s_clause 0x3
	buffer_load_b128 v[2:5] /*v[770:773]*/, v46 /*v814*/, s[36:39], null offen
	buffer_load_b128 v[6:9] /*v[774:777]*/, v46 /*v814*/, s[36:39], null offen offset:32
	buffer_load_b128 v[10:13] /*v[778:781]*/, v46 /*v814*/, s[36:39], null offen offset:64
	buffer_load_b128 v[14:17] /*v[782:785]*/, v46 /*v814*/, s[36:39], null offen offset:96
	s_set_vgpr_msb 0xcb9a
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[130:137] /*v[642:649]*/, v[146:153] /*v[658:665]*/, v[114:121] /*v[370:377]*/
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[122:129] /*v[634:641]*/, v[154:161] /*v[666:673]*/, v[106:113] /*v[362:369]*/
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[122:129] /*v[634:641]*/, v[186:193] /*v[698:705]*/, v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[130:137] /*v[642:649]*/, v[194:201] /*v[706:713]*/, v[82:89] /*v[338:345]*/
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[122:129] /*v[634:641]*/, v[202:209] /*v[714:721]*/, v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[130:137] /*v[642:649]*/, v[210:217] /*v[722:729]*/, v[66:73] /*v[322:329]*/
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[122:129] /*v[634:641]*/, v[218:225] /*v[730:737]*/, v[58:65] /*v[314:321]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[130:137] /*v[642:649]*/, v[226:233] /*v[738:745]*/, v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x9a8a
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[122:129] /*v[634:641]*/, v[234:241] /*v[746:753]*/, v[186:193]
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[130:137] /*v[642:649]*/, v[242:249] /*v[754:761]*/, v[162:169]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[122:129] /*v[634:641]*/, v[250:257] /*v[762:769]*/, v[154:161]
	s_set_vgpr_msb 0x8a8e
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[130:137] /*v[642:649]*/, v[90:97] /*v[858:865]*/, v[146:153]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[122:129] /*v[634:641]*/, v[122:129] /*v[890:897]*/, v[138:145]
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[130:137] /*v[642:649]*/, v[160:167] /*v[928:935]*/, v[130:137]
	s_set_vgpr_msb 0x8e8b
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[168:175] /*v[936:943]*/, v[138:145] /*v[650:657]*/, v[122:129]
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[650:657]*/, v[168:175] /*v[936:943]*/, v[154:161] /*v[666:673]*/, v[106:113]
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[666:673]*/, v[168:175] /*v[936:943]*/, v[202:209] /*v[714:721]*/, v[74:81]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x8b8c
	v_or_b32_e32 v202 /*v714*/, 16, v18 /*v786*/
	s_set_vgpr_msb 0x8c8b
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[168:175] /*v[936:943]*/, v[218:225] /*v[730:737]*/, v[58:65]
	v_nop
	v_nop
	v_nop
	v_nop
	v_mad_u32 v222 /*v734*/, s35, v202 /*v714*/, s2
	s_set_vgpr_msb 0x8b8c
	v_or_b32_e32 v218 /*v730*/, 0x80, v46 /*v814*/
	v_add_nc_u32_e32 v219 /*v731*/, 0xa0, v46 /*v814*/
	s_set_vgpr_msb 0x8c8b
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[714:721]*/, v[168:175] /*v[936:943]*/, v[234:241] /*v[746:753]*/, v[42:49]
	s_set_vgpr_msb 0x8b8c
	v_add_nc_u32_e32 v220 /*v732*/, 0xc0, v46 /*v814*/
	v_add_nc_u32_e32 v221 /*v733*/, 0xe0, v46 /*v814*/
	s_set_vgpr_msb 0x8cc2
	s_clause 0x3
	buffer_load_b128 v[18:21] /*v[786:789]*/, v218 /*v730*/, s[36:39], null offen
	buffer_load_b128 v[22:25] /*v[790:793]*/, v219 /*v731*/, s[36:39], null offen
	buffer_load_b128 v[26:29] /*v[794:797]*/, v220 /*v732*/, s[36:39], null offen
	buffer_load_b128 v[34:37] /*v[802:805]*/, v221 /*v733*/, s[36:39], null offen
	s_set_vgpr_msb 0xc28e
	v_or_b32_e32 v222 /*v734*/, v222 /*v734*/, v143 /*v911*/
	s_set_vgpr_msb 0x8ec3
	s_clause 0x3
	buffer_load_b128 v[30:33] /*v[798:801]*/, v46 /*v814*/, s[28:31], null offen
	buffer_load_b128 v[38:41] /*v[806:809]*/, v46 /*v814*/, s[28:31], null offen offset:32
	buffer_load_b128 v[42:45] /*v[810:813]*/, v46 /*v814*/, s[28:31], null offen offset:64
	buffer_load_b128 v[46:49] /*v[814:817]*/, v46 /*v814*/, s[28:31], null offen offset:96
	s_mov_b32 s2, 0
	s_set_vgpr_msb 0xc38b
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[674:681]*/, v[176:183] /*v[944:951]*/, v[146:153] /*v[658:665]*/, v[114:121]
	v_lshlrev_b32_e32 v234 /*v746*/, 4, v222 /*v734*/
	s_set_vgpr_msb 0x8bc2
	s_clause 0x3
	buffer_load_b128 v[50:53] /*v[818:821]*/, v218 /*v730*/, s[28:31], null offen
	buffer_load_b128 v[54:57] /*v[822:825]*/, v219 /*v731*/, s[28:31], null offen
	buffer_load_b128 v[58:61] /*v[826:829]*/, v220 /*v732*/, s[28:31], null offen
	buffer_load_b128 v[62:65] /*v[830:833]*/, v221 /*v733*/, s[28:31], null offen
	s_clause 0x3
	buffer_load_b128 v[66:69] /*v[834:837]*/, v234 /*v746*/, s[36:39], null offen
	buffer_load_b128 v[70:73] /*v[838:841]*/, v234 /*v746*/, s[36:39], null offen offset:32
	buffer_load_b128 v[74:77] /*v[842:845]*/, v234 /*v746*/, s[36:39], null offen offset:64
	buffer_load_b128 v[78:81] /*v[846:849]*/, v234 /*v746*/, s[36:39], null offen offset:96
	s_set_vgpr_msb 0xc28b
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[168:175] /*v[936:943]*/, v[250:257] /*v[762:769]*/, v[26:33]
	v_or_b32_e32 v235 /*v747*/, 0x80, v234 /*v746*/
	v_add_nc_u32_e32 v236 /*v748*/, 0xa0, v234 /*v746*/
	s_set_vgpr_msb 0x8bc2
	s_clause 0x1
	buffer_load_b128 v[82:85] /*v[850:853]*/, v235 /*v747*/, s[36:39], null offen
	buffer_load_b128 v[86:89] /*v[854:857]*/, v236 /*v748*/, s[36:39], null offen
	v_nop
	v_nop
	s_set_vgpr_msb 0xc28b
	v_add_nc_u32_e32 v250 /*v762*/, 0xc0, v234 /*v746*/
	v_add_nc_u32_e32 v251 /*v763*/, 0xe0, v234 /*v746*/
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[168:175] /*v[936:943]*/, v[186:193] /*v[698:705]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[698:705]*/, v[176:183] /*v[944:951]*/, v[194:201] /*v[706:713]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[706:713]*/, v[176:183] /*v[944:951]*/, v[210:217] /*v[722:729]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[722:729]*/, v[176:183] /*v[944:951]*/, v[226:233] /*v[738:745]*/, v[50:57]
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[176:183] /*v[944:951]*/, v[242:249] /*v[754:761]*/, v[34:41]
	s_set_vgpr_msb 0x8b8f
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[176:183] /*v[944:951]*/, v[90:97] /*v[858:865]*/, v[18:25]
	s_set_vgpr_msb 0x8fc2
	s_clause 0x1
	buffer_load_b128 v[94:97] /*v[862:865]*/, v250 /*v762*/, s[36:39], null offen
	buffer_load_b128 v[102:105] /*v[870:873]*/, v251 /*v763*/, s[36:39], null offen
	s_clause 0x5
	buffer_load_b128 v[90:93] /*v[858:861]*/, v234 /*v746*/, s[28:31], null offen
	buffer_load_b128 v[98:101] /*v[866:869]*/, v234 /*v746*/, s[28:31], null offen offset:32
	buffer_load_b128 v[106:109] /*v[874:877]*/, v234 /*v746*/, s[28:31], null offen offset:64
	buffer_load_b128 v[110:113] /*v[878:881]*/, v234 /*v746*/, s[28:31], null offen offset:96
	buffer_load_b128 v[114:117] /*v[882:885]*/, v235 /*v747*/, s[28:31], null offen
	buffer_load_b128 v[118:121] /*v[886:889]*/, v236 /*v748*/, s[28:31], null offen
	s_set_vgpr_msb 0xc28f
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[168:175] /*v[936:943]*/, v[122:129] /*v[890:897]*/, v[10:17]
	s_set_vgpr_msb 0x8fc2
	s_clause 0x1
	buffer_load_b128 v[122:125] /*v[890:893]*/, v250 /*v762*/, s[28:31], null offen
	buffer_load_b128 v[126:129] /*v[894:897]*/, v251 /*v763*/, s[28:31], null offen
	s_set_vgpr_msb 0xc28f
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[176:183] /*v[944:951]*/, v[160:167] /*v[928:935]*/, v[2:9]
	s_set_vgpr_msb 0x8f00
	s_branch .LBB0_1
.LBB0_4:
	s_sub_co_i32 s2, s59, s60
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s2, s2, s21
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB0_7
	s_set_vgpr_msb 0x8c
	v_or_b32_e32 v30 /*v542*/, 0xe0, v158 /*v926*/
	s_add_co_i32 s60, s60, s58
	s_ashr_i32 s3, s2, 31
	s_mov_b64 s[4:5], 0
	s_mov_b32 s30, s38
	s_mov_b32 s31, s39
	s_mov_b32 s46, s42
	s_mov_b32 s47, s43
	s_mov_b32 s6, 0x3fb8aa3b
	s_set_vgpr_msb 0x8c00
.LBB0_6:
	s_abs_i32 s7, s4
	s_ashr_i32 s8, s4, 31
	s_mul_hi_u32 s9, s7, s57
	s_xor_b32 s8, s8, s56
	s_mul_i32 s10, s9, s55
	s_add_co_i32 s11, s9, 1
	s_sub_co_i32 s7, s7, s10
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[170:177], v[250:257] /*v[506:513]*/, 0
	s_sub_co_i32 s10, s7, s55
	s_cmp_ge_u32 s7, s55
	v_nop
	v_nop
	v_mov_b64_e32 v[88:89] /*v[600:601]*/, s[18:19]
	s_cselect_b32 s9, s11, s9
	s_cselect_b32 s7, s10, s7
	s_add_co_i32 s10, s9, 1
	s_cmp_ge_u32 s7, s55
	s_set_vgpr_msb 0x8485
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[576:583]*/, v[2:9] /*v[258:265]*/, v[186:193] /*v[442:449]*/, 0
	s_cselect_b32 s7, s10, s9
	s_set_vgpr_msb 0x858f
	v_dual_add_nc_u32 v83 /*v595*/, v148 /*v916*/, v152 /*v920*/ :: v_dual_add_nc_u32 v85 /*v597*/, v149 /*v917*/, v152 /*v920*/
	s_xor_b32 s7, s7, s8
	v_dual_add_nc_u32 v87 /*v599*/, v148 /*v916*/, v153 /*v921*/ :: v_dual_add_nc_u32 v90 /*v602*/, v149 /*v917*/, v153 /*v921*/
	s_sub_co_i32 s9, s7, s8
	s_set_vgpr_msb 0x8f84
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[584:591]*/, v[170:177], v[186:193] /*v[442:449]*/, 0
	s_mul_i32 s9, s9, s21
	s_set_vgpr_msb 0x848f
	v_dual_add_nc_u32 v91 /*v603*/, v148 /*v916*/, v154 /*v922*/ :: v_dual_add_nc_u32 v92 /*v604*/, v149 /*v917*/, v154 /*v922*/
	s_cmp_lg_u32 s4, s9
	v_dual_add_nc_u32 v93 /*v605*/, v148 /*v916*/, v155 /*v923*/ :: v_dual_add_nc_u32 v94 /*v606*/, v149 /*v917*/, v155 /*v923*/
	s_cselect_b32 s9, -1, 0
	s_xor_b32 s10, s21, s4
	s_set_vgpr_msb 0x8fa4
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[178:185], v[242:249] /*v[498:505]*/, v[32:39] /*v[544:551]*/
	s_cmp_lt_i32 s10, 0
	s_set_vgpr_msb 0xa48f
	v_dual_add_nc_u32 v95 /*v607*/, v148 /*v916*/, v156 /*v924*/ :: v_dual_add_nc_u32 v196 /*v708*/, v149 /*v917*/, v156 /*v924*/
	s_cselect_b32 s10, -1, 0
	v_dual_add_nc_u32 v197 /*v709*/, v148 /*v916*/, v157 /*v925*/ :: v_dual_add_nc_u32 v198 /*v710*/, v149 /*v917*/, v157 /*v925*/
	s_and_b32 s9, s10, s9
	s_sub_co_ci_u32 s7, s7, s8
	s_add_co_i32 s8, s4, 1
	s_set_vgpr_msb 0x8fa4
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[584:591]*/, v[178:185], v[178:185] /*v[434:441]*/, v[72:79] /*v[584:591]*/
	s_abs_i32 s9, s8
	s_ashr_i32 s12, s8, 31
	s_mul_hi_u32 s10, s9, s57
	s_set_vgpr_msb 0xa48b
	v_dual_add_nc_u32 v199 /*v711*/, v148 /*v916*/, v30 /*v542*/ :: v_dual_add_nc_u32 v200 /*v712*/, v149 /*v917*/, v30 /*v542*/
	s_mul_i32 s11, s10, s55
	s_set_vgpr_msb 0x8b8f
	v_dual_add_nc_u32 v201 /*v713*/, v150 /*v918*/, v147 /*v915*/ :: v_dual_add_nc_u32 v202 /*v714*/, v151 /*v919*/, v147 /*v915*/
	s_sub_co_i32 s9, s9, s11
	s_xor_b32 s11, s12, s56
	s_add_co_i32 s12, s10, 1
	s_sub_co_i32 s13, s9, s55
	s_cmp_ge_u32 s9, s55
	s_set_vgpr_msb 0x8fa5
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[576:583]*/, v[10:17] /*v[266:273]*/, v[178:185] /*v[434:441]*/, v[64:71] /*v[576:583]*/
	s_cselect_b32 s10, s12, s10
	s_cselect_b32 s9, s13, s9
	s_add_co_i32 s12, s10, 1
	s_cmp_ge_u32 s9, s55
	s_set_vgpr_msb 0xa58f
	v_dual_add_nc_u32 v203 /*v715*/, v150 /*v918*/, v152 /*v920*/ :: v_dual_add_nc_u32 v204 /*v716*/, v151 /*v919*/, v152 /*v920*/
	s_cselect_b32 s9, s12, s10
	s_set_vgpr_msb 0x8fa4
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[210:217], v[234:241] /*v[490:497]*/, v[32:39] /*v[544:551]*/
	s_xor_b32 s9, s9, s11
	s_set_vgpr_msb 0xa48f
	v_add_nc_u32_e32 v81 /*v593*/, v149 /*v917*/, v147 /*v915*/
	s_sub_co_i32 s10, s9, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s10, s10, s21
	s_cmp_lg_u32 s8, s10
	s_set_vgpr_msb 0x8fb4
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[560:567]*/, v[194:201], v[218:225] /*v[474:481]*/, 0
	s_cselect_b32 s10, -1, 0
	s_xor_b32 s12, s8, s21
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s12, 0
	s_cselect_b32 s12, -1, 0
	s_and_b32 s10, s12, s10
	s_sub_co_ci_u32 s9, s9, s11
	s_mul_i32 s10, s21, s7
	s_add_co_i32 s11, s54, s4
	s_add_co_i32 s7, s7, s60
	s_sub_co_i32 s10, s11, s10
	v_lshl_or_b32 v2 /*v514*/, s7, 5, v141 /*v909*/
	s_mul_i32 s7, s10, s17
	s_add_co_i32 s8, s8, s52
	s_add_co_i32 s10, s7, 16
	s_set_vgpr_msb 0xb4a4
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[584:591]*/, v[210:217], v[170:177] /*v[426:433]*/, v[72:79] /*v[584:591]*/
	s_set_vgpr_msb 0xa48a
	v_add_lshl_u32 v3 /*v515*/, s7, v2 /*v514*/, 2
	v_add_lshl_u32 v2 /*v514*/, s10, v2 /*v514*/, 2
	s_clause 0x1
	buffer_load_b32 v80 /*v592*/, v3 /*v515*/, s[40:43], null offen
	buffer_load_b32 v82 /*v594*/, v2 /*v514*/, s[40:43], null offen
	s_clause 0x1
	buffer_load_b32 v84 /*v596*/, v3 /*v515*/, s[44:47], null offen
	buffer_load_b32 v86 /*v598*/, v2 /*v514*/, s[44:47], null offen
	s_add_co_i32 s7, s9, s60
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[576:583]*/, v[242:249], v[170:177] /*v[426:433]*/, v[64:71] /*v[576:583]*/
	s_set_vgpr_msb 0xa4b8
	v_lshl_or_b32 v2 /*v514*/, s7, 5, v141 /*v909*/
	s_mul_i32 s7, s21, s9
	s_add_nc_u64 s[4:5], s[4:5], 1
	s_sub_co_i32 s7, s8, s7
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v3 /*v515*/, 16, v2 /*v514*/
	s_lshl4_add_u32 s7, s7, s53
	s_set_vgpr_msb 0xb885
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[568:575]*/, v[18:25] /*v[274:281]*/, v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x858e
	v_mad_u32 v2 /*v514*/, v2 /*v514*/, s35, s7
	s_cmp_lg_u64 s[4:5], s[2:3]
	v_mad_u32 v3 /*v515*/, v3 /*v515*/, s35, s7
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v2 /*v514*/, v2 /*v514*/, v143 /*v911*/
	s_set_vgpr_msb 0x8ea4
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[544:551]*/, v[226:233], v[226:233] /*v[482:489]*/, v[32:39] /*v[544:551]*/
	s_set_vgpr_msb 0xa48e
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b32_e32 v3 /*v515*/, v3 /*v515*/, v143 /*v911*/
	s_set_vgpr_msb 0x8e8a
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_dual_lshlrev_b32 v2 /*v514*/, 4, v2 /*v514*/ :: v_dual_lshlrev_b32 v31 /*v543*/, 4, v3 /*v515*/
	v_nop
	v_nop
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[88:89] /*v[600:601]*/, v[32:33] /*v[544:545]*/
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v3 /*v515*/, 0x80, v2 /*v514*/
	v_add_nc_u32_e32 v4 /*v516*/, 0xa0, v2 /*v514*/
	v_add_nc_u32_e32 v5 /*v517*/, 0xc0, v2 /*v514*/
	v_add_nc_u32_e32 v6 /*v518*/, 0xe0, v2 /*v514*/
	v_or_b32_e32 v40 /*v552*/, 0x80, v31 /*v543*/
	v_add_nc_u32_e32 v41 /*v553*/, 0xa0, v31 /*v543*/
	v_add_nc_u32_e32 v42 /*v554*/, 0xc0, v31 /*v543*/
	v_add_nc_u32_e32 v43 /*v555*/, 0xe0, v31 /*v543*/
	s_clause 0x7
	buffer_load_b128 v[96:99] /*v[608:611]*/, v2 /*v514*/, s[36:39], null offen
	buffer_load_b128 v[100:103] /*v[612:615]*/, v2 /*v514*/, s[36:39], null offen offset:32
	buffer_load_b128 v[104:107] /*v[616:619]*/, v2 /*v514*/, s[36:39], null offen offset:64
	buffer_load_b128 v[108:111] /*v[620:623]*/, v2 /*v514*/, s[36:39], null offen offset:96
	buffer_load_b128 v[112:115] /*v[624:627]*/, v3 /*v515*/, s[36:39], null offen
	buffer_load_b128 v[116:119] /*v[628:631]*/, v4 /*v516*/, s[36:39], null offen
	buffer_load_b128 v[120:123] /*v[632:635]*/, v5 /*v517*/, s[36:39], null offen
	buffer_load_b128 v[124:127] /*v[636:639]*/, v6 /*v518*/, s[36:39], null offen
	s_clause 0x7
	buffer_load_b128 v[128:131] /*v[640:643]*/, v2 /*v514*/, s[28:31], null offen
	buffer_load_b128 v[132:135] /*v[644:647]*/, v2 /*v514*/, s[28:31], null offen offset:32
	buffer_load_b128 v[136:139] /*v[648:651]*/, v2 /*v514*/, s[28:31], null offen offset:64
	buffer_load_b128 v[140:143] /*v[652:655]*/, v2 /*v514*/, s[28:31], null offen offset:96
	buffer_load_b128 v[144:147] /*v[656:659]*/, v3 /*v515*/, s[28:31], null offen
	buffer_load_b128 v[148:151] /*v[660:663]*/, v4 /*v516*/, s[28:31], null offen
	buffer_load_b128 v[152:155] /*v[664:667]*/, v5 /*v517*/, s[28:31], null offen
	buffer_load_b128 v[156:159] /*v[668:671]*/, v6 /*v518*/, s[28:31], null offen
	s_clause 0x7
	buffer_load_b128 v[2:5] /*v[514:517]*/, v31 /*v543*/, s[36:39], null offen
	buffer_load_b128 v[6:9] /*v[518:521]*/, v31 /*v543*/, s[36:39], null offen offset:32
	buffer_load_b128 v[10:13] /*v[522:525]*/, v31 /*v543*/, s[36:39], null offen offset:64
	buffer_load_b128 v[14:17] /*v[526:529]*/, v31 /*v543*/, s[36:39], null offen offset:96
	buffer_load_b128 v[18:21] /*v[530:533]*/, v40 /*v552*/, s[36:39], null offen
	buffer_load_b128 v[22:25] /*v[534:537]*/, v41 /*v553*/, s[36:39], null offen
	buffer_load_b128 v[26:29] /*v[538:541]*/, v42 /*v554*/, s[36:39], null offen
	buffer_load_b128 v[160:163] /*v[672:675]*/, v43 /*v555*/, s[36:39], null offen
	s_clause 0x7
	buffer_load_b128 v[164:167] /*v[676:679]*/, v31 /*v543*/, s[28:31], null offen
	buffer_load_b128 v[168:171] /*v[680:683]*/, v31 /*v543*/, s[28:31], null offen offset:32
	buffer_load_b128 v[172:175] /*v[684:687]*/, v31 /*v543*/, s[28:31], null offen offset:64
	buffer_load_b128 v[176:179] /*v[688:691]*/, v31 /*v543*/, s[28:31], null offen offset:96
	buffer_load_b128 v[180:183] /*v[692:695]*/, v40 /*v552*/, s[28:31], null offen
	buffer_load_b128 v[184:187] /*v[696:699]*/, v41 /*v553*/, s[28:31], null offen
	buffer_load_b128 v[188:191] /*v[700:703]*/, v42 /*v554*/, s[28:31], null offen
	buffer_load_b128 v[192:195] /*v[704:707]*/, v43 /*v555*/, s[28:31], null offen
	s_set_vgpr_msb 0x8a85
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[2:9] /*v[258:265]*/, v[250:257] /*v[506:513]*/, 0
	s_set_vgpr_msb 0x8507
	ds_store_b128 v145 /*v913*/, v[218:221] /*v[474:477]*/
	ds_store_b128 v145 /*v913*/, v[222:225] /*v[478:481]*/ offset:32
	ds_store_b128 v145 /*v913*/, v[250:253] /*v[506:509]*/ offset:8704
	ds_store_b128 v145 /*v913*/, v[254:257] /*v[510:513]*/ offset:8736
	ds_store_b128 v145 /*v913*/, v[210:213] /*v[466:469]*/ offset:64
	ds_store_b128 v145 /*v913*/, v[214:217] /*v[470:473]*/ offset:96
	ds_store_b128 v145 /*v913*/, v[242:245] /*v[498:501]*/ offset:8768
	ds_store_b128 v145 /*v913*/, v[246:249] /*v[502:505]*/ offset:8800
	ds_store_b128 v145 /*v913*/, v[202:205] /*v[458:461]*/ offset:128
	s_set_vgpr_msb 0x78a
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[88:89] /*v[600:601]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[88:89] /*v[600:601]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[88:89] /*v[600:601]*/, v[38:39] /*v[550:551]*/
	s_set_vgpr_msb 0x8a07
	ds_store_b128 v145 /*v913*/, v[234:237] /*v[490:493]*/ offset:8832
	ds_store_b128 v145 /*v913*/, v[238:241] /*v[494:497]*/ offset:8864
	ds_store_b128 v145 /*v913*/, v[206:209] /*v[462:465]*/ offset:160
	ds_store_b128 v145 /*v913*/, v[194:197] /*v[450:453]*/ offset:192
	ds_store_b128 v145 /*v913*/, v[198:201] /*v[454:457]*/ offset:224
	ds_store_b128 v145 /*v913*/, v[226:229] /*v[482:485]*/ offset:8896
	ds_store_b128 v145 /*v913*/, v[230:233] /*v[486:489]*/ offset:8928
	s_set_vgpr_msb 0x78f
	v_add_nc_u32_e32 v31 /*v543*/, v148 /*v916*/, v147 /*v915*/
	s_set_vgpr_msb 0x8fa5
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[10:17] /*v[266:273]*/, v[242:249] /*v[498:505]*/, v[40:47] /*v[552:559]*/
	s_set_vgpr_msb 0xa5a4
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[242:249], v[234:241] /*v[490:497]*/, v[40:47] /*v[552:559]*/
	s_wait_loadcnt 0x23
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa44a
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[36:37] /*v[548:549]*/, v[80:81] /*v[592:593]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a45
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[18:25] /*v[274:281]*/, v[218:225] /*v[474:481]*/, 0
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[240:241] /*v[496:497]*/, v[38:39] /*v[550:551]*/, v[80:81] /*v[592:593]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a44
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[194:201], v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x44a4
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[560:567]*/, v[202:209], v[210:217] /*v[466:473]*/, v[48:55] /*v[560:567]*/
	s_set_vgpr_msb 0xa4a5
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[552:559]*/, v[42:49] /*v[298:305]*/, v[226:233] /*v[482:489]*/, v[40:47] /*v[552:559]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa54a
	v_pk_add_f32 v[226:227] /*v[482:483]*/, v[32:33] /*v[544:545]*/, v[80:81] /*v[592:593]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a8a
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[88:89] /*v[600:601]*/, v[40:41] /*v[552:553]*/
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[584:591]*/, v[226:233], v[154:161] /*v[410:417]*/, v[72:79] /*v[584:591]*/
	s_set_vgpr_msb 0xa48a
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[88:89] /*v[600:601]*/, v[42:43] /*v[554:555]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[88:89] /*v[600:601]*/, v[44:45] /*v[556:557]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[88:89] /*v[600:601]*/, v[46:47] /*v[558:559]*/
	s_set_vgpr_msb 0x8a4a
	v_pk_add_f32 v[250:251] /*v[506:507]*/, v[40:41] /*v[552:553]*/, v[80:81] /*v[592:593]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[252:253] /*v[508:509]*/, v[42:43] /*v[554:555]*/, v[80:81] /*v[592:593]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[44:45] /*v[556:557]*/, v[80:81] /*v[592:593]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa5
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[576:583]*/, v[42:49] /*v[298:305]*/, v[154:161] /*v[410:417]*/, v[64:71] /*v[576:583]*/
	s_set_vgpr_msb 0xa54a
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[88:89] /*v[600:601]*/, v[76:77] /*v[588:589]*/
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[88:89] /*v[600:601]*/, v[78:79] /*v[590:591]*/
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[88:89] /*v[600:601]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[88:89] /*v[600:601]*/, v[74:75] /*v[586:587]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[0:1] /*v[512:513]*/, v[46:47] /*v[558:559]*/, v[80:81] /*v[592:593]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a49
	s_wait_loadcnt 0x22
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[232:233] /*v[488:489]*/, v[82:83] /*v[594:595]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[234:235] /*v[490:491]*/, v[82:83] /*v[594:595]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4955
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[26:33] /*v[282:289]*/, v[210:217] /*v[466:473]*/, v[242:249] /*v[498:505]*/
	s_set_vgpr_msb 0x554a
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[88:89] /*v[600:601]*/, v[64:65] /*v[576:577]*/
	s_set_vgpr_msb 0x4a49
	v_pk_add_f32 v[228:229] /*v[484:485]*/, v[228:229] /*v[484:485]*/, v[82:83] /*v[594:595]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, v[82:83] /*v[594:595]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_delay_alu instid0(VALU_DEP_2)
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[228:229] /*v[484:485]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x49a5
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[568:575]*/, v[26:33] /*v[282:289]*/, v[146:153] /*v[402:409]*/, v[56:63] /*v[568:575]*/
	s_set_vgpr_msb 0xa54a
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[88:89] /*v[600:601]*/, v[66:67] /*v[578:579]*/
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[88:89] /*v[600:601]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[88:89] /*v[600:601]*/, v[70:71] /*v[582:583]*/
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[34:35] /*v[546:547]*/, v[80:81] /*v[592:593]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a41
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v228 /*v484*/, v228 /*v484*/
	v_exp_f32_e32 v229 /*v485*/, v229 /*v485*/
	s_set_vgpr_msb 0x4154
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[202:209], v[146:153] /*v[402:409]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5441
	v_exp_f32_e32 v230 /*v486*/, v230 /*v486*/
	v_exp_f32_e32 v231 /*v487*/, v231 /*v487*/
	s_set_vgpr_msb 0x41a4
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[560:567]*/, v[218:225], v[202:209] /*v[458:465]*/, v[48:55] /*v[560:567]*/
	s_set_vgpr_msb 0xa454
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[250:257], v[202:209] /*v[458:465]*/, v[242:249] /*v[498:505]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5449
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[236:237] /*v[492:493]*/, v[82:83] /*v[594:595]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[210:211] /*v[466:467]*/, v[82:83] /*v[594:595]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[212:213] /*v[468:469]*/, v[82:83] /*v[594:595]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x49a4
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[568:575]*/, v[250:257], v[138:145] /*v[394:401]*/, v[56:63] /*v[568:575]*/
	s_set_vgpr_msb 0xa449
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[214:215] /*v[470:471]*/, v[82:83] /*v[594:595]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[226:227] /*v[482:483]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[216:217] /*v[472:473]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[238:239] /*v[494:495]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[240:241] /*v[496:497]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[250:251] /*v[506:507]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[252:253] /*v[508:509]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4954
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[218:225], v[138:145] /*v[394:401]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5441
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[254:255] /*v[510:511]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x4142
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[0:1] /*v[512:513]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x42a4
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[560:567]*/, v[234:241], v[194:201] /*v[450:457]*/, v[48:55] /*v[560:567]*/
	s_set_vgpr_msb 0xa455
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[50:57] /*v[306:313]*/, v[194:201] /*v[450:457]*/, v[242:249] /*v[498:505]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[232:233] /*v[488:489]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[234:235] /*v[490:491]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[202:203] /*v[458:459]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[204:205] /*v[460:461]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x55a5
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[568:575]*/, v[50:57] /*v[306:313]*/, v[130:137] /*v[386:393]*/, v[56:63] /*v[568:575]*/
	s_set_vgpr_msb 0xa541
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[206:207] /*v[462:463]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[208:209] /*v[464:465]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v206 /*v462*/, v210 /*v466*/
	v_exp_f32_e32 v207 /*v463*/, v211 /*v467*/
	v_exp_f32_e32 v208 /*v464*/, v212 /*v468*/
	v_exp_f32_e32 v209 /*v465*/, v213 /*v469*/
	v_exp_f32_e32 v210 /*v466*/, v214 /*v470*/
	s_set_vgpr_msb 0x4154
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[234:241], v[130:137] /*v[386:393]*/, v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5441
	v_exp_f32_e32 v211 /*v467*/, v215 /*v471*/
	v_exp_f32_e32 v212 /*v468*/, v216 /*v472*/
	v_exp_f32_e32 v213 /*v469*/, v217 /*v473*/
	v_exp_f32_e32 v214 /*v470*/, v226 /*v482*/
	v_exp_f32_e32 v215 /*v471*/, v227 /*v483*/
	v_exp_f32_e32 v216 /*v472*/, v236 /*v492*/
	v_exp_f32_e32 v217 /*v473*/, v237 /*v493*/
	v_exp_f32_e32 v226 /*v482*/, v238 /*v494*/
	v_exp_f32_e32 v227 /*v483*/, v239 /*v495*/
	v_exp_f32_e32 v232 /*v488*/, v240 /*v496*/
	v_exp_f32_e32 v233 /*v489*/, v241 /*v497*/
	v_exp_f32_e32 v234 /*v490*/, v194 /*v450*/
	v_exp_f32_e32 v235 /*v491*/, v195 /*v451*/
	v_exp_f32_e32 v236 /*v492*/, v196 /*v452*/
	v_exp_f32_e32 v237 /*v493*/, v197 /*v453*/
	v_exp_f32_e32 v238 /*v494*/, v198 /*v454*/
	v_exp_f32_e32 v239 /*v495*/, v199 /*v455*/
	v_exp_f32_e32 v240 /*v496*/, v200 /*v456*/
	v_exp_f32_e32 v241 /*v497*/, v201 /*v457*/
	s_set_vgpr_msb 0x414a
	s_wait_loadcnt 0x21
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[48:49] /*v[560:561]*/, v[84:85] /*v[596:597]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[50:51] /*v[562:563]*/, v[84:85] /*v[596:597]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[52:53] /*v[564:565]*/, v[84:85] /*v[596:597]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[54:55] /*v[566:567]*/, v[84:85] /*v[596:597]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a49
	v_exp_f32_e32 v250 /*v506*/, v202 /*v458*/
	v_exp_f32_e32 v251 /*v507*/, v203 /*v459*/
	v_exp_f32_e32 v252 /*v508*/, v204 /*v460*/
	v_exp_f32_e32 v253 /*v509*/, v205 /*v461*/
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[242:243] /*v[498:499]*/, v[84:85] /*v[596:597]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[244:245] /*v[500:501]*/, v[84:85] /*v[596:597]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[242:243] /*v[498:499]*/, v[246:247] /*v[502:503]*/, v[84:85] /*v[596:597]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[244:245] /*v[500:501]*/, v[248:249] /*v[504:505]*/, v[84:85] /*v[596:597]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_wait_loadcnt 0x20
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[86:87] /*v[598:599]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, v[86:87] /*v[598:599]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[86:87] /*v[598:599]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[86:87] /*v[598:599]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[56:57] /*v[568:569]*/, v[86:87] /*v[598:599]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[58:59] /*v[570:571]*/, v[86:87] /*v[598:599]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[60:61] /*v[572:573]*/, v[86:87] /*v[598:599]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[0:1] /*v[512:513]*/, v[62:63] /*v[574:575]*/, v[86:87] /*v[598:599]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a45
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[206:207] /*v[462:463]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[208:209] /*v[464:465]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[210:211] /*v[466:467]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[212:213] /*v[468:469]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, v[214:215] /*v[470:471]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[204:205] /*v[460:461]*/, v[216:217] /*v[472:473]*/
	v_pk_mul_f32 v[242:243] /*v[498:499]*/, v[242:243] /*v[498:499]*/, v[226:227] /*v[482:483]*/
	v_pk_mul_f32 v[244:245] /*v[500:501]*/, v[244:245] /*v[500:501]*/, v[232:233] /*v[488:489]*/
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, v[230:231] /*v[486:487]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[234:235] /*v[490:491]*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[246:247] /*v[502:503]*/, v[238:239] /*v[494:495]*/
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[248:249] /*v[504:505]*/, v[240:241] /*v[496:497]*/
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, v[250:251] /*v[506:507]*/
	s_set_vgpr_msb 0x4586
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[0:1] /*v[512:513]*/, v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x8646
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[88:89] /*v[600:601]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[88:89] /*v[600:601]*/, v[196:197] /*v[452:453]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[88:89] /*v[600:601]*/, v[198:199] /*v[454:455]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[88:89] /*v[600:601]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[88:89] /*v[600:601]*/, v[202:203] /*v[458:459]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[88:89] /*v[600:601]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[242:243] /*v[498:499]*/, v[88:89] /*v[600:601]*/, v[242:243] /*v[498:499]*/
	v_pk_mul_f32 v[244:245] /*v[500:501]*/, v[88:89] /*v[600:601]*/, v[244:245] /*v[500:501]*/
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[88:89] /*v[600:601]*/, v[218:219] /*v[474:475]*/
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[88:89] /*v[600:601]*/, v[220:221] /*v[476:477]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[88:89] /*v[600:601]*/, v[222:223] /*v[478:479]*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[88:89] /*v[600:601]*/, v[224:225] /*v[480:481]*/
	v_pk_mul_f32 v[246:247] /*v[502:503]*/, v[88:89] /*v[600:601]*/, v[246:247] /*v[502:503]*/
	v_pk_mul_f32 v[248:249] /*v[504:505]*/, v[88:89] /*v[600:601]*/, v[248:249] /*v[504:505]*/
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[88:89] /*v[600:601]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x468a
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[88:89] /*v[600:601]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x8a45
	v_cvt_pk_bf16_f32 v194 /*v450*/, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v196 /*v452*/, v197 /*v453*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v198 /*v454*/, v199 /*v455*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v200 /*v456*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v201 /*v457*/, v212 /*v468*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v200 /*v456*/, v210 /*v466*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v199 /*v455*/, v208 /*v464*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v209 /*v465*/, v232 /*v488*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v208 /*v464*/, v226 /*v482*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v207 /*v463*/, v216 /*v472*/, v217 /*v473*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v214 /*v470*/, v215 /*v471*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v203 /*v459*/, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v242 /*v498*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v244 /*v500*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v210 /*v466*/, v218 /*v474*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v211 /*v467*/, v220 /*v476*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v212 /*v468*/, v222 /*v478*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v213 /*v469*/, v224 /*v480*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v217 /*v473*/, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v216 /*v472*/, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v215 /*v471*/, v230 /*v486*/, v231 /*v487*/
	v_cvt_pk_bf16_f32 v214 /*v470*/, v228 /*v484*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v218 /*v474*/, v246 /*v502*/, v247 /*v503*/
	v_cvt_pk_bf16_f32 v219 /*v475*/, v248 /*v504*/, v249 /*v505*/
	v_cvt_pk_bf16_f32 v220 /*v476*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v221 /*v477*/, v0 /*v512*/, v1 /*v513*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v225 /*v481*/, v252 /*v508*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v224 /*v480*/, v250 /*v506*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v223 /*v479*/, v240 /*v496*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v222 /*v478*/, v238 /*v494*/, v239 /*v495*/
	s_set_vgpr_msb 0x4507
	ds_store_b128 v146 /*v914*/, v[198:201] /*v[454:457]*/
	ds_store_b128 v146 /*v914*/, v[206:209] /*v[462:465]*/ offset:32
	ds_store_b128 v146 /*v914*/, v[194:197] /*v[450:453]*/ offset:2560
	ds_store_b128 v146 /*v914*/, v[202:205] /*v[458:461]*/ offset:2592
	ds_store_b128 v145 /*v913*/, v[162:165] /*v[418:421]*/ offset:4352
	ds_store_b128 v145 /*v913*/, v[166:169] /*v[422:425]*/ offset:4384
	ds_store_b128 v145 /*v913*/, v[186:189] /*v[442:445]*/ offset:13056
	ds_store_b128 v145 /*v913*/, v[190:193] /*v[446:449]*/ offset:13088
	ds_store_b128 v145 /*v913*/, v[146:149] /*v[402:405]*/ offset:4416
	ds_store_b128 v145 /*v913*/, v[150:153] /*v[406:409]*/ offset:4448
	ds_store_b128 v145 /*v913*/, v[178:181] /*v[434:437]*/ offset:13120
	ds_store_b128 v145 /*v913*/, v[182:185] /*v[438:441]*/ offset:13152
	ds_store_b128 v145 /*v913*/, v[138:141] /*v[394:397]*/ offset:4480
	ds_store_b128 v145 /*v913*/, v[142:145] /*v[398:401]*/ offset:4512
	ds_store_b128 v145 /*v913*/, v[170:173] /*v[426:429]*/ offset:13184
	ds_store_b128 v145 /*v913*/, v[174:177] /*v[430:433]*/ offset:13216
	ds_store_b128 v145 /*v913*/, v[130:133] /*v[386:389]*/ offset:4544
	ds_store_b128 v145 /*v913*/, v[134:137] /*v[390:393]*/ offset:4576
	ds_store_b128 v145 /*v913*/, v[154:157] /*v[410:413]*/ offset:13248
	ds_store_b128 v145 /*v913*/, v[158:161] /*v[414:417]*/ offset:13280
	ds_store_b128 v146 /*v914*/, v[214:217] /*v[470:473]*/ offset:1280
	ds_store_b128 v146 /*v914*/, v[222:225] /*v[478:481]*/ offset:1312
	ds_store_b128 v146 /*v914*/, v[210:213] /*v[466:469]*/ offset:3840
	ds_store_b128 v146 /*v914*/, v[218:221] /*v[474:477]*/ offset:3872
	s_set_vgpr_msb 0x742
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v31 /*v543*/
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v31 /*v543*/ offset:4352
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v81 /*v593*/
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v81 /*v593*/ offset:4352
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v83 /*v595*/
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v83 /*v595*/ offset:4352
	ds_load_tr16_b128 v[250:253] /*v[506:509]*/, v85 /*v597*/
	ds_load_tr16_b128 v[254:257] /*v[510:513]*/, v85 /*v597*/ offset:4352
	ds_load_tr16_b128 v[130:133] /*v[386:389]*/, v87 /*v599*/
	ds_load_tr16_b128 v[134:137] /*v[390:393]*/, v87 /*v599*/ offset:4352
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v90 /*v602*/
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v90 /*v602*/ offset:4352
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v91 /*v603*/
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v91 /*v603*/ offset:4352
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v92 /*v604*/
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v92 /*v604*/ offset:4352
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v93 /*v605*/
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v93 /*v605*/ offset:4352
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v94 /*v606*/
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v94 /*v606*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v95 /*v607*/
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v95 /*v607*/ offset:4352
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v196 /*v708*/
	ds_load_tr16_b128 v[206:209] /*v[462:465]*/, v196 /*v708*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v197 /*v709*/
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v197 /*v709*/ offset:4352
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v198 /*v710*/
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v198 /*v710*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v199 /*v711*/
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v199 /*v711*/ offset:4352
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v200 /*v712*/
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v200 /*v712*/ offset:4352
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v201 /*v713*/
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v201 /*v713*/ offset:1280
	ds_load_tr16_b128 v[72:75] /*v[584:587]*/, v202 /*v714*/
	ds_load_tr16_b128 v[76:79] /*v[588:591]*/, v202 /*v714*/ offset:1280
	ds_load_tr16_b128 v[80:83] /*v[592:595]*/, v203 /*v715*/
	ds_load_tr16_b128 v[84:87] /*v[596:599]*/, v203 /*v715*/ offset:1280
	ds_load_tr16_b128 v[88:91] /*v[600:603]*/, v204 /*v716*/
	ds_load_tr16_b128 v[92:95] /*v[604:607]*/, v204 /*v716*/ offset:1280
	s_set_vgpr_msb 0x8256
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[64:71] /*v[576:583]*/, v[170:177] /*v[426:433]*/, v[122:129] /*v[378:385]*/
	s_wait_loadcnt 0x16
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[132:133] /*v[644:645]*/
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[134:135] /*v[646:647]*/
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[128:129] /*v[640:641]*/
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[130:131] /*v[642:643]*/
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[124:125] /*v[636:637]*/
	v_mov_b64_e32 v[232:233] /*v[488:489]*/, v[126:127] /*v[638:639]*/
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[120:121] /*v[632:633]*/
	s_set_vgpr_msb 0x5606
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[106:113], v[80:87] /*v[592:599]*/, v[186:193] /*v[442:449]*/, v[106:113]
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[122:123] /*v[634:635]*/
	v_mov_b64_e32 v[238:239] /*v[494:495]*/, v[116:117] /*v[628:629]*/
	v_mov_b64_e32 v[240:241] /*v[496:497]*/, v[118:119] /*v[630:631]*/
	v_mov_b64_e32 v[234:235] /*v[490:491]*/, v[112:113] /*v[624:625]*/
	v_mov_b64_e32 v[236:237] /*v[492:493]*/, v[114:115] /*v[626:627]*/
	v_mov_b64_e32 v[246:247] /*v[502:503]*/, v[108:109] /*v[620:621]*/
	v_mov_b64_e32 v[248:249] /*v[504:505]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x4206
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[88:95] /*v[600:607]*/, v[250:257] /*v[506:513]*/, v[98:105]
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[242:243] /*v[498:499]*/, v[104:105] /*v[616:617]*/
	v_mov_b64_e32 v[244:245] /*v[500:501]*/, v[106:107] /*v[618:619]*/
	s_set_vgpr_msb 0x4206
	v_wmma_f32_16x16x32_bf16 v[82:89], v[88:95] /*v[600:607]*/, v[138:145] /*v[394:401]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[66:73], v[88:95] /*v[600:607]*/, v[154:161] /*v[410:417]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[88:95] /*v[600:607]*/, v[194:201] /*v[450:457]*/, v[50:57]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[88:95] /*v[600:607]*/, v[202:209] /*v[458:465]*/, v[34:41]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[88:95] /*v[600:607]*/, v[210:217] /*v[466:473]*/, v[18:25]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[80:87] /*v[592:599]*/, v[130:137] /*v[386:393]*/, v[90:97]
	v_wmma_f32_16x16x32_bf16 v[74:81], v[80:87] /*v[592:599]*/, v[146:153] /*v[402:409]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[80:87] /*v[592:599]*/, v[162:169] /*v[418:425]*/, v[58:65]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[42:49], v[80:87] /*v[592:599]*/, v[32:39] /*v[544:551]*/, v[42:49]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[80:87] /*v[592:599]*/, v[40:47] /*v[552:559]*/, v[26:33]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[80:87] /*v[592:599]*/, v[48:55] /*v[560:567]*/, v[10:17]
	s_set_vgpr_msb 0xa56
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[72:79] /*v[584:591]*/, v[138:145] /*v[394:401]*/, v[82:89] /*v[338:345]*/
	s_wait_loadcnt 0x2
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[184:185] /*v[696:697]*/
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[186:187] /*v[698:699]*/
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[180:181] /*v[692:693]*/
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[72:79] /*v[584:591]*/, v[154:161] /*v[410:417]*/, v[66:73] /*v[322:329]*/
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[182:183] /*v[694:695]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[158:159] /*v[414:415]*/, v[160:161] /*v[672:673]*/
	v_mov_b64_e32 v[160:161] /*v[416:417]*/, v[162:163] /*v[674:675]*/
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[26:27] /*v[538:539]*/
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[72:79] /*v[584:591]*/, v[194:201] /*v[450:457]*/, v[34:41] /*v[290:297]*/
	v_mov_b64_e32 v[156:157] /*v[412:413]*/, v[28:29] /*v[540:541]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[198:199] /*v[454:455]*/, v[156:157] /*v[668:669]*/
	v_mov_b64_e32 v[200:201] /*v[456:457]*/, v[158:159] /*v[670:671]*/
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, v[152:153] /*v[664:665]*/
	s_set_vgpr_msb 0x5606
	v_wmma_f32_16x16x32_bf16 v[162:169], v[72:79] /*v[584:591]*/, v[202:209] /*v[458:465]*/, v[162:169]
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[196:197] /*v[452:453]*/, v[154:155] /*v[666:667]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[206:207] /*v[462:463]*/, v[148:149] /*v[660:661]*/
	v_mov_b64_e32 v[208:209] /*v[464:465]*/, v[150:151] /*v[662:663]*/
	v_mov_b64_e32 v[202:203] /*v[458:459]*/, v[144:145] /*v[656:657]*/
	s_set_vgpr_msb 0x4206
	v_wmma_f32_16x16x32_bf16 v[146:153], v[72:79] /*v[584:591]*/, v[210:217] /*v[466:473]*/, v[146:153]
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[204:205] /*v[460:461]*/, v[146:147] /*v[658:659]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[214:215] /*v[470:471]*/, v[140:141] /*v[652:653]*/
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[142:143] /*v[654:655]*/
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[136:137] /*v[648:649]*/
	s_set_vgpr_msb 0x420a
	v_wmma_f32_16x16x32_bf16 v[130:137], v[72:79] /*v[584:591]*/, v[56:63] /*v[568:575]*/, v[130:137]
	s_set_vgpr_msb 0xa56
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[138:139] /*v[650:651]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[64:71] /*v[576:583]*/, v[130:137] /*v[386:393]*/, v[90:97] /*v[346:353]*/
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[134:135] /*v[390:391]*/, v[192:193] /*v[704:705]*/
	v_mov_b64_e32 v[136:137] /*v[392:393]*/, v[194:195] /*v[706:707]*/
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[188:189] /*v[700:701]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[64:71] /*v[576:583]*/, v[146:153] /*v[402:409]*/, v[74:81] /*v[330:337]*/
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, v[190:191] /*v[702:703]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[176:177] /*v[688:689]*/
	v_mov_b64_e32 v[152:153] /*v[408:409]*/, v[178:179] /*v[690:691]*/
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[172:173] /*v[684:685]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[64:71] /*v[576:583]*/, v[162:169] /*v[418:425]*/, v[58:65] /*v[314:321]*/
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[174:175] /*v[686:687]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[166:167] /*v[422:423]*/, v[168:169] /*v[680:681]*/
	v_mov_b64_e32 v[168:169] /*v[424:425]*/, v[170:171] /*v[682:683]*/
	v_mov_b64_e32 v[162:163] /*v[418:419]*/, v[164:165] /*v[676:677]*/
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[72:79] /*v[584:591]*/, v[178:185] /*v[434:441]*/, v[114:121] /*v[370:377]*/
	v_mov_b64_e32 v[164:165] /*v[420:421]*/, v[166:167] /*v[678:679]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[64:71] /*v[576:583]*/, v[186:193] /*v[442:449]*/, v[106:113] /*v[362:369]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[190:191] /*v[446:447]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[192:193] /*v[448:449]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[186:187] /*v[442:443]*/, v[2:3] /*v[514:515]*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[72:79] /*v[584:591]*/, v[250:257] /*v[506:513]*/, v[98:105] /*v[354:361]*/
	v_mov_b64_e32 v[188:189] /*v[444:445]*/, v[4:5] /*v[516:517]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[254:255] /*v[510:511]*/, v[100:101] /*v[612:613]*/
	s_set_vgpr_msb 0x5682
	v_mov_b64_e32 v[0:1] /*v[512:513]*/, v[102:103] /*v[614:615]*/
	s_set_vgpr_msb 0x8242
	v_mov_b64_e32 v[250:251] /*v[506:507]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x420a
	v_wmma_f32_16x16x32_bf16 v[186:193], v[64:71] /*v[576:583]*/, v[32:39] /*v[544:551]*/, v[186:193]
	s_set_vgpr_msb 0xa42
	v_mov_b64_e32 v[252:253] /*v[508:509]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x420a
	v_wmma_f32_16x16x32_bf16 v[154:161], v[64:71] /*v[576:583]*/, v[40:47] /*v[552:559]*/, v[154:161]
	v_wmma_f32_16x16x32_bf16 v[138:145], v[64:71] /*v[576:583]*/, v[48:55] /*v[560:567]*/, v[138:145]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[122:129], v[80:87] /*v[592:599]*/, v[170:177] /*v[426:433]*/, v[122:129]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[22:23] /*v[534:535]*/
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[24:25] /*v[536:537]*/
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x4206
	v_wmma_f32_16x16x32_bf16 v[114:121], v[88:95] /*v[600:607]*/, v[178:185] /*v[434:441]*/, v[114:121]
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[20:21] /*v[532:533]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[182:183] /*v[438:439]*/, v[14:15] /*v[526:527]*/
	v_mov_b64_e32 v[184:185] /*v[440:441]*/, v[16:17] /*v[528:529]*/
	v_mov_b64_e32 v[178:179] /*v[434:435]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x420a
	v_wmma_f32_16x16x32_bf16 v[2:9], v[88:95] /*v[600:607]*/, v[56:63] /*v[568:575]*/, v[2:9]
	s_set_vgpr_msb 0xa42
	v_mov_b64_e32 v[180:181] /*v[436:437]*/, v[12:13] /*v[524:525]*/
	s_set_vgpr_msb 0x4200
	s_cbranch_scc1 .LBB0_6
.LBB0_7:
	s_mul_i32 s4, s20, s51
	s_clause 0x1
	s_load_b64 s[6:7], s[0:1], 0x110 nv
	s_load_b64 s[8:9], s[0:1], 0x140 nv
	s_add_co_i32 s4, s4, s50
	s_set_vgpr_msb 13
	v_mul_lo_u32 v176, s20, v140 /*v908*/
	v_mad_u32 v170, s20, v144 /*v912*/, s4
	v_mul_lo_u32 v177, s20, v139 /*v907*/
	s_mov_b32 s0, 0
	s_lshl_b32 s1, s34, 25
	v_cvt_pk_bf16_f32 v171, v122 /*v378*/, s0
	v_cvt_pk_bf16_f32 v173, v123 /*v379*/, s0
	v_cvt_pk_bf16_f32 v172, v114 /*v370*/, s0
	s_mov_b32 s2, s26
	s_set_vgpr_msb 0xd00
	v_lshlrev_b32_e32 v174, 7, v170
	v_add_lshl_u32 v178, v170, s20, 7
	s_mov_b32 s3, s27
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v175, v115 /*v371*/, s0
	v_add_lshl_u32 v176, s4, v176, 7
	s_set_vgpr_msb 0x10c
	v_or_b32_e32 v179, v174, v141 /*v909*/
	v_or_b32_e32 v180, v178, v141 /*v909*/
	v_mul_lo_u32 v182, s20, v138 /*v906*/
	s_wait_kmcnt 0x0
	s_or_b64 s[24:25], s[6:7], s[0:1]
	s_or_b64 s[0:1], s[8:9], s[0:1]
	s_set_vgpr_msb 0xc00
	v_dual_lshlrev_b32 v179, 2, v179 :: v_dual_lshlrev_b32 v180, 2, v180
	s_set_vgpr_msb 13
	v_mul_lo_u32 v184, s20, v137 /*v905*/
	v_cvt_pk_bf16_f32 v181, v125 /*v381*/, s0
	v_cvt_pk_bf16_f32 v183, v117 /*v373*/, s0
	s_set_vgpr_msb 0xd00
	buffer_store_b16 v171, v179, s[24:27], null offen
	buffer_store_b16 v172, v179, s[0:3], null offen
	buffer_store_b16 v173, v180, s[24:27], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v173, s4, v177, 7
	v_mov_b16_e64 v171.l, v175.l
	s_set_vgpr_msb 12
	v_or_b32_e32 v172, v176, v141 /*v909*/
	s_set_vgpr_msb 0xc0d
	v_cvt_pk_bf16_f32 v175, v124 /*v380*/, s0
	v_mul_lo_u32 v195, s20, v135 /*v903*/
	s_set_vgpr_msb 0xd0c
	v_or_b32_e32 v177, v173, v141 /*v909*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v194, v127 /*v383*/, s0
	v_cvt_pk_bf16_f32 v198, v129 /*v385*/, s0
	v_cvt_pk_bf16_f32 v196, v120 /*v376*/, s0
	v_cvt_pk_bf16_f32 v199, v107 /*v363*/, s0
	v_lshlrev_b32_e32 v177, 2, v177
	s_set_vgpr_msb 0x100
	buffer_store_b16 v171, v180, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v171, 2, v172
	v_mov_b16_e64 v172.l, v175.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v175, v116 /*v372*/, s0
	v_add_lshl_u32 v195, s4, v195, 7
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v186, v186, s0
	v_cvt_pk_bf16_f32 v188, v188, s0
	buffer_store_b16 v172, v171, s[24:27], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v172, s4, v182, 7
	buffer_store_b16 v175, v171, s[0:3], null offen
	buffer_store_b16 v181, v177, s[24:27], null offen
	buffer_store_b16 v183, v177, s[0:3], null offen
	s_wait_xcnt 0x2
	v_add_lshl_u32 v175, s4, v184, 7
	s_set_vgpr_msb 13
	v_mul_lo_u32 v184, s20, v136 /*v904*/
	v_cvt_pk_bf16_f32 v182, v126 /*v382*/, s0
	s_set_vgpr_msb 0xd0c
	v_or_b32_e32 v181, v172, v141 /*v909*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v183, v118 /*v374*/, s0
	s_set_vgpr_msb 0x10c
	v_or_b32_e32 v185, v175, v141 /*v909*/
	v_or_b32_e32 v197, v195, v141 /*v909*/
	v_cvt_pk_bf16_f32 v162, v162, s0
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v181, 2, v181
	v_add_lshl_u32 v184, s4, v184, 7
	v_lshlrev_b32_e32 v185, 2, v185
	buffer_store_b16 v182, v181, s[24:27], null offen
	buffer_store_b16 v183, v181, s[0:3], null offen
	buffer_store_b16 v194, v185, s[24:27], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v183, v184, v141 /*v909*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v182, v119 /*v375*/, s0
	v_cvt_pk_bf16_f32 v194, v128 /*v384*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v187, v187, s0
	v_cvt_pk_bf16_f32 v163, v163, s0
	v_lshlrev_b32_e32 v183, 2, v183
	buffer_store_b16 v182, v185, s[0:3], null offen
	buffer_store_b16 v194, v183, s[24:27], null offen
	buffer_store_b16 v196, v183, s[0:3], null offen
	s_wait_xcnt 0x2
	v_lshlrev_b32_e32 v182, 2, v197
	s_wait_xcnt 0x1
	v_mov_b16_e64 v194.l, v198.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v121 /*v377*/, s0
	v_cvt_pk_bf16_f32 v197, v106 /*v362*/, s0
	v_cvt_pk_bf16_f32 v198, v98 /*v354*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v165, v165, s0
	buffer_store_b16 v194, v182, s[24:27], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v99 /*v355*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v182, s[0:3], null offen
	buffer_store_b16 v196, v179, s[24:27], null offen offset:64
	buffer_store_b16 v197, v179, s[0:3], null offen offset:64
	buffer_store_b16 v198, v180, s[24:27], null offen offset:64
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v108 /*v364*/, s0
	v_cvt_pk_bf16_f32 v197, v100 /*v356*/, s0
	v_cvt_pk_bf16_f32 v198, v109 /*v365*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v101 /*v357*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v166, v166, s0
	v_cvt_pk_bf16_f32 v147, v147, s0
	v_cvt_pk_bf16_f32 v154, v154, s0
	buffer_store_b16 v194, v180, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v110 /*v366*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v171, s[24:27], null offen offset:64
	buffer_store_b16 v196, v171, s[0:3], null offen offset:64
	buffer_store_b16 v197, v177, s[24:27], null offen offset:64
	buffer_store_b16 v198, v177, s[0:3], null offen offset:64
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v102 /*v358*/, s0
	v_cvt_pk_bf16_f32 v197, v111 /*v367*/, s0
	v_cvt_pk_bf16_f32 v198, v103 /*v359*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v112 /*v368*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v146, v146, s0
	v_cvt_pk_bf16_f32 v155, v155, s0
	v_cvt_pk_bf16_f32 v148, v148, s0
	buffer_store_b16 v194, v181, s[24:27], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v104 /*v360*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v181, s[0:3], null offen offset:64
	buffer_store_b16 v196, v185, s[24:27], null offen offset:64
	buffer_store_b16 v197, v185, s[0:3], null offen offset:64
	buffer_store_b16 v198, v183, s[24:27], null offen offset:64
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v113 /*v369*/, s0
	v_cvt_pk_bf16_f32 v197, v105 /*v361*/, s0
	v_cvt_pk_bf16_f32 v198, v90 /*v346*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v82 /*v338*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v149, v149, s0
	v_cvt_pk_bf16_f32 v138, v138, s0
	v_cvt_pk_bf16_f32 v130, v130, s0
	buffer_store_b16 v194, v183, s[0:3], null offen offset:64
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v91 /*v347*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v182, s[24:27], null offen offset:64
	buffer_store_b16 v196, v182, s[0:3], null offen offset:64
	buffer_store_b16 v197, v179, s[24:27], null offen offset:128
	buffer_store_b16 v198, v179, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v83 /*v339*/, s0
	v_cvt_pk_bf16_f32 v197, v92 /*v348*/, s0
	v_cvt_pk_bf16_f32 v198, v84 /*v340*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v93 /*v349*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v131, v131, s0
	v_cvt_pk_bf16_f32 v140, v140, s0
	v_cvt_pk_bf16_f32 v132, v132, s0
	buffer_store_b16 v194, v180, s[24:27], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v85 /*v341*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v180, s[0:3], null offen offset:128
	buffer_store_b16 v196, v171, s[24:27], null offen offset:128
	buffer_store_b16 v197, v171, s[0:3], null offen offset:128
	buffer_store_b16 v198, v177, s[24:27], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v94 /*v350*/, s0
	v_cvt_pk_bf16_f32 v197, v86 /*v342*/, s0
	v_cvt_pk_bf16_f32 v198, v95 /*v351*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v87 /*v343*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v134, v134, s0
	v_cvt_pk_bf16_f32 v136, v136, s0
	v_cvt_pk_bf16_f32 v122, v122, s0
	buffer_store_b16 v194, v177, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v96 /*v352*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v181, s[24:27], null offen offset:128
	buffer_store_b16 v196, v181, s[0:3], null offen offset:128
	buffer_store_b16 v197, v185, s[24:27], null offen offset:128
	buffer_store_b16 v198, v185, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v88 /*v344*/, s0
	v_cvt_pk_bf16_f32 v197, v97 /*v353*/, s0
	v_cvt_pk_bf16_f32 v198, v89 /*v345*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v74 /*v330*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v114, v114, s0
	v_cvt_pk_bf16_f32 v115, v115, s0
	v_cvt_pk_bf16_f32 v124, v124, s0
	buffer_store_b16 v194, v183, s[24:27], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v66 /*v322*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v183, s[0:3], null offen offset:128
	buffer_store_b16 v196, v182, s[24:27], null offen offset:128
	buffer_store_b16 v197, v182, s[0:3], null offen offset:128
	buffer_store_b16 v198, v179, s[24:27], null offen offset:192
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v75 /*v331*/, s0
	v_cvt_pk_bf16_f32 v197, v67 /*v323*/, s0
	v_cvt_pk_bf16_f32 v198, v76 /*v332*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v68 /*v324*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v116, v116, s0
	v_cvt_pk_bf16_f32 v118, v118, s0
	v_cvt_pk_bf16_f32 v117, v117, s0
	buffer_store_b16 v194, v179, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v77 /*v333*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v180, s[24:27], null offen offset:192
	buffer_store_b16 v196, v180, s[0:3], null offen offset:192
	buffer_store_b16 v197, v171, s[24:27], null offen offset:192
	buffer_store_b16 v198, v171, s[0:3], null offen offset:192
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v69 /*v325*/, s0
	v_cvt_pk_bf16_f32 v197, v78 /*v334*/, s0
	v_cvt_pk_bf16_f32 v198, v70 /*v326*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v79 /*v335*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v126, v126, s0
	v_mul_lo_u32 v1, s20, v1
	v_cvt_pk_bf16_f32 v119, v119, s0
	buffer_store_b16 v194, v177, s[24:27], null offen offset:192
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v71 /*v327*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v177, s[0:3], null offen offset:192
	buffer_store_b16 v196, v181, s[24:27], null offen offset:192
	buffer_store_b16 v197, v181, s[0:3], null offen offset:192
	buffer_store_b16 v198, v185, s[24:27], null offen offset:192
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v80 /*v336*/, s0
	v_cvt_pk_bf16_f32 v197, v72 /*v328*/, s0
	v_cvt_pk_bf16_f32 v198, v81 /*v337*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v73 /*v329*/, s0
	v_add_lshl_u32 v1, s4, v1, 7
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v120, v120, s0
	v_cvt_pk_bf16_f32 v121, v121, s0
	buffer_store_b16 v194, v185, s[0:3], null offen offset:192
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v58 /*v314*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v183, s[24:27], null offen offset:192
	buffer_store_b16 v196, v183, s[0:3], null offen offset:192
	buffer_store_b16 v197, v182, s[24:27], null offen offset:192
	buffer_store_b16 v198, v182, s[0:3], null offen offset:192
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v34 /*v290*/, s0
	v_cvt_pk_bf16_f32 v197, v59 /*v315*/, s0
	v_cvt_pk_bf16_f32 v198, v35 /*v291*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v60 /*v316*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v106, v106, s0
	v_cvt_pk_bf16_f32 v98, v98, s0
	v_cvt_pk_bf16_f32 v99, v99, s0
	buffer_store_b16 v194, v179, s[24:27], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v36 /*v292*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v179, s[0:3], null offen offset:256
	buffer_store_b16 v196, v180, s[24:27], null offen offset:256
	buffer_store_b16 v197, v180, s[0:3], null offen offset:256
	buffer_store_b16 v198, v171, s[24:27], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v61 /*v317*/, s0
	v_cvt_pk_bf16_f32 v197, v37 /*v293*/, s0
	v_cvt_pk_bf16_f32 v198, v62 /*v318*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v38 /*v294*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v100, v100, s0
	v_cvt_pk_bf16_f32 v90, v90, s0
	v_cvt_pk_bf16_f32 v82, v82, s0
	buffer_store_b16 v194, v171, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v63 /*v319*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v177, s[24:27], null offen offset:256
	buffer_store_b16 v196, v177, s[0:3], null offen offset:256
	buffer_store_b16 v197, v181, s[24:27], null offen offset:256
	buffer_store_b16 v198, v181, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v196, v39 /*v295*/, s0
	v_cvt_pk_bf16_f32 v197, v64 /*v320*/, s0
	v_cvt_pk_bf16_f32 v198, v40 /*v296*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v194.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v65 /*v321*/, s0
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v91, v91, s0
	v_cvt_pk_bf16_f32 v83, v83, s0
	v_cvt_pk_bf16_f32 v92, v92, s0
	buffer_store_b16 v194, v185, s[24:27], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v194.l, v196.l
	v_mov_b16_e64 v196.l, v197.l
	v_mov_b16_e64 v197.l, v198.l
	v_mov_b16_e64 v198.l, v199.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v199, v41 /*v297*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v194, v185, s[0:3], null offen offset:256
	buffer_store_b16 v196, v183, s[24:27], null offen offset:256
	buffer_store_b16 v197, v183, s[0:3], null offen offset:256
	buffer_store_b16 v198, v182, s[24:27], null offen offset:256
	v_cvt_pk_bf16_f32 v86, v86, s0
	v_cvt_pk_bf16_f32 v74, v74, s0
	v_cvt_pk_bf16_f32 v66, v66, s0
	s_wait_xcnt 0x3
	v_mov_b16_e64 v194.l, v199.l
	v_cvt_pk_bf16_f32 v75, v75, s0
	v_cvt_pk_bf16_f32 v67, v67, s0
	v_cvt_pk_bf16_f32 v69, v69, s0
	v_cvt_pk_bf16_f32 v58, v58, s0
	buffer_store_b16 v194, v182, s[0:3], null offen offset:256
	buffer_store_b16 v186, v179, s[24:27], null offen offset:320
	buffer_store_b16 v162, v179, s[0:3], null offen offset:320
	buffer_store_b16 v187, v180, s[24:27], null offen offset:320
	buffer_store_b16 v163, v180, s[0:3], null offen offset:320
	s_wait_xcnt 0x2
	v_mov_b16_e64 v162.l, v188.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v163, v164, s0
	v_cvt_pk_bf16_f32 v164, v189, s0
	v_cvt_pk_bf16_f32 v186, v190, s0
	v_cvt_pk_bf16_f32 v50, v50, s0
	buffer_store_b16 v162, v171, s[24:27], null offen offset:320
	s_wait_xcnt 0x0
	v_mov_b16_e64 v162.l, v163.l
	v_mov_b16_e64 v163.l, v164.l
	v_mov_b16_e64 v164.l, v165.l
	v_mov_b16_e64 v165.l, v186.l
	buffer_store_b16 v162, v171, s[0:3], null offen offset:320
	buffer_store_b16 v163, v177, s[24:27], null offen offset:320
	buffer_store_b16 v164, v177, s[0:3], null offen offset:320
	buffer_store_b16 v165, v181, s[24:27], null offen offset:320
	s_wait_xcnt 0x3
	v_mov_b16_e64 v162.l, v166.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v163, v191, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v164, v167, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v165, v192, s0
	v_cvt_pk_bf16_f32 v166, v168, s0
	buffer_store_b16 v162, v181, s[0:3], null offen offset:320
	s_wait_xcnt 0x0
	v_mov_b16_e64 v162.l, v163.l
	v_mov_b16_e64 v163.l, v164.l
	v_mov_b16_e64 v164.l, v165.l
	v_mov_b16_e64 v165.l, v166.l
	v_cvt_pk_bf16_f32 v166, v193, s0
	buffer_store_b16 v162, v185, s[24:27], null offen offset:320
	buffer_store_b16 v163, v185, s[0:3], null offen offset:320
	buffer_store_b16 v164, v183, s[24:27], null offen offset:320
	buffer_store_b16 v165, v183, s[0:3], null offen offset:320
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v163, v169, s0
	v_cvt_pk_bf16_f32 v59, v59, s0
	v_cvt_pk_bf16_f32 v52, v52, s0
	v_mov_b16_e64 v162.l, v166.l
	v_cvt_pk_bf16_f32 v53, v53, s0
	v_cvt_pk_bf16_f32 v42, v42, s0
	v_cvt_pk_bf16_f32 v34, v34, s0
	v_cvt_pk_bf16_f32 v35, v35, s0
	buffer_store_b16 v162, v182, s[24:27], null offen offset:320
	s_wait_xcnt 0x0
	v_mov_b16_e64 v162.l, v163.l
	buffer_store_b16 v162, v182, s[0:3], null offen offset:320
	buffer_store_b16 v154, v179, s[24:27], null offen offset:384
	buffer_store_b16 v146, v179, s[0:3], null offen offset:384
	buffer_store_b16 v155, v180, s[24:27], null offen offset:384
	s_wait_xcnt 0x1
	v_mov_b16_e64 v146.l, v147.l
	v_cvt_pk_bf16_f32 v147, v156, s0
	v_cvt_pk_bf16_f32 v154, v157, s0
	v_cvt_pk_bf16_f32 v36, v36, s0
	v_cvt_pk_bf16_f32 v26, v26, s0
	buffer_store_b16 v146, v180, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v146.l, v147.l
	v_mov_b16_e64 v147.l, v148.l
	v_mov_b16_e64 v148.l, v154.l
	v_cvt_pk_bf16_f32 v154, v158, s0
	buffer_store_b16 v146, v171, s[24:27], null offen offset:384
	buffer_store_b16 v147, v171, s[0:3], null offen offset:384
	buffer_store_b16 v148, v177, s[24:27], null offen offset:384
	buffer_store_b16 v149, v177, s[0:3], null offen offset:384
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v147, v150, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v148, v159, s0
	v_mov_b16_e64 v146.l, v154.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v149, v151, s0
	v_cvt_pk_bf16_f32 v150, v160, s0
	v_cvt_pk_bf16_f32 v18, v18, s0
	v_cvt_pk_bf16_f32 v19, v19, s0
	buffer_store_b16 v146, v181, s[24:27], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v146.l, v147.l
	v_mov_b16_e64 v147.l, v148.l
	v_mov_b16_e64 v148.l, v149.l
	v_mov_b16_e64 v149.l, v150.l
	v_cvt_pk_bf16_f32 v150, v152, s0
	buffer_store_b16 v146, v181, s[0:3], null offen offset:384
	buffer_store_b16 v147, v185, s[24:27], null offen offset:384
	buffer_store_b16 v148, v185, s[0:3], null offen offset:384
	buffer_store_b16 v149, v183, s[24:27], null offen offset:384
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v147, v161, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v148, v153, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v149, v174, v0
	v_mov_b16_e64 v146.l, v150.l
	v_cvt_pk_bf16_f32 v20, v20, s0
	v_cvt_pk_bf16_f32 v10, v10, s0
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_cvt_pk_bf16_f32 v11, v11, s0
	buffer_store_b16 v146, v183, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v146.l, v147.l
	v_mov_b16_e64 v147.l, v148.l
	v_lshl_or_b32 v148, v149, 2, 0x1c0
	v_cvt_pk_bf16_f32 v3, v3, s0
	v_cvt_pk_bf16_f32 v4, v4, s0
	buffer_store_b16 v146, v182, s[24:27], null offen offset:384
	s_wait_xcnt 0x0
	v_or_b32_e32 v146, v178, v0
	buffer_store_b16 v147, v182, s[0:3], null offen offset:384
	buffer_store_b16 v138, v148, s[24:27], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v138, v139, s0
	v_cvt_pk_bf16_f32 v5, v5, s0
	v_cvt_pk_bf16_f32 v7, v7, s0
	v_lshl_or_b32 v139, v146, 2, 0x1c0
	v_or_b32_e32 v146, v176, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v146, v146, 2, 0x1c0
	buffer_store_b16 v130, v148, s[0:3], null offen
	buffer_store_b16 v138, v139, s[24:27], null offen
	buffer_store_b16 v131, v139, s[0:3], null offen
	buffer_store_b16 v140, v146, s[24:27], null offen
	buffer_store_b16 v132, v146, s[0:3], null offen
	s_wait_xcnt 0x4
	v_or_b32_e32 v130, v173, v0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v132, v133, s0
	v_or_b32_e32 v133, v172, v0
	v_cvt_pk_bf16_f32 v131, v141, s0
	v_cvt_pk_bf16_f32 v138, v142, s0
	v_lshl_or_b32 v130, v130, 2, 0x1c0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v133, v133, 2, 0x1c0
	buffer_store_b16 v131, v130, s[24:27], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v131, v175, v0
	buffer_store_b16 v132, v130, s[0:3], null offen
	buffer_store_b16 v138, v133, s[24:27], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v130.l, v134.l
	v_cvt_pk_bf16_f32 v134, v135, s0
	v_or_b32_e32 v135, v184, v0
	v_cvt_pk_bf16_f32 v132, v143, s0
	v_lshl_or_b32 v131, v131, 2, 0x1c0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v138, v144, s0
	v_lshl_or_b32 v135, v135, 2, 0x1c0
	buffer_store_b16 v130, v133, s[0:3], null offen
	buffer_store_b16 v132, v131, s[24:27], null offen
	buffer_store_b16 v134, v131, s[0:3], null offen
	buffer_store_b16 v138, v135, s[24:27], null offen
	buffer_store_b16 v136, v135, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v130, 1, v142 /*v910*/
	v_add_lshl_u32 v131, v170, s33, 7
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v132, v195, v0
	s_set_vgpr_msb 12
	v_mul_lo_u32 v136, s20, v134 /*v902*/
	v_cvt_pk_bf16_f32 v133, v145, s0
	v_mul_lo_u32 v130, v130, s20
	v_or_b32_e32 v135, v131, v141 /*v909*/
	v_lshl_or_b32 v132, v132, 2, 0x1c0
	v_cvt_pk_bf16_f32 v134, v137, s0
	buffer_store_b16 v133, v132, s[24:27], null offen
	buffer_store_b16 v134, v132, s[0:3], null offen
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v135, 2, v135
	v_add_lshl_u32 v130, s4, v130, 7
	s_set_vgpr_msb 12
	v_mul_lo_u32 v133, s20, v133 /*v901*/
	v_mul_lo_u32 v134, s20, v132 /*v900*/
	buffer_store_b16 v122, v135, s[24:27], null offen
	v_or_b32_e32 v132, v130, v141 /*v909*/
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v122, v123, s0
	v_add_lshl_u32 v123, v136, s4, 7
	buffer_store_b16 v114, v135, s[0:3], null offen
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v114, 2, v132
	s_set_vgpr_msb 12
	v_or_b32_e32 v132, v123, v141 /*v909*/
	s_set_vgpr_msb 0xc00
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v132, 2, v132
	buffer_store_b16 v122, v114, s[24:27], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v122, s4, v133, 7
	buffer_store_b16 v115, v114, s[0:3], null offen
	buffer_store_b16 v124, v132, s[24:27], null offen
	s_wait_xcnt 0x1
	v_add_lshl_u32 v115, s4, v134, 7
	s_set_vgpr_msb 12
	v_mul_lo_u32 v133, s20, v131 /*v899*/
	buffer_store_b16 v116, v132, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v116, v122, v141 /*v909*/
	v_cvt_pk_bf16_f32 v124, v125, s0
	v_or_b32_e32 v125, v115, v141 /*v909*/
	v_mul_lo_u32 v134, s20, v130 /*v898*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v116, 2, v116
	v_add_lshl_u32 v133, s4, v133, 7
	v_lshlrev_b32_e32 v125, 2, v125
	buffer_store_b16 v124, v116, s[24:27], null offen
	buffer_store_b16 v117, v116, s[0:3], null offen
	buffer_store_b16 v126, v125, s[24:27], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e32 v117.l, v118.l
	s_set_vgpr_msb 12
	v_or_b32_e32 v118, v133, v141 /*v909*/
	v_add_lshl_u32 v126, v134, s4, 7
	v_cvt_pk_bf16_f32 v124, v127, s0
	v_cvt_pk_bf16_f32 v127, v128, s0
	buffer_store_b16 v117, v125, s[0:3], null offen
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v118, 2, v118
	v_mov_b16_e32 v117.l, v119.l
	s_set_vgpr_msb 12
	v_or_b32_e32 v119, v126, v141 /*v909*/
	buffer_store_b16 v124, v118, s[24:27], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v124, v1, v141 /*v909*/
	buffer_store_b16 v117, v118, s[0:3], null offen
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v117, 2, v119
	v_mov_b16_e32 v119.l, v127.l
	v_cvt_pk_bf16_f32 v127, v129, s0
	v_lshlrev_b32_e32 v124, 2, v124
	buffer_store_b16 v119, v117, s[24:27], null offen
	buffer_store_b16 v120, v117, s[0:3], null offen
	buffer_store_b16 v127, v124, s[24:27], null offen
	buffer_store_b16 v121, v124, s[0:3], null offen
	buffer_store_b16 v106, v135, s[24:27], null offen offset:64
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v106, v107, s0
	v_cvt_pk_bf16_f32 v107, v108, s0
	buffer_store_b16 v98, v135, s[0:3], null offen offset:64
	buffer_store_b16 v106, v114, s[24:27], null offen offset:64
	buffer_store_b16 v99, v114, s[0:3], null offen offset:64
	buffer_store_b16 v107, v132, s[24:27], null offen offset:64
	buffer_store_b16 v100, v132, s[0:3], null offen offset:64
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v98, v109, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v99, v101, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v100, v110, s0
	v_cvt_pk_bf16_f32 v101, v102, s0
	v_cvt_pk_bf16_f32 v102, v111, s0
	buffer_store_b16 v98, v116, s[24:27], null offen offset:64
	buffer_store_b16 v99, v116, s[0:3], null offen offset:64
	buffer_store_b16 v100, v125, s[24:27], null offen offset:64
	buffer_store_b16 v101, v125, s[0:3], null offen offset:64
	buffer_store_b16 v102, v118, s[24:27], null offen offset:64
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v98, v103, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v99, v112, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v100, v104, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v101, v113, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v102, v105, s0
	buffer_store_b16 v98, v118, s[0:3], null offen offset:64
	buffer_store_b16 v99, v117, s[24:27], null offen offset:64
	buffer_store_b16 v100, v117, s[0:3], null offen offset:64
	buffer_store_b16 v101, v124, s[24:27], null offen offset:64
	buffer_store_b16 v102, v124, s[0:3], null offen offset:64
	buffer_store_b16 v90, v135, s[24:27], null offen offset:128
	buffer_store_b16 v82, v135, s[0:3], null offen offset:128
	buffer_store_b16 v91, v114, s[24:27], null offen offset:128
	buffer_store_b16 v83, v114, s[0:3], null offen offset:128
	buffer_store_b16 v92, v132, s[24:27], null offen offset:128
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v82, v84, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v83, v93, s0
	v_cvt_pk_bf16_f32 v84, v85, s0
	v_cvt_pk_bf16_f32 v85, v94, s0
	buffer_store_b16 v82, v132, s[0:3], null offen offset:128
	buffer_store_b16 v83, v116, s[24:27], null offen offset:128
	buffer_store_b16 v84, v116, s[0:3], null offen offset:128
	buffer_store_b16 v85, v125, s[24:27], null offen offset:128
	buffer_store_b16 v86, v125, s[0:3], null offen offset:128
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v82, v95, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v83, v87, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v84, v96, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v85, v88, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v86, v97, s0
	buffer_store_b16 v82, v118, s[24:27], null offen offset:128
	buffer_store_b16 v83, v118, s[0:3], null offen offset:128
	buffer_store_b16 v84, v117, s[24:27], null offen offset:128
	buffer_store_b16 v85, v117, s[0:3], null offen offset:128
	buffer_store_b16 v86, v124, s[24:27], null offen offset:128
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v82, v89, s0
	buffer_store_b16 v82, v124, s[0:3], null offen offset:128
	buffer_store_b16 v74, v135, s[24:27], null offen offset:192
	buffer_store_b16 v66, v135, s[0:3], null offen offset:192
	buffer_store_b16 v75, v114, s[24:27], null offen offset:192
	buffer_store_b16 v67, v114, s[0:3], null offen offset:192
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v66, v76, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v67, v68, s0
	v_cvt_pk_bf16_f32 v68, v77, s0
	v_cvt_pk_bf16_f32 v74, v78, s0
	buffer_store_b16 v66, v132, s[24:27], null offen offset:192
	buffer_store_b16 v67, v132, s[0:3], null offen offset:192
	buffer_store_b16 v68, v116, s[24:27], null offen offset:192
	buffer_store_b16 v69, v116, s[0:3], null offen offset:192
	buffer_store_b16 v74, v125, s[24:27], null offen offset:192
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v66, v70, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v67, v79, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v68, v71, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v69, v80, s0
	v_cvt_pk_bf16_f32 v70, v72, s0
	buffer_store_b16 v66, v125, s[0:3], null offen offset:192
	buffer_store_b16 v67, v118, s[24:27], null offen offset:192
	buffer_store_b16 v68, v118, s[0:3], null offen offset:192
	buffer_store_b16 v69, v117, s[24:27], null offen offset:192
	buffer_store_b16 v70, v117, s[0:3], null offen offset:192
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v66, v81, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v67, v73, s0
	buffer_store_b16 v66, v124, s[24:27], null offen offset:192
	buffer_store_b16 v67, v124, s[0:3], null offen offset:192
	buffer_store_b16 v58, v135, s[24:27], null offen offset:256
	buffer_store_b16 v50, v135, s[0:3], null offen offset:256
	buffer_store_b16 v59, v114, s[24:27], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v50, v51, s0
	v_cvt_pk_bf16_f32 v51, v60, s0
	v_cvt_pk_bf16_f32 v58, v61, s0
	buffer_store_b16 v50, v114, s[0:3], null offen offset:256
	buffer_store_b16 v51, v132, s[24:27], null offen offset:256
	buffer_store_b16 v52, v132, s[0:3], null offen offset:256
	buffer_store_b16 v58, v116, s[24:27], null offen offset:256
	buffer_store_b16 v53, v116, s[0:3], null offen offset:256
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v50, v62, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v51, v54, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v52, v63, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v53, v55, s0
	v_cvt_pk_bf16_f32 v54, v64, s0
	buffer_store_b16 v50, v125, s[24:27], null offen offset:256
	buffer_store_b16 v51, v125, s[0:3], null offen offset:256
	buffer_store_b16 v52, v118, s[24:27], null offen offset:256
	buffer_store_b16 v53, v118, s[0:3], null offen offset:256
	buffer_store_b16 v54, v117, s[24:27], null offen offset:256
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v50, v56, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v51, v65, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v52, v57, s0
	buffer_store_b16 v50, v117, s[0:3], null offen offset:256
	buffer_store_b16 v51, v124, s[24:27], null offen offset:256
	buffer_store_b16 v52, v124, s[0:3], null offen offset:256
	buffer_store_b16 v42, v135, s[24:27], null offen offset:320
	buffer_store_b16 v34, v135, s[0:3], null offen offset:320
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v34, v43, s0
	v_cvt_pk_bf16_f32 v42, v44, s0
	v_cvt_pk_bf16_f32 v43, v45, s0
	buffer_store_b16 v34, v114, s[24:27], null offen offset:320
	buffer_store_b16 v35, v114, s[0:3], null offen offset:320
	buffer_store_b16 v42, v132, s[24:27], null offen offset:320
	buffer_store_b16 v36, v132, s[0:3], null offen offset:320
	buffer_store_b16 v43, v116, s[24:27], null offen offset:320
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v34, v37, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v35, v46, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v36, v38, s0
	v_cvt_pk_bf16_f32 v37, v47, s0
	v_cvt_pk_bf16_f32 v38, v39, s0
	buffer_store_b16 v34, v116, s[0:3], null offen offset:320
	buffer_store_b16 v35, v125, s[24:27], null offen offset:320
	buffer_store_b16 v36, v125, s[0:3], null offen offset:320
	buffer_store_b16 v37, v118, s[24:27], null offen offset:320
	buffer_store_b16 v38, v118, s[0:3], null offen offset:320
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v34, v48, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v35, v40, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v36, v49, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v37, v41, s0
	buffer_store_b16 v34, v117, s[24:27], null offen offset:320
	buffer_store_b16 v35, v117, s[0:3], null offen offset:320
	buffer_store_b16 v36, v124, s[24:27], null offen offset:320
	buffer_store_b16 v37, v124, s[0:3], null offen offset:320
	buffer_store_b16 v26, v135, s[24:27], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v26, v27, s0
	v_cvt_pk_bf16_f32 v27, v28, s0
	buffer_store_b16 v18, v135, s[0:3], null offen offset:384
	buffer_store_b16 v26, v114, s[24:27], null offen offset:384
	buffer_store_b16 v19, v114, s[0:3], null offen offset:384
	buffer_store_b16 v27, v132, s[24:27], null offen offset:384
	buffer_store_b16 v20, v132, s[0:3], null offen offset:384
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v18, v29, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v19, v21, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v20, v30, s0
	v_cvt_pk_bf16_f32 v21, v22, s0
	v_cvt_pk_bf16_f32 v22, v31, s0
	buffer_store_b16 v18, v116, s[24:27], null offen offset:384
	buffer_store_b16 v19, v116, s[0:3], null offen offset:384
	buffer_store_b16 v20, v125, s[24:27], null offen offset:384
	buffer_store_b16 v21, v125, s[0:3], null offen offset:384
	buffer_store_b16 v22, v118, s[24:27], null offen offset:384
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v18, v23, s0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v19, v32, s0
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v20, v24, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v21, v33, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v22, v25, s0
	buffer_store_b16 v18, v118, s[0:3], null offen offset:384
	buffer_store_b16 v19, v117, s[24:27], null offen offset:384
	buffer_store_b16 v20, v117, s[0:3], null offen offset:384
	buffer_store_b16 v21, v124, s[24:27], null offen offset:384
	buffer_store_b16 v22, v124, s[0:3], null offen offset:384
	s_wait_xcnt 0x4
	v_or_b32_e32 v18, v131, v0
	s_wait_xcnt 0x3
	v_or_b32_e32 v19, v130, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshl_or_b32 v18, v18, 2, 0x1c0
	v_lshl_or_b32 v19, v19, 2, 0x1c0
	buffer_store_b16 v10, v18, s[24:27], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v10, v123, v0
	buffer_store_b16 v2, v18, s[0:3], null offen
	buffer_store_b16 v11, v19, s[24:27], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, v122, v0
	v_mov_b16_e32 v2.l, v3.l
	v_cvt_pk_bf16_f32 v3, v12, s0
	v_lshl_or_b32 v10, v10, 2, 0x1c0
	v_cvt_pk_bf16_f32 v12, v13, s0
	v_lshl_or_b32 v11, v11, 2, 0x1c0
	buffer_store_b16 v2, v19, s[0:3], null offen
	buffer_store_b16 v3, v10, s[24:27], null offen
	buffer_store_b16 v4, v10, s[0:3], null offen
	buffer_store_b16 v12, v11, s[24:27], null offen
	buffer_store_b16 v5, v11, s[0:3], null offen
	s_wait_xcnt 0x4
	v_or_b32_e32 v2, v115, v0
	s_wait_xcnt 0x3
	v_cvt_pk_bf16_f32 v3, v14, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v5, v133, v0
	v_cvt_pk_bf16_f32 v4, v6, s0
	v_cvt_pk_bf16_f32 v6, v15, s0
	v_lshl_or_b32 v2, v2, 2, 0x1c0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v5, v5, 2, 0x1c0
	buffer_store_b16 v3, v2, s[24:27], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v3, v126, v0
	v_or_b32_e32 v0, v1, v0
	buffer_store_b16 v4, v2, s[0:3], null offen
	buffer_store_b16 v6, v5, s[24:27], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e32 v2.l, v7.l
	v_cvt_pk_bf16_f32 v4, v16, s0
	v_lshl_or_b32 v3, v3, 2, 0x1c0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v6, v8, s0
	v_cvt_pk_bf16_f32 v1, v17, s0
	v_lshl_or_b32 v0, v0, 2, 0x1c0
	v_cvt_pk_bf16_f32 v7, v9, s0
	buffer_store_b16 v2, v5, s[0:3], null offen
	buffer_store_b16 v4, v3, s[24:27], null offen
	buffer_store_b16 v6, v3, s[0:3], null offen
	buffer_store_b16 v1, v0, s[24:27], null offen
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
		.amdhsa_next_free_vgpr 952
		.amdhsa_next_free_sgpr 62
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

	.set .Lk_dkdv_0.num_vgpr, 952
	.set .Lk_dkdv_0.num_agpr, 0
	.set .Lk_dkdv_0.numbered_sgpr, 62
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
    .sgpr_count:     64
    .sgpr_spill_count: 0
    .symbol:         k_dkdv_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     952
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
