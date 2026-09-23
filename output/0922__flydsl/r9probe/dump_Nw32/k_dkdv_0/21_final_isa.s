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
	v_dual_lshrrev_b32 v143 /*v911*/, 4, v0 :: v_dual_bitop2_b32 v141 /*v909*/, 15, v0 bitop3:0x40
	s_cselect_b32 s8, s6, s5
	s_cselect_b32 s4, s3, s4
	s_bfe_u32 s2, ttmp6, 0x4000c
	s_and_b32 s3, ttmp6, 15
	s_add_co_i32 s2, s2, 1
	s_wait_kmcnt 0x0
	s_mul_i32 s11, s38, s40
	s_mul_i32 s2, ttmp9, s2
	s_mul_i32 s68, s37, s8
	s_add_co_i32 s5, s3, s2
	s_cmp_eq_u32 s7, 0
	s_load_b64 s[2:3], s[0:1], 0x190 nv
	s_cselect_b32 s66, ttmp9, s5
	s_lshr_b32 s5, s42, 31
	s_lshl_b32 s9, s4, 5
	s_add_co_i32 s5, s42, s5
	s_set_vgpr_msb 0xc00c
	v_or_b32_e32 v1, s9, v141 /*v909*/
	s_and_b32 s4, s5, -2
	s_ashr_i32 s5, s5, 1
	s_cmp_lg_u32 s42, s4
	s_set_vgpr_msb 0xcc0
	v_or_b32_e32 v144 /*v912*/, 16, v0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s42, 0
	s_mul_i32 s42, s38, s8
	s_cselect_b32 s6, -1, 0
	s_set_vgpr_msb 0xc082
	v_mov_b32_e32 v2 /*v514*/, 0
	s_and_b32 s4, s6, s4
	s_sub_co_ci_u32 s6, s5, 0
	s_sub_co_i32 s7, s9, s43
	s_mov_b32 s38, s36
	s_max_i32 s7, s7, 0
	s_wait_kmcnt 0x0
	s_mul_i32 s11, s11, s3
	s_lshr_b32 s7, s7, 5
	s_cmp_lg_u32 s2, 0
	v_dual_mov_b32 v3 /*v515*/, v2 /*v514*/ :: v_dual_mov_b32 v4 /*v516*/, v2 /*v514*/
	s_cselect_b32 s10, -1, 0
	v_dual_mov_b32 v5 /*v517*/, v2 /*v514*/ :: v_dual_mov_b32 v6 /*v518*/, v2 /*v514*/
	s_and_b32 s10, s10, exec_lo
	s_cselect_b32 s73, s7, 0
	s_cmp_lg_u32 s4, 0
	v_dual_mov_b32 v7 /*v519*/, v2 /*v514*/ :: v_dual_mov_b32 v8 /*v520*/, v2 /*v514*/
	s_sub_co_ci_u32 s74, s5, s73
	s_or_b32 s4, s9, 31
	v_mov_b32_e32 v9 /*v521*/, v2 /*v514*/
	s_sub_co_i32 s4, s4, s43
	v_mov_b64_e32 v[14:15] /*v[526:527]*/, v[6:7] /*v[518:519]*/
	s_add_co_i32 s5, s4, 31
	v_mov_b64_e32 v[12:13] /*v[524:525]*/, v[4:5] /*v[516:517]*/
	s_ashr_i32 s7, s5, 31
	v_mov_b64_e32 v[16:17] /*v[528:529]*/, v[8:9] /*v[520:521]*/
	s_lshr_b32 s7, s7, 27
	v_mov_b64_e32 v[10:11] /*v[522:523]*/, v[2:3] /*v[514:515]*/
	s_add_co_i32 s7, s5, s7
	v_mov_b64_e32 v[24:25] /*v[536:537]*/, v[8:9] /*v[520:521]*/
	s_and_b32 s10, s7, 0xffffffe0
	s_ashr_i32 s7, s7, 5
	s_cmp_lg_u32 s5, s10
	v_mov_b64_e32 v[22:23] /*v[534:535]*/, v[6:7] /*v[518:519]*/
	s_cselect_b32 s10, -1, 0
	s_cmp_lt_i32 s5, 0
	v_mov_b64_e32 v[20:21] /*v[532:533]*/, v[4:5] /*v[516:517]*/
	s_cselect_b32 s5, -1, 0
	v_mov_b64_e32 v[18:19] /*v[530:531]*/, v[2:3] /*v[514:515]*/
	s_and_b32 s5, s5, s10
	s_sub_co_ci_u32 s5, s7, 0
	s_cmp_gt_i32 s4, -1
	v_mov_b64_e32 v[40:41] /*v[552:553]*/, v[8:9] /*v[520:521]*/
	s_cselect_b32 s7, s5, 0
	s_clause 0x3
	s_load_b64 s[48:49], s[0:1], 0x0 nv
	s_load_b64 s[44:45], s[0:1], 0x30 nv
	s_load_b64 s[4:5], s[0:1], 0x60 nv
	s_load_b64 s[52:53], s[0:1], 0x90 nv
	s_min_i32 s6, s7, s6
	s_mul_i32 s7, s39, s37
	s_sub_co_i32 s6, s6, s73
	v_mov_b64_e32 v[38:39] /*v[550:551]*/, v[6:7] /*v[518:519]*/
	s_max_i32 s6, s6, 0
	v_mov_b64_e32 v[36:37] /*v[548:549]*/, v[4:5] /*v[516:517]*/
	s_min_i32 s6, s6, s74
	s_cmp_lg_u32 s2, 0
	v_mov_b64_e32 v[34:35] /*v[546:547]*/, v[2:3] /*v[514:515]*/
	s_cselect_b32 s76, -1, 0
	v_mov_b64_e32 v[56:57] /*v[568:569]*/, v[8:9] /*v[520:521]*/
	s_and_b32 s2, s76, exec_lo
	s_cselect_b32 s75, s6, 0
	s_lshl_b32 s12, s40, 4
	s_mul_i32 s2, s7, s3
	s_mul_i32 s6, s12, s42
	s_or_b32 s10, s9, 16
	s_lshl4_add_u32 s7, s66, s6
	s_set_vgpr_msb 0x820c
	v_or_b32_e32 v2, s10, v141 /*v909*/
	v_mad_u32 v1, v1, s12, s7
	s_lshl_b32 s34, s11, 8
	s_lshl_b32 s3, s73, 5
	s_lshl_b32 s67, s39, 4
	s_ashr_i32 s35, s34, 31
	v_mad_u32 v2, v2, s12, s7
	v_or_b32_e32 v3, s3, v141 /*v909*/
	s_lshr_b64 s[46:47], s[34:35], 7
	s_mul_i32 s35, s41, s66
	s_mul_i32 s68, s68, s67
	v_or_b32_e32 v1, v1, v143 /*v911*/
	s_lshl4_add_u32 s11, s35, s68
	s_lshl_b32 s6, s2, 8
	v_mad_u32 v3, v3, s67, s11
	v_or_b32_e32 v2, v2, v143 /*v911*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v1, 4, v1
	s_ashr_i32 s7, s6, 31
	s_mov_b32 s12, 0
	s_lshr_b64 s[50:51], s[6:7], 7
	s_mov_b32 s6, s46
	s_mov_b32 s7, s47
	v_lshlrev_b32_e32 v2, 4, v2
	s_wait_kmcnt 0x0
	s_clause 0xb
	buffer_load_b128 v[146:149], v1, s[44:47], null offen
	buffer_load_b128 v[150:153], v1, s[44:47], null offen offset:32
	buffer_load_b128 v[154:157], v1, s[44:47], null offen offset:64
	buffer_load_b128 v[158:161], v1, s[44:47], null offen offset:96
	buffer_load_b128 v[162:165], v1, s[44:47], null offen offset:128
	buffer_load_b128 v[166:169], v1, s[44:47], null offen offset:160
	buffer_load_b128 v[170:173], v1, s[44:47], null offen offset:192
	buffer_load_b128 v[174:177], v1, s[44:47], null offen offset:224
	buffer_load_b128 v[178:181], v2, s[44:47], null offen
	buffer_load_b128 v[182:185], v2, s[44:47], null offen offset:32
	buffer_load_b128 v[186:189], v2, s[44:47], null offen offset:64
	buffer_load_b128 v[190:193], v2, s[44:47], null offen offset:96
	s_clause 0x8
	buffer_load_b128 v[242:245], v1, s[4:7], null offen
	buffer_load_b128 v[246:249], v1, s[4:7], null offen offset:32
	buffer_load_b128 v[250:253], v1, s[4:7], null offen offset:64
	buffer_load_b128 v[254:257], v1, s[4:7], null offen offset:96
	s_set_vgpr_msb 64
	buffer_load_b128 v[2:5] /*v[258:261]*/, v1, s[4:7], null offen offset:128
	buffer_load_b128 v[6:9] /*v[262:265]*/, v1, s[4:7], null offen offset:160
	buffer_load_b128 v[10:13] /*v[266:269]*/, v1, s[4:7], null offen offset:192
	buffer_load_b128 v[14:17] /*v[270:273]*/, v1, s[4:7], null offen offset:224
	s_set_vgpr_msb 0x400c
	v_or_b32_e32 v1, v3, v143 /*v911*/
	v_or_b32_e32 v3, s3, v144 /*v912*/
	s_clause 0x3
	buffer_load_b128 v[226:229], v2, s[44:47], null offen offset:128
	buffer_load_b128 v[230:233], v2, s[44:47], null offen offset:160
	buffer_load_b128 v[234:237], v2, s[44:47], null offen offset:192
	buffer_load_b128 v[238:241], v2, s[44:47], null offen offset:224
	s_set_vgpr_msb 0xc40
	s_clause 0x7
	buffer_load_b128 v[42:45] /*v[298:301]*/, v2, s[4:7], null offen
	buffer_load_b128 v[46:49] /*v[302:305]*/, v2, s[4:7], null offen offset:32
	buffer_load_b128 v[50:53] /*v[306:309]*/, v2, s[4:7], null offen offset:64
	buffer_load_b128 v[54:57] /*v[310:313]*/, v2, s[4:7], null offen offset:96
	buffer_load_b128 v[58:61] /*v[314:317]*/, v2, s[4:7], null offen offset:128
	buffer_load_b128 v[62:65] /*v[318:321]*/, v2, s[4:7], null offen offset:160
	buffer_load_b128 v[66:69] /*v[322:325]*/, v2, s[4:7], null offen offset:192
	buffer_load_b128 v[70:73] /*v[326:329]*/, v2, s[4:7], null offen offset:224
	s_set_vgpr_msb 0x400c
	v_mad_u32 v2, v3, s67, s11
	s_mov_b32 s54, s50
	s_mov_b32 s55, s51
	s_clause 0x1
	s_load_b64 s[4:5], s[0:1], 0xc0 nv
	s_load_b64 s[6:7], s[0:1], 0xe8 nv
	s_lshl_b32 s13, s2, 27
	s_abs_i32 s70, s41
	s_lshl_b32 s14, s2, 2
	s_mul_i32 s2, s39, s8
	v_or_b32_e32 v2, v2, v143 /*v911*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v1, 4, v1
	s_add_co_i32 s69, s35, s2
	s_movk_i32 s2, 0x1400
	s_mul_i32 s3, s75, s41
	v_lshlrev_b32_e32 v2, 4, v2
	s_set_vgpr_msb 0xc0
	s_clause 0x3
	buffer_load_b128 v[50:53] /*v[818:821]*/, v1, s[52:55], null offen offset:128
	buffer_load_b128 v[54:57] /*v[822:825]*/, v1, s[52:55], null offen offset:160
	buffer_load_b128 v[58:61] /*v[826:829]*/, v1, s[52:55], null offen offset:192
	buffer_load_b128 v[62:65] /*v[830:833]*/, v1, s[52:55], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[66:69] /*v[834:837]*/, v2, s[48:51], null offen
	buffer_load_b128 v[70:73] /*v[838:841]*/, v2, s[48:51], null offen offset:32
	buffer_load_b128 v[74:77] /*v[842:845]*/, v2, s[48:51], null offen offset:64
	buffer_load_b128 v[78:81] /*v[846:849]*/, v2, s[48:51], null offen offset:96
	buffer_load_b128 v[82:85] /*v[850:853]*/, v2, s[48:51], null offen offset:128
	buffer_load_b128 v[86:89] /*v[854:857]*/, v2, s[48:51], null offen offset:160
	buffer_load_b128 v[90:93] /*v[858:861]*/, v2, s[48:51], null offen offset:192
	buffer_load_b128 v[94:97] /*v[862:865]*/, v2, s[48:51], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[98:101] /*v[866:869]*/, v2, s[52:55], null offen
	buffer_load_b128 v[102:105] /*v[870:873]*/, v2, s[52:55], null offen offset:32
	buffer_load_b128 v[106:109] /*v[874:877]*/, v2, s[52:55], null offen offset:64
	buffer_load_b128 v[110:113] /*v[878:881]*/, v2, s[52:55], null offen offset:96
	buffer_load_b128 v[114:117] /*v[882:885]*/, v2, s[52:55], null offen offset:128
	buffer_load_b128 v[118:121] /*v[886:889]*/, v2, s[52:55], null offen offset:160
	buffer_load_b128 v[122:125] /*v[890:893]*/, v2, s[52:55], null offen offset:192
	buffer_load_b128 v[126:129] /*v[894:897]*/, v2, s[52:55], null offen offset:224
	s_clause 0x7
	buffer_load_b128 v[2:5] /*v[770:773]*/, v1, s[48:51], null offen
	buffer_load_b128 v[6:9] /*v[774:777]*/, v1, s[48:51], null offen offset:32
	buffer_load_b128 v[10:13] /*v[778:781]*/, v1, s[48:51], null offen offset:64
	buffer_load_b128 v[14:17] /*v[782:785]*/, v1, s[48:51], null offen offset:96
	buffer_load_b128 v[18:21] /*v[786:789]*/, v1, s[48:51], null offen offset:128
	buffer_load_b128 v[22:25] /*v[790:793]*/, v1, s[48:51], null offen offset:160
	buffer_load_b128 v[26:29] /*v[794:797]*/, v1, s[48:51], null offen offset:192
	buffer_load_b128 v[30:33] /*v[798:801]*/, v1, s[48:51], null offen offset:224
	s_clause 0x3
	buffer_load_b128 v[34:37] /*v[802:805]*/, v1, s[52:55], null offen
	buffer_load_b128 v[38:41] /*v[806:809]*/, v1, s[52:55], null offen offset:32
	buffer_load_b128 v[42:45] /*v[810:813]*/, v1, s[52:55], null offen offset:64
	buffer_load_b128 v[46:49] /*v[814:817]*/, v1, s[52:55], null offen offset:96
	s_set_vgpr_msb 0xc00c
	v_lshlrev_b32_e32 v2, 3, v143 /*v911*/
	s_set_vgpr_msb 0xc00
	v_and_b32_e32 v1, 16, v0
	s_max_i32 s44, s3, 0
	s_sub_co_i32 s3, 0, s70
	s_set_vgpr_msb 0x82
	v_mov_b64_e32 v[54:55] /*v[566:567]*/, v[6:7] /*v[518:519]*/
	s_set_vgpr_msb 0x8200
	v_and_or_b32 v3, v0, 7, v2
	s_wait_kmcnt 0x0
	s_or_b64 s[56:57], s[4:5], s[12:13]
	s_cvt_f32_u32 s5, s70
	s_set_vgpr_msb 0xc0
	v_or_b32_e32 v145 /*v913*/, s9, v2
	v_or_b32_e32 v142 /*v910*/, s10, v2
	v_mad_u32_u24 v152 /*v920*/, 0x110, v3, s2
	v_s_rcp_f32 s2, s5
	s_set_vgpr_msb 0xc000
	v_lshlrev_b32_e32 v2, 1, v0
	s_set_vgpr_msb 0xcc
	v_mad_u32_u24 v146 /*v914*/, 0x110, v141 /*v909*/, v1
	v_mad_u32_u24 v148 /*v916*/, 0x110, v144 /*v912*/, v1
	s_movk_i32 s4, 0x3600
	s_set_vgpr_msb 0xcc82
	v_mov_b64_e32 v[52:53] /*v[564:565]*/, v[4:5] /*v[516:517]*/
	s_set_vgpr_msb 0x82c0
	v_and_b32_e32 v161 /*v929*/, 48, v2
	v_and_b32_e32 v150 /*v918*/, 16, v2
	s_mul_f32 s2, s2, 0x4f7ffffe
	v_mad_u32_u24 v153 /*v921*/, 0x110, v3, s4
	s_movk_i32 s4, 0xa00
	s_set_vgpr_msb 0xc082
	v_mov_b64_e32 v[50:51] /*v[562:563]*/, v[2:3] /*v[514:515]*/
	s_cvt_u32_f32 s2, s2
	v_mov_b64_e32 v[72:73] /*v[584:585]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[70:71] /*v[582:583]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[68:69] /*v[580:581]*/, v[4:5] /*v[516:517]*/
	s_mul_i32 s3, s3, s2
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
	s_set_vgpr_msb 0x82fc
	v_or_b32_e32 v139 /*v907*/, 3, v145 /*v913*/
	v_or_b32_e32 v140 /*v908*/, 2, v145 /*v913*/
	v_or_b32_e32 v137 /*v905*/, 5, v145 /*v913*/
	v_or_b32_e32 v138 /*v906*/, 4, v145 /*v913*/
	v_or_b32_e32 v135 /*v903*/, 7, v145 /*v913*/
	v_or_b32_e32 v136 /*v904*/, 6, v145 /*v913*/
	v_mad_i32_i24 v147 /*v915*/, 0xffffff40, v141 /*v909*/, v146 /*v914*/
	v_or_b32_e32 v133 /*v901*/, 3, v142 /*v910*/
	v_or_b32_e32 v134 /*v902*/, 2, v142 /*v910*/
	v_or_b32_e32 v131 /*v899*/, 5, v142 /*v910*/
	v_or_b32_e32 v132 /*v900*/, 4, v142 /*v910*/
	s_set_vgpr_msb 0xfc0c
	v_or_b32_e32 v1, 7, v142 /*v910*/
	s_set_vgpr_msb 0xcfc
	v_or_b32_e32 v130 /*v898*/, 6, v142 /*v910*/
	v_mad_i32_i24 v149 /*v917*/, 0xffffff40, v144 /*v912*/, v148 /*v916*/
	v_or_b32_e32 v151 /*v919*/, 32, v161 /*v929*/
	v_or_b32_e32 v160 /*v928*/, 64, v150 /*v918*/
	v_or_b32_e32 v159 /*v927*/, 0x60, v161 /*v929*/
	v_or_b32_e32 v158 /*v926*/, 0x80, v150 /*v918*/
	v_or_b32_e32 v157 /*v925*/, 0xa0, v161 /*v929*/
	v_or_b32_e32 v156 /*v924*/, 0xc0, v150 /*v918*/
	v_or_b32_e32 v162 /*v930*/, 0xe0, v150 /*v918*/
	s_set_vgpr_msb 0xfcc0
	v_mul_u32_u24_e32 v154 /*v922*/, 0x50, v3
	v_mad_u32_u24 v155 /*v923*/, 0x50, v3, s4
	s_ashr_i32 s15, s14, 31
	s_mul_hi_u32 s3, s2, s3
	s_lshr_b64 s[58:59], s[14:15], 7
	s_or_b64 s[60:61], s[6:7], s[12:13]
	s_mov_b32 s45, s12
	s_mov_b32 s39, s36
	s_ashr_i32 s71, s41, 31
	s_add_co_i32 s72, s2, s3
	s_mov_b64 s[64:65], 0
	s_mov_b32 s36, 0x3fb8aa3b
	s_set_vgpr_msb 0xc000
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
	v_mov_b64_e32 v[194:195], v[114:115] /*v[626:627]*/
	v_mov_b64_e32 v[196:197], v[116:117] /*v[628:629]*/
	v_mov_b64_e32 v[198:199], v[118:119] /*v[630:631]*/
	v_mov_b64_e32 v[200:201], v[120:121] /*v[632:633]*/
	v_mov_b64_e32 v[210:211], v[106:107] /*v[618:619]*/
	v_mov_b64_e32 v[212:213], v[108:109] /*v[620:621]*/
	v_mov_b64_e32 v[214:215], v[110:111] /*v[622:623]*/
	v_mov_b64_e32 v[216:217], v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x242
	v_mov_b64_e32 v[18:19] /*v[274:275]*/, v[90:91] /*v[602:603]*/
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[92:93] /*v[604:605]*/
	v_mov_b64_e32 v[22:23] /*v[278:279]*/, v[94:95] /*v[606:607]*/
	v_mov_b64_e32 v[24:25] /*v[280:281]*/, v[96:97] /*v[608:609]*/
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[74:75] /*v[586:587]*/
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[76:77] /*v[588:589]*/
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[78:79] /*v[590:591]*/
	v_mov_b64_e32 v[40:41] /*v[296:297]*/, v[80:81] /*v[592:593]*/
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
	v_mov_b64_e32 v[202:203], v[82:83] /*v[594:595]*/
	v_mov_b64_e32 v[204:205], v[84:85] /*v[596:597]*/
	v_mov_b64_e32 v[206:207], v[86:87] /*v[598:599]*/
	v_mov_b64_e32 v[208:209], v[88:89] /*v[600:601]*/
	v_mov_b64_e32 v[218:219], v[66:67] /*v[578:579]*/
	v_mov_b64_e32 v[220:221], v[68:69] /*v[580:581]*/
	v_mov_b64_e32 v[222:223], v[70:71] /*v[582:583]*/
	v_mov_b64_e32 v[224:225], v[72:73] /*v[584:585]*/
	s_set_vgpr_msb 0x242
	v_mov_b64_e32 v[26:27] /*v[282:283]*/, v[50:51] /*v[562:563]*/
	v_mov_b64_e32 v[28:29] /*v[284:285]*/, v[52:53] /*v[564:565]*/
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[54:55] /*v[566:567]*/
	v_mov_b64_e32 v[32:33] /*v[288:289]*/, v[56:57] /*v[568:569]*/
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
	v_dual_mov_b32 v166 /*v422*/, v102 /*v870*/ :: v_dual_mov_b32 v167 /*v423*/, v103 /*v871*/
	v_dual_mov_b32 v168 /*v424*/, v104 /*v872*/ :: v_dual_mov_b32 v169 /*v425*/, v105 /*v873*/
	v_dual_mov_b32 v162 /*v418*/, v98 /*v866*/ :: v_dual_mov_b32 v163 /*v419*/, v99 /*v867*/
	v_dual_mov_b32 v164 /*v420*/, v100 /*v868*/ :: v_dual_mov_b32 v165 /*v421*/, v101 /*v869*/
	v_dual_mov_b32 v158 /*v414*/, v94 /*v862*/ :: v_dual_mov_b32 v159 /*v415*/, v95 /*v863*/
	v_dual_mov_b32 v160 /*v416*/, v96 /*v864*/ :: v_dual_mov_b32 v161 /*v417*/, v97 /*v865*/
	v_dual_mov_b32 v154 /*v410*/, v90 /*v858*/ :: v_dual_mov_b32 v155 /*v411*/, v91 /*v859*/
	v_dual_mov_b32 v156 /*v412*/, v92 /*v860*/ :: v_dual_mov_b32 v157 /*v413*/, v93 /*v861*/
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
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v214 /*v470*/, v46 /*v814*/ :: v_dual_mov_b32 v215 /*v471*/, v47 /*v815*/
	v_dual_mov_b32 v216 /*v472*/, v48 /*v816*/ :: v_dual_mov_b32 v217 /*v473*/, v49 /*v817*/
	v_dual_mov_b32 v210 /*v466*/, v42 /*v810*/ :: v_dual_mov_b32 v211 /*v467*/, v43 /*v811*/
	v_dual_mov_b32 v212 /*v468*/, v44 /*v812*/ :: v_dual_mov_b32 v213 /*v469*/, v45 /*v813*/
	v_dual_mov_b32 v230 /*v486*/, v38 /*v806*/ :: v_dual_mov_b32 v231 /*v487*/, v39 /*v807*/
	v_dual_mov_b32 v232 /*v488*/, v40 /*v808*/ :: v_dual_mov_b32 v233 /*v489*/, v41 /*v809*/
	v_dual_mov_b32 v226 /*v482*/, v34 /*v802*/ :: v_dual_mov_b32 v227 /*v483*/, v35 /*v803*/
	v_dual_mov_b32 v228 /*v484*/, v36 /*v804*/ :: v_dual_mov_b32 v229 /*v485*/, v37 /*v805*/
	v_dual_mov_b32 v222 /*v478*/, v30 /*v798*/ :: v_dual_mov_b32 v223 /*v479*/, v31 /*v799*/
	v_dual_mov_b32 v224 /*v480*/, v32 /*v800*/ :: v_dual_mov_b32 v225 /*v481*/, v33 /*v801*/
	v_dual_mov_b32 v218 /*v474*/, v26 /*v794*/ :: v_dual_mov_b32 v219 /*v475*/, v27 /*v795*/
	v_dual_mov_b32 v220 /*v476*/, v28 /*v796*/ :: v_dual_mov_b32 v221 /*v477*/, v29 /*v797*/
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
	s_cmp_eq_u64 s[64:65], s[44:45]
	s_mov_b32 s2, -1
	s_set_vgpr_msb 0x4300
	s_cbranch_scc1 .LBB0_1
	s_abs_i32 s3, s64
	s_ashr_i32 s2, s64, 31
	s_mul_hi_u32 s4, s3, s72
	s_xor_b32 s2, s2, s71
	s_mul_i32 s5, s4, s70
	s_add_co_i32 s6, s4, 1
	s_sub_co_i32 s3, s3, s5
	s_mov_b32 s62, s58
	s_sub_co_i32 s5, s3, s70
	s_cmp_ge_u32 s3, s70
	s_mov_b32 s63, s59
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s3, s5, s3
	s_add_co_i32 s5, s4, 1
	s_cmp_ge_u32 s3, s70
	s_set_vgpr_msb 0x84
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[146:153], v[250:257] /*v[506:513]*/, 0
	s_cselect_b32 s3, s5, s4
	v_mov_b64_e32 v[74:75] /*v[586:587]*/, s[38:39]
	s_xor_b32 s3, s3, s2
	s_set_vgpr_msb 0x8407
	ds_store_b128 v146 /*v914*/, v[226:229] /*v[482:485]*/ offset:5120
	ds_store_b128 v146 /*v914*/, v[230:233] /*v[486:489]*/ offset:5152
	ds_store_b128 v146 /*v914*/, v[250:253] /*v[506:509]*/ offset:13824
	ds_store_b128 v146 /*v914*/, v[254:257] /*v[510:513]*/ offset:13856
	s_sub_co_i32 s4, s3, s2
	ds_store_b128 v146 /*v914*/, v[210:213] /*v[466:469]*/ offset:5184
	ds_store_b128 v146 /*v914*/, v[214:217] /*v[470:473]*/ offset:5216
	ds_store_b128 v146 /*v914*/, v[242:245] /*v[498:501]*/ offset:13888
	ds_store_b128 v146 /*v914*/, v[246:249] /*v[502:505]*/ offset:13920
	ds_store_b128 v146 /*v914*/, v[202:205] /*v[458:461]*/ offset:5248
	ds_store_b128 v146 /*v914*/, v[206:209] /*v[462:465]*/ offset:5280
	ds_store_b128 v146 /*v914*/, v[234:237] /*v[490:493]*/ offset:13952
	ds_store_b128 v146 /*v914*/, v[238:241] /*v[494:497]*/ offset:13984
	s_mul_i32 s4, s4, s41
	s_set_vgpr_msb 0x7a4
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[154:161], v[242:249] /*v[498:505]*/, v[10:17] /*v[522:529]*/
	s_cmp_lg_u32 s64, s4
	s_set_vgpr_msb 0xa407
	ds_store_b128 v146 /*v914*/, v[194:197] /*v[450:453]*/ offset:5312
	ds_store_b128 v146 /*v914*/, v[198:201] /*v[454:457]*/ offset:5344
	s_cselect_b32 s4, -1, 0
	s_xor_b32 s5, s41, s64
	ds_store_b128 v146 /*v914*/, v[218:221] /*v[474:477]*/ offset:14016
	ds_store_b128 v146 /*v914*/, v[222:225] /*v[478:481]*/ offset:14048
	s_cmp_lt_i32 s5, 0
	s_set_vgpr_msb 0x78f
	v_add_nc_u32_e32 v76 /*v588*/, v152 /*v920*/, v150 /*v918*/
	s_cselect_b32 s5, -1, 0
	v_add_nc_u32_e32 v77 /*v589*/, v154 /*v922*/, v150 /*v918*/
	s_and_b32 s4, s5, s4
	s_sub_co_ci_u32 s2, s3, s2
	s_add_co_i32 s54, s64, 1
	s_mul_i32 s3, s41, s2
	s_abs_i32 s5, s54
	s_ashr_i32 s4, s54, 31
	s_mul_hi_u32 s6, s5, s72
	s_xor_b32 s4, s4, s71
	s_mul_i32 s7, s6, s70
	s_add_co_i32 s8, s6, 1
	s_sub_co_i32 s5, s5, s7
	s_set_vgpr_msb 0x8fa4
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[162:169], v[234:241] /*v[490:497]*/, v[10:17] /*v[522:529]*/
	s_sub_co_i32 s7, s5, s70
	s_cmp_ge_u32 s5, s70
	s_set_vgpr_msb 0xa48f
	v_add_nc_u32_e32 v82 /*v594*/, v153 /*v921*/, v157 /*v925*/
	s_cselect_b32 s6, s8, s6
	s_cselect_b32 s5, s7, s5
	s_add_co_i32 s7, s6, 1
	s_cmp_ge_u32 s5, s70
	s_set_vgpr_msb 0x8fa4
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[170:177], v[218:225] /*v[474:481]*/, v[10:17] /*v[522:529]*/
	s_cselect_b32 s5, s7, s6
	s_set_vgpr_msb 0xa48f
	v_add_nc_u32_e32 v98 /*v610*/, v153 /*v921*/, v156 /*v924*/
	s_xor_b32 s5, s5, s4
	v_add_nc_u32_e32 v162 /*v674*/, v153 /*v921*/, v162 /*v930*/
	s_sub_co_i32 s6, s5, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s6, s6, s41
	s_set_vgpr_msb 0x8fa4
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[146:153], v[186:193] /*v[442:449]*/, 0
	s_cmp_lg_u32 s54, s6
	s_cselect_b32 s6, -1, 0
	s_xor_b32 s7, s54, s41
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s7, -1, 0
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[154:161], v[178:185] /*v[434:441]*/, v[34:41] /*v[546:553]*/
	s_and_b32 s6, s7, s6
	s_sub_co_ci_u32 s55, s5, s4
	s_add_co_i32 s2, s2, s73
	s_add_co_i32 s4, s69, s64
	s_lshl_b32 s5, s2, 5
	s_sub_co_i32 s2, s4, s3
	s_set_vgpr_msb 0xa48c
	v_or_b32_e32 v67 /*v579*/, s5, v141 /*v909*/
	s_mul_i32 s2, s2, s37
	v_or_b32_e32 v69 /*v581*/, s5, v144 /*v912*/
	s_set_vgpr_msb 0x8c8a
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[74:75] /*v[586:587]*/, v[10:11] /*v[522:523]*/
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[74:75] /*v[586:587]*/, v[12:13] /*v[524:525]*/
	v_add_lshl_u32 v18 /*v530*/, s2, v67 /*v579*/, 2
	buffer_load_b32 v66 /*v578*/, v18 /*v530*/, s[56:59], null offen
	buffer_load_b32 v68 /*v580*/, v18 /*v530*/, s[60:63], null offen
	v_add_lshl_u32 v71 /*v583*/, s2, v69 /*v581*/, 2
	s_set_vgpr_msb 0x8a84
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[178:185], v[250:257] /*v[506:513]*/, 0
	s_set_vgpr_msb 0x848a
	v_dual_add_nc_u32 v67 /*v579*/, s43, v67 /*v579*/ :: v_dual_add_nc_u32 v69 /*v581*/, s43, v69 /*v581*/
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[74:75] /*v[586:587]*/, v[14:15] /*v[526:527]*/
	buffer_load_b32 v70 /*v582*/, v71 /*v583*/, s[56:59], null offen
	buffer_load_b32 v72 /*v584*/, v71 /*v583*/, s[60:63], null offen
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s2, v145 /*v913*/, v67 /*v579*/
	v_cmp_gt_i32_e64 s3, v139 /*v907*/, v67 /*v579*/
	s_set_vgpr_msb 0xba4
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[186:193], v[242:249] /*v[498:505]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa40b
	v_cmp_gt_i32_e64 s4, v140 /*v908*/, v67 /*v579*/
	v_cmp_gt_i32_e64 s5, v137 /*v905*/, v67 /*v579*/
	s_and_b32 s2, s76, s2
	v_cmp_gt_i32_e64 s6, v138 /*v906*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v10 /*v522*/, v10 /*v522*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s3
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s7, v135 /*v903*/, v67 /*v579*/
	s_set_vgpr_msb 0xba4
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[226:233], v[234:241] /*v[490:497]*/, v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa48a
	v_cndmask_b32_e64 v13 /*v525*/, v13 /*v525*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s4
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[74:75] /*v[586:587]*/, v[16:17] /*v[528:529]*/
	v_cndmask_b32_e64 v12 /*v524*/, v12 /*v524*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s5
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s8, v136 /*v904*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v15 /*v527*/, v15 /*v527*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[234:241], v[218:225] /*v[474:481]*/, v[18:25] /*v[530:537]*/
	s_and_b32 s2, s76, s6
	s_set_vgpr_msb 0xa40b
	v_cmp_ge_i32_e64 s9, v142 /*v910*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v14 /*v526*/, v14 /*v526*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s7
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s10, v142 /*v910*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v17 /*v529*/, v17 /*v529*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s8
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[162:169], v[170:177] /*v[426:433]*/, v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa48a
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[74:75] /*v[586:587]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s11, v133 /*v901*/, v67 /*v579*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v16 /*v528*/, v16 /*v528*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s9
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[74:75] /*v[586:587]*/, v[20:21] /*v[532:533]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s12, v134 /*v902*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v19 /*v531*/, v19 /*v531*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x8284
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[178:185], v[186:193] /*v[442:449]*/, 0
	s_and_b32 s2, s76, s10
	s_set_vgpr_msb 0x840b
	v_cmp_gt_i32_e64 s13, v131 /*v899*/, v67 /*v579*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v18 /*v530*/, v18 /*v530*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s11
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[74:75] /*v[586:587]*/, v[22:23] /*v[534:535]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s14, v132 /*v900*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v21 /*v533*/, v21 /*v533*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[170:177], v[154:161] /*v[410:417]*/, v[34:41] /*v[546:553]*/
	s_and_b32 s2, s76, s12
	s_set_vgpr_msb 0xa408
	v_cmp_gt_i32_e64 s15, v1, v67 /*v579*/
	s_set_vgpr_msb 0x88a
	v_cndmask_b32_e64 v20 /*v532*/, v20 /*v532*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s13
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[74:75] /*v[586:587]*/, v[24:25] /*v[536:537]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s16, v130 /*v898*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v23 /*v535*/, v23 /*v535*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[186:193], v[178:185] /*v[434:441]*/, v[50:57] /*v[562:569]*/
	s_and_b32 s2, s76, s14
	s_set_vgpr_msb 0xa40b
	v_cmp_ge_i32_e64 s17, v145 /*v913*/, v69 /*v581*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v22 /*v534*/, v22 /*v534*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s15
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[74:75] /*v[586:587]*/, v[34:35] /*v[546:547]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s18, v145 /*v913*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v25 /*v537*/, v25 /*v537*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[226:233], v[170:177] /*v[426:433]*/, v[50:57] /*v[562:569]*/
	s_and_b32 s2, s76, s16
	s_set_vgpr_msb 0xa40b
	v_cmp_gt_i32_e64 s19, v139 /*v907*/, v69 /*v581*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v24 /*v536*/, v24 /*v536*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s17
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[74:75] /*v[586:587]*/, v[36:37] /*v[548:549]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s20, v140 /*v908*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v35 /*v547*/, v35 /*v547*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s18
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s21, v137 /*v905*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v34 /*v546*/, v34 /*v546*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s19
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[234:241], v[154:161] /*v[410:417]*/, v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa48a
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[74:75] /*v[586:587]*/, v[38:39] /*v[550:551]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s22, v138 /*v906*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v37 /*v549*/, v37 /*v549*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s20
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s23, v135 /*v903*/, v69 /*v581*/
	s_set_vgpr_msb 0xb8a
	v_cndmask_b32_e64 v36 /*v548*/, v36 /*v548*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s21
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[74:75] /*v[586:587]*/, v[40:41] /*v[552:553]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s24, v136 /*v904*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v39 /*v551*/, v39 /*v551*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s22
	s_set_vgpr_msb 0x820b
	v_cmp_ge_i32_e64 s25, v142 /*v910*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v38 /*v550*/, v38 /*v550*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s23
	s_set_vgpr_msb 0x8284
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[242:249], v[226:233] /*v[482:489]*/, 0
	s_set_vgpr_msb 0x848a
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[74:75] /*v[586:587]*/, v[50:51] /*v[562:563]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s26, v142 /*v910*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v41 /*v553*/, v41 /*v553*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s24
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s27, v133 /*v901*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v40 /*v552*/, v40 /*v552*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s25
	s_set_vgpr_msb 0x8285
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[42:49] /*v[298:305]*/, v[226:233] /*v[482:489]*/, 0
	s_set_vgpr_msb 0x858a
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[74:75] /*v[586:587]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s28, v134 /*v902*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v51 /*v563*/, v51 /*v563*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s26
	s_set_vgpr_msb 0x820b
	v_cmp_gt_i32_e64 s29, v131 /*v899*/, v69 /*v581*/
	v_cmp_ge_i32_e32 vcc_lo, v145 /*v913*/, v67 /*v579*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v50 /*v562*/, v50 /*v562*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x8284
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[242:249], v[162:169] /*v[418:425]*/, 0
	s_and_b32 s2, s76, s27
	s_set_vgpr_msb 0x848a
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[74:75] /*v[586:587]*/, v[54:55] /*v[566:567]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s30, v132 /*v900*/, v69 /*v581*/
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v53 /*v565*/, v53 /*v565*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s28
	v_cmp_lt_i32_e64 s31, v69 /*v581*/, v1
	v_cndmask_b32_e64 v52 /*v564*/, v52 /*v564*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x8285
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[42:49] /*v[298:305]*/, v[162:169] /*v[418:425]*/, 0
	s_and_b32 s2, s76, s29
	s_set_vgpr_msb 0x858a
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[74:75] /*v[586:587]*/, v[56:57] /*v[568:569]*/
	s_set_vgpr_msb 0x8a0b
	v_cmp_gt_i32_e64 s33, v130 /*v898*/, v69 /*v581*/
	s_and_b32 s62, s76, vcc_lo
	s_set_vgpr_msb 0xb82
	v_cndmask_b32_e64 v55 /*v567*/, v55 /*v567*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s30
	v_cndmask_b32_e64 v11 /*v523*/, v11 /*v523*/, 0xff61b1e6, s62
	s_set_vgpr_msb 0x82a4
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[250:257], v[210:217] /*v[466:473]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa482
	v_cndmask_b32_e64 v54 /*v566*/, v54 /*v566*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s31
	s_mul_i32 s3, s41, s55
	v_cndmask_b32_e64 v57 /*v569*/, v57 /*v569*/, 0xff61b1e6, s2
	s_and_b32 s2, s76, s33
	s_add_co_i32 s54, s54, s35
	v_cndmask_b32_e64 v56 /*v568*/, v56 /*v568*/, 0xff61b1e6, s2
	s_set_vgpr_msb 0x82a5
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[50:57] /*v[306:313]*/, v[210:217] /*v[466:473]*/, v[26:33] /*v[538:545]*/
	s_add_co_i32 s2, s55, s73
	s_sub_co_i32 s3, s54, s3
	s_lshl_b32 s2, s2, 5
	s_lshl4_add_u32 s3, s3, s68
	s_mov_b32 s54, s50
	s_mov_b32 s55, s51
	s_add_nc_u64 s[64:65], s[64:65], 1
	s_set_vgpr_msb 0xa5a4
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[250:257], v[146:153] /*v[402:409]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa48a
	s_wait_loadcnt 0x3
	v_pk_add_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[50:57] /*v[306:313]*/, v[146:153] /*v[402:409]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa58a
	v_pk_add_f32 v[12:13] /*v[524:525]*/, v[12:13] /*v[524:525]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, v[66:67] /*v[578:579]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa5
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[2:9] /*v[258:265]*/, v[202:209] /*v[458:465]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa58a
	s_wait_loadcnt 0x1
	v_pk_add_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa5
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[58:65] /*v[314:321]*/, v[202:209] /*v[458:465]*/, v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa58a
	v_pk_add_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, v[70:71] /*v[582:583]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_mul_f32 v[10:11] /*v[522:523]*/, v[10:11] /*v[522:523]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13] /*v[524:525]*/, v[12:13] /*v[524:525]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15] /*v[526:527]*/, v[14:15] /*v[526:527]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17] /*v[528:529]*/, v[16:17] /*v[528:529]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[18:19] /*v[530:531]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[20:21] /*v[532:533]*/, s[36:37] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x8aa5
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[2:9] /*v[258:265]*/, v[138:145] /*v[394:401]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa582
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[22:23] /*v[534:535]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[24:25] /*v[536:537]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[34:35] /*v[546:547]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[36:37] /*v[548:549]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[38:39] /*v[550:551]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[40:41] /*v[552:553]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51] /*v[562:563]*/, v[50:51] /*v[562:563]*/, s[36:37] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x82a5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[58:65] /*v[314:321]*/, v[138:145] /*v[394:401]*/, v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa582
	v_pk_mul_f32 v[52:53] /*v[564:565]*/, v[52:53] /*v[564:565]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55] /*v[566:567]*/, v[54:55] /*v[566:567]*/, s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57] /*v[568:569]*/, v[56:57] /*v[568:569]*/, s[36:37] op_sel_hi:[1,0]
	v_exp_f32_e32 v10 /*v522*/, v10 /*v522*/
	v_exp_f32_e32 v11 /*v523*/, v11 /*v523*/
	v_exp_f32_e32 v12 /*v524*/, v12 /*v524*/
	v_exp_f32_e32 v13 /*v525*/, v13 /*v525*/
	s_set_vgpr_msb 0x82a5
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[10:17] /*v[266:273]*/, v[194:201] /*v[450:457]*/, v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa582
	v_exp_f32_e32 v14 /*v526*/, v14 /*v526*/
	v_exp_f32_e32 v15 /*v527*/, v15 /*v527*/
	v_exp_f32_e32 v16 /*v528*/, v16 /*v528*/
	v_exp_f32_e32 v17 /*v529*/, v17 /*v529*/
	v_exp_f32_e32 v18 /*v530*/, v18 /*v530*/
	v_exp_f32_e32 v19 /*v531*/, v19 /*v531*/
	v_exp_f32_e32 v20 /*v532*/, v20 /*v532*/
	s_set_vgpr_msb 0x82a5
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[66:73] /*v[322:329]*/, v[194:201] /*v[450:457]*/, v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa58a
	v_exp_f32_e32 v21 /*v533*/, v21 /*v533*/
	v_exp_f32_e32 v22 /*v534*/, v22 /*v534*/
	v_exp_f32_e32 v23 /*v535*/, v23 /*v535*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v25 /*v537*/, v25 /*v537*/
	v_pk_add_f32 v[2:3] /*v[514:515]*/, v[2:3] /*v[514:515]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[4:5] /*v[516:517]*/, v[4:5] /*v[516:517]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8aa5
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[10:17] /*v[266:273]*/, v[130:137] /*v[386:393]*/, v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa58a
	v_pk_add_f32 v[6:7] /*v[518:519]*/, v[6:7] /*v[518:519]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[8:9] /*v[520:521]*/, v[8:9] /*v[520:521]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[28:29] /*v[540:541]*/, v[28:29] /*v[540:541]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[30:31] /*v[542:543]*/, v[30:31] /*v[542:543]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, v[68:69] /*v[580:581]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_exp_f32_e32 v34 /*v546*/, v34 /*v546*/
	s_set_vgpr_msb 0x8aa5
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[66:73] /*v[322:329]*/, v[130:137] /*v[386:393]*/, v[58:65] /*v[570:577]*/
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
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[26:27] /*v[538:539]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[28:29] /*v[540:541]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[30:31] /*v[542:543]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[32:33] /*v[544:545]*/, v[24:25] /*v[536:537]*/
	v_cvt_pk_bf16_f32 v5 /*v517*/, v16 /*v528*/, v17 /*v529*/
	v_cvt_pk_bf16_f32 v4 /*v516*/, v14 /*v526*/, v15 /*v527*/
	v_cvt_pk_bf16_f32 v3 /*v515*/, v12 /*v524*/, v13 /*v525*/
	v_cvt_pk_bf16_f32 v2 /*v514*/, v10 /*v522*/, v11 /*v523*/
	v_cvt_pk_bf16_f32 v9 /*v521*/, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v8 /*v520*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v7 /*v519*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v6 /*v518*/, v18 /*v530*/, v19 /*v531*/
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[42:43] /*v[554:555]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[44:45] /*v[556:557]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[46:47] /*v[558:559]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[48:49] /*v[560:561]*/, v[40:41] /*v[552:553]*/
	v_cvt_pk_bf16_f32 v13 /*v525*/, v40 /*v552*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v12 /*v524*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v11 /*v523*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v10 /*v522*/, v34 /*v546*/, v35 /*v547*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[58:59] /*v[570:571]*/, v[50:51] /*v[562:563]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[60:61] /*v[572:573]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[62:63] /*v[574:575]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[64:65] /*v[576:577]*/, v[56:57] /*v[568:569]*/
	v_pk_mul_f32 v[42:43] /*v[554:555]*/, v[74:75] /*v[586:587]*/, v[66:67] /*v[578:579]*/
	v_pk_mul_f32 v[44:45] /*v[556:557]*/, v[74:75] /*v[586:587]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[46:47] /*v[558:559]*/, v[74:75] /*v[586:587]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[48:49] /*v[560:561]*/, v[74:75] /*v[586:587]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[26:27] /*v[538:539]*/, v[74:75] /*v[586:587]*/, v[26:27] /*v[538:539]*/
	v_pk_mul_f32 v[28:29] /*v[540:541]*/, v[74:75] /*v[586:587]*/, v[28:29] /*v[540:541]*/
	v_pk_mul_f32 v[30:31] /*v[542:543]*/, v[74:75] /*v[586:587]*/, v[30:31] /*v[542:543]*/
	v_pk_mul_f32 v[32:33] /*v[544:545]*/, v[74:75] /*v[586:587]*/, v[32:33] /*v[544:545]*/
	s_set_vgpr_msb 0x8a0b
	ds_store_b128 v147 /*v915*/, v[2:5] /*v[514:517]*/
	ds_store_b128 v147 /*v915*/, v[6:9] /*v[518:521]*/ offset:32
	s_set_vgpr_msb 0xb8a
	v_pk_mul_f32 v[18:19] /*v[530:531]*/, v[74:75] /*v[586:587]*/, v[18:19] /*v[530:531]*/
	v_pk_mul_f32 v[20:21] /*v[532:533]*/, v[74:75] /*v[586:587]*/, v[20:21] /*v[532:533]*/
	v_pk_mul_f32 v[22:23] /*v[534:535]*/, v[74:75] /*v[586:587]*/, v[22:23] /*v[534:535]*/
	v_pk_mul_f32 v[24:25] /*v[536:537]*/, v[74:75] /*v[586:587]*/, v[24:25] /*v[536:537]*/
	v_pk_mul_f32 v[34:35] /*v[546:547]*/, v[74:75] /*v[586:587]*/, v[34:35] /*v[546:547]*/
	v_pk_mul_f32 v[36:37] /*v[548:549]*/, v[74:75] /*v[586:587]*/, v[36:37] /*v[548:549]*/
	v_pk_mul_f32 v[38:39] /*v[550:551]*/, v[74:75] /*v[586:587]*/, v[38:39] /*v[550:551]*/
	v_pk_mul_f32 v[40:41] /*v[552:553]*/, v[74:75] /*v[586:587]*/, v[40:41] /*v[552:553]*/
	v_cvt_pk_bf16_f32 v2 /*v514*/, v42 /*v554*/, v43 /*v555*/
	v_cvt_pk_bf16_f32 v3 /*v515*/, v44 /*v556*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v4 /*v516*/, v46 /*v558*/, v47 /*v559*/
	v_cvt_pk_bf16_f32 v5 /*v517*/, v48 /*v560*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v6 /*v518*/, v26 /*v538*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v7 /*v519*/, v28 /*v540*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v8 /*v520*/, v30 /*v542*/, v31 /*v543*/
	v_cvt_pk_bf16_f32 v9 /*v521*/, v32 /*v544*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v17 /*v529*/, v56 /*v568*/, v57 /*v569*/
	v_cvt_pk_bf16_f32 v16 /*v528*/, v54 /*v566*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v15 /*v527*/, v52 /*v564*/, v53 /*v565*/
	v_cvt_pk_bf16_f32 v14 /*v526*/, v50 /*v562*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v18 /*v530*/, v18 /*v530*/, v19 /*v531*/
	v_cvt_pk_bf16_f32 v19 /*v531*/, v20 /*v532*/, v21 /*v533*/
	v_cvt_pk_bf16_f32 v20 /*v532*/, v22 /*v534*/, v23 /*v535*/
	v_cvt_pk_bf16_f32 v21 /*v533*/, v24 /*v536*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v22 /*v534*/, v34 /*v546*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v23 /*v535*/, v36 /*v548*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v24 /*v536*/, v38 /*v550*/, v39 /*v551*/
	v_cvt_pk_bf16_f32 v25 /*v537*/, v40 /*v552*/, v41 /*v553*/
	s_set_vgpr_msb 0x8a0b
	ds_store_b128 v147 /*v915*/, v[2:5] /*v[514:517]*/ offset:2560
	ds_store_b128 v147 /*v915*/, v[6:9] /*v[518:521]*/ offset:2592
	s_set_vgpr_msb 0xb07
	ds_store_b128 v148 /*v916*/, v[162:165] /*v[418:421]*/ offset:5120
	ds_store_b128 v148 /*v916*/, v[166:169] /*v[422:425]*/ offset:5152
	ds_store_b128 v148 /*v916*/, v[186:189] /*v[442:445]*/ offset:13824
	ds_store_b128 v148 /*v916*/, v[190:193] /*v[446:449]*/ offset:13856
	ds_store_b128 v148 /*v916*/, v[146:149] /*v[402:405]*/ offset:5184
	ds_store_b128 v148 /*v916*/, v[150:153] /*v[406:409]*/ offset:5216
	ds_store_b128 v148 /*v916*/, v[178:181] /*v[434:437]*/ offset:13888
	ds_store_b128 v148 /*v916*/, v[182:185] /*v[438:441]*/ offset:13920
	ds_store_b128 v148 /*v916*/, v[138:141] /*v[394:397]*/ offset:5248
	ds_store_b128 v148 /*v916*/, v[142:145] /*v[398:401]*/ offset:5280
	ds_store_b128 v148 /*v916*/, v[170:173] /*v[426:429]*/ offset:13952
	ds_store_b128 v148 /*v916*/, v[174:177] /*v[430:433]*/ offset:13984
	ds_store_b128 v148 /*v916*/, v[130:133] /*v[386:389]*/ offset:5312
	ds_store_b128 v148 /*v916*/, v[134:137] /*v[390:393]*/ offset:5344
	ds_store_b128 v148 /*v916*/, v[154:157] /*v[410:413]*/ offset:14016
	ds_store_b128 v148 /*v916*/, v[158:161] /*v[414:417]*/ offset:14048
	s_set_vgpr_msb 0x70b
	ds_store_b128 v149 /*v917*/, v[10:13] /*v[522:525]*/
	ds_store_b128 v149 /*v917*/, v[14:17] /*v[526:529]*/ offset:32
	ds_store_b128 v149 /*v917*/, v[18:21] /*v[530:533]*/ offset:2560
	ds_store_b128 v149 /*v917*/, v[22:25] /*v[534:537]*/ offset:2592
	s_set_vgpr_msb 0xb8f
	v_dual_add_nc_u32 v2 /*v514*/, v153 /*v921*/, v150 /*v918*/ :: v_dual_add_nc_u32 v10 /*v522*/, v155 /*v923*/, v150 /*v918*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[130:133] /*v[642:645]*/, v10 /*v522*/
	ds_load_tr16_b128 v[134:137] /*v[646:649]*/, v10 /*v522*/ offset:1280
	s_set_vgpr_msb 0x828f
	v_dual_add_nc_u32 v10 /*v522*/, v152 /*v920*/, v151 /*v919*/ :: v_dual_add_nc_u32 v18 /*v530*/, v153 /*v921*/, v151 /*v919*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[178:181] /*v[690:693]*/, v18 /*v530*/
	ds_load_tr16_b128 v[182:185] /*v[694:697]*/, v18 /*v530*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_dual_add_nc_u32 v18 /*v530*/, v152 /*v920*/, v160 /*v928*/ :: v_dual_add_nc_u32 v34 /*v546*/, v153 /*v921*/, v160 /*v928*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[194:197] /*v[706:709]*/, v34 /*v546*/
	ds_load_tr16_b128 v[198:201] /*v[710:713]*/, v34 /*v546*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_dual_add_nc_u32 v34 /*v546*/, v152 /*v920*/, v159 /*v927*/ :: v_dual_add_nc_u32 v50 /*v562*/, v153 /*v921*/, v159 /*v927*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[210:213] /*v[722:725]*/, v50 /*v562*/
	ds_load_tr16_b128 v[214:217] /*v[726:729]*/, v50 /*v562*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_dual_add_nc_u32 v50 /*v562*/, v152 /*v920*/, v158 /*v926*/ :: v_dual_add_nc_u32 v66 /*v578*/, v153 /*v921*/, v158 /*v926*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[226:229] /*v[738:741]*/, v66 /*v578*/
	ds_load_tr16_b128 v[230:233] /*v[742:745]*/, v66 /*v578*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_add_nc_u32_e32 v66 /*v578*/, v152 /*v920*/, v157 /*v925*/
	s_set_vgpr_msb 0x8f82
	ds_load_tr16_b128 v[242:245] /*v[754:757]*/, v82 /*v594*/
	ds_load_tr16_b128 v[246:249] /*v[758:761]*/, v82 /*v594*/ offset:4352
	s_set_vgpr_msb 0x828f
	v_add_nc_u32_e32 v82 /*v594*/, v152 /*v920*/, v156 /*v924*/
	s_set_vgpr_msb 0x8fc2
	ds_load_tr16_b128 v[90:93] /*v[858:861]*/, v98 /*v610*/
	ds_load_tr16_b128 v[94:97] /*v[862:865]*/, v98 /*v610*/ offset:4352
	s_set_vgpr_msb 0xc28f
	v_add_nc_u32_e32 v98 /*v610*/, v152 /*v920*/, v162 /*v930*/
	s_set_vgpr_msb 0x8fc2
	ds_load_tr16_b128 v[164:167] /*v[932:935]*/, v162 /*v674*/
	ds_load_tr16_b128 v[168:171] /*v[936:939]*/, v162 /*v674*/ offset:4352
	s_set_vgpr_msb 0xc28f
	v_add_nc_u32_e32 v162 /*v674*/, v154 /*v922*/, v151 /*v919*/
	s_set_vgpr_msb 0x8f82
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
	ds_load_tr16_b128 v[172:175] /*v[940:943]*/, v162 /*v674*/
	ds_load_tr16_b128 v[176:179] /*v[944:947]*/, v162 /*v674*/ offset:1280
	s_set_vgpr_msb 0xc28f
	v_add_nc_u32_e32 v162 /*v674*/, v155 /*v923*/, v151 /*v919*/
	s_set_vgpr_msb 0x8fc2
	ds_load_tr16_b128 v[180:183] /*v[948:951]*/, v162 /*v674*/
	ds_load_tr16_b128 v[184:187] /*v[952:955]*/, v162 /*v674*/ offset:1280
	s_set_vgpr_msb 0xc29a
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[122:129] /*v[634:641]*/, v[138:145] /*v[650:657]*/, v[122:129] /*v[378:385]*/
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[130:137] /*v[642:649]*/, v[146:153] /*v[658:665]*/, v[114:121] /*v[370:377]*/
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[122:129] /*v[634:641]*/, v[154:161] /*v[666:673]*/, v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[130:137] /*v[642:649]*/, v[178:185] /*v[690:697]*/, v[98:105] /*v[354:361]*/
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[122:129] /*v[634:641]*/, v[186:193] /*v[698:705]*/, v[90:97] /*v[346:353]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[130:137] /*v[642:649]*/, v[194:201] /*v[706:713]*/, v[82:89] /*v[338:345]*/
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[122:129] /*v[634:641]*/, v[202:209] /*v[714:721]*/, v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[586:593]*/, v[130:137] /*v[642:649]*/, v[210:217] /*v[722:729]*/, v[34:41] /*v[290:297]*/
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[122:129] /*v[634:641]*/, v[218:225] /*v[730:737]*/, v[26:33] /*v[282:289]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[602:609]*/, v[130:137] /*v[642:649]*/, v[226:233] /*v[738:745]*/, v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x9a8a
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[578:585]*/, v[122:129] /*v[634:641]*/, v[234:241] /*v[746:753]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[618:625]*/, v[130:137] /*v[642:649]*/, v[242:249] /*v[754:761]*/, v[210:217]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[594:601]*/, v[122:129] /*v[634:641]*/, v[250:257] /*v[762:769]*/, v[202:209]
	s_set_vgpr_msb 0x8a8e
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[626:633]*/, v[130:137] /*v[642:649]*/, v[90:97] /*v[858:865]*/, v[194:201]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[610:617]*/, v[122:129] /*v[634:641]*/, v[122:129] /*v[890:897]*/, v[138:145]
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[634:641]*/, v[130:137] /*v[642:649]*/, v[164:171] /*v[932:939]*/, v[130:137]
	s_set_vgpr_msb 0x8e8b
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[642:649]*/, v[172:179] /*v[940:947]*/, v[138:145] /*v[650:657]*/, v[122:129]
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[650:657]*/, v[172:179] /*v[940:947]*/, v[154:161] /*v[666:673]*/, v[106:113]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x8b8c
	v_or_b32_e32 v154 /*v666*/, s2, v141 /*v909*/
	s_set_vgpr_msb 0x8c8b
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[180:187] /*v[948:955]*/, v[178:185] /*v[690:697]*/, v[98:105]
	v_nop
	v_nop
	v_nop
	v_nop
	v_mad_u32 v178 /*v690*/, s67, v154 /*v666*/, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b32_e32 v178 /*v690*/, v143 /*v911*/, v178 /*v690*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[666:673]*/, v[172:179] /*v[940:947]*/, v[202:209] /*v[714:721]*/, v[74:81]
	s_set_vgpr_msb 0x8bc8
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v34 /*v802*/, 4, v178 /*v690*/
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xc88c
	v_or_b32_e32 v202 /*v714*/, s2, v144 /*v912*/
	s_mov_b32 s2, 0
	s_set_vgpr_msb 0x8ccc
	v_or_b32_e32 v42 /*v810*/, 64, v34 /*v802*/
	s_set_vgpr_msb 0xcc82
	v_mad_u32 v202 /*v714*/, v202 /*v714*/, s67, s3
	s_set_vgpr_msb 0x828b
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[172:179] /*v[940:947]*/, v[218:225] /*v[730:737]*/, v[58:65]
	s_set_vgpr_msb 0x8bcf
	v_or_b32_e32 v46 /*v814*/, 0x60, v34 /*v802*/
	s_clause 0x1
	buffer_load_b128 v[10:13] /*v[778:781]*/, v42 /*v810*/, s[48:51], null offen
	buffer_load_b128 v[14:17] /*v[782:785]*/, v46 /*v814*/, s[48:51], null offen
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xcf8e
	v_or_b32_e32 v218 /*v730*/, 0x80, v34 /*v802*/
	v_or_b32_e32 v222 /*v734*/, v202 /*v714*/, v143 /*v911*/
	s_set_vgpr_msb 0x8e8b
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[714:721]*/, v[172:179] /*v[940:947]*/, v[234:241] /*v[746:753]*/, v[42:49]
	s_set_vgpr_msb 0x8b8c
	v_or_b32_e32 v220 /*v732*/, 0xc0, v34 /*v802*/
	v_or_b32_e32 v219 /*v731*/, 0xa0, v34 /*v802*/
	v_or_b32_e32 v221 /*v733*/, 0xe0, v34 /*v802*/
	s_set_vgpr_msb 0x8cc2
	s_clause 0x3
	buffer_load_b128 v[18:21] /*v[786:789]*/, v218 /*v730*/, s[48:51], null offen
	buffer_load_b128 v[22:25] /*v[790:793]*/, v219 /*v731*/, s[48:51], null offen
	buffer_load_b128 v[26:29] /*v[794:797]*/, v220 /*v732*/, s[48:51], null offen
	buffer_load_b128 v[30:33] /*v[798:801]*/, v221 /*v733*/, s[48:51], null offen
	v_nop
	s_set_vgpr_msb 0xc288
	v_lshlrev_b32_e32 v234 /*v746*/, 4, v222 /*v734*/
	s_set_vgpr_msb 0x88c3
	s_clause 0x4
	buffer_load_b128 v[42:45] /*v[810:813]*/, v42 /*v810*/, s[52:55], null offen
	buffer_load_b128 v[46:49] /*v[814:817]*/, v46 /*v814*/, s[52:55], null offen
	s_set_vgpr_msb 0xc3c2
	buffer_load_b128 v[50:53] /*v[818:821]*/, v218 /*v730*/, s[52:55], null offen
	buffer_load_b128 v[54:57] /*v[822:825]*/, v219 /*v731*/, s[52:55], null offen
	s_set_vgpr_msb 0xc28b
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[674:681]*/, v[180:187] /*v[948:955]*/, v[146:153] /*v[658:665]*/, v[114:121]
	s_set_vgpr_msb 0x8bcf
	v_or_b32_e32 v38 /*v806*/, 32, v34 /*v802*/
	s_clause 0x1
	buffer_load_b128 v[2:5] /*v[770:773]*/, v34 /*v802*/, s[48:51], null offen
	buffer_load_b128 v[6:9] /*v[774:777]*/, v38 /*v806*/, s[48:51], null offen
	s_set_vgpr_msb 0xcf88
	v_or_b32_e32 v235 /*v747*/, 32, v234 /*v746*/
	s_set_vgpr_msb 0x88c2
	s_clause 0x1
	buffer_load_b128 v[58:61] /*v[826:829]*/, v220 /*v732*/, s[52:55], null offen
	buffer_load_b128 v[62:65] /*v[830:833]*/, v221 /*v733*/, s[52:55], null offen
	s_clause 0x1
	buffer_load_b128 v[66:69] /*v[834:837]*/, v234 /*v746*/, s[48:51], null offen
	buffer_load_b128 v[70:73] /*v[838:841]*/, v235 /*v747*/, s[48:51], null offen
	s_set_vgpr_msb 0xc28b
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[730:737]*/, v[172:179] /*v[940:947]*/, v[250:257] /*v[762:769]*/, v[26:33]
	v_or_b32_e32 v236 /*v748*/, 64, v234 /*v746*/
	v_or_b32_e32 v237 /*v749*/, 0x60, v234 /*v746*/
	v_or_b32_e32 v238 /*v750*/, 0x80, v234 /*v746*/
	v_or_b32_e32 v239 /*v751*/, 0xa0, v234 /*v746*/
	v_or_b32_e32 v250 /*v762*/, 0xc0, v234 /*v746*/
	v_or_b32_e32 v251 /*v763*/, 0xe0, v234 /*v746*/
	s_set_vgpr_msb 0x8bc3
	s_clause 0x1
	buffer_load_b128 v[34:37] /*v[802:805]*/, v34 /*v802*/, s[52:55], null offen
	buffer_load_b128 v[38:41] /*v[806:809]*/, v38 /*v806*/, s[52:55], null offen
	s_set_vgpr_msb 0xc38b
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[658:665]*/, v[172:179] /*v[940:947]*/, v[186:193] /*v[698:705]*/, v[90:97]
	s_set_vgpr_msb 0x8bc2
	s_clause 0x3
	buffer_load_b128 v[74:77] /*v[842:845]*/, v236 /*v748*/, s[48:51], null offen
	buffer_load_b128 v[78:81] /*v[846:849]*/, v237 /*v749*/, s[48:51], null offen
	buffer_load_b128 v[82:85] /*v[850:853]*/, v238 /*v750*/, s[48:51], null offen
	buffer_load_b128 v[86:89] /*v[854:857]*/, v239 /*v751*/, s[48:51], null offen
	s_set_vgpr_msb 0xc28b
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[698:705]*/, v[180:187] /*v[948:955]*/, v[194:201] /*v[706:713]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[706:713]*/, v[180:187] /*v[948:955]*/, v[210:217] /*v[722:729]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[722:729]*/, v[180:187] /*v[948:955]*/, v[226:233] /*v[738:745]*/, v[50:57]
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[738:745]*/, v[180:187] /*v[948:955]*/, v[242:249] /*v[754:761]*/, v[34:41]
	s_set_vgpr_msb 0x8b8f
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[754:761]*/, v[180:187] /*v[948:955]*/, v[90:97] /*v[858:865]*/, v[18:25]
	s_set_vgpr_msb 0x8fc2
	s_clause 0x1
	buffer_load_b128 v[90:93] /*v[858:861]*/, v250 /*v762*/, s[48:51], null offen
	buffer_load_b128 v[94:97] /*v[862:865]*/, v251 /*v763*/, s[48:51], null offen
	s_clause 0x5
	buffer_load_b128 v[98:101] /*v[866:869]*/, v234 /*v746*/, s[52:55], null offen
	buffer_load_b128 v[102:105] /*v[870:873]*/, v235 /*v747*/, s[52:55], null offen
	buffer_load_b128 v[106:109] /*v[874:877]*/, v236 /*v748*/, s[52:55], null offen
	buffer_load_b128 v[110:113] /*v[878:881]*/, v237 /*v749*/, s[52:55], null offen
	buffer_load_b128 v[114:117] /*v[882:885]*/, v238 /*v750*/, s[52:55], null offen
	buffer_load_b128 v[118:121] /*v[886:889]*/, v239 /*v751*/, s[52:55], null offen
	s_set_vgpr_msb 0xc28f
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[746:753]*/, v[172:179] /*v[940:947]*/, v[122:129] /*v[890:897]*/, v[10:17]
	s_set_vgpr_msb 0x8fc2
	s_clause 0x1
	buffer_load_b128 v[122:125] /*v[890:893]*/, v250 /*v762*/, s[52:55], null offen
	buffer_load_b128 v[126:129] /*v[894:897]*/, v251 /*v763*/, s[52:55], null offen
	s_set_vgpr_msb 0xc28f
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[762:769]*/, v[180:187] /*v[948:955]*/, v[164:171] /*v[932:939]*/, v[2:9]
	s_set_vgpr_msb 0x8f00
	s_branch .LBB0_1
.LBB0_4:
	s_sub_co_i32 s2, s74, s75
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s2, s2, s41
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB0_7
	s_set_vgpr_msb 0x8f
	v_or_b32_e32 v2 /*v514*/, 0xe0, v161 /*v929*/
	v_mov_b64_e32 v[30:31] /*v[542:543]*/, s[38:39]
	v_dual_add_nc_u32 v32 /*v544*/, v152 /*v920*/, v150 /*v918*/ :: v_dual_add_nc_u32 v33 /*v545*/, v153 /*v921*/, v150 /*v918*/
	v_dual_add_nc_u32 v34 /*v546*/, v152 /*v920*/, v151 /*v919*/ :: v_dual_add_nc_u32 v35 /*v547*/, v153 /*v921*/, v151 /*v919*/
	v_dual_add_nc_u32 v36 /*v548*/, v152 /*v920*/, v160 /*v928*/ :: v_dual_add_nc_u32 v37 /*v549*/, v153 /*v921*/, v160 /*v928*/
	v_dual_add_nc_u32 v38 /*v550*/, v152 /*v920*/, v159 /*v927*/ :: v_dual_add_nc_u32 v39 /*v551*/, v153 /*v921*/, v159 /*v927*/
	v_dual_add_nc_u32 v40 /*v552*/, v152 /*v920*/, v158 /*v926*/ :: v_dual_add_nc_u32 v41 /*v553*/, v153 /*v921*/, v158 /*v926*/
	v_dual_add_nc_u32 v42 /*v554*/, v152 /*v920*/, v157 /*v925*/ :: v_dual_add_nc_u32 v43 /*v555*/, v153 /*v921*/, v157 /*v925*/
	v_dual_add_nc_u32 v44 /*v556*/, v152 /*v920*/, v156 /*v924*/ :: v_dual_add_nc_u32 v45 /*v557*/, v153 /*v921*/, v156 /*v924*/
	s_set_vgpr_msb 0x8f8b
	v_dual_add_nc_u32 v46 /*v558*/, v152 /*v920*/, v2 /*v514*/ :: v_dual_add_nc_u32 v47 /*v559*/, v153 /*v921*/, v2 /*v514*/
	s_set_vgpr_msb 0x8b8f
	v_dual_add_nc_u32 v48 /*v560*/, v154 /*v922*/, v150 /*v918*/ :: v_dual_add_nc_u32 v49 /*v561*/, v155 /*v923*/, v150 /*v918*/
	v_dual_add_nc_u32 v50 /*v562*/, v154 /*v922*/, v151 /*v919*/ :: v_dual_add_nc_u32 v51 /*v563*/, v155 /*v923*/, v151 /*v919*/
	s_add_co_i32 s75, s75, s73
	s_ashr_i32 s3, s2, 31
	s_mov_b64 s[4:5], 0
	s_mov_b32 s54, s50
	s_mov_b32 s55, s51
	s_mov_b32 s62, s58
	s_mov_b32 s63, s59
	s_mov_b32 s6, 0x3fb8aa3b
	s_set_vgpr_msb 0x8f00
.LBB0_6:
	s_abs_i32 s7, s4
	s_ashr_i32 s8, s4, 31
	s_mul_hi_u32 s9, s7, s72
	s_xor_b32 s8, s8, s71
	s_mul_i32 s10, s9, s70
	s_add_co_i32 s11, s9, 1
	s_sub_co_i32 s7, s7, s10
	s_set_vgpr_msb 0xa4
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[146:153], v[250:257] /*v[506:513]*/, 0
	s_sub_co_i32 s10, s7, s70
	s_cmp_ge_u32 s7, s70
	s_cselect_b32 s9, s11, s9
	s_cselect_b32 s7, s10, s7
	s_add_co_i32 s10, s9, 1
	s_cmp_ge_u32 s7, s70
	v_wmma_f32_16x16x32_bf16 v[68:75] /*v[580:587]*/, v[178:185], v[186:193] /*v[442:449]*/, 0
	s_cselect_b32 s7, s10, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s7, s7, s8
	s_sub_co_i32 s9, s7, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_mul_i32 s9, s9, s41
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[146:153], v[186:193] /*v[442:449]*/, 0
	s_cmp_lg_u32 s4, s9
	s_cselect_b32 s9, -1, 0
	s_xor_b32 s10, s41, s4
	s_cmp_lt_i32 s10, 0
	s_cselect_b32 s10, -1, 0
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[154:161], v[242:249] /*v[498:505]*/, v[52:59] /*v[564:571]*/
	s_and_b32 s9, s10, s9
	s_sub_co_ci_u32 s7, s7, s8
	s_add_co_i32 s8, s4, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_abs_i32 s9, s8
	s_ashr_i32 s12, s8, 31
	s_mul_hi_u32 s10, s9, s72
	v_wmma_f32_16x16x32_bf16 v[68:75] /*v[580:587]*/, v[186:193], v[178:185] /*v[434:441]*/, v[68:75] /*v[580:587]*/
	s_mul_i32 s11, s10, s70
	s_sub_co_i32 s9, s9, s11
	s_xor_b32 s11, s12, s71
	s_add_co_i32 s12, s10, 1
	s_sub_co_i32 s13, s9, s70
	s_cmp_ge_u32 s9, s70
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[154:161], v[178:185] /*v[434:441]*/, v[84:91] /*v[596:603]*/
	s_cselect_b32 s10, s12, s10
	s_cselect_b32 s9, s13, s9
	s_add_co_i32 s12, s10, 1
	s_cmp_ge_u32 s9, s70
	s_cselect_b32 s9, s12, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_xor_b32 s9, s9, s11
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[162:169], v[234:241] /*v[490:497]*/, v[52:59] /*v[564:571]*/
	s_sub_co_i32 s10, s9, s11
	s_mul_i32 s10, s10, s41
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s8, s10
	s_cselect_b32 s10, -1, 0
	s_xor_b32 s12, s8, s41
	v_wmma_f32_16x16x32_bf16 v[68:75] /*v[580:587]*/, v[226:233], v[170:177] /*v[426:433]*/, v[68:75] /*v[580:587]*/
	s_cmp_lt_i32 s12, 0
	s_cselect_b32 s12, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s10, s12, s10
	s_sub_co_ci_u32 s9, s9, s11
	s_add_co_i32 s10, s7, s75
	s_mul_i32 s7, s41, s7
	s_add_co_i32 s11, s69, s4
	s_lshl_b32 s10, s10, 5
	s_sub_co_i32 s7, s11, s7
	s_set_vgpr_msb 0xa48c
	v_or_b32_e32 v2 /*v514*/, s10, v144 /*v912*/
	s_mul_i32 s7, s7, s37
	s_add_co_i32 s8, s8, s35
	s_add_co_i32 s10, s7, s10
	s_set_vgpr_msb 0x8ca4
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[162:169], v[170:177] /*v[426:433]*/, v[84:91] /*v[596:603]*/
	s_set_vgpr_msb 0xa48c
	v_add_lshl_u32 v3 /*v515*/, s10, v141 /*v909*/, 2
	s_set_vgpr_msb 0x8c8a
	v_add_lshl_u32 v2 /*v514*/, s7, v2 /*v514*/, 2
	s_clause 0x1
	buffer_load_b32 v92 /*v604*/, v3 /*v515*/, s[56:59], null offen
	buffer_load_b32 v94 /*v606*/, v2 /*v514*/, s[56:59], null offen
	s_clause 0x1
	buffer_load_b32 v96 /*v608*/, v3 /*v515*/, s[60:63], null offen
	buffer_load_b32 v98 /*v610*/, v2 /*v514*/, s[60:63], null offen
	s_add_co_i32 s7, s9, s75
	s_mul_i32 s9, s41, s9
	s_lshl_b32 s7, s7, 5
	s_set_vgpr_msb 0x8a85
	v_wmma_f32_16x16x32_bf16 v[76:83] /*v[588:595]*/, v[42:49] /*v[298:305]*/, v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x858e
	v_or_b32_e32 v2 /*v514*/, s7, v141 /*v909*/
	v_or_b32_e32 v3 /*v515*/, s7, v144 /*v912*/
	s_sub_co_i32 s7, s8, s9
	s_add_nc_u64 s[4:5], s[4:5], 1
	s_lshl4_add_u32 s7, s7, s68
	s_cmp_lg_u64 s[4:5], s[2:3]
	v_mad_u32 v2 /*v514*/, v2 /*v514*/, s67, s7
	v_mad_u32 v3 /*v515*/, v3 /*v515*/, s67, s7
	s_set_vgpr_msb 0x8ea4
	v_wmma_f32_16x16x32_bf16 v[52:59] /*v[564:571]*/, v[170:177], v[218:225] /*v[474:481]*/, v[52:59] /*v[564:571]*/
	s_set_vgpr_msb 0xa48e
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v2 /*v514*/, v2 /*v514*/, v143 /*v911*/
	v_or_b32_e32 v3 /*v515*/, v3 /*v515*/, v143 /*v911*/
	s_set_vgpr_msb 0x8ea4
	v_wmma_f32_16x16x32_bf16 v[68:75] /*v[580:587]*/, v[234:241], v[154:161] /*v[410:417]*/, v[68:75] /*v[580:587]*/
	s_set_vgpr_msb 0xa48a
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_lshlrev_b32 v2 /*v514*/, 4, v2 /*v514*/ :: v_dual_lshlrev_b32 v60 /*v572*/, 4, v3 /*v515*/
	v_or_b32_e32 v3 /*v515*/, 32, v2 /*v514*/
	v_or_b32_e32 v4 /*v516*/, 64, v2 /*v514*/
	v_or_b32_e32 v5 /*v517*/, 0x60, v2 /*v514*/
	v_or_b32_e32 v6 /*v518*/, 0x80, v2 /*v514*/
	v_or_b32_e32 v7 /*v519*/, 0xa0, v2 /*v514*/
	v_or_b32_e32 v8 /*v520*/, 0xc0, v2 /*v514*/
	v_or_b32_e32 v9 /*v521*/, 0xe0, v2 /*v514*/
	v_or_b32_e32 v61 /*v573*/, 32, v60 /*v572*/
	v_or_b32_e32 v62 /*v574*/, 64, v60 /*v572*/
	v_or_b32_e32 v63 /*v575*/, 0x60, v60 /*v572*/
	v_or_b32_e32 v64 /*v576*/, 0x80, v60 /*v572*/
	v_or_b32_e32 v65 /*v577*/, 0xa0, v60 /*v572*/
	v_or_b32_e32 v66 /*v578*/, 0xc0, v60 /*v572*/
	v_or_b32_e32 v67 /*v579*/, 0xe0, v60 /*v572*/
	s_clause 0x7
	buffer_load_b128 v[124:127] /*v[636:639]*/, v2 /*v514*/, s[48:51], null offen
	buffer_load_b128 v[128:131] /*v[640:643]*/, v3 /*v515*/, s[48:51], null offen
	buffer_load_b128 v[132:135] /*v[644:647]*/, v4 /*v516*/, s[48:51], null offen
	buffer_load_b128 v[136:139] /*v[648:651]*/, v5 /*v517*/, s[48:51], null offen
	buffer_load_b128 v[140:143] /*v[652:655]*/, v6 /*v518*/, s[48:51], null offen
	buffer_load_b128 v[144:147] /*v[656:659]*/, v7 /*v519*/, s[48:51], null offen
	buffer_load_b128 v[148:151] /*v[660:663]*/, v8 /*v520*/, s[48:51], null offen
	buffer_load_b128 v[152:155] /*v[664:667]*/, v9 /*v521*/, s[48:51], null offen
	s_clause 0x7
	buffer_load_b128 v[156:159] /*v[668:671]*/, v2 /*v514*/, s[52:55], null offen
	buffer_load_b128 v[160:163] /*v[672:675]*/, v3 /*v515*/, s[52:55], null offen
	buffer_load_b128 v[164:167] /*v[676:679]*/, v4 /*v516*/, s[52:55], null offen
	buffer_load_b128 v[168:171] /*v[680:683]*/, v5 /*v517*/, s[52:55], null offen
	buffer_load_b128 v[172:175] /*v[684:687]*/, v6 /*v518*/, s[52:55], null offen
	buffer_load_b128 v[176:179] /*v[688:691]*/, v7 /*v519*/, s[52:55], null offen
	buffer_load_b128 v[180:183] /*v[692:695]*/, v8 /*v520*/, s[52:55], null offen
	buffer_load_b128 v[184:187] /*v[696:699]*/, v9 /*v521*/, s[52:55], null offen
	s_clause 0x7
	buffer_load_b128 v[2:5] /*v[514:517]*/, v60 /*v572*/, s[48:51], null offen
	buffer_load_b128 v[6:9] /*v[518:521]*/, v61 /*v573*/, s[48:51], null offen
	buffer_load_b128 v[10:13] /*v[522:525]*/, v62 /*v574*/, s[48:51], null offen
	buffer_load_b128 v[14:17] /*v[526:529]*/, v63 /*v575*/, s[48:51], null offen
	buffer_load_b128 v[18:21] /*v[530:533]*/, v64 /*v576*/, s[48:51], null offen
	buffer_load_b128 v[22:25] /*v[534:537]*/, v65 /*v577*/, s[48:51], null offen
	buffer_load_b128 v[26:29] /*v[538:541]*/, v66 /*v578*/, s[48:51], null offen
	buffer_load_b128 v[188:191] /*v[700:703]*/, v67 /*v579*/, s[48:51], null offen
	s_clause 0x7
	buffer_load_b128 v[192:195] /*v[704:707]*/, v60 /*v572*/, s[52:55], null offen
	buffer_load_b128 v[196:199] /*v[708:711]*/, v61 /*v573*/, s[52:55], null offen
	buffer_load_b128 v[200:203] /*v[712:715]*/, v62 /*v574*/, s[52:55], null offen
	buffer_load_b128 v[204:207] /*v[716:719]*/, v63 /*v575*/, s[52:55], null offen
	buffer_load_b128 v[208:211] /*v[720:723]*/, v64 /*v576*/, s[52:55], null offen
	buffer_load_b128 v[212:215] /*v[724:727]*/, v65 /*v577*/, s[52:55], null offen
	buffer_load_b128 v[216:219] /*v[728:731]*/, v66 /*v578*/, s[52:55], null offen
	buffer_load_b128 v[220:223] /*v[732:735]*/, v67 /*v579*/, s[52:55], null offen
	s_set_vgpr_msb 0x8a84
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[178:185], v[250:257] /*v[506:513]*/, 0
	s_set_vgpr_msb 0x8407
	ds_store_b128 v146 /*v914*/, v[226:229] /*v[482:485]*/ offset:5120
	ds_store_b128 v146 /*v914*/, v[230:233] /*v[486:489]*/ offset:5152
	ds_store_b128 v146 /*v914*/, v[250:253] /*v[506:509]*/ offset:13824
	ds_store_b128 v146 /*v914*/, v[254:257] /*v[510:513]*/ offset:13856
	ds_store_b128 v146 /*v914*/, v[210:213] /*v[466:469]*/ offset:5184
	ds_store_b128 v146 /*v914*/, v[214:217] /*v[470:473]*/ offset:5216
	ds_store_b128 v146 /*v914*/, v[242:245] /*v[498:501]*/ offset:13888
	ds_store_b128 v146 /*v914*/, v[246:249] /*v[502:505]*/ offset:13920
	ds_store_b128 v146 /*v914*/, v[202:205] /*v[458:461]*/ offset:5248
	ds_store_b128 v146 /*v914*/, v[206:209] /*v[462:465]*/ offset:5280
	ds_store_b128 v146 /*v914*/, v[234:237] /*v[490:493]*/ offset:13952
	ds_store_b128 v146 /*v914*/, v[238:241] /*v[494:497]*/ offset:13984
	ds_store_b128 v146 /*v914*/, v[194:197] /*v[450:453]*/ offset:5312
	ds_store_b128 v146 /*v914*/, v[218:221] /*v[474:477]*/ offset:14016
	ds_store_b128 v146 /*v914*/, v[222:225] /*v[478:481]*/ offset:14048
	ds_store_b128 v146 /*v914*/, v[198:201] /*v[454:457]*/ offset:5344
	s_set_vgpr_msb 0x78a
	v_pk_mul_f32 v[68:69] /*v[580:581]*/, v[30:31] /*v[542:543]*/, v[68:69] /*v[580:581]*/
	v_pk_mul_f32 v[70:71] /*v[582:583]*/, v[30:31] /*v[542:543]*/, v[70:71] /*v[582:583]*/
	v_pk_mul_f32 v[72:73] /*v[584:585]*/, v[30:31] /*v[542:543]*/, v[72:73] /*v[584:585]*/
	v_pk_mul_f32 v[74:75] /*v[586:587]*/, v[30:31] /*v[542:543]*/, v[74:75] /*v[586:587]*/
	s_set_vgpr_msb 0x8aa4
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[186:193], v[242:249] /*v[498:505]*/, v[60:67] /*v[572:579]*/
	s_wait_loadcnt 0x22
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa44a
	v_pk_add_f32 v[242:243] /*v[498:499]*/, v[68:69] /*v[580:581]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa4
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[226:233], v[234:241] /*v[490:497]*/, v[60:67] /*v[572:579]*/
	s_set_vgpr_msb 0xa44a
	v_pk_add_f32 v[244:245] /*v[500:501]*/, v[70:71] /*v[582:583]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[246:247] /*v[502:503]*/, v[72:73] /*v[584:585]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[74:75] /*v[586:587]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa4
	s_delay_alu instid0(TRANS32_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[234:241], v[218:225] /*v[474:481]*/, v[60:67] /*v[572:579]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa44a
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[30:31] /*v[542:543]*/, v[52:53] /*v[564:565]*/
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[30:31] /*v[542:543]*/, v[54:55] /*v[566:567]*/
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[30:31] /*v[542:543]*/, v[56:57] /*v[568:569]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_mul_f32 v[100:101] /*v[612:613]*/, v[30:31] /*v[542:543]*/, v[60:61] /*v[572:573]*/
	v_pk_mul_f32 v[102:103] /*v[614:615]*/, v[30:31] /*v[542:543]*/, v[62:63] /*v[574:575]*/
	v_pk_mul_f32 v[104:105] /*v[616:617]*/, v[30:31] /*v[542:543]*/, v[64:65] /*v[576:577]*/
	v_pk_mul_f32 v[106:107] /*v[618:619]*/, v[30:31] /*v[542:543]*/, v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x8a84
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[242:249], v[226:233] /*v[482:489]*/, 0
	s_set_vgpr_msb 0x844a
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[30:31] /*v[542:543]*/, v[58:59] /*v[570:571]*/
	s_set_vgpr_msb 0x4a49
	v_pk_add_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[234:235] /*v[490:491]*/, v[100:101] /*v[612:613]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[236:237] /*v[492:493]*/, v[102:103] /*v[614:615]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a45
	v_pk_mul_f32 v[218:219] /*v[474:475]*/, v[218:219] /*v[474:475]*/, s[6:7] op_sel_hi:[1,0]
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[42:49] /*v[298:305]*/, v[226:233] /*v[482:489]*/, 0
	v_pk_mul_f32 v[220:221] /*v[476:477]*/, v[220:221] /*v[476:477]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223] /*v[478:479]*/, v[222:223] /*v[478:479]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[234:235] /*v[490:491]*/, v[234:235] /*v[490:491]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[236:237] /*v[492:493]*/, v[236:237] /*v[492:493]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v218 /*v474*/, v218 /*v474*/
	v_exp_f32_e32 v219 /*v475*/, v219 /*v475*/
	v_exp_f32_e32 v220 /*v476*/, v220 /*v476*/
	s_set_vgpr_msb 0x4544
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[242:249], v[162:169] /*v[418:425]*/, 0
	s_set_vgpr_msb 0x4441
	v_exp_f32_e32 v221 /*v477*/, v221 /*v477*/
	v_exp_f32_e32 v222 /*v478*/, v222 /*v478*/
	v_exp_f32_e32 v223 /*v479*/, v223 /*v479*/
	v_exp_f32_e32 v234 /*v490*/, v234 /*v490*/
	v_exp_f32_e32 v235 /*v491*/, v235 /*v491*/
	v_exp_f32_e32 v236 /*v492*/, v236 /*v492*/
	v_exp_f32_e32 v237 /*v493*/, v237 /*v493*/
	s_set_vgpr_msb 0x41a4
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[250:257], v[210:217] /*v[466:473]*/, v[60:67] /*v[572:579]*/
	s_set_vgpr_msb 0xa455
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[50:57] /*v[306:313]*/, v[210:217] /*v[466:473]*/, v[250:257] /*v[506:513]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5549
	v_pk_add_f32 v[210:211] /*v[466:467]*/, v[224:225] /*v[480:481]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[212:213] /*v[468:469]*/, v[104:105] /*v[616:617]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[214:215] /*v[470:471]*/, v[106:107] /*v[618:619]*/, v[92:93] /*v[604:605]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa4
	v_wmma_f32_16x16x32_bf16 v[84:91] /*v[596:603]*/, v[170:177], v[154:161] /*v[410:417]*/, v[84:91] /*v[596:603]*/
	s_set_vgpr_msb 0xa441
	v_pk_mul_f32 v[210:211] /*v[466:467]*/, v[210:211] /*v[466:467]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[212:213] /*v[468:469]*/, v[212:213] /*v[468:469]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[214:215] /*v[470:471]*/, v[214:215] /*v[470:471]*/, s[6:7] op_sel_hi:[1,0]
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v210 /*v466*/, v210 /*v466*/
	s_set_vgpr_msb 0x41a5
	v_wmma_f32_16x16x32_bf16 v[76:83] /*v[588:595]*/, v[50:57] /*v[306:313]*/, v[146:153] /*v[402:409]*/, v[76:83] /*v[588:595]*/
	s_set_vgpr_msb 0xa58a
	v_pk_mul_f32 v[84:85] /*v[596:597]*/, v[30:31] /*v[542:543]*/, v[84:85] /*v[596:597]*/
	v_pk_mul_f32 v[88:89] /*v[600:601]*/, v[30:31] /*v[542:543]*/, v[88:89] /*v[600:601]*/
	v_pk_mul_f32 v[90:91] /*v[602:603]*/, v[30:31] /*v[542:543]*/, v[90:91] /*v[602:603]*/
	v_pk_mul_f32 v[86:87] /*v[598:599]*/, v[30:31] /*v[542:543]*/, v[86:87] /*v[598:599]*/
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v211 /*v467*/, v211 /*v467*/
	s_set_vgpr_msb 0x414a
	v_pk_add_f32 v[216:217] /*v[472:473]*/, v[84:85] /*v[596:597]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[238:239] /*v[494:495]*/, v[88:89] /*v[600:601]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a54
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[250:257], v[146:153] /*v[402:409]*/, v[226:233] /*v[482:489]*/
	s_set_vgpr_msb 0x544a
	v_pk_add_f32 v[240:241] /*v[496:497]*/, v[90:91] /*v[602:603]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[224:225] /*v[480:481]*/, v[86:87] /*v[598:599]*/, v[94:95] /*v[606:607]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a41
	v_pk_mul_f32 v[216:217] /*v[472:473]*/, v[216:217] /*v[472:473]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v212 /*v468*/, v212 /*v468*/
	v_exp_f32_e32 v213 /*v469*/, v213 /*v469*/
	v_exp_f32_e32 v214 /*v470*/, v214 /*v470*/
	v_pk_mul_f32 v[224:225] /*v[480:481]*/, v[224:225] /*v[480:481]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x41a5
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[2:9] /*v[258:265]*/, v[202:209] /*v[458:465]*/, v[60:67] /*v[572:579]*/
	s_set_vgpr_msb 0xa555
	v_exp_f32_e32 v215 /*v471*/, v215 /*v471*/
	v_exp_f32_e32 v224 /*v480*/, v224 /*v480*/
	v_exp_f32_e32 v225 /*v481*/, v225 /*v481*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[58:65] /*v[314:321]*/, v[202:209] /*v[458:465]*/, v[250:257] /*v[506:513]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[238:239] /*v[494:495]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[240:241] /*v[496:497]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[242:243] /*v[498:499]*/, s[6:7] op_sel_hi:[1,0]
	s_set_vgpr_msb 0x55a5
	v_wmma_f32_16x16x32_bf16 v[76:83] /*v[588:595]*/, v[58:65] /*v[314:321]*/, v[138:145] /*v[394:401]*/, v[76:83] /*v[588:595]*/
	s_set_vgpr_msb 0xa555
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[244:245] /*v[500:501]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[238:239] /*v[494:495]*/, v[246:247] /*v[502:503]*/, s[6:7] op_sel_hi:[1,0]
	v_pk_mul_f32 v[240:241] /*v[496:497]*/, v[248:249] /*v[504:505]*/, s[6:7] op_sel_hi:[1,0]
	v_exp_f32_e32 v242 /*v498*/, v216 /*v472*/
	v_exp_f32_e32 v243 /*v499*/, v217 /*v473*/
	v_exp_f32_e32 v244 /*v500*/, v202 /*v458*/
	v_exp_f32_e32 v245 /*v501*/, v203 /*v459*/
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[2:9] /*v[258:265]*/, v[138:145] /*v[394:401]*/, v[226:233] /*v[482:489]*/
	v_exp_f32_e32 v216 /*v472*/, v204 /*v460*/
	v_exp_f32_e32 v217 /*v473*/, v205 /*v461*/
	v_exp_f32_e32 v246 /*v502*/, v206 /*v462*/
	v_exp_f32_e32 v247 /*v503*/, v207 /*v463*/
	v_exp_f32_e32 v248 /*v504*/, v208 /*v464*/
	v_exp_f32_e32 v249 /*v505*/, v209 /*v465*/
	v_exp_f32_e32 v238 /*v494*/, v238 /*v494*/
	s_set_vgpr_msb 0x55a5
	v_wmma_f32_16x16x32_bf16 v[60:67] /*v[572:579]*/, v[10:17] /*v[266:273]*/, v[194:201] /*v[450:457]*/, v[60:67] /*v[572:579]*/
	s_set_vgpr_msb 0xa555
	v_exp_f32_e32 v239 /*v495*/, v239 /*v495*/
	v_exp_f32_e32 v240 /*v496*/, v240 /*v496*/
	v_exp_f32_e32 v241 /*v497*/, v241 /*v497*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[66:73] /*v[322:329]*/, v[194:201] /*v[450:457]*/, v[250:257] /*v[506:513]*/
	s_wait_loadcnt 0x21
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x554a
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[60:61] /*v[572:573]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[62:63] /*v[574:575]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[64:65] /*v[576:577]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4aa5
	v_wmma_f32_16x16x32_bf16 v[76:83] /*v[588:595]*/, v[66:73] /*v[322:329]*/, v[130:137] /*v[386:393]*/, v[76:83] /*v[588:595]*/
	s_set_vgpr_msb 0xa54a
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[66:67] /*v[578:579]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a49
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[250:251] /*v[506:507]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[204:205] /*v[460:461]*/, v[252:253] /*v[508:509]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[206:207] /*v[462:463]*/, v[254:255] /*v[510:511]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[208:209] /*v[464:465]*/, v[0:1] /*v[512:513]*/, v[96:97] /*v[608:609]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a55
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, v[218:219] /*v[474:475]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[196:197] /*v[452:453]*/, v[220:221] /*v[476:477]*/
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[10:17] /*v[266:273]*/, v[130:137] /*v[386:393]*/, v[226:233] /*v[482:489]*/
	s_set_vgpr_msb 0x554a
	s_wait_loadcnt 0x20
	v_pk_add_f32 v[250:251] /*v[506:507]*/, v[76:77] /*v[588:589]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[252:253] /*v[508:509]*/, v[78:79] /*v[590:591]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[254:255] /*v[510:511]*/, v[80:81] /*v[592:593]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[0:1] /*v[512:513]*/, v[82:83] /*v[594:595]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x8a45
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, v[222:223] /*v[478:479]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, v[210:211] /*v[466:467]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[202:203] /*v[458:459]*/, v[234:235] /*v[490:491]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[226:227] /*v[482:483]*/, v[226:227] /*v[482:483]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[228:229] /*v[484:485]*/, v[228:229] /*v[484:485]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[232:233] /*v[488:489]*/, v[232:233] /*v[488:489]*/, v[98:99] /*v[610:611]*/ op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	s_set_vgpr_msb 0x4945
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[204:205] /*v[460:461]*/, v[236:237] /*v[492:493]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[206:207] /*v[462:463]*/, v[212:213] /*v[468:469]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[208:209] /*v[464:465]*/, v[214:215] /*v[470:471]*/
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[226:227] /*v[482:483]*/, v[242:243] /*v[498:499]*/
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[228:229] /*v[484:485]*/, v[224:225] /*v[480:481]*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[230:231] /*v[486:487]*/, v[244:245] /*v[500:501]*/
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[232:233] /*v[488:489]*/, v[216:217] /*v[472:473]*/
	v_pk_mul_f32 v[250:251] /*v[506:507]*/, v[250:251] /*v[506:507]*/, v[246:247] /*v[502:503]*/
	v_pk_mul_f32 v[252:253] /*v[508:509]*/, v[252:253] /*v[508:509]*/, v[248:249] /*v[504:505]*/
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[254:255] /*v[510:511]*/, v[238:239] /*v[494:495]*/
	s_set_vgpr_msb 0x4586
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[0:1] /*v[512:513]*/, v[240:241] /*v[496:497]*/
	s_set_vgpr_msb 0x8646
	v_pk_mul_f32 v[194:195] /*v[450:451]*/, v[30:31] /*v[542:543]*/, v[194:195] /*v[450:451]*/
	v_pk_mul_f32 v[196:197] /*v[452:453]*/, v[30:31] /*v[542:543]*/, v[196:197] /*v[452:453]*/
	v_pk_mul_f32 v[198:199] /*v[454:455]*/, v[30:31] /*v[542:543]*/, v[198:199] /*v[454:455]*/
	v_pk_mul_f32 v[200:201] /*v[456:457]*/, v[30:31] /*v[542:543]*/, v[200:201] /*v[456:457]*/
	v_pk_mul_f32 v[202:203] /*v[458:459]*/, v[30:31] /*v[542:543]*/, v[202:203] /*v[458:459]*/
	v_pk_mul_f32 v[204:205] /*v[460:461]*/, v[30:31] /*v[542:543]*/, v[204:205] /*v[460:461]*/
	v_pk_mul_f32 v[206:207] /*v[462:463]*/, v[30:31] /*v[542:543]*/, v[206:207] /*v[462:463]*/
	v_pk_mul_f32 v[208:209] /*v[464:465]*/, v[30:31] /*v[542:543]*/, v[208:209] /*v[464:465]*/
	v_pk_mul_f32 v[226:227] /*v[482:483]*/, v[30:31] /*v[542:543]*/, v[226:227] /*v[482:483]*/
	v_pk_mul_f32 v[228:229] /*v[484:485]*/, v[30:31] /*v[542:543]*/, v[228:229] /*v[484:485]*/
	v_pk_mul_f32 v[230:231] /*v[486:487]*/, v[30:31] /*v[542:543]*/, v[230:231] /*v[486:487]*/
	v_pk_mul_f32 v[232:233] /*v[488:489]*/, v[30:31] /*v[542:543]*/, v[232:233] /*v[488:489]*/
	v_pk_mul_f32 v[250:251] /*v[506:507]*/, v[30:31] /*v[542:543]*/, v[250:251] /*v[506:507]*/
	v_pk_mul_f32 v[252:253] /*v[508:509]*/, v[30:31] /*v[542:543]*/, v[252:253] /*v[508:509]*/
	v_pk_mul_f32 v[254:255] /*v[510:511]*/, v[30:31] /*v[542:543]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x468a
	v_pk_mul_f32 v[0:1] /*v[512:513]*/, v[30:31] /*v[542:543]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x8a45
	v_cvt_pk_bf16_f32 v194 /*v450*/, v194 /*v450*/, v195 /*v451*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v196 /*v452*/, v197 /*v453*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v198 /*v454*/, v199 /*v455*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v200 /*v456*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v201 /*v457*/, v210 /*v466*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v200 /*v456*/, v222 /*v478*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v199 /*v455*/, v220 /*v476*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v218 /*v474*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v202 /*v458*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v203 /*v459*/, v204 /*v460*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v206 /*v462*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v208 /*v464*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v209 /*v465*/, v214 /*v470*/, v215 /*v471*/
	v_cvt_pk_bf16_f32 v208 /*v464*/, v212 /*v468*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v207 /*v463*/, v236 /*v492*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v234 /*v490*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v210 /*v466*/, v226 /*v482*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v211 /*v467*/, v228 /*v484*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v212 /*v468*/, v230 /*v486*/, v231 /*v487*/
	v_cvt_pk_bf16_f32 v213 /*v469*/, v232 /*v488*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v217 /*v473*/, v216 /*v472*/, v217 /*v473*/
	v_cvt_pk_bf16_f32 v216 /*v472*/, v244 /*v500*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v215 /*v471*/, v224 /*v480*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v214 /*v470*/, v242 /*v498*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v218 /*v474*/, v250 /*v506*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v219 /*v475*/, v252 /*v508*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v220 /*v476*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v221 /*v477*/, v0 /*v512*/, v1 /*v513*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v225 /*v481*/, v240 /*v496*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v224 /*v480*/, v238 /*v494*/, v239 /*v495*/
	v_cvt_pk_bf16_f32 v223 /*v479*/, v248 /*v504*/, v249 /*v505*/
	v_cvt_pk_bf16_f32 v222 /*v478*/, v246 /*v502*/, v247 /*v503*/
	s_set_vgpr_msb 0x4507
	ds_store_b128 v147 /*v915*/, v[198:201] /*v[454:457]*/
	ds_store_b128 v147 /*v915*/, v[206:209] /*v[462:465]*/ offset:32
	ds_store_b128 v147 /*v915*/, v[194:197] /*v[450:453]*/ offset:2560
	ds_store_b128 v147 /*v915*/, v[202:205] /*v[458:461]*/ offset:2592
	ds_store_b128 v148 /*v916*/, v[162:165] /*v[418:421]*/ offset:5120
	ds_store_b128 v148 /*v916*/, v[166:169] /*v[422:425]*/ offset:5152
	ds_store_b128 v148 /*v916*/, v[186:189] /*v[442:445]*/ offset:13824
	ds_store_b128 v148 /*v916*/, v[190:193] /*v[446:449]*/ offset:13856
	ds_store_b128 v148 /*v916*/, v[146:149] /*v[402:405]*/ offset:5184
	ds_store_b128 v148 /*v916*/, v[150:153] /*v[406:409]*/ offset:5216
	ds_store_b128 v148 /*v916*/, v[178:181] /*v[434:437]*/ offset:13888
	ds_store_b128 v148 /*v916*/, v[182:185] /*v[438:441]*/ offset:13920
	ds_store_b128 v148 /*v916*/, v[138:141] /*v[394:397]*/ offset:5248
	ds_store_b128 v148 /*v916*/, v[142:145] /*v[398:401]*/ offset:5280
	ds_store_b128 v148 /*v916*/, v[170:173] /*v[426:429]*/ offset:13952
	ds_store_b128 v148 /*v916*/, v[174:177] /*v[430:433]*/ offset:13984
	ds_store_b128 v148 /*v916*/, v[130:133] /*v[386:389]*/ offset:5312
	ds_store_b128 v148 /*v916*/, v[134:137] /*v[390:393]*/ offset:5344
	ds_store_b128 v148 /*v916*/, v[154:157] /*v[410:413]*/ offset:14016
	ds_store_b128 v148 /*v916*/, v[158:161] /*v[414:417]*/ offset:14048
	ds_store_b128 v149 /*v917*/, v[214:217] /*v[470:473]*/
	ds_store_b128 v149 /*v917*/, v[222:225] /*v[478:481]*/ offset:32
	ds_store_b128 v149 /*v917*/, v[210:213] /*v[466:469]*/ offset:2560
	ds_store_b128 v149 /*v917*/, v[218:221] /*v[474:477]*/ offset:2592
	s_set_vgpr_msb 0x742
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v32 /*v544*/
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v32 /*v544*/ offset:4352
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v33 /*v545*/
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v33 /*v545*/ offset:4352
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v34 /*v546*/
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v34 /*v546*/ offset:4352
	ds_load_tr16_b128 v[242:245] /*v[498:501]*/, v35 /*v547*/
	ds_load_tr16_b128 v[246:249] /*v[502:505]*/, v35 /*v547*/ offset:4352
	ds_load_tr16_b128 v[130:133] /*v[386:389]*/, v36 /*v548*/
	ds_load_tr16_b128 v[134:137] /*v[390:393]*/, v36 /*v548*/ offset:4352
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v37 /*v549*/
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v37 /*v549*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v38 /*v550*/
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v38 /*v550*/ offset:4352
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v39 /*v551*/
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v39 /*v551*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v40 /*v552*/
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v40 /*v552*/ offset:4352
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v41 /*v553*/
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v41 /*v553*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v42 /*v554*/
	ds_load_tr16_b128 v[72:75] /*v[584:587]*/, v42 /*v554*/ offset:4352
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v43 /*v555*/
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v43 /*v555*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[76:79] /*v[588:591]*/, v44 /*v556*/
	ds_load_tr16_b128 v[80:83] /*v[592:595]*/, v44 /*v556*/ offset:4352
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v45 /*v557*/
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v45 /*v557*/ offset:4352
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[84:87] /*v[596:599]*/, v46 /*v558*/
	ds_load_tr16_b128 v[88:91] /*v[600:603]*/, v46 /*v558*/ offset:4352
	ds_load_tr16_b128 v[92:95] /*v[604:607]*/, v47 /*v559*/
	ds_load_tr16_b128 v[96:99] /*v[608:611]*/, v47 /*v559*/ offset:4352
	ds_load_tr16_b128 v[100:103] /*v[612:615]*/, v48 /*v560*/
	ds_load_tr16_b128 v[104:107] /*v[616:619]*/, v48 /*v560*/ offset:1280
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[250:253] /*v[506:509]*/, v49 /*v561*/
	ds_load_tr16_b128 v[254:257] /*v[510:513]*/, v49 /*v561*/ offset:1280
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[108:111] /*v[620:623]*/, v50 /*v562*/
	ds_load_tr16_b128 v[112:115] /*v[624:627]*/, v50 /*v562*/ offset:1280
	ds_load_tr16_b128 v[116:119] /*v[628:631]*/, v51 /*v563*/
	ds_load_tr16_b128 v[120:123] /*v[632:635]*/, v51 /*v563*/ offset:1280
	s_set_vgpr_msb 0x8256
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[100:107] /*v[612:619]*/, v[170:177] /*v[426:433]*/, v[122:129] /*v[378:385]*/
	s_wait_loadcnt 0x12
	v_mov_b64_e32 v[206:207] /*v[462:463]*/, v[176:177] /*v[688:689]*/
	v_mov_b64_e32 v[208:209] /*v[464:465]*/, v[178:179] /*v[690:691]*/
	v_mov_b64_e32 v[202:203] /*v[458:459]*/, v[172:173] /*v[684:685]*/
	v_mov_b64_e32 v[204:205] /*v[460:461]*/, v[174:175] /*v[686:687]*/
	v_mov_b64_e32 v[214:215] /*v[470:471]*/, v[168:169] /*v[680:681]*/
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[170:171] /*v[682:683]*/
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[164:165] /*v[676:677]*/
	s_set_vgpr_msb 0x5606
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[106:113], v[108:115] /*v[620:627]*/, v[186:193] /*v[442:449]*/, v[106:113]
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[166:167] /*v[678:679]*/
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[160:161] /*v[672:673]*/
	v_mov_b64_e32 v[232:233] /*v[488:489]*/, v[162:163] /*v[674:675]*/
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[156:157] /*v[668:669]*/
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[158:159] /*v[670:671]*/
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[152:153] /*v[664:665]*/
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[154:155] /*v[666:667]*/
	s_set_vgpr_msb 0x4206
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[98:105], v[116:123] /*v[628:635]*/, v[242:249] /*v[498:505]*/, v[98:105]
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[148:149] /*v[660:661]*/
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[150:151] /*v[662:663]*/
	v_mov_b64_e32 v[238:239] /*v[494:495]*/, v[144:145] /*v[656:657]*/
	v_mov_b64_e32 v[240:241] /*v[496:497]*/, v[146:147] /*v[658:659]*/
	v_mov_b64_e32 v[234:235] /*v[490:491]*/, v[140:141] /*v[652:653]*/
	v_mov_b64_e32 v[236:237] /*v[492:493]*/, v[142:143] /*v[654:655]*/
	s_set_vgpr_msb 0x4206
	v_wmma_f32_16x16x32_bf16 v[82:89], v[116:123] /*v[628:635]*/, v[138:145] /*v[394:401]*/, v[82:89]
	v_wmma_f32_16x16x32_bf16 v[66:73], v[116:123] /*v[628:635]*/, v[146:153] /*v[402:409]*/, v[66:73]
	v_wmma_f32_16x16x32_bf16 v[50:57], v[116:123] /*v[628:635]*/, v[154:161] /*v[410:417]*/, v[50:57]
	v_wmma_f32_16x16x32_bf16 v[34:41], v[116:123] /*v[628:635]*/, v[162:169] /*v[418:425]*/, v[34:41]
	v_wmma_f32_16x16x32_bf16 v[18:25], v[116:123] /*v[628:635]*/, v[194:201] /*v[450:457]*/, v[18:25]
	v_wmma_f32_16x16x32_bf16 v[90:97], v[108:115] /*v[620:627]*/, v[130:137] /*v[386:393]*/, v[90:97]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[74:81], v[108:115] /*v[620:627]*/, v[52:59] /*v[564:571]*/, v[74:81]
	v_wmma_f32_16x16x32_bf16 v[58:65], v[108:115] /*v[620:627]*/, v[60:67] /*v[572:579]*/, v[58:65]
	v_wmma_f32_16x16x32_bf16 v[42:49], v[108:115] /*v[620:627]*/, v[68:75] /*v[580:587]*/, v[42:49]
	v_wmma_f32_16x16x32_bf16 v[26:33], v[108:115] /*v[620:627]*/, v[76:83] /*v[588:595]*/, v[26:33]
	v_wmma_f32_16x16x32_bf16 v[10:17], v[108:115] /*v[620:627]*/, v[84:91] /*v[596:603]*/, v[10:17]
	s_set_vgpr_msb 0xa55
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[250:257] /*v[506:513]*/, v[138:145] /*v[394:401]*/, v[82:89] /*v[338:345]*/
	s_wait_loadcnt 0x2
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5542
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[212:213] /*v[724:725]*/
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[214:215] /*v[726:727]*/
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[208:209] /*v[720:721]*/
	s_set_vgpr_msb 0x4255
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[250:257] /*v[506:513]*/, v[146:153] /*v[402:409]*/, v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5542
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[210:211] /*v[722:723]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[204:205] /*v[716:717]*/
	v_mov_b64_e32 v[152:153] /*v[408:409]*/, v[206:207] /*v[718:719]*/
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0x4255
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[250:257] /*v[506:513]*/, v[154:161] /*v[410:417]*/, v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5542
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[202:203] /*v[714:715]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[158:159] /*v[414:415]*/, v[188:189] /*v[700:701]*/
	v_mov_b64_e32 v[160:161] /*v[416:417]*/, v[190:191] /*v[702:703]*/
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[26:27] /*v[538:539]*/
	s_set_vgpr_msb 0x4205
	v_wmma_f32_16x16x32_bf16 v[210:217], v[250:257] /*v[506:513]*/, v[162:169] /*v[418:425]*/, v[210:217]
	s_set_vgpr_msb 0x542
	v_mov_b64_e32 v[156:157] /*v[412:413]*/, v[28:29] /*v[540:541]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[166:167] /*v[422:423]*/, v[196:197] /*v[708:709]*/
	v_mov_b64_e32 v[168:169] /*v[424:425]*/, v[198:199] /*v[710:711]*/
	v_mov_b64_e32 v[162:163] /*v[418:419]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0x4205
	v_wmma_f32_16x16x32_bf16 v[194:201], v[250:257] /*v[506:513]*/, v[194:201] /*v[450:457]*/, v[194:201]
	s_set_vgpr_msb 0x542
	v_mov_b64_e32 v[164:165] /*v[420:421]*/, v[194:195] /*v[706:707]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[198:199] /*v[454:455]*/, v[184:185] /*v[696:697]*/
	v_mov_b64_e32 v[200:201] /*v[456:457]*/, v[186:187] /*v[698:699]*/
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, v[180:181] /*v[692:693]*/
	s_set_vgpr_msb 0x4209
	v_wmma_f32_16x16x32_bf16 v[130:137], v[250:257] /*v[506:513]*/, v[92:99] /*v[604:611]*/, v[130:137]
	s_set_vgpr_msb 0x956
	v_mov_b64_e32 v[196:197] /*v[452:453]*/, v[182:183] /*v[694:695]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[100:107] /*v[612:619]*/, v[130:137] /*v[386:393]*/, v[90:97] /*v[346:353]*/
	s_wait_loadcnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[134:135] /*v[390:391]*/, v[220:221] /*v[732:733]*/
	v_mov_b64_e32 v[136:137] /*v[392:393]*/, v[222:223] /*v[734:735]*/
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[216:217] /*v[728:729]*/
	s_set_vgpr_msb 0x5655
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[250:257] /*v[506:513]*/, v[178:185] /*v[434:441]*/, v[114:121] /*v[370:377]*/
	s_set_vgpr_msb 0x5556
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, v[218:219] /*v[730:731]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[100:107] /*v[612:619]*/, v[186:193] /*v[442:449]*/, v[106:113] /*v[362:369]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[190:191] /*v[446:447]*/, v[6:7] /*v[518:519]*/
	v_mov_b64_e32 v[192:193] /*v[448:449]*/, v[8:9] /*v[520:521]*/
	v_mov_b64_e32 v[186:187] /*v[442:443]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x5655
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[250:257] /*v[506:513]*/, v[242:249] /*v[498:505]*/, v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x555a
	v_mov_b64_e32 v[188:189] /*v[444:445]*/, v[4:5] /*v[516:517]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[246:247] /*v[502:503]*/, v[136:137] /*v[648:649]*/
	v_mov_b64_e32 v[248:249] /*v[504:505]*/, v[138:139] /*v[650:651]*/
	v_mov_b64_e32 v[242:243] /*v[498:499]*/, v[132:133] /*v[644:645]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[100:107] /*v[612:619]*/, v[52:59] /*v[564:571]*/, v[74:81] /*v[330:337]*/
	v_mov_b64_e32 v[244:245] /*v[500:501]*/, v[134:135] /*v[646:647]*/
	v_mov_b64_e32 v[254:255] /*v[510:511]*/, v[128:129] /*v[640:641]*/
	s_set_vgpr_msb 0x5a82
	v_mov_b64_e32 v[0:1] /*v[512:513]*/, v[130:131] /*v[642:643]*/
	s_set_vgpr_msb 0x825a
	v_mov_b64_e32 v[250:251] /*v[506:507]*/, v[124:125] /*v[636:637]*/
	v_mov_b64_e32 v[252:253] /*v[508:509]*/, v[126:127] /*v[638:639]*/
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[100:107] /*v[612:619]*/, v[60:67] /*v[572:579]*/, v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5a0a
	v_wmma_f32_16x16x32_bf16 v[218:225], v[100:107] /*v[612:619]*/, v[68:75] /*v[580:587]*/, v[218:225]
	v_wmma_f32_16x16x32_bf16 v[202:209], v[100:107] /*v[612:619]*/, v[76:83] /*v[588:595]*/, v[202:209]
	v_wmma_f32_16x16x32_bf16 v[138:145], v[100:107] /*v[612:619]*/, v[84:91] /*v[596:603]*/, v[138:145]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[122:129], v[108:115] /*v[620:627]*/, v[170:177] /*v[426:433]*/, v[122:129]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[22:23] /*v[534:535]*/
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[24:25] /*v[536:537]*/
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x4206
	v_wmma_f32_16x16x32_bf16 v[114:121], v[116:123] /*v[628:635]*/, v[178:185] /*v[434:441]*/, v[114:121]
	s_set_vgpr_msb 0x642
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[20:21] /*v[532:533]*/
	v_nop
	v_nop
	v_nop
	v_mov_b64_e32 v[182:183] /*v[438:439]*/, v[14:15] /*v[526:527]*/
	v_mov_b64_e32 v[184:185] /*v[440:441]*/, v[16:17] /*v[528:529]*/
	v_mov_b64_e32 v[178:179] /*v[434:435]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x420a
	v_wmma_f32_16x16x32_bf16 v[2:9], v[116:123] /*v[628:635]*/, v[92:99] /*v[604:611]*/, v[2:9]
	s_set_vgpr_msb 0xa42
	v_mov_b64_e32 v[180:181] /*v[436:437]*/, v[12:13] /*v[524:525]*/
	s_set_vgpr_msb 0x4200
	s_cbranch_scc1 .LBB0_6
.LBB0_7:
	s_clause 0x1
	s_load_b64 s[6:7], s[0:1], 0x110 nv
	s_load_b64 s[8:9], s[0:1], 0x140 nv
	s_set_vgpr_msb 12
	v_or_b32_e32 v147, 1, v145 /*v913*/
	v_mul_lo_u32 v146, s40, v145 /*v913*/
	s_mul_i32 s4, s40, s42
	s_lshl_b32 s1, s34, 25
	s_add_co_i32 s4, s4, s66
	v_mul_lo_u32 v147, v147, s40
	s_mov_b32 s0, 0
	v_mul_lo_u32 v153, s40, v140 /*v908*/
	v_mul_lo_u32 v155, s40, v139 /*v907*/
	v_add_lshl_u32 v146, v146, s4, 7
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v148, v122 /*v378*/, s0
	v_cvt_pk_bf16_f32 v150, v114 /*v370*/, s0
	s_mov_b32 s2, s46
	v_add_lshl_u32 v147, s4, v147, 7
	s_set_vgpr_msb 0x10c
	v_or_b32_e32 v149, v146, v141 /*v909*/
	s_mov_b32 s3, s47
	v_add_lshl_u32 v153, v153, s4, 7
	v_mul_lo_u32 v156, s40, v138 /*v906*/
	v_or_b32_e32 v151, v147, v141 /*v909*/
	s_wait_kmcnt 0x0
	s_or_b64 s[44:45], s[6:7], s[0:1]
	s_or_b64 s[0:1], s[8:9], s[0:1]
	s_set_vgpr_msb 0xc01
	v_lshlrev_b32_e32 v149, 2, v149
	v_cvt_pk_bf16_f32 v154, v115 /*v371*/, s0
	v_cvt_pk_bf16_f32 v152, v123 /*v379*/, s0
	v_lshlrev_b32_e32 v151, 2, v151
	s_set_vgpr_msb 0x10c
	v_mul_lo_u32 v159, s40, v137 /*v905*/
	buffer_store_b16 v148, v149, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v148.l, v154.l
	buffer_store_b16 v150, v149, s[0:3], null offen
	buffer_store_b16 v152, v151, s[44:47], null offen
	s_wait_xcnt 0x1
	v_add_lshl_u32 v150, v155, s4, 7
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v152, v124 /*v380*/, s0
	v_cvt_pk_bf16_f32 v154, v116 /*v372*/, s0
	s_set_vgpr_msb 0x10c
	buffer_store_b16 v148, v151, s[0:3], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v148, v153, v141 /*v909*/
	v_or_b32_e32 v155, v150, v141 /*v909*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v157, v125 /*v381*/, s0
	v_cvt_pk_bf16_f32 v158, v117 /*v373*/, s0
	v_add_lshl_u32 v156, s4, v156, 7
	v_dual_lshlrev_b32 v148, 2, v148 :: v_dual_lshlrev_b32 v155, 2, v155
	s_set_vgpr_msb 0x10d
	v_mul_lo_u32 v160, s40, v136 /*v904*/
	v_mul_lo_u32 v163, s40, v135 /*v903*/
	v_cvt_pk_bf16_f32 v162, v119 /*v375*/, s0
	s_set_vgpr_msb 0xd00
	buffer_store_b16 v152, v148, s[44:47], null offen
	buffer_store_b16 v154, v148, s[0:3], null offen
	buffer_store_b16 v157, v155, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v157, s4, v159, 7
	v_mov_b16_e64 v152.l, v158.l
	s_set_vgpr_msb 12
	v_or_b32_e32 v154, v156, v141 /*v909*/
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v158, v118 /*v374*/, s0
	v_add_lshl_u32 v160, s4, v160, 7
	s_set_vgpr_msb 0x10c
	v_or_b32_e32 v159, v157, v141 /*v909*/
	buffer_store_b16 v152, v155, s[0:3], null offen
	s_set_vgpr_msb 0xc01
	v_cvt_pk_bf16_f32 v152, v126 /*v382*/, s0
	v_lshlrev_b32_e32 v154, 2, v154
	v_cvt_pk_bf16_f32 v161, v127 /*v383*/, s0
	s_set_vgpr_msb 0x100
	v_dual_lshlrev_b32 v159, 2, v159 :: v_dual_bitop2_b32 v150, v150, v0 bitop3:0x54
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v165, v121 /*v377*/, s0
	s_set_vgpr_msb 0x10c
	buffer_store_b16 v152, v154, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v152.l, v162.l
	buffer_store_b16 v158, v154, s[0:3], null offen
	buffer_store_b16 v161, v159, s[44:47], null offen
	s_wait_xcnt 0x1
	v_or_b32_e32 v158, v160, v141 /*v909*/
	s_wait_xcnt 0x0
	v_add_lshl_u32 v161, v163, s4, 7
	s_set_vgpr_msb 0xc00
	v_dual_lshlrev_b32 v150, 2, v150 :: v_dual_bitop2_b32 v147, v147, v0 bitop3:0x54
	buffer_store_b16 v152, v159, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v152, v128 /*v384*/, s0
	s_set_vgpr_msb 0x10c
	v_or_b32_e32 v163, v161, v141 /*v909*/
	s_set_vgpr_msb 0xc00
	v_dual_lshlrev_b32 v158, 2, v158 :: v_dual_bitop2_b32 v146, v146, v0 bitop3:0x54
	v_lshlrev_b32_e32 v147, 2, v147
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v162, v120 /*v376*/, s0
	v_lshlrev_b32_e32 v163, 2, v163
	s_set_vgpr_msb 0x100
	buffer_store_b16 v152, v158, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v152.l, v165.l
	v_dual_lshlrev_b32 v146, 2, v146 :: v_dual_bitop2_b32 v167, 64, v147 bitop3:0x54
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v164, v129 /*v385*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v162, v158, s[0:3], null offen
	buffer_store_b16 v164, v163, s[44:47], null offen
	buffer_store_b16 v152, v163, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v152, v106 /*v362*/, s0
	v_or_b32_e32 v164, 64, v146
	v_cvt_pk_bf16_f32 v166, v99 /*v355*/, s0
	s_set_vgpr_msb 0x100
	v_or_b32_e32 v157, v157, v0
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v162, v98 /*v354*/, s0
	v_cvt_pk_bf16_f32 v165, v107 /*v363*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v152, v164, s[44:47], null offen
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v157, 2, v157 :: v_dual_bitop2_b32 v152, v153, v0 bitop3:0x54
	v_mov_b16_e64 v153.l, v166.l
	v_or_b32_e32 v156, v156, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_lshlrev_b32 v152, 2, v152 :: v_dual_bitop2_b32 v166, 64, v150 bitop3:0x54
	buffer_store_b16 v162, v164, s[0:3], null offen
	buffer_store_b16 v165, v167, s[44:47], null offen
	buffer_store_b16 v153, v167, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v153, v108 /*v364*/, s0
	v_cvt_pk_bf16_f32 v167, v101 /*v357*/, s0
	v_or_b32_e32 v164, 64, v152
	v_cvt_pk_bf16_f32 v162, v100 /*v356*/, s0
	v_cvt_pk_bf16_f32 v165, v109 /*v365*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v153, v164, s[44:47], null offen
	buffer_store_b16 v162, v164, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v153.l, v167.l
	v_lshlrev_b32_e32 v156, 2, v156
	buffer_store_b16 v165, v166, s[44:47], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v162, v110 /*v366*/, s0
	s_set_vgpr_msb 0x100
	v_or_b32_e32 v160, v160, v0
	buffer_store_b16 v153, v166, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v153, v102 /*v358*/, s0
	v_or_b32_e32 v164, 64, v156
	v_cvt_pk_bf16_f32 v166, v103 /*v359*/, s0
	v_cvt_pk_bf16_f32 v165, v111 /*v367*/, s0
	v_or_b32_e32 v167, 64, v157
	v_cvt_pk_bf16_f32 v169, v37 /*v293*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v162, v164, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e64 v162.l, v166.l
	buffer_store_b16 v153, v164, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v153, 2, v160
	v_or_b32_e32 v160, v161, v0
	buffer_store_b16 v165, v167, s[44:47], null offen
	buffer_store_b16 v162, v167, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v161, v112 /*v368*/, s0
	v_dual_lshlrev_b32 v160, 2, v160 :: v_dual_bitop2_b32 v162, 64, v153 bitop3:0x54
	v_cvt_pk_bf16_f32 v165, v113 /*v369*/, s0
	v_cvt_pk_bf16_f32 v164, v104 /*v360*/, s0
	v_cvt_pk_bf16_f32 v166, v105 /*v361*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v162, s[44:47], null offen
	buffer_store_b16 v164, v162, s[0:3], null offen
	v_or_b32_e32 v167, 64, v160
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v161, v90 /*v346*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v165, v167, s[44:47], null offen
	buffer_store_b16 v166, v167, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v165, v83 /*v339*/, s0
	v_cvt_pk_bf16_f32 v166, v92 /*v348*/, s0
	v_cvt_pk_bf16_f32 v162, v82 /*v338*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v149, s[44:47], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v164, v91 /*v347*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v161.l, v165.l
	v_mov_b16_e64 v165.l, v166.l
	buffer_store_b16 v162, v149, s[0:3], null offen offset:128
	buffer_store_b16 v164, v151, s[44:47], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v162, v84 /*v340*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v151, s[0:3], null offen offset:128
	buffer_store_b16 v165, v148, s[44:47], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v165, v94 /*v350*/, s0
	v_cvt_pk_bf16_f32 v166, v86 /*v342*/, s0
	v_cvt_pk_bf16_f32 v161, v93 /*v349*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v162, v148, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v164, v85 /*v341*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v162.l, v165.l
	v_mov_b16_e64 v165.l, v166.l
	buffer_store_b16 v161, v155, s[44:47], null offen offset:128
	buffer_store_b16 v164, v155, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v161, v95 /*v351*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v162, v154, s[44:47], null offen offset:128
	buffer_store_b16 v165, v154, s[0:3], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v165, v88 /*v344*/, s0
	v_cvt_pk_bf16_f32 v166, v97 /*v353*/, s0
	v_cvt_pk_bf16_f32 v162, v87 /*v343*/, s0
	v_cvt_pk_bf16_f32 v164, v96 /*v352*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v159, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e64 v161.l, v165.l
	v_mov_b16_e64 v165.l, v166.l
	buffer_store_b16 v162, v159, s[0:3], null offen offset:128
	buffer_store_b16 v164, v158, s[44:47], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v162, v89 /*v345*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v158, s[0:3], null offen offset:128
	buffer_store_b16 v165, v163, s[44:47], null offen offset:128
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v161, v74 /*v330*/, s0
	v_cvt_pk_bf16_f32 v164, v34 /*v290*/, s0
	v_or_b32_e32 v166, 0xc0, v146
	v_cvt_pk_bf16_f32 v165, v75 /*v331*/, s0
	v_or_b32_e32 v167, 0xc0, v147
	s_set_vgpr_msb 0x100
	buffer_store_b16 v162, v163, s[0:3], null offen offset:128
	buffer_store_b16 v161, v166, s[44:47], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v161, v35 /*v291*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v164, v166, s[0:3], null offen
	buffer_store_b16 v165, v167, s[44:47], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v162, v76 /*v332*/, s0
	v_or_b32_e32 v166, 0xc0, v152
	v_cvt_pk_bf16_f32 v164, v36 /*v292*/, s0
	v_cvt_pk_bf16_f32 v165, v77 /*v333*/, s0
	v_or_b32_e32 v168, 0xc0, v150
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v167, s[0:3], null offen
	buffer_store_b16 v162, v166, s[44:47], null offen
	buffer_store_b16 v164, v166, s[0:3], null offen
	buffer_store_b16 v165, v168, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e64 v161.l, v169.l
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v162, v78 /*v334*/, s0
	v_or_b32_e32 v165, 0xc0, v156
	v_cvt_pk_bf16_f32 v164, v38 /*v294*/, s0
	v_cvt_pk_bf16_f32 v166, v79 /*v335*/, s0
	v_or_b32_e32 v169, 0xc0, v157
	v_cvt_pk_bf16_f32 v167, v39 /*v295*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v168, s[0:3], null offen
	buffer_store_b16 v162, v165, s[44:47], null offen
	buffer_store_b16 v164, v165, s[0:3], null offen
	buffer_store_b16 v166, v169, s[44:47], null offen
	buffer_store_b16 v167, v169, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v161, v80 /*v336*/, s0
	v_or_b32_e32 v164, 0xc0, v153
	v_cvt_pk_bf16_f32 v165, v81 /*v337*/, s0
	v_or_b32_e32 v167, 0xc0, v160
	v_cvt_pk_bf16_f32 v162, v40 /*v296*/, s0
	v_cvt_pk_bf16_f32 v166, v41 /*v297*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v164, s[44:47], null offen
	buffer_store_b16 v162, v164, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v161, v26 /*v282*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v165, v167, s[44:47], null offen
	buffer_store_b16 v166, v167, s[0:3], null offen
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v165, v19 /*v275*/, s0
	v_cvt_pk_bf16_f32 v166, v28 /*v284*/, s0
	v_cvt_pk_bf16_f32 v162, v18 /*v274*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v149, s[44:47], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v164, v27 /*v283*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v161.l, v165.l
	v_mov_b16_e64 v165.l, v166.l
	buffer_store_b16 v162, v149, s[0:3], null offen offset:256
	buffer_store_b16 v164, v151, s[44:47], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v162, v20 /*v276*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v151, s[0:3], null offen offset:256
	buffer_store_b16 v165, v148, s[44:47], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v165, v30 /*v286*/, s0
	v_cvt_pk_bf16_f32 v166, v22 /*v278*/, s0
	v_cvt_pk_bf16_f32 v161, v29 /*v285*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v162, v148, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v164, v21 /*v277*/, s0
	s_set_vgpr_msb 0x100
	v_mov_b16_e64 v162.l, v165.l
	v_mov_b16_e64 v165.l, v166.l
	buffer_store_b16 v161, v155, s[44:47], null offen offset:256
	buffer_store_b16 v164, v155, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v161, v31 /*v287*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v162, v154, s[44:47], null offen offset:256
	buffer_store_b16 v165, v154, s[0:3], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v165, v24 /*v280*/, s0
	v_cvt_pk_bf16_f32 v166, v33 /*v289*/, s0
	v_cvt_pk_bf16_f32 v162, v23 /*v279*/, s0
	v_cvt_pk_bf16_f32 v164, v32 /*v288*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v159, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e64 v161.l, v165.l
	v_mov_b16_e64 v165.l, v166.l
	buffer_store_b16 v162, v159, s[0:3], null offen offset:256
	buffer_store_b16 v164, v158, s[44:47], null offen offset:256
	s_set_vgpr_msb 1
	v_cvt_pk_bf16_f32 v162, v25 /*v281*/, s0
	s_set_vgpr_msb 0x100
	buffer_store_b16 v161, v158, s[0:3], null offen offset:256
	buffer_store_b16 v165, v163, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v161, v218, s0
	v_cvt_pk_bf16_f32 v164, v210, s0
	v_or_b32_e32 v166, 0x140, v146
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v165, v219, s0
	v_or_b32_e32 v167, 0x140, v147
	buffer_store_b16 v162, v163, s[0:3], null offen offset:256
	buffer_store_b16 v161, v166, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v161, v211, s0
	v_cvt_pk_bf16_f32 v169, v213, s0
	buffer_store_b16 v164, v166, s[0:3], null offen
	buffer_store_b16 v165, v167, s[44:47], null offen
	v_cvt_pk_bf16_f32 v162, v220, s0
	s_wait_xcnt 0x1
	v_or_b32_e32 v166, 0x140, v152
	v_cvt_pk_bf16_f32 v164, v212, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v165, v221, s0
	v_or_b32_e32 v168, 0x140, v150
	buffer_store_b16 v161, v167, s[0:3], null offen
	buffer_store_b16 v162, v166, s[44:47], null offen
	buffer_store_b16 v164, v166, s[0:3], null offen
	buffer_store_b16 v165, v168, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e64 v161.l, v169.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v162, v222, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v165, 0x140, v156
	v_cvt_pk_bf16_f32 v164, v214, s0
	v_cvt_pk_bf16_f32 v166, v223, s0
	v_or_b32_e32 v169, 0x140, v157
	v_cvt_pk_bf16_f32 v167, v215, s0
	buffer_store_b16 v161, v168, s[0:3], null offen
	buffer_store_b16 v162, v165, s[44:47], null offen
	buffer_store_b16 v164, v165, s[0:3], null offen
	buffer_store_b16 v166, v169, s[44:47], null offen
	buffer_store_b16 v167, v169, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v161, v224, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v164, 0x140, v153
	v_cvt_pk_bf16_f32 v165, v225, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v167, 0x140, v160
	v_cvt_pk_bf16_f32 v162, v216, s0
	v_cvt_pk_bf16_f32 v166, v217, s0
	buffer_store_b16 v161, v164, s[44:47], null offen
	buffer_store_b16 v162, v164, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v161, v202, s0
	buffer_store_b16 v165, v167, s[44:47], null offen
	buffer_store_b16 v166, v167, s[0:3], null offen
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v162, v194, s0
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v165, v195, s0
	v_cvt_pk_bf16_f32 v164, v203, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v166, v204, s0
	buffer_store_b16 v161, v149, s[44:47], null offen offset:384
	buffer_store_b16 v162, v149, s[0:3], null offen offset:384
	buffer_store_b16 v164, v151, s[44:47], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e64 v161.l, v165.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v149, v196, s0
	v_cvt_pk_bf16_f32 v162, v206, s0
	v_mov_b16_e64 v165.l, v166.l
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v164, v198, s0
	buffer_store_b16 v161, v151, s[0:3], null offen offset:384
	buffer_store_b16 v165, v148, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v151, v205, s0
	buffer_store_b16 v149, v148, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e64 v148.l, v162.l
	v_cvt_pk_bf16_f32 v161, v197, s0
	v_mov_b16_e64 v149.l, v164.l
	buffer_store_b16 v151, v155, s[44:47], null offen offset:384
	buffer_store_b16 v161, v155, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v151, v207, s0
	buffer_store_b16 v148, v154, s[44:47], null offen offset:384
	buffer_store_b16 v149, v154, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v148, v199, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v154, v200, s0
	v_cvt_pk_bf16_f32 v149, v208, s0
	v_cvt_pk_bf16_f32 v155, v209, s0
	buffer_store_b16 v151, v159, s[44:47], null offen offset:384
	buffer_store_b16 v148, v159, s[0:3], null offen offset:384
	buffer_store_b16 v149, v158, s[44:47], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e64 v151.l, v154.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v148, v201, s0
	v_cvt_pk_bf16_f32 v130, v130, s0
	v_or_b32_e32 v146, 0x1c0, v146
	v_mov_b16_e64 v154.l, v155.l
	v_cvt_pk_bf16_f32 v138, v138, s0
	v_cvt_pk_bf16_f32 v139, v139, s0
	v_or_b32_e32 v147, 0x1c0, v147
	v_cvt_pk_bf16_f32 v131, v131, s0
	buffer_store_b16 v151, v158, s[0:3], null offen offset:384
	buffer_store_b16 v154, v163, s[44:47], null offen offset:384
	buffer_store_b16 v148, v163, s[0:3], null offen offset:384
	buffer_store_b16 v138, v146, s[44:47], null offen
	buffer_store_b16 v130, v146, s[0:3], null offen
	buffer_store_b16 v139, v147, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v130, v140, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v139, 0x1c0, v152
	v_cvt_pk_bf16_f32 v132, v132, s0
	v_or_b32_e32 v140, 0x1c0, v150
	buffer_store_b16 v131, v147, s[0:3], null offen
	buffer_store_b16 v130, v139, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v130, v133, s0
	v_cvt_pk_bf16_f32 v135, v135, s0
	v_cvt_pk_bf16_f32 v138, v141, s0
	v_cvt_pk_bf16_f32 v131, v142, s0
	v_or_b32_e32 v133, 0x1c0, v156
	buffer_store_b16 v132, v139, s[0:3], null offen
	buffer_store_b16 v138, v140, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v132, v134, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v138, 0x1c0, v157
	buffer_store_b16 v130, v140, s[0:3], null offen
	buffer_store_b16 v131, v133, s[44:47], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v130.l, v135.l
	v_cvt_pk_bf16_f32 v134, v143, s0
	buffer_store_b16 v132, v133, s[0:3], null offen
	buffer_store_b16 v134, v138, s[44:47], null offen
	buffer_store_b16 v130, v138, s[0:3], null offen
	s_set_vgpr_msb 12
	v_mul_lo_u32 v130, s40, v142 /*v910*/
	v_or_b32_e32 v134, 1, v142 /*v910*/
	v_cvt_pk_bf16_f32 v131, v144, s0
	s_set_vgpr_msb 0xc00
	v_or_b32_e32 v133, 0x1c0, v153
	v_cvt_pk_bf16_f32 v137, v137, s0
	v_cvt_pk_bf16_f32 v132, v136, s0
	v_mul_lo_u32 v134, s40, v134
	v_cvt_pk_bf16_f32 v135, v145, s0
	v_add_lshl_u32 v130, s4, v130, 7
	v_or_b32_e32 v136, 0x1c0, v160
	buffer_store_b16 v131, v133, s[44:47], null offen
	buffer_store_b16 v132, v133, s[0:3], null offen
	s_wait_xcnt 0x1
	v_mov_b16_e64 v131.l, v137.l
	v_cvt_pk_bf16_f32 v122, v122, s0
	s_set_vgpr_msb 12
	v_or_b32_e32 v132, v130, v141 /*v909*/
	v_add_lshl_u32 v133, v134, s4, 7
	buffer_store_b16 v135, v136, s[44:47], null offen
	buffer_store_b16 v131, v136, s[0:3], null offen
	v_mul_lo_u32 v134, s40, v134 /*v902*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v131, 2, v132
	s_set_vgpr_msb 12
	v_or_b32_e32 v132, v133, v141 /*v909*/
	v_mul_lo_u32 v135, s40, v133 /*v901*/
	v_cvt_pk_bf16_f32 v114, v114, s0
	v_cvt_pk_bf16_f32 v123, v123, s0
	v_cvt_pk_bf16_f32 v115, v115, s0
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v132, 2, v132
	buffer_store_b16 v122, v131, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v122, s4, v134, 7
	buffer_store_b16 v114, v131, s[0:3], null offen
	buffer_store_b16 v123, v132, s[44:47], null offen
	s_wait_xcnt 0x1
	v_add_lshl_u32 v114, s4, v135, 7
	buffer_store_b16 v115, v132, s[0:3], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v115, v122, v141 /*v909*/
	v_cvt_pk_bf16_f32 v123, v124, s0
	v_mul_lo_u32 v134, s40, v132 /*v900*/
	v_or_b32_e32 v124, v114, v141 /*v909*/
	v_mul_lo_u32 v135, s40, v131 /*v899*/
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v115, 2, v115
	v_cvt_pk_bf16_f32 v116, v116, s0
	v_cvt_pk_bf16_f32 v125, v125, s0
	v_lshlrev_b32_e32 v124, 2, v124
	v_cvt_pk_bf16_f32 v117, v117, s0
	v_add_lshl_u32 v134, s4, v134, 7
	buffer_store_b16 v123, v115, s[44:47], null offen
	s_wait_xcnt 0x0
	v_add_lshl_u32 v123, s4, v135, 7
	buffer_store_b16 v116, v115, s[0:3], null offen
	buffer_store_b16 v125, v124, s[44:47], null offen
	buffer_store_b16 v117, v124, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v117, v126, s0
	s_set_vgpr_msb 12
	v_mul_lo_u32 v126, s40, v130 /*v898*/
	v_mul_lo_u32 v1, v1, s40
	v_or_b32_e32 v116, v134, v141 /*v909*/
	v_or_b32_e32 v125, v123, v141 /*v909*/
	v_cvt_pk_bf16_f32 v118, v118, s0
	v_cvt_pk_bf16_f32 v119, v119, s0
	v_cvt_pk_bf16_f32 v127, v127, s0
	s_set_vgpr_msb 0xc00
	v_lshlrev_b32_e32 v116, 2, v116
	v_add_lshl_u32 v126, s4, v126, 7
	v_lshlrev_b32_e32 v125, 2, v125
	v_add_lshl_u32 v1, s4, v1, 7
	v_cvt_pk_bf16_f32 v121, v121, s0
	buffer_store_b16 v117, v116, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v117.l, v119.l
	buffer_store_b16 v118, v116, s[0:3], null offen
	buffer_store_b16 v127, v125, s[44:47], null offen
	s_set_vgpr_msb 12
	v_or_b32_e32 v118, v126, v141 /*v909*/
	v_cvt_pk_bf16_f32 v119, v120, s0
	v_or_b32_e32 v120, v1, v141 /*v909*/
	buffer_store_b16 v117, v125, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v117, v128, s0
	s_set_vgpr_msb 0xc00
	v_dual_lshlrev_b32 v118, 2, v118 :: v_dual_bitop2_b32 v128, v130, v0 bitop3:0x54
	v_cvt_pk_bf16_f32 v127, v129, s0
	v_lshlrev_b32_e32 v120, 2, v120
	v_cvt_pk_bf16_f32 v106, v106, s0
	buffer_store_b16 v117, v118, s[44:47], null offen
	s_wait_xcnt 0x0
	v_mov_b16_e32 v117.l, v121.l
	buffer_store_b16 v119, v118, s[0:3], null offen
	buffer_store_b16 v127, v120, s[44:47], null offen
	s_wait_xcnt 0x1
	v_lshlrev_b32_e32 v119, 2, v128
	v_or_b32_e32 v121, v133, v0
	v_cvt_pk_bf16_f32 v98, v98, s0
	buffer_store_b16 v117, v120, s[0:3], null offen
	v_cvt_pk_bf16_f32 v107, v107, s0
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v121, 2, v121 :: v_dual_bitop2_b32 v117, 64, v119 bitop3:0x54
	v_cvt_pk_bf16_f32 v99, v99, s0
	v_cvt_pk_bf16_f32 v101, v101, s0
	v_cvt_pk_bf16_f32 v100, v100, s0
	buffer_store_b16 v106, v117, s[44:47], null offen
	s_wait_xcnt 0x0
	v_or_b32_e32 v106, v122, v0
	v_or_b32_e32 v127, 64, v121
	buffer_store_b16 v98, v117, s[0:3], null offen
	buffer_store_b16 v107, v127, s[44:47], null offen
	s_wait_xcnt 0x1
	v_dual_lshlrev_b32 v106, 2, v106 :: v_dual_bitop2_b32 v98, v114, v0 bitop3:0x54
	buffer_store_b16 v99, v127, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v99, v108, s0
	v_dual_lshlrev_b32 v98, 2, v98 :: v_dual_bitop2_b32 v114, v134, v0 bitop3:0x54
	v_or_b32_e32 v107, 64, v106
	v_cvt_pk_bf16_f32 v108, v109, s0
	buffer_store_b16 v99, v107, s[44:47], null offen
	buffer_store_b16 v100, v107, s[0:3], null offen
	v_or_b32_e32 v109, 64, v98
	s_wait_xcnt 0x1
	v_mov_b16_e32 v99.l, v101.l
	s_wait_xcnt 0x0
	v_dual_lshlrev_b32 v100, 2, v114 :: v_dual_bitop2_b32 v101, v123, v0 bitop3:0x54
	v_cvt_pk_bf16_f32 v107, v110, s0
	buffer_store_b16 v108, v109, s[44:47], null offen
	buffer_store_b16 v99, v109, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v99, v102, s0
	v_or_b32_e32 v102, 64, v100
	v_dual_lshlrev_b32 v101, 2, v101 :: v_dual_bitop2_b32 v110, v126, v0 bitop3:0x54
	v_or_b32_e32 v0, v1, v0
	v_cvt_pk_bf16_f32 v108, v111, s0
	buffer_store_b16 v107, v102, s[44:47], null offen
	buffer_store_b16 v99, v102, s[0:3], null offen
	s_wait_xcnt 0x0
	v_lshlrev_b32_e32 v99, 2, v110
	v_cvt_pk_bf16_f32 v103, v103, s0
	v_dual_lshlrev_b32 v0, 2, v0 :: v_dual_bitop2_b32 v109, 64, v101 bitop3:0x54
	v_cvt_pk_bf16_f32 v1, v112, s0
	s_delay_alu instid0(VALU_DEP_4)
	v_or_b32_e32 v102, 64, v99
	v_cvt_pk_bf16_f32 v83, v83, s0
	buffer_store_b16 v108, v109, s[44:47], null offen
	buffer_store_b16 v103, v109, s[0:3], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v103, v104, s0
	v_cvt_pk_bf16_f32 v104, v113, s0
	v_or_b32_e32 v107, 64, v0
	buffer_store_b16 v1, v102, s[44:47], null offen
	buffer_store_b16 v103, v102, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v90, s0
	v_cvt_pk_bf16_f32 v105, v105, s0
	v_cvt_pk_bf16_f32 v90, v91, s0
	v_cvt_pk_bf16_f32 v91, v92, s0
	v_cvt_pk_bf16_f32 v82, v82, s0
	buffer_store_b16 v104, v107, s[44:47], null offen
	buffer_store_b16 v105, v107, s[0:3], null offen
	buffer_store_b16 v1, v131, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v83.l
	v_mov_b16_e32 v83.l, v91.l
	buffer_store_b16 v82, v131, s[0:3], null offen offset:128
	buffer_store_b16 v90, v132, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v82, v84, s0
	v_cvt_pk_bf16_f32 v84, v94, s0
	buffer_store_b16 v1, v132, s[0:3], null offen offset:128
	buffer_store_b16 v83, v115, s[44:47], null offen offset:128
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v83, v85, s0
	v_cvt_pk_bf16_f32 v85, v86, s0
	v_cvt_pk_bf16_f32 v1, v93, s0
	buffer_store_b16 v82, v115, s[0:3], null offen offset:128
	s_wait_xcnt 0x0
	v_mov_b16_e32 v82.l, v84.l
	buffer_store_b16 v1, v124, s[44:47], null offen offset:128
	buffer_store_b16 v83, v124, s[0:3], null offen offset:128
	v_mov_b16_e32 v84.l, v85.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v95, s0
	buffer_store_b16 v82, v116, s[44:47], null offen offset:128
	buffer_store_b16 v84, v116, s[0:3], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v82, v87, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v84, v88, s0
	v_cvt_pk_bf16_f32 v83, v96, s0
	v_cvt_pk_bf16_f32 v85, v97, s0
	buffer_store_b16 v1, v125, s[44:47], null offen offset:128
	buffer_store_b16 v82, v125, s[0:3], null offen offset:128
	buffer_store_b16 v83, v118, s[44:47], null offen offset:128
	s_wait_xcnt 0x2
	v_mov_b16_e32 v1.l, v84.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v82, v89, s0
	v_mov_b16_e32 v84.l, v85.l
	buffer_store_b16 v1, v118, s[0:3], null offen offset:128
	buffer_store_b16 v84, v120, s[44:47], null offen offset:128
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v74, s0
	v_mov_b16_e32 v74.l, v82.l
	v_cvt_pk_bf16_f32 v66, v66, s0
	v_or_b32_e32 v82, 0xc0, v119
	v_cvt_pk_bf16_f32 v75, v75, s0
	v_or_b32_e32 v83, 0xc0, v121
	buffer_store_b16 v74, v120, s[0:3], null offen offset:128
	buffer_store_b16 v1, v82, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v67, s0
	v_cvt_pk_bf16_f32 v69, v69, s0
	buffer_store_b16 v66, v82, s[0:3], null offen
	buffer_store_b16 v75, v83, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v66, v76, s0
	v_or_b32_e32 v74, 0xc0, v106
	v_cvt_pk_bf16_f32 v67, v68, s0
	v_cvt_pk_bf16_f32 v68, v77, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v75, 0xc0, v98
	buffer_store_b16 v1, v83, s[0:3], null offen
	buffer_store_b16 v66, v74, s[44:47], null offen
	buffer_store_b16 v67, v74, s[0:3], null offen
	buffer_store_b16 v68, v75, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v1.l, v69.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v66, v78, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v68, 0xc0, v100
	v_cvt_pk_bf16_f32 v67, v70, s0
	v_cvt_pk_bf16_f32 v69, v79, s0
	v_cvt_pk_bf16_f32 v70, v71, s0
	v_or_b32_e32 v71, 0xc0, v101
	buffer_store_b16 v1, v75, s[0:3], null offen
	buffer_store_b16 v66, v68, s[44:47], null offen
	buffer_store_b16 v67, v68, s[0:3], null offen
	buffer_store_b16 v69, v71, s[44:47], null offen
	buffer_store_b16 v70, v71, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v1, v80, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v67, 0xc0, v99
	v_cvt_pk_bf16_f32 v66, v72, s0
	v_cvt_pk_bf16_f32 v68, v81, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v70, 0xc0, v0
	buffer_store_b16 v1, v67, s[44:47], null offen
	buffer_store_b16 v66, v67, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v58, s0
	v_cvt_pk_bf16_f32 v51, v51, s0
	v_cvt_pk_bf16_f32 v69, v73, s0
	v_cvt_pk_bf16_f32 v58, v59, s0
	v_cvt_pk_bf16_f32 v59, v60, s0
	v_cvt_pk_bf16_f32 v50, v50, s0
	buffer_store_b16 v68, v70, s[44:47], null offen
	buffer_store_b16 v69, v70, s[0:3], null offen
	buffer_store_b16 v1, v131, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v51.l
	v_mov_b16_e32 v51.l, v59.l
	buffer_store_b16 v50, v131, s[0:3], null offen offset:256
	buffer_store_b16 v58, v132, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v50, v52, s0
	v_cvt_pk_bf16_f32 v52, v62, s0
	buffer_store_b16 v1, v132, s[0:3], null offen offset:256
	buffer_store_b16 v51, v115, s[44:47], null offen offset:256
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v51, v53, s0
	v_cvt_pk_bf16_f32 v53, v54, s0
	v_cvt_pk_bf16_f32 v1, v61, s0
	buffer_store_b16 v50, v115, s[0:3], null offen offset:256
	s_wait_xcnt 0x0
	v_mov_b16_e32 v50.l, v52.l
	buffer_store_b16 v1, v124, s[44:47], null offen offset:256
	buffer_store_b16 v51, v124, s[0:3], null offen offset:256
	v_mov_b16_e32 v52.l, v53.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v63, s0
	buffer_store_b16 v50, v116, s[44:47], null offen offset:256
	buffer_store_b16 v52, v116, s[0:3], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v50, v55, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v52, v56, s0
	v_cvt_pk_bf16_f32 v51, v64, s0
	v_cvt_pk_bf16_f32 v53, v65, s0
	buffer_store_b16 v1, v125, s[44:47], null offen offset:256
	buffer_store_b16 v50, v125, s[0:3], null offen offset:256
	buffer_store_b16 v51, v118, s[44:47], null offen offset:256
	s_wait_xcnt 0x2
	v_mov_b16_e32 v1.l, v52.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v50, v57, s0
	v_mov_b16_e32 v52.l, v53.l
	buffer_store_b16 v1, v118, s[0:3], null offen offset:256
	buffer_store_b16 v52, v120, s[44:47], null offen offset:256
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v42, s0
	v_mov_b16_e32 v42.l, v50.l
	v_cvt_pk_bf16_f32 v34, v34, s0
	v_or_b32_e32 v50, 0x140, v119
	v_cvt_pk_bf16_f32 v43, v43, s0
	v_or_b32_e32 v51, 0x140, v121
	buffer_store_b16 v42, v120, s[0:3], null offen offset:256
	buffer_store_b16 v1, v50, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v35, s0
	v_cvt_pk_bf16_f32 v37, v37, s0
	buffer_store_b16 v34, v50, s[0:3], null offen
	buffer_store_b16 v43, v51, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v34, v44, s0
	v_or_b32_e32 v42, 0x140, v106
	v_cvt_pk_bf16_f32 v35, v36, s0
	v_cvt_pk_bf16_f32 v36, v45, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v43, 0x140, v98
	buffer_store_b16 v1, v51, s[0:3], null offen
	buffer_store_b16 v34, v42, s[44:47], null offen
	buffer_store_b16 v35, v42, s[0:3], null offen
	buffer_store_b16 v36, v43, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v1.l, v37.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v34, v46, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v36, 0x140, v100
	v_cvt_pk_bf16_f32 v35, v38, s0
	v_cvt_pk_bf16_f32 v37, v47, s0
	v_cvt_pk_bf16_f32 v38, v39, s0
	v_or_b32_e32 v39, 0x140, v101
	buffer_store_b16 v1, v43, s[0:3], null offen
	buffer_store_b16 v34, v36, s[44:47], null offen
	buffer_store_b16 v35, v36, s[0:3], null offen
	buffer_store_b16 v37, v39, s[44:47], null offen
	buffer_store_b16 v38, v39, s[0:3], null offen
	s_wait_xcnt 0x4
	v_cvt_pk_bf16_f32 v1, v48, s0
	s_wait_xcnt 0x2
	v_or_b32_e32 v35, 0x140, v99
	v_cvt_pk_bf16_f32 v34, v40, s0
	v_cvt_pk_bf16_f32 v36, v49, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v38, 0x140, v0
	buffer_store_b16 v1, v35, s[44:47], null offen
	buffer_store_b16 v34, v35, s[0:3], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v26, s0
	v_cvt_pk_bf16_f32 v19, v19, s0
	v_cvt_pk_bf16_f32 v37, v41, s0
	v_cvt_pk_bf16_f32 v26, v27, s0
	v_cvt_pk_bf16_f32 v27, v28, s0
	v_cvt_pk_bf16_f32 v18, v18, s0
	buffer_store_b16 v36, v38, s[44:47], null offen
	buffer_store_b16 v37, v38, s[0:3], null offen
	buffer_store_b16 v1, v131, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v1.l, v19.l
	v_mov_b16_e32 v19.l, v27.l
	buffer_store_b16 v18, v131, s[0:3], null offen offset:384
	buffer_store_b16 v26, v132, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v18, v20, s0
	v_cvt_pk_bf16_f32 v20, v30, s0
	buffer_store_b16 v1, v132, s[0:3], null offen offset:384
	buffer_store_b16 v19, v115, s[44:47], null offen offset:384
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v19, v21, s0
	v_cvt_pk_bf16_f32 v21, v22, s0
	v_cvt_pk_bf16_f32 v1, v29, s0
	buffer_store_b16 v18, v115, s[0:3], null offen offset:384
	s_wait_xcnt 0x0
	v_mov_b16_e32 v18.l, v20.l
	buffer_store_b16 v1, v124, s[44:47], null offen offset:384
	buffer_store_b16 v19, v124, s[0:3], null offen offset:384
	v_mov_b16_e32 v20.l, v21.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v31, s0
	buffer_store_b16 v18, v116, s[44:47], null offen offset:384
	buffer_store_b16 v20, v116, s[0:3], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v18, v23, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v20, v24, s0
	v_cvt_pk_bf16_f32 v19, v32, s0
	v_cvt_pk_bf16_f32 v21, v33, s0
	buffer_store_b16 v1, v125, s[44:47], null offen offset:384
	buffer_store_b16 v18, v125, s[0:3], null offen offset:384
	buffer_store_b16 v19, v118, s[44:47], null offen offset:384
	s_wait_xcnt 0x2
	v_mov_b16_e32 v1.l, v20.l
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v18, v25, s0
	v_mov_b16_e32 v20.l, v21.l
	buffer_store_b16 v1, v118, s[0:3], null offen offset:384
	buffer_store_b16 v20, v120, s[44:47], null offen offset:384
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v10, s0
	v_mov_b16_e32 v10.l, v18.l
	v_cvt_pk_bf16_f32 v2, v2, s0
	v_or_b32_e32 v18, 0x1c0, v119
	v_cvt_pk_bf16_f32 v11, v11, s0
	v_or_b32_e32 v19, 0x1c0, v121
	buffer_store_b16 v10, v120, s[0:3], null offen offset:384
	buffer_store_b16 v1, v18, s[44:47], null offen
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v1, v3, s0
	buffer_store_b16 v2, v18, s[0:3], null offen
	buffer_store_b16 v11, v19, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v2, v12, s0
	v_or_b32_e32 v10, 0x1c0, v106
	v_cvt_pk_bf16_f32 v3, v4, s0
	v_cvt_pk_bf16_f32 v4, v13, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v11, 0x1c0, v98
	buffer_store_b16 v1, v19, s[0:3], null offen
	buffer_store_b16 v2, v10, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v1, v5, s0
	v_cvt_pk_bf16_f32 v7, v7, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v2, v14, s0
	v_or_b32_e32 v5, 0x1c0, v100
	buffer_store_b16 v3, v10, s[0:3], null offen
	buffer_store_b16 v4, v11, s[44:47], null offen
	s_wait_xcnt 0x1
	v_cvt_pk_bf16_f32 v3, v6, s0
	s_wait_xcnt 0x0
	v_cvt_pk_bf16_f32 v4, v15, s0
	v_or_b32_e32 v6, 0x1c0, v101
	buffer_store_b16 v1, v11, s[0:3], null offen
	buffer_store_b16 v2, v5, s[44:47], null offen
	buffer_store_b16 v3, v5, s[0:3], null offen
	buffer_store_b16 v4, v6, s[44:47], null offen
	s_wait_xcnt 0x3
	v_mov_b16_e32 v1.l, v7.l
	s_wait_xcnt 0x2
	v_cvt_pk_bf16_f32 v2, v16, s0
	s_wait_xcnt 0x0
	v_or_b32_e32 v4, 0x1c0, v99
	v_cvt_pk_bf16_f32 v3, v8, s0
	v_cvt_pk_bf16_f32 v5, v17, s0
	v_or_b32_e32 v0, 0x1c0, v0
	v_cvt_pk_bf16_f32 v7, v9, s0
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
		.amdhsa_next_free_vgpr 956
		.amdhsa_next_free_sgpr 77
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

	.set .Lk_dkdv_0.num_vgpr, 956
	.set .Lk_dkdv_0.num_agpr, 0
	.set .Lk_dkdv_0.numbered_sgpr, 77
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
    .sgpr_count:     79
    .sgpr_spill_count: 0
    .symbol:         k_dkdv_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     956
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
