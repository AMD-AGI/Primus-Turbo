.LBB0_7:
	v_exp_f32_e64 v166, v166 clamp
	v_exp_f32_e64 v167, v167 clamp
	v_exp_f32_e64 v168, v168 clamp
	v_exp_f32_e64 v169, v169 clamp
	v_exp_f32_e64 v162, v162 clamp
	v_exp_f32_e64 v163, v163 clamp
	v_exp_f32_e64 v164, v164 clamp
	v_exp_f32_e64 v165, v165 clamp
	v_exp_f32_e64 v158, v158 clamp
	v_exp_f32_e64 v159, v159 clamp
	v_exp_f32_e64 v160, v160 clamp
	v_exp_f32_e64 v161, v161 clamp
	v_exp_f32_e64 v154, v154 clamp
	v_exp_f32_e64 v155, v155 clamp
	v_exp_f32_e64 v156, v156 clamp
	v_exp_f32_e64 v157, v157 clamp
	s_setprio 0
	s_add_u32 s8, s8, 2
	s_addc_u32 s9, s9, 0
	v_mov_b64_e32 v[170:171], s[2:3]
	s_add_u32 s28, s28, 0x80
	v_cmp_lt_u64_e32 vcc, s[8:9], v[170:171]
	v_add_u32_e32 v237, 0xffffff80, v237
	s_addc_u32 s29, s29, 0
	s_add_i32 s36, s36, s30
	s_add_i32 s31, s31, s30
	s_add_i32 s34, s34, s48
	s_add_i32 s11, s11, s48
	s_cbranch_vccz .LBB0_17
.LBB0_8:
	ds_read_b128 v[188:191], v208 offset:42368
	ds_read_b128 v[198:201], v208 offset:42880
	v_exp_f32_e64 v2, v2 clamp
	v_exp_f32_e64 v3, v3 clamp
	v_exp_f32_e64 v4, v4 clamp
	v_exp_f32_e64 v5, v5 clamp
	v_exp_f32_e64 v6, v6 clamp
	v_exp_f32_e64 v7, v7 clamp
	v_exp_f32_e64 v8, v8 clamp
	v_exp_f32_e64 v9, v9 clamp
	v_exp_f32_e64 v10, v10 clamp
	v_exp_f32_e64 v11, v11 clamp
	v_exp_f32_e64 v12, v12 clamp
	v_exp_f32_e64 v13, v13 clamp
	v_exp_f32_e64 v14, v14 clamp
	v_exp_f32_e64 v15, v15 clamp
	v_exp_f32_e64 v16, v16 clamp
	v_exp_f32_e64 v17, v17 clamp
	v_cvt_pk_bf16_f32 v181, v160, v161
	v_cvt_pk_bf16_f32 v180, v158, v159
	v_cvt_pk_bf16_f32 v179, v168, v169
	v_cvt_pk_bf16_f32 v178, v166, v167
	v_cvt_pk_bf16_f32 v185, v156, v157
	v_cvt_pk_bf16_f32 v184, v154, v155
	v_cvt_pk_bf16_f32 v183, v164, v165
	v_cvt_pk_bf16_f32 v182, v162, v163
	v_cvt_pk_bf16_f32 v173, v12, v13
	v_cvt_pk_bf16_f32 v172, v10, v11
	v_cvt_pk_bf16_f32 v171, v4, v5
	v_cvt_pk_bf16_f32 v170, v2, v3
	v_cvt_pk_bf16_f32 v177, v16, v17
	v_cvt_pk_bf16_f32 v176, v14, v15
	v_cvt_pk_bf16_f32 v175, v8, v9
	v_cvt_pk_bf16_f32 v174, v6, v7
	v_mfma_f32_16x16x32_bf16 v[2:5], v[240:243], v[178:181], v[146:149]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[240:243], v[182:185], v[150:153]
	v_mfma_f32_16x16x32_bf16 v[6:9], v[240:243], v[170:173], v[2:5]
	v_mfma_f32_16x16x32_bf16 v[2:5], v[240:243], v[174:177], v[10:13]
	ds_read_b128 v[162:165], v208 offset:42624
	ds_read_b128 v[166:169], v208 offset:43136
	ds_read_b128 v[202:205], v208 offset:42400
	ds_read_b128 v[244:247], v208 offset:42912
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 v[10:13], v[188:191], v[82:85], v[74:77]
	ds_read_b128 v[212:215], v208 offset:42432
	ds_read_b128 v[248:251], v208 offset:42720
	ds_read_b128 v[216:219], v211 offset:51232
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 v[14:17], v[198:201], v[82:85], v[74:77]
	ds_read_b128 v[220:223], v211 offset:51488
	v_mfma_f32_16x16x32_bf16 v[146:149], v[188:191], v[106:109], v[78:81]
	ds_read_b128 v[188:191], v208 offset:42464
	v_mfma_f32_16x16x32_bf16 v[150:153], v[198:201], v[106:109], v[78:81]
	ds_read_b128 v[198:201], v208 offset:43168
	s_waitcnt lgkmcnt(9)
	v_mfma_f32_16x16x32_bf16 v[154:157], v[162:165], v[82:85], v[130:133]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[158:161], v[166:169], v[82:85], v[130:133]
	v_mfma_f32_16x16x32_bf16 v[162:165], v[162:165], v[106:109], v[134:137]
	v_mfma_f32_16x16x32_bf16 v[166:169], v[166:169], v[106:109], v[134:137]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 v[10:13], v[202:205], v[86:89], v[10:13]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[14:17], v[244:247], v[86:89], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[146:149], v[202:205], v[110:113], v[146:149]
	ds_read_b128 v[202:205], v208 offset:42656
	v_mfma_f32_16x16x32_bf16 v[150:153], v[244:247], v[110:113], v[150:153]
	ds_read_b128 v[244:247], v208 offset:42688
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[154:157], v[202:205], v[86:89], v[154:157]
	v_mfma_f32_16x16x32_bf16 v[158:161], v[198:201], v[86:89], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[162:165], v[202:205], v[110:113], v[162:165]
	ds_read_b128 v[202:205], v208 offset:42976
	v_mfma_f32_16x16x32_bf16 v[166:169], v[198:201], v[110:113], v[166:169]
	ds_read_b128 v[198:201], v208 offset:42944
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[14:17], v[198:201], v[90:93], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[150:153], v[198:201], v[114:117], v[150:153]
	ds_read_b128 v[198:201], v208 offset:43200
	v_mfma_f32_16x16x32_bf16 v[10:13], v[212:215], v[90:93], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[146:149], v[212:215], v[114:117], v[146:149]
	ds_read_b128 v[212:215], v208 offset:43232
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[158:161], v[198:201], v[90:93], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[166:169], v[198:201], v[114:117], v[166:169]
	ds_read_b128 v[198:201], v211 offset:50720
	v_mfma_f32_16x16x32_bf16 v[10:13], v[188:191], v[94:97], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[146:149], v[188:191], v[118:121], v[146:149]
	ds_read_b128 v[188:191], v211 offset:50688
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[158:161], v[212:215], v[94:97], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[166:169], v[212:215], v[118:121], v[166:169]
	ds_read_b128 v[212:215], v211 offset:51200
	v_mfma_f32_16x16x32_bf16 v[14:17], v[202:205], v[94:97], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[150:153], v[202:205], v[118:121], v[150:153]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[14:17], v[212:215], v[98:101], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[202:205], v[188:191], v[122:125], v[146:149]
	v_mfma_f32_16x16x32_bf16 v[150:153], v[212:215], v[122:125], v[150:153]
	s_nop 1
	ds_read_b128 v[146:149], v211 offset:50944
	ds_read_b128 v[212:215], v211 offset:51456
	v_mfma_f32_16x16x32_bf16 v[154:157], v[244:247], v[90:93], v[154:157]
	v_mfma_f32_16x16x32_bf16 v[162:165], v[244:247], v[114:117], v[162:165]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[188:191], v[98:101], v[10:13]
	ds_read_b128 v[188:191], v211 offset:50976
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[154:157], v[248:251], v[94:97], v[154:157]
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[162:165], v[248:251], v[118:121], v[162:165]
	v_mfma_f32_16x16x32_bf16 v[154:157], v[146:149], v[98:101], v[154:157]
	v_mfma_f32_16x16x32_bf16 v[244:247], v[212:215], v[98:101], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[248:251], v[146:149], v[122:125], v[162:165]
	v_mfma_f32_16x16x32_bf16 v[252:255], v[212:215], v[122:125], v[166:169]
	v_mfma_f32_16x16x32_bf16 v[166:169], v[198:201], v[102:105], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[146:149], v[216:219], v[102:105], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[162:165], v[198:201], v[126:129], v[202:205]
	v_mfma_f32_16x16x32_bf16 v[14:17], v[216:219], v[126:129], v[150:153]
	v_mfma_f32_16x16x32_bf16 v[158:161], v[188:191], v[102:105], v[154:157]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[220:223], v[102:105], v[244:247]
	v_mfma_f32_16x16x32_bf16 v[154:157], v[188:191], v[126:129], v[248:251]
	v_mfma_f32_16x16x32_bf16 v[150:153], v[220:223], v[126:129], v[252:255]
	ds_read_b64_tr_b16 v[188:189], v207 offset:24960
	ds_read_b64_tr_b16 v[190:191], v207 offset:25216
	ds_read_b64_tr_b16 v[204:205], v207 offset:25280
	ds_read_b64_tr_b16 v[202:203], v207 offset:25024
	ds_read_b64_tr_b16 v[198:199], v206 offset:24960
	ds_read_b64_tr_b16 v[200:201], v206 offset:25216
	ds_read_b64_tr_b16 v[214:215], v206 offset:25280
	ds_read_b64_tr_b16 v[212:213], v206 offset:25024
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v194, s[12:15], s11 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v195, s[12:15], s11 offen lds
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v196, s[12:15], s11 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v209, s[16:19], s31 offen lds
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v210, s[16:19], s31 offen lds
	s_setprio 1
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 v[70:73], v[202:205], v[178:181], v[70:73]
	ds_read_b64_tr_b16 v[244:245], v207 offset:33664
	ds_read_b64_tr_b16 v[246:247], v207 offset:33920
	ds_read_b64_tr_b16 v[254:255], v207 offset:33984
	ds_read_b64_tr_b16 v[252:253], v207 offset:33728
	v_mfma_f32_16x16x32_bf16 v[66:69], v[202:205], v[182:185], v[66:69]
	ds_read_b64_tr_b16 v[248:249], v206 offset:33664
	ds_read_b64_tr_b16 v[250:251], v206 offset:33920
	ds_read_b64_tr_b16 v[204:205], v206 offset:33984
	ds_read_b64_tr_b16 v[202:203], v206 offset:33728
	v_mfma_f32_16x16x32_bf16 v[54:57], v[188:191], v[178:181], v[54:57]
	v_cmp_le_i64_e32 vcc, s[28:29], v[186:187]
	s_and_b64 vcc, exec, vcc
	v_mfma_f32_16x16x32_bf16 v[50:53], v[188:191], v[182:185], v[50:53]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[198:201], v[178:181], v[62:65]
	v_mfma_f32_16x16x32_bf16 v[58:61], v[198:201], v[182:185], v[58:61]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[142:145], v[212:215], v[178:181], v[142:145]
	v_mfma_f32_16x16x32_bf16 v[138:141], v[212:215], v[182:185], v[138:141]
	s_cbranch_vccnz .LBB0_10
	v_add_u32_e32 v188, 0x60, v237
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 0
	v_cmp_lt_i32_e64 s[52:53], v188, 1
	v_cndmask_b32_e64 v166, v166, v238, s[50:51]
	v_cndmask_b32_e64 v167, v167, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 2
	v_cmp_lt_i32_e64 s[52:53], v188, 3
	v_cndmask_b32_e64 v168, v168, v238, s[50:51]
	v_cndmask_b32_e64 v169, v169, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 16
	v_cmp_lt_i32_e64 s[52:53], v188, 17
	v_cndmask_b32_e64 v158, v158, v238, s[50:51]
	v_cndmask_b32_e64 v159, v159, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 18
	v_cmp_lt_i32_e64 s[52:53], v188, 19
	v_cndmask_b32_e64 v160, v160, v238, s[50:51]
	v_cndmask_b32_e64 v161, v161, v238, s[52:53]
	;;#ASMEND
	v_add_u32_e32 v188, 0x70, v237
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 0
	v_cmp_lt_i32_e64 s[52:53], v188, 1
	v_cndmask_b32_e64 v162, v162, v238, s[50:51]
	v_cndmask_b32_e64 v163, v163, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 2
	v_cmp_lt_i32_e64 s[52:53], v188, 3
	v_cndmask_b32_e64 v164, v164, v238, s[50:51]
	v_cndmask_b32_e64 v165, v165, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 16
	v_cmp_lt_i32_e64 s[52:53], v188, 17
	v_cndmask_b32_e64 v154, v154, v238, s[50:51]
	v_cndmask_b32_e64 v155, v155, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 18
	v_cmp_lt_i32_e64 s[52:53], v188, 19
	v_cndmask_b32_e64 v156, v156, v238, s[50:51]
	v_cndmask_b32_e64 v157, v157, v238, s[52:53]
	;;#ASMEND
	v_add_u32_e32 v188, 64, v237
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 0
	v_cmp_lt_i32_e64 s[52:53], v188, 1
	v_cndmask_b32_e64 v146, v146, v238, s[50:51]
	v_cndmask_b32_e64 v147, v147, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 2
	v_cmp_lt_i32_e64 s[52:53], v188, 3
	v_cndmask_b32_e64 v148, v148, v238, s[50:51]
	v_cndmask_b32_e64 v149, v149, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 16
	v_cmp_lt_i32_e64 s[52:53], v188, 17
	v_cndmask_b32_e64 v10, v10, v238, s[50:51]
	v_cndmask_b32_e64 v11, v11, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 18
	v_cmp_lt_i32_e64 s[52:53], v188, 19
	v_cndmask_b32_e64 v12, v12, v238, s[50:51]
	v_cndmask_b32_e64 v13, v13, v238, s[52:53]
	;;#ASMEND
	v_add_u32_e32 v188, 0x50, v237
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 0
	v_cmp_lt_i32_e64 s[52:53], v188, 1
	v_cndmask_b32_e64 v14, v14, v238, s[50:51]
	v_cndmask_b32_e64 v15, v15, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 2
	v_cmp_lt_i32_e64 s[52:53], v188, 3
	v_cndmask_b32_e64 v16, v16, v238, s[50:51]
	v_cndmask_b32_e64 v17, v17, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 16
	v_cmp_lt_i32_e64 s[52:53], v188, 17
	v_cndmask_b32_e64 v150, v150, v238, s[50:51]
	v_cndmask_b32_e64 v151, v151, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 18
	v_cmp_lt_i32_e64 s[52:53], v188, 19
	v_cndmask_b32_e64 v152, v152, v238, s[50:51]
	v_cndmask_b32_e64 v153, v153, v238, s[52:53]
	;;#ASMEND
.LBB0_10:
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[198:201], v[244:247], v[178:181], v[18:21]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[188:191], v[248:251], v[178:181], v[26:29]
	v_mfma_f32_16x16x32_bf16 v[248:251], v[248:251], v[182:185], v[22:25]
	v_mfma_f32_16x16x32_bf16 v[212:215], v[252:255], v[178:181], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[252:255], v[252:255], v[182:185], v[30:33]
	ds_read_b64_tr_b16 v[18:19], v207 offset:25472
	ds_read_b64_tr_b16 v[20:21], v207 offset:25728
	ds_read_b64_tr_b16 v[22:23], v206 offset:25472
	ds_read_b64_tr_b16 v[24:25], v206 offset:25728
	ds_read_b64_tr_b16 v[26:27], v207 offset:25536
	ds_read_b64_tr_b16 v[28:29], v207 offset:25792
	ds_read_b64_tr_b16 v[30:31], v206 offset:25536
	ds_read_b64_tr_b16 v[32:33], v206 offset:25792
	v_mfma_f32_16x16x32_bf16 v[244:247], v[244:247], v[182:185], v[38:41]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[178:181], v[202:205], v[178:181], v[46:49]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[34:37], v[18:21], v[170:173], v[54:57]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 v[38:41], v[22:25], v[170:173], v[62:65]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[22:25], v[174:177], v[58:61]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[46:49], v[30:33], v[170:173], v[142:145]
	ds_read_b64_tr_b16 v[54:55], v207 offset:34176
	ds_read_b64_tr_b16 v[56:57], v207 offset:34432
	ds_read_b64_tr_b16 v[58:59], v206 offset:34176
	ds_read_b64_tr_b16 v[60:61], v206 offset:34432
	ds_read_b64_tr_b16 v[62:63], v207 offset:34240
	ds_read_b64_tr_b16 v[64:65], v207 offset:34496
	ds_read_b64_tr_b16 v[142:143], v206 offset:34240
	ds_read_b64_tr_b16 v[144:145], v206 offset:34496
	v_mfma_f32_16x16x32_bf16 v[182:185], v[202:205], v[182:185], v[42:45]
	v_mfma_f32_16x16x32_bf16 v[18:21], v[18:21], v[174:177], v[50:53]
	v_mfma_f32_16x16x32_bf16 v[42:45], v[26:29], v[170:173], v[70:73]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[26:29], v[174:177], v[66:69]
	v_mfma_f32_16x16x32_bf16 v[30:33], v[30:33], v[174:177], v[138:141]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[50:53], v[54:57], v[170:173], v[198:201]
	v_mfma_f32_16x16x32_bf16 v[66:69], v[54:57], v[174:177], v[244:247]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 v[54:57], v[58:61], v[170:173], v[188:191]
	v_mfma_f32_16x16x32_bf16 v[70:73], v[58:61], v[174:177], v[248:251]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[58:61], v[62:65], v[170:173], v[212:215]
	v_mfma_f32_16x16x32_bf16 v[138:141], v[62:65], v[174:177], v[252:255]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[142:145], v[170:173], v[178:181]
	v_max3_f32 v170, v166, v167, v168
	v_max_f32_e32 v173, v169, v162
	v_max3_f32 v171, v15, v16, v17
	v_mfma_f32_16x16x32_bf16 v[142:145], v[142:145], v[174:177], v[182:185]
	v_max_f32_e32 v174, v164, v165
	v_max3_f32 v175, v159, v160, v161
	v_max3_f32 v176, v154, v155, v156
	v_max3_f32 v177, v157, v146, v147
	v_max3_f32 v178, v148, v149, v14
	v_max3_f32 v172, v10, v11, v12
	v_max3_f32 v173, v170, v173, v163
	v_max3_f32 v173, v173, v174, v158
	v_max3_f32 v174, v175, v176, v177
	v_max3_f32 v175, v178, v171, v172
	v_max3_f32 v179, v13, v150, v151
	v_max_f32_e32 v180, v152, v153
	v_max3_f32 v173, v173, v174, v175
	v_max3_f32 v173, v173, v179, v180
	v_cmp_lt_f32_e32 vcc, 0, v173
	s_cbranch_vccz .LBB0_12
	v_max_f32_e32 v74, v169, v158
	v_max3_f32 v76, v147, v148, v149
	v_max3_f32 v75, v160, v161, v146
	v_max3_f32 v74, v170, v74, v159
	v_max3_f32 v76, v76, v172, v13
	v_max3_f32 v74, v74, v75, v76
	v_max_f32_e32 v75, v162, v163
	v_max3_f32 v76, v165, v154, v155
	v_max3_f32 v78, v150, v151, v152
	v_max3_f32 v77, v156, v157, v14
	v_max3_f32 v75, v75, v164, v76
	v_max3_f32 v76, v171, v78, v153
	v_max3_f32 v75, v75, v77, v76
	s_nop 1
	v_permlane16_swap_b32_e32 v74, v75
	v_max_f32_e32 v74, v74, v75
	v_mov_b32_e32 v75, v74
	s_nop 1
	v_permlane32_swap_b32_e64 v74, v75 bound_ctrl:1
	v_max3_f32 v74, v74, v75, s21
	v_add_f32_e32 v74, 0x42c00000, v74
	v_max_f32_e32 v74, 0, v74
	v_exp_f32_e64 v76, -v74
	v_mov_b32_e32 v78, v74
	v_add_f32_e32 v1, v74, v1
	s_nop 0
	v_permlane16_swap_b32_e32 v74, v78
	v_pk_add_f32 v[166:167], v[166:167], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[168:169], v[168:169], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[158:159], v[158:159], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[160:161], v[160:161], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[146:147], v[146:147], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[148:149], v[148:149], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[10:11], v[10:11], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[12:13], v[12:13], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_xor_b32_e32 v74, 0x80000000, v1
	v_mov_b32_e32 v80, v76
	v_pk_add_f32 v[162:163], v[162:163], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[164:165], v[164:165], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[154:155], v[154:155], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[156:157], v[156:157], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[14:15], v[14:15], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[16:17], v[16:17], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[150:151], v[150:151], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[152:153], v[152:153], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_mov_b32_e32 v78, v74
	v_permlane16_swap_b32_e32 v76, v80
	v_pk_mul_f32 v[4:5], v[80:81], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[80:81], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[76:77], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[76:77], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[80:81], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[80:81], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[76:77], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[76:77], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[80:81], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[80:81], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[76:77], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[76:77], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[80:81], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[80:81], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[76:77], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[76:77], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[80:81], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[80:81], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[76:77], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[76:77], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[140:141], v[80:81], v[140:141] op_sel_hi:[0,1]
	v_pk_mul_f32 v[138:139], v[80:81], v[138:139] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[76:77], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[76:77], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[80:81], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[80:81], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[76:77], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[76:77], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[80:81], v[144:145] op_sel_hi:[0,1]
	v_pk_mul_f32 v[142:143], v[80:81], v[142:143] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[76:77], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[76:77], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[80:81], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[80:81], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[76:77], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[76:77], v[46:47] op_sel_hi:[0,1]
	v_permlane16_swap_b32_e32 v74, v78
	v_mov_b32_e32 v75, v74
	v_mov_b32_e32 v76, v74
	v_mov_b32_e32 v77, v74
	v_mov_b32_e32 v79, v78
	v_mov_b32_e32 v80, v78
	v_mov_b32_e32 v81, v78
	v_mov_b32_e32 v130, v74
	v_mov_b32_e32 v131, v74
	v_mov_b32_e32 v132, v74
	v_mov_b32_e32 v133, v74
	v_mov_b32_e32 v134, v78
	v_mov_b32_e32 v135, v78
	v_mov_b32_e32 v136, v78
	v_mov_b32_e32 v137, v78
.LBB0_12:
	v_exp_f32_e64 v166, v166 clamp
	v_exp_f32_e64 v167, v167 clamp
	v_exp_f32_e64 v168, v168 clamp
	v_exp_f32_e64 v169, v169 clamp
	v_exp_f32_e64 v162, v162 clamp
	v_exp_f32_e64 v163, v163 clamp
	v_exp_f32_e64 v164, v164 clamp
	v_exp_f32_e64 v165, v165 clamp
	v_exp_f32_e64 v158, v158 clamp
	v_exp_f32_e64 v159, v159 clamp
	v_exp_f32_e64 v160, v160 clamp
	v_exp_f32_e64 v161, v161 clamp
	v_exp_f32_e64 v154, v154 clamp
	v_exp_f32_e64 v155, v155 clamp
	v_exp_f32_e64 v156, v156 clamp
	v_exp_f32_e64 v157, v157 clamp
	s_add_i32 s37, s28, 64
	s_setprio 0
	ds_read_b128 v[188:191], v208
	ds_read_b128 v[198:201], v208 offset:512
	v_exp_f32_e64 v146, v146 clamp
	v_exp_f32_e64 v147, v147 clamp
	v_exp_f32_e64 v148, v148 clamp
	v_exp_f32_e64 v149, v149 clamp
	v_exp_f32_e64 v14, v14 clamp
	v_exp_f32_e64 v15, v15 clamp
	v_exp_f32_e64 v16, v16 clamp
	v_exp_f32_e64 v17, v17 clamp
	v_exp_f32_e64 v10, v10 clamp
	v_exp_f32_e64 v11, v11 clamp
	v_exp_f32_e64 v12, v12 clamp
	v_exp_f32_e64 v13, v13 clamp
	v_exp_f32_e64 v150, v150 clamp
	v_exp_f32_e64 v151, v151 clamp
	v_exp_f32_e64 v152, v152 clamp
	v_exp_f32_e64 v153, v153 clamp
	v_cvt_pk_bf16_f32 v181, v160, v161
	v_cvt_pk_bf16_f32 v180, v158, v159
	v_cvt_pk_bf16_f32 v179, v168, v169
	v_cvt_pk_bf16_f32 v178, v166, v167
	v_cvt_pk_bf16_f32 v185, v156, v157
	v_cvt_pk_bf16_f32 v184, v154, v155
	v_cvt_pk_bf16_f32 v183, v164, v165
	v_cvt_pk_bf16_f32 v182, v162, v163
	v_cvt_pk_bf16_f32 v173, v12, v13
	v_cvt_pk_bf16_f32 v172, v10, v11
	v_cvt_pk_bf16_f32 v171, v148, v149
	v_cvt_pk_bf16_f32 v170, v146, v147
	v_cvt_pk_bf16_f32 v177, v152, v153
	v_cvt_pk_bf16_f32 v176, v150, v151
	v_cvt_pk_bf16_f32 v175, v16, v17
	v_cvt_pk_bf16_f32 v174, v14, v15
	v_mfma_f32_16x16x32_bf16 v[6:9], v[240:243], v[178:181], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[2:5], v[240:243], v[182:185], v[2:5]
	v_mfma_f32_16x16x32_bf16 v[146:149], v[240:243], v[170:173], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[150:153], v[240:243], v[174:177], v[2:5]
	ds_read_b128 v[162:165], v208 offset:256
	ds_read_b128 v[166:169], v208 offset:768
	ds_read_b128 v[202:205], v208 offset:32
	ds_read_b128 v[216:219], v208 offset:544
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 v[2:5], v[188:191], v[82:85], v[74:77]
	ds_read_b128 v[212:215], v208 offset:64
	ds_read_b128 v[220:223], v208 offset:352
	ds_read_b128 v[244:247], v211 offset:9120
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 v[6:9], v[198:201], v[82:85], v[74:77]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[188:191], v[106:109], v[78:81]
	ds_read_b128 v[188:191], v208 offset:96
	v_mfma_f32_16x16x32_bf16 v[14:17], v[198:201], v[106:109], v[78:81]
	ds_read_b128 v[198:201], v208 offset:800
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[154:157], v[162:165], v[82:85], v[130:133]
	s_waitcnt lgkmcnt(7)
	v_mfma_f32_16x16x32_bf16 v[158:161], v[166:169], v[82:85], v[130:133]
	v_mfma_f32_16x16x32_bf16 v[162:165], v[162:165], v[106:109], v[134:137]
	v_mfma_f32_16x16x32_bf16 v[166:169], v[166:169], v[106:109], v[134:137]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[2:5], v[202:205], v[86:89], v[2:5]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_bf16 v[6:9], v[216:219], v[86:89], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[202:205], v[110:113], v[10:13]
	ds_read_b128 v[202:205], v208 offset:288
	v_mfma_f32_16x16x32_bf16 v[14:17], v[216:219], v[110:113], v[14:17]
	ds_read_b128 v[216:219], v208 offset:320
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[154:157], v[202:205], v[86:89], v[154:157]
	v_mfma_f32_16x16x32_bf16 v[158:161], v[198:201], v[86:89], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[162:165], v[202:205], v[110:113], v[162:165]
	ds_read_b128 v[202:205], v208 offset:608
	v_mfma_f32_16x16x32_bf16 v[166:169], v[198:201], v[110:113], v[166:169]
	ds_read_b128 v[198:201], v208 offset:576
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[6:9], v[198:201], v[90:93], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[14:17], v[198:201], v[114:117], v[14:17]
	ds_read_b128 v[198:201], v208 offset:832
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[158:161], v[198:201], v[90:93], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[166:169], v[198:201], v[114:117], v[166:169]
	ds_read_b128 v[198:201], v211 offset:8832
	v_mfma_f32_16x16x32_bf16 v[2:5], v[212:215], v[90:93], v[2:5]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[212:215], v[114:117], v[10:13]
	ds_read_b128 v[212:215], v208 offset:864
	v_mfma_f32_16x16x32_bf16 v[6:9], v[202:205], v[94:97], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[14:17], v[202:205], v[118:121], v[14:17]
	ds_read_b128 v[202:205], v211 offset:9088
	v_mfma_f32_16x16x32_bf16 v[154:157], v[216:219], v[90:93], v[154:157]
	v_mfma_f32_16x16x32_bf16 v[162:165], v[216:219], v[114:117], v[162:165]
	ds_read_b128 v[216:219], v211 offset:8864
	v_mfma_f32_16x16x32_bf16 v[2:5], v[188:191], v[94:97], v[2:5]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[188:191], v[118:121], v[10:13]
	ds_read_b128 v[188:191], v211 offset:8320
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 v[6:9], v[198:201], v[98:101], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[14:17], v[198:201], v[122:125], v[14:17]
	ds_read_b128 v[198:201], v211 offset:8576
	v_mfma_f32_16x16x32_bf16 v[154:157], v[220:223], v[94:97], v[154:157]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 v[158:161], v[212:215], v[94:97], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[162:165], v[220:223], v[118:121], v[162:165]
	ds_read_b128 v[220:223], v211 offset:8608
	v_mfma_f32_16x16x32_bf16 v[166:169], v[212:215], v[118:121], v[166:169]
	ds_read_b128 v[212:215], v211 offset:8352
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[2:5], v[188:191], v[98:101], v[2:5]
	s_barrier
	v_mfma_f32_16x16x32_bf16 v[10:13], v[188:191], v[122:125], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[154:157], v[198:201], v[98:101], v[154:157]
	v_mfma_f32_16x16x32_bf16 v[188:191], v[202:205], v[98:101], v[158:161]
	v_mfma_f32_16x16x32_bf16 v[198:201], v[198:201], v[122:125], v[162:165]
	v_mfma_f32_16x16x32_bf16 v[202:205], v[202:205], v[122:125], v[166:169]
	v_mfma_f32_16x16x32_bf16 v[166:169], v[212:215], v[102:105], v[2:5]
	v_mfma_f32_16x16x32_bf16 v[2:5], v[216:219], v[102:105], v[6:9]
	v_mfma_f32_16x16x32_bf16 v[162:165], v[212:215], v[126:129], v[10:13]
	v_mfma_f32_16x16x32_bf16 v[6:9], v[216:219], v[126:129], v[14:17]
	v_mfma_f32_16x16x32_bf16 v[158:161], v[220:223], v[102:105], v[154:157]
	v_mfma_f32_16x16x32_bf16 v[10:13], v[244:247], v[102:105], v[188:191]
	v_mfma_f32_16x16x32_bf16 v[154:157], v[220:223], v[126:129], v[198:201]
	v_mfma_f32_16x16x32_bf16 v[14:17], v[244:247], v[126:129], v[202:205]
	s_nop 0
	v_add_u32_e32 v188, 0x10700, v207
	s_nop 0
	ds_read_b64_tr_b16 v[202:203], v188
	v_add_u32_e32 v188, 0x10800, v207
	ds_read_b64_tr_b16 v[204:205], v188
	v_add_u32_e32 v188, 0x10700, v206
	ds_read_b64_tr_b16 v[244:245], v188
	v_add_u32_e32 v188, 0x10800, v206
	ds_read_b64_tr_b16 v[246:247], v188
	v_add_u32_e32 v188, 0x10740, v207
	ds_read_b64_tr_b16 v[248:249], v188
	v_add_u32_e32 v188, 0x10840, v207
	ds_read_b64_tr_b16 v[250:251], v188
	v_add_u32_e32 v188, 0x10740, v206
	ds_read_b64_tr_b16 v[252:253], v188
	v_add_u32_e32 v188, 0x10840, v206
	ds_read_b64_tr_b16 v[254:255], v188
	s_cmp_ge_i32 s20, s37
	s_cbranch_scc1 .LBB0_14
	v_add_u32_e32 v188, 32, v237
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 0
	v_cmp_lt_i32_e64 s[52:53], v188, 1
	v_cndmask_b32_e64 v166, v166, v238, s[50:51]
	v_cndmask_b32_e64 v167, v167, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 2
	v_cmp_lt_i32_e64 s[52:53], v188, 3
	v_cndmask_b32_e64 v168, v168, v238, s[50:51]
	v_cndmask_b32_e64 v169, v169, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 16
	v_cmp_lt_i32_e64 s[52:53], v188, 17
	v_cndmask_b32_e64 v158, v158, v238, s[50:51]
	v_cndmask_b32_e64 v159, v159, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 18
	v_cmp_lt_i32_e64 s[52:53], v188, 19
	v_cndmask_b32_e64 v160, v160, v238, s[50:51]
	v_cndmask_b32_e64 v161, v161, v238, s[52:53]
	;;#ASMEND
	v_add_u32_e32 v188, 48, v237
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 0
	v_cmp_lt_i32_e64 s[52:53], v188, 1
	v_cndmask_b32_e64 v162, v162, v238, s[50:51]
	v_cndmask_b32_e64 v163, v163, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 2
	v_cmp_lt_i32_e64 s[52:53], v188, 3
	v_cndmask_b32_e64 v164, v164, v238, s[50:51]
	v_cndmask_b32_e64 v165, v165, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 16
	v_cmp_lt_i32_e64 s[52:53], v188, 17
	v_cndmask_b32_e64 v154, v154, v238, s[50:51]
	v_cndmask_b32_e64 v155, v155, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 18
	v_cmp_lt_i32_e64 s[52:53], v188, 19
	v_cndmask_b32_e64 v156, v156, v238, s[50:51]
	v_cndmask_b32_e64 v157, v157, v238, s[52:53]
	;;#ASMEND
	v_add_u32_e32 v188, 16, v237
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v237, 0
	v_cmp_lt_i32_e64 s[52:53], v237, 1
	v_cndmask_b32_e64 v2, v2, v238, s[50:51]
	v_cndmask_b32_e64 v3, v3, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v237, 2
	v_cmp_lt_i32_e64 s[52:53], v237, 3
	v_cndmask_b32_e64 v4, v4, v238, s[50:51]
	v_cndmask_b32_e64 v5, v5, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v237, 16
	v_cmp_lt_i32_e64 s[52:53], v237, 17
	v_cndmask_b32_e64 v10, v10, v238, s[50:51]
	v_cndmask_b32_e64 v11, v11, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v237, 18
	v_cmp_lt_i32_e64 s[52:53], v237, 19
	v_cndmask_b32_e64 v12, v12, v238, s[50:51]
	v_cndmask_b32_e64 v13, v13, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 0
	v_cmp_lt_i32_e64 s[52:53], v188, 1
	v_cndmask_b32_e64 v6, v6, v238, s[50:51]
	v_cndmask_b32_e64 v7, v7, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 2
	v_cmp_lt_i32_e64 s[52:53], v188, 3
	v_cndmask_b32_e64 v8, v8, v238, s[50:51]
	v_cndmask_b32_e64 v9, v9, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 16
	v_cmp_lt_i32_e64 s[52:53], v188, 17
	v_cndmask_b32_e64 v14, v14, v238, s[50:51]
	v_cndmask_b32_e64 v15, v15, v238, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[50:51], v188, 18
	v_cmp_lt_i32_e64 s[52:53], v188, 19
	v_cndmask_b32_e64 v16, v16, v238, s[50:51]
	v_cndmask_b32_e64 v17, v17, v238, s[52:53]
	;;#ASMEND
.LBB0_14:
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v194, s[12:15], s34 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v195, s[12:15], s34 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v196, s[12:15], s34 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v209, s[16:19], s36 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v210, s[16:19], s36 offen lds
	s_setprio 1
	v_add_u32_e32 v188, 0x12900, v207
	ds_read_b64_tr_b16 v[198:199], v188
	v_add_u32_e32 v188, 0x12a00, v207
	ds_read_b64_tr_b16 v[200:201], v188
	v_add_u32_e32 v188, 0x12900, v206
	ds_read_b64_tr_b16 v[212:213], v188
	v_add_u32_e32 v188, 0x12a00, v206
	ds_read_b64_tr_b16 v[214:215], v188
	v_add_u32_e32 v188, 0x12940, v207
	ds_read_b64_tr_b16 v[216:217], v188
	v_add_u32_e32 v188, 0x12a40, v207
	ds_read_b64_tr_b16 v[218:219], v188
	ds_read_b64_tr_b16 v[220:221], v192
	ds_read_b64_tr_b16 v[222:223], v193
	s_waitcnt lgkmcnt(12)
	v_mfma_f32_16x16x32_bf16 v[38:41], v[244:247], v[178:181], v[38:41]
	v_mfma_f32_16x16x32_bf16 v[22:25], v[244:247], v[182:185], v[22:25]
	s_waitcnt lgkmcnt(10)
	v_mfma_f32_16x16x32_bf16 v[42:45], v[248:251], v[178:181], v[42:45]
	v_mfma_f32_16x16x32_bf16 v[26:29], v[248:251], v[182:185], v[26:29]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[188:191], v[198:201], v[178:181], v[50:53]
	v_mfma_f32_16x16x32_bf16 v[198:201], v[198:201], v[182:185], v[66:69]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[244:247], v[216:219], v[178:181], v[58:61]
	v_mfma_f32_16x16x32_bf16 v[248:251], v[216:219], v[182:185], v[138:141]
	ds_read_b64_tr_b16 v[50:51], v197
	ds_read_b64_tr_b16 v[52:53], v239
	ds_read_b64_tr_b16 v[58:59], v0
	ds_read_b64_tr_b16 v[60:61], v224
	ds_read_b64_tr_b16 v[66:67], v225
	ds_read_b64_tr_b16 v[68:69], v226
	ds_read_b64_tr_b16 v[138:139], v227
	ds_read_b64_tr_b16 v[140:141], v228
	v_mfma_f32_16x16x32_bf16 v[46:49], v[252:255], v[178:181], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[30:33], v[252:255], v[182:185], v[30:33]
	v_mfma_f32_16x16x32_bf16 v[34:37], v[202:205], v[178:181], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[18:21], v[202:205], v[182:185], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[202:205], v[212:215], v[178:181], v[54:57]
	v_mfma_f32_16x16x32_bf16 v[212:215], v[212:215], v[182:185], v[70:73]
	s_waitcnt lgkmcnt(8)
	v_mfma_f32_16x16x32_bf16 v[178:181], v[220:223], v[178:181], v[62:65]
	v_mfma_f32_16x16x32_bf16 v[182:185], v[220:223], v[182:185], v[142:145]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 v[62:65], v[58:61], v[170:173], v[38:41]
	v_mfma_f32_16x16x32_bf16 v[58:61], v[58:61], v[174:177], v[22:25]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[70:73], v[66:69], v[170:173], v[42:45]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[142:145], v[138:141], v[170:173], v[46:49]
	v_mfma_f32_16x16x32_bf16 v[138:141], v[138:141], v[174:177], v[30:33]
	ds_read_b64_tr_b16 v[22:23], v229
	ds_read_b64_tr_b16 v[24:25], v230
	s_nop 0
	ds_read_b64_tr_b16 v[30:31], v231
	ds_read_b64_tr_b16 v[32:33], v232
	ds_read_b64_tr_b16 v[42:43], v233
	ds_read_b64_tr_b16 v[44:45], v234
	ds_read_b64_tr_b16 v[216:217], v235
	ds_read_b64_tr_b16 v[218:219], v236
	v_mfma_f32_16x16x32_bf16 v[54:57], v[50:53], v[170:173], v[34:37]
	v_mfma_f32_16x16x32_bf16 v[50:53], v[50:53], v[174:177], v[18:21]
	v_mfma_f32_16x16x32_bf16 v[66:69], v[66:69], v[174:177], v[26:29]
	s_waitcnt lgkmcnt(6)
	v_mfma_f32_16x16x32_bf16 v[18:21], v[22:25], v[170:173], v[188:191]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_bf16 v[26:29], v[30:33], v[170:173], v[202:205]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_bf16 v[34:37], v[42:45], v[170:173], v[244:247]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[46:49], v[216:219], v[170:173], v[178:181]
	v_max3_f32 v170, v166, v167, v168
	v_max_f32_e32 v173, v169, v162
	v_max3_f32 v171, v7, v8, v9
	v_mfma_f32_16x16x32_bf16 v[38:41], v[22:25], v[174:177], v[198:201]
	v_max3_f32 v178, v4, v5, v6
	v_max3_f32 v172, v10, v11, v12
	v_max3_f32 v173, v170, v173, v163
	v_mfma_f32_16x16x32_bf16 v[22:25], v[30:33], v[174:177], v[212:215]
	v_max3_f32 v179, v13, v14, v15
	v_max_f32_e32 v180, v16, v17
	v_mfma_f32_16x16x32_bf16 v[30:33], v[42:45], v[174:177], v[248:251]
	v_mfma_f32_16x16x32_bf16 v[42:45], v[216:219], v[174:177], v[182:185]
	v_max_f32_e32 v174, v164, v165
	v_max3_f32 v175, v159, v160, v161
	v_max3_f32 v176, v154, v155, v156
	v_max3_f32 v177, v157, v2, v3
	v_max3_f32 v173, v173, v174, v158
	v_max3_f32 v174, v175, v176, v177
	v_max3_f32 v175, v178, v171, v172
	v_max3_f32 v173, v173, v174, v175
	v_max3_f32 v173, v173, v179, v180
	v_cmp_lt_f32_e32 vcc, 0, v173
	s_cbranch_vccz .LBB0_7
