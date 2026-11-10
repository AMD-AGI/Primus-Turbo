L:
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
v_add_u32_e32 v236, 0xffffff80, v236
s_addc_u32 s29, s29, 0
s_add_i32 s36, s36, s30
s_add_i32 s31, s31, s30
s_add_i32 s34, s34, s49
s_add_i32 s11, s11, s49
s_cbranch_vccz L
L:
ds_read_b128 v[198:201], v207 offset:42880
ds_read_b128 v[192:195], v207 offset:42368
v_exp_f32_e64 v138, v138 clamp
v_exp_f32_e64 v139, v139 clamp
v_exp_f32_e64 v140, v140 clamp
v_exp_f32_e64 v141, v141 clamp
v_exp_f32_e64 v70, v70 clamp
v_exp_f32_e64 v71, v71 clamp
v_exp_f32_e64 v72, v72 clamp
v_exp_f32_e64 v73, v73 clamp
v_exp_f32_e64 v66, v66 clamp
v_exp_f32_e64 v67, v67 clamp
v_exp_f32_e64 v68, v68 clamp
v_exp_f32_e64 v69, v69 clamp
v_exp_f32_e64 v142, v142 clamp
v_exp_f32_e64 v143, v143 clamp
v_exp_f32_e64 v144, v144 clamp
v_exp_f32_e64 v145, v145 clamp
v_cvt_pk_bf16_f32 v181, v160, v161
v_cvt_pk_bf16_f32 v180, v158, v159
v_cvt_pk_bf16_f32 v179, v168, v169
v_cvt_pk_bf16_f32 v178, v166, v167
v_cvt_pk_bf16_f32 v185, v156, v157
v_cvt_pk_bf16_f32 v184, v154, v155
v_cvt_pk_bf16_f32 v183, v164, v165
v_cvt_pk_bf16_f32 v182, v162, v163
v_cvt_pk_bf16_f32 v173, v68, v69
v_cvt_pk_bf16_f32 v172, v66, v67
v_cvt_pk_bf16_f32 v171, v140, v141
v_cvt_pk_bf16_f32 v170, v138, v139
v_cvt_pk_bf16_f32 v177, v144, v145
v_cvt_pk_bf16_f32 v176, v142, v143
v_cvt_pk_bf16_f32 v175, v72, v73
v_cvt_pk_bf16_f32 v174, v70, v71
v_mfma_f32_16x16x32_bf16 v[66:69], v[238:241], v[178:181], v[146:149]
v_mfma_f32_16x16x32_bf16 v[138:141], v[238:241], v[182:185], v[150:153]
v_mfma_f32_16x16x32_bf16 v[70:73], v[238:241], v[170:173], v[66:69]
v_mfma_f32_16x16x32_bf16 v[66:69], v[238:241], v[174:177], v[138:141]
ds_read_b128 v[162:165], v207 offset:42624
ds_read_b128 v[166:169], v207 offset:43136
ds_read_b128 v[212:215], v207 offset:42400
ds_read_b128 v[242:245], v207 offset:42912
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[138:141], v[192:195], v[82:85], v[74:77]
ds_read_b128 v[216:219], v207 offset:42432
ds_read_b128 v[246:249], v207 offset:42720
ds_read_b128 v[220:223], v210 offset:51232
v_mfma_f32_16x16x32_bf16 v[142:145], v[198:201], v[82:85], v[74:77]
ds_read_b128 v[224:227], v210 offset:51488
v_mfma_f32_16x16x32_bf16 v[146:149], v[192:195], v[106:109], v[78:81]
ds_read_b128 v[192:195], v207 offset:42464
v_mfma_f32_16x16x32_bf16 v[150:153], v[198:201], v[106:109], v[78:81]
ds_read_b128 v[198:201], v207 offset:43168
s_waitcnt lgkmcnt(9)
v_mfma_f32_16x16x32_bf16 v[154:157], v[162:165], v[82:85], v[130:133]
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[158:161], v[166:169], v[82:85], v[130:133]
v_mfma_f32_16x16x32_bf16 v[162:165], v[162:165], v[106:109], v[134:137]
v_mfma_f32_16x16x32_bf16 v[166:169], v[166:169], v[106:109], v[134:137]
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_bf16 v[138:141], v[212:215], v[86:89], v[138:141]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[142:145], v[242:245], v[86:89], v[142:145]
v_mfma_f32_16x16x32_bf16 v[146:149], v[212:215], v[110:113], v[146:149]
ds_read_b128 v[212:215], v207 offset:42656
v_mfma_f32_16x16x32_bf16 v[150:153], v[242:245], v[110:113], v[150:153]
ds_read_b128 v[242:245], v207 offset:42688
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[154:157], v[212:215], v[86:89], v[154:157]
v_mfma_f32_16x16x32_bf16 v[158:161], v[198:201], v[86:89], v[158:161]
v_mfma_f32_16x16x32_bf16 v[162:165], v[212:215], v[110:113], v[162:165]
ds_read_b128 v[212:215], v207 offset:42976
v_mfma_f32_16x16x32_bf16 v[166:169], v[198:201], v[110:113], v[166:169]
ds_read_b128 v[198:201], v207 offset:42944
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[142:145], v[198:201], v[90:93], v[142:145]
v_mfma_f32_16x16x32_bf16 v[150:153], v[198:201], v[114:117], v[150:153]
ds_read_b128 v[198:201], v207 offset:43200
v_mfma_f32_16x16x32_bf16 v[138:141], v[216:219], v[90:93], v[138:141]
v_mfma_f32_16x16x32_bf16 v[146:149], v[216:219], v[114:117], v[146:149]
ds_read_b128 v[216:219], v207 offset:43232
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[158:161], v[198:201], v[90:93], v[158:161]
v_mfma_f32_16x16x32_bf16 v[166:169], v[198:201], v[114:117], v[166:169]
v_mfma_f32_16x16x32_bf16 v[138:141], v[192:195], v[94:97], v[138:141]
v_mfma_f32_16x16x32_bf16 v[146:149], v[192:195], v[118:121], v[146:149]
ds_read_b128 v[192:195], v210 offset:50688
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[158:161], v[216:219], v[94:97], v[158:161]
v_mfma_f32_16x16x32_bf16 v[166:169], v[216:219], v[118:121], v[166:169]
ds_read_b128 v[216:219], v210 offset:51200
v_mfma_f32_16x16x32_bf16 v[142:145], v[212:215], v[94:97], v[142:145]
v_mfma_f32_16x16x32_bf16 v[150:153], v[212:215], v[118:121], v[150:153]
ds_read_b128 v[212:215], v210 offset:50720
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[142:145], v[216:219], v[98:101], v[142:145]
v_mfma_f32_16x16x32_bf16 v[198:201], v[192:195], v[122:125], v[146:149]
v_mfma_f32_16x16x32_bf16 v[150:153], v[216:219], v[122:125], v[150:153]
s_nop 1
ds_read_b128 v[146:149], v210 offset:50944
ds_read_b128 v[216:219], v210 offset:51456
v_mfma_f32_16x16x32_bf16 v[154:157], v[242:245], v[90:93], v[154:157]
v_mfma_f32_16x16x32_bf16 v[162:165], v[242:245], v[114:117], v[162:165]
v_mfma_f32_16x16x32_bf16 v[138:141], v[192:195], v[98:101], v[138:141]
ds_read_b128 v[192:195], v210 offset:50976
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[154:157], v[246:249], v[94:97], v[154:157]
s_barrier
v_mfma_f32_16x16x32_bf16 v[162:165], v[246:249], v[118:121], v[162:165]
v_mfma_f32_16x16x32_bf16 v[154:157], v[146:149], v[98:101], v[154:157]
v_mfma_f32_16x16x32_bf16 v[242:245], v[216:219], v[98:101], v[158:161]
v_mfma_f32_16x16x32_bf16 v[246:249], v[146:149], v[122:125], v[162:165]
v_mfma_f32_16x16x32_bf16 v[250:253], v[216:219], v[122:125], v[166:169]
v_mfma_f32_16x16x32_bf16 v[166:169], v[212:215], v[102:105], v[138:141]
v_mfma_f32_16x16x32_bf16 v[146:149], v[220:223], v[102:105], v[142:145]
v_mfma_f32_16x16x32_bf16 v[162:165], v[212:215], v[126:129], v[198:201]
v_mfma_f32_16x16x32_bf16 v[142:145], v[220:223], v[126:129], v[150:153]
v_mfma_f32_16x16x32_bf16 v[158:161], v[192:195], v[102:105], v[154:157]
v_mfma_f32_16x16x32_bf16 v[138:141], v[224:227], v[102:105], v[242:245]
v_mfma_f32_16x16x32_bf16 v[154:157], v[192:195], v[126:129], v[246:249]
v_mfma_f32_16x16x32_bf16 v[150:153], v[224:227], v[126:129], v[250:253]
ds_read_b64_tr_b16 v[198:199], v205 offset:24960
ds_read_b64_tr_b16 v[200:201], v205 offset:25216
ds_read_b64_tr_b16 v[218:219], v205 offset:25280
ds_read_b64_tr_b16 v[216:217], v205 offset:25024
ds_read_b64_tr_b16 v[192:193], v206 offset:24960
ds_read_b64_tr_b16 v[194:195], v206 offset:25216
ds_read_b64_tr_b16 v[214:215], v206 offset:25280
ds_read_b64_tr_b16 v[212:213], v206 offset:25024
s_mov_b32 m0, s44
s_nop 0
buffer_load_dwordx4 v188, s[12:15], s11 offen lds
s_mov_b32 m0, s43
s_nop 0
buffer_load_dwordx4 v189, s[12:15], s11 offen lds
s_mov_b32 m0, s42
s_nop 0
buffer_load_dwordx4 v190, s[12:15], s11 offen lds
s_mov_b32 m0, s35
s_nop 0
buffer_load_dwordx4 v208, s[16:19], s31 offen lds
s_mov_b32 m0, s25
s_nop 0
buffer_load_dwordx4 v209, s[16:19], s31 offen lds
s_setprio 1
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[46:49], v[198:201], v[178:181], v[46:49]
ds_read_b64_tr_b16 v[242:243], v206 offset:33664
ds_read_b64_tr_b16 v[244:245], v206 offset:33920
ds_read_b64_tr_b16 v[252:253], v206 offset:33984
ds_read_b64_tr_b16 v[250:251], v206 offset:33728
v_mfma_f32_16x16x32_bf16 v[42:45], v[198:201], v[182:185], v[42:45]
ds_read_b64_tr_b16 v[246:247], v205 offset:33664
ds_read_b64_tr_b16 v[248:249], v205 offset:33920
ds_read_b64_tr_b16 v[200:201], v205 offset:33984
ds_read_b64_tr_b16 v[198:199], v205 offset:33728
s_waitcnt lgkmcnt(10)
v_mfma_f32_16x16x32_bf16 v[38:41], v[192:195], v[178:181], v[38:41]
v_cmp_le_i64_e32 vcc, s[28:29], v[186:187]
s_and_b64 vcc, exec, vcc
v_mfma_f32_16x16x32_bf16 v[34:37], v[192:195], v[182:185], v[34:37]
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[54:57], v[212:215], v[178:181], v[54:57]
v_mfma_f32_16x16x32_bf16 v[50:53], v[212:215], v[182:185], v[50:53]
v_mfma_f32_16x16x32_bf16 v[62:65], v[216:219], v[178:181], v[62:65]
v_mfma_f32_16x16x32_bf16 v[58:61], v[216:219], v[182:185], v[58:61]
s_cbranch_vccnz L
v_add_u32_e32 v192, 0x60, v236
v_cmp_lt_i32_e64 s[50:51], v192, 0
v_cmp_lt_i32_e64 s[52:53], v192, 1
v_cndmask_b32_e64 v166, v166, v237, s[50:51]
v_cndmask_b32_e64 v167, v167, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 2
v_cmp_lt_i32_e64 s[52:53], v192, 3
v_cndmask_b32_e64 v168, v168, v237, s[50:51]
v_cndmask_b32_e64 v169, v169, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 16
v_cmp_lt_i32_e64 s[52:53], v192, 17
v_cndmask_b32_e64 v158, v158, v237, s[50:51]
v_cndmask_b32_e64 v159, v159, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 18
v_cmp_lt_i32_e64 s[52:53], v192, 19
v_cndmask_b32_e64 v160, v160, v237, s[50:51]
v_cndmask_b32_e64 v161, v161, v237, s[52:53]
v_add_u32_e32 v192, 0x70, v236
v_cmp_lt_i32_e64 s[50:51], v192, 0
v_cmp_lt_i32_e64 s[52:53], v192, 1
v_cndmask_b32_e64 v162, v162, v237, s[50:51]
v_cndmask_b32_e64 v163, v163, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 2
v_cmp_lt_i32_e64 s[52:53], v192, 3
v_cndmask_b32_e64 v164, v164, v237, s[50:51]
v_cndmask_b32_e64 v165, v165, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 16
v_cmp_lt_i32_e64 s[52:53], v192, 17
v_cndmask_b32_e64 v154, v154, v237, s[50:51]
v_cndmask_b32_e64 v155, v155, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 18
v_cmp_lt_i32_e64 s[52:53], v192, 19
v_cndmask_b32_e64 v156, v156, v237, s[50:51]
v_cndmask_b32_e64 v157, v157, v237, s[52:53]
v_add_u32_e32 v192, 64, v236
v_cmp_lt_i32_e64 s[50:51], v192, 0
v_cmp_lt_i32_e64 s[52:53], v192, 1
v_cndmask_b32_e64 v146, v146, v237, s[50:51]
v_cndmask_b32_e64 v147, v147, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 2
v_cmp_lt_i32_e64 s[52:53], v192, 3
v_cndmask_b32_e64 v148, v148, v237, s[50:51]
v_cndmask_b32_e64 v149, v149, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 16
v_cmp_lt_i32_e64 s[52:53], v192, 17
v_cndmask_b32_e64 v138, v138, v237, s[50:51]
v_cndmask_b32_e64 v139, v139, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 18
v_cmp_lt_i32_e64 s[52:53], v192, 19
v_cndmask_b32_e64 v140, v140, v237, s[50:51]
v_cndmask_b32_e64 v141, v141, v237, s[52:53]
v_add_u32_e32 v192, 0x50, v236
v_cmp_lt_i32_e64 s[50:51], v192, 0
v_cmp_lt_i32_e64 s[52:53], v192, 1
v_cndmask_b32_e64 v142, v142, v237, s[50:51]
v_cndmask_b32_e64 v143, v143, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 2
v_cmp_lt_i32_e64 s[52:53], v192, 3
v_cndmask_b32_e64 v144, v144, v237, s[50:51]
v_cndmask_b32_e64 v145, v145, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 16
v_cmp_lt_i32_e64 s[52:53], v192, 17
v_cndmask_b32_e64 v150, v150, v237, s[50:51]
v_cndmask_b32_e64 v151, v151, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 18
v_cmp_lt_i32_e64 s[52:53], v192, 19
v_cndmask_b32_e64 v152, v152, v237, s[50:51]
v_cndmask_b32_e64 v153, v153, v237, s[52:53]
L:
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[192:195], v[242:245], v[178:181], v[2:5]
v_mfma_f32_16x16x32_bf16 v[242:245], v[242:245], v[182:185], v[6:9]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[212:215], v[246:249], v[178:181], v[14:17]
v_mfma_f32_16x16x32_bf16 v[246:249], v[246:249], v[182:185], v[10:13]
ds_read_b64_tr_b16 v[2:3], v206 offset:25472
ds_read_b64_tr_b16 v[4:5], v206 offset:25728
ds_read_b64_tr_b16 v[6:7], v205 offset:25472
ds_read_b64_tr_b16 v[8:9], v205 offset:25728
ds_read_b64_tr_b16 v[10:11], v206 offset:25536
ds_read_b64_tr_b16 v[12:13], v206 offset:25792
ds_read_b64_tr_b16 v[14:15], v205 offset:25536
ds_read_b64_tr_b16 v[16:17], v205 offset:25792
v_mfma_f32_16x16x32_bf16 v[216:219], v[250:253], v[178:181], v[22:25]
v_mfma_f32_16x16x32_bf16 v[250:253], v[250:253], v[182:185], v[18:21]
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[178:181], v[198:201], v[178:181], v[30:33]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[18:21], v[2:5], v[170:173], v[38:41]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[22:25], v[6:9], v[170:173], v[46:49]
v_mfma_f32_16x16x32_bf16 v[6:9], v[6:9], v[174:177], v[42:45]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[30:33], v[14:17], v[170:173], v[62:65]
ds_read_b64_tr_b16 v[38:39], v206 offset:34176
ds_read_b64_tr_b16 v[40:41], v206 offset:34432
ds_read_b64_tr_b16 v[42:43], v205 offset:34176
ds_read_b64_tr_b16 v[44:45], v205 offset:34432
ds_read_b64_tr_b16 v[46:47], v206 offset:34240
ds_read_b64_tr_b16 v[48:49], v206 offset:34496
ds_read_b64_tr_b16 v[62:63], v205 offset:34240
ds_read_b64_tr_b16 v[64:65], v205 offset:34496
v_mfma_f32_16x16x32_bf16 v[182:185], v[198:201], v[182:185], v[26:29]
v_mfma_f32_16x16x32_bf16 v[2:5], v[2:5], v[174:177], v[34:37]
v_mfma_f32_16x16x32_bf16 v[26:29], v[10:13], v[170:173], v[54:57]
v_mfma_f32_16x16x32_bf16 v[10:13], v[10:13], v[174:177], v[50:53]
v_mfma_f32_16x16x32_bf16 v[14:17], v[14:17], v[174:177], v[58:61]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[34:37], v[38:41], v[170:173], v[192:195]
v_mfma_f32_16x16x32_bf16 v[50:53], v[38:41], v[174:177], v[242:245]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[38:41], v[42:45], v[170:173], v[212:215]
v_mfma_f32_16x16x32_bf16 v[54:57], v[42:45], v[174:177], v[246:249]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[42:45], v[46:49], v[170:173], v[216:219]
v_mfma_f32_16x16x32_bf16 v[58:61], v[46:49], v[174:177], v[250:253]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[46:49], v[62:65], v[170:173], v[178:181]
v_max3_f32 v170, v166, v167, v168
v_max_f32_e32 v173, v169, v162
v_max3_f32 v171, v143, v144, v145
v_mfma_f32_16x16x32_bf16 v[62:65], v[62:65], v[174:177], v[182:185]
v_max_f32_e32 v174, v164, v165
v_max3_f32 v175, v159, v160, v161
v_max3_f32 v176, v154, v155, v156
v_max3_f32 v177, v157, v146, v147
v_max3_f32 v178, v148, v149, v142
v_max3_f32 v172, v138, v139, v140
v_max3_f32 v173, v170, v173, v163
v_max3_f32 v173, v173, v174, v158
v_max3_f32 v174, v175, v176, v177
v_max3_f32 v175, v178, v171, v172
v_max3_f32 v179, v141, v150, v151
v_max_f32_e32 v180, v152, v153
v_max3_f32 v173, v173, v174, v175
v_max3_f32 v173, v173, v179, v180
v_cmp_lt_f32_e32 vcc, 0, v173
s_cbranch_vccz L
v_max_f32_e32 v74, v169, v158
v_max3_f32 v76, v147, v148, v149
v_max3_f32 v75, v160, v161, v146
v_max3_f32 v74, v170, v74, v159
v_max3_f32 v76, v76, v172, v141
v_max3_f32 v74, v74, v75, v76
v_max_f32_e32 v75, v162, v163
v_max3_f32 v76, v165, v154, v155
v_max3_f32 v78, v150, v151, v152
v_max3_f32 v77, v156, v157, v142
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
v_pk_add_f32 v[138:139], v[138:139], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[140:141], v[140:141], v[74:75] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_xor_b32_e32 v74, 0x80000000, v1
v_mov_b32_e32 v80, v76
v_pk_add_f32 v[162:163], v[162:163], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[164:165], v[164:165], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[154:155], v[154:155], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[156:157], v[156:157], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[142:143], v[142:143], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[144:145], v[144:145], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[150:151], v[150:151], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[152:153], v[152:153], v[78:79] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_mov_b32_e32 v78, v74
v_permlane16_swap_b32_e32 v76, v80
v_pk_mul_f32 v[68:69], v[80:81], v[68:69] op_sel_hi:[0,1]
v_pk_mul_f32 v[66:67], v[80:81], v[66:67] op_sel_hi:[0,1]
v_pk_mul_f32 v[72:73], v[76:77], v[72:73] op_sel_hi:[0,1]
v_pk_mul_f32 v[70:71], v[76:77], v[70:71] op_sel_hi:[0,1]
v_pk_mul_f32 v[52:53], v[80:81], v[52:53] op_sel_hi:[0,1]
v_pk_mul_f32 v[50:51], v[80:81], v[50:51] op_sel_hi:[0,1]
v_pk_mul_f32 v[36:37], v[76:77], v[36:37] op_sel_hi:[0,1]
v_pk_mul_f32 v[34:35], v[76:77], v[34:35] op_sel_hi:[0,1]
v_pk_mul_f32 v[4:5], v[80:81], v[4:5] op_sel_hi:[0,1]
v_pk_mul_f32 v[2:3], v[80:81], v[2:3] op_sel_hi:[0,1]
v_pk_mul_f32 v[20:21], v[76:77], v[20:21] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], v[76:77], v[18:19] op_sel_hi:[0,1]
v_pk_mul_f32 v[56:57], v[80:81], v[56:57] op_sel_hi:[0,1]
v_pk_mul_f32 v[54:55], v[80:81], v[54:55] op_sel_hi:[0,1]
v_pk_mul_f32 v[40:41], v[76:77], v[40:41] op_sel_hi:[0,1]
v_pk_mul_f32 v[38:39], v[76:77], v[38:39] op_sel_hi:[0,1]
v_pk_mul_f32 v[8:9], v[80:81], v[8:9] op_sel_hi:[0,1]
v_pk_mul_f32 v[6:7], v[80:81], v[6:7] op_sel_hi:[0,1]
v_pk_mul_f32 v[24:25], v[76:77], v[24:25] op_sel_hi:[0,1]
v_pk_mul_f32 v[22:23], v[76:77], v[22:23] op_sel_hi:[0,1]
v_pk_mul_f32 v[60:61], v[80:81], v[60:61] op_sel_hi:[0,1]
v_pk_mul_f32 v[58:59], v[80:81], v[58:59] op_sel_hi:[0,1]
v_pk_mul_f32 v[44:45], v[76:77], v[44:45] op_sel_hi:[0,1]
v_pk_mul_f32 v[42:43], v[76:77], v[42:43] op_sel_hi:[0,1]
v_pk_mul_f32 v[12:13], v[80:81], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[10:11], v[80:81], v[10:11] op_sel_hi:[0,1]
v_pk_mul_f32 v[28:29], v[76:77], v[28:29] op_sel_hi:[0,1]
v_pk_mul_f32 v[26:27], v[76:77], v[26:27] op_sel_hi:[0,1]
v_pk_mul_f32 v[64:65], v[80:81], v[64:65] op_sel_hi:[0,1]
v_pk_mul_f32 v[62:63], v[80:81], v[62:63] op_sel_hi:[0,1]
v_pk_mul_f32 v[48:49], v[76:77], v[48:49] op_sel_hi:[0,1]
v_pk_mul_f32 v[46:47], v[76:77], v[46:47] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], v[80:81], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], v[80:81], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[32:33], v[76:77], v[32:33] op_sel_hi:[0,1]
v_pk_mul_f32 v[30:31], v[76:77], v[30:31] op_sel_hi:[0,1]
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
L:
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
ds_read_b128 v[198:201], v207 offset:512
ds_read_b128 v[192:195], v207
v_exp_f32_e64 v146, v146 clamp
v_exp_f32_e64 v147, v147 clamp
v_exp_f32_e64 v148, v148 clamp
v_exp_f32_e64 v149, v149 clamp
v_exp_f32_e64 v142, v142 clamp
v_exp_f32_e64 v143, v143 clamp
v_exp_f32_e64 v144, v144 clamp
v_exp_f32_e64 v145, v145 clamp
v_exp_f32_e64 v138, v138 clamp
v_exp_f32_e64 v139, v139 clamp
v_exp_f32_e64 v140, v140 clamp
v_exp_f32_e64 v141, v141 clamp
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
v_cvt_pk_bf16_f32 v173, v140, v141
v_cvt_pk_bf16_f32 v172, v138, v139
v_cvt_pk_bf16_f32 v171, v148, v149
v_cvt_pk_bf16_f32 v170, v146, v147
v_cvt_pk_bf16_f32 v177, v152, v153
v_cvt_pk_bf16_f32 v176, v150, v151
v_cvt_pk_bf16_f32 v175, v144, v145
v_cvt_pk_bf16_f32 v174, v142, v143
v_mfma_f32_16x16x32_bf16 v[70:73], v[238:241], v[178:181], v[70:73]
v_mfma_f32_16x16x32_bf16 v[66:69], v[238:241], v[182:185], v[66:69]
v_mfma_f32_16x16x32_bf16 v[146:149], v[238:241], v[170:173], v[70:73]
v_mfma_f32_16x16x32_bf16 v[150:153], v[238:241], v[174:177], v[66:69]
ds_read_b128 v[162:165], v207 offset:256
ds_read_b128 v[166:169], v207 offset:768
ds_read_b128 v[212:215], v207 offset:32
ds_read_b128 v[220:223], v207 offset:544
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[66:69], v[192:195], v[82:85], v[74:77]
ds_read_b128 v[216:219], v207 offset:64
ds_read_b128 v[224:227], v207 offset:352
ds_read_b128 v[242:245], v210 offset:8608
v_mfma_f32_16x16x32_bf16 v[70:73], v[198:201], v[82:85], v[74:77]
ds_read_b128 v[246:249], v210 offset:9120
v_mfma_f32_16x16x32_bf16 v[138:141], v[192:195], v[106:109], v[78:81]
ds_read_b128 v[192:195], v207 offset:96
v_mfma_f32_16x16x32_bf16 v[142:145], v[198:201], v[106:109], v[78:81]
ds_read_b128 v[198:201], v207 offset:800
s_waitcnt lgkmcnt(9)
v_mfma_f32_16x16x32_bf16 v[154:157], v[162:165], v[82:85], v[130:133]
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[158:161], v[166:169], v[82:85], v[130:133]
v_mfma_f32_16x16x32_bf16 v[162:165], v[162:165], v[106:109], v[134:137]
v_mfma_f32_16x16x32_bf16 v[166:169], v[166:169], v[106:109], v[134:137]
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_bf16 v[66:69], v[212:215], v[86:89], v[66:69]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[70:73], v[220:223], v[86:89], v[70:73]
v_mfma_f32_16x16x32_bf16 v[138:141], v[212:215], v[110:113], v[138:141]
ds_read_b128 v[212:215], v207 offset:288
v_mfma_f32_16x16x32_bf16 v[142:145], v[220:223], v[110:113], v[142:145]
ds_read_b128 v[220:223], v207 offset:320
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[154:157], v[212:215], v[86:89], v[154:157]
v_mfma_f32_16x16x32_bf16 v[158:161], v[198:201], v[86:89], v[158:161]
v_mfma_f32_16x16x32_bf16 v[162:165], v[212:215], v[110:113], v[162:165]
ds_read_b128 v[212:215], v207 offset:608
v_mfma_f32_16x16x32_bf16 v[166:169], v[198:201], v[110:113], v[166:169]
ds_read_b128 v[198:201], v207 offset:576
v_mfma_f32_16x16x32_bf16 v[66:69], v[216:219], v[90:93], v[66:69]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[70:73], v[198:201], v[90:93], v[70:73]
v_mfma_f32_16x16x32_bf16 v[138:141], v[216:219], v[114:117], v[138:141]
ds_read_b128 v[216:219], v207 offset:864
v_mfma_f32_16x16x32_bf16 v[142:145], v[198:201], v[114:117], v[142:145]
ds_read_b128 v[198:201], v207 offset:832
v_mfma_f32_16x16x32_bf16 v[66:69], v[192:195], v[94:97], v[66:69]
v_mfma_f32_16x16x32_bf16 v[138:141], v[192:195], v[118:121], v[138:141]
ds_read_b128 v[192:195], v210 offset:8320
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[158:161], v[198:201], v[90:93], v[158:161]
v_mfma_f32_16x16x32_bf16 v[166:169], v[198:201], v[114:117], v[166:169]
ds_read_b128 v[198:201], v210 offset:8832
v_mfma_f32_16x16x32_bf16 v[154:157], v[220:223], v[90:93], v[154:157]
v_mfma_f32_16x16x32_bf16 v[162:165], v[220:223], v[114:117], v[162:165]
ds_read_b128 v[220:223], v210 offset:8352
v_mfma_f32_16x16x32_bf16 v[158:161], v[216:219], v[94:97], v[158:161]
v_mfma_f32_16x16x32_bf16 v[166:169], v[216:219], v[118:121], v[166:169]
ds_read_b128 v[216:219], v210 offset:9088
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_bf16 v[66:69], v[192:195], v[98:101], v[66:69]
v_mfma_f32_16x16x32_bf16 v[192:195], v[192:195], v[122:125], v[138:141]
s_nop 2
ds_read_b128 v[138:141], v210 offset:8576
v_mfma_f32_16x16x32_bf16 v[154:157], v[224:227], v[94:97], v[154:157]
v_mfma_f32_16x16x32_bf16 v[162:165], v[224:227], v[118:121], v[162:165]
ds_read_b128 v[224:227], v210 offset:8864
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[70:73], v[212:215], v[94:97], v[70:73]
s_barrier
v_mfma_f32_16x16x32_bf16 v[142:145], v[212:215], v[118:121], v[142:145]
v_mfma_f32_16x16x32_bf16 v[70:73], v[198:201], v[98:101], v[70:73]
v_mfma_f32_16x16x32_bf16 v[142:145], v[198:201], v[122:125], v[142:145]
v_mfma_f32_16x16x32_bf16 v[154:157], v[138:141], v[98:101], v[154:157]
v_mfma_f32_16x16x32_bf16 v[198:201], v[216:219], v[98:101], v[158:161]
v_mfma_f32_16x16x32_bf16 v[212:215], v[138:141], v[122:125], v[162:165]
v_mfma_f32_16x16x32_bf16 v[216:219], v[216:219], v[122:125], v[166:169]
v_mfma_f32_16x16x32_bf16 v[166:169], v[220:223], v[102:105], v[66:69]
v_mfma_f32_16x16x32_bf16 v[138:141], v[224:227], v[102:105], v[70:73]
v_mfma_f32_16x16x32_bf16 v[162:165], v[220:223], v[126:129], v[192:195]
v_mfma_f32_16x16x32_bf16 v[70:73], v[224:227], v[126:129], v[142:145]
v_mfma_f32_16x16x32_bf16 v[158:161], v[242:245], v[102:105], v[154:157]
v_mfma_f32_16x16x32_bf16 v[66:69], v[246:249], v[102:105], v[198:201]
v_mfma_f32_16x16x32_bf16 v[154:157], v[242:245], v[126:129], v[212:215]
v_mfma_f32_16x16x32_bf16 v[142:145], v[246:249], v[126:129], v[216:219]
v_add_u32_e32 v192, 0x10700, v206
ds_read_b64_tr_b16 v[198:199], v192
v_add_u32_e32 v192, 0x10800, v206
ds_read_b64_tr_b16 v[200:201], v192
v_add_u32_e32 v192, 0x10700, v205
ds_read_b64_tr_b16 v[242:243], v192
v_add_u32_e32 v192, 0x10800, v205
ds_read_b64_tr_b16 v[244:245], v192
v_add_u32_e32 v192, 0x10740, v206
ds_read_b64_tr_b16 v[246:247], v192
v_add_u32_e32 v192, 0x10840, v206
ds_read_b64_tr_b16 v[248:249], v192
v_add_u32_e32 v192, 0x10740, v205
ds_read_b64_tr_b16 v[250:251], v192
v_add_u32_e32 v192, 0x10840, v205
ds_read_b64_tr_b16 v[252:253], v192
s_cmp_ge_i32 s20, s37
s_cbranch_scc1 L
v_add_u32_e32 v192, 32, v236
v_cmp_lt_i32_e64 s[50:51], v192, 0
v_cmp_lt_i32_e64 s[52:53], v192, 1
v_cndmask_b32_e64 v166, v166, v237, s[50:51]
v_cndmask_b32_e64 v167, v167, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 2
v_cmp_lt_i32_e64 s[52:53], v192, 3
v_cndmask_b32_e64 v168, v168, v237, s[50:51]
v_cndmask_b32_e64 v169, v169, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 16
v_cmp_lt_i32_e64 s[52:53], v192, 17
v_cndmask_b32_e64 v158, v158, v237, s[50:51]
v_cndmask_b32_e64 v159, v159, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 18
v_cmp_lt_i32_e64 s[52:53], v192, 19
v_cndmask_b32_e64 v160, v160, v237, s[50:51]
v_cndmask_b32_e64 v161, v161, v237, s[52:53]
v_add_u32_e32 v192, 48, v236
v_cmp_lt_i32_e64 s[50:51], v192, 0
v_cmp_lt_i32_e64 s[52:53], v192, 1
v_cndmask_b32_e64 v162, v162, v237, s[50:51]
v_cndmask_b32_e64 v163, v163, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 2
v_cmp_lt_i32_e64 s[52:53], v192, 3
v_cndmask_b32_e64 v164, v164, v237, s[50:51]
v_cndmask_b32_e64 v165, v165, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 16
v_cmp_lt_i32_e64 s[52:53], v192, 17
v_cndmask_b32_e64 v154, v154, v237, s[50:51]
v_cndmask_b32_e64 v155, v155, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 18
v_cmp_lt_i32_e64 s[52:53], v192, 19
v_cndmask_b32_e64 v156, v156, v237, s[50:51]
v_cndmask_b32_e64 v157, v157, v237, s[52:53]
v_add_u32_e32 v192, 16, v236
v_cmp_lt_i32_e64 s[50:51], v236, 0
v_cmp_lt_i32_e64 s[52:53], v236, 1
v_cndmask_b32_e64 v138, v138, v237, s[50:51]
v_cndmask_b32_e64 v139, v139, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v236, 2
v_cmp_lt_i32_e64 s[52:53], v236, 3
v_cndmask_b32_e64 v140, v140, v237, s[50:51]
v_cndmask_b32_e64 v141, v141, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v236, 16
v_cmp_lt_i32_e64 s[52:53], v236, 17
v_cndmask_b32_e64 v66, v66, v237, s[50:51]
v_cndmask_b32_e64 v67, v67, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v236, 18
v_cmp_lt_i32_e64 s[52:53], v236, 19
v_cndmask_b32_e64 v68, v68, v237, s[50:51]
v_cndmask_b32_e64 v69, v69, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 0
v_cmp_lt_i32_e64 s[52:53], v192, 1
v_cndmask_b32_e64 v70, v70, v237, s[50:51]
v_cndmask_b32_e64 v71, v71, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 2
v_cmp_lt_i32_e64 s[52:53], v192, 3
v_cndmask_b32_e64 v72, v72, v237, s[50:51]
v_cndmask_b32_e64 v73, v73, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 16
v_cmp_lt_i32_e64 s[52:53], v192, 17
v_cndmask_b32_e64 v142, v142, v237, s[50:51]
v_cndmask_b32_e64 v143, v143, v237, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v192, 18
v_cmp_lt_i32_e64 s[52:53], v192, 19
v_cndmask_b32_e64 v144, v144, v237, s[50:51]
v_cndmask_b32_e64 v145, v145, v237, s[52:53]
L:
s_mov_b32 m0, s45
s_nop 0
buffer_load_dwordx4 v188, s[12:15], s34 offen lds
s_mov_b32 m0, s46
s_nop 0
buffer_load_dwordx4 v189, s[12:15], s34 offen lds
s_mov_b32 m0, s47
s_nop 0
buffer_load_dwordx4 v190, s[12:15], s34 offen lds
s_mov_b32 m0, s40
s_nop 0
buffer_load_dwordx4 v208, s[16:19], s36 offen lds
s_mov_b32 m0, s39
s_nop 0
buffer_load_dwordx4 v209, s[16:19], s36 offen lds
s_setprio 1
v_add_u32_e32 v192, 0x12900, v206
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[18:21], v[198:201], v[178:181], v[18:21]
v_mfma_f32_16x16x32_bf16 v[2:5], v[198:201], v[182:185], v[2:5]
ds_read_b64_tr_b16 v[198:199], v192
v_add_u32_e32 v192, 0x12a00, v206
ds_read_b64_tr_b16 v[200:201], v192
v_add_u32_e32 v192, 0x12900, v205
ds_read_b64_tr_b16 v[216:217], v192
v_add_u32_e32 v192, 0x12a00, v205
ds_read_b64_tr_b16 v[218:219], v192
v_add_u32_e32 v192, 0x12940, v206
ds_read_b64_tr_b16 v[220:221], v192
v_add_u32_e32 v192, 0x12a40, v206
ds_read_b64_tr_b16 v[222:223], v192
ds_read_b64_tr_b16 v[224:225], v202
ds_read_b64_tr_b16 v[226:227], v196
s_waitcnt lgkmcnt(12)
v_mfma_f32_16x16x32_bf16 v[22:25], v[242:245], v[178:181], v[22:25]
v_mfma_f32_16x16x32_bf16 v[6:9], v[242:245], v[182:185], v[6:9]
s_waitcnt lgkmcnt(10)
v_mfma_f32_16x16x32_bf16 v[26:29], v[246:249], v[178:181], v[26:29]
v_mfma_f32_16x16x32_bf16 v[10:13], v[246:249], v[182:185], v[10:13]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[192:195], v[198:201], v[178:181], v[34:37]
v_mfma_f32_16x16x32_bf16 v[198:201], v[198:201], v[182:185], v[50:53]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[242:245], v[220:223], v[178:181], v[42:45]
v_mfma_f32_16x16x32_bf16 v[246:249], v[220:223], v[182:185], v[58:61]
ds_read_b64_tr_b16 v[34:35], v197
ds_read_b64_tr_b16 v[36:37], v211
ds_read_b64_tr_b16 v[42:43], v203
ds_read_b64_tr_b16 v[44:45], v191
ds_read_b64_tr_b16 v[50:51], v254
ds_read_b64_tr_b16 v[52:53], v255
ds_read_b64_tr_b16 v[58:59], v0
ds_read_b64_tr_b16 v[60:61], v204
v_mfma_f32_16x16x32_bf16 v[30:33], v[250:253], v[178:181], v[30:33]
v_mfma_f32_16x16x32_bf16 v[212:215], v[216:219], v[178:181], v[38:41]
v_mfma_f32_16x16x32_bf16 v[216:219], v[216:219], v[182:185], v[54:57]
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[178:181], v[224:227], v[178:181], v[46:49]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[38:41], v[34:37], v[170:173], v[18:21]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[46:49], v[42:45], v[170:173], v[22:25]
v_mfma_f32_16x16x32_bf16 v[42:45], v[42:45], v[174:177], v[6:9]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[54:57], v[50:53], v[170:173], v[26:29]
v_mfma_f32_16x16x32_bf16 v[50:53], v[50:53], v[174:177], v[10:13]
ds_read_b64_tr_b16 v[6:7], v228
ds_read_b64_tr_b16 v[8:9], v229
s_nop 0
ds_read_b64_tr_b16 v[10:11], v230
ds_read_b64_tr_b16 v[12:13], v231
ds_read_b64_tr_b16 v[18:19], v232
ds_read_b64_tr_b16 v[20:21], v233
ds_read_b64_tr_b16 v[26:27], v234
ds_read_b64_tr_b16 v[28:29], v235
v_mfma_f32_16x16x32_bf16 v[14:17], v[250:253], v[182:185], v[14:17]
v_mfma_f32_16x16x32_bf16 v[182:185], v[224:227], v[182:185], v[62:65]
v_mfma_f32_16x16x32_bf16 v[34:37], v[34:37], v[174:177], v[2:5]
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[62:65], v[58:61], v[170:173], v[30:33]
v_mfma_f32_16x16x32_bf16 v[58:61], v[58:61], v[174:177], v[14:17]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[2:5], v[6:9], v[170:173], v[192:195]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[14:17], v[10:13], v[170:173], v[212:215]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[22:25], v[18:21], v[170:173], v[242:245]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[30:33], v[26:29], v[170:173], v[178:181]
v_max3_f32 v170, v166, v167, v168
v_max_f32_e32 v173, v169, v162
v_max3_f32 v171, v71, v72, v73
v_mfma_f32_16x16x32_bf16 v[6:9], v[6:9], v[174:177], v[198:201]
v_max3_f32 v178, v140, v141, v70
v_max3_f32 v172, v66, v67, v68
v_max3_f32 v173, v170, v173, v163
v_mfma_f32_16x16x32_bf16 v[10:13], v[10:13], v[174:177], v[216:219]
v_max3_f32 v179, v69, v142, v143
v_max_f32_e32 v180, v144, v145
v_mfma_f32_16x16x32_bf16 v[18:21], v[18:21], v[174:177], v[246:249]
v_mfma_f32_16x16x32_bf16 v[26:29], v[26:29], v[174:177], v[182:185]
v_max_f32_e32 v174, v164, v165
v_max3_f32 v175, v159, v160, v161
v_max3_f32 v176, v154, v155, v156
v_max3_f32 v177, v157, v138, v139
v_max3_f32 v173, v173, v174, v158
v_max3_f32 v174, v175, v176, v177
v_max3_f32 v175, v178, v171, v172
v_max3_f32 v173, v173, v174, v175
v_max3_f32 v173, v173, v179, v180
v_cmp_lt_f32_e32 vcc, 0, v173
s_cbranch_vccz L
