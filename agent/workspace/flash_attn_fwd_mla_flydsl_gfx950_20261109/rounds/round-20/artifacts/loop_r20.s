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
v_add_u32_e32 v244, 0xffffff80, v244
s_addc_u32 s29, s29, 0
s_add_i32 s37, s37, s30
s_add_i32 s31, s31, s30
s_add_i32 s34, s34, s48
s_add_i32 s36, s36, s48
s_cbranch_vccz L
L:
ds_read_b128 v[186:189], v205 offset:42368
ds_read_b128 v[190:193], v205 offset:42880
v_exp_f32_e64 v138, v138 clamp
v_exp_f32_e64 v139, v139 clamp
v_exp_f32_e64 v140, v140 clamp
v_exp_f32_e64 v141, v141 clamp
v_exp_f32_e64 v134, v134 clamp
v_exp_f32_e64 v135, v135 clamp
v_exp_f32_e64 v136, v136 clamp
v_exp_f32_e64 v137, v137 clamp
v_exp_f32_e64 v130, v130 clamp
v_exp_f32_e64 v131, v131 clamp
v_exp_f32_e64 v132, v132 clamp
v_exp_f32_e64 v133, v133 clamp
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
v_cvt_pk_bf16_f32 v173, v132, v133
v_cvt_pk_bf16_f32 v172, v130, v131
v_cvt_pk_bf16_f32 v171, v140, v141
v_cvt_pk_bf16_f32 v170, v138, v139
v_cvt_pk_bf16_f32 v177, v144, v145
v_cvt_pk_bf16_f32 v176, v142, v143
v_cvt_pk_bf16_f32 v175, v136, v137
v_cvt_pk_bf16_f32 v174, v134, v135
v_mfma_f32_16x16x32_bf16 v[130:133], v[246:249], v[178:181], v[146:149]
v_mfma_f32_16x16x32_bf16 v[138:141], v[246:249], v[182:185], v[150:153]
v_mfma_f32_16x16x32_bf16 v[134:137], v[246:249], v[170:173], v[130:133]
v_mfma_f32_16x16x32_bf16 v[130:133], v[246:249], v[174:177], v[138:141]
ds_read_b128 v[162:165], v205 offset:42624
ds_read_b128 v[166:169], v205 offset:43136
ds_read_b128 v[194:197], v205 offset:42400
ds_read_b128 v[216:219], v205 offset:42912
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_bf16 v[138:141], v[186:189], v[74:77], v[66:69]
ds_read_b128 v[212:215], v205 offset:42432
ds_read_b128 v[220:223], v205 offset:42720
ds_read_b128 v[224:227], v208 offset:51456
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_bf16 v[142:145], v[190:193], v[74:77], v[66:69]
ds_read_b128 v[228:231], v208 offset:51488
v_mfma_f32_16x16x32_bf16 v[146:149], v[186:189], v[98:101], v[70:73]
ds_read_b128 v[186:189], v205 offset:42464
v_mfma_f32_16x16x32_bf16 v[150:153], v[190:193], v[98:101], v[70:73]
ds_read_b128 v[190:193], v205 offset:43168
s_waitcnt lgkmcnt(9)
v_mfma_f32_16x16x32_bf16 v[154:157], v[162:165], v[74:77], v[122:125]
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[158:161], v[166:169], v[74:77], v[122:125]
v_mfma_f32_16x16x32_bf16 v[162:165], v[162:165], v[98:101], v[126:129]
v_mfma_f32_16x16x32_bf16 v[166:169], v[166:169], v[98:101], v[126:129]
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_bf16 v[138:141], v[194:197], v[78:81], v[138:141]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[142:145], v[216:219], v[78:81], v[142:145]
v_mfma_f32_16x16x32_bf16 v[146:149], v[194:197], v[102:105], v[146:149]
ds_read_b128 v[194:197], v205 offset:42656
v_mfma_f32_16x16x32_bf16 v[150:153], v[216:219], v[102:105], v[150:153]
ds_read_b128 v[216:219], v205 offset:42688
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[154:157], v[194:197], v[78:81], v[154:157]
v_mfma_f32_16x16x32_bf16 v[158:161], v[190:193], v[78:81], v[158:161]
v_mfma_f32_16x16x32_bf16 v[162:165], v[194:197], v[102:105], v[162:165]
ds_read_b128 v[194:197], v205 offset:42976
v_mfma_f32_16x16x32_bf16 v[166:169], v[190:193], v[102:105], v[166:169]
ds_read_b128 v[190:193], v205 offset:42944
v_mfma_f32_16x16x32_bf16 v[138:141], v[212:215], v[82:85], v[138:141]
v_mfma_f32_16x16x32_bf16 v[146:149], v[212:215], v[106:109], v[146:149]
ds_read_b128 v[212:215], v205 offset:43232
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[142:145], v[190:193], v[82:85], v[142:145]
v_mfma_f32_16x16x32_bf16 v[150:153], v[190:193], v[106:109], v[150:153]
ds_read_b128 v[190:193], v205 offset:43200
v_mfma_f32_16x16x32_bf16 v[138:141], v[186:189], v[86:89], v[138:141]
v_mfma_f32_16x16x32_bf16 v[146:149], v[186:189], v[110:113], v[146:149]
ds_read_b128 v[186:189], v208 offset:50688
v_mfma_f32_16x16x32_bf16 v[154:157], v[216:219], v[82:85], v[154:157]
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[158:161], v[190:193], v[82:85], v[158:161]
v_mfma_f32_16x16x32_bf16 v[162:165], v[216:219], v[106:109], v[162:165]
ds_read_b128 v[216:219], v208 offset:51232
v_mfma_f32_16x16x32_bf16 v[166:169], v[190:193], v[106:109], v[166:169]
ds_read_b128 v[190:193], v208 offset:51200
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[138:141], v[186:189], v[90:93], v[138:141]
v_mfma_f32_16x16x32_bf16 v[186:189], v[186:189], v[114:117], v[146:149]
s_nop 2
ds_read_b128 v[146:149], v208 offset:50944
v_mfma_f32_16x16x32_bf16 v[154:157], v[220:223], v[86:89], v[154:157]
v_mfma_f32_16x16x32_bf16 v[158:161], v[212:215], v[86:89], v[158:161]
v_mfma_f32_16x16x32_bf16 v[162:165], v[220:223], v[110:113], v[162:165]
ds_read_b128 v[220:223], v208 offset:50976
v_mfma_f32_16x16x32_bf16 v[166:169], v[212:215], v[110:113], v[166:169]
ds_read_b128 v[212:215], v208 offset:50720
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[142:145], v[194:197], v[86:89], v[142:145]
s_barrier
v_mfma_f32_16x16x32_bf16 v[150:153], v[194:197], v[110:113], v[150:153]
v_mfma_f32_16x16x32_bf16 v[142:145], v[190:193], v[90:93], v[142:145]
v_mfma_f32_16x16x32_bf16 v[150:153], v[190:193], v[114:117], v[150:153]
v_mfma_f32_16x16x32_bf16 v[154:157], v[146:149], v[90:93], v[154:157]
v_mfma_f32_16x16x32_bf16 v[190:193], v[224:227], v[90:93], v[158:161]
v_mfma_f32_16x16x32_bf16 v[194:197], v[146:149], v[114:117], v[162:165]
v_mfma_f32_16x16x32_bf16 v[250:253], v[224:227], v[114:117], v[166:169]
v_mfma_f32_16x16x32_bf16 v[166:169], v[212:215], v[94:97], v[138:141]
v_mfma_f32_16x16x32_bf16 v[146:149], v[216:219], v[94:97], v[142:145]
v_mfma_f32_16x16x32_bf16 v[162:165], v[212:215], v[118:121], v[186:189]
v_mfma_f32_16x16x32_bf16 v[142:145], v[216:219], v[118:121], v[150:153]
v_mfma_f32_16x16x32_bf16 v[158:161], v[220:223], v[94:97], v[154:157]
v_mfma_f32_16x16x32_bf16 v[138:141], v[228:231], v[94:97], v[190:193]
v_mfma_f32_16x16x32_bf16 v[154:157], v[220:223], v[118:121], v[194:197]
v_mfma_f32_16x16x32_bf16 v[150:153], v[228:231], v[118:121], v[250:253]
ds_read_b64_tr_b16 v[186:187], v204 offset:24960
ds_read_b64_tr_b16 v[188:189], v204 offset:25216
ds_read_b64_tr_b16 v[196:197], v204 offset:25280
ds_read_b64_tr_b16 v[194:195], v204 offset:25024
ds_read_b64_tr_b16 v[190:191], v201 offset:24960
ds_read_b64_tr_b16 v[192:193], v201 offset:25216
ds_read_b64_tr_b16 v[214:215], v201 offset:25280
ds_read_b64_tr_b16 v[212:213], v201 offset:25024
s_mov_b32 m0, s44
s_nop 0
buffer_load_dwordx4 v209, s[12:15], s36 offen lds
s_mov_b32 m0, s43
s_nop 0
buffer_load_dwordx4 v210, s[12:15], s36 offen lds
s_mov_b32 m0, s42
s_nop 0
buffer_load_dwordx4 v211, s[12:15], s36 offen lds
s_mov_b32 m0, s35
s_nop 0
buffer_load_dwordx4 v206, s[16:19], s31 offen lds
s_mov_b32 m0, s25
s_nop 0
buffer_load_dwordx4 v207, s[16:19], s31 offen lds
s_setprio 1
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[2:5], v[186:189], v[178:181], v[2:5]
ds_read_b64_tr_b16 v[252:253], v204 offset:33984
ds_read_b64_tr_b16 v[250:251], v204 offset:33728
v_cmp_le_i64_e32 vcc, s[28:29], v[202:203]
v_mfma_f32_16x16x32_bf16 v[6:9], v[186:189], v[182:185], v[6:9]
s_and_b64 vcc, exec, vcc
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[186:189], v[190:193], v[178:181], v[18:21]
v_mfma_f32_16x16x32_bf16 v[190:193], v[190:193], v[182:185], v[22:25]
s_nop 1
ds_read_b64_tr_b16 v[18:19], v204 offset:33664
ds_read_b64_tr_b16 v[20:21], v204 offset:33920
v_mfma_f32_16x16x32_bf16 v[34:37], v[194:197], v[178:181], v[34:37]
ds_read_b64_tr_b16 v[22:23], v201 offset:33664
ds_read_b64_tr_b16 v[24:25], v201 offset:33920
v_mfma_f32_16x16x32_bf16 v[38:41], v[194:197], v[182:185], v[38:41]
ds_read_b64_tr_b16 v[196:197], v201 offset:33984
ds_read_b64_tr_b16 v[194:195], v201 offset:33728
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[50:53], v[212:215], v[178:181], v[50:53]
v_mfma_f32_16x16x32_bf16 v[54:57], v[212:215], v[182:185], v[54:57]
s_cbranch_vccnz L
v_add_u32_e32 v212, 0x60, v244
v_cmp_lt_i32_e64 s[50:51], v212, 0
v_cmp_lt_i32_e64 s[52:53], v212, 1
v_cndmask_b32_e64 v166, v166, v245, s[50:51]
v_cndmask_b32_e64 v167, v167, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 2
v_cmp_lt_i32_e64 s[52:53], v212, 3
v_cndmask_b32_e64 v168, v168, v245, s[50:51]
v_cndmask_b32_e64 v169, v169, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 16
v_cmp_lt_i32_e64 s[52:53], v212, 17
v_cndmask_b32_e64 v158, v158, v245, s[50:51]
v_cndmask_b32_e64 v159, v159, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 18
v_cmp_lt_i32_e64 s[52:53], v212, 19
v_cndmask_b32_e64 v160, v160, v245, s[50:51]
v_cndmask_b32_e64 v161, v161, v245, s[52:53]
v_add_u32_e32 v212, 0x70, v244
v_cmp_lt_i32_e64 s[50:51], v212, 0
v_cmp_lt_i32_e64 s[52:53], v212, 1
v_cndmask_b32_e64 v162, v162, v245, s[50:51]
v_cndmask_b32_e64 v163, v163, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 2
v_cmp_lt_i32_e64 s[52:53], v212, 3
v_cndmask_b32_e64 v164, v164, v245, s[50:51]
v_cndmask_b32_e64 v165, v165, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 16
v_cmp_lt_i32_e64 s[52:53], v212, 17
v_cndmask_b32_e64 v154, v154, v245, s[50:51]
v_cndmask_b32_e64 v155, v155, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 18
v_cmp_lt_i32_e64 s[52:53], v212, 19
v_cndmask_b32_e64 v156, v156, v245, s[50:51]
v_cndmask_b32_e64 v157, v157, v245, s[52:53]
v_add_u32_e32 v212, 64, v244
v_cmp_lt_i32_e64 s[50:51], v212, 0
v_cmp_lt_i32_e64 s[52:53], v212, 1
v_cndmask_b32_e64 v146, v146, v245, s[50:51]
v_cndmask_b32_e64 v147, v147, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 2
v_cmp_lt_i32_e64 s[52:53], v212, 3
v_cndmask_b32_e64 v148, v148, v245, s[50:51]
v_cndmask_b32_e64 v149, v149, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 16
v_cmp_lt_i32_e64 s[52:53], v212, 17
v_cndmask_b32_e64 v138, v138, v245, s[50:51]
v_cndmask_b32_e64 v139, v139, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 18
v_cmp_lt_i32_e64 s[52:53], v212, 19
v_cndmask_b32_e64 v140, v140, v245, s[50:51]
v_cndmask_b32_e64 v141, v141, v245, s[52:53]
v_add_u32_e32 v212, 0x50, v244
v_cmp_lt_i32_e64 s[50:51], v212, 0
v_cmp_lt_i32_e64 s[52:53], v212, 1
v_cndmask_b32_e64 v142, v142, v245, s[50:51]
v_cndmask_b32_e64 v143, v143, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 2
v_cmp_lt_i32_e64 s[52:53], v212, 3
v_cndmask_b32_e64 v144, v144, v245, s[50:51]
v_cndmask_b32_e64 v145, v145, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 16
v_cmp_lt_i32_e64 s[52:53], v212, 17
v_cndmask_b32_e64 v150, v150, v245, s[50:51]
v_cndmask_b32_e64 v151, v151, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 18
v_cmp_lt_i32_e64 s[52:53], v212, 19
v_cndmask_b32_e64 v152, v152, v245, s[50:51]
v_cndmask_b32_e64 v153, v153, v245, s[52:53]
L:
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[212:215], v[18:21], v[178:181], v[10:13]
v_mfma_f32_16x16x32_bf16 v[216:219], v[18:21], v[182:185], v[14:17]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[220:223], v[22:25], v[178:181], v[26:29]
v_mfma_f32_16x16x32_bf16 v[224:227], v[22:25], v[182:185], v[30:33]
v_mfma_f32_16x16x32_bf16 v[42:45], v[250:253], v[178:181], v[42:45]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[178:181], v[194:197], v[178:181], v[58:61]
ds_read_b64_tr_b16 v[10:11], v204 offset:25472
ds_read_b64_tr_b16 v[12:13], v204 offset:25728
ds_read_b64_tr_b16 v[14:15], v201 offset:25472
ds_read_b64_tr_b16 v[16:17], v201 offset:25728
ds_read_b64_tr_b16 v[30:31], v204 offset:25536
ds_read_b64_tr_b16 v[32:33], v204 offset:25792
ds_read_b64_tr_b16 v[58:59], v201 offset:25536
ds_read_b64_tr_b16 v[60:61], v201 offset:25792
v_mfma_f32_16x16x32_bf16 v[46:49], v[250:253], v[182:185], v[46:49]
v_mfma_f32_16x16x32_bf16 v[62:65], v[194:197], v[182:185], v[62:65]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[18:21], v[10:13], v[170:173], v[2:5]
v_mfma_f32_16x16x32_bf16 v[2:5], v[10:13], v[174:177], v[6:9]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[22:25], v[14:17], v[170:173], v[186:189]
v_mfma_f32_16x16x32_bf16 v[6:9], v[14:17], v[174:177], v[190:193]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[26:29], v[30:33], v[170:173], v[34:37]
v_mfma_f32_16x16x32_bf16 v[10:13], v[30:33], v[174:177], v[38:41]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[30:33], v[58:61], v[170:173], v[50:53]
v_mfma_f32_16x16x32_bf16 v[14:17], v[58:61], v[174:177], v[54:57]
ds_read_b64_tr_b16 v[38:39], v204 offset:34176
ds_read_b64_tr_b16 v[40:41], v204 offset:34432
s_nop 0
ds_read_b64_tr_b16 v[54:55], v201 offset:34176
ds_read_b64_tr_b16 v[56:57], v201 offset:34432
ds_read_b64_tr_b16 v[58:59], v204 offset:34240
ds_read_b64_tr_b16 v[60:61], v204 offset:34496
ds_read_b64_tr_b16 v[182:183], v201 offset:34240
ds_read_b64_tr_b16 v[184:185], v201 offset:34496
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[34:37], v[38:41], v[170:173], v[212:215]
v_mfma_f32_16x16x32_bf16 v[50:53], v[38:41], v[174:177], v[216:219]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[38:41], v[54:57], v[170:173], v[220:223]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[42:45], v[58:61], v[170:173], v[42:45]
v_mfma_f32_16x16x32_bf16 v[58:61], v[58:61], v[174:177], v[46:49]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[46:49], v[182:185], v[170:173], v[178:181]
v_max3_f32 v170, v166, v167, v168
v_max_f32_e32 v173, v169, v162
v_max3_f32 v171, v143, v144, v145
v_mfma_f32_16x16x32_bf16 v[54:57], v[54:57], v[174:177], v[224:227]
v_max3_f32 v178, v148, v149, v142
v_max3_f32 v172, v138, v139, v140
v_max3_f32 v173, v170, v173, v163
v_mfma_f32_16x16x32_bf16 v[62:65], v[182:185], v[174:177], v[62:65]
v_max_f32_e32 v174, v164, v165
v_max3_f32 v175, v159, v160, v161
v_max3_f32 v176, v154, v155, v156
v_max3_f32 v177, v157, v146, v147
v_max3_f32 v173, v173, v174, v158
v_max3_f32 v174, v175, v176, v177
v_max3_f32 v175, v178, v171, v172
v_max3_f32 v179, v141, v150, v151
v_max_f32_e32 v180, v152, v153
v_max3_f32 v173, v173, v174, v175
v_max3_f32 v173, v173, v179, v180
v_cmp_lt_f32_e32 vcc, 0, v173
s_cbranch_vccz L
v_max_f32_e32 v66, v169, v158
v_max3_f32 v68, v147, v148, v149
v_max3_f32 v67, v160, v161, v146
v_max3_f32 v66, v170, v66, v159
v_max3_f32 v68, v68, v172, v141
v_max3_f32 v66, v66, v67, v68
v_max_f32_e32 v67, v162, v163
v_max3_f32 v68, v165, v154, v155
v_max3_f32 v70, v150, v151, v152
v_max3_f32 v69, v156, v157, v142
v_max3_f32 v67, v67, v164, v68
v_max3_f32 v68, v171, v70, v153
v_max3_f32 v67, v67, v69, v68
s_nop 1
v_permlane16_swap_b32_e32 v66, v67
v_max_f32_e32 v66, v66, v67
v_mov_b32_e32 v67, v66
s_nop 1
v_permlane32_swap_b32_e64 v66, v67 bound_ctrl:1
v_max3_f32 v66, v66, v67, s21
v_add_f32_e32 v66, 0x42c00000, v66
v_max_f32_e32 v66, 0, v66
v_exp_f32_e64 v68, -v66
v_mov_b32_e32 v70, v66
v_add_f32_e32 v1, v66, v1
s_nop 0
v_permlane16_swap_b32_e32 v66, v70
v_pk_add_f32 v[166:167], v[166:167], v[66:67] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[168:169], v[168:169], v[66:67] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[158:159], v[158:159], v[66:67] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[160:161], v[160:161], v[66:67] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[146:147], v[146:147], v[66:67] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[148:149], v[148:149], v[66:67] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[138:139], v[138:139], v[66:67] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[140:141], v[140:141], v[66:67] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_xor_b32_e32 v66, 0x80000000, v1
v_mov_b32_e32 v72, v68
v_pk_add_f32 v[162:163], v[162:163], v[70:71] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[164:165], v[164:165], v[70:71] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[154:155], v[154:155], v[70:71] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[156:157], v[156:157], v[70:71] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[142:143], v[142:143], v[70:71] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[144:145], v[144:145], v[70:71] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[150:151], v[150:151], v[70:71] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_pk_add_f32 v[152:153], v[152:153], v[70:71] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
v_mov_b32_e32 v70, v66
v_permlane16_swap_b32_e32 v68, v72
v_pk_mul_f32 v[132:133], v[72:73], v[132:133] op_sel_hi:[0,1]
v_pk_mul_f32 v[130:131], v[72:73], v[130:131] op_sel_hi:[0,1]
v_pk_mul_f32 v[136:137], v[68:69], v[136:137] op_sel_hi:[0,1]
v_pk_mul_f32 v[134:135], v[68:69], v[134:135] op_sel_hi:[0,1]
v_pk_mul_f32 v[52:53], v[72:73], v[52:53] op_sel_hi:[0,1]
v_pk_mul_f32 v[50:51], v[72:73], v[50:51] op_sel_hi:[0,1]
v_pk_mul_f32 v[36:37], v[68:69], v[36:37] op_sel_hi:[0,1]
v_pk_mul_f32 v[34:35], v[68:69], v[34:35] op_sel_hi:[0,1]
v_pk_mul_f32 v[4:5], v[72:73], v[4:5] op_sel_hi:[0,1]
v_pk_mul_f32 v[2:3], v[72:73], v[2:3] op_sel_hi:[0,1]
v_pk_mul_f32 v[20:21], v[68:69], v[20:21] op_sel_hi:[0,1]
v_pk_mul_f32 v[18:19], v[68:69], v[18:19] op_sel_hi:[0,1]
v_pk_mul_f32 v[56:57], v[72:73], v[56:57] op_sel_hi:[0,1]
v_pk_mul_f32 v[54:55], v[72:73], v[54:55] op_sel_hi:[0,1]
v_pk_mul_f32 v[40:41], v[68:69], v[40:41] op_sel_hi:[0,1]
v_pk_mul_f32 v[38:39], v[68:69], v[38:39] op_sel_hi:[0,1]
v_pk_mul_f32 v[8:9], v[72:73], v[8:9] op_sel_hi:[0,1]
v_pk_mul_f32 v[6:7], v[72:73], v[6:7] op_sel_hi:[0,1]
v_pk_mul_f32 v[24:25], v[68:69], v[24:25] op_sel_hi:[0,1]
v_pk_mul_f32 v[22:23], v[68:69], v[22:23] op_sel_hi:[0,1]
v_pk_mul_f32 v[60:61], v[72:73], v[60:61] op_sel_hi:[0,1]
v_pk_mul_f32 v[58:59], v[72:73], v[58:59] op_sel_hi:[0,1]
v_pk_mul_f32 v[44:45], v[68:69], v[44:45] op_sel_hi:[0,1]
v_pk_mul_f32 v[42:43], v[68:69], v[42:43] op_sel_hi:[0,1]
v_pk_mul_f32 v[12:13], v[72:73], v[12:13] op_sel_hi:[0,1]
v_pk_mul_f32 v[10:11], v[72:73], v[10:11] op_sel_hi:[0,1]
v_pk_mul_f32 v[28:29], v[68:69], v[28:29] op_sel_hi:[0,1]
v_pk_mul_f32 v[26:27], v[68:69], v[26:27] op_sel_hi:[0,1]
v_pk_mul_f32 v[64:65], v[72:73], v[64:65] op_sel_hi:[0,1]
v_pk_mul_f32 v[62:63], v[72:73], v[62:63] op_sel_hi:[0,1]
v_pk_mul_f32 v[48:49], v[68:69], v[48:49] op_sel_hi:[0,1]
v_pk_mul_f32 v[46:47], v[68:69], v[46:47] op_sel_hi:[0,1]
v_pk_mul_f32 v[16:17], v[72:73], v[16:17] op_sel_hi:[0,1]
v_pk_mul_f32 v[14:15], v[72:73], v[14:15] op_sel_hi:[0,1]
v_pk_mul_f32 v[32:33], v[68:69], v[32:33] op_sel_hi:[0,1]
v_pk_mul_f32 v[30:31], v[68:69], v[30:31] op_sel_hi:[0,1]
v_permlane16_swap_b32_e32 v66, v70
v_mov_b32_e32 v67, v66
v_mov_b32_e32 v68, v66
v_mov_b32_e32 v69, v66
v_mov_b32_e32 v71, v70
v_mov_b32_e32 v72, v70
v_mov_b32_e32 v73, v70
v_mov_b32_e32 v122, v66
v_mov_b32_e32 v123, v66
v_mov_b32_e32 v124, v66
v_mov_b32_e32 v125, v66
v_mov_b32_e32 v126, v70
v_mov_b32_e32 v127, v70
v_mov_b32_e32 v128, v70
v_mov_b32_e32 v129, v70
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
s_add_i32 s49, s28, 64
s_setprio 0
ds_read_b128 v[186:189], v205
ds_read_b128 v[190:193], v205 offset:512
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
v_mfma_f32_16x16x32_bf16 v[134:137], v[246:249], v[178:181], v[134:137]
v_mfma_f32_16x16x32_bf16 v[130:133], v[246:249], v[182:185], v[130:133]
v_mfma_f32_16x16x32_bf16 v[146:149], v[246:249], v[170:173], v[134:137]
v_mfma_f32_16x16x32_bf16 v[150:153], v[246:249], v[174:177], v[130:133]
ds_read_b128 v[162:165], v205 offset:256
ds_read_b128 v[166:169], v205 offset:768
ds_read_b128 v[194:197], v205 offset:32
ds_read_b128 v[216:219], v205 offset:544
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_bf16 v[130:133], v[186:189], v[74:77], v[66:69]
ds_read_b128 v[212:215], v205 offset:64
ds_read_b128 v[220:223], v205 offset:352
ds_read_b128 v[224:227], v208 offset:8608
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_bf16 v[134:137], v[190:193], v[74:77], v[66:69]
ds_read_b128 v[228:231], v208 offset:9120
v_mfma_f32_16x16x32_bf16 v[138:141], v[186:189], v[98:101], v[70:73]
ds_read_b128 v[186:189], v205 offset:96
v_mfma_f32_16x16x32_bf16 v[142:145], v[190:193], v[98:101], v[70:73]
ds_read_b128 v[190:193], v205 offset:800
s_waitcnt lgkmcnt(9)
v_mfma_f32_16x16x32_bf16 v[154:157], v[162:165], v[74:77], v[122:125]
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[158:161], v[166:169], v[74:77], v[122:125]
v_mfma_f32_16x16x32_bf16 v[162:165], v[162:165], v[98:101], v[126:129]
v_mfma_f32_16x16x32_bf16 v[166:169], v[166:169], v[98:101], v[126:129]
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x32_bf16 v[130:133], v[194:197], v[78:81], v[130:133]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[134:137], v[216:219], v[78:81], v[134:137]
v_mfma_f32_16x16x32_bf16 v[138:141], v[194:197], v[102:105], v[138:141]
ds_read_b128 v[194:197], v205 offset:288
v_mfma_f32_16x16x32_bf16 v[142:145], v[216:219], v[102:105], v[142:145]
ds_read_b128 v[216:219], v205 offset:320
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[154:157], v[194:197], v[78:81], v[154:157]
v_mfma_f32_16x16x32_bf16 v[158:161], v[190:193], v[78:81], v[158:161]
v_mfma_f32_16x16x32_bf16 v[162:165], v[194:197], v[102:105], v[162:165]
ds_read_b128 v[194:197], v205 offset:608
v_mfma_f32_16x16x32_bf16 v[166:169], v[190:193], v[102:105], v[166:169]
ds_read_b128 v[190:193], v205 offset:576
v_mfma_f32_16x16x32_bf16 v[130:133], v[212:215], v[82:85], v[130:133]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[134:137], v[190:193], v[82:85], v[134:137]
v_mfma_f32_16x16x32_bf16 v[138:141], v[212:215], v[106:109], v[138:141]
ds_read_b128 v[212:215], v205 offset:864
v_mfma_f32_16x16x32_bf16 v[142:145], v[190:193], v[106:109], v[142:145]
ds_read_b128 v[190:193], v205 offset:832
v_mfma_f32_16x16x32_bf16 v[130:133], v[186:189], v[86:89], v[130:133]
v_mfma_f32_16x16x32_bf16 v[138:141], v[186:189], v[110:113], v[138:141]
ds_read_b128 v[186:189], v208 offset:8320
s_waitcnt lgkmcnt(1)
v_mfma_f32_16x16x32_bf16 v[158:161], v[190:193], v[82:85], v[158:161]
v_mfma_f32_16x16x32_bf16 v[166:169], v[190:193], v[106:109], v[166:169]
ds_read_b128 v[190:193], v208 offset:8832
v_mfma_f32_16x16x32_bf16 v[154:157], v[216:219], v[82:85], v[154:157]
v_mfma_f32_16x16x32_bf16 v[162:165], v[216:219], v[106:109], v[162:165]
ds_read_b128 v[216:219], v208 offset:8352
v_mfma_f32_16x16x32_bf16 v[158:161], v[212:215], v[86:89], v[158:161]
v_mfma_f32_16x16x32_bf16 v[166:169], v[212:215], v[110:113], v[166:169]
ds_read_b128 v[212:215], v208 offset:9088
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_bf16 v[130:133], v[186:189], v[90:93], v[130:133]
v_mfma_f32_16x16x32_bf16 v[186:189], v[186:189], v[114:117], v[138:141]
s_nop 2
ds_read_b128 v[138:141], v208 offset:8576
v_mfma_f32_16x16x32_bf16 v[154:157], v[220:223], v[86:89], v[154:157]
v_mfma_f32_16x16x32_bf16 v[162:165], v[220:223], v[110:113], v[162:165]
ds_read_b128 v[220:223], v208 offset:8864
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[134:137], v[194:197], v[86:89], v[134:137]
s_barrier
v_mfma_f32_16x16x32_bf16 v[142:145], v[194:197], v[110:113], v[142:145]
v_mfma_f32_16x16x32_bf16 v[134:137], v[190:193], v[90:93], v[134:137]
v_mfma_f32_16x16x32_bf16 v[142:145], v[190:193], v[114:117], v[142:145]
v_mfma_f32_16x16x32_bf16 v[154:157], v[138:141], v[90:93], v[154:157]
v_mfma_f32_16x16x32_bf16 v[190:193], v[212:215], v[90:93], v[158:161]
v_mfma_f32_16x16x32_bf16 v[194:197], v[138:141], v[114:117], v[162:165]
v_mfma_f32_16x16x32_bf16 v[212:215], v[212:215], v[114:117], v[166:169]
v_mfma_f32_16x16x32_bf16 v[166:169], v[216:219], v[94:97], v[130:133]
v_mfma_f32_16x16x32_bf16 v[138:141], v[220:223], v[94:97], v[134:137]
v_mfma_f32_16x16x32_bf16 v[162:165], v[216:219], v[118:121], v[186:189]
v_mfma_f32_16x16x32_bf16 v[134:137], v[220:223], v[118:121], v[142:145]
v_mfma_f32_16x16x32_bf16 v[158:161], v[224:227], v[94:97], v[154:157]
v_mfma_f32_16x16x32_bf16 v[130:133], v[228:231], v[94:97], v[190:193]
v_mfma_f32_16x16x32_bf16 v[154:157], v[224:227], v[118:121], v[194:197]
v_mfma_f32_16x16x32_bf16 v[142:145], v[228:231], v[118:121], v[212:215]
s_nop 2
v_add_u32_e32 v212, 0x10740, v201
v_add_u32_e32 v186, 0x10700, v204
v_add_u32_e32 v188, 0x10800, v204
v_add_u32_e32 v190, 0x10700, v201
v_add_u32_e32 v192, 0x10800, v201
v_add_u32_e32 v194, 0x10740, v204
v_add_u32_e32 v196, 0x10840, v204
ds_read_b64_tr_b16 v[250:251], v212
v_add_u32_e32 v212, 0x10840, v201
ds_read_b64_tr_b16 v[186:187], v186
ds_read_b64_tr_b16 v[188:189], v188
ds_read_b64_tr_b16 v[190:191], v190
ds_read_b64_tr_b16 v[192:193], v192
ds_read_b64_tr_b16 v[194:195], v194
ds_read_b64_tr_b16 v[196:197], v196
ds_read_b64_tr_b16 v[252:253], v212
s_cmp_ge_i32 s20, s49
s_cbranch_scc1 L
v_add_u32_e32 v212, 32, v244
v_cmp_lt_i32_e64 s[50:51], v212, 0
v_cmp_lt_i32_e64 s[52:53], v212, 1
v_cndmask_b32_e64 v166, v166, v245, s[50:51]
v_cndmask_b32_e64 v167, v167, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 2
v_cmp_lt_i32_e64 s[52:53], v212, 3
v_cndmask_b32_e64 v168, v168, v245, s[50:51]
v_cndmask_b32_e64 v169, v169, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 16
v_cmp_lt_i32_e64 s[52:53], v212, 17
v_cndmask_b32_e64 v158, v158, v245, s[50:51]
v_cndmask_b32_e64 v159, v159, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 18
v_cmp_lt_i32_e64 s[52:53], v212, 19
v_cndmask_b32_e64 v160, v160, v245, s[50:51]
v_cndmask_b32_e64 v161, v161, v245, s[52:53]
v_add_u32_e32 v212, 48, v244
v_cmp_lt_i32_e64 s[50:51], v212, 0
v_cmp_lt_i32_e64 s[52:53], v212, 1
v_cndmask_b32_e64 v162, v162, v245, s[50:51]
v_cndmask_b32_e64 v163, v163, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 2
v_cmp_lt_i32_e64 s[52:53], v212, 3
v_cndmask_b32_e64 v164, v164, v245, s[50:51]
v_cndmask_b32_e64 v165, v165, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 16
v_cmp_lt_i32_e64 s[52:53], v212, 17
v_cndmask_b32_e64 v154, v154, v245, s[50:51]
v_cndmask_b32_e64 v155, v155, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 18
v_cmp_lt_i32_e64 s[52:53], v212, 19
v_cndmask_b32_e64 v156, v156, v245, s[50:51]
v_cndmask_b32_e64 v157, v157, v245, s[52:53]
v_add_u32_e32 v212, 16, v244
v_cmp_lt_i32_e64 s[50:51], v244, 0
v_cmp_lt_i32_e64 s[52:53], v244, 1
v_cndmask_b32_e64 v138, v138, v245, s[50:51]
v_cndmask_b32_e64 v139, v139, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v244, 2
v_cmp_lt_i32_e64 s[52:53], v244, 3
v_cndmask_b32_e64 v140, v140, v245, s[50:51]
v_cndmask_b32_e64 v141, v141, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v244, 16
v_cmp_lt_i32_e64 s[52:53], v244, 17
v_cndmask_b32_e64 v130, v130, v245, s[50:51]
v_cndmask_b32_e64 v131, v131, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v244, 18
v_cmp_lt_i32_e64 s[52:53], v244, 19
v_cndmask_b32_e64 v132, v132, v245, s[50:51]
v_cndmask_b32_e64 v133, v133, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 0
v_cmp_lt_i32_e64 s[52:53], v212, 1
v_cndmask_b32_e64 v134, v134, v245, s[50:51]
v_cndmask_b32_e64 v135, v135, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 2
v_cmp_lt_i32_e64 s[52:53], v212, 3
v_cndmask_b32_e64 v136, v136, v245, s[50:51]
v_cndmask_b32_e64 v137, v137, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 16
v_cmp_lt_i32_e64 s[52:53], v212, 17
v_cndmask_b32_e64 v142, v142, v245, s[50:51]
v_cndmask_b32_e64 v143, v143, v245, s[52:53]
s_nop 0
v_cmp_lt_i32_e64 s[50:51], v212, 18
v_cmp_lt_i32_e64 s[52:53], v212, 19
v_cndmask_b32_e64 v144, v144, v245, s[50:51]
v_cndmask_b32_e64 v145, v145, v245, s[52:53]
L:
s_mov_b32 m0, s45
s_nop 0
buffer_load_dwordx4 v209, s[12:15], s34 offen lds
s_mov_b32 m0, s46
s_nop 0
buffer_load_dwordx4 v210, s[12:15], s34 offen lds
s_mov_b32 m0, s47
s_nop 0
buffer_load_dwordx4 v211, s[12:15], s34 offen lds
s_mov_b32 m0, s40
s_nop 0
buffer_load_dwordx4 v206, s[16:19], s37 offen lds
s_mov_b32 m0, s39
s_nop 0
buffer_load_dwordx4 v207, s[16:19], s37 offen lds
s_setprio 1
s_waitcnt lgkmcnt(5)
v_mfma_f32_16x16x32_bf16 v[18:21], v[186:189], v[178:181], v[18:21]
v_mfma_f32_16x16x32_bf16 v[186:189], v[186:189], v[182:185], v[2:5]
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_bf16 v[22:25], v[190:193], v[178:181], v[22:25]
s_nop 0
v_add_u32_e32 v2, 0x12900, v204
v_add_u32_e32 v4, 0x12a00, v204
ds_read_b64_tr_b16 v[2:3], v2
v_mfma_f32_16x16x32_bf16 v[190:193], v[190:193], v[182:185], v[6:9]
ds_read_b64_tr_b16 v[4:5], v4
s_waitcnt lgkmcnt(3)
v_mfma_f32_16x16x32_bf16 v[26:29], v[194:197], v[178:181], v[26:29]
v_add_u32_e32 v6, 0x12900, v201
v_add_u32_e32 v8, 0x12a00, v201
ds_read_b64_tr_b16 v[6:7], v6
v_mfma_f32_16x16x32_bf16 v[10:13], v[194:197], v[182:185], v[10:13]
v_add_u32_e32 v194, 0x12940, v204
ds_read_b64_tr_b16 v[8:9], v8
ds_read_b64_tr_b16 v[224:225], v194
v_add_u32_e32 v194, 0x12a40, v204
ds_read_b64_tr_b16 v[226:227], v194
ds_read_b64_tr_b16 v[228:229], v0
ds_read_b64_tr_b16 v[230:231], v198
s_waitcnt lgkmcnt(8)
v_mfma_f32_16x16x32_bf16 v[30:33], v[250:253], v[178:181], v[30:33]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[194:197], v[2:5], v[178:181], v[34:37]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[216:219], v[6:9], v[178:181], v[38:41]
v_mfma_f32_16x16x32_bf16 v[220:223], v[6:9], v[182:185], v[54:57]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[42:45], v[224:227], v[178:181], v[42:45]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[178:181], v[228:231], v[178:181], v[46:49]
ds_read_b64_tr_b16 v[6:7], v199
ds_read_b64_tr_b16 v[8:9], v200
ds_read_b64_tr_b16 v[34:35], v254
ds_read_b64_tr_b16 v[36:37], v255
ds_read_b64_tr_b16 v[38:39], v232
ds_read_b64_tr_b16 v[40:41], v233
ds_read_b64_tr_b16 v[46:47], v234
ds_read_b64_tr_b16 v[48:49], v235
v_mfma_f32_16x16x32_bf16 v[14:17], v[250:253], v[182:185], v[14:17]
v_mfma_f32_16x16x32_bf16 v[212:215], v[2:5], v[182:185], v[50:53]
v_mfma_f32_16x16x32_bf16 v[58:61], v[224:227], v[182:185], v[58:61]
v_mfma_f32_16x16x32_bf16 v[62:65], v[228:231], v[182:185], v[62:65]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[50:53], v[46:49], v[170:173], v[30:33]
v_mfma_f32_16x16x32_bf16 v[54:57], v[46:49], v[174:177], v[14:17]
s_nop 2
ds_read_b64_tr_b16 v[14:15], v236
ds_read_b64_tr_b16 v[16:17], v237
ds_read_b64_tr_b16 v[30:31], v238
ds_read_b64_tr_b16 v[32:33], v239
ds_read_b64_tr_b16 v[46:47], v240
ds_read_b64_tr_b16 v[48:49], v241
ds_read_b64_tr_b16 v[182:183], v242
ds_read_b64_tr_b16 v[184:185], v243
v_mfma_f32_16x16x32_bf16 v[2:5], v[6:9], v[170:173], v[18:21]
v_mfma_f32_16x16x32_bf16 v[18:21], v[34:37], v[170:173], v[22:25]
v_mfma_f32_16x16x32_bf16 v[22:25], v[34:37], v[174:177], v[190:193]
v_mfma_f32_16x16x32_bf16 v[34:37], v[38:41], v[170:173], v[26:29]
v_mfma_f32_16x16x32_bf16 v[38:41], v[38:41], v[174:177], v[10:13]
s_waitcnt lgkmcnt(6)
v_mfma_f32_16x16x32_bf16 v[10:13], v[14:17], v[170:173], v[194:197]
s_waitcnt lgkmcnt(4)
v_mfma_f32_16x16x32_bf16 v[26:29], v[30:33], v[170:173], v[216:219]
s_waitcnt lgkmcnt(2)
v_mfma_f32_16x16x32_bf16 v[42:45], v[46:49], v[170:173], v[42:45]
v_mfma_f32_16x16x32_bf16 v[46:49], v[46:49], v[174:177], v[58:61]
s_waitcnt lgkmcnt(0)
v_mfma_f32_16x16x32_bf16 v[58:61], v[182:185], v[170:173], v[178:181]
v_max3_f32 v170, v166, v167, v168
v_max_f32_e32 v173, v169, v162
v_max3_f32 v171, v135, v136, v137
v_mfma_f32_16x16x32_bf16 v[6:9], v[6:9], v[174:177], v[186:189]
v_max3_f32 v178, v140, v141, v134
v_max3_f32 v172, v130, v131, v132
v_max3_f32 v173, v170, v173, v163
v_mfma_f32_16x16x32_bf16 v[14:17], v[14:17], v[174:177], v[212:215]
v_max3_f32 v179, v133, v142, v143
v_max_f32_e32 v180, v144, v145
v_mfma_f32_16x16x32_bf16 v[30:33], v[30:33], v[174:177], v[220:223]
v_mfma_f32_16x16x32_bf16 v[62:65], v[182:185], v[174:177], v[62:65]
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
