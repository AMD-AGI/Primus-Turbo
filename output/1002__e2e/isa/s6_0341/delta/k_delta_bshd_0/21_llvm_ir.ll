; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define amdgpu_kernel void @k_delta_bshd_0(ptr addrspace(1) %0, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %5, i32 %6, i32 %7, i32 %8) #0 !reqd_work_group_size !1 {
  %10 = call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x()
  %11 = sext i32 %10 to i64
  %12 = trunc i64 %11 to i32
  %13 = call i32 @llvm.amdgcn.workgroup.id.x()
  %14 = sext i32 %13 to i64
  %15 = trunc i64 %14 to i32
  %16 = mul i32 %8, 256
  %17 = sext i32 %16 to i64
  %18 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %0, i16 0, i64 %17, i32 159744)
  %19 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 %17, i32 159744)
  %20 = mul i32 %8, 4
  %21 = sext i32 %20 to i64
  %22 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 %21, i32 159744)
  %23 = mul i32 %15, 512
  %24 = add i32 %23, %12
  %25 = mul i32 %24, 16
  %26 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %18, i32 %25, i32 0, i32 0)
  %27 = bitcast i128 %26 to <8 x bfloat>
  %28 = add i32 %24, 256
  %29 = mul i32 %28, 16
  %30 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %18, i32 %29, i32 0, i32 0)
  %31 = bitcast i128 %30 to <8 x bfloat>
  %32 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %19, i32 %25, i32 0, i32 0)
  %33 = bitcast i128 %32 to <8 x bfloat>
  %34 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %19, i32 %29, i32 0, i32 0)
  %35 = bitcast i128 %34 to <8 x bfloat>
  %36 = srem i32 %12, 16
  %37 = sdiv i32 %12, 16
  %38 = mul i32 %37, 16
  %39 = icmp ne i32 %12, %38
  %40 = icmp slt i32 %12, 0
  %41 = icmp ne i1 %40, false
  %42 = and i1 %39, %41
  %43 = add i32 %37, -1
  %44 = select i1 %42, i32 %43, i32 %37
  %45 = extractelement <8 x bfloat> %27, i64 0
  %46 = fpext bfloat %45 to float
  %47 = extractelement <8 x bfloat> %33, i64 0
  %48 = fpext bfloat %47 to float
  %49 = fmul float %46, %48
  %50 = fadd float %49, 0.000000e+00
  %51 = extractelement <8 x bfloat> %27, i64 1
  %52 = fpext bfloat %51 to float
  %53 = extractelement <8 x bfloat> %33, i64 1
  %54 = fpext bfloat %53 to float
  %55 = fmul float %52, %54
  %56 = fadd float %55, 0.000000e+00
  %57 = extractelement <8 x bfloat> %27, i64 2
  %58 = fpext bfloat %57 to float
  %59 = extractelement <8 x bfloat> %33, i64 2
  %60 = fpext bfloat %59 to float
  %61 = fmul float %58, %60
  %62 = fadd float %50, %61
  %63 = extractelement <8 x bfloat> %27, i64 3
  %64 = fpext bfloat %63 to float
  %65 = extractelement <8 x bfloat> %33, i64 3
  %66 = fpext bfloat %65 to float
  %67 = fmul float %64, %66
  %68 = fadd float %56, %67
  %69 = extractelement <8 x bfloat> %27, i64 4
  %70 = fpext bfloat %69 to float
  %71 = extractelement <8 x bfloat> %33, i64 4
  %72 = fpext bfloat %71 to float
  %73 = fmul float %70, %72
  %74 = fadd float %62, %73
  %75 = extractelement <8 x bfloat> %27, i64 5
  %76 = fpext bfloat %75 to float
  %77 = extractelement <8 x bfloat> %33, i64 5
  %78 = fpext bfloat %77 to float
  %79 = fmul float %76, %78
  %80 = fadd float %68, %79
  %81 = extractelement <8 x bfloat> %27, i64 6
  %82 = fpext bfloat %81 to float
  %83 = extractelement <8 x bfloat> %33, i64 6
  %84 = fpext bfloat %83 to float
  %85 = fmul float %82, %84
  %86 = fadd float %74, %85
  %87 = extractelement <8 x bfloat> %27, i64 7
  %88 = fpext bfloat %87 to float
  %89 = extractelement <8 x bfloat> %33, i64 7
  %90 = fpext bfloat %89 to float
  %91 = fmul float %88, %90
  %92 = fadd float %80, %91
  %93 = fadd float %86, %92
  %94 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %95 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %94)
  %96 = add i32 %95, 32
  %97 = and i32 %96, -32
  %98 = xor i32 %95, 1
  %99 = icmp slt i32 %98, %97
  %100 = select i1 %99, i32 %98, i32 %95
  %101 = shl i32 %100, 2
  %102 = bitcast float %93 to i32
  %103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %101, i32 %102)
  %104 = bitcast i32 %103 to float
  %105 = fadd float %93, %104
  %106 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %107 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %106)
  %108 = add i32 %107, 32
  %109 = and i32 %108, -32
  %110 = xor i32 %107, 2
  %111 = icmp slt i32 %110, %109
  %112 = select i1 %111, i32 %110, i32 %107
  %113 = shl i32 %112, 2
  %114 = bitcast float %105 to i32
  %115 = call i32 @llvm.amdgcn.ds.bpermute(i32 %113, i32 %114)
  %116 = bitcast i32 %115 to float
  %117 = fadd float %105, %116
  %118 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %119 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %118)
  %120 = add i32 %119, 32
  %121 = and i32 %120, -32
  %122 = xor i32 %119, 4
  %123 = icmp slt i32 %122, %121
  %124 = select i1 %123, i32 %122, i32 %119
  %125 = shl i32 %124, 2
  %126 = bitcast float %117 to i32
  %127 = call i32 @llvm.amdgcn.ds.bpermute(i32 %125, i32 %126)
  %128 = bitcast i32 %127 to float
  %129 = fadd float %117, %128
  %130 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %131 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %130)
  %132 = add i32 %131, 32
  %133 = and i32 %132, -32
  %134 = xor i32 %131, 8
  %135 = icmp slt i32 %134, %133
  %136 = select i1 %135, i32 %134, i32 %131
  %137 = shl i32 %136, 2
  %138 = bitcast float %129 to i32
  %139 = call i32 @llvm.amdgcn.ds.bpermute(i32 %137, i32 %138)
  %140 = bitcast i32 %139 to float
  %141 = fadd float %129, %140
  %142 = mul i32 %15, 32
  %143 = add i32 %142, %44
  %144 = icmp eq i32 %36, 0
  %145 = icmp slt i32 %143, %8
  %146 = and i1 %144, %145
  %147 = mul i32 %6, %7
  %148 = sdiv i32 %143, %147
  %149 = mul i32 %148, %147
  %150 = icmp ne i32 %143, %149
  %151 = icmp slt i32 %143, 0
  %152 = icmp slt i32 %147, 0
  %153 = icmp ne i1 %151, %152
  %154 = and i1 %150, %153
  %155 = add i32 %148, -1
  %156 = select i1 %154, i32 %155, i32 %148
  %157 = mul i32 %156, %147
  %158 = sub i32 %143, %157
  %159 = sdiv i32 %158, %7
  %160 = mul i32 %159, %7
  %161 = icmp ne i32 %158, %160
  %162 = icmp slt i32 %158, 0
  %163 = icmp slt i32 %7, 0
  %164 = icmp ne i1 %162, %163
  %165 = and i1 %161, %164
  %166 = add i32 %159, -1
  %167 = select i1 %165, i32 %166, i32 %159
  %168 = mul i32 %167, %7
  %169 = sub i32 %158, %168
  %170 = mul i32 %156, %7
  %171 = add i32 %170, %169
  %172 = mul i32 %171, %6
  %173 = add i32 %172, %167
  %174 = select i1 %146, i32 %173, i32 %8
  %175 = mul i32 %174, 4
  %176 = bitcast float %141 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %176, ptr addrspace(8) %22, i32 %175, i32 0, i32 0)
  %177 = extractelement <8 x bfloat> %31, i64 0
  %178 = fpext bfloat %177 to float
  %179 = extractelement <8 x bfloat> %35, i64 0
  %180 = fpext bfloat %179 to float
  %181 = fmul float %178, %180
  %182 = fadd float %181, 0.000000e+00
  %183 = extractelement <8 x bfloat> %31, i64 1
  %184 = fpext bfloat %183 to float
  %185 = extractelement <8 x bfloat> %35, i64 1
  %186 = fpext bfloat %185 to float
  %187 = fmul float %184, %186
  %188 = fadd float %187, 0.000000e+00
  %189 = extractelement <8 x bfloat> %31, i64 2
  %190 = fpext bfloat %189 to float
  %191 = extractelement <8 x bfloat> %35, i64 2
  %192 = fpext bfloat %191 to float
  %193 = fmul float %190, %192
  %194 = fadd float %182, %193
  %195 = extractelement <8 x bfloat> %31, i64 3
  %196 = fpext bfloat %195 to float
  %197 = extractelement <8 x bfloat> %35, i64 3
  %198 = fpext bfloat %197 to float
  %199 = fmul float %196, %198
  %200 = fadd float %188, %199
  %201 = extractelement <8 x bfloat> %31, i64 4
  %202 = fpext bfloat %201 to float
  %203 = extractelement <8 x bfloat> %35, i64 4
  %204 = fpext bfloat %203 to float
  %205 = fmul float %202, %204
  %206 = fadd float %194, %205
  %207 = extractelement <8 x bfloat> %31, i64 5
  %208 = fpext bfloat %207 to float
  %209 = extractelement <8 x bfloat> %35, i64 5
  %210 = fpext bfloat %209 to float
  %211 = fmul float %208, %210
  %212 = fadd float %200, %211
  %213 = extractelement <8 x bfloat> %31, i64 6
  %214 = fpext bfloat %213 to float
  %215 = extractelement <8 x bfloat> %35, i64 6
  %216 = fpext bfloat %215 to float
  %217 = fmul float %214, %216
  %218 = fadd float %206, %217
  %219 = extractelement <8 x bfloat> %31, i64 7
  %220 = fpext bfloat %219 to float
  %221 = extractelement <8 x bfloat> %35, i64 7
  %222 = fpext bfloat %221 to float
  %223 = fmul float %220, %222
  %224 = fadd float %212, %223
  %225 = fadd float %218, %224
  %226 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %227 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %226)
  %228 = add i32 %227, 32
  %229 = and i32 %228, -32
  %230 = xor i32 %227, 1
  %231 = icmp slt i32 %230, %229
  %232 = select i1 %231, i32 %230, i32 %227
  %233 = shl i32 %232, 2
  %234 = bitcast float %225 to i32
  %235 = call i32 @llvm.amdgcn.ds.bpermute(i32 %233, i32 %234)
  %236 = bitcast i32 %235 to float
  %237 = fadd float %225, %236
  %238 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %239 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %238)
  %240 = add i32 %239, 32
  %241 = and i32 %240, -32
  %242 = xor i32 %239, 2
  %243 = icmp slt i32 %242, %241
  %244 = select i1 %243, i32 %242, i32 %239
  %245 = shl i32 %244, 2
  %246 = bitcast float %237 to i32
  %247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %245, i32 %246)
  %248 = bitcast i32 %247 to float
  %249 = fadd float %237, %248
  %250 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %251 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %250)
  %252 = add i32 %251, 32
  %253 = and i32 %252, -32
  %254 = xor i32 %251, 4
  %255 = icmp slt i32 %254, %253
  %256 = select i1 %255, i32 %254, i32 %251
  %257 = shl i32 %256, 2
  %258 = bitcast float %249 to i32
  %259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %257, i32 %258)
  %260 = bitcast i32 %259 to float
  %261 = fadd float %249, %260
  %262 = call noundef range(i32 0, 32) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %263 = call noundef range(i32 0, 64) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %262)
  %264 = add i32 %263, 32
  %265 = and i32 %264, -32
  %266 = xor i32 %263, 8
  %267 = icmp slt i32 %266, %265
  %268 = select i1 %267, i32 %266, i32 %263
  %269 = shl i32 %268, 2
  %270 = bitcast float %261 to i32
  %271 = call i32 @llvm.amdgcn.ds.bpermute(i32 %269, i32 %270)
  %272 = bitcast i32 %271 to float
  %273 = fadd float %261, %272
  %274 = add i32 %142, 16
  %275 = add i32 %274, %44
  %276 = icmp slt i32 %275, %8
  %277 = and i1 %144, %276
  %278 = sdiv i32 %275, %147
  %279 = mul i32 %278, %147
  %280 = icmp ne i32 %275, %279
  %281 = icmp slt i32 %275, 0
  %282 = icmp slt i32 %147, 0
  %283 = icmp ne i1 %281, %282
  %284 = and i1 %280, %283
  %285 = add i32 %278, -1
  %286 = select i1 %284, i32 %285, i32 %278
  %287 = mul i32 %286, %147
  %288 = sub i32 %275, %287
  %289 = sdiv i32 %288, %7
  %290 = mul i32 %289, %7
  %291 = icmp ne i32 %288, %290
  %292 = icmp slt i32 %288, 0
  %293 = icmp slt i32 %7, 0
  %294 = icmp ne i1 %292, %293
  %295 = and i1 %291, %294
  %296 = add i32 %289, -1
  %297 = select i1 %295, i32 %296, i32 %289
  %298 = mul i32 %297, %7
  %299 = sub i32 %288, %298
  %300 = mul i32 %286, %7
  %301 = add i32 %300, %299
  %302 = mul i32 %301, %6
  %303 = add i32 %302, %297
  %304 = select i1 %277, i32 %303, i32 %8
  %305 = mul i32 %304, 4
  %306 = bitcast float %273 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %306, ptr addrspace(8) %22, i32 %305, i32 0, i32 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone, i16, i64, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #6

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 256, i32 1, i32 1}
