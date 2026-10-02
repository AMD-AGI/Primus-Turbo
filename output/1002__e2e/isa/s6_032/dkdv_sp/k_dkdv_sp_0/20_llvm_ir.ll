; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__shared_alloc_0 = external dso_local addrspace(3) global [70656 x i8], align 16

define amdgpu_kernel void @k_dkdv_sp_0(ptr addrspace(1) %0, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %5, ptr addrspace(1) %6, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %7, ptr addrspace(1) %8, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %9, ptr addrspace(1) %10, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %11, ptr addrspace(1) %12, <{ <{ i32, i32, i32, i32, i32 }>, <{ i64, i64, i64, i64 }> }> %13, ptr addrspace(1) %14, <{ <{ i32, i32, i32, i32, i32 }>, <{ i64, i64, i64, i64 }> }> %15, float %16, i32 %17, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 %26) #0 !reqd_work_group_size !1 {
  %28 = call range(i32 0, 32) i32 @llvm.amdgcn.workitem.id.x()
  %29 = sext i32 %28 to i64
  %30 = trunc i64 %29 to i32
  %31 = call i32 @llvm.amdgcn.workgroup.id.x()
  %32 = sext i32 %31 to i64
  %33 = trunc i64 %32 to i32
  %34 = sdiv i32 %33, %26
  %35 = mul i32 %34, %26
  %36 = icmp ne i32 %33, %35
  %37 = icmp slt i32 %33, 0
  %38 = icmp slt i32 %26, 0
  %39 = icmp ne i1 %37, %38
  %40 = and i1 %36, %39
  %41 = add i32 %34, -1
  %42 = select i1 %40, i32 %41, i32 %34
  %43 = mul i32 %42, %26
  %44 = sub i32 %33, %43
  %45 = call i32 @llvm.amdgcn.workgroup.id.y()
  %46 = sext i32 %45 to i64
  %47 = trunc i64 %46 to i32
  %48 = call i32 @llvm.amdgcn.workgroup.id.z()
  %49 = sext i32 %48 to i64
  %50 = trunc i64 %49 to i32
  %51 = srem i32 %30, 16
  %52 = sdiv i32 %30, 16
  %53 = mul i32 %52, 16
  %54 = icmp ne i32 %30, %53
  %55 = icmp slt i32 %30, 0
  %56 = icmp ne i1 %55, false
  %57 = and i1 %54, %56
  %58 = add i32 %52, -1
  %59 = select i1 %57, i32 %58, i32 %52
  %60 = mul i32 %47, 32
  %61 = mul i32 %25, %18
  %62 = mul i32 %61, %20
  %63 = mul i32 %62, 256
  %64 = mul i32 %25, %19
  %65 = mul i32 %64, %17
  %66 = mul i32 %65, 4
  %67 = sext i32 %63 to i64
  %68 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 %67, i32 159744)
  %69 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 %67, i32 159744)
  %70 = sext i32 %66 to i64
  %71 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %8, i16 0, i64 %70, i32 159744)
  %72 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %10, i16 0, i64 %70, i32 159744)
  %73 = mul i32 %26, %25
  %74 = mul i32 %73, %18
  %75 = mul i32 %74, %20
  %76 = mul i32 %75, 512
  %77 = sext i32 %76 to i64
  %78 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %12, i16 0, i64 %77, i32 159744)
  %79 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %14, i16 0, i64 %77, i32 159744)
  %80 = mul i32 %20, 16
  %81 = mul i32 %50, %18
  %82 = mul i32 %81, %80
  %83 = mul i32 %42, 16
  %84 = add i32 %82, %83
  %85 = mul i32 %51, 4
  %86 = mul i32 %19, 128
  %87 = add i32 %60, %51
  %88 = mul i32 %87, %80
  %89 = add i32 %84, %88
  %90 = add i32 %89, %59
  %91 = mul i32 %90, 16
  %92 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %91, i32 0, i32 0)
  %93 = bitcast i128 %92 to <8 x bfloat>
  %94 = add i32 %90, 2
  %95 = mul i32 %94, 16
  %96 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %95, i32 0, i32 0)
  %97 = bitcast i128 %96 to <8 x bfloat>
  %98 = shufflevector <8 x bfloat> %93, <8 x bfloat> %97, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %99 = add i32 %90, 4
  %100 = mul i32 %99, 16
  %101 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %100, i32 0, i32 0)
  %102 = bitcast i128 %101 to <8 x bfloat>
  %103 = add i32 %90, 6
  %104 = mul i32 %103, 16
  %105 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %104, i32 0, i32 0)
  %106 = bitcast i128 %105 to <8 x bfloat>
  %107 = shufflevector <8 x bfloat> %102, <8 x bfloat> %106, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %108 = add i32 %90, 8
  %109 = mul i32 %108, 16
  %110 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %109, i32 0, i32 0)
  %111 = bitcast i128 %110 to <8 x bfloat>
  %112 = add i32 %90, 10
  %113 = mul i32 %112, 16
  %114 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %113, i32 0, i32 0)
  %115 = bitcast i128 %114 to <8 x bfloat>
  %116 = shufflevector <8 x bfloat> %111, <8 x bfloat> %115, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %117 = add i32 %90, 12
  %118 = mul i32 %117, 16
  %119 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %118, i32 0, i32 0)
  %120 = bitcast i128 %119 to <8 x bfloat>
  %121 = add i32 %90, 14
  %122 = mul i32 %121, 16
  %123 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %122, i32 0, i32 0)
  %124 = bitcast i128 %123 to <8 x bfloat>
  %125 = shufflevector <8 x bfloat> %120, <8 x bfloat> %124, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %126 = add i32 %60, 16
  %127 = add i32 %126, %51
  %128 = mul i32 %127, %80
  %129 = add i32 %84, %128
  %130 = add i32 %129, %59
  %131 = mul i32 %130, 16
  %132 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %131, i32 0, i32 0)
  %133 = bitcast i128 %132 to <8 x bfloat>
  %134 = add i32 %130, 2
  %135 = mul i32 %134, 16
  %136 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %135, i32 0, i32 0)
  %137 = bitcast i128 %136 to <8 x bfloat>
  %138 = shufflevector <8 x bfloat> %133, <8 x bfloat> %137, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %139 = add i32 %130, 4
  %140 = mul i32 %139, 16
  %141 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %140, i32 0, i32 0)
  %142 = bitcast i128 %141 to <8 x bfloat>
  %143 = add i32 %130, 6
  %144 = mul i32 %143, 16
  %145 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %144, i32 0, i32 0)
  %146 = bitcast i128 %145 to <8 x bfloat>
  %147 = shufflevector <8 x bfloat> %142, <8 x bfloat> %146, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %148 = add i32 %130, 8
  %149 = mul i32 %148, 16
  %150 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %149, i32 0, i32 0)
  %151 = bitcast i128 %150 to <8 x bfloat>
  %152 = add i32 %130, 10
  %153 = mul i32 %152, 16
  %154 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %153, i32 0, i32 0)
  %155 = bitcast i128 %154 to <8 x bfloat>
  %156 = shufflevector <8 x bfloat> %151, <8 x bfloat> %155, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %157 = add i32 %130, 12
  %158 = mul i32 %157, 16
  %159 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %158, i32 0, i32 0)
  %160 = bitcast i128 %159 to <8 x bfloat>
  %161 = add i32 %130, 14
  %162 = mul i32 %161, 16
  %163 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %68, i32 %162, i32 0, i32 0)
  %164 = bitcast i128 %163 to <8 x bfloat>
  %165 = shufflevector <8 x bfloat> %160, <8 x bfloat> %164, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %166 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %91, i32 0, i32 0)
  %167 = bitcast i128 %166 to <8 x bfloat>
  %168 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %95, i32 0, i32 0)
  %169 = bitcast i128 %168 to <8 x bfloat>
  %170 = shufflevector <8 x bfloat> %167, <8 x bfloat> %169, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %171 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %100, i32 0, i32 0)
  %172 = bitcast i128 %171 to <8 x bfloat>
  %173 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %104, i32 0, i32 0)
  %174 = bitcast i128 %173 to <8 x bfloat>
  %175 = shufflevector <8 x bfloat> %172, <8 x bfloat> %174, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %176 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %109, i32 0, i32 0)
  %177 = bitcast i128 %176 to <8 x bfloat>
  %178 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %113, i32 0, i32 0)
  %179 = bitcast i128 %178 to <8 x bfloat>
  %180 = shufflevector <8 x bfloat> %177, <8 x bfloat> %179, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %181 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %118, i32 0, i32 0)
  %182 = bitcast i128 %181 to <8 x bfloat>
  %183 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %122, i32 0, i32 0)
  %184 = bitcast i128 %183 to <8 x bfloat>
  %185 = shufflevector <8 x bfloat> %182, <8 x bfloat> %184, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %186 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %131, i32 0, i32 0)
  %187 = bitcast i128 %186 to <8 x bfloat>
  %188 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %135, i32 0, i32 0)
  %189 = bitcast i128 %188 to <8 x bfloat>
  %190 = shufflevector <8 x bfloat> %187, <8 x bfloat> %189, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %191 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %140, i32 0, i32 0)
  %192 = bitcast i128 %191 to <8 x bfloat>
  %193 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %144, i32 0, i32 0)
  %194 = bitcast i128 %193 to <8 x bfloat>
  %195 = shufflevector <8 x bfloat> %192, <8 x bfloat> %194, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %196 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %149, i32 0, i32 0)
  %197 = bitcast i128 %196 to <8 x bfloat>
  %198 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %153, i32 0, i32 0)
  %199 = bitcast i128 %198 to <8 x bfloat>
  %200 = shufflevector <8 x bfloat> %197, <8 x bfloat> %199, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %201 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %158, i32 0, i32 0)
  %202 = bitcast i128 %201 to <8 x bfloat>
  %203 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %69, i32 %162, i32 0, i32 0)
  %204 = bitcast i128 %203 to <8 x bfloat>
  %205 = shufflevector <8 x bfloat> %202, <8 x bfloat> %204, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %206 = mul i32 %59, 8
  %207 = srem i32 %30, 8
  %208 = add i32 %206, %207
  %209 = sdiv i32 %30, 8
  %210 = mul i32 %209, 8
  %211 = icmp ne i32 %30, %210
  %212 = icmp slt i32 %30, 0
  %213 = icmp ne i1 %212, false
  %214 = and i1 %211, %213
  %215 = add i32 %209, -1
  %216 = select i1 %214, i32 %215, i32 %209
  %217 = srem i32 %216, 2
  %218 = mul i32 %208, 272
  %219 = mul i32 %217, 16
  %220 = add i32 %218, %219
  %221 = mul i32 %51, 272
  %222 = mul i32 %59, 16
  %223 = add i32 %221, %222
  %224 = sdiv i32 %22, 2
  %225 = mul i32 %224, 2
  %226 = icmp ne i32 %22, %225
  %227 = icmp slt i32 %22, 0
  %228 = icmp ne i1 %227, false
  %229 = and i1 %226, %228
  %230 = add i32 %224, -1
  %231 = select i1 %229, i32 %230, i32 %224
  %232 = sub i32 %60, %23
  %233 = call i32 @llvm.smax.i32(i32 %232, i32 0)
  %234 = sdiv i32 %233, 32
  %235 = mul i32 %234, 32
  %236 = icmp ne i32 %233, %235
  %237 = icmp slt i32 %233, 0
  %238 = icmp ne i1 %237, false
  %239 = and i1 %236, %238
  %240 = add i32 %234, -1
  %241 = select i1 %239, i32 %240, i32 %234
  %242 = icmp ne i32 %24, 0
  %243 = select i1 %242, i32 %241, i32 0
  %244 = sub i32 %231, %243
  %245 = add i32 %60, 31
  %246 = sub i32 %245, %23
  %247 = icmp slt i32 %246, 0
  %248 = add i32 %246, 31
  %249 = sdiv i32 %248, 32
  %250 = mul i32 %249, 32
  %251 = icmp ne i32 %248, %250
  %252 = icmp slt i32 %248, 0
  %253 = icmp ne i1 %252, false
  %254 = and i1 %251, %253
  %255 = add i32 %249, -1
  %256 = select i1 %254, i32 %255, i32 %249
  %257 = select i1 %247, i32 0, i32 %256
  %258 = call i32 @llvm.smin.i32(i32 %257, i32 %231)
  %259 = sub i32 %258, %243
  %260 = call i32 @llvm.smax.i32(i32 %259, i32 0)
  %261 = call i32 @llvm.smin.i32(i32 %260, i32 %244)
  %262 = select i1 %242, i32 %261, i32 0
  %263 = sub i32 %244, %262
  %264 = call i32 @llvm.smax.i32(i32 %263, i32 0)
  %265 = add i32 %264, %26
  %266 = sub i32 %265, 1
  %267 = sdiv i32 %266, %26
  %268 = mul i32 %267, %26
  %269 = icmp ne i32 %266, %268
  %270 = icmp slt i32 %266, 0
  %271 = icmp slt i32 %26, 0
  %272 = icmp ne i1 %270, %271
  %273 = and i1 %269, %272
  %274 = add i32 %267, -1
  %275 = select i1 %273, i32 %274, i32 %267
  %276 = mul i32 %44, %275
  %277 = sub i32 %264, %276
  %278 = call i32 @llvm.smax.i32(i32 %277, i32 0)
  %279 = call i32 @llvm.smin.i32(i32 %278, i32 %275)
  %280 = icmp ne i32 %44, 0
  %281 = select i1 %280, i32 0, i32 %262
  %282 = add i32 %243, %262
  %283 = add i32 %282, %276
  %284 = mul i32 %21, %281
  %285 = sext i32 %284 to i64
  br label %286

286:                                              ; preds = %323, %27
  %287 = phi i64 [ %1271, %323 ], [ 0, %27 ]
  %288 = phi <8 x float> [ %1239, %323 ], [ zeroinitializer, %27 ]
  %289 = phi <8 x float> [ %1240, %323 ], [ zeroinitializer, %27 ]
  %290 = phi <8 x float> [ %1241, %323 ], [ zeroinitializer, %27 ]
  %291 = phi <8 x float> [ %1242, %323 ], [ zeroinitializer, %27 ]
  %292 = phi <8 x float> [ %1243, %323 ], [ zeroinitializer, %27 ]
  %293 = phi <8 x float> [ %1244, %323 ], [ zeroinitializer, %27 ]
  %294 = phi <8 x float> [ %1245, %323 ], [ zeroinitializer, %27 ]
  %295 = phi <8 x float> [ %1246, %323 ], [ zeroinitializer, %27 ]
  %296 = phi <8 x float> [ %1255, %323 ], [ zeroinitializer, %27 ]
  %297 = phi <8 x float> [ %1256, %323 ], [ zeroinitializer, %27 ]
  %298 = phi <8 x float> [ %1257, %323 ], [ zeroinitializer, %27 ]
  %299 = phi <8 x float> [ %1258, %323 ], [ zeroinitializer, %27 ]
  %300 = phi <8 x float> [ %1259, %323 ], [ zeroinitializer, %27 ]
  %301 = phi <8 x float> [ %1260, %323 ], [ zeroinitializer, %27 ]
  %302 = phi <8 x float> [ %1261, %323 ], [ zeroinitializer, %27 ]
  %303 = phi <8 x float> [ %1262, %323 ], [ zeroinitializer, %27 ]
  %304 = phi <8 x float> [ %1247, %323 ], [ zeroinitializer, %27 ]
  %305 = phi <8 x float> [ %1248, %323 ], [ zeroinitializer, %27 ]
  %306 = phi <8 x float> [ %1249, %323 ], [ zeroinitializer, %27 ]
  %307 = phi <8 x float> [ %1250, %323 ], [ zeroinitializer, %27 ]
  %308 = phi <8 x float> [ %1251, %323 ], [ zeroinitializer, %27 ]
  %309 = phi <8 x float> [ %1252, %323 ], [ zeroinitializer, %27 ]
  %310 = phi <8 x float> [ %1253, %323 ], [ zeroinitializer, %27 ]
  %311 = phi <8 x float> [ %1254, %323 ], [ zeroinitializer, %27 ]
  %312 = phi <8 x float> [ %1263, %323 ], [ zeroinitializer, %27 ]
  %313 = phi <8 x float> [ %1264, %323 ], [ zeroinitializer, %27 ]
  %314 = phi <8 x float> [ %1265, %323 ], [ zeroinitializer, %27 ]
  %315 = phi <8 x float> [ %1266, %323 ], [ zeroinitializer, %27 ]
  %316 = phi <8 x float> [ %1267, %323 ], [ zeroinitializer, %27 ]
  %317 = phi <8 x float> [ %1268, %323 ], [ zeroinitializer, %27 ]
  %318 = phi <8 x float> [ %1269, %323 ], [ zeroinitializer, %27 ]
  %319 = phi <8 x float> [ %1270, %323 ], [ zeroinitializer, %27 ]
  %320 = phi i32 [ %327, %323 ], [ 0, %27 ]
  %321 = phi i32 [ %328, %323 ], [ 0, %27 ]
  %322 = icmp slt i64 %287, %285
  br i1 %322, label %323, label %1272

323:                                              ; preds = %286
  %324 = add i32 %321, 1
  %325 = icmp slt i32 %324, %21
  %326 = add i32 %320, 1
  %327 = select i1 %325, i32 %320, i32 %326
  %328 = select i1 %325, i32 %324, i32 0
  %329 = add i32 %243, %320
  %330 = mul i32 %329, 32
  %331 = mul i32 %42, %21
  %332 = add i32 %331, %321
  %333 = mul i32 %50, %19
  %334 = add i32 %333, %332
  %335 = mul i32 %334, %17
  %336 = add i32 %330, %51
  %337 = add i32 %335, %336
  %338 = mul i32 %337, 4
  %339 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %71, i32 %338, i32 0, i32 0)
  %340 = bitcast i32 %339 to float
  %341 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %72, i32 %338, i32 0, i32 0)
  %342 = bitcast i32 %341 to float
  %343 = add i32 %330, 16
  %344 = add i32 %343, %51
  %345 = add i32 %335, %344
  %346 = mul i32 %345, 4
  %347 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %71, i32 %346, i32 0, i32 0)
  %348 = bitcast i32 %347 to float
  %349 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %72, i32 %346, i32 0, i32 0)
  %350 = bitcast i32 %349 to float
  %351 = mul i32 %50, %17
  %352 = add i32 %351, %330
  %353 = sext i32 %352 to i64
  %354 = sext i32 %19 to i64
  %355 = mul i64 %353, %354
  %356 = sext i32 %332 to i64
  %357 = add i64 %355, %356
  %358 = mul i64 %357, 128
  %359 = sub i32 %17, %330
  %360 = getelementptr bfloat, ptr addrspace(1) %6, i64 %358
  %361 = sext i32 %86 to i64
  %362 = icmp eq i64 %361, -2147483648
  %363 = select i1 %362, i64 128, i64 %361
  %364 = ptrtoint ptr addrspace(1) %360 to i64
  %365 = trunc i64 %364 to i32
  %366 = lshr i64 %364, 32
  %367 = trunc i64 %366 to i32
  %368 = or i32 %367, -2147483648
  %369 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 poison, i32 poison>, i32 %365, i64 2
  %370 = insertelement <4 x i32> %369, i32 %368, i64 3
  %371 = call i32 @llvm.smax.i32(i32 %359, i32 0)
  %372 = and i32 %371, 65535
  %373 = shl i32 %372, 16
  %374 = or i32 %373, 32767
  %375 = lshr i32 %371, 16
  %376 = and i32 %375, 65535
  %377 = or i32 %376, 8388608
  %378 = trunc i64 %363 to i32
  %379 = lshr i64 %363, 32
  %380 = trunc i64 %379 to i32
  %381 = and i32 %380, 65535
  %382 = insertelement <8 x i32> <i32 122748928, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %374, i64 2
  %383 = insertelement <8 x i32> %382, i32 %377, i64 3
  %384 = insertelement <8 x i32> %383, i32 32, i64 4
  %385 = insertelement <8 x i32> %384, i32 %378, i64 5
  %386 = insertelement <8 x i32> %385, i32 %381, i64 6
  %387 = insertelement <8 x i32> %386, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %370, <8 x i32> %387, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %388 = getelementptr bfloat, ptr addrspace(1) %0, i64 %358
  %389 = ptrtoint ptr addrspace(1) %388 to i64
  %390 = trunc i64 %389 to i32
  %391 = lshr i64 %389, 32
  %392 = trunc i64 %391 to i32
  %393 = or i32 %392, -2147483648
  %394 = insertelement <4 x i32> <i32 1, i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 8704), i32 poison, i32 poison>, i32 %390, i64 2
  %395 = insertelement <4 x i32> %394, i32 %393, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %395, <8 x i32> %387, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %396 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %223
  %397 = add i32 %396, 8704
  %398 = inttoptr i32 %397 to ptr addrspace(3)
  %399 = load <8 x bfloat>, ptr addrspace(3) %398, align 16
  %400 = add i32 %396, 8736
  %401 = inttoptr i32 %400 to ptr addrspace(3)
  %402 = load <8 x bfloat>, ptr addrspace(3) %401, align 16
  %403 = add i32 %396, 8768
  %404 = inttoptr i32 %403 to ptr addrspace(3)
  %405 = load <8 x bfloat>, ptr addrspace(3) %404, align 16
  %406 = add i32 %396, 8800
  %407 = inttoptr i32 %406 to ptr addrspace(3)
  %408 = load <8 x bfloat>, ptr addrspace(3) %407, align 16
  %409 = add i32 %396, 8832
  %410 = inttoptr i32 %409 to ptr addrspace(3)
  %411 = load <8 x bfloat>, ptr addrspace(3) %410, align 16
  %412 = add i32 %396, 8864
  %413 = inttoptr i32 %412 to ptr addrspace(3)
  %414 = load <8 x bfloat>, ptr addrspace(3) %413, align 16
  %415 = add i32 %396, 8896
  %416 = inttoptr i32 %415 to ptr addrspace(3)
  %417 = load <8 x bfloat>, ptr addrspace(3) %416, align 16
  %418 = add i32 %396, 8928
  %419 = inttoptr i32 %418 to ptr addrspace(3)
  %420 = load <8 x bfloat>, ptr addrspace(3) %419, align 16
  %421 = inttoptr i32 %396 to ptr addrspace(3)
  %422 = load <8 x bfloat>, ptr addrspace(3) %421, align 16
  %423 = add i32 %396, 32
  %424 = inttoptr i32 %423 to ptr addrspace(3)
  %425 = load <8 x bfloat>, ptr addrspace(3) %424, align 16
  %426 = add i32 %396, 64
  %427 = inttoptr i32 %426 to ptr addrspace(3)
  %428 = load <8 x bfloat>, ptr addrspace(3) %427, align 16
  %429 = add i32 %396, 96
  %430 = inttoptr i32 %429 to ptr addrspace(3)
  %431 = load <8 x bfloat>, ptr addrspace(3) %430, align 16
  %432 = add i32 %396, 128
  %433 = inttoptr i32 %432 to ptr addrspace(3)
  %434 = load <8 x bfloat>, ptr addrspace(3) %433, align 16
  %435 = add i32 %396, 160
  %436 = inttoptr i32 %435 to ptr addrspace(3)
  %437 = load <8 x bfloat>, ptr addrspace(3) %436, align 16
  %438 = add i32 %396, 192
  %439 = inttoptr i32 %438 to ptr addrspace(3)
  %440 = load <8 x bfloat>, ptr addrspace(3) %439, align 16
  %441 = add i32 %396, 224
  %442 = inttoptr i32 %441 to ptr addrspace(3)
  %443 = load <8 x bfloat>, ptr addrspace(3) %442, align 16
  %444 = add i32 %396, 13056
  %445 = inttoptr i32 %444 to ptr addrspace(3)
  %446 = load <8 x bfloat>, ptr addrspace(3) %445, align 16
  %447 = add i32 %396, 13088
  %448 = inttoptr i32 %447 to ptr addrspace(3)
  %449 = load <8 x bfloat>, ptr addrspace(3) %448, align 16
  %450 = add i32 %396, 13120
  %451 = inttoptr i32 %450 to ptr addrspace(3)
  %452 = load <8 x bfloat>, ptr addrspace(3) %451, align 16
  %453 = add i32 %396, 13152
  %454 = inttoptr i32 %453 to ptr addrspace(3)
  %455 = load <8 x bfloat>, ptr addrspace(3) %454, align 16
  %456 = add i32 %396, 13184
  %457 = inttoptr i32 %456 to ptr addrspace(3)
  %458 = load <8 x bfloat>, ptr addrspace(3) %457, align 16
  %459 = add i32 %396, 13216
  %460 = inttoptr i32 %459 to ptr addrspace(3)
  %461 = load <8 x bfloat>, ptr addrspace(3) %460, align 16
  %462 = add i32 %396, 13248
  %463 = inttoptr i32 %462 to ptr addrspace(3)
  %464 = load <8 x bfloat>, ptr addrspace(3) %463, align 16
  %465 = add i32 %396, 13280
  %466 = inttoptr i32 %465 to ptr addrspace(3)
  %467 = load <8 x bfloat>, ptr addrspace(3) %466, align 16
  %468 = add i32 %396, 4352
  %469 = inttoptr i32 %468 to ptr addrspace(3)
  %470 = load <8 x bfloat>, ptr addrspace(3) %469, align 16
  %471 = add i32 %396, 4384
  %472 = inttoptr i32 %471 to ptr addrspace(3)
  %473 = load <8 x bfloat>, ptr addrspace(3) %472, align 16
  %474 = add i32 %396, 4416
  %475 = inttoptr i32 %474 to ptr addrspace(3)
  %476 = load <8 x bfloat>, ptr addrspace(3) %475, align 16
  %477 = add i32 %396, 4448
  %478 = inttoptr i32 %477 to ptr addrspace(3)
  %479 = load <8 x bfloat>, ptr addrspace(3) %478, align 16
  %480 = add i32 %396, 4480
  %481 = inttoptr i32 %480 to ptr addrspace(3)
  %482 = load <8 x bfloat>, ptr addrspace(3) %481, align 16
  %483 = add i32 %396, 4512
  %484 = inttoptr i32 %483 to ptr addrspace(3)
  %485 = load <8 x bfloat>, ptr addrspace(3) %484, align 16
  %486 = add i32 %396, 4544
  %487 = inttoptr i32 %486 to ptr addrspace(3)
  %488 = load <8 x bfloat>, ptr addrspace(3) %487, align 16
  %489 = add i32 %396, 4576
  %490 = inttoptr i32 %489 to ptr addrspace(3)
  %491 = load <8 x bfloat>, ptr addrspace(3) %490, align 16
  %492 = shufflevector <8 x bfloat> %399, <8 x bfloat> %402, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %493 = shufflevector <8 x bfloat> %405, <8 x bfloat> %408, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %494 = shufflevector <8 x bfloat> %411, <8 x bfloat> %414, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %495 = shufflevector <8 x bfloat> %417, <8 x bfloat> %420, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %496 = shufflevector <8 x bfloat> %422, <8 x bfloat> %425, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %497 = shufflevector <8 x bfloat> %428, <8 x bfloat> %431, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %498 = shufflevector <8 x bfloat> %434, <8 x bfloat> %437, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %499 = shufflevector <8 x bfloat> %440, <8 x bfloat> %443, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %500 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %98, <16 x bfloat> %492, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %501 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %170, <16 x bfloat> %496, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %502 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %107, <16 x bfloat> %493, i16 0, <8 x float> %500, i1 false, i1 false)
  %503 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %175, <16 x bfloat> %497, i16 0, <8 x float> %501, i1 false, i1 false)
  %504 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %116, <16 x bfloat> %494, i16 0, <8 x float> %502, i1 false, i1 false)
  %505 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %180, <16 x bfloat> %498, i16 0, <8 x float> %503, i1 false, i1 false)
  %506 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %125, <16 x bfloat> %495, i16 0, <8 x float> %504, i1 false, i1 false)
  %507 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %185, <16 x bfloat> %499, i16 0, <8 x float> %505, i1 false, i1 false)
  %508 = add i32 %60, %206
  %509 = add i32 %336, %23
  %510 = icmp sgt i32 %508, %509
  %511 = and i1 %510, %242
  %512 = extractelement <8 x float> %506, i64 0
  %513 = fmul float %512, %16
  %514 = select i1 %511, float -3.000000e+38, float %513
  %515 = add i32 %508, 1
  %516 = icmp sgt i32 %515, %509
  %517 = and i1 %516, %242
  %518 = extractelement <8 x float> %506, i64 1
  %519 = fmul float %518, %16
  %520 = select i1 %517, float -3.000000e+38, float %519
  %521 = add i32 %508, 2
  %522 = icmp sgt i32 %521, %509
  %523 = and i1 %522, %242
  %524 = extractelement <8 x float> %506, i64 2
  %525 = fmul float %524, %16
  %526 = select i1 %523, float -3.000000e+38, float %525
  %527 = add i32 %508, 3
  %528 = icmp sgt i32 %527, %509
  %529 = and i1 %528, %242
  %530 = extractelement <8 x float> %506, i64 3
  %531 = fmul float %530, %16
  %532 = select i1 %529, float -3.000000e+38, float %531
  %533 = add i32 %508, 4
  %534 = icmp sgt i32 %533, %509
  %535 = and i1 %534, %242
  %536 = extractelement <8 x float> %506, i64 4
  %537 = fmul float %536, %16
  %538 = select i1 %535, float -3.000000e+38, float %537
  %539 = add i32 %508, 5
  %540 = icmp sgt i32 %539, %509
  %541 = and i1 %540, %242
  %542 = extractelement <8 x float> %506, i64 5
  %543 = fmul float %542, %16
  %544 = select i1 %541, float -3.000000e+38, float %543
  %545 = add i32 %508, 6
  %546 = icmp sgt i32 %545, %509
  %547 = and i1 %546, %242
  %548 = extractelement <8 x float> %506, i64 6
  %549 = fmul float %548, %16
  %550 = select i1 %547, float -3.000000e+38, float %549
  %551 = add i32 %508, 7
  %552 = icmp sgt i32 %551, %509
  %553 = and i1 %552, %242
  %554 = extractelement <8 x float> %506, i64 7
  %555 = fmul float %554, %16
  %556 = select i1 %553, float -3.000000e+38, float %555
  %557 = fsub float %514, %340
  %558 = fmul float %557, f0x3FB8AA3B
  %559 = call float @llvm.amdgcn.exp2.f32(float %558)
  %560 = fsub float %520, %340
  %561 = fmul float %560, f0x3FB8AA3B
  %562 = call float @llvm.amdgcn.exp2.f32(float %561)
  %563 = fsub float %526, %340
  %564 = fmul float %563, f0x3FB8AA3B
  %565 = call float @llvm.amdgcn.exp2.f32(float %564)
  %566 = fsub float %532, %340
  %567 = fmul float %566, f0x3FB8AA3B
  %568 = call float @llvm.amdgcn.exp2.f32(float %567)
  %569 = fsub float %538, %340
  %570 = fmul float %569, f0x3FB8AA3B
  %571 = call float @llvm.amdgcn.exp2.f32(float %570)
  %572 = fsub float %544, %340
  %573 = fmul float %572, f0x3FB8AA3B
  %574 = call float @llvm.amdgcn.exp2.f32(float %573)
  %575 = fsub float %550, %340
  %576 = fmul float %575, f0x3FB8AA3B
  %577 = call float @llvm.amdgcn.exp2.f32(float %576)
  %578 = fsub float %556, %340
  %579 = fmul float %578, f0x3FB8AA3B
  %580 = call float @llvm.amdgcn.exp2.f32(float %579)
  %581 = fptrunc float %559 to bfloat
  %582 = fptrunc float %562 to bfloat
  %583 = fptrunc float %565 to bfloat
  %584 = fptrunc float %568 to bfloat
  %585 = fptrunc float %571 to bfloat
  %586 = fptrunc float %574 to bfloat
  %587 = fptrunc float %577 to bfloat
  %588 = fptrunc float %580 to bfloat
  %589 = extractelement <8 x float> %507, i64 0
  %590 = fsub float %589, %342
  %591 = fmul float %559, %590
  %592 = fmul float %591, %16
  %593 = fptrunc float %592 to bfloat
  %594 = extractelement <8 x float> %507, i64 1
  %595 = fsub float %594, %342
  %596 = fmul float %562, %595
  %597 = fmul float %596, %16
  %598 = fptrunc float %597 to bfloat
  %599 = extractelement <8 x float> %507, i64 2
  %600 = fsub float %599, %342
  %601 = fmul float %565, %600
  %602 = fmul float %601, %16
  %603 = fptrunc float %602 to bfloat
  %604 = extractelement <8 x float> %507, i64 3
  %605 = fsub float %604, %342
  %606 = fmul float %568, %605
  %607 = fmul float %606, %16
  %608 = fptrunc float %607 to bfloat
  %609 = extractelement <8 x float> %507, i64 4
  %610 = fsub float %609, %342
  %611 = fmul float %571, %610
  %612 = fmul float %611, %16
  %613 = fptrunc float %612 to bfloat
  %614 = extractelement <8 x float> %507, i64 5
  %615 = fsub float %614, %342
  %616 = fmul float %574, %615
  %617 = fmul float %616, %16
  %618 = fptrunc float %617 to bfloat
  %619 = extractelement <8 x float> %507, i64 6
  %620 = fsub float %619, %342
  %621 = fmul float %577, %620
  %622 = fmul float %621, %16
  %623 = fptrunc float %622 to bfloat
  %624 = extractelement <8 x float> %507, i64 7
  %625 = fsub float %624, %342
  %626 = fmul float %580, %625
  %627 = fmul float %626, %16
  %628 = fptrunc float %627 to bfloat
  %629 = mul i32 %51, 80
  %630 = add i32 %629, %222
  %631 = insertelement <8 x bfloat> poison, bfloat %581, i64 0
  %632 = insertelement <8 x bfloat> %631, bfloat %582, i64 1
  %633 = insertelement <8 x bfloat> %632, bfloat %583, i64 2
  %634 = insertelement <8 x bfloat> %633, bfloat %584, i64 3
  %635 = insertelement <8 x bfloat> %634, bfloat %585, i64 4
  %636 = insertelement <8 x bfloat> %635, bfloat %586, i64 5
  %637 = insertelement <8 x bfloat> %636, bfloat %587, i64 6
  %638 = insertelement <8 x bfloat> %637, bfloat %588, i64 7
  %639 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %630
  %640 = insertelement <8 x bfloat> poison, bfloat %593, i64 0
  %641 = insertelement <8 x bfloat> %640, bfloat %598, i64 1
  %642 = insertelement <8 x bfloat> %641, bfloat %603, i64 2
  %643 = insertelement <8 x bfloat> %642, bfloat %608, i64 3
  %644 = insertelement <8 x bfloat> %643, bfloat %613, i64 4
  %645 = insertelement <8 x bfloat> %644, bfloat %618, i64 5
  %646 = insertelement <8 x bfloat> %645, bfloat %623, i64 6
  %647 = insertelement <8 x bfloat> %646, bfloat %628, i64 7
  %648 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %630
  %649 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %138, <16 x bfloat> %492, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %650 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %190, <16 x bfloat> %496, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %651 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %147, <16 x bfloat> %493, i16 0, <8 x float> %649, i1 false, i1 false)
  %652 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %195, <16 x bfloat> %497, i16 0, <8 x float> %650, i1 false, i1 false)
  %653 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %156, <16 x bfloat> %494, i16 0, <8 x float> %651, i1 false, i1 false)
  %654 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %200, <16 x bfloat> %498, i16 0, <8 x float> %652, i1 false, i1 false)
  %655 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %165, <16 x bfloat> %495, i16 0, <8 x float> %653, i1 false, i1 false)
  %656 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %205, <16 x bfloat> %499, i16 0, <8 x float> %654, i1 false, i1 false)
  %657 = add i32 %126, %206
  %658 = icmp sgt i32 %657, %509
  %659 = and i1 %658, %242
  %660 = extractelement <8 x float> %655, i64 0
  %661 = fmul float %660, %16
  %662 = select i1 %659, float -3.000000e+38, float %661
  %663 = add i32 %657, 1
  %664 = icmp sgt i32 %663, %509
  %665 = and i1 %664, %242
  %666 = extractelement <8 x float> %655, i64 1
  %667 = fmul float %666, %16
  %668 = select i1 %665, float -3.000000e+38, float %667
  %669 = add i32 %657, 2
  %670 = icmp sgt i32 %669, %509
  %671 = and i1 %670, %242
  %672 = extractelement <8 x float> %655, i64 2
  %673 = fmul float %672, %16
  %674 = select i1 %671, float -3.000000e+38, float %673
  %675 = add i32 %657, 3
  %676 = icmp sgt i32 %675, %509
  %677 = and i1 %676, %242
  %678 = extractelement <8 x float> %655, i64 3
  %679 = fmul float %678, %16
  %680 = select i1 %677, float -3.000000e+38, float %679
  %681 = add i32 %657, 4
  %682 = icmp sgt i32 %681, %509
  %683 = and i1 %682, %242
  %684 = extractelement <8 x float> %655, i64 4
  %685 = fmul float %684, %16
  %686 = select i1 %683, float -3.000000e+38, float %685
  %687 = add i32 %657, 5
  %688 = icmp sgt i32 %687, %509
  %689 = and i1 %688, %242
  %690 = extractelement <8 x float> %655, i64 5
  %691 = fmul float %690, %16
  %692 = select i1 %689, float -3.000000e+38, float %691
  %693 = add i32 %657, 6
  %694 = icmp sgt i32 %693, %509
  %695 = and i1 %694, %242
  %696 = extractelement <8 x float> %655, i64 6
  %697 = fmul float %696, %16
  %698 = select i1 %695, float -3.000000e+38, float %697
  %699 = add i32 %657, 7
  %700 = icmp sgt i32 %699, %509
  %701 = and i1 %700, %242
  %702 = extractelement <8 x float> %655, i64 7
  %703 = fmul float %702, %16
  %704 = select i1 %701, float -3.000000e+38, float %703
  %705 = fsub float %662, %340
  %706 = fmul float %705, f0x3FB8AA3B
  %707 = call float @llvm.amdgcn.exp2.f32(float %706)
  %708 = fsub float %668, %340
  %709 = fmul float %708, f0x3FB8AA3B
  %710 = call float @llvm.amdgcn.exp2.f32(float %709)
  %711 = fsub float %674, %340
  %712 = fmul float %711, f0x3FB8AA3B
  %713 = call float @llvm.amdgcn.exp2.f32(float %712)
  %714 = fsub float %680, %340
  %715 = fmul float %714, f0x3FB8AA3B
  %716 = call float @llvm.amdgcn.exp2.f32(float %715)
  %717 = fsub float %686, %340
  %718 = fmul float %717, f0x3FB8AA3B
  %719 = call float @llvm.amdgcn.exp2.f32(float %718)
  %720 = fsub float %692, %340
  %721 = fmul float %720, f0x3FB8AA3B
  %722 = call float @llvm.amdgcn.exp2.f32(float %721)
  %723 = fsub float %698, %340
  %724 = fmul float %723, f0x3FB8AA3B
  %725 = call float @llvm.amdgcn.exp2.f32(float %724)
  %726 = fsub float %704, %340
  %727 = fmul float %726, f0x3FB8AA3B
  %728 = call float @llvm.amdgcn.exp2.f32(float %727)
  %729 = fptrunc float %707 to bfloat
  %730 = fptrunc float %710 to bfloat
  %731 = fptrunc float %713 to bfloat
  %732 = fptrunc float %716 to bfloat
  %733 = fptrunc float %719 to bfloat
  %734 = fptrunc float %722 to bfloat
  %735 = fptrunc float %725 to bfloat
  %736 = fptrunc float %728 to bfloat
  %737 = extractelement <8 x float> %656, i64 0
  %738 = fsub float %737, %342
  %739 = fmul float %707, %738
  %740 = fmul float %739, %16
  %741 = fptrunc float %740 to bfloat
  %742 = extractelement <8 x float> %656, i64 1
  %743 = fsub float %742, %342
  %744 = fmul float %710, %743
  %745 = fmul float %744, %16
  %746 = fptrunc float %745 to bfloat
  %747 = extractelement <8 x float> %656, i64 2
  %748 = fsub float %747, %342
  %749 = fmul float %713, %748
  %750 = fmul float %749, %16
  %751 = fptrunc float %750 to bfloat
  %752 = extractelement <8 x float> %656, i64 3
  %753 = fsub float %752, %342
  %754 = fmul float %716, %753
  %755 = fmul float %754, %16
  %756 = fptrunc float %755 to bfloat
  %757 = extractelement <8 x float> %656, i64 4
  %758 = fsub float %757, %342
  %759 = fmul float %719, %758
  %760 = fmul float %759, %16
  %761 = fptrunc float %760 to bfloat
  %762 = extractelement <8 x float> %656, i64 5
  %763 = fsub float %762, %342
  %764 = fmul float %722, %763
  %765 = fmul float %764, %16
  %766 = fptrunc float %765 to bfloat
  %767 = extractelement <8 x float> %656, i64 6
  %768 = fsub float %767, %342
  %769 = fmul float %725, %768
  %770 = fmul float %769, %16
  %771 = fptrunc float %770 to bfloat
  %772 = extractelement <8 x float> %656, i64 7
  %773 = fsub float %772, %342
  %774 = fmul float %728, %773
  %775 = fmul float %774, %16
  %776 = fptrunc float %775 to bfloat
  %777 = add i32 %629, 32
  %778 = add i32 %777, %222
  %779 = insertelement <8 x bfloat> poison, bfloat %729, i64 0
  %780 = insertelement <8 x bfloat> %779, bfloat %730, i64 1
  %781 = insertelement <8 x bfloat> %780, bfloat %731, i64 2
  %782 = insertelement <8 x bfloat> %781, bfloat %732, i64 3
  %783 = insertelement <8 x bfloat> %782, bfloat %733, i64 4
  %784 = insertelement <8 x bfloat> %783, bfloat %734, i64 5
  %785 = insertelement <8 x bfloat> %784, bfloat %735, i64 6
  %786 = insertelement <8 x bfloat> %785, bfloat %736, i64 7
  %787 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %778
  %788 = insertelement <8 x bfloat> poison, bfloat %741, i64 0
  %789 = insertelement <8 x bfloat> %788, bfloat %746, i64 1
  %790 = insertelement <8 x bfloat> %789, bfloat %751, i64 2
  %791 = insertelement <8 x bfloat> %790, bfloat %756, i64 3
  %792 = insertelement <8 x bfloat> %791, bfloat %761, i64 4
  %793 = insertelement <8 x bfloat> %792, bfloat %766, i64 5
  %794 = insertelement <8 x bfloat> %793, bfloat %771, i64 6
  %795 = insertelement <8 x bfloat> %794, bfloat %776, i64 7
  %796 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %778
  %797 = shufflevector <8 x bfloat> %446, <8 x bfloat> %449, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %798 = shufflevector <8 x bfloat> %452, <8 x bfloat> %455, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %799 = shufflevector <8 x bfloat> %458, <8 x bfloat> %461, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %800 = shufflevector <8 x bfloat> %464, <8 x bfloat> %467, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %801 = shufflevector <8 x bfloat> %470, <8 x bfloat> %473, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %802 = shufflevector <8 x bfloat> %476, <8 x bfloat> %479, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %803 = shufflevector <8 x bfloat> %482, <8 x bfloat> %485, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %804 = shufflevector <8 x bfloat> %488, <8 x bfloat> %491, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %805 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %98, <16 x bfloat> %797, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %806 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %170, <16 x bfloat> %801, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %807 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %107, <16 x bfloat> %798, i16 0, <8 x float> %805, i1 false, i1 false)
  %808 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %175, <16 x bfloat> %802, i16 0, <8 x float> %806, i1 false, i1 false)
  %809 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %116, <16 x bfloat> %799, i16 0, <8 x float> %807, i1 false, i1 false)
  %810 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %180, <16 x bfloat> %803, i16 0, <8 x float> %808, i1 false, i1 false)
  %811 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %125, <16 x bfloat> %800, i16 0, <8 x float> %809, i1 false, i1 false)
  %812 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %185, <16 x bfloat> %804, i16 0, <8 x float> %810, i1 false, i1 false)
  %813 = add i32 %344, %23
  %814 = icmp sgt i32 %508, %813
  %815 = and i1 %814, %242
  %816 = extractelement <8 x float> %811, i64 0
  %817 = fmul float %816, %16
  %818 = select i1 %815, float -3.000000e+38, float %817
  %819 = icmp sgt i32 %515, %813
  %820 = and i1 %819, %242
  %821 = extractelement <8 x float> %811, i64 1
  %822 = fmul float %821, %16
  %823 = select i1 %820, float -3.000000e+38, float %822
  %824 = icmp sgt i32 %521, %813
  %825 = and i1 %824, %242
  %826 = extractelement <8 x float> %811, i64 2
  %827 = fmul float %826, %16
  %828 = select i1 %825, float -3.000000e+38, float %827
  %829 = icmp sgt i32 %527, %813
  %830 = and i1 %829, %242
  %831 = extractelement <8 x float> %811, i64 3
  %832 = fmul float %831, %16
  %833 = select i1 %830, float -3.000000e+38, float %832
  %834 = icmp sgt i32 %533, %813
  %835 = and i1 %834, %242
  %836 = extractelement <8 x float> %811, i64 4
  %837 = fmul float %836, %16
  %838 = select i1 %835, float -3.000000e+38, float %837
  %839 = icmp sgt i32 %539, %813
  %840 = and i1 %839, %242
  %841 = extractelement <8 x float> %811, i64 5
  %842 = fmul float %841, %16
  %843 = select i1 %840, float -3.000000e+38, float %842
  %844 = icmp sgt i32 %545, %813
  %845 = and i1 %844, %242
  %846 = extractelement <8 x float> %811, i64 6
  %847 = fmul float %846, %16
  %848 = select i1 %845, float -3.000000e+38, float %847
  %849 = icmp sgt i32 %551, %813
  %850 = and i1 %849, %242
  %851 = extractelement <8 x float> %811, i64 7
  %852 = fmul float %851, %16
  %853 = select i1 %850, float -3.000000e+38, float %852
  %854 = fsub float %818, %348
  %855 = fmul float %854, f0x3FB8AA3B
  %856 = call float @llvm.amdgcn.exp2.f32(float %855)
  %857 = fsub float %823, %348
  %858 = fmul float %857, f0x3FB8AA3B
  %859 = call float @llvm.amdgcn.exp2.f32(float %858)
  %860 = fsub float %828, %348
  %861 = fmul float %860, f0x3FB8AA3B
  %862 = call float @llvm.amdgcn.exp2.f32(float %861)
  %863 = fsub float %833, %348
  %864 = fmul float %863, f0x3FB8AA3B
  %865 = call float @llvm.amdgcn.exp2.f32(float %864)
  %866 = fsub float %838, %348
  %867 = fmul float %866, f0x3FB8AA3B
  %868 = call float @llvm.amdgcn.exp2.f32(float %867)
  %869 = fsub float %843, %348
  %870 = fmul float %869, f0x3FB8AA3B
  %871 = call float @llvm.amdgcn.exp2.f32(float %870)
  %872 = fsub float %848, %348
  %873 = fmul float %872, f0x3FB8AA3B
  %874 = call float @llvm.amdgcn.exp2.f32(float %873)
  %875 = fsub float %853, %348
  %876 = fmul float %875, f0x3FB8AA3B
  %877 = call float @llvm.amdgcn.exp2.f32(float %876)
  %878 = fptrunc float %856 to bfloat
  %879 = fptrunc float %859 to bfloat
  %880 = fptrunc float %862 to bfloat
  %881 = fptrunc float %865 to bfloat
  %882 = fptrunc float %868 to bfloat
  %883 = fptrunc float %871 to bfloat
  %884 = fptrunc float %874 to bfloat
  %885 = fptrunc float %877 to bfloat
  %886 = extractelement <8 x float> %812, i64 0
  %887 = fsub float %886, %350
  %888 = fmul float %856, %887
  %889 = fmul float %888, %16
  %890 = fptrunc float %889 to bfloat
  %891 = extractelement <8 x float> %812, i64 1
  %892 = fsub float %891, %350
  %893 = fmul float %859, %892
  %894 = fmul float %893, %16
  %895 = fptrunc float %894 to bfloat
  %896 = extractelement <8 x float> %812, i64 2
  %897 = fsub float %896, %350
  %898 = fmul float %862, %897
  %899 = fmul float %898, %16
  %900 = fptrunc float %899 to bfloat
  %901 = extractelement <8 x float> %812, i64 3
  %902 = fsub float %901, %350
  %903 = fmul float %865, %902
  %904 = fmul float %903, %16
  %905 = fptrunc float %904 to bfloat
  %906 = extractelement <8 x float> %812, i64 4
  %907 = fsub float %906, %350
  %908 = fmul float %868, %907
  %909 = fmul float %908, %16
  %910 = fptrunc float %909 to bfloat
  %911 = extractelement <8 x float> %812, i64 5
  %912 = fsub float %911, %350
  %913 = fmul float %871, %912
  %914 = fmul float %913, %16
  %915 = fptrunc float %914 to bfloat
  %916 = extractelement <8 x float> %812, i64 6
  %917 = fsub float %916, %350
  %918 = fmul float %874, %917
  %919 = fmul float %918, %16
  %920 = fptrunc float %919 to bfloat
  %921 = extractelement <8 x float> %812, i64 7
  %922 = fsub float %921, %350
  %923 = fmul float %877, %922
  %924 = fmul float %923, %16
  %925 = fptrunc float %924 to bfloat
  %926 = add i32 %51, 16
  %927 = mul i32 %926, 80
  %928 = add i32 %927, %222
  %929 = insertelement <8 x bfloat> poison, bfloat %878, i64 0
  %930 = insertelement <8 x bfloat> %929, bfloat %879, i64 1
  %931 = insertelement <8 x bfloat> %930, bfloat %880, i64 2
  %932 = insertelement <8 x bfloat> %931, bfloat %881, i64 3
  %933 = insertelement <8 x bfloat> %932, bfloat %882, i64 4
  %934 = insertelement <8 x bfloat> %933, bfloat %883, i64 5
  %935 = insertelement <8 x bfloat> %934, bfloat %884, i64 6
  %936 = insertelement <8 x bfloat> %935, bfloat %885, i64 7
  %937 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %928
  %938 = insertelement <8 x bfloat> poison, bfloat %890, i64 0
  %939 = insertelement <8 x bfloat> %938, bfloat %895, i64 1
  %940 = insertelement <8 x bfloat> %939, bfloat %900, i64 2
  %941 = insertelement <8 x bfloat> %940, bfloat %905, i64 3
  %942 = insertelement <8 x bfloat> %941, bfloat %910, i64 4
  %943 = insertelement <8 x bfloat> %942, bfloat %915, i64 5
  %944 = insertelement <8 x bfloat> %943, bfloat %920, i64 6
  %945 = insertelement <8 x bfloat> %944, bfloat %925, i64 7
  %946 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %928
  %947 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %138, <16 x bfloat> %797, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %948 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %190, <16 x bfloat> %801, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %949 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %147, <16 x bfloat> %798, i16 0, <8 x float> %947, i1 false, i1 false)
  %950 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %195, <16 x bfloat> %802, i16 0, <8 x float> %948, i1 false, i1 false)
  %951 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %156, <16 x bfloat> %799, i16 0, <8 x float> %949, i1 false, i1 false)
  %952 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %200, <16 x bfloat> %803, i16 0, <8 x float> %950, i1 false, i1 false)
  %953 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %165, <16 x bfloat> %800, i16 0, <8 x float> %951, i1 false, i1 false)
  %954 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %205, <16 x bfloat> %804, i16 0, <8 x float> %952, i1 false, i1 false)
  %955 = icmp sgt i32 %657, %813
  %956 = and i1 %955, %242
  %957 = extractelement <8 x float> %953, i64 0
  %958 = fmul float %957, %16
  %959 = select i1 %956, float -3.000000e+38, float %958
  %960 = icmp sgt i32 %663, %813
  %961 = and i1 %960, %242
  %962 = extractelement <8 x float> %953, i64 1
  %963 = fmul float %962, %16
  %964 = select i1 %961, float -3.000000e+38, float %963
  %965 = icmp sgt i32 %669, %813
  %966 = and i1 %965, %242
  %967 = extractelement <8 x float> %953, i64 2
  %968 = fmul float %967, %16
  %969 = select i1 %966, float -3.000000e+38, float %968
  %970 = icmp sgt i32 %675, %813
  %971 = and i1 %970, %242
  %972 = extractelement <8 x float> %953, i64 3
  %973 = fmul float %972, %16
  %974 = select i1 %971, float -3.000000e+38, float %973
  %975 = icmp sgt i32 %681, %813
  %976 = and i1 %975, %242
  %977 = extractelement <8 x float> %953, i64 4
  %978 = fmul float %977, %16
  %979 = select i1 %976, float -3.000000e+38, float %978
  %980 = icmp sgt i32 %687, %813
  %981 = and i1 %980, %242
  %982 = extractelement <8 x float> %953, i64 5
  %983 = fmul float %982, %16
  %984 = select i1 %981, float -3.000000e+38, float %983
  %985 = icmp sgt i32 %693, %813
  %986 = and i1 %985, %242
  %987 = extractelement <8 x float> %953, i64 6
  %988 = fmul float %987, %16
  %989 = select i1 %986, float -3.000000e+38, float %988
  %990 = icmp sgt i32 %699, %813
  %991 = and i1 %990, %242
  %992 = extractelement <8 x float> %953, i64 7
  %993 = fmul float %992, %16
  %994 = select i1 %991, float -3.000000e+38, float %993
  %995 = fsub float %959, %348
  %996 = fmul float %995, f0x3FB8AA3B
  %997 = call float @llvm.amdgcn.exp2.f32(float %996)
  %998 = fsub float %964, %348
  %999 = fmul float %998, f0x3FB8AA3B
  %1000 = call float @llvm.amdgcn.exp2.f32(float %999)
  %1001 = fsub float %969, %348
  %1002 = fmul float %1001, f0x3FB8AA3B
  %1003 = call float @llvm.amdgcn.exp2.f32(float %1002)
  %1004 = fsub float %974, %348
  %1005 = fmul float %1004, f0x3FB8AA3B
  %1006 = call float @llvm.amdgcn.exp2.f32(float %1005)
  %1007 = fsub float %979, %348
  %1008 = fmul float %1007, f0x3FB8AA3B
  %1009 = call float @llvm.amdgcn.exp2.f32(float %1008)
  %1010 = fsub float %984, %348
  %1011 = fmul float %1010, f0x3FB8AA3B
  %1012 = call float @llvm.amdgcn.exp2.f32(float %1011)
  %1013 = fsub float %989, %348
  %1014 = fmul float %1013, f0x3FB8AA3B
  %1015 = call float @llvm.amdgcn.exp2.f32(float %1014)
  %1016 = fsub float %994, %348
  %1017 = fmul float %1016, f0x3FB8AA3B
  %1018 = call float @llvm.amdgcn.exp2.f32(float %1017)
  %1019 = fptrunc float %997 to bfloat
  %1020 = fptrunc float %1000 to bfloat
  %1021 = fptrunc float %1003 to bfloat
  %1022 = fptrunc float %1006 to bfloat
  %1023 = fptrunc float %1009 to bfloat
  %1024 = fptrunc float %1012 to bfloat
  %1025 = fptrunc float %1015 to bfloat
  %1026 = fptrunc float %1018 to bfloat
  %1027 = extractelement <8 x float> %954, i64 0
  %1028 = fsub float %1027, %350
  %1029 = fmul float %997, %1028
  %1030 = fmul float %1029, %16
  %1031 = fptrunc float %1030 to bfloat
  %1032 = extractelement <8 x float> %954, i64 1
  %1033 = fsub float %1032, %350
  %1034 = fmul float %1000, %1033
  %1035 = fmul float %1034, %16
  %1036 = fptrunc float %1035 to bfloat
  %1037 = extractelement <8 x float> %954, i64 2
  %1038 = fsub float %1037, %350
  %1039 = fmul float %1003, %1038
  %1040 = fmul float %1039, %16
  %1041 = fptrunc float %1040 to bfloat
  %1042 = extractelement <8 x float> %954, i64 3
  %1043 = fsub float %1042, %350
  %1044 = fmul float %1006, %1043
  %1045 = fmul float %1044, %16
  %1046 = fptrunc float %1045 to bfloat
  %1047 = extractelement <8 x float> %954, i64 4
  %1048 = fsub float %1047, %350
  %1049 = fmul float %1009, %1048
  %1050 = fmul float %1049, %16
  %1051 = fptrunc float %1050 to bfloat
  %1052 = extractelement <8 x float> %954, i64 5
  %1053 = fsub float %1052, %350
  %1054 = fmul float %1012, %1053
  %1055 = fmul float %1054, %16
  %1056 = fptrunc float %1055 to bfloat
  %1057 = extractelement <8 x float> %954, i64 6
  %1058 = fsub float %1057, %350
  %1059 = fmul float %1015, %1058
  %1060 = fmul float %1059, %16
  %1061 = fptrunc float %1060 to bfloat
  %1062 = extractelement <8 x float> %954, i64 7
  %1063 = fsub float %1062, %350
  %1064 = fmul float %1018, %1063
  %1065 = fmul float %1064, %16
  %1066 = fptrunc float %1065 to bfloat
  %1067 = add i32 %927, 32
  %1068 = add i32 %1067, %222
  %1069 = insertelement <8 x bfloat> poison, bfloat %1019, i64 0
  %1070 = insertelement <8 x bfloat> %1069, bfloat %1020, i64 1
  %1071 = insertelement <8 x bfloat> %1070, bfloat %1021, i64 2
  %1072 = insertelement <8 x bfloat> %1071, bfloat %1022, i64 3
  %1073 = insertelement <8 x bfloat> %1072, bfloat %1023, i64 4
  %1074 = insertelement <8 x bfloat> %1073, bfloat %1024, i64 5
  %1075 = insertelement <8 x bfloat> %1074, bfloat %1025, i64 6
  %1076 = insertelement <8 x bfloat> %1075, bfloat %1026, i64 7
  %1077 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %1068
  %1078 = insertelement <8 x bfloat> poison, bfloat %1031, i64 0
  %1079 = insertelement <8 x bfloat> %1078, bfloat %1036, i64 1
  %1080 = insertelement <8 x bfloat> %1079, bfloat %1041, i64 2
  %1081 = insertelement <8 x bfloat> %1080, bfloat %1046, i64 3
  %1082 = insertelement <8 x bfloat> %1081, bfloat %1051, i64 4
  %1083 = insertelement <8 x bfloat> %1082, bfloat %1056, i64 5
  %1084 = insertelement <8 x bfloat> %1083, bfloat %1061, i64 6
  %1085 = insertelement <8 x bfloat> %1084, bfloat %1066, i64 7
  %1086 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %1068
  %1087 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %220
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1088 = inttoptr i32 %639 to ptr addrspace(3)
  store <8 x bfloat> %638, ptr addrspace(3) %1088, align 16
  %1089 = inttoptr i32 %648 to ptr addrspace(3)
  store <8 x bfloat> %647, ptr addrspace(3) %1089, align 16
  %1090 = inttoptr i32 %787 to ptr addrspace(3)
  store <8 x bfloat> %786, ptr addrspace(3) %1090, align 16
  %1091 = inttoptr i32 %796 to ptr addrspace(3)
  store <8 x bfloat> %795, ptr addrspace(3) %1091, align 16
  %1092 = inttoptr i32 %937 to ptr addrspace(3)
  store <8 x bfloat> %936, ptr addrspace(3) %1092, align 16
  %1093 = inttoptr i32 %946 to ptr addrspace(3)
  store <8 x bfloat> %945, ptr addrspace(3) %1093, align 16
  %1094 = inttoptr i32 %1077 to ptr addrspace(3)
  store <8 x bfloat> %1076, ptr addrspace(3) %1094, align 16
  %1095 = inttoptr i32 %1086 to ptr addrspace(3)
  store <8 x bfloat> %1085, ptr addrspace(3) %1095, align 16
  %1096 = mul i32 %208, 80
  %1097 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %1096
  %1098 = add i32 %1097, %219
  %1099 = inttoptr i32 %1098 to ptr addrspace(3)
  %1100 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1099)
  %1101 = add i32 %1098, 1280
  %1102 = inttoptr i32 %1101 to ptr addrspace(3)
  %1103 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1102)
  %1104 = shufflevector <8 x bfloat> %1100, <8 x bfloat> %1103, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1105 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %1096
  %1106 = add i32 %1105, %219
  %1107 = inttoptr i32 %1106 to ptr addrspace(3)
  %1108 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1107)
  %1109 = add i32 %1106, 1280
  %1110 = inttoptr i32 %1109 to ptr addrspace(3)
  %1111 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1110)
  %1112 = shufflevector <8 x bfloat> %1108, <8 x bfloat> %1111, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1113 = add i32 %219, 32
  %1114 = add i32 %1097, %1113
  %1115 = inttoptr i32 %1114 to ptr addrspace(3)
  %1116 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1115)
  %1117 = add i32 %1114, 1280
  %1118 = inttoptr i32 %1117 to ptr addrspace(3)
  %1119 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1118)
  %1120 = shufflevector <8 x bfloat> %1116, <8 x bfloat> %1119, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1121 = add i32 %1105, %1113
  %1122 = inttoptr i32 %1121 to ptr addrspace(3)
  %1123 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1122)
  %1124 = add i32 %1121, 1280
  %1125 = inttoptr i32 %1124 to ptr addrspace(3)
  %1126 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1125)
  %1127 = shufflevector <8 x bfloat> %1123, <8 x bfloat> %1126, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1128 = inttoptr i32 %1087 to ptr addrspace(3)
  %1129 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1128)
  %1130 = add i32 %1087, 4352
  %1131 = inttoptr i32 %1130 to ptr addrspace(3)
  %1132 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1131)
  %1133 = shufflevector <8 x bfloat> %1129, <8 x bfloat> %1132, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1134 = add i32 %1087, 8704
  %1135 = inttoptr i32 %1134 to ptr addrspace(3)
  %1136 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1135)
  %1137 = add i32 %1087, 13056
  %1138 = inttoptr i32 %1137 to ptr addrspace(3)
  %1139 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1138)
  %1140 = shufflevector <8 x bfloat> %1136, <8 x bfloat> %1139, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1141 = add i32 %1087, 32
  %1142 = inttoptr i32 %1141 to ptr addrspace(3)
  %1143 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1142)
  %1144 = add i32 %1087, 4384
  %1145 = inttoptr i32 %1144 to ptr addrspace(3)
  %1146 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1145)
  %1147 = shufflevector <8 x bfloat> %1143, <8 x bfloat> %1146, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1148 = add i32 %1087, 8736
  %1149 = inttoptr i32 %1148 to ptr addrspace(3)
  %1150 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1149)
  %1151 = add i32 %1087, 13088
  %1152 = inttoptr i32 %1151 to ptr addrspace(3)
  %1153 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1152)
  %1154 = shufflevector <8 x bfloat> %1150, <8 x bfloat> %1153, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1155 = add i32 %1087, 64
  %1156 = inttoptr i32 %1155 to ptr addrspace(3)
  %1157 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1156)
  %1158 = add i32 %1087, 4416
  %1159 = inttoptr i32 %1158 to ptr addrspace(3)
  %1160 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1159)
  %1161 = shufflevector <8 x bfloat> %1157, <8 x bfloat> %1160, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1162 = add i32 %1087, 8768
  %1163 = inttoptr i32 %1162 to ptr addrspace(3)
  %1164 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1163)
  %1165 = add i32 %1087, 13120
  %1166 = inttoptr i32 %1165 to ptr addrspace(3)
  %1167 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1166)
  %1168 = shufflevector <8 x bfloat> %1164, <8 x bfloat> %1167, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1169 = add i32 %1087, 96
  %1170 = inttoptr i32 %1169 to ptr addrspace(3)
  %1171 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1170)
  %1172 = add i32 %1087, 4448
  %1173 = inttoptr i32 %1172 to ptr addrspace(3)
  %1174 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1173)
  %1175 = shufflevector <8 x bfloat> %1171, <8 x bfloat> %1174, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1176 = add i32 %1087, 8800
  %1177 = inttoptr i32 %1176 to ptr addrspace(3)
  %1178 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1177)
  %1179 = add i32 %1087, 13152
  %1180 = inttoptr i32 %1179 to ptr addrspace(3)
  %1181 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1180)
  %1182 = shufflevector <8 x bfloat> %1178, <8 x bfloat> %1181, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1183 = add i32 %1087, 128
  %1184 = inttoptr i32 %1183 to ptr addrspace(3)
  %1185 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1184)
  %1186 = add i32 %1087, 4480
  %1187 = inttoptr i32 %1186 to ptr addrspace(3)
  %1188 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1187)
  %1189 = shufflevector <8 x bfloat> %1185, <8 x bfloat> %1188, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1190 = add i32 %1087, 8832
  %1191 = inttoptr i32 %1190 to ptr addrspace(3)
  %1192 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1191)
  %1193 = add i32 %1087, 13184
  %1194 = inttoptr i32 %1193 to ptr addrspace(3)
  %1195 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1194)
  %1196 = shufflevector <8 x bfloat> %1192, <8 x bfloat> %1195, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1197 = add i32 %1087, 160
  %1198 = inttoptr i32 %1197 to ptr addrspace(3)
  %1199 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1198)
  %1200 = add i32 %1087, 4512
  %1201 = inttoptr i32 %1200 to ptr addrspace(3)
  %1202 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1201)
  %1203 = shufflevector <8 x bfloat> %1199, <8 x bfloat> %1202, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1204 = add i32 %1087, 8864
  %1205 = inttoptr i32 %1204 to ptr addrspace(3)
  %1206 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1205)
  %1207 = add i32 %1087, 13216
  %1208 = inttoptr i32 %1207 to ptr addrspace(3)
  %1209 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1208)
  %1210 = shufflevector <8 x bfloat> %1206, <8 x bfloat> %1209, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1211 = add i32 %1087, 192
  %1212 = inttoptr i32 %1211 to ptr addrspace(3)
  %1213 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1212)
  %1214 = add i32 %1087, 4544
  %1215 = inttoptr i32 %1214 to ptr addrspace(3)
  %1216 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1215)
  %1217 = shufflevector <8 x bfloat> %1213, <8 x bfloat> %1216, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1218 = add i32 %1087, 8896
  %1219 = inttoptr i32 %1218 to ptr addrspace(3)
  %1220 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1219)
  %1221 = add i32 %1087, 13248
  %1222 = inttoptr i32 %1221 to ptr addrspace(3)
  %1223 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1222)
  %1224 = shufflevector <8 x bfloat> %1220, <8 x bfloat> %1223, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1225 = add i32 %1087, 224
  %1226 = inttoptr i32 %1225 to ptr addrspace(3)
  %1227 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1226)
  %1228 = add i32 %1087, 4576
  %1229 = inttoptr i32 %1228 to ptr addrspace(3)
  %1230 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1229)
  %1231 = shufflevector <8 x bfloat> %1227, <8 x bfloat> %1230, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1232 = add i32 %1087, 8928
  %1233 = inttoptr i32 %1232 to ptr addrspace(3)
  %1234 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1233)
  %1235 = add i32 %1087, 13280
  %1236 = inttoptr i32 %1235 to ptr addrspace(3)
  %1237 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1236)
  %1238 = shufflevector <8 x bfloat> %1234, <8 x bfloat> %1237, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1239 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1104, <16 x bfloat> %1133, i16 0, <8 x float> %288, i1 false, i1 false)
  %1240 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1104, <16 x bfloat> %1147, i16 0, <8 x float> %289, i1 true, i1 false)
  %1241 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1104, <16 x bfloat> %1161, i16 0, <8 x float> %290, i1 true, i1 false)
  %1242 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1104, <16 x bfloat> %1175, i16 0, <8 x float> %291, i1 true, i1 false)
  %1243 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1104, <16 x bfloat> %1189, i16 0, <8 x float> %292, i1 true, i1 false)
  %1244 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1104, <16 x bfloat> %1203, i16 0, <8 x float> %293, i1 true, i1 false)
  %1245 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1104, <16 x bfloat> %1217, i16 0, <8 x float> %294, i1 true, i1 false)
  %1246 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1104, <16 x bfloat> %1231, i16 0, <8 x float> %295, i1 true, i1 false)
  %1247 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1112, <16 x bfloat> %1140, i16 0, <8 x float> %304, i1 false, i1 false)
  %1248 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1112, <16 x bfloat> %1154, i16 0, <8 x float> %305, i1 true, i1 false)
  %1249 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1112, <16 x bfloat> %1168, i16 0, <8 x float> %306, i1 true, i1 false)
  %1250 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1112, <16 x bfloat> %1182, i16 0, <8 x float> %307, i1 true, i1 false)
  %1251 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1112, <16 x bfloat> %1196, i16 0, <8 x float> %308, i1 true, i1 false)
  %1252 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1112, <16 x bfloat> %1210, i16 0, <8 x float> %309, i1 true, i1 false)
  %1253 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1112, <16 x bfloat> %1224, i16 0, <8 x float> %310, i1 true, i1 false)
  %1254 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1112, <16 x bfloat> %1238, i16 0, <8 x float> %311, i1 true, i1 false)
  %1255 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1120, <16 x bfloat> %1133, i16 0, <8 x float> %296, i1 false, i1 false)
  %1256 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1120, <16 x bfloat> %1147, i16 0, <8 x float> %297, i1 true, i1 false)
  %1257 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1120, <16 x bfloat> %1161, i16 0, <8 x float> %298, i1 true, i1 false)
  %1258 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1120, <16 x bfloat> %1175, i16 0, <8 x float> %299, i1 true, i1 false)
  %1259 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1120, <16 x bfloat> %1189, i16 0, <8 x float> %300, i1 true, i1 false)
  %1260 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1120, <16 x bfloat> %1203, i16 0, <8 x float> %301, i1 true, i1 false)
  %1261 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1120, <16 x bfloat> %1217, i16 0, <8 x float> %302, i1 true, i1 false)
  %1262 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1120, <16 x bfloat> %1231, i16 0, <8 x float> %303, i1 true, i1 false)
  %1263 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1127, <16 x bfloat> %1140, i16 0, <8 x float> %312, i1 false, i1 false)
  %1264 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1127, <16 x bfloat> %1154, i16 0, <8 x float> %313, i1 true, i1 false)
  %1265 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1127, <16 x bfloat> %1168, i16 0, <8 x float> %314, i1 true, i1 false)
  %1266 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1127, <16 x bfloat> %1182, i16 0, <8 x float> %315, i1 true, i1 false)
  %1267 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1127, <16 x bfloat> %1196, i16 0, <8 x float> %316, i1 true, i1 false)
  %1268 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1127, <16 x bfloat> %1210, i16 0, <8 x float> %317, i1 true, i1 false)
  %1269 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1127, <16 x bfloat> %1224, i16 0, <8 x float> %318, i1 true, i1 false)
  %1270 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1127, <16 x bfloat> %1238, i16 0, <8 x float> %319, i1 true, i1 false)
  %1271 = add i64 %287, 1
  br label %286

1272:                                             ; preds = %286
  %1273 = icmp slt i32 %283, %231
  %1274 = sub i32 %231, 1
  %1275 = select i1 %1273, i32 %283, i32 %1274
  %1276 = call i32 @llvm.smax.i32(i32 %1275, i32 0)
  %1277 = mul i32 %42, %21
  %1278 = mul i32 %1276, 32
  %1279 = mul i32 %50, %19
  %1280 = add i32 %1279, %1277
  %1281 = mul i32 %1280, %17
  %1282 = add i32 %1281, %1278
  %1283 = mul i32 %1282, 4
  %1284 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1283)
  %1285 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %71, i32 %85, i32 %1284, i32 0)
  %1286 = insertelement <1 x float> poison, float %1285, i32 0
  %1287 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %72, i32 %85, i32 %1284, i32 0)
  %1288 = insertelement <1 x float> poison, float %1287, i32 0
  %1289 = add i32 %1284, 64
  %1290 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %71, i32 %85, i32 %1289, i32 0)
  %1291 = insertelement <1 x float> poison, float %1290, i32 0
  %1292 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %72, i32 %85, i32 %1289, i32 0)
  %1293 = insertelement <1 x float> poison, float %1292, i32 0
  %1294 = mul i32 %21, %279
  %1295 = mul i32 %50, %17
  %1296 = add i32 %1295, %1278
  %1297 = sext i32 %1296 to i64
  %1298 = sext i32 %19 to i64
  %1299 = mul i64 %1297, %1298
  %1300 = sext i32 %1277 to i64
  %1301 = add i64 %1299, %1300
  %1302 = mul i64 %1301, 128
  %1303 = sub i32 %17, %1278
  %1304 = getelementptr bfloat, ptr addrspace(1) %6, i64 %1302
  %1305 = sext i32 %86 to i64
  %1306 = icmp eq i64 %1305, -2147483648
  %1307 = select i1 %1306, i64 128, i64 %1305
  %1308 = ptrtoint ptr addrspace(1) %1304 to i64
  %1309 = trunc i64 %1308 to i32
  %1310 = lshr i64 %1308, 32
  %1311 = trunc i64 %1310 to i32
  %1312 = or i32 %1311, -2147483648
  %1313 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 poison, i32 poison>, i32 %1309, i64 2
  %1314 = insertelement <4 x i32> %1313, i32 %1312, i64 3
  %1315 = call i32 @llvm.smax.i32(i32 %1303, i32 0)
  %1316 = and i32 %1315, 65535
  %1317 = shl i32 %1316, 16
  %1318 = or i32 %1317, 32767
  %1319 = lshr i32 %1315, 16
  %1320 = and i32 %1319, 65535
  %1321 = or i32 %1320, 8388608
  %1322 = trunc i64 %1307 to i32
  %1323 = lshr i64 %1307, 32
  %1324 = trunc i64 %1323 to i32
  %1325 = and i32 %1324, 65535
  %1326 = insertelement <8 x i32> <i32 122748928, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %1318, i64 2
  %1327 = insertelement <8 x i32> %1326, i32 %1321, i64 3
  %1328 = insertelement <8 x i32> %1327, i32 32, i64 4
  %1329 = insertelement <8 x i32> %1328, i32 %1322, i64 5
  %1330 = insertelement <8 x i32> %1329, i32 %1325, i64 6
  %1331 = insertelement <8 x i32> %1330, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %1314, <8 x i32> %1331, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %1332 = getelementptr bfloat, ptr addrspace(1) %0, i64 %1302
  %1333 = ptrtoint ptr addrspace(1) %1332 to i64
  %1334 = trunc i64 %1333 to i32
  %1335 = lshr i64 %1333, 32
  %1336 = trunc i64 %1335 to i32
  %1337 = or i32 %1336, -2147483648
  %1338 = insertelement <4 x i32> <i32 1, i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 8704), i32 poison, i32 poison>, i32 %1334, i64 2
  %1339 = insertelement <4 x i32> %1338, i32 %1337, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %1339, <8 x i32> %1331, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %1340 = icmp sgt i32 %21, 1
  %1341 = icmp sle i32 %21, 1
  %1342 = zext i1 %1341 to i32
  %1343 = zext i1 %1340 to i32
  %1344 = icmp sgt i32 %1294, 1
  %1345 = select i1 %1344, i32 %1342, i32 0
  %1346 = select i1 %1344, i32 %1343, i32 0
  %1347 = add i32 %283, %1345
  %1348 = icmp slt i32 %1347, %231
  %1349 = select i1 %1348, i32 %1347, i32 %1274
  %1350 = call i32 @llvm.smax.i32(i32 %1349, i32 0)
  %1351 = add i32 %1277, %1346
  %1352 = mul i32 %1350, 32
  %1353 = add i32 %1295, %1352
  %1354 = sext i32 %1353 to i64
  %1355 = mul i64 %1354, %1298
  %1356 = sext i32 %1351 to i64
  %1357 = add i64 %1355, %1356
  %1358 = mul i64 %1357, 128
  %1359 = sub i32 %17, %1352
  %1360 = getelementptr bfloat, ptr addrspace(1) %6, i64 %1358
  %1361 = ptrtoint ptr addrspace(1) %1360 to i64
  %1362 = trunc i64 %1361 to i32
  %1363 = lshr i64 %1361, 32
  %1364 = trunc i64 %1363 to i32
  %1365 = or i32 %1364, -2147483648
  %1366 = insertelement <4 x i32> <i32 1, i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 17408), i32 poison, i32 poison>, i32 %1362, i64 2
  %1367 = insertelement <4 x i32> %1366, i32 %1365, i64 3
  %1368 = call i32 @llvm.smax.i32(i32 %1359, i32 0)
  %1369 = and i32 %1368, 65535
  %1370 = shl i32 %1369, 16
  %1371 = or i32 %1370, 32767
  %1372 = lshr i32 %1368, 16
  %1373 = and i32 %1372, 65535
  %1374 = or i32 %1373, 8388608
  %1375 = insertelement <8 x i32> <i32 122748928, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %1371, i64 2
  %1376 = insertelement <8 x i32> %1375, i32 %1374, i64 3
  %1377 = insertelement <8 x i32> %1376, i32 32, i64 4
  %1378 = insertelement <8 x i32> %1377, i32 %1322, i64 5
  %1379 = insertelement <8 x i32> %1378, i32 %1325, i64 6
  %1380 = insertelement <8 x i32> %1379, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %1367, <8 x i32> %1380, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %1381 = getelementptr bfloat, ptr addrspace(1) %0, i64 %1358
  %1382 = ptrtoint ptr addrspace(1) %1381 to i64
  %1383 = trunc i64 %1382 to i32
  %1384 = lshr i64 %1382, 32
  %1385 = trunc i64 %1384 to i32
  %1386 = or i32 %1385, -2147483648
  %1387 = insertelement <4 x i32> <i32 1, i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 26112), i32 poison, i32 poison>, i32 %1383, i64 2
  %1388 = insertelement <4 x i32> %1387, i32 %1386, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %1388, <8 x i32> %1380, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1389 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %223
  %1390 = add i32 %1389, 8704
  %1391 = inttoptr i32 %1390 to ptr addrspace(3)
  %1392 = load <8 x bfloat>, ptr addrspace(3) %1391, align 16
  %1393 = add i32 %1389, 8736
  %1394 = inttoptr i32 %1393 to ptr addrspace(3)
  %1395 = load <8 x bfloat>, ptr addrspace(3) %1394, align 16
  %1396 = add i32 %1389, 8768
  %1397 = inttoptr i32 %1396 to ptr addrspace(3)
  %1398 = load <8 x bfloat>, ptr addrspace(3) %1397, align 16
  %1399 = add i32 %1389, 8800
  %1400 = inttoptr i32 %1399 to ptr addrspace(3)
  %1401 = load <8 x bfloat>, ptr addrspace(3) %1400, align 16
  %1402 = add i32 %1389, 8832
  %1403 = inttoptr i32 %1402 to ptr addrspace(3)
  %1404 = load <8 x bfloat>, ptr addrspace(3) %1403, align 16
  %1405 = add i32 %1389, 8864
  %1406 = inttoptr i32 %1405 to ptr addrspace(3)
  %1407 = load <8 x bfloat>, ptr addrspace(3) %1406, align 16
  %1408 = add i32 %1389, 8896
  %1409 = inttoptr i32 %1408 to ptr addrspace(3)
  %1410 = load <8 x bfloat>, ptr addrspace(3) %1409, align 16
  %1411 = add i32 %1389, 8928
  %1412 = inttoptr i32 %1411 to ptr addrspace(3)
  %1413 = load <8 x bfloat>, ptr addrspace(3) %1412, align 16
  %1414 = inttoptr i32 %1389 to ptr addrspace(3)
  %1415 = load <8 x bfloat>, ptr addrspace(3) %1414, align 16
  %1416 = add i32 %1389, 32
  %1417 = inttoptr i32 %1416 to ptr addrspace(3)
  %1418 = load <8 x bfloat>, ptr addrspace(3) %1417, align 16
  %1419 = add i32 %1389, 64
  %1420 = inttoptr i32 %1419 to ptr addrspace(3)
  %1421 = load <8 x bfloat>, ptr addrspace(3) %1420, align 16
  %1422 = add i32 %1389, 96
  %1423 = inttoptr i32 %1422 to ptr addrspace(3)
  %1424 = load <8 x bfloat>, ptr addrspace(3) %1423, align 16
  %1425 = add i32 %1389, 128
  %1426 = inttoptr i32 %1425 to ptr addrspace(3)
  %1427 = load <8 x bfloat>, ptr addrspace(3) %1426, align 16
  %1428 = add i32 %1389, 160
  %1429 = inttoptr i32 %1428 to ptr addrspace(3)
  %1430 = load <8 x bfloat>, ptr addrspace(3) %1429, align 16
  %1431 = add i32 %1389, 192
  %1432 = inttoptr i32 %1431 to ptr addrspace(3)
  %1433 = load <8 x bfloat>, ptr addrspace(3) %1432, align 16
  %1434 = add i32 %1389, 224
  %1435 = inttoptr i32 %1434 to ptr addrspace(3)
  %1436 = load <8 x bfloat>, ptr addrspace(3) %1435, align 16
  %1437 = add i32 %1389, 13056
  %1438 = inttoptr i32 %1437 to ptr addrspace(3)
  %1439 = load <8 x bfloat>, ptr addrspace(3) %1438, align 16
  %1440 = add i32 %1389, 13088
  %1441 = inttoptr i32 %1440 to ptr addrspace(3)
  %1442 = load <8 x bfloat>, ptr addrspace(3) %1441, align 16
  %1443 = add i32 %1389, 13120
  %1444 = inttoptr i32 %1443 to ptr addrspace(3)
  %1445 = load <8 x bfloat>, ptr addrspace(3) %1444, align 16
  %1446 = add i32 %1389, 13152
  %1447 = inttoptr i32 %1446 to ptr addrspace(3)
  %1448 = load <8 x bfloat>, ptr addrspace(3) %1447, align 16
  %1449 = add i32 %1389, 13184
  %1450 = inttoptr i32 %1449 to ptr addrspace(3)
  %1451 = load <8 x bfloat>, ptr addrspace(3) %1450, align 16
  %1452 = add i32 %1389, 13216
  %1453 = inttoptr i32 %1452 to ptr addrspace(3)
  %1454 = load <8 x bfloat>, ptr addrspace(3) %1453, align 16
  %1455 = add i32 %1389, 13248
  %1456 = inttoptr i32 %1455 to ptr addrspace(3)
  %1457 = load <8 x bfloat>, ptr addrspace(3) %1456, align 16
  %1458 = add i32 %1389, 13280
  %1459 = inttoptr i32 %1458 to ptr addrspace(3)
  %1460 = load <8 x bfloat>, ptr addrspace(3) %1459, align 16
  %1461 = add i32 %1389, 4352
  %1462 = inttoptr i32 %1461 to ptr addrspace(3)
  %1463 = load <8 x bfloat>, ptr addrspace(3) %1462, align 16
  %1464 = add i32 %1389, 4384
  %1465 = inttoptr i32 %1464 to ptr addrspace(3)
  %1466 = load <8 x bfloat>, ptr addrspace(3) %1465, align 16
  %1467 = add i32 %1389, 4416
  %1468 = inttoptr i32 %1467 to ptr addrspace(3)
  %1469 = load <8 x bfloat>, ptr addrspace(3) %1468, align 16
  %1470 = add i32 %1389, 4448
  %1471 = inttoptr i32 %1470 to ptr addrspace(3)
  %1472 = load <8 x bfloat>, ptr addrspace(3) %1471, align 16
  %1473 = add i32 %1389, 4480
  %1474 = inttoptr i32 %1473 to ptr addrspace(3)
  %1475 = load <8 x bfloat>, ptr addrspace(3) %1474, align 16
  %1476 = add i32 %1389, 4512
  %1477 = inttoptr i32 %1476 to ptr addrspace(3)
  %1478 = load <8 x bfloat>, ptr addrspace(3) %1477, align 16
  %1479 = add i32 %1389, 4544
  %1480 = inttoptr i32 %1479 to ptr addrspace(3)
  %1481 = load <8 x bfloat>, ptr addrspace(3) %1480, align 16
  %1482 = add i32 %1389, 4576
  %1483 = inttoptr i32 %1482 to ptr addrspace(3)
  %1484 = load <8 x bfloat>, ptr addrspace(3) %1483, align 16
  %1485 = sext i32 %1294 to i64
  br label %1486

1486:                                             ; preds = %1560, %1272
  %1487 = phi i64 [ %2417, %1560 ], [ 0, %1272 ]
  %1488 = phi <8 x float> [ %2385, %1560 ], [ %288, %1272 ]
  %1489 = phi <8 x float> [ %2386, %1560 ], [ %289, %1272 ]
  %1490 = phi <8 x float> [ %2387, %1560 ], [ %290, %1272 ]
  %1491 = phi <8 x float> [ %2388, %1560 ], [ %291, %1272 ]
  %1492 = phi <8 x float> [ %2389, %1560 ], [ %292, %1272 ]
  %1493 = phi <8 x float> [ %2390, %1560 ], [ %293, %1272 ]
  %1494 = phi <8 x float> [ %2391, %1560 ], [ %294, %1272 ]
  %1495 = phi <8 x float> [ %2392, %1560 ], [ %295, %1272 ]
  %1496 = phi <8 x float> [ %2401, %1560 ], [ %296, %1272 ]
  %1497 = phi <8 x float> [ %2402, %1560 ], [ %297, %1272 ]
  %1498 = phi <8 x float> [ %2403, %1560 ], [ %298, %1272 ]
  %1499 = phi <8 x float> [ %2404, %1560 ], [ %299, %1272 ]
  %1500 = phi <8 x float> [ %2405, %1560 ], [ %300, %1272 ]
  %1501 = phi <8 x float> [ %2406, %1560 ], [ %301, %1272 ]
  %1502 = phi <8 x float> [ %2407, %1560 ], [ %302, %1272 ]
  %1503 = phi <8 x float> [ %2408, %1560 ], [ %303, %1272 ]
  %1504 = phi <8 x float> [ %2393, %1560 ], [ %304, %1272 ]
  %1505 = phi <8 x float> [ %2394, %1560 ], [ %305, %1272 ]
  %1506 = phi <8 x float> [ %2395, %1560 ], [ %306, %1272 ]
  %1507 = phi <8 x float> [ %2396, %1560 ], [ %307, %1272 ]
  %1508 = phi <8 x float> [ %2397, %1560 ], [ %308, %1272 ]
  %1509 = phi <8 x float> [ %2398, %1560 ], [ %309, %1272 ]
  %1510 = phi <8 x float> [ %2399, %1560 ], [ %310, %1272 ]
  %1511 = phi <8 x float> [ %2400, %1560 ], [ %311, %1272 ]
  %1512 = phi <8 x float> [ %2409, %1560 ], [ %312, %1272 ]
  %1513 = phi <8 x float> [ %2410, %1560 ], [ %313, %1272 ]
  %1514 = phi <8 x float> [ %2411, %1560 ], [ %314, %1272 ]
  %1515 = phi <8 x float> [ %2412, %1560 ], [ %315, %1272 ]
  %1516 = phi <8 x float> [ %2413, %1560 ], [ %316, %1272 ]
  %1517 = phi <8 x float> [ %2414, %1560 ], [ %317, %1272 ]
  %1518 = phi <8 x float> [ %2415, %1560 ], [ %318, %1272 ]
  %1519 = phi <8 x float> [ %2416, %1560 ], [ %319, %1272 ]
  %1520 = phi <1 x float> [ %1642, %1560 ], [ %1286, %1272 ]
  %1521 = phi <1 x float> [ %1644, %1560 ], [ %1288, %1272 ]
  %1522 = phi <1 x float> [ %1647, %1560 ], [ %1291, %1272 ]
  %1523 = phi <1 x float> [ %1649, %1560 ], [ %1293, %1272 ]
  %1524 = phi <8 x bfloat> [ %2292, %1560 ], [ %1392, %1272 ]
  %1525 = phi <8 x bfloat> [ %2295, %1560 ], [ %1395, %1272 ]
  %1526 = phi <8 x bfloat> [ %2298, %1560 ], [ %1398, %1272 ]
  %1527 = phi <8 x bfloat> [ %2301, %1560 ], [ %1401, %1272 ]
  %1528 = phi <8 x bfloat> [ %2304, %1560 ], [ %1404, %1272 ]
  %1529 = phi <8 x bfloat> [ %2307, %1560 ], [ %1407, %1272 ]
  %1530 = phi <8 x bfloat> [ %2310, %1560 ], [ %1410, %1272 ]
  %1531 = phi <8 x bfloat> [ %2313, %1560 ], [ %1413, %1272 ]
  %1532 = phi <8 x bfloat> [ %2315, %1560 ], [ %1415, %1272 ]
  %1533 = phi <8 x bfloat> [ %2318, %1560 ], [ %1418, %1272 ]
  %1534 = phi <8 x bfloat> [ %2321, %1560 ], [ %1421, %1272 ]
  %1535 = phi <8 x bfloat> [ %2324, %1560 ], [ %1424, %1272 ]
  %1536 = phi <8 x bfloat> [ %2327, %1560 ], [ %1427, %1272 ]
  %1537 = phi <8 x bfloat> [ %2330, %1560 ], [ %1430, %1272 ]
  %1538 = phi <8 x bfloat> [ %2333, %1560 ], [ %1433, %1272 ]
  %1539 = phi <8 x bfloat> [ %2336, %1560 ], [ %1436, %1272 ]
  %1540 = phi <8 x bfloat> [ %2339, %1560 ], [ %1439, %1272 ]
  %1541 = phi <8 x bfloat> [ %2342, %1560 ], [ %1442, %1272 ]
  %1542 = phi <8 x bfloat> [ %2345, %1560 ], [ %1445, %1272 ]
  %1543 = phi <8 x bfloat> [ %2348, %1560 ], [ %1448, %1272 ]
  %1544 = phi <8 x bfloat> [ %2351, %1560 ], [ %1451, %1272 ]
  %1545 = phi <8 x bfloat> [ %2354, %1560 ], [ %1454, %1272 ]
  %1546 = phi <8 x bfloat> [ %2357, %1560 ], [ %1457, %1272 ]
  %1547 = phi <8 x bfloat> [ %2360, %1560 ], [ %1460, %1272 ]
  %1548 = phi <8 x bfloat> [ %2363, %1560 ], [ %1463, %1272 ]
  %1549 = phi <8 x bfloat> [ %2366, %1560 ], [ %1466, %1272 ]
  %1550 = phi <8 x bfloat> [ %2369, %1560 ], [ %1469, %1272 ]
  %1551 = phi <8 x bfloat> [ %2372, %1560 ], [ %1472, %1272 ]
  %1552 = phi <8 x bfloat> [ %2375, %1560 ], [ %1475, %1272 ]
  %1553 = phi <8 x bfloat> [ %2378, %1560 ], [ %1478, %1272 ]
  %1554 = phi <8 x bfloat> [ %2381, %1560 ], [ %1481, %1272 ]
  %1555 = phi <8 x bfloat> [ %2384, %1560 ], [ %1484, %1272 ]
  %1556 = phi i32 [ %1586, %1560 ], [ 0, %1272 ]
  %1557 = phi i32 [ %1566, %1560 ], [ 0, %1272 ]
  %1558 = phi i32 [ %1567, %1560 ], [ 0, %1272 ]
  %1559 = icmp slt i64 %1487, %1485
  br i1 %1559, label %1560, label %2418

1560:                                             ; preds = %1486
  %1561 = trunc i64 %1487 to i32
  %1562 = add i32 %1561, 1
  %1563 = add i32 %1558, 1
  %1564 = icmp slt i32 %1563, %21
  %1565 = add i32 %1557, 1
  %1566 = select i1 %1564, i32 %1557, i32 %1565
  %1567 = select i1 %1564, i32 %1563, i32 0
  %1568 = icmp slt i32 %1562, %1294
  %1569 = select i1 %1568, i32 %1566, i32 %1557
  %1570 = select i1 %1568, i32 %1567, i32 %1558
  %1571 = add i32 %1567, 1
  %1572 = icmp slt i32 %1571, %21
  %1573 = add i32 %1566, 1
  %1574 = select i1 %1572, i32 %1566, i32 %1573
  %1575 = select i1 %1572, i32 %1571, i32 0
  %1576 = add i32 %1561, 2
  %1577 = icmp slt i32 %1576, %1294
  %1578 = select i1 %1577, i32 %1574, i32 %1569
  %1579 = add i32 %283, %1578
  %1580 = select i1 %1577, i32 %1575, i32 %1570
  %1581 = icmp eq i32 %1556, 0
  %1582 = sub i32 %1556, 17408
  %1583 = select i1 %1581, i32 34816, i32 %1582
  %1584 = icmp eq i32 %1556, 34816
  %1585 = add i32 %1556, 17408
  %1586 = select i1 %1584, i32 0, i32 %1585
  %1587 = add i32 %283, %1569
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1588 = add i32 %1277, %1580
  %1589 = mul i32 %1579, 32
  %1590 = add i32 %1295, %1589
  %1591 = sext i32 %1590 to i64
  %1592 = mul i64 %1591, %1298
  %1593 = sext i32 %1588 to i64
  %1594 = add i64 %1592, %1593
  %1595 = mul i64 %1594, 128
  %1596 = sub i32 %17, %1589
  %1597 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1583
  %1598 = add i32 %1597, 8704
  %1599 = getelementptr bfloat, ptr addrspace(1) %6, i64 %1595
  %1600 = inttoptr i32 %1597 to ptr addrspace(3)
  %1601 = ptrtoint ptr addrspace(1) %1599 to i64
  %1602 = ptrtoint ptr addrspace(3) %1600 to i32
  %1603 = trunc i64 %1601 to i32
  %1604 = lshr i64 %1601, 32
  %1605 = trunc i64 %1604 to i32
  %1606 = or i32 %1605, -2147483648
  %1607 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %1602, i64 1
  %1608 = insertelement <4 x i32> %1607, i32 %1603, i64 2
  %1609 = insertelement <4 x i32> %1608, i32 %1606, i64 3
  %1610 = call i32 @llvm.smax.i32(i32 %1596, i32 0)
  %1611 = and i32 %1610, 65535
  %1612 = shl i32 %1611, 16
  %1613 = or i32 %1612, 32767
  %1614 = lshr i32 %1610, 16
  %1615 = and i32 %1614, 65535
  %1616 = or i32 %1615, 8388608
  %1617 = insertelement <8 x i32> <i32 122748928, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %1613, i64 2
  %1618 = insertelement <8 x i32> %1617, i32 %1616, i64 3
  %1619 = insertelement <8 x i32> %1618, i32 32, i64 4
  %1620 = insertelement <8 x i32> %1619, i32 %1322, i64 5
  %1621 = insertelement <8 x i32> %1620, i32 %1325, i64 6
  %1622 = insertelement <8 x i32> %1621, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %1609, <8 x i32> %1622, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %1623 = getelementptr bfloat, ptr addrspace(1) %0, i64 %1595
  %1624 = inttoptr i32 %1598 to ptr addrspace(3)
  %1625 = ptrtoint ptr addrspace(1) %1623 to i64
  %1626 = ptrtoint ptr addrspace(3) %1624 to i32
  %1627 = trunc i64 %1625 to i32
  %1628 = lshr i64 %1625, 32
  %1629 = trunc i64 %1628 to i32
  %1630 = or i32 %1629, -2147483648
  %1631 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %1626, i64 1
  %1632 = insertelement <4 x i32> %1631, i32 %1627, i64 2
  %1633 = insertelement <4 x i32> %1632, i32 %1630, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %1633, <8 x i32> %1622, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1634 = add i32 %1277, %1570
  %1635 = mul i32 %1587, 32
  %1636 = add i32 %1279, %1634
  %1637 = mul i32 %1636, %17
  %1638 = add i32 %1637, %1635
  %1639 = mul i32 %1638, 4
  %1640 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1639)
  %1641 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %71, i32 %85, i32 %1640, i32 0)
  %1642 = insertelement <1 x float> poison, float %1641, i32 0
  %1643 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %72, i32 %85, i32 %1640, i32 0)
  %1644 = insertelement <1 x float> poison, float %1643, i32 0
  %1645 = add i32 %1640, 64
  %1646 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %71, i32 %85, i32 %1645, i32 0)
  %1647 = insertelement <1 x float> poison, float %1646, i32 0
  %1648 = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %72, i32 %85, i32 %1645, i32 0)
  %1649 = insertelement <1 x float> poison, float %1648, i32 0
  %1650 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1556
  %1651 = shufflevector <8 x bfloat> %1524, <8 x bfloat> %1525, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1652 = shufflevector <8 x bfloat> %1526, <8 x bfloat> %1527, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1653 = shufflevector <8 x bfloat> %1528, <8 x bfloat> %1529, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1654 = shufflevector <8 x bfloat> %1530, <8 x bfloat> %1531, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1655 = shufflevector <8 x bfloat> %1532, <8 x bfloat> %1533, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1656 = shufflevector <8 x bfloat> %1534, <8 x bfloat> %1535, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1657 = shufflevector <8 x bfloat> %1536, <8 x bfloat> %1537, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1658 = shufflevector <8 x bfloat> %1538, <8 x bfloat> %1539, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1659 = extractelement <1 x float> %1520, i64 0
  %1660 = extractelement <1 x float> %1521, i64 0
  %1661 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %98, <16 x bfloat> %1651, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1662 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %170, <16 x bfloat> %1655, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1663 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %107, <16 x bfloat> %1652, i16 0, <8 x float> %1661, i1 false, i1 false)
  %1664 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %175, <16 x bfloat> %1656, i16 0, <8 x float> %1662, i1 false, i1 false)
  %1665 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %116, <16 x bfloat> %1653, i16 0, <8 x float> %1663, i1 false, i1 false)
  %1666 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %180, <16 x bfloat> %1657, i16 0, <8 x float> %1664, i1 false, i1 false)
  %1667 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %125, <16 x bfloat> %1654, i16 0, <8 x float> %1665, i1 false, i1 false)
  %1668 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %185, <16 x bfloat> %1658, i16 0, <8 x float> %1666, i1 false, i1 false)
  %1669 = extractelement <8 x float> %1667, i64 0
  %1670 = fmul float %1669, %16
  %1671 = extractelement <8 x float> %1667, i64 1
  %1672 = fmul float %1671, %16
  %1673 = extractelement <8 x float> %1667, i64 2
  %1674 = fmul float %1673, %16
  %1675 = extractelement <8 x float> %1667, i64 3
  %1676 = fmul float %1675, %16
  %1677 = extractelement <8 x float> %1667, i64 4
  %1678 = fmul float %1677, %16
  %1679 = extractelement <8 x float> %1667, i64 5
  %1680 = fmul float %1679, %16
  %1681 = extractelement <8 x float> %1667, i64 6
  %1682 = fmul float %1681, %16
  %1683 = extractelement <8 x float> %1667, i64 7
  %1684 = fmul float %1683, %16
  %1685 = fsub float %1670, %1659
  %1686 = fmul float %1685, f0x3FB8AA3B
  %1687 = call float @llvm.amdgcn.exp2.f32(float %1686)
  %1688 = fsub float %1672, %1659
  %1689 = fmul float %1688, f0x3FB8AA3B
  %1690 = call float @llvm.amdgcn.exp2.f32(float %1689)
  %1691 = fsub float %1674, %1659
  %1692 = fmul float %1691, f0x3FB8AA3B
  %1693 = call float @llvm.amdgcn.exp2.f32(float %1692)
  %1694 = fsub float %1676, %1659
  %1695 = fmul float %1694, f0x3FB8AA3B
  %1696 = call float @llvm.amdgcn.exp2.f32(float %1695)
  %1697 = fsub float %1678, %1659
  %1698 = fmul float %1697, f0x3FB8AA3B
  %1699 = call float @llvm.amdgcn.exp2.f32(float %1698)
  %1700 = fsub float %1680, %1659
  %1701 = fmul float %1700, f0x3FB8AA3B
  %1702 = call float @llvm.amdgcn.exp2.f32(float %1701)
  %1703 = fsub float %1682, %1659
  %1704 = fmul float %1703, f0x3FB8AA3B
  %1705 = call float @llvm.amdgcn.exp2.f32(float %1704)
  %1706 = fsub float %1684, %1659
  %1707 = fmul float %1706, f0x3FB8AA3B
  %1708 = call float @llvm.amdgcn.exp2.f32(float %1707)
  %1709 = fptrunc float %1687 to bfloat
  %1710 = fptrunc float %1690 to bfloat
  %1711 = fptrunc float %1693 to bfloat
  %1712 = fptrunc float %1696 to bfloat
  %1713 = fptrunc float %1699 to bfloat
  %1714 = fptrunc float %1702 to bfloat
  %1715 = fptrunc float %1705 to bfloat
  %1716 = fptrunc float %1708 to bfloat
  %1717 = extractelement <8 x float> %1668, i64 0
  %1718 = fsub float %1717, %1660
  %1719 = fmul float %1687, %1718
  %1720 = fmul float %1719, %16
  %1721 = fptrunc float %1720 to bfloat
  %1722 = extractelement <8 x float> %1668, i64 1
  %1723 = fsub float %1722, %1660
  %1724 = fmul float %1690, %1723
  %1725 = fmul float %1724, %16
  %1726 = fptrunc float %1725 to bfloat
  %1727 = extractelement <8 x float> %1668, i64 2
  %1728 = fsub float %1727, %1660
  %1729 = fmul float %1693, %1728
  %1730 = fmul float %1729, %16
  %1731 = fptrunc float %1730 to bfloat
  %1732 = extractelement <8 x float> %1668, i64 3
  %1733 = fsub float %1732, %1660
  %1734 = fmul float %1696, %1733
  %1735 = fmul float %1734, %16
  %1736 = fptrunc float %1735 to bfloat
  %1737 = extractelement <8 x float> %1668, i64 4
  %1738 = fsub float %1737, %1660
  %1739 = fmul float %1699, %1738
  %1740 = fmul float %1739, %16
  %1741 = fptrunc float %1740 to bfloat
  %1742 = extractelement <8 x float> %1668, i64 5
  %1743 = fsub float %1742, %1660
  %1744 = fmul float %1702, %1743
  %1745 = fmul float %1744, %16
  %1746 = fptrunc float %1745 to bfloat
  %1747 = extractelement <8 x float> %1668, i64 6
  %1748 = fsub float %1747, %1660
  %1749 = fmul float %1705, %1748
  %1750 = fmul float %1749, %16
  %1751 = fptrunc float %1750 to bfloat
  %1752 = extractelement <8 x float> %1668, i64 7
  %1753 = fsub float %1752, %1660
  %1754 = fmul float %1708, %1753
  %1755 = fmul float %1754, %16
  %1756 = fptrunc float %1755 to bfloat
  %1757 = mul i32 %51, 80
  %1758 = add i32 %1757, %222
  %1759 = insertelement <8 x bfloat> poison, bfloat %1709, i64 0
  %1760 = insertelement <8 x bfloat> %1759, bfloat %1710, i64 1
  %1761 = insertelement <8 x bfloat> %1760, bfloat %1711, i64 2
  %1762 = insertelement <8 x bfloat> %1761, bfloat %1712, i64 3
  %1763 = insertelement <8 x bfloat> %1762, bfloat %1713, i64 4
  %1764 = insertelement <8 x bfloat> %1763, bfloat %1714, i64 5
  %1765 = insertelement <8 x bfloat> %1764, bfloat %1715, i64 6
  %1766 = insertelement <8 x bfloat> %1765, bfloat %1716, i64 7
  %1767 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %1758
  %1768 = insertelement <8 x bfloat> poison, bfloat %1721, i64 0
  %1769 = insertelement <8 x bfloat> %1768, bfloat %1726, i64 1
  %1770 = insertelement <8 x bfloat> %1769, bfloat %1731, i64 2
  %1771 = insertelement <8 x bfloat> %1770, bfloat %1736, i64 3
  %1772 = insertelement <8 x bfloat> %1771, bfloat %1741, i64 4
  %1773 = insertelement <8 x bfloat> %1772, bfloat %1746, i64 5
  %1774 = insertelement <8 x bfloat> %1773, bfloat %1751, i64 6
  %1775 = insertelement <8 x bfloat> %1774, bfloat %1756, i64 7
  %1776 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %1758
  %1777 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %138, <16 x bfloat> %1651, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1778 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %190, <16 x bfloat> %1655, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1779 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %147, <16 x bfloat> %1652, i16 0, <8 x float> %1777, i1 false, i1 false)
  %1780 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %195, <16 x bfloat> %1656, i16 0, <8 x float> %1778, i1 false, i1 false)
  %1781 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %156, <16 x bfloat> %1653, i16 0, <8 x float> %1779, i1 false, i1 false)
  %1782 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %200, <16 x bfloat> %1657, i16 0, <8 x float> %1780, i1 false, i1 false)
  %1783 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %165, <16 x bfloat> %1654, i16 0, <8 x float> %1781, i1 false, i1 false)
  %1784 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %205, <16 x bfloat> %1658, i16 0, <8 x float> %1782, i1 false, i1 false)
  %1785 = extractelement <8 x float> %1783, i64 0
  %1786 = fmul float %1785, %16
  %1787 = extractelement <8 x float> %1783, i64 1
  %1788 = fmul float %1787, %16
  %1789 = extractelement <8 x float> %1783, i64 2
  %1790 = fmul float %1789, %16
  %1791 = extractelement <8 x float> %1783, i64 3
  %1792 = fmul float %1791, %16
  %1793 = extractelement <8 x float> %1783, i64 4
  %1794 = fmul float %1793, %16
  %1795 = extractelement <8 x float> %1783, i64 5
  %1796 = fmul float %1795, %16
  %1797 = extractelement <8 x float> %1783, i64 6
  %1798 = fmul float %1797, %16
  %1799 = extractelement <8 x float> %1783, i64 7
  %1800 = fmul float %1799, %16
  %1801 = fsub float %1786, %1659
  %1802 = fmul float %1801, f0x3FB8AA3B
  %1803 = call float @llvm.amdgcn.exp2.f32(float %1802)
  %1804 = fsub float %1788, %1659
  %1805 = fmul float %1804, f0x3FB8AA3B
  %1806 = call float @llvm.amdgcn.exp2.f32(float %1805)
  %1807 = fsub float %1790, %1659
  %1808 = fmul float %1807, f0x3FB8AA3B
  %1809 = call float @llvm.amdgcn.exp2.f32(float %1808)
  %1810 = fsub float %1792, %1659
  %1811 = fmul float %1810, f0x3FB8AA3B
  %1812 = call float @llvm.amdgcn.exp2.f32(float %1811)
  %1813 = fsub float %1794, %1659
  %1814 = fmul float %1813, f0x3FB8AA3B
  %1815 = call float @llvm.amdgcn.exp2.f32(float %1814)
  %1816 = fsub float %1796, %1659
  %1817 = fmul float %1816, f0x3FB8AA3B
  %1818 = call float @llvm.amdgcn.exp2.f32(float %1817)
  %1819 = fsub float %1798, %1659
  %1820 = fmul float %1819, f0x3FB8AA3B
  %1821 = call float @llvm.amdgcn.exp2.f32(float %1820)
  %1822 = fsub float %1800, %1659
  %1823 = fmul float %1822, f0x3FB8AA3B
  %1824 = call float @llvm.amdgcn.exp2.f32(float %1823)
  %1825 = fptrunc float %1803 to bfloat
  %1826 = fptrunc float %1806 to bfloat
  %1827 = fptrunc float %1809 to bfloat
  %1828 = fptrunc float %1812 to bfloat
  %1829 = fptrunc float %1815 to bfloat
  %1830 = fptrunc float %1818 to bfloat
  %1831 = fptrunc float %1821 to bfloat
  %1832 = fptrunc float %1824 to bfloat
  %1833 = extractelement <8 x float> %1784, i64 0
  %1834 = fsub float %1833, %1660
  %1835 = fmul float %1803, %1834
  %1836 = fmul float %1835, %16
  %1837 = fptrunc float %1836 to bfloat
  %1838 = extractelement <8 x float> %1784, i64 1
  %1839 = fsub float %1838, %1660
  %1840 = fmul float %1806, %1839
  %1841 = fmul float %1840, %16
  %1842 = fptrunc float %1841 to bfloat
  %1843 = extractelement <8 x float> %1784, i64 2
  %1844 = fsub float %1843, %1660
  %1845 = fmul float %1809, %1844
  %1846 = fmul float %1845, %16
  %1847 = fptrunc float %1846 to bfloat
  %1848 = extractelement <8 x float> %1784, i64 3
  %1849 = fsub float %1848, %1660
  %1850 = fmul float %1812, %1849
  %1851 = fmul float %1850, %16
  %1852 = fptrunc float %1851 to bfloat
  %1853 = extractelement <8 x float> %1784, i64 4
  %1854 = fsub float %1853, %1660
  %1855 = fmul float %1815, %1854
  %1856 = fmul float %1855, %16
  %1857 = fptrunc float %1856 to bfloat
  %1858 = extractelement <8 x float> %1784, i64 5
  %1859 = fsub float %1858, %1660
  %1860 = fmul float %1818, %1859
  %1861 = fmul float %1860, %16
  %1862 = fptrunc float %1861 to bfloat
  %1863 = extractelement <8 x float> %1784, i64 6
  %1864 = fsub float %1863, %1660
  %1865 = fmul float %1821, %1864
  %1866 = fmul float %1865, %16
  %1867 = fptrunc float %1866 to bfloat
  %1868 = extractelement <8 x float> %1784, i64 7
  %1869 = fsub float %1868, %1660
  %1870 = fmul float %1824, %1869
  %1871 = fmul float %1870, %16
  %1872 = fptrunc float %1871 to bfloat
  %1873 = add i32 %1757, 32
  %1874 = add i32 %1873, %222
  %1875 = insertelement <8 x bfloat> poison, bfloat %1825, i64 0
  %1876 = insertelement <8 x bfloat> %1875, bfloat %1826, i64 1
  %1877 = insertelement <8 x bfloat> %1876, bfloat %1827, i64 2
  %1878 = insertelement <8 x bfloat> %1877, bfloat %1828, i64 3
  %1879 = insertelement <8 x bfloat> %1878, bfloat %1829, i64 4
  %1880 = insertelement <8 x bfloat> %1879, bfloat %1830, i64 5
  %1881 = insertelement <8 x bfloat> %1880, bfloat %1831, i64 6
  %1882 = insertelement <8 x bfloat> %1881, bfloat %1832, i64 7
  %1883 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %1874
  %1884 = insertelement <8 x bfloat> poison, bfloat %1837, i64 0
  %1885 = insertelement <8 x bfloat> %1884, bfloat %1842, i64 1
  %1886 = insertelement <8 x bfloat> %1885, bfloat %1847, i64 2
  %1887 = insertelement <8 x bfloat> %1886, bfloat %1852, i64 3
  %1888 = insertelement <8 x bfloat> %1887, bfloat %1857, i64 4
  %1889 = insertelement <8 x bfloat> %1888, bfloat %1862, i64 5
  %1890 = insertelement <8 x bfloat> %1889, bfloat %1867, i64 6
  %1891 = insertelement <8 x bfloat> %1890, bfloat %1872, i64 7
  %1892 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %1874
  %1893 = shufflevector <8 x bfloat> %1540, <8 x bfloat> %1541, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1894 = shufflevector <8 x bfloat> %1542, <8 x bfloat> %1543, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1895 = shufflevector <8 x bfloat> %1544, <8 x bfloat> %1545, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1896 = shufflevector <8 x bfloat> %1546, <8 x bfloat> %1547, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1897 = shufflevector <8 x bfloat> %1548, <8 x bfloat> %1549, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1898 = shufflevector <8 x bfloat> %1550, <8 x bfloat> %1551, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1899 = shufflevector <8 x bfloat> %1552, <8 x bfloat> %1553, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1900 = shufflevector <8 x bfloat> %1554, <8 x bfloat> %1555, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1901 = extractelement <1 x float> %1522, i64 0
  %1902 = extractelement <1 x float> %1523, i64 0
  %1903 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %98, <16 x bfloat> %1893, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1904 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %170, <16 x bfloat> %1897, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1905 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %107, <16 x bfloat> %1894, i16 0, <8 x float> %1903, i1 false, i1 false)
  %1906 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %175, <16 x bfloat> %1898, i16 0, <8 x float> %1904, i1 false, i1 false)
  %1907 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %116, <16 x bfloat> %1895, i16 0, <8 x float> %1905, i1 false, i1 false)
  %1908 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %180, <16 x bfloat> %1899, i16 0, <8 x float> %1906, i1 false, i1 false)
  %1909 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %125, <16 x bfloat> %1896, i16 0, <8 x float> %1907, i1 false, i1 false)
  %1910 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %185, <16 x bfloat> %1900, i16 0, <8 x float> %1908, i1 false, i1 false)
  %1911 = extractelement <8 x float> %1909, i64 0
  %1912 = fmul float %1911, %16
  %1913 = extractelement <8 x float> %1909, i64 1
  %1914 = fmul float %1913, %16
  %1915 = extractelement <8 x float> %1909, i64 2
  %1916 = fmul float %1915, %16
  %1917 = extractelement <8 x float> %1909, i64 3
  %1918 = fmul float %1917, %16
  %1919 = extractelement <8 x float> %1909, i64 4
  %1920 = fmul float %1919, %16
  %1921 = extractelement <8 x float> %1909, i64 5
  %1922 = fmul float %1921, %16
  %1923 = extractelement <8 x float> %1909, i64 6
  %1924 = fmul float %1923, %16
  %1925 = extractelement <8 x float> %1909, i64 7
  %1926 = fmul float %1925, %16
  %1927 = fsub float %1912, %1901
  %1928 = fmul float %1927, f0x3FB8AA3B
  %1929 = call float @llvm.amdgcn.exp2.f32(float %1928)
  %1930 = fsub float %1914, %1901
  %1931 = fmul float %1930, f0x3FB8AA3B
  %1932 = call float @llvm.amdgcn.exp2.f32(float %1931)
  %1933 = fsub float %1916, %1901
  %1934 = fmul float %1933, f0x3FB8AA3B
  %1935 = call float @llvm.amdgcn.exp2.f32(float %1934)
  %1936 = fsub float %1918, %1901
  %1937 = fmul float %1936, f0x3FB8AA3B
  %1938 = call float @llvm.amdgcn.exp2.f32(float %1937)
  %1939 = fsub float %1920, %1901
  %1940 = fmul float %1939, f0x3FB8AA3B
  %1941 = call float @llvm.amdgcn.exp2.f32(float %1940)
  %1942 = fsub float %1922, %1901
  %1943 = fmul float %1942, f0x3FB8AA3B
  %1944 = call float @llvm.amdgcn.exp2.f32(float %1943)
  %1945 = fsub float %1924, %1901
  %1946 = fmul float %1945, f0x3FB8AA3B
  %1947 = call float @llvm.amdgcn.exp2.f32(float %1946)
  %1948 = fsub float %1926, %1901
  %1949 = fmul float %1948, f0x3FB8AA3B
  %1950 = call float @llvm.amdgcn.exp2.f32(float %1949)
  %1951 = fptrunc float %1929 to bfloat
  %1952 = fptrunc float %1932 to bfloat
  %1953 = fptrunc float %1935 to bfloat
  %1954 = fptrunc float %1938 to bfloat
  %1955 = fptrunc float %1941 to bfloat
  %1956 = fptrunc float %1944 to bfloat
  %1957 = fptrunc float %1947 to bfloat
  %1958 = fptrunc float %1950 to bfloat
  %1959 = extractelement <8 x float> %1910, i64 0
  %1960 = fsub float %1959, %1902
  %1961 = fmul float %1929, %1960
  %1962 = fmul float %1961, %16
  %1963 = fptrunc float %1962 to bfloat
  %1964 = extractelement <8 x float> %1910, i64 1
  %1965 = fsub float %1964, %1902
  %1966 = fmul float %1932, %1965
  %1967 = fmul float %1966, %16
  %1968 = fptrunc float %1967 to bfloat
  %1969 = extractelement <8 x float> %1910, i64 2
  %1970 = fsub float %1969, %1902
  %1971 = fmul float %1935, %1970
  %1972 = fmul float %1971, %16
  %1973 = fptrunc float %1972 to bfloat
  %1974 = extractelement <8 x float> %1910, i64 3
  %1975 = fsub float %1974, %1902
  %1976 = fmul float %1938, %1975
  %1977 = fmul float %1976, %16
  %1978 = fptrunc float %1977 to bfloat
  %1979 = extractelement <8 x float> %1910, i64 4
  %1980 = fsub float %1979, %1902
  %1981 = fmul float %1941, %1980
  %1982 = fmul float %1981, %16
  %1983 = fptrunc float %1982 to bfloat
  %1984 = extractelement <8 x float> %1910, i64 5
  %1985 = fsub float %1984, %1902
  %1986 = fmul float %1944, %1985
  %1987 = fmul float %1986, %16
  %1988 = fptrunc float %1987 to bfloat
  %1989 = extractelement <8 x float> %1910, i64 6
  %1990 = fsub float %1989, %1902
  %1991 = fmul float %1947, %1990
  %1992 = fmul float %1991, %16
  %1993 = fptrunc float %1992 to bfloat
  %1994 = extractelement <8 x float> %1910, i64 7
  %1995 = fsub float %1994, %1902
  %1996 = fmul float %1950, %1995
  %1997 = fmul float %1996, %16
  %1998 = fptrunc float %1997 to bfloat
  %1999 = add i32 %51, 16
  %2000 = mul i32 %1999, 80
  %2001 = add i32 %2000, %222
  %2002 = insertelement <8 x bfloat> poison, bfloat %1951, i64 0
  %2003 = insertelement <8 x bfloat> %2002, bfloat %1952, i64 1
  %2004 = insertelement <8 x bfloat> %2003, bfloat %1953, i64 2
  %2005 = insertelement <8 x bfloat> %2004, bfloat %1954, i64 3
  %2006 = insertelement <8 x bfloat> %2005, bfloat %1955, i64 4
  %2007 = insertelement <8 x bfloat> %2006, bfloat %1956, i64 5
  %2008 = insertelement <8 x bfloat> %2007, bfloat %1957, i64 6
  %2009 = insertelement <8 x bfloat> %2008, bfloat %1958, i64 7
  %2010 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %2001
  %2011 = insertelement <8 x bfloat> poison, bfloat %1963, i64 0
  %2012 = insertelement <8 x bfloat> %2011, bfloat %1968, i64 1
  %2013 = insertelement <8 x bfloat> %2012, bfloat %1973, i64 2
  %2014 = insertelement <8 x bfloat> %2013, bfloat %1978, i64 3
  %2015 = insertelement <8 x bfloat> %2014, bfloat %1983, i64 4
  %2016 = insertelement <8 x bfloat> %2015, bfloat %1988, i64 5
  %2017 = insertelement <8 x bfloat> %2016, bfloat %1993, i64 6
  %2018 = insertelement <8 x bfloat> %2017, bfloat %1998, i64 7
  %2019 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %2001
  %2020 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %138, <16 x bfloat> %1893, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %2021 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %190, <16 x bfloat> %1897, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %2022 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %147, <16 x bfloat> %1894, i16 0, <8 x float> %2020, i1 false, i1 false)
  %2023 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %195, <16 x bfloat> %1898, i16 0, <8 x float> %2021, i1 false, i1 false)
  %2024 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %156, <16 x bfloat> %1895, i16 0, <8 x float> %2022, i1 false, i1 false)
  %2025 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %200, <16 x bfloat> %1899, i16 0, <8 x float> %2023, i1 false, i1 false)
  %2026 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %165, <16 x bfloat> %1896, i16 0, <8 x float> %2024, i1 false, i1 false)
  %2027 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %205, <16 x bfloat> %1900, i16 0, <8 x float> %2025, i1 false, i1 false)
  %2028 = extractelement <8 x float> %2026, i64 0
  %2029 = fmul float %2028, %16
  %2030 = extractelement <8 x float> %2026, i64 1
  %2031 = fmul float %2030, %16
  %2032 = extractelement <8 x float> %2026, i64 2
  %2033 = fmul float %2032, %16
  %2034 = extractelement <8 x float> %2026, i64 3
  %2035 = fmul float %2034, %16
  %2036 = extractelement <8 x float> %2026, i64 4
  %2037 = fmul float %2036, %16
  %2038 = extractelement <8 x float> %2026, i64 5
  %2039 = fmul float %2038, %16
  %2040 = extractelement <8 x float> %2026, i64 6
  %2041 = fmul float %2040, %16
  %2042 = extractelement <8 x float> %2026, i64 7
  %2043 = fmul float %2042, %16
  %2044 = fsub float %2029, %1901
  %2045 = fmul float %2044, f0x3FB8AA3B
  %2046 = call float @llvm.amdgcn.exp2.f32(float %2045)
  %2047 = fsub float %2031, %1901
  %2048 = fmul float %2047, f0x3FB8AA3B
  %2049 = call float @llvm.amdgcn.exp2.f32(float %2048)
  %2050 = fsub float %2033, %1901
  %2051 = fmul float %2050, f0x3FB8AA3B
  %2052 = call float @llvm.amdgcn.exp2.f32(float %2051)
  %2053 = fsub float %2035, %1901
  %2054 = fmul float %2053, f0x3FB8AA3B
  %2055 = call float @llvm.amdgcn.exp2.f32(float %2054)
  %2056 = fsub float %2037, %1901
  %2057 = fmul float %2056, f0x3FB8AA3B
  %2058 = call float @llvm.amdgcn.exp2.f32(float %2057)
  %2059 = fsub float %2039, %1901
  %2060 = fmul float %2059, f0x3FB8AA3B
  %2061 = call float @llvm.amdgcn.exp2.f32(float %2060)
  %2062 = fsub float %2041, %1901
  %2063 = fmul float %2062, f0x3FB8AA3B
  %2064 = call float @llvm.amdgcn.exp2.f32(float %2063)
  %2065 = fsub float %2043, %1901
  %2066 = fmul float %2065, f0x3FB8AA3B
  %2067 = call float @llvm.amdgcn.exp2.f32(float %2066)
  %2068 = fptrunc float %2046 to bfloat
  %2069 = fptrunc float %2049 to bfloat
  %2070 = fptrunc float %2052 to bfloat
  %2071 = fptrunc float %2055 to bfloat
  %2072 = fptrunc float %2058 to bfloat
  %2073 = fptrunc float %2061 to bfloat
  %2074 = fptrunc float %2064 to bfloat
  %2075 = fptrunc float %2067 to bfloat
  %2076 = extractelement <8 x float> %2027, i64 0
  %2077 = fsub float %2076, %1902
  %2078 = fmul float %2046, %2077
  %2079 = fmul float %2078, %16
  %2080 = fptrunc float %2079 to bfloat
  %2081 = extractelement <8 x float> %2027, i64 1
  %2082 = fsub float %2081, %1902
  %2083 = fmul float %2049, %2082
  %2084 = fmul float %2083, %16
  %2085 = fptrunc float %2084 to bfloat
  %2086 = extractelement <8 x float> %2027, i64 2
  %2087 = fsub float %2086, %1902
  %2088 = fmul float %2052, %2087
  %2089 = fmul float %2088, %16
  %2090 = fptrunc float %2089 to bfloat
  %2091 = extractelement <8 x float> %2027, i64 3
  %2092 = fsub float %2091, %1902
  %2093 = fmul float %2055, %2092
  %2094 = fmul float %2093, %16
  %2095 = fptrunc float %2094 to bfloat
  %2096 = extractelement <8 x float> %2027, i64 4
  %2097 = fsub float %2096, %1902
  %2098 = fmul float %2058, %2097
  %2099 = fmul float %2098, %16
  %2100 = fptrunc float %2099 to bfloat
  %2101 = extractelement <8 x float> %2027, i64 5
  %2102 = fsub float %2101, %1902
  %2103 = fmul float %2061, %2102
  %2104 = fmul float %2103, %16
  %2105 = fptrunc float %2104 to bfloat
  %2106 = extractelement <8 x float> %2027, i64 6
  %2107 = fsub float %2106, %1902
  %2108 = fmul float %2064, %2107
  %2109 = fmul float %2108, %16
  %2110 = fptrunc float %2109 to bfloat
  %2111 = extractelement <8 x float> %2027, i64 7
  %2112 = fsub float %2111, %1902
  %2113 = fmul float %2067, %2112
  %2114 = fmul float %2113, %16
  %2115 = fptrunc float %2114 to bfloat
  %2116 = add i32 %2000, 32
  %2117 = add i32 %2116, %222
  %2118 = insertelement <8 x bfloat> poison, bfloat %2068, i64 0
  %2119 = insertelement <8 x bfloat> %2118, bfloat %2069, i64 1
  %2120 = insertelement <8 x bfloat> %2119, bfloat %2070, i64 2
  %2121 = insertelement <8 x bfloat> %2120, bfloat %2071, i64 3
  %2122 = insertelement <8 x bfloat> %2121, bfloat %2072, i64 4
  %2123 = insertelement <8 x bfloat> %2122, bfloat %2073, i64 5
  %2124 = insertelement <8 x bfloat> %2123, bfloat %2074, i64 6
  %2125 = insertelement <8 x bfloat> %2124, bfloat %2075, i64 7
  %2126 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %2117
  %2127 = insertelement <8 x bfloat> poison, bfloat %2080, i64 0
  %2128 = insertelement <8 x bfloat> %2127, bfloat %2085, i64 1
  %2129 = insertelement <8 x bfloat> %2128, bfloat %2090, i64 2
  %2130 = insertelement <8 x bfloat> %2129, bfloat %2095, i64 3
  %2131 = insertelement <8 x bfloat> %2130, bfloat %2100, i64 4
  %2132 = insertelement <8 x bfloat> %2131, bfloat %2105, i64 5
  %2133 = insertelement <8 x bfloat> %2132, bfloat %2110, i64 6
  %2134 = insertelement <8 x bfloat> %2133, bfloat %2115, i64 7
  %2135 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %2117
  %2136 = add i32 %1650, %220
  %2137 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1586
  %2138 = add i32 %2137, %223
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2139 = inttoptr i32 %1767 to ptr addrspace(3)
  store <8 x bfloat> %1766, ptr addrspace(3) %2139, align 16
  %2140 = inttoptr i32 %1776 to ptr addrspace(3)
  store <8 x bfloat> %1775, ptr addrspace(3) %2140, align 16
  %2141 = inttoptr i32 %1883 to ptr addrspace(3)
  store <8 x bfloat> %1882, ptr addrspace(3) %2141, align 16
  %2142 = inttoptr i32 %1892 to ptr addrspace(3)
  store <8 x bfloat> %1891, ptr addrspace(3) %2142, align 16
  %2143 = inttoptr i32 %2010 to ptr addrspace(3)
  store <8 x bfloat> %2009, ptr addrspace(3) %2143, align 16
  %2144 = inttoptr i32 %2019 to ptr addrspace(3)
  store <8 x bfloat> %2018, ptr addrspace(3) %2144, align 16
  %2145 = inttoptr i32 %2126 to ptr addrspace(3)
  store <8 x bfloat> %2125, ptr addrspace(3) %2145, align 16
  %2146 = inttoptr i32 %2135 to ptr addrspace(3)
  store <8 x bfloat> %2134, ptr addrspace(3) %2146, align 16
  %2147 = mul i32 %208, 80
  %2148 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 65536), %2147
  %2149 = add i32 %2148, %219
  %2150 = inttoptr i32 %2149 to ptr addrspace(3)
  %2151 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2150)
  %2152 = add i32 %2149, 1280
  %2153 = inttoptr i32 %2152 to ptr addrspace(3)
  %2154 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2153)
  %2155 = shufflevector <8 x bfloat> %2151, <8 x bfloat> %2154, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2156 = inttoptr i32 %2136 to ptr addrspace(3)
  %2157 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2156)
  %2158 = add i32 %2136, 4352
  %2159 = inttoptr i32 %2158 to ptr addrspace(3)
  %2160 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2159)
  %2161 = shufflevector <8 x bfloat> %2157, <8 x bfloat> %2160, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2162 = add i32 %2136, 32
  %2163 = inttoptr i32 %2162 to ptr addrspace(3)
  %2164 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2163)
  %2165 = add i32 %2136, 4384
  %2166 = inttoptr i32 %2165 to ptr addrspace(3)
  %2167 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2166)
  %2168 = shufflevector <8 x bfloat> %2164, <8 x bfloat> %2167, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2169 = add i32 %2136, 64
  %2170 = inttoptr i32 %2169 to ptr addrspace(3)
  %2171 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2170)
  %2172 = add i32 %2136, 4416
  %2173 = inttoptr i32 %2172 to ptr addrspace(3)
  %2174 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2173)
  %2175 = shufflevector <8 x bfloat> %2171, <8 x bfloat> %2174, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2176 = add i32 %2136, 96
  %2177 = inttoptr i32 %2176 to ptr addrspace(3)
  %2178 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2177)
  %2179 = add i32 %2136, 4448
  %2180 = inttoptr i32 %2179 to ptr addrspace(3)
  %2181 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2180)
  %2182 = shufflevector <8 x bfloat> %2178, <8 x bfloat> %2181, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2183 = add i32 %2136, 128
  %2184 = inttoptr i32 %2183 to ptr addrspace(3)
  %2185 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2184)
  %2186 = add i32 %2136, 4480
  %2187 = inttoptr i32 %2186 to ptr addrspace(3)
  %2188 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2187)
  %2189 = shufflevector <8 x bfloat> %2185, <8 x bfloat> %2188, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2190 = add i32 %2136, 160
  %2191 = inttoptr i32 %2190 to ptr addrspace(3)
  %2192 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2191)
  %2193 = add i32 %2136, 4512
  %2194 = inttoptr i32 %2193 to ptr addrspace(3)
  %2195 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2194)
  %2196 = shufflevector <8 x bfloat> %2192, <8 x bfloat> %2195, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2197 = add i32 %2136, 192
  %2198 = inttoptr i32 %2197 to ptr addrspace(3)
  %2199 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2198)
  %2200 = add i32 %2136, 4544
  %2201 = inttoptr i32 %2200 to ptr addrspace(3)
  %2202 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2201)
  %2203 = shufflevector <8 x bfloat> %2199, <8 x bfloat> %2202, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2204 = add i32 %2136, 224
  %2205 = inttoptr i32 %2204 to ptr addrspace(3)
  %2206 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2205)
  %2207 = add i32 %2136, 4576
  %2208 = inttoptr i32 %2207 to ptr addrspace(3)
  %2209 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2208)
  %2210 = shufflevector <8 x bfloat> %2206, <8 x bfloat> %2209, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2211 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 68096), %2147
  %2212 = add i32 %2211, %219
  %2213 = inttoptr i32 %2212 to ptr addrspace(3)
  %2214 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2213)
  %2215 = add i32 %2212, 1280
  %2216 = inttoptr i32 %2215 to ptr addrspace(3)
  %2217 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2216)
  %2218 = shufflevector <8 x bfloat> %2214, <8 x bfloat> %2217, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2219 = add i32 %2136, 8704
  %2220 = inttoptr i32 %2219 to ptr addrspace(3)
  %2221 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2220)
  %2222 = add i32 %2136, 13056
  %2223 = inttoptr i32 %2222 to ptr addrspace(3)
  %2224 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2223)
  %2225 = shufflevector <8 x bfloat> %2221, <8 x bfloat> %2224, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2226 = add i32 %2136, 8736
  %2227 = inttoptr i32 %2226 to ptr addrspace(3)
  %2228 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2227)
  %2229 = add i32 %2136, 13088
  %2230 = inttoptr i32 %2229 to ptr addrspace(3)
  %2231 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2230)
  %2232 = shufflevector <8 x bfloat> %2228, <8 x bfloat> %2231, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2233 = add i32 %2136, 8768
  %2234 = inttoptr i32 %2233 to ptr addrspace(3)
  %2235 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2234)
  %2236 = add i32 %2136, 13120
  %2237 = inttoptr i32 %2236 to ptr addrspace(3)
  %2238 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2237)
  %2239 = shufflevector <8 x bfloat> %2235, <8 x bfloat> %2238, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2240 = add i32 %2136, 8800
  %2241 = inttoptr i32 %2240 to ptr addrspace(3)
  %2242 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2241)
  %2243 = add i32 %2136, 13152
  %2244 = inttoptr i32 %2243 to ptr addrspace(3)
  %2245 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2244)
  %2246 = shufflevector <8 x bfloat> %2242, <8 x bfloat> %2245, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2247 = add i32 %2136, 8832
  %2248 = inttoptr i32 %2247 to ptr addrspace(3)
  %2249 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2248)
  %2250 = add i32 %2136, 13184
  %2251 = inttoptr i32 %2250 to ptr addrspace(3)
  %2252 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2251)
  %2253 = shufflevector <8 x bfloat> %2249, <8 x bfloat> %2252, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2254 = add i32 %2136, 8864
  %2255 = inttoptr i32 %2254 to ptr addrspace(3)
  %2256 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2255)
  %2257 = add i32 %2136, 13216
  %2258 = inttoptr i32 %2257 to ptr addrspace(3)
  %2259 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2258)
  %2260 = shufflevector <8 x bfloat> %2256, <8 x bfloat> %2259, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2261 = add i32 %2136, 8896
  %2262 = inttoptr i32 %2261 to ptr addrspace(3)
  %2263 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2262)
  %2264 = add i32 %2136, 13248
  %2265 = inttoptr i32 %2264 to ptr addrspace(3)
  %2266 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2265)
  %2267 = shufflevector <8 x bfloat> %2263, <8 x bfloat> %2266, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2268 = add i32 %2136, 8928
  %2269 = inttoptr i32 %2268 to ptr addrspace(3)
  %2270 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2269)
  %2271 = add i32 %2136, 13280
  %2272 = inttoptr i32 %2271 to ptr addrspace(3)
  %2273 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2272)
  %2274 = shufflevector <8 x bfloat> %2270, <8 x bfloat> %2273, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2275 = add i32 %219, 32
  %2276 = add i32 %2148, %2275
  %2277 = inttoptr i32 %2276 to ptr addrspace(3)
  %2278 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2277)
  %2279 = add i32 %2276, 1280
  %2280 = inttoptr i32 %2279 to ptr addrspace(3)
  %2281 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2280)
  %2282 = shufflevector <8 x bfloat> %2278, <8 x bfloat> %2281, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2283 = add i32 %2211, %2275
  %2284 = inttoptr i32 %2283 to ptr addrspace(3)
  %2285 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2284)
  %2286 = add i32 %2283, 1280
  %2287 = inttoptr i32 %2286 to ptr addrspace(3)
  %2288 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2287)
  %2289 = shufflevector <8 x bfloat> %2285, <8 x bfloat> %2288, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2290 = add i32 %2138, 8704
  %2291 = inttoptr i32 %2290 to ptr addrspace(3)
  %2292 = load <8 x bfloat>, ptr addrspace(3) %2291, align 16
  %2293 = add i32 %2138, 8736
  %2294 = inttoptr i32 %2293 to ptr addrspace(3)
  %2295 = load <8 x bfloat>, ptr addrspace(3) %2294, align 16
  %2296 = add i32 %2138, 8768
  %2297 = inttoptr i32 %2296 to ptr addrspace(3)
  %2298 = load <8 x bfloat>, ptr addrspace(3) %2297, align 16
  %2299 = add i32 %2138, 8800
  %2300 = inttoptr i32 %2299 to ptr addrspace(3)
  %2301 = load <8 x bfloat>, ptr addrspace(3) %2300, align 16
  %2302 = add i32 %2138, 8832
  %2303 = inttoptr i32 %2302 to ptr addrspace(3)
  %2304 = load <8 x bfloat>, ptr addrspace(3) %2303, align 16
  %2305 = add i32 %2138, 8864
  %2306 = inttoptr i32 %2305 to ptr addrspace(3)
  %2307 = load <8 x bfloat>, ptr addrspace(3) %2306, align 16
  %2308 = add i32 %2138, 8896
  %2309 = inttoptr i32 %2308 to ptr addrspace(3)
  %2310 = load <8 x bfloat>, ptr addrspace(3) %2309, align 16
  %2311 = add i32 %2138, 8928
  %2312 = inttoptr i32 %2311 to ptr addrspace(3)
  %2313 = load <8 x bfloat>, ptr addrspace(3) %2312, align 16
  %2314 = inttoptr i32 %2138 to ptr addrspace(3)
  %2315 = load <8 x bfloat>, ptr addrspace(3) %2314, align 16
  %2316 = add i32 %2138, 32
  %2317 = inttoptr i32 %2316 to ptr addrspace(3)
  %2318 = load <8 x bfloat>, ptr addrspace(3) %2317, align 16
  %2319 = add i32 %2138, 64
  %2320 = inttoptr i32 %2319 to ptr addrspace(3)
  %2321 = load <8 x bfloat>, ptr addrspace(3) %2320, align 16
  %2322 = add i32 %2138, 96
  %2323 = inttoptr i32 %2322 to ptr addrspace(3)
  %2324 = load <8 x bfloat>, ptr addrspace(3) %2323, align 16
  %2325 = add i32 %2138, 128
  %2326 = inttoptr i32 %2325 to ptr addrspace(3)
  %2327 = load <8 x bfloat>, ptr addrspace(3) %2326, align 16
  %2328 = add i32 %2138, 160
  %2329 = inttoptr i32 %2328 to ptr addrspace(3)
  %2330 = load <8 x bfloat>, ptr addrspace(3) %2329, align 16
  %2331 = add i32 %2138, 192
  %2332 = inttoptr i32 %2331 to ptr addrspace(3)
  %2333 = load <8 x bfloat>, ptr addrspace(3) %2332, align 16
  %2334 = add i32 %2138, 224
  %2335 = inttoptr i32 %2334 to ptr addrspace(3)
  %2336 = load <8 x bfloat>, ptr addrspace(3) %2335, align 16
  %2337 = add i32 %2138, 13056
  %2338 = inttoptr i32 %2337 to ptr addrspace(3)
  %2339 = load <8 x bfloat>, ptr addrspace(3) %2338, align 16
  %2340 = add i32 %2138, 13088
  %2341 = inttoptr i32 %2340 to ptr addrspace(3)
  %2342 = load <8 x bfloat>, ptr addrspace(3) %2341, align 16
  %2343 = add i32 %2138, 13120
  %2344 = inttoptr i32 %2343 to ptr addrspace(3)
  %2345 = load <8 x bfloat>, ptr addrspace(3) %2344, align 16
  %2346 = add i32 %2138, 13152
  %2347 = inttoptr i32 %2346 to ptr addrspace(3)
  %2348 = load <8 x bfloat>, ptr addrspace(3) %2347, align 16
  %2349 = add i32 %2138, 13184
  %2350 = inttoptr i32 %2349 to ptr addrspace(3)
  %2351 = load <8 x bfloat>, ptr addrspace(3) %2350, align 16
  %2352 = add i32 %2138, 13216
  %2353 = inttoptr i32 %2352 to ptr addrspace(3)
  %2354 = load <8 x bfloat>, ptr addrspace(3) %2353, align 16
  %2355 = add i32 %2138, 13248
  %2356 = inttoptr i32 %2355 to ptr addrspace(3)
  %2357 = load <8 x bfloat>, ptr addrspace(3) %2356, align 16
  %2358 = add i32 %2138, 13280
  %2359 = inttoptr i32 %2358 to ptr addrspace(3)
  %2360 = load <8 x bfloat>, ptr addrspace(3) %2359, align 16
  %2361 = add i32 %2138, 4352
  %2362 = inttoptr i32 %2361 to ptr addrspace(3)
  %2363 = load <8 x bfloat>, ptr addrspace(3) %2362, align 16
  %2364 = add i32 %2138, 4384
  %2365 = inttoptr i32 %2364 to ptr addrspace(3)
  %2366 = load <8 x bfloat>, ptr addrspace(3) %2365, align 16
  %2367 = add i32 %2138, 4416
  %2368 = inttoptr i32 %2367 to ptr addrspace(3)
  %2369 = load <8 x bfloat>, ptr addrspace(3) %2368, align 16
  %2370 = add i32 %2138, 4448
  %2371 = inttoptr i32 %2370 to ptr addrspace(3)
  %2372 = load <8 x bfloat>, ptr addrspace(3) %2371, align 16
  %2373 = add i32 %2138, 4480
  %2374 = inttoptr i32 %2373 to ptr addrspace(3)
  %2375 = load <8 x bfloat>, ptr addrspace(3) %2374, align 16
  %2376 = add i32 %2138, 4512
  %2377 = inttoptr i32 %2376 to ptr addrspace(3)
  %2378 = load <8 x bfloat>, ptr addrspace(3) %2377, align 16
  %2379 = add i32 %2138, 4544
  %2380 = inttoptr i32 %2379 to ptr addrspace(3)
  %2381 = load <8 x bfloat>, ptr addrspace(3) %2380, align 16
  %2382 = add i32 %2138, 4576
  %2383 = inttoptr i32 %2382 to ptr addrspace(3)
  %2384 = load <8 x bfloat>, ptr addrspace(3) %2383, align 16
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2385 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2155, <16 x bfloat> %2161, i16 0, <8 x float> %1488, i1 false, i1 false)
  %2386 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2155, <16 x bfloat> %2168, i16 0, <8 x float> %1489, i1 true, i1 false)
  %2387 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2155, <16 x bfloat> %2175, i16 0, <8 x float> %1490, i1 true, i1 false)
  %2388 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2155, <16 x bfloat> %2182, i16 0, <8 x float> %1491, i1 true, i1 false)
  %2389 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2155, <16 x bfloat> %2189, i16 0, <8 x float> %1492, i1 true, i1 false)
  %2390 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2155, <16 x bfloat> %2196, i16 0, <8 x float> %1493, i1 true, i1 false)
  %2391 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2155, <16 x bfloat> %2203, i16 0, <8 x float> %1494, i1 true, i1 false)
  %2392 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2155, <16 x bfloat> %2210, i16 0, <8 x float> %1495, i1 true, i1 false)
  %2393 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2218, <16 x bfloat> %2225, i16 0, <8 x float> %1504, i1 false, i1 false)
  %2394 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2218, <16 x bfloat> %2232, i16 0, <8 x float> %1505, i1 true, i1 false)
  %2395 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2218, <16 x bfloat> %2239, i16 0, <8 x float> %1506, i1 true, i1 false)
  %2396 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2218, <16 x bfloat> %2246, i16 0, <8 x float> %1507, i1 true, i1 false)
  %2397 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2218, <16 x bfloat> %2253, i16 0, <8 x float> %1508, i1 true, i1 false)
  %2398 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2218, <16 x bfloat> %2260, i16 0, <8 x float> %1509, i1 true, i1 false)
  %2399 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2218, <16 x bfloat> %2267, i16 0, <8 x float> %1510, i1 true, i1 false)
  %2400 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2218, <16 x bfloat> %2274, i16 0, <8 x float> %1511, i1 true, i1 false)
  %2401 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2282, <16 x bfloat> %2161, i16 0, <8 x float> %1496, i1 false, i1 false)
  %2402 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2282, <16 x bfloat> %2168, i16 0, <8 x float> %1497, i1 true, i1 false)
  %2403 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2282, <16 x bfloat> %2175, i16 0, <8 x float> %1498, i1 true, i1 false)
  %2404 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2282, <16 x bfloat> %2182, i16 0, <8 x float> %1499, i1 true, i1 false)
  %2405 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2282, <16 x bfloat> %2189, i16 0, <8 x float> %1500, i1 true, i1 false)
  %2406 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2282, <16 x bfloat> %2196, i16 0, <8 x float> %1501, i1 true, i1 false)
  %2407 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2282, <16 x bfloat> %2203, i16 0, <8 x float> %1502, i1 true, i1 false)
  %2408 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2282, <16 x bfloat> %2210, i16 0, <8 x float> %1503, i1 true, i1 false)
  %2409 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2289, <16 x bfloat> %2225, i16 0, <8 x float> %1512, i1 false, i1 false)
  %2410 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2289, <16 x bfloat> %2232, i16 0, <8 x float> %1513, i1 true, i1 false)
  %2411 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2289, <16 x bfloat> %2239, i16 0, <8 x float> %1514, i1 true, i1 false)
  %2412 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2289, <16 x bfloat> %2246, i16 0, <8 x float> %1515, i1 true, i1 false)
  %2413 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2289, <16 x bfloat> %2253, i16 0, <8 x float> %1516, i1 true, i1 false)
  %2414 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2289, <16 x bfloat> %2260, i16 0, <8 x float> %1517, i1 true, i1 false)
  %2415 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2289, <16 x bfloat> %2267, i16 0, <8 x float> %1518, i1 true, i1 false)
  %2416 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2289, <16 x bfloat> %2274, i16 0, <8 x float> %1519, i1 true, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %2417 = add i64 %1487, 1
  br label %1486

2418:                                             ; preds = %1486
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  %2419 = mul i32 %44, %25
  %2420 = add i32 %2419, %50
  %2421 = mul i32 %2420, %18
  %2422 = mul i32 %2421, %20
  %2423 = mul i32 %2422, 128
  %2424 = mul i32 %42, 128
  %2425 = add i32 %2423, %2424
  %2426 = add i32 %60, %206
  %2427 = mul i32 %2426, %20
  %2428 = mul i32 %2427, 128
  %2429 = add i32 %2425, %2428
  %2430 = add i32 %2429, %51
  %2431 = extractelement <8 x float> %1488, i64 0
  %2432 = mul i32 %2430, 4
  %2433 = bitcast float %2431 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2433, ptr addrspace(8) %78, i32 %2432, i32 0, i32 0)
  %2434 = extractelement <8 x float> %1504, i64 0
  %2435 = bitcast float %2434 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2435, ptr addrspace(8) %79, i32 %2432, i32 0, i32 0)
  %2436 = add i32 %2426, 1
  %2437 = mul i32 %2436, %20
  %2438 = mul i32 %2437, 128
  %2439 = add i32 %2425, %2438
  %2440 = add i32 %2439, %51
  %2441 = extractelement <8 x float> %1488, i64 1
  %2442 = mul i32 %2440, 4
  %2443 = bitcast float %2441 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2443, ptr addrspace(8) %78, i32 %2442, i32 0, i32 0)
  %2444 = extractelement <8 x float> %1504, i64 1
  %2445 = bitcast float %2444 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2445, ptr addrspace(8) %79, i32 %2442, i32 0, i32 0)
  %2446 = add i32 %2426, 2
  %2447 = mul i32 %2446, %20
  %2448 = mul i32 %2447, 128
  %2449 = add i32 %2425, %2448
  %2450 = add i32 %2449, %51
  %2451 = extractelement <8 x float> %1488, i64 2
  %2452 = mul i32 %2450, 4
  %2453 = bitcast float %2451 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2453, ptr addrspace(8) %78, i32 %2452, i32 0, i32 0)
  %2454 = extractelement <8 x float> %1504, i64 2
  %2455 = bitcast float %2454 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2455, ptr addrspace(8) %79, i32 %2452, i32 0, i32 0)
  %2456 = add i32 %2426, 3
  %2457 = mul i32 %2456, %20
  %2458 = mul i32 %2457, 128
  %2459 = add i32 %2425, %2458
  %2460 = add i32 %2459, %51
  %2461 = extractelement <8 x float> %1488, i64 3
  %2462 = mul i32 %2460, 4
  %2463 = bitcast float %2461 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2463, ptr addrspace(8) %78, i32 %2462, i32 0, i32 0)
  %2464 = extractelement <8 x float> %1504, i64 3
  %2465 = bitcast float %2464 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2465, ptr addrspace(8) %79, i32 %2462, i32 0, i32 0)
  %2466 = add i32 %2426, 4
  %2467 = mul i32 %2466, %20
  %2468 = mul i32 %2467, 128
  %2469 = add i32 %2425, %2468
  %2470 = add i32 %2469, %51
  %2471 = extractelement <8 x float> %1488, i64 4
  %2472 = mul i32 %2470, 4
  %2473 = bitcast float %2471 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2473, ptr addrspace(8) %78, i32 %2472, i32 0, i32 0)
  %2474 = extractelement <8 x float> %1504, i64 4
  %2475 = bitcast float %2474 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2475, ptr addrspace(8) %79, i32 %2472, i32 0, i32 0)
  %2476 = add i32 %2426, 5
  %2477 = mul i32 %2476, %20
  %2478 = mul i32 %2477, 128
  %2479 = add i32 %2425, %2478
  %2480 = add i32 %2479, %51
  %2481 = extractelement <8 x float> %1488, i64 5
  %2482 = mul i32 %2480, 4
  %2483 = bitcast float %2481 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2483, ptr addrspace(8) %78, i32 %2482, i32 0, i32 0)
  %2484 = extractelement <8 x float> %1504, i64 5
  %2485 = bitcast float %2484 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2485, ptr addrspace(8) %79, i32 %2482, i32 0, i32 0)
  %2486 = add i32 %2426, 6
  %2487 = mul i32 %2486, %20
  %2488 = mul i32 %2487, 128
  %2489 = add i32 %2425, %2488
  %2490 = add i32 %2489, %51
  %2491 = extractelement <8 x float> %1488, i64 6
  %2492 = mul i32 %2490, 4
  %2493 = bitcast float %2491 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2493, ptr addrspace(8) %78, i32 %2492, i32 0, i32 0)
  %2494 = extractelement <8 x float> %1504, i64 6
  %2495 = bitcast float %2494 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2495, ptr addrspace(8) %79, i32 %2492, i32 0, i32 0)
  %2496 = add i32 %2426, 7
  %2497 = mul i32 %2496, %20
  %2498 = mul i32 %2497, 128
  %2499 = add i32 %2425, %2498
  %2500 = add i32 %2499, %51
  %2501 = extractelement <8 x float> %1488, i64 7
  %2502 = mul i32 %2500, 4
  %2503 = bitcast float %2501 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2503, ptr addrspace(8) %78, i32 %2502, i32 0, i32 0)
  %2504 = extractelement <8 x float> %1504, i64 7
  %2505 = bitcast float %2504 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2505, ptr addrspace(8) %79, i32 %2502, i32 0, i32 0)
  %2506 = add i32 %2429, 16
  %2507 = add i32 %2506, %51
  %2508 = extractelement <8 x float> %1489, i64 0
  %2509 = mul i32 %2507, 4
  %2510 = bitcast float %2508 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2510, ptr addrspace(8) %78, i32 %2509, i32 0, i32 0)
  %2511 = extractelement <8 x float> %1505, i64 0
  %2512 = bitcast float %2511 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2512, ptr addrspace(8) %79, i32 %2509, i32 0, i32 0)
  %2513 = add i32 %2439, 16
  %2514 = add i32 %2513, %51
  %2515 = extractelement <8 x float> %1489, i64 1
  %2516 = mul i32 %2514, 4
  %2517 = bitcast float %2515 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2517, ptr addrspace(8) %78, i32 %2516, i32 0, i32 0)
  %2518 = extractelement <8 x float> %1505, i64 1
  %2519 = bitcast float %2518 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2519, ptr addrspace(8) %79, i32 %2516, i32 0, i32 0)
  %2520 = add i32 %2449, 16
  %2521 = add i32 %2520, %51
  %2522 = extractelement <8 x float> %1489, i64 2
  %2523 = mul i32 %2521, 4
  %2524 = bitcast float %2522 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2524, ptr addrspace(8) %78, i32 %2523, i32 0, i32 0)
  %2525 = extractelement <8 x float> %1505, i64 2
  %2526 = bitcast float %2525 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2526, ptr addrspace(8) %79, i32 %2523, i32 0, i32 0)
  %2527 = add i32 %2459, 16
  %2528 = add i32 %2527, %51
  %2529 = extractelement <8 x float> %1489, i64 3
  %2530 = mul i32 %2528, 4
  %2531 = bitcast float %2529 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2531, ptr addrspace(8) %78, i32 %2530, i32 0, i32 0)
  %2532 = extractelement <8 x float> %1505, i64 3
  %2533 = bitcast float %2532 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2533, ptr addrspace(8) %79, i32 %2530, i32 0, i32 0)
  %2534 = add i32 %2469, 16
  %2535 = add i32 %2534, %51
  %2536 = extractelement <8 x float> %1489, i64 4
  %2537 = mul i32 %2535, 4
  %2538 = bitcast float %2536 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2538, ptr addrspace(8) %78, i32 %2537, i32 0, i32 0)
  %2539 = extractelement <8 x float> %1505, i64 4
  %2540 = bitcast float %2539 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2540, ptr addrspace(8) %79, i32 %2537, i32 0, i32 0)
  %2541 = add i32 %2479, 16
  %2542 = add i32 %2541, %51
  %2543 = extractelement <8 x float> %1489, i64 5
  %2544 = mul i32 %2542, 4
  %2545 = bitcast float %2543 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2545, ptr addrspace(8) %78, i32 %2544, i32 0, i32 0)
  %2546 = extractelement <8 x float> %1505, i64 5
  %2547 = bitcast float %2546 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2547, ptr addrspace(8) %79, i32 %2544, i32 0, i32 0)
  %2548 = add i32 %2489, 16
  %2549 = add i32 %2548, %51
  %2550 = extractelement <8 x float> %1489, i64 6
  %2551 = mul i32 %2549, 4
  %2552 = bitcast float %2550 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2552, ptr addrspace(8) %78, i32 %2551, i32 0, i32 0)
  %2553 = extractelement <8 x float> %1505, i64 6
  %2554 = bitcast float %2553 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2554, ptr addrspace(8) %79, i32 %2551, i32 0, i32 0)
  %2555 = add i32 %2499, 16
  %2556 = add i32 %2555, %51
  %2557 = extractelement <8 x float> %1489, i64 7
  %2558 = mul i32 %2556, 4
  %2559 = bitcast float %2557 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2559, ptr addrspace(8) %78, i32 %2558, i32 0, i32 0)
  %2560 = extractelement <8 x float> %1505, i64 7
  %2561 = bitcast float %2560 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2561, ptr addrspace(8) %79, i32 %2558, i32 0, i32 0)
  %2562 = add i32 %2429, 32
  %2563 = add i32 %2562, %51
  %2564 = extractelement <8 x float> %1490, i64 0
  %2565 = mul i32 %2563, 4
  %2566 = bitcast float %2564 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2566, ptr addrspace(8) %78, i32 %2565, i32 0, i32 0)
  %2567 = extractelement <8 x float> %1506, i64 0
  %2568 = bitcast float %2567 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2568, ptr addrspace(8) %79, i32 %2565, i32 0, i32 0)
  %2569 = add i32 %2439, 32
  %2570 = add i32 %2569, %51
  %2571 = extractelement <8 x float> %1490, i64 1
  %2572 = mul i32 %2570, 4
  %2573 = bitcast float %2571 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2573, ptr addrspace(8) %78, i32 %2572, i32 0, i32 0)
  %2574 = extractelement <8 x float> %1506, i64 1
  %2575 = bitcast float %2574 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2575, ptr addrspace(8) %79, i32 %2572, i32 0, i32 0)
  %2576 = add i32 %2449, 32
  %2577 = add i32 %2576, %51
  %2578 = extractelement <8 x float> %1490, i64 2
  %2579 = mul i32 %2577, 4
  %2580 = bitcast float %2578 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2580, ptr addrspace(8) %78, i32 %2579, i32 0, i32 0)
  %2581 = extractelement <8 x float> %1506, i64 2
  %2582 = bitcast float %2581 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2582, ptr addrspace(8) %79, i32 %2579, i32 0, i32 0)
  %2583 = add i32 %2459, 32
  %2584 = add i32 %2583, %51
  %2585 = extractelement <8 x float> %1490, i64 3
  %2586 = mul i32 %2584, 4
  %2587 = bitcast float %2585 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2587, ptr addrspace(8) %78, i32 %2586, i32 0, i32 0)
  %2588 = extractelement <8 x float> %1506, i64 3
  %2589 = bitcast float %2588 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2589, ptr addrspace(8) %79, i32 %2586, i32 0, i32 0)
  %2590 = add i32 %2469, 32
  %2591 = add i32 %2590, %51
  %2592 = extractelement <8 x float> %1490, i64 4
  %2593 = mul i32 %2591, 4
  %2594 = bitcast float %2592 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2594, ptr addrspace(8) %78, i32 %2593, i32 0, i32 0)
  %2595 = extractelement <8 x float> %1506, i64 4
  %2596 = bitcast float %2595 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2596, ptr addrspace(8) %79, i32 %2593, i32 0, i32 0)
  %2597 = add i32 %2479, 32
  %2598 = add i32 %2597, %51
  %2599 = extractelement <8 x float> %1490, i64 5
  %2600 = mul i32 %2598, 4
  %2601 = bitcast float %2599 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2601, ptr addrspace(8) %78, i32 %2600, i32 0, i32 0)
  %2602 = extractelement <8 x float> %1506, i64 5
  %2603 = bitcast float %2602 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2603, ptr addrspace(8) %79, i32 %2600, i32 0, i32 0)
  %2604 = add i32 %2489, 32
  %2605 = add i32 %2604, %51
  %2606 = extractelement <8 x float> %1490, i64 6
  %2607 = mul i32 %2605, 4
  %2608 = bitcast float %2606 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2608, ptr addrspace(8) %78, i32 %2607, i32 0, i32 0)
  %2609 = extractelement <8 x float> %1506, i64 6
  %2610 = bitcast float %2609 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2610, ptr addrspace(8) %79, i32 %2607, i32 0, i32 0)
  %2611 = add i32 %2499, 32
  %2612 = add i32 %2611, %51
  %2613 = extractelement <8 x float> %1490, i64 7
  %2614 = mul i32 %2612, 4
  %2615 = bitcast float %2613 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2615, ptr addrspace(8) %78, i32 %2614, i32 0, i32 0)
  %2616 = extractelement <8 x float> %1506, i64 7
  %2617 = bitcast float %2616 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2617, ptr addrspace(8) %79, i32 %2614, i32 0, i32 0)
  %2618 = add i32 %2429, 48
  %2619 = add i32 %2618, %51
  %2620 = extractelement <8 x float> %1491, i64 0
  %2621 = mul i32 %2619, 4
  %2622 = bitcast float %2620 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2622, ptr addrspace(8) %78, i32 %2621, i32 0, i32 0)
  %2623 = extractelement <8 x float> %1507, i64 0
  %2624 = bitcast float %2623 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2624, ptr addrspace(8) %79, i32 %2621, i32 0, i32 0)
  %2625 = add i32 %2439, 48
  %2626 = add i32 %2625, %51
  %2627 = extractelement <8 x float> %1491, i64 1
  %2628 = mul i32 %2626, 4
  %2629 = bitcast float %2627 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2629, ptr addrspace(8) %78, i32 %2628, i32 0, i32 0)
  %2630 = extractelement <8 x float> %1507, i64 1
  %2631 = bitcast float %2630 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2631, ptr addrspace(8) %79, i32 %2628, i32 0, i32 0)
  %2632 = add i32 %2449, 48
  %2633 = add i32 %2632, %51
  %2634 = extractelement <8 x float> %1491, i64 2
  %2635 = mul i32 %2633, 4
  %2636 = bitcast float %2634 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2636, ptr addrspace(8) %78, i32 %2635, i32 0, i32 0)
  %2637 = extractelement <8 x float> %1507, i64 2
  %2638 = bitcast float %2637 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2638, ptr addrspace(8) %79, i32 %2635, i32 0, i32 0)
  %2639 = add i32 %2459, 48
  %2640 = add i32 %2639, %51
  %2641 = extractelement <8 x float> %1491, i64 3
  %2642 = mul i32 %2640, 4
  %2643 = bitcast float %2641 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2643, ptr addrspace(8) %78, i32 %2642, i32 0, i32 0)
  %2644 = extractelement <8 x float> %1507, i64 3
  %2645 = bitcast float %2644 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2645, ptr addrspace(8) %79, i32 %2642, i32 0, i32 0)
  %2646 = add i32 %2469, 48
  %2647 = add i32 %2646, %51
  %2648 = extractelement <8 x float> %1491, i64 4
  %2649 = mul i32 %2647, 4
  %2650 = bitcast float %2648 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2650, ptr addrspace(8) %78, i32 %2649, i32 0, i32 0)
  %2651 = extractelement <8 x float> %1507, i64 4
  %2652 = bitcast float %2651 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2652, ptr addrspace(8) %79, i32 %2649, i32 0, i32 0)
  %2653 = add i32 %2479, 48
  %2654 = add i32 %2653, %51
  %2655 = extractelement <8 x float> %1491, i64 5
  %2656 = mul i32 %2654, 4
  %2657 = bitcast float %2655 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2657, ptr addrspace(8) %78, i32 %2656, i32 0, i32 0)
  %2658 = extractelement <8 x float> %1507, i64 5
  %2659 = bitcast float %2658 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2659, ptr addrspace(8) %79, i32 %2656, i32 0, i32 0)
  %2660 = add i32 %2489, 48
  %2661 = add i32 %2660, %51
  %2662 = extractelement <8 x float> %1491, i64 6
  %2663 = mul i32 %2661, 4
  %2664 = bitcast float %2662 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2664, ptr addrspace(8) %78, i32 %2663, i32 0, i32 0)
  %2665 = extractelement <8 x float> %1507, i64 6
  %2666 = bitcast float %2665 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2666, ptr addrspace(8) %79, i32 %2663, i32 0, i32 0)
  %2667 = add i32 %2499, 48
  %2668 = add i32 %2667, %51
  %2669 = extractelement <8 x float> %1491, i64 7
  %2670 = mul i32 %2668, 4
  %2671 = bitcast float %2669 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2671, ptr addrspace(8) %78, i32 %2670, i32 0, i32 0)
  %2672 = extractelement <8 x float> %1507, i64 7
  %2673 = bitcast float %2672 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2673, ptr addrspace(8) %79, i32 %2670, i32 0, i32 0)
  %2674 = add i32 %2429, 64
  %2675 = add i32 %2674, %51
  %2676 = extractelement <8 x float> %1492, i64 0
  %2677 = mul i32 %2675, 4
  %2678 = bitcast float %2676 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2678, ptr addrspace(8) %78, i32 %2677, i32 0, i32 0)
  %2679 = extractelement <8 x float> %1508, i64 0
  %2680 = bitcast float %2679 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2680, ptr addrspace(8) %79, i32 %2677, i32 0, i32 0)
  %2681 = add i32 %2439, 64
  %2682 = add i32 %2681, %51
  %2683 = extractelement <8 x float> %1492, i64 1
  %2684 = mul i32 %2682, 4
  %2685 = bitcast float %2683 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2685, ptr addrspace(8) %78, i32 %2684, i32 0, i32 0)
  %2686 = extractelement <8 x float> %1508, i64 1
  %2687 = bitcast float %2686 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2687, ptr addrspace(8) %79, i32 %2684, i32 0, i32 0)
  %2688 = add i32 %2449, 64
  %2689 = add i32 %2688, %51
  %2690 = extractelement <8 x float> %1492, i64 2
  %2691 = mul i32 %2689, 4
  %2692 = bitcast float %2690 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2692, ptr addrspace(8) %78, i32 %2691, i32 0, i32 0)
  %2693 = extractelement <8 x float> %1508, i64 2
  %2694 = bitcast float %2693 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2694, ptr addrspace(8) %79, i32 %2691, i32 0, i32 0)
  %2695 = add i32 %2459, 64
  %2696 = add i32 %2695, %51
  %2697 = extractelement <8 x float> %1492, i64 3
  %2698 = mul i32 %2696, 4
  %2699 = bitcast float %2697 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2699, ptr addrspace(8) %78, i32 %2698, i32 0, i32 0)
  %2700 = extractelement <8 x float> %1508, i64 3
  %2701 = bitcast float %2700 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2701, ptr addrspace(8) %79, i32 %2698, i32 0, i32 0)
  %2702 = add i32 %2469, 64
  %2703 = add i32 %2702, %51
  %2704 = extractelement <8 x float> %1492, i64 4
  %2705 = mul i32 %2703, 4
  %2706 = bitcast float %2704 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2706, ptr addrspace(8) %78, i32 %2705, i32 0, i32 0)
  %2707 = extractelement <8 x float> %1508, i64 4
  %2708 = bitcast float %2707 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2708, ptr addrspace(8) %79, i32 %2705, i32 0, i32 0)
  %2709 = add i32 %2479, 64
  %2710 = add i32 %2709, %51
  %2711 = extractelement <8 x float> %1492, i64 5
  %2712 = mul i32 %2710, 4
  %2713 = bitcast float %2711 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2713, ptr addrspace(8) %78, i32 %2712, i32 0, i32 0)
  %2714 = extractelement <8 x float> %1508, i64 5
  %2715 = bitcast float %2714 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2715, ptr addrspace(8) %79, i32 %2712, i32 0, i32 0)
  %2716 = add i32 %2489, 64
  %2717 = add i32 %2716, %51
  %2718 = extractelement <8 x float> %1492, i64 6
  %2719 = mul i32 %2717, 4
  %2720 = bitcast float %2718 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2720, ptr addrspace(8) %78, i32 %2719, i32 0, i32 0)
  %2721 = extractelement <8 x float> %1508, i64 6
  %2722 = bitcast float %2721 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2722, ptr addrspace(8) %79, i32 %2719, i32 0, i32 0)
  %2723 = add i32 %2499, 64
  %2724 = add i32 %2723, %51
  %2725 = extractelement <8 x float> %1492, i64 7
  %2726 = mul i32 %2724, 4
  %2727 = bitcast float %2725 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2727, ptr addrspace(8) %78, i32 %2726, i32 0, i32 0)
  %2728 = extractelement <8 x float> %1508, i64 7
  %2729 = bitcast float %2728 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2729, ptr addrspace(8) %79, i32 %2726, i32 0, i32 0)
  %2730 = add i32 %2429, 80
  %2731 = add i32 %2730, %51
  %2732 = extractelement <8 x float> %1493, i64 0
  %2733 = mul i32 %2731, 4
  %2734 = bitcast float %2732 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2734, ptr addrspace(8) %78, i32 %2733, i32 0, i32 0)
  %2735 = extractelement <8 x float> %1509, i64 0
  %2736 = bitcast float %2735 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2736, ptr addrspace(8) %79, i32 %2733, i32 0, i32 0)
  %2737 = add i32 %2439, 80
  %2738 = add i32 %2737, %51
  %2739 = extractelement <8 x float> %1493, i64 1
  %2740 = mul i32 %2738, 4
  %2741 = bitcast float %2739 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2741, ptr addrspace(8) %78, i32 %2740, i32 0, i32 0)
  %2742 = extractelement <8 x float> %1509, i64 1
  %2743 = bitcast float %2742 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2743, ptr addrspace(8) %79, i32 %2740, i32 0, i32 0)
  %2744 = add i32 %2449, 80
  %2745 = add i32 %2744, %51
  %2746 = extractelement <8 x float> %1493, i64 2
  %2747 = mul i32 %2745, 4
  %2748 = bitcast float %2746 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2748, ptr addrspace(8) %78, i32 %2747, i32 0, i32 0)
  %2749 = extractelement <8 x float> %1509, i64 2
  %2750 = bitcast float %2749 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2750, ptr addrspace(8) %79, i32 %2747, i32 0, i32 0)
  %2751 = add i32 %2459, 80
  %2752 = add i32 %2751, %51
  %2753 = extractelement <8 x float> %1493, i64 3
  %2754 = mul i32 %2752, 4
  %2755 = bitcast float %2753 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2755, ptr addrspace(8) %78, i32 %2754, i32 0, i32 0)
  %2756 = extractelement <8 x float> %1509, i64 3
  %2757 = bitcast float %2756 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2757, ptr addrspace(8) %79, i32 %2754, i32 0, i32 0)
  %2758 = add i32 %2469, 80
  %2759 = add i32 %2758, %51
  %2760 = extractelement <8 x float> %1493, i64 4
  %2761 = mul i32 %2759, 4
  %2762 = bitcast float %2760 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2762, ptr addrspace(8) %78, i32 %2761, i32 0, i32 0)
  %2763 = extractelement <8 x float> %1509, i64 4
  %2764 = bitcast float %2763 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2764, ptr addrspace(8) %79, i32 %2761, i32 0, i32 0)
  %2765 = add i32 %2479, 80
  %2766 = add i32 %2765, %51
  %2767 = extractelement <8 x float> %1493, i64 5
  %2768 = mul i32 %2766, 4
  %2769 = bitcast float %2767 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2769, ptr addrspace(8) %78, i32 %2768, i32 0, i32 0)
  %2770 = extractelement <8 x float> %1509, i64 5
  %2771 = bitcast float %2770 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2771, ptr addrspace(8) %79, i32 %2768, i32 0, i32 0)
  %2772 = add i32 %2489, 80
  %2773 = add i32 %2772, %51
  %2774 = extractelement <8 x float> %1493, i64 6
  %2775 = mul i32 %2773, 4
  %2776 = bitcast float %2774 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2776, ptr addrspace(8) %78, i32 %2775, i32 0, i32 0)
  %2777 = extractelement <8 x float> %1509, i64 6
  %2778 = bitcast float %2777 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2778, ptr addrspace(8) %79, i32 %2775, i32 0, i32 0)
  %2779 = add i32 %2499, 80
  %2780 = add i32 %2779, %51
  %2781 = extractelement <8 x float> %1493, i64 7
  %2782 = mul i32 %2780, 4
  %2783 = bitcast float %2781 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2783, ptr addrspace(8) %78, i32 %2782, i32 0, i32 0)
  %2784 = extractelement <8 x float> %1509, i64 7
  %2785 = bitcast float %2784 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2785, ptr addrspace(8) %79, i32 %2782, i32 0, i32 0)
  %2786 = add i32 %2429, 96
  %2787 = add i32 %2786, %51
  %2788 = extractelement <8 x float> %1494, i64 0
  %2789 = mul i32 %2787, 4
  %2790 = bitcast float %2788 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2790, ptr addrspace(8) %78, i32 %2789, i32 0, i32 0)
  %2791 = extractelement <8 x float> %1510, i64 0
  %2792 = bitcast float %2791 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2792, ptr addrspace(8) %79, i32 %2789, i32 0, i32 0)
  %2793 = add i32 %2439, 96
  %2794 = add i32 %2793, %51
  %2795 = extractelement <8 x float> %1494, i64 1
  %2796 = mul i32 %2794, 4
  %2797 = bitcast float %2795 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2797, ptr addrspace(8) %78, i32 %2796, i32 0, i32 0)
  %2798 = extractelement <8 x float> %1510, i64 1
  %2799 = bitcast float %2798 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2799, ptr addrspace(8) %79, i32 %2796, i32 0, i32 0)
  %2800 = add i32 %2449, 96
  %2801 = add i32 %2800, %51
  %2802 = extractelement <8 x float> %1494, i64 2
  %2803 = mul i32 %2801, 4
  %2804 = bitcast float %2802 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2804, ptr addrspace(8) %78, i32 %2803, i32 0, i32 0)
  %2805 = extractelement <8 x float> %1510, i64 2
  %2806 = bitcast float %2805 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2806, ptr addrspace(8) %79, i32 %2803, i32 0, i32 0)
  %2807 = add i32 %2459, 96
  %2808 = add i32 %2807, %51
  %2809 = extractelement <8 x float> %1494, i64 3
  %2810 = mul i32 %2808, 4
  %2811 = bitcast float %2809 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2811, ptr addrspace(8) %78, i32 %2810, i32 0, i32 0)
  %2812 = extractelement <8 x float> %1510, i64 3
  %2813 = bitcast float %2812 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2813, ptr addrspace(8) %79, i32 %2810, i32 0, i32 0)
  %2814 = add i32 %2469, 96
  %2815 = add i32 %2814, %51
  %2816 = extractelement <8 x float> %1494, i64 4
  %2817 = mul i32 %2815, 4
  %2818 = bitcast float %2816 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2818, ptr addrspace(8) %78, i32 %2817, i32 0, i32 0)
  %2819 = extractelement <8 x float> %1510, i64 4
  %2820 = bitcast float %2819 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2820, ptr addrspace(8) %79, i32 %2817, i32 0, i32 0)
  %2821 = add i32 %2479, 96
  %2822 = add i32 %2821, %51
  %2823 = extractelement <8 x float> %1494, i64 5
  %2824 = mul i32 %2822, 4
  %2825 = bitcast float %2823 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2825, ptr addrspace(8) %78, i32 %2824, i32 0, i32 0)
  %2826 = extractelement <8 x float> %1510, i64 5
  %2827 = bitcast float %2826 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2827, ptr addrspace(8) %79, i32 %2824, i32 0, i32 0)
  %2828 = add i32 %2489, 96
  %2829 = add i32 %2828, %51
  %2830 = extractelement <8 x float> %1494, i64 6
  %2831 = mul i32 %2829, 4
  %2832 = bitcast float %2830 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2832, ptr addrspace(8) %78, i32 %2831, i32 0, i32 0)
  %2833 = extractelement <8 x float> %1510, i64 6
  %2834 = bitcast float %2833 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2834, ptr addrspace(8) %79, i32 %2831, i32 0, i32 0)
  %2835 = add i32 %2499, 96
  %2836 = add i32 %2835, %51
  %2837 = extractelement <8 x float> %1494, i64 7
  %2838 = mul i32 %2836, 4
  %2839 = bitcast float %2837 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2839, ptr addrspace(8) %78, i32 %2838, i32 0, i32 0)
  %2840 = extractelement <8 x float> %1510, i64 7
  %2841 = bitcast float %2840 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2841, ptr addrspace(8) %79, i32 %2838, i32 0, i32 0)
  %2842 = add i32 %2429, 112
  %2843 = add i32 %2842, %51
  %2844 = extractelement <8 x float> %1495, i64 0
  %2845 = mul i32 %2843, 4
  %2846 = bitcast float %2844 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2846, ptr addrspace(8) %78, i32 %2845, i32 0, i32 0)
  %2847 = extractelement <8 x float> %1511, i64 0
  %2848 = bitcast float %2847 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2848, ptr addrspace(8) %79, i32 %2845, i32 0, i32 0)
  %2849 = add i32 %2439, 112
  %2850 = add i32 %2849, %51
  %2851 = extractelement <8 x float> %1495, i64 1
  %2852 = mul i32 %2850, 4
  %2853 = bitcast float %2851 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2853, ptr addrspace(8) %78, i32 %2852, i32 0, i32 0)
  %2854 = extractelement <8 x float> %1511, i64 1
  %2855 = bitcast float %2854 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2855, ptr addrspace(8) %79, i32 %2852, i32 0, i32 0)
  %2856 = add i32 %2449, 112
  %2857 = add i32 %2856, %51
  %2858 = extractelement <8 x float> %1495, i64 2
  %2859 = mul i32 %2857, 4
  %2860 = bitcast float %2858 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2860, ptr addrspace(8) %78, i32 %2859, i32 0, i32 0)
  %2861 = extractelement <8 x float> %1511, i64 2
  %2862 = bitcast float %2861 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2862, ptr addrspace(8) %79, i32 %2859, i32 0, i32 0)
  %2863 = add i32 %2459, 112
  %2864 = add i32 %2863, %51
  %2865 = extractelement <8 x float> %1495, i64 3
  %2866 = mul i32 %2864, 4
  %2867 = bitcast float %2865 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2867, ptr addrspace(8) %78, i32 %2866, i32 0, i32 0)
  %2868 = extractelement <8 x float> %1511, i64 3
  %2869 = bitcast float %2868 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2869, ptr addrspace(8) %79, i32 %2866, i32 0, i32 0)
  %2870 = add i32 %2469, 112
  %2871 = add i32 %2870, %51
  %2872 = extractelement <8 x float> %1495, i64 4
  %2873 = mul i32 %2871, 4
  %2874 = bitcast float %2872 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2874, ptr addrspace(8) %78, i32 %2873, i32 0, i32 0)
  %2875 = extractelement <8 x float> %1511, i64 4
  %2876 = bitcast float %2875 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2876, ptr addrspace(8) %79, i32 %2873, i32 0, i32 0)
  %2877 = add i32 %2479, 112
  %2878 = add i32 %2877, %51
  %2879 = extractelement <8 x float> %1495, i64 5
  %2880 = mul i32 %2878, 4
  %2881 = bitcast float %2879 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2881, ptr addrspace(8) %78, i32 %2880, i32 0, i32 0)
  %2882 = extractelement <8 x float> %1511, i64 5
  %2883 = bitcast float %2882 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2883, ptr addrspace(8) %79, i32 %2880, i32 0, i32 0)
  %2884 = add i32 %2489, 112
  %2885 = add i32 %2884, %51
  %2886 = extractelement <8 x float> %1495, i64 6
  %2887 = mul i32 %2885, 4
  %2888 = bitcast float %2886 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2888, ptr addrspace(8) %78, i32 %2887, i32 0, i32 0)
  %2889 = extractelement <8 x float> %1511, i64 6
  %2890 = bitcast float %2889 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2890, ptr addrspace(8) %79, i32 %2887, i32 0, i32 0)
  %2891 = add i32 %2499, 112
  %2892 = add i32 %2891, %51
  %2893 = extractelement <8 x float> %1495, i64 7
  %2894 = mul i32 %2892, 4
  %2895 = bitcast float %2893 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2895, ptr addrspace(8) %78, i32 %2894, i32 0, i32 0)
  %2896 = extractelement <8 x float> %1511, i64 7
  %2897 = bitcast float %2896 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2897, ptr addrspace(8) %79, i32 %2894, i32 0, i32 0)
  %2898 = add i32 %126, %206
  %2899 = mul i32 %2898, %20
  %2900 = mul i32 %2899, 128
  %2901 = add i32 %2425, %2900
  %2902 = add i32 %2901, %51
  %2903 = extractelement <8 x float> %1496, i64 0
  %2904 = mul i32 %2902, 4
  %2905 = bitcast float %2903 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2905, ptr addrspace(8) %78, i32 %2904, i32 0, i32 0)
  %2906 = extractelement <8 x float> %1512, i64 0
  %2907 = bitcast float %2906 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2907, ptr addrspace(8) %79, i32 %2904, i32 0, i32 0)
  %2908 = add i32 %2898, 1
  %2909 = mul i32 %2908, %20
  %2910 = mul i32 %2909, 128
  %2911 = add i32 %2425, %2910
  %2912 = add i32 %2911, %51
  %2913 = extractelement <8 x float> %1496, i64 1
  %2914 = mul i32 %2912, 4
  %2915 = bitcast float %2913 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2915, ptr addrspace(8) %78, i32 %2914, i32 0, i32 0)
  %2916 = extractelement <8 x float> %1512, i64 1
  %2917 = bitcast float %2916 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2917, ptr addrspace(8) %79, i32 %2914, i32 0, i32 0)
  %2918 = add i32 %2898, 2
  %2919 = mul i32 %2918, %20
  %2920 = mul i32 %2919, 128
  %2921 = add i32 %2425, %2920
  %2922 = add i32 %2921, %51
  %2923 = extractelement <8 x float> %1496, i64 2
  %2924 = mul i32 %2922, 4
  %2925 = bitcast float %2923 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2925, ptr addrspace(8) %78, i32 %2924, i32 0, i32 0)
  %2926 = extractelement <8 x float> %1512, i64 2
  %2927 = bitcast float %2926 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2927, ptr addrspace(8) %79, i32 %2924, i32 0, i32 0)
  %2928 = add i32 %2898, 3
  %2929 = mul i32 %2928, %20
  %2930 = mul i32 %2929, 128
  %2931 = add i32 %2425, %2930
  %2932 = add i32 %2931, %51
  %2933 = extractelement <8 x float> %1496, i64 3
  %2934 = mul i32 %2932, 4
  %2935 = bitcast float %2933 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2935, ptr addrspace(8) %78, i32 %2934, i32 0, i32 0)
  %2936 = extractelement <8 x float> %1512, i64 3
  %2937 = bitcast float %2936 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2937, ptr addrspace(8) %79, i32 %2934, i32 0, i32 0)
  %2938 = add i32 %2898, 4
  %2939 = mul i32 %2938, %20
  %2940 = mul i32 %2939, 128
  %2941 = add i32 %2425, %2940
  %2942 = add i32 %2941, %51
  %2943 = extractelement <8 x float> %1496, i64 4
  %2944 = mul i32 %2942, 4
  %2945 = bitcast float %2943 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2945, ptr addrspace(8) %78, i32 %2944, i32 0, i32 0)
  %2946 = extractelement <8 x float> %1512, i64 4
  %2947 = bitcast float %2946 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2947, ptr addrspace(8) %79, i32 %2944, i32 0, i32 0)
  %2948 = add i32 %2898, 5
  %2949 = mul i32 %2948, %20
  %2950 = mul i32 %2949, 128
  %2951 = add i32 %2425, %2950
  %2952 = add i32 %2951, %51
  %2953 = extractelement <8 x float> %1496, i64 5
  %2954 = mul i32 %2952, 4
  %2955 = bitcast float %2953 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2955, ptr addrspace(8) %78, i32 %2954, i32 0, i32 0)
  %2956 = extractelement <8 x float> %1512, i64 5
  %2957 = bitcast float %2956 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2957, ptr addrspace(8) %79, i32 %2954, i32 0, i32 0)
  %2958 = add i32 %2898, 6
  %2959 = mul i32 %2958, %20
  %2960 = mul i32 %2959, 128
  %2961 = add i32 %2425, %2960
  %2962 = add i32 %2961, %51
  %2963 = extractelement <8 x float> %1496, i64 6
  %2964 = mul i32 %2962, 4
  %2965 = bitcast float %2963 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2965, ptr addrspace(8) %78, i32 %2964, i32 0, i32 0)
  %2966 = extractelement <8 x float> %1512, i64 6
  %2967 = bitcast float %2966 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2967, ptr addrspace(8) %79, i32 %2964, i32 0, i32 0)
  %2968 = add i32 %2898, 7
  %2969 = mul i32 %2968, %20
  %2970 = mul i32 %2969, 128
  %2971 = add i32 %2425, %2970
  %2972 = add i32 %2971, %51
  %2973 = extractelement <8 x float> %1496, i64 7
  %2974 = mul i32 %2972, 4
  %2975 = bitcast float %2973 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2975, ptr addrspace(8) %78, i32 %2974, i32 0, i32 0)
  %2976 = extractelement <8 x float> %1512, i64 7
  %2977 = bitcast float %2976 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2977, ptr addrspace(8) %79, i32 %2974, i32 0, i32 0)
  %2978 = add i32 %2901, 16
  %2979 = add i32 %2978, %51
  %2980 = extractelement <8 x float> %1497, i64 0
  %2981 = mul i32 %2979, 4
  %2982 = bitcast float %2980 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2982, ptr addrspace(8) %78, i32 %2981, i32 0, i32 0)
  %2983 = extractelement <8 x float> %1513, i64 0
  %2984 = bitcast float %2983 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2984, ptr addrspace(8) %79, i32 %2981, i32 0, i32 0)
  %2985 = add i32 %2911, 16
  %2986 = add i32 %2985, %51
  %2987 = extractelement <8 x float> %1497, i64 1
  %2988 = mul i32 %2986, 4
  %2989 = bitcast float %2987 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2989, ptr addrspace(8) %78, i32 %2988, i32 0, i32 0)
  %2990 = extractelement <8 x float> %1513, i64 1
  %2991 = bitcast float %2990 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2991, ptr addrspace(8) %79, i32 %2988, i32 0, i32 0)
  %2992 = add i32 %2921, 16
  %2993 = add i32 %2992, %51
  %2994 = extractelement <8 x float> %1497, i64 2
  %2995 = mul i32 %2993, 4
  %2996 = bitcast float %2994 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2996, ptr addrspace(8) %78, i32 %2995, i32 0, i32 0)
  %2997 = extractelement <8 x float> %1513, i64 2
  %2998 = bitcast float %2997 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %2998, ptr addrspace(8) %79, i32 %2995, i32 0, i32 0)
  %2999 = add i32 %2931, 16
  %3000 = add i32 %2999, %51
  %3001 = extractelement <8 x float> %1497, i64 3
  %3002 = mul i32 %3000, 4
  %3003 = bitcast float %3001 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3003, ptr addrspace(8) %78, i32 %3002, i32 0, i32 0)
  %3004 = extractelement <8 x float> %1513, i64 3
  %3005 = bitcast float %3004 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3005, ptr addrspace(8) %79, i32 %3002, i32 0, i32 0)
  %3006 = add i32 %2941, 16
  %3007 = add i32 %3006, %51
  %3008 = extractelement <8 x float> %1497, i64 4
  %3009 = mul i32 %3007, 4
  %3010 = bitcast float %3008 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3010, ptr addrspace(8) %78, i32 %3009, i32 0, i32 0)
  %3011 = extractelement <8 x float> %1513, i64 4
  %3012 = bitcast float %3011 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3012, ptr addrspace(8) %79, i32 %3009, i32 0, i32 0)
  %3013 = add i32 %2951, 16
  %3014 = add i32 %3013, %51
  %3015 = extractelement <8 x float> %1497, i64 5
  %3016 = mul i32 %3014, 4
  %3017 = bitcast float %3015 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3017, ptr addrspace(8) %78, i32 %3016, i32 0, i32 0)
  %3018 = extractelement <8 x float> %1513, i64 5
  %3019 = bitcast float %3018 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3019, ptr addrspace(8) %79, i32 %3016, i32 0, i32 0)
  %3020 = add i32 %2961, 16
  %3021 = add i32 %3020, %51
  %3022 = extractelement <8 x float> %1497, i64 6
  %3023 = mul i32 %3021, 4
  %3024 = bitcast float %3022 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3024, ptr addrspace(8) %78, i32 %3023, i32 0, i32 0)
  %3025 = extractelement <8 x float> %1513, i64 6
  %3026 = bitcast float %3025 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3026, ptr addrspace(8) %79, i32 %3023, i32 0, i32 0)
  %3027 = add i32 %2971, 16
  %3028 = add i32 %3027, %51
  %3029 = extractelement <8 x float> %1497, i64 7
  %3030 = mul i32 %3028, 4
  %3031 = bitcast float %3029 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3031, ptr addrspace(8) %78, i32 %3030, i32 0, i32 0)
  %3032 = extractelement <8 x float> %1513, i64 7
  %3033 = bitcast float %3032 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3033, ptr addrspace(8) %79, i32 %3030, i32 0, i32 0)
  %3034 = add i32 %2901, 32
  %3035 = add i32 %3034, %51
  %3036 = extractelement <8 x float> %1498, i64 0
  %3037 = mul i32 %3035, 4
  %3038 = bitcast float %3036 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3038, ptr addrspace(8) %78, i32 %3037, i32 0, i32 0)
  %3039 = extractelement <8 x float> %1514, i64 0
  %3040 = bitcast float %3039 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3040, ptr addrspace(8) %79, i32 %3037, i32 0, i32 0)
  %3041 = add i32 %2911, 32
  %3042 = add i32 %3041, %51
  %3043 = extractelement <8 x float> %1498, i64 1
  %3044 = mul i32 %3042, 4
  %3045 = bitcast float %3043 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3045, ptr addrspace(8) %78, i32 %3044, i32 0, i32 0)
  %3046 = extractelement <8 x float> %1514, i64 1
  %3047 = bitcast float %3046 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3047, ptr addrspace(8) %79, i32 %3044, i32 0, i32 0)
  %3048 = add i32 %2921, 32
  %3049 = add i32 %3048, %51
  %3050 = extractelement <8 x float> %1498, i64 2
  %3051 = mul i32 %3049, 4
  %3052 = bitcast float %3050 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3052, ptr addrspace(8) %78, i32 %3051, i32 0, i32 0)
  %3053 = extractelement <8 x float> %1514, i64 2
  %3054 = bitcast float %3053 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3054, ptr addrspace(8) %79, i32 %3051, i32 0, i32 0)
  %3055 = add i32 %2931, 32
  %3056 = add i32 %3055, %51
  %3057 = extractelement <8 x float> %1498, i64 3
  %3058 = mul i32 %3056, 4
  %3059 = bitcast float %3057 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3059, ptr addrspace(8) %78, i32 %3058, i32 0, i32 0)
  %3060 = extractelement <8 x float> %1514, i64 3
  %3061 = bitcast float %3060 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3061, ptr addrspace(8) %79, i32 %3058, i32 0, i32 0)
  %3062 = add i32 %2941, 32
  %3063 = add i32 %3062, %51
  %3064 = extractelement <8 x float> %1498, i64 4
  %3065 = mul i32 %3063, 4
  %3066 = bitcast float %3064 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3066, ptr addrspace(8) %78, i32 %3065, i32 0, i32 0)
  %3067 = extractelement <8 x float> %1514, i64 4
  %3068 = bitcast float %3067 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3068, ptr addrspace(8) %79, i32 %3065, i32 0, i32 0)
  %3069 = add i32 %2951, 32
  %3070 = add i32 %3069, %51
  %3071 = extractelement <8 x float> %1498, i64 5
  %3072 = mul i32 %3070, 4
  %3073 = bitcast float %3071 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3073, ptr addrspace(8) %78, i32 %3072, i32 0, i32 0)
  %3074 = extractelement <8 x float> %1514, i64 5
  %3075 = bitcast float %3074 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3075, ptr addrspace(8) %79, i32 %3072, i32 0, i32 0)
  %3076 = add i32 %2961, 32
  %3077 = add i32 %3076, %51
  %3078 = extractelement <8 x float> %1498, i64 6
  %3079 = mul i32 %3077, 4
  %3080 = bitcast float %3078 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3080, ptr addrspace(8) %78, i32 %3079, i32 0, i32 0)
  %3081 = extractelement <8 x float> %1514, i64 6
  %3082 = bitcast float %3081 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3082, ptr addrspace(8) %79, i32 %3079, i32 0, i32 0)
  %3083 = add i32 %2971, 32
  %3084 = add i32 %3083, %51
  %3085 = extractelement <8 x float> %1498, i64 7
  %3086 = mul i32 %3084, 4
  %3087 = bitcast float %3085 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3087, ptr addrspace(8) %78, i32 %3086, i32 0, i32 0)
  %3088 = extractelement <8 x float> %1514, i64 7
  %3089 = bitcast float %3088 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3089, ptr addrspace(8) %79, i32 %3086, i32 0, i32 0)
  %3090 = add i32 %2901, 48
  %3091 = add i32 %3090, %51
  %3092 = extractelement <8 x float> %1499, i64 0
  %3093 = mul i32 %3091, 4
  %3094 = bitcast float %3092 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3094, ptr addrspace(8) %78, i32 %3093, i32 0, i32 0)
  %3095 = extractelement <8 x float> %1515, i64 0
  %3096 = bitcast float %3095 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3096, ptr addrspace(8) %79, i32 %3093, i32 0, i32 0)
  %3097 = add i32 %2911, 48
  %3098 = add i32 %3097, %51
  %3099 = extractelement <8 x float> %1499, i64 1
  %3100 = mul i32 %3098, 4
  %3101 = bitcast float %3099 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3101, ptr addrspace(8) %78, i32 %3100, i32 0, i32 0)
  %3102 = extractelement <8 x float> %1515, i64 1
  %3103 = bitcast float %3102 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3103, ptr addrspace(8) %79, i32 %3100, i32 0, i32 0)
  %3104 = add i32 %2921, 48
  %3105 = add i32 %3104, %51
  %3106 = extractelement <8 x float> %1499, i64 2
  %3107 = mul i32 %3105, 4
  %3108 = bitcast float %3106 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3108, ptr addrspace(8) %78, i32 %3107, i32 0, i32 0)
  %3109 = extractelement <8 x float> %1515, i64 2
  %3110 = bitcast float %3109 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3110, ptr addrspace(8) %79, i32 %3107, i32 0, i32 0)
  %3111 = add i32 %2931, 48
  %3112 = add i32 %3111, %51
  %3113 = extractelement <8 x float> %1499, i64 3
  %3114 = mul i32 %3112, 4
  %3115 = bitcast float %3113 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3115, ptr addrspace(8) %78, i32 %3114, i32 0, i32 0)
  %3116 = extractelement <8 x float> %1515, i64 3
  %3117 = bitcast float %3116 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3117, ptr addrspace(8) %79, i32 %3114, i32 0, i32 0)
  %3118 = add i32 %2941, 48
  %3119 = add i32 %3118, %51
  %3120 = extractelement <8 x float> %1499, i64 4
  %3121 = mul i32 %3119, 4
  %3122 = bitcast float %3120 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3122, ptr addrspace(8) %78, i32 %3121, i32 0, i32 0)
  %3123 = extractelement <8 x float> %1515, i64 4
  %3124 = bitcast float %3123 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3124, ptr addrspace(8) %79, i32 %3121, i32 0, i32 0)
  %3125 = add i32 %2951, 48
  %3126 = add i32 %3125, %51
  %3127 = extractelement <8 x float> %1499, i64 5
  %3128 = mul i32 %3126, 4
  %3129 = bitcast float %3127 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3129, ptr addrspace(8) %78, i32 %3128, i32 0, i32 0)
  %3130 = extractelement <8 x float> %1515, i64 5
  %3131 = bitcast float %3130 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3131, ptr addrspace(8) %79, i32 %3128, i32 0, i32 0)
  %3132 = add i32 %2961, 48
  %3133 = add i32 %3132, %51
  %3134 = extractelement <8 x float> %1499, i64 6
  %3135 = mul i32 %3133, 4
  %3136 = bitcast float %3134 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3136, ptr addrspace(8) %78, i32 %3135, i32 0, i32 0)
  %3137 = extractelement <8 x float> %1515, i64 6
  %3138 = bitcast float %3137 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3138, ptr addrspace(8) %79, i32 %3135, i32 0, i32 0)
  %3139 = add i32 %2971, 48
  %3140 = add i32 %3139, %51
  %3141 = extractelement <8 x float> %1499, i64 7
  %3142 = mul i32 %3140, 4
  %3143 = bitcast float %3141 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3143, ptr addrspace(8) %78, i32 %3142, i32 0, i32 0)
  %3144 = extractelement <8 x float> %1515, i64 7
  %3145 = bitcast float %3144 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3145, ptr addrspace(8) %79, i32 %3142, i32 0, i32 0)
  %3146 = add i32 %2901, 64
  %3147 = add i32 %3146, %51
  %3148 = extractelement <8 x float> %1500, i64 0
  %3149 = mul i32 %3147, 4
  %3150 = bitcast float %3148 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3150, ptr addrspace(8) %78, i32 %3149, i32 0, i32 0)
  %3151 = extractelement <8 x float> %1516, i64 0
  %3152 = bitcast float %3151 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3152, ptr addrspace(8) %79, i32 %3149, i32 0, i32 0)
  %3153 = add i32 %2911, 64
  %3154 = add i32 %3153, %51
  %3155 = extractelement <8 x float> %1500, i64 1
  %3156 = mul i32 %3154, 4
  %3157 = bitcast float %3155 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3157, ptr addrspace(8) %78, i32 %3156, i32 0, i32 0)
  %3158 = extractelement <8 x float> %1516, i64 1
  %3159 = bitcast float %3158 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3159, ptr addrspace(8) %79, i32 %3156, i32 0, i32 0)
  %3160 = add i32 %2921, 64
  %3161 = add i32 %3160, %51
  %3162 = extractelement <8 x float> %1500, i64 2
  %3163 = mul i32 %3161, 4
  %3164 = bitcast float %3162 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3164, ptr addrspace(8) %78, i32 %3163, i32 0, i32 0)
  %3165 = extractelement <8 x float> %1516, i64 2
  %3166 = bitcast float %3165 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3166, ptr addrspace(8) %79, i32 %3163, i32 0, i32 0)
  %3167 = add i32 %2931, 64
  %3168 = add i32 %3167, %51
  %3169 = extractelement <8 x float> %1500, i64 3
  %3170 = mul i32 %3168, 4
  %3171 = bitcast float %3169 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3171, ptr addrspace(8) %78, i32 %3170, i32 0, i32 0)
  %3172 = extractelement <8 x float> %1516, i64 3
  %3173 = bitcast float %3172 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3173, ptr addrspace(8) %79, i32 %3170, i32 0, i32 0)
  %3174 = add i32 %2941, 64
  %3175 = add i32 %3174, %51
  %3176 = extractelement <8 x float> %1500, i64 4
  %3177 = mul i32 %3175, 4
  %3178 = bitcast float %3176 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3178, ptr addrspace(8) %78, i32 %3177, i32 0, i32 0)
  %3179 = extractelement <8 x float> %1516, i64 4
  %3180 = bitcast float %3179 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3180, ptr addrspace(8) %79, i32 %3177, i32 0, i32 0)
  %3181 = add i32 %2951, 64
  %3182 = add i32 %3181, %51
  %3183 = extractelement <8 x float> %1500, i64 5
  %3184 = mul i32 %3182, 4
  %3185 = bitcast float %3183 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3185, ptr addrspace(8) %78, i32 %3184, i32 0, i32 0)
  %3186 = extractelement <8 x float> %1516, i64 5
  %3187 = bitcast float %3186 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3187, ptr addrspace(8) %79, i32 %3184, i32 0, i32 0)
  %3188 = add i32 %2961, 64
  %3189 = add i32 %3188, %51
  %3190 = extractelement <8 x float> %1500, i64 6
  %3191 = mul i32 %3189, 4
  %3192 = bitcast float %3190 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3192, ptr addrspace(8) %78, i32 %3191, i32 0, i32 0)
  %3193 = extractelement <8 x float> %1516, i64 6
  %3194 = bitcast float %3193 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3194, ptr addrspace(8) %79, i32 %3191, i32 0, i32 0)
  %3195 = add i32 %2971, 64
  %3196 = add i32 %3195, %51
  %3197 = extractelement <8 x float> %1500, i64 7
  %3198 = mul i32 %3196, 4
  %3199 = bitcast float %3197 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3199, ptr addrspace(8) %78, i32 %3198, i32 0, i32 0)
  %3200 = extractelement <8 x float> %1516, i64 7
  %3201 = bitcast float %3200 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3201, ptr addrspace(8) %79, i32 %3198, i32 0, i32 0)
  %3202 = add i32 %2901, 80
  %3203 = add i32 %3202, %51
  %3204 = extractelement <8 x float> %1501, i64 0
  %3205 = mul i32 %3203, 4
  %3206 = bitcast float %3204 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3206, ptr addrspace(8) %78, i32 %3205, i32 0, i32 0)
  %3207 = extractelement <8 x float> %1517, i64 0
  %3208 = bitcast float %3207 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3208, ptr addrspace(8) %79, i32 %3205, i32 0, i32 0)
  %3209 = add i32 %2911, 80
  %3210 = add i32 %3209, %51
  %3211 = extractelement <8 x float> %1501, i64 1
  %3212 = mul i32 %3210, 4
  %3213 = bitcast float %3211 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3213, ptr addrspace(8) %78, i32 %3212, i32 0, i32 0)
  %3214 = extractelement <8 x float> %1517, i64 1
  %3215 = bitcast float %3214 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3215, ptr addrspace(8) %79, i32 %3212, i32 0, i32 0)
  %3216 = add i32 %2921, 80
  %3217 = add i32 %3216, %51
  %3218 = extractelement <8 x float> %1501, i64 2
  %3219 = mul i32 %3217, 4
  %3220 = bitcast float %3218 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3220, ptr addrspace(8) %78, i32 %3219, i32 0, i32 0)
  %3221 = extractelement <8 x float> %1517, i64 2
  %3222 = bitcast float %3221 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3222, ptr addrspace(8) %79, i32 %3219, i32 0, i32 0)
  %3223 = add i32 %2931, 80
  %3224 = add i32 %3223, %51
  %3225 = extractelement <8 x float> %1501, i64 3
  %3226 = mul i32 %3224, 4
  %3227 = bitcast float %3225 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3227, ptr addrspace(8) %78, i32 %3226, i32 0, i32 0)
  %3228 = extractelement <8 x float> %1517, i64 3
  %3229 = bitcast float %3228 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3229, ptr addrspace(8) %79, i32 %3226, i32 0, i32 0)
  %3230 = add i32 %2941, 80
  %3231 = add i32 %3230, %51
  %3232 = extractelement <8 x float> %1501, i64 4
  %3233 = mul i32 %3231, 4
  %3234 = bitcast float %3232 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3234, ptr addrspace(8) %78, i32 %3233, i32 0, i32 0)
  %3235 = extractelement <8 x float> %1517, i64 4
  %3236 = bitcast float %3235 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3236, ptr addrspace(8) %79, i32 %3233, i32 0, i32 0)
  %3237 = add i32 %2951, 80
  %3238 = add i32 %3237, %51
  %3239 = extractelement <8 x float> %1501, i64 5
  %3240 = mul i32 %3238, 4
  %3241 = bitcast float %3239 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3241, ptr addrspace(8) %78, i32 %3240, i32 0, i32 0)
  %3242 = extractelement <8 x float> %1517, i64 5
  %3243 = bitcast float %3242 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3243, ptr addrspace(8) %79, i32 %3240, i32 0, i32 0)
  %3244 = add i32 %2961, 80
  %3245 = add i32 %3244, %51
  %3246 = extractelement <8 x float> %1501, i64 6
  %3247 = mul i32 %3245, 4
  %3248 = bitcast float %3246 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3248, ptr addrspace(8) %78, i32 %3247, i32 0, i32 0)
  %3249 = extractelement <8 x float> %1517, i64 6
  %3250 = bitcast float %3249 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3250, ptr addrspace(8) %79, i32 %3247, i32 0, i32 0)
  %3251 = add i32 %2971, 80
  %3252 = add i32 %3251, %51
  %3253 = extractelement <8 x float> %1501, i64 7
  %3254 = mul i32 %3252, 4
  %3255 = bitcast float %3253 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3255, ptr addrspace(8) %78, i32 %3254, i32 0, i32 0)
  %3256 = extractelement <8 x float> %1517, i64 7
  %3257 = bitcast float %3256 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3257, ptr addrspace(8) %79, i32 %3254, i32 0, i32 0)
  %3258 = add i32 %2901, 96
  %3259 = add i32 %3258, %51
  %3260 = extractelement <8 x float> %1502, i64 0
  %3261 = mul i32 %3259, 4
  %3262 = bitcast float %3260 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3262, ptr addrspace(8) %78, i32 %3261, i32 0, i32 0)
  %3263 = extractelement <8 x float> %1518, i64 0
  %3264 = bitcast float %3263 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3264, ptr addrspace(8) %79, i32 %3261, i32 0, i32 0)
  %3265 = add i32 %2911, 96
  %3266 = add i32 %3265, %51
  %3267 = extractelement <8 x float> %1502, i64 1
  %3268 = mul i32 %3266, 4
  %3269 = bitcast float %3267 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3269, ptr addrspace(8) %78, i32 %3268, i32 0, i32 0)
  %3270 = extractelement <8 x float> %1518, i64 1
  %3271 = bitcast float %3270 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3271, ptr addrspace(8) %79, i32 %3268, i32 0, i32 0)
  %3272 = add i32 %2921, 96
  %3273 = add i32 %3272, %51
  %3274 = extractelement <8 x float> %1502, i64 2
  %3275 = mul i32 %3273, 4
  %3276 = bitcast float %3274 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3276, ptr addrspace(8) %78, i32 %3275, i32 0, i32 0)
  %3277 = extractelement <8 x float> %1518, i64 2
  %3278 = bitcast float %3277 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3278, ptr addrspace(8) %79, i32 %3275, i32 0, i32 0)
  %3279 = add i32 %2931, 96
  %3280 = add i32 %3279, %51
  %3281 = extractelement <8 x float> %1502, i64 3
  %3282 = mul i32 %3280, 4
  %3283 = bitcast float %3281 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3283, ptr addrspace(8) %78, i32 %3282, i32 0, i32 0)
  %3284 = extractelement <8 x float> %1518, i64 3
  %3285 = bitcast float %3284 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3285, ptr addrspace(8) %79, i32 %3282, i32 0, i32 0)
  %3286 = add i32 %2941, 96
  %3287 = add i32 %3286, %51
  %3288 = extractelement <8 x float> %1502, i64 4
  %3289 = mul i32 %3287, 4
  %3290 = bitcast float %3288 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3290, ptr addrspace(8) %78, i32 %3289, i32 0, i32 0)
  %3291 = extractelement <8 x float> %1518, i64 4
  %3292 = bitcast float %3291 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3292, ptr addrspace(8) %79, i32 %3289, i32 0, i32 0)
  %3293 = add i32 %2951, 96
  %3294 = add i32 %3293, %51
  %3295 = extractelement <8 x float> %1502, i64 5
  %3296 = mul i32 %3294, 4
  %3297 = bitcast float %3295 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3297, ptr addrspace(8) %78, i32 %3296, i32 0, i32 0)
  %3298 = extractelement <8 x float> %1518, i64 5
  %3299 = bitcast float %3298 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3299, ptr addrspace(8) %79, i32 %3296, i32 0, i32 0)
  %3300 = add i32 %2961, 96
  %3301 = add i32 %3300, %51
  %3302 = extractelement <8 x float> %1502, i64 6
  %3303 = mul i32 %3301, 4
  %3304 = bitcast float %3302 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3304, ptr addrspace(8) %78, i32 %3303, i32 0, i32 0)
  %3305 = extractelement <8 x float> %1518, i64 6
  %3306 = bitcast float %3305 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3306, ptr addrspace(8) %79, i32 %3303, i32 0, i32 0)
  %3307 = add i32 %2971, 96
  %3308 = add i32 %3307, %51
  %3309 = extractelement <8 x float> %1502, i64 7
  %3310 = mul i32 %3308, 4
  %3311 = bitcast float %3309 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3311, ptr addrspace(8) %78, i32 %3310, i32 0, i32 0)
  %3312 = extractelement <8 x float> %1518, i64 7
  %3313 = bitcast float %3312 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3313, ptr addrspace(8) %79, i32 %3310, i32 0, i32 0)
  %3314 = add i32 %2901, 112
  %3315 = add i32 %3314, %51
  %3316 = extractelement <8 x float> %1503, i64 0
  %3317 = mul i32 %3315, 4
  %3318 = bitcast float %3316 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3318, ptr addrspace(8) %78, i32 %3317, i32 0, i32 0)
  %3319 = extractelement <8 x float> %1519, i64 0
  %3320 = bitcast float %3319 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3320, ptr addrspace(8) %79, i32 %3317, i32 0, i32 0)
  %3321 = add i32 %2911, 112
  %3322 = add i32 %3321, %51
  %3323 = extractelement <8 x float> %1503, i64 1
  %3324 = mul i32 %3322, 4
  %3325 = bitcast float %3323 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3325, ptr addrspace(8) %78, i32 %3324, i32 0, i32 0)
  %3326 = extractelement <8 x float> %1519, i64 1
  %3327 = bitcast float %3326 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3327, ptr addrspace(8) %79, i32 %3324, i32 0, i32 0)
  %3328 = add i32 %2921, 112
  %3329 = add i32 %3328, %51
  %3330 = extractelement <8 x float> %1503, i64 2
  %3331 = mul i32 %3329, 4
  %3332 = bitcast float %3330 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3332, ptr addrspace(8) %78, i32 %3331, i32 0, i32 0)
  %3333 = extractelement <8 x float> %1519, i64 2
  %3334 = bitcast float %3333 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3334, ptr addrspace(8) %79, i32 %3331, i32 0, i32 0)
  %3335 = add i32 %2931, 112
  %3336 = add i32 %3335, %51
  %3337 = extractelement <8 x float> %1503, i64 3
  %3338 = mul i32 %3336, 4
  %3339 = bitcast float %3337 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3339, ptr addrspace(8) %78, i32 %3338, i32 0, i32 0)
  %3340 = extractelement <8 x float> %1519, i64 3
  %3341 = bitcast float %3340 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3341, ptr addrspace(8) %79, i32 %3338, i32 0, i32 0)
  %3342 = add i32 %2941, 112
  %3343 = add i32 %3342, %51
  %3344 = extractelement <8 x float> %1503, i64 4
  %3345 = mul i32 %3343, 4
  %3346 = bitcast float %3344 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3346, ptr addrspace(8) %78, i32 %3345, i32 0, i32 0)
  %3347 = extractelement <8 x float> %1519, i64 4
  %3348 = bitcast float %3347 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3348, ptr addrspace(8) %79, i32 %3345, i32 0, i32 0)
  %3349 = add i32 %2951, 112
  %3350 = add i32 %3349, %51
  %3351 = extractelement <8 x float> %1503, i64 5
  %3352 = mul i32 %3350, 4
  %3353 = bitcast float %3351 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3353, ptr addrspace(8) %78, i32 %3352, i32 0, i32 0)
  %3354 = extractelement <8 x float> %1519, i64 5
  %3355 = bitcast float %3354 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3355, ptr addrspace(8) %79, i32 %3352, i32 0, i32 0)
  %3356 = add i32 %2961, 112
  %3357 = add i32 %3356, %51
  %3358 = extractelement <8 x float> %1503, i64 6
  %3359 = mul i32 %3357, 4
  %3360 = bitcast float %3358 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3360, ptr addrspace(8) %78, i32 %3359, i32 0, i32 0)
  %3361 = extractelement <8 x float> %1519, i64 6
  %3362 = bitcast float %3361 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3362, ptr addrspace(8) %79, i32 %3359, i32 0, i32 0)
  %3363 = add i32 %2971, 112
  %3364 = add i32 %3363, %51
  %3365 = extractelement <8 x float> %1503, i64 7
  %3366 = mul i32 %3364, 4
  %3367 = bitcast float %3365 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3367, ptr addrspace(8) %78, i32 %3366, i32 0, i32 0)
  %3368 = extractelement <8 x float> %1519, i64 7
  %3369 = bitcast float %3368 to i32
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %3369, ptr addrspace(8) %79, i32 %3366, i32 0, i32 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone, i16, i64, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #5

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

; Function Attrs: nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.amdgcn.s.wait.tensorcnt(i16 immarg) #7

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #8

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat>, <16 x bfloat>, i16 immarg, <8 x float>, i1 immarg, i1 immarg) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: read)
declare <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) captures(none)) #9

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

attributes #0 = { "amdgpu-flat-work-group-size"="32,32" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #8 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #9 = { convergent nocallback nofree nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 32, i32 1, i32 1}
