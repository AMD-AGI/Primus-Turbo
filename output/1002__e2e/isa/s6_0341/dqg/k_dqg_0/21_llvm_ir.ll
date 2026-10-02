; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__shared_alloc_0 = external dso_local addrspace(3) global [52224 x i8], align 16

define amdgpu_kernel void @k_dqg_0(ptr addrspace(1) %0, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %5, ptr addrspace(1) %6, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %7, ptr addrspace(1) %8, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %9, ptr addrspace(1) %10, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %11, ptr addrspace(1) %12, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %13, ptr addrspace(1) %14, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %15, float %16, i32 %17, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24) #0 !reqd_work_group_size !1 {
  %26 = call range(i32 0, 32) i32 @llvm.amdgcn.workitem.id.x()
  %27 = sext i32 %26 to i64
  %28 = trunc i64 %27 to i32
  %29 = call i32 @llvm.amdgcn.workgroup.id.x()
  %30 = sext i32 %29 to i64
  %31 = trunc i64 %30 to i32
  %32 = srem i32 %19, 8
  %33 = icmp eq i32 %32, 0
  %34 = srem i32 %31, 8
  %35 = sdiv i32 %19, 8
  %36 = mul i32 %35, 8
  %37 = icmp ne i32 %19, %36
  %38 = icmp slt i32 %19, 0
  %39 = icmp ne i1 %38, false
  %40 = and i1 %37, %39
  %41 = add i32 %35, -1
  %42 = select i1 %40, i32 %41, i32 %35
  %43 = mul i32 %34, %42
  %44 = sdiv i32 %31, 8
  %45 = mul i32 %44, 8
  %46 = icmp ne i32 %31, %45
  %47 = icmp slt i32 %31, 0
  %48 = icmp ne i1 %47, false
  %49 = and i1 %46, %48
  %50 = add i32 %44, -1
  %51 = select i1 %49, i32 %50, i32 %44
  %52 = add i32 %43, %51
  %53 = select i1 %33, i32 %52, i32 %31
  %54 = sdiv i32 %17, 64
  %55 = mul i32 %54, 64
  %56 = icmp ne i32 %17, %55
  %57 = icmp slt i32 %17, 0
  %58 = icmp ne i1 %57, false
  %59 = and i1 %56, %58
  %60 = add i32 %54, -1
  %61 = select i1 %59, i32 %60, i32 %54
  %62 = sub i32 %61, 1
  %63 = call i32 @llvm.amdgcn.workgroup.id.y()
  %64 = sext i32 %63 to i64
  %65 = trunc i64 %64 to i32
  %66 = sub i32 %62, %65
  %67 = call i32 @llvm.amdgcn.workgroup.id.z()
  %68 = sext i32 %67 to i64
  %69 = trunc i64 %68 to i32
  %70 = srem i32 %28, 16
  %71 = sdiv i32 %28, 16
  %72 = mul i32 %71, 16
  %73 = icmp ne i32 %28, %72
  %74 = icmp slt i32 %28, 0
  %75 = icmp ne i1 %74, false
  %76 = and i1 %73, %75
  %77 = add i32 %71, -1
  %78 = select i1 %76, i32 %77, i32 %71
  %79 = mul i32 %66, 64
  %80 = sdiv i32 %53, %21
  %81 = mul i32 %80, %21
  %82 = icmp ne i32 %53, %81
  %83 = icmp slt i32 %53, 0
  %84 = icmp slt i32 %21, 0
  %85 = icmp ne i1 %83, %84
  %86 = and i1 %82, %85
  %87 = add i32 %80, -1
  %88 = select i1 %86, i32 %87, i32 %80
  %89 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %0, i16 0, i64 1073741824, i32 159744)
  %90 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %6, i16 0, i64 1073741824, i32 159744)
  %91 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %10, i16 0, i64 268435456, i32 159744)
  %92 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %12, i16 0, i64 268435456, i32 159744)
  %93 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %14, i16 0, i64 1073741824, i32 159744)
  %94 = mul i32 %19, 16
  %95 = mul i32 %69, %17
  %96 = mul i32 %95, %94
  %97 = mul i32 %53, 16
  %98 = add i32 %96, %97
  %99 = mul i32 %69, %19
  %100 = add i32 %99, %53
  %101 = mul i32 %100, %17
  %102 = add i32 %79, %70
  %103 = mul i32 %102, %94
  %104 = add i32 %98, %103
  %105 = add i32 %104, %78
  %106 = mul i32 %105, 16
  %107 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %106, i32 0, i32 0)
  %108 = bitcast i128 %107 to <8 x bfloat>
  %109 = add i32 %105, 2
  %110 = mul i32 %109, 16
  %111 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %110, i32 0, i32 0)
  %112 = bitcast i128 %111 to <8 x bfloat>
  %113 = shufflevector <8 x bfloat> %108, <8 x bfloat> %112, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %114 = add i32 %105, 4
  %115 = mul i32 %114, 16
  %116 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %115, i32 0, i32 0)
  %117 = bitcast i128 %116 to <8 x bfloat>
  %118 = add i32 %105, 6
  %119 = mul i32 %118, 16
  %120 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %119, i32 0, i32 0)
  %121 = bitcast i128 %120 to <8 x bfloat>
  %122 = shufflevector <8 x bfloat> %117, <8 x bfloat> %121, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %123 = add i32 %105, 8
  %124 = mul i32 %123, 16
  %125 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %124, i32 0, i32 0)
  %126 = bitcast i128 %125 to <8 x bfloat>
  %127 = add i32 %105, 10
  %128 = mul i32 %127, 16
  %129 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %128, i32 0, i32 0)
  %130 = bitcast i128 %129 to <8 x bfloat>
  %131 = shufflevector <8 x bfloat> %126, <8 x bfloat> %130, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %132 = add i32 %105, 12
  %133 = mul i32 %132, 16
  %134 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %133, i32 0, i32 0)
  %135 = bitcast i128 %134 to <8 x bfloat>
  %136 = add i32 %105, 14
  %137 = mul i32 %136, 16
  %138 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %137, i32 0, i32 0)
  %139 = bitcast i128 %138 to <8 x bfloat>
  %140 = shufflevector <8 x bfloat> %135, <8 x bfloat> %139, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %141 = add i32 %79, 16
  %142 = add i32 %141, %70
  %143 = mul i32 %142, %94
  %144 = add i32 %98, %143
  %145 = add i32 %144, %78
  %146 = mul i32 %145, 16
  %147 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %146, i32 0, i32 0)
  %148 = bitcast i128 %147 to <8 x bfloat>
  %149 = add i32 %145, 2
  %150 = mul i32 %149, 16
  %151 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %150, i32 0, i32 0)
  %152 = bitcast i128 %151 to <8 x bfloat>
  %153 = shufflevector <8 x bfloat> %148, <8 x bfloat> %152, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %154 = add i32 %145, 4
  %155 = mul i32 %154, 16
  %156 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %155, i32 0, i32 0)
  %157 = bitcast i128 %156 to <8 x bfloat>
  %158 = add i32 %145, 6
  %159 = mul i32 %158, 16
  %160 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %159, i32 0, i32 0)
  %161 = bitcast i128 %160 to <8 x bfloat>
  %162 = shufflevector <8 x bfloat> %157, <8 x bfloat> %161, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %163 = add i32 %145, 8
  %164 = mul i32 %163, 16
  %165 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %164, i32 0, i32 0)
  %166 = bitcast i128 %165 to <8 x bfloat>
  %167 = add i32 %145, 10
  %168 = mul i32 %167, 16
  %169 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %168, i32 0, i32 0)
  %170 = bitcast i128 %169 to <8 x bfloat>
  %171 = shufflevector <8 x bfloat> %166, <8 x bfloat> %170, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %172 = add i32 %145, 12
  %173 = mul i32 %172, 16
  %174 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %173, i32 0, i32 0)
  %175 = bitcast i128 %174 to <8 x bfloat>
  %176 = add i32 %145, 14
  %177 = mul i32 %176, 16
  %178 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %177, i32 0, i32 0)
  %179 = bitcast i128 %178 to <8 x bfloat>
  %180 = shufflevector <8 x bfloat> %175, <8 x bfloat> %179, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %181 = add i32 %79, 32
  %182 = add i32 %181, %70
  %183 = mul i32 %182, %94
  %184 = add i32 %98, %183
  %185 = add i32 %184, %78
  %186 = mul i32 %185, 16
  %187 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %186, i32 0, i32 0)
  %188 = bitcast i128 %187 to <8 x bfloat>
  %189 = add i32 %185, 2
  %190 = mul i32 %189, 16
  %191 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %190, i32 0, i32 0)
  %192 = bitcast i128 %191 to <8 x bfloat>
  %193 = shufflevector <8 x bfloat> %188, <8 x bfloat> %192, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %194 = add i32 %185, 4
  %195 = mul i32 %194, 16
  %196 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %195, i32 0, i32 0)
  %197 = bitcast i128 %196 to <8 x bfloat>
  %198 = add i32 %185, 6
  %199 = mul i32 %198, 16
  %200 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %199, i32 0, i32 0)
  %201 = bitcast i128 %200 to <8 x bfloat>
  %202 = shufflevector <8 x bfloat> %197, <8 x bfloat> %201, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %203 = add i32 %185, 8
  %204 = mul i32 %203, 16
  %205 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %204, i32 0, i32 0)
  %206 = bitcast i128 %205 to <8 x bfloat>
  %207 = add i32 %185, 10
  %208 = mul i32 %207, 16
  %209 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %208, i32 0, i32 0)
  %210 = bitcast i128 %209 to <8 x bfloat>
  %211 = shufflevector <8 x bfloat> %206, <8 x bfloat> %210, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %212 = add i32 %185, 12
  %213 = mul i32 %212, 16
  %214 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %213, i32 0, i32 0)
  %215 = bitcast i128 %214 to <8 x bfloat>
  %216 = add i32 %185, 14
  %217 = mul i32 %216, 16
  %218 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %217, i32 0, i32 0)
  %219 = bitcast i128 %218 to <8 x bfloat>
  %220 = shufflevector <8 x bfloat> %215, <8 x bfloat> %219, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %221 = add i32 %79, 48
  %222 = add i32 %221, %70
  %223 = mul i32 %222, %94
  %224 = add i32 %98, %223
  %225 = add i32 %224, %78
  %226 = mul i32 %225, 16
  %227 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %226, i32 0, i32 0)
  %228 = bitcast i128 %227 to <8 x bfloat>
  %229 = add i32 %225, 2
  %230 = mul i32 %229, 16
  %231 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %230, i32 0, i32 0)
  %232 = bitcast i128 %231 to <8 x bfloat>
  %233 = shufflevector <8 x bfloat> %228, <8 x bfloat> %232, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %234 = add i32 %225, 4
  %235 = mul i32 %234, 16
  %236 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %235, i32 0, i32 0)
  %237 = bitcast i128 %236 to <8 x bfloat>
  %238 = add i32 %225, 6
  %239 = mul i32 %238, 16
  %240 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %239, i32 0, i32 0)
  %241 = bitcast i128 %240 to <8 x bfloat>
  %242 = shufflevector <8 x bfloat> %237, <8 x bfloat> %241, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %243 = add i32 %225, 8
  %244 = mul i32 %243, 16
  %245 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %244, i32 0, i32 0)
  %246 = bitcast i128 %245 to <8 x bfloat>
  %247 = add i32 %225, 10
  %248 = mul i32 %247, 16
  %249 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %248, i32 0, i32 0)
  %250 = bitcast i128 %249 to <8 x bfloat>
  %251 = shufflevector <8 x bfloat> %246, <8 x bfloat> %250, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %252 = add i32 %225, 12
  %253 = mul i32 %252, 16
  %254 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %253, i32 0, i32 0)
  %255 = bitcast i128 %254 to <8 x bfloat>
  %256 = add i32 %225, 14
  %257 = mul i32 %256, 16
  %258 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %89, i32 %257, i32 0, i32 0)
  %259 = bitcast i128 %258 to <8 x bfloat>
  %260 = shufflevector <8 x bfloat> %255, <8 x bfloat> %259, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %261 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %106, i32 0, i32 0)
  %262 = bitcast i128 %261 to <8 x bfloat>
  %263 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %110, i32 0, i32 0)
  %264 = bitcast i128 %263 to <8 x bfloat>
  %265 = shufflevector <8 x bfloat> %262, <8 x bfloat> %264, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %266 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %115, i32 0, i32 0)
  %267 = bitcast i128 %266 to <8 x bfloat>
  %268 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %119, i32 0, i32 0)
  %269 = bitcast i128 %268 to <8 x bfloat>
  %270 = shufflevector <8 x bfloat> %267, <8 x bfloat> %269, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %271 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %124, i32 0, i32 0)
  %272 = bitcast i128 %271 to <8 x bfloat>
  %273 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %128, i32 0, i32 0)
  %274 = bitcast i128 %273 to <8 x bfloat>
  %275 = shufflevector <8 x bfloat> %272, <8 x bfloat> %274, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %276 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %133, i32 0, i32 0)
  %277 = bitcast i128 %276 to <8 x bfloat>
  %278 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %137, i32 0, i32 0)
  %279 = bitcast i128 %278 to <8 x bfloat>
  %280 = shufflevector <8 x bfloat> %277, <8 x bfloat> %279, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %281 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %146, i32 0, i32 0)
  %282 = bitcast i128 %281 to <8 x bfloat>
  %283 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %150, i32 0, i32 0)
  %284 = bitcast i128 %283 to <8 x bfloat>
  %285 = shufflevector <8 x bfloat> %282, <8 x bfloat> %284, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %286 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %155, i32 0, i32 0)
  %287 = bitcast i128 %286 to <8 x bfloat>
  %288 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %159, i32 0, i32 0)
  %289 = bitcast i128 %288 to <8 x bfloat>
  %290 = shufflevector <8 x bfloat> %287, <8 x bfloat> %289, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %291 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %164, i32 0, i32 0)
  %292 = bitcast i128 %291 to <8 x bfloat>
  %293 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %168, i32 0, i32 0)
  %294 = bitcast i128 %293 to <8 x bfloat>
  %295 = shufflevector <8 x bfloat> %292, <8 x bfloat> %294, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %296 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %173, i32 0, i32 0)
  %297 = bitcast i128 %296 to <8 x bfloat>
  %298 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %177, i32 0, i32 0)
  %299 = bitcast i128 %298 to <8 x bfloat>
  %300 = shufflevector <8 x bfloat> %297, <8 x bfloat> %299, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %301 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %186, i32 0, i32 0)
  %302 = bitcast i128 %301 to <8 x bfloat>
  %303 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %190, i32 0, i32 0)
  %304 = bitcast i128 %303 to <8 x bfloat>
  %305 = shufflevector <8 x bfloat> %302, <8 x bfloat> %304, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %306 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %195, i32 0, i32 0)
  %307 = bitcast i128 %306 to <8 x bfloat>
  %308 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %199, i32 0, i32 0)
  %309 = bitcast i128 %308 to <8 x bfloat>
  %310 = shufflevector <8 x bfloat> %307, <8 x bfloat> %309, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %311 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %204, i32 0, i32 0)
  %312 = bitcast i128 %311 to <8 x bfloat>
  %313 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %208, i32 0, i32 0)
  %314 = bitcast i128 %313 to <8 x bfloat>
  %315 = shufflevector <8 x bfloat> %312, <8 x bfloat> %314, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %316 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %213, i32 0, i32 0)
  %317 = bitcast i128 %316 to <8 x bfloat>
  %318 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %217, i32 0, i32 0)
  %319 = bitcast i128 %318 to <8 x bfloat>
  %320 = shufflevector <8 x bfloat> %317, <8 x bfloat> %319, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %321 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %226, i32 0, i32 0)
  %322 = bitcast i128 %321 to <8 x bfloat>
  %323 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %230, i32 0, i32 0)
  %324 = bitcast i128 %323 to <8 x bfloat>
  %325 = shufflevector <8 x bfloat> %322, <8 x bfloat> %324, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %326 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %235, i32 0, i32 0)
  %327 = bitcast i128 %326 to <8 x bfloat>
  %328 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %239, i32 0, i32 0)
  %329 = bitcast i128 %328 to <8 x bfloat>
  %330 = shufflevector <8 x bfloat> %327, <8 x bfloat> %329, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %331 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %244, i32 0, i32 0)
  %332 = bitcast i128 %331 to <8 x bfloat>
  %333 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %248, i32 0, i32 0)
  %334 = bitcast i128 %333 to <8 x bfloat>
  %335 = shufflevector <8 x bfloat> %332, <8 x bfloat> %334, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %336 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %253, i32 0, i32 0)
  %337 = bitcast i128 %336 to <8 x bfloat>
  %338 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %90, i32 %257, i32 0, i32 0)
  %339 = bitcast i128 %338 to <8 x bfloat>
  %340 = shufflevector <8 x bfloat> %337, <8 x bfloat> %339, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %341 = add i32 %101, %102
  %342 = mul i32 %341, 4
  %343 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %91, i32 %342, i32 0, i32 0)
  %344 = bitcast i32 %343 to float
  %345 = add i32 %101, %142
  %346 = mul i32 %345, 4
  %347 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %91, i32 %346, i32 0, i32 0)
  %348 = bitcast i32 %347 to float
  %349 = add i32 %101, %182
  %350 = mul i32 %349, 4
  %351 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %91, i32 %350, i32 0, i32 0)
  %352 = bitcast i32 %351 to float
  %353 = add i32 %101, %222
  %354 = mul i32 %353, 4
  %355 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %91, i32 %354, i32 0, i32 0)
  %356 = bitcast i32 %355 to float
  %357 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %92, i32 %342, i32 0, i32 0)
  %358 = bitcast i32 %357 to float
  %359 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %92, i32 %346, i32 0, i32 0)
  %360 = bitcast i32 %359 to float
  %361 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %92, i32 %350, i32 0, i32 0)
  %362 = bitcast i32 %361 to float
  %363 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %92, i32 %354, i32 0, i32 0)
  %364 = bitcast i32 %363 to float
  %365 = fmul float %16, f0x3FB8AA3B
  %366 = fmul float %344, f0xBFB8AA3B
  %367 = fmul float %348, f0xBFB8AA3B
  %368 = fmul float %352, f0xBFB8AA3B
  %369 = fmul float %356, f0xBFB8AA3B
  %370 = fsub float 0.000000e+00, %16
  %371 = fmul float %358, %370
  %372 = fmul float %360, %370
  %373 = fmul float %362, %370
  %374 = fmul float %364, %370
  %375 = mul i32 %78, 8
  %376 = srem i32 %28, 8
  %377 = add i32 %375, %376
  %378 = sdiv i32 %28, 8
  %379 = mul i32 %378, 8
  %380 = icmp ne i32 %28, %379
  %381 = icmp slt i32 %28, 0
  %382 = icmp ne i1 %381, false
  %383 = and i1 %380, %382
  %384 = add i32 %378, -1
  %385 = select i1 %383, i32 %384, i32 %378
  %386 = srem i32 %385, 2
  %387 = mul i32 %377, 272
  %388 = mul i32 %386, 16
  %389 = add i32 %387, %388
  %390 = mul i32 %70, 272
  %391 = mul i32 %78, 16
  %392 = add i32 %390, %391
  %393 = mul i32 %20, 128
  %394 = add i32 %79, 64
  %395 = add i32 %394, %23
  %396 = add i32 %395, 31
  %397 = sdiv i32 %396, 32
  %398 = mul i32 %397, 32
  %399 = icmp ne i32 %396, %398
  %400 = icmp slt i32 %396, 0
  %401 = icmp ne i1 %400, false
  %402 = and i1 %399, %401
  %403 = add i32 %397, -1
  %404 = select i1 %402, i32 %403, i32 %397
  %405 = call i32 @llvm.smax.i32(i32 %404, i32 1)
  %406 = call i32 @llvm.smin.i32(i32 %405, i32 %22)
  %407 = icmp ne i32 %24, 0
  %408 = select i1 %407, i32 %406, i32 %22
  %409 = add i32 %79, %23
  %410 = add i32 %409, 1
  %411 = icmp slt i32 %410, 0
  %412 = sdiv i32 %410, 32
  %413 = mul i32 %412, 32
  %414 = icmp ne i32 %410, %413
  %415 = icmp slt i32 %410, 0
  %416 = icmp ne i1 %415, false
  %417 = and i1 %414, %416
  %418 = add i32 %412, -1
  %419 = select i1 %417, i32 %418, i32 %412
  %420 = select i1 %411, i32 0, i32 %419
  %421 = call i32 @llvm.smin.i32(i32 %420, i32 %408)
  %422 = select i1 %407, i32 %421, i32 %22
  %423 = sub i32 %408, 1
  %424 = mul i32 %69, %18
  %425 = sext i32 %424 to i64
  %426 = sext i32 %20 to i64
  %427 = mul i64 %425, %426
  %428 = sext i32 %88 to i64
  %429 = add i64 %427, %428
  %430 = mul i64 %429, 128
  %431 = getelementptr bfloat, ptr addrspace(1) %2, i64 %430
  %432 = sext i32 %393 to i64
  %433 = icmp eq i64 %432, -2147483648
  %434 = select i1 %433, i64 128, i64 %432
  %435 = ptrtoint ptr addrspace(1) %431 to i64
  %436 = trunc i64 %435 to i32
  %437 = lshr i64 %435, 32
  %438 = trunc i64 %437 to i32
  %439 = or i32 %438, -2147483648
  %440 = insertelement <4 x i32> <i32 1, i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 poison, i32 poison>, i32 %436, i64 2
  %441 = insertelement <4 x i32> %440, i32 %439, i64 3
  %442 = call i32 @llvm.smax.i32(i32 %18, i32 0)
  %443 = and i32 %442, 65535
  %444 = shl i32 %443, 16
  %445 = or i32 %444, 32767
  %446 = lshr i32 %442, 16
  %447 = and i32 %446, 65535
  %448 = or i32 %447, 8388608
  %449 = trunc i64 %434 to i32
  %450 = lshr i64 %434, 32
  %451 = trunc i64 %450 to i32
  %452 = and i32 %451, 65535
  %453 = insertelement <8 x i32> <i32 122748928, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %445, i64 2
  %454 = insertelement <8 x i32> %453, i32 %448, i64 3
  %455 = insertelement <8 x i32> %454, i32 32, i64 4
  %456 = insertelement <8 x i32> %455, i32 %449, i64 5
  %457 = insertelement <8 x i32> %456, i32 %452, i64 6
  %458 = insertelement <8 x i32> %457, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %441, <8 x i32> %458, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %459 = getelementptr bfloat, ptr addrspace(1) %4, i64 %430
  %460 = ptrtoint ptr addrspace(1) %459 to i64
  %461 = trunc i64 %460 to i32
  %462 = lshr i64 %460, 32
  %463 = trunc i64 %462 to i32
  %464 = or i32 %463, -2147483648
  %465 = insertelement <4 x i32> <i32 1, i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 8704), i32 poison, i32 poison>, i32 %461, i64 2
  %466 = insertelement <4 x i32> %465, i32 %464, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %466, <8 x i32> %458, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %467 = call i32 @llvm.smin.i32(i32 %423, i32 1)
  %468 = mul i32 %467, 32
  %469 = add i32 %424, %468
  %470 = sext i32 %469 to i64
  %471 = mul i64 %470, %426
  %472 = add i64 %471, %428
  %473 = mul i64 %472, 128
  %474 = sub i32 %18, %468
  %475 = getelementptr bfloat, ptr addrspace(1) %2, i64 %473
  %476 = ptrtoint ptr addrspace(1) %475 to i64
  %477 = trunc i64 %476 to i32
  %478 = lshr i64 %476, 32
  %479 = trunc i64 %478 to i32
  %480 = or i32 %479, -2147483648
  %481 = insertelement <4 x i32> <i32 1, i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 17408), i32 poison, i32 poison>, i32 %477, i64 2
  %482 = insertelement <4 x i32> %481, i32 %480, i64 3
  %483 = call i32 @llvm.smax.i32(i32 %474, i32 0)
  %484 = and i32 %483, 65535
  %485 = shl i32 %484, 16
  %486 = or i32 %485, 32767
  %487 = lshr i32 %483, 16
  %488 = and i32 %487, 65535
  %489 = or i32 %488, 8388608
  %490 = insertelement <8 x i32> <i32 122748928, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %486, i64 2
  %491 = insertelement <8 x i32> %490, i32 %489, i64 3
  %492 = insertelement <8 x i32> %491, i32 32, i64 4
  %493 = insertelement <8 x i32> %492, i32 %449, i64 5
  %494 = insertelement <8 x i32> %493, i32 %452, i64 6
  %495 = insertelement <8 x i32> %494, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %482, <8 x i32> %495, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %496 = getelementptr bfloat, ptr addrspace(1) %4, i64 %473
  %497 = ptrtoint ptr addrspace(1) %496 to i64
  %498 = trunc i64 %497 to i32
  %499 = lshr i64 %497, 32
  %500 = trunc i64 %499 to i32
  %501 = or i32 %500, -2147483648
  %502 = insertelement <4 x i32> <i32 1, i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 26112), i32 poison, i32 poison>, i32 %498, i64 2
  %503 = insertelement <4 x i32> %502, i32 %501, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %503, <8 x i32> %495, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %504 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %392
  %505 = inttoptr i32 %504 to ptr addrspace(3)
  %506 = load <8 x bfloat>, ptr addrspace(3) %505, align 16
  %507 = add i32 %504, 32
  %508 = inttoptr i32 %507 to ptr addrspace(3)
  %509 = load <8 x bfloat>, ptr addrspace(3) %508, align 16
  %510 = add i32 %504, 8704
  %511 = inttoptr i32 %510 to ptr addrspace(3)
  %512 = load <8 x bfloat>, ptr addrspace(3) %511, align 16
  %513 = add i32 %504, 8736
  %514 = inttoptr i32 %513 to ptr addrspace(3)
  %515 = load <8 x bfloat>, ptr addrspace(3) %514, align 16
  %516 = add i32 %504, 64
  %517 = inttoptr i32 %516 to ptr addrspace(3)
  %518 = load <8 x bfloat>, ptr addrspace(3) %517, align 16
  %519 = add i32 %504, 96
  %520 = inttoptr i32 %519 to ptr addrspace(3)
  %521 = load <8 x bfloat>, ptr addrspace(3) %520, align 16
  %522 = add i32 %504, 8768
  %523 = inttoptr i32 %522 to ptr addrspace(3)
  %524 = load <8 x bfloat>, ptr addrspace(3) %523, align 16
  %525 = add i32 %504, 8800
  %526 = inttoptr i32 %525 to ptr addrspace(3)
  %527 = load <8 x bfloat>, ptr addrspace(3) %526, align 16
  %528 = add i32 %504, 128
  %529 = inttoptr i32 %528 to ptr addrspace(3)
  %530 = load <8 x bfloat>, ptr addrspace(3) %529, align 16
  %531 = add i32 %504, 160
  %532 = inttoptr i32 %531 to ptr addrspace(3)
  %533 = load <8 x bfloat>, ptr addrspace(3) %532, align 16
  %534 = add i32 %504, 8832
  %535 = inttoptr i32 %534 to ptr addrspace(3)
  %536 = load <8 x bfloat>, ptr addrspace(3) %535, align 16
  %537 = add i32 %504, 8864
  %538 = inttoptr i32 %537 to ptr addrspace(3)
  %539 = load <8 x bfloat>, ptr addrspace(3) %538, align 16
  %540 = add i32 %504, 192
  %541 = inttoptr i32 %540 to ptr addrspace(3)
  %542 = load <8 x bfloat>, ptr addrspace(3) %541, align 16
  %543 = add i32 %504, 224
  %544 = inttoptr i32 %543 to ptr addrspace(3)
  %545 = load <8 x bfloat>, ptr addrspace(3) %544, align 16
  %546 = add i32 %504, 8896
  %547 = inttoptr i32 %546 to ptr addrspace(3)
  %548 = load <8 x bfloat>, ptr addrspace(3) %547, align 16
  %549 = add i32 %504, 8928
  %550 = inttoptr i32 %549 to ptr addrspace(3)
  %551 = load <8 x bfloat>, ptr addrspace(3) %550, align 16
  %552 = add i32 %504, 4352
  %553 = inttoptr i32 %552 to ptr addrspace(3)
  %554 = load <8 x bfloat>, ptr addrspace(3) %553, align 16
  %555 = add i32 %504, 4384
  %556 = inttoptr i32 %555 to ptr addrspace(3)
  %557 = load <8 x bfloat>, ptr addrspace(3) %556, align 16
  %558 = add i32 %504, 13056
  %559 = inttoptr i32 %558 to ptr addrspace(3)
  %560 = load <8 x bfloat>, ptr addrspace(3) %559, align 16
  %561 = add i32 %504, 13088
  %562 = inttoptr i32 %561 to ptr addrspace(3)
  %563 = load <8 x bfloat>, ptr addrspace(3) %562, align 16
  %564 = add i32 %504, 4416
  %565 = inttoptr i32 %564 to ptr addrspace(3)
  %566 = load <8 x bfloat>, ptr addrspace(3) %565, align 16
  %567 = add i32 %504, 4448
  %568 = inttoptr i32 %567 to ptr addrspace(3)
  %569 = load <8 x bfloat>, ptr addrspace(3) %568, align 16
  %570 = add i32 %504, 13120
  %571 = inttoptr i32 %570 to ptr addrspace(3)
  %572 = load <8 x bfloat>, ptr addrspace(3) %571, align 16
  %573 = add i32 %504, 13152
  %574 = inttoptr i32 %573 to ptr addrspace(3)
  %575 = load <8 x bfloat>, ptr addrspace(3) %574, align 16
  %576 = add i32 %504, 4480
  %577 = inttoptr i32 %576 to ptr addrspace(3)
  %578 = load <8 x bfloat>, ptr addrspace(3) %577, align 16
  %579 = add i32 %504, 4512
  %580 = inttoptr i32 %579 to ptr addrspace(3)
  %581 = load <8 x bfloat>, ptr addrspace(3) %580, align 16
  %582 = add i32 %504, 13184
  %583 = inttoptr i32 %582 to ptr addrspace(3)
  %584 = load <8 x bfloat>, ptr addrspace(3) %583, align 16
  %585 = add i32 %504, 13216
  %586 = inttoptr i32 %585 to ptr addrspace(3)
  %587 = load <8 x bfloat>, ptr addrspace(3) %586, align 16
  %588 = add i32 %504, 4544
  %589 = inttoptr i32 %588 to ptr addrspace(3)
  %590 = load <8 x bfloat>, ptr addrspace(3) %589, align 16
  %591 = add i32 %504, 4576
  %592 = inttoptr i32 %591 to ptr addrspace(3)
  %593 = load <8 x bfloat>, ptr addrspace(3) %592, align 16
  %594 = add i32 %504, 13248
  %595 = inttoptr i32 %594 to ptr addrspace(3)
  %596 = load <8 x bfloat>, ptr addrspace(3) %595, align 16
  %597 = add i32 %504, 13280
  %598 = inttoptr i32 %597 to ptr addrspace(3)
  %599 = load <8 x bfloat>, ptr addrspace(3) %598, align 16
  %600 = sext i32 %422 to i64
  br label %601

601:                                              ; preds = %669, %25
  %602 = phi i64 [ %1501, %669 ], [ 0, %25 ]
  %603 = phi <8 x float> [ %1253, %669 ], [ zeroinitializer, %25 ]
  %604 = phi <8 x float> [ %1254, %669 ], [ zeroinitializer, %25 ]
  %605 = phi <8 x float> [ %1255, %669 ], [ zeroinitializer, %25 ]
  %606 = phi <8 x float> [ %1256, %669 ], [ zeroinitializer, %25 ]
  %607 = phi <8 x float> [ %1257, %669 ], [ zeroinitializer, %25 ]
  %608 = phi <8 x float> [ %1258, %669 ], [ zeroinitializer, %25 ]
  %609 = phi <8 x float> [ %1259, %669 ], [ zeroinitializer, %25 ]
  %610 = phi <8 x float> [ %1260, %669 ], [ zeroinitializer, %25 ]
  %611 = phi <8 x float> [ %1333, %669 ], [ zeroinitializer, %25 ]
  %612 = phi <8 x float> [ %1334, %669 ], [ zeroinitializer, %25 ]
  %613 = phi <8 x float> [ %1335, %669 ], [ zeroinitializer, %25 ]
  %614 = phi <8 x float> [ %1336, %669 ], [ zeroinitializer, %25 ]
  %615 = phi <8 x float> [ %1337, %669 ], [ zeroinitializer, %25 ]
  %616 = phi <8 x float> [ %1338, %669 ], [ zeroinitializer, %25 ]
  %617 = phi <8 x float> [ %1339, %669 ], [ zeroinitializer, %25 ]
  %618 = phi <8 x float> [ %1340, %669 ], [ zeroinitializer, %25 ]
  %619 = phi <8 x float> [ %1413, %669 ], [ zeroinitializer, %25 ]
  %620 = phi <8 x float> [ %1414, %669 ], [ zeroinitializer, %25 ]
  %621 = phi <8 x float> [ %1415, %669 ], [ zeroinitializer, %25 ]
  %622 = phi <8 x float> [ %1416, %669 ], [ zeroinitializer, %25 ]
  %623 = phi <8 x float> [ %1417, %669 ], [ zeroinitializer, %25 ]
  %624 = phi <8 x float> [ %1418, %669 ], [ zeroinitializer, %25 ]
  %625 = phi <8 x float> [ %1419, %669 ], [ zeroinitializer, %25 ]
  %626 = phi <8 x float> [ %1420, %669 ], [ zeroinitializer, %25 ]
  %627 = phi <8 x float> [ %1493, %669 ], [ zeroinitializer, %25 ]
  %628 = phi <8 x float> [ %1494, %669 ], [ zeroinitializer, %25 ]
  %629 = phi <8 x float> [ %1495, %669 ], [ zeroinitializer, %25 ]
  %630 = phi <8 x float> [ %1496, %669 ], [ zeroinitializer, %25 ]
  %631 = phi <8 x float> [ %1497, %669 ], [ zeroinitializer, %25 ]
  %632 = phi <8 x float> [ %1498, %669 ], [ zeroinitializer, %25 ]
  %633 = phi <8 x float> [ %1499, %669 ], [ zeroinitializer, %25 ]
  %634 = phi <8 x float> [ %1500, %669 ], [ zeroinitializer, %25 ]
  %635 = phi <8 x bfloat> [ %1087, %669 ], [ %506, %25 ]
  %636 = phi <8 x bfloat> [ %1090, %669 ], [ %509, %25 ]
  %637 = phi <8 x bfloat> [ %1093, %669 ], [ %512, %25 ]
  %638 = phi <8 x bfloat> [ %1096, %669 ], [ %515, %25 ]
  %639 = phi <8 x bfloat> [ %1099, %669 ], [ %518, %25 ]
  %640 = phi <8 x bfloat> [ %1102, %669 ], [ %521, %25 ]
  %641 = phi <8 x bfloat> [ %1105, %669 ], [ %524, %25 ]
  %642 = phi <8 x bfloat> [ %1108, %669 ], [ %527, %25 ]
  %643 = phi <8 x bfloat> [ %1111, %669 ], [ %530, %25 ]
  %644 = phi <8 x bfloat> [ %1114, %669 ], [ %533, %25 ]
  %645 = phi <8 x bfloat> [ %1117, %669 ], [ %536, %25 ]
  %646 = phi <8 x bfloat> [ %1120, %669 ], [ %539, %25 ]
  %647 = phi <8 x bfloat> [ %1123, %669 ], [ %542, %25 ]
  %648 = phi <8 x bfloat> [ %1126, %669 ], [ %545, %25 ]
  %649 = phi <8 x bfloat> [ %1129, %669 ], [ %548, %25 ]
  %650 = phi <8 x bfloat> [ %1132, %669 ], [ %551, %25 ]
  %651 = phi <8 x bfloat> [ %1135, %669 ], [ %554, %25 ]
  %652 = phi <8 x bfloat> [ %1138, %669 ], [ %557, %25 ]
  %653 = phi <8 x bfloat> [ %1141, %669 ], [ %560, %25 ]
  %654 = phi <8 x bfloat> [ %1144, %669 ], [ %563, %25 ]
  %655 = phi <8 x bfloat> [ %1147, %669 ], [ %566, %25 ]
  %656 = phi <8 x bfloat> [ %1150, %669 ], [ %569, %25 ]
  %657 = phi <8 x bfloat> [ %1153, %669 ], [ %572, %25 ]
  %658 = phi <8 x bfloat> [ %1156, %669 ], [ %575, %25 ]
  %659 = phi <8 x bfloat> [ %1159, %669 ], [ %578, %25 ]
  %660 = phi <8 x bfloat> [ %1162, %669 ], [ %581, %25 ]
  %661 = phi <8 x bfloat> [ %1165, %669 ], [ %584, %25 ]
  %662 = phi <8 x bfloat> [ %1168, %669 ], [ %587, %25 ]
  %663 = phi <8 x bfloat> [ %1171, %669 ], [ %590, %25 ]
  %664 = phi <8 x bfloat> [ %1174, %669 ], [ %593, %25 ]
  %665 = phi <8 x bfloat> [ %1177, %669 ], [ %596, %25 ]
  %666 = phi <8 x bfloat> [ %1180, %669 ], [ %599, %25 ]
  %667 = phi i32 [ %678, %669 ], [ 0, %25 ]
  %668 = icmp slt i64 %602, %600
  br i1 %668, label %669, label %1502

669:                                              ; preds = %601
  %670 = trunc i64 %602 to i32
  %671 = add i32 %670, 2
  %672 = call i32 @llvm.smin.i32(i32 %671, i32 %423)
  %673 = icmp eq i32 %667, 0
  %674 = sub i32 %667, 17408
  %675 = select i1 %673, i32 34816, i32 %674
  %676 = icmp eq i32 %667, 34816
  %677 = add i32 %667, 17408
  %678 = select i1 %676, i32 0, i32 %677
  %679 = mul i32 %672, 32
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %680 = add i32 %424, %679
  %681 = sext i32 %680 to i64
  %682 = mul i64 %681, %426
  %683 = add i64 %682, %428
  %684 = mul i64 %683, 128
  %685 = sub i32 %18, %679
  %686 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %675
  %687 = add i32 %686, 8704
  %688 = getelementptr bfloat, ptr addrspace(1) %2, i64 %684
  %689 = inttoptr i32 %686 to ptr addrspace(3)
  %690 = ptrtoint ptr addrspace(1) %688 to i64
  %691 = ptrtoint ptr addrspace(3) %689 to i32
  %692 = trunc i64 %690 to i32
  %693 = lshr i64 %690, 32
  %694 = trunc i64 %693 to i32
  %695 = or i32 %694, -2147483648
  %696 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %691, i64 1
  %697 = insertelement <4 x i32> %696, i32 %692, i64 2
  %698 = insertelement <4 x i32> %697, i32 %695, i64 3
  %699 = call i32 @llvm.smax.i32(i32 %685, i32 0)
  %700 = and i32 %699, 65535
  %701 = shl i32 %700, 16
  %702 = or i32 %701, 32767
  %703 = lshr i32 %699, 16
  %704 = and i32 %703, 65535
  %705 = or i32 %704, 8388608
  %706 = insertelement <8 x i32> <i32 122748928, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %702, i64 2
  %707 = insertelement <8 x i32> %706, i32 %705, i64 3
  %708 = insertelement <8 x i32> %707, i32 32, i64 4
  %709 = insertelement <8 x i32> %708, i32 %449, i64 5
  %710 = insertelement <8 x i32> %709, i32 %452, i64 6
  %711 = insertelement <8 x i32> %710, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %698, <8 x i32> %711, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %712 = getelementptr bfloat, ptr addrspace(1) %4, i64 %684
  %713 = inttoptr i32 %687 to ptr addrspace(3)
  %714 = ptrtoint ptr addrspace(1) %712 to i64
  %715 = ptrtoint ptr addrspace(3) %713 to i32
  %716 = trunc i64 %714 to i32
  %717 = lshr i64 %714, 32
  %718 = trunc i64 %717 to i32
  %719 = or i32 %718, -2147483648
  %720 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %715, i64 1
  %721 = insertelement <4 x i32> %720, i32 %716, i64 2
  %722 = insertelement <4 x i32> %721, i32 %719, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %722, <8 x i32> %711, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %723 = shufflevector <8 x bfloat> %635, <8 x bfloat> %636, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %724 = shufflevector <8 x bfloat> %637, <8 x bfloat> %638, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %725 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %723, <16 x bfloat> %113, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %726 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %724, <16 x bfloat> %265, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %727 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %723, <16 x bfloat> %153, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %728 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %724, <16 x bfloat> %285, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %729 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %723, <16 x bfloat> %193, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %730 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %724, <16 x bfloat> %305, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %731 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %723, <16 x bfloat> %233, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %732 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %724, <16 x bfloat> %325, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %733 = shufflevector <8 x bfloat> %639, <8 x bfloat> %640, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %734 = shufflevector <8 x bfloat> %641, <8 x bfloat> %642, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %735 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %733, <16 x bfloat> %122, i16 0, <8 x float> %725, i1 false, i1 false)
  %736 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %734, <16 x bfloat> %270, i16 0, <8 x float> %726, i1 false, i1 false)
  %737 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %733, <16 x bfloat> %162, i16 0, <8 x float> %727, i1 false, i1 false)
  %738 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %734, <16 x bfloat> %290, i16 0, <8 x float> %728, i1 false, i1 false)
  %739 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %733, <16 x bfloat> %202, i16 0, <8 x float> %729, i1 false, i1 false)
  %740 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %734, <16 x bfloat> %310, i16 0, <8 x float> %730, i1 false, i1 false)
  %741 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %733, <16 x bfloat> %242, i16 0, <8 x float> %731, i1 false, i1 false)
  %742 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %734, <16 x bfloat> %330, i16 0, <8 x float> %732, i1 false, i1 false)
  %743 = shufflevector <8 x bfloat> %643, <8 x bfloat> %644, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %744 = shufflevector <8 x bfloat> %645, <8 x bfloat> %646, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %745 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %743, <16 x bfloat> %131, i16 0, <8 x float> %735, i1 false, i1 false)
  %746 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %744, <16 x bfloat> %275, i16 0, <8 x float> %736, i1 false, i1 false)
  %747 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %743, <16 x bfloat> %171, i16 0, <8 x float> %737, i1 false, i1 false)
  %748 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %744, <16 x bfloat> %295, i16 0, <8 x float> %738, i1 false, i1 false)
  %749 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %743, <16 x bfloat> %211, i16 0, <8 x float> %739, i1 false, i1 false)
  %750 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %744, <16 x bfloat> %315, i16 0, <8 x float> %740, i1 false, i1 false)
  %751 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %743, <16 x bfloat> %251, i16 0, <8 x float> %741, i1 false, i1 false)
  %752 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %744, <16 x bfloat> %335, i16 0, <8 x float> %742, i1 false, i1 false)
  %753 = shufflevector <8 x bfloat> %647, <8 x bfloat> %648, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %754 = shufflevector <8 x bfloat> %649, <8 x bfloat> %650, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %755 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %753, <16 x bfloat> %140, i16 0, <8 x float> %745, i1 false, i1 false)
  %756 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %754, <16 x bfloat> %280, i16 0, <8 x float> %746, i1 false, i1 false)
  %757 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %753, <16 x bfloat> %180, i16 0, <8 x float> %747, i1 false, i1 false)
  %758 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %754, <16 x bfloat> %300, i16 0, <8 x float> %748, i1 false, i1 false)
  %759 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %753, <16 x bfloat> %220, i16 0, <8 x float> %749, i1 false, i1 false)
  %760 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %754, <16 x bfloat> %320, i16 0, <8 x float> %750, i1 false, i1 false)
  %761 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %753, <16 x bfloat> %260, i16 0, <8 x float> %751, i1 false, i1 false)
  %762 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %754, <16 x bfloat> %340, i16 0, <8 x float> %752, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %763 = shufflevector <8 x bfloat> %651, <8 x bfloat> %652, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %764 = shufflevector <8 x bfloat> %653, <8 x bfloat> %654, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %765 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %763, <16 x bfloat> %113, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %766 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %764, <16 x bfloat> %265, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %767 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %763, <16 x bfloat> %153, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %768 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %764, <16 x bfloat> %285, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %769 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %763, <16 x bfloat> %193, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %770 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %764, <16 x bfloat> %305, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %771 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %763, <16 x bfloat> %233, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %772 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %764, <16 x bfloat> %325, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %773 = extractelement <8 x float> %755, i64 0
  %774 = call float @llvm.fma.f32(float %773, float %365, float %366)
  %775 = extractelement <8 x float> %755, i64 1
  %776 = call float @llvm.fma.f32(float %775, float %365, float %366)
  %777 = extractelement <8 x float> %755, i64 2
  %778 = call float @llvm.fma.f32(float %777, float %365, float %366)
  %779 = extractelement <8 x float> %755, i64 3
  %780 = call float @llvm.fma.f32(float %779, float %365, float %366)
  %781 = extractelement <8 x float> %755, i64 4
  %782 = call float @llvm.fma.f32(float %781, float %365, float %366)
  %783 = extractelement <8 x float> %755, i64 5
  %784 = call float @llvm.fma.f32(float %783, float %365, float %366)
  %785 = extractelement <8 x float> %755, i64 6
  %786 = call float @llvm.fma.f32(float %785, float %365, float %366)
  %787 = extractelement <8 x float> %755, i64 7
  %788 = call float @llvm.fma.f32(float %787, float %365, float %366)
  %789 = call float @llvm.amdgcn.exp2.f32(float %774)
  %790 = call float @llvm.amdgcn.exp2.f32(float %776)
  %791 = call float @llvm.amdgcn.exp2.f32(float %778)
  %792 = call float @llvm.amdgcn.exp2.f32(float %780)
  %793 = call float @llvm.amdgcn.exp2.f32(float %782)
  %794 = call float @llvm.amdgcn.exp2.f32(float %784)
  %795 = call float @llvm.amdgcn.exp2.f32(float %786)
  %796 = call float @llvm.amdgcn.exp2.f32(float %788)
  %797 = extractelement <8 x float> %756, i64 0
  %798 = call float @llvm.fma.f32(float %797, float %16, float %371)
  %799 = fmul float %789, %798
  %800 = fptrunc float %799 to bfloat
  %801 = extractelement <8 x float> %756, i64 1
  %802 = call float @llvm.fma.f32(float %801, float %16, float %371)
  %803 = fmul float %790, %802
  %804 = fptrunc float %803 to bfloat
  %805 = extractelement <8 x float> %756, i64 2
  %806 = call float @llvm.fma.f32(float %805, float %16, float %371)
  %807 = fmul float %791, %806
  %808 = fptrunc float %807 to bfloat
  %809 = extractelement <8 x float> %756, i64 3
  %810 = call float @llvm.fma.f32(float %809, float %16, float %371)
  %811 = fmul float %792, %810
  %812 = fptrunc float %811 to bfloat
  %813 = extractelement <8 x float> %756, i64 4
  %814 = call float @llvm.fma.f32(float %813, float %16, float %371)
  %815 = fmul float %793, %814
  %816 = fptrunc float %815 to bfloat
  %817 = extractelement <8 x float> %756, i64 5
  %818 = call float @llvm.fma.f32(float %817, float %16, float %371)
  %819 = fmul float %794, %818
  %820 = fptrunc float %819 to bfloat
  %821 = extractelement <8 x float> %756, i64 6
  %822 = call float @llvm.fma.f32(float %821, float %16, float %371)
  %823 = fmul float %795, %822
  %824 = fptrunc float %823 to bfloat
  %825 = extractelement <8 x float> %756, i64 7
  %826 = call float @llvm.fma.f32(float %825, float %16, float %371)
  %827 = fmul float %796, %826
  %828 = fptrunc float %827 to bfloat
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %829 = shufflevector <8 x bfloat> %655, <8 x bfloat> %656, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %830 = shufflevector <8 x bfloat> %657, <8 x bfloat> %658, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %831 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %829, <16 x bfloat> %122, i16 0, <8 x float> %765, i1 false, i1 false)
  %832 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %830, <16 x bfloat> %270, i16 0, <8 x float> %766, i1 false, i1 false)
  %833 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %829, <16 x bfloat> %162, i16 0, <8 x float> %767, i1 false, i1 false)
  %834 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %830, <16 x bfloat> %290, i16 0, <8 x float> %768, i1 false, i1 false)
  %835 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %829, <16 x bfloat> %202, i16 0, <8 x float> %769, i1 false, i1 false)
  %836 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %830, <16 x bfloat> %310, i16 0, <8 x float> %770, i1 false, i1 false)
  %837 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %829, <16 x bfloat> %242, i16 0, <8 x float> %771, i1 false, i1 false)
  %838 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %830, <16 x bfloat> %330, i16 0, <8 x float> %772, i1 false, i1 false)
  %839 = extractelement <8 x float> %757, i64 0
  %840 = call float @llvm.fma.f32(float %839, float %365, float %367)
  %841 = extractelement <8 x float> %757, i64 1
  %842 = call float @llvm.fma.f32(float %841, float %365, float %367)
  %843 = extractelement <8 x float> %757, i64 2
  %844 = call float @llvm.fma.f32(float %843, float %365, float %367)
  %845 = extractelement <8 x float> %757, i64 3
  %846 = call float @llvm.fma.f32(float %845, float %365, float %367)
  %847 = extractelement <8 x float> %757, i64 4
  %848 = call float @llvm.fma.f32(float %847, float %365, float %367)
  %849 = extractelement <8 x float> %757, i64 5
  %850 = call float @llvm.fma.f32(float %849, float %365, float %367)
  %851 = extractelement <8 x float> %757, i64 6
  %852 = call float @llvm.fma.f32(float %851, float %365, float %367)
  %853 = extractelement <8 x float> %757, i64 7
  %854 = call float @llvm.fma.f32(float %853, float %365, float %367)
  %855 = call float @llvm.amdgcn.exp2.f32(float %840)
  %856 = call float @llvm.amdgcn.exp2.f32(float %842)
  %857 = call float @llvm.amdgcn.exp2.f32(float %844)
  %858 = call float @llvm.amdgcn.exp2.f32(float %846)
  %859 = call float @llvm.amdgcn.exp2.f32(float %848)
  %860 = call float @llvm.amdgcn.exp2.f32(float %850)
  %861 = call float @llvm.amdgcn.exp2.f32(float %852)
  %862 = call float @llvm.amdgcn.exp2.f32(float %854)
  %863 = extractelement <8 x float> %758, i64 0
  %864 = call float @llvm.fma.f32(float %863, float %16, float %372)
  %865 = fmul float %855, %864
  %866 = fptrunc float %865 to bfloat
  %867 = extractelement <8 x float> %758, i64 1
  %868 = call float @llvm.fma.f32(float %867, float %16, float %372)
  %869 = fmul float %856, %868
  %870 = fptrunc float %869 to bfloat
  %871 = extractelement <8 x float> %758, i64 2
  %872 = call float @llvm.fma.f32(float %871, float %16, float %372)
  %873 = fmul float %857, %872
  %874 = fptrunc float %873 to bfloat
  %875 = extractelement <8 x float> %758, i64 3
  %876 = call float @llvm.fma.f32(float %875, float %16, float %372)
  %877 = fmul float %858, %876
  %878 = fptrunc float %877 to bfloat
  %879 = extractelement <8 x float> %758, i64 4
  %880 = call float @llvm.fma.f32(float %879, float %16, float %372)
  %881 = fmul float %859, %880
  %882 = fptrunc float %881 to bfloat
  %883 = extractelement <8 x float> %758, i64 5
  %884 = call float @llvm.fma.f32(float %883, float %16, float %372)
  %885 = fmul float %860, %884
  %886 = fptrunc float %885 to bfloat
  %887 = extractelement <8 x float> %758, i64 6
  %888 = call float @llvm.fma.f32(float %887, float %16, float %372)
  %889 = fmul float %861, %888
  %890 = fptrunc float %889 to bfloat
  %891 = extractelement <8 x float> %758, i64 7
  %892 = call float @llvm.fma.f32(float %891, float %16, float %372)
  %893 = fmul float %862, %892
  %894 = fptrunc float %893 to bfloat
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %895 = shufflevector <8 x bfloat> %659, <8 x bfloat> %660, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %896 = shufflevector <8 x bfloat> %661, <8 x bfloat> %662, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %897 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %895, <16 x bfloat> %131, i16 0, <8 x float> %831, i1 false, i1 false)
  %898 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %896, <16 x bfloat> %275, i16 0, <8 x float> %832, i1 false, i1 false)
  %899 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %895, <16 x bfloat> %171, i16 0, <8 x float> %833, i1 false, i1 false)
  %900 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %896, <16 x bfloat> %295, i16 0, <8 x float> %834, i1 false, i1 false)
  %901 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %895, <16 x bfloat> %211, i16 0, <8 x float> %835, i1 false, i1 false)
  %902 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %896, <16 x bfloat> %315, i16 0, <8 x float> %836, i1 false, i1 false)
  %903 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %895, <16 x bfloat> %251, i16 0, <8 x float> %837, i1 false, i1 false)
  %904 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %896, <16 x bfloat> %335, i16 0, <8 x float> %838, i1 false, i1 false)
  %905 = extractelement <8 x float> %759, i64 0
  %906 = call float @llvm.fma.f32(float %905, float %365, float %368)
  %907 = extractelement <8 x float> %759, i64 1
  %908 = call float @llvm.fma.f32(float %907, float %365, float %368)
  %909 = extractelement <8 x float> %759, i64 2
  %910 = call float @llvm.fma.f32(float %909, float %365, float %368)
  %911 = extractelement <8 x float> %759, i64 3
  %912 = call float @llvm.fma.f32(float %911, float %365, float %368)
  %913 = extractelement <8 x float> %759, i64 4
  %914 = call float @llvm.fma.f32(float %913, float %365, float %368)
  %915 = extractelement <8 x float> %759, i64 5
  %916 = call float @llvm.fma.f32(float %915, float %365, float %368)
  %917 = extractelement <8 x float> %759, i64 6
  %918 = call float @llvm.fma.f32(float %917, float %365, float %368)
  %919 = extractelement <8 x float> %759, i64 7
  %920 = call float @llvm.fma.f32(float %919, float %365, float %368)
  %921 = call float @llvm.amdgcn.exp2.f32(float %906)
  %922 = call float @llvm.amdgcn.exp2.f32(float %908)
  %923 = call float @llvm.amdgcn.exp2.f32(float %910)
  %924 = call float @llvm.amdgcn.exp2.f32(float %912)
  %925 = call float @llvm.amdgcn.exp2.f32(float %914)
  %926 = call float @llvm.amdgcn.exp2.f32(float %916)
  %927 = call float @llvm.amdgcn.exp2.f32(float %918)
  %928 = call float @llvm.amdgcn.exp2.f32(float %920)
  %929 = extractelement <8 x float> %760, i64 0
  %930 = call float @llvm.fma.f32(float %929, float %16, float %373)
  %931 = fmul float %921, %930
  %932 = fptrunc float %931 to bfloat
  %933 = extractelement <8 x float> %760, i64 1
  %934 = call float @llvm.fma.f32(float %933, float %16, float %373)
  %935 = fmul float %922, %934
  %936 = fptrunc float %935 to bfloat
  %937 = extractelement <8 x float> %760, i64 2
  %938 = call float @llvm.fma.f32(float %937, float %16, float %373)
  %939 = fmul float %923, %938
  %940 = fptrunc float %939 to bfloat
  %941 = extractelement <8 x float> %760, i64 3
  %942 = call float @llvm.fma.f32(float %941, float %16, float %373)
  %943 = fmul float %924, %942
  %944 = fptrunc float %943 to bfloat
  %945 = extractelement <8 x float> %760, i64 4
  %946 = call float @llvm.fma.f32(float %945, float %16, float %373)
  %947 = fmul float %925, %946
  %948 = fptrunc float %947 to bfloat
  %949 = extractelement <8 x float> %760, i64 5
  %950 = call float @llvm.fma.f32(float %949, float %16, float %373)
  %951 = fmul float %926, %950
  %952 = fptrunc float %951 to bfloat
  %953 = extractelement <8 x float> %760, i64 6
  %954 = call float @llvm.fma.f32(float %953, float %16, float %373)
  %955 = fmul float %927, %954
  %956 = fptrunc float %955 to bfloat
  %957 = extractelement <8 x float> %760, i64 7
  %958 = call float @llvm.fma.f32(float %957, float %16, float %373)
  %959 = fmul float %928, %958
  %960 = fptrunc float %959 to bfloat
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %961 = shufflevector <8 x bfloat> %663, <8 x bfloat> %664, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %962 = shufflevector <8 x bfloat> %665, <8 x bfloat> %666, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %963 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %961, <16 x bfloat> %140, i16 0, <8 x float> %897, i1 false, i1 false)
  %964 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %962, <16 x bfloat> %280, i16 0, <8 x float> %898, i1 false, i1 false)
  %965 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %961, <16 x bfloat> %180, i16 0, <8 x float> %899, i1 false, i1 false)
  %966 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %962, <16 x bfloat> %300, i16 0, <8 x float> %900, i1 false, i1 false)
  %967 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %961, <16 x bfloat> %220, i16 0, <8 x float> %901, i1 false, i1 false)
  %968 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %962, <16 x bfloat> %320, i16 0, <8 x float> %902, i1 false, i1 false)
  %969 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %961, <16 x bfloat> %260, i16 0, <8 x float> %903, i1 false, i1 false)
  %970 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %962, <16 x bfloat> %340, i16 0, <8 x float> %904, i1 false, i1 false)
  %971 = extractelement <8 x float> %761, i64 0
  %972 = call float @llvm.fma.f32(float %971, float %365, float %369)
  %973 = extractelement <8 x float> %761, i64 1
  %974 = call float @llvm.fma.f32(float %973, float %365, float %369)
  %975 = extractelement <8 x float> %761, i64 2
  %976 = call float @llvm.fma.f32(float %975, float %365, float %369)
  %977 = extractelement <8 x float> %761, i64 3
  %978 = call float @llvm.fma.f32(float %977, float %365, float %369)
  %979 = extractelement <8 x float> %761, i64 4
  %980 = call float @llvm.fma.f32(float %979, float %365, float %369)
  %981 = extractelement <8 x float> %761, i64 5
  %982 = call float @llvm.fma.f32(float %981, float %365, float %369)
  %983 = extractelement <8 x float> %761, i64 6
  %984 = call float @llvm.fma.f32(float %983, float %365, float %369)
  %985 = extractelement <8 x float> %761, i64 7
  %986 = call float @llvm.fma.f32(float %985, float %365, float %369)
  %987 = call float @llvm.amdgcn.exp2.f32(float %972)
  %988 = call float @llvm.amdgcn.exp2.f32(float %974)
  %989 = call float @llvm.amdgcn.exp2.f32(float %976)
  %990 = call float @llvm.amdgcn.exp2.f32(float %978)
  %991 = call float @llvm.amdgcn.exp2.f32(float %980)
  %992 = call float @llvm.amdgcn.exp2.f32(float %982)
  %993 = call float @llvm.amdgcn.exp2.f32(float %984)
  %994 = call float @llvm.amdgcn.exp2.f32(float %986)
  %995 = extractelement <8 x float> %762, i64 0
  %996 = call float @llvm.fma.f32(float %995, float %16, float %374)
  %997 = fmul float %987, %996
  %998 = fptrunc float %997 to bfloat
  %999 = extractelement <8 x float> %762, i64 1
  %1000 = call float @llvm.fma.f32(float %999, float %16, float %374)
  %1001 = fmul float %988, %1000
  %1002 = fptrunc float %1001 to bfloat
  %1003 = extractelement <8 x float> %762, i64 2
  %1004 = call float @llvm.fma.f32(float %1003, float %16, float %374)
  %1005 = fmul float %989, %1004
  %1006 = fptrunc float %1005 to bfloat
  %1007 = extractelement <8 x float> %762, i64 3
  %1008 = call float @llvm.fma.f32(float %1007, float %16, float %374)
  %1009 = fmul float %990, %1008
  %1010 = fptrunc float %1009 to bfloat
  %1011 = extractelement <8 x float> %762, i64 4
  %1012 = call float @llvm.fma.f32(float %1011, float %16, float %374)
  %1013 = fmul float %991, %1012
  %1014 = fptrunc float %1013 to bfloat
  %1015 = extractelement <8 x float> %762, i64 5
  %1016 = call float @llvm.fma.f32(float %1015, float %16, float %374)
  %1017 = fmul float %992, %1016
  %1018 = fptrunc float %1017 to bfloat
  %1019 = extractelement <8 x float> %762, i64 6
  %1020 = call float @llvm.fma.f32(float %1019, float %16, float %374)
  %1021 = fmul float %993, %1020
  %1022 = fptrunc float %1021 to bfloat
  %1023 = extractelement <8 x float> %762, i64 7
  %1024 = call float @llvm.fma.f32(float %1023, float %16, float %374)
  %1025 = fmul float %994, %1024
  %1026 = fptrunc float %1025 to bfloat
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1027 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %667
  %1028 = add i32 %1027, %389
  %1029 = inttoptr i32 %1028 to ptr addrspace(3)
  %1030 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1029)
  %1031 = add i32 %1028, 4352
  %1032 = inttoptr i32 %1031 to ptr addrspace(3)
  %1033 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1032)
  %1034 = shufflevector <8 x bfloat> %1030, <8 x bfloat> %1033, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1035 = add i32 %1028, 32
  %1036 = inttoptr i32 %1035 to ptr addrspace(3)
  %1037 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1036)
  %1038 = add i32 %1028, 4384
  %1039 = inttoptr i32 %1038 to ptr addrspace(3)
  %1040 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1039)
  %1041 = shufflevector <8 x bfloat> %1037, <8 x bfloat> %1040, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1042 = add i32 %1028, 64
  %1043 = inttoptr i32 %1042 to ptr addrspace(3)
  %1044 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1043)
  %1045 = add i32 %1028, 4416
  %1046 = inttoptr i32 %1045 to ptr addrspace(3)
  %1047 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1046)
  %1048 = shufflevector <8 x bfloat> %1044, <8 x bfloat> %1047, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1049 = add i32 %1028, 96
  %1050 = inttoptr i32 %1049 to ptr addrspace(3)
  %1051 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1050)
  %1052 = add i32 %1028, 4448
  %1053 = inttoptr i32 %1052 to ptr addrspace(3)
  %1054 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1053)
  %1055 = shufflevector <8 x bfloat> %1051, <8 x bfloat> %1054, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1056 = add i32 %1028, 128
  %1057 = inttoptr i32 %1056 to ptr addrspace(3)
  %1058 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1057)
  %1059 = add i32 %1028, 4480
  %1060 = inttoptr i32 %1059 to ptr addrspace(3)
  %1061 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1060)
  %1062 = shufflevector <8 x bfloat> %1058, <8 x bfloat> %1061, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1063 = add i32 %1028, 160
  %1064 = inttoptr i32 %1063 to ptr addrspace(3)
  %1065 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1064)
  %1066 = add i32 %1028, 4512
  %1067 = inttoptr i32 %1066 to ptr addrspace(3)
  %1068 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1067)
  %1069 = shufflevector <8 x bfloat> %1065, <8 x bfloat> %1068, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1070 = add i32 %1028, 192
  %1071 = inttoptr i32 %1070 to ptr addrspace(3)
  %1072 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1071)
  %1073 = add i32 %1028, 4544
  %1074 = inttoptr i32 %1073 to ptr addrspace(3)
  %1075 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1074)
  %1076 = shufflevector <8 x bfloat> %1072, <8 x bfloat> %1075, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1077 = add i32 %1028, 224
  %1078 = inttoptr i32 %1077 to ptr addrspace(3)
  %1079 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1078)
  %1080 = add i32 %1028, 4576
  %1081 = inttoptr i32 %1080 to ptr addrspace(3)
  %1082 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1081)
  %1083 = shufflevector <8 x bfloat> %1079, <8 x bfloat> %1082, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1084 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %678
  %1085 = add i32 %1084, %392
  %1086 = inttoptr i32 %1085 to ptr addrspace(3)
  %1087 = load <8 x bfloat>, ptr addrspace(3) %1086, align 16
  %1088 = add i32 %1085, 32
  %1089 = inttoptr i32 %1088 to ptr addrspace(3)
  %1090 = load <8 x bfloat>, ptr addrspace(3) %1089, align 16
  %1091 = add i32 %1085, 8704
  %1092 = inttoptr i32 %1091 to ptr addrspace(3)
  %1093 = load <8 x bfloat>, ptr addrspace(3) %1092, align 16
  %1094 = add i32 %1085, 8736
  %1095 = inttoptr i32 %1094 to ptr addrspace(3)
  %1096 = load <8 x bfloat>, ptr addrspace(3) %1095, align 16
  %1097 = add i32 %1085, 64
  %1098 = inttoptr i32 %1097 to ptr addrspace(3)
  %1099 = load <8 x bfloat>, ptr addrspace(3) %1098, align 16
  %1100 = add i32 %1085, 96
  %1101 = inttoptr i32 %1100 to ptr addrspace(3)
  %1102 = load <8 x bfloat>, ptr addrspace(3) %1101, align 16
  %1103 = add i32 %1085, 8768
  %1104 = inttoptr i32 %1103 to ptr addrspace(3)
  %1105 = load <8 x bfloat>, ptr addrspace(3) %1104, align 16
  %1106 = add i32 %1085, 8800
  %1107 = inttoptr i32 %1106 to ptr addrspace(3)
  %1108 = load <8 x bfloat>, ptr addrspace(3) %1107, align 16
  %1109 = add i32 %1085, 128
  %1110 = inttoptr i32 %1109 to ptr addrspace(3)
  %1111 = load <8 x bfloat>, ptr addrspace(3) %1110, align 16
  %1112 = add i32 %1085, 160
  %1113 = inttoptr i32 %1112 to ptr addrspace(3)
  %1114 = load <8 x bfloat>, ptr addrspace(3) %1113, align 16
  %1115 = add i32 %1085, 8832
  %1116 = inttoptr i32 %1115 to ptr addrspace(3)
  %1117 = load <8 x bfloat>, ptr addrspace(3) %1116, align 16
  %1118 = add i32 %1085, 8864
  %1119 = inttoptr i32 %1118 to ptr addrspace(3)
  %1120 = load <8 x bfloat>, ptr addrspace(3) %1119, align 16
  %1121 = add i32 %1085, 192
  %1122 = inttoptr i32 %1121 to ptr addrspace(3)
  %1123 = load <8 x bfloat>, ptr addrspace(3) %1122, align 16
  %1124 = add i32 %1085, 224
  %1125 = inttoptr i32 %1124 to ptr addrspace(3)
  %1126 = load <8 x bfloat>, ptr addrspace(3) %1125, align 16
  %1127 = add i32 %1085, 8896
  %1128 = inttoptr i32 %1127 to ptr addrspace(3)
  %1129 = load <8 x bfloat>, ptr addrspace(3) %1128, align 16
  %1130 = add i32 %1085, 8928
  %1131 = inttoptr i32 %1130 to ptr addrspace(3)
  %1132 = load <8 x bfloat>, ptr addrspace(3) %1131, align 16
  %1133 = add i32 %1085, 4352
  %1134 = inttoptr i32 %1133 to ptr addrspace(3)
  %1135 = load <8 x bfloat>, ptr addrspace(3) %1134, align 16
  %1136 = add i32 %1085, 4384
  %1137 = inttoptr i32 %1136 to ptr addrspace(3)
  %1138 = load <8 x bfloat>, ptr addrspace(3) %1137, align 16
  %1139 = add i32 %1085, 13056
  %1140 = inttoptr i32 %1139 to ptr addrspace(3)
  %1141 = load <8 x bfloat>, ptr addrspace(3) %1140, align 16
  %1142 = add i32 %1085, 13088
  %1143 = inttoptr i32 %1142 to ptr addrspace(3)
  %1144 = load <8 x bfloat>, ptr addrspace(3) %1143, align 16
  %1145 = add i32 %1085, 4416
  %1146 = inttoptr i32 %1145 to ptr addrspace(3)
  %1147 = load <8 x bfloat>, ptr addrspace(3) %1146, align 16
  %1148 = add i32 %1085, 4448
  %1149 = inttoptr i32 %1148 to ptr addrspace(3)
  %1150 = load <8 x bfloat>, ptr addrspace(3) %1149, align 16
  %1151 = add i32 %1085, 13120
  %1152 = inttoptr i32 %1151 to ptr addrspace(3)
  %1153 = load <8 x bfloat>, ptr addrspace(3) %1152, align 16
  %1154 = add i32 %1085, 13152
  %1155 = inttoptr i32 %1154 to ptr addrspace(3)
  %1156 = load <8 x bfloat>, ptr addrspace(3) %1155, align 16
  %1157 = add i32 %1085, 4480
  %1158 = inttoptr i32 %1157 to ptr addrspace(3)
  %1159 = load <8 x bfloat>, ptr addrspace(3) %1158, align 16
  %1160 = add i32 %1085, 4512
  %1161 = inttoptr i32 %1160 to ptr addrspace(3)
  %1162 = load <8 x bfloat>, ptr addrspace(3) %1161, align 16
  %1163 = add i32 %1085, 13184
  %1164 = inttoptr i32 %1163 to ptr addrspace(3)
  %1165 = load <8 x bfloat>, ptr addrspace(3) %1164, align 16
  %1166 = add i32 %1085, 13216
  %1167 = inttoptr i32 %1166 to ptr addrspace(3)
  %1168 = load <8 x bfloat>, ptr addrspace(3) %1167, align 16
  %1169 = add i32 %1085, 4544
  %1170 = inttoptr i32 %1169 to ptr addrspace(3)
  %1171 = load <8 x bfloat>, ptr addrspace(3) %1170, align 16
  %1172 = add i32 %1085, 4576
  %1173 = inttoptr i32 %1172 to ptr addrspace(3)
  %1174 = load <8 x bfloat>, ptr addrspace(3) %1173, align 16
  %1175 = add i32 %1085, 13248
  %1176 = inttoptr i32 %1175 to ptr addrspace(3)
  %1177 = load <8 x bfloat>, ptr addrspace(3) %1176, align 16
  %1178 = add i32 %1085, 13280
  %1179 = inttoptr i32 %1178 to ptr addrspace(3)
  %1180 = load <8 x bfloat>, ptr addrspace(3) %1179, align 16
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1181 = extractelement <8 x float> %963, i64 0
  %1182 = call float @llvm.fma.f32(float %1181, float %365, float %366)
  %1183 = extractelement <8 x float> %963, i64 1
  %1184 = call float @llvm.fma.f32(float %1183, float %365, float %366)
  %1185 = extractelement <8 x float> %963, i64 2
  %1186 = call float @llvm.fma.f32(float %1185, float %365, float %366)
  %1187 = extractelement <8 x float> %963, i64 3
  %1188 = call float @llvm.fma.f32(float %1187, float %365, float %366)
  %1189 = extractelement <8 x float> %963, i64 4
  %1190 = call float @llvm.fma.f32(float %1189, float %365, float %366)
  %1191 = extractelement <8 x float> %963, i64 5
  %1192 = call float @llvm.fma.f32(float %1191, float %365, float %366)
  %1193 = extractelement <8 x float> %963, i64 6
  %1194 = call float @llvm.fma.f32(float %1193, float %365, float %366)
  %1195 = extractelement <8 x float> %963, i64 7
  %1196 = call float @llvm.fma.f32(float %1195, float %365, float %366)
  %1197 = call float @llvm.amdgcn.exp2.f32(float %1182)
  %1198 = call float @llvm.amdgcn.exp2.f32(float %1184)
  %1199 = call float @llvm.amdgcn.exp2.f32(float %1186)
  %1200 = call float @llvm.amdgcn.exp2.f32(float %1188)
  %1201 = call float @llvm.amdgcn.exp2.f32(float %1190)
  %1202 = call float @llvm.amdgcn.exp2.f32(float %1192)
  %1203 = call float @llvm.amdgcn.exp2.f32(float %1194)
  %1204 = call float @llvm.amdgcn.exp2.f32(float %1196)
  %1205 = extractelement <8 x float> %964, i64 0
  %1206 = call float @llvm.fma.f32(float %1205, float %16, float %371)
  %1207 = fmul float %1197, %1206
  %1208 = fptrunc float %1207 to bfloat
  %1209 = extractelement <8 x float> %964, i64 1
  %1210 = call float @llvm.fma.f32(float %1209, float %16, float %371)
  %1211 = fmul float %1198, %1210
  %1212 = fptrunc float %1211 to bfloat
  %1213 = extractelement <8 x float> %964, i64 2
  %1214 = call float @llvm.fma.f32(float %1213, float %16, float %371)
  %1215 = fmul float %1199, %1214
  %1216 = fptrunc float %1215 to bfloat
  %1217 = extractelement <8 x float> %964, i64 3
  %1218 = call float @llvm.fma.f32(float %1217, float %16, float %371)
  %1219 = fmul float %1200, %1218
  %1220 = fptrunc float %1219 to bfloat
  %1221 = extractelement <8 x float> %964, i64 4
  %1222 = call float @llvm.fma.f32(float %1221, float %16, float %371)
  %1223 = fmul float %1201, %1222
  %1224 = fptrunc float %1223 to bfloat
  %1225 = extractelement <8 x float> %964, i64 5
  %1226 = call float @llvm.fma.f32(float %1225, float %16, float %371)
  %1227 = fmul float %1202, %1226
  %1228 = fptrunc float %1227 to bfloat
  %1229 = extractelement <8 x float> %964, i64 6
  %1230 = call float @llvm.fma.f32(float %1229, float %16, float %371)
  %1231 = fmul float %1203, %1230
  %1232 = fptrunc float %1231 to bfloat
  %1233 = extractelement <8 x float> %964, i64 7
  %1234 = call float @llvm.fma.f32(float %1233, float %16, float %371)
  %1235 = fmul float %1204, %1234
  %1236 = fptrunc float %1235 to bfloat
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1237 = insertelement <16 x bfloat> poison, bfloat %800, i64 0
  %1238 = insertelement <16 x bfloat> %1237, bfloat %804, i64 1
  %1239 = insertelement <16 x bfloat> %1238, bfloat %808, i64 2
  %1240 = insertelement <16 x bfloat> %1239, bfloat %812, i64 3
  %1241 = insertelement <16 x bfloat> %1240, bfloat %816, i64 4
  %1242 = insertelement <16 x bfloat> %1241, bfloat %820, i64 5
  %1243 = insertelement <16 x bfloat> %1242, bfloat %824, i64 6
  %1244 = insertelement <16 x bfloat> %1243, bfloat %828, i64 7
  %1245 = insertelement <16 x bfloat> %1244, bfloat %1208, i64 8
  %1246 = insertelement <16 x bfloat> %1245, bfloat %1212, i64 9
  %1247 = insertelement <16 x bfloat> %1246, bfloat %1216, i64 10
  %1248 = insertelement <16 x bfloat> %1247, bfloat %1220, i64 11
  %1249 = insertelement <16 x bfloat> %1248, bfloat %1224, i64 12
  %1250 = insertelement <16 x bfloat> %1249, bfloat %1228, i64 13
  %1251 = insertelement <16 x bfloat> %1250, bfloat %1232, i64 14
  %1252 = insertelement <16 x bfloat> %1251, bfloat %1236, i64 15
  %1253 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1252, <16 x bfloat> %1034, i16 0, <8 x float> %603, i1 false, i1 false)
  %1254 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1252, <16 x bfloat> %1041, i16 0, <8 x float> %604, i1 false, i1 false)
  %1255 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1252, <16 x bfloat> %1048, i16 0, <8 x float> %605, i1 false, i1 false)
  %1256 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1252, <16 x bfloat> %1055, i16 0, <8 x float> %606, i1 false, i1 false)
  %1257 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1252, <16 x bfloat> %1062, i16 0, <8 x float> %607, i1 false, i1 false)
  %1258 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1252, <16 x bfloat> %1069, i16 0, <8 x float> %608, i1 false, i1 false)
  %1259 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1252, <16 x bfloat> %1076, i16 0, <8 x float> %609, i1 false, i1 false)
  %1260 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1252, <16 x bfloat> %1083, i16 0, <8 x float> %610, i1 false, i1 false)
  %1261 = extractelement <8 x float> %965, i64 0
  %1262 = call float @llvm.fma.f32(float %1261, float %365, float %367)
  %1263 = extractelement <8 x float> %965, i64 1
  %1264 = call float @llvm.fma.f32(float %1263, float %365, float %367)
  %1265 = extractelement <8 x float> %965, i64 2
  %1266 = call float @llvm.fma.f32(float %1265, float %365, float %367)
  %1267 = extractelement <8 x float> %965, i64 3
  %1268 = call float @llvm.fma.f32(float %1267, float %365, float %367)
  %1269 = extractelement <8 x float> %965, i64 4
  %1270 = call float @llvm.fma.f32(float %1269, float %365, float %367)
  %1271 = extractelement <8 x float> %965, i64 5
  %1272 = call float @llvm.fma.f32(float %1271, float %365, float %367)
  %1273 = extractelement <8 x float> %965, i64 6
  %1274 = call float @llvm.fma.f32(float %1273, float %365, float %367)
  %1275 = extractelement <8 x float> %965, i64 7
  %1276 = call float @llvm.fma.f32(float %1275, float %365, float %367)
  %1277 = call float @llvm.amdgcn.exp2.f32(float %1262)
  %1278 = call float @llvm.amdgcn.exp2.f32(float %1264)
  %1279 = call float @llvm.amdgcn.exp2.f32(float %1266)
  %1280 = call float @llvm.amdgcn.exp2.f32(float %1268)
  %1281 = call float @llvm.amdgcn.exp2.f32(float %1270)
  %1282 = call float @llvm.amdgcn.exp2.f32(float %1272)
  %1283 = call float @llvm.amdgcn.exp2.f32(float %1274)
  %1284 = call float @llvm.amdgcn.exp2.f32(float %1276)
  %1285 = extractelement <8 x float> %966, i64 0
  %1286 = call float @llvm.fma.f32(float %1285, float %16, float %372)
  %1287 = fmul float %1277, %1286
  %1288 = fptrunc float %1287 to bfloat
  %1289 = extractelement <8 x float> %966, i64 1
  %1290 = call float @llvm.fma.f32(float %1289, float %16, float %372)
  %1291 = fmul float %1278, %1290
  %1292 = fptrunc float %1291 to bfloat
  %1293 = extractelement <8 x float> %966, i64 2
  %1294 = call float @llvm.fma.f32(float %1293, float %16, float %372)
  %1295 = fmul float %1279, %1294
  %1296 = fptrunc float %1295 to bfloat
  %1297 = extractelement <8 x float> %966, i64 3
  %1298 = call float @llvm.fma.f32(float %1297, float %16, float %372)
  %1299 = fmul float %1280, %1298
  %1300 = fptrunc float %1299 to bfloat
  %1301 = extractelement <8 x float> %966, i64 4
  %1302 = call float @llvm.fma.f32(float %1301, float %16, float %372)
  %1303 = fmul float %1281, %1302
  %1304 = fptrunc float %1303 to bfloat
  %1305 = extractelement <8 x float> %966, i64 5
  %1306 = call float @llvm.fma.f32(float %1305, float %16, float %372)
  %1307 = fmul float %1282, %1306
  %1308 = fptrunc float %1307 to bfloat
  %1309 = extractelement <8 x float> %966, i64 6
  %1310 = call float @llvm.fma.f32(float %1309, float %16, float %372)
  %1311 = fmul float %1283, %1310
  %1312 = fptrunc float %1311 to bfloat
  %1313 = extractelement <8 x float> %966, i64 7
  %1314 = call float @llvm.fma.f32(float %1313, float %16, float %372)
  %1315 = fmul float %1284, %1314
  %1316 = fptrunc float %1315 to bfloat
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1317 = insertelement <16 x bfloat> poison, bfloat %866, i64 0
  %1318 = insertelement <16 x bfloat> %1317, bfloat %870, i64 1
  %1319 = insertelement <16 x bfloat> %1318, bfloat %874, i64 2
  %1320 = insertelement <16 x bfloat> %1319, bfloat %878, i64 3
  %1321 = insertelement <16 x bfloat> %1320, bfloat %882, i64 4
  %1322 = insertelement <16 x bfloat> %1321, bfloat %886, i64 5
  %1323 = insertelement <16 x bfloat> %1322, bfloat %890, i64 6
  %1324 = insertelement <16 x bfloat> %1323, bfloat %894, i64 7
  %1325 = insertelement <16 x bfloat> %1324, bfloat %1288, i64 8
  %1326 = insertelement <16 x bfloat> %1325, bfloat %1292, i64 9
  %1327 = insertelement <16 x bfloat> %1326, bfloat %1296, i64 10
  %1328 = insertelement <16 x bfloat> %1327, bfloat %1300, i64 11
  %1329 = insertelement <16 x bfloat> %1328, bfloat %1304, i64 12
  %1330 = insertelement <16 x bfloat> %1329, bfloat %1308, i64 13
  %1331 = insertelement <16 x bfloat> %1330, bfloat %1312, i64 14
  %1332 = insertelement <16 x bfloat> %1331, bfloat %1316, i64 15
  %1333 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1332, <16 x bfloat> %1034, i16 0, <8 x float> %611, i1 false, i1 false)
  %1334 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1332, <16 x bfloat> %1041, i16 0, <8 x float> %612, i1 false, i1 false)
  %1335 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1332, <16 x bfloat> %1048, i16 0, <8 x float> %613, i1 false, i1 false)
  %1336 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1332, <16 x bfloat> %1055, i16 0, <8 x float> %614, i1 false, i1 false)
  %1337 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1332, <16 x bfloat> %1062, i16 0, <8 x float> %615, i1 false, i1 false)
  %1338 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1332, <16 x bfloat> %1069, i16 0, <8 x float> %616, i1 false, i1 false)
  %1339 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1332, <16 x bfloat> %1076, i16 0, <8 x float> %617, i1 false, i1 false)
  %1340 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1332, <16 x bfloat> %1083, i16 0, <8 x float> %618, i1 false, i1 false)
  %1341 = extractelement <8 x float> %967, i64 0
  %1342 = call float @llvm.fma.f32(float %1341, float %365, float %368)
  %1343 = extractelement <8 x float> %967, i64 1
  %1344 = call float @llvm.fma.f32(float %1343, float %365, float %368)
  %1345 = extractelement <8 x float> %967, i64 2
  %1346 = call float @llvm.fma.f32(float %1345, float %365, float %368)
  %1347 = extractelement <8 x float> %967, i64 3
  %1348 = call float @llvm.fma.f32(float %1347, float %365, float %368)
  %1349 = extractelement <8 x float> %967, i64 4
  %1350 = call float @llvm.fma.f32(float %1349, float %365, float %368)
  %1351 = extractelement <8 x float> %967, i64 5
  %1352 = call float @llvm.fma.f32(float %1351, float %365, float %368)
  %1353 = extractelement <8 x float> %967, i64 6
  %1354 = call float @llvm.fma.f32(float %1353, float %365, float %368)
  %1355 = extractelement <8 x float> %967, i64 7
  %1356 = call float @llvm.fma.f32(float %1355, float %365, float %368)
  %1357 = call float @llvm.amdgcn.exp2.f32(float %1342)
  %1358 = call float @llvm.amdgcn.exp2.f32(float %1344)
  %1359 = call float @llvm.amdgcn.exp2.f32(float %1346)
  %1360 = call float @llvm.amdgcn.exp2.f32(float %1348)
  %1361 = call float @llvm.amdgcn.exp2.f32(float %1350)
  %1362 = call float @llvm.amdgcn.exp2.f32(float %1352)
  %1363 = call float @llvm.amdgcn.exp2.f32(float %1354)
  %1364 = call float @llvm.amdgcn.exp2.f32(float %1356)
  %1365 = extractelement <8 x float> %968, i64 0
  %1366 = call float @llvm.fma.f32(float %1365, float %16, float %373)
  %1367 = fmul float %1357, %1366
  %1368 = fptrunc float %1367 to bfloat
  %1369 = extractelement <8 x float> %968, i64 1
  %1370 = call float @llvm.fma.f32(float %1369, float %16, float %373)
  %1371 = fmul float %1358, %1370
  %1372 = fptrunc float %1371 to bfloat
  %1373 = extractelement <8 x float> %968, i64 2
  %1374 = call float @llvm.fma.f32(float %1373, float %16, float %373)
  %1375 = fmul float %1359, %1374
  %1376 = fptrunc float %1375 to bfloat
  %1377 = extractelement <8 x float> %968, i64 3
  %1378 = call float @llvm.fma.f32(float %1377, float %16, float %373)
  %1379 = fmul float %1360, %1378
  %1380 = fptrunc float %1379 to bfloat
  %1381 = extractelement <8 x float> %968, i64 4
  %1382 = call float @llvm.fma.f32(float %1381, float %16, float %373)
  %1383 = fmul float %1361, %1382
  %1384 = fptrunc float %1383 to bfloat
  %1385 = extractelement <8 x float> %968, i64 5
  %1386 = call float @llvm.fma.f32(float %1385, float %16, float %373)
  %1387 = fmul float %1362, %1386
  %1388 = fptrunc float %1387 to bfloat
  %1389 = extractelement <8 x float> %968, i64 6
  %1390 = call float @llvm.fma.f32(float %1389, float %16, float %373)
  %1391 = fmul float %1363, %1390
  %1392 = fptrunc float %1391 to bfloat
  %1393 = extractelement <8 x float> %968, i64 7
  %1394 = call float @llvm.fma.f32(float %1393, float %16, float %373)
  %1395 = fmul float %1364, %1394
  %1396 = fptrunc float %1395 to bfloat
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1397 = insertelement <16 x bfloat> poison, bfloat %932, i64 0
  %1398 = insertelement <16 x bfloat> %1397, bfloat %936, i64 1
  %1399 = insertelement <16 x bfloat> %1398, bfloat %940, i64 2
  %1400 = insertelement <16 x bfloat> %1399, bfloat %944, i64 3
  %1401 = insertelement <16 x bfloat> %1400, bfloat %948, i64 4
  %1402 = insertelement <16 x bfloat> %1401, bfloat %952, i64 5
  %1403 = insertelement <16 x bfloat> %1402, bfloat %956, i64 6
  %1404 = insertelement <16 x bfloat> %1403, bfloat %960, i64 7
  %1405 = insertelement <16 x bfloat> %1404, bfloat %1368, i64 8
  %1406 = insertelement <16 x bfloat> %1405, bfloat %1372, i64 9
  %1407 = insertelement <16 x bfloat> %1406, bfloat %1376, i64 10
  %1408 = insertelement <16 x bfloat> %1407, bfloat %1380, i64 11
  %1409 = insertelement <16 x bfloat> %1408, bfloat %1384, i64 12
  %1410 = insertelement <16 x bfloat> %1409, bfloat %1388, i64 13
  %1411 = insertelement <16 x bfloat> %1410, bfloat %1392, i64 14
  %1412 = insertelement <16 x bfloat> %1411, bfloat %1396, i64 15
  %1413 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1412, <16 x bfloat> %1034, i16 0, <8 x float> %619, i1 false, i1 false)
  %1414 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1412, <16 x bfloat> %1041, i16 0, <8 x float> %620, i1 false, i1 false)
  %1415 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1412, <16 x bfloat> %1048, i16 0, <8 x float> %621, i1 false, i1 false)
  %1416 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1412, <16 x bfloat> %1055, i16 0, <8 x float> %622, i1 false, i1 false)
  %1417 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1412, <16 x bfloat> %1062, i16 0, <8 x float> %623, i1 false, i1 false)
  %1418 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1412, <16 x bfloat> %1069, i16 0, <8 x float> %624, i1 false, i1 false)
  %1419 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1412, <16 x bfloat> %1076, i16 0, <8 x float> %625, i1 false, i1 false)
  %1420 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1412, <16 x bfloat> %1083, i16 0, <8 x float> %626, i1 false, i1 false)
  %1421 = extractelement <8 x float> %969, i64 0
  %1422 = call float @llvm.fma.f32(float %1421, float %365, float %369)
  %1423 = extractelement <8 x float> %969, i64 1
  %1424 = call float @llvm.fma.f32(float %1423, float %365, float %369)
  %1425 = extractelement <8 x float> %969, i64 2
  %1426 = call float @llvm.fma.f32(float %1425, float %365, float %369)
  %1427 = extractelement <8 x float> %969, i64 3
  %1428 = call float @llvm.fma.f32(float %1427, float %365, float %369)
  %1429 = extractelement <8 x float> %969, i64 4
  %1430 = call float @llvm.fma.f32(float %1429, float %365, float %369)
  %1431 = extractelement <8 x float> %969, i64 5
  %1432 = call float @llvm.fma.f32(float %1431, float %365, float %369)
  %1433 = extractelement <8 x float> %969, i64 6
  %1434 = call float @llvm.fma.f32(float %1433, float %365, float %369)
  %1435 = extractelement <8 x float> %969, i64 7
  %1436 = call float @llvm.fma.f32(float %1435, float %365, float %369)
  %1437 = call float @llvm.amdgcn.exp2.f32(float %1422)
  %1438 = call float @llvm.amdgcn.exp2.f32(float %1424)
  %1439 = call float @llvm.amdgcn.exp2.f32(float %1426)
  %1440 = call float @llvm.amdgcn.exp2.f32(float %1428)
  %1441 = call float @llvm.amdgcn.exp2.f32(float %1430)
  %1442 = call float @llvm.amdgcn.exp2.f32(float %1432)
  %1443 = call float @llvm.amdgcn.exp2.f32(float %1434)
  %1444 = call float @llvm.amdgcn.exp2.f32(float %1436)
  %1445 = extractelement <8 x float> %970, i64 0
  %1446 = call float @llvm.fma.f32(float %1445, float %16, float %374)
  %1447 = fmul float %1437, %1446
  %1448 = fptrunc float %1447 to bfloat
  %1449 = extractelement <8 x float> %970, i64 1
  %1450 = call float @llvm.fma.f32(float %1449, float %16, float %374)
  %1451 = fmul float %1438, %1450
  %1452 = fptrunc float %1451 to bfloat
  %1453 = extractelement <8 x float> %970, i64 2
  %1454 = call float @llvm.fma.f32(float %1453, float %16, float %374)
  %1455 = fmul float %1439, %1454
  %1456 = fptrunc float %1455 to bfloat
  %1457 = extractelement <8 x float> %970, i64 3
  %1458 = call float @llvm.fma.f32(float %1457, float %16, float %374)
  %1459 = fmul float %1440, %1458
  %1460 = fptrunc float %1459 to bfloat
  %1461 = extractelement <8 x float> %970, i64 4
  %1462 = call float @llvm.fma.f32(float %1461, float %16, float %374)
  %1463 = fmul float %1441, %1462
  %1464 = fptrunc float %1463 to bfloat
  %1465 = extractelement <8 x float> %970, i64 5
  %1466 = call float @llvm.fma.f32(float %1465, float %16, float %374)
  %1467 = fmul float %1442, %1466
  %1468 = fptrunc float %1467 to bfloat
  %1469 = extractelement <8 x float> %970, i64 6
  %1470 = call float @llvm.fma.f32(float %1469, float %16, float %374)
  %1471 = fmul float %1443, %1470
  %1472 = fptrunc float %1471 to bfloat
  %1473 = extractelement <8 x float> %970, i64 7
  %1474 = call float @llvm.fma.f32(float %1473, float %16, float %374)
  %1475 = fmul float %1444, %1474
  %1476 = fptrunc float %1475 to bfloat
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 2, i32 0)
  call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 1, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1477 = insertelement <16 x bfloat> poison, bfloat %998, i64 0
  %1478 = insertelement <16 x bfloat> %1477, bfloat %1002, i64 1
  %1479 = insertelement <16 x bfloat> %1478, bfloat %1006, i64 2
  %1480 = insertelement <16 x bfloat> %1479, bfloat %1010, i64 3
  %1481 = insertelement <16 x bfloat> %1480, bfloat %1014, i64 4
  %1482 = insertelement <16 x bfloat> %1481, bfloat %1018, i64 5
  %1483 = insertelement <16 x bfloat> %1482, bfloat %1022, i64 6
  %1484 = insertelement <16 x bfloat> %1483, bfloat %1026, i64 7
  %1485 = insertelement <16 x bfloat> %1484, bfloat %1448, i64 8
  %1486 = insertelement <16 x bfloat> %1485, bfloat %1452, i64 9
  %1487 = insertelement <16 x bfloat> %1486, bfloat %1456, i64 10
  %1488 = insertelement <16 x bfloat> %1487, bfloat %1460, i64 11
  %1489 = insertelement <16 x bfloat> %1488, bfloat %1464, i64 12
  %1490 = insertelement <16 x bfloat> %1489, bfloat %1468, i64 13
  %1491 = insertelement <16 x bfloat> %1490, bfloat %1472, i64 14
  %1492 = insertelement <16 x bfloat> %1491, bfloat %1476, i64 15
  %1493 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1492, <16 x bfloat> %1034, i16 0, <8 x float> %627, i1 false, i1 false)
  %1494 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1492, <16 x bfloat> %1041, i16 0, <8 x float> %628, i1 false, i1 false)
  %1495 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1492, <16 x bfloat> %1048, i16 0, <8 x float> %629, i1 false, i1 false)
  %1496 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1492, <16 x bfloat> %1055, i16 0, <8 x float> %630, i1 false, i1 false)
  %1497 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1492, <16 x bfloat> %1062, i16 0, <8 x float> %631, i1 false, i1 false)
  %1498 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1492, <16 x bfloat> %1069, i16 0, <8 x float> %632, i1 false, i1 false)
  %1499 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1492, <16 x bfloat> %1076, i16 0, <8 x float> %633, i1 false, i1 false)
  %1500 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1492, <16 x bfloat> %1083, i16 0, <8 x float> %634, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1501 = add i64 %602, 1
  br label %601

1502:                                             ; preds = %601
  %1503 = sub i32 %408, %422
  %1504 = sext i32 %1503 to i64
  br label %1505

1505:                                             ; preds = %1573, %1502
  %1506 = phi i64 [ %2620, %1573 ], [ 0, %1502 ]
  %1507 = phi <8 x float> [ %2588, %1573 ], [ %603, %1502 ]
  %1508 = phi <8 x float> [ %2589, %1573 ], [ %604, %1502 ]
  %1509 = phi <8 x float> [ %2590, %1573 ], [ %605, %1502 ]
  %1510 = phi <8 x float> [ %2591, %1573 ], [ %606, %1502 ]
  %1511 = phi <8 x float> [ %2592, %1573 ], [ %607, %1502 ]
  %1512 = phi <8 x float> [ %2593, %1573 ], [ %608, %1502 ]
  %1513 = phi <8 x float> [ %2594, %1573 ], [ %609, %1502 ]
  %1514 = phi <8 x float> [ %2595, %1573 ], [ %610, %1502 ]
  %1515 = phi <8 x float> [ %2596, %1573 ], [ %611, %1502 ]
  %1516 = phi <8 x float> [ %2597, %1573 ], [ %612, %1502 ]
  %1517 = phi <8 x float> [ %2598, %1573 ], [ %613, %1502 ]
  %1518 = phi <8 x float> [ %2599, %1573 ], [ %614, %1502 ]
  %1519 = phi <8 x float> [ %2600, %1573 ], [ %615, %1502 ]
  %1520 = phi <8 x float> [ %2601, %1573 ], [ %616, %1502 ]
  %1521 = phi <8 x float> [ %2602, %1573 ], [ %617, %1502 ]
  %1522 = phi <8 x float> [ %2603, %1573 ], [ %618, %1502 ]
  %1523 = phi <8 x float> [ %2604, %1573 ], [ %619, %1502 ]
  %1524 = phi <8 x float> [ %2605, %1573 ], [ %620, %1502 ]
  %1525 = phi <8 x float> [ %2606, %1573 ], [ %621, %1502 ]
  %1526 = phi <8 x float> [ %2607, %1573 ], [ %622, %1502 ]
  %1527 = phi <8 x float> [ %2608, %1573 ], [ %623, %1502 ]
  %1528 = phi <8 x float> [ %2609, %1573 ], [ %624, %1502 ]
  %1529 = phi <8 x float> [ %2610, %1573 ], [ %625, %1502 ]
  %1530 = phi <8 x float> [ %2611, %1573 ], [ %626, %1502 ]
  %1531 = phi <8 x float> [ %2612, %1573 ], [ %627, %1502 ]
  %1532 = phi <8 x float> [ %2613, %1573 ], [ %628, %1502 ]
  %1533 = phi <8 x float> [ %2614, %1573 ], [ %629, %1502 ]
  %1534 = phi <8 x float> [ %2615, %1573 ], [ %630, %1502 ]
  %1535 = phi <8 x float> [ %2616, %1573 ], [ %631, %1502 ]
  %1536 = phi <8 x float> [ %2617, %1573 ], [ %632, %1502 ]
  %1537 = phi <8 x float> [ %2618, %1573 ], [ %633, %1502 ]
  %1538 = phi <8 x float> [ %2619, %1573 ], [ %634, %1502 ]
  %1539 = phi <8 x bfloat> [ %1769, %1573 ], [ %635, %1502 ]
  %1540 = phi <8 x bfloat> [ %1772, %1573 ], [ %636, %1502 ]
  %1541 = phi <8 x bfloat> [ %1775, %1573 ], [ %637, %1502 ]
  %1542 = phi <8 x bfloat> [ %1778, %1573 ], [ %638, %1502 ]
  %1543 = phi <8 x bfloat> [ %1781, %1573 ], [ %639, %1502 ]
  %1544 = phi <8 x bfloat> [ %1784, %1573 ], [ %640, %1502 ]
  %1545 = phi <8 x bfloat> [ %1787, %1573 ], [ %641, %1502 ]
  %1546 = phi <8 x bfloat> [ %1790, %1573 ], [ %642, %1502 ]
  %1547 = phi <8 x bfloat> [ %1793, %1573 ], [ %643, %1502 ]
  %1548 = phi <8 x bfloat> [ %1796, %1573 ], [ %644, %1502 ]
  %1549 = phi <8 x bfloat> [ %1799, %1573 ], [ %645, %1502 ]
  %1550 = phi <8 x bfloat> [ %1802, %1573 ], [ %646, %1502 ]
  %1551 = phi <8 x bfloat> [ %1805, %1573 ], [ %647, %1502 ]
  %1552 = phi <8 x bfloat> [ %1808, %1573 ], [ %648, %1502 ]
  %1553 = phi <8 x bfloat> [ %1811, %1573 ], [ %649, %1502 ]
  %1554 = phi <8 x bfloat> [ %1814, %1573 ], [ %650, %1502 ]
  %1555 = phi <8 x bfloat> [ %1817, %1573 ], [ %651, %1502 ]
  %1556 = phi <8 x bfloat> [ %1820, %1573 ], [ %652, %1502 ]
  %1557 = phi <8 x bfloat> [ %1823, %1573 ], [ %653, %1502 ]
  %1558 = phi <8 x bfloat> [ %1826, %1573 ], [ %654, %1502 ]
  %1559 = phi <8 x bfloat> [ %1829, %1573 ], [ %655, %1502 ]
  %1560 = phi <8 x bfloat> [ %1832, %1573 ], [ %656, %1502 ]
  %1561 = phi <8 x bfloat> [ %1835, %1573 ], [ %657, %1502 ]
  %1562 = phi <8 x bfloat> [ %1838, %1573 ], [ %658, %1502 ]
  %1563 = phi <8 x bfloat> [ %1841, %1573 ], [ %659, %1502 ]
  %1564 = phi <8 x bfloat> [ %1844, %1573 ], [ %660, %1502 ]
  %1565 = phi <8 x bfloat> [ %1847, %1573 ], [ %661, %1502 ]
  %1566 = phi <8 x bfloat> [ %1850, %1573 ], [ %662, %1502 ]
  %1567 = phi <8 x bfloat> [ %1853, %1573 ], [ %663, %1502 ]
  %1568 = phi <8 x bfloat> [ %1856, %1573 ], [ %664, %1502 ]
  %1569 = phi <8 x bfloat> [ %1859, %1573 ], [ %665, %1502 ]
  %1570 = phi <8 x bfloat> [ %1862, %1573 ], [ %666, %1502 ]
  %1571 = phi i32 [ %1583, %1573 ], [ %667, %1502 ]
  %1572 = icmp slt i64 %1506, %1504
  br i1 %1572, label %1573, label %2621

1573:                                             ; preds = %1505
  %1574 = trunc i64 %1506 to i32
  %1575 = add i32 %1574, %422
  %1576 = add i32 %1575, 2
  %1577 = call i32 @llvm.smin.i32(i32 %1576, i32 %423)
  %1578 = icmp eq i32 %1571, 0
  %1579 = sub i32 %1571, 17408
  %1580 = select i1 %1578, i32 34816, i32 %1579
  %1581 = icmp eq i32 %1571, 34816
  %1582 = add i32 %1571, 17408
  %1583 = select i1 %1581, i32 0, i32 %1582
  %1584 = mul i32 %1575, 32
  %1585 = mul i32 %1577, 32
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1586 = add i32 %424, %1585
  %1587 = sext i32 %1586 to i64
  %1588 = mul i64 %1587, %426
  %1589 = add i64 %1588, %428
  %1590 = mul i64 %1589, 128
  %1591 = sub i32 %18, %1585
  %1592 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1580
  %1593 = add i32 %1592, 8704
  %1594 = getelementptr bfloat, ptr addrspace(1) %2, i64 %1590
  %1595 = inttoptr i32 %1592 to ptr addrspace(3)
  %1596 = ptrtoint ptr addrspace(1) %1594 to i64
  %1597 = ptrtoint ptr addrspace(3) %1595 to i32
  %1598 = trunc i64 %1596 to i32
  %1599 = lshr i64 %1596, 32
  %1600 = trunc i64 %1599 to i32
  %1601 = or i32 %1600, -2147483648
  %1602 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %1597, i64 1
  %1603 = insertelement <4 x i32> %1602, i32 %1598, i64 2
  %1604 = insertelement <4 x i32> %1603, i32 %1601, i64 3
  %1605 = call i32 @llvm.smax.i32(i32 %1591, i32 0)
  %1606 = and i32 %1605, 65535
  %1607 = shl i32 %1606, 16
  %1608 = or i32 %1607, 32767
  %1609 = lshr i32 %1605, 16
  %1610 = and i32 %1609, 65535
  %1611 = or i32 %1610, 8388608
  %1612 = insertelement <8 x i32> <i32 122748928, i32 -65536, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, i32 %1608, i64 2
  %1613 = insertelement <8 x i32> %1612, i32 %1611, i64 3
  %1614 = insertelement <8 x i32> %1613, i32 32, i64 4
  %1615 = insertelement <8 x i32> %1614, i32 %449, i64 5
  %1616 = insertelement <8 x i32> %1615, i32 %452, i64 6
  %1617 = insertelement <8 x i32> %1616, i32 0, i64 7
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %1604, <8 x i32> %1617, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  %1618 = getelementptr bfloat, ptr addrspace(1) %4, i64 %1590
  %1619 = inttoptr i32 %1593 to ptr addrspace(3)
  %1620 = ptrtoint ptr addrspace(1) %1618 to i64
  %1621 = ptrtoint ptr addrspace(3) %1619 to i32
  %1622 = trunc i64 %1620 to i32
  %1623 = lshr i64 %1620, 32
  %1624 = trunc i64 %1623 to i32
  %1625 = or i32 %1624, -2147483648
  %1626 = insertelement <4 x i32> <i32 1, i32 poison, i32 poison, i32 poison>, i32 %1621, i64 1
  %1627 = insertelement <4 x i32> %1626, i32 %1622, i64 2
  %1628 = insertelement <4 x i32> %1627, i32 %1625, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %1628, <8 x i32> %1617, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1629 = shufflevector <8 x bfloat> %1539, <8 x bfloat> %1540, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1630 = shufflevector <8 x bfloat> %1541, <8 x bfloat> %1542, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1631 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1629, <16 x bfloat> %113, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1632 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1630, <16 x bfloat> %265, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1633 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1629, <16 x bfloat> %153, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1634 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1630, <16 x bfloat> %285, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1635 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1629, <16 x bfloat> %193, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1636 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1630, <16 x bfloat> %305, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1637 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1629, <16 x bfloat> %233, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1638 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1630, <16 x bfloat> %325, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1639 = shufflevector <8 x bfloat> %1543, <8 x bfloat> %1544, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1640 = shufflevector <8 x bfloat> %1545, <8 x bfloat> %1546, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1641 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1639, <16 x bfloat> %122, i16 0, <8 x float> %1631, i1 false, i1 false)
  %1642 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1640, <16 x bfloat> %270, i16 0, <8 x float> %1632, i1 false, i1 false)
  %1643 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1639, <16 x bfloat> %162, i16 0, <8 x float> %1633, i1 false, i1 false)
  %1644 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1640, <16 x bfloat> %290, i16 0, <8 x float> %1634, i1 false, i1 false)
  %1645 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1639, <16 x bfloat> %202, i16 0, <8 x float> %1635, i1 false, i1 false)
  %1646 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1640, <16 x bfloat> %310, i16 0, <8 x float> %1636, i1 false, i1 false)
  %1647 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1639, <16 x bfloat> %242, i16 0, <8 x float> %1637, i1 false, i1 false)
  %1648 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1640, <16 x bfloat> %330, i16 0, <8 x float> %1638, i1 false, i1 false)
  %1649 = shufflevector <8 x bfloat> %1547, <8 x bfloat> %1548, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1650 = shufflevector <8 x bfloat> %1549, <8 x bfloat> %1550, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1651 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1649, <16 x bfloat> %131, i16 0, <8 x float> %1641, i1 false, i1 false)
  %1652 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1650, <16 x bfloat> %275, i16 0, <8 x float> %1642, i1 false, i1 false)
  %1653 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1649, <16 x bfloat> %171, i16 0, <8 x float> %1643, i1 false, i1 false)
  %1654 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1650, <16 x bfloat> %295, i16 0, <8 x float> %1644, i1 false, i1 false)
  %1655 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1649, <16 x bfloat> %211, i16 0, <8 x float> %1645, i1 false, i1 false)
  %1656 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1650, <16 x bfloat> %315, i16 0, <8 x float> %1646, i1 false, i1 false)
  %1657 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1649, <16 x bfloat> %251, i16 0, <8 x float> %1647, i1 false, i1 false)
  %1658 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1650, <16 x bfloat> %335, i16 0, <8 x float> %1648, i1 false, i1 false)
  %1659 = shufflevector <8 x bfloat> %1551, <8 x bfloat> %1552, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1660 = shufflevector <8 x bfloat> %1553, <8 x bfloat> %1554, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1661 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1659, <16 x bfloat> %140, i16 0, <8 x float> %1651, i1 false, i1 false)
  %1662 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1660, <16 x bfloat> %280, i16 0, <8 x float> %1652, i1 false, i1 false)
  %1663 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1659, <16 x bfloat> %180, i16 0, <8 x float> %1653, i1 false, i1 false)
  %1664 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1660, <16 x bfloat> %300, i16 0, <8 x float> %1654, i1 false, i1 false)
  %1665 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1659, <16 x bfloat> %220, i16 0, <8 x float> %1655, i1 false, i1 false)
  %1666 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1660, <16 x bfloat> %320, i16 0, <8 x float> %1656, i1 false, i1 false)
  %1667 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1659, <16 x bfloat> %260, i16 0, <8 x float> %1657, i1 false, i1 false)
  %1668 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1660, <16 x bfloat> %340, i16 0, <8 x float> %1658, i1 false, i1 false)
  %1669 = shufflevector <8 x bfloat> %1555, <8 x bfloat> %1556, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1670 = shufflevector <8 x bfloat> %1557, <8 x bfloat> %1558, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1671 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1669, <16 x bfloat> %113, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1672 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1670, <16 x bfloat> %265, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1673 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1669, <16 x bfloat> %153, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1674 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1670, <16 x bfloat> %285, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1675 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1669, <16 x bfloat> %193, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1676 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1670, <16 x bfloat> %305, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1677 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1669, <16 x bfloat> %233, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1678 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1670, <16 x bfloat> %325, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1679 = shufflevector <8 x bfloat> %1559, <8 x bfloat> %1560, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1680 = shufflevector <8 x bfloat> %1561, <8 x bfloat> %1562, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1681 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1679, <16 x bfloat> %122, i16 0, <8 x float> %1671, i1 false, i1 false)
  %1682 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1680, <16 x bfloat> %270, i16 0, <8 x float> %1672, i1 false, i1 false)
  %1683 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1679, <16 x bfloat> %162, i16 0, <8 x float> %1673, i1 false, i1 false)
  %1684 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1680, <16 x bfloat> %290, i16 0, <8 x float> %1674, i1 false, i1 false)
  %1685 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1679, <16 x bfloat> %202, i16 0, <8 x float> %1675, i1 false, i1 false)
  %1686 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1680, <16 x bfloat> %310, i16 0, <8 x float> %1676, i1 false, i1 false)
  %1687 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1679, <16 x bfloat> %242, i16 0, <8 x float> %1677, i1 false, i1 false)
  %1688 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1680, <16 x bfloat> %330, i16 0, <8 x float> %1678, i1 false, i1 false)
  %1689 = shufflevector <8 x bfloat> %1563, <8 x bfloat> %1564, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1690 = shufflevector <8 x bfloat> %1565, <8 x bfloat> %1566, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1691 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1689, <16 x bfloat> %131, i16 0, <8 x float> %1681, i1 false, i1 false)
  %1692 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1690, <16 x bfloat> %275, i16 0, <8 x float> %1682, i1 false, i1 false)
  %1693 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1689, <16 x bfloat> %171, i16 0, <8 x float> %1683, i1 false, i1 false)
  %1694 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1690, <16 x bfloat> %295, i16 0, <8 x float> %1684, i1 false, i1 false)
  %1695 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1689, <16 x bfloat> %211, i16 0, <8 x float> %1685, i1 false, i1 false)
  %1696 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1690, <16 x bfloat> %315, i16 0, <8 x float> %1686, i1 false, i1 false)
  %1697 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1689, <16 x bfloat> %251, i16 0, <8 x float> %1687, i1 false, i1 false)
  %1698 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1690, <16 x bfloat> %335, i16 0, <8 x float> %1688, i1 false, i1 false)
  %1699 = shufflevector <8 x bfloat> %1567, <8 x bfloat> %1568, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1700 = shufflevector <8 x bfloat> %1569, <8 x bfloat> %1570, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1701 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1699, <16 x bfloat> %140, i16 0, <8 x float> %1691, i1 false, i1 false)
  %1702 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1700, <16 x bfloat> %280, i16 0, <8 x float> %1692, i1 false, i1 false)
  %1703 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1699, <16 x bfloat> %180, i16 0, <8 x float> %1693, i1 false, i1 false)
  %1704 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1700, <16 x bfloat> %300, i16 0, <8 x float> %1694, i1 false, i1 false)
  %1705 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1699, <16 x bfloat> %220, i16 0, <8 x float> %1695, i1 false, i1 false)
  %1706 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1700, <16 x bfloat> %320, i16 0, <8 x float> %1696, i1 false, i1 false)
  %1707 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1699, <16 x bfloat> %260, i16 0, <8 x float> %1697, i1 false, i1 false)
  %1708 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1700, <16 x bfloat> %340, i16 0, <8 x float> %1698, i1 false, i1 false)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1709 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1571
  %1710 = add i32 %1709, %389
  %1711 = inttoptr i32 %1710 to ptr addrspace(3)
  %1712 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1711)
  %1713 = add i32 %1710, 4352
  %1714 = inttoptr i32 %1713 to ptr addrspace(3)
  %1715 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1714)
  %1716 = shufflevector <8 x bfloat> %1712, <8 x bfloat> %1715, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1717 = add i32 %1710, 32
  %1718 = inttoptr i32 %1717 to ptr addrspace(3)
  %1719 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1718)
  %1720 = add i32 %1710, 4384
  %1721 = inttoptr i32 %1720 to ptr addrspace(3)
  %1722 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1721)
  %1723 = shufflevector <8 x bfloat> %1719, <8 x bfloat> %1722, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1724 = add i32 %1710, 64
  %1725 = inttoptr i32 %1724 to ptr addrspace(3)
  %1726 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1725)
  %1727 = add i32 %1710, 4416
  %1728 = inttoptr i32 %1727 to ptr addrspace(3)
  %1729 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1728)
  %1730 = shufflevector <8 x bfloat> %1726, <8 x bfloat> %1729, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1731 = add i32 %1710, 96
  %1732 = inttoptr i32 %1731 to ptr addrspace(3)
  %1733 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1732)
  %1734 = add i32 %1710, 4448
  %1735 = inttoptr i32 %1734 to ptr addrspace(3)
  %1736 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1735)
  %1737 = shufflevector <8 x bfloat> %1733, <8 x bfloat> %1736, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1738 = add i32 %1710, 128
  %1739 = inttoptr i32 %1738 to ptr addrspace(3)
  %1740 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1739)
  %1741 = add i32 %1710, 4480
  %1742 = inttoptr i32 %1741 to ptr addrspace(3)
  %1743 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1742)
  %1744 = shufflevector <8 x bfloat> %1740, <8 x bfloat> %1743, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1745 = add i32 %1710, 160
  %1746 = inttoptr i32 %1745 to ptr addrspace(3)
  %1747 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1746)
  %1748 = add i32 %1710, 4512
  %1749 = inttoptr i32 %1748 to ptr addrspace(3)
  %1750 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1749)
  %1751 = shufflevector <8 x bfloat> %1747, <8 x bfloat> %1750, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1752 = add i32 %1710, 192
  %1753 = inttoptr i32 %1752 to ptr addrspace(3)
  %1754 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1753)
  %1755 = add i32 %1710, 4544
  %1756 = inttoptr i32 %1755 to ptr addrspace(3)
  %1757 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1756)
  %1758 = shufflevector <8 x bfloat> %1754, <8 x bfloat> %1757, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1759 = add i32 %1710, 224
  %1760 = inttoptr i32 %1759 to ptr addrspace(3)
  %1761 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1760)
  %1762 = add i32 %1710, 4576
  %1763 = inttoptr i32 %1762 to ptr addrspace(3)
  %1764 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1763)
  %1765 = shufflevector <8 x bfloat> %1761, <8 x bfloat> %1764, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 2)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1766 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1583
  %1767 = add i32 %1766, %392
  %1768 = inttoptr i32 %1767 to ptr addrspace(3)
  %1769 = load <8 x bfloat>, ptr addrspace(3) %1768, align 16
  %1770 = add i32 %1767, 32
  %1771 = inttoptr i32 %1770 to ptr addrspace(3)
  %1772 = load <8 x bfloat>, ptr addrspace(3) %1771, align 16
  %1773 = add i32 %1767, 8704
  %1774 = inttoptr i32 %1773 to ptr addrspace(3)
  %1775 = load <8 x bfloat>, ptr addrspace(3) %1774, align 16
  %1776 = add i32 %1767, 8736
  %1777 = inttoptr i32 %1776 to ptr addrspace(3)
  %1778 = load <8 x bfloat>, ptr addrspace(3) %1777, align 16
  %1779 = add i32 %1767, 64
  %1780 = inttoptr i32 %1779 to ptr addrspace(3)
  %1781 = load <8 x bfloat>, ptr addrspace(3) %1780, align 16
  %1782 = add i32 %1767, 96
  %1783 = inttoptr i32 %1782 to ptr addrspace(3)
  %1784 = load <8 x bfloat>, ptr addrspace(3) %1783, align 16
  %1785 = add i32 %1767, 8768
  %1786 = inttoptr i32 %1785 to ptr addrspace(3)
  %1787 = load <8 x bfloat>, ptr addrspace(3) %1786, align 16
  %1788 = add i32 %1767, 8800
  %1789 = inttoptr i32 %1788 to ptr addrspace(3)
  %1790 = load <8 x bfloat>, ptr addrspace(3) %1789, align 16
  %1791 = add i32 %1767, 128
  %1792 = inttoptr i32 %1791 to ptr addrspace(3)
  %1793 = load <8 x bfloat>, ptr addrspace(3) %1792, align 16
  %1794 = add i32 %1767, 160
  %1795 = inttoptr i32 %1794 to ptr addrspace(3)
  %1796 = load <8 x bfloat>, ptr addrspace(3) %1795, align 16
  %1797 = add i32 %1767, 8832
  %1798 = inttoptr i32 %1797 to ptr addrspace(3)
  %1799 = load <8 x bfloat>, ptr addrspace(3) %1798, align 16
  %1800 = add i32 %1767, 8864
  %1801 = inttoptr i32 %1800 to ptr addrspace(3)
  %1802 = load <8 x bfloat>, ptr addrspace(3) %1801, align 16
  %1803 = add i32 %1767, 192
  %1804 = inttoptr i32 %1803 to ptr addrspace(3)
  %1805 = load <8 x bfloat>, ptr addrspace(3) %1804, align 16
  %1806 = add i32 %1767, 224
  %1807 = inttoptr i32 %1806 to ptr addrspace(3)
  %1808 = load <8 x bfloat>, ptr addrspace(3) %1807, align 16
  %1809 = add i32 %1767, 8896
  %1810 = inttoptr i32 %1809 to ptr addrspace(3)
  %1811 = load <8 x bfloat>, ptr addrspace(3) %1810, align 16
  %1812 = add i32 %1767, 8928
  %1813 = inttoptr i32 %1812 to ptr addrspace(3)
  %1814 = load <8 x bfloat>, ptr addrspace(3) %1813, align 16
  %1815 = add i32 %1767, 4352
  %1816 = inttoptr i32 %1815 to ptr addrspace(3)
  %1817 = load <8 x bfloat>, ptr addrspace(3) %1816, align 16
  %1818 = add i32 %1767, 4384
  %1819 = inttoptr i32 %1818 to ptr addrspace(3)
  %1820 = load <8 x bfloat>, ptr addrspace(3) %1819, align 16
  %1821 = add i32 %1767, 13056
  %1822 = inttoptr i32 %1821 to ptr addrspace(3)
  %1823 = load <8 x bfloat>, ptr addrspace(3) %1822, align 16
  %1824 = add i32 %1767, 13088
  %1825 = inttoptr i32 %1824 to ptr addrspace(3)
  %1826 = load <8 x bfloat>, ptr addrspace(3) %1825, align 16
  %1827 = add i32 %1767, 4416
  %1828 = inttoptr i32 %1827 to ptr addrspace(3)
  %1829 = load <8 x bfloat>, ptr addrspace(3) %1828, align 16
  %1830 = add i32 %1767, 4448
  %1831 = inttoptr i32 %1830 to ptr addrspace(3)
  %1832 = load <8 x bfloat>, ptr addrspace(3) %1831, align 16
  %1833 = add i32 %1767, 13120
  %1834 = inttoptr i32 %1833 to ptr addrspace(3)
  %1835 = load <8 x bfloat>, ptr addrspace(3) %1834, align 16
  %1836 = add i32 %1767, 13152
  %1837 = inttoptr i32 %1836 to ptr addrspace(3)
  %1838 = load <8 x bfloat>, ptr addrspace(3) %1837, align 16
  %1839 = add i32 %1767, 4480
  %1840 = inttoptr i32 %1839 to ptr addrspace(3)
  %1841 = load <8 x bfloat>, ptr addrspace(3) %1840, align 16
  %1842 = add i32 %1767, 4512
  %1843 = inttoptr i32 %1842 to ptr addrspace(3)
  %1844 = load <8 x bfloat>, ptr addrspace(3) %1843, align 16
  %1845 = add i32 %1767, 13184
  %1846 = inttoptr i32 %1845 to ptr addrspace(3)
  %1847 = load <8 x bfloat>, ptr addrspace(3) %1846, align 16
  %1848 = add i32 %1767, 13216
  %1849 = inttoptr i32 %1848 to ptr addrspace(3)
  %1850 = load <8 x bfloat>, ptr addrspace(3) %1849, align 16
  %1851 = add i32 %1767, 4544
  %1852 = inttoptr i32 %1851 to ptr addrspace(3)
  %1853 = load <8 x bfloat>, ptr addrspace(3) %1852, align 16
  %1854 = add i32 %1767, 4576
  %1855 = inttoptr i32 %1854 to ptr addrspace(3)
  %1856 = load <8 x bfloat>, ptr addrspace(3) %1855, align 16
  %1857 = add i32 %1767, 13248
  %1858 = inttoptr i32 %1857 to ptr addrspace(3)
  %1859 = load <8 x bfloat>, ptr addrspace(3) %1858, align 16
  %1860 = add i32 %1767, 13280
  %1861 = inttoptr i32 %1860 to ptr addrspace(3)
  %1862 = load <8 x bfloat>, ptr addrspace(3) %1861, align 16
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %1863 = extractelement <8 x float> %1661, i64 0
  %1864 = call float @llvm.fma.f32(float %1863, float %365, float %366)
  %1865 = extractelement <8 x float> %1661, i64 1
  %1866 = call float @llvm.fma.f32(float %1865, float %365, float %366)
  %1867 = extractelement <8 x float> %1661, i64 2
  %1868 = call float @llvm.fma.f32(float %1867, float %365, float %366)
  %1869 = extractelement <8 x float> %1661, i64 3
  %1870 = call float @llvm.fma.f32(float %1869, float %365, float %366)
  %1871 = extractelement <8 x float> %1661, i64 4
  %1872 = call float @llvm.fma.f32(float %1871, float %365, float %366)
  %1873 = extractelement <8 x float> %1661, i64 5
  %1874 = call float @llvm.fma.f32(float %1873, float %365, float %366)
  %1875 = extractelement <8 x float> %1661, i64 6
  %1876 = call float @llvm.fma.f32(float %1875, float %365, float %366)
  %1877 = extractelement <8 x float> %1661, i64 7
  %1878 = call float @llvm.fma.f32(float %1877, float %365, float %366)
  %1879 = add i32 %1584, %375
  %1880 = add i32 %102, %23
  %1881 = icmp sgt i32 %1879, %1880
  %1882 = and i1 %1881, %407
  %1883 = select i1 %1882, float -3.000000e+38, float %1864
  %1884 = add i32 %1879, 1
  %1885 = icmp sgt i32 %1884, %1880
  %1886 = and i1 %1885, %407
  %1887 = select i1 %1886, float -3.000000e+38, float %1866
  %1888 = add i32 %1879, 2
  %1889 = icmp sgt i32 %1888, %1880
  %1890 = and i1 %1889, %407
  %1891 = select i1 %1890, float -3.000000e+38, float %1868
  %1892 = add i32 %1879, 3
  %1893 = icmp sgt i32 %1892, %1880
  %1894 = and i1 %1893, %407
  %1895 = select i1 %1894, float -3.000000e+38, float %1870
  %1896 = add i32 %1879, 4
  %1897 = icmp sgt i32 %1896, %1880
  %1898 = and i1 %1897, %407
  %1899 = select i1 %1898, float -3.000000e+38, float %1872
  %1900 = add i32 %1879, 5
  %1901 = icmp sgt i32 %1900, %1880
  %1902 = and i1 %1901, %407
  %1903 = select i1 %1902, float -3.000000e+38, float %1874
  %1904 = add i32 %1879, 6
  %1905 = icmp sgt i32 %1904, %1880
  %1906 = and i1 %1905, %407
  %1907 = select i1 %1906, float -3.000000e+38, float %1876
  %1908 = add i32 %1879, 7
  %1909 = icmp sgt i32 %1908, %1880
  %1910 = and i1 %1909, %407
  %1911 = select i1 %1910, float -3.000000e+38, float %1878
  %1912 = call float @llvm.amdgcn.exp2.f32(float %1883)
  %1913 = call float @llvm.amdgcn.exp2.f32(float %1887)
  %1914 = call float @llvm.amdgcn.exp2.f32(float %1891)
  %1915 = call float @llvm.amdgcn.exp2.f32(float %1895)
  %1916 = call float @llvm.amdgcn.exp2.f32(float %1899)
  %1917 = call float @llvm.amdgcn.exp2.f32(float %1903)
  %1918 = call float @llvm.amdgcn.exp2.f32(float %1907)
  %1919 = call float @llvm.amdgcn.exp2.f32(float %1911)
  %1920 = extractelement <8 x float> %1662, i64 0
  %1921 = call float @llvm.fma.f32(float %1920, float %16, float %371)
  %1922 = fmul float %1912, %1921
  %1923 = fptrunc float %1922 to bfloat
  %1924 = extractelement <8 x float> %1662, i64 1
  %1925 = call float @llvm.fma.f32(float %1924, float %16, float %371)
  %1926 = fmul float %1913, %1925
  %1927 = fptrunc float %1926 to bfloat
  %1928 = extractelement <8 x float> %1662, i64 2
  %1929 = call float @llvm.fma.f32(float %1928, float %16, float %371)
  %1930 = fmul float %1914, %1929
  %1931 = fptrunc float %1930 to bfloat
  %1932 = extractelement <8 x float> %1662, i64 3
  %1933 = call float @llvm.fma.f32(float %1932, float %16, float %371)
  %1934 = fmul float %1915, %1933
  %1935 = fptrunc float %1934 to bfloat
  %1936 = extractelement <8 x float> %1662, i64 4
  %1937 = call float @llvm.fma.f32(float %1936, float %16, float %371)
  %1938 = fmul float %1916, %1937
  %1939 = fptrunc float %1938 to bfloat
  %1940 = extractelement <8 x float> %1662, i64 5
  %1941 = call float @llvm.fma.f32(float %1940, float %16, float %371)
  %1942 = fmul float %1917, %1941
  %1943 = fptrunc float %1942 to bfloat
  %1944 = extractelement <8 x float> %1662, i64 6
  %1945 = call float @llvm.fma.f32(float %1944, float %16, float %371)
  %1946 = fmul float %1918, %1945
  %1947 = fptrunc float %1946 to bfloat
  %1948 = extractelement <8 x float> %1662, i64 7
  %1949 = call float @llvm.fma.f32(float %1948, float %16, float %371)
  %1950 = fmul float %1919, %1949
  %1951 = fptrunc float %1950 to bfloat
  %1952 = extractelement <8 x float> %1663, i64 0
  %1953 = call float @llvm.fma.f32(float %1952, float %365, float %367)
  %1954 = extractelement <8 x float> %1663, i64 1
  %1955 = call float @llvm.fma.f32(float %1954, float %365, float %367)
  %1956 = extractelement <8 x float> %1663, i64 2
  %1957 = call float @llvm.fma.f32(float %1956, float %365, float %367)
  %1958 = extractelement <8 x float> %1663, i64 3
  %1959 = call float @llvm.fma.f32(float %1958, float %365, float %367)
  %1960 = extractelement <8 x float> %1663, i64 4
  %1961 = call float @llvm.fma.f32(float %1960, float %365, float %367)
  %1962 = extractelement <8 x float> %1663, i64 5
  %1963 = call float @llvm.fma.f32(float %1962, float %365, float %367)
  %1964 = extractelement <8 x float> %1663, i64 6
  %1965 = call float @llvm.fma.f32(float %1964, float %365, float %367)
  %1966 = extractelement <8 x float> %1663, i64 7
  %1967 = call float @llvm.fma.f32(float %1966, float %365, float %367)
  %1968 = add i32 %142, %23
  %1969 = icmp sgt i32 %1879, %1968
  %1970 = and i1 %1969, %407
  %1971 = select i1 %1970, float -3.000000e+38, float %1953
  %1972 = icmp sgt i32 %1884, %1968
  %1973 = and i1 %1972, %407
  %1974 = select i1 %1973, float -3.000000e+38, float %1955
  %1975 = icmp sgt i32 %1888, %1968
  %1976 = and i1 %1975, %407
  %1977 = select i1 %1976, float -3.000000e+38, float %1957
  %1978 = icmp sgt i32 %1892, %1968
  %1979 = and i1 %1978, %407
  %1980 = select i1 %1979, float -3.000000e+38, float %1959
  %1981 = icmp sgt i32 %1896, %1968
  %1982 = and i1 %1981, %407
  %1983 = select i1 %1982, float -3.000000e+38, float %1961
  %1984 = icmp sgt i32 %1900, %1968
  %1985 = and i1 %1984, %407
  %1986 = select i1 %1985, float -3.000000e+38, float %1963
  %1987 = icmp sgt i32 %1904, %1968
  %1988 = and i1 %1987, %407
  %1989 = select i1 %1988, float -3.000000e+38, float %1965
  %1990 = icmp sgt i32 %1908, %1968
  %1991 = and i1 %1990, %407
  %1992 = select i1 %1991, float -3.000000e+38, float %1967
  %1993 = call float @llvm.amdgcn.exp2.f32(float %1971)
  %1994 = call float @llvm.amdgcn.exp2.f32(float %1974)
  %1995 = call float @llvm.amdgcn.exp2.f32(float %1977)
  %1996 = call float @llvm.amdgcn.exp2.f32(float %1980)
  %1997 = call float @llvm.amdgcn.exp2.f32(float %1983)
  %1998 = call float @llvm.amdgcn.exp2.f32(float %1986)
  %1999 = call float @llvm.amdgcn.exp2.f32(float %1989)
  %2000 = call float @llvm.amdgcn.exp2.f32(float %1992)
  %2001 = extractelement <8 x float> %1664, i64 0
  %2002 = call float @llvm.fma.f32(float %2001, float %16, float %372)
  %2003 = fmul float %1993, %2002
  %2004 = fptrunc float %2003 to bfloat
  %2005 = extractelement <8 x float> %1664, i64 1
  %2006 = call float @llvm.fma.f32(float %2005, float %16, float %372)
  %2007 = fmul float %1994, %2006
  %2008 = fptrunc float %2007 to bfloat
  %2009 = extractelement <8 x float> %1664, i64 2
  %2010 = call float @llvm.fma.f32(float %2009, float %16, float %372)
  %2011 = fmul float %1995, %2010
  %2012 = fptrunc float %2011 to bfloat
  %2013 = extractelement <8 x float> %1664, i64 3
  %2014 = call float @llvm.fma.f32(float %2013, float %16, float %372)
  %2015 = fmul float %1996, %2014
  %2016 = fptrunc float %2015 to bfloat
  %2017 = extractelement <8 x float> %1664, i64 4
  %2018 = call float @llvm.fma.f32(float %2017, float %16, float %372)
  %2019 = fmul float %1997, %2018
  %2020 = fptrunc float %2019 to bfloat
  %2021 = extractelement <8 x float> %1664, i64 5
  %2022 = call float @llvm.fma.f32(float %2021, float %16, float %372)
  %2023 = fmul float %1998, %2022
  %2024 = fptrunc float %2023 to bfloat
  %2025 = extractelement <8 x float> %1664, i64 6
  %2026 = call float @llvm.fma.f32(float %2025, float %16, float %372)
  %2027 = fmul float %1999, %2026
  %2028 = fptrunc float %2027 to bfloat
  %2029 = extractelement <8 x float> %1664, i64 7
  %2030 = call float @llvm.fma.f32(float %2029, float %16, float %372)
  %2031 = fmul float %2000, %2030
  %2032 = fptrunc float %2031 to bfloat
  %2033 = extractelement <8 x float> %1665, i64 0
  %2034 = call float @llvm.fma.f32(float %2033, float %365, float %368)
  %2035 = extractelement <8 x float> %1665, i64 1
  %2036 = call float @llvm.fma.f32(float %2035, float %365, float %368)
  %2037 = extractelement <8 x float> %1665, i64 2
  %2038 = call float @llvm.fma.f32(float %2037, float %365, float %368)
  %2039 = extractelement <8 x float> %1665, i64 3
  %2040 = call float @llvm.fma.f32(float %2039, float %365, float %368)
  %2041 = extractelement <8 x float> %1665, i64 4
  %2042 = call float @llvm.fma.f32(float %2041, float %365, float %368)
  %2043 = extractelement <8 x float> %1665, i64 5
  %2044 = call float @llvm.fma.f32(float %2043, float %365, float %368)
  %2045 = extractelement <8 x float> %1665, i64 6
  %2046 = call float @llvm.fma.f32(float %2045, float %365, float %368)
  %2047 = extractelement <8 x float> %1665, i64 7
  %2048 = call float @llvm.fma.f32(float %2047, float %365, float %368)
  %2049 = add i32 %182, %23
  %2050 = icmp sgt i32 %1879, %2049
  %2051 = and i1 %2050, %407
  %2052 = select i1 %2051, float -3.000000e+38, float %2034
  %2053 = icmp sgt i32 %1884, %2049
  %2054 = and i1 %2053, %407
  %2055 = select i1 %2054, float -3.000000e+38, float %2036
  %2056 = icmp sgt i32 %1888, %2049
  %2057 = and i1 %2056, %407
  %2058 = select i1 %2057, float -3.000000e+38, float %2038
  %2059 = icmp sgt i32 %1892, %2049
  %2060 = and i1 %2059, %407
  %2061 = select i1 %2060, float -3.000000e+38, float %2040
  %2062 = icmp sgt i32 %1896, %2049
  %2063 = and i1 %2062, %407
  %2064 = select i1 %2063, float -3.000000e+38, float %2042
  %2065 = icmp sgt i32 %1900, %2049
  %2066 = and i1 %2065, %407
  %2067 = select i1 %2066, float -3.000000e+38, float %2044
  %2068 = icmp sgt i32 %1904, %2049
  %2069 = and i1 %2068, %407
  %2070 = select i1 %2069, float -3.000000e+38, float %2046
  %2071 = icmp sgt i32 %1908, %2049
  %2072 = and i1 %2071, %407
  %2073 = select i1 %2072, float -3.000000e+38, float %2048
  %2074 = call float @llvm.amdgcn.exp2.f32(float %2052)
  %2075 = call float @llvm.amdgcn.exp2.f32(float %2055)
  %2076 = call float @llvm.amdgcn.exp2.f32(float %2058)
  %2077 = call float @llvm.amdgcn.exp2.f32(float %2061)
  %2078 = call float @llvm.amdgcn.exp2.f32(float %2064)
  %2079 = call float @llvm.amdgcn.exp2.f32(float %2067)
  %2080 = call float @llvm.amdgcn.exp2.f32(float %2070)
  %2081 = call float @llvm.amdgcn.exp2.f32(float %2073)
  %2082 = extractelement <8 x float> %1666, i64 0
  %2083 = call float @llvm.fma.f32(float %2082, float %16, float %373)
  %2084 = fmul float %2074, %2083
  %2085 = fptrunc float %2084 to bfloat
  %2086 = extractelement <8 x float> %1666, i64 1
  %2087 = call float @llvm.fma.f32(float %2086, float %16, float %373)
  %2088 = fmul float %2075, %2087
  %2089 = fptrunc float %2088 to bfloat
  %2090 = extractelement <8 x float> %1666, i64 2
  %2091 = call float @llvm.fma.f32(float %2090, float %16, float %373)
  %2092 = fmul float %2076, %2091
  %2093 = fptrunc float %2092 to bfloat
  %2094 = extractelement <8 x float> %1666, i64 3
  %2095 = call float @llvm.fma.f32(float %2094, float %16, float %373)
  %2096 = fmul float %2077, %2095
  %2097 = fptrunc float %2096 to bfloat
  %2098 = extractelement <8 x float> %1666, i64 4
  %2099 = call float @llvm.fma.f32(float %2098, float %16, float %373)
  %2100 = fmul float %2078, %2099
  %2101 = fptrunc float %2100 to bfloat
  %2102 = extractelement <8 x float> %1666, i64 5
  %2103 = call float @llvm.fma.f32(float %2102, float %16, float %373)
  %2104 = fmul float %2079, %2103
  %2105 = fptrunc float %2104 to bfloat
  %2106 = extractelement <8 x float> %1666, i64 6
  %2107 = call float @llvm.fma.f32(float %2106, float %16, float %373)
  %2108 = fmul float %2080, %2107
  %2109 = fptrunc float %2108 to bfloat
  %2110 = extractelement <8 x float> %1666, i64 7
  %2111 = call float @llvm.fma.f32(float %2110, float %16, float %373)
  %2112 = fmul float %2081, %2111
  %2113 = fptrunc float %2112 to bfloat
  %2114 = extractelement <8 x float> %1667, i64 0
  %2115 = call float @llvm.fma.f32(float %2114, float %365, float %369)
  %2116 = extractelement <8 x float> %1667, i64 1
  %2117 = call float @llvm.fma.f32(float %2116, float %365, float %369)
  %2118 = extractelement <8 x float> %1667, i64 2
  %2119 = call float @llvm.fma.f32(float %2118, float %365, float %369)
  %2120 = extractelement <8 x float> %1667, i64 3
  %2121 = call float @llvm.fma.f32(float %2120, float %365, float %369)
  %2122 = extractelement <8 x float> %1667, i64 4
  %2123 = call float @llvm.fma.f32(float %2122, float %365, float %369)
  %2124 = extractelement <8 x float> %1667, i64 5
  %2125 = call float @llvm.fma.f32(float %2124, float %365, float %369)
  %2126 = extractelement <8 x float> %1667, i64 6
  %2127 = call float @llvm.fma.f32(float %2126, float %365, float %369)
  %2128 = extractelement <8 x float> %1667, i64 7
  %2129 = call float @llvm.fma.f32(float %2128, float %365, float %369)
  %2130 = add i32 %222, %23
  %2131 = icmp sgt i32 %1879, %2130
  %2132 = and i1 %2131, %407
  %2133 = select i1 %2132, float -3.000000e+38, float %2115
  %2134 = icmp sgt i32 %1884, %2130
  %2135 = and i1 %2134, %407
  %2136 = select i1 %2135, float -3.000000e+38, float %2117
  %2137 = icmp sgt i32 %1888, %2130
  %2138 = and i1 %2137, %407
  %2139 = select i1 %2138, float -3.000000e+38, float %2119
  %2140 = icmp sgt i32 %1892, %2130
  %2141 = and i1 %2140, %407
  %2142 = select i1 %2141, float -3.000000e+38, float %2121
  %2143 = icmp sgt i32 %1896, %2130
  %2144 = and i1 %2143, %407
  %2145 = select i1 %2144, float -3.000000e+38, float %2123
  %2146 = icmp sgt i32 %1900, %2130
  %2147 = and i1 %2146, %407
  %2148 = select i1 %2147, float -3.000000e+38, float %2125
  %2149 = icmp sgt i32 %1904, %2130
  %2150 = and i1 %2149, %407
  %2151 = select i1 %2150, float -3.000000e+38, float %2127
  %2152 = icmp sgt i32 %1908, %2130
  %2153 = and i1 %2152, %407
  %2154 = select i1 %2153, float -3.000000e+38, float %2129
  %2155 = call float @llvm.amdgcn.exp2.f32(float %2133)
  %2156 = call float @llvm.amdgcn.exp2.f32(float %2136)
  %2157 = call float @llvm.amdgcn.exp2.f32(float %2139)
  %2158 = call float @llvm.amdgcn.exp2.f32(float %2142)
  %2159 = call float @llvm.amdgcn.exp2.f32(float %2145)
  %2160 = call float @llvm.amdgcn.exp2.f32(float %2148)
  %2161 = call float @llvm.amdgcn.exp2.f32(float %2151)
  %2162 = call float @llvm.amdgcn.exp2.f32(float %2154)
  %2163 = extractelement <8 x float> %1668, i64 0
  %2164 = call float @llvm.fma.f32(float %2163, float %16, float %374)
  %2165 = fmul float %2155, %2164
  %2166 = fptrunc float %2165 to bfloat
  %2167 = extractelement <8 x float> %1668, i64 1
  %2168 = call float @llvm.fma.f32(float %2167, float %16, float %374)
  %2169 = fmul float %2156, %2168
  %2170 = fptrunc float %2169 to bfloat
  %2171 = extractelement <8 x float> %1668, i64 2
  %2172 = call float @llvm.fma.f32(float %2171, float %16, float %374)
  %2173 = fmul float %2157, %2172
  %2174 = fptrunc float %2173 to bfloat
  %2175 = extractelement <8 x float> %1668, i64 3
  %2176 = call float @llvm.fma.f32(float %2175, float %16, float %374)
  %2177 = fmul float %2158, %2176
  %2178 = fptrunc float %2177 to bfloat
  %2179 = extractelement <8 x float> %1668, i64 4
  %2180 = call float @llvm.fma.f32(float %2179, float %16, float %374)
  %2181 = fmul float %2159, %2180
  %2182 = fptrunc float %2181 to bfloat
  %2183 = extractelement <8 x float> %1668, i64 5
  %2184 = call float @llvm.fma.f32(float %2183, float %16, float %374)
  %2185 = fmul float %2160, %2184
  %2186 = fptrunc float %2185 to bfloat
  %2187 = extractelement <8 x float> %1668, i64 6
  %2188 = call float @llvm.fma.f32(float %2187, float %16, float %374)
  %2189 = fmul float %2161, %2188
  %2190 = fptrunc float %2189 to bfloat
  %2191 = extractelement <8 x float> %1668, i64 7
  %2192 = call float @llvm.fma.f32(float %2191, float %16, float %374)
  %2193 = fmul float %2162, %2192
  %2194 = fptrunc float %2193 to bfloat
  %2195 = extractelement <8 x float> %1701, i64 0
  %2196 = call float @llvm.fma.f32(float %2195, float %365, float %366)
  %2197 = extractelement <8 x float> %1701, i64 1
  %2198 = call float @llvm.fma.f32(float %2197, float %365, float %366)
  %2199 = extractelement <8 x float> %1701, i64 2
  %2200 = call float @llvm.fma.f32(float %2199, float %365, float %366)
  %2201 = extractelement <8 x float> %1701, i64 3
  %2202 = call float @llvm.fma.f32(float %2201, float %365, float %366)
  %2203 = extractelement <8 x float> %1701, i64 4
  %2204 = call float @llvm.fma.f32(float %2203, float %365, float %366)
  %2205 = extractelement <8 x float> %1701, i64 5
  %2206 = call float @llvm.fma.f32(float %2205, float %365, float %366)
  %2207 = extractelement <8 x float> %1701, i64 6
  %2208 = call float @llvm.fma.f32(float %2207, float %365, float %366)
  %2209 = extractelement <8 x float> %1701, i64 7
  %2210 = call float @llvm.fma.f32(float %2209, float %365, float %366)
  %2211 = add i32 %1584, 16
  %2212 = add i32 %2211, %375
  %2213 = icmp sgt i32 %2212, %1880
  %2214 = and i1 %2213, %407
  %2215 = select i1 %2214, float -3.000000e+38, float %2196
  %2216 = add i32 %2212, 1
  %2217 = icmp sgt i32 %2216, %1880
  %2218 = and i1 %2217, %407
  %2219 = select i1 %2218, float -3.000000e+38, float %2198
  %2220 = add i32 %2212, 2
  %2221 = icmp sgt i32 %2220, %1880
  %2222 = and i1 %2221, %407
  %2223 = select i1 %2222, float -3.000000e+38, float %2200
  %2224 = add i32 %2212, 3
  %2225 = icmp sgt i32 %2224, %1880
  %2226 = and i1 %2225, %407
  %2227 = select i1 %2226, float -3.000000e+38, float %2202
  %2228 = add i32 %2212, 4
  %2229 = icmp sgt i32 %2228, %1880
  %2230 = and i1 %2229, %407
  %2231 = select i1 %2230, float -3.000000e+38, float %2204
  %2232 = add i32 %2212, 5
  %2233 = icmp sgt i32 %2232, %1880
  %2234 = and i1 %2233, %407
  %2235 = select i1 %2234, float -3.000000e+38, float %2206
  %2236 = add i32 %2212, 6
  %2237 = icmp sgt i32 %2236, %1880
  %2238 = and i1 %2237, %407
  %2239 = select i1 %2238, float -3.000000e+38, float %2208
  %2240 = add i32 %2212, 7
  %2241 = icmp sgt i32 %2240, %1880
  %2242 = and i1 %2241, %407
  %2243 = select i1 %2242, float -3.000000e+38, float %2210
  %2244 = call float @llvm.amdgcn.exp2.f32(float %2215)
  %2245 = call float @llvm.amdgcn.exp2.f32(float %2219)
  %2246 = call float @llvm.amdgcn.exp2.f32(float %2223)
  %2247 = call float @llvm.amdgcn.exp2.f32(float %2227)
  %2248 = call float @llvm.amdgcn.exp2.f32(float %2231)
  %2249 = call float @llvm.amdgcn.exp2.f32(float %2235)
  %2250 = call float @llvm.amdgcn.exp2.f32(float %2239)
  %2251 = call float @llvm.amdgcn.exp2.f32(float %2243)
  %2252 = extractelement <8 x float> %1702, i64 0
  %2253 = call float @llvm.fma.f32(float %2252, float %16, float %371)
  %2254 = fmul float %2244, %2253
  %2255 = fptrunc float %2254 to bfloat
  %2256 = extractelement <8 x float> %1702, i64 1
  %2257 = call float @llvm.fma.f32(float %2256, float %16, float %371)
  %2258 = fmul float %2245, %2257
  %2259 = fptrunc float %2258 to bfloat
  %2260 = extractelement <8 x float> %1702, i64 2
  %2261 = call float @llvm.fma.f32(float %2260, float %16, float %371)
  %2262 = fmul float %2246, %2261
  %2263 = fptrunc float %2262 to bfloat
  %2264 = extractelement <8 x float> %1702, i64 3
  %2265 = call float @llvm.fma.f32(float %2264, float %16, float %371)
  %2266 = fmul float %2247, %2265
  %2267 = fptrunc float %2266 to bfloat
  %2268 = extractelement <8 x float> %1702, i64 4
  %2269 = call float @llvm.fma.f32(float %2268, float %16, float %371)
  %2270 = fmul float %2248, %2269
  %2271 = fptrunc float %2270 to bfloat
  %2272 = extractelement <8 x float> %1702, i64 5
  %2273 = call float @llvm.fma.f32(float %2272, float %16, float %371)
  %2274 = fmul float %2249, %2273
  %2275 = fptrunc float %2274 to bfloat
  %2276 = extractelement <8 x float> %1702, i64 6
  %2277 = call float @llvm.fma.f32(float %2276, float %16, float %371)
  %2278 = fmul float %2250, %2277
  %2279 = fptrunc float %2278 to bfloat
  %2280 = extractelement <8 x float> %1702, i64 7
  %2281 = call float @llvm.fma.f32(float %2280, float %16, float %371)
  %2282 = fmul float %2251, %2281
  %2283 = fptrunc float %2282 to bfloat
  %2284 = extractelement <8 x float> %1703, i64 0
  %2285 = call float @llvm.fma.f32(float %2284, float %365, float %367)
  %2286 = extractelement <8 x float> %1703, i64 1
  %2287 = call float @llvm.fma.f32(float %2286, float %365, float %367)
  %2288 = extractelement <8 x float> %1703, i64 2
  %2289 = call float @llvm.fma.f32(float %2288, float %365, float %367)
  %2290 = extractelement <8 x float> %1703, i64 3
  %2291 = call float @llvm.fma.f32(float %2290, float %365, float %367)
  %2292 = extractelement <8 x float> %1703, i64 4
  %2293 = call float @llvm.fma.f32(float %2292, float %365, float %367)
  %2294 = extractelement <8 x float> %1703, i64 5
  %2295 = call float @llvm.fma.f32(float %2294, float %365, float %367)
  %2296 = extractelement <8 x float> %1703, i64 6
  %2297 = call float @llvm.fma.f32(float %2296, float %365, float %367)
  %2298 = extractelement <8 x float> %1703, i64 7
  %2299 = call float @llvm.fma.f32(float %2298, float %365, float %367)
  %2300 = icmp sgt i32 %2212, %1968
  %2301 = and i1 %2300, %407
  %2302 = select i1 %2301, float -3.000000e+38, float %2285
  %2303 = icmp sgt i32 %2216, %1968
  %2304 = and i1 %2303, %407
  %2305 = select i1 %2304, float -3.000000e+38, float %2287
  %2306 = icmp sgt i32 %2220, %1968
  %2307 = and i1 %2306, %407
  %2308 = select i1 %2307, float -3.000000e+38, float %2289
  %2309 = icmp sgt i32 %2224, %1968
  %2310 = and i1 %2309, %407
  %2311 = select i1 %2310, float -3.000000e+38, float %2291
  %2312 = icmp sgt i32 %2228, %1968
  %2313 = and i1 %2312, %407
  %2314 = select i1 %2313, float -3.000000e+38, float %2293
  %2315 = icmp sgt i32 %2232, %1968
  %2316 = and i1 %2315, %407
  %2317 = select i1 %2316, float -3.000000e+38, float %2295
  %2318 = icmp sgt i32 %2236, %1968
  %2319 = and i1 %2318, %407
  %2320 = select i1 %2319, float -3.000000e+38, float %2297
  %2321 = icmp sgt i32 %2240, %1968
  %2322 = and i1 %2321, %407
  %2323 = select i1 %2322, float -3.000000e+38, float %2299
  %2324 = call float @llvm.amdgcn.exp2.f32(float %2302)
  %2325 = call float @llvm.amdgcn.exp2.f32(float %2305)
  %2326 = call float @llvm.amdgcn.exp2.f32(float %2308)
  %2327 = call float @llvm.amdgcn.exp2.f32(float %2311)
  %2328 = call float @llvm.amdgcn.exp2.f32(float %2314)
  %2329 = call float @llvm.amdgcn.exp2.f32(float %2317)
  %2330 = call float @llvm.amdgcn.exp2.f32(float %2320)
  %2331 = call float @llvm.amdgcn.exp2.f32(float %2323)
  %2332 = extractelement <8 x float> %1704, i64 0
  %2333 = call float @llvm.fma.f32(float %2332, float %16, float %372)
  %2334 = fmul float %2324, %2333
  %2335 = fptrunc float %2334 to bfloat
  %2336 = extractelement <8 x float> %1704, i64 1
  %2337 = call float @llvm.fma.f32(float %2336, float %16, float %372)
  %2338 = fmul float %2325, %2337
  %2339 = fptrunc float %2338 to bfloat
  %2340 = extractelement <8 x float> %1704, i64 2
  %2341 = call float @llvm.fma.f32(float %2340, float %16, float %372)
  %2342 = fmul float %2326, %2341
  %2343 = fptrunc float %2342 to bfloat
  %2344 = extractelement <8 x float> %1704, i64 3
  %2345 = call float @llvm.fma.f32(float %2344, float %16, float %372)
  %2346 = fmul float %2327, %2345
  %2347 = fptrunc float %2346 to bfloat
  %2348 = extractelement <8 x float> %1704, i64 4
  %2349 = call float @llvm.fma.f32(float %2348, float %16, float %372)
  %2350 = fmul float %2328, %2349
  %2351 = fptrunc float %2350 to bfloat
  %2352 = extractelement <8 x float> %1704, i64 5
  %2353 = call float @llvm.fma.f32(float %2352, float %16, float %372)
  %2354 = fmul float %2329, %2353
  %2355 = fptrunc float %2354 to bfloat
  %2356 = extractelement <8 x float> %1704, i64 6
  %2357 = call float @llvm.fma.f32(float %2356, float %16, float %372)
  %2358 = fmul float %2330, %2357
  %2359 = fptrunc float %2358 to bfloat
  %2360 = extractelement <8 x float> %1704, i64 7
  %2361 = call float @llvm.fma.f32(float %2360, float %16, float %372)
  %2362 = fmul float %2331, %2361
  %2363 = fptrunc float %2362 to bfloat
  %2364 = extractelement <8 x float> %1705, i64 0
  %2365 = call float @llvm.fma.f32(float %2364, float %365, float %368)
  %2366 = extractelement <8 x float> %1705, i64 1
  %2367 = call float @llvm.fma.f32(float %2366, float %365, float %368)
  %2368 = extractelement <8 x float> %1705, i64 2
  %2369 = call float @llvm.fma.f32(float %2368, float %365, float %368)
  %2370 = extractelement <8 x float> %1705, i64 3
  %2371 = call float @llvm.fma.f32(float %2370, float %365, float %368)
  %2372 = extractelement <8 x float> %1705, i64 4
  %2373 = call float @llvm.fma.f32(float %2372, float %365, float %368)
  %2374 = extractelement <8 x float> %1705, i64 5
  %2375 = call float @llvm.fma.f32(float %2374, float %365, float %368)
  %2376 = extractelement <8 x float> %1705, i64 6
  %2377 = call float @llvm.fma.f32(float %2376, float %365, float %368)
  %2378 = extractelement <8 x float> %1705, i64 7
  %2379 = call float @llvm.fma.f32(float %2378, float %365, float %368)
  %2380 = icmp sgt i32 %2212, %2049
  %2381 = and i1 %2380, %407
  %2382 = select i1 %2381, float -3.000000e+38, float %2365
  %2383 = icmp sgt i32 %2216, %2049
  %2384 = and i1 %2383, %407
  %2385 = select i1 %2384, float -3.000000e+38, float %2367
  %2386 = icmp sgt i32 %2220, %2049
  %2387 = and i1 %2386, %407
  %2388 = select i1 %2387, float -3.000000e+38, float %2369
  %2389 = icmp sgt i32 %2224, %2049
  %2390 = and i1 %2389, %407
  %2391 = select i1 %2390, float -3.000000e+38, float %2371
  %2392 = icmp sgt i32 %2228, %2049
  %2393 = and i1 %2392, %407
  %2394 = select i1 %2393, float -3.000000e+38, float %2373
  %2395 = icmp sgt i32 %2232, %2049
  %2396 = and i1 %2395, %407
  %2397 = select i1 %2396, float -3.000000e+38, float %2375
  %2398 = icmp sgt i32 %2236, %2049
  %2399 = and i1 %2398, %407
  %2400 = select i1 %2399, float -3.000000e+38, float %2377
  %2401 = icmp sgt i32 %2240, %2049
  %2402 = and i1 %2401, %407
  %2403 = select i1 %2402, float -3.000000e+38, float %2379
  %2404 = call float @llvm.amdgcn.exp2.f32(float %2382)
  %2405 = call float @llvm.amdgcn.exp2.f32(float %2385)
  %2406 = call float @llvm.amdgcn.exp2.f32(float %2388)
  %2407 = call float @llvm.amdgcn.exp2.f32(float %2391)
  %2408 = call float @llvm.amdgcn.exp2.f32(float %2394)
  %2409 = call float @llvm.amdgcn.exp2.f32(float %2397)
  %2410 = call float @llvm.amdgcn.exp2.f32(float %2400)
  %2411 = call float @llvm.amdgcn.exp2.f32(float %2403)
  %2412 = extractelement <8 x float> %1706, i64 0
  %2413 = call float @llvm.fma.f32(float %2412, float %16, float %373)
  %2414 = fmul float %2404, %2413
  %2415 = fptrunc float %2414 to bfloat
  %2416 = extractelement <8 x float> %1706, i64 1
  %2417 = call float @llvm.fma.f32(float %2416, float %16, float %373)
  %2418 = fmul float %2405, %2417
  %2419 = fptrunc float %2418 to bfloat
  %2420 = extractelement <8 x float> %1706, i64 2
  %2421 = call float @llvm.fma.f32(float %2420, float %16, float %373)
  %2422 = fmul float %2406, %2421
  %2423 = fptrunc float %2422 to bfloat
  %2424 = extractelement <8 x float> %1706, i64 3
  %2425 = call float @llvm.fma.f32(float %2424, float %16, float %373)
  %2426 = fmul float %2407, %2425
  %2427 = fptrunc float %2426 to bfloat
  %2428 = extractelement <8 x float> %1706, i64 4
  %2429 = call float @llvm.fma.f32(float %2428, float %16, float %373)
  %2430 = fmul float %2408, %2429
  %2431 = fptrunc float %2430 to bfloat
  %2432 = extractelement <8 x float> %1706, i64 5
  %2433 = call float @llvm.fma.f32(float %2432, float %16, float %373)
  %2434 = fmul float %2409, %2433
  %2435 = fptrunc float %2434 to bfloat
  %2436 = extractelement <8 x float> %1706, i64 6
  %2437 = call float @llvm.fma.f32(float %2436, float %16, float %373)
  %2438 = fmul float %2410, %2437
  %2439 = fptrunc float %2438 to bfloat
  %2440 = extractelement <8 x float> %1706, i64 7
  %2441 = call float @llvm.fma.f32(float %2440, float %16, float %373)
  %2442 = fmul float %2411, %2441
  %2443 = fptrunc float %2442 to bfloat
  %2444 = extractelement <8 x float> %1707, i64 0
  %2445 = call float @llvm.fma.f32(float %2444, float %365, float %369)
  %2446 = extractelement <8 x float> %1707, i64 1
  %2447 = call float @llvm.fma.f32(float %2446, float %365, float %369)
  %2448 = extractelement <8 x float> %1707, i64 2
  %2449 = call float @llvm.fma.f32(float %2448, float %365, float %369)
  %2450 = extractelement <8 x float> %1707, i64 3
  %2451 = call float @llvm.fma.f32(float %2450, float %365, float %369)
  %2452 = extractelement <8 x float> %1707, i64 4
  %2453 = call float @llvm.fma.f32(float %2452, float %365, float %369)
  %2454 = extractelement <8 x float> %1707, i64 5
  %2455 = call float @llvm.fma.f32(float %2454, float %365, float %369)
  %2456 = extractelement <8 x float> %1707, i64 6
  %2457 = call float @llvm.fma.f32(float %2456, float %365, float %369)
  %2458 = extractelement <8 x float> %1707, i64 7
  %2459 = call float @llvm.fma.f32(float %2458, float %365, float %369)
  %2460 = icmp sgt i32 %2212, %2130
  %2461 = and i1 %2460, %407
  %2462 = select i1 %2461, float -3.000000e+38, float %2445
  %2463 = icmp sgt i32 %2216, %2130
  %2464 = and i1 %2463, %407
  %2465 = select i1 %2464, float -3.000000e+38, float %2447
  %2466 = icmp sgt i32 %2220, %2130
  %2467 = and i1 %2466, %407
  %2468 = select i1 %2467, float -3.000000e+38, float %2449
  %2469 = icmp sgt i32 %2224, %2130
  %2470 = and i1 %2469, %407
  %2471 = select i1 %2470, float -3.000000e+38, float %2451
  %2472 = icmp sgt i32 %2228, %2130
  %2473 = and i1 %2472, %407
  %2474 = select i1 %2473, float -3.000000e+38, float %2453
  %2475 = icmp sgt i32 %2232, %2130
  %2476 = and i1 %2475, %407
  %2477 = select i1 %2476, float -3.000000e+38, float %2455
  %2478 = icmp sgt i32 %2236, %2130
  %2479 = and i1 %2478, %407
  %2480 = select i1 %2479, float -3.000000e+38, float %2457
  %2481 = icmp sgt i32 %2240, %2130
  %2482 = and i1 %2481, %407
  %2483 = select i1 %2482, float -3.000000e+38, float %2459
  %2484 = call float @llvm.amdgcn.exp2.f32(float %2462)
  %2485 = call float @llvm.amdgcn.exp2.f32(float %2465)
  %2486 = call float @llvm.amdgcn.exp2.f32(float %2468)
  %2487 = call float @llvm.amdgcn.exp2.f32(float %2471)
  %2488 = call float @llvm.amdgcn.exp2.f32(float %2474)
  %2489 = call float @llvm.amdgcn.exp2.f32(float %2477)
  %2490 = call float @llvm.amdgcn.exp2.f32(float %2480)
  %2491 = call float @llvm.amdgcn.exp2.f32(float %2483)
  %2492 = extractelement <8 x float> %1708, i64 0
  %2493 = call float @llvm.fma.f32(float %2492, float %16, float %374)
  %2494 = fmul float %2484, %2493
  %2495 = fptrunc float %2494 to bfloat
  %2496 = extractelement <8 x float> %1708, i64 1
  %2497 = call float @llvm.fma.f32(float %2496, float %16, float %374)
  %2498 = fmul float %2485, %2497
  %2499 = fptrunc float %2498 to bfloat
  %2500 = extractelement <8 x float> %1708, i64 2
  %2501 = call float @llvm.fma.f32(float %2500, float %16, float %374)
  %2502 = fmul float %2486, %2501
  %2503 = fptrunc float %2502 to bfloat
  %2504 = extractelement <8 x float> %1708, i64 3
  %2505 = call float @llvm.fma.f32(float %2504, float %16, float %374)
  %2506 = fmul float %2487, %2505
  %2507 = fptrunc float %2506 to bfloat
  %2508 = extractelement <8 x float> %1708, i64 4
  %2509 = call float @llvm.fma.f32(float %2508, float %16, float %374)
  %2510 = fmul float %2488, %2509
  %2511 = fptrunc float %2510 to bfloat
  %2512 = extractelement <8 x float> %1708, i64 5
  %2513 = call float @llvm.fma.f32(float %2512, float %16, float %374)
  %2514 = fmul float %2489, %2513
  %2515 = fptrunc float %2514 to bfloat
  %2516 = extractelement <8 x float> %1708, i64 6
  %2517 = call float @llvm.fma.f32(float %2516, float %16, float %374)
  %2518 = fmul float %2490, %2517
  %2519 = fptrunc float %2518 to bfloat
  %2520 = extractelement <8 x float> %1708, i64 7
  %2521 = call float @llvm.fma.f32(float %2520, float %16, float %374)
  %2522 = fmul float %2491, %2521
  %2523 = fptrunc float %2522 to bfloat
  %2524 = insertelement <16 x bfloat> poison, bfloat %1923, i64 0
  %2525 = insertelement <16 x bfloat> %2524, bfloat %1927, i64 1
  %2526 = insertelement <16 x bfloat> %2525, bfloat %1931, i64 2
  %2527 = insertelement <16 x bfloat> %2526, bfloat %1935, i64 3
  %2528 = insertelement <16 x bfloat> %2527, bfloat %1939, i64 4
  %2529 = insertelement <16 x bfloat> %2528, bfloat %1943, i64 5
  %2530 = insertelement <16 x bfloat> %2529, bfloat %1947, i64 6
  %2531 = insertelement <16 x bfloat> %2530, bfloat %1951, i64 7
  %2532 = insertelement <16 x bfloat> %2531, bfloat %2255, i64 8
  %2533 = insertelement <16 x bfloat> %2532, bfloat %2259, i64 9
  %2534 = insertelement <16 x bfloat> %2533, bfloat %2263, i64 10
  %2535 = insertelement <16 x bfloat> %2534, bfloat %2267, i64 11
  %2536 = insertelement <16 x bfloat> %2535, bfloat %2271, i64 12
  %2537 = insertelement <16 x bfloat> %2536, bfloat %2275, i64 13
  %2538 = insertelement <16 x bfloat> %2537, bfloat %2279, i64 14
  %2539 = insertelement <16 x bfloat> %2538, bfloat %2283, i64 15
  %2540 = insertelement <16 x bfloat> poison, bfloat %2004, i64 0
  %2541 = insertelement <16 x bfloat> %2540, bfloat %2008, i64 1
  %2542 = insertelement <16 x bfloat> %2541, bfloat %2012, i64 2
  %2543 = insertelement <16 x bfloat> %2542, bfloat %2016, i64 3
  %2544 = insertelement <16 x bfloat> %2543, bfloat %2020, i64 4
  %2545 = insertelement <16 x bfloat> %2544, bfloat %2024, i64 5
  %2546 = insertelement <16 x bfloat> %2545, bfloat %2028, i64 6
  %2547 = insertelement <16 x bfloat> %2546, bfloat %2032, i64 7
  %2548 = insertelement <16 x bfloat> %2547, bfloat %2335, i64 8
  %2549 = insertelement <16 x bfloat> %2548, bfloat %2339, i64 9
  %2550 = insertelement <16 x bfloat> %2549, bfloat %2343, i64 10
  %2551 = insertelement <16 x bfloat> %2550, bfloat %2347, i64 11
  %2552 = insertelement <16 x bfloat> %2551, bfloat %2351, i64 12
  %2553 = insertelement <16 x bfloat> %2552, bfloat %2355, i64 13
  %2554 = insertelement <16 x bfloat> %2553, bfloat %2359, i64 14
  %2555 = insertelement <16 x bfloat> %2554, bfloat %2363, i64 15
  %2556 = insertelement <16 x bfloat> poison, bfloat %2085, i64 0
  %2557 = insertelement <16 x bfloat> %2556, bfloat %2089, i64 1
  %2558 = insertelement <16 x bfloat> %2557, bfloat %2093, i64 2
  %2559 = insertelement <16 x bfloat> %2558, bfloat %2097, i64 3
  %2560 = insertelement <16 x bfloat> %2559, bfloat %2101, i64 4
  %2561 = insertelement <16 x bfloat> %2560, bfloat %2105, i64 5
  %2562 = insertelement <16 x bfloat> %2561, bfloat %2109, i64 6
  %2563 = insertelement <16 x bfloat> %2562, bfloat %2113, i64 7
  %2564 = insertelement <16 x bfloat> %2563, bfloat %2415, i64 8
  %2565 = insertelement <16 x bfloat> %2564, bfloat %2419, i64 9
  %2566 = insertelement <16 x bfloat> %2565, bfloat %2423, i64 10
  %2567 = insertelement <16 x bfloat> %2566, bfloat %2427, i64 11
  %2568 = insertelement <16 x bfloat> %2567, bfloat %2431, i64 12
  %2569 = insertelement <16 x bfloat> %2568, bfloat %2435, i64 13
  %2570 = insertelement <16 x bfloat> %2569, bfloat %2439, i64 14
  %2571 = insertelement <16 x bfloat> %2570, bfloat %2443, i64 15
  %2572 = insertelement <16 x bfloat> poison, bfloat %2166, i64 0
  %2573 = insertelement <16 x bfloat> %2572, bfloat %2170, i64 1
  %2574 = insertelement <16 x bfloat> %2573, bfloat %2174, i64 2
  %2575 = insertelement <16 x bfloat> %2574, bfloat %2178, i64 3
  %2576 = insertelement <16 x bfloat> %2575, bfloat %2182, i64 4
  %2577 = insertelement <16 x bfloat> %2576, bfloat %2186, i64 5
  %2578 = insertelement <16 x bfloat> %2577, bfloat %2190, i64 6
  %2579 = insertelement <16 x bfloat> %2578, bfloat %2194, i64 7
  %2580 = insertelement <16 x bfloat> %2579, bfloat %2495, i64 8
  %2581 = insertelement <16 x bfloat> %2580, bfloat %2499, i64 9
  %2582 = insertelement <16 x bfloat> %2581, bfloat %2503, i64 10
  %2583 = insertelement <16 x bfloat> %2582, bfloat %2507, i64 11
  %2584 = insertelement <16 x bfloat> %2583, bfloat %2511, i64 12
  %2585 = insertelement <16 x bfloat> %2584, bfloat %2515, i64 13
  %2586 = insertelement <16 x bfloat> %2585, bfloat %2519, i64 14
  %2587 = insertelement <16 x bfloat> %2586, bfloat %2523, i64 15
  %2588 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2539, <16 x bfloat> %1716, i16 0, <8 x float> %1507, i1 false, i1 false)
  %2589 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2539, <16 x bfloat> %1723, i16 0, <8 x float> %1508, i1 false, i1 false)
  %2590 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2539, <16 x bfloat> %1730, i16 0, <8 x float> %1509, i1 false, i1 false)
  %2591 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2539, <16 x bfloat> %1737, i16 0, <8 x float> %1510, i1 false, i1 false)
  %2592 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2539, <16 x bfloat> %1744, i16 0, <8 x float> %1511, i1 false, i1 false)
  %2593 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2539, <16 x bfloat> %1751, i16 0, <8 x float> %1512, i1 false, i1 false)
  %2594 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2539, <16 x bfloat> %1758, i16 0, <8 x float> %1513, i1 false, i1 false)
  %2595 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2539, <16 x bfloat> %1765, i16 0, <8 x float> %1514, i1 false, i1 false)
  %2596 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2555, <16 x bfloat> %1716, i16 0, <8 x float> %1515, i1 false, i1 false)
  %2597 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2555, <16 x bfloat> %1723, i16 0, <8 x float> %1516, i1 false, i1 false)
  %2598 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2555, <16 x bfloat> %1730, i16 0, <8 x float> %1517, i1 false, i1 false)
  %2599 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2555, <16 x bfloat> %1737, i16 0, <8 x float> %1518, i1 false, i1 false)
  %2600 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2555, <16 x bfloat> %1744, i16 0, <8 x float> %1519, i1 false, i1 false)
  %2601 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2555, <16 x bfloat> %1751, i16 0, <8 x float> %1520, i1 false, i1 false)
  %2602 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2555, <16 x bfloat> %1758, i16 0, <8 x float> %1521, i1 false, i1 false)
  %2603 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2555, <16 x bfloat> %1765, i16 0, <8 x float> %1522, i1 false, i1 false)
  %2604 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2571, <16 x bfloat> %1716, i16 0, <8 x float> %1523, i1 false, i1 false)
  %2605 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2571, <16 x bfloat> %1723, i16 0, <8 x float> %1524, i1 false, i1 false)
  %2606 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2571, <16 x bfloat> %1730, i16 0, <8 x float> %1525, i1 false, i1 false)
  %2607 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2571, <16 x bfloat> %1737, i16 0, <8 x float> %1526, i1 false, i1 false)
  %2608 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2571, <16 x bfloat> %1744, i16 0, <8 x float> %1527, i1 false, i1 false)
  %2609 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2571, <16 x bfloat> %1751, i16 0, <8 x float> %1528, i1 false, i1 false)
  %2610 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2571, <16 x bfloat> %1758, i16 0, <8 x float> %1529, i1 false, i1 false)
  %2611 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2571, <16 x bfloat> %1765, i16 0, <8 x float> %1530, i1 false, i1 false)
  %2612 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2587, <16 x bfloat> %1716, i16 0, <8 x float> %1531, i1 false, i1 false)
  %2613 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2587, <16 x bfloat> %1723, i16 0, <8 x float> %1532, i1 false, i1 false)
  %2614 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2587, <16 x bfloat> %1730, i16 0, <8 x float> %1533, i1 false, i1 false)
  %2615 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2587, <16 x bfloat> %1737, i16 0, <8 x float> %1534, i1 false, i1 false)
  %2616 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2587, <16 x bfloat> %1744, i16 0, <8 x float> %1535, i1 false, i1 false)
  %2617 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2587, <16 x bfloat> %1751, i16 0, <8 x float> %1536, i1 false, i1 false)
  %2618 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2587, <16 x bfloat> %1758, i16 0, <8 x float> %1537, i1 false, i1 false)
  %2619 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2587, <16 x bfloat> %1765, i16 0, <8 x float> %1538, i1 false, i1 false)
  %2620 = add i64 %1506, 1
  br label %1505

2621:                                             ; preds = %1505
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  %2622 = mul i32 %95, %19
  %2623 = mul i32 %2622, 128
  %2624 = mul i32 %53, 128
  %2625 = add i32 %2623, %2624
  %2626 = add i32 %79, %375
  %2627 = extractelement <8 x float> %1507, i64 0
  %2628 = fptrunc float %2627 to bfloat
  %2629 = mul i32 %2626, %19
  %2630 = mul i32 %2629, 128
  %2631 = add i32 %2625, %2630
  %2632 = add i32 %2631, %70
  %2633 = mul i32 %2632, 2
  %2634 = bitcast bfloat %2628 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2634, ptr addrspace(8) %93, i32 %2633, i32 0, i32 0)
  %2635 = add i32 %2626, 1
  %2636 = extractelement <8 x float> %1507, i64 1
  %2637 = fptrunc float %2636 to bfloat
  %2638 = mul i32 %2635, %19
  %2639 = mul i32 %2638, 128
  %2640 = add i32 %2625, %2639
  %2641 = add i32 %2640, %70
  %2642 = mul i32 %2641, 2
  %2643 = bitcast bfloat %2637 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2643, ptr addrspace(8) %93, i32 %2642, i32 0, i32 0)
  %2644 = add i32 %2626, 2
  %2645 = extractelement <8 x float> %1507, i64 2
  %2646 = fptrunc float %2645 to bfloat
  %2647 = mul i32 %2644, %19
  %2648 = mul i32 %2647, 128
  %2649 = add i32 %2625, %2648
  %2650 = add i32 %2649, %70
  %2651 = mul i32 %2650, 2
  %2652 = bitcast bfloat %2646 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2652, ptr addrspace(8) %93, i32 %2651, i32 0, i32 0)
  %2653 = add i32 %2626, 3
  %2654 = extractelement <8 x float> %1507, i64 3
  %2655 = fptrunc float %2654 to bfloat
  %2656 = mul i32 %2653, %19
  %2657 = mul i32 %2656, 128
  %2658 = add i32 %2625, %2657
  %2659 = add i32 %2658, %70
  %2660 = mul i32 %2659, 2
  %2661 = bitcast bfloat %2655 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2661, ptr addrspace(8) %93, i32 %2660, i32 0, i32 0)
  %2662 = add i32 %2626, 4
  %2663 = extractelement <8 x float> %1507, i64 4
  %2664 = fptrunc float %2663 to bfloat
  %2665 = mul i32 %2662, %19
  %2666 = mul i32 %2665, 128
  %2667 = add i32 %2625, %2666
  %2668 = add i32 %2667, %70
  %2669 = mul i32 %2668, 2
  %2670 = bitcast bfloat %2664 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2670, ptr addrspace(8) %93, i32 %2669, i32 0, i32 0)
  %2671 = add i32 %2626, 5
  %2672 = extractelement <8 x float> %1507, i64 5
  %2673 = fptrunc float %2672 to bfloat
  %2674 = mul i32 %2671, %19
  %2675 = mul i32 %2674, 128
  %2676 = add i32 %2625, %2675
  %2677 = add i32 %2676, %70
  %2678 = mul i32 %2677, 2
  %2679 = bitcast bfloat %2673 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2679, ptr addrspace(8) %93, i32 %2678, i32 0, i32 0)
  %2680 = add i32 %2626, 6
  %2681 = extractelement <8 x float> %1507, i64 6
  %2682 = fptrunc float %2681 to bfloat
  %2683 = mul i32 %2680, %19
  %2684 = mul i32 %2683, 128
  %2685 = add i32 %2625, %2684
  %2686 = add i32 %2685, %70
  %2687 = mul i32 %2686, 2
  %2688 = bitcast bfloat %2682 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2688, ptr addrspace(8) %93, i32 %2687, i32 0, i32 0)
  %2689 = add i32 %2626, 7
  %2690 = extractelement <8 x float> %1507, i64 7
  %2691 = fptrunc float %2690 to bfloat
  %2692 = mul i32 %2689, %19
  %2693 = mul i32 %2692, 128
  %2694 = add i32 %2625, %2693
  %2695 = add i32 %2694, %70
  %2696 = mul i32 %2695, 2
  %2697 = bitcast bfloat %2691 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2697, ptr addrspace(8) %93, i32 %2696, i32 0, i32 0)
  %2698 = extractelement <8 x float> %1508, i64 0
  %2699 = fptrunc float %2698 to bfloat
  %2700 = add i32 %2631, 16
  %2701 = add i32 %2700, %70
  %2702 = mul i32 %2701, 2
  %2703 = bitcast bfloat %2699 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2703, ptr addrspace(8) %93, i32 %2702, i32 0, i32 0)
  %2704 = extractelement <8 x float> %1508, i64 1
  %2705 = fptrunc float %2704 to bfloat
  %2706 = add i32 %2640, 16
  %2707 = add i32 %2706, %70
  %2708 = mul i32 %2707, 2
  %2709 = bitcast bfloat %2705 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2709, ptr addrspace(8) %93, i32 %2708, i32 0, i32 0)
  %2710 = extractelement <8 x float> %1508, i64 2
  %2711 = fptrunc float %2710 to bfloat
  %2712 = add i32 %2649, 16
  %2713 = add i32 %2712, %70
  %2714 = mul i32 %2713, 2
  %2715 = bitcast bfloat %2711 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2715, ptr addrspace(8) %93, i32 %2714, i32 0, i32 0)
  %2716 = extractelement <8 x float> %1508, i64 3
  %2717 = fptrunc float %2716 to bfloat
  %2718 = add i32 %2658, 16
  %2719 = add i32 %2718, %70
  %2720 = mul i32 %2719, 2
  %2721 = bitcast bfloat %2717 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2721, ptr addrspace(8) %93, i32 %2720, i32 0, i32 0)
  %2722 = extractelement <8 x float> %1508, i64 4
  %2723 = fptrunc float %2722 to bfloat
  %2724 = add i32 %2667, 16
  %2725 = add i32 %2724, %70
  %2726 = mul i32 %2725, 2
  %2727 = bitcast bfloat %2723 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2727, ptr addrspace(8) %93, i32 %2726, i32 0, i32 0)
  %2728 = extractelement <8 x float> %1508, i64 5
  %2729 = fptrunc float %2728 to bfloat
  %2730 = add i32 %2676, 16
  %2731 = add i32 %2730, %70
  %2732 = mul i32 %2731, 2
  %2733 = bitcast bfloat %2729 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2733, ptr addrspace(8) %93, i32 %2732, i32 0, i32 0)
  %2734 = extractelement <8 x float> %1508, i64 6
  %2735 = fptrunc float %2734 to bfloat
  %2736 = add i32 %2685, 16
  %2737 = add i32 %2736, %70
  %2738 = mul i32 %2737, 2
  %2739 = bitcast bfloat %2735 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2739, ptr addrspace(8) %93, i32 %2738, i32 0, i32 0)
  %2740 = extractelement <8 x float> %1508, i64 7
  %2741 = fptrunc float %2740 to bfloat
  %2742 = add i32 %2694, 16
  %2743 = add i32 %2742, %70
  %2744 = mul i32 %2743, 2
  %2745 = bitcast bfloat %2741 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2745, ptr addrspace(8) %93, i32 %2744, i32 0, i32 0)
  %2746 = extractelement <8 x float> %1509, i64 0
  %2747 = fptrunc float %2746 to bfloat
  %2748 = add i32 %2631, 32
  %2749 = add i32 %2748, %70
  %2750 = mul i32 %2749, 2
  %2751 = bitcast bfloat %2747 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2751, ptr addrspace(8) %93, i32 %2750, i32 0, i32 0)
  %2752 = extractelement <8 x float> %1509, i64 1
  %2753 = fptrunc float %2752 to bfloat
  %2754 = add i32 %2640, 32
  %2755 = add i32 %2754, %70
  %2756 = mul i32 %2755, 2
  %2757 = bitcast bfloat %2753 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2757, ptr addrspace(8) %93, i32 %2756, i32 0, i32 0)
  %2758 = extractelement <8 x float> %1509, i64 2
  %2759 = fptrunc float %2758 to bfloat
  %2760 = add i32 %2649, 32
  %2761 = add i32 %2760, %70
  %2762 = mul i32 %2761, 2
  %2763 = bitcast bfloat %2759 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2763, ptr addrspace(8) %93, i32 %2762, i32 0, i32 0)
  %2764 = extractelement <8 x float> %1509, i64 3
  %2765 = fptrunc float %2764 to bfloat
  %2766 = add i32 %2658, 32
  %2767 = add i32 %2766, %70
  %2768 = mul i32 %2767, 2
  %2769 = bitcast bfloat %2765 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2769, ptr addrspace(8) %93, i32 %2768, i32 0, i32 0)
  %2770 = extractelement <8 x float> %1509, i64 4
  %2771 = fptrunc float %2770 to bfloat
  %2772 = add i32 %2667, 32
  %2773 = add i32 %2772, %70
  %2774 = mul i32 %2773, 2
  %2775 = bitcast bfloat %2771 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2775, ptr addrspace(8) %93, i32 %2774, i32 0, i32 0)
  %2776 = extractelement <8 x float> %1509, i64 5
  %2777 = fptrunc float %2776 to bfloat
  %2778 = add i32 %2676, 32
  %2779 = add i32 %2778, %70
  %2780 = mul i32 %2779, 2
  %2781 = bitcast bfloat %2777 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2781, ptr addrspace(8) %93, i32 %2780, i32 0, i32 0)
  %2782 = extractelement <8 x float> %1509, i64 6
  %2783 = fptrunc float %2782 to bfloat
  %2784 = add i32 %2685, 32
  %2785 = add i32 %2784, %70
  %2786 = mul i32 %2785, 2
  %2787 = bitcast bfloat %2783 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2787, ptr addrspace(8) %93, i32 %2786, i32 0, i32 0)
  %2788 = extractelement <8 x float> %1509, i64 7
  %2789 = fptrunc float %2788 to bfloat
  %2790 = add i32 %2694, 32
  %2791 = add i32 %2790, %70
  %2792 = mul i32 %2791, 2
  %2793 = bitcast bfloat %2789 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2793, ptr addrspace(8) %93, i32 %2792, i32 0, i32 0)
  %2794 = extractelement <8 x float> %1510, i64 0
  %2795 = fptrunc float %2794 to bfloat
  %2796 = add i32 %2631, 48
  %2797 = add i32 %2796, %70
  %2798 = mul i32 %2797, 2
  %2799 = bitcast bfloat %2795 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2799, ptr addrspace(8) %93, i32 %2798, i32 0, i32 0)
  %2800 = extractelement <8 x float> %1510, i64 1
  %2801 = fptrunc float %2800 to bfloat
  %2802 = add i32 %2640, 48
  %2803 = add i32 %2802, %70
  %2804 = mul i32 %2803, 2
  %2805 = bitcast bfloat %2801 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2805, ptr addrspace(8) %93, i32 %2804, i32 0, i32 0)
  %2806 = extractelement <8 x float> %1510, i64 2
  %2807 = fptrunc float %2806 to bfloat
  %2808 = add i32 %2649, 48
  %2809 = add i32 %2808, %70
  %2810 = mul i32 %2809, 2
  %2811 = bitcast bfloat %2807 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2811, ptr addrspace(8) %93, i32 %2810, i32 0, i32 0)
  %2812 = extractelement <8 x float> %1510, i64 3
  %2813 = fptrunc float %2812 to bfloat
  %2814 = add i32 %2658, 48
  %2815 = add i32 %2814, %70
  %2816 = mul i32 %2815, 2
  %2817 = bitcast bfloat %2813 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2817, ptr addrspace(8) %93, i32 %2816, i32 0, i32 0)
  %2818 = extractelement <8 x float> %1510, i64 4
  %2819 = fptrunc float %2818 to bfloat
  %2820 = add i32 %2667, 48
  %2821 = add i32 %2820, %70
  %2822 = mul i32 %2821, 2
  %2823 = bitcast bfloat %2819 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2823, ptr addrspace(8) %93, i32 %2822, i32 0, i32 0)
  %2824 = extractelement <8 x float> %1510, i64 5
  %2825 = fptrunc float %2824 to bfloat
  %2826 = add i32 %2676, 48
  %2827 = add i32 %2826, %70
  %2828 = mul i32 %2827, 2
  %2829 = bitcast bfloat %2825 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2829, ptr addrspace(8) %93, i32 %2828, i32 0, i32 0)
  %2830 = extractelement <8 x float> %1510, i64 6
  %2831 = fptrunc float %2830 to bfloat
  %2832 = add i32 %2685, 48
  %2833 = add i32 %2832, %70
  %2834 = mul i32 %2833, 2
  %2835 = bitcast bfloat %2831 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2835, ptr addrspace(8) %93, i32 %2834, i32 0, i32 0)
  %2836 = extractelement <8 x float> %1510, i64 7
  %2837 = fptrunc float %2836 to bfloat
  %2838 = add i32 %2694, 48
  %2839 = add i32 %2838, %70
  %2840 = mul i32 %2839, 2
  %2841 = bitcast bfloat %2837 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2841, ptr addrspace(8) %93, i32 %2840, i32 0, i32 0)
  %2842 = extractelement <8 x float> %1511, i64 0
  %2843 = fptrunc float %2842 to bfloat
  %2844 = add i32 %2631, 64
  %2845 = add i32 %2844, %70
  %2846 = mul i32 %2845, 2
  %2847 = bitcast bfloat %2843 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2847, ptr addrspace(8) %93, i32 %2846, i32 0, i32 0)
  %2848 = extractelement <8 x float> %1511, i64 1
  %2849 = fptrunc float %2848 to bfloat
  %2850 = add i32 %2640, 64
  %2851 = add i32 %2850, %70
  %2852 = mul i32 %2851, 2
  %2853 = bitcast bfloat %2849 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2853, ptr addrspace(8) %93, i32 %2852, i32 0, i32 0)
  %2854 = extractelement <8 x float> %1511, i64 2
  %2855 = fptrunc float %2854 to bfloat
  %2856 = add i32 %2649, 64
  %2857 = add i32 %2856, %70
  %2858 = mul i32 %2857, 2
  %2859 = bitcast bfloat %2855 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2859, ptr addrspace(8) %93, i32 %2858, i32 0, i32 0)
  %2860 = extractelement <8 x float> %1511, i64 3
  %2861 = fptrunc float %2860 to bfloat
  %2862 = add i32 %2658, 64
  %2863 = add i32 %2862, %70
  %2864 = mul i32 %2863, 2
  %2865 = bitcast bfloat %2861 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2865, ptr addrspace(8) %93, i32 %2864, i32 0, i32 0)
  %2866 = extractelement <8 x float> %1511, i64 4
  %2867 = fptrunc float %2866 to bfloat
  %2868 = add i32 %2667, 64
  %2869 = add i32 %2868, %70
  %2870 = mul i32 %2869, 2
  %2871 = bitcast bfloat %2867 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2871, ptr addrspace(8) %93, i32 %2870, i32 0, i32 0)
  %2872 = extractelement <8 x float> %1511, i64 5
  %2873 = fptrunc float %2872 to bfloat
  %2874 = add i32 %2676, 64
  %2875 = add i32 %2874, %70
  %2876 = mul i32 %2875, 2
  %2877 = bitcast bfloat %2873 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2877, ptr addrspace(8) %93, i32 %2876, i32 0, i32 0)
  %2878 = extractelement <8 x float> %1511, i64 6
  %2879 = fptrunc float %2878 to bfloat
  %2880 = add i32 %2685, 64
  %2881 = add i32 %2880, %70
  %2882 = mul i32 %2881, 2
  %2883 = bitcast bfloat %2879 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2883, ptr addrspace(8) %93, i32 %2882, i32 0, i32 0)
  %2884 = extractelement <8 x float> %1511, i64 7
  %2885 = fptrunc float %2884 to bfloat
  %2886 = add i32 %2694, 64
  %2887 = add i32 %2886, %70
  %2888 = mul i32 %2887, 2
  %2889 = bitcast bfloat %2885 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2889, ptr addrspace(8) %93, i32 %2888, i32 0, i32 0)
  %2890 = extractelement <8 x float> %1512, i64 0
  %2891 = fptrunc float %2890 to bfloat
  %2892 = add i32 %2631, 80
  %2893 = add i32 %2892, %70
  %2894 = mul i32 %2893, 2
  %2895 = bitcast bfloat %2891 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2895, ptr addrspace(8) %93, i32 %2894, i32 0, i32 0)
  %2896 = extractelement <8 x float> %1512, i64 1
  %2897 = fptrunc float %2896 to bfloat
  %2898 = add i32 %2640, 80
  %2899 = add i32 %2898, %70
  %2900 = mul i32 %2899, 2
  %2901 = bitcast bfloat %2897 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2901, ptr addrspace(8) %93, i32 %2900, i32 0, i32 0)
  %2902 = extractelement <8 x float> %1512, i64 2
  %2903 = fptrunc float %2902 to bfloat
  %2904 = add i32 %2649, 80
  %2905 = add i32 %2904, %70
  %2906 = mul i32 %2905, 2
  %2907 = bitcast bfloat %2903 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2907, ptr addrspace(8) %93, i32 %2906, i32 0, i32 0)
  %2908 = extractelement <8 x float> %1512, i64 3
  %2909 = fptrunc float %2908 to bfloat
  %2910 = add i32 %2658, 80
  %2911 = add i32 %2910, %70
  %2912 = mul i32 %2911, 2
  %2913 = bitcast bfloat %2909 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2913, ptr addrspace(8) %93, i32 %2912, i32 0, i32 0)
  %2914 = extractelement <8 x float> %1512, i64 4
  %2915 = fptrunc float %2914 to bfloat
  %2916 = add i32 %2667, 80
  %2917 = add i32 %2916, %70
  %2918 = mul i32 %2917, 2
  %2919 = bitcast bfloat %2915 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2919, ptr addrspace(8) %93, i32 %2918, i32 0, i32 0)
  %2920 = extractelement <8 x float> %1512, i64 5
  %2921 = fptrunc float %2920 to bfloat
  %2922 = add i32 %2676, 80
  %2923 = add i32 %2922, %70
  %2924 = mul i32 %2923, 2
  %2925 = bitcast bfloat %2921 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2925, ptr addrspace(8) %93, i32 %2924, i32 0, i32 0)
  %2926 = extractelement <8 x float> %1512, i64 6
  %2927 = fptrunc float %2926 to bfloat
  %2928 = add i32 %2685, 80
  %2929 = add i32 %2928, %70
  %2930 = mul i32 %2929, 2
  %2931 = bitcast bfloat %2927 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2931, ptr addrspace(8) %93, i32 %2930, i32 0, i32 0)
  %2932 = extractelement <8 x float> %1512, i64 7
  %2933 = fptrunc float %2932 to bfloat
  %2934 = add i32 %2694, 80
  %2935 = add i32 %2934, %70
  %2936 = mul i32 %2935, 2
  %2937 = bitcast bfloat %2933 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2937, ptr addrspace(8) %93, i32 %2936, i32 0, i32 0)
  %2938 = extractelement <8 x float> %1513, i64 0
  %2939 = fptrunc float %2938 to bfloat
  %2940 = add i32 %2631, 96
  %2941 = add i32 %2940, %70
  %2942 = mul i32 %2941, 2
  %2943 = bitcast bfloat %2939 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2943, ptr addrspace(8) %93, i32 %2942, i32 0, i32 0)
  %2944 = extractelement <8 x float> %1513, i64 1
  %2945 = fptrunc float %2944 to bfloat
  %2946 = add i32 %2640, 96
  %2947 = add i32 %2946, %70
  %2948 = mul i32 %2947, 2
  %2949 = bitcast bfloat %2945 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2949, ptr addrspace(8) %93, i32 %2948, i32 0, i32 0)
  %2950 = extractelement <8 x float> %1513, i64 2
  %2951 = fptrunc float %2950 to bfloat
  %2952 = add i32 %2649, 96
  %2953 = add i32 %2952, %70
  %2954 = mul i32 %2953, 2
  %2955 = bitcast bfloat %2951 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2955, ptr addrspace(8) %93, i32 %2954, i32 0, i32 0)
  %2956 = extractelement <8 x float> %1513, i64 3
  %2957 = fptrunc float %2956 to bfloat
  %2958 = add i32 %2658, 96
  %2959 = add i32 %2958, %70
  %2960 = mul i32 %2959, 2
  %2961 = bitcast bfloat %2957 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2961, ptr addrspace(8) %93, i32 %2960, i32 0, i32 0)
  %2962 = extractelement <8 x float> %1513, i64 4
  %2963 = fptrunc float %2962 to bfloat
  %2964 = add i32 %2667, 96
  %2965 = add i32 %2964, %70
  %2966 = mul i32 %2965, 2
  %2967 = bitcast bfloat %2963 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2967, ptr addrspace(8) %93, i32 %2966, i32 0, i32 0)
  %2968 = extractelement <8 x float> %1513, i64 5
  %2969 = fptrunc float %2968 to bfloat
  %2970 = add i32 %2676, 96
  %2971 = add i32 %2970, %70
  %2972 = mul i32 %2971, 2
  %2973 = bitcast bfloat %2969 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2973, ptr addrspace(8) %93, i32 %2972, i32 0, i32 0)
  %2974 = extractelement <8 x float> %1513, i64 6
  %2975 = fptrunc float %2974 to bfloat
  %2976 = add i32 %2685, 96
  %2977 = add i32 %2976, %70
  %2978 = mul i32 %2977, 2
  %2979 = bitcast bfloat %2975 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2979, ptr addrspace(8) %93, i32 %2978, i32 0, i32 0)
  %2980 = extractelement <8 x float> %1513, i64 7
  %2981 = fptrunc float %2980 to bfloat
  %2982 = add i32 %2694, 96
  %2983 = add i32 %2982, %70
  %2984 = mul i32 %2983, 2
  %2985 = bitcast bfloat %2981 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2985, ptr addrspace(8) %93, i32 %2984, i32 0, i32 0)
  %2986 = extractelement <8 x float> %1514, i64 0
  %2987 = fptrunc float %2986 to bfloat
  %2988 = add i32 %2631, 112
  %2989 = add i32 %2988, %70
  %2990 = mul i32 %2989, 2
  %2991 = bitcast bfloat %2987 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2991, ptr addrspace(8) %93, i32 %2990, i32 0, i32 0)
  %2992 = extractelement <8 x float> %1514, i64 1
  %2993 = fptrunc float %2992 to bfloat
  %2994 = add i32 %2640, 112
  %2995 = add i32 %2994, %70
  %2996 = mul i32 %2995, 2
  %2997 = bitcast bfloat %2993 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2997, ptr addrspace(8) %93, i32 %2996, i32 0, i32 0)
  %2998 = extractelement <8 x float> %1514, i64 2
  %2999 = fptrunc float %2998 to bfloat
  %3000 = add i32 %2649, 112
  %3001 = add i32 %3000, %70
  %3002 = mul i32 %3001, 2
  %3003 = bitcast bfloat %2999 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3003, ptr addrspace(8) %93, i32 %3002, i32 0, i32 0)
  %3004 = extractelement <8 x float> %1514, i64 3
  %3005 = fptrunc float %3004 to bfloat
  %3006 = add i32 %2658, 112
  %3007 = add i32 %3006, %70
  %3008 = mul i32 %3007, 2
  %3009 = bitcast bfloat %3005 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3009, ptr addrspace(8) %93, i32 %3008, i32 0, i32 0)
  %3010 = extractelement <8 x float> %1514, i64 4
  %3011 = fptrunc float %3010 to bfloat
  %3012 = add i32 %2667, 112
  %3013 = add i32 %3012, %70
  %3014 = mul i32 %3013, 2
  %3015 = bitcast bfloat %3011 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3015, ptr addrspace(8) %93, i32 %3014, i32 0, i32 0)
  %3016 = extractelement <8 x float> %1514, i64 5
  %3017 = fptrunc float %3016 to bfloat
  %3018 = add i32 %2676, 112
  %3019 = add i32 %3018, %70
  %3020 = mul i32 %3019, 2
  %3021 = bitcast bfloat %3017 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3021, ptr addrspace(8) %93, i32 %3020, i32 0, i32 0)
  %3022 = extractelement <8 x float> %1514, i64 6
  %3023 = fptrunc float %3022 to bfloat
  %3024 = add i32 %2685, 112
  %3025 = add i32 %3024, %70
  %3026 = mul i32 %3025, 2
  %3027 = bitcast bfloat %3023 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3027, ptr addrspace(8) %93, i32 %3026, i32 0, i32 0)
  %3028 = extractelement <8 x float> %1514, i64 7
  %3029 = fptrunc float %3028 to bfloat
  %3030 = add i32 %2694, 112
  %3031 = add i32 %3030, %70
  %3032 = mul i32 %3031, 2
  %3033 = bitcast bfloat %3029 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3033, ptr addrspace(8) %93, i32 %3032, i32 0, i32 0)
  %3034 = add i32 %141, %375
  %3035 = extractelement <8 x float> %1515, i64 0
  %3036 = fptrunc float %3035 to bfloat
  %3037 = mul i32 %3034, %19
  %3038 = mul i32 %3037, 128
  %3039 = add i32 %2625, %3038
  %3040 = add i32 %3039, %70
  %3041 = mul i32 %3040, 2
  %3042 = bitcast bfloat %3036 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3042, ptr addrspace(8) %93, i32 %3041, i32 0, i32 0)
  %3043 = add i32 %3034, 1
  %3044 = extractelement <8 x float> %1515, i64 1
  %3045 = fptrunc float %3044 to bfloat
  %3046 = mul i32 %3043, %19
  %3047 = mul i32 %3046, 128
  %3048 = add i32 %2625, %3047
  %3049 = add i32 %3048, %70
  %3050 = mul i32 %3049, 2
  %3051 = bitcast bfloat %3045 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3051, ptr addrspace(8) %93, i32 %3050, i32 0, i32 0)
  %3052 = add i32 %3034, 2
  %3053 = extractelement <8 x float> %1515, i64 2
  %3054 = fptrunc float %3053 to bfloat
  %3055 = mul i32 %3052, %19
  %3056 = mul i32 %3055, 128
  %3057 = add i32 %2625, %3056
  %3058 = add i32 %3057, %70
  %3059 = mul i32 %3058, 2
  %3060 = bitcast bfloat %3054 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3060, ptr addrspace(8) %93, i32 %3059, i32 0, i32 0)
  %3061 = add i32 %3034, 3
  %3062 = extractelement <8 x float> %1515, i64 3
  %3063 = fptrunc float %3062 to bfloat
  %3064 = mul i32 %3061, %19
  %3065 = mul i32 %3064, 128
  %3066 = add i32 %2625, %3065
  %3067 = add i32 %3066, %70
  %3068 = mul i32 %3067, 2
  %3069 = bitcast bfloat %3063 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3069, ptr addrspace(8) %93, i32 %3068, i32 0, i32 0)
  %3070 = add i32 %3034, 4
  %3071 = extractelement <8 x float> %1515, i64 4
  %3072 = fptrunc float %3071 to bfloat
  %3073 = mul i32 %3070, %19
  %3074 = mul i32 %3073, 128
  %3075 = add i32 %2625, %3074
  %3076 = add i32 %3075, %70
  %3077 = mul i32 %3076, 2
  %3078 = bitcast bfloat %3072 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3078, ptr addrspace(8) %93, i32 %3077, i32 0, i32 0)
  %3079 = add i32 %3034, 5
  %3080 = extractelement <8 x float> %1515, i64 5
  %3081 = fptrunc float %3080 to bfloat
  %3082 = mul i32 %3079, %19
  %3083 = mul i32 %3082, 128
  %3084 = add i32 %2625, %3083
  %3085 = add i32 %3084, %70
  %3086 = mul i32 %3085, 2
  %3087 = bitcast bfloat %3081 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3087, ptr addrspace(8) %93, i32 %3086, i32 0, i32 0)
  %3088 = add i32 %3034, 6
  %3089 = extractelement <8 x float> %1515, i64 6
  %3090 = fptrunc float %3089 to bfloat
  %3091 = mul i32 %3088, %19
  %3092 = mul i32 %3091, 128
  %3093 = add i32 %2625, %3092
  %3094 = add i32 %3093, %70
  %3095 = mul i32 %3094, 2
  %3096 = bitcast bfloat %3090 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3096, ptr addrspace(8) %93, i32 %3095, i32 0, i32 0)
  %3097 = add i32 %3034, 7
  %3098 = extractelement <8 x float> %1515, i64 7
  %3099 = fptrunc float %3098 to bfloat
  %3100 = mul i32 %3097, %19
  %3101 = mul i32 %3100, 128
  %3102 = add i32 %2625, %3101
  %3103 = add i32 %3102, %70
  %3104 = mul i32 %3103, 2
  %3105 = bitcast bfloat %3099 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3105, ptr addrspace(8) %93, i32 %3104, i32 0, i32 0)
  %3106 = extractelement <8 x float> %1516, i64 0
  %3107 = fptrunc float %3106 to bfloat
  %3108 = add i32 %3039, 16
  %3109 = add i32 %3108, %70
  %3110 = mul i32 %3109, 2
  %3111 = bitcast bfloat %3107 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3111, ptr addrspace(8) %93, i32 %3110, i32 0, i32 0)
  %3112 = extractelement <8 x float> %1516, i64 1
  %3113 = fptrunc float %3112 to bfloat
  %3114 = add i32 %3048, 16
  %3115 = add i32 %3114, %70
  %3116 = mul i32 %3115, 2
  %3117 = bitcast bfloat %3113 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3117, ptr addrspace(8) %93, i32 %3116, i32 0, i32 0)
  %3118 = extractelement <8 x float> %1516, i64 2
  %3119 = fptrunc float %3118 to bfloat
  %3120 = add i32 %3057, 16
  %3121 = add i32 %3120, %70
  %3122 = mul i32 %3121, 2
  %3123 = bitcast bfloat %3119 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3123, ptr addrspace(8) %93, i32 %3122, i32 0, i32 0)
  %3124 = extractelement <8 x float> %1516, i64 3
  %3125 = fptrunc float %3124 to bfloat
  %3126 = add i32 %3066, 16
  %3127 = add i32 %3126, %70
  %3128 = mul i32 %3127, 2
  %3129 = bitcast bfloat %3125 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3129, ptr addrspace(8) %93, i32 %3128, i32 0, i32 0)
  %3130 = extractelement <8 x float> %1516, i64 4
  %3131 = fptrunc float %3130 to bfloat
  %3132 = add i32 %3075, 16
  %3133 = add i32 %3132, %70
  %3134 = mul i32 %3133, 2
  %3135 = bitcast bfloat %3131 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3135, ptr addrspace(8) %93, i32 %3134, i32 0, i32 0)
  %3136 = extractelement <8 x float> %1516, i64 5
  %3137 = fptrunc float %3136 to bfloat
  %3138 = add i32 %3084, 16
  %3139 = add i32 %3138, %70
  %3140 = mul i32 %3139, 2
  %3141 = bitcast bfloat %3137 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3141, ptr addrspace(8) %93, i32 %3140, i32 0, i32 0)
  %3142 = extractelement <8 x float> %1516, i64 6
  %3143 = fptrunc float %3142 to bfloat
  %3144 = add i32 %3093, 16
  %3145 = add i32 %3144, %70
  %3146 = mul i32 %3145, 2
  %3147 = bitcast bfloat %3143 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3147, ptr addrspace(8) %93, i32 %3146, i32 0, i32 0)
  %3148 = extractelement <8 x float> %1516, i64 7
  %3149 = fptrunc float %3148 to bfloat
  %3150 = add i32 %3102, 16
  %3151 = add i32 %3150, %70
  %3152 = mul i32 %3151, 2
  %3153 = bitcast bfloat %3149 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3153, ptr addrspace(8) %93, i32 %3152, i32 0, i32 0)
  %3154 = extractelement <8 x float> %1517, i64 0
  %3155 = fptrunc float %3154 to bfloat
  %3156 = add i32 %3039, 32
  %3157 = add i32 %3156, %70
  %3158 = mul i32 %3157, 2
  %3159 = bitcast bfloat %3155 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3159, ptr addrspace(8) %93, i32 %3158, i32 0, i32 0)
  %3160 = extractelement <8 x float> %1517, i64 1
  %3161 = fptrunc float %3160 to bfloat
  %3162 = add i32 %3048, 32
  %3163 = add i32 %3162, %70
  %3164 = mul i32 %3163, 2
  %3165 = bitcast bfloat %3161 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3165, ptr addrspace(8) %93, i32 %3164, i32 0, i32 0)
  %3166 = extractelement <8 x float> %1517, i64 2
  %3167 = fptrunc float %3166 to bfloat
  %3168 = add i32 %3057, 32
  %3169 = add i32 %3168, %70
  %3170 = mul i32 %3169, 2
  %3171 = bitcast bfloat %3167 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3171, ptr addrspace(8) %93, i32 %3170, i32 0, i32 0)
  %3172 = extractelement <8 x float> %1517, i64 3
  %3173 = fptrunc float %3172 to bfloat
  %3174 = add i32 %3066, 32
  %3175 = add i32 %3174, %70
  %3176 = mul i32 %3175, 2
  %3177 = bitcast bfloat %3173 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3177, ptr addrspace(8) %93, i32 %3176, i32 0, i32 0)
  %3178 = extractelement <8 x float> %1517, i64 4
  %3179 = fptrunc float %3178 to bfloat
  %3180 = add i32 %3075, 32
  %3181 = add i32 %3180, %70
  %3182 = mul i32 %3181, 2
  %3183 = bitcast bfloat %3179 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3183, ptr addrspace(8) %93, i32 %3182, i32 0, i32 0)
  %3184 = extractelement <8 x float> %1517, i64 5
  %3185 = fptrunc float %3184 to bfloat
  %3186 = add i32 %3084, 32
  %3187 = add i32 %3186, %70
  %3188 = mul i32 %3187, 2
  %3189 = bitcast bfloat %3185 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3189, ptr addrspace(8) %93, i32 %3188, i32 0, i32 0)
  %3190 = extractelement <8 x float> %1517, i64 6
  %3191 = fptrunc float %3190 to bfloat
  %3192 = add i32 %3093, 32
  %3193 = add i32 %3192, %70
  %3194 = mul i32 %3193, 2
  %3195 = bitcast bfloat %3191 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3195, ptr addrspace(8) %93, i32 %3194, i32 0, i32 0)
  %3196 = extractelement <8 x float> %1517, i64 7
  %3197 = fptrunc float %3196 to bfloat
  %3198 = add i32 %3102, 32
  %3199 = add i32 %3198, %70
  %3200 = mul i32 %3199, 2
  %3201 = bitcast bfloat %3197 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3201, ptr addrspace(8) %93, i32 %3200, i32 0, i32 0)
  %3202 = extractelement <8 x float> %1518, i64 0
  %3203 = fptrunc float %3202 to bfloat
  %3204 = add i32 %3039, 48
  %3205 = add i32 %3204, %70
  %3206 = mul i32 %3205, 2
  %3207 = bitcast bfloat %3203 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3207, ptr addrspace(8) %93, i32 %3206, i32 0, i32 0)
  %3208 = extractelement <8 x float> %1518, i64 1
  %3209 = fptrunc float %3208 to bfloat
  %3210 = add i32 %3048, 48
  %3211 = add i32 %3210, %70
  %3212 = mul i32 %3211, 2
  %3213 = bitcast bfloat %3209 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3213, ptr addrspace(8) %93, i32 %3212, i32 0, i32 0)
  %3214 = extractelement <8 x float> %1518, i64 2
  %3215 = fptrunc float %3214 to bfloat
  %3216 = add i32 %3057, 48
  %3217 = add i32 %3216, %70
  %3218 = mul i32 %3217, 2
  %3219 = bitcast bfloat %3215 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3219, ptr addrspace(8) %93, i32 %3218, i32 0, i32 0)
  %3220 = extractelement <8 x float> %1518, i64 3
  %3221 = fptrunc float %3220 to bfloat
  %3222 = add i32 %3066, 48
  %3223 = add i32 %3222, %70
  %3224 = mul i32 %3223, 2
  %3225 = bitcast bfloat %3221 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3225, ptr addrspace(8) %93, i32 %3224, i32 0, i32 0)
  %3226 = extractelement <8 x float> %1518, i64 4
  %3227 = fptrunc float %3226 to bfloat
  %3228 = add i32 %3075, 48
  %3229 = add i32 %3228, %70
  %3230 = mul i32 %3229, 2
  %3231 = bitcast bfloat %3227 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3231, ptr addrspace(8) %93, i32 %3230, i32 0, i32 0)
  %3232 = extractelement <8 x float> %1518, i64 5
  %3233 = fptrunc float %3232 to bfloat
  %3234 = add i32 %3084, 48
  %3235 = add i32 %3234, %70
  %3236 = mul i32 %3235, 2
  %3237 = bitcast bfloat %3233 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3237, ptr addrspace(8) %93, i32 %3236, i32 0, i32 0)
  %3238 = extractelement <8 x float> %1518, i64 6
  %3239 = fptrunc float %3238 to bfloat
  %3240 = add i32 %3093, 48
  %3241 = add i32 %3240, %70
  %3242 = mul i32 %3241, 2
  %3243 = bitcast bfloat %3239 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3243, ptr addrspace(8) %93, i32 %3242, i32 0, i32 0)
  %3244 = extractelement <8 x float> %1518, i64 7
  %3245 = fptrunc float %3244 to bfloat
  %3246 = add i32 %3102, 48
  %3247 = add i32 %3246, %70
  %3248 = mul i32 %3247, 2
  %3249 = bitcast bfloat %3245 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3249, ptr addrspace(8) %93, i32 %3248, i32 0, i32 0)
  %3250 = extractelement <8 x float> %1519, i64 0
  %3251 = fptrunc float %3250 to bfloat
  %3252 = add i32 %3039, 64
  %3253 = add i32 %3252, %70
  %3254 = mul i32 %3253, 2
  %3255 = bitcast bfloat %3251 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3255, ptr addrspace(8) %93, i32 %3254, i32 0, i32 0)
  %3256 = extractelement <8 x float> %1519, i64 1
  %3257 = fptrunc float %3256 to bfloat
  %3258 = add i32 %3048, 64
  %3259 = add i32 %3258, %70
  %3260 = mul i32 %3259, 2
  %3261 = bitcast bfloat %3257 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3261, ptr addrspace(8) %93, i32 %3260, i32 0, i32 0)
  %3262 = extractelement <8 x float> %1519, i64 2
  %3263 = fptrunc float %3262 to bfloat
  %3264 = add i32 %3057, 64
  %3265 = add i32 %3264, %70
  %3266 = mul i32 %3265, 2
  %3267 = bitcast bfloat %3263 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3267, ptr addrspace(8) %93, i32 %3266, i32 0, i32 0)
  %3268 = extractelement <8 x float> %1519, i64 3
  %3269 = fptrunc float %3268 to bfloat
  %3270 = add i32 %3066, 64
  %3271 = add i32 %3270, %70
  %3272 = mul i32 %3271, 2
  %3273 = bitcast bfloat %3269 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3273, ptr addrspace(8) %93, i32 %3272, i32 0, i32 0)
  %3274 = extractelement <8 x float> %1519, i64 4
  %3275 = fptrunc float %3274 to bfloat
  %3276 = add i32 %3075, 64
  %3277 = add i32 %3276, %70
  %3278 = mul i32 %3277, 2
  %3279 = bitcast bfloat %3275 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3279, ptr addrspace(8) %93, i32 %3278, i32 0, i32 0)
  %3280 = extractelement <8 x float> %1519, i64 5
  %3281 = fptrunc float %3280 to bfloat
  %3282 = add i32 %3084, 64
  %3283 = add i32 %3282, %70
  %3284 = mul i32 %3283, 2
  %3285 = bitcast bfloat %3281 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3285, ptr addrspace(8) %93, i32 %3284, i32 0, i32 0)
  %3286 = extractelement <8 x float> %1519, i64 6
  %3287 = fptrunc float %3286 to bfloat
  %3288 = add i32 %3093, 64
  %3289 = add i32 %3288, %70
  %3290 = mul i32 %3289, 2
  %3291 = bitcast bfloat %3287 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3291, ptr addrspace(8) %93, i32 %3290, i32 0, i32 0)
  %3292 = extractelement <8 x float> %1519, i64 7
  %3293 = fptrunc float %3292 to bfloat
  %3294 = add i32 %3102, 64
  %3295 = add i32 %3294, %70
  %3296 = mul i32 %3295, 2
  %3297 = bitcast bfloat %3293 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3297, ptr addrspace(8) %93, i32 %3296, i32 0, i32 0)
  %3298 = extractelement <8 x float> %1520, i64 0
  %3299 = fptrunc float %3298 to bfloat
  %3300 = add i32 %3039, 80
  %3301 = add i32 %3300, %70
  %3302 = mul i32 %3301, 2
  %3303 = bitcast bfloat %3299 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3303, ptr addrspace(8) %93, i32 %3302, i32 0, i32 0)
  %3304 = extractelement <8 x float> %1520, i64 1
  %3305 = fptrunc float %3304 to bfloat
  %3306 = add i32 %3048, 80
  %3307 = add i32 %3306, %70
  %3308 = mul i32 %3307, 2
  %3309 = bitcast bfloat %3305 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3309, ptr addrspace(8) %93, i32 %3308, i32 0, i32 0)
  %3310 = extractelement <8 x float> %1520, i64 2
  %3311 = fptrunc float %3310 to bfloat
  %3312 = add i32 %3057, 80
  %3313 = add i32 %3312, %70
  %3314 = mul i32 %3313, 2
  %3315 = bitcast bfloat %3311 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3315, ptr addrspace(8) %93, i32 %3314, i32 0, i32 0)
  %3316 = extractelement <8 x float> %1520, i64 3
  %3317 = fptrunc float %3316 to bfloat
  %3318 = add i32 %3066, 80
  %3319 = add i32 %3318, %70
  %3320 = mul i32 %3319, 2
  %3321 = bitcast bfloat %3317 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3321, ptr addrspace(8) %93, i32 %3320, i32 0, i32 0)
  %3322 = extractelement <8 x float> %1520, i64 4
  %3323 = fptrunc float %3322 to bfloat
  %3324 = add i32 %3075, 80
  %3325 = add i32 %3324, %70
  %3326 = mul i32 %3325, 2
  %3327 = bitcast bfloat %3323 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3327, ptr addrspace(8) %93, i32 %3326, i32 0, i32 0)
  %3328 = extractelement <8 x float> %1520, i64 5
  %3329 = fptrunc float %3328 to bfloat
  %3330 = add i32 %3084, 80
  %3331 = add i32 %3330, %70
  %3332 = mul i32 %3331, 2
  %3333 = bitcast bfloat %3329 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3333, ptr addrspace(8) %93, i32 %3332, i32 0, i32 0)
  %3334 = extractelement <8 x float> %1520, i64 6
  %3335 = fptrunc float %3334 to bfloat
  %3336 = add i32 %3093, 80
  %3337 = add i32 %3336, %70
  %3338 = mul i32 %3337, 2
  %3339 = bitcast bfloat %3335 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3339, ptr addrspace(8) %93, i32 %3338, i32 0, i32 0)
  %3340 = extractelement <8 x float> %1520, i64 7
  %3341 = fptrunc float %3340 to bfloat
  %3342 = add i32 %3102, 80
  %3343 = add i32 %3342, %70
  %3344 = mul i32 %3343, 2
  %3345 = bitcast bfloat %3341 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3345, ptr addrspace(8) %93, i32 %3344, i32 0, i32 0)
  %3346 = extractelement <8 x float> %1521, i64 0
  %3347 = fptrunc float %3346 to bfloat
  %3348 = add i32 %3039, 96
  %3349 = add i32 %3348, %70
  %3350 = mul i32 %3349, 2
  %3351 = bitcast bfloat %3347 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3351, ptr addrspace(8) %93, i32 %3350, i32 0, i32 0)
  %3352 = extractelement <8 x float> %1521, i64 1
  %3353 = fptrunc float %3352 to bfloat
  %3354 = add i32 %3048, 96
  %3355 = add i32 %3354, %70
  %3356 = mul i32 %3355, 2
  %3357 = bitcast bfloat %3353 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3357, ptr addrspace(8) %93, i32 %3356, i32 0, i32 0)
  %3358 = extractelement <8 x float> %1521, i64 2
  %3359 = fptrunc float %3358 to bfloat
  %3360 = add i32 %3057, 96
  %3361 = add i32 %3360, %70
  %3362 = mul i32 %3361, 2
  %3363 = bitcast bfloat %3359 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3363, ptr addrspace(8) %93, i32 %3362, i32 0, i32 0)
  %3364 = extractelement <8 x float> %1521, i64 3
  %3365 = fptrunc float %3364 to bfloat
  %3366 = add i32 %3066, 96
  %3367 = add i32 %3366, %70
  %3368 = mul i32 %3367, 2
  %3369 = bitcast bfloat %3365 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3369, ptr addrspace(8) %93, i32 %3368, i32 0, i32 0)
  %3370 = extractelement <8 x float> %1521, i64 4
  %3371 = fptrunc float %3370 to bfloat
  %3372 = add i32 %3075, 96
  %3373 = add i32 %3372, %70
  %3374 = mul i32 %3373, 2
  %3375 = bitcast bfloat %3371 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3375, ptr addrspace(8) %93, i32 %3374, i32 0, i32 0)
  %3376 = extractelement <8 x float> %1521, i64 5
  %3377 = fptrunc float %3376 to bfloat
  %3378 = add i32 %3084, 96
  %3379 = add i32 %3378, %70
  %3380 = mul i32 %3379, 2
  %3381 = bitcast bfloat %3377 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3381, ptr addrspace(8) %93, i32 %3380, i32 0, i32 0)
  %3382 = extractelement <8 x float> %1521, i64 6
  %3383 = fptrunc float %3382 to bfloat
  %3384 = add i32 %3093, 96
  %3385 = add i32 %3384, %70
  %3386 = mul i32 %3385, 2
  %3387 = bitcast bfloat %3383 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3387, ptr addrspace(8) %93, i32 %3386, i32 0, i32 0)
  %3388 = extractelement <8 x float> %1521, i64 7
  %3389 = fptrunc float %3388 to bfloat
  %3390 = add i32 %3102, 96
  %3391 = add i32 %3390, %70
  %3392 = mul i32 %3391, 2
  %3393 = bitcast bfloat %3389 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3393, ptr addrspace(8) %93, i32 %3392, i32 0, i32 0)
  %3394 = extractelement <8 x float> %1522, i64 0
  %3395 = fptrunc float %3394 to bfloat
  %3396 = add i32 %3039, 112
  %3397 = add i32 %3396, %70
  %3398 = mul i32 %3397, 2
  %3399 = bitcast bfloat %3395 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3399, ptr addrspace(8) %93, i32 %3398, i32 0, i32 0)
  %3400 = extractelement <8 x float> %1522, i64 1
  %3401 = fptrunc float %3400 to bfloat
  %3402 = add i32 %3048, 112
  %3403 = add i32 %3402, %70
  %3404 = mul i32 %3403, 2
  %3405 = bitcast bfloat %3401 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3405, ptr addrspace(8) %93, i32 %3404, i32 0, i32 0)
  %3406 = extractelement <8 x float> %1522, i64 2
  %3407 = fptrunc float %3406 to bfloat
  %3408 = add i32 %3057, 112
  %3409 = add i32 %3408, %70
  %3410 = mul i32 %3409, 2
  %3411 = bitcast bfloat %3407 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3411, ptr addrspace(8) %93, i32 %3410, i32 0, i32 0)
  %3412 = extractelement <8 x float> %1522, i64 3
  %3413 = fptrunc float %3412 to bfloat
  %3414 = add i32 %3066, 112
  %3415 = add i32 %3414, %70
  %3416 = mul i32 %3415, 2
  %3417 = bitcast bfloat %3413 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3417, ptr addrspace(8) %93, i32 %3416, i32 0, i32 0)
  %3418 = extractelement <8 x float> %1522, i64 4
  %3419 = fptrunc float %3418 to bfloat
  %3420 = add i32 %3075, 112
  %3421 = add i32 %3420, %70
  %3422 = mul i32 %3421, 2
  %3423 = bitcast bfloat %3419 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3423, ptr addrspace(8) %93, i32 %3422, i32 0, i32 0)
  %3424 = extractelement <8 x float> %1522, i64 5
  %3425 = fptrunc float %3424 to bfloat
  %3426 = add i32 %3084, 112
  %3427 = add i32 %3426, %70
  %3428 = mul i32 %3427, 2
  %3429 = bitcast bfloat %3425 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3429, ptr addrspace(8) %93, i32 %3428, i32 0, i32 0)
  %3430 = extractelement <8 x float> %1522, i64 6
  %3431 = fptrunc float %3430 to bfloat
  %3432 = add i32 %3093, 112
  %3433 = add i32 %3432, %70
  %3434 = mul i32 %3433, 2
  %3435 = bitcast bfloat %3431 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3435, ptr addrspace(8) %93, i32 %3434, i32 0, i32 0)
  %3436 = extractelement <8 x float> %1522, i64 7
  %3437 = fptrunc float %3436 to bfloat
  %3438 = add i32 %3102, 112
  %3439 = add i32 %3438, %70
  %3440 = mul i32 %3439, 2
  %3441 = bitcast bfloat %3437 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3441, ptr addrspace(8) %93, i32 %3440, i32 0, i32 0)
  %3442 = add i32 %181, %375
  %3443 = extractelement <8 x float> %1523, i64 0
  %3444 = fptrunc float %3443 to bfloat
  %3445 = mul i32 %3442, %19
  %3446 = mul i32 %3445, 128
  %3447 = add i32 %2625, %3446
  %3448 = add i32 %3447, %70
  %3449 = mul i32 %3448, 2
  %3450 = bitcast bfloat %3444 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3450, ptr addrspace(8) %93, i32 %3449, i32 0, i32 0)
  %3451 = add i32 %3442, 1
  %3452 = extractelement <8 x float> %1523, i64 1
  %3453 = fptrunc float %3452 to bfloat
  %3454 = mul i32 %3451, %19
  %3455 = mul i32 %3454, 128
  %3456 = add i32 %2625, %3455
  %3457 = add i32 %3456, %70
  %3458 = mul i32 %3457, 2
  %3459 = bitcast bfloat %3453 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3459, ptr addrspace(8) %93, i32 %3458, i32 0, i32 0)
  %3460 = add i32 %3442, 2
  %3461 = extractelement <8 x float> %1523, i64 2
  %3462 = fptrunc float %3461 to bfloat
  %3463 = mul i32 %3460, %19
  %3464 = mul i32 %3463, 128
  %3465 = add i32 %2625, %3464
  %3466 = add i32 %3465, %70
  %3467 = mul i32 %3466, 2
  %3468 = bitcast bfloat %3462 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3468, ptr addrspace(8) %93, i32 %3467, i32 0, i32 0)
  %3469 = add i32 %3442, 3
  %3470 = extractelement <8 x float> %1523, i64 3
  %3471 = fptrunc float %3470 to bfloat
  %3472 = mul i32 %3469, %19
  %3473 = mul i32 %3472, 128
  %3474 = add i32 %2625, %3473
  %3475 = add i32 %3474, %70
  %3476 = mul i32 %3475, 2
  %3477 = bitcast bfloat %3471 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3477, ptr addrspace(8) %93, i32 %3476, i32 0, i32 0)
  %3478 = add i32 %3442, 4
  %3479 = extractelement <8 x float> %1523, i64 4
  %3480 = fptrunc float %3479 to bfloat
  %3481 = mul i32 %3478, %19
  %3482 = mul i32 %3481, 128
  %3483 = add i32 %2625, %3482
  %3484 = add i32 %3483, %70
  %3485 = mul i32 %3484, 2
  %3486 = bitcast bfloat %3480 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3486, ptr addrspace(8) %93, i32 %3485, i32 0, i32 0)
  %3487 = add i32 %3442, 5
  %3488 = extractelement <8 x float> %1523, i64 5
  %3489 = fptrunc float %3488 to bfloat
  %3490 = mul i32 %3487, %19
  %3491 = mul i32 %3490, 128
  %3492 = add i32 %2625, %3491
  %3493 = add i32 %3492, %70
  %3494 = mul i32 %3493, 2
  %3495 = bitcast bfloat %3489 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3495, ptr addrspace(8) %93, i32 %3494, i32 0, i32 0)
  %3496 = add i32 %3442, 6
  %3497 = extractelement <8 x float> %1523, i64 6
  %3498 = fptrunc float %3497 to bfloat
  %3499 = mul i32 %3496, %19
  %3500 = mul i32 %3499, 128
  %3501 = add i32 %2625, %3500
  %3502 = add i32 %3501, %70
  %3503 = mul i32 %3502, 2
  %3504 = bitcast bfloat %3498 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3504, ptr addrspace(8) %93, i32 %3503, i32 0, i32 0)
  %3505 = add i32 %3442, 7
  %3506 = extractelement <8 x float> %1523, i64 7
  %3507 = fptrunc float %3506 to bfloat
  %3508 = mul i32 %3505, %19
  %3509 = mul i32 %3508, 128
  %3510 = add i32 %2625, %3509
  %3511 = add i32 %3510, %70
  %3512 = mul i32 %3511, 2
  %3513 = bitcast bfloat %3507 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3513, ptr addrspace(8) %93, i32 %3512, i32 0, i32 0)
  %3514 = extractelement <8 x float> %1524, i64 0
  %3515 = fptrunc float %3514 to bfloat
  %3516 = add i32 %3447, 16
  %3517 = add i32 %3516, %70
  %3518 = mul i32 %3517, 2
  %3519 = bitcast bfloat %3515 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3519, ptr addrspace(8) %93, i32 %3518, i32 0, i32 0)
  %3520 = extractelement <8 x float> %1524, i64 1
  %3521 = fptrunc float %3520 to bfloat
  %3522 = add i32 %3456, 16
  %3523 = add i32 %3522, %70
  %3524 = mul i32 %3523, 2
  %3525 = bitcast bfloat %3521 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3525, ptr addrspace(8) %93, i32 %3524, i32 0, i32 0)
  %3526 = extractelement <8 x float> %1524, i64 2
  %3527 = fptrunc float %3526 to bfloat
  %3528 = add i32 %3465, 16
  %3529 = add i32 %3528, %70
  %3530 = mul i32 %3529, 2
  %3531 = bitcast bfloat %3527 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3531, ptr addrspace(8) %93, i32 %3530, i32 0, i32 0)
  %3532 = extractelement <8 x float> %1524, i64 3
  %3533 = fptrunc float %3532 to bfloat
  %3534 = add i32 %3474, 16
  %3535 = add i32 %3534, %70
  %3536 = mul i32 %3535, 2
  %3537 = bitcast bfloat %3533 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3537, ptr addrspace(8) %93, i32 %3536, i32 0, i32 0)
  %3538 = extractelement <8 x float> %1524, i64 4
  %3539 = fptrunc float %3538 to bfloat
  %3540 = add i32 %3483, 16
  %3541 = add i32 %3540, %70
  %3542 = mul i32 %3541, 2
  %3543 = bitcast bfloat %3539 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3543, ptr addrspace(8) %93, i32 %3542, i32 0, i32 0)
  %3544 = extractelement <8 x float> %1524, i64 5
  %3545 = fptrunc float %3544 to bfloat
  %3546 = add i32 %3492, 16
  %3547 = add i32 %3546, %70
  %3548 = mul i32 %3547, 2
  %3549 = bitcast bfloat %3545 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3549, ptr addrspace(8) %93, i32 %3548, i32 0, i32 0)
  %3550 = extractelement <8 x float> %1524, i64 6
  %3551 = fptrunc float %3550 to bfloat
  %3552 = add i32 %3501, 16
  %3553 = add i32 %3552, %70
  %3554 = mul i32 %3553, 2
  %3555 = bitcast bfloat %3551 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3555, ptr addrspace(8) %93, i32 %3554, i32 0, i32 0)
  %3556 = extractelement <8 x float> %1524, i64 7
  %3557 = fptrunc float %3556 to bfloat
  %3558 = add i32 %3510, 16
  %3559 = add i32 %3558, %70
  %3560 = mul i32 %3559, 2
  %3561 = bitcast bfloat %3557 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3561, ptr addrspace(8) %93, i32 %3560, i32 0, i32 0)
  %3562 = extractelement <8 x float> %1525, i64 0
  %3563 = fptrunc float %3562 to bfloat
  %3564 = add i32 %3447, 32
  %3565 = add i32 %3564, %70
  %3566 = mul i32 %3565, 2
  %3567 = bitcast bfloat %3563 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3567, ptr addrspace(8) %93, i32 %3566, i32 0, i32 0)
  %3568 = extractelement <8 x float> %1525, i64 1
  %3569 = fptrunc float %3568 to bfloat
  %3570 = add i32 %3456, 32
  %3571 = add i32 %3570, %70
  %3572 = mul i32 %3571, 2
  %3573 = bitcast bfloat %3569 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3573, ptr addrspace(8) %93, i32 %3572, i32 0, i32 0)
  %3574 = extractelement <8 x float> %1525, i64 2
  %3575 = fptrunc float %3574 to bfloat
  %3576 = add i32 %3465, 32
  %3577 = add i32 %3576, %70
  %3578 = mul i32 %3577, 2
  %3579 = bitcast bfloat %3575 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3579, ptr addrspace(8) %93, i32 %3578, i32 0, i32 0)
  %3580 = extractelement <8 x float> %1525, i64 3
  %3581 = fptrunc float %3580 to bfloat
  %3582 = add i32 %3474, 32
  %3583 = add i32 %3582, %70
  %3584 = mul i32 %3583, 2
  %3585 = bitcast bfloat %3581 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3585, ptr addrspace(8) %93, i32 %3584, i32 0, i32 0)
  %3586 = extractelement <8 x float> %1525, i64 4
  %3587 = fptrunc float %3586 to bfloat
  %3588 = add i32 %3483, 32
  %3589 = add i32 %3588, %70
  %3590 = mul i32 %3589, 2
  %3591 = bitcast bfloat %3587 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3591, ptr addrspace(8) %93, i32 %3590, i32 0, i32 0)
  %3592 = extractelement <8 x float> %1525, i64 5
  %3593 = fptrunc float %3592 to bfloat
  %3594 = add i32 %3492, 32
  %3595 = add i32 %3594, %70
  %3596 = mul i32 %3595, 2
  %3597 = bitcast bfloat %3593 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3597, ptr addrspace(8) %93, i32 %3596, i32 0, i32 0)
  %3598 = extractelement <8 x float> %1525, i64 6
  %3599 = fptrunc float %3598 to bfloat
  %3600 = add i32 %3501, 32
  %3601 = add i32 %3600, %70
  %3602 = mul i32 %3601, 2
  %3603 = bitcast bfloat %3599 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3603, ptr addrspace(8) %93, i32 %3602, i32 0, i32 0)
  %3604 = extractelement <8 x float> %1525, i64 7
  %3605 = fptrunc float %3604 to bfloat
  %3606 = add i32 %3510, 32
  %3607 = add i32 %3606, %70
  %3608 = mul i32 %3607, 2
  %3609 = bitcast bfloat %3605 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3609, ptr addrspace(8) %93, i32 %3608, i32 0, i32 0)
  %3610 = extractelement <8 x float> %1526, i64 0
  %3611 = fptrunc float %3610 to bfloat
  %3612 = add i32 %3447, 48
  %3613 = add i32 %3612, %70
  %3614 = mul i32 %3613, 2
  %3615 = bitcast bfloat %3611 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3615, ptr addrspace(8) %93, i32 %3614, i32 0, i32 0)
  %3616 = extractelement <8 x float> %1526, i64 1
  %3617 = fptrunc float %3616 to bfloat
  %3618 = add i32 %3456, 48
  %3619 = add i32 %3618, %70
  %3620 = mul i32 %3619, 2
  %3621 = bitcast bfloat %3617 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3621, ptr addrspace(8) %93, i32 %3620, i32 0, i32 0)
  %3622 = extractelement <8 x float> %1526, i64 2
  %3623 = fptrunc float %3622 to bfloat
  %3624 = add i32 %3465, 48
  %3625 = add i32 %3624, %70
  %3626 = mul i32 %3625, 2
  %3627 = bitcast bfloat %3623 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3627, ptr addrspace(8) %93, i32 %3626, i32 0, i32 0)
  %3628 = extractelement <8 x float> %1526, i64 3
  %3629 = fptrunc float %3628 to bfloat
  %3630 = add i32 %3474, 48
  %3631 = add i32 %3630, %70
  %3632 = mul i32 %3631, 2
  %3633 = bitcast bfloat %3629 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3633, ptr addrspace(8) %93, i32 %3632, i32 0, i32 0)
  %3634 = extractelement <8 x float> %1526, i64 4
  %3635 = fptrunc float %3634 to bfloat
  %3636 = add i32 %3483, 48
  %3637 = add i32 %3636, %70
  %3638 = mul i32 %3637, 2
  %3639 = bitcast bfloat %3635 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3639, ptr addrspace(8) %93, i32 %3638, i32 0, i32 0)
  %3640 = extractelement <8 x float> %1526, i64 5
  %3641 = fptrunc float %3640 to bfloat
  %3642 = add i32 %3492, 48
  %3643 = add i32 %3642, %70
  %3644 = mul i32 %3643, 2
  %3645 = bitcast bfloat %3641 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3645, ptr addrspace(8) %93, i32 %3644, i32 0, i32 0)
  %3646 = extractelement <8 x float> %1526, i64 6
  %3647 = fptrunc float %3646 to bfloat
  %3648 = add i32 %3501, 48
  %3649 = add i32 %3648, %70
  %3650 = mul i32 %3649, 2
  %3651 = bitcast bfloat %3647 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3651, ptr addrspace(8) %93, i32 %3650, i32 0, i32 0)
  %3652 = extractelement <8 x float> %1526, i64 7
  %3653 = fptrunc float %3652 to bfloat
  %3654 = add i32 %3510, 48
  %3655 = add i32 %3654, %70
  %3656 = mul i32 %3655, 2
  %3657 = bitcast bfloat %3653 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3657, ptr addrspace(8) %93, i32 %3656, i32 0, i32 0)
  %3658 = extractelement <8 x float> %1527, i64 0
  %3659 = fptrunc float %3658 to bfloat
  %3660 = add i32 %3447, 64
  %3661 = add i32 %3660, %70
  %3662 = mul i32 %3661, 2
  %3663 = bitcast bfloat %3659 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3663, ptr addrspace(8) %93, i32 %3662, i32 0, i32 0)
  %3664 = extractelement <8 x float> %1527, i64 1
  %3665 = fptrunc float %3664 to bfloat
  %3666 = add i32 %3456, 64
  %3667 = add i32 %3666, %70
  %3668 = mul i32 %3667, 2
  %3669 = bitcast bfloat %3665 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3669, ptr addrspace(8) %93, i32 %3668, i32 0, i32 0)
  %3670 = extractelement <8 x float> %1527, i64 2
  %3671 = fptrunc float %3670 to bfloat
  %3672 = add i32 %3465, 64
  %3673 = add i32 %3672, %70
  %3674 = mul i32 %3673, 2
  %3675 = bitcast bfloat %3671 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3675, ptr addrspace(8) %93, i32 %3674, i32 0, i32 0)
  %3676 = extractelement <8 x float> %1527, i64 3
  %3677 = fptrunc float %3676 to bfloat
  %3678 = add i32 %3474, 64
  %3679 = add i32 %3678, %70
  %3680 = mul i32 %3679, 2
  %3681 = bitcast bfloat %3677 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3681, ptr addrspace(8) %93, i32 %3680, i32 0, i32 0)
  %3682 = extractelement <8 x float> %1527, i64 4
  %3683 = fptrunc float %3682 to bfloat
  %3684 = add i32 %3483, 64
  %3685 = add i32 %3684, %70
  %3686 = mul i32 %3685, 2
  %3687 = bitcast bfloat %3683 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3687, ptr addrspace(8) %93, i32 %3686, i32 0, i32 0)
  %3688 = extractelement <8 x float> %1527, i64 5
  %3689 = fptrunc float %3688 to bfloat
  %3690 = add i32 %3492, 64
  %3691 = add i32 %3690, %70
  %3692 = mul i32 %3691, 2
  %3693 = bitcast bfloat %3689 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3693, ptr addrspace(8) %93, i32 %3692, i32 0, i32 0)
  %3694 = extractelement <8 x float> %1527, i64 6
  %3695 = fptrunc float %3694 to bfloat
  %3696 = add i32 %3501, 64
  %3697 = add i32 %3696, %70
  %3698 = mul i32 %3697, 2
  %3699 = bitcast bfloat %3695 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3699, ptr addrspace(8) %93, i32 %3698, i32 0, i32 0)
  %3700 = extractelement <8 x float> %1527, i64 7
  %3701 = fptrunc float %3700 to bfloat
  %3702 = add i32 %3510, 64
  %3703 = add i32 %3702, %70
  %3704 = mul i32 %3703, 2
  %3705 = bitcast bfloat %3701 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3705, ptr addrspace(8) %93, i32 %3704, i32 0, i32 0)
  %3706 = extractelement <8 x float> %1528, i64 0
  %3707 = fptrunc float %3706 to bfloat
  %3708 = add i32 %3447, 80
  %3709 = add i32 %3708, %70
  %3710 = mul i32 %3709, 2
  %3711 = bitcast bfloat %3707 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3711, ptr addrspace(8) %93, i32 %3710, i32 0, i32 0)
  %3712 = extractelement <8 x float> %1528, i64 1
  %3713 = fptrunc float %3712 to bfloat
  %3714 = add i32 %3456, 80
  %3715 = add i32 %3714, %70
  %3716 = mul i32 %3715, 2
  %3717 = bitcast bfloat %3713 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3717, ptr addrspace(8) %93, i32 %3716, i32 0, i32 0)
  %3718 = extractelement <8 x float> %1528, i64 2
  %3719 = fptrunc float %3718 to bfloat
  %3720 = add i32 %3465, 80
  %3721 = add i32 %3720, %70
  %3722 = mul i32 %3721, 2
  %3723 = bitcast bfloat %3719 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3723, ptr addrspace(8) %93, i32 %3722, i32 0, i32 0)
  %3724 = extractelement <8 x float> %1528, i64 3
  %3725 = fptrunc float %3724 to bfloat
  %3726 = add i32 %3474, 80
  %3727 = add i32 %3726, %70
  %3728 = mul i32 %3727, 2
  %3729 = bitcast bfloat %3725 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3729, ptr addrspace(8) %93, i32 %3728, i32 0, i32 0)
  %3730 = extractelement <8 x float> %1528, i64 4
  %3731 = fptrunc float %3730 to bfloat
  %3732 = add i32 %3483, 80
  %3733 = add i32 %3732, %70
  %3734 = mul i32 %3733, 2
  %3735 = bitcast bfloat %3731 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3735, ptr addrspace(8) %93, i32 %3734, i32 0, i32 0)
  %3736 = extractelement <8 x float> %1528, i64 5
  %3737 = fptrunc float %3736 to bfloat
  %3738 = add i32 %3492, 80
  %3739 = add i32 %3738, %70
  %3740 = mul i32 %3739, 2
  %3741 = bitcast bfloat %3737 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3741, ptr addrspace(8) %93, i32 %3740, i32 0, i32 0)
  %3742 = extractelement <8 x float> %1528, i64 6
  %3743 = fptrunc float %3742 to bfloat
  %3744 = add i32 %3501, 80
  %3745 = add i32 %3744, %70
  %3746 = mul i32 %3745, 2
  %3747 = bitcast bfloat %3743 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3747, ptr addrspace(8) %93, i32 %3746, i32 0, i32 0)
  %3748 = extractelement <8 x float> %1528, i64 7
  %3749 = fptrunc float %3748 to bfloat
  %3750 = add i32 %3510, 80
  %3751 = add i32 %3750, %70
  %3752 = mul i32 %3751, 2
  %3753 = bitcast bfloat %3749 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3753, ptr addrspace(8) %93, i32 %3752, i32 0, i32 0)
  %3754 = extractelement <8 x float> %1529, i64 0
  %3755 = fptrunc float %3754 to bfloat
  %3756 = add i32 %3447, 96
  %3757 = add i32 %3756, %70
  %3758 = mul i32 %3757, 2
  %3759 = bitcast bfloat %3755 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3759, ptr addrspace(8) %93, i32 %3758, i32 0, i32 0)
  %3760 = extractelement <8 x float> %1529, i64 1
  %3761 = fptrunc float %3760 to bfloat
  %3762 = add i32 %3456, 96
  %3763 = add i32 %3762, %70
  %3764 = mul i32 %3763, 2
  %3765 = bitcast bfloat %3761 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3765, ptr addrspace(8) %93, i32 %3764, i32 0, i32 0)
  %3766 = extractelement <8 x float> %1529, i64 2
  %3767 = fptrunc float %3766 to bfloat
  %3768 = add i32 %3465, 96
  %3769 = add i32 %3768, %70
  %3770 = mul i32 %3769, 2
  %3771 = bitcast bfloat %3767 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3771, ptr addrspace(8) %93, i32 %3770, i32 0, i32 0)
  %3772 = extractelement <8 x float> %1529, i64 3
  %3773 = fptrunc float %3772 to bfloat
  %3774 = add i32 %3474, 96
  %3775 = add i32 %3774, %70
  %3776 = mul i32 %3775, 2
  %3777 = bitcast bfloat %3773 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3777, ptr addrspace(8) %93, i32 %3776, i32 0, i32 0)
  %3778 = extractelement <8 x float> %1529, i64 4
  %3779 = fptrunc float %3778 to bfloat
  %3780 = add i32 %3483, 96
  %3781 = add i32 %3780, %70
  %3782 = mul i32 %3781, 2
  %3783 = bitcast bfloat %3779 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3783, ptr addrspace(8) %93, i32 %3782, i32 0, i32 0)
  %3784 = extractelement <8 x float> %1529, i64 5
  %3785 = fptrunc float %3784 to bfloat
  %3786 = add i32 %3492, 96
  %3787 = add i32 %3786, %70
  %3788 = mul i32 %3787, 2
  %3789 = bitcast bfloat %3785 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3789, ptr addrspace(8) %93, i32 %3788, i32 0, i32 0)
  %3790 = extractelement <8 x float> %1529, i64 6
  %3791 = fptrunc float %3790 to bfloat
  %3792 = add i32 %3501, 96
  %3793 = add i32 %3792, %70
  %3794 = mul i32 %3793, 2
  %3795 = bitcast bfloat %3791 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3795, ptr addrspace(8) %93, i32 %3794, i32 0, i32 0)
  %3796 = extractelement <8 x float> %1529, i64 7
  %3797 = fptrunc float %3796 to bfloat
  %3798 = add i32 %3510, 96
  %3799 = add i32 %3798, %70
  %3800 = mul i32 %3799, 2
  %3801 = bitcast bfloat %3797 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3801, ptr addrspace(8) %93, i32 %3800, i32 0, i32 0)
  %3802 = extractelement <8 x float> %1530, i64 0
  %3803 = fptrunc float %3802 to bfloat
  %3804 = add i32 %3447, 112
  %3805 = add i32 %3804, %70
  %3806 = mul i32 %3805, 2
  %3807 = bitcast bfloat %3803 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3807, ptr addrspace(8) %93, i32 %3806, i32 0, i32 0)
  %3808 = extractelement <8 x float> %1530, i64 1
  %3809 = fptrunc float %3808 to bfloat
  %3810 = add i32 %3456, 112
  %3811 = add i32 %3810, %70
  %3812 = mul i32 %3811, 2
  %3813 = bitcast bfloat %3809 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3813, ptr addrspace(8) %93, i32 %3812, i32 0, i32 0)
  %3814 = extractelement <8 x float> %1530, i64 2
  %3815 = fptrunc float %3814 to bfloat
  %3816 = add i32 %3465, 112
  %3817 = add i32 %3816, %70
  %3818 = mul i32 %3817, 2
  %3819 = bitcast bfloat %3815 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3819, ptr addrspace(8) %93, i32 %3818, i32 0, i32 0)
  %3820 = extractelement <8 x float> %1530, i64 3
  %3821 = fptrunc float %3820 to bfloat
  %3822 = add i32 %3474, 112
  %3823 = add i32 %3822, %70
  %3824 = mul i32 %3823, 2
  %3825 = bitcast bfloat %3821 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3825, ptr addrspace(8) %93, i32 %3824, i32 0, i32 0)
  %3826 = extractelement <8 x float> %1530, i64 4
  %3827 = fptrunc float %3826 to bfloat
  %3828 = add i32 %3483, 112
  %3829 = add i32 %3828, %70
  %3830 = mul i32 %3829, 2
  %3831 = bitcast bfloat %3827 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3831, ptr addrspace(8) %93, i32 %3830, i32 0, i32 0)
  %3832 = extractelement <8 x float> %1530, i64 5
  %3833 = fptrunc float %3832 to bfloat
  %3834 = add i32 %3492, 112
  %3835 = add i32 %3834, %70
  %3836 = mul i32 %3835, 2
  %3837 = bitcast bfloat %3833 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3837, ptr addrspace(8) %93, i32 %3836, i32 0, i32 0)
  %3838 = extractelement <8 x float> %1530, i64 6
  %3839 = fptrunc float %3838 to bfloat
  %3840 = add i32 %3501, 112
  %3841 = add i32 %3840, %70
  %3842 = mul i32 %3841, 2
  %3843 = bitcast bfloat %3839 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3843, ptr addrspace(8) %93, i32 %3842, i32 0, i32 0)
  %3844 = extractelement <8 x float> %1530, i64 7
  %3845 = fptrunc float %3844 to bfloat
  %3846 = add i32 %3510, 112
  %3847 = add i32 %3846, %70
  %3848 = mul i32 %3847, 2
  %3849 = bitcast bfloat %3845 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3849, ptr addrspace(8) %93, i32 %3848, i32 0, i32 0)
  %3850 = add i32 %221, %375
  %3851 = extractelement <8 x float> %1531, i64 0
  %3852 = fptrunc float %3851 to bfloat
  %3853 = mul i32 %3850, %19
  %3854 = mul i32 %3853, 128
  %3855 = add i32 %2625, %3854
  %3856 = add i32 %3855, %70
  %3857 = mul i32 %3856, 2
  %3858 = bitcast bfloat %3852 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3858, ptr addrspace(8) %93, i32 %3857, i32 0, i32 0)
  %3859 = add i32 %3850, 1
  %3860 = extractelement <8 x float> %1531, i64 1
  %3861 = fptrunc float %3860 to bfloat
  %3862 = mul i32 %3859, %19
  %3863 = mul i32 %3862, 128
  %3864 = add i32 %2625, %3863
  %3865 = add i32 %3864, %70
  %3866 = mul i32 %3865, 2
  %3867 = bitcast bfloat %3861 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3867, ptr addrspace(8) %93, i32 %3866, i32 0, i32 0)
  %3868 = add i32 %3850, 2
  %3869 = extractelement <8 x float> %1531, i64 2
  %3870 = fptrunc float %3869 to bfloat
  %3871 = mul i32 %3868, %19
  %3872 = mul i32 %3871, 128
  %3873 = add i32 %2625, %3872
  %3874 = add i32 %3873, %70
  %3875 = mul i32 %3874, 2
  %3876 = bitcast bfloat %3870 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3876, ptr addrspace(8) %93, i32 %3875, i32 0, i32 0)
  %3877 = add i32 %3850, 3
  %3878 = extractelement <8 x float> %1531, i64 3
  %3879 = fptrunc float %3878 to bfloat
  %3880 = mul i32 %3877, %19
  %3881 = mul i32 %3880, 128
  %3882 = add i32 %2625, %3881
  %3883 = add i32 %3882, %70
  %3884 = mul i32 %3883, 2
  %3885 = bitcast bfloat %3879 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3885, ptr addrspace(8) %93, i32 %3884, i32 0, i32 0)
  %3886 = add i32 %3850, 4
  %3887 = extractelement <8 x float> %1531, i64 4
  %3888 = fptrunc float %3887 to bfloat
  %3889 = mul i32 %3886, %19
  %3890 = mul i32 %3889, 128
  %3891 = add i32 %2625, %3890
  %3892 = add i32 %3891, %70
  %3893 = mul i32 %3892, 2
  %3894 = bitcast bfloat %3888 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3894, ptr addrspace(8) %93, i32 %3893, i32 0, i32 0)
  %3895 = add i32 %3850, 5
  %3896 = extractelement <8 x float> %1531, i64 5
  %3897 = fptrunc float %3896 to bfloat
  %3898 = mul i32 %3895, %19
  %3899 = mul i32 %3898, 128
  %3900 = add i32 %2625, %3899
  %3901 = add i32 %3900, %70
  %3902 = mul i32 %3901, 2
  %3903 = bitcast bfloat %3897 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3903, ptr addrspace(8) %93, i32 %3902, i32 0, i32 0)
  %3904 = add i32 %3850, 6
  %3905 = extractelement <8 x float> %1531, i64 6
  %3906 = fptrunc float %3905 to bfloat
  %3907 = mul i32 %3904, %19
  %3908 = mul i32 %3907, 128
  %3909 = add i32 %2625, %3908
  %3910 = add i32 %3909, %70
  %3911 = mul i32 %3910, 2
  %3912 = bitcast bfloat %3906 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3912, ptr addrspace(8) %93, i32 %3911, i32 0, i32 0)
  %3913 = add i32 %3850, 7
  %3914 = extractelement <8 x float> %1531, i64 7
  %3915 = fptrunc float %3914 to bfloat
  %3916 = mul i32 %3913, %19
  %3917 = mul i32 %3916, 128
  %3918 = add i32 %2625, %3917
  %3919 = add i32 %3918, %70
  %3920 = mul i32 %3919, 2
  %3921 = bitcast bfloat %3915 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3921, ptr addrspace(8) %93, i32 %3920, i32 0, i32 0)
  %3922 = extractelement <8 x float> %1532, i64 0
  %3923 = fptrunc float %3922 to bfloat
  %3924 = add i32 %3855, 16
  %3925 = add i32 %3924, %70
  %3926 = mul i32 %3925, 2
  %3927 = bitcast bfloat %3923 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3927, ptr addrspace(8) %93, i32 %3926, i32 0, i32 0)
  %3928 = extractelement <8 x float> %1532, i64 1
  %3929 = fptrunc float %3928 to bfloat
  %3930 = add i32 %3864, 16
  %3931 = add i32 %3930, %70
  %3932 = mul i32 %3931, 2
  %3933 = bitcast bfloat %3929 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3933, ptr addrspace(8) %93, i32 %3932, i32 0, i32 0)
  %3934 = extractelement <8 x float> %1532, i64 2
  %3935 = fptrunc float %3934 to bfloat
  %3936 = add i32 %3873, 16
  %3937 = add i32 %3936, %70
  %3938 = mul i32 %3937, 2
  %3939 = bitcast bfloat %3935 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3939, ptr addrspace(8) %93, i32 %3938, i32 0, i32 0)
  %3940 = extractelement <8 x float> %1532, i64 3
  %3941 = fptrunc float %3940 to bfloat
  %3942 = add i32 %3882, 16
  %3943 = add i32 %3942, %70
  %3944 = mul i32 %3943, 2
  %3945 = bitcast bfloat %3941 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3945, ptr addrspace(8) %93, i32 %3944, i32 0, i32 0)
  %3946 = extractelement <8 x float> %1532, i64 4
  %3947 = fptrunc float %3946 to bfloat
  %3948 = add i32 %3891, 16
  %3949 = add i32 %3948, %70
  %3950 = mul i32 %3949, 2
  %3951 = bitcast bfloat %3947 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3951, ptr addrspace(8) %93, i32 %3950, i32 0, i32 0)
  %3952 = extractelement <8 x float> %1532, i64 5
  %3953 = fptrunc float %3952 to bfloat
  %3954 = add i32 %3900, 16
  %3955 = add i32 %3954, %70
  %3956 = mul i32 %3955, 2
  %3957 = bitcast bfloat %3953 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3957, ptr addrspace(8) %93, i32 %3956, i32 0, i32 0)
  %3958 = extractelement <8 x float> %1532, i64 6
  %3959 = fptrunc float %3958 to bfloat
  %3960 = add i32 %3909, 16
  %3961 = add i32 %3960, %70
  %3962 = mul i32 %3961, 2
  %3963 = bitcast bfloat %3959 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3963, ptr addrspace(8) %93, i32 %3962, i32 0, i32 0)
  %3964 = extractelement <8 x float> %1532, i64 7
  %3965 = fptrunc float %3964 to bfloat
  %3966 = add i32 %3918, 16
  %3967 = add i32 %3966, %70
  %3968 = mul i32 %3967, 2
  %3969 = bitcast bfloat %3965 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3969, ptr addrspace(8) %93, i32 %3968, i32 0, i32 0)
  %3970 = extractelement <8 x float> %1533, i64 0
  %3971 = fptrunc float %3970 to bfloat
  %3972 = add i32 %3855, 32
  %3973 = add i32 %3972, %70
  %3974 = mul i32 %3973, 2
  %3975 = bitcast bfloat %3971 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3975, ptr addrspace(8) %93, i32 %3974, i32 0, i32 0)
  %3976 = extractelement <8 x float> %1533, i64 1
  %3977 = fptrunc float %3976 to bfloat
  %3978 = add i32 %3864, 32
  %3979 = add i32 %3978, %70
  %3980 = mul i32 %3979, 2
  %3981 = bitcast bfloat %3977 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3981, ptr addrspace(8) %93, i32 %3980, i32 0, i32 0)
  %3982 = extractelement <8 x float> %1533, i64 2
  %3983 = fptrunc float %3982 to bfloat
  %3984 = add i32 %3873, 32
  %3985 = add i32 %3984, %70
  %3986 = mul i32 %3985, 2
  %3987 = bitcast bfloat %3983 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3987, ptr addrspace(8) %93, i32 %3986, i32 0, i32 0)
  %3988 = extractelement <8 x float> %1533, i64 3
  %3989 = fptrunc float %3988 to bfloat
  %3990 = add i32 %3882, 32
  %3991 = add i32 %3990, %70
  %3992 = mul i32 %3991, 2
  %3993 = bitcast bfloat %3989 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3993, ptr addrspace(8) %93, i32 %3992, i32 0, i32 0)
  %3994 = extractelement <8 x float> %1533, i64 4
  %3995 = fptrunc float %3994 to bfloat
  %3996 = add i32 %3891, 32
  %3997 = add i32 %3996, %70
  %3998 = mul i32 %3997, 2
  %3999 = bitcast bfloat %3995 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3999, ptr addrspace(8) %93, i32 %3998, i32 0, i32 0)
  %4000 = extractelement <8 x float> %1533, i64 5
  %4001 = fptrunc float %4000 to bfloat
  %4002 = add i32 %3900, 32
  %4003 = add i32 %4002, %70
  %4004 = mul i32 %4003, 2
  %4005 = bitcast bfloat %4001 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4005, ptr addrspace(8) %93, i32 %4004, i32 0, i32 0)
  %4006 = extractelement <8 x float> %1533, i64 6
  %4007 = fptrunc float %4006 to bfloat
  %4008 = add i32 %3909, 32
  %4009 = add i32 %4008, %70
  %4010 = mul i32 %4009, 2
  %4011 = bitcast bfloat %4007 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4011, ptr addrspace(8) %93, i32 %4010, i32 0, i32 0)
  %4012 = extractelement <8 x float> %1533, i64 7
  %4013 = fptrunc float %4012 to bfloat
  %4014 = add i32 %3918, 32
  %4015 = add i32 %4014, %70
  %4016 = mul i32 %4015, 2
  %4017 = bitcast bfloat %4013 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4017, ptr addrspace(8) %93, i32 %4016, i32 0, i32 0)
  %4018 = extractelement <8 x float> %1534, i64 0
  %4019 = fptrunc float %4018 to bfloat
  %4020 = add i32 %3855, 48
  %4021 = add i32 %4020, %70
  %4022 = mul i32 %4021, 2
  %4023 = bitcast bfloat %4019 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4023, ptr addrspace(8) %93, i32 %4022, i32 0, i32 0)
  %4024 = extractelement <8 x float> %1534, i64 1
  %4025 = fptrunc float %4024 to bfloat
  %4026 = add i32 %3864, 48
  %4027 = add i32 %4026, %70
  %4028 = mul i32 %4027, 2
  %4029 = bitcast bfloat %4025 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4029, ptr addrspace(8) %93, i32 %4028, i32 0, i32 0)
  %4030 = extractelement <8 x float> %1534, i64 2
  %4031 = fptrunc float %4030 to bfloat
  %4032 = add i32 %3873, 48
  %4033 = add i32 %4032, %70
  %4034 = mul i32 %4033, 2
  %4035 = bitcast bfloat %4031 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4035, ptr addrspace(8) %93, i32 %4034, i32 0, i32 0)
  %4036 = extractelement <8 x float> %1534, i64 3
  %4037 = fptrunc float %4036 to bfloat
  %4038 = add i32 %3882, 48
  %4039 = add i32 %4038, %70
  %4040 = mul i32 %4039, 2
  %4041 = bitcast bfloat %4037 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4041, ptr addrspace(8) %93, i32 %4040, i32 0, i32 0)
  %4042 = extractelement <8 x float> %1534, i64 4
  %4043 = fptrunc float %4042 to bfloat
  %4044 = add i32 %3891, 48
  %4045 = add i32 %4044, %70
  %4046 = mul i32 %4045, 2
  %4047 = bitcast bfloat %4043 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4047, ptr addrspace(8) %93, i32 %4046, i32 0, i32 0)
  %4048 = extractelement <8 x float> %1534, i64 5
  %4049 = fptrunc float %4048 to bfloat
  %4050 = add i32 %3900, 48
  %4051 = add i32 %4050, %70
  %4052 = mul i32 %4051, 2
  %4053 = bitcast bfloat %4049 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4053, ptr addrspace(8) %93, i32 %4052, i32 0, i32 0)
  %4054 = extractelement <8 x float> %1534, i64 6
  %4055 = fptrunc float %4054 to bfloat
  %4056 = add i32 %3909, 48
  %4057 = add i32 %4056, %70
  %4058 = mul i32 %4057, 2
  %4059 = bitcast bfloat %4055 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4059, ptr addrspace(8) %93, i32 %4058, i32 0, i32 0)
  %4060 = extractelement <8 x float> %1534, i64 7
  %4061 = fptrunc float %4060 to bfloat
  %4062 = add i32 %3918, 48
  %4063 = add i32 %4062, %70
  %4064 = mul i32 %4063, 2
  %4065 = bitcast bfloat %4061 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4065, ptr addrspace(8) %93, i32 %4064, i32 0, i32 0)
  %4066 = extractelement <8 x float> %1535, i64 0
  %4067 = fptrunc float %4066 to bfloat
  %4068 = add i32 %3855, 64
  %4069 = add i32 %4068, %70
  %4070 = mul i32 %4069, 2
  %4071 = bitcast bfloat %4067 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4071, ptr addrspace(8) %93, i32 %4070, i32 0, i32 0)
  %4072 = extractelement <8 x float> %1535, i64 1
  %4073 = fptrunc float %4072 to bfloat
  %4074 = add i32 %3864, 64
  %4075 = add i32 %4074, %70
  %4076 = mul i32 %4075, 2
  %4077 = bitcast bfloat %4073 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4077, ptr addrspace(8) %93, i32 %4076, i32 0, i32 0)
  %4078 = extractelement <8 x float> %1535, i64 2
  %4079 = fptrunc float %4078 to bfloat
  %4080 = add i32 %3873, 64
  %4081 = add i32 %4080, %70
  %4082 = mul i32 %4081, 2
  %4083 = bitcast bfloat %4079 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4083, ptr addrspace(8) %93, i32 %4082, i32 0, i32 0)
  %4084 = extractelement <8 x float> %1535, i64 3
  %4085 = fptrunc float %4084 to bfloat
  %4086 = add i32 %3882, 64
  %4087 = add i32 %4086, %70
  %4088 = mul i32 %4087, 2
  %4089 = bitcast bfloat %4085 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4089, ptr addrspace(8) %93, i32 %4088, i32 0, i32 0)
  %4090 = extractelement <8 x float> %1535, i64 4
  %4091 = fptrunc float %4090 to bfloat
  %4092 = add i32 %3891, 64
  %4093 = add i32 %4092, %70
  %4094 = mul i32 %4093, 2
  %4095 = bitcast bfloat %4091 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4095, ptr addrspace(8) %93, i32 %4094, i32 0, i32 0)
  %4096 = extractelement <8 x float> %1535, i64 5
  %4097 = fptrunc float %4096 to bfloat
  %4098 = add i32 %3900, 64
  %4099 = add i32 %4098, %70
  %4100 = mul i32 %4099, 2
  %4101 = bitcast bfloat %4097 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4101, ptr addrspace(8) %93, i32 %4100, i32 0, i32 0)
  %4102 = extractelement <8 x float> %1535, i64 6
  %4103 = fptrunc float %4102 to bfloat
  %4104 = add i32 %3909, 64
  %4105 = add i32 %4104, %70
  %4106 = mul i32 %4105, 2
  %4107 = bitcast bfloat %4103 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4107, ptr addrspace(8) %93, i32 %4106, i32 0, i32 0)
  %4108 = extractelement <8 x float> %1535, i64 7
  %4109 = fptrunc float %4108 to bfloat
  %4110 = add i32 %3918, 64
  %4111 = add i32 %4110, %70
  %4112 = mul i32 %4111, 2
  %4113 = bitcast bfloat %4109 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4113, ptr addrspace(8) %93, i32 %4112, i32 0, i32 0)
  %4114 = extractelement <8 x float> %1536, i64 0
  %4115 = fptrunc float %4114 to bfloat
  %4116 = add i32 %3855, 80
  %4117 = add i32 %4116, %70
  %4118 = mul i32 %4117, 2
  %4119 = bitcast bfloat %4115 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4119, ptr addrspace(8) %93, i32 %4118, i32 0, i32 0)
  %4120 = extractelement <8 x float> %1536, i64 1
  %4121 = fptrunc float %4120 to bfloat
  %4122 = add i32 %3864, 80
  %4123 = add i32 %4122, %70
  %4124 = mul i32 %4123, 2
  %4125 = bitcast bfloat %4121 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4125, ptr addrspace(8) %93, i32 %4124, i32 0, i32 0)
  %4126 = extractelement <8 x float> %1536, i64 2
  %4127 = fptrunc float %4126 to bfloat
  %4128 = add i32 %3873, 80
  %4129 = add i32 %4128, %70
  %4130 = mul i32 %4129, 2
  %4131 = bitcast bfloat %4127 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4131, ptr addrspace(8) %93, i32 %4130, i32 0, i32 0)
  %4132 = extractelement <8 x float> %1536, i64 3
  %4133 = fptrunc float %4132 to bfloat
  %4134 = add i32 %3882, 80
  %4135 = add i32 %4134, %70
  %4136 = mul i32 %4135, 2
  %4137 = bitcast bfloat %4133 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4137, ptr addrspace(8) %93, i32 %4136, i32 0, i32 0)
  %4138 = extractelement <8 x float> %1536, i64 4
  %4139 = fptrunc float %4138 to bfloat
  %4140 = add i32 %3891, 80
  %4141 = add i32 %4140, %70
  %4142 = mul i32 %4141, 2
  %4143 = bitcast bfloat %4139 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4143, ptr addrspace(8) %93, i32 %4142, i32 0, i32 0)
  %4144 = extractelement <8 x float> %1536, i64 5
  %4145 = fptrunc float %4144 to bfloat
  %4146 = add i32 %3900, 80
  %4147 = add i32 %4146, %70
  %4148 = mul i32 %4147, 2
  %4149 = bitcast bfloat %4145 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4149, ptr addrspace(8) %93, i32 %4148, i32 0, i32 0)
  %4150 = extractelement <8 x float> %1536, i64 6
  %4151 = fptrunc float %4150 to bfloat
  %4152 = add i32 %3909, 80
  %4153 = add i32 %4152, %70
  %4154 = mul i32 %4153, 2
  %4155 = bitcast bfloat %4151 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4155, ptr addrspace(8) %93, i32 %4154, i32 0, i32 0)
  %4156 = extractelement <8 x float> %1536, i64 7
  %4157 = fptrunc float %4156 to bfloat
  %4158 = add i32 %3918, 80
  %4159 = add i32 %4158, %70
  %4160 = mul i32 %4159, 2
  %4161 = bitcast bfloat %4157 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4161, ptr addrspace(8) %93, i32 %4160, i32 0, i32 0)
  %4162 = extractelement <8 x float> %1537, i64 0
  %4163 = fptrunc float %4162 to bfloat
  %4164 = add i32 %3855, 96
  %4165 = add i32 %4164, %70
  %4166 = mul i32 %4165, 2
  %4167 = bitcast bfloat %4163 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4167, ptr addrspace(8) %93, i32 %4166, i32 0, i32 0)
  %4168 = extractelement <8 x float> %1537, i64 1
  %4169 = fptrunc float %4168 to bfloat
  %4170 = add i32 %3864, 96
  %4171 = add i32 %4170, %70
  %4172 = mul i32 %4171, 2
  %4173 = bitcast bfloat %4169 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4173, ptr addrspace(8) %93, i32 %4172, i32 0, i32 0)
  %4174 = extractelement <8 x float> %1537, i64 2
  %4175 = fptrunc float %4174 to bfloat
  %4176 = add i32 %3873, 96
  %4177 = add i32 %4176, %70
  %4178 = mul i32 %4177, 2
  %4179 = bitcast bfloat %4175 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4179, ptr addrspace(8) %93, i32 %4178, i32 0, i32 0)
  %4180 = extractelement <8 x float> %1537, i64 3
  %4181 = fptrunc float %4180 to bfloat
  %4182 = add i32 %3882, 96
  %4183 = add i32 %4182, %70
  %4184 = mul i32 %4183, 2
  %4185 = bitcast bfloat %4181 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4185, ptr addrspace(8) %93, i32 %4184, i32 0, i32 0)
  %4186 = extractelement <8 x float> %1537, i64 4
  %4187 = fptrunc float %4186 to bfloat
  %4188 = add i32 %3891, 96
  %4189 = add i32 %4188, %70
  %4190 = mul i32 %4189, 2
  %4191 = bitcast bfloat %4187 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4191, ptr addrspace(8) %93, i32 %4190, i32 0, i32 0)
  %4192 = extractelement <8 x float> %1537, i64 5
  %4193 = fptrunc float %4192 to bfloat
  %4194 = add i32 %3900, 96
  %4195 = add i32 %4194, %70
  %4196 = mul i32 %4195, 2
  %4197 = bitcast bfloat %4193 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4197, ptr addrspace(8) %93, i32 %4196, i32 0, i32 0)
  %4198 = extractelement <8 x float> %1537, i64 6
  %4199 = fptrunc float %4198 to bfloat
  %4200 = add i32 %3909, 96
  %4201 = add i32 %4200, %70
  %4202 = mul i32 %4201, 2
  %4203 = bitcast bfloat %4199 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4203, ptr addrspace(8) %93, i32 %4202, i32 0, i32 0)
  %4204 = extractelement <8 x float> %1537, i64 7
  %4205 = fptrunc float %4204 to bfloat
  %4206 = add i32 %3918, 96
  %4207 = add i32 %4206, %70
  %4208 = mul i32 %4207, 2
  %4209 = bitcast bfloat %4205 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4209, ptr addrspace(8) %93, i32 %4208, i32 0, i32 0)
  %4210 = extractelement <8 x float> %1538, i64 0
  %4211 = fptrunc float %4210 to bfloat
  %4212 = add i32 %3855, 112
  %4213 = add i32 %4212, %70
  %4214 = mul i32 %4213, 2
  %4215 = bitcast bfloat %4211 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4215, ptr addrspace(8) %93, i32 %4214, i32 0, i32 0)
  %4216 = extractelement <8 x float> %1538, i64 1
  %4217 = fptrunc float %4216 to bfloat
  %4218 = add i32 %3864, 112
  %4219 = add i32 %4218, %70
  %4220 = mul i32 %4219, 2
  %4221 = bitcast bfloat %4217 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4221, ptr addrspace(8) %93, i32 %4220, i32 0, i32 0)
  %4222 = extractelement <8 x float> %1538, i64 2
  %4223 = fptrunc float %4222 to bfloat
  %4224 = add i32 %3873, 112
  %4225 = add i32 %4224, %70
  %4226 = mul i32 %4225, 2
  %4227 = bitcast bfloat %4223 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4227, ptr addrspace(8) %93, i32 %4226, i32 0, i32 0)
  %4228 = extractelement <8 x float> %1538, i64 3
  %4229 = fptrunc float %4228 to bfloat
  %4230 = add i32 %3882, 112
  %4231 = add i32 %4230, %70
  %4232 = mul i32 %4231, 2
  %4233 = bitcast bfloat %4229 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4233, ptr addrspace(8) %93, i32 %4232, i32 0, i32 0)
  %4234 = extractelement <8 x float> %1538, i64 4
  %4235 = fptrunc float %4234 to bfloat
  %4236 = add i32 %3891, 112
  %4237 = add i32 %4236, %70
  %4238 = mul i32 %4237, 2
  %4239 = bitcast bfloat %4235 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4239, ptr addrspace(8) %93, i32 %4238, i32 0, i32 0)
  %4240 = extractelement <8 x float> %1538, i64 5
  %4241 = fptrunc float %4240 to bfloat
  %4242 = add i32 %3900, 112
  %4243 = add i32 %4242, %70
  %4244 = mul i32 %4243, 2
  %4245 = bitcast bfloat %4241 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4245, ptr addrspace(8) %93, i32 %4244, i32 0, i32 0)
  %4246 = extractelement <8 x float> %1538, i64 6
  %4247 = fptrunc float %4246 to bfloat
  %4248 = add i32 %3909, 112
  %4249 = add i32 %4248, %70
  %4250 = mul i32 %4249, 2
  %4251 = bitcast bfloat %4247 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4251, ptr addrspace(8) %93, i32 %4250, i32 0, i32 0)
  %4252 = extractelement <8 x float> %1538, i64 7
  %4253 = fptrunc float %4252 to bfloat
  %4254 = add i32 %3918, 112
  %4255 = add i32 %4254, %70
  %4256 = mul i32 %4255, 2
  %4257 = bitcast bfloat %4253 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %4257, ptr addrspace(8) %93, i32 %4256, i32 0, i32 0)
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

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

; Function Attrs: nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.amdgcn.s.wait.tensorcnt(i16 immarg) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #7

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat>, <16 x bfloat>, i16 immarg, <8 x float>, i1 immarg, i1 immarg) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: read)
declare <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) captures(none)) #9

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg) #5

attributes #0 = { "amdgpu-flat-work-group-size"="32,32" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nofree nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #7 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #8 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #9 = { convergent nocallback nofree nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 32, i32 1, i32 1}
