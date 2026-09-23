; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

@__shared_alloc_0 = external dso_local addrspace(3) global [22528 x i8], align 16

define amdgpu_kernel void @k_dkdv_0(ptr addrspace(1) %0, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %5, ptr addrspace(1) %6, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %7, ptr addrspace(1) %8, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %9, ptr addrspace(1) %10, <{ <{ i32, i32, i32 }>, <{ i64, i64 }> }> %11, ptr addrspace(1) %12, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %13, ptr addrspace(1) %14, <{ <{ i32, i32, i32, i32 }>, <{ i64, i64, i64 }> }> %15, float %16, i32 %17, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25) #0 !reqd_work_group_size !1 {
  %27 = call range(i32 0, 64) i32 @llvm.amdgcn.workitem.id.x()
  %28 = sext i32 %27 to i64
  %29 = trunc i64 %28 to i32
  %30 = call i32 @llvm.amdgcn.workgroup.id.x()
  %31 = sext i32 %30 to i64
  %32 = trunc i64 %31 to i32
  %33 = call i32 @llvm.amdgcn.workgroup.id.y()
  %34 = sext i32 %33 to i64
  %35 = trunc i64 %34 to i32
  %36 = call i32 @llvm.amdgcn.workgroup.id.z()
  %37 = sext i32 %36 to i64
  %38 = trunc i64 %37 to i32
  %39 = srem i32 %29, 16
  %40 = sdiv i32 %29, 16
  %41 = mul i32 %40, 16
  %42 = icmp ne i32 %29, %41
  %43 = icmp slt i32 %29, 0
  %44 = icmp ne i1 %43, false
  %45 = and i1 %42, %44
  %46 = add i32 %40, -1
  %47 = select i1 %45, i32 %46, i32 %40
  %48 = mul i32 %35, 32
  %49 = mul i32 %25, %17
  %50 = mul i32 %49, %19
  %51 = mul i32 %50, 256
  %52 = mul i32 %25, %18
  %53 = mul i32 %52, %20
  %54 = mul i32 %53, 256
  %55 = mul i32 %25, %19
  %56 = mul i32 %55, %17
  %57 = mul i32 %56, 4
  %58 = sext i32 %51 to i64
  %59 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %0, i16 0, i64 %58, i32 159744)
  %60 = sext i32 %54 to i64
  %61 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 %60, i32 159744)
  %62 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 %60, i32 159744)
  %63 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %6, i16 0, i64 %58, i32 159744)
  %64 = sext i32 %57 to i64
  %65 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %8, i16 0, i64 %64, i32 159744)
  %66 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %10, i16 0, i64 %64, i32 159744)
  %67 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %12, i16 0, i64 %60, i32 159744)
  %68 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %14, i16 0, i64 %60, i32 159744)
  %69 = mul i32 %19, 16
  %70 = mul i32 %20, 16
  %71 = mul i32 %38, %18
  %72 = mul i32 %71, %70
  %73 = mul i32 %32, 16
  %74 = add i32 %72, %73
  %75 = add i32 %48, %39
  %76 = mul i32 %75, %70
  %77 = add i32 %74, %76
  %78 = add i32 %77, %47
  %79 = mul i32 %78, 16
  %80 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %79, i32 0, i32 0)
  %81 = bitcast i128 %80 to <8 x bfloat>
  %82 = add i32 %78, 2
  %83 = mul i32 %82, 16
  %84 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %83, i32 0, i32 0)
  %85 = bitcast i128 %84 to <8 x bfloat>
  %86 = shufflevector <8 x bfloat> %81, <8 x bfloat> %85, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %87 = add i32 %78, 4
  %88 = mul i32 %87, 16
  %89 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %88, i32 0, i32 0)
  %90 = bitcast i128 %89 to <8 x bfloat>
  %91 = add i32 %78, 6
  %92 = mul i32 %91, 16
  %93 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %92, i32 0, i32 0)
  %94 = bitcast i128 %93 to <8 x bfloat>
  %95 = shufflevector <8 x bfloat> %90, <8 x bfloat> %94, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %96 = add i32 %78, 8
  %97 = mul i32 %96, 16
  %98 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %97, i32 0, i32 0)
  %99 = bitcast i128 %98 to <8 x bfloat>
  %100 = add i32 %78, 10
  %101 = mul i32 %100, 16
  %102 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %101, i32 0, i32 0)
  %103 = bitcast i128 %102 to <8 x bfloat>
  %104 = shufflevector <8 x bfloat> %99, <8 x bfloat> %103, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %105 = add i32 %78, 12
  %106 = mul i32 %105, 16
  %107 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %106, i32 0, i32 0)
  %108 = bitcast i128 %107 to <8 x bfloat>
  %109 = add i32 %78, 14
  %110 = mul i32 %109, 16
  %111 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %110, i32 0, i32 0)
  %112 = bitcast i128 %111 to <8 x bfloat>
  %113 = shufflevector <8 x bfloat> %108, <8 x bfloat> %112, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %114 = add i32 %48, 16
  %115 = add i32 %114, %39
  %116 = mul i32 %115, %70
  %117 = add i32 %74, %116
  %118 = add i32 %117, %47
  %119 = mul i32 %118, 16
  %120 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %119, i32 0, i32 0)
  %121 = bitcast i128 %120 to <8 x bfloat>
  %122 = add i32 %118, 2
  %123 = mul i32 %122, 16
  %124 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %123, i32 0, i32 0)
  %125 = bitcast i128 %124 to <8 x bfloat>
  %126 = shufflevector <8 x bfloat> %121, <8 x bfloat> %125, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %127 = add i32 %118, 4
  %128 = mul i32 %127, 16
  %129 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %128, i32 0, i32 0)
  %130 = bitcast i128 %129 to <8 x bfloat>
  %131 = add i32 %118, 6
  %132 = mul i32 %131, 16
  %133 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %132, i32 0, i32 0)
  %134 = bitcast i128 %133 to <8 x bfloat>
  %135 = shufflevector <8 x bfloat> %130, <8 x bfloat> %134, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %136 = add i32 %118, 8
  %137 = mul i32 %136, 16
  %138 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %137, i32 0, i32 0)
  %139 = bitcast i128 %138 to <8 x bfloat>
  %140 = add i32 %118, 10
  %141 = mul i32 %140, 16
  %142 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %141, i32 0, i32 0)
  %143 = bitcast i128 %142 to <8 x bfloat>
  %144 = shufflevector <8 x bfloat> %139, <8 x bfloat> %143, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %145 = add i32 %118, 12
  %146 = mul i32 %145, 16
  %147 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %146, i32 0, i32 0)
  %148 = bitcast i128 %147 to <8 x bfloat>
  %149 = add i32 %118, 14
  %150 = mul i32 %149, 16
  %151 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %61, i32 %150, i32 0, i32 0)
  %152 = bitcast i128 %151 to <8 x bfloat>
  %153 = shufflevector <8 x bfloat> %148, <8 x bfloat> %152, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %154 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %79, i32 0, i32 0)
  %155 = bitcast i128 %154 to <8 x bfloat>
  %156 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %83, i32 0, i32 0)
  %157 = bitcast i128 %156 to <8 x bfloat>
  %158 = shufflevector <8 x bfloat> %155, <8 x bfloat> %157, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %159 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %88, i32 0, i32 0)
  %160 = bitcast i128 %159 to <8 x bfloat>
  %161 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %92, i32 0, i32 0)
  %162 = bitcast i128 %161 to <8 x bfloat>
  %163 = shufflevector <8 x bfloat> %160, <8 x bfloat> %162, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %164 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %97, i32 0, i32 0)
  %165 = bitcast i128 %164 to <8 x bfloat>
  %166 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %101, i32 0, i32 0)
  %167 = bitcast i128 %166 to <8 x bfloat>
  %168 = shufflevector <8 x bfloat> %165, <8 x bfloat> %167, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %169 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %106, i32 0, i32 0)
  %170 = bitcast i128 %169 to <8 x bfloat>
  %171 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %110, i32 0, i32 0)
  %172 = bitcast i128 %171 to <8 x bfloat>
  %173 = shufflevector <8 x bfloat> %170, <8 x bfloat> %172, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %174 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %119, i32 0, i32 0)
  %175 = bitcast i128 %174 to <8 x bfloat>
  %176 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %123, i32 0, i32 0)
  %177 = bitcast i128 %176 to <8 x bfloat>
  %178 = shufflevector <8 x bfloat> %175, <8 x bfloat> %177, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %179 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %128, i32 0, i32 0)
  %180 = bitcast i128 %179 to <8 x bfloat>
  %181 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %132, i32 0, i32 0)
  %182 = bitcast i128 %181 to <8 x bfloat>
  %183 = shufflevector <8 x bfloat> %180, <8 x bfloat> %182, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %184 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %137, i32 0, i32 0)
  %185 = bitcast i128 %184 to <8 x bfloat>
  %186 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %141, i32 0, i32 0)
  %187 = bitcast i128 %186 to <8 x bfloat>
  %188 = shufflevector <8 x bfloat> %185, <8 x bfloat> %187, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %189 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %146, i32 0, i32 0)
  %190 = bitcast i128 %189 to <8 x bfloat>
  %191 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %62, i32 %150, i32 0, i32 0)
  %192 = bitcast i128 %191 to <8 x bfloat>
  %193 = shufflevector <8 x bfloat> %190, <8 x bfloat> %192, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %194 = mul i32 %47, 8
  %195 = srem i32 %29, 8
  %196 = add i32 %194, %195
  %197 = sdiv i32 %29, 8
  %198 = mul i32 %197, 8
  %199 = icmp ne i32 %29, %198
  %200 = icmp slt i32 %29, 0
  %201 = icmp ne i1 %200, false
  %202 = and i1 %199, %201
  %203 = add i32 %197, -1
  %204 = select i1 %202, i32 %203, i32 %197
  %205 = srem i32 %204, 2
  %206 = mul i32 %205, 8
  %207 = sdiv i32 %22, 2
  %208 = mul i32 %207, 2
  %209 = icmp ne i32 %22, %208
  %210 = icmp slt i32 %22, 0
  %211 = icmp ne i1 %210, false
  %212 = and i1 %209, %211
  %213 = add i32 %207, -1
  %214 = select i1 %212, i32 %213, i32 %207
  %215 = sub i32 %48, %23
  %216 = call i32 @llvm.smax.i32(i32 %215, i32 0)
  %217 = sdiv i32 %216, 32
  %218 = mul i32 %217, 32
  %219 = icmp ne i32 %216, %218
  %220 = icmp slt i32 %216, 0
  %221 = icmp ne i1 %220, false
  %222 = and i1 %219, %221
  %223 = add i32 %217, -1
  %224 = select i1 %222, i32 %223, i32 %217
  %225 = icmp ne i32 %24, 0
  %226 = select i1 %225, i32 %224, i32 0
  %227 = sub i32 %214, %226
  %228 = add i32 %48, 31
  %229 = sub i32 %228, %23
  %230 = icmp slt i32 %229, 0
  %231 = add i32 %229, 31
  %232 = sdiv i32 %231, 32
  %233 = mul i32 %232, 32
  %234 = icmp ne i32 %231, %233
  %235 = icmp slt i32 %231, 0
  %236 = icmp ne i1 %235, false
  %237 = and i1 %234, %236
  %238 = add i32 %232, -1
  %239 = select i1 %237, i32 %238, i32 %232
  %240 = select i1 %230, i32 0, i32 %239
  %241 = call i32 @llvm.smin.i32(i32 %240, i32 %214)
  %242 = sub i32 %241, %226
  %243 = call i32 @llvm.smax.i32(i32 %242, i32 0)
  %244 = call i32 @llvm.smin.i32(i32 %243, i32 %227)
  %245 = select i1 %225, i32 %244, i32 0
  %246 = mul i32 %21, %245
  %247 = sext i32 %246 to i64
  br label %248

248:                                              ; preds = %283, %26
  %249 = phi i64 [ %1302, %283 ], [ 0, %26 ]
  %250 = phi <8 x float> [ %1270, %283 ], [ zeroinitializer, %26 ]
  %251 = phi <8 x float> [ %1272, %283 ], [ zeroinitializer, %26 ]
  %252 = phi <8 x float> [ %1274, %283 ], [ zeroinitializer, %26 ]
  %253 = phi <8 x float> [ %1276, %283 ], [ zeroinitializer, %26 ]
  %254 = phi <8 x float> [ %1278, %283 ], [ zeroinitializer, %26 ]
  %255 = phi <8 x float> [ %1280, %283 ], [ zeroinitializer, %26 ]
  %256 = phi <8 x float> [ %1282, %283 ], [ zeroinitializer, %26 ]
  %257 = phi <8 x float> [ %1284, %283 ], [ zeroinitializer, %26 ]
  %258 = phi <8 x float> [ %1286, %283 ], [ zeroinitializer, %26 ]
  %259 = phi <8 x float> [ %1288, %283 ], [ zeroinitializer, %26 ]
  %260 = phi <8 x float> [ %1290, %283 ], [ zeroinitializer, %26 ]
  %261 = phi <8 x float> [ %1292, %283 ], [ zeroinitializer, %26 ]
  %262 = phi <8 x float> [ %1294, %283 ], [ zeroinitializer, %26 ]
  %263 = phi <8 x float> [ %1296, %283 ], [ zeroinitializer, %26 ]
  %264 = phi <8 x float> [ %1298, %283 ], [ zeroinitializer, %26 ]
  %265 = phi <8 x float> [ %1300, %283 ], [ zeroinitializer, %26 ]
  %266 = phi <8 x float> [ %1271, %283 ], [ zeroinitializer, %26 ]
  %267 = phi <8 x float> [ %1273, %283 ], [ zeroinitializer, %26 ]
  %268 = phi <8 x float> [ %1275, %283 ], [ zeroinitializer, %26 ]
  %269 = phi <8 x float> [ %1277, %283 ], [ zeroinitializer, %26 ]
  %270 = phi <8 x float> [ %1279, %283 ], [ zeroinitializer, %26 ]
  %271 = phi <8 x float> [ %1281, %283 ], [ zeroinitializer, %26 ]
  %272 = phi <8 x float> [ %1283, %283 ], [ zeroinitializer, %26 ]
  %273 = phi <8 x float> [ %1285, %283 ], [ zeroinitializer, %26 ]
  %274 = phi <8 x float> [ %1287, %283 ], [ zeroinitializer, %26 ]
  %275 = phi <8 x float> [ %1289, %283 ], [ zeroinitializer, %26 ]
  %276 = phi <8 x float> [ %1291, %283 ], [ zeroinitializer, %26 ]
  %277 = phi <8 x float> [ %1293, %283 ], [ zeroinitializer, %26 ]
  %278 = phi <8 x float> [ %1295, %283 ], [ zeroinitializer, %26 ]
  %279 = phi <8 x float> [ %1297, %283 ], [ zeroinitializer, %26 ]
  %280 = phi <8 x float> [ %1299, %283 ], [ zeroinitializer, %26 ]
  %281 = phi <8 x float> [ %1301, %283 ], [ zeroinitializer, %26 ]
  %282 = icmp slt i64 %249, %247
  br i1 %282, label %283, label %1303

283:                                              ; preds = %248
  %284 = trunc i64 %249 to i32
  %285 = sdiv i32 %284, %21
  %286 = mul i32 %285, %21
  %287 = icmp ne i32 %284, %286
  %288 = icmp slt i32 %284, 0
  %289 = icmp slt i32 %21, 0
  %290 = icmp ne i1 %288, %289
  %291 = and i1 %287, %290
  %292 = add i32 %285, -1
  %293 = select i1 %291, i32 %292, i32 %285
  %294 = add i32 %226, %293
  %295 = mul i32 %293, %21
  %296 = sub i32 %284, %295
  %297 = mul i32 %32, %21
  %298 = add i32 %297, %296
  %299 = mul i32 %38, %17
  %300 = mul i32 %299, %69
  %301 = mul i32 %298, 16
  %302 = add i32 %300, %301
  %303 = mul i32 %38, %19
  %304 = add i32 %303, %298
  %305 = mul i32 %304, %17
  %306 = mul i32 %294, 32
  %307 = add i32 %306, %39
  %308 = mul i32 %307, %69
  %309 = add i32 %302, %308
  %310 = add i32 %309, %47
  %311 = mul i32 %310, 16
  %312 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %311, i32 0, i32 0)
  %313 = bitcast i128 %312 to <8 x bfloat>
  %314 = add i32 %310, 2
  %315 = mul i32 %314, 16
  %316 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %315, i32 0, i32 0)
  %317 = bitcast i128 %316 to <8 x bfloat>
  %318 = add i32 %310, 4
  %319 = mul i32 %318, 16
  %320 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %319, i32 0, i32 0)
  %321 = bitcast i128 %320 to <8 x bfloat>
  %322 = add i32 %310, 6
  %323 = mul i32 %322, 16
  %324 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %323, i32 0, i32 0)
  %325 = bitcast i128 %324 to <8 x bfloat>
  %326 = add i32 %310, 8
  %327 = mul i32 %326, 16
  %328 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %327, i32 0, i32 0)
  %329 = bitcast i128 %328 to <8 x bfloat>
  %330 = add i32 %310, 10
  %331 = mul i32 %330, 16
  %332 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %331, i32 0, i32 0)
  %333 = bitcast i128 %332 to <8 x bfloat>
  %334 = add i32 %310, 12
  %335 = mul i32 %334, 16
  %336 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %335, i32 0, i32 0)
  %337 = bitcast i128 %336 to <8 x bfloat>
  %338 = add i32 %310, 14
  %339 = mul i32 %338, 16
  %340 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %339, i32 0, i32 0)
  %341 = bitcast i128 %340 to <8 x bfloat>
  %342 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %311, i32 0, i32 0)
  %343 = bitcast i128 %342 to <8 x bfloat>
  %344 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %315, i32 0, i32 0)
  %345 = bitcast i128 %344 to <8 x bfloat>
  %346 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %319, i32 0, i32 0)
  %347 = bitcast i128 %346 to <8 x bfloat>
  %348 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %323, i32 0, i32 0)
  %349 = bitcast i128 %348 to <8 x bfloat>
  %350 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %327, i32 0, i32 0)
  %351 = bitcast i128 %350 to <8 x bfloat>
  %352 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %331, i32 0, i32 0)
  %353 = bitcast i128 %352 to <8 x bfloat>
  %354 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %335, i32 0, i32 0)
  %355 = bitcast i128 %354 to <8 x bfloat>
  %356 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %339, i32 0, i32 0)
  %357 = bitcast i128 %356 to <8 x bfloat>
  %358 = mul i32 %39, 272
  %359 = mul i32 %47, 16
  %360 = add i32 %358, %359
  %361 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %360
  %362 = inttoptr i32 %361 to ptr addrspace(3)
  store <8 x bfloat> %343, ptr addrspace(3) %362, align 16
  %363 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %360
  %364 = inttoptr i32 %363 to ptr addrspace(3)
  store <8 x bfloat> %313, ptr addrspace(3) %364, align 16
  %365 = add i32 %360, 32
  %366 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %365
  %367 = inttoptr i32 %366 to ptr addrspace(3)
  store <8 x bfloat> %345, ptr addrspace(3) %367, align 16
  %368 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %365
  %369 = inttoptr i32 %368 to ptr addrspace(3)
  store <8 x bfloat> %317, ptr addrspace(3) %369, align 16
  %370 = add i32 %360, 64
  %371 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %370
  %372 = inttoptr i32 %371 to ptr addrspace(3)
  store <8 x bfloat> %347, ptr addrspace(3) %372, align 16
  %373 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %370
  %374 = inttoptr i32 %373 to ptr addrspace(3)
  store <8 x bfloat> %321, ptr addrspace(3) %374, align 16
  %375 = add i32 %360, 96
  %376 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %375
  %377 = inttoptr i32 %376 to ptr addrspace(3)
  store <8 x bfloat> %349, ptr addrspace(3) %377, align 16
  %378 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %375
  %379 = inttoptr i32 %378 to ptr addrspace(3)
  store <8 x bfloat> %325, ptr addrspace(3) %379, align 16
  %380 = add i32 %360, 128
  %381 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %380
  %382 = inttoptr i32 %381 to ptr addrspace(3)
  store <8 x bfloat> %351, ptr addrspace(3) %382, align 16
  %383 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %380
  %384 = inttoptr i32 %383 to ptr addrspace(3)
  store <8 x bfloat> %329, ptr addrspace(3) %384, align 16
  %385 = add i32 %360, 160
  %386 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %385
  %387 = inttoptr i32 %386 to ptr addrspace(3)
  store <8 x bfloat> %353, ptr addrspace(3) %387, align 16
  %388 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %385
  %389 = inttoptr i32 %388 to ptr addrspace(3)
  store <8 x bfloat> %333, ptr addrspace(3) %389, align 16
  %390 = add i32 %360, 192
  %391 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %390
  %392 = inttoptr i32 %391 to ptr addrspace(3)
  store <8 x bfloat> %355, ptr addrspace(3) %392, align 16
  %393 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %390
  %394 = inttoptr i32 %393 to ptr addrspace(3)
  store <8 x bfloat> %337, ptr addrspace(3) %394, align 16
  %395 = add i32 %360, 224
  %396 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %395
  %397 = inttoptr i32 %396 to ptr addrspace(3)
  store <8 x bfloat> %357, ptr addrspace(3) %397, align 16
  %398 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %395
  %399 = inttoptr i32 %398 to ptr addrspace(3)
  store <8 x bfloat> %341, ptr addrspace(3) %399, align 16
  %400 = shufflevector <8 x bfloat> %313, <8 x bfloat> %317, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %401 = shufflevector <8 x bfloat> %321, <8 x bfloat> %325, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %402 = shufflevector <8 x bfloat> %329, <8 x bfloat> %333, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %403 = shufflevector <8 x bfloat> %337, <8 x bfloat> %341, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %404 = shufflevector <8 x bfloat> %343, <8 x bfloat> %345, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %405 = shufflevector <8 x bfloat> %347, <8 x bfloat> %349, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %406 = shufflevector <8 x bfloat> %351, <8 x bfloat> %353, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %407 = shufflevector <8 x bfloat> %355, <8 x bfloat> %357, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %408 = add i32 %305, %307
  %409 = mul i32 %408, 4
  %410 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %65, i32 %409, i32 0, i32 0)
  %411 = bitcast i32 %410 to float
  %412 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %66, i32 %409, i32 0, i32 0)
  %413 = bitcast i32 %412 to float
  %414 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %86, <16 x bfloat> %400, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %415 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %158, <16 x bfloat> %404, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %416 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %95, <16 x bfloat> %401, i16 0, <8 x float> %414, i1 false, i1 false)
  %417 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %163, <16 x bfloat> %405, i16 0, <8 x float> %415, i1 false, i1 false)
  %418 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %104, <16 x bfloat> %402, i16 0, <8 x float> %416, i1 false, i1 false)
  %419 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %168, <16 x bfloat> %406, i16 0, <8 x float> %417, i1 false, i1 false)
  %420 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %113, <16 x bfloat> %403, i16 0, <8 x float> %418, i1 false, i1 false)
  %421 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %173, <16 x bfloat> %407, i16 0, <8 x float> %419, i1 false, i1 false)
  %422 = add i32 %48, %194
  %423 = add i32 %307, %23
  %424 = icmp sgt i32 %422, %423
  %425 = and i1 %424, %225
  %426 = extractelement <8 x float> %420, i64 0
  %427 = fmul float %426, %16
  %428 = select i1 %425, float -3.000000e+38, float %427
  %429 = add i32 %422, 1
  %430 = icmp sgt i32 %429, %423
  %431 = and i1 %430, %225
  %432 = extractelement <8 x float> %420, i64 1
  %433 = fmul float %432, %16
  %434 = select i1 %431, float -3.000000e+38, float %433
  %435 = add i32 %422, 2
  %436 = icmp sgt i32 %435, %423
  %437 = and i1 %436, %225
  %438 = extractelement <8 x float> %420, i64 2
  %439 = fmul float %438, %16
  %440 = select i1 %437, float -3.000000e+38, float %439
  %441 = add i32 %422, 3
  %442 = icmp sgt i32 %441, %423
  %443 = and i1 %442, %225
  %444 = extractelement <8 x float> %420, i64 3
  %445 = fmul float %444, %16
  %446 = select i1 %443, float -3.000000e+38, float %445
  %447 = add i32 %422, 4
  %448 = icmp sgt i32 %447, %423
  %449 = and i1 %448, %225
  %450 = extractelement <8 x float> %420, i64 4
  %451 = fmul float %450, %16
  %452 = select i1 %449, float -3.000000e+38, float %451
  %453 = add i32 %422, 5
  %454 = icmp sgt i32 %453, %423
  %455 = and i1 %454, %225
  %456 = extractelement <8 x float> %420, i64 5
  %457 = fmul float %456, %16
  %458 = select i1 %455, float -3.000000e+38, float %457
  %459 = add i32 %422, 6
  %460 = icmp sgt i32 %459, %423
  %461 = and i1 %460, %225
  %462 = extractelement <8 x float> %420, i64 6
  %463 = fmul float %462, %16
  %464 = select i1 %461, float -3.000000e+38, float %463
  %465 = add i32 %422, 7
  %466 = icmp sgt i32 %465, %423
  %467 = and i1 %466, %225
  %468 = extractelement <8 x float> %420, i64 7
  %469 = fmul float %468, %16
  %470 = select i1 %467, float -3.000000e+38, float %469
  %471 = fsub float %428, %411
  %472 = fmul float %471, f0x3FB8AA3B
  %473 = call float @llvm.amdgcn.exp2.f32(float %472)
  %474 = fsub float %434, %411
  %475 = fmul float %474, f0x3FB8AA3B
  %476 = call float @llvm.amdgcn.exp2.f32(float %475)
  %477 = fsub float %440, %411
  %478 = fmul float %477, f0x3FB8AA3B
  %479 = call float @llvm.amdgcn.exp2.f32(float %478)
  %480 = fsub float %446, %411
  %481 = fmul float %480, f0x3FB8AA3B
  %482 = call float @llvm.amdgcn.exp2.f32(float %481)
  %483 = fsub float %452, %411
  %484 = fmul float %483, f0x3FB8AA3B
  %485 = call float @llvm.amdgcn.exp2.f32(float %484)
  %486 = fsub float %458, %411
  %487 = fmul float %486, f0x3FB8AA3B
  %488 = call float @llvm.amdgcn.exp2.f32(float %487)
  %489 = fsub float %464, %411
  %490 = fmul float %489, f0x3FB8AA3B
  %491 = call float @llvm.amdgcn.exp2.f32(float %490)
  %492 = fsub float %470, %411
  %493 = fmul float %492, f0x3FB8AA3B
  %494 = call float @llvm.amdgcn.exp2.f32(float %493)
  %495 = fptrunc float %473 to bfloat
  %496 = fptrunc float %476 to bfloat
  %497 = fptrunc float %479 to bfloat
  %498 = fptrunc float %482 to bfloat
  %499 = fptrunc float %485 to bfloat
  %500 = fptrunc float %488 to bfloat
  %501 = fptrunc float %491 to bfloat
  %502 = fptrunc float %494 to bfloat
  %503 = extractelement <8 x float> %421, i64 0
  %504 = fsub float %503, %413
  %505 = fmul float %473, %504
  %506 = fmul float %505, %16
  %507 = fptrunc float %506 to bfloat
  %508 = extractelement <8 x float> %421, i64 1
  %509 = fsub float %508, %413
  %510 = fmul float %476, %509
  %511 = fmul float %510, %16
  %512 = fptrunc float %511 to bfloat
  %513 = extractelement <8 x float> %421, i64 2
  %514 = fsub float %513, %413
  %515 = fmul float %479, %514
  %516 = fmul float %515, %16
  %517 = fptrunc float %516 to bfloat
  %518 = extractelement <8 x float> %421, i64 3
  %519 = fsub float %518, %413
  %520 = fmul float %482, %519
  %521 = fmul float %520, %16
  %522 = fptrunc float %521 to bfloat
  %523 = extractelement <8 x float> %421, i64 4
  %524 = fsub float %523, %413
  %525 = fmul float %485, %524
  %526 = fmul float %525, %16
  %527 = fptrunc float %526 to bfloat
  %528 = extractelement <8 x float> %421, i64 5
  %529 = fsub float %528, %413
  %530 = fmul float %488, %529
  %531 = fmul float %530, %16
  %532 = fptrunc float %531 to bfloat
  %533 = extractelement <8 x float> %421, i64 6
  %534 = fsub float %533, %413
  %535 = fmul float %491, %534
  %536 = fmul float %535, %16
  %537 = fptrunc float %536 to bfloat
  %538 = extractelement <8 x float> %421, i64 7
  %539 = fsub float %538, %413
  %540 = fmul float %494, %539
  %541 = fmul float %540, %16
  %542 = fptrunc float %541 to bfloat
  %543 = mul i32 %39, 80
  %544 = add i32 %543, %359
  %545 = insertelement <8 x bfloat> poison, bfloat %495, i64 0
  %546 = insertelement <8 x bfloat> %545, bfloat %496, i64 1
  %547 = insertelement <8 x bfloat> %546, bfloat %497, i64 2
  %548 = insertelement <8 x bfloat> %547, bfloat %498, i64 3
  %549 = insertelement <8 x bfloat> %548, bfloat %499, i64 4
  %550 = insertelement <8 x bfloat> %549, bfloat %500, i64 5
  %551 = insertelement <8 x bfloat> %550, bfloat %501, i64 6
  %552 = insertelement <8 x bfloat> %551, bfloat %502, i64 7
  %553 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %544
  %554 = inttoptr i32 %553 to ptr addrspace(3)
  store <8 x bfloat> %552, ptr addrspace(3) %554, align 16
  %555 = insertelement <8 x bfloat> poison, bfloat %507, i64 0
  %556 = insertelement <8 x bfloat> %555, bfloat %512, i64 1
  %557 = insertelement <8 x bfloat> %556, bfloat %517, i64 2
  %558 = insertelement <8 x bfloat> %557, bfloat %522, i64 3
  %559 = insertelement <8 x bfloat> %558, bfloat %527, i64 4
  %560 = insertelement <8 x bfloat> %559, bfloat %532, i64 5
  %561 = insertelement <8 x bfloat> %560, bfloat %537, i64 6
  %562 = insertelement <8 x bfloat> %561, bfloat %542, i64 7
  %563 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %544
  %564 = inttoptr i32 %563 to ptr addrspace(3)
  store <8 x bfloat> %562, ptr addrspace(3) %564, align 16
  %565 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %126, <16 x bfloat> %400, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %566 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %178, <16 x bfloat> %404, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %567 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %135, <16 x bfloat> %401, i16 0, <8 x float> %565, i1 false, i1 false)
  %568 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %183, <16 x bfloat> %405, i16 0, <8 x float> %566, i1 false, i1 false)
  %569 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %144, <16 x bfloat> %402, i16 0, <8 x float> %567, i1 false, i1 false)
  %570 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %188, <16 x bfloat> %406, i16 0, <8 x float> %568, i1 false, i1 false)
  %571 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %153, <16 x bfloat> %403, i16 0, <8 x float> %569, i1 false, i1 false)
  %572 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %193, <16 x bfloat> %407, i16 0, <8 x float> %570, i1 false, i1 false)
  %573 = add i32 %114, %194
  %574 = icmp sgt i32 %573, %423
  %575 = and i1 %574, %225
  %576 = extractelement <8 x float> %571, i64 0
  %577 = fmul float %576, %16
  %578 = select i1 %575, float -3.000000e+38, float %577
  %579 = add i32 %573, 1
  %580 = icmp sgt i32 %579, %423
  %581 = and i1 %580, %225
  %582 = extractelement <8 x float> %571, i64 1
  %583 = fmul float %582, %16
  %584 = select i1 %581, float -3.000000e+38, float %583
  %585 = add i32 %573, 2
  %586 = icmp sgt i32 %585, %423
  %587 = and i1 %586, %225
  %588 = extractelement <8 x float> %571, i64 2
  %589 = fmul float %588, %16
  %590 = select i1 %587, float -3.000000e+38, float %589
  %591 = add i32 %573, 3
  %592 = icmp sgt i32 %591, %423
  %593 = and i1 %592, %225
  %594 = extractelement <8 x float> %571, i64 3
  %595 = fmul float %594, %16
  %596 = select i1 %593, float -3.000000e+38, float %595
  %597 = add i32 %573, 4
  %598 = icmp sgt i32 %597, %423
  %599 = and i1 %598, %225
  %600 = extractelement <8 x float> %571, i64 4
  %601 = fmul float %600, %16
  %602 = select i1 %599, float -3.000000e+38, float %601
  %603 = add i32 %573, 5
  %604 = icmp sgt i32 %603, %423
  %605 = and i1 %604, %225
  %606 = extractelement <8 x float> %571, i64 5
  %607 = fmul float %606, %16
  %608 = select i1 %605, float -3.000000e+38, float %607
  %609 = add i32 %573, 6
  %610 = icmp sgt i32 %609, %423
  %611 = and i1 %610, %225
  %612 = extractelement <8 x float> %571, i64 6
  %613 = fmul float %612, %16
  %614 = select i1 %611, float -3.000000e+38, float %613
  %615 = add i32 %573, 7
  %616 = icmp sgt i32 %615, %423
  %617 = and i1 %616, %225
  %618 = extractelement <8 x float> %571, i64 7
  %619 = fmul float %618, %16
  %620 = select i1 %617, float -3.000000e+38, float %619
  %621 = fsub float %578, %411
  %622 = fmul float %621, f0x3FB8AA3B
  %623 = call float @llvm.amdgcn.exp2.f32(float %622)
  %624 = fsub float %584, %411
  %625 = fmul float %624, f0x3FB8AA3B
  %626 = call float @llvm.amdgcn.exp2.f32(float %625)
  %627 = fsub float %590, %411
  %628 = fmul float %627, f0x3FB8AA3B
  %629 = call float @llvm.amdgcn.exp2.f32(float %628)
  %630 = fsub float %596, %411
  %631 = fmul float %630, f0x3FB8AA3B
  %632 = call float @llvm.amdgcn.exp2.f32(float %631)
  %633 = fsub float %602, %411
  %634 = fmul float %633, f0x3FB8AA3B
  %635 = call float @llvm.amdgcn.exp2.f32(float %634)
  %636 = fsub float %608, %411
  %637 = fmul float %636, f0x3FB8AA3B
  %638 = call float @llvm.amdgcn.exp2.f32(float %637)
  %639 = fsub float %614, %411
  %640 = fmul float %639, f0x3FB8AA3B
  %641 = call float @llvm.amdgcn.exp2.f32(float %640)
  %642 = fsub float %620, %411
  %643 = fmul float %642, f0x3FB8AA3B
  %644 = call float @llvm.amdgcn.exp2.f32(float %643)
  %645 = fptrunc float %623 to bfloat
  %646 = fptrunc float %626 to bfloat
  %647 = fptrunc float %629 to bfloat
  %648 = fptrunc float %632 to bfloat
  %649 = fptrunc float %635 to bfloat
  %650 = fptrunc float %638 to bfloat
  %651 = fptrunc float %641 to bfloat
  %652 = fptrunc float %644 to bfloat
  %653 = extractelement <8 x float> %572, i64 0
  %654 = fsub float %653, %413
  %655 = fmul float %623, %654
  %656 = fmul float %655, %16
  %657 = fptrunc float %656 to bfloat
  %658 = extractelement <8 x float> %572, i64 1
  %659 = fsub float %658, %413
  %660 = fmul float %626, %659
  %661 = fmul float %660, %16
  %662 = fptrunc float %661 to bfloat
  %663 = extractelement <8 x float> %572, i64 2
  %664 = fsub float %663, %413
  %665 = fmul float %629, %664
  %666 = fmul float %665, %16
  %667 = fptrunc float %666 to bfloat
  %668 = extractelement <8 x float> %572, i64 3
  %669 = fsub float %668, %413
  %670 = fmul float %632, %669
  %671 = fmul float %670, %16
  %672 = fptrunc float %671 to bfloat
  %673 = extractelement <8 x float> %572, i64 4
  %674 = fsub float %673, %413
  %675 = fmul float %635, %674
  %676 = fmul float %675, %16
  %677 = fptrunc float %676 to bfloat
  %678 = extractelement <8 x float> %572, i64 5
  %679 = fsub float %678, %413
  %680 = fmul float %638, %679
  %681 = fmul float %680, %16
  %682 = fptrunc float %681 to bfloat
  %683 = extractelement <8 x float> %572, i64 6
  %684 = fsub float %683, %413
  %685 = fmul float %641, %684
  %686 = fmul float %685, %16
  %687 = fptrunc float %686 to bfloat
  %688 = extractelement <8 x float> %572, i64 7
  %689 = fsub float %688, %413
  %690 = fmul float %644, %689
  %691 = fmul float %690, %16
  %692 = fptrunc float %691 to bfloat
  %693 = add i32 %543, 32
  %694 = add i32 %693, %359
  %695 = insertelement <8 x bfloat> poison, bfloat %645, i64 0
  %696 = insertelement <8 x bfloat> %695, bfloat %646, i64 1
  %697 = insertelement <8 x bfloat> %696, bfloat %647, i64 2
  %698 = insertelement <8 x bfloat> %697, bfloat %648, i64 3
  %699 = insertelement <8 x bfloat> %698, bfloat %649, i64 4
  %700 = insertelement <8 x bfloat> %699, bfloat %650, i64 5
  %701 = insertelement <8 x bfloat> %700, bfloat %651, i64 6
  %702 = insertelement <8 x bfloat> %701, bfloat %652, i64 7
  %703 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %694
  %704 = inttoptr i32 %703 to ptr addrspace(3)
  store <8 x bfloat> %702, ptr addrspace(3) %704, align 16
  %705 = insertelement <8 x bfloat> poison, bfloat %657, i64 0
  %706 = insertelement <8 x bfloat> %705, bfloat %662, i64 1
  %707 = insertelement <8 x bfloat> %706, bfloat %667, i64 2
  %708 = insertelement <8 x bfloat> %707, bfloat %672, i64 3
  %709 = insertelement <8 x bfloat> %708, bfloat %677, i64 4
  %710 = insertelement <8 x bfloat> %709, bfloat %682, i64 5
  %711 = insertelement <8 x bfloat> %710, bfloat %687, i64 6
  %712 = insertelement <8 x bfloat> %711, bfloat %692, i64 7
  %713 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %694
  %714 = inttoptr i32 %713 to ptr addrspace(3)
  store <8 x bfloat> %712, ptr addrspace(3) %714, align 16
  %715 = add i32 %306, 16
  %716 = add i32 %715, %39
  %717 = mul i32 %716, %69
  %718 = add i32 %302, %717
  %719 = add i32 %718, %47
  %720 = mul i32 %719, 16
  %721 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %720, i32 0, i32 0)
  %722 = bitcast i128 %721 to <8 x bfloat>
  %723 = add i32 %719, 2
  %724 = mul i32 %723, 16
  %725 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %724, i32 0, i32 0)
  %726 = bitcast i128 %725 to <8 x bfloat>
  %727 = add i32 %719, 4
  %728 = mul i32 %727, 16
  %729 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %728, i32 0, i32 0)
  %730 = bitcast i128 %729 to <8 x bfloat>
  %731 = add i32 %719, 6
  %732 = mul i32 %731, 16
  %733 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %732, i32 0, i32 0)
  %734 = bitcast i128 %733 to <8 x bfloat>
  %735 = add i32 %719, 8
  %736 = mul i32 %735, 16
  %737 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %736, i32 0, i32 0)
  %738 = bitcast i128 %737 to <8 x bfloat>
  %739 = add i32 %719, 10
  %740 = mul i32 %739, 16
  %741 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %740, i32 0, i32 0)
  %742 = bitcast i128 %741 to <8 x bfloat>
  %743 = add i32 %719, 12
  %744 = mul i32 %743, 16
  %745 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %744, i32 0, i32 0)
  %746 = bitcast i128 %745 to <8 x bfloat>
  %747 = add i32 %719, 14
  %748 = mul i32 %747, 16
  %749 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %748, i32 0, i32 0)
  %750 = bitcast i128 %749 to <8 x bfloat>
  %751 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %720, i32 0, i32 0)
  %752 = bitcast i128 %751 to <8 x bfloat>
  %753 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %724, i32 0, i32 0)
  %754 = bitcast i128 %753 to <8 x bfloat>
  %755 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %728, i32 0, i32 0)
  %756 = bitcast i128 %755 to <8 x bfloat>
  %757 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %732, i32 0, i32 0)
  %758 = bitcast i128 %757 to <8 x bfloat>
  %759 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %736, i32 0, i32 0)
  %760 = bitcast i128 %759 to <8 x bfloat>
  %761 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %740, i32 0, i32 0)
  %762 = bitcast i128 %761 to <8 x bfloat>
  %763 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %744, i32 0, i32 0)
  %764 = bitcast i128 %763 to <8 x bfloat>
  %765 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %748, i32 0, i32 0)
  %766 = bitcast i128 %765 to <8 x bfloat>
  %767 = add i32 %39, 16
  %768 = mul i32 %767, 272
  %769 = add i32 %768, %359
  %770 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %769
  %771 = inttoptr i32 %770 to ptr addrspace(3)
  store <8 x bfloat> %752, ptr addrspace(3) %771, align 16
  %772 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %769
  %773 = inttoptr i32 %772 to ptr addrspace(3)
  store <8 x bfloat> %722, ptr addrspace(3) %773, align 16
  %774 = add i32 %769, 32
  %775 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %774
  %776 = inttoptr i32 %775 to ptr addrspace(3)
  store <8 x bfloat> %754, ptr addrspace(3) %776, align 16
  %777 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %774
  %778 = inttoptr i32 %777 to ptr addrspace(3)
  store <8 x bfloat> %726, ptr addrspace(3) %778, align 16
  %779 = add i32 %769, 64
  %780 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %779
  %781 = inttoptr i32 %780 to ptr addrspace(3)
  store <8 x bfloat> %756, ptr addrspace(3) %781, align 16
  %782 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %779
  %783 = inttoptr i32 %782 to ptr addrspace(3)
  store <8 x bfloat> %730, ptr addrspace(3) %783, align 16
  %784 = add i32 %769, 96
  %785 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %784
  %786 = inttoptr i32 %785 to ptr addrspace(3)
  store <8 x bfloat> %758, ptr addrspace(3) %786, align 16
  %787 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %784
  %788 = inttoptr i32 %787 to ptr addrspace(3)
  store <8 x bfloat> %734, ptr addrspace(3) %788, align 16
  %789 = add i32 %769, 128
  %790 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %789
  %791 = inttoptr i32 %790 to ptr addrspace(3)
  store <8 x bfloat> %760, ptr addrspace(3) %791, align 16
  %792 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %789
  %793 = inttoptr i32 %792 to ptr addrspace(3)
  store <8 x bfloat> %738, ptr addrspace(3) %793, align 16
  %794 = add i32 %769, 160
  %795 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %794
  %796 = inttoptr i32 %795 to ptr addrspace(3)
  store <8 x bfloat> %762, ptr addrspace(3) %796, align 16
  %797 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %794
  %798 = inttoptr i32 %797 to ptr addrspace(3)
  store <8 x bfloat> %742, ptr addrspace(3) %798, align 16
  %799 = add i32 %769, 192
  %800 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %799
  %801 = inttoptr i32 %800 to ptr addrspace(3)
  store <8 x bfloat> %764, ptr addrspace(3) %801, align 16
  %802 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %799
  %803 = inttoptr i32 %802 to ptr addrspace(3)
  store <8 x bfloat> %746, ptr addrspace(3) %803, align 16
  %804 = add i32 %769, 224
  %805 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %804
  %806 = inttoptr i32 %805 to ptr addrspace(3)
  store <8 x bfloat> %766, ptr addrspace(3) %806, align 16
  %807 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %804
  %808 = inttoptr i32 %807 to ptr addrspace(3)
  store <8 x bfloat> %750, ptr addrspace(3) %808, align 16
  %809 = shufflevector <8 x bfloat> %722, <8 x bfloat> %726, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %810 = shufflevector <8 x bfloat> %730, <8 x bfloat> %734, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %811 = shufflevector <8 x bfloat> %738, <8 x bfloat> %742, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %812 = shufflevector <8 x bfloat> %746, <8 x bfloat> %750, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %813 = shufflevector <8 x bfloat> %752, <8 x bfloat> %754, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %814 = shufflevector <8 x bfloat> %756, <8 x bfloat> %758, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %815 = shufflevector <8 x bfloat> %760, <8 x bfloat> %762, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %816 = shufflevector <8 x bfloat> %764, <8 x bfloat> %766, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %817 = add i32 %305, %716
  %818 = mul i32 %817, 4
  %819 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %65, i32 %818, i32 0, i32 0)
  %820 = bitcast i32 %819 to float
  %821 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %66, i32 %818, i32 0, i32 0)
  %822 = bitcast i32 %821 to float
  %823 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %86, <16 x bfloat> %809, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %824 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %158, <16 x bfloat> %813, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %825 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %95, <16 x bfloat> %810, i16 0, <8 x float> %823, i1 false, i1 false)
  %826 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %163, <16 x bfloat> %814, i16 0, <8 x float> %824, i1 false, i1 false)
  %827 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %104, <16 x bfloat> %811, i16 0, <8 x float> %825, i1 false, i1 false)
  %828 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %168, <16 x bfloat> %815, i16 0, <8 x float> %826, i1 false, i1 false)
  %829 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %113, <16 x bfloat> %812, i16 0, <8 x float> %827, i1 false, i1 false)
  %830 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %173, <16 x bfloat> %816, i16 0, <8 x float> %828, i1 false, i1 false)
  %831 = add i32 %716, %23
  %832 = icmp sgt i32 %422, %831
  %833 = and i1 %832, %225
  %834 = extractelement <8 x float> %829, i64 0
  %835 = fmul float %834, %16
  %836 = select i1 %833, float -3.000000e+38, float %835
  %837 = icmp sgt i32 %429, %831
  %838 = and i1 %837, %225
  %839 = extractelement <8 x float> %829, i64 1
  %840 = fmul float %839, %16
  %841 = select i1 %838, float -3.000000e+38, float %840
  %842 = icmp sgt i32 %435, %831
  %843 = and i1 %842, %225
  %844 = extractelement <8 x float> %829, i64 2
  %845 = fmul float %844, %16
  %846 = select i1 %843, float -3.000000e+38, float %845
  %847 = icmp sgt i32 %441, %831
  %848 = and i1 %847, %225
  %849 = extractelement <8 x float> %829, i64 3
  %850 = fmul float %849, %16
  %851 = select i1 %848, float -3.000000e+38, float %850
  %852 = icmp sgt i32 %447, %831
  %853 = and i1 %852, %225
  %854 = extractelement <8 x float> %829, i64 4
  %855 = fmul float %854, %16
  %856 = select i1 %853, float -3.000000e+38, float %855
  %857 = icmp sgt i32 %453, %831
  %858 = and i1 %857, %225
  %859 = extractelement <8 x float> %829, i64 5
  %860 = fmul float %859, %16
  %861 = select i1 %858, float -3.000000e+38, float %860
  %862 = icmp sgt i32 %459, %831
  %863 = and i1 %862, %225
  %864 = extractelement <8 x float> %829, i64 6
  %865 = fmul float %864, %16
  %866 = select i1 %863, float -3.000000e+38, float %865
  %867 = icmp sgt i32 %465, %831
  %868 = and i1 %867, %225
  %869 = extractelement <8 x float> %829, i64 7
  %870 = fmul float %869, %16
  %871 = select i1 %868, float -3.000000e+38, float %870
  %872 = fsub float %836, %820
  %873 = fmul float %872, f0x3FB8AA3B
  %874 = call float @llvm.amdgcn.exp2.f32(float %873)
  %875 = fsub float %841, %820
  %876 = fmul float %875, f0x3FB8AA3B
  %877 = call float @llvm.amdgcn.exp2.f32(float %876)
  %878 = fsub float %846, %820
  %879 = fmul float %878, f0x3FB8AA3B
  %880 = call float @llvm.amdgcn.exp2.f32(float %879)
  %881 = fsub float %851, %820
  %882 = fmul float %881, f0x3FB8AA3B
  %883 = call float @llvm.amdgcn.exp2.f32(float %882)
  %884 = fsub float %856, %820
  %885 = fmul float %884, f0x3FB8AA3B
  %886 = call float @llvm.amdgcn.exp2.f32(float %885)
  %887 = fsub float %861, %820
  %888 = fmul float %887, f0x3FB8AA3B
  %889 = call float @llvm.amdgcn.exp2.f32(float %888)
  %890 = fsub float %866, %820
  %891 = fmul float %890, f0x3FB8AA3B
  %892 = call float @llvm.amdgcn.exp2.f32(float %891)
  %893 = fsub float %871, %820
  %894 = fmul float %893, f0x3FB8AA3B
  %895 = call float @llvm.amdgcn.exp2.f32(float %894)
  %896 = fptrunc float %874 to bfloat
  %897 = fptrunc float %877 to bfloat
  %898 = fptrunc float %880 to bfloat
  %899 = fptrunc float %883 to bfloat
  %900 = fptrunc float %886 to bfloat
  %901 = fptrunc float %889 to bfloat
  %902 = fptrunc float %892 to bfloat
  %903 = fptrunc float %895 to bfloat
  %904 = extractelement <8 x float> %830, i64 0
  %905 = fsub float %904, %822
  %906 = fmul float %874, %905
  %907 = fmul float %906, %16
  %908 = fptrunc float %907 to bfloat
  %909 = extractelement <8 x float> %830, i64 1
  %910 = fsub float %909, %822
  %911 = fmul float %877, %910
  %912 = fmul float %911, %16
  %913 = fptrunc float %912 to bfloat
  %914 = extractelement <8 x float> %830, i64 2
  %915 = fsub float %914, %822
  %916 = fmul float %880, %915
  %917 = fmul float %916, %16
  %918 = fptrunc float %917 to bfloat
  %919 = extractelement <8 x float> %830, i64 3
  %920 = fsub float %919, %822
  %921 = fmul float %883, %920
  %922 = fmul float %921, %16
  %923 = fptrunc float %922 to bfloat
  %924 = extractelement <8 x float> %830, i64 4
  %925 = fsub float %924, %822
  %926 = fmul float %886, %925
  %927 = fmul float %926, %16
  %928 = fptrunc float %927 to bfloat
  %929 = extractelement <8 x float> %830, i64 5
  %930 = fsub float %929, %822
  %931 = fmul float %889, %930
  %932 = fmul float %931, %16
  %933 = fptrunc float %932 to bfloat
  %934 = extractelement <8 x float> %830, i64 6
  %935 = fsub float %934, %822
  %936 = fmul float %892, %935
  %937 = fmul float %936, %16
  %938 = fptrunc float %937 to bfloat
  %939 = extractelement <8 x float> %830, i64 7
  %940 = fsub float %939, %822
  %941 = fmul float %895, %940
  %942 = fmul float %941, %16
  %943 = fptrunc float %942 to bfloat
  %944 = mul i32 %767, 80
  %945 = add i32 %944, %359
  %946 = insertelement <8 x bfloat> poison, bfloat %896, i64 0
  %947 = insertelement <8 x bfloat> %946, bfloat %897, i64 1
  %948 = insertelement <8 x bfloat> %947, bfloat %898, i64 2
  %949 = insertelement <8 x bfloat> %948, bfloat %899, i64 3
  %950 = insertelement <8 x bfloat> %949, bfloat %900, i64 4
  %951 = insertelement <8 x bfloat> %950, bfloat %901, i64 5
  %952 = insertelement <8 x bfloat> %951, bfloat %902, i64 6
  %953 = insertelement <8 x bfloat> %952, bfloat %903, i64 7
  %954 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %945
  %955 = inttoptr i32 %954 to ptr addrspace(3)
  store <8 x bfloat> %953, ptr addrspace(3) %955, align 16
  %956 = insertelement <8 x bfloat> poison, bfloat %908, i64 0
  %957 = insertelement <8 x bfloat> %956, bfloat %913, i64 1
  %958 = insertelement <8 x bfloat> %957, bfloat %918, i64 2
  %959 = insertelement <8 x bfloat> %958, bfloat %923, i64 3
  %960 = insertelement <8 x bfloat> %959, bfloat %928, i64 4
  %961 = insertelement <8 x bfloat> %960, bfloat %933, i64 5
  %962 = insertelement <8 x bfloat> %961, bfloat %938, i64 6
  %963 = insertelement <8 x bfloat> %962, bfloat %943, i64 7
  %964 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %945
  %965 = inttoptr i32 %964 to ptr addrspace(3)
  store <8 x bfloat> %963, ptr addrspace(3) %965, align 16
  %966 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %126, <16 x bfloat> %809, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %967 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %178, <16 x bfloat> %813, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %968 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %135, <16 x bfloat> %810, i16 0, <8 x float> %966, i1 false, i1 false)
  %969 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %183, <16 x bfloat> %814, i16 0, <8 x float> %967, i1 false, i1 false)
  %970 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %144, <16 x bfloat> %811, i16 0, <8 x float> %968, i1 false, i1 false)
  %971 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %188, <16 x bfloat> %815, i16 0, <8 x float> %969, i1 false, i1 false)
  %972 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %153, <16 x bfloat> %812, i16 0, <8 x float> %970, i1 false, i1 false)
  %973 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %193, <16 x bfloat> %816, i16 0, <8 x float> %971, i1 false, i1 false)
  %974 = icmp sgt i32 %573, %831
  %975 = and i1 %974, %225
  %976 = extractelement <8 x float> %972, i64 0
  %977 = fmul float %976, %16
  %978 = select i1 %975, float -3.000000e+38, float %977
  %979 = icmp sgt i32 %579, %831
  %980 = and i1 %979, %225
  %981 = extractelement <8 x float> %972, i64 1
  %982 = fmul float %981, %16
  %983 = select i1 %980, float -3.000000e+38, float %982
  %984 = icmp sgt i32 %585, %831
  %985 = and i1 %984, %225
  %986 = extractelement <8 x float> %972, i64 2
  %987 = fmul float %986, %16
  %988 = select i1 %985, float -3.000000e+38, float %987
  %989 = icmp sgt i32 %591, %831
  %990 = and i1 %989, %225
  %991 = extractelement <8 x float> %972, i64 3
  %992 = fmul float %991, %16
  %993 = select i1 %990, float -3.000000e+38, float %992
  %994 = icmp sgt i32 %597, %831
  %995 = and i1 %994, %225
  %996 = extractelement <8 x float> %972, i64 4
  %997 = fmul float %996, %16
  %998 = select i1 %995, float -3.000000e+38, float %997
  %999 = icmp sgt i32 %603, %831
  %1000 = and i1 %999, %225
  %1001 = extractelement <8 x float> %972, i64 5
  %1002 = fmul float %1001, %16
  %1003 = select i1 %1000, float -3.000000e+38, float %1002
  %1004 = icmp sgt i32 %609, %831
  %1005 = and i1 %1004, %225
  %1006 = extractelement <8 x float> %972, i64 6
  %1007 = fmul float %1006, %16
  %1008 = select i1 %1005, float -3.000000e+38, float %1007
  %1009 = icmp sgt i32 %615, %831
  %1010 = and i1 %1009, %225
  %1011 = extractelement <8 x float> %972, i64 7
  %1012 = fmul float %1011, %16
  %1013 = select i1 %1010, float -3.000000e+38, float %1012
  %1014 = fsub float %978, %820
  %1015 = fmul float %1014, f0x3FB8AA3B
  %1016 = call float @llvm.amdgcn.exp2.f32(float %1015)
  %1017 = fsub float %983, %820
  %1018 = fmul float %1017, f0x3FB8AA3B
  %1019 = call float @llvm.amdgcn.exp2.f32(float %1018)
  %1020 = fsub float %988, %820
  %1021 = fmul float %1020, f0x3FB8AA3B
  %1022 = call float @llvm.amdgcn.exp2.f32(float %1021)
  %1023 = fsub float %993, %820
  %1024 = fmul float %1023, f0x3FB8AA3B
  %1025 = call float @llvm.amdgcn.exp2.f32(float %1024)
  %1026 = fsub float %998, %820
  %1027 = fmul float %1026, f0x3FB8AA3B
  %1028 = call float @llvm.amdgcn.exp2.f32(float %1027)
  %1029 = fsub float %1003, %820
  %1030 = fmul float %1029, f0x3FB8AA3B
  %1031 = call float @llvm.amdgcn.exp2.f32(float %1030)
  %1032 = fsub float %1008, %820
  %1033 = fmul float %1032, f0x3FB8AA3B
  %1034 = call float @llvm.amdgcn.exp2.f32(float %1033)
  %1035 = fsub float %1013, %820
  %1036 = fmul float %1035, f0x3FB8AA3B
  %1037 = call float @llvm.amdgcn.exp2.f32(float %1036)
  %1038 = fptrunc float %1016 to bfloat
  %1039 = fptrunc float %1019 to bfloat
  %1040 = fptrunc float %1022 to bfloat
  %1041 = fptrunc float %1025 to bfloat
  %1042 = fptrunc float %1028 to bfloat
  %1043 = fptrunc float %1031 to bfloat
  %1044 = fptrunc float %1034 to bfloat
  %1045 = fptrunc float %1037 to bfloat
  %1046 = extractelement <8 x float> %973, i64 0
  %1047 = fsub float %1046, %822
  %1048 = fmul float %1016, %1047
  %1049 = fmul float %1048, %16
  %1050 = fptrunc float %1049 to bfloat
  %1051 = extractelement <8 x float> %973, i64 1
  %1052 = fsub float %1051, %822
  %1053 = fmul float %1019, %1052
  %1054 = fmul float %1053, %16
  %1055 = fptrunc float %1054 to bfloat
  %1056 = extractelement <8 x float> %973, i64 2
  %1057 = fsub float %1056, %822
  %1058 = fmul float %1022, %1057
  %1059 = fmul float %1058, %16
  %1060 = fptrunc float %1059 to bfloat
  %1061 = extractelement <8 x float> %973, i64 3
  %1062 = fsub float %1061, %822
  %1063 = fmul float %1025, %1062
  %1064 = fmul float %1063, %16
  %1065 = fptrunc float %1064 to bfloat
  %1066 = extractelement <8 x float> %973, i64 4
  %1067 = fsub float %1066, %822
  %1068 = fmul float %1028, %1067
  %1069 = fmul float %1068, %16
  %1070 = fptrunc float %1069 to bfloat
  %1071 = extractelement <8 x float> %973, i64 5
  %1072 = fsub float %1071, %822
  %1073 = fmul float %1031, %1072
  %1074 = fmul float %1073, %16
  %1075 = fptrunc float %1074 to bfloat
  %1076 = extractelement <8 x float> %973, i64 6
  %1077 = fsub float %1076, %822
  %1078 = fmul float %1034, %1077
  %1079 = fmul float %1078, %16
  %1080 = fptrunc float %1079 to bfloat
  %1081 = extractelement <8 x float> %973, i64 7
  %1082 = fsub float %1081, %822
  %1083 = fmul float %1037, %1082
  %1084 = fmul float %1083, %16
  %1085 = fptrunc float %1084 to bfloat
  %1086 = add i32 %944, 32
  %1087 = add i32 %1086, %359
  %1088 = insertelement <8 x bfloat> poison, bfloat %1038, i64 0
  %1089 = insertelement <8 x bfloat> %1088, bfloat %1039, i64 1
  %1090 = insertelement <8 x bfloat> %1089, bfloat %1040, i64 2
  %1091 = insertelement <8 x bfloat> %1090, bfloat %1041, i64 3
  %1092 = insertelement <8 x bfloat> %1091, bfloat %1042, i64 4
  %1093 = insertelement <8 x bfloat> %1092, bfloat %1043, i64 5
  %1094 = insertelement <8 x bfloat> %1093, bfloat %1044, i64 6
  %1095 = insertelement <8 x bfloat> %1094, bfloat %1045, i64 7
  %1096 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1087
  %1097 = inttoptr i32 %1096 to ptr addrspace(3)
  store <8 x bfloat> %1095, ptr addrspace(3) %1097, align 16
  %1098 = insertelement <8 x bfloat> poison, bfloat %1050, i64 0
  %1099 = insertelement <8 x bfloat> %1098, bfloat %1055, i64 1
  %1100 = insertelement <8 x bfloat> %1099, bfloat %1060, i64 2
  %1101 = insertelement <8 x bfloat> %1100, bfloat %1065, i64 3
  %1102 = insertelement <8 x bfloat> %1101, bfloat %1070, i64 4
  %1103 = insertelement <8 x bfloat> %1102, bfloat %1075, i64 5
  %1104 = insertelement <8 x bfloat> %1103, bfloat %1080, i64 6
  %1105 = insertelement <8 x bfloat> %1104, bfloat %1085, i64 7
  %1106 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %1087
  %1107 = inttoptr i32 %1106 to ptr addrspace(3)
  store <8 x bfloat> %1105, ptr addrspace(3) %1107, align 16
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1108 = mul i32 %205, 16
  %1109 = mul i32 %196, 272
  %1110 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1109
  %1111 = add i32 %1110, %1108
  %1112 = inttoptr i32 %1111 to ptr addrspace(3)
  %1113 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1112)
  %1114 = add i32 %1111, 4352
  %1115 = inttoptr i32 %1114 to ptr addrspace(3)
  %1116 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1115)
  %1117 = shufflevector <8 x bfloat> %1113, <8 x bfloat> %1116, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1118 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1109
  %1119 = add i32 %1118, %1108
  %1120 = inttoptr i32 %1119 to ptr addrspace(3)
  %1121 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1120)
  %1122 = add i32 %1119, 4352
  %1123 = inttoptr i32 %1122 to ptr addrspace(3)
  %1124 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1123)
  %1125 = shufflevector <8 x bfloat> %1121, <8 x bfloat> %1124, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1126 = add i32 %206, 16
  %1127 = mul i32 %1126, 2
  %1128 = add i32 %1110, %1127
  %1129 = inttoptr i32 %1128 to ptr addrspace(3)
  %1130 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1129)
  %1131 = add i32 %1128, 4352
  %1132 = inttoptr i32 %1131 to ptr addrspace(3)
  %1133 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1132)
  %1134 = shufflevector <8 x bfloat> %1130, <8 x bfloat> %1133, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1135 = add i32 %1118, %1127
  %1136 = inttoptr i32 %1135 to ptr addrspace(3)
  %1137 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1136)
  %1138 = add i32 %1135, 4352
  %1139 = inttoptr i32 %1138 to ptr addrspace(3)
  %1140 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1139)
  %1141 = shufflevector <8 x bfloat> %1137, <8 x bfloat> %1140, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1142 = add i32 %206, 32
  %1143 = mul i32 %1142, 2
  %1144 = add i32 %1110, %1143
  %1145 = inttoptr i32 %1144 to ptr addrspace(3)
  %1146 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1145)
  %1147 = add i32 %1144, 4352
  %1148 = inttoptr i32 %1147 to ptr addrspace(3)
  %1149 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1148)
  %1150 = shufflevector <8 x bfloat> %1146, <8 x bfloat> %1149, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1151 = add i32 %1118, %1143
  %1152 = inttoptr i32 %1151 to ptr addrspace(3)
  %1153 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1152)
  %1154 = add i32 %1151, 4352
  %1155 = inttoptr i32 %1154 to ptr addrspace(3)
  %1156 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1155)
  %1157 = shufflevector <8 x bfloat> %1153, <8 x bfloat> %1156, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1158 = add i32 %206, 48
  %1159 = mul i32 %1158, 2
  %1160 = add i32 %1110, %1159
  %1161 = inttoptr i32 %1160 to ptr addrspace(3)
  %1162 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1161)
  %1163 = add i32 %1160, 4352
  %1164 = inttoptr i32 %1163 to ptr addrspace(3)
  %1165 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1164)
  %1166 = shufflevector <8 x bfloat> %1162, <8 x bfloat> %1165, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1167 = add i32 %1118, %1159
  %1168 = inttoptr i32 %1167 to ptr addrspace(3)
  %1169 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1168)
  %1170 = add i32 %1167, 4352
  %1171 = inttoptr i32 %1170 to ptr addrspace(3)
  %1172 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1171)
  %1173 = shufflevector <8 x bfloat> %1169, <8 x bfloat> %1172, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1174 = add i32 %206, 64
  %1175 = mul i32 %1174, 2
  %1176 = add i32 %1110, %1175
  %1177 = inttoptr i32 %1176 to ptr addrspace(3)
  %1178 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1177)
  %1179 = add i32 %1176, 4352
  %1180 = inttoptr i32 %1179 to ptr addrspace(3)
  %1181 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1180)
  %1182 = shufflevector <8 x bfloat> %1178, <8 x bfloat> %1181, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1183 = add i32 %1118, %1175
  %1184 = inttoptr i32 %1183 to ptr addrspace(3)
  %1185 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1184)
  %1186 = add i32 %1183, 4352
  %1187 = inttoptr i32 %1186 to ptr addrspace(3)
  %1188 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1187)
  %1189 = shufflevector <8 x bfloat> %1185, <8 x bfloat> %1188, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1190 = add i32 %206, 80
  %1191 = mul i32 %1190, 2
  %1192 = add i32 %1110, %1191
  %1193 = inttoptr i32 %1192 to ptr addrspace(3)
  %1194 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1193)
  %1195 = add i32 %1192, 4352
  %1196 = inttoptr i32 %1195 to ptr addrspace(3)
  %1197 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1196)
  %1198 = shufflevector <8 x bfloat> %1194, <8 x bfloat> %1197, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1199 = add i32 %1118, %1191
  %1200 = inttoptr i32 %1199 to ptr addrspace(3)
  %1201 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1200)
  %1202 = add i32 %1199, 4352
  %1203 = inttoptr i32 %1202 to ptr addrspace(3)
  %1204 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1203)
  %1205 = shufflevector <8 x bfloat> %1201, <8 x bfloat> %1204, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1206 = add i32 %206, 96
  %1207 = mul i32 %1206, 2
  %1208 = add i32 %1110, %1207
  %1209 = inttoptr i32 %1208 to ptr addrspace(3)
  %1210 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1209)
  %1211 = add i32 %1208, 4352
  %1212 = inttoptr i32 %1211 to ptr addrspace(3)
  %1213 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1212)
  %1214 = shufflevector <8 x bfloat> %1210, <8 x bfloat> %1213, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1215 = add i32 %1118, %1207
  %1216 = inttoptr i32 %1215 to ptr addrspace(3)
  %1217 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1216)
  %1218 = add i32 %1215, 4352
  %1219 = inttoptr i32 %1218 to ptr addrspace(3)
  %1220 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1219)
  %1221 = shufflevector <8 x bfloat> %1217, <8 x bfloat> %1220, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1222 = add i32 %206, 112
  %1223 = mul i32 %1222, 2
  %1224 = add i32 %1110, %1223
  %1225 = inttoptr i32 %1224 to ptr addrspace(3)
  %1226 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1225)
  %1227 = add i32 %1224, 4352
  %1228 = inttoptr i32 %1227 to ptr addrspace(3)
  %1229 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1228)
  %1230 = shufflevector <8 x bfloat> %1226, <8 x bfloat> %1229, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1231 = add i32 %1118, %1223
  %1232 = inttoptr i32 %1231 to ptr addrspace(3)
  %1233 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1232)
  %1234 = add i32 %1231, 4352
  %1235 = inttoptr i32 %1234 to ptr addrspace(3)
  %1236 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1235)
  %1237 = shufflevector <8 x bfloat> %1233, <8 x bfloat> %1236, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1238 = mul i32 %196, 80
  %1239 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1238
  %1240 = add i32 %1239, %1108
  %1241 = inttoptr i32 %1240 to ptr addrspace(3)
  %1242 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1241)
  %1243 = add i32 %1240, 1280
  %1244 = inttoptr i32 %1243 to ptr addrspace(3)
  %1245 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1244)
  %1246 = shufflevector <8 x bfloat> %1242, <8 x bfloat> %1245, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1247 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %1238
  %1248 = add i32 %1247, %1108
  %1249 = inttoptr i32 %1248 to ptr addrspace(3)
  %1250 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1249)
  %1251 = add i32 %1248, 1280
  %1252 = inttoptr i32 %1251 to ptr addrspace(3)
  %1253 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1252)
  %1254 = shufflevector <8 x bfloat> %1250, <8 x bfloat> %1253, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1255 = add i32 %1108, 32
  %1256 = add i32 %1239, %1255
  %1257 = inttoptr i32 %1256 to ptr addrspace(3)
  %1258 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1257)
  %1259 = add i32 %1256, 1280
  %1260 = inttoptr i32 %1259 to ptr addrspace(3)
  %1261 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1260)
  %1262 = shufflevector <8 x bfloat> %1258, <8 x bfloat> %1261, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1263 = add i32 %1247, %1255
  %1264 = inttoptr i32 %1263 to ptr addrspace(3)
  %1265 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1264)
  %1266 = add i32 %1263, 1280
  %1267 = inttoptr i32 %1266 to ptr addrspace(3)
  %1268 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %1267)
  %1269 = shufflevector <8 x bfloat> %1265, <8 x bfloat> %1268, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1270 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1246, <16 x bfloat> %1117, i16 0, <8 x float> %250, i1 false, i1 false)
  %1271 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1254, <16 x bfloat> %1125, i16 0, <8 x float> %266, i1 false, i1 false)
  %1272 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1246, <16 x bfloat> %1134, i16 0, <8 x float> %251, i1 false, i1 false)
  %1273 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1254, <16 x bfloat> %1141, i16 0, <8 x float> %267, i1 false, i1 false)
  %1274 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1246, <16 x bfloat> %1150, i16 0, <8 x float> %252, i1 false, i1 false)
  %1275 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1254, <16 x bfloat> %1157, i16 0, <8 x float> %268, i1 false, i1 false)
  %1276 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1246, <16 x bfloat> %1166, i16 0, <8 x float> %253, i1 false, i1 false)
  %1277 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1254, <16 x bfloat> %1173, i16 0, <8 x float> %269, i1 false, i1 false)
  %1278 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1246, <16 x bfloat> %1182, i16 0, <8 x float> %254, i1 false, i1 false)
  %1279 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1254, <16 x bfloat> %1189, i16 0, <8 x float> %270, i1 false, i1 false)
  %1280 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1246, <16 x bfloat> %1198, i16 0, <8 x float> %255, i1 false, i1 false)
  %1281 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1254, <16 x bfloat> %1205, i16 0, <8 x float> %271, i1 false, i1 false)
  %1282 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1246, <16 x bfloat> %1214, i16 0, <8 x float> %256, i1 false, i1 false)
  %1283 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1254, <16 x bfloat> %1221, i16 0, <8 x float> %272, i1 false, i1 false)
  %1284 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1246, <16 x bfloat> %1230, i16 0, <8 x float> %257, i1 false, i1 false)
  %1285 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1254, <16 x bfloat> %1237, i16 0, <8 x float> %273, i1 false, i1 false)
  %1286 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1262, <16 x bfloat> %1117, i16 0, <8 x float> %258, i1 false, i1 false)
  %1287 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1269, <16 x bfloat> %1125, i16 0, <8 x float> %274, i1 false, i1 false)
  %1288 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1262, <16 x bfloat> %1134, i16 0, <8 x float> %259, i1 false, i1 false)
  %1289 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1269, <16 x bfloat> %1141, i16 0, <8 x float> %275, i1 false, i1 false)
  %1290 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1262, <16 x bfloat> %1150, i16 0, <8 x float> %260, i1 false, i1 false)
  %1291 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1269, <16 x bfloat> %1157, i16 0, <8 x float> %276, i1 false, i1 false)
  %1292 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1262, <16 x bfloat> %1166, i16 0, <8 x float> %261, i1 false, i1 false)
  %1293 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1269, <16 x bfloat> %1173, i16 0, <8 x float> %277, i1 false, i1 false)
  %1294 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1262, <16 x bfloat> %1182, i16 0, <8 x float> %262, i1 false, i1 false)
  %1295 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1269, <16 x bfloat> %1189, i16 0, <8 x float> %278, i1 false, i1 false)
  %1296 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1262, <16 x bfloat> %1198, i16 0, <8 x float> %263, i1 false, i1 false)
  %1297 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1269, <16 x bfloat> %1205, i16 0, <8 x float> %279, i1 false, i1 false)
  %1298 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1262, <16 x bfloat> %1214, i16 0, <8 x float> %264, i1 false, i1 false)
  %1299 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1269, <16 x bfloat> %1221, i16 0, <8 x float> %280, i1 false, i1 false)
  %1300 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1262, <16 x bfloat> %1230, i16 0, <8 x float> %265, i1 false, i1 false)
  %1301 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %1269, <16 x bfloat> %1237, i16 0, <8 x float> %281, i1 false, i1 false)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %1302 = add i64 %249, 1
  br label %248

1303:                                             ; preds = %248
  %1304 = sub i32 %227, %245
  %1305 = mul i32 %21, %1304
  %1306 = add i32 %226, %245
  %1307 = sext i32 %1305 to i64
  br label %1308

1308:                                             ; preds = %1343, %1303
  %1309 = phi i64 [ %2248, %1343 ], [ 0, %1303 ]
  %1310 = phi <8 x float> [ %2216, %1343 ], [ %250, %1303 ]
  %1311 = phi <8 x float> [ %2218, %1343 ], [ %251, %1303 ]
  %1312 = phi <8 x float> [ %2220, %1343 ], [ %252, %1303 ]
  %1313 = phi <8 x float> [ %2222, %1343 ], [ %253, %1303 ]
  %1314 = phi <8 x float> [ %2224, %1343 ], [ %254, %1303 ]
  %1315 = phi <8 x float> [ %2226, %1343 ], [ %255, %1303 ]
  %1316 = phi <8 x float> [ %2228, %1343 ], [ %256, %1303 ]
  %1317 = phi <8 x float> [ %2230, %1343 ], [ %257, %1303 ]
  %1318 = phi <8 x float> [ %2232, %1343 ], [ %258, %1303 ]
  %1319 = phi <8 x float> [ %2234, %1343 ], [ %259, %1303 ]
  %1320 = phi <8 x float> [ %2236, %1343 ], [ %260, %1303 ]
  %1321 = phi <8 x float> [ %2238, %1343 ], [ %261, %1303 ]
  %1322 = phi <8 x float> [ %2240, %1343 ], [ %262, %1303 ]
  %1323 = phi <8 x float> [ %2242, %1343 ], [ %263, %1303 ]
  %1324 = phi <8 x float> [ %2244, %1343 ], [ %264, %1303 ]
  %1325 = phi <8 x float> [ %2246, %1343 ], [ %265, %1303 ]
  %1326 = phi <8 x float> [ %2217, %1343 ], [ %266, %1303 ]
  %1327 = phi <8 x float> [ %2219, %1343 ], [ %267, %1303 ]
  %1328 = phi <8 x float> [ %2221, %1343 ], [ %268, %1303 ]
  %1329 = phi <8 x float> [ %2223, %1343 ], [ %269, %1303 ]
  %1330 = phi <8 x float> [ %2225, %1343 ], [ %270, %1303 ]
  %1331 = phi <8 x float> [ %2227, %1343 ], [ %271, %1303 ]
  %1332 = phi <8 x float> [ %2229, %1343 ], [ %272, %1303 ]
  %1333 = phi <8 x float> [ %2231, %1343 ], [ %273, %1303 ]
  %1334 = phi <8 x float> [ %2233, %1343 ], [ %274, %1303 ]
  %1335 = phi <8 x float> [ %2235, %1343 ], [ %275, %1303 ]
  %1336 = phi <8 x float> [ %2237, %1343 ], [ %276, %1303 ]
  %1337 = phi <8 x float> [ %2239, %1343 ], [ %277, %1303 ]
  %1338 = phi <8 x float> [ %2241, %1343 ], [ %278, %1303 ]
  %1339 = phi <8 x float> [ %2243, %1343 ], [ %279, %1303 ]
  %1340 = phi <8 x float> [ %2245, %1343 ], [ %280, %1303 ]
  %1341 = phi <8 x float> [ %2247, %1343 ], [ %281, %1303 ]
  %1342 = icmp slt i64 %1309, %1307
  br i1 %1342, label %1343, label %2249

1343:                                             ; preds = %1308
  %1344 = trunc i64 %1309 to i32
  %1345 = sdiv i32 %1344, %21
  %1346 = mul i32 %1345, %21
  %1347 = icmp ne i32 %1344, %1346
  %1348 = icmp slt i32 %1344, 0
  %1349 = icmp slt i32 %21, 0
  %1350 = icmp ne i1 %1348, %1349
  %1351 = and i1 %1347, %1350
  %1352 = add i32 %1345, -1
  %1353 = select i1 %1351, i32 %1352, i32 %1345
  %1354 = add i32 %1306, %1353
  %1355 = mul i32 %1353, %21
  %1356 = sub i32 %1344, %1355
  %1357 = mul i32 %32, %21
  %1358 = add i32 %1357, %1356
  %1359 = mul i32 %38, %17
  %1360 = mul i32 %1359, %69
  %1361 = mul i32 %1358, 16
  %1362 = add i32 %1360, %1361
  %1363 = mul i32 %38, %19
  %1364 = add i32 %1363, %1358
  %1365 = mul i32 %1364, %17
  %1366 = mul i32 %1354, 32
  %1367 = add i32 %1366, %39
  %1368 = mul i32 %1367, %69
  %1369 = add i32 %1362, %1368
  %1370 = add i32 %1369, %47
  %1371 = mul i32 %1370, 16
  %1372 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1371, i32 0, i32 0)
  %1373 = bitcast i128 %1372 to <8 x bfloat>
  %1374 = add i32 %1370, 2
  %1375 = mul i32 %1374, 16
  %1376 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1375, i32 0, i32 0)
  %1377 = bitcast i128 %1376 to <8 x bfloat>
  %1378 = add i32 %1370, 4
  %1379 = mul i32 %1378, 16
  %1380 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1379, i32 0, i32 0)
  %1381 = bitcast i128 %1380 to <8 x bfloat>
  %1382 = add i32 %1370, 6
  %1383 = mul i32 %1382, 16
  %1384 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1383, i32 0, i32 0)
  %1385 = bitcast i128 %1384 to <8 x bfloat>
  %1386 = add i32 %1370, 8
  %1387 = mul i32 %1386, 16
  %1388 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1387, i32 0, i32 0)
  %1389 = bitcast i128 %1388 to <8 x bfloat>
  %1390 = add i32 %1370, 10
  %1391 = mul i32 %1390, 16
  %1392 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1391, i32 0, i32 0)
  %1393 = bitcast i128 %1392 to <8 x bfloat>
  %1394 = add i32 %1370, 12
  %1395 = mul i32 %1394, 16
  %1396 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1395, i32 0, i32 0)
  %1397 = bitcast i128 %1396 to <8 x bfloat>
  %1398 = add i32 %1370, 14
  %1399 = mul i32 %1398, 16
  %1400 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1399, i32 0, i32 0)
  %1401 = bitcast i128 %1400 to <8 x bfloat>
  %1402 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1371, i32 0, i32 0)
  %1403 = bitcast i128 %1402 to <8 x bfloat>
  %1404 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1375, i32 0, i32 0)
  %1405 = bitcast i128 %1404 to <8 x bfloat>
  %1406 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1379, i32 0, i32 0)
  %1407 = bitcast i128 %1406 to <8 x bfloat>
  %1408 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1383, i32 0, i32 0)
  %1409 = bitcast i128 %1408 to <8 x bfloat>
  %1410 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1387, i32 0, i32 0)
  %1411 = bitcast i128 %1410 to <8 x bfloat>
  %1412 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1391, i32 0, i32 0)
  %1413 = bitcast i128 %1412 to <8 x bfloat>
  %1414 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1395, i32 0, i32 0)
  %1415 = bitcast i128 %1414 to <8 x bfloat>
  %1416 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1399, i32 0, i32 0)
  %1417 = bitcast i128 %1416 to <8 x bfloat>
  %1418 = mul i32 %39, 272
  %1419 = mul i32 %47, 16
  %1420 = add i32 %1418, %1419
  %1421 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1420
  %1422 = inttoptr i32 %1421 to ptr addrspace(3)
  store <8 x bfloat> %1403, ptr addrspace(3) %1422, align 16
  %1423 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1420
  %1424 = inttoptr i32 %1423 to ptr addrspace(3)
  store <8 x bfloat> %1373, ptr addrspace(3) %1424, align 16
  %1425 = add i32 %1420, 32
  %1426 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1425
  %1427 = inttoptr i32 %1426 to ptr addrspace(3)
  store <8 x bfloat> %1405, ptr addrspace(3) %1427, align 16
  %1428 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1425
  %1429 = inttoptr i32 %1428 to ptr addrspace(3)
  store <8 x bfloat> %1377, ptr addrspace(3) %1429, align 16
  %1430 = add i32 %1420, 64
  %1431 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1430
  %1432 = inttoptr i32 %1431 to ptr addrspace(3)
  store <8 x bfloat> %1407, ptr addrspace(3) %1432, align 16
  %1433 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1430
  %1434 = inttoptr i32 %1433 to ptr addrspace(3)
  store <8 x bfloat> %1381, ptr addrspace(3) %1434, align 16
  %1435 = add i32 %1420, 96
  %1436 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1435
  %1437 = inttoptr i32 %1436 to ptr addrspace(3)
  store <8 x bfloat> %1409, ptr addrspace(3) %1437, align 16
  %1438 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1435
  %1439 = inttoptr i32 %1438 to ptr addrspace(3)
  store <8 x bfloat> %1385, ptr addrspace(3) %1439, align 16
  %1440 = add i32 %1420, 128
  %1441 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1440
  %1442 = inttoptr i32 %1441 to ptr addrspace(3)
  store <8 x bfloat> %1411, ptr addrspace(3) %1442, align 16
  %1443 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1440
  %1444 = inttoptr i32 %1443 to ptr addrspace(3)
  store <8 x bfloat> %1389, ptr addrspace(3) %1444, align 16
  %1445 = add i32 %1420, 160
  %1446 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1445
  %1447 = inttoptr i32 %1446 to ptr addrspace(3)
  store <8 x bfloat> %1413, ptr addrspace(3) %1447, align 16
  %1448 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1445
  %1449 = inttoptr i32 %1448 to ptr addrspace(3)
  store <8 x bfloat> %1393, ptr addrspace(3) %1449, align 16
  %1450 = add i32 %1420, 192
  %1451 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1450
  %1452 = inttoptr i32 %1451 to ptr addrspace(3)
  store <8 x bfloat> %1415, ptr addrspace(3) %1452, align 16
  %1453 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1450
  %1454 = inttoptr i32 %1453 to ptr addrspace(3)
  store <8 x bfloat> %1397, ptr addrspace(3) %1454, align 16
  %1455 = add i32 %1420, 224
  %1456 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1455
  %1457 = inttoptr i32 %1456 to ptr addrspace(3)
  store <8 x bfloat> %1417, ptr addrspace(3) %1457, align 16
  %1458 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1455
  %1459 = inttoptr i32 %1458 to ptr addrspace(3)
  store <8 x bfloat> %1401, ptr addrspace(3) %1459, align 16
  %1460 = shufflevector <8 x bfloat> %1373, <8 x bfloat> %1377, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1461 = shufflevector <8 x bfloat> %1381, <8 x bfloat> %1385, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1462 = shufflevector <8 x bfloat> %1389, <8 x bfloat> %1393, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1463 = shufflevector <8 x bfloat> %1397, <8 x bfloat> %1401, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1464 = shufflevector <8 x bfloat> %1403, <8 x bfloat> %1405, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1465 = shufflevector <8 x bfloat> %1407, <8 x bfloat> %1409, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1466 = shufflevector <8 x bfloat> %1411, <8 x bfloat> %1413, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1467 = shufflevector <8 x bfloat> %1415, <8 x bfloat> %1417, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1468 = add i32 %1365, %1367
  %1469 = mul i32 %1468, 4
  %1470 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %65, i32 %1469, i32 0, i32 0)
  %1471 = bitcast i32 %1470 to float
  %1472 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %66, i32 %1469, i32 0, i32 0)
  %1473 = bitcast i32 %1472 to float
  %1474 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %86, <16 x bfloat> %1460, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1475 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %158, <16 x bfloat> %1464, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1476 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %95, <16 x bfloat> %1461, i16 0, <8 x float> %1474, i1 false, i1 false)
  %1477 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %163, <16 x bfloat> %1465, i16 0, <8 x float> %1475, i1 false, i1 false)
  %1478 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %104, <16 x bfloat> %1462, i16 0, <8 x float> %1476, i1 false, i1 false)
  %1479 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %168, <16 x bfloat> %1466, i16 0, <8 x float> %1477, i1 false, i1 false)
  %1480 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %113, <16 x bfloat> %1463, i16 0, <8 x float> %1478, i1 false, i1 false)
  %1481 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %173, <16 x bfloat> %1467, i16 0, <8 x float> %1479, i1 false, i1 false)
  %1482 = extractelement <8 x float> %1480, i64 0
  %1483 = fmul float %1482, %16
  %1484 = extractelement <8 x float> %1480, i64 1
  %1485 = fmul float %1484, %16
  %1486 = extractelement <8 x float> %1480, i64 2
  %1487 = fmul float %1486, %16
  %1488 = extractelement <8 x float> %1480, i64 3
  %1489 = fmul float %1488, %16
  %1490 = extractelement <8 x float> %1480, i64 4
  %1491 = fmul float %1490, %16
  %1492 = extractelement <8 x float> %1480, i64 5
  %1493 = fmul float %1492, %16
  %1494 = extractelement <8 x float> %1480, i64 6
  %1495 = fmul float %1494, %16
  %1496 = extractelement <8 x float> %1480, i64 7
  %1497 = fmul float %1496, %16
  %1498 = fsub float %1483, %1471
  %1499 = fmul float %1498, f0x3FB8AA3B
  %1500 = call float @llvm.amdgcn.exp2.f32(float %1499)
  %1501 = fsub float %1485, %1471
  %1502 = fmul float %1501, f0x3FB8AA3B
  %1503 = call float @llvm.amdgcn.exp2.f32(float %1502)
  %1504 = fsub float %1487, %1471
  %1505 = fmul float %1504, f0x3FB8AA3B
  %1506 = call float @llvm.amdgcn.exp2.f32(float %1505)
  %1507 = fsub float %1489, %1471
  %1508 = fmul float %1507, f0x3FB8AA3B
  %1509 = call float @llvm.amdgcn.exp2.f32(float %1508)
  %1510 = fsub float %1491, %1471
  %1511 = fmul float %1510, f0x3FB8AA3B
  %1512 = call float @llvm.amdgcn.exp2.f32(float %1511)
  %1513 = fsub float %1493, %1471
  %1514 = fmul float %1513, f0x3FB8AA3B
  %1515 = call float @llvm.amdgcn.exp2.f32(float %1514)
  %1516 = fsub float %1495, %1471
  %1517 = fmul float %1516, f0x3FB8AA3B
  %1518 = call float @llvm.amdgcn.exp2.f32(float %1517)
  %1519 = fsub float %1497, %1471
  %1520 = fmul float %1519, f0x3FB8AA3B
  %1521 = call float @llvm.amdgcn.exp2.f32(float %1520)
  %1522 = fptrunc float %1500 to bfloat
  %1523 = fptrunc float %1503 to bfloat
  %1524 = fptrunc float %1506 to bfloat
  %1525 = fptrunc float %1509 to bfloat
  %1526 = fptrunc float %1512 to bfloat
  %1527 = fptrunc float %1515 to bfloat
  %1528 = fptrunc float %1518 to bfloat
  %1529 = fptrunc float %1521 to bfloat
  %1530 = extractelement <8 x float> %1481, i64 0
  %1531 = fsub float %1530, %1473
  %1532 = fmul float %1500, %1531
  %1533 = fmul float %1532, %16
  %1534 = fptrunc float %1533 to bfloat
  %1535 = extractelement <8 x float> %1481, i64 1
  %1536 = fsub float %1535, %1473
  %1537 = fmul float %1503, %1536
  %1538 = fmul float %1537, %16
  %1539 = fptrunc float %1538 to bfloat
  %1540 = extractelement <8 x float> %1481, i64 2
  %1541 = fsub float %1540, %1473
  %1542 = fmul float %1506, %1541
  %1543 = fmul float %1542, %16
  %1544 = fptrunc float %1543 to bfloat
  %1545 = extractelement <8 x float> %1481, i64 3
  %1546 = fsub float %1545, %1473
  %1547 = fmul float %1509, %1546
  %1548 = fmul float %1547, %16
  %1549 = fptrunc float %1548 to bfloat
  %1550 = extractelement <8 x float> %1481, i64 4
  %1551 = fsub float %1550, %1473
  %1552 = fmul float %1512, %1551
  %1553 = fmul float %1552, %16
  %1554 = fptrunc float %1553 to bfloat
  %1555 = extractelement <8 x float> %1481, i64 5
  %1556 = fsub float %1555, %1473
  %1557 = fmul float %1515, %1556
  %1558 = fmul float %1557, %16
  %1559 = fptrunc float %1558 to bfloat
  %1560 = extractelement <8 x float> %1481, i64 6
  %1561 = fsub float %1560, %1473
  %1562 = fmul float %1518, %1561
  %1563 = fmul float %1562, %16
  %1564 = fptrunc float %1563 to bfloat
  %1565 = extractelement <8 x float> %1481, i64 7
  %1566 = fsub float %1565, %1473
  %1567 = fmul float %1521, %1566
  %1568 = fmul float %1567, %16
  %1569 = fptrunc float %1568 to bfloat
  %1570 = mul i32 %39, 80
  %1571 = add i32 %1570, %1419
  %1572 = insertelement <8 x bfloat> poison, bfloat %1522, i64 0
  %1573 = insertelement <8 x bfloat> %1572, bfloat %1523, i64 1
  %1574 = insertelement <8 x bfloat> %1573, bfloat %1524, i64 2
  %1575 = insertelement <8 x bfloat> %1574, bfloat %1525, i64 3
  %1576 = insertelement <8 x bfloat> %1575, bfloat %1526, i64 4
  %1577 = insertelement <8 x bfloat> %1576, bfloat %1527, i64 5
  %1578 = insertelement <8 x bfloat> %1577, bfloat %1528, i64 6
  %1579 = insertelement <8 x bfloat> %1578, bfloat %1529, i64 7
  %1580 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1571
  %1581 = inttoptr i32 %1580 to ptr addrspace(3)
  store <8 x bfloat> %1579, ptr addrspace(3) %1581, align 16
  %1582 = insertelement <8 x bfloat> poison, bfloat %1534, i64 0
  %1583 = insertelement <8 x bfloat> %1582, bfloat %1539, i64 1
  %1584 = insertelement <8 x bfloat> %1583, bfloat %1544, i64 2
  %1585 = insertelement <8 x bfloat> %1584, bfloat %1549, i64 3
  %1586 = insertelement <8 x bfloat> %1585, bfloat %1554, i64 4
  %1587 = insertelement <8 x bfloat> %1586, bfloat %1559, i64 5
  %1588 = insertelement <8 x bfloat> %1587, bfloat %1564, i64 6
  %1589 = insertelement <8 x bfloat> %1588, bfloat %1569, i64 7
  %1590 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %1571
  %1591 = inttoptr i32 %1590 to ptr addrspace(3)
  store <8 x bfloat> %1589, ptr addrspace(3) %1591, align 16
  %1592 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %126, <16 x bfloat> %1460, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1593 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %178, <16 x bfloat> %1464, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1594 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %135, <16 x bfloat> %1461, i16 0, <8 x float> %1592, i1 false, i1 false)
  %1595 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %183, <16 x bfloat> %1465, i16 0, <8 x float> %1593, i1 false, i1 false)
  %1596 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %144, <16 x bfloat> %1462, i16 0, <8 x float> %1594, i1 false, i1 false)
  %1597 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %188, <16 x bfloat> %1466, i16 0, <8 x float> %1595, i1 false, i1 false)
  %1598 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %153, <16 x bfloat> %1463, i16 0, <8 x float> %1596, i1 false, i1 false)
  %1599 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %193, <16 x bfloat> %1467, i16 0, <8 x float> %1597, i1 false, i1 false)
  %1600 = extractelement <8 x float> %1598, i64 0
  %1601 = fmul float %1600, %16
  %1602 = extractelement <8 x float> %1598, i64 1
  %1603 = fmul float %1602, %16
  %1604 = extractelement <8 x float> %1598, i64 2
  %1605 = fmul float %1604, %16
  %1606 = extractelement <8 x float> %1598, i64 3
  %1607 = fmul float %1606, %16
  %1608 = extractelement <8 x float> %1598, i64 4
  %1609 = fmul float %1608, %16
  %1610 = extractelement <8 x float> %1598, i64 5
  %1611 = fmul float %1610, %16
  %1612 = extractelement <8 x float> %1598, i64 6
  %1613 = fmul float %1612, %16
  %1614 = extractelement <8 x float> %1598, i64 7
  %1615 = fmul float %1614, %16
  %1616 = fsub float %1601, %1471
  %1617 = fmul float %1616, f0x3FB8AA3B
  %1618 = call float @llvm.amdgcn.exp2.f32(float %1617)
  %1619 = fsub float %1603, %1471
  %1620 = fmul float %1619, f0x3FB8AA3B
  %1621 = call float @llvm.amdgcn.exp2.f32(float %1620)
  %1622 = fsub float %1605, %1471
  %1623 = fmul float %1622, f0x3FB8AA3B
  %1624 = call float @llvm.amdgcn.exp2.f32(float %1623)
  %1625 = fsub float %1607, %1471
  %1626 = fmul float %1625, f0x3FB8AA3B
  %1627 = call float @llvm.amdgcn.exp2.f32(float %1626)
  %1628 = fsub float %1609, %1471
  %1629 = fmul float %1628, f0x3FB8AA3B
  %1630 = call float @llvm.amdgcn.exp2.f32(float %1629)
  %1631 = fsub float %1611, %1471
  %1632 = fmul float %1631, f0x3FB8AA3B
  %1633 = call float @llvm.amdgcn.exp2.f32(float %1632)
  %1634 = fsub float %1613, %1471
  %1635 = fmul float %1634, f0x3FB8AA3B
  %1636 = call float @llvm.amdgcn.exp2.f32(float %1635)
  %1637 = fsub float %1615, %1471
  %1638 = fmul float %1637, f0x3FB8AA3B
  %1639 = call float @llvm.amdgcn.exp2.f32(float %1638)
  %1640 = fptrunc float %1618 to bfloat
  %1641 = fptrunc float %1621 to bfloat
  %1642 = fptrunc float %1624 to bfloat
  %1643 = fptrunc float %1627 to bfloat
  %1644 = fptrunc float %1630 to bfloat
  %1645 = fptrunc float %1633 to bfloat
  %1646 = fptrunc float %1636 to bfloat
  %1647 = fptrunc float %1639 to bfloat
  %1648 = extractelement <8 x float> %1599, i64 0
  %1649 = fsub float %1648, %1473
  %1650 = fmul float %1618, %1649
  %1651 = fmul float %1650, %16
  %1652 = fptrunc float %1651 to bfloat
  %1653 = extractelement <8 x float> %1599, i64 1
  %1654 = fsub float %1653, %1473
  %1655 = fmul float %1621, %1654
  %1656 = fmul float %1655, %16
  %1657 = fptrunc float %1656 to bfloat
  %1658 = extractelement <8 x float> %1599, i64 2
  %1659 = fsub float %1658, %1473
  %1660 = fmul float %1624, %1659
  %1661 = fmul float %1660, %16
  %1662 = fptrunc float %1661 to bfloat
  %1663 = extractelement <8 x float> %1599, i64 3
  %1664 = fsub float %1663, %1473
  %1665 = fmul float %1627, %1664
  %1666 = fmul float %1665, %16
  %1667 = fptrunc float %1666 to bfloat
  %1668 = extractelement <8 x float> %1599, i64 4
  %1669 = fsub float %1668, %1473
  %1670 = fmul float %1630, %1669
  %1671 = fmul float %1670, %16
  %1672 = fptrunc float %1671 to bfloat
  %1673 = extractelement <8 x float> %1599, i64 5
  %1674 = fsub float %1673, %1473
  %1675 = fmul float %1633, %1674
  %1676 = fmul float %1675, %16
  %1677 = fptrunc float %1676 to bfloat
  %1678 = extractelement <8 x float> %1599, i64 6
  %1679 = fsub float %1678, %1473
  %1680 = fmul float %1636, %1679
  %1681 = fmul float %1680, %16
  %1682 = fptrunc float %1681 to bfloat
  %1683 = extractelement <8 x float> %1599, i64 7
  %1684 = fsub float %1683, %1473
  %1685 = fmul float %1639, %1684
  %1686 = fmul float %1685, %16
  %1687 = fptrunc float %1686 to bfloat
  %1688 = add i32 %1570, 32
  %1689 = add i32 %1688, %1419
  %1690 = insertelement <8 x bfloat> poison, bfloat %1640, i64 0
  %1691 = insertelement <8 x bfloat> %1690, bfloat %1641, i64 1
  %1692 = insertelement <8 x bfloat> %1691, bfloat %1642, i64 2
  %1693 = insertelement <8 x bfloat> %1692, bfloat %1643, i64 3
  %1694 = insertelement <8 x bfloat> %1693, bfloat %1644, i64 4
  %1695 = insertelement <8 x bfloat> %1694, bfloat %1645, i64 5
  %1696 = insertelement <8 x bfloat> %1695, bfloat %1646, i64 6
  %1697 = insertelement <8 x bfloat> %1696, bfloat %1647, i64 7
  %1698 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1689
  %1699 = inttoptr i32 %1698 to ptr addrspace(3)
  store <8 x bfloat> %1697, ptr addrspace(3) %1699, align 16
  %1700 = insertelement <8 x bfloat> poison, bfloat %1652, i64 0
  %1701 = insertelement <8 x bfloat> %1700, bfloat %1657, i64 1
  %1702 = insertelement <8 x bfloat> %1701, bfloat %1662, i64 2
  %1703 = insertelement <8 x bfloat> %1702, bfloat %1667, i64 3
  %1704 = insertelement <8 x bfloat> %1703, bfloat %1672, i64 4
  %1705 = insertelement <8 x bfloat> %1704, bfloat %1677, i64 5
  %1706 = insertelement <8 x bfloat> %1705, bfloat %1682, i64 6
  %1707 = insertelement <8 x bfloat> %1706, bfloat %1687, i64 7
  %1708 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %1689
  %1709 = inttoptr i32 %1708 to ptr addrspace(3)
  store <8 x bfloat> %1707, ptr addrspace(3) %1709, align 16
  %1710 = add i32 %1366, 16
  %1711 = add i32 %1710, %39
  %1712 = mul i32 %1711, %69
  %1713 = add i32 %1362, %1712
  %1714 = add i32 %1713, %47
  %1715 = mul i32 %1714, 16
  %1716 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1715, i32 0, i32 0)
  %1717 = bitcast i128 %1716 to <8 x bfloat>
  %1718 = add i32 %1714, 2
  %1719 = mul i32 %1718, 16
  %1720 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1719, i32 0, i32 0)
  %1721 = bitcast i128 %1720 to <8 x bfloat>
  %1722 = add i32 %1714, 4
  %1723 = mul i32 %1722, 16
  %1724 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1723, i32 0, i32 0)
  %1725 = bitcast i128 %1724 to <8 x bfloat>
  %1726 = add i32 %1714, 6
  %1727 = mul i32 %1726, 16
  %1728 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1727, i32 0, i32 0)
  %1729 = bitcast i128 %1728 to <8 x bfloat>
  %1730 = add i32 %1714, 8
  %1731 = mul i32 %1730, 16
  %1732 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1731, i32 0, i32 0)
  %1733 = bitcast i128 %1732 to <8 x bfloat>
  %1734 = add i32 %1714, 10
  %1735 = mul i32 %1734, 16
  %1736 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1735, i32 0, i32 0)
  %1737 = bitcast i128 %1736 to <8 x bfloat>
  %1738 = add i32 %1714, 12
  %1739 = mul i32 %1738, 16
  %1740 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1739, i32 0, i32 0)
  %1741 = bitcast i128 %1740 to <8 x bfloat>
  %1742 = add i32 %1714, 14
  %1743 = mul i32 %1742, 16
  %1744 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %59, i32 %1743, i32 0, i32 0)
  %1745 = bitcast i128 %1744 to <8 x bfloat>
  %1746 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1715, i32 0, i32 0)
  %1747 = bitcast i128 %1746 to <8 x bfloat>
  %1748 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1719, i32 0, i32 0)
  %1749 = bitcast i128 %1748 to <8 x bfloat>
  %1750 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1723, i32 0, i32 0)
  %1751 = bitcast i128 %1750 to <8 x bfloat>
  %1752 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1727, i32 0, i32 0)
  %1753 = bitcast i128 %1752 to <8 x bfloat>
  %1754 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1731, i32 0, i32 0)
  %1755 = bitcast i128 %1754 to <8 x bfloat>
  %1756 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1735, i32 0, i32 0)
  %1757 = bitcast i128 %1756 to <8 x bfloat>
  %1758 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1739, i32 0, i32 0)
  %1759 = bitcast i128 %1758 to <8 x bfloat>
  %1760 = call i128 @llvm.amdgcn.raw.ptr.buffer.load.i128(ptr addrspace(8) %63, i32 %1743, i32 0, i32 0)
  %1761 = bitcast i128 %1760 to <8 x bfloat>
  %1762 = add i32 %39, 16
  %1763 = mul i32 %1762, 272
  %1764 = add i32 %1763, %1419
  %1765 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1764
  %1766 = inttoptr i32 %1765 to ptr addrspace(3)
  store <8 x bfloat> %1747, ptr addrspace(3) %1766, align 16
  %1767 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1764
  %1768 = inttoptr i32 %1767 to ptr addrspace(3)
  store <8 x bfloat> %1717, ptr addrspace(3) %1768, align 16
  %1769 = add i32 %1764, 32
  %1770 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1769
  %1771 = inttoptr i32 %1770 to ptr addrspace(3)
  store <8 x bfloat> %1749, ptr addrspace(3) %1771, align 16
  %1772 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1769
  %1773 = inttoptr i32 %1772 to ptr addrspace(3)
  store <8 x bfloat> %1721, ptr addrspace(3) %1773, align 16
  %1774 = add i32 %1764, 64
  %1775 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1774
  %1776 = inttoptr i32 %1775 to ptr addrspace(3)
  store <8 x bfloat> %1751, ptr addrspace(3) %1776, align 16
  %1777 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1774
  %1778 = inttoptr i32 %1777 to ptr addrspace(3)
  store <8 x bfloat> %1725, ptr addrspace(3) %1778, align 16
  %1779 = add i32 %1764, 96
  %1780 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1779
  %1781 = inttoptr i32 %1780 to ptr addrspace(3)
  store <8 x bfloat> %1753, ptr addrspace(3) %1781, align 16
  %1782 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1779
  %1783 = inttoptr i32 %1782 to ptr addrspace(3)
  store <8 x bfloat> %1729, ptr addrspace(3) %1783, align 16
  %1784 = add i32 %1764, 128
  %1785 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1784
  %1786 = inttoptr i32 %1785 to ptr addrspace(3)
  store <8 x bfloat> %1755, ptr addrspace(3) %1786, align 16
  %1787 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1784
  %1788 = inttoptr i32 %1787 to ptr addrspace(3)
  store <8 x bfloat> %1733, ptr addrspace(3) %1788, align 16
  %1789 = add i32 %1764, 160
  %1790 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1789
  %1791 = inttoptr i32 %1790 to ptr addrspace(3)
  store <8 x bfloat> %1757, ptr addrspace(3) %1791, align 16
  %1792 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1789
  %1793 = inttoptr i32 %1792 to ptr addrspace(3)
  store <8 x bfloat> %1737, ptr addrspace(3) %1793, align 16
  %1794 = add i32 %1764, 192
  %1795 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1794
  %1796 = inttoptr i32 %1795 to ptr addrspace(3)
  store <8 x bfloat> %1759, ptr addrspace(3) %1796, align 16
  %1797 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1794
  %1798 = inttoptr i32 %1797 to ptr addrspace(3)
  store <8 x bfloat> %1741, ptr addrspace(3) %1798, align 16
  %1799 = add i32 %1764, 224
  %1800 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %1799
  %1801 = inttoptr i32 %1800 to ptr addrspace(3)
  store <8 x bfloat> %1761, ptr addrspace(3) %1801, align 16
  %1802 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %1799
  %1803 = inttoptr i32 %1802 to ptr addrspace(3)
  store <8 x bfloat> %1745, ptr addrspace(3) %1803, align 16
  %1804 = shufflevector <8 x bfloat> %1717, <8 x bfloat> %1721, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1805 = shufflevector <8 x bfloat> %1725, <8 x bfloat> %1729, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1806 = shufflevector <8 x bfloat> %1733, <8 x bfloat> %1737, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1807 = shufflevector <8 x bfloat> %1741, <8 x bfloat> %1745, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1808 = shufflevector <8 x bfloat> %1747, <8 x bfloat> %1749, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1809 = shufflevector <8 x bfloat> %1751, <8 x bfloat> %1753, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1810 = shufflevector <8 x bfloat> %1755, <8 x bfloat> %1757, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1811 = shufflevector <8 x bfloat> %1759, <8 x bfloat> %1761, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1812 = add i32 %1365, %1711
  %1813 = mul i32 %1812, 4
  %1814 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %65, i32 %1813, i32 0, i32 0)
  %1815 = bitcast i32 %1814 to float
  %1816 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %66, i32 %1813, i32 0, i32 0)
  %1817 = bitcast i32 %1816 to float
  %1818 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %86, <16 x bfloat> %1804, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1819 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %158, <16 x bfloat> %1808, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1820 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %95, <16 x bfloat> %1805, i16 0, <8 x float> %1818, i1 false, i1 false)
  %1821 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %163, <16 x bfloat> %1809, i16 0, <8 x float> %1819, i1 false, i1 false)
  %1822 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %104, <16 x bfloat> %1806, i16 0, <8 x float> %1820, i1 false, i1 false)
  %1823 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %168, <16 x bfloat> %1810, i16 0, <8 x float> %1821, i1 false, i1 false)
  %1824 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %113, <16 x bfloat> %1807, i16 0, <8 x float> %1822, i1 false, i1 false)
  %1825 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %173, <16 x bfloat> %1811, i16 0, <8 x float> %1823, i1 false, i1 false)
  %1826 = extractelement <8 x float> %1824, i64 0
  %1827 = fmul float %1826, %16
  %1828 = extractelement <8 x float> %1824, i64 1
  %1829 = fmul float %1828, %16
  %1830 = extractelement <8 x float> %1824, i64 2
  %1831 = fmul float %1830, %16
  %1832 = extractelement <8 x float> %1824, i64 3
  %1833 = fmul float %1832, %16
  %1834 = extractelement <8 x float> %1824, i64 4
  %1835 = fmul float %1834, %16
  %1836 = extractelement <8 x float> %1824, i64 5
  %1837 = fmul float %1836, %16
  %1838 = extractelement <8 x float> %1824, i64 6
  %1839 = fmul float %1838, %16
  %1840 = extractelement <8 x float> %1824, i64 7
  %1841 = fmul float %1840, %16
  %1842 = fsub float %1827, %1815
  %1843 = fmul float %1842, f0x3FB8AA3B
  %1844 = call float @llvm.amdgcn.exp2.f32(float %1843)
  %1845 = fsub float %1829, %1815
  %1846 = fmul float %1845, f0x3FB8AA3B
  %1847 = call float @llvm.amdgcn.exp2.f32(float %1846)
  %1848 = fsub float %1831, %1815
  %1849 = fmul float %1848, f0x3FB8AA3B
  %1850 = call float @llvm.amdgcn.exp2.f32(float %1849)
  %1851 = fsub float %1833, %1815
  %1852 = fmul float %1851, f0x3FB8AA3B
  %1853 = call float @llvm.amdgcn.exp2.f32(float %1852)
  %1854 = fsub float %1835, %1815
  %1855 = fmul float %1854, f0x3FB8AA3B
  %1856 = call float @llvm.amdgcn.exp2.f32(float %1855)
  %1857 = fsub float %1837, %1815
  %1858 = fmul float %1857, f0x3FB8AA3B
  %1859 = call float @llvm.amdgcn.exp2.f32(float %1858)
  %1860 = fsub float %1839, %1815
  %1861 = fmul float %1860, f0x3FB8AA3B
  %1862 = call float @llvm.amdgcn.exp2.f32(float %1861)
  %1863 = fsub float %1841, %1815
  %1864 = fmul float %1863, f0x3FB8AA3B
  %1865 = call float @llvm.amdgcn.exp2.f32(float %1864)
  %1866 = fptrunc float %1844 to bfloat
  %1867 = fptrunc float %1847 to bfloat
  %1868 = fptrunc float %1850 to bfloat
  %1869 = fptrunc float %1853 to bfloat
  %1870 = fptrunc float %1856 to bfloat
  %1871 = fptrunc float %1859 to bfloat
  %1872 = fptrunc float %1862 to bfloat
  %1873 = fptrunc float %1865 to bfloat
  %1874 = extractelement <8 x float> %1825, i64 0
  %1875 = fsub float %1874, %1817
  %1876 = fmul float %1844, %1875
  %1877 = fmul float %1876, %16
  %1878 = fptrunc float %1877 to bfloat
  %1879 = extractelement <8 x float> %1825, i64 1
  %1880 = fsub float %1879, %1817
  %1881 = fmul float %1847, %1880
  %1882 = fmul float %1881, %16
  %1883 = fptrunc float %1882 to bfloat
  %1884 = extractelement <8 x float> %1825, i64 2
  %1885 = fsub float %1884, %1817
  %1886 = fmul float %1850, %1885
  %1887 = fmul float %1886, %16
  %1888 = fptrunc float %1887 to bfloat
  %1889 = extractelement <8 x float> %1825, i64 3
  %1890 = fsub float %1889, %1817
  %1891 = fmul float %1853, %1890
  %1892 = fmul float %1891, %16
  %1893 = fptrunc float %1892 to bfloat
  %1894 = extractelement <8 x float> %1825, i64 4
  %1895 = fsub float %1894, %1817
  %1896 = fmul float %1856, %1895
  %1897 = fmul float %1896, %16
  %1898 = fptrunc float %1897 to bfloat
  %1899 = extractelement <8 x float> %1825, i64 5
  %1900 = fsub float %1899, %1817
  %1901 = fmul float %1859, %1900
  %1902 = fmul float %1901, %16
  %1903 = fptrunc float %1902 to bfloat
  %1904 = extractelement <8 x float> %1825, i64 6
  %1905 = fsub float %1904, %1817
  %1906 = fmul float %1862, %1905
  %1907 = fmul float %1906, %16
  %1908 = fptrunc float %1907 to bfloat
  %1909 = extractelement <8 x float> %1825, i64 7
  %1910 = fsub float %1909, %1817
  %1911 = fmul float %1865, %1910
  %1912 = fmul float %1911, %16
  %1913 = fptrunc float %1912 to bfloat
  %1914 = mul i32 %1762, 80
  %1915 = add i32 %1914, %1419
  %1916 = insertelement <8 x bfloat> poison, bfloat %1866, i64 0
  %1917 = insertelement <8 x bfloat> %1916, bfloat %1867, i64 1
  %1918 = insertelement <8 x bfloat> %1917, bfloat %1868, i64 2
  %1919 = insertelement <8 x bfloat> %1918, bfloat %1869, i64 3
  %1920 = insertelement <8 x bfloat> %1919, bfloat %1870, i64 4
  %1921 = insertelement <8 x bfloat> %1920, bfloat %1871, i64 5
  %1922 = insertelement <8 x bfloat> %1921, bfloat %1872, i64 6
  %1923 = insertelement <8 x bfloat> %1922, bfloat %1873, i64 7
  %1924 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %1915
  %1925 = inttoptr i32 %1924 to ptr addrspace(3)
  store <8 x bfloat> %1923, ptr addrspace(3) %1925, align 16
  %1926 = insertelement <8 x bfloat> poison, bfloat %1878, i64 0
  %1927 = insertelement <8 x bfloat> %1926, bfloat %1883, i64 1
  %1928 = insertelement <8 x bfloat> %1927, bfloat %1888, i64 2
  %1929 = insertelement <8 x bfloat> %1928, bfloat %1893, i64 3
  %1930 = insertelement <8 x bfloat> %1929, bfloat %1898, i64 4
  %1931 = insertelement <8 x bfloat> %1930, bfloat %1903, i64 5
  %1932 = insertelement <8 x bfloat> %1931, bfloat %1908, i64 6
  %1933 = insertelement <8 x bfloat> %1932, bfloat %1913, i64 7
  %1934 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %1915
  %1935 = inttoptr i32 %1934 to ptr addrspace(3)
  store <8 x bfloat> %1933, ptr addrspace(3) %1935, align 16
  %1936 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %126, <16 x bfloat> %1804, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1937 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %178, <16 x bfloat> %1808, i16 0, <8 x float> zeroinitializer, i1 false, i1 false)
  %1938 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %135, <16 x bfloat> %1805, i16 0, <8 x float> %1936, i1 false, i1 false)
  %1939 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %183, <16 x bfloat> %1809, i16 0, <8 x float> %1937, i1 false, i1 false)
  %1940 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %144, <16 x bfloat> %1806, i16 0, <8 x float> %1938, i1 false, i1 false)
  %1941 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %188, <16 x bfloat> %1810, i16 0, <8 x float> %1939, i1 false, i1 false)
  %1942 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %153, <16 x bfloat> %1807, i16 0, <8 x float> %1940, i1 false, i1 false)
  %1943 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %193, <16 x bfloat> %1811, i16 0, <8 x float> %1941, i1 false, i1 false)
  %1944 = extractelement <8 x float> %1942, i64 0
  %1945 = fmul float %1944, %16
  %1946 = extractelement <8 x float> %1942, i64 1
  %1947 = fmul float %1946, %16
  %1948 = extractelement <8 x float> %1942, i64 2
  %1949 = fmul float %1948, %16
  %1950 = extractelement <8 x float> %1942, i64 3
  %1951 = fmul float %1950, %16
  %1952 = extractelement <8 x float> %1942, i64 4
  %1953 = fmul float %1952, %16
  %1954 = extractelement <8 x float> %1942, i64 5
  %1955 = fmul float %1954, %16
  %1956 = extractelement <8 x float> %1942, i64 6
  %1957 = fmul float %1956, %16
  %1958 = extractelement <8 x float> %1942, i64 7
  %1959 = fmul float %1958, %16
  %1960 = fsub float %1945, %1815
  %1961 = fmul float %1960, f0x3FB8AA3B
  %1962 = call float @llvm.amdgcn.exp2.f32(float %1961)
  %1963 = fsub float %1947, %1815
  %1964 = fmul float %1963, f0x3FB8AA3B
  %1965 = call float @llvm.amdgcn.exp2.f32(float %1964)
  %1966 = fsub float %1949, %1815
  %1967 = fmul float %1966, f0x3FB8AA3B
  %1968 = call float @llvm.amdgcn.exp2.f32(float %1967)
  %1969 = fsub float %1951, %1815
  %1970 = fmul float %1969, f0x3FB8AA3B
  %1971 = call float @llvm.amdgcn.exp2.f32(float %1970)
  %1972 = fsub float %1953, %1815
  %1973 = fmul float %1972, f0x3FB8AA3B
  %1974 = call float @llvm.amdgcn.exp2.f32(float %1973)
  %1975 = fsub float %1955, %1815
  %1976 = fmul float %1975, f0x3FB8AA3B
  %1977 = call float @llvm.amdgcn.exp2.f32(float %1976)
  %1978 = fsub float %1957, %1815
  %1979 = fmul float %1978, f0x3FB8AA3B
  %1980 = call float @llvm.amdgcn.exp2.f32(float %1979)
  %1981 = fsub float %1959, %1815
  %1982 = fmul float %1981, f0x3FB8AA3B
  %1983 = call float @llvm.amdgcn.exp2.f32(float %1982)
  %1984 = fptrunc float %1962 to bfloat
  %1985 = fptrunc float %1965 to bfloat
  %1986 = fptrunc float %1968 to bfloat
  %1987 = fptrunc float %1971 to bfloat
  %1988 = fptrunc float %1974 to bfloat
  %1989 = fptrunc float %1977 to bfloat
  %1990 = fptrunc float %1980 to bfloat
  %1991 = fptrunc float %1983 to bfloat
  %1992 = extractelement <8 x float> %1943, i64 0
  %1993 = fsub float %1992, %1817
  %1994 = fmul float %1962, %1993
  %1995 = fmul float %1994, %16
  %1996 = fptrunc float %1995 to bfloat
  %1997 = extractelement <8 x float> %1943, i64 1
  %1998 = fsub float %1997, %1817
  %1999 = fmul float %1965, %1998
  %2000 = fmul float %1999, %16
  %2001 = fptrunc float %2000 to bfloat
  %2002 = extractelement <8 x float> %1943, i64 2
  %2003 = fsub float %2002, %1817
  %2004 = fmul float %1968, %2003
  %2005 = fmul float %2004, %16
  %2006 = fptrunc float %2005 to bfloat
  %2007 = extractelement <8 x float> %1943, i64 3
  %2008 = fsub float %2007, %1817
  %2009 = fmul float %1971, %2008
  %2010 = fmul float %2009, %16
  %2011 = fptrunc float %2010 to bfloat
  %2012 = extractelement <8 x float> %1943, i64 4
  %2013 = fsub float %2012, %1817
  %2014 = fmul float %1974, %2013
  %2015 = fmul float %2014, %16
  %2016 = fptrunc float %2015 to bfloat
  %2017 = extractelement <8 x float> %1943, i64 5
  %2018 = fsub float %2017, %1817
  %2019 = fmul float %1977, %2018
  %2020 = fmul float %2019, %16
  %2021 = fptrunc float %2020 to bfloat
  %2022 = extractelement <8 x float> %1943, i64 6
  %2023 = fsub float %2022, %1817
  %2024 = fmul float %1980, %2023
  %2025 = fmul float %2024, %16
  %2026 = fptrunc float %2025 to bfloat
  %2027 = extractelement <8 x float> %1943, i64 7
  %2028 = fsub float %2027, %1817
  %2029 = fmul float %1983, %2028
  %2030 = fmul float %2029, %16
  %2031 = fptrunc float %2030 to bfloat
  %2032 = add i32 %1914, 32
  %2033 = add i32 %2032, %1419
  %2034 = insertelement <8 x bfloat> poison, bfloat %1984, i64 0
  %2035 = insertelement <8 x bfloat> %2034, bfloat %1985, i64 1
  %2036 = insertelement <8 x bfloat> %2035, bfloat %1986, i64 2
  %2037 = insertelement <8 x bfloat> %2036, bfloat %1987, i64 3
  %2038 = insertelement <8 x bfloat> %2037, bfloat %1988, i64 4
  %2039 = insertelement <8 x bfloat> %2038, bfloat %1989, i64 5
  %2040 = insertelement <8 x bfloat> %2039, bfloat %1990, i64 6
  %2041 = insertelement <8 x bfloat> %2040, bfloat %1991, i64 7
  %2042 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %2033
  %2043 = inttoptr i32 %2042 to ptr addrspace(3)
  store <8 x bfloat> %2041, ptr addrspace(3) %2043, align 16
  %2044 = insertelement <8 x bfloat> poison, bfloat %1996, i64 0
  %2045 = insertelement <8 x bfloat> %2044, bfloat %2001, i64 1
  %2046 = insertelement <8 x bfloat> %2045, bfloat %2006, i64 2
  %2047 = insertelement <8 x bfloat> %2046, bfloat %2011, i64 3
  %2048 = insertelement <8 x bfloat> %2047, bfloat %2016, i64 4
  %2049 = insertelement <8 x bfloat> %2048, bfloat %2021, i64 5
  %2050 = insertelement <8 x bfloat> %2049, bfloat %2026, i64 6
  %2051 = insertelement <8 x bfloat> %2050, bfloat %2031, i64 7
  %2052 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %2033
  %2053 = inttoptr i32 %2052 to ptr addrspace(3)
  store <8 x bfloat> %2051, ptr addrspace(3) %2053, align 16
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %2054 = mul i32 %205, 16
  %2055 = mul i32 %196, 272
  %2056 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 5120), %2055
  %2057 = add i32 %2056, %2054
  %2058 = inttoptr i32 %2057 to ptr addrspace(3)
  %2059 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2058)
  %2060 = add i32 %2057, 4352
  %2061 = inttoptr i32 %2060 to ptr addrspace(3)
  %2062 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2061)
  %2063 = shufflevector <8 x bfloat> %2059, <8 x bfloat> %2062, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2064 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 13824), %2055
  %2065 = add i32 %2064, %2054
  %2066 = inttoptr i32 %2065 to ptr addrspace(3)
  %2067 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2066)
  %2068 = add i32 %2065, 4352
  %2069 = inttoptr i32 %2068 to ptr addrspace(3)
  %2070 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2069)
  %2071 = shufflevector <8 x bfloat> %2067, <8 x bfloat> %2070, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2072 = add i32 %206, 16
  %2073 = mul i32 %2072, 2
  %2074 = add i32 %2056, %2073
  %2075 = inttoptr i32 %2074 to ptr addrspace(3)
  %2076 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2075)
  %2077 = add i32 %2074, 4352
  %2078 = inttoptr i32 %2077 to ptr addrspace(3)
  %2079 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2078)
  %2080 = shufflevector <8 x bfloat> %2076, <8 x bfloat> %2079, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2081 = add i32 %2064, %2073
  %2082 = inttoptr i32 %2081 to ptr addrspace(3)
  %2083 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2082)
  %2084 = add i32 %2081, 4352
  %2085 = inttoptr i32 %2084 to ptr addrspace(3)
  %2086 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2085)
  %2087 = shufflevector <8 x bfloat> %2083, <8 x bfloat> %2086, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2088 = add i32 %206, 32
  %2089 = mul i32 %2088, 2
  %2090 = add i32 %2056, %2089
  %2091 = inttoptr i32 %2090 to ptr addrspace(3)
  %2092 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2091)
  %2093 = add i32 %2090, 4352
  %2094 = inttoptr i32 %2093 to ptr addrspace(3)
  %2095 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2094)
  %2096 = shufflevector <8 x bfloat> %2092, <8 x bfloat> %2095, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2097 = add i32 %2064, %2089
  %2098 = inttoptr i32 %2097 to ptr addrspace(3)
  %2099 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2098)
  %2100 = add i32 %2097, 4352
  %2101 = inttoptr i32 %2100 to ptr addrspace(3)
  %2102 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2101)
  %2103 = shufflevector <8 x bfloat> %2099, <8 x bfloat> %2102, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2104 = add i32 %206, 48
  %2105 = mul i32 %2104, 2
  %2106 = add i32 %2056, %2105
  %2107 = inttoptr i32 %2106 to ptr addrspace(3)
  %2108 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2107)
  %2109 = add i32 %2106, 4352
  %2110 = inttoptr i32 %2109 to ptr addrspace(3)
  %2111 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2110)
  %2112 = shufflevector <8 x bfloat> %2108, <8 x bfloat> %2111, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2113 = add i32 %2064, %2105
  %2114 = inttoptr i32 %2113 to ptr addrspace(3)
  %2115 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2114)
  %2116 = add i32 %2113, 4352
  %2117 = inttoptr i32 %2116 to ptr addrspace(3)
  %2118 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2117)
  %2119 = shufflevector <8 x bfloat> %2115, <8 x bfloat> %2118, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2120 = add i32 %206, 64
  %2121 = mul i32 %2120, 2
  %2122 = add i32 %2056, %2121
  %2123 = inttoptr i32 %2122 to ptr addrspace(3)
  %2124 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2123)
  %2125 = add i32 %2122, 4352
  %2126 = inttoptr i32 %2125 to ptr addrspace(3)
  %2127 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2126)
  %2128 = shufflevector <8 x bfloat> %2124, <8 x bfloat> %2127, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2129 = add i32 %2064, %2121
  %2130 = inttoptr i32 %2129 to ptr addrspace(3)
  %2131 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2130)
  %2132 = add i32 %2129, 4352
  %2133 = inttoptr i32 %2132 to ptr addrspace(3)
  %2134 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2133)
  %2135 = shufflevector <8 x bfloat> %2131, <8 x bfloat> %2134, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2136 = add i32 %206, 80
  %2137 = mul i32 %2136, 2
  %2138 = add i32 %2056, %2137
  %2139 = inttoptr i32 %2138 to ptr addrspace(3)
  %2140 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2139)
  %2141 = add i32 %2138, 4352
  %2142 = inttoptr i32 %2141 to ptr addrspace(3)
  %2143 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2142)
  %2144 = shufflevector <8 x bfloat> %2140, <8 x bfloat> %2143, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2145 = add i32 %2064, %2137
  %2146 = inttoptr i32 %2145 to ptr addrspace(3)
  %2147 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2146)
  %2148 = add i32 %2145, 4352
  %2149 = inttoptr i32 %2148 to ptr addrspace(3)
  %2150 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2149)
  %2151 = shufflevector <8 x bfloat> %2147, <8 x bfloat> %2150, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2152 = add i32 %206, 96
  %2153 = mul i32 %2152, 2
  %2154 = add i32 %2056, %2153
  %2155 = inttoptr i32 %2154 to ptr addrspace(3)
  %2156 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2155)
  %2157 = add i32 %2154, 4352
  %2158 = inttoptr i32 %2157 to ptr addrspace(3)
  %2159 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2158)
  %2160 = shufflevector <8 x bfloat> %2156, <8 x bfloat> %2159, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2161 = add i32 %2064, %2153
  %2162 = inttoptr i32 %2161 to ptr addrspace(3)
  %2163 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2162)
  %2164 = add i32 %2161, 4352
  %2165 = inttoptr i32 %2164 to ptr addrspace(3)
  %2166 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2165)
  %2167 = shufflevector <8 x bfloat> %2163, <8 x bfloat> %2166, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2168 = add i32 %206, 112
  %2169 = mul i32 %2168, 2
  %2170 = add i32 %2056, %2169
  %2171 = inttoptr i32 %2170 to ptr addrspace(3)
  %2172 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2171)
  %2173 = add i32 %2170, 4352
  %2174 = inttoptr i32 %2173 to ptr addrspace(3)
  %2175 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2174)
  %2176 = shufflevector <8 x bfloat> %2172, <8 x bfloat> %2175, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2177 = add i32 %2064, %2169
  %2178 = inttoptr i32 %2177 to ptr addrspace(3)
  %2179 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2178)
  %2180 = add i32 %2177, 4352
  %2181 = inttoptr i32 %2180 to ptr addrspace(3)
  %2182 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2181)
  %2183 = shufflevector <8 x bfloat> %2179, <8 x bfloat> %2182, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2184 = mul i32 %196, 80
  %2185 = add i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), %2184
  %2186 = add i32 %2185, %2054
  %2187 = inttoptr i32 %2186 to ptr addrspace(3)
  %2188 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2187)
  %2189 = add i32 %2186, 1280
  %2190 = inttoptr i32 %2189 to ptr addrspace(3)
  %2191 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2190)
  %2192 = shufflevector <8 x bfloat> %2188, <8 x bfloat> %2191, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2193 = add i32 add (i32 ptrtoint (ptr addrspace(3) @__shared_alloc_0 to i32), i32 2560), %2184
  %2194 = add i32 %2193, %2054
  %2195 = inttoptr i32 %2194 to ptr addrspace(3)
  %2196 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2195)
  %2197 = add i32 %2194, 1280
  %2198 = inttoptr i32 %2197 to ptr addrspace(3)
  %2199 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2198)
  %2200 = shufflevector <8 x bfloat> %2196, <8 x bfloat> %2199, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2201 = add i32 %2054, 32
  %2202 = add i32 %2185, %2201
  %2203 = inttoptr i32 %2202 to ptr addrspace(3)
  %2204 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2203)
  %2205 = add i32 %2202, 1280
  %2206 = inttoptr i32 %2205 to ptr addrspace(3)
  %2207 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2206)
  %2208 = shufflevector <8 x bfloat> %2204, <8 x bfloat> %2207, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2209 = add i32 %2193, %2201
  %2210 = inttoptr i32 %2209 to ptr addrspace(3)
  %2211 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2210)
  %2212 = add i32 %2209, 1280
  %2213 = inttoptr i32 %2212 to ptr addrspace(3)
  %2214 = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %2213)
  %2215 = shufflevector <8 x bfloat> %2211, <8 x bfloat> %2214, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %2216 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2192, <16 x bfloat> %2063, i16 0, <8 x float> %1310, i1 false, i1 false)
  %2217 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2200, <16 x bfloat> %2071, i16 0, <8 x float> %1326, i1 false, i1 false)
  %2218 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2192, <16 x bfloat> %2080, i16 0, <8 x float> %1311, i1 false, i1 false)
  %2219 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2200, <16 x bfloat> %2087, i16 0, <8 x float> %1327, i1 false, i1 false)
  %2220 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2192, <16 x bfloat> %2096, i16 0, <8 x float> %1312, i1 false, i1 false)
  %2221 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2200, <16 x bfloat> %2103, i16 0, <8 x float> %1328, i1 false, i1 false)
  %2222 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2192, <16 x bfloat> %2112, i16 0, <8 x float> %1313, i1 false, i1 false)
  %2223 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2200, <16 x bfloat> %2119, i16 0, <8 x float> %1329, i1 false, i1 false)
  %2224 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2192, <16 x bfloat> %2128, i16 0, <8 x float> %1314, i1 false, i1 false)
  %2225 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2200, <16 x bfloat> %2135, i16 0, <8 x float> %1330, i1 false, i1 false)
  %2226 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2192, <16 x bfloat> %2144, i16 0, <8 x float> %1315, i1 false, i1 false)
  %2227 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2200, <16 x bfloat> %2151, i16 0, <8 x float> %1331, i1 false, i1 false)
  %2228 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2192, <16 x bfloat> %2160, i16 0, <8 x float> %1316, i1 false, i1 false)
  %2229 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2200, <16 x bfloat> %2167, i16 0, <8 x float> %1332, i1 false, i1 false)
  %2230 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2192, <16 x bfloat> %2176, i16 0, <8 x float> %1317, i1 false, i1 false)
  %2231 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2200, <16 x bfloat> %2183, i16 0, <8 x float> %1333, i1 false, i1 false)
  %2232 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2208, <16 x bfloat> %2063, i16 0, <8 x float> %1318, i1 false, i1 false)
  %2233 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2215, <16 x bfloat> %2071, i16 0, <8 x float> %1334, i1 false, i1 false)
  %2234 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2208, <16 x bfloat> %2080, i16 0, <8 x float> %1319, i1 false, i1 false)
  %2235 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2215, <16 x bfloat> %2087, i16 0, <8 x float> %1335, i1 false, i1 false)
  %2236 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2208, <16 x bfloat> %2096, i16 0, <8 x float> %1320, i1 false, i1 false)
  %2237 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2215, <16 x bfloat> %2103, i16 0, <8 x float> %1336, i1 false, i1 false)
  %2238 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2208, <16 x bfloat> %2112, i16 0, <8 x float> %1321, i1 false, i1 false)
  %2239 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2215, <16 x bfloat> %2119, i16 0, <8 x float> %1337, i1 false, i1 false)
  %2240 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2208, <16 x bfloat> %2128, i16 0, <8 x float> %1322, i1 false, i1 false)
  %2241 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2215, <16 x bfloat> %2135, i16 0, <8 x float> %1338, i1 false, i1 false)
  %2242 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2208, <16 x bfloat> %2144, i16 0, <8 x float> %1323, i1 false, i1 false)
  %2243 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2215, <16 x bfloat> %2151, i16 0, <8 x float> %1339, i1 false, i1 false)
  %2244 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2208, <16 x bfloat> %2160, i16 0, <8 x float> %1324, i1 false, i1 false)
  %2245 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2215, <16 x bfloat> %2167, i16 0, <8 x float> %1340, i1 false, i1 false)
  %2246 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2208, <16 x bfloat> %2176, i16 0, <8 x float> %1325, i1 false, i1 false)
  %2247 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %2215, <16 x bfloat> %2183, i16 0, <8 x float> %1341, i1 false, i1 false)
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  fence syncscope("workgroup") acquire
  %2248 = add i64 %1309, 1
  br label %1308

2249:                                             ; preds = %1308
  %2250 = mul i32 %71, %20
  %2251 = mul i32 %2250, 128
  %2252 = mul i32 %32, 128
  %2253 = add i32 %2251, %2252
  %2254 = add i32 %48, %194
  %2255 = mul i32 %2254, %20
  %2256 = mul i32 %2255, 128
  %2257 = add i32 %2253, %2256
  %2258 = add i32 %2257, %39
  %2259 = extractelement <8 x float> %1310, i64 0
  %2260 = fptrunc float %2259 to bfloat
  %2261 = mul i32 %2258, 4
  %2262 = bitcast bfloat %2260 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2262, ptr addrspace(8) %67, i32 %2261, i32 0, i32 0)
  %2263 = extractelement <8 x float> %1326, i64 0
  %2264 = fptrunc float %2263 to bfloat
  %2265 = bitcast bfloat %2264 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2265, ptr addrspace(8) %68, i32 %2261, i32 0, i32 0)
  %2266 = add i32 %2254, 1
  %2267 = mul i32 %2266, %20
  %2268 = mul i32 %2267, 128
  %2269 = add i32 %2253, %2268
  %2270 = add i32 %2269, %39
  %2271 = extractelement <8 x float> %1310, i64 1
  %2272 = fptrunc float %2271 to bfloat
  %2273 = mul i32 %2270, 4
  %2274 = bitcast bfloat %2272 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2274, ptr addrspace(8) %67, i32 %2273, i32 0, i32 0)
  %2275 = extractelement <8 x float> %1326, i64 1
  %2276 = fptrunc float %2275 to bfloat
  %2277 = bitcast bfloat %2276 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2277, ptr addrspace(8) %68, i32 %2273, i32 0, i32 0)
  %2278 = add i32 %2254, 2
  %2279 = mul i32 %2278, %20
  %2280 = mul i32 %2279, 128
  %2281 = add i32 %2253, %2280
  %2282 = add i32 %2281, %39
  %2283 = extractelement <8 x float> %1310, i64 2
  %2284 = fptrunc float %2283 to bfloat
  %2285 = mul i32 %2282, 4
  %2286 = bitcast bfloat %2284 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2286, ptr addrspace(8) %67, i32 %2285, i32 0, i32 0)
  %2287 = extractelement <8 x float> %1326, i64 2
  %2288 = fptrunc float %2287 to bfloat
  %2289 = bitcast bfloat %2288 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2289, ptr addrspace(8) %68, i32 %2285, i32 0, i32 0)
  %2290 = add i32 %2254, 3
  %2291 = mul i32 %2290, %20
  %2292 = mul i32 %2291, 128
  %2293 = add i32 %2253, %2292
  %2294 = add i32 %2293, %39
  %2295 = extractelement <8 x float> %1310, i64 3
  %2296 = fptrunc float %2295 to bfloat
  %2297 = mul i32 %2294, 4
  %2298 = bitcast bfloat %2296 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2298, ptr addrspace(8) %67, i32 %2297, i32 0, i32 0)
  %2299 = extractelement <8 x float> %1326, i64 3
  %2300 = fptrunc float %2299 to bfloat
  %2301 = bitcast bfloat %2300 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2301, ptr addrspace(8) %68, i32 %2297, i32 0, i32 0)
  %2302 = add i32 %2254, 4
  %2303 = mul i32 %2302, %20
  %2304 = mul i32 %2303, 128
  %2305 = add i32 %2253, %2304
  %2306 = add i32 %2305, %39
  %2307 = extractelement <8 x float> %1310, i64 4
  %2308 = fptrunc float %2307 to bfloat
  %2309 = mul i32 %2306, 4
  %2310 = bitcast bfloat %2308 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2310, ptr addrspace(8) %67, i32 %2309, i32 0, i32 0)
  %2311 = extractelement <8 x float> %1326, i64 4
  %2312 = fptrunc float %2311 to bfloat
  %2313 = bitcast bfloat %2312 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2313, ptr addrspace(8) %68, i32 %2309, i32 0, i32 0)
  %2314 = add i32 %2254, 5
  %2315 = mul i32 %2314, %20
  %2316 = mul i32 %2315, 128
  %2317 = add i32 %2253, %2316
  %2318 = add i32 %2317, %39
  %2319 = extractelement <8 x float> %1310, i64 5
  %2320 = fptrunc float %2319 to bfloat
  %2321 = mul i32 %2318, 4
  %2322 = bitcast bfloat %2320 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2322, ptr addrspace(8) %67, i32 %2321, i32 0, i32 0)
  %2323 = extractelement <8 x float> %1326, i64 5
  %2324 = fptrunc float %2323 to bfloat
  %2325 = bitcast bfloat %2324 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2325, ptr addrspace(8) %68, i32 %2321, i32 0, i32 0)
  %2326 = add i32 %2254, 6
  %2327 = mul i32 %2326, %20
  %2328 = mul i32 %2327, 128
  %2329 = add i32 %2253, %2328
  %2330 = add i32 %2329, %39
  %2331 = extractelement <8 x float> %1310, i64 6
  %2332 = fptrunc float %2331 to bfloat
  %2333 = mul i32 %2330, 4
  %2334 = bitcast bfloat %2332 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2334, ptr addrspace(8) %67, i32 %2333, i32 0, i32 0)
  %2335 = extractelement <8 x float> %1326, i64 6
  %2336 = fptrunc float %2335 to bfloat
  %2337 = bitcast bfloat %2336 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2337, ptr addrspace(8) %68, i32 %2333, i32 0, i32 0)
  %2338 = add i32 %2254, 7
  %2339 = mul i32 %2338, %20
  %2340 = mul i32 %2339, 128
  %2341 = add i32 %2253, %2340
  %2342 = add i32 %2341, %39
  %2343 = extractelement <8 x float> %1310, i64 7
  %2344 = fptrunc float %2343 to bfloat
  %2345 = mul i32 %2342, 4
  %2346 = bitcast bfloat %2344 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2346, ptr addrspace(8) %67, i32 %2345, i32 0, i32 0)
  %2347 = extractelement <8 x float> %1326, i64 7
  %2348 = fptrunc float %2347 to bfloat
  %2349 = bitcast bfloat %2348 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2349, ptr addrspace(8) %68, i32 %2345, i32 0, i32 0)
  %2350 = add i32 %2257, 16
  %2351 = add i32 %2350, %39
  %2352 = extractelement <8 x float> %1311, i64 0
  %2353 = fptrunc float %2352 to bfloat
  %2354 = mul i32 %2351, 4
  %2355 = bitcast bfloat %2353 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2355, ptr addrspace(8) %67, i32 %2354, i32 0, i32 0)
  %2356 = extractelement <8 x float> %1327, i64 0
  %2357 = fptrunc float %2356 to bfloat
  %2358 = bitcast bfloat %2357 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2358, ptr addrspace(8) %68, i32 %2354, i32 0, i32 0)
  %2359 = add i32 %2269, 16
  %2360 = add i32 %2359, %39
  %2361 = extractelement <8 x float> %1311, i64 1
  %2362 = fptrunc float %2361 to bfloat
  %2363 = mul i32 %2360, 4
  %2364 = bitcast bfloat %2362 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2364, ptr addrspace(8) %67, i32 %2363, i32 0, i32 0)
  %2365 = extractelement <8 x float> %1327, i64 1
  %2366 = fptrunc float %2365 to bfloat
  %2367 = bitcast bfloat %2366 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2367, ptr addrspace(8) %68, i32 %2363, i32 0, i32 0)
  %2368 = add i32 %2281, 16
  %2369 = add i32 %2368, %39
  %2370 = extractelement <8 x float> %1311, i64 2
  %2371 = fptrunc float %2370 to bfloat
  %2372 = mul i32 %2369, 4
  %2373 = bitcast bfloat %2371 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2373, ptr addrspace(8) %67, i32 %2372, i32 0, i32 0)
  %2374 = extractelement <8 x float> %1327, i64 2
  %2375 = fptrunc float %2374 to bfloat
  %2376 = bitcast bfloat %2375 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2376, ptr addrspace(8) %68, i32 %2372, i32 0, i32 0)
  %2377 = add i32 %2293, 16
  %2378 = add i32 %2377, %39
  %2379 = extractelement <8 x float> %1311, i64 3
  %2380 = fptrunc float %2379 to bfloat
  %2381 = mul i32 %2378, 4
  %2382 = bitcast bfloat %2380 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2382, ptr addrspace(8) %67, i32 %2381, i32 0, i32 0)
  %2383 = extractelement <8 x float> %1327, i64 3
  %2384 = fptrunc float %2383 to bfloat
  %2385 = bitcast bfloat %2384 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2385, ptr addrspace(8) %68, i32 %2381, i32 0, i32 0)
  %2386 = add i32 %2305, 16
  %2387 = add i32 %2386, %39
  %2388 = extractelement <8 x float> %1311, i64 4
  %2389 = fptrunc float %2388 to bfloat
  %2390 = mul i32 %2387, 4
  %2391 = bitcast bfloat %2389 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2391, ptr addrspace(8) %67, i32 %2390, i32 0, i32 0)
  %2392 = extractelement <8 x float> %1327, i64 4
  %2393 = fptrunc float %2392 to bfloat
  %2394 = bitcast bfloat %2393 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2394, ptr addrspace(8) %68, i32 %2390, i32 0, i32 0)
  %2395 = add i32 %2317, 16
  %2396 = add i32 %2395, %39
  %2397 = extractelement <8 x float> %1311, i64 5
  %2398 = fptrunc float %2397 to bfloat
  %2399 = mul i32 %2396, 4
  %2400 = bitcast bfloat %2398 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2400, ptr addrspace(8) %67, i32 %2399, i32 0, i32 0)
  %2401 = extractelement <8 x float> %1327, i64 5
  %2402 = fptrunc float %2401 to bfloat
  %2403 = bitcast bfloat %2402 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2403, ptr addrspace(8) %68, i32 %2399, i32 0, i32 0)
  %2404 = add i32 %2329, 16
  %2405 = add i32 %2404, %39
  %2406 = extractelement <8 x float> %1311, i64 6
  %2407 = fptrunc float %2406 to bfloat
  %2408 = mul i32 %2405, 4
  %2409 = bitcast bfloat %2407 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2409, ptr addrspace(8) %67, i32 %2408, i32 0, i32 0)
  %2410 = extractelement <8 x float> %1327, i64 6
  %2411 = fptrunc float %2410 to bfloat
  %2412 = bitcast bfloat %2411 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2412, ptr addrspace(8) %68, i32 %2408, i32 0, i32 0)
  %2413 = add i32 %2341, 16
  %2414 = add i32 %2413, %39
  %2415 = extractelement <8 x float> %1311, i64 7
  %2416 = fptrunc float %2415 to bfloat
  %2417 = mul i32 %2414, 4
  %2418 = bitcast bfloat %2416 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2418, ptr addrspace(8) %67, i32 %2417, i32 0, i32 0)
  %2419 = extractelement <8 x float> %1327, i64 7
  %2420 = fptrunc float %2419 to bfloat
  %2421 = bitcast bfloat %2420 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2421, ptr addrspace(8) %68, i32 %2417, i32 0, i32 0)
  %2422 = add i32 %2257, 32
  %2423 = add i32 %2422, %39
  %2424 = extractelement <8 x float> %1312, i64 0
  %2425 = fptrunc float %2424 to bfloat
  %2426 = mul i32 %2423, 4
  %2427 = bitcast bfloat %2425 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2427, ptr addrspace(8) %67, i32 %2426, i32 0, i32 0)
  %2428 = extractelement <8 x float> %1328, i64 0
  %2429 = fptrunc float %2428 to bfloat
  %2430 = bitcast bfloat %2429 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2430, ptr addrspace(8) %68, i32 %2426, i32 0, i32 0)
  %2431 = add i32 %2269, 32
  %2432 = add i32 %2431, %39
  %2433 = extractelement <8 x float> %1312, i64 1
  %2434 = fptrunc float %2433 to bfloat
  %2435 = mul i32 %2432, 4
  %2436 = bitcast bfloat %2434 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2436, ptr addrspace(8) %67, i32 %2435, i32 0, i32 0)
  %2437 = extractelement <8 x float> %1328, i64 1
  %2438 = fptrunc float %2437 to bfloat
  %2439 = bitcast bfloat %2438 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2439, ptr addrspace(8) %68, i32 %2435, i32 0, i32 0)
  %2440 = add i32 %2281, 32
  %2441 = add i32 %2440, %39
  %2442 = extractelement <8 x float> %1312, i64 2
  %2443 = fptrunc float %2442 to bfloat
  %2444 = mul i32 %2441, 4
  %2445 = bitcast bfloat %2443 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2445, ptr addrspace(8) %67, i32 %2444, i32 0, i32 0)
  %2446 = extractelement <8 x float> %1328, i64 2
  %2447 = fptrunc float %2446 to bfloat
  %2448 = bitcast bfloat %2447 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2448, ptr addrspace(8) %68, i32 %2444, i32 0, i32 0)
  %2449 = add i32 %2293, 32
  %2450 = add i32 %2449, %39
  %2451 = extractelement <8 x float> %1312, i64 3
  %2452 = fptrunc float %2451 to bfloat
  %2453 = mul i32 %2450, 4
  %2454 = bitcast bfloat %2452 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2454, ptr addrspace(8) %67, i32 %2453, i32 0, i32 0)
  %2455 = extractelement <8 x float> %1328, i64 3
  %2456 = fptrunc float %2455 to bfloat
  %2457 = bitcast bfloat %2456 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2457, ptr addrspace(8) %68, i32 %2453, i32 0, i32 0)
  %2458 = add i32 %2305, 32
  %2459 = add i32 %2458, %39
  %2460 = extractelement <8 x float> %1312, i64 4
  %2461 = fptrunc float %2460 to bfloat
  %2462 = mul i32 %2459, 4
  %2463 = bitcast bfloat %2461 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2463, ptr addrspace(8) %67, i32 %2462, i32 0, i32 0)
  %2464 = extractelement <8 x float> %1328, i64 4
  %2465 = fptrunc float %2464 to bfloat
  %2466 = bitcast bfloat %2465 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2466, ptr addrspace(8) %68, i32 %2462, i32 0, i32 0)
  %2467 = add i32 %2317, 32
  %2468 = add i32 %2467, %39
  %2469 = extractelement <8 x float> %1312, i64 5
  %2470 = fptrunc float %2469 to bfloat
  %2471 = mul i32 %2468, 4
  %2472 = bitcast bfloat %2470 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2472, ptr addrspace(8) %67, i32 %2471, i32 0, i32 0)
  %2473 = extractelement <8 x float> %1328, i64 5
  %2474 = fptrunc float %2473 to bfloat
  %2475 = bitcast bfloat %2474 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2475, ptr addrspace(8) %68, i32 %2471, i32 0, i32 0)
  %2476 = add i32 %2329, 32
  %2477 = add i32 %2476, %39
  %2478 = extractelement <8 x float> %1312, i64 6
  %2479 = fptrunc float %2478 to bfloat
  %2480 = mul i32 %2477, 4
  %2481 = bitcast bfloat %2479 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2481, ptr addrspace(8) %67, i32 %2480, i32 0, i32 0)
  %2482 = extractelement <8 x float> %1328, i64 6
  %2483 = fptrunc float %2482 to bfloat
  %2484 = bitcast bfloat %2483 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2484, ptr addrspace(8) %68, i32 %2480, i32 0, i32 0)
  %2485 = add i32 %2341, 32
  %2486 = add i32 %2485, %39
  %2487 = extractelement <8 x float> %1312, i64 7
  %2488 = fptrunc float %2487 to bfloat
  %2489 = mul i32 %2486, 4
  %2490 = bitcast bfloat %2488 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2490, ptr addrspace(8) %67, i32 %2489, i32 0, i32 0)
  %2491 = extractelement <8 x float> %1328, i64 7
  %2492 = fptrunc float %2491 to bfloat
  %2493 = bitcast bfloat %2492 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2493, ptr addrspace(8) %68, i32 %2489, i32 0, i32 0)
  %2494 = add i32 %2257, 48
  %2495 = add i32 %2494, %39
  %2496 = extractelement <8 x float> %1313, i64 0
  %2497 = fptrunc float %2496 to bfloat
  %2498 = mul i32 %2495, 4
  %2499 = bitcast bfloat %2497 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2499, ptr addrspace(8) %67, i32 %2498, i32 0, i32 0)
  %2500 = extractelement <8 x float> %1329, i64 0
  %2501 = fptrunc float %2500 to bfloat
  %2502 = bitcast bfloat %2501 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2502, ptr addrspace(8) %68, i32 %2498, i32 0, i32 0)
  %2503 = add i32 %2269, 48
  %2504 = add i32 %2503, %39
  %2505 = extractelement <8 x float> %1313, i64 1
  %2506 = fptrunc float %2505 to bfloat
  %2507 = mul i32 %2504, 4
  %2508 = bitcast bfloat %2506 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2508, ptr addrspace(8) %67, i32 %2507, i32 0, i32 0)
  %2509 = extractelement <8 x float> %1329, i64 1
  %2510 = fptrunc float %2509 to bfloat
  %2511 = bitcast bfloat %2510 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2511, ptr addrspace(8) %68, i32 %2507, i32 0, i32 0)
  %2512 = add i32 %2281, 48
  %2513 = add i32 %2512, %39
  %2514 = extractelement <8 x float> %1313, i64 2
  %2515 = fptrunc float %2514 to bfloat
  %2516 = mul i32 %2513, 4
  %2517 = bitcast bfloat %2515 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2517, ptr addrspace(8) %67, i32 %2516, i32 0, i32 0)
  %2518 = extractelement <8 x float> %1329, i64 2
  %2519 = fptrunc float %2518 to bfloat
  %2520 = bitcast bfloat %2519 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2520, ptr addrspace(8) %68, i32 %2516, i32 0, i32 0)
  %2521 = add i32 %2293, 48
  %2522 = add i32 %2521, %39
  %2523 = extractelement <8 x float> %1313, i64 3
  %2524 = fptrunc float %2523 to bfloat
  %2525 = mul i32 %2522, 4
  %2526 = bitcast bfloat %2524 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2526, ptr addrspace(8) %67, i32 %2525, i32 0, i32 0)
  %2527 = extractelement <8 x float> %1329, i64 3
  %2528 = fptrunc float %2527 to bfloat
  %2529 = bitcast bfloat %2528 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2529, ptr addrspace(8) %68, i32 %2525, i32 0, i32 0)
  %2530 = add i32 %2305, 48
  %2531 = add i32 %2530, %39
  %2532 = extractelement <8 x float> %1313, i64 4
  %2533 = fptrunc float %2532 to bfloat
  %2534 = mul i32 %2531, 4
  %2535 = bitcast bfloat %2533 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2535, ptr addrspace(8) %67, i32 %2534, i32 0, i32 0)
  %2536 = extractelement <8 x float> %1329, i64 4
  %2537 = fptrunc float %2536 to bfloat
  %2538 = bitcast bfloat %2537 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2538, ptr addrspace(8) %68, i32 %2534, i32 0, i32 0)
  %2539 = add i32 %2317, 48
  %2540 = add i32 %2539, %39
  %2541 = extractelement <8 x float> %1313, i64 5
  %2542 = fptrunc float %2541 to bfloat
  %2543 = mul i32 %2540, 4
  %2544 = bitcast bfloat %2542 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2544, ptr addrspace(8) %67, i32 %2543, i32 0, i32 0)
  %2545 = extractelement <8 x float> %1329, i64 5
  %2546 = fptrunc float %2545 to bfloat
  %2547 = bitcast bfloat %2546 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2547, ptr addrspace(8) %68, i32 %2543, i32 0, i32 0)
  %2548 = add i32 %2329, 48
  %2549 = add i32 %2548, %39
  %2550 = extractelement <8 x float> %1313, i64 6
  %2551 = fptrunc float %2550 to bfloat
  %2552 = mul i32 %2549, 4
  %2553 = bitcast bfloat %2551 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2553, ptr addrspace(8) %67, i32 %2552, i32 0, i32 0)
  %2554 = extractelement <8 x float> %1329, i64 6
  %2555 = fptrunc float %2554 to bfloat
  %2556 = bitcast bfloat %2555 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2556, ptr addrspace(8) %68, i32 %2552, i32 0, i32 0)
  %2557 = add i32 %2341, 48
  %2558 = add i32 %2557, %39
  %2559 = extractelement <8 x float> %1313, i64 7
  %2560 = fptrunc float %2559 to bfloat
  %2561 = mul i32 %2558, 4
  %2562 = bitcast bfloat %2560 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2562, ptr addrspace(8) %67, i32 %2561, i32 0, i32 0)
  %2563 = extractelement <8 x float> %1329, i64 7
  %2564 = fptrunc float %2563 to bfloat
  %2565 = bitcast bfloat %2564 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2565, ptr addrspace(8) %68, i32 %2561, i32 0, i32 0)
  %2566 = add i32 %2257, 64
  %2567 = add i32 %2566, %39
  %2568 = extractelement <8 x float> %1314, i64 0
  %2569 = fptrunc float %2568 to bfloat
  %2570 = mul i32 %2567, 4
  %2571 = bitcast bfloat %2569 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2571, ptr addrspace(8) %67, i32 %2570, i32 0, i32 0)
  %2572 = extractelement <8 x float> %1330, i64 0
  %2573 = fptrunc float %2572 to bfloat
  %2574 = bitcast bfloat %2573 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2574, ptr addrspace(8) %68, i32 %2570, i32 0, i32 0)
  %2575 = add i32 %2269, 64
  %2576 = add i32 %2575, %39
  %2577 = extractelement <8 x float> %1314, i64 1
  %2578 = fptrunc float %2577 to bfloat
  %2579 = mul i32 %2576, 4
  %2580 = bitcast bfloat %2578 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2580, ptr addrspace(8) %67, i32 %2579, i32 0, i32 0)
  %2581 = extractelement <8 x float> %1330, i64 1
  %2582 = fptrunc float %2581 to bfloat
  %2583 = bitcast bfloat %2582 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2583, ptr addrspace(8) %68, i32 %2579, i32 0, i32 0)
  %2584 = add i32 %2281, 64
  %2585 = add i32 %2584, %39
  %2586 = extractelement <8 x float> %1314, i64 2
  %2587 = fptrunc float %2586 to bfloat
  %2588 = mul i32 %2585, 4
  %2589 = bitcast bfloat %2587 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2589, ptr addrspace(8) %67, i32 %2588, i32 0, i32 0)
  %2590 = extractelement <8 x float> %1330, i64 2
  %2591 = fptrunc float %2590 to bfloat
  %2592 = bitcast bfloat %2591 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2592, ptr addrspace(8) %68, i32 %2588, i32 0, i32 0)
  %2593 = add i32 %2293, 64
  %2594 = add i32 %2593, %39
  %2595 = extractelement <8 x float> %1314, i64 3
  %2596 = fptrunc float %2595 to bfloat
  %2597 = mul i32 %2594, 4
  %2598 = bitcast bfloat %2596 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2598, ptr addrspace(8) %67, i32 %2597, i32 0, i32 0)
  %2599 = extractelement <8 x float> %1330, i64 3
  %2600 = fptrunc float %2599 to bfloat
  %2601 = bitcast bfloat %2600 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2601, ptr addrspace(8) %68, i32 %2597, i32 0, i32 0)
  %2602 = add i32 %2305, 64
  %2603 = add i32 %2602, %39
  %2604 = extractelement <8 x float> %1314, i64 4
  %2605 = fptrunc float %2604 to bfloat
  %2606 = mul i32 %2603, 4
  %2607 = bitcast bfloat %2605 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2607, ptr addrspace(8) %67, i32 %2606, i32 0, i32 0)
  %2608 = extractelement <8 x float> %1330, i64 4
  %2609 = fptrunc float %2608 to bfloat
  %2610 = bitcast bfloat %2609 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2610, ptr addrspace(8) %68, i32 %2606, i32 0, i32 0)
  %2611 = add i32 %2317, 64
  %2612 = add i32 %2611, %39
  %2613 = extractelement <8 x float> %1314, i64 5
  %2614 = fptrunc float %2613 to bfloat
  %2615 = mul i32 %2612, 4
  %2616 = bitcast bfloat %2614 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2616, ptr addrspace(8) %67, i32 %2615, i32 0, i32 0)
  %2617 = extractelement <8 x float> %1330, i64 5
  %2618 = fptrunc float %2617 to bfloat
  %2619 = bitcast bfloat %2618 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2619, ptr addrspace(8) %68, i32 %2615, i32 0, i32 0)
  %2620 = add i32 %2329, 64
  %2621 = add i32 %2620, %39
  %2622 = extractelement <8 x float> %1314, i64 6
  %2623 = fptrunc float %2622 to bfloat
  %2624 = mul i32 %2621, 4
  %2625 = bitcast bfloat %2623 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2625, ptr addrspace(8) %67, i32 %2624, i32 0, i32 0)
  %2626 = extractelement <8 x float> %1330, i64 6
  %2627 = fptrunc float %2626 to bfloat
  %2628 = bitcast bfloat %2627 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2628, ptr addrspace(8) %68, i32 %2624, i32 0, i32 0)
  %2629 = add i32 %2341, 64
  %2630 = add i32 %2629, %39
  %2631 = extractelement <8 x float> %1314, i64 7
  %2632 = fptrunc float %2631 to bfloat
  %2633 = mul i32 %2630, 4
  %2634 = bitcast bfloat %2632 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2634, ptr addrspace(8) %67, i32 %2633, i32 0, i32 0)
  %2635 = extractelement <8 x float> %1330, i64 7
  %2636 = fptrunc float %2635 to bfloat
  %2637 = bitcast bfloat %2636 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2637, ptr addrspace(8) %68, i32 %2633, i32 0, i32 0)
  %2638 = add i32 %2257, 80
  %2639 = add i32 %2638, %39
  %2640 = extractelement <8 x float> %1315, i64 0
  %2641 = fptrunc float %2640 to bfloat
  %2642 = mul i32 %2639, 4
  %2643 = bitcast bfloat %2641 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2643, ptr addrspace(8) %67, i32 %2642, i32 0, i32 0)
  %2644 = extractelement <8 x float> %1331, i64 0
  %2645 = fptrunc float %2644 to bfloat
  %2646 = bitcast bfloat %2645 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2646, ptr addrspace(8) %68, i32 %2642, i32 0, i32 0)
  %2647 = add i32 %2269, 80
  %2648 = add i32 %2647, %39
  %2649 = extractelement <8 x float> %1315, i64 1
  %2650 = fptrunc float %2649 to bfloat
  %2651 = mul i32 %2648, 4
  %2652 = bitcast bfloat %2650 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2652, ptr addrspace(8) %67, i32 %2651, i32 0, i32 0)
  %2653 = extractelement <8 x float> %1331, i64 1
  %2654 = fptrunc float %2653 to bfloat
  %2655 = bitcast bfloat %2654 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2655, ptr addrspace(8) %68, i32 %2651, i32 0, i32 0)
  %2656 = add i32 %2281, 80
  %2657 = add i32 %2656, %39
  %2658 = extractelement <8 x float> %1315, i64 2
  %2659 = fptrunc float %2658 to bfloat
  %2660 = mul i32 %2657, 4
  %2661 = bitcast bfloat %2659 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2661, ptr addrspace(8) %67, i32 %2660, i32 0, i32 0)
  %2662 = extractelement <8 x float> %1331, i64 2
  %2663 = fptrunc float %2662 to bfloat
  %2664 = bitcast bfloat %2663 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2664, ptr addrspace(8) %68, i32 %2660, i32 0, i32 0)
  %2665 = add i32 %2293, 80
  %2666 = add i32 %2665, %39
  %2667 = extractelement <8 x float> %1315, i64 3
  %2668 = fptrunc float %2667 to bfloat
  %2669 = mul i32 %2666, 4
  %2670 = bitcast bfloat %2668 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2670, ptr addrspace(8) %67, i32 %2669, i32 0, i32 0)
  %2671 = extractelement <8 x float> %1331, i64 3
  %2672 = fptrunc float %2671 to bfloat
  %2673 = bitcast bfloat %2672 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2673, ptr addrspace(8) %68, i32 %2669, i32 0, i32 0)
  %2674 = add i32 %2305, 80
  %2675 = add i32 %2674, %39
  %2676 = extractelement <8 x float> %1315, i64 4
  %2677 = fptrunc float %2676 to bfloat
  %2678 = mul i32 %2675, 4
  %2679 = bitcast bfloat %2677 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2679, ptr addrspace(8) %67, i32 %2678, i32 0, i32 0)
  %2680 = extractelement <8 x float> %1331, i64 4
  %2681 = fptrunc float %2680 to bfloat
  %2682 = bitcast bfloat %2681 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2682, ptr addrspace(8) %68, i32 %2678, i32 0, i32 0)
  %2683 = add i32 %2317, 80
  %2684 = add i32 %2683, %39
  %2685 = extractelement <8 x float> %1315, i64 5
  %2686 = fptrunc float %2685 to bfloat
  %2687 = mul i32 %2684, 4
  %2688 = bitcast bfloat %2686 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2688, ptr addrspace(8) %67, i32 %2687, i32 0, i32 0)
  %2689 = extractelement <8 x float> %1331, i64 5
  %2690 = fptrunc float %2689 to bfloat
  %2691 = bitcast bfloat %2690 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2691, ptr addrspace(8) %68, i32 %2687, i32 0, i32 0)
  %2692 = add i32 %2329, 80
  %2693 = add i32 %2692, %39
  %2694 = extractelement <8 x float> %1315, i64 6
  %2695 = fptrunc float %2694 to bfloat
  %2696 = mul i32 %2693, 4
  %2697 = bitcast bfloat %2695 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2697, ptr addrspace(8) %67, i32 %2696, i32 0, i32 0)
  %2698 = extractelement <8 x float> %1331, i64 6
  %2699 = fptrunc float %2698 to bfloat
  %2700 = bitcast bfloat %2699 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2700, ptr addrspace(8) %68, i32 %2696, i32 0, i32 0)
  %2701 = add i32 %2341, 80
  %2702 = add i32 %2701, %39
  %2703 = extractelement <8 x float> %1315, i64 7
  %2704 = fptrunc float %2703 to bfloat
  %2705 = mul i32 %2702, 4
  %2706 = bitcast bfloat %2704 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2706, ptr addrspace(8) %67, i32 %2705, i32 0, i32 0)
  %2707 = extractelement <8 x float> %1331, i64 7
  %2708 = fptrunc float %2707 to bfloat
  %2709 = bitcast bfloat %2708 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2709, ptr addrspace(8) %68, i32 %2705, i32 0, i32 0)
  %2710 = add i32 %2257, 96
  %2711 = add i32 %2710, %39
  %2712 = extractelement <8 x float> %1316, i64 0
  %2713 = fptrunc float %2712 to bfloat
  %2714 = mul i32 %2711, 4
  %2715 = bitcast bfloat %2713 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2715, ptr addrspace(8) %67, i32 %2714, i32 0, i32 0)
  %2716 = extractelement <8 x float> %1332, i64 0
  %2717 = fptrunc float %2716 to bfloat
  %2718 = bitcast bfloat %2717 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2718, ptr addrspace(8) %68, i32 %2714, i32 0, i32 0)
  %2719 = add i32 %2269, 96
  %2720 = add i32 %2719, %39
  %2721 = extractelement <8 x float> %1316, i64 1
  %2722 = fptrunc float %2721 to bfloat
  %2723 = mul i32 %2720, 4
  %2724 = bitcast bfloat %2722 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2724, ptr addrspace(8) %67, i32 %2723, i32 0, i32 0)
  %2725 = extractelement <8 x float> %1332, i64 1
  %2726 = fptrunc float %2725 to bfloat
  %2727 = bitcast bfloat %2726 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2727, ptr addrspace(8) %68, i32 %2723, i32 0, i32 0)
  %2728 = add i32 %2281, 96
  %2729 = add i32 %2728, %39
  %2730 = extractelement <8 x float> %1316, i64 2
  %2731 = fptrunc float %2730 to bfloat
  %2732 = mul i32 %2729, 4
  %2733 = bitcast bfloat %2731 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2733, ptr addrspace(8) %67, i32 %2732, i32 0, i32 0)
  %2734 = extractelement <8 x float> %1332, i64 2
  %2735 = fptrunc float %2734 to bfloat
  %2736 = bitcast bfloat %2735 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2736, ptr addrspace(8) %68, i32 %2732, i32 0, i32 0)
  %2737 = add i32 %2293, 96
  %2738 = add i32 %2737, %39
  %2739 = extractelement <8 x float> %1316, i64 3
  %2740 = fptrunc float %2739 to bfloat
  %2741 = mul i32 %2738, 4
  %2742 = bitcast bfloat %2740 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2742, ptr addrspace(8) %67, i32 %2741, i32 0, i32 0)
  %2743 = extractelement <8 x float> %1332, i64 3
  %2744 = fptrunc float %2743 to bfloat
  %2745 = bitcast bfloat %2744 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2745, ptr addrspace(8) %68, i32 %2741, i32 0, i32 0)
  %2746 = add i32 %2305, 96
  %2747 = add i32 %2746, %39
  %2748 = extractelement <8 x float> %1316, i64 4
  %2749 = fptrunc float %2748 to bfloat
  %2750 = mul i32 %2747, 4
  %2751 = bitcast bfloat %2749 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2751, ptr addrspace(8) %67, i32 %2750, i32 0, i32 0)
  %2752 = extractelement <8 x float> %1332, i64 4
  %2753 = fptrunc float %2752 to bfloat
  %2754 = bitcast bfloat %2753 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2754, ptr addrspace(8) %68, i32 %2750, i32 0, i32 0)
  %2755 = add i32 %2317, 96
  %2756 = add i32 %2755, %39
  %2757 = extractelement <8 x float> %1316, i64 5
  %2758 = fptrunc float %2757 to bfloat
  %2759 = mul i32 %2756, 4
  %2760 = bitcast bfloat %2758 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2760, ptr addrspace(8) %67, i32 %2759, i32 0, i32 0)
  %2761 = extractelement <8 x float> %1332, i64 5
  %2762 = fptrunc float %2761 to bfloat
  %2763 = bitcast bfloat %2762 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2763, ptr addrspace(8) %68, i32 %2759, i32 0, i32 0)
  %2764 = add i32 %2329, 96
  %2765 = add i32 %2764, %39
  %2766 = extractelement <8 x float> %1316, i64 6
  %2767 = fptrunc float %2766 to bfloat
  %2768 = mul i32 %2765, 4
  %2769 = bitcast bfloat %2767 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2769, ptr addrspace(8) %67, i32 %2768, i32 0, i32 0)
  %2770 = extractelement <8 x float> %1332, i64 6
  %2771 = fptrunc float %2770 to bfloat
  %2772 = bitcast bfloat %2771 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2772, ptr addrspace(8) %68, i32 %2768, i32 0, i32 0)
  %2773 = add i32 %2341, 96
  %2774 = add i32 %2773, %39
  %2775 = extractelement <8 x float> %1316, i64 7
  %2776 = fptrunc float %2775 to bfloat
  %2777 = mul i32 %2774, 4
  %2778 = bitcast bfloat %2776 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2778, ptr addrspace(8) %67, i32 %2777, i32 0, i32 0)
  %2779 = extractelement <8 x float> %1332, i64 7
  %2780 = fptrunc float %2779 to bfloat
  %2781 = bitcast bfloat %2780 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2781, ptr addrspace(8) %68, i32 %2777, i32 0, i32 0)
  %2782 = add i32 %2257, 112
  %2783 = add i32 %2782, %39
  %2784 = extractelement <8 x float> %1317, i64 0
  %2785 = fptrunc float %2784 to bfloat
  %2786 = mul i32 %2783, 4
  %2787 = bitcast bfloat %2785 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2787, ptr addrspace(8) %67, i32 %2786, i32 0, i32 0)
  %2788 = extractelement <8 x float> %1333, i64 0
  %2789 = fptrunc float %2788 to bfloat
  %2790 = bitcast bfloat %2789 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2790, ptr addrspace(8) %68, i32 %2786, i32 0, i32 0)
  %2791 = add i32 %2269, 112
  %2792 = add i32 %2791, %39
  %2793 = extractelement <8 x float> %1317, i64 1
  %2794 = fptrunc float %2793 to bfloat
  %2795 = mul i32 %2792, 4
  %2796 = bitcast bfloat %2794 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2796, ptr addrspace(8) %67, i32 %2795, i32 0, i32 0)
  %2797 = extractelement <8 x float> %1333, i64 1
  %2798 = fptrunc float %2797 to bfloat
  %2799 = bitcast bfloat %2798 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2799, ptr addrspace(8) %68, i32 %2795, i32 0, i32 0)
  %2800 = add i32 %2281, 112
  %2801 = add i32 %2800, %39
  %2802 = extractelement <8 x float> %1317, i64 2
  %2803 = fptrunc float %2802 to bfloat
  %2804 = mul i32 %2801, 4
  %2805 = bitcast bfloat %2803 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2805, ptr addrspace(8) %67, i32 %2804, i32 0, i32 0)
  %2806 = extractelement <8 x float> %1333, i64 2
  %2807 = fptrunc float %2806 to bfloat
  %2808 = bitcast bfloat %2807 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2808, ptr addrspace(8) %68, i32 %2804, i32 0, i32 0)
  %2809 = add i32 %2293, 112
  %2810 = add i32 %2809, %39
  %2811 = extractelement <8 x float> %1317, i64 3
  %2812 = fptrunc float %2811 to bfloat
  %2813 = mul i32 %2810, 4
  %2814 = bitcast bfloat %2812 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2814, ptr addrspace(8) %67, i32 %2813, i32 0, i32 0)
  %2815 = extractelement <8 x float> %1333, i64 3
  %2816 = fptrunc float %2815 to bfloat
  %2817 = bitcast bfloat %2816 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2817, ptr addrspace(8) %68, i32 %2813, i32 0, i32 0)
  %2818 = add i32 %2305, 112
  %2819 = add i32 %2818, %39
  %2820 = extractelement <8 x float> %1317, i64 4
  %2821 = fptrunc float %2820 to bfloat
  %2822 = mul i32 %2819, 4
  %2823 = bitcast bfloat %2821 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2823, ptr addrspace(8) %67, i32 %2822, i32 0, i32 0)
  %2824 = extractelement <8 x float> %1333, i64 4
  %2825 = fptrunc float %2824 to bfloat
  %2826 = bitcast bfloat %2825 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2826, ptr addrspace(8) %68, i32 %2822, i32 0, i32 0)
  %2827 = add i32 %2317, 112
  %2828 = add i32 %2827, %39
  %2829 = extractelement <8 x float> %1317, i64 5
  %2830 = fptrunc float %2829 to bfloat
  %2831 = mul i32 %2828, 4
  %2832 = bitcast bfloat %2830 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2832, ptr addrspace(8) %67, i32 %2831, i32 0, i32 0)
  %2833 = extractelement <8 x float> %1333, i64 5
  %2834 = fptrunc float %2833 to bfloat
  %2835 = bitcast bfloat %2834 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2835, ptr addrspace(8) %68, i32 %2831, i32 0, i32 0)
  %2836 = add i32 %2329, 112
  %2837 = add i32 %2836, %39
  %2838 = extractelement <8 x float> %1317, i64 6
  %2839 = fptrunc float %2838 to bfloat
  %2840 = mul i32 %2837, 4
  %2841 = bitcast bfloat %2839 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2841, ptr addrspace(8) %67, i32 %2840, i32 0, i32 0)
  %2842 = extractelement <8 x float> %1333, i64 6
  %2843 = fptrunc float %2842 to bfloat
  %2844 = bitcast bfloat %2843 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2844, ptr addrspace(8) %68, i32 %2840, i32 0, i32 0)
  %2845 = add i32 %2341, 112
  %2846 = add i32 %2845, %39
  %2847 = extractelement <8 x float> %1317, i64 7
  %2848 = fptrunc float %2847 to bfloat
  %2849 = mul i32 %2846, 4
  %2850 = bitcast bfloat %2848 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2850, ptr addrspace(8) %67, i32 %2849, i32 0, i32 0)
  %2851 = extractelement <8 x float> %1333, i64 7
  %2852 = fptrunc float %2851 to bfloat
  %2853 = bitcast bfloat %2852 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2853, ptr addrspace(8) %68, i32 %2849, i32 0, i32 0)
  %2854 = add i32 %114, %194
  %2855 = mul i32 %2854, %20
  %2856 = mul i32 %2855, 128
  %2857 = add i32 %2253, %2856
  %2858 = add i32 %2857, %39
  %2859 = extractelement <8 x float> %1318, i64 0
  %2860 = fptrunc float %2859 to bfloat
  %2861 = mul i32 %2858, 4
  %2862 = bitcast bfloat %2860 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2862, ptr addrspace(8) %67, i32 %2861, i32 0, i32 0)
  %2863 = extractelement <8 x float> %1334, i64 0
  %2864 = fptrunc float %2863 to bfloat
  %2865 = bitcast bfloat %2864 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2865, ptr addrspace(8) %68, i32 %2861, i32 0, i32 0)
  %2866 = add i32 %2854, 1
  %2867 = mul i32 %2866, %20
  %2868 = mul i32 %2867, 128
  %2869 = add i32 %2253, %2868
  %2870 = add i32 %2869, %39
  %2871 = extractelement <8 x float> %1318, i64 1
  %2872 = fptrunc float %2871 to bfloat
  %2873 = mul i32 %2870, 4
  %2874 = bitcast bfloat %2872 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2874, ptr addrspace(8) %67, i32 %2873, i32 0, i32 0)
  %2875 = extractelement <8 x float> %1334, i64 1
  %2876 = fptrunc float %2875 to bfloat
  %2877 = bitcast bfloat %2876 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2877, ptr addrspace(8) %68, i32 %2873, i32 0, i32 0)
  %2878 = add i32 %2854, 2
  %2879 = mul i32 %2878, %20
  %2880 = mul i32 %2879, 128
  %2881 = add i32 %2253, %2880
  %2882 = add i32 %2881, %39
  %2883 = extractelement <8 x float> %1318, i64 2
  %2884 = fptrunc float %2883 to bfloat
  %2885 = mul i32 %2882, 4
  %2886 = bitcast bfloat %2884 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2886, ptr addrspace(8) %67, i32 %2885, i32 0, i32 0)
  %2887 = extractelement <8 x float> %1334, i64 2
  %2888 = fptrunc float %2887 to bfloat
  %2889 = bitcast bfloat %2888 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2889, ptr addrspace(8) %68, i32 %2885, i32 0, i32 0)
  %2890 = add i32 %2854, 3
  %2891 = mul i32 %2890, %20
  %2892 = mul i32 %2891, 128
  %2893 = add i32 %2253, %2892
  %2894 = add i32 %2893, %39
  %2895 = extractelement <8 x float> %1318, i64 3
  %2896 = fptrunc float %2895 to bfloat
  %2897 = mul i32 %2894, 4
  %2898 = bitcast bfloat %2896 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2898, ptr addrspace(8) %67, i32 %2897, i32 0, i32 0)
  %2899 = extractelement <8 x float> %1334, i64 3
  %2900 = fptrunc float %2899 to bfloat
  %2901 = bitcast bfloat %2900 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2901, ptr addrspace(8) %68, i32 %2897, i32 0, i32 0)
  %2902 = add i32 %2854, 4
  %2903 = mul i32 %2902, %20
  %2904 = mul i32 %2903, 128
  %2905 = add i32 %2253, %2904
  %2906 = add i32 %2905, %39
  %2907 = extractelement <8 x float> %1318, i64 4
  %2908 = fptrunc float %2907 to bfloat
  %2909 = mul i32 %2906, 4
  %2910 = bitcast bfloat %2908 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2910, ptr addrspace(8) %67, i32 %2909, i32 0, i32 0)
  %2911 = extractelement <8 x float> %1334, i64 4
  %2912 = fptrunc float %2911 to bfloat
  %2913 = bitcast bfloat %2912 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2913, ptr addrspace(8) %68, i32 %2909, i32 0, i32 0)
  %2914 = add i32 %2854, 5
  %2915 = mul i32 %2914, %20
  %2916 = mul i32 %2915, 128
  %2917 = add i32 %2253, %2916
  %2918 = add i32 %2917, %39
  %2919 = extractelement <8 x float> %1318, i64 5
  %2920 = fptrunc float %2919 to bfloat
  %2921 = mul i32 %2918, 4
  %2922 = bitcast bfloat %2920 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2922, ptr addrspace(8) %67, i32 %2921, i32 0, i32 0)
  %2923 = extractelement <8 x float> %1334, i64 5
  %2924 = fptrunc float %2923 to bfloat
  %2925 = bitcast bfloat %2924 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2925, ptr addrspace(8) %68, i32 %2921, i32 0, i32 0)
  %2926 = add i32 %2854, 6
  %2927 = mul i32 %2926, %20
  %2928 = mul i32 %2927, 128
  %2929 = add i32 %2253, %2928
  %2930 = add i32 %2929, %39
  %2931 = extractelement <8 x float> %1318, i64 6
  %2932 = fptrunc float %2931 to bfloat
  %2933 = mul i32 %2930, 4
  %2934 = bitcast bfloat %2932 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2934, ptr addrspace(8) %67, i32 %2933, i32 0, i32 0)
  %2935 = extractelement <8 x float> %1334, i64 6
  %2936 = fptrunc float %2935 to bfloat
  %2937 = bitcast bfloat %2936 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2937, ptr addrspace(8) %68, i32 %2933, i32 0, i32 0)
  %2938 = add i32 %2854, 7
  %2939 = mul i32 %2938, %20
  %2940 = mul i32 %2939, 128
  %2941 = add i32 %2253, %2940
  %2942 = add i32 %2941, %39
  %2943 = extractelement <8 x float> %1318, i64 7
  %2944 = fptrunc float %2943 to bfloat
  %2945 = mul i32 %2942, 4
  %2946 = bitcast bfloat %2944 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2946, ptr addrspace(8) %67, i32 %2945, i32 0, i32 0)
  %2947 = extractelement <8 x float> %1334, i64 7
  %2948 = fptrunc float %2947 to bfloat
  %2949 = bitcast bfloat %2948 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2949, ptr addrspace(8) %68, i32 %2945, i32 0, i32 0)
  %2950 = add i32 %2857, 16
  %2951 = add i32 %2950, %39
  %2952 = extractelement <8 x float> %1319, i64 0
  %2953 = fptrunc float %2952 to bfloat
  %2954 = mul i32 %2951, 4
  %2955 = bitcast bfloat %2953 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2955, ptr addrspace(8) %67, i32 %2954, i32 0, i32 0)
  %2956 = extractelement <8 x float> %1335, i64 0
  %2957 = fptrunc float %2956 to bfloat
  %2958 = bitcast bfloat %2957 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2958, ptr addrspace(8) %68, i32 %2954, i32 0, i32 0)
  %2959 = add i32 %2869, 16
  %2960 = add i32 %2959, %39
  %2961 = extractelement <8 x float> %1319, i64 1
  %2962 = fptrunc float %2961 to bfloat
  %2963 = mul i32 %2960, 4
  %2964 = bitcast bfloat %2962 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2964, ptr addrspace(8) %67, i32 %2963, i32 0, i32 0)
  %2965 = extractelement <8 x float> %1335, i64 1
  %2966 = fptrunc float %2965 to bfloat
  %2967 = bitcast bfloat %2966 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2967, ptr addrspace(8) %68, i32 %2963, i32 0, i32 0)
  %2968 = add i32 %2881, 16
  %2969 = add i32 %2968, %39
  %2970 = extractelement <8 x float> %1319, i64 2
  %2971 = fptrunc float %2970 to bfloat
  %2972 = mul i32 %2969, 4
  %2973 = bitcast bfloat %2971 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2973, ptr addrspace(8) %67, i32 %2972, i32 0, i32 0)
  %2974 = extractelement <8 x float> %1335, i64 2
  %2975 = fptrunc float %2974 to bfloat
  %2976 = bitcast bfloat %2975 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2976, ptr addrspace(8) %68, i32 %2972, i32 0, i32 0)
  %2977 = add i32 %2893, 16
  %2978 = add i32 %2977, %39
  %2979 = extractelement <8 x float> %1319, i64 3
  %2980 = fptrunc float %2979 to bfloat
  %2981 = mul i32 %2978, 4
  %2982 = bitcast bfloat %2980 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2982, ptr addrspace(8) %67, i32 %2981, i32 0, i32 0)
  %2983 = extractelement <8 x float> %1335, i64 3
  %2984 = fptrunc float %2983 to bfloat
  %2985 = bitcast bfloat %2984 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2985, ptr addrspace(8) %68, i32 %2981, i32 0, i32 0)
  %2986 = add i32 %2905, 16
  %2987 = add i32 %2986, %39
  %2988 = extractelement <8 x float> %1319, i64 4
  %2989 = fptrunc float %2988 to bfloat
  %2990 = mul i32 %2987, 4
  %2991 = bitcast bfloat %2989 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2991, ptr addrspace(8) %67, i32 %2990, i32 0, i32 0)
  %2992 = extractelement <8 x float> %1335, i64 4
  %2993 = fptrunc float %2992 to bfloat
  %2994 = bitcast bfloat %2993 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %2994, ptr addrspace(8) %68, i32 %2990, i32 0, i32 0)
  %2995 = add i32 %2917, 16
  %2996 = add i32 %2995, %39
  %2997 = extractelement <8 x float> %1319, i64 5
  %2998 = fptrunc float %2997 to bfloat
  %2999 = mul i32 %2996, 4
  %3000 = bitcast bfloat %2998 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3000, ptr addrspace(8) %67, i32 %2999, i32 0, i32 0)
  %3001 = extractelement <8 x float> %1335, i64 5
  %3002 = fptrunc float %3001 to bfloat
  %3003 = bitcast bfloat %3002 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3003, ptr addrspace(8) %68, i32 %2999, i32 0, i32 0)
  %3004 = add i32 %2929, 16
  %3005 = add i32 %3004, %39
  %3006 = extractelement <8 x float> %1319, i64 6
  %3007 = fptrunc float %3006 to bfloat
  %3008 = mul i32 %3005, 4
  %3009 = bitcast bfloat %3007 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3009, ptr addrspace(8) %67, i32 %3008, i32 0, i32 0)
  %3010 = extractelement <8 x float> %1335, i64 6
  %3011 = fptrunc float %3010 to bfloat
  %3012 = bitcast bfloat %3011 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3012, ptr addrspace(8) %68, i32 %3008, i32 0, i32 0)
  %3013 = add i32 %2941, 16
  %3014 = add i32 %3013, %39
  %3015 = extractelement <8 x float> %1319, i64 7
  %3016 = fptrunc float %3015 to bfloat
  %3017 = mul i32 %3014, 4
  %3018 = bitcast bfloat %3016 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3018, ptr addrspace(8) %67, i32 %3017, i32 0, i32 0)
  %3019 = extractelement <8 x float> %1335, i64 7
  %3020 = fptrunc float %3019 to bfloat
  %3021 = bitcast bfloat %3020 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3021, ptr addrspace(8) %68, i32 %3017, i32 0, i32 0)
  %3022 = add i32 %2857, 32
  %3023 = add i32 %3022, %39
  %3024 = extractelement <8 x float> %1320, i64 0
  %3025 = fptrunc float %3024 to bfloat
  %3026 = mul i32 %3023, 4
  %3027 = bitcast bfloat %3025 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3027, ptr addrspace(8) %67, i32 %3026, i32 0, i32 0)
  %3028 = extractelement <8 x float> %1336, i64 0
  %3029 = fptrunc float %3028 to bfloat
  %3030 = bitcast bfloat %3029 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3030, ptr addrspace(8) %68, i32 %3026, i32 0, i32 0)
  %3031 = add i32 %2869, 32
  %3032 = add i32 %3031, %39
  %3033 = extractelement <8 x float> %1320, i64 1
  %3034 = fptrunc float %3033 to bfloat
  %3035 = mul i32 %3032, 4
  %3036 = bitcast bfloat %3034 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3036, ptr addrspace(8) %67, i32 %3035, i32 0, i32 0)
  %3037 = extractelement <8 x float> %1336, i64 1
  %3038 = fptrunc float %3037 to bfloat
  %3039 = bitcast bfloat %3038 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3039, ptr addrspace(8) %68, i32 %3035, i32 0, i32 0)
  %3040 = add i32 %2881, 32
  %3041 = add i32 %3040, %39
  %3042 = extractelement <8 x float> %1320, i64 2
  %3043 = fptrunc float %3042 to bfloat
  %3044 = mul i32 %3041, 4
  %3045 = bitcast bfloat %3043 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3045, ptr addrspace(8) %67, i32 %3044, i32 0, i32 0)
  %3046 = extractelement <8 x float> %1336, i64 2
  %3047 = fptrunc float %3046 to bfloat
  %3048 = bitcast bfloat %3047 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3048, ptr addrspace(8) %68, i32 %3044, i32 0, i32 0)
  %3049 = add i32 %2893, 32
  %3050 = add i32 %3049, %39
  %3051 = extractelement <8 x float> %1320, i64 3
  %3052 = fptrunc float %3051 to bfloat
  %3053 = mul i32 %3050, 4
  %3054 = bitcast bfloat %3052 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3054, ptr addrspace(8) %67, i32 %3053, i32 0, i32 0)
  %3055 = extractelement <8 x float> %1336, i64 3
  %3056 = fptrunc float %3055 to bfloat
  %3057 = bitcast bfloat %3056 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3057, ptr addrspace(8) %68, i32 %3053, i32 0, i32 0)
  %3058 = add i32 %2905, 32
  %3059 = add i32 %3058, %39
  %3060 = extractelement <8 x float> %1320, i64 4
  %3061 = fptrunc float %3060 to bfloat
  %3062 = mul i32 %3059, 4
  %3063 = bitcast bfloat %3061 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3063, ptr addrspace(8) %67, i32 %3062, i32 0, i32 0)
  %3064 = extractelement <8 x float> %1336, i64 4
  %3065 = fptrunc float %3064 to bfloat
  %3066 = bitcast bfloat %3065 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3066, ptr addrspace(8) %68, i32 %3062, i32 0, i32 0)
  %3067 = add i32 %2917, 32
  %3068 = add i32 %3067, %39
  %3069 = extractelement <8 x float> %1320, i64 5
  %3070 = fptrunc float %3069 to bfloat
  %3071 = mul i32 %3068, 4
  %3072 = bitcast bfloat %3070 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3072, ptr addrspace(8) %67, i32 %3071, i32 0, i32 0)
  %3073 = extractelement <8 x float> %1336, i64 5
  %3074 = fptrunc float %3073 to bfloat
  %3075 = bitcast bfloat %3074 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3075, ptr addrspace(8) %68, i32 %3071, i32 0, i32 0)
  %3076 = add i32 %2929, 32
  %3077 = add i32 %3076, %39
  %3078 = extractelement <8 x float> %1320, i64 6
  %3079 = fptrunc float %3078 to bfloat
  %3080 = mul i32 %3077, 4
  %3081 = bitcast bfloat %3079 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3081, ptr addrspace(8) %67, i32 %3080, i32 0, i32 0)
  %3082 = extractelement <8 x float> %1336, i64 6
  %3083 = fptrunc float %3082 to bfloat
  %3084 = bitcast bfloat %3083 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3084, ptr addrspace(8) %68, i32 %3080, i32 0, i32 0)
  %3085 = add i32 %2941, 32
  %3086 = add i32 %3085, %39
  %3087 = extractelement <8 x float> %1320, i64 7
  %3088 = fptrunc float %3087 to bfloat
  %3089 = mul i32 %3086, 4
  %3090 = bitcast bfloat %3088 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3090, ptr addrspace(8) %67, i32 %3089, i32 0, i32 0)
  %3091 = extractelement <8 x float> %1336, i64 7
  %3092 = fptrunc float %3091 to bfloat
  %3093 = bitcast bfloat %3092 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3093, ptr addrspace(8) %68, i32 %3089, i32 0, i32 0)
  %3094 = add i32 %2857, 48
  %3095 = add i32 %3094, %39
  %3096 = extractelement <8 x float> %1321, i64 0
  %3097 = fptrunc float %3096 to bfloat
  %3098 = mul i32 %3095, 4
  %3099 = bitcast bfloat %3097 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3099, ptr addrspace(8) %67, i32 %3098, i32 0, i32 0)
  %3100 = extractelement <8 x float> %1337, i64 0
  %3101 = fptrunc float %3100 to bfloat
  %3102 = bitcast bfloat %3101 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3102, ptr addrspace(8) %68, i32 %3098, i32 0, i32 0)
  %3103 = add i32 %2869, 48
  %3104 = add i32 %3103, %39
  %3105 = extractelement <8 x float> %1321, i64 1
  %3106 = fptrunc float %3105 to bfloat
  %3107 = mul i32 %3104, 4
  %3108 = bitcast bfloat %3106 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3108, ptr addrspace(8) %67, i32 %3107, i32 0, i32 0)
  %3109 = extractelement <8 x float> %1337, i64 1
  %3110 = fptrunc float %3109 to bfloat
  %3111 = bitcast bfloat %3110 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3111, ptr addrspace(8) %68, i32 %3107, i32 0, i32 0)
  %3112 = add i32 %2881, 48
  %3113 = add i32 %3112, %39
  %3114 = extractelement <8 x float> %1321, i64 2
  %3115 = fptrunc float %3114 to bfloat
  %3116 = mul i32 %3113, 4
  %3117 = bitcast bfloat %3115 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3117, ptr addrspace(8) %67, i32 %3116, i32 0, i32 0)
  %3118 = extractelement <8 x float> %1337, i64 2
  %3119 = fptrunc float %3118 to bfloat
  %3120 = bitcast bfloat %3119 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3120, ptr addrspace(8) %68, i32 %3116, i32 0, i32 0)
  %3121 = add i32 %2893, 48
  %3122 = add i32 %3121, %39
  %3123 = extractelement <8 x float> %1321, i64 3
  %3124 = fptrunc float %3123 to bfloat
  %3125 = mul i32 %3122, 4
  %3126 = bitcast bfloat %3124 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3126, ptr addrspace(8) %67, i32 %3125, i32 0, i32 0)
  %3127 = extractelement <8 x float> %1337, i64 3
  %3128 = fptrunc float %3127 to bfloat
  %3129 = bitcast bfloat %3128 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3129, ptr addrspace(8) %68, i32 %3125, i32 0, i32 0)
  %3130 = add i32 %2905, 48
  %3131 = add i32 %3130, %39
  %3132 = extractelement <8 x float> %1321, i64 4
  %3133 = fptrunc float %3132 to bfloat
  %3134 = mul i32 %3131, 4
  %3135 = bitcast bfloat %3133 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3135, ptr addrspace(8) %67, i32 %3134, i32 0, i32 0)
  %3136 = extractelement <8 x float> %1337, i64 4
  %3137 = fptrunc float %3136 to bfloat
  %3138 = bitcast bfloat %3137 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3138, ptr addrspace(8) %68, i32 %3134, i32 0, i32 0)
  %3139 = add i32 %2917, 48
  %3140 = add i32 %3139, %39
  %3141 = extractelement <8 x float> %1321, i64 5
  %3142 = fptrunc float %3141 to bfloat
  %3143 = mul i32 %3140, 4
  %3144 = bitcast bfloat %3142 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3144, ptr addrspace(8) %67, i32 %3143, i32 0, i32 0)
  %3145 = extractelement <8 x float> %1337, i64 5
  %3146 = fptrunc float %3145 to bfloat
  %3147 = bitcast bfloat %3146 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3147, ptr addrspace(8) %68, i32 %3143, i32 0, i32 0)
  %3148 = add i32 %2929, 48
  %3149 = add i32 %3148, %39
  %3150 = extractelement <8 x float> %1321, i64 6
  %3151 = fptrunc float %3150 to bfloat
  %3152 = mul i32 %3149, 4
  %3153 = bitcast bfloat %3151 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3153, ptr addrspace(8) %67, i32 %3152, i32 0, i32 0)
  %3154 = extractelement <8 x float> %1337, i64 6
  %3155 = fptrunc float %3154 to bfloat
  %3156 = bitcast bfloat %3155 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3156, ptr addrspace(8) %68, i32 %3152, i32 0, i32 0)
  %3157 = add i32 %2941, 48
  %3158 = add i32 %3157, %39
  %3159 = extractelement <8 x float> %1321, i64 7
  %3160 = fptrunc float %3159 to bfloat
  %3161 = mul i32 %3158, 4
  %3162 = bitcast bfloat %3160 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3162, ptr addrspace(8) %67, i32 %3161, i32 0, i32 0)
  %3163 = extractelement <8 x float> %1337, i64 7
  %3164 = fptrunc float %3163 to bfloat
  %3165 = bitcast bfloat %3164 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3165, ptr addrspace(8) %68, i32 %3161, i32 0, i32 0)
  %3166 = add i32 %2857, 64
  %3167 = add i32 %3166, %39
  %3168 = extractelement <8 x float> %1322, i64 0
  %3169 = fptrunc float %3168 to bfloat
  %3170 = mul i32 %3167, 4
  %3171 = bitcast bfloat %3169 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3171, ptr addrspace(8) %67, i32 %3170, i32 0, i32 0)
  %3172 = extractelement <8 x float> %1338, i64 0
  %3173 = fptrunc float %3172 to bfloat
  %3174 = bitcast bfloat %3173 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3174, ptr addrspace(8) %68, i32 %3170, i32 0, i32 0)
  %3175 = add i32 %2869, 64
  %3176 = add i32 %3175, %39
  %3177 = extractelement <8 x float> %1322, i64 1
  %3178 = fptrunc float %3177 to bfloat
  %3179 = mul i32 %3176, 4
  %3180 = bitcast bfloat %3178 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3180, ptr addrspace(8) %67, i32 %3179, i32 0, i32 0)
  %3181 = extractelement <8 x float> %1338, i64 1
  %3182 = fptrunc float %3181 to bfloat
  %3183 = bitcast bfloat %3182 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3183, ptr addrspace(8) %68, i32 %3179, i32 0, i32 0)
  %3184 = add i32 %2881, 64
  %3185 = add i32 %3184, %39
  %3186 = extractelement <8 x float> %1322, i64 2
  %3187 = fptrunc float %3186 to bfloat
  %3188 = mul i32 %3185, 4
  %3189 = bitcast bfloat %3187 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3189, ptr addrspace(8) %67, i32 %3188, i32 0, i32 0)
  %3190 = extractelement <8 x float> %1338, i64 2
  %3191 = fptrunc float %3190 to bfloat
  %3192 = bitcast bfloat %3191 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3192, ptr addrspace(8) %68, i32 %3188, i32 0, i32 0)
  %3193 = add i32 %2893, 64
  %3194 = add i32 %3193, %39
  %3195 = extractelement <8 x float> %1322, i64 3
  %3196 = fptrunc float %3195 to bfloat
  %3197 = mul i32 %3194, 4
  %3198 = bitcast bfloat %3196 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3198, ptr addrspace(8) %67, i32 %3197, i32 0, i32 0)
  %3199 = extractelement <8 x float> %1338, i64 3
  %3200 = fptrunc float %3199 to bfloat
  %3201 = bitcast bfloat %3200 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3201, ptr addrspace(8) %68, i32 %3197, i32 0, i32 0)
  %3202 = add i32 %2905, 64
  %3203 = add i32 %3202, %39
  %3204 = extractelement <8 x float> %1322, i64 4
  %3205 = fptrunc float %3204 to bfloat
  %3206 = mul i32 %3203, 4
  %3207 = bitcast bfloat %3205 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3207, ptr addrspace(8) %67, i32 %3206, i32 0, i32 0)
  %3208 = extractelement <8 x float> %1338, i64 4
  %3209 = fptrunc float %3208 to bfloat
  %3210 = bitcast bfloat %3209 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3210, ptr addrspace(8) %68, i32 %3206, i32 0, i32 0)
  %3211 = add i32 %2917, 64
  %3212 = add i32 %3211, %39
  %3213 = extractelement <8 x float> %1322, i64 5
  %3214 = fptrunc float %3213 to bfloat
  %3215 = mul i32 %3212, 4
  %3216 = bitcast bfloat %3214 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3216, ptr addrspace(8) %67, i32 %3215, i32 0, i32 0)
  %3217 = extractelement <8 x float> %1338, i64 5
  %3218 = fptrunc float %3217 to bfloat
  %3219 = bitcast bfloat %3218 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3219, ptr addrspace(8) %68, i32 %3215, i32 0, i32 0)
  %3220 = add i32 %2929, 64
  %3221 = add i32 %3220, %39
  %3222 = extractelement <8 x float> %1322, i64 6
  %3223 = fptrunc float %3222 to bfloat
  %3224 = mul i32 %3221, 4
  %3225 = bitcast bfloat %3223 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3225, ptr addrspace(8) %67, i32 %3224, i32 0, i32 0)
  %3226 = extractelement <8 x float> %1338, i64 6
  %3227 = fptrunc float %3226 to bfloat
  %3228 = bitcast bfloat %3227 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3228, ptr addrspace(8) %68, i32 %3224, i32 0, i32 0)
  %3229 = add i32 %2941, 64
  %3230 = add i32 %3229, %39
  %3231 = extractelement <8 x float> %1322, i64 7
  %3232 = fptrunc float %3231 to bfloat
  %3233 = mul i32 %3230, 4
  %3234 = bitcast bfloat %3232 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3234, ptr addrspace(8) %67, i32 %3233, i32 0, i32 0)
  %3235 = extractelement <8 x float> %1338, i64 7
  %3236 = fptrunc float %3235 to bfloat
  %3237 = bitcast bfloat %3236 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3237, ptr addrspace(8) %68, i32 %3233, i32 0, i32 0)
  %3238 = add i32 %2857, 80
  %3239 = add i32 %3238, %39
  %3240 = extractelement <8 x float> %1323, i64 0
  %3241 = fptrunc float %3240 to bfloat
  %3242 = mul i32 %3239, 4
  %3243 = bitcast bfloat %3241 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3243, ptr addrspace(8) %67, i32 %3242, i32 0, i32 0)
  %3244 = extractelement <8 x float> %1339, i64 0
  %3245 = fptrunc float %3244 to bfloat
  %3246 = bitcast bfloat %3245 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3246, ptr addrspace(8) %68, i32 %3242, i32 0, i32 0)
  %3247 = add i32 %2869, 80
  %3248 = add i32 %3247, %39
  %3249 = extractelement <8 x float> %1323, i64 1
  %3250 = fptrunc float %3249 to bfloat
  %3251 = mul i32 %3248, 4
  %3252 = bitcast bfloat %3250 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3252, ptr addrspace(8) %67, i32 %3251, i32 0, i32 0)
  %3253 = extractelement <8 x float> %1339, i64 1
  %3254 = fptrunc float %3253 to bfloat
  %3255 = bitcast bfloat %3254 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3255, ptr addrspace(8) %68, i32 %3251, i32 0, i32 0)
  %3256 = add i32 %2881, 80
  %3257 = add i32 %3256, %39
  %3258 = extractelement <8 x float> %1323, i64 2
  %3259 = fptrunc float %3258 to bfloat
  %3260 = mul i32 %3257, 4
  %3261 = bitcast bfloat %3259 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3261, ptr addrspace(8) %67, i32 %3260, i32 0, i32 0)
  %3262 = extractelement <8 x float> %1339, i64 2
  %3263 = fptrunc float %3262 to bfloat
  %3264 = bitcast bfloat %3263 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3264, ptr addrspace(8) %68, i32 %3260, i32 0, i32 0)
  %3265 = add i32 %2893, 80
  %3266 = add i32 %3265, %39
  %3267 = extractelement <8 x float> %1323, i64 3
  %3268 = fptrunc float %3267 to bfloat
  %3269 = mul i32 %3266, 4
  %3270 = bitcast bfloat %3268 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3270, ptr addrspace(8) %67, i32 %3269, i32 0, i32 0)
  %3271 = extractelement <8 x float> %1339, i64 3
  %3272 = fptrunc float %3271 to bfloat
  %3273 = bitcast bfloat %3272 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3273, ptr addrspace(8) %68, i32 %3269, i32 0, i32 0)
  %3274 = add i32 %2905, 80
  %3275 = add i32 %3274, %39
  %3276 = extractelement <8 x float> %1323, i64 4
  %3277 = fptrunc float %3276 to bfloat
  %3278 = mul i32 %3275, 4
  %3279 = bitcast bfloat %3277 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3279, ptr addrspace(8) %67, i32 %3278, i32 0, i32 0)
  %3280 = extractelement <8 x float> %1339, i64 4
  %3281 = fptrunc float %3280 to bfloat
  %3282 = bitcast bfloat %3281 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3282, ptr addrspace(8) %68, i32 %3278, i32 0, i32 0)
  %3283 = add i32 %2917, 80
  %3284 = add i32 %3283, %39
  %3285 = extractelement <8 x float> %1323, i64 5
  %3286 = fptrunc float %3285 to bfloat
  %3287 = mul i32 %3284, 4
  %3288 = bitcast bfloat %3286 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3288, ptr addrspace(8) %67, i32 %3287, i32 0, i32 0)
  %3289 = extractelement <8 x float> %1339, i64 5
  %3290 = fptrunc float %3289 to bfloat
  %3291 = bitcast bfloat %3290 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3291, ptr addrspace(8) %68, i32 %3287, i32 0, i32 0)
  %3292 = add i32 %2929, 80
  %3293 = add i32 %3292, %39
  %3294 = extractelement <8 x float> %1323, i64 6
  %3295 = fptrunc float %3294 to bfloat
  %3296 = mul i32 %3293, 4
  %3297 = bitcast bfloat %3295 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3297, ptr addrspace(8) %67, i32 %3296, i32 0, i32 0)
  %3298 = extractelement <8 x float> %1339, i64 6
  %3299 = fptrunc float %3298 to bfloat
  %3300 = bitcast bfloat %3299 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3300, ptr addrspace(8) %68, i32 %3296, i32 0, i32 0)
  %3301 = add i32 %2941, 80
  %3302 = add i32 %3301, %39
  %3303 = extractelement <8 x float> %1323, i64 7
  %3304 = fptrunc float %3303 to bfloat
  %3305 = mul i32 %3302, 4
  %3306 = bitcast bfloat %3304 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3306, ptr addrspace(8) %67, i32 %3305, i32 0, i32 0)
  %3307 = extractelement <8 x float> %1339, i64 7
  %3308 = fptrunc float %3307 to bfloat
  %3309 = bitcast bfloat %3308 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3309, ptr addrspace(8) %68, i32 %3305, i32 0, i32 0)
  %3310 = add i32 %2857, 96
  %3311 = add i32 %3310, %39
  %3312 = extractelement <8 x float> %1324, i64 0
  %3313 = fptrunc float %3312 to bfloat
  %3314 = mul i32 %3311, 4
  %3315 = bitcast bfloat %3313 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3315, ptr addrspace(8) %67, i32 %3314, i32 0, i32 0)
  %3316 = extractelement <8 x float> %1340, i64 0
  %3317 = fptrunc float %3316 to bfloat
  %3318 = bitcast bfloat %3317 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3318, ptr addrspace(8) %68, i32 %3314, i32 0, i32 0)
  %3319 = add i32 %2869, 96
  %3320 = add i32 %3319, %39
  %3321 = extractelement <8 x float> %1324, i64 1
  %3322 = fptrunc float %3321 to bfloat
  %3323 = mul i32 %3320, 4
  %3324 = bitcast bfloat %3322 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3324, ptr addrspace(8) %67, i32 %3323, i32 0, i32 0)
  %3325 = extractelement <8 x float> %1340, i64 1
  %3326 = fptrunc float %3325 to bfloat
  %3327 = bitcast bfloat %3326 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3327, ptr addrspace(8) %68, i32 %3323, i32 0, i32 0)
  %3328 = add i32 %2881, 96
  %3329 = add i32 %3328, %39
  %3330 = extractelement <8 x float> %1324, i64 2
  %3331 = fptrunc float %3330 to bfloat
  %3332 = mul i32 %3329, 4
  %3333 = bitcast bfloat %3331 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3333, ptr addrspace(8) %67, i32 %3332, i32 0, i32 0)
  %3334 = extractelement <8 x float> %1340, i64 2
  %3335 = fptrunc float %3334 to bfloat
  %3336 = bitcast bfloat %3335 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3336, ptr addrspace(8) %68, i32 %3332, i32 0, i32 0)
  %3337 = add i32 %2893, 96
  %3338 = add i32 %3337, %39
  %3339 = extractelement <8 x float> %1324, i64 3
  %3340 = fptrunc float %3339 to bfloat
  %3341 = mul i32 %3338, 4
  %3342 = bitcast bfloat %3340 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3342, ptr addrspace(8) %67, i32 %3341, i32 0, i32 0)
  %3343 = extractelement <8 x float> %1340, i64 3
  %3344 = fptrunc float %3343 to bfloat
  %3345 = bitcast bfloat %3344 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3345, ptr addrspace(8) %68, i32 %3341, i32 0, i32 0)
  %3346 = add i32 %2905, 96
  %3347 = add i32 %3346, %39
  %3348 = extractelement <8 x float> %1324, i64 4
  %3349 = fptrunc float %3348 to bfloat
  %3350 = mul i32 %3347, 4
  %3351 = bitcast bfloat %3349 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3351, ptr addrspace(8) %67, i32 %3350, i32 0, i32 0)
  %3352 = extractelement <8 x float> %1340, i64 4
  %3353 = fptrunc float %3352 to bfloat
  %3354 = bitcast bfloat %3353 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3354, ptr addrspace(8) %68, i32 %3350, i32 0, i32 0)
  %3355 = add i32 %2917, 96
  %3356 = add i32 %3355, %39
  %3357 = extractelement <8 x float> %1324, i64 5
  %3358 = fptrunc float %3357 to bfloat
  %3359 = mul i32 %3356, 4
  %3360 = bitcast bfloat %3358 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3360, ptr addrspace(8) %67, i32 %3359, i32 0, i32 0)
  %3361 = extractelement <8 x float> %1340, i64 5
  %3362 = fptrunc float %3361 to bfloat
  %3363 = bitcast bfloat %3362 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3363, ptr addrspace(8) %68, i32 %3359, i32 0, i32 0)
  %3364 = add i32 %2929, 96
  %3365 = add i32 %3364, %39
  %3366 = extractelement <8 x float> %1324, i64 6
  %3367 = fptrunc float %3366 to bfloat
  %3368 = mul i32 %3365, 4
  %3369 = bitcast bfloat %3367 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3369, ptr addrspace(8) %67, i32 %3368, i32 0, i32 0)
  %3370 = extractelement <8 x float> %1340, i64 6
  %3371 = fptrunc float %3370 to bfloat
  %3372 = bitcast bfloat %3371 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3372, ptr addrspace(8) %68, i32 %3368, i32 0, i32 0)
  %3373 = add i32 %2941, 96
  %3374 = add i32 %3373, %39
  %3375 = extractelement <8 x float> %1324, i64 7
  %3376 = fptrunc float %3375 to bfloat
  %3377 = mul i32 %3374, 4
  %3378 = bitcast bfloat %3376 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3378, ptr addrspace(8) %67, i32 %3377, i32 0, i32 0)
  %3379 = extractelement <8 x float> %1340, i64 7
  %3380 = fptrunc float %3379 to bfloat
  %3381 = bitcast bfloat %3380 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3381, ptr addrspace(8) %68, i32 %3377, i32 0, i32 0)
  %3382 = add i32 %2857, 112
  %3383 = add i32 %3382, %39
  %3384 = extractelement <8 x float> %1325, i64 0
  %3385 = fptrunc float %3384 to bfloat
  %3386 = mul i32 %3383, 4
  %3387 = bitcast bfloat %3385 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3387, ptr addrspace(8) %67, i32 %3386, i32 0, i32 0)
  %3388 = extractelement <8 x float> %1341, i64 0
  %3389 = fptrunc float %3388 to bfloat
  %3390 = bitcast bfloat %3389 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3390, ptr addrspace(8) %68, i32 %3386, i32 0, i32 0)
  %3391 = add i32 %2869, 112
  %3392 = add i32 %3391, %39
  %3393 = extractelement <8 x float> %1325, i64 1
  %3394 = fptrunc float %3393 to bfloat
  %3395 = mul i32 %3392, 4
  %3396 = bitcast bfloat %3394 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3396, ptr addrspace(8) %67, i32 %3395, i32 0, i32 0)
  %3397 = extractelement <8 x float> %1341, i64 1
  %3398 = fptrunc float %3397 to bfloat
  %3399 = bitcast bfloat %3398 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3399, ptr addrspace(8) %68, i32 %3395, i32 0, i32 0)
  %3400 = add i32 %2881, 112
  %3401 = add i32 %3400, %39
  %3402 = extractelement <8 x float> %1325, i64 2
  %3403 = fptrunc float %3402 to bfloat
  %3404 = mul i32 %3401, 4
  %3405 = bitcast bfloat %3403 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3405, ptr addrspace(8) %67, i32 %3404, i32 0, i32 0)
  %3406 = extractelement <8 x float> %1341, i64 2
  %3407 = fptrunc float %3406 to bfloat
  %3408 = bitcast bfloat %3407 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3408, ptr addrspace(8) %68, i32 %3404, i32 0, i32 0)
  %3409 = add i32 %2893, 112
  %3410 = add i32 %3409, %39
  %3411 = extractelement <8 x float> %1325, i64 3
  %3412 = fptrunc float %3411 to bfloat
  %3413 = mul i32 %3410, 4
  %3414 = bitcast bfloat %3412 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3414, ptr addrspace(8) %67, i32 %3413, i32 0, i32 0)
  %3415 = extractelement <8 x float> %1341, i64 3
  %3416 = fptrunc float %3415 to bfloat
  %3417 = bitcast bfloat %3416 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3417, ptr addrspace(8) %68, i32 %3413, i32 0, i32 0)
  %3418 = add i32 %2905, 112
  %3419 = add i32 %3418, %39
  %3420 = extractelement <8 x float> %1325, i64 4
  %3421 = fptrunc float %3420 to bfloat
  %3422 = mul i32 %3419, 4
  %3423 = bitcast bfloat %3421 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3423, ptr addrspace(8) %67, i32 %3422, i32 0, i32 0)
  %3424 = extractelement <8 x float> %1341, i64 4
  %3425 = fptrunc float %3424 to bfloat
  %3426 = bitcast bfloat %3425 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3426, ptr addrspace(8) %68, i32 %3422, i32 0, i32 0)
  %3427 = add i32 %2917, 112
  %3428 = add i32 %3427, %39
  %3429 = extractelement <8 x float> %1325, i64 5
  %3430 = fptrunc float %3429 to bfloat
  %3431 = mul i32 %3428, 4
  %3432 = bitcast bfloat %3430 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3432, ptr addrspace(8) %67, i32 %3431, i32 0, i32 0)
  %3433 = extractelement <8 x float> %1341, i64 5
  %3434 = fptrunc float %3433 to bfloat
  %3435 = bitcast bfloat %3434 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3435, ptr addrspace(8) %68, i32 %3431, i32 0, i32 0)
  %3436 = add i32 %2929, 112
  %3437 = add i32 %3436, %39
  %3438 = extractelement <8 x float> %1325, i64 6
  %3439 = fptrunc float %3438 to bfloat
  %3440 = mul i32 %3437, 4
  %3441 = bitcast bfloat %3439 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3441, ptr addrspace(8) %67, i32 %3440, i32 0, i32 0)
  %3442 = extractelement <8 x float> %1341, i64 6
  %3443 = fptrunc float %3442 to bfloat
  %3444 = bitcast bfloat %3443 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3444, ptr addrspace(8) %68, i32 %3440, i32 0, i32 0)
  %3445 = add i32 %2941, 112
  %3446 = add i32 %3445, %39
  %3447 = extractelement <8 x float> %1325, i64 7
  %3448 = fptrunc float %3447 to bfloat
  %3449 = mul i32 %3446, 4
  %3450 = bitcast bfloat %3448 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3450, ptr addrspace(8) %67, i32 %3449, i32 0, i32 0)
  %3451 = extractelement <8 x float> %1341, i64 7
  %3452 = fptrunc float %3451 to bfloat
  %3453 = bitcast bfloat %3452 to i16
  call void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16 %3453, ptr addrspace(8) %68, i32 %3449, i32 0, i32 0)
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

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat>, <16 x bfloat>, i16 immarg, <8 x float>, i1 immarg, i1 immarg) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.signal(i32 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier.wait(i16 immarg) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(argmem: read)
declare <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) captures(none)) #7

attributes #0 = { "amdgpu-flat-work-group-size"="64,64" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #5 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 64, i32 1, i32 1}
