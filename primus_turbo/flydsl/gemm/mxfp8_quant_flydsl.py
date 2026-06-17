# Fused dual-cast mxfp8 quant. Tile [BM=32 x BK=256]. 512 threads/block:
# 256 threads coalesced-load BM×BK input -> LDS; then halves run concurrently:
#   ROW half (0..255): fp8 + layout-1 A-preshuffled scale  -> Qr, ASp
#   COL half (256..511): fp8 + b-comb preshuffled scale    -> AtQd, AtSp
#
# Outputs:
#   Qr   [M, K]         fp8 e4m3   -- row cast (fwd A operand)
#   ASp  [SP_A_ELEMS]   int32      -- A layout-1 preshuffled row scale (fwd gemm ScaleS2R)
#       dword[((grp*K128+kg)*64+lane)*4+sub] = broadcast_u8_to_u32(e8m0)
#       grp=row//64, sub=(row%64)//16, r=row%16
#       kcol=bk*(BK//32)+kb; g=kcol&3, kg=kcol>>2, lane=g*16+r
#
#   AtQd [K, M]         fp8 e4m3   -- col cast (bwd at operand, stored transposed)
#   AtSp [SP_AT_ELEMS]  int32      -- B-comb preshuffled col scale (bwd gemm ScaleBComb)
#       grp_bc=bk*4+wn; w=c; low=c&127; s_bc=((c>>7)<<1)|((low&31)>>4)
#       wn=low>>5; r_bc=low&15; g_bc=br&3; kg_bc=br>>2
#       dword[((grp_bc*K128p_M+kg_bc)*64+lane_bc)*4+s_bc] = broadcast(e8m0)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo
from flydsl.expr import math as fm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.utils.gemm_helper import make_row_band_resource

BM = 32
BK = 256
PAD = BK
NTH = 512
VPR = BK // 4  # vec4 per row (64 for BK=256)
_CM = 1  # glc: bypass L2 on output stores


def _ep(amax):
    I32 = fx.Int32
    ai = I32(amax.bitcast(fx.Int32.ir_type)) + I32(1 << 19)
    ep = ((ai >> I32(23)) & I32(0x1FF)) - I32(135)
    ep = (ep < I32(-127)).select(I32(-127), ep)
    ep = (ep > I32(128)).select(I32(128), ep)
    return I32(ep)


def compile_qdual(M, K):
    K128 = K // 128  # K-direction: K-blocks of 128
    K128p_M = M // 128  # M-direction: M-blocks of 128 (for b-comb K128p)
    SP_A_ELEMS = (M // 64) * K128 * 64 * 4  # layout-1 A-scale dwords
    SP_A_BYTES = SP_A_ELEMS * 4
    SP_AT_ELEMS = (K // 256) * 4 * K128p_M * 64 * 4  # b-comb col-scale dwords
    SP_AT_BYTES = SP_AT_ELEMS * 4

    @flyc.kernel(known_block_size=[NTH, 1, 1])
    def kern(X: fx.Tensor, Qr: fx.Tensor, ASp: fx.Tensor, AtQd: fx.Tensor, AtSp: fx.Tensor):
        I32 = fx.Int32
        BF = fx.BFloat16.ir_type
        F32 = fx.Float32
        z = I32(0)
        IRI = fx.Int32.ir_type

        @fx.struct
        class Smem:
            tile: fx.Array[fx.BFloat16, BM * PAD, 16]

        sm = fx.SharedAllocator().allocate(Smem).peek()
        tile = sm.tile
        t = fx.thread_idx.x
        nbk = I32(K // BK)
        pid = fx.block_idx.x
        br = pid // nbk
        bk = pid - br * nbk  # br: M-block, bk: K-block
        rx = make_row_band_resource(bo.extract_base_index(X), z, I32(M), I32(K), 2)
        gbase = (br * I32(BM)) * I32(K) + bk * I32(BK)
        # Coalesced load -> LDS: 512 threads × 4 iters → [BM=32 rows × BK=256 cols]
        for ls in range_constexpr(4):
            lin = t + I32(ls * NTH)
            lr = lin // I32(VPR)
            cv = (lin - lr * I32(VPR)) * I32(4)
            v = bo.buffer_load(rx, gbase + lr * I32(K) + cv, vec_width=4, dtype=BF)
            p = fx.add_offset(tile.ptr, fx.make_int_tuple(lr * I32(PAD) + cv))
            fx.make_view(p, fx.make_layout(4, 1)).store(Vec(v))
        rocdl.s_barrier()
        half = t // I32(256)
        lt = t - half * I32(256)
        if half == z:
            # ROW half: thread lt -> row=lt//8, kb=lt%8 within tile
            row = lt // I32(8)
            kb = lt - row * I32(8)
            loff = row * I32(PAD) + kb * I32(32)
            chunks = []
            for i in range_constexpr(8):
                p = fx.add_offset(tile.ptr, fx.make_int_tuple(loff + I32(4 * i)))
                chunks.append(Vec(fx.make_view(p, fx.make_layout(4, 1)).load()).to(F32))
            amax = F32(0.0)
            for i in range_constexpr(8):
                a = fm.absf(chunks[i]).reduce("max")
                amax = (amax > a).select(amax, a)
            grow = br * I32(BM) + row
            gcol0 = bk * I32(BK) + kb * I32(32)
            ep = _ep(amax)
            inv = F32(1.0) / fm.exp2(ep.to(F32))
            rqr = make_row_band_resource(bo.extract_base_index(Qr), z, I32(M), I32(K), 1)
            for wi in range_constexpr(8):
                qf = chunks[wi] * inv
                word = I32(rocdl.cvt_pk_fp8_f32(IRI, qf[0], qf[1], z, 0))
                word = I32(rocdl.cvt_pk_fp8_f32(IRI, qf[2], qf[3], word, 1))
                bo.buffer_store(
                    word, rqr, grow * I32(K) + gcol0 + I32(4 * wi), cache_modifier=_CM, offset_is_bytes=True
                )
            # A layout-1 preshuffled scale (keep as Int32, do NOT cast to Uint8).
            kcol = bk * I32(BK // 32) + kb
            g = kcol & I32(3)
            kg = kcol >> I32(2)
            grp = grow >> I32(6)
            sub = (grow >> I32(4)) & I32(3)
            r = grow & I32(15)
            lane = g * I32(16) + r
            dword = ((grp * I32(K128) + kg) * I32(64) + lane) * I32(4) + sub
            e8_i32 = ep + I32(127)
            bcast = e8_i32 | (e8_i32 << I32(8)) | (e8_i32 << I32(16)) | (e8_i32 << I32(24))
            rasp = bo.create_buffer_resource(ASp, max_size=False, num_records_bytes=I32(SP_A_BYTES))
            bo.buffer_store(bcast, rasp, dword, cache_modifier=_CM)
        else:
            # COL half: thread lt -> col c=lt (0..BK-1), processes all BM rows from LDS.
            c = lt
            base = fx.add_offset(tile.ptr, fx.make_int_tuple(c))
            cv2 = Vec(fx.make_view(base, fx.make_layout(32, PAD)).load()).to(F32)
            ca = fm.absf(cv2).reduce("max")
            camax = (F32(0.0) > ca).select(F32(0.0), ca)
            cep = _ep(camax)
            cinv = F32(1.0) / fm.exp2(cep.to(F32))
            cq = cv2 * cinv
            # Col fp8 -> AtQd[K, M] (gc-th K-row, br*BM..+BM M-cols).
            gc = bk * I32(BK) + c  # global K-col index (= row in AtQd)
            raqd = make_row_band_resource(bo.extract_base_index(AtQd), z, I32(K), I32(M), 1)
            mbase = br * I32(BM)
            for wi in range_constexpr(8):
                word = I32(rocdl.cvt_pk_fp8_f32(IRI, cq[4 * wi + 0], cq[4 * wi + 1], z, 0))
                word = I32(rocdl.cvt_pk_fp8_f32(IRI, cq[4 * wi + 2], cq[4 * wi + 3], word, 1))
                bo.buffer_store(
                    word, raqd, gc * I32(M) + mbase + I32(4 * wi), cache_modifier=_CM, offset_is_bytes=True
                )
            # B-comb preshuffled col scale (keep as Int32 throughout for bcast).
            # gc = bk*256+c: block_n=bk, w=c; br=M-row-block = mc (M-direction k-block).
            low = c & I32(127)
            s_bc = ((c >> I32(7)) << I32(1)) | ((low & I32(31)) >> I32(4))
            wn = low >> I32(5)
            r_bc = low & I32(15)
            g_bc = I32(br) & I32(3)
            kg_bc = I32(br) >> I32(2)
            grp_bc = bk * I32(4) + wn
            lane_bc = g_bc * I32(16) + r_bc
            dword_bc = ((grp_bc * I32(K128p_M) + kg_bc) * I32(64) + lane_bc) * I32(4) + s_bc
            e8c_i32 = cep + I32(127)
            bcast_bc = e8c_i32 | (e8c_i32 << I32(8)) | (e8c_i32 << I32(16)) | (e8c_i32 << I32(24))
            ratsp = bo.create_buffer_resource(AtSp, max_size=False, num_records_bytes=I32(SP_AT_BYTES))
            bo.buffer_store(bcast_bc, ratsp, dword_bc, cache_modifier=_CM)

    @flyc.jit
    def launch(
        X: fx.Tensor, Qr: fx.Tensor, ASp: fx.Tensor, AtQd: fx.Tensor, AtSp: fx.Tensor, stream: fx.Stream
    ):
        grid = (M // BM) * (K // BK)
        kern(X, Qr, ASp, AtQd, AtSp).launch(grid=(grid, 1, 1), block=(NTH, 1, 1), stream=stream)

    return launch
