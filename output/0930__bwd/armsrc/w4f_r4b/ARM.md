# arm w4f_r4b

Copy of arms/w4f code with these constants set in kernels.py (see arms/w4f/ARM.md section 12):

    W4_RELAX = True
    W4_ABL_NOATOM = False
    W4_LDS_LSE = True
    W4_XCD = False
    W4_DQ_TILED = True
    W4_LOCK = False              # descending q sweep (the part that is kept)
    W4_GRPMAJOR = False
    W4_DELTA_ZERO = False
    W4_CVT_SRC = False
    W4_SIG_LATE = True
    W4_TDM_LSE_LATE = True

Compile: ./compile_w4f.sh w4f w4f_sp cvt_s cvt_t delta_z delta redsp. Proof: python3 bounds_proof.py (log .bounds_proof.log).
