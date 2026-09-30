# arm w4f_r3

Copy of arms/w4f code with (see arms/w4f/ARM.md section 11):

    W4_RELAX = True
    W4_ABL_NOATOM = False
    W4_LDS_LSE = True
    W4_XCD = False
    W4_DQ_TILED = True

Compile: ./compile_w4f.sh w4f w4f_sp cvt_t cvt redsp delta. Proof: python3 bounds_proof.py (ALL PASS, .bounds_proof.log).
impl.py launches k_dq_cvt_t when W4_DQ_TILED.
