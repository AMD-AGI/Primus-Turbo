# arm w4f_r2b

Copy of arms/w4f code with these constants set in kernels.py (see arms/w4f/ARM.md sections 9 and 10):

    W4_RELAX = True
    W4_ABL_NOATOM = False
    W4_LDS_LSE = False
    W4_XCD = True

Compile: ./compile_w4f.sh w4f w4f_sp cvt redsp delta. Proof: python3 bounds_proof.py (log .bounds_proof.log).
