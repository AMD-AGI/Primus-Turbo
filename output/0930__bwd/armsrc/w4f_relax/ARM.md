# arm w4f_relax

Byte copy of arms/w4f except the variant constants in kernels.py (see arms/w4f/ARM.md section 9):

    W4_RELAX = True
    W4_ABL_NOATOM = False

Compile: ./compile_w4f.sh w4f w4f_sp cvt redsp delta. Proof: python3 bounds_proof.py (ALL PASS, log .bounds_proof.log).
