set -u
SC="$1"; OP="$2"
run(){ env $1 OPDIR=$OP TRITON_CACHE_DIR=$SC/tcy SWEEP="$2" python $SC/sweep2.py 2>&1 | grep -E "bwd_ms|FAIL"; }
run "TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1" '[[256,"waves_per_eu=0"],[256,"waves_per_eu=0,matrix_instr_nonkdim=32"],[256,"waves_per_eu=0,BLK_SLICE_FACTOR=2"],[256,"waves_per_eu=0,num_stages=2"],[128,"waves_per_eu=0"],[512,"waves_per_eu=0"],[256,"waves_per_eu=0,BLOCK_M1=16,BLOCK_N2=16"],[256,"waves_per_eu=0,BLOCK_M1=64,BLOCK_N2=64"],[256,""]]'
