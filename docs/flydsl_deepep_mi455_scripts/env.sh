cd /home/xiaompen/Turbo-dev
export PYTHONPATH=/home/xiaompen/Turbo-dev
export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/primus_turbo/lib:${LD_LIBRARY_PATH}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-/tmp/fcache}
export PYTHONWARNINGS=ignore
