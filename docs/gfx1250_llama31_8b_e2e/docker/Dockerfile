###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Image for the Llama-3.1-8B MI455X (gfx1250) tutorial in ../README.md. Build context is
# this directory:
#
#   docker build -t llama31-8b-mi455x -f Dockerfile .
#
# The base is the gfx1250 Primus image of 2026-09-10 (Python 3.12 venv /opt/venv, ROCm 7.14
# as pip wheels, torch 2.11, triton 3.6, hipBLASLt with a tuned gfx1250 library), pinned by
# digest. Pulling it needs access to that repository.
FROM amdprimus/amdprimus:gfx1250-20260910@sha256:6e656de79e6c8d7f3b3db3690606c3e6cb0536ec871f0521735746015297c9a1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore
COPY requirements.txt constraints.txt /tmp/llama31_8b/

# 1) TorchTitan's missing dependencies and flydsl 0.3.4.1, which replaces the base image's
#    0.2.4, at the versions in constraints.txt. Nothing in the base image depends on flydsl.
# 2) The base image ships an editable Primus-Turbo without the gfx1250 attention. Primus'
#    launcher puts site-packages ahead of PYTHONPATH, so that copy would shadow the checkout
#    run_llama31_8b_e2e.sh puts on PYTHONPATH: remove it, and fail the build if any copy remains.
# 3) safe.directory: the container runs as root, and git (run by the Primus-Turbo build and by
#    Primus' launcher) refuses repositories owned by another user, here the host user's
#    checkouts in the mounted work directory. '*' trusts every repository the container sees;
#    mount only directories whose repositories you trust (README, section 2).
RUN cd /tmp/llama31_8b \
 && pip install -c constraints.txt -r requirements.txt \
 && (pip uninstall -y primus_turbo || true) \
 && git config --global --add safe.directory '*' \
 && python -c "import flydsl, torch, triton; assert flydsl.__version__ == '0.3.4.1', flydsl.__version__; print('flydsl', flydsl.__version__, 'torch', torch.__version__, 'triton', triton.__version__)" \
 && python -c "import importlib.util as u, sys; s = u.find_spec('primus_turbo'); sys.exit(s and 'a primus_turbo is still installed: %s' % s.origin)" \
 && cd / && rm -rf /tmp/llama31_8b

# Do not add anything that sets TORCH_BLAS_PREFER_HIPBLASLT=0 (profile.d, sitecustomize): Primus'
# MI455X settings and run_llama31_8b_e2e.sh use hipBLASLt with HIPBLASLT_TENSILE_LIBPATH pointing at the
# image's gfx1250 library.
CMD ["sleep", "infinity"]
