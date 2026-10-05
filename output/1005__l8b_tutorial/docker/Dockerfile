# fa-tune:deps for Llama-3.1-8B e2e on MI455X: amdprimus/amdprimus:gfx1250-20260910 + torchtitan deps,
# minus two traps of the stock image. Build context = this directory:
#   docker build -t fa-tune:deps -f Dockerfile .
# The base is private on Docker Hub: docker login, or docker save / docker load it from a host that has it.
FROM amdprimus/amdprimus:gfx1250-20260910@sha256:6e656de79e6c8d7f3b3db3690606c3e6cb0536ec871f0521735746015297c9a1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore
COPY requirements.fa-tune-deps-layer.txt requirements.fa-repro-container.txt constraints.a0-fa-tune-deps.txt /tmp/fa-tune/

# 1) torchtitan's deps (tyro, torchdata, tabulate, tensorboard, wandb + their deps). The constraints file is the
#    A0 fa-tune:deps freeze, so torch, triton, the ROCm 7.14 wheels, TE and flydsl 0.2.4 are never replaced.
#    Without tensorboard, Primus' turbo-attention patch is skipped and torchtitan fails with an enable_gqa TypeError.
# 2) tomli / psutil, which the A0 container picked up later.
# 3) The base ships an editable primus-turbo (0.4.1.dev12, /workspace/Primus-Turbo, no gfx1250 attention) whose
#    import hook wins over PYTHONPATH: remove it so the branch checkout is the one that runs.
RUN cd /tmp/fa-tune \
 && pip install -c constraints.a0-fa-tune-deps.txt -r requirements.fa-tune-deps-layer.txt \
 && pip install -c constraints.a0-fa-tune-deps.txt -r requirements.fa-repro-container.txt \
 && (pip uninstall -y primus_turbo || true) \
 && git config --global --add safe.directory '*' \
 && python -c "import flydsl, torch, triton; assert flydsl.__version__ == '0.2.4', flydsl.__version__; print(torch.__version__, triton.__version__)" \
 && rm -rf /tmp/fa-tune

# Do NOT add anything that sets TORCH_BLAS_PREFER_HIPBLASLT=0 (profile.d / sitecustomize): hipBLASLt works on
# gfx1250 with the image's tuned library once HIPBLASLT_TENSILE_LIBPATH points at it (the launcher sets both).
CMD ["sleep", "infinity"]
