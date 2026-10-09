#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Llama-3.1-8B BF16 pretraining on one MI455X (gfx1250) with Primus (TorchTitan backend) and
# Primus-Turbo. Runs Primus' MI455X recipe, see README.md.
#
#   run_llama31_8b_e2e.sh [steps=20] [tag]
#
# Runs on the host; training runs inside the container $CT (image: docker/Dockerfile), which
# must see host paths at the same place (mount the work directory at the same path, README
# section 2).
#
# The recipe is used unchanged: the launcher sets the variables it reads from the environment
# (PRIMUS_HF_ASSETS_PATH, PRIMUS_WORKSPACE, PRIMUS_EXP_NAME) and passes Primus config
# overrides on the command line: --training.steps <steps>, plus
# --primus_turbo.enable_mm_layout_workaround false when MM_LAYOUT_WORKAROUND=0.
#
# Required environment:
#   PRIMUS      Primus checkout with the MI455X Llama-3.1-8B recipe and the
#               third_party/torchtitan submodule.
#   HF_ASSETS   directory holding the Llama-3.1-8B tokenizer and config files.
# Optional:
#   TURBO       Primus-Turbo checkout with its C++ extension built (default: the checkout this
#               script is in).
#   CT          container name (default: llama31-8b-mi455x).
#   WORKSPACE   directory for the logs and Primus' output (default: llama31_8b_runs next to
#               $TURBO).
#   MM_LAYOUT_WORKAROUND=0   run the backward GEMMs without the workaround (~10x slower).
#   SKIP_GPU_IDLE_CHECK=1    start even though another process holds a GPU on this host.
#
# Exit status: that of the training run, or 2 when training succeeded but a log check failed.
set -euo pipefail

STEPS=${1:-20}
TAG=${2:-llama31_8b_$(date +%m%d_%H%M%S)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PRIMUS=${PRIMUS:?set PRIMUS to the Primus checkout}
HF_ASSETS=${HF_ASSETS:?set HF_ASSETS to the Llama-3.1-8B tokenizer/config directory}
TURBO=${TURBO:-$(cd "$HERE/../.." && pwd)}
CT=${CT:-llama31-8b-mi455x}
WORKSPACE=${WORKSPACE:-$(dirname "$TURBO")/llama31_8b_runs}
MM_LAYOUT_WORKAROUND=${MM_LAYOUT_WORKAROUND:-1}
RECIPE_REL=examples/torchtitan/configs/MI455X/llama3.1_8B-BF16-pretrain.yaml
FLYDSL_VERSION_PY=primus_turbo/flydsl/attention/gfx1250/flydsl_version.py

die() {
  echo "run_llama31_8b_e2e.sh: $*" >&2
  exit 1
}

[[ $STEPS =~ ^[1-9][0-9]*$ ]] || die "steps must be a positive integer, got '$STEPS'"
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || die "the tag may contain only letters, digits, '.', '_' and '-'"
OVERRIDES=(--training.steps "$STEPS")
case $MM_LAYOUT_WORKAROUND in
  1) MMLW=true ;;
  0) MMLW=false OVERRIDES+=(--primus_turbo.enable_mm_layout_workaround false) ;;
  *) die "MM_LAYOUT_WORKAROUND must be 0 or 1, got '$MM_LAYOUT_WORKAROUND'" ;;
esac
# Absolute paths: the container resolves them, from another working directory.
PRIMUS=$(cd "$PRIMUS" && pwd) || die "PRIMUS=$PRIMUS is not a directory"
HF_ASSETS=$(cd "$HF_ASSETS" && pwd) || die "HF_ASSETS=$HF_ASSETS is not a directory"
TURBO=$(cd "$TURBO" && pwd) || die "TURBO=$TURBO is not a directory"
mkdir -p "$WORKSPACE" && WORKSPACE=$(cd "$WORKSPACE" && pwd)
RECIPE=$PRIMUS/$RECIPE_REL

# ---- Checks that need no GPU ----------------------------------------------------------------
[[ -f $PRIMUS/runner/primus-cli ]] || die "$PRIMUS is not a Primus checkout (no runner/primus-cli)"
[[ -f $RECIPE ]] ||
  die "$PRIMUS has no $RECIPE_REL: use a Primus with the MI455X Llama-3.1-8B recipe (README, Requirements)"
[[ -f $PRIMUS/third_party/torchtitan/pyproject.toml ]] ||
  die "no TorchTitan in $PRIMUS: git -C $PRIMUS submodule update --init third_party/torchtitan"
[[ -f $TURBO/$FLYDSL_VERSION_PY && -f $TURBO/primus_turbo/pytorch/core/mm_layout_workaround.py ]] ||
  die "$TURBO lacks the gfx1250 FlyDSL attention or the mm layout workaround"
compgen -G "$TURBO/primus_turbo/pytorch/_C*.so" >/dev/null ||
  die "the Primus-Turbo C++ extension is not built in $TURBO (README, section 3)"
[[ -f $HF_ASSETS/tokenizer.json && -f $HF_ASSETS/tokenizer_config.json ]] ||
  die "no tokenizer.json / tokenizer_config.json in $HF_ASSETS"
[[ $(docker inspect -f '{{.State.Running}}' "$CT" 2>/dev/null) == true ]] ||
  die "container $CT is not running (README, section 2)"
for p in "$PRIMUS" "$TURBO" "$HF_ASSETS" "$WORKSPACE"; do
  docker exec "$CT" test -d "$p" ||
    die "container $CT does not see $p: mount the work directory at the same path (README, section 2)"
done
# The flydsl release the Primus-Turbo checkout pins in setup.py (empty when it pins none).
TURBO_FLYDSL=$(grep -oE 'flydsl==[0-9][0-9A-Za-z.+]*' "$TURBO/setup.py" 2>/dev/null | head -n 1 | cut -d= -f3 || true)

# In the container, importing nothing that opens the GPU: the installed flydsl must satisfy
# the gfx1250 kernels' FLYDSL_REQUIREMENT (read from the checkout) and be the release the
# checkout pins; no installed primus_turbo may shadow $TURBO (Primus' launcher puts
# site-packages ahead of PYTHONPATH); and the image's own hipBLASLt gfx1250 library is located
# for HIPBLASLT_TENSILE_LIBPATH (not the incomplete copy under _rocm_sdk_devel). No bytecode is
# written into the checkout.
BLAS_LIB=$(docker exec -w /tmp -e PYTHONDONTWRITEBYTECODE=1 -e TURBO="$TURBO" -e TURBO_FLYDSL="$TURBO_FLYDSL" \
  -e FLYDSL_VERSION_PY="$TURBO/$FLYDSL_VERSION_PY" "$CT" python -c '
import glob, importlib.metadata as md, importlib.util as u, os, sys
from packaging.specifiers import SpecifierSet
from packaging.version import Version

spec = u.spec_from_file_location("gfx1250_flydsl_version", os.environ["FLYDSL_VERSION_PY"])
fv = u.module_from_spec(spec)
spec.loader.exec_module(fv)  # standard library only; does not import flydsl
try:
    v = md.version("flydsl")
except md.PackageNotFoundError:
    v = None
if v is None or not SpecifierSet(fv.FLYDSL_REQUIREMENT).contains(Version(v), prereleases=True):
    sys.exit("the gfx1250 attention needs flydsl%s, the container has %s: rebuild the image from docker/"
             % (fv.FLYDSL_REQUIREMENT, v))
pin = os.environ["TURBO_FLYDSL"]
if pin and pin != v:
    sys.exit("%s pins flydsl==%s in setup.py but the container has flydsl %s: the checkout and the image"
             " must use the same release (README, Requirements)" % (os.environ["TURBO"], pin, v))
s = u.find_spec("primus_turbo")
if s:
    sys.exit("the installed primus_turbo %s would shadow the checkout: pip uninstall -y primus_turbo" % s.origin)
s = u.find_spec("_rocm_sdk_libraries_gfx1250")
lib = os.path.join(list(s.submodule_search_locations)[0], "lib/hipblaslt/library/gfx1250") if s else ""
if not glob.glob(os.path.join(lib, "TensileLibrary_lazy_gfx1250.dat*")):
    sys.exit("the image has no hipBLASLt gfx1250 library (_rocm_sdk_libraries_gfx1250)")
print(lib)
') || die "container check failed (see above)"

# One GPU process per card: do not start while another process holds a GPU.
if [[ ${SKIP_GPU_IDLE_CHECK:-0} != 1 && -n $(ls -A /sys/class/kfd/kfd/proc 2>/dev/null) ]]; then
  die "another process holds a GPU on this host (/sys/class/kfd/kfd/proc is not empty);" \
    "set SKIP_GPU_IDLE_CHECK=1 if it is on a GPU this container does not use"
fi

# ---- Run ------------------------------------------------------------------------------------
LOG=$WORKSPACE/$TAG.log
rev() { git -C "$1" describe --always --dirty --abbrev=8 2>/dev/null || echo unknown; }
echo "[$(date +%T)] $TAG: steps=$STEPS mm_layout_workaround=$MMLW log=$LOG"
echo "  primus-turbo $TURBO ($(rev "$TURBO")), primus $PRIMUS ($(rev "$PRIMUS"))"
echo "  config $RECIPE_REL ${OVERRIDES[*]}"

# Runs in the container, which expands the variables (they arrive through docker exec -e) and
# receives the overrides as its arguments. bash -c, not bash -lc: a login shell may source
# profile scripts that override the BLAS settings. FLYDSL_RUNTIME_CACHE_DIR is fresh per run,
# so step 1 includes the FlyDSL JIT and no kernel compiled from another checkout can be picked
# up; the Triton cache persists.
read -r -d '' INNER <<'EOS' || true
ulimit -c 0
if [[ -n ${PRIMUS_TURBO_ATTN_BACKEND:-} ]]; then
  echo "note: PRIMUS_TURBO_ATTN_BACKEND=$PRIMUS_TURBO_ATTN_BACKEND pins the attention backend"
fi
cd "$PRIMUS"
exec bash runner/primus-cli direct --log_file "$LAUNCHER_LOG" -- train pretrain --config "$RECIPE" "$@"
EOS

T0=$(date +%s)
set +e
docker exec \
  -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 -e MASTER_PORT=$((20000 + RANDOM % 20000)) \
  -e PRIMUS_GPU_MODEL=MI455X -e PRIMUS_EXP_NAME="$TAG" -e PRIMUS_WORKSPACE="$WORKSPACE" \
  -e PRIMUS_HF_ASSETS_PATH="$HF_ASSETS" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 -e HIPBLASLT_TENSILE_LIBPATH="$BLAS_LIB" \
  -e FLYDSL_RUNTIME_CACHE_DIR="/tmp/flydsl_cache_$TAG" -e TRITON_CACHE_DIR=/tmp/triton_cache_llama31_8b \
  -e PYTHONPATH="$TURBO" -e PRIMUS="$PRIMUS" -e RECIPE="$RECIPE" \
  -e LAUNCHER_LOG="$WORKSPACE/$TAG.launcher.log" \
  "$CT" bash -c "$INNER" run_llama31_8b_e2e "${OVERRIDES[@]}" >"$LOG" 2>&1
RC=$?
set -e
T1=$(date +%s)

# ---- Summary --------------------------------------------------------------------------------
# Steps 1-5 include the warm-up (FlyDSL JIT on first use, allocator growth): report 6..N.
# Tokens per step come from torchtitan's "Trainer is initialized with ..." line.
if command -v python3 >/dev/null; then PY=(python3); else PY=(docker exec -i "$CT" python); fi
CHECKS_FAILED=0
"${PY[@]}" - "$LOG" "$MMLW" <<'EOF' || CHECKS_FAILED=1
import math
import re
import statistics
import sys

log, mmlw = sys.argv[1], sys.argv[2] == "true"
with open(log, errors="replace") as f:
    text = re.sub(r"\x1b\[[0-9;]*m", "", f.read())
step_re = re.compile(
    r"step:\s*(\d+)\s+loss:\s*(\S+)\s+grad_norm:\s*(\S+)\s+"
    r"memory:\s*([\d.]+)GiB\(([\d.]+)%\)\s+tps:\s*([\d,]+)"
)
init = re.search(
    r"local batch size (\d+), global batch size \d+, gradient accumulation steps (\d+), sequence length (\d+)",
    text,
)
tokens_per_step = int(init[1]) * int(init[2]) * int(init[3]) if init else None


def num(s):
    try:
        return float(s)
    except ValueError:
        return math.nan


rows = [
    (int(m[1]), num(m[2]), num(m[3]), float(m[4]), float(m[5]), int(m[6].replace(",", "")))
    for m in step_re.finditer(text)
]
if rows:
    steady = [r for r in rows if r[0] >= 6] or rows
    tps = statistics.median(r[5] for r in steady)
    print(
        f"steps logged: {len(rows)}, loss {rows[0][1]:.4f} -> {rows[-1][1]:.4f}, "
        f"peak memory {max(r[3] for r in rows):.2f} GiB ({max(r[4] for r in rows):.2f}% of HBM)"
    )
    if tps > 0:
        ms = f" = {tokens_per_step / tps * 1000:,.1f} ms/step" if tokens_per_step else ""
        print(f"steady state (steps {steady[0][0]}-{steady[-1][0]}): median {tps:,.0f} tokens/s{ms}")
    print("loss and grad_norm finite:", all(math.isfinite(r[1]) and math.isfinite(r[2]) for r in rows))
else:
    print("no training steps found in the log")

checks = [
    ("TurboAttention installed", "Primus-Turbo Attention successfully installed" in text),
    ("no Primus-Turbo backend fallback", "fallback backend" not in text),
    ("Triton transpose available", "Triton is unavailable" not in text),
]
if mmlw:
    checks.append(("GEMM layout workaround installed", "aten::mm layout workaround installed" in text))
failed = [name for name, ok in checks if not ok]
print("log checks:", "FAILED: " + ", ".join(failed) if failed else "all passed")
sys.exit(1 if failed else 0)
EOF
echo "rc=$RC wall=$((T1 - T0)) s (launch to exit, including model init and JIT) log=$LOG"
if ((RC != 0)); then exit "$RC"; fi
if ((CHECKS_FAILED)); then exit 2; fi
