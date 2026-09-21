#!/bin/sh
# Stand-in for rocminfo inside a container started WITHOUT /dev/kfd.
#
# aiter's chip_info.get_gfx_runtime() calls _detect_native() directly and ignores the
# GPU_ARCHS override, so `import aiter` cannot complete where there is no card. It only
# needs one line matching \b(gfx\w+)\b. This is bind-mounted read-only over the real
# binary for the lifetime of a --rm container; nothing on the host is modified.
#
# ONLY for the card-less compile-only screen. Never mount this into a container that has
# /dev/kfd: a fake arch answer on a real card is how an audit reports on the wrong target.
echo "  Name:                    amdgcn-amd-amdhsa--gfx1250"
echo "  Marketing Name:          AMD Radeon Graphics"
exit 0
