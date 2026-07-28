#!/usr/bin/env bash
# install.sh — build ns-3.40 + ns3-ai + the LyMAPPO scenario (Linux/WSL2).
#
# Usage:
#   NS3_ROOT=~/ns-3-dev bash install.sh     # NS3_ROOT defaults to ~/ns-3-dev
#
# Prerequisites (see docs/INSTALL.md):
#   - apt packages installed (build-essential, cmake, ninja-build, git,
#     python3-dev, pybind11-dev, libgtk-3-dev, libxml2-dev, libssl-dev,
#     libsqlite3-dev, sqlite3, boost, protobuf, gcc-12/g++-12)
#   - ns-3.40 cloned at $NS3_ROOT

set -euo pipefail

# WSL inherits the Windows PATH; mingw/Anaconda headers leaking into the
# CMake include search break the ns-3 stats module and cairo linking.
# Build with a Linux-only environment.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH PKG_CONFIG_PATH \
    || true

# GCC 13.3 (Ubuntu 24.04 default) hits an internal compiler error
# (try_forward_edges) on ns-3.40 callback/propagation headers at -O2/-O3.
if command -v g++-12 >/dev/null 2>&1; then
    export CC=gcc-12
    export CXX=g++-12
    echo "[install] using ${CXX} (avoids GCC 13 ICE on ns-3.40)"
fi

NS3_ROOT="${NS3_ROOT:-${HOME}/ns-3-dev}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "[install] NS3_ROOT  = ${NS3_ROOT}"
echo "[install] REPO_ROOT = ${REPO_ROOT}"

# ---------- pre-flight ------------------------------------------------------
if [ ! -d "${NS3_ROOT}" ]; then
    echo "[install] ERROR: ${NS3_ROOT} not found. Clone ns-3.40 first:"
    echo "  git clone --depth 1 --branch ns-3.40 https://gitlab.com/nsnam/ns-3-dev.git ${NS3_ROOT}"
    exit 1
fi
for cmd in git cmake g++ python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[install] ERROR: '$cmd' not on PATH. Run the apt step in docs/INSTALL.md."
        exit 2
    fi
done

# ---------- ns3-ai contrib --------------------------------------------------
NS3AI_DIR="${NS3_ROOT}/contrib/ai"
if [ ! -d "${NS3AI_DIR}" ]; then
    echo "[install] cloning ns3-ai"
    git clone --depth 1 https://github.com/hust-diangroup/ns3-ai.git "${NS3AI_DIR}"
    pushd "${NS3AI_DIR}" >/dev/null
    if [ -x ./install.sh ]; then
        ./install.sh
    fi
    popd >/dev/null
else
    echo "[install] ns3-ai already present -- skip clone"
fi

# Some bundled ns3-ai examples do not compile against the ns-3.40 API.
# Removing the directories is not enough: their add_subdirectory() references
# must also be commented out, or CMake configure fails.
if [ -d "${NS3AI_DIR}/examples" ]; then
    rm -rf "${NS3AI_DIR}/examples/rate-control/thompson-sampling"
    rm -rf "${NS3AI_DIR}/examples/rl-tcp"
    rm -rf "${NS3AI_DIR}/examples/multi-bss"
    sed -i -E 's/^([[:space:]]*)add_subdirectory\((rl-tcp|multi-bss)\)/\1# pruned (ns-3.40 incompatible): add_subdirectory(\2)/' \
        "${NS3AI_DIR}/examples/CMakeLists.txt"
    if [ -f "${NS3AI_DIR}/examples/rate-control/CMakeLists.txt" ]; then
        sed -i -E 's|^([[:space:]]*)add_subdirectory\((thompson-sampling)\)|\1# pruned (ns-3.40 incompatible): add_subdirectory(\2)|' \
            "${NS3AI_DIR}/examples/rate-control/CMakeLists.txt"
    fi
    echo "[install] pruned ns3-ai examples incompatible with ns-3.40"
fi

# ---------- sync scenario + drivers ----------------------------------------
DST="${NS3AI_DIR}/examples/feddrl"
mkdir -p "${DST}"
cp -f "${REPO_ROOT}"/scenario/*.h "${DST}/"
cp -f "${REPO_ROOT}"/scenario/*.cc "${DST}/"
cp -f "${REPO_ROOT}/scenario/CMakeLists.txt" "${DST}/"
cp -f "${REPO_ROOT}"/drivers/*.py "${DST}/"
cp -f "${REPO_ROOT}/models/mappo.py" "${REPO_ROOT}/models/networks.py" "${DST}/"
echo "[install] synced scenario/ + drivers/ + models/ -> contrib/ai/examples/feddrl/"

EX_CMAKE="${NS3AI_DIR}/examples/CMakeLists.txt"
if ! grep -q '^add_subdirectory(feddrl)' "${EX_CMAKE}"; then
    echo 'add_subdirectory(feddrl)' >> "${EX_CMAKE}"
    echo "[install] registered feddrl in contrib/ai/examples/CMakeLists.txt"
fi

# ---------- build -----------------------------------------------------------
pushd "${NS3_ROOT}" >/dev/null
if [ ! -f .build_configured ]; then
    # 'default' profile (-O2): the 'optimized' profile (-O3 -march=native)
    # triggers the GCC ICE class above on ns-3.40.
    ./ns3 configure --build-profile=default --enable-examples --disable-tests
    touch .build_configured
fi
export CXXFLAGS="-DNS3_AI_AVAILABLE ${CXXFLAGS:-}"
./ns3 build

# ---------- verify ----------------------------------------------------------
BIN="$(find build/contrib/ai/examples/feddrl -maxdepth 1 -type f -name '*feddrl*' 2>/dev/null | head -1 || true)"
popd >/dev/null
if [ -n "${BIN}" ]; then
    echo "[install] DONE — scenario binary: ${BIN}"
    echo "[install] next: bash scripts/smoke.sh"
else
    echo "[install] WARN: build finished but scenario binary not found under build/contrib/ai/examples/feddrl"
    exit 3
fi
