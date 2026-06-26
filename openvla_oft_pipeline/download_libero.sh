#!/usr/bin/env bash
# Download all 4 LIBERO task suite datasets.
#
# DISK REQUIREMENTS:
#   ~4-8 GB compressed (zip), ~8-16 GB uncompressed.
#   Check free space first: df -h /path/to/your/data/drive
#
# USAGE:
#   # Default: downloads to datasets/libero/ relative to this script's location
#   bash download_libero.sh
#
#   # Download to a different drive (recommended if /home is full):
#   LIBERO_DATA_ROOT=/nvme/data/libero bash download_libero.sh
#
# NOTE: Box.com blocks HEAD requests but GET works fine — wget/curl will work.
# Do NOT use curl -I to test the link, use the actual download.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${SCRIPT_DIR}/../datasets/libero}"

mkdir -p "${LIBERO_DATA_ROOT}"
cd "${LIBERO_DATA_ROOT}"

echo "[Download] Target directory: ${LIBERO_DATA_ROOT}"
echo "[Download] Free space: $(df -h . | awk 'NR==2{print $4}') available"
echo ""

# Check minimum free space (need at least 10 GB)
AVAIL_KB=$(df -k . | awk 'NR==2{print $4}')
if [[ ${AVAIL_KB} -lt 10485760 ]]; then
    echo "ERROR: Less than 10 GB free in ${LIBERO_DATA_ROOT}"
    echo "       Current free: $(df -h . | awk 'NR==2{print $4}')"
    echo "       Free up space or set LIBERO_DATA_ROOT to a different filesystem:"
    echo "         LIBERO_DATA_ROOT=/nvme/data bash download_libero.sh"
    exit 1
fi

# ── Download function ─────────────────────────────────────────────────────────
download_suite() {
    local suite_name="$1"
    local url="$2"
    local zip_file="${suite_name}.zip"

    if [[ -d "${suite_name}" ]]; then
        local n_hdf5
        n_hdf5=$(find "${suite_name}" -name "*.hdf5" | wc -l)
        if [[ ${n_hdf5} -gt 0 ]]; then
            echo "[${suite_name}] Already downloaded (${n_hdf5} HDF5 files). Skipping."
            return
        fi
    fi

    echo "[${suite_name}] Downloading from Box.com..."
    # Box.com blocks HEAD/conditional requests — always do a full GET with -L for redirects
    if command -v wget &>/dev/null; then
        wget --content-disposition --show-progress -q -O "${zip_file}" "${url}"
    else
        curl -L --progress-bar -o "${zip_file}" "${url}"
    fi

    echo "[${suite_name}] Extracting..."
    unzip -q "${zip_file}" -d "${suite_name}_tmp"

    # LIBERO zips sometimes have an extra nesting level — flatten it
    # Structure varies: either suite_name/*.hdf5 or suite_name/suite_name/*.hdf5
    if [[ -d "${suite_name}_tmp/${suite_name}" ]]; then
        mv "${suite_name}_tmp/${suite_name}" "${suite_name}"
        rmdir "${suite_name}_tmp"
    else
        mv "${suite_name}_tmp" "${suite_name}"
    fi

    rm -f "${zip_file}"

    local n_hdf5
    n_hdf5=$(find "${suite_name}" -name "*.hdf5" | wc -l)
    echo "[${suite_name}] Done. ${n_hdf5} HDF5 files extracted."
}

# ── Suite downloads ───────────────────────────────────────────────────────────
# URLs from https://libero-project.github.io/datasets (verified June 2026)
download_suite "libero_spatial" \
    "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip"

download_suite "libero_object" \
    "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip"

download_suite "libero_goal" \
    "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip"

download_suite "libero_long" \
    "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== LIBERO Download Summary ==="
for suite in libero_spatial libero_object libero_goal libero_long; do
    if [[ -d "${suite}" ]]; then
        n=$(find "${suite}" -name "*.hdf5" | wc -l)
        size=$(du -sh "${suite}" | cut -f1)
        echo "  ${suite}: ${n} HDF5 files, ${size}"
    else
        echo "  ${suite}: MISSING"
    fi
done

echo ""
echo "Data root: ${LIBERO_DATA_ROOT}"
echo "Set this path as --data_root when running training and evaluation scripts."
