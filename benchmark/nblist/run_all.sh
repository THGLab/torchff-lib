#!/usr/bin/env bash
# Run all neighbor-list benchmarks end-to-end, then generate the plot.
#
# Usage (via Slurm interactive QOS):
#   srun --nodes 1 --qos interactive --time 1:00:00 --constraint gpu --account=m2834 \
#     bash benchmark/nblist/run_all.sh
#
# Optional: pass a CSV path as $1 to override the default output location.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/benchmark/nblist"
OUTPUT_CSV="${1:-${SCRIPT_DIR}/nblist_benchmark.csv}"

echo "============================================="
echo "  Neighbor-list benchmark suite"
echo "  Output CSV: ${OUTPUT_CSV}"
echo "============================================="

module load conda 2>/dev/null || true
eval "$(conda shell.bash hook)"

run_benchmark() {
    local env_name="$1"
    local script="$2"
    echo ""
    echo ">>> Activating env: ${env_name}"
    mamba activate "${env_name}"
    echo ">>> Running: ${script}"
    python "${script}" -o "${OUTPUT_CSV}"
    echo ">>> Done: ${script}"
    mamba deactivate
}

# TorchFF (requires torchff to be installed)
run_benchmark "openmm-torch-py312-cu124" "${SCRIPT_DIR}/benchmark_torchff.py"

# Vesin
run_benchmark "vesin-bench" "${SCRIPT_DIR}/benchmark_vesin.py"

# ALCHEMI
run_benchmark "alchemi-bench" "${SCRIPT_DIR}/benchmark_alchemi.py"

# TorchMD-Net
run_benchmark "torchmd-net" "${SCRIPT_DIR}/benchmark_torchmd.py"

# Plot (use any env that has matplotlib + numpy)
echo ""
echo ">>> Generating plot from ${OUTPUT_CSV}"
mamba activate "openmm-torch-py312-cu124"
PDF_PATH="${OUTPUT_CSV%.csv}.pdf"
python "${SCRIPT_DIR}/plot_nblist.py" --input "${OUTPUT_CSV}" --output "${PDF_PATH}"
mamba deactivate

echo ""
echo "============================================="
echo "  All benchmarks complete."
echo "  CSV: ${OUTPUT_CSV}"
echo "  PDF: ${PDF_PATH}"
echo "============================================="
