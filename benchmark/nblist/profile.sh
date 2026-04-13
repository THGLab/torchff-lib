#!/usr/bin/bash -c
module load conda
mamba activate openmm-torch-py312-cu124
module load cudatoolkit
# OUTPUT_REP="benchmark_torchff.nsys-rep"
# nsys profile \
#     --force-overwrite=true \
#     -o "'"$OUTPUT_REP"'" \
#     --stats=true \
#     --trace=cuda,nvtx,cudnn,cublas \
#     --cuda-graph-trace=node \
#     python benchmark_torchff.py

# ncu --set full -o ncu_report python benchmark_torchff.py

for k in \
  "assign_cell_index_kernel" \
  "compute_bounding_box_kernel" \
  "find_interacting_clusters_kernel" \
  "build_neighbor_list_cell_list_kernel"
do
  ncu \
    --set basic \
    --kernel-name-base demangled \
    -k "regex:.*${k}.*" \
    --launch-count 5 \
    -o "ncu_${k}" \
    -f \
    python benchmark_torchff.py --atoms 1000000
done