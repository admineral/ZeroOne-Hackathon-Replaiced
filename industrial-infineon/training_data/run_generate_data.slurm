#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:20:00
#SBATCH --output=outputs/generate_data.out
#SBATCH --error=outputs/generate_data.err

set -eo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
# Workdir comes from the dashboard (.env REMOTE_WORKDIR); fallback keeps the
# script runnable standalone. Pixi cache is a sibling of the project dir.
WORKDIR="${REMOTE_WORKDIR:-/leonardo_work/EUHPC_D30_031/industrial-infineon}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$(dirname "$WORKDIR")/pixi-cache-${USER}}"
cd "$WORKDIR"
mkdir -p outputs "$PIXI_CACHE_DIR"

# Rule-based data generation is CPU-only; the GPU request just matches the
# training reservation so the job schedules. Large counts (100k+ per family)
# hold all sequences + a dedup set in RAM — 120 GB avoids Slurm OOM kills.
# OUTPUT_DIR (set by the dashboard, e.g. datasets/<id>) keeps each generated
# dataset in its own folder instead of overwriting training_data/. generate_all
# mkdir -p's it, but we create it up front too so the run never races on it.
if [ -n "${OUTPUT_DIR:-}" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

pixi run python training_data/generate_all.py \
  ${COUNT:+--count $COUNT} \
  ${SEED:+--seed $SEED} \
  ${FAMILIES:+--families $FAMILIES} \
  ${OUTPUT_DIR:+--output-dir $OUTPUT_DIR}
