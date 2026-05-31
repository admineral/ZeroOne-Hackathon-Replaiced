#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=outputs/preprocess.out
#SBATCH --error=outputs/preprocess.err

set -eo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
# Workdir comes from the dashboard (.env REMOTE_WORKDIR); fallback keeps the
# script runnable standalone. Pixi cache is a sibling of the project dir.
WORKDIR="${REMOTE_WORKDIR:-/leonardo_work/EUHPC_D30_031/industrial-infineon}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$(dirname "$WORKDIR")/pixi-cache-${USER}}"
cd "$WORKDIR"
mkdir -p outputs "$PIXI_CACHE_DIR"

# One-time CPU preprocessing: stream the long-format CSVs into a packed memmap
# blob under <data-dir>/packed/ (tokens.bin + offsets.npy + families.npy +
# vocab.json + meta.json) so training/eval can run on the full ~30M-sequence
# dataset with near-zero RAM. This is CPU/IO-bound -- the GPU on the boost node
# (required by the reservation) stays idle. DATA_DIR selects which dataset to
# pack; MAX_PER_FAMILY caps sequences per family (mostly for quick tests).
pixi run python training_data/preprocess_dataset.py \
  ${DATA_DIR:+--data-dir $DATA_DIR} \
  ${MAX_PER_FAMILY:+--max-per-family $MAX_PER_FAMILY} \
  ${SEED:+--seed $SEED}
