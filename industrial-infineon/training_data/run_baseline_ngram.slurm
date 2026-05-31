#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --output=outputs/baseline_ngram.out
#SBATCH --error=outputs/baseline_ngram.err

set -eo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
# Workdir comes from the dashboard (.env REMOTE_WORKDIR); fallback keeps the
# script runnable standalone. Pixi cache is a sibling of the project dir.
WORKDIR="${REMOTE_WORKDIR:-/leonardo_work/EUHPC_D30_031/industrial-infineon}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$(dirname "$WORKDIR")/pixi-cache-${USER}}"
cd "$WORKDIR"
mkdir -p outputs "$PIXI_CACHE_DIR"

pixi run python training_data/baseline_ngram.py \
  ${DATA_DIR:+--data-dir $DATA_DIR}

