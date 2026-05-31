#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/transformer/rule_eval.out
#SBATCH --error=outputs/transformer/rule_eval.err

set -eo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
# Workdir comes from the dashboard (.env REMOTE_WORKDIR); fallback keeps the
# script runnable standalone. Pixi cache is a sibling of the project dir.
WORKDIR="${REMOTE_WORKDIR:-/leonardo_work/EUHPC_D30_031/industrial-infineon}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$(dirname "$WORKDIR")/pixi-cache-${USER}}"
cd "$WORKDIR"
mkdir -p outputs/transformer "$PIXI_CACHE_DIR"

# Evaluate the most recently trained transformer checkpoint (overwritten by
# each transformer training run). Writes rule_eval_summary.csv +
# generated_rule_counts.csv into outputs/transformer/ for the dashboard.
# CHECKPOINT/VOCAB can override the defaults to re-evaluate a specific archived
# run (runs/<job_id>/...); argparse takes the last --checkpoint/--vocab given.
pixi run python training_data/evaluate_rules.py \
  --checkpoint outputs/transformer/transformer_model.pt \
  --vocab outputs/transformer/vocab.json \
  --output-dir outputs/transformer \
  ${TRAIN_RATIO:+--train-ratio $TRAIN_RATIO} \
  ${VAL_RATIO:+--val-ratio $VAL_RATIO} \
  ${MAX_SEQUENCES:+--max-sequences $MAX_SEQUENCES} \
  ${CHECKPOINT:+--checkpoint $CHECKPOINT} \
  ${VOCAB:+--vocab $VOCAB} \
  ${DATA_DIR:+--data-dir $DATA_DIR}
