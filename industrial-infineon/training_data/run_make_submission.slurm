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
#SBATCH --output=outputs/transformer/make_submission.out
#SBATCH --error=outputs/transformer/make_submission.err

set -eo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
# Workdir comes from the dashboard (.env REMOTE_WORKDIR); fallback keeps the
# script runnable standalone. Pixi cache is a sibling of the project dir.
WORKDIR="${REMOTE_WORKDIR:-/leonardo_work/EUHPC_D30_031/industrial-infineon}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$(dirname "$WORKDIR")/pixi-cache-${USER}}"
cd "$WORKDIR"
mkdir -p outputs/transformer/submission "$PIXI_CACHE_DIR"

# Organizer eval inputs (eval_input_valid.csv / eval_input_anomaly.csv) are
# uploaded by the dashboard. INPUT_DIR can override their location.
INPUT_DIR="${INPUT_DIR:-participant_files}"
# Which task CSVs to (re)generate: "all" or a space-separated subset of
# "next-step", "completion", "anomaly".
TASKS="${TASKS:-all}"

# CHECKPOINT/VOCAB override the canonical model so a specific archived run
# (outputs/transformer/runs/<job_id>/...) can be scored. argparse keeps the
# last --checkpoint/--vocab it sees, so the overrides come after the defaults.
pixi run python training_data/make_submission.py \
  --input-dir "$INPUT_DIR" \
  --output-dir outputs/transformer/submission \
  --checkpoint outputs/transformer/transformer_model.pt \
  --vocab outputs/transformer/vocab.json \
  ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
  ${VOCAB:+--vocab "$VOCAB"} \
  --tasks $TASKS
