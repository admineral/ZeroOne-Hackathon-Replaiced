#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D30_031
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --output=outputs/transformer/train_transformer.out
#SBATCH --error=outputs/transformer/train_transformer.err

set -eo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
# Workdir comes from the dashboard (.env REMOTE_WORKDIR); fallback keeps the
# script runnable standalone. Pixi cache is a sibling of the project dir.
WORKDIR="${REMOTE_WORKDIR:-/leonardo_work/EUHPC_D30_031/industrial-infineon}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$(dirname "$WORKDIR")/pixi-cache-${USER}}"
cd "$WORKDIR"
mkdir -p outputs/transformer "$PIXI_CACHE_DIR"

# Clear the previous run's loss log up front so the dashboard shows "waiting"
# instead of the last run's curve while this run loads data / warms up. The
# training script reopens it ("w") once it starts logging epochs.
: > outputs/transformer/train_log.csv

# Sample GPU utilisation / memory / power every 2s for the whole run so the
# dashboard can show a real time-series (not just the peak). The trap ensures
# the sampler is stopped even if training fails under `set -e`.
nvidia-smi -i 0 \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,nounits -l 2 > outputs/transformer/gpu_timeline.csv 2>/dev/null &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# Single-process by default; for multi-GPU (DDP=1 with GPUS>1) launch one
# process per GPU via torchrun. train_transformer.py reads the torchrun env
# (WORLD_SIZE/RANK/LOCAL_RANK) to set up DistributedDataParallel.
LAUNCHER="pixi run python training_data/train_transformer.py"
if [ "${DDP:-0}" = "1" ] && [ "${GPUS:-1}" -gt 1 ]; then
  # Use the module form so we don't depend on the torchrun console script
  # being on PATH inside the pixi env.
  LAUNCHER="pixi run python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=${GPUS} training_data/train_transformer.py"
  echo "launching DDP across ${GPUS} GPUs via torch.distributed.run"
fi

$LAUNCHER \
  ${EPOCHS:+--epochs $EPOCHS} \
  ${LEARNING_RATE:+--learning-rate $LEARNING_RATE} \
  ${BATCH_SIZE:+--batch-size $BATCH_SIZE} \
  ${MAX_SEQ_LEN:+--max-seq-len $MAX_SEQ_LEN} \
  ${D_MODEL:+--d-model $D_MODEL} \
  ${NUM_LAYERS:+--num-layers $NUM_LAYERS} \
  ${NUM_HEADS:+--num-heads $NUM_HEADS} \
  ${DROPOUT:+--dropout $DROPOUT} \
  ${FAMILY_DROPOUT:+--family-dropout $FAMILY_DROPOUT} \
  ${WEIGHT_DECAY:+--weight-decay $WEIGHT_DECAY} \
  ${LABEL_SMOOTHING:+--label-smoothing $LABEL_SMOOTHING} \
  ${LR_SCHEDULE:+--lr-schedule $LR_SCHEDULE} \
  ${WARMUP_RATIO:+--warmup-ratio $WARMUP_RATIO} \
  ${NUM_WORKERS:+--num-workers $NUM_WORKERS} \
  ${TRAIN_RATIO:+--train-ratio $TRAIN_RATIO} \
  ${VAL_RATIO:+--val-ratio $VAL_RATIO} \
  ${SEED:+--seed $SEED} \
  ${MAX_SEQUENCES:+--max-sequences $MAX_SEQUENCES} \
  ${DATA_DIR:+--data-dir $DATA_DIR}

