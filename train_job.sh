#!/bin/bash
set -e

# Test locally - don't need to clone since we're already in the repo

# Set environment variables for local testing
export WANDB_API_KEY=${WANDB_API_KEY:-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}
export EPOCHS=${EPOCHS:-30}
export BATCH_SIZE=${BATCH_SIZE:-128}
export LR=${LR:-1e-4}
export D_MODEL=${D_MODEL:-256}
export NUM_MP_LAYERS=${NUM_MP_LAYERS:-4}
export MAX_GENES=${MAX_GENES:-2000}
export GRN_STRATEGY=${GRN_STRATEGY:-message_passing}
export JOB_NAME=${JOB_NAME:-causalflow-local-test-$(date +%Y%m%d-%H%M%S)}

# Use local data directory
DATA_DIR="/tmp"
OUTPUT_DIR="/tmp/causalflow_output"
mkdir -p $OUTPUT_DIR

# Don't need to download - we have the file locally
echo "Using local Norman dataset..."

echo 'Starting training locally...'
python train.py \
  --data-dir $DATA_DIR \
  --output-dir $OUTPUT_DIR \
  --max-genes $MAX_GENES \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --d-model $D_MODEL \
  --num-layers $NUM_MP_LAYERS \
  --grn-strategy $GRN_STRATEGY \
  --grn-reg \
  --wandb \
  --wandb-project causalflow \
  --run-name $JOB_NAME \
  --save-interval 5

echo 'Done!'
