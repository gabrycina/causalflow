#!/bin/bash
# =============================================================================
# CausalFlow: Launch Training Job on Nebius AI Cloud
# =============================================================================
#
# Prerequisites:
#   1. Run setup_nebius.sh first (already done - S3 credentials configured)
#   2. Code uploaded to S3: s3://causalflow-experiments/code/causalflow.tar.gz
#
# =============================================================================

set -e

# ---- Configuration ----
PROJECT_ID="project-e00h8qe9pp3twpwbzf"
SUBNET_ID="vpcsubnet-e00w88z4k3eq6s6wk5"
BUCKET_NAME="causalflow-experiments"
REGION="eu-north1"
IMAGE="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"
PLATFORM="gpu-a100-sxm"
PRESET="1gpu-16vcpu-200gb"
DISK_SIZE="450Gi"
MAX_TIME="48h"

# Training hyperparameters
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-4}
D_MODEL=${D_MODEL:-256}
NUM_MP_LAYERS=${NUM_MP_LAYERS:-4}
MAX_GENES=${MAX_GENES:-2000}
GRN_STRATEGY=${GRN_STRATEGY:-message_passing}

# Container paths
CONTAINER_CODE_DIR="/workspace/code"
CONTAINER_DATA_DIR="/workspace/data"
CONTAINER_OUTPUT_DIR="/workspace/output"

JOB_NAME="causalflow-$(date +%Y%m%d-%H%M%S)"

echo "========================================="
echo "CausalFlow Training Job: $JOB_NAME"
echo "========================================="
echo "Image: $IMAGE"
echo "Platform: $PLATFORM / $PRESET}"
echo "Max time: $MAX_TIME"
echo ""
echo "Training config:"
echo "  Epochs: $EPOCHS, Batch: $BATCH_SIZE, LR: $LR"
echo "  D_model: $D_MODEL, MP layers: $NUM_MP_LAYERS"
echo "========================================="

# Training command
TRAIN_CMD="bash -c \"
set -e
echo 'Extracting code from S3...'
aws s3 cp s3://$BUCKET_NAME/code/causalflow.tar.gz /tmp/causalflow.tar.gz \
  --profile nebius --endpoint-url https://storage.$REGION.nebius.cloud
mkdir -p $CONTAINER_CODE_DIR
tar -xzf /tmp/causalflow.tar.gz -C $CONTAINER_CODE_DIR
cd $CONTAINER_CODE_DIR

echo 'Installing dependencies...'
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
pip install --quiet decoupler numpy scipy scikit-learn anndata scanpy scvi-tools wandb tqdm pyyaml pandas
pip install --quiet pertpy

echo 'Starting training...'
python train.py \
  --data-dir $CONTAINER_DATA_DIR \
  --output-dir $CONTAINER_OUTPUT_DIR \
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

echo 'Copying outputs to S3...'
aws s3 cp $CONTAINER_OUTPUT_DIR s3://$BUCKET_NAME/output/$JOB_NAME/ \
  --profile nebius --endpoint-url https://storage.$REGION.nebius.cloud --recursive
echo 'Done!'
\""

# Launch job
JOB_RESULT=$(nebius ai job create \
  --name "$JOB_NAME" \
  --image "$IMAGE" \
  --platform "$PLATFORM" \
  --preset "$PRESET" \
  --disk-size "$DISK_SIZE" \
  --volume "nebius-storage://$BUCKET_NAME:$CONTAINER_DATA_DIR" \
  --volume "nebius-storage://$BUCKET_NAME:$CONTAINER_OUTPUT_DIR" \
  --container-command bash \
  --args "-c '$TRAIN_CMD'" \
  --working-dir "$CONTAINER_CODE_DIR" \
  --subnet-id "$SUBNET_ID" \
  --timeout "$MAX_TIME" \
  --restart-policy never \
  --format json 2>&1)

JOB_ID=$(echo "$JOB_RESULT" | jq -r '.metadata.id' 2>/dev/null || echo "")

if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "null" ]; then
  echo "ERROR: Failed to create job"
  echo "$JOB_RESULT"
  exit 1
fi

echo ""
echo "Job created: $JOB_ID"
echo ""
echo "Monitor:"
echo "  nebius ai job get $JOB_ID"
echo ""
echo "Watch logs:"
echo "  nebius ai job logs $JOB_ID -f"
