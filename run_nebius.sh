#!/bin/bash
# =============================================================================
# CausalFlow: Launch Training Job on Nebius AI Cloud
# =============================================================================
#
# Prerequisites:
#   - Code pushed to GitHub: https://github.com/gabrycina/causalflow
#   - AWS credentials configured (for saving outputs to S3)
#
# =============================================================================

set -e

# Add nebius to PATH
export PATH="$HOME/.nebius/bin:$PATH"

# ---- Configuration ----
PROJECT_ID="project-e00h8qe9pp3twpwbzf"
SUBNET_ID="vpcsubnet-e00w88z4k3eq6s6wk5"
BUCKET_NAME="causalflow-experiments"
BUCKET_ID="storagebucket-e009709783125504239130"
REGION="eu-north1"
IMAGE="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime"
PLATFORM="gpu-h100-sxm"
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
CONTAINER_DATA_DIR="/workspace/data"
CONTAINER_OUTPUT_DIR="/workspace/output"

JOB_NAME="causalflow-$(date +%Y%m%d-%H%M%S)"

echo "========================================="
echo "CausalFlow Training Job: $JOB_NAME"
echo "========================================="
echo "Platform: $PLATFORM / $PRESET"
echo "Max time: $MAX_TIME"
echo ""
echo "Training config:"
echo "  Epochs: $EPOCHS, Batch: $BATCH_SIZE, LR: $LR"
echo "  D_model: $D_MODEL, MP layers: $NUM_MP_LAYERS"
echo "========================================="

# Launch job using a simple bash -c with the script inline
# Use @ instead of $ for variable substitution in the train script to avoid outer shell expansion
JOB_RESULT=$(nebius ai job create \
  --name "$JOB_NAME" \
  --image "$IMAGE" \
  --platform "$PLATFORM" \
  --preset "$PRESET" \
  --disk-size "$DISK_SIZE" \
  --volume "$BUCKET_ID:$CONTAINER_DATA_DIR" \
  --volume "$BUCKET_ID:$CONTAINER_OUTPUT_DIR" \
  --env "WANDB_API_KEY=$WANDB_API_KEY" \
  --container-command bash \
  --args "-c" \
  --args "set -e && \
apt-get update && apt-get install -y git && \
git clone https://github.com/gabrycina/causalflow.git /workspace/causalflow && \
cd /workspace/causalflow && \
pip install --quiet scipy scikit-learn anndata scanpy scvi-tools wandb tqdm pyyaml pandas && \
pip install --quiet decoupler && \
pip install --quiet 'numpy<2' && \
mkdir -p /workspace/data && \
curl -L -o /workspace/data/NormanWeissman2019_filtered.h5ad https://zenodo.org/records/10044268/files/NormanWeissman2019_filtered.h5ad && \
wandb login \$WANDB_API_KEY && \
python train.py \
  --data-dir /workspace/data \
  --output-dir /workspace/output \
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
  --save-interval 5 && \
aws s3 cp /workspace/output s3://causalflow-experiments/output/$JOB_NAME/ \
  --profile nebius --endpoint-url https://storage.$REGION.nebius.cloud --recursive" \
  --working-dir "/workspace/causalflow" \
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
