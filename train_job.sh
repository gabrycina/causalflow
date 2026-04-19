#!/bin/bash
set -e

# We assume the repo has already been cloned by the parent script
# Just cd into it and run
cd /workspace/causalflow

echo 'Installing dependencies...'
pip install --quiet "numpy<2" --force-reinstall
pip install --quiet scipy scikit-learn anndata scanpy scvi-tools wandb tqdm pyyaml pandas scikit-misc
pip install --quiet decoupler

echo 'Setting up data...'
mkdir -p /workspace/data
if [ ! -f /workspace/data/NormanWeissman2019_filtered.h5ad ]; then
  echo 'Downloading Norman 2019 dataset from Zenodo...'
  curl -L -o /workspace/data/NormanWeissman2019_filtered.h5ad https://zenodo.org/records/10044268/files/NormanWeissman2019_filtered.h5ad
else
  echo 'Data file already exists, skipping download'
fi

echo 'Logging into wandb...'
wandb login $WANDB_API_KEY

echo 'Starting training...'
python train.py \
  --data-dir /workspace/data \
  --output-dir /workspace/output \
  --max-genes ${MAX_GENES:-2000} \
  --epochs ${EPOCHS:-30} \
  --batch-size ${BATCH_SIZE:-128} \
  --lr ${LR:-1e-4} \
  --d-model ${D_MODEL:-256} \
  --num-layers ${NUM_MP_LAYERS:-4} \
  --grn-strategy ${GRN_STRATEGY:-message_passing} \
  --grn-reg \
  --wandb \
  --wandb-project causalflow \
  --run-name ${JOB_NAME:-causalflow-$(date +%Y%m%d-%H%M%S)} \
  --save-interval 5

echo 'Copying outputs to S3...'
aws s3 cp /workspace/output s3://causalflow-experiments/output/${JOB_NAME}/ \
  --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud --recursive
echo 'Done!'
