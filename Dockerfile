FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install core ML dependencies
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision \
    numpy \
    scipy \
    pandas \
    scikit-learn

# Install transformers and related
RUN pip install --no-cache-dir \
    transformers>=4.30.0 \
    huggingface-hub

# Install single-cell packages
RUN pip install --no-cache-dir \
    scanpy>=1.9.0 \
    scvi-tools>=0.20.0 \
    anndata>=0.9.0 \
    pertpy>=0.5.0

# Install decoupler for DoRothEA GRN
RUN pip install --no-cache-dir \
    decoupler>=1.4.0

# Install training utilities
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    tqdm \
    pyyaml

# Install scGPT from GitHub
RUN pip install --no-cache-dir \
    scgpt @ git+https://github.com/bowang-lab/scGPT.git

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV WANDB_API_KEY=${WANDB_API_KEY}

ENTRYPOINT ["python", "train.py"]
