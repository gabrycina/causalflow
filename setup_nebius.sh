#!/bin/bash
# =============================================================================
# CausalFlow: Nebius AI Cloud Setup and Upload Script
# =============================================================================
# This script:
#   1. Configures AWS CLI for Nebius S3 (Object Storage)
#   2. Packages and uploads the CausalFlow code to S3
#   3. Verifies the upload
#
# Requirements:
#   - Nebius CLI installed and configured (already done)
#   - Project ID, subnet ID known (already in run_nebius.sh)
#
# =============================================================================

set -e

PROJECT_ID="project-e00h8qe9pp3twpwbzf"
REGION="eu-north1"
BUCKET_NAME="causalflow-experiments"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "CausalFlow: Nebius S3 Setup"
echo "============================================"

# Step 1: Create service account for S3 access
echo ""
echo "[1/4] Creating service account for S3 access..."

# Check if service account already exists
SA_NAME="object-storage-sa"
SA_ID=$(nebius iam service-account list --parent-id $PROJECT_ID --format json 2>/dev/null | \
  jq -r ".items[] | select(.metadata.name == \"$SA_NAME\") | .metadata.id" 2>/dev/null || echo "")

if [ -z "$SA_ID" ]; then
  SA_ID=$(nebius iam service-account create \
    --name "$SA_NAME" \
    --parent-id "$PROJECT_ID" \
    --format json 2>/dev/null | jq -r '.resource.id')
  echo "  Created service account: $SA_ID"
else
  echo "  Service account already exists: $SA_ID"
fi

# Step 2: Grant storage permissions to service account
echo ""
echo "[2/4] Granting storage permissions..."

# Get editors group ID
EDITORS_GROUP_ID=$(nebius iam group list --parent-id $PROJECT_ID --format json 2>/dev/null | \
  jq -r '.items[] | select(.metadata.name | contains("editors")) | .metadata.id' 2>/dev/null | head -1)

if [ -n "$EDITORS_GROUP_ID" ]; then
  nebius iam group-membership create \
    --parent-id "$EDITORS_GROUP_ID" \
    --member-id "$SA_ID" 2>/dev/null && echo "  Added to editors group" || echo "  Already in group or cannot add"
fi

# Step 3: Create access key
echo ""
echo "[3/4] Creating S3 access key..."

ACCESS_KEY_ID=$(nebius iam access-key create \
  --account-service-account-id "$SA_ID" \
  --description 'AWS CLI for CausalFlow' \
  --format json 2>/dev/null | jq -r '.resource.id')

if [ -z "$ACCESS_KEY_ID" ] || [ "$ACCESS_KEY_ID" = "null" ]; then
  echo "  ERROR: Failed to create access key"
  echo "  Check if you have permissions to create access keys"
  exit 1
fi

# Get the actual AWS key ID and secret
AWS_KEY_ID=$(nebius iam access-key get-by-id --id "$ACCESS_KEY_ID" --format json 2>/dev/null | jq -r '.status.aws_access_key_id')
AWS_SECRET=$(nebius iam access-key get-secret-once --id "$ACCESS_KEY_ID" --format json 2>/dev/null | jq -r '.secret')

if [ -z "$AWS_KEY_ID" ] || [ -z "$AWS_SECRET" ]; then
  echo "  ERROR: Failed to retrieve access key credentials"
  exit 1
fi

echo "  Access key created successfully"

# Step 4: Configure AWS CLI
echo ""
echo "[4/4] Configuring AWS CLI for Nebius S3..."

mkdir -p ~/.aws

aws configure set aws_access_key_id "$AWS_KEY_ID" --profile nebius
aws configure set aws_secret_access_key "$AWS_SECRET" --profile nebius
aws configure set region "$REGION" --profile nebius
aws configure set endpoint_url "https://storage.$REGION.nebius.cloud" --profile nebius

echo "  AWS CLI configured with Nebius S3 endpoint"

# Verify connection
echo ""
echo "Verifying S3 access..."
aws s3 ls --profile nebius --endpoint-url "https://storage.$REGION.nebius.cloud" 2>/dev/null && echo "  S3 access verified!" || echo "  Warning: S3 access check failed"

# Package and upload code
echo ""
echo "Packaging CausalFlow code..."
cd "$SCRIPT_DIR"
tar -czf /tmp/causalflow.tar.gz \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude=".venv" \
  --exclude=".git" \
  .

echo "Uploading to S3..."
aws s3 cp /tmp/causalflow.tar.gz \
  "s3://$BUCKET_NAME/code/causalflow.tar.gz" \
  --profile nebius \
  --endpoint-url "https://storage.$REGION.nebius.cloud"

echo ""
echo "Verifying upload..."
aws s3 ls \
  "s3://$BUCKET_NAME/code/" \
  --profile nebius \
  --endpoint-url "https://storage.$REGION.nebius.cloud"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Review and adjust config in run_nebius.sh if needed"
echo "  2. Launch training: bash run_nebius.sh"
echo "  3. Monitor job: nebius ai job list"
echo ""
echo "To check job logs:"
echo "  nebius ai job list"
echo "  nebius ai job logs <job-id>"
