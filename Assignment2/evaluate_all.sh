#!/usr/bin/env bash
# filepath: /Users/vickeyfeng/Code/COMP4703-NLP/Assignment2/evaluate_all.sh

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate COMP4703A2
export PATH=/opt/miniconda3/envs/COMP4703A2/bin:$PATH

PYTHON="/opt/miniconda3/envs/COMP4703A2/bin/python"
WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${WORKING_DIR}/logs"
OUTPUT_DIR="${WORKING_DIR}/output"

# Create logs directory if not exists
mkdir -p "${LOGDIR}"

# Define ranker configurations
declare -A RANKERS=(
  ["rankerA"]="BAAI/bge-base-en-v1.5"
  ["rankerB"]="sentence-transformers/all-MiniLM-L6-v2"
  ["rankerC"]="intfloat/e5-base-v2"
  ["rankerD"]="BAAI/bge-small-en-v1.5"
)