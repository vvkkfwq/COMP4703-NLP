#!/usr/bin/env bash

# Activate conda environment
source /conda/etc/profile.d/conda.sh
conda activate NLPA2
export PATH=/conda/envs/NLPA2/bin:$PATH

PYTHON="/conda/envs/NLPA2/bin/python"
WORKING_DIR="$HOME/A2-v2.0"

# Read STAGING status from config.py
IS_STAGING=$(grep "is_STAGING" config.py | grep -o "True\|False")

if [ "$IS_STAGING" = "True" ]; then
    LOG_DIR="${WORKING_DIR}/logs/staging"
    echo "======================================"
    echo "Running in STAGING mode"
else
    LOG_DIR="${WORKING_DIR}/logs/production"
    echo "======================================"
    echo "Running in PRODUCTION mode"
fi

mkdir -p ${LOG_DIR}

echo "Running All RAGs"
echo "======================================"

# Run rankers
echo "[1/4] Running RAGA (meta-llama/Llama-2-7b-chat-hf)..."
${PYTHON} RAGA.py > ${LOG_DIR}/RAGA.log 2>&1

echo "[2/4] Running RAGB (meta-llama/Meta-Llama-3-8B-Instruct)..."
${PYTHON} RAGB.py > ${LOG_DIR}/RAGB.log 2>&1

echo "[3/4] Running RAGC (mistralai/Mistral-7B-Instruct-v0.3)..."
${PYTHON} RAGC.py > ${LOG_DIR}/RAGC.log 2>&1

echo "[4/4] Running RAGD (Qwen/Qwen3-8B)..."
${PYTHON} RAGD.py > ${LOG_DIR}/RAGD.log 2>&1

echo "======================================"
echo "All tasks completed!"
echo "Logs saved to: ${LOG_DIR}"
echo "======================================"
