#!/usr/bin/env bash

# # Activate conda environment
# source /conda/etc/profile.d/conda.sh
# conda activate NLPA2
# export PATH=/conda/envs/NLPA2/bin:$PATH

# PYTHON="/conda/envs/NLPA2/bin/python"
# WORKING_DIR="$HOME/A2-v2.0"

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate COMP4703A2
export PATH=/root/miniconda3/envs/COMP4704A2/bin:$PATH

PYTHON="/root/miniconda3/envs/COMP4703A2/bin/python"
WORKING_DIR="/workspace/COMP4703-NLP/A2-v2.0"

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

echo "Running All Rankers and Rerankers"
echo "======================================"

# Run rankers
echo "[1/6] Running rankerA..."
${PYTHON} rankerA.py > ${LOG_DIR}/rankerA.log 2>&1

echo "[2/6] Running rankerB..."
${PYTHON} rankerB.py > ${LOG_DIR}/rankerB.log 2>&1

echo "[3/6] Running rankerC..."
${PYTHON} rankerC.py > ${LOG_DIR}/rankerC.log 2>&1

echo "[4/6] Running rankerD..."
${PYTHON} rankerD.py > ${LOG_DIR}/rankerD.log 2>&1

# Run rerankers
echo "[5/6] Running rerankerA..."
${PYTHON} rerankerA.py > ${LOG_DIR}/rerankerA.log 2>&1

echo "[6/6] Running rerankerB..."
${PYTHON} rerankerB.py > ${LOG_DIR}/rerankerB.log 2>&1

echo "======================================"
echo "All tasks completed!"
echo "Logs saved to: ${LOG_DIR}"
echo "======================================"
