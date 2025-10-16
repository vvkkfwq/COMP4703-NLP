#!/usr/bin/env bash

# Activate conda environment
source /conda/etc/profile.d/conda.sh
conda activate NLPA2
export PATH=/conda/envs/NLPA2/bin:$PATH

PYTHON="/conda/envs/NLPA2/bin/python"
WORKING_DIR="$HOME/COMP4703_NLP/A2-v2.0"
LOGFILE="${WORKING_DIR}/run.log"

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
${PYTHON} rankerA.py 2>&1 | tee ${LOG_DIR}/rankerA.log

echo "[2/6] Running rankerB..."
${PYTHON} rankerB.py 2>&1 | tee ${LOG_DIR}/rankerB.log

echo "[3/6] Running rankerC..."
${PYTHON} rankerC.py 2>&1 | tee ${LOG_DIR}/rankerC.log

echo "[4/6] Running rankerD..."
${PYTHON} rankerD.py 2>&1 | tee ${LOG_DIR}/rankerD.log

# Run rerankers
echo "[5/6] Running rerankerA..."
${PYTHON} rerankerA.py 2>&1 | tee ${LOG_DIR}/rerankerA.log

echo "[6/6] Running rerankerB..."
${PYTHON} rerankerB.py 2>&1 | tee ${LOG_DIR}/rerankerB.log

echo "======================================"
echo "All tasks completed!"
echo "Logs saved to: ${LOG_DIR}"
echo "======================================"
