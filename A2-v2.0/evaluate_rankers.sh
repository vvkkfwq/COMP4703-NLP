#!/usr/bin/env bash

# Activate conda environment
source /conda/etc/profile.d/conda.sh
conda activate NLPA2
export PATH=/conda/envs/NLPA2/bin:$PATH

PYTHON="/conda/envs/NLPA2/bin/python"
WORKING_DIR="$HOME/A2-v2.0"
cd "$WORKING_DIR"

# Read STAGING status
IS_STAGING=$(grep "is_STAGING" config.py | grep -o "True\|False")

if [ "$IS_STAGING" = "True" ]; then
    OUTPUT_DIR="${WORKING_DIR}/output/staging"
    LOG_DIR="${WORKING_DIR}/logs/staging"
    echo "Evaluating in STAGING mode"
else
    OUTPUT_DIR="${WORKING_DIR}/output/production"
    LOG_DIR="${WORKING_DIR}/logs/production"
    echo "Evaluating in PRODUCTION mode"
fi

mkdir -p "${LOG_DIR}"
SUMMARY_FILE="${LOG_DIR}/evaluation_summary.txt"

echo "======================================"
echo "Evaluating All Rankers and Rerankers"
echo "======================================"

RANKERS=("rankerA" "rankerB" "rankerC" "rankerD" "rerankerA" "rerankerB")

echo "======================================" > "${SUMMARY_FILE}"
echo "Ranker Evaluation Summary" >> "${SUMMARY_FILE}"
echo "Date: $(date)" >> "${SUMMARY_FILE}"
echo "======================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

for i in "${!RANKERS[@]}"; do
    ranker="${RANKERS[$i]}"
    input_file="${OUTPUT_DIR}/${ranker}.json"
    eval_log="${LOG_DIR}/${ranker}.eval.log"
    
    idx=$((i+1))
    echo "[${idx}/6] Evaluating ${ranker}..."
    
    if [ -f "$input_file" ]; then
        # Execute in the activated environment
        ${PYTHON} MyRetEval.py --input "$input_file" 2>&1 | tee "$eval_log"
        
        echo "=== ${ranker} ===" >> "${SUMMARY_FILE}"
        grep -E "Hits@|MAP@|MRR@|NDCG@" "$eval_log" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    else
        echo "Warning: ${input_file} not found"
        echo "=== ${ranker} ===" >> "${SUMMARY_FILE}"
        echo "ERROR: Output file not found" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    fi
done

echo "======================================"
echo "All evaluations completed!"
echo "Summary: ${SUMMARY_FILE}"
echo "======================================"
cat "${SUMMARY_FILE}"