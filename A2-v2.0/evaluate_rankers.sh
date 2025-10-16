#!/usr/bin/env bash

# Activate conda environment
source /conda/etc/profile.d/conda.sh
conda activate COMP4703A2
export PATH=/conda/envs/COMP4703A2/bin:$PATH

# Use which to find python from activated environment
PYTHON=$(which python)
WORKING_DIR="$HOME/code/COMP4703-NLP/A2-v2.0"

# Read STAGING status from config.py
IS_STAGING=$(grep "is_STAGING" config.py | grep -o "True\|False")

if [ "$IS_STAGING" = "True" ]; then
    OUTPUT_DIR="${WORKING_DIR}/output/staging"
    LOG_DIR="${WORKING_DIR}/logs/staging"
    SUMMARY_FILE="${LOG_DIR}/evaluation_summary.txt"
    echo "======================================"
    echo "Evaluating in STAGING mode"
else
    OUTPUT_DIR="${WORKING_DIR}/output/production"
    LOG_DIR="${WORKING_DIR}/logs/production"
    SUMMARY_FILE="${LOG_DIR}/evaluation_summary.txt"
    echo "======================================"
    echo "Evaluating in PRODUCTION mode"
fi

mkdir -p ${LOG_DIR}

echo "Evaluating All Rankers and Rerankers"
echo "======================================"

# Define ranker list
RANKERS=("llm-embedder-ranker" "all-MiniLM-L6-v2-ranker" "bge-large-en-v1.5-ranker" "multilingual-e5-large-ranker" "llm-embedder-reranker" "all-MiniLM-L6-reranker")

# Initialize summary file
echo "====================================== " > ${SUMMARY_FILE}
echo "Ranker Evaluation Summary" >> ${SUMMARY_FILE}
echo "Date: $(date)" >> ${SUMMARY_FILE}
echo "====================================== " >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

# Evaluate each ranker
for i in "${!RANKERS[@]}"; do
    ranker="${RANKERS[$i]}"
    input_file="${OUTPUT_DIR}/${ranker}.json"
    eval_log="${LOG_DIR}/${ranker}.eval.log"

    idx=$((i+1))
    echo "[${idx}/6] Evaluating ${ranker}..."

    if [ -f "$input_file" ]; then
        # Run evaluation and save to .eval.log
        ${PYTHON} MyRetEval.py --input "$input_file" 2>&1 | tee "$eval_log"

        # Extract metrics and append to summary
        echo "=== ${ranker} ===" >> ${SUMMARY_FILE}
        grep -E "Hits@|MAP@|MRR@" "$eval_log" >> ${SUMMARY_FILE}
        echo "" >> ${SUMMARY_FILE}
    else
        echo "Warning: ${input_file} not found, skipping..."
        echo "=== ${ranker} ===" >> ${SUMMARY_FILE}
        echo "ERROR: Output file not found" >> ${SUMMARY_FILE}
        echo "" >> ${SUMMARY_FILE}
    fi
done

echo "======================================"
echo "All evaluations completed!"
echo "Individual logs saved to: ${LOG_DIR}/*.eval.log"
echo "Summary saved to: ${SUMMARY_FILE}"
echo "======================================"

# Display summary
cat ${SUMMARY_FILE}
