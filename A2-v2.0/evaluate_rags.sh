#!/usr/bin/env bash

# Use the conda run command instead of activating
CONDA_ENV="COMP4703A2"
WORKING_DIR="/Users/vickeyfeng/Code/COMP4703-NLP/A2-v2.0"

cd "${WORKING_DIR}" || {
    echo "Failed to access working directory: ${WORKING_DIR}" >&2
    exit 1
}

# Read STAGING status from config.py
IS_STAGING=$(grep "is_STAGING" config.py | grep -o "True\|False")

if [ "${IS_STAGING}" = "True" ]; then
    OUTPUT_DIR="${WORKING_DIR}/output/staging"
    LOG_DIR="${WORKING_DIR}/logs/staging"
    SUMMARY_FILE="${LOG_DIR}/rag_evaluation_summary.txt"
    echo "======================================"
    echo "Evaluating in STAGING mode"
else
    OUTPUT_DIR="${WORKING_DIR}/output/production"
    LOG_DIR="${WORKING_DIR}/logs/production"
    SUMMARY_FILE="${LOG_DIR}/rag_evaluation_summary.txt"
    echo "======================================"
    echo "Evaluating in PRODUCTION mode"
fi

mkdir -p "${LOG_DIR}"

echo "Evaluating All RAG Pipelines"
echo "======================================"

RAG_RUNS=(
    "llm-embedder-reranker_llama2|LLM Embedder Reranker + Llama-2"
    "llm-embedder-reranker_llama3|LLM Embedder Reranker + Llama-3"
    "llm-embedder-reranker_Mistral-7B|LLM Embedder Reranker + Mistral-7B"
    "llm-embedder-reranker_Qwen3-8B|LLM Embedder Reranker + Qwen3-8B"
    "bge-large_llama2|BGE Large + Llama-2"
    "bge-large_llama3|BGE Large + Llama-3"
    "bge-large_Mistral-7B|BGE Large + Mistral-7B"
    "bge-large_Qwen3-8B|BGE Large + Qwen3-8B"
)

total=${#RAG_RUNS[@]}

# Initialize summary file
echo "====================================== " > "${SUMMARY_FILE}"
echo "RAG Evaluation Summary" >> "${SUMMARY_FILE}"
echo "Date: $(date)" >> "${SUMMARY_FILE}"
echo "====================================== " >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

for idx in "${!RAG_RUNS[@]}"; do
    IFS='|' read -r rag_key rag_label <<< "${RAG_RUNS[$idx]}"
    input_file="${OUTPUT_DIR}/${rag_key}.json"
    eval_log="${LOG_DIR}/${rag_key}.rag.eval.log"

    seq=$((idx + 1))
    echo "[${seq}/${total}] Evaluating ${rag_label}..."

    if [ -f "${input_file}" ]; then
        # Use conda run to execute in the environment
        conda run -n ${CONDA_ENV} python MyRagEval.py "${input_file}" 2>&1 | tee "${eval_log}"

        echo "=== ${rag_label} ===" >> "${SUMMARY_FILE}"
        if grep -q "Overall Metrics:" "${eval_log}"; then
            awk '/Overall Metrics:/ {print; getline; print; getline; print; getline; print; getline; print}' "${eval_log}" >> "${SUMMARY_FILE}"
        else
            echo "Overall metrics not found in log." >> "${SUMMARY_FILE}"
        fi
        echo "" >> "${SUMMARY_FILE}"
    else
        echo "Warning: ${input_file} not found, skipping..."
        echo "=== ${rag_label} ===" >> "${SUMMARY_FILE}"
        echo "ERROR: Output file not found" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    fi

done

echo "======================================"
echo "All evaluations completed!"
echo "Individual logs saved to: ${LOG_DIR}/*.rag.eval.log"
echo "Summary saved to: ${SUMMARY_FILE}"
echo "======================================"

# Display summary
cat "${SUMMARY_FILE}"