#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

OUTPUT_DIR="sweep_results_merged"
mkdir -p $OUTPUT_DIR

MODEL_ID="FacebookAI/xlm-roberta-base"
EPOCHS=5
NUM_CONFIGS=6

get_gpu_id() {
    local config_index=$1
    echo $((config_index % 4))
}

echo " Starting merged dataset experiments..."

for i in $(seq 0 $((NUM_CONFIGS-1))); do
    RUN_NAME="merged_config_${i}"
    
    RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
    mkdir -p $RUN_DIR
    
    GPU_ID=$(get_gpu_id $i)
    
    echo " Starting config $i on GPU $GPU_ID"
    
    nohup python3 "${SCRIPT_DIR}/text_classification_merged.py" \
        --model_id $MODEL_ID \
        --num_epochs $EPOCHS \
        --use_single_gpu \
        --gpu_id $GPU_ID \
        --output_dir $RUN_DIR \
        --config_index $i \
        --train_file "${WORKSPACE_DIR}/merged_datasets/merged_train.csv" \
        --valid_file "${WORKSPACE_DIR}/merged_datasets/merged_valid.csv" \
        --test_file "${WORKSPACE_DIR}/merged_datasets/merged_test.csv" \
        > "${RUN_DIR}/nohup.out" 2>&1 &
    
    sleep 30
done

echo " All $NUM_CONFIGS experiments started!"
echo " Results in: $OUTPUT_DIR/" 