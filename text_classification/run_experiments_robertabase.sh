#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

OUTPUT_DIR="sweep_results_all"
mkdir -p $OUTPUT_DIR

MODEL_ID="FacebookAI/xlm-roberta-base"
EPOCHS=5
NUM_CONFIGS=6

get_gpu_id() {
    local config_index=$1
    echo $((config_index % 4))
}

for i in $(seq 0 $((NUM_CONFIGS-1))); do
    RUN_NAME="config_${i}"
    
    RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
    mkdir -p $RUN_DIR
    
    GPU_ID=$(get_gpu_id $i)
    
    nohup python3 "${SCRIPT_DIR}/text_classification_all.py" \
        --model_id $MODEL_ID \
        --num_epochs $EPOCHS \
        --use_single_gpu \
        --gpu_id $GPU_ID \
        --output_dir $RUN_DIR \
        --config_index $i \
        --train_file "${WORKSPACE_DIR}/mmsd_image_description_with_QwenVL/mmsd_image_descriptions_train.csv" \
        --valid_file "${WORKSPACE_DIR}/mmsd_image_description_with_QwenVL/mmsd_image_descriptions_valid.csv" \
        --test_file "${WORKSPACE_DIR}/mmsd_image_description_with_QwenVL/mmsd_image_descriptions_test.csv" \
        > "${RUN_DIR}/nohup.out" 2>&1 &
    
    sleep 30
done

echo "All experiments started. Check nohup.out files in each run directory for progress." 