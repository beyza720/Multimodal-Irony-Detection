#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

OUTPUT_DIR="sweep_results_merged_large"
mkdir -p $OUTPUT_DIR

MODEL_ID="FacebookAI/xlm-roberta-large"
EPOCHS=3
NUM_CONFIGS=6

get_gpu_id() {
    local config_index=$1
    echo $((config_index % 4))
}

echo "🚀 Starting merged dataset experiments with XLM-RoBERTa Large..."
echo "📊 Dataset info:"
echo "   Train: $(wc -l < "${WORKSPACE_DIR}/merged_datasets/merged_train.csv") samples"
echo "   Valid: $(wc -l < "${WORKSPACE_DIR}/merged_datasets/merged_valid.csv") samples"
echo "   Test:  $(wc -l < "${WORKSPACE_DIR}/merged_datasets/merged_test.csv") samples"
echo ""
echo "🔧 Large Model Optimizations:"
echo "   • Reduced epochs: $EPOCHS (vs 5 for base model)"
echo "   • Smaller batch sizes: 2-4 (vs 4-16 for base model)"
echo "   • Memory-efficient settings enabled"
echo "   • Text truncation: 400 characters"
echo ""

for i in $(seq 0 $((NUM_CONFIGS-1))); do
    RUN_NAME="merged_large_config_${i}"
    
    RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
    mkdir -p $RUN_DIR
    
    GPU_ID=$(get_gpu_id $i)
    
    echo "▶️  Starting Large model config $i on GPU $GPU_ID"
    
    # Show config details
    case $i in
        0) echo "     Config: batch_size=2, lr=5e-5" ;;
        1) echo "     Config: batch_size=2, lr=2e-5" ;;
        2) echo "     Config: batch_size=2, lr=1e-4" ;;
        3) echo "     Config: batch_size=4, lr=5e-5" ;;
        4) echo "     Config: batch_size=4, lr=2e-5" ;;
        5) echo "     Config: batch_size=4, lr=1e-4" ;;
    esac
    
    nohup python3 "${SCRIPT_DIR}/text_classification_merged_large.py" \
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
    
    echo "   ✓ Process started, logs: ${RUN_DIR}/nohup.out"
    sleep 45  # Longer wait for large model initialization
done

echo ""
echo "🎉 All $NUM_CONFIGS Large model experiments started!"
echo "📁 Results will be saved in: $OUTPUT_DIR/"
echo "📊 Monitor progress with:"
echo "   tail -f $OUTPUT_DIR/*/nohup.out"
echo ""
echo "⏱️  Estimated completion time: ~4-6 hours (Large model takes longer)" 
echo "💾 Memory usage will be higher - monitor with: nvidia-smi"
echo ""
echo "🔍 Key differences from base model:"
echo "   • Using XLM-RoBERTa Large (355M parameters vs 270M)"
echo "   • Optimized for memory efficiency"
echo "   • Better performance expected with longer training time" 