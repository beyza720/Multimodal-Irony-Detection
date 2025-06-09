# Multimodal Sarcasm Detection Project

## Overview

This project focuses on multimodal sarcasm detection using the MMSD2.0 dataset. The goal is to develop and compare various machine learning models that can identify sarcasm in social media content by analyzing both textual and visual information simultaneously.

## Project Description

Sarcasm detection is a challenging natural language processing task that becomes even more complex in multimodal settings where text is accompanied by images. This project explores different approaches to tackle this problem, including:

- **Text-only classification models** using transformer-based architectures
- **Image description generation** using vision-language models
- **Multimodal approaches** combining textual and visual features
- **Zero-shot classification** experiments
- **Fine-tuning experiments** with various pre-trained models

## Dataset

This project uses multiple datasets for multimodal sarcasm detection and analysis:

### Primary Dataset: MMSD2.0
The **MMSD2.0 (Multimodal Sarcasm Detection Dataset 2.0)** is a comprehensive dataset containing social media posts with both text and images, labeled for sarcasm detection.

**Dataset Source**: [https://github.com/JoeYing1019/MMSD2.0](https://github.com/JoeYing1019/MMSD2.0)

### Additional Dataset: MuSE
The **MuSE (Multimodal Sarcasm Explanation)** dataset, introduced in the AAAI-22 paper "Nice perfume. How long did you marinate in it? Multimodal Sarcasm Explanation", contains 3,510 sarcastic multimodal posts with natural language explanations. For our experiments, we utilize only the text and image components from this dataset, applying preprocessing steps similar to MMSD2.0 (emoji removal and hashtag cleaning) to ensure consistency.

**Dataset Source**: [https://github.com/LCS2-IIITD/Multimodal-Sarcasm-Explanation-MuSE](https://github.com/LCS2-IIITD/Multimodal-Sarcasm-Explanation-MuSE)

**Note**: Due to copyright and ethical considerations, the original images and texts from the dataset are not included in this repository. Please refer to the official MMSD2.0 repository linked above to access the complete dataset.

### Dataset Setup

After downloading the MMSD2.0 dataset, organize it as follows:

```
mmsd_dataset/
├── extracted_part_1/    # Images directory 1
├── extracted_part_2/    # Images directory 2
├── extracted_part_3/    # Images directory 3
├── extracted_part_4/    # Images directory 4
├── extracted_part_5/    # Images directory 5
├── extracted_part_6/    # Images directory 6
├── train.json          # Training split annotations
├── valid.json          # Validation split annotations
└── test.json           # Test split annotations
```

**Important Notes:**
- Images are distributed across 6 separate directories (`extracted_part_1` through `extracted_part_6`)
- Each JSON file contains image IDs, text content, and sarcasm labels (0: non-sarcastic, 1: sarcastic)
- Update the `dataset_path` variable in scripts to point to your `mmsd_dataset` directory location

## Key Components

- **Image Description Generation**: Scripts for generating textual descriptions of images using multiple vision-language models
  - **Qwen2.5-VL**: High-quality descriptions with quantization support
  - **InternVL3**: Alternative model for comparison and ensemble approaches
- **Text Combination**: Utilities for combining original text with generated image descriptions
- **Text Classification**: Implementation of various transformer models for sarcasm detection
- **Multimodal Analysis**: Experiments combining visual and textual features
- **Zero-shot Learning**: Testing models' performance without specific training on the target task

## Technology Stack

- **Python 3.10+**
- **PyTorch 2.2.2** for deep learning with CUDA support
- **Transformers 4.51.3+** library for pre-trained models
- **LMDeploy 0.7.3** for InternVL model deployment
- **Weights & Biases** for experiment tracking
- **Multiple vision-language models**:
  - Qwen2.5-VL-7B-Instruct
  - InternVL3-8B
- **Quantization support** via bitsandbytes and triton

## Environment Setup

### Option 1: Qwen2.5-VL Environment 
```bash
python3 -m venv qwen_env
source qwen_env/bin/activate  # On Windows: qwen_env\Scripts\activate
pip install -r requirements_qwenvl.txt
```

### Option 2: InternVL Environment
```bash
python3 -m venv internvl_env
source internvl_env/bin/activate  # On Windows: internvl_env\Scripts\activate
pip install -r requirements_internvl.txt
```

### Option 3: Conda Environment
```bash
conda create -n multimodal-sarcasm python=3.10
conda activate multimodal-sarcasm
# Choose one: pip install -r requirements.txt OR pip install -r requirements_internvl.txt
```

## Getting Started

1. Clone this repository
2. Set up your environment for your chosen VL model (see Environment Setup above)
3. Download the MMSD2.0 dataset from the official repository
4. Organize the dataset according to the structure described in Dataset Setup
5. Update dataset paths in the scripts to match your local setup
6. Run the desired experiments following the pipeline below

## Usage Pipeline

### Step 1: Image Description Generation

**Option A: Using Qwen2.5-VL**
```bash
# For training data
python image_description_scripts/qwen_vl_image_description.py

# Note: Script is configured for train.json by default
# For validation/test data, modify the script to use valid.json or test.json
```

**Option B: Using InternVL3**
```bash
# For training and validation data
python image_description_scripts/intern_vl_image_description.py

# Note: This script processes both train and val splits automatically
```

### Step 2: Text Combination (Multimodal Input Creation)
```bash
# Combine original text with generated image descriptions
python image_description_scripts/combine_texts.py \
    --input mmsd_image_descriptions_train.csv \
    --output mmsd_combined_train.csv
```

### Step 3: Text Classification Training

**For QwenVL + RoBERTa experiments:**
```bash
# Run all configurations with QwenVL-generated descriptions
cd text_classification
./run_experiments_robertabase.sh
```

**For InternVL + RoBERTa experiments:**
Before running the script, you need to modify the data paths in `run_experiments_robertabase.sh`:

```bash
# Change these lines in run_experiments_robertabase.sh:
--train_file "${WORKSPACE_DIR}/mmsd_image_description_with_QwenVL/mmsd_image_descriptions_train.csv" \
--valid_file "${WORKSPACE_DIR}/mmsd_image_description_with_QwenVL/mmsd_image_descriptions_valid.csv" \
--test_file "${WORKSPACE_DIR}/mmsd_image_description_with_QwenVL/mmsd_image_descriptions_test.csv" \

# To:
--train_file "${WORKSPACE_DIR}/mmsd_image_description_with_InternVL/mmsd_image_descriptions_train.csv" \
--valid_file "${WORKSPACE_DIR}/mmsd_image_description_with_InternVL/mmsd_image_descriptions_valid.csv" \
--test_file "${WORKSPACE_DIR}/mmsd_image_description_with_InternVL/mmsd_image_descriptions_test.csv" \
```

Then run:
```bash
cd text_classification
./run_experiments_robertabase.sh
```


## Script Adaptability

### Image Description Generation
The repository includes two different VL models for image description generation:

**Qwen2.5-VL (`qwen_vl_image_description.py`)**:
- Uses Qwen2.5-VL-7B-Instruct model
- Supports 4-bit quantization for memory efficiency
- Designed for MMSD2.0 dataset with train/valid/test split flexibility
- Generates detailed descriptions up to 250 tokens
- Requires larger GPU memory but provides high-quality descriptions

**InternVL3 (`intern_vl_image_description.py`)**:
- Uses InternVL3-8B model with LMDeploy backend
- Optimized for inference speed and efficiency
- Processes train and validation splits automatically
- Designed for MUSE dataset format but adaptable
- More memory-efficient deployment

Both scripts generate CSV outputs with the same structure for consistency in downstream tasks.

### Text Combination
The `combine_texts.py` script takes any CSV file with `text` and `image_description` columns and creates a `combined_text` column for multimodal classification.

## Sample Data

This repository includes sample datasets with 50 examples from the processed training data, demonstrating the data structure after image description generation with different vision-language models:

### Sample Files
- **`mmsd_sample_data_qwenvl.csv`**: Sample data with image descriptions generated using Qwen2.5-VL model (MMSD2.0 dataset, 50 examples)
- **`mmsd_sample_data_internvl.csv`**: Sample data with image descriptions generated using InternVL3 model (MMSD2.0 dataset, 50 examples)
- **`muse_sample_data_qwenvl.csv`**: Sample data with image descriptions generated using Qwen2.5-VL model (MuSE dataset, 36 examples)

Both MMSD2.0 sample files use the same image IDs, allowing for direct comparison of how different vision-language models describe the same images. The MuSE sample demonstrates the data structure for the additional dataset used in our experiments.

### Data Structure
| Column | Description |
|--------|-------------|
| `image_id` | Unique identifier for the image (e.g., "840006160660983809.jpg") |
| `text` | Original social media post text |
| `label` | Sarcasm label (0: non-sarcastic, 1: sarcastic) |
| `image_description` | Generated description (Qwen2.5-VL or InternVL3 depending on file) |
| `image_location` | Path where the image was found during processing |
| `combined_text` | Concatenation of original text and image description |

**Note:** The sample data is provided for format reference, model comparison, and testing purposes. For full experiments, process the complete MMSD2.0 dataset using the provided scripts.

## Citations

## Results

We conducted a series of experiments using different configurations (learning rate and batch size) for various models. The results for the sarcasm detection task are summarized below:

### Table: Best performing configuration per setting (no config details)

This table summarizes the best performance achieved for each input modality and model combination, providing a concise overview of the highest metrics across all configurations.

| Input             | Model             | Acc   | F1    | Prec  | Recall |
|-------------------|-------------------|-------|-------|-------|--------|
| text-only         | Qwen + Roberta    | 78.42 | 77.71 | 78.60 | 78.42  |
| text-only         | Qwen + ModernBERT | **84.72** | **84.59** | **85.21** | **84.72** |
| text-only         | InternVL + Roberta| 78.08 | 77.86 | 78.48 | 78.08  |
| text+image        | Qwen + Roberta    | 74.51 | 73.88 | 74.42 | 74.51  |
| text+image        | InternVL + Roberta| 76.26 | 75.93 | 76.47 | 76.26  |
| combined          | Qwen + Roberta    | 82.07 | 81.88 | 82.42 | 82.07  |
| combined          | Qwen + ModernBERT | 82.61 | 82.36 | 82.67 | 82.61  |
| combined          | InternVL + Roberta| 81.82 | 81.62 | 82.14 | 81.82  |
| Baseline          | MMSD2.0           | 85.64 | 84.10 | 80.33 | 88.24  |

*Note: The best performing configuration across all settings is highlighted in bold.*

### All Experimental Results

### Table: QwenVL + roberta base (all dataset) results

| Config (lr, bs) | F1    | Accuracy | Precision | Recall |
|-----------------|-------|----------|-----------|--------|
| (5e-05, 4)      | 80.41 | 80.53    | 81.32     | 80.53  |
| (2e-05, 4)      | 81.12 | 81.40    | 81.52     | 81.40  |
| (5e-05, 8)      | **81.88** | **82.07** | **82.42** | **82.07** |
| (2e-05, 8)      | 81.36 | 81.53    | 82.01     | 81.53  |
| (5e-05, 16)     | 81.32 | 82.04    | 81.72     | 81.61  |
| (2e-05, 16)     | 80.66 | 80.86    | 81.21     | 80.86  |

*Note: The best performing configuration is highlighted in bold.*

### Table: InternVL3-8B + roberta base results

| Config (lr, bs) | F1    | Accuracy | Precision | Recall |
|-----------------|-------|----------|-----------|--------|
| (5e-05, 4)      | 79.68 | 79.83    | 80.49     | 79.83  |
| (2e-05, 4)      | 79.94 | 80.37    | 80.34     | 80.37  |
| (5e-05, 8)      | **81.62** | **81.82** | **82.14** | **81.82** |
| (2e-05, 8)      | 79.67 | 80.03    | 80.07     | 80.03  |
| (5e-05, 16)     | 80.30 | 80.68    | 80.68     | 80.70  |
| (2e-05, 16)     | 79.31 | 79.87    | 79.80     | 79.87  |

*Note: The best performing configuration is highlighted in bold.*

We also conducted experiments with different data configurations to analyze the contribution of different modalities:

### Table: QwenVL + roberta base (only test set) results

This table shows the performance of the QwenVL + Roberta Base model when trained and evaluated on the **test set only**, using combined text and image descriptions.

| Config (lr, bs) | F1    | Accuracy | Precision | Recall |
|-----------------|-------|----------|-----------|--------|
| (5e-05, 4)      | 73.40 | 75.10    | 74.85     | 75.10  |
| (2e-05, 4)      | **77.71** | **78.42**    | **78.60**     | **78.42**  |
| (5e-05, 8)      | 76.93 | 78.01    | 77.87     | 78.01  |
| (2e-05, 8)      | 73.02 | 74.69    | 74.42     | 74.69  |
| (5e-05, 16)     | 74.51 | 75.93    | 75.71     | 75.93  |
| (2e-05, 16)     | 74.12 | 75.52    | 75.29     | 75.52  |

---

### Table: QwenVL + roberta base (only text) results

This table presents the results of the QwenVL + Roberta Base model trained and evaluated using **only the textual content** from the dataset.

| Config (lr, bs) | F1    | Accuracy | Precision | Recall |
|-----------------|-------|----------|-----------|--------|
| (5e-05, 4)      | 66.37 | 66.67    | 67.21     | 66.67  |
| (2e-05, 4)      | **76.01** | **76.38**    | **76.51**     | **76.38**  |
| (5e-05, 8)      | 36.29 | 56.95    | 56.95     | 56.95  |
| (2e-05, 8)      | 74.18 | 74.97    | 74.84     | 74.97  |
| (5e-05, 16)     | 75.77 | 76.25    | 76.30     | 76.30  |
| (2e-05, 16)     | 75.93 | 76.09    | 76.81     | 76.09  |

---

### Table: QwenVL + roberta base (only image description) results

This table details the performance of the QwenVL + Roberta Base model when trained and evaluated using **only the generated image descriptions** as input.

| Config (lr, bs) | F1    | Accuracy | Precision | Recall |
|-----------------|-------|----------|-----------|--------|
| (5e-05, 4)      | 36.29 | 56.95    | 32.44     | 56.95  |
| (2e-05, 4)      | **73.88** | **74.51**    | **74.42**     | **74.51** |
| (5e-05, 8)      | 68.35 | 68.87    | 68.98     | 68.87  |
| (2e-05, 8)      | 73.65 | 74.35    | 74.23     | 74.35  |
| (5e-05, 16)     | 74.53 | 74.48    | 74.51     | 74.48  |
| (2e-05, 16)     | 74.53 | 75.18    | 75.08     | 75.18  |

---

### Table: InternVL3-8B + roberta-base results (text-only)

This table shows the performance of the InternVL3-8B + Roberta Base model when trained and evaluated using **only the textual content** from the dataset.

| Config (lr, bs) | F1    | Accuracy | Precision | Recall |
|-----------------|-------|----------|-----------|--------|
| (5e-05, 4)      | 36.29 | 56.95    | 32.44     | 56.95  |
| (2e-05, 4)      | **77.86** | **78.08** | **78.48** | **78.08** |
| (5e-05, 8)      | 76.09 | 76.26    | 76.96     | 76.26  |
| (2e-05, 8)      | 76.21 | 76.38    | 77.07     | 76.38  |
| (5e-05, 16)     | 77.20 | 77.38    | 77.97     | 77.38  |
| (2e-05, 16)     | 77.17 | 77.38    | 77.86     | 77.38  |

---

### Table: InternVL3-8B + roberta-base results (image description only)

This table details the performance of the InternVL3-8B + Roberta Base model when trained and evaluated using **only the generated image descriptions** as input.

| Config (lr, bs) | F1    | Accuracy | Precision | Recall |
|-----------------|-------|----------|-----------|--------|
| (5e-05, 4)      | 67.81 | 68.12    | 68.60     | 68.12  |
| (2e-05, 4)      | **75.93** | **76.26** | **76.47** | **76.26** |
| (5e-05, 8)      | 74.31 | 74.84    | 74.81     | 74.84  |
| (2e-05, 8)      | 75.50 | 75.84    | 76.04     | 75.84  |
| (5e-05, 16)     | 74.40 | 74.68    | 75.05     | 74.68  |
| (2e-05, 16)     | 74.83 | 75.22    | 75.35     | 75.22  |

---

### Table: QwenVL + roberta large results

This table presents the performance of the QwenVL + Roberta Large model when using combined text and image descriptions.

| Config (lr, bs) | F1    | Accuracy | Precision | Recall |
|-----------------|-------|----------|-----------|--------|
| (5e-05, 4)      | 36.29 | 56.95    | 32.44     | 56.95  |
| (2e-05, 4)      | **82.31** | **82.61** | **82.67** | **82.61** |
| (5e-05, 8)      | 79.96 | 80.24    | 80.40     | 80.24  |
| (2e-05, 8)      | 82.36 | 82.57    | 82.86     | 82.57  |
| (5e-05, 16)     | 65.17 | 68.04    | 68.36     | 68.04  |
| (2e-05, 16)     | 81.99 | 82.19    | 82.48     | 82.19  |

---

### Table: QwenVL + ModernBERT-large results

This table displays the performance metrics for the ModernBERT-large model.

| Config (lr, bs) | F1    | Accuracy | Precision | Recall |
|-----------------|-------|----------|-----------|--------|
| (1e-05, 8)      | 0.8387 | 0.8402   | 0.8446    | 0.8402 |
| (2e-05, 8)      | **0.8459** | **0.8472** | **0.8521** | **0.8472** |
| (5e-05, 8)      | 0.8158 | 0.8174   | 0.8225    | 0.8174 |
| (1e-05, 16)     | 0.8285 | 0.8294   | 0.8381    | 0.8294 |
| (2e-05, 16)     | 0.8310 | 0.8319   | 0.8406    | 0.8319 |
| (5e-05, 16)     | 0.8114 | 0.8120   | 0.8255    | 0.8120 |

*Note: ModernBERT-large experiments were conducted by a collaborating research team. The implementation scripts for this model are not included in this repository.*

*Note: The best performing configuration for each experimental setup is highlighted in bold.*

## License

This project is for research purposes. Please refer to the MMSD2.0 dataset license for data usage terms.

## Acknowledgments

- Original MMSD2.0 dataset creators and contributors
- The open-source community for providing the foundational models and libraries used in this project