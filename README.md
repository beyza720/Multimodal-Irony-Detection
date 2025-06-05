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

This project uses the **MMSD2.0 (Multimodal Sarcasm Detection Dataset 2.0)**, which is a comprehensive dataset containing social media posts with both text and images, labeled for sarcasm detection.

### Dataset Source
The MMSD2.0 dataset is available at: [https://github.com/JoeYing1019/MMSD2.0](https://github.com/JoeYing1019/MMSD2.0)

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
- **`sample_data_qwenvl.csv`**: Sample data with image descriptions generated using Qwen2.5-VL model
- **`sample_data_internvl.csv`**: Sample data with image descriptions generated using InternVL3 model

Both sample files use the same image IDs, allowing for direct comparison of how different vision-language models describe the same images.

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

## License

This project is for research purposes. Please refer to the MMSD2.0 dataset license for data usage terms.

## Acknowledgments

- Original MMSD2.0 dataset creators and contributors
- The open-source community for providing the foundational models and libraries used in this project