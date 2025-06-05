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

- **Image Description Generation**: Scripts for generating textual descriptions of images using models like InternVL and QwenVL
- **Text Classification**: Implementation of various transformer models for sarcasm detection
- **Multimodal Analysis**: Experiments combining visual and textual features
- **Zero-shot Learning**: Testing models' performance without specific training on the target task

## Technology Stack

- **Python 3.10+**
- **PyTorch 2.2.2** for deep learning with CUDA support
- **Transformers 4.51.3+** library for pre-trained models
- **Weights & Biases** for experiment tracking
- **Various vision-language models** (InternVL, QwenVL, PaliGemma)
- **Quantization support** via bitsandbytes and triton

## Environment Setup

### Option 1: Virtual Environment (Recommended)
```bash
python3 -m venv qwen_env
source qwen_env/bin/activate  # On Windows: qwen_env\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Conda Environment
```bash
conda create -n multimodal-sarcasm python=3.10
conda activate multimodal-sarcasm
pip install -r requirements.txt
```

## Getting Started

1. Clone this repository
2. Set up your environment (see Environment Setup above)
3. Download the MMSD2.0 dataset from the official repository
4. Organize the dataset according to the structure described in Dataset Setup
5. Update dataset paths in the scripts to match your local setup
6. Run the desired experiments

### Example Usage

**Image Description Generation:**
```bash
# For training data
python image_description_scripts/qwen_vl_image_description.py

# Note: Script is configured for train.json by default
# For validation/test data, modify the script to use valid.json or test.json
```

**Text Classification:**
```bash
python text_classification/text_classification.py
```

## Script Adaptability

The image description generation script (`qwen_vl_image_description.py`) is designed for the training split by default. To process validation or test data:

1. Change `train_file = os.path.join(dataset_path, "train.json")` to:
   - `valid_file = os.path.join(dataset_path, "valid.json")` for validation data
   - `test_file = os.path.join(dataset_path, "test.json")` for test data

2. Update the corresponding DataFrame loading and output file names accordingly

## Citation


```

## License

This project is for research purposes. Please refer to the MMSD2.0 dataset license for data usage terms.

## Acknowledgments

- Original MMSD2.0 dataset creators and contributors
- The open-source community for providing the foundational models and libraries used in this project 
