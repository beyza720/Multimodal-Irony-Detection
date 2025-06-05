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

## Key Components

- **Image Description Generation**: Scripts for generating textual descriptions of images using models like InternVL and QwenVL
- **Text Classification**: Implementation of various transformer models for sarcasm detection
- **Multimodal Analysis**: Experiments combining visual and textual features
- **Zero-shot Learning**: Testing models' performance without specific training on the target task

## Technology Stack



## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the MMSD2.0 dataset from the official repository
4. Configure your paths and experiment settings
5. Run the desired experiments

## Citation



## License

This project is for research purposes. Please refer to the MMSD2.0 dataset license for data usage terms.

## Acknowledgments

- Original MMSD2.0 dataset creators and contributors
- The open-source community for providing the foundational models and libraries used in this project 
