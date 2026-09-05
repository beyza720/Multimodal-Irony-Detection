# Lightweight Multimodal Irony Detection by Image Caption Generation

Code for **"Lightweight Approach for Multi-Modal Irony Detection by Image Caption Generation"**, presented at **IEEE BigData 2025**.

Beyza Nur Koç, Recep Fırat Çekinel, Pınar Karagöz — Department of Computer Engineering, Middle East Technical University.

📄 [Paper (PDF)](Lightweight%20Approach%20for%20Multi-Modal%20Irony%20Detection%20by%20Image%20Caption%20Generation.pdf) · 🔗 [DOI: 10.1109/BigData66926.2025.11402099](https://doi.org/10.1109/BigData66926.2025.11402099) · 📚 [IEEE Xplore](https://ieeexplore.ieee.org/document/11402099)

---

## What this does

Multimodal irony detection usually requires end-to-end training over both a vision encoder and a text encoder. We avoid that entirely.

Instead, we **turn the image into text and then classify text**:

1. A vision–language model (Qwen2.5-VL-7B or InternVL3-8B) generates a caption for the image. **No fine-tuning** — inference only.
2. The caption is concatenated with the original post text.
3. A transformer text classifier (XLM-RoBERTa or ModernBERT) is fine-tuned on the combined text.

The multimodal problem becomes a text classification problem, so only the text encoder is ever trained.

**Result:** this matches Multi-view CLIP on MMSD2.0 (84.59 vs. 84.64 macro-F1) and outperforms it on the augmented MMSD2.0+MORE benchmark (85.95 vs. 84.81 macro-F1) — without any joint visual–textual training.

---

## Results

All numbers are from the published paper. Macro-F1 is the primary metric; all evaluation is on the MMSD2.0 test set unless stated otherwise.

### Main comparison on MMSD2.0

| Model | Input | Image encoder | Text encoder | Setup | Acc | **F1** | Prec | Rec |
|---|---|---|---|---|---|---|---|---|
| Zero-shot-1 | image + text (vanilla prompt) | Dynamic ViT | Qwen2.5 LLM | Zero-shot | 69.12 | 67.73 | 66.78 | 56.22 |
| Zero-shot-2 | image + text (detailed prompt) | Dynamic ViT | Qwen2.5 LLM | Zero-shot | 61.15 | 60.99 | 78.40 | 53.31 |
| Zero-shot-3 | image caption + text | – | Mistral-7B | Zero-shot | 67.04 | 61.05 | 71.52 | 63.17 |
| Zero-shot-4 | text only | – | Mistral-7B | Zero-shot | 64.80 | 64.45 | 64.45 | 64.68 |
| Zero-shot-5 | image caption only | – | Mistral-7B | Zero-shot | 64.51 | 55.87 | 71.02 | 59.51 |
| Baseline-text | text only | Qwen2.5-VL-7B | xlm-roberta-base | Fine-tuned | 78.79 | 78.58 | 79.21 | 78.79 |
| Baseline-text | text only | InternVL3-8B | xlm-roberta-base | Fine-tuned | 78.33 | 78.15 | 78.87 | 78.33 |
| Baseline-caption | image caption only | Qwen2.5-VL-7B | xlm-roberta-base | Fine-tuned | 74.55 | 74.19 | 74.74 | 74.55 |
| Baseline-caption | image caption only | InternVL3-8B | xlm-roberta-base | Fine-tuned | 75.34 | 75.02 | 75.59 | 75.34 |
| Multi-view CLIP | image + text | clip-vit-base | CLIP Transformer | Fine-tuned | 84.81 | **84.64** | 84.49 | 86.42 |
| **Ours** | image caption + text | Qwen2.5-VL-7B | ModernBERT-large | Fine-tuned | 84.72 | **84.59** | 85.21 | 84.72 |

Adding captions to the text lifts macro-F1 from 78.58 to 84.59 — visual information carries complementary signal that text alone misses.

### Data augmentation: MMSD2.0 + cleaned MORE

Multi-view CLIP was re-implemented and retrained on the identical merged splits for a fair comparison.

| Model | Input | Image encoder | Text classifier | Acc | **F1** | Prec | Rec |
|---|---|---|---|---|---|---|---|
| Multi-view CLIP | image + text | clip-vit-base | CLIP Transformer | 84.97 | 84.81 | 84.66 | 86.60 |
| Ours | image caption + text | Qwen2.5-VL-7B | xlm-roberta-large | 84.28 | 84.27 | 84.34 | 84.28 |
| **Ours** | image caption + text | Qwen2.5-VL-7B | ModernBERT-large | **85.98** | **85.95** | **86.27** | **85.98** |

Adding ~3.5k posts from MORE gains +1.14 F1 and cuts sarcastic false negatives by roughly 20%, overtaking Multi-view CLIP on the same data.

### Text classifier comparison

| Input | Image encoder | Text classifier | Acc | **F1** | Prec | Rec |
|---|---|---|---|---|---|---|
| image caption + text | Qwen2.5-VL-7B | xlm-roberta-base | 82.27 | 82.00 | 82.39 | 82.27 |
| image caption + text | InternVL3-8B | xlm-roberta-base | 82.57 | 82.42 | 83.10 | 82.57 |
| image caption + text | Qwen2.5-VL-7B | xlm-roberta-large | 83.48 | 83.35 | 84.04 | 83.48 |
| image caption + text | Qwen2.5-VL-7B | ModernBERT-large | **84.72** | **84.59** | 85.21 | 84.72 |

InternVL3 produces better captions in the caption-only setting, but once modalities are concatenated the two VLMs are effectively equivalent — the fusion step neutralises caption quality differences. Qwen2.5-VL-7B was therefore used for all classifier comparisons.

<details>
<summary><b>Hyperparameter sweeps (click to expand)</b></summary>

Search space: learning rate ∈ {1e-5, 2e-5, 5e-5}, batch size ∈ {4, 8, 16}. Best configuration selected by validation macro-F1.

**Qwen2.5-VL-7B + xlm-roberta-base, text only**

| Config (lr, bs) | F1 | Acc | Prec | Rec |
|---|---|---|---|---|
| (5e-05, 4) | 66.37 | 66.67 | 67.21 | 66.67 |
| (2e-05, 4) | **78.58** | **78.79** | **79.21** | **78.79** |
| (5e-05, 8) | 76.58 | 76.67 | 77.79 | 76.67 |
| (2e-05, 8) | 77.41 | 77.63 | 78.07 | 77.63 |
| (5e-05, 16) | 77.42 | 77.54 | 78.40 | 77.54 |
| (2e-05, 16) | 78.09 | 78.25 | 78.90 | 78.25 |

**Qwen2.5-VL-7B + xlm-roberta-base, image caption only**

| Config (lr, bs) | F1 | Acc | Prec | Rec |
|---|---|---|---|---|
| (5e-05, 4) | 68.03 | 68.83 | 68.69 | 68.83 |
| (2e-05, 4) | 73.60 | 73.97 | 74.15 | 73.97 |
| (5e-05, 8) | 36.29 | 56.95 | 32.44 | 56.95 |
| (2e-05, 8) | 72.82 | 73.18 | 73.41 | 73.18 |
| (5e-05, 16) | 74.03 | 74.39 | 74.59 | 74.39 |
| (2e-05, 16) | **74.19** | **74.55** | **74.74** | **74.55** |

**InternVL3-8B + xlm-roberta-base, text only**

| Config (lr, bs) | F1 | Acc | Prec | Rec |
|---|---|---|---|---|
| (5e-05, 4) | 36.29 | 56.95 | 32.44 | 56.95 |
| (2e-05, 4) | 69.29 | 69.49 | 70.24 | 69.49 |
| (5e-05, 8) | 65.28 | 65.34 | 66.89 | 65.34 |
| (2e-05, 8) | 77.45 | 77.67 | 78.12 | 77.67 |
| (5e-05, 16) | **78.15** | **78.33** | **78.87** | **78.33** |
| (2e-05, 16) | 77.15 | 77.33 | 77.89 | 77.33 |

**InternVL3-8B + xlm-roberta-base, image caption only**

| Config (lr, bs) | F1 | Acc | Prec | Rec |
|---|---|---|---|---|
| (5e-05, 4) | 67.81 | 68.12 | 68.60 | 68.12 |
| (2e-05, 4) | 73.81 | 74.01 | 74.64 | 74.01 |
| (5e-05, 8) | **75.02** | **75.34** | **75.59** | **75.34** |
| (2e-05, 8) | 74.02 | 74.30 | 74.68 | 74.30 |
| (5e-05, 16) | 73.28 | 73.56 | 73.96 | 73.56 |
| (2e-05, 16) | 74.77 | 75.01 | 75.49 | 75.01 |

**Qwen2.5-VL-7B + ModernBERT-large, text only**

| Config (lr, bs) | F1 | Acc | Prec | Rec |
|---|---|---|---|---|
| (1e-05, 8) | 83.87 | 84.02 | 84.46 | 84.02 |
| (2e-05, 8) | **84.59** | **84.72** | **85.21** | **84.72** |
| (5e-05, 8) | 81.58 | 81.74 | 82.24 | 81.73 |
| (1e-05, 16) | 82.84 | 82.94 | 83.81 | 82.94 |
| (2e-05, 16) | 83.09 | 83.19 | 84.06 | 83.19 |
| (5e-05, 16) | 81.14 | 81.19 | 82.54 | 81.20 |

**Qwen2.5-VL-7B + xlm-roberta-large, MMSD2.0 + MORE (cleaned)**

| Config (lr, bs) | F1 | Acc | Prec | Rec |
|---|---|---|---|---|
| (2e-05, 16) | **84.27** | **84.28** | **84.34** | **84.28** |
| (1e-05, 8) | 83.85 | 83.85 | 83.85 | 83.85 |
| (1e-05, 16) | 83.02 | 83.05 | 83.23 | 83.05 |

**Qwen2.5-VL-7B + ModernBERT-large, MMSD2.0 + MORE (cleaned)**

| Config (lr, bs) | F1 | Acc | Prec | Rec |
|---|---|---|---|---|
| (1e-05, 16) | 84.56 | 84.57 | 84.69 | 84.57 |
| (2e-05, 16) | 83.71 | 83.74 | 83.87 | 83.74 |
| (5e-05, 16) | **85.95** | **85.98** | **86.27** | **85.98** |

</details>

> **Note on ModernBERT.** The ModernBERT-large experiments were run by a collaborating research team, so those training scripts are not part of this repository. Everything else reported above is reproducible from the code here.

---

## Method

### 1. Image caption generation

Each image is converted to a natural-language description by a VLM in **inference mode only** — no fine-tuning. Qwen2.5-VL-7B and InternVL3-8B were selected based on Open VLM Leaderboard results and preliminary caption-quality checks.

Prompt used for both models:

> Please describe this image in detail. Focus on the visual elements, composition, and any notable features. Keep the description under 250 tokens.

### 2. Merging text and image information

The generated caption is concatenated with the original post text into a single string, which becomes the classifier input. This is what reduces the multimodal task to text classification.

### 3. Irony classification

The combined text is tokenized and passed to a fine-tuned transformer classifier (XLM-RoBERTa or ModernBERT) for binary irony detection.

### Training setup

- Up to 5 epochs, AdamW optimizer, cosine learning rate schedule
- Best checkpoint selected by validation macro-F1
- Trained on GPUs with ≥ 40 GB memory
- Reported metrics: accuracy, macro-F1, precision, recall (macro-F1 primary)

---

## Datasets

Neither dataset is redistributed here, for copyright and ethical reasons. Download both from their official sources.

### MMSD2.0 — primary

Multimodal posts from Twitter, Instagram and Tumblr, collected by distant supervision on hashtags such as `#sarcasm` and `#irony`, then manually annotated to remove false positives and balance the classes.

| Split | Posts | Sarcastic | Non-sarcastic |
|---|---|---|---|
| Train | 19,816 | 9,572 | 10,240 |
| Validation | 2,410 | – | – |
| Test | 2,409 | 1,037 | 1,372 |
| **Total** | **24,635** | | |

Source: https://github.com/JoeYing1019/MMSD2.0

### MORE — augmentation

Released for multimodal sarcasm *explanation* rather than detection, so **every instance is sarcastic** — there are no negative examples. It contains 3,510 sarcastic posts with expert-written explanations (train 2,983 / validation 175 / test 352).

We adapt it for detection by treating the presence of an explanation as a positive sarcasm label, apply the MMSD2.0 cleaning strategy (emoji removal, spurious hashtag removal), and merge it with MMSD2.0.

Source: https://github.com/LCS2-IIITD/Multimodal-Sarcasm-Explanation-MuSE
*(The repository is named MuSE; the dataset is referred to as MORE in the paper and throughout this README.)*

### Expected directory layout

```
mmsd_dataset/
├── extracted_part_1/    # images, split across 6 directories
├── extracted_part_2/
├── extracted_part_3/
├── extracted_part_4/
├── extracted_part_5/
├── extracted_part_6/
├── train.json           # image ids, text, sarcasm labels (0 / 1)
├── valid.json
└── test.json
```

Point the `dataset_path` variable in the scripts at this directory.

---

## Setup

Python 3.10+. The two VLMs need different dependency sets, so use separate environments.

**Qwen2.5-VL**

```bash
python3 -m venv qwen_env
source qwen_env/bin/activate        # Windows: qwen_env\Scripts\activate
pip install -r requirements_qwenvl.txt
```

**InternVL3**

```bash
python3 -m venv internvl_env
source internvl_env/bin/activate    # Windows: internvl_env\Scripts\activate
pip install -r requirements_internvl.txt
```

Core stack: PyTorch 2.2.2 (CUDA), Transformers 4.51.3+, LMDeploy 0.7.3 (InternVL serving), bitsandbytes + triton (4-bit quantization), Weights & Biases (experiment tracking).

---

## Usage

### Step 1 — Generate image captions

```bash
# Qwen2.5-VL-7B-Instruct, 4-bit quantization supported
python image_description_scripts/qwen_vl_image_description.py

# InternVL3-8B via LMDeploy; processes train and validation splits automatically
python image_description_scripts/intern_vl_image_description.py

# MORE dataset
python image_description_scripts/muse_image_description_batch.py
```

The Qwen script defaults to `train.json`; edit it to target `valid.json` or `test.json`. Both scripts emit CSVs with an identical schema so downstream steps are interchangeable.

### Step 2 — Combine text with captions

```bash
python image_description_scripts/combine_texts.py \
    --input  mmsd_image_descriptions_train.csv \
    --output mmsd_combined_train.csv
```

Takes any CSV with `text` and `image_description` columns and adds a `combined_text` column.

### Step 3 — Train the classifier

```bash
cd text_classification
./run_experiments_robertabase.sh
```

To run against InternVL captions instead, change the `--train_file` / `--valid_file` / `--test_file` paths in the script from `mmsd_image_description_with_QwenVL/` to `mmsd_image_description_with_InternVL/`.

### Step 4 — Merged MMSD2.0 + MORE experiments

```bash
cd text_classification
./run_experiments_merged.sh
```

Requires both datasets already processed through Steps 1–2, cleaned and merged. `text_classification_merged.py` handles the differing column structures and logs the MMSD/MORE composition of each split.

---

## Repository layout

```
.
├── image_description_scripts/
│   ├── qwen_vl_image_description.py     # Qwen2.5-VL-7B captioning
│   ├── intern_vl_image_description.py   # InternVL3-8B captioning (LMDeploy)
│   ├── muse_image_description_batch.py  # MORE dataset captioning
│   └── combine_texts.py                 # text + caption concatenation
├── text_classification/
│   ├── text_classification_robertabase.py
│   ├── text_classification_all_large.py
│   ├── text_classification_merged.py
│   ├── text_classification_merged_large.py
│   └── run_experiments_*.sh             # sweep drivers
├── mmsd_sample_data_qwenvl.csv          # 50 MMSD2.0 rows, Qwen captions
├── mmsd_sample_data_internvl.csv        # 50 MMSD2.0 rows, InternVL captions
├── muse_sample_data_qwenvl.csv          # 36 MORE rows, Qwen captions
├── requirements_qwenvl.txt
└── requirements_internvl.txt
```

### Sample data

Small samples are included so the data format is inspectable without downloading the full datasets. The two MMSD2.0 samples use the **same image IDs**, so you can compare directly how Qwen2.5-VL and InternVL3 describe an identical image.

| Column | Description |
|---|---|
| `image_id` | Image identifier, e.g. `840006160660983809.jpg` |
| `text` | Original social media post text |
| `label` | Sarcasm label — 0 non-sarcastic, 1 sarcastic |
| `image_description` | VLM-generated caption |
| `image_location` | Path where the image was found during processing |
| `combined_text` | `text` + `image_description`, the classifier input |

---

## Citation

```bibtex
@INPROCEEDINGS{11402099,
  author    = {Koc, Beyza Nur and Cekinel, Recep Firat and Karagoz, Pinar},
  booktitle = {2025 IEEE International Conference on Big Data (BigData)},
  title     = {Lightweight Approach for Multi-Modal Irony Detection by Image Caption Generation},
  year      = {2025},
  pages     = {2908-2914},
  doi       = {10.1109/BigData66926.2025.11402099}
}
```

Please also cite the dataset papers if you use them:

- **MMSD2.0** — Qin et al., *MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System*, Findings of ACL 2023.
- **MORE** — Desai et al., *Nice Perfume. How Long Did You Marinate in It? Multimodal Sarcasm Explanation*, AAAI 2022.

## Acknowledgments

This work was carried out as guided research (CENG488) at the Department of Computer Engineering, Middle East Technical University, supervised by Prof. Dr. Pınar Karagöz and Recep Fırat Çekinel.

## License

MIT — see [LICENSE](LICENSE).
