# Natural Language Inference Data Generation with OOD Detection

This repository provides the official implementation of our EMNLP 2025 paper:
**"Rule Discovery for Natural Language Inference Data Generation Using Out-of-Distribution Detection"**.

---

## Overview

We propose a semi-automated framework that combines OOD detection, clustering, and LLM-based prompting to discover new sentence transformation rules for Natural Language Inference (NLI). Our method identifies unexplained patterns in SNLI and generates high-quality synthetic data for training NLI models.

This package includes all code and datasets necessary to fully reproduce the experiments described in our EMNLP 2025 paper. To further improve usability and long-term maintainability, we plan to modularize and release the framework on GitHub in a future update.

---

## Requirements

### Computing Infrastructure
Our experiments were conducted using the following hardware and software environment:

#### Hardware
- GPU: NVIDIA RTX A6000 (48GB VRAM)
- CPU: Intel Core i9-9820X (10 cores)
- RAM: 16GB

#### Software
- CUDA: 12.2
- cuDNN: 8.9.2
- Python: ≥ 3.8

### Python Dependencies
```bash
torch>=1.5.0
transformers==2.9.0
pytorch-lightning==0.7.3
spacy==2.2.4
torch-optimizer==0.0.1a9
matplotlib>=3.1.1
scikit-learn>=0.24
numpy>=1.19
```

## Directory Overview

```
├── 1. generation_15_transformation_with_CoT/        # Data generation for 15 rules (prompts and outputs)
│   ├── code_make_data_15_transformation_rules/     # Scripts to generate PHLs from prompts (CoT_AM.py, CoT_Con.py, ..., CoT_SSNCV.py)
│   └── prompt_15_transformation_rules/             # CoT prompts for 15 rules (AM_prompt.txt, CON_prompt.txt, ..., SSNCV_prompt.txt)
├── 2. train_test_split_generated_data/              # Contains script for splitting generated data (train_test_split.py) 
├── 3. fine_tuning_with_15_transformation_data/      # BERT fine-tuning scripts (run_sst.py, modeling_bert.py, classification_15_transformation_rules.py, finetuning_file_for_15_transformation_rules.sh)      # BERT fine-tuning scripts for 15-rule classification
├── 4. ood_detection/                                # OOD detection using MSP + TS + IP
├── 5. clustering_with_ood_data/                     # Clustering low-confidence pairs, t-SNE visualization
├── 6. discovered_data/                              # LLM-based rule discovery and similarity scoring
│   └── automated_discovered/                        # Contains rule generation and validation modules
│       ├── code_discover_new_rule/                  # LLM-based rule induction scripts (e.g., automated_discovered_rule.py)
│       ├── code_similarity_with_s_bert/             # Semantic similarity analysis using SBERT (e.g., analysis_hypothesis_LLM.py) 
│       └── prompt_discover_new_rule/                # Prompt templates for rule and hypothesis generation
├── 7. data_augmentation/                            # GPT-generated data for newly discovered rules
│   ├── code_new_rule_data_augmentation/            # Scripts to generate new-rule-based data (e.g., CoT_new_CA.py, CoT_new_EI.py, CoT_new_RG.py, CoT_new_VS.py) 
│   └── prompt_new_rule_data_augmentation/          # CoT prompts for new rule data generation (e.g., CA_prompt.txt, EI_prompt.txt, RG_prompt.txt, VS_prompt.txt) 
├── 8. experiments_with_augmentation_data/           # Scripts to evaluate impact of augmented datasets (e.g., experiments_for_distribution.py)  
│
├── transformation_rules/                            # Rule-specific generation scripts
│   ├── 15_rules/                                     # 15 original rules (e.g., CoT_AM.py, ...)
│   └── new_rules/                                    # 4 discovered rules (e.g., CoT_RG.py, ...)
```

## Experimental Pipeline

### 1. Data Generation Using 15 Transformation Rules
- **Input**: `snli_trainset_for_generation_15_transformation_rules/snli_train_original.jsonl`
- **Prompt**: `prompt_15_transformation_rules/`
- **Generation Code**: `code_make_data_15_transformation_rules/`
- **Output**: Generated data saved to `dataset/data_15_transformation_rules/original/`
- **Split**: Train/test split (8:2) using:
  - `train_test_split_generated_data/train_test_split.py`
  - Output: `train_test_split_data/15rules_ml_train_data`, `15rules_ml_test_data`

---

### 2. Fine-tuning on 15-rule Data
- **Training Code**:
  - `finetuning_file_for_15_transformation_rules.sh`
  - `run_sst.py`, `classification_15_transformation_rules.py`, `modeling_bert.py`
- **Input**: 15-rule training/test data
- **Output**: Fine-tuned BERT classification model

---

### 3. OOD Detection
- **Notebook**: `ood_detection/OOD_Detection.ipynb`
- **Methods**:
  - MSP (Maximum Softmax Probability)
  - MSP + TS (Temperature Scaling)
  - MSP + TS + IP (Input Preprocessing)
- **Output**: OOD detection scores and ROC analysis

---

### 4. Clustering of OOD Samples
- **Input**: 50,000 low-confidence SNLI PHL pairs (OOD)
- **Code**: `clustering_with_ood_detected_data/clustering_with_ood_detected_data.ipynb`
- **Output**:
  - Cluster visualization (t-SNE)
  - Cluster metrics: `Cluster_output_10000.xlsx`
  - Cluster analysis: `Cluster_analysis/`

---

### 5. Automated Rule Discovery
- **Folder**: `automated_discovered/`
- **Code**:
  - Rule discovery: `code_discover_new_rule/`
  - Hypothesis similarity check: `code_similarity_with_s_bert/analysis_hypothesis_LLM.py`
- **Prompts**:
  - Rule evaluation: `prompt_discover_new_rule/prompt_discover_existing_vs_new.txt`
  - Hypothesis generation: `prompt_discover_new_rule/prompt_generate_hypothesis.txt`

---

### 6. Data Augmentation with New Rules
- **Code**: `data_augmentation/code_new_rule_data_augmentation/`
- **Prompt**: `data_augmentation/prompt_new_rule_data_augmentation/`
- **Output Dataset**: `generated_data_for_distribution/distribution_experiments/`
- **Goal**: Add 1,000 PHL pairs per rule using GPT-4o-mini + CoT prompting

---

### 7. Final Evaluation
- **Script**: `experiments_for_distribution`
- **Metrics**:
  - Accuracy, AUROC, FPR@TPR
  - Semantic similarity for rule validity
- **Findings**:
  - New rules (RG, CA, VS, EI) yield +0.74%p improvement
  - Distribution-aware augmentation performs best in all dataset sizes (2k to 550k)

---

## New Transformation Rules

In addition to the original 15 rules, we discovered 4 new transformation rules through OOD detection and clustering:

1. **Role Generalization (RG)**: Transforms specific roles into general expressions  
   _e.g._ "baseball player" → "athlete"

2. **Contextual Augmentation (CA)**: Derives purpose and background information  
   _e.g._ "performing on street" → "performing on street to collect donations"

3. **Visual Specification (VS)**: Adds detailed visual characteristics  
   _e.g._ "straw hat" → "dirty straw hat"

4. **Emotion Inference (EI)**: Infers emotions based on actions  
   _e.g._ "throwing rocks" → "throwing rocks because he is bored"

---

## Performance Summary

| Dataset                      | Type            | 550k            | 50k             | 10k             | 2k              |
|-----------------------------|------------------|------------------|------------------|------------------|------------------|
| **Original**                | —                | 89.85 (±0.362)   | 86.06 (±0.240)   | 82.29 (±0.347)   | 76.31 (±0.513)   |
| **+15 rules (+4,500)**      | Uniform          | 89.95 (±0.359)   | 86.14 (±0.255)   | 82.49 (±0.392)   | 76.71 (±0.806)   |
|                             | Dist.-aware      | 89.99 (±0.158)   | 86.19 (±0.222)   | 82.62 (±0.178)   | 76.43 (±0.883)   |
| **+19 rules (+5,700)**      | Uniform          | 90.00 (±0.405)   | 86.17 (±0.224)   | 82.64 (±0.383)   | 76.94 (±0.835)   |
| **+19 rules (Ours)**        | Dist.-aware      | **90.00 (±0.134)** | **86.24 (±0.175)** | **82.72 (±0.258)** | **77.16 (±0.392)** |



---