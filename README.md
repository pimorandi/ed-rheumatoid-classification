# ED Rheumatoid Classification

A text classification system for identifying rheumatoid symptoms in emergency department clinical notes using natural language processing and deep learning.

## Overview

This project develops a machine learning model to classify patients as having rheumatoid symptoms based on clinical notes written during emergency department (ED) admissions. The model uses transformer-based embeddings to process free-text clinical documentation and predict the presence of rheumatoid pathologies.

## Project Structure
```
ed-rheumatoid-classification/
├── data/
│   ├── raw/                # Raw data from hospital database (not included)
│   └── processed/          # Preprocessed, tokenized data (not included)
├── configs/
│   ├── fast_classification.json # Classification settings
│   └── preprocess_data.json     # Preprocessing settings
├── procedures/             # Data extraction and SQL queries
├── preprocess_data.py      # Data preprocessing and propensity score matching
├── fast_classification.py  # Main training and evaluation script
├── get_data.py             # Script to extract data from hospital database
├── misclass_analysis.py    # Script that extracts other insights from the data in a results folder
├── results/                # Model outputs and performance metrics
└── README.md
```

## Features

- **Propensity Score Matching**: Balanced control group selection to reduce confounding
- **Transformer-based Embeddings**: Neural representation of clinical text
- **Class Weighting**: Mitigation of class imbalance in loss function calculation
- **Comprehensive Evaluation**: ROC curves, probability distributions, and performance metrics

## Usage

### 1. Data Extraction

Extract raw data from the hospital database:
```bash
python get_data.py --query ./procedures/<query_name.sql> --filename <filename>
```

This generates a dataset with the following fields:
- Patient ID, Gender, Age
- Priority level, ED Admission ID, Timestamp
- Associated diagnosis, Symptoms description, Anamnesis

### 2. Data Preprocessing

Preprocess raw data with propensity score matching and tokenization:
```bash
python preprocess_data.py --filename ./data/raw/<filename>
```

Outputs JSONL files with:
- Anonymized patient IDs
- Classification labels (1 = rheumatoid symptoms, 0 = control)
- Tokenized input sequences (input_ids, token_type_ids, attention_mask)

### 3. Model Training

Train the classification model:
```bash
python fast_classification.py --train ./data/processed/<train_set> --test ./data/processed/<test_set>
```

Each run creates a results folder containing:
- `config.json` - Training parameters
- `train_predictions.csv` / `test_predictions.csv` - Model predictions
- ROC curves, probability distributions, and performance visualizations

## Model Details

The classification model:
- Uses clinical text embeddings as input features
- Implements class weights in the loss function to handle class imbalance
- Avoids SMOTE oversampling (inappropriate for embedding space, as it would generate synthetic reports with no clinical or linguistic validity)

## Data Privacy

⚠️ **Important**: This repository does not include actual patient data due to privacy regulations and hospital data protection policies.

All data files (`data/raw/`, `data/processed/`, and prediction outputs) are excluded from version control. 

