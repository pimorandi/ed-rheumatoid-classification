# Results

This directory contains the outputs from each run of `fast_classification.py`. Each run creates a separate subfolder with the following files:

## Folder Structure

Each run folder contains:

- **`config.json`** - Configuration file with all model training parameters
- **`train_predictions.csv`** - Model predictions on the training set
- **`test_predictions.csv`** - Model predictions on the test set
- **Performance metrics visualizations:**
  - `roc_auc_train.png` - ROC curve for training set
  - `roc_auc_test.png` - ROC curve for test set
  - `probability_distribution_train.png` - Output probability distribution for training set
  - `probability_distribution_test.png` - Output probability distribution for test set
  - `performance_train.png` - Performance metrics for training set
  - `performance_test.png` - Performance metrics for test set

## Note on Data Privacy

Due to privacy regulations, prediction files and results generated from hospital data are not included in this repository. The folder structure is provided for reproducibility purposes. Users with appropriate data access can generate results by running the classification script with their own data.