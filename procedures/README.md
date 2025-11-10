# Procedures

This directory contains SQL queries and data extraction scripts for retrieving clinical data from the hospital database.

## Scripts

- **`get_data.py`** - Python script that executes SQL queries to extract emergency department data

## Data Extraction

The `get_data.py` script retrieves patient records from emergency department admissions with the following fields:

- **Patient ID** - Unique patient identifier
- **Gender** - Patient gender
- **Age** - Patient age at admission
- **Priority Level** - Triage priority level
- **ED Admission ID** - Emergency department admission identifier
- **ED Timestamp** - Date and time of emergency department admission
- **Associated Diagnosis** - Clinical diagnosis associated with the visit
- **Symptoms Description** - Documented patient symptoms
- **Anamnesis** - Patient medical history

## Output

Query results are stored in the `data/raw/` folder.

## Note on Data Privacy

Due to patient privacy regulations and hospital data protection policies, raw data files are not included in this repository. The SQL queries and extraction scripts are provided for reproducibility purposes. Users must have appropriate institutional data access and ethical approval to run these scripts.