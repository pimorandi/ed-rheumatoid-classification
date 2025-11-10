# Raw Data

This directory contains raw data extracted from the hospital database using the `procedures/get_data.py` script.

## Data Description

The raw dataset includes emergency department admission records with the following fields:

| Field | Description |
|-------|-------------|
| Patient ID | Unique patient identifier |
| Gender | Patient gender |
| Age | Patient age at admission |
| Priority Level | Triage priority level assigned at admission |
| ED Admission ID | Emergency department admission identifier |
| ED Timestamp | Date and time of emergency department admission |
| Associated Diagnosis | Clinical diagnosis associated with the visit |
| Symptoms Description | Documented patient symptoms |
| Anamnesis | Patient medical history |

## Data Format

Data is stored in CSV format with one row per emergency department admission.

## Note on Data Privacy

⚠️ **This folder does not contain actual patient data.**

Due to patient privacy regulations and hospital data protection policies, raw data files containing protected health information (PHI) are excluded from this repository. 

Raw data files are listed in `.gitignore` to prevent accidental commits of sensitive information.