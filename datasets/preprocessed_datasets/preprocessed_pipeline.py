# ============================================================
# MULTIMODAL MRI PREPROCESSING PIPELINE FOR ASD DETECTION
# ============================================================
# This script:
# 1. Loads Anatomical, DTI, and Functional MRI datasets
# 2. Selects important features
# 3. Renames duplicate columns
# 4. Merges datasets using Sub_ID
# 5. Handles missing values
# 6. Normalizes features
# 7. Saves the final preprocessed dataset
#
# Author: Your Name
# ============================================================

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# STEP 1: LOAD DATASETS
# ============================================================

# File paths
anat_file = "anat_qap.csv"
dti_file = "dti_qap.csv"
func_file = "functional_qap.csv"

# Read CSV files
anat_df = pd.read_csv(anat_file)
dti_df = pd.read_csv(dti_file)
func_df = pd.read_csv(func_file)

print("Datasets Loaded Successfully")
print()

# ============================================================
# STEP 2: SELECT IMPORTANT FEATURES
# ============================================================

# ---------- Anatomical MRI Features ----------
anat_features = anat_df[
    [
        'Sub_ID',
        'CNR',
        'SNR',
        'QI1',
        'EFC',
        'FBER'
    ]
]

# ---------- DTI Features ----------
dti_features = dti_df[
    [
        'Sub_ID',
        'SNR',
        'PercentFD_greater_than_0.20',
        'MeanFD',
        'DVARS'
    ]
]

# ---------- Functional MRI Features ----------
func_features = func_df[
    [
        'Sub_ID',
        'SNR',
        'PercentFD_greater_than_0.20',
        'MeanFD',
        'DVARS',
        'GCOR'
    ]
]

print("Feature Selection Completed")
print()

# ============================================================
# STEP 3: RENAME COLUMNS TO AVOID DUPLICATES
# ============================================================

# Rename Anatomical columns
anat_features = anat_features.rename(columns={
    'SNR': 'SNR_anat',
    'QI1': 'QI1_anat',
    'EFC': 'EFC_anat',
    'FBER': 'FBER_anat'
})

# Rename DTI columns
dti_features = dti_features.rename(columns={
    'SNR': 'SNR_dti',
    'PercentFD_greater_than_0.20': 'FD_dti',
    'MeanFD': 'MeanFD_dti',
    'DVARS': 'DVARS_dti'
})

# Rename Functional columns
func_features = func_features.rename(columns={
    'SNR': 'SNR_func',
    'PercentFD_greater_than_0.20': 'FD_func',
    'MeanFD': 'MeanFD_func',
    'DVARS': 'DVARS_func',
    'GCOR': 'GCOR_func'
})

print("Column Renaming Completed")
print()

# ============================================================
# STEP 4: MERGE DATASETS USING Sub_ID
# ============================================================

# Merge Anatomical and DTI datasets
merged_df = pd.merge(
    anat_features,
    dti_features,
    on='Sub_ID',
    how='inner'
)

# Merge with Functional dataset
merged_df = pd.merge(
    merged_df,
    func_features,
    on='Sub_ID',
    how='inner'
)

print("Datasets Merged Successfully")
print()

# ============================================================
# STEP 5: HANDLE MISSING VALUES
# ============================================================

# Check missing values
print("Missing Values Before Cleaning:")
print(merged_df.isnull().sum())
print()

# Remove rows with missing values
merged_df = merged_df.dropna()

print("Missing Values Removed")
print()

# ============================================================
# STEP 6: NORMALIZE FEATURES
# ============================================================

# Separate features and subject IDs
subject_ids = merged_df['Sub_ID']

# Drop Sub_ID before normalization
feature_data = merged_df.drop(columns=['Sub_ID'])

# Initialize scaler
scaler = MinMaxScaler()

# Normalize features
normalized_features = scaler.fit_transform(feature_data)

# Convert back to DataFrame
normalized_df = pd.DataFrame(
    normalized_features,
    columns=feature_data.columns
)

# Add Sub_ID back
normalized_df.insert(0, 'Sub_ID', subject_ids.values)

print("Feature Normalization Completed")
print()

# ============================================================
# STEP 7: SAVE PREPROCESSED DATASET
# ============================================================

output_file = "preprocessed_features.csv"

normalized_df.to_csv(output_file, index=False)

print("Preprocessed Dataset Saved Successfully")
print(f"Output File: {output_file}")
print()

# ============================================================
# STEP 8: DISPLAY FINAL INFORMATION
# ============================================================

print("Final Dataset Shape:")
print(normalized_df.shape)
print()

print("First 5 Rows:")
print(normalized_df.head())
print()

print("Preprocessing Pipeline Completed Successfully")