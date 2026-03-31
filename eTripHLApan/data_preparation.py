#!/usr/bin/env python3
"""
Data preparation script for converting MHC binding CSV data to eTripHLApan format.

Input format (CSV):
    HLA.Class.I.Allele, Epitope...Peptide, Assay.Qualitative.Measurement, Expression, ...

Output format (for eTripHLApan training):
    Peptide\tHLA_Allele\tAffinity_Value\tBinary_Label
    NAKEFEKEI\tHLA-A*68:02\t122\t1
    
Binary Label mapping:
    0 = Negative (non-binder)
    1 = Positive (binder: "Positive", "Positive-High", "Positive-Intermediate", "Positive-Low")
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def prepare_data(csv_path, output_dir='../for_prediction/', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Convert MHC binding CSV data to eTripHLApan training format.
    
    Args:
        csv_path: Path to MHC_Binding_Dataset_Final.csv
        output_dir: Directory to save output files
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
    
    Returns:
        dict with statistics about the prepared data
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract relevant columns
    df_processed = df[['HLA.Class.I.Allele', 'Epitope...Peptide', 'Expression', 'Assay.Qualitative.Measurement']].copy()
    df_processed.columns = ['hla', 'peptide', 'affinity', 'binding_class']
    
    # Convert binding class to binary label
    # 0 = Negative, 1 = Any Positive
    df_processed['label'] = df_processed['binding_class'].apply(
        lambda x: 0 if x == 'Negative' else 1
    )
    
    # Remove rows with missing values
    initial_count = len(df_processed)
    df_processed = df_processed.dropna()
    print(f"Removed {initial_count - len(df_processed)} rows with missing values")
    
    # Verify peptide lengths (should be 8-15 for HLA-I, but allow some variation)
    print("\nPeptide length distribution:")
    print(df_processed['peptide'].str.len().describe())
    
    # Filter out very short or very long peptides (optional, adjust as needed)
    # TripHLApan typically handles peptides up to length 15, but can be flexible
    valid_lengths = df_processed['peptide'].str.len()
    df_processed = df_processed[(valid_lengths >= 8) & (valid_lengths <= 15)]
    print(f"After filtering by peptide length (8-15 aa): {len(df_processed)} samples")
    
    # Print label distribution
    print(f"\nLabel distribution:")
    print(f"  Negative (0): {(df_processed['label'] == 0).sum()}")
    print(f"  Positive (1): {(df_processed['label'] == 1).sum()}")
    
    # Split data: train -> validation -> test
    # First split: train + temp (validation + test)
    train_df, temp_df = train_test_split(
        df_processed, 
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=df_processed['label']
    )
    
    # Second split: validation and test from temp
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42,
        stratify=temp_df['label']
    )
    
    print(f"\nData split:")
    print(f"  Training:   {len(train_df)} samples ({len(train_df)/len(df_processed)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df_processed)*100:.1f}%)")
    print(f"  Testing:    {len(test_df)} samples ({len(test_df)/len(df_processed)*100:.1f}%)")
    
    # Function to save data in TripHLApan format
    def save_data(df, filename):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            for _, row in df.iterrows():
                f.write(f"{row['peptide']}\t{row['hla']}\t{row['affinity']}\t{row['label']}\n")
        print(f"  Saved: {filepath}")
        return filepath
    
    # Save splits
    print(f"\nSaving data to {output_dir}...")
    save_data(train_df, 'training_data.txt')
    save_data(val_df, 'validation_data.txt')
    save_data(test_df, 'test_data.txt')
    
    # Also save combined training+validation for cross-validation
    train_val_df = pd.concat([train_df, val_df])
    save_data(train_val_df, 'train_val_data.txt')
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, 'data_preparation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Data Preparation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source CSV: {csv_path}\n")
        f.write(f"Total samples processed: {len(df_processed)}\n\n")
        f.write("Label Distribution:\n")
        f.write(f"  Negative (0): {(df_processed['label'] == 0).sum()}\n")
        f.write(f"  Positive (1): {(df_processed['label'] == 1).sum()}\n\n")
        f.write("Data Split:\n")
        f.write(f"  Training:   {len(train_df)} samples\n")
        f.write(f"  Validation: {len(val_df)} samples\n")
        f.write(f"  Testing:    {len(test_df)} samples\n")
        f.write(f"  Train+Val:  {len(train_val_df)} samples\n\n")
        f.write("Format: Peptide\\tHLA\\tAffinity\\tLabel\n")
        f.write("Label: 0=Negative, 1=Positive\n")
    
    print(f"  Saved: {summary_path}")
    
    return {
        'total_samples': len(df_processed),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'negative_count': (df_processed['label'] == 0).sum(),
        'positive_count': (df_processed['label'] == 1).sum(),
    }


if __name__ == '__main__':
    # Path to your CSV file (update this to your local path)
    csv_path = 'MHC_Binding_Dataset_Final.csv'
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'for_prediction')
    
    # Prepare data with default splits (70% train, 15% val, 15% test)
    stats = prepare_data(csv_path, output_dir)
    
    print("\n" + "=" * 50)
    print("Data preparation completed successfully!")
    print("=" * 50)
    print(f"\nNext steps:")
    print(f"1. Review the prepared data in: {output_dir}")
    print(f"2. For independent testing: use test_data.txt")
    print(f"3. For training: modify eTripHLApan to use training_data.txt")
    print(f"4. Consider using pre-trained models with transfer learning")
