"""
Data preprocessing script - Process CSV data and split into datasets
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import boto3
import s3fs

def main():
    parser = argparse.ArgumentParser(description='Process Jigsaw toxicity data')
    parser.add_argument('--input-path', type=str, required=True, help='S3 input path (s3://bucket/path)')
    parser.add_argument('--output-path', type=str, required=True, help='S3 output path (s3://bucket/path)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size ratio')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sample-size', type=float, default=1.0, 
                        help='Sample size ratio (0.0-1.0). Default uses all data.')
    
    args = parser.parse_args()
    
    # Parse S3 paths
    input_parts = args.input_path.replace('s3://', '').split('/', 1)
    output_parts = args.output_path.replace('s3://', '').split('/', 1)
    
    input_bucket = input_parts[0]
    input_key = input_parts[1] if len(input_parts) > 1 else ''
    
    output_bucket = output_parts[0]
    output_prefix = output_parts[1] if len(output_parts) > 1 else ''
    
    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem(anon=False)
    
    # Create output directories in S3 (these are virtual, just for organization)
    train_prefix = f"{output_prefix}/train"
    val_prefix = f"{output_prefix}/validation"
    test_prefix = f"{output_prefix}/test"
    
    # Read data directly from S3
    print(f"Reading data from {args.input_path}")
    with fs.open(args.input_path, 'rb') as f:
        df = pd.read_csv(f)
    print(f"Original data shape: {df.shape}")
    
    # Create a binary target column for stratification
    # This ensures we have clear binary classes rather than potentially many small classes
    df['target_binary'] = (df['target'] >= 0.5).astype(int)
    
    # Sample data if needed
    if args.sample_size < 1.0:
        # Stratified sampling to maintain class distribution
        df_positive = df[df['target_binary'] == 1]
        df_negative = df[df['target_binary'] == 0]
        
        # Calculate sample sizes, ensuring at least 2 samples per class
        pos_sample_size = max(int(len(df_positive) * args.sample_size), 2)
        neg_sample_size = max(int(len(df_negative) * args.sample_size), 2)
        
        # Check if we have enough samples
        if len(df_positive) < 2 or len(df_negative) < 2:
            print("Warning: Not enough samples in one or both classes for stratification.")
            print(f"Positive samples: {len(df_positive)}, Negative samples: {len(df_negative)}")
            print("Disabling stratification for sampling.")
            # Simple random sampling if not enough samples
            sample_size = max(int(len(df) * args.sample_size), 4)
            df = df.sample(sample_size, random_state=args.random_seed)
        else:
            # Proceed with stratified sampling
            df_positive_sampled = df_positive.sample(pos_sample_size, random_state=args.random_seed)
            df_negative_sampled = df_negative.sample(neg_sample_size, random_state=args.random_seed)
            df = pd.concat([df_positive_sampled, df_negative_sampled])
        
        print(f"Sampled data shape: {df.shape}")
        positive_count = df['target_binary'].sum()
        negative_count = len(df) - positive_count
        print(f"Positive samples: {positive_count}, Negative samples: {negative_count}")
    
    # Verify we have at least 2 samples per class for stratification
    class_counts = df['target_binary'].value_counts()
    min_samples_per_class = class_counts.min()
    
    # Split datasets
    if min_samples_per_class < 2:
        print(f"Warning: Minimum samples per class is {min_samples_per_class}, which is less than 2.")
        print("Disabling stratification for train-test split.")
        
        train_df, temp_df = train_test_split(
            df, 
            test_size=args.test_size + args.val_size,
            stratify=None,  # Disable stratification
            random_state=args.random_seed
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=args.test_size/(args.test_size + args.val_size),
            stratify=None,  # Disable stratification
            random_state=args.random_seed
        )
    else:
        print("Using stratified sampling for train-test split.")
        train_df, temp_df = train_test_split(
            df, 
            test_size=args.test_size + args.val_size,
            stratify=df['target_binary'],  # Use binary target for stratification
            random_state=args.random_seed
        )
        
        val_ratio = args.val_size / (args.test_size + args.val_size)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=args.test_size/(args.test_size + args.val_size),
            stratify=temp_df['target_binary'],  # Use binary target for stratification
            random_state=args.random_seed
        )
    
    # Calculate class weights
    n_negative = (train_df['target_binary'] == 0).sum()
    n_positive = (train_df['target_binary'] == 1).sum()
    
    # Total number of samples
    n_samples = n_negative + n_positive
    
    # Number of classes
    n_classes = 2  # Binary classification
    
    # Calculate class counts for np.bincount format
    y = train_df['target_binary'].values
    class_counts = np.array([n_negative, n_positive])
    
    # Calculate weights using the square root formula
    class_weights = np.sqrt(n_samples / (n_classes * class_counts))
    
    # If you need it as a dictionary (e.g., for sklearn)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # If you need the pos_weight for PyTorch BCE loss
    pos_weight = class_weights[1] / class_weights[0]
    
    weight_df = pd.DataFrame({
        'class': ['negative', 'positive'],
        'count': [n_negative, n_positive],
        'weight': [1.0, pos_weight]
    })
    
    # Save datasets directly to S3
    train_path = f"s3://{output_bucket}/{train_prefix}/train.csv"
    val_path = f"s3://{output_bucket}/{val_prefix}/validation.csv"
    test_path = f"s3://{output_bucket}/{test_prefix}/test.csv"
    weight_path = f"s3://{output_bucket}/{output_prefix}/class_weights.csv"
    
    # Drop the temporary target_binary column before saving if you don't need it
    if 'target_binary' in train_df.columns and 'target_binary' not in df.columns:
        train_df = train_df.drop('target_binary', axis=1)
        val_df = val_df.drop('target_binary', axis=1) 
        test_df = test_df.drop('target_binary', axis=1)
    
    with fs.open(train_path, 'w') as f:
        train_df.to_csv(f, index=False)
    
    with fs.open(val_path, 'w') as f:
        val_df.to_csv(f, index=False)
    
    with fs.open(test_path, 'w') as f:
        test_df.to_csv(f, index=False)
    
    with fs.open(weight_path, 'w') as f:
        weight_df.to_csv(f, index=False)
    
    # Print statistics
    print(f"Data split complete.")
    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Class weights: positive={pos_weight:.2f}, negative=1.00")
    print(f"Data saved to S3: {args.output_path}")

if __name__ == "__main__":
    main()