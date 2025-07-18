"""
SVM model training script - Uses TF-IDF features and class weights
Direct S3 integration for data access and model storage
"""
import argparse
import os
import pandas as pd
import numpy as np
import json
import pickle
import time
import s3fs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def main():
    parser = argparse.ArgumentParser(description='Train SVM for toxicity classification')
    parser.add_argument('--train-dir', type=str, required=True, help='Training data directory (s3://bucket/path)')
    parser.add_argument('--val-dir', type=str, required=True, help='Validation data directory (s3://bucket/path)')
    parser.add_argument('--test-dir', type=str, required=True, help='Test data directory (s3://bucket/path)')
    parser.add_argument('--model-dir', type=str, required=True, help='Model output directory (s3://bucket/path)')
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter')
    parser.add_argument('--max-iter', type=int, default=1000, help='Maximum iterations')
    parser.add_argument('--use-class-weight', action='store_true', help='Use class weights')
    parser.add_argument('--max-features', type=int, default=10000, help='Maximum number of features for TF-IDF')
    
    args = parser.parse_args()
    
    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem(anon=False)
    
    # Helper function to read CSV from S3 or local
    def read_csv(path):
        if path.startswith('s3://'):
            with fs.open(path, 'rb') as f:
                return pd.read_csv(f)
        else:
            return pd.read_csv(path)
    
    # Helper function to write data to S3 or local
    def write_to_path(data, path, mode='w', is_binary=False):
        if path.startswith('s3://'):
            mode = mode + 'b' if is_binary and 'b' not in mode else mode
            with fs.open(path, mode) as f:
                return f.write(data)
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            mode = mode + 'b' if is_binary and 'b' not in mode else mode
            with open(path, mode) as f:
                return f.write(data)
    
    # Read data from S3 or local
    print(f"Reading training data from {args.train_dir}/train.csv")
    train_path = f"{args.train_dir}/train.csv"
    val_path = f"{args.val_dir}/validation.csv"
    test_path = f"{args.test_dir}/test.csv"
    
    train_df = read_csv(train_path)
    val_df = read_csv(val_path)
    test_df = read_csv(test_path)
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Read class weights
    base_dir = args.train_dir.rsplit('/', 1)[0] if '/' in args.train_dir else ''
    weights_path = f"{base_dir}/class_weights.csv"
    
    try:
        weights_df = read_csv(weights_path)
        pos_weight = weights_df[weights_df['class'] == 'positive']['weight'].values[0]
    except Exception as e:
        print(f"Error reading class weights: {e}. Calculating from training data.")
        # Calculate weights if file doesn't exist
        n_negative = (train_df['target'] < 0.5).sum()
        n_positive = (train_df['target'] >= 0.5).sum()
        pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    print(f"Class weight for positive class: {pos_weight}")
    
    # Feature extraction
    start_time = time.time()
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    X_train = tfidf.fit_transform(train_df['comment_text'].fillna(''))
    X_val = tfidf.transform(val_df['comment_text'].fillna(''))
    X_test = tfidf.transform(test_df['comment_text'].fillna(''))
    
    y_train = (train_df['target'] >= 0.5).astype(int).values
    y_val = (val_df['target'] >= 0.5).astype(int).values
    y_test = (test_df['target'] >= 0.5).astype(int).values
    
    print(f"TF-IDF features extracted. Shape: {X_train.shape}")
    
    # Train model
    print("Training SVM model...")
    
    # Configure class weights
    class_weight = None
    if args.use_class_weight:
        class_weight = {0: 1.0, 1: pos_weight}
    
    # Create base SVM model
    base_model = LinearSVC(
        C=args.C,
        max_iter=args.max_iter,
        class_weight=class_weight,
        random_state=42
    )
    
    # Wrap with CalibratedClassifierCV to get probability estimates
    model = CalibratedClassifierCV(base_model, cv=3)
    
    # Train
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    model_path = f"{args.model_dir}/svm_model.pkl"
    vectorizer_path = f"{args.model_dir}/tfidf_vectorizer.pkl"
    
    # Save model
    model_data = pickle.dumps(model)
    write_to_path(model_data, model_path, is_binary=True)
    
    # Save vectorizer
    vectorizer_data = pickle.dumps(tfidf)
    write_to_path(vectorizer_data, vectorizer_path, is_binary=True)
    
    # Evaluate model
    print("Evaluating model...")
    # Validation set
    val_preds_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds_proba)
    val_binary_preds = (val_preds_proba >= 0.5).astype(int)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, val_binary_preds, average='binary', zero_division=0
    )
    
    # Test set
    test_preds_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds_proba)
    test_binary_preds = (test_preds_proba >= 0.5).astype(int)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, test_binary_preds, average='binary', zero_division=0
    )
    
    # Calculate minority class F1
    test_pos_indices = np.where(y_test == 1)[0]
    test_pos_preds = test_binary_preds[test_pos_indices]
    test_pos_true = y_test[test_pos_indices]
    minority_precision, minority_recall, test_minority_f1, _ = precision_recall_fscore_support(
        test_pos_true, test_pos_preds, average='binary', zero_division=0
    )
    
    # Save results
    results = {
        'model': 'SVM',
        'parameters': {
            'C': args.C,
            'max_iter': args.max_iter,
            'use_class_weight': args.use_class_weight,
            'max_features': args.max_features,
            'class_weight': float(pos_weight) if args.use_class_weight else None
        },
        'training_time_seconds': training_time,
        'validation': {
            'auc': float(val_auc),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1': float(val_f1)
        },
        'test': {
            'auc': float(test_auc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1),
            'minority_f1': float(test_minority_f1)
        }
    }
    
    results_path = f"{args.model_dir}/svm_results.json"
    write_to_path(json.dumps(results, indent=2), results_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")
    print(f"Validation AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}, F1: {test_f1:.4f}, Minority F1: {test_minority_f1:.4f}")

if __name__ == "__main__":
    main()