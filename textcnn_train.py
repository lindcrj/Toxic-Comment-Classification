"""
TextCNN model training script - Uses pretrained embeddings and class weights
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer


class TextCNNDataset(Dataset):
    """Dataset for TextCNN model"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class TextCNN(nn.Module):
    """Text CNN model for toxicity classification"""
    def __init__(self, vocab_size, embedding_dim, filter_sizes, num_filters, dropout, output_dim=1, embedding_matrix=None):
        super(TextCNN, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize with pretrained embeddings if provided
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, 
                      kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
    def forward(self, text, mask=None):
        # text: [batch_size, seq_len]
        
        # Apply embeddings: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(text)
        
        # Add channel dimension for CNN: [batch_size, 1, seq_len, embedding_dim]
        embedded = embedded.unsqueeze(1)
        
        # Apply convolutions and pooling
        # Each conv_i is [batch_size, num_filters, seq_len - filter_size + 1, 1]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        # Apply max pooling over time
        # Each pooled_i is [batch_size, num_filters, 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate the pooled features
        # cat is [batch_size, len(filter_sizes) * num_filters]
        cat = torch.cat(pooled, dim=1)
        
        # Apply dropout and the final linear layer
        # output is [batch_size, output_dim]
        output = self.fc(self.dropout(cat))
        
        return output.squeeze()


def train_epoch(model, dataloader, optimizer, device, class_weight=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(total=len(dataloader), desc="Training", leave=True)
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        predictions = model(input_ids, mask)
        
        # Apply class weights for positive examples
        if class_weight > 1:
            pos_weight = torch.tensor([class_weight], device=device)
            loss = F.binary_cross_entropy_with_logits(predictions, labels, pos_weight=pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    progress_bar.close()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, class_weight=1.0, desc="Evaluating"):
    """Evaluate model on a dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(total=len(dataloader), desc=desc, leave=True)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            predictions = model(input_ids, mask)
            
            # Apply sigmoid to get probability
            preds = torch.sigmoid(predictions).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Binarize predictions and labels
    binary_preds = (all_preds >= 0.5).astype(int)
    binary_labels = (all_labels >= 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(binary_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, average='binary', zero_division=0
    )
    
    # Calculate minority class F1 (toxic texts)
    pos_indices = np.where(binary_labels == 1)[0]
    if len(pos_indices) > 0:
        pos_preds = binary_preds[pos_indices]
        pos_labels = binary_labels[pos_indices]
        _, _, minority_f1, _ = precision_recall_fscore_support(
            pos_labels, pos_preds, average='binary', zero_division=0
        )
    else:
        minority_f1 = 0.0
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'minority_f1': minority_f1
    }


def main():
    parser = argparse.ArgumentParser(description='Train TextCNN for toxicity classification')
    parser.add_argument('--train-dir', type=str, required=True, help='Training data directory (s3://bucket/path)')
    parser.add_argument('--val-dir', type=str, required=True, help='Validation data directory (s3://bucket/path)')
    parser.add_argument('--test-dir', type=str, required=True, help='Test data directory (s3://bucket/path)')
    parser.add_argument('--model-dir', type=str, required=True, help='Model output directory (s3://bucket/path)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--max-length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--embedding-dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--filter-sizes', type=str, default='3,4,5', help='Comma-separated filter sizes')
    parser.add_argument('--num-filters', type=int, default=100, help='Number of filters per size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--use-class-weight', action='store_true', help='Use class weights')
    
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
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer (using DistilBERT tokenizer for efficiency)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TextCNNDataset(
        train_df['comment_text'].fillna(''),
        (train_df['target'] >= 0.5).astype(float),
        tokenizer,
        args.max_length
    )
    
    val_dataset = TextCNNDataset(
        val_df['comment_text'].fillna(''),
        (val_df['target'] >= 0.5).astype(float),
        tokenizer,
        args.max_length
    )
    
    test_dataset = TextCNNDataset(
        test_df['comment_text'].fillna(''),
        (test_df['target'] >= 0.5).astype(float),
        tokenizer,
        args.max_length
    )
    
    # Create data loaders
    print(f"Creating data loaders with batch size {args.batch_size}...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size
    )
    
    # Parse filter sizes
    filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
    
    # Initialize model
    print("Initializing TextCNN model...")
    vocab_size = len(tokenizer.vocab)
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        filter_sizes=filter_sizes,
        num_filters=args.num_filters,
        dropout=args.dropout
    ).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    
    start_time = time.time()
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            device,
            class_weight=pos_weight if args.use_class_weight else 1.0
        )
        
        print(f"Training loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        val_metrics = evaluate(
            model, 
            val_dataloader, 
            device,
            class_weight=pos_weight if args.use_class_weight else 1.0
        )
        
        print(f"Validation metrics:")
        print(f"  AUC: {val_metrics['auc']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
            print(f"New best model! Validation F1: {best_val_f1:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model
    print("Saving model...")
    model_path = f"{args.model_dir}/textcnn_model.pt"
    
    # For S3 path, save locally first then upload
    if model_path.startswith('s3://'):
        local_model_path = '/tmp/textcnn_model.pt'
        torch.save(model.state_dict(), local_model_path)
        
        with open(local_model_path, 'rb') as f:
            model_data = f.read()
            write_to_path(model_data, model_path, is_binary=True)
        
        os.remove(local_model_path)
    else:
        torch.save(model.state_dict(), model_path)
    
    # Save tokenizer configuration
    tokenizer_path = f"{args.model_dir}/tokenizer_config.json"
    tokenizer_config = tokenizer.save_pretrained('/tmp/tokenizer')
    
    # For S3 path, save tokenizer files
    if tokenizer_path.startswith('s3://'):
        for file_name in os.listdir('/tmp/tokenizer'):
            local_file_path = os.path.join('/tmp/tokenizer', file_name)
            s3_file_path = f"{args.model_dir}/{file_name}"
            
            with open(local_file_path, 'rb') as f:
                file_data = f.read()
                write_to_path(file_data, s3_file_path, is_binary=True)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate(
        model, 
        test_dataloader, 
        device,
        class_weight=pos_weight if args.use_class_weight else 1.0,
        desc="Testing"
    )
    
    print(f"Test metrics:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Minority F1: {test_metrics['minority_f1']:.4f}")
    
    # Save model configuration and results
    model_config = {
        'vocab_size': vocab_size,
        'embedding_dim': args.embedding_dim,
        'filter_sizes': filter_sizes,
        'num_filters': args.num_filters,
        'dropout': args.dropout
    }
    
    results = {
        'model': 'TextCNN',
        'parameters': {
            'embedding_dim': args.embedding_dim,
            'filter_sizes': filter_sizes,
            'num_filters': args.num_filters,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'use_class_weight': args.use_class_weight,
            'class_weight': float(pos_weight) if args.use_class_weight else None
        },
        'model_config': model_config,
        'training_time_seconds': training_time,
        'validation': {
            'auc': float(val_metrics['auc']),
            'precision': float(val_metrics['precision']),
            'recall': float(val_metrics['recall']),
            'f1': float(val_metrics['f1'])
        },
        'test': {
            'auc': float(test_metrics['auc']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'minority_f1': float(test_metrics['minority_f1'])
        }
    }
    
    results_path = f"{args.model_dir}/textcnn_results.json"
    write_to_path(json.dumps(results, indent=2), results_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()