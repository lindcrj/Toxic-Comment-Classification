"""
TinyBERT model training script - Uses adversarial training and class weights
Direct S3 integration for data access and model storage with progress tracking
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
import json
import time
import s3fs
import io
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import sys
from tqdm.auto import tqdm  # Import tqdm for progress bars
sys.path.append('./scripts')
from spot_monitor import SpotInstanceMonitor

class ToxicityDataset(Dataset):
    """Dataset for toxicity classification"""
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

def fgsm_attack(model, loss, emb, epsilon=0.01):
    """
    Fast Gradient Sign Method for generating adversarial examples
    """
    # Calculate gradients
    loss.backward(retain_graph=True)
    
    # Get embedding layer gradients
    emb_grad = emb.grad.data
    
    # Create perturbation
    perturbed_emb = emb + epsilon * emb_grad.sign()
    
    return perturbed_emb

def save_checkpoint(model, optimizer, scheduler, epoch, step, path):
    """
    Save model checkpoint to file with progress tracking
    """
    print(f"Creating checkpoint at epoch {epoch}, step {step}...")
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    
    # Check if path is S3 path
    if path.startswith('s3://'):
        # Save locally first
        local_path = f"/tmp/checkpoint_epoch_{epoch}_step_{step}.pt"
        torch.save(checkpoint, local_path)
        
        # Upload to S3 with progress indication
        print(f"Uploading checkpoint to {path}...")
        fs = s3fs.S3FileSystem(anon=False)
        fs.put(local_path, path)
        
        # Clean up
        os.remove(local_path)
    else:
        # Create directory if doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
    
    print(f"Checkpoint saved successfully")

def train_epoch(model, dataloader, optimizer, scheduler, device, class_weight=1.0, 
                epsilon=0.01, use_adversarial=True, checkpoint_interval=100, 
                checkpoint_dir=None, epoch=0):
    """
    Train for one epoch with a single progress bar
    """
    model.train()
    total_loss = 0
    total_batches = len(dataloader)
    
    # Create a single progress bar for the entire epoch
    progress_bar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}", leave=True)
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Get embedding layer - for TinyBERT
        embeddings = model.bert.embeddings.word_embeddings(input_ids)
        embeddings.retain_grad()
        
        # Forward pass
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()
        
        # Calculate loss
        if class_weight > 1:
            pos_weight = torch.tensor([class_weight], device=device)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # If using adversarial training
        if use_adversarial:
            # Generate adversarial examples
            perturbed_emb = fgsm_attack(model, loss, embeddings, epsilon)
            
            # Recalculate loss with perturbed embeddings
            adv_outputs = model(inputs_embeds=perturbed_emb, attention_mask=attention_mask)
            adv_logits = adv_outputs.logits.squeeze()
            
            if class_weight > 1:
                adv_loss = F.binary_cross_entropy_with_logits(adv_logits, labels, pos_weight=pos_weight)
            else:
                adv_loss = F.binary_cross_entropy_with_logits(adv_logits, labels)
            
            # Total loss = original loss + adversarial loss
            total_batch_loss = loss + adv_loss
        else:
            total_batch_loss = loss
        
        # Backward pass
        total_batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer and scheduler step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        batch_loss = total_batch_loss.item()
        total_loss += batch_loss
        
        # Update the progress bar
        current_loss = total_loss / (step + 1)
        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}", 
            "batch": f"{step+1}/{total_batches}"
        })
        progress_bar.update(1)
        
        # Save checkpoint
        if checkpoint_dir and checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            global_step = epoch * len(dataloader) + step
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}_step_{global_step}.pt"
            # Save checkpoint without disrupting progress bar
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}", 
                "batch": f"{step+1}/{total_batches}",
                "saving": "checkpoint"
            })
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path)
            # Reset postfix after saving
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}", 
                "batch": f"{step+1}/{total_batches}"
            })
    
    # Close the progress bar
    progress_bar.close()
    
    return total_loss / total_batches

def evaluate(model, dataloader, device, class_weight=1.0, desc="Evaluating"):
    """
    Evaluate model on a dataset with a single clean progress bar
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_batches = len(dataloader)
    
    # Add a single progress bar for the entire evaluation
    progress_bar = tqdm(total=total_batches, desc=desc, leave=True)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds if isinstance(preds, np.ndarray) else [preds])
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with processed percentage
            progress_bar.update(1)
            progress_bar.set_postfix({
                "processed": f"{len(all_preds)}", 
                "batch": f"{progress_bar.n}/{total_batches}"
            })
    
    # Close the progress bar
    progress_bar.close()
    
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

def read_csv_from_path(path):
    """Read CSV file from local path or S3 with a simple progress indication"""
    print(f"Reading data from {path}...")
    
    if path.startswith('s3://'):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path, 'rb') as f:
            return pd.read_csv(f)
    else:
        return pd.read_csv(path)

def str2bool(v):
    """Convert string representations of boolean values to actual booleans"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Train TinyBERT for toxicity classification')
    parser.add_argument('--train-dir', type=str, required=True, help='Training data directory (s3://bucket/path or local)')
    parser.add_argument('--val-dir', type=str, required=True, help='Validation data directory (s3://bucket/path or local)')
    parser.add_argument('--test-dir', type=str, required=True, help='Test data directory (s3://bucket/path or local)')
    parser.add_argument('--model-dir', type=str, required=True, help='Model output directory (s3://bucket/path or local)')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory (s3://bucket/path or local)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--use-class-weight', type=str2bool, nargs='?', const=True, default=False, help='Use class weights')
    parser.add_argument('--use-adversarial', type=str2bool, nargs='?', const=True, default=False, help='Use adversarial training')
    parser.add_argument('--adversarial-epsilon', type=float, default=0.01, help='FGSM epsilon value')
    parser.add_argument('--checkpoint-interval', type=int, default=100, help='Steps between checkpoints (0 to disable)')
    parser.add_argument('--enable-spot-monitoring', type=str2bool, nargs='?', const=True, default=False, help='Enable spot instance monitoring')
    parser.add_argument('--use-spot', type=str2bool, nargs='?', const=True, default=False, help='Running on a spot instance')
    
    args = parser.parse_args()
    
    # Create output directories (for local paths)
    if not args.model_dir.startswith('s3://'):
        os.makedirs(args.model_dir, exist_ok=True)
    if args.checkpoint_dir and not args.checkpoint_dir.startswith('s3://'):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Read data with progress tracking
    print("\n=== Loading Datasets ===")
    train_path = f"{args.train_dir}/train.csv"
    val_path = f"{args.val_dir}/validation.csv"
    test_path = f"{args.test_dir}/test.csv"
    
    train_df = read_csv_from_path(train_path)
    val_df = read_csv_from_path(val_path)
    test_df = read_csv_from_path(test_path)
    print("All datasets loaded successfully")
    
    # Read class weights
    # Extract base directory from train_dir (parent directory)
    if args.train_dir.startswith('s3://'):
        parts = args.train_dir.replace('s3://', '').split('/')
        # Remove 'train' from path
        if parts[-1] == 'train':
            parts = parts[:-1]
        base_dir = f"s3://{'/'.join(parts)}"
    else:
        base_dir = os.path.dirname(args.train_dir.rstrip('/'))
    
    weights_path = f"{base_dir}/class_weights.csv"
    try:
        weights_df = read_csv_from_path(weights_path)
        pos_weight = weights_df[weights_df['class'] == 'positive']['weight'].values[0]
    except Exception as e:
        print(f"Error reading class weights, calculating from training data: {e}")
        # Calculate weights if file doesn't exist
        n_negative = (train_df['target'] < 0.5).sum()
        n_positive = (train_df['target'] >= 0.5).sum()
        pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    class_weight = pos_weight if args.use_class_weight else 1.0
    print(f"Class weight for positive class: {class_weight}")
    
    # Initialize tokenizer and model with progress indication
    print("\n=== Model Initialization ===")
    # Use TinyBERT 4-layer model
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1
    ).to(device)
    print("TinyBERT model initialized successfully")
    
    # Create datasets with progress indication
    print("\n=== Dataset Preparation ===")
    
    # Display dataset sizes
    print(f"Training examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    print(f"Test examples: {len(test_df)}")
    
    # Create datasets
    train_dataset = ToxicityDataset(
        train_df['comment_text'].fillna(''),
        (train_df['target'] >= 0.5).astype(float),
        tokenizer,
        args.max_length
    )
    
    val_dataset = ToxicityDataset(
        val_df['comment_text'].fillna(''),
        (val_df['target'] >= 0.5).astype(float),
        tokenizer,
        args.max_length
    )
    
    test_dataset = ToxicityDataset(
        test_df['comment_text'].fillna(''),
        (test_df['target'] >= 0.5).astype(float),
        tokenizer,
        args.max_length
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size
    )
    print(f"Created dataloaders with {len(train_dataloader)} training batches")
    
    # Initialize optimizer and learning rate scheduler
    print("\n=== Optimizer Setup ===")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Initialize Spot instance monitor if enabled
    spot_monitor = None
    if args.enable_spot_monitoring and args.use_spot:
        print("Initializing Spot instance monitoring...")
        try:
            spot_monitor = SpotInstanceMonitor(
                lambda: save_checkpoint(
                    model, optimizer, scheduler, 
                    -1, -1, 
                    f"{args.checkpoint_dir}/emergency_checkpoint.pt"
                )
            )
            spot_monitor.start()
            print("Spot monitoring started successfully")
        except Exception as e:
            print(f"Error starting spot monitoring: {e}")
    
    # Training loop with overall epoch progress bar
    start_time = time.time()
    best_val_auc = 0
    best_model_path = f"{args.model_dir}/best_model.pt"
    
    print("\n=== Starting Training ===")
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        
        # Train with a single progress bar per epoch
        train_loss = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            scheduler, 
            device,
            class_weight=class_weight,
            epsilon=args.adversarial_epsilon,
            use_adversarial=args.use_adversarial,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            epoch=epoch
        )
        
        print(f"Epoch {epoch+1} completed with average loss: {train_loss:.4f}")
        
        # Validate with a single progress bar
        print("\nValidating model...")
        val_metrics = evaluate(model, val_dataloader, device, class_weight, desc=f"Validation (Epoch {epoch+1})")
        
        print(f"Validation metrics:")
        print(f"  AUC: {val_metrics['auc']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        
        # Save best model with progress indication
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            print(f"New best model found! Saving model with validation AUC: {best_val_auc:.4f}")
            
            # Save to S3 or local
            if best_model_path.startswith('s3://'):
                local_path = '/tmp/best_model.pt'
                torch.save(model.state_dict(), local_path)
                
                # Upload to S3 with progress indication
                print(f"Uploading best model to S3...")
                fs = s3fs.S3FileSystem(anon=False)
                fs.put(local_path, best_model_path)
                
                # Clean up
                os.remove(local_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            
            print("Model saved successfully")
    
    training_time = time.time() - start_time
    print(f"\n=== Training Summary ===")
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    
    # Stop spot monitoring
    if spot_monitor:
        spot_monitor.stop()
    
    # Load best model for testing with progress indication
    print("\n=== Final Evaluation ===")
    if best_model_path.startswith('s3://'):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(best_model_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer, map_location=device)
            model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Test with a single progress bar
    test_metrics = evaluate(model, test_dataloader, device, class_weight, desc="Test Evaluation")
    
    print(f"\nTest metrics:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Minority F1: {test_metrics['minority_f1']:.4f}")
    
    # Save results with progress indication
    print("\n=== Saving Results ===")
    results = {
        'model': 'TinyBERT',
        'parameters': {
            'variant': '4L_312D',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'use_class_weight': args.use_class_weight,
            'class_weight': float(class_weight) if args.use_class_weight else 1.0,
            'use_adversarial': args.use_adversarial,
            'adversarial_epsilon': args.adversarial_epsilon if args.use_adversarial else None
        },
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
    
    results_path = f"{args.model_dir}/tinybert_results.json"
    
    # Save results to S3 or local with progress indication
    if results_path.startswith('s3://'):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save tokenizer config with progress indication
    print("\nSaving tokenizer configuration...")
    if args.model_dir.startswith('s3://'):
        # For S3, save locally first then upload
        local_tokenizer_dir = '/tmp/tokenizer'
        tokenizer.save_pretrained(local_tokenizer_dir)
        
        # Upload to S3 with progress bar
        print("Uploading tokenizer files to S3...")
        fs = s3fs.S3FileSystem(anon=False)
        tokenizer_files = os.listdir(local_tokenizer_dir)
        for file in tqdm(tokenizer_files, desc="Uploading tokenizer files"):
            local_file_path = os.path.join(local_tokenizer_dir, file)
            s3_file_path = f"{args.model_dir}/{file}"
            fs.put(local_file_path, s3_file_path)
            
        # Clean up
        import shutil
        shutil.rmtree(local_tokenizer_dir)
    else:
        tokenizer.save_pretrained(args.model_dir)
    
    print(f"\nModel and results saved to: {args.model_dir}")
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()