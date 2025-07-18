#!/usr/bin/env python
"""
Enhanced model evaluation script for toxicity classification
Focuses on imbalanced classification metrics with improved visualizations
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import time
import s3fs
import io
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set the aesthetic style for plots
plt.style.use('seaborn-v0_8-whitegrid') 
sns.set_palette("viridis")
custom_colors = ['#4287f5', '#f542a7', '#42f5b3', '#f5a742', '#8e42f5']
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("", custom_colors)

def is_s3_path(path):
    """Check if a path is an S3 path"""
    return path.startswith('s3://')

def read_json_from_path(path):
    """Read JSON from a file on S3 or local path"""
    if is_s3_path(path):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path.replace('s3://', ''), 'r') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)

def save_to_path(data, path, is_json=False, is_binary=False):
    """Save data to a file on S3 or local path"""
    mode = 'wb' if is_binary else 'w'
    
    if is_s3_path(path):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path.replace('s3://', ''), mode) as f:
            if is_json:
                json.dump(data, f, indent=2, ensure_ascii=False)  # Added ensure_ascii=False for better Unicode handling
            elif is_binary:
                f.write(data)
            else:
                f.write(data)
    else:
        # Create directory if doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, mode) as f:
            if is_json:
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif is_binary:
                f.write(data)
            else:
                f.write(data)

def list_files(path, pattern=''):
    """List files in a directory on S3 or local path"""
    if is_s3_path(path):
        fs = s3fs.S3FileSystem(anon=False)
        files = []
        
        # Strip 's3://' prefix for fs.ls
        s3_path = path.replace('s3://', '')
        
        try:
            for file_path in fs.ls(s3_path):
                # Add back the 's3://' prefix
                full_path = f"s3://{file_path}"
                if pattern in full_path and not fs.isdir(file_path):
                    files.append(full_path)
        except FileNotFoundError:
            print(f"Path not found: {path}")
            return []
                
        return files
    else:
        import glob
        return glob.glob(os.path.join(path, f"*{pattern}*"))

def load_results(result_files):
    """
    Load and parse model result files from S3 or local path
    """
    results = {}
    for file_path in result_files:
        try:
            # Check if file exists
            if is_s3_path(file_path):
                fs = s3fs.S3FileSystem(anon=False)
                file_exists = fs.exists(file_path.replace('s3://', ''))
            else:
                file_exists = os.path.exists(file_path)
                
            if file_exists:
                model_results = read_json_from_path(file_path)
                model_name = model_results.get('model', os.path.basename(file_path).split('_')[0])
                results[model_name] = model_results
                print(f"Loaded results for {model_name}")
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error loading results from {file_path}: {e}")
    
    return results

def create_comparison_table(results):
    """
    Create a comparison table of model metrics
    """
    rows = []
    for model_name, result in results.items():
        row = {
            'Model': model_name,
            'Training Time (s)': result.get('training_time_seconds', '-'),
            'Test AUC': result.get('test', {}).get('auc', '-'),
            'Test F1': result.get('test', {}).get('f1', '-'),
            'Test Precision': result.get('test', {}).get('precision', '-'),
            'Test Recall': result.get('test', {}).get('recall', '-'),
            'Minority F1': result.get('test', {}).get('minority_f1', '-'),
            'Class Weight': result.get('parameters', {}).get('use_class_weight', False),
            'Adversarial': result.get('parameters', {}).get('use_adversarial', False)
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def plot_radar_chart(results, output_dir):
    """
    Create a radar chart comparing key metrics across models
    """
    # Prepare the data
    models = list(results.keys())
    metrics = ['AUC', 'F1', 'Precision', 'Recall', 'Minority F1']
    
    # Create data for the radar chart
    data = []
    for model in models:
        model_data = [
            results[model].get('test', {}).get('auc', 0),
            results[model].get('test', {}).get('f1', 0),
            results[model].get('test', {}).get('precision', 0),
            results[model].get('test', {}).get('recall', 0),
            results[model].get('test', {}).get('minority_f1', 0)
        ]
        data.append(model_data)
    
    # Number of variables
    categories = metrics
    N = len(categories)
    
    # Create the angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.figure(figsize=(10, 10)), plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw the y-axis labels (0.2, 0.4, 0.6, 0.8, 1.0)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, model in enumerate(models):
        color = custom_colors[i % len(custom_colors)]
        
        values = data[i]
        values += values[:1]  # Close the loop
        
        # Plot the model data
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=color, label=model)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title('Model Performance Comparison - Radar Chart', size=15, y=1.1)
    
    # Save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path = f"{output_dir}/radar_comparison.png"
    save_to_path(buf.getvalue(), plot_path, is_binary=True)
    
    plt.close()
    
    return plot_path

def plot_transformer_comparison(results, output_dir):
    """
    Generate specific comparison between transformer models
    """
    transformer_models = ['DeBERTa', 'DistilBERT', 'TinyBERT']
    available_transformers = [model for model in transformer_models if model in results]
    
    if len(available_transformers) < 2:
        print("Not enough transformer models available for comparison")
        return []
    
    # Extract metrics
    metrics = ['auc', 'f1', 'precision', 'recall', 'minority_f1']
    model_data = {}
    
    for model in available_transformers:
        model_data[model] = {
            metric: results[model].get('test', {}).get(metric, 0) 
            for metric in metrics
        }
        model_data[model]['training_time'] = results[model].get('training_time_seconds', 0)
    
    # Create bar chart comparing metrics
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.8 / len(available_transformers)
    
    # Calculate starting positions for bars
    offsets = np.linspace(-0.35, 0.35, len(available_transformers))
    
    for i, model in enumerate(available_transformers):
        values = [model_data[model][metric] for metric in metrics]
        plt.bar(x + offsets[i], values, width, label=model)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Transformer Models Performance Comparison')
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics])
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path = f"{output_dir}/transformer_comparison.png"
    save_to_path(buf.getvalue(), plot_path, is_binary=True)
    
    plt.close()
    
    # Plot training time comparison
    plt.figure(figsize=(10, 6))
    times = [model_data[model]['training_time'] for model in available_transformers]
    
    # If there's a significant difference, use log scale
    if max(times) / (min(times) + 0.1) > 10:
        plt.bar(available_transformers, np.log10(times), color='skyblue')
        plt.ylabel('Log10(Training Time in seconds)')
    else:
        plt.bar(available_transformers, times, color='skyblue')
        plt.ylabel('Training Time (seconds)')
    
    plt.xlabel('Model')
    plt.title('Training Time Comparison: Transformer Models')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add actual value annotations
    for i, time_val in enumerate(times):
        plt.text(i, np.log10(time_val) if max(times) / (min(times) + 0.1) > 10 else time_val, 
                 f"{time_val:.1f}s", ha='center', va='bottom')
    
    # Save figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path_time = f"{output_dir}/transformer_training_time.png"
    save_to_path(buf.getvalue(), plot_path_time, is_binary=True)
    
    plt.close()
    
    # Create bubble chart showing efficiency (Minority F1 / Training Time)
    plt.figure(figsize=(10, 8))
    
    minority_f1s = [model_data[model]['minority_f1'] for model in available_transformers]
    
    # Calculate efficiency (normalize to avoid extreme values)
    efficiencies = [f1 / (time + 0.1) * 1000 for f1, time in zip(minority_f1s, times)]
    
    # Plot bubbles
    plt.scatter(minority_f1s, times, s=[e * 100 for e in efficiencies], alpha=0.5, 
               color=custom_colors[:len(available_transformers)])
    
    # Add model labels
    for i, model in enumerate(available_transformers):
        plt.annotate(model, xy=(minority_f1s[i], times[i]), 
                    xytext=(5, 0), textcoords='offset points')
    
    plt.xlabel('Minority Class F1')
    plt.ylabel('Training Time (s)')
    plt.title('Efficiency of Transformer Models')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # If there's a significant difference in training time, use log scale
    if max(times) / (min(times) + 0.1) > 10:
        plt.yscale('log')
    
    # Annotate with efficiency value
    for i, (f1, time, eff) in enumerate(zip(minority_f1s, times, efficiencies)):
        plt.text(f1, time, f"\nEff: {eff:.1f}", ha='center', fontsize=8)
    
    # Save figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path_efficiency = f"{output_dir}/transformer_efficiency.png"
    save_to_path(buf.getvalue(), plot_path_efficiency, is_binary=True)
    
    plt.close()
    
    return [plot_path, plot_path_time, plot_path_efficiency]

def plot_impact_of_strategies(results, output_dir):
    """
    Analyze and visualize impact of class weighting and adversarial training
    """
    # Determine which models use which strategies
    models_by_strategy = {
        'base': [],
        'class_weight': [],
        'adversarial': [],
        'both': []
    }
    
    for model, data in results.items():
        params = data.get('parameters', {})
        use_cw = params.get('use_class_weight', False)
        use_adv = params.get('use_adversarial', False)
        
        if use_cw and use_adv:
            models_by_strategy['both'].append(model)
        elif use_cw:
            models_by_strategy['class_weight'].append(model)
        elif use_adv:
            models_by_strategy['adversarial'].append(model)
        else:
            models_by_strategy['base'].append(model)
    
    # Skip if we don't have enough different strategies
    strategies_with_models = [s for s, models in models_by_strategy.items() if models]
    if len(strategies_with_models) < 2:
        print("Not enough different strategies to compare")
        return []
    
    # Calculate average metrics by strategy
    strategy_metrics = {}
    for strategy, model_list in models_by_strategy.items():
        if not model_list:
            continue
            
        auc_values = [results[m].get('test', {}).get('auc', 0) for m in model_list]
        f1_values = [results[m].get('test', {}).get('f1', 0) for m in model_list]
        minority_f1_values = [results[m].get('test', {}).get('minority_f1', 0) for m in model_list]
        
        strategy_metrics[strategy] = {
            'auc': np.mean(auc_values),
            'f1': np.mean(f1_values),
            'minority_f1': np.mean(minority_f1_values),
            'model_count': len(model_list)
        }
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    strategies = list(strategy_metrics.keys())
    x = np.arange(len(strategies))
    width = 0.25
    
    plt.bar(x - width, [strategy_metrics[s]['auc'] for s in strategies], width, label='AUC')
    plt.bar(x, [strategy_metrics[s]['f1'] for s in strategies], width, label='F1')
    plt.bar(x + width, [strategy_metrics[s]['minority_f1'] for s in strategies], width, label='Minority F1')
    
    plt.xlabel('Training Strategy')
    plt.ylabel('Average Score')
    plt.title('Impact of Training Strategies')
    strategy_labels = [f"{s.replace('_', ' ').title()} (n={strategy_metrics[s]['model_count']})" for s in strategies]
    plt.xticks(x, strategy_labels)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path = f"{output_dir}/training_strategies_impact.png"
    save_to_path(buf.getvalue(), plot_path, is_binary=True)
    
    plt.close()
    
    return [plot_path]

def generate_detailed_report(results, comparison_df, output_dir):
    """
    Generate a detailed model comparison report in markdown format with improved formatting for toxicity classification
    """
    # Determine best models
    best_auc_model = comparison_df.loc[comparison_df['Test AUC'].idxmax()]['Model']
    best_f1_model = comparison_df.loc[comparison_df['Test F1'].idxmax()]['Model']
    best_minority_model = comparison_df.loc[comparison_df['Minority F1'].idxmax()]['Model']
    fastest_model = comparison_df.loc[comparison_df['Training Time (s)'].idxmin()]['Model']
    
    # Prepare the transformer comparison if applicable
    transformer_models = ['DeBERTa', 'DistilBERT', 'TinyBERT']
    transformer_comparison = ""
    
    available_transformers = [model for model in transformer_models if model in results]
    if len(available_transformers) >= 2:
        transformer_comparison = "## Transformer Models Comparison\n\n"
        transformer_comparison += "| Metric | " + " | ".join(available_transformers) + " |\n"
        transformer_comparison += "|--------|" + "---------".join(["|" for _ in available_transformers]) + "|\n"
        
        # Add metrics rows
        metrics = [
            ("AUC", "auc", 4),
            ("F1", "f1", 4),
            ("Minority F1", "minority_f1", 4),
            ("Precision", "precision", 4),
            ("Recall", "recall", 4),
            ("Training Time (s)", "training_time_seconds", 2)
        ]
        
        for metric_name, metric_key, decimals in metrics:
            row = f"| {metric_name} | "
            
            if metric_key == "training_time_seconds":
                # For time, we want the raw value
                values = [results[model].get(metric_key, 0) for model in available_transformers]
                row += " | ".join([f"{val:.{decimals}f}" for val in values])
            else:
                # For all other metrics, they're under the 'test' dictionary
                values = [results[model].get('test', {}).get(metric_key, 0) for model in available_transformers]
                row += " | ".join([f"{val:.{decimals}f}" for val in values])
            
            row += " |\n"
            transformer_comparison += row
        
        # Add efficiency metric (Minority F1 / Training Time)
        efficiencies = []
        for model in available_transformers:
            min_f1 = results[model].get('test', {}).get('minority_f1', 0)
            time_s = results[model].get('training_time_seconds', 0.1)
            efficiency = min_f1 / max(time_s, 0.1)  # Prevent division by zero
            efficiencies.append(efficiency)
        
        transformer_comparison += "| Efficiency (Minority F1/Time) | "
        transformer_comparison += " | ".join([f"{eff:.6f}" for eff in efficiencies])
        transformer_comparison += " |\n\n"
        
        # Add summary of best transformer model
        best_performance_idx = np.argmax([results[model].get('test', {}).get('minority_f1', 0) for model in available_transformers])
        best_efficiency_idx = np.argmax(efficiencies)
        
        transformer_comparison += f"**Best transformer for performance**: {available_transformers[best_performance_idx]}  \n"
        transformer_comparison += f"**Best transformer for efficiency**: {available_transformers[best_efficiency_idx]}  \n\n"
    
    # Prepare the report
    report = f"""# Enhanced Toxicity Classification Model Comparison Report
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This report compares the performance of {len(results)} models trained on the Jigsaw toxicity classification dataset with a focus on imbalanced data performance.

## Executive Summary

| Category | Best Model | Score |
|----------|------------|-------|
| Overall AUC | {best_auc_model} | {comparison_df.loc[comparison_df['Test AUC'].idxmax()]['Test AUC']:.4f} |
| Overall F1 | {best_f1_model} | {comparison_df.loc[comparison_df['Test F1'].idxmax()]['Test F1']:.4f} |
| Minority Class F1 | {best_minority_model} | {comparison_df.loc[comparison_df['Minority F1'].idxmax()]['Minority F1']:.4f} |
| Training Speed | {fastest_model} | {comparison_df.loc[comparison_df['Training Time (s)'].idxmin()]['Training Time (s)']:.2f}s |

## Performance on Imbalanced Data

The Jigsaw toxicity classification dataset exhibits class imbalance, which poses challenges for model training and evaluation. The following metrics focus specifically on how well models handle this imbalance:

1. **Minority F1 Score**: F1 score calculated specifically for the toxic class
2. **Precision-Recall Trade-off**: How well models balance precision and recall given the class imbalance
3. **AUC**: Area under the ROC curve, which measures model's ability to discriminate between classes

## Detailed Metrics

{comparison_df.to_markdown(index=False)}

## Impact of Training Strategies

### Class Weighting
Models using class weights: {', '.join([m for m, r in results.items() if r.get('parameters', {}).get('use_class_weight', False)])}

#### Effect on Performance:
- Average increase in Minority F1: {calculate_class_weight_impact(results, 'minority_f1'):.2f}%
- Average increase in Overall AUC: {calculate_class_weight_impact(results, 'auc'):.2f}%

### Adversarial Training
Models using adversarial training: {', '.join([m for m, r in results.items() if r.get('parameters', {}).get('use_adversarial', False)])}

{transformer_comparison}

## Model Efficiency Analysis

| Model | Training Time (s) | Minority F1 | Efficiency Ratio |
|-------|-------------------|-------------|------------------|
{generate_efficiency_table(results)}

## Recommendations

1. **Best model for imbalanced classification**: {best_minority_model}
2. **Best model for overall performance**: {best_auc_model}
3. **Best model for time-constrained deployment**: {fastest_model}

## Next Steps

1. Evaluate the best performing models on a larger validation set
2. Conduct inference time benchmarking for production deployment
3. Implement early stopping to optimize training time for neural models
4. Explore ensembling the best transformer model with the fastest traditional model
5. Add confidence calibration to improve prediction reliability

"""
    # Save the report
    report_path = f"{output_dir}/enhanced_detailed_report.md"
    save_to_path(report, report_path)
    
    return report_path

def calculate_class_weight_impact(results, metric_key):
    """
    Calculate the average percentage impact of using class weights
    for a given metric
    """
    # Group models by whether they use class weights
    weighted_models = []
    unweighted_models = []
    
    for model, data in results.items():
        if data.get('parameters', {}).get('use_class_weight', False):
            weighted_models.append(model)
        else:
            unweighted_models.append(model)
    
    # Skip if we don't have both weighted and unweighted models
    if not weighted_models or not unweighted_models:
        return 0.0
    
    # Calculate average metric values for each group
    weighted_values = [results[m].get('test', {}).get(metric_key, 0) for m in weighted_models]
    weighted_avg = sum(weighted_values) / len(weighted_values) if weighted_values else 0
    
    unweighted_values = [results[m].get('test', {}).get(metric_key, 0) for m in unweighted_models]
    unweighted_avg = sum(unweighted_values) / len(unweighted_values) if unweighted_values else 0
    
    # Calculate percentage difference
    if unweighted_avg > 0:
        return ((weighted_avg - unweighted_avg) / unweighted_avg) * 100
    else:
        return 0.0

def generate_efficiency_table(results):
    """
    Generate a markdown table showing efficiency metrics for each model
    """
    table_rows = []
    
    for model, data in results.items():
        training_time = data.get('training_time_seconds', 0)
        minority_f1 = data.get('test', {}).get('minority_f1', 0)
        
        # Calculate efficiency ratio
        efficiency = minority_f1 / max(training_time, 0.1)  # Avoid division by zero
        
        row = f"| {model} | {training_time:.2f} | {minority_f1:.4f} | {efficiency:.6f} |"
        table_rows.append((model, efficiency, row))
    
    # Sort by efficiency (descending)
    table_rows.sort(key=lambda x: x[1], reverse=True)
    
    # Return the table content
    return "\n".join([row for _, _, row in table_rows])

def plot_model_dashboard(results, comparison_df, output_dir):
    """
    Create a comprehensive dashboard of model performance
    """
    models = list(results.keys())
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(4, 4, figure=fig)
    
    # Set up subplots
    ax_radar = fig.add_subplot(gs[0:2, 0:2], polar=True)
    ax_imbalance = fig.add_subplot(gs[0:2, 2:4])
    ax_pr = fig.add_subplot(gs[2:3, 0:2])
    ax_time = fig.add_subplot(gs[2:3, 2:4])
    ax_table = fig.add_subplot(gs[3:4, :])
    
    # 1. Radar Chart in first subplot
    categories = ['AUC', 'F1', 'Precision', 'Recall', 'Minority F1']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    
    # Draw the y-axis labels
    ax_radar.set_rlabel_position(0)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8)
    ax_radar.set_ylim(0, 1)
    
    # Plot each model on radar
    for i, model in enumerate(models):
        color = custom_colors[i % len(custom_colors)]
        
        # Get model data
        values = [
            results[model].get('test', {}).get('auc', 0),
            results[model].get('test', {}).get('f1', 0),
            results[model].get('test', {}).get('precision', 0),
            results[model].get('test', {}).get('recall', 0),
            results[model].get('test', {}).get('minority_f1', 0)
        ]
        values += values[:1]  # Close the loop
        
        # Plot model data on radar
        ax_radar.plot(angles, values, linewidth=2, linestyle='solid', color=color, label=model)
        ax_radar.fill(angles, values, color=color, alpha=0.1)
    
    ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax_radar.set_title('Model Performance Metrics', pad=20)
    
    # 2. Imbalance Performance in second subplot
    overall_f1 = [results[m].get('test', {}).get('f1', 0) for m in models]
    minority_f1 = [results[m].get('test', {}).get('minority_f1', 0) for m in models]
    f1_gaps = [minority - overall for minority, overall in zip(minority_f1, overall_f1)]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax_imbalance.bar(x - width/2, overall_f1, width, color='#4287f5', label='Overall F1')
    ax_imbalance.bar(x + width/2, minority_f1, width, color='#f542a7', label='Minority F1')
    
    for i, (gap, minority) in enumerate(zip(f1_gaps, minority_f1)):
        ax_imbalance.annotate(f'{gap:+.2f}',
                    xy=(i, max(minority_f1[i], overall_f1[i])), 
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    color='green' if gap > 0 else 'red')
    
    ax_imbalance.set_xlabel('Models')
    ax_imbalance.set_ylabel('Score')
    ax_imbalance.set_title('Performance on Imbalanced Data')
    ax_imbalance.set_xticks(x)
    ax_imbalance.set_xticklabels(models)
    ax_imbalance.legend()
    ax_imbalance.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 3. Precision-Recall Tradeoff in third subplot
    precisions = [results[m].get('test', {}).get('precision', 0) for m in models]
    recalls = [results[m].get('test', {}).get('recall', 0) for m in models]
    f1_scores = [results[m].get('test', {}).get('f1', 0) for m in models]
    
    # Calculate F1 curve points
    f1_curve_x = np.linspace(0.01, 1, 100)
    for f1 in [0.3, 0.5, 0.7]:
        curve_y = [(f1 * x) / (2 * x - f1) if (2 * x - f1) != 0 else 1 for x in f1_curve_x]
        valid_idx = ~(np.isnan(curve_y) | np.isinf(curve_y) | (np.array(curve_y) < 0) | (np.array(curve_y) > 1))
        if any(valid_idx):
            ax_pr.plot(f1_curve_x[valid_idx], np.array(curve_y)[valid_idx], '--', color='gray', alpha=0.5, linewidth=1)
            # Add label for each F1 curve
            idx = len(np.array(curve_y)[valid_idx]) // 2
            if idx < sum(valid_idx):
                point_idx = np.where(valid_idx)[0][idx]
                ax_pr.annotate(f'F1={f1}', xy=(f1_curve_x[point_idx], curve_y[point_idx]), 
                            xytext=(0, 0), textcoords='offset points',
                            color='gray', fontsize=7)
    
    # Plot model points on PR chart
    for i, model in enumerate(models):
        ax_pr.scatter(recalls[i], precisions[i], s=80, 
                    color=custom_colors[i % len(custom_colors)], 
                    label=f"{model}")
        
        # Add annotation with F1 score
        ax_pr.annotate(f"F1={f1_scores[i]:.2f}", xy=(recalls[i], precisions[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)
    
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Tradeoff')
    ax_pr.grid(True, linestyle='--', alpha=0.5)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.legend(loc='lower left', fontsize=8)
    
    # 4. Training Time vs Performance in fourth subplot
    training_times = [results[m].get('training_time_seconds', 0) for m in models]
    minority_f1s = [results[m].get('test', {}).get('minority_f1', 0) for m in models]
    sizes = [f1 * 300 for f1 in f1_scores]
    
    # Plot each model as bubble
    for i, model in enumerate(models):
        # Determine color based on model type
        if any(term in model.lower() for term in ['bert', 'gpt', 'transformer']):
            color = '#4287f5'  # Transformer
            model_type = 'Transformer'
        elif any(term in model.lower() for term in ['lstm', 'rnn', 'nn']):
            color = '#f542a7'  # Neural
            model_type = 'Neural Network'
        else:
            color = '#42f5b3'  # Traditional
            model_type = 'Traditional ML'
            
        ax_time.scatter(np.log10(max(training_times[i], 0.1)), minority_f1s[i], 
                     s=sizes[i], alpha=0.7, color=color, edgecolor='black')
        
        # Add model name annotation
        ax_time.annotate(model, xy=(np.log10(max(training_times[i], 0.1)), minority_f1s[i]),
                      xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add model type legend
    type_handles = [
        mpatches.Patch(color='#4287f5', label='Transformer'),
        mpatches.Patch(color='#f542a7', label='Neural Network'),
        mpatches.Patch(color='#42f5b3', label='Traditional ML')
    ]
    ax_time.legend(handles=type_handles, title="Model Type", loc='upper right', fontsize=8)
    
    # Configure axes
    ax_time.set_xlabel('Log10(Training Time in seconds)')
    ax_time.set_ylabel('Minority Class F1 Score')
    ax_time.set_title('Performance vs. Training Time')
    ax_time.grid(True, linestyle='--', alpha=0.5)
    
    # Add annotation about bubble size
    ax_time.annotate("Bubble size = F1 score", xy=(0.05, 0.05), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # 5. Summary table in the bottom subplot
    ax_table.axis('off')
    
    # Create a summary table of key metrics
    table_data = []
    
    # Table header
    table_data.append(['Model', 'AUC', 'F1', 'Minority F1', 'Time (s)', 'Class Weight'])
    
    # Add model data rows
    for model in models:
        row = [
            model,
            f"{results[model].get('test', {}).get('auc', 0):.4f}",
            f"{results[model].get('test', {}).get('f1', 0):.4f}",
            f"{results[model].get('test', {}).get('minority_f1', 0):.4f}",
            f"{results[model].get('training_time_seconds', 0):.2f}",
            "Yes" if results[model].get('parameters', {}).get('use_class_weight', False) else "No"
        ]
        table_data.append(row)
    
    # Create the table
    table = ax_table.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Adjust table size
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(weight='bold')
    
    # Highlight the best model for each metric
    best_auc_idx = np.argmax([results[m].get('test', {}).get('auc', 0) for m in models])
    best_f1_idx = np.argmax([results[m].get('test', {}).get('f1', 0) for m in models])
    best_minority_idx = np.argmax([results[m].get('test', {}).get('minority_f1', 0) for m in models])
    fastest_idx = np.argmin([results[m].get('training_time_seconds', float('inf')) for m in models])
    
    # Highlight the best metrics
    table[(best_auc_idx + 1, 1)].set_facecolor('#d2f8d2')
    table[(best_f1_idx + 1, 2)].set_facecolor('#d2f8d2')
    table[(best_minority_idx + 1, 3)].set_facecolor('#d2f8d2')
    table[(fastest_idx + 1, 4)].set_facecolor('#d2f8d2')
    
    # Add title to the dashboard
    plt.suptitle('Model Performance Dashboard', fontsize=20, y=0.98)
    
    # Add a timestamp
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    plt.figtext(0.5, 0.01, f"Generated on: {timestamp}", ha='center', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path = f"{output_dir}/model_dashboard.png"
    save_to_path(buf.getvalue(), plot_path, is_binary=True)
    
    plt.close()
    
    return plot_path

def plot_imbalance_performance(results, output_dir):
    """
    Create a specialized visualization focusing on model performance with imbalanced data
    """
    models = list(results.keys())
    
    # Extract metrics relevant to imbalanced classification
    overall_f1 = [results[m].get('test', {}).get('f1', 0) for m in models]
    minority_f1 = [results[m].get('test', {}).get('minority_f1', 0) for m in models]
    
    # Calculate the gap between overall F1 and minority class F1
    f1_gaps = [minority - overall for minority, overall in zip(minority_f1, overall_f1)]
    
    # Find imbalance ratio from parameters if available, default to 10
    imbalance_ratios = []
    for m in models:
        if 'parameters' in results[m] and 'class_weight' in results[m]['parameters']:
            imbalance_ratios.append(results[m]['parameters']['class_weight'])
        else:
            # Default to 10:1 imbalance if not specified
            imbalance_ratios.append(10)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart for Overall F1 vs Minority F1
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, overall_f1, width, color='#4287f5', label='Overall F1')
    bars2 = plt.bar(x + width/2, minority_f1, width, color='#f542a7', label='Minority Class F1')
    
    # Add annotation showing the gap
    for i, (gap, minority) in enumerate(zip(f1_gaps, minority_f1)):
        if gap > 0:  # Minority F1 > Overall F1
            plt.annotate(f'+{gap:.2f}', 
                        xy=(i, minority), 
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        color='green')
        else:  # Minority F1 <= Overall F1
            plt.annotate(f'{gap:.2f}', 
                        xy=(i, minority), 
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        color='red')
    
    # Add imbalance ratio information in subtitle
    ratio_info = ', '.join([f"{m}: {r:.1f}:1" for m, r in zip(models, imbalance_ratios)])
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance on Imbalanced Data\n', fontsize=14)
    plt.figtext(0.5, 0.01, f"Class Imbalance Ratios: {ratio_info}", 
                ha='center', fontsize=10, style='italic')
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal lines for benchmarks
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(len(models)-1, 0.51, 'F1 = 0.5 (Benchmark)', fontsize=8, alpha=0.7)
    
    # Save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path = f"{output_dir}/imbalance_performance.png"
    save_to_path(buf.getvalue(), plot_path, is_binary=True)
    
    plt.close()
    
    return plot_path

def plot_precision_recall_tradeoff(results, output_dir):
    """
    Create a visualization showing precision-recall tradeoff for all models
    """
    models = list(results.keys())
    
    # Extract precision and recall
    precisions = [results[m].get('test', {}).get('precision', 0) for m in models]
    recalls = [results[m].get('test', {}).get('recall', 0) for m in models]
    f1_scores = [results[m].get('test', {}).get('f1', 0) for m in models]
    
    # Calculate F1 curve points for visualization
    f1_curve_x = np.linspace(0.01, 1, 100)
    f1_curves = {}
    
    for f1 in [0.2, 0.4, 0.6, 0.8]:
        f1_curves[f1] = [(f1 * x) / (2 * x - f1) if (2 * x - f1) != 0 else 1 for x in f1_curve_x]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot F1 curves
    for f1, curve in f1_curves.items():
        valid_idx = ~(np.isnan(curve) | np.isinf(curve) | (np.array(curve) < 0) | (np.array(curve) > 1))
        if any(valid_idx):
            plt.plot(f1_curve_x[valid_idx], np.array(curve)[valid_idx], '--', color='gray', alpha=0.5, linewidth=1)
            # Add label for each F1 curve
            idx = len(np.array(curve)[valid_idx]) // 2
            if idx < sum(valid_idx):
                point_idx = np.where(valid_idx)[0][idx]
                plt.annotate(f'F1={f1}', xy=(f1_curve_x[point_idx], curve[point_idx]), 
                           xytext=(0, 0), textcoords='offset points',
                           color='gray', fontsize=8)
    
    # Plot model points
    for i, model in enumerate(models):
        plt.scatter(recalls[i], precisions[i], s=100, 
                    color=custom_colors[i % len(custom_colors)], 
                    label=f"{model} (F1={f1_scores[i]:.3f})")
        
        # Add annotation with model name
        plt.annotate(model, xy=(recalls[i], precisions[i]), xytext=(5, 5),
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Tradeoff for Different Models')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='lower left')
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
                "Precision-Recall tradeoff is especially important for imbalanced datasets.\nCurved lines represent constant F1 scores.", 
                ha='center', fontsize=9, style='italic')
    
    # Save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path = f"{output_dir}/precision_recall_tradeoff.png"
    save_to_path(buf.getvalue(), plot_path, is_binary=True)
    
    plt.close()
    
    return plot_path

def plot_class_weight_impact(results, output_dir):
    """
    Visualize the impact of class weighting on model performance
    """
    # Group models by whether they use class weights
    weighted_models = []
    unweighted_models = []
    
    for model, data in results.items():
        if data.get('parameters', {}).get('use_class_weight', False):
            weighted_models.append(model)
        else:
            unweighted_models.append(model)
    
    # Skip if we don't have both weighted and unweighted models
    if not weighted_models or not unweighted_models:
        print("Cannot generate class weight impact plot - need both weighted and unweighted models")
        return None
    
    # Prepare metrics for comparison
    metrics = ['AUC', 'F1', 'Minority F1', 'Precision', 'Recall']
    metric_keys = ['auc', 'f1', 'minority_f1', 'precision', 'recall']
    
    # Calculate average metrics for each group
    weighted_avgs = []
    unweighted_avgs = []
    
    for key in metric_keys:
        # Calculate weighted average
        weighted_values = [results[m].get('test', {}).get(key, 0) for m in weighted_models]
        weighted_avg = sum(weighted_values) / len(weighted_values) if weighted_values else 0
        weighted_avgs.append(weighted_avg)
        
        # Calculate unweighted average
        unweighted_values = [results[m].get('test', {}).get(key, 0) for m in unweighted_models]
        unweighted_avg = sum(unweighted_values) / len(unweighted_values) if unweighted_values else 0
        unweighted_avgs.append(unweighted_avg)
    
    # Calculate percentage differences
    pct_diffs = [(w - u) / max(u, 0.0001) * 100 for w, u in zip(weighted_avgs, unweighted_avgs)]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
    
    # First subplot: Bar chart comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(metrics))
    width = 0.35
    
    weighted_bars = ax1.bar(x - width/2, weighted_avgs, width, label=f'With Class Weights (n={len(weighted_models)})')
    unweighted_bars = ax1.bar(x + width/2, unweighted_avgs, width, label=f'Without Class Weights (n={len(unweighted_models)})')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Impact of Class Weighting on Model Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i, v in enumerate(weighted_avgs):
        ax1.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    for i, v in enumerate(unweighted_avgs):
        ax1.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # Second subplot: Percentage difference
    ax2 = fig.add_subplot(gs[1, :])
    
    bars = ax2.bar(x, pct_diffs, color=['green' if p > 0 else 'red' for p in pct_diffs])
    
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('% Difference')
    ax2.set_title('Percentage Improvement from Class Weighting')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i, v in enumerate(pct_diffs):
        ax2.text(i, v + np.sign(v) * 2, f'{v:+.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path = f"{output_dir}/class_weight_impact.png"
    save_to_path(buf.getvalue(), plot_path, is_binary=True)
    
    plt.close()
    
    return plot_path

def plot_performance_vs_time(results, output_dir):
    """
    Create a visualization showing model performance vs training time
    """
    models = list(results.keys())
    
    # Extract metrics
    training_times = [results[m].get('training_time_seconds', 0) for m in models]
    aucs = [results[m].get('test', {}).get('auc', 0) for m in models]
    f1_scores = [results[m].get('test', {}).get('f1', 0) for m in models]
    minority_f1s = [results[m].get('test', {}).get('minority_f1', 0) for m in models]
    
    # Create bubble size based on overall F1 score
    sizes = [f1 * 500 for f1 in f1_scores]
    
    # Create a categorical assignment for each model as traditional, neural, or transformer
    # based on name patterns
    model_types = []
    for model in models:
        if any(term in model.lower() for term in ['bert', 'gpt', 'transformer']):
            model_types.append('Transformer')
        elif any(term in model.lower() for term in ['lstm', 'rnn', 'nn']):
            model_types.append('Neural Network')
        else:
            model_types.append('Traditional ML')
    
    # Create a mapping of model types to colors
    type_colors = {
        'Transformer': '#4287f5',
        'Neural Network': '#f542a7',
        'Traditional ML': '#42f5b3'
    }
    
    colors = [type_colors[t] for t in model_types]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot performances
    for i, model in enumerate(models):
        # Plot bubble
        plt.scatter(np.log10(max(training_times[i], 0.1)), minority_f1s[i], 
                   s=sizes[i], alpha=0.6, color=colors[i], edgecolor='black')
        
        # Add model name
        plt.annotate(model, xy=(np.log10(max(training_times[i], 0.1)), minority_f1s[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    # Add legend for model types
    type_handles = [mpatches.Patch(color=color, label=label) for label, color in type_colors.items()]
    plt.legend(handles=type_handles, title="Model Type")
    
    # Add explanation for bubble size
    plt.figtext(0.15, 0.01, "Bubble size represents overall F1 score", ha='left', fontsize=9)
    
    plt.xlabel('Log10(Training Time in seconds)')
    plt.ylabel('Minority Class F1 Score')
    plt.title('Model Performance vs. Training Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a note on the top-right corner explaining the ideal position
    plt.annotate("Ideal models are\ntop-left (fast and accurate)", 
                xy=(0.05, 0.95), xycoords='figure fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    # Save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path = f"{output_dir}/performance_vs_time.png"
    save_to_path(buf.getvalue(), plot_path, is_binary=True)
    
    plt.close()
    
    return plot_path

def plot_imbalanced_focus_dashboard(results, output_dir):
    """
    Create a dedicated dashboard focusing on imbalanced dataset performance
    """
    models = list(results.keys())
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig)
    
    # Set up subplots
    ax_pr_curve = fig.add_subplot(gs[0, 0])
    ax_roc_curve = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])
    
    # 1. Precision-Recall Curves (simulated)
    # (Since we don't have actual PR curves, we'll simulate based on the summary metrics)
    for i, model in enumerate(models):
        precision = results[model].get('test', {}).get('precision', 0)
        recall = results[model].get('test', {}).get('recall', 0)
        f1 = results[model].get('test', {}).get('f1', 0)
        
        # Create simulated PR curve points
        x = np.linspace(0, 1, 100)
        # Create a curve that passes through the model's precision-recall point
        beta = 1.5 if precision > recall else 0.75  # Control curve shape
        y = precision * (1 - (1 - (x/max(recall, 0.001))**beta) ** (1/beta)) if recall > 0 else np.zeros_like(x)
        
        # Plot the curve
        ax_pr_curve.plot(x, y, '-', color=custom_colors[i % len(custom_colors)], 
                        alpha=0.7, linewidth=2, label=model)
        
        # Mark the operating point
        ax_pr_curve.scatter([recall], [precision], s=80, 
                          color=custom_colors[i % len(custom_colors)], edgecolor='black')
        
        # Add F1 annotation
        ax_pr_curve.annotate(f"F1={f1:.3f}", xy=(recall, precision), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)
    
    # Add PR curve details
    ax_pr_curve.set_xlabel('Recall')
    ax_pr_curve.set_ylabel('Precision')
    ax_pr_curve.set_title('Precision-Recall Curves\n(Simulated from metrics)')
    ax_pr_curve.grid(True, linestyle='--', alpha=0.5)
    ax_pr_curve.set_xlim(0, 1)
    ax_pr_curve.set_ylim(0, 1)
    ax_pr_curve.legend(loc='lower left')
    
    # Draw F1 isocurves
    f1_values = [0.2, 0.4, 0.6, 0.8]
    x_f1 = np.linspace(0.01, 1, 100)
    for f1_val in f1_values:
        y_f1 = (f1_val * x_f1) / (2 * x_f1 - f1_val)
        valid_idx = ~(np.isnan(y_f1) | np.isinf(y_f1) | (y_f1 < 0) | (y_f1 > 1))
        if any(valid_idx):
            ax_pr_curve.plot(x_f1[valid_idx], y_f1[valid_idx], '--', color='gray', alpha=0.5)
            # Find a good position for the label
            idx = min(80, sum(valid_idx)-1) if any(valid_idx) else -1
            if idx >= 0:
                point_idx = np.where(valid_idx)[0][idx]
                ax_pr_curve.annotate(f'F1={f1_val}', 
                                  xy=(x_f1[point_idx], y_f1[point_idx]),
                                  color='gray', fontsize=7)
    
    # 2. ROC Curves (simulated)
    for i, model in enumerate(models):
        auc = results[model].get('test', {}).get('auc', 0)
        
        # Create simulated ROC curve
        # Using a simple parametric function to create a curve with the given AUC
        x = np.linspace(0, 1, 100)
        y = x ** (1/max(auc, 0.01))
        y = y / np.max(y)
        
        # Plot the curve
        ax_roc_curve.plot(x, y, '-', color=custom_colors[i % len(custom_colors)], 
                         alpha=0.7, linewidth=2, label=f"{model} (AUC={auc:.3f})")
    
    # Add ROC curve details
    ax_roc_curve.plot([0, 1], [0, 1], '--', color='gray')  # Random classifier line
    ax_roc_curve.set_xlabel('False Positive Rate')
    ax_roc_curve.set_ylabel('True Positive Rate')
    ax_roc_curve.set_title('ROC Curves\n(Simulated from AUC)')
    ax_roc_curve.grid(True, linestyle='--', alpha=0.5)
    ax_roc_curve.set_xlim(0, 1)
    ax_roc_curve.set_ylim(0, 1)
    ax_roc_curve.legend(loc='lower right')
    
    # 3. Imbalance Performance Comparison
    # Extract relevant metrics
    overall_f1 = [results[m].get('test', {}).get('f1', 0) for m in models]
    minority_f1 = [results[m].get('test', {}).get('minority_f1', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, overall_f1, width, color='#4287f5', label='Overall F1')
    bars2 = ax_bar.bar(x + width/2, minority_f1, width, color='#f542a7', label='Minority F1')
    
    # Calculate the difference
    f1_diff = [m - o for m, o in zip(minority_f1, overall_f1)]
    
    # Add annotations for differences
    for i, diff in enumerate(f1_diff):
        color = 'green' if diff >= 0 else 'red'
        ax_bar.annotate(f'{diff:+.2f}', 
                      xy=(i, max(minority_f1[i], overall_f1[i])), 
                      xytext=(0, 5),
                      textcoords='offset points',
                      ha='center', fontsize=9, color=color)
    
    ax_bar.set_xlabel('Models')
    ax_bar.set_ylabel('F1 Score')
    ax_bar.set_title('Overall vs. Minority Class F1 Scores')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models)
    ax_bar.legend()
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 4. Summary metrics table with focus on imbalance
    ax_table.axis('off')
    
    # Create table data
    table_data = []
    
    # Header row
    table_data.append(['Model', 'Minority F1', 'Overall F1', 'Diff', 'Precision', 'Recall'])
    
    # Model data rows
    for i, model in enumerate(models):
        min_f1 = results[model].get('test', {}).get('minority_f1', 0)
        f1 = results[model].get('test', {}).get('f1', 0)
        prec = results[model].get('test', {}).get('precision', 0)
        rec = results[model].get('test', {}).get('recall', 0)
        diff = min_f1 - f1
        
        diff_str = f"{diff:+.4f}"
        
        row = [model, f"{min_f1:.4f}", f"{f1:.4f}", diff_str, f"{prec:.4f}", f"{rec:.4f}"]
        table_data.append(row)
    
    # Create the table
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(weight='bold')
    
    # Style the diff column
    for i in range(1, len(models) + 1):
        cell = table[(i, 3)]
        diff_val = float(table_data[i][3])
        if diff_val > 0:
            cell.set_facecolor('#d2f8d2')  # Light green
        else:
            cell.set_facecolor('#f8d2d2')  # Light red
    
    # Highlight best in each metric
    best_min_f1_idx = np.argmax([results[m].get('test', {}).get('minority_f1', 0) for m in models])
    best_f1_idx = np.argmax([results[m].get('test', {}).get('f1', 0) for m in models])
    best_prec_idx = np.argmax([results[m].get('test', {}).get('precision', 0) for m in models])
    best_rec_idx = np.argmax([results[m].get('test', {}).get('recall', 0) for m in models])
    
    table[(best_min_f1_idx + 1, 1)].set_facecolor('#d2e9ff')
    table[(best_f1_idx + 1, 2)].set_facecolor('#d2e9ff')
    table[(best_prec_idx + 1, 4)].set_facecolor('#d2e9ff')
    table[(best_rec_idx + 1, 5)].set_facecolor('#d2e9ff')
    
    # Add title to the dashboard
    plt.suptitle('Imbalanced Classification Performance Dashboard', fontsize=18, y=0.98)
    
    # Add a subtitle with explanation
    plt.figtext(0.5, 0.94, 
               "Focus on minority class performance and imbalanced metrics",
               ha='center', fontsize=12, fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    plot_path = f"{output_dir}/imbalanced_focus_dashboard.png"
    save_to_path(buf.getvalue(), plot_path, is_binary=True)
    
    plt.close()
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(description='Enhanced model evaluation for toxicity classification')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory containing model results (s3://bucket/path or local)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for evaluation results (s3://bucket/path or local)')
    parser.add_argument('--focus', choices=['general', 'imbalanced', 'all'], default='all', 
                       help='Focus of visualizations: general, imbalanced, or all')
    
    args = parser.parse_args()
    
    # Create output directory if it's local
    if not is_s3_path(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting enhanced model evaluation...")
    print(f"Searching for model results in: {args.results_dir}")
    print(f"Results will be saved to: {args.output_dir}")
    print(f"Visualization focus: {args.focus}")
    
    # Find result files
    result_files = []
    
    # Check if we're working with only specific models
    available_models = ['svm', 'xgboost', 'bilstm', 'tinybert', 'tinybert_2','deberta', 'distilbert']

    # Find results files paths based on model names
    for model_subdir in available_models:
        model_dir = f"{args.results_dir}/{model_subdir}" if args.results_dir.endswith('/') else f"{args.results_dir}/{model_subdir}"
        # For S3 handling
        if is_s3_path(args.results_dir):
            # Direct path to result file
            result_file = f"{model_dir}/{model_subdir}_results.json"
            result_files.append(result_file)
        else:
            # For local path
            try:
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith('_results.json'):
                            result_files.append(os.path.join(root, file))
            except FileNotFoundError:
                print(f"Directory not found: {model_dir}")
    
    print(f"Found {len(result_files)} result files: {result_files}")
    
    # Load results
    results = load_results(result_files)
    
    if not results:
        print("No valid model results found. Please check paths and file formats.")
        return
    
    print(f"Successfully loaded results for models: {', '.join(results.keys())}")
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    table_path = f"{args.output_dir}/model_comparison.csv"
    
    # Save CSV to S3 or local
    if is_s3_path(table_path):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(table_path.replace('s3://', ''), 'w') as f:
            comparison_df.to_csv(f, index=False)
    else:
        comparison_df.to_csv(table_path, index=False)
        
    print(f"Comparison table saved to: {table_path}")
    
    # Initialize plot paths list
    plot_paths = []
    
    # Create standard plots for all focus types
    if args.focus in ['general', 'all']:
        print("Generating general performance visualizations...")
        
        # Basic radar chart
        radar_path = plot_radar_chart(results, args.output_dir)
        plot_paths.append(radar_path)
        print(f"Radar chart saved to: {radar_path}")
        
        # Performance vs time
        time_path = plot_performance_vs_time(results, args.output_dir)
        plot_paths.append(time_path)
        print(f"Performance vs time plot saved to: {time_path}")
        
        # Model dashboard
        dashboard_path = plot_model_dashboard(results, comparison_df, args.output_dir)
        plot_paths.append(dashboard_path)
        print(f"Model dashboard saved to: {dashboard_path}")
    
    # Create imbalance-focused plots
    if args.focus in ['imbalanced', 'all']:
        print("Generating imbalanced data performance visualizations...")
        
        # Imbalance performance
        imbalance_path = plot_imbalance_performance(results, args.output_dir)
        plot_paths.append(imbalance_path)
        print(f"Imbalance performance plot saved to: {imbalance_path}")
        
        # Precision-recall tradeoff
        pr_path = plot_precision_recall_tradeoff(results, args.output_dir)
        plot_paths.append(pr_path)
        print(f"Precision-recall tradeoff plot saved to: {pr_path}")
        
        # Class weight impact if applicable
        class_weight_path = plot_class_weight_impact(results, args.output_dir)
        if class_weight_path:
            plot_paths.append(class_weight_path)
            print(f"Class weight impact plot saved to: {class_weight_path}")
        
        # Imbalanced focus dashboard
        imbalanced_dashboard_path = plot_imbalanced_focus_dashboard(results, args.output_dir)
        plot_paths.append(imbalanced_dashboard_path)
        print(f"Imbalanced focus dashboard saved to: {imbalanced_dashboard_path}")
    
    # Create transformer model comparison if applicable
    transformer_models = ['DeBERTa', 'DistilBERT', 'TinyBERT']
    available_transformers = [model for model in transformer_models if model in results]
    
    if len(available_transformers) >= 2:
        print(f"Found {len(available_transformers)} transformer models, generating transformer-specific comparisons...")
        transformer_plots = plot_transformer_comparison(results, args.output_dir)
        plot_paths.extend(transformer_plots)
        print(f"Transformer comparison plots saved.")
    
    # Create training strategies impact plot if applicable
    strategy_plots = plot_impact_of_strategies(results, args.output_dir)
    if strategy_plots:
        plot_paths.extend(strategy_plots)
        print(f"Training strategies impact plots saved.")
    
    # Generate detailed report
    report_path = generate_detailed_report(results, comparison_df, args.output_dir)
    print(f"Enhanced detailed report saved to: {report_path}")
    
    # Generate summary report
    report = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "models_evaluated": list(results.keys()),
        "best_model_by_auc": comparison_df.loc[comparison_df['Test AUC'].idxmax()]['Model'],
        "best_model_by_f1": comparison_df.loc[comparison_df['Test F1'].idxmax()]['Model'],
        "best_model_by_minority_f1": comparison_df.loc[comparison_df['Minority F1'].idxmax()]['Model'],
        "fastest_model": comparison_df.loc[comparison_df['Training Time (s)'].idxmin()]['Model'],
        "comparison_table": table_path,
        "plots": plot_paths,
        "detailed_report": report_path
    }
    
    # Save evaluation summary with extra error handling
    summary_path = f"{args.output_dir}/evaluation_summary.json"
    backup_summary_path = f"{args.output_dir}/evaluation_summary_backup.json"
    
    try:
        # Save the main summary
        save_to_path(report, summary_path, is_json=True)
        print(f"Evaluation summary saved to: {summary_path}")
    except Exception as e:
        print(f"WARNING: Error saving evaluation summary: {e}")
        print("Creating backup summary...")
        save_backup_json(report, backup_summary_path)
    
    # Print final results
    print("\n========== EVALUATION RESULTS ==========")
    print(f"Total models evaluated: {len(results)}")
    print(f"Best model by AUC: {report['best_model_by_auc']}")
    print(f"Best model by F1: {report['best_model_by_f1']}")
    print(f"Best model by Minority F1: {report['best_model_by_minority_f1']}")
    print(f"Fastest model: {report['fastest_model']}")
    
    # Print model rankings
    print("\nModel Rankings:")
    for metric in ['Test AUC', 'Test F1', 'Minority F1']:
        print(f"\nBy {metric}:")
        ranked_df = comparison_df.sort_values(metric, ascending=False)
        for i, (_, row) in enumerate(ranked_df.iterrows()):
            print(f"  {i+1}. {row['Model']}: {row[metric]:.4f}")
    
    print("\nEvaluation complete!")
    
if __name__ == "__main__":
    main()