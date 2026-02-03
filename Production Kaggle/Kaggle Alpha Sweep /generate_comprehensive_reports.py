"""
Alpha Sweep Results - Comprehensive Report Generator
======================================================
Optimized for MacBook Pro M3 - Analyzes summary files and computes all metrics

Features:
- Reads summary JSON files from alpha_sweep_results folder
- Computes ROUGE metrics using evaluate library
- Computes BERTScore with MPS (Metal) acceleration
- Generates comprehensive reports and visualizations
- Handles missing alphas gracefully
- Optimized for Apple Silicon

Author: Research Experiment
Date: 2026-02-03
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
RESULTS_DIR = '/Users/mrarnav69/Documents/TreeSum/Production Kaggle/Kaggle Alpha Sweep /alpha_sweep_results'  # Folder with summaries_alpha_X.X.json files
OUTPUT_DIR = '/Users/mrarnav69/Documents/TreeSum/Production Kaggle/Kaggle Alpha Sweep /alpha_sweep_reports'    # Where to save reports

# Metrics Configuration
COMPUTE_ROUGE = True
COMPUTE_BERTSCORE = True  # Set to False to skip (faster, but less complete)

# BERTScore Configuration (for Mac M3)
BERTSCORE_MODEL = "roberta-large"  # Good balance of speed/quality
BERTSCORE_BATCH_SIZE = 16          # Optimized for M3 Pro
USE_MPS = True                     # Use Metal Performance Shaders on Mac

# Expected alpha values
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print("="*80)
print("ALPHA SWEEP COMPREHENSIVE REPORT GENERATOR")
print("="*80)
print(f"Input Directory:  {RESULTS_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"ROUGE Enabled:    {COMPUTE_ROUGE}")
print(f"BERTScore Enabled: {COMPUTE_BERTSCORE}")
if COMPUTE_BERTSCORE:
    print(f"BERTScore Model:  {BERTSCORE_MODEL}")
    print(f"Use MPS (Metal):  {USE_MPS}")
print("="*80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DEPENDENCY CHECK & INSTALLATION
# ============================================================================

def check_and_install_dependencies():
    """Check and install required packages"""
    required_packages = {
        'evaluate': 'evaluate',
        'rouge_score': 'rouge-score',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    if COMPUTE_BERTSCORE:
        required_packages['bert_score'] = 'bert-score'
        required_packages['torch'] = 'torch'
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Installing...")
        import subprocess
        import sys
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print("✅ All dependencies installed!\n")
    else:
        print("\n✅ All dependencies available\n")

check_and_install_dependencies()

# Import after installation
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

if COMPUTE_BERTSCORE:
    import torch
    from bert_score import score as bert_score_fn

# ============================================================================
# SECTION 1: DISCOVER AND LOAD SUMMARIES
# ============================================================================

def discover_summary_files(results_dir: str) -> Dict[float, str]:
    """Find all available summary JSON files"""
    print("="*80)
    print("DISCOVERING SUMMARY FILES")
    print("="*80)
    
    available_files = {}
    
    for alpha in ALPHA_VALUES:
        # Try different naming patterns
        patterns = [
            f"summaries_alpha_{alpha:.1f}.json",
            f"summaries_alpha_{str(alpha).replace('.', '_')}.json",
        ]
        
        for pattern in patterns:
            filepath = os.path.join(results_dir, pattern)
            if os.path.exists(filepath):
                available_files[alpha] = filepath
                file_size = os.path.getsize(filepath) / 1024  # KB
                print(f"✅ Found: {pattern:30s} ({file_size:.1f} KB)")
                break
    
    missing_alphas = set(ALPHA_VALUES) - set(available_files.keys())
    if missing_alphas:
        print(f"\n⚠️  Missing alphas: {sorted(missing_alphas)}")
    
    print(f"\n📊 Total: {len(available_files)}/{len(ALPHA_VALUES)} alpha values found")
    
    if len(available_files) == 0:
        print(f"\n❌ ERROR: No summary files found in {results_dir}")
        print("   Expected files like: summaries_alpha_0.0.json")
        exit(1)
    
    return available_files

def load_summaries(filepath: str) -> List[Dict]:
    """Load and validate summary JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        if not isinstance(data, list):
            raise ValueError("Expected list of summaries")
        
        if len(data) == 0:
            raise ValueError("Empty summaries file")
        
        # Check required fields
        required_fields = {'sample_id', 'generated_summary', 'reference_summary'}
        if not required_fields.issubset(data[0].keys()):
            raise ValueError(f"Missing required fields: {required_fields - set(data[0].keys())}")
        
        return data
    
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return []

# ============================================================================
# SECTION 2: COMPUTE METRICS
# ============================================================================

def compute_rouge_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores"""
    print("  Computing ROUGE metrics...")
    
    rouge = evaluate.load('rouge')
    
    # Filter out empty predictions/references
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    
    if len(valid_pairs) == 0:
        print("  ⚠️  No valid prediction-reference pairs")
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'rougeLsum': 0.0
        }
    
    valid_predictions = [p for p, r in valid_pairs]
    valid_references = [r for p, r in valid_pairs]
    
    scores = rouge.compute(
        predictions=valid_predictions,
        references=valid_references,
        use_stemmer=True
    )
    
    # Convert to percentages
    result = {k: v * 100 for k, v in scores.items()}
    
    print(f"  ✅ ROUGE-1: {result['rouge1']:.2f}, ROUGE-2: {result['rouge2']:.2f}, ROUGE-L: {result['rougeL']:.2f}")
    
    return result

def compute_bertscore_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BERTScore using bert-score library (optimized for Mac M3)"""
    print(f"  Computing BERTScore (model: {BERTSCORE_MODEL})...")
    
    # Filter valid pairs
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    
    if len(valid_pairs) == 0:
        print("  ⚠️  No valid prediction-reference pairs")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }
    
    valid_predictions = [p for p, r in valid_pairs]
    valid_references = [r for p, r in valid_pairs]
    
    # Determine device
    if USE_MPS and torch.backends.mps.is_available():
        device = 'mps'
        print(f"  Using MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"  Using CUDA acceleration")
    else:
        device = 'cpu'
        print(f"  Using CPU (this may be slow)")
    
    # Compute BERTScore with batching
    P, R, F1 = bert_score_fn(
        valid_predictions,
        valid_references,
        model_type=BERTSCORE_MODEL,
        device=device,
        batch_size=BERTSCORE_BATCH_SIZE,
        lang='en',
        verbose=False
    )
    
    # Average scores
    result = {
        'bertscore_precision': P.mean().item() * 100,
        'bertscore_recall': R.mean().item() * 100,
        'bertscore_f1': F1.mean().item() * 100
    }
    
    print(f"  ✅ BERTScore F1: {result['bertscore_f1']:.2f}")
    
    return result

def analyze_alpha(alpha: float, filepath: str) -> Optional[Dict]:
    """Analyze summaries for a single alpha value"""
    print(f"\n{'='*80}")
    print(f"ANALYZING ALPHA = {alpha:.1f}")
    print(f"{'='*80}")
    
    # Load summaries
    summaries = load_summaries(filepath)
    if not summaries:
        return None
    
    num_samples = len(summaries)
    print(f"Loaded {num_samples} summaries")
    
    # Extract predictions and references
    predictions = [s['generated_summary'] for s in summaries]
    references = [s['reference_summary'] for s in summaries]
    sample_ids = [s['sample_id'] for s in summaries]
    
    # Check for empty summaries
    empty_predictions = sum(1 for p in predictions if not p.strip())
    if empty_predictions > 0:
        print(f"⚠️  Warning: {empty_predictions} empty predictions found")
    
    # Compute metrics
    results = {
        'alpha': alpha,
        'num_samples': num_samples,
        'sample_ids': sample_ids
    }
    
    if COMPUTE_ROUGE:
        rouge_scores = compute_rouge_metrics(predictions, references)
        results.update(rouge_scores)
    
    if COMPUTE_BERTSCORE:
        bertscore_scores = compute_bertscore_metrics(predictions, references)
        results.update(bertscore_scores)
    
    print(f"✅ Analysis complete for alpha={alpha:.1f}")
    
    return results

# ============================================================================
# SECTION 3: GENERATE REPORTS
# ============================================================================

def create_summary_table(all_results: List[Dict]) -> pd.DataFrame:
    """Create consolidated results table"""
    df = pd.DataFrame(all_results)
    df = df.set_index('alpha')
    
    # Remove sample_ids from display (keep for validation)
    display_columns = [c for c in df.columns if c != 'sample_ids']
    
    return df[display_columns]

def save_results_csv(df: pd.DataFrame, output_dir: str):
    """Save results in multiple CSV formats"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Main results file
    main_path = os.path.join(output_dir, 'alpha_sweep_results.csv')
    df.to_csv(main_path)
    print(f"✅ Saved: {main_path}")
    
    # ROUGE only
    if COMPUTE_ROUGE:
        rouge_cols = [c for c in df.columns if 'rouge' in c.lower()]
        if rouge_cols:
            rouge_path = os.path.join(output_dir, 'rouge_results.csv')
            df[rouge_cols].to_csv(rouge_path)
            print(f"✅ Saved: {rouge_path}")
    
    # BERTScore only
    if COMPUTE_BERTSCORE:
        bert_cols = [c for c in df.columns if 'bertscore' in c.lower()]
        if bert_cols:
            bert_path = os.path.join(output_dir, 'bertscore_results.csv')
            df[bert_cols].to_csv(bert_path)
            print(f"✅ Saved: {bert_path}")
    
    # JSON format
    json_path = os.path.join(output_dir, 'alpha_sweep_results.json')
    df.to_json(json_path, indent=2, orient='index')
    print(f"✅ Saved: {json_path}")

def generate_text_report(df: pd.DataFrame, all_results: List[Dict], output_dir: str):
    """Generate comprehensive text report"""
    print("Generating text report...")
    
    report_lines = []
    
    # Header
    report_lines.append("="*80)
    report_lines.append("ALPHA SWEEP ABLATION STUDY - FINAL RESULTS")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Number of Alpha Values: {len(df)}")
    report_lines.append(f"Alpha Values Tested: {', '.join([f'{a:.1f}' for a in df.index])}")
    
    # Check for consistent sample sizes
    sample_sizes = [r['num_samples'] for r in all_results]
    if len(set(sample_sizes)) == 1:
        report_lines.append(f"Samples per Alpha: {sample_sizes[0]}")
    else:
        report_lines.append(f"⚠️  Inconsistent sample sizes: {sample_sizes}")
    
    # ROUGE Results
    if COMPUTE_ROUGE:
        report_lines.append("\n" + "="*80)
        report_lines.append("ROUGE SCORES")
        report_lines.append("="*80)
        
        rouge_cols = [c for c in df.columns if 'rouge' in c.lower()]
        report_lines.append("\n" + df[rouge_cols].to_string())
        
        # Best alphas
        report_lines.append("\n" + "-"*80)
        report_lines.append("Best Alpha by Metric (ROUGE):")
        report_lines.append("-"*80)
        for col in rouge_cols:
            best_alpha = df[col].idxmax()
            best_score = df.loc[best_alpha, col]
            report_lines.append(f"{col:15s}: alpha={best_alpha:.1f} → {best_score:.2f}")
        
        # Statistics
        report_lines.append("\n" + "-"*80)
        report_lines.append("ROUGE Score Statistics:")
        report_lines.append("-"*80)
        for col in rouge_cols:
            report_lines.append(f"{col}:")
            report_lines.append(f"  Mean:   {df[col].mean():.2f}")
            report_lines.append(f"  Std:    {df[col].std():.2f}")
            report_lines.append(f"  Min:    {df[col].min():.2f} (alpha={df[col].idxmin():.1f})")
            report_lines.append(f"  Max:    {df[col].max():.2f} (alpha={df[col].idxmax():.1f})")
            report_lines.append(f"  Range:  {df[col].max() - df[col].min():.2f}")
    
    # BERTScore Results
    if COMPUTE_BERTSCORE:
        report_lines.append("\n" + "="*80)
        report_lines.append("BERTSCORE METRICS")
        report_lines.append("="*80)
        
        bert_cols = [c for c in df.columns if 'bertscore' in c.lower()]
        report_lines.append("\n" + df[bert_cols].to_string())
        
        # Best alphas
        report_lines.append("\n" + "-"*80)
        report_lines.append("Best Alpha by Metric (BERTScore):")
        report_lines.append("-"*80)
        for col in bert_cols:
            best_alpha = df[col].idxmax()
            best_score = df.loc[best_alpha, col]
            report_lines.append(f"{col:25s}: alpha={best_alpha:.1f} → {best_score:.2f}")
        
        # Statistics
        report_lines.append("\n" + "-"*80)
        report_lines.append("BERTScore Statistics:")
        report_lines.append("-"*80)
        for col in bert_cols:
            report_lines.append(f"{col}:")
            report_lines.append(f"  Mean:   {df[col].mean():.2f}")
            report_lines.append(f"  Std:    {df[col].std():.2f}")
            report_lines.append(f"  Min:    {df[col].min():.2f} (alpha={df[col].idxmin():.1f})")
            report_lines.append(f"  Max:    {df[col].max():.2f} (alpha={df[col].idxmax():.1f})")
            report_lines.append(f"  Range:  {df[col].max() - df[col].min():.2f}")
    
    # Overall Best Alpha
    report_lines.append("\n" + "="*80)
    report_lines.append("OVERALL BEST ALPHA")
    report_lines.append("="*80)
    
    if COMPUTE_ROUGE and 'rouge1' in df.columns:
        best_rouge1_alpha = df['rouge1'].idxmax()
        report_lines.append(f"Best by ROUGE-1: alpha={best_rouge1_alpha:.1f} (score: {df.loc[best_rouge1_alpha, 'rouge1']:.2f})")
    
    if COMPUTE_BERTSCORE and 'bertscore_f1' in df.columns:
        best_bert_alpha = df['bertscore_f1'].idxmax()
        report_lines.append(f"Best by BERTScore F1: alpha={best_bert_alpha:.1f} (score: {df.loc[best_bert_alpha, 'bertscore_f1']:.2f})")
    
    # Recommendations
    report_lines.append("\n" + "="*80)
    report_lines.append("RECOMMENDATIONS FOR PAPER")
    report_lines.append("="*80)
    
    if COMPUTE_ROUGE and 'rouge1' in df.columns:
        best_alpha = df['rouge1'].idxmax()
        report_lines.append(f"\nRecommended optimal alpha: {best_alpha:.1f}")
        report_lines.append(f"Performance at alpha={best_alpha:.1f}:")
        if COMPUTE_ROUGE:
            report_lines.append(f"  ROUGE-1: {df.loc[best_alpha, 'rouge1']:.2f}")
            report_lines.append(f"  ROUGE-2: {df.loc[best_alpha, 'rouge2']:.2f}")
            report_lines.append(f"  ROUGE-L: {df.loc[best_alpha, 'rougeL']:.2f}")
        if COMPUTE_BERTSCORE:
            report_lines.append(f"  BERTScore F1: {df.loc[best_alpha, 'bertscore_f1']:.2f}")
    
    report_lines.append("\n" + "="*80)
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, 'COMPREHENSIVE_REPORT.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ Saved: {report_path}")
    
    # Also print to console
    print("\n" + report_text)

def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualization plots"""
    print("\nGenerating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Determine what metrics are available
    has_rouge = COMPUTE_ROUGE and any('rouge' in c.lower() for c in df.columns)
    has_bertscore = COMPUTE_BERTSCORE and any('bertscore' in c.lower() for c in df.columns)
    
    if has_rouge and has_bertscore:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    elif has_rouge or has_bertscore:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = None
    else:
        print("⚠️  No metrics to visualize")
        return
    
    # Plot ROUGE
    if has_rouge:
        rouge_cols = ['rouge1', 'rouge2', 'rougeL']
        rouge_cols = [c for c in rouge_cols if c in df.columns]
        
        for col in rouge_cols:
            label = col.upper().replace('ROUGE', 'ROUGE-')
            ax1.plot(df.index, df[col], marker='o', linewidth=2, markersize=8, label=label)
        
        ax1.set_xlabel('Alpha (Semantic Weight)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ROUGE Score (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Alpha vs ROUGE Scores', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(df.index)
    
    # Plot BERTScore
    if has_bertscore and ax2 is not None:
        bert_cols = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
        bert_cols = [c for c in bert_cols if c in df.columns]
        
        for col in bert_cols:
            label = col.replace('bertscore_', '').replace('_', ' ').title()
            ax2.plot(df.index, df[col], marker='s', linewidth=2, markersize=8, label=label)
        
        ax2.set_xlabel('Alpha (Semantic Weight)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('BERTScore (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Alpha vs BERTScore', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(df.index)
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(output_dir, 'alpha_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    
    # Close to free memory
    plt.close()
    
    # Create individual metric plots
    if has_rouge:
        create_individual_plot(df, 'rouge1', 'ROUGE-1', output_dir)
    
    if has_bertscore:
        create_individual_plot(df, 'bertscore_f1', 'BERTScore F1', output_dir)

def create_individual_plot(df: pd.DataFrame, metric: str, title: str, output_dir: str):
    """Create individual metric plot"""
    if metric not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Line plot
    ax.plot(df.index, df[metric], marker='o', linewidth=3, markersize=10, color='#2E86AB')
    
    # Highlight best
    best_alpha = df[metric].idxmax()
    best_score = df.loc[best_alpha, metric]
    ax.scatter([best_alpha], [best_score], s=200, c='red', zorder=5, 
               label=f'Best: α={best_alpha:.1f} ({best_score:.2f})')
    
    ax.set_xlabel('Alpha (Semantic Weight)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{title} Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Alpha vs {title}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df.index)
    
    plt.tight_layout()
    
    filename = f"{metric}_analysis.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()

def validate_consistency(all_results: List[Dict], output_dir: str):
    """Validate that all alphas used the same samples"""
    print("\n" + "="*80)
    print("VALIDATING CONSISTENCY")
    print("="*80)
    
    if len(all_results) < 2:
        print("⚠️  Only one alpha available, skipping consistency check")
        return
    
    # Check sample IDs
    reference_ids = sorted(all_results[0]['sample_ids'])
    reference_alpha = all_results[0]['alpha']
    
    all_consistent = True
    
    for result in all_results[1:]:
        current_ids = sorted(result['sample_ids'])
        current_alpha = result['alpha']
        
        if current_ids != reference_ids:
            print(f"❌ INCONSISTENCY: Alpha {current_alpha:.1f} has different samples than alpha {reference_alpha:.1f}")
            all_consistent = False
        else:
            print(f"✅ Alpha {current_alpha:.1f}: Same samples as alpha {reference_alpha:.1f}")
    
    if all_consistent:
        print(f"\n✅ VALIDATION PASSED: All {len(all_results)} alphas used the same {len(reference_ids)} samples")
        print("   Experiment is scientifically valid for comparison")
    else:
        print(f"\n❌ VALIDATION FAILED: Alphas used different samples")
        print("   ⚠️  WARNING: Results may not be directly comparable!")
    
    # Save validation report
    validation_report = {
        'validation_status': 'PASSED' if all_consistent else 'FAILED',
        'num_alphas': len(all_results),
        'reference_alpha': reference_alpha,
        'num_samples': len(reference_ids),
        'sample_ids': reference_ids
    }
    
    validation_path = os.path.join(output_dir, 'validation_report.json')
    with open(validation_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    print(f"\n✅ Validation report saved: {validation_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow"""
    
    start_time = pd.Timestamp.now()
    
    # Step 1: Discover files
    summary_files = discover_summary_files(RESULTS_DIR)
    
    # Step 2: Analyze each alpha
    all_results = []
    for alpha in sorted(summary_files.keys()):
        filepath = summary_files[alpha]
        result = analyze_alpha(alpha, filepath)
        if result:
            all_results.append(result)
    
    if len(all_results) == 0:
        print("\n❌ ERROR: No results generated")
        return
    
    # Step 3: Create summary table
    df = create_summary_table(all_results)
    
    # Step 4: Save results
    save_results_csv(df, OUTPUT_DIR)
    
    # Step 5: Generate text report
    generate_text_report(df, all_results, OUTPUT_DIR)
    
    # Step 6: Create visualizations
    create_visualizations(df, OUTPUT_DIR)
    
    # Step 7: Validate consistency
    validate_consistency(all_results, OUTPUT_DIR)
    
    # Summary
    end_time = pd.Timestamp.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ REPORT GENERATION COMPLETE!")
    print("="*80)
    print(f"Time Elapsed: {duration:.1f} seconds")
    print(f"Results Directory: {OUTPUT_DIR}")
    print(f"\nGenerated Files:")
    print(f"  1. alpha_sweep_results.csv")
    print(f"  2. COMPREHENSIVE_REPORT.txt")
    print(f"  3. alpha_analysis.png")
    if COMPUTE_ROUGE:
        print(f"  4. rouge_results.csv")
        print(f"  5. rouge1_analysis.png")
    if COMPUTE_BERTSCORE:
        print(f"  6. bertscore_results.csv")
        print(f"  7. bertscore_f1_analysis.png")
    print(f"  8. validation_report.json")
    print("\n🎉 All reports generated successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
