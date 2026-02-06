import os
import sys
import subprocess
import json

def setup_environment():
    """Installs missing dependencies for the report generator."""
    required_packages = ["pandas", "tabulate"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    print("✓ Report generator dependencies verified.\n")

# Run setup
setup_environment()

import pandas as pd
from datetime import datetime

"""
================================================================================
FINAL REPORT GENERATOR FOR ABLATION STUDY
================================================================================
This script automates the evaluation of multiple result directories and
generates a consolidated comparison report.

Usage:
    python generate_final_report.py
================================================================================
"""

def get_result_dirs():
    """Identify directories containing summarization results."""
    # Base search on script's own location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    
    candidates = [
        'results_flat_overlap', 
        'results_flat_1024', 
        'Overlap 1024', 
        'Flat 1024'
    ]
    
    found = []
    for cand in candidates:
        # Check script-relative first (robust local) and then CWD-relative
        for base in [script_dir, cwd]:
            full_path = os.path.join(base, cand)
            if os.path.exists(full_path) and os.path.isdir(full_path):
                # Check if it has batches or a nested results dir
                if any(f.startswith('summaries_batch_') for f in os.listdir(full_path)):
                    found.append(full_path)
                    break
                else:
                    # Check one level deep
                    for entry in os.scandir(full_path):
                        if entry.is_dir() and any(f.startswith('summaries_batch_') for f in os.listdir(entry.path)):
                            found.append(full_path)
                            break
                if full_path in found: break
    return list(set(found))

def run_evaluation(dir_path):
    """Run evaluate_results.py on a specific directory."""
    print(f"\n>>> Evaluating: {os.path.basename(dir_path)}...")
    
    # Resolve the absolute path to evaluate_results.py (it should be in the same dir as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "evaluate_results.py")
    
    cmd = ["python", eval_script, "--results_dir", dir_path]
    try:
        subprocess.check_call(cmd)
        return True
    except Exception as e:
        print(f"❌ Evaluation failed for {dir_path}: {e}")
        return False

def collect_metrics(result_dirs):
    """Collect all metrics_*.json files into a list."""
    all_metrics = []
    
    for d in result_dirs:
        # Search for metrics file in the dir or subdirs
        metrics_files = []
        for root, _, files in os.walk(d):
            for f in files:
                if f.startswith('metrics_') and f.endswith('.json'):
                    metrics_files.append(os.path.join(root, f))
        
        if not metrics_files:
            # Try running evaluation if missing
            if run_evaluation(d):
                # Re-search
                for root, _, files in os.walk(d):
                    for f in files:
                        if f.startswith('metrics_') and f.endswith('.json'):
                            metrics_files.append(os.path.join(root, f))
        
        for mf in metrics_files:
            with open(mf, 'r') as f:
                data = json.load(f)
                data['source_folder'] = os.path.basename(d)
                all_metrics.append(data)
                
    return all_metrics

def generate_report(metrics_list):
    """Generate a Markdown report from metrics."""
    if not metrics_list:
        print("No metrics found to report.")
        return
    
    df = pd.DataFrame(metrics_list)
    
    # Reorder columns for readability
    cols = ['source_folder', 'method', 'num_samples', 'rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]
    
    # Format numbers
    for col in ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    try:
        markdown_table = df.to_markdown(index=False)
    except ImportError:
        # Robust fallback if tabulate/pandas markdown is broken
        print("   (Note: 'tabulate' missing, using manual markdown formatter)")
        header = "| " + " | ".join(df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(map(str, row.values)) + " |")
        markdown_table = "\n".join([header, separator] + rows)
    
    report = f"""# Ablation Study: Final Comparison Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Results Summary

{markdown_table}

## Analysis
- **Flat 1024**: Baseline chunking strategy.
- **Overlap 1024**: Sliding window strategy (128-token overlap).

> [!NOTE]
> Metrics were calculated locally on Mac M3 Pro using the DeBERTa-XLarge-MNLI model for BERTScore.
"""
    
    report_file = "final_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print("\n" + "="*70)
    print("FINAL COMPARISON REPORT")
    print("="*70)
    print(markdown_table)
    print("="*70)
    print(f"Report saved to: {os.path.abspath(report_file)}")

def main():
    print("Searching for result directories...")
    dirs = get_result_dirs()
    if not dirs:
        print("❌ No result directories found.")
        return
    
    print(f"Found {len(dirs)} directories: {[os.path.basename(d) for d in dirs]}")
    
    metrics = collect_metrics(dirs)
    generate_report(metrics)

if __name__ == "__main__":
    main()
