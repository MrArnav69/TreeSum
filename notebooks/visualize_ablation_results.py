
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

# Paths
RESULT_DIR = os.path.join(os.path.dirname(__file__), '../results/ablation')
CSV_PATH = os.path.join(RESULT_DIR, 'ablation_results.csv')
CHART_PATH = os.path.join(RESULT_DIR, 'ablation_comparison.png')
LATEX_PATH = os.path.join(RESULT_DIR, 'ablation_table.tex')

def generate_visualizations():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run ablation study first.")
        return

    # Load Data
    df = pd.read_csv(CSV_PATH)
    # Transpose back if needed or handle the format from run_ablation_study
    # The script saves df.to_csv() from a transposed frame: index=Models, cols=Metrics
    # So the CSV will have an unnamed column 0 for Model Names.
    
    # Reload with correct index
    df = pd.read_csv(CSV_PATH, index_col=0)
    
    print("\nLoaded Data:")
    print(df)
    
    # 1. Generate Bar Chart
    plt.figure(figsize=(10, 6))
    
    # Prepare data for Seaborn (Long Format)
    df_reset = df.reset_index().rename(columns={'index': 'Method'})
    df_melt = df_reset.melt(id_vars='Method', value_vars=['rouge1', 'rouge2', 'rougeL'], 
                            var_name='Metric', value_name='Score')
    
    # Plot
    ax = sns.barplot(data=df_melt, x='Metric', y='Score', hue='Method', palette='viridis')
    
    plt.title('ROUGE Score Comparison: TreeSum vs Baselines', fontsize=14, fontweight='bold')
    plt.ylabel('ROUGE Score (Higher is Better)', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.legend(title='Summarization Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(CHART_PATH, dpi=300)
    print(f"\nSaved Chart to: {CHART_PATH}")
    
    # 2. Generate LaTeX Table
    latex_str = df[['rouge1', 'rouge2', 'rougeL']].to_latex(
        float_format="%.2f",
        caption="Ablation Study Results: Comparison of different chunking strategies on Multi-News subset.",
        label="tab:ablation_results"
    )
    
    with open(LATEX_PATH, 'w') as f:
        f.write(latex_str)
        
    print(f"\nSaved LaTeX Table to: {LATEX_PATH}")
    print("\n=== LaTeX Code ===")
    print(latex_str)

if __name__ == "__main__":
    generate_visualizations()
