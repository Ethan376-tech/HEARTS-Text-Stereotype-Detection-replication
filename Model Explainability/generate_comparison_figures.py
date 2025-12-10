import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Load data
shap_df = pd.read_csv('shap_results.csv')
lime_df = pd.read_csv('lime_results.csv')

# Merge SHAP and LIME results
common_columns = ['sentence_id', 'token', 'sentence', 'dataset', 'categorisation', 
                  'predicted_label', 'actual_label']
merged_df = pd.merge(shap_df[common_columns + ['value_shap']], 
                     lime_df[common_columns + ['value_lime']], 
                     on=common_columns, 
                     how='inner')

# Add column to identify correct vs incorrect predictions
merged_df['prediction_correct'] = merged_df['predicted_label'] == merged_df['actual_label']

def get_token_rankings_for_sentence(sentence_data):
    """Get token rankings for a single sentence"""
    # Sort by absolute values for ranking
    sentence_data = sentence_data.copy()
    sentence_data['shap_abs'] = sentence_data['value_shap'].abs()
    sentence_data['lime_abs'] = sentence_data['value_lime'].abs()
    
    # Rank tokens by absolute importance (1 = most important)
    sentence_data['shap_rank'] = sentence_data['shap_abs'].rank(ascending=False, method='min')
    sentence_data['lime_rank'] = sentence_data['lime_abs'].rank(ascending=False, method='min')
    
    return sentence_data[['token', 'value_shap', 'value_lime', 'shap_rank', 'lime_rank']]

def plot_token_rankings_comparison(sentence_data, title, save_path):
    """Plot comparison of SHAP and LIME token rankings for a sentence"""
    tokens = sentence_data['token'].values
    shap_ranks = sentence_data['shap_rank'].values
    lime_ranks = sentence_data['lime_rank'].values
    
    # Calculate Spearman correlation
    spearman_corr, p_value = spearmanr(shap_ranks, lime_ranks)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot rankings
    x_pos = np.arange(len(tokens))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, shap_ranks, width, label='SHAP', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x_pos + width/2, lime_ranks, width, label='LIME', alpha=0.8, color='#A23B72')
    
    # Customize plot
    ax.set_xlabel('Token Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Token Rank (1 = Most Important)', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\nSpearman Correlation: {spearman_corr:.3f} (p={p_value:.3f})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.invert_yaxis()  # Lower rank (1) at top
    
    # Add value annotations on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        if height1 <= len(tokens) * 0.3:  # Only annotate if bar is not too tall
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{sentence_data.iloc[i]["value_shap"]:.3f}',
                   ha='center', va='bottom', fontsize=7, rotation=90)
        if height2 <= len(tokens) * 0.3:
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{sentence_data.iloc[i]["value_lime"]:.3f}',
                   ha='center', va='bottom', fontsize=7, rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return spearman_corr, p_value

# Separate correct and incorrect predictions
correct_predictions = merged_df[merged_df['prediction_correct'] == True]
incorrect_predictions = merged_df[merged_df['prediction_correct'] == False]

# Sample sentences for visualization
# For correct predictions - sample a few representative sentences
correct_sentences = correct_predictions.groupby('sentence_id').first().reset_index()
if len(correct_sentences) > 0:
    # Select a sentence with moderate alignment (not too high, not too low correlation)
    sample_correct = correct_sentences.sample(n=min(3, len(correct_sentences)), random_state=42)
    
    for idx, row in sample_correct.iterrows():
        sentence_id = row['sentence_id']
        sentence_data = get_token_rankings_for_sentence(
            correct_predictions[correct_predictions['sentence_id'] == sentence_id]
        )
        sentence_text = row['sentence'][:80] + '...' if len(row['sentence']) > 80 else row['sentence']
        title = f'Figure 3: Correct Prediction - Loose Alignment\n"{sentence_text}"'
        corr, p = plot_token_rankings_comparison(
            sentence_data, 
            title,
            f'figure3_correct_prediction_{sentence_id}.png'
        )
        print(f"Correct prediction (sentence {sentence_id}): Spearman correlation = {corr:.3f}")

# For incorrect predictions
incorrect_sentences = incorrect_predictions.groupby('sentence_id').first().reset_index()
if len(incorrect_sentences) > 0:
    sample_incorrect = incorrect_sentences.sample(n=min(3, len(incorrect_sentences)), random_state=42)
    
    for idx, row in sample_incorrect.iterrows():
        sentence_id = row['sentence_id']
        sentence_data = get_token_rankings_for_sentence(
            incorrect_predictions[incorrect_predictions['sentence_id'] == sentence_id]
        )
        sentence_text = row['sentence'][:80] + '...' if len(row['sentence']) > 80 else row['sentence']
        title = f'Figure 4: Incorrect Prediction - Divergent Outcomes\n"{sentence_text}"'
        corr, p = plot_token_rankings_comparison(
            sentence_data,
            title,
            f'figure4_incorrect_prediction_{sentence_id}.png'
        )
        print(f"Incorrect prediction (sentence {sentence_id}): Spearman correlation = {corr:.3f}")

# Create a summary comparison plot showing distribution of correlations
def create_summary_comparison_plot():
    """Create a summary plot comparing correlations for correct vs incorrect predictions"""
    correlations_correct = []
    correlations_incorrect = []
    
    # Calculate correlations for all sentences
    for sentence_id in correct_predictions['sentence_id'].unique():
        sent_data = correct_predictions[correct_predictions['sentence_id'] == sentence_id]
        if len(sent_data) > 1:
            rankings = get_token_rankings_for_sentence(sent_data)
            corr, _ = spearmanr(rankings['shap_rank'], rankings['lime_rank'])
            if not np.isnan(corr):
                correlations_correct.append(corr)
    
    for sentence_id in incorrect_predictions['sentence_id'].unique():
        sent_data = incorrect_predictions[incorrect_predictions['sentence_id'] == sentence_id]
        if len(sent_data) > 1:
            rankings = get_token_rankings_for_sentence(sent_data)
            corr, _ = spearmanr(rankings['shap_rank'], rankings['lime_rank'])
            if not np.isnan(corr):
                correlations_incorrect.append(corr)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram comparison
    ax1.hist(correlations_correct, bins=20, alpha=0.7, label='Correct Predictions', color='#2E86AB', edgecolor='black')
    ax1.hist(correlations_incorrect, bins=20, alpha=0.7, label='Incorrect Predictions', color='#A23B72', edgecolor='black')
    ax1.set_xlabel('Spearman Correlation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of SHAP-LIME Correlation\nby Prediction Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot comparison
    data_to_plot = [correlations_correct, correlations_incorrect]
    bp = ax2.boxplot(data_to_plot, labels=['Correct\nPredictions', 'Incorrect\nPredictions'], 
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][1].set_facecolor('#A23B72')
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    ax2.set_ylabel('Spearman Correlation', fontsize=12, fontweight='bold')
    ax2.set_title('SHAP-LIME Correlation Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    mean_correct = np.mean(correlations_correct) if correlations_correct else 0
    mean_incorrect = np.mean(correlations_incorrect) if correlations_incorrect else 0
    stats_text = f'Mean Correlation:\nCorrect: {mean_correct:.3f}\nIncorrect: {mean_incorrect:.3f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('shap_lime_correlation_summary.png', bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary Statistics:")
    print(f"Correct predictions - Mean correlation: {mean_correct:.3f}, Count: {len(correlations_correct)}")
    print(f"Incorrect predictions - Mean correlation: {mean_incorrect:.3f}, Count: {len(correlations_incorrect)}")

create_summary_comparison_plot()

print("\nFigures generated successfully!")
print("Generated files:")
print("- figure3_correct_prediction_*.png (for correct predictions)")
print("- figure4_incorrect_prediction_*.png (for incorrect predictions)")
print("- shap_lime_correlation_summary.png (summary comparison)")

