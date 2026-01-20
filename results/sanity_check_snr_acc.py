from itertools import product
from snr.stats import compute_decision_accuracy
from snr.dataloader import get_slice
from snr.metrics import decision_acc_fast, signal_to_noise_ratio
import pandas as pd
from snr.download.hf import pull_predictions_from_hf
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from IPython.display import display
from scipy import stats
from snr_utils import compute_score_and_stepstd, compute_mean_std


local_path = pull_predictions_from_hf("allenai/signal-and-noise", split_name='datadecide_intermediate')
df = pd.read_parquet(local_path)

# Reproduce the result from the paper (SNR vs decision accuracy in datadecide.ipyn) for sizes in SIZES models


# Setup

#MIXES = df['mix'].unique()
MIXES = df['group'].unique()
TASKS = df['task'].unique() 
#['arc_challenge' 'arc_easy' 'boolq' 'csqa' 'hellaswag' 'openbookqa' 'piqa'
#'socialiqa' 'winogrande' 'mmlu' 'olmes_10_macro_avg']
SEEDS = df['seed'].unique()
selected_tasks = TASKS
metric = 'primary_metric'
SIZES = ['60M', '90M', '150M', '300M', '530M', '750M', '1B']

selected_sizes = SIZES[4:]  # only 530M, 750M and 1B
SEED = 6198


assert "Your MIXES:", sorted(MIXES) == sorted(df['group'].unique()) 

#print(df.columns)

metrics_df = df[df['size'].isin(selected_sizes)]
metrics_df = metrics_df[metrics_df['task'].isin(selected_tasks)]
metrics_df = metrics_df[['task', 'mix', 'size', 'seed', 'step', 'primary_metric']]


# Compute score as the average of the last 10% checkpoints
# Compute step_std as the std of the last 30% checkpoints

compute_score_and_stepstd(selected_tasks, SEEDS, MIXES, selected_sizes, metrics_df, metric)




compute_mean_std(metrics_df, selected_tasks, selected_sizes, SEEDS)

# Compute decision accuracy

for seed, task, size in tqdm(product(SEEDS, selected_tasks, selected_sizes[:-1]), total = len(SEEDS)*len(selected_tasks)*len(selected_sizes[:-1])): # decision accuracy for 1B models is always 1.0
    score_1b = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == '1B')]
    score_size = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == size) & (metrics_df['seed'] == seed)]
    
    if score_size.empty or score_1b.empty:
        continue

    score_1b_sorted   = score_1b.sort_values('score', ascending=False)
    score_size_sorted = score_size.sort_values('score', ascending=False)

    decision_accuracy = compute_decision_accuracy(
        mixes_size = score_size_sorted['mix'].tolist(),
        mixes_1b = score_1b_sorted['mix'].tolist()
    )
    metrics_df.loc[(metrics_df['task'] == task) & (metrics_df['size'] == size) & (metrics_df['seed'] == seed), 'decision_accuracy'] = decision_accuracy

#display(metrics_df[metrics_df['task'] == 'arc_challenge'].head())

# Get mix_data

recap = []

for size in selected_sizes[:-1]:
    for task in metrics_df['task'].unique():
        if task == 'olmes_10_macro_avg': continue
        if task == 'boolq': continue
        mix_data = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == size)]
        
        if len(mix_data) == 0:
            continue

        m_mean = np.mean(mix_data['mean'])

        if not np.isfinite(m_mean) or m_mean <= 0:
            # mean used for normalization must be positive and finite
            continue

        rel_spread   = np.mean(mix_data['std']) / m_mean
        smoothness   = np.mean(mix_data['step_std']) / m_mean
        rel_dispersion     = (np.mean(mix_data['max']) - np.mean(mix_data['min'])) / m_mean
        decision_accuracy         = np.mean(mix_data['decision_accuracy'])


        # Require finite and positive SNR ingredients
        if not (np.isfinite(rel_spread) and np.isfinite(smoothness) and np.isfinite(rel_dispersion) and np.isfinite(decision_accuracy)):
            continue
        if smoothness <= 0:  # avoid division by 0 and log10 issues downstream
            continue
        
        recap += [{
            'size': size,
            'task': task,
            'SNR (rel_dispersion/smoothness)': rel_dispersion / smoothness, # TO DEFINE
            'SNR (rel_spread/smoothness)': rel_spread / smoothness,
            'decision_accuracy': decision_accuracy
        }]
        

recap_df = pd.DataFrame(recap)

# Plot SNR vs decision accuracy

size_symbols = {
    '4M': 'o',
    '6M': 's', 
    '8M': '^',
    '10M': 'D',
    '14M': 'v',
    '16M': 'p',
    '20M': 'h',
    '60M': '8',
    '90M': '*',
    '150M': 'H',
    '300M': 'X',
    '530M': 'd',
    '750M': 'P',
    '1B': '<'
}

fig, (ax_paper, ax_figure) = plt.subplots(1, 2, figsize=(12, 5))

# Collect data for fits
snr_dispersion_all = []
snr_spread_all = []
dacc_dispersion_all = []
dacc_spread_all = []

for size in selected_sizes[:-1]:
    subset = recap_df[recap_df['size'] == size]
    ax_paper.scatter(subset['SNR (rel_dispersion/smoothness)'], subset['decision_accuracy'], label=size, marker=size_symbols[size], s=100)
    ax_figure.scatter(subset['SNR (rel_spread/smoothness)'], subset['decision_accuracy'], label=size, marker=size_symbols[size], s=100)
    
    # Collect data for fit
    snr_dispersion_all.extend(subset['SNR (rel_dispersion/smoothness)'].tolist())
    snr_spread_all.extend(subset['SNR (rel_spread/smoothness)'].tolist())
    dacc_dispersion_all.extend(subset['decision_accuracy'].tolist())
    dacc_spread_all.extend(subset['decision_accuracy'].tolist())
    
    # Annotate points with task names
    for _, row in subset.iterrows():
        ax_paper.annotate(row['task'], (row['SNR (rel_dispersion/smoothness)'], row['decision_accuracy']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        ax_figure.annotate(row['task'], (row['SNR (rel_spread/smoothness)'], row['decision_accuracy']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Filter out non-positive SNR values for fitting
def safe_log_fit_and_ci(x_vals, y_vals, ax):
    x_raw = np.asarray(x_vals, dtype=float)
    y_raw = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_raw) & np.isfinite(y_raw) & (x_raw > 0)
    x = x_raw[mask]; y = y_raw[mask]
    if x.size < 3 or np.allclose(x, x.mean()):
        return  # insufficient points or zero variance

    x_log = np.log10(x)
    try:
        z = np.polyfit(x_log, y, 1)
    except np.linalg.LinAlgError:
        return
    p = np.poly1d(z)
    x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    y_line = p(np.log10(x_line))
    ax.plot(x_line, y_line, '--', color='black', alpha=0.5, linewidth=2)

    n = x_log.size
    y_hat = p(x_log)
    denom = np.sum((x_log - x_log.mean())**2)
    if n <= 2 or denom == 0:
        return
    s_err = np.sqrt(np.sum((y - y_hat)**2) / (n - 2))
    tcrit = stats.t.ppf(0.975, n - 2)
    x_new = np.log10(x_line)
    conf = tcrit * s_err * np.sqrt(1/n + (x_new - x_log.mean())**2 / denom)
    ax.fill_between(x_line, y_line - conf, y_line + conf, color='gray', alpha=0.2)

    r = np.corrcoef(x_log, y)[0, 1]; r2 = r**2
    stderr = s_err * np.sqrt((1 - r2) / (n - 2))
    ax.text(0.95, 0.05, f'R = {r:.3f} ± {stderr:.3f}\nR² = {r2:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

safe_log_fit_and_ci(snr_dispersion_all, dacc_dispersion_all, ax_paper)
safe_log_fit_and_ci(snr_spread_all,     dacc_spread_all,     ax_figure)

# Check

print(f"Dispersion points: {np.sum(np.isfinite(snr_dispersion_all) & (np.array(snr_dispersion_all)>0))}")
print(f"Spread points: {np.sum(np.isfinite(snr_spread_all) & (np.array(snr_spread_all)>0))}")

# Fit and plot for ax_paper (dispersion)
x_log = np.log10(snr_dispersion_all)
z = np.polyfit(x_log, dacc_dispersion_all, 1)
p = np.poly1d(z)
x_line = np.logspace(np.log10(min(snr_dispersion_all)), np.log10(max(snr_dispersion_all)), 100)
y_line = p(np.log10(x_line))

# Calculate confidence interval for dispersion
n = len(snr_dispersion_all)
y_mean = np.mean(dacc_dispersion_all)
x_mean = np.mean(x_log)
s_err = np.sqrt(np.sum((dacc_dispersion_all - p(x_log))**2)/(n-2))
x_new = np.log10(x_line)
conf = stats.t.ppf(0.975, n-2) * s_err * np.sqrt(1/n + (x_new - x_mean)**2 / np.sum((x_log - x_mean)**2))

ax_paper.plot(x_line, y_line, '--', color='black', alpha=0.5, linewidth=2)
ax_paper.fill_between(x_line, y_line-conf, y_line+conf, color='gray', alpha=0.2)

# Calculate correlation and standard error
r_dispersion = np.corrcoef(x_log, dacc_dispersion_all)[0, 1]
r2_dispersion = r_dispersion**2
stderr_dispersion = s_err * np.sqrt((1-r2_dispersion)/(n-2))
ax_paper.text(0.95, 0.05, f'R = {r_dispersion:.3f} ± {stderr_dispersion:.3f}\nR² = {r2_dispersion:.3f}', 
              transform=ax_paper.transAxes, fontsize=10, 
              verticalalignment='bottom', horizontalalignment='right',
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Fit and plot for ax_figure (spread)
x_log2 = np.log10(snr_spread_all)
z2 = np.polyfit(x_log2, dacc_spread_all, 1)
p2 = np.poly1d(z2)
x_line2 = np.logspace(np.log10(min(snr_spread_all)), np.log10(max(snr_spread_all)), 100)
y_line2 = p2(np.log10(x_line2))

# Calculate confidence interval for spread
n2 = len(snr_spread_all)
y_mean2 = np.mean(dacc_spread_all)
x_mean2 = np.mean(x_log2)
s_err2 = np.sqrt(np.sum((dacc_spread_all - p2(x_log2))**2)/(n2-2))
x_new2 = np.log10(x_line2)
conf2 = stats.t.ppf(0.975, n2-2) * s_err2 * np.sqrt(1/n2 + (x_new2 - x_mean2)**2 / np.sum((x_log2 - x_mean2)**2))

ax_figure.plot(x_line2, y_line2, '--', color='black', alpha=0.5, linewidth=2)
ax_figure.fill_between(x_line2, y_line2-conf2, y_line2+conf2, color='gray', alpha=0.2)

# Calculate correlation and standard error
r_spread = np.corrcoef(x_log2, dacc_spread_all)[0, 1]
r2_spread = r_spread**2
stderr_spread = s_err2 * np.sqrt((1-r2_spread)/(n2-2))
ax_figure.text(0.95, 0.05, f'R = {r_spread:.3f} ± {stderr_spread:.3f}\nR² = {r2_spread:.3f}', 
               transform=ax_figure.transAxes, fontsize=10, 
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Configure ax_paper
ax_paper.set_xlabel('Signal-to-Noise Ratio (SNR) using relative dispersion')
ax_paper.set_ylabel('Decision Accuracy')
ax_paper.legend(title='Model Size')
ax_paper.grid(True, linestyle='--', alpha=0.3)
ax_paper.set_xscale('log')
ax_paper.set_ylim(0.5, 1.0)

# Configure ax_figure
ax_figure.set_xlabel('Signal-to-Noise Ratio (SNR) using relative spread')
ax_figure.set_ylabel('Decision Accuracy')
ax_figure.legend(title='Model Size')
ax_figure.grid(True, linestyle='--', alpha=0.3)
ax_figure.set_xscale('log')
ax_figure.set_ylim(0.5, 1.0)

fig.suptitle('Signal-to-Noise Ratio vs Decision Accuracy')
#fig.savefig('snr_vs_decision_accuracy_selected_sizes.png', dpi=300)
plt.tight_layout()
plt.show()