from tqdm import tqdm
from snr.dataloader import get_slice
from snr.metrics import decision_acc_fast
import pandas as pd
from snr.download.hf import pull_predictions_from_hf
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
import json
import os
from IPython.display import display
from itertools import product
from snr.stats import compute_decision_accuracy
from scipy import stats
import matplotlib.pyplot as plt
from snr_utils import compute_score_and_stepstd, compute_mean_std, compute_score_model


local_path = pull_predictions_from_hf("allenai/signal-and-noise", split_name='datadecide_intermediate')
df = pd.read_parquet(local_path)

MIXES = df['group'].unique()
TASKS = df['task'].unique() 
SEEDS = df['seed'].unique()
SEED = 6198
SIZES = ['60M', '90M', '150M', '300M', '530M', '750M', '1B']
k = 5  # number of folds per task
metric = 'primary_metric'
selected_tasks = ['arc_easy', 'csqa', 'hellaswag', 'openbookqa', 'socialiqa', 'winogrande'] # removed 'boolq', 'piqa', 'arc_challenge'
selected_sizes = [ '150M', '300M', '530M', '750M', '1B']
selected_seeds = SEEDS

# Setup to get models (i.e. (mix, seed, size, step)) evaluated on k-folds
model_used = ['DCLM-baseline-20M-5xC', 'dolma17-25p-DCLM-baseline-75p-150M-5xC', 'falcon_and_cc_tulu_qc_top10-530M-5xC']
# used 'allenai/DataDecide-dclm-baseline-20M' # 'DCLM-baseline-20M-5xC'
# and 'allenai/DataDecide-dclm-baseline-75p-dolma1.7-25p-150M', 'allenai/DataDecide-falcon-and-cc-qc-orig-10p-530M'
mix_used = ['DCLM-baseline', 'dolma17-25p-DCLM-baseline-75p', 'falcon_and_cc_tulu_qc_top10']
size_used = ['20M', '150M', '530M']
seed_used = [6198, 6198, 6198] # default seed
step_used = [14594, 38157, 57776] # changed 57500 to 57776


""" model_used = model_used[:1]
mix_used = mix_used[:1]
size_used = size_used[:1]
seed_used = seed_used[:1] """
# STEP 0 : get dataframe with [task, mix, seed, size, score_per_fold, step, primary_metric]

# Pull data to get task, mix, seed, size, step, primary_metric

# Create metrics_df for summary stats (no fold info)
metrics_df = df[df['size'].isin(selected_sizes) | df['size'].isin(size_used)]
metrics_df = metrics_df[metrics_df['task'].isin(selected_tasks) | metrics_df['task'].isin(TASKS)]
metrics_df = metrics_df[['task', 'mix', 'size', 'seed', 'step', 'primary_metric']]

# Create folds_df for all fold-based operations
folds_df = metrics_df.copy()

# Duplicate each row corresponding to the model(s) evaluated on the k-folds k times and assign fold = 0 … k-1.
for task in selected_tasks:
    for i in range(len(mix_used)):
        mix = mix_used[i]
        size = size_used[i]
        step = step_used[i]
        seed = seed_used[i]
        mask = (
            (folds_df['task'] == task) &
            (folds_df['seed'] == seed) &
            (folds_df['mix'] == mix) &
            (folds_df['size'] == size) &
            (folds_df['step'] == step)
        )
        expanded = (
            folds_df.loc[mask]
            .assign(fold=[list(range(k))] * mask.sum()) # type: ignore
            .explode('fold')
        )
        folds_df = pd.concat(
            [folds_df.loc[~mask], expanded],
            ignore_index=True
        )

display(folds_df.head(20))

folds_df['score_per_fold'] = np.nan

# Get score_per_fold for each task folds in folds_df

for i in range(k):  
    score_fold = compute_score_model(selected_tasks, model_used, k) # Need to update fn if different model are evaluated on tasks
    print("ATTENCION score_fold is ", score_fold)
    
    for task in selected_tasks:

        for j in range(len(model_used)):
            mix = mix_used[j]
            size = size_used[j]
            step = step_used[j]
            seed = seed_used[j]

            mask = (
                (folds_df['task'] == task) &
                (folds_df['mix'] == mix) &
                (folds_df['size'] == size) &
                (folds_df['step'] == step) &
                (folds_df['seed'] == seed) &
                (folds_df['fold'] == i)
            )

            fold_name = f"fold_{i}"
            # Defensive: check keys exist
            score = None
            try:
                score = score_fold[model_used[j]][task][fold_name]
            except KeyError:
                print(f"Warning: missing score for model={model_used[j]}, task={task}, fold={fold_name}")
            folds_df.loc[mask, 'score_per_fold'] = score

            # Instead of doing that, since there are several scores per fold,
        # we have to compute first the benchmark noise using the score associated to each model 
        # and then average the result to obtain the benchmark noise of the task

        # So far: folds_df has score_per_fold for each (model, fold) for each task
        # Next: compute benchmark noise per task wrt each model, then average over models

display(folds_df[folds_df['fold'].notna()].head(20))


# STEP 1 : Get measures of signal and noises

# Get benchmark noise "fold_std" (the relative stddev of the score_per_fold for each (mix, task, size, seed))

# Compute per-task benchmark noise for each (mix, step, size, seed) combination (i.e. model) 
# and average over models to get per-task benchmark noise

benchmark_noise_per_task_and_model = {}
benchmark_noise_per_task = {}

for task in selected_tasks:
    for j in range(len(mix_used)):
        
        base_mask = (
            (folds_df['task'] == task) &
            (folds_df['mix'] == mix_used[j]) &
            (folds_df['size'] == size_used[j]) &
            (folds_df['seed'] == seed_used[j]) &
            (folds_df['step'] == step_used[j]) &
            (folds_df['fold'].notna())
        )

        # Base mask focuses on 1 model and 1 task
        # Next : compute benchmark noise for that model and task

        scores = folds_df.loc[base_mask, 'score_per_fold'].values
        print(f"Scores for task {task}, model {model_used[j]}: {scores}")
        
        if np.isnan(scores).any():
            print(f"Warning: NaN scores found for task {task} and model {model_used[j]} in benchmark noise computation.")
        if len(scores) < 2:
            print(f"Warning: Not enough scores to compute benchmark noise for task {task} and model {model_used[j]}.")
            continue
        
        mean_score = np.mean(scores) # type: ignore
        std_score = np.std(scores) # type: ignore

        if mean_score == 0:
            print(f"Warning: Mean score is zero for task {task} and model {model_used[j]}, cannot compute relative benchmark noise.")
            benchmark_noise_per_task_and_model[(task, model_used[j])] = np.nan

        benchmark_noise_per_task_and_model[(task, model_used[j])] = std_score / mean_score
        
# Average benchmark noise over models for that task
benchmark_noise_per_task = {
    task: np.nanmean([benchmark_noise_per_task_and_model.get((task, model), np.nan) for model in model_used])
    for task in selected_tasks
}

# Store the same per-task benchmark noise on all rows of that task in metrics_df
for task, bn in benchmark_noise_per_task.items():
    metrics_df.loc[metrics_df['task'] == task, 'fold_std'] = bn

# Get step_std, score, mean, std 

compute_score_and_stepstd(selected_tasks, selected_seeds, MIXES, selected_sizes, metrics_df, metric)

# Remove 'step' and 'primary_metric' columns
metrics_df = metrics_df.drop(columns=['step', 'primary_metric'])
metrics_df = metrics_df.drop_duplicates(subset = ['task', 'mix', 'size', 'seed'], keep = 'first')

compute_mean_std(metrics_df, selected_tasks, selected_sizes, selected_seeds)

#display(metrics_df.head(10))

# STEP 2 : Compute decision accuracy

for seed, task, size in tqdm(product(selected_seeds, selected_tasks, selected_sizes[:-1]), total = len(selected_seeds)*len(selected_tasks)*len(selected_sizes[:-1])): # decision accuracy for 1B models is always 1.0
    score_1b = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == '1B')]
    score_size = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == size) & (metrics_df['seed'] == seed)]
    
    if score_size.empty or score_1b.empty:
        continue

    score_1b_sorted   = score_1b.sort_values('score', ascending=False)
    score_size_sorted = score_size.sort_values('score', ascending=False)

    decision_accuracy = compute_decision_accuracy(
        mixes_1b = score_1b_sorted['mix'].tolist(),
        mixes_size = score_size_sorted['mix'].tolist()
        
    )
    metrics_df.loc[(metrics_df['task'] == task) & (metrics_df['size'] == size) & (metrics_df['seed'] == seed), 'decision_accuracy'] = decision_accuracy

#display(metrics_df.head(10))

# STEP 3 : Get mix_data to average noise and signal measures over seeds


recap = []

for size in selected_sizes[:-1]:
    for task in metrics_df['task'].unique():
        if task == 'olmes_10_macro_avg': continue
        if task == 'boolq': continue
        
        mix_data = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == size)]
        
        if len(mix_data) == 0:
            continue

        m_mean = mix_data['mean'].values[0] # All mean values are the same

        if not np.isfinite(m_mean) or m_mean <= 0:
            # mean used for normalization must be positive and finite
            continue

        rel_spread   = np.mean(mix_data['std']) / m_mean
        step_noise   = np.mean(mix_data['step_std']) / m_mean
        rel_dispersion     = (np.mean(mix_data['max']) - np.mean(mix_data['min'])) / m_mean
        decision_accuracy         = np.mean(mix_data['decision_accuracy'])
        bench_noise = benchmark_noise_per_task.get(task, np.nan)
                
        # Require finite and positive SNR ingredients
        if not (np.isfinite(rel_spread) and np.isfinite(step_noise) and np.isfinite(rel_dispersion) and np.isfinite(decision_accuracy)):
            continue
        if step_noise <= 0:  # avoid division by 0 and log10 issues downstream
            continue
        
        recap += [{
            'size': size,
            'task': task,
            'bench_noise': bench_noise,
            'step_noise': step_noise,
            'SNR (rel_dispersion/step_noise)': rel_dispersion / step_noise, # TO DEFINE
            'SNR (rel_spread/step_noise)': rel_spread / step_noise,
            'SNR (rel_dispersion/bench_noise)': rel_dispersion / bench_noise,
            'SNR (rel_spread/bench_noise)': rel_spread / bench_noise,
            'decision_accuracy': decision_accuracy
        }]
        #print(f"bench_noise for task {task} size {size} is {bench_noise}")
        #print(f"step_noise for task {task} size {size} is {step_noise}")

recap_df = pd.DataFrame(recap)

#display(recap_df.head(10))

# STEP 4 : Plot correlations between decision accuracy and SNR (using checkpoint-to-checkpoint noise) 

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

#fig, (ax_disp_step, ax_spread_step, ax_disp_bench, ax_spread_bench) = plt.subplots(1, 4, figsize=(24, 5))
fig, (ax_disp_step, ax_disp_bench) = plt.subplots(1, 2, figsize=(12, 5))

# Collect data for fits
snr_disp_step_all = [] # SNR values using signal = relative dispersion and noise = step_noise
#snr_spread_step_all = [] # SNR values using signal = relative spread and noise = step_noise
dacc_disp_step_all = [] # Decision accuracy values aligned with snr_disp_step_all
#dacc_spread_step_all = [] # Decision accuracy values aligned with snr_spread_step_all

snr_disp_bench_all = [] # SNR values using signal = relative dispersion and noise = step_noise (benchmark)
#snr_spread_bench_all = [] # SNR values using signal = relative spread and noise = step_noise (benchmark)
dacc_disp_bench_all = [] # Decision accuracy values aligned with snr_disp_bench_all
#dacc_spread_bench_all = [] # Decision accuracy values aligned with snr_spread_bench_all

for size in selected_sizes[:-1]:
    subset = recap_df[recap_df['size'] == size]

    # Collect data for plotting
    ax_disp_step.scatter(subset['SNR (rel_dispersion/step_noise)'], subset['decision_accuracy'], label=size, marker=size_symbols[size], s=100)
    #ax_spread_step.scatter(subset['SNR (rel_spread/step_noise)'], subset['decision_accuracy'], label=size, marker=size_symbols[size], s=100)
    
    ax_disp_bench.scatter(subset['SNR (rel_dispersion/bench_noise)'], subset['decision_accuracy'], label=size, marker=size_symbols[size], s=100)
    #ax_spread_bench.scatter(subset['SNR (rel_spread/bench_noise)'], subset['decision_accuracy'], label=size, marker=size_symbols[size], s=100)

    # Collect data for fit
    snr_disp_step_all.extend(subset['SNR (rel_dispersion/step_noise)'].tolist())
    #snr_spread_step_all.extend(subset['SNR (rel_spread/step_noise)'].tolist())
    dacc_disp_step_all.extend(subset['decision_accuracy'].tolist())
    #dacc_spread_step_all.extend(subset['decision_accuracy'].tolist())

    snr_disp_bench_all.extend(subset['SNR (rel_dispersion/bench_noise)'].tolist())
    #snr_spread_bench_all.extend(subset['SNR (rel_spread/bench_noise)'].tolist())
    dacc_disp_bench_all.extend(subset['decision_accuracy'].tolist())
    #dacc_spread_bench_all.extend(subset['decision_accuracy'].tolist())
    
    # Annotate points with task names
    for _, row in subset.iterrows():
        ax_disp_step.annotate(row['task'], (row['SNR (rel_dispersion/step_noise)'], row['decision_accuracy']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        #ax_spread_step.annotate(row['task'], (row['SNR (rel_spread/step_noise)'], row['decision_accuracy']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        ax_disp_bench.annotate(row['task'], (row['SNR (rel_dispersion/bench_noise)'], row['decision_accuracy']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)    
        #ax_spread_bench.annotate(row['task'], (row['SNR (rel_spread/bench_noise)'], row['decision_accuracy']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

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

safe_log_fit_and_ci(snr_disp_step_all, dacc_disp_step_all, ax_disp_step)
#safe_log_fit_and_ci(snr_spread_step_all,     dacc_spread_step_all,     ax_spread_step)
safe_log_fit_and_ci(snr_disp_bench_all, dacc_disp_bench_all, ax_disp_bench)
#safe_log_fit_and_ci(snr_spread_bench_all,     dacc_spread_bench_all,     ax_spread_bench)

# Check

print(f"Dispersion points: {np.sum(np.isfinite(snr_disp_step_all) & (np.array(snr_disp_step_all)>0))}")
#print(f"Spread points: {np.sum(np.isfinite(snr_spread_step_all) & (np.array(snr_spread_step_all)>0))}")
print(f"Dispersion benchmark points: {np.sum(np.isfinite(snr_disp_bench_all) & (np.array(snr_disp_bench_all)>0))}")
#print(f"Spread benchmark points: {np.sum(np.isfinite(snr_spread_bench_all) & (np.array(snr_spread_bench_all)>0))}")

# Fit and plot for ax_disp_step (dispersion)
x_log = np.log10(snr_disp_step_all)
z = np.polyfit(x_log, dacc_disp_step_all, 1)
p = np.poly1d(z)
x_line = np.logspace(np.log10(min(snr_disp_step_all)), np.log10(max(snr_disp_step_all)), 100)
y_line = p(np.log10(x_line))

# Calculate confidence interval for dispersion
n = len(snr_disp_step_all)
y_mean = np.mean(dacc_disp_step_all)
x_mean = np.mean(x_log)
s_err = np.sqrt(np.sum((dacc_disp_step_all - p(x_log))**2)/(n-2))
x_new = np.log10(x_line)
conf = stats.t.ppf(0.975, n-2) * s_err * np.sqrt(1/n + (x_new - x_mean)**2 / np.sum((x_log - x_mean)**2))

ax_disp_step.plot(x_line, y_line, '--', color='black', alpha=0.5, linewidth=2)
ax_disp_step.fill_between(x_line, y_line-conf, y_line+conf, color='gray', alpha=0.2)

# Calculate correlation and standard error
r_dispersion = np.corrcoef(x_log, dacc_disp_step_all)[0, 1]
r2_dispersion = r_dispersion**2
stderr_dispersion = s_err * np.sqrt((1-r2_dispersion)/(n-2))
ax_disp_step.text(0.95, 0.05, f'R = {r_dispersion:.3f} ± {stderr_dispersion:.3f}\nR² = {r2_dispersion:.3f}', 
              transform=ax_disp_step.transAxes, fontsize=10, 
              verticalalignment='bottom', horizontalalignment='right',
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Fit and plot for ax_spread_step (spread)
#x_log2 = np.log10(snr_spread_step_all)#
#z2 = np.polyfit(x_log2, dacc_spread_step_all, 1)
#p2 = np.poly1d(z2)
#x_line2 = np.logspace(np.log10(min(snr_spread_step_all)), np.log10(max(snr_spread_step_all)), 100)
#y_line2 = p2(np.log10(x_line2))

# Calculate confidence interval for spread
""" n2 = len(snr_spread_step_all)
y_mean2 = np.mean(dacc_spread_step_all)
x_mean2 = np.mean(x_log2)
s_err2 = np.sqrt(np.sum((dacc_spread_step_all - p2(x_log2))**2)/(n2-2))
x_new2 = np.log10(x_line2)
conf2 = stats.t.ppf(0.975, n2-2) * s_err2 * np.sqrt(1/n2 + (x_new2 - x_mean2)**2 / np.sum((x_log2 - x_mean2)**2))

ax_spread_step.plot(x_line2, y_line2, '--', color='black', alpha=0.5, linewidth=2)
ax_spread_step.fill_between(x_line2, y_line2-conf2, y_line2+conf2, color='gray', alpha=0.2)

# Calculate correlation and standard error
r_spread = np.corrcoef(x_log2, dacc_spread_step_all)[0, 1]
r2_spread = r_spread**2
stderr_spread = s_err2 * np.sqrt((1-r2_spread)/(n2-2))
ax_spread_step.text(0.95, 0.05, f'R = {r_spread:.3f} ± {stderr_spread:.3f}\nR² = {r2_spread:.3f}', 
               transform=ax_spread_step.transAxes, fontsize=10, 
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')) """

# Configure ax_disp_step
ax_disp_step.set_xlabel('SNR using relative dispersion / step-to-step noise')
ax_disp_step.set_ylabel('Decision Accuracy')
ax_disp_step.legend(title='Model Size')
ax_disp_step.grid(True, linestyle='--', alpha=0.3)
ax_disp_step.set_xscale('log')
ax_disp_step.set_ylim(0.5, 1.0)

# Configure ax_spread_step
""" ax_spread_step.set_xlabel('SNR using relative spread')
ax_spread_step.set_ylabel('Decision Accuracy')
ax_spread_step.legend(title='Model Size')
ax_spread_step.grid(True, linestyle='--', alpha=0.3)
ax_spread_step.set_xscale('log')
ax_spread_step.set_ylim(0.5, 1.0)
 """
# Configure ax_disp_bench
ax_disp_bench.set_xlabel('SNR using relative dispersion / benchmark noise')
ax_disp_bench.set_ylabel('Decision Accuracy')
ax_disp_bench.legend(title='Model Size')
ax_disp_bench.grid(True, linestyle='--', alpha=0.3)
ax_disp_bench.set_xscale('log')
ax_disp_bench.set_ylim(0.5, 1.0)
# Configure ax_spread_bench
""" ax_spread_bench.set_xlabel('SNR using relative spread / benchmark noise')
ax_spread_bench.set_ylabel('Decision Accuracy')
ax_spread_bench.legend(title='Model Size')
ax_spread_bench.grid(True, linestyle='--', alpha=0.3)
ax_spread_bench.set_xscale('log')
ax_spread_bench.set_ylim(0.5, 1.0) """


fig.suptitle('Signal-to-Noise Ratio vs Decision Accuracy')
fig.savefig('snr_vs_decision_accuracy_selected_sizes.png', dpi=300)
plt.tight_layout()
plt.show()

# ===== Be careful ====
"""
We computed benchmark noise using the scores of a 20M model and applying it to all sizes.
If possible, run the experiment using different model sizes.
"""

# Step 5 : Plot correlation step-to-step noise vs. benchmark_noise.

fig, ax_step_bench = plt.subplots(figsize=(12, 5))

# Collect data for fit
step_all = [] # step noise values
bench_all = [] # benchmark noise values

for size in selected_sizes[:-1]:
    subset = recap_df[recap_df['size'] == size]
    #print(f"subset is {display(subset)}")

    # Collect data for plotting
    ax_step_bench.scatter(subset['step_noise'], subset['bench_noise'], label=size, marker=size_symbols[size], s=100)
    # Collect data for fit
    step_all.extend(subset['step_noise'].tolist())
    bench_all.extend(subset['bench_noise'].tolist())

    # Annotate points with task names
    for _, row in subset.iterrows():
        ax_step_bench.annotate(row['task'], (row['step_noise'], row['bench_noise']),textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Fit and plot for ax_step_bench
safe_log_fit_and_ci(step_all, bench_all, ax_step_bench)
ax_step_bench.set_xlabel('Step-to-step noise')
ax_step_bench.set_ylabel('Benchmark noise')
ax_step_bench.legend(title='Model Size')
ax_step_bench.grid(True, linestyle='--', alpha=0.3)
ax_step_bench.set_xscale('log')
ax_step_bench.set_yscale('log')
fig.suptitle('Step-to-step noise vs Benchmark noise')
fig.savefig('step_to_step_vs_benchmark_noise_complete.png', dpi=300)
plt.tight_layout()
#plt.show()
















#INCORRECT PREVIOUS CODE BELOW






"""

# ============= Goal 1 : correlate decision accuracy and benchmark noise ===================

benchmark_noise = {}
decision_acc = {}

# ============= Step 1 : measure dataset noise from evaluation runs ===============

MODEL_FOR_NOISE = 'allenai/DataDecide-dclm-baseline-20M' # 'DCLM-baseline-20M-5xC'

# details: --revision step14594-seed-default

for task in selected_tasks:
    scores = []
    for i in range(5):
        if task in ['hellaswag','arc_easy', 'boolq', 'csqa', 'openbookqa', 'piqa', 'socialiqa', 'winogrande']:
            metrics_file = f"results/k_folds/{task}_fold_{i}/task-000-{task}_flexible-metrics.json"

        if os.path.exists(metrics_file):# type: ignore
            with open(metrics_file, 'r') as f: # type: ignore
                metrics = json.load(f)
                scores.append(metrics['metrics']['primary_score'])
                #print(scores)
        else:
            print(f"Warning: {metrics_file} not found, skipping fold {i}")# type: ignore
    res = np.std(scores) / np.mean(scores)
    
    benchmark_noise[task] = res
    #print(f"Benchmark scores for task {task}: {scores}")
    #print(f"Noise of benchmark {task}: {np.std(scores) / np.mean(scores)}")

# ============= Step 2 : compute decision accuracy =================================

for task in selected_tasks:
    dacc = []
    for size in SIZES:
        scores_small  = get_slice(df, size=size, task=task, step=38157)
        scores_target = get_slice(df, size='1B', task=task, step=69369)
        
        # Find common models
        common_mixes = set(scores_small['mix']) & set(scores_target['mix'])
        scores_small_filtered = scores_small[scores_small['mix'].isin(common_mixes)]
        scores_target_filtered = scores_target[scores_target['mix'].isin(common_mixes)]
        # Debug: Check lengths match
        if len(scores_small_filtered) != len(scores_target_filtered):
            print(f"Skipping {task} size {size}: mismatched lengths {len(scores_small_filtered)} vs {len(scores_target_filtered)}")
            continue
        if len(scores_small_filtered) == 0:
            print(f"Skipping {task} size {size}: no common mixes")
            continue
        acc = decision_acc_fast(
            scores_small = scores_small.sort_values('mix')['primary_score'],
            scores_target = scores_target.sort_values('mix')['primary_score']
        )
        if task in benchmark_noise:
            dacc.append(acc)
    if len(dacc) == 0:
        continue
    decision_acc[task] = np.mean(dacc)
    #print(f"Decision accuracy for task {task}: {decision_acc[task]}")

#print(f"Decision accuracy of all tasks: {decision_acc}")

# ============ Step 3 : compute correlation =========================================

#print("benchmark_noise is", benchmark_noise)
#print("decision_acc is", decision_acc)

# Filter to only tasks that have noise data
valid_tasks = [b for b in tasks if b in signal_to_noise_ratios_benchmark]
#print(f"Valid tasks for correlation: {valid_tasks}")

pearson_correlation = pearsonr(
    [benchmark_noise[b] for b in valid_tasks],
    [decision_acc[b] for b in valid_tasks]
)
print(f"\n")


#print(f"Pearson's correlation of benchmark noise and decision accuracy is {pearson_correlation}")

spearman_correlation = spearmanr(
    [benchmark_noise[b] for b in valid_tasks],
    [decision_acc[b] for b in valid_tasks]
)
#print(f"Spearman's correlation of benchmark noise and decision accuracy is {spearman_correlation}")

kendall_correlation = kendalltau(
    [benchmark_noise[b] for b in valid_tasks],
    [decision_acc[b] for b in valid_tasks]
)
#print(f"Kendall Tau's correlation of benchmark noise and decision accuracy is {kendall_correlation}")

print("GOAL 1 DONEEEEEEEEE")

"""
# ============= Goal 2 : correlate decision accuracy and step-to-step noise ===================

# ==================== Step 1 : compute checkpoint-to-checkpoint noise ==========================
"""
local_path = pull_predictions_from_hf("allenai/signal-and-noise", split_name='datadecide_intermediate')
df = pd.read_parquet(local_path)
"""

""" metrics_df = df[df['size'].isin(selected_sizes) | df['size'].isin(size_used)]
metrics_df = metrics_df[metrics_df['task'].isin(selected_tasks) | metrics_df['task'].isin(TASKS)]
metrics_df = metrics_df[['task', 'mix', 'size', 'seed', 'step', 'primary_metric']]
 """
"""
# rename primary_metric to primary_score for clarity
df = df.rename(columns={'primary_metric': 'primary_score'})

# Compute per mix checkpoint-to-checkpoint noise for each task and size
step_to_step_noise = {}
for task in selected_tasks:
    noise_values = []
    for size in selected_sizes:
        scores_df = get_slice(df, size=size, task=task).sort_values('step')
        # take last 30% of steps
        num_steps = len(scores_df['step'].unique())
        last_30_percent_steps = scores_df['step'].unique()[int(num_steps*0.7):]
        scores_df = scores_df[scores_df['step'].isin(last_30_percent_steps)]

        # group by mix and compute per-mix checkpoint variability
        per_mix_noise = []
        for mix, group in scores_df.groupby('mix'):
            scores = group['primary_score'].values
            if len(scores) < 2:
                continue
            mix_mean = np.mean(scores)
            if not np.isfinite(mix_mean) or mix_mean == 0:
                continue
            # Compute relative standard deviation for this mix
            mix_noise = np.std(scores, ddof=0) / mix_mean
            per_mix_noise.append(mix_noise)
        
        if len(per_mix_noise) == 0:
            print(f"Skipping {task} size {size}: no valid per-mix noise")
            continue
        
        # Average the per-mix noise values
        noise = np.mean(per_mix_noise)
        noise_values.append(noise)
    
    if len(noise_values) > 0:
        step_to_step_noise[task] = np.mean(noise_values)

#print(f"Step-to-step noise for all tasks: {step_to_step_noise}")

# ==================== Step 2 : compute correlation with decision accuracy ==========================
pearson_correlation_step = pearsonr(
    [step_to_step_noise[b] for b in tasks],
    [decision_acc[b] for b in tasks]
)


spearman_correlation_step = spearmanr(
    [step_to_step_noise[b] for b in tasks],
    [decision_acc[b] for b in tasks]
)

kendall_correlation = kendalltau(
    [step_to_step_noise[b] for b in tasks],
    [decision_acc[b] for b in tasks]
)
print(f"Pearson's correlation of step-to-step noise and decision accuracy is {pearson_correlation_step} \n")
print(f"Spearman's correlation of step-to-step noise and decision accuracy is {spearman_correlation_step}\n")
print(f"Kendall Tau's correlation of step-to-step noise and decision accuracy is {kendall_correlation}\n")

print("GOAL 2 DONEEEEEEEEE")

# ========================= Plot =============================
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")
# Plot decision accuracy vs checkpoint-to-checkpoint noise
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=[step_to_step_noise[b] for b in tasks],
    y=[decision_acc[b] for b in tasks],
    s=100
)
# Diplay linear fit line and R and R^2 as legend
ax = plt.gca()

x_vals = np.array([step_to_step_noise[b] for b in tasks], dtype=float)
y_vals = np.array([decision_acc[b] for b in tasks], dtype=float)

mask = np.isfinite(x_vals) & np.isfinite(y_vals)
if mask.sum() >= 2:
    x_fit = x_vals[mask]
    y_fit = y_vals[mask]

    # Linear fit
    coef = np.polyfit(x_fit, y_fit, 1)
    x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
    y_line = np.polyval(coef, x_line)
    ax.plot(x_line, y_line, '--', color='black',
            label=f"fit: y={coef[0]:.2f}x+{coef[1]:.2f}")

    # Correlation
    r = np.corrcoef(x_fit, y_fit)[0, 1]
    r2 = r**2
    ax.text(0.98, 0.02, f"R={r:.3f}\nR²={r2:.3f}",
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.legend()

for i, task in enumerate(tasks):
    plt.text(
        step_to_step_noise[task],
        decision_acc[task],
        task,
        fontsize=9,
        ha='right'
    )
plt.title('Decision Accuracy vs Checkpoint-to-Checkpoint Noise')
plt.xlabel('Checkpoint-to-Checkpoint Noise (StdDev / Mean)')
plt.ylabel('Decision Accuracy')
plt.tight_layout()
#plt.savefig('decision_accuracy_vs_step_noise.png')
#plt.show()

# =============== Goal 3: Compute correlation between SNRs and decision accuracy ===================


# =============== Step 1: Compute signal to noise ratio for both types of noise ====================
from snr.metrics import signal_to_noise_ratio

signal_to_noise_ratios_benchmark = {}
signal_to_noise_ratios_step = {}

for task in tasks:
    snr_steps = []
    snr_benchmarks = []
    for size in SIZES:
        # Signal of a task is the relative dispersion of primary scores of mixes on task
        scores_df = get_slice(df, size=size, task=task).sort_values('step')
        # numpy array of scores in shape (mix, checkpoint)
        scores_arr = np.array([lst for lst in scores_df.groupby('mix')['primary_score'].apply(list)])
        signal = [np.std(scores) for scores in scores_arr]

        if len(signal) < 2:
            print(f"Skipping SNR computation for {task} size {size}: less than 2 mixes")
            continue
        
        # benchmark noise

        noise_bench = [signal_to_noise_ratios_benchmark[b] for b in valid_tasks]

        if len(signal) != len(noise_bench):
            print(f"Skipping SNR computation for {task} size {size}: mismatched signal and noise lengths {len(signal)} vs {len(noise)}")
            continue
        
        snr_benchmark = signal_to_noise_ratio(
            signal_scores = np.array(signal),
            noise_scores = np.array(noise_bench) # review noise_scores
        )

        snr_benchmarks.append(snr_benchmark)

        # step-to-step noise

        noise_step = [step_to_step_noise[b] for b in tasks]

        if len(signal) != len(noise_step):
            print(f"Skipping SNR computation for {task} size {size}: mismatched signal and noise lengths {len(signal)} vs {len(noise)}")
            continue

        snr_step = signal_to_noise_ratio(
            signal_scores = np.array(signal),
            noise_scores = np.array(noise_step)
        )

        snr_steps.append(snr_step)
    signal_to_noise_ratios_benchmark[task] = np.mean(snr_benchmarks)
    signal_to_noise_ratios_step[task] = np.mean(snr_steps)

print(f"Signal to Noise Ratio (benchmark noise): {signal_to_noise_ratios_benchmark}")
print(f"Signal to Noise Ratio (step-to-step noise): {signal_to_noise_ratios_step}")

# =============== Step 2: Compute correlation between SNRs and decision accuracy ====================

print("\n")
print("Decision Accuracy is", decision_acc)
print(f"Signal to Noise Ratios (benchmark noise): {signal_to_noise_ratios_benchmark}")
print(f"Signal to Noise Ratios (step-to-step noise): {signal_to_noise_ratios_step} \n")
pearson_snr_benchmark = pearsonr(
    [signal_to_noise_ratios_benchmark[b] for b in valid_tasks],
    [decision_acc[b] for b in valid_tasks]
)
pearson_snr_step = pearsonr(
    [signal_to_noise_ratios_step[b] for b in tasks if b != 'boolq'],
    [decision_acc[b] for b in tasks if b != 'boolq']
)
print(f"Pearson's correlation of SNR (benchmark noise) and decision accuracy is {pearson_snr_benchmark}")
print(f"Pearson's correlation of SNR (step-to-step noise) and decision accuracy is {pearson_snr_step}")


print(f"Kendall Tau's correlation of SNR (benchmark noise) and decision accuracy is {kendalltau([signal_to_noise_ratios_benchmark[b] for b in valid_tasks if b != 'boolq'],[decision_acc[b] for b in valid_tasks if b != 'boolq'])}  ")
print(f"Kendall Tau's correlation of SNR (step-to-step noise) and decision accuracy is {kendalltau([signal_to_noise_ratios_step[b] for b in tasks if b != 'boolq'],[decision_acc[b] for b in tasks if b != 'boolq'])}  ")

# ========================= Plot =============================

# Plot decision accuracy vs signal to noise ratio using step-to-step noise
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=[signal_to_noise_ratios_step[b] for b in tasks],
    y=[decision_acc[b] for b in tasks],
    s=100
)

ax = plt.gca()

x_vals = np.array([signal_to_noise_ratios_step[b] for b in tasks], dtype=float)
y_vals = np.array([decision_acc[b] for b in tasks], dtype=float)

mask = np.isfinite(x_vals) & np.isfinite(y_vals)
if mask.sum() >= 2:
    x_fit = x_vals[mask]
    y_fit = y_vals[mask]

    # Linear fit
    coef = np.polyfit(x_fit, y_fit, 1)
    x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
    y_line = np.polyval(coef, x_line)
    ax.plot(x_line, y_line, '--', color='black',
            label=f"fit: y={coef[0]:.2f}x+{coef[1]:.2f}")

    # Correlation
    r = np.corrcoef(x_fit, y_fit)[0, 1]
    r2 = r**2
    ax.text(0.98, 0.02, f"R={r:.3f}\nR²={r2:.3f}",
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.legend()

for i, task in enumerate(tasks):
    plt.text(
        signal_to_noise_ratios_step[task],
        decision_acc[task],
        task,
        fontsize=9,
        ha='right'
    )
plt.title('Decision Accuracy vs Signal to Noise Ratio (Step-to-Step Noise)')
plt.xlabel('Signal to Noise Ratio (Step-to-Step Noise)')
plt.ylabel('Decision Accuracy')
plt.tight_layout()
plt.show()

# Plot decision accuracy vs signal to noise ratio using benchmark noise
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=[signal_to_noise_ratios_benchmark[b] for b in valid_tasks],
    y=[decision_acc[b] for b in valid_tasks],
    s=100
)
for i, task in enumerate(valid_tasks):
    plt.text(
        signal_to_noise_ratios_benchmark[task],
        decision_acc[task],
        task,
        fontsize=9,
        ha='right'
    )
plt.title('Decision Accuracy vs Signal to Noise Ratio (Benchmark Noise)')
plt.xlabel('Signal to Noise Ratio (Benchmark Noise)')
plt.ylabel('Decision Accuracy')
plt.tight_layout()
#plt.show()

# Plot checkpoint-to-checkpoint noise vs benchmark noise
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=[signal_to_noise_ratios_benchmark[b] for b in valid_tasks],
    y=[signal_to_noise_ratios_step[b] for b in valid_tasks],
    s=100
)
for i, task in enumerate(valid_tasks):
    plt.text(
        signal_to_noise_ratios_benchmark[task],
        signal_to_noise_ratios_step[task],
        task,
        fontsize=9,
        ha='right'
    )
plt.title('Step-to-Step Noise vs Benchmark Noise')
plt.xlabel('Benchmark Noise (StdDev / Mean)')
plt.ylabel('Step-to-Step Noise (StdDev / Mean)')
plt.tight_layout()
#plt.savefig('step_to_step_vs_benchmark_noise.png')
#plt.show()

# ====== Goal 4 : Plot decision accuracy vs step-to-step noise for all tasks using 150M mixes ======

# =============== Step 1 : compute decision accuracy for all tasks =================

print(df.columns)

all_tasks = df['task'].unique()
all_decision_acc = {}



for task in all_tasks:
    if task == 'boolq' or task == 'piqa':
        continue

    scores_small  = get_slice(df, size='150M', task=task, step=38157)
    scores_target = get_slice(df, size='1B', task=task, step=69369)

    # Find common mixes
    common_mixes = set(scores_small['mix']) & set(scores_target['mix'])
    scores_small_filtered = scores_small[scores_small['mix'].isin(common_mixes)]
    scores_target_filtered = scores_target[scores_target['mix'].isin(common_mixes)]
    
    # Debug: Check lengths match
    if len(scores_small_filtered) != len(scores_target_filtered):
        print(f"Skipping {task}: mismatched lengths {len(scores_small_filtered)} vs {len(scores_target_filtered)}")
        continue
    
    if len(scores_small_filtered) == 0:
        print(f"Skipping {task}: no common mixes")
        continue

    acc = decision_acc_fast(
        scores_small = scores_small_filtered.sort_values('mix')['primary_score'].values,
        scores_target = scores_target_filtered.sort_values('mix')['primary_score'].values
    )
    if isinstance(acc, (int, float)) and not np.isnan(acc):
        all_decision_acc[task] = acc
# =============== Step 2 : get step-to-step noise for all tasks on 150M mixes =================

all_step_to_step_noise = {}
for task in all_tasks:
    scores_df = get_slice(df, size='150M', task=task).sort_values('step')

    grouped = scores_df.groupby('mix')['primary_score'].apply(list)
    if len(grouped) < 2:
        continue

    min_len = min(len(lst) for lst in grouped)
    if min_len == 0:
        continue

    scores_arr = np.array([lst[-min_len:] for lst in grouped])

    arr_mean = np.mean(scores_arr)
    arr_std  = np.std(scores_arr)

    if not np.isfinite(arr_mean) or arr_mean == 0:
        print(f"Skipping {task}: non-finite or zero mean ({arr_mean})")
        continue

    noise = arr_std / arr_mean



# =============== Step 3 : filter out None and incomplete values =================
for task in all_tasks:
    if task not in all_decision_acc or task not in all_step_to_step_noise:
        if task in all_decision_acc:
            del all_decision_acc[task]
        if task in all_step_to_step_noise:
            del all_step_to_step_noise[task]
    if task == 'boolq' or task == 'piqa':  # these tasks have nan values
        if task in all_decision_acc:
            del all_decision_acc[task]
        if task in all_step_to_step_noise:
            del all_step_to_step_noise[task]
# ========================= Plot =============================

all_tasks = [t for t in all_tasks if t in all_step_to_step_noise and t in all_decision_acc]
import matplotlib.pyplot as plt
# Plot decision accuracy vs checkpoint-to-checkpoint noise (150M models)
import seaborn as sns
sns.set(style="whitegrid")

x_vals = []
y_vals = []
plot_tasks = []
for task in all_tasks:
    if task in all_step_to_step_noise and task in all_decision_acc:
        x_vals.append(all_step_to_step_noise[task])
        y_vals.append(all_decision_acc[task])
        plot_tasks.append(task)

if len(x_vals) == 0:
    print("Skipping plot: no overlapping decision accuracy and noise values.")
else:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_vals, y=y_vals, s=100)
    for xv, yv, task in zip(x_vals, y_vals, plot_tasks):
        plt.text(xv, yv, task, fontsize=9, ha='right')

    # Add linear regression line only if enough points
    if len(x_vals) >= 2:
        coef = np.polyfit(x_vals, y_vals, 1)
        sns.regplot(x=x_vals, y=y_vals, scatter=False, color='blue',
                    line_kws={'label': f"y={coef[0]:.2f}x+{coef[1]:.2f}"})

    plt.title('Decision Accuracy vs Checkpoint-to-Checkpoint Noise (150M models)')
    plt.xlabel('Checkpoint-to-Checkpoint Noise (StdDev / Mean)')
    plt.ylabel('Decision Accuracy')
    plt.tight_layout()
    plt.legend()
    #plt.show()

    """