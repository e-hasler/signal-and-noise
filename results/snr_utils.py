import numpy as np
from itertools import product
import pandas as pd
from tqdm import tqdm
from snr.download.hf import pull_predictions_from_hf
from IPython.display import display
from math import comb
import os


def compute_score_and_stepstd(ta, se, mi, si, dtf, metric):
    for task, seed, mix, size in tqdm(product(ta, se, mi, si), total = len(ta)*len(se)*len(mi)*len(si)):
        score_df = dtf[(dtf['task'] == task) & (dtf['mix'] == mix) &
                                (dtf['size'] == size) & (dtf['seed'] == seed)]
        
        if len(score_df) == 0:  # Skip empty combinations
            continue
        
        score_df = score_df.sort_values('step')
        curve_values = score_df[metric].values  # Get values as array, like datadecide
        
        if len(curve_values) == 0:  # Double-check after sorting
            continue

        # step_std computation
        last_vals = curve_values[int(-0.3 * len(curve_values)):]
        step_std = np.std(last_vals)
        
        # score computation
        avg_score = np.mean(curve_values[int(-0.1 * len(curve_values)):])

        # assign score and step_std to metrics_df
        dtf.loc[(dtf['task'] == task) & (dtf['mix'] == mix) &
                        (dtf['size'] == size) & (dtf['seed'] == seed), 'score'] = avg_score


        dtf.loc[(dtf['task'] == task) & (dtf['mix'] == mix) &
                (dtf['size'] == size) & (dtf['seed'] == seed), 'step_std'] = step_std # Add a new column 'step_std' with the std of the score
        

def compute_mean_std(dtf, ta, si, se):
    # Create columns for mean and std if not exist
    dtf['mean'] = np.nan
    dtf['std'] = np.nan

    # Remove 'step' and 'primary_metric' columns
    if 'step' in dtf.columns and 'primary_metric' in dtf.columns:
        dtf.drop(columns=['step', 'primary_metric'], inplace=True)
        dtf.drop_duplicates(subset = ['task', 'mix', 'size', 'seed'], keep = 'first', inplace=True)


    # Compute mean and std of scores over same task, size, seed but different mixes
    for task, seed, size in tqdm(product(ta, se, si), total = len(ta)*len(se)*len(si)):
        subset = dtf[(dtf['task'] == task) & (dtf['size'] == size) & (dtf['seed'] == seed)]
        
        if len(subset) < 2:  # skip if fewer than 2 mixes
            continue
        
        mean_score = subset['score'].mean()
        std_score = subset['score'].std()

        max_score = subset['score'].max()
        min_score = subset['score'].min()
        
        dtf.loc[(dtf['task'] == task) & (dtf['size'] == size) & 
                    (dtf['seed'] == seed), 'mean'] = mean_score
        dtf.loc[(dtf['task'] == task) & (dtf['size'] == size) & 
                    (dtf['seed'] == seed), 'std'] = std_score
        dtf.loc[(dtf['task'] == task) & (dtf['size'] == size) & 
                    (dtf['seed'] == seed), 'max'] = max_score
        dtf.loc[(dtf['task'] == task) & (dtf['size'] == size) & 
                    (dtf['seed'] == seed), 'min'] = min_score


# Compute the score of the model on each fold
def compute_score_model(benchmarks: list, all_models_names: list, nb_folds: int):
    """
    Load and return the aggregated primary metric score for a model on a fold.
    
    Reads metrics from metrics-all.jsonl where each line is one instance.
    Aggregates all instance scores by averaging them to get the fold-level metric.
    
    Args:
        benchmark: Benchmark name (e.g., 'hellaswag')
        model_name: Model name (e.g., 'dolma17-25p-DCLM-baseline-75p-150M-5xC')
        nb_folds: Number of folds (e.g., 5)
    
    Returns:
        float: List of benchmark scores for the fold, or None if loading fails.
    """
    import json

    # Nested dict: scores[model][benchmark][fold] = avg_score
    scores = {}

    for benchmark in benchmarks:
        for model_name in all_models_names:
            if model_name not in scores:
                scores[model_name] = {}
            if benchmark not in scores[model_name]:
                scores[model_name][benchmark] = {}
            for nb in range(nb_folds):
                fold_name = f"fold_{nb}"
                output_dir = os.path.join(
                    "results/k_folds",
                    f"{benchmark}_{str(fold_name)}"
                )
                final_output_dir = f"{output_dir}_{model_name}"
                metrics_file = os.path.join(final_output_dir, "metrics-all.jsonl")

                if not os.path.exists(metrics_file):
                    print(f"Warning: Metrics file not found at {metrics_file}")
                    scores[model_name][benchmark][fold_name] = None
                    continue

                fold_scores = []
                try:
                    with open(metrics_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                result = json.loads(line)
                                if 'metrics' in result and 'primary_score' in result['metrics']:
                                    fold_scores.append(result['metrics']['primary_score'])
                except Exception as e:
                    print(f"Error loading metrics from {metrics_file}: {e}")
                    scores[model_name][benchmark][fold_name] = None
                    continue

                if not fold_scores:
                    print(f"Warning: No primary_score found in {metrics_file}")
                    scores[model_name][benchmark][fold_name] = None
                else:
                    avg_score = sum(fold_scores) / len(fold_scores)
                    scores[model_name][benchmark][fold_name] = avg_score

    return scores
        
        
