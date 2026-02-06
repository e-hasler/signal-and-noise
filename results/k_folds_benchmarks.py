"""
Create k-folds on benchmarks and push to HuggingFace hub
"""

import numpy as np
import pandas as pd
from snr.download.hf import pull_predictions_from_hf
from datasets import load_dataset, DatasetDict, DownloadConfig
import random
import os
import subprocess
import math
from snr_utils import compute_score_model



# Pull all the model evaluations used in the project to find models and compute noise and decision accuracy
df = pd.read_parquet(pull_predictions_from_hf("allenai/signal-and-noise", split_name='datadecide_intermediate'))

# Split into k folds and save as custom HuggingFace dataset
#DatasetDict with k splits named 'fold_0', 'fold_1', ..., 'fold_{k-1}'
def make_k_folds_dataset(tasks_list, k: int, seed: int = 0, save_path = None, force_recompute: bool = False):

    # Check if folds already made
    if save_path and os.path.exists(save_path) and not force_recompute:
        print(f"Loading cached k-fold dataset from {save_path}")
        dataset_dict = DatasetDict.load_from_disk(save_path)
        print(f"Loaded {len(dataset_dict)} folds from cache")
        for fold_name, fold_data in dataset_dict.items():
            print(f"  {fold_name}: {len(fold_data)} instances")
        return dataset_dict
    
    print(f"Creating k-fold dataset (k={k}, seed={seed})...")
    
    # Shuffle the dataset
    shuffled = tasks_list.shuffle(seed=seed)
    
    total_size = len(shuffled)
    fold_size = total_size // k
    
    folds_dict = {}
    
    for i in range(k):
        start_idx = i * fold_size
        # Last fold gets remaining items
        end_idx = start_idx + fold_size if i < k - 1 else total_size
        
        fold_indices = list(range(start_idx, end_idx))
        fold_data = shuffled.select(fold_indices)
        folds_dict[f'fold_{i}'] = fold_data
        
        print(f"Fold {i}: {len(fold_data)} instances (indices {start_idx}-{end_idx-1})")
    
    dataset_dict = DatasetDict(folds_dict)
    
    # Save to disk if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dataset_dict.save_to_disk(save_path)
        print(f"Saved k-fold dataset to {save_path}")
    
    return dataset_dict

# Extract last n checkpoints for each model
def get_last_n_checkpoints(model_steps_dict, n):
    """
    Get the last n checkpoints for each model.
    
    Returns:
        dict: {model_name: [list of last n steps]}
    """
    last_checkpoints = {}
    for model, steps in model_steps_dict.items():
        # Take the last n steps
        last_checkpoints[model] = sorted(steps)[-n:]
    return last_checkpoints

# Run model on the folds TODO: generalize to other datasets
def run_model_on_fold(model: str, task_name: str, fold_dataset_path: str, fold_name: str, output_dir: str, revision = None):
    """
    Run evaluation on a single fold using custom dataset split.
    
    Args:
        model: HuggingFace model name (e.g., 'allenai/DataDecide-dclm-baseline-150M')
        task_name: Task name from OLMES registry (e.g., 'hellaswag')
        fold_dataset_path: Path to the saved fold dataset (local directory)
        fold_name: Name of the fold split (e.g., 'fold_0')
        output_dir: Directory for results
        revision: Optional HF revision/tag (e.g., 'step38750')
    
    Returns:
        output_dir if successful, None otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the oe-eval command using local dataset path and split
    cmd = [
        "python", "./deps/olmes/oe_eval/run_eval.py",
        "--model", model,
        "--task", task_name,
        "--split", fold_name,  # Use the custom fold split
        "--output-dir", output_dir,
        "--dataset-path data/k_folds/hellaswag_k5_seed0/"
        "--limit 0.01"
    ]

    if revision:
        cmd.extend(["--revision", revision])
    
    print(f"\nRunning evaluation on {fold_name}...")
    print(f"Dataset path: {fold_dataset_path}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"Error running {fold_name}:")
        print(result.stderr)
        return None
    
    print(f"✓ {fold_name} completed")
    return output_dir


# For each model, average the score on the last n-checkpoints to reduce noise
def smooth_scores(results_df, model, fold_name, last_n_steps, metric='primary_metric'):
    """
    Compute S_{i,f} = (1/n) * sum of scores for last n checkpoints.
    
    Args:
        results_df: DataFrame with evaluation results
        model: Model name
        fold_name: Fold identifier (e.g., 'fold_0')
        last_n_steps: List of step numbers to average over
        metric: Metric to use for scoring
    
    Returns:
        float: Smoothed score
    """
    # Filter results for this model, fold, and steps
    mask = (results_df['model'] == model) & \
           (results_df['fold'] == fold_name) & \
           (results_df['step'].isin(last_n_steps))
    
    scores = results_df[mask][metric].values
    
    if len(scores) == 0:
        print(f"Warning: No scores found for {model} on {fold_name}")
        return np.nan
    
    return np.mean(scores)

# Compute the standard deviation of the averaged scores on each fold for each model
def get_std_benchmark(smoothed_scores_per_fold):
    """
    Compute Var_fold(i) = (1/(k-1)) * sum((S_{i,f} - mean(S_{i,f}))^2).
    
    Args:
        smoothed_scores_per_fold: List of smoothed scores [S_{i,f1}, S_{i,f2}, ...]
    
    Returns:
        float: Fold variance
    """
    scores = np.array(smoothed_scores_per_fold)
    k = len(scores)
    
    if k <= 1:
        return np.nan
    
    mean_score = np.mean(scores)
    variance = np.sum((scores - mean_score)**2) / (k - 1)
    
    return math.sqrt(variance)

# Visualize correlation with  seed noise, checkpoint-to-checkpoint noise, decision accuracy
def visualize():
    pass

# Retrieve model name, score from results folder
def main():
    # Configuration
    benchmarks = ["hellaswag", "arc_easy", "arc_challenge",'boolq','csqa', \
                  'openbookqa', 'piqa', 'socialiqa', 'winogrande', 'mmlu']
    k = 5
    n_last_checkpoints = [1]
    
    for i, benchmark in enumerate(benchmarks):
        # Get the tasks on the benchmark
        if benchmark == "hellaswag":
            tasks = load_dataset('Rowan/hellaswag', split='validation')
        elif benchmark == "arc_easy":
            tasks = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='validation') 
        elif benchmark == "arc_challenge":
            tasks = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='validation') 
        elif benchmark == "boolq":
            tasks = load_dataset('google/boolq', split='validation', trust_remote_code=True)
        elif benchmark == "csqa":
            tasks = load_dataset('commonsense_qa', split='validation', trust_remote_code=True)
        elif benchmark == "openbookqa":
            tasks = load_dataset('openbookqa', split='validation', trust_remote_code=True)
        elif benchmark == "piqa":
            tasks = load_dataset('piqa', split='validation', trust_remote_code=True)
        elif benchmark == "socialiqa":
            tasks = load_dataset('social_i_qa', split='validation', trust_remote_code=True)
        elif benchmark == "winogrande":
            tasks = load_dataset('winogrande', 'winogrande_debiased', split='validation', trust_remote_code=True)
        elif benchmark == "mmlu":
            tasks = load_dataset('cais/mmlu', 'all', split='dev', trust_remote_code=True)

        print(f"\n{'='*60}")
        print(f"Processing benchmark: {benchmark}")
        print(f"Total instances: {len(tasks)}")  # type: ignore
        print(f"{'='*60}")
        
        # Create and save k-fold dataset
        fold_dataset_path = f"data/k_folds/{benchmark}_k{k}_seed0"
        folds_dataset = make_k_folds_dataset(tasks, k, seed=0, save_path=fold_dataset_path) # type: ignore
        
        # TODO: Modify the name to push other datasets
        # TODO: Add flag so that it does not push again the dataset
        #folds_dataset.push_to_hub(repo_id = "ehasler/hellaswag-k5-folds", private = True)
        #if benchmark == "arc_easy":
        #    folds_dataset.push_to_hub(repo_id = "ehasler/arc-easy-k5-folds", private = True)
        #if benchmark == "arc_challenge":
        #    folds_dataset.push_to_hub(repo_id = "ehasler/arc-challenge-k5-folds", private = True)
        #if benchmark not in ["hellaswag", "arc_easy", "arc_challenge"]:
            #folds_dataset.push_to_hub(repo_id = f"ehasler/{benchmark}-k5-folds", private = True) #TODO: verify


        """
        # Get last n checkpoints for the model
        if 'step' in df.columns:
            # bench_df contains all models evaluated on {benchmark}
            bench_df = df[df['task'] == benchmark]
            # model_steps contains the list of all checkpoints of those models
            model_steps = bench_df.groupby('model')['step'].apply(lambda x: sorted(x.unique())).to_dict()
            
            last_checkpoints = get_last_n_checkpoints(model_steps, n=n_last_checkpoints[i])
            
            print(f"\n{'='*60}")
            print(f"Last {n_last_checkpoints[i]} checkpoints per model (first 5 shown):")
            print(f"{'='*60}")
            for model, steps in sorted(last_checkpoints.items())[:5]:  # Show first 5 models
                print(f"{model}: {steps}")
        else:
            print("Warning: No 'step' column found. Cannot select last checkpoints.")
            return
        
        # Select which models to evaluate (example: take first model for testing)
        models_to_evaluate = list(last_checkpoints.keys())[:1]  # TODO: Change to all models
        
        print(f"\n{'='*60}")
        print(f"Will evaluate {len(models_to_evaluate)} models on {k} folds")
        print(f"{'='*60}")


        # Run evaluations for each model checkpoint on each fold
        for model_base in models_to_evaluate:
            checkpoints = last_checkpoints[model_base]
            
            print(f"\nEvaluating model: {model_base}")
            print(f"  Checkpoints: {checkpoints}")
            # For each checkpoint
            for step in checkpoints:
                # Use base repo + HF revision tag (steps are stored as revisions)
                # TODO: Modify for all benchmarks
                #model_repo = f"allenai/DataDecide-{model_base}"
                model_repo = 'allenai/DataDecide-dclm-baseline-20M'
                #model_revision = f"step{step}"
                model_revision = 'step14594-seed-default'                
            
                
                # Prepare fold and output directory to run evaluations
                # TODO: Change name of output_dir for any benchmark or model_base
                for fold_name in folds_dataset.keys():
                    output_dir = os.path.join(
                        "results/k_folds",
                        f"smoke_hellaswag_dclm-baseline-20M",
                        f"step_{step}",
                        str(fold_name)
                    )
                    
                    # Uncomment to actually evaluate a model on each fold
                    
                    print(f"  Running {fold_name} at step {step}...")

                    result = run_model_on_fold(
                        model=model_repo,
                        task_name='hellaswag',
                        fold_dataset_path=fold_dataset_path,
                        fold_name=str(fold_name),
                        output_dir=output_dir,
                        revision=model_revision
                    )
                    
                    if result:
                        print(f"    ✓ Results saved to {output_dir}")
"""
        """
        # After all evaluations, compute smoothed scores and fold variance
        print(f"\n{'='*60}")
        print("Computing smoothed scores and fold variance...")
        print(f"{'='*60}")
        """
        # TODO: Load results and compute metrics
        # This would involve:
        # 1. Loading all result files
        # 2. For each model and fold, compute smoothed score across last n checkpoints
        # 3. For each model, compute fold variance across all folds
        # 4. Save/visualize results
        """


        """
if __name__ == "__main__":
    main()









# ================== EXTRA ======================
# Uncomment to check which models and checkpoints (steps) have been evaluated on hellaswag bechmark
"""
hellaswag_df = df[df['task'] == 'hellaswag']
cols_to_show = ['model']
if 'step' in df.columns:
    cols_to_show.append('step')
if 'checkpoint' in df.columns:
    cols_to_show.append('checkpoint')

print("\nHellaSwag models:")
print(hellaswag_df[cols_to_show].drop_duplicates().sort_values('model'))
"""

# Uncomment to check which models and checkpoints (steps) have been evaluated on arc_easy bechmark
"""
arc_easy_df = df[df['task'] == 'arc_easy']
cols_to_show = ['model']
if 'step' in df.columns:
    cols_to_show.append('step')
if 'checkpoint' in df.columns:
    cols_to_show.append('checkpoint')

print("\nArcEasy models:")
print(arc_easy_df[cols_to_show].drop_duplicates().sort_values('model'))
"""

# Show steps for each model (if 'step' column exists)
"""
if 'step' in df.columns:
    print("\nSteps per model:")
    model_steps = hellaswag_df.groupby('model')['step'].apply(lambda x: sorted(x.unique())).to_dict()
    for model, steps in sorted(model_steps.items()):
        print(f"{model}: {steps}")
else:
    print("\nNo 'step' column found in dataframe")
"""
