"""
DATADECIDE PROJECT DATA STRUCTURE EXPLORER
==========================================
This file helps understand the structure and quirks of the signal-and-noise project data.
Run this to get a comprehensive overview of:
- DataFrame structure and columns
- Task naming conventions
- Metric types and locations
- Model/checkpoint organization
- Common pitfalls and particularities
"""

import numpy as np
import pandas as pd
from snr.download.hf import pull_predictions_from_hf

# Load the main results dataset
print("="*80)
print("LOADING DATADECIDE RESULTS")
print("="*80)
df = pd.read_parquet(pull_predictions_from_hf("allenai/signal-and-noise", split_name='datadecide_intermediate'))

print(f"\n✓ Loaded {len(df)} evaluation results")
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================================================
# SECTION 1: DATAFRAME STRUCTURE
# ============================================================================
print("\n" + "="*80)
print("1. DATAFRAME COLUMNS AND TYPES")
print("="*80)
print("\nAll columns:")
for col in df.columns:
    dtype = df[col].dtype
    nunique = df[col].nunique()
    print(f"  {col:30s} | {str(dtype):15s} | {nunique:,} unique values")

print("\n" + "-"*80)
print("Key Dimensions (MIXES, MODELS, TASKS, SEEDS, STEPS):")
print("-"*80)
for dim_col in ['model', 'task', 'step', 'seed']:
    if dim_col in df.columns:
        n = df[dim_col].nunique()
        print(f"  {dim_col:15s}: {n:,} unique values")
        if n < 20:
            print(f"    Values: {sorted(df[dim_col].unique())}")

# ============================================================================
# SECTION 2: TASK NAMING CONVENTIONS
# ============================================================================
print("\n" + "="*80)
print("2. TASK NAMING AND IDS")
print("="*80)

if 'task' in df.columns:
    tasks = df['task'].unique()
    print(f"\n{len(tasks)} unique tasks found:")
    
    # Group by task family
    task_families = {}
    for task in sorted(tasks):
        # Extract task family (before underscore or dash)
        if '_' in task:
            family = task.split('_')[0]
        else:
            family = task
        
        if family not in task_families:
            task_families[family] = []
        task_families[family].append(task)
    
    for family, tasks_in_family in sorted(task_families.items()):
        print(f"\n  {family.upper()} ({len(tasks_in_family)} tasks):")
        for t in tasks_in_family[:5]:  # Show first 5
            print(f"    - {t}")
        if len(tasks_in_family) > 5:
            print(f"    ... and {len(tasks_in_family) - 5} more")

print("\nTask naming convention:")
print("  - Simple names: 'hellaswag', 'arc_easy', 'winogrande'")
print("  - MMLU subtasks: 'mmlu_abstract_algebra', 'mmlu_anatomy', etc.")
print("  - No format/regime/version suffixes in task column")
print("  - Instance IDs typically in 'id' or 'native_id' column")

# Check for ID columns
id_cols = [c for c in df.columns if 'id' in c.lower()]
if id_cols:
    print(f"\nID columns found: {id_cols}")
    for col in id_cols:
        print(f"  {col}: {df[col].dtype}, {df[col].nunique()} unique")

# ============================================================================
# SECTION 3: METRICS EXPLAINED
# ============================================================================
print("\n" + "="*80)
print("3. METRICS: TYPES, LOCATIONS, AND MEANINGS")
print("="*80)

# Find all metric columns
metric_cols = [c for c in df.columns if any(x in c for x in ['acc', 'score', 'metric', 'f1', 'bleu', 'rouge'])]
print(f"\n{len(metric_cols)} metric columns found:")

for col in sorted(metric_cols):
    dtype = df[col].dtype
    if pd.api.types.is_numeric_dtype(df[col]):
        non_null = df[col].notna().sum()
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"\n  {col}:")
        print(f"    Type: {dtype}")
        print(f"    Non-null: {non_null:,} / {len(df):,} ({100*non_null/len(df):.1f}%)")
        print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"    Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
    else:
        print(f"\n  {col}: {dtype} (non-numeric)")

print("\n" + "-"*80)
print("Common Metrics Explained:")
print("-"*80)
print("""
  primary_score     : The main metric for the task (varies by task)
  acc_raw           : Raw accuracy (0 or 1 per instance)
  acc_per_token     : Accuracy normalized by token count
  acc_per_char      : Accuracy normalized by character count (used by MMLU)
  acc_per_byte      : Accuracy normalized by byte count
  sum_logits_corr   : Sum of log probabilities for correct answer
  logits_per_*_corr : Logits normalized by token/char/byte for correct answer
  bits_per_byte_corr: Information-theoretic metric (bits per byte)
  acc_uncond        : Unconditional accuracy baseline
  
  NOTE: Different tasks use different primary metrics!
        - HellaSwag: typically acc_per_token
        - MMLU: typically acc_per_char
        - Check task config or first instance to determine primary metric
""")

# ============================================================================
# SECTION 4: MODEL AND CHECKPOINT ORGANIZATION
# ============================================================================
print("\n" + "="*80)
print("4. MODEL AND CHECKPOINT STRUCTURE")
print("="*80)

if 'model' in df.columns and 'step' in df.columns:
    print("\nModels with training checkpoints:")
    model_steps = df.groupby('model')['step'].apply(lambda x: sorted(x.unique())).to_dict()
    
    print(f"\n{len(model_steps)} models found")
    print("\nCheckpoint structure examples (first 5 models):")
    for i, (model, steps) in enumerate(sorted(model_steps.items())[:5]):
        print(f"\n  {model}:")
        print(f"    Steps: {steps}")
        print(f"    Total checkpoints: {len(steps)}")
        
        # Check if steps are evenly spaced
        if len(steps) > 1:
            diffs = [steps[i+1] - steps[i] for i in range(len(steps)-1)]
            if len(set(diffs)) == 1:
                print(f"    Spacing: uniform ({diffs[0]} steps)")
            else:
                print(f"    Spacing: non-uniform (varies from {min(diffs)} to {max(diffs)})")
    
    # Count models by number of checkpoints
    print("\n" + "-"*80)
    print("Checkpoint counts across models:")
    print("-"*80)
    checkpoint_counts = pd.Series({m: len(s) for m, s in model_steps.items()})
    print(checkpoint_counts.value_counts().sort_index().to_string())
    
    # Identify model families
    print("\n" + "-"*80)
    print("Model size/family structure:")
    print("-"*80)
    print("""
    Model naming convention: typically includes size (10M, 150M, 1B, etc.)
    Example: 'DCLM-baseline-150M-5xC' = 150M params, DCLM baseline, 5x compute
    
    IMPORTANT: Different sizes are DIFFERENT models, not checkpoints!
      - DCLM-baseline-150M and DCLM-baseline-1B are separate models
      - 'step' column tracks training checkpoints WITHIN a model
      - Checkpoint-to-checkpoint noise = variance across steps for one model
      - Seed noise = variance across different random seeds for same config
    """)

# ============================================================================
# SECTION 5: COMMON PITFALLS AND PARTICULARITIES
# ============================================================================
print("\n" + "="*80)
print("5. COMMON PITFALLS AND PROJECT QUIRKS")
print("="*80)
print("""
1. Task names in data vs. OLMES registry:
   - DataFrame uses simple names: 'hellaswag', 'mmlu_anatomy'
   - No format/regime/version suffixes needed for task specification
   - OLMES registry keys match these simple names

2. Metric inconsistency across tasks:
   - Different tasks use different primary metrics!
   - Always check primary_score or task config
   - Don't assume 'accuracy' means the same thing across tasks

3. Instance-level vs. aggregate scores:
   - Results in metrics-all.jsonl: ONE LINE PER INSTANCE
   - Each line has individual instance score
   - Fold-level score = mean of all instance scores
   - Don't confuse with aggregate summary metrics

4. Checkpoint access on HuggingFace:
   - Repo format: allenai/{model_base}
   - Checkpoint format: use --revision step{N}
   - NOT separate repos like allenai/{model}-step{N}

5. DataFrame dimensions:
   - Each row = one instance evaluation (question-level)
   - Multiple rows per (model, task, step) combination
   - Group by (model, task, step) to get aggregate metrics

6. Missing data patterns:
   - Not all models evaluated on all tasks
   - Not all models have multiple checkpoints
   - Check for nulls before computing metrics

7. K-fold evaluation specifics:
   - Custom fold splits saved as HuggingFace DatasetDict
   - Use --split fold_0 (not --split 0 or --split=fold_0)
   - Results stored per fold, per checkpoint, per model
""")

# ============================================================================
# SECTION 6: QUICK REFERENCE QUERIES
# ============================================================================
print("\n" + "="*80)
print("6. QUICK REFERENCE: COMMON QUERIES")
print("="*80)

print("\nExample 1: Get all hellaswag results for one model")
print("  df[(df['task'] == 'hellaswag') & (df['model'] == 'DCLM-baseline-150M-5xC')]")

print("\nExample 2: Compute checkpoint noise for a model on a task")
print("  h_df = df[(df['task'] == 'hellaswag') & (df['model'] == 'my_model')]")
print("  noise = h_df.groupby('step')['primary_score'].mean().std()")

print("\nExample 3: List all models evaluated on a specific task")
print("  df[df['task'] == 'mmlu_anatomy']['model'].unique()")

print("\nExample 4: Get last N checkpoints for a model")
print("  steps = df[df['model'] == 'my_model']['step'].unique()")
print("  last_n = sorted(steps)[-5:]")

print("\nExample 5: Check which metric is primary for a task")
print("  # Load task config or check first result's metrics-all.jsonl")
print("  # Look for 'task_config.primary_metric' field")

# ============================================================================
# SECTION 7: SAMPLE DATA INSPECTION
# ============================================================================
print("\n" + "="*80)
print("7. SAMPLE DATA (First 5 rows)")
print("="*80)

# Display subset of important columns
display_cols = ['model', 'task', 'step', 'primary_score']
display_cols = [c for c in display_cols if c in df.columns]
if display_cols:
    print("\n", df[display_cols].head(5).to_string(index=False))

print("\n" + "="*80)
print("END OF DATA STRUCTURE OVERVIEW")
print("="*80)
print("\nTo dive deeper into specific aspects:")
print("  - Checkpoint noise: focus on 'step' column variance")
print("  - Task comparison: group by 'task' and compare metrics")
print("  - Model families: parse 'model' column for size/config patterns")
print("  - Fold variance: requires k-fold evaluation results (not in this dataset)")
