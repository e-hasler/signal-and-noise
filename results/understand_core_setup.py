"""
SIGNAL-AND-NOISE PROJECT: CORE DATASET EXPLORER
================================================
This file helps understand the comprehensive 'core' split of the signal-and-noise dataset.
Run this to get an overview of:
- All available tasks (244 tasks across multiple benchmarks)
- Models evaluated
- Metrics and data structure
- Task groupings and families
- Coverage and statistics
"""

import numpy as np
import pandas as pd
from snr.download.hf import pull_predictions_from_hf
from collections import defaultdict

# Load the main core dataset
print("="*80)
print("LOADING CORE DATASET")
print("="*80)
df = pd.read_parquet(pull_predictions_from_hf("allenai/signal-and-noise", split_name='core'))

print(f"\n✓ Loaded {len(df)} evaluation results")
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================================================
# SECTION 1: BASIC STATISTICS
# ============================================================================
print("\n" + "="*80)
print("1. DATASET OVERVIEW")
print("="*80)
print(f"\nTotal evaluation instances: {len(df):,}")
print(f"Unique tasks: {df['task'].nunique()}")
print(f"Unique models: {df['model'].nunique()}")

if 'step' in df.columns:
    print(f"Unique steps/checkpoints: {df['step'].nunique()}")
if 'seed' in df.columns:
    print(f"Unique seeds: {df['seed'].nunique()}")

# ============================================================================
# SECTION 2: TASK COVERAGE BY FAMILY
# ============================================================================
print("\n" + "="*80)
print("2. TASKS BY FAMILY")
print("="*80)

tasks = sorted(df['task'].unique())

# Organize by family
task_families = defaultdict(list)
for task in tasks:
    # Extract family (first part before underscore or colon)
    if ':' in task:
        family = task.split(':')[0]
    elif '_' in task:
        family = task.split('_')[0]
    else:
        family = task
    
    task_families[family].append(task)

for family in sorted(task_families.keys()):
    tasks_list = task_families[family]
    total = sum((df['task'] == t).sum() for t in tasks_list)
    print(f"\n{family.upper()} ({len(tasks_list)} tasks, {total:,} instances):")
    
    # Show first 5 tasks
    for t in tasks_list[:5]:
        count = (df['task'] == t).sum()
        print(f"  - {t}: {count:,}")
    
    if len(tasks_list) > 5:
        print(f"  ... and {len(tasks_list) - 5} more")

# ============================================================================
# SECTION 3: MODELS AND CHECKPOINTS
# ============================================================================
print("\n" + "="*80)
print("3. MODELS AND CHECKPOINTS")
print("="*80)

models = sorted(df['model'].unique())
print(f"\n{len(models)} models found")

if 'step' in df.columns:
    print("\nModel checkpoint structure:")
    model_steps = df.groupby('model')['step'].apply(lambda x: sorted(x.unique())).to_dict()
    
    # Show first 10 models
    for i, (model, steps) in enumerate(sorted(model_steps.items())[:10]):
        print(f"\n  {model}:")
        print(f"    Steps: {steps}")
        print(f"    Total checkpoints: {len(steps)}")
    
    if len(model_steps) > 10:
        print(f"\n  ... and {len(model_steps) - 10} more models")
    
    # Statistics
    checkpoint_counts = pd.Series({m: len(s) for m, s in model_steps.items()})
    print("\n" + "-"*80)
    print("Checkpoint distribution:")
    print("-"*80)
    print(checkpoint_counts.value_counts().sort_index().to_string())
    print(f"\nAverage checkpoints per model: {checkpoint_counts.mean():.1f}")
    print(f"Max checkpoints: {checkpoint_counts.max()}")
    print(f"Min checkpoints: {checkpoint_counts.min()}")

# ============================================================================
# SECTION 4: TASK COVERAGE PER MODEL
# ============================================================================
print("\n" + "="*80)
print("4. TASK COVERAGE")
print("="*80)

# Which tasks appear in which models?
model_task_counts = df.groupby('model')['task'].nunique().sort_values(ascending=False)
print(f"\nTasks evaluated per model:")
print(f"  Average: {model_task_counts.mean():.1f}")
print(f"  Min: {model_task_counts.min()}")
print(f"  Max: {model_task_counts.max()}")

print(f"\nModels with most task coverage (top 10):")
for model, count in model_task_counts.head(10).items():
    total_evals = (df['model'] == model).sum()
    print(f"  {model}: {count} tasks ({total_evals:,} evaluations)")

# Which tasks have most model coverage?
task_model_counts = df.groupby('task')['model'].nunique().sort_values(ascending=False)
print(f"\nModels per task (top 20 most-evaluated tasks):")
for task, count in task_model_counts.head(20).items():
    total_evals = (df['task'] == task).sum()
    print(f"  {task}: {count} models ({total_evals:,} evaluations)")

# ============================================================================
# SECTION 5: METRICS AND COLUMNS
# ============================================================================
print("\n" + "="*80)
print("5. AVAILABLE METRICS")
print("="*80)

print(f"\nColumns in dataset:")
for col in df.columns:
    dtype = df[col].dtype
    try:
        nunique = df[col].nunique()
    except TypeError:
        # Column contains non-hashable types (dicts, lists, etc.)
        nunique = "N/A (complex type)"
    print(f"  {col:30s} | {str(dtype):15s} | {nunique}")

# Find metric columns
metric_cols = [c for c in df.columns if any(x in c for x in ['acc', 'score', 'metric', 'loss', 'perplexity'])]
print(f"\n{len(metric_cols)} metric columns found:")

for col in sorted(metric_cols)[:10]:
    dtype = df[col].dtype
    if pd.api.types.is_numeric_dtype(df[col]):
        non_null = df[col].notna().sum()
        mean_val = df[col].mean()
        print(f"  {col}: mean={mean_val:.4f}, non-null={non_null:,}/{len(df):,}")
    else:
        print(f"  {col}: {dtype}")

if len(metric_cols) > 10:
    print(f"  ... and {len(metric_cols) - 10} more")

# ============================================================================
# SECTION 6: DATA DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("6. EVALUATION DISTRIBUTION")
print("="*80)

print(f"\nEvaluations per task:")
task_counts = df['task'].value_counts().sort_values(ascending=False)
print(f"  Mean: {task_counts.mean():.1f}")
print(f"  Median: {task_counts.median():.1f}")
print(f"  Std: {task_counts.std():.1f}")
print(f"  Min: {task_counts.min()}")
print(f"  Max: {task_counts.max()}")

print(f"\nEvaluations per model:")
model_counts = df['model'].value_counts().sort_values(ascending=False)
print(f"  Mean: {model_counts.mean():.1f}")
print(f"  Median: {model_counts.median():.1f}")
print(f"  Min: {model_counts.min()}")
print(f"  Max: {model_counts.max()}")

# ============================================================================
# SECTION 7: SPECIAL TASK GROUPS
# ============================================================================
print("\n" + "="*80)
print("7. DEFINED TASK GROUPS IN THIS DATASET")
print("="*80)

# Import task groups
try:
    from snr.constants.tasks import (
        OLMES, MMLU, MMLU_PRO, BBH, AGI_EVAL, 
        MULTITASK_MATH, MULTITASK_CODE, MULTITASK_KNOWLEDGE
    )
    
    groups = {
        'OLMES': OLMES,
        'MMLU (all subtasks)': MMLU,
        'MMLU_PRO': MMLU_PRO,
        'BBH': BBH,
        'AGI_EVAL': AGI_EVAL,
        'MULTITASK_MATH': MULTITASK_MATH,
        'MULTITASK_CODE': MULTITASK_CODE,
        'MULTITASK_KNOWLEDGE': MULTITASK_KNOWLEDGE,
    }
    
    for group_name, group_tasks in groups.items():
        # Check which tasks from this group are in the core dataset
        available = [t for t in group_tasks if t in tasks]
        missing = [t for t in group_tasks if t not in tasks]
        
        total_instances = sum((df['task'] == t).sum() for t in available)
        
        print(f"\n{group_name}:")
        print(f"  Available: {len(available)}/{len(group_tasks)}")
        print(f"  Total instances: {total_instances:,}")
        
        if missing and len(missing) <= 5:
            print(f"  Missing: {missing}")
        elif missing:
            print(f"  Missing: {len(missing)} tasks")
except ImportError:
    print("Could not import task groups from snr.constants.tasks")

# ============================================================================
# SECTION 8: RECOMMENDATIONS FOR K-FOLD ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8. RECOMMENDATIONS FOR K-FOLD ANALYSIS")
print("="*80)

print("""
The 'core' split contains comprehensive results for:
  ✓ 244 different tasks/formats
  ✓ All major benchmarks (MMLU, BBH, OLMES, AGI_EVAL, etc.)
  ✓ Multiple models with checkpoint data
  ✓ ~389k total evaluation instances

For k-fold noise analysis, you can:

1. START SMALL:
   - Pick 1 benchmark (e.g., MMLU) with high coverage
   - Pick 1 model with multiple checkpoints
   - Run k-fold analysis to validate methodology

2. SCALE UP:
   - Add more models
   - Expand to other benchmarks (BBH, OLMES, AGI_EVAL)
   - Compare noise patterns across task families

3. CHOOSE TASKS STRATEGICALLY:
   - Look at task_model_counts to find well-covered benchmarks
   - Prioritize tasks with many model evaluations
   - Mix different task families (knowledge, math, reasoning)

IMPORTANT: Core vs. datadecide_intermediate:
   - core: Comprehensive results for all tasks/models
   - datadecide_intermediate: Focused subset for checkpoint noise analysis
   
Choose based on your research goal:
   - Broad benchmark noise: use 'core'
   - Checkpoint-specific noise: use 'datadecide_intermediate'
""")

# ============================================================================
# SECTION 9: QUICK REFERENCE
# ============================================================================
print("\n" + "="*80)
print("9. QUICK REFERENCE QUERIES")
print("="*80)

print("""
Get all tasks in a family:
  df[df['task'].str.startswith('mmlu_')]

Get evaluations for one model on one task:
  df[(df['model'] == 'MODEL_NAME') & (df['task'] == 'TASK_NAME')]

Get tasks with full model coverage:
  tasks_full_coverage = task_model_counts[task_model_counts == len(models)]

Get models that evaluate on a specific task:
  df[df['task'] == 'TASK_NAME']['model'].unique()

Get all tasks in one benchmark family:
  df[df['task'].str.contains('mmlu')]['task'].unique()

Compare metrics across tasks:
  df.groupby('task')['primary_score'].agg(['mean', 'std', 'count'])

Get checkpoint structure for a model:
  steps = df[df['model'] == 'MODEL']['step'].unique()
  sorted(steps)
""")

# ============================================================================
# SECTION 10: SAMPLE DATA
# ============================================================================
print("\n" + "="*80)
print("10. SAMPLE DATA (First 5 rows)")
print("="*80)

display_cols = ['model', 'task', 'step', 'primary_score']
display_cols = [c for c in display_cols if c in df.columns]
if display_cols:
    print("\n", df[display_cols].head(5).to_string(index=False))

print("\n" + "="*80)
print("END OF CORE SETUP OVERVIEW")
print("="*80)
