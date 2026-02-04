"""
Wrapper that registers FlexibleHellaswag task, then calls oe_eval
"""
import os
from typing import Optional
from datasets import load_dataset, load_from_disk, DatasetDict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deps.olmes.oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY

# Get the original HellaSwag task class
OriginalHellaswag = TASK_REGISTRY['hellaswag']
OriginalARCEasy = TASK_REGISTRY['arc_easy']

# Define and register custom hellaswag task
class FlexibleHellaswag(OriginalHellaswag):
    """HellaSwag task that accepts custom splits like fold_0, fold_1"""
    
    def __init__(self, *args, **kwargs):
        # Store the original split request before calling parent
        self._fold_split = kwargs.get('task_config', {}).get('split')
        super().__init__(*args, **kwargs)
    
    # Method to override to download dataset from a custom API.
    def download(self, data_dir: Optional[str] = None, cache_dir: Optional[str] = None, download_mode=None):  # type: ignore
        """Load fold-specific dataset and present it as the validation split.

        If split is fold_*, we try to load from local saved DatasetDict at
        data/k_folds/hellaswag_k5_seed0; if not present, fall back to HF hub
        repo ehasler/hellaswag-k5-folds. Otherwise, use the parent download.
        """
        split = self.task_config.get("split")
        if isinstance(split, str) and split.startswith("fold_"):
            fold_name = split
            local_path = os.path.join("data", "k_folds", "hellaswag_k5_seed0")
            ds = None
            if os.path.exists(local_path):
                try:
                    dd = load_from_disk(local_path)
                    if fold_name not in dd:
                        raise KeyError(f"Fold {fold_name} not found in {local_path}")
                    ds = dd[fold_name]
                    print(f"✓ Loaded {fold_name} from local: {len(ds)} instances")
                except Exception as e:
                    print(f"Failed to load local folds from {local_path}: {e}")
            if ds is None:
                try:
                    ds = load_dataset("ehasler/hellaswag-k5-folds", split=fold_name)
                    print(f"✓ Loaded {fold_name} from HF hub: {len(ds)} instances")
                except Exception as e:
                    print(f"Failed to load HF dataset ehasler/hellaswag-k5-folds:{fold_name}: {e}")
                    # As a last resort, fallback to parent download and normal validation
                    return super().download(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)
            # Expose fold as validation split
            self.dataset = DatasetDict({"validation": ds})  # type: ignore
            # Force split to validation for downstream logic
            self.task_config["split"] = "validation"
            return
        # Default behavior for non-fold splits
        return super().download(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)

# Register the custom task in TASK_REGISTRY
TASK_REGISTRY['hellaswag_flexible'] = FlexibleHellaswag  # type: ignore
print("✓ Registered hellaswag_flexible task")

class FlexibleARCEasy(OriginalARCEasy):
    """ARCEasy task that accepts custom splits like fold_0, fold_1"""
    
    def __init__(self, *args, **kwargs):
        # Store the original split request before calling parent
        self._fold_split = kwargs.get('task_config', {}).get('split')
        super().__init__(*args, **kwargs)
    
    # Method to override to download dataset from a custom API.
    def download(self, data_dir: Optional[str] = None, cache_dir: Optional[str] = None, download_mode=None):  # type: ignore
        """Load fold-specific dataset and present it as the validation split.

        If split is fold_*, we try to load from local saved DatasetDict at
        data/k_folds/hellaswag_k5_seed0; if not present, fall back to HF hub
        repo ehasler/arc-easy-k5-folds. Otherwise, use the parent download.
        """
        split = self.task_config.get("split")
        if isinstance(split, str) and split.startswith("fold_"):
            fold_name = split
            local_path = os.path.join("data", "k_folds", "arc_easy_k5_seed0")
            ds = None
            if os.path.exists(local_path):
                try:
                    dd = load_from_disk(local_path)
                    if fold_name not in dd:
                        raise KeyError(f"Fold {fold_name} not found in {local_path}")
                    ds = dd[fold_name]
                    print(f"✓ Loaded {fold_name} from local: {len(ds)} instances")
                except Exception as e:
                    print(f"Failed to load local folds from {local_path}: {e}")
            if ds is None:
                try:
                    ds = load_dataset("ehasler/arc-easy-k5-folds", split=fold_name)
                    print(f"✓ Loaded {fold_name} from HF hub: {len(ds)} instances")
                except Exception as e:
                    print(f"Failed to load HF dataset ehasler/arc-easy-k5-folds:{fold_name}: {e}")
                    # As a last resort, fallback to parent download and normal validation
                    return super().download(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)
            # Expose fold as validation split
            self.dataset = DatasetDict({"validation": ds})  # type: ignore
            # Force split to validation for downstream logic
            self.task_config["split"] = "validation"
            return
        # Default behavior for non-fold splits
        return super().download(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)



# Register the custom arc easy task in TASK_REGISTRY
TASK_REGISTRY['arc_easy_flexible'] = FlexibleARCEasy  # type: ignore
print("✓ Registered arc_easy_flexible task")

# Get task classes for remaining benchmarks
OriginalBoolq = TASK_REGISTRY.get('boolq')
OriginalCSQA = TASK_REGISTRY.get('csqa')
OriginalOpenBookQA = TASK_REGISTRY.get('openbookqa')
OriginalPIQA = TASK_REGISTRY.get('piqa')
OriginalSocialIQA = TASK_REGISTRY.get('socialiqa')  # Note: not 'social_i_qa'
OriginalWinogrande = TASK_REGISTRY.get('winogrande')
# OriginalMMLU is skipped - MMLU in oe_eval is per-subject, not a single task

# Create flexible wrapper for each task
def create_flexible_task(task_name: str, original_task_class):
    """Factory function to create flexible task wrapper"""
    if original_task_class is None:
        print(f"⚠ Warning: Task {task_name} not found in TASK_REGISTRY")
        return None
    
    class FlexibleTask(original_task_class):
        def __init__(self, *args, **kwargs):
            self._fold_split = kwargs.get('task_config', {}).get('split')
            super().__init__(*args, **kwargs)
        
        def download(self, data_dir: Optional[str] = None, cache_dir: Optional[str] = None, download_mode=None):  # type: ignore
            split = self.task_config.get("split")
            if isinstance(split, str) and split.startswith("fold_"):
                fold_name = split
                local_path = os.path.join("data", "k_folds", f"{task_name}_k5_seed0")
                ds = None
                if os.path.exists(local_path):
                    try:
                        dd = load_from_disk(local_path)
                        if fold_name not in dd:
                            raise KeyError(f"Fold {fold_name} not found in {local_path}")
                        ds = dd[fold_name]
                        print(f"✓ Loaded {fold_name} from local: {len(ds)} instances")
                    except Exception as e:
                        print(f"Failed to load local folds from {local_path}: {e}")
                if ds is None:
                    try:
                        ds = load_dataset(f"ehasler/{task_name}-k5-folds", split=fold_name)
                        print(f"✓ Loaded {fold_name} from HF hub: {len(ds)} instances")
                    except Exception as e:
                        print(f"Failed to load HF dataset ehasler/{task_name}-k5-folds:{fold_name}: {e}")
                        return super().download(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)
                
                # Add missing idx/id field if needed
                native_id_field = self.task_config.get("native_id_field") or "idx"
                if native_id_field not in ds.column_names: # type: ignore
                    ds = ds.map(lambda example, idx: {**example, native_id_field: idx}, with_indices=True) # type: ignore
                    print(f"✓ Added {native_id_field} field to fold dataset")
                
                # Remap fields for compatibility with oe_eval tasks
                field_mapping = {
                    'boolq': {'answer': 'label'},  # boolq uses 'answer' but task expects 'label'
                }
                if task_name in field_mapping:
                    rename_map = field_mapping[task_name]
                    for old_name, new_name in rename_map.items():
                        if old_name in ds.column_names and new_name not in ds.column_names: # type: ignore
                            ds = ds.rename_column(old_name, new_name) # type: ignore
                            print(f"✓ Renamed {old_name} → {new_name}")
                
                self.dataset = DatasetDict({"validation": ds})  # type: ignore
                self.task_config["split"] = "validation"
                return
            return super().download(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)
    
    return FlexibleTask

# Register flexible tasks
if OriginalBoolq:
    FlexibleBoolq = create_flexible_task('boolq', OriginalBoolq)
    TASK_REGISTRY['boolq_flexible'] = FlexibleBoolq  # type: ignore
    print("✓ Registered boolq_flexible task")

if OriginalCSQA:
    FlexibleCSQA = create_flexible_task('csqa', OriginalCSQA)
    TASK_REGISTRY['csqa_flexible'] = FlexibleCSQA  # type: ignore
    print("✓ Registered csqa_flexible task")

if OriginalOpenBookQA:
    FlexibleOpenBookQA = create_flexible_task('openbookqa', OriginalOpenBookQA)
    TASK_REGISTRY['openbookqa_flexible'] = FlexibleOpenBookQA  # type: ignore
    print("✓ Registered openbookqa_flexible task")

if OriginalPIQA:
    FlexiblePIQA = create_flexible_task('piqa', OriginalPIQA)
    TASK_REGISTRY['piqa_flexible'] = FlexiblePIQA  # type: ignore
    print("✓ Registered piqa_flexible task")

if OriginalSocialIQA:
    FlexibleSocialIQA = create_flexible_task('socialiqa', OriginalSocialIQA)
    TASK_REGISTRY['socialiqa_flexible'] = FlexibleSocialIQA  # type: ignore
    print("✓ Registered socialiqa_flexible task")

if OriginalWinogrande:
    FlexibleWinogrande = create_flexible_task('winogrande', OriginalWinogrande)
    TASK_REGISTRY['winogrande_flexible'] = FlexibleWinogrande  # type: ignore
    print("✓ Registered winogrande_flexible task")

# MMLU is skipped - it's per-subject tasks (mmlu_abstract_algebra, etc.), not a single task

# Now call the actual evaluator with the registered task available
if __name__ == "__main__":
    from oe_eval.run_eval import run_eval
    from oe_eval import run_eval as run_eval_module
    
    # Parse arguments using oe_eval's parser
    args = run_eval_module._parser.parse_args()
    args_dict = vars(args)
    
    # Run with the custom task registered
    run_eval(args_dict)
    