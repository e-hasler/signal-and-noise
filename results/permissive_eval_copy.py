"""
Wrapper that registers FlexibleHellaswag task, then calls oe_eval
"""
import os
from typing import Optional
from datasets import load_dataset, load_from_disk, DatasetDict
from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY

print(type(TASK_REGISTRY))

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
#TASK_REGISTRY['hellaswag_flexible'] = FlexibleHellaswag  # type: ignore
#print("✓ Registered hellaswag_flexible task")
