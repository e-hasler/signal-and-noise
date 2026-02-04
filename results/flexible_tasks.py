"""Flexible task wrappers that accept fold_* splits"""
import os
from typing import Optional
from datasets import load_dataset, load_from_disk, DatasetDict
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deps.olmes.oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY

def create_flexible_task(task_name: str, original_task_class):
    class FlexibleTask(original_task_class):
        def download(self, data_dir: Optional[str] = None, cache_dir: Optional[str] = None, download_mode=None):
            split = self.task_config.get("split")
            if isinstance(split, str) and split.startswith("fold_"):
                fold_name = split
                local_path = os.path.join("data", "k_folds", f"{task_name}_k5_seed0")
                ds = None
                if os.path.exists(local_path):
                    try:
                        dd = load_from_disk(local_path)
                        ds = dd[fold_name]
                        print(f"✓ Loaded {fold_name} from local: {len(ds)} instances")
                    except Exception as e:
                        print(f"Failed to load local: {e}")
                if ds is None:
                    try:
                        ds = load_dataset(f"ehasler/{task_name}-k5-folds", split=fold_name)
                        print(f"✓ Loaded {fold_name} from HF hub: {len(ds)} instances")
                    except Exception as e:
                        print(f"Failed to load HF: {e}")
                        return super().download(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)
                
                field_mapping = {'boolq': {'answer': 'label'}}
                if task_name in field_mapping:
                    for old_name, new_name in field_mapping[task_name].items():
                        if old_name in ds.column_names and new_name not in ds.column_names:# type: ignore
                            ds = ds.rename_column(old_name, new_name)# type: ignore
                
                self.dataset = DatasetDict({"validation": ds}) # type: ignore
                self.task_config["split"] = "validation"
                return
            return super().download(data_dir=data_dir, cache_dir=cache_dir, download_mode=download_mode)
    return FlexibleTask

# Register all flexible tasks
print("Registering flexible tasks...", flush=True)
TASK_REGISTRY['hellaswag_flexible'] = create_flexible_task('hellaswag', TASK_REGISTRY['hellaswag'])
TASK_REGISTRY['arc_easy_flexible'] = create_flexible_task('arc_easy', TASK_REGISTRY['arc_easy'])
TASK_REGISTRY['boolq_flexible'] = create_flexible_task('boolq', TASK_REGISTRY['boolq'])
TASK_REGISTRY['csqa_flexible'] = create_flexible_task('csqa', TASK_REGISTRY['csqa'])
TASK_REGISTRY['openbookqa_flexible'] = create_flexible_task('openbookqa', TASK_REGISTRY['openbookqa'])
TASK_REGISTRY['piqa_flexible'] = create_flexible_task('piqa', TASK_REGISTRY['piqa'])
TASK_REGISTRY['socialiqa_flexible'] = create_flexible_task('socialiqa', TASK_REGISTRY['socialiqa'])
TASK_REGISTRY['winogrande_flexible'] = create_flexible_task('winogrande', TASK_REGISTRY['winogrande'])
print(f"Registered {len([k for k in TASK_REGISTRY.keys() if 'flexible' in k])} flexible tasks", flush=True)