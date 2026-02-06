"""
Verify consistency of k-fold splits for HellaSwag dataset
"""

from datasets import DatasetDict, load_dataset
import os
from collections import Counter

# Load the original dataset
print("Loading original HellaSwag dataset...")
original = load_dataset('Rowan/hellaswag', split='validation')
print(f"Original dataset size: {len(original)} instances") # type: ignore

# Check for duplicate IDs in the original dataset
print("\nChecking for duplicate IDs in original dataset...")
original_ids = original['ind'] # type: ignore
id_counts = Counter(original_ids)
duplicates = {id_val: count for id_val, count in id_counts.items() if count > 1}

if duplicates:
    print(f"  ⚠ Found {len(duplicates)} duplicate IDs in original dataset!")
    print(f"  Most common duplicates:")
    for id_val, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    ID {id_val}: appears {count} times")
else:
    print("  ✓ No duplicate IDs in original dataset")

# Load the saved k-fold dataset
fold_path = "data/k_folds/hellaswag_k5_seed0"
folds = DatasetDict.load_from_disk(fold_path)

print(f"\nNumber of folds: {len(folds)}")
print(f"\nFold sizes:")
for fold_name, fold_data in folds.items():
    print(f"  {fold_name}: {len(fold_data)} instances")

# Check total coverage
total_instances = sum(len(fold) for fold in folds.values())
print(f"\nTotal instances across folds: {total_instances}")

# Check for overlaps (verify non-overlapping folds)
print("\nChecking for overlaps in folds...")
all_ids = set()
has_overlap = False
overlap_details = {}

for fold_name, fold_data in folds.items():
    fold_ids = set(fold_data['ind'])  # 'ind' is HellaSwag's instance ID field
    overlap = all_ids & fold_ids
    if overlap:
        print(f"  ⚠ {fold_name} has {len(overlap)} overlapping IDs with previous folds!")
        overlap_details[fold_name] = overlap
        has_overlap = True
    all_ids.update(fold_ids)

if not has_overlap:
    print("  ✓ No overlaps detected - folds are properly separated")
else:
    # Determine if overlaps are due to original duplicates
    print(f"\n  Analyzing overlap causes:")
    total_overlap_ids = set()
    for fold_overlaps in overlap_details.values():
        total_overlap_ids.update(fold_overlaps)
    
    original_duplicates_in_overlap = sum(1 for id_val in total_overlap_ids if duplicates.get(id_val, 0) > 0)
    print(f"    - Overlapping IDs that are duplicates in original: {original_duplicates_in_overlap}")
    print(f"    - Overlapping IDs that are NOT duplicates in original: {len(total_overlap_ids) - original_duplicates_in_overlap}")
    
    if len(total_overlap_ids) - original_duplicates_in_overlap > 0:
        print(f"    ⚠ WARNING: Some overlaps are NOT due to original duplicates! Folds may not be properly separated.")

# Sample questions from each fold
print("\n" + "="*60)
print("Sample questions from each fold:")
print("="*60)
for fold_name in sorted(folds.keys())[:2]:  # Show first 2 folds
    fold = folds[fold_name]
    print(f"\n{fold_name} - first 2 instances:")
    for i in range(min(2, len(fold))):
        item = fold[i]
        print(f"\n  Instance {i}:")
        print(f"    ID: {item['ind']}")
        print(f"    Question: {item['ctx'][:100]}...")
        print(f"    Answer: {item['label']}")