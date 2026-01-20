"""
Basic example: Push a HuggingFace dataset to the hub
"""

from datasets import load_dataset, DatasetDict
"""
# Example 1: Push a single dataset
dataset = load_dataset('Rowan/hellaswag', split='validation')

# Push to HuggingFace hub
# Note: You need to be authenticated first
# Run: huggingface-cli login
# Or: huggingface_hub.login(token="your_token")

dataset.push_to_hub(
    repo_id="ehasler/hellaswag-sample",  # Replace with your username
    private=False,  # Set to True if you want it private
    max_shard_size="500MB"
)

print("✓ Dataset pushed successfully!")

# Example 2: Push a DatasetDict (multiple splits)
# This would push all your k-folds as separate splits
folds = DatasetDict({
    "fold_0": dataset.select(range(0, 1000)),
    "fold_1": dataset.select(range(1000, 2000)),
    "fold_2": dataset.select(range(2000, 3000)),
})

folds.push_to_hub(
    repo_id="ehasler/hellaswag-k3-folds",
    private=False,
    max_shard_size="500MB"
)

print("✓ K-fold dataset pushed successfully!")
"""