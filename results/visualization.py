import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


METRICS= ['results/hellaswag-125M/task-000-hellaswag-metrics.json']
PREDS = ['results/hellaswag-125M/task-000-hellaswag-predictions.jsonl']

# Load metrics and identify the primary metric configured for the task
for metric_file in METRICS:
    with open(metric_file) as f:
        metrics = json.load(f)

primary_metric_key = metrics.get("task_config", {}).get("primary_metric", "primary_score") # type: ignore
primary_task_score = metrics.get("metrics", {}).get(primary_metric_key, metrics.get("metrics", {}).get("primary_score"))# type: ignore
print(f"Primary metric key: {primary_metric_key}")
print(f"Task-level primary score: {primary_task_score}")

# Load predictions
predictions = []
with open(PREDS[0]) as f:
    for line in f:
        predictions.append(json.loads(line))
    #print(len(predictions))

df = pd.DataFrame(predictions)
display(df.head())
print(f"Total instances: {len(df)}")

# Extract the per-instance value for the primary metric (falls back to primary_score if missing)
def get_primary_metric(row):
    return row.get(primary_metric_key, row.get("primary_score"))

df["primary_metric_value"] = df["metrics"].apply(get_primary_metric)

# Plot distribution of the primary metric over instances
plt.figure(figsize=(8, 4))
df["primary_metric_value"].hist(bins=30)
plt.xlabel(primary_metric_key)
plt.ylabel("Count")
plt.title(f"Distribution of {primary_metric_key} across instances")
plt.tight_layout()
# plt.show()

# Optional: line plot over index to see drift/variance
plt.figure(figsize=(10, 3))
plt.plot(df.index, df["primary_metric_value"], marker='.', linestyle='')
plt.xlabel('Instance Index')
plt.ylabel(primary_metric_key)
plt.title(f"{primary_metric_key} by instance")
plt.tight_layout()
plt.show()











"""

# Optional: plot accuracy per choice
choice_accuracy = {}
for choice_idx in range(4):
    mask = df['gold'] == choice_idx
    if mask.sum() > 0:
        acc = (df[mask]['pred'] == choice_idx).sum() / mask.sum()
        choice_accuracy[f'Choice {choice_idx}'] = acc

pd.Series(choice_accuracy).plot(kind='bar')
plt.ylabel('Accuracy')
plt.title('HellaSwag Accuracy by Gold Choice')
plt.show()
"""