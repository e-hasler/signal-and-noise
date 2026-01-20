import pandas as pd
from snr.download.hf import pull_predictions_from_hf
from datasets import load_dataset, DatasetDict, DownloadConfig

df = pd.read_parquet(pull_predictions_from_hf("allenai/signal-and-noise", split_name='datadecide_intermediate'))

print(df['task'].unique())