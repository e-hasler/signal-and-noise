
import os
from snr.download.hf import pull_predictions_from_hf
import pandas as pd



local_path = pull_predictions_from_hf("allenai/signal-and-noise", split_name='datadecide_intermediate')
df = pd.read_parquet(local_path)

selected_tasks = ['arc_easy', 'csqa', 'hellaswag', 'openbookqa', 'socialiqa', 'winogrande']
models_to_evaluate = ['allenai/DataDecide-dclm-baseline-75p-dolma1.7-25p-150M', 'allenai/DataDecide-falcon-and-cc-qc-orig-10p-530M']

corr_models = ['dolma17-25p-DCLM-baseline-75p-150M-5xC', 'falcon_and_cc_tulu_qc_top10-530M-5xC']
corr_mixes = ['dolma17-25p-DCLM-baseline-75p' , 'falcon_and_cc_tulu_qc_top10']
corr_steps = [38157, 57500]


def run_on_fold():
    for task in selected_tasks:
        print(f"Running folds for task: {task}_flexible")
        for j, model in enumerate(models_to_evaluate[0:]):
            for i in range(5):
                os.system(f"""
                    python results/permissive_eval.py \
                    --model {model} \
                    --revision step{corr_steps[j]}-seed-default \
                    --task {task}_flexible \
                    --split fold_{i} \
                    --output-dir results/k_folds/{task}_fold_{i}_{corr_models[j]} \
                    --gpus 0
                    """)
            
def main():
    run_on_fold()

if __name__ == "__main__":
    main()