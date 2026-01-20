# submit_fold_jobs.py
import os

selected_tasks = ['hellaswag', 'openbookqa', 'socialiqa', 'winogrande']
models_to_evaluate = ['allenai/DataDecide-dclm-baseline-75p-dolma1.7-25p-150M', 
                      'allenai/DataDecide-falcon-and-cc-qc-orig-10p-530M']
corr_models = ['dolma17-25p-DCLM-baseline-75p-150M-5xC', 'falcon_and_cc_tulu_qc_top10-530M-5xC']
corr_steps = [38157, 57500]

for task in selected_tasks:
    if task == 'hellaswag':
        # evaluate only models_to_evaluate[1]
        model = models_to_evaluate[1]
        j = 1
        for i in range(1, 5):
            job_name = f"eval-{task}-f{i}-m{j}"
            cmd = f"cd /mloscratch/homes/ehasler/signal-and-noise && " \
                  f"python results/permissive_eval.py " \
                  f"--model {model} " \
                  f"--revision step{corr_steps[j]}-seed-default " \
                  f"--task {task}_flexible " \
                  f"--split fold_{i} " \
                  f"--output-dir results/k_folds/{task}_fold_{i}_{corr_models[j]} " \
                  f"--gpus 0"
            
            os.system(f'python csub.py -n {job_name} -g 1 --train --command "{cmd}"')
    else:
        for j, model in enumerate(models_to_evaluate):
            for i in range(5):
                job_name = f"eval-{task}-f{i}-m{j}"
                cmd = f"cd /mloscratch/homes/ehasler/signal-and-noise && " \
                      f"python results/permissive_eval.py " \
                      f"--model {model} " \
                      f"--revision step{corr_steps[j]}-seed-default " \
                      f"--task {task}_flexible " \
                      f"--split fold_{i} " \
                      f"--output-dir results/k_folds/{task}_fold_{i}_{corr_models[j]} " \
                      f"--gpus 0"
                
                os.system(f'python csub.py -n {job_name} -g 1 --train --command "{cmd}"')