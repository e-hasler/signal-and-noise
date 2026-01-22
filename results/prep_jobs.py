# submit_fold_jobs.py
import os
import shlex

selected_tasks = ['hellaswag', 'openbookqa', 'socialiqa', 'winogrande']
models_to_evaluate = ['allenai/DataDecide-dclm-baseline-75p-dolma1.7-25p-150M', 
                      'allenai/DataDecide-falcon-and-cc-qc-orig-10p-530M']
corr_models = ['dolma17-25p-DCLM-baseline-75p-150M-5xC', 'falcon_and_cc_tulu_qc_top10-530M-5xC']
corr_steps = [38157, 57776]  # Updated: 57500 -> 57776 (actual available revision for falcon model)

for task in selected_tasks:
    if task == 'hellaswag':
        # evaluate only models_to_evaluate[1]
        model = models_to_evaluate[1]
        j = 1
        for i in range(1, 5):
            job_name = f"eval-{task}-f{i}-m{j}"
            cmd = f"cd /nlpscratch/home/ehasler/signal-and-noise && " \
                  f"export HF_HOME=/nlpscratch/home/ehasler/.cache " \
                  f"python results/permissive_eval.py " \
                  f"--model {model} " \
                  f"--revision step{corr_steps[j]}-seed-default " \
                  f"--task {task}_flexible " \
                  f"--split fold_{i} " \
                  f"--output-dir /nlpscratch/home/ehasler/signal-and-noise/results/k_folds/{task}_fold_{i}_{corr_models[j]} " \
                  f"--gpus 0"
            safe_cmd = shlex.quote(cmd)
            #os.system(f"python getting-started/csub.py -n {job_name} -g 1 --train --train --secret-name runai-nlp-ehasler-env --command {safe_cmd}")
            os.system(f'python /Users/eleonore.hasler/getting-started/csub.py -n {job_name} -g 1 --train --secret-name runai-nlp-ehasler-env --env-file /Users/eleonore.hasler/getting-started/.env --command "{safe_cmd}"')
            break
    else:
        break
        for j, model in enumerate(models_to_evaluate):
            for i in range(5):
                job_name = f"eval-{task}-f{i}-m{j}"
                cmd = f"cd /nlpscratch/home/ehasler/signal-and-noise && " \
                      f"export HF_HOME=/nlpscratch/home/ehasler/.cache " \
                      f"python results/permissive_eval.py " \
                      f"--model {model} " \
                      f"--revision step{corr_steps[j]}-seed-default " \
                      f"--task {task}_flexible " \
                      f"--split fold_{i} " \
                      f"--output-dir /nlpscratch/home/ehasler/signal-and-noise/results/k_folds/{task}_fold_{i}_{corr_models[j]} " \
                      f"--gpus 0"

                safe_cmd = shlex.quote(cmd)
                #os.system(f"python getting-started/csub.py -n {job_name} -g 1 --train --train --secret-name runai-nlp-ehasler-env --command {safe_cmd}")
                os.system(f'python /Users/eleonore.hasler/getting-started/csub.py -n {job_name} -g 1 --train --secret-name runai-nlp-ehasler-env --env-file /Users/eleonore.hasler/getting-started/.env --command "{safe_cmd}"')

