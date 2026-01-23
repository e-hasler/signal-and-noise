# submit_fold_jobs.py
import os
import shlex

from sympy import python

selected_tasks = ['hellaswag', 'openbookqa', 'socialiqa', 'winogrande']
models_to_evaluate = ['allenai/DataDecide-dclm-baseline-75p-dolma1.7-25p-150M', 
                      'allenai/DataDecide-falcon-and-cc-qc-orig-10p-530M']
corr_models = ['dolma17-25p-DCLM-baseline-75p-150M-5xC', 'falcon_and_cc_tulu_qc_top10-530M-5xC']
corr_steps = [38157, 57776]

def all_jobs():
    for task in selected_tasks:
        if task == 'hellaswag':
            # evaluate only models_to_evaluate[1]
            model = models_to_evaluate[1]
            j = 1
            for i in range(1, 5):
                job_name = f"eval-{task}-f{i}-m{j}"
                cmd = f"cd /home/ehasler/signal-and-noise && " \
                    f"export HF_HOME=$HOME/.cache/huggingface && " \
                    f"python results/permissive_eval_copy.py " \
                    f"--model {model} " \
                    f"--revision step{corr_steps[j]}-seed-default " \
                    f"--task {task}_flexible " \
                    f"--split fold_{i} " \
                    f"--output-dir /home/ehasler/signal-and-noise/results/k_folds/{task}_fold_{i}_{corr_models[j]} " \
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
                    cmd = f"cd /home/ehasler/signal-and-noise && " \
                        f"export HF_HOME=$HOME/.cache/huggingface && " \
                        f"python results/permissive_eval_copy.py " \
                        f"--model {model} " \
                        f"--revision step{corr_steps[j]}-seed-default " \
                        f"--task {task}_flexible " \
                        f"--split fold_{i} " \
                        f"--output-dir /home/ehasler/signal-and-noise/results/k_folds/{task}_fold_{i}_{corr_models[j]} " \
                        f"--gpus 0"

                    safe_cmd = shlex.quote(cmd)
                    #os.system(f"python getting-started/csub.py -n {job_name} -g 1 --train --train --secret-name runai-nlp-ehasler-env --command {safe_cmd}")
                    os.system(f'python /Users/eleonore.hasler/getting-started/csub.py -n {job_name} -g 1 --train --secret-name runai-nlp-ehasler-env --env-file /Users/eleonore.hasler/getting-started/.env --command "{safe_cmd}"')

def first_job():
    job_name = f"eval-hellaswag-f2-m1"
    cmd = f"export HOME=/nlpscratch/home/ehasler " \
    f"export HF_HOME=$HOME/.cache/huggingface " \
    f"python results/permissive_eval_copy.py " \
    f"--model allenai/DataDecide-falcon-and-cc-qc-orig-10p-530M " \
    f"--revision step57776-seed-default " \
    f"--task hellaswag_flexible " \
    f"--split fold_2 " \
    f"--output-dir /nlpscratch/home/ehasler/signal-and-noise/results/k_folds/hellaswag_fold_2_falcon_and_cc_tulu_qc_top10-530M-5xC " \
    f"--gpus 0"

    os.system(f'python /Users/eleonore.hasler/getting-started/csub.py -n {job_name} -g 1 --train --secret-name runai-nlp-ehasler-env --env-file /Users/eleonore.hasler/getting-started/.env --command "{cmd}"')

def main():
    first_job()

if __name__ == "__main__":
    main()