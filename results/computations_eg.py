import numpy as np
import pandas as pd
from snr.download.hf import pull_predictions_from_hf
import statistics
from math import comb
from IPython.display import display
from itertools import combinations  

"""
Goal : Get in hand some ways of measuring from the paper
"""

df = pd.read_parquet('analysis/data')


# ===========================================
# Compute checkpoint-to-checkpoint noise
# ===========================================

# First example: 'DCLM-baseline-150M-5xC' on the 'hellaswag' benchmark

"""
My to do:
get all the checkpoints of the model
get the score of each checkpoint
compute the standard deviation of the scores

"""
"""
#print(df.columns)

dres = df.loc[:, ['task','model', 'primary_score', 'primary_metric', 'step']].dropna()
#display(dres)

# 1. get all the checkpoints of the model

d_dclm = dres[dres['model'].str.contains('DCLM-baseline-150M-5xC')]
#display(d_dclm)

d2 = d_dclm[d_dclm['model'] == 'DCLM-baseline-150M-5xC-2']
d2 = d2[d2['task'] == 'hellaswag']
#print(d2['primary_metric'].unique())

display(d2)

# 2. compute the standard deviation of the scores

noise = statistics.stdev(d2['primary_score'])
print(f"checkpoint-to-checkpoint noise of 'DCLM-baseline-150M-5xC' on the 'hellaswag' benchmark is {noise}")

"""

# ===========================================
# Compute decision accuracy
# ===========================================

# First example: use 3 models, 2 sizes (150M-5xC and 1B) on Hellaswag

"""
Models :
'DCLM-baseline-150M-5xC-2'
'baseline-150M-5xC-2'
'c4-150M-5xC-2'

'DCLM-baseline-1B-5xC-2'
'baseline-1B-5xC-2'
'c4-1B-5xC-2'
"""

df = pd.read_parquet('analysis/data')
dres = df.loc[:, ['task','model', 'primary_score', 'primary_metric', 'step']].dropna()

# Define the models we want
models_150M = [
    'DCLM-baseline-150M-5xC-2',
    'baseline-150M-5xC-2',
    'c4-150M-5xC-2'
]

models_1B = [
    'DCLM-baseline-1B-5xC-2',
    'baseline-1B-5xC-2',
    'c4-1B-5xC-2'
]

all_150M = {}
for modl in models_150M:
    subset = dres[dres['model'] == modl]
    all_150M[modl] = subset.loc[subset['step'].idxmax(), 'primary_score']

all_1B = {}
for modl in models_1B:
    subset = dres[dres['model'] == modl]
    all_1B[modl] = subset.loc[subset['step'].idxmax(), 'primary_score']

print("150M models:", all_150M)
print("1B models:", all_1B)

for model in ['DCLM-baseline-150M-5xC-2', 'baseline-150M-5xC-2', 'c4-150M-5xC-2']:
    res = dres[dres['model'] == model]['task'].unique()
    if 'hellaswag' in res:
        print(True)
    print(dres[(dres['model'] == model) & (dres['task'] == 'hellaswag')]['primary_metric'])

# Get score of each from 'primary_score' (based on their primary metric 'acc_per_char', take last checkpoint)


def decision_accuracy(small, big):
    s = len(small)
    b = len(big)
    if s == b:
        indexes = list(range(s))
        P = list(combinations(indexes, 2))
        print(P)
    else:
        print("Small and big do not have the same size")

    res = 0
    for a,b in P: # type: ignore


        if np.sign(small[list(small.keys())[a]] - small[list(small.keys())[b]]) \
            == np.sign(big[list(big.keys())[a]] - big[list(big.keys())[b]]):
            res += 1 
    #print(res)
    dec = 1/len(P) * res # type: ignore
    #print(dec)
    return dec


decision_accuracy(all_150M, all_1B)
