import torch
import numpy as np
import pandas as pd
from PIL import Image
from os import makedirs
from os.path import exists
from os import listdir
from Models import *

""" Script to create accuracy datasets
"""

# inpust
tench_dir = 'data/tench/'
tenches = listdir(tench_dir)

# outputs
out_dir = 'results/acc_datasets/'
if not exists(out_dir):
    makedirs(out_dir)
results = []

model = VGG()
model.eval()
with torch.no_grad():
    miss_count = 0
    for tench in tenches:
        img = Image.open(tench_dir+tench)
        img_transf = torch.unsqueeze(model.transforms(img),dim=0)
        logs = model.simple_eval(img_transf)[0]
        if torch.argmax(logs) != 0:
            miss_count += 1
        else:
            results.append([0, logs[0], logs[1]])
    res_np = np.array(results)
    tench_df = pd.DataFrame(res_np, columns=['Class', 'Class 0 acc', 'Class 1 acc'])
    tench_df.to_csv('results/acc_datasets/tench.csv')
    print(f'Found {miss_count} misclassified images')