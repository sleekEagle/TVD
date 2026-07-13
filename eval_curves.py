from dataloaders import data_paths
from models import get_model
import torch
import random
import func
import os
import json
import h5py
import torch.nn.functional as F
import numpy as np
from scipy.integrate import trapezoid
import CONF

def calc_auc(points):
    x = np.linspace(0, 1, len(points))
    return trapezoid(points, x)


dir_path = CONF.OUT_PATH
def eval_curves(dataset, model, method):
    curve_file = os.path.join(dir_path, method, f'curves_{dataset}_{model}.h5')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)

    js_auc_avg, sim_auc_avg = 0,0
    with h5py.File(curve_file, "r") as f:
        for path in path_list:
            fname = os.path.basename(path)
            g = f[fname]
            js_ar = g['js_ar'][:]
            sim_ar = g['sim_ar'][:]

            # normalize 
            # sim_ar = (sim_ar-sim_ar.min())/(sim_ar.max()-sim_ar.min())
            # js_ar = (js_ar-js_ar.min())/(js_ar.max()-js_ar.min())

            js_auc = calc_auc(js_ar)
            sim_auc = calc_auc(sim_ar)

            js_auc_avg += js_auc
            sim_auc_avg += sim_auc

    js_auc_avg /= len(path_list)
    sim_auc_avg /= len(path_list)
    print(f'sim_auc: {sim_auc_avg}, js_auc: {js_auc_avg}')

def eval_compression_quality(dataset, model, method):
    curve_file = os.path.join(dir_path, method, f'curves_{dataset}_{model}.h5')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)

    comp = {
        1e-3: 0,
        5e-4: 0,
        1e-4: 0,
        5e-5: 0,
        1e-5: 0,
    }

    with h5py.File(curve_file, "r") as f:
        for path in path_list:
            fname = os.path.basename(path)
            g = f[fname]
            js_ar = g['js_ar'][:]

            for thr in comp.keys():
                n = np.argwhere(js_ar<thr).min()
                comp[thr] += int(n)
    for thr in comp.keys():
        comp[thr]/=len(path_list)
    print(comp)


if __name__ == "__main__":
    eval_compression_quality('ucf101', 'mc3-18', 'facility')
    pass