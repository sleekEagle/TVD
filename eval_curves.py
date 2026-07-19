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
from tqdm import tqdm

def calc_auc(points):
    x = np.linspace(0, 1, len(points))
    return trapezoid(points, x)


dir_path = CONF.OUT_PATH
def eval_curves(dataset, model, method, forward):
    if method in ['greedy', 'foolish', 'brute']:
        ward = 'forward' if forward else 'backward'
        curve_file = os.path.join(dir_path, method, f'curves_{dataset}_{model}_{ward}.jsonl')
    else:
        curve_file = os.path.join(dir_path, method, f'curves_{dataset}_{model}.jsonl')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)

    js_auc_avg, sim_auc_avg = 0,0

    data = func.load_jsonl_to_dict(curve_file)
    for path in path_list:
        fname = os.path.basename(path)
        g = data[fname]
        js_ar = g['js_ar']
        sim_ar = g['sim_ar']

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


def eval_compression_quality(dataset, model, method, forward):
    COMP = {
        1e-3: 0,
        5e-4: 0,
        1e-4: 0,
        5e-5: 0,
        1e-5: 0,
    }
        
    ward = 'forward' if forward else 'backward'
    curve_file = os.path.join(dir_path, method, f'curves_{dataset}_{model}_{ward}.jsonl')
    data = func.load_jsonl_to_dict(curve_file)

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    for path in path_list:
        fname = os.path.basename(path)
        g = data[fname]
        js_ar = np.array(g['js_ar'])

        for thr in COMP.keys():
            n = np.argwhere(js_ar<thr).min()
            COMP[thr] += int(n)
    for thr in COMP.keys():
        COMP[thr]/=len(path_list)
    print(COMP)

def eval_acc_comp(dataset, model, method, forward):
    COMP = {
        1e-1: 0,
        1e-2: 0,
        1e-3: 0,
        5e-4: 0,
        1e-4: 0,
        5e-5: 0,
        1e-5: 0
    }
    ACC = {
        1e-1: 0,
        1e-2: 0,
        1e-3: 0,
        5e-4: 0,
        1e-4: 0,
        5e-5: 0,
        1e-5: 0
    }
    
    if method in ['random', 'facility']:
        curve_file = os.path.join(dir_path, method, f'curves_{dataset}_{model}.jsonl')
    else:
        ward = 'forward' if forward else 'backward'
        curve_file = os.path.join(dir_path, method, f'curves_{dataset}_{model}_{ward}.jsonl')
    data = func.load_jsonl_to_dict(curve_file)
    model = get_model.get_model(dataset, model)

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    for i in tqdm(range(len(path_list))):
        path = path_list[i]
        fname = os.path.basename(path)
        g = data[fname]
        js_ar = np.array(g['js_ar'])
        idx_ar = g['idx']
        video = model.get_video(path)
        L = video.size(2)

        for thr in COMP.keys():
            n = np.argwhere(js_ar<thr).min()+1
            COMP[thr] += int(n) 
            idx_used = idx_ar[:n]

            # eval
            if len(idx_used)<L:
                fvideo = func.fill_with_keep(idx_used, video)
            else:
                fvideo = video
            pred_cls = torch.argmax(model.predict_video(fvideo))
            if idx_list[i]==pred_cls:
                ACC[thr] += 1
    for thr in COMP.keys():
        COMP[thr]/=len(path_list)
        ACC[thr]/=len(path_list)

    print(COMP)
    print(ACC)

if __name__ == "__main__":
    eval_curves('ucf101', 'r3d-18', 'random', forward=False)