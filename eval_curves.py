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

def eval_curves_cls(dataset, model_name, method):
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    
    curve_file = os.path.join(dir_path, method, f'top1_{dataset}_{model_name}.jsonl')
    data = func.load_jsonl_to_dict(curve_file)

    sm_auc_avg, smnorm_auc_avg, l_auc_avg = 0,0,0
    for path in tqdm(path_list):
        bn = os.path.basename(path)
        dn = os.path.basename(os.path.dirname(path))
        fname = dn+'\\'+bn
        dict = data[fname]
        o_l = dict['orig_l']
        l = np.array([o_l] + dict['l'])
        o_sm = dict['orig_sm']
        sm = np.array([o_sm] + dict['sm'])
        s_norm = sm/o_sm
        l_norm = l/o_l

        smnorm_auc_avg += calc_auc(s_norm)
        sm_auc_avg += calc_auc(sm)
        l_auc_avg += calc_auc(l_norm)
    sm_auc_avg/=len(path_list)
    l_auc_avg/=len(path_list)
    smnorm_auc_avg/=len(path_list)

    print(f'avg sm AUC = {sm_auc_avg}, avg sm(norm) AUC = {smnorm_auc_avg}, avg logit ACU = {l_auc_avg}')




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

def eval_compression_quality_cls(dataset, model, method):
    COMP = {
        1e-3: 0,
        5e-4: 0,
        1e-4: 0,
        5e-5: 0,
        1e-5: 0,
    }
        
    curve_file = os.path.join(dir_path, method, f'top1_{dataset}_{model}.jsonl')
    data = func.load_jsonl_to_dict(curve_file)

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    for path in path_list:
        bn = os.path.basename(path)
        dn = os.path.basename(os.path.dirname(path))
        fname = dn + '\\' + bn
        g = data[fname]
        sm = np.array([g['orig_sm']] + g['sm'])
        sm_norm = (sm - sm[0])/sm[0]
        N = len(g['start_frames'])
        
        for thr in COMP.keys():
            n = N - np.argwhere(sm_norm>-1*thr).max()
            COMP[thr] += int(n)
    for thr in COMP.keys():
        COMP[thr]/=len(path_list)
    print(COMP)


'''
'ucf101', 'mc3-18'

print(COMP)
{0.1: 1.1030927835051547, 0.01: 1.225218080888184, 0.001: 1.2767644726407612, 0.0005: 1.2807295796986518, 0.0001: 1.2854877081681206, 5e-05: 1.2857520486386467, 1e-05: 1.2860163891091727}
print(ACC)
{0.1: 0.8437747819191118, 0.01: 0.8456251652127941, 0.001: 0.8456251652127941, 0.0005: 0.8456251652127941, 0.0001: 0.8456251652127941, 5e-05: 0.8456251652127941, 1e-05: 0.8456251652127941}

{0.2: 1.062648691514671, 0.3: 1.0354216230504891, 0.5: 1.0124240021147237}
{0.2: 0.8427174200370077, 0.3: 0.8413957176843775, 0.5: 0.8387523129791171}
{0: 1.2860163891091727}
{0: 0.8456251652127941}

'ucf101', 'r3d-18'

{0.1: 1.7739888977002378, 0.2: 1.5434840074015332, 0.3: 1.3822363203806503, 0.5: 1.182659265133492}
{0.1: 0.8112609040444092, 0.2: 0.8102035421623051, 0.3: 0.8072957969865187, 0.5: 0.8020089875759979}
{0: 2.316415543219667}
{0: 0.8128469468675654}

'''

def eval_acc_comp_cls(dataset, model, method):
    COMP = {
        0: 0
    }
    ACC = {}
    for k in COMP:
        ACC[k] = 0

    curve_file = os.path.join(dir_path, method, f'top1_{dataset}_{model}.jsonl')
    data = func.load_jsonl_to_dict(curve_file)

    model = get_model.get_model(dataset, model)

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    for i in tqdm(range(len(path_list))):
        path = path_list[i]
        bn = os.path.basename(path)
        dn = os.path.basename(os.path.dirname(path))
        fname = dn + '\\' + bn
        g = data[fname]
        sm = np.array([g['orig_sm']] + g['sm'])
        sm_norm = (sm - sm[0])/sm[0]
        N = len(g['start_frames'])
        rem_frames = g['rem_f']
        assert len(rem_frames)==N-1
    
        video = model.get_video(path)
        video = video.to(model.device)

        for thr in COMP.keys():
            aw = np.argwhere(sm_norm>-1*thr)
            if len(aw)==0: # nothing can be removed
                n=0
            else:
                n = aw.max()
            #what frames satisfy this
            removed_frames = rem_frames[:n]
            frames_left = [f for f in g['start_frames'] if f not in removed_frames]

            COMP[thr] += len(frames_left) 

            # eval
            if len(frames_left)<N:
                fvideo = func.fill_with_keep(frames_left, video)
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
        video = video.to(model.device)
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

def eval_mul(dataset, model, method, forward):
    if method in ['random', 'facility']:
        mult_file = os.path.join(dir_path, method, f'multi_{dataset}_{model}.jsonl')
    else:
        ward = 'forward' if forward else 'backward'
        mult_file = os.path.join(dir_path, method, f'multi_{dataset}_{model}_{ward}.jsonl')

    data = func.load_jsonl_to_dict(mult_file)

    n_sfs = []
    k_list = []
    for k in data:
        n = len(data[k])+1
        n_sfs.append(n)
        k_list.append(k)

    total = 12
    n_sfs = np.array(n_sfs)
    hist, bin_edges = np.histogram(n_sfs, bins=total, range=(1, total+1))


if __name__ == "__main__":
    # eval_curves_cls('ssv2', 'vjepa2', 'random')
    eval_acc_comp_cls('ucf101', 'mc3-18', 'sfs')
    # eval_acc_comp('ssv2', 'tformer_base', 'random', forward=False)
    # eval_mul('ucf101', 'r3d-18', 'brute', forward=True)