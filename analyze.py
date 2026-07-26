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
import CONF
from tqdm import tqdm
import json

def get_video_curve(model, video, idx, forward):
    o_sm = F.softmax(model.predict_video(video),dim=1)
    o_feat = model.get_features()
    L = video.size(2)

    sim_ar, js_ar = [], []
    filled = []
    for i in idx:
        fvideo = video.clone()
        filled.append(i)
        tofill, fillwith = func.past_fill(filled, L)
        if len(tofill)>0 and len(fillwith)>0:
            func.fill_video(tofill, fillwith, fvideo)
        pred = model.predict_video(fvideo)  
        sm = F.softmax(pred, dim=1).to(o_feat.device)
        feat = model.get_features().to(o_feat.device)

        similarity = F.cosine_similarity(feat, o_feat, dim=0)
        js = func.jensen_shannon(sm, o_sm)
        sim_ar.append(similarity.item())
        js_ar.append(js.item())
    
    return sim_ar, js_ar

def get_greedy_js(video, model, forward):
    L = video.size(2)
    o_logits = model.predict_video(video)
    o_sm = F.softmax(o_logits, dim=1)

    js_t = torch.empty(0).to(model.device)
    for i in range(L):
        if forward:
            keep_forward = [i]
            filled_video = func.fill_with_keep(keep_forward, video)
            pred = model.predict_video(filled_video)
            sm = F.softmax(pred, dim=1)
            js = func.jensen_shannon(sm, o_sm)
            js_t = torch.concatenate([js_t, js])
        else:
            keep_backward = [idx for idx in range(L) if idx!=i]
            filled_video = func.fill_with_keep(keep_backward, video)
            pred = model.predict_video(filled_video)
            sm = F.softmax(pred, dim=1)
            js = func.jensen_shannon(sm, o_sm)
            js_t = torch.concatenate([js_t, js])
    return js_t

def emb_facilitylocation(emb, k=16):
    from apricot import FacilityLocationSelection
    selector = FacilityLocationSelection(
        n_samples=k,
        metric="cosine"
    )
    selector.fit(emb)
    keyframe_indices = selector.ranking
    return keyframe_indices

def brute(video, model, greedy_js, forward):
    L = video.size(2)
    o_logits = model.predict_video(video)
    o_sm = F.softmax(o_logits, dim=1)

    idx_sort = np.argsort(greedy_js)
    best_idx = int(idx_sort[0])

    def get_best_idx_forward(model, video, idx_present, o_sm):
        idx_left = list(set(range(video.size(2)))-set(idx_present))
        pred_sm_ar = torch.empty(0).to(model.device)
        for idx in idx_left:
            keep = idx_present + [idx]
            tofill, fillwith = func.past_fill(keep, video.size(2))
            fvideo = video.clone()
            func.fill_video(tofill, fillwith, fvideo)

            pred = model.predict_video(fvideo)
            pred_sm = F.softmax(pred,dim=1)
            pred_sm_ar = torch.concatenate([pred_sm_ar, pred_sm])

        js = func.jensen_shannon(pred_sm_ar.to(o_sm.device), o_sm.repeat(pred_sm_ar.size(0),1))
        bi = idx_left[torch.argmin(js)]
        return bi
    
    def get_best_idx_backward(model, video, idx_remove, o_sm):
        idx_left = list(set(range(video.size(2)))-set(idx_remove))
        pred_sm_ar = torch.empty(0).to(model.device)
        for idx in idx_left:
            remove = idx_remove + [idx]
            keep = list(set(range(video.size(2)))-set(remove))
            tofill, fillwith = func.past_fill(keep, video.size(2))
            fvideo = video.clone()
            func.fill_video(tofill, fillwith, fvideo)

            pred = model.predict_video(fvideo)
            pred_sm = F.softmax(pred,dim=1)
            pred_sm_ar = torch.concatenate([pred_sm_ar, pred_sm])

        js = func.jensen_shannon(pred_sm_ar.to(o_sm.device), o_sm.repeat(pred_sm_ar.size(0),1))
        bi = idx_left[torch.argmin(js)]
        return bi
    
    sel_idx = [best_idx]
    for _ in range(video.size(2)-2):
        if forward:
            bi = get_best_idx_forward(model, video, sel_idx, o_sm)
        else:
            bi = get_best_idx_backward(model, video, sel_idx, o_sm)
        sel_idx += [bi]
    sel_idx += list(set(range(video.size(2))) - set(sel_idx))

    if not forward: # most important must be the first
        sel_idx = sel_idx[::-1]

    return sel_idx


def find_sfs_single(video, model, existing, totry, o_sm):
    js_vals = []
    for i in range(len(totry)):
        tt = [totry[i]]
        fvideo = func.fill_with_keep(existing + tt, video, fill='past')
        pred = model.predict_video(fvideo)
        sm = F.softmax(pred, dim=1)
        js = func.jensen_shannon(sm, o_sm)
        js_vals.append(js.item())
    return np.array(js_vals)

def find_sfs_cummulative(video, model, existing, totry, o_sm, thr=1e-3):
    for i in range(len(totry)+1):
        if len(existing) +  len(totry[:i]) == 0:
            continue
        fvideo = func.fill_with_keep(existing + totry[:i], video, fill='past')
        pred = model.predict_video(fvideo)
        sm = F.softmax(pred, dim=1)
        js = func.jensen_shannon(sm, o_sm)
        if js<=thr:
            return totry[:i]
    return -1




'''
forward: forward or backward selection. not applicable for method=facility
'''



def dataset_curves(dataset, model, method, forward = True):
    out_path = CONF.OUT_PATH
    out_file = os.path.join(out_path, method)
    os.makedirs(out_file, exist_ok=True)
    if method in ['random', 'facility']:
        out_file = os.path.join(out_file, f'curves_{dataset}_{model}.jsonl') 
    else:
        ward = 'forward' if forward else 'backward'
        out_file = os.path.join(out_file, f'curves_{dataset}_{model}_{ward}.jsonl') 

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model)

    # resume from earlier file
    if os.path.exists(out_file):
        existing_data = func.load_jsonl_to_dict(out_file)
    else:
        existing_data = {}

    with open(out_file, 'a') as f:
        for i in tqdm(range(len(path_list))):
            # print(f'{i} of {len(path_list)} is done.', end='\r', flush=True)

            video = model.get_video(path_list[i])
            video = video.to(model.device)
            fname = os.path.basename(path_list[i])
            L = video.size(2)
            
            if fname in existing_data: continue

            if method in ['greedy','foolish','brute']:
                greedy_js = get_greedy_js(video, model, forward).cpu().numpy()
            if method == 'greedy':
                if forward: 
                    idx = np.argsort(greedy_js)
                else:
                    idx = np.argsort(-1*greedy_js)
            if method == 'foolish':
                if forward: 
                    idx = np.argsort(-1*greedy_js)
                else:
                    idx = np.argsort(greedy_js)
            if method == 'random':
                idx = list(range(L))
                random.shuffle(idx)
            elif method == 'facility': # facility location
                emb = torch.empty(0)
                for i in range(L):
                    _ = model.predict_video(video)
                    emb = torch.concatenate([emb, model.get_features()[None,:].to(emb.device)], dim=0)
                idx = emb_facilitylocation(emb, video.size(2))
            elif method == 'brute':
                idx = brute(video, model, greedy_js, forward) 

            sim_ar, js_ar = get_video_curve(model, video, idx, forward)

            # import matplotlib.pyplot as plt
            # plt.plot(js_ar_f)
            # plt.plot(js_ar_b)
            if isinstance(sim_ar, np.ndarray):
                sim_ar = sim_ar.tolist()
            if isinstance(js_ar, np.ndarray):
                js_ar = js_ar.tolist()
            if isinstance(idx, np.ndarray):
                idx = idx.tolist()
            d={
                fname: {'sim_ar': sim_ar, 'js_ar': js_ar, 'idx': idx}
            }

            f.write(json.dumps(d) + '\n')
            f.flush()


def dataset_multiple_SFS(dataset, model_name, method, forward = True, thr=1e-3):
    out_path = CONF.OUT_PATH
    out_file = os.path.join(out_path, method)
    if method in ['random', 'facility']:
        out_path = os.path.join(out_file, f'multi_{dataset}_{model_name}.jsonl') 
        data_path = os.path.join(out_file, f'curves_{dataset}_{model_name}.jsonl') 
    else:
        ward = 'forward' if forward else 'backward'
        out_path = os.path.join(out_file, f'multi_{dataset}_{model_name}_{ward}.jsonl') 
        data_path = os.path.join(out_file, f'curves_{dataset}_{model_name}_{ward}.jsonl')

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    data = func.load_jsonl_to_dict(data_path)
    model = get_model.get_model(dataset, model_name)

    with open(out_path, 'a') as f:
        for path in tqdm(path_list):
            # base_list = [os.path.basename(p) for p in path_list]
            # path_idx = base_list.index(fname)
            # path = path_list[path_idx]
            video = model.get_video(path)
            L = video.size(2)
            fname = os.path.basename(path)
            f_idx = data[fname]['idx']
            js_ar = np.array(data[fname]['js_ar'])
            n = np.argwhere(js_ar<thr).min()
            valid_idx = f_idx[:n+1]
            other_idx = f_idx[n+1:]

            o_logits = model.predict_video(video)
            o_sm = F.softmax(o_logits, dim=1)

            new_idx = []
            new_idx_list = []
            used_idx_list = []
            search_idx = other_idx
            while True:
                new_idx = find_sfs_cummulative(video, model, [] , search_idx, o_sm)
                if new_idx==-1: break

                used_idx_list += new_idx
                search_idx = [item for item in other_idx if item not in used_idx_list]
                new_idx_list.append(new_idx)

            d={fname: new_idx_list}
            f.write(json.dumps(d) + '\n')
            f.flush()

def get_preds(video, model):
    logits = model.predict_video(video)
    sm = F.softmax(logits, dim=1)
    f = model.get_features()
    return sm, f

def distribution_shift(dataset, model_name, forward = True, thr=1e-3):
    out_path = CONF.OUT_PATH
    out_file = os.path.join(out_path, 'brute')
    ward = 'forward' if forward else 'backward'
    data_path = os.path.join(out_file, f'curves_{dataset}_{model_name}_{ward}.jsonl')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    data = func.load_jsonl_to_dict(data_path)
    model = get_model.get_model(dataset, model_name)
    N_ITR = 4
    methods = ['future', 'past', 'zero', 'mean', 'interp']
    big_metrics = {}
    for N in range(N_ITR):
        method_js = {}
        method_sim = {}
        for m in methods:
            method_js[m] = []
            method_sim[m] = []
        big_metrics[N] = {'js': method_js, 'sim': method_sim}

    for path in tqdm(path_list):
        video = model.get_video(path)
        L = video.size(2)
        fname = os.path.basename(path)
        f_idx = data[fname]['idx']
        js_ar = np.array(data[fname]['js_ar'])
        n = np.argwhere(js_ar<thr).min()
        valid_idx = f_idx[:n+1]

        # original prediction
        sm_orig, f_orig = get_preds(video, model)

        # remove random frames
        all_idx = list(range(0,L))
        used_idx = []
        for n in range(N_ITR):
            if len(used_idx) == 0:
                rand_idx = random.randint(0,L-1)
                used_idx.append(rand_idx)
            else:
                #expand the region
                if max(used_idx)<L-1:
                    used_idx.append(max(used_idx)+1)
                elif min(used_idx)>0:
                    used_idx.append(min(used_idx)-1)
                    
            keep_idx = [i for i in all_idx if i not in used_idx]
            
            for method in methods:
                fvideo = func.fill_with_keep(keep_idx, video, method)
                sm, f = get_preds(fvideo, model)
                js_sm = func.jensen_shannon(sm, sm_orig)
                big_metrics[n]['js'][method].append(js_sm.item())
                similarity = F.cosine_similarity(f, f_orig, dim=0)
                big_metrics[n]['sim'][method].append(similarity.item())

    for n in range(N_ITR):
        print(f'N: {n+1}')
        for m in methods:
            ar_js = np.array(big_metrics[n]['js'][m])
            ar_sim = np.array(big_metrics[n]['sim'][m])
            print(f'{m:>6} **** JS: mean = {ar_js.mean():.8f} , std = {ar_js.std():.8f} SIM: mean = {ar_sim.mean():.8f} , std = {ar_sim.std():.8f}')

            


if __name__ == "__main__":
    # dataset_multiple_SFS('ucf101', 'r3d-18', 'brute', forward=True)
    # dataset_curves('ssv2', 'tformer_base', 'facility', forward=False)
    distribution_shift('ucf101', 'mc3-18', forward = True)
    pass
