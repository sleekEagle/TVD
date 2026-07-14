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

def get_video_curve(model, video, idx):
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

    if forward:
        idx_sort = np.argsort(greedy_js)
    else:
        idx_sort = np.argsort(-1*greedy_js)

    def get_best_idx(model, video, idx_present, o_sm):
        idx_left = list(set(range(video.size(2)))-set(idx_present))
        pred_sm_ar = torch.empty(0).to(model.device)
        for idx in idx_left:
            keep = idx_present + [idx]
            tofill, fillwith = func.past_fill(keep)
            fvideo = video.clone()
            func.fill_video(tofill, fillwith, fvideo)

            pred = model.predict_video(fvideo)
            pred_sm = F.softmax(pred,dim=1)
            pred_sm_ar = torch.concatenate([pred_sm_ar, pred_sm])

        js = func.jensen_shannon(pred_sm_ar.to(o_sm.device), o_sm.repeat(pred_sm_ar.size(0),1))
        bi = idx_left[torch.argmin(js)]
        return bi
    
    sel_idx = [int(best_idx)]
    for _ in range(video.size(2)-2):
        bi = get_best_idx(model, video, sel_idx, o_sm)
        sel_idx += [bi]
    sel_idx += list(set(range(video.size(2))) - set(sel_idx))

    return sel_idx


'''
forward: forward or backward selection. not applicable for method=facility
'''
def dataset_curves(dataset, model, method, forward = True):
    out_path = CONF.OUT_PATH
    out_file = os.path.join(out_path, method)
    os.makedirs(out_file, exist_ok=True)
    if method!='facility':
        ward = 'forward' if forward else 'backward'
        out_file = os.path.join(out_file, f'curves_{dataset}_{model}_{ward}.h5') 
    else:
        out_file = os.path.join(out_file, f'curves_{dataset}_{model}.h5') 


    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model)

    with h5py.File(out_file, "w") as f:
        for i in range(len(path_list)):
            print(f'{i} of {len(path_list)} is done.', end='\r')

            video = model.get_video(path_list[i])
            fname = os.path.basename(path_list[i])
            L = video.size(2)

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
                idx = emb_facilitylocation(emb)
            elif method == 'brute':
                idx = brute(video, model, greedy_js, forward) 

            sim_ar, js_ar = get_video_curve(model, video, idx)

            # import matplotlib.pyplot as plt
            # plt.plot(js_ar_f)
            # plt.plot(js_ar_b)
            d={
                fname: {'sim_ar': torch.tensor(sim_ar), 'js_ar': torch.tensor(js_ar), 'idx': torch.tensor(idx)}
            }

            func.save_dict_to_h5(f, d)

if __name__ == "__main__":
    dataset_curves('ucf101', 'r3d-18', 'brute', forward=True)
