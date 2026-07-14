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

def get_greedy_js(video, model):
    L = video.size(2)
    o_logits = model.predict_video(video)
    o_sm = F.softmax(o_logits, dim=1)

    f_js_t, b_js_t = torch.empty(0).to(model.device), torch.empty(0).to(model.device)
    for i in range(L):
        keep_forward = [i]
        keep_backward = [idx for idx in range(L) if idx!=i]

        f_video = func.fill_with_keep(keep_forward, video)
        f_pred = model.predict_video(f_video)
        f_sm = F.softmax(f_pred, dim=1)
        f_js = func.jensen_shannon(f_sm, o_sm)
        f_js_t = torch.concatenate([f_js_t, f_js])

        b_video = func.fill_with_keep(keep_backward, video)
        b_pred = model.predict_video(b_video)
        b_sm = F.softmax(b_pred, dim=1)
        b_js = func.jensen_shannon(b_sm, o_sm)
        b_js_t = torch.concatenate([b_js_t, b_js])
    return {
        'forward': f_js_t,
        'backward': b_js_t
    }

def dataset_curves(dataset, model, method):
    level1_file = os.path.join(CONF.LEVEL_1_PATH, f'{dataset}_{model}.h5')
    out_path = CONF.OUT_PATH
    out_file = os.path.join(out_path, method)
    os.makedirs(out_file, exist_ok=True)
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
                greedy_js = get_greedy_js(video, model)
                idx_f = torch.argsort(greedy_js['forward'])
                idx_b = torch.argsort(greedy_js['backward'])
            if method == 'greedy':
                idx = {}
                for k in greedy_js:
                    idx[k] = torch.argsort(greedy_js[k])
            if method == 'foolish':
                idx = {}
                for k in greedy_js:
                    idx[k] = torch.argsort(-1*greedy_js[k])
            if method == 'random':
                idx = list(range(L))
                random.shuffle(idx)
            elif method == 'facility': # facility location
                emb = []
                for i in range(L):
                    emb.append(data[str(i)]['feat'][None,:])
                emb = np.concatenate(emb, axis=0)
                idx = func.emb_facilitylocation(emb)
            elif method == 'brute':
                best_idx = torch.argmin(js)
                o_logits = data['full']['logits']
                o_sm = F.softmax(torch.tensor(o_logits[None,:]), dim=1)
                idx = func.brute(video, best_idx, model, o_sm) 

            sim_ar_f, js_ar_f = get_video_curve(model, video, idx_f.cpu())
            sim_ar_b, js_ar_b = get_video_curve(model, video, idx_b.cpu())
            d={
                fname: {'sim_ar_f': torch.tensor(sim_ar_f), 'js_ar_f': torch.tensor(js_ar_f),
                        'sim_ar_b': torch.tensor(sim_ar_b), 'js_ar_b': torch.tensor(js_ar_b)}
            }

            func.save_dict_to_h5(f, d)
            
if __name__ == "__main__":
    dataset_curves('ucf101', 'r3d-18', 'greedy')
