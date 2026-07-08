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

def jensen_shannon(p, q, eps=1e-10):
    """Jensen-Shannon divergence between two probability distributions"""
    # Add small epsilon to avoid log(0)
    p = p + eps
    q = q + eps
    
    # Normalize to ensure they sum to 1 (if not already)
    p = p / p.sum()
    q = q / q.sum()
    
    # Compute KL divergences
    m = 0.5 * (p + q)
    kl_pm = (p * torch.log(p / m)).sum()
    kl_qm = (q * torch.log(q / m)).sum()
    
    return 0.5 * (kl_pm + kl_qm)

def get_video_curve(model, video, data, idx):
    o_feat = torch.from_numpy(data['full']['feat'])
    o_logit = torch.from_numpy(data['full']['logits'])
    o_sm = F.softmax(o_logit, dim=0)
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
        js = jensen_shannon(sm, o_sm)
        sim_ar.append(similarity.item())
        js_ar.append(js.item())
    
    return sim_ar, js_ar

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
            data = func.get_h5_item(level1_file, fname)
            L = video.size(2)
            idx = list(range(L))

            if method == 'random':
                random.shuffle(idx)
            elif method in ['greedy','foolish']:
                o_logits = data['full']['logits']
                o_cls = np.argmax(o_logits)

                max_l_list = []
                for i in range(L):
                    max_l_list.append(data[str(i)]['logits'][o_cls])
                max_l_list = np.array(max_l_list)
                idx = np.argsort(-1*max_l_list)

                if method == 'foolish':
                    idx = np.argsort(max_l_list)
            elif method == 'facility': # facility location
                emb = []
                for i in range(L):
                    emb.append(data[str(i)]['feat'][None,:])
                emb = np.concatenate(emb, axis=0)
                idx = func.emb_facilitylocation(emb)


            sim_ar, js_ar = get_video_curve(model, video, data, idx)
            d={
                fname: {'sim_ar': torch.tensor(sim_ar), 'js_ar': torch.tensor(js_ar)}
            }

            func.save_dict_to_h5(f, d)

if __name__ == "__main__":
    dataset_curves('ucf101', 'mc3-18', 'facility')

