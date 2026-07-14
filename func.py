import os
import random
import numpy as np
import h5py
import torch
import torch.nn.functional as F

def get_pred(model, path):
    fname = os.path.basename(path)
    video = model.get_video(path)
    pred = model.predict_video(video).squeeze()
    feat = model.get_features()  
    return fname, pred, feat

#****************************************************************************************************************
#****************************************************************************************************************
# *****************  Temporal Freezing Code *********************************************************************
#****************************************************************************************************************
#****************************************************************************************************************

'''
how to use:
    tofill, fillwith = future_fill([0,4,7,8,13,15])
    import torch
    video = torch.rand([1,3,16,112,112])
    fvideo = video.clone()
    fill_video(tofill, fillwith, fvideo)
'''

def past_fill(keep, l=16):
    tofill = [i for i in range(l) if i not in keep]
    
    fillwith = []
    keep = np.array(keep)
    for idx in tofill:
        # get immediate past item
        ar = np.sort(keep[keep<idx])
        if len(ar) > 0:
            k = int(ar[-1])
        else:
            ar = np.sort(keep[keep>idx])
            assert len(ar) > 0, 'no items found to fill'
            k = int(ar[0])
        fillwith.append(k)

    return tofill, fillwith

def future_fill(keep, l=16):
    tofill = [i for i in range(l) if i not in keep]
    
    fillwith = []
    keep = np.array(keep)
    for idx in tofill:
        # get immediate past item
        ar = np.sort(keep[keep>idx])
        if len(ar) > 0:
            k = int(ar[0])
        else:
            ar = np.sort(keep[keep<idx])
            assert len(ar) > 0, 'no items found to fill'
            k = int(ar[-1])
        fillwith.append(k)

    return tofill, fillwith

#in-place fill video [1, 3, 16, 112, 112]
def fill_video(tofill, fillwith, video):
    import torch
    tofill_t = torch.tensor(tofill)
    fillwith_t = torch.tensor(fillwith)
    video[:, :, tofill_t] = video[:, :, fillwith_t].clone()

def fill_with_keep(keep, video, fill='past'):
    if fill == 'past':
        tofill, fillwith = past_fill(keep)
    fvideo = video.clone()
    fill_video(tofill, fillwith, fvideo)
    return fvideo



#****************************************************************************************************************
#****************************************************************************************************************
# reading .h5 files
#****************************************************************************************************************
#****************************************************************************************************************
# get all keys from a given .h5 file
def get_h5_keys(filename):
    with h5py.File(filename, "r") as f:
        return list(f.keys())

# read 'our' nested dict structure from a given h5 file and a key
# use: get_h5_item(r"D:\output\TVD\level1\ucf101_r3d-18.h5", "v_ApplyEyeMakeup_g01_c01.avi" )
def get_h5_item(h5file, key):
    with h5py.File(h5file, "r") as f:
        g = f[key]
        d = {}
        for k in g.keys():
            d_ = {}
            for sub_k in g[k].keys():
                d_[sub_k] = g[k][sub_k][:]
            d[k] = d_
        return d
    
def save_dict_to_h5(group, dictionary):
    for key, value in dictionary.items():

        if isinstance(value, dict):
            # create nested group
            subgroup = group.create_group(key)
            save_dict_to_h5(subgroup, value)

        elif torch.is_tensor(value):
            # save tensor
            group.create_dataset(
                key,
                data=value.cpu().numpy()
            )

        elif isinstance(value, str):
            # save string as attribute
            group.attrs[key] = value


#****************************************************************************************************************
#****************************************************************************************************************
# select keyframes
#****************************************************************************************************************
#****************************************************************************************************************
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
    kl_pm = (p * torch.log(p / m)).sum(dim=1)
    kl_qm = (q * torch.log(q / m)).sum(dim=1)
    
    return 0.5 * (kl_pm + kl_qm)

def emb_facilitylocation(emb, k=16):
    from apricot import FacilityLocationSelection
    selector = FacilityLocationSelection(
        n_samples=k,
        metric="cosine"
    )
    selector.fit(emb)
    keyframe_indices = selector.ranking
    return keyframe_indices

def get_js_video(data):
    o_logits = data['full']['logits']
    o_sm = F.softmax(torch.tensor(o_logits[None,:]), dim=1)

    i_logits = []
    for i in range(len(data.keys())-1):
        i_logits.append(data[str(i)]['logits'][None,:])
    i_logits = np.concatenate(i_logits)
    i_logits = torch.tensor(i_logits)
    sm = F.softmax(i_logits, dim=1)
    js = jensen_shannon(sm, o_sm.repeat(16,1))

    return js

def brute(video, best_idx, model, o_sm):
    def get_best_idx(model, video, idx_present, o_sm):
        idx_left = list(set(range(video.size(2)))-set(idx_present))
        pred_sm_ar = torch.empty(0).to(model.device)
        for idx in idx_left:
            keep = idx_present + [idx]
            tofill, fillwith = past_fill(keep)
            fvideo = video.clone()
            fill_video(tofill, fillwith, fvideo)

            pred = model.predict_video(fvideo)
            pred_sm = F.softmax(pred,dim=1)
            pred_sm_ar = torch.concatenate([pred_sm_ar, pred_sm])

        js = jensen_shannon(pred_sm_ar.to(o_sm.device), o_sm.repeat(pred_sm_ar.size(0),1))
        bi = idx_left[torch.argmin(js)]
        return bi
    
    sel_idx = [int(best_idx)]
    for _ in range(video.size(2)-2):
        bi = get_best_idx(model, video, sel_idx, o_sm)
        sel_idx += [bi]
    sel_idx += list(set(range(video.size(2))) - set(sel_idx))

    return sel_idx




if __name__ == "__main__":
    tofill, fillwith = future_fill([0,4,7,8,13,15])
    import torch
    video = torch.rand([1,3,16,112,112])
    fvideo = video.clone()
    fill_video(tofill, fillwith, fvideo)