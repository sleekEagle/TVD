import os
import random
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import json

def get_pred(model, path):
    fname = os.path.basename(path)
    video = model.get_video(path)
    pred = model.predict_video(video).squeeze()
    feat = model.get_features()  
    return fname, pred, feat

#****************************************************************************************************************
#****************************************************************************************************************
# *****************  Temporal Freezing / interpolation Code *********************************************************************
#****************************************************************************************************************
#****************************************************************************************************************

def linear_interpolate_frames(video, keep):
    """
    video: Tensor (B, C, T, H, W)
    remove_indices: list of int, e.g., [3, 7, 12]
    Returns: video with removed frames linearly interpolated from neighbors
    """
    out = video.clone()
    T = video.shape[2]
    remove_indices = set(range(T)) - set(keep)
    
    for idx in sorted(remove_indices):
        before = max([i for i in keep if i < idx], default=None)
        after = min([i for i in keep if i > idx], default=None)
        
        if before is None:
            out[:, :, idx] = video[:, :, after]      # hold first valid
        elif after is None:
            out[:, :, idx] = video[:, :, before]     # hold last valid
        else:
            alpha = (idx - before) / (after - before)
            out[:, :, idx] = (1 - alpha) * video[:, :, before] + alpha * video[:, :, after]
    return out

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
    tofill_t = torch.tensor(tofill, device=video.device)
    fillwith_t = torch.tensor(fillwith, device=video.device)
    video[:, :, tofill_t] = video[:, :, fillwith_t].clone()

def fill_with_keep(keep, video, fill='past'):
    if fill != 'interp':
        fvideo = video.clone()
    if fill in ['past','future']:
        if fill == 'past':
            tofill, fillwith = past_fill(keep, video.size(2))
        if fill == 'future':
            tofill, fillwith = future_fill(keep, video.size(2))
        fill_video(tofill, fillwith, fvideo)
    elif fill == 'zero':
        z_idx = [i for i in range(video.size(2)) if i not in keep]
        fvideo[:,:,z_idx,:] = 0
    elif fill == 'mean':
        m_idx = [i for i in range(video.size(2)) if i not in keep]
        vid_mean = video[:,:,m_idx,:].mean(dim=(0,2,3,4),keepdim=True)
        fvideo[:,:,m_idx,:] = vid_mean
    elif fill == 'interp':
        fvideo = linear_interpolate_frames(video, keep)

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

def load_jsonl_to_dict(filepath):   
    """Load JSON Lines file into a dictionary keyed by filename"""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                entry = json.loads(line)
                k = list(entry.keys())[0]
                data[k] = entry[k]
    return data


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


from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from dataloaders import data_paths
from models import get_model

if __name__ == "__main__":
    # tofill, fillwith = future_fill([0,4,7,8,13,15])
    # import torch
    # video = torch.rand([1,3,16,112,112])


    # dataset = 'ucf101'
    # path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    # model = get_model.get_model(dataset, 'mc3-18')
    # video = model.get_video(path_list[102])
    # keep = [0,1,2,3,8,9,10,11,12,13,14,15]
    # fvideo = fill_with_keep(keep, video, 'interp')


    # fvideo_ = linear_interpolate_frames(video, keep)

    # grid = make_grid(video.squeeze(0).permute(1,0,2,3), nrow=video.size(2), normalize=True, pad_value=1)
    # fgrid = make_grid(fvideo.squeeze(0).permute(1,0,2,3), nrow=video.size(2), normalize=True, pad_value=1)
    # img = torch.concatenate([grid,fgrid],dim=1)

    
    # plt.imshow(img.permute(1,2,0).cpu())
    # plt.show()


    # # fvideo = fill_with_keep(keep, video, fill='mean')
    # # fvideo = video.clone()
    # # fill_video(tofill, fillwith, fvideo)
    pass