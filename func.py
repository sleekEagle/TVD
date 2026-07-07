import os
import random
import numpy as np
import h5py
import torch

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

if __name__ == "__main__":
    tofill, fillwith = future_fill([0,4,7,8,13,15])
    import torch
    video = torch.rand([1,3,16,112,112])
    fvideo = video.clone()
    fill_video(tofill, fillwith, fvideo)