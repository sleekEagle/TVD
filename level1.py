from dataloaders import data_paths
from models import get_model
import torch
import random
import func
import os
import json
import h5py

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


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


out_path = r'D:\output\TVD\level1'
def save(dataset, model):
    out_file = os.path.join(out_path, f'{dataset}_{model}.h5')

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model)

    with h5py.File(out_file, "w") as f:
        for i in range(len(path_list)):
            print(f'{i} of {len(path_list)} is done.', end='\r')
            d_ = {}
            path = path_list[i]
            fname = os.path.basename(path)
            video = model.get_video(path)
            pred = model.predict_video(video).squeeze()
            feat = model.get_features()

            d_['full'] = {
                'logits': pred,
                'feat': feat
            }

            # get single-frame model beliefs
            for fidx in range(video.size(2)):
                tofill, fillwith = func.future_fill([fidx])
                fvideo = video.clone()
                func.fill_video(tofill, fillwith, fvideo)
                pred = model.predict_video(fvideo).squeeze()
                feat = model.get_features()
                d_[str(fidx)] = {
                    'logits': pred,
                    'feat': feat
                }
            d={
                fname: d_
            }

            save_dict_to_h5(f, d)

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
    
if __name__ == "__main__":
    # for ucf101 dataset
    save('ucf101', 'mc3-18')
    # save('ucf101', 'r3d-18')
