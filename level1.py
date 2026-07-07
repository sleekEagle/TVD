from dataloaders import data_paths
from models import get_model
import torch
import random
import func
import os
import json
import h5py

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


out_path = r'D:\output\TVD\level1'
def save(dataset, model):
    out_file = os.path.join(out_path, f'{dataset}_{model}.pt')

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model)

    data = {}
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
            d_[fidx] = {
                'logits': pred,
                'feat': feat
            }

        data[fname] = d_

        if i%100 == 0 or i==len(path_list)-1:
            torch.save(data, out_file)


if __name__ == "__main__":
    # for ucf101 dataset
    # eval('ucf101', 'mc3-18')
    save('ucf101', 'r3d-18')

    # d = torch.load(r"D:\output\TVD\level1\ucf101_r3d-18.pt")
    # pass