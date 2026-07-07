from dataloaders import data_paths
from models import get_model
import torch
import random
import func
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def save(dataset, model):
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model)

    data = {}
    for i in range(len(path_list)):
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



if __name__ == "__main__":
    # for ucf101 dataset
    # eval('ucf101', 'mc3-18')
    save('ucf101', 'r3d-18')