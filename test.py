from dataloaders import data_paths
from models import get_model
import torch
import random

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def eval(dataset, model):
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model)

    correct = 0
    for i in range(len(path_list)):
        print(f'{i} of {len(path_list)} is done.', end='\r')
        video = model.get_video(path_list[i])
        pred = model.predict_video(video)        
        pred_cls = torch.argmax(pred,dim=1)
        if idx_list[i]==pred_cls:
            correct += 1

    print(f'Avg acc: {correct/len(path_list) * 100}')


if __name__ == "__main__":
    # for ucf101 dataset
    # eval('ucf101', 'mc3-18')
    # eval('ucf101', 'r3d-18')

    #for ssv2 dataset
    eval('ssv2','tformer_hr')
    # eval('ssv2','tformer_base')
    # eval('ssv2','vjepa2')

    # for diving48  dataset
    # eval('diving48', 'vjepa2')



