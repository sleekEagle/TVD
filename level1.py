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


    for i in range(len(path_list)):
        path = path_list[i]
        fname, pred, feat = func.get_pred(model, path)
        pass



if __name__ == "__main__":
    # for ucf101 dataset
    # eval('ucf101', 'mc3-18')
    save('ucf101', 'r3d-18')