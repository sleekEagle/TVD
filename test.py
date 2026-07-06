from dataloaders import data_paths
from models import get_model
import torch
import random

def eval(dataset, model):
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model)

    correct = 0
    for i in range(len(path_list)):
        print(f'{i} of {len(path_list)} is done.', end='\r')
        pred = model.predict_video(path_list[i])
        pred_cls = torch.argmax(pred,dim=1)
        if idx_list[i]==pred_cls:
            correct += 1

    print(f'Avg acc: {correct/len(path_list) * 100}')


if __name__ == "__main__":
    # for ucf101 dataset
    eval('ucf101', 'mc3-18')
    # eval('ucf101', 'r3d-18')

    #for ssv2 dataset
    

