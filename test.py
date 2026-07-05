from dataloaders import data_paths
from models import get_model
import torch
import random

def eval(dataset, model):
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model)

    # idx = list(range(0,len(path_list)))
    # random.shuffle(idx)
    # path_list = [path_list[i] for i in idx]
    # cls_list = [cls_list[i] for i in idx]
    # idx_list = [idx_list[i] for i in idx]
    

    correct = 0
    for i in range(len(path_list)):
        print(f'{i} of {len(path_list)} is done.', end='\r')
        pred = model.predict_video(path_list[i])
        pred_cls = torch.argmax(pred,dim=1)
        if idx_list[i]==pred_cls:
            correct += 1

    print(f'Avg acc: {correct/len(path_list) * 100}')


if __name__ == "__main__":
    # eval('ucf101', 'mc3-18')
    eval('ucf101', 'r3d-18')


