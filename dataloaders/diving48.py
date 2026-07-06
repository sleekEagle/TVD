import numpy as np
import os
import json

def get_paths():
    data_path = r'D:\datasets\Diving48_rgb'
    #read class names
    with open(os.path.join(data_path, 'id2label.json'), 'r') as f:
        id2label = json.load(f)

    #split files downloaded from https://github.com/GaganKanojia/Attentive-spatio-temporal-representation-learning-for-diving-classification/tree/master
    test_data = np.load(os.path.join(data_path, 'test_data_64.npy'), allow_pickle=True)
    # train_data = np.load(r"D:\datasets\Diving48_rgb\train_data_64.npy", allow_pickle=True)
    # vocab = np.load(r"D:\datasets\Diving48_rgb\vocab.npy")

    data_path = os.path.join(data_path, 'rgb')
    paths, cls_names, cls_ind = [], [], []
    for d in test_data:
        l = d['label']
        name = d['vid_name']
        path = os.path.join(data_path, f'{name}.mp4')
        cls_name = id2label[str(l)]

        paths.append(path)
        cls_names.append(cls_name)
        cls_ind.append(l)
    
    return paths, cls_names, cls_ind

