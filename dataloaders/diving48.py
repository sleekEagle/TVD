import numpy as np
import os
import json
import CONF
from datasets import load_dataset

def get_paths():
    data_path = CONF.DIVING_PATH
    #read class names
    with open(os.path.join(data_path, 'id2label.json'), 'r') as f:
        id2label = json.load(f)

    # ds = load_dataset("bkprocovid19/diving48")

    train_path = os.path.join(data_path, "Diving48_V2_train.json")
    test_path = os.path.join(data_path, "Diving48_V2_test.json")

    train_data = json.load(open(train_path))
    test_data = json.load(open(test_path))
    test_data = train_data # comment after training

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


