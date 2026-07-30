import numpy as np
import os
import json
import CONF

def get_paths():
    data_path = CONF.DIVING_PATH
    annot_path = os.path.join(data_path, 'Diving48_V2_test.json')

    with open(os.path.join(data_path, 'id2label.json'), 'r') as f:
        id2label = json.load(f)

    paths, labels, cls_names = [], [], []
    num_samples_per_dataset = []
    rgb_path = os.path.join(data_path, 'rgb')
    data = json.load(open(annot_path))
    num_samples = len(data)
    num_samples_per_dataset.append(len(data))
    for d in data:
        paths += [os.path.join(rgb_path, f'{d['vid_name']}.mp4')]
        labels += [d['label']]
        cls_name = id2label[str(d['label'])]
        cls_names.append(cls_name)

    return paths, cls_names, labels


