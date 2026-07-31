import numpy as np
import os
import json
import CONF

def get_paths():
    annot_path = os.path.join(CONF.DIVING_META_PATH, 'Diving48_V2_test.json')

    with open(os.path.join(CONF.DIVING_META_PATH, 'id2label.json'), 'r') as f:
        id2label = json.load(f)

    paths, labels, cls_names = [], [], []
    num_samples_per_dataset = []
    data = json.load(open(annot_path))
    num_samples_per_dataset.append(len(data))
    for d in data:
        paths += [os.path.join(CONF.DIVING_DATA_PATH, f'{d['vid_name']}.mp4')]
        labels += [d['label']]
        cls_name = id2label[str(d['label'])]
        cls_names.append(cls_name)

    return paths, cls_names, labels


