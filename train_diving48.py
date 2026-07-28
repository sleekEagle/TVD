from dataloaders import data_paths
from models import get_model
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import CONF
import os

def save_features(dataset='diving48', model_name='vjepa2'):
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model_name)

    out_path = os.path.join(CONF.OUT_PATH,'training', dataset, model_name)
    os.makedirs(out_path, exist_ok=True)
    features, labels = torch.empty(0), []
    for i in tqdm(range(len(path_list))):
        if i==5: break
        video = model.get_video(path_list[i])
        _=model.predict_video(video)
        f = model.get_features()
        features = torch.concatenate([features, f[None,:].cpu()],dim=0)
        cls = idx_list[i]
        labels.append(cls)
    np.save(os.path.join(out_path, 'features.npy'), features.numpy())
    np.save(os.path.join(out_path, 'labels.npy'), np.array(labels))

    # np.load(os.path.join(out_path, 'labels.npy')).shape

if __name__ == "__main__":
    save_features()
    pass