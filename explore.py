from dataloaders import data_paths
from models import get_model
import torch
import random
import func
import os
import json
import h5py
import torch.nn.functional as F
import numpy as np
import CONF

from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt


def plot_point_traj(dataset, model, fname):
    level1_file = os.path.join(CONF.LEVEL_1_PATH, f'{dataset}_{model}.h5')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    path = [pl for pl in path_list if os.path.basename(pl) == fname][0]
    model = get_model.get_model(dataset, model)

    data = func.get_h5_item(level1_file, fname)
    video = model.get_video(path)
    L = video.size(2)

    # lets compare greedy vs brute in the embedding space
    # greedy
    greedy_idx = torch.argsort(js)

    js_scaled = (js-js.min())/(js.max()-js.min())
    point_sizes = (400-20)*(1-js_scaled)+40
    point_sizes = point_sizes.to(torch.int32)

    # brute
    js = func.get_js_emb(data)
    best_idx = torch.argmin(js)
    o_logits = data['full']['logits']
    o_sm = F.softmax(torch.tensor(o_logits[None,:]), dim=1)
    brute_idx = func.brute(video, best_idx, model, o_sm) 

    # get all individual + all embeddings (L+1). last emb is full 
    all_feat = torch.empty(0)
    for k in range(len(data.keys())-1):
        all_feat = torch.concatenate([all_feat, torch.tensor(data[str(k)]['feat'])[None,]],dim=0)
    all_feat = torch.concatenate([all_feat, torch.tensor(data['full']['feat'])[None,]],dim=0)

    embeddings_np = all_feat.numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=4)
    embeddings_2d = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], s=point_sizes)
    plt.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], 
            s=200,            # Size of the 'x'
            c='red',          # Color
            marker='x',       # 'x' marker
            linewidth=3,      # Thickness of the 'x'
            label=f'full')
    for i in range(embeddings_2d.shape[0]):
        plt.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.title('Embeddings Visualization (t-SNE)')
    plt.show()


def js_vs_dist(dataset, model_name):
    level1_file = os.path.join(CONF.LEVEL_1_PATH, f'{dataset}_{model_name}.h5')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    out_path = os.path.join(CONF.OUT_PATH, 'plots', 'js_vs_dist')
    os.makedirs(out_path, exist_ok=True)
    out_path_plot = os.path.join(out_path, f'{dataset}_{model_name}.png')
    out_path_txt = os.path.join(out_path, f'{dataset}_{model_name}.txt')
    model = get_model.get_model(dataset, model_name)

    js_vals, sim = [],[]
    avg_corr = 0
    for i in range(len(path_list)):
        print(f'{i} of {len(path_list)} is done.', end='\r')

        video = model.get_video(path_list[i])
        fname = os.path.basename(path_list[i])
        data = func.get_h5_item(level1_file, fname)
        L = video.size(2)
        idx = list(range(L))

        full_feat = data['full']['feat']
        ind_feat = torch.empty(0)
        for k in range(len(data.keys())-1):
            ind_feat = torch.concatenate([ind_feat, torch.tensor(data[str(k)]['feat'][None,:])], dim=0)

        js_vals_ = func.get_js_video(data).tolist()
        js_vals += js_vals_
        sim_ =  F.cosine_similarity(ind_feat, torch.tensor(full_feat)[None,:], dim=1).tolist()
        sim += sim_

        corr = np.corrcoef(js_vals_, sim_)[0, 1]
        avg_corr += corr
    avg_corr /= len(path_list)

    plt.figure(figsize=(8, 6))
    plt.scatter(js_vals, sim, s=5)
    plt.xlabel('JS')
    plt.ylabel('cosine')
    plt.savefig(out_path_plot, dpi=300)

    with open(out_path_txt, 'w') as file:
        file.write(f'Avg correlation coeff: {avg_corr}')


        
        



if __name__ == "__main__":
    # plot_point_traj('ucf101', 'r3d-18', 'v_Diving_g02_c04.avi')
    js_vs_dist('ucf101', 'mc3-18')