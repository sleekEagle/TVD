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
import analyze

def plot_JS_seq(dataset, model_name, fname):
    out_path = os.path.join(CONF.OUT_PATH, 'plots', 'JS_seq')
    os.makedirs(out_path, exist_ok=True)
    out_path_plot = os.path.join(out_path, f'{dataset}_{model_name}_{fname}.png')

    level1_file = os.path.join(CONF.LEVEL_1_PATH, f'{dataset}_{model_name}.h5')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    path = [pl for pl in path_list if os.path.basename(pl) == fname][0]
    model = get_model.get_model(dataset, model_name)

    data = func.get_h5_item(level1_file, fname)
    video = model.get_video(path)
    L = video.size(2)
    js = func.get_js_video(data)

    # lets compare greedy vs brute in the embedding space
    # greedy
    greedy_idx = torch.argsort(js)

    # brute
    best_idx = torch.argmin(js)
    o_logits = data['full']['logits']
    o_sm = F.softmax(torch.tensor(o_logits[None,:]), dim=1)
    brute_idx = func.brute(video, best_idx, model, o_sm) 

    assert brute_idx[0]==greedy_idx[0], 'brute and greedy does not match!'

    def normalize(ar):
        ar=np.array(ar)
        ar = (ar-ar.min())/(ar.max()-ar.min())
        return ar
    
    greedy_seq = analyze.get_video_curve(model, video, data, greedy_idx)[1]
    brute_seq = analyze.get_video_curve(model, video, data, brute_idx)[1]

    plt.figure(figsize=(8, 6))   
    plt.plot(greedy_seq, label='Greedy')
    plt.plot(brute_seq, label= 'Brute')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('JS Div')
    plt.savefig(out_path_plot, dpi=300)
    plt.show()
    pass


def plot_point_traj(dataset, model, fname):
    level1_file = os.path.join(CONF.LEVEL_1_PATH, f'{dataset}_{model}.h5')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    path = [pl for pl in path_list if os.path.basename(pl) == fname][0]
    model = get_model.get_model(dataset, model)

    data = func.get_h5_item(level1_file, fname)
    video = model.get_video(path)
    L = video.size(2)
    js = func.get_js_video(data)

    # get point sizes based on JS
    js_ = js
    js_scaled = (js_-js_.min())/(js_.max()-js_.min())
    point_sizes = (((300-20)*(1-js_scaled)+20))**1.2
    point_sizes = point_sizes.to(torch.int32)

    # lets compare greedy vs brute in the embedding space
    # greedy
    greedy_idx = torch.argsort(js)

    # brute
    best_idx = torch.argmin(js)
    o_logits = data['full']['logits']
    o_sm = F.softmax(torch.tensor(o_logits[None,:]), dim=1)
    brute_idx = func.brute(video, best_idx, model, o_sm) 

    assert brute_idx[0]==greedy_idx[0], 'brute and greedy does not match!'

    # get all individual + all embeddings (L+1). last emb is full 
    all_feat = torch.empty(0)
    for k in range(len(data.keys())-1):
        all_feat = torch.concatenate([all_feat, torch.tensor(data[str(k)]['feat'])[None,]],dim=0)
    all_feat = torch.concatenate([all_feat, torch.tensor(data['full']['feat'])[None,]],dim=0)
    # all_feat = F.softmax(all_feat,dim=1)

    embeddings_np = all_feat.numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=6)
    embeddings_2d = tsne.fit_transform(embeddings_np)

    order = brute_idx
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(embeddings_2d[:-1, 0], embeddings_2d[:-1, 1], s=point_sizes)
    ax.scatter(embeddings_2d[greedy_idx[0], 0], embeddings_2d[greedy_idx[0], 1], s=point_sizes[greedy_idx[0]], c='green')
    ax.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], 
            s=200,            # Size of the 'x'
            c='red',          # Color
            marker='x',       # 'x' marker
            linewidth=3,      # Thickness of the 'x'
            label=f'full')
    # show the first greedy point
    ax.scatter(embeddings_2d[greedy_idx[0], 0], embeddings_2d[greedy_idx[0], 1], s=200, c='green')
    for i, seq_idx in enumerate(order):
        plt.annotate(str(i), (embeddings_2d[seq_idx, 0], embeddings_2d[seq_idx, 1]))
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
    plot_JS_seq('ucf101', 'r3d-18', 'v_ApplyLipstick_g01_c02.avi')
    # js_vs_dist('ucf101', 'mc3-18')