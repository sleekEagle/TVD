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
from torchvision.utils import make_grid

def plot_JS_seq(dataset, model_name, fname, forward):
    out_path = os.path.join(CONF.OUT_PATH, 'plots', 'JS_seq')
    os.makedirs(out_path, exist_ok=True)
    out_path_plot = os.path.join(out_path, f'{dataset}_{model_name}_{fname}.png')

    ward = 'forward' if forward else 'backward'
    methods = ['brute', 'greedy', 'foolish', 'facility', 'random']
    js_ar = []

    styles = {
        'brute': {'marker': 'o', 'linestyle': '-'},
        'greedy': {'marker': 's', 'linestyle': '--'},
        'foolish': {'marker': '^', 'linestyle': '-.'},
        'facility': {'marker': 'D', 'linestyle': ':'},
        'random': {'marker': 'P', 'linestyle': '-'}
    }
    marker_loc = [5,5,5,5,5]
    plt.figure(figsize=(8, 6))
    for i,meth in enumerate(methods):
        if meth in ['brute', 'greedy', 'foolish']:
            stat_path = os.path.join(CONF.OUT_PATH,meth,f'curves_{dataset}_{model_name}_{ward}.jsonl')
        else:
            stat_path = os.path.join(CONF.OUT_PATH,meth,f'curves_{dataset}_{model_name}.jsonl')

        data = func.load_jsonl_to_dict(stat_path)[fname]
        js = data['js_ar']
        js_ar.append(js)
        style = styles.get(meth, {'marker': 'o', 'linestyle': '-'})
        plt.plot(js, label=meth, marker=style['marker'], 
             linestyle=style['linestyle'], markersize=6, markevery=[marker_loc[i]])

    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('JS Div')
    plt.ylim(0,0.02)

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

def plot_frames(dataset, model_name, fname, forward, thr=1e-3):
    out_path = r'D:\output\TVD\plots\frames'
    out_path = os.path.join(out_path, f'{fname}.png')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    basenames = [os.path.basename(s) for s in path_list]
    path_idx = basenames.index(fname)

    ward = 'forward' if forward else 'backward'
    stat_path = os.path.join(CONF.OUT_PATH, 'brute' ,f'curves_{dataset}_{model_name}_{ward}.jsonl')
    data = func.load_jsonl_to_dict(stat_path)[fname]
    js = np.array(data['js_ar'])
    idx = min(np.argwhere(js<thr))
    frames = data['idx'][:int(idx[0])]
    print(f'Frames: {frames}')

    model = get_model.get_model(dataset, model_name)
    video = model.get_video(path_list[path_idx])

    grid = make_grid(video.squeeze(0).permute(1,0,2,3), nrow=video.size(2), normalize=True, pad_value=1)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid.permute(1,2,0).cpu().numpy())
    ax.axis('off')

    n_frames = video.size(2)
    frame_height = video.size(3)
    frame_width = video.size(4)
    for idx in frames:
        x_center = idx * (frame_width + 2) + frame_width // 2
        y_top = -10 # Position above the frame
        ax.plot(x_center, y_top, marker='v', markersize=6, 
            color='red', linestyle='none')
        # circle = plt.Circle((x_center, y_top), radius=10, 
        #                     color='red', fill=True, linewidth=0)
        # ax.add_patch(circle)

    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max-14) 
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_frames('ucf101', 'mc3-18', 'v_BalanceBeam_g03_c04.avi', forward = True)
    # js_vs_dist('ucf101', 'mc3-18')
    pass