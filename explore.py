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
from tqdm import tqdm


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
    
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    if dataset=='ucf101':
        basenames = [os.path.basename(s) for s in path_list]
        out_path = os.path.join(out_path, f'{fname}.png')
    elif dataset == 'ssv2':
        fn = fname.replace('/','_')
        out_path = os.path.join(out_path, f'{fn}.png')
        basenames = []
        for s in path_list:
            parent = os.path.basename(os.path.dirname(s))
            base = os.path.basename(s)
            str = f'{parent}/{base}'
            basenames.append(str)

    path_idx = basenames.index(fname)

    ward = 'forward' if forward else 'backward'
    stat_path = os.path.join(CONF.OUT_PATH, 'brute' ,f'curves_{dataset}_{model_name}_{ward}.jsonl')
    data = list(func.load_jsonl_to_dict(stat_path).items())[path_idx][1]
    js = np.array(data['js_ar'])
    idx = min(np.argwhere(js<thr))
    frames = data['idx'][:int(idx[0])+1]
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

def get_idx_to_remove_next(model, video, existing, analyze_cls):
    l_list = torch.empty(0).to(model.device)
    for idx in existing:
        keep = [i for i in existing if i!=idx]
        fvideo = func.fill_with_keep(keep, video, 'past')
        pred = model.predict_video(fvideo)
        pred_l = pred[:,analyze_cls]
        l_list = torch.concatenate([l_list, pred_l])

    amax = torch.argmax(l_list)
    bi = existing[amax]
    return bi, l_list[amax]
    
def plot_cls_importance(dataset, model_name, fname, forward, thr=1e-3, cls_thr=-1e-4):
    plot_path = os.path.join(CONF.OUT_PATH, 'plots' ,'cls_sfs')
    # get SFS
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    basenames = [os.path.basename(s) for s in path_list]
    path_idx = basenames.index(fname)

    # read JS data
    ward = 'forward' if forward else 'backward'
    js_path = os.path.join(CONF.OUT_PATH, 'brute' ,f'curves_{dataset}_{model_name}_{ward}.jsonl')
    data = list(func.load_jsonl_to_dict(js_path).items())[path_idx]
    assert basenames[path_idx] == data[0] , 'fname mismatch'

    # get pred class
    model = get_model.get_model(dataset, model_name)
    video = model.get_video(path_list[path_idx])

    #save image
    grid = make_grid(video.squeeze(0).permute(1,0,2,3), nrow=video.size(2), normalize=True, pad_value=1)
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_file_name = fname.split('.')[0]+'.png'
    plt.savefig(os.path.join(plot_path, f'{out_file_name}'), bbox_inches='tight', pad_inches=0, dpi=300)

    pred = model.predict_video(video)
    pred_cls_idx = torch.argmax(pred,dim=1).item()
    pred_cls = cls_list[idx_list.index(pred_cls_idx)]
    gt_cls = cls_list[path_idx]

    sm = F.softmax(pred, dim=1)
    sm_vals, indices = torch.topk(sm, 3, dim=1)
    
    js = np.array(data[1]['js_ar'])
    idx = min(np.argwhere(js<thr))
    frames = data[1]['idx'][:int(idx[0])+1]

    # read cls wise data
    cls_path = os.path.join(CONF.OUT_PATH, 'brute' ,f'cls_{dataset}_{model_name}_{ward}.jsonl')
    data = list(func.load_jsonl_to_dict(cls_path).items())[path_idx]
    assert basenames[path_idx] == data[0] , 'fname mismatch'

    data = data[1]
    f_left_list = {}
    for k in data:
        valid_ar = np.argwhere(np.array(data[k]['norm_l']) > cls_thr)
        if len(valid_ar)==0: continue
        idx = max(valid_ar)[0]
        start_f = data[k]['start_frames']
        rem_f = data[k]['rem_f'][:idx+1]
        f_left = [f for f in start_f if f not in rem_f]
        d_ = {'f_left': f_left,
         'cls_idx': data[k]['cls'],
         'cls_name': cls_list[idx_list.index(data[k]['cls'])]}
        f_left_list[k] = d_

    print('**********************************************************************')
    print(f'sm values : {sm_vals.cpu()[0].tolist()}')
    print(f'GT cls: {gt_cls}, pred cls: {pred_cls}')
    print(f'original frames: {frames}')
    print(f_left_list)
    print('**********************************************************************')


def cls_metrics(dataset, model_name, thr=-1e-3):
    data_path = os.path.join(CONF.OUT_PATH, 'brute', f'cls_{dataset}_{model_name}_backward.jsonl')
    data = func.load_jsonl_to_dict(data_path)

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model_name)

    # analyze the top class
    correct = 0
    n_frames = 0
    for i, k in tqdm(enumerate(data)):
        # get original pred
        path = path_list[i]
        video = model.get_video(path)
        pred = model.predict_video(video)
        pred = F.softmax(pred, dim=1)
        pred_cls = torch.argmax(pred, dim=1)
        o_sm = pred[0,pred_cls].item()
        sm_ar = np.array(data[k]['0']['sm'])
        sm_change = (sm_ar-o_sm)/o_sm
        argwhere = np.argwhere(sm_change > thr)

        if len(argwhere)>0: 
            idx = int(max(argwhere[:,0]))
            rem_f = data[k]['0']['rem_f'][:idx+1]
            existing_f = [f for f in data[k]['0']['start_frames'] if f not in rem_f]
            # assert torch.abs(max_l - data[k]['0']['l'][idx])<0.01, 'Does not match'
        else:
            existing_f = data[k]['0']['start_frames']

        #calc accuracy
        if len(existing_f) == video.size(2):
            fvideo = video
        else:
            fvideo = func.fill_with_keep(existing_f, video, 'past')
        pred = model.predict_video(fvideo)
        pred_cls = torch.argmax(pred[0,:])
        max_l = pred[0,pred_cls]

        if pred_cls==idx_list[i]:
            correct += 1

        n_frames += len(existing_f)

    print(f'acc: {correct/len(path_list)}, n_frames: {n_frames/len(path_list)}')


def print_sutable_samples():
    path_list, cls_list, idx_list = data_paths.get_paths('ssv2')
    data = func.load_jsonl_to_dict(r"D:\output\TVD\brute\curves_ssv2_vjepa2_backward.jsonl")
    for i, k in enumerate(data):
        js_ar = np.array(data[k]['js_ar'])
        idx = np.array(data[k]['idx'])
        sel_idx = idx[:min(np.argwhere(js_ar<1e-3))[0]+1]
        if len(sel_idx) >8:
            print(path_list[i])


'''
ucf101:
    mc3-18: 1.9947131905894793
    r3d-18: 1.1726143272535026

ssv2:
    vjepa2: 1.2088122605363985
    tformer-base: 1.0842911877394636
'''
def calc_multi_sfs_metrics(dataset, model_name, forward, thr=1e-3, cls_thr=-1e-4):
    ward = 'forward' if forward else 'backward'
    data_path = os.path.join(CONF.OUT_PATH, 'brute' ,f'multi_{dataset}_{model_name}_{ward}.jsonl')
    data_dict = func.load_jsonl_to_dict(data_path)

    n_sum = 0
    for k in data_dict:
        sfs = data_dict[k]
        n_sum += (len(sfs)+1)

    mean_sum = n_sum / len(data_dict)
    print('*********************************')
    print(f'mean n SFS: {mean_sum}')
    print('*********************************')

    # for i,k in enumerate(data_dict):
    #     sfs = data_dict[k]
    #     if len(sfs)>=1:
    #         print(i)

def plot_multi_sfs(dataset, model_name, forward, i, thr=1e-3):
    plot_path = os.path.join(CONF.OUT_PATH, 'results', 'plots', 'mul_sfs')

    ward = 'forward' if forward else 'backward'
    data_path = os.path.join(CONF.OUT_PATH, 'brute' ,f'multi_{dataset}_{model_name}_{ward}.jsonl')
    mul_data_dict = func.load_jsonl_to_dict(data_path)

    data_path = os.path.join(CONF.OUT_PATH, 'brute' ,f'curves_{dataset}_{model_name}_{ward}.jsonl')
    data_dict = func.load_jsonl_to_dict(data_path)

    sfs1 = data_dict[list(data_dict.keys())[i]]
    js_ar = np.array(sfs1['js_ar'])
    idx = min(np.argwhere(js_ar<thr))[0]
    sfs1 = [sfs1['idx'][:idx+1]]
    sfs2 = mul_data_dict[list(mul_data_dict.keys())[i]]
    mul_sfs = sfs1 + sfs2
    print(mul_sfs)

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model_name)
    video = model.get_video(path_list[i])

    #save image
    grid = make_grid(video.squeeze(0).permute(1,0,2,3), nrow=video.size(2), normalize=True, pad_value=1)
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_file_name = os.path.basename(os.path.dirname(path_list[i])) + '_' + os.path.basename(path_list[i]).split('.')[0] + '.png'
    print(f'{out_file_name} {mul_sfs}')

    plt.savefig(os.path.join(plot_path, f'{out_file_name}'), bbox_inches='tight', pad_inches=0, dpi=300)


def exchange_SFS(dataset, cls_model_name, sfs_model_name, thr=0.1):
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, cls_model_name)
    save_path = os.path.join(CONF.SAVE_PATH, 'results', 'exchange_SFS')
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{dataset}_cls-{cls_model_name}_sfs-{sfs_model_name}.txt')

    # reasd SFS
    sfs_path = os.path.join(CONF.SAVE_PATH, 'SFS', f'top1_{dataset}_{sfs_model_name}.jsonl')
    sfs_data = func.load_jsonl_to_dict(sfs_path)

    with open(save_path, 'a') as f:
        acc = 0
        for i, path in tqdm(enumerate(path_list)):
            bn = os.path.basename(path)
            dn = os.path.basename(os.path.dirname(path))
            fname = dn + '\\' + bn
            sfs = sfs_data[fname]

            sm = np.array([sfs['orig_sm']] + sfs['sm'])
            sm_norm = (sm-sfs['orig_sm'])/sfs['orig_sm']
            where = np.argwhere(sm_norm>-1*thr)
            n = where.max()
            rem_f = sfs['rem_f'][:n]
            keep = [f for f in sfs['start_frames'] if f not in rem_f]

            video = model.get_video(path)
            fvideo = func.fill_with_keep(keep, video, 'past')
            pred = model.predict_video(fvideo)
            pred_sm = F.softmax(pred, dim=1)
            pred_sm_cls = str(pred_sm[0, sfs['cls']].item())

            pred_cls = torch.argmax(pred_sm, dim=1)
            if pred_cls == idx_list[i]: acc += 1

            f.write(f'{pred_sm_cls} \n')
            f.flush()

        acc /= len(path_list)
        f.write(f'avg acc: {acc} \n')
        f.flush()





if __name__ == "__main__":
    # print_sutable_samples()
    # plot_cls_importance('ucf101', 'mc3-18', 'v_Archery_g02_c02.avi', forward = True, thr=1e-3)
    # plot_multi_sfs('ssv2', 'vjepa2', False, 1035)

    # js_vs_dist('ucf101', 'mc3-18')
    # cls_metrics('ucf101', 'r3d-18', thr=1e-2)
    exchange_SFS('ucf101', 'mc3-18', 'r3d-18')
    exchange_SFS('ucf101', 'mc3-18', 'mc3-18')
    pass