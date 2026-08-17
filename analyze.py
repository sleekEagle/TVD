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
from tqdm import tqdm
import json
import random

def get_video_curve(model, video, idx, forward):
    o_sm = F.softmax(model.predict_video(video),dim=1)
    o_feat = model.get_features()
    L = video.size(2)

    sim_ar, js_ar = [], []
    filled = []
    for i in idx:
        fvideo = video.clone()
        filled.append(i)
        tofill, fillwith = func.past_fill(filled, L)
        if len(tofill)>0 and len(fillwith)>0:
            func.fill_video(tofill, fillwith, fvideo)
        pred = model.predict_video(fvideo)  
        sm = F.softmax(pred, dim=1).to(o_feat.device)
        feat = model.get_features().to(o_feat.device)

        similarity = F.cosine_similarity(feat, o_feat, dim=0)
        js = func.jensen_shannon(sm, o_sm)
        sim_ar.append(similarity.item())
        js_ar.append(js.item())
    
    return sim_ar, js_ar

def get_video_logit(model, video, idx, analyze_cls=None):
    if not analyze_cls:
        L = video.size(2)
        pred = model.predict_video(video)
        pred = F.softmax(pred, dim=1)
        o_cls = torch.argmax(pred,dim=1)
        o_l = pred[0,o_cls]
    else: 
        o_cls = analyze_cls

    change_ar, cls_ar = [], []
    filled = []
    for i in idx:
        fvideo = video.clone()
        filled.append(i)
        tofill, fillwith = func.past_fill(filled, L)
        if len(tofill)>0 and len(fillwith)>0:
            func.fill_video(tofill, fillwith, fvideo)
        pred = model.predict_video(fvideo) 
        p_cls = torch.argmax(pred, dim=1) 
        pred = F.softmax(pred, dim=1)
        p_l = pred[0,o_cls]
        change = (p_l)/o_l
        change_ar.append(change.item())
        cls_ar.append(p_cls.item())
    
    return change_ar, cls_ar

'''
forward:
    lowest js: best
backward:
    highest js: best
'''
def get_greedy_js(video, model, forward):
    L = video.size(2)
    o_logits = model.predict_video(video)
    o_sm = F.softmax(o_logits, dim=1)

    js_t = torch.empty(0).to(model.device)
    for i in range(L):
        if forward:
            keep_forward = [i]
            filled_video = func.fill_with_keep(keep_forward, video)
            pred = model.predict_video(filled_video)
            sm = F.softmax(pred, dim=1)
            js = func.jensen_shannon(sm, o_sm)
            js_t = torch.concatenate([js_t, js])
        else:
            keep_backward = [idx for idx in range(L) if idx!=i]
            filled_video = func.fill_with_keep(keep_backward, video)
            pred = model.predict_video(filled_video)
            sm = F.softmax(pred, dim=1)
            js = func.jensen_shannon(sm, o_sm)
            js_t = torch.concatenate([js_t, js])
    return js_t

def get_greedy_logit(video, model, forward, analyze_cls=None):
    if not analyze_cls:
        L = video.size(2)
        o_logits = model.predict_video(video)
        o_cls = torch.argmax(o_logits,dim=1)
        o_l = o_logits[0,o_cls]
    else: 
        o_cls = analyze_cls
 
    logit_t = torch.empty(0).to(model.device)
    for i in range(L):
        if forward:
            keep_forward = [i]
            filled_video = func.fill_with_keep(keep_forward, video)
            p_logits = model.predict_video(filled_video)
            p_cls = torch.argmax(p_logits,dim=1)
            p_l = p_logits[0,o_cls]
            logit_t = torch.concatenate([logit_t, p_l])
        else:
            keep_backward = [idx for idx in range(L) if idx!=i]
            filled_video = func.fill_with_keep(keep_backward, video)
            p_logits = model.predict_video(filled_video)
            p_cls = torch.argmax(p_logits,dim=1)
            p_l = p_logits[0,o_cls]
            logit_t = torch.concatenate([logit_t, p_l])
    return logit_t

def emb_facilitylocation(emb, k=16):
    from apricot import FacilityLocationSelection
    selector = FacilityLocationSelection(
        n_samples=k,
        metric="cosine"
    )
    selector.fit(emb)
    keyframe_indices = selector.ranking
    return keyframe_indices


def brute_logit(video, model, greedy_l, forward, analyze_cls=None):
    L = video.size(2)
    o_l = model.predict_video(video)
    if not analyze_cls:
        o_cls = torch.argmax(o_l, dim=1)
    else:
        o_cls = analyze_cls

    idx_sort = np.argsort(-1*greedy_l)
    best_idx = int(idx_sort[0])

    def get_best_idx_forward(model, video, idx_present, analyze_cls):
        idx_left = list(set(range(video.size(2)))-set(idx_present))
        l_list = torch.empty(0).to(model.device)
        for idx in idx_left:
            keep = idx_present + [idx]
            tofill, fillwith = func.past_fill(keep, video.size(2))
            fvideo = video.clone()
            func.fill_video(tofill, fillwith, fvideo)

            pred = model.predict_video(fvideo)
            pred_l = pred[0,analyze_cls]
            l_list = torch.concatenate([l_list, pred_l])

        bi = idx_left[torch.argmax(l_list)]
        return bi
    
    def get_best_idx_backward(model, video, idx_remove, analyze_cls):
        idx_left = list(set(range(video.size(2)))-set(idx_remove))
        l_list = torch.empty(0).to(model.device)
        for idx in idx_left:
            remove = idx_remove + [idx]
            keep = list(set(range(video.size(2)))-set(remove))
            tofill, fillwith = func.past_fill(keep, video.size(2))
            fvideo = video.clone()
            func.fill_video(tofill, fillwith, fvideo)

            pred = model.predict_video(fvideo)
            pred_l = pred[0,analyze_cls]
            l_list = torch.concatenate([l_list, pred_l])

        bi = idx_left[torch.argmax(l_list)]
        return bi
    
    sel_idx = [best_idx]
    for _ in range(video.size(2)-2):
        if forward:
            bi = get_best_idx_forward(model, video, sel_idx, o_cls)
        else:
            bi = get_best_idx_backward(model, video, sel_idx, o_cls)
        sel_idx += [bi]
    sel_idx += list(set(range(video.size(2))) - set(sel_idx))

    if not forward: # most important must be the first
        sel_idx = sel_idx[::-1]

    return sel_idx


def brute(video, model, greedy_js, forward):
    L = video.size(2)
    o_logits = model.predict_video(video)
    o_sm = F.softmax(o_logits, dim=1)

    idx_sort = np.argsort(greedy_js)
    best_idx = int(idx_sort[0])

    def get_best_idx_forward(model, video, idx_present, o_sm):
        idx_left = list(set(range(video.size(2)))-set(idx_present))
        pred_sm_ar = torch.empty(0).to(model.device)
        for idx in idx_left:
            keep = idx_present + [idx]
            tofill, fillwith = func.past_fill(keep, video.size(2))
            fvideo = video.clone()
            func.fill_video(tofill, fillwith, fvideo)

            pred = model.predict_video(fvideo)
            pred_sm = F.softmax(pred,dim=1)
            pred_sm_ar = torch.concatenate([pred_sm_ar, pred_sm])

        js = func.jensen_shannon(pred_sm_ar.to(o_sm.device), o_sm.repeat(pred_sm_ar.size(0),1))
        bi = idx_left[torch.argmin(js)]
        return bi
    
    def get_best_idx_backward(model, video, idx_remove, o_sm):
        idx_left = list(set(range(video.size(2)))-set(idx_remove))
        pred_sm_ar = torch.empty(0).to(model.device)
        for idx in idx_left:
            remove = idx_remove + [idx]
            keep = list(set(range(video.size(2)))-set(remove))
            tofill, fillwith = func.past_fill(keep, video.size(2))
            fvideo = video.clone()
            func.fill_video(tofill, fillwith, fvideo)

            pred = model.predict_video(fvideo)
            pred_sm = F.softmax(pred,dim=1)
            pred_sm_ar = torch.concatenate([pred_sm_ar, pred_sm])

        js = func.jensen_shannon(pred_sm_ar.to(o_sm.device), o_sm.repeat(pred_sm_ar.size(0),1))
        bi = idx_left[torch.argmin(js)]
        return bi
    
    sel_idx = [best_idx]
    for _ in range(video.size(2)-2):
        if forward:
            bi = get_best_idx_forward(model, video, sel_idx, o_sm)
        else:
            bi = get_best_idx_backward(model, video, sel_idx, o_sm)
        sel_idx += [bi]
    sel_idx += list(set(range(video.size(2))) - set(sel_idx))

    if not forward: # most important must be the first
        sel_idx = sel_idx[::-1]

    return sel_idx


def find_sfs_single(video, model, existing, totry, o_sm):
    js_vals = []
    for i in range(len(totry)):
        tt = [totry[i]]
        fvideo = func.fill_with_keep(existing + tt, video, fill='past')
        pred = model.predict_video(fvideo)
        sm = F.softmax(pred, dim=1)
        js = func.jensen_shannon(sm, o_sm)
        js_vals.append(js.item())
    return np.array(js_vals)

def find_sfs_cummulative(video, model, existing, totry, o_sm, thr=1e-3):
    for i in range(len(totry)+1):
        if len(existing) +  len(totry[:i]) == 0:
            continue
        fvideo = func.fill_with_keep(existing + totry[:i], video, fill='past')
        pred = model.predict_video(fvideo)
        sm = F.softmax(pred, dim=1)
        js = func.jensen_shannon(sm, o_sm)
        if js<=thr:
            return totry[:i]
    return -1




'''
forward: forward or backward selection. not applicable for method=facility
'''



def dataset_curves(dataset, model, method, forward = True):
    out_path = CONF.OUT_PATH
    out_file = os.path.join(out_path, method)
    os.makedirs(out_file, exist_ok=True)
    if method in ['random', 'facility']:
        out_file = os.path.join(out_file, f'curves_{dataset}_{model}.jsonl') 
    else:
        ward = 'forward' if forward else 'backward'
        out_file = os.path.join(out_file, f'curves_{dataset}_{model}_{ward}.jsonl') 

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model)

    # resume from earlier file
    if os.path.exists(out_file):
        existing_data = func.load_jsonl_to_dict(out_file)
    else:
        existing_data = {}

    with open(out_file, 'a') as f:
        for i in tqdm(range(len(path_list))):
            # print(f'{i} of {len(path_list)} is done.', end='\r', flush=True)

            video = model.get_video(path_list[i])
            video = video.to(model.device)
            fname = os.path.basename(path_list[i])
            L = video.size(2)
            
            if fname in existing_data: continue

            if method in ['greedy','foolish','brute']:
                greedy_js = get_greedy_js(video, model, forward).cpu().numpy()
            if method == 'greedy':
                if forward: 
                    idx = np.argsort(greedy_js)
                else:
                    idx = np.argsort(-1*greedy_js)
            if method == 'foolish':
                if forward: 
                    idx = np.argsort(-1*greedy_js)
                else:
                    idx = np.argsort(greedy_js)
            if method == 'random':
                idx = list(range(L))
                random.shuffle(idx)
            elif method == 'facility': # facility location
                emb = torch.empty(0)
                for i in range(L):
                    _ = model.predict_video(video)
                    emb = torch.concatenate([emb, model.get_features()[None,:].to(emb.device)], dim=0)
                idx = emb_facilitylocation(emb, video.size(2))
            elif method == 'brute':
                idx = brute(video, model, greedy_js, forward) 

            sim_ar, js_ar = get_video_curve(model, video, idx, forward)

            # import matplotlib.pyplot as plt
            # plt.plot(js_ar_f)
            # plt.plot(js_ar_b)
            if isinstance(sim_ar, np.ndarray):
                sim_ar = sim_ar.tolist()
            if isinstance(js_ar, np.ndarray):
                js_ar = js_ar.tolist()
            if isinstance(idx, np.ndarray):
                idx = idx.tolist()
            d={
                fname: {'sim_ar': sim_ar, 'js_ar': js_ar, 'idx': idx}
            }

            f.write(json.dumps(d) + '\n')
            f.flush()


def get_idx_to_remove_next(model, video, existing, analyze_cls):
    # o_pred = model.predict_video(video)
    # o_sm = F.softmax(o_pred, dim=1)

    l_list = torch.empty(0).to(model.device)
    sm_list = torch.empty(0).to(model.device)
    for idx in existing:
        keep = [i for i in existing if i!=idx]
        fvideo = func.fill_with_keep(keep, video, 'past')
        pred = model.predict_video(fvideo)
        pred_l = pred[:,analyze_cls]
        pred_sm = F.softmax(pred, dim=1)[:,analyze_cls]
        l_list = torch.concatenate([l_list, pred_l])
        sm_list = torch.concatenate([sm_list, pred_sm])

    amax = torch.argmax(l_list)
    bi = existing[amax]
    return bi, l_list[amax], sm_list[amax]

def calc_SFS(video, model, analyze_cls):
    frames = list(range(video.size(2)))
    frame_totry = frames.copy()
    removed_frame = []
    result_logits = []
    result_sm = []
    while len(frame_totry)>=2:
        bi, bl, bsm = get_idx_to_remove_next(model, video, frame_totry, analyze_cls=analyze_cls)
        removed_frame.append(bi)
        result_logits.append(bl.item())
        result_sm.append(bsm.item())
        frame_totry = [f for f in frame_totry if f!=bi]
    res = {
        'start_frames': frames,
        'rem_f': removed_frame,
        'l': result_logits,
        'sm': result_sm
    }
    return res

def resume_SFS(video, model, existing, analyze_cls):
    frames = existing
    frame_totry = frames.copy()
    removed_frame = []
    result_logits = []
    result_sm = []
    # get start sm
    fvideo = func.fill_with_keep(existing, video)
    pred = model.predict_video(fvideo)
    existing_sm = F.softmax(pred, dim=1)[0,analyze_cls]

    while len(frame_totry)>=2:
        bi, bl, bsm = get_idx_to_remove_next(model, video, frame_totry, analyze_cls=analyze_cls)
        removed_frame.append(bi)
        result_logits.append(bl.item())
        result_sm.append(bsm.item())
        frame_totry = [f for f in frame_totry if f!=bi]
    res = {
        'start_frames': frames,
        'rem_f': removed_frame,
        'l': result_logits,
        'sm': result_sm,
        'existing_sm': existing_sm
    }
    return res

def calc_metrics(video, model, analyze_cls, ordered_frames):
    result_logits = []
    result_sm = []
    removed_frame = []
    for i in range(len(ordered_frames)-1):
        keep = ordered_frames[i+1:]
        fvideo = func.fill_with_keep(keep, video, 'past')
        pred = model.predict_video(fvideo)
        pred_l = pred[:,analyze_cls]
        pred_sm = F.softmax(pred, dim=1)[:,analyze_cls]
        result_logits.append(pred_l.item())
        result_sm.append(pred_sm.item())
    removed_frame = ordered_frames[:-1]

    return removed_frame, result_logits, result_sm

def calc_random(video, model, analyze_cls):
    frames = list(range(video.size(2)))
    frame_totry = frames.copy()
    random.shuffle(frame_totry)

    removed_frame, result_logits, result_sm =  calc_metrics(video, model, analyze_cls, frame_totry)

    res = {
        'start_frames': frames,
        'rem_f': removed_frame,
        'l': result_logits,
        'sm': result_sm
    }
    return res

def calc_gradcam(video, model, analyze_cls):
    from captum.attr import LayerGradCam
    grad_cam = LayerGradCam(model, model.gradcam_layer)
    attribution = grad_cam.attribute(
        video,
        target=analyze_cls
    )
    f_attrib = model.prep_gradcam_feat(attribution)
    if f_attrib.size(0)!=video.size(2):
        f_reshaped = f_attrib.view(1, 1, -1)
        x_interpolated = F.interpolate(f_reshaped, size=video.size(2), mode='linear', align_corners=False)
        f_attrib = x_interpolated.squeeze()

    frame_totry = torch.argsort(f_attrib).cpu().tolist()
    removed_frame, result_logits, result_sm =  calc_metrics(video, model, analyze_cls, frame_totry)

    res = {
        'start_frames': list(range(video.size(2))),
        'rem_f': removed_frame,
        'l': result_logits,
        'sm': result_sm
    }
    return res

def calc_IG(video, model, analyze_cls):
    from captum.attr import IntegratedGradients
    model.eval()
    ig = IntegratedGradients(model)
    attribution = ig.attribute(
        video,
        target=analyze_cls,
        n_steps=10,
        internal_batch_size=2
    )
    f_attrib = torch.mean(attribution, dim=(1,3,4)).squeeze()
    frame_totry = torch.argsort(f_attrib).cpu().tolist()
    removed_frame, result_logits, result_sm =  calc_metrics(video, model, analyze_cls, frame_totry)

    res = {
        'start_frames': list(range(video.size(2))),
        'rem_f': removed_frame,
        'l': result_logits,
        'sm': result_sm
    }
    return res

def calc_saliency(video, model, analyze_cls):
    from captum.attr import Saliency

    saliency = Saliency(model)
    attribution = saliency.attribute(
        video,
        target=analyze_cls
    )
    f_attrib = torch.mean(attribution, dim=(3,4)).squeeze()
    frame_totry = torch.argsort(f_attrib).cpu().tolist()
    removed_frame, result_logits, result_sm =  calc_metrics(video, model, analyze_cls, frame_totry)

    res = {
        'start_frames': list(range(video.size(2))),
        'rem_f': removed_frame,
        'l': result_logits,
        'sm': result_sm
    }
    return res

    


def dataset_curves_cls_refine(dataset, model_name, forward = True, js_thr=1e-3):
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model_name)
    ward = 'forward' if forward else 'backward'
    data_path = os.path.join(CONF.OUT_PATH, 'brute', f'curves_{dataset}_{model_name}_{ward}.jsonl')
    data = func.load_jsonl_to_dict(data_path)

    out_path = os.path.join(CONF.SAVE_PATH, 'brute', f'cls_{dataset}_{model_name}_{ward}.jsonl')
    
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    #sanity check
    for i,k in enumerate(data.keys()):
        assert k == os.path.basename(path_list[i]), 'key mismatch'
    
    # skip when its already there
    if os.path.exists(out_path):
        existing = func.load_jsonl_to_dict(out_path)
    else:
        existing = {}

    with open(out_path, 'a') as f:
        for i in tqdm(range(len(path_list))):
            video = model.get_video(path_list[i])
            video = video.to(model.device)
            fname = list(data.keys())[i]
            d = data[fname]

            if fname in existing: continue # skip its there

            js = np.array(d['js_ar'])
            idx = min(np.argwhere(js<js_thr))
            frames = d['idx'][:int(idx[0])+1]
            o_logits = model.predict_video(video)
            o_sm = F.softmax(o_logits, dim=1)

            K=3
            logits, indices = torch.topk(o_logits, K, dim=1)
            sms, _ = torch.topk(o_sm, K, dim=1)

            # filled sm
            if len(frames) == video.size(2):
                fsms = sms
            else:
                fvideo = func.fill_with_keep(frames, video, 'past')
                f_pred = model.predict_video(fvideo)
                f_sm = F.softmax(f_pred, dim=1)
                fsms, _ = torch.topk(f_sm, K, dim=1)

            res = {}
            for k in range(K):
                cls_idx = indices[0,k]
                o_logit = logits[0,k]

                frame_totry = frames.copy()
                removed_frame = []
                result_logits = []
                result_sm = []
                norm_logits = []
                while len(frame_totry)>=2:
                    bi, bl, bsm = get_idx_to_remove_next(model, video, frame_totry, analyze_cls=cls_idx)

                    # fvideo = func.fill_with_keep([i for i in frame_totry if i!=bi], video, 'past')
                    # pred = model.predict_video(fvideo)
                    # pred[0,i]

                    removed_frame.append(bi)
                    result_logits.append(bl.item())
                    result_sm.append(bsm.item())
                    frame_totry = [f for f in frame_totry if f!=bi]
                    norm_logits.append(((bl - o_logit) / o_logit).item())
                res[k] = {
                    'start_frames': frames,
                    'rem_f': removed_frame,
                    'l': result_logits,
                    'sm': result_sm,
                    'norm_l': norm_logits,
                    'cls': cls_idx.item()
                }
            res['orig_sm'] = sms.cpu()[0].tolist()
            res['filled_sm'] = fsms.cpu()[0].tolist()
            d = {fname: res}
            f.write(json.dumps(d) + '\n')
            f.flush()

            # fvideo = func.fill_with_keep([i for i in frames if i not in [5,12, 8,1, 10]], video, 'past')
            # pred = model.predict_video(fvideo)
            # pred[0,0]

def dataset_curves_cls(dataset, model_name, method):
    out_path = CONF.SAVE_PATH
    out_file = os.path.join(out_path, method, f'top1_{dataset}_{model_name}.jsonl')

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    model = get_model.get_model(dataset, model_name)

    if method=='facility':
        fac_file = os.path.join(os.path.dirname(out_file), f'curves_{dataset}_{model_name}.jsonl')
        fac_d = func.load_jsonl_to_dict(fac_file)

    # skip when its already there
    if os.path.exists(out_file):
        existing = func.load_jsonl_to_dict(out_file)
    else:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        existing = {}

    with open(out_file, 'a') as f:
        for i in tqdm(range(len(path_list))):
            if i<955:continue
            video = model.get_video(path_list[i])
            video = video.to(model.device)
            logits = model.predict_video(video)
            sms = F.softmax(logits, dim=1)
            argmax = torch.argmax(logits, dim=1)
            o_logit = logits[0,argmax][0]
            o_sm = sms[0,argmax][0]
            cls_idx = argmax[0]

            fname = os.path.basename(os.path.dirname(path_list[i]))+'\\'+os.path.basename(path_list[i])
            if fname in existing: continue # skip its there

            if method=='sfs':
                res = calc_SFS(video, model, cls_idx)
            elif method=='random':
                 res = calc_random(video, model, cls_idx)
            elif method=='gradcam':
                res = calc_gradcam(video, model, cls_idx)
            elif method=='ig':
                res = calc_IG(video, model, cls_idx)
            elif method=='sal':
                res = calc_saliency(video, model, cls_idx)
            elif method=='facility':
                k = list(fac_d.keys())[i]
                frames = fac_d[k]['idx'][::-1] # turn into least to most important
                removed_frame, result_logits, result_sm =  calc_metrics(video, model, cls_idx, frames)
                res = {
                    'start_frames': list(range(video.size(2))),
                    'rem_f': removed_frame,
                    'l': result_logits,
                    'sm': result_sm
                 }
            

            res['orig_l'] = o_logit.item()
            res['orig_sm'] = o_sm.item()
            res['cls'] = cls_idx.item()
            res['gt_cls'] = idx_list[i]

            d = {fname: res}
            f.write(json.dumps(d) + '\n')
            f.flush()

            # fvideo = func.fill_with_keep([i for i in frames if i not in [5,12, 8,1, 10]], video, 'past')
            # pred = model.predict_video(fvideo)
            # pred[0,0]



def dataset_multiple_SFS(dataset, model_name, method, forward = True, thr=1e-3):
    out_path = CONF.SAVE_PATH
    out_file = os.path.join(out_path, method)
    if method in ['random', 'facility']:
        out_path = os.path.join(out_file, f'multi_{dataset}_{model_name}.jsonl') 
        data_path = os.path.join(out_file, f'curves_{dataset}_{model_name}.jsonl') 
    else:
        ward = 'forward' if forward else 'backward'
        out_path = os.path.join(out_file, f'multi_{dataset}_{model_name}_{ward}.jsonl') 
        data_path = os.path.join(out_file, f'curves_{dataset}_{model_name}_{ward}.jsonl')

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    data = func.load_jsonl_to_dict(data_path)
    model = get_model.get_model(dataset, model_name)

    with open(out_path, 'a') as f:
        for path in tqdm(path_list):
            # base_list = [os.path.basename(p) for p in path_list]
            # path_idx = base_list.index(fname)
            # path = path_list[path_idx]
            video = model.get_video(path)
            L = video.size(2)
            fname = os.path.basename(path)
            f_idx = data[fname]['idx']
            js_ar = np.array(data[fname]['js_ar'])
            n = np.argwhere(js_ar<thr).min()
            valid_idx = f_idx[:n+1]
            other_idx = f_idx[n+1:]

            o_logits = model.predict_video(video)
            o_sm = F.softmax(o_logits, dim=1)

            new_idx = []
            new_idx_list = []
            used_idx_list = []
            search_idx = other_idx
            while True:
                new_idx = find_sfs_cummulative(video, model, [] , search_idx, o_sm)
                if new_idx==-1: break

                used_idx_list += new_idx
                search_idx = [item for item in other_idx if item not in used_idx_list]
                new_idx_list.append(new_idx)

            d={fname: new_idx_list}
            f.write(json.dumps(d) + '\n')
            f.flush()

def dataset_multiple_SFS_cls(dataset, model_name, method, thr=0.1):
    out_path = CONF.SAVE_PATH
    data_path = os.path.join(out_path, method, f'top1_{dataset}_{model_name}.jsonl')
    write_path = os.path.join(out_path, method, f'multop1_{dataset}_{model_name}.jsonl')

    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    data = func.load_jsonl_to_dict(data_path)
    model = get_model.get_model(dataset, model_name)

    with open(write_path, 'a') as f:
        for path in tqdm(path_list):
            bn = os.path.basename(path)
            dn = os.path.basename(os.path.dirname(path))
            fname = dn + '\\' + bn
            d = data[fname]
            orig_sm = d['orig_sm']

            def get_rem_idx(d, orig_sm):
                sm = np.array(d['sm'])
                sm_norm = (sm-orig_sm)/orig_sm
                where = np.argwhere(sm_norm>-1*thr)
                if len(where)==0:
                    rem_f=[]
                else:
                    n = where.max()
                    rem_f = d['rem_f'][:n+1]
                # sel_idx = [f for f in d['start_frames'] if f not in rem_f]
                return rem_f

            sfs_list = []
            frames = d['start_frames']
            existing = get_rem_idx(d, orig_sm)
            video = model.get_video(path)
            analyze_cls = d['cls']
            sfs_list.append([f for f in frames if f not in existing])

            while len(existing)>=1:
                res = resume_SFS(video, model, existing, analyze_cls)
                existing_ = get_rem_idx(res, orig_sm)
                if len(existing_)==0:
                    break
                new_sfs = [f for f in existing if f not in existing_]
                sfs_list.append(new_sfs)
                existing=existing_
                
            # check the sfs validity. only for debugging
            for sfs in sfs_list:
                fvideo = func.fill_with_keep(sfs, video, 'past')
                pred = model.predict_video(fvideo)
                pred_sm = F.softmax(pred,dim=1)[0,d['cls']]
                assert (pred_sm - orig_sm)/orig_sm > -1*thr, f'{sfs} non consistant threshold'




            # new_idx = []
            # new_idx_list = []
            # used_idx_list = []
            # search_idx = other_idx
            # while True:
            #     new_idx = find_sfs_cummulative(video, model, [] , search_idx, o_sm)
            #     if new_idx==-1: break

            #     used_idx_list += new_idx
            #     search_idx = [item for item in other_idx if item not in used_idx_list]
            #     new_idx_list.append(new_idx)

            # d={fname: new_idx_list}
            # f.write(json.dumps(d) + '\n')
            # f.flush()

def get_preds(video, model):
    logits = model.predict_video(video)
    sm = F.softmax(logits, dim=1)
    f = model.get_features()
    return sm, f

'''
select: random- remove random frames, worst : remove the worst frames according to brute
'''
def distribution_shift_sample(dataset, model_name, forward = True, thr=1e-3, select = 'random'):
    out_path = CONF.OUT_PATH
    result_file = os.path.join(out_path, 'distrib.txt')
    out_file = os.path.join(out_path, 'brute')
    ward = 'forward' if forward else 'backward'
    data_path = os.path.join(out_file, f'curves_{dataset}_{model_name}_{ward}.jsonl')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    if select != 'random':
        data = func.load_jsonl_to_dict(data_path)
    model = get_model.get_model(dataset, model_name)
    N_ITR = 4
    methods = ['future', 'past', 'zero', 'mean', 'interp']
    big_metrics = {}
    for m in methods:
        big_metrics[m] = {'js': {i: [] for i in range(N_ITR)}, 'norm': {i: [] for i in range(N_ITR)}, 'cos': {i: [] for i in range(N_ITR)}}

    # for N in range(N_ITR):
    #     method_js = {}
    #     method_sim = {}
    #     method_cosin = {}
    #     for m in methods:
    #         method_js[m] = []
    #         method_sim[m] = []
    #         method_cosin[m] = []

    #     big_metrics[N] = {'js': method_js, 'norm': method_sim, 'cosine': method_cosin}

    for path in tqdm(path_list):
        video = model.get_video(path)
        L = video.size(2)
        fname = os.path.basename(path)
        if select != 'random':
            f_idx = data[fname]['idx']
        else:
            f_idx = list(range(video.size(2)))

        # original prediction
        sm_orig, f_orig = get_preds(video, model)

        # remove random frames
        all_idx = list(range(0,L))
        used_idx = []
        for n in range(N_ITR):
            if select == 'random':
                used_idx = random.sample(f_idx, n+1)
            elif select == 'worst':
                used_idx = f_idx[-(n+1):]
                    
            keep_idx = [i for i in all_idx if i not in used_idx]
            
            for method in methods:
                fvideo = func.fill_with_keep(keep_idx, video, method)
                sm, f = get_preds(fvideo, model)
                js_sm = func.jensen_shannon(sm, sm_orig)
                big_metrics[method]['js'][n].append(js_sm.item())
                f_norm = ((f**2).sum())**0.5
                f_orig_norm = ((f_orig**2).sum())**0.5
                similarity = F.cosine_similarity(f, f_orig, dim=0)
                big_metrics[method]['norm'][n].append((f_norm / f_orig_norm).item())
                big_metrics[method]['cos'][n].append(similarity.item())

    with open(result_file, 'a') as f:
        f.write(f'{dataset},  {model_name},  forward={forward}, thr = {thr},  select={select} \n')
        for n in range(N_ITR):
            f.write(f'N: {n+1} \n')
            for m in methods:
                ar_js = np.array(big_metrics[m]['js'][n])
                ar_sim = np.array(big_metrics[m]['norm'][n])
                ar_cosin = np.array(big_metrics[m]['cos'][n])
                str = f'{m:>6} **** JS: mean = {ar_js.mean():.8f} , std = {ar_js.std():.8f} NORM_ratio: mean = {ar_sim.mean():.8f} , std = {ar_sim.std():.8f} cosine: mean = {ar_cosin.mean():.8f} , std = {ar_cosin.std():.8f}\n'
                f.write(str)
        # write means
        f.write('Mean of means metrics : \n')
        for m in methods:
            f.write(f'{m}: ')
            for metric in ['js', 'norm', 'cos']:
                sum = 0
                for n in range(N_ITR):
                    sum += np.mean(big_metrics[m][metric][n])
                f.write(f'{metric} : {sum/N_ITR} ')
            f.write('\n')


def mmd_rbf(X, Y, gamma=1.0):
    XX = torch.cdist(X, X) ** 2
    YY = torch.cdist(Y, Y) ** 2
    XY = torch.cdist(X, Y) ** 2

    Kxx = torch.exp(-gamma * XX)
    Kyy = torch.exp(-gamma * YY)
    Kxy = torch.exp(-gamma * XY)

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

def distribution_mmd(dataset, model_name, forward = True, thr=1e-3, select = 'random', N_ITR=1):
    out_path = CONF.OUT_PATH
    out_file = os.path.join(out_path, 'brute')
    ward = 'forward' if forward else 'backward'
    data_path = os.path.join(out_file, f'curves_{dataset}_{model_name}_{ward}.jsonl')
    path_list, cls_list, idx_list = data_paths.get_paths(dataset)
    if select != 'random':
        data = func.load_jsonl_to_dict(data_path)
    model = get_model.get_model(dataset, model_name)

    N_ITR = N_ITR
    methods = ['future', 'past', 'zero', 'mean', 'interp']
    feature_dict = {}
    for m in methods:
        feature_dict[m] = [torch.empty(0), torch.empty(0)]

    for path in tqdm(path_list):
        video = model.get_video(path)
        L = video.size(2)
        fname = os.path.basename(path)
        if select != 'random':
            f_idx = data[fname]['idx']
        else:
            f_idx = list(range(video.size(2)))

        # original prediction
        sm_orig, f_orig = get_preds(video, model)

        # remove random frames
        all_idx = list(range(0,L))
        used_idx = []
        for n in range(N_ITR):
            if select == 'random':
                used_idx = random.sample(f_idx, n+1)
            elif select == 'worst':
                used_idx = f_idx[-(n+1):]
                    
            keep_idx = [i for i in all_idx if i not in used_idx]
            
            for method in methods:
                fvideo = func.fill_with_keep(keep_idx, video, method)
                sm, f = get_preds(fvideo, model)
                feature_dict[method][0] = torch.concatenate([feature_dict[method][0], f_orig[None,:].cpu()], dim=0)
                feature_dict[method][1] = torch.concatenate([feature_dict[method][1], f[None,:].cpu()], dim=0)

    for method in methods:
        X = feature_dict[method][0]
        Y = feature_dict[method][1]
        Z = torch.cat([X, Y], dim=0)
        dists = torch.cdist(Z, Z)
        sigma2 = torch.median(dists**2)
        gamma = 1 / (2 * sigma2)

        for g in [0.01, 0.1, 1, 10, 100, gamma.item()]:
            mmd = mmd_rbf(X, Y, gamma=g).item()
            print(f'method: {method} Gamma: {g} MMD: {mmd}')
        print('')





if __name__ == "__main__":
    dataset_multiple_SFS_cls('ucf101', 'mc3-18', 'sfs', thr=0.1)
    # dataset_curves('ssv2', 'tformer_base', 'facility', forward=False)
    # distribution_shift('ucf101', 'mc3-18', forward = False, select='random')
    # distribution_mmd('ucf101', 'mc3-18', forward = True, select='random')
    # dataset_curves_cls('ucf101', 'mc3-18', forward = False)
    # dataset_curves_cls('ssv2', 'vjepa2', 'ig')