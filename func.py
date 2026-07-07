import os
import random

def get_pred(model, path):
    fname = os.path.basename(path)
    pred = model.predict_video(path).squeeze()   
    feat = model.get_features()  
    return fname, pred, feat

#****************************************************************************************************************
#****************************************************************************************************************
# *****************  Temporal Freezing Code *********************************************************************
#****************************************************************************************************************
#****************************************************************************************************************
def deep_copy_dict(d):
    new_dict = {}
    for k in d:
        if isinstance(d[k], dict):
            new_dict[k] = deep_copy_dict(d[k])
        elif isinstance(d[k], list):
            new_dict[k] = d[k].copy()
        else:
            new_dict[k] = d[k]
    return new_dict

"""
break the list into two sub lists. 
Divide into equal length lists if the length is even.
If the length is odd, randommly choose the dividing point from the two possible options
"""
def break_list_middle(items):
    n = len(items)
    
    if n == 0:
        return None  # or raise ValueError("Cannot select from empty list")
    
    if n % 2 == 0:
        pass

    if n % 2 == 1:
        r = random.choice([0,1])
        mid = n // 2 + r
    else:
        mid = n //2

    past = items[:mid]
    future = items[mid:]

    return past, future

def break_list_random(items):
    n = len(items)
    idx = random.randint(0, n) 
    past = items[:idx]
    future = items[idx:]
    return past, future

'''
    modifies the group dict inplace
    use:     key_to_fill = 1
    future_fill(key_to_fill, mask, groups)
'''
def _future_fill(fill_key, mask, groups):
    # #create a deep copy of the dict
    # new_groups = {}
    # for k in groups:
    #     new_groups[k] = groups[k].copy()
    ord_keys = sorted(mask.keys())
    l = [k for k in ord_keys if (k>fill_key and mask[k])]
    if len(l)==0:
        return -1
    first_true_key = l[0]
    groups[first_true_key].extend(list(set(groups[fill_key]+[fill_key])))
    groups.pop(fill_key)

def _past_fill(fill_key, mask, groups):
    ord_keys = sorted(mask.keys(),reverse=True)
    l = [k for k in ord_keys if (k<fill_key and mask[k])]
    if len(l)==0:
        return -1
    first_true_key = l[0]
    groups[first_true_key].extend(list(set(groups[fill_key]+[fill_key])))
    groups.pop(fill_key)

def _hybrid_mid_fill(fill_key, mask, groups, method):
    ord_keys_rev = sorted(mask.keys(),reverse=True)
    past_grp = [ok for ok in ord_keys_rev if (ok<fill_key and mask[ok])]
    ord_keys = sorted(mask.keys(),reverse=False)
    future_grp = [ok for ok in ord_keys if (ok>fill_key and mask[ok])]

    if len(past_grp)>0 and len(future_grp)>0:
        frames = groups[fill_key] + [fill_key]
        frames.sort()
        if method == 'middle':
            to_past, to_future = break_list_middle(frames)
        elif method == 'random':
            to_past, to_future = break_list_random(frames)

        groups[past_grp[0]].extend(to_past)
        groups[future_grp[0]].extend(to_future)
        groups.pop(fill_key)

    elif len(past_grp)>0:
        _past_fill(fill_key, mask, groups)
    elif len(future_grp)>0:
        _future_fill(fill_key, mask, groups)
    else:
        return -1


'''
temporally freeze a given groups. 
the groups where mask==false will be filled from another group from the future which is True.
If there are no such frames, it is filled from the past. If there are no such groups either, groups is not modified.
mask: e.g: [0,0,1] : keep the last group. remove the rest
'''
def future_fill_all(mask, groups):
    ord_keys = sorted(list(groups.keys()))
    m = {}
    for i,k in enumerate(ord_keys):
        m[int(k)] = bool(mask[i])

    groups = deep_copy_dict(groups)
    ord_keys = sorted(m.keys())
    for k in ord_keys[:-1]:
        if not m[k]:
            ret = _future_fill(k, m, groups)
            if ret ==-1:
                _past_fill(k, m, groups)
    if not m[ord_keys[-1]]:
        _past_fill(ord_keys[-1], m, groups)
    return groups

'''
temporally freeze a given groups. 
the groups where mask==false will be filled from another group from the past which is True.
If there are no such frames, it is filled from the future. If there are no such groups either, groups is not modified. 

mask: e.g: [0,0,1] : keep the last group. remove the rest
'''
def past_fill_all(mask, groups):
    # groups_ = {1:[0,2], 5:[3,4,6,7], 8:[], 10:[9,11,12], 14:[13,15]}
    # mask = [False, False, True, True, True]

    ord_keys = sorted(list(groups.keys()))
    m = {}
    for i,k in enumerate(ord_keys):
        m[int(k)] = bool(mask[i])

    groups = deep_copy_dict(groups)
    if not m[ord_keys[0]]:
        _future_fill(ord_keys[0], m, groups)
    for k in ord_keys[1:]:
        if not m[k]:
            ret = _past_fill(k, m, groups)
            if ret ==-1:
                _future_fill(k, m, groups)

    return groups


def hybrid_fill_all(mask, groups, method):
    # mask = [False,True,False,True,False]
    # groups = {1:[0,2], 4:[3,5,6], 7:[], 10:[8,9,11,12],13:[14,15]}

    ord_keys = sorted(list(groups.keys()))
    m = {}
    for i,k in enumerate(ord_keys):
        m[int(k)] = bool(mask[i])
    groups = deep_copy_dict(groups)

    # handle edge cases
    if not m[ord_keys[0]]:
        _future_fill(ord_keys[0], m, groups)
    if not m[ord_keys[-1]]:
        _past_fill(ord_keys[-1], m, groups)

    for k in ord_keys[1:-1]:
        if not m[k]:
            _hybrid_mid_fill(k, m, groups, method)
    
    return groups

#****************************************************************************************************************
#****************************************************************************************************************