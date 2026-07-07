import os
import random
import numpy as np

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

def past_fill(keep, l=16):
    tofill = [i for i in range(l) if i not in keep]
    
    fillwith = []
    keep = np.array(keep)
    for idx in tofill:
        # get immediate past item
        ar = np.sort(keep[keep<idx])
        if len(ar) > 0:
            k = int(ar[-1])
        else:
            ar = np.sort(keep[keep>idx])
            assert len(ar) > 0, 'no items found to fill'
            k = int(ar[0])
        fillwith.append(k)

    return tofill, fillwith

def future_fill(keep, l=16):
    tofill = [i for i in range(l) if i not in keep]
    
    fillwith = []
    keep = np.array(keep)
    for idx in tofill:
        # get immediate past item
        ar = np.sort(keep[keep>idx])
        if len(ar) > 0:
            k = int(ar[0])
        else:
            ar = np.sort(keep[keep<idx])
            assert len(ar) > 0, 'no items found to fill'
            k = int(ar[-1])
        fillwith.append(k)

    return tofill, fillwith

#in-place fill video [1, 3, 16, 112, 112]
def fill_video(tofill, fillwith, video):
    tofill_t = torch.tensor(tofill)
    fillwith_t = torch.tensor(fillwith)
    video[:, :, tofill_t] = video[:, :, fillwith_t].clone()


#****************************************************************************************************************
#****************************************************************************************************************

if __name__ == "__main__":
    tofill, fillwith = future_fill([0,4,7,8,13,15])
    import torch
    video = torch.rand([1,3,16,112,112])
    fvideo = video.clone()
    fill_video(tofill, fillwith, fvideo)