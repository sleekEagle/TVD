import torch
import numpy as np
import torch.nn as nn
from models.diving48_vjp import model

device = "cuda" if torch.cuda.is_available() else "cpu"

# https://huggingface.co/facebook/vjepa2-vitl-fpc32-256-diving48
# config (look for stride): https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/inference/vitl/diving48.yaml
class VJEPA2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model.VJEPA2()
        pass

    def hook_fn(self, module, input, output):
        self.features['features'] = output.mean(dim=1).detach()
        

    def get_features(self):
        return self.features['features'].squeeze()
    
    def remove_hook(self):
        self.handle.remove()
        
    def get_video(self, path):
        # fname = os.path.basename(path).split('.')[0]
        # data = self.data_dict[fname]
        # data_nf = data['end_frame'] - data['start_frame'] + 1

        vr = VideoDecoder(path)
        total_frames = len(vr)

        # assert data_nf == total_frames, 'n frames not consistant' # this does not get triggered

        required_frames = self.model.config.frames_per_clip
        
        # Sample available frames (use stride of 2 as in your example)
        stride = 2
        frame_idx = np.arange(0, total_frames, stride)
        
        # If we have enough frames, sample normally
        if len(frame_idx) >= required_frames:
            # Take first 'required_frames' frames
            frame_idx = frame_idx[:required_frames]
        else:
            # Pad by repeating the last frame
            frame_idx = list(frame_idx)
            while len(frame_idx) < required_frames:
                frame_idx.append(frame_idx[-1])
        
        video = vr.get_frames_at(indices=np.array(frame_idx)).data
        inputs = self.processor(video, return_tensors="pt").to(self.model.device)
        return inputs['pixel_values_videos'].permute(0,2,1,3,4) # [1,3,32,224,224]
    
    
    def predict_video(self, video):
        with torch.no_grad():
            outputs = self.model(video.permute(0,2,1,3,4))
        return outputs.logits
