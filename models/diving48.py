import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
import torch.nn as nn
from transformers import AutoModelForVideoClassification, AutoVideoProcessor
from huggingface_hub import hf_hub_download
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import CONF
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# Disable all progress bars
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# https://huggingface.co/facebook/vjepa2-vitl-fpc32-256-diving48
# config (look for stride): https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/inference/vitl/diving48.yaml
class VJEPA2(nn.Module):
    def __init__(self):
        super().__init__()
        hf_repo = "facebook/vjepa2-vitl-fpc32-256-diving48"
        self.model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(device)
        self.processor = AutoVideoProcessor.from_pretrained(hf_repo)
        self.num_frames=self.model.config.frames_per_clip
        # data_path = CONF.DIVING_PATH
        self.features = {}
        self.handle = self.model.pooler.self_attention_layers[2].mlp.fc2.register_forward_hook(self.hook_fn)

        # test_path = os.path.join(data_path, "Diving48_V2_test.json")
        # data = json.load(open(test_path))
        # data_dict = {}
        # for d in data:
        #     fname = d['vid_name']
        #     data_dict[fname] = {'label':d['label'], 'start_frame':d['start_frame'], 'end_frame':d['end_frame']}
        # self.data_dict = data_dict

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
