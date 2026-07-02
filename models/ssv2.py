import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
import torch.nn as nn
from transformers import AutoModelForVideoClassification, AutoVideoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

class VJEPA2(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
        self.model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(self.device)
        self.processor = AutoVideoProcessor.from_pretrained(hf_repo)

        id2label = self.model.config.id2label
        self.label2id = {}
        for k in id2label:
            new_k = id2label[k].replace('[','').replace(']','').replace('\'','')
            self.label2id[new_k] = k


    #input shape of x: 1,3,16,224,224
    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        output = self.model(x)
        logits = output.logits
        return logits

    def sample_frames(self, video_path, num_frames=16):
        decoder = VideoDecoder(video_path)
        vid_len = len(decoder)
        frame_idx = np.linspace(0, vid_len - 1, self.model.config.frames_per_clip, dtype=int)
        video = decoder.get_frames_at(indices=frame_idx).data
        return video 

    def video_from_path(self, path):
        frames = self.sample_frames(str(path), num_frames=16)
        inputs = self.processor(frames, return_tensors="pt").to(self.device)
        return inputs
    
    def predict_from_path(self, path):
        inputs = self.video_from_path(path)
        with torch.no_grad():
            outputs = self.model(**inputs)
        pred_cls = outputs.logits.argmax(-1).item()
        return pred_cls 
    
    def predict_from_batch_path(self, paths):
        bs = 32
        paths = [paths[i:i + bs] for i in range(0, len(paths), bs)]

        pred_t = torch.empty(0)
        for i, pbatch in enumerate(paths):
            print(f'Processing batch {(i+1)/len(paths):.2%} %', end='\r')
            vid_list = []
            for p in pbatch:
                frames = self.sample_frames(str(p), num_frames=16)
                vid_list.append(frames)
            inputs = self.processor(vid_list, return_tensors="pt").to(device)
            with torch.no_grad():
                p = self.model(**inputs)
            pred_t = torch.cat((pred_t, p.logits.cpu()), dim=0)

        return pred_t