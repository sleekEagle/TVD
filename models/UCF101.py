import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
import torch.nn as nn
from transformers import AutoModelForVideoClassification, AutoVideoProcessor
from huggingface_hub import hf_hub_download
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2

device = "cuda" if torch.cuda.is_available() else "cpu"

class HF_MODEL(nn.Module):
    def __init__(self):
        super().__init__()

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # hf_repo = "facebook/vjepa2-vitl-fpc16-256-ssv2"
        # self.model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(self.device)
        # self.processor = AutoVideoProcessor.from_pretrained(hf_repo)

        # id2label = self.model.config.id2label
        # self.label2id = {}
        # for k in id2label:
        #     new_k = id2label[k].replace('[','').replace(']','').replace('\'','')
        #     self.label2id[new_k] = k


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
    
class MC3_18(HF_MODEL):
    def __init__(self):
        super().__init__()
        from torchvision.models.video import mc3_18
        model = mc3_18(pretrained=False, num_classes=101)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = hf_hub_download(repo_id="dronefreak/mc3-18-ucf101", filename="mc318-ufc101-split-1.pth")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        #rename dict
        d = {}
        for k in checkpoint['model_state_dict']:
            new_k = '.'.join(k.split('.')[1:])
            d[new_k] = checkpoint['model_state_dict'][k]
        model.load_state_dict(d)

        self.model = model
        self.model.eval()

        self.transform = v2.Compose([
            v2.Resize((128, 171)),
            v2.CenterCrop(112),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                    std=[0.22803, 0.22145, 0.216989])
        ])

    def predict_from_path(self, path):
        decoder = VideoDecoder(path)
        indices = torch.linspace(0, decoder.metadata.num_frames - 1, 16).long().tolist() 
        sampled_frames = decoder.get_frames_at(indices=indices) 
        if hasattr(sampled_frames, 'data'):
            frame_tensors = sampled_frames.data  # Shape: (N, C, H, W) or (N, H, W, C)
        else:
            # Option B: Convert directly - torchcodec returns frames as tensor list
            frame_tensors = torch.stack([sampled_frames[i] for i in range(len(sampled_frames))])
        
        self.transform(frame_tensors[0])

        processed_frames = torch.stack([self.transform(frame) for frame in frame_tensors])



        inputs = self.video_from_path(path)
        with torch.no_grad():
            outputs = self.model(**inputs)
        pred_cls = outputs.logits.argmax(-1).item()
        return pred_cls 
