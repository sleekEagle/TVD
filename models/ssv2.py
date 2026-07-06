import torch
import numpy as np
from torchcodec.decoders import VideoDecoder
import torch.nn as nn
from transformers import AutoModelForVideoClassification, AutoVideoProcessor
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import cv2
import numpy as np

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
    
    def predict_video(self, path):
        inputs = self.video_from_path(path)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits 
    
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
    
#tformer from https://github.com/mit-han-lab/temporal-shift-module
class TFORMER_b(nn.Module):
    def __init__(self, num_frames=8):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_frames = num_frames

        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2")
        self.model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-ssv2")

    def _load_video(self, video_path):
        """Load video and extract frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")

        return frames
    
    def _sample_frames(self, frames):
        """Sample num_frames uniformly from video."""
        total_frames = len(frames)

        if total_frames < self.num_frames:
            # Repeat frames if video too short
            indices = np.arange(total_frames)
            indices = np.tile(indices, int(np.ceil(self.num_frames / total_frames)))
            indices = indices[: self.num_frames]
        else:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

        return [frames[i] for i in indices]
    
    def predict_video(self, video_path):
        # Load and sample frames
        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)
        inputs = self.processor(images=frames, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits
    

class TFORMER_hr(TFORMER_b):
    def __init__(self, num_frames=16):
        super().__init__()
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_frames = num_frames

        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-hr-finetuned-ssv2")
        self.model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-hr-finetuned-ssv2")