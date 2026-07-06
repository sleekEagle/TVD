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

device = "cuda" if torch.cuda.is_available() else "cpu"

# Disable all progress bars
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# https://huggingface.co/facebook/vjepa2-vitg-fpc32-384-diving48

class VJEPA2(nn.Module):
    def __init__(self):
        super().__init__()
        hf_repo = "facebook/vjepa2-vitg-fpc32-384-diving48"
        self.model = AutoModelForVideoClassification.from_pretrained(hf_repo).to(device)
        self.processor = AutoVideoProcessor.from_pretrained(hf_repo)
        self.num_frames=self.model.config.frames_per_clip

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
        
    
    # Transform frames
    def _process_frames(self, frames):
        inputs = self.processor(
            list(frames),  # Each frame as a numpy array
            return_tensors="pt"
        )
        return inputs
    
    def predict_video(self, video_path):

        # Load and sample frames
        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)
        inputs = self._process_frames(frames)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits
