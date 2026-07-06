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



class HF_MODEL(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.processor = None
    
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
    
    # Transform frames
    def _process_frames(self, frames):
        transformed_frames = []
        for frame in frames:
            frame_pil = Image.fromarray(frame)
            frame_tensor = self.processor(frame_pil)
            transformed_frames.append(frame_tensor)
        return transformed_frames
    
    def predict_video(self, video_path):
        """Predict action from video.

        Args:
            video_path: Path to video file
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions
        """
        # Load and sample frames
        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)
        transformed_frames = self._process_frames(frames)

        # Stack: (T, C, H, W) → (C, T, H, W)
        video_tensor = torch.stack(transformed_frames).permute(1, 0, 2, 3)
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dim: (1, C, T, H, W)
        video_tensor = video_tensor.to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(video_tensor)

        return outputs

'''
**********************************************************
**********************************************************
Drone Freak Models
**********************************************************
**********************************************************
'''

# acc: 0.85223367697594595
class MC3_18(HF_MODEL):
    def __init__(self,num_frames=16, spatial_size=112):
        super().__init__()
        self.num_frames=num_frames
        self.spatial_size=spatial_size
        # Transform
        self.processor = transforms.Compose(
            [
                transforms.Resize((128, 171)),
                transforms.CenterCrop(spatial_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
                ),
            ]
        )

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
        self.model.to(self.device)


# acc: 81.52260111022997
class R3D_18(HF_MODEL):
    def __init__(self, num_frames=16, spatial_size=112):
        super().__init__()

        self.num_frames=num_frames
        self.spatial_size=spatial_size
        # Transform
        self.processor = transforms.Compose(
            [
                transforms.Resize((128, 171)),
                transforms.CenterCrop(spatial_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
                ),
            ]
        )

        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=False, num_classes=101)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = hf_hub_download(repo_id="dronefreak/r3d-18-ucf101", filename="r3d18-ufc101-split-1.pth")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        #rename dict
        d = {}
        for k in checkpoint['model_state_dict']:
            new_k = '.'.join(k.split('.')[1:])
            d[new_k] = checkpoint['model_state_dict'][k]
        model.load_state_dict(d)

        self.model = model
        self.model.eval()
        self.model.to(self.device)


'''
**********************************************************
**********************************************************
Other HF Models
**********************************************************
**********************************************************
'''

# Disable all progress bars
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# https://huggingface.co/nateraw/videomae-base-finetuned-ucf101
class MAE_B(HF_MODEL):
    def __init__(self, num_frames=16):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("nateraw/videomae-base-finetuned-ucf101")
        self.model = AutoModelForVideoClassification.from_pretrained("nateraw/videomae-base-finetuned-ucf101")
        self.num_frames=num_frames

    # Transform frames
    def _process_frames(self, frames):
        inputs = self.processor(
            list(frames),  # Each frame as a numpy array
            return_tensors="pt"
        )
        return inputs
    
    def predict_video(self, video_path):
        """Predict action from video.

        Args:
            video_path: Path to video file
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions
        """
        # Load and sample frames
        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)
        inputs = self._process_frames(frames)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits

    
        