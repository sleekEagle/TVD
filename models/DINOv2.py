import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from typing import Union, List
import os


class DINOv2Embedder:
    def __init__(self, model_name="facebook/dinov2-base"):
        """Initialize DINOv2 model and processor.
        
        Available model sizes:
        - facebook/dinov2-small
        - facebook/dinov2-base  (recommended)
        - facebook/dinov2-large
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_embeddings_batch(self, image_t, batch_size=32):
        image_t = image_t.squeeze().permute(1,0,2,3)

        embeddings = []
        for i in range(0, len(image_t), batch_size):
            batch = image_t[i:i+batch_size]
            #to 0 - 1
            batch = (batch-batch.min())/(batch.max()-batch.min())
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeds = outputs.last_hidden_state[:, 0, :]
                embeddings.append(batch_embeds.cpu().numpy())
                
        return np.vstack(embeddings)