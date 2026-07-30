
import math
import pprint
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import models.vision_transformer as vit
from models.attentive_pooler import AttentiveClassifier
# from evals.video_classification_frozen.utils import make_transforms
# from src.datasets.data_manager import init_data
import torch
import torch.nn as nn
from torch.serialization import MAP_LOCATION
import time
import random
from typing import Any
import yaml
import os

def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class ClipAggregation(nn.Module):
    """
    Process each clip independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=128,
        use_pos_embed=False,
        out_layers=None,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads

        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        outputs = self.model(x)
        outputs = torch.cat(outputs, dim=1)

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size  # num temporal indices
            S = N // T  # num spatial tokens

            # Unroll outputs into a 2D array [spatial_views x temporal_views]
            eff_B = B * num_views_per_clip
            all_outputs = [[] for _ in range(num_views_per_clip)]
            for i in range(num_clips):
                o = outputs[i * eff_B : (i + 1) * eff_B]
                for j in range(num_views_per_clip):
                    all_outputs[j].append(o[j * B : (j + 1) * B])

            for i, outputs in enumerate(all_outputs):
                # Concatenate along temporal dimension
                outputs = [o.reshape(B, T, S, D) for o in outputs]
                outputs = torch.cat(outputs, dim=1).flatten(1, 2)
                # Compute positional embedding
                if (self.pos_embed is not None) and (clip_indices is not None):
                    _indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                    pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, max_T, D]
                    pos_embed = apply_masks(pos_embed, _indices, concat=False)  # list(Tensor([B, T, D]))
                    pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                    pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)  # [B, T*num_clips, S, D]
                    pos_embed = pos_embed.flatten(1, 2)
                    outputs += pos_embed
                all_outputs[i] = outputs

            return all_outputs

        return multiviews_postprocess(outputs)

def robust_checkpoint_loader(r_path: str, map_location: MAP_LOCATION = "cpu", max_retries: int = 3) -> Any:
    """
    Loads a checkpoint from a path, retrying up to max_retries times if the checkpoint is not found.
    """
    retries = 0

    while retries < max_retries:
        try:
            return torch.load(r_path, map_location=map_location)
        except Exception as e:
            retries += 1
            if retries < max_retries:
                sleep_time_s = (2**retries) * random.uniform(1.0, 1.1)
                time.sleep(sleep_time_s)
                continue
            else:
                raise e
            
def load_checkpoint(device, r_path, classifiers):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    # -- loading encoder
    pretrained_dict = checkpoint["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}

    # msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]
    
    msg = classifiers.load_state_dict(pretrained_dict)
    print(msg)

    return classifiers


def init_encoder(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    checkpoint = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    out_layers = wrapper_kwargs.get("out_layers")

    model = vit.__dict__[enc_model_name](
        img_size=resolution, num_frames=frames_per_clip, out_layers=out_layers, **enc_kwargs
    )

    pretrained_dict = checkpoint[enc_ckp_key]
    # --
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in model.state_dict().items():
        if k not in pretrained_dict:
            print(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            print(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(msg)

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        **wrapper_kwargs,
    )
    del checkpoint
    return model


class VJEPA2(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(script_dir,'configs', 'diving48.yaml')
        params = None
        with open(fname, "r") as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)

        args_exp = params.get("experiment")
        args_data = args_exp.get("data")
        args_pretrain = params.get("model_kwargs")
        args_classifier = args_exp.get("classifier")


        frames_per_clip = args_data.get("frames_per_clip", 16)
        resolution = args_data.get("resolution", 224)
        num_classes = args_data.get("num_classes")

        checkpoint = args_pretrain.get("checkpoint")
        args_model = args_pretrain.get("pretrain_kwargs")
        args_wrapper = args_pretrain.get("wrapper_kwargs")

        eval_tag = params.get("tag", None)

        num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
        num_heads = args_classifier.get("num_heads", 16)

        pretrain_folder = params.get("folder", None)
        
        folder = os.path.join(pretrain_folder, "video_classification_frozen/")
        if eval_tag is not None:
            folder = os.path.join(folder, eval_tag)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        latest_path = os.path.join(folder, "latest.pt")

        # -- init models
        model = (init_encoder(
                frames_per_clip=frames_per_clip,
                resolution=resolution,
                checkpoint=checkpoint,
                model_kwargs=args_model,
                wrapper_kwargs=args_wrapper,
            )
            .to(device)
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        encoder = model
        classifier = AttentiveClassifier(
                embed_dim=encoder.embed_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes,
                use_activation_checkpointing=True,
            ).to(device)
        if os.path.exists(latest_path):
            classifier = load_checkpoint(
                device=device,
                r_path=latest_path,
                classifiers=classifier
            )

        pass
        
        self.features = {}
        self.handle = self.model.pooler.self_attention_layers[2].mlp.fc2.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features['features'] = output.mean(dim=1).detach()
        

    def get_features(self):
        return self.features['features'].squeeze()
    
    def remove_hook(self):
        self.handle.remove()

    #input shape of x: 1,3,16,224,224
    def forward(self, x):
        pass

    def sample_frames(self, video_path, num_frames=16):
        pass

    def get_video(self, path):
        pass
    
    def predict_video(self, video):
        pass


if __name__ == "__main__":
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    model = VJEPA2()



    # Initialize model



