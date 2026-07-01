
from UCF101_models.resnet_lstm import ResNetLSTM
from UCF101_models.ResidualSE import ResidualSE
from UCF101_models.tsm import models
from UCF101_models.i3d_shufflenet import EnhancedI3DShuffleNet
from UCF101_models.enhanced_r3d import R3DModel
from UCF101_models.r21d import R2Plus1DClassifier
from UCF101_models.resnet_fnn import ResNetFNN
import torch

from CONF import ucf_classes as classes

def choose_model(model_name: str, n_frames: int) -> object:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'resnet-lstm':
        model = ResNetLSTM(num_classes=len(classes)).to(device)
    elif model_name == 'resnet-fnn':
        model = ResNetFNN(num_classes=len(classes)).to(device)
    elif model_name == 'residualSE':
        model = ResidualSE(num_classes=len(classes)).to(device)
    elif model_name == 'tsm':
        model = models.TSN(num_classes=len(classes), num_segments=n_frames, modality="RGB",
                   base_model='resnet50', before_softmax=False,
                   is_shift=True, shift_place='blockres').to(device)
    elif model_name == 'i3d':
        model = EnhancedI3DShuffleNet(len(classes))
    elif model_name == 'enhanced_r3d':
        model = R3DModel(num_classes=len(classes), pretrained='update_r3d', dropout_prob=0.5)
    elif model_name == 'r21d':
        model = R2Plus1DClassifier(num_classes=len(classes), layer_sizes=[2,2,2,2]).to(device)
    return model