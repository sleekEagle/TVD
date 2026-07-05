import models.ssv2 as ssv2
import models.UCF101 as ucf101

# UCF101 model from https://github.com/sonho4ng/Human-Action-Recognition-UCF101/tree/main
def get_model(dataset_name, model_name):
    if dataset_name == 'SSV2' and model_name == 'VJEPA2':
        model = ssv2.VJEPA2()
    elif dataset_name == 'UCF101' and model_name == 'mc3-18':
        model = ucf101.MC3_18()
    return model