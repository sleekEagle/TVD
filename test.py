from dataloaders import data_paths
path_list, cls_list, idx_list = data_paths.get_paths('ssv2')

from models import get_model
model = get_model.get_model('UCF101','mc3-18')


model.predict_from_path(path_list[0])

pass
