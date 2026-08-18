import models.DINOv2 as dino
from models import get_model
from dataloaders import data_paths

dataset = 'ucf101'
cls_model = get_model.get_model(dataset, 'mc3-18')
path_list, cls_list, idx_list = data_paths.get_paths(dataset)
video = cls_model.get_video(path_list[0])

emb_model = dino.DINOv2Embedder()
emb = emb_model.extract_embeddings_batch(video)

pass