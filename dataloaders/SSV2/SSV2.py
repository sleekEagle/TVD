from pathlib import Path
import os
import CONF
SSV2_PATH = CONF.SSV2_PATH

def get_ssv2_paths():
    path = Path(SSV2_PATH)
    dirs = [p.name for p in path.iterdir() if p.is_dir()]
    dirs = [p for p in path.iterdir() if p.is_dir()]
    n_files = len([p for p in path.rglob("*") if p.is_file()])

    d_names = []
    paths = []
    for dir in dirs:
        d_name = dir.name
        files = [p for p in dir.iterdir() if p.is_file()]
        d_names.extend([d_name]*len(files))
        paths.extend(files)

    return d_names, paths


def get_paths():
    import json
    #read class indices 
    with open('dataloaders/SSV2/label2id.json', 'r') as f:
        label2id = json.load(f)

    path = Path(SSV2_PATH)
    cls_list, path_list , idx_list = [], [], []
    with open('dataloaders/SSV2/ssv2_paths.txt', 'r') as f:
        for line in f:
            full = Path(line.strip()).as_posix()
            cls = str(Path(full).parent)
            full = os.path.join(path, full)
            cls_list.append(cls)
            path_list.append(full)
            idx_list.append(label2id[cls])
    return path_list, cls_list, idx_list
