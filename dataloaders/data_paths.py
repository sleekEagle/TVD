
def get_paths(dataset):
    if dataset == 'ssv2':
        import dataloaders.SSV2.SSV2 as SSV2
        path_list, cls_list, idx_list = SSV2.get_paths()
    elif dataset == 'ucf101':
        import dataloaders.UCF101 as UCF101
        path_list, cls_list, idx_list = UCF101.get_paths()
    elif dataset == 'diving48':
        import dataloaders.diving48 as diving48
        path_list, cls_list, idx_list = diving48.get_paths()


    return path_list, cls_list, idx_list
        


