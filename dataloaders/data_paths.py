
def get_paths(dataset):
    if dataset == 'ssv2':
        import dataloaders.SSV2.SSV2 as SSV2
        paths = SSV2.get_paths()
    elif dataset == 'ucf101':
        import dataloaders.UCF101 as UCF101
        paths = UCF101.get_paths()

    return paths
        


