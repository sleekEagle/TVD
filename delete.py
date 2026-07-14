import func
import h5py



k = func.get_h5_keys(r"D:\output\TVD\greedy\curves_ucf101_r3d-18.h5")[0]
with h5py.File(r"D:\output\TVD\greedy\curves_ucf101_r3d-18.h5", "r") as f:
    g = f[k]
    print(g['js_ar_f'][:])

with h5py.File(r"D:\output\TVD - Copy\greedy\curves_ucf101_r3d-18.h5", "r") as f:
    g = f[k]
    print(g['js_ar'][:])