import numpy as np

def save_scaler(scaler, filepath):
    np.save(filepath, [scaler.mean_, scaler.scale_])

def load_scaler(filepath):
    mean_, scale_ = np.load(filepath, allow_pickle=True)
    return mean_, scale_
