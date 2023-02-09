from sklearn.utils import resample
import numpy as np

def normalization(signal):
    #normalize_signal = (signal - np.mean(signal)) / (np.std(np.abs(signal)))
    normalize_signal = (signal - np.mean(signal)) / (np.std(signal))
    return normalize_signal

