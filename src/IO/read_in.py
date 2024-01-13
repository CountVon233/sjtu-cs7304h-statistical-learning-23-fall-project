import numpy as np
import pickle
from pathlib import Path

def read_file_pkl(relative_path):
    dir = Path(__file__).parent
    absolute_path = dir.joinpath(relative_path)
    Data = pickle.load(open(absolute_path, "rb"))
    return Data

def read_file_npy(relative_path):
    dir = Path(__file__).parent
    absolute_path = dir.joinpath(relative_path)
    Data = np.load(open(absolute_path, "rb"))
    return Data
