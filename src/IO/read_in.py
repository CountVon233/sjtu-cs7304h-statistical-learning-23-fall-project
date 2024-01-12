import numpy as np
import pickle
from pathlib import Path

def read_file(relative_path):
    dir = Path(__file__).parent
    absolute_path = dir.joinpath(relative_path)
    Data = pickle.load(open(absolute_path, "rb"))
    return Data
