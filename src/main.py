import numpy as np
import pickle
import scipy
from pathlib import Path

relative_path = "..\\..\\dataset\\train_feature.pkl"
dir = Path(__file__).parent
absolute_path = dir.joinpath(relative_path)

TrainFeature = pickle.load(open(absolute_path, "rb"))
for i in TrainFeature:
    print(i)
    break