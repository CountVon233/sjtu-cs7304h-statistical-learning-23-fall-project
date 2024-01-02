import numpy as np
import pickle
from pathlib import Path

def read_file(relative_train_feature_path, relative_train_label_path, relative_test_feature_path):
    dir = Path(__file__).parent
    absolute_train_feature_path = dir.joinpath(relative_train_feature_path)
    absolute_train_label_path = dir.joinpath(relative_train_label_path)
    absolute_test_feature_path = dir.joinpath(relative_test_feature_path)
    TrainFeature = pickle.load(open(absolute_train_feature_path, "rb"))
    TrainLabel = np.load(open(absolute_train_label_path, "rb"))
    TestFeature = pickle.load(open(absolute_test_feature_path, "rb"))
    return TrainFeature, TrainLabel, TestFeature
