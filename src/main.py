import numpy as np
import pickle
import scipy
from pathlib import Path
import run_DL
import run_LR

relative_path = "..\\dataset\\train_feature.pkl"
dir = Path(__file__).parent
absolute_path = dir.joinpath(relative_path)

TrainFeature = pickle.load(open(absolute_path, "rb"))

# print(np.eye(10)@TrainFeature[0:10,:])
# print(TrainFeature[0:1,:].T@np.eye(1))

test_x = np.random.rand(1000,2)
test_y = ((test_x[:,0] > test_x[:,1]) * 1).reshape(1000,1)
# print(test_x)
# print(test_y.reshape(10,1))

# run_DL.run_py(test_x, test_y)
run_LR.run_lr(test_x, test_y)