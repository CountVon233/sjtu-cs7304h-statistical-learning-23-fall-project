import numpy as np
import pickle
import scipy
from pathlib import Path
from deep_learning_model import DL

def run_py(train_data, train_label):
    train_x = train_data[0:800,:]
    train_y = train_label[0:800,:]
    test_x = train_data[800:train_data.shape[0],:]
    test_y = train_label[800:train_label.shape[0],:]
    # print(train_data)
    # print(train_x)
    # print(test_x)
    # print(train_label)
    # print(train_y)
    # print(test_y)
    DL_net = DL.network(train_data.shape[1], 20, 2)
    DL_net.train(train_x, train_y, 50, test_x, test_y)