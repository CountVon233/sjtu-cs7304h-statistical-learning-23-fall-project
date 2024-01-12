import numpy as np
import pickle
import scipy
from pathlib import Path
from deep_learning_model import DL

def run_dl(train_data, train_label, test_data):
    train_x = train_data[0:10000,:]
    train_y = train_label[0:10000,:]
    test_x = train_data[10000:train_data.shape[0],:]
    test_y = train_label[10000:train_label.shape[0],:]
    # print(train_data)
    # print(train_x)
    # print(test_x)
    # print(train_label)
    # print(train_y)
    # print(test_y)
    DL_net = DL.network(train_data.shape[1], 1000, 20)
    DL_net.train(train_x, train_y, 1000, test_x, test_y, 100)
    DL_net.plot()
    return DL_net.test(test_data, 1000)