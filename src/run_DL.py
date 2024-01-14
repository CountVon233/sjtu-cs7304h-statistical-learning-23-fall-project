import numpy as np
import pickle
import scipy
from pathlib import Path
from deep_learning_model import DL

def run_dl(train_data, train_label, test_data):
    train_x = train_data[0:1000,:]
    train_y = train_label[0:1000]
    test_x = train_data[1000:train_data.shape[0],:]
    test_y = train_label[1000:train_label.shape[0]]
    
    DL_net = DL.network(train_data.shape[1], 500, 20)
    DL_net.train(train_x, train_y, 1000, test_x, test_y, 200)
    print("AIC of DL model is {AIC}".format(AIC = DL_net.AIC(train_x, train_y)))
    # DL_net.plot()
    return DL_net.test(test_data, 1000)