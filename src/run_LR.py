import numpy as np
import pickle
import scipy
from pathlib import Path
from logistic_regression import LR

def run_lr(train_data, train_label, test_data):
    LR_solver = LR.solver(train_data.shape[1], 20)
    LR_solver.train(train_data[:1000,:], train_label[:1000], 5000)
    # LR_solver.train(train_data, train_label, 4000)
    
    for i in range(0,10):
        result = LR_solver.test(train_data[i * 1000:(i + 1) * 1000,:]).reshape([-1,1])
        acc = np.sum(train_label[i * 1000:(i + 1) * 1000].reshape([-1,1]) == result) / result.shape[0]
        print("accuracy for the {i}~{j} is {acc}".format(i = i * 1000, j = i * 1000 + 999, acc = acc))

    result = LR_solver.test(test_data)
    return result