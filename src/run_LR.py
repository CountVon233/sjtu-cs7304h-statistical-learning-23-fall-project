import numpy as np
import pickle
import scipy
from pathlib import Path
from logistic_regression import LR

def run_lr(train_data, train_label, test_data, test_label):
    LR_solver = LR.solver(train_data.shape[1], 2)
    LR_solver.train(train_data, train_label, 100)
    LR_solver.test(test_data, test_label)