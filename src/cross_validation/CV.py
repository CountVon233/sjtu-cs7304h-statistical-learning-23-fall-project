import numpy as np
from deep_learning_model.DL import network
from logistic_regression.LR import solver
from support_vector_machine.multi_class import MultiSVM

def run_CV(train_data, train_label):
    DL_net = network(train_data.shape[1], 500, 20)
    LR_solver = solver(train_data.shape[1], 20)
    SVC_linear = MultiSVM('linear', 1)
    SVC_RBF = MultiSVM('rbf', 1)
    acc_DL = 0.0
    acc_LR = 0.0
    acc_SVC_linear = 0.0
    acc_SVC_RBF = 0.0
    for i in range(0,5):
        test_x = train_data[i*200:(i+1)*200,:]
        test_y = train_label[i*200:(i+1)*200]
        train_x = np.concatenate((train_data[:i*200,:], train_data[(i+1)*200:1000,:]), axis=0)
        train_y = np.concatenate((train_label[:i*200], train_label[(i+1)*200:1000]), axis=0)
        # train DL
        DL_net.train(train_x, train_y, 200, train_x, train_y, 200)
        # train LR
        LR_solver.train(train_x, train_y, 3000)
        # train SVM
        SVC_linear.fit(train_x, train_y)
        SVC_RBF.fit(train_x, train_y)
        # test DL
        result = DL_net.test(test_x, 200).reshape([-1,1])
        acc_DL += np.sum(test_y.reshape([-1,1]) == result) / result.shape[0]
        # test LR
        result = LR_solver.test(test_x).reshape([-1,1])
        acc_LR += np.sum(test_y.reshape([-1,1]) == result) / result.shape[0]
        # test SVM
        result = SVC_linear.predict(test_x).reshape([-1,1])
        acc_SVC_linear += np.sum(test_y.reshape([-1,1]) == result) / result.shape[0]
        result = SVC_RBF.predict(test_x).reshape([-1,1])
        acc_SVC_RBF += np.sum(test_y.reshape([-1,1]) == result) / result.shape[0]
    print(acc_DL / 5.0)
    print(acc_LR / 5.0)
    print(acc_SVC_linear / 5.0)
    print(acc_SVC_RBF / 5.0)
