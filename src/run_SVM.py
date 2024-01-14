import numpy as np
from support_vector_machine.multi_class import MultiSVM

def run_svm(train_data, train_label, test_data):
    SVC = MultiSVM('linear', 1)
    # SVC = MultiSVM('rbf', 1)
    SVC.fit(train_data[:1000,:], train_label[:1000])
    
    for i in range(0,10):
        result = SVC.predict(train_data[i * 1000: (i + 1) * 1000,:]).reshape([-1,1])
        acc = np.sum(train_label[i * 1000: (i + 1) * 1000].reshape([-1,1]) == result) / result.shape[0]
        print("accuracy for the {i}~{j} is {acc}".format(i = i * 1000, j = i * 1000 + 999, acc = acc))

    return SVC.predict(test_data).reshape([-1,1])
