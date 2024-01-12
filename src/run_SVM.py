from support_vector_machine.multi_class import MultiSVM

def run_svm(train_data, train_label, test_data):
    SVC = MultiSVM('linear', 1)
    SVC.fit(train_data, train_label)
    return SVC.predict(test_data).reshape([-1,1])
