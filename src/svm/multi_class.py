import numpy

from svm import SVM

class MultiSVM():
    def __init__(self, kernel, C) -> None:
        self.classifier_list = []
        self.num_class = 0
        self.kernel = kernel
        self.C = C

    def fit(self, feature, label):
        self.num_class = numpy.max(label) + 1
        for class_i in range(self.num_class):
            self.classifier_list.append(SVM(self.kernel, C=self.C))
            train_y = (label == class_i)*2.0 - 1
            self.classifier_list[class_i].fit(feature, train_y)

    def predict(self, feature):
        result_array = numpy.zeros([feature.shape[0], self.num_class])
        dist_array = numpy.zeros([feature.shape[0], self.num_class])
        for class_i in range(self.num_class):
            result_array[:,class_i], dist_array[:,class_i] = self.classifier_list[class_i].predict(feature, value=True)
        am = numpy.argmax(dist_array, axis=1)
        result = [ result_array[i, am[i]] for i in range(feature.shape[0]) ]
        return numpy.array(result)