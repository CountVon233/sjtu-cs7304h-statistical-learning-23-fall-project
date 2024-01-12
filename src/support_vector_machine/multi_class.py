import numpy

from svmcopy import SVM

class MultiSVM():
    def __init__(self, kernel, C) -> None:
        self.classifier_list = []
        self.num_class = 0
        self.kernel = kernel
        self.C = C

    def fit(self, feature, label):
        self.num_class = int(numpy.max(label)) + 1
        for class_i in range(self.num_class):
            self.classifier_list.append(SVM(self.kernel, C=self.C))
            train_y = (label == class_i)*2.0 - 1
            self.classifier_list[class_i].fit(feature, train_y)

    def predict(self, feature):
        result_array = numpy.zeros([feature.shape[0], self.num_class])
        dist_array = numpy.zeros([feature.shape[0], self.num_class])
        for class_i in range(self.num_class):
            result_array[:,class_i:class_i+1], dist_array[:,class_i:class_i+1] = self.classifier_list[class_i].predict(feature, value=True)
        am = numpy.argmax(dist_array, axis=1)
        return am
    
if __name__ == "__main__":
    sample = 5
    center = numpy.array([[10, 10], [10, -10], [-10, 10], [-10, -10]])
    data = numpy.random.randn(sample*4, 2)
    label = numpy.zeros([sample*4,1],)
    for i in range(4):
        data[i*sample:(i+1)*sample, :] += center[i:i+1, :]
        label[i*sample:(i+1)*sample, :] = i
    SVC = MultiSVM("linear", 1)
    SVC.fit(data, label)
    result = SVC.predict(data)
    print(result)