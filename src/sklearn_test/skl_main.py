from pca import PCA
import numpy
import pickle
import sklearn

path = '/data/Documents/study/2023-Fall/CS7304H-StatisticalLearning/statistical-learning-project/dataset/train_feature.pkl'
with open(path, "rb") as file:
    TrainFeature = pickle.load(file)
pca = PCA()
feature = pca.train_and_proj(TrainFeature, 320)
# feature = numpy.load("./dataset/pca_feature.npy")
label = numpy.load("./dataset/train_labels.npy")
train_x = feature[:10000,:]
train_y = label[:10000]
test_x = feature[10000:,:]
test_y = label[10000:]

print("training")

##  svm
# from sklearn import svm
# predictor = svm.SVC(gamma='scale', C=1.0, kernel="linear", decision_function_shape='ovr')
# predictor.fit(train_x, train_y)
# result = predictor.predict(test_x)
# print(numpy.sum(result == test_y)/result.shape[0])

##  Log reg
# from sklearn.linear_model import LogisticRegression
# predictor = LogisticRegression()
# predictor = predictor.fit(train_x, train_y)
# # result = predictor.predict(train_x)
# # print(numpy.sum(result == train_y)/result.shape[0])
# result = predictor.predict(test_x)
# print(numpy.sum(result == test_y)/result.shape[0])


## kNN
# from sklearn.neighbors import KNeighborsClassifier
# predictor = KNeighborsClassifier(n_neighbors=17, p=2, metric="minkowski")
# predictor.fit(train_x, train_y)
# result = predictor.predict(test_x)
# print(numpy.sum(result == test_y)/result.shape[0])

## naive bayes
from sklearn.naive_bayes import GaussianNB
predictor = GaussianNB()
predictor = predictor.fit(train_x, train_y)
result = predictor.predict(test_x)
print(numpy.sum(result == test_y)/result.shape[0])



