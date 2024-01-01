import numpy as np
import pickle
import numpy as np
from scipy import sparse, linalg, stats
from scipy.sparse.linalg import svds, aslinearoperator, LinearOperator
from pca import PCA


path = '/data/Documents/study/2023-Fall/CS7304H-StatisticalLearning/statistical-learning-project/dataset/train_feature.pkl'

with open(path, "rb") as file:
    TrainFeature = pickle.load(file)


pca = PCA()

feature = pca.train_and_proj(TrainFeature, 10)
np.save("./dataset/pca_feature.npy", feature)
