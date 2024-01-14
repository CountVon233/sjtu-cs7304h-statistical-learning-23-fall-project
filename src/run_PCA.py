import numpy as np
from degree_reduce import pca

def run_pca(TrainFeature, TestFeature, target_dim):
    PCA = pca.PCA()
    feature = PCA.train_and_proj(TrainFeature, target_dim)
    np.save(".\\dataset\\pca_train_feature.npy", feature)
    feature = PCA.project(TestFeature)
    np.save(".\\dataset\\pca_test_feature.npy", feature)
