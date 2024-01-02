import numpy as np
from degree_reduce import pca

def run_pca(TrainFeature, TestFeature, target_dim):
    feature = pca.PCA().train_and_proj(TrainFeature, target_dim)
    np.save("..\\dataset\\pca_train_feature.npy", feature)
    feature = pca.PCA().train_and_proj(TestFeature, target_dim)
    np.save("..\\dataset\\pca_test_feature.npy", feature)
