
import numpy
from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import norm




class PCA:
    def __init__(self) -> None:
        self.feature_degree = 0                 # feature degree of input data
        self.trunc_degree = 0                   # feature degree of output data
        self.projectMatrix = numpy.zeros([0,0]) # project matrix
        

    # generate projMtrx by training data
    #   data: N * m sparse matrix, N=samples, m=features
    #   trunc_degree: desire output feature degree
    def train_and_proj(self, data:sparse.csr_matrix, trunc_degree):
        assert data.shape[1] >= trunc_degree
        data = data[:,norm(data, axis=0).nonzero()[0]]
        data = data - numpy.mean(data, axis=0)
        ut, st, vt = svds(data, trunc_degree)

        self.feature_degree = data.shape[1]
        self.trunc_degree = trunc_degree
        self.projectMatrix = vt

        return ut@numpy.diag(st)


    def project(self, data):
        return data @ self.projectMatrix
