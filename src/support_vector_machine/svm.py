from typing import Tuple
import numpy
from numpy.core.numeric import identity
from sklearn.svm import SVC
import random
from matplotlib import pyplot as plt

class SVM():
    def __init__(self, kernel = 'linear',C = 1, gamma = 1) -> None:
        self.vec = None
        self.lab = None
        self.a = None
        self.KM = None
        self.fx = None
        self.b = 0
        self.C = C
        self.epsilon = 0.001
        self.kernel = kernel
        self.gamma = gamma
    def fit(self, feature, label, max_iter = 10000):
        self.vec = feature
        self.lab = label.reshape((-1,1))
        self.getKM()
        self.a = numpy.zeros_like(self.lab, dtype=numpy.float32) * self.C
        # self.a = numpy.random.rand(self.lab.shape[0], 1)*self.C
        self.SMO(max_iter=max_iter)
        self.simplify()
        return (self.fx > 0) * 2 -1
    def getKM(self, feature = None):
        # linear kernel
        if self.kernel == 'linear':
            if feature is None:
                self.KM = numpy.dot(self.vec, self.vec.T)
            else:
                return numpy.dot(self.vec, feature.T)
        elif self.kernel == 'rbf':
            if feature is None:
                vec = self.vec.reshape((1,self.vec.shape[0],-1))
                vec = numpy.concatenate( [vec]*self.vec.shape[0], 0 )
                km = vec - vec.transpose((1,0,2))
                km = numpy.linalg.norm(km,axis=2)**2
                km = numpy.exp(-self.gamma * km)
                self.KM = km
                pass
            else:
                vec = self.vec.reshape((1,self.vec.shape[0],-1))
                vec = numpy.concatenate( [vec]*feature.shape[0], 0 )
                ftr = feature.reshape((1,feature.shape[0],-1))
                ftr = numpy.concatenate( [ftr]*self.vec.shape[0], 0 )
                km = ftr - vec.transpose((1,0,2))
                km = numpy.linalg.norm(km,axis=2)**2
                km = numpy.exp(-self.gamma * km)
                return km
    def simplify(self):
        idx = numpy.array(range(self.a.shape[0])).reshape((-1,1))
        idx  = idx[self.a!=0]
        self.vec = self.vec[idx]
        self.lab = self.lab[idx]
        self.fx = self.fx[idx]
        self.a = self.a[idx]
        self.getKM()
        pass
    def SMO(self, max_iter):
        self.fx = numpy.dot(self.KM , self.lab * self.a) + self.b
        # self.fx = (self.fx > 0)*2 - 1
        for stp in range(max_iter):
            if stp < 10 * self.lab.shape[0]:
                ij = self.choice(stp=stp % self.lab.shape[0])
            else:
                ij = self.choice()
            if ij is not None:
                self.step(ij[0],ij[1])
            else:
                break
            # if stp % 1000 == 0:
                # print("iter = %d"%stp)
    def step(self,i:int,j:int):
        E_i = self.fx[i] - self.lab[i]
        E_j = self.fx[j] - self.lab[j]
        eta = self.KM[i,i] + self.KM[j,j] - self.KM[i,j] - self.KM[j,i]
        a_j = self.a[j] + self.lab[j] * (E_i - E_j) /eta

        L = max(0, self.a[j] - self.a[i]) \
            if (self.lab[i] != self.lab[j])\
                else max(0, self.a[i] + self.a[j] - self.C)
        H = min(self.C, self.C + self.a[j] - self.a[i] )\
            if (self.lab[i] != self.lab[j])\
                else min( self.C, self.a[i] + self.a[j] )
        assert H >= 0
        assert L <= self.C
        a_j = numpy.clip(a_j,L,H)
        a_i = numpy.clip(self.a[i] + self.lab[i] * self.lab[j] * (self.a[j] - a_j),0, self.C)

        b_i = -E_i \
            - self.lab[i] * self.KM[i,i] * ( a_i - self.a[i] )\
                - self.lab[j] * self.KM[j,i] * ( a_j - self.a[j] )\
                    + self.b
        b_j = -E_j \
            - self.lab[i] * self.KM[i,j] * ( a_i - self.a[i] )\
                - self.lab[j] * self.KM[j,j] * ( a_j - self.a[j] )\
                    + self.b
        if a_i > 0 and a_i < self.C:
            self.b = b_i
        elif a_j > 0 and a_j < self.C:
            self.b = b_j
        else:
            self.b = b_i/2 + b_j/2
        # self.b = b_i/2 + b_j/2
        self.a[i] = a_i
        self.a[j] = a_j

        self.fx = numpy.dot(self.KM , self.lab * self.a) + self.b
        # self.fx = (self.fx > 0)*2 - 1
        pass
    def choice(self,stp = None) -> Tuple[int,int]:
        if stp is not None:
            i = stp

            # E = self.fx - self.lab
            # E_lr = numpy.abs(E-E[i]).reshape([-1])
            # j = numpy.argsort(E_lr)[-10:]
            # j = random.choice(j)

            # E = self.fx - self.lab
            # if E[i] > 0:
            #     j = numpy.argmin(E.reshape([-1]))
            # else:
            #     j = numpy.argmax(E.reshape([-1]))

            j = random.randint(0,self.lab.shape[0]-1)
            while j == i:
                j = random.randint(0,self.lab.shape[0]-1)

            return (i,j)

        idx_p = (self.a > 0) * 1
        idx_c = (self.a == self.C) * 1
        idx_s = idx_c + idx_p
        found_i = False
        idx = numpy.array(list(range(self.lab.shape[0]))).reshape((-1,1))
        support_vec = idx[idx_s == 1]
        c_vec = idx[idx_s == 2]
        zero_vec = idx[idx_s == 0]
        numpy.random.shuffle(support_vec)
        if len(support_vec) > 0:
            for i in support_vec:
                if abs(self.lab[i] * self.fx[i] - 1) >= self.epsilon:
                    found_i = True
                    break
        if len(c_vec) > 0 and found_i == False :
            for i in c_vec:
                if self.lab[i] * self.fx[i] > 1:
                    found_i = True
                    break
        if len(zero_vec) > 0 and found_i == False:
            for i in zero_vec:
                if self.lab[i] * self.fx[i] < 1:
                    found_i = True
                    break
        # found_i = True
        # i = random.randint(0,221)
        if found_i == True:
            E = self.fx - self.lab
            E_lr = numpy.abs(E-E[i])
            j = numpy.argpartition(E_lr[:,0],-5)[-5:]
            j = random.choice(j)
            # j = numpy.argmax(E_lr)
            return (i,j)
        return None
    def predict(self, feature, value = False):
        ker = self.getKM(feature=feature)
        fx = numpy.dot(ker.T, self.lab * self.a) + self.b
        if value:
            return (fx > 0) * 2 -1, abs(fx)
        return (fx > 0) * 2 -1
        pass



if __name__ == "__main__":

    C = 1
    svm = SVM(kernel='linear',C=C)
    numpy.random.seed(5)
    # data_x = numpy.concatenate([numpy.random.randn(500, 2)+2, numpy.random.randn(500, 2)-2])
    # data_y = numpy.concatenate([numpy.ones([500, 1]), -1*numpy.ones([500, 1])] )
    data_x = numpy.random.rand(500,2)
    data_y = ((data_x[:,0] > data_x[:,1]) * 1).reshape([-1])*2-1
    data_x = data_x - numpy.mean(data_x, axis=0)
    data_x = data_x / numpy.std(data_x, axis=0)
    train_x = data_x[:int(0.8*data_x.shape[0]), :]
    train_y = data_y[:int(0.8*data_x.shape[0])]
    test_x = data_x[int(0.8*data_x.shape[0]):, :]
    test_y = data_y[int(0.8*data_x.shape[0]):]
    
    svm.fit(train_x, train_y)
    # result = svm.predict(train_x)
    # print( sum(result.reshape([-1]) == train_y.reshape([-1]))/train_y.shape[0] )
    result = svm.predict(test_x)
    print( sum(result.reshape([-1]) == test_y.reshape([-1]))/test_y.shape[0] )

    posi_idx = numpy.nonzero(train_y==1)
    nega_idx = numpy.nonzero(train_y==-1)

    plt.scatter(train_x[posi_idx][:,0], train_x[posi_idx][:,1], color='blue', s=10, marker='.')
    plt.scatter(train_x[nega_idx][:,0], train_x[nega_idx][:,1], color='yellow', s=10, marker='.')
    plt.scatter(svm.vec[:,0], svm.vec[:,1], marker='+')
    plt.savefig("./fig.png")


    svc = SVC(C=1, kernel="linear")
    svc.fit(train_x, train_y)
    result = svc.predict(test_x)
    print( sum(result.reshape([-1]) == test_y.reshape([-1]))/test_y.shape[0] )


