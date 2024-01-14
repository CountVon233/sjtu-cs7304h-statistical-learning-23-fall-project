import numpy as np

class solver:
    def __init__(self, in_feature, out_class) -> None:
        self.in_feature = in_feature
        self.out_class = out_class
        self.w = np.ones([out_class, in_feature])
        self.b = np.zeros([out_class, 1])
        self.lr = 0.1
        self.gamma = 1
    
    def train(self, train_data, train_label, epoch):
        for iter in range(0, epoch):
            _w = np.zeros_like(self.w)
            _b = np.zeros_like(self.b)
            log_likelihood = 0.0
            for i in range(0, train_data.shape[0]):
                x = train_data[i:i+1,:].T
                y = train_label[i]
                out = self.w@x + self.b
                p = np.exp(out)/( np.sum( np.exp(out), axis=0).reshape([1,-1]) )
                log_likelihood += np.log(p[y])
                _w += ( - p + np.eye(self.out_class)[y.reshape(-1)].T)@x.T
                _b += - p + np.eye(self.out_class)[y.reshape(-1)].T
            self.w += self.lr * _w
            self.b += self.lr * _b
            self.lr *= self.gamma 
            print("log_likelyhood in iter {i} is {log}".format(i = iter, log = log_likelihood))
    
    def test(self, test_data):
        result = []
        for i in range(0, test_data.shape[0]):
            x = test_data[i:i+1,:].T
            out = self.w@x + self.b
            p = np.exp(out)/( np.sum( np.exp(out), axis=0).reshape([1,-1]) )
            result.append(np.argmax(p, axis = 0))
        return np.concatenate(result, axis = 0).reshape([-1,1])
    
    def AIC(self, train_data, train_label):
        log_lik = 0.0
        for i in range(0, train_data.shape[0]):
            x = train_data[i:i+1,:].T
            y = train_label[i]
            out = self.w@x + self.b
            p = np.exp(out)/( np.sum( np.exp(out), axis=0).reshape([1,-1]) )
            log_lik += np.log(p[y])
        return 2.0 * (- log_lik + self.out_class * self.in_feature + self.out_class) / train_data.shape[0]

