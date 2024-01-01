import numpy as np

class solver:
    def __init__(self, in_feature, out_class) -> None:
        self.in_feature = in_feature
        self.out_class = out_class
        self.w = np.ones([out_class, in_feature])
        self.b = np.zeros([out_class, 1])
        self.lr = 0.1
        self.gama = 0.97
    
    def train(self, train_data, train_label, epoch):
        for iter in range(0, epoch):
            _w = np.zeros_like(self.w)
            _b = np.zeros_like(self.b)
            log_likelyhood = 0.0
            for i in range(0, train_data.shape[0]):
                x = train_data[i:i+1,:].T
                y = train_label[i:i+1,:].T
                out = self.w@x + self.b
                p = np.exp(out)/( np.sum( np.exp(out), axis=0).reshape([1,-1]) )
                log_likelyhood += np.log(p)
                _w += ( - p + np.eye(self.out_class)[y, :])@x.T
                _b += - p + np.eye(self.out_class)[y, :]
            self.w += self.lr * _w
            self.b += self.lr * _b
            self.lr *= self.gama 
            print(log_likelyhood)

