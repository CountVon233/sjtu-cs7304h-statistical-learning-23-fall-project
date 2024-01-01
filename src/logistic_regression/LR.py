import numpy as np

class solver:
    def __init__(self, in_feature, out_class) -> None:
        self.in_feature = in_feature
        self.out_class = out_class
        self.w = np.ones(out_class, in_feature)
        self.b = np.zeros(out_class, 1)
    
    def train(self, train_data, train_label):
        for i in range(0, train_data.shape[0]):
            x = train_data[i:i+1,:]
            y = train_label[i:i+1,:]
            out = self.w@x + self.b
            p = np.exp(out)/( np.sum( np.exp(out), axis=0).reshape([1,-1]) )
            _w = np.ones(self.out_class, self.in_feature)@x * ( - p + np.eye(self.out_class)[y, :])