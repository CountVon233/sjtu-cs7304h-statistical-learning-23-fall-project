import numpy as np
import pickle
import scipy
from pathlib import Path
import matplotlib.pyplot as plt

class network:
    def __init__(self, in_feature, hidden, out_class,):
        self.in_feature = in_feature          # 设置输入特征维度
        self.hidden = hidden        # 隐藏层神经元
        self.out_class = out_class            # 类别数
        self.lr = 0.1                # 学习率
        self.gama = 0.97             # 下降因子
        self.w_hidden = np.random.rand(hidden, in_feature)    # 隐藏层权重
        self.b_hidden = np.random.rand(hidden, 1)             # 隐藏层偏置
        self.w_out = np.random.rand(out_class, hidden)        # 输出层权重
        self.b_out = np.random.rand(out_class, 1)             # 输出层偏置
        self.LOSS = []       # 记录损失
        self.ACC  = []       # 正确率
        self.LR   = []       # 学习率变化的列表


    def plot(self):    #画图
        plt.subplot(3,1,1)
        plt.plot(self.LOSS)
        plt.title('Test result')
        plt.ylabel('Loss')
        plt.subplot(3,1,2)
        plt.plot(self.ACC)
        plt.ylabel('Acc')
        plt.subplot(3,1,3)
        plt.plot(self.LR)
        plt.ylabel('lr')
        plt.savefig('..\\..\\output\\DNN_fig.png')


    def train(self, train_data, train_label, mini_batch, test_data, test_label, epochs):
        for epoch in range(epochs):
            losses = []
            for i in range(0, train_data.shape[0], mini_batch):
                x_1 = train_data[i:min(i+mini_batch, train_data.shape[0]),:].T
                y_1 = train_label[i:i+mini_batch,:].T
                x_2 = self.w_hidden@x_1 + self.b_hidden
                x_3 = np.exp(x_2)/(1 + np.exp(x_2))
                x_4 = (self.w_out@x_3 + self.b_out)
                x_5 = np.exp(x_4)/( np.sum( np.exp(x_4), axis=0).reshape([1,-1]) )  
                x_6 = np.log(x_5)
                y_1_onehot = np.eye(self.out_class)[y_1.reshape(-1)]
                loss = - np.trace(y_1_onehot@x_6)

                losses.append(loss)

                _x_6 = - y_1_onehot.T
                _x_5 = _x_6/x_5
                _x_4 = np.zeros_like(x_4)
                for i in range(0, mini_batch):
                    x_5_i = x_5[:,i:i+1]
                    _x_5_i = _x_5[:,i:i+1]
                    _x_4[:,i:i+1] = (- x_5_i@x_5_i.T + np.identity(self.out_class) * x_5_i)@_x_5_i

                _b_out = np.sum(_x_4, axis = 1, keepdims = True)        #计算输出层参数的梯度
                _w_out = _x_4@x_3.T
                _x_3    = self.w_out.T@_x_4

                _x_2    = x_3 * (1 - x_3) * _x_3  #sigmoid函数 梯度反向传播

                _w_hidden =  _x_2@x_1.T  #计算隐藏层参数的梯度
                _b_hidden = np.sum(_x_2, axis = 1, keepdims = True)


                self.w_hidden -= self.lr * _w_hidden / mini_batch            #梯度下降
                self.b_hidden -= self.lr * _b_hidden / mini_batch
                self.w_out    -= self.lr * _w_out / mini_batch
                self.b_out    -= self.lr * _b_out.reshape(self.b_out.shape) / mini_batch
            
            acc = 0
            for i in range(0, test_data.shape[0], mini_batch):
                x_1 = test_data[i:min(i+mini_batch, test_data.shape[0]),:].T
                y_1 = test_label[i:i+mini_batch,:].T
                x_2 = self.w_hidden@x_1 + self.b_hidden
                x_3 = np.exp(x_2)/(1 + np.exp(x_2))
                x_4 = (self.w_out@x_3 + self.b_out)
                x_5 = np.exp(x_4)/( np.sum( np.exp(x_4), axis=0).reshape([1,-1]) )
                x_5 = (x_5 == np.max(x_5, axis=1)) 
                acc += mini_batch - np.sum(np.absolute(x_5 - y_1)) / 2
            
            self.lr *= self.gama
            print('loss = %f,\tacc = %f, \tlr = %f'%(sum(losses)/train_data.shape[0], acc/test_data.shape[0], self.lr))
            self.LOSS.append(sum(losses)/train_data.shape[0])
            self.ACC .append(acc/test_data.shape[0])
            self.LR  .append(self.lr)
        # self.plot()


    def test(self, data, mini_batch):
        predict_result = []
        for i in range(0, data.shape[0], mini_batch):
            x_1 = data[i:min(i+mini_batch, data.shape[0]),:].T
            x_2 = self.w_hidden@x_1 + self.b_hidden
            x_3 = np.exp(x_2)/(1 + np.exp(x_2))
            x_4 = (self.w_out@x_3 + self.b_out)
            x_5 = np.exp(x_4)/( np.sum( np.exp(x_4) ) )
            predict_result.append(np.argmax(x_5, axis=1).reshape([-1,1]))
        return np.concatenate(predict_result, axis = 0)


