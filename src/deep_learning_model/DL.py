import numpy as np
import pickle
import scipy
from pathlib import Path
import matplotlib.pyplot as plt

class network:
    def __init__(self, data, in_feature, hidden, out_class,):
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


    def plot(loss, acc, lr):    #画图
        plt.subplot(3,1,1)
        plt.plot(loss)
        plt.title('Test result')
        plt.ylabel('Loss')
        plt.subplot(3,1,2)
        plt.plot(acc)
        plt.ylabel('Acc')
        plt.subplot(3,1,3)
        plt.plot(lr)
        plt.ylabel('lr')
        plt.savefig('..\\..\\output\\DNN_fig.png')


    def forward(self):
        pass


    def train(self, data, label):
        epochs = 100


