import numpy as np
import pickle
from scipy import sparse, linalg, stats
from scipy.sparse.linalg import svds, aslinearoperator, LinearOperator
from pca import PCA
from matplotlib import pyplot as plt
from pathlib import Path

train_relative_path = "..\\..\\dataset\\train_feature.pkl"
dir = Path(__file__).parent
train_absolute_path = dir.joinpath(train_relative_path)

label_relative_path = "..\\..\\dataset\\train_labels.npy"
label_absolute_path = dir.joinpath(label_relative_path)

with open(train_absolute_path, "rb") as file:
    TrainFeature = pickle.load(file)

label = np.load(label_absolute_path)
color = [ plt.get_cmap("seismic", 20)(i) for i in label[:500] ]
pca = PCA()

feature = pca.train_and_proj(TrainFeature, 3)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
plt.set_cmap(plt.get_cmap('seismic', 20))
im = ax.scatter( feature[:500, 0], feature[:500, 1], feature[:500, 2], s=20, c=color, marker='.' )

plt.show()
plt.savefig('.\\colored.png')


