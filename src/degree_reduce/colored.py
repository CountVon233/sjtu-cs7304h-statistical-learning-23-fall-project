import numpy as np
import pickle
from scipy import sparse, linalg, stats
from scipy.sparse.linalg import svds, aslinearoperator, LinearOperator
from pca import PCA
from matplotlib import pyplot as plt
from pathlib import Path

relative_data_path = "..\\..\\dataset\\train_feature.pkl"
dir = Path(__file__).parent
absolute_data_path = dir.joinpath(relative_data_path)

relative_label_path = "..\\..\\dataset\\train_labels.npy"
absolute_label_path = dir.joinpath(relative_label_path)

with open(absolute_data_path, "rb") as file:
    TrainFeature = pickle.load(file)

label = np.load(absolute_label_path)
color = [ plt.get_cmap("seismic", 20)(i) for i in label[:1000] ]
pca = PCA()

feature = pca.train_and_proj(TrainFeature, 3)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
plt.set_cmap(plt.get_cmap('seismic', 20))
im = ax.scatter( feature[:1000, 0], feature[:1000, 1], feature[:1000, 2], s=20, c=color, marker='.' )

plt.show()
plt.savefig('./src/degree_reduce/colored.png', dpi=300)


