import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model

from encoders.vse.src.autoencoders import Simple3Layers


classes = [6, 10, 13, 15, 3, 50, 9, 47, 7, 31, 5, 38]
class_dict = {6: 'persian+cat', 10: 'siamese+cat', 13: 'tiger', 15: 'leopard',
              3: 'killer+whale', 50: 'dolphin', 9: 'blue+whale', 47: 'walrus',
              7: 'horse', 31: 'giraffe', 5: 'dalmatian', 38: 'zebra'}

with open('../../../Datasets/AWA2/AWA2_x_train_sem.txt') as f:
    sem_fts = np.array([list(map(float, row.split())) for row in f.readlines()])

with open('../../../Datasets/AWA2/AWA2_x_train_vis.txt') as f:
    vis_fts = np.array([list(map(float, row.split())) for row in f.readlines()])

with open('../../../Datasets/AWA2/AWA2_y_train.txt') as f:
    labels = np.array([int(row) for row in f.readlines()])

mask = np.array([True if lb in classes else False for lb in labels])
y_data = labels[mask]

sem_length = sem_fts.shape[1]
vis_mask = np.ones(vis_fts.shape[1])
x_data = np.hstack((vis_fts[mask, :], sem_fts[mask, :]))

del sem_fts, vis_fts, labels

model = Simple3Layers(x_data.shape[1], sem_length, x_data.shape[1]).define_ae()
model.load_weights('../../../Desktop/ae_model.h5')
encoder = Model(model.input, outputs=[model.get_layer('code').output])


def plot_pca_space(data):
    principal_components = pca.fit_transform(data)

    for target, color in zip(classes, colors):
        indices_to_keep = y_data == target
        plt.scatter(principal_components[indices_to_keep, 0],
                    principal_components[indices_to_keep, 1],
                    label=class_dict[target],
                    c=color)

    plt.tight_layout()
    plt.axis('off')


pca = PCA(n_components=2)

colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#34a02c', '#fb9a99', '#e31a1c',
          '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#666666']

fig = plt.figure(figsize=(14, 4))

ax = fig.add_subplot(141)
color_mask = np.array([1 if i in [0, 1, 2, 3, 4, 5, 6, 7] else 0 for i in range(sem_length)])
data_est = encoder.predict(x_data * np.hstack((vis_mask, color_mask)))
plot_pca_space(data_est)
plt.title('PCA of COLOR Features', weight='bold', size=10)

ax = fig.add_subplot(142)
texture_mask = np.array([1 if i in [8, 9, 10, 11, 12, 13] else 0 for i in range(sem_length)])
data_est = encoder.predict(x_data * np.hstack((vis_mask, texture_mask)))
plot_pca_space(data_est)
plt.title('PCA of TEXTURE Features', weight='bold', size=10)

ax = fig.add_subplot(143)
shape_mask = np.array([1 if i in [16, 17, 23, 24, 44, 45] else 0 for i in range(sem_length)])
data_est = encoder.predict(x_data * np.hstack((vis_mask, shape_mask)))
plot_pca_space(data_est)
plt.title('PCA of SHAPE Features', weight='bold', size=10)

ax = fig.add_subplot(144)
parts_mask = np.array([1 if i in [18, 19, 22, 25, 30, 32] else 0 for i in range(sem_length)])
data_est = encoder.predict(x_data * np.hstack((vis_mask, parts_mask)))
plot_pca_space(data_est)
plt.title('PCA of PARTS Features', weight='bold', size=10)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6)

plt.show()
