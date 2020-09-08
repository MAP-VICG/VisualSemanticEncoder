import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model

from encoders.sec.src.autoencoders import Simple3Layers


classes = [6, 10, 13, 15, 3, 50, 9, 47, 7, 31, 5, 38]
class_dict = {6: 'persian+cat', 10: 'siamese+cat', 13: 'tiger', 15: 'leopard',
              3: 'killer+whale', 50: 'dolphin', 9: 'blue+whale', 47: 'walrus',
              7: 'horse', 31: 'giraffe', 5: 'dalmatian', 38: 'zebra'}

with open('../../Datasets/AWA2/AWA2_x_train_sem.txt') as f:
    sem_fts = np.array([list(map(float, row.split())) for row in f.readlines()])

with open('../../Datasets/AWA2/AWA2_x_train_vis.txt') as f:
    vis_fts = np.array([list(map(float, row.split())) for row in f.readlines()])

with open('../../Datasets/AWA2/AWA2_y_train.txt') as f:
    labels = np.array([int(row) for row in f.readlines()])

mask = np.array([True if lb in classes else False for lb in labels])

y_data = labels[mask]
x_vis = vis_fts[mask, :]
x_sem = sem_fts[mask, :]
x_data = np.hstack((x_vis, x_sem))

del sem_fts, vis_fts, labels

model = Simple3Layers(x_data.shape[1], x_data.shape[1] - 2048, x_data.shape[1]).define_ae()
model.save_weights('../../../Desktop/ae_model.h5')
encoder = Model(model.input, outputs=[model.get_layer('code').output])
data_est = encoder.predict(x_data)


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


np.random.seed(19680801)
pca = PCA(n_components=2)

colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#34a02c', '#fb9a99', '#e31a1c',
          '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#666666']

fig = plt.figure(figsize=(10, 4))

plt.subplot(131)
plot_pca_space(x_vis)
plt.title('PCA of Visual Features', weight='bold', size=10)

plt.subplot(132)
plot_pca_space(x_sem)
plt.title('PCA of Semantic Features', weight='bold', size=10)

ax = fig.add_subplot(133)
plot_pca_space(data_est)
plt.title('PCA of SEC Features', weight='bold', size=10)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6)

plt.show()
