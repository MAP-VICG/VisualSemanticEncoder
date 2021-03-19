import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from sklearn.metrics import silhouette_score

from encoders.vse.src.autoencoders import StraightAutoencoder


classes = [6, 10, 13, 15, 3, 50, 9, 47, 7, 31, 5, 38]
class_dict = {6: 'persian+cat', 10: 'siamese+cat', 13: 'tiger', 15: 'leopard',
              3: 'killer+whale', 50: 'dolphin', 9: 'blue+whale', 47: 'walrus',
              7: 'horse', 31: 'giraffe', 5: 'dalmatian', 38: 'zebra'}

data = loadmat('../../../Datasets/awa2_data_resnet50.mat')
lbs = {data['param']['testclasses_id'][0][0][i][0]: attrs for i, attrs in enumerate(data['S_te_pro'])}
sem_fts = np.vstack((data['S_tr'], np.array([lbs[label[0]] for label in data['param']['test_labels'][0][0]])))
vis_fts = np.vstack((data['X_tr'], data['X_te']))
labels = np.vstack((data['param']['train_labels'][0][0], data['param']['test_labels'][0][0]))

mask = np.array([True if lb in classes else False for lb in labels])

y_data = labels[mask]
x_vis = vis_fts[mask, :]
x_sem = sem_fts[mask, :]
x_data = np.hstack((x_vis, x_sem))

del sem_fts, vis_fts, labels

ae_model = StraightAutoencoder(x_data.shape[1], x_data.shape[1] - 2048, x_data.shape[1])
ae_model.define_ae()
model = ae_model.ae
model.load_weights('../data/ae_model_wacv.h5')
encoder = Model(model.input, outputs=[model.get_layer('code').output])
data_est = encoder.predict(x_data)


def plot_pca_space(type, data):
    pca_ten = PCA(n_components=10)
    pca_ten.fit(data)
    print(type + ': ' + ' & '.join([str(round(v, 4)) for v in pca_ten.explained_variance_ratio_]))

    principal_components = pca.fit_transform(data)

    for target, color in zip(classes, colors):
        indices_to_keep = y_data == target
        indices_to_keep = [i[0] for i in indices_to_keep]
        plt.scatter(principal_components[indices_to_keep, 0],
                    principal_components[indices_to_keep, 1],
                    label=class_dict[target],
                    c=color)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - 0.1, box.width * 0.9, box.height * 0.9])

    plt.tight_layout()
    plt.axis('off')
    return silhouette_score(principal_components, [i[0] for i in y_data])


pca = PCA(n_components=2)

colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#34a02c', '#fb9a99', '#e31a1c',
          '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#666666']

fig = plt.figure(figsize=(10, 3))

ax = fig.add_subplot(131)
coef = plot_pca_space('VIS', x_vis)
plt.title('PCA of Visual Features\n[Silhouette %0.4f]' % coef, weight='bold', size=10)

ax = fig.add_subplot(132)
coef = plot_pca_space('SEM', x_sem)
plt.title('PCA of Semantic Features\n[Silhouette %0.4f]' % coef, weight='bold', size=10)

ax = fig.add_subplot(133)
coef = plot_pca_space('VSE', data_est)
plt.title('PCA of VSE Features\n[Silhouette %0.4f]' % coef, weight='bold', size=10)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6)

plt.show()
