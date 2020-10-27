import json
import warnings
import numpy as np
from os import listdir, path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import MatplotlibDeprecationWarning


def plot_loss(tr, te, _label):
    x = np.linspace(1, 50, 50)

    plt.errorbar(x, np.mean(np.array(tr), axis=0), yerr=np.std(np.array(tr), axis=0), label=_label[0] + '_loss_tr')
    plt.errorbar(x, np.mean(np.array(te), axis=0), yerr=np.std(np.array(te), axis=0), label=_label[0] + '_loss_vl')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - 0.1, box.width, box.height])
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))

    plt.xlabel('Epoch', weight='bold', size=9)
    plt.ylabel('Mean Square Error', weight='bold', size=9)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-3, 0))
    plt.title('Training Loss %s' % _label.split('_')[1].upper(), weight='bold', size=10)
    plt.tight_layout()


def plot_svm(tr, te, _label):
    x = np.linspace(1, 50, 50)
    plt.errorbar(x, np.mean(np.array(tr), axis=0), yerr=np.std(np.array(tr), axis=0), label=_label[0] + '_svm_tr')
    plt.errorbar(x, np.mean(np.array(te), axis=0), yerr=np.std(np.array(te), axis=0), label=_label[0] + '_svm_vl')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - 0.1, box.width, box.height])

    plt.xlabel('Epoch', weight='bold', size=9)
    plt.ylabel('Balanced Accuracy Score', weight='bold', size=9)
    plt.title('Classification Accuracy %s' % _label.split('_')[1].upper(), weight='bold', size=10)
    plt.tight_layout()


def get_history(base_path):
    _loss_tr = list()
    _loss_te = list()
    _svm_tr = list()
    _svm_te = list()

    for file in listdir(base_path):
        if not file.endswith('.json'):
            continue

        with open(path.join(base_path, file)) as f:
            data = json.load(f)
            _loss_tr.append(data['loss'])
            _loss_te.append(data['val_loss'])
            _svm_tr.append(data['svm_train'])
            _svm_te.append(data['svm_test'])

    return _loss_tr, _loss_te, _svm_tr, _svm_te


warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
fig = plt.figure(figsize=(14, 6.5))
for i, example in enumerate([('i_apy', 'r_apy'), ('i_awa', 'r_awa'), ('i_cub', 'r_cub'), ('i_sun', 'r_sun')]):
    for label in example:
        loss_tr, loss_te, svm_tr, svm_te = get_history(path.join('../results/history/', label))

        ax = fig.add_subplot(2, 4, i + 1)
        plot_loss(loss_tr, loss_te, label)
        handles, labels = ax.get_legend_handles_labels()

        if i + 1 == 4:
            fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.52))

        ax = fig.add_subplot(2, 4, i + 5)
        plot_svm(svm_tr, svm_te, label)
        handles, labels = ax.get_legend_handles_labels()

        if i + 5 == 8:
            fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.05))

fig.subplots_adjust(hspace=0.7)
plt.show()
