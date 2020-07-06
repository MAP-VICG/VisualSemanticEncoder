"""
Plots chart to display classification accuracy mean and standard deviation for each
degradation rate analyzed.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: May 23, 2020

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC)
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
"""
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy(data_path, label):
    """
    Computes errors from the standard deviation and plots accuracy details

    :param data_path: string with path to json file with results
    :param label: type of data being checked: awa or cub
    :return: None
    """
    degradation_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    with open(data_path) as f:
        data = json.loads(json.load(f))

    err = []
    y = []

    for rate in degradation_rates:
        y.append(data[str(rate)]['mean'])
        err.append(data[str(rate)]['std'])

    plt.errorbar(np.array(degradation_rates) * 100, y, yerr=err, label=label)


fig = plt.figure()

plt.subplot(221)
plot_accuracy(data_path='data/awa_v2s_projection_random.json', label='g_awa_zsl')
plot_accuracy(data_path='data/cub_v2s_projection_random.json', label='g_cub_zsl')
plot_accuracy(data_path='data/awa_v2s_projection_random_resnet.json', label='r_awa_zsl')
plot_accuracy(data_path='data/cub_v2s_projection_random_resnet.json', label='r_cub_zsl')
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('ZSL Classification - SAE')

plt.legend(loc='upper right')
plt.tight_layout()

plt.subplot(222)
plot_accuracy(data_path='data/awa_svm_classification_random.json', label='g_awa_svm')
plot_accuracy(data_path='data/cub_svm_classification_random.json', label='g_cub_svm')
plot_accuracy(data_path='data/awa_svm_classification_random_resnet.json', label='r_awa_svm')
plot_accuracy(data_path='data/cub_svm_classification_resnet_rbf.json', label='r_cub_svm_rbf')
plot_accuracy(data_path='data/cub_svm_classification_resnet_linear.json', label='r_cub_svm_linear')
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('SVM Classification - SAE')

plt.legend(loc='upper right')
plt.tight_layout()

plt.subplot(223)
plot_accuracy(data_path='data/awa_v2s_projection_random_sec.json', label='g_awa_zsl')
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('ZSL Classification - SEC')

plt.legend(loc='upper right')
plt.tight_layout()

plt.subplot(224)
plot_accuracy(data_path='data/awa_svm_classification_random_sec.json', label='g_awa_svm')
plot_accuracy(data_path='data/cub_svm_classification_random_sec.json', label='g_cub_svm')
plot_accuracy(data_path='data/awa_svm_classification_random_resnet_sec.json', label='r_awa_svm')
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('SVM Classification - SEC')

plt.legend(loc='upper right')
plt.tight_layout()

plt.show()
