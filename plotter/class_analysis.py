"""
Plots chart to display classification accuracy mean and standard deviation for
SVM and Random Forest Classifiers

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


fig = plt.figure(figsize=(14, 4))

plt.subplot(131)
plot_accuracy(data_path='classification/svm_linear/awa_svm_classification.json', label='g_awa_zsl')
plot_accuracy(data_path='classification/svm_linear/awa_svm_classification_resnet.json', label='r_awa_zsl')
plot_accuracy(data_path='classification/svm_linear/cub_svm_classification.json', label='g_cub_zsl')
plot_accuracy(data_path='classification/svm_linear/cub_svm_classification_resnet.json', label='r_cub_zsl')
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('SVM LINEAR')

plt.legend(loc='upper right')
plt.tight_layout()

plt.subplot(132)
plot_accuracy(data_path='classification/svm_rbf/awa_svm_classification.json', label='g_awa_zsl')
plot_accuracy(data_path='classification/svm_rbf/awa_svm_classification_resnet.json', label='r_awa_zsl')
plot_accuracy(data_path='classification/svm_rbf/cub_svm_classification.json', label='g_cub_zsl')
plot_accuracy(data_path='classification/svm_rbf/cub_svm_classification_resnet.json', label='r_cub_zsl')
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('SVM RBF')

plt.legend(loc='upper right')
plt.tight_layout()

plt.subplot(133)
plot_accuracy(data_path='classification/forest/awa_svm_classification.json', label='g_awa_zsl')
plot_accuracy(data_path='classification/forest/awa_svm_classification_resnet.json', label='r_awa_zsl')
plot_accuracy(data_path='classification/forest/cub_svm_classification.json', label='g_cub_zsl')
plot_accuracy(data_path='classification/forest/cub_svm_classification_resnet.json', label='r_cub_zsl')
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('RANDOM FOREST')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
