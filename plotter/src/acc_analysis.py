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

    plt.errorbar(degradation_rates, y, yerr=err, label=label)


fig = plt.figure()
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate of Semantic Data')
plt.title('Accuracy analysis for ZSL and SVM Classification')

plot_accuracy(data_path='../data/awa_v2s_projection_random.json', label='awa_zsl')
plot_accuracy(data_path='../data/cub_v2s_projection_random.json', label='cub_zsl')
plot_accuracy(data_path='../data/awa_svm_classification_random.json', label='awa_svm')
plot_accuracy(data_path='../data/cub_svm_classification_random.json', label='cub_svm')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
