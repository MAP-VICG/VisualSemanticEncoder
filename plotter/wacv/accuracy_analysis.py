import os
import json
import numpy as np
import matplotlib.pyplot as plt

dataset_names = {'i_awa': 'AwA GoogleNet', 'r_awa': 'AwA2 ResNet', 'i_cub': 'CUB GoogleNet', 'r_cub': 'CUB ResNet'}
result_files = sorted(['../results/accuracy/' + file for file in os.listdir('../../results/wacv/accuracy') if file.endswith('.json')])


def plot_classification_results():
    ax = None
    fig = plt.figure(figsize=(11, 6.5))
    for p, dataset in enumerate(['i_awa', 'r_awa', 'i_cub', 'r_cub']):
        rates = []
        k = len(result_files)
        curves = {'cat': {'mean': np.zeros(k), 'std': np.zeros(k)}, 's2s': {'mean': np.zeros(k), 'std': np.zeros(k)},
                  'sae': {'mean': np.zeros(k), 'std': np.zeros(k)}, 'vse': {'mean': np.zeros(k), 'std': np.zeros(k)},
                  'sem': {'mean': np.zeros(k), 'std': np.zeros(k)}, 'vis': {'mean': np.zeros(k), 'std': np.zeros(k)},
                  'pca': {'mean': np.zeros(k), 'std': np.zeros(k)}}

        for i, file in enumerate(result_files):
            if file.split('_')[-1].split('.')[0] == '2':
                rates.append(10)
            else:
                rates.append(int(file.split('_')[-1].split('.')[0]))

            with open(file) as f:
                data = json.load(f)

            curves['cat']['mean'][i], curves['cat']['std'][i] = np.mean(data[dataset]['cat']), np.std(data[dataset]['cat'])
            curves['s2s']['mean'][i], curves['s2s']['std'][i] = np.mean(data[dataset]['s2s_v1']), np.std(data[dataset]['s2s_v1'])
            curves['sae']['mean'][i], curves['sae']['std'][i] = np.mean(data[dataset]['sae']), np.std(data[dataset]['sae'])
            curves['vse']['mean'][i], curves['vse']['std'][i] = np.mean(data[dataset]['vse_v1']), np.std(data[dataset]['vse_v1'])
            curves['sem']['mean'][i], curves['sem']['std'][i] = np.mean(data[dataset]['sem']), np.std(data[dataset]['sem'])
            curves['vis']['mean'][i], curves['vis']['std'][i] = np.mean(data[dataset]['vis']), np.std(data[dataset]['vis'])
            curves['pca']['mean'][i], curves['pca']['std'][i] = np.mean(data[dataset]['pca']), np.std(data[dataset]['pca'])

        ax = plt.subplot(2, 2, p + 1)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - 0.1, box.width, box.height])

        for key in curves.keys():
            if key == 'vis':
                plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key, ls='-.', color='#C0C0C0')
            elif key == 'sae':
                plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key, ls='--', color='#808080')
            else:
                plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key)

        plt.ylabel('Accuracy', weight='bold')
        plt.xlabel('Degradation Rate (%)', weight='bold')
        plt.title('SVM - %s' % dataset_names[dataset], weight='bold')
        plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, 0.05))

    plt.show()


def plot_v1_vs_v2():
    ax = None
    fig = plt.figure(figsize=(11, 6.5))
    for p, dataset in enumerate(['i_awa', 'r_awa', 'i_cub', 'r_cub']):
        rates = []
        k = len(result_files)
        curves = {'s2s_v1': {'mean': np.zeros(k), 'std': np.zeros(k)}, 's2s_v2': {'mean': np.zeros(k), 'std': np.zeros(k)},
                  'vse_v1': {'mean': np.zeros(k), 'std': np.zeros(k)}, 'vse_v2': {'mean': np.zeros(k), 'std': np.zeros(k)}}

        for i, file in enumerate(result_files):
            if file.split('_')[-1].split('.')[0] == '2':
                rates.append(10)
            else:
                rates.append(int(file.split('_')[-1].split('.')[0]))

            with open(file) as f:
                data = json.load(f)

            curves['s2s_v1']['mean'][i], curves['s2s_v1']['std'][i] = np.mean(data[dataset]['s2s_v1']), np.std(data[dataset]['s2s_v1'])
            curves['s2s_v2']['mean'][i], curves['s2s_v2']['std'][i] = np.mean(data[dataset]['s2s_v2']), np.std(data[dataset]['s2s_v2'])
            curves['vse_v1']['mean'][i], curves['vse_v1']['std'][i] = np.mean(data[dataset]['vse_v1']), np.std(data[dataset]['vse_v1'])
            curves['vse_v2']['mean'][i], curves['vse_v2']['std'][i] = np.mean(data[dataset]['vse_v2']), np.std(data[dataset]['vse_v2'])

        ax = plt.subplot(2, 2, p + 1)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - 0.1, box.width, box.height])

        for key in curves.keys():
            plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key)

        plt.ylabel('Accuracy', weight='bold')
        plt.xlabel('Degradation Rate (%)', weight='bold')
        plt.title('SVM - %s' % dataset_names[dataset], weight='bold')
        plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.05))
    plt.show()


plot_classification_results()
plot_v1_vs_v2()
