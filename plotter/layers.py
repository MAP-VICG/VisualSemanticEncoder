import os
import json
import numpy as np
import matplotlib.pyplot as plt

dataset_names = {'i_awa': 'AwA GoogleNet', 'r_awa': 'AwA2 ResNet', 'i_cub': 'CUB GoogleNet', 'r_cub': 'CUB ResNet'}
result_files = sorted(['../results/layers/' + file for file in os.listdir('../results/layers') if file.endswith('.json')])


def plot_classification_results():
    ax = None
    fig = plt.figure(figsize=(14, 6.5))
    for d, klass in enumerate(['vse', 's2s']):
        for p, dataset in enumerate(['i_awa', 'r_awa', 'i_cub', 'r_cub']):
            rates = []
            k = len(result_files)
            curves = {klass + '_1l': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_2l': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_3l': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_4l': {'mean': np.zeros(k), 'std': np.zeros(k)}}

            for i, file in enumerate(result_files):
                rates.append(int(file.split('_')[-1].split('.')[0]))

                with open(file) as f:
                    data = json.load(f)

                try:
                    curves[klass + '_1l']['mean'][i] = np.mean(data[dataset][klass + '_1l'])
                    curves[klass + '_2l']['mean'][i] = np.mean(data[dataset][klass + '_2l'])
                    curves[klass + '_3l']['mean'][i] = np.mean(data[dataset][klass + '_3l'])
                    curves[klass + '_4l']['mean'][i] = np.mean(data[dataset][klass + '_4l'])

                    curves[klass + '_1l']['std'][i] = np.std(data[dataset][klass + '_1l'])
                    curves[klass + '_2l']['std'][i] = np.std(data[dataset][klass + '_2l'])
                    curves[klass + '_3l']['std'][i] = np.std(data[dataset][klass + '_3l'])
                    curves[klass + '_4l']['std'][i] = np.std(data[dataset][klass + '_4l'])
                except KeyError:
                    pass

            ax = fig.add_subplot(2, 4, d * 4 + p + 1)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 - 0.1, box.width, box.height])

            for key in curves.keys():
                plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key.split('_')[-1].upper())

            plt.ylabel('Accuracy', weight='bold', size=9)
            plt.xlabel('Degradation Rate (%)', weight='bold', size=9)
            plt.title('%s - %s' % (klass.upper(), dataset_names[dataset]),  weight='bold', size=10)
            plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.05))
    plt.show()


plot_classification_results()
