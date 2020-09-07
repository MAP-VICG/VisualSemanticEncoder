import json
import numpy as np
import matplotlib.pyplot as plt

dataset_names = {'i_awa': 'AwA GoogleNet', 'r_awa': 'AwA2 ResNet', 'i_cub': 'CUB GoogleNet', 'r_cub': 'CUB ResNet'}
result_files = ['../results/layers/classification_results_000.json',
                '../results/layers/classification_results_010.json',
                '../results/layers/classification_results_020.json',
                '../results/layers/classification_results_030.json',
                '../results/layers/classification_results_040.json',
                '../results/layers/classification_results_050.json',
                '../results/layers/classification_results_060.json',
                '../results/layers/classification_results_070.json',
                '../results/layers/classification_results_080.json',
                '../results/layers/classification_results_090.json',
                '../results/layers/classification_results_100.json']


def plot_classification_results():
    plt.figure(figsize=(14, 5.5))
    for d, klass in enumerate(['sec', 's2s']):
        for p, dataset in enumerate(['i_awa', 'r_awa', 'i_cub', 'r_cub']):
            rates = []
            k = len(result_files)
            curves = {klass + '_1l': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_2l': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_3l': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_4l': {'mean': np.zeros(k), 'std': np.zeros(k)}}

            for i, file in enumerate(result_files):
                if file.split('_')[-1].split('.')[0] == '2':
                    rates.append(10)
                else:
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

            plt.subplot(2, 4, d * 4 + p + 1)
            for key in curves.keys():
                plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key)

            if dataset in ['i_awa', 'r_awa']:
                plt.legend(loc='lower left')
            else:
                plt.legend(loc='upper right')

            plt.ylabel('Accuracy', weight='bold', size=9)
            plt.xlabel('Degradation Rate (%)', weight='bold', size=9)
            plt.title('SVM - %s' % dataset_names[dataset],  weight='bold', size=10)
            plt.tight_layout()

    plt.show()


plot_classification_results()
