import json
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(11, 6.5))
dataset_names = {'i_awa': 'AwA GoogleNet', 'r_awa': 'AwA2 ResNet', 'i_cub': 'CUB GoogleNet', 'r_cub': 'CUB ResNet'}
result_files = ['../results/classification_results_000.json', '../results/classification_results_010.json',
                '../results/classification_results_020.json', '../results/classification_results_030.json',
                '../results/classification_results_040.json', '../results/classification_results_050.json',
                '../results/classification_results_060.json', '../results/classification_results_070.json',
                '../results/classification_results_080.json', '../results/classification_results_090.json',
                '../results/classification_results_100.json']

for p, dataset in enumerate(['i_awa', 'r_awa', 'i_cub', 'r_cub']):
    rates = []
    k = len(result_files)
    curves = {'cat': {'mean': np.zeros(k), 'std': np.zeros(k)}, 's2s': {'mean': np.zeros(k), 'std': np.zeros(k)},
              'sae': {'mean': np.zeros(k), 'std': np.zeros(k)}, 'sec': {'mean': np.zeros(k), 'std': np.zeros(k)},
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
        curves['s2s']['mean'][i], curves['s2s']['std'][i] = np.mean(data[dataset]['s2s']), np.std(data[dataset]['s2s'])
        curves['sae']['mean'][i], curves['sae']['std'][i] = np.mean(data[dataset]['sae']), np.std(data[dataset]['sae'])
        curves['sec']['mean'][i], curves['sec']['std'][i] = np.mean(data[dataset]['sec']), np.std(data[dataset]['sec'])
        curves['sem']['mean'][i], curves['sem']['std'][i] = np.mean(data[dataset]['sem']), np.std(data[dataset]['sem'])
        curves['vis']['mean'][i], curves['vis']['std'][i] = np.mean(data[dataset]['vis']), np.std(data[dataset]['vis'])
        curves['pca']['mean'][i], curves['pca']['std'][i] = np.mean(data[dataset]['pca']), np.std(data[dataset]['pca'])

    plt.subplot(2, 2, p + 1)
    for key in curves.keys():
        if key == 'vis':
            plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key, ls='-.', color='#C0C0C0')
        elif key == 'sae':
            plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key, ls='--', color='#808080')
        else:
            plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key)

    if dataset in ['i_awa', 'r_awa']:
        plt.legend(loc='lower left')
    else:
        plt.legend(loc='upper right')

    plt.ylabel('Accuracy')
    plt.xlabel('Degradation Rate (%)')
    plt.title('SVM - %s' % dataset_names[dataset])
    plt.tight_layout()

plt.show()
