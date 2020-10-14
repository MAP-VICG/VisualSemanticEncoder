import os
import json
import numpy as np
import matplotlib.pyplot as plt

dataset_names = {'i_awa': 'AwA GoogleNet', 'r_awa': 'AwA2 ResNet', 'i_cub': 'CUB GoogleNet', 'r_cub': 'CUB ResNet'}
result_files = sorted(['../results/dropout/' + file for file in os.listdir('../results/dropout') if file.endswith('.json')])


def plot_classification_results():
    ax = None
    fig = plt.figure(figsize=(14, 6.5))
    for d, klass in enumerate(['vse', 's2s']):
        for p, dataset in enumerate(['i_awa', 'r_awa', 'i_cub', 'r_cub']):
            rates = []
            k = len(result_files)
            curves = {klass + '_dp00': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_dpb1': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_dpa1': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_dba1': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_dpb2': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_dpa2': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_dba2': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_dpb3': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_dpa3': {'mean': np.zeros(k), 'std': np.zeros(k)},
                      klass + '_dba3': {'mean': np.zeros(k), 'std': np.zeros(k)}
                      }

            for i, file in enumerate(sorted(result_files)):
                rates.append(int(file.split('_')[-1].split('.')[0]))

                with open(file) as f:
                    data = json.load(f)

                curves[klass + '_dp00']['mean'][i] = np.mean(data[dataset][klass + '_dp00'])
                curves[klass + '_dpb1']['mean'][i] = np.mean(data[dataset][klass + '_dpb1'])
                curves[klass + '_dpa1']['mean'][i] = np.mean(data[dataset][klass + '_dpa1'])
                curves[klass + '_dba1']['mean'][i] = np.mean(data[dataset][klass + '_dba1'])
                curves[klass + '_dpb2']['mean'][i] = np.mean(data[dataset][klass + '_dpb2'])
                curves[klass + '_dpa2']['mean'][i] = np.mean(data[dataset][klass + '_dpa2'])
                curves[klass + '_dba2']['mean'][i] = np.mean(data[dataset][klass + '_dba2'])
                curves[klass + '_dpb3']['mean'][i] = np.mean(data[dataset][klass + '_dpb3'])
                curves[klass + '_dpa3']['mean'][i] = np.mean(data[dataset][klass + '_dpa3'])
                curves[klass + '_dba3']['mean'][i] = np.mean(data[dataset][klass + '_dba3'])

                curves[klass + '_dp00']['std'][i] = np.std(data[dataset][klass + '_dp00'])
                curves[klass + '_dpb1']['std'][i] = np.std(data[dataset][klass + '_dpb1'])
                curves[klass + '_dpa1']['std'][i] = np.std(data[dataset][klass + '_dpa1'])
                curves[klass + '_dba1']['std'][i] = np.std(data[dataset][klass + '_dba1'])
                curves[klass + '_dpb2']['std'][i] = np.std(data[dataset][klass + '_dpb2'])
                curves[klass + '_dpa2']['std'][i] = np.std(data[dataset][klass + '_dpa2'])
                curves[klass + '_dba2']['std'][i] = np.std(data[dataset][klass + '_dba2'])
                curves[klass + '_dpb3']['std'][i] = np.std(data[dataset][klass + '_dpb3'])
                curves[klass + '_dpa3']['std'][i] = np.std(data[dataset][klass + '_dpa3'])
                curves[klass + '_dba3']['std'][i] = np.std(data[dataset][klass + '_dba3'])

            ax = fig.add_subplot(2, 4, d * 4 + p + 1)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 - 0.1, box.width * 0.9, box.height * 0.9])

            for key in curves.keys():
                plt.errorbar(rates, curves[key]['mean'], yerr=curves[key]['std'], label=key.split('_')[-1].upper())

            plt.ylabel('Accuracy', weight='bold', size=9)
            plt.xlabel('Degradation Rate (%)', weight='bold', size=9)
            plt.title('%s - %s' % (klass.upper(), dataset_names[dataset]),  weight='bold', size=10)
            plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=10, bbox_to_anchor=(0.5, 0.05))
    plt.show()


plot_classification_results()
