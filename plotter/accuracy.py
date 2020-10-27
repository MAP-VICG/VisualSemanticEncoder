import json
import numpy as np
from os import listdir, path
import matplotlib.pyplot as plt

algorithms = ['cat', 'iso', 'lle', 'pca', 's2s', 'sae', 'sem', 'vis', 'vse']
datasets = ['i_apy', 'i_awa', 'i_cub', 'i_sun', 'r_apy', 'r_awa', 'r_cub', 'r_sun']
curves = {dt: {ag: {'mean': [], 'std': []} for ag in algorithms} for dt in datasets}

for file in sorted(listdir('../results/accuracy/')):
    with open(path.join('../results/accuracy/', file)) as f:
        data = json.load(f)

    for key in data.keys():
        for sub_key in algorithms:
            if data[key].get(sub_key) is None:
                curves[key][sub_key]['mean'].append(-1)
                curves[key][sub_key]['std'].append(0)
            else:
                curves[key][sub_key]['mean'].append(np.mean(np.array(data[key][sub_key])))
                curves[key][sub_key]['std'].append(np.std(np.array(data[key][sub_key])))

ax = None
x = np.linspace(0, 10, 11) * 10
fig = plt.figure(figsize=(14, 6.5))

for i, ds in enumerate(datasets):
    if ds.startswith('i'):
        ax = fig.add_subplot(2, 4, i + 1)
        for key in sorted(list(curves[ds].keys())):
            if key == 'vis':
                plt.errorbar(x, curves[ds][key]['mean'], yerr=curves[ds][key]['std'], label=key, ls='-.', color='#C0C0C0')
            elif key == 'sae':
                plt.errorbar(x, curves[ds][key]['mean'], yerr=curves[ds][key]['std'], label=key, ls='--', color='#808080')
            else:
                plt.errorbar(x, curves[ds][key]['mean'], yerr=curves[ds][key]['std'], label=key)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - 0.1, box.width, box.height])

        plt.xlabel('Degradation Level (%)', weight='bold', size=9)
        plt.ylabel('Balanced Accuracy Score', weight='bold', size=9)
        plt.title('SVM Accuracy %s' % ds.upper(), weight='bold', size=10)
        plt.tight_layout()
    else:
        ax = fig.add_subplot(2, 4, i + 1)
        for key in sorted(list(curves[ds].keys())):
            if key == 'vis':
                plt.errorbar(x, curves[ds][key]['mean'], yerr=curves[ds][key]['std'], label=key, ls='-.', color='#C0C0C0')
            elif key == 'sae':
                plt.errorbar(x, curves[ds][key]['mean'], yerr=curves[ds][key]['std'], label=key, ls='--', color='#808080')
            else:
                plt.errorbar(x, curves[ds][key]['mean'], yerr=curves[ds][key]['std'], label=key)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - 0.1, box.width, box.height])

        plt.xlabel('Degradation Level (%)', weight='bold', size=9)
        plt.ylabel('Balanced Accuracy Score', weight='bold', size=9)
        plt.title('SVM Accuracy %s' % ds.upper(), weight='bold', size=10)
        plt.tight_layout()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=9, bbox_to_anchor=(0.5, 0.05))

plt.show()
