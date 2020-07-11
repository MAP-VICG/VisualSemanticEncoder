import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy(path, files):
    for file_name in files:
        with open(os.path.join(path, file_name)) as f:
            data = json.loads(json.load(f))

        rates = sorted([float(key) * 100 for key in data.keys() if key != 'ref'])
        label = 'r_' + file_name.split('_')[1] if file_name.startswith('resnet') else 'i_' + file_name.split('_')[1]

        mean_ref = [np.mean(np.array(data['ref']))] * len(rates)
        std_ref = [np.std(np.array(data['ref']))] * len(rates)

        mean = [float(data[str(rate / 100) if rate > 0 else '0']['mean']) for rate in rates]
        std = [float(data[str(rate / 100) if rate > 0 else '0']['std']) for rate in rates]

        color = '#C0C0C0' if label.startswith('r') else '#808080'
        style = '-.' if label.startswith('r') else '--'

        plt.errorbar(rates, mean_ref, yerr=std_ref, label='%s_sae' % label, ls=style, color=color)
        plt.errorbar(rates, mean, yerr=std, label='%s_sec' % label)


results_path = 'data/'
result_files = [file for file in os.listdir(results_path) if file.endswith('.json')]

fig = plt.figure(figsize=(13, 6.5))

plt.subplot(221)
plot_accuracy(results_path, [file for file in result_files if 'awa_svm' in file])
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('SVM Classification - AwA2')

handles, labels = plt.gca().get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].split('_')[-1] + t[0].split('_')[-2]))
plt.legend(handles, labels, loc='upper right')
plt.tight_layout()

plt.subplot(222)
plot_accuracy(results_path, [file for file in result_files if 'awa_zsl' in file])
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('ZSL Classification - AwA2 ')

handles, labels = plt.gca().get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].split('_')[-1] + t[0].split('_')[-2]))
plt.legend(handles, labels, loc='upper right')
plt.tight_layout()

plt.subplot(223)
plot_accuracy(results_path, [file for file in result_files if 'cub_svm' in file])
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('SVM Classification - CUB200')

handles, labels = plt.gca().get_legend_handles_labels()
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].split('_')[-1] + t[0].split('_')[-2]))
plt.legend(handles, labels, loc='upper right')
plt.tight_layout()

plt.subplot(224)
plot_accuracy(results_path, [file for file in result_files if 'cub_zsl' in file])
plt.ylabel('Accuracy')
plt.xlabel('Degradation Rate (%)')
plt.title('ZSL Classification - CUB200')

handles, labels = plt.gca().get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].split('_')[-1] + t[0].split('_')[-2]))
plt.legend(handles, labels, loc='upper right')
plt.tight_layout()

plt.show()
