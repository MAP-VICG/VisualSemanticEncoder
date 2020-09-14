import os
import json
import numpy as np
import matplotlib.pyplot as plt

# keys = ['acc', 'val_acc', 'loss', 'val_loss', 'svm_test', 'svm_val']
# data_types = ['i_awa', 'r_awa', 'i_cub', 'r_cub']
# svm_dict = {key: dict() for key in keys}
#
# for data_type in data_types:
#     for file in os.listdir('../../../Desktop/' + data_type):
#         if not file.endswith('.json'):
#             continue
#
#         with open('../../../Desktop/%s/%s' % (data_type, file)) as f:
#             data = json.load(f)
#
#         for key in svm_dict.keys():
#             svm_dict[key][file.split('_')[-1].split('.')[0].strip()] = data[key]
#
#     with open('../results/history/history_%s.json' % data_type, 'w+') as f:
#         json.dump(svm_dict, f, indent=4, sort_keys=True)


def plot_history(results, id_w, title):
    for data_type in results.keys():
        with open('../results/history/history_%s.json' % data_type) as f:
            data = json.load(f)

        for key in data.keys():
            values = []

            for fold in data[key].keys():
                values.append(data[key][fold])

            results[data_type][key] = dict()
            values = np.array(values)
            results[data_type][key]['mean'], results[data_type][key]['std'] = np.mean(values, axis=0), np.std(values, axis=0)

    plt.subplot(2, 3, id_w * 3 + 1)
    x = np.linspace(1, 50, 50)
    for dt in results.keys():
        plt.errorbar(x, results[dt]['loss']['mean'], yerr=results[dt]['loss']['std'], label='%s_trn_loss' % dt)
        plt.errorbar(x, results[dt]['val_loss']['mean'], yerr=results[dt]['val_loss']['std'], label='%s_val_loss' % dt)

    plt.title('%s Training Loss' % title, weight='bold', size=10)
    plt.ylabel('MSE', weight='bold', size=9)
    plt.xlabel('Epochs', weight='bold', size=9)
    plt.legend(loc='upper right', prop={'size': 9})
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.subplot(2, 3, id_w * 3 + 2)
    x = np.linspace(1, 50, 50)
    for dt in results.keys():
        plt.errorbar(x, results[dt]['acc']['mean'], yerr=results[dt]['acc']['std'], label='%s_trn_acc' % dt)
        plt.errorbar(x, results[dt]['val_acc']['mean'], yerr=results[dt]['val_acc']['std'], label='%s_val_acc' % dt)

    plt.title('%s Reconstruction Accuracy' % title, weight='bold', size=10)
    plt.ylabel('Accuracy', weight='bold', size=9)
    plt.xlabel('Epochs', weight='bold', size=9)
    plt.legend(loc='lower right', prop={'size': 9})

    plt.subplot(2, 3, id_w * 3 + 3)
    x = np.linspace(1, 50, 50)
    for dt in results.keys():
        plt.errorbar(x, results[dt]['svm_val']['mean'], yerr=results[dt]['svm_val']['std'], label='%s_val_svm' % dt)
        plt.errorbar(x, results[dt]['svm_test']['mean'], yerr=results[dt]['svm_test']['std'], label='%s_test_svm' % dt)

    plt.title('%s SVM Classification' % title, weight='bold', size=10)
    plt.ylabel('Accuracy', weight='bold', size=9)
    plt.xlabel('Epochs', weight='bold', size=9)
    plt.legend(loc='lower right', prop={'size': 9})


fig = plt.figure(figsize=(14, 7))
plot_history(results={'r_awa': dict(), 'i_awa': dict()}, id_w=0, title='AwA2')
plot_history(results={'r_cub': dict(), 'i_cub': dict()}, id_w=1, title='CUB200')

plt.tight_layout()
plt.show()
