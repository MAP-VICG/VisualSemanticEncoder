import json
import numpy as np
import matplotlib.pyplot as plt


def get_loss_details(file, ae_type):
    with open(file) as f:
        data = json.load(f)

        loss = []
        val_loss = []
        for key in data[ae_type].keys():
            loss.append(data[ae_type][key]['loss'])
            val_loss.append(data[ae_type][key]['val_loss'])

        loss = np.array(loss)
        loss_mean, loss_std = np.mean(loss, axis=0), np.std(loss, axis=0)

        val_loss = np.array(val_loss)
        val_loss_mean, val_loss_std = np.mean(val_loss, axis=0), np.std(val_loss, axis=0)

    return loss_mean, loss_std, val_loss_mean, val_loss_std


def plot_loss(file, title, y_label):
    x_values = np.linspace(1, 50, 50)

    loss_mean, loss_std, val_loss_mean, val_loss_std = get_loss_details(file, 'sec')
    plt.errorbar(x_values, loss_mean, yerr=loss_std, label='loss_sec')
    plt.errorbar(x_values, val_loss_mean, yerr=val_loss_std, label='val_loss_sec')

    loss_mean, loss_std, val_loss_mean, val_loss_std = get_loss_details(file, 's2s')
    plt.errorbar(x_values, loss_mean, yerr=loss_std, label='loss_s2s')
    plt.errorbar(x_values, val_loss_mean, yerr=val_loss_std, label='val_loss_s2s')

    plt.ylabel(y_label, weight='bold', size=9)
    plt.xlabel('Epochs', weight='bold', size=9)
    plt.title(title, weight='bold', size=10)
    plt.legend(loc='upper right',  prop={'size': 8})
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()


plt.figure(figsize=(14, 5.5))

plt.subplot(2, 4, 1)
plot_loss('../results/losses/v1_loss_i_cub.json', 'V1 - CUB200 GN', 'MSE')
plt.subplot(2, 4, 2)
plot_loss('../results/losses/v1_loss_r_cub.json', 'V1 - CUB200 RN', 'MSE')
plt.subplot(2, 4, 3)
plot_loss('../results/losses/v1_loss_i_awa.json', 'V1 - AwA GN', 'MSE')
plt.subplot(2, 4, 4)
plot_loss('../results/losses/v1_loss_r_awa.json', 'V1 - AwA2 RN', 'MSE')

plt.subplot(2, 4, 5)
plot_loss('../results/losses/v2_loss_i_cub.json', 'V2 - CUB200 GN', 'MSE(vis) + L * MSE(sem)')
plt.subplot(2, 4, 6)
plot_loss('../results/losses/v2_loss_r_cub.json', 'V2 - CUB200 RN', 'MSE(vis) + L * MSE(sem)')
plt.subplot(2, 4, 7)
plot_loss('../results/losses/v2_loss_i_awa.json', 'V2 - AwA GN', 'MSE(vis) + L * MSE(sem)')
plt.subplot(2, 4, 8)
plot_loss('../results/losses/v2_loss_r_awa.json', 'V2 - AwA2 RN', 'MSE(vis) + L * MSE(sem)')

plt.show()
