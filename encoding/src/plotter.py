import os
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')


class Plotter:
    def __init__(self, base_path):
        self.base_path = base_path

    def plot_evaluation(self, history, accuracy, confusion_matrix, baseline=0):
        fig = plt.figure(figsize=(18, 6))
        plt.rcParams.update({'font.size': 12})
        plt.subplots_adjust(wspace=0.4, hspace=0.9)

        file_name = os.path.join(self.base_path, 'ae_evaluation.png')

        ax = plt.subplot(131)
        ax.set_title('Classification Accuracy')
        plt.plot([acc for acc in accuracy['train']])
        plt.plot([acc for acc in accuracy['test']])

        if baseline != 0:
            plt.plot([baseline for _ in range(len(accuracy['test']))], linestyle='dashed', linewidth=2, color='k')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'test', 'baseline'])

        ax = plt.subplot(132)
        ax.set_title('Training Loss')
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])

        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend(['train', 'val'])

        ax = plt.subplot(133)
        ax.set_title('Confusion Matrix')
        ax = plt.matshow(confusion_matrix, fignum=False, cmap=plt.cm.Blues)
        ax.axes.get_xaxis().set_visible(False)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=10)

        plt.tight_layout()
        plt.savefig(file_name)
        plt.close(fig)
