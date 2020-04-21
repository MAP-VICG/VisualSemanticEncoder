import os
import matplotlib
import numpy as np
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix

matplotlib.use('Agg')


class Plotter:
    def __init__(self, ae, base_path):
        """
        Initialize main variables.

        @param ae: ae model
        @param base_path: string with base path to save results
        """
        self.ae = ae
        self.base_path = base_path

    def plot_evaluation(self, baseline):
        """
        Plots statistics related to the training of the autoencoder model including error and accuracy charts,
        covariance and confusion matrices, variation of encoding values and PCA distribution for chosen classes.

        @param baseline: float number to be reference in plot
        @return: None
        """
        fig = plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 12})
        plt.subplots_adjust(wspace=0.4, hspace=0.9)

        file_name = os.path.join(self.base_path, 'ae_evaluation.png')

        ax = plt.subplot(1, 2, 1)
        ax.set_title('Classification Accuracy')

        n_epochs = len(self.ae.zsl_accuracies)
        plt.plot([baseline for _ in range(n_epochs)], linestyle='dashed', linewidth=2, color='k')

        plt.plot([acc for acc in self.ae.zsl_accuracies], linewidth=2)
        plt.legend(['baseline', 'test'], loc='upper left')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        ax = plt.subplot(1, 2, 2)
        ax.set_title('Training Loss')
        plt.plot(self.ae.history.history['loss'], linewidth=2)
        plt.plot(self.ae.history.history['val_loss'], linewidth=2)

        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend(['train', 'val'], loc='upper right')

        plt.tight_layout()
        plt.savefig(file_name)
        plt.close(fig)
