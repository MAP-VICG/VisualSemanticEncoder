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

    def plot_evaluation(self, x_test, y_test, label, ae_type):
        """
        Plots statistics related to the training of the autoencoder model including error and accuracy charts,
        covariance and confusion matrices, variation of encoding values and PCA distribution for chosen classes.

        @param x_test: 2D numpy array for test set
        @param y_test: 1D numpy array for labels
        @param label: plot name
        @param ae_type: type of AE
        @return: None
        """
        fig = plt.figure(figsize=(33, 18))
        plt.rcParams.update({'font.size': 12})
        plt.subplots_adjust(wspace=0.4, hspace=0.9)

        file_name = os.path.join(self.base_path, 'ae_evaluation_%s_%s.png' % (label, ae_type))
        encoder = Model(self.ae.model.input, outputs=[self.ae.model.get_layer('code').output])

        if ae_type == 'B0':
            encoding = encoder.predict(x_test[:, :2048])
        else:
            encoding = encoder.predict(x_test)

        ax = plt.subplot(2, 3, 1)
        ax.set_title('Classification Accuracy')

        n_epochs = len(self.ae.accuracies['test'])
        plt.plot([0.6151 for _ in range(n_epochs)], linestyle='dashed', linewidth=2)

        plt.plot([acc for acc in self.ae.accuracies['train']], linewidth=2)
        plt.plot([acc for acc in self.ae.accuracies['test']], linewidth=2)

        if ae_type in ('A1', 'A2'):
            plt.plot([acc for acc in self.ae.accuracies['vis test']], linewidth=2)
            plt.plot([acc for acc in self.ae.accuracies['sem test']], linewidth=2)

            if ae_type == 'A1':
                plt.plot([acc for acc in self.ae.accuracies['vs50 test']], linewidth=2)
                plt.legend(['baseline', 'train', 'test', 'vis test', 'sem test', 'vs50 test'], loc='upper left')
            else:
                plt.legend(['baseline', 'train', 'test', 'vis test', 'sem test'], loc='upper left')
        else:
            plt.legend(['baseline', 'train', 'test'], loc='upper left')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        ax = plt.subplot(2, 3, 2)
        ax.set_title('Training Loss')
        plt.plot(self.ae.history.history['loss'], linewidth=2)
        plt.plot(self.ae.history.history['val_loss'], linewidth=2)

        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend(['train', 'val'], loc='upper right')

        ax = plt.subplot(2, 3, 3)
        ax.set_title('Reconstruction Accuracy')
        plt.plot(self.ae.history.history['acc'], linewidth=2)
        plt.plot(self.ae.history.history['val_acc'], linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'val'], loc='upper left')

        ax = plt.subplot(2, 3, 4)
        ax.set_title('Encoding Variation')
        zeros = [1 for value in np.mean(encoding, axis=0) if abs(value) <= 0.05]
        plt.plot([0.05 for _ in range(encoding.shape[1])], linestyle='dashed', color='k')
        plt.plot([-0.05 for _ in range(encoding.shape[1])], linestyle='dashed', color='k')
        plt.errorbar([x for x in range(encoding.shape[1])], encoding[0], encoding[1], fmt='.')

        plt.xlabel('Encoding Dimension')
        plt.ylabel('Amplitude')
        plt.legend(['%d/%d zeros' % (len(zeros), encoding.shape[1])])

        ax = plt.subplot(2, 3, 5)
        ax.set_title('Covariance Matrix')
        code_cov = np.cov(np.array(encoding).transpose())
        ax = plt.matshow(code_cov, fignum=False, cmap='plasma')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        ax.axes.get_xaxis().set_visible(False)

        ax = plt.subplot(2, 3, 6)
        ax.set_title('Confusion Matrix')
        plot_confusion_matrix(self.ae.svm_model, encoding, y_test, cmap=plt.cm.Blues, normalize='true',
                              include_values=False, ax=ax)

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        plt.tight_layout()
        plt.savefig(file_name)
        plt.close(fig)
