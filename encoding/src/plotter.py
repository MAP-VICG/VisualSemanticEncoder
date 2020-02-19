import os
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from sklearn.metrics import plot_confusion_matrix

matplotlib.use('Agg')


class Plotter:
    def __init__(self, encoder, base_path, chosen_classes, classes_names):
        """
        Initialize main variables.

        @param encoder: encoder model
        @param base_path: string with base path to save allresults
        @param chosen_classes: array of numbers with id of classes to plot PCA
        @param classes_names: array of strings with classes names to plot PCA
        """
        self.base_path = base_path
        self.encoder = encoder

        self.chosen_classes = chosen_classes
        self.classes_names = classes_names
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']

    def plot_evaluation(self, x_test, y_test, baseline=0):
        """
        Plots statistics related to the training allresults of the autoencoder model including error and accuracy
        charts, covariance and confusion matrices, variation of encoding values and PCA distribution for chosen classes.

        @param x_test: 2D numpy array for test set
        @param y_test: 1D numpy array for labels
        @param baseline: value that accuracy must reach
        @return: None
        """
        fig = plt.figure(figsize=(20, 12))
        plt.rcParams.update({'font.size': 12})
        plt.subplots_adjust(wspace=0.4, hspace=0.9)

        file_name = os.path.join(self.base_path, 'ae_evaluation.png')
        encoder = Model(self.encoder.autoencoder.input, outputs=[self.encoder.autoencoder.get_layer('code').output])
        encoding = encoder.predict(x_test)

        ax = plt.subplot(231)
        ax.set_title('Classification Accuracy')
        plt.plot([acc for acc in self.encoder.accuracies['train']])
        plt.plot([acc for acc in self.encoder.accuracies['test']])

        if baseline != 0:
            nepochs = len(self.encoder.accuracies['test'])
            plt.plot([baseline for _ in range(nepochs)], linestyle='dashed', linewidth=2, color='k')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'test', 'baseline'])

        ax = plt.subplot(232)
        ax.set_title('Training Loss')
        plt.plot(self.encoder.history.history['loss'])
        plt.plot(self.encoder.history.history['val_loss'])

        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend(['train', 'val'])

        ax = plt.subplot(233)
        zeros = [1 for value in np.mean(encoding, axis=0) if abs(value) <= 0.05]
        plt.plot([0.05 for _ in range(encoding.shape[1])], linestyle='dashed', color='k')
        plt.plot([-0.05 for _ in range(encoding.shape[1])], linestyle='dashed', color='k')
        plt.errorbar([x for x in range(encoding.shape[1])], encoding[0], encoding[1], fmt='.')

        ax.set_title('Encoding Variation')
        plt.xlabel('Encoding Dimension')
        plt.ylabel('Amplitude')
        plt.legend(['%d/%d zeros' % (len(zeros), encoding.shape[1])])

        ax = plt.subplot(234)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Encoding')
        mask = self._define_mask(y_test)
        encoding_fts = pca.fit_transform(encoding[mask, :])
        self._scatter_plot(ax, encoding_fts, y_test[mask])

        ax = plt.subplot(235)
        ax.set_title('Covariance Matrix')
        code_cov = np.cov(np.array(encoding).transpose())
        ax = plt.matshow(code_cov, fignum=False, cmap='plasma')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        ax.axes.get_xaxis().set_visible(False)

        ax = plt.subplot(236)
        ax.set_title('Confusion Matrix')
        plot_confusion_matrix(self.encoder.svm_model, encoding, y_test, cmap=plt.cm.Blues, normalize='true',
                              include_values=False, ax=ax)

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        plt.tight_layout()
        plt.savefig(file_name)
        plt.close(fig)

    def _define_mask(self, classes):
        """
        Builds boolean mask to filter chosen classes

        @param classes: array of classes
        @return: boolean array to mask classes array
        """
        mask = [False] * len(classes)
        for i in range(len(classes)):
            if classes[i] in self.chosen_classes:
                mask[i] = True
        return mask

    def _scatter_plot(self, ax, features, labels):
        """
        Plots PCA distribution in a scatter plot indicating classes labels

        @param ax: pyplot axes
        @param features: 2D numpy array with features to plot
        @param labels: list of labels
        @return: None
        """
        for k, label in enumerate(self.chosen_classes):
            plot_mask = [False] * len(labels)
            for i in range(len(labels)):
                if labels[i] == label:
                    plot_mask[i] = True

            plt.scatter(features[plot_mask, 0], features[plot_mask, 1], c=self.colors[k],
                        # s=np.ones(self.labels[plot_mask].shape)*5,
                        label=self.classes_names[k])
        ax.legend(prop={'size': 10})
