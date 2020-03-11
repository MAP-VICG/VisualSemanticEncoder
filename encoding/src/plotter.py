import os
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import balanced_accuracy_score

matplotlib.use('Agg')


class Plotter:
    def __init__(self, encoder, base_path, chosen_classes, classes_names):
        """
        Initialize main variables.

        @param encoder: encoder model
        @param base_path: string with base path to save oldresults
        @param chosen_classes: array of numbers with id of classes to plot PCA
        @param classes_names: array of strings with classes names to plot PCA
        """
        self.base_path = base_path
        self.encoder = encoder

        self.chosen_classes = chosen_classes
        self.classes_names = classes_names
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']

    def plot_evaluation(self, x_test, y_test, output_length=2048, baseline=None):
        """
        Plots statistics related to the training of the autoencoder model including error and accuracy charts,
        covariance and confusion matrices, variation of encoding values and PCA distribution for chosen classes.

        @param x_test: 2D numpy array for test set
        @param y_test: 1D numpy array for labels
        @param output_length: AE output length
        @param baseline: value that accuracy must reach
        @return: None
        """
        fig = plt.figure(figsize=(33, 18))
        plt.rcParams.update({'font.size': 12})
        plt.subplots_adjust(wspace=0.4, hspace=0.9)

        file_name = os.path.join(self.base_path, 'ae_evaluation.png')
        legend = []

        encoder = Model(self.encoder.autoencoder.input, outputs=[self.encoder.autoencoder.get_layer('code').output])
        encoding = encoder.predict(x_test)
        output = self.encoder.autoencoder.predict(x_test)
        mask = self._define_mask(y_test)

        ax = plt.subplot(3, 5, 2)
        ax.set_title('Training Loss')
        plt.plot(self.encoder.history.history['loss'])
        plt.plot(self.encoder.history.history['val_loss'])

        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend(['train', 'val'])

        ax = plt.subplot(3, 5, 3)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Input VS')
        input_fts = pca.fit_transform(x_test[mask, :])
        self._scatter_plot(ax, input_fts, y_test[mask])

        ax = plt.subplot(3, 5, 4)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Encoding VS')
        encoding_fts = pca.fit_transform(encoding[mask, :])
        self._scatter_plot(ax, encoding_fts, y_test[mask])

        ax = plt.subplot(3, 5, 5)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Output VS')
        encoding_fts = pca.fit_transform(output[mask, :])
        self._scatter_plot(ax, encoding_fts, y_test[mask])

        ax = plt.subplot(3, 5, 6)
        ax.set_title('Reconstruction Accuracy')
        plt.plot(self.encoder.history.history['acc'])
        plt.plot(self.encoder.history.history['val_acc'])

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'val'])

        ax = plt.subplot(3, 5, 7)
        ax.set_title('Encoding Variation')
        zeros = [1 for value in np.mean(encoding, axis=0) if abs(value) <= 0.05]
        plt.plot([0.05 for _ in range(encoding.shape[1])], linestyle='dashed', color='k')
        plt.plot([-0.05 for _ in range(encoding.shape[1])], linestyle='dashed', color='k')
        plt.errorbar([x for x in range(encoding.shape[1])], encoding[0], encoding[1], fmt='.')

        plt.xlabel('Encoding Dimension')
        plt.ylabel('Amplitude')
        plt.legend(['%d/%d zeros' % (len(zeros), encoding.shape[1])])

        ax = plt.subplot(3, 5, 8)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Input V')
        input_fts = pca.fit_transform(x_test[mask, :2048])
        self._scatter_plot(ax, input_fts, y_test[mask])

        x_test_v = np.copy(x_test)
        x_test_v[:, 2048:] = x_test_v[:, 2048:] * 0
        encoding_v = encoder.predict(x_test_v)
        output_v = self.encoder.autoencoder.predict(x_test_v)

        prediction = self.encoder.svm_model.predict(encoding_v)
        acc_v = balanced_accuracy_score(prediction, y_test)

        ax = plt.subplot(3, 5, 9)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Encoding V')
        encoding_fts = pca.fit_transform(encoding_v[mask, :])
        self._scatter_plot(ax, encoding_fts, y_test[mask])

        ax = plt.subplot(3, 5, 10)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Output V')
        encoding_fts = pca.fit_transform(output_v[mask, :2048])
        self._scatter_plot(ax, encoding_fts, y_test[mask])

        ax = plt.subplot(3, 5, 11)
        ax.set_title('Covariance Matrix')
        code_cov = np.cov(np.array(encoding).transpose())
        ax = plt.matshow(code_cov, fignum=False, cmap='plasma')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        ax.axes.get_xaxis().set_visible(False)

        ax = plt.subplot(3, 5, 12)
        ax.set_title('Confusion Matrix')
        plot_confusion_matrix(self.encoder.svm_model, encoding, y_test, cmap=plt.cm.Blues, normalize='true',
                              include_values=False, ax=ax)

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        ax = plt.subplot(3, 5, 13)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Input S')
        input_fts = pca.fit_transform(x_test[mask, 2048:])
        self._scatter_plot(ax, input_fts, y_test[mask])

        x_test_s = np.copy(x_test)
        x_test_s[:, :2048] = x_test_s[:, :2048] * 0
        encoding_s = encoder.predict(x_test_s)
        output_s = self.encoder.autoencoder.predict(x_test_s)

        prediction = self.encoder.svm_model.predict(encoding_s)
        acc_s = balanced_accuracy_score(prediction, y_test)

        ax = plt.subplot(3, 5, 14)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Encoding S')
        encoding_fts = pca.fit_transform(encoding_s[mask, :])
        self._scatter_plot(ax, encoding_fts, y_test[mask])

        ax = plt.subplot(3, 5, 15)
        pca = PCA(n_components=2)
        ax.set_title('PCA - Output S')
        encoding_fts = pca.fit_transform(output_s[mask, 2048:])
        self._scatter_plot(ax, encoding_fts, y_test[mask])

        ax = plt.subplot(3, 5, 1)
        ax.set_title('Classification Accuracy')
        if baseline and isinstance(baseline, dict):
            n_epochs = len(self.encoder.accuracies['test'])

            plt.plot([baseline['vis'] for _ in range(n_epochs)], linestyle='dashed', linewidth=1, color='g')
            legend.append('vis')

            plt.plot([acc_v for _ in range(n_epochs)], linestyle='dashed', linewidth=1, color='r')
            legend.append('ae_v')

            plt.plot([acc_s for _ in range(n_epochs)], linestyle='dashed', linewidth=1, color='k')
            legend.append('ae_s')

        plt.plot([acc for acc in self.encoder.accuracies['train']], linewidth=2)
        plt.plot([acc for acc in self.encoder.accuracies['test']], linewidth=2)
        legend.extend(['train', 'test'])

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(legend)

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
