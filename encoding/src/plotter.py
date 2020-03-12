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
    def __init__(self, ae, base_path, chosen_classes, classes_names):
        """
        Initialize main variables.

        @param ae: ae model
        @param base_path: string with base path to save oldresults
        @param chosen_classes: array of numbers with id of classes to plot PCA
        @param classes_names: array of strings with classes names to plot PCA
        """
        self.ae = ae
        self.base_path = base_path
        self.chosen_classes = chosen_classes
        self.classes_names = classes_names
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']

    def plot_evaluation(self, x_test, y_test, baseline=None):
        """
        Plots statistics related to the training of the autoencoder model including error and accuracy charts,
        covariance and confusion matrices, variation of encoding values and PCA distribution for chosen classes.

        @param x_test: 2D numpy array for test set
        @param y_test: 1D numpy array for labels
        @param baseline: value that accuracy must reach
        @return: None
        """
        x_test = x_test[:, :2048]
        fig = plt.figure(figsize=(33, 18))
        plt.rcParams.update({'font.size': 12})
        plt.subplots_adjust(wspace=0.4, hspace=0.9)

        legend = []
        mask = self._define_mask(y_test)
        file_name = os.path.join(self.base_path, 'ae_evaluation.png')
        encoder = Model(self.ae.autoencoder.input, outputs=[self.ae.autoencoder.get_layer('code').output])

        ax = plt.subplot(3, 5, 2)
        ax.set_title('Training Loss')
        plt.plot(self.ae.history.history['loss'], linewidth=2)
        plt.plot(self.ae.history.history['val_loss'], linewidth=2)

        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend(['train', 'val'])

        input_vs, encoding_vs, output_vs, baseline['iboth'] = self._get_pca_plot_inputs(x_test, y_test, encoder, 'VS')
        self._plot_pca_grid(input_vs, encoding_vs, output_vs, y_test, mask, (3, 5, 3), 'Vis-Sem')

        ax = plt.subplot(3, 5, 6)
        ax.set_title('Reconstruction Accuracy')
        try:
            plt.plot(self.ae.history.history['acc'], linewidth=2)
            plt.plot(self.ae.history.history['val_acc'], linewidth=2)
        except KeyError:
            pass

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'val'])

        ax = plt.subplot(3, 5, 7)
        ax.set_title('Encoding Variation')
        zeros = [1 for value in np.mean(encoding_vs, axis=0) if abs(value) <= 0.05]
        plt.plot([0.05 for _ in range(encoding_vs.shape[1])], linestyle='dashed', color='k')
        plt.plot([-0.05 for _ in range(encoding_vs.shape[1])], linestyle='dashed', color='k')
        plt.errorbar([x for x in range(encoding_vs.shape[1])], encoding_vs[0], encoding_vs[1], fmt='.')

        plt.xlabel('Encoding Dimension')
        plt.ylabel('Amplitude')
        plt.legend(['%d/%d zeros' % (len(zeros), encoding_vs.shape[1])])

        input_v, encoding_v, output_v, baseline['ivis'] = self._get_pca_plot_inputs(x_test, y_test, encoder, 'V')
        self._plot_pca_grid(input_v, encoding_v, output_v, y_test, mask, (3, 5, 8), 'Visual')

        ax = plt.subplot(3, 5, 11)
        ax.set_title('Covariance Matrix')
        code_cov = np.cov(np.array(encoding_vs).transpose())
        ax = plt.matshow(code_cov, fignum=False, cmap='plasma')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        ax.axes.get_xaxis().set_visible(False)

        ax = plt.subplot(3, 5, 12)
        ax.set_title('Confusion Matrix')
        plot_confusion_matrix(self.ae.svm_model, encoding_vs, y_test, cmap=plt.cm.Blues, normalize='true',
                              include_values=False, ax=ax)

        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        input_s, encoding_s, output_s, baseline['isem'] = self._get_pca_plot_inputs(x_test, y_test, encoder, 'S')
        self._plot_pca_grid(input_s, encoding_s, output_s, y_test, mask, (3, 5, 13), 'Semantic')

        ax = plt.subplot(3, 5, 1)
        ax.set_title('Classification Accuracy')
        if baseline and isinstance(baseline, dict):
            n_epochs = len(self.ae.accuracies['test'])

            for i, key in enumerate(baseline.keys()):
                if baseline[key] != 0:
                    plt.plot([baseline[key] for _ in range(n_epochs)], linestyle='dashed', linewidth=2)
                    legend.append(key)

        plt.plot([acc for acc in self.ae.accuracies['train']], linewidth=2)
        plt.plot([acc for acc in self.ae.accuracies['test']], linewidth=2)
        legend.extend(['train', 'test'])

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(legend)

        plt.tight_layout()
        plt.savefig(file_name)
        plt.close(fig)

    def _get_pca_plot_inputs(self, x_test, y_test, encoder, type_):
        """
        Retrieves the input, encoding, output and and classification accuracy results depending
        on the type of investigation set

        @param x_test: 2D numpy array for test set
        @param y_test: 1D numpy array for labels
        @param encoder: encoder model (first half of the AE)
        @param type_: type of analysis: VS, V or S
        @return: tuple with input, encoding and output arrays plus float with classification accuracy
        """
        if type_ == 'VS':
            input_ = x_test
        elif type_ == 'V':
            input_ = np.copy(x_test)
            input_[:, 2048:] = input_[:, 2048:] * 0
        elif type_ == 'S':
            input_ = np.copy(x_test)
            input_[:, :2048] = input_[:, :2048] * 0
        else:
            raise ValueError('Unknown Type')

        encoding_ = encoder.predict(input_)
        output_ = self.ae.autoencoder.predict(input_)
        acc = balanced_accuracy_score(self.ae.svm_model.predict(encoding_), y_test)

        return input_, encoding_, output_, acc

    def _plot_pca_grid(self, input_, encoding_, output_, labels, mask, grid_pos, name):
        """
        Plots PCA result in a grid for input, encoding and output data

        @param input_: 2D numpy array for input data
        @param encoding_: 2D numpy array encoding data
        @param output_: 2D numpy array output data
        @param labels: 1D numpy array with set labels
        @param mask: boolean array where True indicates the examples to be used
        @param grid_pos: grid position of the first subplot
        @param name: string with identification of the plot
        @return: None
        """
        try:
            ax = plt.subplot(grid_pos[0], grid_pos[1], grid_pos[2])
            pca = PCA(n_components=2)
            ax.set_title('PCA %s - Input' % name)
            input_fts = pca.fit_transform(input_[mask, :])
            self._scatter_plot(ax, input_fts, labels[mask])

            ax = plt.subplot(grid_pos[0], grid_pos[1], grid_pos[2] + 1)
            pca = PCA(n_components=2)
            ax.set_title('PCA %s - Encoding' % name)
            encoding_fts = pca.fit_transform(encoding_[mask, :])
            self._scatter_plot(ax, encoding_fts, labels[mask])

            ax = plt.subplot(grid_pos[0], grid_pos[1], grid_pos[2] + 2)
            pca = PCA(n_components=2)
            ax.set_title('PCA %s - Output' % name)
            encoding_fts = pca.fit_transform(output_[mask, :])
            self._scatter_plot(ax, encoding_fts, labels[mask])
        except ValueError:
            pass

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
