'''
Autoencoder for visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 23, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''
import io
import os
import sys
import numpy as np
from random import randint
from matplotlib import pyplot as plt

from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.layers import Input, Dense, BatchNormalization, Concatenate, Flatten, Conv1D, Add

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from core.vsclassifier import SVMClassifier
from core.featuresparser import FeaturesParser
from utils.logwriter import Logger, MessageType


class VSAutoencoderSingleInput:
    
    def __init__(self, cv=5, njobs=1, **kwargs):
        '''
        Initialize all models and runs grid search on SVM
        
        @param kwargs: dictionary with training and testing data
        @param cv: grid search number of folds in cross validation
        @param njobs: grid search number of jobs to run in parallel
        '''
        self.svm = None
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.svm_history = []
    
        self.x_train = kwargs.get('x_train')
        self.y_train = kwargs.get('y_train')
        self.x_test = kwargs.get('x_test')
        self.y_test = kwargs.get('y_test')

        self.svm = SVMClassifier()
        self.svm.run_classifier(self.x_train, self.y_train, cv, njobs)
        
    def run_autoencoder(self, enc_dim, nepochs, results_path):
        '''
        Builds and trains autoencoder model
        
        @param enc_dim: encoding size
        @param nepochs: number of epochs
        @param results_path: string with path to svm save results under
        @return object with training details and history
        '''
        def svm_callback(epoch, logs):
            '''
            Runs SVM and saves prediction results
            
            @param epoch: default callback parameter. Epoch index
            @param logs:default callback parameter. Loss result
            '''
            old_stderr = sys.stderr
            sys.stderr = buffer = io.StringIO()
            
#             self.svm.model.fit(self.encoder.predict([self.x_train]), self.y_train)
            self.svm.model.best_estimator_.fit(self.encoder.predict([self.x_train]), self.y_train)
            
            sys.stderr = old_stderr
            Logger().write_message('Epoch %d SVM\n%s' % (epoch, buffer.getvalue()), MessageType.ERR)
            
            pred_dict, prediction = self.svm.predict(self.encoder.predict([self.x_test]), self.y_test)
            self.svm_history.append(pred_dict)
            
            if epoch == nepochs - 1:
                self.svm.save_results(prediction, results_path, {'epoch': epoch + 1, 'AE Loss': logs})

        input_fts = Input(shape=(self.x_train.shape[1],))
        
        encoded = Dense(1426, activation='relu')(input_fts)
        encoded = Dense(732, activation='relu')(encoded)
        encoded = Dense(328, activation='relu')(encoded)
        
#         encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
        encoded = Dense(enc_dim, activation='relu')(encoded)
#         encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
        
        decoded = Dense(328, activation='relu')(encoded)
        decoded = Dense(732, activation='relu')(decoded)
        decoded = Dense(1426, activation='relu')(decoded)
        decoded = Dense(self.x_train.shape[1], activation='relu')(decoded)
     
        self.autoencoder = Model(inputs=[input_fts], outputs=decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.autoencoder.summary()
        
        sys.stdout = old_stdout
        
        Logger().write_message('Model summary:\n' + buffer.getvalue(), MessageType.INF)

        encoded_input = Input(shape=(enc_dim,))
        decoder_layer = self.autoencoder.layers[-4](encoded_input)
        decoder_layer = self.autoencoder.layers[-3](decoder_layer)
        decoder_layer = self.autoencoder.layers[-2](decoder_layer)
        decoder_layer = self.autoencoder.layers[-1](decoder_layer)

        self.encoder = Model(inputs=[input_fts], outputs=encoded)
        self.decoder = Model(encoded_input, decoder_layer)
        
        svm = LambdaCallback(on_epoch_end=svm_callback)
        
        noise = (np.random.normal(loc=0.5, scale=0.5, size=self.x_train.shape)) / 10
        history = self.autoencoder.fit([self.x_train + noise], 
                                       self.x_train,
                                       epochs=nepochs,
                                       batch_size=128,
                                       shuffle=True,
                                       verbose=1,
                                       validation_split=0.2,
                                       callbacks=[svm])
        return history
    
    def plot_loss(self, history, results_path):
        '''
        Plots loss and validation loss along training
        
        @param history: dictionary with training history
        @param results_path: string with path to save results under
        '''
        try:
            plt.figure()
            plt.rcParams.update({'font.size': 10})
            
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Autoencoder Loss')
            
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend(['train', 'test'], loc='upper right')
            
            root_path = os.sep.join(results_path.split(os.sep)[:-1])
            if not os.path.isdir(root_path):
                os.mkdir(root_path)
            
            plt.savefig(results_path)
        except OSError:
            Logger().write_message('Loss image could not be saved under %s.' % results_path, MessageType.ERR)
    
    def plot_encoding(self, input_set, encoding, output_set, results_path):
        '''
        Plots input example vs encoded example vs decoded example of 5 random examples
        in test set
        
        @param input_set: autoencoder input
        @param encoding: autoencoder encoded features
        @param output_set: autoencoder output
        @param results_path: string with path to save results under
        '''
        ex_idx = set()
        while len(ex_idx) < 5:
            ex_idx.add(randint(0, input_set.shape[0] - 1))
        
        error = output_set - input_set
    
        try:
            plt.figure()
            plt.rcParams.update({'font.size': 6})
            plt.subplots_adjust(wspace=0.4, hspace=0.9)
            
            for i, idx in enumerate(ex_idx):
                ax = plt.subplot(5, 4, 4 * i + 1)
                plt.plot(input_set[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('%d - Input' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 4, 4 * i + 2)
                plt.plot(encoding[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('%d - Encoding' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 4, 4 * i + 3)
                plt.plot(output_set[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('%d - Output' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 4, 4 * i + 4)
                plt.plot(error[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('Error')
                ax.axes.get_xaxis().set_visible(False)
            
            root_path = os.sep.join(results_path.split(os.sep)[:-1])
            if not os.path.isdir(root_path):
                os.mkdir(root_path)
            
            plt.savefig(results_path)
        except OSError:
            Logger().write_message('Error image could not be saved under %s.' % results_path, MessageType.ERR)
            
    def plot_spatial_distribution(self, input_set, encoding, output_set, labels, results_path):
        '''
        Plots the spatial distribution of input, encoding and output using PCA and TSNE
        
        @param input_set: autoencoder input
        @param encoding: autoencoder encoded features
        @param output_set: autoencoder output
        @param labels: data set labels
        @param results_path: string with path to save results under
        ''' 
        plt.figure()
        plt.rcParams.update({'font.size': 8})
             
        try:    
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            pca = PCA(n_components=2)
             
            ax = plt.subplot(231)
            ax.set_title('PCA - Input')
            input_fts = pca.fit_transform(input_set)
            plt.scatter(input_fts[:,0], input_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))
            
            ax = plt.subplot(232)
            ax.set_title('PCA - Encoding')
            encoding_fts = pca.fit_transform(encoding)
            plt.scatter(encoding_fts[:,0], encoding_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))
             
            ax = plt.subplot(233)
            ax.set_title('PCA - Output')
            output_fts = pca.fit_transform(output_set)
            plt.scatter(output_fts[:,0], output_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))
        except ValueError:
            Logger().write_message('PCA could not be computed.', MessageType.ERR)
            
        try:
            tsne = TSNE(n_components=2)
             
            ax = plt.subplot(234)
            ax.set_title('TSNE - Input')
            input_fts = tsne.fit_transform(input_set)
            plt.scatter(input_fts[:,0], input_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))

            ax = plt.subplot(235)
            ax.set_title('TSNE - Encoding')
            encoding_fts = tsne.fit_transform(encoding)
            plt.scatter(encoding_fts[:,0], encoding_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))
                         
            ax = plt.subplot(236)
            ax.set_title('TSNE - Output')
            output_fts = tsne.fit_transform(output_set)
            plt.scatter(output_fts[:,0], output_fts[:,1], c=labels, cmap='hsv', s=np.ones(labels.shape))  
        except ValueError:
            Logger().write_message('TSNE could not be computed.', MessageType.ERR)
        
        try:
            root_path = os.sep.join(results_path.split(os.sep)[:-1])
            if not os.path.isdir(root_path):
                os.mkdir(root_path)

            plt.savefig(results_path)
        except OSError:
            Logger().write_message('Scatter plots could not be saved under %s.' % results_path, MessageType.ERR)
            
    def plot_pca_vs_encoding(self, input_set, encoding, results_path):
        '''
        Plots PCA against encoding components
        
        @param input_set: autoencoder input
        @param encoding: autoencoder encoded features
        @param results_path: string with path to save results under
        ''' 
        ex_idx = set()
        while len(ex_idx) < 5:
            ex_idx.add(randint(0, input_set.shape[0] - 1))
            
        pca = PCA(n_components=encoding.shape[1])
        input_fts = pca.fit_transform(input_set)
    
        try:
            plt.figure()
            plt.rcParams.update({'font.size': 6})
            plt.subplots_adjust(wspace=0.4, hspace=0.9)
            
            for i, idx in enumerate(ex_idx):
                ax = plt.subplot(5, 2, 2 * i + 1)
                plt.plot(input_fts[idx, :], linestyle='None', marker='o', markersize=3)
                ax.set_title('%d - PCA' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 2, 2 * i + 2)
                plt.plot(encoding[idx, :], linestyle='None', marker='o', markersize=3)
                ax.set_title('%d - Encoding' % idx)
                ax.axes.get_xaxis().set_visible(False)
            
            root_path = os.sep.join(results_path.split(os.sep)[:-1])
            if not os.path.isdir(root_path):
                os.mkdir(root_path)
            
            plt.savefig(results_path)
        except (OSError, ValueError):
            Logger().write_message('PCA vs Encoding image could not be saved under %s.' 
                                   % results_path, MessageType.ERR)
            
            
class VSAutoencoderDoubleInput(VSAutoencoderSingleInput):
    
    def __init__(self, cv=5, njobs=1, **kwargs):
        '''
        Initialize all models and runs grid search on SVM
        
        @param kwargs: dictionary with training and testing data
        @param cv: grid search number of folds in cross validation
        @param njobs: grid search number of jobs to run in parallel
        '''
        self.svm = None
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.svm_history = []
    
        split = kwargs.get('split')
        self.x_train_vis = kwargs.get('x_train')[:,:split]
        self.x_test_vis = kwargs.get('x_test')[:,:split]
        self.x_train_sem = kwargs.get('x_train')[:,split:]
        self.x_test_sem = kwargs.get('x_test')[:,split:]
        
        self.y_train = kwargs.get('y_train')
        self.y_test = kwargs.get('y_test')

        self.svm = SVMClassifier()
        self.svm.run_classifier(kwargs.get('x_train'), self.y_train, cv, njobs)
        
    def run_autoencoder(self, enc_dim, nepochs, results_path):
        '''
        Builds and trains autoencoder model
        
        @param enc_dim: encoding size
        @param nepochs: number of epochs
        @param results_path: string with path to svm save results under
        @return object with training details and history
        '''
        def svm_callback(epoch, logs):
            '''
            Runs SVM and saves prediction results
            
            @param epoch: default callback parameter. Epoch index
            @param logs:default callback parameter. Loss result
            '''
            old_stderr = sys.stderr
            sys.stderr = buffer = io.StringIO()
            
            self.svm.model.best_estimator_.fit(
                self.encoder.predict([self.x_train_vis, np.expand_dims(self.x_train_sem, axis=-1)]), self.y_train)
            
#             self.svm.model.fit(
#                 self.encoder.predict([self.x_train_vis, np.expand_dims(self.x_train_sem, axis=-1)]), self.y_train)
            
            sys.stderr = old_stderr
            Logger().write_message('Epoch %d SVM\n%s' % (epoch, buffer.getvalue()), MessageType.INF)
            
            pred_dict, prediction = self.svm.predict(self.encoder.predict(
                [self.x_test_vis, np.expand_dims(self.x_test_sem, axis=-1)]), self.y_test)
            self.svm_history.append(pred_dict)
            
            if epoch == nepochs - 1:
                self.svm.save_results(prediction, results_path, {'epoch': epoch + 1, 'AE Loss': logs})

        input_vis_fts = Input(shape=(self.x_train_vis.shape[1],))
        input_sem_fts = Input(shape=(self.x_train_sem.shape[1],1))
        embedding_sem = Conv1D(5, 25, use_bias=False, padding='same')(input_sem_fts)
#         embedding_sem = Embedding(input_dim=2, output_dim=45, input_length=85)(input_sem_fts)
#         embedding_sem = Lambda(lambda x: K.tf.Print(x, [x], 'Oi'))(embedding_sem)
        
        flatten_sem = Flatten()(embedding_sem)
        encoded = Concatenate()([input_vis_fts, flatten_sem])
        
        encoded = Dense(1426, activation='relu')(encoded)
        encoded = Dense(732, activation='relu')(encoded)
        encoded = Dense(328, activation='relu')(encoded)
        
#         encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
        encoded = Dense(enc_dim, activation='relu')(encoded)
#         encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
        
        decoded = Dense(328, activation='relu')(encoded)
        decoded = Dense(732, activation='relu')(decoded)
        decoded = Dense(1426, activation='relu')(decoded)
        decoded = Dense(self.x_train_vis.shape[1] + self.x_train_sem.shape[1], activation='relu')(decoded)

        self.autoencoder = Model(inputs=[input_vis_fts, input_sem_fts], outputs=decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.autoencoder.summary()
        
        sys.stdout = old_stdout
        
        Logger().write_message('Model summary:\n' + buffer.getvalue(), MessageType.INF)
            
        encoded_input = Input(shape=(enc_dim,))
        decoder_layer = self.autoencoder.layers[-4](encoded_input)
        decoder_layer = self.autoencoder.layers[-3](decoder_layer)
        decoder_layer = self.autoencoder.layers[-2](decoder_layer)
        decoder_layer = self.autoencoder.layers[-1](decoder_layer)

        self.encoder = Model(inputs=[input_vis_fts, input_sem_fts], outputs=encoded)
        self.decoder = Model(encoded_input, decoder_layer)
        
        svm = LambdaCallback(on_epoch_end=svm_callback)
        
        noise = (np.random.normal(loc=0.5, scale=0.5, size=self.x_train_vis.shape)) / 10
#         noise_sem = (np.random.normal(loc=0.5, scale=0.5, size=self.x_train_sem.shape)) * 100

        history = self.autoencoder.fit([self.x_train_vis + noise, np.expand_dims(self.x_train_sem, axis=-1)], 
                                       FeaturesParser.concatenate_features(self.x_train_vis, self.x_train_sem, 23),
                                       epochs=nepochs,
                                       batch_size=128,
                                       shuffle=True,
                                       verbose=1,
                                       validation_split=0.2,
                                       callbacks=[svm])
        return history


class VSAutoencoderUNet(VSAutoencoderSingleInput):
        
    def run_autoencoder(self, enc_dim, nepochs, results_path):
        '''
        Builds and trains autoencoder model
        
        @param enc_dim: encoding size
        @param nepochs: number of epochs
        @param results_path: string with path to svm save results under
        @return object with training details and history
        '''
        def svm_callback(epoch, logs):
            '''
            Runs SVM and saves prediction results
            
            @param epoch: default callback parameter. Epoch index
            @param logs:default callback parameter. Loss result
            '''
            old_stderr = sys.stderr
            sys.stderr = buffer = io.StringIO()
            
            self.svm.model.best_estimator_.fit(self.encoder.predict([self.x_train]), self.y_train)
            
            sys.stderr = old_stderr
            Logger().write_message('Epoch %d SVM\n%s' % (epoch, buffer.getvalue()), MessageType.INF)
            
            pred_dict, prediction = self.svm.predict(self.encoder.predict([self.x_test]), self.y_test)
            self.svm_history.append(pred_dict)
            
            if epoch == nepochs - 1:
                self.svm.save_results(prediction, results_path, {'epoch': epoch + 1, 'AE Loss': logs})

        input_fts = Input(shape=(self.x_train.shape[1],))
        
        encoded_1 = Dense(1426, activation='relu')(input_fts)
        encoded_2 = Dense(732, activation='relu')(encoded_1)
        encoded_3 = Dense(328, activation='relu')(encoded_2)
        
        encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded_3)
        encoded = Dense(enc_dim, activation='relu')(encoded)
        encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
        
        decoded = Dense(328, activation='relu')(encoded)
        decoded = Dense(732, activation='relu')(Add()([encoded_3, decoded]))
        decoded = Dense(1426, activation='relu')(Add()([encoded_2, decoded]))
        decoded = Dense(self.x_train.shape[1], activation='relu')(Add()([encoded_1, decoded]))
     
        self.autoencoder = Model(inputs=[input_fts], outputs=decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.autoencoder.summary()
        
        sys.stdout = old_stdout
        
        Logger().write_message('Model summary:\n' + buffer.getvalue(), MessageType.INF)

        encoded_input = Input(shape=(enc_dim,))
        decoder_layer = self.autoencoder.layers[-4](encoded_input)
        decoder_layer = self.autoencoder.layers[-3](decoder_layer)
        decoder_layer = self.autoencoder.layers[-2](decoder_layer)
        decoder_layer = self.autoencoder.layers[-1](decoder_layer)

        self.encoder = Model(inputs=[input_fts], outputs=encoded)
        self.decoder = Model(encoded_input, decoder_layer)
        
        svm = LambdaCallback(on_epoch_end=svm_callback)
        
        noise = (np.random.normal(loc=0.5, scale=0.5, size=self.x_train.shape)) / 10
        history = self.autoencoder.fit([self.x_train + noise], 
                                       self.x_train,
                                       epochs=nepochs,
                                       batch_size=128,
                                       shuffle=True,
                                       verbose=1,
                                       validation_split=0.2,
                                       callbacks=[svm])
        return history