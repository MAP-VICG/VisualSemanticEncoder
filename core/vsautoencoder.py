'''
Autoencoder for visual and semantic features of images

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Mar 23, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

import os
from random import randint
from keras.models import Model
from keras.layers import Input, Dense
from matplotlib import pyplot as plt


class VSAutoencoder:
    
    def __init__(self):
        '''
        Initialize all models
        '''
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
    
    def run_autoencoder(self, x_train, enc_dim, nepochs):
        '''
        Builds and trains autoencoder model
        
        @param io_dim: autoencoder input/output size
        @param enc_dim: encoding size
        @param nepochs: number of epochs
        @return object with training details and history
        '''
        input_fts = Input(shape=(x_train.shape[1],))
        
        encoded = Dense(1426, activation='relu')(input_fts)
        encoded = Dense(732, activation='relu')(encoded)
        encoded = Dense(328, activation='relu')(encoded)
        
        encoded = Dense(enc_dim, activation='relu')(encoded)
        
        decoded = Dense(328, activation='relu')(encoded)
        decoded = Dense(732, activation='relu')(decoded)
        decoded = Dense(1426, activation='relu')(decoded)
        decoded = Dense(x_train.shape[1], activation='relu')(decoded)

        self.autoencoder = Model(inputs=input_fts, outputs=decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        history =  self.autoencoder.fit(x_train, 
                                        x_train,
                                        epochs=nepochs,
                                        batch_size=256,
                                        shuffle=True,
                                        verbose=1,
                                        validation_split=0.2)
        
        encoded_input = Input(shape=(enc_dim,))
        decoder_layer = self.autoencoder.layers[-4](encoded_input)
        decoder_layer = self.autoencoder.layers[-3](decoder_layer)
        decoder_layer = self.autoencoder.layers[-2](decoder_layer)
        decoder_layer = self.autoencoder.layers[-1](decoder_layer)
        
        self.encoder = Model(input_fts, encoded)
        self.decoder = Model(encoded_input, decoder_layer)
        
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
            print('>> ERROR: Loss image could not be saved under %s' % results_path)
    
    def plot_error(self, x_test, results_path):
        '''
        Plots input example vs encoded example vs decoded example of 5 random examples
        in test set
        
        @param x_test: test set
        @param results_path: string with path to save results under
        '''
        ex_idx = set()
        while len(ex_idx) < 5:
            ex_idx.add(randint(0, x_test.shape[0] - 1))
            
        encoded_fts = self.encoder.predict(x_test)
        decoded_fts = self.decoder.predict(encoded_fts)
        error = x_test - decoded_fts
    
        try:
            plt.figure()
            plt.rcParams.update({'font.size': 8})
            plt.subplots_adjust(wspace=0.3, hspace=0.9)
            
            for i, idx in enumerate(ex_idx):
                ax = plt.subplot(5, 4, 4 * i + 1)
                plt.plot(x_test[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('Ex %d - Input' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 4, 4 * i + 2)
                plt.plot(encoded_fts[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('Ex %d - Encoding' % idx)
                ax.axes.get_xaxis().set_visible(False)
                
                ax = plt.subplot(5, 4, 4 * i + 3)
                plt.plot(decoded_fts[idx, :], linestyle='None', marker='o', markersize=1)
                ax.set_title('Ex %d - Output' % idx)
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
            print('>> ERROR: Error image could not be saved under %s' % results_path)