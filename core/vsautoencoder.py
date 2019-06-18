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
import sys
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Flatten, Concatenate
from keras.callbacks import LambdaCallback

from core.vsclassifier import SVMClassifier
from utils.logwriter import Logger, MessageType


class VSAutoencoder:
    
    def __init__(self, cv=5, njobs=1, console=False, **kwargs):
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
        self.logger = Logger(console=console)
        
    def run_autoencoder(self, enc_dim, nepochs, tag=None):
        '''
        Builds and trains autoencoder model
        
        @param enc_dim: encoding size
        @param nepochs: number of epochs
        @param tag: string with folder name to saver results under
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
            
            self.svm.model.best_estimator_.fit(self.encoder.predict([self.x_train[:,:2048],
                                                                     np.expand_dims(self.x_train[:,2048:], 
                                                                                    axis=-1)]), self.y_train)
            
            sys.stderr = old_stderr
            self.logger.write_message('Epoch %d SVM\n%s' % (epoch, buffer.getvalue()), MessageType.INF)
            
            pred_dict, prediction = self.svm.predict(self.encoder.predict([self.x_test[:,:2048], 
                                                                           np.expand_dims(self.x_test[:,2048:], 
                                                                                          axis=-1)]), self.y_test)
            self.svm_history.append(pred_dict)
            
            if epoch == nepochs - 1:
                self.svm.save_results(prediction, {'epoch': epoch + 1, 'AE Loss': logs}, tag)

        input_vis_fts = Input(shape=(self.x_train[:,:2048].shape[1],))
        input_sem_fts = Input(shape=(self.x_train[:,2048:].shape[1],1))
        embedding_sem = Conv1D(5, 25, use_bias=False, padding='same')(input_sem_fts)
        
        flatten_sem = Flatten()(embedding_sem)
        encoded = Concatenate()([input_vis_fts, flatten_sem])
        
        encoded = Dense(1426, activation='relu')(encoded)
        encoded = Dense(732, activation='relu')(encoded)
        encoded = Dense(328, activation='relu')(encoded)
        
        #encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
        encoded = Dense(enc_dim, activation='relu')(encoded)
        #encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
        
        decoded = Dense(328, activation='relu')(encoded)
        decoded = Dense(732, activation='relu')(decoded)
        decoded = Dense(1426, activation='relu')(decoded)
        decoded = Dense(self.x_train[:,:2048].shape[1] + self.x_train[:,2048:].shape[1], activation='relu')(decoded)
     
        self.autoencoder = Model(inputs=[input_vis_fts, input_sem_fts], outputs=decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.autoencoder.summary()
        
        sys.stdout = old_stdout
        
        self.logger.write_message('Model summary:\n' + buffer.getvalue(), MessageType.INF)

        encoded_input = Input(shape=(enc_dim,))
        decoder_layer = self.autoencoder.layers[-4](encoded_input)
        decoder_layer = self.autoencoder.layers[-3](decoder_layer)
        decoder_layer = self.autoencoder.layers[-2](decoder_layer)
        decoder_layer = self.autoencoder.layers[-1](decoder_layer)

        self.encoder = Model(inputs=[input_vis_fts, input_sem_fts], outputs=encoded)
        self.decoder = Model(encoded_input, decoder_layer)
        
        svm = LambdaCallback(on_epoch_end=svm_callback)
        noise = (np.random.normal(loc=0.5, scale=0.5, size=self.x_train[:,:2048].shape)) / 10
        
        history = self.autoencoder.fit([self.x_train[:,:2048] + noise, 
                                        np.expand_dims(self.x_train[:,2048:], axis=-1)], 
                                       self.x_train,
                                       epochs=nepochs,
                                       batch_size=128,
                                       shuffle=True,
                                       verbose=1,
                                       validation_split=0.2,
                                       callbacks=[svm])
        return history
