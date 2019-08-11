'''
Created on Jul 27, 2019

@author: damaresresende
'''
import numpy as np
from random import randint
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense

from core.vsencoder import SemanticEncoder

import gc
import tensorflow as tf
from keras import backend as K

def create_enc_model(data):
    input_fts = Input(shape=(data.shape[1],))
        
    encoded = Dense(1426, activation='relu')(input_fts)
    encoded = Dense(732, activation='relu')(encoded)
    encoded = Dense(328, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoder = Model(inputs=input_fts, outputs=encoded)

    return encoder

def create_dec_model(data):
    input_fts = Input(shape=(128,))
    decoded = Dense(328, activation='relu')(input_fts)
    decoded = Dense(732, activation='relu')(decoded)
    decoded = Dense(1426, activation='relu')(decoded)
    decoded = Dense(data.shape[1], activation='relu')(decoded)
    decoder = Model(inputs=input_fts, outputs=decoded)

    return decoder

def load_data():
    data = []
    with open('../_files/results/test_set.txt') as f:
        for line in f.readlines():
            if line != '\n' and line != '':
                data.append([float(v) for v in line.split(',')])
               
    with open('../_files/results/test_labels.txt') as f:
        labels = [float(v) for v in f.readline().split(',')]
    
    return np.array(data), np.array(labels)
        
def filter_data(data, labels):
    # horse, blue+whale, giraffe, zebra, dolphin
    classes = [7, 9, 31, 38, 50]
    mask = [False] * labels.shape[0]
    
    for idx, lb in enumerate(labels):
        if lb in classes:
            mask[idx] = True
    
    return data[mask, :], labels[mask]

def plot_encoding(input_set, encoding, output_set):
    '''
    Plots input example vs encoded example vs decoded example of 5 random examples
    in test set
    
    @param input_set: autoencoder input
    @param encoding: autoencoder encoded features
    @param output_set: autoencoder output
    @param tag: string with folder name to saver results under
    '''
    ex_idx = set()
    while len(ex_idx) < 5:
        ex_idx.add(randint(0, input_set.shape[0] - 1))
    
    error = output_set - input_set

    
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
    
    plt.show()
   
         
data, lbs = load_data()
data, lbs = filter_data(data, lbs)
# data = data[:, :2048]

enc = SemanticEncoder(50, 128, console=True)
data = enc.pick_semantic_features('SHAPE', data, opposite=False)

labels = {7: 'horse', 9: 'blue+whale', 31: 'giraffe', 38: 'zebra', 50: 'dolphin'}

# with open('test_data_class.tsv', 'w+') as f:
#     for l in lbs:
#         f.write(labels[int(l)] + '\n')
         
with open('input_data_shape.tsv', 'w+') as f:
#     f.write('\t'.join(['a' + str(i) for i in range(data.shape[1])]) + '\tclass\n')
      
    for i, ex in enumerate(data):
        f.write('\t'.join([str(v) for v in ex]) +  '\n')
#         f.write(labels[lbs[i]] + '\t' + '\t'.join([str(v) for v in ex]) +  '\n')
 
model_enc = create_enc_model(data)
model_enc.load_weights('../_files/results/SHAPE/encoder.h5')
enc = model_enc.predict(data)
  
with open('coded_data_shape.tsv', 'w+') as f:
#     f.write('\t'.join(['a' + str(i) for i in range(enc.shape[1])]) + '\tclass\n')
      
    for i, ex in enumerate(enc):
        f.write('\t'.join([str(v) for v in ex]) +  '\n')
#         f.write(labels[lbs[i]] + '\t' + '\t'.join([str(v) for v in ex]) +  '\n')
  
tf.reset_default_graph()
K.clear_session()
gc.collect()
          
model_dec = create_dec_model(data)
model_dec.load_weights('../_files/results/SHAPE/decoder.h5')
dec = model_dec.predict(enc) 
  
with open('output_data_shape.tsv', 'w+') as f:
#     f.write('\t'.join(['a' + str(i) for i in range(dec.shape[1])]) + '\tclass\n')
      
    for i, ex in enumerate(dec):
        f.write('\t'.join([str(v) for v in ex]) +  '\n')
#         f.write(labels[lbs[i]] + '\t' + '\t'.join([str(v) for v in ex]) +  '\n')
          
# plot_encoding(data, enc, dec)