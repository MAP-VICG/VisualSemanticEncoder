'''
Created on Jul 27, 2019

@author: damaresresende
'''
import numpy as np
from random import randint
from matplotlib import pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Conv1D, BatchNormalization

from core.vsencoder import SemanticEncoder

import gc
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops

def create_enc_model(data):
#     input_vis_fts = Input(shape=(data[:,:2048].shape[1],))
#     input_sem_fts = Input(shape=(data[:,2048:].shape[1],1))
#     embedding_sem = Conv1D(5, 25, use_bias=False, padding='same')(input_sem_fts)
#  
#     flatten_sem = Flatten()(embedding_sem)
#          
#     encoded = Concatenate()([input_vis_fts, flatten_sem])
#     encoded = Dense(1426, activation='relu')(encoded)
#     encoded = Dense(732, activation='relu')(encoded)
#     encoded = Dense(328, activation='relu')(encoded)
#          
#     encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
#     code = Dense(128, activation='relu')(encoded)
#     encoder = Model(inputs=[input_vis_fts, input_sem_fts], outputs=code)

    input_fts = Input(shape=(data.shape[1],))
         
    encoded = Dense(1426, activation='relu')(input_fts)
    encoded = Dense(732, activation='relu')(encoded)
    encoded = Dense(328, activation='relu')(encoded)
    encoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoder = Model(inputs=input_fts, outputs=encoded)

    return encoder

def create_dec_model(data):
#     input_fts = Input(shape=(128,))
#     decoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(input_fts)
#     decoded = Dense(328, activation='relu')(decoded)
#         
#     decoded = Dense(732, activation='relu')(decoded)
#     decoded = Dense(1426, activation='relu')(decoded)
#     decoded = Dense(data[:,:2048].shape[1] + data[:,2048:].shape[1], activation='relu')(decoded)
#     decoder = Model(input_fts, decoded)
        
    input_fts = Input(shape=(128,))
    decoded = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(input_fts)
    decoded = Dense(328, activation='relu')(decoded)
    decoded = Dense(732, activation='relu')(decoded)
    decoded = Dense(1426, activation='relu')(decoded)
    decoded = Dense(data.shape[1], activation='relu')(decoded)
    decoder = Model(inputs=input_fts, outputs=decoded)

    return decoder

def load_data():
    data = []
    with open('/Users/damaresresende/Desktop/quali_results/encoder/_datasets/test_set.txt') as f:
        for line in f.readlines():
            if line != '\n' and line != '':
                data.append([float(v) for v in line.split(',')])
               
    with open('/Users/damaresresende/Desktop/quali_results/encoder/_datasets/test_labels.txt') as f:
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
   
         
data_f, lbs = load_data()
data_f, lbs = filter_data(data_f, lbs)
# data = data[:, :2048]

labels = {7: 'horse', 9: 'blue+whale', 31: 'giraffe', 38: 'zebra', 50: 'dolphin'}

with open('/Users/damaresresende/Desktop/test_data_class.tsv', 'w+') as f:
    for l in lbs:
        f.write(labels[int(l)] + '\n')

for tag in ['_ALL']:#['ALL', 'COLOR', 'TEXTURE', 'PARTS', 'SHAPE', '_COLOR', '_TEXTURE', '_PARTS', '_SHAPE']:
    print(tag)
    if tag.startswith('_'):
        tagname = 'not' + tag.lower()
        ptag = tag[1:]
        opp = True
    else:
        tagname = tag.lower()
        ptag = tag
        opp = False
         
    enc = SemanticEncoder(50, 128, console=True)
    data = data_f[:,:2048]
#     data = enc.pick_semantic_features(ptag, data_f, opposite=opp)
     
    with open('/Users/damaresresende/Desktop/input_data_%s.tsv' % tagname, 'w+') as f:
    #     f.write('\t'.join(['a' + str(i) for i in range(data.shape[1])]) + '\tclass\n')
           
        for i, ex in enumerate(data):
            f.write('\t'.join([str(v) for v in ex]) +  '\n')
    #         f.write(labels[lbs[i]] + '\t' + '\t'.join([str(v) for v in ex]) +  '\n')
      
    model_enc = create_enc_model(data)
    model_enc.load_weights('/Users/damaresresende/Desktop/quali_results/encoder/idx/results_simple_batch/%s/encoder.h5' % tag)
#     input_vis_fts = data[:,:2048]
#     input_sem_fts = np.expand_dims(data[:,2048:], axis=-1)
#     enc = model_enc.predict([input_vis_fts, input_sem_fts])
    enc = model_enc.predict(data)
       
    with open('/Users/damaresresende/Desktop/coded_data_%s.tsv' % tagname, 'w+') as f:
    #     f.write('\t'.join(['a' + str(i) for i in range(enc.shape[1])]) + '\tclass\n')
           
        for i, ex in enumerate(enc):
            f.write('\t'.join([str(v) for v in ex]) +  '\n')
    #         f.write(labels[lbs[i]] + '\t' + '\t'.join([str(v) for v in ex]) +  '\n')
       
    ops.reset_default_graph()
    K.clear_session()
    gc.collect()
               
    model_dec = create_dec_model(data)
    model_dec.load_weights('/Users/damaresresende/Desktop/quali_results/encoder/idx/results_simple_batch/%s/decoder.h5' % tag)
    dec = model_dec.predict(enc) 
       
    with open('/Users/damaresresende/Desktop/output_data_%s.tsv' % tagname, 'w+') as f:
    #     f.write('\t'.join(['a' + str(i) for i in range(dec.shape[1])]) + '\tclass\n')
           
        for i, ex in enumerate(dec):
            f.write('\t'.join([str(v) for v in ex]) +  '\n')
    #         f.write(labels[lbs[i]] + '\t' + '\t'.join([str(v) for v in ex]) +  '\n')
               
    # plot_encoding(data, enc, dec)
    ops.reset_default_graph()
    K.clear_session()
    gc.collect()