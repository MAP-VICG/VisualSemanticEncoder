'''
Created on Jun 17, 2019

@author: damaresresende
'''
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt

idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 45, 46, 24, 25, 19, 20, 23, 26, 31]     
    
x = [22739., 18333.,  2576., 24533., 17092.,  3373.,   664.,  3571.,
       13616.,  9335.,  2747., 29666.,  8114., 17876., 26535., 12111.,
       18253., 18662.,  4081.,  1891., 12901., 10461., 17393., 12842.,
        6065., 29463., 31030., 14984.,  8224.,  2266.,  7953., 14447.,
        1949., 18675.,   383.,  3067.,  5826.,  1646., 31577., 29831.,
       15133., 27285.,  5957., 22457.,  6068., 33616., 27725., 18614.,
        7124.,  6864., 24325., 10695., 13283.,  1174., 21823.,  1521.,
       18811., 16443., 11782.,  2847.,   883.,  7752., 29473., 34097.,
        5296.,  4968.,  1019., 10140., 17688., 14440., 15408.,  8182.,
        9947.,  4949., 32567.,  5826.,  6210.,  2454., 13991., 25839.,
       24201., 24182., 17796., 13112., 15856.]
    
values = []
for i in idx:
    values.append(x[i-1])
       
       
### CLASSES
 
values = [1046.,  852.,  291.,  193.,  549.,  747., 1645., 1033.,  174.,
        500.,  188.,  100.,  877.,  684.,  720.,  704.,  291.,  709.,
       1038.,  872.,  728.,  664., 1420.,  988.,  728.,  779., 1200.,
        696., 1088.,  383., 1202.,  589.,  567.,  310.,  272.,  758.,
        895., 1170.,  874., 1344.,  630.,  713., 1019.,  185.,  868.,
       1028.,  215.,  512., 1338.,  946.]
 
# z = [37322] * 23
 
# label = ['black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow',
#          'patches', 'spots', 'stripes', 'furry', 'hairless', 'toughskin', 'bulbous', 'bipedal', 
#          'quadrupedal', 'longleg', 'longneck', 'flippers', 'hands', 'paws', 'tail', 'horns']
       
# label = ['preto', 'branco', 'azul', 'marrom', 'cinza', 'laranjado', 'vermelho', 'amarelo',
#          'manchas', 'pintas', 'listras', 'peludo', 'sem pelo', 'pele dura', 'bulboso', 'bípede', 
#          'quadrúpede', 'perna longa', 'pescoço longo', 'nadadeiras', 'mãos', 'patas', 'rabo', 'chifres']
 
label = [ 'antílope', 'urso pardo', 'orca', 'castor', 'dálmata', 'gato persa', 'cavalo', 'pastor alemão', 
         'baleia azul', 'gato siamês', 'gambá', 'toupeira', 'tigre', 'hipopótamo', 'leopardo', 'alce', 
         'macaco aranha', 'jubarte', 'elefante', 'gorila', 'boi', 'raposa', 'ovelha', 'foca', 'chipanzé',
         'hamister', 'esquilo', 'rinoceronte', 'coelho', 'morcego', 'girafa', 'lobo', 'chihuahua', 'rato',
         'doninha', 'lontra', 'búfalo', 'zebra', 'panda', 'veado', 'lince', 'porco', 'leão', 'camundongo',
         'urso polar', 'collie', 'morsa', 'guaxinim', 'vaca', 'golfinho' ] 
      
plt.figure(figsize=(20,15))
index = np.arange(len(label))
# plt.bar(index, values, color=['#FF6666', '#FF6666', '#FF6666', '#FF6666', 
#                               '#FF6666', '#FF6666', '#FF6666', '#FF6666', 
#                               '#00CC66', '#00CC66', '#00CC66', '#00CC66', '#00CC66', '#00CC66', 
#                               '#6666FF', '#6666FF', '#6666FF', '#6666FF', '#6666FF',
#                               '#B266FF', '#B266FF', '#B266FF', '#B266FF', '#B266FF'])
plt.bar(index, values, color=['#3399FF'])
  
# plt.bar(index, [22739., 18333.,  2576., 24533.], color='#FF6666')
# plt.bar(index, [22739., 18333.,  2576., 24533.], color='#00CC66')
# plt.bar(index, [22739., 18333.,  2576., 24533.], color='#6666FF')
# plt.bar(index, [22739., 18333.,  2576., 24533.], color='#B266FF')
  
# plt.plot(z, linestyle='--', color='k')
# plt.xlabel('Classes de Animais', fontsize=18)
plt.xlabel('Classes', fontsize=18)
plt.ylabel('Número de Imagens', fontsize=18)
plt.xticks(index, label, fontsize=12, rotation='vertical')
plt.yticks(fontsize=12)
# plt.legend(['color', 'texture', 'shape', 'parts'], loc='top left')
# plt.title('Attributes Distribution')
plt.savefig('/Users/damaresresende/Desktop/classes_dist.png', bbox_inches='tight')
# plt.show()



# tree = ET.parse('/Users/damaresresende/Desktop/quali_results/encoder/idx/results_conv/ae_results.xml')
# root = tree.getroot()
#    
# ref = [0.92] * 50
# _all = [float(v) for v in root.find('ALL/accuracy/f1-score').text.split(',')]
# parts = [float(v) for v in root.find('PARTS/accuracy/f1-score').text.split(',')]
# texture = [float(v) for v in root.find('TEXTURE/accuracy/f1-score').text.split(',')]
# color = [float(v) for v in root.find('COLOR/accuracy/f1-score').text.split(',')]
# shape = [float(v) for v in root.find('SHAPE/accuracy/f1-score').text.split(',')]
#    
# _parts = [float(v) for v in root.find('_PARTS/accuracy/f1-score').text.split(',')]
# _texture = [float(v) for v in root.find('_TEXTURE/accuracy/f1-score').text.split(',')]
# _color = [float(v) for v in root.find('_COLOR/accuracy/f1-score').text.split(',')]
# _shape = [float(v) for v in root.find('_SHAPE/accuracy/f1-score').text.split(',')]
#    
# plt.figure(figsize=(12,3.5))
# plt.rcParams.update({'font.size': 10})
#  
# plt.subplot(121)
# plt.plot(ref, linestyle='--', color='#000000')
# plt.plot(_all, color='#C0C0C0')
# plt.plot(parts, color='#0000CC')
# plt.plot(texture, color='#CC0000')
# plt.plot(color, color='#00CC00')
# plt.plot(shape, color='#FF8000')
#    
# plt.xlabel('Épocas')
# plt.ylabel('F1-Score')
# # plt.title('SVM Prediction - Categories')
# plt.legend(['referência', 'todos', 'partes', 'textura', 'cor', 'forma'], fontsize=7)
#    
# plt.subplot(122)
# plt.plot(ref, linestyle='--', color='#000000')
# plt.plot(_all, color='#C0C0C0')
# plt.plot(_parts, color='#0000CC')
# plt.plot(_texture, color='#CC0000')
# plt.plot(_color, color='#00CC00')
# plt.plot(_shape, color='#FF8000')
#     
# plt.xlabel('Épocas')
# plt.ylabel('F1-Score')
# # plt.title('SVM Prediction - Negation')
# plt.legend(['referência', 'todos', 'sem partes', 'sem textura', 'sem cor', 'sem forma'], fontsize=7)
#                 
# plt.savefig('/Users/damaresresende/Desktop/ae_results_ind_conv.png', bbox_inches='tight')
