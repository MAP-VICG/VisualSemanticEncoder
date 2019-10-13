import numpy as np
import statistics as st
import xml.etree.ElementTree as ET

values = {'simple': [], 'conv': [], 'simple_batch': [], 'conv_batch': []}
# values = {'simple': [], 'conv': []}

for name in values.keys():
    tree = ET.parse('/Users/damaresresende/Desktop/bin/results_' + name + '/ae_results.xml')
    root = tree.getroot()

    for typ in ['ALL', 'COLOR', 'TEXTURE', 'PARTS', 'SHAPE', '_COLOR', '_TEXTURE', '_PARTS', '_SHAPE']:
        try:
            score = [float(v) for v in root.find(typ).find('accuracy').find('f1-score').text.split(',')]
            values[name].append([max(score), st.mean(score), st.stdev(score)])
        except AttributeError:
            print('Issue in ' + typ + ' for ' + name)

for i in range(9):
    res = ''
    for name in values.keys():
        # res += str(values[name][i][0]) + ' & ' + str(values[name][i][1]) + ' & ' + str(values[name][i][2]) + ' & '
        res += "& {:.2f} & {:.2f} & {:.2f} ".format(values[name][i][0], values[name][i][1], values[name][i][2])
    print(res)# + '& . & . & . & . & . & .')


res = ''
for name in values.keys():
    res += "& {:.2f} & {:.2f} & {:.2f} ".format(np.mean(np.array(values[name])[:,0]), np.mean(np.array(values[name])[:,1]), np.mean(np.array(values[name])[:,2]))
    
print(res)