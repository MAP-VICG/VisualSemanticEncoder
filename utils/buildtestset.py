'''
Generates smaller features dataset based on a given limit of instances
per label.

@author: Damares Resende
@contact: damaresresende@usp.br
@since: Apr 03, 2019

@organization: University of Sao Paulo (USP)
    Institute of Mathematics and Computer Science (ICMC) 
    Laboratory of Visualization, Imaging and Computer Graphics (VICG)
'''

from os import remove

mask = []
limit = 300

try:
    remove('../test/_mockfiles/awa2/features/ResNet101/AwA2-labels2.txt')
    remove('../test/_mockfiles/awa2/features/ResNet101/AwA2-features2.txt')
except (FileNotFoundError, OSError):
    pass

with open('../_files/awa2/features/ResNet101/AwA2-labels.txt') as f:
    
    labels = {5: 0, 6: 0, 7: 0, 13: 0, 15: 0, 17: 0, 20: 0, 25: 0, 38: 0, 39: 0, 43: 0, 49: 0}
    
    for i, line in enumerate(f.readlines()):
        lb = int(line)
        
        if lb in labels.keys() and labels[lb] < limit:
            mask.append((i, line))
            labels[lb] += 1
        
with open('../_files/awa2/features/ResNet101/AwA2-features.txt') as f:
    index = 0
    for i, line in enumerate(f.readlines()):
        if i == mask[index][0]:
            with open('../test/_mockfiles/awa2/features/ResNet101/AwA2-features2.txt', 'a+') as w:
                w.write(line)
            with open('../test/_mockfiles/awa2/features/ResNet101/AwA2-labels2.txt', 'a+') as w:
                w.write(mask[index][1])
            index += 1

        if index == len(mask):
            break