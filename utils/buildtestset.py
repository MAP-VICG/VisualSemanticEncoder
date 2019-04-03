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
limit = 20

try:
    remove('../test/_mockfiles/awa2/features/ResNet101/AwA2-labels.txt')
    remove('../test/_mockfiles/awa2/features/ResNet101/AwA2-features.txt')
except (FileNotFoundError, OSError):
    pass

with open('../_files/awa2/features/ResNet101/AwA2-labels.txt') as f:
    
    labels = [0 for i in range(0, 50)]
    
    for i, line in enumerate(f.readlines()):
        lb = int(line)
        
        if labels[lb-1] < limit:
            mask.append((i, line))
            labels[lb-1] += 1
        
with open('../_files/awa2/features/ResNet101/AwA2-features.txt') as f:
    index = 0
    for i, line in enumerate(f.readlines()):
        if i == mask[index][0]:
            with open('../test/_mockfiles/awa2/features/ResNet101/AwA2-features.txt', 'a+') as w:
                w.write(line)
            with open('../test/_mockfiles/awa2/features/ResNet101/AwA2-labels.txt', 'a+') as w:
                w.write(mask[index][1])
            index += 1

        if index == len(mask):
            break