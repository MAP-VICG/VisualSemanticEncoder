import numpy as np
from matplotlib import pyplot as plt

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_x_test.txt') as f:
    x_test = np.array([list(map(float, row.strip().split())) for row in f.readlines()])

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_x_train.txt') as f:
    x_train = np.array([list(map(float, row.strip().split())) for row in f.readlines()])

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_y_test.txt') as f:
    y_test = np.array([int(value) for value in f.readlines()])

with open('/Users/damaresresende/Projects/Datasets/CUB200/CUB200_y_train.txt') as f:
    y_train = np.array([int(value) for value in f.readlines()])

print('x_test shape: %s' % str(x_test.shape))
print('x_train shape: %s' % str(x_train.shape))
print('y_test shape: %s' % str(y_test.shape))
print('y_train shape: %s' % str(y_train.shape))

train_distribution = np.zeros(200)
test_distribution = np.zeros(200)

for label in y_test:
    test_distribution[label-1] += 1

for label in y_train:
    train_distribution[label-1] += 1

x = [i+1 for i in range(200)]

plt.subplot(2, 1, 1)
plt.bar(x, test_distribution)
plt.title('Test Distribution')

plt.subplot(2, 1, 2)
plt.bar(x, train_distribution)
plt.title('Train Distribution')

plt.tight_layout()
plt.show()

test_mask = [False] * x_test.shape[0]
for klass, value in enumerate(test_distribution):
    k = 0
    for i in range(len(test_mask)):
        if y_test[i] == klass:
            test_mask[i] = True
            k += 1
        if k >= value * 0.2:
            break

train_mask = [False] * x_train.shape[0]
for klass, value in enumerate(train_distribution):
    k = 0
    for i in range(len(train_mask)):
        if y_train[i] == klass:
            train_mask[i] = True
            k += 1
        if k >= value * 0.2:
            break

count = 0
for value in test_mask:
    if value:
        count += 1
print('Using %d examples from test set' % count)

count = 0
for value in train_mask:
    if value:
        count += 1
print('Using %d examples from train set' % count)

x_train_mk = x_train[train_mask, :]
y_train_mk = y_train[train_mask]

x_test_mk = x_test[test_mask, :]
y_test_mk = y_test[test_mask]

print('x_test_mk shape: %s' % str(x_test_mk.shape))
print('x_train_mk shape: %s' % str(x_train_mk.shape))
print('y_test_mk shape: %s' % str(y_test_mk.shape))
print('y_train_mk shape: %s' % str(y_train_mk.shape))

with open('/encoding/test/mockfiles/CUB200_x_test.txt', 'w+') as f:
    for row in x_test_mk:
        f.write(' '.join(list(map(str, row))) + '\n')

with open('/encoding/test/mockfiles/CUB200_x_train.txt', 'w+') as f:
    for row in x_train_mk:
        f.write(' '.join(list(map(str, row))) + '\n')

with open('/encoding/test/mockfiles/CUB200_y_test.txt', 'w+') as f:
    for row in y_test_mk:
        f.write(str(row) + '\n')

with open('/encoding/test/mockfiles/CUB200_y_train.txt', 'w+') as f:
    for row in y_train_mk:
        f.write(str(row) + '\n')
