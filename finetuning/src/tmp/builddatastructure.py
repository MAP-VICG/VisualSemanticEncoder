import os
from shutil import copyfile

data_path = '../../../../Datasets/Birds/images'
test_split = '../../../Datasets/Birds/lists/test.txt'
train_split = '../../../Datasets/Birds/lists/train.txt'

training_path = '../../../../Datasets/Birds/training'
test_path = '../../../../Datasets/Birds/test'


print('>> Copying training images')
with open(train_split) as f:
    if not os.path.exists(training_path):
        os.mkdir(training_path)

    for img in f.readlines():
        img = img.strip()
        print('   >>> Copying image %s' % img)
        img_dir = os.path.join(training_path, img.split('/')[0])
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        copyfile(os.path.join(data_path, img), os.path.join(training_path, img))

print('>> Copying test images')
with open(test_split) as f:
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    for img in f.readlines():
        img = img.strip()
        print('   >>> Copying image %s' % img)
        img_dir = os.path.join(test_path, img.split('/')[0])
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        copyfile(os.path.join(data_path, img), os.path.join(test_path, img))

print('>>> Job done. Bye o/')
