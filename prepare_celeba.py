import random
import pickle
import zipfile

import numpy as np

from scipy import misc
import tqdm

archive = zipfile.ZipFile('CelebAHQ128PNGLANCZOS.zip', 'r')

names = archive.namelist()

names = [x for x in names if x[-4:] == '.png']

count = len(names)
print("Count: %d" % count)

folds = 1

random.shuffle(names)

images = {}

count = len(names)
print("Count: %d" % count)
count_per_fold = count // folds

i = 0
im = 0
for x in tqdm.tqdm(names):
    imgfile = archive.open(x)
    image = misc.imread(imgfile)
    images[x] = image
    im += 1

    if im == count_per_fold:
        output = open('data_fold_%d.pkl' % i, 'wb')
        pickle.dump(list(images.values()), output)
        output.close()
        i += 1
        im = 0
        images.clear()
