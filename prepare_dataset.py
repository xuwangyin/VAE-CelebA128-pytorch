import random
import pickle
import zipfile
import glob
import argparse
import pathlib
import os
import PIL.Image

import numpy as np

from scipy import misc
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--folds', type=int, default=1)
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--savedir', type=str, required=True)
args = parser.parse_args()

pathlib.Path(args.savedir).mkdir(exist_ok=True)

names = glob.glob(os.path.join(args.datadir, 'png/*.png'))

count = len(names)
print("Count: %d" % count)
assert count > 0


random.shuffle(names)

images = {}

count = len(names)
print("Count: %d" % count)
count_per_fold = count // args.folds

i = 0
im = 0
for x in tqdm.tqdm(names):
    image = np.array(PIL.Image.open(x))
    images[x] = image
    im += 1

    if im == count_per_fold:
        output = open(os.path.join(args.savedir, 'data_fold_%d.pkl' % i), 'wb')
        pickle.dump(list(images.values()), output)
        output.close()
        i += 1
        im = 0
        images.clear()
