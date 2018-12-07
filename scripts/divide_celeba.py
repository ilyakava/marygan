
import os.path
from shutil import copyfile

from tqdm import tqdm

import pdb

# attr_file = '/Users/artsyinc/Documents/MATH630/research/data/celeba/list_attr_celeba.txt'
attr_file = '/scratch0/ilya/locDoc/data/celeba/list_attr_celeba.txt'
src_folder = '/scratch0/ilya/locDoc/data/celeba_thirds/img'
data1folder = '/scratch0/ilya/locDoc/data/celeba_partitions/male_close/img'
data2folder = '/scratch0/ilya/locDoc/data/celeba_partitions/female_close/img'

male_idx = 20

with open(attr_file,'r') as f:
    for i, x in enumerate(tqdm(f)):
        x = x.rstrip()
        if i < 2:
            continue
        fname = x.split(' ')[0]
        attrs = x.split(' ')[1:]
        attrs = [a for a in attrs if len(a) > 0]
        srcpath = '%s/%s' % (src_folder, fname)
        if len(attrs) == 40 and os.path.exists(srcpath):
            attr = int(attrs[male_idx])
            if attr == 1:
                copyfile(srcpath, '%s/%s' % (data1folder, fname))
            elif attr == -1:
                copyfile(srcpath, '%s/%s' % (data2folder, fname))
        