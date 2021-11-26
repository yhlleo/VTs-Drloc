
import os
import scipy
from scipy import io
import shutil
import numpy as np

data_dir = "./datasets"

datasets = [
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch"
]

for da in datasets:
    cur_dir = os.path.join(data_dir, da)
    train_dir = os.path.join(cur_dir, "train")
    val_dir = os.path.join(cur_dir, "val")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    with open(os.path.join(cur_dir, "{}_train.txt".format(da)), 'r') as fin:
        for line in fin:
            im_path = line.strip().split()[0]
            splits = im_path.split("/")
            sub_folder = splits[1]
            fname = splits[-1]
            save_dir = os.path.join(train_dir, sub_folder)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            shutil.move(os.path.join(cur_dir, im_path), os.path.join(save_dir, fname))

    with open(os.path.join(cur_dir, "{}_test.txt".format(da)), 'r') as fin:
        for line in fin:
            im_path = line.strip().split()[0]
            splits = im_path.split("/")
            sub_folder = splits[1]
            fname = splits[-1]
            save_dir = os.path.join(val_dir, sub_folder)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            shutil.move(os.path.join(cur_dir, im_path), os.path.join(save_dir, fname))
            
    