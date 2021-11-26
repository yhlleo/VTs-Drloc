import os
import scipy
from scipy import io
import shutil
import numpy as np

data_dir = "./datasets/flowers102"

labels = scipy.io.loadmat(os.path.join(data_dir, "imagelabels.mat"))["labels"][0]
splits = scipy.io.loadmat(os.path.join(data_dir, "setid.mat"))

train_list = np.concatenate((splits["trnid"], splits["valid"]), axis=1)[0]
val_list = splits["tstid"][0]

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

for i in range(1, 103):
    if not os.path.exists(os.path.join(train_dir, str(i))):
        os.mkdir(os.path.join(train_dir, str(i)))
    if not os.path.exists(os.path.join(test_dir, str(i))):
        os.mkdir(os.path.join(test_dir, str(i)))

for idx, lab in enumerate(labels):
    fname = 'image_{:05d}.jpg'.format(idx+1)
    if idx+1 in train_list:
        shutil.move(os.path.join(data_dir, "jpg", fname), os.path.join(train_dir, str(lab), fname))
    else:
        shutil.move(os.path.join(data_dir, "jpg", fname), os.path.join(test_dir, str(lab), fname))
