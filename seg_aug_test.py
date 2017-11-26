import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

from train_classifier import *
from train_segmenter import *
from utils import *
import random

import time
import csv

print(datagen.__dict__)

print('-'*30)
print('Loading and preprocessing train data for classifier...')
print('-'*30)

classifier_img_rows = 64
classifier_img_cols = 80

imgs_train, imgs_mask_train = load_train_data()
subject_ids = load_subject_ids()
subject_ids = subject_ids[:,0]

imgs_train = imgs_train.astype(np.uint8)

imgs_mask_train = imgs_mask_train.astype('float32')
imgs_mask_train /= 255

y = imgs_mask_train
y = y.reshape(y.shape[0],y.shape[2]*y.shape[3])
y = y.sum(axis=1)
y[np.where(y != 0)] = 1

subjects = np.unique(subject_ids)
print(np.sort(subjects))
train_subjects = random.sample(subjects,np.int(np.round(len(subjects) * 0.9)))
test_subjects = [x for x in subjects if x not in train_subjects]

train_inds = [x for x in range(0,len(subject_ids)) if subject_ids[x] in train_subjects]
test_inds = [x for x in range(0,len(subject_ids)) if subject_ids[x] in test_subjects]


### Train the classifier ##
clf_imgs = imgs_train.astype('float32')
clf_imgs = preprocess(clf_imgs,classifier_img_rows,classifier_img_cols)

X_train = clf_imgs[train_inds]
X_test = clf_imgs[test_inds]
y_train_clf = y[train_inds]
y_test_clf = y[test_inds]

clf_mean = np.mean(X_train)  # mean for data centering
clf_std = np.std(X_train)  # std for data normalization

X_train -= clf_mean
X_train /= clf_std

X_test -= clf_mean
X_test /= clf_std


## Train the segmenter ##
segmenter_img_rows = 64
segmenter_img_cols = 80

seg_imgs = imgs_train.astype('float32')
seg_imgs = preprocess(seg_imgs,segmenter_img_rows,segmenter_img_cols)


## Resizing makes this weird. You get values between -0.246 and 1.25
seg_masks = preprocess(imgs_mask_train,segmenter_img_rows,segmenter_img_cols)
seg_masks[np.where(seg_masks > 0)] = 1
seg_masks[np.where(seg_masks <= 0)] = 0
#seg_masks[np.where(seg_masks < 0.5)] = 0
X_train = seg_imgs[train_inds]
X_test = seg_imgs[test_inds]

y_train = seg_masks[train_inds]
y_test = seg_masks[test_inds]

non_zero_train = y_train.reshape(y_train.shape[0],y_train.shape[2]*y_train.shape[3])
non_zero_train = non_zero_train.sum(axis=1)
non_zero_train = np.where(non_zero_train != 0)

non_zero_test = y_test.reshape(y_test.shape[0],y_test.shape[2]*y_test.shape[3])
non_zero_test = non_zero_test.sum(axis=1)
non_zero_test = np.where(non_zero_test != 0)

#X_train = X_train[non_zero_train]
#X_test = X_test[non_zero_test]

#y_train =  y_train[non_zero_train]
#y_test = y_test[non_zero_test]

seg_mean = np.mean(X_train)  # mean for data centering
seg_std = np.std(X_train)  # std for data normalization

X_train -= seg_mean
X_train /= seg_std

X_test -= seg_mean
X_test /= seg_std

X_test_all = ( seg_imgs[test_inds] - seg_mean ) / seg_std

seg = segmenter_train_aug(X_train,y_train,(X_test,y_test), nb_epoch=2)
