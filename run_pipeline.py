import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
import os 
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
train_subjects = random.sample(subjects,np.int(np.round(len(subjects) * 0.9)))
if 42 in train_subjects:
    train_subjects.remove(42)
test_subjects = [x for x in subjects if x not in train_subjects]
print('Test subjects',test_subjects)
train_inds = [x for x in range(0,len(subject_ids)) if subject_ids[x] in train_subjects]
test_inds = [x for x in range(0,len(subject_ids)) if subject_ids[x] in test_subjects]

for subject in test_subjects:
    subject_data = imgs_train[np.where(subject_ids==subject)]
    print('Test Subject '+str(int(subject))+' '+str(len(subject_data)))

for subject in train_subjects:
    subject_data = imgs_train[np.where(subject_ids==subject)]
    print('Train Subject '+str(int(subject))+' '+str(len(subject_data)))


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

print("Validation accuracy for predicting all-zeros: " + str(1.0 - y_test_clf.sum()/len(y_test_clf)))
clf = clf_train(X_train,y_train_clf,(X_test,y_test_clf),nb_epoch=2)
#clf.optimizer.lr.set_value(0.00001)
#clf_train_more(clf,X_train,y_train_clf,(X_test,y_test_clf),nb_epoch=100)
clf.save_weights('/home/ben/hst_summer/kaggle/vgg_weights.h5',overwrite=True)
test_probs = predict_over_augmentations(clf,X_test,y_test_clf,n_aug=20,evaluate=True)

## Train the segmenter ##
segmenter_img_rows = 64
segmenter_img_cols = 80

seg_imgs = imgs_train.astype('float32')
#seg_imgs = preprocess(seg_imgs,segmenter_img_rows,segmenter_img_cols)

seg_masks = imgs_mask_train
## Resizing makes this weird. You get values between -0.246 and 1.25
#seg_masks = preprocess(imgs_mask_train,segmenter_img_rows,segmenter_img_cols)
#seg_masks[np.where(seg_masks > 0)] = 1
#seg_masks[np.where(seg_masks <= 0)] = 0
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

seg = segmenter_train_crop(X_train,y_train,(X_test,y_test),nb_epoch=100)
seg.save_weights('/home/ben/hst_summer/kaggle/segment_weights.h5',overwrite=True)
test_masks = seg.predict(X_test_all,verbose=1)


## We need to do this in a more intelligent way that mirrors what thesubmission looks like ##
## The evaluation should be done on 420 x 580 predictions on the *original* masks ##

resize_row = 420
resize_col = 580

all_zeros = np.zeros_like(imgs_train[test_inds])
zero_dice = dice(imgs_mask_train[test_inds],all_zeros)
print("All zero-dice score: " + str(zero_dice))

score = dice(imgs_mask_train[test_inds],np.round(preprocess(test_masks,resize_row,resize_col)))
print("Dice using just the segmenter: " + str(score))

final_masks = naive_combine(test_probs,test_masks,p=0.5)
score = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
print("Naive Dice: " + str(score))

final_masks = optimal_combine(test_probs,test_masks)
score = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
print("Optimal Dice: " + str(score))


## Now predict on the unlableled data ##
imgs_test, imgs_id_test = load_test_data()
imgs_test_clf = preprocess(imgs_test,classifier_img_rows,classifier_img_cols)
imgs_test_clf -= clf_mean
imgs_test_clf /= clf_std
imgs_test_probs = predict_over_augmentations(clf,imgs_test_clf,None,n_aug=20,evaluate=False)

imgs_test_seg = preprocess(imgs_test,segmenter_img_rows,segmenter_img_cols)
imgs_test_seg -= seg_mean
imgs_test_seg /= seg_std

imgs_test_mask = seg.predict(imgs_test_seg,verbose=1)

final_test_masks = optimal_combine(imgs_test_probs,imgs_test_mask)
final_test_masks = preprocess(final_test_masks,resize_row,resize_col)

if os.path.isfile('/home/ben/hst_summer/kaggle/predictions_loop.npy'):
   predictions = np.load('/home/ben/hst_summer/kaggle/predictions_loop.npy')
   combine = np.add(predictions,final_test_masks)
   np.save('/home/ben/hst_summer/kaggle/predictions_loop.npy',combine)
if not os.path.isfile('/home/ben/hst_summer/kaggle/predictions_loop.npy'):
   np.save('/home/ben/hst_summer/kaggle/predictions_loop.npy',final_test_masks)

print('-'*30)
print('Doing patient wise validation error...')
print('-'*30)

dice_score_subjects = {}
subject_split = {}
classifier_acc = {}

clf_imgs = imgs_train.astype('float32')
clf_imgs = preprocess(clf_imgs,classifier_img_rows,classifier_img_cols)
clf_imgs -= np.mean(clf_imgs)
clf_imgs /= np.std(clf_imgs)
#validation for test subject
for subject in test_subjects:
    subject_split[int(subject)] = 'test'
    subject_data = imgs_train[np.where(subject_ids==subject)].astype('float32')
    subject_masks = imgs_mask_train[np.where(subject_ids==subject)]
    print('Validating testing subject '+str(subject)+' with '+str(len(subject_data))+' data points')
    subject_data_clf = subject_data
    subject_data_clf -= np.mean(subject_data)
    subject_data_clf /= np.std(subject_data)
    subject_data_clf = preprocess(subject_data_clf,classifier_img_rows,classifier_img_cols)
    subject_prob = clf.predict(subject_data_clf)

    subject_data_seg = subject_data
    subject_data_seg -= np.mean(subject_data)
    subject_data_seg /= np.std(subject_data)
    subject_data_seg = preprocess(subject_data_seg,segmenter_img_rows,segmenter_img_cols)
    subject_predicted_masks = seg.predict(subject_data_seg, verbose=1)
    subject_predicted_masks = np.round(preprocess(subject_predicted_masks,resize_row,resize_col))

    subject_final_masks = optimal_combine(subject_prob, subject_predicted_masks)
    subject_score = dice(subject_masks, subject_final_masks)
    print('Subject '+str(subject)+': '+str(subject_score))
    dice_score_subjects[int(subject)] = dice_score_subjects.get(int(subject),0)+subject_score

    clf_subject_img = clf_imgs[np.where(subject_ids==subject)]
    clf_subject_mask = y[np.where(subject_ids==subject)]
    loss, acc = clf.evaluate(clf_subject_img,clf_subject_mask,batch_size=32)
    classifier_acc[int(subject)] = acc


#validation for training subjects
for subject in train_subjects:
    subject_split[int(subject)] = 'train'
    subject_data = imgs_train[np.where(subject_ids==subject)].astype('float32')
    subject_masks = imgs_mask_train[np.where(subject_ids==subject)]
    print('Validating training subject '+str(subject)+' with '+str(len(subject_data))+' data points')
    subject_data_clf = subject_data
    subject_data_clf -= np.mean(subject_data)
    subject_data_clf /= np.std(subject_data)
    subject_data_clf = preprocess(subject_data_clf,classifier_img_rows,classifier_img_cols)
    subject_prob = clf.predict(subject_data_clf)

    subject_data_seg = subject_data
    subject_data_seg -= np.mean(subject_data)
    subject_data_seg /= np.std(subject_data)
    subject_data_seg = preprocess(subject_data_seg,segmenter_img_rows,segmenter_img_cols)
    subject_predicted_masks = seg.predict(subject_data_seg, verbose=1)
    subject_predicted_masks = np.round(preprocess(subject_predicted_masks,resize_row,resize_col))

    subject_final_masks = optimal_combine(subject_prob, subject_predicted_masks)
    subject_score = dice(subject_masks, subject_final_masks)
    print('Subject '+str(subject)+': '+str(subject_score))
    dice_score_subjects[int(subject)] = dice_score_subjects.get(int(subject),0)+subject_score

    clf_subject_img = clf_imgs[np.where(subject_ids==subject)]
    clf_subject_mask = y[np.where(subject_ids==subject)]
    loss, acc = clf.evaluate(clf_subject_img,clf_subject_mask,batch_size=32)
    classifier_acc[int(subject)] = acc

print('-'*30)
print('Writing to log file...')
print('-'*30)

timestr = time.strftime("%Y%m%d-%H%M%S")
sorted_subjects = sorted(dice_score_subjects)
subject_split['timestamp'] = timestr
classifier_acc['timestamp'] = timestr
dice_score_subjects['timestamp'] = dice_score_subjects.get('timestamp','')+timestr
fieldnames = []
fieldnames.append('timestamp')
for num in sorted_subjects:
    fieldnames.append(num)
with open('log5.csv', 'a') as csvfile:
    writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
    print(fieldnames)
    print(dice_score_subjects)
    writer.writeheader()
    writer.writerow(dice_score_subjects)
    writer.writerow(subject_split)
    writer.writerow(classifier_acc)
