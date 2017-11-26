import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

from train_classifier import *
from train_segmenter import *
from utils import *
import random

import time
import csv
import os


col_names = ["UNET Single 50 Zero","UNET Single 50 Seg Dice", "UNET Single 50 Naive Dice", "UNET Single 50 Optimal Dice","UNET Single 100 Zero","UNET Single 100 Seg Dice", "UNET Single 100 Naive Dice", "UNET Single 100 Optimal Dice","UNET Nonzero Single 50 Zero","UNET Nonzero Single 50 Seg Dice", "UNET Nonzero Single 50 Naive Dice", "UNET Nonzero Single 50 Optimal Dice","UNET Nonzero Single 100 Zero","UNET Nonzero Single 100 Seg Dice", "UNET Nonzero Single 100 Naive Dice", "UNET Nonzero Single 100 Optimal Dice"]
with open('seg_cv_log.csv', 'a') as csvfile:
    writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=col_names)
    writer.writeheader()

n_cv = 20
for cv in range(n_cv):

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
    clf = clf_train(X_train,y_train_clf,(X_test,y_test_clf),nb_epoch=50)
    #clf.optimizer.lr.set_value(0.00001)
    #clf_train_more(clf,X_train,y_train_clf,(X_test,y_test_clf),nb_epoch=100)
    clf.save_weights('/home/ben/hst_summer/kaggle/vgg_weights.h5',overwrite=True)
    test_probs = predict_over_augmentations(clf,X_test,y_test_clf,n_aug=20,evaluate=True)

    ## Train the segmenter ##
    segmenter_img_rows = 64
    segmenter_img_cols = 80

    seg_imgs = imgs_train.astype('float32')
    seg_imgs = preprocess(seg_imgs,segmenter_img_rows,segmenter_img_cols)

    seg_masks = imgs_mask_train
    ## Resizing makes this weird. You get values between -0.246 and 1.25
    seg_masks = preprocess(imgs_mask_train,segmenter_img_rows,segmenter_img_cols)
    seg_masks[np.where(seg_masks > 0)] = 1
    seg_masks[np.where(seg_masks <= 0)] = 0
    seg_masks[np.where(seg_masks < 0.5)] = 0
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

    seg = segmenter_train_aug(X_train,y_train,(X_test,y_test),nb_epoch=50)
    seg.save_weights('/home/ben/hst_summer/kaggle/segment_weights.h5',overwrite=True)
    test_masks = seg.predict(X_test_all,verbose=1)


    ## We need to do this in a more intelligent way that mirrors what thesubmission looks like ##
    ## The evaluation should be done on 420 x 580 predictions on the *original* masks ##

    resize_row = 420
    resize_col = 580

    print("\n----- Accuracy report after 50 epochs -----")
    all_zeros = np.zeros_like(imgs_train[test_inds])
    zero_dice_50 = dice(imgs_mask_train[test_inds],all_zeros)
    print("All zero-dice score: " + str(zero_dice_50))

    score_seg_50 = dice(imgs_mask_train[test_inds],np.round(preprocess(test_masks,resize_row,resize_col)))
    print("Dice using just the segmenter: " + str(score_seg_50))

    final_masks = naive_combine(test_probs,test_masks,p=0.5)
    score_naive_50 = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
    print("Naive Dice: " + str(score_naive_50))

    final_masks = optimal_combine(test_probs,test_masks)
    score_optimal_50 = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
    print("Optimal Dice: " + str(score_optimal_50))

    seg = segmenter_train_aug_more(X_train,y_train,(X_test,y_test),nb_epoch=50)
    seg.save_weights('/home/ben/hst_summer/kaggle/segment_weights.h5',overwrite=True)
    test_masks = seg.predict(X_test_all,verbose=1)

    print("\n----- Accuracy report after 100 epochs -----")
    all_zeros = np.zeros_like(imgs_train[test_inds])
    zero_dice_100 = dice(imgs_mask_train[test_inds],all_zeros)
    print("All zero-dice score: " + str(zero_dice_100))

    score_seg_100 = dice(imgs_mask_train[test_inds],np.round(preprocess(test_masks,resize_row,resize_col)))
    print("Dice using just the segmenter: " + str(score_seg_100))

    final_masks = naive_combine(test_probs,test_masks,p=0.5)
    score_naive_100 = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
    print("Naive Dice: " + str(score_naive_100))

    final_masks = optimal_combine(test_probs,test_masks)
    score_optimal_100 = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
    print("Optimal Dice: " + str(score_optimal_100))

    X_train = X_train[non_zero_train]
    X_test = X_test[non_zero_test]

    y_train =  y_train[non_zero_train]
    y_test = y_test[non_zero_test]

    seg_mean = np.mean(X_train)  # mean for data centering
    seg_std = np.std(X_train)  # std for data normalization

    X_train -= seg_mean
    X_train /= seg_std

    X_test -= seg_mean
    X_test /= seg_std

    X_test_all = ( seg_imgs[test_inds] - seg_mean ) / seg_std

    seg = segmenter_train_aug(X_train,y_train,(X_test,y_test),nb_epoch=50)
    seg.save_weights('/home/ben/hst_summer/kaggle/segment_weights.h5',overwrite=True)
    test_masks = seg.predict(X_test_all,verbose=1)


    ## We need to do this in a more intelligent way that mirrors what thesubmission looks like ##
    ## The evaluation should be done on 420 x 580 predictions on the *original* masks ##

    resize_row = 420
    resize_col = 580

    print("\n----- Accuracy report after 50 epochs -----")
    all_zeros = np.zeros_like(imgs_train[test_inds])
    zero_dice_nonzero50 = dice(imgs_mask_train[test_inds],all_zeros)
    print("All nonzero zero-dice score: " + str(zero_dice_nonzero50))

    score_seg_nonzero50 = dice(imgs_mask_train[test_inds],np.round(preprocess(test_masks,resize_row,resize_col)))
    print("Dice nonzero using just the segmenter: " + str(score_seg_nonzero50))

    final_masks = naive_combine(test_probs,test_masks,p=0.5)
    score_naive_nonzero50 = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
    print("Naive nonzero Dice: " + str(score_naive_nonzero50))

    final_masks = optimal_combine(test_probs,test_masks)
    score_optimal_nonzero50 = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
    print("Optimal nonzero Dice: " + str(score_optimal_nonzero50))

    seg = segmenter_train_aug_more(X_train,y_train,(X_test,y_test),nb_epoch=50)
    seg.save_weights('/home/ben/hst_summer/kaggle/segment_weights.h5',overwrite=True)
    test_masks = seg.predict(X_test_all,verbose=1)

    print("\n----- Accuracy report after 100 epochs -----")
    all_zeros = np.zeros_like(imgs_train[test_inds])
    zero_dice_nonzero100 = dice(imgs_mask_train[test_inds],all_zeros)
    print("All nonzero zero-dice score: " + str(zero_dice_nonzero100))

    score_seg_nonzero100 = dice(imgs_mask_train[test_inds],np.round(preprocess(test_masks,resize_row,resize_col)))
    print("Dice nonzero using just the segmenter: " + str(score_seg_nonzero100))

    final_masks = naive_combine(test_probs,test_masks,p=0.5)
    score_naive_nonzero100 = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
    print("Naive nonzero Dice: " + str(score_naive_nonzero100))

    final_masks = optimal_combine(test_probs,test_masks)
    score_optimal_nonzero100 = dice(imgs_mask_train[test_inds],np.round(preprocess(final_masks,resize_row,resize_col)))
    print("Optimal nonzero Dice: " + str(score_optimal_nonzero100))


    with open('seg_cv_log.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=col_names)
        out = dict(zip(col_names,[zero_dice_50,score_seg_50,score_naive_50,score_optimal_50,zero_dice_100,score_seg_100,score_naive_100,score_optimal_100,zero_dice_nonzero50,score_seg_nonzero50,score_naive_nonzero50,score_optimal_nonzero50,zero_dice_nonzero100,score_seg_nonzero100,score_naive_nonzero100,score_optimal_nonzero100]))
        writer.writerow(out)
