from __future__ import print_function
from keras.optimizers import Adam


from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, merge,Activation,UpSampling2D


from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import *

import cv2, numpy as np
import math

from mask_flow import *

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))


def segmenter_train(X_train,y_train,validation_data):
    image_shape = (1,X_train.shape[2],X_train.shape[3])
    model = UNET(image_shape)
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])
    model.fit(X_train, y_train, batch_size=64,
                        validation_data=validation_data, nb_epoch=20)
    return model

def segmenter_train_aug(X_train, y_train, validation_data, nb_epoch=20):
    image_shape = (1,X_train.shape[2], X_train.shape[3])
    model = UNET(image_shape)
    opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])

    batch_size = 8
    for i in range(nb_epoch):
        print('Epoch '+str(i+1)+'/'+str(nb_epoch))
        progbar = Progbar(X_train.shape[0])
        for j in range(int(math.ceil(float(X_train.shape[0])/batch_size))):
            #def flow(X_train, Y_train, batches_done, batch_size,rotation_range, height_shift_range, width_shift_range, shear_range,
            #zoom_range, horizontal_flip, vertical_flip, standardize):
            batch_X, batch_y = flow(X_train, y_train, j, batch_size, 10, .1, .1, .05, (.95,1.05), True, True, False)
            history = model.train_on_batch(batch_X,batch_y)
            progbar.add(min(X_train.shape[0]-batch_size,batch_size))
        print(model.evaluate(validation_data[0],validation_data[1],batch_size))
    return model

#[10, .1, .1, .05, .95,1.05, True, True, False]
def segmenter_train_aug_elastic(X_train, y_train, validation_data, nb_epoch=20):
    image_shape = (1,X_train.shape[2], X_train.shape[3])
    model = UNET(image_shape)
    opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])

    batch_size = 32
    for i in range(nb_epoch):
        print('Epoch '+str(i+1)+'/'+str(nb_epoch))
        progbar = Progbar(X_train.shape[0])
        for j in range(int(math.ceil(float(X_train.shape[0])/batch_size))):
            #def flow(X_train, Y_train, batches_done, batch_size,rotation_range, height_shift_range, width_shift_range, shear_range,
            #zoom_range, horizontal_flip, vertical_flip, standardize):
            batch_X, batch_y = flow_elastic(X_train, y_train, j, batch_size, 0, 0, 0, 0, (1,1), False, False, False)
            history = model.train_on_batch(batch_X,batch_y)
            progbar.add(min(X_train.shape[0]-batch_size,batch_size))
        print(model.evaluate(validation_data[0],validation_data[1],batch_size))
    return model

def segmenter_train_aug_more(model, X_train, y_train, validation_data, nb_epoch=20):
    batch_size = 32
    for i in range(nb_epoch):
        print('Epoch '+str(i+1)+'/'+str(nb_epoch))
        progbar = Progbar(X_train.shape[0])
        for j in range(int(math.ceil(float(X_train.shape[0])/batch_size))):
            #def flow(X_train, Y_train, batches_done, batch_size,rotation_range, height_shift_range, width_shift_range, shear_range,
            #zoom_range, horizontal_flip, vertical_flip, standardize):
            batch_X, batch_y = flow(X_train, y_train, j, batch_size, 10, .1, .1, .05, (.95,1.05), True, True, False)
            model.train_on_batch(batch_X, batch_y)
            progbar.add(min(X_train.shape[0]-batch_size,batch_size))
        print(model.evaluate(validation_data[0],validation_data[1],batch_size))
    return model

def segmenter_train_crop(X_train, y_train, validation_data, nb_epoch=20):
    image_shape = (1,96,96)
    model = UNET(image_shape)
    opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])

    batch_size = 32

    #crop the validation data
    crop_valid_X = np.zeros((validation_data[0].shape[0],1,96,96))
    crop_valid_Y = np.zeros((validation_data[1].shape[0],1,96,96))
    for i in range(validation_data[0].shape[0]):
        row = np.random.randint(0,validation_data[0].shape[2]-96)
        col = np.random.randint(0,validation_data[0].shape[3]-96)
        crop_valid_X[i] = crop(validation_data[0][i],row,col)
        crop_valid_Y[i] = crop(validation_data[1][i],row,col)
    for i in range(nb_epoch):
	print('Epoch '+str(i+1)+'/'+str(nb_epoch))
        progbar = Progbar(X_train.shape[0])
        for j in range(int(math.ceil(float(X_train.shape[0])/batch_size))):
            #def flow(X_train, Y_train, batches_done, batch_size,rotation_range, height_shift_range, width_shift_range, shear_range,
            #zoom_range, horizontal_flip, vertical_flip, standardize):
            #batch_X, batch_y = flow_crop(X_train, y_train, j, batch_size, 10, .1, .1, 0, (.9,1.1), True, True, False)
            batch_X, batch_y = flow_crop(X_train, y_train, j, batch_size, 0, 0, 0, 0, (1,1), False, False, False)
	    history = model.train_on_batch(batch_X, batch_y)
            print('Training stats', history)
            progbar.add(min(X_train.shape[0]-batch_size,batch_size))
        print(model.evaluate(crop_valid_X,crop_valid_Y,batch_size))
    return model

def UNET(image_shape):
    bn_mode = 0
    bn_axis = 1
    activation = 'linear'
    inputs = Input(image_shape)
    conv1 = Convolution2D(32, 3, 3, activation=activation, border_mode='same')(inputs)
    conv1 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Convolution2D(32, 3, 3, activation=activation, border_mode='same')(conv1)
    conv1 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Convolution2D(64, 3, 3, activation=activation, border_mode='same')(pool1)
    conv2 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Convolution2D(64, 3, 3, activation=activation, border_mode='same')(conv2)
    conv2 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Convolution2D(128, 3, 3, activation=activation, border_mode='same')(pool2)
    conv3 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Convolution2D(64, 3, 3, activation=activation, border_mode='same')(conv3)
    conv3 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Convolution2D(128, 3, 3, activation=activation, border_mode='same')(conv3)
    conv3 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Convolution2D(256, 3, 3, activation=activation, border_mode='same')(pool3)
    conv4 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Convolution2D(256, 3, 3, activation=activation, border_mode='same')(conv4)
    conv4 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Convolution2D(512, 3, 3, activation=activation, border_mode='same')(pool4)
    conv5 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Convolution2D(512, 3, 3, activation=activation, border_mode='same')(conv5)
    conv5 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(p=0.5)(conv5)
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation=activation, border_mode='same')(up6)
    #conv6 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Convolution2D(256, 3, 3, activation=activation, border_mode='same')(conv6)
    #conv6 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv6)
    conv6 = Activation('relu')(conv6)
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = Dropout(p=0.5)(up7)
    conv7 = Convolution2D(128, 3, 3, activation=activation, border_mode='same')(up7)
    #conv7 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Convolution2D(128, 3, 3, activation=activation, border_mode='same')(conv7)
    #conv7 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv7)
    conv7 = Activation('relu')(conv7)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = Dropout(p=0.5)(up8)
    conv8 = Convolution2D(64, 3, 3, activation=activation, border_mode='same')(up8)
    #conv8 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Convolution2D(64, 3, 3, activation=activation, border_mode='same')(conv8)
    #conv8 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv8)
    conv8 = Activation('relu')(conv8)
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = Dropout(p=0.5)(up9)
    conv9 = Convolution2D(32, 3, 3, activation=activation, border_mode='same')(up9)
    #conv9 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Convolution2D(32, 3, 3, activation=activation, border_mode='same')(conv9)
    #conv9 = BatchNormalization(mode=bn_mode, axis=bn_axis)(conv9)
    conv9 = Activation('relu')(conv9)
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    model = Model(input=inputs, output=conv10)
    print(model.summary())
    return model

count=0
for i in range(0,len(cut)):
    if(cut[i,2]>non_aug[i,2]):
        count+=1
        print(cut[i,:2])

results = {}

#no data aug
arr = benchmark_cutoffs(preprocess(test_masks,420,580), imgs_mask_train[test_inds], verbose=False)
results[0] = results.get(0,0)+np.max(arr[:,2])

for i in [5, 10, 20, 50, 100, 500]:
    print('Augmenting {} times'.format(i))
    preds = predict_over_aug(X_test, seg, i, [10, .1, .1, .05, .95,1.05, True, True, False])
    arr = benchmark_cutoffs(preprocess(preds,420,580), imgs_mask_train[test_inds], verbose=False)
    results[i] = results.get(i,0)+np.max(arr[:,2])
