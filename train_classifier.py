from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Activation, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils.generic_utils import Progbar
from resnet import *

datagen = ImageDataGenerator(
        rotation_range = 10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True)

def clf_train(X_train,y_train,validation_data,nb_epoch=30,MODEL_TYPE=RESNET):
    image_shape = (1,X_train.shape[2],X_train.shape[3])
    #model = VGG_16(image_shape)
    model = MODEL_TYPE(image_shape)
    opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=opt)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=validation_data,
                        samples_per_epoch=len(X_train),
                        nb_epoch=nb_epoch)
    return model

def clf_train_more(model,X_train,y_train,validation_data,nb_epoch):
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=validation_data,
                        samples_per_epoch=len(X_train),
                        nb_epoch=nb_epoch)
    return model

def train_vs_test(train_preds,test_preds,nb_epoch=25):
    X = np.concatenate((train_preds,test_preds))
    y = np.zeros((len(X),))
    y[len(train_preds):] = 1
    image_shape = (1,X.shape[2],X.shape[3])
    model = VGG_16(image_shape)
    #model = resnet(image_shape)
    opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=opt)
    model.fit(X, y, batch_size=32,validation_split=0.2,nb_epoch=nb_epoch)

def predict_over_augmentations(model,X,y,n_aug=10,evaluate=True):
    progbar = Progbar(n_aug)  # progress bar for pre-processing status tracking
    probs = np.zeros(len(X),)
    gen_iter = datagen.flow(X,batch_size = len(X),shuffle = False)
    for i in range(n_aug):
        progbar.add(1)
        X_aug = gen_iter.next()
        preds = model.predict(X_aug)
        probs += preds[:,0]
    probs /= np.float(n_aug)
    if evaluate:
        pred_classes = np.round(probs)
        accuracy = 1.0 - (np.abs(y - pred_classes).sum())/len(y)
        print("Accuracy: " + str(accuracy))
    return probs

def VGG_16(image_shape):
    bn_mode = 0
    bn_axis = 1
    decay = 1e-5
    activation = 'linear'
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=image_shape))
    model.add(Convolution2D(64, 3, 3,activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation=activation,W_regularizer=l2(decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
