import os
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import pylab as pl

def import_tiffs():
    img_path = 'train/'
    training_path = os.path.join(os.getcwd(),img_path)
    images = os.listdir(training_path)
    num_images = len(images)/2
    rows=420
    cols=580
    i = 0
    img_array = np.ndarray((num_images,1,rows,cols), dtype=np.uint8)
    mask_array = np.ndarray((num_images,1,rows,cols), dtype=np.uint8)
    print('Loading tiff files... ')
    for image in images:
        if 'mask' in image:
            continue
        tiff_path = os.path.join(training_path,image)
        mask_path = tiff_path.replace('.tif','_mask.tif')
        tiff = plt.imread(tiff_path)
        mask = plt.imread(mask_path)
        img_array[i][0] = Image.fromarray(tiff,'L')
        mask_array[i][0] = Image.fromarray(mask,'L')
        i += 1
    np.save('imgs_train.npy', img_array)
    np.save('masks_train.npy', mask_array)
    print('Saving to .npy files done')
    return

def load_train_data():
    return np.load('imgs_train.npy'), mask_to_binary(np.load('masks_train.npy'))

def preprocess(array):
    dim1,dim2,dim3 = array.shape
    processed = np.ndarray(array.shape)
    for i in range(dim1):
        mu = np.mean(array[i])
        sigma = np.std(array[i])
        processed[i] = array[i]-mu
        processed[i] /= sigma
    return processed

def mask_to_binary(array):
    return array/255

def binary_to_mask(array):
    return array*255
