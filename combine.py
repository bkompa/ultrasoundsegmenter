import numpy as np
import cv2

def naive_combine(image_probs,image_masks,p):
    final_masks = np.zeros_like(image_masks)
    non_zero =  np.where(image_probs > p)
    final_masks[non_zero,0] = image_masks[non_zero,0]
    return final_masks

def optimal_combine(image_probs,image_masks):
    final_masks = np.zeros_like(image_masks)
    for i in range(len(image_masks)):
        zero_score = 1 - image_probs[i]
        preds = np.round(image_masks[i,0])
        pred_score = image_probs[i]*(2.*((image_masks[i,0]*preds).sum())) / ( image_masks[i,0].sum() + preds.sum() )
        if pred_score > zero_score:
            final_masks[i,0] = image_masks[i,0]
    return final_masks

def element_wise_dice(y_true,y_pred):
    dice_values = np.zeros((len(y_true)))
    for i in range(len(y_true)):
        val = 2. * ( ( y_true[i,0] * y_pred[i,0] ).sum() ) / ( y_true[i,0].sum() + y_pred[i,0] )
        dice_values[i] = val
    return dice_values

def dice(y_true, y_pred):
    y_true_f = y_true.reshape(y_true.shape[0],y_true.shape[2]*y_true.shape[3])
    y_pred_f = y_pred.reshape(y_pred.shape[0],y_pred.shape[2]*y_pred.shape[3])
    intersection = (y_true_f*y_pred_f).sum()
    return (2. * intersection) / (y_true_f.sum() + y_pred_f.sum())

def preprocess(imgs,rows,cols):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], rows, cols))
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (cols, rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def load_train_data():
    imgs_train = np.load('/home/beam/projects/kaggle/ultrasound-nerve-segmentation/data/imgs_train.npy')
    imgs_mask_train = np.load('/home/beam/projects/kaggle/ultrasound-nerve-segmentation/data/imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_test_data():
    imgs_test = np.load('/home/beam/projects/kaggle/ultrasound-nerve-segmentation/data/imgs_test.npy')
    imgs_id = np.load('/home/beam/projects/kaggle/ultrasound-nerve-segmentation/data/imgs_id_test.npy')
    return imgs_test, imgs_id
