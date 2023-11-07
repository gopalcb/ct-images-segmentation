"""
"""
from __future__ import print_function
import os
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage import io
from unet.unet import *


def predict(imgs_test):
    '''
    this block tests the trained classifier.
    draw the predicted mask on the original test images.
    store the images with drawn masks.
    '''

    # compute mean for data centering
    mean = np.mean(imgs_test)
    # compute std for data normalization
    std = np.std(imgs_test)
    # normalize test data
    imgs_test -= mean
    imgs_test /= std

    # load saved weights
    model = build_unet_architecture()
    print('loading weights')
    model.load_weights('weights.h5')

    # predict masks on test data
    print('predicting masks...')
    imgs_mask_test = model.predict(imgs_test, verbose=1)

    # save prediction
    np.save('imgs_mask_test.npy', imgs_mask_test)
    print('saving prediction...')
    pred_dir = 'preds'

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for k in range(len(imgs_mask_test)):
        a = rescale_intensity(imgs_test[k][:,:,0],out_range=(-1,1))
        b = (imgs_mask_test[k][:,:,0]).astype('uint8')
        io.imsave(os.path.join(pred_dir, str(k) + '_pred.png'),mark_boundaries(a, b))