"""
"""

from __future__ import print_function
from keras import backend as K
import os
import numpy as np



def dice_coef(y_true, y_pred, smooth = 1.0):
    '''
    compute dice coefficient given GT and prediction.
    
    params:
    y_true: array; ground truths
    y_pred: array; prediction
    smooth: float
    
    returns:
    dice_score: float
    '''
    # flatten GT and prediction
    y = K.flatten(y_true)
    y_hat = K.flatten(y_pred)
    
    # find overlap/intersection
    intersection = K.sum(y * y_hat)
    
    # compute dice score
    dice_score = (2. * intersection + smooth) / (K.sum(y) + K.sum(y_hat) + smooth)
    
    return dice_score


def dice_coef_loss(y, y_pred):
    '''
    compute dice coefficient loss given GT and prediction.
    
    params:
    y: array; ground truths
    y_pred: array; prediction
    
    returns:
    loss: float
    '''
    return 1 - dice_coef(y, y_pred)