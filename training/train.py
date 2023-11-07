"""
"""

from __future__ import print_function
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
import os
import numpy as np
from unet.unet import *


# set hyperparameters
BATCH_SIZE = 10
EPOCHS = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2


def train_model(imgs_train, imgs_mask_train):
    '''
    this block compiles and fits the model.
    for each fold, it will perform the following operation.
    weights file name: weights.h5
    monitor: ['loss', 'accuracy', 'dice_coef']
    '''
    # kfold cross validation
    # k = 5
    kf = KFold(n_splits = 2, shuffle = False)

    # compute mean for data centering
    mean = np.mean(imgs_train)
    # compute std for data normalization
    std = np.std(imgs_train)

    # normalize train data
    imgs_train -= mean
    imgs_train /= std

    # save training history
    histories = []
    scores = []
    itr = 0
    for train_index, test_index in kf.split(imgs_train):
        itr += 1
        print(f'training for fold {itr}')
        
        X_train, X_test = imgs_train[train_index], imgs_train[test_index]
        y_train, y_test = imgs_mask_train[train_index], imgs_mask_train[test_index]
        
        # create and compile unet model
        print('compiling model...')
        model = build_unet_architecture()

        # save the weights and the loss of the best predictions
        model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

        # fit model
        print('fitting model...')
        history = model.fit(X_train,
                            y_train,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            shuffle=True,
                            validation_split=VALIDATION_SPLIT,
                            callbacks=[model_checkpoint])
        
        histories.append(history)

    return histories
