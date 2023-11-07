"""
read training and validation sets
author @gopal
"""

import os
import numpy as np
import nibabel
import random


ROOT = '/path/to/root'
ds_path = f'{ROOT}/ircad-dataset'


def extract_images_and_masks(files):
    images_, masks_ = [], []
    for index, file in enumerate(files):
        images = nibabel.load(f'{ds_path}/{file}')
        images = images.get_fdata()  # get pixels as nd array

        # extract original image name - to be used for mask naming
        common_name = file.split('.')[0].replace('orig', '')
        masks = nibabel.load(f'{ds_path}/{common_name}liver.nii.gz')
        masks = masks.get_fdata()  # get mask pixels as nd array

        # traverse each slice and store original images and masks
        for i in range(masks.shape[2]):
            image = images[:, :, i]  # get ith image slice
            mask = masks[:, :, i]  # get ith mask slice
            if np.sum(mask) > 0:
                images_.append(image)
                masks_.append(mask)

    images_, masks_ = np.array(images_), np.array(masks_)
    return images_, masks_


def read_random_training_and_validation_data():
    """
    read trainingn and validation dataset randomly
    out of 16 sets, use 10 sets for training and 6 sets for validation 
    return:
        train_set, val_set
    """
    # load all original image gz files
    orig_gz_files = [f for f in os.listdir(ds_path) if '_orig' in f]
    # randomly select 10 sets
    train_files = random.choices(orig_gz_files, k=10)
    print(f'INFO: randomly selected train files: {train_files}')
    val_files = [f for f in orig_gz_files if f not in train_files]
    print(f'INFO: randomly selected val files: {val_files}')

    # prepare trainset
    print('INFO: reading trainset..')
    train_images, train_masks = extract_images_and_masks(train_files)
    print('INFO: trainset reading complete')

    # prepare valset
    print('INFO: reading valset..')
    val_images, val_masks = extract_images_and_masks(val_files)
    print('INFO: valset reading complete')

    return train_images, train_masks, val_images, val_masks
