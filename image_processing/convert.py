"""
author @gopal
extract images and masks from gz files and store them separately
"""

import copy
import os
import numpy as np
import nibabel
from PIL import Image as pimage
from PIL import ImageOps as iops
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = '/Users/gopalcbala/Desktop/Jupyter_NB_Projects/PROJECTS/IRCAD/Untitled/ct-images-semantic-segmentation'
ds_path = f'{ROOT}/ircad-dataset'


def apply_dataset_masking():
    """
    split up original image and mask and store them separately
    params:
        none
    return:
        bool
    """
    # load all original image gz files
    orig_gz_files = [f for f in os.listdir(ds_path) if '_orig' in f]
    print(f'total loaded image file: {len(orig_gz_files)}')

    # traverse and store images and masks
    print('INFO: splitting images and masks..')
    for index, file in enumerate(orig_gz_files):
        print(f'INFO: processing - {file}')
        images = nibabel.load(f'{ds_path}/{file}')
        images = images.get_fdata()  # get pixels as nd array

        # extract original image name - to be used for mask naming
        common_name = file.split('.')[0].replace('orig', '')
        masks = nibabel.load(f'{ds_path}/{common_name}liver.nii.gz')
        masks = masks.get_fdata()  # get mask pixels as nd array
        # print(images.shape)
        # print(masks.shape)

        # create file wise directory if not exists
        container_dir = f'{ROOT}/dataset/images_and_masks/e{index+1}'
        if not os.path.exists(container_dir):
            os.makedirs(container_dir)

        # traverse each slice and store original images and masks
        for i in range(images.shape[2]):
            image = images[:, :, i]  # get ith slice
            # print(image.shape)
            image_path = f'{ROOT}/dataset/images_and_masks/e{index+1}/{file.split(".")[0]}_{i}.jpg'
            pimage.fromarray(np.uint8(image)).save(image_path)  # save image

            if i < masks.shape[2]:
                mask = masks[:, :, i]  # get ith slice
                image[mask == 0] = 0
                # print(image_cp.shape)
                mask_path = f'{ROOT}/dataset/images_and_masks/e{index+1}/{common_name}liver_mask_{i}.jpg'
                pimage.fromarray(np.uint8(image)).save(mask_path)  # save mask

    print('INFO: processing complete')
    return True