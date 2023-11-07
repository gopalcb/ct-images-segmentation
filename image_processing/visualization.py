"""
image dataset visualization using matplotlib
each row in the visualization set contains 3 images
    1. orig image 2. mask 3. image and mask combined
"""

# import libraries
import os
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import traceback

ROOT = '/path/to/root'
def plot_images(images, masks):
    '''
    plot images, masks, and combined images and masks.
    params:
        images: ndarray
        masks: ndarray
    '''
    for i in range(len(images)):
        image, mask = images[i], masks[i]
        
        # original image visualization
        fig, ax = plt.subplots(1,3,figsize = (13,11))
        ax[0].imshow(image, cmap = 'gray')
        ax[0].axis('off')

        # mask visualization
        ax[1].imshow(mask, cmap = 'gray')
        ax[1].axis('off')

        # draw mask on top of original image
        ax[2].imshow(image, cmap = 'gray', interpolation = 'none')
        ax[2].imshow(mask, cmap = 'jet', interpolation = 'none', alpha = 0.7)
        ax[2].axis('off')
