"""
"""
import pickle
import matplotlib.pyplot as plt
import imageio
import os
import numpy as np


ROOT = '/path/to/root'
out_path = f'{ROOT}/display_preds'


def plot_histories(histories):
    '''
    plot training and validation loss, accuracy, and dice score.
    the histories list contains all folds training history.
    '''

    # show plots for each fold
    for h, history in enumerate(histories):
        # get metrics keys
        keys = history.history.keys()
        fig, axs = plt.subplots(1, len(keys)//2, figsize = (25, 5))
        fig.suptitle('No. ' + str(h+1) + ' Fold Results', fontsize=30)

        for k, key in enumerate(list(keys)[:len(keys)//2]):
            training = history.history[key]
            validation = history.history['val_' + key]

            epoch_count = range(1, len(training) + 1)

            axs[k].plot(epoch_count, training, 'r--')
            axs[k].plot(epoch_count, validation, 'b-')
            axs[k].legend(['Training ' + key, 'Validation ' + key])
                    
        with open(str(h+1) + '_lungs_trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


def display_segmented_images():
    '''
    plot predicted images.
    display 9 images in a 3*3 grid
    '''

    # get all file names in display_preds dir
    image_names = os.listdir(out_path)
    # convert to numpy array
    image_names=np.asarray(image_names)
    # reshape the array into 3*3
    image_names = image_names.reshape(3, 3).T 

    for names in image_names:
        # read .png images into numpy array
        image1 = imageio.imread(f'{out_path}/{names[0]}')
        image2 = imageio.imread(f'{out_path}/{names[1]}')
        image3 = imageio.imread(f'{out_path}/{names[2]}')

        # show image 1
        fig, ax = plt.subplots(1,3,figsize = (16,12))
        ax[0].imshow(image1, cmap = 'gray')

        # show image 2
        ax[1].imshow(image2, cmap = 'gray')

        # show image 3
        ax[2].imshow(image3, cmap = 'gray')
