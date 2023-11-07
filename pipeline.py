"""
pipeline py file
    contains all sequential steps
"""

from unet.random_selection import *
from image_processing.convert import *
from image_processing.data_augmentation import *
from training.train import *
from testing.predict import *
from testing.output import *
from testing.output import *


def pipeline():
    """
    step 1: apply dataset masking
        extract the CT scan images and masks
        store original images and masks separately
    """
    apply_dataset_masking()
    

    """
    step 2: load train and validation images and masks
        use numpy to load the train and validation dataset
        the process of dataset selection is random
    """
    train_images, train_masks, val_images, val_masks = read_random_training_and_validation_data()


    """
    step 3: image augmentation
        increase the number of sample by applying image augmentation
        currently implemented augmentation techniques
            left to right flip
            top to bottom flip
            image rotation
            gaussian blur
    """
    train_images, train_masks = apply_image_augmentation(train_images, train_masks)
    val_images, val_masks = apply_image_augmentation(val_images, val_masks)


    """
    step 4: train segmentation model
        backbone network: unet
    """
    history = train_model(train_images, train_masks)


    """
    step 5: load and test the trained model
        apply prediction on validation images
    """
    predict(val_images)


    """
    step 6: results and plotting
        plot training and validation loss, accuracy, and dice score.
        the histories list contains all folds training history.
    """
    plot_histories(histories)

    """
    plot training and validation loss, accuracy, and dice score.
    the histories list contains all folds training history.
    """
    display_segmented_images()