"""
apply image augmentation - 
    L-R flip
    R-L flip
    B-U flip
    histogram equalization
    auto contrast
    image bluring
    image rotation
"""
import numpy as np
import copy


def apply_image_augmentation(np_images, np_masks):
    """
    apply image augmentation
    params:
        np_images: (nd-array) [slices, H, W]
        
    return:
        augmented_images, augmented_masks
    """
    augmented_images, augmented_masks = [], []
    np_images_ = copy.deepcopy(np_images)
    np_masks_ = copy.deepcopy(np_masks)
    
    for i in range(np_images_.shape[0]):
        np_image = np_images_[i, :, :]
        np_mask = np_masks_[i, :, :]
        
        arr = np.fliplr(np_image)  # flip image left to right
        augmented_images.append(arr)
        arr = np.fliplr(np_mask)  # flip mask left to right
        augmented_masks.append(arr)

        arr = np.flipud(np_image)  # flip image up down
        augmented_images.append(arr)
        arr = np.flipud(np_mask)  # flip mask up down
        augmented_masks.append(arr)

        # adjust image autocontrast 
        # pil_image = pimage.fromarray(np.uint8(np_image))
        # arr = iops.autocontrast(pil_image)
        # arr = np.array(arr)
        # augmented_images.append(arr)

        # # apply image histogram equalization
        # arr = iops.equalize(pil_image)
        # arr = np.array(arr)
        # augmented_images.append(arr)

        # # apply image gaussian blur
        # arr = pil_image.filter(ImageFilter.GaussianBlur(radius=1.5))
        # arr = np.array(arr)
        # augmented_images.append(arr)

        # rotate image - 90 deg
        # arr = pil_image.rotate(90, pimage.NEAREST, expand = 1)
        # arr = np.array(arr)
        # augmented_images.append(arr)

    augmented_images = np.array(augmented_images)
    augmented_masks = np.array(augmented_masks)
    print(f'INFO: total augmented images: {augmented_images.shape[0] - np_images.shape[0]}')
    return augmented_images, augmented_masks


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
        fig, ax = plt.subplots(1,3,figsize = (16,12))
        ax[0].imshow(image, cmap = 'gray')

        # mask visualization
        ax[1].imshow(mask, cmap = 'gray')

        # draw mask on top of original image
        ax[2].imshow(image, cmap = 'gray', interpolation = 'none')
        ax[2].imshow(mask, cmap = 'jet', interpolation = 'none', alpha = 0.7)


def plot_augmented_images(aug_images, aug_masks):
    '''
    plot augmented images and masks.
    
    params:
    aug_images: ndarray of train images
    aug_masks: ndarray of train masks
    '''
    
    # plot original images and masks vs. augmented images and masks
    plot_images([aug_images[0], aug_images[1]], [aug_masks[0], aug_masks[1]])
