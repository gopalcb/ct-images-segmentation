"""
prepare image dataset and create batch for traning and validation
source: https://github.com/usuyama/pytorch-unet
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from random_selection import *
from image_processing.data_augmentation import *


# train and validation batch size
BATCH_SIZE = 25
ROOT = '/path/to/root'
ds_path = f'{ROOT}/ircad-dataset'


class ImageDataset(Dataset):
    def __init__(self, input_images, target_masks, transform):
        self.input_images = input_images
        self.target_masks = target_masks
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]
    

def get_train_val_data_loaders():
    """
    prepare data loaders of train and validation dataset
    return:
        dataloaders, image_datasets
    """
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    # read train and validation dataset
    train_images, train_masks, val_images, val_masks = read_random_training_and_validation_data()

    # apply image augmentation
    print('INFO: augment trainset')
    train_images, train_masks = apply_image_augmentation(train_images, train_masks)

    print('INFO: augment valset')
    val_images, val_masks = apply_image_augmentation(val_images, val_masks)

    train_set = ImageDataset(train_images, train_masks, trans)
    val_set = ImageDataset(val_images, val_masks, trans)

    # prepare dataset and data loader
    image_datasets = {
        'train': train_set, 'val': val_set
    }

    dataloaders = {
        'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    }
    print('INFO: data loader created')

    return dataloaders, image_datasets


def normalize_data(imgs_train):
    '''
    data normalization.
    
    params:
    imgs_train: array
    
    returns:
    imgs_train: normalized array
    '''
    # compute mean for data centering
    mean = np.mean(imgs_train)
    # compute std for data normalization
    std = np.std(imgs_train)

    # normalize train data
    imgs_train -= mean
    imgs_train /= std
    
    return imgs_train
