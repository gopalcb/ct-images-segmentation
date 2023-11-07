"""
ResNetUnet fully connected layer
the model uses pretrained resnet weight
source: https://github.com/usuyama/pytorch-unet
"""

from __future__ import print_function
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from training.metrics import *

import torch
import torch.nn as nn


# set hyperparameters
LEARNING_RATE = 1e-3


def build_unet_architecture(rows, cols):
    '''
    u-net model architecture.
    source: https://github.com/zhixuhao/unet
    
    returns:
    model: keras.model
    '''
    # layer 1
    # initial input shape 256*256
    # conv 3x3, ReLU
    # max pool 2x2
    inputs = Input((rows, cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # layer 2
    # conv 3x3, ReLU
    # max pool 2x2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # layer 3
    # conv 3x3, ReLU
    # max pool 2x2
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # layer 4
    # conv 3x3, ReLU
    # max pool 2x2
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # layer 5
    # conv 3x3, ReLU
    # copy and crop
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # layer 6
    # up-conv 2x2
    # conv 3x3, ReLU
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    # layer 7
    # up-conv 2x2
    # conv 3x3, ReLU
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    
    # layer 8
    # up-conv 2x2
    # conv 3x3, ReLU
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    
    # layer 9
    # up-conv 2x2
    # conv 3x3, ReLU
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=dice_coef_loss, metrics=['accuracy', dice_coef])

    return model



def convolution(inchannels, outchannels):
    conv_with_relu = nn.Sequential(
        nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, padding=1),
        # inplace=True means that it will modify the input directly, without allocating any additional output
        # it can sometimes slightly decrease the memory usage, but the original input is destroyed
        nn.ReLU(inplace=True)
    )

    return conv_with_relu


def double_convolution(inchannels, outchannels):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

    return dconv


class Unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder block / down sample
        self.encoder1 = self.double_convolution(inchannels=3, outchannels=64)
        self.encoder2 = self.double_convolution(inchannels=64, outchannels=128)
        self.encoder3 = self.double_convolution(inchannels=128, outchannels=256)
        self.encoder4 = self.double_convolution(inchannels=256, outchannels=512)
        self.encoder5 = self.double_convolution(inchannels=512, outchannels=1024)

        # decoder block / up sample
        # expand
        self.updeconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        # concatinate
        self.upconv1 = self.double_convolution(inchannels=1024, outchannels=512)

        self.updeconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.upconv2 = self.double_convolution(inchannels=512, outchannels=256)

        self.updeconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upconv3 = self.double_convolution(inchannels=256, outchannels=128)

        self.updeconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upconv4 = self.double_convolution(inchannels=128, outchannels=64)

        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)


    def double_convolution(self, inchannels, outchannels):
        double_conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
        return double_conv


    def forward(self, x):
        # apply encoding
        encode1 = self.encoder1(x)
        maxpool1 = self.max_pool2d(encode1)

        encode2 = self.encoder2(maxpool1)
        maxpool2 = self.max_pool2d(encode2)

        encode3 = self.encoder3(maxpool2)
        maxpool3 = self.max_pool2d(encode3)

        encode4 = self.encoder4(maxpool3)
        maxpool4 = self.max_pool2d(encode4)

        encode5 = self.encoder5(maxpool4)

        # apply decoding
        decode1 = self.updeconv1(encode5)
        decode1_ = self.upconv1(torch.cat([encode4, decode1], 1))

        decode2 = self.updeconv2(decode1_)
        decode2_ = self.upconv2(torch.cat([encode3, decode2], 1))

        decode3 = self.updeconv3(decode2_)
        decode3_ = self.upconv3(torch.cat([encode2, decode3], 1))

        decode4 = self.updeconv4(decode3_)
        decode4_ = self.upconv4(torch.cat([encode1, decode4], 1))
        
        # output layer
        output = self.output(decode4_)

        return output
