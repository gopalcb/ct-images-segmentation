## Project Title:
### Semantic segmentation of Liver organ in medical CT scan images using Unet as the backend architecture
<hr>

<ul>
    <li>Automatically segment livers using the U-net architecture.</li>
    <li>Data source: <a href="https://www.dropbox.com/s/8h2avwtk8cfzl49/ircad-dataset.zip?dl=0">Data download URL</a></li>
    <li>Nibabel python library is used to load the NifTi (Neuroimaging Informatics Technology Initiative) data.</li>
    <li>The dataset consists of 20 medical examinations in 3D.</li>
    <li>Each original image has its corresponding binary mask.</li>
</ul>

The following diagram shows the overall U-net architecture.

![png](display_preds/img-unet.png)

#### Pipeline Steps:

<ol>
    <li>Images and masks extraction</li>
    <li>Load training and validation images and masks</li>
    <li>Image augmentation</li>
    <li>Training segmentation model</li>
    <li>Load and test the model</li>
    <li>Results and plotting</li>
</ol>

### Project Structure:

.
├── README.md
├── config.py
├── dataset
│   └── images_and_masks
├── deployment
├── display_preds
├── image_processing
│   ├── convert.py
│   ├── data_augmentation.py
│   └── visualization.py
├── ircad-dataset
├── ircad.ipynb
├── model_serving_api
├── models
├── pipeline.py
├── preds
├── testing
│   ├── output.py
│   └── predict.py
├── training
│   ├── metrics.py
│   └── train.py
└── unet
    ├── data_loader.py
    ├── random_selection.py
    └── unet.py

### Hyperparameters:


```python
# Hyperparameters
BATCH_SIZE = 10
EPOCHS = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
```

### Image data visualization:


```python
'''
set data paths.
root data dir: data
'''
ROOT = '/Users/gopalcbala/Desktop/Jupyter_NB_Projects/PROJECTS/IRCAD/Untitled/ct-images-semantic-segmentation'
ds_path = f'{ROOT}/ircad-dataset'
```


```python
'''
view data shape.
both images and masks will have the same shape.
only masks shape is shown here.
'''
# load a mask/image to view shape
train_masks = nibabel.load(f'{train_data_path}/ircad_e01_liver.nii.gz')
# get 3d numpy array
train_masks = train_masks.get_data()
print(f'mask shape: {train_masks.shape}')
```

    (512, 512, 129)



```python
'''
train image and mask visualization.
slice 72 in the train image and mask has better/full view.
'''
from image_processing.visualization import *

train_mask = nibabel.load(f'{train_data_path}/ircad_e01_liver.nii.gz')
train_mask = train_mask.get_data()

train_image = nibabel.load(f'{train_data_path}/ircad_e01_orig.nii.gz')
train_image = train_image.get_data()

# pick a slice (72)
image = train_image[:, :, 72]
mask = train_mask[:, :, 72]

# plot an image, mask, and combined
plot_images([image], [mask])
```

    /var/folders/cg/h7_fl0497c7dyn_wxbcjdl4c0000gn/T/ipykernel_4244/544972443.py:6: DeprecationWarning: get_data() is deprecated in favor of get_fdata(), which has a more predictable return type. To obtain get_data() behavior going forward, use numpy.asanyarray(img.dataobj).
    
    * deprecated from version: 3.0
    * Will raise <class 'nibabel.deprecator.ExpiredDeprecationError'> as of version: 5.0
      train_mask = train_mask.get_data()
    /var/folders/cg/h7_fl0497c7dyn_wxbcjdl4c0000gn/T/ipykernel_4244/544972443.py:9: DeprecationWarning: get_data() is deprecated in favor of get_fdata(), which has a more predictable return type. To obtain get_data() behavior going forward, use numpy.asanyarray(img.dataobj).
    
    * deprecated from version: 3.0
    * Will raise <class 'nibabel.deprecator.ExpiredDeprecationError'> as of version: 5.0
      train_image = train_image.get_data()



    
![png](display_preds/output_8_1.png)
    


### 1. Images and masks extraction:


```python
from unet.random_selection import *

apply_dataset_masking()
```

    INFO: splitting images and masks
    INFO: process complete


### 2. Load train and validation images and masks:


```python
from image_processing.convert import *

train_images, train_masks, val_images, val_masks = read_random_training_and_validation_data()
```

    INFO: randomly selected train files: ['ircad_e14_orig.nii.gz', 'ircad_e06_orig.nii.gz', 'ircad_e05_orig.nii.gz', 'ircad_e06_orig.nii.gz', 'ircad_e01_orig.nii.gz', 'ircad_e16_orig.nii.gz', 'ircad_e06_orig.nii.gz', 'ircad_e10_orig.nii.gz', 'ircad_e01_orig.nii.gz', 'ircad_e07_orig.nii.gz']
    INFO: randomly selected val files: ['ircad_e08_orig.nii.gz', 'ircad_e11_orig.nii.gz', 'ircad_e04_orig.nii.gz', 'ircad_e18_orig.nii.gz', 'ircad_e17_orig.nii.gz', 'ircad_e02_orig.nii.gz', 'ircad_e19_orig.nii.gz', 'ircad_e15_orig.nii.gz', 'ircad_e09_orig.nii.gz', 'ircad_e13_orig.nii.gz']
    INFO: reading trainset..
    INFO: trainset reading complete
    INFO: reading valset..
    INFO: valset reading complete


### 3. Image augmentation:


```python
from image_processing.data_augmentation import *

train_images, train_masks = apply_image_augmentation(train_images, train_masks)
val_images, val_masks = apply_image_augmentation(val_images, val_masks)
```

    INFO: augment trainset
    INFO: trainset augmentation complete
    INFO: augment validation set
    INFO: validation set augmentation complete
    


### 4. Training segmentation model:


```python
from training.train import *

history = train_model(train_images, train_masks)
```

    training for fold 1
    compiling model...
    fitting model...
    Epoch 1/50
    2021-12-21 18:26:45.618382: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
    58/58 [==============================] - ETA: 0s - loss: 0.3608 - accuracy: 0.9682 - dice_coef: 0.6402
    2021-12-21 18:27:23.655058: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
    58/58 [==============================] - 42s 701ms/step - loss: 0.3608 - accuracy: 0.9682 - dice_coef: 0.6402 - val_loss: 0.3293 - val_accuracy: 0.9743 - val_dice_coef: 0.6491
    Epoch 2/50
    58/58 [==============================] - 38s 660ms/step - loss: 0.0958 - accuracy: 0.9892 - dice_coef: 0.9044 - val_loss: 0.4429 - val_accuracy: 0.9689 - val_dice_coef: 0.5385
    Epoch 3/50
    58/58 [==============================] - 38s 652ms/step - loss: 0.0820 - accuracy: 0.9909 - dice_coef: 0.9178 - val_loss: 0.5201 - val_accuracy: 0.9670 - val_dice_coef: 0.4639
    Epoch 4/50
    58/58 [==============================] - 38s 659ms/step - loss: 0.0707 - accuracy: 0.9919 - dice_coef: 0.9295 - val_loss: 0.3744 - val_accuracy: 0.9766 - val_dice_coef: 0.6053
    Epoch 5/50
    58/58 [==============================] - 37s 644ms/step - loss: 0.0655 - accuracy: 0.9926 - dice_coef: 0.9346 - val_loss: 0.2618 - val_accuracy: 0.9824 - val_dice_coef: 0.7139
    Epoch 6/50
    58/58 [==============================] - 37s 636ms/step - loss: 0.0542 - accuracy: 0.9938 - dice_coef: 0.9457 - val_loss: 0.2883 - val_accuracy: 0.9786 - val_dice_coef: 0.6894
    Epoch 7/50
    58/58 [==============================] - 37s 642ms/step - loss: 0.0479 - accuracy: 0.9945 - dice_coef: 0.9517 - val_loss: 0.2140 - val_accuracy: 0.9835 - val_dice_coef: 0.7615
    Epoch 8/50
    58/58 [==============================] - 37s 639ms/step - loss: 0.0504 - accuracy: 0.9942 - dice_coef: 0.9496 - val_loss: 0.5240 - val_accuracy: 0.9659 - val_dice_coef: 0.4601
    Epoch 9/50
    58/58 [==============================] - 37s 635ms/step - loss: 0.0453 - accuracy: 0.9949 - dice_coef: 0.9548 - val_loss: 0.3427 - val_accuracy: 0.9789 - val_dice_coef: 0.6378
    Epoch 10/50
    58/58 [==============================] - 37s 634ms/step - loss: 0.0390 - accuracy: 0.9956 - dice_coef: 0.9610 - val_loss: 0.2443 - val_accuracy: 0.9832 - val_dice_coef: 0.7334
    Epoch 11/50
    58/58 [==============================] - 37s 637ms/step - loss: 0.0523 - accuracy: 0.9941 - dice_coef: 0.9476 - val_loss: 0.6009 - val_accuracy: 0.9594 - val_dice_coef: 0.3858
    Epoch 12/50
    58/58 [==============================] - 37s 636ms/step - loss: 0.0589 - accuracy: 0.9935 - dice_coef: 0.9412 - val_loss: 0.4885 - val_accuracy: 0.9670 - val_dice_coef: 0.5121
    Epoch 13/50
    58/58 [==============================] - 37s 635ms/step - loss: 0.0478 - accuracy: 0.9946 - dice_coef: 0.9523 - val_loss: 0.3815 - val_accuracy: 0.9753 - val_dice_coef: 0.5998
    Epoch 14/50
    58/58 [==============================] - 37s 632ms/step - loss: 0.0386 - accuracy: 0.9956 - dice_coef: 0.9614 - val_loss: 0.3003 - val_accuracy: 0.9791 - val_dice_coef: 0.6764
    Epoch 15/50
    58/58 [==============================] - 37s 637ms/step - loss: 0.0331 - accuracy: 0.9962 - dice_coef: 0.9669 - val_loss: 0.2743 - val_accuracy: 0.9770 - val_dice_coef: 0.7039
    Epoch 16/50
    58/58 [==============================] - 37s 635ms/step - loss: 0.0296 - accuracy: 0.9966 - dice_coef: 0.9705 - val_loss: 0.2876 - val_accuracy: 0.9808 - val_dice_coef: 0.6903
    Epoch 17/50
    58/58 [==============================] - 37s 633ms/step - loss: 0.0268 - accuracy: 0.9969 - dice_coef: 0.9731 - val_loss: 0.2977 - val_accuracy: 0.9802 - val_dice_coef: 0.6822
    Epoch 18/50
    58/58 [==============================] - 37s 638ms/step - loss: 0.0252 - accuracy: 0.9972 - dice_coef: 0.9748 - val_loss: 0.2257 - val_accuracy: 0.9824 - val_dice_coef: 0.7505
    Epoch 19/50
    58/58 [==============================] - 37s 637ms/step - loss: 0.0258 - accuracy: 0.9971 - dice_coef: 0.9741 - val_loss: 0.2042 - val_accuracy: 0.9839 - val_dice_coef: 0.7711
    Epoch 20/50
    58/58 [==============================] - 37s 638ms/step - loss: 0.0230 - accuracy: 0.9974 - dice_coef: 0.9770 - val_loss: 0.2063 - val_accuracy: 0.9824 - val_dice_coef: 0.7689
    Epoch 21/50
    58/58 [==============================] - 37s 636ms/step - loss: 0.0217 - accuracy: 0.9975 - dice_coef: 0.9783 - val_loss: 0.2295 - val_accuracy: 0.9813 - val_dice_coef: 0.7477
    Epoch 22/50
    58/58 [==============================] - 37s 635ms/step - loss: 0.0234 - accuracy: 0.9973 - dice_coef: 0.9767 - val_loss: 0.2313 - val_accuracy: 0.9827 - val_dice_coef: 0.7476
    Epoch 23/50
    58/58 [==============================] - 37s 632ms/step - loss: 0.0214 - accuracy: 0.9976 - dice_coef: 0.9786 - val_loss: 0.2384 - val_accuracy: 0.9812 - val_dice_coef: 0.7429
    Epoch 24/50
    58/58 [==============================] - 37s 638ms/step - loss: 0.0197 - accuracy: 0.9978 - dice_coef: 0.9803 - val_loss: 0.1857 - val_accuracy: 0.9855 - val_dice_coef: 0.7935
    Epoch 25/50
    58/58 [==============================] - 37s 638ms/step - loss: 0.0180 - accuracy: 0.9979 - dice_coef: 0.9820 - val_loss: 0.2501 - val_accuracy: 0.9807 - val_dice_coef: 0.7312
    Epoch 26/50
    58/58 [==============================] - 37s 636ms/step - loss: 0.0173 - accuracy: 0.9980 - dice_coef: 0.9827 - val_loss: 0.1671 - val_accuracy: 0.9866 - val_dice_coef: 0.8100
    Epoch 27/50
    58/58 [==============================] - 37s 638ms/step - loss: 0.0177 - accuracy: 0.9980 - dice_coef: 0.9823 - val_loss: 0.1479 - val_accuracy: 0.9883 - val_dice_coef: 0.8264
    Epoch 28/50
    58/58 [==============================] - 37s 639ms/step - loss: 0.0168 - accuracy: 0.9981 - dice_coef: 0.9832 - val_loss: 0.2063 - val_accuracy: 0.9849 - val_dice_coef: 0.7704
    Epoch 29/50
    58/58 [==============================] - 37s 632ms/step - loss: 0.0394 - accuracy: 0.9955 - dice_coef: 0.9603 - val_loss: 0.3966 - val_accuracy: 0.9692 - val_dice_coef: 0.5850
    Epoch 30/50
    58/58 [==============================] - 37s 633ms/step - loss: 0.0540 - accuracy: 0.9940 - dice_coef: 0.9461 - val_loss: 0.2225 - val_accuracy: 0.9828 - val_dice_coef: 0.7541
    Epoch 31/50
    58/58 [==============================] - 37s 638ms/step - loss: 0.0289 - accuracy: 0.9967 - dice_coef: 0.9712 - val_loss: 0.2336 - val_accuracy: 0.9831 - val_dice_coef: 0.7427
    Epoch 32/50
    58/58 [==============================] - 37s 635ms/step - loss: 0.0238 - accuracy: 0.9973 - dice_coef: 0.9762 - val_loss: 0.1962 - val_accuracy: 0.9847 - val_dice_coef: 0.7770
    Epoch 33/50
    58/58 [==============================] - 37s 632ms/step - loss: 0.0205 - accuracy: 0.9976 - dice_coef: 0.9795 - val_loss: 0.1853 - val_accuracy: 0.9852 - val_dice_coef: 0.7909
    Epoch 34/50
    58/58 [==============================] - 37s 637ms/step - loss: 0.0186 - accuracy: 0.9979 - dice_coef: 0.9815 - val_loss: 0.2639 - val_accuracy: 0.9837 - val_dice_coef: 0.7153
    Epoch 35/50
    58/58 [==============================] - 37s 635ms/step - loss: 0.0170 - accuracy: 0.9981 - dice_coef: 0.9830 - val_loss: 0.1718 - val_accuracy: 0.9858 - val_dice_coef: 0.8076
    Epoch 36/50
    58/58 [==============================] - 37s 633ms/step - loss: 0.0188 - accuracy: 0.9979 - dice_coef: 0.9812 - val_loss: 0.3572 - val_accuracy: 0.9743 - val_dice_coef: 0.6214
    Epoch 37/50
    58/58 [==============================] - 38s 660ms/step - loss: 0.0169 - accuracy: 0.9981 - dice_coef: 0.9831 - val_loss: 0.1931 - val_accuracy: 0.9808 - val_dice_coef: 0.7834
    Epoch 38/50
    58/58 [==============================] - 42s 724ms/step - loss: 0.0166 - accuracy: 0.9981 - dice_coef: 0.9834 - val_loss: 0.1737 - val_accuracy: 0.9852 - val_dice_coef: 0.8051
    Epoch 39/50
    58/58 [==============================] - 42s 726ms/step - loss: 0.0166 - accuracy: 0.9981 - dice_coef: 0.9834 - val_loss: 0.2341 - val_accuracy: 0.9834 - val_dice_coef: 0.7448
    Epoch 40/50
    58/58 [==============================] - 42s 726ms/step - loss: 0.0153 - accuracy: 0.9983 - dice_coef: 0.9847 - val_loss: 0.2130 - val_accuracy: 0.9819 - val_dice_coef: 0.7645
    Epoch 41/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0146 - accuracy: 0.9983 - dice_coef: 0.9854 - val_loss: 0.2095 - val_accuracy: 0.9835 - val_dice_coef: 0.7700
    Epoch 42/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0139 - accuracy: 0.9984 - dice_coef: 0.9861 - val_loss: 0.2358 - val_accuracy: 0.9807 - val_dice_coef: 0.7450
    Epoch 43/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0138 - accuracy: 0.9984 - dice_coef: 0.9862 - val_loss: 0.2855 - val_accuracy: 0.9820 - val_dice_coef: 0.6959
    Epoch 44/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0133 - accuracy: 0.9985 - dice_coef: 0.9867 - val_loss: 0.2237 - val_accuracy: 0.9832 - val_dice_coef: 0.7579
    Epoch 45/50
    58/58 [==============================] - 42s 727ms/step - loss: 0.0130 - accuracy: 0.9985 - dice_coef: 0.9870 - val_loss: 0.2149 - val_accuracy: 0.9828 - val_dice_coef: 0.7673
    Epoch 46/50
    58/58 [==============================] - 42s 721ms/step - loss: 0.0138 - accuracy: 0.9984 - dice_coef: 0.9862 - val_loss: 0.1896 - val_accuracy: 0.9851 - val_dice_coef: 0.7895
    Epoch 47/50
    58/58 [==============================] - 42s 721ms/step - loss: 0.0126 - accuracy: 0.9986 - dice_coef: 0.9874 - val_loss: 0.1927 - val_accuracy: 0.9856 - val_dice_coef: 0.7871
    Epoch 48/50
    58/58 [==============================] - 42s 721ms/step - loss: 0.0130 - accuracy: 0.9985 - dice_coef: 0.9869 - val_loss: 0.1917 - val_accuracy: 0.9851 - val_dice_coef: 0.7887
    Epoch 49/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0125 - accuracy: 0.9986 - dice_coef: 0.9875 - val_loss: 0.2072 - val_accuracy: 0.9840 - val_dice_coef: 0.7752
    Epoch 50/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0123 - accuracy: 0.9986 - dice_coef: 0.9877 - val_loss: 0.1597 - val_accuracy: 0.9866 - val_dice_coef: 0.8198
    training for fold 2
    compiling model...
    fitting model...
    Epoch 1/50
    2021-12-21 18:58:48.446957: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
    58/58 [==============================] - ETA: 0s - loss: 0.4943 - accuracy: 0.8784 - dice_coef: 0.5075
    2021-12-21 18:59:28.779922: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:112] Plugin optimizer for device_type GPU is enabled.
    58/58 [==============================] - 47s 750ms/step - loss: 0.4943 - accuracy: 0.8784 - dice_coef: 0.5075 - val_loss: 0.2444 - val_accuracy: 0.9667 - val_dice_coef: 0.7601
    Epoch 2/50
    58/58 [==============================] - 42s 726ms/step - loss: 0.1460 - accuracy: 0.9832 - dice_coef: 0.8540 - val_loss: 0.3830 - val_accuracy: 0.9508 - val_dice_coef: 0.6161
    Epoch 3/50
    58/58 [==============================] - 42s 724ms/step - loss: 0.1012 - accuracy: 0.9883 - dice_coef: 0.8990 - val_loss: 0.3321 - val_accuracy: 0.9607 - val_dice_coef: 0.6710
    Epoch 4/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0880 - accuracy: 0.9900 - dice_coef: 0.9117 - val_loss: 0.4143 - val_accuracy: 0.9453 - val_dice_coef: 0.5881
    Epoch 5/50
    58/58 [==============================] - 42s 724ms/step - loss: 0.0784 - accuracy: 0.9909 - dice_coef: 0.9217 - val_loss: 0.4067 - val_accuracy: 0.9526 - val_dice_coef: 0.5961
    Epoch 6/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0569 - accuracy: 0.9933 - dice_coef: 0.9431 - val_loss: 0.7410 - val_accuracy: 0.9263 - val_dice_coef: 0.2508
    Epoch 7/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.0427 - accuracy: 0.9950 - dice_coef: 0.9573 - val_loss: 0.7172 - val_accuracy: 0.9274 - val_dice_coef: 0.2753
    Epoch 8/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0423 - accuracy: 0.9951 - dice_coef: 0.9577 - val_loss: 0.6935 - val_accuracy: 0.9291 - val_dice_coef: 0.3014
    Epoch 9/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0352 - accuracy: 0.9959 - dice_coef: 0.9648 - val_loss: 0.5895 - val_accuracy: 0.9375 - val_dice_coef: 0.4090
    Epoch 10/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.0349 - accuracy: 0.9959 - dice_coef: 0.9651 - val_loss: 0.7237 - val_accuracy: 0.9260 - val_dice_coef: 0.2682
    Epoch 11/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.0364 - accuracy: 0.9957 - dice_coef: 0.9636 - val_loss: 0.6954 - val_accuracy: 0.9283 - val_dice_coef: 0.2989
    Epoch 12/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.0373 - accuracy: 0.9957 - dice_coef: 0.9628 - val_loss: 0.7148 - val_accuracy: 0.9275 - val_dice_coef: 0.2767
    Epoch 13/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0306 - accuracy: 0.9965 - dice_coef: 0.9695 - val_loss: 0.7043 - val_accuracy: 0.9283 - val_dice_coef: 0.2892
    Epoch 14/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0261 - accuracy: 0.9969 - dice_coef: 0.9739 - val_loss: 0.6886 - val_accuracy: 0.9294 - val_dice_coef: 0.3067
    Epoch 15/50
    58/58 [==============================] - 42s 727ms/step - loss: 0.0247 - accuracy: 0.9971 - dice_coef: 0.9753 - val_loss: 0.6810 - val_accuracy: 0.9300 - val_dice_coef: 0.3155
    Epoch 16/50
    58/58 [==============================] - 42s 730ms/step - loss: 0.0236 - accuracy: 0.9973 - dice_coef: 0.9764 - val_loss: 0.6800 - val_accuracy: 0.9306 - val_dice_coef: 0.3192
    Epoch 17/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0244 - accuracy: 0.9971 - dice_coef: 0.9755 - val_loss: 0.6568 - val_accuracy: 0.9324 - val_dice_coef: 0.3433
    Epoch 18/50
    58/58 [==============================] - 42s 727ms/step - loss: 0.0230 - accuracy: 0.9973 - dice_coef: 0.9771 - val_loss: 0.6553 - val_accuracy: 0.9327 - val_dice_coef: 0.3481
    Epoch 19/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0220 - accuracy: 0.9974 - dice_coef: 0.9780 - val_loss: 0.6617 - val_accuracy: 0.9327 - val_dice_coef: 0.3379
    Epoch 20/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.0214 - accuracy: 0.9975 - dice_coef: 0.9786 - val_loss: 0.6811 - val_accuracy: 0.9298 - val_dice_coef: 0.3152
    Epoch 21/50
    58/58 [==============================] - 42s 724ms/step - loss: 0.0209 - accuracy: 0.9976 - dice_coef: 0.9791 - val_loss: 0.6896 - val_accuracy: 0.9298 - val_dice_coef: 0.3046
    Epoch 22/50
    58/58 [==============================] - 42s 727ms/step - loss: 0.0195 - accuracy: 0.9977 - dice_coef: 0.9804 - val_loss: 0.6413 - val_accuracy: 0.9346 - val_dice_coef: 0.3622
    Epoch 23/50
    58/58 [==============================] - 42s 727ms/step - loss: 0.0186 - accuracy: 0.9978 - dice_coef: 0.9815 - val_loss: 0.6630 - val_accuracy: 0.9316 - val_dice_coef: 0.3379
    Epoch 24/50
    58/58 [==============================] - 42s 724ms/step - loss: 0.0196 - accuracy: 0.9977 - dice_coef: 0.9804 - val_loss: 0.5806 - val_accuracy: 0.9398 - val_dice_coef: 0.4251
    Epoch 25/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0200 - accuracy: 0.9977 - dice_coef: 0.9800 - val_loss: 0.6399 - val_accuracy: 0.9343 - val_dice_coef: 0.3645
    Epoch 26/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0182 - accuracy: 0.9979 - dice_coef: 0.9818 - val_loss: 0.6527 - val_accuracy: 0.9326 - val_dice_coef: 0.3500
    Epoch 27/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0167 - accuracy: 0.9981 - dice_coef: 0.9833 - val_loss: 0.6382 - val_accuracy: 0.9343 - val_dice_coef: 0.3664
    Epoch 28/50
    58/58 [==============================] - 42s 727ms/step - loss: 0.0169 - accuracy: 0.9980 - dice_coef: 0.9831 - val_loss: 0.6501 - val_accuracy: 0.9333 - val_dice_coef: 0.3541
    Epoch 29/50
    58/58 [==============================] - 42s 728ms/step - loss: 0.0166 - accuracy: 0.9980 - dice_coef: 0.9834 - val_loss: 0.6286 - val_accuracy: 0.9347 - val_dice_coef: 0.3768
    Epoch 30/50
    58/58 [==============================] - 42s 724ms/step - loss: 0.0154 - accuracy: 0.9982 - dice_coef: 0.9846 - val_loss: 0.6644 - val_accuracy: 0.9313 - val_dice_coef: 0.3361
    Epoch 31/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.0159 - accuracy: 0.9981 - dice_coef: 0.9840 - val_loss: 0.6875 - val_accuracy: 0.9294 - val_dice_coef: 0.3118
    Epoch 32/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0152 - accuracy: 0.9982 - dice_coef: 0.9848 - val_loss: 0.6290 - val_accuracy: 0.9357 - val_dice_coef: 0.3767
    Epoch 33/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0144 - accuracy: 0.9983 - dice_coef: 0.9856 - val_loss: 0.6449 - val_accuracy: 0.9345 - val_dice_coef: 0.3606
    Epoch 34/50
    58/58 [==============================] - 42s 731ms/step - loss: 0.0147 - accuracy: 0.9983 - dice_coef: 0.9853 - val_loss: 0.6336 - val_accuracy: 0.9349 - val_dice_coef: 0.3716
    Epoch 35/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0143 - accuracy: 0.9983 - dice_coef: 0.9857 - val_loss: 0.6508 - val_accuracy: 0.9350 - val_dice_coef: 0.3554
    Epoch 36/50
    58/58 [==============================] - 42s 721ms/step - loss: 0.0143 - accuracy: 0.9983 - dice_coef: 0.9857 - val_loss: 0.6629 - val_accuracy: 0.9322 - val_dice_coef: 0.3382
    Epoch 37/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0227 - accuracy: 0.9973 - dice_coef: 0.9772 - val_loss: 0.7196 - val_accuracy: 0.9254 - val_dice_coef: 0.2710
    Epoch 38/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.1108 - accuracy: 0.9870 - dice_coef: 0.8891 - val_loss: 0.6943 - val_accuracy: 0.9256 - val_dice_coef: 0.3046
    Epoch 39/50
    58/58 [==============================] - 42s 727ms/step - loss: 0.0818 - accuracy: 0.9901 - dice_coef: 0.9184 - val_loss: 0.7034 - val_accuracy: 0.9266 - val_dice_coef: 0.2881
    Epoch 40/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0377 - accuracy: 0.9956 - dice_coef: 0.9623 - val_loss: 0.7012 - val_accuracy: 0.9286 - val_dice_coef: 0.2934
    Epoch 41/50
    58/58 [==============================] - 42s 723ms/step - loss: 0.0300 - accuracy: 0.9965 - dice_coef: 0.9700 - val_loss: 0.7064 - val_accuracy: 0.9276 - val_dice_coef: 0.2848
    Epoch 42/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0255 - accuracy: 0.9970 - dice_coef: 0.9745 - val_loss: 0.7101 - val_accuracy: 0.9280 - val_dice_coef: 0.2802
    Epoch 43/50
    58/58 [==============================] - 42s 726ms/step - loss: 0.0239 - accuracy: 0.9972 - dice_coef: 0.9761 - val_loss: 0.7035 - val_accuracy: 0.9280 - val_dice_coef: 0.2869
    Epoch 44/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.0248 - accuracy: 0.9971 - dice_coef: 0.9752 - val_loss: 0.7184 - val_accuracy: 0.9272 - val_dice_coef: 0.2742
    Epoch 45/50
    58/58 [==============================] - 42s 720ms/step - loss: 0.0219 - accuracy: 0.9974 - dice_coef: 0.9781 - val_loss: 0.7088 - val_accuracy: 0.9278 - val_dice_coef: 0.2850
    Epoch 46/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.0208 - accuracy: 0.9976 - dice_coef: 0.9792 - val_loss: 0.7142 - val_accuracy: 0.9276 - val_dice_coef: 0.2765
    Epoch 47/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0188 - accuracy: 0.9978 - dice_coef: 0.9812 - val_loss: 0.7050 - val_accuracy: 0.9282 - val_dice_coef: 0.2870
    Epoch 48/50
    58/58 [==============================] - 42s 725ms/step - loss: 0.0174 - accuracy: 0.9980 - dice_coef: 0.9826 - val_loss: 0.7003 - val_accuracy: 0.9282 - val_dice_coef: 0.2902
    Epoch 49/50
    58/58 [==============================] - 42s 722ms/step - loss: 0.0178 - accuracy: 0.9979 - dice_coef: 0.9822 - val_loss: 0.7055 - val_accuracy: 0.9280 - val_dice_coef: 0.2857
    Epoch 50/50
    58/58 [==============================] - 42s 732ms/step - loss: 0.0172 - accuracy: 0.9980 - dice_coef: 0.9828 - val_loss: 0.7142 - val_accuracy: 0.9276 - val_dice_coef: 0.2763
    


### 5. Load and test the model:


```python
from testing.predict import *

predict(val_images)
```

    loading weights
    predicting masks...
    20/20 [==============================] - 13s 666ms/step
    saving prediction...
    


### 6. Results and plotting:


```python
'''
plot training and validation loss, accuracy, and dice score.
the histories list contains all folds training history.
'''
from testing.output import *

plot_histories(histories)
```


    
![png](display_preds/output_20_0.png)
    



    
![png](display_preds/output_20_1.png)
    



```python
print('average accuracy : ', np.mean(np.array(history.history['val_accuracy'])))
print('average loss : ', np.mean(np.array(history.history['val_loss'])))
print('average dicecoefs : ', np.mean(np.array(history.history['val_dice_coef'])))
```

    average accuracy :  0.9329276859760285
    average loss :  0.6465253165364265
    average dicecoefs :  0.3512564218044281



```python
'''
plot training and validation loss, accuracy, and dice score.
the histories list contains all folds training history.
'''
from testing.output import *

display_segmented_images()
```


    
![png](display_preds/output_22_0.png)
    



    
![png](display_preds/output_22_1.png)
    



    
![png](display_preds/output_22_2.png)
    


### Pipeline:


```python
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
```

### Citations:

```{bibliography}
arxiv.org/pdf/1505.04597.pdf
```
```{bibliography}
github.com/zhixuhao/unet
```
```{bibliography}
IRCAD Dataset
