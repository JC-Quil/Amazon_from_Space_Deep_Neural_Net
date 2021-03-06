# -*- coding: utf-8 -*-
'''Model derived as a smaller version of the Xception V1 model for Keras.

I selected this high performance base-model as a case study for
understanding of depthwise separable convolution layers.
On amazon_from_space, the base model gets to a top-1 validation accuracy of 0.790.
and a top-5 validation accuracy of 0.945.

At the date of posting this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers.
# Reference:
All code in this repository is under the MIT license.

# Reference:
The model is derived from the one released by Francois Chollet 
under the MIT license.
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
'''

import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage import io
from sklearn.metrics import fbeta_score
from skimage import transform

from keras.preprocessing import image

from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape


# Weight of the pre-trained model
#TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'


def Xception_derived_model(include_top=False, 
             weights=None,
             input_tensor=Input(shape=(124, 124, 5)),
             input_shape=(124, 124, 5),
             pooling=None,
             classes=17):
    """Instantiates the Xception-derived architecture.
    Optionally loads weights pre-trained on Planet's Amazon from Space
    dataset. This model is available for TensorFlow only, and can only 
    be used with inputs following the TensorFlow data format 
    `(width, height, channels)`.
    You should set `image_data_format="channels_last"` in your Keras config
    located at ~/.keras/keras.json.
    Note that the default input image size for this model is 299x299.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization) 
            or "amazon_from_space" (pre-training on Planet's dataset).
        input_tensor: Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape tuple to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: number of classes to classify images into, E.g. 17 in
        the study case if `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    # Check weight argument validity.
    if weights not in {'amazon_from_space', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `amazon_from_space` '
                         '(pre-training on Planet Amazon_from_space dataset).')

    # Check classes and weight argument compatibility
    if weights == 'amazon_from_space' and classes != 17:
        raise ValueError('If using `weights` as amazon_from_space with `include_top`'
                         ' as true, `classes` should be 17')

    # Check that TensoFlow backend is operational
    if K.backend() != 'tensorflow':
        raise RuntimeError('The Xception model is only available with '
                           'the TensorFlow backend.')
   
    if K.image_data_format() != 'channels_last':
        warnings.warn('The Xception model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape
    #input_shape = _obtain_input_shape(input_shape,
    #                                  default_size=256,
    #                                  min_size=71,
    #                                  data_format=K.image_data_format(),
    #                                  include_top=False) #### Pourquoi ici?

    ### input_tensor question. Could be optional.
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    # Define the model architecture
    # Entry flow
    x = Conv2D(16, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(32, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    # Residual connection
    #residual = Conv2D(16, (1, 1), strides=(2, 2),
    #                  padding='same', use_bias=False)(x)
    #residual = BatchNormalization()(residual)

    #x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    #x = BatchNormalization(name='block2_sepconv1_bn')(x)
    #x = Activation('relu', name='block2_sepconv2_act')(x)
    #x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    #x = BatchNormalization(name='block2_sepconv2_bn')(x)

    #x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    #x = layers.add([x, residual])

    residual = Conv2D(48, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    #x = Activation('relu', name='block3_sepconv1_act')(x)
    #x = SeparableConv2D(48, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    #x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(48, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    #x = Activation('relu', name='block4_sepconv1_act')(x)
    #x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    #x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    # Middle flow
    for i in range(2):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    # Exit flow
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    #x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    ### Include top is correlated to Xception V1 model.
    ###if include_top:
    ###    x = GlobalAveragePooling2D(name='avg_pool')(x)
    ###    x = Dense(classes, activation='softmax', name='predictions')(x)
    ###else:
    ###    if pooling == 'avg':
    ###        x = GlobalAveragePooling2D()(x)
    ###    elif pooling == 'max':
    ###        x = GlobalMaxPooling2D()(x)

    # Includes a fully-connected layer at the top of the network
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Create model.
    model = Model(inputs, x, name='xception_derived_model')

    # load weights for prediction
    if weights == 'Amazon_from_space':    
        weights_path = get_file('xception_derived_model_weights.h5',
                                TF_WEIGHTS_PATH,
                                cache_subdir='models')
        model.load_weights(weights_path)

    ### Remet les images au vieux format s'il est different de celui attendu par le modele.
    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


 
def data_processing(labels_df, x_train, y_train, label_map):
    """
    Pre-process inputs listed  "labels_df" in np.arrays for the images and the label.
    Include normalization of values, calculation of NDVI (Vegetal Index).
    Output: 
        Save the X_train and Y_train ndarrays into multiple batches into .ndy files.
    """
    subset = str()

    if labels_df.shape[0] == 32384 or labels_df.shape[0] == 3120 or labels_df.shape[0] == 16:
        batch_size = 8
        subset = "train"
    elif labels_df.shape[0] == 8080 or labels_df.shape[0] == 1920 or labels_df.shape[0] == 8:
        batch_size = 4
        subset = "valid"
    elif labels_df.shape[0] == 40669:
        batch_size = 607
        subset = "test"   
    else:
        raise ValueError('The dataset format is different than expected')

    label_map = label_map
#    images_size = (256, 256)
    images_size = (124, 124)

    # Iterate through batches of rows of the dataset
    for i in range(labels_df.shape[0]//batch_size):
        
        temp_labels_df = labels_df.iloc[i*batch_size:((i+1)*batch_size) , :]
        
        # Iterate through the samples batch and create x and y for training
        for f, tags in tqdm(temp_labels_df.values, miniters=100):
            # load a .tif file
            img = io.imread('data/train-tif/{}.tif'.format(f))
            img = transform.resize(img, images_size)

            # Add NDVI layer
            np.seterr(all='warn') # divide by zero, NaN values
            img_ndvi = np.expand_dims((img[:, :, 3] - img[:, :, 2]) / (img[:, :, 3] + img[:, :, 2]), axis=2) # (NIR - RED) / (NIR + RED)
            img = np.concatenate((img, img_ndvi), axis=2)
            
            # Create the target array for an image
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1 

            x_train.append(img)
            y_train.append(targets)

        # Format values
        y_train = np.array(y_train, np.uint8)
        x_train = np.array(x_train, np.float16) / 65536.
        x_train -= 0.5
        x_train *= 2        

        # Save subsets in npz files
        np.save('data/{}-npy/npdatasetX{}'.format(subset, i), x_train)
        x_train = []
        np.save('data/{}-npy/npdatasetY{}'.format(subset, i), y_train)
        y_train = []
        #print "{} data saved".format(subset)
