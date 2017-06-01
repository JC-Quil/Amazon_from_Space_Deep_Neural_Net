'''
Classic deep neural network
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
from keras.layers import Dropout
from keras.layers import Flatten
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


def Xception_derived_model(include_top=False, 
             weights=None,
             input_tensor=Input(shape=(64, 64, 3)),
             input_shape=(64, 64, 3),
             pooling=None,
             classes=17):
    """
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
        warnings.warn('The model is only available for the '
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
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='block1_conv1')(img_input)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(32, (3, 3), use_bias=False, name='block1_conv2')(x)##############
    x = Activation('relu', name='block1_conv2_act')(x)

    x = MaxPooling2D(pool_size=2, name='block1_pool')(x) ###
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='block1_conv1')(img_input)##############
    x = Activation('relu', name='block1_conv1_act')(x)##############
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)##############
    x = Activation('relu', name='block1_conv2_act')(x)##############

    x = MaxPooling2D(pool_size=2, name='block1_pool')(x) ###
    x = Dropout(0.25)(x)##############


    x = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='block14_conv1')(x)
    x = Activation('relu', name='block14_conv1_act')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', kernel_initializer='glorot_uniform', bias_initializer='zeros', name='block14_conv2')(x)
    x = Activation('relu', name='block14_conv2_act')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None, name='block14_conv1_avg_pool')(x)
    x = Dropout(0.25)(x)    

    x = Flatten()(x)
    x = Dense(512, name='block15_den1')(x)
    x = BatchNormalization(name='block15_den1_bn')(x)
    x = Activation('relu', name='block15_den1_act')(x)
    
    
    x = Dropout(0.5)(x)  
    x = Dense(classes, activation='sigmoid', name='predictions')(x)


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

    if labels_df.shape[0] == 32384 or labels_df.shape[0] == 3120 or labels_df.shape[0] == 16 or labels_df.shape[0] == 64:
        batch_size = 8 ### Modified for smaller images
        subset = "train"
    elif labels_df.shape[0] == 8080 or labels_df.shape[0] == 1920 or labels_df.shape[0] == 8:
        batch_size = 4
        subset = "valid"
    elif labels_df.shape[0] == 40669:
        batch_size = 4
        subset = "test"   
    elif labels_df.shape[0] == 20522:
        batch_size = 2
        subset = "test-add"  
    else:
        raise ValueError('The dataset format is different than expected')

    label_map = label_map
#    images_size = (256, 256)
    images_size = (64, 64)

    # Iterate through batches of rows of the dataset
    for i in range(labels_df.shape[0]//batch_size):
        
        temp_labels_df = labels_df.iloc[i*batch_size:((i+1)*batch_size) , :]
        
        # Iterate through the samples batch and create x and y for training
        for f, tags in tqdm(temp_labels_df.values, miniters=100):
            # load a .tif file
            img = io.imread('data/{}-jpg/{}.jpg'.format(subset,f)) ######## Modified for train jpg folder
            img = transform.resize(img, images_size)

### Removed for use of JPEG files:
#            # Add NDVI layer // Removed for usage of JPG files
#            np.seterr(all='warn') # divide by zero, NaN values
#            img_ndvi = np.expand_dims((img[:, :, 3] - img[:, :, 2]) / (img[:, :, 3] + img[:, :, 2]), axis=2) # (NIR - RED) / (NIR + RED)
#            img = np.concatenate((img, img_ndvi), axis=2)
            
            # Create the target array for an image
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1 

            x_train.append(img)
            y_train.append(targets)

        # Format values
        y_train = np.array(y_train, np.uint8)
        x_train = np.array(x_train, np.float16) / 255.

### Removed for use of JPEG files:  
#        x_train = np.array(x_train, np.float16) / 65536.
####        x_train -= 0.5
####        x_train *= 2        


        # Save subsets in npz files
        np.save('data/{}-npy/npdatasetX{}'.format(subset, i), x_train)
        x_train = []
        np.save('data/{}-npy/npdatasetY{}'.format(subset, i), y_train)
        y_train = []
        #print "{} data saved".format(subset)
