#import numpy as np
#from scipy.misc import imread, imresize
#import os
#root_dir = r'D:\vm_share_folder\learning_nerualtalk\nerualtalk_myself\TF_vgg'
#file_name = 'vgg16_weights.npz'
#
#def load_weights(weight_file):
#    weights = np.load(weight_file)
#    keys = sorted(weights.keys())
#    for i, k in enumerate(keys):
#        print(i, k, np.shape(weights[k]))
#    return weights
#
#weights = load_weights(os.path.join(root_dir, file_name))
# https://keras-cn.readthedocs.io/en/latest/other/application/
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.models import Model
from keras.layers import (Flatten, Dense, Input, Conv2D, 
                          MaxPooling2D, GlobalAveragePooling2D,
                          GlobalMaxPooling2D)
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
#from keras.utils.data_utils import get_file

#import numpy as np
#weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
#model = VGG16(weights='imagenet', include_top=True)

# pring all layer name in model
#for i, layer in enumerate(model.layers):
#    print(i, layer.name)


def vgg16_model(weights_path, last_layer=False):
    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', 
                     input_shape=(224, 224, 3), name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    
    # Block 6, fc
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(1000, activation='softmax', name='predictions'))
    model.load_weights(weights_path)
    if last_layer:
        return model
    return model.pop()  # delete the last layer


weights_path = r'C:\Users\Belter\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model = vgg16_model(weights_path, last_layer=True)
img_path = 'IMG_20170528_122525_E5A.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print('Predicted:', decode_predictions(features, top=3)[0])


# see model's summary
model.summary()
























