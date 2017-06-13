# https://keras-cn.readthedocs.io/en/latest/other/application/
from keras.models import Sequential
from keras.layers import (Flatten, Dense, Conv2D, MaxPooling2D)
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os

def vgg16_model(weights_path):
    # this modle totaly has 22 layers with polling 
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
    return model
#    if last_layer:
#        return model
#    else:
#        return model.pop()  # delete the last layer


weights_path = r'C:\Users\Belter\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model = vgg16_model(weights_path)

def process_pic(img_path, model='', predict=True):
    img_path = img_path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    if predict:  # predict pic's class
        features = model.predict(x)  # 4096 features
        # print('Predicted:', decode_predictions(features, top=3)[0])
        return decode_predictions(features, top=3)[0]
    else:  # return 4096 features
        if len(model.layers) == 22:
            model.pop()  # delete the last layer
        features = model.predict(x)
        return features

self_pic_dir = r'D:\vm_share_folder\learn_neuralTalk\nerualtalk_myself\self_pic'

# predict images' classes
line = 0
for root, dirs, files in os.walk(os.path.join(self_pic_dir, 'img')):
    for f in files:
        if f.endswith('jpg'):
            # print(f)
            img_path = os.path.join(root, f)
            predict_results = process_pic(img_path, model=model)
            predict_list = [': '.join(str(i) for i in list(_[1:])) for _ in predict_results]
            output_str = '\t'.join([f, str(line)] + predict_list)
            line += 1
            with open(os.path.join(self_pic_dir, 'predict_images_class.txt'), 'a') as f_handle:
                f_handle.write(output_str + '\n')

# get images' features
features = np.zeros([line, 4096])
line2 = 0
for root, dirs, files in os.walk(os.path.join(self_pic_dir, 'img')):
    for f in files:
        if f.endswith('jpg'):
            print(f)
            img_path = os.path.join(root, f)
            features[line2] = process_pic(img_path, model=model, predict=False)
            line2 += 1
np.save(os.path.join(self_pic_dir, 'images_features'), features)         
    


# see model's summary
#model.summary()
























