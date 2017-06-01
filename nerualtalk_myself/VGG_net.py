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
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils.data_utils import get_file

#import numpy as np
weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
model = VGG16(weights='imagenet', include_top=True)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)