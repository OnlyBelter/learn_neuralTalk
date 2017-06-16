# learn_neuralTalk


# Overview
This repository mainly services our final project of computer animation class this term.

- Our group member: FeiSun and I.

- Andrej Karpathy's code can be found [here](https://github.com/karpathy/neuraltalk)

- Andrej Karpathy's paper, Deep Visual-Semantic Alignments for Generating Image Descriptions, can be found [here](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

# Dependencies
Because We need to use tensorflow(it may not support Python2.7 in windows), We ues Python 3.5.
- argparse
- tensorflow
- keras
- numpy

Most of these are okay to install with pip. To install all dependencies at once, run the command `pip install -r requirements.txt`

# To predict images of yourself
We use pretrained model [VGG16](https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py) and [LSTM](http://cs.stanford.edu/people/karpathy/neuraltalk/), so we need to import their parameters at first. You also can train these models by yourself.


- step1: put yourself's images into 'self_pic/img/'
- step2: download 'vgg16_weights_tf_dim_ordering_tf_kernels.h5' and put it into 'VGG16_weights/'
- step3: download 'model_checkpoint_coco_visionlab43.stanford.edu_lstm_11.14.p' and put it into 'cv/'
- step4: in terminal, run `python get_img_features_VGG16.py` to get 'vgg_feats.npy' and 'self_img_dataset.json' file
- step5: in terminal, run `python predict_on_images.py "cv/model_checkpoint_coco_visionlab43.stanford.edu_lstm_11.14.p" -r self_pic`, then you can see the result in 'result.html'

You can download above files from [here](https://pan.baidu.com/s/1dEA0sXb)

#### We tested this model by ourselves images.

Some of images get pretty well result, like this one
<div align="center">
    <img src="https://github.com/OnlyBelter/learn_neuralTalk/blob/master/demo_images/001_bridge.png?raw=true">
</div>

and this one
<div align="center">
    <img src="https://github.com/OnlyBelter/learn_neuralTalk/blob/master/demo_images/002_ski.png?raw=true">
</div>

This one is not bad
<div align="center">
    <img src="https://github.com/OnlyBelter/learn_neuralTalk/blob/master/demo_images/003_dog.png?raw=true">
</div>

But this image get a very strange description, may because she dressed a cat-like sweater.
<div align="center">
    <img src="https://github.com/OnlyBelter/learn_neuralTalk/blob/master/demo_images/004_person.png?raw=true">
</div>



# Improvement
- Instead of using matlab code written by Andrej Karpathy, we create ourselves VGG16 model to extract the features of each image by [keras](https://keras.io/), you can see the source code in file '[get_img_features_VGG16.py](https://github.com/OnlyBelter/learn_neuralTalk/blob/master/get_img_features_VGG16.py)'.
- We also can predict the classes of each image and have a look at the relation between 'prediction of image class' and 'generation of image description'. Our fundamental hypothesis is if we can get a very precise prediction on image classification, this may imply that we extract most important features about this image, then we can use these features to generate a better image description.