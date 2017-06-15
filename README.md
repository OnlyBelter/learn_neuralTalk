# learn_neuralTalk


# Overview
 Andrej Karpathy's code can find [here](https://github.com/karpathy/neuraltalk)


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
- step5: in terminal, run `python predict_on_images.py "cv/model_checkpoint_coco_visionlab43.stanford.edu_lstm_11.14.p" -r self_pic`, and you can see the result in 'result.html'

You can download above files from [here](https://pan.baidu.com/s/1dEA0sXb)