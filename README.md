# learn_neuralTalk


# dependence
- python3.5
- tensorflow
- keras
- numpy
- cPickle

# to predict self images
- step1: put yourself's images into 'self_pic/img/'
- step2: download 'vgg16_weights_tf_dim_ordering_tf_kernels.h5' and put it into 'VGG16_weights/'
- step3: download 'model_checkpoint_coco_visionlab43.stanford.edu_lstm_11.14.p' and put it into 'cv/'
- in terminal, run 'python get_img_features_VGG16.py' to get 'vgg_feats.npy' and 'self_img_dataset.json' file