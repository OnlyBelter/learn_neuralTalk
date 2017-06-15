#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:35:47 2017

@author: belter
"""

import matplotlib.pyplot as plt
import scipy
import numpy as np
features_path = '/media/sf_vm_share_folder/neuraltalk/data/flickr8k/vgg_feats.mat'
features_struct = scipy.io.loadmat(features_path)
features = features_struct['feats']
a_feat = features[:, 0].reshape(64, 64)
plt.imshow(a_feat)
