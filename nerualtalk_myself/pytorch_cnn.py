#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:07:35 2017

@author: belter
"""

import scipy
import json
import os
from collections import defaultdict


dataset = 'flickr8k'
dataset_root = r'/media/sf_vm_share_folder/neuraltalk/data/flickr8k'
features_path = os.path.join(dataset_root, 'vgg_feats.mat')
dataset_path = os.path.join(dataset_root, 'dataset.json')
features_struct = scipy.io.loadmat(features_path)

dataset = json.load(open(dataset_path, 'r'))
split = defaultdict(list)
for img in dataset['images']:
    split[img['split']].append(img)
    
with open(os.path.join(dataset_root, 'dataset_2.json'), 'w') as file_handle:
    file_handle.write(json.dumps(dataset, indent = 2))
