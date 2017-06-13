# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 21:48:40 2017

@author: Belter
"""

import numpy as np
import os
import json

root_dir = r'D:\vm_share_folder\learn_neuralTalk\nerualtalk_myself\self_pic'
features_file = 'images_features.npy'
class_file = 'predict_images_class.txt'

def create_data_json(root_d, file_n):
    a_dic = {'images': [], 'dataset': 'self_img'}
    with open(os.path.join(root_d, file_n)) as f_handle:
        for x in f_handle:
            each_img_dic = {}
            x = x.strip()
            x_list = x.split('\t')
            each_img_dic['filename'] = x_list[0]
            each_img_dic['imgid'] = x_list[1]
            each_img_dic['senences'] = []
            each_img_dic['split'] = 'train'
            each_img_dic['sentids'] = []
            each_img_dic['predict_classes'] = x_list[2:]
            a_dic['images'].append(each_img_dic)
    with open(os.path.join(root_d, 'self_img_dataset.json'), 'a') as f_handle:
        f_handle.write(json.dumps(a_dic, indent=2))


create_data_json(root_dir, class_file)
self_img_features = np.load(os.path.join(root_dir, features_file))
