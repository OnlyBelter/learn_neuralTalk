#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:38:27 2017

@author: belter
"""

import json
import os


dataset_root = r'/media/sf_vm_share_folder/neuraltalk'
dataset_path = os.path.join(dataset_root, 'result_struct.json')

dataset = json.load(open(dataset_path, 'r'))
with open(os.path.join(dataset_root, 'result_struct_2.json'), 'w') as file_handle:
    file_handle.write(json.dumps(dataset, indent = 2))
