#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:52:49 2017

@author: belter
"""
import os
import json
import io
# http://stackoverflow.com/a/37795053/2803344

file_dir = r'/media/sf_vm_share_folder/neuraltalk/data/flickr8k'
input_file = r'dataset.json'
output_file = r'dataset.json2'

data = {}
with open(os.path.join(file_dir, input_file)) as file_handle:
    data = json.load(file_handle)
    
with open(os.path.join(file_dir, output_file), 'w') as f_handle:
    f_handle.write(json.dumps(data, indent=4))