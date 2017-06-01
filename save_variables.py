#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:10:48 2017

@author: belter
"""

import json
import os

out_dir = r'/media/sf_vm_share_folder/learning_nerualtalk'
file_name1 = 'ixtoword.json'
file_name2 = 'wordtoix.sort.json'

with open(os.path.join(out_dir, file_name2), 'w') as f_handle:
    f_handle.write(json.dumps(sorted(wordtoix.items(), key=lambda x: x[1], reverse=False), indent=4))