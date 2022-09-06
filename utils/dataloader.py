#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gustav
@edited: caroline
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:15:18 2020

@author: gustav
"""

import numpy as np
from torch.utils.data import Dataset
import torch

class mri_dset(Dataset):
    '''
    Custom dataset class for mri images during training and validation.
    
    '''
    def __init__(self, df, partition=None, 
                 input_transform=None, is_training=False):
        self.df = df
        if partition is not None:
            self.df = self.df.query('partition==@partition') # query for chosen partition
        self.is_training=is_training
        self.input_transform=input_transform

    def __getitem__(self, index):
        
        subj= self.df.iloc[index]
        path = subj['path_registered']
        
        img = self.input_transform(path)
        return img, subj['age_at_scan'], subj['uid']     

    def __len__(self):
        return len(self.df)

