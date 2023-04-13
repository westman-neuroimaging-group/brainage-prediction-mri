'''
@author: GM
@edited: CD
'''

import numpy as np
import torch.nn as nn
import torch
from model.modules import ResidualNet3D

class ResNet3D(nn.Module):
    '''
    
    Args:
        input_dims = [h,w,c] - list or tuple containing dimensions of each input slice
    '''
    def __init__(self, input_dims,width_f=1):
        super(ResNet3D, self).__init__()
        self.features = ResidualNet3D(z=1,width_f=width_f)
        input_size = [input_dims[2],input_dims[0],input_dims[1]]

        self.l = self.get_flat_fts(input_dims, self.features)
        a=self.l
        N= 512 # number of neurons in fully connected layer
        
        self.fc1 = nn.Sequential(
            nn.Linear(a,N),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(N, 1))

    def get_flat_fts(self, in_size, fts):
            # Calculate output dimensions for feature exctration in each plane (with varying dimensions)
            f = fts(torch.Tensor(torch.ones(1,1,*in_size)))
            return int(np.prod(f.size()[1:]))
    def forward(self, x):
        x=x.unsqueeze(1)
        x = self.features(x)
        x_cnn = x.view(x.size()[0], -1)
        x= self.fc1(x_cnn)
        return x,x_cnn
