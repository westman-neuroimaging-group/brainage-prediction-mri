#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gustav
@edited: caroline
"""
import transforms.transforms as tfs
def load_transforms(args,random_chance=1):

    # s determins random offset factor. If random_chance==0 (e.g. during testing), then we want no random offset.
    if random_chance==0:
        s=0
    else:
        s=1
    
    transforms=tfs.ComposeMRI([
            tfs.LoadNifti(),
            tfs.RandomScaling(scale_range=[.95,1.05]),#.95,1.05
            tfs.RandomRotation(angle_interval=[-10,10],rotation_axis=None),#-10,10
            tfs.ApplyAffine(so=2,chance=random_chance),
            tfs.Gamma(gamma_range=[.9,1.1],chance=random_chance),#.9,1.1
            tfs.ReturnImageData(),
            tfs.Crop(dims=args['img_dim'],offset=[0,-0,0],rand_offset = 5*s),
            tfs.ReduceSlices(2,2),
            tfs.PrcCap(),
            tfs.UnitInterval(),
            tfs.ToTensor(),
            ])

    return transforms
