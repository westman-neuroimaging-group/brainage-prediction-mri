#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: GM
@edited: CD
"""
from __future__ import print_function, division
import nibabel
from scipy import ndimage
import numpy as np
import torch
import matplotlib.pyplot as plt
import math


class LoadNifti(object):
    
    def __call__(self,file):
    # Load mri feils with same orientation and numpy format
        #print('---')
        #print(file)
        
        a = nibabel.load(file) # load file
        a= nibabel.as_closest_canonical(a) # transform into RAS orientation
        pixdim = a.header.get('pixdim')[1:4]
        #a = np.array(a.dataobj,dtype=np.float32) # fetch data as float 32bit
        a = np.array(a.dataobj) # fetch data as float 32bit
        a=np.float32(a)
        return {'data':a,'pixdim':pixdim,'affine':[]}
class CropShift(object):
    '''
    Shift center voxel of croping
    '''
    def __init__(self,shift):
        self.shift =np.array(shift)
    def __call__(self,image):
        ndims = len(image['data'].shape)
        
        T = np.identity(ndims+1)
        
        T[0:ndims,-1]=self.shift

        image['crop_shift']=T
        image['affine'].append('crop_shift')
        return image


class Gamma(object):
    ''' apply gamma transform '''
    def __init__(self, gamma_range = [.5,2],chance=0):
        self.chance=chance
        self.gamma_range=gamma_range
    def __call__(self, image):
        if np.random.rand()<self.chance: # only do gamma if random number < chance
            img_min= image['data'].min()
            img_max = image['data'].max()
            image['data']-=img_min
            image['data']/=(np.abs(img_max - img_min) + 1e-8)
            gamma=(np.random.rand())*(self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
            
            image['data']=np.power(image['data'],gamma)
            image['data'] = image['data']*(img_max - img_min) + img_min
            
        return image



class RandomShift(object):
    '''
    RandomShift center voxel of croping +-
    '''
    def __init__(self,max_shift):
        self.max_shift =np.array(max_shift)
    def __call__(self,image):
        ndims = len(image['data'].shape)
        
        shift = np.random.rand(ndims)*self.max_shift
        
        T = np.identity(ndims+1)
        
        T[0:ndims,-1]=shift

        image['random_shift']=T
        image['affine'].append('random_shift')
        return image

class RandomScaling(object):
    def __init__(self,scale_range=[1,1]):
        self.scale_range=scale_range
        
    def __call__(self,image):
        old_res = np.array(image['pixdim'])
        
        scale_factor = np.random.rand()*np.diff(self.scale_range)+self.scale_range[0]
        scale_factor = np.ones(len(old_res))*scale_factor
        
        S = np.ones(old_res.size+1)
        S[0:len(scale_factor)] = scale_factor
        S = np.diag(S)
        
        image['random_scale']=S
        image['affine'].append('random_scale')
        return image


class SetResolution(object):
    def __init__(self,new_dim,new_res=None):
        self.new_res=new_res
        self.new_dim = np.array(new_dim)
    def __call__(self,image):
        old_res = np.array(image['pixdim'])
        if self.new_res==None: # don't change resolution, only matrix size
            new_res_tmp=old_res
        else:
            new_res_tmp=self.new_res
        new_res_tmp=np.array(new_res_tmp)
        old_size = np.array(image['data'].shape)
        scale_factor = (old_res/new_res_tmp)
        #scale_factor *= (self.new_dim/old_size)
        S = np.ones(old_res.size+1)
        S[0:len(scale_factor)] = scale_factor
        S = np.diag(S)
        #print(S)
        image['scale']=S
        image['affine'].append('scale')
        return image

class TranslateToCom(object):
    def __init__(self,scale_f=4):
        self.f = scale_f
    def __call__(self,image):
        img_tmp=image['data'][::self.f,::self.f,::self.f]
        prc5 = np.percentile(img_tmp.ravel()[::4],5)
        img_tmp=img_tmp>prc5

        com = self.f*np.array(ndimage.center_of_mass(img_tmp))
#        com = self.f*np.array(ndimage.center_of_mass(image['data'][::self.f,::self.f,::self.f]))
        
        mid = np.array(image['data'].shape)/2
        T = np.identity(len(mid)+1)
        T[0:len(mid),-1] = mid-com
        image['com'] = T
        image['affine'].append('com')
        return image

class RandomRotation(object):
    def __init__(self,angle_interval=[-5,5],rotation_axis=None):
        self.a_l,self.a_u =angle_interval
        self.rotation_axis =rotation_axis
    def unit_vector(self,data, axis=None, out=None):
        """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
        """
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data*data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data
    def rotation_matrix(self,angle, direction, point=None):
        """Return matrix to rotate about axis defined by point and direction.
        TODO: make 2D/&3D version?
        """
        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = self.unit_vector(direction[:3])
        # rotation matrix around unit vector
        R = np.diag([cosa, cosa, cosa])
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array([[ 0.0,         -direction[2],  direction[1]],
                          [ direction[2], 0.0,          -direction[0]],
                          [-direction[1], direction[0],  0.0]])
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M
    def __call__(self,image):
        theta = np.random.uniform(self.a_l,self.a_u)
#        "theta=3
        #print('theta: ' + str(theta))
        angle =theta/180*np.pi
        if self.rotation_axis is None:
            u=np.random.rand(3)-.5
            #u = np.array([1,1,1])
        else:
            u=self.rotation_axis
        u = u/(np.dot(u,u))            
        R = self.rotation_matrix(-angle,u)
        
        image['rotation']=R
        image['affine'].append('rotation')
        return image

class ApplyAffine(object):
    def __init__(self,new_dim=None,new_res=None, so = 3,chance=1):
        self.new_res=new_res
        self.chance=chance # if random number is below self.chance then apply transformation
        if not new_dim==None:
            self.new_dim = np.array(new_dim)
        else:
            self.new_dim = new_dim
        self.so=so
    def __call__(self,image):
        if image['affine']==[] or np.random.rand()>self.chance:
            return image
        else:
            # forward mapping
            if self.new_dim is None:
                new_dim = image['data'].shape
            else:
                new_dim = self.new_dim
            
            ndims = len(image['pixdim'])
            T=np.identity(ndims+1)
            for a in image['affine']:
                T = np.dot(image[a],T)
            T_inv = np.linalg.inv(T)
            
            # compute offset for centering translation
            c_in = np.array(image['data'].shape)*.5
            c_out=np.array(new_dim)*.5
            s=c_in-np.dot(T_inv[:ndims,:ndims],c_out)
            #tx,ty = -s
            translation =np.identity(ndims +1)
            translation[0:ndims,-1]=-s
            T_inv = np.dot(np.linalg.inv(translation),T_inv)

            image['data'] = ndimage.affine_transform(
                    image['data'],T_inv,output_shape=new_dim,order=self.so)
            return image
class ReturnImageData(object):
    def __call__(self,image):
        return image['data']

class ToTensor(object):
    """Convert np arrays in sample to Tensors."""

    def __call__(self,image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = torch.from_numpy(image)
        return image

# class ComposeCT(object):
#     """ Composes several co_transforms together.
#     For example:
#     >>> co_transforms.Compose([
#     >>>     co_transforms.CenterCrop(10),
#     >>>     co_transforms.ToTensor(),
#     >>>  ])
#     """

#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, input, center_voxels):
#         for t in self.transforms:
            
#             try:
#                 input= t(input,center_voxels)
#             except:
#                 input= t(input)
    
        return input
class SwapAxes(object):
    """Switch axes so that convolution is applied in axial plane

    Args:
        axis1: dim1 to be swapped with axis2
    """

    def __init__(self,axis1,axis2):
       self.axis1 =axis1
       self.axis2 =axis2
    def __call__(self, image):
        
        return np.swapaxes(image,self.axis1,self.axis2)


class ReduceSlices(object):
    """Reduce the number of slices in all dimensions by selecting every factor_hw:th element
    in hw-dimensions and every factor_d:th element in depth dimension

    Args:
        factor_hw, factor_d_d (int): Desired reduction factor.
    """

    def __init__(self, factor_hw, factor_d):

        self.f_h = factor_hw
        self.f_w = factor_hw
        self.f_d = factor_d
        
    def __call__(self, image):
        # h, w, d = image.shape[:3]
        image = image[0::self.f_h,0::self.f_w,0::self.f_d]

        return image


class Threshold(object):
    """Threshold numpy values in image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, lower_limit, upper_limit):
        
        self.ll = lower_limit
        self.ul = upper_limit
    def __call__(self, image):
        image[image<=self.ll]=self.ll
        image[image>=self.ul]=self.ul
        return image


class RandomNoise(object):
    """Add random normally distributed noise to image tensor.

    Args:
        noise_var (float): maximum variance of added noise 
        p (float): probability of adding noise
    """

    def __init__(self, noise_var=.1, p=.5):
        
        self.noise_var = noise_var
        self.p = p
        
    def __call__(self, image):
        if torch.rand(1)[0]<self.p:
            var = torch.rand(1)[0]*self.noise_var
            plt.imshow(image[5,:,:],vmax=.3)
            plt.title('before noise')
            plt.colorbar()
            plt.show()
            
            plt.imshow(torch.randn(image.shape)[5,:,:]*var)
            plt.colorbar()
            plt.title('before noise')
            plt.show()

            image += torch.randn(image.shape)*var
            plt.imshow(image[5,:,:],vmax=.3)
            plt.colorbar()
            plt.title('after flip')
            plt.show()
            input('hej')
        return image

class Crop(object):
    def __init__(self,dims=[128,128,128],offset = [0,0,0],rand_offset =None):
        self.dims=dims
        self.offset = offset
        self.rand_offset = rand_offset
    def __call__(self, image):    
        dims_org = image.shape[:3]
        center = np.array([d/2 for d in dims_org]) # center coordinates
        if not (self.rand_offset is None or self.rand_offset==0):
            center += np.random.randint(-self.rand_offset,self.rand_offset,len(center))
        corner = center - np.array([of/2 for of in self.dims])
        corner -=np.array(self.offset)
        corner[corner<1]=0
        c=[int(c) for c in corner]
        image = image[c[0]:c[0]+self.dims[0],c[1]:c[1]+self.dims[1],c[2]:c[2]+self.dims[2]]
        #print(np.sum(image))
        return image

# class RandomCrop(object):
#     """Crop randomly the image in a sample.

#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """

#     def __init__(self, output_x, output_y, output_z):
        
#         self.output_x = output_x
#         self.output_y = output_y
#         self.output_z = output_z
#     def __call__(self, image):
#         #image, landmarks = sample['image'], sample['landmarks']
        
#         x, y, z = image.shape[:3]
#         new_x, new_y, new_z = self.output_x, self.output_y, self.output_z

#         x = np.random.randint(0, x - new_x)
#         y = np.random.randint(0, y - new_y)
#         z = np.random.randint(0, z - new_z)

#         image = image[x: x + new_x,
#                       y: y + new_y, z:z+new_z]
#         return image#.copy()#{'image': image, 'landmarks': landmarks}

# class RandomMirrorLR(object):
#     """Randomly mirror an image in the sagittal plane (left-right)"""
#     def __init__(self, axis):
#         self.axis=axis
#     def __call__(self, image):
#         randbol = np.random.randn()>0 # 50/50 if to rotate
#         #image, landmarks = sample['image'], sample['landmarks']

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         if randbol:#randbol:
        
#             image = np.flip(image,self.axis).copy()
        
#         return image    
        
class PerImageNormalization(object):
    """ Transforms all pixel values to to have total mean= 0 and std = 1"""

    def __call__(self, image):
        # transform data to 
        
        image -=image.mean()
        image /=image.std()

        return image
class Window(object):
    """ Cap image to be between [low,high]"""
    def __init__(self, low,high):
        self.low=low
        self.high=high
    def __call__(self, image):
        # transform data to 
        
        image[image<self.low] =self.low
        image[image>self.high] =self.high

        return image
    
class PrcCap(object):
    """ Cap all pixel values to prc 5,95 
    low: lower cap
    high: upper percentile value to cap high pixel values to.
    """
    def __init__(self, low=5,high=99):
        self.low = low
        self.high= high
    def __call__(self, image):
        # transform data to 
        
        l= np.percentile(image,self.low)
        h= np.percentile(image,self.high)
        image[image<l] = l; image[image>h] = h

        return image
class UnitInterval(object):
    """ Transforms all pixel values to be in [0,1]"""

    def __call__(self, image):
        # transform data to 
        
        image -=image.min()
        image /=image.max()
        image = (image-.5)*2
        return image


class ComposeMRI(object):
    """ Composes transforms. Same as built-in by PyTorch but possible to 
    customize if needed.
    
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            #print(t)
            input= t(input)   
        return input

