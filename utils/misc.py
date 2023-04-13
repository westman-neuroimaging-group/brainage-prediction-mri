#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: GM
@edited: CD
"""
import numpy as np 
import pandas as pd
import os
import nipype.interfaces.fsl as fsl
from shutil import copyfile
from sklearn.metrics import mean_squared_error, mean_absolute_error
#%%
class StoreOutput(object):
    """accumulate outputs and labels and compute performance metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.pred = []
        self.label = []
        self.uid= [] 
        self.guid= []
        self.df= [] 

    
    def get_df(self):
        # return dataframe with predictions
        self.df = pd.DataFrame({'age_at_scan':self.label,'predicted_age':self.pred,'uid':self.uid,'guid':self.guid})
        return self.df
 
    def update(self, new_pred, new_labels, new_uids=None, new_guids=None):
        # update object with new labels and predictions
        new_pred=new_pred.detach().to('cpu').numpy()
        new_labels=new_labels.detach().to('cpu').numpy()
        
        self.pred = np.concatenate((self.pred,new_pred))
        self.label = np.concatenate((self.label,new_labels))
        if new_uids is not None:
            self.uid= np.concatenate((self.uid,new_uids))
        if new_guids is not None:
            self.guid= np.concatenate((self.guid,new_guids))

    def mse(self,j=None):
        # get mean squared error of predictions
        cm=mean_squared_error(self.pred, self.label)
        return cm
    
    def mae(self):
        # get mean absolute error of predictions
        cm=mean_absolute_error(self.pred, self.label)
        return cm
    def save_df(self,fname='output.csv'):
        # get mean absolute error of predictions
        self.get_df().to_csv(fname)



#%%
def native_to_tal_fsl(path_to_img, force_new_transform=False,dof=6,output_folder = '',guid='',remove_tmp_files=True):
    '''
    path_to_img - input image in native space
    force_new_transform - if True, the native image will be transformed regardless of if 
    a transformed image exists
    returns ac/pc alinged imaged by taliarach transformation with voxel size of 1x1x1mm3
    Function that inputs a native image of the brain and 
        1) conforms it to 1x1x1mm3 voxel size + lrflip if needed
        2) performs rigid talariach transformation for ac/pc-alignment and centering of brain.
        3) if to_nifti: convert to nii
    ''' 
    if os.path.exists(path_to_img):
        if '.mgz' in path_to_img:
            print('No support for .mgz format. Convert to .nii or .nii.gz')
            return
        
        native_img = os.path.basename(path_to_img)
        if not guid: # if no output guid soecified
            if 'nii.gz' in native_img:
                guid = os.path.splitext(os.path.splitext(native_img)[0])[0]
            else:
                guid = os.path.splitext(native_img)[0]
        
        #TODO: add support for other formats than .nii.gz
        tal_img = guid + '_mni_dof_'+str(dof) + '.nii'
        bet_img = guid + '_bet.nii'
        bet_img_cp = guid + '_bet_cp.nii'
        tmp_img = guid + '_tmp.nii'
        # path_to_folder = os.path.dirname(path_to_img)
        tmp_img_path = os.path.join(output_folder,tmp_img)
        tal_img_path= os.path.join(output_folder,tal_img)
        bet_img_path= os.path.join(output_folder,bet_img)
        bet_img_path_cp= os.path.join(output_folder,bet_img_cp)

        xfm_path = os.path.join(output_folder,guid + '_mni_dof_' + str(dof)+ '.mat')
        xfm_path_cp = os.path.join(output_folder,guid + '_mni_dof_' + str(dof)+ '_cp.mat')
        
        xfm_path2 = os.path.join(output_folder,guid + '_mni_dof_' + str(dof)+ '_2.mat')
        
        try:
            fsl_path=os.environ['FSLDIR']
        except:
            fsl_path='/usr/local/fsl'
            print('please install fsl and test $FSLDIR. Trying default path: ' + fsl_path)
        template_img = os.path.join(fsl_path,'data','standard','MNI152_T1_1mm.nii.gz')
        tal_img_exist = os.path.exists(tal_img_path)
        # xfm_exist = os.path.exists(xfm_path)
        fsl_1 = fsl.FLIRT()
        fsl_2 = fsl.FLIRT()
        fsl_pre = fsl.Reorient2Std()
        
        
        if not tal_img_exist or force_new_transform:
            
            # pre-reorient first images
            fsl_pre.inputs.in_file = path_to_img
            fsl_pre.inputs.out_file = tmp_img_path
            
            fsl_pre.inputs.output_type ='NIFTI'
            fsl_pre.run()
            
            # run skull strip to calculate transformation matrix
            btr = fsl.BET()
            btr.inputs.in_file = tmp_img_path
            btr.inputs.frac = 0.7
            btr.inputs.out_file = bet_img_path
            btr.inputs.output_type ='NIFTI'
            btr.inputs.robust = True
            btr.cmdline
            btr.run() 
            
            # calculate transformation matrix - 1st attempt
            fsl_1.inputs.in_file = bet_img_path
            fsl_1.inputs.reference = template_img
            fsl_1.inputs.out_file = bet_img_path
            fsl_1.inputs.output_type ='NIFTI'
            fsl_1.inputs.dof = dof
            fsl_1.inputs.out_matrix_file = xfm_path
            fsl_1.run()
            
            # read .mat file for a quick-and-dirty assessment of whether AC-PC alignment failed completely by looking at the diagonal elements (should be close to 1's)
            f = open(xfm_path, 'r')
            l = [[num for num in line.split('  ')] for line in f ]
            matrix_1 = np.zeros((4,4))
            for m in range(4):
                for n in range(4):
                    matrix_1[m,n] = float(l[m][n])
            
            dist_1 = np.sum(np.square(np.diag(matrix_1)-1))
            print('Dist (below 0.01 generally OK): ' + str(dist_1))
            print('Transformation matrix path: ' + xfm_path)
            dist_lim = .01
            translate_lim  = 30
            if dist_1>dist_lim or matrix_1[2,3]>translate_lim : # if ac-PC failed, run without bet
                # copy bet files for debuging                
                copyfile(bet_img_path, bet_img_path_cp)
                copyfile(xfm_path, xfm_path_cp)
                
                print('------ Rerunning registration without bet ---')
                fsl_1.inputs.in_file = tmp_img_path
                fsl_1.run()
            
                f = open(xfm_path, 'r')
                l = [[num for num in line.split('  ')] for line in f ]
                matrix_2 = np.zeros((4,4))
                for m in range(4):
                    for n in range(4):
                        matrix_2[m,n] = float(l[m][n])
                
                dist_2 = np.sum(np.square(np.diag(matrix_2)-1))
                print([dist_1,dist_2])
                if (dist_1<dist_lim and dist_2<dist_lim):
                    if matrix_1[2,3]<matrix_2[2,3]:
                        # use the transform from the bet image if that was "better"
                        xfm_path=xfm_path_cp
                        print('Using bet transform, both below dist and translate smaller')
                elif dist_1<dist_2:
                    xfm_path=xfm_path_cp
                    print('Using bet transform, bet dist smaller')
            
            # apply transform
            fsl_2.inputs.in_file = tmp_img_path
            fsl_2.inputs.reference = template_img
            fsl_2.inputs.out_file = tal_img_path
            fsl_2.inputs.output_type ='NIFTI'
            fsl_2.inputs.in_matrix_file = xfm_path
            fsl_2.inputs.apply_xfm = True
            fsl_2.inputs.out_matrix_file = xfm_path2     
            fsl_2.run()

            if remove_tmp_files:
                for img in [tmp_img_path,bet_img_path,xfm_path2, bet_img_path_cp,xfm_path_cp]:#,xfm_path
                    if os.path.exists(img):
                        os.remove(img)
            else:
                print('OBS: NOT REMOVING TEMPORARY TRANSFORM FILES ')
    else:
        print(path_to_img + ' did not exist')
