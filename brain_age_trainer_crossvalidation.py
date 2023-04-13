#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
@author: GM
@edit: CD
#-----------------------------------------

     Brain AGE v2 - cross-validation
               
#-----------------------------------------

Age prediction of brain images with convolutional neural networks. 

---
Before using this script, please be sure that you have preprocessed your images using the "brain_age_trainer_preprocessing.py" script with:

python brain_age_trainer_preprocessing.py --input-csv your_csv_file.csv --output-csv your_new_csv_file.csv --output-dir /path/to/folder/to/plave/registered/images

After that, the files will be ready to be a input in this script.
---

This script briefly:
    1. Takes processed T1-weighted native MRI image in .nii.gz or .nii format.
    2. Divides the set in K train and development sets;
    3. Apply the next steps in training (K-1 folds), and evaluate the results on the development set (1 fold):
        3.1 Apply augmentation on the training set;
        3.2 Parameter tuning is realized with 5 convolutional neural network with 20 epochs with SGD and initial LR of lambda=0.002 that decreased with a factor of 10 every 5 epochs
        3.3 The CNN is trained in the K-1 folds, and evaluate in the development set.
        3.4 The weights of the K*5 CVs is saved on tensorboard and updated in the output folder as .pth files
        3.5 A .csv file is created with the average of precited age from the ensemble models for training and development sets


The brainAGE-v2 aggregates the K-folds development. 
Also, the CV model is in a stratified-fashion, and the input table need to describe in the column 'Project', to which project the images are.

The input .csv file needs to contain, at least, the follow columns:
    'indx': a sample order of the images in decrescent order to be able to follow changes of indexed data.
    'Project': the name of the original project of the image. It will be necessary in the stratification of groups.
    'uid': the individual number of identification of subject.
    'path_registered': the path for the registered image.
    'age_at_scan': the chronological age of the subject in the image acquisition time.
    'partition': a column that need to be fullfilled with the word 'main'. This column will be used to identify in each partition (train or test) the data will be used in each KFold.
    

"""
import os
import datetime
import numpy as np 
from utils.dataloader import mri_dset
from transforms.load_transform import load_transforms
import pandas as pd
import matplotlib.pyplot as plt
from model.model import ResNet3D
import time
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import copy
import utils.misc
import glob
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import monai
from monai.data import DataLoader, ThreadDataLoader
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# %% variables

parser = argparse.ArgumentParser(description='Training of model for brain age predictions from T1-weighted nifti images')

parser.add_argument('--input-csv', default='../brain_age/data/your_data.csv', help='Path to csv file with paths to img files and labels (chronological age)')
parser.add_argument('--output-dir', default='/path/to/brain_age/output_dir', help='Path to directory where output folder is created.')
parser.add_argument('--evaluate-test-set', dest='evaluate_test_set', action='store_false',help='not necessary because is the cross-validation model')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,metavar='LR', help='initial learning rate') 
parser.add_argument('-bs', '--batch-size', default=20, type=int,metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('-epochs', default=20, type=int,metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('-kfolds', '--kfolds', default=10, type=int,metavar='N', help='Number of KFolds cross-validation to be used (default: 10)')
parser.add_argument('--comment', default='test_public_script', help='Add comment to training session for outputdir')
parser.add_argument('--print-frequency', default=10, type=int, metavar='N',help='num batches to process before printing prediction error')
parser.set_defaults(evaluate_test_set=False)
args = parser.parse_args()

print('Settings: KFolds = ', args.kfolds)
print('N of epochs = ', args.epochs)
print('Learning Rate = ', args.lr)
print('Batch size = ', args.batch_size) 

cfg = {
        'img_dim':[160,192,160],
        'device':torch.device('cuda'),
        }

# %%
t = time.localtime()
time_str = '%2d%02d%02d_%02d.%02d.%02d' % (t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)

print('fixing rand seed')
np.random.seed(0)
torch.random.manual_seed(0)

# %% Create datasets and data loaders
print('Creating dataset from %s' % args.input_csv)
print('Images in column path_registered are assumed to have been registered ')
df_ = pd.read_csv(args.input_csv,usecols=['indx', 'Project','uid', 'guid','path_registered','age_at_scan','partition'])

# transforms
transforms_train = load_transforms(cfg,random_chance=.7)
transforms_test = load_transforms(cfg,random_chance=0)

kf = StratifiedGroupKFold(n_splits = args.kfolds)#, shuffle = True)
cv = 0
distribution = pd.DataFrame({'CV':[],'Number of samples':[], 'Distribution':[], 'Distribution related to the whole dataset':[]})
dist=pd.DataFrame()
df = df_

for train_index, dev_index in kf.split(df, df_['Project'], df_['uid']):
    df_train, df_dev = df_.loc[train_index,:], df_.loc[dev_index,:]
    cv = cv+1
    print('#######################################   FOLD ', str(cv),'  ##################################################')
    print('Index of development set: ', dev_index)
    df_train = pd.DataFrame(df_train, columns = df.columns)
    df_dev = pd.DataFrame(df_dev, columns = df.columns)
    df_train['partition'].replace({'main':'train'}, inplace = True)
    df_train['partition'].replace({'dev':'train'}, inplace = True)
    df_dev['partition'].replace({'main':'dev'}, inplace = True)
    df_dev['partition'].replace({'train':'dev'}, inplace = True)
    df = pd.concat([df_train, df_dev], ignore_index = True)
    distribution = pd.DataFrame({'CV':cv,'Number of samples':df.pivot_table(columns = ['Project'], aggfunc = 'size'), 'Distribution on test set':df_dev['Project'].value_counts(),
                                 'Distribution related to the whole dataset':df_dev['Project'].value_counts()/df_dev['Project'].count()})
    dist= pd.concat([dist,distribution], axis=0)


    print('Train size', df_train.shape)
    print('Dev size', df_dev.shape)
    results_dir=os.path.join(args.output_dir,time_str + '_'+args.comment.replace(' ','_').replace(',','_').replace('[','').replace(']','').replace('__','_'))
    writer = SummaryWriter(results_dir,comment='')
    writer.add_text('comment',args.comment)
    print('Output directory created in %s' % results_dir)
    
    #Verify existence or create a output just for data splits and predictions
    splits_folder = (os.path.join(results_dir, 'splits'))
    CHECK_FOLDER = os.path.isdir(splits_folder)
    if not CHECK_FOLDER:
        os.makedirs(splits_folder)
    
    predictions_folder = (os.path.join(results_dir, 'predictions'))
    CHECK_FOLDER = os.path.isdir(predictions_folder)
    if not CHECK_FOLDER:
        os.makedirs(predictions_folder)
        
    #Save splits
    fname = os.path.join(splits_folder, 'data_split_'+str(cv)+'.csv')
    df.to_csv(fname)
    
    #Save data distribution in the folds
    fname = os.path.join(results_dir, 'data_distribution_folds.csv')
    dist.to_csv(fname)
    
    # datasets for training and development sets
    dset_training=mri_dset(df,
                           partition='train',
                           is_training=True,
                           input_transform=transforms_train,
                           )

    dset_dev=mri_dset(df,
                           partition='dev',
                           is_training=False,
                           input_transform=transforms_test,
                           )

    loader_training = monai.data.ThreadDataLoader(
            dset_training, batch_size=args.batch_size, 
            shuffle=True,
            num_workers=10,
            pin_memory=True,
            drop_last=True
            )


    loader_dev = monai.data.ThreadDataLoader(
            dset_dev, batch_size=args.batch_size, 
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            )


    # %% Test dset and dataloader, get img needed for initializing network
    index=0
    img,label,uid,guid = dset_training.__getitem__(index)
    for img_tmp in [img,]:
        ix=32
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(img_tmp[ix,:,:])
        plt.subplot(1,3,2)
        plt.imshow(img_tmp[:,ix,:])
        plt.subplot(1,3,3)
        plt.imshow(img_tmp[:,:,ix]);plt.colorbar()
        plt.show(block=False) 
        print([img_tmp.min(),img_tmp.mean(),img_tmp.max()])
        plt.pause(3) #Shows and close the image after 3 seconds
        plt.close()
    

# %% Initialize models and optimizers

    loss_func = torch.nn.L1Loss() # optimize for mean absolute error


    models ={ # create five models and use as ensemble predictions
           'ResNet3D_3x_0':ResNet3D(img.shape,width_f=3),
           'ResNet3D_3x_1':ResNet3D(img.shape,width_f=3),
           'ResNet3D_3x_2':ResNet3D(img.shape,width_f=3),
           'ResNet3D_3x_3':ResNet3D(img.shape,width_f=3),
           'ResNet3D_3x_4':ResNet3D(img.shape,width_f=3),
           }
    optimizers = {}
    schedulers= {}
    for key in models.keys():
        #Dividing models to run in different GPUs
        print([key,count_parameters(models[key])])
        models[key] = models[key].to(cfg['device'])
        models[key]= torch.nn.DataParallel(models[key], device_ids=range(torch.cuda.device_count()))

        optimizers[key] = torch.optim.SGD(models[key].parameters(), lr=args.lr,momentum=.9)

        schedulers[key] = torch.optim.lr_scheduler.StepLR(optimizers[key], step_size=5, gamma=0.1)

    # %%
    for epoch in range(args.epochs):
        phases = ['train','dev']

        ratings = {}
        for phase in phases:
            ratings[phase] = {}
            for key in models.keys():
                ratings[phase][key] = utils.misc.StoreOutput()
                
        print('--- starting training epoch ' + str(epoch) + ' ---')
        phase='train'
        start=time.time()
        for i,(img,age,uid,guid) in enumerate(loader_training):
            age=age.type(torch.FloatTensor).to(cfg['device'])
            # since dataloading is the bottleneck we train all models at once
            for key in models.keys():
                loss=0
                start_model=time.time()
                models[key].train()
                tmp_pred,_ = models[key](img.detach().to(cfg['device']))

                loss += loss_func(tmp_pred.squeeze(),age)
                optimizers[key].zero_grad()
                loss.backward()

                optimizers[key].step()
                ratings[phase][key].update(tmp_pred.squeeze(),age,uid,guid)
                if i%args.print_frequency==0:
                    print([epoch,i,len(loader_training),key,loss.detach().to('cpu')])

        # %%
        # -------------------------- evaulate dev set ----------------------
        phase='dev'
        with torch.no_grad():
            for i,(img,age,uid,guid) in enumerate(loader_dev):

                age=age.type(torch.FloatTensor).to(cfg['device'])

                for key in models.keys():
                    models[key].eval()           
                    tmp_pred,_ = models[key](img.detach().to(cfg['device']))
                    ratings[phase][key].update(tmp_pred.squeeze_(1),age,uid,guid)
        print(datetime.datetime.now())
        print('finished epoch %d' % epoch) 

        # %% Compute epoch statistics and add to tensorboard summary writer

        maes=  {} # dictionary with MAE
        for phase in phases:
            predictions = [] # for calculating ensemble predictions
            for key in models.keys():
                mae = ratings[phase][key].mae()
                maes[phase+'_'+key] = mae

                df_tmp = ratings[phase][key].get_df()
                predictions.append(df_tmp['predicted_age'].to_numpy())
                correlation = np.corrcoef(df_tmp['predicted_age'],df_tmp['age_at_scan'])[0][1] # pearson correlation
                lims = [df_tmp['age_at_scan'].min()-3,df_tmp['age_at_scan'].max()+3] # x and y lims for plotting

                # generate scatterplots and add to tensorboard
                fig = plt.figure(figsize=(12,8))
                plt.scatter(df_tmp['age_at_scan'],df_tmp['predicted_age'],alpha=1)
                tstr= '%s - MAE: %.2f - rho: %.3f (%s)'% (phase,mae,correlation,key)
                plt.title(tstr,fontsize=16)
                plt.plot(lims,lims,'k:');plt.grid();
                plt.xlim(lims);plt.ylim(lims);
                plt.xlabel('Chronological age',fontsize=16)
                plt.ylabel('Predicted age',fontsize=16)
                writer.add_figure('predictions/'+phase+'_'+key,fig,epoch,cv)
                plt.close()
                fname = os.path.join(predictions_folder,'predictions_'+phase+'_'+key+'_cv_'+str(cv)+'.csv')
                ratings[phase][key].save_df(fname)

            maes[phase + '_ensemble'] = mean_absolute_error(df_tmp['age_at_scan'], df_tmp['predicted_age'])
            fname = os.path.join(results_dir,'maes'+phase+str(cv)+'.csv')
            
            # calculate ensemble predictions
            df_tmp['predicted_age'] = np.mean(predictions,axis=0)
            fname = os.path.join(results_dir,'predictions_'+phase+'_ensemble_cv_'+str(cv)+'.csv')
            df_tmp.to_csv(fname,index=False)

            #For iteractive view of predictions in each folder
            for cvs in str(cv):
                correlation = np.corrcoef(df_tmp['predicted_age'],df_tmp['age_at_scan'])[0][1] # pearson correlation
                lims = [df_tmp['age_at_scan'].min()-3,df_tmp['age_at_scan'].max()+3] # x and y lims for plotting
                key = 'ensemble'
                # generate scatterplots and add to tensorboard
                fig = plt.figure(figsize=(12,8))
                plt.scatter(df_tmp['age_at_scan'],df_tmp['predicted_age'],alpha=1)
                tstr= '%s - MAE: %.2f - rho: %.3f (%s)'% (phase,mae,correlation,key)
                plt.title(tstr,fontsize=16)
                plt.plot(lims,lims,'k:');plt.grid();
                plt.xlim(lims);plt.ylim(lims);
                plt.xlabel('Chronological age',fontsize=16)
                plt.ylabel('Predicted age',fontsize=16)
                writer.add_figure('predictions_ensemble/'+phase,fig,cv)
                plt.close()
                fname = os.path.join(predictions_folder,'predictions_'+phase+'_ensemble.csv')

        #add MAE to tensorboard    
        writer.add_scalars('mae',maes,epoch)
        maes_list=maes.items()
        maes_list=pd.DataFrame(maes_list)
        maes_list.to_csv(fname)

        # add learning rates to tensorboatd and update learning rate scheduler
        lrs = {}
        for key in models.keys(): # loop over models
            lrs[key] = optimizers[key].param_groups[0]['lr']
            schedulers[key].step()

        writer.add_scalars('lr',lrs,epoch)
        # %% save model weights
        for key in models.keys():
            fname=results_dir+'/'+key+'_cv_'+str(cv)+'.pth'
            print('saving weights in %s' % fname)
            torch.save(models[key].to('cpu').state_dict(), fname)
            models[key].to(cfg['device'])
            plt.show()
            

# open all csv files with the ensemble predictions in each CV for both phases and all CVphases = ['dev', 'train']
phases = ['dev', 'train']

for phase in phases:
    all_files = glob.glob(os.path.join(results_dir, 'predictions_'+phase+'_ensemble*'))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True).sort_values(by=['guid'], axis=0, ascending=True)

    bio_age = df_merged.groupby('guid', as_index=False)['age_at_scan'].mean()
    pred_age_mean = df_merged.groupby('guid', as_index=False)['predicted_age'].mean()
    pred_age_std = df_merged.groupby('guid', as_index=False)['predicted_age'].std()

    bio_age['predicted_age_mean']=pred_age_mean['predicted_age']
    bio_age['predicted_age_std']=pred_age_std['predicted_age']
    bio_age['uid']=df_merged['uid']

    mae = mean_absolute_error(bio_age['age_at_scan'], bio_age['predicted_age_mean'])
    correlation_ensemble = np.corrcoef(bio_age['predicted_age_mean'],bio_age['age_at_scan'])[0][1] # pearson correlation
    lims = [bio_age['age_at_scan'].min()-3,bio_age['age_at_scan'].max()+3] # x and y lims for plotting
    key = 'ensemble_'+phase
    # generate scatterplots and add to tensorboard
    fig = plt.figure(figsize=(12,8))
    plt.scatter(bio_age['age_at_scan'],bio_age['predicted_age_mean'],alpha=1)
    tstr= '%s - MAE: %.2f - rho: %.3f (%s)'% (phase,mae,correlation,key)
    plt.title(tstr,fontsize=16)
    plt.plot(lims,lims,'k:');plt.grid();
    plt.xlim(lims);plt.ylim(lims);
    plt.xlabel('Chronological age',fontsize=16)
    plt.ylabel('Predicted age',fontsize=16)
    writer.add_figure('predictions_ensemble_mean/'+phase+'_'+key,fig,epoch,cv)
    plt.close()
    
    fname = os.path.join(results_dir, 'predictions_'+phase+'_final_result.csv')
    bio_age.to_csv(fname)
