"""
@author: GM
@edited: CD

Age prediction of brain images convolutional neural networks. The script briefly:
    1. Takes a single unprocessed T1-weighted native MRI image in .nii.gz or .nii format.
    2. Uses FSL Flirt for a rigid registration (AC-PC alignment) and interpolation 
    to 1x1x1mm3 voxel size.
    3. An ensemble of convolutional neural networks  predicts the age of the indivual, 
    4. A .csv file 'uid'.csv is created with the average predicted age from the ensemble models.

The model was trained on more than 15000 images from the following cohorts: UK biobank, ADNI, AIBL and GENIC.

Method and results are detailed in paper: #TODO

To run:
python3 brain_age.py  --input-file /path/to/img.nii.gz --uid output_file_name_prefix --model-dir /path/to/trained/model/weights --output-dir /path/to/output/dir

"""
import pandas as pd
import torch
import argparse
import os 
import numpy as np

from collections import OrderedDict
import glob
import datetime

from utils.misc import native_to_tal_fsl
from model.model import ResNet3D
from transforms.load_transform import load_transforms


parser = argparse.ArgumentParser(description='Brain age prediction from unprocessed T1-weighted nifti images')
parser.add_argument('--model-dir', type=str,default='/path/to/the/path/with/trained/model/weights', help='Path to directory containing the folders trained network weights')
parser.add_argument('--input-file', default='/path/to/your/input/nifti/registered/file/my_registered_nifti.nii.gz', help='Absoulute path to the input MRI file in (file assumed to be in .nii or .nii.gz format)')
parser.add_argument('--output-dir', default='/path/to/brain_age/output_dir', help='Path to directory where all output files. Directory will be created if it doesn\'t exist')
parser.add_argument('--uid', default='', type=str,help='Chosen unique id for output files that will located at output-dir/{uid_mni_dof_6.nii,uid.csv,uid_coronal.jpg}')
parser.add_argument('--no-new-registration', dest='registration', action='store_false',help='If a previous AC/PC-alignment exists (file output_folder/uid_mni_dof_6.nii) then setting this flag will use previous registration. If there is no previous transform, the transform will be performed anyway.')
parser.set_defaults(registration=True) 
args = parser.parse_args()

print('---- Started age prediction ----')
print('Input file: %s' % args.input_file)
print('Output files %s: ' % args.output_dir + '/'+args.uid+'*' )
print('Model directory %s: ' % args.model_dir)
print('Force new registration: %s ' % str(args.registration))

args.device =  torch.device('cpu')
timestamp= '{:%Y-%m-%d_%H_%M_%S}'.format(datetime.datetime.now())
fname= os.path.join(args.output_dir,args.uid + '_info.log') 

# Check that input parameters are OK
assert os.path.exists(args.input_file), 'input-file not specified or does not exist'
assert args.output_dir, 'output-dir not specified'
assert '.nii' in os.path.basename(args.input_file), 'input-file %s should be in .nii or .nii.gz format' % args.input_file
if not args.uid:
    args.uid=os.path.basename(args.input_file)
    args.uid = args.uid[:args.uid.find('.nii')]
    print('uid not specified. Automatically setting uid to %s' % args.uid)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    print('Creating output-dir')

# Perform registration using FSL
if args.registration:
    print('Performing rigid registration of input image to MNI template (AC-PC alignment)')
else:
    print('If registration not found rigid registration of input image to MNI template (AC-PC alignment) is performed, else use previous registration')
dof=6 # degrees of freedom of transform
native_to_tal_fsl(args.input_file, force_new_transform=args.registration,dof=dof,output_folder =args.output_dir,guid=args.uid)
tal_path = os.path.join(args.output_dir,args.uid + '_mni_dof_'+str(dof) +'.nii')

# Log parameters in output csv file
rating_dict = OrderedDict()
rating_dict['uid'] = [args.uid]
rating_dict['model-dir'] = [args.model_dir]
#%% Load transforms and parameters needed for this
params = {'ac_pc':True,'img_dim':[160,192,160]} # img_dim is dimensions used for training
transforms_test = load_transforms(params,random_chance=0)
#%% Load image with transforms
img = transforms_test(tal_path).unsqueeze(0)

#%% Initialize model
model_paths = np.sort(glob.glob(args.model_dir +'/*.pth'))
models ={}
for i,m in enumerate(model_paths):
    models[i] = ResNet3D(np.array(params['img_dim'])//2,width_f=3)

#%% Load weights and run prediction

predicted_ages = np.zeros(len(models))
for i,key in enumerate(models.keys()):
    # loading weights
    model_checkpoint = torch.load(model_paths[i],map_location='cpu')
    new_model_checkpoint = OrderedDict()
    for k, v in model_checkpoint.items():
        name = k[7:] # remove module. from items in model_checkpoint
        new_model_checkpoint[name] = v
    models[key].load_state_dict(new_model_checkpoint)
    models[key] = models[key].to('cpu')
    # evaluating model
    with torch.no_grad():
        models[key].eval()
        tmp,_ = models[key](img)
        predicted_ages[i] = tmp.detach().numpy()
        
print('---- Ages predicted from each individual model ---')
print(predicted_ages)
print('--'*20)
#%% Save
rating_dict['predicted_age_mean'] = predicted_ages.mean()
rating_dict['predicted_age_std'] = predicted_ages.std()
csv_name= args.output_dir + '/'+args.uid+'.csv'
print('Saving ensemble mean %.3f with std %.3f in csv file at %s' % (rating_dict['predicted_age_mean'],rating_dict['predicted_age_std'],csv_name))
pd.DataFrame(rating_dict).to_csv(csv_name,index=False)
print('Done!')
