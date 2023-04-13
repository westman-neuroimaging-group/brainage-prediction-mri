#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for preprocessing data with FSL prior to training. This step is done within 
the script brain_age.py but is too time consuming to do on-the-fly during training.

This script uses FSL to do a rigid registration (6 DOF) to the MNI brain, so FSL FLIRT needs to be installed (see README).

The input-csv file should have a column `path` with the full path to each image.
The output-csv file will have an additional column `path_registered`, which includes paths to the registered images located in output-dir.

@author: GM
@edited: CD
"""

import pandas as pd
from utils.misc import native_to_tal_fsl
import os
from concurrent import futures
import argparse
parser = argparse.ArgumentParser(description='Perform rigid registration to MNI brain of a dataset')
parser.add_argument('--input-csv', default='../brain_age/data/sample_short.csv', help='Path to csv file with paths to img files')
parser.add_argument('--output-csv', default='', help='Path to output csv file')
parser.add_argument('--output-dir', default='/path/to/output_file', help='Path to output csv file')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    print('Creating output-dir')
if not args.output_csv:
    args.output_csv = args.input_csv.replace('.csv','_with_registration_paths.csv')
    print('output-csv not specified, setting it to %s' % args.output_csv)
    
df = pd.read_csv(args.input_csv)


print('Starting registration. May take up to 1min/image.')
#% Running registrations in parallel
executor = futures.ProcessPoolExecutor()
futures = [executor.submit(native_to_tal_fsl,p, False, 6,args.output_dir) for p in df['path']]
for i,future in enumerate(futures):
    pass
    # print('Registering case %d of %d' % (i, len(df)))
executor.shutdown()

#% Create new column with paths to registered brains
def registered_file_names(input_file,output_dir=''):
    input_file = os.path.basename(input_file)
    output_file = os.path.join(output_dir,input_file.replace('.gz','').replace('.nii','_mni_dof_6.nii'))
    return output_file

df['path_registered'] =  [registered_file_names(f,args.output_dir) if os.path.exists(registered_file_names(f,args.output_dir)) else '' for f in df['path']]

df.to_csv(args.output_csv,index=False)
