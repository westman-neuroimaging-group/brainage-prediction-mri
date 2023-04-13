# Predicting the Age of the Brain with Minimally Processed T1-weighted MRI data
*Current version: 0.02*

*brainage-prediction-mri* is a tool that takes an unprocessed T1-weighted brain MRI(in .nii or .nii.gz format) and automatically predicts the age of the subject. The model was trained on more than 15000 images from healthy individuals.

The methods used in the releasev0.02 in this repository is described in:
> TBA

The methods used in the releasev0.01 in this repository is described in:
>https://www.medrxiv.org/content/10.1101/2022.09.06.22279594v1



The network architecture is a 3D version of ResNet [He et al. (2017)](https://arxiv.org/abs/1512.03385).


**Please note that the model has not been extensively validated across different protocols, scanners or populations and should be used for research purposes only.** 
## Table of  contents

* [Getting Started](#getting-started)
   * [Prerequisites](#prerequisites)
   * [Installing](#installation)
   * [Download model weights](#download-model-weights)
* [Usage](#usage)
   * [Single case](#single-case)
   * [Multiple files](#multiple-files)
* [Train model](#train-model)
* [Troubleshooting](#troubleshooting)
* [Citation](#citation)
* [License](#license)
* [Contact](#contact)

## Getting Started

These instructions show you code prerequisites needed to run the the script and how to install them. Please note that to use this tool you need a unix-based OS (i.e. Linux or macOS) or running a Linux Virtual machine. 

### Prerequisites

**Prerequisites**
- To run `brain_age.py` you need to have the following installed:
  - Python 3.5 or higher, with the following packages:
    - numpy
    - matplotlib
    - argparse
    - glob3
    - nibabel
    - h5py
    - nipype
    - tensorboard
    - monai
  - [PyTorch 1.2](pytorch.org) or higher.
  - [FSL v6.0](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)

See below for installation instructions.

### Installation

**Download brain_age python scripts**
- Open a terminal and cd to the folder where you wish to install the tool and clone repository:
``` 
cd /path/to/your/installation/folder 
git clone https://github.com/westman-neuroimaging-group/brainage-prediction-mri.git
``` 

or press "Clone or Download" in the top right corner and unzip the repository in your folder of choice.

**Install requirements**
- Install latest FSL version (currently v6.0) by following the instructions [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation). FSL is needed for the automated registration, and called from the nibabel library. Make sure that the FSLDIR environment variable is set (run `echo $FSLDIR` in the terminal which should display the directory FSL was installed in).

We suggest that you install all python libraries in a conda environment, see instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html?#). This is not necessary in order to use the model, however.

- Install required python libraries (except for PyTorch) by executing the following (inside the brain_age_prediction_public folder):
``` 
pip install -r requirements.txt
```
- Install the latest PyTorch version (at time of writing: 1.7.1) by following the instructions [here](https://pytorch.org/). If you want to train your own model you need to download the CUDA version of PyTorch (and have a GPU + CUDA installed).

### Download model weights
You can either [train your own model](#train-model) or download our pretrained model weights that we used in our papers. To use our pretrained model weights go to [release-v0.02](https://github.com/westman-neuroimaging-group/brainage-prediction-mri/releases/tag/Brainage-release-v0.02) or [release-v0.01](https://github.com/westman-neuroimaging-group/brainage-prediction-mri/releases/tag/Brainage-release-v0.01).

## Usage
### Single case
To process an image through the model you can use the following command (inside the folder `brain_age_prediction_public`):
```
python brain_age.py --input-file /path/to/image_folder/input_filename.nii.gz --model-dir /path/to/model_weights_folder --uid new_output_filename_prefix --output-dir /path/to/output_folder
```

This command would input the image `input_filename.nii.gz`, load the pretrained weights located in `/path/to/model_weights_folder/*.pth` and produce the files:

- `new_output_filename_prefix.csv`: csv file with the predicted brain age. 
- `new_output_filename_prefix_mni_dof_6.nii`: MNI registered .nii of `input_filename.nii.gz`.
- `new_output_filename_prefix_mni_dof_6.mat`: Computed transformation matrix for the registration of FSL. Since the FLS registration is the most time consuming step of `brain_age.py`, saving the intermediate processing step can save time if you want to re-run `brain_age.py` with e.g. new trained weights in future.

If `--uid` is not provided, the output file prefix will instead be the basename of the input-file.

### Multiple files
To process all .nii images in one folder, you can e.g. execute the following command in your terminal:
```
for f in /path/to/images/*.nii; do python brain_age.py --input-file $f --output-dir /path/to/output/folder --model-dir /path/to/model/weights; done
```
This will process all .nii or .nii.gz images in the folder /path/to/images/img1.nii with the output files img1.csv, etc being saved in the specified output-folder. All images will have their own .csv file with it's associated ratings. To merge all .csv files in your output directory into a single .csv file you can run

```
awk '(NR == 1) || (FNR > 1)' /path/to/output/folder/*.csv > merged_file.csv
```
## Train model
If you want to train your own model you need to have a GPU and a GPU version of pytorch installed.

- Prepare a csv file `your_csv_file.csv` that includes full paths to the images (`path`), the chronological age at the time of the scan (`age_at_scan`), a unique ID of each image (`uid`), a unique index for each image (`indx`), to which project/cohort they belong (`Project`), and if the image belongs to train, dev or test set (for hold-out approach) or to the main partition (for cross-validation approach) (`partition`). See `sample_file_for_preprocessing_script.csv` for an example. Please note that these six columns are needed for the training script to work.
- Run the script `brain_age_trainer_preprocessing.py` with the following flags (may take several hours depending on sample size):
``` 
python brain_age_trainer_preprocessing.py --input-csv your_csv_file.csv --output-csv your_new_csv_file.csv --output-dir /path/to/folder/to/plave/registered/images
```
- The output file `your_new_csv_file.csv` contains the same info as `your_csv_file.csv` but with an added column `path_registered` containing paths to the FSL-registered files in the chosen `output-dir`. 
- To start training the model using the **hold-out** approach, run the following:
``` 
python brain_age_trainer-holdout.py --input-csv your_new_csv_file.csv --output-dir /path/to/output/folder
```
- The path specified in `output-dir` is where a timestamped folder will be created containing the trained weights of the model(s), predictions on the training and development sets (add flag `--evaluate-test-set` if you want to evaluate your test set in the end of the script), and tensorboard files monitoring the training process. (See https://pytorch.org/docs/stable/tensorboard.html for help on tensorboard.)

- To start training the model using the **cross-validation** approach, run the following:
``` 
python brain_age_trainer-crossvalidation.py --input-csv your_new_csv_file.csv --output-dir /path/to/output/folder
```
- The path specified in `output-dir` is where a timestamped folder will be created containing the trained weights of the model(s), predictions, a file with the data distribution within different cohorts, and tensorboard files monitoring the training process. (See https://pytorch.org/docs/stable/tensorboard.html for help on tensorboard.)
- To use your models, you can follow the steps in [Usage](#usage) providing the path to your output folder as `--model-dir` instead of the path to the downloaded weights.

## Troubleshooting

- Occasionally the FSL registration fails (<1% in our experience), causing the age prediction to be useless. There is currently no tool implemented to catch when this happens besides opening them with e.g. fsleyes and looking at them.

- To run FSL:s installer script python 2.x  is required, which is not included in newer versions of Ubuntu for instance. If you don't have it installed you can
```
conda create --name py2 python=2.7
conda activate py2
```
and the run `python2.7 fslinstaller.py`.

- Please report any problems with running the tool [here](https://github.com/westman-neuroimaging-group/brainage-prediction-mri/issues)


## Citation
If you use this tool in your research, in its release-v0.02 (hold-out and cross-validation approaches), please cite:
> TBA

If you use this tool in your research, in its release-v0.01 (hold-out), please cite:

>Dartora, Caroline, et al. "Predicting the Age of the Brain with Minimally Processed T1-weighted MRI Data." medRxiv (2022): 2022-09. https://www.medrxiv.org/content/10.1101/2022.09.06.22279594v1
>

## License

The code in this project is licensed under the MIT License - see [License](LICENSE.md) for details.

Please note that this tool relies on third-party software with other licesenses:
- FSL - see [FSL license](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Licence) for details.
- nipype - see [license](https://github.com/nipy/nipype/blob/master/LICENSE) for details.
- nibabel - see [license](http://nipy.org/nibabel/legal.html) for details.

## Contact
caroline.dartora@ki.se/eric.westman@ki.se
