import os
import multiprocessing

from joblib import Parallel, delayed
from _preprocessing_functions import *
import time
from tqdm import tqdm

"""
The purpose of this is to resample and register predicted ROIs to a reference volume
"""

#path to nifti directory
nifti_dir = '/home/neural_network_code/Data/Patients/'
croped_name = "2222"
rois_to_resample = ['tumor_pred_label_' + cropped_name + '.nii.gz', "tumor_label.nii.gz"]
reference_volume = 'early_T1C.nii.gz'
name_after_resampling = ['tumor-pred-label_' + cropped_name + 'og-res.nii.gz', 'tumor_label_padded.nii.gz']
run_in_parallel = True

#path to SLICER3D (used for reorienting, registering, resampling volumes)
slicer_dir = '/home/shared_software/Slicer-4.10.2-linux-amd64/Slicer'
#path to ROBEX (used for skull stripping volumes)
robex_dir = '/home/shared_software/ROBEX/runROBEX.sh'
# parameters to run resampling module
interp_type_roi = 'nearestNeighbor'
#####################################################################################
#run preprocessing over all patients
patients = nested_folder_filepaths(nifti_dir, rois_to_resample)
patients.sort()

def resample_back(patient):    
    #resample the ROI and register to reference volume
    resampled_rois = resample_volume_using_reference(nifti_dir, patient, rois_to_resample, reference_volume, interp_type_roi, slicer_dir)

    #binarize the ROI
    binarized_rois = binarize_segmentation(nifti_dir, patient, resampled_rois)

    #delete in between files
    delete_in_between_files(nifti_dir, patient, [resampled_rois])

    #rename files
    rename_files(nifti_dir, patient, binarized_rois, name_after_resampling)
    

#parallel version
start_time = time.time()
print("Starting resampling....")
print("This may take awhile")
if run_in_parallel:
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(resample_back)(patient) for patient in patients)
else:
    for patient in tqdm(patients):
        resample_back(patient)
end_time = time.time()

summarize_resampling(start_time, end_time, patients)

