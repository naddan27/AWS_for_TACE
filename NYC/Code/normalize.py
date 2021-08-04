import os
import multiprocessing
from numpy.lib.function_base import _append_dispatcher
from tqdm import tqdm
from joblib import Parallel, delayed
import psutil
from _preprocessing_functions import *
import time
from pqdm.processes import pqdm

#path to nifti directory
run_in_parallel = True
keep_postprocessed_liver_masks = False
nifti_dir = '/home/neural_network_code/Data/Patients/'
vols_to_process = [
        'early_minus_t1_before_norm.nii.gz',
        'early_minus_delayed_before_norm.nii.gz'
        ]

#path to SLICER3D (used for reorienting, registering, resampling volumes)
slicer_dir = '/home/shared_software/Slicer-4.10.2-linux-amd64/Slicer'

#####################################################################################
#run preprocessing over all patients
patients = nested_folder_filepaths(nifti_dir, vols_to_process)
patients.sort()

def normalize(patient):
    normalized_liver_volumes = normalize_volume(nifti_dir, patient, vols_to_process, None, append_tag = "NORM-for_tumor_pred")

if run_in_parallel:
    num_cores = psutil.cpu_count(logical = False)
    pqdm(patients, normalize, n_jobs = num_cores)
else:
    for patient in tqdm(patients):
        normalize(patient)
