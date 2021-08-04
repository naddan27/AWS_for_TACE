import numpy as np
import nibabel as nib
import os
from glob import glob
import shutil 
from tqdm import tqdm

early = "early_T1C_RAI_RESAMPLED_REG_croppedTo2222_N4_nls.nii.gz"
delayed = "delayed_T1C_RAI_RESAMPLED_REG_croppedTo2222_N4_nls.nii.gz"
t1 = "t1_nocontrast_RAI_RESAMPLED_REG_croppedTo2222_N4_nls.nii.gz"

nifti_dir = "/home/neural_network_code/Data/Patients"
patients = []
for root, dir, files in os.walk(nifti_dir):
    if early in files:
        patients.append(root)
patients.sort()

#perform the subtraction
def subtract(patient, img_name1, img_name2, save_name):
    img1 = nib.load(os.path.join(patient, img_name1))
    img2 = nib.load(os.path.join(patient, img_name2))
    img1_data = np.array(img1.dataobj)
    img2_data = np.array(img2.dataobj)

    subtraction = img1_data - img2_data
    new_img = nib.Nifti1Image(subtraction, img1.affine)
    nib.save(new_img, os.path.join(patient, save_name))

for patient in tqdm(patients):
    subtract(patient, early, t1, "early_minus_t1_before_norm.nii.gz")
    subtract(patient, early, delayed, "early_minus_delayed_before_norm.nii.gz")
