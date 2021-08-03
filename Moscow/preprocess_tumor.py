import os
import multiprocessing
from numpy.lib.function_base import _append_dispatcher
from tqdm import tqdm
from joblib import Parallel, delayed
import psutil
from _preprocessing_functions import *
import time

#path to nifti directory
run_in_parallel = True
keep_postprocessed_liver_masks = False
nifti_dir = '/home/neural_network_code/Data/Patients/'
vols_to_process = [
        'early_T1C.nii.gz',
        'delayed_T1C.nii.gz',
        't1_nocontrast.nii.gz',
        #'portal_T1C.nii.gz'
        ]
rois_to_process = [
        'tumor_label.nii.gz' #uncomment for preprocessing for model training, comment for preprocessing for prediction
        ]
liver_prediction_name = "liver-pred-label.nii.gz"

#path to SLICER3D (used for reorienting, registering, resampling volumes)
slicer_dir = '/home/shared_software/Slicer-4.10.2-linux-amd64/Slicer'

# parameter to run orientation module
orientation = 'RAI'
# parameters to run resampling module
spacing = '1,1,5'
interp_type_vol = 'bspline'
interp_type_roi = 'nearestNeighbor'
# parameters to run registration module (will register T2 to T1 image)
transform_mode = 'Off'
transform_type='Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine'
interpolation_mode = 'BSpline'
sampling_percentage = .02
affine_transform_filename = 'affine_transform.h5'
# parameter to run bias correction module
n4_iterations = [50,50,30,20]

#####################################################################################
#run preprocessing over all patients
patients = nested_folder_filepaths(nifti_dir, vols_to_process)
patients.sort()
print(patients[0])

def all_preprocessing(patient):
    inbetween_files = []

    # 1) reorient volumes and rois
    reoriented_volumes = reorient_volume(nifti_dir, patient, vols_to_process, orientation, slicer_dir)
    inbetween_files.append(reoriented_volumes)

    reoriented_liver = reorient_volume(nifti_dir, patient, [liver_prediction_name], orientation, slicer_dir)
    inbetween_files.append(reoriented_liver)
    if len(rois_to_process) > 0:
        #first figure out if ground truth is on T2 or FLAIR
        reoriented_rois = reorient_volume(nifti_dir, patient, rois_to_process, orientation, slicer_dir)
        inbetween_files.append(reoriented_rois)

    # 2) resample to isotropic resolution
    resampled_volumes = resample_volume(nifti_dir, patient, reoriented_volumes, spacing, interp_type_vol, slicer_dir)
    inbetween_files.append(resampled_volumes)

    resampled_liver = resample_volume_using_reference(nifti_dir, patient, reoriented_liver, resampled_volumes[0], interp_type_roi, slicer_dir)
    inbetween_files.append(resampled_liver)
    if len(rois_to_process) > 0:
        #resample ROI using reference modality
        resampled_rois = resample_volume_using_reference(nifti_dir, patient, reoriented_rois, resampled_volumes[0], interp_type_roi, slicer_dir)
        inbetween_files.append(resampled_rois)

    # 3) register all patients to reference modality (i.e. modality on which ground truth was performed)
    registered_volumes = register_volume(nifti_dir, patient, resampled_volumes[0], resampled_volumes, transform_mode, transform_type, interpolation_mode, sampling_percentage, affine_transform_filename, slicer_dir)
    inbetween_files.append(registered_volumes)

    registered_liver = resampled_liver
    if len(rois_to_process) > 0:
        # use found affine to register other label maps to input space
        resampled_rois_reference = resampled_rois.pop(0)
        registered_rois = resample_volume_using_reference(nifti_dir, patient, resampled_rois, None, interp_type_roi, slicer_dir, output_transform_filename=affine_transform_filename, append_tag='')
        registered_rois.insert(0, resampled_rois_reference)
        inbetween_files.append(registered_rois)

    # 4) correct size differences between volumes here to prevent downstream issues
    ground_truth_reference_volume = registered_volumes.pop(0)
    registered_volumes = resample_volume_using_reference(nifti_dir, patient, registered_volumes, ground_truth_reference_volume, interp_type_vol, slicer_dir, append_tag='')
    registered_volumes.insert(0, ground_truth_reference_volume)

    #get postprocessed liver prediction mask to use in N4
    postprocessed_livers, postprocessed_liver_save_names = postprocess_liver_predictions(nifti_dir, patient, registered_liver)
    if not keep_postprocessed_liver_masks:
        inbetween_files.append(postprocessed_livers)

    #sometimes there is a mismatch between the mask and the volume, so we will copy the affine/header from the first volume
    registered_volumes = replace_affine_header(nifti_dir, patient, registered_volumes, ground_truth_reference_volume)
    postprocessed_livers = replace_affine_header(nifti_dir, patient, postprocessed_livers, ground_truth_reference_volume)

    for postprocessed_liver, postprocessed_liver_save_name in zip(postprocessed_livers, postprocessed_liver_save_names):
        #5) crop volumes to liver
        cropped_volumes, cropped_liver = crop_to_liver(nifti_dir, patient, registered_volumes, postprocessed_liver, append_tag='croppedTo' + postprocessed_liver_save_name)
        inbetween_files.append(cropped_volumes)
        inbetween_files.append([cropped_liver])

        if len(rois_to_process) > 0:
            cropped_rois, cropped_liver = crop_to_liver(nifti_dir, patient, registered_rois, postprocessed_liver, append_tag = 'croppedTo' + postprocessed_liver_save_name)
            inbetween_files.append(cropped_rois)
            inbetween_files.append([cropped_liver])
        
        #6) n4 bias correction
        #start looking here
        #look at the N4 mask function on documentation
        bias_corrected_volumes = n4_bias_correction(nifti_dir, patient, cropped_volumes, n4_iterations, mask_image=cropped_liver)
        inbetween_files.append(bias_corrected_volumes)

        #7) liver extraction
        liver_extracted_volumes = liver_extraction(nifti_dir, patient, bias_corrected_volumes, cropped_liver)
        #inbetween_files.append(liver_extracted_volumes)

        if len(rois_to_process) > 0:
            liver_extracted_rois = liver_extraction(nifti_dir, patient, cropped_rois, cropped_liver)
            #inbetween_files.append(liver_extracted_rois) 

        #8) normalization using voxels within liver prediction (final volume)
        normalized_liver_volumes = normalize_volume(nifti_dir, patient, liver_extracted_volumes, cropped_liver, append_tag = "NORM-for_tumor_pred")
    
        #9) binarize ROI (final roi names)
        if len(rois_to_process) > 0:
            binarized_rois = binarize_segmentation(nifti_dir, patient, cropped_rois)
            binarized_rois_nls = binarize_segmentation(nifti_dir, patient, liver_extracted_rois)

        #10) ensure label matches volume size since this will cause problems downstream (final roi data)
        if len(rois_to_process) > 0:
            final_rois = resample_volume_using_reference(nifti_dir, patient, binarized_rois, normalized_liver_volumes[0], interp_type_roi, slicer_dir, append_tag='')
            final_rois_nls = resample_volume_using_reference(nifti_dir, patient, binarized_rois_nls, normalized_liver_volumes[0], interp_type_roi, slicer_dir, append_tag='')

    delete_in_between_files(nifti_dir, patient, inbetween_files)

    #remove affine files if there were multiple inputs
    for root, dir, files in os.walk(nifti_dir):
        for f in files:
            if ".h5" in f:
                os.remove(os.path.join(root, f))

#run the preprocessing scripts
start_time = time.time()
if run_in_parallel:
    num_cores = psutil.cpu_count(logical = False)
    Parallel(n_jobs=num_cores)(delayed(all_preprocessing)(patient) for patient in patients)
else:
    for patient in tqdm(patients):
        all_preprocessing(patient)
end_time = time.time()
summarize_preprocessing(start_time, end_time, patients)
