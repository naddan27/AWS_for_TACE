import os
import errno
import shutil
import numpy as np
import nibabel as nib
from scipy.ndimage.measurements import label as label2
import operator
from subprocess import call
from nipype.interfaces.ants import N4BiasFieldCorrection

#Function to convert all dicoms to Nifti format
def convert_dicom_to_nifti(dicom_dir, nifti_dir, patient, dcm2niix_dir, output_vol_name=None):
    #filepath to input dicom folder
    dicom_folder = dicom_dir + patient
    os.chdir(dicom_folder)
    #make output patient folder if it does not exist
    output_nifti_folder = nifti_dir + patient
    if not os.path.exists(output_nifti_folder):
        os.makedirs(output_nifti_folder)
    if output_vol_name == None:
        #grab all folders in dicom directory
        dicom_volumes = next(os.walk('.'))[1]
        for dicom_volume in dicom_volumes:
            input_dicom_volume_folder = dicom_folder + '/' + dicom_volume
            convert_command = [dcm2niix_dir + ' -z y -f ' + dicom_volume + ' -o "' + output_nifti_folder + '" "' + input_dicom_volume_folder + '"']
            call(' '.join(convert_command), shell=True)
    else:
        convert_command = [dcm2niix_dir + ' -z y -f ' + output_vol_name + ' -o "' + output_nifti_folder + '" "' + dicom_folder + '"']
        call(' '.join(convert_command), shell=True)
    #return created file names
    os.chdir(output_nifti_folder)
    output_filenames = next(os.walk('.'))[2]
    return output_filenames

#Function to change all images to desired orientation
def reorient_volume(nifti_dir, patient, vols_to_process, orientation, slicer_dir):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, orientation)
    #orientation module
    module_name = 'OrientScalarVolume'
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        orientation_command = [slicer_dir, '--launch', module_name,  '"' + input_filepath + '" "' + output_filepath + '"', '-o', orientation]
        call(' '.join(orientation_command), shell=True)
    #return created file names
    return output_filenames

#Function to compute affine registration between moving (low res scan) and fixed (high res scan that you are registering all other sequences to) volume
def register_volume(nifti_dir, patient, fixed_volume, vols_to_process, transform_mode, transform_type, interpolation_mode, sampling_percentage, output_transform_filename, slicer_dir, append_tag='REG'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    input_fixed_vol_filepath, output_fixed_vol_filepath, input_others_filepaths, output_others_filepaths = choose_volume(fixed_volume, vols_to_process, input_filepaths, output_filepaths)
    #rename the fixed volume for consistency
    os.rename(input_fixed_vol_filepath[0], output_fixed_vol_filepath[0])
    for i, (input_others, output_others) in enumerate(zip(input_others_filepaths, output_others_filepaths)):
        if len(input_others_filepaths) > 1:
            temp_output_transform_filename = output_transform_filename[:-3] + '_' + str(i) + '.h5'
        else:
            temp_output_transform_filename = output_transform_filename
        affine_registration_command = [slicer_dir,'--launch', 'BRAINSFit', '--fixedVolume', '"' + output_fixed_vol_filepath[0] + '"', '--movingVolume', '"' + input_others + '"', '--transformType', transform_type, '--initializeTransformMode', transform_mode, '--interpolationMode', interpolation_mode, '--samplingPercentage', str(sampling_percentage), '--outputTransform', temp_output_transform_filename, '--outputVolume', output_others]
        call(' '.join(affine_registration_command), shell=True)
    #return created file names
    return output_filenames

#Function to resample all volumes to desired spacing
def resample_volume(nifti_dir, patient, vols_to_process, spacing, interp_type, slicer_dir, append_tag='RESAMPLED'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    #resampling module
    module_name = 'ResampleScalarVolume'
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        resample_scalar_volume_command = [slicer_dir,'--launch', module_name, '"' + input_filepath + '" "' + output_filepath + '"', '-i', interp_type, '-s', spacing]
        call(' '.join(resample_scalar_volume_command), shell=True)
    #return created file names
    return output_filenames

#Function to resample all volumes using a reference volume
def resample_volume_using_reference(nifti_dir, patient, vols_to_process, reference_volume, interp_type, slicer_dir, output_transform_filename=None, append_tag='RESAMPLED'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    #resampling module
    module_name = 'ResampleScalarVectorDWIVolume'
    if reference_volume != None:
        reference_volume_filepath = nifti_dir + patient + '/' + reference_volume
    if interp_type == 'nearestNeighbor':
        interp_type = 'nn'
    else:
        interp_type = 'bs'
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
        if reference_volume != None:
            resample_scalar_volume_command = [slicer_dir,'--launch', module_name, '"' + input_filepath + '" "' + output_filepath + '"', '-i', interp_type, '-R', reference_volume_filepath]
        else:
            if len(input_filepaths) > 1 or not os.path.exists(nifti_dir + patient + '/' + output_transform_filename):
                temp_output_transform_filename = output_transform_filename[:-3] + '_' + str(i) + '.h5'
            else:
                temp_output_transform_filename = output_transform_filename
            resample_scalar_volume_command = [slicer_dir,'--launch', module_name, '"' + input_filepath + '" "' + output_filepath + '"', '-i', interp_type, '-f', temp_output_transform_filename]
        call(' '.join(resample_scalar_volume_command), shell=True)
    #return created file names
    return output_filenames

#Function to perform N4 bias correction
def n4_bias_correction(nifti_dir, patient, vols_to_process, n4_iterations, mask_image=None, append_tag='N4'):
	#input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        n4 = N4BiasFieldCorrection(output_image = output_filepath)
        n4.inputs.input_image = input_filepath
        n4.inputs.n_iterations = n4_iterations
        if mask_image != None:
            n4.inputs.mask_image = os.path.join(nifti_dir + patient, mask_image)
        n4.run()
    #return created file names
    return output_filenames

def get_non_zero_mask(nifti_dir, patient, vols_to_process, append_tag='mask'):
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
        vol_mask = (vol != 0).astype(np.int)
        nib_vol_mask = nib.Nifti1Image(vol_mask, affine, header=header)
        nib.save(nib_vol_mask, output_filepath)
    #return created file names
    return output_filenames

def get_first_nonzero_ix(array, reverse):
    if reverse:
        for i, x in enumerate(array[::-1]):
            if x != 0:
                return len(array) - 1 - i
    else:
        for i, x in enumerate(array):
            if x != 0:
                return i
    return -1

def replace_affine_header(nifti_dir, patient, vols_to_process, reference_volume, append_tag=''):
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    reference_vol_nib = nib.load(os.path.join(nifti_dir + patient, reference_volume))
    affine = reference_vol_nib.get_affine()
    header = reference_vol_nib.get_header()
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
        #load nifti volume
        vol = nib.load(input_filepath).get_data()
        vol_new = nib.Nifti1Image(vol, affine, header=header)
        nib.save(vol_new, output_filepath)
    #return created file names
    return output_filenames

def mask_volume(nifti_dir, patient, vols_to_process, mask, append_tag='masked'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    mask_vol = nib.load(mask).get_data()
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
	    #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
        vol_masked = vol * mask_vol
        nib_vol_masked = nib.Nifti1Image(vol_masked, affine, header=header)
        nib.save(nib_vol_masked, output_filepath)
    #return created file names
    return output_filenames

#Function to perform normalization (if no mean/std is given, will perform per-volume mean zero, standard deviation one normalization by default); reference volume will be used to generate appropriate skull mask; skull_mask_volume is a shortcut used in ALD pre-processing
def normalize_volume(nifti_dir, patient, vols_to_process, reference_mask, append_tag='NORM'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
	    #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()

        #load the reference mask
        if reference_mask != None:
            ref_img = nib.load(os.path.join(nifti_dir, patient, reference_mask))
            ref_data = ref_img.get_data()
            non_zero_ref_ix = np.nonzero(ref_data)
        else:
            non_zero_ref_ix = np.nonzero(vol)

        #normalize the data
        mean, std = np.mean(vol[non_zero_ref_ix]), np.std(vol[non_zero_ref_ix])
        vol_norm = np.copy(vol)
        vol_norm[non_zero_ref_ix] = (vol_norm[non_zero_ref_ix] - mean) / std

        nib_vol_norm = nib.Nifti1Image(vol_norm, affine, header=header)
        nib.save(nib_vol_norm, output_filepath)
    #return created file names
    return output_filenames

#function to binarize ROI
def binarize_segmentation(nifti_dir, patient, roi_to_process, append_tag='BINARY-label'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, roi_to_process, append_tag)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
	    #load nifti volume
        nib_roi = nib.load(input_filepath)
        affine = nib_roi.get_affine()
        header = nib_roi.get_header()
        roi = nib_roi.get_data()
	    #binarize non-zero intensity values
        roi[np.nonzero(roi)] = 1
        nib_roi_binary = nib.Nifti1Image(roi, affine, header=header)
        nib.save(nib_roi_binary, output_filepath)
    #return created file names
    return output_filenames

#function to rescale intensity range (if no range is given, will default to rescaling range to [0,1])
def intensity_rescale_volume(nifti_dir, patient, vols_to_process, rescale_range=np.array([0,1]), append_tag='RESCALED'):
    if len(rescale_range.shape) == 1:
        rescale_range = np.tile(rescale_range, (len(vols_to_process), 1))
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
	    #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
	    #rescale intensities to new min and max
        old_min, old_max = np.min(vol), np.max(vol)
        new_min, new_max = rescale_range[i, :]
        rescaled_vol = (vol - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
        nib_rescaled_vol = nib.Nifti1Image(rescaled_vol, affine, header=header)
        nib.save(nib_rescaled_vol, output_filepath)
    #return created file names
    return output_filenames

#function to change extremely small values to 0 (in case there were rounding errors during resampling/resizing operations)
def round_volume(nifti_dir, patient, vols_to_process, decimals=5, append_tag=''):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
	    #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
	    #round small intensities to zero
        vol[np.abs(vol) < (10**(-decimals))] = 0
        nib_round_vol = nib.Nifti1Image(vol, affine, header=header)
        nib.save(nib_round_vol, output_filepath)
    #return created file names
    return output_filenames

#function to rename files
def rename_volumes(nifti_dir, patient, vols_to_process, save_names, append_tag=''):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag, save_names=save_names)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
	    #rename volumes
        os.rename(input_filepath, output_filepath)
    #return created file names
    return output_filenames

#function to generate ADC map from B0 and B1000 image
def create_adc_volume(nifti_dir, patient, vols_to_process, append_tag='ADC'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for i, input_filepath in enumerate(input_filepaths):
        #load nifti volume
        nib_vol = nib.load(input_filepath)
        vol = nib_vol.get_data()
        if i == 0:
            affine = nib_vol.get_affine()
            header = nib_vol.get_header()
            b_vols = np.zeros(vol.shape + (2,))
            idx_zero = []
        idx_zero.append(np.where(vol <= 0))
        vol[idx_zero[i]] = 1
        b_vols[...,i] = vol
    ADC = np.log(np.divide(b_vols[...,0], b_vols[...,1])) / -1000
    for indexes in idx_zero:
	    ADC[indexes] = 0
    nib_ADC = nib.Nifti1Image(ADC, affine, header=header)
    save_name = append_tag + '.nii.gz'
    nib.save(nib_ADC, nifti_dir + patient + '/' + save_name)
    return [save_name]

#function to threshold probability masks as requested thresholds (will binarize at 0.5 as default with no tags appended to generated label maps)
def threshold_probability_mask(nifti_dir, patient, vols_to_process, thresholds=[0.5]):
    #input/output filepaths
    input_folder = nifti_dir + patient
    input_filepaths = [input_folder + '/' + i for i in vols_to_process]
    for input_filepath in input_filepaths:
        probability_vol, affine, header = load_nifti_volume(input_filepaths=[input_filepath])
        #binarize predicted label map at requested thresholds
        for threshold in thresholds:
            probability_vol_binarized = (probability_vol[...,0] >= threshold).astype(int)
            #save output
            save_name_vol = 'threshold_' + str(threshold) + '_pred-label.nii.gz'
            save_nifti_volume(input_filepath, [save_name_vol], [probability_vol_binarized], affine=affine, header=header)

#function to get all filepaths if there are nested folders (and only choose folders that have all the necessary volumes)
def nested_folder_filepaths(nifti_dir, vols_to_process=None):
    if vols_to_process == None:
        relative_filepaths = [os.path.relpath(directory_paths, nifti_dir) for (directory_paths, directory_names, filenames) in os.walk(nifti_dir) if len(filenames)!=0]
    else:
        relative_filepaths = [os.path.relpath(directory_paths, nifti_dir) for (directory_paths, directory_names, filenames) in os.walk(nifti_dir) if all(vol_to_process in filenames for vol_to_process in vols_to_process)]
    return relative_filepaths 

#function to copy files (if output names are not given, will use source directory names)
def copy_and_move_files(input_dir, output_dir, file_names, output_file_names=None):
    input_filepaths = [input_dir + '/' + i for i in file_names]
    if all(os.path.exists(input_filepath) for input_filepath in input_filepaths):
        if output_file_names == None:
            output_file_names = file_names
        output_filepaths = [output_dir + '/' + i for i in output_file_names]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
            shutil.copy(input_filepath, output_filepath)

#function to copy entire folder and subdirectories to new location
def copy_and_move_folders(input_dir, output_dir):
    try:
        shutil.copytree(input_dir, output_dir)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(input_dir, input_dir)
        else:
            print('Directory not copied. Error: %s' % e)

#helper function to load nifti volumes as numpy arrays (along with affine and header information if requested)
def load_nifti_volume(input_filepaths=None, vols_to_process=None, load_affine_header=True):
    if vols_to_process != None:
        input_filepaths = [input_filepaths + vol_to_process for vol_to_process in vols_to_process]
    for j, input_filepath in enumerate(input_filepaths):
        nib_vol = nib.load(input_filepath)
        image = nib_vol.get_data()
        if j == 0:
            affine = nib_vol.get_affine()
            header = nib_vol.get_header()
            all_volumes = np.zeros((image.shape) + ((len(input_filepaths),)))
        all_volumes[...,j] = image
    if load_affine_header == True:
        return all_volumes, affine, header
    else:
        return all_volumes

#helper function to save numpy arrays as nifti volumes (using affine and header if given)
def save_nifti_volume(input_filepath, save_names, numpy_volume_list, affine=None, header=None):
    for j, (save_name, save_vol) in enumerate(zip(save_names, numpy_volume_list)):
        if header==None and affine==None:
            affine = np.eye(len(save_vol.shape) + 1)
            save_vol_nib = nib.Nifti1Image(save_vol, affine)
        else:
            save_vol_nib = nib.Nifti1Image(save_vol, affine, header=header)
        nib.save(save_vol_nib, input_filepath + save_name)

#helper function to generate input/output filepaths
def generate_filepaths(data_dir, patient_name, vols_to_process, append_tag, save_names=None):
    #filepath to input patient folder
    input_folder = data_dir + patient_name
    os.chdir(input_folder)
    #filepath to volumes
    input_filepaths = [input_folder + '/' + i for i in vols_to_process]
    #output volume names and file paths
    if save_names != None:
        output_filenames = save_names
        output_filepaths = [input_folder + '/' + i for i in output_filenames]
    elif append_tag == '' or append_tag == None:
        output_filenames = vols_to_process
        output_filepaths = input_filepaths
    else:
        output_filenames = [i[:i.find('.nii')] + '_' + append_tag + '.nii.gz' for i in vols_to_process]
        output_filepaths = [input_folder + '/' + i for i in output_filenames]
    return input_filepaths, output_filenames, output_filepaths

#helper function to find volume of interest from list of volumes to process
def choose_volume(vol_special, vols_to_process, input_filepaths, output_filepaths):
    index_vol_special = np.array([vol_special in vol for vol in vols_to_process]) 
    input_vol_special_path = [i for (i, j) in zip(input_filepaths, index_vol_special) if j]
    output_vol_special_path = [i for (i, j) in zip(output_filepaths, index_vol_special) if j]
    input_vol_paths = [i for (i, j) in zip(input_filepaths, ~index_vol_special) if j]
    output_vol_paths = [i for (i, j) in zip(output_filepaths, ~index_vol_special) if j]
    return input_vol_special_path, output_vol_special_path, input_vol_paths, output_vol_paths

def summarize_resampling(start_time, end_time, patients):
    total_time = end_time - start_time
    print()
    print("***********************************************")
    print("*************  Resampling Summary *************")
    print("Patients resampled:", len(patients))
    print("Time to finish:", total_time)
    print("***********************************************")

def summarize_preprocessing(start_time, end_time, patients):
    total_time = end_time - start_time
    print()
    print("***********************************************")
    print("***********  Preprocessing Summary ************")
    print("Patients Preprocessed:", len(patients))
    print("Time to finish:", total_time)
    print("***********************************************")

def delete_in_between_files(nifti_dir, patient, array_of_file_names):
    for file_names in array_of_file_names:
        for x in file_names:
            fp = os.path.join(nifti_dir, patient, x)
            if os.path.exists(fp):
                os.remove(fp)

def rename_files(nifti_dir, patient, current_file_names, new_file_names):
    for current_file_name, new_file_name in zip(current_file_names, new_file_names):
        src = os.path.join(nifti_dir, patient, current_file_name)
        dest = os.path.join(nifti_dir, patient, new_file_name)
        shutil.move(src, dest)

def extract_largest_connected_comp2(liver_data, voxel_vol = 5, pick_right_of_top_two = True):    
    #get a mask where all the connected regions have the same label
    # conn_comp = label(np.round(liver_data),connectivity=3).astype(np.float)
    conn_comp, num_features = label2(np.round(liver_data), structure = np.ones((3,3,3)))
    
    #get the size of each connected component and put in into a dictionary
    conn_comp_size = dict()
    
    #get the size of each connected component
    for i in range(1,len(np.unique(conn_comp))): #skip the first one as this is background
        conn_comp_size[i] = np.sum(conn_comp == np.unique(conn_comp)[i])
    
    #sort the dictionary by the size of the region
    sorted_conn_comp_ct = sorted(conn_comp_size.items(), key=operator.itemgetter(1))
    sorted_conn_comp_ct.reverse()

    #if there is only one connected component, return that
    if len(sorted_conn_comp_ct) == 1:
        conn_comp_label = sorted_conn_comp_ct[0][0]
        curr_mask = np.zeros(liver_data.shape)
        idx = np.where(conn_comp==conn_comp_label)
        curr_mask[idx] = 1
        return curr_mask

    #get a mask of the top largest connected component subset
    nPicktop = 2
    conn_comp_selected_masks = []
    sum_dist = []
    for i in range(nPicktop):
        conn_comp_label = sorted_conn_comp_ct[i][0]
        curr_mask = np.zeros(liver_data.shape)
        idx = np.where(conn_comp==conn_comp_label)
        curr_mask[idx] = 1
        conn_comp_selected_masks.append(curr_mask)
        sum_dist.append(np.sum(curr_mask))
    
    vol_dist = [x*voxel_vol for x in sum_dist]
    vol_dist = np.array(vol_dist)

    if pick_right_of_top_two:
        if all(vol_dist > (150000 * 5)): #check which one is more right (patient POV) if both meet size requirement
            ix_liver = [np.where(x == 1) for x in conn_comp_selected_masks]
            x_location_mean = [np.mean(x[1]) for x in ix_liver]
            if x_location_mean[1] < x_location_mean[0]: #second largest is more right than first
                return conn_comp_selected_masks[1]

    #return the new liver data with only the largest connected component as np array
    return conn_comp_selected_masks[0]

def box_liver_2D(liver_data):
    postprocessed_data = np.zeros(liver_data.shape)

    assert len(liver_data.shape) == 3
    for i in range(liver_data.shape[2]):
        liver_slice = liver_data[:,:,i]

        #find the indices on the left and right sides
        collapse_to_horizontal_vector = np.sum(liver_slice, axis = 0)
        left_ix = get_first_nonzero_ix(collapse_to_horizontal_vector, False)

        if left_ix == -1: #slice is empty
            continue
        right_ix = get_first_nonzero_ix(collapse_to_horizontal_vector, True)

        #find the indices on the top and bottom side
        collapse_to_vertical_vector = np.sum(liver_slice, axis = 1)
        top_ix = get_first_nonzero_ix(collapse_to_vertical_vector, False)
        bottom_ix = get_first_nonzero_ix(collapse_to_vertical_vector, True)

        postprocessed_data[top_ix:bottom_ix+1, left_ix:right_ix+1, i] = 1
    
    return postprocessed_data
 
def expand_border_2(offset_baseline, offsets):
    offsets = np.array(offsets)

    new_liver_seg = offset_baseline.copy()

    for i in range(2): #move 0-left, 1-right
        for j in range(2): #move 0-forward, 1-backward
            for axis0_shift in range(np.abs(offsets[0,i])+1):
                for axis1_shift in range(np.abs(offsets[1,j])+1):
                    axis0_shift_cp = axis0_shift
                    if offsets[0,i] < 0:
                        axis0_shift_cp *= -1

                    axis1_shift_cp = axis1_shift
                    if offsets[1,j] < 0:
                        axis1_shift_cp *= -1
                    #print("i:", i, ",j:", j, "axis0:", axis0_shift_cp, "axis1:", axis1_shift_cp)
                    ix_nonzero = np.transpose(np.array(np.where(offset_baseline != 0)))

                    ix_nonzero[:,0] += axis0_shift_cp
                    less_than_0 = np.where(ix_nonzero[:,0] < 0)
                    greater_than_shape = np.where(ix_nonzero[:,0] >= offset_baseline.shape[0])
                    ix_nonzero[less_than_0,0] = 0
                    ix_nonzero[greater_than_shape,0] = offset_baseline.shape[0]-1

                    ix_nonzero[:,1] += axis1_shift_cp
                    less_than_0 = np.where(ix_nonzero[:,1] < 0)
                    greater_than_shape = np.where(ix_nonzero[:,1] >= offset_baseline.shape[1])
                    ix_nonzero[less_than_0,1] = 0
                    ix_nonzero[greater_than_shape,1] = offset_baseline.shape[1]-1
                    
                    ix_nonzero = np.transpose(ix_nonzero)
                    ix_nonzero = tuple([ix_nonzero[i, :] for i in range(len(ix_nonzero))])

                    new_liver_seg[ix_nonzero] = 1
    return new_liver_seg

def postprocess_liver_predictions(nifti_dir, patient, liver_file_names):
    postprocessed_file_names = []
    liver_save_names = []

    liver_file_name = liver_file_names[0]

    #load the data
    liver_img = nib.load(os.path.join(nifti_dir, patient, liver_file_name))
    liver_data = liver_img.get_data()

    #postprocess the data
    large_comp_liver = extract_largest_connected_comp2(liver_data, voxel_vol = np.prod(liver_img.header.get_zooms()), pick_right_of_top_two = True)
    
    #no dilation
    no_dilation_img = nib.Nifti1Image(large_comp_liver, liver_img.affine)
    liver_save_names.append("NoDilation")
    postprocessed_file_names.append(liver_file_name[:liver_file_name.find('.nii')] + "_" + liver_save_names[-1] + ".nii.gz")
    nib.save(no_dilation_img, os.path.join(nifti_dir, patient, postprocessed_file_names[-1]))

    #raw box
    #raw_box_data = box_liver_2D(large_comp_liver)
    #raw_box_img = nib.Nifti1Image(raw_box_data, liver_img.affine)
    #liver_save_names.append("RawBox")
    #postprocessed_file_names.append(liver_file_name[:liver_file_name.find('.nii')] + "_" + liver_save_names[-1] + ".nii.gz")
    #nib.save(raw_box_img, os.path.join(nifti_dir, patient, postprocessed_file_names[-1]))

    #box with 1111 offset
    #offset_1111 = expand_border_2(large_comp_liver, [[-1,1], [1,-1]])
    #box1111_data = box_liver_2D(offset_1111)
    #box1111_img = nib.Nifti1Image(box1111_data, liver_img.affine)
    #liver_save_names.append("Box1111")
    #postprocessed_file_names.append(liver_file_name[:liver_file_name.find('.nii')] + "_" + liver_save_names[-1] + ".nii.gz")
    #nib.save(box1111_img, os.path.join(nifti_dir, patient, postprocessed_file_names[-1]))

    #mask with 2222 offset
    offset_2222 = expand_border_2(large_comp_liver, [[-2,2], [2,-2]])
    offset_2222_img = nib.Nifti1Image(offset_2222, liver_img.affine)
    liver_save_names.append("2222")
    postprocessed_file_names.append(liver_file_name[:liver_file_name.find('.nii')] + "_" + liver_save_names[-1] + ".nii.gz")
    nib.save(offset_2222_img, os.path.join(nifti_dir, patient, postprocessed_file_names[-1]))

    #mask with 4444 offset
    #offset_4444 = expand_border_2(large_comp_liver, [[-4,4], [4,-4]])
    #offset_4444_img = nib.Nifti1Image(offset_4444, liver_img.affine)
    #liver_save_names.append("4444")
    #postprocessed_file_names.append(liver_file_name[:liver_file_name.find('.nii')] + "_" + liver_save_names[-1] + ".nii.gz")
    #nib.save(offset_4444_img, os.path.join(nifti_dir, patient, postprocessed_file_names[-1]))

    return postprocessed_file_names, liver_save_names

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    ix = int(np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val))
    if ix == -1:
        ix = 0
    return ix

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    ix = np.where(mask.any(axis=axis), val, invalid_val) + 1
    if ix == 0:
        ix = len(arr)
    return ix

def crop_to_liver(nifti_dir, patient, vols_to_process, liver_mask, append_tag = "cropped"):
    liver_img = nib.load(os.path.join(nifti_dir, patient, liver_mask))
    liver_data = liver_img.get_data()
    #volume input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)

    #find where the liver starts and ends
    collapsed_but_axial = np.sum(np.sum(liver_data, axis = 0), axis = 0).flatten(order = 'C')
    collapsed_but_sagital = np.sum(np.sum(liver_data, axis = 2), axis = 1).flatten(order = 'C')
    collapsed_but_coronal = np.sum(np.sum(liver_data, axis = 2), axis = 0).flatten(order = 'C')

    first_axial = first_nonzero(collapsed_but_axial, 0, invalid_val=-1)
    last_axial = last_nonzero(collapsed_but_axial, 0, invalid_val=-1)
    first_sagital = first_nonzero(collapsed_but_sagital, 0, invalid_val = -1)
    last_sagital = last_nonzero(collapsed_but_sagital, 0, invalid_val=-1)
    first_coronal = first_nonzero(collapsed_but_coronal, 0, invalid_val = -1)
    last_coronal = last_nonzero(collapsed_but_coronal, 0, invalid_val=-1)

    #crop and save the images
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        #load the image
        img = nib.load(input_filepath)
 
        img_cropped = img.slicer[first_sagital:last_sagital+1, first_coronal:last_coronal+1, first_axial:last_axial+1]

        nib.save(img_cropped, output_filepath)

    #crop and save the liver
    liver_img_cropped = liver_img.slicer[first_sagital:last_sagital+1, first_coronal:last_coronal+1, first_axial:last_axial+1]
    liver_img_cropped_name = liver_mask[:liver_mask.find('.nii')] + "_" + append_tag + ".nii.gz"
    nib.save(liver_img_cropped, os.path.join(nifti_dir, patient, liver_img_cropped_name))

    return output_filenames, liver_img_cropped_name
    
def liver_extraction(nifti_dir, patient, vols_to_process, liver_mask, append_tag = "nls"):
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)

    liver_img = nib.load(os.path.join(nifti_dir, patient, liver_mask))
    liver_data = liver_img.get_data()
    non_liver_ix = liver_data == 0

    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        img = nib.load(input_filepath)
        img_data = img.get_data()

        img_data[non_liver_ix] = 0
        new_img = nib.Nifti1Image(img_data, img.affine)

        nib.save(new_img, output_filepath)
    
    return output_filenames
