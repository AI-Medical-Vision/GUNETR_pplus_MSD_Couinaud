import numpy as np
from multiprocessing import Pool
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import torchio as tio
import torch


def unpack_dataset(folder, extension, task_type, threads=8, key="data"):
    """
    unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved unter key)
    :param folder:
    :param threads:
    :param key:
    :return:
    """

    p = Pool(threads)
    nii_files = subfiles(folder, True, None, extension, True) # npz->mha
    print(len(nii_files)) # 5
    if task_type == 'whole':
        p.map(convert_to_npy, zip(nii_files, [key] * len(nii_files)))
    else:
        print("Couinaud segmentation")
        p.map(convert_couinaud_to_npy, zip(nii_files, [key] * len(nii_files)))
    p.close()
    p.join()

def convert_to_npy(args):
    # LiTS dataset rotation
    #tmp_lst=['volume-15.nii', 'volume-18.nii', 'volume-28.nii', 'volume-3.nii', 'volume-33.nii', 'volume-37.nii', 'volume-42.nii', 'volume-47.nii', 'volume-5.nii', 'volume-54.nii', 'volume-70.nii', 'volume-73.nii', 'volume-80.nii']
    
    if not isinstance(args, tuple):
        key = "data"
        nii_files = args
    else:
        nii_files, key = args
        
    if (nii_files[-3:] == 'nii'):
        num = 3
        extension = 'nii'
    elif (nii_files[-3:] == 'mhd'):
        num = 3
        extension = 'mhd'
    elif nii_files[-6:] == 'nii.gz':
        num = 6
        extension = 'nii.gz'

    # resize
    #resize = tio.Resize((256,256,-1))
    try:
        #print(nii_files)
        gt_path = nii_files.replace('unetr_pp_Data_plans_v2.1_stage1', 'seg_gt')
        name_OR = gt_path.split('/')[-1]
        #print(name_OR)
        gt_path = gt_path[:-len(name_OR)]
        gt_path = gt_path + name_OR
        #print("gt:", gt_path)

        ####################################
        a = tio.ScalarImage(nii_files)
        resample = tio.Resample()  # default is 1 mm isotropic
        resampled = resample(a)
        a = resampled.data

        a = np.array(a)
        
        a = (a - np.min(a)) / (np.max(a) - np.min(a)) # normalize - preprocessing
        a = np.transpose(a[0], (2,1,0)) # TODO: 9/24 axial
        #####################################

        #####################################
        gt = tio.ScalarImage(gt_path)
        gt.data = np.where(gt.data >= 1, 1, 0) # Not LiTS, it remove (tumor containing)
        resample = tio.Resample()  # default is 1 mm isotropic
        resampled = resample(gt)
        gt = resampled.data

        gt = np.array(gt)
        #gt = resize(gt) # input: 4D
        if len(np.unique(gt)) != 2: # error check
            print(name_OR)
            raise Exception("No label")
        
        gt = np.transpose(gt[0], (2,1,0)) # TODO: 9/24 axial
        #####################################
        
    except Exception as e:
        print("error file:",nii_files)
        print(e)    
    
    # npy format 생성
    a = a[np.newaxis, ...]
    gt = gt[np.newaxis, ...]
    print(a.shape)
    print(gt.shape)
    if a.shape != gt.shape:
        print(nii_files)
        
    print("saving...")
    final = np.concatenate([a, gt], axis=0) # (2, x, y, x)
    np.save(nii_files[:-num] + "npy", final)

def convert_couinaud_to_npy(args):
    # LiTS dataset rotation
    #tmp_lst=['volume-15.nii', 'volume-18.nii', 'volume-28.nii', 'volume-3.nii', 'volume-33.nii', 'volume-37.nii', 'volume-42.nii', 'volume-47.nii', 'volume-5.nii', 'volume-54.nii', 'volume-70.nii', 'volume-73.nii', 'volume-80.nii']
    
    if not isinstance(args, tuple):
        key = "data"
        nii_files = args
    else:
        nii_files, key = args
        
    if (nii_files[-3:] == 'nii'):
        num = 3
        extension = 'nii'
    elif (nii_files[-3:] == 'mhd'):
        num = 3
        extension = 'mhd'
    elif nii_files[-6:] == 'nii.gz':
        num = 6
        extension = 'nii.gz'

    # resize
    #resize = tio.Resize((256,256,-1))
    mask_path = './DATASET_Synapse/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse/Task002_Synapse/masking/test/'
    try:
        #print(nii_files)
        gt_path = nii_files.replace('unetr_pp_Data_plans_v2.1_stage1', 'seg_gt')
        name_OR = gt_path.split('/')[-1]
        #print(name_OR)
        gt_path = gt_path[:-len(name_OR)]
        gt_path = gt_path + name_OR
        #print("gt:", gt_path)

        ####################################
        a = tio.ScalarImage(nii_files)
        resample = tio.Resample()  # default is 1 mm isotropic
        resampled = resample(a)
        a = resampled.data

        a = np.array(a)
        input_shape = a[0].shape
        
        a = (a - np.min(a)) / (np.max(a) - np.min(a)) # normalize - preprocessing
        a = np.transpose(a[0], (2,1,0)) # TODO: 9/24 axial

        ## Masking
        mask_path = mask_path + name_OR.replace('hepaticvessel', 'masking')
        masking = tio.ScalarImage(mask_path)
        masking = np.array(masking)
        masking = np.transpose(masking[0], (2,1,0))
        a = np.where(masking==0, 0, a)
        #####################################

        #####################################
        ## GT: couinaud segmentation
        #print(gt_path)
        gt = tio.ScalarImage(gt_path)
        couinaud_gt = np.zeros(input_shape)

        gt_copy_data = gt.data.clone()
        unique_label = np.unique(gt.data)
        resample = tio.Resample()

        if len(unique_label) != 9:
            raise Exception('Bug data?')
        else:
            for k in unique_label:
                #print(k)
                if k != 0:
                    #a = tio.ScalarImage(path+'hepaticvessel_020.nii.gz')
                    gt.data = np.where(gt.data>=k, 1, 0)
                    sub_gt = resample(gt)
                    sub_gt = sub_gt.data[0]
                    sub_gt = np.array(sub_gt)
                    #sub_gt = np.where(sub_gt==1, k, 0)
                    couinaud_gt += sub_gt
                    gt.data = gt_copy_data

        gt = np.transpose(couinaud_gt, (2,1,0)) # TODO: 9/24 axial
        #####################################
        
    except Exception as e:
        print("error file:",nii_files)
        print(e)    
    
    # npy format 생성
    a = a[np.newaxis, ...]
    gt = gt[np.newaxis, ...]
    print(a.shape)
    print(gt.shape)
    if a.shape != gt.shape:
        print(nii_files)
        
    print("saving...")
    final = np.concatenate([a, gt], axis=0) # (2, x, y, x)
    np.save(nii_files[:-num] + "npy", final)


if __name__ == '__main__':

    while True:
        
        task_type = input("whole or couinaud:")

        #data = input("Dataset names(3Dircadb, LiTS, Sliver):")
        data = 'MSD'
        if data in ['MSD']:
            if data == 'LiTS':
                extension = '.nii'
            elif data == 'MSD':
                extension = '.nii.gz'
            else:
                extension = '.mhd'
            print("Extension:", extension)
            
            folder = './DATASET_Synapse/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse/Task002_Synapse/unetr_pp_Data_plans_v2.1_stage1/test/'
            unpack_dataset(folder, extension, task_type)
            break

        else:
            print("No dataset")