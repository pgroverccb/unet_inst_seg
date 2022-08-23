from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from scipy.signal import convolve2d
from skimage.morphology import erosion, dilation
from typing import Optional, Tuple
import torch
import scipy
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes
from skimage.measure import label as label_cc  # avoid namespace conflict


def seg_to_instance_bd(seg: np.ndarray, tsz_h: int = 1, do_bg: bool = True, do_convolve: bool = True) -> np.ndarray:
    if do_bg == False:
        do_convolve = False
    sz = seg.shape
    bd = np.zeros(sz, np.uint8)
    tsz = tsz_h*2+1

    if do_convolve:
        sobel = [1, 0, -1]
        sobel_x = np.array(sobel).reshape(3, 1)
        sobel_y = np.array(sobel).reshape(1, 3)
        for z in range(sz[0]):
            slide = seg[z]
            edge_x = convolve2d(slide, sobel_x, 'same', boundary='symm')
            edge_y = convolve2d(slide, sobel_y, 'same', boundary='symm')
            edge = np.maximum(np.abs(edge_x), np.abs(edge_y))
            contour = (edge != 0).astype(np.uint8)
            bd[z] = dilation(contour, np.ones((tsz, tsz), dtype=np.uint8))
        return bd

    mm = seg.max()
    for z in range(sz[0]):
        patch = im2col(
            np.pad(seg[z], ((tsz_h, tsz_h), (tsz_h, tsz_h)), 'reflect'), [tsz, tsz])
        p0 = patch.max(axis=1)
        if do_bg:  # at least one non-zero seg
            p1 = patch.min(axis=1)
            bd[z] = ((p0 > 0)*(p0 != p1)).reshape(sz[1:])
        else:  # between two non-zero seg
            patch[patch == 0] = mm+1
            p1 = patch.min(axis=1)
            bd[z] = ((p0 != 0)*(p1 != 0)*(p0 != p1)).reshape(sz[1:])
    return bd

def _edt_binary_mask(mask, resolution, alpha):
    if (mask == 1).all():  # tanh(5) = 0.99991
        return np.ones_like(mask).astype(float) * 5

    return distance_transform_edt(mask, resolution) / alpha

def edt_semantic(
        label: np.ndarray,
        mode: str = '2d',
        alpha_fore: float = 8.0,
        alpha_back: float = 50.0):
    """Euclidean distance transform (DT or EDT) for binary semantic mask.
    """
    assert mode in ['2d', '3d']
    do_2d = (label.ndim == 2)

    resolution = (6.0, 1.0, 1.0)  # anisotropic data
    if mode == '2d' or do_2d:
        resolution = (1.0, 1.0)

    fore = (label != 0).astype(np.uint8)
    back = (label == 0).astype(np.uint8)

    if mode == '3d' or do_2d:
        fore_edt = _edt_binary_mask(fore, resolution, alpha_fore)
        back_edt = _edt_binary_mask(back, resolution, alpha_back)
    else:
        fore_edt = [_edt_binary_mask(fore[i], resolution, alpha_fore)
                    for i in range(label.shape[0])]
        back_edt = [_edt_binary_mask(back[i], resolution, alpha_back)
                    for i in range(label.shape[0])]
        fore_edt, back_edt = np.stack(fore_edt, 0), np.stack(back_edt, 0)
    distance = fore_edt - back_edt
    return np.tanh(distance)

dir_path = '/mnt/ceph/users/lbrown/Labels3DMouse/TrainingSets/2022_64x256x256/'
folders = os.listdir(dir_path)
folders = folders[1:]

image_paths = []
mask_paths = []

for folder_name in folders:
      print("Folder Name : " + str(folder_name))
      if (folder_name[0] == 'F'):
          images_folder_path = dir_path + '/' + folder_name + '/' + folder_name + '/images/' 
          files_images = os.listdir(images_folder_path)
          masks_folder_path = dir_path + '/' + folder_name + '/' + folder_name + '/masks/' 
          files_masks = os.listdir(masks_folder_path)
          for frame in range(1, 10):
              try:
                  curr_image_path = dir_path + '/' + folder_name + '/' + folder_name + '/images/' + folder_name + "_image_000" + str(frame) + ".npy" 
                  curr_mask_path = dir_path + '/' + folder_name + '/' + folder_name + '/masks/' + folder_name + "_masks_000" + str(frame) + ".npy" 
                  # curr_image = np.load(curr_image_path)
                  # curr_mask = np.load(curr_mask_path)
                  # curr_image = (curr_image - np.mean(curr_image))/(np.std(curr_image))
                  # scaler = MinMaxScaler()
                  # scaler.fit(curr_image.flatten().reshape(-1, 1))
                  # curr_image = scaler.transform(curr_image.flatten().reshape(-1, 1)).reshape((64, 256, 256))
                  # bg, fg = np.percentile(curr_image, (1, 99))
                  # if (fg/bg > 1.5):
                  image_paths.append(curr_image_path)
                  mask_paths.append(curr_mask_path)
              except:
                  continue


dataset_path = '/mnt/ceph/users/pgrover/bcd_dataset/'
true_labels_path = '/mnt/ceph/users/pgrover/act_dataset/'
curr_count = 1
for i in range(0, len(image_paths)):
    curr_image_path = image_paths[i]
    curr_mask_path = mask_paths[i]
    curr_image = np.load(curr_image_path)
    curr_mask = np.load(curr_mask_path)
    curr_image = (curr_image - np.mean(curr_image))/(np.std(curr_image))
    scaler = MinMaxScaler()
    scaler.fit(curr_image.flatten().reshape(-1, 1))
    curr_image = scaler.transform(curr_image.flatten().reshape(-1, 1)).reshape((64, 256, 256))
    np.save(true_labels_path + str(curr_count) + ".npy", curr_mask)
    curr_mask [curr_mask > 1] = 1
    curr_contour = seg_to_instance_bd(curr_mask)
    curr_dist = []
    for i in range(0, 64):
        curr_dist_slice = edt_semantic(curr_mask[i], mode = '2d', alpha_fore = 8.0, alpha_back = 50.0)
        curr_dist.append(curr_dist_slice)
    curr_dist = np.array(curr_dist)
    complete_sample = np.array([curr_image, curr_mask, curr_contour, curr_dist])
    np.savez_compressed(dataset_path + '/' + str(curr_count), a = complete_sample)
    print(curr_count)
    curr_count += 1