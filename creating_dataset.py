import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import MinMaxScaler
import random

class Dataset(torch.utils.data.Dataset):
  def __init__(self, list_IDs):
        self.list_IDs = list_IDs

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        ID = self.list_IDs[index]
        data = np.load("/mnt/ceph/users/pgrover/bcd_dataset/" + str(ID + 1) + ".npz")['a']
        curr_image = data[0]
        curr_mask = data[1]
        curr_mask_contour = data[2]
        curr_mask_dist = data[3]
        X = curr_image
        X = X.reshape((1, 64, 256, 256))
        y = np.array([curr_mask, curr_mask_contour, curr_mask_dist])
        y = y.reshape((3, 64, 256, 256))
        z_offset = random.randint(0, 48)
        x_offset = random.randint(0, 127)
        y_offset = random.randint(0, 127)
        X = X[:, z_offset : z_offset + 16, y_offset : y_offset + 128, x_offset : x_offset + 128]
        y = y[:, z_offset : z_offset + 16, y_offset : y_offset + 128, x_offset : x_offset + 128]
        return X, y