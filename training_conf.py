import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import builtins
import os
from pytorch_connectomics.connectomics.model.arch.unet import UNet3D
from creating_dataset import Dataset
import random

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)      
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)         
        return 1 - dice

sys.stdout = open("/mnt/home/pgrover/3dunet/logs/cont_logging_training.txt", "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

print("Processing Inputs")
partition = {'train' : [], 'validation' : []}
for i in range(1, 3196):
    prob = random.random()
    if (prob > 0.85):
        partition['validation'].append(i)
    else:
        partition['train'].append(i)

# Generators
training_set = Dataset(partition['train'])
training_generator = torch.utils.data.DataLoader(training_set, batch_size = 6, num_workers = 16, shuffle = True)

validation_set = Dataset(partition['validation'])
validation_generator = torch.utils.data.DataLoader(validation_set)

model = UNet3D()
# model = model.cuda()
print("Loaded UNet3D")

sig = torch.nn.Sigmoid()
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
dice_loss = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# true_labels = []
# pred_labels = []

for e in range(1, 500+1):
    print("Training Network")

    train_bce_mask_loss = 0.0
    train_bce_contour_loss = 0.0
    train_dice_mask_loss = 0.0
    train_dice_contour_loss = 0.0
    train_mse_dist_loss = 0.0
    train_epoch_loss = 0.0

    # val_bce_mask_loss = 0.0
    # val_bce_contour_loss = 0.0
    # val_dice_mask_loss = 0.0
    # val_dice_contour_loss = 0.0
    # val_mse_dist_loss = 0.0
    # val_epoch_loss = 0
    
    model.train()
    batch_num = 0

    for X_train_batch, y_train_batch in training_generator:
            batch_num += 1
            X_train_batch, y_train_batch = X_train_batch.to(device, dtype = torch.float), y_train_batch.to(device, dtype = torch.float)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            dice_loss_mask = dice_loss(y_train_pred[:, 0, :, :, :], y_train_batch[:, 0, :, :, :])
            dice_loss_contour = dice_loss(y_train_pred[:, 1, :, :, :], y_train_batch[:, 1, :, :, :]) 
            mse_loss_dist = 1 * mse_loss(y_train_pred[:, 2, :, :, :], y_train_batch[:, 2, :, :, :]) + 0 * l1_loss(y_train_pred[:, 2, :, :, :], y_train_batch[:, 2, :, :, :])
            train_loss = dice_loss_mask + dice_loss_contour + 1 * mse_loss_dist
            print("Epoch : " + str(e) + "| Batch " + str(batch_num) + "| Mask : " + str(round(dice_loss_mask.item(), 3)) +  "| Contour : " + str(round(dice_loss_contour.item(), 3)) + "| Distance Map : " + str(round(mse_loss_dist.item(), 3)) + "| Complete Loss : " + str(round(train_loss.item(), 3)))            
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_dice_mask_loss += dice_loss_mask.item()
            train_dice_contour_loss += dice_loss_contour.item()
            train_mse_dist_loss += mse_loss_dist.item()
            train_epoch_loss += train_loss

            if (batch_num%100 == 0):
                torch.save(model.state_dict(), '/mnt/home/pgrover/3dunet/3dunet_bcd_saved.pth')

    torch.save(model.state_dict(), '/mnt/home/pgrover/3dunet/3dunet_bcd_saved.pth')     
    print("Training Epoch : " + str(e))
    print("Dice Mask Loss : " + str(round(train_dice_mask_loss/batch_num, 3)) + "| Dice Contour Loss : " + str(round(train_dice_contour_loss/batch_num, 3)) + "| MSE Dist Loss : " + str(round(train_mse_dist_loss/batch_num, 3)))