import numpy as np
import torch
from tqdm import tqdm
from pytorch_connectomics.connectomics.model.arch.unet import UNet3D
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ToTensord,
)
from monai.data import (
    DataLoader,
    CacheDataset,
)

inference_path = "/mnt/ceph/users/pgrover/inst_inference_dataset/"
saved_weights_path = "/mnt/ceph/users/pgrover/inst_inference_dataset/3dunet_bcd_saved_weights.pth"
patch_size = (16, 128, 128)

test_files = list(np.load(inference_path + "test_files.npy", allow_pickle=True))
test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
)

test_ds = CacheDataset(
    data=test_files, 
    transform=test_transforms,
    cache_num=1, cache_rate=0.0, num_workers=1
)

test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
)

test_iterator = tqdm(
                test_loader, desc="Testing (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )

test_set_labels = ['F55_185']


model = UNet3D()
model.load_state_dict(torch.load(saved_weights_path, map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
    for step, batch in enumerate(test_iterator):
        current_label = test_set_labels[step]
        print(" ")
        print("Processing : ", current_label)
        val_inputs = batch["image"]
        val_outputs = sliding_window_inference(val_inputs, patch_size, 4, model)
        np.save(inference_path + "unet_outputs_" + current_label + ".npy", val_outputs)