import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

inference_path = "/mnt/ceph/users/pgrover/inst_inference_dataset/"

test_set_labels = ['F8_72', 'F24_1', 'F24_2', 'F24_6', 'F24_10', 'F25_2', 'F25_8', 'F26_8',
                  'F27_7', 'F27_9', 'F27_10', 'F29_3', 'F29_4', 'F30_4', 'F30_8', 'F30_9',
                  'F33_67', 'F34_73', 'F39_117', 'F40_136', 'F41_56', 'F42_63', 'F44_87',
                  'F44_89', 'F49_148', 'F55_185', 'M6_12', 'M6_21', 'M7_0', 'M7_4']

if not os.path.isdir(inference_path):
      os.makedirs(inference_path)
      print("Created Inference Folder", inference_path)
else:
      print("Folder already exists")

test_files = []

for series in test_set_labels:
      input_volume = np.load("/mnt/ceph/users/lbrown/Labels3DMouse/GTSets/2022_Full/" + series + "/" + series + "/images/" + series + "_image_0001.npy")
      orig_shape = input_volume.shape
      input_volume = (input_volume - np.mean(input_volume))/(np.std(input_volume))
      scaler = MinMaxScaler()
      scaler.fit(input_volume.flatten().reshape(-1, 1))
      input_volume = scaler.transform(input_volume.flatten().reshape(-1, 1)).reshape(orig_shape)
      image_path = inference_path + "image_" + series + ".npy"
      test_files.append({'image' : image_path})
      np.save(image_path, input_volume)
      print("Completed operation for ", series)

test_files = np.array(test_files)
np.save(inference_path + "test_files.npy", test_files)           