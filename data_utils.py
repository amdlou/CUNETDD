import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
import numpy as np
import glob



def normalize_data(data):
    # Check if there's a batch dimension, and add one if there isn't
    was_singleton = False
    if len(data.shape) == 3:  # Shape is [1, H, W]
        data = data.unsqueeze(0)  # Add a batch dimension [1, 1, H, W]
        was_singleton = True

    # Find the maximum value for each image in the batch
    max_vals = data.view(data.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    
    # Avoid division by zero for images with all pixels equal to zero
    max_vals[max_vals == 0] = 1
    
    # Normalize each image individually
    normalized_data = data / max_vals

    # Remove the batch dimension if it was added
    if was_singleton:
        normalized_data = normalized_data.squeeze(0)
    
    return normalized_data

class ParseDataset(Dataset):
    def __init__(self, filepath='', image_size=256, out_channel=1):
        self.datasets = []
        self.ds = None
        self.filepath = Path(filepath)

        if self.filepath.is_dir():
            self.file_lists = list(self.filepath.glob('**/*training.h5'))
            self.from_dir = True
        else:
            self.file_lists = [self.filepath]
            self.from_dir = False

        self.ext = self.file_lists[0].suffix.lstrip('.')
        assert self.ext in ['h5', 'tfrecords'], "Currently only supports hdf5 or tfrecords as dataset"

        assert isinstance(image_size, (int, list)), 'image_size must be integer (when height=width) or list (height, width)'
        if isinstance(image_size, int):
            self.height = self.width = image_size
        else:
            self.height, self.width = image_size

        self.out_channel = out_channel

    def read(self, batch_size=256, shuffle=True, mode='default', task='system', one_hot=False):
        self.task = task
        self.one_hot = one_hot
        if self.ext == 'h5':
            self.ds = self.prepare_dataset_from_hdf5(batch_size, shuffle, mode)
        elif self.ext == 'tfrecords':
            self.ds = self.prepare_dataset_from_tfrecords(batch_size, shuffle, mode)
        return DataLoader(self.ds, batch_size=batch_size, shuffle=shuffle)

    def prepare_dataset_from_tfrecords(self, batch_size, shuffle, mode):
        # Implement loading TFRecords data into a PyTorch dataset
        pass


    def prepare_dataset_from_hdf5(self, batch_size, shuffle, mode):
        for file in self.file_lists:
            
            try:
                with h5py.File(file, 'r') as data:
                    # Assuming data['dataMeas'] and data['dataPots'] are numpy arrays of shape (256, 256, 25)
                    cbed = [torch.from_numpy(np.reshape(data['dataMeas'][..., i], (256, 256, 1))).unsqueeze(0).permute(0, 3, 1, 2) for i in range(25)]
                    probe = torch.from_numpy(data['dataProbe'][...]).unsqueeze(0).unsqueeze(0)
                    pot = [torch.from_numpy(np.reshape(data['dataPots'][..., i], (256, 256, 1))).unsqueeze(0).permute(0, 3, 1,2) for i in range(25)]
                    
                    for i in range(25):
                        ds = torch.utils.data.TensorDataset(cbed[i], probe, pot[i])
                        self.datasets.append(ds)
                    
                    self.ds = torch.utils.data.ConcatDataset(self.datasets)
                    print(f"File: {file}, Num items: {len(self.ds)}")
            except OSError as e:
            # If an error occurs, skip the file and print/log the error
                print("Skipped corrupted or incompatible file Error: {e}")
        if shuffle:
            indices = torch.randperm(len(self.ds))
            self.ds = torch.utils.data.Subset(self.ds, indices)

        return self.ds
    def __len__(self):
        total_len = sum([len(ds) for ds in self.datasets])
        return total_len
    
    def __getitem__(self, idx):
        for ds in self.datasets:
            if idx < len(ds):
                cbed, probe, pot = ds[idx]
                #pot = normalize_data(pot)
                #print(pot.size)
                return (cbed, probe, pot)
            idx -= len(ds)



