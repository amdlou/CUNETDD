"""prepare_dataset_from_hdf5 or TFRecords that loads data from HDF5 files.
 The data includes datameas with dimensions 256x256x25, 
 dataprobe with dimensions 256x256,
 and datapots with dimensions 256x256x25."""

from pathlib import Path
from typing import Union, List, Tuple
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
import h5py


def normalize_data(data):
    """
    Normalize the input data by dividing each image by its maximum value.

    Args:
        data (torch.Tensor): Input data tensor of shape [B, C, H, W], where B is the batch size,
                             C is the number of channels, H is the height, and W is the width.

    Returns:
        torch.Tensor: Normalized data tensor of the same shape as the input data.

    """
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
    """
    A PyTorch dataset class for parsing and preparing datasets from HDF5 or TFRecords files.

    Args:
        filepath (str): The path to the dataset file or directory.
        image_size (int or list): The size of the images in the dataset. 
        Can be an integer (when height=width) or a list (height, width).
        out_channel (int): The number of output channels.

    Attributes:
        datasets (list): A list to store the individual datasets.
        ds (torch.utils.data.ConcatDataset): The concatenated dataset.
        filepath (Path): The path to the dataset file or directory.
        file_lists (list): A list of file paths.
        from_dir (bool): Indicates whether the dataset is loaded
                                from a directory or a single file.
        ext (str): The file extension of the dataset file.
        height (int): The height of the images in the dataset.
        width (int): The width of the images in the dataset.
        out_channel (int): The number of output channels.
        task (str): The task associated with the dataset.
        one_hot (bool): Indicates whether the labels are one-hot encoded.

    """

    def __init__(self, filepath: str = '',
                 image_size: Union[int, List[int]] = 256,
                 out_channel: int = 1):

        self.ds: Optional[ConcatDataset] = None
        self.datasets = []
        self.filepath = Path(filepath)

        if self.filepath.is_dir():
            self.file_lists = list(self.filepath.glob('**/*training.h5'))
            self.from_dir = True
        else:
            self.file_lists = [self.filepath]
            self.from_dir = False

        self.ext = self.file_lists[0].suffix.lstrip('.')
        assert self.ext in ['h5', 'tfrecords'], "Currently only supports hdf5 or tfrecords as dataset"

        assert isinstance(image_size, (int, list)),'image_size must be integer (when height=width) or list (height, width)'
        if isinstance(image_size, int):
            self.height = self.width = image_size
        else:
            self.height, self.width = image_size

        self.out_channel = out_channel

    def read(self, batch_size: int = 256, shuffle: bool = True,
             mode: str = 'default') -> DataLoader:
        """
        Reads and prepares the dataset for training or evaluation.

        Args:
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Indicates whether to shuffle the dataset.
            mode (str): The mode for preparing the dataset.
            task (str): The task associated with the dataset.
            one_hot (bool): Indicates whether the labels are one-hot encoded.

        Returns:
            torch.utils.data.DataLoader:
            The DataLoader object for the prepared dataset.

        """
        self.ds = self.prepare_dataset_from_hdf5(mode)
        
        return DataLoader(self.ds, batch_size=batch_size, shuffle=shuffle)

    def prepare_dataset_from_hdf5(self, mode: str) -> ConcatDataset:
        """
        Loads the dataset from HDF5 files.

        Args:
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Indicates whether to shuffle the dataset.
            mode (str): The mode for preparing the dataset.

        Returns:
            torch.utils.data.ConcatDataset: The concatenated dataset.
        """
        
        for file in self.file_lists:
            try:
                with h5py.File(file, 'r') as data:
                    # Assuming data['dataMeas'] and data['dataPots']
                    # are numpy arrays of shape (256, 256, 25)
                    data_meas = np.array(data['dataMeas'])
                    data_probe = np.array(data['dataProbe'])
                    data_pots = np.array(data['dataPots'])

                    cbed = [torch.from_numpy(data_meas[..., i].reshape((256, 256, 1)))
                            .unsqueeze(0).permute(0, 3, 1, 2)
                            for i in range(25)]
                    probe = torch.from_numpy(data_probe[...]).unsqueeze(0).unsqueeze(0)
                    pot = [torch.from_numpy(data_pots[..., i].reshape((256, 256, 1)))
                           .unsqueeze(0).permute(0, 3, 1, 2) for i in range(25)]

                    if mode == 'default':
                        ds = TensorDataset(cbed[0], probe, pot[0])
                        self.datasets.append(ds)
                        for i in range(1, 25):
                            ds = TensorDataset(cbed[i], probe, pot[i])
                            self.datasets.append(ds)

                    elif mode == 'norm':
                        max_val = torch.max(probe)
                        pot = [p / max_val for p in pot]
                        for i in range(25):
                            ds = TensorDataset(cbed[i], probe, pot[i])
                            self.datasets.append(ds)

                    self.ds = ConcatDataset(self.datasets)
                    print(f"File: {file}, Num items: {len(self.ds)}")
            except OSError as e:
                # If an error occurs, skip the file and print/log the error
                print(f"Skipped corrupted or incompatible file. Error: {e}")

        return self.ds

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.

        """
        total_len = sum([len(ds) for ds in self.datasets])
        return total_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor]:
        """
        Returns the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: A tuple containing the input features and labels.

        """
        for ds in self.datasets:
            if idx < len(ds):
                cbed, probe, pot = ds[idx]
                #pot = normalize_data(pot)
                return (cbed, probe, pot)
            idx -= len(ds)
        
        # Return a default value if the index is out of range
        return (torch.Tensor(), torch.Tensor(), torch.Tensor())




