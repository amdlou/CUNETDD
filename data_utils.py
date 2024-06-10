"""prepare_dataset_from_hdf5 or TFRecords that loads data from HDF5 files.
 The data includes datameas(cbed) with dimensions 256x256x25,
                     dataprobe(probe) with dimensions 256x256,
                     and datapots(Structure Factors,real part) with
                     dimensions 256x256x25."""

from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from augment import Image_Augmentation
import tensorflow as tf

def normalize_data(data):
    """
    Normalize the input data by dividing each image by its maximum value.

    Args:
        data (torch.Tensor): Input data tensor of shape [B, C, H, W],
                             where B is the batch size,
                             C is the number of channels, H is the height,
                             and W is the width.

    Returns:
        torch.Tensor: Normalized data tensor of the same shape
                                             as the input data.

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
    A custom PyTorch dataset class for parsing and accessing data
      from hdf5 or tfrecords files.

    Args:
        filepath (str): The path to the dataset file or directory.
        image_size (int or List[int]): The size of the images in the dataset.
        If an integer is provided, the images will be resized
        to a square of that size. If a list of two integers is provided,
        the images will be resized to the specified height and width.
        out_channel (int): The number of output channels for the images.
        batch_size (int): The batch size for the dataset.

    Attributes:
        filepath (Path): The path to the dataset file or directory.
        batch_size (int): The batch size for the dataset.
        file_lists (List[Path]): A list of paths to the dataset files.
        from_dir (bool): Indicates whether the dataset is loaded from
        a directory or a single file.
        ext (str): The file extension of the dataset files.
        height (int): The height of the images in the dataset.
        width (int): The width of the images in the dataset.
        out_channel (int): The number of output channels for the images.
        lengths (List[int]): The lengths of the individual dataset files.
        cumulative_lengths (ndarray):
        The cumulative lengths of the dataset files.

    Methods:
        _replace_nan(tensor): Replaces NaN values in a tensor with zeros.
        __getitem__(idx): Retrieves an item from the dataset
                          based on the given index.
        __len__(): Returns the length of the dataset.

    """

    def __init__(self, filepath: str = '', image_size: Union[int,
                 List[int]] = 256, out_channel: int = 1):

        assert isinstance(image_size, (int, list)), 'image_size must be integer (when height=width) or list (height, width)'
        self.filepath: Path = Path(filepath)
        self.file_lists: List[Path] = list(self.filepath.glob(
            '**/*training.h5')) if self.filepath.is_dir() else [self.filepath]
        self.file_lists = self._filter_valid_files(self.file_lists)
        self.from_dir: bool = self.filepath.is_dir()
        self.ext: str = self.file_lists[0].suffix.lstrip('.')
        assert self.ext in ['h5', 'tfrecords'], \
            "Currently only supports hdf5 or tfrecords as dataset"
        if isinstance(image_size, int):
            self.height = self.width = image_size
        else:
            self.height, self.width = image_size
        self.out_channel = out_channel
        self.lengths = [25 for _ in self.file_lists]
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.augmenter = Image_Augmentation()


    def _filter_valid_files(self, file_lists: List[Path]) -> List[Path]:
        valid_files = []
        for file in file_lists:
            try:
                with h5py.File(file, 'r') as f:
                    pass
                valid_files.append(file)
            except OSError:
                print(f"Skipped corrupted or incompatible file: {file}")
        return valid_files

    def _replace_nan(self, tensor: torch.Tensor) -> torch.Tensor:
        """Replaces NaN values in a tensor with zeros."""
        return np.nan_to_num(tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        """Retrieves an item from the dataset based on the given index."""
        file_idx = np.searchsorted(self.cumulative_lengths, idx + 1)
        if file_idx > 0:
            idx -= self.cumulative_lengths[file_idx - 1]
        with h5py.File(self.file_lists[file_idx], 'r') as file:
            data_meas = torch.from_numpy(file['dataMeas'][..., idx])
            data_probe = torch.from_numpy(file['dataProbe'][...])
            data_pots = torch.from_numpy(file['dataPots'][..., idx])

        cbed = data_meas.unsqueeze(0)
        probe = data_probe.unsqueeze(0)
        pot = data_pots.unsqueeze(0)
        batch_size = 32
        # Expand dimensions
        cbed1 = tf.expand_dims(cbed, axis=-1)  # Now `cbed1` has shape (1, 256, 256, 1)
        probe1 = tf.expand_dims(probe, axis=-1)  # Now `probe1` has shape (1, 256, 256, 1)

        # Replicate along the batch dimension
        cbed1 = tf.tile(cbed1, [batch_size, 1, 1, 1])  # Now `cbed1` has shape (batch_size, 256, 256, 1)
        probe1 = tf.tile(probe1, [batch_size, 1, 1, 1])  # Now `probe1` has shape (batch_size, 256, 256, 1)

        cbed = self.augmenter.augment_img(cbed1, probe1)
        return (self._replace_nan(cbed), self._replace_nan(probe),
                self._replace_nan(pot))

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.cumulative_lengths[-1]


