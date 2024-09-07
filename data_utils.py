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

def filter_hot_pixels_pytorch(datacube: torch.Tensor, thresh: float, ind_compare: int = 1, return_mask: bool = False):
    """
    Perform pixel filtering to remove hot or bright pixels using PyTorch.
    Parameters
    ----------
    datacube : torch.Tensor
        The 4D datacube with shape (n, c, h, w)
    thresh : float
        Threshold for replacing hot pixels, if pixel value minus local ordering
        filter exceeds it.
    ind_compare : int
        Which ordered pixel value to compare against. 0 = brightest pixel,
        1 = next brightest, etc.
    return_mask : bool
        If True, returns the filter mask

    Returns
    -------
    datacube : torch.Tensor
    mask : torch.Tensor (optional)
        The bad pixel mask
    """

    # Mean image over all probe positions
    diff_mean = torch.mean(datacube, dim=(0, 1))
    if len(diff_mean.shape) == 1:
        diff_mean = diff_mean.unsqueeze(0)
    shape = diff_mean.shape

    # Moving local ordered pixel values
    shifts = [
        (-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (-1, -2), (0, -2), (1, -2),
        (-1, 2), (0, 2), (1, 2), (-2, -1), (-2, 0), (-2, 1),
        (2, -1), (2, 0), (2, 1)
    ]
    
    diff_local_med = torch.stack([
        torch.roll(diff_mean, shifts=shift, dims=(0, 1)).flatten() if len(shift) > 1 else torch.roll(diff_mean, shifts=shift, dims=0).flatten()
        for shift in shifts
    ], dim=0).sort(dim=0).values
    
    # Get the ind_compare'th pixel intensity
    diff_compare = diff_local_med[-ind_compare - 1, :].view(shape)

    # Generate mask
    mask = (diff_mean - diff_compare) > thresh

    # If the mask is empty, return
    if mask.sum().item() == 0:
        print("No hot pixels detected")
        return (datacube, mask) if return_mask else datacube

    # Otherwise, apply filtering

    # Get masked indices
    x_ma, y_ma = mask.nonzero(as_tuple=True)

    # Get local windows for each masked pixel
    xslices, yslices = [], []
    for xm, ym in zip(x_ma, y_ma):
        xslice = slice(max(xm - 1, 0), min(xm + 2, shape[0]))
        yslice = slice(max(ym - 1, 0), min(ym + 2, shape[1]))
        xslices.append(xslice)
        yslices.append(yslice)

    # Loop and replace pixels
    for ax in range(datacube.shape[0]):
        for ay in range(datacube.shape[1]):
            for xm, ym, xs, ys in zip(x_ma, y_ma, xslices, yslices):
                datacube[ax, ay, xm, ym] = torch.median(datacube[ax, ay, xs, ys])

    # Return
    return (datacube, mask) if return_mask else datacube


def min_max_normalize(tensor):
    """
    Normalize a tensor using min-max normalization.

    Args:
        tensor (torch.Tensor): The input tensor to be normalized.

    Returns:
        torch.Tensor: The normalized tensor.

    """
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


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
                 List[int]] = 256, out_channel: int = 1,
                 batch_size: int = 32):

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
        self.batch_size = batch_size
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
        if torch.isnan(tensor).any():
            return torch.nan_to_num(tensor)
        return tensor

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
        
        self.augmenter.generate_params()
        cbed = self.augmenter.augment_img(cbed, probe)
        
        return (self._replace_nan(cbed), self._replace_nan(probe),
                self._replace_nan(pot))

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.cumulative_lengths[-1]
