"""
This file contains the arguments for the CUNETD model.
"""

from typing import Dict, Any
from torch import nn


def get_args() -> Dict[str, Any]:

    """
    Get the arguments for the CUNETD model.

    Returns:
        args (Dict[str, Any]): A dictionary containing the arguments for
        the CUNETD model.
    """
    args: Dict[str, Any] = {

        # #### Main arguments #####

        'input_channel': 1,  # Set the number of input channels
        'image_size': 256,  # Set the size of the input images
        'batch_size': 2,  # Set the batch size
        'filter_size': 4,  # Set the initial number of filters
        'n_depth': 1,  # Set the depth of the network
        'dp_rate': 0.3,  # Set the dropout rate
        'num_workers': 4,  # Set the number of workers for data loading
        'plot_frequency': 10,  # Set the frequency of plotting
        'num_images_to_plot': 4,  # Set the number of images to plot
        'learning_rate': 0.001,  # Add the learning rate
        'log_every_n_steps': 50,  # Set the number of steps between each log
        'max_epochs': 10,  # Set the maximum number of epochs
        'chek_val_every_n_epoch': 1,  # Set the frequency of validation
        'activation': nn.ReLU,  # Note: Use the module directly

        # #### Boolean arguments#####

        'shuffle': True,  # Set to False to disable shuffling
        'fast_dev_run': False,  # Set to True for a quick test run
        'use_profiler': False,  # Set to True to use profiler, False to not use
        'sync_bnorm': False,  # Set to True to sync batch norm across GPUs
        'gpus': None,  # Set to None for CPU
        'mode': 'fit',  # Set to 'fit' for training, 'test' for testing
        'pin_memory': True,  # Set to True to use pinned memory
        'persistent_workers': True,  # Set to True to use persistent workers


        # #### Directory arguments#####

        'checkpoint_pth': None,  # Set to the path of the checkpoint to laod
        'checkpoint_dir': './',  # Set the directory for saving checkpoints
        'train_dataset_dir': './train',  # Add the directory for the training
        'test_dataset_dir': './test',  # Add the directory for the test
        'val_dataset_dir': './val',  # Add the directory for the validation
    }
    return args
