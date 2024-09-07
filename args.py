"""
This file contains the arguments for the CUNETD model.
"""
from typing import Dict, Any
import argparse
from torch import nn


def get_args() -> Dict[str, Any]:

    """
    Get the arguments for the CUNETD model.

    Returns:
        args (Dict[str, Any]): A dictionary containing the arguments for
        the CUNETD model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_nodes', type=int, default=1, help='Set the number of nodes')
    parser.add_argument('--num_workers', type=int, default=0, help='Set the number of workers for data loading')
    parsed_args = parser.parse_args()
    
    args: Dict[str, Any] = {

        # #### Model arguments #####

        'input_channel': 1,  # Set the number of input channels
        'image_size': 256,  # Set the size of the input images
        'batch_size': 16,  # Set the batch size
        'filter_size': 32,  # Set the initial number of filters
        'n_depth': 3,  # Set Number of convolutional layers in each block
        'dp_rate': 0.3,  # Set the dropout rate
        'learning_rate': 0.001,  # Add the learning rate
        'activation': nn.ReLU,  # Note: Use the module directly
        'shuffle': True,  # Set to False to disable shuffling
        'drop_last': True,  # Set to False to keep the last batch
        'pin_memory': True,  # Set to True to use pinned memory
        'persistent_workers': True,  # Set to True to use persistent workers
        'plot_frequency': 10,  # Set the frequency of plotting
        'num_images_to_plot': 4,  # Set the number of images to plot
        'num_workers': parsed_args.num_workers,  # Use the parsed num_workers

        # #### Trainer arguments#####
        'num_nodes': parsed_args.num_nodes,  # Use the parsed num_nodes
        'gpus': -1,  # Set to None for CPU
        'strategy': 'ddp_find_unused_parameters_true',  # Set the strategy for distributed training
        'mode': 'fit',  # Set to 'fit' for training, 'test' for testing
        'max_epochs': 200,  # Set the maximum number of epochs
        'accumulate_grad_batches': 16,  # Set the number of batches to accumulate
        'limit_train_batches': 1.0,  # Set the fraction of training data
        'track_grad_norm': -1,  # Set the norm to track
        'gradient_clip_val': 0.5,  # Set the value for gradient clipping
        'fast_dev_run': False,  # Set to True for a quick test run
        'use_profiler': False,  # Set to True to use profiler, False to not use
        'log_every_n_steps': 10,  # Set the number of steps between each log
        'check_val_every_n_epoch': 1,  # Set the frequency of validation
        'precision': '16',  # Set to 'mixed' to enable mixed training
        'benchmark': True,  # Set to True to enable benchmarking
        'gradient_clip_algorithm': 'value',  # Set the algorithm for gradient clipping
        'deterministic': False,  # Set to True to enable deterministic training
        'enable_progress_bar': False,  # Set to True to enable progress bar
        'sync_bnorm': True,  # Set to True to sync batch norm across GPUs


        # #### Directory arguments####

        'image_folder_name': 'validation_image',  # Set the name of the main folder
        'checkpoint_dir': './',  # Set the directory for saving checkpoints
        'train_dataset_dir': './train',  # '/ourdisk/hpc/disc/amin/auto_archive_notyet/tape_2copies/4DSTEM_DATA/rotated_data',  # Add the directory for the training
        'checkpoint_pth': None, #'./FCUnet-epoch=69.ckpt',  # './FCUnet-epoch=798.ckpt',  # Set to the path of the checkpoint to laod or None
    }
    return args
