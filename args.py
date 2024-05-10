
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

        # #### Model arguments #####


        'input_channel': 1,  # Set the number of input channels
        'image_size': 256,  # Set the size of the input images
        'batch_size': 32,  # Set the batch size
        'filter_size': 8,  # Set the initial number of filters
        'n_depth': 1,  # Set the depth of the network
        'dp_rate': 0.1,  # Set the dropout rate
        'learning_rate': 0.0001,  # Add the learning rate
        'activation': nn.ReLU,  # Note: Use the module directly
        'shuffle': True,  # Set to False to disable shuffling
        'drop_last': True,  # Set to False to keep the last batch
        'pin_memory': False,  # Set to True to use pinned memory
        'persistent_workers': True,  # Set to True to use persistent workers
        'plot_frequency': 50,  # Set the frequency of plotting
        'num_images_to_plot': 4,  # Set the number of images to plot
        'num_workers': 9,  # Set the number of workers for data loading


        # #### Trainer arguments#####

        'gpus': None, #[0,1],  # Set to None for CPU
        'strategy': 'ddp',  # Set the strategy for distributed training
        'num_nodes': 1,  # Set the number of nodes
        'mode': 'fit',  # Set to 'fit' for training, 'test' for testing
        'max_epochs': 3,  # Set the maximum number of epochs
        'accumulate_grad_batches': 16 ,  # Set the number of batches to accumulate
        'limit_train_batches' : 1.0,  # Set the fraction of training data to us
        'track_grad_norm': -1,  # Set the norm to track
        'gradient_clip_val': 0.5,  # Set the value for gradient clipping
        'fast_dev_run': False,  # Set to True for a quick test run
        'use_profiler': False,  # Set to True to use profiler, False to not use
        'log_every_n_steps': 100,  # Set the number of steps between each log
        'check_val_every_n_epoch': 1,  # Set the frequency of validation
        'precision': '16',  # Set to 'mixed' to enable mixed training
        'benchmark': True,  # Set to True to enable benchmarking
        'gradient_clip_algorithm': 'value',  # Set the algorithm for gradient clipping
        'deterministic': False,  # Set to True to enable deterministic training
        'enable_progress_bar': False,  # Set to True to enable progress bar
        'sync_bnorm': True,  # Set to True to sync batch norm across GPUs


        # #### Directory arguments####

        'checkpoint_dir': './',  # Set the directory for saving checkpoints
        'train_dataset_dir': './train', #'/ourdisk/hpc/disc/amin/auto_archive_notyet/tape_2copies/4DSTEM_DATA/rotated_data',  # Add the directory for the training
        'test_dataset_dir': './test',  #Addthe directory for the test
        'val_dataset_dir': './val', #  '/ourdisk/hpc/disc/amin/auto_archive_notyet/tape_2copies/val/rotate',  # Add the directory for the validation
        'checkpoint_pth': None, #'./FCUnet-epoch=798.ckpt',  # Set to the path of the checkpoint to laod or None
    }
    return args
