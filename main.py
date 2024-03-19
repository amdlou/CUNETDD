"""MAIN.PY: Main file for training
ComplexUNet model using PyTorch Lightning.
custom callbacks are used to monitor the training process.
compile method uses openai/triton which converts the model to gpu machine code.
compile imporves the performance of the model.
benchmark help to find fastest algorithm for the model on cuda.
Script is flexible and allows to control various hyperparameters
and training settings.
fast_dev_run helps to run the model for a quick test run.
mixed precision is used to reduce the memory usage and improve the performance.
easy to save and continue the training process.
validation loss is monitored to evaluate the model performance while training.
profiler is used to monitor the training process and find the bottlenecks.
inference is faster and more efficient.
Distributed training is also supported.
Multi processing is used to load the data faster.
accelerator is used to control the training process on cpu or gpu.
"""

from typing import Any, List, Dict
from argparse import Namespace
import torch
from torch import nn
from pytorch_lightning.profilers import PyTorchProfiler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model_pl import ComplexUNetLightning


def configure_callbacks(params) -> List[Callback]:
    """
    Configure the callbacks for the training process.

    Args:
        params (Namespace): Hyperparameters for the training process.

    Returns:
        list: List of callbacks to be used during training.
    """
    early_stop = EarlyStopping(
        monitor='Train_loss_2',
        min_delta=0.00,
        patience=1000,
        verbose=True,
        mode='min'
    )

    checkpoint = ModelCheckpoint(
        monitor='Train_loss_2',
        dirpath=params.checkpoint_dir,
        filename='FCUnet-{epoch:02d}',
        save_top_k=1,
        verbose=True,
        mode='min'
    )
    return [early_stop, checkpoint]


def main(params: Namespace) -> None:
    """
    Main function for training a ComplexUNet model.

    Args:
        params (Namespace): Hyperparameters for training.

    Returns:
        None
    """
    callbacks = configure_callbacks(params)
    if params.checkpoint_pth:
        model = ComplexUNetLightning.load_from_checkpoint(
            checkpoint_path=params.checkpoint_pth,
            # map_location=torch.device('cpu'),
            input_channel=params.input_channel,
            image_size=params.image_size,
            filter_size=params.filter_size,
            n_depth=params.n_depth,
            dp_rate=params.dp_rate,
            activation=params.activation,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            num_workers=params.num_workers,
            shuffle=params.shuffle,
            val_dataset_dir=params.val_dataset_dir,
            train_dataset_dir=params.training_dataset_dir,
            test_dataset_dir=params.test_dataset_dir)
    else:
        model = ComplexUNetLightning(
            input_channel=params.input_channel,
            image_size=params.image_size,
            filter_size=params.filter_size,
            n_depth=params.n_depth,
            dp_rate=params.dp_rate,
            activation=params.activation,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
            num_workers=params.num_workers,
            shuffle=params.shuffle,
            val_dataset_dir=params.val_dataset_dir,
            train_dataset_dir=params.train_dataset_dir,
            test_dataset_dir=params.test_dataset_dir)
    torch.compile(model)

    profiler = PyTorchProfiler(dirpath='./', filename='profiler_report')
    trainer = pl.Trainer(
        profiler=profiler,
        max_epochs=params.max_epochs,
        accelerator='cpu' if params.gpus is None else 'gpu',
        enable_progress_bar=False,
        callbacks=callbacks,
        fast_dev_run=params.fast_dev_run,
        log_every_n_steps=50,  # 50 is the default value
        precision=16,
        benchmark=True,
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    args: Dict[str, Any] = {
        'input_channel': 1,  # Set the number of input channels
        'fast_dev_run': False,  # Set to True for a quick test run
        'image_size': 256,  # Set the size of the input images
        'batch_size': 2,  # Set the batch size
        'filter_size': 4,  # Set the initial number of filters
        'n_depth': 1,  # Set the depth of the network
        'dp_rate': 0.3,  # Set the dropout rate
        'gpus': None,  # Set to None for CPU
        'activation': nn.ReLU,  # Note: Use the module directly
        'max_epochs': 5,  # Set the maximum number of epochs
        'checkpoint_dir': './',  # Set the directory for saving checkpoints
        'shuffle': True,  # Set to False to disable shuffling
        'num_workers': 4,  # Set the number of workers for data loading
        'checkpoint_pth': None,  # Set to the path of the checkpoint to laod
        'learning_rate': 0.001,  # Add the learning rate
        'train_dataset_dir': './train',  # Add the directory for the training
        'test_dataset_dir': './test',  # Add the directory for the test
        'val_dataset_dir': './val',  # Add the directory for the validation
    }

    hparams = Namespace(**args)
    main(hparams)
