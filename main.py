"""MAIN.PY: Main file for training
the ComplexUNet model using PyTorch Lightning.
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

from typing import List
from argparse import Namespace
import torch
from pytorch_lightning.profilers import PyTorchProfiler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model_pl import ComplexUNetLightning
from args import get_args


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


def create_model(params: Namespace) -> ComplexUNetLightning:
    """
    Create a ComplexUNetLightning model based on the given parameters.

    Args:
        params (Namespace): The namespace object containing
                                       the model parameters.

    Returns:
        ComplexUNetLightning: The created ComplexUNetLightning model.
    """
    model_params = {
        key: value for key, value in vars(params).items()
        if key not in [
            'mode', 'use_profiler', 'max_epochs', 'gpus',
            'fast_dev_run', 'checkpoint_dir', 'checkpoint_pth',
            'log_every_n_steps', 'sync_bnorm', 'accelerator',
            'check_val_every_n_epoch', 'precision', 'benchmark',
            'deterministic', 'enable_progress_bar'
        ]
    }
    if params.checkpoint_pth:
        return ComplexUNetLightning.load_from_checkpoint(
            checkpoint_path=params.checkpoint_pth, **model_params
        )
    return ComplexUNetLightning(**model_params)


def main(params: Namespace) -> None:
    """
    Main function for training a ComplexUNet model.

    Args:
        params (Namespace): Hyperparameters for training.

    Returns:
        None
    """

    model = create_model(params)
    torch.compile(model)

    trainer = pl.Trainer(
        profiler=PyTorchProfiler(dirpath='./', filename='profiler_report')
        if params.use_profiler else None,
        max_epochs=params.max_epochs,
        accelerator='cpu' if params.gpus is None else 'gpu',
        enable_progress_bar=params.enable_progress_bar,
        callbacks=configure_callbacks(params),
        fast_dev_run=params.fast_dev_run,
        sync_batchnorm=params.sync_bnorm,
        log_every_n_steps=params.log_every_n_steps,
        precision=params.precision,
        benchmark=params.benchmark,
        deterministic=params.deterministic,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
    )

    getattr(trainer, params.mode)(model)


if __name__ == '__main__':
    args = get_args()
    hparams = Namespace(**args)
    main(hparams)
