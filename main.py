"""MAIN.PY: Main file for training the ComplexUNet model using PyTorch Lightning."""
from pickle import TRUE
from typing import Any, List, Dict
from argparse import Namespace
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import Callback
from model_pl import ComplexUNetLightning

def configure_callbacks(hparams: Namespace) -> List[Callback]:
    """
    Configure the callbacks for the training process.

    Args:
        hparams (Namespace): Hyperparameters for the training process.

    Returns:
        list: List of callbacks to be used during training.
    """
    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=1000,
        verbose=True,
        mode='min')
      
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=hparams.checkpoint_dir,
        filename='FCUnet-{epoch:02d}',
        save_top_k=1,
        verbose=True,
        mode='min')
    return [early_stop, checkpoint]
    
def main(hparams: Namespace) -> None:
    """
    Main function for training a ComplexUNet model.

    Args:
        hparams (Namespace): Hyperparameters for training.

    Returns:
        None
    """
    callbacks = configure_callbacks(hparams)
    torch.backends.cudnn.benchmark = True

    model = ComplexUNetLightning(
        input_channel=hparams.input_channel,
        image_size=hparams.image_size,
        filter_size=hparams.filter_size,
        n_depth=hparams.n_depth,
        dp_rate=hparams.dp_rate,
        activation=hparams.activation,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=hparams.shuffle,
    )
    torch.compile(model, mode="reduce-overhead")
    
    #profiler = PyTorchProfiler(dirpath='./',
    #                           filename='profiler_report')
    trainer = pl.Trainer(
    #    profiler=profiler,
        max_epochs=hparams.max_epochs,
        accelerator='cpu' if hparams.gpus is None else 'gpu',  # Set gpus to the number of GPUs you want to use or None for CPU
        enable_progress_bar=False,
        callbacks=callbacks,
        fast_dev_run=hparams.fast_dev_run,
        log_every_n_steps=50,  # 50 is the default value
        precision=16,  # Enable mixed precision training why??
    ) 
    trainer.fit(model)


if __name__ == '__main__':
    args: Dict[str, Any] = {
        'input_channel': 1,
        'fast_dev_run': False,  # Set to True for a quick test run
        'image_size': 256,
        'batch_size':2,
        'filter_size': 4,
        'n_depth': 1,
        'dp_rate': 0.3,
        'gpus': None,  # Set to None for CPU
        'activation': nn.ReLU,
        'max_epochs': 5,
        'checkpoint_dir': './',
        'shuffle': True,  # Set to False to disable shuffling
        'num_workers': 4,  # Set the number of workers for data loading
    }
        
    hparams = Namespace(**args)
    main(hparams)
    