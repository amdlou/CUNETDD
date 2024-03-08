import os
import torch
import torch.nn as nn
from model_pl import ComplexUNetLightning
from pytorch_lightning.profilers import PyTorchProfiler
from argparse import Namespace
import pytorch_lightning as pl



def configure_callbacks(hparams):
    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00, #if min_delta is 0.1, then even if the reduction between two consequetive steps is 0.05, it is considered as a not-seccusful step
        patience=1000,#number of not-seccussful iterations before failure 
        verbose=True,
         mode='min')#'min' means descending trend is favorite and if the value increases, it is considered as a bad iteration
      
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=hparams.checkpoint_dir,
        filename='FCUnet-{epoch:02d}',
        save_top_k=1,#number of best models to save
        verbose=True,
        mode='min')
    return [early_stop, checkpoint]
    
def main(hparams):

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
    profiler = PyTorchProfiler()
    trainer = pl.Trainer(
                         max_epochs=hparams.max_epochs,
                        accelerator='cpu' if hparams.gpus is None else 'gpu',  # Set gpus to the number of GPUs you want to use or None for CPU
                        enable_progress_bar=False,
                        callbacks=callbacks,
                        fast_dev_run=hparams.fast_dev_run,
                        log_every_n_steps=50, #50 is the default value
                        precision=16,  # Enable mixed precision training why??

                        ) 
    #torch.compile(model)
    trainer.fit(model)
    #trainer.test(model)


if __name__ == '__main__':
    args = {
        'input_channel': 1,
        'fast_dev_run': False,  # Set to True for a quick test run
        'image_size': 256,
        'batch_size':2,
        'filter_size': 16,
        'n_depth': 2,
        'dp_rate': 0.3,
        'gpus': None,  # Set to None for CPU
        'activation': nn.ReLU,
        'max_epochs': 2,
        'checkpoint_dir': './',
        'shuffle': True,  # Set to False to disable shuffling
        'num_workers': 4,  # Set the number of workers for data loading
    }
    
    hparams = Namespace(**args)
    main(hparams)

