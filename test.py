import torch
import torch.nn as nn
import pytorch_lightning as pl
from argparse import Namespace

from torchmetrics import Precision
from fcunet import ComplexUNet  
from datautils import ParseDataset 
from lightningunet import ComplexUNetLightning
from pytorch_lightning.callbacks import ModelCheckpoint
from torchinfo import summary
from pytorch_lightning.profilers import PyTorchProfiler

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
   
hparams = Namespace(
    max_epochs=2,
    input_channel=1,
    image_size=256,
    filter_size=16,
    n_depth=3,
    dp_rate=0.3,
    activation=nn.ReLU,  # Note: Use the module directly, not an instance
    batch_size=2,
    gpus=None ,
    checkpoint_dir='./' ) # Set to None for CPU)
callbacks = configure_callbacks(hparams)
model = ComplexUNetLightning.load_from_checkpoint( #checkpoint_path="exp_f_0/FCUnet-epoch=492.ckpt",
                                                 input_channel=hparams.input_channel,
                                                 #map_location=torch.device('cpu'), 
                                                 image_size=hparams.image_size,
                                                 filter_size=hparams.filter_size,
                                                 n_depth=hparams.n_depth,
                                                 dp_rate=hparams.dp_rate,
                                                 activation=hparams.activation,
                                                 batch_size=hparams.batch_size,
                                                  
                                                  )





profiler = PyTorchProfiler(
    #dirpath="./",  # Directory to save profiler reports
    #filename="profiler_report",    # Base filename for the reports
    #record_shapes=True,            # Whether to record tensor shapes
    #profile_memory=True,           # Whether to profile memory usage
    #with_stack=True                # Whether to capture the stack trace of operations
)

#model.setup(stage='test')                                                 
trainer = pl.Trainer(max_epochs=hparams.max_epochs ,accelerator='cpu' if hparams.gpus is None else 'gpu',callbacks=callbacks)
trainer.fit(model)
#trainer.test(model=model, dataloaders=model.test_dataloader())

#trainer.test(model)
