
ComplexUNet Training with PyTorch Lightning

Welcome to the ComplexUNet training repository. This project uses PyTorch Lightning to facilitate the training of the ComplexUNet model, offering advanced features like model checkpointing, automatic optimization, and effective visualization strategies for monitoring training progress.


Configuration

Modify settings in `args.py` to adjust model parameters and training settings according to your needs.


Key Configuration Parameters

Model Parameters: `input_channel`, `batch_size`, `learning_rate`, etc.

Training Settings: `max_epochs`, `precision`, `benchmark`, etc.


Running the Training

Start the training process by running the `main.py` script.


Training Arguments

`max_epochs`: Defines the maximum number of training epochs.

`precision`: Sets the computational precision (16 or 32 bits).

`benchmark`: Activates benchmarking to optimize performance.


Additional Features


Training Modes

Fit Mode: For training the model.

Test Mode: For evaluating the model on the test dataset.


Directory Management


Checkpoint Directory

Path: Configure the checkpoint directory in `args.py` to determine where model checkpoints will be saved.

You can also load a model from a checkpoint to resume training or test by providing the checkpoint path.


Dataset Directory

Set up directories for training, validation, and test datasets.


Model Compilation

Compiles the model for optimized GPU execution using Triton (Windows not supported).


Visualization and Evaluation

Plot Frequency: Determines how often to plot training progress images using the val_dataset.


Custom Callbacks

Use callbacks for enhanced training management, such as early stopping and checkpoints.


Profiling

Activate profiling to diagnose performance issues:

This feature leverages PyTorch Lightning's built-in profiler and generates a detailed performance report.


Hardware Strategies

CPU Usage: Set `gpus=None` to force training on the CPU when GPU resources are unavailable or undesired.

Single GPU: Set `gpus=1` to utilize a single GPU for training, harnessing GPU acceleration for faster computations.

Multi-GPU Strategy: Use `gpus=4` and `accelerator='ddp'` (or choose another strategy like 'ddp2' or 'horovod' depending on your specific needs) for distributed training across multiple GPUs, optimizing performance and scaling.

Synchronized Batch Normalization: Enable `sync_batchnorm=True` when using multiple GPUs to ensure consistent batch normalization across all devices.


Help and Support

For more detailed usage instructions or troubleshooting, refer to the inline comments in the `args.py` and `main.py` files or raise issues in this repository's issue tracker.
