"""
Module providing pytorch lightning for complex FCU_net.
It is designed to make research easier while providing
the tools to scale to production.
Lightning is a way to organize your PyTorch code to decouple
the science code from the engineering code.
Lightning is a PyTorch wrapper for high-performance AI research
that includes a simple, minimal interface and
a ton of functionality, including distributed training,
16-bit precision, automatic scaling, and more.
"""
from typing import Optional, Type, List
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import ComplexUNet
from loss_utils import custom_ssim_loss
from data_utils import ParseDataset


# Define MAX_SAMPLES
MAX_SAMPLES = 500


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the given image by dividing it by the maximum pixel value.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The normalized image.
    """
    max_val = np.max(image)
    if max_val > 0:
        return image / max_val
    return image


class ComplexUNetLightning(pl.LightningModule):
    """
    LightningModule implementation for the ComplexUNet model.

    Args:
        input_channel (int): Number of input channels.
        image_size (int): Size of the input image.
        filter_size (int): Size of the filters in the model.
        n_depth (int): Number of downsampling and upsampling blocks
          in the model.
        val_dataset_dir (str): Directory path of the validation dataset.
        train_dataset_dir (str): Directory path of the training dataset.
        test_dataset_dir (str): Directory path of the test dataset.
        dp_rate (float, optional): Dropout rate. Defaults to 0.3.
        activation (Optional[Type[nn.Module]], optional): Activation function.
        Defaults to nn.ReLU.
        batch_size (int, optional): Batch size. Defaults to 256.
        learning_rate (float, optional): Learning rate. Defaults to 0.001.
        num_workers (int, optional): Number of workers for data loading.
        Defaults to 4.
        shuffle (bool, optional): Whether to shuffle the data during training.
        Defaults to True.
    """

    def __init__(self, input_channel: int, image_size: int, filter_size: int,
                 n_depth: int, val_dataset_dir, train_dataset_dir,
                 test_dataset_dir, dp_rate: float = 0.3,
                 activation: Optional[Type[nn.Module]] = nn.ReLU,
                 batch_size: int = 256, learning_rate: float = 0.001,
                 plot_frequency: int = 10,
                 num_workers: int = 4, shuffle: bool = True,
                 drop_last: bool = True, pin_memory: bool = False,
                 persistent_workers: bool = False,
                 num_images_to_plot: int = 4) -> None:
        super().__init__()
        self.complex_unet = ComplexUNet(input_channel, image_size, filter_size,
                                        n_depth, dp_rate, activation)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = False
        self.drop_last = True
        self.persistent_workers = False
        self.learning_rate = learning_rate
        self.loss: List[float] = []
        self.epochs: List[int] = []
        self.targets: List[torch.Tensor] = []
        self.outputs: List[torch.Tensor] = []
        self.learning_rate = learning_rate
        self.sample_counter = 0
        self.plot_frequency = plot_frequency
        self.num_images_to_plot = num_images_to_plot
        self.val_dataset_dir = val_dataset_dir
        self.train_dataset_dir = train_dataset_dir
        self.test_dataset_dir = test_dataset_dir
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.loss_fn = custom_ssim_loss

    def forward(self, inputs_cb: torch.Tensor,
                inputs_pr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs_cb (torch.Tensor): The input for stream cb.
            inputs_pr (torch.Tensor): The input for stream pr.

        Returns:
            torch.Tensor: The output of the complex UNet model.
        """
        return self.complex_unet(inputs_cb, inputs_pr)

    def collect_samples(self, targets: torch.Tensor,
                        outputs: torch.Tensor, max_samples: int) -> None:
        """
        Collects a specified number of samples from the targets and
                                                             outputs tensors.

        Args:
            targets (torch.Tensor): The target tensor containing
                                                    the ground truth values.
            outputs (torch.Tensor): The output tensor containing
                                                    the predicted values.
            max_samples (int): The maximum number of samples to collect.

        Returns:
            None
        """
        with torch.no_grad():
            samples_to_collect = min(targets.size(0),
                                     max_samples - self.sample_counter)
            if samples_to_collect > 0:
                self.targets.append(targets[:samples_to_collect])
                self.outputs.append(outputs[:samples_to_collect])
                self.sample_counter += samples_to_collect

    def setup(self, stage=None):
        """
        Set up the datasets for training, validation, and testing.
        """
        # Load the datasets from each respective folder
        if stage == 'fit' or stage is None:
            train_dataset = ParseDataset(filepath=self.train_dataset_dir)
            self.train_dataset = train_dataset.read(mode='default')
            val_dataset = ParseDataset(filepath=self.val_dataset_dir)
            self.val_dataset = val_dataset.read(mode='default')
        if stage == 'test':
            test_dataset = ParseDataset(filepath=self.test_dataset_dir)
            self.test_dataset = test_dataset.read(mode='default')

    def train_dataloader(self):
        """
        Get the DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=self.drop_last,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        """
        Get the DataLoader for the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=0,
                          pin_memory=False, drop_last=True,
                          persistent_workers=False)

    def test_dataloader(self):
        """
        Get the DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=0,
                          pin_memory=False, drop_last=True,
                          persistent_workers=False)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch: The input batch containing inputs_cb, inputsB, and targets.
            batch_idx: The index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the loss value.
        """
        inputs_cb, inputs_pr, targets = batch
        outputs = self(inputs_cb, inputs_pr)
        total_loss, loss_1, loss_2 = self.loss_fn(targets, outputs)
        self.log('Train_loss_1', loss_1, on_step=False, on_epoch=True)
        self.log('Train_loss_2', loss_2, on_step=False, on_epoch=True)
        self.log('Train_loss', total_loss, on_step=False, on_epoch=True)
        return {'loss': total_loss}

    def on_validation_epoch_start(self):
        """
        Perform actions at the start of each validation epoch.

        Returns:
            None
        """
        self.targets.clear()
        self.outputs.clear()
        self.sample_counter = 0

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        Args:
            batch: The input batch containing inputs_cb, inputsB, and targets.
            batch_idx: The index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing
            the validation loss value.
        """
        inputs_cb, inputs_pr, targets = batch
        with torch.no_grad():
            outputs = self(inputs_cb, inputs_pr)
            total_loss, loss_1, loss_2 = self.loss_fn(targets, outputs)
            self.log('val_loss_1', loss_1, on_step=False, on_epoch=True)
            self.log('val_loss_2', loss_2, on_step=False, on_epoch=True)
            self.log('val_loss', total_loss, on_step=False, on_epoch=True)
            self.collect_samples(targets, outputs, MAX_SAMPLES)
            return {'val_loss': total_loss}

    def test_step(self, batch, batch_idx):
        """
        Perform a test step.

        Args:
            batch: The input batch containing inputs_cb, inputsB, and targets.
            batch_idx: The index of the current batch.

        Returns:
            None
        """
        with torch.no_grad():
            inputs_cb, inputs_pr, targets = batch
            outputs = self(inputs_cb, inputs_pr)
            total_loss, loss_1, loss_2 = self.loss_fn(targets, outputs)
            self.log('test_loss_1', loss_1, on_step=False, on_epoch=True)
            self.log('test_loss_2', loss_2, on_step=False, on_epoch=True)
            self.log('test_loss', total_loss, on_step=False, on_epoch=True)
            self.collect_samples(targets, outputs, MAX_SAMPLES)
            return {'test_loss': total_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['Train_loss']
        if isinstance(avg_loss, torch.Tensor):
            avg_loss = avg_loss.cpu().numpy()

        self.loss.append(avg_loss)
        self.epochs.append(self.current_epoch)

    # Check if the current epoch is a multiple of 10
        if self.current_epoch % self.plot_frequency == 0:
            plt.figure()  # Create a new figure
            plt.plot(self.epochs, self.loss, 'ro-')
            plt.title(f"Training Loss at Epoch {self.current_epoch}")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            # Ensure the directory for saving the plots exists
            save_dir = "training_plot"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/plot_{self.current_epoch}.png")
            plt.close()

    def process_epoch_end(self, num_images_to_plot: int,
                          save_dir: str) -> None:
        """
        Process the end of an epoch by concatenating targets and outputs,
        converting them to numpy arrays, saving images, and clearing the lists.

        Args:
            num_images_to_plot (int): Number of images to plot and save.
            save_dir (str): Directory to save the images.

        Returns:
            None
        """
        # Concatenate all targets and outputs
        targets = torch.cat(self.targets, dim=0)
        outputs = torch.cat(self.outputs, dim=0)

        # Convert to Float32 before converting to numpy
        # Convert tensors to numpy arrays for plotting
        targets_np = targets.cpu().to(torch.float32).numpy()
        outputs_np = outputs.cpu().to(torch.float32).numpy()

        # Save images
        self.save_images(targets_np, outputs_np, num_images_to_plot, save_dir)

        # Clear the lists
        self.targets = []
        self.outputs = []

    def on_validation_epoch_end(self):
        if self.current_epoch % self.plot_frequency == 0:
            self.process_epoch_end(self.num_images_to_plot,
                                   f"validation_image_{self.current_epoch}")

    def on_test_epoch_end(self):
        self.process_epoch_end(self.num_images_to_plot, "test_image")

    def save_images(self, targets_np: np.ndarray, outputs_np: np.ndarray,
                    num_images_to_plot: int, save_dir: str) -> None:
        """
        Save a specified number of target and output images to a directory.

        Args:
            targets_np (numpy.ndarray): Array of target images.
            outputs_np (numpy.ndarray): Array of output images.
            num_images_to_plot (int): Number of images to save and plot.
            save_dir (str): Directory path to save the images.

        Returns:
            None
        """
        # Determine how many images to save
        num_images_to_plot = min(num_images_to_plot, len(targets_np))

        # Randomly select indices to save
        indices_to_save = np.random.choice(len(targets_np),
                                           num_images_to_plot, replace=False)

        # Ensure the directory for saving the images exists
        os.makedirs(save_dir, exist_ok=True)

        # Loop through the selected indices and save each image
        for i in indices_to_save:
            # Normalize the target and output images before saving
            target_img = normalize_image(targets_np[i][0])
            output_img = normalize_image(outputs_np[i][0])

            # Save the target and output images
            # directly without scaling them to 255
            plt.imsave(os.path.join(save_dir, f"target_{i}.png"), target_img,
                       cmap='gray', format='png')
            plt.imsave(os.path.join(save_dir, f"output_{i}.png"), output_img,
                       cmap='gray', format='png')

        # Plot the first N images
        fig, axes = plt.subplots(num_images_to_plot, 2,
                                 figsize=(10, num_images_to_plot * 2))
        for j in range(num_images_to_plot):
            index = indices_to_save[j]
            axes[j, 0].imshow(normalize_image(targets_np[index][0]),
                              cmap='gray')
            axes[j, 0].set_title('Ground Truth')
            axes[j, 0].axis('off')

            axes[j, 1].imshow(normalize_image(outputs_np[index][0]),
                              cmap='gray')
            axes[j, 1].set_title('Prediction')
            axes[j, 1].axis('off')

        plt.tight_layout()
        # Optionally, save the figure to a file
        fig.savefig(os.path.join(save_dir,
                                 f'gt_vs_pred_{num_images_to_plot}.png'))
        plt.close(fig)
