import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model import ComplexUNet
from loss_utils import CustomSSIMLoss
from data_utils import ParseDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# Define MAX_SAMPLES
MAX_SAMPLES = 500

def normalize_image(image):
        max_val = np.max(image)
        if max_val > 0:
            return image / max_val
        return image

class ComplexUNetLightning(pl.LightningModule):
    def __init__(self, input_channel, image_size, filter_size, n_depth, dp_rate=0.3, activation=nn.ReLU,batch_size=256,num_workers=4,shuffle=True):
        super(ComplexUNetLightning, self).__init__()
        self.complex_unet = ComplexUNet(input_channel, image_size, filter_size, n_depth, dp_rate, activation)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.loss=[]
        self.epochs=[]
        self.targets=[]
        self.outputs=[]
        self.sample_counter = 0
        self.los_fn=F.mse_loss
        self.SSIM_loss = CustomSSIMLoss

    def forward(self, inputsA, inputsB):
        return self.complex_unet(inputsA, inputsB)
    
    def calculate_loss(self, targets, outputs):
        loss_1 = self.SSIM_loss(targets, outputs)
        loss_2 = F.mse_loss(targets, outputs)
        total_loss = loss_1 + loss_2
        return total_loss, loss_1, loss_2

    def create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True, num_workers=self.num_workers ,persistent_workers=True, pin_memory=True)

    def train_dataloader(self):
        return self.create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self.create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.create_dataloader(self.test_dataset)
   
    def collect_samples(self, targets, outputs, max_samples):
        with torch.no_grad():
            samples_to_collect = min(targets.size(0), max_samples - self.sample_counter)
            if samples_to_collect > 0:
                self.targets.append(targets[:samples_to_collect])
                self.outputs.append(outputs[:samples_to_collect])
                self.sample_counter += samples_to_collect
    
    def training_step(self, batch, batch_idx):
        inputsA, inputsB, targets = batch
        outputs = self(inputsA, inputsB)
        total_loss, loss_1, loss_2 = self.calculate_loss(targets, outputs)
        self.log('Train_loss_1', loss_1)
        self.log('Train_loss_2', loss_2)
        self.log('Train_loss', total_loss)
        return {'loss': total_loss}
    
    def on_validation_epoch_start(self):
        self.targets.clear()  # Clear the list of targets at the start of validation epoch
        self.outputs.clear()  # Clear the list of outputs at the start of validation epoch
        self.sample_counter = 0  # Reset the sample counter at the start of validation epoch
 
    def validation_step(self, batch, batch_idx):
        inputsA, inputsB, targets = batch
        with torch.no_grad():
            outputs = self(inputsA, inputsB)
            total_loss, loss_1, loss_2 = self.calculate_loss(targets, outputs)
            self.log('val_loss_1', loss_1)
            self.log('val_loss_2', loss_2)
            self.log('val_loss', total_loss)
            self.collect_samples(targets, outputs, MAX_SAMPLES)
            return {'val_loss': loss_2}

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            inputsA, inputsB, targets = batch
            outputs = self(inputsA, inputsB)
            total_loss, loss_1, loss_2 = self.calculate_loss(targets, outputs)
            self.log('test_loss_1', loss_1)
            self.log('test_loss_2', loss_2)
            self.log('test_loss', total_loss)
            self.collect_samples(targets, outputs, MAX_SAMPLES)
            return {'test_loss': total_loss}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def setup(self, stage=None):
        # Load the full dataset
        torch.manual_seed(42)
        full_dataset = ParseDataset(filepath='3fb20ba5-ed3d-4c55-9e35-cfaa97b85cdd_training.h5')
        full_dataset.read(batch_size=self.batch_size, shuffle=True, mode='default', task='system', one_hot=False)
        # Split the dataset into training, validation, and test datasets
        train_size = int(0.7 * len(full_dataset))  # Use 70% of the data for training
        val_size = int(0.15 * len(full_dataset))  # Use 15% of the data for validation
        test_size = len(full_dataset) - train_size - val_size
        print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")
                
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        ''''
        # Create the DataLoaders once in the setup method
        train_folder = 'Train'
        val_folder = 'val'
        test_folder = 'test'

        # Load the datasets from each respective folder
        self.train_dataset = ParseDataset(filepath=train_folder)
        self.val_dataset = ParseDataset(filepath=val_folder)
        self.test_dataset = ParseDataset(filepath=test_folder)
        
        # Read the dataset
        self.train_dataset.read(batch_size=self.batch_size, shuffle=True, mode='default', task='system', one_hot=False)
        self.val_dataset.read(batch_size=self.batch_size, shuffle=False, mode='default', task='system', one_hot=False)
        self.test_dataset.read(batch_size=self.batch_size, shuffle=False, mode='default', task='system', one_hot=False)
      '''
    def on_train_epoch_end(self, num_images_to_plot=2):
        
        avg_loss = self.trainer.callback_metrics['Train_loss']
        if isinstance(avg_loss, torch.Tensor):
           avg_loss = avg_loss.cpu().numpy()
    
        self.loss.append(avg_loss)
        self.epochs.append(self.current_epoch)

    # Check if the current epoch is a multiple of 10
        if self.current_epoch % 10 == 0: 
           plt.figure()  # Create a new figure
           plt.plot(self.epochs, self.loss, 'ro-')
           plt.title('Training loss')
           plt.xlabel('Epoch')
           plt.ylabel('Loss')
        
           # Ensure the directory for saving the plots exists
           save_dir = "training_plot"
           os.makedirs(save_dir, exist_ok=True)
        
           # Save the plot with a unique filename for each epoch
           #plt.savefig(os.path.join(save_dir, f'training_loss_epoch_{self.current_epoch}.png'))
        
           # Clear the plot after saving to avoid memory issues with multiple plots
           plt.close()
    
    def on_test_epoch_end(self, num_images_to_plot=10):
        # Concatenate all targets and outputs
        targets = torch.cat(self.targets, dim=0)
        outputs = torch.cat(self.outputs, dim=0)

        # Convert tensors to numpy arrays for plotting
        targets_np = targets.cpu().to(torch.float32).numpy()
        outputs_np = outputs.cpu().to(torch.float32).numpy()

        # Save images
        self.save_images(targets_np, outputs_np, num_images_to_plot, "test_image")
        #clear the lists
        self.targets=[]
        self.outputs=[]

    def on_validation_epoch_end(self,num_images_to_plot=10):
        #print("Validation epoch end")
        #only for each 10th epoch
        if self.current_epoch % 10 == 0:
            # Concatenate all targets and outputs
            targets = torch.cat(self.targets, dim=0)
            outputs = torch.cat(self.outputs, dim=0)

            # Convert tensors to numpy arrays for plotting
            targets_np = targets.cpu().to(torch.float32).numpy()
            outputs_np = outputs.cpu().to(torch.float32).numpy() 
            
            # Save images
            #foldername should be unique for each epoch
            #self.save_images(targets_np, outputs_np, num_images_to_plot, f"validation_image_{self.current_epoch}")           
            #clear the lists
            self.targets=[]
            self.outputs=[]
  
    def save_images(self, targets_np, outputs_np, num_images_to_plot, save_dir):
        # Determine how many images to save
        num_images_to_plot = min(num_images_to_plot, len(targets_np))

        # Randomly select indices to save
        indices_to_save = np.random.choice(len(targets_np), num_images_to_plot, replace=False)

        # Ensure the directory for saving the images exists
        os.makedirs(save_dir, exist_ok=True)

        # Loop through the selected indices and save each image
        for i in indices_to_save:
            # Normalize the target and output images before saving
            target_img = normalize_image(targets_np[i][0])  # Assuming single-channel images
            output_img = normalize_image(outputs_np[i][0])

            # Save the target and output images directly without scaling them to 255
            plt.imsave(os.path.join(save_dir, f"target_{i}.png"), target_img, cmap='gray', format='png')
            plt.imsave(os.path.join(save_dir, f"output_{i}.png"), output_img, cmap='gray', format='png')

        # Plot the first N images
        fig, axes = plt.subplots(num_images_to_plot, 2, figsize=(10, num_images_to_plot * 2))
        for j in range(num_images_to_plot):
            index = indices_to_save[j]
            axes[j, 0].imshow(normalize_image(targets_np[index][0]), cmap='gray')
            axes[j, 0].set_title('Ground Truth')
            axes[j, 0].axis('off')

            axes[j, 1].imshow(normalize_image(outputs_np[index][0]), cmap='gray')
            axes[j, 1].set_title('Prediction')
            axes[j, 1].axis('off')

        plt.tight_layout()
        # Optionally, save the figure to a file
        fig.savefig(os.path.join(save_dir, f'gt_vs_pred_{num_images_to_plot}.png'))
        plt.close(fig)
