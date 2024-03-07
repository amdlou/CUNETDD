import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import pytorch_lightning as pl
from model import ComplexUNet
from loss_utils import CustomSSIMLoss
from data_utils import ParseDataset



def normalize_image(image):
        max_val = np.max(image)
        if max_val > 0:
            return image / max_val
        return image


class ComplexUNetLightning(pl.LightningModule):
    def __init__(self, input_channel, image_size, filter_size, n_depth, dp_rate=0.3, activation=nn.ReLU,batch_size=256):
        super(ComplexUNetLightning, self).__init__()
        self.complex_unet = ComplexUNet(input_channel, image_size, filter_size, n_depth, dp_rate, activation)
        self.batch_size = batch_size
        self.loss=[]
        self.epochs=[]
        #self.all_targets=[]
        #self.all_outputs=[]
        self.targets=[]
        self.outputs=[]
        self.sample_counter = 0
        
        self.loss_fn = CustomSSIMLoss
        self.los_fn=F.mse_loss

    def forward(self, inputsA, inputsB):
        return self.complex_unet(inputsA, inputsB)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,drop_last=True,num_workers=0,pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,drop_last=True,num_workers=0,pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,drop_last=True)
    #def on_training_epoch_start(self):
        #print("Validation epoch start")
        #self.targets.clear()  # Clear the list of targets at the start of validation epoch
        #self.outputs.clear()  # Clear the list of outputs at the start of validation epoch
        #self.sample_counter = 0  # Reset the sample counter at the start of validation epoch
    
    def training_step(self, batch, batch_idx):
        inputsA, inputsB, targets = batch
        #print(inputsA.shape,inputsB.shape,targets.shape)
        outputs = self(inputsA, inputsB)
        #print(f"Output: {outputs.size()}")
        
        loss_1 =self.loss_fn(targets, outputs)
        self.log('Train_loss_1', loss_1)
        loss_2 = self.los_fn(targets, outputs)
        self.log('Train_loss_2', loss_2)
        loss=loss_1+loss_2
        self.log('Train_loss', loss)
        
        #print(f"Loss: {loss}")
        return {'loss': loss}
    
    def on_validation_epoch_start(self):
        #print("Validation epoch start")
        self.targets.clear()  # Clear the list of targets at the start of validation epoch
        self.outputs.clear()  # Clear the list of outputs at the start of validation epoch
        self.sample_counter = 0  # Reset the sample counter at the start of validation epoch
        #make sure that the lists are empty
        #print(f"Targets: {len(self.targets)}, Outputs: {len(self.outputs)}")
        #print(f"Sample counter: {self.sample_counter}")

        
    def validation_step(self, batch, batch_idx):
        inputsA, inputsB, targets = batch
        outputs = self(inputsA, inputsB)
        loss_1 =self.loss_fn(targets, outputs)
        self.log('val_loss_1', loss_1)
        loss_2 = self.los_fn(targets, outputs)
        self.log('val_loss_2', loss_2)
        loss=loss_1+loss_2
        loss = F.mse_loss(targets, outputs)
        self.log('val_loss', loss) 
        #samples should be cleared before each epoch      
        MAX_SAMPLES = 500
        samples_to_collect = min(targets.size(0), MAX_SAMPLES - self.sample_counter)
        if samples_to_collect > 0:
            self.targets.append(targets[:samples_to_collect].detach())
            self.outputs.append(outputs[:samples_to_collect].detach())
            self.sample_counter += samples_to_collect
        #print(f"Sample counter: {self.sample_counter}")
        
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        inputsA, inputsB, targets = batch
        outputs = self(inputsA, inputsB)
        loss_1=self.loss_fn(targets, outputs)
        self.log('test_loss_1', loss_1)
        loss_2 = self.los_fn(targets, outputs)
        self.log('test_loss_2', loss_2)
        loss=loss_1+loss_2
        self.log('test_loss', loss)
        MAX_SAMPLES = 500
        samples_to_collect = min(targets.size(0), MAX_SAMPLES - self.sample_counter)
        if samples_to_collect > 0:
            self.targets.append(targets[:samples_to_collect].detach())
            self.outputs.append(outputs[:samples_to_collect].detach())
            self.sample_counter += samples_to_collect
        #self.targets.append(targets.detach())
        #self.outputs.append(outputs.detach())
        return {'test_loss': loss}
        

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
        targets_np = targets.cpu().numpy()
        outputs_np = outputs.cpu().numpy()

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
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            #print(f"Targets: {targets_np.shape}, Outputs: {outputs_np.shape}")

            # Save images
            #foldername should be unique for each epoch
            self.save_images(targets_np, outputs_np, num_images_to_plot, f"validation_image_{self.current_epoch}")           
            #clear the lists
            self.targets=[]
            self.outputs=[]
            #priint count of tartgets and outputs
            #print(f"Targets: {len(self.targets)}, Outputs: {len(self.outputs)}")
        



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
