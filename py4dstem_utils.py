"""
This code snippet is modifiied from crystal4D repository.
https://github.com/AI-ML-4DSTEM/crystal4D/blob/dev/crystal4D/crystal4D/utils/py4dstem_utils.py
This file contains the py4dstemModel class which is used to detect disks
in an image and score the detection results.
"""
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import py4DSTEM



class Peakdet(object):
    '''Class for disk detection using py4DSTEM library.'''
    def __init__(self, maxNumPeaks=100, relativeToPeak=0,
                 probe_kernel_type='gaussian'):
        self.maxNumPeaks = maxNumPeaks
        self.relativeToPeak = relativeToPeak
        self.probe_kernel_type = probe_kernel_type

    def integrate_disks(self, image, maxima_x, maxima_y,
                        thresold=1):
        """
        Integrate disks in an image.
        """
        disks = []
        for x, y in zip(maxima_x, maxima_y):
            disk = image[int(x)-thresold:int(x)+thresold,
                         int(y)-thresold:int(y)+thresold]
            disks.append(np.average(disk))
        disks = disks/max(disks)
        return (maxima_x, maxima_y, disks)

    def aimlDiskDet(self, image,
                    aiml_param_dict,
                    integrate=False,
                    thresold=1):
        """
        Detect disks in an image using the py4DSTEM library.
        """
        maxima = py4DSTEM.process.utils.get_maxima_2D(
                image,
                minRelativeIntensity=aiml_param_dict['minRelativeIntensity'],
                edgeBoundary=aiml_param_dict['edgeBoundary'],
                maxNumPeaks=aiml_param_dict['maxNumPeaks'],
                minSpacing=aiml_param_dict['minSpacing'],
                subpixel=aiml_param_dict['subpixel'],
                upsample_factor=aiml_param_dict['upsample_factor'],
                sigma=1)

        maxima_x = maxima['x']
        maxima_y = maxima['y']
        maxima_int = maxima['intensity']
        if integrate:
            maxima_x, maxima_y, maxima_int = self.integrate_disks(
                image, maxima_x, maxima_y, thresold=thresold)
        peaks = np.zeros(len(maxima_x), dtype=[('qx', float),
                                               ('qy', float),
                                               ('intensity', float)])
        peaks['qx'] = maxima_x
        peaks['qy'] = maxima_y
        peaks['intensity'] = maxima_int
        return peaks

    def score(self, ground_truth,
              prediction,
              tr_image=None,
              pred_image=None,
              cutoff=0.3,
              pixel_size=0.0217,
              integrate_disk=False):
        """
        Compute the accuracy of the disk detection.
        """
        if integrate_disk:
            true_x, true_y, true_int = self.integrate_disks(
             tr_image, ground_truth[0], ground_truth[1], ground_truth[2])
            pred_x, pred_y, pred_int = self.integrate_disks(
             pred_image, prediction[0], prediction[1], prediction[2])
        else:
            if (len(ground_truth[2]), len(ground_truth[2]),
                    len(ground_truth[2])) == (0, 0, 0):
                return 0, 0, (0, 0, 0), (0, 0, 0)
            true_x, true_y, true_int = ground_truth[0], ground_truth[1], ground_truth[2]
            if (len(prediction[2]), len(prediction[1]),
                    len(prediction[0])) == (0, 0, 0):
                return 0, 0, (true_x, true_y, true_int), (0, 0, 0)
            pred_x, pred_y, pred_int = prediction[0], prediction[1], prediction[2]

        true_coord = np.asarray((true_x, true_y)).T
        pred_coord = np.asarray((pred_x, pred_y)).T
        true_coord = np.delete(true_coord, np.argmax(true_int), axis=0)
        pred_coord = np.delete(pred_coord, np.argmax(pred_int), axis=0)
        true_int_ = np.delete(true_int, np.argmax(true_int), axis=0)
        pred_int_ = np.delete(pred_int, np.argmax(pred_int), axis=0)
        closest_true = self.find_closest_disks(true_coord, pred_coord)
        closest_pred = self.find_closest_disks(pred_coord, true_coord)
        dist_true = np.sum((true_coord - closest_true)**2, axis=1)
        dist_pred = np.sum((pred_coord - closest_pred)**2, axis=1)
        sub_true = dist_true <= (cutoff/pixel_size)**2
        sub_pred = dist_pred <= (cutoff/pixel_size)**2

        # Intensity weighted
        TP = np.sum(sub_pred)
        FN = np.sum(np.logical_not(sub_true))
        FP = np.sum(np.logical_not(sub_pred))
        accuracy = TP/(TP+FP+FN)
        epsilon = 1e-7  # small constant

        TP_int = np.sum(pred_int_[sub_pred]) / (np.sum(pred_int_) + epsilon)
        FN_int = np.sum(true_int_[np.logical_not(sub_true)]) / (np.sum(true_int_) + epsilon)
        FP_int = np.sum(pred_int_[np.logical_not(sub_pred)]) / (np.sum(pred_int_) + epsilon)
        accuracy_int = TP_int/(TP_int+FP_int+FN_int)
        if math.isnan(accuracy) or math.isnan(accuracy_int):
            return 0, 0, (0, 0, 0), (0, 0, 0)
        return accuracy, accuracy_int, (true_x, true_y, true_int), (
            pred_x, pred_y, pred_int)

    def find_closest_disks(self, coord1, coord2):
        """
        Find the closest disks in coord2 to coord1.
        """
        if coord1.shape[0] == 0 or coord2.shape[0] == 0:
            return np.array([0, 0])
        closest = []
        for coord in range(coord1.shape[0]):
            dist_sum = np.sum((coord2 - coord1[coord])**2, axis=1)
            closest.append(coord2[np.argmin(dist_sum), :])
        return np.asarray(closest)


class Accuracy:
    """
    Accuracy class for complex unet model.
    Process batch of images and compute the accuracy.
    """
    def __init__(self, integrate_disk=False):
        """
        Initializes the class with the disk detection model and parameters.

        Parameters:
        - integrate_disk: Whether to integrate disks for scoring (boolean).
        """
        self.model = Peakdet(maxNumPeaks=100, relativeToPeak=0,
                             probe_kernel_type='gaussian')
        self.aiml_param_dict = {
            'minRelativeIntensity': 0.0,  # Ensure float
            'edgeBoundary': 10,
            'maxNumPeaks': 100,
            'minSpacing': 10,
            'subpixel': "pixel",
            'upsample_factor': 16
        }
        self.integrate_disk = integrate_disk

    def process_image(self, image):
        """
        Process a single image: remove channel dimension and ensure the
        result is a float array.

        Parameters:
        - image: Image tensor of shape [c, h, w].

        Returns:
        - Processed image numpy array of shape [h, w], as float.
        """
        if image.shape[0] == 1:  # Assuming c=1
            image = image.squeeze(0)  # Remove channel dimension
        return image
    # Ensure the image is float64

    def score(self, gt_batch, pred_batch):
        """
        Computes the average accuracy across a batch of g
        round truth and predicted images.

        Parameters:
        - gt_batch: Batch of ground truth images
        (numpy array of shape [b, c, h, w]).
        - pred_batch: Batch of predicted images
        (numpy array of shape [b, c, h, w]).

        Returns:
        - Average accuracy across the batch, as float.
        """
        accuracies = []
        for gt, pred in zip(gt_batch, pred_batch):
            gt_processed = self.process_image(gt)
            pred_processed = self.process_image(pred)
            gt_peaks = self.model.aimlDiskDet(gt_processed,
                                              self.aiml_param_dict)
            pred_peaks = self.model.aimlDiskDet(pred_processed,
                                                self.aiml_param_dict)
            _, accuracy_int, _, _ = self.model.score(
                ground_truth=(gt_peaks['qx'],
                              gt_peaks['qy'],
                              gt_peaks['intensity']),
                prediction=(pred_peaks['qx'],
                            pred_peaks['qy'],
                            pred_peaks['intensity']),
                tr_image=gt_processed,
                pred_image=pred_processed,
                integrate_disk=self.integrate_disk
            )
            accuracies.append(accuracy_int)
        return np.mean(accuracies)

    def plot_coord(self, gt_batch, pred_batch, num_to_plot,
                   save_dir):
        """
        Plot and save comparisons for a selected number of images from ground
        truth and prediction batches,
        dynamically computing and including accuracy in the plot legend.

        Parameters:
        - gt_batch: Batch of ground truth images.
        - pred_batch: Batch of predicted images.
        - num_to_plot: Number of images to plot.
        - current_epoch: The current training/testing epoch.
        - save_dir: Directory to save the plots.
        """
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Limit the number of plots to the minimum of
        # num_to_plot and the batch size
        num_to_plot = min(num_to_plot, len(gt_batch))

        for i in range(num_to_plot):
            gt_processed = self.process_image(gt_batch[i])
            pred_processed = self.process_image(pred_batch[i])
            gt_peaks = self.model.aimlDiskDet(gt_processed,
                                              self.aiml_param_dict)
            pred_peaks = self.model.aimlDiskDet(pred_processed,
                                                self.aiml_param_dict)
            _, accuracy, _, _ = self.model.score(
                                ground_truth=(gt_peaks['qx'],
                                              gt_peaks['qy'],
                                              gt_peaks['intensity']),
                                              prediction=(pred_peaks['qx'],
                                              pred_peaks['qy'],
                                            pred_peaks['intensity']),)
            # Combined plot configuration
            fig, axs = plt.subplots(1, 2, figsize=(20, 5))  # 1 row, 2 columns

            # Plot peak coordinates
            axs[0].scatter(gt_peaks['qx'], gt_peaks['qy'], c='blue',
                           label=f'GT (Acc: {accuracy:.2f})', alpha=0.5)
            axs[0].scatter(pred_peaks['qx'], pred_peaks['qy'], c='red',
                           label='Pred', alpha=0.5)
            axs[0].set_title("Peaks")
            axs[0].set_xlabel("qx")
            axs[0].set_ylabel("qy")
            axs[0].legend()
            axs[0].grid(True)
            max_length = max(len(gt_peaks['intensity']),
                             len(pred_peaks['intensity']))
            gt = np.pad(gt_peaks['intensity'],
                        (0, max_length - len(gt_peaks['intensity'])),
                        'constant')
            pred = np.pad(pred_peaks['intensity'],
                          (0, max_length - len(pred_peaks['intensity'])),
                          'constant')
            # Normalize GT intensities
            gt_min, gt_max = np.min(gt), np.max(gt)
            gt = (gt - gt_min) / ((gt_max - gt_min)+1e-7)

            # Normalize Predicted intensities
            pred_min, pred_max = np.min(pred), np.max(pred)
            pred = (pred - pred_min) / ((pred_max - pred_min)+1e-7)

            axs[1].scatter(gt, pred, alpha=0.5, c='blue', marker='o',
                           label='GT vs. Pred')

            # Adding a diagonal line for perfect agreement
            axs[1].plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
            axs[1].set_title("Intensities")
            axs[1].set_xlabel("GT Intensity")
            axs[1].set_ylabel("Pred Intensity")
            # Refine the scale to a more fine-grained view, if needed
            axs[1].set_xlim([0, 1])
            axs[1].set_ylim([0, 1])
            axs[1].grid(True)
            # Customize the ticks for a finer scale
            axs[1].set_xticks(np.arange(0, 1.05, 0.05))
            axs[1].set_yticks(np.arange(0, 1.05, 0.05))
            axs[1].legend()

            plt.suptitle(f"peaks - Image {i+1} (Acc: {accuracy:.2f})")

            # Saving the combined plot
            plt.savefig(os.path.join(save_dir,
                                     f"peak_analysis_img_{i+1}.png"))
            plt.close(fig)


# Usage:
# Initialize the disk detection with the model and parameter dictionary
acc = Accuracy()
