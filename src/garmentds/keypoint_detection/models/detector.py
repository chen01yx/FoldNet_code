import os
import re
import argparse
import shutil

import omegaconf
from typing import Any, Dict, List
import pandas as pd

import torch
import torch.nn as nn
import torchvision

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import wandb
import lightning.pytorch as pl

from garmentds.keypoint_detection.models.backbones.backbone_factory import BackboneFactory
from garmentds.keypoint_detection.models.metrics import (
    DetectedKeypoint, Keypoint, KeypointAPMetrics )
from garmentds.keypoint_detection.utils.heatmap import (
    BCE_loss, create_heatmap_batch, get_keypoints_from_heatmap_batch_maxpool )
from garmentds.keypoint_detection.utils.visualization import (
    get_logging_label_from_channel_configuration,
    visualize_predicted_heatmaps,
    visualize_predicted_keypoints, 
)


class KeypointDetector(pl.LightningModule):
    """
    keypoint Detector using Spatial Heatmaps.
    There can be N channels of keypoints, each with its own set of ground truth keypoints.
    The mean Average precision is used to calculate the performance.

    """
    def __init__(self, cfg_pl: omegaconf.DictConfig, **kwargs):
        super().__init__()
        ## No need to manage devices ourselves, pytorch.lightning does all of that.
        ## device can be accessed through self.device if required.

        # to add new hyperparameters:
        # 1. define as named arg in the init (and use them)
        # 2. add to the config yaml file of this module
        # 3. pass them along when calling the train.py file to override their default value

        cfg_model:omegaconf.DictConfig = cfg_pl.model
        cfg_learn:omegaconf.DictConfig = cfg_pl.learn
        self.cfg_ablation:omegaconf.DictConfig = cfg_pl.ablation
        #self.cfg_ablation:omegaconf.DictConfig = omegaconf.OmegaConf.create()
        #self.cfg_ablation.only_use_mask_as_input = False

        self.learn_kwargs = omegaconf.OmegaConf.to_container(cfg_learn)
        self.backbone = BackboneFactory.create_backbone(cfg_model)
        self.heatmap_sigma = cfg_model.heatmap_sigma
        self.ap_epoch_start = cfg_model.ap_epoch_start
        self.ap_epoch_freq = cfg_model.ap_epoch_freq
        self.max_keypoints = cfg_model.max_keypoints
        self.keypoint_channel_configuration = cfg_model.keypoint_channel_configuration
        self.maximal_gt_keypoint_pixel_distances = cfg_model.maximal_gt_keypoint_pixel_distances
        self.minimal_keypoint_pixel_distance = cfg_model.minimal_keypoint_extraction_pixel_distance
        # parse the gt pixel distances
        if isinstance(self.maximal_gt_keypoint_pixel_distances, str):
            self.maximal_gt_keypoint_pixel_distances = [
                int(val) for val in self.maximal_gt_keypoint_pixel_distances.strip().split(" ")
            ]

        self.ap_training_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channel_configuration
        ]
        self.ap_validation_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channel_configuration
        ]

        self.ap_test_metrics = [
            KeypointAPMetrics(self.maximal_gt_keypoint_pixel_distances) for _ in self.keypoint_channel_configuration
        ]

        self.n_heatmaps = len(self.keypoint_channel_configuration)

        head = nn.Conv2d(
            in_channels=self.backbone.get_n_channels_out(),
            out_channels=self.n_heatmaps,
            kernel_size=(3, 3),
            padding="same",
        )

        # expect output of backbone to be normalized!
        # so by filling bias to -4, the sigmoid should be on avg sigmoid(-4) =  0.02
        # which is consistent with the desired heatmaps that are zero almost everywhere.
        # setting too low would result in loss of gradients..
        head.bias.data.fill_(-4)

        self.unnormalized_model = nn.Sequential(
            self.backbone,
            head,
        )  # NO sigmoid to combine it in the loss! (needed for FP16)

        # save hyperparameters to logger, to make sure the model hparams are saved even if
        # they are not included in the config (i.e. if they are kept at the defaults).
        # this is for later reference (e.g. checkpoint loading) and consistency.
        self.save_hyperparameters(ignore=["**kwargs"])
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self._most_recent_val_mean_ap = 0.0 # used to store the most recent validation mean AP and log it in each epoch, so that checkpoint can be chosen based on this one.

    def forward(self, x: torch.Tensor):
        """
        x shape must be of shape (N,3,H,W)
        returns tensor with shape (N, n_heatmaps, H,W)
        """
        return torch.sigmoid(self.forward_unnormalized(x))

    def forward_unnormalized(self, x: torch.Tensor):
        return self.unnormalized_model(x)

    def preprocess_batch(self, batch):
        """
        batch:  
            N x dict {
                "rgb": 3xHxW numpy array, augmented rgb image
                "mask": 3xHxW numpy array, mask of object in the image
                "keypoints": num_keypointsx1x3 numpy array

                "rgb_raw": 3xHxW numpy array, original rgb image (before augmentation)
                "mask_raw": 3xHxW numpy array, original mask of object in the image
                "keypoints_raw": num_keypointsx1x3 numpy array, original keypoints in the image (before augmentation)
                }
        """

        rgb = batch["rgb"]               # [N, 3, H, W]
        rgb_raw = batch["rgb_raw"]       # [N, 3, H, W]
        rgb_path = batch["rgb_path"]

        mask = batch["mask"]             # [N, 3, H, W]
        keypoints = batch["keypoints"]   # [N, num_keypoints, 1, 3]

        if self.cfg_ablation.only_use_mask_as_input:
            rgb_masked = torch.where(mask < 0.5, torch.ones_like(rgb)*-2, torch.ones_like(rgb))
        else:
            rgb_masked = torch.where(mask < 0.5, torch.ones_like(rgb)*-2, rgb)

        scale_factor = torch.tensor(rgb_raw.shape[-2:]) / torch.tensor(rgb.shape[-2:])
        return rgb_masked.to(self.dtype), keypoints.transpose(0, 1).to(self.dtype), \
                rgb_raw.to(self.dtype) , rgb_path, scale_factor

    def shared_step(self, batch, batch_idx, include_visualization_data_in_result_dict=False) -> Dict[str, Any]:
        """
        shared step for train and validation step that computes the heatmaps and losses and
        creates a result dict for later use in the train, validate and test step.

        returns:
            shared_dict (Dict): a dict with a.o. heatmaps, gt_keypoints and losses
        """
        input_images, keypoint_channels, original_images, rgb_paths, scale_factor = self.preprocess_batch(batch)
        heatmap_shape = input_images[0].shape[1:]

        gt_heatmaps = [
            create_heatmap_batch(heatmap_shape, keypoint_channel, self.heatmap_sigma, self.device)
            for keypoint_channel in keypoint_channels
        ] # num_keypoints x [N, H, W]

        input_images = input_images.to(self.device)

        ## predict and compute losses
        predicted_unnormalized_maps = self.forward_unnormalized(input_images)
        predicted_heatmaps = torch.sigmoid(predicted_unnormalized_maps)
        channel_losses = []
        channel_gt_losses = []

        result_dict = {}
        for channel_idx in range(len(self.keypoint_channel_configuration)):
            channel_losses.append(
                # combines sigmoid with BCE for increased stability.
                nn.functional.binary_cross_entropy_with_logits(
                    predicted_unnormalized_maps[:, channel_idx, :, :], gt_heatmaps[channel_idx],
                    weight=(keypoint_channels[channel_idx, :, :, -1] > 0.5).unsqueeze(-1)
                )
            )
            #with torch.no_grad():
            #    print("weight: ", (keypoint_channels[channel_idx, :, :, -1] > 0.5))
            #    print("with_weight: ", channel_losses[-1])
            #    print("without_weight: ", nn.functional.binary_cross_entropy_with_logits(
            #        predicted_unnormalized_maps[:, channel_idx, :, :], gt_heatmaps[channel_idx],
            #        ))           

            # pass losses and other info to result dict
            result_dict.update(
                {f"{self.keypoint_channel_configuration[channel_idx]}_loss": channel_losses[channel_idx].detach()}
            )

        loss = sum(channel_losses)
        #gt_loss = sum(channel_gt_losses)
        result_dict.update({"loss": loss, }) #"gt_loss": gt_loss})

        if include_visualization_data_in_result_dict:
            result_dict.update(
                {
                    "input_images": input_images,             # [N, 3, H, W]
                    "gt_keypoints": keypoint_channels,        # [num_keypoints, N, 1, 3]
                    "predicted_heatmaps": predicted_heatmaps, # [N, num_keypoints, H, W]
                    "gt_heatmaps": gt_heatmaps,               # [num_keypoints, N, H, W]
                }
            )

        result_dict.update({"rgb_paths": rgb_paths})
        result_dict.update({"scale_factor": scale_factor})
        result_dict.update({"original_images": original_images})
        return result_dict

    def training_step(self, train_batch, batch_idx):
        should_log_ap = self.is_ap_epoch() and batch_idx < 20  # limit AP calculation to first 20 batches to save time
        log_images = batch_idx == 0 and self.current_epoch > 0 and self.is_ap_epoch()
        include_vis_data = log_images or should_log_ap

        result_dict = self.shared_step(
            train_batch, batch_idx, include_visualization_data_in_result_dict=include_vis_data
        )

        if should_log_ap:
            self.update_ap_metrics(result_dict, self.ap_training_metrics)

        for channel_name in self.keypoint_channel_configuration:
            self.log(f"train/{channel_name}_loss", 
                     result_dict[f"{channel_name}_loss"], 
                     sync_dist=True)

        self.log("train/loss", result_dict["loss"], sync_dist=True)  # also logs steps?
        return result_dict

    def validation_step(self, val_batch, batch_idx):
        # no need to switch model to eval mode, this is handled by pytorch lightning
        result_dict = self.shared_step(val_batch, batch_idx, include_visualization_data_in_result_dict=True)

        self.update_ap_metrics(result_dict, self.ap_validation_metrics)

        log_images = batch_idx == 0 and self.current_epoch % 10 == 0
        print(f"[ INFO ]Log images: {log_images}")
        if log_images:
            channel_grids, _ = self.visualize_predictions_channels(result_dict, num_to_visualize=1)
            self.log_channel_predictions_grids(channel_grids, mode="validation")
            keypoint_grids, _ = self.visualize_predicted_keypoints(result_dict, num_to_visualize=1)
            self.log_predicted_keypoints(keypoint_grids, mode="validation")

        ## log (defaults to on_epoch, which aggregates the logged values over entire validation set)
        self.log("validation/epoch_loss", 
                 result_dict["loss"], 
                 on_epoch=True, sync_dist=True)
        return result_dict

    def test_step(self, test_batch, batch_idx):
        # no need to switch model to eval mode, this is handled by pytorch lightning
        result_dict = self.shared_step(test_batch, batch_idx, include_visualization_data_in_result_dict=True)
        
        base_dir = self.learn_kwargs["test"]["output_dir"]
        self.update_ap_metrics(result_dict, self.ap_test_metrics)
        
        # only log first few batches to reduce storage space
        if batch_idx < self.learn_kwargs["test"]["num_batches_to_visualize"]:
            batch_dir = os.path.join(base_dir, f"batch_{batch_idx}")
            os.makedirs(batch_dir, exist_ok=True)
            num_to_visualize = min(32, len(result_dict["rgb_paths"]))

            image_grids, all_channels = self.visualize_predictions_channels(result_dict, num_to_visualize)
            keypoint_grids, all_keypoints = self.visualize_predicted_keypoints(result_dict, num_to_visualize)

            ## visualize all channels in a single heatmap
            if all_channels != None:
                for idx, img in enumerate(all_channels):
                    Image.fromarray((img*255).permute(1,2,0).numpy().astype(np.uint8))\
                         .save(os.path.join(batch_dir, f"heatmap_all_channels_{idx}.png"))

            ## visualize all the keypoints in a single image
            if all_keypoints != None:
                for idx, img in enumerate(all_keypoints):
                    img.save(os.path.join(batch_dir, f"keypoint_all_channels_{idx}.png"))

            num_channels = len(self.keypoint_channel_configuration)
            _, C, H, W = keypoint_grids[0].shape
            for i in range(0): # num_to_visualize): 
                fig = plt.figure(figsize=(10*3, 10*(H/W)*num_channels))
                for channel_idx, channel_name in enumerate(self.keypoint_channel_configuration):
                    heatmaps_channel = image_grids[channel_idx][[i, num_to_visualize+i]] # [2, 3, H, W]
                    keypoint_channel = keypoint_grids[channel_idx][i:i+1]  # [1, 3, H, W]
                
                    # In a row, we have the pred heatmap, gt heatmap, and keypoint
                    ax1 = fig.add_subplot(num_channels, 3, 3*channel_idx+1)
                    ax1.imshow((heatmaps_channel[0]*255).numpy().astype(np.uint8).transpose(1,2,0))
                    ax1.axis('off')
                    ax2 = fig.add_subplot(num_channels, 3, 3*channel_idx+2)
                    ax2.imshow((heatmaps_channel[1]*255).numpy().astype(np.uint8).transpose(1,2,0))
                    ax2.axis('off')
                    ax3 = fig.add_subplot(num_channels, 3, 3*channel_idx+3)
                    ax3.imshow((keypoint_channel[0]*255).numpy().astype(np.uint8).transpose(1,2,0))
                    ax3.axis('off')

                # add caption to the image and export it
                export_path = os.path.join(batch_dir, f"cloth_{i}.pdf")
                plt.figtext(0.5, 1.0, "Left: predicted heatmap, Middle: gt heatmap, Right: keypoint", ha='center', va='bottom', fontsize=50)
                plt.tight_layout()
                fig.savefig(export_path, format="pdf")

        self.log("test/epoch_loss", (result_dict["loss"]), 
                 on_epoch=True, sync_dist=True)
        #self.log("test/gt_loss", result_dict["gt_loss"])
        return result_dict

    def on_train_epoch_end(self):
        """
        Called on the end of a training epoch.
        Used to compute and log the AP metrics.
        """
        if self.is_ap_epoch():
            self.log_and_reset_ap_kd("train")

    def on_validation_epoch_end(self):
        """
        Called on the end of a validation epoch.
        Used to compute and log the AP metrics.
        """
        self.log_and_reset_ap_kd("validation")
        if type(self._most_recent_val_mean_ap) == torch.Tensor:
            self.log("checkpointing_metrics/valmeanAP", 
                self._most_recent_val_mean_ap.clone().detach().to(self.device), sync_dist=True)
        else:
            self.log("checkpointing_metrics/valmeanAP", 
                torch.tensor(self._most_recent_val_mean_ap, device=self.device), sync_dist=True)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """
        Called on the end of a test epoch.
        Used to compute and log the AP metrics.
        """
        self.log_and_reset_ap_kd("test")
        self.test_step_outputs.clear()

    def log_predicted_keypoints(self, grid, mode=str):
        label = f"predicted_keypoints_{mode}"
        image_caption = "predicted keypoints"
        keypoints = torch.cat(grid, dim=0)
        grided_keypoints = torchvision.utils.make_grid(keypoints, nrow=len(keypoints))
        self.logger.experiment.add_image(label, grided_keypoints, self.global_rank)

    def log_channel_predictions_grids(self, image_grids, mode: str):
        heatmaps = torch.stack(image_grids).transpose(0, 1)
        _, num_channels, C, H, W = heatmaps.shape
        grid = torchvision.utils.make_grid(heatmaps.reshape(-1, C, H, W), nrow=num_channels)
        label = "top: predicted heatmaps, bottom: gt heatmaps"
        self.logger.experiment.add_image(label, grid, self.global_step)

    def log_and_reset_ap_kd(self, mode: str):
        mean_ap_kd_per_threshold = torch.zeros((len(self.maximal_gt_keypoint_pixel_distances),2))
        kd_percentiles = []
        if mode == "train":
            metrics = self.ap_training_metrics
        elif mode == "validation":
            metrics = self.ap_validation_metrics
        elif mode == "test":
            metrics = self.ap_test_metrics
        else:
            raise ValueError(f"mode {mode} not recognized")

        # calculate APs for each channel and each threshold distance, and log them
        print(f" # {mode} metrics:")
        for channel_idx, channel_name in enumerate(self.keypoint_channel_configuration):
            channel_aps_kds, channel_kp_percentile = \
                self.compute_and_log_metrics_for_channel(metrics[channel_idx], channel_name, mode)
            mean_ap_kd_per_threshold += torch.tensor(channel_aps_kds)
            kd_percentiles.append(channel_kp_percentile)

        # calculate the mAP over all channels for each threshold distance, and log them
        for i, maximal_distance in enumerate(self.maximal_gt_keypoint_pixel_distances):
            self.log(
                f"{mode}/meanAP/d={float(maximal_distance):.1f}",
                (mean_ap_kd_per_threshold[i][0] / len(self.keypoint_channel_configuration)).clone().detach().to(self.device),
                sync_dist=True,
            )

        # calculate the mAP over all channels and all threshold distances, and log it
        mean_ap_kd = mean_ap_kd_per_threshold.mean(dim=0) / len(self.keypoint_channel_configuration)
        self.log(f"{mode}/meanAP", mean_ap_kd[0].clone().detach().to(self.device), sync_dist=True)
        self.log(f"{mode}/avgKD", mean_ap_kd[1].clone().detach().to(self.device), sync_dist=True)
        self.log(f"{mode}/meanAP/meanAP", mean_ap_kd[0].clone().detach().to(self.device), sync_dist=True)

        if mode== "validation":
            self._most_recent_val_mean_ap = mean_ap_kd[0]
        if mode == "test":
            kd_percentiles = np.stack(kd_percentiles)
            percentiles_df = pd.DataFrame(kd_percentiles)
            percentiles_df.columns = list(range(10, 101, 10))
            percentiles_df.index = self.keypoint_channel_configuration
            print(f" # {mode} KD percentiles:")
            print(percentiles_df)

    def update_ap_metrics(self, result_dict, ap_metrics):
        predicted_heatmaps = result_dict["predicted_heatmaps"]
        gt_keypoints = result_dict["gt_keypoints"]
        scale_factor = result_dict["scale_factor"]
        for channel_idx in range(len(self.keypoint_channel_configuration)):
            predicted_heatmaps_channel = predicted_heatmaps[:, channel_idx, :, :]
            gt_keypoints_channel = gt_keypoints[channel_idx]
            self.update_channel_ap_metrics(predicted_heatmaps_channel, gt_keypoints_channel, 
                                           ap_metrics[channel_idx], scale_factor)

    def update_channel_ap_metrics(
        self, predicted_heatmaps: torch.Tensor, gt_keypoints: List[torch.Tensor], 
        validation_metric: KeypointAPMetrics, scale_factor: torch.Tensor
    ):
        """
        Updates the AP metric for a batch of heatmaps and keypoins of a single channel (!)
        This is done by extracting the detected keypoints for each heatmap and combining them with the gt keypoints for the same frame, so that
        the confusion matrix can be determined together with the distance thresholds.

        predicted_heatmaps: N x H x W tensor with the batch of predicted heatmaps for a single channel
        gt_keypoints: List of size N, containing K_i x 2 tensors with the ground truth keypoints for the channel of that sample
        """
        # log corner keypoints to AP metrics for all images in this batch
        formatted_gt_keypoints = [
            [Keypoint(int(k[0]), int(k[1])) for k in frame_gt_keypoints] for frame_gt_keypoints in gt_keypoints if not torch.all(frame_gt_keypoints[:, -1]<0.5)
        ]
        batch_detected_channel_keypoints = self.extract_detected_keypoints_from_heatmap(
            predicted_heatmaps.unsqueeze(1)
        ) # [N, 1, max_keypoints] of DetectedKeypoint objects
        batch_detected_channel_keypoints = [batch_detected_channel_keypoints[i][0] for i in range(len(gt_keypoints)) if not torch.all(gt_keypoints[i][:, -1]<0.5)]
        
        #print("before filter, len of keypoints: ", len(gt_keypoints))
        #print("after filter, len of keypoints: ", len(formatted_gt_keypoints))
        #print("after filter, len of detected keypoints: ", len(batch_detected_channel_keypoints))
        
        for i, detected_channel_keypoints in enumerate(batch_detected_channel_keypoints):
            validation_metric.update(detected_channel_keypoints, formatted_gt_keypoints[i], scale_factor)

    def compute_and_log_metrics_for_channel(
        self, metrics: KeypointAPMetrics, channel: str, training_mode: str
    ) -> List[float]:
        """
        logs AP of predictions of single Channel for each threshold distance.
        Also resets metric and returns resulting AP for all distances.
        """
        ap_kd_metrics, kd_percentile = metrics.compute()
        #rounded_ap_metrics = {k: round(v, 3) for k, v in ap_metrics.items()}
        #print(f"{channel} : {rounded_ap_metrics}")
        aps, kds = [], []
        for maximal_distance, (ap, kd) in ap_kd_metrics.items():
            aps.append(ap)
            kds.append(kd)
            self.log(f"{training_mode}/{channel}_ap/d={float(maximal_distance):.1f}", 
                     torch.tensor(ap, device=self.device), sync_dist=True)
            
        mean_ap = sum(aps) / len(aps)
        self.log(f"{training_mode}/{channel}_ap/meanAP", 
                 torch.tensor(mean_ap, device=self.device), sync_dist=True)
        mean_kd = sum(kds) / len(kds)
        self.log(f"{training_mode}/{channel}_kd", 
                 torch.tensor(mean_kd, device=self.device), sync_dist=True)

        metrics.reset()
        return list(ap_kd_metrics.values()), kd_percentile

    def is_ap_epoch(self) -> bool:
        """Returns True if the AP should be calculated in this epoch."""
        is_epch = self.ap_epoch_start <= self.current_epoch and self.current_epoch % self.ap_epoch_freq == 0
        # always log the AP in the last epoch
        is_epch = is_epch or self.current_epoch == self.trainer.max_epochs - 1

        # if user manually specified a validation frequency, we should always log the AP in that epoch
        # is_epch = is_epch or (self.current_epoch > 0 and self.trainer.check_val_every_n_epoch > 1)
        return is_epch

    def extract_detected_keypoints_from_heatmap(self, heatmap: torch.Tensor) -> List[DetectedKeypoint]:
        """
        Extract keypoints from a single channel prediction and format them for AP calculation.

        Args:
        heatmap (torch.Tensor) : N x 1 x H x W tensor that represents a heatmap.
        """
        if heatmap.dtype == torch.float16:
            # Maxpool_2d not implemented for FP16 apparently
            heatmap_to_extract_from = heatmap.float()
        else:
            heatmap_to_extract_from = heatmap

        keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
            heatmap_to_extract_from, self.max_keypoints, self.minimal_keypoint_pixel_distance, return_scores=True
        )
        detected_keypoints = [
            [[] for _ in range(heatmap_to_extract_from.shape[1])] for _ in range(heatmap_to_extract_from.shape[0])
        ]
        for batch_idx in range(len(detected_keypoints)):
            for channel_idx in range(len(detected_keypoints[batch_idx])):
                for kp_idx in range(len(keypoints[batch_idx][channel_idx])):
                    detected_keypoints[batch_idx][channel_idx].append(
                        DetectedKeypoint(
                            keypoints[batch_idx][channel_idx][kp_idx][0],
                            keypoints[batch_idx][channel_idx][kp_idx][1],
                            scores[batch_idx][channel_idx][kp_idx],
                        )
                    )

        return detected_keypoints
    
    def visualize_predictions_channels(self, result_dict, num_to_visualize):
        original_images = result_dict["original_images"]
        gt_heatmaps = result_dict["gt_heatmaps"]
        predicted_heatmaps = result_dict["predicted_heatmaps"]

        one_channel_grid, all_channel_grid = visualize_predicted_heatmaps(
            original_images, predicted_heatmaps, gt_heatmaps, 
            self.keypoint_channel_configuration  ,num_to_visualize)

        return (one_channel_grid, # num_channels x [2*num_images, 3, H, W]
                all_channel_grid) # [num_images, 3, H, W]

    def visualize_predicted_keypoints(self, result_dict, num_to_visualize):
        scale_factor = result_dict["scale_factor"]
        original_images = result_dict["original_images"]
        gt_keypoints = result_dict["gt_keypoints"] # [num_keypoints, N, 1, 3]
        predicted_heatmaps = result_dict["predicted_heatmaps"]
        # get the keypoints from the heatmaps
        predicted_heatmaps = predicted_heatmaps.detach().float()
        predicted_keypoints, predicted_scores = get_keypoints_from_heatmap_batch_maxpool(
            predicted_heatmaps, self.max_keypoints, self.minimal_keypoint_pixel_distance, 
            abs_max_threshold=0.1, return_scores=True
        )
        # overlay the images with the keypoints
        gt_keypoints = gt_keypoints.detach().cpu().numpy().transpose(1,0,2,3)
        one_channel_grid, all_channel_grid = visualize_predicted_keypoints(
            original_images, scale_factor, predicted_keypoints, predicted_scores, 
            gt_keypoints, self.keypoint_channel_configuration, num_to_visualize)

        return (one_channel_grid, # num_channels x Tensor([num_images, 3, H, W])
                all_channel_grid) # num_images x PIL.Image([H, W, 3])

    def configure_optimizers(self):
        optimizer = [
            getattr(torch.optim, self.learn_kwargs["optimizer"]["name"])(
                self.backbone.parameters(), 
                **(self.learn_kwargs["optimizer"]["cfg"]),
            )
        ]
        schedule = [
            getattr(torch.optim.lr_scheduler, self.learn_kwargs["schedule"]["name"])(
                optimizer[0],
                **(self.learn_kwargs["schedule"]["cfg"]),
            )
        ]
        return optimizer, schedule

if __name__ == "__main__":
    pass
