from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont

from garmentds.keypoint_detection.utils.heatmap import generate_channel_heatmap

# Kelly's 22 colors for max contrast
# cf. https://gist.github.com/ollieglass/f6ddd781eeae1d24e391265432297538
DISTINCT_COLORS = [
    "#F2F3F4",
    "#F3C300",
    "#875692",
    "#F38400",
    "#A1CAF1",
    "#BE0032",
    "#C2B280",
    "#848482",
    "#008856",
    "#E68FAC",
    "#0067A5",
    "#F99379",
    "#604E97",
    "#F6A600",
    "#B3446C",
    "#DCD300",
    "#222222",
    "#882D17",
    "#8DB600",
    "#654522",
    "#E25822",
    "#2B3D26",
    "#FF0000",
    "#00FF00"
]


def get_logging_label_from_channel_configuration(channel_configuration: List[List[str]], mode: str) -> str:
    channel_name = channel_configuration

    if isinstance(channel_configuration, list):
        if len(channel_configuration) == 1:
            channel_name = channel_configuration[0]
        else:
            channel_name = f"{channel_configuration[0]}+{channel_configuration[1]}+..."

    channel_name_short = (channel_name[:40] + "...") if len(channel_name) > 40 else channel_name
    if mode != "":
        label = f"{channel_name_short}_{mode}"
    else:
        label = channel_name_short
    return label

#########################################################
# The following routines are used to visualize heatmaps #
#########################################################

def draw_one_channel_heatmaps_on_image(
    images: torch.Tensor,   # [num_images, 3, H, W]
    heatmaps: torch.Tensor, # [num_images, H', W']
    label: str,
    alpha=0.3
) -> torch.Tensor:
    viridis = cm.get_cmap("viridis")
    heatmaps = viridis(heatmaps.detach().cpu().numpy())[..., :3]  # viridis: grayscale -> RGBa
    heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
    heatmaps = heatmaps.permute((0, 3, 1, 2))  # HxWxC -> CxHxW for pytorch

    # scale the heatmap to put the max value at 1.0
    max_val = heatmaps.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    heatmaps = torch.square(heatmaps / max_val)

    heatmaps = torch.nn.functional.interpolate(
        heatmaps, size=images.shape[2:], mode="bilinear", align_corners=False,
    )  # resize to match image size
    images = images.detach().cpu()

    overlayed_images = alpha * (images/255.0) + (1 - alpha) * heatmaps
    return overlayed_images

def draw_all_channel_heatmaps_on_image(
    images: torch.Tensor,   # [num_images, 3, H, W]
    heatmaps: torch.Tensor, # [num_images, num_channels, H', W']
    label: str,
    alpha=0.3
):
    viridis = cm.get_cmap("viridis")
    mix_heatmap = torch.max(heatmaps, dim=1)[0]
    mix_heatmap = viridis(mix_heatmap.detach().cpu().numpy())[..., :3]  # viridis: grayscale -> RGBa
    mix_heatmap = torch.tensor(mix_heatmap, dtype=torch.float32)
    mix_heatmap = mix_heatmap.permute((0, 3, 1, 2))  # HxWxC -> CxHxW for pytorch

    # scale the heatmap to put the max value at 1.0
    max_val = mix_heatmap.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    mix_heatmap = torch.square(mix_heatmap / max_val)

    mix_heatmap = torch.nn.functional.interpolate(
        mix_heatmap, size=images.shape[2:], mode="bilinear", align_corners=False,
    )  # resize to match image size
    images = images.detach().cpu()

    overlayed_images = alpha * (images/255.0) + (1-alpha) * mix_heatmap

    return overlayed_images # [num_images, 3, H, W]

def visualize_predicted_heatmaps(
    imgs: torch.Tensor,
    predicted_heatmaps: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    channel_configuration: List[List[str]],
    num_images: int = 1,
):

    one_channel_image_grid = []
    all_channel_image_grid = None

    for channel_idx in range(len(channel_configuration)):
        pred_heatmaps_channel = predicted_heatmaps[:, channel_idx, :, :]
        gt_heatmaps_channel = gt_heatmaps[channel_idx]
        predicted_heatmap_overlays = draw_one_channel_heatmaps_on_image(
            imgs[:num_images], pred_heatmaps_channel[:num_images], "predicted")
        gt_heatmap_overlays = draw_one_channel_heatmaps_on_image(
            imgs[:num_images], gt_heatmaps_channel[:num_images], "ground truth")

        image = torch.cat([predicted_heatmap_overlays, gt_heatmap_overlays]) # [2*num_images, 3, H, W]
        one_channel_image_grid.append(image)

    all_channel_image_grid = draw_all_channel_heatmaps_on_image(
        imgs[:num_images], predicted_heatmaps[:num_images], "predicted")

    return (one_channel_image_grid, # num_channels x Tensor([2*num_images, 3, H, W])
            all_channel_image_grid) # Tensor([num_images, 3, H, W])


##########################################################
# The following routines are used to visualize keypoints #
##########################################################

def get_coord(triangle:np.ndarray, center_xy:np.ndarray):
    """
    This is a helper function to generate coordinates of a tirangle
    in a specific format that can be used to draw on an image.
    """
    coord = triangle + center_xy
    return [tuple(xy) for xy in coord]

def draw_one_channel_keypoints_on_image(
    image: Image, 
    scale_factor: torch.Tensor, 
    image_keypoints: List[Tuple[int, int]], 
    image_scores: List[float], 
    gt_keypoints: List[Tuple[int, int, int]], 
    channel_configuration: List[List[str]]
) -> Image:
    """adds all keypoints to the PIL image, with different colors for each channel."""
    color_pool = DISTINCT_COLORS
    image_size = image.size
    min_size = min(image_size)
    radius = 1 + (min_size // 256)
    scale_y, scale_x = scale_factor.tolist()

    draw = ImageDraw.Draw(image)
    tiny_triangle = np.array([[-np.sin(np.pi/3), np.cos(np.pi/3)], 
                              [np.sin(np.pi/3), np.cos(np.pi/3)], [0, -1]]) * 4*radius

    # draw caption
    draw.ellipse((20 - 3*radius,4*radius,20 + 3*radius,10*radius), fill=color_pool[-2])
    draw.text(
        (20 + 6 * radius, 2 * radius), f"=pred",
        fill=color_pool[-2], font=ImageFont.truetype("FreeMono.ttf", size=10 * radius))
    draw.polygon(get_coord(tiny_triangle, np.array([150, 7*radius])), fill=color_pool[-1])
    draw.text(
        (150 + 5 * radius, 2 * radius), f"=ground truth",
        fill=color_pool[-1], font=ImageFont.truetype("FreeMono.ttf", size=10 * radius))
    
    # draw keypoint
    for keypoint, gt_keypoint in zip(image_keypoints, gt_keypoints):
        u, v = keypoint
        u, v = int(u * scale_x), int(v * scale_y)
        gt_u, gt_v, gt_score = gt_keypoint
        gt_u, gt_v = int(gt_u * scale_x), int(gt_v * scale_y)
        draw.ellipse((u - 3*radius, v - 3*radius, u + 3*radius, v + 3*radius), fill=color_pool[-2])
        if gt_score > 0.5: # visible
            draw.polygon(get_coord(tiny_triangle, np.array([gt_u, gt_v])), fill=color_pool[-1])
        break

    if len(image_scores) > 0:
        score = image_scores[0]
    else:
        score = -1.0
    draw.text(
        (10, 10 * radius),
        #get_logging_label_from_channel_configuration(channel_configuration[channel_idx], f"{score:.4f}"),
        f"p={score:.4f}, gt={gt_score:.4f}",
        fill=color_pool[0],
        font=ImageFont.truetype("FreeMono.ttf", size=10 * radius),
    )

    return image

def draw_all_channel_keypoints_on_image(
    image: Image, 
    scale_factor: torch.Tensor, 
    image_keypoints: List[List[Tuple[int, int]]], 
    image_scores: List[List[float]], 
    gt_keypoints: List[List[Tuple[int, int, int]]], 
    channel_configuration: List[List[str]]
) -> Image:

    image_size = image.size
    min_size = min(image_size)
    radius = 1 + (min_size // 256)
    scale_y, scale_x = scale_factor.tolist()

    draw = ImageDraw.Draw(image)
    color_pool = DISTINCT_COLORS
    tiny_triangle = np.array([[-np.sin(np.pi/3), np.cos(np.pi/3)], 
                      [np.sin(np.pi/3), np.cos(np.pi/3)], [0, -1]]) * 5*radius

    for channel_idx, keypoints in enumerate(image_keypoints):
        keypoint = keypoints[0]
        gt_keypoint = gt_keypoints[channel_idx][0]
        u, v = keypoint
        u, v = int(u * scale_x), int(v * scale_y)
        gt_u, gt_v, gt_score = gt_keypoint
        gt_u, gt_v = int(gt_u * scale_x), int(gt_v * scale_y)
        draw.ellipse((u - 4*radius, v - 4*radius, u + 4*radius, v + 4*radius), fill=color_pool[channel_idx])
        #draw.ellipse((gt_u - 5*radius, gt_v - 5*radius, gt_u + 5*radius, gt_v + 5*radius), fill="#BE0032")
        
        #draw.polygon(get_coord(tiny_triangle, np.array([gt_u, gt_v])), fill=color_pool[channel_idx])

        #draw.text(
        #    (gt_u, gt_v),
        #    #get_logging_label_from_channel_configuration(channel_configuration[channel_idx], f"{score:.4f}"),
        #    f"{channel_idx}",
        #    fill="#FFFFFF",
        #    font=ImageFont.truetype("FreeMono.ttf", size=10 * radius),
        #)
        draw.text(
            (u, v),
            f"{channel_idx}",
            #f"{channel_idx}_{np.sqrt((u-gt_u)**2+(v-gt_v)**2):.1f}",
            
            fill="#000000",
            font=ImageFont.truetype("FreeMono.ttf", size=4 * radius),
            stroke_width=1,
            stroke_fill="#FFFFFF",
        )

    return image

def visualize_predicted_keypoints(
    images: torch.Tensor, 
    scale_factor: torch.Tensor, 
    keypoints: List[List[List[List[int]]]], 
    scores: List[List[List[float]]], 
    gt_keypoints: List[List[List[List[int]]]], 
    channel_configuration: List[List[str]],
    num_images: int = 1,
):
    images = images.detach().cpu()
    one_channel_image_grid = []
    all_channel_image_grid = []

    for channel_idx in range(len(channel_configuration)):
        channel_images = []
        for i in range(num_images):
            # PIL expects uint8 images
            image = images[i].permute(1, 2, 0).numpy()
            image = Image.fromarray(image.astype(np.uint8))
            keypoint_channel = keypoints[i][channel_idx]
            score_channel = scores[i][channel_idx]
            gt_keypoint_channel = gt_keypoints[i][channel_idx]
            image_overlay = draw_one_channel_keypoints_on_image(
                image, scale_factor, keypoint_channel, score_channel, 
                gt_keypoint_channel, channel_configuration)

            channel_images.append(torch.from_numpy(np.array(image_overlay)).permute(2, 0, 1) / 255)
        one_channel_image_grid.append(torch.stack(channel_images))

    for i in range(num_images):
        image = images[i].permute(1, 2, 0).numpy()
        image = Image.fromarray(image.astype(np.uint8))
        image_overlay = draw_all_channel_keypoints_on_image(
            image, scale_factor, keypoints[i], scores[i], gt_keypoints[i], channel_configuration)
        all_channel_image_grid.append(image_overlay)

    return (one_channel_image_grid, # num_channels x Tensor([num_images, 3, H, W])
            all_channel_image_grid) # num_images x PIL.Image([H, W, 3])

if __name__ == "__main__":
    pass

    """Script to visualize dataset"""
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
    from keypoint_detection.tasks.train import parse_channel_configuration
    from keypoint_detection.utils.heatmap import create_heatmap_batch

    parser = ArgumentParser()
    parser.add_argument("json_dataset_path")
    parser.add_argument("keypoint_channel_configuration")
    args = parser.parse_args()

    hparams = vars(parser.parse_args())
    hparams["keypoint_channel_configuration"] = parse_channel_configuration(hparams["keypoint_channel_configuration"])

    dataset = COCOKeypointsDataset(**hparams)
    batch_size = 6
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    images, keypoint_channels = next(iter(dataloader))

    shape = images.shape[2:]

    heatmaps = create_heatmap_batch(shape, keypoint_channels[0], sigma=6.0, device="cpu")
    grid = visualize_predicted_heatmaps(images, heatmaps, heatmaps, 6)

    image_numpy = grid.permute(1, 2, 0).numpy()
    plt.imshow(image_numpy)
    plt.show()
    """
    imgs = torch.rand(6, 3, 256, 256)
    heatmaps = torch.rand(6, 256, 256)
    gt_heatmaps = torch.rand(6, 256, 256)

    visualize_predicted_heatmaps(imgs, heatmaps, gt_heatmaps)
