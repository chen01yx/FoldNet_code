import os
from dataclasses import dataclass
from typing import Optional, Union
import copy
import pickle

import numpy as np
import torch
from PIL import Image, ImageDraw
import cv2

# disable warning
import torchvision
torchvision.disable_beta_transforms_warning()

import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv
from supervision.draw.color import ColorPalette


@dataclass(eq=True)
class SAMCfg:
    device: str = "cuda"
    use_cache: bool = True
    dino_model_id: str = "IDEA-Research/grounding-dino-tiny"


class VideoTracker:
    method: str = "box"
    sample_pos: int = 5
    sample_neg: int = 5
    box_expand_ratio: float = 0.0
    def __init__(self, sam2_predictor: SAM2ImagePredictor):
        self.sam2_predictor = sam2_predictor
        self.current_mask: np.ndarray = None
    
    def set_init_mask(self, init_mask: np.ndarray):
        self.current_mask = init_mask
    
    def segment_and_update(self, image: Union[np.ndarray, Image.Image]):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self.sam2_predictor.set_image(image)

        image_new = np.array(image).copy()
        if self.method == "mask":
            masks, scores, logits = self.sam2_predictor.predict(
                mask_input=cv2.resize(self.current_mask, (256, 256))[None, :, :],
                multimask_output=False,
            )
        elif self.method == "sample":
            i0, j0 = np.where(self.current_mask == 0.)
            n0 = min(self.sample_neg, len(i0))
            idx0 = np.random.choice(len(i0), n0, replace=False)
            i0, j0 = i0[idx0], j0[idx0]
            i1, j1 = np.where(self.current_mask == 1.)
            n1 = min(self.sample_pos, len(i1))
            idx1 = np.random.choice(len(i1), n1, replace=False)
            i1, j1 = i1[idx1], j1[idx1]

            point_coords = np.concatenate([np.stack([i0, j0], axis=1), np.stack([i1, j1], axis=1)], axis=0)
            point_labels = np.concatenate([np.zeros(n0), np.ones(n1)], axis=0)
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            image_new[i0, j0, :] = [255, 0, 0]
            image_new[i1, j1, :] = [0, 255, 0]
        elif self.method == "box":
            i, j = np.where(self.current_mask == 1.)
            x1, y1, x2, y2 = min(j), min(i), max(j), max(i)
            x1, y1, x2, y2 = x1 - (x2 - x1) * self.box_expand_ratio, y1 - (y2 - y1) * self.box_expand_ratio, x2 + (x2 - x1) * self.box_expand_ratio, y2 + (y2 - y1) * self.box_expand_ratio
            def clip_x(x): return max(0, min(image.width - 1, x))
            def clip_y(y): return max(0, min(image.height - 1, y))
            x1, y1, x2, y2 = clip_x(x1), clip_y(y1), clip_x(x2), clip_y(y2)
            box = np.array([x1, y1, x2, y2])
            masks, scores, logits = self.sam2_predictor.predict(
                box=box,
                multimask_output=False,
            )
            image_new = Image.fromarray(image_new)
            draw = ImageDraw.Draw(image_new)
            draw.rectangle(box.tolist(), fill=None, outline="red")
            image_new = np.array(image_new)
        else:
            raise ValueError(f"Invalid method: {self.method}")

        if masks.ndim == 4:
            masks = masks.squeeze(1)
        mask = masks[0] # select the first mask
        self.current_mask = mask
        return mask, image_new
    
    def reset(self):
        self.current_mask = None
    
    def need_init_mask(self):
        return self.current_mask is None


class GroundedSAM:
    CACHE_PATH = os.path.join(os.path.dirname(__file__), "GS_tmp_cache.pkl")

    def __init__(self, cfg: Optional[SAMCfg] = None):
        if cfg is None:
            cfg = SAMCfg()
        self.cfg = copy.deepcopy(cfg)

        cache_path = os.path.join(os.path.dirname(__file__), f"GS_tmp_cache_{cfg}.pkl".replace(" ", "").replace("/", "_"))

        cwd = os.getcwd()
        os.chdir(os.path.dirname(sam2.__path__[0]))

        processor, grounding_model, sam2_model, load_cache_success = None, None, None, False
        if cfg.use_cache:
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                    processor, grounding_model, sam2_model, cfg_cache = data["processor"], data["grounding_model"], data["sam2_model"], data["cfg"]
                    assert cfg == cfg_cache, f"cfg is different from the cached one: {cfg} vs {cfg_cache}"
                    load_cache_success = True
            except Exception as e:
                print(e)
                print("[WARN] load from cache failed ...")

        print("[INFO] loading SAM2 model ...")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        if sam2_model is None:
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, cfg.device)
        self.sam2_model = sam2_model
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        self.video_tracker = VideoTracker(self.sam2_predictor)

        print("[INFO] loading grounding dino model ...")
        if processor is None:
            processor = AutoProcessor.from_pretrained(cfg.dino_model_id)
        self.processor = processor
        if grounding_model is None:
            grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.dino_model_id).to(cfg.device)
        self.grounding_model = grounding_model
        
        if not load_cache_success:
            with open(cache_path, "wb") as f:
                pickle.dump(dict(processor=processor, grounding_model=grounding_model, sam2_model=sam2_model, cfg=cfg), f)
        
        print("[INFO] GroundedSAM loaded.")
        os.chdir(cwd)
    
    def _get_annotated_frame(
        self, 
        image: Image.Image, 
        results: list, 
        masks: np.ndarray, 
        input_boxes: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
    ):
        CUSTOM_COLOR_MAP = [
            "#e6194b",
            "#3cb44b",
            "#ffe119",
            "#0082c8",
            "#f58231",
            "#911eb4",
            "#46f0f0",
            "#f032e6",
            "#d2f53c",
            "#fabebe",
            "#008080",
            "#e6beff",
            "#aa6e28",
            "#fffac8",
            "#800000",
            "#aaffc3",
        ]
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids, 
        )
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        labels = [str(i) for i in range(len(masks))]
        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        if point_coords is not None and point_labels is not None:
            draw = ImageDraw.Draw(annotated_frame)
            for (x, y), label in zip(point_coords, point_labels):
                radius = 5
                if label == 0:
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
                else:
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="green")

        return annotated_frame
    
    def inference_sam(
        self,
        image: Union[np.ndarray, Image.Image], 
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        xyxy: Optional[np.ndarray] = None,
    ):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        self.sam2_predictor.set_image(image)
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=xyxy,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks

    def inference(
        self, text: str, 
        image: Union[np.ndarray, Image.Image], 
        point_coords: Optional[np.ndarray] = None, 
        point_labels: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Args:
            img (np.ndarray or PIL Image): The input image to embed in RGB format. 
            The image should be in HWC format if np.ndarray, 
            or WHC format if PIL Image with pixel values in [0, 255].

        Return:
            masks (np.ndarray): (n, H, W)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        self.sam2_predictor.set_image(image)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.2,
            target_sizes=[image.size[::-1]]
        )
        input_boxes = results[0]["boxes"].cpu().numpy().copy() # [n, 4]
        if point_coords is not None and point_labels is not None:
            assert input_boxes.shape[0] > 0
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=np.array([point_coords] * len(input_boxes)) if point_coords is not None else None,
            point_labels=np.array([point_labels] * len(input_boxes)) if point_labels is not None else None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # annotate the image
        annotated_frame = self._get_annotated_frame(image, results, masks, input_boxes, point_coords, point_labels)
        info = dict(annotated_frame=annotated_frame)
        return masks, info


if __name__ == '__main__':
    img_dir = "tmp/video_track/src"
    mask_gt_dir = "tmp/video_track/mask_gt"
    output_dir = "tmp/video_track/output"
    info_dir = "tmp/video_track/info"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    sam = GroundedSAM()
    import tqdm

    for i in tqdm.tqdm(range(10, 167)):
        img = np.array(Image.open(os.path.join(img_dir, f"{str(i).zfill(4)}.png")))
        if sam.video_tracker.need_init_mask():
            mask = (np.load(os.path.join(mask_gt_dir, f"{str(i).zfill(4)}.npy")) == 1).astype(np.float32)
            sam.video_tracker.set_init_mask(mask)
            image_new = img.copy()
        else:
            mask, image_new = sam.video_tracker.segment_and_update(img)
        Image.fromarray((mask[:, :, None] * 127 + img).clip(0, 255).astype(np.uint8)).save(os.path.join(output_dir, f"{str(i).zfill(4)}.png"))
        Image.fromarray(image_new.astype(np.uint8)).save(os.path.join(info_dir, f"{str(i).zfill(4)}.png"))

