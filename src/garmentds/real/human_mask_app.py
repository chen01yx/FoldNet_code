import os, json, sys, copy
from typing import Optional
from dataclasses import dataclass, field

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

from garmentds.real.sam_utils import SAMCfg, GroundedSAM

if sys.platform == "darwin":
    button_map = {
        "left": "<Button-1>",
        "right": "<Button-2>",
    }
elif sys.platform == "linux":
    button_map = {
        "left": "<Button-1>",
        "right": "<Button-3>",
    }
else:
    raise NotImplementedError(sys.platform)

@dataclass
class AppCfg:
    """Configuration for the app layout."""
    image_width: int = 640
    image_height: int = 480
    image_scale: float = 1.5
    r_width: int = 600
    r_height: int = 600
    x: int = 100
    y: int = 100
    l_width: int = field(init=False)
    l_height: int = field(init=False)
    button_width: int = field(init=False)
    button_height: int = field(init=False)
    button_num: int = field(init=False)

    freq: int = 30

    def __post_init__(self):
        self.l_width = int(self.image_width * self.image_scale)
        self.l_height = int(self.image_height * self.image_scale)
        self.button_num = 5
        self.button_width = self.r_width // self.button_num
        self.button_height = self.l_height - self.r_height


@dataclass
class AppState:
    """Internal state of the app."""
    current_index: Optional[int] = None
    mask: Optional[np.ndarray] = None
    xyxy: list[tuple[int, int, int, int]] = field(default_factory=list)
    cursor_xy: Optional[tuple[int, int]] = None
    prev_xy: Optional[tuple[int, int]] = None
    mask_cache: dict[tuple[int, int, int, int], np.ndarray] = field(default_factory=dict)

    def reset_state(self):
        self.prev_xy = self.cursor_xy = None
        self.xyxy = []
        self.mask_cache = {}

    def reset_labels(self):
        self.mask = None

    def write_to_npy(self, npy_path: str):
        if self.mask is not None:
            np.save(npy_path, self.mask)
    
    def load_from_npy(self, npy_path: str):
        self.reset_state()
        if os.path.exists(npy_path):
            self.mask = np.load(npy_path)
        else:
            self.reset_labels()


class App:
    def __init__(self, img_dir: str, app_cfg: AppCfg, current_index: int = 0):
        self._img_dir = img_dir
        self._cfg = cfg = copy.deepcopy(app_cfg)
        self._images = self._load_images()
        assert len(self._images)

        self._state = AppState()

        # sam setup
        self._sam = GroundedSAM()

        # GUI setup
        self._root = root = tk.Tk()
        root.geometry(f"{app_cfg.l_width + app_cfg.r_width}x{app_cfg.l_height}+{app_cfg.x}+{app_cfg.y}")
        root.title("Generate Mask")

        ## left
        left_frame = tk.Frame(root)
        left_frame.place(x=0, y=0, width=app_cfg.l_width, height=app_cfg.l_height)

        self._canvas = canvas = tk.Canvas(left_frame, bg="grey", cursor="cross")
        canvas.place(x=0, y=0, width=app_cfg.l_width, height=app_cfg.l_height)

        canvas.bind(button_map["left"], self._canvas_on_left)
        canvas.bind(button_map["right"], self._canvas_on_right)
        canvas.bind("<Motion>", self._canvas_on_motion)

        ## right
        right_frame = tk.Frame(root)
        right_frame.place(x=app_cfg.l_width, y=0, width=app_cfg.r_width, height=app_cfg.l_height)

        button_frame = tk.Frame(right_frame)
        button_frame.place(x=0, y=0, width=app_cfg.r_width, height=app_cfg.button_height)
        self._button: dict[str, tk.Button] = {}
        for button_index, (text, command) in enumerate(zip(
            ["Mask", "Save", "Next", "Previous", "Quit"],
            [self._on_mask_button, self._on_save_button, self._on_next_button, self._on_previous_button, self._root.quit]
        )):
            button = tk.Button(button_frame, text=text, command=command)
            button.place(width=cfg.button_width, height=cfg.button_height, x=cfg.button_width * button_index, y=0)
            self._button[text] = button

        example_frame = tk.Frame(right_frame)
        example_frame.place(x=0, y=cfg.button_height, width=cfg.r_width, height=cfg.r_height)
        self._canvas_example = canvas_example = tk.Canvas(example_frame)
        canvas_example.place(x=0, y=0, width=app_cfg.r_width, height=app_cfg.r_height)

        self._update_current_index(current_index)

    def _load_images(self):
        return sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(self._img_dir)
            for file in files if file.endswith(".png")
        ])
    
    def _get_current_mask_path(self):
        path: str = self._images[self._state.current_index]
        return path.replace("color.png", "mask.npy")
    
    def _get_current_json_path(self):
        path: str = self._images[self._state.current_index]
        return path + ".json"
    
    def _get_current_image_path(self):
        return self._images[self._state.current_index]

    def _update_all_impl(self):
        cfg, state = self._cfg, self._state
        current_path = self._get_current_image_path()

        # example image
        self._canvas_example.delete("all")
        if state.mask is not None:
            mask_arr = np.array(Image.open(current_path))
            mask_arr[np.where(np.tile(state.mask[:, :, None], (1, 1, 3)))] = 255
        else:
            mask_arr = np.zeros((cfg.l_height, cfg.l_width), dtype=np.uint8)
        self._example_image = Image.fromarray(mask_arr).resize((cfg.r_width, cfg.r_height))
        self._example_photo = ImageTk.PhotoImage(self._example_image)
        self._canvas_example.create_image(0, 0, anchor="nw", image=self._example_photo)
        self._canvas_example.create_text(5, 5, text=current_path, anchor="nw", fill="yellow")

        # image to label
        self._canvas.delete("all")
        self._current_image = Image.open(current_path).resize((cfg.l_width, cfg.l_height))
        self._current_photo = ImageTk.PhotoImage(self._current_image)
        self._canvas.create_image(0, 0, anchor="nw", image=self._current_photo)

        # bounding box
        s = cfg.image_scale
        for x1, y1, x2, y2 in state.xyxy:
            self._canvas.create_rectangle(x1 * s, y1 * s, x2 * s, y2 * s, outline="red", width=2)
        if state.prev_xy is not None and state.cursor_xy is not None:
            x1, y1 = state.prev_xy
            x2, y2 = state.cursor_xy
            self._canvas.create_rectangle(x1 * s, y1 * s, x2, y2, outline="red", width=2)
        
    def _update_all(self):
        self._update_all_impl()
        self._root.after(int(1000. / self._cfg.freq), self._update_all)

    def run(self):
        self._root.after(int(1000. / self._cfg.freq), self._update_all)
        self._root.mainloop()
    
    def _update_current_index(self, new_index: int):
        state = self._state
        if new_index < 0 or new_index >= len(self._images):
            raise ValueError(f"Invalid index: {new_index} not in [0, {len(self._images)}]")
        if new_index != state.current_index:
            state.current_index = new_index
            state.load_from_npy(self._get_current_mask_path())
            if os.path.exists(self._get_current_json_path()):
                with open(self._get_current_json_path(), "r") as f:
                    data = json.load(f)
                xys = np.array([xy for xy in data["keypoints"] if xy is not None])
                xy0, xy1 = np.min(xys, axis=0), np.max(xys, axis=0)
                xyc, xyd = (xy0 + xy1) / 2, (xy1 - xy0) / 2
                state.xyxy.append((*(xyc - xyd * 1.1), *(xyc + xyd * 1.1))) # auto load keypoints bounding box
    
    def _next_image(self):
        state = self._state
        if state.current_index < len(self._images) - 1:
            self._update_current_index(state.current_index + 1)
    
    def _previous_image(self):
        state = self._state
        if state.current_index > 0:
            self._update_current_index(state.current_index - 1)
    
    # caller function
    def _on_mask_button(self):
        cfg, state = self._cfg, self._state
        if state.xyxy:
            mask_all = np.zeros((cfg.image_height, cfg.image_width), dtype=np.uint8)
            for x1, y1, x2, y2 in state.xyxy:
                if (x1, y1, x2, y2) in state.mask_cache:
                    mask = state.mask_cache[(x1, y1, x2, y2)]
                else:
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    mask = self._sam.inference_sam(
                        image=Image.open(self._get_current_image_path()), 
                        xyxy=np.array([x1, y1, x2, y2])
                    )[0]
                    state.mask_cache[(x1, y1, x2, y2)] = mask
                mask_all = np.logical_or(mask_all, mask)
            state.mask = mask_all

    def _on_next_button(self):
        self._next_image()

    def _on_previous_button(self):
        self._previous_image()

    def _on_save_button(self):
        self._state.write_to_npy(self._get_current_mask_path())
        self._next_image()
    
    def _on_quit_button(self):
        self._root.quit()

    def _canvas_on_left(self, event: tk.Event):
        cfg, state = self._cfg, self._state
        s = cfg.image_scale
        if state.prev_xy is None:
            state.prev_xy = (event.x / s, event.y / s)
        else:
            x, y = state.prev_xy
            state.xyxy.append((x, y, event.x / s, event.y / s))
            state.prev_xy = None

    def _canvas_on_right(self, event: tk.Event):
        state = self._state
        if state.prev_xy is not None:
            state.prev_xy = None
        else:
            if state.xyxy:
                state.xyxy.pop()
    
    def _canvas_on_motion(self, event: tk.Event):
        state = self._state
        state.cursor_xy = (event.x, event.y)


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="tmp_image")
    args = parser.parse_args()

    app_cfg = AppCfg()
    app = App(
        img_dir=args.img_dir, 
        app_cfg=app_cfg
    )
    app.run()
