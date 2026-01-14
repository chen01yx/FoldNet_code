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
    r_width: int = 300
    r_height: int = field(init=False)
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
        self.r_height = self.l_height
        self.button_num = 2
        self.button_width = self.r_width
        self.button_height = self.r_height // self.button_num


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
    def __init__(self, sam: GroundedSAM, img: Image.Image, cfg: AppCfg):
        self._cfg = cfg = copy.deepcopy(cfg)
        self._state = AppState()

        # sam setup
        self._sam = sam

        # GUI setup
        self._root = root = tk.Tk()
        root.geometry(f"{cfg.l_width + cfg.r_width}x{cfg.l_height}+{cfg.x}+{cfg.y}")
        root.title("Generate Mask")

        ## left
        left_frame = tk.Frame(root)
        left_frame.place(x=0, y=0, width=cfg.l_width, height=cfg.l_height)

        self._canvas = canvas = tk.Canvas(left_frame, bg="grey", cursor="cross")
        canvas.place(x=0, y=0, width=cfg.l_width, height=cfg.l_height)

        canvas.bind(button_map["left"], self._canvas_on_left)
        canvas.bind(button_map["right"], self._canvas_on_right)
        canvas.bind("<Motion>", self._canvas_on_motion)

        self._img = img.copy()

        ## right
        right_frame = tk.Frame(root)
        right_frame.place(x=cfg.l_width, y=0, width=cfg.r_width, height=cfg.l_height)

        self._button: dict[str, tk.Button] = {}
        for button_index, (text, command) in enumerate(zip(
            ["Mask", "Save"],
            [self._on_mask_button, self._on_save_button]
        )):
            button = tk.Button(right_frame, text=text, command=command)
            button.place(width=cfg.button_width, height=cfg.button_height, x=0, y=cfg.button_height * button_index)
            self._button[text] = button
        
    def _update_all_impl(self):
        cfg, state = self._cfg, self._state

        # image to label
        self._canvas.delete("all")
        mask_arr = np.array(self._img)
        if state.mask is not None:
            mask_arr[np.where(np.tile(state.mask[:, :, None], (1, 1, 3)))] = 255
        self._current_photo = ImageTk.PhotoImage(Image.fromarray(mask_arr).resize((cfg.l_width, cfg.l_height)))
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

    def run(self) -> np.ndarray:
        self._root.after(int(1000. / self._cfg.freq), self._update_all)
        self._root.mainloop()
        self._root.quit()
        self._root.destroy()
        return self._state.mask

    # caller function
    def _on_mask_button(self):
        cfg, state = self._cfg, self._state
        mask_all = np.zeros((cfg.image_height, cfg.image_width), dtype=np.uint8)
        for x1, y1, x2, y2 in state.xyxy:
            if (x1, y1, x2, y2) in state.mask_cache:
                mask = state.mask_cache[(x1, y1, x2, y2)]
            else:
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                mask = self._sam.inference_sam(
                    image=self._img, 
                    xyxy=np.array([x1, y1, x2, y2])
                )[0]
                state.mask_cache[(x1, y1, x2, y2)] = mask
            mask_all = np.logical_or(mask_all, mask)
        state.mask = mask_all

    def _on_save_button(self):
        if self._state.mask is not None:
            self._root.quit()
        else:
            print("No mask generated yet.")

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
    cfg = AppCfg()
    app = App(
        # sam=None, 
        sam=GroundedSAM(),
        img=Image.open("asset/fig/keypoint_example/tshirt.jpg").resize((640, 480)),
        cfg=cfg,
    )
    app.run()
