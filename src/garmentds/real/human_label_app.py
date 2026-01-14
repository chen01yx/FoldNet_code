import os, json, sys, copy
from typing import Optional
from dataclasses import dataclass, field

import tkinter as tk
from PIL import Image, ImageTk

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

    keypoint_num: int = 14
    keypoint_radius: int = 5
    freq: int = 30

    def __post_init__(self):
        self.l_width = int(self.image_width * self.image_scale)
        self.l_height = int(self.image_height * self.image_scale)
        self.button_num = 6
        self.button_width = self.r_width // self.button_num
        self.button_height = self.l_height - self.r_height


@dataclass
class AppState:
    """Internal state of the app."""
    cfg: AppCfg
    cursor_xy: Optional[tuple[int, int]] = None
    current_index: Optional[int] = None
    is_faceup: bool = True
    keypoints: list[Optional[tuple[int, int]]] = field(default_factory=list)

    def reset_labels(self):
        self.keypoints = []
    
    def write_to_json(self, json_path: str):
        if len(self.keypoints) != self.cfg.keypoint_num:
            print(f"[WARN] keypoint number not match, expected {self.cfg.keypoint_num}, got {len(self.keypoints)}. auto padding with None")
            self.keypoints = self.keypoints + [None] * (self.cfg.keypoint_num - len(self.keypoints))
        data = {"is_faceup": self.is_faceup, "keypoints": self.keypoints}
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    def load_from_json(self, json_path: str):
        """try to load labels from json file, if not exist, reset labels"""
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            if len(data["keypoints"]) != self.cfg.keypoint_num:
                print(f"[WARN] keypoint number not match, expected {self.cfg.keypoint_num}, got {len(data['keypoints'])}")
                self.reset_labels()
            else:
                self.is_faceup = data["is_faceup"]
                self.keypoints = data["keypoints"]
        else:
            self.reset_labels()


class App:
    def __init__(self, img_dir: str, category: str, app_cfg: AppCfg, current_index: int = 0):
        self._img_dir = img_dir
        self._example_path = f"asset/fig/keypoint_example/{category}.jpg"
        self._cfg = cfg = copy.deepcopy(app_cfg)
        self._images = self._load_images()
        assert len(self._images)

        self._state = AppState(cfg=cfg)

        # GUI setup
        self._root = root = tk.Tk()
        root.geometry(f"{app_cfg.l_width + app_cfg.r_width}x{app_cfg.l_height}+{app_cfg.x}+{app_cfg.y}")
        root.title("Keypoint Labeling")

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
            ["Flip", "Skip", "Save", "Next", "Previous", "Quit"],
            [self._on_flip_button, self._on_skip_button, self._on_save_button, self._on_next_button, self._on_previous_button, self._root.quit]
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
    
    def _get_current_json_path(self):
        return self._images[self._state.current_index] + ".json"
    
    def _get_current_image_path(self):
        return self._images[self._state.current_index]

    def _update_all_impl(self):
        cfg, state = self._cfg, self._state
        current_path = self._get_current_image_path()

        # example image
        self._canvas_example.delete("all")
        self._example_image = Image.open(self._example_path).resize((cfg.r_width, cfg.r_height))
        self._example_photo = ImageTk.PhotoImage(self._example_image)
        self._canvas_example.create_image(0, 0, anchor="nw", image=self._example_photo)
        self._canvas_example.create_text(5, 5, text=current_path, anchor="nw", fill="yellow")

        # image to label
        self._canvas.delete("all")
        self._current_image = Image.open(current_path).resize((cfg.l_width, cfg.l_height))
        self._current_photo = ImageTk.PhotoImage(self._current_image)
        self._canvas.create_image(0, 0, anchor="nw", image=self._current_photo)

        # draw keypoints
        s, r = cfg.image_scale, cfg.keypoint_radius
        for i, xy in enumerate(state.keypoints):
            if xy is None:
                continue
            x, y = xy
            x, y = x * s, y * s
            self._canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")
            self._canvas.create_text(x + r, y - r, text=str(i + 1), fill="yellow")
        if state.cursor_xy is not None:
            x, y = state.cursor_xy
            self._canvas.create_oval(x - r, y - r, x + r, y + r, fill="red")
        
        # button name
        self._button["Flip"].config(text=("current:\n\nfaceup\n\nclick to flip" if state.is_faceup else "current:\n\nfacedown\n\nclick to flip"))
        self._button["Skip"].config(text="Skip:\n" + " ".join([str(i + 1) for i, k in enumerate(state.keypoints) if k is None]))
        
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
            state.load_from_json(self._get_current_json_path())
    
    def _next_image(self):
        state = self._state
        if state.current_index < len(self._images) - 1:
            self._update_current_index(state.current_index + 1)
    
    def _previous_image(self):
        state = self._state
        if state.current_index > 0:
            self._update_current_index(state.current_index - 1)
    
    # caller function
    def _on_flip_button(self):
        self._state.is_faceup = not self._state.is_faceup
    
    def _on_skip_button(self):
        self._state.keypoints.append(None)

    def _on_next_button(self):
        self._next_image()

    def _on_previous_button(self):
        self._previous_image()

    def _on_save_button(self):
        self._state.write_to_json(self._get_current_json_path())
        self._next_image()
    
    def _on_quit_button(self):
        self._root.quit()

    def _canvas_on_left(self, event: tk.Event):
        cfg, state = self._cfg, self._state
        state.keypoints.append((event.x / cfg.image_scale, event.y / cfg.image_scale))

    def _canvas_on_right(self, event: tk.Event):
        state = self._state
        if state.keypoints:
            state.keypoints.pop()
    
    def _canvas_on_motion(self, event: tk.Event):
        state = self._state
        state.cursor_xy = (event.x, event.y)


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="tmp_image")
    parser.add_argument("--category", type=str, default="tshirt")
    args = parser.parse_args()

    app_cfg = AppCfg(keypoint_num=dict(tshirt=14, trousers=8, hooded_close=14, vest_close=10)[args.category])
    app = App(
        img_dir=args.img_dir, 
        category=args.category, 
        app_cfg=app_cfg
    )
    app.run()
