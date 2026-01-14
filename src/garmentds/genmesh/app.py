import sys
from dataclasses import dataclass, asdict
from typing import Optional, Literal
import json
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext

from PIL import ImageTk
import numpy as np

from .base_cls import GarmentTemplateABC

import omegaconf


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


class TwoInputDialog(simpledialog.Dialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_result = (None, None)

    def body(self, master):
        tk.Label(master, text="x:").grid(row=0)
        tk.Label(master, text="y:").grid(row=1)

        self.entry1 = tk.Entry(master)
        self.entry2 = tk.Entry(master)

        self.entry1.grid(row=0, column=1)
        self.entry2.grid(row=1, column=1)

        return self.entry1

    def apply(self):
        try:
            self.result = (float(self.entry1.get()), float(self.entry2.get()))
        except Exception as e:
            print(e)
            self.result = self.default_result


@dataclass
class AppCfg:
    # size
    l_width: int = 960
    l_height: int = 960
    r_width: int = 480
    r_u_height: int = 80
    x: int = 200
    y: int = 50

    # frequency
    freq: int = 30

    # background grid
    grid_x_num: int = 8
    grid_x_range: tuple[float, float] = (-0.8, 0.8)
    grid_y_num: int = 8
    grid_y_range: tuple[float, float] = (-0.8, 0.8)
    text_offset: int = 20

    # control
    select_threshold: int = 10 # in pixel


@dataclass
class AppState:
    current_pos_ij: Optional[tuple[int, int]] = None
    right_button_press: bool = False
    left_button_press: bool = False
    left_button_press_ij: Optional[tuple[int, int]] = None
    selected_pos_xy: Optional[tuple[float, float]] = None
    selected_keypoint: Optional[str] = None


class RedirectText(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class DesignApp:
    def __init__(self, garment: GarmentTemplateABC) -> None:
        # param
        self._app_cfg = AppCfg()

        # state
        self._state = AppState()

        # object
        self._garment = garment
    
    ### helper function
    def _xy2ij(self, x: float, y: float):
        app_cfg, state = self._app_cfg, self._state
        
        i = max(0, min(app_cfg.l_height, int(
            app_cfg.l_height * (-y - app_cfg.grid_y_range[0]) / 
            (app_cfg.grid_y_range[1] - app_cfg.grid_y_range[0])
        )))
        j = max(0, min(app_cfg.l_width, int(
            app_cfg.l_width * (x - app_cfg.grid_x_range[0]) / 
            (app_cfg.grid_x_range[1] - app_cfg.grid_x_range[0])
        )))
        return (i, j)
    
    def _ij2xy(self, i: int, j: int):
        app_cfg, state = self._app_cfg, self._state
        x = (
            app_cfg.grid_x_range[0] + j / app_cfg.l_width * 
            (app_cfg.grid_x_range[1] - app_cfg.grid_x_range[0])
        )
        y = -(
            app_cfg.grid_y_range[0] + i / app_cfg.l_height * 
            (app_cfg.grid_y_range[1] - app_cfg.grid_y_range[0])
        )
        return (x, y)
    
    def _nearby_keypoint(self, i: int, j: int, threshold: int):
        for k, v in self._garment.asdict_keypoints().items():
            i1, j1 = self._xy2ij(*v)
            if (i - i1) ** 2 + (j - j1) ** 2 < threshold ** 2:
                return k, (i1, j1)
        
        return None
    
    ### modifier api
    def _modify_state(self, name: str, value):
        setattr(self._state, name, value)
    
    def _modify_garment(self, mode: Literal["update", "symmetry"]="update", name=None, value=None, put_in_stack=True):
        if put_in_stack:
            print(f"[INFO] modify garment {mode} {name} {value}")
        if mode == "update":
            self._garment.update_keypoints(name, value, put_in_stack)
        elif mode == "symmetry":
            self._garment.symmetry(put_in_stack)
        else:
            raise NotImplementedError(mode)
    
    ### worker function
    def _get_user_input_xy(self):
        dialog = TwoInputDialog(self._root, "input target xy position")
        result = dialog.result
        if result is None:
            result = dialog.default_result
        return result
    
    def _update_canvas_impl(self):
        app_cfg, state = self._app_cfg, self._state

        self._canvas.delete("all")

        # draw grid
        text_offset = app_cfg.text_offset
        num = app_cfg.grid_x_num
        for i in range(num * 2 - 1):
            x0 = x1 = self._canvas.winfo_width() / (num * 2) * (i + 1)
            y0, y1 = 0, self._canvas.winfo_height()
            self._canvas.create_line(x0, y0, x1, y1)
            self._canvas.create_text(x0, text_offset, text=f"{(i - num + 1) * 0.1:.1f}", fill="white")
        self._canvas.create_text(self._canvas.winfo_width() - text_offset, text_offset, text="(x)", fill="white")

        num = app_cfg.grid_y_num
        for i in range(num * 2 - 1):
            y0 = y1 = self._canvas.winfo_height() / (num * 2) * (i + 1)
            x0, x1 = 0, self._canvas.winfo_width()
            self._canvas.create_line(x0, y0, x1, y1)
            self._canvas.create_text(text_offset, y0, text=f"{(i - num + 1) * -0.1:.1f}", fill="white")
        self._canvas.create_text(text_offset, self._canvas.winfo_height() - text_offset, text="(y)", fill="white")
        
        # draw garment
        img = self._garment.draw(width=app_cfg.l_width, height=app_cfg.l_height, xy2ij=self._xy2ij)
        self._photo = ImageTk.PhotoImage(img) # use 'self.' to keep reference
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # draw annotation
        if state.current_pos_ij is not None:
            selected_keypoint = self._nearby_keypoint(state.current_pos_ij[0], state.current_pos_ij[1], app_cfg.select_threshold)
            if selected_keypoint is not None:
                _, (i, j) = selected_keypoint
                radius = 10
                self._canvas.create_oval(j - radius, i - radius, j + radius, i + radius, fill="red", width=2)

    def _update_object_use_state(self):
        app_cfg, state = self._app_cfg, self._state
        if state.right_button_press:
            if state.selected_pos_xy is not None:
                # right button to cancel movement
                self._modify_garment(mode="update", name=state.selected_keypoint, value=state.selected_pos_xy, put_in_stack=False)
                self._modify_state("selected_pos_xy", None)
                self._modify_state("selected_keypoint", None)
            else:
                # direct move keypoint to xy
                selected_keypoint = self._nearby_keypoint(state.current_pos_ij[0], state.current_pos_ij[1], app_cfg.select_threshold)
                if selected_keypoint is not None:
                    keypoint_name = selected_keypoint[0]
                    x, y = self._get_user_input_xy()
                    if x is not None and y is not None:
                        self._modify_garment(mode="update", name=keypoint_name, value=(x, y))
            self._modify_state("right_button_press", False) # state.right_button_press is used, restore it to False
        elif state.left_button_press:
            if state.selected_pos_xy is not None:
                # left button again to confirm update
                self._modify_garment(mode="update", name=state.selected_keypoint, value=self._ij2xy(*state.left_button_press_ij))
                self._modify_state("selected_pos_xy", None)
                self._modify_state("selected_keypoint", None)
            else:
                # select the first point
                selected_keypoint = self._nearby_keypoint(*state.left_button_press_ij, app_cfg.select_threshold)
                if selected_keypoint is not None:
                    keypoint_name = selected_keypoint[0]
                    self._modify_state("selected_pos_xy", self._garment.access_keypoints(keypoint_name))
                    self._modify_state("selected_keypoint", keypoint_name)
            self._modify_state("left_button_press", False) # state.left_button_press is used, restore it to False
        else:
            # just moving around
            if state.selected_pos_xy is not None:
                i, j = state.current_pos_ij
                self._modify_garment(mode="update", name=state.selected_keypoint, value=self._ij2xy(i, j), put_in_stack=False)

    ### caller
    def _update_canvas(self):
        self._update_canvas_impl()
        self._root.after(int(1000. / self._app_cfg.freq), self._update_canvas)
    
    def _on_motion(self, event: tk.Event):
        j, i = event.x, event.y
        self._modify_state("current_pos_ij", (i, j))
        self._update_object_use_state()
    
    def _on_button_right(self, event: tk.Event):
        self._modify_state("right_button_press", True)
        self._update_object_use_state()
    
    def _on_button_left(self, event: tk.Event):
        self._modify_state("left_button_press", True)
        self._modify_state("left_button_press_ij", (event.y, event.x))
        self._update_object_use_state()

    def _on_symmetry_button(self):
        self._modify_garment(mode="symmetry")
        self._update_object_use_state()
    
    def _on_triangulation_button(self):
        self._garment.triangulation()
    
    def _on_quick_save_mesh_button(self):
        self._garment.quick_export("mesh.obj")
    
    def _on_quick_save_cfg_button(self):
        raise DeprecationWarning("use save mesh")
        omegaconf.OmegaConf.save(omegaconf.DictConfig(dict(cfg=asdict(self._garment.cfg))), "cfg.yaml")
        print("[INFO] cfg saved !")
    
    def _on_ctrl_z(self, event):
        self._garment.ctrl_z()
    
    def _on_ctrl_shift_z(self, event):
        self._garment.ctrl_shift_z()

    def _init(self):
        app_cfg, state = self._app_cfg, self._state

        # root
        self._root = root = tk.Tk()
        root.title("design UI")
        root.geometry(f"{app_cfg.l_width + app_cfg.r_width}x{app_cfg.l_height}+{app_cfg.x}+{app_cfg.y}")
        root.bind('<Control-z>', self._on_ctrl_z)
        root.bind('<Control-Z>', self._on_ctrl_shift_z)

        # left
        left_frame = tk.Frame(root)
        left_frame.place(x=0, y=0, width=app_cfg.l_width, height=app_cfg.l_height)
        
        ## canvas
        self._canvas = canvas = tk.Canvas(left_frame, bg="grey", cursor="cross")
        canvas.pack(side=tk.LEFT)
        canvas.bind("<Motion>", self._on_motion)
        canvas.bind(button_map["right"], self._on_button_right)
        canvas.bind(button_map["left"], self._on_button_left)
        canvas.place(x=0, y=0, width=app_cfg.l_width, height=app_cfg.l_height)

        # right
        right_frame = tk.Frame(root)
        right_frame.place(x=app_cfg.l_width, y=0, width=app_cfg.r_width, height=app_cfg.l_height)

        ## button
        button_frame = tk.Frame(right_frame)
        button_frame.place(x=0, y=0, width=app_cfg.r_width, height=app_cfg.r_u_height)

        button1 = tk.Button(button_frame, text="symmetry", command=self._on_symmetry_button)
        button1.place(x=0, y=0, width=app_cfg.r_width // 4, height=app_cfg.r_u_height)
        button2 = tk.Button(button_frame, text="triangulation", command=self._on_triangulation_button)
        button2.place(x=app_cfg.r_width // 4, y=0, width=app_cfg.r_width // 4, height=app_cfg.r_u_height)
        button3 = tk.Button(button_frame, text="quick_save_mesh", command=self._on_quick_save_mesh_button)
        button3.place(x=app_cfg.r_width // 4 * 2, y=0, width=app_cfg.r_width // 4, height=app_cfg.r_u_height)
        button4 = tk.Button(button_frame, text="quick_save_cfg", command=self._on_quick_save_cfg_button)
        button4.place(x=app_cfg.r_width // 4 * 3, y=0, width=app_cfg.r_width // 4, height=app_cfg.r_u_height)

        ## redirect stdout
        text_area = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=20, width=40)
        text_area.place(x=0, y=app_cfg.r_u_height, width=app_cfg.r_width, height=app_cfg.l_height - app_cfg.r_u_height)
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        redirect = RedirectText(text_area)
        sys.stdout = sys.stderr = redirect

    def _mainloop(self):
        self._root.after(int(1000. / self._app_cfg.freq), self._update_canvas)
        self._root.mainloop()
    
    def _close(self):
        pass

    def run(self):
        try:
            self._init()
            self._mainloop()
        finally:
            self._close()


if __name__ == "__main__":
    app = DesignApp()
    app.run()