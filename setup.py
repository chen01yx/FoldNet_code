import setuptools
import os
import subprocess

if __name__ == '__main__':
    try:
        setuptools.setup(
            name='garmentds',
            version='0.0',
            packages=setuptools.find_packages('src'),
            package_dir={'':'src'},
            python_requires='>=3.9',
            install_requires=[
                "numpy<2.0",
                "torch",
                "trimesh",
                "taichi==1.6.0",
                "tqdm",
                "opencv-python",
                "omegaconf",
                "matplotlib",
                "scipy",
                "open3d",
                "cgal",
                "shapely",
                "hydra-core",
                "pyrealsense2",
                "sapien",
                "lightning",
                "pybind11",
                "einops",
                "diffusers", 
                "psutil", 
                "torchvision", 
                "tensorboard", 
                "diffusers",
                "accelerate",
                "openai",
                "transformers", 
                "sentencepiece", 
            ]
        )
    except:
        raise RuntimeError("An error occured during setup.")