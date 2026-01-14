1. Conda environment
    ```
    conda create -n FoldNet python=3.9.20
    pip install -e . --use-pep517
    sh setup.sh
    sudo apt install ffmpeg
    ```
2. Blender
- Download [blender](https://www.blender.org/download/release/Blender4.2/blender-4.2.9-linux-x64.tar.xz)
- Append the following code to ~/.zshrc or ~/.bashrc
    ```
    export BLENDER_PATH="/your/path/to/blender-4.2.9-linux-x64"
    export PATH="$BLENDER_PATH:$PATH"
    alias blender_python="$BLENDER_PATH/4.2/python/bin/python3.11"
    ```
- Install packages for blender's python
    ```
    cd FoldNet_code
    cd external/batch_urdf && blender_python -m pip install -e . && cd ../..
    cd external/bpycv && blender_python -m pip install -e . && cd ../..
    blender_python -m pip install psutil
    apt install libsm6
    ```
- Test blender
    ```
    cd FoldNet_code
    blender src/garmentds/foldenv/scene.blend --python src/garmentds/foldenv/blender_script.py --background -- --run_test
    ```
- Download blender asset
    ```
    blender src/garmentds/foldenv/scene.blend --python src/garmentds/foldenv/blender_script.py --background -- --run_init
    ```
3. PyFlex
We provide compiled pyflex for python 3.9. Please refer to src/pyflex/libs/how_to_run_without_docker.md for more details.
- Append the following code to ~/.zshrc or ~/.bashrc
    ```
    PYFLEX_PATH=/your/path/to/FoldNet_code/src/pyflex
    export PYTHONPATH="$PYFLEX_PATH/libs":$PYTHONPATH
    export LD_LIBRARY_PATH="$PYFLEX_PATH/libs":$LD_LIBRARY_PATH
    ```
- Install
    ```
    sudo apt install libasound2
    sudo apt install libegl1
    ```
- Test
    ```
    import pyflex
    pyflex.init(True, False, 0, 0, 0)
    ```
4. Full test
    ```
    CUDA_VISIBLE_DEVICES=0 python run/fold_multi_cat.py env.cloth_obj_path=asset/garment_example/mesh.obj env.render_process_num=1 '+env.init_cloth_vel_range=[1.,2.]'
    ```