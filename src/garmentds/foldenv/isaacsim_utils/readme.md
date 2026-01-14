for i in {1..1}
do
    docker run --name isaac-sim-450-$i --entrypoint bash -it --runtime=nvidia --gpus "device=$i" --network=host \
        -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
        -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
        -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
        -v ~/docker/isaac-sim/documents:/root/Documents:rw \
        -v .:/root/garmentds:rw \
        nvcr.io/nvidia/isaac-sim:4.5.0
done

for i in {1..1}
do
    docker start isaac-sim-450-$i && docker exec isaac-sim-450-$i /isaac-sim/python.sh -m pip install torch trimesh tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
done

for i in {1..1}
do
    docker stop isaac-sim-450-$i && docker rm isaac-sim-450-$i
done