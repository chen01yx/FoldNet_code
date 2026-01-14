### compile pyflex
1. run docker 
```
sudo docker run \
-v ./src/pyflex/PyFlex:/workspace/PyFleX \
-v ~/app/miniconda3/envs/garmentds:/workspace/anaconda \
-it yunzhuli/pyflex_16_04_cuda_9_1:latest
```
2. in docker, `cd /workspace/PyFleX && source prepare.sh`
3. compile `cd /workspace/PyFleX && sh compile.sh`