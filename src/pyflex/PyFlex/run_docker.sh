sudo docker run \
 -v root-directory-of-this-repository/src/pyflex/PyFlex:/workspace/PyFleX \
 -v your-conda-env-path:/workspace/anaconda \
 -it yunzhuli/pyflex_16_04_cuda_9_1:latest
