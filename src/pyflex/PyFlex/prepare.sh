export PATH="/workspace/anaconda/bin:$PATH"
cd /workspace/PyFleX
export PYFLEXROOT=${PWD}
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

apt-get update
apt-get install libsdl2-dev
apt-get install libgl1-mesa-dev

rm -rf /workspace/PyFleX/bindings/build