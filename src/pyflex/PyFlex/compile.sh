if [ -d "bindings/build" ]; then
    rm -rf bindings/build
fi

cd bindings/

mkdir build; cd build; cmake -DCMAKE_PREFIX_PATH=/workspace/anaconda/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 ..; make -j

cd ../..