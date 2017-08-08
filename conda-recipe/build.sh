mkdir build
cd build

export TBB_LINK=${PREFIX}/lib
export EMBREE_LINK=${PREFIX}/lib

if [ "$(uname)" == "Darwin" ]; then

cmake ../ \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.8 \
      -DCMAKE_CXX_FLAGS="-mmacosx-version-min=10.8 -stdlib=libc++" \
      -DCMAKE_INSTALL_PREFIX=${SP_DIR} \
      -DPYTHON_EXECUTABLE=${PYTHON} \
      -DENABLE_CUDA=off \
      -DENABLE_OPTIX=off

make install -j ${CPU_COUNT}

else
# Linux build
CC=${PREFIX}/bin/gcc
CXX=${PREFIX}/bin/g++

cmake ../ \
      -DCMAKE_INSTALL_PREFIX=${SP_DIR} \
      -DPYTHON_EXECUTABLE=${PYTHON} \
      -DENABLE_CUDA=off \
      -DENABLE_OPTIX=off

make install -j ${CPU_COUNT}
fi
