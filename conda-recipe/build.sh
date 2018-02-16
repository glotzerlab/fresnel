mkdir -p build_conda
cd build_conda
rm -f CMakeCache.txt

export TBB_LINK=${PREFIX}/lib
export EMBREE_LINK=${PREFIX}/lib

if [ "$(uname)" == "Darwin" ]; then

cmake ../ \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.8 \
      -DCMAKE_CXX_FLAGS="-mmacosx-version-min=10.8 -stdlib=libc++ -march=core2" \
      -DCMAKE_C_FLAGS="-march=core2" \
      -DCMAKE_INSTALL_PREFIX=${SP_DIR} \
      -DPYTHON_EXECUTABLE=${PYTHON} \
      -DENABLE_CUDA=off \
      -DENABLE_OPTIX=off

make install -j 2

else
# Linux build
cmake ../ \
      -DCMAKE_INSTALL_PREFIX=${SP_DIR} \
      -DPYTHON_EXECUTABLE=${PYTHON} \
      -DCMAKE_CXX_FLAGS="-march=core2" \
      -DCMAKE_C_FLAGS="-march=core2" \
      -DENABLE_CUDA=off \
      -DENABLE_OPTIX=off

make install -j 2
fi
