rm -rf ./nvcc.build ./nvcc.install 
mkdir ./nvcc.build
mkdir ./nvcc.install
cd nvcc.build
rm -f CMakeCache.txt

CUDACXX=/usr/local/cuda/bin/nvcc \
CXXFLAGS="-O3 -Wall -Wextra -Wfatal-errors" \
CUDAFLAGS="$(for x in `mpic++ --showme:incdirs`; do echo -n -I$x" " ; done) -D_DISABLE_CUDA_SLOW -O3 --expt-relaxed-constexpr --expt-extended-lambda --Werror=cross-execution-space-call --compiler-options -Ofast,-std=c++14,-Wall,-Wextra,-Wfatal-errors" \
LIBS="$(for x in `mpic++ --showme:libs`; do echo -n -l$x" " ; done)" \
LDFLAGS="$(for x in `mpic++ --showme:libdirs`; do echo -n -L$x" " ; done)" \
../../configure --enable-cuda --prefix=`pwd`/../nvcc.install $* \
  && make -j $(($(nproc) + 1)) \
  && make install \
  && cd src && ctest -j $(nproc) --output-on-failure || exit 1

