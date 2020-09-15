rm -rf ./gnu.build ./gnu.install 
mkdir -p ./gnu.build
mkdir -p ./gnu.install
cd gnu.build

OMPIC_CXX=g++ OMPI_CC=gcc CC=mpicc CXX=mpic++ CXXFLAGS="-O3 -Wall -Wextra -Werror" \
	../../configure --prefix=`pwd`/../gnu.install $* \
  && make -j $(($(nproc) + 1)) \
  && make install \
  && cd src && ctest -j $(nproc) --output-on-failure || exit 1

