rm -rf ./gnu.build ./gnu.install 
mkdir -p ./gnu.build
mkdir -p ./gnu.install
cd gnu.build
rm 
cmake ../.. \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCMAKE_C_COMPILER=gcc \
	-DCMAKE_CXX_COMPILER=g++ \
	-DCMAKE_LINKER=g++ \
	-DCMAKE_INSTALL_PREFIX=`pwd`/../intel.dir.install \
	-DCMAKE_C_COMPILE_FLAGS="$(mpicxx -showme:compile || mpicxx -cxx= -compile_info)" \
	-DCMAKE_CXX_COMPILE_FLAGS="$(mpicxx -showme:compile || mpicxx -cxx= -compile_info)" \
	-DCMAKE_CXX_FLAGS="-O3 -Wall -Wextra -Werror" \
	-DCMAKE_EXE_LINKER_FLAGS="$(mpicxx -showme:link || mpicxx -cxx= -link_info)" \
	$* && \
make -j $(($(nproc)/2 + 1)) && \
make install && 
ctest -j $(nproc) --output-on-failure || exit 1

