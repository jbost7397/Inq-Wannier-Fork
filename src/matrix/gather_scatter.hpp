/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MATRIX__GATHER_SCATTER
#define INQ__MATRIX__GATHER_SCATTER

// Copyright (C) 2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <matrix/distributed.hpp>

#include <gpu/array.hpp>

namespace inq {
namespace matrix {

template <class DistributedType>
auto gather(DistributedType const & matrix) -> gpu::array<typename DistributedType::element_type, 2> {

	CALI_CXX_MARK_SCOPE("matrix::gather");
	
  using type = typename DistributedType::element_type;
  
  auto mpi_type = boost::mpi3::detail::basic_datatype<type>{};
  auto recvbuffer = gpu::array<type, 1>{};
  auto full_matrix = gpu::array<type, 2>{};

  std::vector<int> recvcounts;
  std::vector<int> displs;
	
  int pos = 0;
  for(int iproc = 0; iproc < matrix.comm().size(); iproc++){
    auto coords = matrix.comm().coordinates(iproc);
    auto size = matrix.partx().local_size(coords[0])*matrix.party().local_size(coords[1]);
    recvcounts.emplace_back(size);
    displs.emplace_back(pos);
    pos += size;
  }

  assert(pos == matrix.sizex()*matrix.sizey());
  assert(int(recvcounts.size()) == matrix.comm().size());
  assert(recvcounts[matrix.comm().rank()] == matrix.block().num_elements()); 
  
	if(matrix.comm().root()) recvbuffer.reextent(matrix.sizex()*matrix.sizey());

	MPI_Gatherv(raw_pointer_cast(matrix.block().data_elements()), matrix.block().num_elements(), mpi_type,
							raw_pointer_cast(recvbuffer.data_elements()), recvcounts.data(), displs.data(), mpi_type, 0, matrix.comm().get());
	
	if(matrix.comm().root()) {
		full_matrix.reextent({matrix.sizex(), matrix.sizey()});

		for(int iproc = 0; iproc < matrix.comm().size(); iproc++){
      auto coords = matrix.comm().coordinates(iproc);
      auto xsize = matrix.partx().local_size(coords[0]);
      auto ysize = matrix.party().local_size(coords[1]);

			gpu::run(ysize, xsize, [ful = begin(full_matrix), rec = begin(recvbuffer), startx = matrix.partx().start(coords[0]), starty = matrix.party().start(coords[1]), disp = displs[iproc], ysize]
							 GPU_LAMBDA (auto iy, auto ix){
				ful[startx + ix][starty + iy] = rec[disp + ix*ysize + iy];
			});
		}
  }
	
  return full_matrix;
}

////////////////////////////////////////////////////////////////////////////////////

template <typename ArrayType>
auto scatter(parallel::cartesian_communicator<2> comm, ArrayType const & full_matrix) -> matrix::distributed<typename ArrayType::element_type>{

	CALI_CXX_MARK_SCOPE("matrix::scatter");

  using type = typename ArrayType::element_type;
  auto mpi_type = boost::mpi3::detail::basic_datatype<type>{};

	int szs[2];
	if(comm.root()){
		szs[0] = std::get<0>(sizes(full_matrix));
		szs[1] = std::get<1>(sizes(full_matrix));
	}

	comm.broadcast_n(szs, 2);
	comm.barrier();
	
	auto matrix = matrix::distributed<type>(comm, szs[0], szs[1]);
	auto sendbuffer = gpu::array<type, 1>{};
  
	std::vector<int> sendcounts;
  std::vector<int> displs;
	
  int pos = 0;
  for(int iproc = 0; iproc < matrix.comm().size(); iproc++){
    auto coords = matrix.comm().coordinates(iproc);
    auto size = matrix.partx().local_size(coords[0])*matrix.party().local_size(coords[1]);
    sendcounts.emplace_back(size);
    displs.emplace_back(pos);
    pos += size;
  }

  assert(pos == matrix.sizex()*matrix.sizey());
  assert(int(sendcounts.size()) == matrix.comm().size());
  assert(sendcounts[matrix.comm().rank()] == matrix.block().num_elements()); 
	
	if(matrix.comm().root()) {
		sendbuffer.reextent(matrix.sizex()*matrix.sizey());			
		
		for(int iproc = 0; iproc < matrix.comm().size(); iproc++){
      auto coords = matrix.comm().coordinates(iproc);
      auto xsize = matrix.partx().local_size(coords[0]);
      auto ysize = matrix.party().local_size(coords[1]);

			gpu::run(ysize, xsize, [ful = begin(full_matrix), sen = begin(sendbuffer), startx = matrix.partx().start(coords[0]), starty = matrix.party().start(coords[1]), disp = displs[iproc], ysize]
							 GPU_LAMBDA (auto iy, auto ix){
				sen[disp + ix*ysize + iy] = ful[startx + ix][starty + iy];
			});
		}
  }

	MPI_Scatterv(raw_pointer_cast(sendbuffer.data_elements()), sendcounts.data(), displs.data(), mpi_type,
							 raw_pointer_cast(matrix.block().data_elements()), matrix.block().num_elements(), mpi_type, 0, matrix.comm().get());

	return matrix;
}


}
}
#endif

#ifdef INQ_MATRIX_GATHER_SCATTER_UNIT_TEST
#undef INQ_MATRIX_GATHER_SCATTER_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <mpi3/environment.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

  parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
  parallel::cartesian_communicator<2> cart_comm(comm, {});
  
	auto element = [](auto ix, auto iy){
		return (sqrt(ix) + 1.0)*cos(iy + 1.0);
  };

  auto mm = 278;
  auto nn = 323;
  
  matrix::distributed<double> mat(cart_comm, mm, nn);
		
  for(int ix = 0; ix < mat.sizex(); ix++){
    for(int iy = 0; iy < mat.sizey(); iy++){
      auto ixg = parallel::global_index(ix);
      auto iyg = parallel::global_index(iy);			
      if(mat.is_local(ixg, iyg)) mat.block()[mat.partx().global_to_local(ixg)][mat.party().global_to_local(iyg)] = element(ix, iy);
    }
  }

  auto full_mat = matrix::gather(mat);
	
	if(cart_comm.root()){

		CHECK(full_mat.size() == mat.sizex());
		CHECK((~full_mat).size() == mat.sizey());
		
		for(int ix = 0; ix < mat.sizex(); ix++){
			for(int iy = 0; iy < mat.sizey(); iy++){
				CHECK(full_mat[ix][iy] == element(ix, iy));
			}
		}
	}

	auto mat2 = matrix::scatter(cart_comm, full_mat);	

	CHECK(mat2.sizex() == mm);
	CHECK(mat2.sizey() == nn);	

	for(int ix = 0; ix < mat.partx().local_size(); ix++){
		for(int iy = 0; iy < mat.party().local_size(); iy++){
			CHECK(mat2.block()[ix][iy] == mat.block()[ix][iy]);
		}
	}
    
}
#endif
