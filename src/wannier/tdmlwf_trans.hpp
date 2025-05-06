/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__WANNIER__TDMLWF_TRANS
#define INQ__WANNIER__TDMLWF_TRANS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>
#include <input/environment.hpp>
#include <multi/adaptors/fftw.hpp>
#include <math/complex.hpp>
#include <utils/profiling.hpp>
#include <math/complex.hpp>
#include <math/vector3.hpp>
#include <systems/cell.hpp>
#include <systems/electrons.hpp>
#include <basis/field.hpp>
#include <basis/real_space.hpp>
#include <matrix/gather_scatter.hpp>
#include <states/ks_states.hpp>
#include <states/orbital_set.hpp>
#include <operations/rotate.hpp>
#include <parallel/communicator.hpp>
#include <wannier/jade_complex.hpp>
#include <gpu/array.hpp>
#include <gpu/run.hpp>
#include <iostream>
#include <vector>

namespace inq {
namespace wannier {

class tdmlwf_trans {

private:
	gpu::array<complex,2> u_; //JB
	gpu::array<complex,3> a_;
	gpu::array<complex,2> adiag_;
	states::orbital_set<basis::real_space, complex> wavefunctions_;

public:

////////////////////////////////////////////////////////////////////////////////
tdmlwf_trans(states::orbital_set<basis::real_space, complex> const & wavefunctions) : wavefunctions_(wavefunctions) {
  const int n_states = wavefunctions_.set_size();
  a_.reextent({6, n_states, n_states});
  adiag_.reextent({6, n_states});
  u_.reextent({n_states, n_states});

}//constructor
////////////////////////////////////////////////////////////////////////////////
template <class CommType>
void normalize(CommType & comm) {
  CALI_CXX_MARK_SCOPE("wannier_normalize");
  int n_states_local = wavefunctions_.local_set_size();
  int n_states_global = wavefunctions_.set_size();
  int nx = wavefunctions_.basis().local_sizes()[0];
  int ny = wavefunctions_.basis().local_sizes()[1];
  int nz = wavefunctions_.basis().local_sizes()[2];
  auto rank = comm.rank();
  int offset = rank * n_states_local;
  /*if(n_states_local == n_states_global) {
     offset = 0;
  }*/

  gpu::array<double, 1> norm_squared_per_state({n_states_global}, 0.0);

  gpu::run(n_states_local, [hypercubic = begin(wavefunctions_.hypercubic()), offset, nx, ny, nz, nsp = begin(norm_squared_per_state)] GPU_LAMBDA (auto k_wf) {
    for (int ix = 0; ix < nx; ++ix) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
          complex wf_component = hypercubic[ix][iy][iz][k_wf];
          gpu::atomic::add(&nsp[k_wf + offset], norm(wf_component));
        }
      }
    }
  });

  comm.barrier();
  comm.all_reduce_in_place_n(raw_pointer_cast(norm_squared_per_state.data_elements()), norm_squared_per_state.num_elements(), std::plus<>());

  gpu::run(n_states_local, [hypercubic = begin(wavefunctions_.hypercubic()), offset, nx, ny, nz, nsp = begin(norm_squared_per_state)] GPU_LAMBDA (auto k_wf) {
    double norm_factor = 1.0 / sqroot(nsp[k_wf + offset]);
          gpu::atomic::add(&nsp[k_wf], norm(wf_component));
        }
      }

    }
  });

  auto comm = wavefunctions_.basis().comm();
  comm.all_reduce_in_place_n(raw_pointer_cast(norm_squared_per_state.data_elements()), norm_squared_per_state.num_elements(), std::plus<>());

  gpu::run(n_states, [hypercubic = begin(wavefunctions_.hypercubic()), nx, ny, nz, nsp = begin(norm_squared_per_state)] GPU_LAMBDA (auto k_wf) {
    double norm_factor = 1.0 / sqroot(nsp[k_wf]);
    for (int ix = 0; ix < nx; ++ix) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
          hypercubic[ix][iy][iz][k_wf] *= norm_factor;
        }
      }
    }
  });
}//normalize 
////////////////////////////////////////////////////////////////////////////////
template <class CommType>
gpu::array<complex, 2> prepare_buffer(CommType & comm){
    int n_states = wavefunctions_.set_size();
    int n_states_local = wavefunctions_.local_set_size();
    int nx = wavefunctions_.basis().local_sizes()[0];
    int ny = wavefunctions_.basis().local_sizes()[1];
    int nz = wavefunctions_.basis().local_sizes()[2];
    int local_data_elements = nx * ny * nz * n_states_local;

    // Prepare alltoall buffer
    gpu::array<complex, 2> send_buffer({comm.size(), local_data_elements});
    gpu::run(n_states_local, comm.size(), [nx, ny, nz, n_states_local, hypercubic = begin(wavefunctions_.hypercubic()), send_buf = begin(send_buffer)] GPU_LAMBDA (auto l_wf, auto cur_rank) {
	for (int ix = 0; ix < nx; ++ix){
	    for (int iy = 0; iy < ny; ++iy) {
	        for (int iz = 0; iz < nz; ++iz) {
		    int index = ix * ny * nz * n_states_local + iy * nz * n_states_local + iz * n_states_local + l_wf;
		    send_buf[cur_rank][index] = hypercubic[ix][iy][iz][l_wf];
		}
	    }
	}
    });
    comm.barrier();
    return send_buffer;
}
////////////////////////////////////////////////////////////////////////////////
template <class CommType>
void update(const states::orbital_set<basis::real_space, complex>& wavefunctions, CommType & comm) {
  wavefunctions_ = wavefunctions;
  CALI_CXX_MARK_SCOPE("wannier_update");
  int n_states_global = wavefunctions_.set_size();
  int n_states_local = wavefunctions_.local_set_size();
  int nx = wavefunctions_.basis().local_sizes()[0];
  int ny = wavefunctions_.basis().local_sizes()[1];
  int nz = wavefunctions_.basis().local_sizes()[2];

  double lx = sqroot(wavefunctions_.basis().cell()[0].norm());
  double ly = sqroot(wavefunctions_.basis().cell()[1].norm());
  double lz = sqroot(wavefunctions_.basis().cell()[2].norm());
  auto point_op = wavefunctions_.basis().point_op();
  auto cubic_part_x = wavefunctions_.basis().cubic_part(0);
  auto cubic_part_y = wavefunctions_.basis().cubic_part(1);
  auto cubic_part_z = wavefunctions_.basis().cubic_part(2);
  normalize(comm);

  gpu::run(n_states_global, n_states_global, 6, [a_int=begin(a_)] GPU_LAMBDA (auto k, auto j, auto i) {
    a_int[i][j][k] = complex(0.0);
  });

  gpu::run(n_states_global, n_states_global, [u_int=begin(u_)] GPU_LAMBDA (auto j, auto i) {
      u_int[i][j] = (i == j) ? complex(1.0,0.0) : complex(0.0,0.0);
  });

  gpu::run(n_states_global, 6, [adiag_int=begin(adiag_)] GPU_LAMBDA (auto j, auto i) {
    adiag_int[i][j] = complex(0.0);
  });

  gpu::array<double, 4> trig_array({6, nx, ny, nz});

  gpu::run(nz, ny, nx, [cubic_part_x, cubic_part_y, cubic_part_z, point_op, ta = begin(trig_array), lx, ly, lz] GPU_LAMBDA (auto iz, auto iy, auto ix) {
    auto ixg = cubic_part_x.local_to_global(ix);
    auto iyg = cubic_part_y.local_to_global(iy);
    auto izg = cubic_part_z.local_to_global(iz);
    auto coords = point_op.rvector_cartesian(ixg, iyg, izg);
    ta[0][ix][iy][iz] = cos(2.0 * M_PI * coords[0] / lx);
    ta[1][ix][iy][iz] = sin(2.0 * M_PI * coords[0] / lx);
    ta[2][ix][iy][iz] = cos(2.0 * M_PI * coords[1] / ly);
    ta[3][ix][iy][iz] = sin(2.0 * M_PI * coords[1] / ly);
    ta[4][ix][iy][iz] = cos(2.0 * M_PI * coords[2] / lz);
    ta[5][ix][iy][iz] = sin(2.0 * M_PI * coords[2] / lz);
  });
  
  if (n_states_global > n_states_local) { //parallelized along states
    parallel::communicator comm2{boost::mpi3::environment::get_world_instance()};
    auto buf = prepare_buffer(comm2);
    gpu::sync();
    comm.barrier();
    parallel::alltoall(buf, comm2);
    gpu::sync();
    comm.barrier();
    gpu::run(n_states_global, n_states_local, [rec_buf = begin(buf), n_states_local, ta = begin(trig_array), nx, ny, nz, a = begin(a_), rank = comm.rank()] GPU_LAMBDA (auto l_wf, auto k_wf) {
      int global_k_wf = rank * n_states_local + k_wf;
      int owner_rank_l = l_wf / n_states_local;
      int local_index_l = l_wf % n_states_local;

      for (int ix = 0; ix < nx; ix++){
        for (int iy = 0; iy < ny; iy++){
          for (int iz = 0; iz < nz; iz++){
            int index_k = k_wf + ix * ny * nz * n_states_local + iy * nz * n_states_local + iz * n_states_local;
            int index_l = local_index_l + ix * ny * nz * n_states_local + iy * nz * n_states_local + iz * n_states_local;

            complex conj_ik = conj_cplx(rec_buf[rank][index_k]); 
            complex c_jl = rec_buf[owner_rank_l][index_l];           

            a[0][global_k_wf][l_wf] += conj_ik * c_jl * ta[0][ix][iy][iz];
            a[1][global_k_wf][l_wf] += conj_ik * c_jl * ta[1][ix][iy][iz];
            a[2][global_k_wf][l_wf] += conj_ik * c_jl * ta[2][ix][iy][iz];
            a[3][global_k_wf][l_wf] += conj_ik * c_jl * ta[3][ix][iy][iz];
            a[4][global_k_wf][l_wf] += conj_ik * c_jl * ta[4][ix][iy][iz];
            a[5][global_k_wf][l_wf] += conj_ik * c_jl * ta[5][ix][iy][iz];

          }
        }
      }
    });
    
    gpu::sync();
    comm.barrier();

    auto rank = comm.rank();

    /*std::cout << "Rank " << rank << ": Initial a_[0][0][0] = " << a_[0][0][0] << std::endl;
    std::cout << "Rank " << rank << ": Initial a_[4][555][555] = " << a_[4][555][555] << std::endl;*/

    CALI_CXX_MARK_SCOPE("wannier_update::reduce_a");
    comm.all_reduce_in_place_n(raw_pointer_cast(a_.data_elements()), a_.num_elements(), std::plus<>());

    comm.barrier();

    if(rank == 0){
    	std::ofstream output_file("a_mat.dat", std::ios_base::app);
    	for (int i = 0; i < n_states_global; i++){
		    for (int j = 0; j < n_states_global; j++){
			    output_file << a_[0][i][j] << std::endl;
			    output_file << a_[1][i][j] << std::endl;
			    output_file << a_[2][i][j] << std::endl;
			    output_file << a_[3][i][j] << std::endl;
			    output_file << a_[4][i][j] << std::endl;
			    output_file << a_[5][i][j] << std::endl;
		    }
    	}
    }
    /*std::cout << "Rank " << rank << ": Reduced a_[0][0][0] = " << a_[0][0][0] << std::endl;
    std::cout << "Rank " << rank << ": Reduced a_[4][555][555] = " << a_[4][555][555] << std::endl;*/
  }
  
  else{
    gpu::run(n_states_global, n_states_global, [hypercubic = begin(wavefunctions_.hypercubic()), ta = begin(trig_array), nx, ny, nz, a = begin(a_)] GPU_LAMBDA (auto l_wf, auto k_wf) {
      for (int ix = 0; ix < nx; ix++){
        for (int iy = 0; iy < ny; iy++){
          for (int iz = 0; iz < nz; iz++){

            complex c_ik = hypercubic[ix][iy][iz][k_wf];
            auto conj_ik = conj_cplx(c_ik);
            complex c_jl = hypercubic[ix][iy][iz][l_wf];
            a[0][k_wf][l_wf] += conj_ik * c_jl * ta[0][ix][iy][iz];
            a[1][k_wf][l_wf] += conj_ik * c_jl * ta[1][ix][iy][iz];
            a[2][k_wf][l_wf] += conj_ik * c_jl * ta[2][ix][iy][iz];
            a[3][k_wf][l_wf] += conj_ik * c_jl * ta[3][ix][iy][iz];
            a[4][k_wf][l_wf] += conj_ik * c_jl * ta[4][ix][iy][iz];
            a[5][k_wf][l_wf] += conj_ik * c_jl * ta[5][ix][iy][iz];

	  }
        }
      }
    });

    if(comm.size() > 1){
	comm.all_reduce_in_place_n(raw_pointer_cast(a_.data_elements()), a_.num_elements(), std::plus<>());
    }
  }


}//update
////////////////////////////////////////////////////////////////////////////////
void compute_transform(double tol)
{
  const int maxsweep = 100;
  jade_complex(maxsweep,tol,a_,u_,adiag_);
}
////////////////////////////////////////////////////////////////////////////////
=======
	    /*gpu::atomic::add(&a[0][k_wf][l_wf], conj_ik * c_jl * ta[0][ix][iy][iz]);
	    gpu::atomic::add(&a[1][k_wf][l_wf], conj_ik * c_jl * ta[1][ix][iy][iz]);
	    gpu::atomic::add(&a[2][k_wf][l_wf], conj_ik * c_jl * ta[2][ix][iy][iz]);
	    gpu::atomic::add(&a[3][k_wf][l_wf], conj_ik * c_jl * ta[3][ix][iy][iz]);
	    gpu::atomic::add(&a[4][k_wf][l_wf], conj_ik * c_jl * ta[4][ix][iy][iz]);
	    gpu::atomic::add(&a[5][k_wf][l_wf], conj_ik * c_jl * ta[5][ix][iy][iz]);*/

	}
      }
    }
  });

  gpu::sync();

  if (wavefunctions.basis().comm().size() > 1) {
    CALI_CXX_MARK_SCOPE("wannier_update::reduce_a");
    wavefunctions.basis().comm().all_reduce_in_place_n(raw_pointer_cast(a_.data_elements()), a_.num_elements(), std::plus<>());
  }

  /*auto rank = wavefunctions.basis().comm().rank();

  std::cout << "Rank " << rank << ": Reduced a_[0][0][0] = " << a_[0][0][0] << std::endl;*/

}//update
////////////////////////////////////////////////////////////////////////////////
void compute_transform(void)
{
  const int maxsweep = 100;
  const double tol = 1.e-8;
  jade_complex(maxsweep,tol,a_,u_,adiag_);
}
////////////////////////////////////////////////////////////////////////////////
auto get_a(void){
  return a_;
}
///
const states::orbital_set<basis::real_space, complex>& get_wavefunctions() const {
  return wavefunctions_;
}
////////////////////////////////////////////////////////////////////////////////
template <typename T>
auto center(T i, const systems::cell & cell_) const {
  assert(i >= 0 && i < wavefunctions_.set_size());
  const double cx = real(adiag_[0][i]);
  const double sx = real(adiag_[1][i]);
  const double cy = real(adiag_[2][i]);
  const double sy = real(adiag_[3][i]);
  const double cz = real(adiag_[4][i]);
  const double sz = real(adiag_[5][i]);
  // Ratios for inputs into atan functions below
  //const complex<double> sxcx = sx / cx;
  //const complex<double> sycy = sy / cy;
  //const complex<double> szcz = sz / cz;
  const double itwopi = 1.0 / ( 2.0 * M_PI );
  const double t0 = (itwopi * atan2(sx,cx));
  const double t1 = (itwopi * atan2(sy,cy));
  const double t2 = (itwopi * atan2(sz,cz));
  const double x = (t0*cell_[0][0] + t1*cell_[0][1] + t2*cell_[0][2]);
  const double y = (t0*cell_[1][0] + t1*cell_[1][1] + t2*cell_[1][2]);
  const double z = (t0*cell_[2][0] + t1*cell_[2][1] + t2*cell_[2][2]);
  vector3<double> center_d3{x,y,z};

  return center_d3;
}
////////////////////////////////////////////////////////////////////////////////
template <typename T>
double wannier_distance(T i, T j, const systems::cell & cell_) const {
  assert(i >=0 && i < wavefunctions_.set_size());
  assert(j >=0 && j < wavefunctions_.set_size());
  vector3<double>ctr_i = center(i, cell_);
  vector3<double>ctr_j = center(j, cell_);
  double x_dist = ctr_i[0] - ctr_j[0];
  double y_dist = ctr_i[1] - ctr_j[1];
  double z_dist = ctr_i[2] - ctr_j[2];
  double total_dist = x_dist*x_dist + y_dist*y_dist + z_dist*z_dist;
  double root = sqroot(total_dist);
  return abs(root);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2>
bool overlap(T1 epsilon, T2 i, T2 j, const systems::cell & cell_) const {
  // overlap: return true if the functions i and j overlap according to distance
  double x = cell_[0][0]*cell_[0][0] + cell_[0][1]*cell_[0][1] + cell_[0][2]*cell_[0][2];
  double y = cell_[1][1]*cell_[1][1] + cell_[1][2]*cell_[1][2] + cell_[1][2]*cell_[1][2];
  double z = cell_[2][2]*cell_[2][2] + cell_[2][1]*cell_[2][1] + cell_[2][2]*cell_[2][2];
  double len = sqrt(x+y+z);
  if (wannier_distance(i,j, cell_) <= epsilon || wannier_distance(i,j, cell_) >= (len - epsilon) )
      return true;  //need sqrt(a0^2 + a1^2 + a2^2) for cell diagonal distance. Diagonal dist - epsilon for pbc
  // return false if the states don't overlap
  return false;
}
////////////////////////////////////////////////////////////////////////////////
template <typename T>
auto get_overlaps_of_j(T epsilon, int j, const systems::cell & cell_) const {
  	const int n_states = wavefunctions_.set_size();
	gpu::array<int, 1> olap_j(n_states);
	int count = 0;
	for(int i = 0; i < n_states; i++){
		if(overlap(epsilon, i, j, cell_)){
			olap_j[count++] = i;
		}
	}
	olap_j.reextent(count);
	return olap_j;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double total_overlaps(T epsilon, const systems::cell & cell_) {

  int n = wavefunctions_.set_size();
  gpu::array<int,1> sum({1}, 0);
  gpu::run(n, n, [epsilon, cell_, sum_int=begin(sum)] GPU_LAMBDA (auto i, auto j) {
    if (overlap(epsilon, i, j, cell_)) {
      gpu::atomic::add(&sum_int[0], 1);
    }
  });
  gpu::sync(); //CS probably don't need 

  return static_cast<double>(sum[0]) / (n * n);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double pair_fraction(T epsilon, const systems::cell & cell_) {
  // pair_fraction: return fraction of pairs having non-zero overlap
  // count pairs (i,j) having non-zero overlap for i != j only
  int n = wavefunctions_.set_size();
  gpu::array<int,1> sum({1}, 0);

  gpu::run(n, n, [epsilon, cell_, sum_int=begin(sum)] GPU_LAMBDA (auto i, auto j) {
    if (j > i) { //CS avoid duplicates 
        if (overlap(epsilon, i, j, cell_)) {
            gpu::atomic::add(&sum_int[0], 1);
        }
    }
  });
  gpu::sync(); //CS probably don't need

  // add overlap with self: (i,i)
  int total = sum[0] + n;
  return static_cast<double>(total)/((n*(n+1))/2);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread2(T i, T j, const systems::cell & cell) {
  assert(i >= 0 && i < wavefunctions_.set_size());
  assert(j >= 0 && j < 3);
  const complex c = adiag_[2*j][i]; //DCY
  const complex s = adiag_[2*j+1][i]; //DCY
  auto recip = cell.reciprocal(j);
  double length = sqrt(recip[0]*recip[0]+ recip[1]*recip[1] + recip[2]*recip[2]);
  const double fac = 1.0 / length;
  return fac*fac * ( 1.0 - norm(c) - norm(s) );
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread2(T i, const systems::cell & cell) {
  assert(i >= 0 & i < wavefunctions_.set_size());
  return spread2(i,0,cell) + spread2(i,1,cell) + spread2(i,2,cell);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread(T i, const systems::cell & cell) {
  return sqroot(spread2(i,cell));
}

////////////////////////////////////////////////////////////////////////////////
double spread2(const systems::cell & cell) {
  double sum = 0.0;
  for (int i = 0; i < wavefunctions_.set_size(); i++ )
    sum += spread2(i, cell);
  return sum;
}

////////////////////////////////////////////////////////////////////////////////
double spread(const systems::cell & cell) {
  return sqroot(spread2(cell));
}
////////////////////////////////////////////////////////////////////////////////
auto dipole(const systems::cell & cell) {
  // total electronic dipole
  vector3<double> sum{0.0,0.0,0.0};
  for ( int i = 0; i < wavefunctions_.set_size(); i++ )
    sum -= 2.0 * center(i,cell);  //CS need to pass state occupations (assume fully occupied for now) How?
  return sum;
}
////////////////////////////////////////////////////////////////////////////////
template <class CommType>
void apply_transform(states::orbital_set<basis::real_space, complex> & phi, CommType & comm) {
  	parallel::cartesian_communicator<2> cart_comm(comm, {});
	auto rot = matrix::scatter(cart_comm, u_, /* root = */ 0);	
	operations::rotate(rot, phi);
}
////////////////////////////////////////////////////////////////////////////////
}; //tdmlwf
} //wannier
} //inq
#endif

///////////////////////////////////////////////////////////////////
#ifdef INQ_WANNIER_TDMLWF_TRANS_UNIT_TEST
#undef INQ_WANNIER_TDMLWF_TRANS_UNIT_TEST

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

        using namespace inq;
        using namespace Catch::literals;
        using Catch::Approx;

  	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	inq::systems::ions sys(inq::systems::cell::cubic(20.0_b).periodic());
        sys.insert(ionic::species("He").pseudo_file(inq::config::path::pseudo() + "He_ONCV_PBE-1.2.upf.gz"), {-7.0_b, -7.0_b, -7.0_b});
        sys.insert(ionic::species("He").pseudo_file(inq::config::path::pseudo() + "He_ONCV_PBE-1.2.upf.gz"), {8.0_b, 8.0_b, 8.0_b});
	inq::systems::electrons el(sys, options::electrons{}.cutoff(30.0_Ry));
	inq::ground_state::initial_guess(sys, el);

	inq::ground_state::calculate(sys, el, inq::options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-10_Ha));

	wannier::tdmlwf_trans mlwf_transformer(el.kpin()[0]);
        mlwf_transformer.update(el.kpin()[0], comm);
	mlwf_transformer.compute_transform(1e-8);

	int i = 0;
        auto center = mlwf_transformer.center(i, el.states_basis().cell());

        CHECK(center[0] == Approx(8.0_a));
        CHECK(center[1] == Approx(8.0_a));
        CHECK(center[2] == Approx(8.0_a));

	double spread = mlwf_transformer.spread(i, el.states_basis().cell());
        CHECK(spread == Approx(1.16_a));

	i = 1;
        auto center2 = mlwf_transformer.center(i, el.states_basis().cell());

        CHECK(center2[0] == Approx(-7.0_a));
        CHECK(center2[1] == Approx(-7.0_a));
        CHECK(center2[2] == Approx(-7.0_a));

	double spread2 = mlwf_transformer.spread(i, el.states_basis().cell());
        CHECK(spread2 == Approx(1.16_a));
}
#endif
