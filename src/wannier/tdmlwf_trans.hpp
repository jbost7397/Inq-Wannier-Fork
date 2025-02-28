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
#include <iostream>
#include <vector>

namespace inq {
namespace wannier {

class tdmlwf_trans {

private:
	gpu::array<complex,2> u_; //JB: have to consider between gpu::array, std::vector of std::vectors, or perhaps matrix::distributed for parallelism?
	gpu::array<complex,3> a_;
	gpu::array<complex,2> adiag_;
	//std::vector<std::vector<std::vector<complex>>> a_;
	//std::vector<std::vector<inq::complex>> adiag_;
	//std::vector<std::vector<complex>> u_;
	states::orbital_set<basis::real_space, complex> wavefunctions_;

public:

////////////////////////////////////////////////////////////////////////////////
tdmlwf_trans(states::orbital_set<basis::real_space, complex> const & wavefunctions) : wavefunctions_(wavefunctions) {
  const int n_states = wavefunctions_.set_size();
  int nx = wavefunctions_.basis().local_sizes()[0];
  int ny = wavefunctions_.basis().local_sizes()[1];
  int nz = wavefunctions_.basis().local_sizes()[2];
  a_.reextent({6, n_states, n_states});
  adiag_.reextent({6, n_states});
  u_.reextent({n_states, n_states});

}//constructor

void normalize(void) {
  CALI_CXX_MARK_SCOPE("wannier_normalize");
  int n_states = wavefunctions_.set_size();
  int nx = wavefunctions_.basis().local_sizes()[0];
  int ny = wavefunctions_.basis().local_sizes()[1];
  int nz = wavefunctions_.basis().local_sizes()[2];

  gpu::array<double, 1> norm_squared_per_state({n_states}, 0.0);

  gpu::run(n_states, [hypercubic = begin(wavefunctions_.hypercubic()), nx, ny, nz, nsp = begin(norm_squared_per_state)] GPU_LAMBDA (auto k_wf) {
    for (int ix = 0; ix < nx; ++ix) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
          complex wf_component = hypercubic[ix][iy][iz][k_wf];
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

}

////////////////////////////////////////////////////////////////////////////////
void update(const states::orbital_set<basis::real_space, complex>& wavefunctions) {
  wavefunctions_ = wavefunctions;
  CALI_CXX_MARK_SCOPE("wannier_update");
  int n_states = wavefunctions_.set_size();
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
  normalize();

  gpu::run(6, n_states, n_states, [n_states, a_int=begin(a_)] GPU_LAMBDA (auto i, auto j, auto k) {
    a_int[i][j][k] = complex(0.0);
  });

  gpu::run(n_states, n_states, [u_int=begin(u_)] GPU_LAMBDA (auto i, auto j) {
      u_int[i][j] = (i == j) ? complex(1.0,0.0) : complex(0.0,0.0);
  });

  gpu::run(6, n_states, [adiag_int=begin(adiag_)] GPU_LAMBDA (auto i, auto j) {
    adiag_int[i][j] = complex(0.0);
  });

  gpu::array<double, 4> trig_array;
  trig_array.reextent({6, nx, ny, nz});

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
  
  gpu::run(n_states, n_states, [hypercubic = begin(wavefunctions_.hypercubic()), ta = begin(trig_array), nx, ny, nz, a = begin(a_)] GPU_LAMBDA (auto l_wf, auto k_wf) {
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
auto center(T i, const systems::cell & cell_) {
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
double wannier_distance(T i, T j, const systems::cell & cell_) {
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
bool overlap(T1 epsilon, T2 i, T2 j, const systems::cell & cell_) {
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
double total_overlaps(T epsilon, const systems::cell & cell_) {
  int sum = 0;
  for ( int i = 0; i < wavefunctions_.set_size(); i++ )
  {
    int count = 0;
    for ( int j = 0; j < wavefunctions_.set_size(); j++ )
    {
      if ( overlap(epsilon,i,j,cell_) )
        count++;
    }
    sum += count;
  }
    return ((double) sum)/(wavefunctions_.set_size()*wavefunctions_.set_size());
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double pair_fraction(T epsilon, const systems::cell & cell) {
  // pair_fraction: return fraction of pairs having non-zero overlap
  // count pairs (i,j) having non-zero overlap for i != j only
  int sum = 0;
  for ( int i = 0; i < wavefunctions_.set_size(); i++ )
  {
    int count = 0;
    for ( int j = i+1; j < wavefunctions_.set_size(); j++ )
    {
      if ( overlap(epsilon,i,j,cell) )
        count++;
    }
    sum += count;
  }
  // add overlap with self: (i,i)
  sum += wavefunctions_.set_size();
  return ((double) sum)/((wavefunctions_.set_size()*(wavefunctions_.set_size()+1))/2);
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
  auto tst = 1.0 - norm(c) - norm(s);
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
  for ( int i = 0; i < wavefunctions_.set_size(); i++ )
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
void apply_transform(states::orbital_set<basis::real_space, complex> & phi) {
  	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
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

	inq::systems::ions sys(inq::systems::cell::cubic(20.0_b).periodic());
        sys.insert(ionic::species("He").pseudo_file(inq::config::path::pseudo() + "He_ONCV_PBE-1.2.upf.gz"), {-7.0_b, -7.0_b, -7.0_b});
        sys.insert(ionic::species("He").pseudo_file(inq::config::path::pseudo() + "He_ONCV_PBE-1.2.upf.gz"), {8.0_b, 8.0_b, 8.0_b});
	inq::systems::electrons el(sys, options::electrons{}.cutoff(30.0_Ry));
	inq::ground_state::initial_guess(sys, el);

	inq::ground_state::calculate(sys, el, inq::options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-10_Ha));

	wannier::tdmlwf_trans mlwf_transformer(el.kpin()[0]);
        mlwf_transformer.update(el.kpin()[0]);
	mlwf_transformer.compute_transform();

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
