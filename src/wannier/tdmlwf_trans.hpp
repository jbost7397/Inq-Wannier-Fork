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
#include <states/ks_states.hpp>
#include <states/orbital_set.hpp>
#include <wannier/jade_complex.hpp>
#include <gpu/array.hpp>
#include <iostream>
#include <vector> 

namespace inq {
namespace wannier {

class tdmlwf_trans {

private:
	//gpu::array<complex,2> u_; //JB: have to consider between gpu::array, std::vector of std::vectors, or perhaps matrix::distributed for parallelism?
	//gpu::array<complex,3> a_;
	//gpu::array<complex,2> adiag_;
	std::vector<std::vector<std::vector<complex>>> a_; 
	std::vector<std::vector<inq::complex>> adiag_;
	std::vector<std::vector<complex>> u_;
	states::orbital_set<basis::real_space, complex> wavefunctions_;

public:

////////////////////////////////////////////////////////////////////////////////
tdmlwf_trans(states::orbital_set<basis::real_space, complex> const & wavefunctions) : wavefunctions_(wavefunctions) {

  const int n = wavefunctions_.set_size();
  int nx = wavefunctions_.basis().local_sizes()[0];
  int ny = wavefunctions_.basis().local_sizes()[1];
  int nz = wavefunctions_.basis().local_sizes()[2];
  int six = 6;
  //gpu::array<complex,3> a_ ({6,n,n});
  //gpu::array<complex,2> adiag_ ({6,n});
  a_.resize(6);
  adiag_.resize(6);
  int k; 
  for (k = 0; k < 6; k++) {
    //gpu::array<complex,2> a_[k];
    a_[k].resize(n);
    for (int iwf = 0; iwf < n; iwf++) {
	    a_[k][iwf].resize(n, 0);
    }
    adiag_[k].resize(n, 0);
  }

  //gpu::array<complex,2> u_({n,n});
  std::vector<std::vector<complex>> u_(n, std::vector<complex>(n));
}
////////////////////////////////////////////////////////////////////////////////
void normalize(void) {
    int n_states = wavefunctions_.set_size();
    int nx = wavefunctions_.basis().local_sizes()[0];
    int ny = wavefunctions_.basis().local_sizes()[1];
    int nz = wavefunctions_.basis().local_sizes()[2];
    for (int k_wf = 0; k_wf < n_states; ++k_wf) {
        double norm_squared = 0.0;
        for (int ix = 0; ix < nx; ++ix) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int iz = 0; iz < nz; ++iz) {
                    complex wf_component = wavefunctions_.hypercubic()[ix][iy][iz][k_wf];
                    norm_squared += norm(wf_component); 
                }
            }
        }


        double norm_factor = 1.0 / sqroot(norm_squared);
        for (int ix = 0; ix < nx; ++ix) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int iz = 0; iz < nz; ++iz) {
                    wavefunctions_.hypercubic()[ix][iy][iz][k_wf] *= norm_factor;
                }
            }
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
void update(void) {
  int n_states = wavefunctions_.set_size();
  int nprocs = wavefunctions_.basis().comm().size();
  std::array<int, 2> shape;
  int dim_x = std::round(std::cbrt(nprocs)); 
  shape[0] = dim_x;
  shape[1] = nprocs / dim_x;
  auto comm = parallel::cartesian_communicator<2>(wavefunctions_.basis().comm(), shape);
  int nx = wavefunctions_.basis().local_sizes()[0];
  int ny = wavefunctions_.basis().local_sizes()[1];
  int nz = wavefunctions_.basis().local_sizes()[2];

  double lx = sqroot(wavefunctions_.basis().cell()[0].norm());
  double ly = sqroot(wavefunctions_.basis().cell()[1].norm());
  double lz = sqroot(wavefunctions_.basis().cell()[2].norm());

  normalize();

  for (int k_wf = 0; k_wf < n_states; ++k_wf) {
    for (int l_wf = 0; l_wf < n_states; ++l_wf) {
      for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
          for (int iz = 0; iz < nz; ++iz) {
            complex c_ik = wavefunctions_.hypercubic()[ix][iy][iz][k_wf];
            auto conj_ik = conj_cplx(c_ik);

            auto coords = wavefunctions_.basis().point_op().rvector_cartesian(ix, iy, iz);

            double cos_x = cos(2.0 * M_PI * coords[0] / lx);
            double sin_x = sin(2.0 * M_PI * coords[0] / lx);
            double cos_y = cos(2.0 * M_PI * coords[1] / ly);
            double sin_y = sin(2.0 * M_PI * coords[1] / ly);
            double cos_z = cos(2.0 * M_PI * coords[2] / lz);
            double sin_z = sin(2.0 * M_PI * coords[2] / lz);

            complex c_jl = wavefunctions_.hypercubic()[ix][iy][iz][l_wf];

            a_[0][k_wf][l_wf] += conj_ik * c_jl * cos_x;
            a_[1][k_wf][l_wf] += conj_ik * c_jl * sin_x;
            a_[2][k_wf][l_wf] += conj_ik * c_jl * cos_y;
            a_[3][k_wf][l_wf] += conj_ik * c_jl * sin_y;
            a_[4][k_wf][l_wf] += conj_ik * c_jl * cos_z;
            a_[5][k_wf][l_wf] += conj_ik * c_jl * sin_z;
	    
	  }
        }
      }
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
void compute_transform(void)
{
  const int maxsweep = 100;
  const double tol = 1.e-8;
  auto sweep = jade_complex(maxsweep,tol,a_,u_,adiag_);
}
////////////////////////////////////////////////////////////////////////////////
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
  // Next lines: M_1_PI = 1.0/pi // DCY explicit arctan(sx,cx) to work for complex numbers
  const double itwopi = 1.0 / ( 2.0 * M_PI );
  const double t0 = (itwopi * atan2(sx,cx));
  const double t1 = (itwopi * atan2(sy,cy));
  const double t2 = (itwopi * atan2(sz,cz));
  const double x = (t0*cell_[0][0] + t1*cell_[0][1] + t2*cell_[0][2]);
  const double y = (t0*cell_[1][0] + t1*cell_[1][1] + t2*cell_[1][2]);
  const double z = (t0*cell_[2][0] + t1*cell_[2][1] + t2*cell_[2][2]);
  //const double x = (t0*cell_.a(0).x + t1*cell_.a(1).x + t2*cell_.a(2).x);
  //const double y = (t0*cell_.a(0).y + t1*cell_.a(1).y + t2*cell_.a(2).y);
  //const double z = (t0*cell_.a(0).z + t1*cell_.a(1).z + t2*cell_.a(2).z);
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
  const std::complex<double> c = adiag_[2*j][i]; //DCY
  const std::complex<double> s = adiag_[2*j+1][i]; //DCY
  // Next line: M_1_PI = 1.0/pi
  auto recip = cell.reciprocal(j);
  double length = sqrt(recip[0]*recip[0]+ recip[1]*recip[1] + recip[2]*recip[2]);
  const double fac = 1.0 / length;
  auto tst = 1.0 - std::norm(c) - std::norm(s);
  return fac*fac * ( 1.0 - std::norm(c) - std::norm(s) );
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
  return sqrt(spread2(cell));
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
void apply_transform(void) {

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
        sys.insert(ionic::species("O").pseudo_file(inq::config::path::unit_tests_data() + "O_ONCV_PBE-1.2.upf.gz"), {2.19744213_b, 3.16473822_b, -6.15854709_b});
        sys.insert(ionic::species("H").pseudo_file(inq::config::path::unit_tests_data() + "H_ONCV_PBE-1.2.upf.gz"), {3.65385704_b, 4.22625626_b, -5.59004071_b});
        sys.insert(ionic::species("H").pseudo_file(inq::config::path::unit_tests_data() + "H_ONCV_PBE-1.2.upf.gz"), {0.89470006_b, 4.26612956_b, -6.97158452_b});
	inq::systems::electrons el(sys, options::electrons{}.cutoff(30.0_Ry));
	inq::ground_state::initial_guess(sys, el);
	
	inq::ground_state::calculate(sys, el, inq::options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-10_Ha));

	wannier::tdmlwf_trans mlwf_transformer(el.kpin()[0]);
        mlwf_transformer.update();
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

        i = 2;
        auto center3 = mlwf_transformer.center(i, el.states_basis().cell());

        CHECK(center3[0] == Approx(0.0_a));
        CHECK(center3[1] == Approx(0.0_a));
        CHECK(center3[2] == Approx(0.0_a));

        double spread3 = mlwf_transformer.spread(i, el.states_basis().cell());
        CHECK(spread3 == Approx(1.16_a));

}
#endif  

