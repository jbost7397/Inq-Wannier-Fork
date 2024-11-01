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
  	states::ks_states states_;
  	basis::real_space basis_; 

public:

////////////////////////////////////////////////////////////////////////////////
tdmlwf_trans(const basis::real_space & basis, states::ks_states const & states) :
      basis_(basis), states_(states) {

  const int n = states_.num_states();
  int nprocs = basis_.comm().size();
  std::array<int, 2> shape;
  int dim_x = std::round(std::cbrt(nprocs)); 
  shape[0] = dim_x;
  shape[1] = nprocs / dim_x;
  auto comm = parallel::cartesian_communicator<2>(basis_.comm(), shape);
  a_.resize(6);
  adiag_.resize(6);
  int k; 
  for (k = 0; k < 6; k++) {
    //gpu::array<complex,2> a_[k];
    a_[k].resize(n);
    adiag_[k].resize(n);
  }

  //gpu::array<complex,2> u_({n,n});
  std::vector<std::vector<complex>> u_(n, std::vector<complex>(n));


  
}	  

//JB: compute a matrices in real space. Should give exact result, but obviously inefficent (O(n^4)) for larger systems so still need to implement reciprocal space algorithm
void update(void) {
  int n_states = states_.num_states();
  int nprocs = basis_.comm().size();
  std::array<int, 2> shape;
  int dim_x = std::round(std::cbrt(nprocs)); 
  shape[0] = dim_x;
  shape[1] = nprocs / dim_x;
  auto comm = parallel::cartesian_communicator<2>(basis_.comm(), shape);
  int nx = basis_.local_sizes()[0];
  int ny = basis_.local_sizes()[1];
  int nz = basis_.local_sizes()[2];

  double lx = basis_.cell()[0].norm();
  double ly = basis_.cell()[1].norm();
  double lz = basis_.cell()[2].norm();

  states::orbital_set<basis::real_space, complex> orbital_set(basis_, n_states, 1, {0.0, 0.0, 0.0}, 0, comm);

  const auto& wavefunctions = orbital_set.hypercubic();

  for (int iwf = 0; iwf < n_states; ++iwf) {
    for (int ix = 0; ix < nx; ++ix) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
          auto coords = basis_.point_op().rvector_cartesian(ix, iy, iz);

          double cos_x = cos(2.0 * M_PI * coords[0] / lx);
          double sin_x = sin(2.0 * M_PI * coords[0] / lx);
          double cos_y = cos(2.0 * M_PI * coords[1] / ly);
          double sin_y = sin(2.0 * M_PI * coords[1] / ly);
          double cos_z = cos(2.0 * M_PI * coords[2] / lz);
          double sin_z = sin(2.0 * M_PI * coords[2] / lz);

	  complex wf_coeff = wavefunctions[ix][iy][iz][iwf];

          a_[0][iwf][ix * ny * nz + iy * nz + iz] += wf_coeff * cos_x;
          a_[1][iwf][ix * ny * nz + iy * nz + iz] += wf_coeff * sin_x;
          a_[2][iwf][ix * ny * nz + iy * nz + iz] += wf_coeff * cos_y;
          a_[3][iwf][ix * ny * nz + iy * nz + iz] += wf_coeff * sin_y;
          a_[4][iwf][ix * ny * nz + iy * nz + iz] += wf_coeff * cos_z;
          a_[5][iwf][ix * ny * nz + iy * nz + iz] += wf_coeff * sin_z;
        }
      }
    }
  }
}


//JB: actual update implementation, wip
/*
void update(void)
{
  const int n = states_.num_states();
  int nprocs = basis_.comm().size();
  std::array<int, 2> shape;
  int dim_x = std::round(std::cbrt(nprocs)); 
  shape[0] = dim_x;
  shape[1] = nprocs / dim_x;
  auto comm = parallel::cartesian_communicator<2>(basis_.comm(), shape);
  auto sd = std::make_unique<states::orbital_set<basis::real_space, math::complex>>(basis_, n, 1, {0.0, 0.0, 0.0}, 0, comm);
  auto& c = sd->matrix();
  auto f = operations::transform::to_fourier(c);
 
  for ( int i = 0; i < 6; i++ )
  {
    a_[i]->resize(c.size(), c.size());
    adiag_[i].resize(c.size());
  }
  u_->resize(c.size(), c.size());

  for ( int ist = 0; ist < c.local_set_size(); ist++ )
  {
    auto state_view_fourier = f.matrix()[ist]; 
    auto fs = std::make_unique<gpu::array<math::complex,1>>(state_view_fourier.size());
    auto fc = std::make_unique<gpu::array<math::complex,1>>(state_view_fourier.size());

    std::vector<inq::math::complex> flattened_fourier_data(state_view_fourier.size());
    std::copy(state_view_fourier.data(), state_view_fourier.data() + state_view_fourier.size(), flattened_fourier_data.begin());

    compute_sincos(state_view_fourier.size(), flattened_fourier_data.data(), fc.get(), fs.get()); 
  }

  auto real_c = operations::transform::to_real(f);

  // dot products a_[0] = <cos x>, a_[1] = <sin x>
  a_[0]->gemm('c','n',1.0,c,ccosx,0.0);
  a_[0]->zger(-1.0,c,0,ccosx,0);
  a_[1]->gemm('c','n',1.0,c,csinx,0.0);
  a_[1]->zger(-1.0,c,0,csinx,0);

  // dot products a_[2] = <cos y>, a_[3] = <sin y>
  a_[2]->gemm('c','n',1.0,c,ccosy,0.0);
  a_[2]->zger(-1.0,c,0,ccosy,0);
  a_[3]->gemm('c','n',1.0,c,csiny,0.0);
  a_[3]->zger(-1.0,c,0,csiny,0);

  // dot products a_[4] = <cos z>, a_[5] = <sin z>
  a_[4]->gemm('c','n',1.0,c,ccosz,0.0);
  a_[4]->zger(-1.0,c,0,ccosz,0);
  a_[5]->gemm('c','n',1.0,c,csinz,0.0);
  a_[5]->zger(-1.0,c,0,csinz,0);
}*/

////////////////////////////////////////////////////////////////////////////////
template <typename T, class vector_type1, class vector_type2 > 
void compute_sincos(T n, vector_type1* f, vector_type2* fc, vector_type2* fs) { 
//void compute_sincos(const int n, const std::complex<double>* f, std::complex<double>* fc, std::complex<double>* fs) {

  // fc[i] =     0.5 * ( f[i-1] + f[i+1] )
  // fs[i] = (0.5/i) * ( f[i-1] - f[i+1] )

  // i = 0
  std::complex<double> zp = f[n-1];
  std::complex<double> zm = f[1];
  fc[0] = 0.5 * ( zp + zm );
  std::complex<double> zdiff = zp - zm;
  fs[0] = 0.5 * std::complex<double>(imag(zdiff),-real(zdiff));
  for ( int i = 1; i < n-1; i++ )
  {
    const std::complex<double> zzp = f[i-1];
    const std::complex<double> zzm = f[i+1];
    fc[i] = 0.5 * ( zzp + zzm );
    const std::complex<double> zzdiff = zzp - zzm;
    fs[i] = 0.5 * std::complex<double>(imag(zzdiff),-real(zzdiff));
  }
  // i = n-1
  zp = f[n-2];
  zm = f[0];
  fc[n-1] = 0.5 * ( zp + zm );
  zdiff = zp - zm;
  fs[n-1] = 0.5 * std::complex<double>(imag(zdiff),-real(zdiff));
}

void compute_transform(void)
{
  const int maxsweep = 100;
  const double tol = 1.e-8;
  auto sweep = jade_complex(maxsweep,tol,a_,u_,adiag_);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
auto center(T i, const systems::cell & cell_) {
  assert(i >= 0 && i < st.num_states());
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
  assert(i >=0 && i < st.num_states());
  assert(j >=0 && j < st.num_states());
  vector3<double>ctr_i = center(i, cell_);
  vector3<double>ctr_j = center(j, cell_);
  double x_dist = ctr_i[0] - ctr_j[0];
  double y_dist = ctr_i[1] - ctr_j[1];
  double z_dist = ctr_i[2] - ctr_j[2];
  double total_dist = x_dist*x_dist + y_dist*y_dist + z_dist*z_dist;
  double root = std::sqrt(total_dist);
  return std::abs(root);
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
/*template <typename T>
double total_overlaps(T epsilon, const systems::cell & cell_) {
  int sum = 0;
  for ( int i = 0; i < st.num_states(); i++ )
  {
    int count = 0;
    for ( int j = 0; j < st.num_states(); j++ )
    {
      if ( overlap(epsilon,i,j,cell_) )
        count++;
    }
    sum += count;
  }
    return ((double) sum)/(st.num_states()*st.num_states());
}*/

////////////////////////////////////////////////////////////////////////////////
/*template <typename T>
double pair_fraction(T epsilon, const systems::cell & cell) {
  // pair_fraction: return fraction of pairs having non-zero overlap
  // count pairs (i,j) having non-zero overlap for i != j only
  int sum = 0;
  for ( int i = 0; i < st.num_states(); i++ )
  {
    int count = 0;
    for ( int j = i+1; j < st.num_states(); j++ )
    {
      if ( overlap(epsilon,i,j,cell) )
        count++;
    }
    sum += count;
  }
  // add overlap with self: (i,i)
  sum += st.num_states();
  return ((double) sum)/((st.num_states()*(st.num_states()+1))/2);
}*/

////////////////////////////////////////////////////////////////////////////////
/*template <typename T>
double spread2(T i, T j, const systems::cell & cell) {
  assert(i >= 0 && i < st.num_states());
  assert(j >= 0 && j < 3);
  const std::complex<double> c = adiag_[2*j][i]; //DCY
  const std::complex<double> s = adiag_[2*j+1][i]; //DCY
  // Next line: M_1_PI = 1.0/pi
  auto recip = cell.reciprocal(j);
  double length = sqrt(recip[0]*recip[0]+ recip[1]*recip[1] + recip[2]*recip[2]);
  const double fac = 1.0 / length;
  return fac*fac * ( 1.0 - std::norm(c) - std::norm(s) );
}*/

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread2(T i, const systems::cell & cell) {
  assert(i >= 0 & i < st.num_states());
  return spread2(i,0,cell) + spread2(i,1,cell) + spread2(i,2,cell);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread(T i, const systems::cell & cell) {
  return std::sqrt(spread2(i,cell));
}

////////////////////////////////////////////////////////////////////////////////
/*double spread2(const systems::cell & cell) {
  double sum = 0.0;
  for ( int i = 0; i < st.num_states(); i++ )
    sum += spread2(i, cell);
  return sum;
}*/

////////////////////////////////////////////////////////////////////////////////
/*double spread(const systems::cell & cell) {
  return sqrt(spread2(cell));
}*/

////////////////////////////////////////////////////////////////////////////////
/*auto dipole(const systems::cell & cell) {
  // total electronic dipole
  vector3<double> sum{0.0,0.0,0.0};
  for ( int i = 0; i < st.num_states(); i++ )
    sum -= 2.0 * center(i,cell);  //CS need to pass state occupations (assume fully occupied for now) How?
  return sum;
}*/


////////////////////////////////////////////////////////////////////////////////
};
}
}
#endif 

///////////////////////////////////////////////////////////////////
#ifdef INQ_WANNIER_TDMLWF_TRANS_UNIT_TEST
#undef INQ_WANNIER_TDMLWF_TRANS_UNIT_TEST

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

        using namespace inq;
        using namespace Catch::literals;
        using Catch::Approx;

	/*
		//auto cell = systems::cell::cubic(15.0_b).periodicity(3); 
	
        //CS adiag_ values correspond to H2 gs. Should give <mlwf center="    0.000000    0.000000    0.711503 " spread=" 1.578285 "/>

	SECTION("Wannier Centers"){

		//systems::cell cell(vector3<double>(15.0, 0.0, 0.0), vector3<double>(0.0, 15.0, 0.0), vector3<double>(0.0, 0.0, 15.0)); 
		auto cell = systems::cell::cubic(15.0_b).periodicity(3); //CS more conventional call to a cubic cell with pbc
		int i = 0;
		auto center = wannier::center(i, cell); 

		CHECK(center[0] == 0.000000_a);
		CHECK(center[1] == 0.000000_a);
		CHECK(center[2] == 0.711503_a); 
		
		int j = 0;
		auto dist = wannier::wannier_distance(i, j, cell);
		CHECK(dist == 0.00_a);
		
		double epsilon = 10.0;
		auto overlap = wannier::overlap(epsilon, i, j, cell);
		CHECK(overlap == true);
		
		auto ols = wannier::total_overlaps(epsilon, cell);
		CHECK(ols == 1.00_a);
		auto pair = wannier::pair_fraction(epsilon,cell);
		CHECK(pair == 1.00_a);

		double j_0 = wannier::spread2(i,j,cell);
		CHECK(j_0 == 0.749738_a);
		double tot = wannier::spread2(i,cell);
		CHECK(tot == 2.490984_a);
		double spread = wannier::spread(i,cell);
		CHECK(spread == 1.578285_a);
		double sum = wannier::spread2(cell);
		CHECK(sum == 4.981968_a);
		double val = wannier::spread(cell);
		CHECK(val == 2.232032_a);
		auto dp = wannier::dipole(cell);
                CHECK(dp[0] == 0.000000_a);
                CHECK(dp[1] == 0.000000_a);
                CHECK(dp[2] == -2.846012_a);
  }*/	
  	
}
#endif  

