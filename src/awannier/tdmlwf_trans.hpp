/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__AWANNIER__TDMLWF_TRANS
#define INQ__AWANNIER__TDMLWF_TRANS

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
#include <awannier/jade_complex.hpp>
#include <iostream>
#include <vector> 

namespace inq {
namespace wannier {

////////////////////////////////////////////////////////////////////////////////
// CS tmp stuff to be able to test these functions
states::ks_states st(states::spin_config::UNPOLARIZED, 4.0); //will be added to inital declaration later

std::vector<std::vector<std::vector<complex>>> a = {
    {{complex(-0.68433137, -0.00000000), complex(-0.00103429,  0.10810869)},
     {complex(-0.00103429, -0.10810876), complex(-0.68104163,  0.00000000)}},

    {{complex(-0.09765152,  0.00000003), complex( 0.00727211, -0.68256214)},
     {complex( 0.00727210,  0.68256259), complex(-0.11860232, -0.00000003)}},

    {{complex(-0.68433137, -0.00000000), complex(-0.00103429,  0.10810870)},
     {complex(-0.00103429, -0.10810877), complex(-0.68104162,  0.00000000)}},

    {{complex(-0.09765152,  0.00000003), complex( 0.00727211, -0.68256215)},
     {complex( 0.00727210,  0.68256259), complex(-0.11860232, -0.00000003)}},

    {{complex(-0.68433130, -0.00000000), complex(-0.00103425,  0.10810870)},
     {complex(-0.00103453, -0.10810874), complex(-0.68104179,  0.00000005)}},

    {{complex(-0.09765150,  0.00000002), complex( 0.00727212, -0.68256221)},
     {complex( 0.00727207,  0.68256246), complex(-0.11860233, -0.00000030)}}
};
	
std::vector<std::vector<complex>> u = {
    {{complex(1.0, 0.0), complex(0.0, 0.0)},
     {complex(0.0, 0.0), complex(1.0, 0.0)}}
};

std::vector<std::vector<complex>> adiag(a.size(), std::vector<complex>(2)); // Assuming single diagonal element per input matrix

const int maxsweep = 100;
const double tol = 1.e-8;
auto nsweep = wannier::jade_complex(maxsweep,tol,a,u,adiag);
// Joint approximate diagonalization step.

////////////////////////////////////////////////////////////////////////////////
template <typename T, class vector_type1, class vector_type2> 
void compute_sincos(T n, vector_type1* f, vector_type2* fc, vector_type2* fs) { 
//void compute_sincos(const int n, const std::complex<double>* f, std::complex<double>* fc, std::complex<double>* fs) {

  // fc[i] =     0.5 * ( f[i-1] + f[i+1] )
  // fs[i] = (0.5/i) * ( f[i-1] - f[i+1] )

  // i = 0
  complex zp = f[n-1];
  complex zm = f[1];
  fc[0] = 0.5 * ( zp + zm );
  complex zdiff = zp - zm;
  fs[0] = 0.5 * complex(imag(zdiff),-real(zdiff));
  for ( int i = 1; i < n-1; i++ )
  {
    const complex zzp = f[i-1];
    const complex zzm = f[i+1];
    fc[i] = 0.5 * ( zzp + zzm );
    const complex zzdiff = zzp - zzm;
    fs[i] = 0.5 * complex(imag(zzdiff),-real(zzdiff));
  }
  // i = n-1
  zp = f[n-2];
  zm = f[0];
  fc[n-1] = 0.5 * ( zp + zm );
  zdiff = zp - zm;
  fs[n-1] = 0.5 * complex(imag(zdiff),-real(zdiff));
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
auto center(T i, const systems::cell & cell_) {
  assert(i >= 0 && i < st.num_states());
  const double cx = real(adiag[0][i]);
  const double sx = real(adiag[1][i]);
  const double cy = real(adiag[2][i]);
  const double sy = real(adiag[3][i]);
  const double cz = real(adiag[4][i]);
  const double sz = real(adiag[5][i]);
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
  double root = sqroot(total_dist);
  return fabs(root);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2>
bool overlap(T1 epsilon, T2 i, T2 j, const systems::cell & cell_) {
  // overlap: return true if the functions i and j overlap according to distance
  double x = cell_[0][0]*cell_[0][0] + cell_[0][1]*cell_[0][1] + cell_[0][2]*cell_[0][2];
  double y = cell_[1][1]*cell_[1][1] + cell_[1][2]*cell_[1][2] + cell_[1][2]*cell_[1][2];
  double z = cell_[2][2]*cell_[2][2] + cell_[2][1]*cell_[2][1] + cell_[2][2]*cell_[2][2];
  double len = sqroot(x+y+z);
  if (wannier_distance(i,j, cell_) <= epsilon || wannier_distance(i,j, cell_) >= (len - epsilon) )
      return true;  //need sqrt(a0^2 + a1^2 + a2^2) for cell diagonal distance. Diagonal dist - epsilon for pbc

  // return false if the states don't overlap
  return false;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
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
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
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
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread2(T i, T j, const systems::cell & cell) {
  assert(i >= 0 && i < st.num_states());
  assert(j >= 0 && j < 3);
  const complex c = adiag[2*j][i]; //DCY
  const complex s = adiag[2*j+1][i]; //DCY
  // Next line: M_1_PI = 1.0/pi
  auto recip = cell.reciprocal(j);
  double length = sqrt(recip[0]*recip[0]+ recip[1]*recip[1] + recip[2]*recip[2]);
  const double fac = 1.0 / length;
  return fac*fac * ( 1.0 - norm(c) - norm(s) );
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread2(T i, const systems::cell & cell) {
  assert(i >= 0 & i < st.num_states());
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
  for ( int i = 0; i < st.num_states(); i++ )
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
  for ( int i = 0; i < st.num_states(); i++ )
    sum -= 2.0 * center(i,cell);  //CS need to pass state occupations (assume fully occupied for now) How?
  return sum;
}
///////////////////////////////////////////////////////////////////

} //wannier
} //inq
#endif 

///////////////////////////////////////////////////////////////////
#ifdef INQ_AWANNIER_TDMLWF_TRANS_UNIT_TEST
#undef INQ_AWANNIER_TDMLWF_TRANS_UNIT_TEST

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

        using namespace inq;
        using namespace magnitude;
        using namespace Catch::literals;
        using Catch::Approx;

	SECTION("Wannier Centers"){

		//CS testing a 2He atom test; 20x20x20 cell, 1He at <-7,-7,-7> and 1He at <8,8,8>
		auto cell = systems::cell::cubic(20.0_b).periodic(); //CS more conventional call to a cubic cell with pbc 
		int i = 0;
		auto center = wannier::center(i, cell); 

		CHECK(center[0] == -7.000012_a);
		CHECK(center[1] == -7.000012_a);
		CHECK(center[2] == -7.000012_a); 

		int j = 1;
                auto center1 = wannier::center(j, cell);
		CHECK(center1[0] == 8.000012_a);
                CHECK(center1[1] == 8.000012_a);
                CHECK(center1[2] == 8.000012_a);
	
		auto dist = wannier::wannier_distance(i, j, cell);
		CHECK(dist == 25.980803_a);
		
		double epsilon = 5.0;
		auto overlap = wannier::overlap(epsilon, i, j, cell);
		CHECK(overlap == false);
		
		auto ols = wannier::total_overlaps(epsilon, cell);
		CHECK(ols == 0.5_a);
		auto pair = wannier::pair_fraction(epsilon,cell);
		CHECK(pair == 0.6666666_a);

		double j_0 = wannier::spread2(i,j,cell);
		CHECK(j_0 == 0.450904_a);
		double tot = wannier::spread2(i,cell);
		CHECK(tot == 1.352712_a);
		double spread = wannier::spread(i,cell);
		CHECK(spread == 1.163062_a);
		double sum = wannier::spread2(cell);
		CHECK(sum == 2.705424_a);
		double val = wannier::spread(cell);
		CHECK(val == 1.644817_a);
		auto dp = wannier::dipole(cell);
                CHECK(dp[0] == -2.000000_a);
                CHECK(dp[1] == -2.000000_a);
                CHECK(dp[2] == -2.000000_a);
  }	
  	
}
#endif  
