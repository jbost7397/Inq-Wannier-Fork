/* -*- indent-tabs-mode: t -*- */

#ifndef SHARMONIC
#define SHARMONIC

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifdef HAVE_SHARMONIC_CONFIG_H
#include <sharmonic_config.h>
#endif

#ifdef __cplusplus
#include <cmath>
#include <complex>
#include <cassert>
#include <array>

#ifdef HAVE_BOOST_SPHERICAL_HARMONIC
#include <boost/math/special_functions/spherical_harmonic.hpp>
#endif

#define ComplexConstr(real, imag) Complex{real, imag}

#define SH_FUNC(func) func
#else

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <stdlib.h>

#define Complex double complex
#define ComplexConstr(real, imag) real + I*imag

#define SH_FUNC(func) sharmonic_##func
#endif


#define SHARMONIC_MAX_L 9

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SH_GPU_FUNCTION __host__ __device__
#else
#define SH_GPU_FUNCTION
#endif

#ifdef __cplusplus
namespace sharmonic {
namespace internal {
#endif
///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION void SH_FUNC(swap)(double * aa, double  * bb) {
	double cc = *aa;
	*aa = *bb;
	*bb = cc;
};

SH_GPU_FUNCTION double SH_FUNC(norm)(double xx, double yy, double zz) {
	xx = fabs(xx);
	yy = fabs(yy);
	zz = fabs(zz);

	if (xx < yy) SH_FUNC(swap)(&xx, &yy);
	if (yy < zz) SH_FUNC(swap)(&yy, &zz);
	if (xx < yy) SH_FUNC(swap)(&xx, &yy);

	if(xx == 0.0) return 0.0;
	
	yy /= xx;
	zz /= xx;

	return xx*sqrt(1.0 + yy*yy + zz*zz);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION double SH_FUNC(normalize)(double * xx, double * yy, double * zz) {
	double nn = SH_FUNC(norm)(*xx, *yy, *zz);
	if(nn == 0.0) return 0.0;
	
  *xx /= nn;
	*yy /= nn;
	*zz /= nn;
	return nn;
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION void SH_FUNC(angular_to_normalized_cartesian)(double theta, double phi, double * xx, double * yy, double * zz) {
	*xx = sin(theta)*cos(phi);
	*yy = sin(theta)*sin(phi);
	*zz = cos(theta);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION void SH_FUNC(normalized_cartesian_to_angular)(double xx, double yy, double zz, double * theta, double * phi) {
	*phi = atan2(yy, xx);
	*theta = acos(zz);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION void SH_FUNC(cartesian_to_angular)(double xx, double yy, double zz, double * rr, double * theta, double * phi) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	*rr = SH_FUNC(normalize)(&xx, &yy, &zz);

	if(*rr < 1.0e-15) {
		*theta = 0.0;
		*phi   = 0.0;
		return;
	}

	SH_FUNC(normalized_cartesian_to_angular)(xx, yy, zz, theta, phi);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION double SH_FUNC(powsign)(int mm) {
	if(mm%2 == 0) return 1.0;
	return -1.0;
}

///////////////////////////////////////////////////////////////////

//this functions converts an mm > 0 complex harmonic into the -mm complex harmonic
#ifdef __cplusplus
template <typename Complex>
#endif
SH_GPU_FUNCTION Complex SH_FUNC(neg)(int mm, double re, double im) {
	if(mm >= 0) return ComplexConstr(re, im);
	if(abs(mm)%2 == 0) return ComplexConstr(re, -im);
	return ComplexConstr(-re, im);
}

SH_GPU_FUNCTION double rpair(double const a, double const b) {
	return (a - b)*(a + b);
}

#ifdef __cplusplus
} //namespace internal
#endif

///////////////////////////////////////////////////////////////////

#if defined(__cplusplus) && defined(HAVE_SPH_LEGENDRE)
namespace ref_std {

auto angular_real(int const ll, int const mm, double theta, double phi) {
	
	// This is the general implementation using the C++ standard library (that doesn't work on the GPU and it slower)
	
	auto sh = std::sph_legendre(ll, abs(mm), theta);
	if(mm > 0){
		return internal::powsign(mm)*std::sqrt(2.0)*cos(mm*phi)*sh;
	} else if (mm == 0) {
		return sh;
	} else {
		return internal::powsign(mm)*std::sqrt(2.0)*sin(-mm*phi)*sh;
	}
}

//////////////////////////////////////////////////////////////////

auto normalized_cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_std::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

auto cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_std::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
Complex angular_complex(int const ll, int const mm, double theta, double phi) {
	using namespace std::complex_literals;

	// This is the general implementation using the C++ standard library (that doesn't work on the GPU and it slower)
	auto leg = std::sph_legendre(ll, abs(mm), theta);
	if(mm < 0) leg *= internal::powsign(mm); //only positive mm values are supported
	return std::polar(leg, mm*phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
Complex normalized_cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_std::angular_complex<Complex>(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
Complex cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_std::angular_complex<Complex>(ll, mm, theta, phi);
}

} //namespace ref_std
#endif

///////////////////////////////////////////////////////////////////

#if defined(__cplusplus) && defined(HAVE_BOOST_SPHERICAL_HARMONIC)
namespace ref_boost {

auto angular_real(int const ll, int const mm, double theta, double phi) {
	
	// this is a reference implementation using boost.

	if(phi < 0.0) phi += 2.0*M_PI; //boost wants phi in the [0, 2pi) range

	auto sh = boost::math::spherical_harmonic(ll, abs(mm), theta, phi);
	if(mm < 0) return sqrt(2.0)*pow(-1.0, mm)*imag(sh);
	if(mm > 0) return sqrt(2.0)*pow(-1.0, mm)*real(sh);
	return real(sh);
}

///////////////////////////////////////////////////////////////////

auto normalized_cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_boost::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

auto cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_boost::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
Complex angular_complex(int const ll, int const mm, double theta, double phi) {
	
	// this is a reference implementation using boost.
	if(phi < 0.0) phi += 2.0*M_PI; //boost wants phi in the [0, 2pi) range
	return boost::math::spherical_harmonic(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
Complex normalized_cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_boost::angular_complex<Complex>(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
Complex cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_boost::angular_complex<Complex>(ll, mm, theta, phi);
}

} //namespace ref_boost
#endif

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION double SH_FUNC(normalized_cartesian_real)(int const ll, int const mm, double x, double y, double z) {
	assert(ll >= 0);
	assert(ll <= SHARMONIC_MAX_L);
	assert(abs(mm) <= ll);

#ifdef __cplusplus
	using internal::rpair;
	using internal::powsign;
#endif
	
	if(mm > 0 && mm%2 == 1) return SH_FUNC(powsign)(mm/2)*SH_FUNC(normalized_cartesian_real)(ll, -mm, -y, x, z); // 90 degree rotation
	if(mm == 2) return SH_FUNC(normalized_cartesian_real)(ll, -mm, (x - y)/sqrt(2), (x + y)/sqrt(2), z);          // 45 degree rotation

	// Functions taken from:
	//   [1] https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
	//   [2] https://www.doiserbia.nb.rs/img/doi/1450-698X/2009/1450-698X0979107M.pdf
	//   [3] https://github.com/elerac/sh_table/blob/main/docs/table.txt

	double z2 = z*z;
	
	switch(ll){
	case 0:
		return 1.0/sqrt(4*M_PI);
	case 1:
		switch(mm){
		case -1: return sqrt(3.0/(4*M_PI))*y;
		case	0: return sqrt(3.0/(4*M_PI))*z;
		default: return 0.0;
		}
	case 2:
		switch(mm){		
		case -2: return 1.0/2.0*sqrt(15/M_PI)*x*y;
		case -1: return 1.0/2.0*sqrt(15/M_PI)*y*z;
		case	0: return 1.0/4.0*sqrt(5/M_PI)*rpair(sqrt(3)*z, 1);
		default: return 0.0;
		}
	case 3:
		switch(mm){		
		case -3: return 1.0/8.0*sqrt(70/M_PI)*y*rpair(sqrt(3)*x, y);
		case -2: return 1.0/2.0*sqrt(105/M_PI)*x*y*z;
		case -1: return 1.0/8.0*sqrt(42/M_PI)*y*rpair(sqrt(5)*z, 1);
		case	0: return 1.0/4.0*sqrt(7/M_PI)*rpair(sqrt(5)*z, sqrt(3))*z;
		default: return 0.0;
		}
	case 4:
		switch(mm){
		case -4: return 3.0/4.0*sqrt(35/M_PI)*x*y*rpair(x, y);
		case -3: return 3.0/8.0*sqrt(70/M_PI)*y*rpair(sqrt(3)*x, y)*z;
		case -2: return 3.0/4.0*sqrt(5/M_PI)*x*y*rpair(sqrt(7)*z, 1);
		case -1: return 3.0/8.0*sqrt(10/M_PI)*y*rpair(sqrt(7)*z, sqrt(3))*z;
		case	0: return 105.0/16.0*sqrt(1/M_PI)*rpair(sqrt((3 + (2*sqrt(6/5.0)))/7.0), z)*rpair(sqrt((3 - (2*sqrt(6/5.0)))/7.0), z);
		case	4: return 3.0/16.0*sqrt(35/M_PI)*rpair((sqrt(2) + 1)*x, y)*rpair((sqrt(2) - 1)*x, y);
		default: return 0.0;
		}
	case 5:
		switch(mm){		
		case -5: return 30.0/64.0*sqrt(154/M_PI)*y*rpair(sqrt(1 - 2/sqrt(5))*y, x)*rpair(sqrt(1 + 2/sqrt(5))*y, x);
		case -4: return 3.0/4.0*sqrt(385.0/M_PI)*rpair(x, y)*y*z*x;
		case -3: return 1.0/32.0*sqrt(770.0/M_PI)*rpair(1, 3*z)*rpair(y, sqrt(3)*x)*y;
		case -2: return 1.0/4.0*sqrt(1155.0/M_PI)*y*rpair(sqrt(3)*z, 1)*z*x;
		case -1: return 21.0/16.0*sqrt(165.0/M_PI)*y*rpair(sqrt((7 - 2*sqrt(7))/21), z)*rpair(sqrt((7 + 2*sqrt(7))/21), z);
		case	0: return 63.0/16.0*sqrt(11.0/M_PI)*z*rpair(sqrt(5 - 2*sqrt(10/7.0))/3, z)*rpair(sqrt(5 + 2*sqrt(10/7.0))/3, z);
		case	4: return 3.0/16.0*sqrt(385.0/M_PI)*z*rpair((sqrt(2) + 1)*x, y)*rpair((sqrt(2) - 1)*x, y);
		default: return 0.0;
		}
	case 6:
		switch(mm){
		case -6: return 1.0/32.0*sqrt(6006/M_PI)*x*y*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y);
		case -5: return 3.0/32*sqrt(2002/M_PI)*y*z*rpair(sqrt(5 - 2*sqrt(5))*x, y)*rpair(sqrt(5 + 2*sqrt(5))*x, y);			
		case -4: return 3.0/8.0*sqrt(91/M_PI)*x*y*rpair(x, y)*rpair(sqrt(11)*z, 1);
		case -3: return 1.0/32.0*sqrt(2730/M_PI)*y*z*rpair(sqrt(3)*x, y)*rpair(sqrt(11)*z, sqrt(3));
		case -2: return 33.0/32.0*sqrt(2730.0/M_PI)*x*y*rpair(sqrt((3 + 4/sqrt(3))/11), z)*rpair(sqrt((3 - 4/sqrt(3))/11), z);
		case -1: return 33.0/16.0*sqrt(273.0/M_PI)*z*rpair(sqrt((15 - 2*sqrt(15))/33), z)*y*rpair(sqrt((15 + 2*sqrt(15))/33), z);
		case  0: return 1.0/32*sqrt(13.0/M_PI)*(z2*((231*z2 - 315)*z2 + 105) - 5);
		case  4: return 3.0/32.0*sqrt(91.0/M_PI)*rpair(sqrt(11.0)*z, 1.0)*rpair((sqrt(2) + 1)*x, y)*rpair((sqrt(2) - 1)*x, y);
		case  6: return 1.0/64.0*sqrt(6006/M_PI)*rpair(x, y)*rpair((sqrt(3) - 2)*x, y)*rpair((sqrt(3) + 2)*x, y);
		default: return 0.0;
		}
	case 7:
		switch(mm){
		case -7: {
			//ANALYTIC: return 3.0/64.0*sqrt(715/M_PI)*y*(7.0*x6 - 35.0*x4*y2 + 21.0*x2*y4 - y6);
			//ROOTS:    https://www.wolframalpha.com/input?i=roots+7.0*x6+-+35.0*x4*y2+%2B+21.0*x2*y4+-+y6

			double r1 = 0.481574618807528644332162353056970575219;
			double r2 = 1.253960337662703837570910978336464443221;
			double r3 = 4.3812862675348230724046890850326954441502;
			return 3.0/64.0*sqrt(715/M_PI)*y*rpair(r1*x, y)*rpair(r2*x, y)*rpair(r3*x, y);
		}
		case -6: return 3.0/32.0*sqrt(10010/M_PI)*x*y*z*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y);
		case -5: return 3.0/64.0*sqrt(385/M_PI)*y*rpair(sqrt(13)*z, 1)*rpair(sqrt(5 - 2*sqrt(5))*x,y)*rpair(sqrt(5 + 2*sqrt(5))*x, y);
		case -4: return 3.0/8.0*sqrt(385/M_PI)*x*y*z*rpair(x, y)*rpair(sqrt(13.0)*z, sqrt(3));
		case -3: return 429.0/64.0*sqrt(35/M_PI)*y*rpair(sqrt(3)*x, y)*rpair(sqrt((33 - 2*sqrt(165))/143), z)*rpair(sqrt((33 + 2*sqrt(165))/143), z);
		case -2: return 429.0/32.0*sqrt(70/M_PI)*x*y*z*rpair(sqrt((55 - 4*sqrt(55))/143), z)*rpair(sqrt((55 + 4*sqrt(55))/143), z);
		case -1: return 1.0/64.0*sqrt(105/M_PI)*y*(z2*((429*z2 - 495)*z2 + 135) - 5);
		case  0: return 1.0/32.0*sqrt(15/M_PI)*z*(-35 + z2*(315 + z2*(-693 + 429*z2)));
		case  4: return 3.0/32.0*sqrt(385/M_PI)*z*rpair(sqrt(13)*z, sqrt(3))*rpair((1 + sqrt(2))*x, y)*rpair((1 - sqrt(2))*x, y);
		case  6: return 3.0/64.0*sqrt(10010/M_PI)*z*rpair(x, y)*rpair((sqrt(3) - 2)*x, y)*rpair((sqrt(3) + 2)*x, y);
		default: return 0.0;
		}
	case 8:
		switch(mm){
		case -8: return 12.0/128.0*sqrt(12155/M_PI)*x*y*rpair(x, y)*rpair((1 + sqrt(2))*x, y)*rpair((1 - sqrt(2))*x, y);
		case -7: {
			// ANALYTIC: return 3.0/64.0*sqrt(12155/M_PI)*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6);
			// ROOTS:    https://www.wolframalpha.com/input?i=roots+%287+x%5E6+-+35+x%5E4+y%5E2+%2B+21+x%5E2+y%5E4+-+1+y%5E6%29
			double r1 = 0.481574618807528644332162353056970575219;
			double r2 = 1.253960337662703837570910978336464443221;
			double r3 = 4.3812862675348230724046890850326954441502;

			return 3.0/64.0*sqrt(12155/M_PI)*y*z*rpair(r1*x, y)*rpair(r2*x, y)*rpair(r3*x, y);
		}
		case -6: return 1.0/64.0*sqrt(14586/M_PI)*x*y*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y)*rpair(sqrt(15)*z, 1);
		case -5: return 3.0/64.0*sqrt(17017/M_PI)*y*z*rpair(sqrt(5)*z, 1)*rpair(sqrt(5 + 2*sqrt(5))*x, y)*rpair(sqrt(5 - 2*sqrt(5))*x, y);
		case -4: return 195.0/32.0*sqrt(1309/M_PI)*x*y*rpair(x, y)*rpair(sqrt((13 - 2*sqrt(26))/65), z)*rpair(sqrt((13 + 2*sqrt(26))/65), z);
		case -3: return 39.0/64.0*sqrt(19635/M_PI)*y*z*rpair(sqrt(3)*x, y)*rpair(sqrt((1 - 2/sqrt(13))/3), z)*rpair(sqrt((1 + 2/sqrt(13))/3), z);
		case -2: return 3.0/64.0*sqrt(1190/M_PI)*x*y*(z2*((143*z2 - 143)*z2 + 33) - 1);
		case -1: {
			//ANALYTIC: return 3.0/64.0*sqrt(17/M_PI)*y*z*(z2*((715*z2 - 1001)*z2 + 385) - 35);
			//ROOTS:    https://www.wolframalpha.com/input?i=roots+z2*%28%28715*z2+-+1001%29*z2+%2B+385%29+-+35
			double r1 = 0.36311746382617815871075206870865921;
			double r2 = 0.67718627951073775344588542709134245;
			double r3 = 0.89975799541146015731234524441833796;
			return 2145.0/64.0*sqrt(17/M_PI)*y*z*rpair(z, r1)*rpair(z, r2)*rpair(z, r3);
		}
		case  0: {
			// ANALYTIC: return 1.0/256.0*sqrt(17/M_PI)*((z2*((6435*z2 - 12012)*z2 + 6930) - 1260)*z2 + 35);
			// ROOTS:    https://www.wolframalpha.com/input?i=roots+%28%28z2*%28%286435*z2+-+12012%29*z2+%2B+6930%29+-+1260%29*z2+%2B+35%29  
			double r1 = 0.18343464249564980493947614236018398;
			double r2 = 0.52553240991632898581773904918924635;
			double r3 = 0.79666647741362673959155393647583044;
			double r4 = 0.96028985649753623168356086856947299;
			return 6435.0/256.0*sqrt(17/M_PI)*rpair(r1, z)*rpair(r2, z)*rpair(r3, z)*rpair(r4, z);
		}
		case  4: return 195.0/128.0*sqrt(1309/M_PI)*rpair((sqrt(2) + 1)*x, y)*rpair((sqrt(2) - 1)*x, y)*rpair(sqrt((13 + 2*sqrt(26))/65), z)*rpair(sqrt((13 - 2*sqrt(26))/65), z);
		case  6: return 1.0/128.0*sqrt(14586/M_PI)*rpair(x, y)*rpair(y, (sqrt(3.0) - 2.0)*x)*rpair(y, (sqrt(3.0) + 2.0)*x)*rpair(sqrt(15)*z, 1);
		case  8: return 3.0/256.0*sqrt(12155/M_PI)
				*rpair((1 - sqrt(2) - sqrt(4 - 2*sqrt(2)))*x, y)*rpair((1 - sqrt(2) + sqrt(4 - 2*sqrt(2)))*x, y)
				*rpair((1 + sqrt(2) + sqrt(4 + 2*sqrt(2)))*x, y)*rpair((1 + sqrt(2) - sqrt(4 + 2*sqrt(2)))*x, y);
		default: return 0.0;
		}
	case 9:
		switch(mm){		
		case -9: {
			// ANALYTIC: return 1.0/512.0*sqrt(461890/M_PI)*y*(sqrt(3)*x - y)*(sqrt(3)*x + y)*(3*x6 - 27*x4*y2 + 33*x2*y4 - y6);
			// Roots:    https://www.wolframalpha.com/input?i=roots+3*x6+-+27*x4*y2+%2B+33*x2*y4+-+y6
			double r1 = 0.363970234266202361351047882776834043890;
			double r2 = 0.839099631177280011763127298123181364687;
			double r3 = 5.6712818196177095309944184398639644216254;
			
			return 1.0/512.0*sqrt(461890/M_PI)*y*rpair(y, sqrt(3)*x)*rpair(y, r1*x)*rpair(y, r2*x)*rpair(y, r3*x);
		}
		case -8: return 3.0/32.0*sqrt(230945/M_PI)*x*y*z*rpair(x, y)*rpair((1 + sqrt(2))*x, y)*rpair((1 - sqrt(2))*x, y);
		case -7: {
			//ANALYTIC: return -3.0/512.0*sqrt(27170/M_PI)*y*(1 - sqrt(17)*z)*(1 + sqrt(17)*z)*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6);
			//ROOTS:    https://www.wolframalpha.com/input?i=roots+7*x6+-+35*x4*y2+%2B+21*x2*y4+-+y6
			double r1 = 0.481574618807528644332162353056970575219;
			double r2 = 1.253960337662703837570910978336464443221;
			double r3 = 4.3812862675348230724046890850326954441502;
			return 3.0/512.0*sqrt(27170/M_PI)*y*rpair(1, sqrt(17)*z)*rpair(y, r1*x)*rpair(y, r2*x)*rpair(y, r3*x);
		}
		case -6: return 1.0/64.0*sqrt(81510/M_PI)*x*y*z*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y)*rpair(sqrt(17)*z, sqrt(3));
		case -5: return 255.0/256.0*sqrt(5434/M_PI)*y*rpair(sqrt(5 - 2*sqrt(5))*x, y)*rpair(sqrt(5 + 2*sqrt(5))*x, y)*rpair(sqrt((15 - 2*sqrt(35))/85), z)*rpair(sqrt((15 + 2*sqrt(35))/85), z);
		case -4: return 51.0/32.0*sqrt(95095/M_PI)*x*y*z*rpair(x, y)*rpair(sqrt((5 - 2*sqrt(2))/17), z)*rpair(sqrt((5 + 2*sqrt(2))/17), z);
		case -3: return 1.0/256.0*sqrt(43890/M_PI)*y*rpair(sqrt(3)*x, y)*(z2*((221*z2 - 195)*z2 + 39) - 1);
		case -2: return -3.0/64.0*sqrt(2090/M_PI)*x*y*z*((z2*(273 - 221*z2) - 91)*z2 + 7);
		case -1: {
			//			ANALYTIC:   return 3.0/256.0*sqrt(95/M_PI)*y*((z2*((2431*z2 - 4004)*z2 + 2002) - 308)*z2 + 7);
			//			ROOTS:      https://www.wolframalpha.com/input?i=roots+%28z2*%28%282431*z2+-+4004%29*z2+%2B+2002%29+-+308%29*z2+%2B+7
			double r1 = 0.16527895766638702462621976595817353;
			double r2 = 0.47792494981044449566117509273125800;
			double r3 = 0.73877386510550507500310617485983073;
			double r4 = 0.91953390816645881382893266082233813;
			return 7293.0/256.0*sqrt(95/M_PI)*y*rpair(r1, z)*rpair(r2, z)*rpair(r3, z)*rpair(r4, z);
		}
		case  0: {
			//ANALYTIC: return 1.0/256.0*sqrt(19/M_PI)*z*((z2*(rpair(sqrt(12155)*z, sqrt(25740))*z2 + 18018) - 4620)*z2 + 315);
			//ROOTS:    https://www.wolframalpha.com/input?i=roots+%28z2*%28%2812155*z2+-+25740%29*z2+%2B+18018%29+-+4620%29*z2+%2B+315

			double r1 = 0.32425342340380892903853801464333661;
			double r2 = 0.61337143270059039730870203934147418;
			double r3 = 0.83603110732663579429942978806973488;
			double r4 = 0.96816023950762608983557620290367287;

			return 12155.0/256.0*sqrt(19/M_PI)*z*rpair(r1, z)*rpair(r2, z)*rpair(r3, z)*rpair(r4, z);
		}
		case  4: return 51.0/128.0*sqrt(95095/M_PI)*z*rpair((sqrt(2) - 1)*x, y)*rpair((sqrt(2) + 1)*x, y)*rpair(sqrt((5 - 2*sqrt(2))/17), z)*rpair(sqrt((5 + 2*sqrt(2))/17), z);
		case  6: return 1.0/128.0*sqrt(81510/M_PI)*z*rpair(x, y)*rpair(sqrt(17)*z, sqrt(3))*rpair((sqrt(3) - 2)*x, y)*rpair((sqrt(3) + 2)*x, y);
		case  8: return 3.0/256.0*sqrt(230945/M_PI)*z
				*rpair((1 + sqrt(2) + sqrt(2*(2 + sqrt(2))))*x, y)*rpair((1 - sqrt(2) - sqrt(2*(2 - sqrt(2))))*x, y)
				*rpair((1 + sqrt(2) - sqrt(2*(2 + sqrt(2))))*x, y)*rpair((1 - sqrt(2) + sqrt(2*(2 - sqrt(2))))*x, y);


		default: return 0.0;
		}
	default: return 0.0;
	}
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION double SH_FUNC(cartesian_real)(int const ll, int const mm, double xx, double yy, double zz) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double norm = SH_FUNC(normalize)(&xx, &yy, &zz);
	if(norm < 1.0e-15) return 0.0;
	return SH_FUNC(normalized_cartesian_real)(ll, mm, xx, yy, zz);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION double SH_FUNC(angular_real)(int const ll, int const mm, double theta, double phi) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);

	double xx, yy, zz;
	SH_FUNC(angular_to_normalized_cartesian)(theta, phi, &xx, &yy, &zz);
	return SH_FUNC(normalized_cartesian_real)(ll, mm, xx, yy, zz);;
}

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
#endif
SH_GPU_FUNCTION Complex SH_FUNC(normalized_cartesian_complex)(int const ll, int const mm, double xx, double yy, double zz) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	if(mm == 0) return SH_FUNC(normalized_cartesian_real)(ll, mm, xx, yy, zz);
	double factor = SH_FUNC(powsign)(abs(mm))*0.5*sqrt(2.0);
	double re = factor*SH_FUNC(normalized_cartesian_real)(ll, abs(mm), xx, yy, zz);
	double im = factor*SH_FUNC(normalized_cartesian_real)(ll, -abs(mm), xx, yy, zz);

#ifdef __cplusplus
	return SH_FUNC(neg)<Complex>(mm, re, im);
#else
	return SH_FUNC(neg)(mm, re, im);
#endif
	
}

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
#endif
SH_GPU_FUNCTION Complex SH_FUNC(cartesian_complex)(int const ll, int const mm, double xx, double yy, double zz) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);

	double norm = SH_FUNC(normalize)(&xx, &yy, &zz);
	if(norm < 1.0e-15) return 0.0;

#ifdef __cplusplus
	return SH_FUNC(normalized_cartesian_complex)<Complex>(ll, mm, xx, yy, zz);
#else
	return SH_FUNC(normalized_cartesian_complex)(ll, mm, xx, yy, zz);
#endif
	
}

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
#endif
SH_GPU_FUNCTION Complex SH_FUNC(angular_complex)(int const ll, int const mm, double theta, double phi) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double xx, yy, zz;
	SH_FUNC(angular_to_normalized_cartesian)(theta, phi, &xx, &yy, &zz);

#ifdef __cplusplus
	return SH_FUNC(normalized_cartesian_complex)<Complex>(ll, mm, xx, yy, zz);
#else
	return SH_FUNC(normalized_cartesian_complex)(ll, mm, xx, yy, zz);
#endif
}

#ifdef __cplusplus

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus

template <typename Complex>
#endif
struct spinor {
#ifdef __cplusplus
	SH_GPU_FUNCTION spinor(Complex const & s0, Complex const & s1){
		components[0] = s0;
		components[1] = s1;
	}
	SH_GPU_FUNCTION auto & operator[](int index) const{
		return components[index];
	}
	SH_GPU_FUNCTION auto & operator[](int index){
		return components[index];
	}
private:
#endif	
	Complex components[2];
};


///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
auto
#else
spinor
#endif
SH_GPU_FUNCTION SH_FUNC(normalized_cartesian_spinor)(int const twice_j, int const twice_mj, int const twice_s, double xx, double yy, double zz) {

	// These come from https://en.wikipedia.org/wiki/Spinor_spherical_harmonics
	// we use that '2*ll = twice_jj - sgn' to simplify the expressions
	assert(twice_j%2 != 0);
	assert(twice_mj%2 != 0);
	assert(abs(twice_mj) <= twice_j);
	assert(abs(twice_s) == 1);
	
	int ll = (twice_j - twice_s)/2;
	double sgn = twice_j - 2*ll;
	double den = 2*ll + 1;
	double cc0 = sgn*sqrt((ll + sgn*twice_mj/2.0 + 0.5)/den);
	double cc1 =     sqrt((ll - sgn*twice_mj/2.0 + 0.5)/den);
	int ml0 = (twice_mj - 1)/2;
	int ml1 = (twice_mj + 1)/2;

	Complex s0 = 0.0;
	Complex s1 = 0.0;
	if(abs(ml0) <= ll) s0 = cc0*sharmonic::cartesian_complex<Complex>(ll, ml0, xx, yy, zz);
	if(abs(ml1) <= ll) s1 = cc1*sharmonic::cartesian_complex<Complex>(ll, ml1, xx, yy, zz);
	
	return spinor<Complex>{s0, s1};
}

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
auto
#else
spinor
#endif
SH_GPU_FUNCTION SH_FUNC(cartesian_spinor)(int const twice_j, int const twice_mj, int const twice_s, double xx, double yy, double zz) {

#ifdef __cplusplus
	using namespace internal;
#endif

	int ll = (twice_j - twice_s)/2;
	double norm = SH_FUNC(normalize)(&xx, &yy, &zz);
	if(ll != 0 and norm < 1.0e-15) return spinor<Complex>{0.0, 0.0};

#ifdef __cplusplus
	return SH_FUNC(normalized_cartesian_spinor)<Complex>(twice_j, twice_mj, twice_s, xx, yy, zz);
#else
	return SH_FUNC(normalized_cartesian_spinor)(twice_j, twice_mj, twice_s, xx, yy, zz);
#endif
	
}

///////////////////////////////////////////////////////////////////

template <typename Vector3 = std::array<double, 3>>
SH_GPU_FUNCTION auto SH_FUNC(normalized_cartesian_real)(int const ll, int const mm, Vector3 const & vec3) {
	return SH_FUNC(normalized_cartesian_real)(ll, mm, vec3[0], vec3[1], vec3[2]);
}

///////////////////////////////////////////////////////////////////

template <typename Vector3 = std::array<double, 3>>
SH_GPU_FUNCTION auto SH_FUNC(cartesian_real)(int const ll, int const mm, Vector3 const & vec3) {
	return SH_FUNC(cartesian_real)(ll, mm, vec3[0], vec3[1], vec3[2]);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>, typename Vector3 = std::array<double, 3>>
SH_GPU_FUNCTION auto SH_FUNC(normalized_cartesian_complex)(int const ll, int const mm, Vector3 const & vec3) {
	return SH_FUNC(normalized_cartesian_complex)<Complex>(ll, mm, vec3[0], vec3[1], vec3[2]);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>, typename Vector3 = std::array<double, 3>>
SH_GPU_FUNCTION auto SH_FUNC(cartesian_complex)(int const ll, int const mm, Vector3 const & vec3) {
	return SH_FUNC(cartesian_complex)<Complex>(ll, mm, vec3[0], vec3[1], vec3[2]);
}
///////////////////////////////////////////////////////////////////
#endif


#ifdef __cplusplus
} //namespace sharmonic
#endif

#endif
