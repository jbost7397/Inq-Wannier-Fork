/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__WANNIER__PLANE_ROT
#define INQ__WANNIER__PLANE_ROT

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//CS specific call for plane rotation used in jade_complex

#include <inq_config.h>
#include <math/complex.hpp>
#include <vector>
#include <utils/raw_pointer_cast.hpp>

#include "FC.h"

#include <utils/profiling.hpp>

#define zrot FC_GLOBAL(zrot, ZROT) 
extern "C" void zrot(int*, inq::complex *, int*, inq::complex *, int*, std::complex<double>*, std::complex<double>*);

namespace inq {
namespace wannier {

template<class vec_type, class T>
auto plane_rot(vec_type & vector1, vec_type & vector2, T c, T s_conj){

	CALI_CXX_MARK_FUNCTION;

	// the vectors must have the same dimension 
	assert(std::get<0>(sizes(vector1)) == std::get<0>(sizes(vector2)));

	int nn = std::get<0>(sizes(vector1));
	int one = 1;

	zrot(&nn, raw_pointer_cast(vector1.data_elements()), &one, raw_pointer_cast(vector2.data_elements()), &one, &c, &s_conj);

}
}  // namespace wannier
}  // namespace inq

#endif  // INQ__WANNIER__PLANE_ROT

///////////////////////////////////////////////////////////////////

#ifdef INQ_WANNIER_PLANE_ROT_UNIT_TEST
#undef INQ_WANNIER_PLANE_ROT_UNIT_TEST

#include <gpu/array.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) { 

	SECTION("Real rotation to real vectors") {

		using namespace inq;
  		using namespace Catch::literals;
  
		gpu::array<complex,1> vector1 = {1.0, 2.0, 3.0, 4.0}; 
  		gpu::array<complex,1> vector2 = {-1.0, -2.0, -3.0, -4.0};

		std::complex<double> c1 = 0.5;
		std::complex<double> s1 = std::sqrt(3.0)/2.0;
		//complex c1 = 0.5;
		//complex s1 = sqrt(3.0)/2.0;
 
		wannier::plane_rot(vector1, vector2, c1, s1);

		CHECK(real(vector1[0]) == -0.366025_a);
		CHECK(real(vector2[0]) == -1.366025_a);
		CHECK(real(vector1[1]) == -0.732051_a);
 		CHECK(real(vector2[1]) == -2.732051_a);
		CHECK(real(vector1[2]) == -1.098076_a);
		CHECK(real(vector2[2]) == -4.098076_a);
		CHECK(real(vector1[3]) == -1.464102_a);
		CHECK(real(vector2[3]) == -5.464102_a);
		CHECK(imag(vector1[0]) == 0.0_a);
		CHECK(imag(vector2[0]) == 0.0_a);
		CHECK(imag(vector1[3]) == 0.0_a);
		CHECK(imag(vector2[3]) == 0.0_a);

  } 

	SECTION("Complex rotation to complex vectors") {

		using namespace inq;
                using namespace Catch::literals;

		gpu::array<complex,1> vector1 = {complex(1.0,2.0), complex(2.0,3.0), complex(3.0,4.0)};
		gpu::array<complex,1> vector2 = {complex(-1.0,5.0), complex(-2.0,4.0), complex(-3.0,3.0)};

		std::complex<double> c1 = 0.5;
		std::complex<double> s1 = complex(0.75,0.50);

		wannier::plane_rot(vector1, vector2, c1, s1);

		CHECK(real(vector1[0]) == -2.750_a);
                CHECK(imag(vector1[0]) == 4.250_a);
                CHECK(real(vector2[0]) == -2.250_a);
                CHECK(imag(vector2[0]) == 1.500_a);
                CHECK(real(vector1[1]) == -2.500_a);
                CHECK(imag(vector1[1]) == 3.500_a);
                CHECK(real(vector2[1]) == -4.000_a);
                CHECK(imag(vector2[1]) == 0.750_a);
                CHECK(real(vector1[2]) == -2.250_a);
                CHECK(imag(vector1[2]) == 2.750_a);
                CHECK(real(vector2[2]) == -5.750_a);
                CHECK(imag(vector2[2]) == 0.00_a);

  }

        SECTION("Real rotation to complex vectors") {

                using namespace inq;
                using namespace Catch::literals;

                gpu::array<complex,1> vector1 = {complex(1.0,2.0), complex(2.0,3.0), complex(3.0,4.0)};
                gpu::array<complex,1> vector2 = {complex(-1.0,5.0), complex(-2.0,4.0), complex(-3.0,3.0)};

                std::complex<double> c1 = 0.5;
                std::complex<double> s1 = std::sqrt(3.0)/2.0;

                wannier::plane_rot(vector1, vector2, c1, s1);

                CHECK(real(vector1[0]) == -0.366025_a);
                CHECK(imag(vector1[0]) == 5.330127_a);
                CHECK(real(vector2[0]) == -1.366025_a);
                CHECK(imag(vector2[0]) == 0.76795_a);
                CHECK(real(vector1[1]) == -0.732051_a);
                CHECK(imag(vector1[1]) == 4.964102_a);
                CHECK(real(vector2[1]) == -2.732051_a);
                CHECK(imag(vector2[1]) == -0.598076_a);
                CHECK(real(vector1[2]) == -1.098076_a);
                CHECK(imag(vector1[2]) == 4.598076_a);
                CHECK(real(vector2[2]) == -4.098076_a);
                CHECK(imag(vector2[2]) == -1.964102_a);
  }
  ///////////////////////////////////////////////////////////////////
}
#endif
