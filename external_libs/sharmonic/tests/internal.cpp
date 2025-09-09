/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <sharmonic.h>

#include <complex>
#include <cassert>

#include <iostream>
#include <catch2/catch_all.hpp>
#include <fstream>

TEST_CASE("internal", "[internal]") {
	using namespace Catch::literals;
	using Catch::Approx;

	using namespace sharmonic;

	SECTION("normalize") {

		{
			auto xx =  3.0;
			auto yy = -5.0;
			auto zz =  4.0;
			CHECK(internal::normalize(&xx, &yy, &zz) == 7.0710678119_a);
			CHECK(internal::normalize(&xx, &yy, &zz) == 1.0_a);
		}

		{
			auto xx =  3.0e-15;
			auto yy = -5.0e-15;
			auto zz =  4.0e-15;
			CHECK(internal::normalize(&xx, &yy, &zz) == 7.0710678119e-15_a);
			CHECK(internal::normalize(&xx, &yy, &zz) == 1.0_a);
		}
		
		{
			auto xx =  3.0e-20;
			auto yy = -5.0e-20;
			auto zz =  4.0e-20;
			CHECK(internal::normalize(&xx, &yy, &zz) == 7.0710678119e-20_a);
			CHECK(internal::normalize(&xx, &yy, &zz) == 1.0_a);
		}
		
		{
			auto xx =  0.0;
			auto yy =  0.0;
			auto zz =  0.0;
			CHECK(internal::normalize(&xx, &yy, &zz) == 0.0_a);
		}
		
	}

	SECTION("cartesian/angular conversion") {

		{
			double xx, yy, zz;
			internal::angular_to_normalized_cartesian(M_PI/2, 0.0, &xx, &yy, &zz);
			CHECK(xx == 1.0_a);
			CHECK(yy == Approx(0.0).margin(1e-12));
			CHECK(zz == Approx(0.0).margin(1e-12));
			double rr, theta, phi;
			internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
			CHECK(rr    == 1.0_a);			
			CHECK(theta == Approx(M_PI/2));
			CHECK(phi   == Approx(0.0).margin(1e-12));
		}

		{
			double xx, yy, zz;
			internal::angular_to_normalized_cartesian(0.0, 0.0, &xx, &yy, &zz);
			CHECK(xx == Approx(0.0).margin(1e-12));
			CHECK(yy == Approx(0.0).margin(1e-12));
			CHECK(zz == 1.0_a);
			double rr, theta, phi;
			internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
			CHECK(rr    == 1.0_a);			
			CHECK(theta == Approx(0.0).margin(1e-12));
			CHECK(phi   == Approx(0.0).margin(1e-12));
		}

		{
			double xx, yy, zz;
			internal::angular_to_normalized_cartesian(1.8, 4.5, &xx, &yy, &zz);
			CHECK(xx == -0.2052829899_a);
			CHECK(yy == -0.9519653892_a);
			CHECK(zz == -0.2272020947_a);
			double rr, theta, phi;			
			internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
			CHECK(rr    == 1.0_a);			
			CHECK(theta == 1.8_a);
			CHECK(phi   == Approx(4.5 - 2.0*M_PI));
		}

		{
			double rr, theta, phi;
			internal::cartesian_to_angular(-3.0, 4.0, -1.0, &rr, &theta, &phi);
			CHECK(rr    == Approx(sqrt(26.0)));
			CHECK(theta == 1.7681918866_a);
			CHECK(phi   == 2.2142974356_a);
			double xx, yy, zz;
			internal::angular_to_normalized_cartesian(theta, phi, &xx, &yy, &zz);
			CHECK(xx == Approx(-3.0/sqrt(26.0)));
			CHECK(yy == Approx( 4.0/sqrt(26.0)));
			CHECK(zz == Approx(-1.0/sqrt(26.0)));
		}

		{
			double rr, theta, phi;
			internal::cartesian_to_angular(0.0, 0.0, 0.0, &rr, &theta, &phi);
			CHECK(rr    == Approx(0.0).margin(1e-12));
			CHECK(theta == Approx(0.0).margin(1e-12));
			CHECK(phi   == Approx(0.0).margin(1e-12));
			double xx, yy, zz;
			internal::angular_to_normalized_cartesian(theta, phi, &xx, &yy, &zz);
			CHECK(xx == Approx(0.0).margin(1e-12));
			CHECK(yy == Approx(0.0).margin(1e-12));
			CHECK(zz == 1.0_a);
		}
		
	}

#ifdef HAVE_SPH_LEGENDRE
	SECTION("reference std real"){
		CHECK(ref_std::cartesian_real(3, -3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -3, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -3, 0.0, 3.0, 0.0) == -0.5900435899_a);
		CHECK(ref_std::cartesian_real(3, -3, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -3, 0.5, 1.5, 2.5) == -0.0512925789_a);
		
		CHECK(ref_std::cartesian_real(3, -2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -2, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -2, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -2, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -2, 0.5, 1.5, 2.5) == 0.2094010765_a);
		
		CHECK(ref_std::cartesian_real(3, -1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -1, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -1, 0.0, 3.0, 0.0) == -0.4570457995_a);
		CHECK(ref_std::cartesian_real(3, -1, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3, -1, 0.5, 1.5, 2.5) == 0.5959659117_a);
		
		CHECK(ref_std::cartesian_real(3,	0, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	0, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	0, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	0, 0.0, 0.0, 2.0) == 0.7463526652_a);
		CHECK(ref_std::cartesian_real(3,	0, 0.5, 1.5, 2.5) == 0.1802237516_a);
		
		CHECK(ref_std::cartesian_real(3,	1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	1, 2.0, 0.0, 0.0) == -0.4570457995_a);
		CHECK(ref_std::cartesian_real(3,	1, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	1, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	1, 0.5, 1.5, 2.5) == 0.1986553039_a);

		CHECK(ref_std::cartesian_real(3,	2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	2, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	2, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	2, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	2, 0.5, 1.5, 2.5) == -0.2792014354_a);

		CHECK(ref_std::cartesian_real(3,	3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	3, 2.0, 0.0, 0.0) == 0.5900435899_a);
		CHECK(ref_std::cartesian_real(3,	3, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	3, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(ref_std::cartesian_real(3,	3, 0.5, 1.5, 2.5) == -0.0740892806_a);
	}

	SECTION("reference std complex"){
		
		CHECK(real(ref_std::cartesian_complex(5, -5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -5, 3.0, 0.0, 0.0)) == 0.4641322034_a);
		CHECK(real(ref_std::cartesian_complex(5, -5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -5, 0.5, 1.5, 2.5)) == 0.0202375845_a);
		
		CHECK(real(ref_std::cartesian_complex(5, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -4, 0.5, 1.5, 2.5)) == 0.0283530398_a);
		
		CHECK(real(ref_std::cartesian_complex(5, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -3, 3.0, 0.0, 0.0)) == -0.3459437191_a);
		CHECK(real(ref_std::cartesian_complex(5, -3, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -3, 0.5, 1.5, 2.5)) == -0.2358100379_a);
		
		CHECK(real(ref_std::cartesian_complex(5, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -2, 0.5, 1.5, 2.5)) == -0.3741630893_a);
		
		CHECK(real(ref_std::cartesian_complex(5, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -1, 3.0, 0.0, 0.0)) == 0.3202816486_a);
		CHECK(real(ref_std::cartesian_complex(5, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5, -1, 0.5, 1.5, 2.5)) == 0.0928071079_a);
		
		CHECK(real(ref_std::cartesian_complex(5,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 0, 0.0, 0.0, 2.0)) == 0.9356025796_a);
		CHECK(real(ref_std::cartesian_complex(5,	 0, 0.5, 1.5, 2.5)) == -0.282403036_a);
		
		CHECK(real(ref_std::cartesian_complex(5,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 1, 2.0, 0.0, 0.0)) == -0.3202816486_a);
		CHECK(real(ref_std::cartesian_complex(5,	 1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 1, 0.5, 1.5, 2.5)) == -0.0928071079_a);

		CHECK(real(ref_std::cartesian_complex(5,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 2, 0.5, 1.5, 2.5)) == -0.3741630893_a);

		CHECK(real(ref_std::cartesian_complex(5,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 3, 2.0, 0.0, 0.0)) == 0.3459437191_a);
		CHECK(real(ref_std::cartesian_complex(5,	 3, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 3, 0.5, 1.5, 2.5)) == 0.2358100379_a);
		
		CHECK(real(ref_std::cartesian_complex(5,	 4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 4, 0.5, 1.5, 2.5)) == 0.0283530398_a);

		CHECK(real(ref_std::cartesian_complex(5,	 5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 5, 3.0, 0.0, 0.0)) == -0.4641322034_a);
		CHECK(real(ref_std::cartesian_complex(5,	 5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_std::cartesian_complex(5,	 5, 0.5, 1.5, 2.5)) == -0.0202375845_a);

		CHECK(imag(ref_std::cartesian_complex(5, -5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -5, 0.0, 3.0, 0.0)) == -0.4641322034_a);
		CHECK(imag(ref_std::cartesian_complex(5, -5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -5, 0.5, 1.5, 2.5)) == 0.0007685159_a);
		
		CHECK(imag(ref_std::cartesian_complex(5, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -4, 0.5, 1.5, 2.5)) == 0.0972104222_a);
		
		CHECK(imag(ref_std::cartesian_complex(5, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -3, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -3, 0.0, 3.0, 0.0)) == -0.3459437191_a);
		CHECK(imag(ref_std::cartesian_complex(5, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -3, 0.5, 1.5, 2.5)) == 0.1632531032_a);
		
		CHECK(imag(ref_std::cartesian_complex(5, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -2, 0.5, 1.5, 2.5)) == -0.280622317_a);
		
		CHECK(imag(ref_std::cartesian_complex(5, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -1, 0.0, 3.0, 0.0)) == -0.3202816486_a);
		CHECK(imag(ref_std::cartesian_complex(5, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5, -1, 0.5, 1.5, 2.5)) == -0.2784213237_a);
		
		CHECK(imag(ref_std::cartesian_complex(5,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 0, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 0, 0.5, 1.5, 2.5)) == (0.0_a).margin(1e-12));
		
		CHECK(imag(ref_std::cartesian_complex(5,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 1, 0.0, 2.0, 0.0)) == -0.3202816486_a);
		CHECK(imag(ref_std::cartesian_complex(5,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 1, 0.5, 1.5, 2.5)) == -0.2784213237_a);

		CHECK(imag(ref_std::cartesian_complex(5,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 2, 0.5, 1.5, 2.5)) == 0.280622317_a);

		CHECK(imag(ref_std::cartesian_complex(5,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 3, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 3, 0.0, 2.0, 0.0)) == -0.3459437191_a);
		CHECK(imag(ref_std::cartesian_complex(5,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 3, 0.5, 1.5, 2.5)) == 0.1632531032_a);
		
		CHECK(imag(ref_std::cartesian_complex(5,	 4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 4, 0.5, 1.5, 2.5)) == -0.0972104222_a);

		CHECK(imag(ref_std::cartesian_complex(5,	 5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 5, 0.0, 3.0, 0.0)) == -0.4641322034_a);
		CHECK(imag(ref_std::cartesian_complex(5,	 5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_std::cartesian_complex(5,	 5, 0.5, 1.5, 2.5)) == 0.0007685159_a);
		
	}
#endif

#ifdef HAVE_BOOST_SPHERICAL_HARMONIC
	SECTION("reference boost real"){
		CHECK(ref_boost::cartesian_real(3, -3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -3, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -3, 0.0, 3.0, 0.0) == -0.5900435899_a);
		CHECK(ref_boost::cartesian_real(3, -3, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -3, 0.5, 1.5, 2.5) == -0.0512925789_a);
		
		CHECK(ref_boost::cartesian_real(3, -2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -2, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -2, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -2, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -2, 0.5, 1.5, 2.5) == 0.2094010765_a);
		
		CHECK(ref_boost::cartesian_real(3, -1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -1, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -1, 0.0, 3.0, 0.0) == -0.4570457995_a);
		CHECK(ref_boost::cartesian_real(3, -1, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3, -1, 0.5, 1.5, 2.5) == 0.5959659117_a);
		
		CHECK(ref_boost::cartesian_real(3,	0, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	0, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	0, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	0, 0.0, 0.0, 2.0) == 0.7463526652_a);
		CHECK(ref_boost::cartesian_real(3,	0, 0.5, 1.5, 2.5) == 0.1802237516_a);
		
		CHECK(ref_boost::cartesian_real(3,	1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	1, 2.0, 0.0, 0.0) == -0.4570457995_a);
		CHECK(ref_boost::cartesian_real(3,	1, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	1, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	1, 0.5, 1.5, 2.5) == 0.1986553039_a);

		CHECK(ref_boost::cartesian_real(3,	2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	2, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	2, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	2, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	2, 0.5, 1.5, 2.5) == -0.2792014354_a);

		CHECK(ref_boost::cartesian_real(3,	3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	3, 2.0, 0.0, 0.0) == 0.5900435899_a);
		CHECK(ref_boost::cartesian_real(3,	3, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	3, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(ref_boost::cartesian_real(3,	3, 0.5, 1.5, 2.5) == -0.0740892806_a);
	}

	SECTION("reference boost complex"){
		
		CHECK(real(ref_boost::cartesian_complex(5, -5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -5, 3.0, 0.0, 0.0)) == 0.4641322034_a);
		CHECK(real(ref_boost::cartesian_complex(5, -5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -5, 0.5, 1.5, 2.5)) == 0.0202375845_a);
		
		CHECK(real(ref_boost::cartesian_complex(5, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -4, 0.5, 1.5, 2.5)) == 0.0283530398_a);
		
		CHECK(real(ref_boost::cartesian_complex(5, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -3, 3.0, 0.0, 0.0)) == -0.3459437191_a);
		CHECK(real(ref_boost::cartesian_complex(5, -3, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -3, 0.5, 1.5, 2.5)) == -0.2358100379_a);
		
		CHECK(real(ref_boost::cartesian_complex(5, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -2, 0.5, 1.5, 2.5)) == -0.3741630893_a);
		
		CHECK(real(ref_boost::cartesian_complex(5, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -1, 3.0, 0.0, 0.0)) == 0.3202816486_a);
		CHECK(real(ref_boost::cartesian_complex(5, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5, -1, 0.5, 1.5, 2.5)) == 0.0928071079_a);
		
		CHECK(real(ref_boost::cartesian_complex(5,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 0, 0.0, 0.0, 2.0)) == 0.9356025796_a);
		CHECK(real(ref_boost::cartesian_complex(5,	 0, 0.5, 1.5, 2.5)) == -0.282403036_a);
		
		CHECK(real(ref_boost::cartesian_complex(5,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 1, 2.0, 0.0, 0.0)) == -0.3202816486_a);
		CHECK(real(ref_boost::cartesian_complex(5,	 1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 1, 0.5, 1.5, 2.5)) == -0.0928071079_a);

		CHECK(real(ref_boost::cartesian_complex(5,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 2, 0.5, 1.5, 2.5)) == -0.3741630893_a);

		CHECK(real(ref_boost::cartesian_complex(5,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 3, 2.0, 0.0, 0.0)) == 0.3459437191_a);
		CHECK(real(ref_boost::cartesian_complex(5,	 3, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 3, 0.5, 1.5, 2.5)) == 0.2358100379_a);
		
		CHECK(real(ref_boost::cartesian_complex(5,	 4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 4, 0.5, 1.5, 2.5)) == 0.0283530398_a);

		CHECK(real(ref_boost::cartesian_complex(5,	 5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 5, 3.0, 0.0, 0.0)) == -0.4641322034_a);
		CHECK(real(ref_boost::cartesian_complex(5,	 5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(ref_boost::cartesian_complex(5,	 5, 0.5, 1.5, 2.5)) == -0.0202375845_a);

		CHECK(imag(ref_boost::cartesian_complex(5, -5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -5, 0.0, 3.0, 0.0)) == -0.4641322034_a);
		CHECK(imag(ref_boost::cartesian_complex(5, -5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -5, 0.5, 1.5, 2.5)) == 0.0007685159_a);
		
		CHECK(imag(ref_boost::cartesian_complex(5, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -4, 0.5, 1.5, 2.5)) == 0.0972104222_a);
		
		CHECK(imag(ref_boost::cartesian_complex(5, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -3, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -3, 0.0, 3.0, 0.0)) == -0.3459437191_a);
		CHECK(imag(ref_boost::cartesian_complex(5, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -3, 0.5, 1.5, 2.5)) == 0.1632531032_a);
		
		CHECK(imag(ref_boost::cartesian_complex(5, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -2, 0.5, 1.5, 2.5)) == -0.280622317_a);
		
		CHECK(imag(ref_boost::cartesian_complex(5, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -1, 0.0, 3.0, 0.0)) == -0.3202816486_a);
		CHECK(imag(ref_boost::cartesian_complex(5, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5, -1, 0.5, 1.5, 2.5)) == -0.2784213237_a);
		
		CHECK(imag(ref_boost::cartesian_complex(5,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 0, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 0, 0.5, 1.5, 2.5)) == (0.0_a).margin(1e-12));
		
		CHECK(imag(ref_boost::cartesian_complex(5,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 1, 0.0, 2.0, 0.0)) == -0.3202816486_a);
		CHECK(imag(ref_boost::cartesian_complex(5,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 1, 0.5, 1.5, 2.5)) == -0.2784213237_a);

		CHECK(imag(ref_boost::cartesian_complex(5,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 2, 0.5, 1.5, 2.5)) == 0.280622317_a);

		CHECK(imag(ref_boost::cartesian_complex(5,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 3, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 3, 0.0, 2.0, 0.0)) == -0.3459437191_a);
		CHECK(imag(ref_boost::cartesian_complex(5,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 3, 0.5, 1.5, 2.5)) == 0.1632531032_a);
		
		CHECK(imag(ref_boost::cartesian_complex(5,	 4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 4, 0.5, 1.5, 2.5)) == -0.0972104222_a);

		CHECK(imag(ref_boost::cartesian_complex(5,	 5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 5, 0.0, 3.0, 0.0)) == -0.4641322034_a);
		CHECK(imag(ref_boost::cartesian_complex(5,	 5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(ref_boost::cartesian_complex(5,	 5, 0.5, 1.5, 2.5)) == 0.0007685159_a);		
	}
#endif
	
}
