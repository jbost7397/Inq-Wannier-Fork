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
#include <vector>

#ifdef ENABLE_GPU
#include <thrust/complex.h>
using complex = thrust::complex<double>;
#else
#include <complex>
using complex = std::complex<double>;
#endif

SH_GPU_FUNCTION inline double real(const complex & z){
	return z.real();
}

SH_GPU_FUNCTION inline auto imag(const complex & z){
	return z.imag();
}

struct comparison {
	std::vector<int> ll;
	std::vector<int> mm;
	std::vector<double> av_diff_lm;
	std::vector<double> max_diff_lm;
	std::vector<double> av_diff_l;
	std::vector<double> max_diff_l;
	double av_diff;
	double max_diff;	
};

template <typename Impl1, typename Impl2>
auto compare(Impl1 impl1, Impl2 impl2, double tol) {

	auto reps = 10000;
	comparison comp;

	comp.av_diff = 0.0;
	comp.max_diff = 0.0;
	
	for(int ll = 0; ll <= SHARMONIC_MAX_L; ll++) {
		auto avdiff_l  = 0.0;
		auto maxdiff_l = 0.0;
		
		for(int mm = -ll; mm <= ll; mm++) {
			
			auto avdiff_lm  = 0.0;
			auto maxdiff_lm = 0.0;

			srand48(0xFC769E60887BB49Cl);

			auto neff = 0;
			for(int iter = 0; iter < reps; iter++){
				double xx = 2.0*(drand48() - 0.5);
				double yy = 2.0*(drand48() - 0.5);
				double zz = 2.0*(drand48() - 0.5);

				if(xx*xx + yy*yy + zz*zz > 1.0) continue;
				
				auto val1 = impl1(ll, mm, xx, yy, zz);
				auto val2 = impl2(ll, mm, xx, yy, zz);

				auto diff = std::abs(val1 - val2);

				CHECK(diff < tol);

				if(diff > 1e-14) {
					std::cout << ll << '\t' << mm << '\t' << xx << '\t' << yy << '\t' << zz;
					sharmonic::internal::normalize(&xx, &yy, &zz);					
					std::cout << " z/r = " << zz << '\t' << diff << std::endl;;
				}
				
				avdiff_lm += diff;
				maxdiff_lm = std::max(maxdiff_lm, diff);
				neff++;
			}

			avdiff_lm /= neff;
			
			comp.ll.push_back(ll);
			comp.mm.push_back(mm);
			comp.av_diff_lm.push_back(avdiff_lm);
			comp.max_diff_lm.push_back(maxdiff_lm);	

			avdiff_l += avdiff_lm/(2*ll + 1);
			maxdiff_l = std::max(maxdiff_l, maxdiff_lm);
		}
		
		comp.av_diff_l.push_back(avdiff_l);
		comp.max_diff_l.push_back(maxdiff_l);	

		comp.av_diff += avdiff_l/(SHARMONIC_MAX_L + 1.0);
		comp.max_diff = std::max(comp.max_diff, maxdiff_l);
		
	}

	return comp;
}
TEST_CASE("reference", "[reference]") {
	using namespace Catch::literals;
	using Catch::Approx;

	using namespace sharmonic;
	using sharmonic::internal::normalize;
	using sharmonic::cartesian_real;
	using sharmonic::cartesian_complex;

#if defined(HAVE_SPH_LEGENDRE) && defined(HAVE_BOOST_SPHERICAL_HARMONIC)
	SECTION("reference values") {

		auto tol = 8e-14;
		auto ref_tol = 2e-14;

		std::cout << "SHarmonic v/s std" << std::endl;
		auto comp_sh_std  = compare([](auto l, auto m, auto x, auto y, auto z) { return sharmonic::cartesian_real(l, m, x, y, z); }, ref_std::cartesian_real, tol);
		std::cout << "SHarmonic v/s boost" << std::endl;		
		auto comp_sh_boo  = compare([](auto l, auto m, auto x, auto y, auto z) { return sharmonic::cartesian_real(l, m, x, y, z); }, ref_boost::cartesian_real, tol);
		std::cout << "std v/s boost" << std::endl;		
		auto comp_std_boo = compare(ref_std::cartesian_real, ref_boost::cartesian_real, ref_tol);

		std::cout << "Average difference by l" << std::endl;
		for(int ll = 0; ll <= SHARMONIC_MAX_L; ll++) {
			std::cout << ll << '\t' <<  comp_sh_std.av_diff_l[ll] << '\t'  <<  comp_sh_boo.av_diff_l[ll] << '\t'  <<  comp_std_boo.av_diff_l[ll] << '\t' << std::endl;
		}
		std::cout << std::endl;

		std::cout << "Maximum difference by l" << std::endl;
		for(int ll = 0; ll <= SHARMONIC_MAX_L; ll++) {
			std::cout << ll << '\t' <<  comp_sh_std.max_diff_l[ll] << '\t'  <<  comp_sh_boo.max_diff_l[ll] << '\t'  <<  comp_std_boo.max_diff_l[ll] << '\t' << std::endl;
		}
		std::cout << std::endl;

		std::cout << "Maximum difference by lm" << std::endl;
		for(auto ii = 0ul; ii < comp_sh_std.ll.size(); ii++){
			std::cout << comp_sh_std.ll[ii] << '\t' <<  comp_sh_std.mm[ii] << '\t' << comp_sh_std.max_diff_lm[ii] << '\t'  <<  comp_sh_boo.max_diff_lm[ii] << '\t'  <<  comp_std_boo.max_diff_lm[ii] << '\t' << std::endl;
		}
	}
#endif	

	SECTION("interfaces") {

		CHECK(normalized_cartesian_real(2, 0, 0.16903085094570331550, 0.50709255283710994650, 0.84515425472851657751) == 0.3604475031_a);

		CHECK(normalized_cartesian_real(2, 0, {0.16903085094570331550, 0.50709255283710994650, 0.84515425472851657751}) == 0.3604475031_a);

		CHECK(cartesian_real(2, 0, 0.5, 1.5, 2.5) == 0.3604475031_a);

		CHECK(cartesian_real(2, 0, {0.5, 1.5, 2.5}) == 0.3604475031_a);

		CHECK(cartesian_real(2, 0, std::vector{0.5, 1.5, 2.5}) == 0.3604475031_a);

		CHECK(angular_real(2, 0, 0.563942641, 1.24904577) == 0.3604475031_a);
		

		CHECK(real(normalized_cartesian_complex<complex>(3, -1, 0.16903085094570331550, 0.50709255283710994650, 0.84515425472851657751)) == 0.1404705125_a);
		CHECK(imag(normalized_cartesian_complex<complex>(3, -1, 0.16903085094570331550, 0.50709255283710994650, 0.84515425472851657751)) == -0.4214115375_a);
		
		CHECK(real(normalized_cartesian_complex<complex>(3, -1, {0.16903085094570331550, 0.50709255283710994650, 0.84515425472851657751})) == 0.1404705125_a);
		CHECK(imag(normalized_cartesian_complex<complex>(3, -1, {0.16903085094570331550, 0.50709255283710994650, 0.84515425472851657751})) == -0.4214115375_a);

		CHECK(real(cartesian_complex<complex>(3, -1, 0.5, 1.5, 2.5)) == 0.1404705125_a);
		CHECK(imag(cartesian_complex<complex>(3, -1, 0.5, 1.5, 2.5)) == -0.4214115375_a);

		CHECK(real(cartesian_complex<complex>(3, -1, {0.5, 1.5, 2.5})) == 0.1404705125_a);
		CHECK(imag(cartesian_complex<complex>(3, -1, std::vector{0.5, 1.5, 2.5})) == -0.4214115375_a);
		
		CHECK(real(angular_complex<complex>(3, -1, 0.563942641, 1.24904577)) == 0.1404705125_a);
		CHECK(imag(angular_complex<complex>(3, -1, 0.563942641, 1.24904577)) == -0.4214115375_a);
		
	}
	
	SECTION("l=0"){
		CHECK(cartesian_real(0, 0, 0.0, 0.0, 0.0) == 0.2820947918_a);

		CHECK(cartesian_real(0, 0, 1.0, 0.0, 0.0) == 0.2820947918_a);
		CHECK(cartesian_real(0, 0, 0.0, 1.0, 0.0) == 0.2820947918_a);
		CHECK(cartesian_real(0, 0, 0.0, 0.0, 1.0) == 0.2820947918_a);
	}

	SECTION("l=1"){
		CHECK(cartesian_real(1, -1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(1, -1, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(1, -1, 0.0, 3.0, 0.0) == 0.4886025119_a);
		CHECK(cartesian_real(1, -1, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(1, -1, 0.5, 1.5, 2.5) == 0.2477666951_a);
		
		CHECK(cartesian_real(1,	0, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(1,	0, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(1,	0, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(1,	0, 0.0, 0.0, 2.0) == 0.4886025119_a);
		CHECK(cartesian_real(1,	0, 0.5, 1.5, 2.5) == 0.4129444918_a);
		
		CHECK(cartesian_real(1,	1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(1,	1, 2.0, 0.0, 0.0) == 0.4886025119_a);
		CHECK(cartesian_real(1,	1, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(1,	1, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(1,	1, 0.5, 1.5, 2.5) == 0.0825888984_a);
	}
	
	SECTION("l=2"){
		CHECK(cartesian_real(2, -2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2, -2, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2, -2, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2, -2, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2, -2, 0.5, 1.5, 2.5) == 0.0936470083_a);
		
		CHECK(cartesian_real(2, -1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2, -1, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2, -1, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2, -1, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2, -1, 0.5, 1.5, 2.5) == 0.4682350417_a);
		
		CHECK(cartesian_real(2,	0, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2,	0, 2.0, 0.0, 0.0) == -0.3153915653_a);
		CHECK(cartesian_real(2,	0, 0.0, 2.0, 0.0) == -0.3153915653_a);
		CHECK(cartesian_real(2,	0, 0.0, 0.0, 2.0) == 0.6307831305_a);
		CHECK(cartesian_real(2,	0, 0.5, 1.5, 2.5) == 0.3604475031_a);
		
		CHECK(cartesian_real(2,	1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2,	1, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2,	1, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2,	1, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2,	1, 0.5, 1.5, 2.5) == 0.1560783472_a);

		CHECK(cartesian_real(2,	2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2,	2, 2.0, 0.0, 0.0) == 0.5462742153_a);
		CHECK(cartesian_real(2,	2, 0.0, 2.0, 0.0) == -0.5462742153_a);
		CHECK(cartesian_real(2,	2, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(2,	2, 0.5, 1.5, 2.5) == -0.1248626778_a);
	}

	SECTION("l=3"){
		CHECK(cartesian_real(3, -3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -3, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -3, 0.0, 3.0, 0.0) == -0.5900435899_a);
		CHECK(cartesian_real(3, -3, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -3, 0.5, 1.5, 2.5) == -0.0512925789_a);
		
		CHECK(cartesian_real(3, -2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -2, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -2, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -2, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -2, 0.5, 1.5, 2.5) == 0.2094010765_a);
		
		CHECK(cartesian_real(3, -1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -1, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -1, 0.0, 3.0, 0.0) == -0.4570457995_a);
		CHECK(cartesian_real(3, -1, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3, -1, 0.5, 1.5, 2.5) == 0.5959659117_a);
		
		CHECK(cartesian_real(3,	0, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	0, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	0, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	0, 0.0, 0.0, 2.0) == 0.7463526652_a);
		CHECK(cartesian_real(3,	0, 0.5, 1.5, 2.5) == 0.1802237516_a);
		
		CHECK(cartesian_real(3,	1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	1, 2.0, 0.0, 0.0) == -0.4570457995_a);
		CHECK(cartesian_real(3,	1, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	1, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	1, 0.5, 1.5, 2.5) == 0.1986553039_a);

		CHECK(cartesian_real(3,	2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	2, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	2, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	2, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	2, 0.5, 1.5, 2.5) == -0.2792014354_a);

		CHECK(cartesian_real(3,	3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	3, 2.0, 0.0, 0.0) == 0.5900435899_a);
		CHECK(cartesian_real(3,	3, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	3, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(3,	3, 0.5, 1.5, 2.5) == -0.0740892806_a);
	}

	SECTION("l=4"){
		CHECK(cartesian_real(4, -4, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -4, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -4, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -4, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -4, 0.5, 1.5, 2.5) == -0.0490450862_a);
		
		CHECK(cartesian_real(4, -3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -3, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -3, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -3, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -3, 0.5, 1.5, 2.5) == -0.1300504239_a);
		
		CHECK(cartesian_real(4, -2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -2, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -2, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -2, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -2, 0.5, 1.5, 2.5) == 0.3244027528_a);
		
		CHECK(cartesian_real(4, -1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -1, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -1, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -1, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4, -1, 0.5, 1.5, 2.5) == 0.5734684659_a);
		
		CHECK(cartesian_real(4,	0, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	0, 2.0, 0.0, 0.0) == 0.3173566407_a);
		CHECK(cartesian_real(4,	0, 0.0, 2.0, 0.0) == 0.3173566407_a);
		CHECK(cartesian_real(4,	0, 0.0, 0.0, 2.0) == 0.8462843753_a);
		CHECK(cartesian_real(4,	0, 0.5, 1.5, 2.5) == -0.060448884_a);
		
		CHECK(cartesian_real(4,	1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	1, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	1, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	1, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	1, 0.5, 1.5, 2.5) == 0.1911561553_a);

		CHECK(cartesian_real(4,	2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	2, 2.0, 0.0, 0.0) == -0.4730873479_a);
		CHECK(cartesian_real(4,	2, 0.0, 2.0, 0.0) == 0.4730873479_a);
		CHECK(cartesian_real(4,	2, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	2, 0.5, 1.5, 2.5) == -0.4325370038_a);

		CHECK(cartesian_real(4,	3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	3, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	3, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	3, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	3, 0.5, 1.5, 2.5) == -0.1878506123_a);
		
		CHECK(cartesian_real(4,	4, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	4, 3.0, 0.0, 0.0) == 0.6258357354_a);
		CHECK(cartesian_real(4,	4, 0.0, 3.0, 0.0) == 0.6258357354_a);
		CHECK(cartesian_real(4,	4, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(4,	4, 0.5, 1.5, 2.5) == 0.0143048168_a);
	}
	
	SECTION("l=5"){
		CHECK(cartesian_real(5, -5, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -5, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -5, 0.0, 3.0, 0.0) == 0.6563820568_a);
		CHECK(cartesian_real(5, -5, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -5, 0.5, 1.5, 2.5) == -0.0010868456_a);
		
		CHECK(cartesian_real(5, -4, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -4, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -4, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -4, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -4, 0.5, 1.5, 2.5) == -0.1374762974_a);
		
		CHECK(cartesian_real(5, -3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -3, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -3, 0.0, 3.0, 0.0) == 0.4892382994_a);
		CHECK(cartesian_real(5, -3, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -3, 0.5, 1.5, 2.5) == -0.2308747526_a);
		
		CHECK(cartesian_real(5, -2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -2, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -2, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -2, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -2, 0.5, 1.5, 2.5) == 0.3968598866_a);
		
		CHECK(cartesian_real(5, -1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -1, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -1, 0.0, 3.0, 0.0) == 0.4529466512_a);
		CHECK(cartesian_real(5, -1, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5, -1, 0.5, 1.5, 2.5) == 0.393747212_a);
		
		CHECK(cartesian_real(5,	0, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	0, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	0, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	0, 0.0, 0.0, 2.0) == 0.9356025796_a);
		CHECK(cartesian_real(5,	0, 0.5, 1.5, 2.5) == -0.282403036_a);
		
		CHECK(cartesian_real(5,	1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	1, 2.0, 0.0, 0.0) == 0.4529466512_a);
		CHECK(cartesian_real(5,	1, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	1, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	1, 0.5, 1.5, 2.5) == 0.1312490707_a);

		CHECK(cartesian_real(5,	2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	2, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	2, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	2, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	2, 0.5, 1.5, 2.5) == -0.5291465155_a);

		CHECK(cartesian_real(5,	3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	3, 2.0, 0.0, 0.0) == -0.4892382994_a);
		CHECK(cartesian_real(5,	3, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	3, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	3, 0.5, 1.5, 2.5) == -0.3334857538_a);
		
		CHECK(cartesian_real(5,	4, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	4, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	4, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	4, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	4, 0.5, 1.5, 2.5) == 0.0400972534_a);

		CHECK(cartesian_real(5,	5, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	5, 3.0, 0.0, 0.0) == 0.6563820568_a);
		CHECK(cartesian_real(5,	5, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	5, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(5,	5, 0.5, 1.5, 2.5) == 0.0286202664_a);
	}
	
	SECTION("l=6"){
		CHECK(cartesian_real(6, -6, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -6, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -6, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -6, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -6, 0.5, 1.5, 2.5) == 0.0149145265_a);
		
		CHECK(cartesian_real(6, -5, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -5, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -5, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -5, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -5, 0.5, 1.5, 2.5) == -0.0033118869_a);
		
		CHECK(cartesian_real(6, -4, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -4, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -4, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -4, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -4, 0.5, 1.5, 2.5) == -0.2711411152_a);
		
		CHECK(cartesian_real(6, -3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -3, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -3, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -3, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -3, 0.5, 1.5, 2.5) == -0.3287333054_a);
		
		CHECK(cartesian_real(6, -2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -2, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -2, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -2, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -2, 0.5, 1.5, 2.5) == 0.3931908163_a);
		
		CHECK(cartesian_real(6, -1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -1, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -1, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -1, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6, -1, 0.5, 1.5, 2.5) == 0.1019162733_a);
		
		CHECK(cartesian_real(6,	0, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	0, 2.0, 0.0, 0.0) == -0.3178460113_a);
		CHECK(cartesian_real(6,	0, 0.0, 2.0, 0.0) == -0.3178460113_a);
		CHECK(cartesian_real(6,	0, 0.0, 0.0, 2.0) == 1.0171072363_a);
		CHECK(cartesian_real(6,	0, 0.5, 1.5, 2.5) == -0.4151458107_a);
		
		CHECK(cartesian_real(6,	1, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	1, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	1, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	1, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	1, 0.5, 1.5, 2.5) == 0.0339720911_a);

		CHECK(cartesian_real(6,	2, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	2, 2.0, 0.0, 0.0) == 0.4606026298_a);
		CHECK(cartesian_real(6,	2, 0.0, 2.0, 0.0) == -0.4606026298_a);
		CHECK(cartesian_real(6,	2, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	2, 0.5, 1.5, 2.5) == -0.5242544217_a);

		CHECK(cartesian_real(6,	3, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	3, 2.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	3, 0.0, 2.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	3, 0.0, 0.0, 2.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	3, 0.5, 1.5, 2.5) == -0.4748369967_a);
		
		CHECK(cartesian_real(6,	4, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	4, 3.0, 0.0, 0.0) == -0.5045649007_a);
		CHECK(cartesian_real(6,	4, 0.0, 3.0, 0.0) == -0.5045649007_a);
		CHECK(cartesian_real(6,	4, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	4, 0.5, 1.5, 2.5) == 0.0790828253_a);

		CHECK(cartesian_real(6,	5, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	5, 3.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	5, 0.0, 3.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	5, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	5, 0.5, 1.5, 2.5) == 0.087213021_a);

		CHECK(cartesian_real(6,	6, 0.0, 0.0, 0.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	6, 3.0, 0.0, 0.0) == 0.6831841052_a);
		CHECK(cartesian_real(6,	6, 0.0, 3.0, 0.0) == -0.6831841052_a);
		CHECK(cartesian_real(6,	6, 0.0, 0.0, 3.0) == (0.0_a).margin(1e-12));
		CHECK(cartesian_real(6,	6, 0.5, 1.5, 2.5) == 0.0056088817_a);
	}

	SECTION("complex l=0"){
		CHECK(real(cartesian_complex<complex>(0, 0, 0.0, 0.0, 0.0)) == 0.2820947918_a);	
		
		CHECK(real(cartesian_complex<complex>(0, 0, 1.0, 0.0, 0.0)) == 0.2820947918_a);
		CHECK(real(cartesian_complex<complex>(0, 0, 0.0, 1.0, 0.0)) == 0.2820947918_a);
		CHECK(real(cartesian_complex<complex>(0, 0, 0.0, 0.0, 1.0)) == 0.2820947918_a);

		
		CHECK(imag(cartesian_complex<complex>(0, 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));

		CHECK(imag(cartesian_complex<complex>(0, 0, 1.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(0, 0, 0.0, 1.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(0, 0, 0.0, 0.0, 1.0)) == (0.0_a).margin(1e-12));

	}

	SECTION("complex l=1"){

#ifdef GENERATE_REFERENCE
		CHECK(real(bsh(1, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(bsh(1, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
#endif
		
		CHECK(real(cartesian_complex<complex>(1, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(1, -1, 3.0, 0.0, 0.0)) == 0.3454941495_a);
		CHECK(real(cartesian_complex<complex>(1, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(1, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(1, -1, 0.5, 1.5, 2.5)) == 0.0583991701_a);

		CHECK(imag(cartesian_complex<complex>(1, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1, -1, 0.0, 3.0, 0.0)) == -0.3454941495_a);
		CHECK(imag(cartesian_complex<complex>(1, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1, -1, 0.5, 1.5, 2.5)) == -0.1751975102_a);
		
		CHECK(real(cartesian_complex<complex>(1,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(1,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(1,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(1,	 0, 0.0, 0.0, 2.0)) == 0.4886025119_a);
		CHECK(real(cartesian_complex<complex>(1,	 0, 0.5, 1.5, 2.5)) == 0.4129444918_a);

		CHECK(imag(cartesian_complex<complex>(1,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1,	 0, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1,	 0, 0.5, 1.5, 2.5)) == (0.0_a).margin(1e-12));

		CHECK(real(cartesian_complex<complex>(1,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(1,	 1, 2.0, 0.0, 0.0)) == -0.3454941495_a);
		CHECK(real(cartesian_complex<complex>(1,	 1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(1,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(1,	 1, 0.5, 1.5, 2.5)) == -0.0583991701_a);

		CHECK(imag(cartesian_complex<complex>(1,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1,	 1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1,	 1, 0.0, 2.0, 0.0)) == -0.3454941495_a);
		CHECK(imag(cartesian_complex<complex>(1,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(1,	 1, 0.5, 1.5, 2.5)) == -0.1751975102_a);
	}

	SECTION("complex l=2"){

		CHECK(real(cartesian_complex<complex>(2, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2, -2, 3.0, 0.0, 0.0)) ==	0.386274202_a);
		CHECK(real(cartesian_complex<complex>(2, -2, 0.0, 3.0, 0.0)) == -0.386274202_a);
		CHECK(real(cartesian_complex<complex>(2, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2, -2, 0.5, 1.5, 2.5)) == -0.0882912462_a);

		CHECK(imag(cartesian_complex<complex>(2, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2, -2, 0.5, 1.5, 2.5)) == -0.0662184346_a);

		CHECK(real(cartesian_complex<complex>(2, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2, -1, 0.5, 1.5, 2.5)) == 0.1103640577_a);

		CHECK(imag(cartesian_complex<complex>(2, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2, -1, 0.5, 1.5, 2.5)) == -0.3310921732_a);

		CHECK(real(cartesian_complex<complex>(2,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2,	 0, 2.0, 0.0, 0.0)) == -0.3153915653_a);
		CHECK(real(cartesian_complex<complex>(2,	 0, 0.0, 2.0, 0.0)) == -0.3153915653_a);
		CHECK(real(cartesian_complex<complex>(2,	 0, 0.0, 0.0, 2.0)) == 0.6307831305_a);
		CHECK(real(cartesian_complex<complex>(2,	 0, 0.5, 1.5, 2.5)) == 0.3604475031_a);

		CHECK(imag(cartesian_complex<complex>(2,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 0, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 0, 0.5, 1.5, 2.5)) == (0.0_a).margin(1e-12));

		CHECK(real(cartesian_complex<complex>(2,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2,	 1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2,	 1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2,	 1, 0.5, 1.5, 2.5)) == -0.1103640577_a);

		CHECK(imag(cartesian_complex<complex>(2,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 1, 0.5, 1.5, 2.5)) == -0.3310921732_a);

		CHECK(real(cartesian_complex<complex>(2,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2,	 2, 2.0, 0.0, 0.0)) == 0.386274202_a);
		CHECK(real(cartesian_complex<complex>(2,	 2, 0.0, 2.0, 0.0)) == -0.386274202_a);
		CHECK(real(cartesian_complex<complex>(2,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(2,	 2, 0.5, 1.5, 2.5)) == -0.0882912462_a);
		
		CHECK(imag(cartesian_complex<complex>(2,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(2,	 2, 0.5, 1.5, 2.5)) == 0.0662184346_a);
		
	}

	SECTION("complex l=3"){
		
		CHECK(real(cartesian_complex<complex>(3, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -3, 3.0, 0.0, 0.0)) == 0.4172238236_a);
		CHECK(real(cartesian_complex<complex>(3, -3, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -3, 0.5, 1.5, 2.5)) == -0.0523890328_a);
		
		CHECK(imag(cartesian_complex<complex>(3, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -3, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -3, 0.0, 3.0, 0.0)) == 0.4172238236_a);
		CHECK(imag(cartesian_complex<complex>(3, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -3, 0.5, 1.5, 2.5)) == 0.0362693304_a);
		
		CHECK(real(cartesian_complex<complex>(3, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -2, 0.5, 1.5, 2.5)) == -0.1974252283_a);

		CHECK(imag(cartesian_complex<complex>(3, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -2, 0.5, 1.5, 2.5)) == -0.1480689212_a);
		
		CHECK(real(cartesian_complex<complex>(3, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -1, 3.0, 0.0, 0.0)) == -0.3231801841_a);
		CHECK(real(cartesian_complex<complex>(3, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3, -1, 0.5, 1.5, 2.5)) == 0.1404705125_a);

		CHECK(imag(cartesian_complex<complex>(3, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -1, 0.0, 3.0, 0.0)) == 0.3231801841_a);
		CHECK(imag(cartesian_complex<complex>(3, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3, -1, 0.5, 1.5, 2.5)) == -0.4214115375_a);
		
		CHECK(real(cartesian_complex<complex>(3,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 0, 0.0, 0.0, 2.0)) == 0.7463526652_a);
		CHECK(real(cartesian_complex<complex>(3,	 0, 0.5, 1.5, 2.5)) == 0.1802237516_a);

		CHECK(imag(cartesian_complex<complex>(3,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 0, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 0, 0.5, 1.5, 2.5)) == (0.0_a).margin(1e-12));
		
		CHECK(real(cartesian_complex<complex>(3,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 1, 2.0, 0.0, 0.0)) == 0.3231801841_a);
		CHECK(real(cartesian_complex<complex>(3,	 1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 1, 0.5, 1.5, 2.5)) == -0.1404705125_a);

		CHECK(real(cartesian_complex<complex>(3,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 2, 0.5, 1.5, 2.5)) == -0.1974252283_a);

		CHECK(imag(cartesian_complex<complex>(3,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 2, 0.5, 1.5, 2.5)) == 0.1480689212_a);
		
		CHECK(real(cartesian_complex<complex>(3,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 3, 2.0, 0.0, 0.0)) == -0.4172238236_a);
		CHECK(real(cartesian_complex<complex>(3,	 3, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(3,	 3, 0.5, 1.5, 2.5)) == 0.0523890328_a);

		CHECK(imag(cartesian_complex<complex>(3,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 3, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 3, 0.0, 2.0, 0.0)) == 0.4172238236_a);
		CHECK(imag(cartesian_complex<complex>(3,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(3,	 3, 0.5, 1.5, 2.5)) == 0.0362693304_a);		
	}

	SECTION("complex l=4"){
		
		CHECK(real(cartesian_complex<complex>(4, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -4, 3.0, 0.0, 0.0)) == 0.4425326924_a);
		CHECK(real(cartesian_complex<complex>(4, -4, 0.0, 3.0, 0.0)) == 0.4425326924_a);
		CHECK(real(cartesian_complex<complex>(4, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -4, 0.5, 1.5, 2.5)) == 0.010115033_a);

		CHECK(real(cartesian_complex<complex>(4, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -3, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -3, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -3, 0.5, 1.5, 2.5)) == -0.1328304418_a);
		
		CHECK(real(cartesian_complex<complex>(4, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -2, 3.0, 0.0, 0.0)) == -0.3345232718_a);
		CHECK(real(cartesian_complex<complex>(4, -2, 0.0, 3.0, 0.0)) == 0.3345232718_a);
		CHECK(real(cartesian_complex<complex>(4, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -2, 0.5, 1.5, 2.5)) == -0.3058498485_a);
		
		CHECK(real(cartesian_complex<complex>(4, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4, -1, 0.5, 1.5, 2.5)) == 0.1351678137_a);
		
		CHECK(real(cartesian_complex<complex>(4,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 0, 2.0, 0.0, 0.0)) == 0.3173566407_a);
		CHECK(real(cartesian_complex<complex>(4,	 0, 0.0, 2.0, 0.0)) == 0.3173566407_a);
		CHECK(real(cartesian_complex<complex>(4,	 0, 0.0, 0.0, 2.0)) == 0.8462843753_a);
		CHECK(real(cartesian_complex<complex>(4,	 0, 0.5, 1.5, 2.5)) == -0.060448884_a);
		
		CHECK(real(cartesian_complex<complex>(4,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 1, 0.5, 1.5, 2.5)) == -0.1351678137_a);

		CHECK(real(cartesian_complex<complex>(4,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 2, 2.0, 0.0, 0.0)) == -0.3345232718_a);
		CHECK(real(cartesian_complex<complex>(4,	 2, 0.0, 2.0, 0.0)) == 0.3345232718_a);
		CHECK(real(cartesian_complex<complex>(4,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 2, 0.5, 1.5, 2.5)) == -0.3058498485_a);

		CHECK(real(cartesian_complex<complex>(4,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 3, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 3, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 3, 0.5, 1.5, 2.5)) == 0.1328304418_a);
		
		CHECK(real(cartesian_complex<complex>(4,	 4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 4, 3.0, 0.0, 0.0)) == 0.4425326924_a);
		CHECK(real(cartesian_complex<complex>(4,	 4, 0.0, 3.0, 0.0)) == 0.4425326924_a);
		CHECK(real(cartesian_complex<complex>(4,	 4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(4,	 4, 0.5, 1.5, 2.5)) == 0.010115033_a);

		CHECK(imag(cartesian_complex<complex>(4, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -4, 0.5, 1.5, 2.5)) == 0.034680113_a);

		CHECK(imag(cartesian_complex<complex>(4, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -3, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -3, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -3, 0.5, 1.5, 2.5)) == 0.0919595366_a);
		
		CHECK(imag(cartesian_complex<complex>(4, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -2, 0.5, 1.5, 2.5)) == -0.2293873864_a);
		
		CHECK(imag(cartesian_complex<complex>(4, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4, -1, 0.5, 1.5, 2.5)) == -0.405503441_a);
		
		CHECK(imag(cartesian_complex<complex>(4,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 0, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 0, 0.5, 1.5, 2.5)) == (0.0_a).margin(1e-12));
		
		CHECK(imag(cartesian_complex<complex>(4,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 1, 0.5, 1.5, 2.5)) == -0.405503441_a);

		CHECK(imag(cartesian_complex<complex>(4,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 2, 0.5, 1.5, 2.5)) == 0.2293873864_a);

		CHECK(imag(cartesian_complex<complex>(4,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 3, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 3, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 3, 0.5, 1.5, 2.5)) == 0.0919595366_a);
		
		CHECK(imag(cartesian_complex<complex>(4,	 4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(4,	 4, 0.5, 1.5, 2.5)) == -0.034680113_a);
		
	}
	
	SECTION("complex l=5"){
		
		CHECK(real(cartesian_complex<complex>(5, -5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -5, 3.0, 0.0, 0.0)) == 0.4641322034_a);
		CHECK(real(cartesian_complex<complex>(5, -5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -5, 0.5, 1.5, 2.5)) == 0.0202375845_a);
		
		CHECK(real(cartesian_complex<complex>(5, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -4, 0.5, 1.5, 2.5)) == 0.0283530398_a);
		
		CHECK(real(cartesian_complex<complex>(5, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -3, 3.0, 0.0, 0.0)) == -0.3459437191_a);
		CHECK(real(cartesian_complex<complex>(5, -3, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -3, 0.5, 1.5, 2.5)) == -0.2358100379_a);
		
		CHECK(real(cartesian_complex<complex>(5, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -2, 0.5, 1.5, 2.5)) == -0.3741630893_a);
		
		CHECK(real(cartesian_complex<complex>(5, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -1, 3.0, 0.0, 0.0)) == 0.3202816486_a);
		CHECK(real(cartesian_complex<complex>(5, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5, -1, 0.5, 1.5, 2.5)) == 0.0928071079_a);
		
		CHECK(real(cartesian_complex<complex>(5,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 0, 0.0, 0.0, 2.0)) == 0.9356025796_a);
		CHECK(real(cartesian_complex<complex>(5,	 0, 0.5, 1.5, 2.5)) == -0.282403036_a);
		
		CHECK(real(cartesian_complex<complex>(5,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 1, 2.0, 0.0, 0.0)) == -0.3202816486_a);
		CHECK(real(cartesian_complex<complex>(5,	 1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 1, 0.5, 1.5, 2.5)) == -0.0928071079_a);

		CHECK(real(cartesian_complex<complex>(5,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 2, 0.5, 1.5, 2.5)) == -0.3741630893_a);

		CHECK(real(cartesian_complex<complex>(5,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 3, 2.0, 0.0, 0.0)) == 0.3459437191_a);
		CHECK(real(cartesian_complex<complex>(5,	 3, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 3, 0.5, 1.5, 2.5)) == 0.2358100379_a);
		
		CHECK(real(cartesian_complex<complex>(5,	 4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 4, 0.5, 1.5, 2.5)) == 0.0283530398_a);

		CHECK(real(cartesian_complex<complex>(5,	 5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 5, 3.0, 0.0, 0.0)) == -0.4641322034_a);
		CHECK(real(cartesian_complex<complex>(5,	 5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(5,	 5, 0.5, 1.5, 2.5)) == -0.0202375845_a);

		CHECK(imag(cartesian_complex<complex>(5, -5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -5, 0.0, 3.0, 0.0)) == -0.4641322034_a);
		CHECK(imag(cartesian_complex<complex>(5, -5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -5, 0.5, 1.5, 2.5)) == 0.0007685159_a);
		
		CHECK(imag(cartesian_complex<complex>(5, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -4, 0.5, 1.5, 2.5)) == 0.0972104222_a);
		
		CHECK(imag(cartesian_complex<complex>(5, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -3, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -3, 0.0, 3.0, 0.0)) == -0.3459437191_a);
		CHECK(imag(cartesian_complex<complex>(5, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -3, 0.5, 1.5, 2.5)) == 0.1632531032_a);
		
		CHECK(imag(cartesian_complex<complex>(5, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -2, 0.5, 1.5, 2.5)) == -0.280622317_a);
		
		CHECK(imag(cartesian_complex<complex>(5, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -1, 0.0, 3.0, 0.0)) == -0.3202816486_a);
		CHECK(imag(cartesian_complex<complex>(5, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5, -1, 0.5, 1.5, 2.5)) == -0.2784213237_a);
		
		CHECK(imag(cartesian_complex<complex>(5,	 0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 0, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 0, 0.5, 1.5, 2.5)) == (0.0_a).margin(1e-12));
		
		CHECK(imag(cartesian_complex<complex>(5,	 1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 1, 0.0, 2.0, 0.0)) == -0.3202816486_a);
		CHECK(imag(cartesian_complex<complex>(5,	 1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 1, 0.5, 1.5, 2.5)) == -0.2784213237_a);

		CHECK(imag(cartesian_complex<complex>(5,	 2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 2, 0.5, 1.5, 2.5)) == 0.280622317_a);

		CHECK(imag(cartesian_complex<complex>(5,	 3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 3, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 3, 0.0, 2.0, 0.0)) == -0.3459437191_a);
		CHECK(imag(cartesian_complex<complex>(5,	 3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 3, 0.5, 1.5, 2.5)) == 0.1632531032_a);
		
		CHECK(imag(cartesian_complex<complex>(5,	 4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 4, 0.5, 1.5, 2.5)) == -0.0972104222_a);

		CHECK(imag(cartesian_complex<complex>(5,	 5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 5, 0.0, 3.0, 0.0)) == -0.4641322034_a);
		CHECK(imag(cartesian_complex<complex>(5,	 5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(5,	 5, 0.5, 1.5, 2.5)) == 0.0007685159_a);
		
	}
	
	SECTION("complex l=6"){
		
		CHECK(real(cartesian_complex<complex>(6, -6, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -6, 3.0, 0.0, 0.0)) == 0.4830841136_a);
		CHECK(real(cartesian_complex<complex>(6, -6, 0.0, 3.0, 0.0)) == -0.4830841136_a);
		CHECK(real(cartesian_complex<complex>(6, -6, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -6, 0.5, 1.5, 2.5)) == 0.0039660783_a);
		
		CHECK(real(cartesian_complex<complex>(6, -5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -5, 0.5, 1.5, 2.5)) == 0.0616689186_a);
		
		CHECK(real(cartesian_complex<complex>(6, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -4, 3.0, 0.0, 0.0)) == -0.3567812629_a);
		CHECK(real(cartesian_complex<complex>(6, -4, 0.0, 3.0, 0.0)) == -0.3567812629_a);
		CHECK(real(cartesian_complex<complex>(6, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -4, 0.5, 1.5, 2.5)) == 0.055920002_a);
		
		CHECK(real(cartesian_complex<complex>(6, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -3, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -3, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -3, 0.5, 1.5, 2.5)) == -0.3357604604_a);
		
		CHECK(real(cartesian_complex<complex>(6, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -2, 3.0, 0.0, 0.0)) == 0.3256952429_a);
		CHECK(real(cartesian_complex<complex>(6, -2, 0.0, 3.0, 0.0)) == -0.3256952429_a);
		CHECK(real(cartesian_complex<complex>(6, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -2, 0.5, 1.5, 2.5)) == -0.3707038567_a);
		
		CHECK(real(cartesian_complex<complex>(6, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6, -1, 0.5, 1.5, 2.5)) == 0.024021896_a);
		
		CHECK(real(cartesian_complex<complex>(6,	0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	0, 2.0, 0.0, 0.0)) == -0.3178460113_a);
		CHECK(real(cartesian_complex<complex>(6,	0, 0.0, 2.0, 0.0)) == -0.3178460113_a);
		CHECK(real(cartesian_complex<complex>(6,	0, 0.0, 0.0, 2.0)) == 1.0171072363_a);
		CHECK(real(cartesian_complex<complex>(6,	0, 0.5, 1.5, 2.5)) == -0.4151458107_a);
		
		CHECK(real(cartesian_complex<complex>(6,	1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	1, 0.5, 1.5, 2.5)) == -0.024021896_a);

		CHECK(real(cartesian_complex<complex>(6,	2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	2, 2.0, 0.0, 0.0)) == 0.3256952429_a);
		CHECK(real(cartesian_complex<complex>(6,	2, 0.0, 2.0, 0.0)) == -0.3256952429_a);
		CHECK(real(cartesian_complex<complex>(6,	2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	2, 0.5, 1.5, 2.5)) == -0.3707038567_a);

		CHECK(real(cartesian_complex<complex>(6,	3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	3, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	3, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	3, 0.5, 1.5, 2.5)) == 0.3357604604_a);
		
		CHECK(real(cartesian_complex<complex>(6,	4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	4, 3.0, 0.0, 0.0)) == -0.3567812629_a);
		CHECK(real(cartesian_complex<complex>(6,	4, 0.0, 3.0, 0.0)) == -0.3567812629_a);
		CHECK(real(cartesian_complex<complex>(6,	4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	4, 0.5, 1.5, 2.5)) == 0.055920002_a);

		CHECK(real(cartesian_complex<complex>(6,	5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	5, 0.5, 1.5, 2.5)) == -0.0616689186_a);

		CHECK(real(cartesian_complex<complex>(6,	6, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	6, 3.0, 0.0, 0.0)) == 0.4830841136_a);
		CHECK(real(cartesian_complex<complex>(6,	6, 0.0, 3.0, 0.0)) == -0.4830841136_a);
		CHECK(real(cartesian_complex<complex>(6,	6, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_complex<complex>(6,	6, 0.5, 1.5, 2.5)) == 0.0039660783_a);

		CHECK(imag(cartesian_complex<complex>(6, -6, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -6, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -6, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -6, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -6, 0.5, 1.5, 2.5)) == -0.0105461628_a);
		
		CHECK(imag(cartesian_complex<complex>(6, -5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -5, 0.5, 1.5, 2.5)) == 0.0023418577_a);
		
		CHECK(imag(cartesian_complex<complex>(6, -4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -4, 0.5, 1.5, 2.5)) == 0.1917257212_a);
		
		CHECK(imag(cartesian_complex<complex>(6, -3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -3, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -3, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -3, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -3, 0.5, 1.5, 2.5)) == 0.2324495495_a);
		
		CHECK(imag(cartesian_complex<complex>(6, -2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -2, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -2, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -2, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -2, 0.5, 1.5, 2.5)) == -0.2780278925_a);
		
		CHECK(imag(cartesian_complex<complex>(6, -1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -1, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -1, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -1, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6, -1, 0.5, 1.5, 2.5)) == -0.072065688_a);
		
		CHECK(imag(cartesian_complex<complex>(6,	0, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	0, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	0, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	0, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	0, 0.5, 1.5, 2.5)) == (0.0_a).margin(1e-12));
		
		CHECK(imag(cartesian_complex<complex>(6,	1, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	1, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	1, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	1, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	1, 0.5, 1.5, 2.5)) == -0.072065688_a);

		CHECK(imag(cartesian_complex<complex>(6,	2, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	2, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	2, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	2, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	2, 0.5, 1.5, 2.5)) == 0.2780278925_a);

		CHECK(imag(cartesian_complex<complex>(6,	3, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	3, 2.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	3, 0.0, 2.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	3, 0.0, 0.0, 2.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	3, 0.5, 1.5, 2.5)) == 0.2324495495_a);
		
		CHECK(imag(cartesian_complex<complex>(6,	4, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	4, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	4, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	4, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	4, 0.5, 1.5, 2.5)) == -0.1917257212_a);

		CHECK(imag(cartesian_complex<complex>(6,	5, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	5, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	5, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	5, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	5, 0.5, 1.5, 2.5)) == 0.0023418577_a);

		CHECK(imag(cartesian_complex<complex>(6,	6, 0.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	6, 3.0, 0.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	6, 0.0, 3.0, 0.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	6, 0.0, 0.0, 3.0)) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_complex<complex>(6,	6, 0.5, 1.5, 2.5)) == 0.0105461628_a);
		
	}
	
	SECTION("spinor j = 1/2 s = +1/2"){
		CHECK(real(cartesian_spinor<complex>(1, -1, 1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(1, -1, 1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_spinor<complex>(1, -1, 1, 0.0, 0.0, 0.0)[1]) == 0.2820947918_a);
		CHECK(imag(cartesian_spinor<complex>(1, -1, 1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));		

		CHECK(real(cartesian_spinor<complex>(1, 1, 1, 0.0, 0.0, 0.0)[0]) == 0.2820947918_a);
		CHECK(imag(cartesian_spinor<complex>(1, 1, 1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_spinor<complex>(1, 1, 1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(1, 1, 1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));

		CHECK(real(cartesian_spinor<complex>(1, -1, 1, 1.0, 2.0, 3.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(1, -1, 1, 1.0, 2.0, 3.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_spinor<complex>(1, -1, 1, 1.0, 2.0, 3.0)[1]) == 0.2820947918_a);
		CHECK(imag(cartesian_spinor<complex>(1, -1, 1, 1.0, 2.0, 3.0)[1]) == (0.0_a).margin(1e-12));		

		CHECK(real(cartesian_spinor<complex>(1, 1, 1, 1.0, 2.0, 3.0)[0]) == 0.2820947918_a);
		CHECK(imag(cartesian_spinor<complex>(1, 1, 1, 1.0, 2.0, 3.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_spinor<complex>(1, 1, 1, 1.0, 2.0, 3.0)[1]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(1, 1, 1, 1.0, 2.0, 3.0)[1]) == (0.0_a).margin(1e-12));
	}


	SECTION("spinor j = 3/2 s = -1/2"){
		CHECK(real(cartesian_spinor<complex>(3, -3, -1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(3, -3, -1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_spinor<complex>(3, -3, -1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(3, -3, -1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));		
		
		CHECK(real(cartesian_spinor<complex>(3, -1, -1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(3, -1, -1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_spinor<complex>(3, -1, -1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(3, -1, -1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));		

		CHECK(real(cartesian_spinor<complex>(3,  1, -1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(3,  1, -1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_spinor<complex>(3,  1, -1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(3,  1, -1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));

		CHECK(real(cartesian_spinor<complex>(3,  3, -1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(3,  3, -1, 0.0, 0.0, 0.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_spinor<complex>(3,  3, -1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));
		CHECK(imag(cartesian_spinor<complex>(3,  3, -1, 0.0, 0.0, 0.0)[1]) == (0.0_a).margin(1e-12));

		
		CHECK(real(cartesian_spinor<complex>(3, -3, -1, 1.0, 2.0, 3.0)[0]) ==  0.0740344606_a);
		CHECK(imag(cartesian_spinor<complex>(3, -3, -1, 1.0, 2.0, 3.0)[0]) ==  0.0987126141_a);
		CHECK(real(cartesian_spinor<complex>(3, -3, -1, 1.0, 2.0, 3.0)[1]) ==  0.0740344606_a);
		CHECK(imag(cartesian_spinor<complex>(3, -3, -1, 1.0, 2.0, 3.0)[1]) == -0.1480689212_a);		
		
		CHECK(real(cartesian_spinor<complex>(3, -1, -1, 1.0, 2.0, 3.0)[0]) == -0.1282314473_a);
		CHECK(imag(cartesian_spinor<complex>(3, -1, -1, 1.0, 2.0, 3.0)[0]) ==  0.2564628945_a);
		CHECK(real(cartesian_spinor<complex>(3, -1, -1, 1.0, 2.0, 3.0)[1]) ==  0.1852232016_a);
		CHECK(imag(cartesian_spinor<complex>(3, -1, -1, 1.0, 2.0, 3.0)[1]) ==  (0.0_a).margin(1e-12));

		CHECK(real(cartesian_spinor<complex>(3,  1, -1, 1.0, 2.0, 3.0)[0]) == -0.1852232016_a);
		CHECK(imag(cartesian_spinor<complex>(3,  1, -1, 1.0, 2.0, 3.0)[0]) == (0.0_a).margin(1e-12));
		CHECK(real(cartesian_spinor<complex>(3,  1, -1, 1.0, 2.0, 3.0)[1]) == -0.1282314473_a);
		CHECK(imag(cartesian_spinor<complex>(3,  1, -1, 1.0, 2.0, 3.0)[1]) == -0.2564628945_a);

		CHECK(real(cartesian_spinor<complex>(3,  3, -1, 1.0, 2.0, 3.0)[0]) ==  0.0740344606_a);
		CHECK(imag(cartesian_spinor<complex>(3,  3, -1, 1.0, 2.0, 3.0)[0]) ==  0.1480689212_a);
		CHECK(real(cartesian_spinor<complex>(3,  3, -1, 1.0, 2.0, 3.0)[1]) == -0.0740344606_a);
		CHECK(imag(cartesian_spinor<complex>(3,  3, -1, 1.0, 2.0, 3.0)[1]) ==  0.0987126141_a);
	}

}
