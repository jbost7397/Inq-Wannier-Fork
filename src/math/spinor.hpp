/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MATH__SPINOR
#define INQ__MATH__SPINOR

/*
 Copyright (C) 2023 Xavier Andrade

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.
  
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
  
 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <inq_config.h>

#include <math/complex.hpp>

namespace inq {

class spinor {

	complex comp_[2];

public:
	
	GPU_FUNCTION spinor() = default;

	GPU_FUNCTION spinor(complex const & aa, complex const & bb){
		comp_[0] = aa;
		comp_[1] = bb;	
	}

	GPU_FUNCTION spinor(double const & aa, double const & bb, double const & cc, double const & dd){
		comp_[0] = complex{aa, bb};
		comp_[1] = complex{cc, dd};	
	}
	
	GPU_FUNCTION constexpr auto operator[](int ii) const {
		assert(ii >= 0 and ii < 2);
		return comp_[ii];
	}
	
	GPU_FUNCTION friend auto operator+(spinor const & aa, spinor const & bb){
		return spinor{aa[0] + bb[0], aa[1] + bb[1]};
	}

	GPU_FUNCTION friend auto operator-(spinor const & aa, spinor const & bb){
		return spinor{aa[0] - bb[0], aa[1] - bb[1]};
	}
	
	GPU_FUNCTION friend auto operator*(spinor const & aa, spinor const & bb){
		return spinor{aa[0]*bb[0], aa[1]*bb[1]};
	}

};

}

#ifdef INQ_MATH_SPINOR_UNIT_TEST
#undef INQ_MATH_SPINOR_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE("Class math::spinor", "[math::spinor]"){

	using namespace inq;
	using namespace Catch::literals;

	spinor uu;
	
}

#endif
#endif
