/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__BASIS__CONTAINING_CUBE
#define INQ__BASIS__CONTAINING_CUBE

/*
 Copyright (C) 2019-2021 Xavier Andrade

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

#include <math/vector3.hpp>

namespace inq {
namespace basis {

//returns the cube that contains the sphere, this makes the initialization O(1) instead of O(N)
template <class BasisType, typename PosType>
void containing_cube(const BasisType & grid, PosType const & pos, double radius, math::vector3<int> & lo, math::vector3<int> & hi){

	for(int idir = 0; idir < 3; idir++){
		auto rec = grid.cell().reciprocal(idir);
		auto lointer = pos - radius/length(rec)*rec;
		auto hiinter = pos + radius/length(rec)*rec;

		auto dlo = grid.cell().metric().to_contravariant(lointer)[idir];
		auto dhi = grid.cell().metric().to_contravariant(hiinter)[idir];

		lo[idir] = floor(dlo/grid.contravariant_spacing()[idir]);
		hi[idir] = ceil(dhi/grid.contravariant_spacing()[idir]) + 1;
		
		lo[idir] = std::clamp(lo[idir], grid.symmetric_range_begin(idir), grid.symmetric_range_end(idir));
		hi[idir] = std::clamp(hi[idir], grid.symmetric_range_begin(idir), grid.symmetric_range_end(idir));
	}
	
}

}
}

#ifdef INQ_BASIS_CONTAINING_CUBE_UNIT_TEST
#undef INQ_BASIS_CONTAINING_CUBE_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <ions/unit_cell.hpp>
#include <math/array.hpp>
#include <basis/real_space.hpp>

TEST_CASE("class basis::containing_cube", "[basis::containing_cube]") {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	using math::vector3;

	auto comm = boost::mpi3::environment::get_world_instance();
	
	SECTION("Orthogonal box"){
		systems::box box = systems::box::orthorhombic(12.0_b, 14.0_b, 16.0_b).cutoff_energy(45.0_Ha);
		basis::real_space rs(box, comm);

		auto center = math::vector3{3.0, 2.0, 1.0};
		auto radius = 3.0;

		math::vector3<int> lo, hi;
		containing_cube(rs, center, radius, lo, hi);

		CHECK(lo[0] == 0);
		CHECK(lo[1] == -3);
		CHECK(lo[2] == -6);
		CHECK(hi[0] == 18);
		CHECK(hi[1] == 17);
		CHECK(hi[2] == 13);		
		
		for(int ix = 0; ix < rs.sizes()[0]; ix++){
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					auto ii = rs.point_op().to_symmetric_range(ix, iy, iz);
					
					if(ii[0] >= lo[0] and ii[0] < hi[0] and
						 ii[1] >= lo[1] and ii[1] < hi[1] and
						 ii[2] >= lo[2] and ii[2] < hi[2]) continue;

					auto dist2 = norm(center - rs.point_op().rvector_cartesian(parallel::global_index(ix), parallel::global_index(iy), parallel::global_index(iz)));
					CHECK(dist2 > radius*radius);
				}
			}
		}
		
	}
	
	SECTION("Non-orthogonal box"){

		auto aa = 23.3_b;
		systems::box box = systems::box::lattice({0.0_b, aa/2.0, aa/2.0}, {aa/2, 0.0_b, aa/2.0}, {aa/2.0, aa/2.0, 0.0_b}).cutoff_energy(75.0_Ha);
		basis::real_space rs(box, comm);

		auto center = math::vector3{-0.5, 0.666, -1.0};
		auto radius = 4.2;

		math::vector3<int> lo, hi;
		containing_cube(rs, center, radius, lo, hi);

		CHECK(lo[0] == -20);
		CHECK(lo[1] == -26);
		CHECK(lo[2] == -17);
		CHECK(hi[0] == 22);
		CHECK(hi[1] == 16);
		CHECK(hi[2] == 25);

		for(int ix = 0; ix < rs.sizes()[0]; ix++){
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					
					auto ii = rs.to_symmetric_range(ix, iy, iz);
					
					if(ii[0] >= lo[0] and ii[0] < hi[0] and
						 ii[1] >= lo[1] and ii[1] < hi[1] and
						 ii[2] >= lo[2] and ii[2] < hi[2]) continue;

					auto dist2 = norm(center - rs.point_op().rvector_cartesian(parallel::global_index(ix), parallel::global_index(iy), parallel::global_index(iz)));
					CHECK(dist2 > radius*radius);
				}
			}
		}
				
	}

	SECTION("Non-orthogonal box 2"){

		auto aa = 5.5_b;
		systems::box box = systems::box::lattice({0.0_b, aa/2.0, aa/2.0}, {aa/2, 0.0_b, aa/2.0}, {aa/2.0, aa/2.0, 0.0_b}).cutoff_energy(102.0_Ha);
		basis::real_space rs(box, comm);

		auto center = math::vector3{-0.5, 0.666, -1.0};
		auto radius = 4.2;

		math::vector3<int> lo, hi;
		containing_cube(rs, center, radius, lo, hi);

		CHECK(lo[0] == -9);
		CHECK(lo[1] == -9);
		CHECK(lo[2] == -9);
		CHECK(hi[0] == 9);
		CHECK(hi[1] == 9);
		CHECK(hi[2] == 9);		
		
		for(int ix = 0; ix < rs.sizes()[0]; ix++){
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					auto ii = rs.point_op().to_symmetric_range(ix, iy, iz);
					
					if(ii[0] >= lo[0] and ii[0] < hi[0] and
						 ii[1] >= lo[1] and ii[1] < hi[1] and
						 ii[2] >= lo[2] and ii[2] < hi[2]) continue;

					auto dist2 = norm(center - rs.point_op().rvector_cartesian(parallel::global_index(ix), parallel::global_index(iy), parallel::global_index(iz)));					
					CHECK(dist2 > radius*radius);
				}
			}
		}
		
	}

}
#endif

#endif
