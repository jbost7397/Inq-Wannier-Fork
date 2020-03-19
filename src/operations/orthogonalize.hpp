/* -*- indent-tabs-mode: t -*- */

#ifndef OPERATIONS__ORTHOGONALIZE
#define OPERATIONS__ORTHOGONALIZE

/*
 Copyright (C) 2019 Xavier Andrade, Alfredo A. Correa

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

#include <config.h>
#include <math/complex.hpp>
#include <basis/field_set.hpp>
#include <cstdlib>
#include <multi/adaptors/blas/trsm.hpp>

namespace operations {

	template <class field_set_type>
  void orthogonalize(field_set_type & phi){

		auto olap = overlap_slate(phi);

		slate::potrf(olap);

		auto olap_triangular = slate::TriangularMatrix<typename field_set_type::element_type>(slate::Diag::NonUnit, olap);
		auto phi_matrix = phi.as_slate_matrix();
		
		slate::trsm(slate::Side::Left, (typename field_set_type::element_type) 1.0, olap_triangular, phi_matrix);

  }
	
	template <class field_set_type>
  void orthogonalize_single(field_set_type & vec, field_set_type const & phi, int num_states = -1){

		if(num_states == -1) num_states = phi.set_size();
		
		assert(num_states <= phi.set_size());
		
		for(int ist = 0; ist < num_states; ist++){


			typename field_set_type::element_type olap = 0.0;
			typename field_set_type::element_type norm = 0.0;
			for(long ip = 0; ip < phi.basis().size(); ip++){
				olap += conj(phi.matrix()[ip][ist])*vec.matrix()[ip][0];
				norm += conj(phi.matrix()[ip][ist])*phi.matrix()[ip][ist];
			}

			//reduce olap, norm

			for(long ip = 0; ip < phi.basis().size(); ip++)	vec.matrix()[ip][0] -= olap/real(norm)*phi.matrix()[ip][ist];

#if 0
			{
				typename field_set_type::element_type olap = 0.0;
				
				for(long ip = 0; ip < phi.basis().size(); ip++){
					olap += conj(phi.matrix()[ip][ist])*vec.matrix()[ip][0];
				}
				
				//reduce olap, norm
				
				std::cout << ist << '\t' << num_states << '\t' << fabs(olap) << std::endl;
			}
#endif
			
		}
		
	}
}

#ifdef UNIT_TEST
#include <catch2/catch.hpp>

#include <operations/randomize.hpp>

TEST_CASE("function operations::orthogonalize", "[operations::orthogonalize]") {

	using namespace Catch::literals;
	using math::vec3d;

	double ecut = 25.0;
	double ll = 6.3;

	auto comm = boost::mpi3::environment::get_world_instance();
	boost::mpi3::cartesian_communicator<2> cart_comm(comm, {1, comm.size()});
	
	auto basis_comm = cart_comm.axis(1);
		
	ions::UnitCell cell(vec3d(ll, 0.0, 0.0), vec3d(0.0, ll, 0.0), vec3d(0.0, 0.0, ll));
	basis::real_space basis(cell, input::basis::cutoff_energy(ecut), basis_comm);

	SECTION("Dimension 3"){
		basis::field_set<basis::real_space, complex> phi(basis, 3, cart_comm);
		
		operations::randomize(phi);
		
		operations::orthogonalize(phi);
		
		auto olap = operations::overlap(phi);
		
		std::cout << "------" << std::endl;
		
		std::cout << olap[0][0] << '\t' << olap[0][1] << '\t' << olap[0][2] << std::endl;
		std::cout << olap[1][0] << '\t' << olap[1][1] << '\t' << olap[1][2] << std::endl;
		std::cout << olap[2][0] << '\t' << olap[2][1] << '\t' << olap[2][2] << std::endl;
		
		
		for(int ii = 0; ii < phi.set_size(); ii++){
			for(int jj = 0; jj < phi.set_size(); jj++){
				if(ii == jj) {
					REQUIRE(real(olap[ii][ii]) == 1.0_a);
					REQUIRE(fabs(imag(olap[ii][ii])) < 1e-14);
			} else {
					REQUIRE(fabs(olap[ii][jj]) < 1e-14);
				}
			}
		}
	}

	SECTION("Dimension 100"){
		basis::field_set<basis::real_space, complex> phi(basis, 100, cart_comm);
		
		operations::randomize(phi);
		
		operations::orthogonalize(phi);
		
		auto olap = operations::overlap(phi);
		
		for(int ii = 0; ii < phi.set_size(); ii++){
			for(int jj = 0; jj < phi.set_size(); jj++){
				if(ii == jj) {
					REQUIRE(real(olap[ii][ii]) == 1.0_a);
					REQUIRE(fabs(imag(olap[ii][ii])) < 1e-14);
				} else {
					REQUIRE(fabs(olap[ii][jj]) < 1e-13);
				}
			}
		}
	}


	SECTION("Dimension 37 - double orthogonalize"){
		basis::field_set<basis::real_space, complex> phi(basis, 37, cart_comm);
		
		operations::randomize(phi);
		
		operations::orthogonalize(phi);
		operations::orthogonalize(phi);
		
		auto olap = operations::overlap(phi);
		
		for(int ii = 0; ii < phi.set_size(); ii++){
			for(int jj = 0; jj < phi.set_size(); jj++){
				if(ii == jj) {
					REQUIRE(real(olap[ii][ii]) == 1.0_a);
					REQUIRE(fabs(imag(olap[ii][ii])) < 1e-16);
				} else {
					REQUIRE(fabs(olap[ii][jj]) < 5e-16);
				}
			}
		}
	}
	/*
	SECTION("single -- Dimension 3"){
		basis::field_set<basis::real_space, complex> phi(basis, 3, cart_comm);
		basis::field_set<basis::real_space, complex> vec(basis, 1, cart_comm);
		
		operations::randomize(phi);
		operations::orthogonalize(phi);
		
		operations::randomize(vec);

		operations::orthogonalize_single(vec, phi, 2);
		
		operations::randomize(vec);
		
		operations::orthogonalize_single(vec, phi);
		
		}*/
}


#endif

#endif
