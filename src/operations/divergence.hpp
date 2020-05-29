/* -*- indent-tabs-mode: t -*- */

#ifndef OPERATIONS__DIVERGENCE
#define OPERATIONS__DIVERGENCE

/*
 Copyright (C) 2020 Xavier Andrade, Alfredo A. Correa, Alexey Karstev.

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

#include <basis/field.hpp>
#include <cassert>

namespace operations {
	auto divergence(basis::field_set<basis::fourier_space, complex> const & ff){ // Divergence function for the field-set type defined in the Fourier space which return a filed 'diverg'
		basis::field<basis::fourier_space, complex> diverg(ff.basis());
		for(int ix = 0; ix < ff.basis().sizes()[0]; ix++){		// Iterating over x-,y- and z- components of the input field 
			for(int iy = 0; iy < ff.basis().sizes()[1]; iy++){
				for(int iz = 0; iz < ff.basis().sizes()[2]; iz++){
					auto gvec = ff.basis().gvector(ix, iy, iz);
					// Iterating over each vectorial components of input field-set and corresponding G-vector at (ix,iy,iz) point in the space
					for(int idir = 0; idir < 3 ; idir++) diverg.cubic()[ix][iy][iz] += complex(0.0, 1.0)*gvec[idir]*ff.cubic()[ix][iy][iz][idir]; 
				}
			}
		}
		return diverg;
	}
	auto divergence(basis::field_set<basis::real_space, complex> const & ff){	// Divergence function for the field-set type defined in the Real space which return a filed 'diverg_real'
		auto ff_fourier = operations::space::to_fourier(ff); 			// Tranform input field-set to Fourier space
		auto diverg_fourier = divergence(ff_fourier); 				// To calculate the divergence in Fourier space with the use of the above-defined function 'diverg'
		auto diverg_real = operations::space::to_real(diverg_fourier); 	// Transform output field to Real space
		return diverg_real;
		}
	auto divergence(basis::field_set<basis::real_space, double> const & ff){	// Divergence function for the field-set type defined in the Real space which return a filed 'diverg_real'
		auto ff_fourier = operations::space::to_fourier(ff.complex()); 			// Tranform input field-set to Fourier space
		auto diverg_fourier = divergence(ff_fourier); 				// To calculate the divergence in Fourier space with the use of the above-defined function 'diverg'
		auto diverg_real = operations::space::to_real(diverg_fourier); 		// Transform output field to Real space
		return diverg_real.real();						// Return a real part off the divergency in the real space
		}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef INQ_UNIT_TEST

#include <catch2/catch.hpp>
#include <math/vec3d.hpp>

        //Define test function 1
        complex df_analytic (math::vec3d k , math::vec3d r , int idir){
                using math::vec3d;
                complex f = 0.0;
		if ( idir == 0){f = 1.0 * exp(complex(0.0, 1.0) * (k | r ));}
		if ( idir == 1){f = 2.0 * exp(complex(0.0, 1.0) * (k | r ));}
		if ( idir == 2){f = 3.0 * exp(complex(0.0, 1.0) * (k | r ));}
                return f;
        }

        //Define analytic form of the divergence of the test function 1
        complex dg_analytic (math::vec3d k , math::vec3d r) {
                using math::vec3d;
                complex g = 0.0;
                complex factor = complex(0.0, 1.0)*exp(complex(0.0,1.0)*(k | r ));
                g = (1.0 * factor * k[0]) +(2.0 * factor * k[1]) + (3.0 * factor * k[2]);
                return g;
        }

        //Define test function 2
        double df_analytic2 (math::vec3d k , math::vec3d r, int idir){
                using math::vec3d;
                double f = 0.0;
		if ( idir == 0){f = 1.0 * sin(k | r );}
		if ( idir == 1){f = 2.0 * sin(k | r );}
		if ( idir == 2){f = 3.0 * sin(k | r );}
                return f;
        }

        //Define analytic form of the divergence of the test function 1
        double dg_analytic2 (math::vec3d k , math::vec3d r) {
                using math::vec3d;
                double g = 0.0;
                g = (1.0 * k [0] * cos (k | r)) + (2.0 * k [1] * cos (k | r)) + (3.0 * k [2] * cos (k | r));
                return g;
        }


TEST_CASE("function operations::divergence", "[operations::divergence]") {

	using namespace Catch::literals;
	using namespace operations;
	using math::vec3d;

	//UnitCell size
	double lx = 9;
	double ly = 12;
	double lz = 10;

	ions::geometry geo;
 	ions::UnitCell cell(vec3d(lx, 0.0, 0.0), vec3d(0.0, ly, 0.0), vec3d(0.0, 0.0, lz));

	basis::real_space rs(cell, input::basis::cutoff_energy(20.0));

	SECTION("Vectored plane-wave"){ 
		basis::field_set<basis::real_space, complex> f_test(rs, 3);
	
		//Define k-vector for test function
		vec3d kvec = 2.0 * M_PI * vec3d(1.0/lx, 1.0/ly, 1.0/lz);

		for(int ix = 0; ix < rs.sizes()[0]; ix++){ 			// Iterating over each x-,y- and z- components of the input field 
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					auto vec = rs.rvector(ix, iy, iz);
					for(int idir = 0; idir < 3 ; idir++) f_test.cubic()[ix][iy][iz][idir] = df_analytic (kvec, vec, idir);
				}
			}
		}

		auto g_test = divergence(f_test);

		double diff = 0.0;
		for(int ix = 0; ix < rs.sizes()[0]; ix++){ 			// Iterating over each x-,y- and z- components of the input field-set 
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					auto vec = rs.rvector(ix, iy, iz);
					diff += abs(g_test.cubic()[ix][iy][iz] - dg_analytic (kvec, vec));
				}
			}
		}
		CHECK( diff < 1.0e-10 ); 
	}
	
	SECTION("Vectored real function"){

		basis::field_set<basis::real_space, double> f_test2(rs , 3);
	
		//Define k-vector for test function
		vec3d kvec = 2.0 * M_PI * vec3d(1.0/lx, 1.0/ly, 1.0/lz);

		for(int ix = 0; ix < rs.sizes()[0]; ix++){ 			// Iterating over each x-,y- and z- components of the input field 
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					auto vec = rs.rvector(ix, iy, iz);
					for(int idir = 0; idir < 3 ; idir++) f_test2.cubic()[ix][iy][iz][idir] = df_analytic2 (kvec, vec, idir);
				}
			}
		}
		
		auto g_test2 = divergence(f_test2);

		double diff2 = 0.0;
		for(int ix = 0; ix < rs.sizes()[0]; ix++){ 			// Iterating over each x-,y- and z- components of the input field-set 
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					auto vec = rs.rvector(ix, iy, iz);
					diff2 += abs(g_test2.cubic()[ix][iy][iz] - dg_analytic2 (kvec, vec));
				}
			}
		}
		CHECK( diff2 < 1.0e-10 ); 
	}
}

#endif
#endif

