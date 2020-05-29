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
	auto divergence(basis::field_set<basis::real_space, complex> const & ff){	// Divergence function for the field-set type defined in the Real space which return a filed 'diverg_real'
		auto ff_fourier = operations::space::to_fourier(ff); 			// Tranform input field-set to Fourier space
		auto diverg_fourier = diverg(ff_fourier); 				// To calculate the divergence in Fourier space with the use of the above-defined function 'diverg'
		auto diverg_real = operations::space::to_real(diverg_fourier); 	// Transform output field to Real space
		return diverg_real;
		}
	auto divergence(basis::field_set<basis::real_space, double> const & ff){	// Divergence function for the field-set type defined in the Real space which return a filed 'diverg_real'
		auto ff_fourier = operations::space::to_fourier(ff); 			// Tranform input field-set to Fourier space
		auto diverg_fourier = diverg(ff_fourier); 				// To calculate the divergence in Fourier space with the use of the above-defined function 'diverg'
		auto diverg_real = operations::space::to_real(diverg_fourier); 		// Transform output field to Real space
		return diverg_real.real();						// Return a real part off the divergency in the real space
		}

	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef UNIT_TEST

#include <catch2/catch.hpp>
#include <math/vec3d.hpp>

        //Define test function 1
        complex f_analytic (math::vec3d k , math::vec3d r){
                using math::vec3d;
                complex f;
                f = exp(complex(0.0,1.0)*(k | r ));
                return f;
        }

        //Define analytic form of the gradient of the test function 1
        complex* g_analytic (math::vec3d k , math::vec3d r) {
                using math::vec3d;
                static complex g[3];
                complex factor = complex(0.0, 1.0)*exp(complex(0.0,1.0)*(k | r ));
                for(int idir = 0; idir < 3 ; idir++) g [idir] = factor * k [idir] ;
                return g;
        }

        //Define test function 2
        double f_analytic2 (math::vec3d k , math::vec3d r){
                using math::vec3d;
                double f;
                f = sin(k | r );
                return f;
        }

        //Define analytic form of the gradient of the test function 1
        double* g_analytic2 (math::vec3d k , math::vec3d r) {
                using math::vec3d;
                static double g[3];
                for(int idir = 0; idir < 3 ; idir++) g [idir] = k [idir] * cos (k | r);
                return g;
        }

#include <catch2/catch.hpp>

TEST_CASE("function operations::divergence", "[operations::divergence]") {

//	using namespace Catch::literals;

}


#endif
#endif

