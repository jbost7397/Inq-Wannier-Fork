/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__WANNIER__JADE_COMPLEX
#define INQ__WANNIER__JADE_COMPLEX

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>
#include <math/complex.hpp>
#include <matrix/distributed.hpp>
#include <matrix/gather_scatter.hpp>
#include <parallel/communicator.hpp>
#include <gpu/run.hpp>
#include <wannier/jacobi_eigenvalue.hpp>
#include <wannier/plane_rot.hpp>
#include <utils/raw_pointer_cast.hpp>
#include <utils/profiling.hpp>

#include <vector>
#include <deque>
#include <limits>
#include <cmath>
#include <cassert>

namespace inq {
namespace wannier {

template <typename T, typename T1, class MatrixType1, class MatrixType2, class MatrixType3>      //JB: proper function declaration consistent w/inq style
auto jade_complex(T maxsweep, T1 tol, MatrixType1& a, MatrixType2& u, MatrixType3& adiag) {

    //const double eps = std::numeric_limits<double>::epsilon();
    CALI_CXX_MARK_SCOPE("wannier_jade");
    assert(tol > std::numeric_limits<double>::epsilon());

    int nloc = a[0].size();  //cols
    int mloc = a[0][0].size();  //rows //CS nloc = mloc (NxN) for all wannier routines
    int n = a.size(); // 6 for all wannier

    // Initialize u as identity
    u = std::vector<std::vector<complex>> (mloc, std::vector<complex>(nloc)); //CS same size as a[k]
    for(int m = 0; m < mloc; ++m){
      for(int n = 0; n < nloc; ++n){
        u[m][n] = complex{0.0, 0.0};
          if(m == n) u[m][n] = complex{1.0,0.0};
      } //n
    } //m

    // eigenvalue array
    adiag.resize(a.size());
    for ( int k = 0; k < a.size(); k++ ){
      adiag[k].resize(mloc);
    }

    //check if number of rows is odd
    const bool nloc_odd = (mloc % 2 != 0);

    //if nloc is odd need auxiliary arrays for an extra column
    std::vector<std::vector<complex>> a_aux(a.size());
    std::vector<complex> u_aux;
    if (nloc_odd) {
      for (int k=0; k < a.size(); ++k)
	a_aux[k].resize(mloc);
      u_aux.resize(mloc);
     }

    const int nploc = (nloc + 1) / 2; //when parallel replace nloc with column distributor
    std::deque<int> top(nploc), bot(nploc);
    int np = nploc; //CS this will always be true when non-parallel

    // initialize top and bot arrays
    // the pair i is (top[i],bot[i])
    // top[i] is the local index of the top column of pair i
    // bot[i] is the local index of the bottom column of pair i
    for (int i = 0; i < nploc; ++i) {
        top[i] = i;
        bot[nploc - i - 1] = nploc + i;
    }

    //when parallel need routine to store global column address for reordering

    //std::vec here since this will depend on parralelization
    std::vector<std::vector<complex*>> acol(a.size());
    std::vector<complex*> ucol(2*nploc);

    for (int k = 0; k < a.size(); ++k) {
      acol[k].resize(2*nploc);
      for (int i = 0; i < a[k].size(); ++i ){
        acol[k][i] = a[k][i].data(); //a[k] will always be square 
      }
      if (nloc_odd)
       acol[k][2*nploc-1] = &a_aux[k][0];
    } // for k
    for ( int i = 0; i < u.size(); ++i ) {
      ucol[i] = u[i].data();
    }
    if (nloc_odd)
      ucol[2*nploc-1] = &u_aux[0];

    int nsweep = 0;
    bool done = false;
    // allocate matrix element packed array apq
    // apq[3*ipair   + k*3*nploc] = apq[k][ipair]
    // apq[3*ipair+1 + k*3*nploc] = app[k][ipair]
    // apq[3*ipair+2 + k*3*nploc] = aqq[k][ipair]
    std::vector<complex> apq(a.size()*3*nploc);
    std::vector<double> tapq(a.size()*3*2*nploc); //CS need for summation over all

    while (!done) {
        ++nsweep;
        double diag_change = 0.0;
        // sweep local pairs and rotate 2*np -1 times
        for (int irot = 0; irot < 2 * np - 1; ++irot) {
            //jacobi rotations for local pairs
            //of diagonal elements for all pairs (apq)
            for (int k = 0; k < a.size(); ++k) {
                for (int ipair = 0; ipair < nploc; ++ipair) {
                    const int iapq = 3 * ipair + k * 3 * nploc;
                    apq[iapq] = complex(0.0, 0.0);
                    apq[iapq + 1] = complex(0.0, 0.0);
                    apq[iapq + 2] = complex(0.0, 0.0);

		    if (top[ipair] >= 0 && bot[ipair] >= 0 ){
                      const complex *ap = acol[k][top[ipair]];
                      const complex *aq = acol[k][bot[ipair]];
                      const complex *up = ucol[top[ipair]];
                      const complex *uq = ucol[bot[ipair]];
                      for (int ii = 0; ii < mloc; ++ii) {
                        apq[iapq] += conj_cplx(ap[ii]) * uq[ii];
                        apq[iapq + 1] += conj_cplx(ap[ii]) * up[ii];
                        apq[iapq + 2] += conj_cplx(aq[ii]) * uq[ii];
                        } //for ii
		    } //top bot
                } //for ipair
            } //for k

	   //now need summation routine for parallel, probably from sum.hpp
	   //sum into tapq and pass back (dsum w/qbach)
	   //or a gather, sum, scatter routine or comm_allreduce

            for (int ipair = 0; ipair < nploc; ++ipair) {
                if (top[ipair] >= 0 && bot[ipair] >= 0) {
                    double g11 = 0.0, g12 = 0.0, g13 = 0.0;
                    double g21 = 0.0, g22 = 0.0, g23 = 0.0;
                    double g31 = 0.0, g32 = 0.0, g33 = 0.0;

                    for (int k = 0; k < a.size(); ++k) {
                        const int iapq = 3 * ipair + k * 3 * nploc;
                        const complex aij = apq[iapq];
                        const complex aii = apq[iapq + 1];
                        const complex ajj = apq[iapq + 2];

                        const complex h1 = aii - ajj;
                        const complex h2 = aij + conj_cplx(aij);
                        const complex h3 = complex(0.0, 1.0) * (aij - conj_cplx(aij));

                        g11 += real(conj_cplx(h1) * h1);
                        g12 += real(conj_cplx(h1) * h2);
                        g13 += real(conj_cplx(h1) * h3);
                        g21 += real(conj_cplx(h2) * h1);
                        g22 += real(conj_cplx(h2) * h2);
                        g23 += real(conj_cplx(h2) * h3);
                        g31 += real(conj_cplx(h3) * h1);
                        g32 += real(conj_cplx(h3) * h2);
                        g33 += real(conj_cplx(h3) * h3);
                    } //for k

                    int N = 3; // For Wannier 3x3 matrix size
                    gpu::array<double,1> G = {g11, g12, g13, g21, g22, g23, g31, g32, g33};  // Matrix to be diagonalized
                    gpu::array<double,1> Q(9); // Eigenvectors
                    gpu::array<double,1> D(3); // Eigenvalues
                    jacobi_eigenvalue(N, G, Q, D);

                    // Extract the largest eigenvalue's vector
                    double x = Q[6], y = Q[7], z = Q[8];
                    if (x < 0.0) {
                        x = -x; y = -y; z = -z;
                    }

		    double one = 1.0;
                    double r = sqroot((x + one) / 2.0); 
                    complex c = complex(r, 0.0);
                    complex s = complex(y / (2.0 * r), -z / (2.0 * r));
                    complex sconj = conj_cplx(s);

                    for (int k = 0; k < a.size(); ++k) {
                      complex *ap = acol[k][top[ipair]];
                      complex *aq = acol[k][bot[ipair]];
                      //Apply plane rotation
                      //plane_rot(ap, aq, c, sconj); //CS skip using plane_rot, probably until clarity on cublas zrot functionality 
		      //routine now internal
		      std::vector<complex> ap_tmp(mloc);
                      std::vector<complex> aq_tmp(mloc);
            		for (int ii = 0; ii < mloc; ++ii) {
              		  ap_tmp[ii] = c * ap[ii] + sconj * aq[ii];
          		  aq_tmp[ii] = -s*ap[ii] + c * aq[ii];
            		}
            		for (int ii = 0; ii < mloc; ++ii) {
              		ap[ii] = ap_tmp[ii];
              		aq[ii] = aq_tmp[ii];
              		}
		    }

		    //rotate u 
                    //plane_rot(up, uq, c, sconj);
          	    complex *up = ucol[top[ipair]];
         	    complex *uq = ucol[bot[ipair]];
          	    std::vector<complex> up_tmp(mloc);
          	    std::vector<complex> uq_tmp(mloc);
          	    for (int ii = 0; ii < mloc; ++ii) {
            	      up_tmp[ii] = c * up[ii] + sconj * uq[ii];
            	      uq_tmp[ii] = -s*up[ii] + c * uq[ii];
          	     }
          	     for (int ii = 0; ii < mloc; ++ii) {
            	       up[ii] = up_tmp[ii];
            	       uq[ii] = uq_tmp[ii];
                     }

                    // new value of off-diag element apq
                    double diag_change_ipair = 0.0;
                    for (int k = 0; k < a.size(); ++k) {
                        const int iapq = 3 * ipair + k * 3 * nploc;
                        const complex aii = apq[iapq + 1];
                        const complex ajj = apq[iapq + 2];
                        const complex v1 = conj_cplx(c) * c - sconj * s;

                        double apq_new = real(v1 * (aii - ajj) + 2.0 * c * s * apq[iapq] + 2.0 * sconj * c * apq[iapq]);
                        diag_change_ipair += 2.0 * fabs(apq_new - real(aii - ajj));
			}
                    diag_change += diag_change_ipair;
                }
            }//for ipair

            // Rotate top and bot arrays
            if (nploc > 0) {
                    bot.push_back(top.back());
                    top.pop_back();
                    top.push_front(bot.front());
                    bot.pop_front();
            	    if (nploc > 1) {
	              std::swap(top[0], top[1]);
	            } else {
	              std::swap(top[0], bot[0]);
	            } 
	      } //if nploc >0 
	} //irot
       done = (fabs(diag_change) < tol) || (nsweep >= maxsweep);
    } //while 

    // Compute diagonal elements
  for (int k = 0; k < a.size(); ++k) {
    for (int i = 0; i < a[k].size(); ++i) {
      adiag[k][i] = complex(0.0, 0.0);    
    }
    for (int i = 0; i < a[k].size(); ++i) {
      const complex *ap = acol[k][i];
      const complex *up = ucol[i];
      for (int ii = 0; ii < mloc; ii++)
      {
        adiag[k][i] += conj(ap[ii])*up[ii];
      }
   }
 }
    return adiag;
    //return nsweep 

} //jade_complex
} // namespace wannier
} // namespace inq

#endif
///////////////////////////////////////////////////////////////////
#ifdef INQ_WANNIER_JADE_COMPLEX_UNIT_TEST
#undef INQ_WANNIER_JADE_COMPLEX_UNIT_TEST

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

    using namespace inq;
    using namespace Catch::literals;
    using Catch::Approx;

    int maxsweep = 100;
    double tol = 1e-6;

    // Create a vector of 6 1x1 matrices (H2 test case) //coressponds to gs in a 20x20x20 cell
    std::vector<std::vector<std::vector<complex>>> a(6, std::vector<std::vector<complex>>(2, std::vector<complex>(2)));

    // Fill a mats
    a[0][0][0] = complex(-0.68433137,-0.00000000);
    a[0][1][0] = complex(-0.00103429,0.10810869);
    a[0][0][1] = complex(-0.00103429,-0.10810876);
    a[0][1][1] = complex(-0.68104163,0.00000000);
    a[1][0][0] = complex(-0.09765152,0.00000003);
    a[1][1][0] = complex(0.00727211,-0.68256214);
    a[1][0][1] = complex(0.00727210,0.68256259);
    a[1][1][1] = complex(-0.11860232,-0.00000003);
    a[2][0][0] = complex(-0.68433137,-0.00000000);
    a[2][1][0] = complex(-0.00103429,0.10810870);
    a[2][0][1] = complex(-0.00103429,-0.10810877);
    a[2][1][1] = complex(-0.68104162,0.00000000);
    a[3][0][0] = complex(-0.09765152,0.00000003);
    a[3][1][0] = complex(0.00727211,-0.68256215);
    a[3][0][1] = complex(0.00727210,0.68256259);
    a[3][1][1] = complex(-0.11860232,-0.00000003);
    a[4][0][0] = complex(-0.68433130,-0.00000000);
    a[4][1][0] = complex(-0.00103425,0.10810870);
    a[4][0][1] = complex(-0.00103453,-0.10810874);
    a[4][1][1] = complex(-0.68104179,0.00000005);
    a[5][0][0] = complex(-0.09765150,0.00000002);
    a[5][1][0] = complex(0.00727212,-0.68256221);
    a[5][0][1] = complex(0.00727207,0.68256246);
    a[5][1][1] = complex(-0.11860233,-0.00000030);

    // Create matrix u (initially identity)
    std::vector<std::vector<complex>> u(2, std::vector<complex>(2));
    u[0][0] = complex(1.0, 0.0);  // Identity element
    u[0][1] = complex(0.0, 0.0);
    u[1][0] = complex(0.0, 0.0);
    u[1][1] = complex(1.0, 0.0);  // Identity element

    // Prepare adiag to hold diagonal elements (size should match number of a matrices and their dimensions)
    std::vector<std::vector<complex>> adiag(a.size(), std::vector<complex>(2)); // Assuming single diagonal element per input matrix

    // Call the jade_complex function
    auto sweep = wannier::jade_complex(maxsweep, tol, a, u, adiag);

    	CHECK(u.size() == 2);
    	CHECK(adiag.size() == 6);
    	CHECK(adiag[0].size() == 2);
        /*CHECK(real(sweep[0]) == -0.00103429_a);
	CHECK(imag(sweep[0]) == 0.10810876_a);
        CHECK(real(sweep[1]) == -0.68433137_a);
        CHECK(imag(sweep[1]) == 0.00000000_a);
        CHECK(real(sweep[2]) == -0.68104163_a);
        CHECK(imag(sweep[2]) == 0.00000000_a);
        CHECK(real(sweep[3]) == 0.00727210_a);
        CHECK(imag(sweep[3]) ==-0.68256259_a);
        CHECK(real(sweep[4]) == -0.09765152_a);
        CHECK(imag(sweep[4]) == -0.00000003_a);
        CHECK(real(sweep[5]) == -0.11860232_a);
        CHECK(imag(sweep[5]) == 0.00000003_a);
        CHECK(real(sweep[6]) == -0.00103429_a);
        CHECK(imag(sweep[6]) == 0.10810877_a);
        CHECK(real(sweep[7]) == -0.68433137_a);
        CHECK(imag(sweep[7]) == 0.0000000_a);
        CHECK(real(sweep[8]) == -0.68104162_a);
        CHECK(imag(sweep[8]) == 0.0000000_a);*/ //for apq 0-8, all check out 	
        //CHECK(real(sweep) == 0.00590959_a ); 
	//CHECK(imag(sweep) == -0.55484695_a ); //acol is correct upon return (this is for acol[1][1]) thus a is returned correctly 
	//CHECK(sweep == 2);
        CHECK(real(sweep[0][0]) == -0.79081263_a);
        CHECK(imag(sweep[0][0]) == 0.00000000_a);
        CHECK(real(sweep[0][1]) == -0.57456037_a);
        CHECK(imag(sweep[0][1]) == -0.00000000_a);
        CHECK(real(sweep[1][0]) == 0.57455456_a);
        CHECK(imag(sweep[1][0]) == -0.00000001_a);
        CHECK(real(sweep[1][1]) == -0.79080839_a);
        CHECK(imag(sweep[1][1]) == 0.00000001_a);
        CHECK(real(sweep[2][0]) == -0.79081263_a);
        CHECK(imag(sweep[2][0]) == 0.00000000_a);
        CHECK(real(sweep[2][1]) == -0.57456037_a);
        CHECK(imag(sweep[2][1]) == -0.00000000_a);
        CHECK(real(sweep[3][0]) == 0.57455456_a);
        CHECK(imag(sweep[3][0]) == -0.00000001_a);
        CHECK(real(sweep[3][1]) == -0.79080840_a);
        CHECK(imag(sweep[3][1]) == 0.00000001_a);
}
#endif
