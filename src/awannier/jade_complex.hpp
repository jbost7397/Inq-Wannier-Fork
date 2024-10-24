/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__AWANNIER__JADE_COMPLEX
#define INQ__AWANNIER__JADE_COMPLEX

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
#include <awannier/jacobi_eigenvalue.hpp>
#include <awannier/plane_rot.hpp>
#include <utils/raw_pointer_cast.hpp>

#include <vector>
#include <deque>
#include <limits>
#include <cmath>
#include <cassert>

namespace inq {
namespace wannier {

template <typename T, typename T1, class MatrixType1, class MatrixType2, class MatrixType3>      //JB: proper function declaration consistent w/inq style
auto jade_complex(T maxsweep, T1 tol, MatrixType1& a, MatrixType2& u, MatrixType3& adiag) {

    const double eps = std::numeric_limits<double>::epsilon();
    assert(tol > eps);

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
        acol[k][i] = &a[k][i][i*mloc]; //a[k] will always be square
      }
      if (nloc_odd)
       acol[k][2*nploc-1] = &a_aux[k][0];
    } // for k
    for ( int i = 0; i < u.size(); ++i ) {
      ucol[i] = &u[i][i*mloc];
    }
    if (nloc_odd)
      ucol[2*nploc-1] = &u_aux[0];

    int nsweep = 0;
    bool done = false;
    double diag_change = 0.0;
    // allocate matrix element packed array apq
    // apq[3*ipair   + k*3*nploc] = apq[k][ipair]
    // apq[3*ipair+1 + k*3*nploc] = app[k][ipair]
    // apq[3*ipair+2 + k*3*nploc] = aqq[k][ipair]
    std::vector<complex> apq(a.size()*3*nploc);
    std::vector<double> tapq(a.size()*3*2*nploc); //CS need for summation over all

    while (!done && nsweep < maxsweep) {
        ++nsweep;
       // sweep local pairs and rotate 2*np -1 times
        for (int irot = 0; irot < 2 * np - 1; ++irot) {
            //jacobi rotations for local pairs
            //of diagonal elements for all pairs (apq)
            for (int k = 0; k < a.size(); ++k) {
                for (int ipair = 0; ipair < nploc; ++ipair) {
                    const int iapq = 3 * ipair + k * 3 * nploc;
                    apq[iapq] = complex{0.0, 0.0};
                    apq[iapq + 1] = complex{0.0, 0.0};
                    apq[iapq + 2] = complex{0.0, 0.0};

		    if (top[ipair] >= 0 && bot[ipair] >= 0 ){
                    const complex* ap = acol[k][top[ipair]];
                    const complex* aq = acol[k][bot[ipair]];
                    const complex* up = ucol[top[ipair]];
                    const complex* uq = ucol[bot[ipair]];

                    for (int ii = 0; ii < mloc; ++ii) {
                      apq[iapq] += conj_cplx(ap[ii]) * uq[ii];
                      apq[iapq + 1] += conj_cplx(ap[ii]) * up[ii];
                      apq[iapq + 2] += conj_cplx(aq[ii]) * uq[ii];
                    } //for ii
		  } //top bot
                } //for ipair
            }; //for k
//CS all correct until here
  }
}
/*
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

                    double r = sqroot((x + 1) / 2.0);
                    complex c = complex(r, 0.0);
                    complex s = complex(y / (2.0 * r), -z / (2.0 * r));
                    complex sconj = conj_cplx(s);

                    for (int k = 0; k < a.size(); ++k) {
                        complex* tmp = (top[ipair] < mloc) ? acol[k][top[ipair]*mloc] : &a_aux[k][0];
                        gpu::array<complex,1> ap = {*tmp};
			complex* tmp1 = (bot[ipair] < mloc) ? acol[k][bot[ipair]*mloc] : &a_aux[k][0];
			gpu::array<complex,1> aq = {*tmp1};
			delete tmp;
			delete tmp1;

                        // Apply plane rotation
                        plane_rot(ap, aq, c, sconj);
                    }

                    complex* tmp2 = {(top[ipair] < mloc) ? ucol[top[ipair]*mloc] : &u_aux[0]};
		    gpu::array<complex,1> up = {*tmp2};
                    complex* tmp3 = {(bot[ipair] < mloc) ? ucol[bot[ipair]*mloc] : &u_aux[0]};
		    gpu::array<complex,1> uq = {*tmp3};
		    delete tmp2;
		    delete tmp3;

                    plane_rot(up, uq, c, sconj);

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
            }
	    if (nploc > 1) {
	      std::swap(top[0], top[1]);
	    } else {
	      std::swap(top[0], bot[1]);
	    }
	} //irot
        done = (fabs(diag_change) < tol) || (nsweep >= maxsweep);
    } //while */
/*
    // Reorder matrix columns and compute diagonal elements
    std::vector<complex> tmpmat(nloc * mloc);
    for (int k = 0; k < a.size(); ++k) {
        for (int ipair = 0; ipair < nploc; ++ipair) {
            if (top[ipair] >= 0) {
                std::copy(a[k].begin() + top[ipair] * mloc, a[k].begin() + (top[ipair] + 1) * mloc, tmpmat.begin() + ipair * mloc);
            }
        }

        for (int ipair = 0; ipair < nploc; ++ipair) {
            if (bot[nploc-ipair-1] >= 0) {
                std::copy(a[k].begin() + bot[ipair] * mloc, a[k].begin() + (bot[ipair] + 1) * mloc, tmpmat.begin() + (nploc + ipair) * mloc);
            }
        }
    }

    // Store diagonal elements
    for (int k = 0; k < a.size(); ++k) {
        for (int ipair = 0; ipair < nploc; ++ipair) {
            adiag[k][ipair] = tmpmat[ipair * mloc];  // Store first element or use a more refined approach based on your needs
        }
    }
*/
    return apq.size();

} //jade_complex
} // namespace wannier
} // namespace inq

#endif

///////////////////////////////////////////////////////////////////
#ifdef INQ_AWANNIER_JADE_COMPLEX_UNIT_TEST
#undef INQ_AWANNIER_JADE_COMPLEX_UNIT_TEST

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

    using namespace inq;
    using namespace Catch::literals;
    using Catch::Approx;

    int maxsweep = 100;
    double tol = 1e-6;

    // Create a vector of 6 1x1 matrices (H2 test case) //coressponds to gs in a 20x20x20 cell
    std::vector<std::vector<std::vector<complex>>> a(6, std::vector<std::vector<complex>>(2, std::vector<complex>(2)));

    // Fill a mats
    a[0][0][0] = complex(-0.6843,-0.0000);
    a[0][0][1] = complex(-0.0010,-0.1081);
    a[0][1][0] = complex(-0.6810,0.0000);
    a[0][1][1] = complex(0.0000,0.0000);
    a[1][0][0] = complex(-0.0977,0.0000);
    a[1][0][1] = complex(0.0073,0.6826);
    a[1][1][0] = complex(0.0000,0.0000);
    a[1][1][1] = complex(0.0000,0.0000);
    a[2][0][0] = complex(-0.6843,-0.0000);
    a[2][0][1] = complex(-0.0010,-0.1081);
    a[2][1][0] = complex(0.0000,0.0000);
    a[2][1][1] = complex(0.0000,0.0000);
    a[3][0][0] = complex(-0.0977,0.0000);
    a[3][0][1] = complex(0.0073,0.6826);
    a[3][1][0] = complex(0.0000,0.0000);
    a[3][1][1] = complex(0.0000,0.0000);
    a[4][0][0] = complex(-0.6843,-0.0000);
    a[4][0][1] = complex(-0.0010,-0.1081);
    a[4][1][0] = complex(0.0000,0.0000);
    a[4][1][1] = complex(0.0000,0.0000);
    a[5][0][0] = complex(-0.0977,0.0000);
    a[5][0][1] = complex(0.0073,0.6826);
    a[5][1][0] = complex(0.0000,0.0000);
    a[5][1][1] = complex(0.0000,0.0000);

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
        CHECK(real(sweep) == 18_a );
}
#endif
