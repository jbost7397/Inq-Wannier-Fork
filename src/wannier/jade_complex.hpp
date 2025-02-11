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
#include <matrix/diagonalize.hpp>
#include <parallel/communicator.hpp>
#include <operations/rotate.hpp>
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
template <typename T, typename T1, class MatrixType1, class MatrixType2, class MatrixType3>
void jade_complex(T maxsweep, T1 tol, MatrixType1& a, MatrixType2& u, MatrixType3& adiag) {

    assert(tol > std::numeric_limits<double>::epsilon());

    int n = std::get<0>(sizes(a)); // 6 for all wannier
    int nloc = std::get<1>(sizes(a)); 
    int mloc = std::get<2>(sizes(a)); //always equals nloc  

    // Initialize u as identity
    u.reextent({mloc, mloc});
    gpu::run(mloc, mloc, [mloc, u_int=begin(u)] GPU_LAMBDA (auto ii, auto jj) {
      u_int[ii][jj] = (ii == jj) ? complex(1.0,0.0) : complex(0.0,0.0);
    });
    gpu::sync();

    const int nploc = (nloc + 1) / 2;
    gpu::array<int,1> top(nploc);
    gpu::array<int,1> bot(nploc);
    int np = nploc; 

    // initialize top and bot arrays
    // the pair i is (top[i],bot[i])
    // top[i] is the local index of the top column of pair i
    // bot[i] is the local index of the bottom column of pair i
    gpu::run(nploc, [nploc, bot_int=begin(bot), top_int=begin(top)] GPU_LAMBDA (auto i) {
      top_int[i] = i; 
      bot_int[nploc - i - 1] = nploc + i;
    });
    gpu::sync();

    int nsweep = 0;
    bool done = false;
    // allocate matrix element packed array apq
    // apq[3*ipair   + k*3*nploc] = apq[k][ipair]
    // apq[3*ipair+1 + k*3*nploc] = app[k][ipair]
    // apq[3*ipair+2 + k*3*nploc] = aqq[k][ipair]
    gpu::array<complex,1> apq(n * 3 * nploc);

    CALI_CXX_MARK_SCOPE("jade");
    {

      while (!done) {
        ++nsweep;
        double diag_change = 0.0;
        // sweep local pairs and rotate 2*np -1 times
        for (int irot = 0; irot < 2*np-1; ++irot) {

	  //CS initalize rot_array within loop so it resets to identity every time 
          gpu::array<complex,2> rot_array({mloc, mloc}, complex(0.0, 0.0));
          gpu::run(mloc, mloc, [mloc, r=begin(rot_array)] GPU_LAMBDA (auto ii, auto jj) {
            r[ii][jj] = (ii == jj) ? complex(1.0,0.0) : complex(0.0,0.0);
          });
          gpu::sync();

	  //CS store original sum of diag elements of a	  
	  gpu::array<double, 1> diag_sum_init(1, 0.0);
	  gpu::run(n, mloc, [a_int=begin(a), diag_sum=begin(diag_sum_init)] GPU_LAMBDA (auto k, auto i) {
	    gpu::atomic::add(&diag_sum[0], real(a_int[k][i][i])); 
	  });
          gpu::sync();

            //jacobi rotations for local pairs of diagonal elements for all pairs (apq)
          {     CALI_CXX_MARK_SCOPE("gpu_run_loop1");
	    //CS profile shows this is pretty fast (and correct)
	    gpu::run(n, nploc, mloc, [n, nploc, mloc, a_int=begin(a), u_int=begin(u), apq_int=begin(apq), top_int=begin(top), bot_int=begin(bot)] GPU_LAMBDA(auto k, auto ipair, auto ii) { 
	      const int iapq = 3 * ipair + k * 3 * nploc;
              if (ii == 0) {
                apq_int[iapq] = complex(0.0, 0.0);
                apq_int[iapq + 1] = complex(0.0, 0.0);
                apq_int[iapq + 2] = complex(0.0, 0.0);
	      }
	      if (top_int[ipair] < mloc && bot_int[ipair] < mloc ){
                gpu::atomic::add(&apq_int[iapq], conj_cplx(a_int[k][top_int[ipair]][ii]) * u_int[bot_int[ipair]][ii]);
                gpu::atomic::add(&apq_int[iapq + 1], conj_cplx(a_int[k][top_int[ipair]][ii]) * u_int[top_int[ipair]][ii]);
                gpu::atomic::add(&apq_int[iapq + 2], conj_cplx(a_int[k][bot_int[ipair]][ii]) * u_int[bot_int[ipair]][ii]);
	      }
            });
            gpu::sync();
          }

          { CALI_CXX_MARK_SCOPE("gpu_run_loop2");

	      //CS loop over nploc and for all pairs construct G to be diagonalized
	      gpu::run(nploc, [mloc, nploc, n, apq_int=begin(apq), bot_int=begin(bot), top_int=begin(top), rot_array_int=begin(rot_array)] GPU_LAMBDA (auto ipair) {
	      for (int ipair = 0; ipair < nploc; ++ipair) {
                if (top_int[ipair] < mloc && bot_int[ipair] < mloc) {
		  double G[9] = {0.0};
		  for (int k = 0; k < n; ++k) {
		    const int iapq = 3 * ipair + k * 3 * nploc;
                    const complex aij = apq_int[iapq];
                    const complex aii = apq_int[iapq + 1];
                    const complex ajj = apq_int[iapq + 2];

                    const complex h1 = aii - ajj;
                    const complex h2 = aij + conj_cplx(aij);
                    const complex h3 = complex(0.0, 1.0) * (aij - conj_cplx(aij));
                    G[0] += real(conj_cplx(h1) * h1);
                    G[1] += real(conj_cplx(h1) * h2);
                    G[2] += real(conj_cplx(h1) * h3);
                    G[3] += real(conj_cplx(h2) * h1);
                    G[4] += real(conj_cplx(h2) * h2);
                    G[5] += real(conj_cplx(h2) * h3);
                    G[6] += real(conj_cplx(h3) * h1);
                    G[7] += real(conj_cplx(h3) * h2);
                    G[8] += real(conj_cplx(h3) * h3);
                  }

		   //CS Jacobi within loop, initalize eigenvector array
                   double v[9] = {0.0};
                   double bw[3] = {0.0};
                   double zw[3] = {0.0};
                   double d[3] = {0.0};

 		   d[0] = G[0]; d[1] = G[4]; d[2] = G[8];
	           v[0] = 1.0; v[4] = 1.0; v[8] = 1.0; 
	           bw[0] = G[0]; bw[1] = G[4]; bw[2] = G[8];

	           int it_num = 0;
		   const int it_max = 10000;
	           int rot_num = 0; 

    		   while (it_num < it_max) {
                   it_num = it_num + 1;

                   double thresh = 0.0;
    		   for (int j = 0; j < 3; j++ ) {
	             for (int i = 0; i < j; i++ ) {
	               thresh = thresh + G[i+j*3] * G[i+j*3];
      		     }
    		   }

    		   thresh = sqroot(thresh)/(12);
	           if (thresh == 0.0) {
                     break;
                   }

	           for (int p = 0; p < 3; ++p) {
                    for (int q = p + 1; q < 3; ++q) {
                      double gapq = 10.0 * fabs(G[p + q * 3]);
                      double termp = gapq + fabs(d[p]);
                      double termq = gapq + fabs(d[q]);
                      if (4 < it_num && termp == fabs(d[p]) && termq == fabs(d[q])) {
                        G[p + q * 3] = 0.0;
                      } else if (thresh <= fabs(G[p + q * 3])) {
                        double h = d[q] - d[p];
                        double term = fabs(h) + gapq;

                        double t;
                        if (term == fabs(h)) {
                          t = G[p + q * 3] / h;
                        } else {
                          double theta = 0.5 * h / G[p + q * 3];
                          t = 1.0 / (fabs(theta) + sqroot(1.0 + theta * theta));
                          if (theta < 0.0) {
                            t = -t;
                          }
                        }

                       double c = 1.0 / sqroot(1.0 + t * t);
                       double s = t * c;
                       double tau = s / (1.0 + c);
                       h = t * G[p + q * 3];
                       zw[p] -= h;
                       zw[q] += h;
                       d[p] -= h;
                       d[q] += h;
                       G[p + q * 3] = 0.0;

                       for (int j = 0; j < p; ++j) {
                         double g = G[j + p * 3];
                         double h = G[j + q * 3];
                         G[j + p * 3] = g - s * (h + g * tau);
                         G[j + q * 3] = h + s * (g - h * tau);
                       }

                       for (int j = p + 1; j < q; ++j) {
                         double g = G[p + j * 3];
                         double h = G[j + q * 3];
                         G[p + j * 3] = g - s * (h + g * tau);
                         G[j + q * 3] = h + s * (g - h * tau);
                       }

                       for (int j = q + 1; j < 3; ++j) {
                         double g = G[p + j * 3];
                         double h = G[q + j * 3];
                         G[p + j * 3] = g - s * (h + g * tau);
                         G[q + j * 3] = h + s * (g - h * tau);
                       }

                       for (int j = 0; j < 3; ++j) {
                         double g = v[j + p * 3];
                         double h = v[j + q * 3];
                         v[j + p * 3] = g - s * (h + g * tau);
                         v[j + q * 3] = h + s * (g - h * tau);
                       }

                      rot_num++;
                      }
                    }
                  }
                  for (int i = 0; i < 3; ++i) {
                    bw[i] += zw[i];
                    d[i] = bw[i];
                    zw[i] = 0.0;
                  }
                }

                for (int j = 0; j < 3; ++j) {
                  for (int i = 0; i < j; ++i) {
                    G[i + j * 3] = G[j + i * 3];
                  }
                }

    	       for (int k = 0; k < 2; ++k) {
               int m = k;
                 for (int l = k + 1; l < 3; ++l) {
                   if (d[l] < d[m]) {
                     m = l;
                   }
                 }
                 if (m != k) {
		   auto tmp = d[m];
		   d[m] = d[k];
		   d[k] = tmp; 
                   for (int i = 0; i < 3; ++i) {
		     auto tmp = v[i + m * 3];
		     v[i + m * 3] = v[i + k * 3];
		     v[i + k * 3] = tmp; 
                   }
                 }
               }

	       //CS v now contains the eigenvectors 
	       double x = v[6], y = v[7], z = v[8];
	       if (v[6] < 0.0) {
                 x = -x; y = -y; z = z;
               }

               double one = 1.0;
               double r = sqroot((x + one) / 2.0);
               complex c = complex(r, 0.0);
               complex s = complex(y / (2.0 * r), -z / (2.0 * r));
               complex sconj = conj_cplx(s);
	       //CS construct rotations as array and apply all at once 
               rot_array_int[top_int[ipair]][top_int[ipair]] = c;
               rot_array_int[bot_int[ipair]][bot_int[ipair]] = c;
	       rot_array_int[top_int[ipair]][bot_int[ipair]] = sconj;
	       rot_array_int[bot_int[ipair]][top_int[ipair]] = -s;
       	     } //if 
           } //ipair 
        }); //loop
        gpu::sync();
        } //timer

          {     CALI_CXX_MARK_SCOPE("gpu_run_loop3");
	       //CS apply rotation 
               namespace blas = boost::multi::blas;

               u = +blas::gemm(1.0, rot_array, u);
               gpu::sync();

	       for (int k = 0; k < n; ++k) {
	         a[k] = +blas::gemm(1.0, rot_array, a[k]);
	       }
	       gpu::sync();

	       //CS get resulting diag sum and find change 
               gpu::array<double, 1> diag_sum_end(1, 0.0);
               gpu::run(n, mloc, [a_int=begin(a), diag_sum=begin(diag_sum_end)] GPU_LAMBDA (auto k, auto i) {
                 gpu::atomic::add(&diag_sum[0], real(a_int[k][i][i]));
               });
               gpu::sync();

	       diag_change += 2.0 * fabs(diag_sum_end[0] - diag_sum_init[0]);

	    }

            // Rotate top and bot arrays //CS ~85% speed up now 
            if (nploc > 0) {
		int top_back = top[nploc - 1];
		int bot_front = bot[0];
		gpu::run(nploc-1, [bot_int=begin(bot), top_int=begin(top)] GPU_LAMBDA (auto j) { 
	          bot_int[j] = bot_int[j+1];
		  top_int[j + 1] = top_int[j];
                });
	        gpu::run(1, [nploc, top_back, bot_front, top_int=begin(top), bot_int=begin(bot)] GPU_LAMBDA (auto i) {
		    bot_int[nploc - 1] = top_back;
		    top_int[0] = bot_front;
            	    if (nploc > 1) {
		        int tmp = top_int[0];
			top_int[0] = top_int[1];
			top_int[1] = tmp;
	            } else {
			  int tmp = top_int[0];
			  top_int[0] = bot_int[0];
			  bot_int[0] = tmp;
		    }
		});
		gpu::sync();
	    } //if nploc >0 
	} //irot
       std::cout << "nsweep:  " <<  nsweep << " diag change:  " << diag_change << std::endl;
       done = (fabs(diag_change) < tol) || (nsweep >= maxsweep);
      } //while 
    } //scope

    //eigenvalue array
    adiag.reextent({n, mloc}); 
    gpu::run(n, mloc, [adiag_int=begin(adiag)] GPU_LAMBDA (auto i, auto k) {
      adiag_int[i][k] = complex(0.0, 0.0);
    });
    gpu::sync();

    //Compute diagonal elements
    gpu::run(n, mloc, nloc, [n, mloc, nloc, a_int=begin(a), u_int=begin(u), adiag_int=begin(adiag)] GPU_LAMBDA (auto kk, auto ii, auto jj) {
      gpu::atomic::add(&adiag_int[kk][ii], conj_cplx(a_int[kk][ii][jj]) * u_int[ii][jj]);
    });
    gpu::sync();

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

    SECTION("Even number of Centers"){

      int maxsweep = 100;
      double tol = 1e-6;
      int six = 6;

      // Create a vector of 6 2x2 matrices (2He, 2 center test case) //coressponds to gs in a 20x20x20 cell
      gpu::array<complex,3> a({six,2,2});

      // Fill a matricies 
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
      gpu::array<complex,2> u({2,2});

      // Prepare adiag to hold diagonal elements (size should match number of a matrices and their dimensions)
      gpu::array<complex,2> adiag({six,2});

      // Call the jade_complex function
      wannier::jade_complex(maxsweep, tol, a, u, adiag);


	  //Inital checks of matrices 
    	  CHECK(u.size() == 2);
    	  CHECK(adiag.size() == 6);
    	  CHECK(adiag[0].size() == 2);

          //Check the diagonal elements of the amats upon return
          CHECK(real(adiag[0][0]) == -0.79081263_a);
          CHECK(real(adiag[0][1]) == -0.57456037_a);
          CHECK(real(adiag[1][0]) == 0.57455456_a);
          CHECK(real(adiag[1][1]) == -0.79080839_a);
          CHECK(real(adiag[2][0]) == -0.79081263_a);
          CHECK(real(adiag[2][1]) == -0.57456037_a);
          CHECK(real(adiag[3][0]) == 0.57455456_a);
          CHECK(real(adiag[3][1]) == -0.79080840_a);
          CHECK(real(adiag[4][0]) == -0.79081266_a);
          CHECK(real(adiag[4][1]) == -0.57456043_a);
          CHECK(real(adiag[5][0]) == 0.57455453_a);
          CHECK(real(adiag[5][1]) == -0.79080836_a); 
          //imag values are correct, but so close to zero they fail in make check 
	  //also only real values needed for wannier

          //Check the transform matrix u that is returned
          //only values that are greater than 1e-7
          CHECK(real(u[0][0]) == 0.71250999_a);
          CHECK(real(u[0][1]) == 0.00745655_a);
          CHECK(imag(u[0][1]) == 0.70162234_a);
          CHECK(real(u[1][0]) == -0.00745655_a);
          CHECK(imag(u[1][0]) == 0.70162234_a);
          CHECK(real(u[1][1]) == 0.71250999_a); 
        }//even 

    SECTION("Odd number of Centers"){

      int maxsweep = 100;
      double tol = 1e-8;
      int six = 6;

      // Create a vector of 6 3x3 matrices (3He, 3 centers test case) coresponds to gs in a 20x20x20 cell
      gpu::array<complex,3> a({six,3,3});

      // Fill the 6 a matrices 
      a[0][0][0] = complex(0.52756279,0.00000025);
      a[0][0][1] = complex(0.61712854,0.03799883);
      a[0][0][2] = complex(0.41610786,0.00450980);
      a[0][1][0] = complex(0.61712858,-0.03799882);
      a[0][1][1] = complex(-0.39874645,-0.00000004);
      a[0][1][2] = complex(0.18879589,0.11069471);
      a[0][2][0] = complex(0.41610770,-0.00451033);
      a[0][2][1] = complex(0.18879578,-0.11069498);
      a[0][2][2] = complex(-0.51669249,-0.00000021);
      a[1][0][0] = complex(0.00043530,-0.00000008);
      a[1][0][1] = complex(-0.03299494,-0.19531448);
      a[1][0][2] = complex(0.01604519,0.28682604);
      a[1][1][0] = complex(-0.03299519,0.19531473);
      a[1][1][1] = complex(0.02496930,0.00000002);
      a[1][1][2] = complex(0.12669977,-0.55715202);
      a[1][2][0] = complex(0.01604516,-0.28682602);
      a[1][2][1] = complex(0.12669996,0.55715167);
      a[1][2][2] = complex(-0.24165857,0.00000006);
      a[2][0][0] = complex(0.52756279,0.00000025);
      a[2][0][1] = complex(0.61712854,0.03799883);
      a[2][0][2] = complex(0.41610786,0.00450980);
      a[2][1][0] = complex(0.61712858,-0.03799882);
      a[2][1][1] = complex(-0.39874645,-0.00000004);
      a[2][1][2] = complex(0.18879589,0.11069471);
      a[2][2][0] = complex(0.41610770,-0.00451033);
      a[2][2][1] = complex(0.18879578,-0.11069498);
      a[2][2][2] = complex(-0.51669249,-0.00000021);
      a[3][0][0] = complex(0.00043529,-0.00000008);
      a[3][0][1] = complex(-0.03299494,-0.19531448);
      a[3][0][2] = complex(0.01604519,0.28682604);
      a[3][1][0] = complex(-0.03299519,0.19531473);
      a[3][1][1] = complex(0.02496930,0.00000002);
      a[3][1][2] = complex(0.12669977,-0.55715202);
      a[3][2][0] = complex(0.01604516,-0.28682602);
      a[3][2][1] = complex(0.12669996,0.55715167);
      a[3][2][2] = complex(-0.24165857,0.00000006);
      a[4][0][0] = complex(0.52756291,0.00000022);
      a[4][0][1] = complex(0.61712859,0.03799886);
      a[4][0][2] = complex(0.41610782,0.00450994);
      a[4][1][0] = complex(0.61712846,-0.03799893);
      a[4][1][1] = complex(-0.39874645,-0.00000012);
      a[4][1][2] = complex(0.18879609,0.11069467);
      a[4][2][0] = complex(0.41610788,-0.00451026);
      a[4][2][1] = complex(0.18879585,-0.11069491);
      a[4][2][2] = complex(-0.51669268,-0.00000006);
      a[5][0][0] = complex(0.00043542,0.00000007);
      a[5][0][1] = complex(-0.03299496,-0.19531438);
      a[5][0][2] = complex(0.01604498,0.28682606);
      a[5][1][0] = complex(-0.03299513,0.19531470);
      a[5][1][1] = complex(0.02496932,0.00000003);
      a[5][1][2] = complex(0.12669977,-0.55715195);
      a[5][2][0] = complex(0.01604504,-0.28682576);
      a[5][2][1] = complex(0.12669985,0.55715171);
      a[5][2][2] = complex(-0.24165874,-0.00000019);

      // Create matrix u (initially identity)
      gpu::array<complex,2> u({3,3});

      // Prepare adiag to hold diagonal elements (size should match number of a matrices and their dimensions)
      gpu::array<complex,2> adiag({six,3});

      // Call the jade_complex function
      wannier::jade_complex(maxsweep, tol, a, u, adiag);

	  //Check that data is initalized correctly 
          CHECK(u.size() == 3);
          CHECK(adiag.size() == 6);
          CHECK(adiag[0].size() == 3);

	  //Check the transform matrix u that is returned 
          CHECK(real(u[0][0]) == 0.85542060_a);
          CHECK(real(u[0][1]) == 0.42645081_a);
          CHECK(real(u[0][2]) == 0.29206758_a);
          CHECK(real(u[1][0]) == -0.28503313_a);
          CHECK(real(u[1][1]) == 0.54964807_a);
          CHECK(real(u[1][2]) == -0.05412787_a);
          CHECK(real(u[2][0]) == -0.18919209_a);
          CHECK(real(u[2][1]) == -0.03741007_a);
          CHECK(real(u[2][2]) == 0.59031761_a);
	  CHECK(imag(u[0][0]) == -0.00819384_a);
          CHECK(imag(u[0][1]) == 0.00084339_a);
          CHECK(imag(u[0][2]) == 0.03199962_a);
          CHECK(imag(u[1][0]) == -0.17689330_a);
          CHECK(imag(u[1][1]) == -0.15451794_a);
          CHECK(imag(u[1][2]) == 0.74735952_a);
          CHECK(imag(u[2][0]) == -0.34620758_a);
          CHECK(imag(u[2][1]) == 0.70053600_a);
          CHECK(imag(u[2][2]) == 0.06100488_a);

          //Check the diagonal elements of the amats upon return 
          CHECK(real(adiag[0][0]) == 0.97749613_a);
          CHECK(real(adiag[0][1]) == -0.57455995_a);
          CHECK(real(adiag[0][2]) == -0.79081233_a);
          CHECK(real(adiag[1][1]) == -0.79080838_a);
          CHECK(real(adiag[1][2]) == 0.57455451_a);
          CHECK(real(adiag[2][0]) == 0.97749613_a);
          CHECK(real(adiag[2][1]) == -0.57455995_a);
          CHECK(real(adiag[2][2]) == -0.79081233_a);
          CHECK(real(adiag[3][1]) == -0.79080838_a);
          CHECK(real(adiag[3][2]) == 0.57455451_a);
          CHECK(real(adiag[4][0]) == 0.97749625_a);
          CHECK(real(adiag[4][1]) == -0.57456014_a);
          CHECK(real(adiag[4][2]) == -0.79081232_a);
          CHECK(real(adiag[5][1]) == -0.79080831_a); 
	  CHECK(real(adiag[5][2]) == 0.57455443_a); 
  }//odd 
}
#endif
