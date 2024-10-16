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
double jade_complex(T maxsweep, T1 tol, MatrixType1& a, MatrixType2& u, MatrixType3& adiag) {

    const double eps = std::numeric_limits<double>::epsilon();
    assert(tol > eps);

    int n = a[0].size();  // Assuming a is non-empty and rectangular
    int mloc = a.size();  // Number of rows in a (size of first dimension)

    // Initialize u as identity
    gpu::array<complex,2> u_tmp({mloc,mloc}, complex(0.0, 0.0));
    //u.resize(mloc, std::vector<complex>(mloc, complex(0.0, 0.0))); //CS can't resize, what is the inq equivalent 
    for (int i = 0; i < mloc; ++i) {
        u_tmp[i][i] = 1.0;
    }
    u = u_tmp;

    //adiag.resize(a.size(), std::vector<complex>(n));
    gpu::array<complex,2> adiag_tmp(a.size(), gpu::array<complex,1>(n));
    adiag = adiag_tmp;

    bool nloc_odd = (mloc % 2 != 0);
    //std::vector<std::vector<complex>> a_aux(a.size(), std::vector<complex>(mloc));
    //std::vector<complex> u_aux(mloc);
    gpu::array<complex,2> a_aux(a.size(), gpu::array<complex,1>(mloc));
    gpu::array<complex,1> u_aux(mloc);


    const int nploc = (mloc + 1) / 2;
    std::deque<int> top(nploc), bot(nploc);

    for (int i = 0; i < nploc; ++i) {
        top[i] = i;
        bot[nploc - i - 1] = nploc + i;
    }

    int nsweep = 0;
    bool done = false;
    double diag_change = 0.0;

    while (!done && nsweep < maxsweep) {
        ++nsweep;

        for (int irot = 0; irot < 2 * nploc - 1; ++irot) {
            gpu::array<complex,1> apq(a.size() * 3 * nploc, complex(0.0, 0.0));
            for (int k = 0; k < a.size(); ++k) {
                for (int ipair = 0; ipair < nploc; ++ipair) {
                    const int iapq = 3 * ipair + k * 3 * nploc;
                    apq[iapq] = complex{0.0, 0.0};
                    apq[iapq + 1] = complex{0.0, 0.0};
                    apq[iapq + 2] = complex{0.0, 0.0};

                    const complex* ap = (top[ipair] < mloc) ? a[k][top[ipair] * mloc] : &a_aux[k][0];
                    const complex* aq = (bot[ipair] < mloc) ? a[k] + bot[ipair] * mloc : &a_aux[k][0];
                    const complex* up = (top[ipair] < mloc) ? u[k] + top[ipair] * mloc : &u_aux[0];
                    const complex* uq = (bot[ipair] < mloc) ? u[k] + bot[ipair] * mloc : &u_aux[0];

                    for (int ii = 0; ii < mloc; ++ii) {
                      apq[iapq] += conj_cplx(ap[ii]) * uq[ii];
                      apq[iapq + 1] += conj_cplx(ap[ii]) * up[ii];
                      apq[iapq + 2] += conj_cplx(aq[ii]) * uq[ii];
                    }
                }
            };

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
                    }

                    int N = 3; // For Wannier 3x3 matrix size
                    std::vector<double> G = {g11, g12, g13, g21, g22, g23, g31, g32, g33};  // Matrix to be diagonalized
                    std::vector<double> Q(9); // Eigenvectors
                    std::vector<double> D(3); // Eigenvalues

                    // Implement the Jacobi diagonalization algorithm here (or call an external function)
                    jacobi_eigenvalue(N, G, Q, D); // You will need to implement this function or use an existing library

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
                        gpu::array<complex,1> ap = (top[ipair] < mloc) ? a[k].data() + top[ipair] * mloc : &a_aux[k][0];
                        gpu::array<complex,1> aq = (bot[ipair] < mloc) ? a[k].data() + bot[ipair] * mloc : &a_aux[k][0];

                        // Apply plane rotation
                        plane_rot(ap, aq, c, sconj);
                    }

                    gpu::array<complex,1> up = (top[ipair] < mloc) ? u.data() + top[ipair] * mloc : &u_aux[0];
                    gpu::array<complex,1> uq = (bot[ipair] < mloc) ? u.data() + bot[ipair] * mloc : &u_aux[0];

                    plane_rot(up, uq, c, sconj);

                    double diag_change_ipair = 0.0;
                    for (int k = 0; k < a.size(); ++k) {
                        const int iapq = 3 * ipair + k * 3 * nploc;
                        const complex aii = apq[iapq + 1];
                        const complex ajj = apq[iapq + 2];
                        const complex v1 = conj_cplx(c) * c - sconj * s;

                        double apq_new = real(v1 * (aii - ajj) + 2.0 * c * s * apq[iapq] + 2.0 * sconj * c * apq[iapq]);
                        diag_change_ipair += 2.0 * fabs(apq_new - real(aii - ajj));
                        diag_change += diag_change_ipair;
                    }
                }
            }

            // Rotate top and bot arrays
            if (nploc > 0) {
                std::swap(top[0], bot[0]);  // Simplified for this example; you can implement more logic if needed
            }
        }

        done = (fabs(diag_change) < tol) || (nsweep >= maxsweep);
    }

    // Reorder matrix columns and compute diagonal elements
    gpu::array<complex,1> tmpmat(n * mloc);
    for (int k = 0; k < a.size(); ++k) {
        for (int ipair = 0; ipair < nploc; ++ipair) {
            if (top[ipair] >= 0) {
                std::copy(a[k].begin() + top[ipair] * mloc, a[k].begin() + (top[ipair] + 1) * mloc, tmpmat.begin() + ipair * mloc);
            }
        }

        for (int ipair = 0; ipair < nploc; ++ipair) {
            if (bot[ipair] >= 0) {
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

    return diag_change;
}

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

    // Create a vector of 2 matrices (2x2 matrices)
    gpu::array<complex,3> a({2,2,2}); 

    // Fill first matrix a[0]
    a[0][0][0] = complex(1.0, 0.0);   // 1 + 0i
    a[0][0][1] = complex(0.5, 0.5);   // 0.5 + 0.5i
    a[0][1][0] = complex(0.5, -0.5);  // 0.5 - 0.5i
    a[0][1][1] = complex(1.0, 1.0);   // 1 + 1i

    // Fill second matrix a[1]
    a[1][0][0] = complex(2.0, 0.0);   // 2 + 0i
    a[1][0][1] = complex(1.0, 1.0);   // 1 + 1i
    a[1][1][0] = complex(1.0, -1.0);  // 1 - 1i
    a[1][1][1] = complex(2.0, 2.0);   // 2 + 2i

    // Create matrix u (initially identity)
    gpu::array<complex,2> u({2,2});
    u[0][0] = complex(1.0, 0.0);  // Identity element
    u[0][1] = complex(0.0, 0.0);
    u[1][0] = complex(0.0, 0.0);
    u[1][1] = complex(1.0, 0.0);  // Identity element

    // Prepare adiag to hold diagonal elements (size should match number of a matrices and their dimensions)
    gpu::array<complex,2> adiag({2,2}); // Assuming single diagonal element per input matrix

    // Call the jade_complex function
    double diag_change = wannier::jade_complex(maxsweep, tol, a, u, adiag);









}
#endif
