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

#include <vector>
#include <deque>
#include <limits>
#include <cmath>
#include <cassert>

namespace inq {
namespace wannier {

//template <typename T, typename T1, class MatrixType1, class MatrixType2, class MatrixType3>      JB: proper function declaration consistent w/inq style
//int jade_complex(T maxsweep, T1 tol,
//                 MatrixType1& a,
//                 MatrixType2& u,
//                 MatrixType3& adiag)

int jade_complex(int maxsweep, double tol, std::vector<inq::matrix::distributed<inq::complex>*> a,
                 inq::matrix::distributed<inq::complex>& u, std::vector<std::vector<inq::complex>>& adiag) {     //JB: testing matrix::distributed for parallelization purposes
    const double eps = std::numeric_limits<double>::epsilon();
    assert(tol > eps);

    auto& comm = u.comm(); 
    int n = u.sizex();    
    int mloc = u.block().local_size();

    // JB: can also use r8mat_identity from jacobi_eigenvalue
    u.block() = gpu::array<inq::complex, 2>({mloc, mloc}, inq::complex(0.0, 0.0));
    gpu::run(mloc, [&](auto i) { u.block()[i][i] = 1.0; });

    adiag.resize(a.size());
    for (int k = 0; k < a.size(); ++k) {
        adiag[k].resize(n);
    }

    bool nloc_odd = (mloc % 2 != 0);
    std::vector<std::vector<inq::complex>> a_aux(a.size(), std::vector<inq::complex>(mloc));
    std::vector<inq::complex> u_aux(mloc);

    int nploc = (mloc + 1) / 2;
    std::deque<int> top(nploc), bot(nploc);
    for (int i = 0; i < nploc; ++i) {
        top[i] = i;
        bot[nploc - i - 1] = nploc + i;
    }

    auto gathered_a = std::vector<gpu::array<inq::complex, 2>>(a.size());
    for (size_t k = 0; k < a.size(); ++k) gathered_a[k] = gather(*a[k], 0);

    if (comm.root()) {
        int nsweep = 0;
        bool done = false;
        double diag_change = 0.0;

        while (!done && nsweep < maxsweep) {
            ++nsweep;
            
            for (int irot = 0; irot < 2 * nploc - 1; ++irot) {
                std::vector<inq::complex> apq(a.size() * 3 * nploc);  
                gpu::run(a.size(), [&](int k) {
                    for (int ipair = 0; ipair < nploc; ++ipair) {
                        const int iapq = 3 * ipair + k * 3 * nploc;
                        apq[iapq] = (0.0, 0.0);
                        apq[iapq + 1] = (0.0, 0.0);
                        apq[iapq + 2] = (0.0, 0.0);
                        
                        const inq::complex* ap = (top[ipair] < mloc) ? gathered_a[k].data() + top[ipair] * mloc : &a_aux[k][0];
                        const inq::complex* aq = (bot[ipair] < mloc) ? gathered_a[k].data() + bot[ipair] * mloc : &a_aux[k][0];
                        const inq::complex* up = (top[ipair] < mloc) ? u.block().data() + top[ipair] * mloc : &u_aux[0];
                        const inq::complex* uq = (bot[ipair] < mloc) ? u.block().data() + bot[ipair] * mloc : &u_aux[0];

                        for (int ii = 0; ii < mloc; ++ii) {
                            apq[iapq] += std::conj(ap[ii]) * uq[ii];
                            apq[iapq + 1] += std::conj(ap[ii]) * up[ii];
                            apq[iapq + 2] += std::conj(aq[ii]) * uq[ii];
                        }
                    }
                });

	        gpu::run(nploc, [&](int ipair) {
                    if (top[ipair] >= 0 && bot[ipair] >= 0) {
                        double g11 = 0.0, g12 = 0.0, g13 = 0.0;
                        double g21 = 0.0, g22 = 0.0, g23 = 0.0;
                        double g31 = 0.0, g32 = 0.0, g33 = 0.0;

                        for (int k = 0; k < a.size(); ++k) {
                            const int iapq = 3 * ipair + k * 3 * nploc;
                            const inq::complex aij = apq[iapq];
                            const inq::complex aii = apq[iapq + 1];
                            const inq::complex ajj = apq[iapq + 2];

                            const inq::complex h1 = aii - ajj;
                            const inq::complex h2 = aij + std::conj(aij);
                            const inq::complex h3 = inq::complex(0.0, 1.0) * (aij - std::conj(aij));

                            g11 += std::real(std::conj(h1) * h1);
                            g12 += std::real(std::conj(h1) * h2);
                            g13 += std::real(std::conj(h1) * h3);
                            g21 += std::real(std::conj(h2) * h1);
                            g22 += std::real(std::conj(h2) * h2);
                            g23 += std::real(std::conj(h2) * h3);
                            g31 += std::real(std::conj(h3) * h1);
                            g32 += std::real(std::conj(h3) * h2);
                            g33 += std::real(std::conj(h3) * h3);
                        }

                        gpu::array<double, 9> G({g11, g12, g13, g21, g22, g23, g31, g32, g33});
                        gpu::array<double, 9> Q(9);
                        gpu::array<double, 3> D(3);

                        jacobi_eigenvalue(3, G, Q, D);

                        double x = Q[6], y = Q[7], z = Q[8];
                        if (x < 0.0) {
                            x = -x; y = -y; z = -z;
                        }

                        double r = std::sqrt((x + 1) / 2.0);
                        inq::complex c = inq::complex(r, 0.0);
                        inq::complex s = inq::complex(y / (2.0 * r), -z / (2.0 * r));

                        for (int k = 0; k < a.size(); ++k) {
                            inq::complex* ap = (top[ipair] < mloc) ? gathered_a[k].data() + top[ipair] * mloc : &a_aux[k][0];
                            inq::complex* aq = (bot[ipair] < mloc) ? gathered_a[k].data() + bot[ipair] * mloc : &a_aux[k][0];

                            plane_rot(ap, aq, mloc, c, std::conj(s), comm);
                        }

                        inq::complex* up = (top[ipair] < mloc) ? u.block().data() + top[ipair] * mloc : &u_aux[0];
                        inq::complex* uq = (bot[ipair] < mloc) ? u.block().data() + bot[ipair] * mloc : &u_aux[0];

                        plane_rot(up, uq, mloc, c, std::conj(s), comm);

                        double diag_change_ipair = 0.0;
                        for (int k = 0; k < a.size(); ++k) {
                            const int iapq = 3 * ipair + k * 3 * nploc;
                            const inq::complex aii = apq[iapq+1];
                            const inq::complex ajj = apq[iapq+2];
                            const inq::complex v1 = std::conj(c) * c - std::conj(s) * s;

                            const double apq_new = std::real(v1 * (aii - ajj) + 2.0 * c * s * apq[iapq] + 2.0 * std::conj(s) * c * apq[iapq]);
                            diag_change_ipair += 2.0 * std::abs(apq_new - std::real(aii - ajj));

                            diag_change += diag_change_ipair;
                        }
                    }
                });

                // Rotate top and bot arrays
                if (nploc > 0) {
                    bot.push_back(top.back());
                    top.pop_back();
                    top.push_front(bot.front());
                    bot.pop_front();

                    // Ensure rotation is skipped on the first process column
                    if (comm.rank() == 0) {
                        if (nploc > 1) {
                            std::swap(top[0], top[1]);
                        } else {
                            std::swap(top[0], bot[0]);
                        }
                    }
                }
            } // end irot loop

            done = (std::abs(diag_change) < tol) || (nsweep >= maxsweep);
        }

        // Reorder matrix columns
        std::vector<inq::complex> tmpmat(nloc * mloc);
        for (int k = 0; k < a.size(); ++k) {
            for (int ipair = 0; ipair < nploc; ++ipair) {
                if (top[ipair] >= 0) {
                    std::copy(gathered_a[k].data() + top[ipair] * mloc,
                              gathered_a[k].data() + (top[ipair] + 1) * mloc, tmpmat.begin() + ipair * mloc);
                }
                if (bot[nploc - ipair - 1] >= 0) {
                    std::copy(gathered_a[k].data() + bot[nploc - ipair - 1] * mloc,
                              gathered_a[k].data() + (bot[nploc - ipair - 1] + 1) * mloc, tmpmat.begin() + (nploc + ipair) * mloc);
                }
            }
            std::copy(tmpmat.begin(), tmpmat.end(), gathered_a[k].data());
        }

        // Reorder u similarly
        for (int ipair = 0; ipair < nploc; ++ipair) {
            if (top[ipair] >= 0) {
                std::copy(u.block().data() + top[ipair] * mloc,
                          u.block().data() + (top[ipair] + 1) * mloc, tmpmat.begin() + ipair * mloc);
            }
            if (bot[nploc - ipair - 1] >= 0) {
                std::copy(u.block().data() + bot[nploc - ipair - 1] * mloc,
                          u.block().data() + (bot[nploc - ipair - 1] + 1) * mloc, tmpmat.begin() + (nploc + ipair) * mloc);
            }
        }
        std::copy(tmpmat.begin(), tmpmat.end(), u.block().data());

        // Compute diagonal elements and scatter back
	for (int k = 0; k < a.size(); ++k) {
	    std::fill(adiag[k].begin(), adiag[k].end(), inq::complex(0.0, 0.0));
	    auto& a_block = a[k]->block();
	    auto& u_block = u.block();

	    // Iterate over local block rows
	    gpu::run(mloc, [&](int i) {
    	    for (int j = 0; j < n; ++j) {  // Iterate over columns of the global matrix
    	        if (a_block.partx().contains(j)) {
        	        int local_j = a_block.partx().global_to_local(j).value();
                	adiag[k][j] += std::conj(a_block[i][local_j]) * u_block[i][local_j];
            		}
        	}
    	});

            // Apply MPI reduction for final diagonal values
            //int len = 2 * a[k]->n();
            //MPI_Allreduce(MPI_IN_PLACE, &adiag[k][0], len, MPI_DOUBLE, MPI_SUM, comm.get());
	    comm.all_reduce(adiag[k], std::plus<>(), comm.get());
        }
    }

    // Scatter results to all processes
    for (size_t k = 0; k < a.size(); ++k) {
        scatter(gathered_a[k], *a[k], 0);
    }
    
    return nsweep;
}

} // namespace wannier
} // namespace inq

///////////////////////////////////////////////////////////////////
#ifdef INQ_WANNIER_JADE_COMPLEX_UNIT_TEST
#undef INQ_WANNIER_JADE_COMPLEX_UNIT_TEST

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

    using namespace inq;
    using namespace Catch::literals;
    using Catch::Approx;
    using complex_t = inq::complex;

    parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

    int maxsweep = 100;
    double tol = 1e-6;
    int matrix_size = 3;

    matrix::distributed<complex_t> a1(comm, matrix_size, matrix_size);
    matrix::distributed<complex_t> a2(comm, matrix_size, matrix_size);
    matrix::distributed<complex_t> u(comm, matrix_size, matrix_size);

    if (comm.rank() == 0) {
        a1.block() = {{1.0, 0.5, 0.0}, {0.5, 1.0, 0.5}, {0.0, 0.5, 1.0}};
        a2.block() = {{1.0, -0.5, 0.0}, {-0.5, 1.0, -0.5}, {0.0, -0.5, 1.0}};
        u.block() = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    }

    std::vector<matrix::distributed<complex_t>*> input_matrices = {&a1, &a2};
    std::vector<std::vector<complex_t>> adiag(input_matrices.size(), std::vector<complex_t>(matrix_size, 0.0));

    int sweeps = inq::wannier::jade_complex(maxsweep, tol, input_matrices, u, adiag);

    CHECK(sweeps <= maxsweep);

    if (comm.rank() == 0) {
        for (size_t i = 0; i < input_matrices.size(); ++i) {
            for (size_t j = 0; j < matrix_size; ++j) {
                CHECK(std::abs(adiag[i][j]) <= 1.0_a);
            }
        }
    }
}
#endif 

