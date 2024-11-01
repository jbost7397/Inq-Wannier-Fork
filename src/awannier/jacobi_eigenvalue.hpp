/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__AWANNIER__JACOBI_EIGENVALUE
#define INQ__AWANNIER__JACOBI_EIGENVALUE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//  Licensing:
//    This code is distributed under the GNU LGPL license.
//  Author:
//    C++ version originally by John Burkardt

#include <inq_config.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <math/complex.hpp>

//CS updated by chatgpt, compiles and passes tests
//can determine later if performance is better w/ inq::matrix::diagonalize routine
 
namespace inq {
namespace wannier {

// Function to get the diagonal of a matrix
template <typename T, typename VectorType>
void r8mat_diag_get_vector(T nn, const VectorType& a, VectorType& v) {
    for (int i = 0; i < nn; ++i) {
        v[i] = a[i + i * nn];
    }
}

// Function to set a matrix to the identity matrix
template <typename T, typename VectorType>
void r8mat_identity(T nn, VectorType& a) {
    std::fill(a.begin(), a.end(), 0.0);
    for (int i = 0; i < nn; ++i) {
        a[i + i * nn] = 1.0;
    }
}

// Jacobi eigenvalue iteration
template <typename T, typename VectorType>
void jacobi_eigenvalue(T n, VectorType& a, VectorType& v, VectorType& d) {
    std::vector<double> bw(n, 0.0), zw(n, 0.0);

    r8mat_identity(n, v);
    r8mat_diag_get_vector(n, a, d);

    std::copy(d.begin(), d.end(), bw.begin());
    int it_num = 0;
    const int it_max = 10000;
    int rot_num = 0;

    while (it_num < it_max) {
        it_num++;
        double thresh = 0.0;

        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < j; ++i) {
                thresh += a[i + j * n] * a[i + j * n];
            }
        }

        thresh = sqroot(thresh) / (4 * n);
        if (thresh == 0.0) {
            break;
        }

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double gapq = 10.0 * fabs(a[p + q * n]);
                double termp = gapq + fabs(d[p]);
                double termq = gapq + fabs(d[q]);

                if (4 < it_num && termp == fabs(d[p]) && termq == fabs(d[q])) {
                    a[p + q * n] = 0.0;
                } else if (thresh <= fabs(a[p + q * n])) {
                    double h = d[q] - d[p];
                    double term = fabs(h) + gapq;

                    double t;
                    if (term == fabs(h)) {
                        t = a[p + q * n] / h;
                    } else {
                        double theta = 0.5 * h / a[p + q * n];
                        t = 1.0 / (fabs(theta) + sqroot(1.0 + theta * theta));
                        if (theta < 0.0) {
                            t = -t;
                        }
                    }

                    double c = 1.0 / sqroot(1.0 + t * t);
                    double s = t * c;
                    double tau = s / (1.0 + c);
                    h = t * a[p + q * n];
                    zw[p] -= h;
                    zw[q] += h;
                    d[p] -= h;
                    d[q] += h;
                    a[p + q * n] = 0.0;

                    for (int j = 0; j < p; ++j) {
                        double g = a[j + p * n];
                        double h = a[j + q * n];
                        a[j + p * n] = g - s * (h + g * tau);
                        a[j + q * n] = h + s * (g - h * tau);
                    }

                    for (int j = p + 1; j < q; ++j) {
                        double g = a[p + j * n];
                        double h = a[j + q * n];
                        a[p + j * n] = g - s * (h + g * tau);
                        a[j + q * n] = h + s * (g - h * tau);
                    }

                    for (int j = q + 1; j < n; ++j) {
                        double g = a[p + j * n];
                        double h = a[q + j * n];
                        a[p + j * n] = g - s * (h + g * tau);
                        a[q + j * n] = h + s * (g - h * tau);
                    }

                    for (int j = 0; j < n; ++j) {
                        double g = v[j + p * n];
                        double h = v[j + q * n];
                        v[j + p * n] = g - s * (h + g * tau);
                        v[j + q * n] = h + s * (g - h * tau);
                    }

                    rot_num++;
                }
            }
        }

        for (int i = 0; i < n; ++i) {
            bw[i] += zw[i];
            d[i] = bw[i];
            zw[i] = 0.0;
        }
    }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            a[i + j * n] = a[j + i * n];
        }
    }

    for (int k = 0; k < n - 1; ++k) {
        int m = k;
        for (int l = k + 1; l < n; ++l) {
            if (d[l] < d[m]) {
                m = l;
            }
        }

        if (m != k) {
            std::swap(d[m], d[k]);
            for (int i = 0; i < n; ++i) {
                std::swap(v[i + m * n], v[i + k * n]);
            }
        }
    }
}

}
}

#ifdef INQ_AWANNIER_JACOBI_EIGENVALUE_UNIT_TEST
#undef INQ_AWANNIER_JACOBI_EIGENVALUE_UNIT_TEST

#include <gpu/array.hpp>
#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
    using namespace inq;
    using namespace Catch::literals;

    //CS for wannier only need this for real 3x3 matrcies so only one test for each function

    int n = 3;
    gpu::array<double, 1> array{0.088958, 1.183407, 1.191946, 1.183407, 1.371884, 0.705297, 1.191946, 0.705297, 0.392459};
    gpu::array<double, 1> vector(3);

    wannier::r8mat_diag_get_vector(n, array, vector);

    CHECK(vector[0] == 0.088958_a);
    CHECK(vector[1] == 1.371884_a);
    CHECK(vector[2] == 0.392459_a);

    gpu::array<double, 1> mat(9);
    wannier::r8mat_identity(n, mat);

    CHECK(mat[0] == 1.0_a);
    CHECK(mat[1] == 0.0_a);
    CHECK(mat[2] == 0.0_a);
    CHECK(mat[3] == 0.0_a);
    CHECK(mat[4] == 1.0_a);
    CHECK(mat[5] == 0.0_a);
    CHECK(mat[6] == 0.0_a);
    CHECK(mat[7] == 0.0_a);
    CHECK(mat[8] == 1.0_a);

    gpu::array<double, 1> d(3);
    gpu::array<double, 1> v(9);

    wannier::jacobi_eigenvalue(n, array, v, d);

    CHECK(d[0] == -1.0626903983_a);
    CHECK(d[1] == 0.1733844724_a);
    CHECK(d[2] == 2.7426069258_a);
}
#endif

#endif

