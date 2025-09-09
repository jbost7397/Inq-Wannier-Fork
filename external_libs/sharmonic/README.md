# sharmonic

SHarmonic is a header only library C and C++ library that implements the spherical harmonics.
It is based on a direct implementation of the spherical harmonic formulas in Cartesian coordinates.
This makes it very efficient as the evaluations of the spherical harmonics is done mostly with multiplications and sums.
It implements both real and complex spherical harmonics up to order l=9 in both cartesian and angular (spherical) coordinates.

Using sharmonic
===============

sharmonic is a header only library.
To use it in your project you can simply download the `sharmonic.h` file and include it from your source code.
You can also download the whole project that includes a CMake build system.
The cmake build compiles several tests for the library that compare against reference values and other implementations (standard c++ and boost) if available.

Note that sharmonic is distribute under the MPL2 open source license.
This license allows to use sharmonic from (almost) any code independently of the license it uses.
Check https://mozilla.org/MPL/2.0/ for details.


C++ interface
=============

```cpp
#include <sharmonic.h>

double sharmonic::normalized_cartesian_real(int l, int m, double x, double y, double z);
template <typename Vector> double normalized_cartesian_real(int ll, int mm, Vector vec3);

double sharmonic::cartesian_real(int l, int m, double x, double y, double z);
template <typename Vector> double cartesian_real(int ll, int mm, Vector vec);

double sharmonic::angular_real(int l, int m, double theta, double phi);

template <typename Complex = std::complex<double>> Complex sharmonic::normalized_cartesian_complex(int l, int m, double x, double y, double z);
template <typename Complex = std::complex<double>, typename Vector> Complex sharmonic::normalized_cartesian_complex(int l, int m, Vector vec);

template <typename Complex = std::complex<double>> Complex sharmonic::cartesian_complex(int l, int m, double x, double y, double z);
template <typename Complex = std::complex<double>, typename Vector> Complex sharmonic::cartesian_complex(int l, int m, Vector vec);

template <typename Complex = std::complex<double>> Complex sharmonic::angular_complex(int l, int m, double theta, double phi);
```

The interface for sharmonic has several functions.
All sharmonic functions take arguments l and m, with l >= 0, m >= -l, and m <= l.
Values outside these conditions produce undefined results.

The functions differ in the spherical harmonic function they calculate (real or complex) and the coordinate arguments they take: normalized cartesian, cartesian, or angular.

The normalized cartesian functions take three coordinates with the constraint that x^2 + y^2 + z^2 must be equal to 1, if they do not the result is undefined.
This is the preferred interface in terms of numerical performance and should be used if normalized values are available.

The regular cartesian function also takes three coordinates but it calculates its normalization so it is more expensive numerically.
In the particular case of coordinates (0.0, 0.0, 0.0) sharmonic will return 0.0, except for l = 0. 

Lastly the angular version of the functions take the two angular components of the spherical coordinates: theta and phi.
We follow the convention where theta is the polar angle (theta = acos(z/r)) and phi is the azimuthal angle (phi = atan2(y, x)).
Both angles are expected in radians and do not need to be in any specific range.
Note that the angular functions internally convert to normalized cartesian coordinates by calling `sin` and `cos`.
So it is recommended to use this functions only if you already have the values in shperical coordinates.

For complex functions sharmonic can receive a template argument for a complex type, if this argument is not passed sharmonic returns an `std::complex<double>`.
The only requirement for the `Complex` type is that it must have a constructor that receives two doubles.
It is not necessary that the type defines complex arithmetic.

C interface
===========

```C
#include <sharmonic.h>

double sharmonic_normalized_cartesian_real(int l, int m, double x, double y, double z);
double sharmonic_cartesian_real(int l, int m, double x, double y, double z);
double sharmonic_angular_real(int l, int m, double theta, double phi);

double complex sharmonic_normalized_cartesian_complex(int l, int m, double x, double y, double z);
double complex sharmonic_cartesian_complex(int l, int m, double x, double y, double z);
double complex sharmonic_angular_complex(int l, int m, double theta, double phi);
```

The C interface for sharmonic is very similar to the C++ interface but simpler.
The first difference is that all functions start with `sharmonic_` (instead of the `sharmonic` namespace in C++).
In second place, the complex functions return a value of the C99 native type `double complex`.
Lastly, there are no vector argument versions of the functions.

CUDA and HIP support for GPUs
=============================

All the C++ functions are defined as `__host__ __device__` when compiling with the CUDA or HIP, so they can be called from CPU or GPU code.
However, note that for complex function the default type `std::complex<double>` normally cannot be used in device code.
We recommend to use the `thrust::complex<double>` type instead by calling sharmonic with a template argument like this:

```C++
auto zsh = cartesian_complex<thrust::complex<double>>(2, 0, 0.5, 1.5, 2.5)
```

The type `thrust::complex` is availabe in CUDA Thrust (NVidia) and rocThurst (AMD).

