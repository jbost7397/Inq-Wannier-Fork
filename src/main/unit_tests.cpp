/* -*- indent-tabs-mode: t -*- */

/*
 Copyright (C) 2019 Xavier Andrade

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

#include <catch2/catch.hpp>


#define PSEUDOPOD_UNIT_TEST

#include <pseudopod/erf_range_separation.hpp>
#include <pseudopod/element.hpp>
#include <pseudopod/pseudopotential.hpp>
#include <pseudopod/spherical_harmonic.hpp>
#include <pseudopod/math/bessel_transform.hpp>
#include <pseudopod/math/spherical_bessel.hpp>
#include <pseudopod/math/spline.hpp>

#define INQ_UNIT_TEST

#include <utils/partition.hpp>
#include <utils/match.hpp>

#include <gpu/run.hpp>

#include <input/basis.hpp>
#include <input/cell.hpp>
#include <input/species.hpp>
#include <input/scf.hpp>

#include <config/path.hpp>

#include <math/vec3d.hpp>
#include <math/vector3.hpp>

#include <ions/geometry.hpp>
#include <ions/unitcell.hpp>
#include <ions/interaction.hpp>
#include <ions/periodic_replicas.hpp>

#include <basis/grid.hpp>
#include <basis/spherical_grid.hpp>
#include <basis/real_space.hpp>
#include <basis/field.hpp>
#include <basis/field_set.hpp>

#include <states/ks_states.hpp>

#include <hamiltonian/atomic_potential.hpp>
#include <hamiltonian/ks_hamiltonian.hpp>
#include <hamiltonian/projector.hpp>
#include <hamiltonian/projector_fourier.hpp>

#include <operations/add.hpp>
#include <operations/sum.hpp>
#include <operations/overlap.hpp>
#include <operations/orthogonalize.hpp>
#include <operations/diagonalize.hpp>
#include <operations/gradient.hpp>
#include <operations/shift.hpp>
#include <operations/randomize.hpp>
#include <operations/matrix_operator.hpp>
#include <operations/transfer.hpp>
#include <operations/laplacian.hpp>
#include <operations/exponential.hpp>
#include <operations/io.hpp>
#include <operations/gradient.hpp>
#include <operations/divergence.hpp>

#include <density/calculate.hpp>
#include <density/normalize.hpp>

#include <perturbations/kick.hpp>

#include <observables/dipole.hpp>

#include <solvers/poisson.hpp>
#include <solvers/linear.hpp>
#include <solvers/least_squares.hpp>

#include <mixers/base.hpp>
#include <mixers/broyden.hpp>
#include <mixers/linear.hpp>
#include <mixers/pulay.hpp>

#include <eigensolvers/conjugate_gradient.hpp>
#include <eigensolvers/steepest_descent.hpp>

#include <ground_state/calculate.hpp>

#include <real_time/propagate.hpp>
