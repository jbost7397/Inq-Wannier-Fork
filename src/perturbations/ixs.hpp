/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PERTURBATIONS__IXS
#define INQ__PERTURBATIONS__IXS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <math/vector3.hpp>
#include <magnitude/energy.hpp>
#include <perturbations/none.hpp>

namespace inq {
namespace perturbations {

class ixs : public perturbations::none {

public:
	ixs(quantity<magnitude::energy> amplitude, vector3<int> wavenumber, quantity<magnitude::time> tdelay, quantity<magnitude::time> twidth, std::string shape="exp"):
		amplitude_(amplitude.in_atomic_units()),
        	wavenumber_(wavenumber),
		tdelay_(tdelay.in_atomic_units()),
		twidth_(twidth.in_atomic_units()),
        	shape_(shape)
	{}

	auto has_potential() const {
		return true;
	}

	auto envelope(const double time) const {
		return amplitude_/sqrt(2.0*M_PI)*exp(-0.5*pow((time - tdelay_)/twidth_, 2));
	}
	
	template<typename PotentialType>
	void potential(const double time, PotentialType & potential) const {

		auto qix = parallel::global_index(wavenumber_[0]);
		auto qiy = parallel::global_index(wavenumber_[1]);
		auto qiz = parallel::global_index(wavenumber_[2]);

        	// get the G vector corresponding to the q indices
		vector3<double, covariant> qcov = potential.basis().reciprocal().point_op().gvector(qix, qiy, qiz);

		if (shape_ == "cos") {
			gpu::run(potential.basis().local_sizes()[2], potential.basis().local_sizes()[1], potential.basis().local_sizes()[0],
				[point_op = potential.basis().point_op(), vk = begin(potential.cubic()), env = envelope(time), q = qcov] GPU_LAMBDA (auto iz, auto iy, auto ix) {
					auto rr = point_op.rvector(ix, iy, iz);
					vk[ix][iy][iz] += env*cos(dot(q, rr));
			});
		}

		else if (shape_ == "sin") {
			gpu::run(potential.basis().local_sizes()[2], potential.basis().local_sizes()[1], potential.basis().local_sizes()[0],
				[point_op = potential.basis().point_op(), vk = begin(potential.cubic()), env = envelope(time), q = qcov] GPU_LAMBDA (auto iz, auto iy, auto ix) {
					auto rr = point_op.rvector(ix, iy, iz);
					vk[ix][iy][iz] += env*sin(dot(q, rr));
			});
		}

         // complex potentials appear unsupported currently
/*        else if (shape_ == "exp") {
            gpu::run(potential.basis().local_sizes()[2], potential.basis().local_sizes()[1], potential.basis().local_sizes()[0],
                         [point_op = potential.basis().point_op(), vk = begin(potential.cubic()), env = envelope(time), q = qcov] GPU_LAMBDA (auto iz, auto iy, auto ix) {
                             auto rr = point_op.rvector(ix, iy, iz);
                             vk[ix][iy][iz] += env*complex(cos(dot(q, rr)), sin(dot(q, rr)));
                         });
        }
*/

		else {
			throw std::runtime_error("INQ error: Invalid IXS envelope type");
		}
	}

private:
	double amplitude_;
	vector3<int> wavenumber_;
	double tdelay_;
	double twidth_;
	std::string shape_;
};

}
}
#endif

#ifdef INQ_PERTURBATIONS_IXS_UNIT_TEST
#undef INQ_PERTURBATIONS_IXS_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

using namespace inq;
using namespace Catch::literals;
using namespace magnitude;


TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	auto amplitude = 1.0_Ha;
	auto tdelay = 0.3_fs;

	perturbations::ixs probe(amplitude, {0,0,1}, tdelay, 0.1_fs);

	CHECK(not probe.has_uniform_electric_field());
	CHECK(not probe.has_uniform_vector_potential());
	CHECK(probe.has_potential());
	CHECK(probe.envelope(tdelay.in_atomic_units()) == amplitude.in_atomic_units()/sqrt(2.0*M_PI));
}
#endif
