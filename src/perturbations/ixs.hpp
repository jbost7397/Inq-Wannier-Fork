/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PERTURBATIONS__IXS
#define INQ__PERTURBATIONS__IXS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa, Yifan Yao
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
	ixs(quantity<magnitude::energy> amplitude, vector3<int> q, quantity<magnitude::time> tdelay, quantity<magnitude::time> twidth, std::string envtype="exp"):
		amplitude_(amplitude.in_atomic_units()),
        	q_(q),
		tdelay_(tdelay.in_atomic_units()),
		twidth_(twidth.in_atomic_units()),
        	envtype_(envtype)
	{}

	auto has_potential() const {
		return true;
	}

    auto envelope(const double time) const {
        return amplitude_/sqrt(2.0*M_PI) * exp( -0.5*pow((time-tdelay_)/(twidth_),2) );
    }
	
	template<typename PotentialType>
	void potential(const double time, PotentialType & potential) const {

        // something like this may be needed in parallel
//        auto qix = parallel::global_index(qi[0]);
//        auto qiy = parallel::global_index(qi[1]);
//        auto qiz = parallel::global_index(qi[2]);

        // get the G vector corresponding to the q indices
        vector3<double, covariant> qcov = potential.basis().reciprocal().point_op().gvector(q_[0], q_[1], q_[2]);


        if (envtype_ == "cos") {
            gpu::run(potential.basis().local_sizes()[2], potential.basis().local_sizes()[1], potential.basis().local_sizes()[0],
                         [point_op = potential.basis().point_op(), vk = begin(potential.cubic()), env=envelope(time), q=qcov] GPU_LAMBDA (auto iz, auto iy, auto ix) {
                             auto rr = point_op.rvector(ix, iy, iz);
                             vk[ix][iy][iz] += env * cos(dot(q,rr));
                        });
        }

        else if (envtype_ == "sin") {
            gpu::run(potential.basis().local_sizes()[2], potential.basis().local_sizes()[1], potential.basis().local_sizes()[0],
                         [point_op = potential.basis().point_op(), vk = begin(potential.cubic()), env=envelope(time), q=qcov] GPU_LAMBDA (auto iz, auto iy, auto ix) {
                             auto rr = point_op.rvector(ix, iy, iz);
                             vk[ix][iy][iz] += env * sin(dot(q,rr));
                        });
        }

         // complex potentials appear unsupported currently
/*        else if (envtype_ == "exp") {
            gpu::run(potential.basis().local_sizes()[2], potential.basis().local_sizes()[1], potential.basis().local_sizes()[0],
                         [point_op = potential.basis().point_op(), vk = begin(potential.cubic()), env=envelope(time), q=qcov] GPU_LAMBDA (auto iz, auto iy, auto ix) {
                             auto rr = point_op.rvector(ix, iy, iz);
                             vk[ix][iy][iz] += env * complex(cos(dot(q,rr)), sin(dot(q,rr)));
                         });
        }
*/

        else {
            throw std::runtime_error("INQ error: Invalid IXS envelope type");
        }


	}

private:
	double amplitude_;
	vector3<int> q_;
	double tdelay_;
	double twidth_;
	std::string envtype_;
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
    perturbations::ixs nop(1.0_Ha, {0,0,1}, 0.3_fs, 0.1_fs);
    CHECK(not nop.has_uniform_electric_field());
}
#endif
