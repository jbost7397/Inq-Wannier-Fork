/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq/inq.hpp>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;

	auto & env = input::environment::global();
	auto ions = systems::ions::parse(inq::config::path::unit_tests_data() + "al.poscar");
	systems::electrons electrons(env.par(), ions, options::electrons{}.cutoff(500.0_eV).temperature(1.0_eV).extra_states(14));
	auto functional = options::theory{}.lda();

	// check wavenumber indexing
        vector3<int> qi = {1,0,0};
        auto qix = parallel::global_index(qi[0]);
        auto qiy = parallel::global_index(qi[1]);
        auto qiz = parallel::global_index(qi[2]);
        vector3<double, covariant> qcov = electrons.density_basis().reciprocal().point_op().gvector(qix, qiy, qiz);
        assert(ions.cell().metric().to_cartesian(qcov) == 2.0*M_PI/ions.cell()[0]);

	ground_state::initial_guess(ions, electrons);
        ground_state::calculate(ions, electrons, functional, inq::options::ground_state{}.energy_tolerance(1e-8_Ha));


	auto td = 0.01_fs;
        auto tw = 0.002_fs;
        auto probe = perturbations::ixs{0.001_eV * (1.0_fs/tw), qi, td, tw, "sin"};

	auto const dt = 0.001_fs;
	long nsteps = 10;
	utils::match match(1.0e-10);

	gpu::array<complex, 1> nq(nsteps);
	gpu::array<double, 1> envt(nsteps);	

	auto output = [&](auto data){
		auto iter = data.iter();
		nq[iter] = data.density_q(qi);
		envt[iter] = data.ixs_envelope();
	};
	
	real_time::propagate<>(ions, electrons, output, functional, options::real_time{}.num_steps(nsteps).dt(dt).etrs(), probe);

	match.check("response step 1",  nq[1] - nq[0],   complex(4.30840696e-10, 3.39704043e-09));
	match.check("response step 2",  nq[2] - nq[0],   complex(1.92667615e-09, 1.97149934e-08));
	match.check("response step 3",  nq[3] - nq[0],   complex(4.89638285e-09, 9.55781739e-08));
	match.check("response step 4",  nq[4] - nq[0],   complex(1.16475252e-08, 4.89930341e-07));
	match.check("response step 5",  nq[5] - nq[0],   complex(3.22301243e-08, 2.32612950e-06));
	match.check("response step 6",  nq[6] - nq[0],   complex(1.00525924e-07, 9.38872018e-06));
	match.check("response step 7",  nq[7] - nq[0],   complex(3.04273250e-07, 3.14804131e-05));
	match.check("response step 8",  nq[8] - nq[0],   complex(8.14973105e-07, 8.77786273e-05));
	match.check("response step 9",  nq[9] - nq[0],   complex(1.87665709e-06, 2.05622074e-04));

	return match.fail();
	
}
