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

	utils::match energy_match(3.0e-5);

	auto a = 10.18_b;
	systems::ions ions(systems::cell::cubic(a));
	
	ions.insert_fractional("Si", {0.0,  0.0,  0.0 });
	ions.insert_fractional("Si", {0.25, 0.25, 0.25});
	ions.insert_fractional("Si", {0.5,  0.5,  0.0 });
	ions.insert_fractional("Si", {0.75, 0.75, 0.25});
	ions.insert_fractional("Si", {0.5,  0.0,  0.5 });
	ions.insert_fractional("Si", {0.75, 0.25, 0.75});
	ions.insert_fractional("Si", {0.0,  0.5,  0.5 });
	ions.insert_fractional("Si", {0.25, 0.75, 0.75});

	auto nk = 1;

	
	systems::electrons electrons(ions, input::kpoints::grid({nk, nk, nk}, false), options::electrons{}.cutoff(50.0_Ha));

	auto functional = options::theory{}.pbe();
	
	if(not electrons.try_load("silicon_restart")){
		ground_state::initial_guess(ions, electrons);
		ground_state::calculate(ions, electrons, options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-1_Ha));
		electrons.save("silicon_restart");
	}

//        auto output = [&](auto data){	
	auto const dt = 0.030000;
	long nsteps = 5000; //413.41373/dt;
	

	
	real_time::propagate<>(ions, electrons, [](auto){}, functional, options::real_time{}.num_steps(15).dt(0.01*1.0_atomictime).im_etrs());
	real_time::propagate<>(ions, electrons, [](auto){}, functional, options::real_time{}.num_steps(nsteps).dt(dt*1.0_atomictime).im_etrs().imetrs_thresh(1e-4_Ha,5).final_subspace_diag(true));
}
