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
        bool groundstate_only = false;

	auto a = 10.18_b;
	systems::ions ions(systems::cell::orthorhombic(2*a, a, a));
	
	ions.insert_fractional("Si", {0.0,  0.0,  0.0 });
	ions.insert_fractional("Si", {0.5,  0.5,  0.5 });
	ions.insert_fractional("Si", {0.125, 0.125, 0.125});
	ions.insert_fractional("Si", {0.25, 0.25, 0.25});
	ions.insert_fractional("Si", {0.25,  0.25,  0.0 });
	ions.insert_fractional("Si", {0.5,  0.5,  0.0 });
	ions.insert_fractional("Si", {0.375, 0.375, 0.125});
	ions.insert_fractional("Si", {0.75, 0.75, 0.25});
	ions.insert_fractional("Si", {0.25,  0.0,  0.25 });
	ions.insert_fractional("Si", {0.5,  0.0,  0.5 });
	ions.insert_fractional("Si", {0.375, 0.125, 0.375});
	ions.insert_fractional("Si", {0.75, 0.25, 0.75});
	ions.insert_fractional("Si", {0.0,  0.25,  0.25 });
	ions.insert_fractional("Si", {0.0,  0.5,  0.5 });
	ions.insert_fractional("Si", {0.125, 0.375, 0.375});
	ions.insert_fractional("Si", {0.25, 0.75, 0.75});

	systems::electrons el(ions, options::electrons{}.cutoff(30.0_Ry), input::kpoints::grid({1, 1, 1}, true));
	
        std::string restart_dir = "Si8_restart";
        auto not_found_gs = groundstate_only or not el.try_load(restart_dir);
        if(not_found_gs){
                inq::ground_state::initial_guess(ions, el);
                try { inq::ground_state::calculate(ions, el, inq::options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-8_Ha)); }
                catch(...){ }
                el.save(restart_dir);
        }

        real_time::propagate(ions, el, [](auto){}, options::theory{}.pbe(), options::real_time{}.num_steps(100).dt(0.0565_atomictime).tdmlwf());

        return 1;
}

