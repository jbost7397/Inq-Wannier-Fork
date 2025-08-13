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

	auto & env = inq::input::environment::global();
        auto ions = systems::ions::parse(config::path::unit_tests_data() + "benzne.xyz", systems::cell::cubic(30.0_b).periodic());
        ions.species_list().pseudopotentials() = pseudo::set_id::sg15();
	inq::systems::electrons el(env.par().states().domains(1), ions, options::electrons{}.cutoff(30.0_Ry));

        std::string restart_dir = "benzene_restart";
        auto not_found_gs = groundstate_only or not el.try_load(restart_dir);
        if(not_found_gs){
                inq::ground_state::initial_guess(ions, el);
                try { inq::ground_state::calculate(ions, el, inq::options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-8_Ha)); }
                catch(...){ }
                el.save(restart_dir);
        }

        if(not groundstate_only){
                        real_time::propagate(ions, el, [](auto){}, options::theory{}.pbe(), options::real_time{}.num_steps(100).dt(0.0565_atomictime).tdmlwf());
        }

        return 1;
}
