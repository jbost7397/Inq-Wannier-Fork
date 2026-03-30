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

	inq::systems::ions sys(inq::systems::cell::cubic(20.0_b).periodic());
        sys.insert(ionic::species("He").pseudo_file(inq::config::path::pseudo() + "He_ONCV_PBE-1.2.upf.gz"), {3.0_b, 3.0_b, 3.0_b});
        sys.insert(ionic::species("He").pseudo_file(inq::config::path::pseudo() + "He_ONCV_PBE-1.2.upf.gz"), {18.0_b, 18.0_b, 18.0_b});
	inq::systems::electrons el(sys, options::electrons{}.cutoff(30.0_Ry));
        
	std::string restart_dir = "helium_restart";
        auto not_found_gs = not el.try_load(restart_dir);
        if(not_found_gs){
                inq::ground_state::initial_guess(sys, el);
                try { inq::ground_state::calculate(sys, el, inq::options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-10_Ha)); }
                catch(...){ }
                el.save(restart_dir);
        }
	real_time::propagate(sys, el, [](auto){}, options::theory{}.pbe(), options::real_time{}.num_steps(10).dt(0.0565_atomictime).tdmlwf());
	
	return 1;
	
}
