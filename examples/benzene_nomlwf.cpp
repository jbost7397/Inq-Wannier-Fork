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

	inq::systems::ions sys(inq::systems::cell::cubic(25.0_b).periodic());
	sys.insert(ionic::species("C").pseudo_file(inq::config::path::pseudo() + "C_ONCV_PBE-1.2.upf.gz"), {0.6825_A, -0.0924_A, 1.2087_A});
        sys.insert(ionic::species("C").pseudo_file(inq::config::path::pseudo() + "C_ONCV_PBE-1.2.upf.gz"), {-0.7075_A, -0.0352_A, 1.1973_A});
        sys.insert(ionic::species("C").pseudo_file(inq::config::path::pseudo() + "C_ONCV_PBE-1.2.upf.gz"), {-1.3898_A, 0.0572_A, -0.0114_A});
        sys.insert(ionic::species("C").pseudo_file(inq::config::path::pseudo() + "C_ONCV_PBE-1.2.upf.gz"), {-0.6824_A, 0.0925_A, -1.2088_A});
        sys.insert(ionic::species("C").pseudo_file(inq::config::path::pseudo() + "C_ONCV_PBE-1.2.upf.gz"), {0.7075_A, 0.0352_A, -1.1973_A});
        sys.insert(ionic::species("C").pseudo_file(inq::config::path::pseudo() + "C_ONCV_PBE-1.2.upf.gz"), {1.3899_A, -0.0572_A, 0.0114_A});
        sys.insert(ionic::species("H").pseudo_file(inq::config::path::pseudo() + "H_ONCV_PBE-1.2.upf.gz"), {1.2194_A, -0.1652_A, 2.1600_A}); 
        sys.insert(ionic::species("H").pseudo_file(inq::config::path::pseudo() + "H_ONCV_PBE-1.2.upf.gz"), {-1.2644_A, -0.0630_A, 2.1393_A});
        sys.insert(ionic::species("H").pseudo_file(inq::config::path::pseudo() + "H_ONCV_PBE-1.2.upf.gz"), {-2.4836_A, 0.1021_A, -0.0204_A});
        sys.insert(ionic::species("H").pseudo_file(inq::config::path::pseudo() + "H_ONCV_PBE-1.2.upf.gz"), {-1.2194_A, 0.1652_A, -2.1599_A});
        sys.insert(ionic::species("H").pseudo_file(inq::config::path::pseudo() + "H_ONCV_PBE-1.2.upf.gz"), {1.2641_A, 0.0628_A, -2.1395_A});
        sys.insert(ionic::species("H").pseudo_file(inq::config::path::pseudo() + "H_ONCV_PBE-1.2.upf.gz"), {2.4836_A, -0.1022_A, 0.0205_A});
	inq::systems::electrons el(sys, options::electrons{}.cutoff(30.0_Ry));
	inq::ground_state::initial_guess(sys, el);
	
	inq::ground_state::calculate(sys, el, inq::options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-8_Ha));
	real_time::propagate(sys, el, [](auto){}, options::theory{}.pbe(), options::real_time{}.num_steps(100).dt(0.0565_atomictime));
	
	return 1;
}
