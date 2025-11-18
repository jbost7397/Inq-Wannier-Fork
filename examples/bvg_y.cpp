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
	int pardomains = 1;
	int mlwf_freq = 0;
	auto functional = options::theory{}.pbe();

	{
                int opt;
                while ((opt = getopt(argc, argv, "p:?gm:yi:")) != EOF){
                        switch(opt){
                        case 'p':
                                pardomains = atoi(optarg);
                                break;
			case 'm':
				mlwf_freq = atoi(optarg);
				break;
                        case 'y':
                                functional = options::theory{}.pbe0();
                                break;
                        case '?':
                                std::cerr << "usage is " << std::endl;
                                std::cerr << "-p N to set the number of processors in the domain partition (1 by default)." << std::endl;
				std::cerr << "-m MLWF_FREQUENCY: Set the frequency for printing MLWF properties (0 = never, default = 0)." << std::endl;
                                std::cerr << "-y use the PBE0 hybrid functional (the default is PBE)." << std::endl;
                                exit(1);
                        default:
                                abort();
                        }
                }
        }

	auto & env = inq::input::environment::global();

	auto ions = systems::ions::parse("../../examples/benzene.xyz", systems::cell::cubic(100.0_b));
	auto electrons = systems::electrons(env.par().states().domains(pardomains), ions, options::electrons{}.cutoff(30.0_Ry));

	if(not electrons.try_load("benzene_restart")){
		ground_state::initial_guess(ions, electrons);
		ground_state::calculate(ions, electrons, functional, inq::options::ground_state{}.energy_tolerance(1e-4_Ha));
		ground_state::calculate(ions, electrons, functional, inq::options::ground_state{}.energy_tolerance(1e-8_Ha));
		electrons.save("benzene_restart");
	}

	auto kick = perturbations::kick(ions.cell(), {0.0, 0.001, 0.0}, perturbations::gauge::velocity);

	auto const timestep = 0.05_atomictime;
	auto const nsteps = 250.0_atomictime/timestep;

	gpu::array<double, 1> time(nsteps);
	gpu::array<double, 1> dip(nsteps);
	gpu::array<double, 1> cur(nsteps);
	gpu::array<double, 1> en(nsteps);

	auto output = [&](auto data){

		auto iter = data.iter();

		time[iter] = data.time();
		cur[iter] = data.current()[1];
		en[iter] = data.energy().total();
		dip[iter] = data.dipole()[1];

                if (data.root()) {
                        std::ofstream tfile("time_benz_y.dat", std::ios::app);
                        std::ofstream dfile("dipole_benz_y.dat", std::ios::app);
                        std::ofstream efile("energy_benz_y.dat", std::ios::app);
                        std::ofstream cfile("current_benz_y.dat", std::ios::app);
 
                        tfile << std::fixed << std::setprecision(6);
                        dfile << std::fixed << std::setprecision(8);
                        efile << std::fixed << std::setprecision(6);
                        cfile << std::fixed << std::setprecision(8);
 
                        cfile << cur[iter] << '\n';
                        tfile << time[iter] << '\n';
                        dfile << dip[iter] << '\n';
                        efile << en[iter] << '\n';
               }
	};

	real_time::propagate(ions, electrons, output, functional, options::real_time{}.num_steps(nsteps).dt(timestep), kick);

	return 1;
}
