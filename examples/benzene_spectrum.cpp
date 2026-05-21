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
	//int mlwf_freq = 0;
	auto functional = options::theory{}.pbe();

	{
                int opt;
                while ((opt = getopt(argc, argv, "p:?gm:yi:")) != EOF){
                        switch(opt){
                        case 'p':
                                pardomains = atoi(optarg);
                                break;
                        case 'y':
                                functional = options::theory{}.pbe0();
                                break;
                        case '?':
                                std::cerr << "usage is " << std::endl;
                                std::cerr << "-p N to set the number of processors in the domain partition (1 by default)." << std::endl;
                                std::cerr << "-y use the PBE0 hybrid functional (the default is PBE)." << std::endl;
                                exit(1);
                        default:
                                abort();
                        }
                }
        }

	auto & env = inq::input::environment::global();

	auto ions = systems::ions::parse("../../examples/benzene.xyz", 10.0_b);
	auto electrons = systems::electrons(env.par().states().domains(pardomains), ions, options::electrons{}.spacing(0.43_b));

	if(not electrons.try_load("benzene_restart")){
		ground_state::initial_guess(ions, electrons);
		ground_state::calculate(ions, electrons, functional, inq::options::ground_state{}.energy_tolerance(1e-4_Ha));
		ground_state::calculate(ions, electrons, functional, inq::options::ground_state{}.energy_tolerance(1e-8_Ha));
		electrons.save("benzene_restart");
	}

	auto kick = perturbations::kick(ions.cell(), {0.01, 0.0, 0.0});

	auto const timestep = 0.02_atomictime;
	auto const nsteps = 200.0_atomictime/timestep;

	gpu::array<double, 1> time(nsteps);
	gpu::array<double, 1> dip(nsteps);

	auto output = [&](auto data){

		auto iter = data.iter();

		time[iter] = data.time();
  		dip[iter] = data.dipole()[0];

		if(data.root()){
			std::ofstream file("dipole.dat", std::ios_base::app);
			file << iter << '\t' << data.dipole()[0] << '\t' << data.dipole()[1] << '\t' << data.dipole()[2] << std::endl;
		}

 		if(data.root() and data.every(1000)){
 			auto spectrum = observables::spectrum(20.0_eV, 0.01_eV, time({0, iter - 1}), dip({0, iter - 1}));

    			std::ofstream file("spectrum.dat");

    			for(int ifreq = 0; ifreq < spectrum.size(); ifreq++){
     				file << ifreq*0.01 << '\t' << real(spectrum[ifreq])  << '\t' << imag(spectrum[ifreq]) << std::endl;
    			}
  		}
	};

	real_time::propagate(ions, electrons, output, functional, options::real_time{}.num_steps(nsteps).dt(timestep), kick);

	return 1;
}
