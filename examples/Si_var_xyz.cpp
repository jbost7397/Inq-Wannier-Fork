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
	int pardomains = 1;
	int mlwf_freq = 1;
	int niter = 10;
	int x_repeats = 1;
	int y_repeats = 1;
	int z_repeats = 1;
	auto functional = options::theory{}.pbe();
	auto extra_states = 0;
	auto nk = 1;
	auto mix = 0.3;
	auto max = 200;
	auto time_step = 1.0;
	auto cut = 20.0;

	{
                int opt;
                while ((opt = getopt(argc, argv, "p:?gf:hli:x:y:z:e:n:m:s:t:c:")) != EOF){
                        switch(opt){
                        case 'p':
                                pardomains = atoi(optarg);
                                break;
                        case 'g':
                                groundstate_only = true;
                                break;
			case 'f':
				mlwf_freq = atoi(optarg);
				break;
                        case 'h':
                                functional = options::theory{}.pbe0();
                                break;
                        case 'l':
                                functional = options::theory{}.lda();
                                break;
                  	case 'i':
                                niter = atoi(optarg);
                                break;
			case 'x':
				x_repeats = atoi(optarg);
				break;
			case 'y':
				y_repeats = atoi(optarg);
				break;
			case 'z':
				z_repeats = atoi(optarg);
				break;
			case 'e':
				extra_states = atoi(optarg);
				break;
			case 'n':
				nk = atoi(optarg);
				break;
			case 'm':
				mix = atof(optarg);
				break;
			case 's':
				max = atoi(optarg);
				break;
			case 't':
				time_step = atof(optarg);
				break;
			case 'c':
				cut = atof(optarg);
				break;
                        case '?':
                                std::cerr << "usage is " << std::endl;
                                std::cerr << "-p N to set the number of processors in the domain partition (1 by default)." << std::endl;
                                std::cerr << "-g only calculate the ground state." << std::endl;
				std::cerr << "-f MLWF_FREQUENCY: Set the frequency for printing MLWF properties (0 = never, default = 1)." << std::endl;
                                std::cerr << "-y use the PBE0 hybrid functional (the default is PBE)." << std::endl;
                                std::cerr << "-i N to set the number of SCF iterations to run (10 by default)." << std::endl;
                                std::cerr << "-x Number of times to repeat in the x direction (1 by default)." << std::endl;
                                exit(1);
                        default:
                                abort();
                        }
                }
	}

	auto & env = inq::input::environment::global();

	auto a = 10.26314_b;
	auto x_dim = a * x_repeats;
	auto y_dim = a * y_repeats;
	auto z_dim = a * z_repeats;
	systems::ions ions(systems::cell::orthorhombic(x_dim, y_dim, z_dim));
	
        //Base positions of the 8 atoms in the unit cell
	std::vector<vector3<double>> base_positions = {
        {0.0,  0.0,  0.0 },
        {0.25, 0.25, 0.25},
        {0.5,  0.5,  0.0 },
        {0.75, 0.75, 0.25},
        {0.5,  0.0,  0.5 },
        {0.75, 0.25, 0.75},
        {0.0,  0.5,  0.5 },
        {0.25, 0.75, 0.75}
	};
    	for (int ix = 0; ix < x_repeats; ++ix) {
        	for (int iy = 0; iy < y_repeats; ++iy) {
        		for (int iz = 0; iz < z_repeats; ++iz) {
                		for (const auto& pos : base_positions) {
                			// Add each base position to the supercell.
					ions.insert_fractional("Si", {(pos[0] + ix)/x_repeats, (pos[1] + iy)/y_repeats, (pos[2] + iz)/z_repeats});
                		}
            	    	}
        	}
    	}

	auto ecut = cut * 1.0_Ry;
	auto dt = time_step * 1.0_as;

	systems::electrons el(env.par().states().domains(pardomains), ions, options::electrons{}.cutoff(ecut).extra_states(extra_states), input::kpoints::grid({nk, nk, nk}));
	
        std::string restart_dir = "Si" + std::to_string(8 * x_repeats * y_repeats * z_repeats) +  "_xyz_restart";
        auto not_found_gs = groundstate_only or not el.try_load(restart_dir);
        if(not_found_gs){
                inq::ground_state::initial_guess(ions, el);
                try { inq::ground_state::calculate(ions, el, functional, inq::options::ground_state{}.energy_tolerance(1e-5_Ha).mixing(mix).max_steps(max).broyden_mixing()); }
                catch(...){ }
                el.save(restart_dir);
        }

        real_time::propagate(ions, el, [](auto){}, functional, options::real_time{}.num_steps(niter).dt(dt).tdmlwf(mlwf_freq));

        return 1;
}

