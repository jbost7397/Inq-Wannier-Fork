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

	input::environment env{};

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

	auto const nk = 6;  // increase later for production run

	systems::electrons electrons(env.par(), ions, input::kpoints::grid({nk, nk, nk}, false), options::electrons{}.spacing(a/32));

	auto functional = options::theory{}.pbe();
	
	// try{
	//  electrons.load("silicon_restart");
	// } catch (...) {
	//  ground_state::initial_guess(ions, electrons);
	//  ground_state::calculate(ions, electrons, options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-4_Ha));
	//  ground_state::calculate(ions, electrons, functional, inq::options::ground_state{}.energy_tolerance(1e-8_Ha));
	//  electrons.save("silicon_restart");
	// }

	if(not electrons.try_load("silicon_restart")){
		ground_state::initial_guess(ions, electrons);
		ground_state::calculate(ions, electrons, options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-4_Ha));
		ground_state::calculate(ions, electrons, functional, inq::options::ground_state{}.energy_tolerance(1e-8_Ha));
		electrons.save("silicon_restart");
	}


	auto const dt = 0.010000;
	long nsteps = 10000; //413.41373/dt;
	
	gpu::array<double, 1> time(nsteps);
	gpu::array<double, 1> cur(nsteps);
	gpu::array<double, 1> en(nsteps);   
	gpu::array<double, 1> aind(nsteps);

	std::ofstream file;
	if(electrons.root()) {
		file.open("current.dat");
		file << "# time, current_x, A_x total (ext + induced)\n";
	}
	
	auto output = [&](auto data){
		
		auto iter = data.iter();
		
		time[iter] = data.time();
		cur[iter] = data.current()[0];
		en[iter] = data.energy().total();
		aind[iter] = data.uniform_vector_potential()[0];
		
		if(data.root()) file << time[iter] << '\t' << cur[iter] << '\t' << aind[iter] << std::endl;
		
		if(data.root() and data.every(50)){
			auto spectrum = observables::spectrum(20.0_eV, 0.01_eV, time({0, iter - 1}), cur({0, iter - 1}));  
			
			{
				std::ofstream sfile("spectrum.dat");
				sfile << "# Frequency (eV) \t Current Real \t Imaginary\n";
				
				for(int ifreq = 0; ifreq != spectrum.size(); ++ifreq) {
					sfile << ifreq*in_atomic_units(0.01_eV) << '\t' << real(spectrum[ifreq]) << '\t' << imag(spectrum[ifreq]) << std::endl;
				}
			}
			{
				std::vector<double> hhg(spectrum.size());
				for(int ifreq = 0; ifreq != spectrum.size(); ++ifreq) {
					hhg[ifreq] = norm(spectrum[ifreq] * ifreq*in_atomic_units(0.01_eV));
				}

				std::ofstream hfile("hhg.dat");
				hfile << "# Frequency (eV) \t HHG Real \t Imaginary\n";

				for(int ifreq = 0; ifreq != hhg.size(); ++ifreq) {
					hfile << ifreq*in_atomic_units(0.01_eV) << '\t' << hhg[ifreq] << std::endl;
				}
			}
		}
	};

	// 0.01
	auto kick = perturbations::kick{ions.cell(), {0.0001, 0.0, 0.0}, perturbations::gauge::velocity};

	real_time::propagate<>(
		ions, electrons, output, functional.induced_vector_potential(4.0*M_PI), options::real_time{}.num_steps(nsteps).dt(dt*1.0_atomictime).etrs(), ions::propagator::fixed{}, 
		// perturbations::laser({0.1, 0.0, 0.0}, 1.0_eV, perturbations::gauge::velocity)
		kick
	);
	
	return energy_match.fail();
	
}


