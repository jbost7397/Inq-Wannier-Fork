/* -*- indent-tabs-mode: t -*- */
#ifndef INQ__REAL_TIME__PROPAGATE
#define INQ__REAL_TIME__PROPAGATE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <systems/ions.hpp>
#include <hamiltonian/self_consistency.hpp>
#include <operations/overlap_diagonal.hpp>
#include <observables/dipole.hpp>
#include <observables/forces_stress.hpp>
#include <observables/mlwf_properties.hpp>
#include <options/real_time.hpp>
#include <perturbations/none.hpp>
#include <ionic/propagator.hpp>
#include <systems/electrons.hpp>
#include <real_time/crank_nicolson.hpp>
#include <real_time/etrs.hpp>
#include <real_time/im_etrs.hpp>
#include <real_time/viewables.hpp>
#include <utils/profiling.hpp>
#include <wannier/tdmlwf_trans.hpp>

#include <chrono>

namespace inq {
namespace real_time {

template <typename ProcessFunction, typename IonSubPropagator = ionic::propagator::fixed, typename Perturbation = perturbations::none>
void propagate(systems::ions & ions, systems::electrons & electrons, ProcessFunction func, const options::theory & inter,
							 const options::real_time & opts, Perturbation const & pert = {}, int const start_step = 0){

	assert(start_step >= 0);

		CALI_CXX_MARK_FUNCTION;

		auto console = electrons.logger();

		ionic::propagator::runtime ion_propagator{opts.ion_dynamics_value()};

		if(start_step > 0) assert(ion_propagator.static_ions()); //restart doesn't work with moving ions for now

		const double dt = opts.dt();
		const int numsteps = opts.num_steps();

		if(console) {
			console->trace(std::string("initializing real-time propagation:\n") +
										 std::string("  time step        = {} atomictime ({:.2f} as)\n") +
										 std::string("  number of steps  = {}\n") +
										 std::string("  propagation time = {} atomictime ({:.2f} fs)"), dt, dt/0.041341373, numsteps, numsteps*dt, numsteps*dt/41.341373);
		if(start_step > 0) console->trace("restarting propagation from step {}", start_step);
			console->trace("\n{}", pert);
		}

	if(start_step == 0) {
		for(auto & phi : electrons.kpin()) pert.zero_step(phi);
		
	}
		observables::mlwf_properties mlwf_props;
		if (opts.wf_diag_value() == options::real_time::wavefunction_diag::TDMLWF && start_step == 0) { //JLB
			mlwf_props.set_mlwf_transformer(wannier::tdmlwf_trans(electrons.kpin()[0]));
    			std::ofstream output_file("mlwf_results.dat", std::ios_base::app);
			mlwf_props.calculate(output_file, -1, electrons.kpin()[0], opts.mlwf_freq());
       			output_file.close();
		}

		electrons.spin_density() = observables::density::calculate(electrons);

		hamiltonian::self_consistency sc(inter, electrons.states_basis(), electrons.density_basis(), electrons.states().num_density_components(), pert);
		hamiltonian::ks_hamiltonian<complex> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(),
																						 ions, sc.exx_coefficients(), /* use_ace = */ opts.propagator() == options::real_time::electron_propagator::CRANK_NICOLSON, mlwf_props.get_mlwf_transformer(), opts.epsilon());

		hamiltonian::energy energy;

		sc.update_ionic_fields(electrons.states_comm(), ions, electrons.atomic_pot());
	sc.update_hamiltonian(ham, energy, electrons.spin_density(), /* time = */ start_step*dt);

		ham.exchange().update(electrons);

		energy.calculate(ham, electrons);
		energy.ion(ionic::interaction_energy(ions.cell(), ions, electrons.atomic_pot()));

	auto forces = decltype(observables::forces_stress{ions, electrons, ham, energy}.forces){};
	if(ion_propagator.needs_force()) forces = observables::forces_stress{ions, electrons, ham, energy}.forces;

		auto current = vector3<double, covariant>{0.0, 0.0, 0.0};
		if(sc.has_induced_vector_potential()) current = observables::current(ions, electrons, ham);

	if(start_step == 0) func(real_time::viewables{false, start_step, start_step*dt, ions, electrons, energy, forces, ham, pert});
	if (opts.wf_diag_value() == options::real_time::wavefunction_diag::TDMLWF) { //JLB
		auto transformer = wannier::tdmlwf_trans(electrons.kpin()[0]); //might have done this previously if starting from GS, so maybe this is a waste but probably doesn't matter too much
	        mlwf_props.set_mlwf_transformer(transformer);
	}

		if(console) console->trace("starting real-time propagation");
	if(console) console->info("step {:9d} :  t =  {:9.3f}  e = {:.12f}", start_step, start_step*dt, energy.total());

		auto iter_start_time = std::chrono::high_resolution_clock::now();

                double last_etot = energy.total(); //VS
                int conv_count = 0;
                int last_step = start_step;

	for(int istep = start_step; istep < numsteps; istep++){
			CALI_CXX_MARK_SCOPE("time_step");

			switch(opts.propagator()){
			case options::real_time::electron_propagator::ETRS :
				etrs(istep*dt, dt, ions, electrons, ion_propagator, forces, current, ham, sc, energy);
				break;
			case options::real_time::electron_propagator::CRANK_NICOLSON :
				crank_nicolson(istep*dt, dt, ions, electrons, ion_propagator, forces, ham, sc, energy);
				break;
			case options::real_time::electron_propagator::IM_ETRS :
				im_etrs(istep*dt, dt, ions, electrons, ion_propagator, forces, current, ham, sc, energy);
				break;
			}

			if (opts.wf_diag_value() == options::real_time::wavefunction_diag::TDMLWF) { //JLB
				{
    					std::ofstream output_file("mlwf_results.dat", std::ios_base::app);
			        	mlwf_props.calculate(output_file, istep, electrons.kpin()[0], opts.mlwf_freq());
       					output_file.close();
				}
			}

			energy.calculate(ham, electrons);

		if (opts.propagator() == options::real_time::electron_propagator::IM_ETRS && opts.im_etrs_thresh()) {

			double etot = energy.total();
		        if (fabs(etot - last_etot) < opts.im_etrs_etol()) {
				conv_count++;
			} else {
				conv_count = 0;
			}

		        if (conv_count == opts.im_etrs_covg_steps()) {
				last_step = istep + 1;

				if (console) console->info("Change in energy less than {:.8f} for {:2d} steps", opts.im_etrs_etol(), opts.im_etrs_covg_steps());
      			        if (console) console->info("IM_ETRS converged at step {:5d} with a total energy of {:.8f}", last_step, etot);

	        	        func(real_time::viewables{true, last_step, (istep + 1.0)*dt, ions, electrons, energy, forces, ham, pert});

				for (int ik = 0; ik < electrons.kpin_size(); ++ik) {
					auto & phi = electrons.kpin()[ik];
         	       	 		auto Hsub = operations::overlap(phi, ham(phi));
			                auto evals = matrix::diagonalize(Hsub);
			                operations::rotate(Hsub, phi);
					auto kpt = electrons.brillouin_zone().kpoint(ik) / (2.0 * M_PI);
					if (console) {
					    	if (electrons.brillouin_zone().size() > 1) {						 
            					   console->info("k-point {:4d}  ({: .6f}, {: .6f}, {: .6f})", ik + 1, kpt[0], kpt[1], kpt[2]);
						} 
        					for (int st = 0; st < evals.size(); ++st) {
					            double e_Ha = real(evals[st]);
					            double e_eV = e_Ha * 27.211383;
					            console->info("    st = {:4d}  evalue = {:18.12f} Ha ({:18.12f} eV)", st + 1, e_Ha, e_eV);
						}
					}	
			        }
        	        	break;
        		}
			last_etot = etot; 
		}			

		if(ion_propagator.needs_force()) forces = observables::forces_stress{ions, electrons, ham, energy}.forces;

			//propagate ionic velocities to t + dt
			ion_propagator.propagate_velocities(dt, ions, forces);

			if(sc.has_induced_vector_potential()) {
				current = observables::current(ions, electrons, ham);
				sc.propagate_induced_vector_potential_derivative(dt, current);
			}

		func(real_time::viewables{istep == numsteps - 1, istep + 1, (istep + 1.0)*dt, ions, electrons, energy, forces, ham, pert});

			auto new_time = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
			last_step++;

			
			if(console) console->info("step {:9d} :  t =  {:9.3f}  e = {:.12f}  wtime = {:9.3f}", istep + 1, (istep + 1)*dt, energy.total(), elapsed_seconds.count());

                        if (opts.propagator() == options::real_time::electron_propagator::IM_ETRS && istep == numsteps - 1) {
                                for (int ik = 0; ik < electrons.kpin_size(); ++ik) {
                                        auto & phi = electrons.kpin()[ik];
                                        auto Hsub = operations::overlap(phi, ham(phi));
                                        auto evals = matrix::diagonalize(Hsub);
                                        operations::rotate(Hsub, phi);
                                        auto kpt = electrons.brillouin_zone().kpoint(ik) / (2.0 * M_PI);
                                        if (console) {
                                                if (electrons.brillouin_zone().size() > 1) {
                                                   console->info("k-point {:4d}  ({: .6f}, {: .6f}, {: .6f})", ik + 1, kpt[0], kpt[1], kpt[2]);
                                                }
                                                for (int st = 0; st < evals.size(); ++st) {
                                                    double e_Ha = real(evals[st]);
                                                    double e_eV = e_Ha * 27.211383;
                                                    console->info("    st = {:4d}  evalue = {:18.12f} Ha ({:18.12f} eV)", st + 1, e_Ha, e_eV);
                                                }
                                        }
                                }
                        }

			iter_start_time = new_time;
		}

		if(console) console->trace("real-time propagation ended normally");
	}
}
}
#endif

#ifdef INQ_REAL_TIME_PROPAGATE_UNIT_TEST
#undef INQ_REAL_TIME_PROPAGATE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
