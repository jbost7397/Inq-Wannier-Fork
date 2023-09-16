/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__GROUND_STATE__CALCULATOR
#define INQ__GROUND_STATE__CALCULATOR

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cfloat>

#include <systems/ions.hpp>
#include <basis/real_space.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <states/ks_states.hpp>
#include <hamiltonian/ks_hamiltonian.hpp>
#include <hamiltonian/self_consistency.hpp>
#include <hamiltonian/energy.hpp>
#include <hamiltonian/forces.hpp>
#include <basis/field_set.hpp>
#include <operations/randomize.hpp>
#include <operations/overlap.hpp>
#include <operations/orthogonalize.hpp>
#include <operations/preconditioner.hpp>
#include <operations/integral.hpp>
#include <observables/density.hpp>
#include <parallel/gather.hpp>
#include <mixers/linear.hpp>
#include <mixers/broyden.hpp>
#include <eigensolvers/steepest_descent.hpp>
#include <math/complex.hpp>
#include <ions/interaction.hpp>
#include <observables/dipole.hpp>
#include <options/ground_state.hpp>
#include <systems/electrons.hpp>
#include <ground_state/eigenvalue_output.hpp>
#include <ground_state/subspace_diagonalization.hpp>

#include<tinyformat/tinyformat.h>

#include<spdlog/spdlog.h>
#include<spdlog/sinks/stdout_color_sinks.h>

#include<memory>

#include <utils/profiling.hpp>

namespace inq {
namespace ground_state {

class calculator {

public:

	using energy_type = hamiltonian::energy;
	using forces_type = gpu::array<vector3<double>, 1>;

private:
	
	systems::ions const & ions_;
	options::theory inter_;
	options::ground_state solver_;
	hamiltonian::self_consistency<> sc_;
	hamiltonian::ks_hamiltonian<double> ham_;
	
	template <typename NormResType>
	static auto state_convergence(systems::electrons & el, NormResType const & normres) {
		auto state_conv = 0.0;
		
		for(int iphi = 0; iphi < el.kpin_size(); iphi++){
			state_conv += operations::sum(el.occupations()[iphi], normres[iphi], [](auto occ, auto nres){ return fabs(occ*nres); });
		}
		
		el.kpin_states_comm().all_reduce_n(&state_conv, 1);
		state_conv /= el.states().num_electrons();
		
		return state_conv;
	}

public:

	calculator(systems::ions const & ions, systems::electrons const & electrons, const options::theory & inter = {}, options::ground_state const & solver = {})
		:ions_(ions),
		 inter_(inter),
		 solver_(solver),
		 sc_(inter, electrons.states_basis(), electrons.density_basis(), electrons.states().num_density_components()),
		 ham_(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions_, sc_.exx_coefficient(), /* use_ace = */ true)
	{
	}

	struct result {
		energy_type energy;
		vector3<double> dipole;
		forces_type forces;
		int total_iter;
	};
	
	result operator()(systems::electrons & electrons){
		
		CALI_CXX_MARK_FUNCTION;
		
		assert(electrons.kpin()[0].full_comm() == electrons.states_basis_comm());
		
		auto console = electrons.logger();
		if(solver_.verbose_output() and console) console->trace("ground-state calculation started");
		
		if(electrons.full_comm().root()) ham_.info(std::cout);
		
		result res;
		operations::preconditioner prec;
		
		using mix_arr_type = std::remove_reference_t<decltype(electrons.spin_density().matrix().flatted())>;
		
		auto mixer = [&]()->std::unique_ptr<mixers::base<mix_arr_type>>{
			switch(solver_.mixing_algorithm()){
			case options::ground_state::mixing_algo::LINEAR : return std::make_unique<mixers::linear <mix_arr_type>>(solver_.mixing());
			case options::ground_state::mixing_algo::BROYDEN: return std::make_unique<mixers::broyden<mix_arr_type>>(4, solver_.mixing(), electrons.spin_density().matrix().flatted().size(), electrons.density_basis().comm());
			} __builtin_unreachable();
		}();
		
		auto old_energy = std::numeric_limits<double>::max();
		
		sc_.update_ionic_fields(electrons.states_comm(), ions_, electrons.atomic_pot());
		sc_.update_hamiltonian(ham_, res.energy, electrons.spin_density());
		
		res.energy.ion(inq::ions::interaction_energy(ions_.cell(), ions_, electrons.atomic_pot()));
		
		double old_exe = ham_.exchange.update(electrons);
		double exe_diff = fabs(old_exe);
		auto update_hf = false;
		
		electrons.full_comm().barrier();
		auto iter_start_time = std::chrono::high_resolution_clock::now();

		res.total_iter = solver_.scf_steps();
		int conv_count = 0;
		for(int iiter = 0; iiter < solver_.scf_steps(); iiter++){
			
			CALI_CXX_MARK_SCOPE("scf_iteration");
			
			if(solver_.subspace_diag()) {
				int ilot = 0;
				for(auto & phi : electrons.kpin()) {
					electrons.eigenvalues()[ilot] = subspace_diagonalization(ham_, phi);
					ilot++;
				}
				electrons.update_occupations(electrons.eigenvalues());
			}
			
			if(update_hf){
				auto exe = ham_.exchange.update(electrons);
				exe_diff = fabs(exe - old_exe);
				old_exe = exe;
			}
			
			for(auto & phi : electrons.kpin()) {
				auto fphi = operations::transform::to_fourier(std::move(phi));
				
				switch(solver_.eigensolver()){
					
				case options::ground_state::scf_eigensolver::STEEPEST_DESCENT:
					eigensolvers::steepest_descent(ham_, prec, fphi);
					break;
					
				default:
					assert(false);
				}
				
				phi = operations::transform::to_real(std::move(fphi));
			}
			
			CALI_MARK_BEGIN("mixing");
			
			double density_diff = 0.0;
			{
				auto new_density = observables::density::calculate(electrons);
				density_diff = operations::integral_sum_absdiff(electrons.spin_density(), new_density);
				density_diff /= electrons.states().num_electrons();
				
				if(inter_.self_consistent()) {
					auto tmp = +electrons.spin_density().matrix().flatted();
					mixer->operator()(tmp, new_density.matrix().flatted());
					electrons.spin_density().matrix().flatted() = tmp;
					observables::density::normalize(electrons.spin_density(), electrons.states().num_electrons());
				} else {
					electrons.spin_density() = std::move(new_density);
				}
			}
			
			sc_.update_hamiltonian(ham_, res.energy, electrons.spin_density());
			
			CALI_MARK_END("mixing");
			
			{
				auto normres = res.energy.calculate(ham_, electrons);
				auto energy_diff = (res.energy.eigenvalues() - old_energy)/electrons.states().num_electrons();

				electrons.full_comm().barrier();
				std::chrono::duration<double> elapsed_seconds = std::chrono::high_resolution_clock::now() - iter_start_time;
				
				electrons.full_comm().barrier();
				iter_start_time = std::chrono::high_resolution_clock::now();
				
				auto state_conv = state_convergence(electrons, normres);
				auto ev_out = eigenvalues_output(electrons, normres);
				
				if(solver_.verbose_output() and console){
					console->info("\nSCF iter {} : wtime = {:5.2f}s e = {:.10f} de = {:5.0e} dexe = {:5.0e} dn = {:5.0e} dst = {:5.0e}\n{}", 
												iiter, elapsed_seconds.count(), res.energy.total(), energy_diff, exe_diff, density_diff, state_conv, ev_out);
				}
				
				if(fabs(energy_diff) < solver_.energy_tolerance()){
					conv_count++;
					if(conv_count > 2 and exe_diff < solver_.energy_tolerance()) {
						res.total_iter = iiter;
						break;
					}
					if(conv_count > 2) update_hf = true;
				} else {
					conv_count = 0; 
				}

				old_energy = res.energy.eigenvalues();
			}
		}

		//make sure we have a density consistent with phi
		electrons.spin_density() = observables::density::calculate(electrons);
		sc_.update_hamiltonian(ham_, res.energy, electrons.spin_density());
		auto normres = res.energy.calculate(ham_, electrons);
			
		if(solver_.calc_forces()) res.forces = hamiltonian::calculate_forces(ions_, electrons, ham_);
		
		auto ev_out = eigenvalues_output(electrons, normres);		
		
		if(solver_.verbose_output() and console) {
			console->info("\nSCF iters ended with resulting eigenvalues and energies:\n\n{}{}", ev_out.full(), res.energy);
		}
		
		if(ions_.cell().periodicity() == 0){
			res.dipole = observables::dipole(ions_, electrons);
		} else {
			res.dipole = vector3<double>(0.);
		}
	
		if(solver_.verbose_output() and console) console->trace("ground-state calculation ended normally");
		return res;
	}

	/////////////////////////////////////////

	auto & hamiltonian() const {
		return ham_;
	}
	
	
};
}
}
#endif

#ifdef INQ_GROUND_STATE_CALCULATOR_UNIT_TEST
#undef INQ_GROUND_STATE_CALCULATOR_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
