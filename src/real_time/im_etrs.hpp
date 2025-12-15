/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__REAL_TIME__IM_ETRS
#define INQ__REAL_TIME__IM_ETRS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <observables/density.hpp>
#include <observables/current.hpp>
#include <operations/exponential.hpp>
#include <systems/electrons.hpp>
#include <systems/ions.hpp>
#include <utils/profiling.hpp>
#include <operations/orthogonalize.hpp>

namespace inq {
namespace real_time {

template <class IonSubPropagator, class ForcesType, class CurrentType, class HamiltonianType, class SelfConsistencyType, class EnergyType>
void im_etrs(double const time, double const dt, systems::ions & ions, systems::electrons & electrons, IonSubPropagator const & ion_propagator, ForcesType const & forces, CurrentType const & current,
					HamiltonianType & ham, SelfConsistencyType & sc, EnergyType & energy){
	CALI_CXX_MARK_FUNCTION;

	int const nscf = 5;
	double const scf_threshold = 5e-5;

	systems::electrons::kpin_type save;

	for(auto & phi : electrons.kpin()){
		
		//propagate half step and full step with H(t)
		auto halfstep_phi = operations::exponential_2_for_1(ham, complex(-dt/2.0, 0.0), complex(-dt, 0.0), phi);
		{ CALI_CXX_MARK_SCOPE("im_etrs:save");
		  save.emplace_back(std::move(halfstep_phi));
		}
	}

	electrons.spin_density() = observables::density::calculate(electrons);
	
	//propagate the Hamiltonian to t + dt
	ion_propagator.propagate_positions(dt, ions, forces);	
	if(not ion_propagator.static_ions()) {
		sc.update_ionic_fields(electrons.states_comm(), ions, electrons.atomic_pot());
		ham.update_projectors(electrons.states_basis(), electrons.atomic_pot(), ions);
		energy.ion(ionic::interaction_energy(ions.cell(), ions, electrons.atomic_pot()));
	}
	sc.propagate_induced_vector_potential(dt, current);
	sc.update_hamiltonian(ham, energy, electrons.spin_density(), time + dt);
	ham.exchange().update(electrons);

	{ CALI_CXX_MARK_SCOPE("im_etrs:restore");
		electrons.kpin() = save;
	}

	//propagate the other half step with H(t + dt) self-consistently
	for(int iscf = 0; iscf < nscf; iscf++){

		int iphi = 0;
		for(auto & phi : electrons.kpin()) {
			if(iscf != 0) phi = save[iphi];
			operations::exponential_in_place(ham, complex(-dt/2.0, 0.0), phi);
			iphi++;
		}

        	for(auto & phi : electrons.kpin()){ //CS seperate step now for merging purposes 
                	operations::orthogonalize(phi);
        	}

		auto old_density = electrons.spin_density();
		electrons.spin_density() = observables::density::calculate(electrons);

		double delta = operations::integral_sum_absdiff(old_density, electrons.spin_density());
		auto done = (delta < scf_threshold) or (iscf == nscf - 1);
		
		sc.update_hamiltonian(ham, energy, electrons.spin_density(), time + dt);
		ham.exchange().update(electrons);
		if(done) break;
	}

	for(auto & phi : electrons.kpin()){
		auto Hsub = operations::overlap(phi, ham(phi));
		auto evals = matrix::diagonalize(Hsub);
		operations::rotate(Hsub, phi);
	}
    }

}
}
#endif

#ifdef INQ_REAL_TIME_IM_ETRS_UNIT_TEST
#undef INQ_REAL_TIME_IM_ETRS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
