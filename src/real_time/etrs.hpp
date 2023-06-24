/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__REAL_TIME__ETRS
#define INQ__REAL_TIME__ETRS

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

namespace inq {
namespace real_time {

template <class IonSubPropagator, class ForcesType, class HamiltonianType, class SelfConsistencyType, class EnergyType>
void etrs(double const time, double const dt, systems::ions & ions, systems::electrons & electrons, IonSubPropagator const & ion_propagator, ForcesType const & forces, HamiltonianType & ham, SelfConsistencyType & sc, EnergyType & energy){

	int const nscf = 5;
	double const scf_threshold = 5e-5;

	systems::electrons::kpin_type save;
	
	int iphi = 0;
	for(auto & phi : electrons.kpin()){
		
		//propagate half step and full step with H(t)
		auto halfstep_phi = operations::exponential_2_for_1(ham, complex(0.0, dt/2.0), complex(0.0, dt), phi);
		save.emplace_back(std::move(halfstep_phi));
									 		
		iphi++;
	}

	electrons.spin_density() = observables::density::calculate(electrons);
	
	//propagate ionic positions to t + dt
	ion_propagator.propagate_positions(dt, ions, forces);
	if(not ion_propagator.static_ions) {
		sc.update_ionic_fields(electrons.states_comm(), ions, electrons.atomic_pot());
		ham.update_projectors(electrons.states_basis(), electrons.atomic_pot(), ions.geo());
		energy.ion(inq::ions::interaction_energy(ions.cell(), ions.geo(), electrons.atomic_pot()));
	}

	auto orig_current = observables::current(ions, electrons, ham);
	auto orig_induced=ham.vp_induced;
	auto orig_velocity=ham.vp_velocity;
	auto orig_accel=ham.vp_accel;
	sc.update_induced_potential(orig_induced, orig_velocity, orig_accel, dt, ions.cell().volume(), orig_current);
	sc.update_hamiltonian(ham, energy, electrons.spin_density(), time + dt);
	ham.exchange.update(electrons);
	
	auto orig_vp=ham.uniform_vector_potential();

	auto tmp_current = observables::current(ions, electrons, ham);
	auto tmp_induced=ham.vp_induced;
	auto tmp_velocity=ham.vp_velocity;
	auto tmp_accel=ham.vp_accel;
	sc.update_induced_potential(tmp_induced, tmp_velocity, tmp_accel, dt, ions.cell().volume(), tmp_current);
	ham.uniform_vector_potential()+=tmp_induced;

	electrons.kpin() = save;
	
	//propagate the other half step with H(t + dt) self-consistently
	for(int iscf = 0; iscf < nscf; iscf++){

		int iphi = 0;
		for(auto & phi : electrons.kpin()) {
			if(iscf != 0) phi = save[iphi];
			operations::exponential_in_place(ham, complex(0.0, dt/2.0), phi);
			iphi++;
		}
		
		auto old_density = electrons.spin_density();
		electrons.spin_density() = observables::density::calculate(electrons);

		double delta = operations::integral_sum_absdiff(old_density, electrons.spin_density());
		auto done = (delta < scf_threshold) or (iscf == nscf - 1);
		
		sc.update_hamiltonian(ham, energy, electrons.spin_density(), time + dt);
		ham.exchange.update(electrons);
		
		tmp_current = observables::current(ions, electrons, ham);
		tmp_induced=ham.vp_induced;
		tmp_velocity=ham.vp_velocity;
		tmp_accel=ham.vp_accel;
		sc.update_induced_potential(tmp_induced, tmp_velocity, tmp_accel, dt, ions.cell().volume(), tmp_current);
		ham.uniform_vector_potential()+=tmp_induced;

		if(done) break;
	}
	
	ham.uniform_vector_potential()=orig_vp+orig_induced;
	ham.vp_induced=orig_induced;
	ham.vp_velocity=orig_velocity;
	ham.vp_accel=orig_accel;
}

}
}
#endif

#ifdef INQ_REAL_TIME_ETRS_UNIT_TEST
#undef INQ_REAL_TIME_ETRS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
