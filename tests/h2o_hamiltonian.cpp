/* -*- indent-tabs-mode: t -*- */

/*
 Copyright (C) 2022 Xavier Andrade

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.
  
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
  
 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <inq/inq.hpp>

//
//
// THIS IS AN EXAMPLE HOW YOU CAN APPLY A HAMILTONIAN USING INQ.  IT
// LOADS THE RESULTS FROM THE h2o_ground_state.cpp FILE.
//
//


int main(int argc, char ** argv){
	using namespace inq;
	using namespace inq::magnitude;	
  using math::vector3;

	input::environment env(argc, argv);

	inq::systems::box box = systems::box::orthorhombic(12.0_b, 11.0_b, 10.0_b).finite().cutoff_energy(30.0_Ha);
	
	inq::systems::ions ions(box);

	ions.insert(inq::input::parse_xyz(inq::config::path::unit_tests_data() + "water.xyz"));
	
	inq::systems::electrons electrons(env.par(), ions, box);

	auto found_gs = electrons.load("h2o_restart");

	assert(found_gs);

	auto inter = input::interaction::dft();
		
	hamiltonian::ks_hamiltonian<basis::real_space> ham(electrons.states_basis_, ions.cell(), electrons.atomic_pot_, /*Pseudos in fourier space */ false, ions.geo(),
																										 electrons.states_.num_states(), inter.exchange_coefficient(), electrons.states_basis_comm_);

	hamiltonian::self_consistency sc(inter, electrons.states_basis_, electrons.density_basis_);

	electrons.density_ = density::calculate(electrons);

	hamiltonian::energy energy;
	ham.scalar_potential = sc.ks_potential(electrons.density_, energy);
		
	for(auto & phi : electrons.lot()) {
		auto hphi = ham(phi);
	}
	
}
