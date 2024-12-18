/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OBSERVABLES__MLWF_PROPERTIES
#define INQ__OBSERVABLES__MLWF_PROPERTIES

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "wannier/tdmlwf_trans.hpp"

namespace inq {
namespace observables {

class mlwf_properties {
public:
  explicit mlwf_properties(const states::orbital_set<basis::real_space, complex>& wavefunctions)
      : wavefunctions_(wavefunctions) {}

  void calculate(std::ofstream& output_file, int time_step) {
    wannier::tdmlwf_trans mlwf_transformer(wavefunctions_);
    mlwf_transformer.update();
    mlwf_transformer.compute_transform();

    output_file << "Time step: " << time_step << "\n"; 
    output_file << "MLWF Centers:\n";
    for (int i = 0; i < mlwf_transformer.get_wavefunctions().set_size(); ++i) {
       auto center = mlwf_transformer.center(i, mlwf_transformer.get_wavefunctions().basis().cell());
       output_file << "  WF " << i << ": " << center << std::endl;
    }

    output_file << "\nMLWF Spreads:\n";
    for (int i = 0; i < mlwf_transformer.get_wavefunctions().set_size(); ++i) {
       auto spread = mlwf_transformer.spread(i, mlwf_transformer.get_wavefunctions().basis().cell());
       output_file << "  WF " << i << ": " << spread << "\n" << std::endl;
    }

  }

private:
  const states::orbital_set<basis::real_space, complex>& wavefunctions_;
};
} // namespace observables
} // namespace inq

#endif

#ifdef INQ_OBSERVABLES_MLWF_PROPERTIES_UNIT_TEST
#undef INQ_OBSERVABLES_MLWF_PROPERTIES_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;
		
	/*parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	auto par = input::parallelization(comm);

	{
		systems::ions ions(systems::cell::orthorhombic(6.0_b, 10.0_b, 6.0_b));
		systems::electrons electrons(par, ions, options::electrons{}.cutoff(15.0_Ha).extra_electrons(20.0));
		ground_state::initial_guess(ions, electrons);
		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions, 0.0, true);
		
		SECTION("Gamma - no atoms"){

			ham.uniform_vector_potential() = vector3<double, covariant>{0.0, 0.0, 0.0};
			
			auto cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] == 62.2452955214_a);
			CHECK(cur[1] == -1.0723045428_a);
			CHECK(cur[2] == 55.1882949624_a);
			
			ham.uniform_vector_potential() = vector3<double, covariant>{1.0, -2.0, 3.0};

			cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] ==  42.2452955214_a);
			CHECK(cur[1] ==  38.9276954572_a);
			CHECK(cur[2] ==  -4.8117050376_a);
			
		}

		SECTION("Gamma - no atoms - zero paramagnetic"){
			
			for(auto & phi : electrons.kpin()) phi.fill(1.0);
			
			auto charge = operations::integral(electrons.density());
			
			CHECK(charge == 20.0_a);
						
			ham.uniform_vector_potential() = vector3<double, covariant>{0.0, 0.0, 0.0};

			auto cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] == Approx(0.0).margin(1e-12));
			CHECK(cur[1] == Approx(0.0).margin(1e-12));
			CHECK(cur[2] == Approx(0.0).margin(1e-12));
			
			ham.uniform_vector_potential() = vector3<double, covariant>{1.0, -2.0, 3.0};

			cur = observables::current(ions, electrons, ham);
			auto cur_an = -charge*ham.uniform_vector_potential()*ions.cell().volume();

			CHECK(cur[0] == Approx(cur_an[0]));
			CHECK(cur[1] == Approx(cur_an[1]));
			CHECK(cur[2] == Approx(cur_an[2]));						
			
		}
	}

	{
		systems::ions ions(systems::cell::orthorhombic(6.0_b, 10.0_b, 6.0_b));
		systems::electrons electrons(par, ions, options::electrons{}.cutoff(15.0_Ha).extra_electrons(20.0), input::kpoints::point(0.25, 0.25, 0.25));
		ground_state::initial_guess(ions, electrons);
		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions, 0.0,  true);
		
		SECTION("1/4 1/4 1/4 - no atoms"){
			
			auto cur = observables::current(ions, electrons, ham);

			ham.uniform_vector_potential() = vector3<double, covariant>{0.0, 0.0, 0.0};
			
			CHECK(cur[0] ==  30.8293689855_a);
			CHECK(cur[1] == -32.4882310787_a);
			CHECK(cur[2] ==  23.7723684265_a);
			
			ham.uniform_vector_potential() = vector3<double, covariant>{1.0, -2.0, 3.0};

			cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] ==  10.8293689855_a);
			CHECK(cur[1] ==   7.5117689213_a);
			CHECK(cur[2] == -36.2276315735_a);
			
		}

		SECTION("1/4 1/4 1/4 - no atoms - zero paramagnetic"){
			
			for(auto & phi : electrons.kpin()) phi.fill(1.0);
			
			auto charge = operations::integral(electrons.density());
			
			CHECK(charge == 20.0_a);

			ham.uniform_vector_potential() = vector3<double, covariant>{0.0, 0.0, 0.0};
			
			auto cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] == Approx(-charge*0.25*2*M_PI*ions.cell().volume()));
			CHECK(cur[1] == Approx(-charge*0.25*2*M_PI*ions.cell().volume()));
			CHECK(cur[2] == Approx(-charge*0.25*2*M_PI*ions.cell().volume()));
			
			ham.uniform_vector_potential() = vector3<double, covariant>{1.0, -2.0, 3.0};

			cur = observables::current(ions, electrons, ham);
			auto cur_an = -charge*(vector3<double, covariant>{0.25, 0.25, 0.25}*2*M_PI + ham.uniform_vector_potential())*ions.cell().volume();

			CHECK(cur[0] == Approx(cur_an[0]));
			CHECK(cur[1] == Approx(cur_an[1]));
			CHECK(cur[2] == Approx(cur_an[2]));
			
		}
	}

	{
		systems::ions ions(systems::cell::orthorhombic(6.0_b, 10.0_b, 6.0_b));
		ions.insert("Cu", {0.0_b, 0.0_b, 0.0_b});
		ions.insert("Ag", {2.0_b, 0.7_b, 0.0_b});	
		systems::electrons electrons(par, ions, options::electrons{}.cutoff(15.0_Ha));
		ground_state::initial_guess(ions, electrons);
		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions, 0.0, true);
		
		SECTION("Gamma - atoms"){

			ham.uniform_vector_potential() = vector3<double, covariant>{0.0, 0.0, 0.0};
			
			auto cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] == 125.4365980468_a);
			CHECK(cur[1] ==  5.708830041_a);
			CHECK(cur[2] == 116.8317675488_a);
			
			ham.uniform_vector_potential() = vector3<double, covariant>{1.0, -2.0, 3.0};

			cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] ==  87.3634379355_a);
			CHECK(cur[1] ==  81.8624161924_a);
			CHECK(cur[2] ==   2.7621022916_a);
			
		}

		SECTION("Gamma - atoms - zero paramagnetic"){
			
			for(auto & phi : electrons.kpin()) phi.fill(1.0);
			
			ham.uniform_vector_potential() = vector3<double, covariant>{0.0, 0.0, 0.0};

			auto cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] == Approx(0.0).margin(1e-12));
			CHECK(cur[1] == Approx(0.0).margin(1e-12));
			CHECK(cur[2] == Approx(0.0).margin(1e-12));
			
			ham.uniform_vector_potential() = vector3<double, covariant>{1.0, -2.0, 3.0};

			cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] == -13946.6008146884_a);
			CHECK(cur[1] ==  27893.1545901583_a);
			CHECK(cur[2] == -41839.3257665369_a);						
			
		}
	}

	{
		systems::ions ions(systems::cell::orthorhombic(6.0_b, 10.0_b, 6.0_b));
		ions.insert("Cu", {0.0_b, 0.0_b, 0.0_b});
		ions.insert("Ag", {2.0_b, 0.7_b, 0.0_b});	
		systems::electrons electrons(par, ions, options::electrons{}.cutoff(15.0_Ha), input::kpoints::point(0.25, 0.25, 0.25));
		ground_state::initial_guess(ions, electrons);
		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions, 0.0, true);
		
		SECTION("Gamma - atoms"){

			ham.uniform_vector_potential() = vector3<double, covariant>{0.0, 0.0, 0.0};
			
			auto cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] ==  65.7213690610_a);
			CHECK(cur[1] == -53.8899219217_a);
			CHECK(cur[2] ==  57.1274385904_a);
			
			ham.uniform_vector_potential() = vector3<double, covariant>{1.0, -2.0, 3.0};

			cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] ==  27.6768119377_a);
			CHECK(cur[1] ==  22.2401572043_a);
			CHECK(cur[2] == -56.9484342345_a);
			
		}

		SECTION("Gamma - atoms - zero paramagnetic"){
			
			for(auto & phi : electrons.kpin()) phi.fill(1.0);
			
			ham.uniform_vector_potential() = vector3<double, covariant>{0.0, 0.0, 0.0};

			auto cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] == -21918.1030676834_a);
			CHECK(cur[1] == -21918.0591750074_a);
			CHECK(cur[2] == -21917.8247537255_a);
			
			ham.uniform_vector_potential() = vector3<double, covariant>{1.0, -2.0, 3.0};

			cur = observables::current(ions, electrons, ham);

			CHECK(cur[0] == -35804.3953857591_a);
			CHECK(cur[1] ==   5977.6646234487_a);
			CHECK(cur[2] == -63658.5829715723_a);
			
		}
	}*/

	
}
#endif
