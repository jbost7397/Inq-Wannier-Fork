/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__RESULTS_GROUND_STATE
#define INQ__INTERFACE__RESULTS_GROUND_STATE

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <ground_state/results.hpp>

namespace inq {
namespace interface {

struct {		

	constexpr auto name() const {
		return "results ground-state";
	}

	constexpr auto one_line() const {
		return "Get information about the results obtained from a ground-state calculation";
	}

	constexpr auto help() const {
		
		return R""""(

The 'results ground-state' command
==================

This command queries the results obtained from a ground-state
calculation. Without arguments, it prints the values calculated.

The options allows you to query a specific value. In this case only
the value will be printed without any other text, so it is suitable
for easy parsing in scripting. The values are returned in atomic
units.

These are the available subcommands:

- `results ground-state`

  When no arguments are given, print the values calculated.

  Example: `inq results ground-state`.


- `results ground-state iterations`

  Print the number of self-consistency iterations done.

  Example: `inq results ground-state iterations`

- `results ground-state magnetization [index]`

  Print the magnetization. If no additional arguments are given, print
  the whole vector. Optionally you can add an index argument to print
  a specific component of the vector. The index can be given as a
  numerical value between _1_ and _3_, or as the letters _x_, _y_ or
  _z_.

  Note that for spin unpolarized systems the magnetization is always
  zero. For spin polarized the magnetization is assumed on the _z_
  direction.

  Example: `inq results ground-state magnetization z`


- `results ground-state dipole [index]`

  Print the dipole of the system in atomic units. The vector is
  printed if no additional arguments are given. You can also add an
  index argument to print a specific component. The index can be given
  as a numerical value between _1_ and _3_, or as the letters _x_, _y_
  or _z_.

  Note that the dipole is only calculated for the non-periodic
  directions. For the periodic directions is set to zero since the
  dipole is not properly defined.

  Example: `inq results ground-state magnetization z`


- `results ground-state energy`

  When no arguments are given, `energy` will print all the energy values available.

  Example: `inq results ground-state energy`.


- `results ground-state energy total`

  Returns the total energy of the calculation. This includes the ionic
  contribution.

  Example: `inq results ground-state energy total`.


- `results ground-state energy kinetic`

  The electronic kinetic energy.

  Example: `inq results ground-state energy kinetic`.


- `results ground-state energy eigenvalues`

  The sum of the eigenvalues, weighed by the occupations.

  Example: `inq results ground-state energy eigenvalues`.


- `results ground-state energy Hartree`

  The classical electrostatic interaction energy between electrons.

  Example: `inq results ground-state energy Hartree`.


- `results ground-state energy external`

  The energy of the interaction of the electrons with the local
  potential generated by the ions. This doesn't include the non-local
  pseudopotential part.

  Example: `inq results ground-state energy external`.


- `results ground-state energy non-local`

  The energy of the interaction of the electrons with the non-local
  part of the ionic pseudo-potentials.

  Example: `inq results ground-state energy non-local`.


- `results ground-state energy xc`

  The exchange and correlation energy from DFT semi-local
  functionals. It doesn't include the contribution from Hartree-Fock
  exchange (see `energy exact_exchange`).

  Example: `inq results ground-state energy xc`.


- `results ground-state energy nvxc`

  The energy of the interaction of the exchange and correlation
  potential and the density. This is different from the exchange and
  correlation energy.

  Example: `inq results ground-state energy nvxc`.


- `results ground-state energy exact-exchange`

  The Hartree-Fock exact-exchange energy. This is calculated for
  Hartree-Fock and hybrid functionals.

  Example: `inq results ground-state energy exact-exchange`.


- `results ground-state energy ion`

  The ion-ion interaction energy. This value is calculated taking into
  account the periodicity of the system.

  Example: `inq results ground-state energy ion`.


)"""";
	}

	void operator()() const {
		auto res = ground_state::results::load(".inq/default_results_ground_state");
		if(input::environment::global().comm().root()) std::cout << res;
	}

	auto iterations() const {
		auto res = ground_state::results::load(".inq/default_results_ground_state");
		return res.total_iter;
	}

	auto magnetization() const {
		auto res = ground_state::results::load(".inq/default_results_ground_state");
		return res.magnetization;
	}

	auto dipole() const {
		auto res = ground_state::results::load(".inq/default_results_ground_state");
		return res.dipole;
	}
	
	void energy() const {
		auto ener = ground_state::results::load(".inq/default_results_ground_state").energy;
		if(input::environment::global().comm().root()) std::cout << ener;
	}
	
  double energy_total() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.total();
  }
	
  double energy_kinetic() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.kinetic();
  }

  double energy_eigenvalues() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.eigenvalues();
  }

  double energy_external() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.external();
  }
  
  double energy_non_local() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.non_local();
  }
  
  double energy_hartree() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.hartree();
  }
  
  double energy_xc() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.xc();
  }

  double energy_nvxc() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.nvxc();
  }

  double energy_exact_exchange() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.exact_exchange();
  }
  
  double energy_ion() const{
    return ground_state::results::load(".inq/default_results_ground_state").energy.ion();
  }

	auto forces() const {
    return ground_state::results::load(".inq/default_results_ground_state").forces;
	}

	template <typename ArgsType>
	void command(ArgsType args, bool quiet) const {

		if(args.size() == 0){
			operator()();
			actions::normal_exit();
		}
		
		if(args.size() == 1 and args[0] == "iterations"){
			std::cout << iterations() << std::endl;
			actions::normal_exit();
		}

		if(args.size() == 1 and args[0] == "magnetization"){
			std::cout << magnetization() << std::endl;
			actions::normal_exit();
		}

		if(args.size() == 2 and args[0] == "magnetization"){
			auto idir = utils::str_to_index(args[1]);
			if(idir == -1) actions::error(input::environment::global().comm(), "Invalid index in the 'results ground-state magnetization' command");
			if(input::environment::global().comm().root())  printf("%.6f\n", magnetization()[idir]);
			actions::normal_exit();
		}
		
		if(args.size() == 1 and args[0] == "dipole"){
			std::cout << dipole() << std::endl;
			actions::normal_exit();
		}

		if(args.size() == 2 and args[0] == "dipole"){
			auto idir = utils::str_to_index(args[1]);
			if(idir == -1) actions::error(input::environment::global().comm(), "Invalid index in the 'results ground-state dipole' command");
			if(input::environment::global().comm().root())  printf("%.6f\n", dipole()[idir]);
			actions::normal_exit();
		}
		
		if(args[0] == "energy"){

			args.erase(args.begin());

			if(args.size() == 0) {
				energy();
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "total"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_total());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "kinetic"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_kinetic());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "eigenvalues"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_eigenvalues());
				actions::normal_exit();
			}
    
			if(args.size() == 1 and args[0] == "external"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_external());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "non-local"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_non_local());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "hartree"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_hartree());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "xc"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_xc());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "nvxc"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_nvxc());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "exact-exchange"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_exact_exchange());
				actions::normal_exit();
			}
        
			if(args.size() == 1 and args[0] == "ion"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_ion());
				actions::normal_exit();
			}
		}
		
		if(args[0] == "forces"){
			auto forces_array = forces();

			if(args.size() == 1) {

				if(input::environment::global().comm().root()) {
					for(auto & force : forces_array) printf("%.20e\t%.20e\t%.20e\n", force[0], force[1], force[2]);
				}
				actions::normal_exit();
					
			} else if (args.size() == 2 or args.size() == 3) {
				auto index = utils::str_to<long>(args[1]);
				if(index < 0 or index >= forces_array.size()) actions::error(input::environment::global().comm(), "Invalid index ", index, " in the 'results ground-state forces' command");

				if(args.size() == 2) {
					if(input::environment::global().comm().root()) printf("%.20e\t%.20e\t%.20e\n", forces_array[index][0], forces_array[index][1], forces_array[index][2]);
					actions::normal_exit();
				}
					
				auto idir = utils::str_to_index(args[2]);
				if(idir == -1) actions::error(input::environment::global().comm(), "Invalid coordinate index in the 'results ground-state forces' command");
				if(input::environment::global().comm().root()) printf("%.20e\n", forces_array[index][idir]);
				actions::normal_exit();
			}
		}
		
		actions::error(input::environment::global().comm(), "Invalid syntax in the 'results ground-state' command");    
	}
	
} const results_ground_state;

}
}
#endif

#ifdef INQ_INTERFACE_RESULTS_GROUND_STATE_UNIT_TEST
#undef INQ_INTERFACE_RESULTS_GROUND_STATE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
