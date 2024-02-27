/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__RESULTS_GROUND_STATE
#define INQ__INTERFACE__RESULTS_GROUND_STATE

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <ground_state/result.hpp>

namespace inq {
namespace interface {

struct {		

	std::string name() const {
		return "results ground-state";
	}

	std::string one_line() const {
		return "Get information about the results obtained from a ground-state calculation";
	}

	void help() const {
		
		std::cout << R""""(

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
		auto res = ground_state::result::load(".inq/default_results_ground_state");
		if(input::environment::global().comm().root()) std::cout << res;
	}

	auto iterations() const {
		auto res = ground_state::result::load(".inq/default_results_ground_state");
		return res.total_iter;
	}

	auto magnetization() const {
		auto res = ground_state::result::load(".inq/default_results_ground_state");
		return res.magnetization;
	}

	auto dipole() const {
		auto res = ground_state::result::load(".inq/default_results_ground_state");
		return res.dipole;
	}
	
	void energy() const {
		auto ener = ground_state::result::load(".inq/default_results_ground_state").energy;
		if(input::environment::global().comm().root()) std::cout << ener;
	}
	
  double energy_total() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.total();
  }
	
  double energy_kinetic() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.kinetic();
  }

  double energy_eigenvalues() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.eigenvalues();
  }

  double energy_external() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.external();
  }
  
  double energy_non_local() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.non_local();
  }
  
  double energy_hartree() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.hartree();
  }
  
  double energy_xc() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.xc();
  }

  double energy_nvxc() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.nvxc();
  }

  double energy_exact_exchange() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.exact_exchange();
  }
  
  double energy_ion() const{
    return ground_state::result::load(".inq/default_results_ground_state").energy.ion();
  }

	template <typename ArgsType>
	void command(ArgsType args, bool quiet) const {

		if(args.size() == 0){
			operator()();
			exit(0);
		}
		
		if(args.size() == 1 and args[0] == "iterations"){
			std::cout << iterations() << std::endl;
			exit(0);
		}

		if(args.size() == 1 and args[0] == "magnetization"){
			std::cout << magnetization() << std::endl;
			exit(0);
		}

		if(args.size() == 2 and args[0] == "magnetization"){
			auto idir = utils::str_to_index(args[1]);

			if(idir == -1) {
				if(input::environment::global().comm().root()) std::cerr << "Error: Invalid index in the 'results ground-state magnetization' command" << std::endl;
				exit(1);
			}

			if(input::environment::global().comm().root())  printf("%.6f\n", magnetization()[idir]);
			exit(0);
		}
		
		if(args.size() == 1 and args[0] == "dipole"){
			std::cout << dipole() << std::endl;
			exit(0);
		}

		if(args.size() == 2 and args[0] == "dipole"){
			auto idir = utils::str_to_index(args[1]);

			if(idir == -1) {
				if(input::environment::global().comm().root()) std::cerr << "Error: Invalid index in the 'results ground-state dipole' command" << std::endl;
				exit(1);
			}

			if(input::environment::global().comm().root())  printf("%.6f\n", dipole()[idir]);
			exit(0);
		}
		
		if(args[0] == "energy"){

			args.erase(args.begin());

			if(args.size() == 0) {
				energy();
				exit(0);
			}

			if(args.size() == 1 and args[0] == "total"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_total());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "kinetic"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_kinetic());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "eigenvalues"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_eigenvalues());
				exit(0);
			}
    
			if(args.size() == 1 and args[0] == "external"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_external());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "non-local"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_non_local());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "hartree"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_hartree());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "xc"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_xc());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "nvxc"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_nvxc());
				exit(0);
			}

			if(args.size() == 1 and args[0] == "exact-exchange"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_exact_exchange());
				exit(0);
			}
        
			if(args.size() == 1 and args[0] == "ion"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_ion());
				exit(0);
			}
		}
      
		if(input::environment::global().comm().root()) std::cerr << "Error: Invalid syntax in the 'results ground-state' command" << std::endl;
		exit(1);
    
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