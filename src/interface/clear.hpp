/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__CLEAR
#define INQ__INTERFACE__CLEAR

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>

namespace inq {
namespace interface {

struct {

	std::string name() const {
		return "clear";
	}

	std::string one_line() const {
		return "Removes any inq information from the current directory.";
	}
	
	void operator()() const {
		if(input::environment::global().comm().root()) {
			std::filesystem::remove_all(".default_ions");
			std::filesystem::remove_all(".default_theory");
			std::filesystem::remove_all(".default_electrons_options");
			std::filesystem::remove_all(".default_orbitals");
		}
		input::environment::global().comm().barrier();
	}

	template <typename ArgsType>
	void command(ArgsType const & args, bool) const {
		if(args.size() != 0) {
			std::cerr << "Error: The 'clear' command doesn't take arguments." << std::endl;
			exit(1);
		}
		operator()();
		exit(0);
	}
	
}	const clear;

}
}
#endif

#ifdef INQ_INTERFACE_CLEAR_UNIT_TEST
#undef INQ_INTERFACE_CLEAR_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
