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
	
}
#endif
