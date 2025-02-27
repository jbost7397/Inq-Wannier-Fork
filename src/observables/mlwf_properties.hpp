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
  mlwf_properties() : mlwf_transformer_(std::nullopt) {}

  void set_mlwf_transformer(const wannier::tdmlwf_trans& transformer) {
    mlwf_transformer_ = transformer;
  }

  explicit mlwf_properties(const wannier::tdmlwf_trans& mlwf_transformer)
      : mlwf_transformer_(mlwf_transformer) {}

  void calculate(std::ofstream& output_file, int time_step, states::orbital_set<basis::real_space, complex>& phi) {
    mlwf_transformer_->update(phi);
    mlwf_transformer_->compute_transform();
    mlwf_transformer_->apply_transform(phi);

    if(phi.basis().comm().rank() == 0){
    	output_file << "Time step: " << time_step + 1 << "\n";
    	output_file << "MLWFs:\n";
    	for (int i = 0; i < phi.set_size(); ++i) {
       		auto center = mlwf_transformer_->center(i, phi.basis().cell());
       		auto spread = mlwf_transformer_->spread(i, phi.basis().cell());
       		output_file << "  WF " << i + 1 << ": " << center[0] << "     " << center[1] << "     " << center[2] << "     Spread: " << spread << std::endl;
    	}
    }
  }

private:

  std::optional<wannier::tdmlwf_trans> mlwf_transformer_;
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
