/* -*- indent-tabs-mode: t -*- */
#ifndef INQ__CONFIG__PATH
#define INQ__CONFIG__PATH

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>
#include <string>

namespace inq {
namespace config {

struct path {
  static std::string share(){ return SHARE_DIR + std::string("/") ; }
  static std::string unit_tests_data(){ return share() + std::string("unit_tests_data/"); }
  static std::string pseudo(){ return SHARE_DIR + std::string("/../pseudopod/pseudopotentials/quantum-simulation.org/sg15/");}
};

}
}
#endif

#ifdef INQ_CONFIG_PATH_UNIT_TEST
#undef INQ_CONFIG_PATH_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
  SECTION("Share path"){
    CHECK(inq::config::path::share() == SHARE_DIR + std::string("/"));
  }
}

#endif
