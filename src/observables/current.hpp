/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OBSERVABLES__CURRENT
#define INQ__OBSERVABLES__CURRENT

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa, Yifan Yao
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <math/vector3.hpp>
#include <basis/real_space.hpp>
#include <basis/field.hpp>
#include <operations/gradient.hpp>
#include <operations/sum.hpp>
#include <matrix/gather_scatter.hpp>
#include <systems/ions.hpp>
#include <systems/electrons.hpp>
#include <physics/constants.hpp>

#include <cassert>
#include <mpi.h>
#include <tuple>

namespace inq {
namespace observables {

template <typename HamiltonianType>
basis::field<basis::real_space, vector3<double, covariant>> current_density(const systems::ions & ions, systems::electrons const & electrons, HamiltonianType const & ham){

	basis::field<basis::real_space, vector3<double, covariant>> cdensity(electrons.density_basis());
	cdensity.fill(vector3<double, covariant>{0.0, 0.0, 0.0});

	auto iphi = 0;
	for(auto & phi : electrons.kpin()){
		
		auto gphi = operations::gradient(phi, /* factor = */ -1.0, /* shift = */ phi.kpoint() + ham.uniform_vector_potential());

		ham.projectors_all().position_commutator(phi, gphi, phi.kpoint() + ham.uniform_vector_potential());
		
    gpu::run(phi.basis().part().local_size(),
             [nst = phi.set_part().local_size(), occ = begin(electrons.occupations()[iphi]),
              ph = begin(phi.matrix()), gph = begin(gphi.matrix()), cdens = begin(cdensity.linear())] GPU_LAMBDA (auto ip){
               for(int ist = 0; ist < nst; ist++) cdens[ip] += 0.5*occ[ist]*imag(conj(ph[ip][ist])*gph[ip][ist] - conj(gph[ip][ist])*ph[ip][ist]);
             });
		iphi++;
	}
  
	cdensity.all_reduce(electrons.kpin_states_comm());
	return cdensity;
}

template <typename HamiltonianType>
auto current(const systems::ions & ions, systems::electrons const & electrons, HamiltonianType const & ham){
  return operations::integral(current_density(ions, electrons, ham));
}

template <typename HamiltonianType>
std::tuple<basis::field<basis::real_space, vector3<double, covariant>>, basis::field<basis::real_space, vector3<double, covariant>>> shift_ballistic_current_density( systems::electrons const & ground_electrons, systems::electrons const & electrons, HamiltonianType const & ham, int band_start, int band_end){

  int nbands = band_end - band_start +1;
  int nkpin_loc = electrons.kpin_size();
  int npoints_loc = electrons.density_basis().local_size();

  assert(electrons.kpin_size() == ground_electrons.kpin_size() );
  assert(electrons.density_basis().local_size() == ground_electrons.density_basis().local_size() );

  basis::field<basis::real_space, vector3<double, covariant>> bcdensity(electrons.density_basis());
  bcdensity.fill(vector3<double, covariant>{0.0, 0.0, 0.0});

  basis::field<basis::real_space, vector3<double, covariant>> scdensity(electrons.density_basis());
  scdensity.fill(vector3<double, covariant>{0.0, 0.0, 0.0});


  for (int ikpin=0; ikpin<nkpin_loc; ikpin++){
   
    //calculate gradient phi
    auto phi = electrons.kpin()[ikpin];
    auto phi0 = ground_electrons.kpin()[ikpin];
    auto gphi = operations::gradient(phi0,  -1.0,   phi0.kpoint() + ham.uniform_vector_potential());
    ham.projectors_all().position_commutator(phi0, gphi, phi0.kpoint() + ham.uniform_vector_potential());

    decltype(phi) gphi_x(phi);
    decltype(phi) gphi_y(phi);
    decltype(phi) gphi_z(phi);

    gphi_x.fill(0.0);
    gphi_y.fill(0.0);
    gphi_z.fill(0.0);

   for(int jp=0; jp < std::get<0>(gphi.matrix().sizes()); jp++){
     for(int jst=0; jst < std::get<1>(gphi.matrix().sizes()); jst++){
       gphi_x.matrix()[jp][jst] = gphi.matrix()[jp][jst][0];
       gphi_y.matrix()[jp][jst] = gphi.matrix()[jp][jst][1];
       gphi_z.matrix()[jp][jst] = gphi.matrix()[jp][jst][2];
     }
   }

    //get phi from other procs
    gpu::array<long, 1> point_list0(phi0.basis().local_size());
    {
    long ip=0;

    for(int ix = 0; ix < phi0.basis().local_sizes()[0]; ix++){
      for(int iy = 0; iy < phi0.basis().local_sizes()[1]; iy++){
        for(int iz = 0; iz < phi0.basis().local_sizes()[2]; iz++){	

          auto ixg = phi0.basis().cubic_part(0).local_to_global(ix);
          auto iyg = phi0.basis().cubic_part(1).local_to_global(iy);
          auto izg = phi0.basis().cubic_part(2).local_to_global(iz);						

          auto ii = phi0.basis().to_symmetric_range(ixg, iyg, izg);
          auto isource = phi0.basis().from_symmetric_range(ii);

          point_list0[ip] = phi0.basis().linear_index(isource[0], isource[1], isource[2]);
          ip++;
        }
      }
    }
    }

    gpu::array<long, 1> point_list(phi.basis().local_size());
    {
    long ip=0;

    for(int ix = 0; ix < phi.basis().local_sizes()[0]; ix++){
      for(int iy = 0; iy < phi.basis().local_sizes()[1]; iy++){
        for(int iz = 0; iz < phi.basis().local_sizes()[2]; iz++){	

          auto ixg = phi.basis().cubic_part(0).local_to_global(ix);
          auto iyg = phi.basis().cubic_part(1).local_to_global(iy);
          auto izg = phi.basis().cubic_part(2).local_to_global(iz);						

          auto ii = phi.basis().to_symmetric_range(ixg, iyg, izg);
          auto isource = phi.basis().from_symmetric_range(ii);

          point_list[ip] = phi.basis().linear_index(isource[0], isource[1], isource[2]);
          ip++;
        }
      }
    }
    }

    auto olap = matrix::all_gather(operations::overlap(phi,phi0));

    gpu::array<long, 1> state_list(nbands);
    for(long ib=0; ib<nbands; ib++) state_list[ib] = (band_start-1)+ib;

    auto remphi = parallel::get_remote_points(phi0, point_list0, state_list);
    auto remgphi_x = parallel::get_remote_points(gphi_x, point_list, state_list);
    auto remgphi_y = parallel::get_remote_points(gphi_y, point_list, state_list);
    auto remgphi_z = parallel::get_remote_points(gphi_z, point_list, state_list);

    for (int ib1=0; ib1<nbands; ib1++){
      for (int ib2=0; ib2<nbands; ib2++){

        gpu::array<vector3<complex, covariant>,1> phigradphi(npoints_loc);

        
        for (int ip=0; ip<npoints_loc; ip++){
          vector3<complex,covariant> gg = {remgphi_x[ip][ib2], remgphi_y[ip][ib2], remgphi_z[ip][ib2]};
          phigradphi[ip] = conj(remphi[ip][ib1]) * gg;
        }

        if (ib1==ib2){//ballistic current
          gpu::run( npoints_loc,
              [nst = phi.set_part().local_size(),  ib1, ib2, ikpin,
              b_part = phi.set_part(),
              occ = begin(electrons.occupations()[ikpin]),
              ol = begin(olap),
              pgp = begin(phigradphi),
              bcdens = begin(bcdensity.linear())]
              GPU_LAMBDA (auto ip){
              for(int ist = 0; ist < nst; ist++){
              auto ist_global = b_part.local_to_global(ist).value();
              auto value = occ[ist]* conj(ol[ib1][ist_global]) * ol[ib2][ist_global] * pgp[ip];
              bcdens[ip] += imag(value);

              }  

              });
        }
        else{//shift current
          gpu::run( npoints_loc,
              [nst = phi.set_part().local_size(),  ib1, ib2, ikpin,
              b_part = phi.set_part(),
              occ = begin(electrons.occupations()[ikpin]),
              ol = begin(olap),
              pgp = begin(phigradphi),
              scdens = begin(scdensity.linear())]
              GPU_LAMBDA (auto ip){
              for(int ist = 0; ist < nst; ist++){
              auto ist_global = b_part.local_to_global(ist).value();
              auto value = occ[ist]* conj(ol[ib1][ist_global]) * ol[ib2][ist_global] * pgp[ip];
              scdens[ip] += imag(value);
              }  

              });
        }

      }
    }
  }


  scdensity.all_reduce(electrons.kpin_states_comm());
  bcdensity.all_reduce(electrons.kpin_states_comm());
  return std::make_tuple(scdensity,bcdensity);
}

template <typename HamiltonianType>
auto shift_ballistic_current(systems::electrons const & ground_electrons, systems::electrons const & electrons, HamiltonianType const & ham, int band_start, int band_end){

  auto sc_bc = shift_ballistic_current_density(ground_electrons, electrons, ham, band_start, band_end);
  auto scdensity = std::get<0>(sc_bc);
  auto bcdensity = std::get<1>(sc_bc);
  return std::make_tuple(operations::integral(scdensity), operations::integral(bcdensity));
}






}
}
#endif

#ifdef INQ_OBSERVABLES_CURRENT_UNIT_TEST
#undef INQ_OBSERVABLES_CURRENT_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;
		
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	auto par = input::parallelization(comm);

	{
		systems::ions ions(systems::cell::orthorhombic(6.0_b, 10.0_b, 6.0_b));
		systems::electrons electrons(par, ions, options::electrons{}.cutoff(15.0_Ha).extra_electrons(20.0));
		ground_state::initial_guess(ions, electrons);
		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions, 0.0, /* use_ace = */ true);
		
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
		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions, 0.0, /* use_ace = */ true);
		
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
		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions, 0.0, /* use_ace = */ true);
		
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
		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions, 0.0, /* use_ace = */ true);
		
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
	}

	
}
#endif
