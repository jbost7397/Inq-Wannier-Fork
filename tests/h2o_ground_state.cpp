/* -*- indent-tabs-mode: t -*- */

/*
 Copyright (C) 2019-2021 Xavier Andrade, Alfredo A. Correa

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

#include <fftw3.h>

#include <systems/ions.hpp>
#include <systems/electrons.hpp>
#include <config/path.hpp>
#include <input/parse_xyz.hpp>
#include <utils/match.hpp>
#include <operations/io.hpp>
#include <perturbations/kick.hpp>
#include <ground_state/initial_guess.hpp>
#include <ground_state/calculate.hpp>

#include <input/environment.hpp>

#include <utils/profiling.hpp>

int main(int argc, char ** argv){

	using namespace inq::input;
	using namespace inq::systems;
	using namespace inq::magnitude;
	using inq::vector3;
	
	inq::input::environment env(argc, argv);
	
	inq::utils::match match(3.0e-4);

	inq::systems::box box = box::orthorhombic(12.0_b, 11.0_b, 10.0_b).finite().cutoff_energy(30.0_Ha);
	
	inq::systems::ions ions(box);

	ions.insert(inq::input::parse_xyz(inq::config::path::unit_tests_data() + "water.xyz"));
	
	auto comm = boost::mpi3::environment::get_world_instance();
	auto parstates = comm.size();
	if(comm.size() == 3 or comm.size() == 5) parstates = 1;
	
	inq::systems::electrons electrons(env.par().states(parstates), ions, box);

	inq::ground_state::initial_guess(ions, electrons);

	auto scf_options = scf::energy_tolerance(1.0e-9_Ha) | scf::broyden_mixing();
	auto result = inq::ground_state::calculate(ions, electrons, interaction::lda(), scf_options);
	
	match.check("total energy",        result.energy.total(),       -17.604152928274);
	match.check("kinetic energy",      result.energy.kinetic(),      12.055671278976);
	match.check("eigenvalues",         result.energy.eigenvalues(),  -4.066524514529);
	match.check("Hartree energy",      result.energy.hartree(),      21.255096093096);
	match.check("external energy",     result.energy.external(),    -50.405054484947);
	match.check("non-local energy",    result.energy.nonlocal(),     -2.732683697594);
	match.check("XC energy",           result.energy.xc(),           -4.762305356613);
	match.check("XC density integral", result.energy.nvxc(),         -5.494649797157);
	match.check("HF exchange energy",  result.energy.hf_exchange(),   0.000000000000);
	match.check("ion-ion energy",      result.energy.ion(),           6.985123238808);

	std::cout << result.dipole << std::endl;
	
	match.check("dipole x", result.dipole[0], -0.000304523);
	match.check("dipole y", result.dipole[1], -0.724304);
	match.check("dipole z", result.dipole[2], -2.78695e-05);
	
	electrons.save("h2o_restart");
	
	fftw_cleanup(); //required for valgrid
	
	return match.fail();

}

