/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INPUT__PARSE_XYZ
#define INQ__INPUT__PARSE_XYZ

/*
 Copyright (C) 2019-2022 Xavier Andrade, Alfredo A. Correa

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

#include <vector>
#include <cmath>

#include <math/vector3.hpp>

#include <pseudopod/element.hpp>
#include <input/atom.hpp>
#include <input/species.hpp>
#include <magnitude/length.hpp>

namespace inq {
namespace input {

auto parse_xyz(const std::string & xyz_file_name, quantity<magnitude::length> unit = magnitude::operator""_angstrom(1.0)){

	using namespace inq;
 
	std::vector<input::atom> geo;

	std::ifstream xyz_file(xyz_file_name.c_str());
	
	assert(xyz_file.is_open());

	int natoms;
	std::string comment_line;
  
	xyz_file >> natoms;
  
	std::getline(xyz_file, comment_line);
	std::getline(xyz_file, comment_line);
  
	std::string atom_name;
	vector3<double> atom_position;
  
	for(int iatom = 0; iatom < natoms; iatom++){
		xyz_file >> atom_name >> atom_position;
		geo.push_back(atom_name | atom_position*unit.in_atomic_units());
	}
  
	xyz_file.close();
  
	assert(unsigned(natoms) == geo.size());

	return geo;
}

template<class AtomsSequence, class Cell>
auto generate_xyz(
	AtomsSequence const& geo,
	Cell const& cell,
	std::ostream& os,
	quantity<magnitude::length> unit = magnitude::operator""_angstrom(1.0)
) -> std::ostream& {  // using https://open-babel.readthedocs.io/en/latest/FileFormats/Extended_XYZ_cartesian_coordinates_format.html
	os << geo.size() <<'\n';
	os << '\n';

	using std::begin; using std::end;
	int i = 0;
	for(auto it = begin(geo); it != end(geo); ++it) {
		os << it->species().symbol() <<' '<< it->position()/unit.in_atomic_units();
		os <<'\n';
		++i;
	}
	os <<'\n';
	os <<"Vector 1 "<< cell[0]/unit.in_atomic_units()<<'\n';
	os <<"Vector 2 "<< cell[1]/unit.in_atomic_units()<<'\n';
	os <<"Vector 3 "<< cell[2]/unit.in_atomic_units()<<'\n';
	os <<"Offset   "<< "0.000000    0.000000    0.000000\n";

	return os;
}

template<class AtomsSequence, class Cell>
void write_xyz(AtomsSequence const& geo, Cell const& cell, const std::string& xyz_file_name, quantity<magnitude::length> unit = magnitude::operator""_angstrom(1.0)) {
	std::ofstream xyz_file{xyz_file_name};
	generate_xyz(geo, cell, xyz_file);
	if(not xyz_file) {throw std::runtime_error{"cannot read xyz file "+ xyz_file_name};}
}

}
}
#endif

#ifdef INQ_INPUT_PARSE_XYZ_UNIT_TEST
#undef INQ_INPUT_PARSE_XYZ_UNIT_TEST

#include <catch2/catch_all.hpp>

#include <systems/box.hpp>
#include <magnitude/length.hpp>
#include <config/path.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

  auto geo = input::parse_xyz(config::path::unit_tests_data() + "benzene.xyz");

  CHECK(geo.size() == 12);

  CHECK(geo[2].species() == pseudo::element("C"));
  CHECK(geo[2].position()[0] == 2.2846788549_a);
  CHECK(geo[2].position()[1] == -1.3190288178_a);
  CHECK(geo[2].position()[2] == 0.0_a);

  CHECK(geo[11].species() == pseudo::element("H"));
  CHECK(geo[11].position()[0] == -4.0572419367_a);
  CHECK(geo[11].position()[1] == 2.343260364_a);
  CHECK(geo[11].position()[2] == 0.0_a);

  geo.push_back("Cl" | vector3<double>(-3.0, 4.0, 5.0));

  CHECK(geo.size() == 13);
  CHECK(geo[12].species() == pseudo::element("Cl"));
  CHECK(geo[12].position()[0] == -3.0_a);
  CHECK(geo[12].position()[1] == 4.0_a);
  CHECK(geo[12].position()[2] == 5.0_a);

	std::ostringstream oss;

	using namespace inq::magnitude;
	auto cell = inq::systems::box::orthorhombic(9.717_A, 11.22023_A, 5.172_A);

	generate_xyz(geo, cell, oss);
	CHECK( oss );
}
#endif
