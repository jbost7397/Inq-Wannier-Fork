/* -*- indent-tabs-mode: t; tab-width: 2 -*- */

/*
 Copyright (C) 2019 Xavier Andrade, Alfredo Correa.

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

#include <systems/electrons.hpp>
#include <systems/ions.hpp>
#include <multi/array.hpp>
#include <multi/adaptors/fftw.hpp>
#include <parser/input_file.hpp>

#include <iostream>

using std::cout;

int main(int argc, char ** argv){

  parser::input_file input(argv[1]);
  
  auto coordinates_file = input.parse<std::string>("Coordinates");
  
  ions::geometry geo(coordinates_file);
  
  auto lx = input.parse<double>("Lx");	
  auto ly = input.parse<double>("Ly");
  auto lz = input.parse<double>("Lz");

	systems::ions ions(input::cell::cubic(lx, ly, lz), geo);
	
  auto ecut = input.parse<double>("CutoffEnergy");
  
	systems::electrons electrons(ions, input::basis::cutoff_energy(ecut));

	electrons.calculate_ground_state();

	std::cout << "Total energy = " << electrons.calculate_energy() << std::endl;
	
}
