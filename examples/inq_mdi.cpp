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
#include <input/atom.hpp>
#include <utils/match.hpp>
#include <operations/io.hpp>
#include <perturbations/kick.hpp>
#include <ground_state/initial_guess.hpp>
#include <ground_state/calculate.hpp>

#include <input/environment.hpp>

#include <utils/profiling.hpp>

#include <mpi.h>
#include "mdi.h"

#include "inq_mdi.h"

using namespace inq::input;
using namespace inq::systems;
using namespace inq::magnitude;
using inq::math::vector3;

bool exit_signal = false;

/* MPI intra-communicator for all processes running this code */
MPI_Comm mpi_world_comm;
int my_rank;

/* Flag whether the current energy is valid */
bool energy_valid;

/* Values defining the system */
std::vector<atom> geo;
std::vector<double> cell(9, 0.0);
double total_energy = 0.0;
double kinetic_energy = 0.0;
std::vector<int> elements(3, 0);
std::vector<double> coords(3*elements.size(), 0.0);
std::vector<double> forces(3*elements.size(), 0.0);
std::vector<int> dimensions(3, 1);




void update_system(std::vector<double> &cell,
		   inq::systems::box *scf_box_ptr,
		   std::vector<int> &dimensions,
		   std::vector<int> &elements,
		   std::vector<double> &coords,
		   std::vector<atom> &geo) {
  double small = 1.0e-10;

  /* Update cell dimensions */
  if ( abs(cell[1]) > small ||
       abs(cell[2]) > small ||
       abs(cell[3]) > small ||
       abs(cell[5]) > small ||
       abs(cell[6]) > small ||
       abs(cell[7]) > small) {
    std::cout << "ERROR: INQ only supports orthorhombic cells." << std::endl;
    MPI_Abort(mpi_world_comm, 1);
  }
  if ( dimensions[0] != dimensions[1] ||
       dimensions[1] != dimensions[2] ||
       (dimensions[0] != 1 && dimensions[0] != 2) ) {
    std::cout << "ERROR: INQ only supports cells that are either non-periodic in all dimensions or periodic in all dimensions."  << std::endl;
    MPI_Abort(mpi_world_comm, 1);
  }
  inq::quantity<length> xlength = inq::quantity<length>::from_atomic_units(cell[0]);
  inq::quantity<length> ylength = inq::quantity<length>::from_atomic_units(cell[4]);
  inq::quantity<length> zlength = inq::quantity<length>::from_atomic_units(cell[8]);
  if ( dimensions[0] == 1 ) { // non-periodic case
    *scf_box_ptr = box::orthorhombic(xlength, ylength, zlength).finite().cutoff_energy(30.0_Ha);
  }
  else { // periodic case
    *scf_box_ptr = box::orthorhombic(xlength, ylength, zlength).periodic().cutoff_energy(30.0_Ha);
  }

  /* Update geometry */
  geo.clear();
  for (int64_t iatom=0; iatom < (int64_t)elements.size(); iatom++) {
    auto spec = species(pseudo::element( elements[iatom] ));
    geo.push_back( spec | vector3<double>( coords[3*iatom+0], coords[3*iatom+1], coords[3*iatom+2]) );
  }

}



void update_scf(bool *energy_valid_ptr,
		std::vector<atom> &geo,
		double *total_energy_ptr,
		double *kinetic_energy_ptr, 
		std::vector<double> &forces,
		inq::systems::box *scf_box_ptr) {

  /* Only rerun the SCF if the energy is not valid */
  if ( *energy_valid_ptr ) { return; }
  

  inq::systems::ions ions(*scf_box_ptr, geo);

  config conf;

  mpi_world_comm = MPI_COMM_WORLD;
  inq::systems::electrons electrons(boost::mpi3::grip(mpi_world_comm), ions, *scf_box_ptr, conf);

  //boost::mpi3::communicator comm_world = boost::mpi3::environment::get_world_instance();
  //inq::systems::electrons electrons(comm_world, ions, *scf_box_ptr, conf);

  inq::ground_state::initial_guess(ions, electrons);

  auto scf_options = scf::conjugate_gradient() | scf::energy_tolerance(1.0e-5_Ha) | scf::density_mixing() | scf::broyden_mixing() | scf::calculate_forces();

  auto result = inq::ground_state::calculate(ions, electrons, interaction::dft(), scf_options);

  /* NOTE: MIGHT NEED TO PERFORM UNIT CONVERSIONS HERE */

  *total_energy_ptr = result.energy.total();
  *kinetic_energy_ptr = result.energy.kinetic();
  for (int64_t iatom=0; iatom < (int64_t)result.forces.size(); iatom++) {
    forces[(3*iatom)+0] = result.forces[iatom][0];
    forces[(3*iatom)+1] = result.forces[iatom][1];
    forces[(3*iatom)+2] = result.forces[iatom][2];
  }

  *energy_valid_ptr = true;
}


int execute_command(const char *command, MDI_Comm mdi_comm, void* class_obj) {

  inq::systems::box scf_box = box::orthorhombic(12.0_b, 11.0_b, 10.0_b).finite().cutoff_energy(30.0_Ha);
 
  /* Respond to the <CELL command */
  if ( strcmp(command, "<CELL") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);

    std::vector<double> cell_send(9, 0.0);
    for (int ivec=0; ivec < 3; ivec++) {
      for (int idim=0; idim < 3; idim++) {
	cell_send[(3*ivec)+idim] = scf_box[ivec][idim];
      }
    }
    MDI_Send(&cell_send[0], 9, MDI_DOUBLE, mdi_comm);

  }

  /* Respond to the >CELL command */
  else if ( strcmp(command, ">CELL") == 0 ) {

    //std::vector<double> cell_recv(9, 0.0);
    MDI_Recv(&cell[0], 9, MDI_DOUBLE, mdi_comm);
    energy_valid = false;

  }

  /* Respond to the <CELL command */
  else if ( strcmp(command, "<CELL_DISPL") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);

    std::vector<double> cell_displ(3, 0.0);
    MDI_Send(&cell_displ[0], 3, MDI_DOUBLE, mdi_comm);

  }

  /* Respond to the <DIMENSIONS command */
  else if ( strcmp(command, "<DIMENSIONS") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);

    MDI_Send(&dimensions[0], 3, MDI_INT, mdi_comm);

  }

  /* Respond to the >DIMENSIONS command */
  else if ( strcmp(command, ">DIMENSIONS") == 0 ) {

    MDI_Recv(&dimensions[0], 3, MDI_INT, mdi_comm);

  }

  /* Respond to the <ELEMENTS command */
  else if ( strcmp(command, "<ELEMENTS") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);

    std::vector<int> elements_send(elements.size(), 0);
    for (int64_t iatom=0; iatom < (int64_t)elements.size(); iatom++) {
      elements_send[iatom] = elements[iatom];
    }
    MDI_Send(&elements_send[0], elements.size(), MDI_INT, mdi_comm);

  }

  /* Respond to the >ELEMENTS command */
  else if ( strcmp(command, ">ELEMENTS") == 0 ) {

    MDI_Recv(&elements[0], elements.size(), MDI_INT, mdi_comm);

  }

  /* Respond to the <COORDS command */
  else if ( strcmp(command, "<COORDS") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);

    MDI_Send(&coords[0], coords.size(), MDI_DOUBLE, mdi_comm);

  }

  /* Respond to the <COORDS command */
  else if ( strcmp(command, ">COORDS") == 0 ) {

    MDI_Recv(&coords[0], coords.size(), MDI_DOUBLE, mdi_comm);

  }

  /* Respond to the <ENERGY command */
  else if ( strcmp(command, "<ENERGY") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);
    update_scf(&energy_valid, geo, &total_energy, &kinetic_energy, forces, &scf_box);

    MDI_Send(&total_energy, 1, MDI_DOUBLE, mdi_comm);

  }

  /* Respond to the EXIT command */
  else if ( strcmp(command, "EXIT") == 0 ) {

    exit_signal = true;

  }

  /* Respond to the <FORCES command */
  else if ( strcmp(command, "<FORCES") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);
    update_scf(&energy_valid, geo, &total_energy, &kinetic_energy, forces, &scf_box);

    MDI_Send(&forces[0], coords.size(), MDI_DOUBLE, mdi_comm);

  }

  /* Respond to the <KE command */
  else if ( strcmp(command, "<KE") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);
    update_scf(&energy_valid, geo, &total_energy, &kinetic_energy, forces, &scf_box);

    MDI_Send(&kinetic_energy, 1, MDI_DOUBLE, mdi_comm);

  }

  /* Respond to the <MASSES command */
  else if ( strcmp(command, "<MASSES") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);

    std::vector<double> masses_send(elements.size(), 0.0);
    for (int64_t iatom=0; iatom < (int64_t)elements.size(); iatom++) {
      masses_send[iatom] = geo[iatom].species().mass();
    }
    MDI_Send(&masses_send[0], elements.size(), MDI_DOUBLE, mdi_comm);

  }

  /* Respond to the <NATOMS command */
  else if ( strcmp(command, "<NATOMS") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);

    int64_t elements_size = elements.size();
    MDI_Send(&elements_size, 1, MDI_INT64_T, mdi_comm);

  }

  /* Respond to the >NATOMS command */
  else if ( strcmp(command, ">NATOMS") == 0 ) {

    int64_t new_natoms;
    MDI_Recv(&new_natoms, 1, MDI_INT64_T, mdi_comm);
    if ( new_natoms < 0 ) {
      std::cout << "ERROR: Invalid value received by >NATOMS command: " << new_natoms << std::endl;
      MPI_Abort(mpi_world_comm, 1);
    }
    else if ( new_natoms < (int64_t)elements.size() ) {
      for ( int64_t iatom=0; iatom < (int64_t)elements.size() - new_natoms; iatom++ ) {
	elements.pop_back();
	coords.pop_back();
	coords.pop_back();
	coords.pop_back();
	forces.pop_back();
	forces.pop_back();
	forces.pop_back();
      }
    }
    else if ( new_natoms > (int64_t)elements.size() ) {
      for ( int64_t iatom=0; iatom < new_natoms - (int64_t)elements.size(); iatom++ ) {
	elements.push_back( 0 );
	coords.push_back( 0.0 );
	coords.push_back( 0.0 );
	coords.push_back( 0.0 );
	forces.push_back( 0.0 );
	forces.push_back( 0.0 );
	forces.push_back( 0.0 );
      }
    }

    energy_valid = false;

  }

  /* Respond to the <KE command */
  else if ( strcmp(command, "<PE") == 0 ) {

    update_system(cell, &scf_box, dimensions, elements, coords, geo);
    update_scf(&energy_valid, geo, &total_energy, &kinetic_energy, forces, &scf_box);

    double potential_energy = total_energy - kinetic_energy;
    MDI_Send(&potential_energy, 1, MDI_DOUBLE, mdi_comm);

  }

  /* Respond to an unrecognized command */
  else {
    /* The received command is not recognized by this engine, so exit
       Note: Replace this with whatever error handling method your code uses */
    MPI_Abort(mpi_world_comm, 1);
  }

  return 0;
}


int initialize_mdi(MDI_Comm* comm_ptr) {
  /* Set values defining the system */
  energy_valid = false;
  cell[0] = 10.0;
  cell[4] = 10.0;
  cell[8] = 10.0;

  elements[0] = 8;
  elements[1] = 1;
  elements[2] = 1;
  coords[1] = -0.553586;
  coords[3] = 1.429937;
  coords[4] = 0.553586;
  coords[6] = -1.429937;
  coords[7] = 0.553586;

  /* Register all supported commands and nodes */
  MDI_Register_node("@DEFAULT");
  MDI_Register_command("@DEFAULT", "<CELL");
  MDI_Register_command("@DEFAULT", ">CELL");
  MDI_Register_command("@DEFAULT", "<COORDS");
  MDI_Register_command("@DEFAULT", ">COORDS");
  MDI_Register_command("@DEFAULT", "<CELL_DISPL");
  MDI_Register_command("@DEFAULT", "<DIMENSIONS");
  MDI_Register_command("@DEFAULT", ">DIMENSIONS");
  MDI_Register_command("@DEFAULT", "<ELEMENTS");
  MDI_Register_command("@DEFAULT", ">ELEMENTS");
  MDI_Register_command("@DEFAULT", "<ENERGY");
  MDI_Register_command("@DEFAULT", "EXIT");
  MDI_Register_command("@DEFAULT", "<FORCES");
  MDI_Register_command("@DEFAULT", "<KE");
  MDI_Register_command("@DEFAULT", "<MASSES");
  MDI_Register_command("@DEFAULT", "<NATOMS");
  MDI_Register_command("@DEFAULT", ">NATOMS");
  MDI_Register_command("@DEFAULT", "<PE");

  MDI_Accept_communicator(comm_ptr);

  // Set the execute_command callback
  int (*generic_command)(const char*, MDI_Comm, void*) = execute_command;
  void* engine_obj = nullptr;
  MDI_Set_execute_command_func(generic_command, engine_obj);

  return 0;
}


int respond_to_commands(MDI_Comm comm) {
  // Respond to the driver's commands
  char* command = new char[MDI_COMMAND_LENGTH];
  while( not exit_signal ) {

    MDI_Recv_command(command, comm);
    MPI_Bcast(command, MDI_COMMAND_LENGTH, MPI_CHAR, 0, mpi_world_comm);

    /* Confirm that this command is actually supported at this node */
    int command_supported = 1;
    if ( my_rank == 0 ) {
      MDI_Check_command_exists("@DEFAULT", command, MDI_COMM_NULL, &command_supported);
    }
    if ( command_supported != 1 ) {
      /* Note: Replace this with whatever error handling method your code uses */
      MPI_Abort(mpi_world_comm, 1);
    }

    execute_command(command, comm, NULL);
  }
  delete [] command;

  return 0;
}


int MDI_Plugin_init_inqmdi() {
  /* MPI intra-communicator for all processes running this code */
  mpi_world_comm = MPI_COMM_WORLD;

  // Get the command-line arguments for this plugin instance
  int mdi_argc;
  if ( MDI_Plugin_get_argc(&mdi_argc) ) {
    MPI_Abort(mpi_world_comm, 1);
  }
  char** mdi_argv;
  if ( MDI_Plugin_get_argv(&mdi_argv) ) {
    MPI_Abort(mpi_world_comm, 1);
  }

  // Call MDI_Init
  MDI_Init(&mdi_argc, &mdi_argv);

  // Get the MPI intra-communicator for this code
  MDI_MPI_get_world_comm(&mpi_world_comm);
  MPI_Comm_rank(mpi_world_comm, &my_rank);

  // Perform one-time operations required to establish a connection with the driver
  MDI_Comm mdi_comm = MDI_COMM_NULL;
  initialize_mdi(&mdi_comm);

  //inq::input::environment env(mdi_argc, mdi_argv);

  // Respond to commands from the driver
  respond_to_commands(mdi_comm);

  fftw_cleanup(); //required for valgrid

  return 0;
}


int main(int argc, char ** argv){

  /* MDI communicator used to communicate with the driver */
  MDI_Comm mdi_comm = MDI_COMM_NULL;

  /* Initialize the MDI Library */
  MDI_Init(&argc, &argv);
  MDI_MPI_get_world_comm(&mpi_world_comm);

  /* Register all supported commands and nodes */
  initialize_mdi(&mdi_comm);

  inq::input::environment env(argc, argv);

  respond_to_commands(mdi_comm);

  fftw_cleanup(); //required for valgrid

  return 0;
}

