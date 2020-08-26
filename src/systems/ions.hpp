/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__SYSTEMS__IONS
#define INQ__SYSTEMS__IONS

#include <cfloat>

#include <ions/geometry.hpp>
#include <ions/unitcell.hpp>
#include <input/cell.hpp>
#include <mpi3/environment.hpp>

namespace inq {
namespace systems {

class ions {

public:

	ions(const input::cell & arg_cell_input, const inq::ions::geometry & geo_arg = inq::ions::geometry()):
		cell_(arg_cell_input, arg_cell_input.periodic_dimensions()),
		geo_(geo_arg){

		if(boost::mpi3::environment::get_world_instance().root()){
			geo_.info(std::cout);
			cell_.info(std::cout);
		}
	}

	auto & geo() const {
		return geo_;
	}

	auto & cell() const {
		return cell_;
	}
    
	inq::ions::UnitCell cell_;
	inq::ions::geometry geo_;

};
  
}
}
#endif

