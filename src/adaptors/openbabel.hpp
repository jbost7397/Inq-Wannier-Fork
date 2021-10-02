// © Alfredo Correa 2021

#include<systems/ions.hpp>

#include<openbabel/atom.h>
#include<openbabel/mol.h> // OpenBabel::OBMol
#include<openbabel/generic.h> // OpenBabel::OBUnitCell

#include<vector>

namespace inq{

static auto to_system_ions(OpenBabel::OBMol mol){
	static auto const A_over_b = 1.889726125; // angstrom over bohr

	namespace input = inq::input;

	input::cell const c = [&mol]{
		auto ouc = static_cast<OpenBabel::OBUnitCell const*>(mol.GetData(OpenBabel::OBGenericDataType::UnitCell));
		assert( ouc and ouc->GetAlpha() == 90. and ouc->GetBeta() == 90. and ouc->GetGamma() == 90. );
		using namespace inq::magnitude;
		return input::cell::orthorhombic(
			ouc->GetA()*A_over_b*1._b,
			ouc->GetB()*A_over_b*1._b,
			ouc->GetC()*A_over_b*1._b
		);
	}();

	std::vector<input::atom> const geo = [&mol]{
		std::vector<input::atom> geo;
		geo.reserve(mol.NumAtoms());
		std::transform(
			mol.BeginAtoms(), mol.EndAtoms(), std::back_inserter(geo),
			[](auto e) -> input::atom{
				auto const p = e->GetVector()*A_over_b;
				return {
					input::species{pseudo::element{e->GetType()}},
					inq::math::vector3{p.x(), p.y(), p.z()} // in bohr
				};
			}
		);
		return geo;
	}();
	
	return inq::systems::ions{c, geo};
}

}

