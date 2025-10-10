/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__XC_FUNCTIONAL
#define INQ__HAMILTONIAN__XC_FUNCTIONAL

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa, Alexey Kartsev
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include <xc.h>
#include <observables/density.hpp>
#include <operations/gradient.hpp>
#include <operations/integral.hpp>
#include <operations/divergence.hpp>
#include <operations/laplacian.hpp>
#include <basis/field.hpp>

#include <utils/profiling.hpp>
#include <utils/raw_pointer_cast.hpp>

#define XC_NONE          0
#define XC_HARTREE_FOCK -4659

namespace inq {
namespace hamiltonian {

namespace set_exchange {
	inline double global_exchange_frac = -1.0; //CS default of -1.0 will triger libxc to use it's default value 
}

namespace set_rsh {
	inline vector3<double> rsh_parameters {0.0, 0.25, 0.11}; 
}

	class xc_functional {

		public:
		
		auto true_functional() const {
			return id_ > 0;			
		}
		
		xc_functional(const int functional_id, int spin_components):
			id_(functional_id),
			nspin_(spin_components){

			if(true_functional()) assert(nspin_ == 1 or nspin_ == 2);
			
			if(true_functional() and xc_func_init(&func_, functional_id, (nspin_==1)?XC_UNPOLARIZED:XC_POLARIZED) != 0){
				fprintf(stderr, "Functional '%d' not found\n", functional_id);
				exit(1);
			}
		}

		xc_functional(xc_functional const & other):
			xc_functional(other.id_, other.nspin_){
		}
		
		xc_functional operator=(xc_functional const & other) {
			if(id_ == other.id_ and nspin_ == other.nspin_) return *this;
			if(true_functional()) xc_func_end(&func_);
			id_ = other.id_;
			nspin_ = other.nspin_;
			if(true_functional() and xc_func_init(&func_, id_, (nspin_==1)?XC_UNPOLARIZED:XC_POLARIZED) != 0){
				fprintf(stderr, "Functional '%d' not found\n", id_);
				exit(1);
			}
			return *this;
		}
		
		~xc_functional(){
			if(true_functional()) xc_func_end(&func_);
		}
		
		auto & family() const {
			return func_.info->family;
		}

		auto & libxc_func() const {
			return func_;
		}
		
		auto libxc_func_ptr() const {
			return &func_;
		}

		auto requires_gradient() const {
			if(not true_functional()) return false;
			return family() != XC_FAMILY_LDA;
		}

		auto exx_coefficients() const {
                        double omega = 0.0, alpha = 0.0, beta = 0.0;
                        if(xc_hyb_type(libxc_func_ptr()) == XC_HYB_HYBRID) {
			  if(set_exchange::global_exchange_frac < 0.0) {
                                alpha = xc_hyb_exx_coef(libxc_func_ptr());
                          } else {
				alpha = set_exchange::global_exchange_frac;
				xc_func_set_ext_params(const_cast<xc_func_type*>(libxc_func_ptr()), &alpha);
			  }           
      			  beta = 0.0;
                          omega = 0.0;
			}
			//Range-seperated hybrid
			else if (xc_hyb_type(libxc_func_ptr()) == XC_HYB_CAM) {
                          xc_hyb_cam_coef(libxc_func_ptr(), &omega, &alpha, &beta);
			  if(const_cast<xc_func_type*>(libxc_func_ptr())->info->number == XC_HYB_GGA_XC_CAM_PBEH) {
			    alpha = set_rsh::rsh_parameters[0];
			    beta = set_rsh::rsh_parameters[1];
			    //beta = set_rsh::rsh_parameters[1] - set_rsh::rsh_parameters[0]; //for libxc
			    omega = set_rsh::rsh_parameters[2];

                            double rsh_par[3] = {alpha, beta, omega}; 
			    xc_func_type* func_ptr = const_cast<xc_func_type*>(libxc_func_ptr());
			    xc_func_set_ext_params(func_ptr, rsh_par);
			  } 
                        }
			std::cout << alpha << " " << beta << " " << omega << std::endl; //this is what libxc sees 
                        return vector3<double>(alpha, beta + alpha, omega); //beta in INQ/Qbach format (what user inputs)
                }

		std::string name() const {
			if(id_ == XC_NONE)         return "";			
			if(id_ == XC_HARTREE_FOCK) return "Exact exchange";
			
			return func_.info->name;
		}

		std::string kind_name() const {
			if(id_ == XC_NONE)         return "";
			if(id_ == XC_HARTREE_FOCK) return "Exchange";
			
			switch (func_.info->kind) {
			case (XC_EXCHANGE):
				return "Exchange";
			case (XC_CORRELATION):
				return "Correlation";
			case (XC_EXCHANGE_CORRELATION):
				return "Exchange-correlation";
			case (XC_KINETIC):
				return "Kinetic energy";
			default:
				return "Unknown";
			}
		}
		
		std::string family_name() const {
			if(id_ == XC_NONE)         return "";
			if(id_ == XC_HARTREE_FOCK) return "Hartree-Fock";
			
			switch (family()) {
			case (XC_FAMILY_LDA):
				return "LDA";
			case (XC_FAMILY_GGA):
				return "GGA";
			case (XC_FAMILY_MGGA):
				return "MGGA";
			default:
				return "Unknown";
			}
		}

		auto references(std::string prefix = {}) const {
			std::string refs;
			
			if(id_ == XC_NONE) return refs;

			if(id_ == XC_HARTREE_FOCK) {
				refs += prefix + "[1] V. A. Fock, Z. Phys. 61, 126 (1930)\n";
				refs += prefix + "[2] V. A. Fock, Z. Phys. 62, 795 (1930)\n";
				refs += prefix + "[3] D. R. Hartree and W. Hartree, Proc. R. Soc. Lond. A. 150, 869 (1935)\n";
				return refs;
			}
														
			for(int ii = 0; func_.info->refs[ii] != NULL; ii++){
				refs += prefix;
				refs += "[" + std::to_string(ii + 1) + "] ";
				refs += func_.info->refs[ii]->ref;
				refs += "\n";
			}
			return refs;
		}

		auto & nspin() const {
			return nspin_;
		}
		
	private:
		
		int id_;
		int nspin_;
		xc_func_type func_;
	};

}
}
#endif

#ifdef INQ_HAMILTONIAN_XC_FUNCTIONAL_UNIT_TEST
#undef INQ_HAMILTONIAN_XC_FUNCTIONAL_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;	
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("LDA"){
		inq::hamiltonian::xc_functional ldafunctional(XC_LDA_X, 1);
		CHECK(ldafunctional.exx_coefficients()[0] == 0.0);
		CHECK(ldafunctional.name() == "Slater exchange");
		CHECK(ldafunctional.kind_name() == "Exchange");
		CHECK(ldafunctional.family_name() == "LDA");
		CHECK(ldafunctional.references() == "[1] P. A. M. Dirac, Math. Proc. Cambridge Philos. Soc. 26, 376 (1930)\n[2] F. Bloch, Z. Phys. 57, 545 (1929)\n");
	}

	SECTION("LSDA"){
		inq::hamiltonian::xc_functional ldafunctional(XC_LDA_X, 2);
		CHECK(ldafunctional.exx_coefficients()[0] == 0.0);
	}
		
	SECTION("GGA"){
		inq::hamiltonian::xc_functional ggafunctional(XC_GGA_X_PBE, 1);
		CHECK(ggafunctional.exx_coefficients()[0] == 0.0);
		CHECK(ggafunctional.name() == "Perdew, Burke & Ernzerhof");
	}

	SECTION("Spin GGA"){
		inq::hamiltonian::xc_functional ggafunctional(XC_GGA_X_PBE, 2);
		CHECK(ggafunctional.exx_coefficients()[0] == 0.0);
		CHECK(ggafunctional.family_name() == "GGA");
	}

	inq::hamiltonian::xc_functional b3lyp(XC_HYB_GGA_XC_B3LYP, 1);
	inq::hamiltonian::xc_functional pbeh(XC_HYB_GGA_XC_PBEH, 1);
	inq::hamiltonian::xc_functional hse(XC_HYB_GGA_XC_HSE06, 1);
	inq::hamiltonian::xc_functional camb3lyp(XC_HYB_GGA_XC_CAM_B3LYP, 1); 
	
	SECTION("HYBRIDS"){
		CHECK(b3lyp.exx_coefficients()[0] == 0.2_a);
		CHECK(b3lyp.kind_name() == "Exchange-correlation");
		CHECK(b3lyp.family_name() == "GGA");
		CHECK(pbeh.exx_coefficients()[0] == 0.25_a);
		CHECK(hse.family_name() == "GGA");
                CHECK(hse.kind_name() == "Exchange-correlation");
		CHECK(hse.exx_coefficients()[0] == 0.0_a);
                CHECK(hse.exx_coefficients()[1] == 0.25_a);
                CHECK(hse.exx_coefficients()[2] == 0.11_a);
                CHECK(camb3lyp.exx_coefficients()[0] == 0.65_a);
                CHECK(camb3lyp.exx_coefficients()[1] == 0.19_a);
                CHECK(camb3lyp.exx_coefficients()[2] == 0.33_a);
	}

	SECTION("COPY AND ASSIGNMENT"){
		auto copy = b3lyp;
		CHECK(copy.exx_coefficients()[0] == 0.2_a);

		copy = pbeh;
		CHECK(copy.exx_coefficients()[0] == 0.25_a);

		auto copy2 = std::move(copy);
		CHECK(copy2.exx_coefficients()[0] == 0.25_a);

		copy2 = std::move(b3lyp);
		CHECK(copy2.exx_coefficients()[0] == 0.2_a);

                auto copy3 = hse;
                CHECK(copy3.exx_coefficients()[2] == 0.11_a);
	}
	
        SECTION("DD_HYBRIDS"){
	        inq::hamiltonian::set_exchange::global_exchange_frac = 0.5;
                CHECK(pbeh.exx_coefficients()[0] == 0.5_a);
		CHECK(b3lyp.exx_coefficients()[0] == 0.5_a);
        }

        SECTION("RSH"){
                inq::hamiltonian::set_rsh::rsh_parameters = {0.65, 0.19, 0.33};
		inq::hamiltonian::xc_functional rsh(XC_HYB_GGA_XC_CAM_PBEH, 1);
                CHECK(rsh.exx_coefficients()[0] == 0.65_a);
                CHECK(rsh.exx_coefficients()[1] == 0.19_a);
                CHECK(rsh.exx_coefficients()[2] == 0.33_a);
        }
}
#endif

