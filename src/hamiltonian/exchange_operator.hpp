/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__EXCHANGE_OPERATOR
#define INQ__HAMILTONIAN__EXCHANGE_OPERATOR

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/real_space.hpp>
#include <hamiltonian/singularity_correction.hpp>
#include <input/parallelization.hpp>
#include <matrix/diagonal.hpp>
#include <matrix/cholesky.hpp>
#include <operations/overlap.hpp>
#include <operations/overlap_diagonal.hpp>
#include <operations/rotate.hpp>
#include <parallel/arbitrary_partition.hpp>
#include <parallel/array_iterator.hpp>
#include <solvers/poisson.hpp>
#include <states/index.hpp>
#include <states/orbital_set.hpp>
#include <wannier/tdmlwf_trans.hpp>

#include <optional>

namespace inq {
namespace hamiltonian {
  class exchange_operator {

		gpu::array<double, 1> occupations_;
		gpu::array<vector3<double, covariant>, 1> kpoints_;
		gpu::array<int, 1> kpoint_indices_;
		std::optional<basis::field_set<basis::real_space, complex, parallel::arbitrary_partition>> orbitals_;
		std::vector<states::orbital_set<basis::real_space, complex>> ace_orbitals_;
		vector3<double> exchange_coefficients_;
		bool use_ace_;
		bool use_cutoff_;
		singularity_correction sing_;
		states::index orbital_index_;
		std::optional<wannier::tdmlwf_trans> mlwf_;
		double epsilon_;

  public:

		exchange_operator(systems::cell const & cell, ionic::brillouin const & bzone, vector3<double> const & exchange_coefficients, bool const use_ace):
			exchange_coefficients_(exchange_coefficients),
			use_ace_(use_ace),
			sing_(cell, bzone){
		}

		exchange_operator(systems::cell const & cell, ionic::brillouin const & bzone, vector3<double> const & exchange_coefficients, wannier::tdmlwf_trans mlwf, bool const use_ace, bool const use_cutoff, double const epsilon):
			exchange_coefficients_(exchange_coefficients),
			use_ace_(use_ace),
			use_cutoff_(use_cutoff),
			sing_(cell, bzone),
	  		epsilon_(epsilon),
	  		mlwf_(mlwf){
		}

		//////////////////////////////////////////////////////////////////////////////////

		template <class ElectronsType>
		double update(ElectronsType const & el){
			if(not enabled()) return 0.0;

			CALI_CXX_MARK_SCOPE("exchage_operator::update");

			auto part = parallel::arbitrary_partition(el.max_local_set_size()*el.kpin_size(), el.kpin_states_comm());

			occupations_ = el.occupations().flatted();
			kpoints_.reextent(part.local_size());
			kpoint_indices_.reextent(part.local_size());

			assert(el.states_comm().size() == 1 or el.kpin_comm().size() == 1); //this is not supported right now since we don't have a way to construct the communicator with combined dimensions
			auto par_dim = input::parallelization::dimension_kpoints();
			if(el.kpin_comm().size() == 1) par_dim = input::parallelization::dimension_states();

			if(not orbitals_.has_value()) orbitals_.emplace(el.states_basis(), part, el.full_comm().plane(input::parallelization::dimension_domains(), par_dim));

			{
				auto ist = 0;
				for(auto & phi : el.kpin()){

					gpu::run(phi.local_set_size(), [ist, kpoints = begin(kpoints_), indices = begin(kpoint_indices_), kp = phi.kpoint(), ind = el.kpoint_index(phi)] GPU_LAMBDA (auto ii) {
						kpoints[ist + ii] = kp;
						indices[ist + ii] = ind;
					});

					orbitals_->matrix()({0, phi.basis().local_size()}, {ist, ist + phi.local_set_size()}) = phi.matrix();

					ist += phi.local_set_size();
				}
			}

			if(not use_ace_) return 0.0;

			ace_orbitals_.clear();

			auto energy = 0.0;
			{
				auto iphi = 0;
				for(auto & phi : el.kpin()){

					orbital_index_[phi.key()] = iphi;

					auto exxphi = direct(phi, -1.0);
					auto exx_matrix = operations::overlap(exxphi, phi);

					energy += -0.5*real(operations::sum_product(el.occupations()[iphi], matrix::diagonal(exx_matrix)));

					matrix::cholesky(exx_matrix);
					operations::rotate_trs(exx_matrix, exxphi);

					ace_orbitals_.emplace_back(std::move(exxphi));

					iphi++;
				}
			}

			el.kpin_states_comm().all_reduce_n(&energy, 1);

			return energy;
		}

		//////////////////////////////////////////////////////////////////////////////////

		auto direct(const states::orbital_set<basis::real_space, complex> & phi, double scale = 1.0) const {
			states::orbital_set<basis::real_space, complex> exxphi(phi.skeleton());
			exxphi.fill(0.0);
			direct(phi, exxphi, scale);
			return exxphi;
		}

		//////////////////////////////////////////////////////////////////////////////////

		template <class HFType, class HFOccType, class KptType, class IdxType, class PhiType, class ExxphiType>
		void block_exchange(double factor, HFType const & hf, HFOccType const & hfocc, KptType const & kpt, IdxType const & idx, PhiType const & phi, ExxphiType & exxphi) const {

			auto nst = phi.local_set_size();
			auto nhf = (~hf).size();
			basis::field_set<basis::real_space, complex> rhoij(phi.basis(), nst);

			for(int jj = 0; jj < nhf; jj++){

				if(fabs(hfocc[jj]) < 1e-10) continue;

				{ CALI_CXX_MARK_SCOPE("exchange_operator::generate_density");
					gpu::run(nst, phi.basis().local_size(),
									 [rho = begin(rhoij.matrix()), hfo = begin(hf), ph = begin(phi.matrix()), jj] GPU_LAMBDA (auto ist, auto ipoint){
										 rho[ipoint][ist] = conj(hfo[ipoint][jj])*ph[ipoint][ist];  
									 });
				}

				solvers::poisson::in_place(rhoij, -phi.kpoint() + kpt[jj], sing_(idx[jj]), exchange_coefficients_);

				{ CALI_CXX_MARK_SCOPE("exchange_operator::mulitplication");
					gpu::run(nst, exxphi.basis().local_size(),
									 [pot = begin(rhoij.matrix()), hfo = begin(hf), exph = begin(exxphi.matrix()), scal = factor*hfocc[jj], jj]
									 GPU_LAMBDA (auto ist, auto ipoint){
										 exph[ipoint][ist] += scal*hfo[ipoint][jj]*pot[ipoint][ist];
									 });
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////////////

		template <class HFType, class HFOccType, class KptType, class IdxType, class PhiType, class ExxphiType, class WannierType>
		void block_exchange_w_cutoff(double factor, HFType const & hf, HFOccType const & hfocc, KptType const & kpt, IdxType const & idx, PhiType const & phi, ExxphiType & exxphi, WannierType & mlwf, double epsilon) const {

			auto nst = phi.local_set_size();
			auto nhf = (~hf).size();

			for(int jj = 0; jj < nhf; jj++){

				if(fabs(hfocc[jj]) < 1e-10) continue;

				auto olaps_j = mlwf->get_overlaps_of_j(epsilon, jj, phi.basis().cell());
				basis::field_set<basis::real_space, complex> rhoij(phi.basis(), olaps_j.size());

				{ CALI_CXX_MARK_SCOPE("exchange_operator::generate_density");
					gpu::run(olaps_j.size(), phi.basis().local_size(),
									 [rho = begin(rhoij.matrix()), hfo = begin(hf), ph = begin(phi.matrix()), oj = begin(olaps_j), jj] GPU_LAMBDA (auto ist, auto ipoint){
										 rho[ipoint][ist] = conj(hfo[ipoint][jj])*ph[ipoint][oj[ist]];
									 });
				}

				solvers::poisson::in_place(rhoij, -phi.kpoint() + kpt[jj], sing_(idx[jj]));

				{ CALI_CXX_MARK_SCOPE("exchange_operator::mulitplication");
					gpu::run(olaps_j.size(), exxphi.basis().local_size(),
									 [pot = begin(rhoij.matrix()), hfo = begin(hf), exph = begin(exxphi.matrix()), scal = factor*hfocc[jj], jj, oj = begin(olaps_j)]
									 GPU_LAMBDA (auto ist, auto ipoint){
										 exph[ipoint][oj[ist]] += scal*hfo[ipoint][jj]*pot[ipoint][ist];
									 });
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////////////

		void direct(const states::orbital_set<basis::real_space, complex> & phi, states::orbital_set<basis::real_space, complex> & exxphi, double scale = 1.0) const {
			if(not enabled()) return;

			CALI_CXX_MARK_SCOPE("exchange_operator::direct");

			//double factor = -0.5*scale*exchange_coefficient_;
			double factor = -0.5 * scale * exchange_coefficients_[0];  //CS with range seperation now vec, test pbe0
			//double factor = -0.5 * scale; 

			if(not orbitals_->set_part().parallel()){
				if(use_cutoff_) block_exchange_w_cutoff(factor, orbitals_->matrix(), occupations_, kpoints_, kpoint_indices_, phi, exxphi, mlwf_, epsilon_);
				else block_exchange(factor, orbitals_->matrix(), occupations_, kpoints_, kpoint_indices_, phi, exxphi);
			} else {
				auto occ_it = parallel::array_iterator(orbitals_->set_part(), orbitals_->set_comm(), occupations_);
				auto kpt_it = parallel::array_iterator(orbitals_->set_part(), orbitals_->set_comm(), kpoints_);
				auto idx_it = parallel::array_iterator(orbitals_->set_part(), orbitals_->set_comm(), kpoint_indices_);
				auto hfo_it = parallel::block_array_iterator(orbitals_->basis().local_size(), orbitals_->set_part(), orbitals_->set_comm(), orbitals_->matrix());
				for(; hfo_it != hfo_it.end(); ++hfo_it){
					if(use_cutoff_) block_exchange_w_cutoff(factor, orbitals_->matrix(), occupations_, kpoints_, kpoint_indices_, phi, exxphi, mlwf_, epsilon_);
					else block_exchange(factor, orbitals_->matrix(), occupations_, kpoints_, kpoint_indices_, phi, exxphi);
					++occ_it;
					++kpt_it;
					++idx_it;
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////////////

		auto ace(const states::orbital_set<basis::real_space, complex> & phi) const {
			states::orbital_set<basis::real_space, complex> exxphi(phi.skeleton());
			exxphi.fill(0.0);
			ace(phi, exxphi);
			return exxphi;
		}

		//////////////////////////////////////////////////////////////////////////////////

		auto operator()(const states::orbital_set<basis::real_space, complex> & phi) const {
			states::orbital_set<basis::real_space, complex> exxphi(phi.skeleton());
			exxphi.fill(0.0);
			operator()(phi, exxphi);
			return exxphi;
		}

		//////////////////////////////////////////////////////////////////////////////////

		void operator()(const states::orbital_set<basis::real_space, complex> & phi, states::orbital_set<basis::real_space, complex> & exxphi) const {
			if(not enabled()) return;

			if(use_ace_) ace(phi, exxphi);
			else direct(phi, exxphi);
		}

		//////////////////////////////////////////////////////////////////////////////////

		void ace(const states::orbital_set<basis::real_space, complex> & phi, states::orbital_set<basis::real_space, complex> & exxphi) const {
			if(not enabled()) return;

			CALI_CXX_MARK_SCOPE("exchange_operator::ace");
			namespace blas = boost::multi::blas;

			auto index = orbital_index_.at(phi.key());

			assert(ace_orbitals_.size() > 0);
			assert(phi.kpoint() == ace_orbitals_[index].kpoint());
			assert(phi.spin_index() == ace_orbitals_[index].spin_index());

			auto olap = operations::overlap(ace_orbitals_[index], phi);
			operations::rotate(olap, ace_orbitals_[index], exxphi, -1.0, 1.0);
		}

		//////////////////////////////////////////////////////////////////////////////////

		bool enabled() const {
			return fabs(exchange_coefficients_[0] + exchange_coefficients_[1] + exchange_coefficients_[2]) > 1.0e-14;
		}

  };

}
}
#endif

#ifdef INQ_HAMILTONIAN_EXCHANGE_OPERATOR_UNIT_TEST
#undef INQ_HAMILTONIAN_EXCHANGE_OPERATOR_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace Catch::literals;

}
#endif
