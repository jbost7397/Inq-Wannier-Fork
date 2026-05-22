/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__WANNIER__TDMLWF_TRANS
#define INQ__WANNIER__TDMLWF_TRANS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>
#include <input/environment.hpp>
#include <math/complex.hpp>
#include <utils/profiling.hpp>
#include <math/vector3.hpp>
#include <systems/cell.hpp>
#include <basis/field.hpp>
#include <matrix/gather_scatter.hpp>
#include <states/ks_states.hpp>
#include <states/orbital_set.hpp>
#include <operations/rotate.hpp>
#include <parallel/communicator.hpp>
#include <parallel/array_iterator.hpp>
#include <wannier/jade_complex.hpp>

namespace inq {
namespace wannier {

class tdmlwf_trans {

private:
	gpu::array<complex,2> u_; 
	gpu::array<complex,3> a_;
	gpu::array<complex,2> adiag_;
	mutable states::orbital_set<basis::real_space, complex> wavefunctions_;

public:

////////////////////////////////////////////////////////////////////////////////
tdmlwf_trans(states::orbital_set<basis::real_space, complex> const & wavefunctions) : wavefunctions_(wavefunctions) {
	const int n_states = wavefunctions_.set_size();
	a_.reextent({6, n_states, n_states});
	adiag_.reextent({6, n_states});
 	u_.reextent({n_states, n_states});
}//constructor

////////////////////////////////////////////////////////////////////////////////
template <class CommType>
void normalize(CommType & comm) {
  	CALI_CXX_MARK_SCOPE("wannier_normalize");
	int n_states_local = wavefunctions_.local_set_size();
	int n_states_global = wavefunctions_.set_size();
	int nx = wavefunctions_.basis().local_sizes()[0];
	int ny = wavefunctions_.basis().local_sizes()[1];
	int nz = wavefunctions_.basis().local_sizes()[2];
	auto rank = comm.rank();
	int offset = rank * n_states_local;
	if(n_states_local == n_states_global) {
        	offset = 0;
        }

	gpu::array<double, 1> norm_squared_per_state({n_states_global}, 0.0);

	gpu::run(n_states_local, [hypercubic = begin(wavefunctions_.hypercubic()), offset, nx, ny, nz, nsp = begin(norm_squared_per_state)] GPU_LAMBDA (auto k_wf) {
  		for (int ix = 0; ix < nx; ++ix) {
      			for (int iy = 0; iy < ny; ++iy) {
        			for (int iz = 0; iz < nz; ++iz) {
          				complex wf_component = hypercubic[ix][iy][iz][k_wf];
          				nsp[k_wf + offset] += norm(wf_component);
        			}
      			}
    		}
  	});

	comm.barrier();
	comm.all_reduce_in_place_n(raw_pointer_cast(norm_squared_per_state.data_elements()), norm_squared_per_state.num_elements(), std::plus<>());

  	gpu::run(n_states_local, [hypercubic = begin(wavefunctions_.hypercubic()), offset, nx, ny, nz, nsp = begin(norm_squared_per_state)] GPU_LAMBDA (auto k_wf) {
  		double norm_factor = 1.0 / sqroot(nsp[k_wf + offset]);
    		for (int ix = 0; ix < nx; ++ix) {
    			for (int iy = 0; iy < ny; ++iy) {
        			for (int iz = 0; iz < nz; ++iz) {
          				hypercubic[ix][iy][iz][k_wf] *= norm_factor;
        			}
      			}
    		}
  	});
}//normalize

////////////////////////////////////////////////////////////////////////////////
template <class CommType>
void update(const states::orbital_set<basis::real_space, complex>& wavefunctions, CommType & comm, const systems::cell & cell_) {
  	wavefunctions_ = wavefunctions;
	CALI_CXX_MARK_SCOPE("wannier_update");
	int n_states_global = wavefunctions_.set_size();
	int n_states_local = wavefunctions_.local_set_size();
 	int nbas = wavefunctions_.basis().local_size();
	int nx = wavefunctions_.basis().local_sizes()[0];
	int ny = wavefunctions_.basis().local_sizes()[1];
	int nz = wavefunctions_.basis().local_sizes()[2];
	auto point_op = wavefunctions_.basis().point_op();
	auto cpx = wavefunctions_.basis().cubic_part(0);
	auto cpy = wavefunctions_.basis().cubic_part(1);
	auto cpz = wavefunctions_.basis().cubic_part(2);
	normalize(comm);

	gpu::run(n_states_global, n_states_global, 6, [a_int=begin(a_)] GPU_LAMBDA (auto k, auto j, auto i) {
        	a_int[i][j][k] = complex(0.0);
	});

  	gpu::array<double, 2> trig_array({6, nbas});

	gpu::array<double,1> cell_dim({9});
        for (int i = 0; i < 9; i++) {
        	cell_dim[i] = cell_[i / 3][i % 3];
	}

	gpu::run(nbas, [cpx, cpy, cpz, point_op, tb = begin(trig_array), nx, ny, nz] GPU_LAMBDA (auto ibas) {
		int ix = ibas / (ny * nz);
    		int remainder_1 = ibas % (ny * nz);
    		int iy = remainder_1 / nz;
    		int iz = remainder_1 % nz;
		auto ixg = cpx.local_to_global(ix);
		auto iyg = cpy.local_to_global(iy);
    		auto izg = cpz.local_to_global(iz);
		auto coords = point_op.rvector(ixg, iyg, izg);
    		tb[0][ibas] = cos(2.0 * M_PI * coords[0]);
    		tb[1][ibas] = sin(2.0 * M_PI * coords[0]);
    		tb[2][ibas] = cos(2.0 * M_PI * coords[1]);
    		tb[3][ibas] = sin(2.0 * M_PI * coords[1]);
    		tb[4][ibas] = cos(2.0 * M_PI * coords[2]);
    		tb[5][ibas] = sin(2.0 * M_PI * coords[2]);
	});

	gpu::array<double,2> cell_array({6, nbas}); 

	gpu::run(nbas, [tb=begin(trig_array), cell=begin(cell_dim), ta=begin(cell_array)] GPU_LAMBDA (auto ibas) {
		ta[0][ibas] = tb[0][ibas]*cell[0] + tb[2][ibas]*cell[3] + tb[4][ibas]*cell[6];
		ta[1][ibas] = tb[1][ibas]*cell[0] + tb[3][ibas]*cell[3] + tb[5][ibas]*cell[6];
		ta[2][ibas] = tb[0][ibas]*cell[1] + tb[2][ibas]*cell[4] + tb[4][ibas]*cell[7];
		ta[3][ibas] = tb[1][ibas]*cell[1] + tb[3][ibas]*cell[4] + tb[5][ibas]*cell[7]; 
		ta[4][ibas] = tb[0][ibas]*cell[2] + tb[2][ibas]*cell[5] + tb[4][ibas]*cell[8];		
		ta[5][ibas] = tb[1][ibas]*cell[2] + tb[3][ibas]*cell[5] + tb[5][ibas]*cell[8];
	});

	if (wavefunctions_.set_part().parallel()) {
  		auto loc_mat = wavefunctions_.matrix();
		auto hypercubic_it = parallel::block_array_iterator(nbas, wavefunctions_.set_part(), wavefunctions_.set_comm(), wavefunctions_.matrix());
		auto init_rank = comm.rank();
		auto cur_rank = comm.rank();
		auto k_offset = wavefunctions_.set_part().start(init_rank);
		for(; hypercubic_it != hypercubic_it.end(); ++hypercubic_it){
			auto gmat = begin(*hypercubic_it);
  			auto cur_local = gmat[0].size();
			auto l_offset = wavefunctions_.set_part().start(cur_rank);

    			gpu::run(cur_local, n_states_local, [lmat=begin(loc_mat), gmat, ta=begin(cell_array), nbas, l_offset, k_offset, a=begin(a_)] 
										GPU_LAMBDA (auto l_wf, auto k_wf) {
				complex a0 = 0.0;
				complex a1 = 0.0;
				complex a2 = 0.0;
                                complex a3 = 0.0;
                                complex a4 = 0.0;
                                complex a5 = 0.0;
				auto cur_k = k_offset + k_wf;
				auto cur_l = l_offset + l_wf;
	    			for (int ibas = 0; ibas < nbas; ibas++){
    		        		complex c_ik = lmat[ibas][k_wf];
        				complex c_jl = gmat[ibas][l_wf];
					complex c = conj_cplx(c_ik) * c_jl;
					a0 += c * ta[0][ibas];
	                                a1 += c * ta[1][ibas];
        	                        a2 += c * ta[2][ibas];
                	                a3 += c * ta[3][ibas];
                        	        a4 += c * ta[4][ibas];
                                	a5 += c * ta[5][ibas];
      		  		}
				a[0][cur_k][cur_l] += a0;
				a[1][cur_k][cur_l] += a1;
				a[2][cur_k][cur_l] += a2;
				a[3][cur_k][cur_l] += a3;
				a[4][cur_k][cur_l] += a4;
				a[5][cur_k][cur_l] += a5;
      			});
			cur_rank += 1;
			cur_rank = cur_rank % comm.size();
  		}	
    		comm.barrier();
    
    		CALI_CXX_MARK_SCOPE("wannier_update::reduce_a");
    		comm.all_reduce_in_place_n(raw_pointer_cast(a_.data_elements()), a_.num_elements(), std::plus<>());

  	} 
        else {

    		gpu::run(n_states_global, n_states_global, [mat = begin(wavefunctions_.matrix()), ta = begin(cell_array), nbas, a = begin(a_)] 
											GPU_LAMBDA (auto l_wf, auto k_wf) {
			complex a0 = 0.0;
			complex a1 = 0.0;
			complex a2 = 0.0;
                        complex a3 = 0.0;
                        complex a4 = 0.0;
                        complex a5 = 0.0;
      			for (int ibas = 0; ibas < nbas; ibas++){
            			complex c_ik = mat[ibas][k_wf];
		                complex c_jl = mat[ibas][l_wf];
				complex c = conj_cplx(c_ik) * c_jl;
				a0 += c * ta[0][ibas];
				a1 += c * ta[1][ibas];
				a2 += c * ta[2][ibas];
                                a3 += c * ta[3][ibas];
                                a4 += c * ta[4][ibas];
                                a5 += c * ta[5][ibas];	 
          		}
			a[0][k_wf][l_wf] += a0;
			a[1][k_wf][l_wf] += a1;
			a[2][k_wf][l_wf] += a2;
                        a[3][k_wf][l_wf] += a3;
                        a[4][k_wf][l_wf] += a4;
                        a[5][k_wf][l_wf] += a5; 
    		});

    		if(comm.size() > 1){
        		CALI_CXX_MARK_SCOPE("wannier_update::reduce_a_basis");
			comm.all_reduce_in_place_n(raw_pointer_cast(a_.data_elements()), a_.num_elements(), std::plus<>());
		}
  	}
}//update

////////////////////////////////////////////////////////////////////////////////
void compute_transform(double tol)
{
	const int maxsweep = 100;
  	jade_complex(maxsweep,tol,a_,u_,adiag_);
}

////////////////////////////////////////////////////////////////////////////////
const states::orbital_set<basis::real_space, complex>& get_wavefunctions() const {
  	return wavefunctions_;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
auto center(T i, const systems::cell & cell_) const {
  	assert(i >= 0 && i < wavefunctions_.set_size());

  	const double c0 = cell_.reciprocal(0)[0] * real(adiag_[0][i]) +
                          cell_.reciprocal(0)[1] * real(adiag_[2][i]) +
              	          cell_.reciprocal(0)[2] * real(adiag_[4][i]);
	const double s0 = cell_.reciprocal(0)[0] * real(adiag_[1][i]) +
                          cell_.reciprocal(0)[1] * real(adiag_[3][i]) +
                          cell_.reciprocal(0)[2] * real(adiag_[5][i]);

  	const double c1 = cell_.reciprocal(1)[0] * real(adiag_[0][i]) +
                          cell_.reciprocal(1)[1] * real(adiag_[2][i]) +
                          cell_.reciprocal(1)[2] * real(adiag_[4][i]);
  	const double s1 = cell_.reciprocal(1)[0] * real(adiag_[1][i]) +
                          cell_.reciprocal(1)[1] * real(adiag_[3][i]) +
                          cell_.reciprocal(1)[2] * real(adiag_[5][i]);

  	const double c2 = cell_.reciprocal(2)[0] * real(adiag_[0][i]) +
                          cell_.reciprocal(2)[1] * real(adiag_[2][i]) +
                          cell_.reciprocal(2)[2] * real(adiag_[4][i]);
  	const double s2 = cell_.reciprocal(2)[0] * real(adiag_[1][i]) +
                          cell_.reciprocal(2)[1] * real(adiag_[3][i]) +
                          cell_.reciprocal(2)[2] * real(adiag_[5][i]);

  	const double itwopi = 1.0 / ( 2.0 * M_PI );
  	const double t0 = (itwopi * atan2(s0,c0));
  	const double t1 = (itwopi * atan2(s1,c1));
  	const double t2 = (itwopi * atan2(s2,c2));

  	const double x = t0 * cell_[0][0] + t1 * cell_[0][1] + t2 * cell_[0][2];
  	const double y = t0 * cell_[1][0] + t1 * cell_[1][1] + t2 * cell_[1][2];
  	const double z = t0 * cell_[2][0] + t1 * cell_[2][1] + t2 * cell_[2][2];
  	vector3<double> center_d3{x,y,z};

  	return center_d3;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double wannier_distance(T i, T j, const systems::cell & cell_) const {
  	assert(i >=0 && i < wavefunctions_.set_size());
  	assert(j >=0 && j < wavefunctions_.set_size());
  	vector3<double>ctr_i = center(i, cell_);
  	vector3<double>ctr_j = center(j, cell_);
  	double x_dist = ctr_i[0] - ctr_j[0];
  	double y_dist = ctr_i[1] - ctr_j[1];
  	double z_dist = ctr_i[2] - ctr_j[2];
  	double dist_root = sqroot(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);
  	return abs(dist_root);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2>
bool overlap(T1 epsilon, T2 i, T2 j, const systems::cell & cell_) const {
	//CS this needs to be updated for non orthorombic cells (generalize min image convention)
  	double x = cell_[0][0]*cell_[0][0] + cell_[0][1]*cell_[0][1] + cell_[0][2]*cell_[0][2];
  	double y = cell_[1][0]*cell_[1][0] + cell_[1][1]*cell_[1][1] + cell_[1][2]*cell_[1][2];
  	double z = cell_[2][0]*cell_[2][0] + cell_[2][1]*cell_[2][1] + cell_[2][2]*cell_[2][2];
  	double len = sqrt(x+y+z);
  	auto dist = wannier_distance(i, j, cell_);
  	if (dist <= epsilon || dist >= (len - epsilon) ) return true;  
  	return false;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
auto get_overlaps_of_j(T epsilon, int j, const systems::cell & cell_, int rank_offset) const {
        CALI_CXX_MARK_SCOPE("wannier_update::overlaps_of_j");
  	const int n_states = wavefunctions_.local_set_size();
	gpu::array<int, 1> olap_j(n_states);
	int count = 0;
	auto i_offset = 0;
  	auto j_offset = 0;
	if(wavefunctions_.set_part().parallel()){
                i_offset = wavefunctions_.set_part().start(wavefunctions_.set_comm().rank());
                j_offset = wavefunctions_.set_part().start(rank_offset);
	}
	for(int i = 0; i < n_states; i++){
		gpu::sync();
		if(overlap(epsilon, i + i_offset, j + j_offset, cell_)){
			olap_j[count] = i;
			count++;
		}
	}
	olap_j.reextent(count);
	return olap_j;
}

////////////////////////////////////////////////////////////////////////////////
template <typename T, class CommType>
double pair_fraction(T epsilon, CommType & comm, const systems::cell & cell) const {
	CALI_CXX_MARK_SCOPE("wannier_update::pair_frac");
  	int n = wavefunctions_.set_size();
  	gpu::array<int,1> sum({1}, 0);
  	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(j > i) {
				if (overlap(epsilon, i, j, comm, cell)) {
					sum[0] += 1;
			  	}
		  	}
  	  	}
  	}

  	int total = sum[0] + n;
  	return static_cast<double>(total)/((n*(n+1))/2);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread2(T i, T j, const systems::cell & cell_) {
	assert(i >= 0 && i < wavefunctions_.set_size());
	assert(j >= 0 && j < 3);
	const double itwopi = 1.0 / ( 2.0 * M_PI );
	const auto recip = cell_.reciprocal(j);

	const complex c = itwopi * ( recip[0] * adiag_[0][i] +
                                     recip[1] * adiag_[2][i] +
                                     recip[2] * adiag_[4][i] );

	const complex s = itwopi * ( recip[0] * adiag_[1][i] +
                                     recip[1] * adiag_[3][i] +
                                     recip[2] * adiag_[5][i] );

	double length = cell_.reciprocal(j).length();
	const double fac = 1.0 / length;
	return fac*fac * ( 1.0 - norm(c) - norm(s) );
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread2(T i, const systems::cell & cell) {
	assert(i >= 0 & i < wavefunctions_.set_size());
  	return spread2(i,0,cell) + spread2(i,1,cell) + spread2(i,2,cell);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T>
double spread(T i, const systems::cell & cell) {
  	return sqroot(spread2(i,cell));
}

////////////////////////////////////////////////////////////////////////////////
auto dipole(const systems::cell & cell) {
  	vector3<double> sum{0.0,0.0,0.0};
  	for ( int i = 0; i < wavefunctions_.set_size(); i++ )
    		sum -= 2.0 * center(i,cell);  //CS need to pass state occupations (assume fully occupied for now) How?
  	return sum;
}

////////////////////////////////////////////////////////////////////////////////
template <class CommType>
void apply_transform(states::orbital_set<basis::real_space, complex> & phi, CommType & comm) {
        CALI_CXX_MARK_SCOPE("wannier::apply_transform");
  	parallel::cartesian_communicator<2> cart_comm(comm, {});
	auto rot = matrix::scatter(cart_comm, u_, /* root = */ 0);	
	operations::rotate(rot, phi);
}

////////////////////////////////////////////////////////////////////////////////
}; //tdmlwf
} //wannier
} //inq
#endif

///////////////////////////////////////////////////////////////////
#ifdef INQ_WANNIER_TDMLWF_TRANS_UNIT_TEST
#undef INQ_WANNIER_TDMLWF_TRANS_UNIT_TEST

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

        using namespace inq;
        using namespace Catch::literals;
        using Catch::Approx;
        parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	SECTION("Single Atom"){

		systems::ions sys(inq::systems::cell::cubic(20.0_b).periodic());
                sys.insert(ionic::species("He"), {8.0_b, 8.0_b, 8.0_b});
        	sys.species_list().pseudopotentials() = pseudo::set_id::sg15();
		systems::electrons el(sys, options::electrons{}.cutoff(30.0_Ry));
		ground_state::initial_guess(sys, el);

		ground_state::calculate(sys, el, options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-10_Ha));

		wannier::tdmlwf_trans mlwf_transformer(el.kpin()[0]);
	        mlwf_transformer.update(el.kpin()[0], comm, el.states_basis().cell());
		mlwf_transformer.compute_transform(1e-8);

	        auto center = mlwf_transformer.center(0, el.states_basis().cell());

	        CHECK(center[0] == Approx(8.0_a));
	        CHECK(center[1] == Approx(8.0_a));
	        CHECK(center[2] == Approx(8.0_a));

		double spread = mlwf_transformer.spread(0, el.states_basis().cell());
	        CHECK(spread == Approx(1.13316_a).epsilon(1e-3));
	}

	SECTION("Water Molecule"){

		auto ions = systems::ions::parse(config::path::unit_tests_data() + "water.xyz", systems::cell::cubic(30.0_b).periodic());
		systems::electrons el(ions, options::electrons{}.cutoff(30.0_Ry));
		ions.species_list().pseudopotentials() = pseudo::set_id::sg15();
		ground_state::initial_guess(ions, el);

                auto scf_options = options::ground_state{}.energy_tolerance(1.0e-6_Ha);
                auto result = ground_state::calculate(ions, el, options::theory{}.lda(), scf_options);		

		wannier::tdmlwf_trans mlwf_transformer(el.kpin()[0]);
                mlwf_transformer.update(el.kpin()[0], comm, el.states_basis().cell());
                mlwf_transformer.compute_transform(1e-8);
		
		int i = 0; 
		auto center = mlwf_transformer.center(i, el.states_basis().cell());
		CHECK(center[0] == Approx(0.000_a).margin(1e-6));
                CHECK(center[1] == Approx(-0.777_a).epsilon(1e-3));
                CHECK(center[2] == Approx(-0.498_a).epsilon(1e-3));

		i = 1;
                center = mlwf_transformer.center(i, el.states_basis().cell());
                CHECK(center[0] == Approx(-0.750_a).epsilon(1e-3));
                CHECK(center[1] == Approx(0.084_a).epsilon(1e-3));
                CHECK(center[2] == Approx(0.000_a).margin(1e-6));
	}
}
#endif
