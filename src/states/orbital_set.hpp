/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__STATES__ORBITAL_SET
#define INQ__STATES__ORBITAL_SET

/*
 Copyright (C) 2019 Xavier Andrade

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

#include <basis/field_set.hpp>

namespace inq {
namespace states {
	
template<class Basis, class Type>
class orbital_set {

public:
	
	using element_type = Type;
	using basis_type = Basis;
	using kpoint_type = math::vector3<double, math::covariant>;
	using internal_array_type = math::array<Type, 2>;
	
	orbital_set(Basis const & basis, int const num_vectors, kpoint_type const & kpoint, int spin_index, parallel::cartesian_communicator<2> comm)
		:full_comm_(std::move(comm)),
		 set_comm_(basis::set_subcomm(full_comm_)),
		 set_part_(num_vectors, set_comm_),
		 matrix_({basis.part().local_size(), set_part_.local_size()}),
		 num_vectors_(num_vectors),
		 basis_(basis),
		 kpoint_(kpoint),
		 spin_index_(spin_index){
		prefetch();
		assert(basis_.part().comm_size() == basis::basis_subcomm(full_comm_).size());
		assert(local_set_size() > 0);
	}
	
	template <class any_type>
	orbital_set(inq::utils::skeleton_wrapper<orbital_set<Basis, any_type>> const & skeleton)
		:orbital_set(skeleton.base.basis(), skeleton.base.set_size(), skeleton.base.kpoint(), skeleton.base.spin_index(), skeleton.base.full_comm()){
	}
	
	// Avoid the default copy constructor since the multi copy constructor is slow
	//		orbital_set(const orbital_set & coeff) = default;
	orbital_set(orbital_set const & other)
			:orbital_set(other.skeleton()){
		matrix_ = other.matrix_;
	}
	
	orbital_set(orbital_set && coeff) = default;
	
	orbital_set(orbital_set && oldset, parallel::cartesian_communicator<2> new_comm):
		orbital_set(Basis{Basis{oldset.basis()}, basis::basis_subcomm(new_comm)}, oldset.set_size(), oldset.kpoint(), oldset.spin_index(), new_comm)
	{
		math::array<int, 1> rem_points(basis().local_size());
		math::array<int, 1> rem_states(local_set_size());
		for(long ip = 0; ip < basis().local_size(); ip++) rem_points[ip] = basis().part().local_to_global(ip).value();
		for(long ist = 0; ist < local_set_size(); ist++) rem_states[ist] = set_part().local_to_global(ist).value();
		matrix_ = parallel::get_remote_points(oldset, rem_points, rem_states);
	}

	orbital_set & operator=(orbital_set && coeff) = default;
	orbital_set & operator=(orbital_set const & coeff) = default;	

	auto skeleton() const {
		return inq::utils::skeleton_wrapper<orbital_set<Basis, Type>>(*this);
	}
	
	template <class OtherType>
	static auto reciprocal(inq::utils::skeleton_wrapper<orbital_set<typename basis_type::reciprocal_space, OtherType>> const & skeleton){
		return orbital_set<basis_type, element_type>(skeleton.base.basis().reciprocal(), skeleton.base.set_size(),  skeleton.base.kpoint(), skeleton.base.spin_index(), skeleton.base.full_comm());
	}
	
	internal_array_type & matrix() {
		return matrix_;
	}
	
	internal_array_type const & matrix() const{
		return matrix_;
	}
	
	auto data() const {
		return raw_pointer_cast(matrix_.data_elements());
	}
	
	auto data() {
		return raw_pointer_cast(matrix_.data_elements());
	}
	
	auto num_elements() const {
		return matrix_.num_elements();
	}
	
	template <typename ScalarType>
	void fill(ScalarType const & scalar) {
		CALI_CXX_MARK_SCOPE("fill(orbital_set)");
		
		gpu::run(matrix_.num_elements(), [lin = raw_pointer_cast(matrix_.data_elements()), scalar] GPU_LAMBDA (auto ii){
			lin[ii] = scalar;
		});
	}
	
	const basis_type & basis() const {
		return basis_;
	}
	
	const int & set_size() const {
		return num_vectors_;
	}
	
	auto local_set_size() const {
		return set_part_.local_size();
	}
		
	auto & set_part() const {
		return set_part_;
	}
	
	auto & set_comm() const {
		return set_comm_;
	}
				
	auto & full_comm() const {
		return full_comm_;
	}
	
	auto hypercubic() const {
		return matrix_.partitioned(basis_.cubic_dist(1).local_size()*basis_.cubic_dist(0).local_size()).partitioned(basis_.cubic_dist(0).local_size());
	}
	
	auto hypercubic() {
		return matrix_.partitioned(basis_.cubic_dist(1).local_size()*basis_.cubic_dist(0).local_size()).partitioned(basis_.cubic_dist(0).local_size());
	}
	
	void prefetch() const {
		math::prefetch(matrix_);
	}
	
	auto & kpoint() const {
		return kpoint_;
	}
	
	auto & spin_index() const {
			assert(spin_index_ >= 0 and spin_index_ < 2);
			return spin_index_;
	}

	class parallel_set_iterator {
		
		internal_array_type matrix_;
		int istep_;
		mutable parallel::cartesian_communicator<1> set_comm_;
		parallel::partition set_part_;
		
	public:
		
		parallel_set_iterator(long basis_local_size, parallel::partition set_part, parallel::cartesian_communicator<1> set_comm, internal_array_type const & data):
			matrix_({basis_local_size, set_part.block_size()}),
			istep_(0),
			set_comm_(std::move(set_comm)),
			set_part_(std::move(set_part)){
			
			CALI_CXX_MARK_SCOPE("field_set_iterator_constructor");
			
			gpu::copy(basis_local_size, set_part.local_size(), data, matrix_);
		};
		
		void operator++(){
			
			CALI_CXX_MARK_SCOPE("field_set_iterator++");
			
			auto mpi_type = boost::mpi3::detail::basic_datatype<element_type>();
			
			auto next_proc = set_comm_.rank() + 1;
			if(next_proc == set_comm_.size()) next_proc = 0;
			auto prev_proc = set_comm_.rank() - 1;
			if(prev_proc == -1) prev_proc = set_comm_.size() - 1;
			
			if(istep_ < set_comm_.size() - 1) {  //there is no need to copy for the last step
				
				set_comm_.nccl_init();
#ifdef ENABLE_NCCL
				ncclGroupStart();
				auto copy = matrix_;
				ncclRecv(raw_pointer_cast(matrix_.data_elements()), matrix_.num_elements()*sizeof(type)/sizeof(double), ncclDouble, next_proc, &set_comm_.nccl_comm(), 0);
				ncclSend(raw_pointer_cast(copy.data_elements()), matrix_.num_elements()*sizeof(type)/sizeof(double), ncclDouble, prev_proc, &set_comm_.nccl_comm(), 0);
				ncclGroupEnd();
				gpu::sync();
#else
				MPI_Sendrecv_replace(raw_pointer_cast(matrix_.data_elements()), matrix_.num_elements(), mpi_type, prev_proc, istep_, next_proc, istep_, set_comm_.get(), MPI_STATUS_IGNORE);
#endif
			}
			
			istep_++;
		}
		
		bool operator!=(int it_istep){
			return istep_ != it_istep;
		}
		
		auto matrix() const {
			return matrix_(boost::multi::ALL, {0, set_part_.local_size(set_ipart())});
		}

		auto set_ipart() const {
			auto ip = istep_ + set_comm_.rank();
			if(ip >= set_comm_.size()) ip -= set_comm_.size();
			return ip;
		}
		
	};

	auto par_set_begin() const {
		return parallel_set_iterator(basis().local_size(), set_part_, set_comm_, matrix());
	}
	
	auto par_set_end() const {
		return set_comm_.size();
	}
	
private:

	mutable parallel::cartesian_communicator<2> full_comm_;
	mutable parallel::cartesian_communicator<1> set_comm_;
	inq::parallel::partition set_part_;
	internal_array_type matrix_;
	int num_vectors_;
	basis_type basis_;
	kpoint_type kpoint_;
	int spin_index_;
		
};

}
}

#ifdef INQ_STATES_ORBITAL_SET_UNIT_TEST
#undef INQ_STATES_ORBITAL_SET_UNIT_TEST

#include <basis/real_space.hpp>

#include <ions/unit_cell.hpp>
#include <catch2/catch_all.hpp>

#include <parallel/communicator.hpp>

TEST_CASE("Class states::orbital_set", "[states::orbital_set]"){
  
	using namespace inq;
	using namespace inq::magnitude;	
	using namespace Catch::literals;
  using math::vector3;
  
  auto ecut = 40.0_Ha;

	auto comm = boost::mpi3::environment::get_world_instance();

	parallel::cartesian_communicator<2> cart_comm(comm, {});

	auto set_comm = basis::set_subcomm(cart_comm);
	auto basis_comm = basis::basis_subcomm(cart_comm);	

	systems::box box = systems::box::orthorhombic(10.0_b, 4.0_b, 7.0_b).cutoff_energy(ecut);
  basis::real_space rs(box, basis_comm);

	states::orbital_set<basis::real_space, double> orb(rs, 12, math::vector3<double, math::covariant>{0.0, 0.0, 0.0}, 0, cart_comm);

	CHECK(sizes(orb.basis())[0] == 28);
	CHECK(sizes(orb.basis())[1] == 11);
	CHECK(sizes(orb.basis())[2] == 20);

	CHECK(orb.local_set_size() == orb.local_set_size());
	CHECK(orb.set_size() == orb.set_size());
	
	states::orbital_set<basis::real_space, double> orbk(rs, 12, {0.4, 0.22, -0.57}, 0, cart_comm);

	CHECK(sizes(orbk.basis())[0] == 28);
	CHECK(sizes(orbk.basis())[1] == 11);
	CHECK(sizes(orbk.basis())[2] == 20);

	CHECK(orbk.kpoint()[0] == 0.4_a);
	CHECK(orbk.kpoint()[1] == 0.22_a);
	CHECK(orbk.kpoint()[2] == -0.57_a);

	CHECK(orbk.local_set_size() == orb.local_set_size());
	CHECK(orbk.set_size() == orb.set_size());

	states::orbital_set<basis::real_space, double> orb_copy(orbk.skeleton());

	CHECK(sizes(orb_copy.basis()) == sizes(orbk.basis()));
	CHECK(orb_copy.kpoint() == orbk.kpoint());
	CHECK(orb_copy.local_set_size() == orbk.local_set_size());
	CHECK(orb_copy.set_size() == orbk.set_size());
	
}

#endif

#endif
