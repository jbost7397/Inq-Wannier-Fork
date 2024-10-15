/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__WANNIER__BASIS_MAPPING
#define INQ__WANNIER__BASIS_MAPPING

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>
#include <cmath>
#include <vector>
#include <math/complex.hpp>
#include <parallel/arbitrary_partition>

//JB: for Wannierization, basically all functionality from Qbach's BasisMapping class is required, including this particular
//    block_cyclic distribution system + accompanying helper functions. Certain aspects of code from Qbach lack a direct equivalent
//    so still need to strategize on implementing those

namespace inq {
namespace wannier {

class block_cyclic_partition : public parallel::arbitrary_partition {
public:
    block_cyclic_partition(long size, int comm_size, int comm_rank, int block_size) :
        parallel::arbitrary_partition(0, comm_size, comm_rank) {  

        int num_blocks = (size + block_size - 1) / block_size;
        int blocks_per_proc = (num_blocks + comm_size - 1) / comm_size;
        local_size_ = std::min((long)blocks_per_proc * block_size, size - (long)comm_rank * blocks_per_proc * block_size);
        lsizes_.resize(comm_size);
        for (int i = 0; i < comm_size; i++) {
            int num_blocks_proc = (num_blocks + comm_size - 1) / comm_size;
            if (i < num_blocks % comm_size) num_blocks_proc++;
            lsizes_[i] = std::min((long)num_blocks_proc * block_size, size - (long)i * num_blocks_proc * block_size);

        }
        size_ = size;
        start_ = 0;
        for (int i = 0; i < comm_rank; ++i) start_ += lsizes_[i];
        end_ = start_ + local_size_;
    }


};

std::vector<int> generate_ipack(const block_cyclic_partition& partition, int nvec, int np2) {
    std::vector<int> ipack;
    ipack.reserve(nvec * np2); 

    long global_index = 0;
    for (int iproc = 0; iproc < comm_.size(); ++iproc) {
        for (int ivec = 0; ivec < nvec; ++ivec) {
            for (long i = partition.start(iproc); i < partition.end(iproc); ++i) {
                ipack.push_back(global_index);
                global_index++;
            }
        }
    }
    return ipack;
}

//need to recover functionality from Qb@ll's 'rod' structure... need to think about equivalent functionality in Inq
//implementation for when the basis is real
std::vector<int> generate_iunpack(const block_cyclic_partition& partition, int np0, int np1, int np2, int nvec) {
    std::vector<int> iunpack;
    iunpack.reserve((2 * basis_.nrods() -1) * partition.local_size());

    for (int l = 0; l < partition.local_size(); ++l) {
        iunpack.push_back(l * np0 * np1); 
    }
    int isource_p = partition.local_size();
    int isource_m = 2 * partition.local_size();


    for (int irod = 1; irod < basis_.nrod_loc(0); irod++) {
        int hp = basis_.rod_h(0, irod);
        int kp = basis_.rod_k(0, irod);
        if (hp < 0) hp += np0;
        if (kp < 0) kp += np1;

        int hm = -hp;
        int km = -kp;
        if (hm < 0) hm += np0;
        if (km < 0) km += np1;

        for (int l = 0; l < partition.local_size(); ++l) {

            int idest_p = hp + np0 * (kp + np1 * l);
            //iunpack_[isource_p+l] = idest_p;
            iunpack.push_back(idest_p);

            int idest_m = hm + np0 * (km + np1 * l);
            //iunpack_[isource_m+l] = idest_m;
            iunpack.push_back(idest_m);
        }
    }

    for (int iproc = 1; iproc < comm_.size(); iproc++) {
        for (int irod = 0; irod < basis_.nrod_loc(iproc); irod++) {
            int hp = basis_.rod_h(iproc, irod);
            int kp = basis_.rod_k(iproc, irod);
            if (hp < 0) hp += np0;
            if (kp < 0) kp += np1;

            int hm = -hp;
            int km = -kp;
            if (hm < 0) hm += np0;
            if (km < 0) km += np1;

            for (int l = 0; l < partition.local_size(); ++l) {
                int idest_p = hp + np0 * (kp + np1 * l);
                iunpack.push_back(idest_p);

                int idest_m = hm + np0 * (km + np1 * l);
                iunpack.push_back(idest_m);
            }
        //isource_p += 2 * np2_loc_[myproc_];
	isource_p += 2 * partition.local_size();
        //isource_m += 2 * np2_loc_[myproc_];
	isource_m += 2 * partition.local_size();
        }
    }
    return iunpack;
}

//case when the basis is complex
std::vector<int> generate_iunpack(const block_cyclic_partition& partition, int np0, int np1, int np2, int nvec) {
    std::vector<int> iunpack;
    iunpack.reserve(basis_.nrods() * partition.local_size());

    int isource = 0;
    for (int iproc = 0; iproc < comm_.size(); ++iproc) {
        for (int irod = 0; irod < basis_.nrod_loc(iproc); ++irod) {
            int h = basis_.rod_h(iproc, irod);
            int k = basis_.rod_k(iproc, irod);
            if (h < 0) h += np0;
            if (k < 0) k += np1;

            for (long l = partition.start(iproc); l < partition.end(iproc); ++l) {
                int idest = h + np0 * (k + np1 * l);
		//iunpack_[isource+l] = idest;
                iunpack.push_back(idest);
            }
	    //isource += np2_loc_[myproc_];
	    isource += partition.local_size();
        }
    }
    return iunpack;
}

void TDMLWFTransform::transpose_fwd(const inq::gpu::array<std::complex<double>, 1>& zvec, inq::gpu::array<std::complex<double>, 1>& ct) {
    int np0 = basis_.sizes()[0];
    int np1 = basis_.sizes()[1];
    int np2 = basis_.sizes()[2];
    int nvec = ...; //how to get this in Inq?
    int block_size = ...; 


    block_cyclic_partition partition(np2, comm_.size(), comm_.rank(), block_size);

    std::vector<int> ipack = generate_ipack(partition, nvec, np2);
    std::vector<int> iunpack = generate_iunpack(partition, np0, np1, np2, nvec);

    inq::gpu::array<std::complex<double>, 1> packed_zvec(zvec.size());
    for (size_t i = 0; i < zvec.size(); ++i) {
        packed_zvec[ipack[i]] = zvec[i];
    }

    inq::gpu::array<std::complex<double>, 1> recv_buffer(ct.size());
    std::vector<int> scounts(comm_.size());
    std::vector<int> sdispl(comm_.size());
    std::vector<int> rcounts(comm_.size());
    std::vector<int> rdispl(comm_.size());

    for (int iproc = 0; iproc < comm_.size(); ++iproc) {
        int nvec_iproc;
        if (basis_.real()) {
            nvec_iproc = (iproc == 0) ? (2 * basis_.nrod_loc(iproc) - 1) : (2 * basis_.nrod_loc(iproc));
        }
        else {
            nvec_iproc = basis_.nrod_loc(iproc);
        }
        scounts[iproc] = 2 * nvec * partition.local_size(iproc);
        rcounts[iproc] = 2 * nvec_iproc * partition.local_size();
    }

    sdispl[0] = 0;
    rdispl[0] = 0;
    for (int iproc = 1; iproc < comm_.size(); ++iproc) {
        sdispl[iproc] = sdispl[iproc - 1] + scounts[iproc - 1];
        rdispl[iproc] = rdispl[iproc - 1] + rcounts[iproc - 1];
    }


    MPI_Alltoallv(packed_zvec.data(), scounts.data(), sdispl.data(), MPI_DOUBLE_COMPLEX, recv_buffer.data(), rcounts.data(), rdispl.data(), MPI_DOUBLE_COMPLEX, comm_);

    ct.resize(recv_buffer.size());
    for (size_t i = 0; i < recv_buffer.size(); ++i) {
        ct[iunpack[i]] = recv_buffer[i];
    }

}

void TDMLWFTransform::transpose_bwd(const inq::gpu::array<std::complex<double>, 1>& ct, inq::gpu::array<std::complex<double>, 1>& zvec) {
    int np0 = basis_.sizes()[0];
    int np1 = basis_.sizes()[1];
    int np2 = basis_.sizes()[2];
    int nvec = ...; 
    int block_size = ...; 


    block_cyclic_partition partition(np2, comm_.size(), comm_.rank(), block_size);

    std::vector<int> ipack = generate_ipack(partition, nvec, np2);
    std::vector<int> iunpack = generate_iunpack(partition, np0, np1, np2, nvec);

    inq::gpu::array<std::complex<double>, 1> packed_ct(ct.size());
    for (size_t i = 0; i < ct.size(); ++i) {
        packed_ct[iunpack[i]] = ct[i];
    }

    inq::gpu::array<std::complex<double>, 1> recv_buffer(zvec.size());
    std::vector<int> scounts(comm_.size());
    std::vector<int> sdispl(comm_.size());
    std::vector<int> rcounts(comm_.size());
    std::vector<int> rdispl(comm_.size());

    //Use the same scounts, sdispl, rcounts, and rdispl vectors as transpose_fwd_inq
    for (int iproc = 0; iproc < comm_.size(); ++iproc) {
        int nvec_iproc;
        if (basis_.real()) {
            nvec_iproc = (iproc == 0) ? (2 * basis_.nrod_loc(iproc) - 1) : (2 * basis_.nrod_loc(iproc));
        }
        else {
            nvec_iproc = basis_.nrod_loc(iproc);
        }
        scounts[iproc] = 2 * nvec * partition.local_size(iproc);
        rcounts[iproc] = 2 * nvec_iproc * partition.local_size();
    }

    sdispl[0] = 0;
    rdispl[0] = 0;
    for (int iproc = 1; iproc < comm_.size(); ++iproc) {
        sdispl[iproc] = sdispl[iproc - 1] + scounts[iproc - 1];
        rdispl[iproc] = rdispl[iproc - 1] + rcounts[iproc - 1];
    }

    MPI_Alltoallv(packed_ct.data(), scounts.data(), sdispl.data(), MPI_DOUBLE_COMPLEX, recv_buffer.data(), rcounts.data(), rdispl.data(), MPI_DOUBLE_COMPLEX, comm_);

    zvec.resize(recv_buffer.size());
    for (size_t i = 0; i < recv_buffer.size(); ++i) {
        zvec[ipack[i]] = recv_buffer[i];
    }

}

//these functions require some equivalent to ip_ and im_ as implemented in Qbach's BasisMapping class
//also, std::conj should be replaced by conj function written by Chris
void TDMLWFTransform::vector_to_zvec(const inq::gpu::array<std::complex<double>, 1>& c, inq::gpu::array<std::complex<double>, 1>& zvec) {
    int ng = basis_.localsize();
    int len = zvec.size();
    zvec.fill(0.0);

    gpu::run(ng, [&](int ig) {
        std::complex<double> val = c[ig];
        if (basis_.real()) {
            zvec[ip_[ig]] = val;
            zvec[im_[ig]] = std::conj(val);
        }
        else {
            zvec[ip_[ig]] = val;
        }
    });
}

void TDMLWFTransform::zvec_to_vector_inq(const inq::gpu::array<std::complex<double>, 1>& zvec, inq::gpu::array<std::complex<double>, 1>& c) {
    int ng = basis_.localsize();

    gpu::run(ng, [&](int ig) {
        c[ig] = zvec[ip_[ig]];
    });
}

}  // namespace wannier
}  // namespace inq

#endif  // INQ__WANNIER__BASIS_MAPPING
	
	
///////////////////////////////////////////////////////////////////
#ifdef INQ_WANNIER_BASIS_MAPPING_UNIT_TEST
#undef INQ_WANNIER_BASIS_MAPPING_UNIT_TEST

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

    using namespace inq;
    using namespace Catch::literals;
    using Catch::Approx;
}
#endif 

