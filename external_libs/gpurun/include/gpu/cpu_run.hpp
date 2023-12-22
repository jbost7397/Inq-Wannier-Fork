/* -*- indent-tabs-mode: t -*- */

#ifndef GPURUN__GPU__CPU_RUN
#define GPURUN__GPU__CPU_RUN

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

namespace cpu {

template <class kernel_type>
void run(kernel_type kernel){
	kernel();
}

template <class kernel_type>
void run(size_t size, kernel_type kernel){
		for(size_t ii = 0; ii < size; ii++) kernel(ii);
}

template <class kernel_type>
void run(size_t sizex, size_t sizey, kernel_type kernel){
	for(size_t iy = 0; iy < sizey; iy++){
		for(size_t ix = 0; ix < sizex; ix++){
			kernel(ix, iy);
		}
	}
}

template <class kernel_type>
void run(size_t sizex, size_t sizey, size_t sizez, kernel_type kernel){

	for(size_t iz = 0; iz < sizez; iz++){
		for(size_t iy = 0; iy < sizey; iy++){
			for(size_t ix = 0; ix < sizex; ix++){
				kernel(ix, iy, iz);
			}
		}
	}
}
 
template <class kernel_type>
void run(size_t sizex, size_t sizey, size_t sizez, size_t sizew, kernel_type kernel){

	if(sizex == 1){
		run(sizey, sizez, sizew, [kernel](auto iy, auto iz, auto iw){ kernel(0, iy, iz, iw); });
		return;
	}
	
	for(size_t iw = 0; iw < sizew; iw++){
		for(size_t iz = 0; iz < sizez; iz++){
			for(size_t iy = 0; iy < sizey; iy++){
				for(size_t ix = 0; ix < sizex; ix++){
					kernel(ix, iy, iz, iw);
				}
			}
		}
	}
}

}
#endif

#ifdef GPURUN__CPU_RUN__UNIT_TEST
#undef GPURUN__CPU_RUN__UNIT_TEST

#include <gpu/array.hpp>
#include <mpi3/environment.hpp>
#include <catch2/catch_all.hpp>
#include <gpu/atomic.hpp>

namespace cpu {

static long check_run(long size){
	
	gpu::array<long, 1> list(size, 0l);

	gpu::run(size,
					 [itlist = begin(list)] (auto ii){
						 gpu::atomic::add(&(itlist[ii]), ii + 1);
					 });
	
	long diff = 0;
	for(long ii = 0; ii < size; ii++) {
		diff += ii + 1 - list[ii];
	}
	return diff;
}

static long check_run(long size1, long size2){
	
	gpu::array<long, 3> list({size1, size2, 2}, 0l);
	
	gpu::run(size1, size2, 
					 [itlist = begin(list)] (auto ii, auto jj){
						 gpu::atomic::add(&(itlist[ii][jj][0]), ii + 1);
						 gpu::atomic::add(&(itlist[ii][jj][1]), jj + 1);
					 });
	
	long diff = 0;
	for(long ii = 0; ii < size1; ii++) {
		for(long jj = 0; jj < size2; jj++) {
			diff += ii + 1 - list[ii][jj][0];
			diff += jj + 1 - list[ii][jj][1];
		}
	}
		
	return diff;
}

static long check_run(long size1, long size2, long size3){
	
	gpu::array<long, 4> list({size1, size2, size3, 3}, 0l);

	gpu::run(size1, size2, size3,
					 [itlist = begin(list)] (auto ii, auto jj, auto kk){
						 gpu::atomic::add(&(itlist[ii][jj][kk][0]), ii + 1);
						 gpu::atomic::add(&(itlist[ii][jj][kk][1]), jj + 1);
						 gpu::atomic::add(&(itlist[ii][jj][kk][2]), kk + 1);
					 });
		
	long diff = 0;
	for(long ii = 0; ii < size1; ii++) {
		for(long jj = 0; jj < size2; jj++) {
			for(long kk = 0; kk < size3; kk++) {
				diff += ii + 1 - list[ii][jj][kk][0];
				diff += jj + 1 - list[ii][jj][kk][1];
				diff += kk + 1 - list[ii][jj][kk][2];
			}
		}
	}

	return diff;
}
	
static long check_run(long size1, long size2, long size3, long size4){

	gpu::array<long, 5> list({size1, size2, size3, size4, 4}, 0l);

	gpu::run(size1, size2, size3, size4,
					 [itlist = begin(list)] (auto ii, auto jj, auto kk, auto ll){
						 gpu::atomic::add(&(itlist[ii][jj][kk][ll][0]), ii + 1);
						 gpu::atomic::add(&(itlist[ii][jj][kk][ll][1]), jj + 1);
						 gpu::atomic::add(&(itlist[ii][jj][kk][ll][2]), kk + 1);
						 gpu::atomic::add(&(itlist[ii][jj][kk][ll][3]), ll + 1);
					 });
		
	long diff = 0;
	for(long ii = 0; ii < size1; ii++) {
		for(long jj = 0; jj < size2; jj++) {
			for(long kk = 0; kk < size3; kk++) {
				for(long ll = 0; ll < size4; ll++) {
					diff += ii + 1 - list[ii][jj][kk][ll][0];
					diff += jj + 1 - list[ii][jj][kk][ll][1];
					diff += kk + 1 - list[ii][jj][kk][ll][2];
					diff += ll + 1 - list[ii][jj][kk][ll][3];
				}
			}
		}
	}

	return diff;
}

}

TEST_CASE(GPURUN_TEST_FILE, GPURUN_TEST_TAG) {

	using namespace Catch::literals;

	SECTION("1D"){
		CHECK(cpu::check_run(200) == 0);
		CHECK(cpu::check_run(1024) == 0);
		CHECK(cpu::check_run(6666) == 0);
	}
	
	SECTION("2D"){
		CHECK(cpu::check_run(200, 200) == 0);
		CHECK(cpu::check_run(256, 1200) == 0);
		CHECK(cpu::check_run(2023, 4) == 0);
		CHECK(cpu::check_run(7, 57*57*57) == 0);
	}

	SECTION("3D"){
		CHECK(cpu::check_run(2, 2, 2) == 0);
		CHECK(cpu::check_run(7, 2, 2) == 0);
		CHECK(cpu::check_run(7, 57, 57) == 0);
		CHECK(cpu::check_run(32, 23, 18) == 0);
		CHECK(cpu::check_run(213, 27, 78) == 0);
		CHECK(cpu::check_run(2500, 10, 12) == 0);
		CHECK(cpu::check_run(7, 1023, 12) == 0);	
		CHECK(cpu::check_run(1, 11, 1229) == 0);	
	}
	
	SECTION("4D"){
		CHECK(cpu::check_run(2, 2, 2, 2) == 0);
		CHECK(cpu::check_run(7, 2, 2, 2) == 0);
		CHECK(cpu::check_run(7, 57, 57, 57) == 0);
		CHECK(cpu::check_run(32, 23, 45, 18) == 0);
		CHECK(cpu::check_run(35, 213, 27, 78) == 0);
		CHECK(cpu::check_run(2500, 10, 11, 12) == 0);
		CHECK(cpu::check_run(7, 1023, 11, 12) == 0);
		CHECK(cpu::check_run(1, 1, 11, 1229) == 0);
		CHECK(cpu::check_run(1, 1023, 11, 12) == 0);
	}

}

#endif
