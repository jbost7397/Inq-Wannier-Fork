/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MATH__ARRAY
#define INQ__MATH__ARRAY

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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <multi/array.hpp>

#ifdef HAVE_CUDA
#include <multi/memory/adaptors/cuda/managed/allocator.hpp>
#endif

namespace inq {
namespace math {

template <class type, size_t dim,
#ifdef HAVE_CUDA
					class allocator = boost::multi::memory::cuda::managed::allocator<type>
#else
					class allocator = std::allocator<type>
#endif
					>
using array = boost::multi::array<type, dim, allocator>;

}
}

#endif
