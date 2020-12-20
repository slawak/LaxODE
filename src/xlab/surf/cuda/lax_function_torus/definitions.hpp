// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_lax_function_torus_definitions_INCLUDED
#define xlab_surf_cuda_lax_function_torus_definitions_INCLUDED

#define XLAB_DEBUG

#include <assert.h>

#ifdef XLAB_DEBUG
	#define XLAB_ASSERT(assertion) assert((assertion))
#else
	#define XLAB_ASSERT(assertion) ((void) 0)
#endif

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

// change to float if your GPU does not support doubles
typedef double value_type_real;

extern
__host__ __device__ inline
std::size_t dimension_lax(uint const &g) { return 3*g+2;}

extern
__host__ __device__ inline
std::size_t dimension_frame() { return 8;}

extern
__host__ __device__ inline
std::size_t dimension_V() { return 5;}

extern
__host__ __device__ inline
std::size_t dimension_R3(uint const &g) { return dimension_lax(g) + dimension_frame();}

} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_definitions_INCLUDED
