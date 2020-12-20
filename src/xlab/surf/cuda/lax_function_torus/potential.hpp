// Copyright (C) 2012 Nicholas Schmitt
// Copyright (C) 2012 Wjatscheslaw Kewlin
// Reworked by W. Kewlin to use on CUDA

#ifndef xlab_surf_cuda_lax_function_torus_potential_INCLUDED
#define xlab_surf_cuda_lax_function_torus_potential_INCLUDED

#include "definitions.hpp"
#include "array_view.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

/**
  Computes V = {V[1], V[2], V[3], V[4], V[5]} from X,
  in form approriate for lax().
*/
extern
__host__ __device__
void
compute_V(
    uint const &g,
    array_view<value_type_real> V,
    array_view<value_type_real const> const &X,
    value_type_real const &dx,
    value_type_real const &dy );

} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_potential_INCLUDED
