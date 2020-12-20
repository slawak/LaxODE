// Copyright (C) 2012 Nicholas Schmitt
// Copyright (C) 2012 Wjatscheslaw Kewlin
// Reworked by W. Kewlin to use on CUDA

#ifndef xlab_surf_cuda_lax_function_torus_potential_cu_INCLUDED
#define xlab_surf_cuda_lax_function_torus_potential_cu_INCLUDED

#include "potential.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

__host__ __device__ inline
void
compute_V(
    uint const &g,
    array_view<value_type_real> V,
    array_view<value_type_real const> const &X,
    value_type_real const &dx,
    value_type_real const &dy )
{
  XLAB_ASSERT( X.size() == 3 * g + 2 );
  XLAB_ASSERT( V.size() == 5 );
  switch (g)
  {
    case 0:
      V[0] = -dy*X[0] - dx*X[1];
      V[1] =  dx*X[0] - dy*X[1];
      V[2] =  dy*X[0] - dx*X[1];
      V[3] = -dx*X[0] - dy*X[1];
      V[4] = 0;
      break;
    case 1:
      V[0] = -dy*X[0] - dx*X[1];
      V[1] =  dx*X[0] - dy*X[1];
      V[2] = -dy*X[2] - dx*X[3];
      V[3] =  dx*X[2] - dy*X[3];
      V[4] = -dy*X[4];
      break;
    default:
      V[0] = -dy*X[0] - dx*X[1];
      V[1] =  dx*X[0] - dy*X[1];
      V[2] = -dy*X[2] - dx*X[3];
      V[3] =  dx*X[2] - dy*X[3];
      V[4] =  dx*X[4] - dy*X[5];
      break;
  }      
}

} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_potential_cu_INCLUDED
