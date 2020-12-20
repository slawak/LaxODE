// Copyright (C) 2012 Nicholas Schmitt
// Copyright (C) 2012 Wjatscheslaw Kewlin
// Reworked by W. Kewlin to use on CUDA

#ifndef xlab_surf_cuda_lax_function_torus_laxR3_cu_INCLUDED
#define xlab_surf_cuda_lax_function_torus_laxR3_cu_INCLUDED

#include "lax.hpp"
#include "potential.hpp"
#include "frameR3.hpp"
#include "laxR3.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

__host__ __device__ inline
void laxR3(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    value_type_real const &dx,
    value_type_real const &dy,
    value_type_real const &sym_re,
    value_type_real const &sym_im,
    array_view<value_type_real> Vtemp)
{
  XLAB_ASSERT(X.size() == dimension_R3(g));
  XLAB_ASSERT(dX.size() == dimension_R3(g));

  std::size_t const X_dim = dimension_lax(g);

  // compute V in packed form {V[0],V[1],V[2],V[3],V[4]}
  compute_V(
      g,
      Vtemp,
      array_view<value_type_real const>( X, X_dim ),
      dx, dy);

  // compute the lax equation dX = [X, V(X)]
  lax(
      g,
      array_view<value_type_real>( dX, X_dim ),
      array_view<value_type_real const>( X, X_dim ),
      Vtemp.getConst() );
  
  // compute the frame equations
  //     dF = F V
  //     dFdot = Fdot V + F Vdot
  frameR3(
      array_view<value_type_real>( dX , 8 , X_dim),
      array_view<value_type_real const>( X , 8 , X_dim),
      Vtemp.getConst(),
      sym_re,
      sym_im);
}

__host__ __device__ inline
void laxR3(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    value_type_real const &dx,
    value_type_real const &dy,
    value_type_real const &sym_re,
    value_type_real const &sym_im) {

	value_type_real V[5];
	laxR3(g, dX, X, dx, dy, sym_re, sym_im, array_view<value_type_real> (V, 5));
}

} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_laxR3_cu_INCLUDED
