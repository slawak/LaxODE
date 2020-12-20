// Copyright (C) 2012 Nicholas Schmitt
// Copyright (C) 2012 Wjatscheslaw Kewlin
// Reworked by W. Kewlin to use on CUDA

#ifndef xlab_surf_cuda_lax_function_torus_frameR3_INCLUDED
#define xlab_surf_cuda_lax_function_torus_frameR3_INCLUDED

#include "definitions.hpp"
#include "array_view.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

/*
  The real data for the potential is {v[0], v[1], v[2], v[3], v[4]},
  The potential is V[-1]/lambda + V[0] + V[1]lambda, where
    V[1] = {{ 0, v[0]+v[1]i}, {v[2]+v[3]i, 0}}
    V[0] = {{ v[4]i, 0}, {0, -v[4]i}}


  The frame equations for R3 are, at the symmpoint

     dF = F V
     dFdot = Fdot V + F Vdot

  where dot is the derivative with respect to theta (lambda = exp(I theta)).
*/

/**
  Computes the derivatives (dF, dFdot) of (F, Fdot), where
  F is the frame and Fdot is its theta derivative, both evaluated at they sympoint.

  The sympoint must be on the unit circle.

  @param dF The pair (dF, dFdot). Must have length 8.
  @param F The pair (F, Fdot). Must have length 8.
  @param V The potential. Must have length 5.
*/
extern
__host__ __device__
void
frameR3(
    array_view<value_type_real> dF,
    array_view<value_type_real const> const &F,
    array_view<value_type_real const> const &V,
    value_type_real const &sym_re,
    value_type_real const &sym_im );

} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_frameR3_INCLUDED
