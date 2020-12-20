// Copyright (C) 2012 Nicholas Schmitt
// Copyright (C) 2012 Wjatscheslaw Kewlin
// Reworked by W. Kewlin to use on CUDA

#ifndef xlab_surf_cuda_lax_function_torus_lax_INCLUDED
#define xlab_surf_cuda_lax_function_torus_lax_INCLUDED

#include "definitions.hpp"
#include "array_view.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

/*
  Polynomial Killing field packing:

  The polynomial Killing field X is of the form
     X = Sum[ X[k] lambda^k, {k, -g, g}]
  where g is the spectral genus.

  X has three symmetries:
    1. trace-free
    2. SU(2) for lambda on the unit circle
    3. Twisted if g is odd; antitwisted if g is even.

  The real data {a[0], ... a[3g+2]} is packed into X as
    X[g] = {{ 0, a[0]+a[1]i}, {a[2]+a[3]i, 0}}
    X[g-1] = {{ a[4]+a[5]i, 0}, {0, -a[4]-a[5]i}}
    X[g-2] = {{ 0, a[6]+a[7]i}, {a[8]+a[9]i, 0}}
    X[g-3] = {{ a[10]+a[11]i, 0}, {0, -a[10]-a[11]i}}
    ...


  Potential packing:

  The real data for the potential is {v[0], v[1], v[2], v[3], v[4]},
  The potential is V[-1]/lambda + V[0] + V[1]lambda, where
    V[1] = {{ 0, v[0]+v[1]i}, {v[2]+v[3]i, 0}}
    V[0] = {{ v[4]i, 0}, {0, -v[4]i}}


  The real dimension of the Lax equation is 3g+2.

*/


/**
  Computes the Lax equation for CMC tori in spaceforms.

  @param g The spectral genus. g >= 0.
  @param dX The differential of the polynomial Killing field. Must have length 3g+2.
  @param X The polynomial Killing field. Must have length 3g+2.
  @param V The potential.
*/
extern
__host__ __device__
void
lax(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V );

} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_lax_INCLUDED
