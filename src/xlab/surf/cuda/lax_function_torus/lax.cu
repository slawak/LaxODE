// Copyright (C) 2012 Nicholas Schmitt
// Copyright (C) 2012 Wjatscheslaw Kewlin
// Reworked by W. Kewlin to use on CUDA

#ifndef xlab_surf_cuda_lax_function_torus_lax_cu_INCLUDED
#define xlab_surf_cuda_lax_function_torus_lax_cu_INCLUDED

#include "lax.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

namespace {

//--------------------------------------------------
// set_A, set_B, set_C and their specializations
// compute individual entries of dX[k]
//--------------------------------------------------

/*
  Computes the upper-right entry of dX[k].
  Generic case.
*/
__host__ __device__ inline
void
set_A(
    uint const &g,
    uint const &k,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 2);
  XLAB_ASSERT(k >= 2);
  XLAB_ASSERT(k <= g-1);

  int const m = 3*(g-k);
  dX[0+m] = 2.0 *(X[4+m]*V[0] - X[5+m]*V[1] - X[-2+m]*V[2] - X[-1+m]*V[3] + X[1+m]*V[4]);
  dX[1+m] = 2.0 *(X[5+m]*V[0] + X[4+m]*V[1] - X[-1+m]*V[2] + X[-2+m]*V[3] - X[0+m]*V[4]);
}

/*
  Computes the lower-left entry of dX[k].
  Generic case.
*/
__host__ __device__ inline
void
set_B(
    uint const &g,
    uint const &k,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 2);
  XLAB_ASSERT(k >= 2);
  XLAB_ASSERT(k <= g-1);

  int const m = 3*(g-k) + 2;
  dX[0+m] = 2.0 *(X[-4+m]*V[0] + X[-3+m]*V[1] - X[2+m]*V[2] + X[3+m]*V[3] - X[1+m]*V[4]);
  dX[1+m] = 2.0 *(X[-3+m]*V[0] - X[-4+m]*V[1] - X[3+m]*V[2] - X[2+m]*V[3] + X[0+m]*V[4]);
}

/*
  Computes the upper-left entry of dX[k].
  Generic case.
*/
__host__ __device__ inline
void
set_C(
    uint const &g,
    uint const &k,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 2);
  XLAB_ASSERT(k >= 2);
  XLAB_ASSERT(k <= g-1);

  int const m = 3*(g-k) + 1;
  dX[0+m] = (-X[-4+m] - X[4+m])*V[0] + (-X[-3+m] + X[5+m])*V[1] + (X[-2+m] + X[2+m])*V[2] + ( X[-1+m] - X[3+m])*V[3];
  dX[1+m] = (-X[-3+m] - X[5+m])*V[0] + ( X[-4+m] - X[4+m])*V[1] + (X[-1+m] + X[3+m])*V[2] + (-X[-2+m] + X[2+m])*V[3];
}

/*
  Computes the upper-right entry of dX[g].
  Special case: k == g.
*/
__host__ __device__ inline
void
set_A_g(
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  dX[0] = 2.0 * (X[4]*V[0] - X[5]*V[1] + X[1]*V[4]);
  dX[1] = 2.0 * (X[5]*V[0] + X[4]*V[1] - X[0]*V[4]);
}

/*
  Computes the lower-left entry of dX[g].
  Special case: k == g.
*/
__host__ __device__ inline
void
set_B_g(
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  dX[2] = 2.0 * (-X[4]*V[2] + X[5]*V[3] - X[3]*V[4]);
  dX[3] = 2.0 * (-X[5]*V[2] - X[4]*V[3] + X[2]*V[4]);
}

/*
  Computes the upper-right entry of dX[1].
  Special case: k == 1.
*/
__host__ __device__ inline
void
set_A_1(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 3);
  XLAB_ASSERT(g % 2 == 1);

  int const m = 3*g - 3;
  dX[0+m] = 2.0 * (-X[4+m]*V[1] - X[-2+m]*V[2] - X[-1+m]*V[3] + X[1+m]*V[4]);
  dX[1+m] = 2.0 * ( X[4+m]*V[0] - X[-1+m]*V[2] + X[-2+m]*V[3] - X[0+m]*V[4]);
}

/*
  Computes the lower-left entry of dX[1].
  Special case: k == 1.
*/
__host__ __device__ inline
void
set_B_1(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 3);
  XLAB_ASSERT(g % 2 == 1);

  int const m = 3*g - 1;
  dX[0+m] = 2.0 * (X[-4+m]*V[0] + X[-3+m]*V[1] + X[2+m]*V[3] - X[1+m]*V[4]);
  dX[1+m] = 2.0 * (X[-3+m]*V[0] - X[-4+m]*V[1] - X[2+m]*V[2] + X[0+m]*V[4]);
}

/*
  Computes the upper-left entry of dX[1].
  Special case: k == 1.
*/
__host__ __device__ inline
void
set_C_1(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 2);
  XLAB_ASSERT(g % 2 == 0);

  int const m = 3*g - 2;
  dX[0+m] = (-X[-4+m] + X[2+m])*V[0] + (-X[-3+m] + X[3+m])*V[1] + (X[-2+m] + X[2+m])*V[2] + ( X[-1+m] - X[3+m])*V[3];
  dX[1+m] = (-X[-3+m] - X[3+m])*V[0] + ( X[-4+m] + X[2+m])*V[1] + (X[-1+m] + X[3+m])*V[2] + (-X[-2+m] + X[2+m])*V[3];
}

/*
  Computes the upper-right entry of dX[1].
  Special case: k == 0.
*/
__host__ __device__ inline
void
set_A_0(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 2);
  XLAB_ASSERT(g % 2 == 0);

  int const m = 3*g;
  dX[0+m] = 2.0 * (-X[-2+m]*V[0] - X[-1+m]*V[1] - X[-2+m]*V[2] - X[-1+m]*V[3] + X[1+m]*V[4]);
  dX[1+m] = 2.0 * ( X[-1+m]*V[0] - X[-2+m]*V[1] - X[-1+m]*V[2] + X[-2+m]*V[3] - X[0+m]*V[4]);
}

/*
  Computes the upper-left entry of dX[1].
  Special case: k == 0.
*/
__host__ __device__ inline
void
set_C_0(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 3);
  XLAB_ASSERT(g % 2 == 1);

  int const m = 3*g + 1;
  dX[0+m] = 2.0 *(-X[-3+m]*V[0] + X[-4+m]*V[1] + X[-1+m]*V[2] - X[-2+m]*V[3]);
}

//--------------------------------------------------
// check_sizes
// check_sizess the sizes of dX, X, V for a specified g
//--------------------------------------------------

__host__ __device__ inline
void
check_sizes(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT( dX.size() == 3 * g + 2 );
  XLAB_ASSERT( X.size() == 3 * g + 2 );
  XLAB_ASSERT( V.size() == 5 );
}

//--------------------------------------------------
// lax and its specializations
// compute the lax equations
//--------------------------------------------------
/*
  Computes the Lax equation for the case g == 0.

  @param dX The differential of the polynomial Killing field. Must have length 3g+2.
  @param X The polynomial Killing field. Must have length 3g+2.
  @param V The potential.
*/
__host__ __device__ inline
void
lax_0(
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  check_sizes(0, dX, X, V);

  /*
    Note: for the case g == 0, the formal equations are
      dX[0] = 2.0 * X[1] * V[4]
      dX[1] = -2.0 * X[1] * V[4].
    Since V[4] == 0, then dX[0] = dX[1] = 0.
  */

  // upper-right entry of dX[0] (off-diagonal)
  dX[0] = 0.0;
  dX[1] = 0.0;
}

/*
  Computes the Lax equation for the case g == 1.

  @param dX The differential of the polynomial Killing field. Must have length 3g+2.
  @param X The polynomial Killing field. Must have length 3g+2.
  @param V The potential.
*/
__host__ __device__ inline
void
lax_1(
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  check_sizes(1, dX, X, V);

  // upper-right entry of dX[1] (off-diagonal)
  dX[0] = 2.0 * (-X[4]*V[1] + X[1]*V[4]);
  dX[1] = 2.0 * ( X[4]*V[0] - X[0]*V[4]);

  // lower-left entry of dX[1] (off-diagonal)
  dX[2] = 2.0 * ( X[4]*V[3] - X[3]*V[4]);
  dX[3] = 2.0 * (-X[4]*V[2] + X[2]*V[4]);

  // upper left entry of dX[0] (diagonal)
  dX[4] = 2.0 * (-X[1]*V[0] + X[0]*V[1] + X[3]*V[2] - X[2]*V[3]);
}

/*
  Computes the Lax equation for the case g is even and g >= 2.

  @param g The spectral genus. Must be even and >= 2.
  @param dX The differential of the polynomial Killing field. Must have length 3g+2.
  @param X The polynomial Killing field. Must have length 3g+2.
  @param V The potential.
*/
__host__ __device__ inline
void
lax_even(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 2);
  XLAB_ASSERT(g % 2 == 0);

  check_sizes(g, dX, X, V);

  // compute dX[g] (off-diagonal)
  set_A_g(dX, X, V);
  set_B_g(dX, X, V);

  // compute dX[g-1],...,dX[2]
  for (uint k = g-1; k >= 3; k -= 2)
  {
    // compute dX[k] (diagonal)
    set_C(g, k, dX, X, V);
    // compute dX[k-1] (off-diagonal)
    set_A(g, k-1, dX, X, V);
    set_B(g, k-1, dX, X, V);
  }

  // compute dX[1] (diagonal)
  set_C_1(g, dX, X, V);

  // compute dX[0] (off-diagonal)
  set_A_0(g, dX, X, V);
}

/*
  Computes the Lax equation for the case g is odd and g >= 3.

  @param g The spectral genus. Must be odd and >= 3.
  @param dX The differential of the polynomial Killing field. Must have length 3g+2.
  @param X The polynomial Killing field. Must have length 3g+2.
  @param V The potential.
*/
__host__ __device__ inline
void
lax_odd(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  XLAB_ASSERT(g >= 3);
  XLAB_ASSERT(g % 2 == 1);

  check_sizes(g, dX, X, V);

  // compute dX[g] (off-diagonal)
  set_A_g(dX, X, V);
  set_B_g(dX, X, V);

  // compute dX[g-1],...,dX[3]
  for (uint k = g-1; k >= 4; k -= 2)
  {
    // compute dX[k] (diagonal)
    set_C(g, k, dX, X, V);
    // compute dX[k-1] (off-diagonal)
    set_A(g, k-1, dX, X, V);
    set_B(g, k-1, dX, X, V);
  }

  // compute dX[2] (diagonal)
  set_C(g, 2, dX, X, V);

  // compute dX[1] (off-diagonal)
  set_A_1(g, dX, X, V);
  set_B_1(g, dX, X, V);

  // compute dX[0] (diagonal)
  set_C_0(g, dX, X, V);
}

} // namespace

//--------------------------------------------------
// lax
// compute the lax equation for arbitrary g
//--------------------------------------------------
__host__ __device__ inline
void
lax(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    array_view<value_type_real const> const &V )
{
  check_sizes(g, dX, X, V);

  if (g % 2 == 0)
  {
    // g even
    if (g == 0)
    {
      lax_0(dX, X, V);
    }
    else
    {
      lax_even(g, dX, X, V);
    }
  }
  else
  {
    // g odd
    if (g == 1)
    {
      lax_1(dX, X, V);
    }
    else
    {
      lax_odd(g, dX, X, V);
    }
  }
}


} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_lax_cu_INCLUDED
