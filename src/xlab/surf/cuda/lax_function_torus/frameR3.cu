// Copyright (C) 2012 Nicholas Schmitt
// Copyright (C) 2012 Wjatscheslaw Kewlin
// Reworked by W. Kewlin to use on CUDA

#ifndef xlab_surf_cuda_lax_function_torus_frameR3_cu_INCLUDED
#define xlab_surf_cuda_lax_function_torus_frameR3_cu_INCLUDED

#include "frameR3.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

__host__ __device__ inline
void
frameR3(
    array_view<value_type_real> dF,
    array_view<value_type_real const> const &F,
    array_view<value_type_real const> const &V,
    value_type_real const &sym_re,
    value_type_real const &sym_im )
{
  XLAB_ASSERT( dF.size() == 8 );
  XLAB_ASSERT( F.size() == 8 );
  XLAB_ASSERT( V.size() == 5 );

  /*
    F at the sympoint is
      {{ F0+I F1, F2+I F3}, {-F2+I F3, F0-I F1}}
    Fdot at the sympoint is
      {{ F4+I F5, F6+I F7}, {-F6+I F7, F4-I F5}}

    V at the sympoint is
      {{ W0+I W1, W2+I W3}, {-W2+I W3, W0-I W1}}
    Vdot at the sympoint is
      {{ W4+I W5, W6+I W7}, {-W6+I W7, W4-I W5}}

    W0, W4, W5 are zero and omitted.
  */

  value_type_real const W1 = V[4];
  value_type_real const W2 = ( V[0] - V[2])*sym_re + (-V[1] + V[3])*sym_im;
  value_type_real const W3 = ( V[1] + V[3])*sym_re + ( V[0] + V[2])*sym_im;

  value_type_real const W6 = (-V[1] + V[3])*sym_re + (-V[0] + V[2])*sym_im;
  value_type_real const W7 = ( V[0] + V[2])*sym_re + (-V[1] - V[3])*sym_im;

  dF[0] = -F[1]*W1 - F[2]*W2 - F[3]*W3; 
  dF[1] = +F[0]*W1 - F[3]*W2 + F[2]*W3; 
  dF[2] = +F[3]*W1 + F[0]*W2 - F[1]*W3; 
  dF[3] = -F[2]*W1 + F[1]*W2 + F[0]*W3; 
  dF[4] = -F[5]*W1 - F[6]*W2 - F[7]*W3 - F[2]*W6 - F[3]*W7; 
  dF[5] = +F[4]*W1 - F[7]*W2 + F[6]*W3 - F[3]*W6 + F[2]*W7; 
  dF[6] = +F[7]*W1 + F[4]*W2 - F[5]*W3 + F[0]*W6 - F[1]*W7; 
  dF[7] = -F[6]*W1 + F[5]*W2 + F[4]*W3 + F[1]*W6 + F[0]*W7;
}

} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_frameR3_cu_INCLUDED
