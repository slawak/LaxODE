// Copyright (C) 2012 Nicholas Schmitt
// Copyright (C) 2012 Wjatscheslaw Kewlin
// Reworked by W. Kewlin to use on CUDA

#ifndef xlab_surf_cuda_lax_function_torus_laxR3_INCLUDED
#define xlab_surf_cuda_lax_function_torus_laxR3_INCLUDED

#include "definitions.hpp"
#include "array_view.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

extern
__host__ __device__
void laxR3(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    value_type_real const &dx,
    value_type_real const &dy,
    value_type_real const &sym_re,
    value_type_real const &sym_im,
    array_view<value_type_real> Vtemp);

extern
__host__ __device__
void laxR3(
    uint const &g,
    array_view<value_type_real> dX,
    array_view<value_type_real const> const &X,
    value_type_real const &dx,
    value_type_real const &dy,
    value_type_real const &sym_re,
    value_type_real const &sym_im);

} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_laxR3_INCLUDED
