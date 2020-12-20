// Copyright (C) 2012 Wjatscheslaw Kewlin
#ifndef xlab_surf_cuda_lax_laxR3_all_INCLUDED
#define xlab_surf_cuda_lax_laxR3_all_INCLUDED

#define CUDA_WHOLE_PROGRAM_COMPILATION

#include "definitions.hpp"
#include "array_view.hpp"
#include "potential.hpp"
#include "lax.hpp"
#include "frameR3.hpp"
#include "laxR3.hpp"

#ifdef CUDA_WHOLE_PROGRAM_COMPILATION
#include "potential.cu"
#include "lax.cu"
#include "frameR3.cu"
#include "laxR3.cu"
#endif

#endif // xlab_surf_cuda_lax_laxR3_all_INCLUDED
