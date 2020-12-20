// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_laxcuda_h_INCLUDED
#define xlab_surf_cuda_laxode_laxcuda_h_INCLUDED

#include "laxdefinitions.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * possible return values
 */
enum LAXODE_ERROR_STATE {
	SUCCESS = 0,
	F_DF_NULL_PTR = 1<<1 ,
	X0_NULL_PTR = 1<<2,
	DOMAIN_N_U_OUT_OFF_BOUNDS = 1<<3 ,
	DOMAIN_N_V_OUT_OFF_BOUNDS = 1<<4 ,
	LINE_METHOD_UNKOWN = 1<<5 ,
	GRID_METHOD_UNKOWN = 1<<6 ,
	PRECISIONGOAL_OUT_OFF_BOUNDS = 1<<7 ,
	GRID_DEVICE_UNKOWN = 1<<8 ,
	CUDA_INIT_ERROR = 1<<9 ,
	THRUST_RUNTIME_ERROR = 1<<10 ,
	MEMORY_ALLOCATION_ERROR = 1<<11 ,
	UNKOWN_ERROR = 1<<20
};

/**
 * Integrate lax equation and the frame equation for cmc torus in R3
 * of a spectral genus g. Use either CUDA or multiple cores of cpu.
 *
 * Parameter:
 * F_DF pointer to an array of results where
 * 		frame F and dF with respect to lambda at sympoint are stored.
 * 		The size of array can be computed by result_size(Nu, Nv) or by 8*Nu*Nv.
 * 		The array has to be preallocated.
 * 		Organization: 4 components of F at origin, then the F along the line to u
 * 					then next grid line till the last line from v to u+v.
 * 					Then the same procedure for dF
 * g	spectral genus
 * X0	pointer to an array with the initial data, size 3*g+2+8
 * sym_re real part of sympoint
 * sym_im imaginary part of sympoint
 * dom	parallelogram shaped grid domain with grid parameters,
 * 		at least 2 grid point need in every direction
 * line_method integration method along o to u (single threaded)
 * grid_method integration method for all other points (multi threaded)
 * precisiongoal maximal error for controlled and dense output methods
 * grid_device choose where to perform multi threaded integration
 *
 * return type is laxode_error_state for various error states
 */
extern
enum LAXODE_ERROR_STATE
laxode_integrate(
		value_type_real *F_DF ,
		uint const g ,
		value_type_real const *X0 ,
		value_type_real const sym_re ,
	    value_type_real const sym_im ,
	    struct laxode_domain const dom ,
	    enum LAXODE_INTEGRATION_METHOD line_method ,
	    enum LAXODE_INTEGRATION_METHOD grid_method ,
	    value_type_real precisiongoal ,
	    enum LAXODE_INTEGRATION_DEVICE grid_device
	    );


#ifdef __cplusplus
}
#endif

#endif // xlab_surf_cuda_laxode_laxcuda_h_INCLUDED
