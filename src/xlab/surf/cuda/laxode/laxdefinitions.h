// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_laxdefinitions_h_INCLUDED
#define xlab_surf_cuda_laxode_laxdefinitions_h_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t size_type;

typedef double value_type_real;

typedef unsigned int uint;

/**
 * Parallelogram shaped grid domain with
 * origin (o_x,o_y) and two vertices at
 * (u_x,u_y) and (v_x,v_y)
 * N_u gives the number of grid points in u direction
 * N_v gives the number of grid points in v direction
 */
struct laxode_domain {
	value_type_real o_x;
	value_type_real o_y;
	value_type_real u_x;
	value_type_real u_y;
	value_type_real v_x;
	value_type_real v_y;
	uint N_u;
	uint N_v;
};

/**
 * Default precisiongoal
 */
value_type_real const precisiongoal_default = 1.0e-9;

/**
 * Integration methods
 */
enum LAXODE_INTEGRATION_METHOD {
	RUNGE_KUTTA4 = 1,
	RUNGE_KUTTA_DOPRI5_CONTROLLED = 2,
	RUNGE_KUTTA_DOPRI5_DENSE = 3
};

/**
 * Devices for integration
 */
enum LAXODE_INTEGRATION_DEVICE {
	CPU_SOLVER = 1,
	CPU_SOLVER_SINGLE_THREAD = 2,
	CUDA_SOLVER = 3,
	CUDA_SOLVER_MULTI = 4 ,
	CUDA_SOLVER_WITH_CPU_FALLBACK = 5
};

inline static
size_type laxode_dimension_lax(uint const g) { return 3*g+2;}

inline static
size_type laxode_dimension_frame() { return 8;}

inline static
size_type laxode_dimension_R3(uint const g) { return laxode_dimension_lax(g) + laxode_dimension_frame();}

inline static
size_type laxode_result_size(size_type Nu, size_type Nv) { return laxode_dimension_frame() * Nu * Nv;}

#ifdef __cplusplus
}
#endif

#endif //xlab_surf_cuda_laxode_laxdefinitions_h_INCLUDED
