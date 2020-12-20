// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_laxcudaerrors_h_INCLUDED
#define xlab_surf_cuda_laxode_laxcudaerrors_h_INCLUDED

#include "laxcuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Translate enum error_state to a C-Style string representation.
 */
extern inline
const char *laxode_get_error_string(enum LAXODE_ERROR_STATE state) {
	switch(state) {
	case SUCCESS:
		return "SUCCESS";
	case F_DF_NULL_PTR:
		return "F_DF_NULL_PTR";
	case X0_NULL_PTR:
		return "X0_NULL_PTR";
	case DOMAIN_N_U_OUT_OFF_BOUNDS:
		return "DOMAIN_N_U_OUT_OFF_BOUNDS";
	case DOMAIN_N_V_OUT_OFF_BOUNDS:
		return "DOMAIN_N_V_OUT_OFF_BOUNDS";
	case LINE_METHOD_UNKOWN:
		return "LINE_METHOD_UNKOWN";
	case GRID_METHOD_UNKOWN:
		return "GRID_METHOD_UNKOWN";
	case PRECISIONGOAL_OUT_OFF_BOUNDS:
		return "PRECISIONGOAL_OUT_OFF_BOUNDS";
	case GRID_DEVICE_UNKOWN:
		return "GRID_DEVICE_UNKOWN";
	case CUDA_INIT_ERROR:
		return "CUDA_INIT_ERROR";
	case THRUST_RUNTIME_ERROR:
		return "THRUST_RUNTIME_ERROR";
	case MEMORY_ALLOCATION_ERROR:
		return "MEMORY_ALLOCATION_ERROR";
	default:
		return "UNKOWN_ERROR";
	}
}

/**
 * Find out the enum error_state for a given string representation
 */
extern inline
enum LAXODE_ERROR_STATE laxode_get_error_state(const char *str) {
	if (!strcmp(str, "SUCCESS")) return SUCCESS;
	if (!strcmp(str, "F_DF_NULL_PTR")) return F_DF_NULL_PTR;
	if (!strcmp(str, "X0_NULL_PTR"))	return X0_NULL_PTR;
	if (!strcmp(str, "DOMAIN_N_U_OUT_OFF_BOUNDS")) return DOMAIN_N_U_OUT_OFF_BOUNDS;
	if (!strcmp(str, "DOMAIN_N_V_OUT_OFF_BOUNDS")) return DOMAIN_N_V_OUT_OFF_BOUNDS;
	if (!strcmp(str, "LINE_METHOD_UNKOWN")) return LINE_METHOD_UNKOWN;
	if (!strcmp(str, "GRID_METHOD_UNKOWN")) return GRID_METHOD_UNKOWN;
	if (!strcmp(str, "PRECISIONGOAL_OUT_OFF_BOUNDS")) return PRECISIONGOAL_OUT_OFF_BOUNDS;
	if (!strcmp(str, "GRID_DEVICE_UNKOWN")) return GRID_DEVICE_UNKOWN;
	if (!strcmp(str, "CUDA_INIT_ERROR"))	return CUDA_INIT_ERROR;
	if (!strcmp(str, "THRUST_RUNTIME_ERROR")) return THRUST_RUNTIME_ERROR;
	if (!strcmp(str, "MEMORY_ALLOCATION_ERROR"))	return MEMORY_ALLOCATION_ERROR;
	return UNKOWN_ERROR;
}

#ifdef __cplusplus
}
#endif

#endif // xlab_surf_cuda_laxode_laxcudaerrors_h_INCLUDED
