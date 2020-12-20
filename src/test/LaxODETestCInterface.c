// Copyright (C) 2012 Wjatscheslaw Kewlin

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "laxcuda.h"
#include "laxcudaerrors.h"

int main(const int argc , const  char** argv)
{
	uint g = 2;
	size_t Nu = 200;
	size_t Nv = 200;

//	enum LAXODE_INTEGRATION_METHOD line_method = RUNGE_KUTTA4;
//	enum LAXODE_INTEGRATION_METHOD line_method = RUNGE_KUTTA_DOPRI5_CONTROLLED;
	enum LAXODE_INTEGRATION_METHOD line_method = RUNGE_KUTTA_DOPRI5_DENSE;
//	enum LAXODE_INTEGRATION_METHOD grid_method = RUNGE_KUTTA4;
//	enum LAXODE_INTEGRATION_METHOD grid_method = RUNGE_KUTTA_DOPRI5_CONTROLLED;
	enum LAXODE_INTEGRATION_METHOD grid_method = RUNGE_KUTTA_DOPRI5_DENSE;

	value_type_real precisiongoal = precisiongoal_default;

	enum LAXODE_INTEGRATION_DEVICE grid_device = CUDA_SOLVER;
	//enum LAXODE_INTEGRATION_DEVICE grid_device = CPU_SOLVER;

	printf("Lax ODE Test\n");

	printf("Loading defaults and evaluating arguments\n");

	size_t res_size = laxode_result_size( Nu , Nv);

	// Wente Torus
	const value_type_real X0[] = {1,0,-0.03031390398319432783124200350541194672841107,0,0,0,-0.2825269372599843036146124128463817986231055,0, 1, 0, 0, 0, 0, 0, 0, 0};

	// Sympoint is 1
	value_type_real sym_re = 1;
	value_type_real sym_im = 0;

	// Wente domain
	struct laxode_domain domain0 = {
			0 ,0 , //origin
			12.865155157904928 ,0 , //first per
			0,19.961498805657186 ,  //second per
			Nu, Nv //gridcount
	};

	value_type_real* results = malloc (res_size * sizeof (value_type_real));

	enum LAXODE_ERROR_STATE result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, CUDA_SOLVER_WITH_CPU_FALLBACK);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, CPU_SOLVER);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	result = laxode_integrate(NULL, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	result = laxode_integrate(results, g, NULL, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, -1, grid_method, precisiongoal, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, -1, precisiongoal, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, -1, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, -1);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	domain0.N_u = 0;
	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	domain0.N_u = 2;
	domain0.N_v = 0;
	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	domain0.N_u = 1<<31;
	domain0.N_v = 2;
	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	domain0.N_u = 2;
	domain0.N_v = 1<<31;
	result = laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, grid_device);
	printf ("Result of integration %s\n",laxode_get_error_string(result));

	free(results);

	printf ("Test finished\n");

	return 0;
}

