#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "laxdefinitions.h"
#include "laxcuda.h"
#include "laxcudaerrors.h"

#include "mathlink.h"

//void integrate P(( int , double*, long , int , int , double , int ));

:Begin:
:Function:      integrate
:Pattern:       LaxODEIntegrate[g_Integer, X0:{___Real}, linemethod_Integer, gridmethod_Integer, precisiongoal_Real, griddevice_Integer, sym_Complex, domain:{__Complex}]
                                /;Length[X0]==3*g+2+8 && Length[domain]==4
:Arguments:     { g, X0, linemethod, gridmethod, precisiongoal, griddevice, sym, domain }
:ArgumentTypes: { Integer, RealList, Integer, Integer, Real, Integer, Manual }
:ReturnType:    Manual
:End:

:Evaluate:      

void integrate( int g, double* X0, long lenX0, int linemethod, int gridmethod, double precisiongoal, int griddevice )
{
	double* sym;
	int* sym_dims;
	char** sym_heads;
	int sym_depth;
	MLGetReal64Array ( stdlink, &sym, &sym_dims, &sym_heads, &sym_depth );
	printf ("Sym= (%f , %f)\n",sym[0],sym[1]);

	double* domain_vector;
	int* domain_dims;
	char** domain_heads;
	int domain_depth;
	MLGetReal64Array ( stdlink, &domain_vector, &domain_dims, &domain_heads, &domain_depth );

	printf ("domain_depth= (%d)\n",domain_depth );

	size_t Nu = abs(floor(domain_vector[3*2+0]));
	size_t Nv = abs(floor(domain_vector[3*2+1]));

	struct laxode_domain domain0 = {
			domain_vector[0*2+0],domain_vector[0*2+1] , //origin
			domain_vector[1*2+0],domain_vector[1*2+1] , //first per
			domain_vector[2*2+0],domain_vector[2*2+1] ,  //second per
			Nu, Nv //gridcount
	};

	printf ("Domain= (%f, %f, %f, %f, %f, %f, %d, %d)\n",domain0.o_x,domain0.o_y,domain0.u_x,domain0.u_y,domain0.v_x,domain0.v_y,domain0.N_u,domain0.N_v);

	size_t res_size = laxode_result_size( Nu , Nv);
	value_type_real* results = malloc (res_size * sizeof (value_type_real));
	
    enum LAXODE_ERROR_STATE result = laxode_integrate(results, g, X0, sym[0], sym[1], domain0, linemethod, gridmethod, precisiongoal, griddevice);
	printf ("Result of integration %s\n",laxode_get_error_string(result));
	
	MLReleaseReal64Array ( stdlink, domain_vector, domain_dims, domain_heads, domain_depth );
	MLReleaseReal64Array ( stdlink, sym, sym_dims, sym_heads, sym_depth );

	MLPutReal64List(stdlink, results, res_size);
	free(results);
}


int main(argc, argv)
	int argc; char* argv[];
{
	return MLMain(argc, argv);
}

