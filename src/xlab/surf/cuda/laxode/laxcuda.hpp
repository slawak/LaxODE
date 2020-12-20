// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_laxcuda_hpp_INCLUDED
#define xlab_surf_cuda_laxode_laxcuda_hpp_INCLUDED

namespace xlab {
namespace surf {
namespace cuda {
namespace laxode {

#include "laxdefinitions.h"

class laxode_integrate_exception : public std::exception
{
	const char *_M_msg;

public:
	/** Takes a C-style character string describing the general cause of
	 *  the current error. */
	explicit laxode_integrate_exception(const char *arg) : _M_msg(arg)
	{ }

	virtual ~laxode_integrate_exception() throw()
	{ }

	/** Returns a C-style character string describing the general cause of
	 *  the current error (the same string passed to the ctor).  */
	virtual const char*
	what() const throw()
	{
		return _M_msg;
	}
};

/**
 * Integrate lax equation and the frame equation for cmc torus in R3
 * of a spectral genus g. Use either CUDA or multiple cores of cpu.
 *
 * Parameter:
 * F_DF vector of results where
 * 		frame F and dF with respect to lambda at sympoint are stored.
 * 		The size of the vector will be automatically set.
 * 		The array has to be preallocated.
 * 		Organization: 4 components of F at origin, then the F along the line to u
 * 					then next grid line till the last line from v to u+v.
 * 					Then the same procedure for dF
 * g	spectral genus
 * X0	vector for the initial data, size 3*g+2+8
 * sym_re real part of sympoint
 * sym_im imaginary part of sympoint
 * dom	parallelogram shaped grid domain with grid parameters,
 * 		at least 2 grid point need in every direction
 * line_method integration method along o to u (single threaded)
 * grid_method integration method for all other points (multi threaded)
 * precisiongoal maximal error for controlled and dense output methods
 * grid_device choose where to perform multi threaded integration
 *
 * Throws
 * integrate_exception for inconsistent input and
 * 						if chosen integration device is not available
 *
 * implementation backend can throw
 * thrust::system_error which are derived from std::runtime_error
 * for internal errors and
 * std::bad_alloc for memory allocation errors,
 * usually if grid size is to big to allocate working memory
 */
extern void
laxode_integrate(
		std::vector<value_type_real> &F_DF ,
		uint const g ,
		std::vector<value_type_real> const &X0 ,
		value_type_real const sym_re ,
		value_type_real const sym_im ,
		struct laxode_domain const dom ,
		enum LAXODE_INTEGRATION_METHOD line_method = RUNGE_KUTTA_DOPRI5_DENSE ,
		enum LAXODE_INTEGRATION_METHOD grid_method = RUNGE_KUTTA_DOPRI5_DENSE ,
		value_type_real precisiongoal = precisiongoal_default,
		enum LAXODE_INTEGRATION_DEVICE grid_device = CUDA_SOLVER_MULTI
		);

} // namespace laxode
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_laxode_laxcuda_hpp_INCLUDED
