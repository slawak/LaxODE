// Copyright (C) 2012 Wjatscheslaw Kewlin

#define THRUST_HOST_SYSTEM THRUST_HOST_SYSTEM_OMP

#include <cmath>
#include <iostream>
#include <iomanip>

#include <boost/ref.hpp>
//#include <boost/thread.hpp>
//#include <boost/shared_ptr.hpp>
//#include <boost/make_shared.hpp>

#include <thrust/device_vector.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <thrust/host_vector.h>
//#include <thrust/system/cpp/vector.h>
//#include <thrust/system/tbb/vector.h>
#include <thrust/system/omp/vector.h>

#include <thrust/copy.h>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_algebra.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_operations.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_resize.hpp>

//#define PRINT_CERR

//#define DEBUG_PRINT_VERBOSE
//#define DEBUG_PRINT
//#define INFO_PRINT_VERBOSE
//#define INFO_PRINT

#ifdef DEBUG_PRINT_VERBOSE
#define DEBUG_PRINT
#endif


#ifdef DEBUG_PRINT
#define INFO_PRINT_VERBOSE
#endif

#ifdef INFO_PRINT_VERBOSE
#define INFO_PRINT
#include <boost/timer/timer.hpp>
#endif

#include "lax_system_single.hpp"
#include "lax_system_thrust.hpp"
#include "lax_observer_single.hpp"
#include "lax_observer_thrust.hpp"

#include "helper_functions.hpp"
#include "laxcuda.hpp"

#include "integrate_line.hpp"
#include "integrate_grid.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace laxode {

#include "laxcuda.h"
#include "laxcudaerrors.h"


//forward declaration
static void
integrate_impl(
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

// C++ Wrapper
extern
void
laxode_integrate(
		std::vector<value_type_real> &F_DF ,
		uint const g ,
		std::vector<value_type_real> const &X0 ,
		value_type_real const sym_re ,
		value_type_real const sym_im ,
		struct laxode_domain const dom ,
		enum LAXODE_INTEGRATION_METHOD line_method ,
		enum LAXODE_INTEGRATION_METHOD grid_method ,
		value_type_real precisiongoal ,
		enum LAXODE_INTEGRATION_DEVICE grid_device
)
{

	F_DF.resize( laxode_result_size( dom.N_u , dom.N_v ) );
	integrate_impl(F_DF.data() , g , X0.data() ,
			sym_re , sym_im , dom ,
			line_method , grid_method , precisiongoal , grid_device);
}

// C Wrapper
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
	    )
{
	enum LAXODE_ERROR_STATE error_state = SUCCESS;
	try {
	integrate_impl(F_DF , g , X0 ,
				sym_re , sym_im , dom ,
				line_method , grid_method , precisiongoal , grid_device);
	} catch (laxode_integrate_exception &e) {
#ifdef PRINT_CERR
		std::cerr << "Error in " << __FILE__ << " : "<< e.what() << std::endl;
#endif
		error_state = laxode_get_error_state(e.what());
	} catch (thrust::system_error &e) {
#ifdef PRINT_CERR
		std::cerr << "Error in " << __FILE__ << " : "<< e.what() << std::endl;
#endif
		error_state = THRUST_RUNTIME_ERROR;
	} catch (std::bad_alloc &e) {
#ifdef PRINT_CERR
		std::cerr << "Error in " << __FILE__ << " : "<< e.what() << std::endl;
#endif
		error_state = MEMORY_ALLOCATION_ERROR;
	} catch (std::exception &e) {
#ifdef PRINT_CERR
		std::cerr << "Error in " << __FILE__ << " : "<< e.what() << std::endl;
#endif
		error_state = UNKOWN_ERROR;
	}
	return error_state;
}

void
check_input(value_type_real *F_DF ,
		uint const g ,
		value_type_real const *X0 ,
		value_type_real const sym_re ,
	    value_type_real const sym_im ,
	    struct laxode_domain const dom ,
	    enum LAXODE_INTEGRATION_METHOD line_method ,
	    enum LAXODE_INTEGRATION_METHOD grid_method ,
	    value_type_real precisiongoal ,
	    enum LAXODE_INTEGRATION_DEVICE grid_device) {

	if (F_DF == NULL)
		throw laxode_integrate_exception(laxode_get_error_string(F_DF_NULL_PTR));

	if (X0 == NULL)
		throw laxode_integrate_exception(laxode_get_error_string(X0_NULL_PTR));

	if (dom.N_u <= 1)
		throw laxode_integrate_exception(laxode_get_error_string(DOMAIN_N_U_OUT_OFF_BOUNDS));

	if (dom.N_v <= 1)
		throw laxode_integrate_exception(laxode_get_error_string(DOMAIN_N_V_OUT_OFF_BOUNDS));

	if (line_method < RUNGE_KUTTA4 || line_method > RUNGE_KUTTA_DOPRI5_DENSE)
		throw laxode_integrate_exception(laxode_get_error_string(LINE_METHOD_UNKOWN));

	if (grid_method < RUNGE_KUTTA4 || grid_method > RUNGE_KUTTA_DOPRI5_DENSE)
		throw laxode_integrate_exception(laxode_get_error_string(GRID_METHOD_UNKOWN));

	if (precisiongoal <= 0)
		throw laxode_integrate_exception(laxode_get_error_string(PRECISIONGOAL_OUT_OFF_BOUNDS));

	if (grid_device < CPU_SOLVER || grid_device > CUDA_SOLVER_WITH_CPU_FALLBACK)
		throw laxode_integrate_exception(laxode_get_error_string(GRID_DEVICE_UNKOWN));
}


// Actual implementation
void
integrate_impl(
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
	    )
{
	check_input(F_DF , g , X0 , sym_re , sym_im , dom , line_method , grid_method , precisiongoal , grid_device);


#ifdef INFO_PRINT
	boost::timer::cpu_timer timer;
	boost::timer::cpu_timer timerall;
	timerall.start();
	std::cout << "Start integration..." << std::endl;
#endif

	typedef thrust::host_vector< value_type_real > state_type_host;
	typedef state_type_host state_type_line;


	std::size_t dim = laxode_dimension_R3(g);

	laxode_domain1d line(dom.o_x,dom.o_y,dom.u_x,dom.u_y,dom.N_u);

	state_type_line Xu;
	state_type_line X0_line(dim);
	thrust::copy(X0, X0 + dim, X0_line.begin());
	integrate_line(Xu , g, X0_line, sym_re, sym_im, line, line_method , precisiongoal);

	if (grid_device == CUDA_SOLVER ||
			grid_device == CUDA_SOLVER_MULTI ||
			grid_device == CUDA_SOLVER_WITH_CPU_FALLBACK) {

		int num_devices;
		cudaError cuda_error = cudaGetDeviceCount(&num_devices);

		// CUDA init not possible
		if (cuda_error == cudaErrorNoDevice || cuda_error == cudaErrorInsufficientDriver) {

			// Either fall back
			if (grid_device == CUDA_SOLVER_WITH_CPU_FALLBACK) {
#ifdef INFO_PRINT
			std::cout << "Warning: No CUDA devices available, falling back to CPU!!!" << std::endl;
			grid_device = CPU_SOLVER;
#endif
			// Or throw an exception
			} else {
				throw laxode_integrate_exception(laxode_get_error_string(CUDA_INIT_ERROR));
			}

		} else {
			if (grid_device == CUDA_SOLVER_WITH_CPU_FALLBACK) {
				grid_device = CUDA_SOLVER_MULTI;
			}
#ifdef INFO_PRINT
			std::cout << "Using CUDA, number of devices found: " << num_devices << std::endl;
#endif
		}

	}

// CPU CODE ----------------------------------------------------------
	if (grid_device == CPU_SOLVER || grid_device == CPU_SOLVER_SINGLE_THREAD) {
#ifdef INFO_PRINT
		std::cout << "Using CPU solver" << std::endl;
#endif

		typedef thrust::omp::tag thrust_tag;
		typedef thrust::host_vector < value_type_real > state_type_grid;


		state_type_grid F_DF_device;
		integrate_grid< state_type_grid , thrust_tag >(
				F_DF_device, g, Xu, sym_re, sym_im, dom, grid_method , precisiongoal);

#ifdef INFO_PRINT
		timer.start();
		std::cout << "Transferring results of grid integration from device .." << std::endl;
#endif

		thrust::copy(F_DF_device.begin(),F_DF_device.end(),F_DF);

#ifdef INFO_PRINT
		std::cout << "Transfered in: " << timer.format() << std::endl;
#endif


	} else

// CUDA CODE ---------------------------------------------------------------
		if (grid_device == CUDA_SOLVER || grid_device == CUDA_SOLVER_MULTI) {
#ifdef INFO_PRINT
		std::cout << "Using CUDA solver" << std::endl;
#endif

		typedef thrust::cuda::tag thrust_tag;
		typedef thrust::device_vector < value_type_real > state_type_grid;
//		typedef thrust::cuda::vector < value_type_real > state_type_grid;
//		typedef thrust::system::cuda::experimental::pinned_allocator < value_type_real > pinned_alloc;
//		typedef thrust::device_vector < pinned_alloc > state_type_grid;
//		typedef thrust::cuda::vector < pinned_alloc > state_type_grid;


#ifdef INFO_PRINT
		timer.start();
		std::cout << "Transferring results of 1D integration to device .." << std::endl;
#endif

		state_type_grid X0_grid(Xu.size());
		thrust::copy(Xu.begin(), Xu.end(), X0_grid.begin());

#ifdef INFO_PRINT
		std::cout << "Transfered in: " << timer.format() << std::endl;
#endif

		state_type_grid F_DF_device;
		integrate_grid< state_type_grid , thrust_tag >(
				F_DF_device, g, X0_grid, sym_re, sym_im, dom, grid_method , precisiongoal);

#ifdef INFO_PRINT
		timer.start();
		std::cout << "Transferring results of grid integration from device .." << std::endl;
#endif

		thrust::copy(F_DF_device.begin(),F_DF_device.end(),F_DF);

#ifdef INFO_PRINT
		std::cout << "Transfered in: " << timer.format() << std::endl;
#endif

	}

#ifdef INFO_PRINT
		std::cout << "Integration finished in:" << timerall.format() << std::endl;
#endif

}

} // namespace laxode
} // namespace cuda
} // namespace surf
} // namespace xlab
