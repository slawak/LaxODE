// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_lax_system_single_INCLUDED
#define xlab_surf_cuda_laxode_lax_system_single_INCLUDED

#include "xlab/surf/cuda/lax_function_torus/laxR3all.hpp"
#include "xlab/surf/cuda/laxode/helper_functions.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace laxode {


template < class state_type , typename value_type_real >
class lax_system_single
{

public:
	lax_system_single(
    		const uint g ,
    		const state_type &X0 ,
    		const value_type_real dx ,
    		const value_type_real dy ,
    		const value_type_real sym_re ,
    		const value_type_real sym_im) :
    			_g ( g ) ,
    			_dim ( xlab::surf::cuda::lax_function_torus::dimension_R3( g ) ) ,
    			_X0( X0 ) ,
    			_X ( state_type ( _X0 ) ) ,
    			_dx ( dx ) ,
    			_dy ( dy ) ,
    			_sym_re ( sym_re ) ,
    			_sym_im ( sym_im )
    {
#ifdef DEBUG_PRINT_VERBOSE
    	std::cout << "X0_raw=";
    	write_vector_cout(_X0);
    	std::cout << "X_raw=";
    	write_vector_cout(_X);
#endif
    }

    void operator()(  const state_type &X , state_type &dXdt , value_type_real t ) const
    {
    	using xlab::surf::cuda::lax_function_torus::array_view;
    	using xlab::surf::cuda::lax_function_torus::laxR3;

#ifdef DEBUG_PRINT_VERBOSE
    	std::cout << "system t=" << t << " X_raw=";
    	write_vector_cout(X);
#endif

    	const value_type_real* ptr_X = thrust::raw_pointer_cast(X.data());
    	value_type_real* ptr_dX = thrust::raw_pointer_cast(dXdt.data());

    	laxR3(
    			_g
    			, array_view<value_type_real> ( ptr_dX , dXdt.size() )
    			, array_view<value_type_real const> ( ptr_X, X.size() )
    			, _dx
    			, _dy
    			, _sym_re
    			, _sym_im);

#ifdef DEBUG_PRINT_VERBOSE
    	std::cout << "system dXdt_raw=";
    	write_vector_cout(dXdt);
#endif

    }

    state_type& getX() {
    	return _X;
    }

private:
    const uint _g;
    const std::size_t _dim;
    const state_type& _X0;
    state_type _X;

    value_type_real _dx;
    value_type_real _dy;
    value_type_real _sym_re;
    value_type_real _sym_im;
};


} // namespace laxode
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_laxode_lax_system_single_INCLUDED
