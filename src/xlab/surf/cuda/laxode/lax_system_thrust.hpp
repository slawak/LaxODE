// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_lax_system_thrust_INCLUDED
#define xlab_surf_cuda_laxode_lax_system_thrust_INCLUDED

#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/memory.h>

#include "xlab/surf/cuda/lax_function_torus/laxR3all.hpp"
#include "xlab/surf/cuda/laxode/helper_functions.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace laxode {


template < class state_type_device , typename value_type_real, class thrust_tag>
class lax_system_thrust
{

public:
	lax_system_thrust(
    		const uint g ,
    		const std::size_t Nu ,
    		const state_type_device &X0 ,
    		const value_type_real dx ,
    		const value_type_real dy ,
    		const value_type_real sym_re ,
    		const value_type_real sym_im) :
    			_g ( g ) ,
    			_dim ( xlab::surf::cuda::lax_function_torus::dimension_R3( g ) ) ,
    			_Nu( Nu ) ,
    			_dim_Nu ( _dim * Nu) ,
    			_X0( X0 ) ,
    			_X ( state_type_device ( _X0 ) ) ,
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

    class lax_functor
    {

    public:
    	template< class T >
    	__host__ __device__
    	void operator()( T t ) const
    	{
    		using xlab::surf::cuda::lax_function_torus::dimension_R3;
    		using xlab::surf::cuda::lax_function_torus::array_view;
    		using xlab::surf::cuda::lax_function_torus::laxR3;

    		std::size_t i = thrust::get< 0 >( t );
    		uint g = thrust::get< 1 >( t );
    		value_type_real* ptr_dX = thrust::get< 2 >( t );
    		const value_type_real* ptr_X = thrust::get< 3 >( t );
    		value_type_real dx = thrust::get< 4 >( t );
    		value_type_real dy = thrust::get< 5 >( t );
    		value_type_real sym_re = thrust::get< 6 >( t );
    		value_type_real sym_im = thrust::get< 7 >( t );
    		std::size_t Nu = thrust::get< 8 >( t );

    		std::size_t dim = dimension_R3( g );

    		laxR3(
    				g
    				, array_view<value_type_real> ( ptr_dX , dim , i , Nu)
    				, array_view<value_type_real const> ( ptr_X, dim , i , Nu)
    				, dx
    				, dy
    				, sym_re
    				, sym_im);
    	}
    };

    void operator()(  const state_type_device &X , state_type_device &dXdt , value_type_real t ) const
    {
#ifdef DEBUG_PRINT_VERBOSE
    	std::cout << "system t=" << t << " X_raw=";
    	write_vector_cout(X);
    	std::cout << "system dXdt_raw=";
    	write_vector_cout(dXdt);
#endif
    	thrust::counting_iterator<std::size_t> begin(0);
    	thrust::counting_iterator<std::size_t> end(_Nu);

    	const value_type_real* ptr_X = thrust::raw_pointer_cast(X.data());
    	value_type_real* ptr_dX = thrust::raw_pointer_cast(dXdt.data());


    	thrust::for_each(
    			thrust::retag<thrust_tag> (
    					thrust::make_zip_iterator(
    							thrust::make_tuple(
    									begin
    									, thrust::make_constant_iterator <uint> ( _g )
    									, thrust::make_constant_iterator <value_type_real* > ( ptr_dX )
    									, thrust::make_constant_iterator <const value_type_real*> ( ptr_X )
    									, thrust::make_constant_iterator <value_type_real> ( _dx )
    									, thrust::make_constant_iterator <value_type_real> ( _dy )
    									, thrust::make_constant_iterator <value_type_real> ( _sym_re )
    									, thrust::make_constant_iterator <value_type_real> ( _sym_im )
    									, thrust::make_constant_iterator <std::size_t> ( _Nu )
    					) ) ),
    			thrust::retag<thrust_tag> (
    					thrust::make_zip_iterator(
    							thrust::make_tuple(
    									end
    									, thrust::make_constant_iterator <uint> ( _g )
    									, thrust::make_constant_iterator <value_type_real* > ( ptr_dX )
    									, thrust::make_constant_iterator <const value_type_real*> ( ptr_X )
    									, thrust::make_constant_iterator <value_type_real> ( _dx )
    									, thrust::make_constant_iterator <value_type_real> ( _dy )
    									, thrust::make_constant_iterator <value_type_real> ( _sym_re )
    									, thrust::make_constant_iterator <value_type_real> ( _sym_im )
    									, thrust::make_constant_iterator <std::size_t> ( _Nu )
                ) ) ),
                lax_functor()
    	);

#ifdef DEBUG_PRINT_VERBOSE
    	std::cout << "system dXdt_raw=";
    	write_vector_cout(dXdt);
#endif

    }

    state_type_device& getX() {
    	return _X;
    }

private:
    const uint _g;
    const std::size_t _dim;
    const std::size_t _Nu;
    const std::size_t _dim_Nu;
    const state_type_device& _X0;
    state_type_device _X;

    value_type_real _dx;
    value_type_real _dy;
    value_type_real _sym_re;
    value_type_real _sym_im;
};


} // namespace laxode
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_laxode_lax_system_thrust_INCLUDED
