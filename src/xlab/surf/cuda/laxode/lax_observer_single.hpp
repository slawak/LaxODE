// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_lax_observer_single_INCLUDED
#define xlab_surf_cuda_laxode_lax_observer_single_INCLUDED

#include "xlab/surf/cuda/lax_function_torus/definitions.hpp"
#include "xlab/surf/cuda/laxode/helper_functions.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace laxode {

template < class state_type, typename value_type_real>
class lax_observer_single
{
public:
	lax_observer_single (
			const uint g ,
			state_type& results,
			const std::size_t Nt ,
			const value_type_real t0 ,
			const value_type_real dt) :
				_g ( g ) ,
				_dim ( xlab::surf::cuda::lax_function_torus::dimension_R3( g ) ) ,
				_results (results) ,
				_Nt( Nt ) ,
				_t0( t0 ) ,
				_dt( dt ) ,
				_offset ( 0 )
	{
#ifdef DEBUG_PRINT_VERBOSE
		cout << "resizing result to " << _dim * _Nt << std::endl;
#endif
		_results.resize(_dim * _Nt );
	}

	void operator()( const state_type &X , value_type_real t )
	{
		//std::size_t offset =  std::floor((t - _t0) / _dt + .5);
		std::size_t offset = _offset++;

#ifdef DEBUG_PRINT_VERBOSE
//		std::cout << "X=";
//		write_vector_cout(X);
		std::cout << "t=" << t;
		std::cout << " offset=" << offset;
		std::cout << " state=";
		write_dim2_cout ( X , 1 , _dim);
#endif

		for (std::size_t i = 0; i < _dim ; i++) {
			_results[i * _Nt + offset] = X[i];
		}

#ifdef DEBUG_PRINT_VERBOSE
//		std::cout << "result_raw=";
//		write_vector_cout(_results);
		std::cout << "results=";
		write_dim2_cout ( _results , _Nt , _dim);
#endif

	}

	template< class X>
	X get_results() const {
		X results (_results.begin(),_results.end());
		return results;
	}

	std::size_t get_dim_results() const{
		return _dim;
	}

private:
	const uint _g;
	const std::size_t _dim;
	state_type& _results;
	const std::size_t _Nt;
	const value_type_real _t0;
	const value_type_real _dt;

	std::size_t _offset;
};

} // namespace laxode
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_laxode_lax_observer_single_INCLUDED
