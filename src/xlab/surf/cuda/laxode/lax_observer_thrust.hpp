// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_lax_observer_thrust_INCLUDED
#define xlab_surf_cuda_laxode_lax_observer_thrust_INCLUDED

//#define COPY_MAP

#ifdef COPY_MAP
#include <thrust/host_vector.h>
#endif

#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/memory.h>

#include "xlab/surf/cuda/lax_function_torus/definitions.hpp"
#include "xlab/surf/cuda/laxode/helper_functions.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace laxode {

#ifdef COPY_MAP
template < class state_type_device, typename value_type_real, class thrust_tag, class index_type_device>
#else
template < class state_type_device, typename value_type_real, class thrust_tag>
#endif
class lax_observer_thrust
{
public:
	lax_observer_thrust (
			const uint g ,
			const std::size_t Nu ,
			state_type_device& results,
			const std::size_t Nt ,
			const value_type_real t0 ,
			const value_type_real dt) :
				_g ( g ) ,
				_dim ( xlab::surf::cuda::lax_function_torus::dimension_R3( g ) ) ,
				_dim_result (xlab::surf::cuda::lax_function_torus::dimension_frame()) ,
				_Nu( Nu ) ,
				_Nt( Nt ) ,
				_t0( t0 ) ,
				_dt( dt ) ,
				_results(results) ,
				_offset ( 0 )
	{
#ifdef DEBUG_PRINT_VERBOSE
		cout << "resizing result to " << _dim_result * _Nu * _Nt << std::endl;
#endif
		_results.resize(_dim_result * _Nu * _Nt);

#ifdef COPY_MAP
		// TODO: computation should be done on device?
		// F and Fdot locations in X
		thrust::host_vector<std::size_t> source_map (_dim_result * _Nu);
		// F and Fdot locations in result
		thrust::host_vector<std::size_t> dest_map (_dim_result * _Nu);

		for(std::size_t iu = 0; iu < _Nu; iu++) {
			for (std::size_t id = 0; id < _dim_result; id++) {

				// first locations of F then Fdot, then next set
				source_map[iu * _dim_result + id] = iu + _Nu * (_dim - _dim_result + id);

				// first all locations of F then all Fdot
				if (id < _dim_result / 2) {
					dest_map[iu * _dim_result + id] = iu * _dim_result / 2 + id;
				} else {
					dest_map[iu * _dim_result + id] = iu * _dim_result / 2 + (id - _dim_result / 2) + (_Nt * _Nu * _dim_result / 2);
				}

			}
		}

		_source_map = source_map;
		_dest_map = dest_map;

#ifdef DEBUG_PRINT_VERBOSE
		std::cout << "_source_map=";
		write_vector_cout(_source_map);
		std::cout << "_dest_map=";
		write_vector_cout(_dest_map);
#endif

#endif

	}

#ifndef COPY_MAP
	class observer_write_out_functor
	{

	public:
		template< class T >
		__host__ __device__
		void operator()( T t ) const
		{
			using xlab::surf::cuda::lax_function_torus::dimension_frame;

			std::size_t iu = thrust::get< 0 >( t );
			value_type_real* ptr_results = thrust::get< 1 >( t );
			const value_type_real* ptr_X = thrust::get< 2 >( t );
			std::size_t dim = thrust::get< 3 >( t );
			std::size_t Nu = thrust::get< 4 >( t );
			std::size_t dF_offset = thrust::get< 5 >( t );
			std::size_t dim_result = dimension_frame();

			std::size_t source;
			std::size_t dest;

#ifdef DEBUG_PRINT_VERBOSE
//			printf("iu=%lu, dF_offset=%lu,  ptr_X=%p,  ptr_results=%p \n",
//					iu, dF_offset, ptr_X, ptr_results);
#endif

			for (std::size_t id = 0; id < dim_result; id++) {
				// first locations of F then Fdot all at the end
				source = iu + Nu * (dim - dim_result + id);

				// first all locations of F then all Fdot
				if (id < dim_result / 2) {
					dest = iu * dim_result / 2 + id;
				} else {
					dest = iu * dim_result / 2 + (id - dim_result / 2) + dF_offset;
				}

				// copy data
				ptr_results[dest] = ptr_X[source];

#ifdef DEBUG_PRINT_VERBOSE
//				printf("ptr_X[0]=%e ptr_results[0]=%e ptr_X[%lu]=%e ptr_results[%lu]=%e \n",
//						ptr_X[0], ptr_results[0] ,source, ptr_X[source], dest, ptr_results[dest]);
#endif
			}
		}
	};
#endif

	template< class State>
	void operator()( const State &X , value_type_real t )
	{
		//std::size_t offset =  std::floor((t - _t0) / _dt + .5);
		std::size_t offset = _offset++;
		offset *= (_dim_result / 2) * _Nu;

#ifdef DEBUG_PRINT_VERBOSE
//		std::cout << "X=";
//		write_vector_cout(X);
		std::cout << "t=" << t;
		std::cout << " offset=" << offset;
		std::cout << " state=";
		write_dim2_cout ( X , _Nu , _dim);
#endif

#ifdef COPY_MAP
		thrust::copy(
				thrust::make_permutation_iterator(X.begin(), _source_map.begin()) ,
				thrust::make_permutation_iterator(X.begin(), _source_map.end()) ,
				thrust::make_permutation_iterator(_results.begin() + offset, _dest_map.begin()));
#else
		thrust::counting_iterator<std::size_t> begin(0);
		thrust::counting_iterator<std::size_t> end(_Nu);

		const value_type_real* ptr_X = thrust::raw_pointer_cast(X.data());
		value_type_real* ptr_results = thrust::raw_pointer_cast(_results.data() + offset);
		std::size_t dF_offset = (_Nt * _Nu * _dim_result / 2);

#ifdef DEBUG_PRINT_VERBOSE
//		std::cout << "offset=" << offset;
//		std::cout << " dF_offset=" << dF_offset;
//		std::cout << " ptr_X=" << ptr_X;
//		std::cout << " ptr_results=" << ptr_results;
//		std::cout << " ptr_results_raw=" << thrust::raw_pointer_cast(_results.data());
//		std::cout << std::endl;
#endif

		thrust::for_each(
				thrust::retag<thrust_tag> (
						thrust::make_zip_iterator(
								thrust::make_tuple(
										begin
										, thrust::make_constant_iterator <value_type_real* > ( ptr_results )
										, thrust::make_constant_iterator <const value_type_real*> ( ptr_X )
										, thrust::make_constant_iterator <value_type_real> ( _dim )
										, thrust::make_constant_iterator <value_type_real> ( _Nu )
										, thrust::make_constant_iterator <value_type_real> ( dF_offset )
								) ) ),
				thrust::retag<thrust_tag> (
						thrust::make_zip_iterator(
								thrust::make_tuple(
										end
										, thrust::make_constant_iterator <value_type_real* > ( ptr_results )
										, thrust::make_constant_iterator <const value_type_real*> ( ptr_X )
										, thrust::make_constant_iterator <value_type_real> ( _dim )
										, thrust::make_constant_iterator <value_type_real> ( _Nu )
										, thrust::make_constant_iterator <value_type_real> ( dF_offset )
								) ) ),
				observer_write_out_functor()
		);

#endif

#ifdef DEBUG_PRINT_VERBOSE
//		std::cout << "result_raw=";
//		write_vector_cout(_results);
		std::cout << "results=";
		write_dim3_cout ( _results , _dim_result/2 , _Nu , _Nt);
		write_dim3_cout ( _results ,
				_dim_result /2 , _Nu , _Nt ,
				true, _Nt * _Nu * _dim_result /2);
#endif

	}

	template< class X>
	X get_results() const {
		X results (_results.begin(),_results.end());
		return results;
	}

	std::size_t get_dim_result() const{
		return _dim_result;
	}

private:
	const uint _g;
	const std::size_t _dim;
	const std::size_t _dim_result;
	const std::size_t _Nu;
	const std::size_t _Nt;
	const value_type_real _t0;
	const value_type_real _dt;
	state_type_device& _results;

	std::size_t _offset;

#ifdef COPY_MAP
	// F and Fdot locations in X
	index_type_device _source_map;
	// F and Fdot locations in result
	index_type_device _dest_map;
#endif

};

} // namespace laxode
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_laxode_lax_observer_thrust_INCLUDED
