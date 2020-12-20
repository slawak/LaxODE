// Copyright (C) 2012 Nicholas Schmitt
// Copyright (C) 2012 Wjatscheslaw Kewlin
// Reworked by W. Kewlin to use on CUDA

#ifndef xlab_surf_cuda_lax_function_torus_array_view_INCLUDED
#define xlab_surf_cuda_lax_function_torus_array_view_INCLUDED

#include "definitions.hpp"

namespace xlab {
namespace surf {
namespace cuda {
namespace lax_function_torus {

/**

*/
template< typename T>
class array_view
{

public:

  // size types
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;


  // value types
  typedef T value_type;
  typedef value_type &reference;
  typedef value_type const &const_reference;
  typedef value_type *pointer;
  typedef value_type const *const_pointer;

  /**
    Constructs a new array_view instance.
  */
  __host__ __device__ inline array_view(T *data_, size_type const &size_, difference_type const &offset = 0, difference_type const &stepsize = 1)
      :
      _data(data_ + offset)
      ,_size(size_)
  	  , _stepsize (stepsize)
    {
      if (_size != 0)
      {
    	  // TODO: Check needed
//        XLAB_CHECK_POINTER(_data);
      }
    }

  /**
     Copy constructor.
  */
  __host__ __device__ inline array_view(array_view const &x )
      :
      _data(x._data)
      ,_size(x._size)
  	  ,_stepsize(x._stepsize)
    {
    }

  __host__ __device__ inline array_view(array_view const &x , size_type const &size, difference_type const &offset = 0)
        :
        _data(x._data + offset * x._stepsize)
        ,_size(size)
    	,_stepsize(x._stepsize)
      {
      }

  __host__ __device__ inline
  array_view<const T> getConst() const
  {
	  return array_view<const T> (_data, _size, 0, _stepsize);
  }


  __host__ __device__ inline  size_type size() const
    { return _size; }

  __host__ __device__ inline T &operator[](size_type const &i)
    {
	  XLAB_ASSERT(i < _size);
      return _data[i * _stepsize];
    }

  __host__ __device__ inline T const &operator[](size_type const &i) const
    {
	  XLAB_ASSERT(i < _size);
      return _data[i * _stepsize];
    }

protected:

private:

  T *_data;
  const size_type _size;
  const difference_type _stepsize;

  /**
     Disabled assignment operator.
  */
  __host__ __device__ array_view &operator=( array_view const & );

};

} // namespace lax
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_lax_function_torus_array_view_INCLUDED

