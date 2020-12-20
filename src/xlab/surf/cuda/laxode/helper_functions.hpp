// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_helper_functions_INCLUDED
#define xlab_surf_cuda_laxode_helper_functions_INCLUDED

#include <iostream>
#include <fstream>
#include <iomanip>

namespace xlab {
namespace surf {
namespace cuda {
namespace laxode {

inline void print_threadnumber (int threadnumber) {
	std::cout << "[" << threadnumber << "]";
}

template < class Vector >
inline void write_vector_cout ( const Vector &X) {
	std::cout << std::setprecision(15) << "{ ";
	for(std::size_t i = 0; i < X.size(); i++) {
		std::cout << X[i];
		std::cout << ( i != X.size() - 1 ? " , " : " " );
	}
	std::cout << "}" << std::endl;
}

template < class Dim2 >
inline void write_dim2_cout ( const Dim2 &X, std::size_t dim1, std::size_t dim2, bool transpose = true, std::size_t offset  = 0,
		std::size_t dim1_start = 0, std::size_t dim2_start = 0) {

	std::cout << std::setprecision(15) << "{ " << std::endl;
	for(std::size_t i1 = dim1_start; i1 < dim1; i1++) {
		std::cout << "{ ";
		for (std::size_t i2 = dim2_start; i2 < dim2; i2++) {

			if (transpose)
				std::cout << X[i1 + i2 * dim1 + offset];
			else
				std::cout << X[i1 * dim2 + i2 + offset];

			std::cout << ( i2 != dim2 - 1 ? " , " : " " );
		}
		std::cout << "} ";
		std::cout << ( i1 != dim1 - 1 ? " , " : " " ) << std::endl;
	}
	std::cout << "}" << std::endl;
}

template < class Dim3 >
inline void write_dim3_cout ( const Dim3 &X,
		std::size_t dim1, std::size_t dim2, std::size_t dim3, bool transpose = true, std::size_t offset  = 0,
		std::size_t dim1_start = 0, std::size_t dim2_start = 0, std::size_t dim3_start = 0) {

	std::cout << std::setprecision(15) << "{ " << std::endl;
	for(std::size_t i3 = dim3_start ; i3 < dim3; i3++) {
		std::cout << "{ ";
		for(std::size_t i2 = dim2_start ; i2 < dim2; i2++) {
			std::cout << "{ ";
			for (std::size_t i1 = dim1_start; i1 < dim1; i1++) {

				if (transpose)
					std::cout << X[i1 + i2 * dim1 + i3 * dim2 * dim1 + offset];
				else
					std::cout << X[i1 * dim2 * dim3 + i2 * dim3 + i3  + offset];

				std::cout << ( i1 != dim1 - 1 ? " , " : " " );
			}
			std::cout << "} ";
			std::cout << ( i2 != dim2 - 1 ? " , " : " " );
		}
		std::cout << "} ";
		std::cout << ( i3 != dim3 - 1 ? " , " : " " ) << std::endl;
	}
	std::cout << "} " << std::endl;
}

template < class Dim3 >
inline void write_dim3_ofstream (std::ofstream &results_stream, const Dim3 &X,
		std::size_t dim1, std::size_t dim2, std::size_t dim3, bool transpose = true, std::size_t offset  = 0,
		std::size_t dim1_start = 0, std::size_t dim2_start = 0, std::size_t dim3_start = 0) {

	results_stream << std::setprecision(15);

//	results_stream << "{";
	for(std::size_t i3 = dim3_start ; i3 < dim3; i3++) {
//		results_stream << "{";
		for(std::size_t i2 = dim2_start ; i2 < dim2; i2++) {
//			results_stream << "{";
			for (std::size_t i1 = dim1_start; i1 < dim1; i1++) {

				if (transpose)
					results_stream << X[i1 + i2 * dim1 + i3 * dim2 * dim1 + offset];
				else
					results_stream << X[i1 * dim2 * dim3 + i2 * dim3 + i3  + offset];

				results_stream << ( i1 != dim1 - 1 ? "," : "" );
			}
//			results_stream << "}";
			results_stream << ( i2 != dim2 - 1 ? "," : "" );
		}
//		results_stream << "}";
//		results_stream << ( i3 != dim3 - 1 ? "," : "" );
		results_stream << std::endl;
	}
//	results_stream << "}";
//	results_stream << std::endl;
}

template < class Vector >
inline void write_vector_ofstream (std::ofstream &results_stream, const Vector &X) {
	results_stream << std::setprecision(15);
	for(std::size_t i = 0; i < X.size(); i++) {
		results_stream << X[i];
		results_stream << ( i != X.size() - 1 ? "," : " " );
	}
	results_stream << std::endl;
}

} // namespace laxode
} // namespace cuda
} // namespace surf
} // namespace xlab


#endif //xlab_surf_cuda_laxode_helper_functions_INCLUDED
