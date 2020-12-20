// Copyright (C) 2012 Wjatscheslaw Kewlin

#include <iostream>
#include <fstream>
#include <iomanip>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>

//#define DEBUG_PRINT

#include "laxcuda.hpp"

using namespace xlab::surf::cuda::laxode;

template < class Dim3 >
inline void write_dim3_ofstream (std::ofstream &results_stream, const Dim3 &X,
		std::size_t dim1, std::size_t dim2, std::size_t dim3, bool transpose = true, std::size_t offset  = 0,
		std::size_t dim1_start = 0, std::size_t dim2_start = 0, std::size_t dim3_start = 0) {

	results_stream << std::setprecision(15);

	for(std::size_t i3 = dim3_start ; i3 < dim3; i3++) {
		for(std::size_t i2 = dim2_start ; i2 < dim2; i2++) {
			for (std::size_t i1 = dim1_start; i1 < dim1; i1++) {

				if (transpose)
					results_stream << X[i1 + i2 * dim1 + i3 * dim2 * dim1 + offset];
				else
					results_stream << X[i1 * dim2 * dim3 + i2 * dim3 + i3  + offset];

				results_stream << ( i1 != dim1 - 1 ? "," : "" );
			}
			results_stream << ( i2 != dim2 - 1 ? "," : "" );
		}
		results_stream << std::endl;
	}

}

int main(const int argc , const  char** argv)
{
	uint g = 2;
	std::size_t Nu = 512;
	std::size_t Nv = 1000;

	uint domain_chooser = 0;

//	LAXODE_INTEGRATION_METHOD line_method = RUNGE_KUTTA4;
//	LAXODE_INTEGRATION_METHOD line_method = RUNGE_KUTTA_DOPRI5_CONTROLLED;
	LAXODE_INTEGRATION_METHOD line_method = RUNGE_KUTTA_DOPRI5_DENSE;
//	LAXODE_INTEGRATION_METHOD grid_method = RUNGE_KUTTA4;
//	LAXODE_INTEGRATION_METHOD grid_method = RUNGE_KUTTA_DOPRI5_CONTROLLED;
	LAXODE_INTEGRATION_METHOD grid_method = RUNGE_KUTTA_DOPRI5_DENSE;

	value_type_real precisiongoal = precisiongoal_default;
//	LAXODE_INTEGRATION_DEVICE grid_device = CUDA_SOLVER;
	LAXODE_INTEGRATION_DEVICE grid_device = CPU_SOLVER;

	std::cout << "Lax ODE Test" << std::endl;

	if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
		std::cout << "Arguments:" << std::endl;
		std::cout << "--Nu=INTEGER (grid points for line integration)" << std::endl;
		std::cout << "--Nv=INTEGER (grid points for grid integration)" << std::endl;
		std::cout << "--line_method=INTEGER" << std::endl;
		std::cout << "--grid_method=INTEGER" << std::endl;
		std::cout << "--grid_device=INTEGER" << std::endl;
		std::cout << "--domain=INTEGER" << std::endl;
		std::cout << "--precisiongoal=FLOAT" << std::endl;
		std::cout << "--csv_output=FILENAME (will be overwritten)" << std::endl;
		return 0;
	}

	std::cout << "Loading defaults and evaluating arguments" << std::endl;

	if (checkCmdLineFlag(argc, (const char **)argv, "grid_device"))
	{
		grid_device = (LAXODE_INTEGRATION_DEVICE)getCmdLineArgumentInt(argc, argv, "grid_device");

		std::cout << "grid_device=" << grid_device << " ";

	}

	if (checkCmdLineFlag(argc, (const char **)argv, "line_method"))
	{
		line_method = (LAXODE_INTEGRATION_METHOD)getCmdLineArgumentInt(argc, argv, "line_method");

		std::cout << "line_method=" << line_method << " ";

	}

	if (checkCmdLineFlag(argc, (const char **)argv, "grid_method"))
	{
		grid_method = (LAXODE_INTEGRATION_METHOD)getCmdLineArgumentInt(argc, argv, "grid_method");

		std::cout << "grid_method=" << grid_method << " ";

	}


	if (checkCmdLineFlag(argc, (const char **)argv, "domain"))
	{
		domain_chooser = getCmdLineArgumentInt(argc, argv, "domain");

		std::cout << "domain=" << domain_chooser << " ";
		std::cout << " ";
		if (domain_chooser < 1 || domain_chooser > 3)
		{
			std::cout << std::endl;
			printf("Illegal argument - domain must be in [1,3]\n");
			return 1;
		}
	}


	if (checkCmdLineFlag(argc, (const char **)argv, "Nu"))
	{
		Nu = getCmdLineArgumentInt(argc, argv, "Nu");

		std::cout << "Nu=" << Nu << " ";

	}

	if (checkCmdLineFlag(argc, (const char **)argv, "Nv"))
	{
		Nv = getCmdLineArgumentInt(argc, argv, "Nv");

		std::cout << "Nv=" << Nv << " ";

	}

	if (checkCmdLineFlag(argc, (const char **)argv, "precisiongoal"))
	{
		precisiongoal = getCmdLineArgumentFloat(argc, argv, "precisiongoal");

		std::cout << "precisiongoal=" << precisiongoal << " ";

	}

	std::cout << std::endl;

	std::size_t dim = xlab::surf::cuda::laxode::laxode_dimension_frame();
	std::size_t res_size = xlab::surf::cuda::laxode::laxode_result_size( Nu , Nv);
	// Wente Torus
	const value_type_real arr_X0[] = {1,0,-0.03031390398319432783124200350541194672841107,0,0,0,-0.2825269372599843036146124128463817986231055,0, 1, 0, 0, 0, 0, 0, 0, 0};
	std::vector<value_type_real> X0 (arr_X0, arr_X0 + sizeof(arr_X0) / sizeof(arr_X0[0]));

	// Sympoint is 1
	value_type_real sym_re = 1;
	value_type_real sym_im = 0;

	// Wente domain
	xlab::surf::cuda::laxode::laxode_domain domain1 = {
			0 ,0 , //origin
			12.865155157904928 ,0 , //first per
			0,19.961498805657186 ,  //second per
			Nu, Nv //gridcount
	};

	//Test domain
	xlab::surf::cuda::laxode::laxode_domain domain2 = {
				0 ,0 , //origin
				1 , 1 , //first per
				1, -1 ,  //second per
				Nu, Nv //gridcount
	};

	//Test domain
	xlab::surf::cuda::laxode::laxode_domain domain3 = {
			0 ,0 , //origin
			1 , 0 , //first per
			0, 1 ,  //second per
			Nu, Nv //gridcount
	};

	xlab::surf::cuda::laxode::laxode_domain domain0;

	switch (domain_chooser) {
	default:
	case 1:
		domain0 = domain1;
		break;
	case 2:
		domain0 = domain2;
		break;
	case 3:
		domain0 = domain3;
		break;
	}

	std::vector<value_type_real> results (res_size);
	laxode_integrate(results, g, X0, sym_re, sym_im, domain0, line_method, grid_method, precisiongoal, grid_device);

	if (checkCmdLineFlag(argc, (const char **)argv, "csv_output"))
	{
		std::vector<char *> filename(50);
		getCmdLineArgumentString(argc, argv, "csv_output", filename.data());
		std::cout << "Saving results to: " << filename[0] << std::endl;

		std::ofstream results_file;
		results_file.open (filename[0]);

		results_file << dim/2 << "," << Nu << "," << Nv << std::endl;

		write_dim3_ofstream (results_file, results , dim /2 , Nu , Nv , true);
		write_dim3_ofstream (results_file, results , dim /2 , Nu , Nv , true, Nv * Nu * dim /2);

		results_file.close();
	}

	std::cout << "Test finished"<< std::endl;

	return 0;
}

