// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_integrate_grid_INCLUDED
#define xlab_surf_cuda_laxode_integrate_grid_INCLUDED

namespace xlab {
namespace surf {
namespace cuda {
namespace laxode {

template < class state_type, class thrust_tag> //, class index_type >
int integrate_grid(
		state_type &results ,
		uint const g ,
		state_type const &X0 ,
		value_type_real const sym_re ,
	    value_type_real const sym_im ,
	    laxode_domain const &dom ,
	    enum LAXODE_INTEGRATION_METHOD grid_method ,
	    value_type_real precisiongoal )
{
#ifdef INFO_PRINT
	boost::timer::cpu_timer timer;
	std::cout << "Setting up 2D integration.. " << std::endl;
	std::cout << std::setprecision(15) << "(" << dom.o_x << " , " << dom.o_y << ")";
	std::cout << " to ";
	std::cout << std::setprecision(15) << "(" << dom.v_x << " , " << dom.v_y << ")";
	std::cout << " with " << dom.N_v << " points";
	std::cout << std::endl;
#endif

#ifdef DEBUG_PRINT
//	std::cout << "X0_raw=";
//	write_vector_cout(X0);
	std::cout << "X0=";
	write_dim2_cout ( X0 , dom.N_u , dimension_R3(g));
#endif

	typedef xlab::surf::cuda::laxode::lax_system_thrust<state_type,value_type_real, thrust_tag> lax_type;
	//typedef xlab::surf::cuda::laxode::lax_observer_thrust<state_type,value_type_real, thrust_tag, index_type > observer_type;
	typedef xlab::surf::cuda::laxode::lax_observer_thrust<state_type,value_type_real, thrust_tag> observer_type;

	std::size_t Nu = dom.N_u;
	std::size_t Nt = dom.N_v;
	value_type_real dx = dom.v_x - dom.o_x;
	value_type_real dy = dom.v_y - dom.o_y;
	value_type_real dt = 1 / (double)(Nt - 1);
	value_type_real t0 = 0;
	lax_type lax ( g , Nu , X0 , dx , dy , sym_re, sym_im );

#ifdef INFO_PRINT_VERBOSE
	std::cout << "Setting up observer..." << std::endl;
	timer.start();
#endif
	observer_type observer(g , Nu , results, Nt , t0 , dt );
#ifdef INFO_PRINT_VERBOSE
	std::cout << "Set up in: " << timer.format() << std::endl;
#endif

#ifdef INFO_PRINT
	std::cout << "Integrating..." << std::endl;
	timer.start();
#endif

	using namespace boost::numeric::odeint;
	// simple non controlled stepper with one step per grid step
	typedef runge_kutta4< state_type , value_type_real , state_type , value_type_real, thrust_algebra , thrust_operations > stepper_type_simple;
	// create error stepper, can be used with make_controlled or make_dense_output
	typedef runge_kutta_dopri5< state_type , value_type_real , state_type , value_type_real, thrust_algebra , thrust_operations > stepper_type;

	switch (grid_method) {
	case RUNGE_KUTTA4 :
#ifdef INFO_PRINT
		std::cout << "Using RUNGE_KUTTA4" << std::endl;
		timer.start();
#endif

		integrate_n_steps( stepper_type_simple() ,
				lax , lax.getX() , t0 , dt , Nt - 1 ,  boost::ref( observer ));
		break;

	case RUNGE_KUTTA_DOPRI5_CONTROLLED :
#ifdef INFO_PRINT
		std::cout << "Using RUNGE_KUTTA_DOPRI5_CONTROLLED ";
		std::cout << "with precisiongoal=" << precisiongoal << std::endl;
		timer.start();
#endif

		integrate_n_steps( make_controlled( precisiongoal , precisiongoal , stepper_type() ) ,
				lax , lax.getX() , t0 , dt , Nt - 1 ,  boost::ref( observer ));
		break;

	case RUNGE_KUTTA_DOPRI5_DENSE :
	default :
#ifdef INFO_PRINT
		std::cout << "Using RUNGE_KUTTA_DOPRI5_DENSE ";
		std::cout << "with precisiongoal=" << precisiongoal << std::endl;
		timer.start();
#endif

		integrate_n_steps( make_dense_output( precisiongoal , precisiongoal , stepper_type() ) ,
				lax , lax.getX() , t0 , dt , Nt - 1 ,  boost::ref( observer ));
		break;
	}

#ifdef INFO_PRINT
	std::cout << "Integrated in: " << timer.format() << std::endl;
#endif

#ifdef DEBUG_PRINT
//	std::cout << "results_raw=";
//	write_vector_cout( results );
	std::cout << "results=";
	write_dim3_cout ( results , observer.get_dim_result() /2 , Nu , Nt);
	write_dim3_cout ( results ,
			observer.get_dim_result() /2 , Nu , Nt ,
			true, Nt * Nu * observer.get_dim_result() /2);
#endif
#ifdef INFO_PRINT_VERBOSE
//	std::cout << "results=";
//	write_dim3_cout ( results , observer.get_dim_result() /2 , Nu , Nt);
//	write_dim3_cout ( results ,
//			observer.get_dim_result() /2 , Nu , Nt ,
//			true, Nt * Nu * observer.get_dim_result() /2);

	std::cout << "result at ";
	std::cout << std::setprecision(15) << "(" << dom.v_x << " , " << dom.v_y << ") = ";
	write_dim3_cout ( results ,
			observer.get_dim_result() /2 , Nu * (Nt - 1) + 1 , 1 ,
			true, 0 ,
			0 , Nu * (Nt - 1), 0);
	write_dim3_cout ( results ,
			observer.get_dim_result() /2 , Nu * (Nt - 1) + 1 , 1 ,
			true, Nt * Nu * observer.get_dim_result() /2 ,
			0 , Nu * (Nt - 1), 0);

	std::cout << "result at ";
	std::cout << std::setprecision(15) << "(" << (dom.u_x + dom.v_x) << " , " << (dom.u_y + dom.v_y) << ") = ";
	write_dim3_cout ( results ,
			observer.get_dim_result() /2 , Nu , Nt ,
			true, 0 ,
			0 , Nu - 1, Nt - 1);
	write_dim3_cout ( results ,
			observer.get_dim_result() /2 , Nu , Nt ,
			true, Nt * Nu * observer.get_dim_result() /2 ,
			0, Nu - 1, Nt - 1);
#endif

	return 0;
}


} // namespace laxode
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_laxode_integrate_grid_INCLUDED
