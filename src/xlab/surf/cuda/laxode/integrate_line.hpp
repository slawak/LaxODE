// Copyright (C) 2012 Wjatscheslaw Kewlin

#ifndef xlab_surf_cuda_laxode_integrate_line_INCLUDED
#define xlab_surf_cuda_laxode_integrate_line_INCLUDED

namespace xlab {
namespace surf {
namespace cuda {
namespace laxode {

struct laxode_domain1d {
	value_type_real o_x;
	value_type_real o_y;
	value_type_real u_x;
	value_type_real u_y;
	uint N_u;

	laxode_domain1d (
			value_type_real o_x_,
			value_type_real o_y_,
			value_type_real u_x_,
			value_type_real u_y_,
			uint N_u_) :
				o_x (o_x_) ,
				o_y (o_y_) ,
				u_x (u_x_) ,
				u_y (u_y_) ,
				N_u (N_u_)
	{}
};

template < class state_type >
int integrate_line(
		state_type &results ,
		uint const g ,
		state_type const &X0 ,
		value_type_real const sym_re ,
	    value_type_real const sym_im ,
	    laxode_domain1d const &line ,
	    enum LAXODE_INTEGRATION_METHOD line_method ,
	    value_type_real precisiongoal )
{

#ifdef INFO_PRINT
	boost::timer::cpu_timer timer;
	std::cout << "Setting up 1D integration.. " << std::endl;
	std::cout << std::setprecision(15) << "(" << line.o_x << " , " << line.o_y << ")";
	std::cout << " to ";
	std::cout << std::setprecision(15) << "(" << line.u_x << " , " << line.u_y << ")";
	std::cout << " with " << line.N_u << " points";
	std::cout << std::endl;
#endif

//#ifdef DEBUG_PRINT
//	std::cout << "X0_raw=";
//	write_vector_cout(X0);
//#endif

	typedef xlab::surf::cuda::laxode::lax_system_single<state_type,value_type_real> lax_type;
	typedef xlab::surf::cuda::laxode::lax_observer_single<state_type,value_type_real> observer_type;

	std::size_t Nt = line.N_u;
	value_type_real dx = line.u_x - line.o_x;
	value_type_real dy = line.u_y - line.o_y;
	value_type_real dt = 1 / (double)(Nt - 1);
	value_type_real t0 = 0;
	lax_type lax ( g , X0 , dx , dy , sym_re, sym_im );
	observer_type observer(g , results , Nt , t0 , dt );

#ifdef INFO_PRINT
	std::cout << "Integrating..." << std::endl;
	timer.start();
#endif

	using namespace boost::numeric::odeint;
	// simple non controlled stepper with one step per grid step
	//typedef runge_kutta4< state_type , value_type_real , state_type , value_type_real , thrust_algebra , thrust_operations > stepper_type_simple;
	typedef runge_kutta4< state_type , value_type_real , state_type , value_type_real > stepper_type_simple;
	// create error stepper, can be used with make_controlled or make_dense_output
	//typedef runge_kutta_dopri5< state_type , value_type_real , state_type , value_type_real, thrust_algebra , thrust_operations > stepper_type;
	typedef runge_kutta_dopri5< state_type , value_type_real , state_type , value_type_real > stepper_type;


	switch (line_method) {
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
	write_dim2_cout ( results , Nt , observer.get_dim_results());
#endif
#ifdef INFO_PRINT_VERBOSE
	std::cout << "result at ";
	std::cout << std::setprecision(15) << "(" << line.u_x << " , " << line.u_y << ") = ";
	write_dim2_cout ( results , Nt , observer.get_dim_results(), true, 0, Nt - 1);
#endif

	return 0;
}


} // namespace laxode
} // namespace cuda
} // namespace surf
} // namespace xlab

#endif // xlab_surf_cuda_laxode_integrate_line_INCLUDED
