#include <ct/core/core.h>
#include <ct/optcon/optcon.h>
#include <ct/optcon/constraint/constraint-impl.h>
#include <ct/optcon/solver/lqp/HPIPMInterface-impl.hpp>
#include <ct/optcon/solver/lqp/HPIPMInterface.hpp>
#include <ct/optcon/nlp/Nlp>
#include "ros/ros.h"
#include "tocabi_controller/foottrajectory.h"
#include <vector>
#include <string>
#include <time.h>

using namespace ct;
using namespace ct::core;
using namespace ct::optcon;
using std::shared_ptr;


double w_ = 0.8;
double m = 70;
double G = 9.81;
double z_c = 0.87;
double dt = 0.005;
const size_t state_dim = 5;
const size_t control_dim = 2;
static const std::string exampleDir = "/home/jhk/catkin_ws/src/my_ct_project";

/*!
  @brief Constraint for flywheel
*/

void rfCallback(const tocabi_controller::foottrajectory &msg)
{
   // int a = msg.posx.size();
}

void lfCallback(const tocabi_controller::foottrajectory &msg)
{ 
   // int a = msg.posx.size();
}

class flywheelLinear : public core::LinearSystem<state_dim, control_dim>
{
public:

    state_matrix_t A_;
    state_control_matrix_t B_;

    const state_matrix_t& getDerivativeState(const core::StateVector<state_dim>& x,
        const core::ControlVector<control_dim>& u,
        const double t = 0.0) override
    {
        A_<< 0, 1, 0, 0, 0,   w_, 0, -w_, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0, 0, 1,   0, 0, 0, 0, 0;
        return A_;
    }

    const state_control_matrix_t& getDerivativeControl(const core::StateVector<state_dim>& x,
        const core::ControlVector<control_dim>& u,
        const double t = 0.0) override
    {
        B_ << 0, 0,  0, 1/(m*G),  1, 0,  0,0,   0,1;
        return B_;
    }

    flywheelLinear* clone() const override { return new flywheelLinear(); };
};

std::shared_ptr<CostFunctionQuadratic<state_dim, control_dim>> createflywheelCostFunction(const core::StateVector<state_dim>& x_final)
{
    Eigen::Matrix<double, state_dim, state_dim> Q;
    Q.setIdentity();
    Q = 1*Q;

    Eigen::Matrix<double, control_dim, control_dim> R;
    R.setIdentity();

    Eigen::Matrix<double, control_dim, 1> u_nominal;
    u_nominal.setZero();

    Eigen::Matrix<double, state_dim, 1> x_nominal;
    x_nominal.setZero();

    Eigen::Matrix<double, state_dim, state_dim> Q_final;
    Q_final.setIdentity();

    std::shared_ptr<CostFunctionQuadratic<state_dim, control_dim>> quadraticCostFunction(
        new CostFunctionQuadraticSimple<state_dim, control_dim>(Q, R, x_nominal, u_nominal, x_final, Q_final));

    return quadraticCostFunction;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mpc_controller");

    ros::NodeHandle n;

    ros::Subscriber RF=n.subscribe("/tocabi/RF",10,rfCallback);
    ros::Subscriber LF=n.subscribe("/tocabi/LF",10,lfCallback);

    std::vector<std::shared_ptr<LQOCSolver<state_dim, control_dim>>> lqocSolvers;
    std::shared_ptr<LQOCSolver<state_dim, control_dim>> hpipmSolver(new HPIPMInterface<state_dim, control_dim>());
    
    ct::core::Time timeHorizon = 1.1;
    NLOptConSettings ilqr_settings;
    ilqr_settings.dt = dt;  // the control discretization in [sec]
    ilqr_settings.max_iterations = 10;
    ilqr_settings.lqoc_solver_settings.num_lqoc_iterations = 10;  
    ilqr_settings.printSummary = false;
    ilqr_settings.nThreads = 6;
    ilqr_settings.debugPrint = false;
    ilqr_settings.lqocp_solver = NLOptConSettings::LQOCP_SOLVER::HPIPM_SOLVER;


    hpipmSolver->configure(ilqr_settings);
    size_t K = ilqr_settings.computeK(timeHorizon);
    ilqr_settings.print();

    lqocSolvers.push_back(hpipmSolver);

    std::vector<std::shared_ptr<LQOCProblem<state_dim, control_dim>>> problems;
    std::shared_ptr<LQOCProblem<state_dim, control_dim>> lqocProblem1(new LQOCProblem<state_dim, control_dim>(K));

    problems.push_back(lqocProblem1);

    ct::core::ControlVector<control_dim> u0;
    u0.setZero();  // by definition
    // initial state
    ct::core::StateVector<state_dim> x0;
    x0.setZero();  // by definition
    x0 << 0.5, 0.5, 0, 0, 0;
    // desired final state
    ct::core::StateVector<state_dim> xf;
    xf.setZero();// << -1, -1, -1, -1, -1;
    xf << -0.8, 0, 0.3, 0.3, 0.3;

    std::shared_ptr<flywheelLinear> flywheelDynamics(new flywheelLinear()); 
    std::shared_ptr<core::LinearSystem<state_dim, control_dim>> exampleSystem(new flywheelLinear());
    core::SensitivityApproximation<state_dim, control_dim> discreteExampleSystem(
      ilqr_settings.dt, exampleSystem, core::SensitivityApproximationSettings::APPROXIMATION::FORWARD_EULER);

    auto costFunction = createflywheelCostFunction(xf);
    
    ct::core::StateVector<state_dim> b;
    b.setZero();

    // initialize the optimal control problems for both solvers
    problems[0]->setFromTimeInvariantLinearQuadraticProblem(discreteExampleSystem, *costFunction, x0, x0, ilqr_settings.dt);
    
    lqocSolvers[0]->setProblem(problems[0]);
    
    //Soft constraint (slack varaible)
    problems[0]->Zl_.resize(K);
    problems[0]->Zu_.resize(K);

    for (size_t i = 0; i < K; i++)
    {
        problems[0]->Zl_[i].resize(8, 8);
        problems[0]->Zu_[i].resize(8, 8);
        problems[0]->Zl_[i].setZero();
        problems[0]->Zu_[i].setZero();
        problems[0]->Zl_[i](7,7) = 1.0;
        problems[0]->Zu_[i](7,7) = 1.0;
        problems[0]->s_ub_[i].setZero();
        problems[0]->s_ub_[i].setZero();
        problems[0]->s_ub_[i](7) = 0;
        problems[0]->s_lb_[i] = -1 * problems[0]->s_ub_[i];
        problems[0]->s_I_[i].setZero();
        problems[0]->s_I_[i](7) = 1;
    }

    lqocSolvers[0]->configureSoftConstraints(problems[0]);
  
    Eigen::VectorXi x_box_sparsity(5);
    x_box_sparsity << 0,1,2,3,4;
    Eigen::VectorXd x_lb(5);
    Eigen::VectorXd x_ub(5);

    x_ub <<2, 2, 2, 2, 2;
    x_lb = -1 * x_ub;  

    Eigen::VectorXd x_s(5);
    x_s.setZero();
    ct::core::StateVectorArray<state_dim> x_nom(K, x_s);
    
    problems[0]->setIntermediateStateBoxConstraints(5, x_lb, x_ub, x_box_sparsity, x_nom);

    x_ub = xf;
    x_lb = x_ub;  

    problems[0]->setTerminalBoxConstraints(5, x_lb, x_ub, x_box_sparsity, x_nom.back());

    lqocSolvers[0]->configureStateBoxConstraints(problems[0]);

    lqocSolvers[0]->setProblem(problems[0]);

    //General constraint
    for (size_t i = 0; i < K; i++)
    {
        problems[0]->d_lb_[i].resize(1, 1);
        problems[0]->d_lb_[i].setZero();
        problems[0]->d_ub_[i].resize(1, 1);
        problems[0]->d_ub_[i].setZero();
        problems[0]->C_[i].resize(1, state_dim);
        problems[0]->C_[i].setZero();
        problems[0]->D_[i].resize(1, control_dim);
        problems[0]->D_[i].setZero();
    }

    for(int i = 0; i < K; i++)
    {
        problems[0]->C_[i](0) = 1.0;
        problems[0]->C_[i](1) = 1.0;
        problems[0]->C_[i](2) = 1.0;
        problems[0]->C_[i](3) = 1.0;
        problems[0]->C_[i](4) = 1.0;  
        problems[0]->d_lb_[i](0) = 0.2;  
        problems[0]->d_ub_[i](0) = 0.2;
        
        problems[0]->C_[i](0) = 1.0;
        problems[0]->C_[i](1) = 1.0;
        problems[0]->C_[i](2) = 1.0;
        problems[0]->C_[i](3) = 1.0;
        problems[0]->C_[i](4) = 1.0;
        problems[0]->d_lb_[i](0) = 0.2;  
        problems[0]->d_ub_[i](0) = 0.2; 
    }

    lqocSolvers[0]->setProblem(problems[0]);

    // allocate memory (if required)
    lqocSolvers[0]->initializeAndAllocate();

    time_t start, end;

    start = clock();

    // solve the problems...
    lqocSolvers[0]->solve();
    
    // postprocess data
    lqocSolvers[0]->computeStatesAndControls();

    end = clock(); 
    lqocSolvers[0]->computeFeedbackMatrices();

    lqocSolvers[0]->compute_lv();

    lqocSolvers[0]->computeSlAndSu();

    // retrieve solutions from both solvers
    auto xSol_riccati = lqocSolvers[0]->getSolutionState();
    auto uSol_riccati = lqocSolvers[0]->getSolutionControl();
    auto slSol_riccati = lqocSolvers[0]->getSolutionSl();
    auto suSol_riccati = lqocSolvers[0]->getSolutionSu();

    ct::core::FeedbackArray<state_dim, control_dim> KSol_riccati = lqocSolvers[0]->getSolutionFeedback();
    ct::core::ControlVectorArray<control_dim> lv_sol_riccati = lqocSolvers[0]->get_lv();

    for (size_t j = 0; j < xSol_riccati.size()/5; j++)
    {
    /*    std::cout << "x solution from riccati solver:" << std::endl;
        std::cout << xSol_riccati[j].transpose() << std::endl;

        std::cout << "u solution from riccati solver:" << std::endl;
        std::cout << uSol_riccati[j].transpose() << std::endl;
*/
        std::cout << "sl solution from riccati solver:" << std::endl;
        std::cout << slSol_riccati[j].transpose()<< std::endl;
        

        std::cout << "su solution from riccati solver:" << std::endl;
        std::cout << suSol_riccati[j].transpose()<< std::endl;
    }
    double result;
    result = (double)(end - start)/ CLOCKS_PER_SEC;
    std::cout << result<< std::endl;;
}

