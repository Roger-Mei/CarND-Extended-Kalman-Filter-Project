#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */

  // Initialization
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // Corner case check
  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
     std::cout << "The estimation or ground truth dataset is invalid!!!" << std::endl;
  }

  // Accumulate squared residuals
  for (int idx = 0; idx < estimations.size(); idx++)
  {
     VectorXd residual = estimations[idx] - ground_truth[idx];
     residual = residual.array() * residual.array();
     rmse += residual;
  }

  // Calculate the mean
  rmse = rmse / estimations.size();

  // Calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */

  // Initialization
  MatrixXd Hj_(3, 4);

  // Retrieve info
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Precomputation for avoiding repeated calculation
  float a1 = px * px + py * py;
  float a2 = sqrt(a1);
  float a3 = a1 * a2;

  // Corner case check
  if (fabs(a1) < 0.0001)
  {
     std::cout << "Invalid calculation for H_j!!!" << std::endl;
     return Hj_;
  }

  // Construct the Jacobian matrix
  Hj_ << (px / a2), (py / a2), 0, 0,
         -(py / a1), (px / a1), 0, 0,
         py*(vx*py - vy*px) / a3, px*(px*vy - py*vx) / a3, px / a2, py / a2;

   return Hj_;
}
