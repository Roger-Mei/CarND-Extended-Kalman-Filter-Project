#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
  
  x_ = F_ * x_; // Predict mean
  MatrixXd F_t = F_.transpose();
  P_ = F_ * P_ * F_t + Q_; // Predict Covariance 
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  // Normalization of angle
  while (y(1)>M_PI)
  {
  y(1) -= 2 * M_PI;
  }
  while (y(1)<-M_PI)
  {
  y(1) += 2 * M_PI;
  }

  MatrixXd H_t = H_.transpose();
  MatrixXd S = H_ * P_ * H_t + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * H_t;
  MatrixXd K = PHt * Si;

  // Estimation
  x_ = x_ + (K * y); // update mean
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_; // update covariance matrix 

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */

  float rho = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
  float phi = atan2(x_(1), x_(0));
  float rho_dot;
  // Corner case check
  if (fabs(rho) < 0.0001)
  {
    rho_dot = 0;
  }
  else
  {
    rho_dot = (x_(0) * x_(2) + x_(1) * x_(3))/rho;
  }

  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;
  
  VectorXd y = z - z_pred;
  // Normalization of angle
  while (y(1)>M_PI)
  {
  y(1) -= 2 * M_PI;
  }
  while (y(1)<-M_PI)
  {
  y(1) += 2 * M_PI;
  }

  MatrixXd H_t = H_.transpose();
  MatrixXd S = H_ * P_ * H_t + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * H_t;
  MatrixXd K = PHt * Si;

  // Estimation
  x_ = x_ + (K * y); // update mean
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_; // update covariance matrix 

}
