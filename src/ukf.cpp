#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if false, ignore measurement type (except during init)
  use_laser_ = true;
  use_radar_ = true;

  // Process noise standard deviations
  //  longitudinal acceleration in m/s^2, yaw accelleration in rad/s^2
  std_a_ = 3;
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviations
  //  position1 in m, position2 in m
  std_laspx_ = 0.15;
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviations
  //  radius in m, angle in rad, radius change in m/s
  std_radr_ = 0.3;
  std_radphi_ = 0.03;
  std_radrd_ = 0.3;

  // measurement dimensions
  n_x_ = 5;
  n_aug_ = 7;
  n_sigpts_ = 1+2*n_aug_;

  // spreading factor
  lambda_ = 3-n_aug_;

  // initialize weights vector
  weights_ = VectorXd(n_sigpts_);
  weights_[0] = lambda_/(lambda_+n_aug_);
  for(int i=1; i<n_sigpts_; ++i){
    weights_[i] = 1/(2*(lambda_+n_aug_));
  }

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrices
  P_ = MatrixXd::Identity(n_x_,n_x_);
  P_aug_ = MatrixXd(n_aug_,n_aug_);
  P_aug_.topLeftCorner(n_x_,n_x_) = P_;
  P_aug_(n_x_,n_x_) = std_a_*std_a_;
  P_aug_(n_x_+1,n_x_+1) = std_yawdd_*std_yawdd_;

  // initial sigma predictions
  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sigpts_);

  NIS_eps_ = 0;

  // filter initialization is not done until first measurement is received
  is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if(!is_initialized_){
    /*
     * Initialize the state with the first measurement
    */
    cout << "UKF:" << endl;

    double px;
    double py;

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      px = rho*cos(phi);
      py = rho*sin(phi);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
      // update px and py covariances
      P_(0,0) = 0.2;
      P_(1,1) = 0.2;
    } else {
      cout << "UKF - ERROR - unknown sensor type: " << meas_package.sensor_type_ << endl;
      return;
    }

    x_[0] = px;
    x_[1] = py;

    // update v, phi, and phi_d covariances to realistic values
    P_(2,2) = 0.5;
    P_(3,3) = 0.5;
    P_(4,4) = 0.5;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  // get time since last measurement
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  cout << "dt: " << dt << endl;
  time_us_ = meas_package.timestamp_;

  // do prediction steps
  cout << "Predicting..." << endl;
  // predicting in small increments improves numerical stability
  while(dt > 0.1){
    Prediction(0.05);
    dt -= 0.05;
  }
  Prediction(dt);

  // do update steps
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    cout << "RADAR Update..." << endl;
  } else if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    cout << "LASER Update..." << endl;
  } else {
    cout << "UKF - ERROR - unknown sensor type: " << meas_package.sensor_type_ << endl;
    return;
  }
  Update(meas_package);

  // print result
  cout << "x: " << x_ << endl;
  cout << "P: " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double dt) {
  // create augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug[n_x_] = 0;
  x_aug[n_x_+1] = 0;
  // re-initialize x_ (will compute this later)
  x_.fill(0.0);

  // update augmented covariance matrix
  P_aug_.topLeftCorner(n_x_,n_x_) = P_;
  // re-initialize P_ (will compute this later)
  P_.fill(0.0);

  // square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  // create augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigpts_);
  Xsig_aug.col(0) = x_aug;
  double c1 = sqrt(lambda_ + n_aug_);
  for(int i=0; i<n_aug_; ++i){
    Xsig_aug.col(i+1       ) = x_aug + c1*L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - c1*L.col(i);
  }

  // predict augmented sigma points
  for(int i=0; i<n_sigpts_; ++i){
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double phi = Xsig_aug(3,i);
    double phi_d = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_phidd = Xsig_aug(6,i);

    double c_phi = cos(phi);
    double s_phi = sin(phi);
    double c_phi_dt = cos(phi+phi_d*dt);
    double s_phi_dt = sin(phi+phi_d*dt);

    double c2 = dt*dt/2;

    if(fabs(phi_d) < 0.0001){
      //state update when phi_d is 0
      Xsig_pred_.col(i) << px    + v*c_phi*dt + c2*c_phi*nu_a,
                           py    + v*s_phi*dt + c2*s_phi*nu_a,
                           v     + 0          + dt*nu_a,
                           phi   + 0          + c2*nu_phidd,
                           phi_d + 0          + dt*nu_phidd;
    } else {
      // state update when phi_d is not 0
      Xsig_pred_.col(i) << px    + v/phi_d*(s_phi_dt-s_phi) + c2*c_phi*nu_a,
                           py    + v/phi_d*(c_phi-c_phi_dt) + c2*s_phi*nu_a,
                           v     + 0                        + dt*nu_a,
                           phi   + phi_d*dt                 + c2*nu_phidd,
                           phi_d + 0                        + dt*nu_phidd;
    }

    // predict state mean
    x_ += weights_[i]*Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  for(int i=0; i<n_sigpts_; ++i){
    VectorXd Xdiff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (Xdiff(3)> M_PI) Xdiff(3)-=2.*M_PI;
    while (Xdiff(3)<-M_PI) Xdiff(3)+=2.*M_PI;

    P_ += weights_[i] * Xdiff * Xdiff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser or radar
 * measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::Update(MeasurementPackage meas_package) {
  int n_z = meas_package.raw_measurements_.size();

  // create measurement sigma points
  MatrixXd Zsig = MatrixXd::Zero(n_z, n_sigpts_);
  for(int i=0; i<n_sigpts_; ++i){
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      double px  = Xsig_pred_(0,i);
      double py  = Xsig_pred_(1,i);
      double v   = Xsig_pred_(2,i);
      double phi = Xsig_pred_(3,i);
      double c1 = sqrt(px*px+py*py);

      Zsig(0,i) = c1;
      // avoid divide by zero
      if(fabs(px) < 0.0001){
        Zsig(1,i) = atan2(py,0.0001);
      } else {
        Zsig(1,i) = atan2(py,px);
      }
      if(fabs(c1) < 0.0001){
        Zsig(2,i) = v*(px*cos(phi)+py*sin(phi))/0.0001;
      } else {
        Zsig(2,i) = v*(px*cos(phi)+py*sin(phi))/c1;
      }
    } else if(meas_package.sensor_type_ == MeasurementPackage::LASER){
      // extract px,py from predicted sigma points
      Zsig.col(i) = Xsig_pred_.block(0,i,2,1);
    }
  }

  // predict mean measuremnt
  VectorXd z_pred = VectorXd::Zero(n_z);
  for(int i=0; i<n_sigpts_; ++i){
    z_pred += weights_[i] * Zsig.col(i);
  }

  // create measurement covariance matrix
  MatrixXd S = MatrixXd::Zero(n_z,n_z);
  for(int i=0; i<n_sigpts_; ++i){
    VectorXd Zdiff = Zsig.col(i) - z_pred;
    //angle normalization
    while (Zdiff(1)> M_PI) Zdiff(1)-=2.*M_PI;
    while (Zdiff(1)<-M_PI) Zdiff(1)+=2.*M_PI;
    S += weights_[i] * Zdiff * Zdiff.transpose();
  }

  // add measurement noises covariance matrix
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    S(0,0) += std_radr_*std_radr_;
    S(1,1) += std_radphi_*std_radphi_;
    S(2,2) += std_radrd_*std_radrd_;
  } else if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    S(0,0) += std_laspx_*std_laspx_;
    S(1,1) += std_laspy_*std_laspy_;
  }

  // create cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_,n_z);
  for(int i=0; i<n_sigpts_; ++i){
    VectorXd Zdiff = Zsig.col(i) - z_pred;
    //angle normalization
    while (Zdiff(1)> M_PI) Zdiff(1)-=2.*M_PI;
    while (Zdiff(1)<-M_PI) Zdiff(1)+=2.*M_PI;
    VectorXd Xdiff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (Xdiff(3)> M_PI) Xdiff(3)-=2.*M_PI;
    while (Xdiff(3)<-M_PI) Xdiff(3)+=2.*M_PI;

    Tc += weights_[i] * Xdiff * Zdiff.transpose();
  }

  // Kalman gain
  MatrixXd Sinv = S.inverse();
  MatrixXd K = Tc*Sinv;

  VectorXd Zdiff = meas_package.raw_measurements_ - z_pred;
  //angle normalization
  while (Zdiff(1)> M_PI) Zdiff(1)-=2.*M_PI;
  while (Zdiff(1)<-M_PI) Zdiff(1)+=2.*M_PI;

  // predicted mean state and covariance
  x_ += K*Zdiff;
  P_ -= K*S*K.transpose();

  // Normalized Innovation Squared measure
  NIS_eps_ = (Zdiff.transpose()*Sinv*Zdiff)(0,0);
}
