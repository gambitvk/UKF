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

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_x_;
  radar_n_ = 3;
  lidar_n_ = 2;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.1;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  x_ = VectorXd(n_x_);
  x_ << 0.1,0.1,0.1,0.1,0.1;

  P_ = MatrixXd(n_x_,n_x_);
  P_ << 0.01, 0, 0, 0, 0,
        0, 0.01, 0, 0, 0,
        0, 0, 0.01, 0, 0,
        0, 0, 0, 0.01, 0,
        0, 0, 0, 0, 0.01;

  Xsig_ =  MatrixXd(n_x_, 2 * n_x_ + 1);
  Xsig_.fill(0.0);

  x_aug_ = VectorXd(n_aug_);
  x_aug_.fill(0.0);

  P_aug_ = MatrixXd(n_aug_, n_aug_);
  P_aug_.fill(0.0);

  XsigAug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  XsigAug_.fill(0.0);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  radarz_pred_ = VectorXd(radar_n_);
  radarz_pred_.fill(0.0);

  radarS_ = MatrixXd(radar_n_,radar_n_);
  radarS_.fill(0.0);

  radarZsig_ = MatrixXd(radar_n_, 2 * n_aug_ + 1);
  radarZsig_.fill(0.0);

  lidarz_pred_ = VectorXd(lidar_n_);
  lidarz_pred_.fill(0.0);

  lidarS_ = MatrixXd(lidar_n_,lidar_n_);
  lidarS_.fill(0.0);

  lidarZsig_ = MatrixXd(lidar_n_, 2 * n_aug_ + 1);
  lidarZsig_.fill(0.0);

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i=1; i<2*n_aug_+1; i++) 
  {  //2n+1 weights
    weights_(i) = 0.5/(n_aug_+lambda_);
  }

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_pack) 
{
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  err( "process measurement");
    if(!is_initialized_)
    {

        err("initialise started");

        if(meas_pack.sensor_type_ == MeasurementPackage::RADAR)
        {
            err("radar");
            setX(
                 meas_pack.raw_measurements_(0),
                 meas_pack.raw_measurements_(1),
                 meas_pack.raw_measurements_(2)
                );
        }
        else
        {
            err("laser");
            setX(meas_pack.raw_measurements_(0),
                 meas_pack.raw_measurements_(1));
        }

        time_us_ = meas_pack.timestamp_;
        is_initialized_ = true;
        err( "initialise ended");
        return;
    }


    double dt = processTime(meas_pack.timestamp_);

   // err("before predition");
   // Prediction(dt);
   // err("After predition");

    while (dt > 0.1)
    {
        const double dtt = 0.05;
        Prediction(dtt);
        dt -= dtt;
    }

    Prediction(dt);

    if (meas_pack.sensor_type_ == MeasurementPackage::RADAR && use_radar_) 
    {
        err("before predict radar");
        PredictRadarMeasurement();
        err("before update radar");
        UpdateRadar(meas_pack);
        err("After update radar");
    }
    else if(meas_pack.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
        err("before predict lidar");
        PredictLidarMeasurement();
        err("before update lidar");
        UpdateLidar(meas_pack);
        err("before update liddar");
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  GenerateSigmaPoints();
  AugmentedSigmaPoints();
  SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();
}


MatrixXd UKF::squareMat(const MatrixXd& mat) const
{
    return mat.llt().matrixL();
}


void UKF::GenerateSigmaPoints() 
{
  MatrixXd A = squareMat(P_);
  //set first column of sigma point matrix
  Xsig_.col(0)  = x_;

  //set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig_.col(i+1)     = x_ + sqrt(lambda_+n_x_) * A.col(i);
    Xsig_.col(i+1+n_x_) = x_ - sqrt(lambda_+n_x_) * A.col(i);
  }
}

void UKF::AugmentedSigmaPoints() 
{
  //create augmented mean state
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  //create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5) = std_a_*std_a_;
  P_aug_(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = squareMat(P_aug_);

  //create augmented sigma points
  XsigAug_.col(0)  = x_aug_;
  for (int i = 0; i< n_aug_; i++)
  {
    XsigAug_.col(i+1)       = x_aug_ + sqrt(lambda_+n_aug_) * L.col(i);
    XsigAug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * L.col(i);
  }
}


void UKF::SigmaPointPrediction(const double &delta_t)
{
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = XsigAug_(0,i);
    double p_y = XsigAug_(1,i);
    double v = XsigAug_(2,i);
    double yaw = XsigAug_(3,i);
    double yawd = XsigAug_(4,i);
    double nu_a = XsigAug_(5,i);
    double nu_yawdd = XsigAug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() 
{
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    //while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    //while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    x_diff(3) = normalize(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}



void UKF::PredictRadarMeasurement() 
{
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  =  Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    radarZsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    radarZsig_(1,i) = atan2(p_y,p_x);                                 //phi
    radarZsig_(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  radarz_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) 
  {
      radarz_pred_ = radarz_pred_ + weights_(i) * radarZsig_.col(i);
  }

  radarS_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  //2n+1 simga points
    //residual
    VectorXd z_diff = radarZsig_.col(i) - radarz_pred_;

    //angle normalization
    //while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    //while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    z_diff(1) = normalize(z_diff(1));

    radarS_ = radarS_ + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(radar_n_,radar_n_);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  radarS_ = radarS_ + R;
}


void UKF::PredictLidarMeasurement() 
{
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  //2n+1 simga points

    // extract values for better readibility
    lidarZsig_(0,i) = Xsig_pred_(0,i);
    lidarZsig_(1,i) = Xsig_pred_(1,i);
  }

  //mean predicted measurement
  lidarz_pred_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) 
  {
      lidarz_pred_ = lidarz_pred_ + weights_(i) * lidarZsig_.col(i);
  }

  lidarS_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) 
  {  //2n+1 simga points
    //residual
    VectorXd z_diff = lidarZsig_.col(i) - lidarz_pred_;

    //angle normalization
    //while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    //while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    z_diff(1)= normalize(z_diff(1));

    lidarS_ = lidarS_ + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(lidar_n_,lidar_n_);
  R <<    std_laspx_*std_laspx_, 0,
          0,std_laspy_*std_laspy_;
  lidarS_ = lidarS_ + R;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(lidar_n_); 
  z <<  meas_package.raw_measurements_(0),  meas_package.raw_measurements_(1);
 
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, lidar_n_);


  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = lidarZsig_.col(i) - lidarz_pred_;
    //angle normalization
   // while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
   // while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    z_diff(1) = normalize(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
   // while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
   // while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
   x_diff(3) = normalize(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * lidarS_.inverse();

  //residual
  //std::cout << z << std::endl;
  //err("----------------");
  //std::cout << lidarz_pred_ << std::endl;
  VectorXd z_diff = z - lidarz_pred_;

  //angle normalization
  //while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  //while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  z_diff(1) = normalize(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*lidarS_*K.transpose();

  NIS_laser_ = z_diff.transpose() * lidarS_.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) 
{
  /**
  TODO:
 
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
 
  You'll also need to calculate the radar NIS.
  */

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(radar_n_); 
  z <<  meas_package.raw_measurements_(0),  meas_package.raw_measurements_(1)
        ,meas_package.raw_measurements_(2);
 
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, radar_n_);


  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = radarZsig_.col(i) - radarz_pred_;
    //angle normalization
    //while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    //while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    z_diff(1) = normalize(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    //while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    //while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    x_diff(3) = normalize(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * radarS_.inverse();

  //residual
  VectorXd z_diff = z - radarz_pred_;

  //angle normalization
  //while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  //while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  z_diff(1) = normalize(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*radarS_*K.transpose();

  NIS_radar_ = z_diff.transpose() * radarS_.inverse() * z_diff;
}


void UKF::setX(const double &r,const  double &p ,const  double &rr)
{
    double vx = rr*cos(p);
    double vy = rr*sin(p);

    x_ << r*cos(p), r* sin(p) , sqrt(vx*vx+vy*vy) ,0,0;
}

void UKF::setX(const double &x,const  double &y)
{
    x_ << x,y,0.0,0.0,0.0;
}

double UKF::processTime(const long long &t)
{
    long long pt =  time_us_;
    time_us_ = t;
    return double((t - pt) / 1000000.0); //dt - expressed in seconds
}

void UKF::err(const std::string& st) const
{
     cout << st << endl;
    //return;
}

double UKF::normalize(const double& ang) const
{
    return atan2(sin(ang),cos(ang));
}
