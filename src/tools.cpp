#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) 
{
  /**
  TODO:
    * Calculate the RMSE here.
    *
  */
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if(estimations.size() == 0 ||
       estimations.size() != ground_truth.size())
    {
        std::cout << "Invalid estimation or ground_truth data" << std::endl;
        return rmse;
    }

    for(unsigned int i=0; i < estimations.size() ; ++i)
    {
        VectorXd res = estimations[i] - ground_truth[i];

        res = res.array() * res.array();
        rmse += res;
    }

    rmse = rmse/estimations.size();
    return rmse.array().sqrt();
}
