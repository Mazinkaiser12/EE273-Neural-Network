#pragma once
#include <Eigen/Dense>

Eigen::MatrixXd sigmoid(const Eigen::MatrixXd m);
Eigen::MatrixXd Tanh(const Eigen::MatrixXd m);
Eigen::MatrixXd softmax(const Eigen::MatrixXd m);
Eigen::MatrixXd ReLU(const Eigen::MatrixXd m);