#include "ActivationFunction.h"
#include <iostream>
#include <Eigen>
#include <string>


Eigen::MatrixXd sigmoid(const Eigen::MatrixXd m)
{
    /**
     * Function array() returns an Eigen::Array object, which is a wrapper around the data of the matrix.
     * The exp() function is applied to each element of the array, and the result is stored in a new array and then automatically converted back to a matrix.
     */
    return 1 / (1 + (-m).array().exp());
}

Eigen::MatrixXd Tanh(const Eigen::MatrixXd m) {
    /**
     * Function array() returns an Eigen::Array object, which is a wrapper around the data of the matrix.
     * The tanh() function is applied to each element of the array, and the result is stored in a new array and then automatically converted back to a matrix.
     */
    return m.array().tanh();
}

Eigen::MatrixXd ReLU(const Eigen::MatrixXd m) {
    // To be implemented
    return m;
}

Eigen::MatrixXd softmax(const Eigen::MatrixXd m) {
    // To be implemented
    return m;
}