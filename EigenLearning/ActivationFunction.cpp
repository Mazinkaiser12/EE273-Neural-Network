#include "ActivationFunction.h"
#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <math.h>
#include "Utils.h"


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


Eigen::MatrixXd ReLU(Eigen::MatrixXd m) {
    /**
     * ReLU re-implemented in Eigen for week 8, was completed for week 7 but not using Eigen
     * ReLU function here does not apply weight and biased mathematics in line with the other active functions
     * Function array() returns an Eigen::Array object, which is a wrapper around the data of the matrix.
     * The ReLu() function is applied to each element of the array, and the result is stored in a new array and then automatically converted back to a matrix.
     */
	for (int i = 0; i < m.rows(); i++)
	{
		for (int j = 0; j < m.cols(); j++)
		{
			if (m(i, j) < 0)
			{
				m(i, j) = 0;
			}
		}
	}

    return m;

}

Eigen::MatrixXd softmax(const Eigen::MatrixXd m) {
    /**
     * ReLu re-implemented in Eigen for week 8, was completed for week 7 but not using Eigen
     * ReLu function here does not apply weight and biased mathematics in line with the other active functions
     * Function array() returns an Eigen::Array object, which is a wrapper around the data of the matrix.
     * The ReLu() function is applied to each element of the array, and the result is stored in a new array and then automatically converted back to a matrix.
     */
    return m.array() / exp(m.size());
}