#include <iostream>
#include <Eigen/Dense>
#include "ActivationFunction.h"
#include "Utils.h"

using Eigen::MatrixXd;
using namespace std;

int main()
{
	//Example code from Eigen website
	MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2.5;
	m(0, 1) = -1;
	m(1, 1) = m(1, 0) + m(0, 1);
	cout << "Example code from Eigen website" << "\n" << m << endl;

	//Comma-initialization and print size
	Eigen::Matrix3f nums;
	nums << 1, 2, -3, 4, 5, 6, 7, 8, 9;
	cout << "Comma-initialization and print size" << "\n" << nums << "\n" << "Size: " << nums.size() << endl;

	//Constant matrix
	MatrixXd n;
	n = MatrixXd::Constant(5, 5, 3);
	cout << "Constant matrix" << "\n" << n << endl;

	//Other creation method + setting a zero matrix
	Eigen::Matrix <float, 4, 4> foour;
	foour.setZero();
	cout << "Zero matrix" << "\n" << foour << endl;

	//Transposing a function
	Eigen::MatrixXf resultz(4, 4);
	resultz << 8, 12, 17, 33, 38, 44, 12, 35, 56, 75, 89, 65, 23, 54, 65, 91;
	cout << "Untransposed: " << "\n" << resultz << endl;
	resultz.transposeInPlace();
	cout << "Transposed: " << "\n" << resultz << endl;

	//Summation
	cout << "Summed matrix" << "\n" << nums.sum() << endl;

	//Product
	cout << "Product matrix" << "\n" << nums.prod() << endl;

	//Mean
	cout << "Mean of a matrix" << "\n" << nums.mean() << endl;

	//Minimal Coefficient
	cout << "Minimal coefficient of a matrix" << "\n" << nums.minCoeff() << endl;

	//Maximum Coefficient
	cout << "Maximum coefficient of a matrix" << "\n" << nums.maxCoeff() << endl;

	//Trace of a matrix
	cout << "Trace of a matrix" << "\n" << nums.trace() << endl;

	//Catchup from week 7 ReLU
	MatrixXd a = ReLU(m);
	cout << "ReLU of a matrix" << "\n" << a << endl;

	//Catchup from week 7 SoftMax
	MatrixXd b = softmax(m);
	cout << "Soft max of a matrix" << "\n" << b << endl;

}