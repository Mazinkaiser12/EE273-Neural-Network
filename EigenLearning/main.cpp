#include <iostream>
#include <Eigen/Dense>
#include "ActivationFunction.h"
#include "Utils.h"
#include "PoolingLayer.h"
#include "Deconvolution.h"
#include <typeinfo>

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
	Eigen::Matrix3d nums;
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
	cout << "Maximum coefficient of a matrix" << "\n" << typeid(nums.maxCoeff()).name() << endl;

	//Trace of a matrix
	cout << "Trace of a matrix" << "\n" << nums.trace() << endl;

	//Catchup from week 7 ReLU
	MatrixXd a = ReLU(m);
	cout << "ReLU of a matrix" << "\n" << a << endl;

	//Catchup from week 7 SoftMax
	MatrixXd b = softmax(m);
	cout << "Soft max of a matrix" << "\n" << b << endl;

	//Testing pooling layer
	Pooling test(false, 3, 0, 1);
	test.returnConFig();
	MatrixXd input(5, 5);
	input <<
		6, 3, 2, 1, 0,
		0, 0, 1, 3, 1,
		3, 1, 2, 2, 3,
		2, 0, 0, 2, 2,
		2, 0, 0, 0, 1;
	cout << "Input matrix : " << endl;
	print(input);
	cout << "Performing pooling..." << endl;
	MatrixXd output = test.launch(input);
	cout << "Output matrix : " << endl;
	print(output);

	//Testing Deconvolution layer
	Deconvolution testy(3, 1, 2, 4);
	test.returnConFig();
	MatrixXd Filter(3, 3);
	Filter <<
		1, 2, 1,
		2, 0, 1,
		0, 2, 1;
	cout << "Filter matrix : " << endl;
	print(Filter);
	testy.setFilter(Filter);
	MatrixXd inputD(2, 2);
	inputD <<
		2, 1,
		3, 2;
	cout << "Input matrix : " << endl;
	print(inputD);
	cout << "Performing deconvolution..." << endl;
	MatrixXd outputD = testy.launch(inputD);
	cout << "Output matrix : " << endl;
	print(outputD);
}