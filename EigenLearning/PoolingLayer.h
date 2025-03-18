#pragma once
#include <iostream>
#include "BaseLayer.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include <map>
#include <string>
#include "ActivationFunction.h"
#include "Utils.h"

using namespace std;
using Eigen::MatrixXd;

class Pooling : public BaseLayer {
public:

	//Constructor with no input information
	Pooling();

	//Constructor with size specified but no activation function
	Pooling(bool type, int filterSize, int padding, int stride);

	//Fully declared pooling contructor
	Pooling(bool type, int filterSize, int padding, int stride, string ActivationFunction);

	//Returning pooling layer's configuration
	void returnConFig();
	void setPooling(bool type, int filterSize, int padding, int stride);
	void setPooling(bool type, int filterSize, int padding, int stride, string ActivationFunction);
	MatrixXd padded(MatrixXd inpMatrix, int padding);
	MatrixXd launch(MatrixXd inpMatrix) override;
private:
	bool type;
	int filterSize;
	int padding;
	int stride;
	string ActivationFunction{ "None" };
};