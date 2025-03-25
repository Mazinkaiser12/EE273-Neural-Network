#pragma once
#include <iostream>
#include "BaseLayer.h"
#include <Eigen>
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
	Pooling(string type, int filterSize, int padding, int stride);

	//Fully declared pooling contructor
	Pooling(string type, int filterSize, int padding, int stride, string ActivationFunction);

	//Returning pooling layer's configuration
	void getConfig();
	void setConfig(string type, int filterSize, int padding, int stride);
	void setConfig(string type, int filterSize, int padding, int stride, string ActivationFunction);
	MatrixXd addPadding(MatrixXd inpMatrix, int padding);
	MatrixXd launch(MatrixXd inpMatrix) override;
private:
	string type;
	int filterSize;
	int padding;
	int stride;
	string ActivationFunction{ "None" };
};