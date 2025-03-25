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

class Deconvolution : public BaseLayer {
public:

	//Constructor with no input information
	Deconvolution();

	//Constructor with size specified but no activation function
	Deconvolution(int filterSize, int padding, int stride);

	//Fully declared pooling contructor
	Deconvolution(int filterSize, int padding, int stride, string ActivationFunction);

	//Returning pooling layer's configuration
	void getConfig();
	void setConfig(int filterSize, int padding, int stride);
	void setConfig(int filterSize, int padding, int stride, string ActivationFunction);
	void setFilter(MatrixXd Filter);
	MatrixXd addPadding(MatrixXd outMatrix, int padding);
	MatrixXd launch(MatrixXd inpMatrix) override;
private:
	int filterSize;
	int padding;
	int stride;
	int outSize;
	MatrixXd Filter;
	string ActivationFunction{ "None" };
};