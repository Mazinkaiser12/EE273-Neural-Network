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

class Deconvolution : public BaseLayer {
public:

	//Constructor with no input information
	Deconvolution();

	//Constructor with size specified but no activation function
	Deconvolution(int filterSize, int padding, int stride, int outSize);

	//Fully declared pooling contructor
	Deconvolution(int filterSize, int padding, int stride, int outSize, string ActivationFunction);

	//Returning pooling layer's configuration
	void returnConFig();
	void setDeconvolution(int filterSize, int padding, int stride, int outSize);
	void setDeconvolution(int filterSize, int padding, int stride, int outSize, string ActivationFunction);
	void setFilter(MatrixXd Filter);
	MatrixXd padded(MatrixXd outMatrix, int padding);
	MatrixXd launch(MatrixXd inpMatrix) override;
private:
	int filterSize;
	int padding;
	int stride;
	int outSize;
	MatrixXd Filter;
	string ActivationFunction{ "None" };
};