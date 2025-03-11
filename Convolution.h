#pragma once
#include "BaseLayer.h"
#include <map>
#include <string>
#include "Utils.h"
#include "ActivationFunction.h"
#include <iostream>

using namespace Eigen;

class ConvolutionLayer : public BaseLayer {
public:

	// Default constructor
	ConvolutionLayer();

	// Constructor without Activation function type specified
	ConvolutionLayer(int filterSize, int padding, int stride);

	// Constructor with Activation function type specified
	ConvolutionLayer(int filterSize, int padding, int stride, std::string ActivationFunction);

	// Displays current configuration of the layer
	void getConfig();
	void setConfig(int filterSize, int padding, int stride);
	void setConfig(int filterSize, int padding, int stride, std::string ActivationFunction);
	void setFilter(MatrixXd Filter);
	MatrixXd addPadding(MatrixXd inputMatrix, int padding);
	MatrixXd launch(MatrixXd inputMatrix) override;

private:
	int filterSize;
	int padding;
	int stride;
	MatrixXd Filter;
	std::string activationFunction = "None";
};