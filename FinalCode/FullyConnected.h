#pragma once
#include "BaseLayer.h"
#include <map>
#include <string>
#include "Utils.h"
#include "ActivationFunction.h"
#include <iostream>

using namespace Eigen;

class FullyConnectedLayer : public BaseLayer {
public:
	// Stndart constructor for fully connected layer
	FullyConnectedLayer();
	// Constructor with weights specified
	FullyConnectedLayer(MatrixXd weights);
	// Constructor with weights and activation function specified
	FullyConnectedLayer(MatrixXd weights, std::string ActivationFunction);

	MatrixXd launch(MatrixXd inputMatrix) override;
	void setWeights(MatrixXd weights);
	void setActivationFunction(std::string ActivationFunction);
	void getConfig();
private:
	MatrixXd weights;
	std::string activationFunction = "None"; 
};

