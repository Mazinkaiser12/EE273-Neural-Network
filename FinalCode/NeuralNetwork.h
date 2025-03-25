#pragma once
#include <vector>
#include "Convolution.h"
#include "FullyConnected.h"

using namespace Eigen;

class NeuralNetwork {
public:
	// Default constructor
	NeuralNetwork();
	// Constructor with number of layers
	NeuralNetwork(int Number_Of_Layers);
	MatrixXd launch(MatrixXd input);
	void addLayer(BaseLayer* layer);
private:
	MatrixXd output;
	int Number_Of_Layers;
	std::vector<BaseLayer*> layers;
};