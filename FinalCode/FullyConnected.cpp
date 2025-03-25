#include "FullyConnected.h"

FullyConnectedLayer::FullyConnectedLayer() {
	print("Fully connected layer created");
	this->getConfig();
}

FullyConnectedLayer::FullyConnectedLayer(MatrixXd weights){
	this->weights = weights;
	print("Fully connected layer created");
	this->getConfig();
}

FullyConnectedLayer::FullyConnectedLayer(MatrixXd weights, std::string ActivationFunction){
	this->weights = weights;
	this->activationFunction = ActivationFunction;
	print("Fully connected layer created");
	this->getConfig();
}

MatrixXd FullyConnectedLayer::launch(MatrixXd inputMatrix){
	// Check if the dimensions of the input matrix and the weights matrix are compatible
	print("Fully connected is running");
	if (inputMatrix.cols() != weights.rows()) {
		//print("Input size:");
		//print(inputMatrix.cols());
		//print(weights.rows());
		error("Input matrix and weights matrix dimensions are not compatible.");
	}
	//This will simply perform matrix multiplication of the input matrix and the weights matrix
	MatrixXd outputMatrix = inputMatrix * weights;

	// Switch case for activation function
	if (activationFunction == "None") {
		return outputMatrix;
	}
	else if (activationFunction == "Sigmoid") {
		sigmoid(outputMatrix);
	}
	else if (activationFunction == "ReLU") {
		ReLU(outputMatrix);
	}
	else if (activationFunction == "Tanh") {
		Tanh(outputMatrix);
	}
	else if (activationFunction == "Softmax") {
		softmax(outputMatrix);
	}
	else {
		error("Activation function not supported.");
	}
	return outputMatrix;
}

void FullyConnectedLayer::setWeights(MatrixXd weights){
	this->weights = weights;
}

void FullyConnectedLayer::setActivationFunction(std::string ActivationFunction){
	this->activationFunction = ActivationFunction;
}

void FullyConnectedLayer::getConfig(){
	print("|---------------Fully Connected Layer---------------|");
	print("Weights:");
	print(this->weights);
	print("Activation function:");
	print(this->activationFunction);
	print("|---------------------------------------------------|");

}
