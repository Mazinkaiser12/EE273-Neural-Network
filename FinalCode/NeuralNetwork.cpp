#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(){
	this->Number_Of_Layers = 0;
	print("You've created a network without any layer try adding them using");
}

NeuralNetwork::NeuralNetwork(int Number_Of_Layers) {
	this->Number_Of_Layers = Number_Of_Layers;
	// Allocates memory in advance, but bector can do it automatically after
	layers.reserve(Number_Of_Layers);
}

MatrixXd NeuralNetwork::launch(MatrixXd input) {
	for (auto layer: layers) {
		/*print("Input before change");
		print(input);*/
		output = layer->launch(input);
		/*print("Output");
		print(output);*/
		input = output;
		/*print("Input");
		print(input);*/
	}
	print("Nerual Network result:");
	return output;
}

void NeuralNetwork::addLayer(BaseLayer* layer){
	layers.push_back(layer);
}
