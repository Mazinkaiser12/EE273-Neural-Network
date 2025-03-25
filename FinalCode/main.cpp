#include <Eigen>
#include "ActivationFunction.h"
#include "Utils.h"
#include "Convolution.h"
#include "FullyConnected.h"
#include "NeuralNetwork.h"
#include "Deconvolution.h"

using namespace Eigen;

int main()
{
	////Testing Eigen library
	// 
	//print("Original matrice");
	//print(m);
	//print("");
	//print("After applying sigmoid fucntion");
	//MatrixXd a = sigmoid(m);
	//print(a);
	//print("");
	//print("After applying tanh fucntion");
	//MatrixXd b = Tanh(m);
	//print(b);
	//print("After applying tanh fucntion");
	//MatrixXd c = ReLU(m);
	//print(c);
	//print("After applying tanh fucntion");
	//MatrixXd d = softmax(m);
	//print(d);
	//// Testing Scalar multiplication
	//MatrixXd c(3, 3);
	//c << 1, 2, 3,
	//	4, 5, 6,
	//	7, 8, 9;
	//print("Matrix multiplication:");
	//MatrixXd b = m * c;
	//print(b);
	//b = (m.array() * c.array());
	//print("");
	//print("Matrix Scalar multiplication");
	//print(b);
	//print("Sum of matrix b");	
	//print(b.sum());

	//Convolution test

	/*MatrixXd m(3, 3);
	m << 0, 1, 2,
		2, 2, 0,
		0, 1, 2;
	ConvolutionLayer K;
	ConvolutionLayer test(3,0,1);
	test.setFilter(m);	
	test.getConfig();
	MatrixXd input(5, 5);
	input << 
		3, 3, 2, 1, 0,
		0, 0, 1, 3, 1,
		3, 1, 2, 2, 3,
		2, 0, 0, 2, 2,
		2, 0, 0, 0, 1;
	print("Input matrix:");
	print("");
	print(input);
	print("Performing convolution...");
	print("");
	MatrixXd output = test.launch(input);
	print("Output matrix:");
	print(output);*/
	//ConvolutionLayer test(2,1,1,"Random");
	
	// Fully connected layer test
	MatrixXd w(3, 3);
	w << 2, 0, 0,
		0, 2, 0,
		0, 0, 2;
	//MatrixXd k(2, 2);
	//k << 1, 0,
	//	0, 1;
	//FullyConnectedLayer test(w,"None");
	//MatrixXd input(3, 3);
	//input << 
	//	1, 2, 3,
	//	4, 5, 6,
	//	7, 8, 9;
	//print("Input matrix:");
	//print("");
	//print(input);
	//print("Performing fully connected layer...");
	//print("");
	//MatrixXd output = test.launch(input);
	//print("Output matrix:");
	//print(output);
	//test.setWeights(k);
	//print("Performing fully connected layer...");
	//print("");
	////Wrong dimensions test
	//MatrixXd output2 = test.launch(input);
	//print("Output matrix:");
	//print(output2);

	// Neural network test

	/*MatrixXd input(5, 5);
	input <<
		3, 3, 2, 1, 0,
		0, 0, 1, 3, 1,
		3, 1, 2, 2, 3,
		2, 0, 0, 2, 2,
		2, 0, 0, 0, 1;
	MatrixXd m(3, 3);
	m << 0, 1, 2,
		2, 2, 0,
		0, 1, 2;
	ConvolutionLayer* layer1 = nullptr;
	layer1 = new ConvolutionLayer(3, 0, 1);
	FullyConnectedLayer* layer2 = nullptr;
	layer2 = new FullyConnectedLayer(w);
	layer1->setFilter(m);
	NeuralNetwork nn(2);
	nn.addLayer(layer1);
	nn.addLayer(layer2);
	print(nn.launch(input));*/

	//Deconvolution test
	/*MatrixXd m(3, 3);
	m <<
		1, 2, 3,
		0, 1, 0,
		2, 1, 2;
	MatrixXd input(4, 4);
	input <<
		1, 3, 2, 1,
		1, 3, 3, 1,
		2, 1, 1, 3,
		3, 2, 3, 3;
	Deconvolution test(3, 0, 1);
	test.setFilter(m);
	MatrixXd output = test.launch(input);
	print(output);*/


	//Convolution test

		MatrixXd m(3, 3);
		m << 0, 0, 0,
			1, -1, 0,
			1, 1, -1;
		ConvolutionLayer test(3,1,2);
		test.setFilter(m);
		test.getConfig();
		MatrixXd input(5, 5);
		input <<
			0, 1, 2, 0, 1,
			1, 2, 2, 0, 0,
			0, 1, 2, 1, 0,
			0, 2, 1, 1, 0,
			0, 0, 1, 0, 2;
		print("Input matrix:");
		print("");
		print(input);
		print("Performing convolution...");
		print("");
		MatrixXd output = test.launch(input);
		print("Output matrix:");
		print(output);

	return 0;

}


