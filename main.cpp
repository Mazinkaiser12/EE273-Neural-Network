#include <Eigen>
#include "ActivationFunction.h"
#include "Utils.h"
#include "Convolution.h"

using namespace Eigen;

int main()
{
	////Testing Eigen library
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
	MatrixXd m(3, 3);
	m << 0, 1, 2,
		2, 2, 0,
		0, 1, 2;
	ConvolutionLayer test(3,2,1);
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
	print(output);
	//ConvolutionLayer test(2,1,1,"Random");
	//return 0;

}

