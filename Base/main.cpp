#include <Eigen>
#include "ActivationFunction.h"
#include "Utils.h"

using namespace Eigen;

int main()
{
	MatrixXd m(3, 3);
	m << 1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	print("Original matrix");
	print(m);
	print("");
	print("After applying sigmoid function");
	MatrixXd a = sigmoid(m);
	print(a);
	print("");
	print("After applying tanh function");
	MatrixXd b = Tanh(m);
	print(b);
	print("After applying tanh function");
	MatrixXd c = ReLU(m);
	print(c);
	print("After applying tanh function");
	MatrixXd d = softmax(m);
	print(d);
	return 0;
	//test
}

