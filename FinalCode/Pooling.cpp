#include "Pooling.h"
#include "Utils.h"
#include <iostream>
#include <Eigen>
#include <vector>
#include <stdexcept>

using namespace std;
using Eigen::MatrixXd;

Pooling::Pooling()
{
	cout << "Pooling layer created, default configuration" << "\n" << endl;
	this->getConfig();
}

Pooling::Pooling(string type, int filterSize, int padding, int stride)
{
	this->type = type;
	this->filterSize = filterSize;
	this->padding = padding;
	this->stride = stride;

	if (filterSize < 1 or padding < 0 or stride < 0)
	{
		error("Input constructor elements must be above 0");
	}

	cout << "Pooling layer created with specified type, filter size, padding and stride" << "\n" << endl;
	this->getConfig();
}

Pooling::Pooling(string type, int filterSize, int padding, int stride, string ActivationFunction)
{
	this->type = type;
	this->filterSize = filterSize;
	this->padding = padding;
	this->stride = stride;
	this->ActivationFunction = ActivationFunction;

	if (filterSize < 1 or padding < 0 or stride < 0)
	{
		error("Input constructor elements must be above 0");
	}

	cout << "Pooling layer created, Act Func configuration" << "\n" << endl;
	this->getConfig();
}

void Pooling::getConfig()
{
	if (type == "max")
	{
		cout << "Filter" << "\t\t\t" << " : Max" << endl;
	}
	else if (type == "average")
	{
		cout << "Filter" << "\t\t\t" << " : Average" << endl;
	}
	cout << "Filter size" << "\t\t" << " : " << this->filterSize << endl;
	cout << "Stride" << "\t\t\t" << " : " << this->stride << endl;
	cout << "Pading" << "\t\t\t" << " : " << this->padding << endl;
	cout << "Activation Function" << "\t" << " : " << this->ActivationFunction << "\n" << endl;
}

void Pooling::setConfig(string type, int filterSize, int padding, int stride)
{
	this->type = type;
	this->filterSize = filterSize;
	this->padding = padding;
	this->stride = stride;
}

void Pooling::setConfig(string type, int filterSize, int padding, int stride, string ActivationFunction)
{
	this->type = type;
	this->filterSize = filterSize;
	this->padding = padding;
	this->stride = stride;
	this->ActivationFunction = ActivationFunction;
}

MatrixXd Pooling::addPadding(MatrixXd inpMatrix, int padding)
{
	if (padding == 0)
	{
		return inpMatrix;
	}
	int rows = inpMatrix.rows();
	int cols = inpMatrix.cols();
	MatrixXd paddedMatrix = MatrixXd::Zero(rows + 2 * padding, cols * 2 + padding);
	paddedMatrix.block(padding, padding, rows, cols) = inpMatrix;
	return paddedMatrix;
}

MatrixXd Pooling::launch(MatrixXd inpMatrix)
{
	//Series of checks to ensure launch is successfull
	//Existance of input matrix
	if (inpMatrix.rows() == 0 or inpMatrix.cols() == 0)
	{
		error("Input matrix is empty");
	}

	//filter size above 0
	if (filterSize == 0)
	{
		error("Filter too small");
	}

	//Filter < Matrix
	if (filterSize > inpMatrix.rows() or filterSize > inpMatrix.cols())
	{
		error("Filter is larger than input matrix");
	}

	int outRowSize = ((inpMatrix.rows() - filterSize) / stride) + 1;
	int outColSize = ((inpMatrix.cols() - filterSize) / stride) + 1;

	//Add padding to input matrix
	inpMatrix = addPadding(inpMatrix, padding);

	//Pooling layer creation
	MatrixXd outMatrix = MatrixXd::Zero(outRowSize, outColSize);
	double result{ 0 };
	for (int i = 0; i < outRowSize; i++)
	{
		for (int j = 0; j < outColSize; j++)
		{
			MatrixXd window = inpMatrix.block(i * stride, j * stride, filterSize, filterSize);
			if (type == "max")
			{
				result = window.maxCoeff();
			}
			else if (type == "average")
			{
				result = window.mean();
			}
			else
			{
				error("Please enter a valid pooling layer type.");
			}
			outMatrix(i, j) = result;
		}
	}

	//Acitvation function application
	if (ActivationFunction == "None")
	{
		return outMatrix;
	}
	else if (ActivationFunction == "Sigmoid")
	{
		return sigmoid(outMatrix);
	}
	else if (ActivationFunction == "ReLU")
	{
		return ReLU(outMatrix);
	}
	else if (ActivationFunction == "Tanh")
	{
		return Tanh(outMatrix);
	}
	else if (ActivationFunction == "SoftMax")
	{
		return softmax(outMatrix);
	}
	else
	{
		error("Please enter a valid activation function");
	}
}