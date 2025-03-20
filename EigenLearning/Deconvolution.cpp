#include "Deconvolution.h"
#include "Utils.h"
#include <iostream>
#include <Eigen>
#include <vector>
#include <stdexcept>

using namespace std;
using Eigen::MatrixXd;

Deconvolution::Deconvolution()
{
	cout << "Pooling layer created, default configuration" << "\n" << endl;
	this->getConfig();
}

Deconvolution::Deconvolution(int filterSize, int padding, int stride)
{
	this->filterSize = filterSize;
	this->padding = padding;
	this->stride = stride;

	if (filterSize < 1 or padding < 0 or stride < 0)
	{
		error("Input constructor elements must be above 0");
	}

	cout << "Deconvolution layer created with specified type, filter size, padding and stride" << "\n" << endl;
	this->getConfig();
}

Deconvolution::Deconvolution(int filterSize, int padding, int stride, string ActivationFunction)
{
	this->filterSize = filterSize;
	this->padding = padding;
	this->stride = stride;
	this->ActivationFunction = ActivationFunction;

	Filter = MatrixXd::Zero(filterSize, filterSize);

	if (filterSize < 1 or padding < 0 or stride < 0)
	{
		error("Input constructor elements must be above 0");
	}

	cout << "Deconvolution layer created, Act Func configuration" << "\n" << endl;
	this->getConfig();
}

void Deconvolution::getConfig()
{
	cout << "Filter size" << "\t\t" << " : " << this->filterSize << endl;
	cout << "Stride" << "\t\t\t" << " : " << this->stride << endl;
	cout << "Pading" << "\t\t\t" << " : " << this->padding << endl;
	cout << "Activation Function" << "\t" << " : " << this->ActivationFunction << "\n" << endl;
}

void Deconvolution::setConfig(int filterSize, int padding, int stride)
{
	this->filterSize = filterSize;
	this->padding = padding;
	this->stride = stride;
}

void Deconvolution::setConfig(int filterSize, int padding, int stride, string ActivationFunction)
{
	this->filterSize = filterSize;
	this->padding = padding;
	this->stride = stride;
	this->ActivationFunction = ActivationFunction;
}

void Deconvolution::setFilter(MatrixXd Filter)
{
	if (this->filterSize != Filter.rows() or this->filterSize != Filter.cols())
	{
		error("Filter doesn't match required size");
	}
	this->Filter = Filter;
}

MatrixXd Deconvolution::addPadding(MatrixXd outMatrix, int padding)
{
	if (padding == 0)
	{
		return outMatrix;
	}
	int rows = outMatrix.rows();
	int cols = outMatrix.cols();
	MatrixXd paddedMatrix = MatrixXd::Zero(rows + 2 * padding, cols * 2 + padding);
	paddedMatrix.block(padding, padding, rows, cols) = outMatrix;
	return paddedMatrix;
}

MatrixXd Deconvolution::launch(MatrixXd inpMatrix)
{
	//Calculating output matrix size
	this->outSize = ((inpMatrix.rows() - 1) * stride) - (2 * padding) + filterSize + 1;
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
	if (filterSize < inpMatrix.rows() or filterSize < inpMatrix.cols())
	{
		error("Filter is larger than input matrix");
	}

	//Gets the size of the output matrix
	if (padding > 3)
	{
		error("Padding too large, try deconvelution layer");
	}

	//Check of the output matrix is large enough
	if (outSize < filterSize or outSize < inpMatrix.rows())
	{
		error("Output matrix must be larger for deconvolution");
	}

	//Pooling layer creation
	MatrixXd outRawMatrix = MatrixXd::Zero(outSize, outSize);
	MatrixXd outMatrix = MatrixXd::Zero(outSize, outSize);
	//Add padding to output matrix
	outRawMatrix = addPadding(outRawMatrix, padding);
	double result{ 0 };
	for (int i = 0; i < inpMatrix.rows(); i++)
	{
		for (int j = 0; j < inpMatrix.cols(); j++)
		{
			MatrixXd window = Filter * double(inpMatrix(i, j));
			outRawMatrix.block(i * stride, j * stride, filterSize, filterSize) = outRawMatrix.block(i * stride, j * stride, filterSize, filterSize) + window;
			outMatrix = outRawMatrix.block(padding, padding, outSize, outSize);
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