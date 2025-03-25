#include "Convolution.h"

ConvolutionLayer::ConvolutionLayer() {
    Filter = MatrixXd::Zero(filterSize, filterSize);
    print("Convolution layer created");
    this->getConfig();
}

ConvolutionLayer::ConvolutionLayer(int filterSize, int padding, int stride) {
    // I like it that way to assing mebers instead of newer construction
    this->filterSize = filterSize;
    this->padding = padding;
    this->stride = stride;

    if (filterSize <= 0 || padding < 0 || stride < 0) {
        error("Invalid value passed to constructor. Values must be non-negative");
    }

    Filter = MatrixXd::Zero(filterSize, filterSize);
    print("Convolution layer created");
    this->getConfig();
}

ConvolutionLayer::ConvolutionLayer(int filterSize, int padding, int stride, std::string ActivationFunction) {
    // I like it that way to assing mebers instead of newer construction
    this->filterSize = filterSize;
    this->padding = padding;
    this->stride = stride;

    if (filterSize < 0 || padding < 0 || stride < 0) {
        error("Invalid value passed to constructor. Values must be non-negative.");
    }

    Filter = MatrixXd::Zero(filterSize, filterSize);
    this->activationFunction = ActivationFunction;
    print("Convolution layer created");
    this->getConfig();
}

void ConvolutionLayer::getConfig() {
    print("|-----------------Convolution Layer-----------------|");
    print("Filter:");
    print(Filter);
    print("Filter size:");
    print(this->filterSize);
    print("Stride:");
    print(this->stride);
    print("Padding:");
    print(this->padding);
    print("Activation function:");
    print(this->activationFunction);
    print("|---------------------------------------------------|");
}

void ConvolutionLayer::setConfig(int filterSize, int padding, int stride)
{
    this->filterSize = filterSize;
    this->padding = padding;
    this->stride = stride;
}

void ConvolutionLayer::setConfig(int filterSize, int padding, int stride, std::string ActivationFunction)
{
    this->filterSize = filterSize;
    this->padding = padding;
    this->stride = stride;
    this->activationFunction = ActivationFunction;
}

void ConvolutionLayer::setFilter(MatrixXd Filter) {
    if (this->filterSize != Filter.rows() || this->filterSize != Filter.cols()) {
        error("Filter size does not match the layer configuration.");
    }
    this->Filter = Filter;
    print("New filter for convolution:");
    print(Filter);
}

MatrixXd ConvolutionLayer::addPadding(MatrixXd inputMatrix, int padding) {
    if (padding == 0) {
        return inputMatrix;
    }
    int rows = inputMatrix.rows();
    int cols = inputMatrix.cols();
    MatrixXd paddedMatrix = MatrixXd::Zero(rows + 2 * padding, cols + 2 * padding);
    paddedMatrix.block(padding, padding, rows, cols) = inputMatrix;
    return paddedMatrix;
}

MatrixXd ConvolutionLayer::launch(MatrixXd inputMatrix)
{
    print("Convoluting");
    // Check if the input matrix is empty
    if (inputMatrix.rows() == 0 || inputMatrix.cols() == 0) {
        error("Empty input matrix.");
    }

    // Check if the filter matrix is empty
    if (filterSize == 0) {
        error("Empty filter matrix. Assing it before launching");
    }

    // Check if the filter size is greater than the input matrix
    if (filterSize > inputMatrix.rows() || filterSize > inputMatrix.cols()) {
        error("Filter size is greater than the input matrix.");
    }
    // Default output size calculation
    if (padding > 1) {
        error("Given padding increase the size of input matrix, try using DeConvolution");
    }
    // General formula for output size convolution
    int outputSizeRows = (inputMatrix.rows() - filterSize + 2 * padding) / stride + 1;
    int outputSizeCols = (inputMatrix.cols() - filterSize + 2 * padding) / stride + 1;
    // Condition for the same padding which means that the output size is the same as the input size
    if (padding == int(filterSize / 2) && filterSize % 2 == 1 && stride == 1) {
        outputSizeRows = inputMatrix.rows();
        outputSizeCols = inputMatrix.cols();
    }
    // Add padding to the input matrix
    inputMatrix = addPadding(inputMatrix, padding);

    MatrixXd outputMatrix = MatrixXd::Zero(outputSizeRows, outputSizeCols);
    for (int i = 0; i < outputSizeRows; i++) {
        for (int j = 0; j < outputSizeCols; j++) {
            // Get the current window of values for filter application
            // block function is used to get a block of values from the matrix
            MatrixXd window = inputMatrix.block(i * stride, j * stride, filterSize, filterSize);
            /*print("Window:");
            print(window);*/
            // Function cwiseProduct makes sort of scalar multiplication of two matrices elementwise 
            // If you would have two matrices and applied it you would get a matrix with the same size and all values would be squared
            MatrixXd result = window.cwiseProduct(Filter);
            /*print("Result:");
            print(result);*/
            // Add the result to the output matrix
            outputMatrix(i, j) = result.sum();
        }
    }
    // Switch case for activation function
    if (activationFunction == "None"){
        return outputMatrix;
    }
    else if (activationFunction == "Sigmoid"){
         return sigmoid(outputMatrix);
    }
    else if (activationFunction == "ReLU"){
        return ReLU(outputMatrix);
    }
    else if (activationFunction == "Tanh") {
        return Tanh(outputMatrix);
    }
    else if (activationFunction == "Softmax") {
        return softmax(outputMatrix);
    }
    else {
        error("Activation function not supported.");
    }
    return outputMatrix;
}