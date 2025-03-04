#include <iostream>
#include <math.h>
using namespace std;

//This code is designed to be used inside a larger structure for a neural network in c++

double isMax(double sum)
{
    if(sum < 0)
    {
        return 0;
    }
    else
    {
        return sum;
    }
}

double Relu(double inputs, double weights, double biases)
{
    //This is the creation of a dummy matrix of two input nodes to three ReLu nodes, replace names to ignore
    //This is to be modified to take in the necessary input, weight and biases information from outside the function
    double inputsT[2] = {{1}, {3}};
    double weightsT[3][2] = {{0.1, 0.2},
                            {0.2, 0.4},
                            {0.3, 0.4}};
    double biasesT[3] = {1,1,1};

    //but of course for real cases, this code has to able to handle any number of sizes for input and output nodes
    //meaning they simply count the sizes of the input matrix
    int weightsRows = sizeof(weightsT) / sizeof(weightsT[0]);
    //The number of input rows will always be 1
    int inputColumns = sizeof(inputsT[0]) / sizeof(inputsT[0][0]);

    //This is the exception, as it is a variable that will be returned by the function and thus made in the function
    double output[weightsRows];

    //Declaration for matric manipulation, normally matrix multiplication in c++ returns straight to new matrix 
    //but since we pass through ReLu we create a sum that frequently returns to 0 during multiplication
    double sum = 0;

    //actual multiplication
    for(int i = 0; i < weightsRows; i++)
    {
        for(int j = 0; j < 1; j++)
        {
            for(int k = 0; k < inputColumns; k++)
            {
                sum += inputsT[k]*weightsT[i][k];
            }
            output[i] = isMax(sum + biasesT[i]);
            sum = 0;
        }
    }

    //output for testing purposes
    for( auto n : output)
    {
        cout << n << " , ";
    }
    return 0;
}