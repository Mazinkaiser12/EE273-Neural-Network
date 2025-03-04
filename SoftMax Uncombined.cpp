#include <iostream>
#include <math.h>
using namespace std;

double SoftMax(double output)
{
    //Code has already been made to simply recieve an output of a node and SoftMax it
    //Finds the size of the array of output nodes
    int outputColumns{ 0 };
    for( auto n : output)
    {
        outputColumns += 1;
    }

    //Sets up the exponential Node creation
    double exponentialSum = 0;

    //Finds the sum of the exponentials of the outputs
    for(int i = 0; i < outputColumns; i++)
    {
        exponentialSum += exp(output[i]);
    }
    //Applies this sum to every item in the output array
    for(int i = 0; i < outputColumns; i++)
    {
        output[i] = output[i] / exponentialSum;
    }

    return output;
}