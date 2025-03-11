#include <iostream>
#include <Eigen/Dense>
#include <string>
#include "Utils.h"

using namespace std;


void print(int data) {
	cout << data << endl;
}

void print(string data) {
	cout << data << endl;
}

void print(Eigen::MatrixXd data) {
	cout << data << endl;
}
