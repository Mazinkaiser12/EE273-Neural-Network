#include <iostream>
#include <Eigen>
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
	cout << data << "\n" << endl;
}

void error(string data) {
	std::cerr << data << std::endl;
	exit(1);
}
