#include <iostream>
#include <Eigen>
#include <string>
#include "Utils.h"

void print(int data) {
	std::cout << data << std::endl;
}

void print(std::string data) {
	std::cout << data << std::endl;
}

void print(Eigen::MatrixXd data) {
	std::cout << data << std::endl;
}

void error(std::string data){
	std::cerr << data << std::endl;
	exit(1);
}
