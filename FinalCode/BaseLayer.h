#pragma once
#include <stdexcept>
#include <Eigen>

class BaseLayer {
public:
	// Launch function which should be implemented within each layer individually to it's purpose
	virtual Eigen::MatrixXd launch(Eigen::MatrixXd input) {
		throw std::runtime_error("Not implemented");
	};
	// Destructor for all inherited classes
	virtual ~BaseLayer() {};
};