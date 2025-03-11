#pragma once
#include <stdexcept>
#include <Eigen/Dense>

using namespace std;

class BaseLayer {
	const int mInSize;
	const int mOutSize;
public:
	// Launch function which should be implemented within each layer individually to it's purpose
	virtual Eigen::MatrixXd launch(Eigen::MatrixXd& input) {
		throw runtime_error("Not implemented");
	};
	BaseLayer(const int inSize, const int outSize) :
		mInSize(inSize),
		mOutSize(outSize) {}
	// Destructor for all inherited classes
	virtual ~BaseLayer() {
		throw runtime_error("Not implemented");
	};
};