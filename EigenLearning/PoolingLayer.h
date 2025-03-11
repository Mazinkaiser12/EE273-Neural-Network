#pragma once
#include <iostream>
#include "BaseLayer.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "ActivationFunction.h"
#include "Utils.h"
#include "MaxNAverage.h"

using namespace std;
using Eigen::MatrixXd;
template<typename Activation>

class Pooling : public BaseLayer {
private:
	using Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	using Eigen::MatrixXi IntMatrix;

	const int mChannelRows;
	const int mChannelCols;
	const int mInChannel;
	const int mPoolRows;
	const int mPoolCols;
	const int mOutRows;
	const int mOutCols;

	IntMatrix mLoc;
	Matrix mZ;
	Matrix mA;
	Matrix mDin;

public:
	Eigen::MatrixXd launch(Eigen::MatrixXd& input) override {
		throw runtime_error("Not implemented");
	};
	Pooling(const int inWidth, const int inHeight, const int inChannel, const int poolingWidth, const int poolingHeight);
		BaseLayer(inWidth*inHeight*inChannel,(inWidth/poolingWidth)*(inHeight/poolingHeight)*inChannel),
		mChannelRows(inHeight),
		mChannelCols(inWidth),
		mInChannel(inChannel)
		mPoolRows(poolingHeight),
		mPoolCols(poolingWidth),
		mOutRows(mChannelRows/mPoolRows)
		mOutCols(mChannelCols / mPoolCols)
	{}

	void forward(const Matrix& prevLayerData)
	{
		const int NoBS = prevLayerData.cols();
		mLoc.resize(this->mOutSize, NoBS);
		mZ.resize(this->mOutSize, NoBS);

		int* locData = mLoc.data();
		const int channelEnd = prevLayerData.size();
		const int channelStride = mChannelRows * mChannelCols;
		const int colEndGap = mChannelRows * mPoolCols * mOutCols;
		const int colStride = mChannelRows * mPoolCols;
		const rowEndGap = mOutRows * mPoolRows;
		for (int channelStart = 0; channelStart < channelEnd; channelStart += channelStride)
		{
			const int colEnd = channelStart + colEndGap;
			for (int colStart = channelStart : colStart < colEnd; colStart += colStride)
			{
				const int rowEnd = colStart + rowEndGap;
				for (int rowStart = colStart; rowStart < rowEnd; rowStart += mPoolRows, locData++)
				{
					locData = rowStart;
				}
			}
		}

		locData = mLoc.data();
		const int* const locEnd = locData + mLoc.size();
		double* zData = mZ.data();
		const double* src = prevLayerData.data();
		for (; locData < locEnd; locData++, zData++)
		{
			const int offset = *locData;
			*zData = internal::findBlockMax(src + offset, mPoolRows, mPoolCols, mChannelRows, *locData);
			*locData += offset;
		}

		mA.resize(this->mOutSize, NoBS);
		mA = ReLU(mZ);
	}

	const Matrix& output() const 
	{
		return mA;
	}

	
};