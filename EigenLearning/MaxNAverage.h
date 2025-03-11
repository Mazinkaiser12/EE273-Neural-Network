#pragma once

namespace internal
{
	inline int findMax(const double* x, const int n)
	{
		int loc = 0;
		for (int i = 1; i < n; i++)
		{
			loc = (x[i] > x[loc]) ? i : loc;
		}
		return loc;
	}

	inline double findBlockMax(const double* x, const int nRow, const int nCol, const int colStride, int& loc)
	{
		//First column - finding max coefficient
		loc = findMax(x, nRow);
		double val = x[loc];

		x += colStride;
		int locNext = findMax(x, nRow);
		double valNext = x[locNext];
		if (valNext > val)
		{
			loc = colStride + locNext; 
			val = valNext;
		}
		if (nCol == 2)
		{
			return val;
		}

		//other columns selection
		for (int i = 2; i < nCol; i++)
		{
			x += colStride;
			locNext = findMax(x, nRow);
			valNext = x[locNext];
			if (valNext > val)
			{
				loc = i * colStride + locNext;
				val = valNext;
			}
		}
		return val;
	}

	inline double sumRow(const double* x, const int n)
	{
		double c = 0;
		for (int i = 0; i < n; i++)
		{
			c += x[i];
		}
		return c;
	}

	inline double averageBlock(const double* x, const int nRow, const int nCol, const int colStride, int& loc)
	{
		double sum = 0;
		for (int i = 0; i < nCol; i++)
		{
			x += colStride;
			sum += sumRow(x, nRow);
		}
		return sum / (nCol * nRow);
	}
}