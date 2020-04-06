#include "complexNumber.h"

CComplexNumber::CComplexNumber(void)
{
	real = 0;
	image = 0;
}

CComplexNumber::CComplexNumber(double rl, double im)
{
	real = rl;
	image = im;
}

CComplexNumber::~CComplexNumber(void)
{
}

void CComplexNumber::SetValue(double rl, double im) {
	real = rl;
	image = im;
}