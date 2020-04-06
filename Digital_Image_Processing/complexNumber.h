#pragma once

class CComplexNumber
{
public:
	CComplexNumber(double real, double image);
	CComplexNumber(void);
	~CComplexNumber(void);

public:
	inline CComplexNumber CComplexNumber::operator +(const CComplexNumber &c) {
		return CComplexNumber(real + c.real, image + c.image);
	}
	inline CComplexNumber CComplexNumber::operator -(const CComplexNumber &c) {
		return CComplexNumber(real - c.real, image - c.image);
	}
	inline CComplexNumber CComplexNumber::operator *(const CComplexNumber &c) {
		return CComplexNumber(real*c.real - image*c.image, image*c.real + real*c.image);
	}

	inline CComplexNumber CComplexNumber::operator /(const CComplexNumber &c) {
		if ((0 == c.real) && (0 == c.image)) {
			return CComplexNumber(real, image);
		}
		return CComplexNumber((real*c.real + image*c.image) / (c.real*c.real + c.image*c.image),
			(image*c.real - real*c.image) / (c.real*c.real + c.image*c.image));
	}

	void   SetValue(double rl, double im);

public:
	double     real;
	double     image;
};