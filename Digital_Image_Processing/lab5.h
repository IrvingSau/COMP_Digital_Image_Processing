#include <iostream> 
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>
#include "complexNumber.h"

using namespace cv;
using namespace std;

int dft_own(cv::Mat image, String filename);

int idft_own(CComplexNumber *dft_values, int width, int height);

int dft_visualize(cv::Mat image);

int dft_samples(Mat src);

int myFFT(cv::Mat& inputImg, double d0);