#include <iostream> 
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>
#include "complexNumber.h"

using namespace cv;
using namespace std;

int highpass_filter_sharpen(cv::Mat image, double d0);

int homomorphic_enhancement(cv::Mat inputImg);

cv::Mat sinusoidal_noise(cv::Mat inputImg);

int bandreject_filter(cv::Mat noise_img);