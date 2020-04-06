#include <iostream> 
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>

using namespace cv;
using namespace std;

int showImg(cv::Mat image, string name);

cv::Mat loadImg(string name);

cv::Mat loadGreyImg(string name);

int image_translation(cv::Mat image, int dx, int dy);

int image_rotation(cv::Mat image, int angle);

int image_shear(cv::Mat image, int type, double d);

int image_smoothing(cv::Mat image, int filter_length, int smooth_type);

int image_smoothing_median(cv::Mat image, int filter_length);

int image_smoothing_binary(cv::Mat image, double filter_length, double threshold);