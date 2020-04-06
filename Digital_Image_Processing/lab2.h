#include <iostream> 
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>

using namespace cv;
using namespace std;

/***************** Lab 2 3.13 ********************/

int showImg(cv::Mat image, string name);

cv::Mat loadImg(string name);

cv::Mat loadGreyImg(string name);

int alternative_line_reduction(cv::Mat image);

int fractional_linear_transformation(cv::Mat image, float d);

int pixel_replication_enlargement(cv::Mat image, float dx, float dy);

int nearest_point_enlargement(cv::Mat image, float d);

int bilinear_interpolation_enlargement(cv::Mat image, float d);

int bicubic_interpolation_enlargement(cv::Mat image, double d);

int fractional_linear_enlargement(cv::Mat image, float d);

int negative_image(cv::Mat image);

/*********************************************************/