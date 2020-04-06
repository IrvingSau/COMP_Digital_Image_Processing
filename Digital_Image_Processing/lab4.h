#include <iostream> 
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>

using namespace cv;
using namespace std;

int showImg(cv::Mat image, string name);

cv::Mat loadImg(string name);

cv::Mat loadGreyImg(string name);

int edge_detection_sobel(cv::Mat image, double threshold);

int image_sharpen_laplacian(cv::Mat image);

int gammar_correction(cv::Mat image, double gammar);

int histogram_enhancement_local(cv::Mat image, double E, double k0, double k1, double k2);

int histogram_enhancement_global(cv::Mat image);