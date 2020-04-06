#include "lab2.h"
#include "lab3.h"

using namespace cv;
using namespace std;


int image_translation(cv::Mat image, int dx, int dy){

	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srccol;
	int dstcol = srcrow;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC3, cv::Scalar::all(0));

	for (int u = 0; u < srcrow; u++){
		for (int v = 0; v < srccol; v++){
			// Calculate the new place
			int x = u + dx;
			int y = v + dy;

			if (x < 0 || x >= dstrow || y < 0 || y >= dstcol){
				continue;
			}

			dst_img.at<cv::Vec3b>(x, y)[0] = image.at<cv::Vec3b>(u, v)[0];
			dst_img.at<cv::Vec3b>(x, y)[1] = image.at<cv::Vec3b>(u, v)[1];
			dst_img.at<cv::Vec3b>(x, y)[2] = image.at<cv::Vec3b>(u, v)[2];

		}
	}
	showImg(dst_img, "Translation");
	return EXIT_SUCCESS;
}

int image_rotation(cv::Mat image, int angle){
	// Convert angle to radius
	double PI = 3.141592653589793;
	double radian = angle * PI / 180;
	double sina = sin(radian);
	double cosa = cos(radian);

	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srccol;
	int dstcol = srcrow;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC3, cv::Scalar::all(0));

	for (int u = 0; u < srcrow; u++){
		for (int v = 0; v < srccol; v++){
			// Calculate the new place
			int x = cosa * u + sina * v;
			int y = -1 * sina *u + cosa*v;

			if (x < 0 || x >= dstrow || y < 0 || y >= dstcol){
				continue;
			}

			dst_img.at<cv::Vec3b>(x, y)[0] = image.at<cv::Vec3b>(u, v)[0];
			dst_img.at<cv::Vec3b>(x, y)[1] = image.at<cv::Vec3b>(u, v)[1];
			dst_img.at<cv::Vec3b>(x, y)[2] = image.at<cv::Vec3b>(u, v)[2];

		}
	}
	showImg(dst_img, "Translation");
	return EXIT_SUCCESS;
}

int image_shear(cv::Mat image, int type, double d){

	double PI = 3.141592653589793;
	double radian = d * PI / 180;
	double tand = tan(radian);

	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srccol;
	int dstcol = srcrow;
	cv::Mat dst_img(2 * dstrow, 2 * dstcol, CV_8UC3, cv::Scalar::all(0));

	for (int u = 0; u < srcrow; u++){
		for (int v = 0; v < srccol; v++){
			// Calculate the new place
			// 0: horizontal; 1: vertical
			int x, y;
			if (type == 0){
				x = u + tand*v;
				y = v;
			}
			else if (type == 1){
				x = u;
				y = tand*v + u;
			}
			if (x < 0 || x >= 2 * dstrow || y < 0 || y >= 2 * dstcol){
				continue;
			}

			dst_img.at<cv::Vec3b>(x, y)[0] = image.at<cv::Vec3b>(u, v)[0];
			dst_img.at<cv::Vec3b>(x, y)[1] = image.at<cv::Vec3b>(u, v)[1];
			dst_img.at<cv::Vec3b>(x, y)[2] = image.at<cv::Vec3b>(u, v)[2];

		}
	}
	showImg(dst_img, "Shear");
	return EXIT_SUCCESS;

}

int image_smoothing(cv::Mat image, int filter_length, int smooth_type){

	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srccol;
	int dstcol = srcrow;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC3, cv::Scalar::all(0));

	for (int u = 0; u < dstrow; u++){
		for (int v = 0; v < dstcol; v++){
			// Loacte the starting coordinate in orginal picture
			int x = u;
			int y = v;

			Vec3f result = { 0, 0, 0 };
			Vec3f sum = { 0, 0, 0 };

			for (int i = x - (filter_length / 2); i <= x + (filter_length / 2); i++){
				for (int j = y - (filter_length / 2); j <= (y + filter_length / 2); j++){
					// Boundary condition
					if (i >= 0 && i < srcrow && j >= 0 && j < srccol){
						sum = sum + Vec3f(image.at<Vec3b>(i, j));
					}
				}
			}


			// type0 - Average; type 1 - median; type 2: binary
			switch (smooth_type)
			{
			case 0:
				result = sum / (filter_length*filter_length);
				break;
			case 1:

				break;
			case 3:
				break;
			default:
				break;
			}


			dst_img.at<Vec3b>(u, v) = (Vec3b)result;
			waitKey(0);
		}
	}

	showImg(dst_img, "smooth");
	return EXIT_SUCCESS;
}

int image_smoothing_median(cv::Mat image, int filter_length){

	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srccol;
	int dstcol = srcrow;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC3, cv::Scalar::all(0));

	for (int u = 0; u < dstrow; u++){
		for (int v = 0; v < dstcol; v++){
			// Loacte the starting coordinate in orginal picture
			int x = u;
			int y = v;

			Vec3f result = { 0, 0, 0 };

			int window[25][3]; int w = 0;
			for (int i = x - (filter_length / 2); i <= x + (filter_length / 2); i++){
				for (int j = y - (filter_length / 2); j <= (y + filter_length / 2); j++){
					// Boundary condition
					if (i >= 0 && i < srcrow && j >= 0 && j < srccol){
						window[w][0] = image.at<Vec3b>(i, j)[0];
						window[w][1] = image.at<Vec3b>(i, j)[1];
						window[w][2] = image.at<Vec3b>(i, j)[2];
						w++;
					}
				}
			}
			for (int c = 0; c < 3; c++){
				for (int i = 1; i<filter_length*filter_length; i++)
				{
					for (int j = 0; j<filter_length*filter_length - 1; j++)
					{
						if (window[j][c]>window[j + 1][c])
						{
							swap(window[j][c], window[j + 1][c]);
						}
					}
				}
			}

			int idmid = 0.5*(filter_length*filter_length + 1);
			result = { float(window[idmid][0]), float(window[idmid][1]), float(window[idmid][2]) };

			dst_img.at<Vec3b>(u, v) = (Vec3b)result;
		}
	}

	showImg(dst_img, "smooth median");
	waitKey(0);
	return EXIT_SUCCESS;

}

int image_smoothing_binary(cv::Mat image, double filter_length, double threshold){

	int rows = image.rows;
	int cols = image.cols;
	cv::Mat result_img(rows, cols, CV_8UC1, cv::Scalar::all(0));

	for (int u = 0; u < rows; u++){
		for (int v = 0; v < cols; v++){
			int x = u;
			int y = v;
			double sum = 0;
			double avg = 0;
			for (int i = x - (filter_length / 2); i <= x + (filter_length / 2); i++){
				for (int j = y - (filter_length / 2); j <= (y + filter_length / 2); j++){
					// Boundary condition
					if (i >= 0 && i < rows && j >= 0 && j < cols){
						sum = sum + image.at<uchar>(i, j);
					}
				}
			}
			avg = sum / (filter_length*filter_length);

			if (avg > threshold * 255)
				result_img.at<uchar>(u, v) = 255;
			else
				result_img.at<uchar>(u, v) = 0;
		}
	}

	showImg(result_img, "Binary Smoothing");

	return EXIT_SUCCESS;
}