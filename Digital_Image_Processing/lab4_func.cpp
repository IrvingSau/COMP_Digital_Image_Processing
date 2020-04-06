#include "lab2.h"
#include "lab4.h"

using namespace cv;
using namespace std;

int edge_detection_sobel(cv::Mat image, double threshold){
	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srccol;
	int dstcol = srcrow;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC1, cv::Scalar::all(0));
	int filter_length = 3;

	int sobelX[3][3] = { { -1.0, 0.0, 1.0 }, { -2.0, 0.0, 2.0 }, { -1.0, 0.0, 1.0 } };
	int sobelY[3][3] = { { 1.0, 2.0, 1.0 }, { 0.0, 0.0, 0.0 }, { -1.0, -2.0, -1.0 } };

	printf("Sobel algorithm for edge detection\n");

	// Travesal each pixel in dst image
	for (int u = 0; u < dstrow; u++){
		for (int v = 0; v < dstcol; v++){
			int x = u;
			int y = v;
			double Gx = 0;
			double Gy = 0;
			double G_Abs = 0;
			// Find corresponding filter in src image
			int k = 0; int m = 0;
			for (int i = x - 1; i <= x + 1; i++){
				for (int j = y - 1; j <= y + 1; j++){
					// Boundary condition
					if (i >= 0 && i < srcrow && j >= 0 && j < srccol){
						// Calculate Gx
						Gx = Gx + sobelX[i - x + 1][j - y + 1] * image.at<uchar>(i, j);
						// Calculate Gy
						Gy = Gy + sobelY[i - x + 1][j - y + 1] * image.at<uchar>(i, j);
						//printf("%d\n", Gy);
					}
				}
			}

			// Find the absolute value of G
			G_Abs = fabs(Gx) + fabs(Gy);
			//printf("Gx=%d, Gy=%d,\t G_Abs=%d\n", Gx, Gy, G_Abs);
			if (G_Abs > threshold * 255)
				dst_img.at<uchar>(u, v) = 255;
			else
				dst_img.at<uchar>(u, v) = 0;
		}
	}

	showImg(dst_img, "sobel");
	waitKey(0);

	return EXIT_SUCCESS;
}

int image_sharpen_laplacian(cv::Mat image){

	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srccol;
	int dstcol = srcrow;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC1, cv::Scalar::all(0));
	int filter_length = 3;

	int laplacian[3][3] = { { 1.0, 1.0, 1.0 }, { 1.0, -8.0, 1.0 }, { 1.0, 1.0, 1.0 } };

	printf("Sobel algorithm for edge detection\n");

	// Travesal each pixel in dst image
	for (int u = 0; u < dstrow; u++){
		for (int v = 0; v < dstcol; v++){
			int x = u;
			int y = v;
			double Gx = 0;
			double Gy = 0;
			double G_Abs = 0;
			// Find corresponding filter in src image
			for (int i = x - 1; i <= x + 1; i++){
				for (int j = y - 1; j <= y + 1; j++){
					// Boundary condition
					if (i >= 0 && i < srcrow && j >= 0 && j < srccol){
						dst_img.at<uchar>(u, v) = image.at<uchar>(i, j) + laplacian[i - x + 1][j - y + 1] * image.at<uchar>(i, j);
					}
				}
			}
		}
	}

	showImg(dst_img, "laplacian");
	waitKey(0);

	return EXIT_SUCCESS;
}

int gammar_correction(cv::Mat image, double gammar){
	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srccol;
	int dstcol = srcrow;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC1, cv::Scalar::all(0));
	int filter_length = 3;

	printf("Sobel algorithm for edge detection\n");

	// Travesal each pixel in dst image
	for (int u = 0; u < dstrow; u++){
		for (int v = 0; v < dstcol; v++){
			if (u >= 0 && u < srcrow && v >= 0 && v < srccol){
				dst_img.at<uchar>(u, v) = exp(gammar) * image.at<uchar>(u, v);
			}

		}
	}

	showImg(dst_img, "gammar");
	waitKey(0);


	return EXIT_SUCCESS;
}

int histogram_enhancement_global(cv::Mat image){

	// Histogram Equalization
	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srcrow;
	int dstcol = srccol;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC1, cv::Scalar::all(0));

	double L = pow(2, 8);

	// Calculate the probability for all candidate value
	int maxIntensity = image.at<uchar>(0, 0);
	for (int i = 0; i < srcrow; i++){
		for (int j = 0; j < srccol; j++){
			if (image.at<uchar>(i, j) > maxIntensity){
				maxIntensity = image.at<uchar>(i, j);
			}
		}
	}

	double sum_count[256];
	// Calculate the prob
	for (int pflag = 0; pflag <= maxIntensity; pflag++){
		int sum_tmp = 0;
		for (int i = 0; i < srcrow; i++){
			for (int j = 0; j < srccol; j++){
				if (image.at<uchar>(i, j) == pflag){
					sum_tmp += 1;
				}
			}
		}
		sum_count[pflag] = sum_tmp;
	}

	// Evaluate the cumulative probability
	double cum_prob[256];
	for (int i = 0; i <= maxIntensity; i++){
		double temp_cum_sum = 0.0;
		for (int j = 0; j <= i; j++){
			temp_cum_sum += sum_count[j];
		}
		cum_prob[i] = temp_cum_sum / (srccol * srcrow);
		printf("i=%d: ", i);
		printf("%f\n", cum_prob[i]);
	}

	for (int i = 0; i < dstrow; i++){
		for (int j = 0; j < dstcol; j++){
			dst_img.at<uchar>(i, j) = int(ceil(cum_prob[int(image.at<uchar>(i, j))] * 255));
		}
	}


	showImg(dst_img, "histo_global");
	waitKey(0);

	return EXIT_SUCCESS;
}

int histogram_enhancement_local(cv::Mat image, double E, double k0, double k1, double k2){

	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srcrow;
	int dstcol = srccol;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC1, cv::Scalar::all(0));

	// Calculate the statistics for whole Graph
	double m_G = 0.0;
	double sigma_G = 0.0;

	double sum = 0.0;

	for (int u = 0; u < dstrow; u++){
		for (int v = 0; v < dstcol; v++){
			sum = sum + image.at<uchar>(u, v);
		}
	}

	m_G = sum / (srcrow * srccol);

	for (int u = 0; u < dstrow; u++){
		for (int v = 0; v < dstcol; v++){
			sum += (image.at<uchar>(u, v) - m_G) * (image.at<uchar>(u, v) - m_G);
		}
	}

	sigma_G = sqrt(sum / (srcrow * srccol));

	//printf("%f, %f", m_G, sigma_G);

	// Evaluate the local statistics; set window size = 3
	for (int u = 0; u < dstrow; u++){
		for (int v = 0; v < dstcol; v++){
			double sum_tmp = 0.0;
			double m_sxy = 0.0;
			for (int i = u - 1; i < u + 1; i++){
				for (int j = v - 1; j < v + 1; j++){
					if (i >= 0 && i < srcrow && j >= 0 && j < srccol){
						sum_tmp = sum_tmp + image.at<uchar>(i, j);
					}
				}
			}
			m_sxy = sum_tmp / 9;

			double sigma_sxy = 0.0;
			sum_tmp = 0.0;
			for (int i = u - 1; i < u + 1; i++){
				for (int j = v - 1; j < v + 1; j++){
					if (i >= 0 && i < srcrow && j >= 0 && j < srccol){
						sum_tmp += (image.at<uchar>(i, j) - m_sxy) * (image.at<uchar>(i, j) - m_sxy);
					}
				}
			}
			sigma_sxy = sqrt(sum_tmp / 9);

			//printf("%f, %f\n", m_sxy, sigma_sxy);

			if (m_sxy <= m_G*k0 && sigma_sxy <= k2*sigma_G && sigma_sxy >= k1*sigma_G){

				dst_img.at<uchar>(u, v) = image.at<uchar>(u, v) * E;
			}

		}
	}

	showImg(dst_img, "histo_global");
	waitKey(0);

	return EXIT_SUCCESS;
}