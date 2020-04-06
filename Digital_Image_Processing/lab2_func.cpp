#include "lab2.h"

using namespace cv;
using namespace std;

int showImg(cv::Mat image, string name){
	if (image.empty()) {
		std::cout << "Hey! Can't read the image!" << std::endl;
		system("PAUSE");
		return EXIT_FAILURE;
	}
	cv::imshow(name, image);
	cv::waitKey(5000);
	return EXIT_SUCCESS;
}

cv::Mat loadImg(string name){
	cv::Mat image = cv::imread(name);
	return image;
}

cv::Mat loadGreyImg(string name){
	cv::Mat image = cv::imread(name);
	cv::Mat grey_img;
	// Turn the original image into grey
	cvtColor(image, grey_img, CV_BGR2GRAY);

	return grey_img;
}


int alternative_line_reduction(cv::Mat image){

	int img_length = image.rows;

	// Create a result image matrix with size in img_size/2
	int len_size = ceil(img_length / 2);
	cv::Mat result_img(len_size, len_size, CV_8UC3, cv::Scalar::all(0));

	int i1 = 0;
	int j1 = 0;

	// Travesal the pixel
	for (int i = 0; i<img_length; i += 2)
	{
		j1 = 0;
		for (int j = 0; j<img_length; j += 2)
		{
			result_img.at<cv::Vec3b>(i1, j1)[0] = image.at<cv::Vec3b>(i, j)[0];
			result_img.at<cv::Vec3b>(i1, j1)[1] = image.at<cv::Vec3b>(i, j)[1];
			result_img.at<cv::Vec3b>(i1, j1)[2] = image.at<cv::Vec3b>(i, j)[2];
			j1++;
		}
		i1++;
	}

	showImg(result_img, "alternative line reduction");

	return EXIT_SUCCESS;
}

int fractional_linear_transformation(cv::Mat image, float d){
	int img_length = image.rows;
	int len_size = d * img_length;
	d = 1 / d;
	cv::Mat result_img(len_size, len_size, CV_8UC3, cv::Scalar::all(0));
	for (int u = 0; u < len_size; u++){
		for (int v = 0; v < len_size; v++){
			// target coordinate and evaluate the coefficient for interpolation
			float org_u = u*d;
			float org_v = v*d;
			float p = org_u - floor(org_u);
			float q = org_v - floor(org_v);

			// Interpolation operation for 3 channels
			result_img.at<cv::Vec3b>(u, v)[0] = (1 - p)*(1 - q)*image.at<cv::Vec3b>(floor(org_u), floor(org_v))[0]
				+ p*(1 - q)*image.at<cv::Vec3b>(ceil(org_u), floor(org_v))[0]
				+ (1 - p)*q*image.at<cv::Vec3b>(floor(org_u), ceil(org_v))[0]
				+ q*p*image.at<cv::Vec3b>(ceil(org_u), ceil(org_v))[0];

			result_img.at<cv::Vec3b>(u, v)[1] = (1 - p)*(1 - q)*image.at<cv::Vec3b>(floor(org_u), floor(org_v))[1]
				+ p*(1 - q)*image.at<cv::Vec3b>(ceil(org_u), floor(org_v))[1]
				+ (1 - p)*q*image.at<cv::Vec3b>(floor(org_u), ceil(org_v))[1]
				+ q*p*image.at<cv::Vec3b>(ceil(org_u), ceil(org_v))[1];

			result_img.at<cv::Vec3b>(u, v)[2] = (1 - p)*(1 - q)*image.at<cv::Vec3b>(floor(org_u), floor(org_v))[2]
				+ p*(1 - q)*image.at<cv::Vec3b>(ceil(org_u), floor(org_v))[2]
				+ (1 - p)*q*image.at<cv::Vec3b>(floor(org_u), ceil(org_v))[2]
				+ q*p*image.at<cv::Vec3b>(ceil(org_u), ceil(org_v))[2];
		}
	}

	showImg(result_img, "fractional reduction");

	return EXIT_SUCCESS;
}

int pixel_replication_enlargement(cv::Mat image, float dx, float dy){

	int srcrow = image.rows;
	int srccol = image.cols;

	int dstrow = srcrow * dx;
	int dstcol = srccol * dy;

	cv::Mat result_img(dstrow, dstcol, CV_8UC3, cv::Scalar::all(0));
	for (int u = 0; u < dstrow; u += dy){
		for (int v = 0; v < dstcol; v += dx){
			int org_u = u / dx;
			int org_v = v / dy;
			result_img.at<cv::Vec3b>(u, v)[0] = image.at<cv::Vec3b>(org_u, org_v)[0];
			result_img.at<cv::Vec3b>(u, v)[1] = image.at<cv::Vec3b>(org_u, org_v)[1];
			result_img.at<cv::Vec3b>(u, v)[2] = image.at<cv::Vec3b>(org_u, org_v)[2];
		}
	}
	showImg(result_img, "replication");
	return EXIT_SUCCESS;
}

int nearest_point_enlargement(cv::Mat image, float d){
	int img_length = image.rows;
	int len_size = d * img_length;
	cv::Mat result_img(len_size, len_size, CV_8UC3, cv::Scalar::all(0));
	for (int u = 0; u < len_size; u++){
		for (int v = 0; v < len_size; v++){
			float org_u = u / d;
			float org_v = v / d;

			// Find the nearest point
			float dist[4];
			int A_u = floor(org_u); int A_v = floor(org_v); dist[0] = sqrt((org_u - A_u) * (org_u - A_u) + (org_v - A_v) * (org_v - A_v));
			int B_u = ceil(org_u); int B_v = floor(org_v); dist[1] = sqrt((org_u - B_u) * (org_u - B_u) + (org_v - B_v) * (org_v - B_v));
			int C_u = floor(org_u); int C_v = ceil(org_v); dist[2] = sqrt((org_u - C_u) * (org_u - C_u) + (org_v - C_v) * (org_v - C_v));
			int D_u = ceil(org_u); int D_v = ceil(org_v); dist[3] = sqrt((org_u - D_u) * (org_u - D_u) + (org_v - D_v) * (org_v - D_v));

			//printf("A = (%d, %d), B = (%d, %d), C = (%d, %d), D = (%d, %d)\n ", A_u, A_v, B_u, B_v, C_u, C_v, D_u, D_v);
			//printf("distA = %f, distB = %f, distC = %f, distD = %f", dist[0], dist[1], dist[2], dist[3]);
			float min = dist[0];
			int index = 0;
			for (int i = 0; i < 4; i++) {
				if (min > dist[i]) {
					min = dist[i];
					index = i;
				}
			}

			//printf("min_dist = %d\n", index);
			switch (index)
			{
			case 0:
				result_img.at<cv::Vec3b>(u, v)[0] = image.at<cv::Vec3b>(A_u, A_v)[0];
				result_img.at<cv::Vec3b>(u, v)[1] = image.at<cv::Vec3b>(A_u, A_v)[1];
				result_img.at<cv::Vec3b>(u, v)[2] = image.at<cv::Vec3b>(A_u, A_v)[2];
				break;
			case 1:
				result_img.at<cv::Vec3b>(u, v)[0] = image.at<cv::Vec3b>(B_u, B_v)[0];
				result_img.at<cv::Vec3b>(u, v)[1] = image.at<cv::Vec3b>(B_u, B_v)[1];
				result_img.at<cv::Vec3b>(u, v)[2] = image.at<cv::Vec3b>(B_u, B_v)[2];
				break;
			case 2:
				result_img.at<cv::Vec3b>(u, v)[0] = image.at<cv::Vec3b>(C_u, C_v)[0];
				result_img.at<cv::Vec3b>(u, v)[1] = image.at<cv::Vec3b>(C_u, C_v)[1];
				result_img.at<cv::Vec3b>(u, v)[2] = image.at<cv::Vec3b>(C_u, C_v)[2];
				break;
			case 3:
				result_img.at<cv::Vec3b>(u, v)[0] = image.at<cv::Vec3b>(D_u, D_v)[0];
				result_img.at<cv::Vec3b>(u, v)[1] = image.at<cv::Vec3b>(D_u, D_v)[1];
				result_img.at<cv::Vec3b>(u, v)[2] = image.at<cv::Vec3b>(D_u, D_v)[2];
				break;
			default:
				break;
			}
		}
	}
	showImg(result_img, "nearest point");
	return EXIT_SUCCESS;
}

int bilinear_interpolation_enlargement(cv::Mat image, float d){

	int img_length = image.rows;
	int len_size = d * img_length;
	cv::Mat result_img(len_size, len_size, CV_8UC3, cv::Scalar::all(0));
	for (int u = 0; u < len_size; u++){
		for (int v = 0; v < len_size; v++){
			float org_u = u / d;
			float org_v = v / d;

			// Find the nearest ABCD point
			float dist[4];
			int A_u = floor(org_u); int A_v = floor(org_v); dist[0] = sqrt((org_u - A_u) * (org_u - A_u) + (org_v - A_v) * (org_v - A_v));
			int B_u = ceil(org_u); int B_v = floor(org_v); dist[1] = sqrt((org_u - B_u) * (org_u - B_u) + (org_v - B_v) * (org_v - B_v));
			int C_u = floor(org_u); int C_v = ceil(org_v); dist[2] = sqrt((org_u - C_u) * (org_u - C_u) + (org_v - C_v) * (org_v - C_v));
			int D_u = ceil(org_u); int D_v = ceil(org_v); dist[3] = sqrt((org_u - D_u) * (org_u - D_u) + (org_v - D_v) * (org_v - D_v));

			// calculating coefficient p and q for interpolation
			float p = org_u - A_u;
			float q = org_v - A_v;

			// Check exception
			if (((org_u >= 0) && (org_u< img_length - 1)) && ((org_v >= 0) && (org_v< img_length - 1))){
				// Bilinear Interpolation: (1-p)(1-q)A + p(1-q)B + q(1-b)C + qpD
				result_img.at<cv::Vec3b>(u, v)[0] = (1 - p)*(1 - q)*image.at<cv::Vec3b>(A_u, A_v)[0]
					+ p*(1 - q)*image.at<cv::Vec3b>(B_u, B_v)[0]
					+ (1 - p)*q*image.at<cv::Vec3b>(C_u, C_v)[0]
					+ q*p*image.at<cv::Vec3b>(D_u, D_v)[0];
				result_img.at<cv::Vec3b>(u, v)[1] = (1 - p)*(1 - q)*image.at<cv::Vec3b>(A_u, A_v)[1]
					+ p*(1 - q)*image.at<cv::Vec3b>(B_u, B_v)[1]
					+ (1 - p)*q*image.at<cv::Vec3b>(C_u, C_v)[1]
					+ q*p*image.at<cv::Vec3b>(D_u, D_v)[1];
				result_img.at<cv::Vec3b>(u, v)[2] = (1 - p)*(1 - q)*image.at<cv::Vec3b>(A_u, A_v)[2]
					+ p*(1 - q)*image.at<cv::Vec3b>(B_u, B_v)[2]
					+ (1 - p)*q*image.at<cv::Vec3b>(C_u, C_v)[2]
					+ q*p*image.at<cv::Vec3b>(D_u, D_v)[2];
			}
		}
	}
	showImg(result_img, "bilinear interpolation");

	return EXIT_SUCCESS;
}

int bicubic_interpolation_enlargement(cv::Mat image, double d){

	int img_length = image.rows;
	int len_size = d * img_length;
	cv::Mat result_img(len_size, len_size, CV_8UC3, cv::Scalar::all(0));

	for (int u = 0; u < len_size; u++){
		for (int v = 0; v < len_size; v++){

			double org_u = u / d;
			double org_v = v / d;

			// Find the weight
			int A_u = floor(org_u); int A_v = floor(org_v);
			int B_u = ceil(org_u); int B_v = floor(org_v);
			int C_u = floor(org_u); int C_v = ceil(org_v);
			int D_u = ceil(org_u); int D_v = ceil(org_v);

			// calculating coefficient p and q for interpolation
			double p = org_u - A_u; // x axis fraction
			double q = org_v - A_v; // y axis fraction

			double w_x[4];
			double w_y[4];

			float a = -0.5;

			// x axis weight for 4 grids
			w_x[0] = a*abs((1 + p) * (1 + p) * (1 + p)) - 5 * a*(1 + p) * (1 + p) + 8 * a*abs((1 + p)) - 4 * a;
			w_x[1] = (a + 2)*abs(p * p * p) - (a + 3)*p * p + 1;
			w_x[2] = (a + 2)*abs((1 - p) * (1 - p) * (1 - p)) - (a + 3)*(1 - p) * (1 - p) + 1;
			w_x[3] = a*abs((2 - p) * (2 - p) * (2 - p)) - 5 * a*(2 - p) * (2 - p) + 8 * a*abs((2 - p)) - 4 * a;

			//y axis weight for 4 grids
			w_y[0] = a*abs((1 + q) * (1 + q) * (1 + q)) - 5 * a*(1 + q) * (1 + q) + 8 * a*abs((1 + q)) - 4 * a;
			w_y[1] = (a + 2)*abs(q * q * q) - (a + 3)*q * q + 1;
			w_y[2] = (a + 2)*abs((1 - q) * (1 - q) * (1 - q)) - (a + 3)*(1 - q) * (1 - q) + 1;
			w_y[3] = a*abs((2 - q) * (2 - q) * (2 - q)) - 5 * a*(2 - q) * (2 - q) + 8 * a*abs((2 - q)) - 4 * a;

			Vec3f temp = { 0, 0, 0 };

			// based on triangle sampling function (Calculating 16 nearest points interpolation)
			for (int m = 0; m <= 3; m++){
				for (int n = 0; n <= 3; n++){
					double f1 = 0.0;
					double f2 = 0.0;

					// exception checking
					if (((org_u >= 1) && (org_u< img_length - 2)) && ((org_v >= 1) && (org_v< img_length - 2))){
						// Interpolation: F' = F(u+m-1)*F(v+n-1)*W_mn
						temp = temp + (Vec3f)(image.at<Vec3b>(int(org_u) + m - 1, int(org_v) + n - 1))*w_x[m] * w_y[n];
					}

				}
			}

			result_img.at<Vec3b>(u, v) = (Vec3b)temp;
		}
	}
	showImg(result_img, "bicubic interpolation");

	return EXIT_SUCCESS;
}

int fractional_linear_enlargement(cv::Mat image, float d){
	bilinear_interpolation_enlargement(image, d);
	return EXIT_SUCCESS;
}

int negative_image(cv::Mat image){

	int rows = image.rows;
	int cols = image.cols;
	cv::Mat result_img(rows, cols, CV_8UC1, cv::Scalar::all(0));

	for (int u = 0; u < rows; u++){
		for (int v = 0; v < cols; v++){
			result_img.at<uchar>(u, v) = 225 - image.at<uchar>(u, v);
		}
	}

	showImg(image, "original");
	showImg(result_img, "bicubic interpolation");

	return EXIT_SUCCESS;
}