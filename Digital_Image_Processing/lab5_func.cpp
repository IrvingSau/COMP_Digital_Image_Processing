#include "lab2.h"
#include "lab4.h"
#include<fstream> 
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#pragma once
#include "complexNumber.h"

using namespace cv;
using namespace std;


/*
class ComplexNumber{
public:
double real;
double image;
ComplexNumber add(const ComplexNumber &adder){
real += adder.real;
image += adder.image;
return *this;
}
double mo_value(){
double mold;
mold = sqrt(real*real+ image*image);
return mold;
}
};*/


double str2double(std::string src)
{
	double ret = 0, sign = 1;
	const char *p = src.c_str();

	if (*p == '+')sign = 1, p++;
	else if (*p == '-') sign = -1, p++;

	while (*p && (*p != '.'))
	{
		ret *= 10;
		ret += (*p) - '0';
		p++;
	}
	if (*p == '.')
	{
		double step = 0.1;
		p++;
		while (*p)
		{
			ret += step*((*p) - '0');
			step /= 10;
			p++;
		}
	}

	return ret*sign;
}


int idft_own(CComplexNumber* dft_values, int width, int height){
	printf("Start IDFT\n");
	double PI = 3.141592653589;

	cv::Mat dst_img(height, width, CV_8UC1, cv::Scalar::all(0));

	double* result_data = new double[width*height];

	memset(result_data, 0, sizeof(result_data)*sizeof(double));

	// Set for power value
	double fixed_factor_for_axisX = (2 * PI) / height;
	double fixed_factor_for_axisY = (2 * PI) / width;

	for (int x = 0; x<height; x++) {
		for (int y = 0; y<width; y++) {
			for (int u = 0; u<height; u++) {
				for (int v = 0; v<width; v++) {
					double powerU = u * x * fixed_factor_for_axisX;         // evaluate i2πux/N
					double powerV = v * y * fixed_factor_for_axisY;         // evaluate i2πux/N
					CComplexNumber cplTemp;
					cplTemp.SetValue(cos(powerU + powerV), sin(powerU + powerV));
					result_data[y + x*width] = result_data[y + x*width] +
						((dft_values[v + u*width] * cplTemp).real
						/ (height*width));
				}
			}
		}
	}

	for (int u = 0; u < height; u++){
		for (int v = 0; v < width; v++){
			dst_img.at<uchar>(u, v) = (unsigned char)result_data[u*width + v];
		}
	}
	printf("End IDFT\n");
	showImg(dst_img, "idft");
	waitKey(0);

	return 0;
}

int dft_own(cv::Mat image, String filename){

	double PI = 3.141592653589;

	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srcrow;
	int dstcol = srccol;
	printf("Start DFT\n");
	// Prepare for the array to store result
	CComplexNumber *dft_values = new CComplexNumber[dstrow*dstcol];
	CComplexNumber   cplTemp(0, 0);

	// Initialize the dft_values array
	for (int j = 0; j < dstrow*dstcol; j++){
		dft_values[j].real = 0;
		dft_values[j].image = 0;
	}
	double fixed_factor_for_axisX = (-2 * PI) / dstrow;                   // evaluate -i2π/N of -i2πux/N, and store the value for computing efficiency
	double fixed_factor_for_axisY = (-2 * PI) / dstcol;
	// Calculate the dft_values
	for (int u = 0; u < dstrow; u++){
		for (int v = 0; v < dstcol; v++){
			// Sum
			for (int x = 0; x < dstrow; x++){
				for (int y = 0; y < dstcol; y++){

					double powerX = u * x * fixed_factor_for_axisX;
					double powerY = v * y * fixed_factor_for_axisY;

					cplTemp.real = image.at<uchar>(x, y) * cos(powerX + powerY);
					cplTemp.image = image.at<uchar>(x, y) * sin(powerX + powerY);

					//printf("index = %d\n", v + u*dstcol);
					dft_values[v + u*dstcol] = dft_values[v + u*dstcol] + cplTemp;
				}
			}
		}
	}
	printf("End DFT\n");
	// idft
	idft_own(dft_values, dstcol, dstrow);

	// Save to file
	ofstream file(filename + ".csv");
	if (file){
		for (int i = 0; i < dstcol*dstrow; i++){
			file << dft_values[i].real << "," << dft_values[i].image << "\n";
		}
	}
	file.close();

	printf("Start Visualize");
	// Convert dft_values to 2d array
	//double dft_values_array[2][65536];
	//int(*dft_values_array)[65536] = new int[2][65536];
	double **dft_values_array = new double*[2];
	for (int i = 0; i < 2; i++)
	{
		dft_values_array[i] = new double[dstcol*dstrow];
	}

	for (int i = 0; i < dstcol*dstrow; i++){
		dft_values_array[0][i] = dft_values[i].real;
		dft_values_array[1][i] = dft_values[i].image;
	}



	cv::Mat dst1[] = { cv::Mat::zeros(dstrow, dstcol, CV_64F), cv::Mat::zeros(dstrow, dstcol, CV_64F) };
	// conver 2d array to mat 2d array
	for (int i = 0; i < dstrow; i++){
		for (int j = 0; j < dstcol; j++){
			dst1[0].at<double>(i, j) = dft_values_array[0][j + i*dstcol];
			dst1[0].at<double>(i, j) = dft_values_array[0][j + i*dstcol];
		}
	}

	for (int i = 0; i < 2; i++)
	{
		delete[] dft_values_array[i];
	}
	delete[] dft_values_array;

	cv::magnitude(dst1[0], dst1[1], dst1[0]);
	cv::Mat magnitudeImage = dst1[0];

	magnitudeImage += Scalar::all(1);
	cv::log(magnitudeImage, magnitudeImage);

	magnitudeImage = magnitudeImage(cv::Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));

	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	cv::Mat q0(magnitudeImage(cv::Rect(0, 0, cx, cy)));
	cv::Mat q1(magnitudeImage(cv::Rect(cx, 0, cx, cy)));
	cv::Mat q2(magnitudeImage(cv::Rect(0, cy, cx, cy)));
	cv::Mat q3(magnitudeImage(cv::Rect(cy, cy, cx, cy)));

	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	cv::normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);

	cv::imshow("spectrum magnitude", magnitudeImage);

	cv::waitKey();



	//imwrite(filename, dst_img);

	return EXIT_SUCCESS;
}

int dft_visualize(cv::Mat image){
	// Visualize (Normalize + centralize)
	int srcrow = image.rows;
	int srccol = image.cols;
	int dstrow = srcrow;
	int dstcol = srccol;

	cv::Mat dst_img(dstrow, dstcol, CV_8UC1, cv::Scalar::all(0));

	ifstream inFile("lena.png.csv", ios::in);
	string lineStr;
	vector<vector<string>> strArray;
	int temp[65536][2];
	int i, j;
	i = 0;
	char* end;
	if (inFile.fail())
		cout << "Read failure" << endl;
	while (getline(inFile, lineStr))
	{
		j = 0;


		stringstream ss(lineStr);
		string str;
		vector<string> lineArray;

		while (getline(ss, str, ','))
		{
			temp[i][j] = static_cast<int>(strtol(str.c_str(), &end, 10));              //string -> int
			j++;
		}
		i++;
	}




	/*ComplexNumber *dft_values = new ComplexNumber[dstrow*dstcol];
	for (int i = 0; i < dstrow*dstcol; i++){
	dft_values[i].real = temp[i][0];
	dft_values[i].image = temp[i][1];
	}*/

	/*
	// Normalize
	double *mo = new double[dstrow*dstcol];
	for (int i = 0; i<dstrow*dstcol; i++){
	mo[i] = dft_values[i].mo_value();
	}
	// Log transformation
	//for (int i = 0; i < dstrow*dstcol; i++){
	//mo[i] = log(mo[i] + 1);
	//}
	// Find the min
	double min = mo[0];
	for (int i = 0; i < dstrow*dstcol; i++){
	if (mo[i] < min)
	min = mo[i];
	}
	// Find the max
	double max = mo[0];
	for (int i = 0; i < dstrow*dstcol; i++){
	if (mo[i] > max)
	max = mo[i];
	}
	// Normalize: (v[i] - min) / (max-min)
	for (int i = 0; i < dstrow*dstcol; i++){
	mo[i] = (mo[i] - min) / (max - min);
	}
	// Visualize
	printf("Start visualize");
	/*for (int u = 0; u < dstrow; u++){
	for (int v = 0; v < dstcol; v++){
	if ((u<(dstrow / 2)) && (v<(dstcol / 2))) {
	dst_img.at<uchar>(u, v) =
	mo[dstcol / 2 + v + (dstrow / 2 + u)*dstcol];
	}
	else if ((u<(dstrow / 2)) && (v >= (dstcol / 2))) {
	dst_img.at<uchar>(u, v) =
	mo[(v - dstcol / 2) + (dstrow / 2 + u)*dstcol];
	}
	else if ((u >= (dstrow / 2)) && (v<(dstcol / 2))) {
	dst_img.at<uchar>(u, v) =
	mo[(dstcol / 2 + v) + (u - dstrow / 2)*dstcol];
	}
	else if ((u >= (dstrow / 2)) && (v >= (dstcol / 2))) {
	dst_img.at<uchar>(u, v) =
	mo[(v - dstcol / 2) + (u - dstrow / 2)*dstcol];
	}
	}
	}*/
	/*
	for (int u = 0; u < dstrow; u++){
	for (int v = 0; v < dstcol; v++){
	if ((u < (dstrow / 2)) && (v < (dstcol / 2))){
	dst_img.at<uchar>(u, v) = 255 * mo[dstcol / 2 + v + (dstrow / 2 + u)*dstcol];
	}
	if ((u < (dstrow / 2)) && (v >= (dstcol / 2))){
	dst_img.at<uchar>(u, v) = 255 * mo[(v - dstcol / 2) + (dstrow / 2 + u)*dstcol];
	}
	if ((u >= (dstrow / 2)) && (v < (dstcol / 2))){
	dst_img.at<uchar>(u, v) = 255 * mo[(dstcol / 2 + v) + (u - dstrow / 2)*dstcol];
	}
	if ((u >= (dstrow / 2)) && (v >= (dstcol / 2))){
	dst_img.at<uchar>(u, v) = 255 * mo[(v - dstcol / 2) + (u - dstrow / 2)*dstcol];
	}
	}
	}
	printf("Finish visualize");
	showImg(dst_img, "dft");
	waitKey(0);
	*/
	return EXIT_SUCCESS;
}

int ilpf(Mat mag, int D0, int center_y, int center_x, Mat tmp, Mat q0, Mat q1, Mat q2, Mat q3){

	// ilpf algorithm using threshold at D0
	for (int y = 0; y < mag.rows; y++){
		// Create an array for recording the result for each row
		double* ilpf_values = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++){
			// Evaluate the distance to center point
			double distance = sqrt((y - center_y)*(y - center_y) + (x - center_x)*(x - center_x));
			if (distance <= D0){
				ilpf_values[x] = 1 * ilpf_values[x];
			}
			else{
				ilpf_values[x] = 0;
			}
		}
	}



	// Visualization of the result
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// Conver to spatial domain, which is same as idft
	Mat invDFT, invDFTcvt;
	idft(mag, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("ilpf", invDFTcvt);
	waitKey();

	return 1;
}

int butterworth(Mat mag, int D0, int center_y, int center_x, Mat tmp, Mat q0, Mat q1, Mat q2, Mat q3){
	int n = 2;
	// ilpf algorithm using threshold at D0
	for (int y = 0; y < mag.rows; y++){
		// Create an array for recording the result for each row
		double* butterworth_values = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++){
			// Evaluate the distance to center point
			double distance = sqrt((y - center_y)*(y - center_y) + (x - center_x)*(x - center_x));
			double h = 1.0 / (1.0 + pow((distance / D0), 2 * n));
			//printf("%f\n", h);
			if (h < 0.5)
				butterworth_values[x] = 0;
			//printf("%f\n", butterworth_values[x]);
		}
	}

	// Visualization of the result
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// Conver to spatial domain, which is same as idft
	Mat invDFT, invDFTcvt;
	idft(mag, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("butterworth", invDFTcvt);
	waitKey();


	return 1;
}

int gaussian(Mat mag, int D0, int center_y, int center_x, Mat tmp, Mat q0, Mat q1, Mat q2, Mat q3){
	int n = 2;
	// ilpf algorithm using threshold at D0
	for (int y = 0; y < mag.rows; y++){
		// Create an array for recording the result for each row
		double* gaussian_values = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++){
			// Evaluate the distance to center point
			double distance = sqrt((y - center_y)*(y - center_y) + (x - center_x)*(x - center_x));
			double h = exp(-distance*distance / (2 * D0*D0));
			//printf("%f, %f\n", distance, h);

			//printf("%f\n", h);
			if (h < 0.5)
				gaussian_values[x] = 0;
			//printf("%f\n", butterworth_values[x]);
		}
	}

	// Visualization of the result
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// Conver to spatial domain, which is same as idft
	Mat invDFT, invDFTcvt;
	idft(mag, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("butterworth", invDFTcvt);
	waitKey();


	return 1;
}

int dft_samples(Mat src){
	Mat img = src;
	//cvtColor(src, img, CV_BGR2GRAY);
	imshow("img", img);
	//调整图像加速傅里叶变换
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	//记录傅里叶变换的实部和虚部
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	//进行傅里叶变换
	dft(complexImg, complexImg);
	//获取图像
	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));//这里为什么&上-2具体查看opencv文档
	//其实是为了把行和列变成偶数 -2的二进制是11111111.......10 最后一位是0
	//获取中心点坐标
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	//调整频域
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//Do为自己设定的阀值具体看公式
	double D0 = 60;

	//ilpf(mag, D0, cy, cx, tmp, q0, q1, q2, q3);
	//butterworth(mag, D0, cy, cx, tmp, q0, q1, q2, q3);
	gaussian(mag, D0, cy, cx, tmp, q0, q1, q2, q3);

	return 1;
}



int myFFT(cv::Mat& inputImg, double d0)
{
	//得到DFT的最佳尺寸（2的指数），以加速计算
	cv::Mat paddedImg;
	int m = cv::getOptimalDFTSize(inputImg.rows);
	int n = cv::getOptimalDFTSize(inputImg.cols);

	std::cout << "图片原始尺寸为：" << inputImg.cols << "*" << inputImg.rows << std::endl;
	std::cout << "DFT最佳尺寸为：" << n << "*" << m << std::endl;

	//填充图像的下端和右端
	cv::copyMakeBorder(inputImg, paddedImg, 0, m - inputImg.rows,
		0, n - inputImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	//将填充的图像组成一个复数的二维数组（两个通道的Mat），用于DFT
	cv::Mat matArray[] = { cv::Mat_<float>(paddedImg), cv::Mat::zeros(paddedImg.size(), CV_32F) };
	cv::Mat complexInput, complexOutput;
	merge(matArray, 2, complexInput);

	cv::dft(complexInput, complexOutput);

	//计算幅度谱（傅里叶谱）
	cv::split(complexOutput, matArray);
	cv::Mat magImg;
	cv::magnitude(matArray[0], matArray[1], magImg);

	//转换到对数坐标
	magImg += cv::Scalar::all(1);
	cv::log(magImg, magImg);

	//将频谱图像裁剪成偶数并将低频部分放到图像中心(以下原文中没被注释，但是我测试时有问题，应该是图像长宽越界了，所以直接注释掉)
	//    int width = (magImg.cols / 2)*2;
	//    int height = (magImg.cols / 2)*2;
	//    magImg = magImg(cv::Rect(0, 0, width, height));

	int colToCut = magImg.cols / 2;
	int rowToCut = magImg.rows / 2;

	//获取图像，分别为左上右上左下右下
	//注意这种方式得到的是magImg的ROI的引用
	//对下面四幅图像进行修改就是直接对magImg进行了修改
	cv::Mat topLeftImg(magImg, cv::Rect(0, 0, colToCut, rowToCut));
	cv::Mat topRightImg(magImg, cv::Rect(colToCut, 0, colToCut, rowToCut));
	cv::Mat bottomLeftImg(magImg, cv::Rect(0, rowToCut, colToCut, rowToCut));
	cv::Mat bottomRightImg(magImg, cv::Rect(colToCut, rowToCut, colToCut, rowToCut));

	//第二象限和第四象限进行交换
	//cv::Mat tmpImg1 = topLeftImg.clone();
	cv::Mat tmpImg;
	topLeftImg.copyTo(tmpImg);
	bottomRightImg.copyTo(topLeftImg);
	tmpImg.copyTo(bottomRightImg);

	//第一象限和第三象限进行交换
	//cv::Mat tmpImg2 = bottomLeftImg.clone();
	bottomLeftImg.copyTo(tmpImg);
	topRightImg.copyTo(bottomLeftImg);
	tmpImg.copyTo(topRightImg);

	//归一化图像
	cv::normalize(magImg, magImg, 0, 1, CV_MINMAX);

	//傅里叶反变换
	cv::Mat complexIDFT, IDFTImg;
	cv::idft(complexOutput, complexIDFT);
	cv::split(complexIDFT, matArray);
	cv::magnitude(matArray[0], matArray[1], IDFTImg);
	cv::normalize(IDFTImg, IDFTImg, 0, 1, CV_MINMAX);

	cv::namedWindow("Input image", cv::WINDOW_NORMAL);
	cv::imshow("Input image", inputImg);
	cv::namedWindow("Spectrum image", cv::WINDOW_NORMAL);
	cv::imshow("Spectrum image", magImg);
	cv::namedWindow("idft img", cv::WINDOW_NORMAL);
	cv::imshow("idft img", IDFTImg);

	/***********************频率域滤波部分*************************/
	// initialize the gaussian filter
	cv::Mat gaussianBlur(paddedImg.size(), CV_32FC2);
	double D0 = d0;
	for (int i = 0; i<paddedImg.rows; i++)
	{
		float*p = gaussianBlur.ptr<float>(i);
		for (int j = 0; j<paddedImg.cols; j++)
		{
			double d = pow(i - paddedImg.rows / 2, 2) + pow(j - paddedImg.cols / 2, 2);
			p[2 * j] = expf(-d / D0);
			p[2 * j + 1] = expf(-d / D0);
		}
	}

	cv::Mat butterworthBlur(paddedImg.size(), CV_32FC2);
	int n_para = 20;
	for (int i = 0; i<paddedImg.rows; i++){
		float*p = butterworthBlur.ptr<float>(i);
		for (int j = 0; j<paddedImg.cols; j++){
			// Evaluate the distance to center point
			double distance = pow(i - paddedImg.rows / 2, 2) + pow(j - paddedImg.cols / 2, 2);
			p[2 * j] = 1 / (1 + pow((distance / D0), 2 * n_para));
			p[2 * j + 1] = 1 / (1 + pow((distance / D0), 2 * n_para));
		}
	}


	cv::split(complexOutput, matArray);

	cv::Mat q1(matArray[0], cv::Rect(0, 0, colToCut, rowToCut));
	cv::Mat q2(matArray[0], cv::Rect(colToCut, 0, colToCut, rowToCut));
	cv::Mat q3(matArray[0], cv::Rect(0, rowToCut, colToCut, rowToCut));
	cv::Mat q4(matArray[0], cv::Rect(colToCut, rowToCut, colToCut, rowToCut));

	q1.copyTo(tmpImg);
	q4.copyTo(q1);
	tmpImg.copyTo(q4);

	q2.copyTo(tmpImg);
	q3.copyTo(q2);
	tmpImg.copyTo(q3);

	cv::Mat qq1(matArray[1], cv::Rect(0, 0, colToCut, rowToCut));
	cv::Mat qq2(matArray[1], cv::Rect(colToCut, 0, colToCut, rowToCut));
	cv::Mat qq3(matArray[1], cv::Rect(0, rowToCut, colToCut, rowToCut));
	cv::Mat qq4(matArray[1], cv::Rect(colToCut, rowToCut, colToCut, rowToCut));

	qq1.copyTo(tmpImg);
	qq4.copyTo(qq1);
	tmpImg.copyTo(qq4);

	qq2.copyTo(tmpImg);
	qq3.copyTo(qq2);
	tmpImg.copyTo(qq3);

	cv::merge(matArray, 2, complexOutput);

	cv::multiply(complexOutput, gaussianBlur, gaussianBlur);
	cv::multiply(complexOutput, butterworthBlur, butterworthBlur);
	//cv::multiply(complexOutput, gaussianSharpen, gaussianSharpen);

	//计算频谱
	cv::split(gaussianBlur, matArray);
	cv::magnitude(matArray[0], matArray[1], magImg);
	magImg += cv::Scalar::all(1);
	cv::log(magImg, magImg);
	cv::normalize(magImg, magImg, 1, 0, CV_MINMAX);
	cv::namedWindow("Spectrum for Gaussian", cv::WINDOW_NORMAL);
	cv::imshow("Spectrum for Gaussian", magImg);

	cv::split(butterworthBlur, matArray);
	cv::magnitude(matArray[0], matArray[1], magImg);
	magImg += cv::Scalar::all(1);
	cv::log(magImg, magImg);
	cv::normalize(magImg, magImg, 1, 0, CV_MINMAX);
	cv::namedWindow("Spectrum for Butterworth", cv::WINDOW_NORMAL);
	cv::imshow("Spectrum for Butterworth", magImg);

	//IDFT得到滤波结果
	cv::Mat gaussianBlurImg;
	cv::idft(gaussianBlur, complexIDFT);
	cv::split(complexIDFT, matArray);
	cv::magnitude(matArray[0], matArray[1], gaussianBlurImg);
	cv::normalize(gaussianBlurImg, gaussianBlurImg, 0, 1, CV_MINMAX);
	cv::namedWindow("gaussianBlurImg", cv::WINDOW_NORMAL);
	cv::imshow("gaussianBlurImg", gaussianBlurImg);

	//IDFT得到滤波结果
	cv::Mat butterworthBlurImg;
	cv::idft(butterworthBlur, complexIDFT);
	cv::split(complexIDFT, matArray);
	cv::magnitude(matArray[0], matArray[1], butterworthBlurImg);
	cv::normalize(butterworthBlurImg, butterworthBlurImg, 0, 1, CV_MINMAX);



	cv::namedWindow("butterworthBlurImg", cv::WINDOW_NORMAL);
	cv::imshow("butterworthBlurImg", butterworthBlurImg);
	/******************************************************/


	cv::waitKey(0);
	return 0;
}