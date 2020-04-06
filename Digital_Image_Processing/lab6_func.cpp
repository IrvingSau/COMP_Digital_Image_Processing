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

cv::Mat dft_own(cv::Mat inputImg){
	// Fetch the best size
	cv::Mat paddedImg;
	int m = cv::getOptimalDFTSize(inputImg.rows);
	int n = cv::getOptimalDFTSize(inputImg.cols);

	std::cout << "图片原始尺寸为：" << inputImg.cols << "*" << inputImg.rows << std::endl;
	std::cout << "DFT最佳尺寸为：" << n << "*" << m << std::endl;

	// Padding the image
	cv::copyMakeBorder(inputImg, paddedImg, 0, m - inputImg.rows,
		0, n - inputImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	// Split the image into matArray, 0 for real and 1 for image
	cv::Mat matArray[] = { cv::Mat_<float>(paddedImg), cv::Mat::zeros(paddedImg.size(), CV_32F) };
	cv::Mat complexInput, complexOutput;
	merge(matArray, 2, complexInput);

	cv::dft(complexInput, complexOutput);
	
	return complexOutput;
}

int dft_visualize(cv::Mat inputImg, ::Mat complexOutput){
	// Fetch the best size
	cv::Mat paddedImg;
	int m = cv::getOptimalDFTSize(inputImg.rows);
	int n = cv::getOptimalDFTSize(inputImg.cols);

	cv::Mat matArray[] = { cv::Mat_<float>(paddedImg), cv::Mat::zeros(paddedImg.size(), CV_32F) };

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

	return 1;
}

int highpass_filter_sharpen(cv::Mat inputImg, double d0){
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
			p[2 * j] = 1 - expf(-d / (2 * D0*D0));
			p[2 * j + 1] = 1 - expf(-d / (2 * D0*D0));
		}
	}

	cv::Mat butterworthBlur(paddedImg.size(), CV_32FC2);
	int n_para = 2;
	for (int i = 0; i<paddedImg.rows; i++){
		float*p = butterworthBlur.ptr<float>(i);
		for (int j = 0; j<paddedImg.cols; j++){
			// Evaluate the distance to center point
			double distance = pow(i - paddedImg.rows / 2, 2) + pow(j - paddedImg.cols / 2, 2);
			p[2 * j] = 1 / (1 + pow((D0 / distance), 2 * n_para));
			p[2 * j + 1] = 1 / (1 + pow((D0 / distance), 2 * n_para));
		}
	}

	cv::Mat idpf(paddedImg.size(), CV_32FC2);

	for (int i = 0; i<paddedImg.rows; i++){
		float*p = idpf.ptr<float>(i);
		for (int j = 0; j<paddedImg.cols; j++){
			double d = sqrt(pow(i - paddedImg.rows / 2, 2) + pow(j - paddedImg.cols / 2, 2));
			if (d < 60){
				p[2 * j] = 0 ;
				p[2 * j + 1] = 0 ;
			}
			else{
					p[2*j] = 1;
					p[2 * j + 1] = 1;
				}
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
	cv::multiply(complexOutput, idpf, idpf);

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

	cv::split(idpf, matArray);
	cv::magnitude(matArray[0], matArray[1], magImg);
	magImg += cv::Scalar::all(1);
	cv::log(magImg, magImg);
	cv::normalize(magImg, magImg, 1, 0, CV_MINMAX);
	cv::namedWindow("Spectrum for idpf", cv::WINDOW_NORMAL);
	cv::imshow("Spectrum for idpf", magImg);

	//IDFT得到滤波结果
	cv::Mat gaussianBlurImg;
	cv::idft(gaussianBlur, complexIDFT);
	cv::split(complexIDFT, matArray);
	cv::magnitude(matArray[0], matArray[1], gaussianBlurImg);
	cv::normalize(gaussianBlurImg, gaussianBlurImg, 0, 1, CV_MINMAX);
	cv::namedWindow("gaussianBlurImg", cv::WINDOW_NORMAL);
	cv::imshow("gaussianBlurImg", gaussianBlurImg);

	//butterworth得到滤波结果
	cv::Mat butterworthBlurImg;
	cv::idft(butterworthBlur, complexIDFT);
	cv::split(complexIDFT, matArray);
	cv::magnitude(matArray[0], matArray[1], butterworthBlurImg);
	cv::normalize(butterworthBlurImg, butterworthBlurImg, 0, 1, CV_MINMAX);

	cv::namedWindow("butterworthBlurImg", cv::WINDOW_NORMAL);
	cv::imshow("butterworthBlurImg", butterworthBlurImg);

	//idpf得到滤波结果
	cv::Mat idpfImg;
	cv::idft(idpf, complexIDFT);
	cv::split(complexIDFT, matArray);
	cv::magnitude(matArray[0], matArray[1], idpfImg);
	cv::normalize(idpfImg, idpfImg, 0, 1, CV_MINMAX);

	cv::namedWindow("idpfBlurImg", cv::WINDOW_NORMAL);
	cv::imshow("idpfBlurImg", idpfImg);
	/******************************************************/


	cv::waitKey(0);
	return 1;
}

int homomorphic_enhancement(cv::Mat image){
	cv::Mat inputImg;
	image.convertTo(inputImg, CV_32FC1, 1.0 / 255, 0);
	//inputImg = image;
	cv::namedWindow("Input image", cv::WINDOW_NORMAL);
	cv::imshow("Input image", image);
	

	for (int i = 0; i < inputImg.rows; i++){
		for (int j = 0; j < inputImg.cols; j++){
			//printf("%f\n", inputImg.at<float>(i, j));
		}
	}

	//Find the best size for image
	cv::Mat paddedImg;
	int m = cv::getOptimalDFTSize(inputImg.rows);
	int n = cv::getOptimalDFTSize(inputImg.cols);

	std::cout << "图片原始尺寸为：" << inputImg.cols << "*" << inputImg.rows << std::endl;
	std::cout << "DFT最佳尺寸为：" << n << "*" << m << std::endl;

	// Padding the image
	cv::copyMakeBorder(inputImg, paddedImg, 0, m - inputImg.rows, 0, n - inputImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::imshow("before log", paddedImg);
	// Take ln transformation for the original image
	paddedImg += cv::Scalar::all(1);
	cv::log(paddedImg, paddedImg);

	cv::imshow("after log", paddedImg);

	// transforme the padded image into a 2d array for dft
	cv::Mat matArray[] = { cv::Mat_<float>(paddedImg), cv::Mat::zeros(paddedImg.size(), CV_32F) };
	cv::Mat complexInput, complexOutput;
	merge(matArray, 2, complexInput);

	// DFT Transformation
	cv::dft(complexInput, complexOutput);

	// Calculate the dft magnitude: store the dft result in compelexOutput 
	// and matArray store the real part and image part
	cv::split(complexOutput, matArray);
	cv::Mat magImg;
	cv::magnitude(matArray[0], matArray[1], magImg);

	// Log transformation
	magImg += cv::Scalar::all(1);
	cv::log(magImg, magImg);

	// shifting
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

	// Visualization for images
	
	cv::namedWindow("Spectrum image", cv::WINDOW_NORMAL);
	cv::imshow("Spectrum image", magImg);

	// Filter In Frequency Domain
	// Here we select the gaussian filter as the candidate for manipulation in frequency domain
	cv::Mat gaussianfilter(paddedImg.size(), CV_32FC2);
	double gammaH = 1.5;
	double gammaL = 0.5;
	double D0 = 2;
	double c = 1;
	for (int i = 0; i<paddedImg.rows; i++)
	{
		float*p = gaussianfilter.ptr<float>(i);
		for (int j = 0; j<paddedImg.cols; j++)
		{
			double d = pow(i - paddedImg.rows / 2, 2) + pow(j - paddedImg.cols / 2, 2);
			p[2 * j] = (1 - expf(-d / (2 * D0*D0))) * (gammaH - gammaL) + gammaL;
			p[2 * j + 1] = (1 - expf(-c * (d*d) / (2 * D0*D0))) * (gammaH - gammaL) + gammaL;
		}
	}

	// Apply the gaussian filter
	printf("%d, %d\n", complexOutput.size().height, complexOutput.size().width);
	printf("%d, %d\n", gaussianfilter.size().height, gaussianfilter.size().width);

	cv::multiply(complexOutput, gaussianfilter, gaussianfilter);
	cv::exp(complexOutput, complexOutput);

	cv::split(gaussianfilter, matArray);
	cv::magnitude(matArray[0], matArray[1], magImg);
	magImg += cv::Scalar::all(1);
	cv::log(magImg, magImg);
	cv::normalize(magImg, magImg, 1, 0, CV_MINMAX);
	cv::namedWindow("Spectrum after Gaussian", cv::WINDOW_NORMAL);
	cv::imshow("Spectrum after Gaussian", magImg);
	

	cv::Mat gaussianBlurImg;
	cv::Mat complexIDFT;
	cv::idft(gaussianfilter, complexIDFT);
	cv::split(complexIDFT, matArray);
	cv::magnitude(matArray[0], matArray[1], gaussianBlurImg);
	cv::normalize(gaussianBlurImg, gaussianBlurImg, 0, 1, CV_MINMAX);
	cv::namedWindow("After applied filter", cv::WINDOW_NORMAL);
	cv::imshow("After applied filter", gaussianBlurImg);
	waitKey();

	return 1;
}

cv::Mat sinusoidal_noise(cv::Mat inputImg){
	
	// Histogram Equalization
	int srcrow = inputImg.rows;
	int srccol = inputImg.cols;
	int dstrow = srcrow;
	int dstcol = srccol;
	cv::Mat dst_img(dstrow, dstcol, CV_8UC1, cv::Scalar::all(0));

	double A = 20;
	double PI = 3.1415926;
	double a = 100;

	for (int i = 0; i < inputImg.rows; i++){
		for (int j = 0; j < inputImg.cols; j++){
			dst_img.at<uchar>(i, j) = inputImg.at<uchar>(i, j) + A*sin(PI* a * i / dstrow) + A*sin(PI* a * j / dstcol);
		}
	}

	/*
	imshow("Sin noise", dst_img);
	
	cv::Mat complexOutput;
	complexOutput = dft_own(dst_img);
	dft_visualize(dst_img, complexOutput);
	*/

	return dst_img;
}

int bandreject_filter(cv::Mat noise_img){
	cv::Mat complexOutput;
	complexOutput = dft_own(noise_img);
	dft_visualize(noise_img, complexOutput);
	
	// Applied the filter to spectrum image
	cv::Mat butterworthFilter(noise_img.size(), CV_32FC2);
	cv::Mat matArray[] = { cv::Mat_<float>(noise_img), cv::Mat::zeros(noise_img.size(), CV_32F) };

	double D0 = 2800;
	double w = 1000;
	double n = 20;

	for (int i = 0; i<noise_img.rows; i++)
	{
		float*p = butterworthFilter.ptr<float>(i);
		for (int j = 0; j<noise_img.cols; j++)
		{
			double distance = pow(i - noise_img.rows / 2, 2) + pow(j - noise_img.cols / 2, 2);
			p[2 * j] = 1 / (1 + pow((distance * w/(distance * distance - D0*D0)), 2*n));
			p[2 * j + 1] = 1 / (1 + pow((distance * w / (distance * distance - D0*D0)), 2 * n));
		}
	}
	cv::Mat complexIDFT, IDFTImg;
	cv::idft(complexOutput, complexIDFT);
	cv::Mat magImg;

	cv::split(complexOutput, matArray);
	int colToCut = noise_img.cols / 2;
	int rowToCut = noise_img.rows / 2;
	cv::Mat q1(matArray[0], cv::Rect(0, 0, colToCut, rowToCut));
	cv::Mat q2(matArray[0], cv::Rect(colToCut, 0, colToCut, rowToCut));
	cv::Mat q3(matArray[0], cv::Rect(0, rowToCut, colToCut, rowToCut));
	cv::Mat q4(matArray[0], cv::Rect(colToCut, rowToCut, colToCut, rowToCut));

	cv::Mat tmpImg;
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


	// Applied the filter by multiplication
	cv::multiply(complexOutput, butterworthFilter, butterworthFilter);

	// Visualize
	
	cv::split(butterworthFilter, matArray);
	cv::magnitude(matArray[0], matArray[1], magImg);
	magImg += cv::Scalar::all(1);
	cv::log(magImg, magImg);
	cv::normalize(magImg, magImg, 1, 0, CV_MINMAX);
	cv::namedWindow("Spectrum for filter", cv::WINDOW_NORMAL);
	cv::imshow("Spectrum for filter", magImg);

	cv::Mat butterworthBlurImg;
	cv::idft(butterworthFilter, complexIDFT);
	cv::split(complexIDFT, matArray);
	cv::magnitude(matArray[0], matArray[1], butterworthBlurImg);
	cv::normalize(butterworthBlurImg, butterworthBlurImg, 0, 1, CV_MINMAX);

	cv::namedWindow("After applied filter", cv::WINDOW_NORMAL);
	cv::imshow("After applied filter", butterworthBlurImg);
	waitKey();

	return 1;
}