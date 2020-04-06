/* First Class Project */
#include "lab2.h"
#include "lab3.h"
#include "lab4.h"
#include "lab5.h"
#include "lab6.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	//cv::Mat image1 = loadImg("C:\\OpenCV\\opencv\\sources\\samples\\data\\labstuff\\lena.bmp");
	//cv::Mat image2 = loadImg("C:\\OpenCV\\opencv\\sources\\samples\\data\\grey_image.jfif");

	/***************** Lab 2 3.13 ********************/

	//showImg(image1, "original image");
	//printf("type = %d", image.type());
	//alternative_line_reduction(image1);
	//alternative_line_reduction(image2);

	//fractional_linear_transformation(image1, 0.5);
	//fractional_linear_transformation(image2, 0.5);

	//pixel_replication_enlargement(image1, 1.1, 1.5);
	//pixel_replication_enlargement(image2, 1.1, 1.5);

	//nearest_point_enlargement(image1, 1.7);
	//nearest_point_enlargement(image2, 1.7);

	//bilinear_interpolation_enlargement(image1, 1.7);
	//bilinear_interpolation_enlargement(image2, 1.7);


	//bicubic_interpolation_enlargement(image1, 1.7);
	//bicubic_interpolation_enlargement(image2, 1.7);

	//fractional_linear_enlargement(image1, 1.7);
	//fractional_linear_enlargement(image2, 1.7);

	//cv::Mat grey_image1 = loadGreyImg("C:\\OpenCV\\opencv\\sources\\samples\\data\\grey_image.jfif");
	//cv::Mat grey_image2 = loadGreyImg("C:\\OpenCV\\opencv\\sources\\samples\\data\\labstuff\\lena.bmp");
	//negative_image(grey_image1);
	//negative_image(grey_image2);




	/***************** Lab 3 3.17 ********************/
	//image_translation(image1, 10, 20);
	//image_translation(image2, 10, 20);

	//image_rotation(image1, 10);
	//image_rotation(image2, 10);

	//int type = 0;
	//image_shear(image1, 0, 30);
	//image_shear(image2, 1, 30);


	//image_smoothing(image1, 5, 0);
	//image_smoothing(image2, 5, 0);

	//image_smoothing_median(image1, 5);
	//image_smoothing_median(image2, 5);

	//image_smoothing_binary(grey_image1, 5, 0.5);
	//image_smoothing_binary(grey_image2, 5, 0.5);
	/*************************************************/


	/***************** Lab 3 3.19 ********************/

	//edge_detection_sobel(grey_image1, 0.5);
	//edge_detection_sobel(grey_image2, 0.5);

	//image_sharpen_laplacian(grey_image1);
	//image_sharpen_laplacian(grey_image2);

	//gammar_correction(grey_image1, 0.9);
	//gammar_correction(grey_image2, 0.9);

	//histogram_enhancement_global(grey_image1);
	//histogram_enhancement_global(grey_image2);

	//histogram_enhancement_local(grey_image1, 4, 2, 0.2, 5);
	//histogram_enhancement_local(grey_image2, 4, 2, 0.2, 5);

	/***************** Lab 5 3.28 ********************/


	//cv::Mat grey_image1 = loadGreyImg("C:\\OpenCV\\opencv\\sources\\samples\\data\\sobel_circle.png");
	//cv::Mat grey_image2 = loadGreyImg("D:\\OpenCV\\opencv\\sources\\samples\\data\\lena.bmp");
	//cv::Mat grey_image3 = loadGreyImg("C:\\OpenCV\\opencv\\sources\\samples\\data\\baby.png");
	//cv::Mat b_img = loadGreyImg("C:\\Users\\61003\\Desktop\\1.jpg");
	//printf("%d, $d", grey_image1.channels(), grey_image2.channels());
	//cv::Mat grey_image3 = loadImg("C:\\OpenCV\\opencv\\sources\\samples\\data\\labstuff\\baby.png");
	//cv::Mat grey_img;
	//dft_own(grey_image1, "sobel_circle.png");
	//dft_own(grey_image1, "line.png");

	//dft_samples(grey_image2);
	//dft_samples(grey_image1);
	//dft_own(grey_image2, "lena");
	//dft_own(grey_image3, "baby.png");
	// myFFT(grey_image2, 1600);
	//dft_visualize(grey_image2);
	/*************************************************/

	/***************** Lab 6 4.5 ********************/
	// HPF and threshold to sharpen fingerprint1.pgm
	//cv::Mat grey_fingerprint1 = loadGreyImg("D:\\OpenCV\\opencv\\sources\\samples\\data\\fingerprint1.pgm");
	//cv::Mat grey_fingerprint2 = loadGreyImg("D:\\OpenCV\\opencv\\sources\\samples\\data\\fingerprint2.pgm");
	//myFFT(grey_fingerprint, 1600);

	// highpass_filter_sharpen(grey_fingerprint1, 1);
	// highpass_filter_sharpen(grey_fingerprint2, 1);

	//cv::Mat bridge_image = loadGreyImg("D:\\OpenCV\\opencv\\sources\\samples\\data\\bridge.pgm");
	//cv::Mat goldhill_image = loadGreyImg("D:\\OpenCV\\opencv\\sources\\samples\\data\\goldhill.pgm");
	//cv::Mat goldhill_image = loadGreyImg("D:\\OpenCV\\opencv\\sources\\samples\\data\\homo_example.png");
	//homomorphic_enhancement(bridge_image);
	//homomorphic_enhancement(goldhill_image);

	cv::Mat lena = loadGreyImg("D:\\OpenCV\\opencv\\sources\\samples\\data\\lena.bmp");
	
	cv::Mat noise_lena;
	noise_lena = sinusoidal_noise(lena);
	bandreject_filter(noise_lena);

	/*************************************************/

	return EXIT_SUCCESS;
}