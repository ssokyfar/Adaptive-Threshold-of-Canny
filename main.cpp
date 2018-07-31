#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
using namespace std;
using namespace cv;

//***********************************************************************************
//              Method 3 (By compute the Maximum gradient variance)
//***********************************************************************************
void AdaptiveFindThreshold(const Mat src, double& low, double& high, int aperture_size = 3)
{
	const int cn = src.channels();
	cv::Mat dx_img(src.rows, src.cols, CV_16SC(cn));
	cv::Mat dy_img(src.rows, src.cols, CV_16SC(cn));

	cv::Sobel(src, dx_img, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(src, dy_img, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_DEFAULT);

	const int width = dx_img.cols;
	const int height = dx_img.rows;

	// 计算边缘的强度, 并存于图像中
	float maxv = 0;
	Mat img_dxy = Mat(height, width, CV_32FC1);
	for (int i = 0; i < height; i++)
	{
		const short* _dx = (short*)(dx_img.data + dx_img.step*i);
		const short* _dy = (short*)(dy_img.data + dy_img.step*i);
		float* _image = (float *)(img_dxy.data + img_dxy.step*i);
		for (int j = 0; j < width; j++)
		{
			_image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
			maxv = maxv < _image[j] ? _image[j] : maxv;
			
			//if(i<height*0.25 && j<width*0.25)std::cout << _image[j] << ",";	
		}
		//if (i<height*0.25 )std::cout << std::endl;
	}
	//std::cout << std::endl << std::endl;
	//std::cout << img_dxy << std::endl;

	std::cout << std::endl << std::endl;
	std::cout << "maxv: " << maxv << std::endl;

	// 计算直方图
	int hist_size = maxv;// (int)(hist_size > maxv ? maxv : hist_size);//255
	//定义一个变量用来存储 单个维度 的数值的取值范围    
	float midRanges[] = { 0, maxv };//---------------------------从cvCalcHist换成calcHist函数，要记得改这个范围（不是0~256！！！）
	//确定每个维度的取值范围，就是横坐标的总数    
	const float *ranges[] = { midRanges };
	double  PercentOfPixelsNotEdges = 0.7;
	//range[1] = maxv;

	Mat hist_dst;
	bool uniform = true;
	bool accumulate = false;
	//需要计算图像的哪个通道（bgr空间需要确定计算 b或g或r空间）    
	const int channels[1] = { 0 };
	calcHist(&img_dxy, 1, channels, Mat(), hist_dst, 1, &hist_size, ranges, uniform, accumulate);

	int total = (int)(height * width * PercentOfPixelsNotEdges);
	float sum = 0;

	//std::cout << hist_dst.cols << ",,,,,,,,," << hist_dst.rows << std::endl;//1,,,,,,1156
	//std::cout << "hist_size" << hist_size << std::endl;//1,,,,,,255
	//std::cout << "hist_dst" << hist_dst << std::endl;


	//寻找最大梯度值
	float Hmax = 0.0f;
	float HmaxNum = 0.0f;
	float *h = (float*)hist_dst.data;
	for (int i=0; i < hist_size; i++)
	{
		if (h[i] > HmaxNum)
		{
			HmaxNum = h[i];
			Hmax = i;
		}
		//std::cout << i << ":" << h[i] << " ";
		//if ((float)hist_dst.data[i] > HmaxNum)
		//{
		//	HmaxNum = (float)hist_dst.data[i];
		//	Hmax = i;
		//}
	}
	std::cout << "最大梯度值，数目：" << Hmax << "," << HmaxNum << std::endl;

	//计算像素最值梯度方差Emax
	float Emax = 0.0f;
	h = (float*)hist_dst.data;
	for (int i = 0; i < hist_size; i++)
	{
		float N = h[i];
		if (N > 0) {
			float temp = (i - Hmax)*(i - Hmax) / N;
			temp = temp > 1.0f ? 1.0f : temp;
			Emax += temp;
			//std::cout << temp << " ";
		}
	}
	std::cout << "像素最值梯度方差：" << Emax << std::endl << std::endl;
	high = Hmax + Emax;

	// 计算高低门限
	//high = (i + 1) * maxv / hist_size;
	low = high * 0.4;

}

int main(int argc, char* argv[])
{
	Mat input = imread("test.jpg"); 
	Mat output(input.rows, input.cols, input.type(), Scalar(0, 0, 255));
	Mat output1 = output.clone();
	Mat output2 = output.clone();
	Mat output3 = output.clone();

	//-------------------------use color image
	std::cout << "//--------【use color image】-------------//" << std::endl;
	Mat out;
	double low = 50.0;
	double high = 200.0;
	Canny(input, out, low, high);
	cout << "Canny:" << endl;
	std::cout << "low threshold:" << low << std::endl << "high threshold:" << high << std::endl;
	imshow("input", input);
	imshow("canny", out);

	out.convertTo(out, CV_8UC1, -1.0, 1);
	input.copyTo(output, out);	
	imshow("canny mix", output);

	AdaptiveFindThreshold(input, low, high);
	cout << "Canny SelfAdaption:" << endl;
	std::cout << "low threshold:" << low << std::endl << "high threshold:" << high << std::endl;
	Canny(input, out, low, high);
	imshow("canny_selfAdaption", out);

	out.convertTo(out, CV_8UC1, -1.0, 1);
	input.copyTo(output1, out);	
	imshow("canny_selfAdaption mix", output1);

	//-------------------------use gray image
	std::cout << "//--------【use gray image】-------------//" << std::endl;
	low = 50.0;
	high = 150.0;
	Mat grayImg;	
	cvtColor(input, grayImg, CV_BGR2GRAY);
	Mat gaussian_image = grayImg;
	//GaussianBlur(grayImg, gaussian_image, Size(5, 5), 2, 2);

	Canny(gaussian_image, out, low, high);
	cout << "Canny:" << endl;
	std::cout << "low threshold:" << low << std::endl << "high threshold:" << high << std::endl;
	imshow("gray input", gaussian_image);
	imshow("gray canny", out);

	out.convertTo(out, CV_8UC1, -1.0, 1);
	input.copyTo(output2, out);	
	imshow("gray canny mix", output2);

	AdaptiveFindThreshold(gaussian_image, low, high);
	cout << "Canny SelfAdaption:" << endl;
	std::cout << "low threshold:" << low << std::endl << "high threshold:" << high << std::endl;
	Canny(gaussian_image, out, low, high);

	imshow("gray canny_selfAdaption", out);

	out.convertTo(out, CV_8UC1, -1.0, 1);
	input.copyTo(output3, out);	
	imshow("gray canny_selfAdaption mix", output3);

	waitKey(0);
	return 0;
}



//***********************************************************************************
//                        Method 1 (use OTSU)
//***********************************************************************************
//void cannyEdgeDetect(Mat& img, Mat& output) {
//
//	Mat gaussian_image, edges, thresholded;
//	double lowthreshold;
//	double highthreshold;
//	int kernel_size = 3;
//
//	imshow("input image", img);
//
//	// -- convert to gray scale
//	Mat gray_image;
//	cvtColor(img, gray_image, CV_BGR2GRAY);
//
//	// --automatic threshold value detection using OTSU
//	highthreshold = threshold(gray_image, thresholded, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
//	lowthreshold = highthreshold*0.4;
//
//	// -- blur image with 3*3 filter and reduce noise
//	GaussianBlur(gray_image, gaussian_image, Size(5, 5), 2, 2);// blur(gray_image, gaussian_image, Size(3, 3));
//
//	// canny detector
//	Canny(gaussian_image, edges, lowthreshold, highthreshold, kernel_size);
//	imshow("canny_result", edges);
//	std::cout << "low threshold:" << lowthreshold << std::endl << "high threshold:" << highthreshold << std::endl;
//
//	edges.convertTo(edges, CV_8UC1, -1.0, 1);
//	img.copyTo(output, edges);
//	imshow("canny_input", output);
//	waitKey(0);
//}
//
//int main(int argc, char** argv) {
//	Mat img = imread("test.jpg");
//
//	// -- if image not found then show error message
//	if (!img.data) {
//		std::cout << "image not found" << std::endl;
//		return -1;
//	}
//	Mat output(img.cols, img.rows,img.type());
//	cannyEdgeDetect(img, output);
//
//	return 0;
//
//}



//***********************************************************************************
//                            Method 2 (By compute gradient histogram)
//***********************************************************************************
// 仿照matlab，自适应求高低两个门限                                            
//void _AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double *low, double *high)
//{
//	CvSize size;
//	IplImage *imge = 0;
//	int i, j;
//	CvHistogram *hist;
//	int hist_size = 255;
//	float range_0[] = { 0,256 };
//	float* ranges[] = { range_0 };
//	double PercentOfPixelsNotEdges = 0.95;
//	size = cvGetSize(dx);
//	imge = cvCreateImage(size, IPL_DEPTH_32F, 1);
//	// 计算边缘的强度, 并存于图像中                                        
//	float maxv = 0;
//	for (i = 0; i < size.height; i++)
//	{
//		const short* _dx = (short*)(dx->data.ptr + dx->step*i);
//		const short* _dy = (short*)(dy->data.ptr + dy->step*i);
//		float* _image = (float *)(imge->imageData + imge->widthStep*i);
//		for (j = 0; j < size.width; j++)
//		{
//			_image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
//			//if(i<size.height*0.25 && j<size.width*0.25)std::cout << _image[j] << ",";
//			maxv = maxv < _image[j] ? _image[j] : maxv;
//
//		}
//		//if (i<size.height*0.25 )std::cout << std::endl;
//	}
//	//std::cout << std::endl << std::endl;
//	if (maxv == 0) {
//		*high = 0;
//		*low = 0;
//		cvReleaseImage(&imge);
//		return;
//	}
//
//	// 计算直方图    
//	std::cout << "maxv" << maxv << std::endl;
//	range_0[1] = maxv;
//	hist_size = maxv;// (int)(hist_size > maxv ? maxv : hist_size);
//	hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
//	cvCalcHist(&imge, hist, 0, NULL);
//	int total = (int)(size.height * size.width * PercentOfPixelsNotEdges);
//	float sum = 0;
//	int icount = hist->mat.dim[0].size;
//	
//	//寻找最大梯度值
//	float Hmax=0.0f;
//	float HmaxNum = 0.0f;
//	float *h = (float*)cvPtr1D(hist->bins, 0);
//	for (i = 0; i < icount; i++)
//	{
//		//std::cout<<i<<":" << h[i] << " ";
//		sum += h[i];
//		if (sum > total)
//			break;
//	}
//	std::cout << "最大梯度值，数目：" << Hmax<<"," <<HmaxNum<< std::endl;
//
//
//	// 计算高低门限                                                        
//	*high = (i + 1) * maxv / hist_size;
//	*low = *high * 0.4;
//	cvReleaseImage(&imge);
//	cvReleaseHist(&hist);
//}
//void AdaptiveFindThreshold(const CvArr* image, double *low, double *high, int aperture_size = 3)
//{
//	cv::Mat src = cv::cvarrToMat(image);
//	const int cn = src.channels();
//	cv::Mat dx(src.rows, src.cols, CV_16SC(cn));
//	cv::Mat dy(src.rows, src.cols, CV_16SC(cn));
//
//	cv::Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_DEFAULT);
//	cv::Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_DEFAULT);//BORDER_REPLIC
//
//	CvMat _dx = dx, _dy = dy;
//	_AdaptiveFindThreshold(&_dx, &_dy, low, high);
//
//}
//int main(int argc, char* argv[])
//{
//	IplImage* pImg;// = cvLoadImage("iphone_1.png");
//	IplImage* pCannyImg = NULL;
//
//	double low_thresh = 50.0;
//	double high_thresh = 200.0;
//
//	if ((pImg = cvLoadImage("iphone_2.png")) != 0)
//	{
//		pCannyImg = cvCreateImage(cvGetSize(pImg), IPL_DEPTH_8U, 1);
//
//		cout << "Canny:" << endl;
//		cout << "low_thresh:  " << low_thresh << endl;
//		cout << "high_thresh: " << high_thresh << endl << endl;
//		cvCanny(pImg, pCannyImg, low_thresh, high_thresh, 3);
//
//		cvNamedWindow("src", 1);
//		cvNamedWindow("canny", 1);
//		cvShowImage("src", pImg);
//		cvShowImage("canny", pCannyImg);
//
//
//		CvMat *dx = (CvMat*)pImg;
//		CvMat *dy = (CvMat*)pCannyImg;
//		AdaptiveFindThreshold(pImg, &low_thresh, &high_thresh);
//		cvCanny(pImg, pCannyImg, low_thresh, high_thresh, 3);
//
//		cout << "Canny SelfAdaption:" << endl;
//		cout << "low_thresh:  " << low_thresh << endl;
//		cout << "high_thresh: " << high_thresh << endl;
//
//		cvNamedWindow("canny_selfAdaption", 1);
//		cvShowImage("canny_selfAdaption", pCannyImg);
//
//		cvWaitKey(0);
//
//		cvDestroyWindow("src0");
//		cvDestroyWindow("canny0");
//		cvDestroyWindow("src1");
//		cvDestroyWindow("canny1");
//
//		cvReleaseImage(&pImg);
//		cvReleaseImage(&pCannyImg);
//	}
//	return 0;
//}