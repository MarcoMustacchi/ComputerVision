#include "filters.h" //declaration of this two functions
#include <opencv2/highgui.hpp>


cv::Mat minFilter(cv::Mat& img, int sizeKernel)
{	
	if ((sizeKernel % 2) == 0)
		return img;

	cv::Mat returnImg(img.rows, img.cols, img.type());
	int a = sizeKernel / 2;

	for (int r = 0; r < img.rows; r++)
		for (int c = 0; c < img.cols; c++)
		{
			unsigned char min = 255;
			for (int i = -a; i <= a; i++)
			{
				for (int j = -a; j <= a; j++)
				{
					if ((r + i) >= 0 && (c + j) >= 0 && (r + i) < img.rows && (c + j) < img.cols)
						if (img.at<unsigned char>(r + i, c + j) < min)
							min = img.at<unsigned char>(r + i, c + j);
				}
			}
			returnImg.at<unsigned char>(r, c) = min;
		}

	return returnImg;
}

cv::Mat maxFilter(cv::Mat& img, int sizeKernel)
{
	if ((sizeKernel % 2) == 0)
		return img;
	
	cv::Mat returnImg(img.rows, img.cols, img.type());
	int a = sizeKernel / 2;
	
	for (int r = 0; r < img.rows; r++)	
		for (int c = 0; c < img.cols; c++)
		{
			unsigned char max = 0;			
			for (int i = -a; i <= a; i++)
			{
				for (int j = -a; j <= a; j++)
					if ( (r+i) >= 0 && (c+j) >= 0 && (r+i) < img.rows && (c+j) < img.cols)					
						if (img.at<unsigned char>(r + i, c + j) > max)
							max = img.at<unsigned char>(r + i, c + j);				
			}
			returnImg.at<unsigned char>(r, c) = max;
		}
	
	return returnImg;
}


