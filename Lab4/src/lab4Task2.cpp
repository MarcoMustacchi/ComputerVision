/*
You now need to detect the white markings on the road. How could you tackle this problem?
Some suggestions:
â— consider edge orientation;
â— consider colors close to edge points
*/

#include <opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include <iostream>

cv::Mat im, gray, dst, detected_edges;
const int Threshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";

int main()
{
  im = cv::imread("../images/street_scene.png");
  cv::namedWindow("Image");
  cv::imshow("Image",im);
  cv::waitKey(0);
  dst.create(im.rows,im.cols,im.type());
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::cvtColor(im, im, cv::COLOR_RGBA2RGB, 0);
  cv::Mat img(im.rows,im.cols,im.type());
  cv::bilateralFilter(im, img, 9, 100, 100, cv::BORDER_DEFAULT);
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  
  int mark = 220;
  
  std::vector<cv::Mat> channels(3);
  split(img,channels);
  cv::Mat red = channels[2];
  cv::Mat green = channels[1];
  cv::Mat blue = channels[0];
  cv::Mat im_mask(red.rows,red.cols,red.type());
  for(int i=0;i<red.rows;i++){
    for(int j=0;j<red.cols;j++){
      if((red.at<uchar>(i,j)>=mark)&&(green.at<uchar>(i,j)>=mark)&&(blue.at<uchar>(i,j)>=mark)){
        im_mask.at<uchar>(i,j) = gray.at<uchar>(i,j);
      }
      else{
        im_mask.at<uchar>(i,j) = 0;
      }
    }
  }
 
  cv::Canny(im_mask, detected_edges, Threshold, Threshold*ratio, kernel_size );
  dst = cv::Scalar::all(0);
  img.copyTo(dst, detected_edges);
  imshow( window_name, dst);
  cv::waitKey(0);
  return 0;
}
