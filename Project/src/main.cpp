#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "detection.h"
#include "segmentationCanny.h"
// #include "segmentationOtsu.h"
#include "iou.h"
#include "pixel_accuracy.h"

using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{
    
        
    int mode;
    
    cv::Mat img;
    
    while ( mode != 4)
    {
        
        std::cout << "-------------------------------------" << std::endl;
        std::cout << "Press 0 for Detection" << std::endl;
        std::cout << "Press 1 for Segmentation" << std::endl;
        std::cout << "Press 2 for Performance Detection" << std::endl;
        std::cout << "Press 3 for Performance Segmentation" << std::endl;
        std::cout << "Press 4 for exiting the program" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        std::cin >> mode; 
        
    	switch (mode)
	    {
	
	        case 0:
		    {    
                std::cout << "Detection is starting" << std::endl;
                
                detection(img);
                
                std::cout << "Result" << std::endl;
                cv::namedWindow("Image");
                cv::imshow("Image", img);
                cv::waitKey(10000);  
                
                cv::destroyAllWindows(); 
                
                break;
                
                
            }
            
            case 1:
            {
                std::cout << "Segmentation is starting" << std::endl;
                
                segmentationCanny(img);
                // segmentationOtsu(img);
                
                cv::namedWindow("Image");
                cv::imshow("Image", img);
                cv::waitKey(10000); 
                
                cv::destroyAllWindows();
                
                break;
            }
            
	        case 2:
		    {    
                std::cout << "Performance Detection is starting" << std::endl;
                
                iou(img);
                
                cv::namedWindow("Image");
                cv::imshow("Image", img);
                cv::waitKey(0); 
                
                cv::destroyAllWindows();
                
                break;
            }
            
            case 3:
            {
                std::cout << "Performance Segmentation is starting" << std::endl;

                pixel_accuracy();
                
                cv::namedWindow("Image");
                cv::imshow("Image", img);
                cv::waitKey(0); 
                
                cv::destroyAllWindows();
                
                break;
            }
            
            case 4:
            {
                std::cout << "Exiting the program" << std::endl;
                
                break;
            }
            
            default:
            {
                std::cout << "Wrong number inserted. Retry" << std::endl;
                
                break;
            
            }   
        
        }

    }

	return 0;
  
}
