#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cmath>
#include <string>

    
    //cv::Mat imgBlurred; 
    //cv::Mat imgCanny;  // definita come variabile globale 
	//int lowThreshold = 0;
	const int max_lowThreshold = 100;
	int kernel_size = 3;
	int highThreshold = 101;
	const int max_highThreshold = 200;
	
// _____________________________________________ Classes _____________________________________________//	
// Create a class to make all variables pass in callback function instead of global variables!
class MyClassCanny {       

  public:             
    int lowThresholdClass;        
    cv::Mat imgBlurredClass; 
    cv::Mat imgCannyClass; 
    
    MyClassCanny(int lowThreshold, cv::Mat imgBlurred, cv::Mat imgCanny)     // constructor
    {
        lowThresholdClass = lowThreshold;
        imgBlurredClass = imgBlurred;
        imgCannyClass = imgCanny;
    }
};

class MyClassHough {       

  public:             
    int rhoClass, thetaClass;
    int threshClass;
    cv::Mat imgCannyClass;
    std::vector<cv::Vec2f> linesClass;
    cv::Mat imgOriginalClass;
    
    MyClassHough(int rho, int theta, int thresh, cv::Mat imgCanny, std::vector<cv::Vec2f> lines, cv::Mat imgOriginal)     // constructor
    {
        rhoClass = rho;
        thetaClass = theta;
        threshClass = thresh;
        imgCannyClass = imgCanny;
        linesClass = lines;
        imgOriginalClass = imgOriginal;
    }
};


//_____________________________________________ Functions _____________________________________________//
// callback function needs to be static?
static void MyCallbackForThreshold(int, void* param)
{

    MyClassCanny test = *(MyClassCanny *)param;
    
    // reference su high o low threshold, quindi solo quello viene modificato
    cv::Canny(test.imgBlurredClass,            // 8-bit input image
        test.imgCannyClass,                    // output edge map; single channels 8-bit image, which has the same size as image 
        test.lowThresholdClass,                // low threshold for the hysteresis procedure
        highThreshold,				// high threshold for the hysteresis procedure
		kernel_size,				// aperture size for the Sobel operator
		false);                    // a flag, indicating whether a more accurate L2 norm should be used    
    
    // cv::namedWindow("imgCanny", cv::WINDOW_AUTOSIZE);
	cv::imshow("Edge Map", test.imgCannyClass);  
}


static void MyCallbackForHoughLines(int, void* param)
{

    MyClassHough test = *(MyClassHough *)param;
    cv::Mat hough = test.imgCannyClass;
    
    // attenzione, se uno dei parametri rho o theta viene portato a 0 nella trackbar --> Erroreee
    cv::HoughLines(test.imgCannyClass,  // Output of the edge detector
        test.linesClass,				   // A vector that will store the parameters (r,theta) of the detected lines
        test.rhoClass,                	// The resolution parameter \rho in pixels.
        test.thetaClass,
        // CV_PI/180,			// The resolution of the parameter \theta in radians.
		test.threshClass,				// The minimum number of intersecting points to detect a line.
		0,					// 
		0);					//    
		
	
	// Draw lines	
    for( size_t i = 0; i < test.linesClass.size(); i++ )
    {
        float rho = test.linesClass[i][0], theta = test.linesClass[i][1];
        cv::Point pt1, pt2;
        double x0 = rho * cos(theta), y0 = rho * sin(theta);
        pt1.x = cvRound(x0 + 1000*(-sin(theta)));
        pt1.y = cvRound(y0 + 1000*(cos(theta)));
        pt2.x = cvRound(x0 - 1000*(-sin(theta)));
        pt2.y = cvRound(y0 - 1000*(cos(theta)));
        cv::line( test.imgOriginalClass, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
    } 
    
    /*
    // reference su high o low threshold, quindi solo quello viene modificato
    cv::Canny(test.imgBlurredClass,            // 8-bit input image
        test.imgCannyClass,                    // output edge map; single channels 8-bit image, which has the same size as image 
        test.lowThreshold,                // low threshold for the hysteresis procedure
        highThreshold,				// high threshold for the hysteresis procedure
		kernel_size,				// aperture size for the Sobel operator
		false);                    // a flag, indicating whether a more accurate L2 norm should be used    
    */
    
    // cv::namedWindow("imgCanny", cv::WINDOW_AUTOSIZE);
	cv::imshow("Hough Lines", test.imgOriginalClass);  
	std::cout << "lines size: " << test.linesClass.size() << std::endl;
}


//_____________________________________________ Main _____________________________________________//
int main( int argc, char** argv )
{

    cv::Mat imgOriginal;        
    imgOriginal = cv::imread("../images/street_scene.png", cv::IMREAD_COLOR);      
            
    if (imgOriginal.empty()) {                                    
        std::cout << "error: image not read from file\n\n";       
        return(0);                                               
    }
	
	cv::Mat imgGrayscale;
    cv::cvtColor(imgOriginal, imgGrayscale, cv::COLOR_BGR2GRAY);   
    
	// difference between blur() and GaussianBlur()
	// why do we need to smooth image? doesn't the canny algorithm already do that?
	
	cv::Mat imgBlurred;
	    
    cv::GaussianBlur(imgGrayscale,            // input image
        imgBlurred,                           // output image
        cv::Size(5, 5),                      // smoothing window width and height in pixels
        1.5);                                // sigma value, determines how much the image will be blurred		
        
    cv::namedWindow("Gaussian Filter", cv::WINDOW_AUTOSIZE); 
    cv::imshow("Gaussian Filter", imgBlurred); 
    cv::waitKey(0);     
    
    cv::bilateralFilter(imgGrayscale, imgBlurred, 9, 75, 75, cv::BORDER_DEFAULT);
    
    cv::namedWindow("Bilateral Filter", cv::WINDOW_AUTOSIZE); 
    cv::imshow("Bilateral Filter", imgBlurred); 
    cv::waitKey(0);     
        
    cv::Mat imgCanny;
    imgCanny.create( imgBlurred.size(), imgBlurred.type() );
    
    MyClassCanny CannyClass(100, imgBlurred, imgCanny);
    
    std::cout << "threshold is: " << CannyClass.lowThresholdClass << std::endl;    					
	
	//_____________________________________________ Important: trackbars need to call a Callback function _____________________________________________
	//Create track bar to change lowThreshold
	
	cv::namedWindow("imgOriginal", cv::WINDOW_AUTOSIZE); 
    cv::imshow("imgOriginal", imgOriginal); 
	
	cv::namedWindow( "Edge Map", cv::WINDOW_AUTOSIZE );
	cv::createTrackbar( "Low Threshold:", "Edge Map", &CannyClass.lowThresholdClass, max_lowThreshold, MyCallbackForThreshold, &CannyClass ); // puo' avere anche il parametro void* userdata per callback
	MyCallbackForThreshold(0, &CannyClass);
	
	
    //Create track bar to change maxThreshold
    cv::createTrackbar( "High Threshold:", "Edge Map", &highThreshold, max_highThreshold, MyCallbackForThreshold, &CannyClass );
    // MyCallbackForThreshold(0, 0);  
    
    cv::waitKey(0);     
	
	
	//_____________________________________________ Hough Transform Lines _____________________________________________
	
    // Copy edges to the images that will display the results in BGR
    // cv::Mat colorDst;
    // cv::cvtColor(imgCanny, colorDst, cv::COLOR_GRAY2BGR);
    
    cv::namedWindow("Detected Lines (in red) - Standard Hough Line Transform", cv::WINDOW_AUTOSIZE);
	cv::imshow("Detected Lines (in red) - Standard Hough Line Transform", CannyClass.imgCannyClass);
	cv::waitKey(0); 
    
    // Standard Hough Line Transform
    std::vector<cv::Vec2f> lines; // will hold the results of the detection
    
    int rho = 1;
    int theta = 1;
    // int theta = CV_PI/180;
    int thresh = 100;
    
    MyClassHough HoughClass(rho, theta, thresh, CannyClass.imgCannyClass, lines, imgOriginal);
    
    cv::namedWindow( "Hough Lines", cv::WINDOW_AUTOSIZE );
    cv::createTrackbar( "rho:", "Hough Lines", &HoughClass.rhoClass, max_highThreshold, MyCallbackForHoughLines, &HoughClass );
    MyCallbackForHoughLines(0, &HoughClass);
    
    cv::createTrackbar( "theta:", "Hough Lines", &HoughClass.thetaClass, max_highThreshold, MyCallbackForHoughLines, &HoughClass );
    cv::createTrackbar( "threshold:", "Hough Lines", &HoughClass.threshClass, max_highThreshold, MyCallbackForHoughLines, &HoughClass );
    
    cv::waitKey(0); 
    /*
    // HoughTransform orders lines descending by number of votes!!
    cv::HoughLines(mmm.imgCannyClass,  // Output of the edge detector
        lines,				   // A vector that will store the parameters (r,theta) of the detected lines
        1,                	// The resolution parameter \rho in pixels.
        CV_PI/180,			// The resolution of the parameter \theta in radians.
		thresh,				// The minimum number of intersecting points to detect a line.
		0,					// 
		0);					// 
    */
    
    std::cout << "Size of lines: " << HoughClass.linesClass.size() << std::endl;
    std::cout << "First Strongest line: " << HoughClass.linesClass.at(1) << std::endl;
    std::cout << "Second Strongest line:: " << HoughClass.linesClass.at(1) << std::endl;
    
    /*
    double rho1 = lines[0][0], rho2 = lines[1][0];
    double theta1 = lines[0][1], theta2 = lines[1][1];
    
    double x0 = rho1 * cos(theta1), y0 = rho1 * sin(theta1);
    double x1 = rho2 * cos(theta2), y1 = rho2 * sin(theta2);
    
    /*
    cv::Point pt1, pt2;
    pt1.x = cvRound(x0 + 1000*(- sin(theta1) ));
    pt1.y = cvRound(y0 + 1000*( cos(theta1) ));
    pt2.x = cvRound(x0 - 1000*(- sin(theta1) ));
    pt2.y = cvRound(y0 - 1000*( cos(theta1) ));
    
    // Draw the lines
    line( imgOriginal, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
    */
    
    /*
    double m = - (cos(theta1)/sin(theta1));  // cotg
    double c = rho1 * (1 / sin(theta1));  // cosecante
    
    // y = (-cotθ)*x + (p*cosecθ)
    // y = m*x + c
    // Calculate where the line crosses the x and y axes and then use those as your points. To my memory, the line endpoints need not be on the image.
    cv::Point pt3, pt4;
    pt3.x = 0;
    pt3.y = c;
    pt4.x = -(c/m);
    pt4.y = 0;
    
    line( imgOriginal, pt3, pt4, cv::Scalar(255,0,0), 3, cv::LINE_AA);
    
    cv::namedWindow("Detected Lines (in red) - Standard Hough Line Transform", cv::WINDOW_AUTOSIZE);
	cv::imshow("Detected Lines (in red) - Standard Hough Line Transform", imgOriginal);
    cv::waitKey(0); 
    */
    
    /*
    double m2 = - (cos(theta2)/sin(theta2));  // cotg
    double c2 = rho2 * (1 / sin(theta2));  // cosecante
    
    // y = (-cotθ)*x + (p*cosecθ)
    // y = m*x + c
    // Calculate where the line crosses the x and y axes and then use those as your points. To my memory, the line endpoints need not be on the image.
    cv::Point pt5, pt6;
    pt5.x = 0;
    pt5.y = c2;
    pt6.x = -(c2/m2);
    pt6.y = 0;
    
    line( imgOriginal, pt5, pt6, cv::Scalar(255,0,0), 3, cv::LINE_AA);
    */
    
    /*
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double x0 = rho * cos(theta), y0 = rho * sin(theta);
        pt1.x = cvRound(x0 + 1000*(-sin(theta)));
        pt1.y = cvRound(y0 + 1000*(cos(theta)));
        pt2.x = cvRound(x0 - 1000*(-sin(theta)));
        pt2.y = cvRound(y0 - 1000*(cos(theta)));
        cv::line( imgOriginal, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
    }
    
	cv::namedWindow("Detected Lines (in red) - Standard Hough Line Transform", cv::WINDOW_AUTOSIZE);
	cv::imshow("Detected Lines (in red) - Standard Hough Line Transform", imgOriginal);
	cv::waitKey(0); 
	
	//_____________________________________________ Hough Transform Circles _____________________________________________
	cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
	cv::imshow("Test", CannyClass.imgCannyClass);
	cv::waitKey(0); 
	
	// Create a vector for detected circles
	std::vector<cv::Vec3f>  circles;
	
	// Apply Hough Transform
	HoughCircles(mmm.imgCannyClass, 
		circles, 		// Each vector is encoded as 3 or 4 element floating-point vector (x,y,radius) or (x,y,radius,votes)
		cv::HOUGH_GRADIENT,  // Detection method
		1, 
		mmm.imgCannyClass.rows/64,  // Minimum distance between the centers of the detected circles.
		200, 
		10, 
		5,  // Minimum circle radius.
		30);  // Maximum circle radius.
	
	// Draw detected circles
	for(size_t i=0; i<circles.size(); i++) {
	    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	    int radius = cvRound(circles[i][2]);
	    circle(imgOriginal, center, radius, cv::Scalar(0, 255, 0), 2, 8, 0);
	}
	
	cv::namedWindow("Detected Circles", cv::WINDOW_AUTOSIZE);
	cv::imshow("Detected Circles", imgOriginal);
	cv::waitKey(0); 
	*/
	
    return 0;
    
}
