#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>

void checksNumberChannels(cv::Mat img);

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread("../images/Lena_color.jpg", cv::IMREAD_COLOR); 
    cv::Mat img2 = cv::imread("../images/Lena_color.jpg", cv::IMREAD_GRAYSCALE);
    
    cv::namedWindow("Example 1");
    cv::imshow("Example 1", img);


    cv::namedWindow("Example 2");
    cv::imshow("Example 2", img2);
    char output = cv::waitKey(0);
    
    std::cout<< "Output: " << output << std::endl;
    
    std::cout<< "Number of channels image 1: " << img.channels() << std::endl;
    std::cout<< "Number of channels image 2: " << img2.channels() << std::endl;
    
    cv::destroyAllWindows();
    
    //__________________________ Vertical and Horizontal Gradient __________________________
    
    cv::Mat image1(256, 256, CV_8UC1, cv::Scalar::all(255));
    cv::Mat image2(256, 256, CV_8UC1, cv::Scalar::all(255));
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image1);
    cv::namedWindow("Original Image2");
    cv::imshow("Original Image2", image2);
    cv::waitKey(0);
    
    for (int r = 0; r < image1.rows; r++) {
        for (int c = 0; c < image1.cols; c++) {
            image1.at<uchar>(r,c) = c;
        } 
    }
    
    for (int r = 0; r < image2.rows; r++) {
        for (int c = 0; c < image2.cols; c++) {
            image2.at<uchar>(r,c) = r;
        } 
    }
    
    cv::namedWindow("Post Image1");
    cv::imshow("Post Image1", image1);
    cv::namedWindow("Post Image2");
    cv::imshow("Post Image2", image2);
    cv::waitKey(0);
    
    //__________________________ Chessboard __________________________
    
    int blockSize = 30;
    int imageSize = blockSize * 10; // Warning: works only for even numbers (odd number each new line start 
                                   // with the same color of the previous one and so we have black and white stripes
    // BGR
    cv::Mat chessBoard(imageSize, imageSize, CV_8UC3, cv::Scalar::all(0)); // in this way image of size we want
    std::cout << "Number of channels: " << chessBoard.channels() << std::endl;
    unsigned char color = 0;

    for(int i=0; i<imageSize; i=i+blockSize) {
        color=~color;
        for(int j=0; j<imageSize; j=j+blockSize) { // cv::Rect(x,y) is using (x,y) as (column,row) while in general image.at(i,j) is using (i,j) as (row,column)
            cv::Mat ROI = chessBoard(cv::Rect(i, j, blockSize, blockSize)); // overloaded function, getting Region Of Interest (ROI)
            ROI.setTo(cv::Scalar::all(color));
            color=~color;
        }
    }
    cv::imshow("Chess board RGB", chessBoard);
    cv::waitKey(0);
    
    // Grayscale
    cv::Mat chessBoard1(imageSize, imageSize, CV_8UC1, cv::Scalar::all(0)); 
    std::cout << "Number of channels: " << chessBoard1.channels() << std::endl;

    for(int i=0; i<imageSize; i=i+blockSize) {
        color=~color;
        for(int j=0; j<imageSize; j=j+blockSize) {
            cv::Mat ROI1 = chessBoard1(cv::Rect(i, j, blockSize, blockSize)); // overloaded function, getting Region Of Interest (ROI)
            ROI1.setTo(cv::Scalar::all(color));
            color=~color;
        }
    }
    cv::imshow("Chess board one channel", chessBoard1);
    cv::waitKey(0);
    
    cv::destroyAllWindows();
    
    //__________________________ Function call __________________________
    
    cv::namedWindow("Image in main");
    cv::imshow("Image in main", img);
    cv::waitKey(0); 
    checksNumberChannels(img); 
    cv::namedWindow("Image in main post function");
    cv::imshow("Image in main post function", img);
    cv::waitKey(0); 
    
    //__________________________ Rotating transformation and Affine transformation __________________________
    
    cv::Mat gurus = cv::imread("../images/DL_gurus.jpg", cv::IMREAD_COLOR); 
    
    cv::Point2f srcTri[3];
    srcTri[0] = cv::Point2f( 0.f, 0.f );
    srcTri[1] = cv::Point2f( gurus.cols - 1.f, 0.f );
    srcTri[2] = cv::Point2f( 0.f, gurus.rows - 1.f );
    cv::Point2f dstTri[3];
    dstTri[0] = cv::Point2f( 0.f, gurus.rows*0.33f );
    dstTri[1] = cv::Point2f( gurus.cols*0.85f, gurus.rows*0.25f );
    dstTri[2] = cv::Point2f( gurus.cols*0.15f, gurus.rows*0.7f );
    cv::Mat warp_Mat = cv::getAffineTransform( srcTri, dstTri );
    cv::Mat warp_dst = cv::Mat::zeros( gurus.rows, gurus.cols, gurus.type() );
    cv::warpAffine( gurus, warp_dst, warp_Mat, warp_dst.size() );
    
    cv::Point center = cv::Point( gurus.cols/2, gurus.rows/2 );
    double angle = -50.0;
    double scale = 1;
    cv::Mat rot_Mat = cv::getRotationMatrix2D( center, angle, scale );
    cv::Mat rot_dst = cv::Mat::zeros( gurus.rows, gurus.cols, gurus.type() );
    warpAffine( gurus, rot_dst, rot_Mat, rot_dst.size() );
    
        
    //__________________________ Multiple images on the same window __________________________ 
    int width = 3*gurus.cols; // width of 2 images next to each other
    int height = gurus.rows; // height of 2 images over reach other

    cv::Mat inputAll = cv::Mat(height, width, gurus.type());

    cv::Rect subImageROI = cv::Rect(0, 0, gurus.cols, gurus.rows);

    // copy to subimage:
    gurus.copyTo(inputAll(subImageROI));

    // move to 2nd image ROI position:
    subImageROI.x = gurus.cols;
    warp_dst.copyTo(inputAll(subImageROI));

    subImageROI.x = 2*gurus.cols;
    rot_dst.copyTo(inputAll(subImageROI));

    cv::imshow("Trasformations", inputAll);
    
    cv::waitKey(0);
        
    return 0;
}

void checksNumberChannels(cv::Mat image) {
    
    if (image.channels() == 3) {
        // "channels" is a vector of 3 Mat arrays:
        std::vector<cv::Mat> channels(3);
        cv::split(image, channels);  // split img
        // get the channels (follow BGR order in OpenCV)
        channels[0] = 0;        
        cv::merge(channels, image);
        // image.setTo(cv::Scalar(0, 255, 255));
    }  
    cv::namedWindow("Image in function");
    cv::imshow("Image in function", image);
    cv::waitKey(0);
 
}
