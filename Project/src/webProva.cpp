#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{

    std::string image_number;
    
    std::cout << "Insert image number from 01 to 30" << std::endl;
    std::cin >> image_number; 
	
	//___________________________ Load Dataset image ___________________________ //

	cv::Mat img0 = cv::imread("../Dataset/rgb/" + image_number + ".jpg", cv::IMREAD_COLOR);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img0);
	cv::waitKey(0);
	
    dnn::Net model = dnn::readNetFromONNX("../model/best.onnx");
    
    std::cout << "test" << endl;
    
    Mat blob;    
    dnn::blobFromImage(img0, blob, 1.0f / 255.0f, Size(640, 640), Scalar(), true, false, CV_32F);
    
    model.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(dnn::DNN_TARGET_CPU);
    model.setInput(blob);
    
    std::cout << "test" << endl;
    
    vector<string> outLayerNames = model.getUnconnectedOutLayersNames();
    vector<Mat> result;
    
    std::cout << "test" << endl;
    
    model.forward(result, outLayerNames);
    
    std::cout << "test" << endl;

    Mat out = Mat(result[0].size[1], result[0].size[2], CV_32F, result[0].ptr<float>());
    
    
    vector<Rect> boxes;
    vector<int> indices;
    vector<float> scores;
    
    std::cout << "test" << endl;
    
    for (int r = 0; r < out.size[0]; r++)
    {
        float cx = out.at<float>(r, 0);
        float cy = out.at<float>(r, 1);
        float w = out.at<float>(r, 2);
        float h = out.at<float>(r, 3);
        float sc = out.at<float>(r, 4);
        
        std::cout << "last" << endl;
        Mat confs = out.row(r).colRange(5,85);
        confs*=sc;
        double minV=0,maxV=0;
        double *minI=&minV;
        double *maxI=&maxV;
        minMaxIdx(confs,minI,maxI);
        scores.push_back(maxV);
        
        boxes.push_back(Rect(cx - w / 2, cy - h / 2, w, h));
        indices.push_back(r);
    }
    
    
    std::cout << boxes[3] << std::endl;
    
    dnn::NMSBoxes(boxes, scores, 0.25f, 0.45f, indices);
    
    return 0;
    
}
