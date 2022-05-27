#include <memory>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>


#ifndef PanoramicUtils_H
#define PanoramicUtils_H


class PanoramicUtils
{
public:
	static cv::Mat cylindricalProj(
		const cv::Mat& image,
		const double angle);

};
#endif
