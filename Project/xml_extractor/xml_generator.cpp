#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include <filesystem>
#include<fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iterator>
namespace fs = std::filesystem;
using namespace std;
using namespace cv;

int main() {
	//path file txt with bounded box with the format : x-y-width-height
	std::string path_square = "hand/bounded";
	//path samples
	std::string path_img = "hand/pos_samples";
	//path where save the xml
	std::string xml_annotation = "hand/xml";
	ifstream myReadFile;
	ofstream WriteFile;
	cout<<"Initialize creation xml"<<"\n";
	for (const auto & entry : fs::directory_iterator(path_img)){
		//text of xml
		std::string xml_file = "";
		std::string img = entry.path();
		xml_file.append("<annotation>\n\t<folder>image</folder>\n\t<filename>"+img.substr(path_img.length()+1, img.length())+"</filename>\n\t<path>...</path>\n\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n");
		Mat new_img = imread(img);
		xml_file.append("\t<size>\n\t\t<width>"+to_string(new_img.cols)+"</width>\n\t\t<height>"+to_string(new_img.rows)+"</height>\n\t\t<depth>"+to_string(new_img.channels())+"</depth>\n\t</size>\n\t<segmented>0</segmented>\n");

		std::string xml_file_path = entry.path();
		std::string normalization = entry.path();
		img.replace(0,path_img.length(),path_square);
		img.replace(img.length()-3,img.length(),"txt");
		//read bounded box of the image from the file
		myReadFile.open(img);
		std::string line;
		if (myReadFile.is_open()) {
			while (getline(myReadFile , line)) {
				if(line!=""){
				std::stringstream ss(line);
				std::istream_iterator<std::string> begin(ss);
				std::istream_iterator<std::string> end;
				std::vector<std::string> vstrings(begin, end);
				std::string x = vstrings[0];
				std::string y = vstrings[1];
				std::string w = vstrings[2];
				std::string h = vstrings[3];
				int x_max = stoi(x) + stoi(w);
				int y_max = stoi(y) + stoi(h);
				xml_file.append("\t<object>\n\t\t<name>hand</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n\t\t\t<xmin>"+x+"</xmin>\n\t\t\t<ymin>"+y+"</ymin>\n\t\t\t<xmax>"+std::to_string(x_max)+"</xmax>\n\t\t\t<ymax>"+std::to_string(y_max)+"</ymax>\n\t\t</bndbox>\n\t</object>\n");
				}
			}	
		}	

		xml_file.append("</annotation>");
		//save file xml
		xml_file_path.replace(0,path_img.length(),xml_annotation);
		xml_file_path.replace(xml_file_path.length()-3,xml_file_path.length(),"xml");
		WriteFile.open(xml_file_path);
		WriteFile << xml_file;
		WriteFile.close();
		myReadFile.close();
	}
	cout<<"Creation Complete"<<"\n";
}
