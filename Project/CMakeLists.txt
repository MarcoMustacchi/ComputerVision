cmake_minimum_required(VERSION 2.8)
project( main )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CMAKE_SOURCE_DIR}/include ) # include the directories where the headers are located                 
add_executable( ${PROJECT_NAME} 
                src/main.cpp
                src/detection.cpp
                src/segmentationMethods.cpp
                src/iou.cpp
                src/pixel_accuracy.cpp
                lib/Detector.cpp
                lib/segmentationAlgorithm.cpp
                lib/Segmentator.cpp
                lib/write_to_file.cpp
                lib/read_sort_BB_matrix.cpp
                lib/removeOutliers.cpp
                lib/fillMaskHoles.cpp
              )  # be sure there's exactly one main() function in the source files
target_link_libraries( ${PROJECT_NAME} PUBLIC 
			${OpenCV_LIBS} 
		      )
