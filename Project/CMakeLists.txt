
set(SRC_FILES
    ./src
    
   )
   
set(LIB_FILES
    ./lib
    
   )
   
set(HEADER_FILES
    ./include
    
   )

cmake_minimum_required(VERSION 2.8)
project( main )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} 
                     ${HEADER_FILES}
                   )
add_executable( ${PROJECT_NAME} 
                ${SRC_FILES}/main.cpp
                ${HEADER_FILES}/detection.h
                ${SRC_FILES}/detection.cpp
                ${HEADER_FILES}/segmentationCanny.h
                ${SRC_FILES}/segmentationCanny.cpp
                ${HEADER_FILES}/iou.h
                ${SRC_FILES}/iou.cpp
                ${HEADER_FILES}/pixel_accuracy.h
                ${SRC_FILES}/pixel_accuracy.cpp
                ${HEADER_FILES}/write_to_file.h
                ${LIB_FILES}/write_to_file.cpp
                ${HEADER_FILES}/read_sort_BB_matrix.h
                ${LIB_FILES}/read_sort_BB_matrix.cpp
                ${HEADER_FILES}/removeOutliers.h
                ${LIB_FILES}/removeOutliers.cpp
                ${HEADER_FILES}/fillMaskHoles.h
                ${LIB_FILES}/fillMaskHoles.cpp
                ${HEADER_FILES}/insertMask.h
                ${LIB_FILES}/insertMask.cpp
                ${HEADER_FILES}/randomColorMask.h
                ${LIB_FILES}/randomColorMask.cpp
              )  # be sure there's exactly one main() function in the source files
#target_include_directories( ${PROJECT_NAME} PUBLIC ${HEADER_FILES}/filters.h) # attenzione, viene letto sequenziale, target_include_directories must be defined after the add_executable
target_link_libraries( ${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} )

