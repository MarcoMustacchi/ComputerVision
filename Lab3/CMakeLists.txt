
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
project( Lab3 )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} 
                     ${HEADER_FILES}
                   )
add_executable( ${PROJECT_NAME} 
                ${SRC_FILES}/lab3.cpp
              )  # be sure there's exactly one main() function in the source files
#target_include_directories( ${PROJECT_NAME} PUBLIC ${HEADER_FILES}/filters.h) # attenzione, viene letto sequenziale, target_include_directories must be defined after the add_executable
target_link_libraries( ${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} )



