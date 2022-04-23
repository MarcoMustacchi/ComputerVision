
set(SRC_FILES
    ./src
    
   )
   
set(LIB_FILES
    ./lib
    
   )
   
set(HEADER_FILES
    ./include
    
   )

cmake_minimum_required( VERSION 2.8 )
project( Lab1 )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( ${PROJECT_NAME} 
                ${SRC_FILES}/lab1.cpp ) # be sure there's exactly one main() function in the source files
target_link_libraries( ${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} )


