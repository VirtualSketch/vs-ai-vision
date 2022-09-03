cmake_minimum_required(VERSION 3.9) # set CMAKE version
set( CMAKE_CXX_STANDARD 11 ) # set C++ standard to c11
project( vs-ai-vision ) # name of the project
find_package( OpenCV REQUIRED ) # find library in system
include_directories( ${OpenCV_INCLUDE_DIRS} ) # include all library dirs
# add_executable( svmtrain_exec svmtrain.cpp ) # executable
# target_link_libraries( svmtrain_exec ${OpenCV_LIBS} ) # link OpenCV libraries
# add_executable( svmtest_exec svmtest.cpp ) # executable
# target_link_libraries( svmtest_exec ${OpenCV_LIBS} ) # link OpenCV libraries

add_library(
    svmutils
    SHARED
    utils/coordinatedMat.hpp
    utils/svm.cpp
    utils/labelassigner.cpp
    utils/descriptordeterminer.cpp
    utils/rectangledeterminer.cpp
    utils/imagepreprocessing.cpp
)

target_link_libraries(svmutils ${OpenCV_LIBS})

add_library(
    train
    SHARED
    train.cpp
)

target_link_libraries(train svmutils)

add_library(
    predict
    SHARED
    predict.cpp
)

target_link_libraries(predict svmutils)

file(COPY "assets/" DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/assets/) # copy assets to build

#MESSAGE( "OpenCV_LIBS: " ${OpenCV_LIBS} ) # simple message to show which OpenCV is used
