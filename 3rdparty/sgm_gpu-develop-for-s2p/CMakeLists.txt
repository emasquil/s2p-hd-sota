project(libSGM_debug)

cmake_minimum_required(VERSION 3.1)


set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)
if (OpenCV_VERSION VERSION_LESS 3.0)
	message(FATAL_ERROR "Error: OpenCV version requires at least 3.0")
endif()

set(OpenCV_DIR "/home/guille/Downloads/opencv/build")
include_directories(${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TIFF_INCLUDE_DIR})
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIR})
find_package( OpenCV REQUIRED )
find_package( TIFF REQUIRED )
find_package( JPEG REQUIRED )
find_package( PNG REQUIRED )

file(GLOB stereosgm_SRC
     "src/*.cu"
     "src/*.h"
     "src/*.c"
     "src/*.hpp"
)

cuda_add_executable(stereosgm src/stereosgm_image.cpp src/stereo_sgm.cpp ${stereosgm_SRC})
target_link_libraries(stereosgm ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TIFF_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES})
