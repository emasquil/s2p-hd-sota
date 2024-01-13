cmake_minimum_required(VERSION 3.1)

project(libSGM_debug)

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall -Werror=switch")
endif()

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)
if (OpenCV_VERSION VERSION_LESS 3.0)
	message(FATAL_ERROR "Error: OpenCV version requires at least 3.0")
endif()

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

cuda_add_executable(stereosgm.bin src/stereosgm_image.cpp src/stereo_sgm.cpp ${stereosgm_SRC})
target_link_libraries(stereosgm.bin ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TIFF_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES})

cuda_add_library(stereosgm src/libsgmgpu.cpp src/stereo_sgm.cpp ${stereosgm_SRC} SHARED)
target_link_libraries(stereosgm ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TIFF_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES})
install(TARGETS stereosgm DESTINATION lib)
install(FILES ./src/libsgmgpu.h DESTINATION lib)
