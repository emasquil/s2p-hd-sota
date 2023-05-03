/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdlib.h>
#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "libsgm.h"

extern "C" {
#include "iio.h"
}

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

int main(int argc, char* argv[])
{
	cv::CommandLineParser parser(argc, argv,
		"{@left_img   | <none> | path to input left image                                                            }"
		"{@right_img  | <none> | path to input right image                                                           }"
		"{@out_disp   | <none> | path to the output image                                                        }"
		"{disp_size   |    256 | maximum possible disparity value                                                    }"
		"{P1          |     10 | penalty on the disparity change by plus or minus 1 between nieghbor pixels          }"
		"{P2          |    120 | penalty on the disparity change by more than 1 between neighbor pixels              }"
		"{uniqueness  |   0.95 | margin in ratio by which the best cost function value should be at least second one }"
		"{subpixel    |        | enable subpixel estimation                                                          }"
		"{num_paths   |      8 | number of scanlines used in cost aggregation                                        }"
		"{min_disp    |      0 | minimum disparity value                                                             }"
		"{LR_max_diff |      1 | maximum allowed difference between left and right disparity                         }"
		"{help h      |        | display this help and exit                                                          }");

	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

 //       cv::Mat  left = cv::imread(parser.get<cv::String>("@left_img"), 0);
//	cv::Mat right = cv::imread(parser.get<cv::String>("@right_img"), 0);
//
//	const char *disp_filename = parser.get<cv::String>("@out_disp").c_str();
	cv::String df = parser.get<cv::String>("@out_disp");
	const char *disp_filename = df.c_str();


   int ncol, nrow, ncol_right, nrow_right;
	uint16_t*left_data  = iio_read_image_uint16((char*) parser.get<cv::String>("@left_img").c_str(), &ncol, &nrow);
	uint16_t*right_data = iio_read_image_uint16((char*) parser.get<cv::String>("@right_img").c_str(), &ncol_right, &nrow_right);


	if (!parser.check()) {
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	const int disp_size = parser.get<int>("disp_size");
	const int P1 = parser.get<int>("P1");
	const int P2 = parser.get<int>("P2");
	const float uniqueness = parser.get<float>("uniqueness");
	const int num_paths = parser.get<int>("num_paths");
	const int min_disp = parser.get<int>("min_disp");
	const int LR_max_diff = parser.get<int>("LR_max_diff");
	const bool subpixel = parser.has("subpixel");

	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");
	ASSERT_MSG(num_paths == 4 || num_paths == 8, "number of scanlines must be 4 or 8.");

	const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
	const int input_depth =  16;
	const int output_depth = 16;

	const sgm::StereoSGM::Parameters param(P1, P2, uniqueness, subpixel, path_type, min_disp, LR_max_diff);
    std::cout << "sgm initialization begin:" << std::endl;
    std::chrono::steady_clock::time_point begin_sgm_ini = std::chrono::steady_clock::now();
	sgm::StereoSGM ssgm(ncol, nrow, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_HOST2HOST, param);
    std::chrono::steady_clock::time_point end_sgm_ini = std::chrono::steady_clock::now();
    std::cout << "\tsgm all initialization time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_sgm_ini - begin_sgm_ini).count() << "[ms]" << std::endl;

	cv::Mat disparity(nrow, ncol, CV_16S);

    std::cout << "sgm processing begin:" << std::endl;
    std::chrono::steady_clock::time_point begin_sgm_proc = std::chrono::steady_clock::now();
	ssgm.execute(left_data, right_data, disparity.data);
    std::chrono::steady_clock::time_point end_sgm_proc = std::chrono::steady_clock::now();
    std::cout << "\tsgm processing time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_sgm_proc - begin_sgm_proc).count() << "[ms]" << std::endl;

	// create mask for invalid disp
	cv::Mat mask = disparity == ssgm.get_invalid_disparity();

	// show image
	cv::Mat disparity_8u; //, disparity_color;
	disparity.convertTo(disparity_8u, CV_32FC1);

	// disparity conventions
	disparity_8u *= -1;
	disparity_8u.setTo(NAN, mask);


	if (subpixel) { 
		//std::cout << sgm::StereoSGM::SUBPIXEL_SCALE;
		disparity_8u /= sgm::StereoSGM::SUBPIXEL_SCALE;
	}


	std::vector<float> array;
	array.assign((float*)disparity_8u.data, (float*)disparity_8u.data + disparity_8u.total()*disparity_8u.channels());
	//char *disp_filename= (char *) "disparity.tif";
	iio_write_image_float((char*) disp_filename, &array[0], disparity_8u.cols, disparity_8u.rows);


    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "\nTime spent all algorithm = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;


	return 0;
}
