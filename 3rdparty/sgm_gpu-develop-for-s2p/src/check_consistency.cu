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

#include "libsgm.h"
#include "internal.h"
#include "utility.hpp"

namespace {
	template<typename SRC_T, typename DST_T>
	__global__ void check_consistency_kernel(DST_T* d_leftDisp, const DST_T* d_rightDisp, const SRC_T* d_left, int width, int height, int src_pitch, int dst_pitch, bool subpixel, int LR_max_diff, int min_disp) {

		const int j = blockIdx.x * blockDim.x + threadIdx.x;
		const int i = blockIdx.y * blockDim.y + threadIdx.y;

		// left-right consistency check, only on leftDisp, but could be done for rightDisp too

		SRC_T mask = d_left[i * src_pitch + j];
		DST_T org = d_leftDisp[i * dst_pitch + j];
		int d = org;
		if (subpixel) {
			d >>= sgm::StereoSGM::SUBPIXEL_SHIFT;
		}
		int k = j - d - (min_disp);
		if (mask == 0 || org == sgm::INVALID_DISP || (k >= 0 && k < width && LR_max_diff >= 0 && abs(d_rightDisp[i * dst_pitch + k] - d) > LR_max_diff)) {
			// masked or left-right inconsistent pixel -> invalid
			d_leftDisp[i * dst_pitch + j] = static_cast<DST_T>(sgm::INVALID_DISP);
		}
	}
}

namespace sgm {
	namespace details {

		void check_consistency(uint8_t* d_left_disp, const uint8_t* d_right_disp, const void* d_src_left, int width, int height, int depth_bits, int src_pitch, int dst_pitch, bool subpixel, int LR_max_diff, int min_disp) {

			const dim3 blocks(width / 16, height / 16);
			const dim3 threads(16, 16);
			if (depth_bits == 16) {
				check_consistency_kernel<uint16_t><<<blocks, threads>>>(d_left_disp, d_right_disp, (uint16_t*)d_src_left, width, height, src_pitch, dst_pitch, subpixel, LR_max_diff, min_disp);
			}
			else if (depth_bits == 8) {
				check_consistency_kernel<uint8_t><<<blocks, threads>>>(d_left_disp, d_right_disp, (uint8_t*)d_src_left, width, height, src_pitch, dst_pitch, subpixel, LR_max_diff, min_disp);
			}

			CudaKernelCheck();
		}

		void check_consistency(uint16_t* d_left_disp, const uint16_t* d_right_disp, const void* d_src_left, int width, int height, int depth_bits, int src_pitch, int dst_pitch, bool subpixel, int LR_max_diff, int min_disp) {

			const dim3 blocks(width / 16, height / 16);
			const dim3 threads(16, 16);
			if (depth_bits == 16) {
				check_consistency_kernel<uint16_t><<<blocks, threads>>>(d_left_disp, d_right_disp, (uint16_t*)d_src_left, width, height, src_pitch, dst_pitch, subpixel, LR_max_diff, min_disp);
			}
			else if (depth_bits == 8) {
				check_consistency_kernel<uint8_t><<<blocks, threads>>>(d_left_disp, d_right_disp, (uint8_t*)d_src_left, width, height, src_pitch, dst_pitch, subpixel, LR_max_diff, min_disp);
			}
			
			CudaKernelCheck();	
		}

	}
}
