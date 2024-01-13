import os
from typing import Any

import cffi
import numpy as np
from numpy.typing import NDArray

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB_FOLDER = os.path.join(ROOT, '..', 'lib')

ffi = cffi.FFI()
ffi.cdef(open(os.path.join(LIB_FOLDER, "libsgmgpu.h")).read())

sgmgpu: Any = ffi.dlopen(os.path.join(LIB_FOLDER, "libstereosgm.so"))


def wrap(array):
    if array.dtype == np.float32:
        typestr = "float*"
    elif array.dtype == bool:
        typestr = "bool*"
    elif array.dtype == np.int32:
        typestr = "int*"
    elif array.dtype == np.uint16:
        typestr = "uint16_t*"
    else:
        assert False
    return ffi.from_buffer(typestr, array, require_writable=True)


def run(
    im1: NDArray[np.float32],
    im2: NDArray[np.float32],
    *,
    nb_dir: int,
    disp_min: int,
    verbose: bool = False,
    P1: int = 10,
    P2: int = 40,
    # see 3rdparty/sgm_gpu-develop-for-s2p/src/census_transform.hpp
    census_transform_size: int = 3,  # 0: 5x5, 1: 7x5, 2: 7x7, 3: 9x7
) -> NDArray[np.float32]:
    h = sgmgpu.make_sgm_gpu(
        # disp_size
        512,
        # P1
        P1,
        # P2
        P2,
        # uniqueness
        0.95,
        # num_paths
        nb_dir,
        # min_disp
        disp_min,
        # LR_max_diff
        1,
        # subpixel
        True,
        # census_transform_size
        census_transform_size,
        # verbose
        verbose,
    )

    im1 = np.nan_to_num(im1, nan=0).astype(np.uint16)
    im2 = np.nan_to_num(im2, nan=0).astype(np.uint16)
    result = np.zeros_like(im1, dtype=np.float32)
    sgmgpu.exec_sgm_gpu(
        h, im1.shape[0], im1.shape[1], wrap(im1), wrap(im2), wrap(result)
    )
    sgmgpu.free_sgm_gpu(h)

    return result
