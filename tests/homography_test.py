import os

import numpy as np
import rasterio

from s2p import homography


def test_success(tmp_path):
    im = os.path.join(tmp_path, "im.tif")
    out = os.path.join(tmp_path, "out.tif")

    with rasterio.open(
        im,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=1,
        dtype=np.float32,
    ) as dst:
        arr = np.empty((100, 100), dtype=np.float32)
        dst.write(arr, 1)

    H = np.asarray([[0.8, 0, 50], [0, 0.7, 20], [0, 0, 1]], dtype=np.float64)
    w, h = 200, 300
    success = homography.image_apply_homography(out, im, H, w, h)
    print(success)

    assert success
    assert os.path.exists(out)


def test_out_of_domain(tmp_path):
    im = os.path.join(tmp_path, "im.tif")
    out = os.path.join(tmp_path, "out.tif")

    with rasterio.open(
        im,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=1,
        dtype=np.float32,
    ) as dst:
        arr = np.empty((100, 100), dtype=np.float32)
        dst.write(arr, 1)

    H = np.asarray([[1, 0, 250], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    w, h = 200, 300
    success = homography.image_apply_homography(out, im, H, w, h)

    assert not success
    assert not os.path.exists(out)
