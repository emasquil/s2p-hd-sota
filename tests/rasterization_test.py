# s2p (Satellite Stereo Pipeline) testing module

import math

import numpy as np
import rasterio
from plyflatten import plyflatten_from_plyfiles_list

from tests_utils import data_path

#    @property
#    def is_tiled(self):
#        return self.block_shapes and self.block_shapes[0][1] != self.width
#        warnings.warn(
#            "is_tiled will be removed in a future version. "
#            "Please consider copying the body of this function "
#            "into your program or module.",
#            PendingDeprecationWarning
#        )
#        # It's rare but possible that a dataset's bands have different
#        # block structure. Therefore we check them all against the
#        # width of the dataset.
#        return self.block_shapes and all(self.width != w for _, w in self.block_shapes)


def test_plyflatten():
    # Test data
    f = data_path("input_ply/cloud.ply")
    raster, profile = plyflatten_from_plyfiles_list([f], resolution=0.4)
    test_raster = raster[:, :, 0]  # keep only band with height

    # Expected data
    e = data_path("expected_output/plyflatten/dsm_40cm.tiff")
    with rasterio.open(e) as src:
        expected_raster = src.read(1)
        expected_crs = src.crs
        expected_transform = src.transform
        T = src.block_shapes and all(src.width != w for _,w in src.block_shapes)
        expected_is_tiled = T
        expected_nodata = src.nodata

    # Check that both rasters are equal pixel-wise within a tolerance
    assert np.allclose(test_raster, expected_raster, equal_nan=True)

    # Check that both images have the same CRS
    test_crs = profile['crs']
    assert test_crs == expected_crs

    # Check that both images have the same transform
    test_transform = profile['transform']
    assert np.allclose(test_transform, expected_transform)

    test_is_tiled = profile['tiled']
    assert test_is_tiled == expected_is_tiled

    test_nodata = profile.get('nodata')
    if expected_nodata and math.isnan(expected_nodata):
        assert math.isnan(test_nodata)
    else:
        assert test_nodata == expected_nodata
