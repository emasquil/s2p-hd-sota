# Copyright (C) 2019, Carlo de Franchis <carlo.de-franchis@cmla.ens-cachan.fr>

import numpy as np
import plyfile


def read_3d_point_cloud_from_ply(path_to_ply_file):
    """
    Read a 3D point cloud from a ply file and return a numpy array.

    Args:
        path_to_ply_file (str): path to a .ply file

    Returns:
        numpy array with the list of 3D points, one point per line
        list of strings with the ply header comments
    """
    plydata = plyfile.PlyData.read(path_to_ply_file)
    d = np.asarray(plydata['vertex'].data)
    array = np.column_stack([d[p.name] for p in plydata['vertex'].properties])
    return array, plydata.comments


def write_3d_point_cloud_to_ply(path_to_ply_file, coordinates, colors=None,
                                extra_properties=None,
                                extra_properties_names=None, comments=[]):
    """
    Write a 3D point cloud to a ply file.

    Args:
        path_to_ply_file (str): path to a .ply file
        coordinates (array): numpy array of shape (n, 3) containing x, y, z coordinates
        colors (array): numpy array of shape (n, 3) or (n, 1) containing either
            r, g, b or gray levels
        extra_properties (array): optional numpy array of shape (n, k)
        extra_properties_names (list): list of k strings with the names of the
            (optional) extra properties
        comments (list): list of strings containing the ply header comments
    """
    dtypes = [('x', coordinates.dtype),
              ('y', coordinates.dtype),
              ('z', coordinates.dtype)]
    arrays = [coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]

    if colors is not None:
        if colors.shape[1] == 1:  # replicate grayscale 3 times
            colors = np.column_stack([colors] * 3)
        elif colors.shape[1] not in [3, 4]:
            raise Exception('Error: colors must have either 1, 3 or 4 channels')
        arrays += [colors[:, 0], colors[:, 1], colors[:, 2]]
        dtypes += [('red', colors.dtype),
                   ('green', colors.dtype),
                   ('blue', colors.dtype)]
        if colors.shape[1] == 4:
            dtypes += [('ir', colors.dtype)]

    if extra_properties is not None:
        if extra_properties.ndim == 1:
            extra_properties = extra_properties[..., None]
        arrays += [extra_properties[:, i] for i in range(extra_properties.shape[1])]
        dtypes += [(s, extra_properties.dtype) for s in extra_properties_names]

    data = np.core.records.fromarrays(arrays, dtypes)
    plydata = plyfile.PlyElement.describe(data, 'vertex')
    plyfile.PlyData([plydata], comments=comments).write(path_to_ply_file)
