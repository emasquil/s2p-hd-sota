# Copyright (C) 2013, Carlo de Franchis <carlo.de-franchis@cmla.ens-cachan.fr>
# Copyright (C) 2013, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2013, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>
# Copyright (C) 2013, Julien Michel <julien.michel@cnes.fr>

# This module contains the default config, with all the parameters of the
# s2p pipeline. This dictionary is updated at runtime with parameters defined
# by the user in the config.json file. All the optional parameters (that the
# user is not forced to define in config.json) must be defined here, otherwise
# they won't have a default value.


def get_default_config() -> dict:
    cfg: dict = {}

    # path to output directory
    cfg['out_dir'] = "s2p_output"

    # path to directory where (many) temporary files will be stored
    cfg['temporary_dir'] = "s2p_tmp"

    # temporary files are erased when s2p terminates. Switch to False to keep them
    cfg['clean_tmp'] = True

    # remove all generated files except from ply point clouds and tif raster dsm
    cfg['clean_intermediate'] = False

    # switch to True if you want to process the whole image
    cfg['full_img'] = False

    # s2p processes the images tile by tile. The tiles are squares cropped from the
    # reference image. The width and height of the tiles are given by this param, in pixels.
    cfg['tile_size'] = 800

    # margins used to increase the footprint of the rectified tiles, to
    # account for poor disparity estimation close to the borders
    cfg['horizontal_margin'] = 50  # for regularity and occlusions
    cfg['vertical_margin'] = 10  # for regularity

    # max number of processes launched in parallel. None means the number of available cores
    cfg['max_processes'] = None

    # max number of processes launched in parallel for stereo_matching
    # Uses the value of cfg['max_processes'] if None
    #   If the GPU has little VRAM, reduce this value so that CUDA contexts (per cpu)
    #   do not fill the VRAM for nothing.
    # It should be set if 'gpu_total_memory' is not set.
    cfg['max_processes_stereo_matching'] = None

    # total GPU memory allowed for this instance of s2p, in MB (eg: 32000 for 32GB GPU)
    # There is no mechanism to restrict s2p to this quantity, but the code will use this value
    # to schedule a reasonnable number of jobs on the GPU in parallel.
    # It should be set if 'max_processes_stereo_matching' is not set.
    cfg['gpu_total_memory'] = None

    # max number of OMP threads used by programs compiled with openMP
    cfg['omp_num_threads'] = 1

    # timeout in seconds, after which a function that runs on a single tile is not
    # waited for
    cfg['timeout'] = 600

    # debug mode (more verbose logs and intermediate results saved)
    cfg['debug'] = False

    # resolution of the output digital surface model, in meters per pixel
    cfg['dsm_resolution'] = 4

    # radius to compute altitudes (and to interpolate the small holes)
    cfg['dsm_radius'] = 0

    # dsm_sigma controls the spread of the blob from each point for the dsm computation
    # (dsm_resolution by default)
    cfg['dsm_sigma'] = None

    # point cloud is aggregated into a DSM by takin the max on each spatial cell, this is now the default dehavior in s2p
    # another possibility is to aggregate with average but this is not recommended on urban areas
    cfg['dsm_aggregation_with_max'] = True

    # relative sift match threshold (else sift match threshold is absolute)
    cfg['relative_sift_match_thresh'] = True

    # use the opencv's implementation of SIFT, which works on 8 bit images so the inputs
    # are rescaled, by default IPOL's SIFT is used, which works on images with any range depth
    # Most of the SIFT options below are not used by opencv's SIFT (it's just faster).
    cfg['sift_use_opencv_implementation'] = False

    # relative sift match threshold (else sift match threshold is absolute)
    cfg['relative_sift_match_thresh'] = True

    # if cfg['relative_sift_match_thresh'] is True :
    # sift threshold on the first over second best match ratio
    # else (absolute) a reasonable value is between 200 and 300 (128-vectors SIFT descriptors)
    cfg['sift_match_thresh'] = 0.6

    # disp range expansion facto
    cfg['disp_range_extra_margin'] = 0.2

    # Maximum disparity range allowed in block matching
    cfg['max_disp_range'] = None

    # estimate rectification homographies either blindly using the rpc data or from
    # the images actual content thanks to sift matches
    cfg['rectification_method'] = 'rpc'  # either 'rpc' or 'sift'

    # register the rectified images with a shear estimated from the rpc data
    cfg['register_with_shear'] = True

    # number of ground control points per axis in matches from rpc generation
    cfg['n_gcp_per_axis'] = 5

    # max distance allowed for a point to the epipolar line of its match
    cfg['epipolar_thresh'] = 0.5

    # maximal pointing error, in pixels
    cfg['max_pointing_error'] = 10

    # set these params if you want to impose the disparity range manually (cfg['disp_range_method'] == 'fixed_pixel_range')
    cfg['disp_min'] = None
    cfg['disp_max'] = None

    # set these params if you want to impose the altitude range manually (cfg['disp_range_method'] == 'fixed_altitude_range')
    cfg['alt_min'] = None
    cfg['alt_max'] = None

    # width of a stripe of pixels to be masked along the reference input image borders
    cfg['border_margin'] = 10

    # radius for erosion of valid disparity areas. Ignored if less than 2
    cfg['msk_erosion'] = 2

    cfg['fusion_operator'] = 'average_if_close'

    # threshold (in meters) used for the fusion of two dems in triplet processing
    cfg['fusion_thresh'] = 3

    cfg['rpc_alt_range_scale_factor'] = 1

    # method to compute the disparity range: "sift", "exogenous", "wider_sift_exogenous", "fixed_pixel_range", "fixed_altitude_range"
    cfg['disp_range_method'] = "wider_sift_exogenous"
    cfg['disp_range_exogenous_low_margin'] = -10
    cfg['disp_range_exogenous_high_margin'] = +100

    # whether or not to use SRTM DEM (downloaded from internet) to estimate:
    #   - the average ground altitude (to project the input geographic AOI to the
    #     correct place in the input images)
    #   - a reasonable altitude range (to get a better rectification when
    #     "rectification_method" is set to "rpc")
    cfg['use_srtm'] = False

    # exogenous dem. If set, it superseeds SRTM.
    cfg['exogenous_dem'] = None
    cfg['exogenous_dem_geoid_mode'] = True

    ### stereo matching parameters

    # stereo matching algorithm: 'tvl1', 'msmw', 'hirschmuller08',
    # hirschmuller08_laplacian', 'sgbm', 'mgm', 'mgm_multi', 'stereosgm_gpu'
    cfg['matching_algorithm'] = 'mgm'

    # this option allows to refine the disparity computed by the fast stereosgm_gpu 
    # it only works in combination with  cfg['matching_algorithm'] = 'stereosgm_gpu'
    # the postrpocess runs an MGM on a narrow band around the previous result
    cfg['postprocess_stereosgm_gpu'] = False

    # size of the Census NCC square windows used in mgm
    cfg['census_ncc_win'] = 5

    # MGM parameter: speckle filter minimum area (REMOVESMALLCC flag)
    cfg['stereo_speckle_filter'] = 25

    # MGM parameter: regularity (multiplies P1 and P2)
    cfg['stereo_regularity_multiplier'] = 1.0

    # MGM parameters:
    # number of directions explored for regularization
    cfg['mgm_nb_directions'] = 8
    # timeout in seconds, after which a running mgm process will be killed
    cfg['mgm_timeout'] = 600
    # distance threshold (in pixels) for the left-right consistency test
    cfg['mgm_leftright_threshold'] = 1.0
    # controls the mgm left-right consistency check. 0: disabled
    #                                                1 (default): enabled at all scales
    #                                                2: enables only at the last scale (faster)
    cfg['mgm_leftright_control'] = 1
    # controls the mgm mindiff filter check. -1: disabled (default), produce denser maps
    #                                         1: enabled, produce conservative results
    cfg['mgm_mindiff_control'] = -1

    # remove isolated 3d points in height maps
    # within a sphere of radius R GSDs, we expect to find A=2pi R^2 samples. 
    # If less than fill factor *A are present then the center is discarded. 
    # Fill factor is usually set to 1/3 or 1/4 
    cfg['3d_filtering_radius_gsd'] = None  # radius of the sphere as number of GSDs : usually set to 3 or 5
    cfg['3d_filtering_fill_factor'] = None  # number of points as fraction of the disk for the given radius : usually set to 1/3 or 1/4


    # clean height maps outliers
    cfg['cargarse_basura'] = True

    # computes a filtered DSM by filtering small holes
    # generates a  dsm-filtered.tif next to the usual dsm.tif
    cfg['fill_dsm_holes_smaller_than'] = 50

    # Output coordinate reference system
    # All formats accepted by `pyproj.CRS()` are allowed, for example:
    # 32740 (int interpreted as an EPSG code), or
    # "epsg:32740+5773" (authority string), or
    # "+proj=utm +zone=40 +south +datum=WGS84 +units=m +vunits=m +no_defs +type=crs" (proj4 string)
    # If None, the local UTM zone will be used
    cfg['out_crs'] = None

    # If the out_crs is not set this parameter determines if the output CRS uses the EGM96 geoid vertical datum (if True)
    # or the WGS84 ellipsoid vertical datum (if False). If out_crs is set, this parameter is ignored.
    cfg['out_geoid'] = False

    # If true, discard tiles that are all nodata (or 0 if 'nodata' is not set in the profile).
    cfg['init_check_all_nodata'] = False

    # Fit a localization RPC using the projection RPC.
    # This speeds-up the triangulation step if the localization rpc is not provided in the images.
    cfg['fit_localization_rpc'] = False

    # Maximum allowed altitude span (in meters) among triangulated SIFT matches.
    # If the span exceeds this value, it is assumed that some matches are likely incorrect.
    # In that case, only points within median altitude ± 'altitude_margin' are kept.
    cfg['max_altitude_span'] = 300

    # Margin (in meters) around the median altitude within which matches are retained
    # when filtering is applied due to a high altitude span.
    cfg['altitude_margin'] = 250

    # Merging method when merging local DSMs into a global one, in case of overlapping tiles
    # "first": reverse painting, i.e value chosen from the first tile merged
    # "last": paint valid new on top of existing
    # "min" or "max": pixel-wise (min or max) of existing and new
    cfg['dsm_merging_method'] = "max"

    # Flag to configure the range of the disparity map.
    cfg['disp_range_flag'] = "positive"

    # Stereo matching parameters for the stereoanywhere correlator
    cfg['stereo_ckpt'] = None  # Path to the stereoanywhere checkpoint
    cfg['mono_ckpt'] = None  # Path to the monoanywhere checkpoint

    cfg['horizontal_translation_margin'] = 0.0  # Margin to add to the horizontal translation for disparity range adjustment

    return cfg
