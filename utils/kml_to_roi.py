from __future__ import print_function
from s2p import common 
import bs4
import sys
import utm 
import rpcm
import datetime
import numpy as np


def kml_roi_process(rpc, kml):
    """
    """
    # extract lon lat from kml
    f = open(kml, 'r')
    a = bs4.BeautifulSoup(f, "lxml").find_all('coordinates')[0].text.split()
    f.close()
    #ll_bbx = np.array([list(map(float, x.split(','))) for x in a])
    #print(ll_bbx)
    ll_bbx = np.array([list(map(float, x.split(','))) for x in a])[:4, :2]

    # save lon lat bounding box to cfg dictionary
    lon_min = min(ll_bbx[:, 0])
    lon_max = max(ll_bbx[:, 0])
    lat_min = min(ll_bbx[:, 1])
    lat_max = max(ll_bbx[:, 1])
    #cfg['ll_bbx'] = (lon_min, lon_max, lat_min, lat_max)

    # convert lon lat bbox to utm
    z = utm.conversion.latlon_to_zone_number((lat_min + lat_max) * .5,
                                             (lon_min + lon_max) * .5)
    utm_bbx = np.array([utm.from_latlon(p[1], p[0], force_zone_number=z)[:2] for
                        p in ll_bbx])
    east_min = min(utm_bbx[:, 0])
    east_max = max(utm_bbx[:, 0])
    nort_min = min(utm_bbx[:, 1])
    nort_max = max(utm_bbx[:, 1])
    #cfg['utm_bbx'] = (east_min, east_max, nort_min, nort_max)

    # project lon lat vertices into the image
    if not isinstance(rpc, rpcm.rpc_model.RPCModel):
        rpc = rpcm.rpc_model.RPCModel(rpc)
    img_pts = [rpc.projection(p[0], p[1], rpc.alt_offset)[:2] for p in ll_bbx]

    # return image roi
    x, y, w, h = common.bounding_box2D(img_pts)
    return {'x': x, 'y': y, 'w': w, 'h': h}




def print_help_and_exit(script_name):
    """
    """
    print("""
    Incorrect syntax, use:
      > %s geotifffile kmlfile
        computes the roi from

      All the parameters, paths to input and output files, are defined in
      the json configuration file.

    """ % script_name)
    sys.exit()


if __name__ == '__main__':
    if len(sys.argv) == 3:
        rpc = rpcm.rpc_from_geotiff(sys.argv[1])
        bbox = kml_roi_process(rpc, sys.argv[2])

        print ("'x': %d, 'y': %d, 'w': %d, 'h': %d"%(bbox['x'], bbox['y'], bbox['w'], bbox['h'] ) )
    else:
        print_help_and_exit(sys.argv[0])
