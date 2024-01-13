"""
speckle filter (remove small cc)

Copyright (C) 2017-2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

import numpy as np

from numba import jit


def specklefilter(off, area=25, th=0):
    '''
    speckle filter of dispairt map off

    Args:
        off:  numpy array with the input disparity map
        area: the surface (in pixels) of the smallest allowed connected component of disparity
        th:   similarity threshold used to determin if two neighboring pixels have the same value

    Returns:
       numpy array with the filtered disparity map, removed points are set to nan
    '''

    @jit(nopython=True)
    def find(i,idx):     # finds the root of a dsf
        if idx.flat[i] == i:
            return i
        else:
            ret = find(idx.flat[i],idx)
            idx.flat[i] = ret    # path compression is useles with idx passed by value
            return ret

    @jit(nopython=True)
    def dsf(D, th=0):    # builds a dsf
        h,w = D.shape[0],D.shape[1]
        idx = np.zeros((h,w),dtype=np.int64)
        for j in range(h):
            for i in range(w):
                idx[j,i] = j*w + i

        for j in range(h):
            for i in range(w):
                if(i>0):
                    if( abs(D[j,i] - D[j,i-1])<= th ):
                        a = find(idx[j,i],idx)
                        b = find(idx[j,i-1],idx)
                        idx[j,i] = idx[j,i-1]
                        idx.flat[a] = b

                if(j>0):
                    if( abs(D[j,i] - D[j-1,i])<= th ):
                        a = find(idx[j,i],idx)
                        b = find(idx[j-1,i],idx)
                        idx[j,i] = idx[j-1,i]
                        idx.flat[a] = b

        return idx

    @jit(nopython=True)
    def labels(idx):
        h,w=idx.shape[0],idx.shape[1]
        lab = idx*0

        for i in range(h*w):
            ind = find(i,idx)
            lab.flat[i] = ind
        return lab

    @jit(nopython=True)
    def areas(lab):
        h,w=lab.shape[0],lab.shape[1]
        area = np.zeros((h,w),dtype=np.int64)
        LL = np.zeros((h,w),dtype=np.int64)
        for i in range(w*h):
            area.flat[lab.flat[i]] += 1
        for i in range(w*h):
            LL.flat[i] = area.flat[lab.flat[i]]
        return LL


    # build the dsf
    ind = dsf(off, th=th)
    # extract the labels of all the regions
    lab = labels(ind)
    # creat e map where all the regions are tagged with their area
    are = areas(lab)
    # filter the disparity map
    filtered = np.where((are>area), off, np.nan)

    return filtered


def call(inf, outf, area=25, th=0):
    import iio
    im=iio.read(inf).squeeze()
    out = specklefilter(im, area, th)
    iio.write(outf, out)


if __name__ == '__main__':
    import fire
    fire.Fire(call)



