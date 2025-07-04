# S2P-hd - Satellite Stereo Pipeline

![Build Status](https://github.com/centreborelli/s2p/actions/workflows/build.yml/badge.svg)
![Tests Status](https://github.com/centreborelli/s2p/actions/workflows/tests.yml/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/s2p)](https://pypi.org/project/s2p)

S2P-hd is a Python library and command line tool that implements a stereo
pipeline which produces elevation models from images taken by high resolution
optical satellites such as Pléiades, WorldView, QuickBird, Spot or Ikonos. It
generates 3D point clouds and digital surface models from stereo pairs (two
images) or tri-stereo sets (three images) in a completely automatic fashion.

S2P-hd builds upon and improves [S2P](https://github.com/centreborelli/s2p), which was used to win the 2016 [IARPA Multi-View Stereo 3D Mapping Challenge](https://www.iarpa.gov/challenges/3dchallenge.html).

A wide variety of stereo correlation algorithms are supported, including a gu-accelerated version of semi-global matching (SGM), MGM, TV-L1 optical flow, etc.

The main language is Python, although several operations are handled by
binaries written in C.

The pipeline is implemented in the Python package `s2p`. It can be used
to produce surface models and 3D point clouds from arbitrarily large regions
of interest or from complete images. If needed, it cuts the region of interest
in several small tiles and process them in parallel.

Its main source code repository is https://github.com/centreborelli/s2p-hd.


# Dependencies

## GDAL
The main dependency is GDAL. Version 2.1.0 or newer is required.

### On Ubuntu 18.04
`gdal` can be installed with `apt-get`.

    sudo apt update
    sudo apt install libgdal-dev

### On macOS
[Download GDAL](http://www.kyngchaos.com/files/software/frameworks/GDAL_Complete-2.4.dmg)
and install the `.dmg` file.

## Other dependencies (fftw, libtiff)

On Ubuntu:

    sudo apt install build-essential libfftw3-dev libgeotiff-dev libopencv-dev libtiff5-dev cmake gdal-bin

On macOS:

    brew install fftw libtiff cmake

If lacking administrative privileges to run `sudo`, all these dependencies exist as conda 
packages and can be installed in a user directory. Then the path to them can be specified
in the s2p makefiles if compiling it from source. 

# Installation

Install it in editable mode from a git clone:

    git clone https://github.com/centreborelli/s2p-hd.git --recursive 
    cd s2p-hd
    pip install -e ".[test]"

Some python rely on external binaries. Most of these binaries
were written on purpose for the needs of the pipeline, and their source code is
provided here in the `c` folder. For the other binaries, the source code is
provided in the `3rdparty` folder.

All the sources (ours and 3rdparties) are compiled from the same makefile. Just
run `make all` from the `s2p-hd` folder to compile them. This will create a `bin`
directory containing all the needed binaries. This makefile is used when
running `pip install .`

You can test if s2p-hd is correctly working using:

    make test

If the test fails due to a comparison with nan, this probably means that the pyproj
data is not correctly downloaded.  You can force its download by running

    pyproj sync -v --file us_nga_egm96_15

If some libraries needed by `s2p-hd` (such as `libfftw3`) are installed in a custom location,
for example `/usr/joe/local`, then the compilation and tests will fail with exit status 127
or mentioning not being able to load shared  libaries.  You can help the compiler to find
these libraries by defining the following variables:

    export CPATH=/usr/joe/local/include
    export LIBRARY_PATH=/usr/joe/local/lib
    
The following invocation can be used then on Linux:

    LD_LIBRARY_PATH=/usr/joe/local/lib make test

and the same for the `s2p` command later. One macOS one may use instead 
`DYLD_FALLBACK_LIBRARY_PATH`.

# Usage

`s2p-hd` is a Python library that can be imported into other applications. It also
comes with a Command Line Interface (CLI).

## From the command line

The `s2p` CLI usage instructions can be printed with the `-h` and `--help` switches.

    $ s2p -h
    usage: s2p.py [-h] config.json

    S2P: Satellite Stereo Pipeline

    positional arguments:
      config.json           path to a json file containing the paths to input and
                            output files and the algorithm parameters

    optional arguments:
      --start_from          Restart from a given step in case of an interruption or to try different parameters.
      -h, --help            show this help message and exit

To run the whole pipeline, call `s2p` with a json configuration file as unique argument:

    s2p tests/data/input_pair/config.json

All the parameters of the algorithm, paths to input and output data are stored
in the json file. See the provided `test.json` file for an example, and the
comments in the file `s2p/config.py` for some explanations about the roles
of these parameters.

Notice that each input image must have RPC coefficients, either in its GeoTIFF
tags or in a companion `.xml` or `.txt` file.

#### ROI definition

The processed Region of interest (ROI) is defined by the image coordinates (x,
y) of its top-left corner, and its dimensions (w, h) in pixels. These four
numbers must be given in the `json` configuration file, as in the `test.json`
example file. They are ignored if the parameter `'full_img'` is set to `true`.
In that case the full image will be processed.

#### File paths in json configuration files

In the json configuration files, input and output paths are relative to the json
file location, not to the current working directory.




## References

If you use this software please cite the following papers:

[*s2p-hd: Gpu-Accelerated Binocular Stereo Pipeline for Large-Scale 
Same-Date Stereo*](https://hal.science/view/index/docid/5051235), Tristan Amadei, 
Enric Meinhardt-Llopis, Carlo de Franchis, Jeremy Anger, Thibaud Ehret, 
Gabriele Facciolo. CVPR EarthVision 2025.

[*An automatic and modular stereo pipeline for pushbroom
images*](http://dx.doi.org/10.5194/isprsannals-II-3-49-2014), Carlo de
Franchis, Enric Meinhardt-Llopis, Julien Michel, Jean-Michel Morel, Gabriele
Facciolo. ISPRS Annals 2014.

[*On Stereo-Rectification of Pushbroom
Images*](http://dx.doi.org/10.1109/ICIP.2014.7026102), Carlo de Franchis, Enric
Meinhardt-Llopis, Julien Michel, Jean-Michel Morel, Gabriele Facciolo.  ICIP
2014.

[*Automatic sensor orientation refinement of Pléiades stereo
images*](http://dx.doi.org/10.1109/IGARSS.2014.6946762), Carlo de Franchis,
Enric Meinhardt-Llopis, Julien Michel, Jean-Michel Morel, Gabriele Facciolo.
IGARSS 2014.
