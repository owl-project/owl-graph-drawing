Sample Code for "Accelerating Force-Directed Graph Drawing with RT Cores"
=========================================================================

This project contains sample code for the paper

[S. Zellmann](https://vis.uni-koeln.de/team/stefan-zellmann), [M. Weier](https://www.h-brs.de/en/inf/martin-weier), [I. Wald](http://www.sci.utah.edu/~wald/), "Accelerating Force-Directed Graph Drawing with RT Cores", IEEE Visualization 2020, Shortpapers.

[Preprint](https://www.researchgate.net/publication/343904020_Accelerating_Force-Directed_Graph_Drawing_with_RT_Cores) (researchgate link)

Source Code Overview
--------------------

While some of the source code files merely contain "boilerplate code", the following implementation files contain the actual graph drawing code:

Implementation of the graph drawing framework: [gd.h](/gd.h) / [gd.cu](/gd.cu)
These files implement the various phases of the graph drawing algorithm with CUDA. For the repulsive phase, the files contain a simple CUDA kernel implementing the naive method, or alternatively call into the OWL/OptiX or LBVH nearest neighbor programs and kernels that are implemented elsewhere.

Implementation of the OptiX device programs: [optixSpringEmbedder.h](/optixSpringEmbedder.h) / [optixSpringEmbedder.cu](/optixSpringEmbedder.cu)
These files contain the optix intersection and bounds programs that implement the ray tracing-based nearest neighbor query. For the OWL/host side, see the routines in [gd.cu](/gd.cu) that call into these programs.

Implementation of the LBVH data structure: [lbvh.h](/lbvh.h) / [lbvh.cu](/lbvh.cu)
These files contain the implementation of our reference method based on the fast BVH tree construction algorithm by Karras.

Implementation of the user interface: [main.cpp](/main.cpp)
This file implements the user interface. The graph drawing algorithm runs in a separate thread from the user interface. We use a simple OpenGL renderer that only draws the graph edges as GL lines. This is not how we generated the images for the paper; for that, we rather use the software [Tulip](https://tulip.labri.fr/TulipDrupal/). Our graph layouts can be  exported to the format supported by Tulip.

Dependencies
------------

The sample code includes the following third-party libraries as submodules:

- [OWL](https://github.com/owl-project/owl) wrappers library on top of OptiX 7
- [Visionaray](https://github.com/szellmann/visionaray) ray tracing library, but only used for GUI and linear algebra
- [CmdLine](https://github.com/abolz/CmdLine) library to handle command line arguments (archived, use [CmdLine2](https://github.com/abolz/CmdLine2) instead)

In addition, the following dependencies should be installed by the user:

- C++11 compliant compiler, tested on Ubuntu 18.04 (g++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0)
- [CMake][1]
- [OpenGL][12]
- [GLEW][3]
- [NVIDIA CUDA Toolkit][4]
- [Optix 7](https://developer.nvidia.com/designworks/optix/download)
- [Boost][2]
- [GLUT][5] or [FreeGLUT][6]

On Ubuntu 18.04, the dependencies can, e.g., be installed using the following command: `sudo apt-get install cmake libglew-dev libboost-all-dev freeglut3-dev`. CUDA and OptiX should be installed via the links provided above and according to the respective instructions. Make sure that you have a proprietary NVIDIA graphics driver (comes, e.g., with the CUDA toolkit) installed and loaded by the kernel.

Building
--------

The following assumes that the OptiX and CUDA base directories can be found via environment variables; otherwise, adjust the paths accordingly:

```Shell
git clone --recursive https://github.com/owl-project/owl-graph-drawing

cd owl-graph-drawing
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DOptiX_INCLUDE:PATH=${OptiX_DIR}/include -DBIN2C=${CUDA_DIR}/bin/bin2c
make
```

Running
-------

After successfully building the application, the application binary will be located in the `build/` directory and is called `gd`. By default, `gd` loads an artificial graph; you can start the iterative graph drawing algorithm by pressing **Key-SPACE**. `gd` can also load `.csv` files exported with [Gephi](https://gephi.org/). Artificial data sets initialized upon application start can also be customized via the command line. For detailed usage info, type `./gd -help`:

```
Usage:
   ./gd [OPTIONS] [files...]

Positional options:
   [files...]             A list of input files

Options:
   -C[=<int>]             Clusters
   -bench[=<bool>]        Benchmarking mode to disable keypress event handling
   -bgcolor               Background color
   -camera=<ARG>          Text file with camera parameters
   -connected[=<bool>]    Generate connected graph
   -dt[=<ARG>]            Select data generation mode:
      =artificial         - Artificial Graph Generation
      =tree               - Tree Generation
      =file               - File Input
   -epc[=<int>]           Edges per Clusters
   -fullscreen            Full screen window
   -height=<ARG>          Window height
   -mode[=<ARG>]          Select graph layout mode:
      =naive              - Naive Implementation
      =rtx                - RTX Mode
      =lbvh               - LBVH Mode
   -n[=<int>]             Maximum number of iterations
   -npc[=<int>]           Nodes per Clusters
   -o[=<string>]          Output tlp file
   -r[=<int>]             Number of repetitions when loading data
   -refit_after[=<ARG>]   Refit RTX BVH after N iterations
   -trDegree[=<int>]      Tree data generation degree
   -trDepth[=<int>]       Tree data generation depth
   -width=<ARG>           Window width
```

Use the following mouse controls to navigate the 3D graph:

* **LMB**: Pan the scene.
* **MMB**: Pan the scene.
* **RMB**: Zoom into the scene.

Citation
--------

If you want to refer to this project in your own scientific work, please cite the following research paper:

```
@INPROCEEDINGS{zellmann:2020,
  author={Zellmann, Stefan and Weier, Martin and Wald, Ingo},
  booktitle={2020 IEEE Visualization Conference (VIS)},
  title={Accelerating Force-Directed Graph Drawing with RT Cores},
  year={2020},
  volume={},
  number={},
  pages={1-5},
}
```

License
-------

This sample code is licensed under the MIT License (MIT)

[1]:    http://www.cmake.org/download/
[2]:    http://www.boost.org/users/download/
[3]:    http://glew.sourceforge.net/
[4]:    https://developer.nvidia.com/cuda-toolkit
[5]:    https://www.opengl.org/resources/libraries/glut/
[6]:    http://freeglut.sourceforge.net/index.php#download
[7]:    http://libjpeg.sourceforge.net/
[8]:    http://libpng.sourceforge.net
[9]:    http://www.libtiff.org
[10]:   http://www.openexr.com/
[12]:   https://www.opengl.org
[13]:   https://github.com/wdas/ptex
