// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <visionaray/math/forward.h>

struct GraphGeom
{
    OptixTraversableHandle world;
    visionaray::vec2* nodes;
    visionaray::vec2* disp;
    unsigned numNodes;
    float k;
};
