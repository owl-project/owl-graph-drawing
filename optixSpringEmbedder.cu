// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <owl/owl.h>
#include <owl/common/math/box.h>

#include <visionaray/math/vector.h>

#include "optixSpringEmbedder.h"

using namespace owl::common;

extern "C" __constant__ GraphGeom optixLaunchParams;

// TODO: dedup (gd.cu)
__device__ visionaray::vec2 frep(visionaray::vec2 u, visionaray::vec2 v, float k)
{
    visionaray::vec2 delta = v - u;
    float len = max(norm(delta), 2e-10f);
    float U = (2*k - len) > 0.f ? 1.f : 2e-10f;
    return (delta / len) * ((k*k)/len) * U;
}

struct PRD
{
    unsigned nodeID;
    unsigned thisNodeID;
};

OPTIX_INTERSECT_PROGRAM(SpringEmbedder)()
{
    const GraphGeom& graph = owl::getProgramData<GraphGeom>();
    int nodeID = optixGetPrimitiveIndex();

    PRD prd = owl::getPRD<PRD>();
    if (prd.thisNodeID == nodeID)
        return;

    vec3f thisNode = optixGetObjectRayOrigin();
    visionaray::vec2 v(thisNode.x,thisNode.y);
    visionaray::vec2 u = optixLaunchParams.nodes[nodeID];
    float radius = optixLaunchParams.k*2.f;

    if (length(v-u) <= radius)// && optixReportIntersection(optixGetRayTmax(),0))
    {
        prd.nodeID = (unsigned)nodeID;

        visionaray::vec2 posv = optixLaunchParams.nodes[prd.thisNodeID];
        visionaray::vec2 posu = optixLaunchParams.nodes[nodeID];

        optixLaunchParams.disp[prd.thisNodeID] += frep(posu, posv, optixLaunchParams.k);
    }
}

OPTIX_BOUNDS_PROGRAM(SpringEmbedder)(const void  *geomData,
                                     box3f       &primBounds,
                                     const int    primID)
{
    const GraphGeom& self = *(const GraphGeom*)geomData;
    visionaray::vec2 node = self.nodes[primID];
    float radius = self.k*2.f;
    vec3f min(node.x-radius,node.y-radius,-2e-10f);
    vec3f max(node.x+radius,node.y+radius,+2e-10f);
    //if (primID == 23)
    //    printf("(%f %f %f) (%f %f %f)\n",min.x,min.y,min.z,max.x,max.y,max.z);
    primBounds
        = box3f()
        .including(min)
        .including(max);
}

OPTIX_RAYGEN_PROGRAM(SpringEmbedder)()
{
    const vec2i pixelID = owl::getLaunchIndex();
    const vec2i launchDim = owl::getLaunchDims();

    if (pixelID.x >= optixLaunchParams.numNodes) return;
    if (pixelID.y != 0) return;

    visionaray::vec2 node = optixLaunchParams.nodes[pixelID.x];

    owl::Ray ray((float3)vec3f(node.x,node.y,0.f),
                 (float3)vec3f(1.f),
                 0.f,
                 2e-10f);

    optixLaunchParams.disp[pixelID.x] = visionaray::vec2(0.f);

    PRD prd;
    prd.thisNodeID = pixelID.x;
    owl::traceRay(optixLaunchParams.world,
                  ray,
                  prd,
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT);
    //printf(
    //    "%f %f\n",
    //    optixLaunchParams.disp[pixelID.x].x,
    //    optixLaunchParams.disp[pixelID.x].y);
}

OPTIX_CLOSEST_HIT_PROGRAM(SpringEmbedder)()
{
}

OPTIX_MISS_PROGRAM(SpringEmbedder)()
{
}
