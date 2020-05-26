// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <iostream>
#include <ostream>
#include <random>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <owl/owl.h>

#include <visionaray/math/math.h>

#include <common/timer.h>

#include "gd.h"
#include "graph.h"
#include "lbvh.h"

// device
#include "optixSpringEmbedder.h"

using namespace visionaray;

extern "C" const char embedded_spring_embedder_programs[];

struct DeviceGraph
{
    Node* nodes;
    size_t numNodes;
    Edge* edges;
    size_t numEdges;
};

// Simply lay out nodes on a 2D grid in order of their appearance
void init_layout(Graph& g)
{
    size_t size = (size_t)ceilf(sqrtf((float)g.nodes.size()));

    std::default_random_engine rng;
    std::uniform_int_distribution<size_t> dist(size_t(0), size - 1);

    for (size_t i = 0; i < g.nodes.size(); ++i)
    {
#if 1
        // At random
        size_t x = dist(rng);
        size_t y = dist(rng);
#else
        // Uniform grid
        size_t x = i / size;
        size_t y = i % size;
#endif

        g.nodes[i] = {(float)x, (float)y};
    }
}

// https://i11www.iti.kit.edu/_media/teaching/winter2016/graphvis/graphvis-ws16-v6.pdf
__device__ vec2 frep(vec2 u, vec2 v, float k)
{
    vec2 delta = v - u;
    float len = max(norm(delta), 2e-10f);
    float U = (2*k - len) > 0.f ? 1.f : 2e-10f;
    return (delta / len) * ((k*k)/len) * U;
    //return k / norm2(u - v) * normalize(v - u);
}

__device__ vec2 fatt(vec2 u, vec2 v, float k)
{
    vec2 delta = v - u;
    float len = max(norm(delta), 2e-10f);
    float len2 = len*len;
    return (delta / len) * (len2/k);
    //float l = 1.f; // what's that??
    //return k * log(norm(u - v) / l) * normalize(v - u);
}

__global__ void repForcesNaive(DeviceGraph dg, vec2* disp, float k)
{
    unsigned nodeID = blockDim.x * blockIdx.x + threadIdx.x;

    if (nodeID >= dg.numNodes)
        return;

    disp[nodeID] = vec2(0.f);

    for (size_t i = 0; i < dg.numNodes; ++i)
    {
        if (i == nodeID)
            continue;

        vec2 posv = dg.nodes[nodeID];
        vec2 posu = dg.nodes[i];

        disp[nodeID] += frep(posu, posv, k);

        //vec2 fff = frep(posu, posv);
        //if (length(posv-posu) < 5.f)
        //    printf("%f %f %f\n",fff.x,fff.y,length(posu-posv));
    }
}

__global__ void attForces(DeviceGraph dg, vec2* disp, float k)
{
    unsigned edgeID = blockDim.x * blockIdx.x + threadIdx.x;

    if (edgeID >= dg.numEdges)
        return;

    unsigned v = dg.edges[edgeID].i1;
    unsigned u = dg.edges[edgeID].i2;
    vec2 posv = dg.nodes[v];
    vec2 posu = dg.nodes[u];

    vec2 F = fatt(posu, posv, k);

    // this might be pretty bad :-/
    atomicAdd(&disp[v].x, -F.x);
    atomicAdd(&disp[v].y, -F.y);

    atomicAdd(&disp[u].x, F.x);
    atomicAdd(&disp[u].y, F.y);
}

__global__ void disperse(DeviceGraph dg, vec2* disp, float t)
{

    unsigned nodeID = blockDim.x * blockIdx.x + threadIdx.x;

    if (nodeID >= dg.numNodes)
        return;

    dg.nodes[nodeID] += normalize(disp[nodeID]) * fminf(length(disp[nodeID]), t);
}

struct Layouter::Impl
{
    
    Impl(Graph& g, LayouterMode mode, unsigned rebuildRTXAccelAfter)
        : graph(g)
        , mode(mode)
        , rebuildRTXAccelAfter(rebuildRTXAccelAfter)
        , d_nodes(g.nodes.size())
        , d_edges(g.edges.size())
    {
        init_layout(graph);

        Bounds bbox = graph.getBounds();

        float A = bbox.size().x * bbox.size().y;
        k = sqrtf(A / graph.nodes.size());

        temperature = bbox.size().x / 10.f;

        thrust::copy(graph.edges.begin(), graph.edges.end(), d_edges.begin());

        if (mode == LayouterMode::Naive)
        {
	    std::cout << "LayouterMode: Naive" << std::endl;
            thrust::copy(graph.nodes.begin(), graph.nodes.end(), d_nodes.begin());

            dg = DeviceGraph{
                thrust::raw_pointer_cast(d_nodes.data()),
                d_nodes.size(),
                thrust::raw_pointer_cast(d_edges.data()),
                d_edges.size()
                };
        }
        else if (mode == LayouterMode::RTX)
        {
	    std::cout << "LayouterMode: RTX" << std::endl;
            context = owlContextCreate();
            module = owlModuleCreate(context, embedded_spring_embedder_programs);

            // -------------------------------------------------------
            // set up empty user geom miss progs
            // -------------------------------------------------------
            OWLMissProg missProg = owlMissProgCreate(context,module,"SpringEmbedder",0,nullptr,-1);

            OWLVarDecl graphVars[] = {
                { "world",    OWL_GROUP,  OWL_OFFSETOF(GraphGeom,world)},
                { "nodes",    OWL_BUFPTR, OWL_OFFSETOF(GraphGeom,nodes)},
                { "disp",     OWL_BUFPTR, OWL_OFFSETOF(GraphGeom,disp)},
                { "numNodes", OWL_UINT,   OWL_OFFSETOF(GraphGeom,numNodes)},
                { "k",        OWL_FLOAT,  OWL_OFFSETOF(GraphGeom,k)},
                { /* sentinel to mark end of list */ }
            };

            graphType = owlGeomTypeCreate(context,
                                          OWL_GEOMETRY_USER,
                                          sizeof(GraphGeom),
                                          graphVars, -1);

            OWLVarDecl launchParamsVars[] = {
                { "world",    OWL_GROUP,  OWL_OFFSETOF(GraphGeom,world)},
                { "nodes",    OWL_BUFPTR, OWL_OFFSETOF(GraphGeom,nodes)},
                { "disp",     OWL_BUFPTR, OWL_OFFSETOF(GraphGeom,disp)},
                { "numNodes", OWL_UINT,   OWL_OFFSETOF(GraphGeom,numNodes)},
                { "k",        OWL_FLOAT,  OWL_OFFSETOF(GraphGeom,k)},
                { /* sentinel to mark end of list */ }
            };

            launchParams = owlParamsCreate(context,sizeof(GraphGeom),
                                           launchParamsVars, -1);

            raygenProg = owlRayGenCreate(context,module,"SpringEmbedder",0,nullptr,-1);

            owlGeomTypeSetBoundsProg(graphType, module, "SpringEmbedder");
            owlGeomTypeSetIntersectProg(graphType, 0, module, "SpringEmbedder");
            owlGeomTypeSetClosestHit(graphType, 0, module, "SpringEmbedder");

            owlBuildPrograms(context);
            owlBuildPipeline(context);

            rtx_nodes = owlDeviceBufferCreate(context,
                                              OWL_USER_TYPE(graph.nodes[0]),
                                              graph.nodes.size(),
                                              graph.nodes.data());

            owlParamsSetBuffer(launchParams, "nodes", rtx_nodes);
            owlParamsSet1ui(launchParams, "numNodes", graph.nodes.size());
            owlParamsSet1f(launchParams, "k", k);

            OWLGeom geom = owlGeomCreate(context, graphType);
            owlGeomSetPrimCount(geom, graph.nodes.size());

            owlGeomSetBuffer(geom, "nodes", rtx_nodes);
            owlGeomSet1ui(geom, "numNodes", graph.nodes.size());
            owlGeomSet1f(geom, "k", k);

            blasGroup = owlUserGeomGroupCreate(context, 1, &geom);
            tlasGroup = owlInstanceGroupCreate(context, 1);
            owlInstanceGroupSetChild(tlasGroup, 0, blasGroup);

            owlParamsSetGroup(launchParams, "world", tlasGroup);
            owlGeomSetGroup(geom, "world", tlasGroup);

            rtx_disp = owlDeviceBufferCreate(context,
                                             OWL_USER_TYPE(graph.nodes[0]),
                                             graph.nodes.size(),
                                             nullptr);
            owlParamsSetBuffer(launchParams, "disp", rtx_disp);

            dg = DeviceGraph{
                (Node*)owlBufferGetPointer(rtx_nodes,0),
                graph.nodes.size(),
                thrust::raw_pointer_cast(d_edges.data()),
                d_edges.size()
                };
        }
        else if (mode == LayouterMode::LBVH)
        {
	    std::cout << "LayouterMode: LBVH" << std::endl;
            thrust::copy(graph.nodes.begin(), graph.nodes.end(), d_nodes.begin());

            dg = DeviceGraph{
                thrust::raw_pointer_cast(d_nodes.data()),
                d_nodes.size(),
                thrust::raw_pointer_cast(d_edges.data()),
                d_edges.size()
                };
        }
        else
        {
            assert(0);
        }
    }

    void goRTX(double &buildTime, cuda::timer &t)
    {
        //Build
        if (iteration % rebuildRTXAccelAfter == 0)
        {
            owlGroupBuildAccel(blasGroup);
            owlGroupBuildAccel(tlasGroup);
        }
        else
        {   
            owlGroupRefitAccel(blasGroup);
            owlGroupRefitAccel(tlasGroup);
        }

        owlBuildSBT(context);

#if BENCHMARK_MODE
        buildTime = t.elapsed();
        t.reset();
#endif

        //Traverse
        owlLaunch2D(raygenProg,graph.nodes.size(),1,launchParams);
    }

    void goLBVH(vec2* disp, float k, double &buildTime, cuda::timer &t)
    {
        //Build
        lbvh.build(thrust::raw_pointer_cast(d_nodes.data()), d_nodes.size());

#if BENCHMARK_MODE
        buildTime = t.elapsed();
        t.reset();
#endif
        
        //Traverse
        lbvh.computeRepForces(thrust::raw_pointer_cast(d_nodes.data()), d_nodes.size(), disp, k);
    }


    Graph& graph;

    // Used with Naive mode
    thrust::device_vector<Node> d_nodes;
    thrust::device_vector<Edge> d_edges;

    // Used with RTX mode
    OWLContext context = 0;
    OWLModule module = 0;
    OWLBuffer rtx_nodes = 0;
    OWLBuffer rtx_disp = 0;
    OWLGeomType graphType = 0;
    OWLGroup blasGroup = 0;
    OWLGroup tlasGroup = 0;
    OWLLaunchParams launchParams = 0;
    OWLRayGen raygenProg = 0;

    unsigned rebuildRTXAccelAfter = 1;

    // Used with LBVH mode
    BVH lbvh;

    DeviceGraph dg;

    LayouterMode mode = LBVH;
    unsigned iteration = 0;
    float k = 1.f;
    float temperature = 1000.f;
};

Layouter::Layouter(Graph& g, LayouterMode mode, unsigned rebuildRTXAccelAfter)
    : impl(new Impl(g, mode, rebuildRTXAccelAfter))
{
}

Layouter::~Layouter()
{
}

void Layouter::iterate()
{
    
    double buildElapsed=0.0;
    double traverseElapsed=0.0;

    cuda::timer tBuildTraverse;   
    cuda::timer t;
    cuda::timer tTotal;

    vec2* dispPointer = nullptr;

    thrust::device_vector<vec2> d_disp;

    if (impl->mode == LayouterMode::Naive)
    {
        d_disp.resize(impl->d_nodes.size());
        dispPointer = thrust::raw_pointer_cast(d_disp.data());
#if BENCHMARK_MODE 
        buildElapsed = tBuildTraverse.elapsed();
        tBuildTraverse.reset();
#endif 
        repForcesNaive<<<div_up(impl->d_nodes.size(), size_t(1024)), 1024>>>(
                impl->dg,
                dispPointer,
                impl->k
                );
#if BENCHMARK_MODE 
        traverseElapsed = tBuildTraverse.elapsed();
        std::cout << "@owl: naive build mem: 0.0K items, 0.0Mb bounds, 0.0Mb temp, 0.0Mb initBVH, 0b finalBVH\n";
#endif 
    }
    else if (impl->mode == LayouterMode::RTX)
    {
        dispPointer = (vec2*)owlBufferGetPointer(impl->rtx_disp,0);
        impl->goRTX(buildElapsed, tBuildTraverse);
    }
    else if (impl->mode == LayouterMode::LBVH)
    {
        d_disp.resize(impl->d_nodes.size());
        dispPointer = thrust::raw_pointer_cast(d_disp.data());
        impl->goLBVH(dispPointer, impl->k, buildElapsed, tBuildTraverse);
    }
#if BENCHMARK_MODE
    traverseElapsed = t.elapsed();
    static double ms = 0.;
    double currentElapsed = t.elapsed();
    ms += currentElapsed;
    std::cout << "Iteration: " << (impl->iteration+1) << '\n';
    std::cout << "TimeRepulsiveForce: " << currentElapsed << '\n';
    std::cout << "AvgTimeRepulsiveForce: " << ms/(impl->iteration+1) << '\n';
    
    std::cout << "TimeBuildAccel: " << buildElapsed << '\n';
    std::cout << "TimeTraverseAccel: " << traverseElapsed << '\n';
    std::cout << "NumNodes: " << impl->d_nodes.size() << '\n';
    std::cout << "NumEdges: " << impl->d_edges.size() << '\n';
    std::cout << "kValue: " << impl->k << '\n';
    std::cout << "Temperature: " << impl->temperature << '\n';
    t.reset();
#endif

    attForces<<<div_up(impl->d_edges.size(), size_t(1024)), 1024>>>(
            impl->dg,
            dispPointer,
            impl->k
            );
#if BENCHMARK_MODE
    std::cout << "TimeAttractiveForces: " << t.elapsed() << '\n';
    t.reset();
#endif

    disperse<<<div_up(impl->d_nodes.size(), size_t(1024)), 1024>>>(
            impl->dg,
            dispPointer,
            impl->temperature
            );
#if BENCHMARK_MODE
    std::cout << "TimeDisperse: " << t.elapsed() << '\n';
    t.reset();
    std::cout << "Elapsed: " << tTotal.elapsed() << '\n';
#endif

    if (impl->mode == LayouterMode::Naive || impl->mode == LayouterMode::LBVH)
    {
        thrust::copy(impl->d_nodes.begin(), impl->d_nodes.end(), impl->graph.nodes.begin());
    }
    else if (impl->mode == LayouterMode::RTX)
    {
        vec2* pointer = (vec2*)owlBufferGetPointer(impl->rtx_nodes,0);
        cudaMemcpy(impl->graph.nodes.data(),
                   pointer,
                   sizeof(vec2)*impl->graph.nodes.size(),
                   cudaMemcpyDeviceToHost);
    }
#if BENCHMARK_MODE
    std::cout << "TimeCopyBack: " << t.elapsed() << '\n';
    std::cout << "\n";
    t.reset();
#endif

    //impl->temperature /= 1.618f;
    impl->temperature /= 1.001f;
    impl->temperature = max(impl->temperature, 1e-4f);

    impl->iteration++;
}
