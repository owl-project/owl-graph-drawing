// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "Compile w/ option --expt-extended-lambda"
#endif

#include <cstdint>
#include <iostream>
#include <ostream>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <visionaray/detail/stack.h> // TODO: detail
#include <visionaray/math/detail/math.h> // div_up
#include <visionaray/math/aabb.h>
#include <visionaray/morton.h>

#include <common/timer.h>

#include "lbvh.h"

using namespace visionaray;

// Stolen from owl
inline std::string prettyNumber(const size_t s)
{
  char buf[1000];
  if (s >= (1024LL*1024LL*1024LL*1024LL)) {
  	snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
  } else if (s >= (1024LL*1024LL*1024LL)) {
  	snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
  } else if (s >= (1024LL*1024LL)) {
  	snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
  } else if (s >= (1024LL)) {
  	snprintf(buf, 1000, "%.2fK",s/(1024.f));
  } else {
  	snprintf(buf,1000,"%zi",s);
  }
  return buf;
}

  
//-------------------------------------------------------------------------------------------------
// Stolen from https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
//

__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}


//-------------------------------------------------------------------------------------------------
// Compact brick data structure. max-corner is explicitly given by adding global brick-size
// to min-corner!
//

struct VSNRAY_ALIGN(16) MortonVertex
{
  unsigned id;
  unsigned morton_code;
};


//-------------------------------------------------------------------------------------------------
// Tree node that can be stored in device memory
//

struct LBVHNode
{
  basic_aabb<float, 2> bbox = {vec2(FLT_MAX),vec2(-FLT_MAX)};
  int left = -1;
  int right = -1;
  int parent = -1;
  int primID = -1;
};


//-------------------------------------------------------------------------------------------------
// Find node range that an inner node overlaps
//

__device__
vec2i determine_range(MortonVertex* verts, int num_verts, int i, int& split)
{
  auto delta = [&](int i, int j)
  {
    // Karras' delta(i,j) function
    // Denotes the length of the longest common
    // prefix between keys k_i and k_j

    // Cf. Figure 4: "for simplicity, we define that
    // delta(i,j) = -1 when j not in [0,n-1]"
    if (j < 0 || j >= num_verts)
      return -1;

    unsigned xord = verts[i].morton_code ^ verts[j].morton_code;
    if (xord == 0)
      return __clz((unsigned)i ^ (unsigned)j) + 32;
    else
      return __clz(verts[i].morton_code ^ verts[j].morton_code);
  };

  // Determine direction of the range (+1 or -1)
  int d = delta(i, i + 1) >= delta(i, i - 1) ? 1 : -1;

  // Compute upper bound for the length of the range
  int delta_min = delta(i, i - d);
  int l_max = 2;
  while (delta(i, i + l_max * d) > delta_min)
  {
    l_max *= 2;
  }

  // Find the other end using binary search
  int l = 0;
  for (int t = l_max >> 1; t >= 1; t >>= 1)
  {
    if (delta(i, i + (l + t) * d) > delta_min)
      l += t;
  }

  int j = i + l * d;

  // Find the split position using binary search
  int delta_node = delta(i, j);
  int s = 0;
  float divf = 2.f;
  int t = ceil(l / divf);
  for(; t >= 1; divf *= 2.f, t = ceil(l / divf))
  {
    if (delta(i, i + (s + t) * d) > delta_node)
      s += t;
  }

  split = i + s * d + min(d, 0);

  if (d == 1)
    return vec2i(i, j);
  else
    return vec2i(j, i);
}


//-------------------------------------------------------------------------------------------------
// Kernels
//

__global__ void computeFrameBounds(const Node* nodes,
                                   unsigned numNodes,
                                   basic_aabb<float, 2>* frameBoundsPtr)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: check if this might become a bottleneck
  if (index < numNodes)
  {
    atomicMin(&frameBoundsPtr->min.x, nodes[index].x);
    atomicMin(&frameBoundsPtr->min.y, nodes[index].y);
    atomicMax(&frameBoundsPtr->max.x, nodes[index].x);
    atomicMax(&frameBoundsPtr->max.y, nodes[index].y);
  }
}

__global__ void assignMortonCodes(MortonVertex* verts,
                                  const Node* nodes,
                                  unsigned numNodes,
                                  basic_aabb<float, 2>* frameBoundsPtr)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < numNodes)
  {
    const basic_aabb<float, 2>& frameBounds = *frameBoundsPtr;

    // Project node to [0..1]
    vec2 pt = nodes[index];
    pt -= frameBounds.center();
    pt = (pt + frameBounds.size() * .5f) / frameBounds.size();

    // Quantize to 10 bit (can do better than that..)
    // pt = min(max(pt * 1024.f, vec2(0.f)), vec2(1023.f));

    // Quantize to 16 bit
    pt = min(max(pt * 65536.f, vec2(0.f)), vec2(65535.f));

    MortonVertex& v = verts[index]; 
    v.id = index;
    v.morton_code = morton_encode2D((unsigned)pt.x, (unsigned)pt.y);
  }
}

__global__ void makeLeaves(MortonVertex* verts, unsigned num_verts, LBVHNode* leaves)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < num_verts)
  {
    leaves[index].primID = (int)verts[index].id;
  }
}

__global__ void nodeSplitting(MortonVertex* verts, int num_verts, LBVHNode* leaves, LBVHNode* inner)
{
  int num_leaves = num_verts;
  int num_inner = num_leaves - 1;

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < num_inner)
  {
    // NOTE: This is [first..last], not [first..last)!!
    int split = -1;
    vec2i range = determine_range(verts, num_verts, index, split);
    int first = range.x;
    int last = range.y;

    int left = split;
    int right = split + 1;

    if (left == first)
    {
      // left child is leaf
      inner[index].left = num_inner + left;
      leaves[left].parent = index;
    }
    else
    {
      // left child is inner
      inner[index].left = left;
      inner[left].parent = index;
    }

    if (right == last)
    {
      // right child is leaf
      inner[index].right = num_inner + right;
      leaves[right].parent = index;
    }
    else
    {
      // right child is inner
      inner[index].right = right;
      inner[right].parent = index;
    }
  }
}

__global__ void buildHierarchy(LBVHNode* inner,
        LBVHNode* leaves,
        const Node* nodes,
        unsigned numNodes,
        MortonVertex* verts)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= numNodes)
    return;

  // Leaf's bounding box
  basic_aabb<float, 2> bbox(nodes[verts[index].id] - vec2(2e-5f),
                            nodes[verts[index].id] + vec2(2e-5f));
  leaves[index].bbox = bbox;


  // Atomically combine child bounding boxes and update parents
  int next = leaves[index].parent;

  while (next >= 0)
  {
    atomicMin(&inner[next].bbox.min.x, bbox.min.x);
    atomicMin(&inner[next].bbox.min.y, bbox.min.y);
    atomicMax(&inner[next].bbox.max.x, bbox.max.x);
    atomicMax(&inner[next].bbox.max.y, bbox.max.y);
    next = inner[next].parent;
  }
}

__global__ void combineHierarchy(const LBVHNode* inner,
        int numInner,
        const LBVHNode* leaves,
        int numLeaves,
        LBVHNode* hierarchy)
{
  // Also set indices while we're at it!

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < numInner)
  {
    hierarchy[index] = inner[index];
  }

  if (index < numLeaves)
  {
    hierarchy[numInner + index] = leaves[index];
  }
}


//-------------------------------------------------------------------------------------------------
// BVH private implementation
//

struct BVH::Impl
{
  thrust::device_vector<LBVHNode> nodes;
};


//-------------------------------------------------------------------------------------------------
// BVH
//

BVH::BVH()
  : impl_(new Impl)
{
}

BVH::~BVH()
{
}

void BVH::build(const Node* nodes, unsigned numNodes)
{
#if BENCHMARK_MODE
  cuda::timer t;
#endif

  thrust::device_vector<MortonVertex> verts(numNodes);

  unsigned numThreads = 1024;

  // First compute frame bounds
  thrust::device_vector<basic_aabb<float, 2>> frameBoundsVector(1);
  computeFrameBounds<<<div_up(numNodes, numThreads), numThreads>>>(
      nodes,
      numNodes,
      thrust::raw_pointer_cast(frameBoundsVector.data()));
#if BENCHMARK_MODE
  cudaDeviceSynchronize();
  std::cout << "computeFrameBounds: " << t.elapsed() << '\n';
  t.reset();
#endif

  // Project on morton curve
  assignMortonCodes<<<div_up(numNodes, numThreads), numThreads>>>(
      thrust::raw_pointer_cast(verts.data()),
      nodes,
      numNodes,
      thrust::raw_pointer_cast(frameBoundsVector.data()));
#if BENCHMARK_MODE
  cudaDeviceSynchronize();
  std::cout << "assignMortonCodes: " << t.elapsed() << '\n';
  t.reset();
#endif

  // Sort on morton curve
  thrust::stable_sort(
      thrust::device,
      verts.begin(),
      verts.end(),
      [] __device__ (MortonVertex l, MortonVertex r)
      {
        return l.morton_code < r.morton_code;
      });
#if BENCHMARK_MODE
  cudaDeviceSynchronize();
  std::cout << "stable_sort: " << t.elapsed() << '\n';
  t.reset();
#endif

  thrust::device_vector<LBVHNode> leaves(numNodes);
  makeLeaves<<<div_up(numNodes, numThreads), numThreads>>>(
      thrust::raw_pointer_cast(verts.data()),
      numNodes,
      thrust::raw_pointer_cast(leaves.data()));
#if BENCHMARK_MODE
  cudaDeviceSynchronize();
  std::cout << "makeLeaves: " << t.elapsed() << '\n';
  t.reset();
#endif

  // Karras' algorithm
  thrust::device_vector<LBVHNode> inner(numNodes - 1);
  nodeSplitting<<<div_up(numNodes, numThreads), numThreads>>>(
      thrust::raw_pointer_cast(verts.data()),
      numNodes,
      thrust::raw_pointer_cast(leaves.data()),
      thrust::raw_pointer_cast(inner.data()));
#if BENCHMARK_MODE
  cudaDeviceSynchronize();
  std::cout << "nodeSplitting: " << t.elapsed() << '\n';
  t.reset();
#endif

  buildHierarchy<<<div_up(numNodes, numThreads), numThreads>>>(
      thrust::raw_pointer_cast(inner.data()),
      thrust::raw_pointer_cast(leaves.data()),
      nodes,
      numNodes,
      thrust::raw_pointer_cast(verts.data()));
#if BENCHMARK_MODE
  cudaDeviceSynchronize();
  std::cout << "buildHierarchy: " << t.elapsed() << '\n';
  t.reset();
#endif


  impl_->nodes.resize(inner.size() + leaves.size());
  combineHierarchy<<<div_up(numNodes, numThreads), numThreads>>>(
      thrust::raw_pointer_cast(inner.data()),
      inner.size(),
      thrust::raw_pointer_cast(leaves.data()),
      leaves.size(),
      thrust::raw_pointer_cast(impl_->nodes.data()));
#if BENCHMARK_MODE
  cudaDeviceSynchronize();
  std::cout << "combineHierarchy: " << t.elapsed() << '\n';
  t.reset();
#endif

#if BENCHMARK_MODE
    // That string can be parsed by our benchmark scripts
    size_t sizeInBytes = impl_->nodes.size() * sizeof(LBVHNode);
    std::cout << "@owl: LBVH build mem: 0.0K items, 0.0Mb bounds, 0.0Mb temp, " << prettyNumber(sizeInBytes) << "b initBVH, 0b finalBVH\n";
#endif
}

// TODO: dedup!
__device__ inline vec2 frep(vec2 u, vec2 v, float k)
{
    vec2 delta = v - u;
    float len = max(norm(delta), 2e-10f);
    float U = (2*k - len) > 0.f ? 1.f : 2e-10f;
    return (delta / len) * ((k*k)/len) * U;
    //return k / norm2(u - v) * normalize(v - u);
}

__device__ inline bool contains(const basic_aabb<float, 2>& bbox, const vec2& v)
{
    return v.x >= bbox.min.x && v.x <= bbox.max.x
        && v.y >= bbox.min.y && v.y <= bbox.max.y;
}

__device__ inline bool overlaps(const basic_aabb<float, 2>& L, const basic_aabb<float, 2>& R)
{
    return contains(L, R.min) || contains(L, R.max);
}

__global__ void repForcesLBVH(LBVHNode* bvhNodes, const Node* nodes, unsigned numNodes, vec2* disp, float k)
{
    unsigned nodeID = blockDim.x * blockIdx.x + threadIdx.x;

    if (nodeID >= numNodes)
        return;

    disp[nodeID] = vec2(0.f);

    detail::stack<32> st;
    st.push(0);

    basic_aabb<float, 2> bbox(nodes[nodeID] - vec2(k*2.f),
                              nodes[nodeID] + vec2(k*2.f));

next:
    while (!st.empty())
    {
        auto node = bvhNodes[st.pop()];

        while (node.left != -1 && node.right != -1)
        {
            LBVHNode children[2] = { bvhNodes[node.left], bvhNodes[node.right] };

            basic_aabb<float, 2> isectL = intersect(bbox, children[0].bbox);
            basic_aabb<float, 2> isectR = intersect(bbox, children[1].bbox);

            vec2f sizeL = isectL.size();
            vec2f sizeR = isectR.size();

            bool overlapsL = sizeL.x * sizeL.y > 0.f;
            bool overlapsR = sizeR.x * sizeR.y > 0.f;

            if (overlapsL && overlapsR)
            {
                st.push(node.left);
                node = children[1];
            }
            else if (overlapsL)
                node = children[0];
            else if (overlapsR)
                node = children[1];
            else
                goto next;
        }

        // traverse leaf
        assert(node.primID >= 0 && node.primID < (int)numNodes);

        vec2 posv = nodes[nodeID];
        vec2 posu = nodes[node.primID];

        if (length(posv-posu) <= 2.f*k)
        {
            disp[nodeID] += frep(posu, posv, k);
        }
    }
}

void BVH::computeRepForces(const Node* nodes, unsigned numNodes, vec2* disp, float k)
{
    repForcesLBVH<<<div_up(numNodes, unsigned(1024)), 1024>>>(
            thrust::raw_pointer_cast(impl_->nodes.data()),
            nodes,
            numNodes,
            disp,
            k
            );
}
