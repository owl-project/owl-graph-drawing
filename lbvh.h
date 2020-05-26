// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <memory>

#include <visionaray/math/forward.h>

#include "graph.h"

class BVH
{
public:

  BVH();
 ~BVH();

  /*! Pass nodes in device global memory */
  void build(const Node* nodes, unsigned numNodes);

  /*! Pass nodes and disp in device global memory */
  void computeRepForces(const Node* nodes, unsigned numNodes, visionaray::vec2* disp, float k);

private:

  struct Impl;
  std::unique_ptr<Impl> impl_;

};

