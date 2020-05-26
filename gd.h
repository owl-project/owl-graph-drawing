// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <memory>

#include "graph.h"

enum LayouterMode
{
        Naive,RTX,LBVH,
};

struct Layouter
{
    Layouter(Graph& g,
             LayouterMode mode=LayouterMode::RTX,
             unsigned rebuildRTXAccelAfter=1);
   ~Layouter();

    void iterate();

    struct Impl;
    std::unique_ptr<Impl> impl;
};
