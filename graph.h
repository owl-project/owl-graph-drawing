// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <vector>

#include <visionaray/math/aabb.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>

typedef visionaray::vec2 Node;

struct Edge
{
    unsigned i1;
    unsigned i2;
};

typedef visionaray::basic_aabb<float, 2> Bounds;

class Graph
{
public:
    std::vector<Node> nodes;
    std::vector<Edge> edges;

    std::vector<std::vector<Edge>> adjacentEdges;

    void append(Graph const& other);

    Bounds getBounds() const;

    void printStatistics(bool extended = false);

    //--- I/O ---------------------------------------------

    void loadArtificial(int numClusters = 80,
                        int nodesPerCluster = 50,
                        int edgesPerCluster = 100,
                        bool clustersAreConnected = true);
    void loadCompleteTree(int depth, int degree);
    void loadGephiCSV(std::string fileName);
    // Can be obtained from https://snap.stanford.edu/data/
    void loadDeezerCSV(std::string fileName);
    void saveTLP(const std::string& outFileName);

private:
    void buildAdjacentEdges();
    unsigned countConnectedComponents();
    unsigned countConnectedComponents(std::vector<unsigned>& countPerComponent);

};
