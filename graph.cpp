// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <map>
#include <numeric>
#include <ostream>
#include <random>
#include <sstream>
#include <stack>

#include <visionaray/math/math.h>

#include "graph.h"

using namespace visionaray;

void Graph::append(Graph const& other)
{
    size_t oldNumNodes = nodes.size();
    size_t oldNumEdges = edges.size();

    nodes.insert(nodes.end(), other.nodes.begin(), other.nodes.end());
    edges.insert(edges.end(), other.edges.begin(), other.edges.end());

    if (oldNumNodes == 0)
        return;

    for (size_t e = oldNumEdges; e < edges.size(); ++e)
    {
        edges[e].i1 += (unsigned)oldNumNodes;
        edges[e].i2 += (unsigned)oldNumNodes;
    }
}

Bounds Graph::getBounds() const
{
    Bounds bbox;
    bbox.invalidate();

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        bbox.insert(nodes[i]);
    }

    return bbox;
}

void Graph::printStatistics(bool extended)
{
    std::cout << "--- Graph Statistics ----------------------------------------\n";
    std::cout << "# Nodes: .............................. " << nodes.size() << '\n';
    std::cout << "# Edges: .............................. " << edges.size() << '\n';

    if (!extended)
        return;

    bool directed = false;

    std::vector<int> outgoing(nodes.size());
    std::vector<int> incoming(nodes.size());
    std::fill(outgoing.begin(), outgoing.end(), 0);
    std::fill(incoming.begin(), incoming.end(), 0);

    float area = 0.0;
    basic_aabb<float, 2> bbox;
    bbox.invalidate();

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        bbox.insert(nodes[i]);
    }

    float minEdgeLen =  FLT_MAX;
    float maxEdgeLen = -FLT_MAX;
    float accEdgeLen = 0.f;

    for (size_t i = 0; i < edges.size(); ++i)
    {
        Edge e = edges[i];
        outgoing[e.i1]++;
        incoming[e.i2]++;

        if (!directed)
        {
            outgoing[e.i2]++;
            incoming[e.i1]++;
        }

        float len = length(nodes[e.i1]-nodes[e.i2]);
        minEdgeLen = std::min(minEdgeLen, len);
        maxEdgeLen = std::max(maxEdgeLen, len);
        accEdgeLen += len;
    }

    std::sort(outgoing.begin(), outgoing.end());
    std::sort(incoming.begin(), incoming.end());

    int numOutgoing = std::accumulate(outgoing.begin(), outgoing.end(), 0);
    int numIncoming = std::accumulate(incoming.begin(), incoming.end(), 0);

    std::vector<unsigned> countPerComponent;
    unsigned cc = countConnectedComponents(countPerComponent);
    assert(cc = countPerComponent.size());
    std::sort(countPerComponent.begin(), countPerComponent.end());
    unsigned accCount = std::accumulate(countPerComponent.begin(), countPerComponent.end(), 0);

    std::cout << "Avg. outgoing edges per node: ......... " << numOutgoing/double(nodes.size()) << '\n';
    std::cout << "Max. outgoing edges per node: ......... " << outgoing.back() << '\n';
    std::cout << "Min. outgoing edges per node: ......... " << outgoing.front() << '\n';

    std::cout << "Avg. incoming edges per node: ......... " << numIncoming/double(nodes.size()) << '\n';
    std::cout << "Max. incoming edges per node: ......... " << incoming.back() << '\n';
    std::cout << "Min. incoming edges per node: ......... " << incoming.front() << '\n';

    std::cout << "Frame size: ........................... " << bbox.size() << '\n';

    std::cout << "Avg. edge length: ..................... " << accEdgeLen/float(edges.size()) << '\n';
    std::cout << "Max. edge length: ..................... " << maxEdgeLen << '\n';
    std::cout << "Min. edge length: ..................... " << minEdgeLen << '\n';

    std::cout << "# connected components: ............... " << cc << '\n';
    std::cout << "Avg. nodes per connected component: ... " << accCount/double(cc) << '\n';
    std::cout << "Max. nodes per connected component: ... " << countPerComponent.back() << '\n';
    std::cout << "Min. nodes per connected component: ... " << countPerComponent.front() << '\n';
}

void Graph::buildAdjacentEdges()
{
    adjacentEdges.resize(nodes.size());
    for (auto& ae : adjacentEdges)
        ae.resize(0);

    for (size_t i = 0; i < edges.size(); ++i)
    {
        Edge e = edges[i];
        adjacentEdges[e.i1].push_back(e);
        adjacentEdges[e.i2].push_back(e);
    }
}

unsigned Graph::countConnectedComponents()
{
    std::vector<unsigned> ignore;
    unsigned count = countConnectedComponents(ignore);
    assert(count == ignore.size());
    return count;
}

unsigned Graph::countConnectedComponents(std::vector<unsigned>& countPerComponent)
{
    if (adjacentEdges.empty())
        buildAdjacentEdges();

    std::vector<bool> seen(nodes.size(), false);

    unsigned count = 0;

    for (size_t i = 0; i < seen.size(); ++i)
    {
        if (seen[i])
            continue;

        count++;

        unsigned vertexCount = 0;

        std::stack<size_t> st;

        st.push(i);

        while (!st.empty())
        {
            size_t s = st.top();
            st.pop();

            if (!seen[s])
            {
                seen[s] = true;
                vertexCount++;
            }

            for (Edge e : adjacentEdges[s])
            {
                unsigned other = e.i1;
                if (e.i1 == s)
                    other = e.i2;

                if (!seen[other])
                    st.push(other);
            }
        }

        countPerComponent.push_back(vertexCount);
    }

    return count;
}

//-------------------------------------------------------------------------------------------------
// I/O functions
//

void Graph::loadArtificial(int numClusters, int nodesPerCluster, int edgesPerCluster, bool clustersAreConnected)
{
    Graph temp;
    for (int c = 0; c < numClusters; ++c)
    {
        for (int n = 0; n < nodesPerCluster; ++n)
        {
            temp.nodes.emplace_back();
        }

        std::vector<bool> isConnected(nodesPerCluster * nodesPerCluster);
        std::fill(isConnected.begin(), isConnected.end(), false);

        std::default_random_engine rng;
        std::uniform_int_distribution<int> dist(0, nodesPerCluster - 1);
        for (int e = 0; e < edgesPerCluster; ++e)
        {
            int n1 = 0;
            int n2 = 0;

            for (;;)
            {
                n1 = dist(rng);
                n2 = dist(rng);

                if (n1 == n2)
                    continue;

                if (isConnected[n1 * nodesPerCluster + n2])
                    continue;

                isConnected[n1 * nodesPerCluster + n2] = true;
                isConnected[n2 * nodesPerCluster + n1] = true;

                unsigned real_n1 = n1 + (int)temp.nodes.size() - nodesPerCluster;
                unsigned real_n2 = n2 + (int)temp.nodes.size() - nodesPerCluster;

                temp.edges.push_back({real_n1,real_n2});

                break;
            }
        }

        constexpr int NumClustersToConnectWith = 4;
        if (clustersAreConnected && c >= NumClustersToConnectWith)
        {
            for (int nc = 0; nc < NumClustersToConnectWith; ++nc)
            {
                // Connect the clusters with one edge
                int n1 = dist(rng);
                int n2 = dist(rng);

                std::uniform_int_distribution<int> connDist(0, NumClustersToConnectWith);
                unsigned real_n1 = n1 + (c-connDist(rng)-1) * nodesPerCluster;
                unsigned real_n2 = n2 + c * nodesPerCluster;

                temp.edges.push_back({real_n1,real_n2});
            }
        }
    }

    append(temp);
}

void recursive_add_children(Graph& g, unsigned prev, int depth, int degree) {
  if (depth > 0) {
    for (int i = 0; i < degree; ++i) {
      g.nodes.emplace_back();
      g.edges.push_back({prev, (unsigned) g.nodes.size()-1});
      recursive_add_children(g, g.nodes.size()-1, depth - 1, degree);
    }
  }
}

void Graph::loadCompleteTree(int depth, int degree)
{
    //Root Node
    nodes.emplace_back();
    recursive_add_children(*this, nodes.size()-1, depth, degree);
}

static std::vector<std::string> stringSplit(std::string s, char delim)
{
    std::vector<std::string> result;

    std::istringstream stream(s);

    for (std::string token; std::getline(stream, token, delim); )
    {
        result.push_back(token);
    }

    return result;
}

void Graph::loadGephiCSV(std::string fileName)
{
    // Assumes "Source,Target,..." csv files
    std::ifstream in(fileName);

    std::string line;
    // ignore first;
    std::getline(in, line);

    std::map<std::string, unsigned> knownNodes;

    Graph temp;
    while (std::getline(in, line))
    {
        std::vector<std::string> columns = stringSplit(line, ',');

        auto sourceIt = knownNodes.find(columns[0]);
        unsigned sourceID;

        if (sourceIt == knownNodes.end())
        {
            sourceID = (unsigned)nodes.size();
            nodes.emplace_back();
            knownNodes.insert({columns[0],sourceID});
        }
        else
            sourceID = sourceIt->second;


        auto targetIt = knownNodes.find(columns[1]);
        unsigned targetID;

        if (targetIt == knownNodes.end())
        {
            targetID = (unsigned)nodes.size();
            nodes.emplace_back();
            knownNodes.insert({columns[1],targetID});
        }
        else
            targetID = targetIt->second;


        edges.push_back({sourceID,targetID});
    }
}

void Graph::loadDeezerCSV(std::string fileName)
{
    std::ifstream in(fileName);

    std::string line;
    // ignore first;
    std::getline(in, line);

    Graph temp;
    while (std::getline(in, line))
    {
        unsigned i, j;
        sscanf(line.c_str(), "%i, %i", &i, &j);

        temp.edges.push_back({i,j});

        if ((i + 1) > temp.nodes.size() || (j + 1) > temp.nodes.size())
        {
            temp.nodes.resize(std::max(i + 1, j + 1));
        }
    }

    append(temp);
}

void Graph::saveTLP(const std::string& outFilename)
{
    std::ofstream out(outFilename);
    
    out << "(tlp \"2.3\"" << std::endl;
    out << "(nb_nodes " << nodes.size() << ")" << std::endl;
    
    //out << "(cluster 0" << std::endl;

    out << "(nodes " << 0 << ".." << nodes.size()-1;
    out << ")" << std::endl;

    for(int eIdx=0; eIdx < edges.size(); eIdx++) {
	out << "(edge " << eIdx << " " << edges[eIdx].i1 << " " << edges[eIdx].i2 << ")" << std::endl;
    }

    out << "(property 0 layout \"viewLayout\"" << std::endl;
    out << "(default \"(886,346,302)\" \"()\")" << std::endl;
    for(int nIdx=0; nIdx < nodes.size(); nIdx++) {
    	out << "(node " << nIdx << " \"(" << nodes[nIdx].x << "," << nodes[nIdx].y << ",0" << ")\")" << std::endl; 
    }
    out << ")" << std::endl;

    //out << ")" << std::endl; //end cluster
    out << ")" << std::endl; //eof

}
