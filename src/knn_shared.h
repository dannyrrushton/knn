// Layouts shared by the host code, the CUDA kernels and the OptiX programs.
#pragma once

#include <optix_types.h>

#include <cuda_runtime.h>

// The two-level cluster hierarchy of Section 3.1.2: C points per cluster and M clusters per cluster
// list. Cluster lists are the primitives of the RT-core tree.
constexpr unsigned int kClusterSize  = 16;
constexpr unsigned int kListSize     = 16;
constexpr unsigned int kPointsPerList = kClusterSize * kListSize;

// Same memory layout as OptixAabb, so range-box arrays can be handed to optixAccelBuild directly.
struct Aabb
{
    float3 lo;
    float3 hi;
};

// Which queries a launch runs.
enum KnnProgram : int
{
    KNN_PROGRAM_FRS       = 0,
    KNN_PROGRAM_CLUSTERED = 1,
    KNN_PROGRAM_ANIMATED  = 2,
};

// Hit group records, selected with the trace SBT offset.
enum KnnHitGroup : unsigned int
{
    KNN_HIT_FRS       = 0,  // one point per primitive
    KNN_HIT_CLUSTERED = 1,  // one cluster list per primitive, collecting neighbours
    KNN_HIT_COUNT     = 2,  // one cluster list per primitive, counting points in range (Section 3.3.1)
    KNN_HIT_GROUP_COUNT
};

struct KnnLaunchParams
{
    OptixTraversableHandle handle;
    int program;  // KnnProgram

    const float3* queries;
    unsigned int numQueries;
    unsigned int k;
    float maxRadius;
    unsigned int iterations;  // ray-time steps for KNN_PROGRAM_ANIMATED

    const float4* points;       // Hilbert-sorted for clustered programs
    const unsigned int* pointIds;  // original index of each point; null means identity
    unsigned int numPoints;
    const Aabb* clusters;       // tight bounds of each cluster of kClusterSize points
    unsigned int numClusters;
    const Aabb* lists;          // tight bounds of each list of kListSize clusters
    unsigned int numLists;

    unsigned int* outIds;
    float* outDist2;
};
