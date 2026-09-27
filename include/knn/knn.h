// K-nearest-neighbour queries over 3D points using RT cores, after Morrical et al., "Hardware
// Accelerated KNN Queries using Ray-Time Marching" (HPG 2023).
//
// Every method answers the same question: for each query point, the (up to) K data points nearest
// to it that lie within the maximum search radius given at build time. Results are written as K
// entries per query, nearest first; entries past the number of neighbours found have id
// kKnnInvalidId and distance INFINITY.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

// Largest supported K: the KNN priority queue lives in the 32 ray payload registers (Section 3.2.2).
constexpr unsigned int kMaxK         = 16;
constexpr unsigned int kKnnInvalidId = 0xffffffffu;

enum class KnnMethod
{
    FRS,           // Fast Radius Search (Evangelou et al. 2021): one range box per point
    ClusteredFRS,  // Hilbert-clustered FRS with cluster and payload priority queues (Sections 3.1-3.2)
    AnimatedFRS,   // Clustered FRS over motion-blur range boxes, with ray-time marching (Section 3.3)
    SoftwareLBVH,  // software LBVH traversal on CUDA cores, in the style of Jakob and Guthe 2021
};

const char* knnMethodName(KnnMethod method);

struct KnnBuildResult
{
    double milliseconds  = 0.0;  // GPU time of the whole build
    size_t treeBytes     = 0;    // acceleration structure (OptiX GAS or LBVH nodes)
    size_t totalBytes    = 0;    // tree plus sorted points, cluster bounds and scratch space in use
    unsigned int primitives = 0; // primitives the tree was built over (points, cluster lists or clusters)
};

class KnnIndex
{
public:
    KnnIndex();
    ~KnnIndex();
    KnnIndex(const KnnIndex&)            = delete;
    KnnIndex& operator=(const KnnIndex&) = delete;

    // Number of ray-time steps the Animated FRS range search marches through (default 8).
    void setIterations(unsigned int iterations);
    unsigned int iterations() const;

    // Builds the index over `numPoints` device points. The points are copied, so the array may be
    // reused once this returns. Synchronous: the result's time covers all the GPU work.
    KnnBuildResult build(KnnMethod method, const float3* dPoints, unsigned int numPoints, float maxRadius);

    // Finds the k nearest neighbours of each device query point, writing numQueries * k ids (indices
    // into the points given to build) and squared distances. Synchronous; returns GPU milliseconds.
    double query(const float3* dQueries, unsigned int numQueries, unsigned int k, unsigned int* dIds,
                 float* dDist2);

    KnnMethod method() const;
    float maxRadius() const;
    unsigned int numPoints() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

// Exhaustive reference search on CUDA cores, with the same semantics as KnnIndex::query. Synchronous;
// returns GPU milliseconds.
double knnBruteForce(const float3* dPoints, unsigned int numPoints, const float3* dQueries,
                     unsigned int numQueries, unsigned int k, float maxRadius, unsigned int* dIds,
                     float* dDist2);
