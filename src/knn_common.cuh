// Device helpers shared by the OptiX programs and the CUDA kernels: squared distances to points and
// boxes, and the register-resident KNN and cluster priority queues (Section 3.2).
//
// The queues are fixed-size arrays that are only ever indexed by compile-time constants (every loop
// over them is unrolled), so they stay in registers and never touch local memory.
#pragma once

#include <cuda_runtime.h>

#include "knn/knn.h"
#include "knn_shared.h"

#define KNN_DEVICE static __forceinline__ __device__

KNN_DEVICE float dist2(float3 a, float3 b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

KNN_DEVICE float dist2(float3 a, float4 b) { return dist2(a, make_float3(b.x, b.y, b.z)); }

// "Optimistic" distance: to the nearest point of the box, 0 inside it.
KNN_DEVICE float minDist2(float3 q, const Aabb& b)
{
    const float dx = fmaxf(fmaxf(b.lo.x - q.x, q.x - b.hi.x), 0.0f);
    const float dy = fmaxf(fmaxf(b.lo.y - q.y, q.y - b.hi.y), 0.0f);
    const float dz = fmaxf(fmaxf(b.lo.z - q.z, q.z - b.hi.z), 0.0f);
    return dx * dx + dy * dy + dz * dz;
}

// To the farthest corner of the box: every point inside is at least this close.
KNN_DEVICE float maxDist2(float3 q, const Aabb& b)
{
    const float dx = fmaxf(fabsf(q.x - b.lo.x), fabsf(q.x - b.hi.x));
    const float dy = fmaxf(fabsf(q.y - b.lo.y), fabsf(q.y - b.hi.y));
    const float dz = fmaxf(fabsf(q.z - b.lo.z), fabsf(q.z - b.hi.z));
    return dx * dx + dy * dy + dz * dz;
}

// "Pessimistic" distance (Figure 7): the smallest, over the six faces, of the distance to the farthest
// point of that face. Every face of a tight bounding box touches at least one of its points, so at
// least one point is this close.
KNN_DEVICE float pessimisticDist2(float3 q, const Aabb& b)
{
    const float nx = fminf(fabsf(q.x - b.lo.x), fabsf(q.x - b.hi.x));
    const float ny = fminf(fabsf(q.y - b.lo.y), fabsf(q.y - b.hi.y));
    const float nz = fminf(fabsf(q.z - b.lo.z), fabsf(q.z - b.hi.z));
    const float fx = fmaxf(fabsf(q.x - b.lo.x), fabsf(q.x - b.hi.x));
    const float fy = fmaxf(fabsf(q.y - b.lo.y), fabsf(q.y - b.hi.y));
    const float fz = fmaxf(fabsf(q.z - b.lo.z), fabsf(q.z - b.hi.z));
    const float fx2 = fx * fx, fy2 = fy * fy, fz2 = fz * fz;
    // The nearer x face, spanning the whole box in y and z; likewise for y and z.
    return fminf(fminf(nx * nx + fy2 + fz2, fx2 + ny * ny + fz2), fx2 + fy2 + nz * nz);
}

// The distance limit a KNN queue starts from: anything farther than the search radius is rejected
// by the same comparison that rejects points farther than the current Kth nearest.
KNN_DEVICE float queueLimit(float radius)
{
    const float r2 = radius * radius;
    return r2 < 3.0e38f ? __uint_as_float(__float_as_uint(r2) + 1u) : __int_as_float(0x7f800000);
}

// K nearest neighbours found so far, as squared distances in descending order: d[0] is the Kth
// nearest, the distance every candidate must beat. Slots from K on hold -1, which sorts after every
// real distance, so the insertion below never shifts past them and works for any runtime K.
struct KnnQueue
{
    float d[kMaxK];
    unsigned int id[kMaxK];

    __device__ __forceinline__ void init(unsigned int k, float limit)
    {
#pragma unroll
        for (unsigned int i = 0; i < kMaxK; ++i)
        {
            d[i]  = i < k ? limit : -1.0f;
            id[i] = kKnnInvalidId;
        }
    }

    __device__ __forceinline__ float worst() const { return d[0]; }

    // Caller checks d2 < worst() first.
    __device__ __forceinline__ void insert(float d2, unsigned int pointId)
    {
        bool placed = false;
#pragma unroll
        for (unsigned int i = 0; i + 1 < kMaxK; ++i)
        {
            if (!placed)
            {
                if (d[i + 1] > d2)
                {
                    d[i]  = d[i + 1];
                    id[i] = id[i + 1];
                }
                else
                {
                    d[i]   = d2;
                    id[i]  = pointId;
                    placed = true;
                }
            }
        }
        if (!placed)
        {
            d[kMaxK - 1]  = d2;
            id[kMaxK - 1] = pointId;
        }
    }

    // Writes the K results nearest first. Slots still holding the initial limit found nothing.
    __device__ __forceinline__ void write(unsigned int k, const unsigned int* pointIds, unsigned int* outIds,
                                          float* outDist2) const
    {
#pragma unroll
        for (unsigned int i = 0; i < kMaxK; ++i)
        {
            if (i < k)
            {
                const bool found     = id[i] != kKnnInvalidId;
                const unsigned int j = k - 1 - i;
                outIds[j]   = found ? (pointIds ? pointIds[id[i]] : id[i]) : kKnnInvalidId;
                outDist2[j] = found ? d[i] : __int_as_float(0x7f800000);
            }
        }
    }

    __device__ __forceinline__ unsigned int count(unsigned int k) const
    {
        unsigned int n = 0;
#pragma unroll
        for (unsigned int i = 0; i < kMaxK; ++i)
            n += (i < k && id[i] != kKnnInvalidId) ? 1u : 0u;
        return n;
    }
};

// Tests every point of cluster `c` against the queue; returns whether any was inserted.
KNN_DEVICE bool collectCluster(KnnQueue& queue, float3 q, const float4* points, unsigned int numPoints,
                               unsigned int c)
{
    const unsigned int begin = c * kClusterSize;
    const unsigned int end   = min(begin + kClusterSize, numPoints);
    bool inserted            = false;
    for (unsigned int p = begin; p < end; ++p)
    {
        const float d2 = dist2(q, points[p]);
        if (d2 < queue.worst())
        {
            queue.insert(d2, p);
            inserted = true;
        }
    }
    return inserted;
}

// The clusters of one cluster list that might hold neighbours, as (distance, cluster offset) pairs in
// ascending distance order (Section 3.2.1). Empty slots hold +inf.
struct ClusterQueue
{
    float d[kListSize];
    unsigned int c[kListSize];

    __device__ __forceinline__ void init()
    {
#pragma unroll
        for (unsigned int i = 0; i < kListSize; ++i)
        {
            d[i] = __int_as_float(0x7f800000);
            c[i] = 0;
        }
    }

    __device__ __forceinline__ void insert(float d2, unsigned int cluster)
    {
        bool placed = false;
#pragma unroll
        for (unsigned int i = kListSize - 1; i > 0; --i)
        {
            if (!placed)
            {
                if (d[i - 1] > d2)
                {
                    d[i] = d[i - 1];
                    c[i] = c[i - 1];
                }
                else
                {
                    d[i]   = d2;
                    c[i]   = cluster;
                    placed = true;
                }
            }
        }
        if (!placed)
        {
            d[0] = d2;
            c[0] = cluster;
        }
    }
};

// Visits the clusters of cluster list `list` nearest first, culling any farther than the current Kth
// nearest neighbour ("upward pruning"), and collects their points. Returns whether the queue changed.
KNN_DEVICE bool collectList(KnnQueue& queue, float3 q, const KnnLaunchParams& p, unsigned int list)
{
    const unsigned int first = list * kListSize;
    const unsigned int count = min(kListSize, p.numClusters - first);

    ClusterQueue clusters;
    clusters.init();
#pragma unroll
    for (unsigned int i = 0; i < kListSize; ++i)
    {
        if (i < count)
        {
            const float d2 = minDist2(q, p.clusters[first + i]);
            if (d2 < queue.worst())
                clusters.insert(d2, i);
        }
    }
    bool inserted = false;
#pragma unroll
    for (unsigned int i = 0; i < kListSize; ++i)
    {
        if (clusters.d[i] >= queue.worst())
            break;
        inserted |= collectCluster(queue, q, p.points, p.numPoints, first + clusters.c[i]);
    }
    return inserted;
}
