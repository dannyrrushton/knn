// Host-side internals shared by knn.cpp and knn_kernels.cu.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include "knn_shared.h"

#define KNN_CUDA_CHECK(call)                                                                        \
    do                                                                                              \
    {                                                                                               \
        const cudaError_t knnErr_ = (call);                                                         \
        if (knnErr_ != cudaSuccess)                                                                 \
        {                                                                                           \
            std::ostringstream knnMsg_;                                                             \
            knnMsg_ << "CUDA error " << cudaGetErrorString(knnErr_) << " at " << __FILE__ << ":"   \
                    << __LINE__ << ": " #call;                                                      \
            throw std::runtime_error(knnMsg_.str());                                                \
        }                                                                                           \
    } while (0)

namespace knn
{

// A device allocation that only grows, so per-frame rebuilds don't allocate once warmed up.
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { cudaFree(m_ptr); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void reserve(size_t bytes)
    {
        if (bytes <= m_capacity)
            return;
        KNN_CUDA_CHECK(cudaFree(m_ptr));
        m_ptr = nullptr;
        m_capacity = 0;
        KNN_CUDA_CHECK(cudaMalloc(&m_ptr, bytes));
        m_capacity = bytes;
    }

    template <class T>
    T* get(size_t count)
    {
        reserve(count * sizeof(T));
        return static_cast<T*>(m_ptr);
    }

    template <class T>
    T* as() const { return static_cast<T*>(m_ptr); }

    size_t capacity() const { return m_capacity; }

private:
    void* m_ptr       = nullptr;
    size_t m_capacity = 0;
};

// Software LBVH node (Karras 2012). Children with kLbvhLeaf set are clusters.
struct LbvhNode
{
    Aabb box;
    unsigned int left;
    unsigned int right;
};
constexpr unsigned int kLbvhLeaf = 0x80000000u;

inline unsigned int divUp(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

// --- knn_kernels.cu -------------------------------------------------------------------------------

// Bounds of all points, written to dBounds[0].
void computeBounds(const float3* points, unsigned int n, Aabb* dBounds, DeviceBuffer& temp, cudaStream_t stream);

// Copies points into float4s, for FRS (no reordering).
void copyPoints(const float3* points, unsigned int n, float4* out, cudaStream_t stream);

// Sorts points along a 3D Hilbert curve over dBounds. Writes the sorted points, the original index of
// each, and the sorted Hilbert keys.
void hilbertSort(const float3* points, unsigned int n, const Aabb* dBounds, float4* sortedPoints,
                 unsigned int* sortedIds, uint64_t* sortedKeys, DeviceBuffer& scratch, DeviceBuffer& temp,
                 cudaStream_t stream);

// Tight bounds of every group of `groupSize` consecutive points / boxes (groupSize must be 16).
void groupBounds(const float4* points, unsigned int n, Aabb* out, cudaStream_t stream);
void groupBounds(const Aabb* boxes, unsigned int n, Aabb* out, cudaStream_t stream);

// Range boxes: every input box (or point) padded by `pad` on each side.
void padBoxes(const Aabb* boxes, unsigned int n, float pad, Aabb* out, cudaStream_t stream);
void pointBoxes(const float4* points, unsigned int n, float pad, Aabb* out, cudaStream_t stream);

// Karras LBVH over clusters, keyed by the Hilbert key of each cluster's first point. `nodes` holds
// numClusters - 1 internal nodes, root first; `scratch` holds parents and counters.
void buildLbvh(const uint64_t* sortedKeys, const Aabb* clusters, unsigned int numClusters, LbvhNode* nodes,
               DeviceBuffer& scratch, cudaStream_t stream);

struct LbvhQuery
{
    const LbvhNode* nodes;
    const Aabb* clusters;
    unsigned int numClusters;
    const float4* points;
    const unsigned int* pointIds;
    unsigned int numPoints;
    const float3* queries;
    unsigned int numQueries;
    unsigned int k;
    float maxRadius;
    unsigned int* outIds;
    float* outDist2;
};
void queryLbvh(const LbvhQuery& q, cudaStream_t stream);

void bruteForce(const float3* points, unsigned int numPoints, const float3* queries, unsigned int numQueries,
                unsigned int k, float maxRadius, unsigned int* outIds, float* outDist2, cudaStream_t stream);

// Marks every result as not found.
void clearResults(unsigned int* ids, float* dist2, size_t count, cudaStream_t stream);

}  // namespace knn
