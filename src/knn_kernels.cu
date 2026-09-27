// CUDA kernels for building KNN indices (Hilbert sort, cluster hierarchy, range boxes), the software
// LBVH baseline, and the brute-force reference.
#include <cub/device/device_radix_sort.cuh>

#include <cfloat>

#include "knn_common.cuh"
#include "knn_internal.h"

namespace knn
{
namespace
{

constexpr unsigned int kBlock = 256;

inline unsigned int blocksFor(size_t n) { return static_cast<unsigned int>((n + kBlock - 1) / kBlock); }

__device__ __forceinline__ Aabb emptyBox()
{
    return {make_float3(FLT_MAX, FLT_MAX, FLT_MAX), make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX)};
}

__device__ __forceinline__ Aabb merge(const Aabb& a, const Aabb& b)
{
    return {make_float3(fminf(a.lo.x, b.lo.x), fminf(a.lo.y, b.lo.y), fminf(a.lo.z, b.lo.z)),
            make_float3(fmaxf(a.hi.x, b.hi.x), fmaxf(a.hi.y, b.hi.y), fmaxf(a.hi.z, b.hi.z))};
}

// Reduces a box over each aligned group of 16 lanes; every lane of the warp must participate.
__device__ __forceinline__ Aabb reduce16(Aabb b)
{
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
    {
        Aabb o;
        o.lo.x = __shfl_xor_sync(0xffffffffu, b.lo.x, offset, 16);
        o.lo.y = __shfl_xor_sync(0xffffffffu, b.lo.y, offset, 16);
        o.lo.z = __shfl_xor_sync(0xffffffffu, b.lo.z, offset, 16);
        o.hi.x = __shfl_xor_sync(0xffffffffu, b.hi.x, offset, 16);
        o.hi.y = __shfl_xor_sync(0xffffffffu, b.hi.y, offset, 16);
        o.hi.z = __shfl_xor_sync(0xffffffffu, b.hi.z, offset, 16);
        b      = merge(b, o);
    }
    return b;
}

// --- Scene bounds -----------------------------------------------------------------------------------

// Floats mapped to unsigned ints that order the same way, for atomicMin/atomicMax.
__device__ __forceinline__ unsigned int orderedBits(float f)
{
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float fromOrderedBits(unsigned int u)
{
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}

__global__ void initBoundsKernel(unsigned int* bits)
{
    if (threadIdx.x < 3)
        bits[threadIdx.x] = 0xffffffffu;
    else if (threadIdx.x < 6)
        bits[threadIdx.x] = 0u;
}

__global__ void boundsKernel(const float3* points, unsigned int n, unsigned int* bits)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    Aabb b               = emptyBox();
    if (i < n)
        b = {points[i], points[i]};
    b = reduce16(b);
    // Lane 0 of each half-warp, then lane 0 of the warp.
    Aabb o;
    o.lo.x = __shfl_down_sync(0xffffffffu, b.lo.x, 16);
    o.lo.y = __shfl_down_sync(0xffffffffu, b.lo.y, 16);
    o.lo.z = __shfl_down_sync(0xffffffffu, b.lo.z, 16);
    o.hi.x = __shfl_down_sync(0xffffffffu, b.hi.x, 16);
    o.hi.y = __shfl_down_sync(0xffffffffu, b.hi.y, 16);
    o.hi.z = __shfl_down_sync(0xffffffffu, b.hi.z, 16);
    b      = merge(b, o);
    if ((threadIdx.x & 31) == 0 && b.lo.x <= b.hi.x)
    {
        atomicMin(&bits[0], orderedBits(b.lo.x));
        atomicMin(&bits[1], orderedBits(b.lo.y));
        atomicMin(&bits[2], orderedBits(b.lo.z));
        atomicMax(&bits[3], orderedBits(b.hi.x));
        atomicMax(&bits[4], orderedBits(b.hi.y));
        atomicMax(&bits[5], orderedBits(b.hi.z));
    }
}

__global__ void finishBoundsKernel(const unsigned int* bits, Aabb* out)
{
    out->lo = make_float3(fromOrderedBits(bits[0]), fromOrderedBits(bits[1]), fromOrderedBits(bits[2]));
    out->hi = make_float3(fromOrderedBits(bits[3]), fromOrderedBits(bits[4]), fromOrderedBits(bits[5]));
}

// --- Hilbert sort -----------------------------------------------------------------------------------

constexpr unsigned int kHilbertBits = 21;  // per axis: 63-bit keys

// Hilbert index of a point on a 2^21 grid, using Skilling's transpose algorithm ("Programming the
// Hilbert curve", 2004), with the transposed bits interleaved into one key.
__device__ uint64_t hilbertKey(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int X[3] = {x, y, z};
    const unsigned int M = 1u << (kHilbertBits - 1);
    for (unsigned int Q = M; Q > 1; Q >>= 1)
    {
        const unsigned int P = Q - 1;
#pragma unroll
        for (int i = 0; i < 3; ++i)
        {
            if (X[i] & Q)
                X[0] ^= P;  // invert
            else
            {
                const unsigned int t = (X[0] ^ X[i]) & P;  // exchange
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }
    X[1] ^= X[0];  // Gray encode
    X[2] ^= X[1];
    unsigned int t = 0;
    for (unsigned int Q = M; Q > 1; Q >>= 1)
        if (X[2] & Q)
            t ^= Q - 1;
    X[0] ^= t;
    X[1] ^= t;
    X[2] ^= t;

    uint64_t key = 0;
    for (int b = kHilbertBits - 1; b >= 0; --b)
    {
        key = (key << 3) | (static_cast<uint64_t>((X[0] >> b) & 1u) << 2) |
              (static_cast<uint64_t>((X[1] >> b) & 1u) << 1) | static_cast<uint64_t>((X[2] >> b) & 1u);
    }
    return key;
}

__device__ __forceinline__ unsigned int quantize(float v, float lo, float scale)
{
    const float cell = (v - lo) * scale;
    return static_cast<unsigned int>(fminf(fmaxf(cell, 0.0f), static_cast<float>((1u << kHilbertBits) - 1)));
}

__global__ void hilbertKeysKernel(const float3* points, unsigned int n, const Aabb* bounds, uint64_t* keys,
                                  unsigned int* ids)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const Aabb b       = *bounds;
    const float extent = fmaxf(fmaxf(b.hi.x - b.lo.x, b.hi.y - b.lo.y), fmaxf(b.hi.z - b.lo.z, 1e-30f));
    const float scale  = static_cast<float>(1u << kHilbertBits) / extent;
    const float3 p     = points[i];
    keys[i] = hilbertKey(quantize(p.x, b.lo.x, scale), quantize(p.y, b.lo.y, scale), quantize(p.z, b.lo.z, scale));
    ids[i]  = i;
}

__global__ void gatherPointsKernel(const float3* points, const unsigned int* ids, unsigned int n, float4* out)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float3 p = points[ids[i]];
    out[i]         = make_float4(p.x, p.y, p.z, 0.0f);
}

__global__ void copyPointsKernel(const float3* points, unsigned int n, float4* out)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float3 p = points[i];
    out[i]         = make_float4(p.x, p.y, p.z, 0.0f);
}

// --- Cluster hierarchy and range boxes ---------------------------------------------------------------

// Launched over whole warps: lanes past n contribute an empty box.
__global__ void pointGroupBoundsKernel(const float4* points, unsigned int n, Aabb* out)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    Aabb b               = emptyBox();
    if (i < n)
    {
        const float4 p = points[i];
        b              = {make_float3(p.x, p.y, p.z), make_float3(p.x, p.y, p.z)};
    }
    b = reduce16(b);
    if (i < n && (i % 16) == 0)
        out[i / 16] = b;
}

__global__ void boxGroupBoundsKernel(const Aabb* boxes, unsigned int n, Aabb* out)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    Aabb b               = i < n ? boxes[i] : emptyBox();
    b                    = reduce16(b);
    if (i < n && (i % 16) == 0)
        out[i / 16] = b;
}

// Rounds away from the box, so a range box always contains every point within `pad` of its contents.
__device__ __forceinline__ Aabb padOutwards(float3 lo, float3 hi, float pad)
{
    return {make_float3(__fsub_rd(lo.x, pad), __fsub_rd(lo.y, pad), __fsub_rd(lo.z, pad)),
            make_float3(__fadd_ru(hi.x, pad), __fadd_ru(hi.y, pad), __fadd_ru(hi.z, pad))};
}

__global__ void padBoxesKernel(const Aabb* boxes, unsigned int n, float pad, Aabb* out)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const Aabb b = boxes[i];
    out[i]       = padOutwards(b.lo, b.hi, pad);
}

__global__ void pointBoxesKernel(const float4* points, unsigned int n, float pad, Aabb* out)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float4 p = points[i];
    const float3 c = make_float3(p.x, p.y, p.z);
    out[i]         = padOutwards(c, c, pad);
}

// --- Software LBVH (Karras 2012) ---------------------------------------------------------------------

// Length of the common prefix of the keys of leaves i and j, with the index breaking ties.
__device__ __forceinline__ int commonPrefix(const uint64_t* keys, int numLeaves, int i, int j)
{
    if (j < 0 || j >= numLeaves)
        return -1;
    const uint64_t ki = keys[static_cast<size_t>(i) * kClusterSize];
    const uint64_t kj = keys[static_cast<size_t>(j) * kClusterSize];
    if (ki == kj)
        return 64 + __clz(static_cast<unsigned int>(i ^ j));
    return __clzll(static_cast<long long>(ki ^ kj));
}

__global__ void lbvhInternalKernel(const uint64_t* keys, int numLeaves, LbvhNode* nodes, unsigned int* internalParent,
                                   unsigned int* leafParent, unsigned int* counters)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLeaves - 1)
        return;
    counters[i] = 0;

    // Direction and extent of the range of leaves this node covers.
    const int d    = commonPrefix(keys, numLeaves, i, i + 1) - commonPrefix(keys, numLeaves, i, i - 1) > 0 ? 1 : -1;
    const int dMin = commonPrefix(keys, numLeaves, i, i - d);
    int lMax       = 2;
    while (commonPrefix(keys, numLeaves, i, i + lMax * d) > dMin)
        lMax *= 2;
    int l = 0;
    for (int t = lMax / 2; t >= 1; t /= 2)
        if (commonPrefix(keys, numLeaves, i, i + (l + t) * d) > dMin)
            l += t;
    const int j     = i + l * d;
    const int dNode = commonPrefix(keys, numLeaves, i, j);

    // Split position.
    int s   = 0;
    int div = 2;
    int t;
    do
    {
        t = (l + div - 1) / div;
        if (commonPrefix(keys, numLeaves, i, i + (s + t) * d) > dNode)
            s += t;
        div *= 2;
    } while (t > 1);
    const int gamma = i + s * d + min(d, 0);

    const bool leftLeaf  = min(i, j) == gamma;
    const bool rightLeaf = max(i, j) == gamma + 1;
    nodes[i].left  = leftLeaf ? (gamma | kLbvhLeaf) : gamma;
    nodes[i].right = rightLeaf ? ((gamma + 1) | kLbvhLeaf) : gamma + 1;
    if (leftLeaf)
        leafParent[gamma] = i;
    else
        internalParent[gamma] = i;
    if (rightLeaf)
        leafParent[gamma + 1] = i;
    else
        internalParent[gamma + 1] = i;
}

__device__ __forceinline__ Aabb loadCoherent(const Aabb* box)
{
    const float* f = reinterpret_cast<const float*>(box);
    return {make_float3(__ldcg(f + 0), __ldcg(f + 1), __ldcg(f + 2)), make_float3(__ldcg(f + 3), __ldcg(f + 4), __ldcg(f + 5))};
}

// Bottom-up bounds: the second child to arrive at a node computes its box.
__global__ void lbvhBoundsKernel(const Aabb* clusters, unsigned int numLeaves, LbvhNode* nodes,
                                 const unsigned int* internalParent, const unsigned int* leafParent,
                                 unsigned int* counters)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLeaves)
        return;
    unsigned int node = leafParent[i];
    while (true)
    {
        if (atomicAdd(&counters[node], 1u) == 0)
            return;
        __threadfence();
        const LbvhNode n = nodes[node];
        const Aabb l     = (n.left & kLbvhLeaf) ? clusters[n.left & ~kLbvhLeaf] : loadCoherent(&nodes[n.left].box);
        const Aabb r     = (n.right & kLbvhLeaf) ? clusters[n.right & ~kLbvhLeaf] : loadCoherent(&nodes[n.right].box);
        nodes[node].box  = merge(l, r);
        __threadfence();
        if (node == 0)
            return;
        node = internalParent[node];
    }
}

constexpr int kLbvhStackSize = 128;

// Depth-first, near-to-far traversal with a register KNN queue. Subtrees are culled against the
// current Kth nearest distance both when pushed and when popped (upward pruning).
__global__ void lbvhQueryKernel(LbvhQuery p)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= p.numQueries)
        return;
    const float3 q = p.queries[index];
    KnnQueue queue;
    queue.init(p.k, queueLimit(p.maxRadius));

    if (p.numClusters == 1)
    {
        collectCluster(queue, q, p.points, p.numPoints, 0);
    }
    else
    {
        unsigned int stackNode[kLbvhStackSize];
        float stackDist[kLbvhStackSize];
        int sp           = 0;
        stackNode[sp]    = 0;
        stackDist[sp++]  = 0.0f;
        while (sp > 0)
        {
            --sp;
            if (stackDist[sp] >= queue.worst())
                continue;
            const unsigned int node = stackNode[sp];
            if (node & kLbvhLeaf)
            {
                collectCluster(queue, q, p.points, p.numPoints, node & ~kLbvhLeaf);
                continue;
            }
            const LbvhNode n = p.nodes[node];
            const float dl   = minDist2(q, (n.left & kLbvhLeaf) ? p.clusters[n.left & ~kLbvhLeaf] : p.nodes[n.left].box);
            const float dr = minDist2(q, (n.right & kLbvhLeaf) ? p.clusters[n.right & ~kLbvhLeaf] : p.nodes[n.right].box);
            const bool leftFirst    = dl <= dr;
            const unsigned int near = leftFirst ? n.left : n.right;
            const unsigned int far  = leftFirst ? n.right : n.left;
            const float dNear       = leftFirst ? dl : dr;
            const float dFar        = leftFirst ? dr : dl;
            if (dFar < queue.worst() && sp < kLbvhStackSize)
            {
                stackNode[sp]   = far;
                stackDist[sp++] = dFar;
            }
            if (dNear < queue.worst() && sp < kLbvhStackSize)
            {
                stackNode[sp]   = near;
                stackDist[sp++] = dNear;
            }
        }
    }

    const size_t base = static_cast<size_t>(index) * p.k;
    queue.write(p.k, p.pointIds, p.outIds + base, p.outDist2 + base);
}

// --- Brute force -------------------------------------------------------------------------------------

__global__ void bruteForceKernel(const float3* points, unsigned int numPoints, const float3* queries,
                                 unsigned int numQueries, unsigned int k, float maxRadius, unsigned int* outIds,
                                 float* outDist2)
{
    __shared__ float3 tile[kBlock];
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const float3 q           = index < numQueries ? queries[index] : make_float3(0.0f, 0.0f, 0.0f);
    KnnQueue queue;
    queue.init(k, queueLimit(maxRadius));
    for (unsigned int base = 0; base < numPoints; base += kBlock)
    {
        __syncthreads();
        if (base + threadIdx.x < numPoints)
            tile[threadIdx.x] = points[base + threadIdx.x];
        __syncthreads();
        const unsigned int count = min(kBlock, numPoints - base);
        for (unsigned int j = 0; j < count; ++j)
        {
            const float d2 = dist2(q, tile[j]);
            if (d2 < queue.worst())
                queue.insert(d2, base + j);
        }
    }
    if (index < numQueries)
    {
        const size_t out = static_cast<size_t>(index) * k;
        queue.write(k, nullptr, outIds + out, outDist2 + out);
    }
}

__global__ void clearResultsKernel(unsigned int* ids, float* dist2, size_t count)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    ids[i]   = kKnnInvalidId;
    dist2[i] = __int_as_float(0x7f800000);
}

}  // namespace

void computeBounds(const float3* points, unsigned int n, Aabb* dBounds, DeviceBuffer& temp, cudaStream_t stream)
{
    unsigned int* bits = temp.get<unsigned int>(6);
    initBoundsKernel<<<1, 32, 0, stream>>>(bits);
    if (n > 0)
        boundsKernel<<<blocksFor(n), kBlock, 0, stream>>>(points, n, bits);
    finishBoundsKernel<<<1, 1, 0, stream>>>(bits, dBounds);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void copyPoints(const float3* points, unsigned int n, float4* out, cudaStream_t stream)
{
    copyPointsKernel<<<blocksFor(n), kBlock, 0, stream>>>(points, n, out);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void hilbertSort(const float3* points, unsigned int n, const Aabb* dBounds, float4* sortedPoints,
                 unsigned int* sortedIds, uint64_t* sortedKeys, DeviceBuffer& scratch, DeviceBuffer& temp,
                 cudaStream_t stream)
{
    // Unsorted keys, then unsorted ids, 16-byte aligned.
    const size_t keyBytes = (static_cast<size_t>(n) * sizeof(uint64_t) + 15) & ~size_t(15);
    char* base            = scratch.get<char>(keyBytes + static_cast<size_t>(n) * sizeof(unsigned int));
    uint64_t* keys        = reinterpret_cast<uint64_t*>(base);
    unsigned int* ids     = reinterpret_cast<unsigned int*>(base + keyBytes);

    hilbertKeysKernel<<<blocksFor(n), kBlock, 0, stream>>>(points, n, dBounds, keys, ids);
    KNN_CUDA_CHECK(cudaGetLastError());

    size_t tempBytes = 0;
    KNN_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, keys, sortedKeys, ids, sortedIds, n, 0,
                                                   3 * kHilbertBits, stream));
    void* tempPtr = temp.get<char>(tempBytes);
    KNN_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(tempPtr, tempBytes, keys, sortedKeys, ids, sortedIds, n, 0,
                                                   3 * kHilbertBits, stream));

    gatherPointsKernel<<<blocksFor(n), kBlock, 0, stream>>>(points, sortedIds, n, sortedPoints);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void groupBounds(const float4* points, unsigned int n, Aabb* out, cudaStream_t stream)
{
    static_assert(kClusterSize == 16 && kListSize == 16, "groupBounds reduces over 16 lanes");
    pointGroupBoundsKernel<<<blocksFor(n), kBlock, 0, stream>>>(points, n, out);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void groupBounds(const Aabb* boxes, unsigned int n, Aabb* out, cudaStream_t stream)
{
    boxGroupBoundsKernel<<<blocksFor(n), kBlock, 0, stream>>>(boxes, n, out);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void padBoxes(const Aabb* boxes, unsigned int n, float pad, Aabb* out, cudaStream_t stream)
{
    padBoxesKernel<<<blocksFor(n), kBlock, 0, stream>>>(boxes, n, pad, out);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void pointBoxes(const float4* points, unsigned int n, float pad, Aabb* out, cudaStream_t stream)
{
    pointBoxesKernel<<<blocksFor(n), kBlock, 0, stream>>>(points, n, pad, out);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void buildLbvh(const uint64_t* sortedKeys, const Aabb* clusters, unsigned int numClusters, LbvhNode* nodes,
               DeviceBuffer& scratch, cudaStream_t stream)
{
    if (numClusters < 2)
        return;
    unsigned int* internalParent = scratch.get<unsigned int>(3 * static_cast<size_t>(numClusters));
    unsigned int* leafParent     = internalParent + numClusters;
    unsigned int* counters       = leafParent + numClusters;
    lbvhInternalKernel<<<blocksFor(numClusters - 1), kBlock, 0, stream>>>(
        sortedKeys, static_cast<int>(numClusters), nodes, internalParent, leafParent, counters);
    KNN_CUDA_CHECK(cudaGetLastError());
    lbvhBoundsKernel<<<blocksFor(numClusters), kBlock, 0, stream>>>(clusters, numClusters, nodes, internalParent,
                                                                    leafParent, counters);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void queryLbvh(const LbvhQuery& q, cudaStream_t stream)
{
    lbvhQueryKernel<<<blocksFor(q.numQueries), kBlock, 0, stream>>>(q);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void bruteForce(const float3* points, unsigned int numPoints, const float3* queries, unsigned int numQueries,
                unsigned int k, float maxRadius, unsigned int* outIds, float* outDist2, cudaStream_t stream)
{
    bruteForceKernel<<<blocksFor(numQueries), kBlock, 0, stream>>>(points, numPoints, queries, numQueries, k,
                                                                   maxRadius, outIds, outDist2);
    KNN_CUDA_CHECK(cudaGetLastError());
}

void clearResults(unsigned int* ids, float* dist2, size_t count, cudaStream_t stream)
{
    if (count == 0)
        return;
    clearResultsKernel<<<blocksFor(count), kBlock, 0, stream>>>(ids, dist2, count);
    KNN_CUDA_CHECK(cudaGetLastError());
}

}  // namespace knn
