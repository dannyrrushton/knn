// OptiX programs for RT-core KNN queries (Morrical et al., HPG 2023).
//
// Queries trace zero-length rays from the query point through a tree of range boxes: a box that
// contains the ray origin belongs to a primitive within range of the query (Figure 2). The
// intersection programs never report hits during collection; they insert the points they find into
// a KNN priority queue that lives in the 32 ray payload registers (Section 3.2.2) and let traversal
// continue. The requested K and the search radius travel in the otherwise unused ray direction.
#include <optix.h>

#include "knn_common.cuh"

extern "C" {
__constant__ KnnLaunchParams params;
}

// Any positive length works: every box that contains the origin is visited.
constexpr float kRayTMax = 1.0e-16f;

// ---------------------------------------------------------------------------------------------------
// Ray payload priority queue: payload registers 0-15 hold the squared distances, 16-31 the ids.
// ---------------------------------------------------------------------------------------------------

#define KNN_LOAD_SLOT(i, j)                                    \
    queue.d[i]  = __uint_as_float(optixGetPayload_##i());      \
    queue.id[i] = optixGetPayload_##j();

#define KNN_STORE_SLOT(i, j)                                   \
    optixSetPayload_##i(__float_as_uint(queue.d[i]));          \
    optixSetPayload_##j(queue.id[i]);

static __forceinline__ __device__ void loadPayloadQueue(KnnQueue& queue)
{
    KNN_LOAD_SLOT(0, 16) KNN_LOAD_SLOT(1, 17) KNN_LOAD_SLOT(2, 18) KNN_LOAD_SLOT(3, 19)
    KNN_LOAD_SLOT(4, 20) KNN_LOAD_SLOT(5, 21) KNN_LOAD_SLOT(6, 22) KNN_LOAD_SLOT(7, 23)
    KNN_LOAD_SLOT(8, 24) KNN_LOAD_SLOT(9, 25) KNN_LOAD_SLOT(10, 26) KNN_LOAD_SLOT(11, 27)
    KNN_LOAD_SLOT(12, 28) KNN_LOAD_SLOT(13, 29) KNN_LOAD_SLOT(14, 30) KNN_LOAD_SLOT(15, 31)
}

static __forceinline__ __device__ void storePayloadQueue(const KnnQueue& queue)
{
    KNN_STORE_SLOT(0, 16) KNN_STORE_SLOT(1, 17) KNN_STORE_SLOT(2, 18) KNN_STORE_SLOT(3, 19)
    KNN_STORE_SLOT(4, 20) KNN_STORE_SLOT(5, 21) KNN_STORE_SLOT(6, 22) KNN_STORE_SLOT(7, 23)
    KNN_STORE_SLOT(8, 24) KNN_STORE_SLOT(9, 25) KNN_STORE_SLOT(10, 26) KNN_STORE_SLOT(11, 27)
    KNN_STORE_SLOT(12, 28) KNN_STORE_SLOT(13, 29) KNN_STORE_SLOT(14, 30) KNN_STORE_SLOT(15, 31)
}

// The Kth nearest distance so far, without reading the rest of the queue.
static __forceinline__ __device__ float payloadWorst() { return __uint_as_float(optixGetPayload_0()); }

// ---------------------------------------------------------------------------------------------------
// collectKNN: the traceRay-like interface of Section 3. Finds the K nearest points within `radius`
// of `origin`, then calls hit(queue, found) or, when nothing is in range, miss().
// ---------------------------------------------------------------------------------------------------

template <class Hit, class Miss>
static __forceinline__ __device__ void collectKNN(OptixTraversableHandle handle, float3 origin, float radius,
                                                  unsigned int k, float time, unsigned int hitGroup, Hit hit,
                                                  Miss miss)
{
    KnnQueue queue;
    queue.init(k, queueLimit(radius));
    unsigned int p[32];
#pragma unroll
    for (unsigned int i = 0; i < kMaxK; ++i)
    {
        p[i]         = __float_as_uint(queue.d[i]);
        p[i + kMaxK] = queue.id[i];
    }

    const float3 direction = make_float3(static_cast<float>(k), radius, 1.0f);
    optixTrace(handle, origin, direction, 0.0f, kRayTMax, time, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, hitGroup, 0, 0,
               p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15],
               p[16], p[17], p[18], p[19], p[20], p[21], p[22], p[23], p[24], p[25], p[26], p[27], p[28], p[29],
               p[30], p[31]);

#pragma unroll
    for (unsigned int i = 0; i < kMaxK; ++i)
    {
        queue.d[i]  = __uint_as_float(p[i]);
        queue.id[i] = p[i + kMaxK];
    }
    const unsigned int found = queue.count(k);
    if (found > 0)
        hit(queue, found);
    else
        miss();
}

// ---------------------------------------------------------------------------------------------------
// Ray-time marching (Section 3.3.1): the range boxes of the motion tree grow from the tight cluster
// list bounds at time 0 to bounds padded by the maximum search radius at time 1, so tracing at time t
// searches radius t * maxRadius. Counting rays march t through `iterations` steps until they find a
// radius that certainly contains K points, which the KNN ray then searches.
// ---------------------------------------------------------------------------------------------------

static __forceinline__ __device__ float rayTimeMarch(OptixTraversableHandle handle, float3 origin, unsigned int k)
{
    const unsigned int steps = max(params.iterations, 1u);
    for (unsigned int s = 1; s < steps; ++s)
    {
        const float time   = static_cast<float>(s) / static_cast<float>(steps);
        unsigned int count = 0;
        unsigned int kk    = k;
        optixTrace(handle, origin, make_float3(1.0f, 1.0f, 1.0f), 0.0f, kRayTMax, time, OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                       OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   KNN_HIT_COUNT, 0, 0, count, kk);
        if (count >= k)
            return time;
    }
    return 1.0f;
}

extern "C" __global__ void __raygen__knn()
{
    const unsigned int index = optixGetLaunchIndex().x;
    if (index >= params.numQueries)
        return;
    const float3 origin = params.queries[index];
    const unsigned int k = params.k;

    float time       = 0.0f;
    float radius     = params.maxRadius;
    unsigned int hit = params.program == KNN_PROGRAM_FRS ? KNN_HIT_FRS : KNN_HIT_CLUSTERED;
    if (params.program == KNN_PROGRAM_ANIMATED)
    {
        time   = rayTimeMarch(params.handle, origin, k);
        radius = time < 1.0f ? time * params.maxRadius : params.maxRadius;
    }

    unsigned int* outIds = params.outIds + static_cast<size_t>(index) * k;
    float* outDist2      = params.outDist2 + static_cast<size_t>(index) * k;
    collectKNN(
        params.handle, origin, radius, k, time, hit,
        [&](const KnnQueue& queue, unsigned int) { queue.write(k, params.pointIds, outIds, outDist2); },
        [&]() {
            for (unsigned int i = 0; i < k; ++i)
            {
                outIds[i]   = kKnnInvalidId;
                outDist2[i] = __int_as_float(0x7f800000);
            }
        });
}

extern "C" __global__ void __miss__knn() {}

// ---------------------------------------------------------------------------------------------------
// Intersection programs
// ---------------------------------------------------------------------------------------------------

// FRS: each primitive is a single point.
extern "C" __global__ void __intersection__frs()
{
    const unsigned int point = optixGetPrimitiveIndex();
    const float d2           = dist2(optixGetWorldRayOrigin(), params.points[point]);
    if (d2 >= payloadWorst())
        return;
    KnnQueue queue;
    loadPayloadQueue(queue);
    queue.insert(d2, point);
    storePayloadQueue(queue);
}

// Clustered FRS: each primitive is a list of clusters. The radius is baked into the queue's initial
// limit, so every test is against the current Kth nearest distance.
extern "C" __global__ void __intersection__clustered()
{
    const unsigned int list = optixGetPrimitiveIndex();
    const float3 q          = optixGetWorldRayOrigin();
    if (minDist2(q, params.lists[list]) >= payloadWorst())
        return;  // the whole list is out of range or behind the Kth nearest
    KnnQueue queue;
    loadPayloadQueue(queue);
    if (collectList(queue, q, params, list))
        storePayloadQueue(queue);
}

// Counting rays: add a conservative count of the points within radius time * maxRadius to payload 0,
// and end traversal once it reaches K (payload 1). Whole lists and clusters within range count all
// their points; a cluster with a face within range counts one (the pessimistic distance, Figure 7).
extern "C" __global__ void __intersection__count()
{
    const unsigned int list = optixGetPrimitiveIndex();
    const float3 q          = optixGetWorldRayOrigin();
    const float radius      = optixGetRayTime() * params.maxRadius;
    const float r2          = radius * radius;

    const Aabb box = params.lists[list];
    if (minDist2(q, box) > r2)
        return;

    const unsigned int firstCluster = list * kListSize;
    const unsigned int numClusters  = min(kListSize, params.numClusters - firstCluster);
    unsigned int count              = optixGetPayload_0();
    if (maxDist2(q, box) <= r2)
    {
        const unsigned int firstPoint = firstCluster * kClusterSize;
        count += min(kPointsPerList, params.numPoints - firstPoint);
    }
    else
    {
        for (unsigned int i = 0; i < numClusters; ++i)
        {
            const unsigned int c = firstCluster + i;
            const Aabb cb        = params.clusters[c];
            if (maxDist2(q, cb) <= r2)
                count += min(kClusterSize, params.numPoints - c * kClusterSize);
            else if (pessimisticDist2(q, cb) <= r2)
                count += 1;
        }
    }
    optixSetPayload_0(count);
    if (count >= optixGetPayload_1())
        optixReportIntersection(optixGetRayTmin(), 0);
}
