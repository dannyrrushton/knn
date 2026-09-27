// Host side of the KNN library: builds the per-method index and launches queries.
#include "knn/knn.h"

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "knn_internal.h"

#define KNN_OPTIX_CHECK(call)                                                                        \
    do                                                                                               \
    {                                                                                                \
        const OptixResult knnRes_ = (call);                                                          \
        if (knnRes_ != OPTIX_SUCCESS)                                                                \
        {                                                                                            \
            std::ostringstream knnMsg_;                                                              \
            knnMsg_ << "OptiX error " << optixGetErrorName(knnRes_) << " at " << __FILE__ << ":"    \
                    << __LINE__ << ": " #call;                                                       \
            throw std::runtime_error(knnMsg_.str());                                                 \
        }                                                                                            \
    } while (0)

static_assert(sizeof(Aabb) == sizeof(OptixAabb), "Aabb must match OptixAabb");

using knn::DeviceBuffer;

const char* knnMethodName(KnnMethod method)
{
    switch (method)
    {
    case KnnMethod::FRS: return "FRS";
    case KnnMethod::ClusteredFRS: return "Clustered FRS";
    case KnnMethod::AnimatedFRS: return "Animated FRS";
    case KnnMethod::SoftwareLBVH: return "Software LBVH";
    }
    return "unknown";
}

namespace
{

struct SbtRecord
{
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

std::vector<char> readDeviceCode()
{
    const char* env   = std::getenv("KNN_DEVICE_CODE");
    const std::string path = env ? env : KNN_DEVICE_CODE_PATH;
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("KNN: cannot open device code " + path + " (set KNN_DEVICE_CODE to its path)");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void optixLog(unsigned int level, const char* tag, const char* message, void*)
{
    std::cerr << "[OptiX " << level << "][" << tag << "] " << message << "\n";
}

// One OptiX pipeline with every KNN program. Motion-blur and static trees need separate pipelines.
struct Pipeline
{
    OptixModule module = nullptr;
    std::vector<OptixProgramGroup> groups;
    OptixPipeline pipeline = nullptr;
    CUdeviceptr sbtRecords = 0;
    OptixShaderBindingTable sbt = {};

    void create(OptixDeviceContext context, const std::vector<char>& code, bool motion)
    {
        OptixModuleCompileOptions moduleOptions = {};
        moduleOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

        OptixPipelineCompileOptions pipelineOptions = {};
        pipelineOptions.usesMotionBlur                   = motion ? 1 : 0;
        pipelineOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineOptions.numPayloadValues                 = 32;  // the ray payload priority queue
        pipelineOptions.numAttributeValues               = 2;
        pipelineOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.pipelineLaunchParamsVariableName = "params";
        pipelineOptions.pipelineLaunchParamsSizeInBytes  = sizeof(KnnLaunchParams);
        pipelineOptions.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

        char log[4096];
        size_t logSize = sizeof(log);
        KNN_OPTIX_CHECK(optixModuleCreate(context, &moduleOptions, &pipelineOptions, code.data(), code.size(), log,
                                          &logSize, &module));

        auto makeGroup = [&](const OptixProgramGroupDesc& desc) {
            OptixProgramGroupOptions options = {};
            OptixProgramGroup group          = nullptr;
            size_t size                      = sizeof(log);
            KNN_OPTIX_CHECK(optixProgramGroupCreate(context, &desc, 1, &options, log, &size, &group));
            groups.push_back(group);
        };
        OptixProgramGroupDesc desc    = {};
        desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module            = module;
        desc.raygen.entryFunctionName = "__raygen__knn";
        makeGroup(desc);
        desc                        = {};
        desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module            = module;
        desc.miss.entryFunctionName = "__miss__knn";
        makeGroup(desc);
        const char* intersections[KNN_HIT_GROUP_COUNT] = {"__intersection__frs", "__intersection__clustered",
                                                          "__intersection__count"};
        for (const char* name : intersections)
        {
            desc                                  = {};
            desc.kind                             = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc.hitgroup.moduleIS                = module;
            desc.hitgroup.entryFunctionNameIS     = name;
            makeGroup(desc);
        }

        OptixPipelineLinkOptions linkOptions = {};
        linkOptions.maxTraceDepth            = 1;
        logSize                              = sizeof(log);
        KNN_OPTIX_CHECK(optixPipelineCreate(context, &pipelineOptions, &linkOptions, groups.data(),
                                            static_cast<unsigned int>(groups.size()), log, &logSize, &pipeline));

        OptixStackSizes stackSizes = {};
        for (OptixProgramGroup group : groups)
            KNN_OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stackSizes, pipeline));
        unsigned int dcFromTraversal = 0, dcFromState = 0, continuation = 0;
        KNN_OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, 1, 0, 0, &dcFromTraversal, &dcFromState, &continuation));
        KNN_OPTIX_CHECK(optixPipelineSetStackSize(pipeline, dcFromTraversal, dcFromState, continuation, 1));

        // Records: raygen, miss, then one hit group per KnnHitGroup.
        std::vector<SbtRecord> records(groups.size());
        for (size_t i = 0; i < groups.size(); ++i)
            KNN_OPTIX_CHECK(optixSbtRecordPackHeader(groups[i], &records[i]));
        KNN_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbtRecords), records.size() * sizeof(SbtRecord)));
        KNN_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbtRecords), records.data(),
                                  records.size() * sizeof(SbtRecord), cudaMemcpyHostToDevice));
        sbt.raygenRecord                = sbtRecords;
        sbt.missRecordBase              = sbtRecords + sizeof(SbtRecord);
        sbt.missRecordStrideInBytes     = sizeof(SbtRecord);
        sbt.missRecordCount             = 1;
        sbt.hitgroupRecordBase          = sbtRecords + 2 * sizeof(SbtRecord);
        sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
        sbt.hitgroupRecordCount         = KNN_HIT_GROUP_COUNT;
    }

    void destroy()
    {
        if (pipeline)
            optixPipelineDestroy(pipeline);
        for (OptixProgramGroup group : groups)
            optixProgramGroupDestroy(group);
        if (module)
            optixModuleDestroy(module);
        cudaFree(reinterpret_cast<void*>(sbtRecords));
        *this = Pipeline();
    }
};

// GPU timing of work queued on a stream.
class StreamTimer
{
public:
    explicit StreamTimer(cudaStream_t stream) : m_stream(stream)
    {
        KNN_CUDA_CHECK(cudaEventCreate(&m_start));
        KNN_CUDA_CHECK(cudaEventCreate(&m_stop));
        KNN_CUDA_CHECK(cudaEventRecord(m_start, m_stream));
    }
    ~StreamTimer()
    {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }
    // Waits for the work and returns its duration.
    double stop()
    {
        KNN_CUDA_CHECK(cudaEventRecord(m_stop, m_stream));
        KNN_CUDA_CHECK(cudaEventSynchronize(m_stop));
        float ms = 0.0f;
        KNN_CUDA_CHECK(cudaEventElapsedTime(&ms, m_start, m_stop));
        return ms;
    }

private:
    cudaStream_t m_stream;
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_stop  = nullptr;
};

}  // namespace

struct KnnIndex::Impl
{
    OptixDeviceContext context = nullptr;
    cudaStream_t stream        = nullptr;
    std::vector<char> deviceCode;
    Pipeline staticPipeline;  // FRS and Clustered FRS
    Pipeline motionPipeline;  // Animated FRS

    KnnMethod method        = KnnMethod::AnimatedFRS;
    float maxRadius         = 0.0f;
    unsigned int iterations = 8;
    unsigned int numPoints   = 0;
    unsigned int numClusters = 0;
    unsigned int numLists    = 0;
    OptixTraversableHandle handle = 0;

    DeviceBuffer points;    // float4, Hilbert-sorted except for FRS
    DeviceBuffer pointIds;  // original index of each sorted point
    DeviceBuffer keys;      // sorted Hilbert keys
    DeviceBuffer clusters;
    DeviceBuffer lists;
    DeviceBuffer rangeBoxes[2];  // one per motion key
    DeviceBuffer gas;
    DeviceBuffer gasTemp;
    DeviceBuffer lbvhNodes;
    DeviceBuffer scratch;
    DeviceBuffer temp;
    DeviceBuffer bounds;
    DeviceBuffer launchParams;

    Pipeline& pipeline(bool motion)
    {
        Pipeline& p = motion ? motionPipeline : staticPipeline;
        if (!p.pipeline)
        {
            if (deviceCode.empty())
                deviceCode = readDeviceCode();
            p.create(context, deviceCode, motion);
        }
        return p;
    }

    size_t buildGas(const Aabb* keys0, const Aabb* keys1, unsigned int count)
    {
        CUdeviceptr aabbs[2] = {reinterpret_cast<CUdeviceptr>(keys0), reinterpret_cast<CUdeviceptr>(keys1)};
        const unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixBuildInput input    = {};
        input.type               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        input.customPrimitiveArray.aabbBuffers   = aabbs;
        input.customPrimitiveArray.numPrimitives = count;
        input.customPrimitiveArray.strideInBytes = sizeof(Aabb);
        input.customPrimitiveArray.flags         = &flags;
        input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions options = {};
        options.buildFlags             = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        options.operation              = OPTIX_BUILD_OPERATION_BUILD;
        if (keys1)
        {
            options.motionOptions.numKeys   = 2;
            options.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
            options.motionOptions.timeBegin = 0.0f;
            options.motionOptions.timeEnd   = 1.0f;
        }

        OptixAccelBufferSizes sizes = {};
        KNN_OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &options, &input, 1, &sizes));
        gasTemp.reserve(sizes.tempSizeInBytes);
        gas.reserve(sizes.outputSizeInBytes);
        KNN_OPTIX_CHECK(optixAccelBuild(context, stream, &options, &input, 1,
                                        reinterpret_cast<CUdeviceptr>(gasTemp.as<void>()), sizes.tempSizeInBytes,
                                        reinterpret_cast<CUdeviceptr>(gas.as<void>()), sizes.outputSizeInBytes,
                                        &handle, nullptr, 0));
        return sizes.outputSizeInBytes;
    }

    // Hilbert sort and the two-level cluster hierarchy (Section 3.1).
    void buildClusters(const float3* dPoints)
    {
        numClusters = knn::divUp(numPoints, kClusterSize);
        numLists    = knn::divUp(numClusters, kListSize);
        Aabb* dBounds = bounds.get<Aabb>(1);
        knn::computeBounds(dPoints, numPoints, dBounds, temp, stream);
        knn::hilbertSort(dPoints, numPoints, dBounds, points.get<float4>(numPoints),
                         pointIds.get<unsigned int>(numPoints), keys.get<uint64_t>(numPoints), scratch, temp,
                         stream);
        knn::groupBounds(points.as<float4>(), numPoints, clusters.get<Aabb>(numClusters), stream);
        knn::groupBounds(clusters.as<Aabb>(), numClusters, lists.get<Aabb>(numLists), stream);
    }

    size_t bytesInUse() const
    {
        size_t total = 0;
        for (const DeviceBuffer* b : {&points, &pointIds, &keys, &clusters, &lists, &rangeBoxes[0], &rangeBoxes[1],
                                      &gas, &gasTemp, &lbvhNodes, &scratch, &temp})
            total += b->capacity();
        return total;
    }
};

KnnIndex::KnnIndex() : m(std::make_unique<Impl>())
{
    KNN_CUDA_CHECK(cudaFree(nullptr));  // make sure the CUDA context exists
    KNN_OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = optixLog;
    options.logCallbackLevel          = 2;  // errors and warnings
    KNN_OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &m->context));
    KNN_CUDA_CHECK(cudaStreamCreateWithFlags(&m->stream, cudaStreamNonBlocking));
}

KnnIndex::~KnnIndex()
{
    if (!m)
        return;
    cudaStreamSynchronize(m->stream);
    m->staticPipeline.destroy();
    m->motionPipeline.destroy();
    optixDeviceContextDestroy(m->context);
    cudaStreamDestroy(m->stream);
}

void KnnIndex::setIterations(unsigned int iterations) { m->iterations = iterations < 1 ? 1 : iterations; }
unsigned int KnnIndex::iterations() const { return m->iterations; }
KnnMethod KnnIndex::method() const { return m->method; }
float KnnIndex::maxRadius() const { return m->maxRadius; }
unsigned int KnnIndex::numPoints() const { return m->numPoints; }

KnnBuildResult KnnIndex::build(KnnMethod method, const float3* dPoints, unsigned int numPoints, float maxRadius)
{
    if (!(maxRadius > 0.0f) || !std::isfinite(maxRadius))
        throw std::invalid_argument("KnnIndex::build: maxRadius must be positive and finite");
    Impl& s     = *m;
    s.method    = method;
    s.maxRadius = maxRadius;
    s.numPoints = numPoints;
    s.numClusters = s.numLists = 0;
    s.handle    = 0;

    KnnBuildResult result;
    if (numPoints == 0)
        return result;
    if (method != KnnMethod::SoftwareLBVH)
        s.pipeline(method == KnnMethod::AnimatedFRS);  // compile outside the timed region

    // Range boxes are padded slightly past the radius (and rounded outwards), so the interpolated
    // boxes of the motion tree still contain every query they should.
    const float pad = maxRadius * 1.001f;

    StreamTimer timer(s.stream);
    switch (method)
    {
    case KnnMethod::FRS:
    {
        knn::copyPoints(dPoints, numPoints, s.points.get<float4>(numPoints), s.stream);
        Aabb* boxes = s.rangeBoxes[0].get<Aabb>(numPoints);
        knn::pointBoxes(s.points.as<float4>(), numPoints, pad, boxes, s.stream);
        result.treeBytes  = s.buildGas(boxes, nullptr, numPoints);
        result.primitives = numPoints;
        break;
    }
    case KnnMethod::ClusteredFRS:
    {
        s.buildClusters(dPoints);
        Aabb* boxes = s.rangeBoxes[0].get<Aabb>(s.numLists);
        knn::padBoxes(s.lists.as<Aabb>(), s.numLists, pad, boxes, s.stream);
        result.treeBytes  = s.buildGas(boxes, nullptr, s.numLists);
        result.primitives = s.numLists;
        break;
    }
    case KnnMethod::AnimatedFRS:
    {
        // Time 0: the tight cluster list bounds. Time 1: padded by the maximum search radius.
        s.buildClusters(dPoints);
        Aabb* keys0 = s.rangeBoxes[0].get<Aabb>(s.numLists);
        Aabb* keys1 = s.rangeBoxes[1].get<Aabb>(s.numLists);
        knn::padBoxes(s.lists.as<Aabb>(), s.numLists, 0.0f, keys0, s.stream);
        knn::padBoxes(s.lists.as<Aabb>(), s.numLists, pad, keys1, s.stream);
        result.treeBytes  = s.buildGas(keys0, keys1, s.numLists);
        result.primitives = s.numLists;
        break;
    }
    case KnnMethod::SoftwareLBVH:
    {
        s.buildClusters(dPoints);
        const size_t numNodes = s.numClusters > 1 ? s.numClusters - 1 : 1;
        knn::buildLbvh(s.keys.as<uint64_t>(), s.clusters.as<Aabb>(), s.numClusters,
                       s.lbvhNodes.get<knn::LbvhNode>(numNodes), s.scratch, s.stream);
        result.treeBytes  = numNodes * sizeof(knn::LbvhNode);
        result.primitives = s.numClusters;
        break;
    }
    }
    result.milliseconds = timer.stop();
    result.totalBytes   = s.bytesInUse();
    return result;
}

double KnnIndex::query(const float3* dQueries, unsigned int numQueries, unsigned int k, unsigned int* dIds,
                       float* dDist2)
{
    if (k < 1 || k > kMaxK)
        throw std::invalid_argument("KnnIndex::query: k must be in [1, 16]");
    Impl& s = *m;
    if (numQueries == 0)
        return 0.0;

    StreamTimer timer(s.stream);
    if (s.numPoints == 0)
    {
        knn::clearResults(dIds, dDist2, static_cast<size_t>(numQueries) * k, s.stream);
        return timer.stop();
    }

    if (s.method == KnnMethod::SoftwareLBVH)
    {
        knn::LbvhQuery q = {};
        q.nodes       = s.lbvhNodes.as<knn::LbvhNode>();
        q.clusters    = s.clusters.as<Aabb>();
        q.numClusters = s.numClusters;
        q.points      = s.points.as<float4>();
        q.pointIds    = s.pointIds.as<unsigned int>();
        q.numPoints   = s.numPoints;
        q.queries     = dQueries;
        q.numQueries  = numQueries;
        q.k           = k;
        q.maxRadius   = s.maxRadius;
        q.outIds      = dIds;
        q.outDist2    = dDist2;
        knn::queryLbvh(q, s.stream);
        return timer.stop();
    }

    const bool clustered = s.method != KnnMethod::FRS;
    KnnLaunchParams p    = {};
    p.handle      = s.handle;
    p.program     = s.method == KnnMethod::FRS            ? KNN_PROGRAM_FRS
                    : s.method == KnnMethod::ClusteredFRS ? KNN_PROGRAM_CLUSTERED
                                                          : KNN_PROGRAM_ANIMATED;
    p.queries     = dQueries;
    p.numQueries  = numQueries;
    p.k           = k;
    p.maxRadius   = s.maxRadius;
    p.iterations  = s.iterations;
    p.points      = s.points.as<float4>();
    p.pointIds    = clustered ? s.pointIds.as<unsigned int>() : nullptr;
    p.numPoints   = s.numPoints;
    p.clusters    = s.clusters.as<Aabb>();
    p.numClusters = s.numClusters;
    p.lists       = s.lists.as<Aabb>();
    p.numLists    = s.numLists;
    p.outIds      = dIds;
    p.outDist2    = dDist2;

    // Stream order keeps the parameter upload ahead of the launch; the pageable copy returns once staged.
    KnnLaunchParams* dParams = s.launchParams.get<KnnLaunchParams>(1);
    KNN_CUDA_CHECK(cudaMemcpyAsync(dParams, &p, sizeof(p), cudaMemcpyHostToDevice, s.stream));
    Pipeline& pipe = s.pipeline(s.method == KnnMethod::AnimatedFRS);
    KNN_OPTIX_CHECK(optixLaunch(pipe.pipeline, s.stream, reinterpret_cast<CUdeviceptr>(dParams), sizeof(p), &pipe.sbt,
                                numQueries, 1, 1));
    return timer.stop();
}

double knnBruteForce(const float3* dPoints, unsigned int numPoints, const float3* dQueries, unsigned int numQueries,
                     unsigned int k, float maxRadius, unsigned int* dIds, float* dDist2)
{
    if (k < 1 || k > kMaxK)
        throw std::invalid_argument("knnBruteForce: k must be in [1, 16]");
    if (numQueries == 0)
        return 0.0;
    StreamTimer timer(nullptr);
    knn::bruteForce(dPoints, numPoints, dQueries, numQueries, k, maxRadius, dIds, dDist2, nullptr);
    return timer.stop();
}
