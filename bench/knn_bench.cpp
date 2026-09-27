// Synthetic KNN benchmark and validation (after Section 4.1 of Morrical et al. 2023).
//
// Generates data and query points, then for every method, search radius, K and (for Animated FRS)
// iteration count, measures build and query times and checks the results against a brute-force
// search.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "knn/knn.h"

#define CHECK_CUDA(call)                                                                              \
    do                                                                                                \
    {                                                                                                 \
        const cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess)                                                                      \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + " in " #call); \
    } while (0)

namespace
{

struct Options
{
    unsigned int numPoints  = 1u << 24;
    unsigned int numQueries = 1u << 15;
    std::vector<unsigned int> ks         = {1, 4, 8, 16};
    std::vector<float> radii             = {0.01f, 0.02f, 0.04f, 0.08f, 0.16f, 0.32f, 0.64f};
    std::vector<unsigned int> iterations = {8};
    std::vector<KnnMethod> methods = {KnnMethod::FRS, KnnMethod::ClusteredFRS, KnnMethod::AnimatedFRS,
                                      KnnMethod::SoftwareLBVH};
    std::string distribution = "uniform";
    unsigned int validate    = 1024;  // queries checked against brute force
    unsigned int trials      = 3;
    double frsLimit          = 2.0e4;  // skip FRS when a range box holds more points than this on average
    std::string csv;
    unsigned int seed = 1;
};

template <class T>
std::vector<T> parseList(const std::string& s, T (*parse)(const std::string&))
{
    std::vector<T> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(parse(item));
    return out;
}

unsigned int parseUint(const std::string& s)
{
    const double v = std::stod(s);  // accepts 1e6 and 16777216 alike
    if (v < 0 || v > 4294967295.0)
        throw std::runtime_error("value out of range: " + s);
    return static_cast<unsigned int>(v);
}
float parseFloat(const std::string& s) { return std::stof(s); }

KnnMethod parseMethod(const std::string& s)
{
    if (s == "frs") return KnnMethod::FRS;
    if (s == "cfrs") return KnnMethod::ClusteredFRS;
    if (s == "afrs") return KnnMethod::AnimatedFRS;
    if (s == "lbvh") return KnnMethod::SoftwareLBVH;
    throw std::runtime_error("unknown method " + s + " (expected frs, cfrs, afrs or lbvh)");
}

const char* methodTag(KnnMethod m)
{
    switch (m)
    {
    case KnnMethod::FRS: return "frs";
    case KnnMethod::ClusteredFRS: return "cfrs";
    case KnnMethod::AnimatedFRS: return "afrs";
    case KnnMethod::SoftwareLBVH: return "lbvh";
    }
    return "?";
}

void usage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  -n <count>          data points (default 2^24)\n"
        << "  -q <count>          query points (default 2^15)\n"
        << "  -k <list>           K values, 1-16 (default 1,4,8,16)\n"
        << "  -r <list>           maximum search radii (default 0.01,...,0.64)\n"
        << "  -i <list>           Animated FRS ray-time iterations (default 8)\n"
        << "  -m <list>           methods: frs,cfrs,afrs,lbvh (default all)\n"
        << "  --dist <name>       uniform: points and queries in the unit cube (default)\n"
        << "                      caustic: points on a plane, mostly in a tight focus and a ring\n"
        << "                      duplicates: ~100 copies of each point\n"
        << "  --validate <count>  queries checked against brute force (default 1024, 0 = off)\n"
        << "  --trials <count>    query timings averaged (default 3)\n"
        << "  --frs-limit <n>     skip FRS when range boxes average more than n points (default 2e4;\n"
        << "                      FRS degrades to exhaustive search and can stall the GPU)\n"
        << "  --csv <file>        also write results as CSV\n"
        << "  --seed <n>          random seed (default 1)\n";
}

Options parseArgs(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "-n") o.numPoints = parseUint(next());
        else if (a == "-q") o.numQueries = parseUint(next());
        else if (a == "-k") o.ks = parseList<unsigned int>(next(), parseUint);
        else if (a == "-r") o.radii = parseList<float>(next(), parseFloat);
        else if (a == "-i") o.iterations = parseList<unsigned int>(next(), parseUint);
        else if (a == "-m") o.methods = parseList<KnnMethod>(next(), parseMethod);
        else if (a == "--dist") o.distribution = next();
        else if (a == "--validate") o.validate = parseUint(next());
        else if (a == "--trials") o.trials = std::max(1u, parseUint(next()));
        else if (a == "--frs-limit") o.frsLimit = std::stod(next());
        else if (a == "--csv") o.csv = next();
        else if (a == "--seed") o.seed = parseUint(next());
        else if (a == "-h" || a == "--help")
        {
            usage(argv[0]);
            std::exit(0);
        }
        else
            throw std::runtime_error("unknown option " + a);
    }
    for (unsigned int k : o.ks)
        if (k < 1 || k > kMaxK)
            throw std::runtime_error("K must be in [1, 16]");
    return o;
}

// Data and query points. "caustic" imitates the glass-sphere scene of Figure 11: photons on a floor
// plane, half focused into a tight spot, a quarter in a ring, the rest spread evenly.
void generate(const Options& o, std::vector<float3>& points, std::vector<float3>& queries)
{
    std::mt19937 rng(o.seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    points.resize(o.numPoints);
    queries.resize(o.numQueries);
    if (o.distribution == "uniform")
    {
        for (float3& p : points)
            p = make_float3(uni(rng), uni(rng), uni(rng));
        for (float3& q : queries)
            q = make_float3(uni(rng), uni(rng), uni(rng));
        return;
    }
    if (o.distribution == "duplicates")
    {
        // Many points at each of a few positions: equal Hilbert keys and zero-size cluster boxes.
        std::vector<float3> sites(std::max(1u, o.numPoints / 100));
        for (float3& s : sites)
            s = make_float3(uni(rng), uni(rng), uni(rng));
        std::uniform_int_distribution<size_t> pick(0, sites.size() - 1);
        for (float3& p : points)
            p = sites[pick(rng)];
        for (float3& q : queries)
            q = make_float3(uni(rng), uni(rng), uni(rng));
        return;
    }
    if (o.distribution != "caustic")
        throw std::runtime_error("unknown distribution " + o.distribution);
    const float floor = 0.25f;
    std::normal_distribution<float> spot(0.0f, 0.004f);
    std::normal_distribution<float> ringWidth(0.0f, 0.003f);
    for (float3& p : points)
    {
        const float u = uni(rng);
        if (u < 0.5f)
            p = make_float3(0.5f + spot(rng), 0.5f + spot(rng), floor);
        else if (u < 0.75f)
        {
            const float a = 6.2831853f * uni(rng);
            const float r = 0.2f + ringWidth(rng);
            p             = make_float3(0.5f + r * std::cos(a), 0.5f + r * std::sin(a), floor);
        }
        else
            p = make_float3(uni(rng), uni(rng), floor);
    }
    for (float3& q : queries)
        q = make_float3(uni(rng), uni(rng), floor);
}

template <class T>
T* upload(const std::vector<T>& v)
{
    T* d = nullptr;
    CHECK_CUDA(cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

float hostDist2(float3 a, float3 b)
{
    const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

// The GPU contracts to fused multiply-adds, so its distances can differ from the host's in the last bit.
bool close(float a, float b) { return std::fabs(a - b) <= 1e-6f * std::max(std::max(a, b), 1e-12f); }

struct Validation
{
    size_t wrongDistance = 0;  // result distance differs from the brute-force one
    size_t wrongId       = 0;  // id does not have the reported distance
};

// Compares the first `count` queries' results with the reference.
Validation validate(unsigned int count, unsigned int k, const std::vector<float3>& points,
                    const std::vector<float3>& queries, const std::vector<unsigned int>& ids,
                    const std::vector<float>& dist2, const std::vector<unsigned int>& refIds,
                    const std::vector<float>& refDist2)
{
    Validation v;
    for (size_t i = 0; i < static_cast<size_t>(count) * k; ++i)
    {
        const bool found    = ids[i] != kKnnInvalidId;
        const bool refFound = refIds[i] != kKnnInvalidId;
        if (found != refFound || (found && !close(dist2[i], refDist2[i])))
            ++v.wrongDistance;
        if (found && (ids[i] >= points.size() || !close(hostDist2(points[ids[i]], queries[i / k]), dist2[i])))
            ++v.wrongId;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options o = parseArgs(argc, argv);
        cudaDeviceProp prop = {};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU: %s\nData: %s, %u points, %u queries\n\n", prop.name, o.distribution.c_str(), o.numPoints,
                    o.numQueries);

        std::vector<float3> points, queries;
        generate(o, points, queries);
        float3* dPoints  = upload(points);
        float3* dQueries = upload(queries);

        const unsigned int validateCount = std::min(o.validate, o.numQueries);
        const size_t maxResults          = static_cast<size_t>(o.numQueries) * kMaxK;
        unsigned int* dIds = nullptr;
        float* dDist2      = nullptr;
        unsigned int* dRefIds = nullptr;
        float* dRefDist2      = nullptr;
        CHECK_CUDA(cudaMalloc(&dIds, maxResults * sizeof(unsigned int)));
        CHECK_CUDA(cudaMalloc(&dDist2, maxResults * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dRefIds, std::max<size_t>(1, validateCount) * kMaxK * sizeof(unsigned int)));
        CHECK_CUDA(cudaMalloc(&dRefDist2, std::max<size_t>(1, validateCount) * kMaxK * sizeof(float)));

        std::ofstream csv;
        if (!o.csv.empty())
        {
            csv.open(o.csv);
            csv << "distribution,method,radius,k,iterations,build_ms,tree_bytes,total_bytes,primitives,query_ms,"
                   "wrong_distances,wrong_ids,validated_queries\n";
        }

        std::printf("%-14s %6s %3s %4s %10s %9s %10s %11s %s\n", "method", "radius", "K", "iter", "build ms",
                    "tree MB", "query ms", "Mquery/s", "validation");
        KnnIndex index;
        size_t failures = 0;
        for (const float radius : o.radii)
        {
            // Reference results for each K on the validation subset.
            std::vector<std::vector<unsigned int>> refIds(o.ks.size());
            std::vector<std::vector<float>> refDist2(o.ks.size());
            if (validateCount > 0)
            {
                for (size_t ki = 0; ki < o.ks.size(); ++ki)
                {
                    const unsigned int k = o.ks[ki];
                    knnBruteForce(dPoints, o.numPoints, dQueries, validateCount, k, radius, dRefIds, dRefDist2);
                    refIds[ki].resize(static_cast<size_t>(validateCount) * k);
                    refDist2[ki].resize(refIds[ki].size());
                    CHECK_CUDA(cudaMemcpy(refIds[ki].data(), dRefIds, refIds[ki].size() * sizeof(unsigned int),
                                          cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(refDist2[ki].data(), dRefDist2, refDist2[ki].size() * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                }
            }

            for (const KnnMethod method : o.methods)
            {
                if (method == KnnMethod::FRS)
                {
                    // Points per range box, for the uniform cube; for the plane, per box footprint.
                    const double side     = 2.0 * radius;
                    const double expected = o.distribution == "uniform" ? o.numPoints * side * side * side
                                                                        : o.numPoints * side * side;
                    if (expected > o.frsLimit)
                    {
                        std::printf("%-14s %6.3f   -    - %10s   (skipped: ~%.0f points per range box)\n",
                                    methodTag(method), radius, "-", expected);
                        continue;
                    }
                }
                const std::vector<unsigned int> iterationList =
                    method == KnnMethod::AnimatedFRS ? o.iterations : std::vector<unsigned int>{0};
                for (const unsigned int iterations : iterationList)
                {
                    if (method == KnnMethod::AnimatedFRS)
                        index.setIterations(iterations);
                    index.build(method, dPoints, o.numPoints, radius);  // warm up
                    const KnnBuildResult build = index.build(method, dPoints, o.numPoints, radius);

                    for (size_t ki = 0; ki < o.ks.size(); ++ki)
                    {
                        const unsigned int k = o.ks[ki];
                        index.query(dQueries, o.numQueries, k, dIds, dDist2);  // warm up
                        double ms = 0.0;
                        for (unsigned int t = 0; t < o.trials; ++t)
                            ms += index.query(dQueries, o.numQueries, k, dIds, dDist2);
                        ms /= o.trials;

                        std::string status = "off";
                        Validation v;
                        if (validateCount > 0)
                        {
                            std::vector<unsigned int> ids(static_cast<size_t>(validateCount) * k);
                            std::vector<float> dist2(ids.size());
                            CHECK_CUDA(cudaMemcpy(ids.data(), dIds, ids.size() * sizeof(unsigned int),
                                                  cudaMemcpyDeviceToHost));
                            CHECK_CUDA(cudaMemcpy(dist2.data(), dDist2, dist2.size() * sizeof(float),
                                                  cudaMemcpyDeviceToHost));
                            v = validate(validateCount, k, points, queries, ids, dist2, refIds[ki], refDist2[ki]);
                            if (v.wrongDistance == 0 && v.wrongId == 0)
                                status = "ok";
                            else
                            {
                                ++failures;
                                status = "FAIL " + std::to_string(v.wrongDistance) + " distances, " +
                                         std::to_string(v.wrongId) + " ids";
                            }
                        }
                        char iter[16] = "-";
                        if (method == KnnMethod::AnimatedFRS)
                            std::snprintf(iter, sizeof(iter), "%u", iterations);
                        std::printf("%-14s %6.3f %3u %4s %10.2f %9.1f %10.3f %11.1f %s\n", methodTag(method), radius, k,
                                    iter, build.milliseconds, build.treeBytes / 1048576.0, ms,
                                    o.numQueries / ms / 1000.0, status.c_str());
                        std::fflush(stdout);
                        if (csv)
                            csv << o.distribution << ',' << methodTag(method) << ',' << radius << ',' << k << ','
                                << (method == KnnMethod::AnimatedFRS ? iterations : 0) << ',' << build.milliseconds
                                << ',' << build.treeBytes << ',' << build.totalBytes << ',' << build.primitives
                                << ',' << ms << ',' << v.wrongDistance << ',' << v.wrongId << ',' << validateCount
                                << '\n';
                    }
                }
            }
        }

        cudaFree(dPoints);
        cudaFree(dQueries);
        cudaFree(dIds);
        cudaFree(dDist2);
        cudaFree(dRefIds);
        cudaFree(dRefDist2);
        if (failures > 0)
        {
            std::printf("\n%zu configurations failed validation\n", failures);
            return 1;
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
