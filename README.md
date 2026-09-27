# knn — RT-core K-nearest-neighbour queries

An implementation of Morrical, Wu, Rushton, Zellmann and Pshriwise, *Hardware Accelerated KNN Queries
using Ray-Time Marching* (HPG 2023), in CUDA and OptiX 9.

Every method answers the same query: for each query point, the K ≤ 16 nearest data points within a
maximum search radius. Results come back nearest first, as ids and squared distances.

| Method | Paper | What it does |
|---|---|---|
| `KnnMethod::FRS` | §3.1, Evangelou et al. 2021 | One range box (point ± radius) per point; zero-length rays; exhaustive within range |
| `KnnMethod::ClusteredFRS` | §3.1–3.2 | Hilbert-sorted points in a 16-point × 16-cluster hierarchy; RT tree over cluster lists; register cluster queue + ray-payload priority queue |
| `KnnMethod::AnimatedFRS` | §3.3 | Clustered FRS over a **motion-blur** tree whose range boxes grow from tight (t = 0) to radius-padded (t = 1); counting rays march ray time to find a tight radius, then one KNN ray collects |
| `KnnMethod::SoftwareLBVH` | baseline, after Jakob & Guthe 2021 | Karras LBVH over the same clusters, near-to-far DFS on CUDA cores with a register queue |

## Where each idea lives

- **Hilbert sort + clustering** (§3.1.1, Fig. 4): `hilbertKey`, `hilbertSort`, `groupBounds` in
  `src/knn_kernels.cu`. 63-bit keys (21 bits per axis), CUB radix sort, 16-lane shuffle reductions for
  cluster and list bounds.
- **Sphere-swept cluster range boxes** (Fig. 3): `padBoxes`, which rounds outwards so boxes stay
  conservative.
- **Cluster hierarchy + upward pruning** (§3.1.2), **cluster priority queue** (§3.2.1):
  `collectList` in `src/knn_common.cuh`.
- **Ray payload priority queue** (§3.2.2): `KnnQueue` plus `load/storePayloadQueue` in
  `src/knn_device.cu`. It uses all 32 payload registers (16 distances, 16 ids). K and the radius ride
  in the ray direction, as the paper suggests. The queue is kept in descending order with `-1`
  padding, so insertion is fully unrolled for any runtime K.
- **Motion-blur range boxes** (§3.3, Fig. 6): a two-key custom-primitive GAS built in
  `KnnIndex::Impl::buildGas`, with a separate motion-enabled pipeline.
- **Ray-time marching** (§3.3.1, Figs. 7–8): `rayTimeMarch` and `__intersection__count`. Counting
  rays add whole lists or clusters whose farthest corner is in range. A cluster that is only partly in
  range adds 1 when its pessimistic distance is in range. Traversal stops, via a reported hit, once the
  count reaches K.
- **`collectKNN(handle, origin, radius, k, time, hitGroup, hit, miss)`** (§3): the
  traceRay-like device interface with lambda hit/miss programs, in `src/knn_device.cu`.

## Building

Requires CUDA (tested with 13.3), CMake ≥ 3.27 and an RTX GPU. The OptiX 9.1 headers are vendored in
`third_party/optix`.

```sh
cmake -S . -B build -G Ninja
cmake --build build
./build/knn_bench -h
```

The device programs are compiled to OptiX IR, and the library loads them at runtime from the build
directory. If you move the binaries, set `KNN_DEVICE_CODE=/path/to/knn_device.optixir`.

The source layout (`include/`, `src/`, `bench/`) matches what the spectral renderer's CMakeLists
expects under `knn/`. Symlink or copy this directory there to use it in the renderer.

## API

```cpp
#include "knn/knn.h"

KnnIndex index;
index.setIterations(8);                                   // Animated FRS ray-time steps
KnnBuildResult b = index.build(KnnMethod::AnimatedFRS, dPoints, numPoints, maxRadius);
double ms = index.query(dQueries, numQueries, k, dIds, dDist2);   // k ids + dist² per query
```

`build` and `query` are synchronous and return GPU time. Buffers only grow, so per-frame rebuilds
stop allocating after warm-up. Unfilled result slots hold `kKnnInvalidId` and `INFINITY`.
`knnBruteForce` is an exhaustive reference with the same semantics.

## Benchmark and validation

`knn_bench` reproduces the synthetic setup of §4.1: 2²⁴ uniform points and 2¹⁵ queries in the unit
cube, radii 0.01–0.64, and K from 1 to 16. It also offers a `caustic` distribution modeled on the
glass-sphere scene of Fig. 11, and a `duplicates` distribution. Every configuration is checked against
brute force, both the distances and that each id's distance matches the reported one.

All methods pass validation on every distribution tested. That includes point counts that don't fill
a cluster or list, from 1 point up, and heavily duplicated points.

Results on an RTX A6000 (Ampere, with motion-blur hardware), 2²⁴ uniform points, 2¹⁵ queries,
K = 16, 8 AFRS iterations:

| radius | FRS query | CFRS query | AFRS query | LBVH query |
|---|---|---|---|---|
| 0.01 | 2.1 ms | 2.1 ms | 4.9 ms | 1.3 ms |
| 0.04 | 90.7 ms | 9.1 ms | 6.4 ms | 1.5 ms |
| 0.16 | skipped | 58 ms | 5.1 ms | 1.6 ms |
| 0.64 | skipped | 507 ms | 25.8 ms (5.8 ms at 32 iterations) | 1.5 ms |

| | FRS | CFRS | AFRS | LBVH |
|---|---|---|---|---|
| Build (2²⁴ points) | 40 ms | 12 ms | 12 ms | 12 ms |
| Tree memory | 599 MB | 2.3 MB | 2.8 MB | 32 MB |

**What reproduces:**
- Clustering cuts tree memory by about 250× and build time by more than 3× versus FRS.
- CFRS is 10× or more faster than FRS once the radius grows.
- The motion-blur range search keeps AFRS nearly flat as the radius grows, where CFRS and FRS blow
  up: 20× or more faster than CFRS at large radii.
- As the paper notes, AFRS gains little for small radii.

**What does not reproduce here:** the software LBVH baseline is faster than every RT method on this
GPU and data. That includes the caustic distribution, where the paper reports its largest wins. The
baseline is my own re-implementation in the style of Jakob & Guthe, not their code. It builds over the
same Hilbert clusters and uses a local-memory stack rather than their stackless traversal, so it may be
stronger than the version the paper measured. The paper's numbers are also from RTX 3090 Ti/4090 GPUs.

## Not implemented

- K > 16. The paper suggests spilling the payload queue to local memory; `query` rejects K > 16.
- A third cluster-hierarchy level. The hierarchy is fixed at 16 × 16.
- The real-scene caustic photon-mapping evaluation (§4.2).
- Exposing `collectKNN` for use from your own OptiX pipelines. It is used internally by the batch
  query raygen.
