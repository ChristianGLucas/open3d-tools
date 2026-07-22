# open3d-tools

Composable Axiom nodes for deterministic **point-cloud** processing — the
operations robotics / LiDAR / 3D-scanning / AR / photogrammetry agents need
on unstructured 3D point sets. Built for the Axiom marketplace
(`christiangeorgelucas`).

Distinct from [`mesh-tools`](https://github.com/ChristianGLucas/mesh-tools)
(triangle-mesh ops: volume, surface area, boolean, decimate, ray
intersection) — this package operates on raw, unstructured `(x, y, z)` point
sets and covers the point-cloud-specific operations mesh-tools does not:
downsampling, normal estimation, registration, segmentation, and outlier
removal.

## Why not Open3D?

Open3D is the obvious reference library for this domain, but every published
Open3D wheel statically links [Eigen](https://eigen.tuxfamily.org/), which is
MPL-2.0 — and the `SimplicialCholesky` module Open3D actually uses is LGPL
unless the build defines `-DEIGEN_MPL2_ONLY` (upstream's published wheels do
not). The Point Cloud Library (PCL) has the same Eigen dependency. Both are
therefore excluded under this package's copyleft-anywhere-in-the-tree
selection gate.

Instead, every node here composes small, individually license-clean,
Eigen-free libraries:

- **numpy / scipy / scikit-learn** (BSD-3-Clause) — arrays, k-d trees
  (nearest-neighbor search, outlier removal, density), convex hull and
  Delaunay triangulation (`scipy.spatial`, wrapping Qhull's permissive
  license), and DBSCAN clustering.
- **[pyransac3d](https://github.com/leomariga/pyRANSAC-3D)** (Apache-2.0) —
  owns the RANSAC plane-segmentation algorithm.
- **[simpleicp](https://github.com/pglira/simpleICP)** (MIT) — owns the
  (point-to-plane) ICP registration algorithm.

PLY/PCD/XYZ file parsing is hand-written (not `plyfile`, which is
GPL-3.0) — straightforward format decoding, not an algorithmically hard part
a library needs to own.

## Nodes

| Node | What it does |
|---|---|
| `GetPointCloudInfo` | Parse a cloud and report point count, colors/normals presence, AABB, centroid |
| `VoxelDownsample` | Grid-based downsampling (merge points per voxel) |
| `DownsampleToCount` | Downsample to an exact target count (uniform stride or seeded random) |
| `EstimateNormals` | Per-point normal via local PCA |
| `RemoveStatisticalOutliers` | Remove points whose neighbor distance is a statistical outlier |
| `RemoveRadiusOutliers` | Remove points with too few neighbors in a radius |
| `SegmentPlaneRANSAC` | Seeded RANSAC dominant-plane segmentation |
| `ClusterDBSCAN` | Density-based clustering |
| `ComputeBoundingBox` | Axis-aligned + PCA-oriented bounding box |
| `ComputeConvexHull` | 3D convex hull of raw points |
| `RegisterICP` | ICP registration (source → target rigid transform) |
| `ComputePointCloudDistance` | Nearest-neighbor distance, cloud to cloud |
| `TransformPointCloud` | Apply a 4x4 transform |
| `CropPointCloud` | Axis-aligned box crop |
| `EstimatePointDensity` | Mean NN distance + points per unit volume |
| `ReconstructSurface` | 2.5D Delaunay height-field mesh reconstruction (PLY out) |
| `VoxelizeOccupancyGrid` | Voxel occupancy-grid summary |
| `ComputeGeometricFeatures` | Per-point linearity/planarity/sphericity/curvature |

## License

MIT. Copyright (c) 2026 Christian George Lucas.
