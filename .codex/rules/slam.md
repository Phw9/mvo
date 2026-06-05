# MVO SLAM Rules

## Architecture

- Keep public headers in `include` and implementation files in `src`.
- Keep filenames lowercase.
- Keep module boundaries aligned with the current files:
  `feature`, `init`, `pose_estimation`, `map_data`, `bundle_adjustment`,
  `loop_closure`, `visualization`, `frame_source`, `parameters`, and `app`.
- Keep `src/cvlib_main.cpp` as a thin executable entry point.
- Do not add broad rewrites unless needed for the requested behavior.

## Dependencies

- Use bundled `thirdparty/cvlib/include` and `thirdparty/cvlib/lib`.
- Use `cvlib` for PnP and bundle adjustment APIs.
- Do not depend on direct local paths outside the repository for `cvlib`.
- Keep OpenCV discovery in CMake.
- Rerun should be installed or fetched by build scripts and enabled by default
  at runtime.
- DBoW2 is optional for loop closure; keep Windows builds valid even when DBoW2
  support is unavailable.

## Runtime Configuration

- Input type is defined by JSON, not hardcoded paths.
- Image sequence input is current priority; video input should remain supported
  by the config schema and frame-source boundary.
- Group tunable parameters by feature area:
  `feature.json`, `pnp.json`, `initializer.json`, `mapping.json`,
  `bundle_adjustment.json`, `loop_closure.json`, and `visualization.json`.
- Do not hide new thresholds in source files when they affect tracking,
  initialization, PnP, map point creation, BA, or visualization.

## Monocular Initialization

- Follow an ORB-SLAM3-style two-view initialization mindset:
  compare model quality, require parallax, verify cheirality, triangulate only
  well-conditioned points, and keep reprojection-error gates explicit.
- Do not accept initialization from low-baseline rotation-only scenes.
- Keep map point IDs tied to feature tracks so later PnP pairs remain correct.

## Tracking And Mapping

- KLT tracking must keep stable track-to-map associations.
- When tracked point count drops below the configured threshold, refresh
  features in a grid so coverage stays spread across the full image.
- New features should not duplicate existing active tracks in the same grid
  cell or near the same pixel.
- PnP should only consume 2D-3D pairs with known current-frame observations.
- Remove or mark weak map points instead of repeatedly using invalid points.
- Keep current-frame visible map points visually distinct from historical map
  points in Rerun.

## Visualization

- Rerun trajectory and 3D points should use large enough radii to inspect.
- Current-frame visible 3D points are red.
- Historical map points are black.
- Log enough counters to debug failures: tracked points, refreshed features,
  PnP pairs, inliers, triangulated points, rejected map points, and BA status.

## Build And Run

- Default config is `Release`; only `Release` and `Debug` are supported.
- Windows: `.\build.ps1` and `.\run.ps1 -Config Release`.
- Linux/Git Bash: `./build.sh` and `./run.sh --config Release`.
- Short validation:
  `.\run.ps1 -Config Release -MaxFrames 3 -NoBa -NoRerun -ParameterDir .\configs\parameters`.

