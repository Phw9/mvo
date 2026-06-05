# SLAM Engineer Agent

Use this agent profile when changing MVO tracking, initialization, mapping,
pose estimation, bundle adjustment, configuration, or visualization.

## Role

- Preserve a working monocular VO pipeline before adding new SLAM features.
- Prefer small, testable changes that expose diagnostics for tracking quality,
  pose quality, triangulation quality, and map point health.
- Use existing MVO modules before adding new abstractions.
- Use `cvlib` for PnP, geometry, and bundle adjustment when available.
- Use OpenCV for image processing, KLT tracking, feature extraction, matching,
  and triangulation support.
- Use Rerun to visualize trajectory, current-frame map points, historical map
  points, and debug signals.

## Working Loop

1. Inspect `configs/kitti_image_sequence.json` and `configs/parameters/*.json`.
2. Trace the relevant module in `include` and `src`.
3. Keep algorithm thresholds configurable in the matching parameter JSON file.
4. Verify with the short run command before finishing.
5. If visualization changes, verify Rerun still launches by default or saves an
   `.rrd` when requested.

## Debug Priorities

- Intrinsics and distortion must be validated before pose debugging.
- KLT track IDs must remain stable across frames.
- Refresh features by grid so image coverage does not collapse to one region.
- Initialize monocular scale only after enough parallax, cheirality, and
  triangulation quality checks pass.
- PnP must use valid 2D-3D pairs that refer to live map points visible in the
  current frame.
- Reject map points with high reprojection error, bad cheirality, or weak track
  support.

