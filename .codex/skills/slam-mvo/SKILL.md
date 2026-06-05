# SLAM MVO Skill

Use this skill when working on MVO, monocular visual odometry, KLT tracking,
ORB-SLAM-style initialization, PnP, triangulation, map points, bundle
adjustment, loop closure, OpenCV, `cvlib`, or Rerun visualization.

## Workflow

1. Read `configs/kitti_image_sequence.json` and the relevant file under
   `configs/parameters`.
2. Inspect the owning headers and sources under `include/mvo` and `src`.
3. Identify whether the issue is calibration, tracking, matching, two-view
   initialization, PnP, triangulation, map management, BA, or visualization.
4. Keep any new threshold configurable in the matching JSON parameter file.
5. Implement the smallest module-local change that keeps the full VO path
   running.
6. Validate with a short run and report the command result.

## Diagnostic Checklist

- Intrinsics: `fx`, `fy`, `cx`, `cy`, distortion, image size, and coordinate
  convention are consistent.
- KLT: track count, forward-backward error, lost tracks, and grid distribution
  are visible in logs or Rerun.
- Matching: 2D observations refer to the correct track ID and map point ID.
- Initialization: enough parallax, enough inliers, good cheirality, and bounded
  reprojection error.
- PnP: enough valid 2D-3D pairs and inliers before accepting pose.
- Map points: positive depth, reasonable triangulation angle, reprojection
  error gate, and observation count are enforced.
- BA: uses `cvlib`, runs only when enough constraints exist, and does not hide
  failures.
- Rerun: trajectory, current visible points, historical points, and counters are
  logged with inspectable sizes.

## Validation Commands

```powershell
.\build.ps1
.\run.ps1 -Config Release -MaxFrames 3 -NoBa -NoRerun -ParameterDir .\configs\parameters
.\run.ps1 -Config Release -MaxFrames 3 -NoBa -RerunSave .\build\mvo_check.rrd -ParameterDir .\configs\parameters
```

```bash
./build.sh
./run.sh --config Release --max-frames 3 --no-ba --no-rerun --parameter-dir ./configs/parameters
```

