# MVO

MVO is a monocular visual odometry project written with my `cvlib`.

Author: Hyunwoo Park <phphww93@gmail.com>

## Build

`Release` is the default build config. Only `Release` and `Debug` are accepted.

Windows PowerShell:

```powershell
.\build.ps1
.\build.ps1 -Config Debug
```

Linux or Git Bash:

```bash
./build.sh
./build.sh --config Debug
```

## Run

The Rerun viewer is spawned by default. Use `-NoRerun` or `--no-rerun` to
disable visualization.

Windows PowerShell:

```powershell
.\run.ps1 -Config Release
.\run.ps1 -Config Release -InputConfig .\configs\kitti_image_sequence.json
.\run.ps1 -Config Release -RerunSave .\build\mvo.rrd
.\run.ps1 -Config Release -NoRerun
```

Linux or Git Bash:

```bash
./run.sh --config Release
./run.sh --config Release --input-config ./configs/kitti_image_sequence.json
./run.sh --config Release --rerun-save ./build/mvo.rrd
./run.sh --config Release --no-rerun
```

Optional parameter directory override:

```powershell
.\run.ps1 -Config Release -ParameterDir .\configs\parameters
```

```bash
./run.sh --config Release --parameter-dir ./configs/parameters
```

## File Structure

```text
MVO/
  CMakeLists.txt
  build.ps1, build.sh
  run.ps1, run.sh
  bundle_cvlib.ps1, bundle_cvlib.sh
  configs/
    kitti_image_sequence.json
    parameters/
      feature.json
      pnp.json
      initializer.json
      mapping.json
      bundle_adjustment.json
      loop_closure.json
      visualization.json
  include/mvo/
    app.h
    bundle_adjustment.h
    config.h
    constants.h
    converter.h
    feature.h
    frame_source.h
    init.h
    loop_closure.h
    map_data.h
    parameters.h
    pose_estimation.h
    types.h
    visualization.h
  src/
    app.cpp
    bundle_adjustment.cpp
    config.cpp
    converter.cpp
    cvlib_main.cpp
    feature.cpp
    frame_source.cpp
    init.cpp
    loop_closure.cpp
    map_data.cpp
    parameters.cpp
    pose_estimation.cpp
    visualization.cpp
  thirdparty/
    cvlib/
    DBoW2/
```
