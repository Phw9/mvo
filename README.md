# MVO

MVO is a monocular visual odometry project written with my `cvlib`.

Author: Hyunwoo Park <phphww93@gmail.com>

## Quick Start

```powershell
.\build.ps1 # default: install all dependencies and build
.\run.ps1 -Config Release -MaxFrames 3 -NoBa -NoRerun -ParameterDir .\configs\parameters # quick run check
.\run.ps1 -Config Release # run with Rerun viewer
```

```bash
bash build.sh # default: install all dependencies and build
bash run.sh --config Release --max-frames 3 --no-ba --no-rerun --parameter-dir ./configs/parameters # quick run check
bash run.sh --config Release # run with Rerun viewer
```

## Quick Install

```powershell
.\build.ps1 # default: same as -InstallAll
.\build.ps1 -InstallAll # install OpenCV and Rerun, then build
.\build.ps1 -InstallOpenCV # install OpenCV, then build
.\build.ps1 -InstallRerun # install Rerun, then build
.\build.ps1 -NoInstall # build only with existing dependencies
$env:OpenCV_DIR = "$env:LOCALAPPDATA\rtk\opencv-4.13.0\opencv\build" # custom OpenCVConfig.cmake path
py -m pip install --user rerun-sdk==0.33.0 # manual Rerun install
$UserScripts = Join-Path (& py -c "import site; print(site.USER_BASE)") "Scripts" # Python user Scripts path
$env:PATH = "$UserScripts;$env:PATH" # add rerun.exe to PATH
```

```bash
bash build.sh # default: same as --install-all
bash build.sh --install-all # install OpenCV and Rerun, then build
bash build.sh --install-opencv # install OpenCV, then build
bash build.sh --install-rerun # install Rerun, then build
bash build.sh --no-install # build only with existing dependencies
export OpenCV_DIR="$LOCALAPPDATA/rtk/opencv-4.13.0/opencv/build" # custom OpenCVConfig.cmake path
python3 -m pip install --user rerun-sdk==0.33.0 # manual Rerun install
export PATH="$(python3 -c 'import site; print(site.USER_BASE)')/bin:$PATH" # add rerun to PATH on Linux/macOS
export PATH="$(cygpath -u "$APPDATA")/Python/Python314/Scripts:$PATH" # add rerun.exe to PATH on Git Bash
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
