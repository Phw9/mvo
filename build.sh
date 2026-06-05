#!/usr/bin/env bash
set -euo pipefail

config="Release"

usage() {
    echo "Usage: ./build.sh [--config Release|Debug]"
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        --config)
            if [[ $# -ne 2 ]]; then
                usage >&2
                exit 2
            fi
            config="$2"
            ;;
        Release|Debug)
            if [[ $# -ne 1 ]]; then
                usage >&2
                exit 2
            fi
            config="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
fi

if [[ "$config" != "Release" && "$config" != "Debug" ]]; then
    echo "Config must be Release or Debug." >&2
    exit 2
fi

to_unix_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -u "$1"
    else
        printf '%s\n' "$1"
    fi
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$script_dir/build"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        cvlib_platform="msvc"
        cvlib_library_name="cvlib_core.lib"
        ;;
    Darwin*)
        cvlib_platform="macos"
        cvlib_library_name="libcvlib_core.a"
        ;;
    *)
        cvlib_platform="linux"
        cvlib_library_name="libcvlib_core.a"
        ;;
esac

config_lower="$(printf '%s' "$config" | tr '[:upper:]' '[:lower:]')"
cvlib_library="$script_dir/thirdparty/cvlib/lib/$cvlib_platform/$config_lower/$cvlib_library_name"
if [[ ! -f "$cvlib_library" ]]; then
    "$script_dir/bundle_cvlib.sh" --config "$config" --platform "$cvlib_platform"
fi

opencv_dir="${OpenCV_DIR:-}"
default_opencv_dir=""
if [[ -n "${LOCALAPPDATA:-}" ]]; then
    default_opencv_dir="$(to_unix_path "$LOCALAPPDATA")/rtk/opencv-4.13.0/opencv/build"
fi
if [[ -z "$opencv_dir" && -n "$default_opencv_dir" && -d "$default_opencv_dir" ]]; then
    opencv_dir="$default_opencv_dir"
fi
if [[ -n "$opencv_dir" ]]; then
    opencv_dir="$(to_unix_path "$opencv_dir")"
fi

cmake_args=(
    -S "$script_dir"
    -B "$build_dir"
    "-DCMAKE_BUILD_TYPE=$config"
    "-DMVO_CVLIB_PLATFORM=$cvlib_platform"
)

if [[ -n "$opencv_dir" ]]; then
    cmake_args+=("-DOpenCV_DIR=$opencv_dir")
fi
if [[ -n "${MVO_RERUN_SDK_URL:-}" ]]; then
    cmake_args+=("-DMVO_RERUN_SDK_URL=$MVO_RERUN_SDK_URL")
fi

cmake "${cmake_args[@]}"
cmake --build "$build_dir" --config "$config"
