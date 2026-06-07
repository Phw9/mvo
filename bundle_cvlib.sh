#!/usr/bin/env bash
set -euo pipefail

config="Release"
cvlib_source_dir=""
generator=""
platform=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            config="$2"
            shift 2
            ;;
        --cvlib-source-dir)
            cvlib_source_dir="$2"
            shift 2
            ;;
        --generator)
            generator="$2"
            shift 2
            ;;
        --platform)
            platform="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./bundle_cvlib.sh [--config Release|Debug] [--cvlib-source-dir PATH] [--platform linux|msvc|macos]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$cvlib_source_dir" ]]; then
    cvlib_source_dir="$(cd "$script_dir/../cvlib/cpp" && pwd)"
fi
if [[ ! -f "$cvlib_source_dir/CMakeLists.txt" ]]; then
    echo "cvlib source not found: $cvlib_source_dir" >&2
    exit 1
fi

if [[ -z "$platform" ]]; then
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) platform="msvc" ;;
        Darwin*) platform="macos" ;;
        *) platform="linux" ;;
    esac
fi
config_lower="$(printf '%s' "$config" | tr '[:upper:]' '[:lower:]')"
build_dir="$script_dir/build/cvlib_package/$platform/$config_lower"
cmake_args=(
    -S "$cvlib_source_dir"
    -B "$build_dir"
    "-DCMAKE_BUILD_TYPE=$config"
    -DCVLIB_BUILD_TESTS=OFF
    -DCVLIB_BUILD_PYTHON=OFF
)
if [[ "$platform" == "linux" || "$platform" == "unix" ]]; then
    linux_cxx_flags="${CXXFLAGS:-}"
    linux_cxx_flags="${linux_cxx_flags:+$linux_cxx_flags }-Wno-unused-function"
    cmake_args+=("-DCMAKE_CXX_FLAGS=$linux_cxx_flags")
fi
if [[ -n "$generator" ]]; then
    cmake_args=(-G "$generator" "${cmake_args[@]}")
fi

cmake "${cmake_args[@]}"
cmake --build "$build_dir" --config "$config"

bundle_root="$script_dir/thirdparty/cvlib"
include_dest="$bundle_root/include"
rm -rf "$include_dest"
mkdir -p "$include_dest"
cvlib_include_source="$cvlib_source_dir/include/cvlib"
if [[ ! -d "$cvlib_include_source" ]]; then
    cvlib_include_source="$cvlib_source_dir/include"
fi
cp -R "$cvlib_include_source/." "$include_dest/"
while IFS= read -r -d '' header; do
    perl -0pi -e 's/#include "cvlib\//#include "/g' "$header"
done < <(find "$include_dest" -type f -name '*.h' -print0)

case "$platform" in
    msvc|windows)
        library_names=(cvlib_core.lib)
        ;;
    linux|unix)
        library_names=(libcvlib_core.a libcvlib_core.so)
        ;;
    macos|darwin)
        library_names=(libcvlib_core.a libcvlib_core.dylib)
        ;;
    *)
        library_names=(cvlib_core.lib libcvlib_core.a libcvlib_core.so libcvlib_core.dylib)
        ;;
esac

libraries=()
for library_name in "${library_names[@]}"; do
    while IFS= read -r library; do
        libraries+=("$library")
    done < <(find "$build_dir" -type f -name "$library_name" | sort)
done
if [[ ${#libraries[@]} -eq 0 ]]; then
    echo "cvlib library for platform '$platform' was not produced under $build_dir" >&2
    exit 1
fi

lib_dest="$bundle_root/lib/$platform/$config_lower"
mkdir -p "$lib_dest"
for library_name in "${library_names[@]}"; do
    rm -f "$lib_dest/$library_name"
done
for library in "${libraries[@]}"; do
    cp "$library" "$lib_dest/"
done

echo "cvlib bundled:"
echo "  headers: $include_dest"
for library in "${libraries[@]}"; do
    echo "  library: $lib_dest/$(basename "$library")"
done
