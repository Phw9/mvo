#!/usr/bin/env bash
set -euo pipefail

config="Release"
install_opencv=false
install_rerun=false
install_option_seen=false
opencv_version="${MVO_OPENCV_VERSION:-4.13.0}"
rerun_version="${MVO_RERUN_VERSION:-0.33.0}"
pip_install_args=(
    install
    --disable-pip-version-check
    --no-warn-script-location
    --user
    "rerun-sdk==$rerun_version"
)

usage() {
    echo "Usage: ./build.sh [--config Release|Debug] [--install-opencv|--install-rerun|--install-all|--no-install]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            if [[ $# -lt 2 ]]; then
                usage >&2
                exit 2
            fi
            config="$2"
            shift 2
            ;;
        Release|Debug)
            config="$1"
            shift
            ;;
        --install-opencv)
            install_option_seen=true
            install_opencv=true
            shift
            ;;
        --install-rerun)
            install_option_seen=true
            install_rerun=true
            shift
            ;;
        --install-all)
            install_option_seen=true
            install_opencv=true
            install_rerun=true
            shift
            ;;
        --no-install)
            install_option_seen=true
            install_opencv=false
            install_rerun=false
            shift
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
done

if [[ "$install_option_seen" == false ]]; then
    install_opencv=true
    install_rerun=true
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

add_path_dir() {
    if [[ -d "$1" && ":$PATH:" != *":$1:"* ]]; then
        export PATH="$1:$PATH"
    fi
}

add_python_user_scripts_to_path() {
    local user_base=""
    local user_base_unix=""
    local python_cmd=""

    for python_cmd in py python3 python; do
        if command -v "$python_cmd" >/dev/null 2>&1; then
            user_base="$("$python_cmd" -c 'import site; print(site.USER_BASE)' 2>/dev/null || true)"
            if [[ -n "$user_base" ]]; then
                user_base_unix="$(to_unix_path "$user_base")"
                add_path_dir "$user_base_unix/Scripts"
                add_path_dir "$user_base_unix/bin"
            fi
        fi
    done

    if [[ -n "${APPDATA:-}" ]]; then
        local appdata_unix
        appdata_unix="$(to_unix_path "$APPDATA")"
        for scripts_dir in "$appdata_unix"/Python/Python*/Scripts; do
            add_path_dir "$scripts_dir"
        done
    fi
}

print_opencv_prerequisites() {
    cat >&2 <<'EOF'
OpenCV was not found.

Install OpenCV, then rerun ./build.sh:
  Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y libopencv-dev pkg-config
  Fedora:        sudo dnf install opencv-devel pkgconf-pkg-config
  Arch:          sudo pacman -S opencv pkgconf
  macOS:         brew install opencv
  Windows:       ./build.sh --install-opencv
  MSYS2:         pacman -S mingw-w64-x86_64-opencv

If OpenCV is already installed in a custom location, set OpenCV_DIR to the
directory containing OpenCVConfig.cmake.
EOF
}

print_rerun_prerequisites() {
    cat >&2 <<EOF
Rerun viewer was not found.

Install Rerun, then rerun ./build.sh:
  Python/pip:    python3 -m pip install --user rerun-sdk==$rerun_version
  Cargo:         cargo install rerun-cli --locked
  Windows:       py -m pip install --user rerun-sdk==$rerun_version

The Rerun viewer version should match the C++ SDK version used by this project.
EOF
}

install_opencv_package() {
    case "$(uname -s)" in
        Linux*)
            if command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update
                sudo apt-get install -y libopencv-dev pkg-config
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y opencv-devel pkgconf-pkg-config
            elif command -v pacman >/dev/null 2>&1; then
                sudo pacman -S --needed opencv pkgconf
            else
                print_opencv_prerequisites
                exit 1
            fi
            ;;
        Darwin*)
            if command -v brew >/dev/null 2>&1; then
                brew install opencv
            else
                print_opencv_prerequisites
                exit 1
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            if command -v powershell.exe >/dev/null 2>&1; then
                powershell.exe \
                    -NoProfile \
                    -ExecutionPolicy Bypass \
                    -File "$(cygpath -w "$script_dir/build.ps1")" \
                    -Config "$config" \
                    -InstallOpenCV \
                    -SkipBuild
            elif command -v pacman >/dev/null 2>&1; then
                pacman -S --needed mingw-w64-x86_64-opencv
            else
                print_opencv_prerequisites
                exit 1
            fi
            ;;
        *)
            print_opencv_prerequisites
            exit 1
            ;;
    esac
}

install_rerun_package() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            if command -v py >/dev/null 2>&1; then
                py -m pip "${pip_install_args[@]}"
            elif command -v python >/dev/null 2>&1; then
                python -m pip "${pip_install_args[@]}"
            elif command -v python3 >/dev/null 2>&1; then
                python3 -m pip "${pip_install_args[@]}"
            else
                print_rerun_prerequisites
                exit 1
            fi
            ;;
        *)
            if command -v python3 >/dev/null 2>&1; then
                python3 -m pip "${pip_install_args[@]}"
            elif command -v python >/dev/null 2>&1; then
                python -m pip "${pip_install_args[@]}"
            elif command -v pip3 >/dev/null 2>&1; then
                pip3 "${pip_install_args[@]}"
            elif command -v cargo >/dev/null 2>&1; then
                cargo install rerun-cli --locked
            else
                print_rerun_prerequisites
                exit 1
            fi
            ;;
    esac
    add_python_user_scripts_to_path
}

opencv_is_available() {
    if [[ -n "${opencv_dir:-}" && -f "${opencv_dir:-}/OpenCVConfig.cmake" ]]; then
        return 0
    fi
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists opencv4; then
        return 0
    fi
    if [[ -f /usr/lib/x86_64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake ||
          -f /usr/local/lib/cmake/opencv4/OpenCVConfig.cmake ||
          -f /opt/homebrew/lib/cmake/opencv4/OpenCVConfig.cmake ||
          -f /usr/local/opt/opencv/lib/cmake/opencv4/OpenCVConfig.cmake ]]; then
        return 0
    fi
    return 1
}

rerun_is_available() {
    command -v rerun >/dev/null 2>&1
}

cvlib_is_available() {
    local lib_dir="$script_dir/thirdparty/cvlib/lib/$cvlib_platform/$config_lower"
    local library_name=""

    for library_name in "${cvlib_library_names[@]}"; do
        if [[ -f "$lib_dir/$library_name" ]]; then
            return 0
        fi
    done

    return 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_root="$script_dir/build"
dependency_root="${MVO_DEPS_ROOT:-}"
if [[ -n "$dependency_root" ]]; then
    dependency_root="$(to_unix_path "$dependency_root")"
elif [[ -n "${LOCALAPPDATA:-}" ]]; then
    dependency_root="$(to_unix_path "$LOCALAPPDATA")/MVO"
elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    dependency_root="$XDG_CACHE_HOME/mvo"
elif [[ -n "${HOME:-}" ]]; then
    dependency_root="$HOME/.cache/mvo"
else
    dependency_root="$script_dir/.deps"
fi

add_python_user_scripts_to_path

opencv_dir="${OpenCV_DIR:-}"
default_opencv_dir="$dependency_root/opencv-$opencv_version/opencv/build"
if [[ -z "$opencv_dir" && -n "$default_opencv_dir" && -f "$default_opencv_dir/OpenCVConfig.cmake" ]]; then
    opencv_dir="$default_opencv_dir"
fi
if [[ -n "$opencv_dir" ]]; then
    opencv_dir="$(to_unix_path "$opencv_dir")"
fi

if [[ "$install_opencv" == true ]] && ! opencv_is_available; then
    install_opencv_package
    if [[ -n "$default_opencv_dir" && -f "$default_opencv_dir/OpenCVConfig.cmake" ]]; then
        opencv_dir="$default_opencv_dir"
    fi
fi

if [[ "$install_rerun" == true ]] && ! rerun_is_available; then
    install_rerun_package
fi

if ! opencv_is_available; then
    print_opencv_prerequisites
    echo >&2
    echo "Tip: ./build.sh --install-opencv can install it on supported systems." >&2
    exit 1
fi

if ! rerun_is_available; then
    print_rerun_prerequisites
    echo >&2
    echo "Tip: ./build.sh --install-rerun can install it on supported systems." >&2
    exit 1
fi

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        cvlib_platform="msvc"
        cvlib_library_names=(cvlib_core.lib)
        ;;
    Darwin*)
        cvlib_platform="macos"
        cvlib_library_names=(libcvlib_core.a libcvlib_core.dylib)
        ;;
    *)
        cvlib_platform="linux"
        cvlib_library_names=(libcvlib_core.a libcvlib_core.so)
        ;;
esac

config_lower="$(printf '%s' "$config" | tr '[:upper:]' '[:lower:]')"
build_dir="$build_root/$cvlib_platform/$config_lower"
if ! cvlib_is_available; then
    "$script_dir/bundle_cvlib.sh" --config "$config" --platform "$cvlib_platform"
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
if [[ -n "${MVO_RERUN_SDK_URL:-}" && "$MVO_RERUN_SDK_URL" =~ [^[:space:]] ]]; then
    cmake_args+=("-DMVO_RERUN_SDK_URL=$MVO_RERUN_SDK_URL")
fi

cmake --log-level=WARNING "${cmake_args[@]}"
cmake --build "$build_dir" --config "$config"
