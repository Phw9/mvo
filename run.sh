#!/usr/bin/env bash
set -euo pipefail

config="Release"
max_frames="3000"
no_ba=()
input_args=()
rerun_enabled=1
rerun_mode="spawn"
rerun_save=""
debug_geometry=()

usage() {
    echo "Usage: ./run.sh [--config Release|Debug] [--max-frames N] [--no-ba] [--input-config PATH] [--parameter-dir DIR] [--debug-geometry] [--rerun|--rerun-save PATH|--no-rerun]"
}

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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            config="$2"
            shift 2
            ;;
        --max-frames)
            max_frames="$2"
            shift 2
            ;;
        --no-ba)
            no_ba=(--no-ba)
            shift
            ;;
        --input-config|--input-type|--input-path|--parameter-dir)
            input_args+=("$1" "$2")
            shift 2
            ;;
        --calib)
            input_args+=("--calib" "$2")
            shift 2
            ;;
        --debug-geometry)
            debug_geometry=(--debug-geometry)
            shift
            ;;
        --rerun|--rerun-spawn)
            rerun_enabled=1
            rerun_mode="spawn"
            rerun_save=""
            shift
            ;;
        --rerun-save)
            rerun_enabled=1
            rerun_mode="save"
            rerun_save="$2"
            shift 2
            ;;
        --no-rerun)
            rerun_enabled=0
            rerun_mode=""
            rerun_save=""
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$config" != "Release" && "$config" != "Debug" ]]; then
    echo "Config must be Release or Debug." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
opencv_version="${MVO_OPENCV_VERSION:-4.13.0}"
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
opencv_bin="$dependency_root/opencv-$opencv_version/opencv/build/x64/vc16/bin"
add_path_dir "$opencv_bin"
add_python_user_scripts_to_path

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) build_platform="msvc" ;;
    Darwin*) build_platform="macos" ;;
    *) build_platform="linux" ;;
esac
config_lower="$(printf '%s' "$config" | tr '[:upper:]' '[:lower:]')"
exe=""
exe_candidates=(
    "$script_dir/build/$build_platform/$config_lower/mvo_cvlib"
    "$script_dir/build/$build_platform/$config_lower/$config/mvo_cvlib.exe"
    "$script_dir/build/$config/mvo_cvlib.exe"
    "$script_dir/build/mvo_cvlib"
)
for exe_candidate in "${exe_candidates[@]}"; do
    if [[ -x "$exe_candidate" ]]; then
        exe="$exe_candidate"
        break
    fi
done
if [[ ! -x "$exe" ]]; then
    echo "mvo_cvlib executable not found. Run ./build.sh first." >&2
    exit 1
fi

if [[ "$rerun_enabled" -eq 1 && "$rerun_mode" == "spawn" ]] && ! command -v rerun >/dev/null 2>&1; then
    echo "warning: rerun viewer was not found in PATH; spawn may fail. Use --rerun-save build/mvo.rrd to record without a viewer." >&2
fi

run_args=(--no-gui)
if [[ -n "$max_frames" ]]; then
    run_args+=(--max-frames "$max_frames")
fi
if [[ "$rerun_enabled" -eq 0 ]]; then
    run_args+=(--no-rerun)
elif [[ "$rerun_mode" == "save" ]]; then
    run_args+=(--rerun-save "$rerun_save")
else
    run_args+=(--rerun-spawn)
fi

"$exe" "${run_args[@]}" "${no_ba[@]}" "${debug_geometry[@]}" "${input_args[@]}"
