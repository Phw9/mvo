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
if [[ -n "${LOCALAPPDATA:-}" ]]; then
    opencv_bin="$(to_unix_path "$LOCALAPPDATA")/rtk/opencv-4.13.0/opencv/build/x64/vc16/bin"
    if [[ -d "$opencv_bin" ]]; then
        export PATH="$opencv_bin:$PATH"
    fi
fi
if command -v python >/dev/null 2>&1; then
    python_user_scripts="$(python -c 'import sysconfig; print(sysconfig.get_path("scripts", scheme="nt_user"))' 2>/dev/null || true)"
    if [[ -n "$python_user_scripts" ]]; then
        python_user_scripts="$(to_unix_path "$python_user_scripts")"
        if [[ -d "$python_user_scripts" ]]; then
            export PATH="$python_user_scripts:$PATH"
        fi
    fi
fi

exe="$script_dir/build/$config/mvo_cvlib.exe"
if [[ ! -x "$exe" ]]; then
    exe="$script_dir/build/mvo_cvlib"
fi
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
