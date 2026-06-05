# AGENTS.md

Project instructions for agents working on MVO.

MVO is a monocular visual odometry project built with the bundled `cvlib`
headers and libraries under `thirdparty/cvlib`.

@.codex/rules/slam.md
@.codex/agents/slam-engineer.md

## Local Priorities

- Keep README compact: project summary, build/run commands, and file structure.
- Keep runtime tunables in JSON under `configs/parameters`.
- Keep source and header filenames lowercase.
- Keep headers split by function under `include/mvo`.
- Keep `cvlib` consumed through bundled headers and library files.
- Do not reintroduce `g2o`; bundle adjustment should use `cvlib`.
- Rerun visualization is enabled by default through the run scripts.

## Validation

- Windows build: `.\build.ps1`
- Windows run check: `.\run.ps1 -Config Release -MaxFrames 3 -NoBa -NoRerun -ParameterDir .\configs\parameters`
- Linux/Git Bash run check: `bash ./run.sh --config Release --max-frames 3 --no-ba --no-rerun --parameter-dir ./configs/parameters`

