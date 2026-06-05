# cvlib package

This directory contains the cvlib public headers and a prebuilt library used by
MVO. The normal MVO build links this package directly and does not read the
sibling `../cvlib/cpp` source directory.

Layout:

- `include`: cvlib public headers.
- `lib/msvc/<config>/cvlib_core.lib`: MSVC static library.
- `lib/linux/<config>/libcvlib_core.a`: Linux static library.
- `lib/linux/<config>/libcvlib_core.so`: Linux shared library alternative.
- `lib/macos/<config>/libcvlib_core.a`: macOS static library.
- `lib/macos/<config>/libcvlib_core.dylib`: macOS shared library alternative.

To refresh this package after cvlib changes:

```powershell
.\bundle_cvlib.ps1 -Config Release
.\bundle_cvlib.ps1 -Config Debug
```

```bash
./bundle_cvlib.sh --config Release
./bundle_cvlib.sh --config Debug
```

Build or refresh the Linux package on a Linux machine:

```bash
./bundle_cvlib.sh --config Release --platform linux
```
