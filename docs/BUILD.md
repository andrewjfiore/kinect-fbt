# Building Marionette

Same CMake flow on Windows and Linux. All third-party dependencies (Eigen,
nlohmann::json, doctest, OpenVR headers) are fetched automatically at configure
time; you only need a compiler and CMake.

See also: [USAGE.md](USAGE.md) (running it), [CALIBRATION.md](CALIBRATION.md)
(sensor setup), [DESIGN.md](DESIGN.md) (architecture).

## Windows

Prerequisites:

- **Visual Studio 2022 Build Tools** (or full VS 2022) with the
  "Desktop development with C++" workload.
- **CMake 3.24+** (bundled with VS 2022, or standalone).

Optional sensor SDKs (skip both to build the hardware-free core + mock nodes):

| SDK | For | Env var set by installer | Detected via |
|---|---|---|---|
| [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561) | Kinect v2 (Xbox One) | `KINECTSDK20_DIR` | `%KINECTSDK20_DIR%\inc\Kinect.h` |
| [Kinect for Windows SDK 1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278) | Kinect v1 (Xbox 360) | `KINECTSDK10_DIR` | `%KINECTSDK10_DIR%\inc\NuiApi.h` |

The installers set the environment variables machine-wide. Open a **new**
shell after installing, then configure with a fresh build directory (or delete
`build/CMakeCache.txt`). The configure output prints `Kinect SDK 2.0 found` /
`Kinect SDK 1.8 found` when a backend is enabled; absent SDKs are skipped
silently, nothing else changes.

Build, from a Developer Command Prompt or any shell with CMake on PATH:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMN_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The CLI lands at `build\app\Release\marionette.exe` (Visual Studio is a
multi-config generator).

## Linux

Prerequisites:

```sh
sudo apt install build-essential cmake
```

(`ninja-build` optional; add `-G Ninja` to the configure line if you prefer it.)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMN_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The CLI lands at `build/app/marionette`.

**WSL note**: the core, mock/replay nodes, OSC endpoint, and the full test
suite build and run fine under WSL. Actually *using* the SteamVR driver needs
SteamVR running in a real desktop session on the same machine; do that on a
native install (Windows or desktop Linux), not WSL.

## CMake options

| Option | Default | Effect |
|---|---|---|
| `MN_BUILD_TESTS` | `ON` | Build the doctest suite (`mn_tests`, registered with CTest) |
| `MN_BUILD_DRIVER` | `ON` | Build the SteamVR driver `driver_marionette` |
| `MN_WITH_OPENVR_CLIENT` | `ON` | Link the prebuilt OpenVR client lib (enables `calibrate playspace`); auto-disables if the platform binary is missing from the OpenVR archive |

## SteamVR driver output and install

Building `driver_marionette` assembles a ready-to-register SteamVR driver
folder at:

```
build/dist/marionette/
  driver.vrdrivermanifest
  resources/...
  bin/win64/driver_marionette.dll      (or bin/linux64/driver_marionette.so)
```

Register it with SteamVR (wraps `vrpathreg adddriver`):

```sh
scripts/install-driver.ps1     # Windows
scripts/install-driver.sh      # Linux
```

Manual fallback: run `vrpathreg adddriver <absolute path to build/dist/marionette>`
using the `vrpathreg` binary in your SteamVR install
(`Steam/steamapps/common/SteamVR/bin/win64/` or `.../bin/linux64/`). Restart
SteamVR afterward. See [USAGE.md](USAGE.md#pcvr-steamvr) for the runtime side.
